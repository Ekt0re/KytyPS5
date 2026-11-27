#ifndef EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_WAITDIAGNOSTICS_H_
#define EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_WAITDIAGNOSTICS_H_

#include "kernel/guestWait.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// KYTY_WAIT_DIAG=1: while no frame is presented, print where every command queue is and what a
// queue spinning in wait_reg_mem waits on. The command processor notes each poll and each label
// write; the reporter only reads those host-side notes, never guest memory. Queue waits are
// described with the same LibKernel::GuestWaitState type and DescribeWait() wording the crash
// report uses for guest threads, so a stall and a crash read the same way.
namespace Libs::Graphics::WaitDiag {

[[nodiscard]] inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_WAIT_DIAG");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

constexpr uint32_t QueueSlots   = 64;
constexpr uint32_t LabelRingLen = 32;
constexpr uint32_t ReadbackLen  = 32;
constexpr uint64_t NearBytes    = 4096;

enum class QueueState : uint32_t {
	Unused,
	Idle,
	Running,
	WaitingMemory,
	SuspendedForShader,
};

struct LabelWrite {
	uint64_t address = 0;
	uint64_t value   = 0;
	uint32_t size    = 0;
	uint32_t queue   = 0;
	uint64_t ms      = 0;
};

struct Readback {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t ms      = 0;
};

struct WaitPacket {
	bool     valid        = false;
	bool     is64         = false;
	uint32_t func         = 0;
	uint64_t address      = 0;
	uint64_t mask         = 0;
	uint64_t reference    = 0;
	uint64_t value        = 0; // read by the command processor at its last poll
	uint64_t buffer       = 0; // address of the command buffer holding the packet
	uint32_t offset_dw    = 0;
	uint64_t polls        = 0;
	uint64_t first_ms     = 0;
	uint64_t last_ms      = 0;
	int      gpu_modified = -1; // -1 not asked, 0 no, 1 yes: a shader wrote the range
	int      gpu_dirty    = -1; // -1 not asked, 0 no, 1 yes: written bytes not read back yet

	// Fill upstream's crash-time wait type from the note (no guest read). A reference is taken
	// because its atomics make GuestWaitState neither copyable nor movable.
	void FillGuestWaitState(LibKernel::GuestWaitState& state, uint64_t now_ms) const {
		state.call.store(is64 ? "wait_reg_mem64" : "wait_reg_mem32", std::memory_order_release);
		state.object.store(address, std::memory_order_relaxed);
		state.value.store(static_cast<int64_t>(value), std::memory_order_relaxed);
		const auto since = now_ms > first_ms ? (now_ms - first_ms) * 1000000ull : 0ull;
		state.since.store(since, std::memory_order_relaxed);
	}
};

struct QueueInfo {
	QueueState state    = QueueState::Unused;
	uint64_t   hash     = 0; // the shader a queue is suspended for
	uint64_t   since_ms = 0;
	WaitPacket wait;
};

[[nodiscard]] inline const char* CompareName(uint32_t func) {
	switch (func) {
		case 0: return "always";
		case 1: return "<";
		case 2: return "<=";
		case 3: return "==";
		case 4: return "!=";
		case 5: return ">=";
		case 6: return ">";
		default: return "?";
	}
}

[[nodiscard]] inline std::string QueueName(uint32_t slot) {
	char text[32];
	if (slot == 0) {
		std::snprintf(text, sizeof(text), "0 (draw)");
	} else {
		std::snprintf(text, sizeof(text), "0x%02x (compute)", 0x20u + slot - 1u);
	}
	return text;
}

#if defined(__clang__) || defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif
inline std::string Fmt(const char* format, ...) {
	char    text[512];
	va_list args;
	va_start(args, format);
	std::vsnprintf(text, sizeof(text), format, args);
	va_end(args);
	return text;
}

class Registry {
public:
	void NoteLabel(uint64_t address, uint64_t value, uint32_t size, uint32_t queue, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		m_labels[m_label_count % LabelRingLen] = {address, value, size, queue, ms};
		m_label_count++;
	}

	void NoteReadback(uint64_t address, uint64_t size, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		m_readbacks[m_readback_count % ReadbackLen] = {address, size, ms};
		m_readback_count++;
	}

	// One poll of a wait packet; returns how often this same packet was polled in a row.
	uint64_t NotePoll(uint32_t slot, const WaitPacket& packet, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		auto&            info = m_queues[slot % QueueSlots];
		auto&            wait = info.wait;
		const bool same = wait.valid && wait.buffer == packet.buffer &&
		                  wait.offset_dw == packet.offset_dw && wait.address == packet.address &&
		                  wait.reference == packet.reference;
		const auto polls    = same ? wait.polls + 1 : 1;
		const auto first_ms = same ? wait.first_ms : ms;
		const auto modified = same ? wait.gpu_modified : -1;
		const auto dirty    = same ? wait.gpu_dirty : -1;
		wait                = packet;
		wait.valid          = true;
		wait.polls          = polls;
		wait.first_ms       = first_ms;
		wait.last_ms        = ms;
		wait.gpu_modified   = packet.gpu_modified >= 0 ? packet.gpu_modified : modified;
		wait.gpu_dirty      = packet.gpu_dirty >= 0 ? packet.gpu_dirty : dirty;
		return polls;
	}

	// A queue starts a slice; a queue already known to wait keeps its state and start time.
	void NoteRunning(uint32_t slot, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		auto&            info = m_queues[slot % QueueSlots];
		if (info.state == QueueState::Unused || info.state == QueueState::Idle) {
			info.state    = QueueState::Running;
			info.hash     = 0;
			info.since_ms = ms;
		}
	}

	void NoteQueue(uint32_t slot, QueueState state, uint64_t hash, uint64_t ms) {
		std::scoped_lock lock(m_mutex);
		auto&            info = m_queues[slot % QueueSlots];
		if (info.state != state || info.hash != hash) {
			info.since_ms = ms;
		}
		info.state = state;
		info.hash  = hash;
		if (state != QueueState::WaitingMemory) {
			info.wait.valid = false;
		}
	}

	// Lines for the stall reporter; empty when no queue is known.
	[[nodiscard]] std::vector<std::string> Report(uint64_t now_ms) const {
		std::scoped_lock         lock(m_mutex);
		std::vector<std::string> lines;
		std::vector<uint64_t>    addresses;
		for (uint32_t slot = 0; slot < QueueSlots; slot++) {
			const auto& info = m_queues[slot];
			if (info.state == QueueState::Unused) {
				continue;
			}
			const double age = static_cast<double>(now_ms - std::min(now_ms, info.since_ms))/1000.0;
			switch (info.state) {
				case QueueState::Idle:
					lines.push_back(Fmt("[wait] queue %s: idle for %.1f s", QueueName(slot).c_str(), age));
					break;
				case QueueState::Running:
					lines.push_back(Fmt("[wait] queue %s: running for %.1f s", QueueName(slot).c_str(), age));
					break;
				case QueueState::SuspendedForShader:
					lines.push_back(Fmt("[wait] queue %s: suspended for shader 0x%016" PRIx64 " for %.1f s",
					                    QueueName(slot).c_str(), info.hash, age));
					break;
				case QueueState::WaitingMemory: {
					const auto& w = info.wait;
					if (!w.valid || now_ms - std::min(now_ms, w.last_ms) > 1500) {
						lines.push_back(Fmt("[wait] queue %s: blocked for %.1f s, not on a wait_reg_mem "
						                    "packet (flip, semaphore or predication)",
						                    QueueName(slot).c_str(), age));
						break;
					}
					LibKernel::GuestWaitState described;
					w.FillGuestWaitState(described, now_ms);
					const double wait_age =
					    static_cast<double>(now_ms - std::min(now_ms, w.first_ms)) / 1000.0;
					lines.push_back(Fmt(
					    "[wait] queue %s: %s; mask 0x%" PRIx64 " compare (value & mask) %s 0x%" PRIx64
					    "; packet at 0x%016" PRIx64 " + %u dw; polled %" PRIu64 " times in %.1f s",
					    QueueName(slot).c_str(),
					    LibKernel::DescribeWait(described, now_ms * 1000000ull).c_str(), w.mask,
					    CompareName(w.func), w.reference, w.buffer, w.offset_dw, w.polls, wait_age));
					const auto yes_no = [](int value) {
						return value < 0 ? "unknown" : (value != 0 ? "yes" : "no");
					};
					lines.push_back(Fmt("[wait]   range written by a shader: %s; written bytes not read "
					                    "back yet: %s",
					                    yes_no(w.gpu_modified), yes_no(w.gpu_dirty)));
					addresses.push_back(w.address);
					break;
				}
				default: break;
			}
		}
		for (const auto address: addresses) {
			AppendNear(lines, address, now_ms);
		}
		return lines;
	}

	void Clear() {
		std::scoped_lock lock(m_mutex);
		m_queues         = {};
		m_labels         = {};
		m_readbacks      = {};
		m_label_count    = 0;
		m_readback_count = 0;
	}

private:
	static bool Near(uint64_t address, uint64_t size, uint64_t waited) {
		const auto lo = waited >= NearBytes ? waited - NearBytes : 0;
		return address + size > lo && address <= waited + NearBytes;
	}

	void AppendNear(std::vector<std::string>& lines, uint64_t waited, uint64_t now_ms) const {
		const auto count = std::min<uint64_t>(m_label_count, LabelRingLen);
		uint64_t   shown = 0;
		for (uint64_t i = 0; i < count; i++) {
			// Newest first.
			const auto& l = m_labels[(m_label_count - 1 - i) % LabelRingLen];
			if (!Near(l.address, l.size, waited)) {
				continue;
			}
			shown++;
			lines.push_back(Fmt("[wait]   label write 0x%016" PRIx64 " <- 0x%" PRIx64
			                    " (%u bytes) by queue %s, %.2f s ago%s",
			                    l.address, l.value, l.size, QueueName(l.queue).c_str(),
			                    static_cast<double>(now_ms - std::min(now_ms, l.ms)) / 1000.0,
			                    l.address == waited ? " [the waited address]" : ""));
		}
		if (shown == 0) {
			const auto oldest = count == 0 ? 0 : m_labels[(m_label_count - count) % LabelRingLen].ms;
			lines.push_back(Fmt("[wait]   no label write within 4 KiB of 0x%016" PRIx64
			                    " among the last %" PRIu64 " (%" PRIu64 " in all, oldest %.1f s ago)",
			                    waited, count, m_label_count,
			                    static_cast<double>(now_ms - std::min(now_ms, oldest)) / 1000.0));
		}
		const auto readbacks = std::min<uint64_t>(m_readback_count, ReadbackLen);
		uint64_t   covering  = 0;
		for (uint64_t i = 0; i < readbacks; i++) {
			const auto& r = m_readbacks[(m_readback_count - 1 - i) % ReadbackLen];
			if (r.address <= waited && waited < r.address + r.size) {
				covering++;
				lines.push_back(Fmt("[wait]   read back 0x%016" PRIx64 " + 0x%" PRIx64 ", %.2f s ago",
				                    r.address, r.size,
				                    static_cast<double>(now_ms - std::min(now_ms, r.ms)) / 1000.0));
				break;
			}
		}
		if (covering == 0) {
			lines.push_back(Fmt("[wait]   no recent read back covers 0x%016" PRIx64
			                    " (last %" PRIu64 " read backs of %" PRIu64 " in all)",
			                    waited, readbacks, m_readback_count));
		}
	}

	mutable std::mutex                    m_mutex;
	std::array<QueueInfo, QueueSlots>     m_queues {};
	std::array<LabelWrite, LabelRingLen>  m_labels {};
	std::array<Readback, ReadbackLen>     m_readbacks {};
	uint64_t                              m_label_count    = 0;
	uint64_t                              m_readback_count = 0;
};

[[nodiscard]] inline Registry& Global() {
	static Registry registry;
	return registry;
}

[[nodiscard]] inline uint64_t NowMs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// The presentation thread tells us a frame was shown; the reporter stays quiet while frames keep
// coming and prints only during a stall (no frame for five seconds).
inline std::atomic<uint64_t>& LastFrameMs() {
	static std::atomic<uint64_t> last_ms {0};
	return last_ms;
}

inline void NoteFramePresented() {
	if (Enabled()) {
		LastFrameMs().store(NowMs(), std::memory_order_relaxed);
	}
}

inline bool Stalled(uint64_t now_ms) {
	return now_ms - std::min(now_ms, LastFrameMs().load(std::memory_order_relaxed)) >= 5000;
}

// Started once from the guest GPU constructor; one detached reporter, like other KYTY_* reports.
inline void StartReporter() {
	if (!Enabled()) {
		return;
	}
	static std::once_flag once;
	std::call_once(once, [] {
		std::thread([] {
			for (;;) {
				std::this_thread::sleep_for(std::chrono::seconds(2));
				const auto now = NowMs();
				if (!Stalled(now)) {
					continue;
				}
				for (const auto& line: Global().Report(now)) {
					std::fprintf(stdout, "%s\n", line.c_str());
				}
				std::fflush(stdout);
			}
		}).detach();
	});
}

} // namespace Libs::Graphics::WaitDiag

#endif // EMULATOR_SRC_GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_WAITDIAGNOSTICS_H_
