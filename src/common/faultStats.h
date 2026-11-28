#ifndef KYTY_COMMON_FAULT_STATS_H_
#define KYTY_COMMON_FAULT_STATS_H_

#include "common/perfSummary.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

// Bookkeeping for the write-fault diagnostics. There is deliberately no reporter here: the
// [fault] lines are printed by PerfSummary's reporter when KYTY_FAULT_STATS=1. Fault counts are
// also PerfSummary counters (WriteFaults/ReadFaults), counted at their existing site in
// RenderContext::HandleFault; nothing in this header counts a fault again.

namespace Common::FaultStats {

[[nodiscard]] inline uint64_t NowNs() noexcept {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

// Numeric switch with a documented fallback and a cap, read once by ConfigureStorm. Zero faults
// (the default) leaves the storm rule off; frames/hold only matter when faults is nonzero.
[[nodiscard]] inline uint32_t ParseStormValue(const char* value, uint32_t fallback,
                                              uint32_t maximum) noexcept {
	if (value == nullptr || value[0] == '\0') {
		return fallback;
	}
	const auto parsed = std::strtoul(value, nullptr, 10);
	if (parsed > maximum) {
		return maximum;
	}
	return static_cast<uint32_t>(parsed);
}

struct StormTuning {
	std::atomic<uint32_t> faults {0};
	std::atomic<uint32_t> frames {8};
	std::atomic<uint32_t> hold {60};
	std::atomic<uint32_t> frame {0}; // The rule's clock, set by the owner of the frame count.
};

inline StormTuning g_storm {};

inline void ConfigureStorm() noexcept {
	static std::atomic<bool> configured {false};
	if (configured.exchange(true, std::memory_order_relaxed)) {
		return;
	}
	const auto faults = ParseStormValue(std::getenv("KYTY_FAULT_STORM_FAULTS"), 0, 250);
	const auto frames = ParseStormValue(std::getenv("KYTY_FAULT_STORM_FRAMES"), 8, 30000);
	const auto hold   = ParseStormValue(std::getenv("KYTY_FAULT_STORM_HOLD"), 60, 30000);
	g_storm.faults.store(faults, std::memory_order_relaxed);
	g_storm.frames.store(frames == 0 ? 1 : frames, std::memory_order_relaxed);
	g_storm.hold.store(hold, std::memory_order_relaxed);
}

[[nodiscard]] inline bool StormEnabled() noexcept {
	return g_storm.faults.load(std::memory_order_relaxed) != 0;
}

inline void SetFrame(uint32_t frame) noexcept {
	g_storm.frame.store(frame, std::memory_order_relaxed);
}

[[nodiscard]] inline uint32_t Frame() noexcept {
	return g_storm.frame.load(std::memory_order_relaxed);
}

inline void NoteLockWait(uint64_t ns) noexcept {
	PerfSummary::Add(PerfSummary::Counter::FaultLockWaitUs, ns / 1000);
}

inline void NoteProtect(uint64_t ns) noexcept {
	PerfSummary::Add(PerfSummary::Counter::FaultProtectUs, ns / 1000);
}

inline void NoteInvalidate(uint64_t ns) noexcept {
	PerfSummary::Add(PerfSummary::Counter::FaultInvalidateUs, ns / 1000);
}

// Faulting 64 KiB blocks of the current interval, in a small open-addressed table. Note() is
// called from the fault handler and only touches atomics; TakeTop() is called by the reporter.
class FaultHotBlocks final {
public:
	static constexpr size_t kSlots  = 1024;
	static constexpr size_t kProbes = 8;
	static constexpr size_t kTop    = 8;

	using Entry = PerfSummary::Detail::FaultBlock; // Address aligned down to 64 KiB, and count.

	void Note(uint64_t address) noexcept {
		const uint64_t key  = (address >> 16) + 1;
		auto           slot = static_cast<size_t>((key * 0x9E3779B97F4A7C15ull) >> 54) % kSlots;
		for (size_t probe = 0; probe < kProbes; probe++, slot = (slot + 1) % kSlots) {
			uint64_t current = m_keys[slot].load(std::memory_order_relaxed);
			if (current == 0 &&
			    m_keys[slot].compare_exchange_strong(current, key, std::memory_order_relaxed)) {
				current = key;
			}
			if (current == key) {
				m_counts[slot].fetch_add(1, std::memory_order_relaxed);
				return;
			}
		}
		m_dropped.fetch_add(1, std::memory_order_relaxed);
	}

	// The most faulting blocks, descending; clears the table and returns how many slots were used.
	size_t TakeTop(std::array<Entry, kTop>& top, uint64_t& dropped) noexcept {
		size_t used = 0;
		for (size_t slot = 0; slot < kSlots; slot++) {
			const auto key = m_keys[slot].exchange(0, std::memory_order_relaxed);
			const auto cnt = m_counts[slot].exchange(0, std::memory_order_relaxed);
			if (key == 0 || cnt == 0) {
				continue;
			}
			Entry entry {(key - 1) << 16, cnt};
			size_t at = used < kTop ? used : kTop;
			if (used < kTop) {
				used++;
			} else if (top[kTop - 1].count >= cnt) {
				continue;
			} else {
				at = kTop - 1;
			}
			top[at] = entry;
			for (; at > 0 && top[at - 1].count < top[at].count; at--) {
				std::swap(top[at - 1], top[at]);
			}
		}
		dropped = m_dropped.exchange(0, std::memory_order_relaxed);
		return used;
	}

private:
	std::array<std::atomic<uint64_t>, kSlots> m_keys {};
	std::array<std::atomic<uint64_t>, kSlots> m_counts {};
	std::atomic<uint64_t>                     m_dropped {0};
};

inline FaultHotBlocks g_hot_blocks {};

} // namespace Common::FaultStats

#endif // KYTY_COMMON_FAULT_STATS_H_
