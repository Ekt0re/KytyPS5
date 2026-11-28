#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_COMPILEPAUSE_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_COMPILEPAUSE_H_

#include "common/timer.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(_M_X64) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#endif

// The guest clock stands still while a GPU queue waits for a shader translation or a pipeline
// build: a console builds no pipelines, and a title watchdog that counts the seconds its main
// thread spends without progress would otherwise fire during the compile. While held,
// sceKernelReadTsc, the process time and the monotonic clocks do not move, and guest sleeps last
// until the guest time they asked for has passed. The wall clock keeps running.
// Nested and concurrent holds share one pause. KYTY_PAUSE_ON_COMPILE=0 turns it off.
// After AnyPS5 62a241e4, 139b502c, e68fe527 (simnJS/AnyPS5, GPL-2.0).
namespace Common::CompilePause {

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PAUSE_ON_COMPILE");
		return value == nullptr || std::strcmp(value, "0") != 0;
	}();
	return enabled;
}

// The host counter that sceKernelReadTsc is based on.
inline uint64_t RawTicks() {
#if defined(_M_X64) || defined(__x86_64__)
	return __rdtsc();
#else
	return SteadyNs();
#endif
}

struct ClockState {
	std::mutex            mutex;
	int                   depth = 0;
	std::atomic<uint32_t> sequence {0};
	std::atomic<bool>     paused {false};
	// Raw readings when the current pause began.
	std::atomic<uint64_t> paused_at_ns {0};
	std::atomic<uint64_t> paused_at_ticks {0};
	// Total held time of the finished pauses.
	std::atomic<uint64_t> held_ns {0};
	std::atomic<uint64_t> held_ticks {0};
};

inline ClockState& State() {
	static ClockState state;
	return state;
}

inline void Pause() {
	if (!Enabled()) return;
	auto&            state = State();
	std::scoped_lock lock(state.mutex);
	if (state.depth++ != 0) return;
	state.sequence.fetch_add(1);
	state.paused_at_ns.store(SteadyNs());
	state.paused_at_ticks.store(RawTicks());
	state.paused.store(true);
	state.sequence.fetch_add(1);
}

inline void Resume() {
	if (!Enabled()) return;
	auto&            state = State();
	std::scoped_lock lock(state.mutex);
	if (state.depth == 0 || --state.depth != 0) return;
	const auto now_ns    = SteadyNs();
	const auto now_ticks = RawTicks();
	const auto at_ns     = state.paused_at_ns.load();
	const auto at_ticks  = state.paused_at_ticks.load();
	state.sequence.fetch_add(1);
	const auto held_before = state.held_ns.load();
	const auto held_after  = held_before + (now_ns > at_ns ? now_ns - at_ns : 0);
	state.held_ns.store(held_after);
	state.held_ticks.store(state.held_ticks.load() + (now_ticks > at_ticks ? now_ticks - at_ticks : 0));
	state.paused.store(false);
	state.sequence.fetch_add(1);
	constexpr uint64_t ReportNs = 5000000000ull;
	if (held_after / ReportNs != held_before / ReportNs) {
		std::fprintf(stderr, "Guest clock: held %.1f s in total while GPU queues waited for compiles\n",
		             static_cast<double>(held_after) / 1e9);
	}
}

inline bool Paused() {
	return State().paused.load(std::memory_order_relaxed);
}

namespace Detail {
// A host clock reading on the guest clock; `at` and `held` pick that clock's pause start and total.
template <typename Read, typename At, typename Held>
uint64_t GuestValue(Read read, At at, Held held) {
	auto& state = State();
	for (;;) {
		const auto sequence = state.sequence.load();
		if ((sequence & 1u) != 0) continue;
		const uint64_t base  = state.paused.load() ? at(state) : read();
		const uint64_t total = held(state);
		if (state.sequence.load() != sequence) continue;
		return base > total ? base - total : 0;
	}
}
} // namespace Detail

// SteadyNs() (CLOCK_MONOTONIC on Linux) on the guest clock.
inline uint64_t GuestNowNs() {
	return Detail::GuestValue(
	    [] { return SteadyNs(); }, [](ClockState& s) { return s.paused_at_ns.load(); },
	    [](ClockState& s) { return s.held_ns.load(); });
}

// RawTicks() on the guest clock.
inline uint64_t GuestTicks() {
	return Detail::GuestValue(
	    [] { return RawTicks(); }, [](ClockState& s) { return s.paused_at_ticks.load(); },
	    [](ClockState& s) { return s.held_ticks.load(); });
}

// Holds the guest clock while a GPU queue waits for a compile.
struct Scope {
	Scope() { Pause(); }
	~Scope() { Resume(); }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
};

// Tracks a presented-frame stall for KYTY_DEFER_NEW_TARGETS. The caller skips Update while the
// threshold is zero, so the default path neither reads the clock nor changes this state.
struct FrameStallTracker {
	enum class Change : uint8_t { None, Started, Ended };

	uint64_t frame    = ~uint64_t {0};
	uint64_t since_ms = 0;
	bool     stalled  = false;
	// Draws actually skipped because the stall made them deferrable.
	uint64_t deferred       = 0;
	uint64_t deferred_total = 0;

	void NoteDeferred() {
		deferred++;
		deferred_total++;
	}

	bool Update(uint64_t frame_now, uint64_t now_ms, uint32_t stall_ms, Change* change) {
		Change edge = Change::None;
		if (stall_ms == 0) {
			if (change != nullptr) *change = edge;
			return false;
		}
		if (frame_now != frame) {
			if (stalled) edge = Change::Ended;
			frame    = frame_now;
			since_ms = now_ms;
			stalled  = false;
		}
		const bool want = now_ms >= since_ms && now_ms - since_ms >= stall_ms;
		if (want && !stalled) {
			stalled  = true;
			deferred = 0;
			edge     = Change::Started;
		}
		if (change != nullptr) *change = edge;
		return stalled;
	}
};

} // namespace Common::CompilePause

#endif
