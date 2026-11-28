#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_PERFAB_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_PERFAB_H_

#include <atomic>
#include <cstdlib>

// Diagnostic (KYTY_PERF_AB=1 with KYTY_SYNC_STATS=n): the per-call CPU cost switches
// KYTY_LRU_TOUCH_ONCE, KYTY_IMAGE_PAGE_COUNTS and KYTY_ASYNC_IMAGE_DESTROY alternate off and on
// every SyncStats period, starting off, so one run measures both arms in the same scene. Each
// SyncStats line names the arm it measured (arm=off|on), and with --profile a Tracy zone
// "PerfAb arm off" or "PerfAb arm on" spans each period on the SyncStats thread.
// KYTY_RANGESET_BLOCKS chooses a data structure at startup and does not alternate.
namespace Common::PerfAb {

inline std::atomic<int> g_arm {-1}; // -1: not alternating, 0: off, 1: on.

inline bool Enabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_AB");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

// The state of a switch whose environment setting is `configured`: the current arm while
// alternating, else the setting.
inline bool Apply(bool configured) {
	const int arm = g_arm.load(std::memory_order_relaxed);
	return arm < 0 ? configured : arm == 1;
}

} // namespace Common::PerfAb

#endif
