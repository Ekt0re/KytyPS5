#ifndef KYTY_COMMON_PERF_SUMMARY_H_
#define KYTY_COMMON_PERF_SUMMARY_H_

#include <atomic>
#include <cstdint>

// KYTY_PERF_SUMMARY=1: a work-based performance summary on stdout, one "PerfSummary ..." line
// every 10 s (KYTY_PERF_SUMMARY=n: every n s, n >= 2) and one "PerfSummary total ..." line at exit.
// It counts work per guest frame (flip) rather than frames per second, so that two builds can be
// compared while other programs load the machine: CPU time and retired instructions of the GPU
// command thread, and how often the expensive paths ran. Off, a counter costs one relaxed load.
namespace Common::PerfSummary {

enum class Counter {
	Flips,
	Draws,
	Dispatches, // Direct and indirect.
	IndirectDispatches,
	WriteFaults, // Guest accesses to GPU-tracked pages.
	ReadFaults,
	BufferReadbacks,
	TextureReadbacks,
	TextureLookups,
	BufferLookups,
	DescriptorWrites,
	HeapDescriptorWrites, // Image heap elements.
	PipelineLookups,
	PipelinesCreated,
	AmmMaps, // AMM map commands of every kind (direct, automatic, PRT).
	AmmUnmaps,
	GpuUnmaps, // Guest unmaps the GPU thread serviced (they drop cached images and buffers).
	HeapRebinds,    // Image heap discoveries (one per draw or dispatch that uses the heap).
	HeapLookups,    // Texture lookups those made.
	HeapRangeSkips, // Lookups saved because the image set changed only at other addresses.
	DepthRecreates, // Depth images replaced by one of another format, layout or binding kind.
	ImageBarriers,     // Image memory barriers of cached images (Image::GetBarriers).
	LayoutTransitions, // Those that change the layout.
	Count
};

namespace Detail {
inline std::atomic<bool>     g_enabled {false};
inline std::atomic<uint64_t> g_counters[static_cast<int>(Counter::Count)] {};
} // namespace Detail

inline void Add(Counter counter, uint64_t value = 1) noexcept {
	if (Detail::g_enabled.load(std::memory_order_relaxed)) {
		Detail::g_counters[static_cast<int>(counter)].fetch_add(value, std::memory_order_relaxed);
	}
}

// Reads KYTY_PERF_SUMMARY and starts the reporter. Safe to call more than once.
void Start();

// Called once on the GPU command thread: its CPU time, instructions and context switches are
// the per-frame cost the summary reports.
void RegisterGpuThread();

} // namespace Common::PerfSummary

#endif /* KYTY_COMMON_PERF_SUMMARY_H_ */
