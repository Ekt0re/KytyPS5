#ifndef KYTY_COMMON_PERF_SUMMARY_H_
#define KYTY_COMMON_PERF_SUMMARY_H_

#include <atomic>
#include <chrono>
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
	// Synchronous buffer readbacks (BufferCache::ReadMemory on the GPU thread), per phase.
	RbCalls,      // ReadMemory requests the GPU thread ran.
	RbBatches,    // Synchronous downloads (one or more buffers copied in one go).
	RbSubmits,    // Queue submissions a synchronous download made (its copy, flushes).
	RbFused,      // Downloads whose copy went into the same queue submission as the open work.
	RbExtraBufs,  // Buffers a batch copied besides the one the request was for.
	RbNeighbour,  // Downloads of a window next to one of the last 8 downloaded (prefetch potential).
	RbHandlerUs,  // Microseconds in ReadMemory on the GPU thread (all phases).
	RbFlushUs,    // ... submitting the open command buffer (a detached download's flush).
	RbSubmitUs,   // ... recording and submitting the copy.
	RbWaitUs,     // ... waiting for the copy (fence, tick and earlier publications).
	RbCopyUs,     // ... copying the bytes into guest memory.
	RbWriteCalls, // ReadMemory requests of guest write faults on GPU-written pages.
	RbWriteUs,    // Their microseconds (part of rb_handler_us).
	// KYTY_RB_QUEUE_PROBE=1, per synchronous download: how much GPU work is queued ahead of it.
	RbQueueTicks,       // Submitted, unfinished ticks when it starts (sum).
	RbProducerDone,     // Downloads whose buffers' last write had already finished.
	RbProducerPending,  // Downloads whose producing tick was submitted and still running.
	RbProducerOpen,     // Downloads whose producing tick is the open (unsubmitted) one.
	RbAfterTicks,       // Pending: submitted ticks queued behind the producing tick (sum).
	RbProducerWaitUs,   // Pending: microseconds until the producing tick finished.
	// Synchronous downloads per path (ReadbackBatch::ReadbackPath): count, microseconds submitting
	// and waiting for the copy (fused: with the open command buffer's submission), microseconds
	// from recording the copy to the end of the wait (with flushes).
	RbPathCq,         // Copy queue (KYTY_COPY_QUEUE): own queue, open command buffer stays open.
	RbPathCqWaitUs,
	RbPathCqTotalUs,
	RbPathFused,      // Detached copy in the same submission as the open command buffer.
	RbPathFusedWaitUs,
	RbPathFusedTotalUs,
	RbPathDetached,   // Detached copy in a submission of its own, then the open command buffer.
	RbPathDetachedWaitUs,
	RbPathDetachedTotalUs,
	RbPathFlush,      // A write of the ranges is in the open command buffer: flush and wait.
	RbPathFlushWaitUs,
	RbPathFlushTotalUs,
	RbCqSubmits,      // Copy-queue submissions.
	RbCqFenceUs,      // ... microseconds waiting for their fence (part of rb_path_cq_wait_us).
	RbCqGpuBusy,      // Copy-queue downloads that found submitted graphics work unfinished.
	// Zero-copy guest memory (KYTY_ZERO_COPY=1, BufferCache).
	ZcResolveUs,  // Microseconds returning GPU-written zero-copy pages on a CPU fault (with waits).
	ZcReleaseUs,  // ... returning them when their last write completed.
	ZcFences,     // Read fences (one per buffer and tick).
	ZcFenceSkips, // Binds that found this tick's fence already in place.
	ZcFenceUs,    // Microseconds in read fences.
	ZcReadWaits,  // CPU writes that waited for a GPU read in flight.
	ZcWaitUs,     // ... their microseconds.
	ZcWaitDrops,  // Buffers moved back to VRAM because the CPU kept waiting.
	ZcStateKiB,   // KiB whose tracking state the zero-copy steps changed.
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

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled.load(std::memory_order_relaxed);
}

// Adds the microseconds of its lifetime to `counter` (no clock read while the summary is off).
class ScopedMicros {
public:
	explicit ScopedMicros(Counter counter, bool active = true) noexcept
	    : m_counter(counter), m_on(active && Enabled()) {
		if (m_on) {
			m_start = std::chrono::steady_clock::now();
		}
	}
	~ScopedMicros() {
		if (m_on) {
			Add(m_counter, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
			                                         std::chrono::steady_clock::now() - m_start)
			                                         .count()));
		}
	}
	ScopedMicros(const ScopedMicros&)            = delete;
	ScopedMicros& operator=(const ScopedMicros&) = delete;

private:
	Counter                               m_counter;
	bool                                  m_on;
	std::chrono::steady_clock::time_point m_start {};
};

// Reads KYTY_PERF_SUMMARY and starts the reporter. Safe to call more than once.
void Start();

// Called once on the GPU command thread: its CPU time, instructions and context switches are
// the per-frame cost the summary reports.
void RegisterGpuThread();

} // namespace Common::PerfSummary

#endif /* KYTY_COMMON_PERF_SUMMARY_H_ */
