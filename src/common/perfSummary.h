#ifndef KYTY_COMMON_PERF_SUMMARY_H_
#define KYTY_COMMON_PERF_SUMMARY_H_

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

// KYTY_PERF_SUMMARY=1: a work-based performance summary on stdout, one "PerfSummary ..." line
// every 10 s (KYTY_PERF_SUMMARY=n: every n s, n >= 2) and one "PerfSummary total ..." line at exit.
// It counts work per guest frame (flip) rather than frames per second, so that two builds can be
// compared while other programs load the machine: CPU time and retired instructions of the GPU
// command thread, and how often the expensive paths ran. Off, a counter costs one relaxed load.
// KYTY_FRAME_COST=1 also prints one [frame] line per interval (5 s by default): presented frames
// and the wall-time split into GPU wait, guest wait and resource materialisation, plus a one-shot
// module-miss warning. It uses this same reporter, never a second thread. Off by default.
// KYTY_FAULT_STATS=1 also prints [fault] rate lines from this same reporter (5 s by default):
// write/read fault rates, handler/protection/lock-wait cost and the hottest 64 KiB blocks. The
// fault-storm thresholds come from KYTY_FAULT_STORM_FAULTS/_FRAMES/_HOLD and are off by default.
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
	DepthRecreateUs,   // Microseconds those replacements took on the GPU thread.
	ImagesCreated,     // Cached images created (a Vulkan image each).
	ImagesDestroyed,   // Cached images destroyed (their Vulkan image released at tick end).
	ParkHits,          // Depth replacements that took a parked image back (no image created).
	ParkMisses,        // Depth replacements with parking on that found no parked image of the shape.
	ParkCapEvictions,  // Parked images destroyed because the pool was at its image or MiB cap.
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
	ZcOpenWaits,  // Read waits for the open tick (the GPU thread submits it, the writer waits).
	ZcSaved,      // Read-backs zero-copy spared (GPU writes released in place, faults resolved).
	// KYTY_FRAME_COST=1: wall-time split of a presented frame, printed by the existing reporter.
	// Presented frames are Counter::Flips, which videoOut already counts; do not count them twice.
	FrameWaitGpuUs,     // Microseconds waiting on the GPU (MasterSemaphore::Wait).
	FrameWaitGuestUs,   // Microseconds the guest command thread waited on blocked queues.
	FrameMaterializeUs, // Microseconds in IR resource materialisation.
	FrameModuleLookups, // Shader-module lookups (Program::FindModule).
	FrameModuleHits,    // Lookups that found a module.
	FrameModuleMisses,  // Lookups that did not find one.
	// KYTY_FAULT_STATS=1: cost and outcomes of the write-fault path. WriteFaults/ReadFaults above
	// are counted at the existing RenderContext::HandleFault site and are not counted again here.
	FaultHandlerUs,   // Microseconds in HandleFault (all paths).
	FaultInvalidateUs, // ... in BufferCache::InvalidateMemory (part of the handler).
	FaultProtectUs,   // ... in guest memory protection (part of the handler).
	FaultLockWaitUs,  // ... waiting for a region tracking lock.
	FaultFastSkips,   // Faults on already CPU-dirty pages: no lock was taken.
	FaultStormMarked, // Pages the fault-storm rule marked as always dirty.
	FaultStormKept,   // Marked pages kept unwatched this interval.
	FaultStormExpired, // Marked pages that were re-armed this interval.
	// KYTY_RECORDER=1: the recording thread (recorder.cpp).
	RecRecords,  // Commands queued for it.
	RecDrains,   // Waits until it replayed everything queued (any thread, mostly the GPU thread).
	RecDrainUs,  // Their microseconds.
	// Readbacks by origin (who asked): ReadMemory requests run on the GPU thread, synchronous
	// downloads they made (one GPU wait each) and microseconds from recording to the end of the wait.
	// guest = guest-thread faults and reads, srt = SRT walker loads, pm4 = other command-thread read
	// faults (the PM4 parser), cmdwr = command-thread write faults, args = indirect dispatch/draw
	// arguments, shader = command-thread backing reads (ReadGuestMemory, shader constants),
	// tex = color metadata and image heap reads, other = unmaps and the rest.
	RbOGuestCalls,
	RbOGuestWaits,
	RbOGuestWaitUs,
	RbOSrtCalls,
	RbOSrtWaits,
	RbOSrtWaitUs,
	RbOPm4Calls,
	RbOPm4Waits,
	RbOPm4WaitUs,
	RbOCmdwrCalls,
	RbOCmdwrWaits,
	RbOCmdwrWaitUs,
	RbOArgsCalls,
	RbOArgsWaits,
	RbOArgsWaitUs,
	RbOShaderCalls,
	RbOShaderWaits,
	RbOShaderWaitUs,
	RbOTexCalls,
	RbOTexWaits,
	RbOTexWaitUs,
	RbOOtherCalls,
	RbOOtherWaits,
	RbOOtherWaitUs,
	Count
};

// Levels rather than counts: each summary line reports the highest level set in its interval
// (the total line: in the whole run).
enum class Gauge {
	ParkedMiB, // Memory of the texture cache's parked images.
	DeviceMiB, // Device-local memory the emulator uses (VK_EXT_memory_budget), at each suspend point.
	Count
};

namespace Detail {
inline std::atomic<bool>     g_enabled {false};
inline std::atomic<uint64_t> g_counters[static_cast<int>(Counter::Count)] {};
inline std::atomic<uint64_t> g_gauge_peaks[static_cast<int>(Gauge::Count)] {};
} // namespace Detail

inline void SetGauge(Gauge gauge, uint64_t value) noexcept {
	if (Detail::g_enabled.load(std::memory_order_relaxed)) {
		auto& peak = Detail::g_gauge_peaks[static_cast<int>(gauge)];
		auto  seen = peak.load(std::memory_order_relaxed);
		while (seen < value && !peak.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
		}
	}
}

inline void Add(Counter counter, uint64_t value = 1) noexcept {
	if (Detail::g_enabled.load(std::memory_order_relaxed)) {
		Detail::g_counters[static_cast<int>(counter)].fetch_add(value, std::memory_order_relaxed);
	}
}

[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_enabled.load(std::memory_order_relaxed);
}

namespace Detail {

// Pure formatters for the [frame] line. They take interval totals (not per-frame values) so the
// reporter and the unit test can share exactly the same arithmetic. The returned strings end in
// a newline; `FormatFrameWarning` returns an empty string until the miss streak is long enough.
std::string FormatFrameLine(double secs, uint64_t frames, uint64_t wait_gpu_us,
                            uint64_t wait_guest_us, uint64_t materialize_us,
                            uint64_t module_lookups, uint64_t module_hits,
                            uint64_t module_misses);
std::string FormatFrameWarning(uint64_t hits, uint64_t misses, double sustained_seconds);
struct FaultBlock {
	uint64_t address; // Aligned down to 64 KiB.
	uint64_t count;
};

// Pure formatters shared by the reporter and the unit test. Values are interval totals; rates and
// milliseconds per second are derived from `secs`. Returned strings end in a newline.
std::string FormatFaultLine(double secs, uint64_t writes, uint64_t reads, uint64_t handler_us,
                            uint64_t invalidate_us, uint64_t protect_us, uint64_t lock_wait_us,
                            uint64_t fast_skips, uint64_t marked, uint64_t kept, uint64_t expired);
std::string FormatFaultHotLine(const FaultBlock* blocks, size_t count, uint64_t dropped);

} // namespace Detail

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

// Called once on the recording thread (KYTY_RECORDER=1): its CPU time is reported as rec_cpu_ms.
void RegisterRecorderThread();

} // namespace Common::PerfSummary

#endif /* KYTY_COMMON_PERF_SUMMARY_H_ */
