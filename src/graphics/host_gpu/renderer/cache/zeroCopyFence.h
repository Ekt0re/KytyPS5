#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_ZEROCOPYFENCE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_ZEROCOPYFENCE_H_

#include "graphics/host_gpu/memoryTracker.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Zero-copy read fences (BufferCache): the newest tick that reads each zero-copy buffer, by buffer
// start. The GPU thread records them; faulting threads look them up before a CPU write proceeds,
// and count the writes that had to wait (a buffer that keeps making the CPU wait goes back to
// VRAM, where an upload is a snapshot).
class ZeroCopyFences {
public:
	struct Recorded {
		bool     fresh = false; // false: this tick fenced the buffer already
		uint32_t waits = 0;     // recent CPU writes that waited for its reads (decaying)
	};

	Recorded Record(uint64_t start, uint64_t end, uint64_t tick) {
		std::scoped_lock lock(m_mutex);
		auto& fence = m_fences[start];
		if (fence.end == end && fence.tick == tick) {
			return {false, fence.waits};
		}
		fence.end  = end;
		fence.tick = tick;
		// Halve the count every 64 fences: it measures waits per use, not since creation.
		if ((++fence.fences & 63u) == 0) {
			fence.waits /= 2;
		}
		m_any.store(true, std::memory_order_release);
		return {true, fence.waits};
	}

	// The newest unreleased tick of the fences that meet [vaddr, vaddr + size), 0 for none.
	[[nodiscard]] uint64_t Find(uint64_t vaddr, uint64_t size) {
		if (!m_any.load(std::memory_order_acquire)) {
			return 0;
		}
		uint64_t tick = 0;
		std::scoped_lock lock(m_mutex);
		ForEachMeeting(vaddr, size, [&](Fence& fence) { tick = std::max(tick, fence.tick); });
		return tick;
	}

	// Counts a CPU write that waited for `tick` on the fences that meet the range.
	void CountWait(uint64_t vaddr, uint64_t size, uint64_t tick) {
		std::scoped_lock lock(m_mutex);
		ForEachMeeting(vaddr, size, [&](Fence& fence) {
			if (fence.tick == tick) {
				fence.waits++;
			}
		});
	}

	// The fence's tick completed; a newer read replaced it otherwise. The wait count stays.
	void Release(uint64_t start, uint64_t tick) {
		std::scoped_lock lock(m_mutex);
		if (const auto it = m_fences.find(start); it != m_fences.end() && it->second.tick == tick) {
			it->second.tick = 0;
		}
	}

	void Forget(uint64_t start) {
		std::scoped_lock lock(m_mutex);
		m_fences.erase(start);
	}

	[[nodiscard]] uint32_t Waits(uint64_t start) {
		std::scoped_lock lock(m_mutex);
		const auto it = m_fences.find(start);
		return it != m_fences.end() ? it->second.waits : 0;
	}

private:
	struct Fence {
		uint64_t end    = 0;
		uint64_t tick   = 0;
		uint32_t waits  = 0;
		uint32_t fences = 0;
	};

	// Buffers never overlap, so the fences that meet a range are adjacent.
	template <typename Func>
	void ForEachMeeting(uint64_t vaddr, uint64_t size, Func&& func) {
		auto it = m_fences.lower_bound(vaddr + size);
		while (it != m_fences.begin()) {
			--it;
			if (it->second.end <= vaddr) {
				break;
			}
			func(it->second);
		}
	}

	std::mutex                m_mutex;
	std::map<uint64_t, Fence> m_fences;
	std::atomic<bool>         m_any {false};
};

// Gives the GPU-written pages of a zero-copy range back to CPU reads once their writes completed,
// run by run: only those pages change protection, and they stay write-watched (a CPU write then
// faults into the read fence instead of the whole buffer turning writable and being protected
// again by the next read). Calls on_run(address, bytes) for each run.
template <typename Func>
void ReleaseGpuWrittenRuns(MemoryTracker& tracker, uint64_t vaddr, uint64_t size, Func&& on_run) {
	std::vector<std::pair<uint64_t, uint64_t>> runs;
	tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept { runs.emplace_back(address, bytes); });
	for (const auto& [address, bytes]: runs) {
		tracker.UnmarkRegionAsGpuModified(address, bytes);
		on_run(address, bytes);
	}
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_ZEROCOPYFENCE_H_ */
