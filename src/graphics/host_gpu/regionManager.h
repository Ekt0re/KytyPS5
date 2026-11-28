#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "common/faultStats.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <windows.h>
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		uint64_t   wait_start = 0;
		uint32_t   spins      = 0;
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
			if (wait_start == 0) {
				wait_start = Common::FaultStats::NowNs();
			}
			// Wait on reads, so the holder's cache line is not written until the lock is free.
			// The holder may be inside a protection syscall: after a short spin give its CPU back
			// instead of burning ours, and measure the wait for the [fault] line.
			while (m_lock.test(std::memory_order_relaxed)) {
				if (++spins > 256) {
					std::this_thread::yield();
					continue;
				}
#if defined(__x86_64__) || defined(_M_X64)
				_mm_pause();
#elif defined(_M_ARM64)
				__yield();
#elif defined(__aarch64__)
				asm volatile("yield");
#endif
			}
		}
		m_owner.store(thread, std::memory_order_relaxed);
		if (wait_start != 0) {
			Common::FaultStats::NoteLockWait(Common::FaultStats::NowNs() - wait_start);
		}
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		m_cpu_dirty.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(offset, size);
		return GetBits<source>().Any(start, end);
	}

	// Whether every page of the range is already in the state. Reads the bits without the lock.
	template <DirtySource source>
	[[nodiscard]] bool IsAllModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(offset, size);
		return GetBits<source>().All(start, end);
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t offset, uint64_t size) {
		const auto [start, end] = GetPageRange(offset, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (m_gpu_dirty.Any(start, end)) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		auto&      bits = GetBits<source>();
		RegionBits changed = enable ? RegionBits(~bits, start, end) : RegionBits(bits, start, end);
		if constexpr (source == DirtySource::Cpu && enable) {
			// Called under the lock. Counts re-armed pages that fault again; off unless the storm
			// threshold was set, so normal runs keep their existing protection behaviour.
			if (Common::FaultStats::StormEnabled()) {
				NoteWriteFaults(changed);
			}
		}
		if constexpr (enable) {
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>(changed);
		} else {
			UpdateProtection<enable, true>(changed);
		}
	}

	// Marks the pages of the range GPU dirty unless they are CPU dirty, and calls func(address,
	// size) for the runs it marked. Callers hold the lock.
	template <typename Func>
	void MarkGpuDirtyUnlessCpuDirty(uint64_t offset, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(offset, size);
		RegionBits mask(~m_cpu_dirty, start, end);
		if (mask.None()) {
			return;
		}
		// Only the pages that were not GPU dirty yet start being read-watched.
		auto either = m_cpu_dirty;
		either |= m_gpu_dirty;
		RegionBits changed(~either, start, end);
		for (const auto [first, last]: mask) {
			m_gpu_dirty.SetRange(first, last);
		}
		UpdateProtection<true, true>(changed);
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// Pages of the range that are neither CPU- nor GPU-modified become CPU-modified, which drops
	// their write watch, in one protection change (a guest write fault's window).
	void MarkCleanAsCpuModified(uint64_t offset, uint64_t size) {
		const auto [start, end] = GetPageRange(offset, size);
		RegionBits changed(~m_cpu_dirty, start, end);
		for (const auto [first, last]: RegionBits(m_gpu_dirty, start, end)) {
			changed.UnsetRange(first, last);
		}
		if (changed.None()) {
			return;
		}
		for (const auto [first, last]: changed) {
			m_cpu_dirty.SetRange(first, last);
		}
		UpdateProtection<false, false>(changed);
	}

	// With hold_hot, pages the storm rule marked are reported but keep their dirty bit and stay
	// unwatched, so the guest writes them without faulting; the caller uploads them again on every
	// pass. Only for CPU-dirty bits, and never for a range the GPU is about to write.
	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t offset, uint64_t size, Func&& func,
	                          bool hold_hot = false) {
		const auto [start, end] = GetPageRange(offset, size);
		auto&      bits         = GetBits<source>();
		if (!bits.Any(start, end)) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			if (source == DirtySource::Cpu && hold_hot && m_storm != nullptr) {
				RegionBits release(mask);
				HoldHotPages(mask, release);
				for (const auto [first, last]: release) {
					bits.UnsetRange(first, last);
				}
				UpdateProtection<true, false>(release);
			} else {
				bits.UnsetRange(start, end);
				if constexpr (source == DirtySource::Cpu) {
					UpdateProtection<true, false>(mask);
				} else {
					UpdateProtection<false, true>(mask);
				}
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	TrackingSpinLock lock;

private:
	// Per page: faults counted in the current window, and the hold of a marked page. Frames are
	// kept modulo 2^16; the window and the hold are at most 30000 frames.
	struct PageStorm {
		uint16_t start  = 0;
		uint16_t until  = 0;
		uint8_t  count  = 0;
		uint8_t  active = 0;
	};

	static uint16_t StormFrame() {
		return static_cast<uint16_t>(Common::FaultStats::Frame());
	}

	// Called under the lock for pages that fault right after being re-armed.
	void NoteWriteFaults(const RegionBits& changed) {
		const auto faults = Common::FaultStats::g_storm.faults.load(std::memory_order_relaxed);
		const auto window = Common::FaultStats::g_storm.frames.load(std::memory_order_relaxed);
		const auto hold   = Common::FaultStats::g_storm.hold.load(std::memory_order_relaxed);
		if (hold == 0) {
			return;
		}
		if (m_storm == nullptr) {
			m_storm = std::make_unique<PageStorm[]>(TRACKER_REGION_PAGES);
		}
		const auto frame = StormFrame();
		for (const auto [first, last]: changed) {
			for (auto page = first; page < last; page++) {
				auto& state = m_storm[page];
				if (state.active != 0) {
					continue;
				}
				if (state.count == 0 || static_cast<uint16_t>(frame - state.start) >= window) {
					state.start = frame;
					state.count = 0;
				}
				if (++state.count >= faults) {
					state.active = 1;
					state.until  = static_cast<uint16_t>(frame + hold);
					state.count  = 0;
					Common::PerfSummary::Add(Common::PerfSummary::Counter::FaultStormMarked);
				}
			}
		}
	}

	// `release` starts as `mask`; hot pages it holds are removed from it, expired ones re-armed.
	void HoldHotPages(const RegionBits& mask, RegionBits& release) {
		const auto frame = StormFrame();
		uint64_t   kept = 0, expired = 0;
		for (const auto [first, last]: mask) {
			for (auto page = first; page < last; page++) {
				auto& state = m_storm[page];
				if (state.active == 0) {
					continue;
				}
				if (static_cast<uint16_t>(frame - state.until) < 0x8000u) {
					state.active = 0;
					state.count  = 0;
					expired++;
					continue;
				}
				release.Unset(page);
				kept++;
			}
		}
		if (kept != 0) {
			Common::PerfSummary::Add(Common::PerfSummary::Counter::FaultStormKept, kept);
		}
		if (expired != 0) {
			Common::PerfSummary::Add(Common::PerfSummary::Counter::FaultStormExpired, expired);
		}
	}

	// CPU-dirty pages are not write-watched; GPU-dirty pages are read-watched.
	template <bool track, bool is_read>
	void UpdateProtection(RegionBits& changed) {
		if (!changed.None()) {
			m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, changed);
		}
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] static std::pair<size_t, size_t> GetPageRange(uint64_t offset, uint64_t size) {
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager&                   m_page_manager;
	uint64_t                       m_cpu_addr = 0;
	RegionBits                     m_cpu_dirty;
	RegionBits                     m_gpu_dirty;
	std::unique_ptr<PageStorm[]>   m_storm;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
