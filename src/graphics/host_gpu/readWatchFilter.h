#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_READWATCHFILTER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_READWATCHFILTER_H_

#include <atomic>
#include <cstdint>

// Which guest pages PageManager made inaccessible (read watched: they hold GPU-written bytes the
// CPU has not got yet), as one counter per page slot. Slots repeat every 4 GiB, so a count may
// stand for another page: a zero count proves the page is not read watched, a nonzero one only
// suggests it. Code that would otherwise take an access fault per load (the SRT walker) asks here
// first. Counts change under PageManager's region locks; readers take no lock, and a stale answer
// only sends a load down the path it took before this filter existed.
namespace Libs::Graphics::ReadWatchFilter {

constexpr uint64_t PageShift = 12;
constexpr uint64_t SlotCount = uint64_t {1} << 20u;

static_assert(std::atomic<uint16_t>::is_always_lock_free && sizeof(std::atomic<uint16_t>) == 2);

inline std::atomic<uint16_t> g_counts[SlotCount] {};

inline void Add(uint64_t vaddr) noexcept {
	g_counts[(vaddr >> PageShift) & (SlotCount - 1)].fetch_add(1, std::memory_order_relaxed);
}

inline void Remove(uint64_t vaddr) noexcept {
	g_counts[(vaddr >> PageShift) & (SlotCount - 1)].fetch_sub(1, std::memory_order_relaxed);
}

} // namespace Libs::Graphics::ReadWatchFilter

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_READWATCHFILTER_H_
