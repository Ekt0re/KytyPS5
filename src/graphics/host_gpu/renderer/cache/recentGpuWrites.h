#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_RECENTGPUWRITES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_RECENTGPUWRITES_H_

#include "graphics/host_gpu/regionDefinitions.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace Libs::Graphics {

inline constexpr uint64_t RecentWriteMaxBytes    = 64ull * 1024;
inline constexpr size_t   RecentGpuWriteCapacity = 1024;

// Small GPU writes with the scheduler tick that recorded them. Entries hold no resource reference;
// the memory tracker and pending-write checks still decide whether a span is safe to download.
class RecentGpuWriteRing {
public:
	void Note(uint64_t address, uint64_t size, uint64_t tick) {
		if (size == 0 || size > RecentWriteMaxBytes) {
			return;
		}
		if (m_count > 0) {
			auto& newest = m_entries[NewestIndex()];
			if (newest.address == address && newest.size == size) {
				newest.tick = tick;
				return;
			}
		}
		m_entries[m_next] = {address, size, tick};
		m_next            = (m_next + 1) % RecentGpuWriteCapacity;
		m_count           = std::min(m_count + 1, RecentGpuWriteCapacity);
	}

	// Newest first, rounded out to tracker pages. `finished` compares the recorded tick with the
	// last completed GPU tick. Returning false from `visit` stops the walk.
	template <class Visit> void ForEachNewest(uint64_t known_tick, Visit&& visit) const {
		for (size_t age = 0; age < m_count; ++age) {
			const auto& entry =
			    m_entries[(NewestIndex() + RecentGpuWriteCapacity - age) % RecentGpuWriteCapacity];
			const uint64_t begin = entry.address & ~(TRACKER_PAGE_SIZE - 1);
			const uint64_t end =
			    (entry.address + entry.size + TRACKER_PAGE_SIZE - 1) & ~(TRACKER_PAGE_SIZE - 1);
			if (!visit(begin, end - begin, entry.tick <= known_tick)) {
				return;
			}
		}
	}

	[[nodiscard]] size_t Size() const { return m_count; }

private:
	struct Entry {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t tick    = 0;
	};
	[[nodiscard]] size_t NewestIndex() const {
		return (m_next + RecentGpuWriteCapacity - 1) % RecentGpuWriteCapacity;
	}
	std::array<Entry, RecentGpuWriteCapacity> m_entries {};
	size_t                                    m_next  = 0;
	size_t                                    m_count = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_RECENTGPUWRITES_H_
