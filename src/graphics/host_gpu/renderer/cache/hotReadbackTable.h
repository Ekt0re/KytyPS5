#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOTREADBACKTABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOTREADBACKTABLE_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Libs::Graphics {

inline constexpr uint64_t HotReadbackTicks      = 512;
inline constexpr size_t   MaxHotReadbackEntries = 64;
inline constexpr uint64_t MaxHotReadbackBytes   = 4ull * 1024 * 1024;

// Recently read-back buffers, addressed without retaining a resource reference. GPU thread only.
class HotReadbackTable {
public:
	void Note(uint64_t address, uint64_t size, uint64_t tick) {
		for (auto& entry: m_entries) {
			if (entry.address == address) {
				entry.size = size;
				entry.tick = tick;
				return;
			}
		}
		if (m_entries.size() >= MaxHotReadbackEntries) {
			const auto oldest = std::min_element(
			    m_entries.begin(), m_entries.end(),
			    [](const Entry& left, const Entry& right) { return left.tick < right.tick; });
			*oldest = {address, size, tick};
			return;
		}
		m_entries.push_back({address, size, tick});
	}

	template <class Visit> void ForEachHot(uint64_t now_tick, Visit&& visit) {
		std::erase_if(m_entries, [now_tick](const Entry& entry) {
			return now_tick > entry.tick && now_tick - entry.tick > HotReadbackTicks;
		});
		for (const auto& entry: m_entries) {
			visit(entry.address, entry.size);
		}
	}

	[[nodiscard]] bool TickOf(uint64_t address, uint64_t& tick) const {
		for (const auto& entry: m_entries) {
			if (entry.address == address) {
				tick = entry.tick;
				return true;
			}
		}
		return false;
	}

	[[nodiscard]] size_t Size() const { return m_entries.size(); }

private:
	struct Entry {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t tick    = 0;
	};
	std::vector<Entry> m_entries;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HOTREADBACKTABLE_H_
