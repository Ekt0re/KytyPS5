#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKWAIT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKWAIT_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

// Shorter GPU waits for buffer readbacks. Pure logic, no Vulkan (BufferCache does the copies).
// Ported from TheCruZ KytyEmu d81455ac (WriteTicks, NoteReadbackWait, TakeFlushRequest, the
// readback mirror's bookkeeping) and 2fdd7744 (mirror slots kept per read).
//
// - WriteTicks: the tick of the command buffer that last wrote each guest range a cached buffer
//   holds, so a readback on the copy queue waits on the GPU for that tick only, not for all work
//   submitted since (toggle rb_tick_wait).
// - HotWriters: ranges whose readbacks found their writer still in the open command buffer (a
//   flush and wait of everything recorded): an operation that writes them again is submitted
//   right after it is recorded, so the next readback finds it submitted (toggle rb_tick_wait).
// - FlushThrottle: such requested submissions happen at most once per interval.
// - MirrorIndex: a host-visible copy of the 64-byte lines the command thread reads again and
//   again, refreshed at the end of each command buffer that wrote them (toggle rb_mirror).
namespace Libs::Graphics::ReadbackWait {

inline constexpr uint64_t UnknownTick = UINT64_MAX;

class WriteTicks {
public:
	// [begin, end) was last written by command buffer `tick`; ticks assigned only grow.
	void Assign(uint64_t begin, uint64_t end, uint64_t tick) {
		if (begin >= end) {
			return;
		}
		// The operations of one command buffer write the same ranges again and again: a range
		// assigned this tick still has it.
		auto& recent = m_recent[((begin ^ (end << 24u)) * 0x9e3779b97f4a7c15ull) >> 56u];
		if (recent.begin == begin && recent.end == end && recent.tick == tick) {
			return;
		}
		recent  = {begin, end, tick};
		auto it = m_ranges.lower_bound(begin);
		if (it != m_ranges.end() && it->first == begin && it->second.end == end) {
			it->second.tick = tick;
			return;
		}
		if (it != m_ranges.begin()) {
			const auto previous = std::prev(it);
			if (previous->second.end > begin) {
				const auto right       = previous->second;
				previous->second.end   = begin;
				it                     = m_ranges.emplace_hint(it, begin, right);
			}
		}
		while (it != m_ranges.end() && it->first < end) {
			if (it->second.end > end) {
				const auto tail = it->second;
				m_ranges.erase(it);
				m_ranges.emplace(end, tail);
				break;
			}
			it = m_ranges.erase(it);
		}
		m_ranges.emplace(begin, Range {end, tick});
	}
	// The latest tick over [begin, end); UnknownTick when part of it has none.
	[[nodiscard]] uint64_t Latest(uint64_t begin, uint64_t end) const {
		auto it = m_ranges.upper_bound(begin);
		if (it != m_ranges.begin()) {
			--it;
		}
		uint64_t latest = 0;
		uint64_t cursor = begin;
		for (; it != m_ranges.end() && it->first < end && cursor < end; ++it) {
			if (it->second.end <= cursor) {
				continue;
			}
			if (it->first > cursor) {
				return UnknownTick;
			}
			latest = std::max(latest, it->second.tick);
			cursor = it->second.end;
		}
		return cursor >= end ? latest : UnknownTick;
	}
	[[nodiscard]] size_t Size() const noexcept { return m_ranges.size(); }

private:
	struct Range {
		uint64_t end  = 0;
		uint64_t tick = 0;
	};
	struct Recent {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t tick  = 0;
	};
	std::map<uint64_t, Range> m_ranges;
	std::array<Recent, 256>   m_recent {};
};

class HotWriters {
public:
	// Readbacks without a hit after which an entry is dropped.
	static constexpr uint64_t Lifetime   = 4096;
	static constexpr size_t   MaxEntries = 32;

	// A readback of [begin, end) waited for writes still being recorded (`drained`) or for
	// submitted ones; `device_address` when they were device-address writes (which name no range).
	// `page` aligns the remembered ranges (a power of two).
	void NoteWait(uint64_t begin, uint64_t end, bool device_address, bool drained, uint64_t page) {
		const auto count = ++m_count;
		if (device_address && (drained || m_device_address_hot != 0)) {
			m_device_address_hot = count;
		}
		std::erase_if(m_entries, [count](const Entry& hot) { return count - hot.last_hit >= Lifetime; });
		begin &= ~(page - 1);
		end = (end + page - 1) & ~(page - 1);
		for (auto& hot: m_entries) {
			if (begin < hot.end && hot.begin < end) {
				hot.begin    = std::min(hot.begin, begin);
				hot.end      = std::max(hot.end, end);
				hot.last_hit = count;
				return;
			}
		}
		if (drained && !device_address) {
			if (m_entries.size() == MaxEntries) {
				m_entries.erase(std::min_element(m_entries.begin(), m_entries.end(),
				                                 [](const Entry& a, const Entry& b) { return a.last_hit < b.last_hit; }));
			}
			m_entries.push_back({begin, end, count});
		}
	}
	// An operation writing [vaddr, vaddr + size) should be submitted at once.
	[[nodiscard]] bool Intersects(uint64_t vaddr, uint64_t size) const noexcept {
		const auto end = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
		return std::any_of(m_entries.begin(), m_entries.end(),
		                   [&](const Entry& hot) { return vaddr < hot.end && hot.begin < end; });
	}
	// Readbacks recently waited for device-address writes still being recorded: an operation that
	// writes through device addresses should be submitted at once.
	[[nodiscard]] bool DeviceAddressHot() const noexcept {
		return m_device_address_hot != 0 && m_count - m_device_address_hot < Lifetime;
	}
	[[nodiscard]] bool   Empty() const noexcept { return m_entries.empty() && !DeviceAddressHot(); }
	[[nodiscard]] size_t Size() const noexcept { return m_entries.size(); }

private:
	struct Entry {
		uint64_t begin    = 0;
		uint64_t end      = 0;
		uint64_t last_hit = 0;
	};
	std::vector<Entry> m_entries;
	uint64_t           m_count              = 0;
	uint64_t           m_device_address_hot = 0; // m_count at the last such readback (0: none)
};

// Requested submissions: at most one per `interval_us`; a request put off waits for a later
// operation and is dropped when another submission sent the work meanwhile.
class FlushThrottle {
public:
	explicit FlushThrottle(int64_t interval_us = 1000) noexcept: m_interval_us(interval_us) {}

	void               Request() noexcept { m_requested = true; }
	void               Cancel() noexcept {
		m_requested     = false;
		m_deferred_tick = 0;
	}
	[[nodiscard]] bool Requested() const noexcept { return m_requested; }
	// After an operation was recorded into command buffer `tick` at time `now_us`: true to submit
	// it now.
	[[nodiscard]] bool Take(uint64_t tick, int64_t now_us) noexcept {
		if (!m_requested) {
			return false;
		}
		if (m_deferred_tick != 0 && tick != m_deferred_tick) {
			m_requested     = false;
			m_deferred_tick = 0;
			return false;
		}
		if (m_last_us != INT64_MIN && now_us - m_last_us < m_interval_us) {
			m_deferred_tick = tick;
			return false;
		}
		m_last_us       = now_us;
		m_requested     = false;
		m_deferred_tick = 0;
		return true;
	}

private:
	int64_t  m_interval_us;
	int64_t  m_last_us       = INT64_MIN;
	uint64_t m_deferred_tick = 0;
	bool     m_requested     = false;
};

// The mirror's lines: each holds a copy of its 64 bytes in a cached buffer, recorded at the end
// of command buffer `tick` (0: none since the line's last write) into `slot` of a host-visible
// buffer. A write of a line clears its tick and queues it for a copy at the end of its command
// buffer, so a copy whose tick is at least the line's last write tick holds the GPU's bytes once
// that tick completed.
class MirrorIndex {
public:
	static constexpr uint64_t LineSize      = 64;
	static constexpr uint64_t MaxRead       = 256;
	static constexpr uint32_t MaxLines      = 8192;
	// Mirror reads within which a line counts as recently used.
	static constexpr uint64_t FlushLifetime = 4096;
	static constexpr size_t   MaxReadLines  = MaxRead / LineSize + 1;

	struct Line {
		uint32_t slot       = 0;
		uint64_t tick       = 0;
		uint64_t dirty_tick = 0;
		uint64_t last_use   = 0;
	};
	struct Lookup {
		bool                                held = false; // every line has a copy at least `needed`
		uint64_t                            wait = 0;     // the newest copy tick of the lines
		uint64_t                            first = 0;    // the first line's address
		std::array<uint32_t, MaxReadLines>  slots {};
	};

	[[nodiscard]] bool Empty() const noexcept { return m_lines.empty(); }
	[[nodiscard]] size_t Size() const noexcept { return m_lines.size(); }

	// [vaddr, vaddr + size) is written in command buffer `tick`. True when a recently read line
	// was among them (its next read soon wants the writer submitted).
	bool MarkWritten(uint64_t vaddr, uint64_t size, uint64_t tick) {
		bool       recent = false;
		const auto end    = size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size;
		for (auto it = m_lines.lower_bound(vaddr & ~(LineSize - 1)); it != m_lines.end() && it->first < end; ++it) {
			auto& line = it->second;
			line.tick  = 0;
			recent     = recent || m_uses - line.last_use < FlushLifetime;
			if (line.dirty_tick != tick) {
				line.dirty_tick = tick;
				m_dirty.push_back(it->first);
			}
		}
		return recent;
	}

	// A read of [vaddr, vaddr + size) whose GPU-written bytes need copies of tick `needed` or
	// newer. Missing lines are added and stale ones queued for a copy at the end of `current`.
	[[nodiscard]] Lookup Find(uint64_t vaddr, uint64_t size, uint64_t needed, uint64_t current) {
		Lookup result;
		if (size == 0 || size > MaxRead) {
			return result;
		}
		const auto first = vaddr & ~(LineSize - 1);
		const auto last  = (vaddr + size + LineSize - 1) & ~(LineSize - 1);
		result.first     = first;
		result.held      = true;
		for (auto line = first; line < last; line += LineSize) {
			const auto it = m_lines.find(line);
			if (it == m_lines.end()) {
				Add(line, current);
				result.held = false;
				continue;
			}
			auto& entry    = it->second;
			entry.last_use = ++m_uses;
			result.slots[(line - first) / LineSize] = entry.slot;
			if (entry.tick == 0 || entry.tick < needed) {
				result.held = false;
				if (entry.dirty_tick != current) {
					entry.dirty_tick = current;
					m_dirty.push_back(line);
				}
			}
			result.wait = std::max(result.wait, entry.tick);
		}
		if (result.held) {
			m_flush_use = m_uses;
		}
		return result;
	}

	// Mirror reads succeeded recently: operations that may write through device addresses
	// should be submitted at once (their writes clear every line's eligibility).
	[[nodiscard]] bool RecentlyHit() const noexcept {
		return m_flush_use != 0 && m_uses - m_flush_use < FlushLifetime;
	}

	// The lines to copy at the end of command buffer `tick`, sorted and unique, as (line, slot);
	// each gets `tick` as its copy tick unless `accept(line)` rejects it (a line outside one
	// cached buffer stays unmirrored).
	template <typename Accept>
	[[nodiscard]] std::vector<std::pair<uint64_t, uint32_t>> TakeDirty(uint64_t tick, Accept&& accept) {
		std::vector<std::pair<uint64_t, uint32_t>> copies;
		if (m_dirty.empty()) {
			return copies;
		}
		std::sort(m_dirty.begin(), m_dirty.end());
		m_dirty.erase(std::unique(m_dirty.begin(), m_dirty.end()), m_dirty.end());
		copies.reserve(m_dirty.size());
		for (const auto line: m_dirty) {
			const auto it = m_lines.find(line);
			if (it == m_lines.end() || !accept(line)) {
				continue;
			}
			it->second.tick = tick;
			copies.emplace_back(line, it->second.slot);
		}
		m_dirty.clear();
		return copies;
	}
	[[nodiscard]] bool HasDirty() const noexcept { return !m_dirty.empty(); }
	[[nodiscard]] const Line* Get(uint64_t line) const {
		const auto it = m_lines.find(line);
		return it == m_lines.end() ? nullptr : &it->second;
	}

private:
	void Add(uint64_t line, uint64_t current) {
		if (m_free_slots.empty()) {
			if (m_lines.size() < MaxLines) {
				m_free_slots.push_back(static_cast<uint32_t>(m_lines.size()));
			} else {
				// Forget the half used least recently; their queued copies are skipped.
				std::vector<uint64_t> uses;
				uses.reserve(m_lines.size());
				for (const auto& [address, entry]: m_lines) {
					uses.push_back(entry.last_use);
				}
				std::nth_element(uses.begin(), uses.begin() + static_cast<std::ptrdiff_t>(uses.size() / 2), uses.end());
				const auto oldest = uses[uses.size() / 2];
				for (auto it = m_lines.begin(); it != m_lines.end();) {
					if (it->second.last_use <= oldest) {
						m_free_slots.push_back(it->second.slot);
						it = m_lines.erase(it);
					} else {
						++it;
					}
				}
			}
		}
		Line entry;
		entry.slot = m_free_slots.back();
		m_free_slots.pop_back();
		entry.last_use = ++m_uses;
		// Nothing recorded the line's bytes yet: copy them at the end of this command buffer.
		entry.dirty_tick = current;
		m_lines.emplace(line, entry);
		m_dirty.push_back(line);
	}

	std::map<uint64_t, Line> m_lines;
	std::vector<uint64_t>    m_dirty;
	std::vector<uint32_t>    m_free_slots;
	uint64_t                 m_uses      = 0;
	uint64_t                 m_flush_use = 0;
};

} // namespace Libs::Graphics::ReadbackWait

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKWAIT_H_
