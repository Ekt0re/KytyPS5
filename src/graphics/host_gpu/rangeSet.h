#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_

#include "common/assert.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace Detail {

// Disjoint, non-adjacent intervals of addresses, one map node per interval.
class RangeSetMap final {
public:
	bool Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second >= address) {
			it = std::prev(it);
		}
		if (it == m_ranges.end() || it->first > address) {
			it = m_ranges.emplace_hint(it, address, end);
		} else {
			if (it->second >= end) {
				return false;
			}
			it->second = end;
		}
		auto next = std::next(it);
		while (next != m_ranges.end() && next->first <= it->second) {
			it->second = std::max(it->second, next->second);
			next       = m_ranges.erase(next);
		}
		return true;
	}

	bool Subtract(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			it = std::prev(it);
		}
		const bool changed = it != m_ranges.end() && it->first < end;
		while (it != m_ranges.end() && it->first < end) {
			const auto begin = it->first;
			const auto last  = it->second;
			it               = m_ranges.erase(it);
			if (begin < address) {
				m_ranges.emplace(begin, address);
			}
			if (last > end) {
				m_ranges.emplace(end, last);
				break;
			}
		}
		return changed;
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& [begin, end]: m_ranges) {
			func(begin, end);
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.lower_bound(address);
		if (it != m_ranges.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_ranges.end() && it->first < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it == m_ranges.begin()) {
			return false;
		}
		--it;
		return it->first <= address && it->second >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = m_ranges.upper_bound(address);
		if (it != m_ranges.begin()) {
			--it;
		}
		for (; it != m_ranges.end() && it->first < end; ++it) {
			const auto begin = std::max(address, it->first);
			const auto last  = std::min(end, it->second);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }
	void               Clear() noexcept { m_ranges.clear(); }

private:
	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	std::map<uint64_t, uint64_t> m_ranges;
};

// The same set in sorted blocks of at most BlockMax intervals, with each block's first address in
// m_heads: a lookup searches two short arrays, and a change moves the tail of one block, where one
// map node per interval costs a cache miss per step of a lookup and an allocation per insertion
// (the GPU-modified ranges of a game hold tens of thousands). Its answers are RangeSetMap's
// (tests/MemoryTrackerTests.cpp compares the two).
class RangeSetBlocks final {
public:
	// Both return whether the set changed.
	bool Add(uint64_t address, uint64_t size) {
		const auto end = End(address, size);
		// Intervals that overlap or touch [address, end) merge with it.
		auto first = LowerBound(address);
		if (first != Pos {} && At(Prev(first)).end >= address) {
			first = Prev(first);
			if (At(first).begin <= address && At(first).end >= end) {
				return false;
			}
		} else if (first != EndPos() && At(first).begin == address && At(first).end >= end) {
			return false;
		}
		uint64_t begin = address;
		uint64_t last  = end;
		auto     it    = first;
		for (; it != EndPos() && At(it).begin <= last; it = Next(it)) {
			begin = std::min(begin, At(it).begin);
			last  = std::max(last, At(it).end);
		}
		const Interval merged {begin, last};
		Replace(first, it, &merged, 1);
		return true;
	}

	bool Subtract(uint64_t address, uint64_t size) {
		const auto end   = End(address, size);
		auto       first = LowerBound(address);
		if (first != Pos {} && At(Prev(first)).end > address) {
			first = Prev(first);
		}
		auto last = first;
		while (last != EndPos() && At(last).begin < end) {
			last = Next(last);
		}
		if (first == last) {
			return false;
		}
		// What remains of the first and the last interval replaces them all.
		Interval parts[2];
		size_t   count = 0;
		if (At(first).begin < address) {
			parts[count++] = {At(first).begin, address};
		}
		if (const auto tail = At(Prev(last)).end; tail > end) {
			parts[count++] = {end, tail};
		}
		Replace(first, last, parts, count);
		return true;
	}

	void Clear() noexcept {
		m_blocks.clear();
		m_heads.clear();
		m_count = 0;
	}

	template <typename Func>
	void ForEach(Func&& func) const {
		for (const auto& block: m_blocks) {
			for (const auto& range: block) {
				func(range.begin, range.end);
			}
		}
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		const auto it  = LowerBound(address);
		if (it != Pos {} && At(Prev(it)).end > address) {
			return true;
		}
		return it != EndPos() && At(it).begin < end;
	}

	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		const auto end = End(address, size);
		const auto it  = UpperBound(address);
		if (it == Pos {}) {
			return false;
		}
		const auto& range = At(Prev(it));
		return range.begin <= address && range.end >= end;
	}

	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		const auto end = End(address, size);
		auto       it  = UpperBound(address);
		if (it != Pos {}) {
			it = Prev(it);
		}
		for (; it != EndPos() && At(it).begin < end; it = Next(it)) {
			const auto begin = std::max(address, At(it).begin);
			const auto last  = std::min(end, At(it).end);
			if (begin < last) {
				func(begin, last);
			}
		}
	}

	[[nodiscard]] bool Empty() const { return m_count == 0; }

private:
	struct Interval {
		uint64_t begin = 0;
		uint64_t end   = 0;
	};
	using Intervals                 = std::vector<Interval>;
	static constexpr size_t BlockMax = 128;

	// An interval's place: its block and its index there ({block count, 0}: the end; {0, 0}: the
	// first interval, and the end of an empty set).
	struct Pos {
		size_t block = 0;
		size_t index = 0;
		bool   operator==(const Pos& other) const = default;
	};
	[[nodiscard]] Pos             EndPos() const { return {m_blocks.size(), 0}; }
	[[nodiscard]] const Interval& At(Pos pos) const { return m_blocks[pos.block][pos.index]; }
	[[nodiscard]] Pos             Next(Pos pos) const {
        return ++pos.index < m_blocks[pos.block].size() ? pos : Pos {pos.block + 1, 0};
	}
	[[nodiscard]] Pos Prev(Pos pos) const {
		return pos.index != 0 ? Pos {pos.block, pos.index - 1}
		                      : Pos {pos.block - 1, m_blocks[pos.block - 1].size() - 1};
	}
	// The first interval beginning at or after the address (LowerBound), after it (UpperBound):
	// in the last block whose first one begins before (at or before) it, else the next block's
	// first.
	[[nodiscard]] Pos LowerBound(uint64_t address) const {
		const auto block = static_cast<size_t>(
		    std::lower_bound(m_heads.begin(), m_heads.end(), address) - m_heads.begin());
		if (block == 0) {
			return {};
		}
		const auto& ranges = m_blocks[block - 1];
		const auto  index  = static_cast<size_t>(
            std::lower_bound(ranges.begin(), ranges.end(), address,
                             [](const Interval& range, uint64_t value) { return range.begin < value; }) -
            ranges.begin());
		return index < ranges.size() ? Pos {block - 1, index} : Pos {block, 0};
	}
	[[nodiscard]] Pos UpperBound(uint64_t address) const {
		const auto block = static_cast<size_t>(
		    std::upper_bound(m_heads.begin(), m_heads.end(), address) - m_heads.begin());
		if (block == 0) {
			return {};
		}
		const auto& ranges = m_blocks[block - 1];
		const auto  index  = static_cast<size_t>(
            std::upper_bound(ranges.begin(), ranges.end(), address,
                             [](uint64_t value, const Interval& range) { return value < range.begin; }) -
            ranges.begin());
		return index < ranges.size() ? Pos {block - 1, index} : Pos {block, 0};
	}

	// The intervals [first, last) replaced with `count` intervals that keep the order.
	void Replace(Pos first, Pos last, const Interval* items, size_t count) {
		if (m_blocks.empty()) {
			if (count == 0) {
				return;
			}
			m_blocks.emplace_back(items, items + count);
			m_heads.push_back(items[0].begin);
			m_count = count;
			Fix(0);
			return;
		}
		// (The end is the last block's end; a block's start, the previous block's end when it
		// closes a removal.)
		const auto at_end = [&](Pos& pos) {
			if (pos.block == m_blocks.size()) {
				pos = {pos.block - 1, m_blocks[pos.block - 1].size()};
			}
		};
		at_end(first);
		at_end(last);
		if (last.index == 0 && last.block > first.block) {
			last = {last.block - 1, m_blocks[last.block - 1].size()};
		}
		if (first.block == last.block) {
			auto& block = m_blocks[first.block];
			m_count     = m_count - (last.index - first.index) + count;
			if (count == last.index - first.index) {
				std::copy(items, items + count, block.begin() + static_cast<ptrdiff_t>(first.index));
			} else {
				block.erase(block.begin() + static_cast<ptrdiff_t>(first.index),
				            block.begin() + static_cast<ptrdiff_t>(last.index));
				block.insert(block.begin() + static_cast<ptrdiff_t>(first.index), items, items + count);
			}
			Fix(first.block);
			return;
		}
		// The first block's tail, the blocks between and the last block's head go; the items end
		// the first block.
		auto&  head_block = m_blocks[first.block];
		auto&  tail_block = m_blocks[last.block];
		size_t removed    = (head_block.size() - first.index) + last.index;
		for (auto b = first.block + 1; b < last.block; ++b) {
			removed += m_blocks[b].size();
		}
		m_count = m_count - removed + count;
		tail_block.erase(tail_block.begin(), tail_block.begin() + static_cast<ptrdiff_t>(last.index));
		head_block.erase(head_block.begin() + static_cast<ptrdiff_t>(first.index), head_block.end());
		head_block.insert(head_block.end(), items, items + count);
		m_blocks.erase(m_blocks.begin() + static_cast<ptrdiff_t>(first.block + 1),
		               m_blocks.begin() + static_cast<ptrdiff_t>(last.block));
		m_heads.erase(m_heads.begin() + static_cast<ptrdiff_t>(first.block + 1),
		              m_heads.begin() + static_cast<ptrdiff_t>(last.block));
		// (The last block, next to the first now: fixed first, the first block's index holds.)
		Fix(first.block + 1);
		Fix(first.block);
	}

	// A changed block: removed when empty, split when over BlockMax, joined with the next when
	// both are small; its head set again.
	void Fix(size_t b) {
		if (m_blocks[b].empty()) {
			m_blocks.erase(m_blocks.begin() + static_cast<ptrdiff_t>(b));
			m_heads.erase(m_heads.begin() + static_cast<ptrdiff_t>(b));
			return;
		}
		if (m_blocks[b].size() > BlockMax) {
			std::vector<Intervals> pieces;
			auto&                  block = m_blocks[b];
			for (size_t at = BlockMax / 2; at < block.size(); at += BlockMax / 2) {
				pieces.emplace_back(
				    block.begin() + static_cast<ptrdiff_t>(at),
				    block.begin() + static_cast<ptrdiff_t>(std::min(block.size(), at + BlockMax / 2)));
			}
			block.resize(BlockMax / 2);
			for (size_t i = 0; i < pieces.size(); ++i) {
				m_heads.insert(m_heads.begin() + static_cast<ptrdiff_t>(b + 1 + i),
				               pieces[i].front().begin);
				m_blocks.insert(m_blocks.begin() + static_cast<ptrdiff_t>(b + 1 + i),
				                std::move(pieces[i]));
			}
		} else if (b + 1 < m_blocks.size() &&
		           m_blocks[b].size() + m_blocks[b + 1].size() <= BlockMax / 2) {
			auto& next = m_blocks[b + 1];
			m_blocks[b].insert(m_blocks[b].end(), next.begin(), next.end());
			m_blocks.erase(m_blocks.begin() + static_cast<ptrdiff_t>(b + 1));
			m_heads.erase(m_heads.begin() + static_cast<ptrdiff_t>(b + 1));
		}
		m_heads[b] = m_blocks[b].front().begin;
	}

	static uint64_t End(uint64_t address, uint64_t size) {
		if (size == 0 || size > UINT64_MAX - address) {
			EXIT("invalid range-set address or size\n");
		}
		return address + size;
	}

	std::vector<Intervals> m_blocks;
	std::vector<uint64_t>  m_heads;
	size_t                 m_count = 0;
};

} // namespace Detail

// Disjoint, non-adjacent intervals of addresses. Add and Subtract return whether the set changed.
// KYTY_RANGESET_BLOCKS=0 keeps every set in a map (one node per interval) instead of sorted blocks;
// both give the same answers, the blocks with fewer cache misses and allocations.
class RangeSet final {
public:
	[[nodiscard]] static bool BlocksEnabled() {
		static const bool enabled = [] {
			const char* value = std::getenv("KYTY_RANGESET_BLOCKS");
			return value == nullptr || value[0] != '0';
		}();
		return enabled;
	}

	bool Add(uint64_t address, uint64_t size) {
		return m_use_blocks ? m_blocks.Add(address, size) : m_map.Add(address, size);
	}
	bool Subtract(uint64_t address, uint64_t size) {
		return m_use_blocks ? m_blocks.Subtract(address, size) : m_map.Subtract(address, size);
	}
	template <typename Func>
	void ForEach(Func&& func) const {
		if (m_use_blocks) {
			m_blocks.ForEach(func);
		} else {
			m_map.ForEach(func);
		}
	}
	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		return m_use_blocks ? m_blocks.Intersects(address, size) : m_map.Intersects(address, size);
	}
	[[nodiscard]] bool Contains(uint64_t address, uint64_t size) const {
		return m_use_blocks ? m_blocks.Contains(address, size) : m_map.Contains(address, size);
	}
	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		if (m_use_blocks) {
			m_blocks.ForEachInRange(address, size, func);
		} else {
			m_map.ForEachInRange(address, size, func);
		}
	}
	[[nodiscard]] bool Empty() const { return m_use_blocks ? m_blocks.Empty() : m_map.Empty(); }
	void               Clear() noexcept {
		m_blocks.Clear();
		m_map.Clear();
	}

private:
	bool                    m_use_blocks = BlocksEnabled();
	Detail::RangeSetBlocks  m_blocks;
	Detail::RangeSetMap     m_map;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RANGESET_H_
