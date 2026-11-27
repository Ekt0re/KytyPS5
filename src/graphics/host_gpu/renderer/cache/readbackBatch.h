#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKBATCH_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKBATCH_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

// Planning of batched buffer readbacks (KYTY_BATCH_READBACK=1 or toggle batch_readback): which
// parts of which buffers one synchronous download copies, so that one CPU access costs one
// submission and one wait however many buffers its window touches. Pure logic, no Vulkan.
namespace Libs::Graphics::ReadbackBatch {

struct Range {
	uint64_t begin = 0;
	uint64_t end   = 0;

	[[nodiscard]] bool     Empty() const noexcept { return end <= begin; }
	[[nodiscard]] uint64_t Size() const noexcept { return Empty() ? 0 : end - begin; }
	[[nodiscard]] Range    Clamp(Range other) const noexcept {
        return {std::max(begin, other.begin), std::min(end, other.end)};
	}
	bool operator==(const Range&) const = default;
};

// The window an access [vaddr, vaddr + size) downloads: the `window`-aligned block around its
// start, extended to the end of the access, widened by `prefetch` bytes on both sides, and
// clamped to `limit` (the guest address range the batch may touch). `window` is a power of two.
[[nodiscard]] inline Range AccessWindow(uint64_t vaddr, uint64_t size, uint64_t window,
                                        uint64_t prefetch, Range limit) noexcept {
	const uint64_t aligned = vaddr & ~(window - 1);
	const uint64_t begin   = aligned >= prefetch ? aligned - prefetch : 0;
	const uint64_t block   = std::max(aligned + window, vaddr + size);
	const uint64_t end     = block <= UINT64_MAX - prefetch ? block + prefetch : UINT64_MAX;
	return limit.Clamp({begin, end});
}

// One buffer's part of a batch window.
template <typename Id>
struct Part {
	Id    id;
	Range range;
};

// Splits `window` by the buffers that own it. `owner(address)` returns the id and guest range of
// the buffer holding `address`, or nothing; ownership is looked up once per `page` bytes and a
// buffer is visited once. The buffer `first` (the one the access is for) comes first with the
// part the caller chose for it (`first_part`); the others get `window` clamped to their range.
template <typename Id, typename Owner>
[[nodiscard]] std::vector<Part<Id>> SplitByOwner(Range window, uint64_t page, Id first,
                                                 Range first_part, Owner&& owner) {
	std::vector<Part<Id>> parts;
	if (!first_part.Empty()) {
		parts.push_back({first, first_part});
	}
	uint64_t address = window.begin & ~(page - 1);
	while (address < window.end) {
		const std::optional<std::pair<Id, Range>> found = owner(address);
		uint64_t next = address + page;
		if (found) {
			const auto& [id, range] = *found;
			const auto part         = window.Clamp(range);
			const bool  seen = std::any_of(parts.begin(), parts.end(),
			                               [&](const Part<Id>& p) { return p.id == id; });
			if (!seen && !part.Empty()) {
				parts.push_back({id, part});
			}
			// Skip the rest of the buffer, page aligned.
			if (range.end > next) {
				next = (range.end + page - 1) & ~(page - 1);
			}
		}
		if (next <= address) {
			break;
		}
		address = next;
	}
	return parts;
}

// Statistics: whether a download's window is next to (or the same as) one of the last few
// windows downloaded, i.e. whether copying the neighbours along with the earlier download would
// have saved this one.
class NeighbourHistory {
public:
	bool Note(uint64_t window_begin, uint64_t window) noexcept {
		bool hit = false;
		for (const auto begin: m_recent) {
			if (begin != Invalid &&
			    (begin + window == window_begin || window_begin + window == begin)) {
				hit = true;
			}
		}
		m_recent[m_next] = window_begin;
		m_next           = (m_next + 1) % m_recent.size();
		return hit;
	}

private:
	static constexpr uint64_t Invalid = UINT64_MAX;
	std::array<uint64_t, 8>   m_recent {Invalid, Invalid, Invalid, Invalid,
	                                    Invalid, Invalid, Invalid, Invalid};
	size_t                    m_next = 0;
};

} // namespace Libs::Graphics::ReadbackBatch

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKBATCH_H_
