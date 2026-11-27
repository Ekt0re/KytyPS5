#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPTAINT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPTAINT_H_

// Guest memory a skipped dispatch was going to write (KYTY_SKIP_TAINT=1). A skipped dispatch
// leaves its output buffers stale; a later shader that reads such a buffer can work on stale data
// or loop on it. The ranges the skipped dispatch would have written are remembered as tainted, a
// later dispatch that reads one is skipped as well (its own outputs become tainted: the taint
// follows the data), and a dispatch that really runs and writes a range clears it. Ranges expire
// after a bounded time so a producer that never runs again cannot silence its consumers for ever.
//
// Only buffers named through descriptors are tracked; memory reached through computed pointers is
// not (the loop guard covers that).

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Internal experiment switch, off unless the environment asks for exactly "1".
[[nodiscard]] inline bool SkipTaintEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_SKIP_TAINT");
		return env != nullptr && env[0] == '1';
	}();
	return enabled;
}

class SkipTaint {
public:
	struct Use {
		uint64_t address = 0;
		uint64_t size    = 0;
		bool     read    = false;
		bool     written = false;
	};

	static constexpr uint64_t DefaultMaxAgeMs = 5000;

	explicit SkipTaint(uint64_t max_age_ms = DefaultMaxAgeMs): m_max_age_ms(max_age_ms) {}

	[[nodiscard]] bool Empty() const { return m_ranges.empty(); }
	[[nodiscard]] size_t Size() const { return m_ranges.size(); }

	// The written ranges of `uses` become tainted (the dispatch was skipped).
	void Taint(const std::vector<Use>& uses, uint64_t now_ms) {
		for (const auto& use: uses) {
			if (use.written && use.size != 0) {
				// The set stays disjoint: a range skipped again replaces its older record instead of
				// piling up beside it (a pile made every later Subtract copy and split all of it).
				Subtract(use.address, use.address + use.size);
				m_ranges[use.address] = Range {use.address + use.size, now_ms};
			}
		}
	}

	// True when a range `uses` reads is tainted and not yet expired.
	[[nodiscard]] bool ReadsTainted(const std::vector<Use>& uses, uint64_t now_ms) {
		Expire(now_ms);
		for (const auto& use: uses) {
			if (!use.read || use.size == 0) continue;
			if (FirstOverlap(use.address, use.address + use.size) != m_ranges.end()) return true;
		}
		return false;
	}

	// The dispatch ran: what it wrote is fresh.
	void ClearWritten(const std::vector<Use>& uses) {
		if (m_ranges.empty()) return;
		for (const auto& use: uses) {
			if (use.written && use.size != 0) Subtract(use.address, use.address + use.size);
		}
	}

private:
	struct Range {
		uint64_t end   = 0;
		uint64_t at_ms = 0;
	};
	using Ranges = std::map<uint64_t, Range>; // begin -> end and time; disjoint

	// The first stored range that overlaps [begin, end), or end().
	[[nodiscard]] Ranges::iterator FirstOverlap(uint64_t begin, uint64_t end) {
		auto it = m_ranges.upper_bound(begin);
		if (it != m_ranges.begin()) {
			auto prev = std::prev(it);
			if (prev->second.end > begin) return prev;
		}
		return it != m_ranges.end() && it->first < end ? it : m_ranges.end();
	}

	void Expire(uint64_t now_ms) {
		std::erase_if(m_ranges, [&](const auto& r) {
			return now_ms >= r.second.at_ms && now_ms - r.second.at_ms > m_max_age_ms;
		});
	}

	void Subtract(uint64_t begin, uint64_t end) {
		auto it = FirstOverlap(begin, end);
		while (it != m_ranges.end() && it->first < end) {
			const auto r_begin = it->first;
			const auto r       = it->second;
			it                 = m_ranges.erase(it);
			if (r_begin < begin) m_ranges[r_begin] = Range {begin, r.at_ms};
			if (end < r.end) {
				m_ranges[end] = Range {r.end, r.at_ms};
				break;
			}
		}
	}

	uint64_t m_max_age_ms;
	Ranges   m_ranges;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPTAINT_H_
