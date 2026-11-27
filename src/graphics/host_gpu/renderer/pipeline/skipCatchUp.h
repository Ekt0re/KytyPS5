#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPCATCHUP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPCATCHUP_H_

#include "graphics/host_gpu/renderer/pipeline/skipTaint.h"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

// Catch-up queue for compute dispatches skipped because translation failed or because they read a
// tainted producer buffer. When the skipped work becomes possible (its program translates), the
// queue yields it in submission order so the renderer re-runs it before later consumers use its
// output, clearing its taint ranges instead of leaving a permanent hole. Pure C++ with no Vulkan
// dependency, so the bookkeeping is unit tested directly.
namespace Libs::Graphics::SkipCatchUp {

enum class ProbeState : uint8_t {
	Pending, // the program is not available to replay yet
	Ready,   // the program is ready to replay now
	Failed,  // the program failed permanently; drop this entry
};

struct Stats {
	uint64_t enqueued         = 0;
	uint64_t replayed         = 0;
	uint64_t superseded       = 0;
	uint64_t dropped_failed   = 0;
	uint64_t dropped_expired  = 0;
	uint64_t dropped_overflow = 0;
};

template <typename Payload>
class Queue {
public:
	static constexpr size_t   MaxPending   = 64;
	static constexpr uint64_t DefaultMaxMs = 30'000;

	struct Entry {
		uint64_t                    key           = 0;
		uint64_t                    shader_addr   = 0;
		uint64_t                    skipped_at_ms = 0;
		std::vector<SkipTaint::Use> uses;
		Payload                     payload {};
	};

	explicit Queue(uint64_t max_age_ms = DefaultMaxMs): m_max_age_ms(max_age_ms) {}

	[[nodiscard]] bool   Empty() const noexcept { return m_entries.empty(); }
	[[nodiscard]] size_t Size() const noexcept { return m_entries.size(); }
	[[nodiscard]] const Stats& GetStats() const noexcept { return m_stats; }

	// Records a skipped dispatch. If an identical key is already queued, refreshes it with the
	// latest payload and timestamp instead of duplicating it.
	void Enqueue(uint64_t key, uint64_t shader_addr, std::vector<SkipTaint::Use> uses,
	             Payload payload, uint64_t now_ms) {
		for (auto& existing: m_entries) {
			if (existing.key == key) {
				existing.shader_addr   = shader_addr;
				existing.skipped_at_ms = now_ms;
				existing.uses          = std::move(uses);
				existing.payload       = std::move(payload);
				return;
			}
		}
		if (m_entries.size() >= MaxPending) {
			m_entries.erase(m_entries.begin());
			m_stats.dropped_overflow++;
		}
		m_entries.push_back(
		    Entry {key, shader_addr, now_ms, std::move(uses), std::move(payload)});
		m_stats.enqueued++;
	}

	// Drops a queued skipped dispatch when a fresh instance of the exact same key ran normally.
	void DropByKey(uint64_t key) {
		const auto removed =
		    std::erase_if(m_entries, [key](const Entry& e) { return e.key == key; });
		m_stats.superseded += removed;
	}

	// Drops queued entries whose known written ranges are all covered by `written_uses` from a
	// newer dispatch that actually executed, so an older skipped copy cannot overwrite fresh data.
	void DropSuperseded(const std::vector<SkipTaint::Use>& written_uses) {
		if (m_entries.empty() || written_uses.empty()) {
			return;
		}
		const auto covered = [&](uint64_t addr, uint64_t size) {
			const uint64_t end = addr + size;
			for (const auto& w: written_uses) {
				if (!w.written || w.size == 0) continue;
				if (w.address <= addr && w.address + w.size >= end) {
					return true;
				}
			}
			return false;
		};
		const auto removed = std::erase_if(m_entries, [&](const Entry& e) {
			bool has_write = false;
			for (const auto& u: e.uses) {
				if (!u.written || u.size == 0) continue;
				has_write = true;
				if (!covered(u.address, u.size)) {
					return false;
				}
			}
			return has_write;
		});
		m_stats.superseded += removed;
	}

	// Probes pending entries in FIFO submission order:
	// - Ready entries are extracted and returned in order to be replayed, unless an earlier entry
	//   that stays queued touches the same memory with a write on either side: replaying the later
	//   one first would read or overwrite data the earlier one has yet to produce, so it waits
	//   behind it (and blocks what depends on it in turn).
	// - Failed or expired entries are dropped (they block nothing).
	// - Pending entries stay queued.
	template <typename Probe>
	[[nodiscard]] std::vector<Entry> TakeReady(uint64_t now_ms, Probe&& probe) {
		std::vector<Entry> ready;
		std::vector<Entry> kept;
		kept.reserve(m_entries.size());
		for (auto& entry: m_entries) {
			const auto state = probe(entry);
			if (state == ProbeState::Failed) {
				m_stats.dropped_failed++;
				continue;
			}
			if (state != ProbeState::Ready && m_max_age_ms != 0 && now_ms >= entry.skipped_at_ms &&
			    now_ms - entry.skipped_at_ms > m_max_age_ms) {
				m_stats.dropped_expired++;
				continue;
			}
			if (state == ProbeState::Ready && !ConflictsWithAny(entry, kept)) {
				ready.push_back(std::move(entry));
				m_stats.replayed++;
				continue;
			}
			kept.push_back(std::move(entry));
		}
		m_entries = std::move(kept);
		return ready;
	}

private:
	// True when `entry` overlaps an earlier queued entry in memory and at least one side writes it.
	static bool ConflictsWithAny(const Entry& entry, const std::vector<Entry>& earlier) {
		for (const auto& other: earlier) {
			for (const auto& a: entry.uses) {
				if (a.size == 0) continue;
				for (const auto& b: other.uses) {
					if (b.size == 0 || (!a.written && !b.written)) continue;
					if (a.address < b.address + b.size && b.address < a.address + a.size) {
						return true;
					}
				}
			}
		}
		return false;
	}

	uint64_t           m_max_age_ms;
	std::vector<Entry> m_entries;
	Stats              m_stats;
};

} // namespace Libs::Graphics::SkipCatchUp

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SKIPCATCHUP_H_
