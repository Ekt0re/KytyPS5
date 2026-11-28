#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CREATESTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CREATESTATS_H_

// Wall time of vkCreate{Compute,Graphics}Pipelines, whoever calls it (store warm-up, workers or the
// render thread): count, total/max time, how many calls the driver reported as cache hits, a
// histogram, and how many calls were inside the driver at once. A warm driver cache shows calls
// under 10 ms with valid cache-hit feedback; a compile shows seconds each. Header-only with a pure
// Line(const Totals&) seam so the arithmetic can be unit-tested without a device.

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace Libs::Graphics::CreateStats {

inline constexpr size_t BucketCount = 5; // <10 ms, <100 ms, <1 s, <10 s, more

[[nodiscard]] constexpr size_t Bucket(uint64_t wall_us) {
	return wall_us < 10'000u ? 0u : wall_us < 100'000u ? 1u : wall_us < 1'000'000u ? 2u
	       : wall_us < 10'000'000u ? 3u : 4u;
}

// KYTY_PIPELINE_CREATE_STATS=1 turns the line on. Read once, off for unset, empty or "0".
[[nodiscard]] inline bool Enabled() noexcept {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_PIPELINE_CREATE_STATS");
		return env != nullptr && env[0] != '\0' && env[0] != '0';
	}();
	return enabled;
}

struct Kind {
	std::atomic<uint64_t>                      count {0};
	std::atomic<uint64_t>                      wall_us {0};
	std::atomic<uint64_t>                      driver_ns {0}; // The driver's own figure, when given.
	std::atomic<uint64_t>                      max_us {0};
	std::atomic<uint64_t>                      feedback_valid {0};
	std::atomic<uint64_t>                      cache_hits {0}; // Of those with valid feedback.
	std::array<std::atomic<uint64_t>, BucketCount> buckets {};
};

struct Totals {
	Kind                  compute;
	Kind                  graphics;
	std::atomic<uint32_t> in_driver {0};
	std::atomic<uint32_t> peak_in_driver {0};
};

inline Totals& Global() {
	static Totals totals;
	return totals;
}

// Around one create call: counts it as inside the driver.
inline void Enter(Totals& totals) {
	const auto now = totals.in_driver.fetch_add(1, std::memory_order_relaxed) + 1;
	auto       peak = totals.peak_in_driver.load(std::memory_order_relaxed);
	while (now > peak &&
	       !totals.peak_in_driver.compare_exchange_weak(peak, now, std::memory_order_relaxed)) {
	}
}

inline void Leave(Totals& totals, Kind& kind, uint64_t wall_us, bool feedback_valid,
                  bool cache_hit, uint64_t driver_ns) {
	totals.in_driver.fetch_sub(1, std::memory_order_relaxed);
	kind.count.fetch_add(1, std::memory_order_relaxed);
	kind.wall_us.fetch_add(wall_us, std::memory_order_relaxed);
	kind.driver_ns.fetch_add(driver_ns, std::memory_order_relaxed);
	kind.buckets[Bucket(wall_us)].fetch_add(1, std::memory_order_relaxed);
	if (feedback_valid) {
		kind.feedback_valid.fetch_add(1, std::memory_order_relaxed);
		kind.cache_hits.fetch_add(cache_hit ? 1u : 0u, std::memory_order_relaxed);
	}
	auto max = kind.max_us.load(std::memory_order_relaxed);
	while (wall_us > max && !kind.max_us.compare_exchange_weak(max, wall_us,
	                                                          std::memory_order_relaxed)) {
	}
}

[[nodiscard]] inline std::string KindText(const char* name, const Kind& kind) {
	char text[384];
	std::snprintf(text, sizeof(text),
	              "%s n=%" PRIu64 " wall=%" PRIu64 "ms driver=%" PRIu64 "ms max=%" PRIu64
	              "ms cache-hit=%" PRIu64 "/%" PRIu64 " <10ms=%" PRIu64 " <100ms=%" PRIu64
	              " <1s=%" PRIu64 " <10s=%" PRIu64 " >=10s=%" PRIu64,
	              name, kind.count.load(), kind.wall_us.load() / 1000u,
	              kind.driver_ns.load() / 1'000'000u, kind.max_us.load() / 1000u,
	              kind.cache_hits.load(), kind.feedback_valid.load(), kind.buckets[0].load(),
	              kind.buckets[1].load(), kind.buckets[2].load(), kind.buckets[3].load(),
	              kind.buckets[4].load());
	return text;
}

[[nodiscard]] inline std::string Line(const Totals& totals = Global()) {
	return "[pipeline-create] " + KindText("cs", totals.compute) + "; " +
	       KindText("gfx", totals.graphics) + "; in-driver now=" +
	       std::to_string(totals.in_driver.load()) + " peak=" +
	       std::to_string(totals.peak_in_driver.load());
}

} // namespace Libs::Graphics::CreateStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_CREATESTATS_H_
