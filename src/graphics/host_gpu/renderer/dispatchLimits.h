#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHLIMITS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHLIMITS_H_

// Size checks and rate-limited diagnostics for compute dispatches whose group counts the guest
// chose. The counts are full 32-bit values: bits above the low byte are part of the count, not
// flags. A count computed from a floating-point value can be large enough to keep the host GPU
// busy until the driver times out; such a dispatch is named and, with KYTY_DISPATCH_LIMITS=1,
// skipped. Multiplication saturates in all cases and does not change which dispatches run by
// default.
//
// GPU-written indirect arguments are judged on the GPU by DispatchArgsGuard. This header only
// judges counts known on the host (direct dispatches and CPU-written arguments), so the two paths
// cannot both clamp the same dispatch.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace Libs::Graphics {

constexpr uint32_t DispatchDefaultMaxGroups    = 1u << 22u;
constexpr uint64_t DispatchReportInvocations   = 1ull << 26u;
constexpr uint32_t DispatchReportsPerShader    = 4;

// x * y * z, saturating: three full 32-bit counts do not fit in 64 bits.
[[nodiscard]] constexpr uint64_t DispatchGroups(uint32_t x, uint32_t y, uint32_t z) {
	const uint64_t xy = uint64_t {x} * y;
	if (z != 0 && xy > UINT64_MAX / z) return UINT64_MAX;
	return xy * z;
}

// Total invocations, saturating before the multiplication.
[[nodiscard]] constexpr uint64_t DispatchInvocations(uint32_t x, uint32_t y, uint32_t z,
                                                     uint64_t threads_per_group) {
	const uint64_t groups = DispatchGroups(x, y, z);
	if (threads_per_group != 0 && groups > UINT64_MAX / threads_per_group) return UINT64_MAX;
	return groups * threads_per_group;
}

// The reports and skips are off unless KYTY_DISPATCH_LIMITS is exactly "1".
[[nodiscard]] inline bool DispatchLimitsEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_DISPATCH_LIMITS");
		return text != nullptr && text[0] == '1' && text[1] == '\0';
	}();
	return enabled;
}

enum class DispatchVerdict : uint8_t {
	Run,    // normal
	Report, // runs, but large enough to be named in the log
	Skip,   // above the group limit: cannot be valid
};

[[nodiscard]] constexpr DispatchVerdict JudgeDispatch(uint32_t x, uint32_t y, uint32_t z,
                                                      uint64_t threads_per_group,
                                                      uint32_t max_groups, bool may_skip) {
	if (may_skip && DispatchGroups(x, y, z) > max_groups) return DispatchVerdict::Skip;
	if (DispatchInvocations(x, y, z, threads_per_group) >= DispatchReportInvocations) {
		return DispatchVerdict::Report;
	}
	return DispatchVerdict::Run;
}

// Which dword of `words` the x count was computed from: the first i with ceil(words[i] / group)
// == x (x > 1, so a common count of one does not match everything). -1 when none does.
[[nodiscard]] inline int FindCountDword(const uint32_t* words, uint32_t num, uint32_t x,
                                        uint32_t threads_x) {
	if (x <= 1 || threads_x == 0) return -1;
	for (uint32_t i = 0; i < num; i++) {
		const uint64_t groups = (uint64_t {words[i]} + threads_x - 1u) / threads_x;
		if (groups == x) return static_cast<int>(i);
	}
	return -1;
}

[[nodiscard]] inline float DwordAsFloat(uint32_t value) {
	float result = 0.0f;
	std::memcpy(&result, &value, sizeof(result));
	return result;
}

// Counts reports per shader so that one shader dispatching every frame is named only a few times.
class DispatchReportLimiter {
public:
	explicit DispatchReportLimiter(uint32_t per_shader = DispatchReportsPerShader)
	    : m_per_shader(per_shader) {}

	// True for the first `per_shader` calls of each key.
	[[nodiscard]] bool Allow(uint64_t key) {
		const std::lock_guard lock(m_mutex);
		return m_counts[key]++ < m_per_shader;
	}

private:
	uint32_t                               m_per_shader;
	std::mutex                             m_mutex;
	std::unordered_map<uint64_t, uint32_t> m_counts;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DISPATCHLIMITS_H_ */
