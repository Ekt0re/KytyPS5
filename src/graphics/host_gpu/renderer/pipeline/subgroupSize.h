#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SUBGROUPSIZE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SUBGROUPSIZE_H_

// Per-stage resolution of the host subgroup size for a compiled shader. Upstream already attaches
// requiredSubgroupSize to fragment and compute stages; KYTY_SUBGROUP_STAGE_SIZE=1 extends that to
// every graphics stage the device supports and emits a one-time warning when a shader that uses
// subgroup operations cannot get the size its guest wave expects. With the switch off this header
// is not used by pipeline creation.

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics {

struct SubgroupSizeCaps {
	uint32_t default_size     = 0;
	uint32_t min_size         = 0;
	uint32_t max_size         = 0;
	uint32_t supported_stages = 0; // VkShaderStageFlags bitmask.
};

struct SubgroupSizeDecision {
	uint32_t expected_size    = 0;
	uint32_t required_size    = 0;
	bool     apply            = false;
	bool     mismatch_warning = false;
};

// Off unless KYTY_SUBGROUP_STAGE_SIZE is exactly "1"; the default keeps upstream's two-stage
// behaviour and no warnings.
[[nodiscard]] inline bool SubgroupStageSizeEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_SUBGROUP_STAGE_SIZE");
		return text != nullptr && text[0] == '1' && text[1] == '\0';
	}();
	return enabled;
}

// Resolves the host subgroup contract of one stage. When the driver supports requiredSubgroupSize
// for `stage_bit` and the expected size is in [min_size, max_size], the caller attaches
// VkPipelineShaderStageRequiredSubgroupSizeCreateInfo with `required_size`. Otherwise a shader
// with subgroup operations that cannot get its expected size asks for a one-time warning.
[[nodiscard]] constexpr SubgroupSizeDecision ResolveStageSubgroupSize(
    uint32_t stage_bit, uint32_t guest_wave_size, bool emulated_wave64_on_32, bool uses_subgroup_ops,
    const SubgroupSizeCaps& caps) noexcept {
	SubgroupSizeDecision decision {};
	if (guest_wave_size != 32u && guest_wave_size != 64u) {
		return decision;
	}
	const uint32_t expected = emulated_wave64_on_32 ? 32u : guest_wave_size;
	decision.expected_size  = expected;

	const bool stage_supported = (caps.supported_stages & stage_bit) != 0u;
	const bool size_supported =
	    caps.min_size != 0u && expected >= caps.min_size && expected <= caps.max_size;
	if (stage_supported && size_supported) {
		decision.required_size = expected;
		decision.apply         = true;
		return decision;
	}

	if (uses_subgroup_ops && caps.default_size != 0u && expected != caps.default_size) {
		decision.mismatch_warning = true;
	}
	return decision;
}

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SUBGROUPSIZE_H_ */
