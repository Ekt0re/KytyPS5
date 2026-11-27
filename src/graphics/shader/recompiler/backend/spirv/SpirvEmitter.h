#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// Host device capabilities and user options that change the emitted SPIR-V. Built once per device.
struct Profile {
	bool maintenance8         = false; // Dynamic texel offsets on image samples.
	bool native_float_min_max = false; // NMin/NMax give the guest's NaN and signed-zero results.
	// OpReadClockKHR Device tick in picoseconds; 0 without shaderDeviceClock.
	uint32_t device_clock_period_ps = 0;
	// Storage buffer elements past the descriptor range, to the byte, read zero and drop writes.
	bool robust_buffer_access = false;
	// 8 and 8_8 sRGB textures sample through sRGB views; without one the shader decodes them.
	bool r8_srgb_views   = false;
	bool r8g8_srgb_views = false;
	// Native float ops: NaN, signed-zero and denormal results may differ from the guest.
	bool relaxed_float = false;
	// Ray queries: recognized traversal loops trace the dispatch's host TLAS.
	bool ray_query = false;
	// Compute shaders that take implicit derivatives (ImageQueryLod) need a derivative group:
	// Quads for even 2D local sizes, Linear for other groups whose size is a multiple of four.
	bool compute_derivative_quads  = true;
	bool compute_derivative_linear = true;
	// A wave64 compute program run as two halves on a 32-lane host subgroup orders its LDS
	// accesses with a subgroup barrier (KYTY_LDS_SYNC=0 turns it off).
	bool lds_lane_sync = true;
	// Length of the resident image heap array; the host binds that many sampled images.
	uint32_t image_heap_capacity = IR::ImageHeapCapacity;
};

inline constexpr Profile DefaultProfile {};

// Scalar-branch guards as ballots compared with zero instead of OpGroupNonUniformAny/All votes.
// NVIDIA's compiler spends ~45 ms on each vote and little on a ballot (KytyPS5 #2469); RADV/ACO
// compiles both alike, so the device picks the default (SetBallotGuardsDefault, on for NVIDIA's
// proprietary driver) and KYTY_BALLOT_GUARDS=0/1 overrides it. Not part of Profile, so a store
// written without ballots keeps its header; the shader store header changes its stamp instead
// while ballots are on.
void SetBallotGuardsDefault(bool enabled);
bool BallotGuardsEnabled();

// KYTY_RT_OFF=1: BVH node tests always miss (an empty scene). Not part of Profile, so the shader
// store header stays; the pipeline cache neither loads nor stores BVH modules while it is set.
bool BvhIntersectAlwaysMisses();

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  IR::PushData::Window push_data       = {},
                                  const Profile&       profile         = DefaultProfile,
                                  bool                 emit_debug_info = false);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
