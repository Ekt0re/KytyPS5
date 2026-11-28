#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// How the host's GLSL.std.450 NMin/NMax compare to the guest's V_MIN/V_MAX rule: ordered
// sources and signed zero always agree; the levels differ on NaN sources.
enum class FloatMinMax : uint8_t {
	Cascade,        // Results differ; lower with compares and selects.
	OtherNanSelect, // The other source for one NaN, a canonical NaN for two: fix the pair.
	Exact,          // The other source for one NaN, the left source for two.
};

// Host device capabilities and user options that change the emitted SPIR-V. Built once per device.
struct Profile {
	bool maintenance8         = false; // Dynamic texel offsets on image samples.
	// Host NMin/NMax behavior against the guest min/max rule, from a probe of the device.
	FloatMinMax float_min_max = FloatMinMax::Cascade;
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

// A guest loop that takes its bound or its next node from memory never ends when that memory
// holds something else (a pass that was left out, a buffer not written yet), and one endless
// invocation hangs the host GPU until the driver resets it. Each invocation therefore leaves the
// shader once its subgroup has run this many loop iterations, all its loops together (structured
// loops only). KYTY_SHADER_LOOP_LIMIT=<n> changes the limit and 0 removes the guard; without it
// the default applies (DefaultLoopIterationLimit; tests change it with
// SetLoopIterationLimitDefault). The limit is part of the emitted code but not of Profile: the
// shader store key changes while it is away from DefaultLoopIterationLimit.
// Off by default (0): with 65,536 (PR #14) GTA VI livelocked at the start of 3D on the world and
// intro saves (a guest thread read-faulted 1-3.6 M times per 10 s on a page no buffer download
// resolves, 0 draws). Its shaders run loops past the cap, so a capped invocation ends early and the
// game waits forever for a result that never comes. KYTY_SHADER_LOOP_LIMIT=65536 turns the guard on
// for debugging GPU hangs.
inline constexpr uint32_t DefaultLoopIterationLimit = 0u;
void                      SetLoopIterationLimitDefault(uint32_t limit);
uint32_t                  LoopIterationLimit();

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
