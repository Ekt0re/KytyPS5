#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  IR::PushData::Window push_data, const Profile& profile,
                                  bool emit_debug_info) {
	using namespace Emitter;

	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program);
	program.bindings = {};
	EmitterState state(program, input_info, profile);
	state.push_data = push_data;
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	// The console keeps lane 63 when every lane of a wave stores to one LDS word; RADV keeps lane 0
	// (AnyPS5 55fe6ebf).
	state.lds_last_lane_stores =
	    program.stage == ShaderType::Compute && workgroup != nullptr && state.lane_count == 1u &&
	    program.wave_size == 64u && workgroup->host_subgroup_size == 64u && program.info.uses_lds &&
	    state.lds_storage_class == spv::StorageClassWorkgroup;
	// The console runs a compute workgroup on one CU, so its waves share one L0 cache and
	// read each other's buffer stores after S_BARRIER with plain (GLC=0) loads; GTA VI's
	// rig-feature interpreter (CS 0x44d7e1bd85bee6c0) passes its values between layers that
	// way. RADV runs compute workgroups in WGP mode: their waves sit on two CUs with separate
	// L0 caches, and half of them read stale values (BufferCrossWaveBarrierVisibility tests).
	// In programs with a workgroup barrier and more than one host wave, loads and stores of
	// the buffers the program writes are volatile (cache-bypassing); other buffers keep L0.
	if (program.stage == ShaderType::Compute && workgroup != nullptr) {
		const auto host_wave = std::min<uint32_t>(
		    program.wave_size, workgroup->host_subgroup_size != 0u ? workgroup->host_subgroup_size : 64u);
		const auto threads = workgroup->threads_num[0] * std::max(workgroup->threads_num[1], 1u) *
		                     std::max(workgroup->threads_num[2], 1u);
		bool barrier = false;
		for (const auto* block: program.blocks) {
			for (const auto& inst: *block) {
				barrier = barrier || inst.GetOpcode() == IR::ValueOpcode::Barrier;
			}
		}
		state.workgroup_buffer_coherence =
		    barrier && threads > host_wave &&
		    std::ranges::any_of(program.info.buffers, [](const auto& b) { return b.read && b.written; });
	}
	DefineModule(state);
	EmitProgram(state);
	state.builder.AddEntryPoint(ExecutionModelForStage(state.program.stage), state.main_func,
	                            "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
