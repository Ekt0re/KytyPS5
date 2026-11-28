#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/frontend/decode/ShaderDecoder.h"
#include "graphics/shader/shader.h"

#include <array>
#include <bitset>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

struct CompileOptions {
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        emit_debug_info            = false;
	const Spirv::Profile*       profile                    = &Spirv::DefaultProfile;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	// Instruction-region sizes in dwords, excluding each program's literal buffer; 0 = whole span.
	uint32_t                    instruction_words      = 0;
	uint32_t                    back_instruction_words = 0;
	ShaderStageInputInfo        input_info;
};

struct TranslateResult {
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

// Decoded source and typed scalar call queries are retained once per cache source.
struct ShaderSource {
	struct Call {
		uint32_t instruction;
		std::array<IR::Value, 2> target;
		std::bitset<128> preserved_registers;
		uint64_t address = 0;
	};
	struct Linked {
		struct Function { uint32_t offset; uint32_t size; };
		std::vector<uint32_t> code;
		Decoder::Program decoded;
		std::vector<Function> functions;
		std::vector<uint32_t> observed_function;
		// Unique per link of this source; `revision` is the id of the current one.
		uint64_t id = 0;
	};
	// Links of earlier callee sets, most recent first. A call shader that alternates between
	// callee sets (one per object) swaps a matching one back in instead of linking again, and
	// the pipeline cache keeps its resource plan by link id.
	static constexpr size_t MaxPreviousLinks = 16;
	std::vector<Linked> previous_links;
	std::vector<uint32_t> code;
	Decoder::Program decoded;
	IR::ResourcePlan call_targets;
	std::vector<Call> calls;
	uint32_t unsupported_call_pc = UINT32_MAX;
	uint32_t wave_size = 64;
	std::optional<Linked> linked;
	uint64_t revision = 0;
	uint64_t links_made = 0;
	std::vector<std::pair<uint64_t, uint64_t>> reads;
};

// The compute programs with a guest BVH walk compiled with ray queries, by distinct shader hash,
// for the "[RT]" summaries of RayTracing: those with a walk the recognizer drives with a ray query,
// and those with a walk it leaves to the guest traversal (a program can be both), with the last
// hash of the latter.
enum class RayTracingProgramKind : uint32_t { Traced, Unrecognized };
struct RayTracingPrograms {
	uint64_t traced            = 0;
	uint64_t unrecognized      = 0;
	uint64_t last_unrecognized = 0;
};
void                             NoteRayTracingProgram(RayTracingProgramKind kind, uint64_t shader_hash);
[[nodiscard]] RayTracingPrograms GetRayTracingPrograms();

[[nodiscard]] const char* StageName(ShaderType stage);

// Optional observer for IR pass durations (benchmarks); null by default, no effect on output.
using PassTimingHook = void (*)(const char* pass_name, uint64_t nanoseconds);
void SetPassTimingHook(PassTimingHook hook);
[[nodiscard]] ShaderSource PrepareShaderSource(std::span<const uint32_t> code,
	                                           const CompileOptions& options);
const Decoder::Program& RefreshShaderSource(ShaderSource& source, const IR::SrtRuntime& runtime);
[[nodiscard]] TranslateResult TranslateProgram(const Decoder::Program& decoded,
	                                           const CompileOptions& options);

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       IR::PushData::Window push_data = {});

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
