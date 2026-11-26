#include "common/compilePause.h"
#include "common/shaderCompileStatus.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <csetjmp>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <ranges>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Common {
extern thread_local std::jmp_buf* g_dbg_recover;
}

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {1};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// PA_SU_SC_MODE_CNTL.POLY_MODE disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

void DumpShader(std::span<const uint32_t> words, const std::string& name) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	const auto path = Config::GetShaderLogFolder() / name;
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(words.data(), words.size_bytes());
}

void ValidateShaderSpirv(const char* stage, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF("%s\n", text.c_str());
	EXIT("ShaderRecompiler %s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s", stage,
	     shader_hash, messages.c_str());
}

} // namespace

// KYTY_LDS_SYNC=0 drops the LDS ordering of wave64 programs run as halves (Profile::lds_lane_sync).
static bool LdsLaneSyncEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_LDS_SYNC");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics) {
	// AMD's own drivers run NMin/NMax as v_min/v_max, which give the guest's NaN and signed-zero
	// results; Mesa's RADV implements the GLSL.std.450 semantics instead, so it keeps the ordered
	// compare. GLSL.std.450 alone does not order -0 below +0.
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = graphics.physical_device_properties.vendorID == 0x1002u &&
	                                  graphics.driver_id != vk::DriverId::eMesaRadv,
	        .device_clock_period_ps = graphics.device_clock_period_ps,
	        .robust_buffer_access   = graphics.robust_buffer_access_enabled,
	        .r8_srgb_views          = graphics.r8_srgb_views,
	        .r8g8_srgb_views        = graphics.r8g8_srgb_views,
	        .relaxed_float          = Config::RelaxedShaderFloatEnabled(),
	        .ray_query              = graphics.ray_query_enabled,
	        .compute_derivative_quads  = graphics.compute_derivative_quads_enabled,
	        .compute_derivative_linear = graphics.compute_derivative_linear_enabled,
	        .lds_lane_sync             = LdsLaneSyncEnabled(),
	        .image_heap_capacity       = ImageHeapCapacity(graphics)};
}

std::filesystem::path ShaderStorePath() {
	if (KYTY_BUILD != KYTY_BUILD_RELEASE || !Config::ShaderCacheEnabled() ||
	    Config::GraphicsDebugDumpEnabled()) {
		return {};
	}
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return {};
	}
	return std::filesystem::path("_PipelineCache") / (title_id + ".shaders");
}

bool GraphicsPipelineKey::operator==(const GraphicsPipelineKey& other) const {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	return XXH3_64bits(&key, sizeof(key));
}

std::span<const uint32_t> PipelineCache::Program::FunctionCode() const {
	return call_source
	    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
	    : std::span<const uint32_t>{};
}

bool PipelineCache::Program::RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
	const auto revision = call_source->revision;
	ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
	resource_plan.source_reads = call_source->reads;
	return revision != call_source->revision;
}

template <bool HasCalls>
ShaderModule* PipelineCache::Program::FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
                                                 ShaderRecompiler::IR::PushData::Window push_data) {
	EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(resource_plan, runtime, resources,
	                                                    specialization));
	for (auto& candidate: modules) {
		const auto& layout = candidate.info.bindings;
		if (layout.push_data_start_dword !=
		        push_data.StartFor(layout.ShaderDataDwords(), candidate.info.stage) ||
		    candidate.specialization != specialization) {
			continue;
		}
		if constexpr (HasCalls) {
			if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
		}
		return &candidate;
	}
	return nullptr;
}

ShaderModule& PipelineCache::CompileModule(Program& program, const ProgramKey& key,
                                           ShaderRecompiler::CompileOptions       options,
                                           ShaderRecompiler::TranslateResult      translated,
                                           ShaderRecompiler::IR::PushData::Window push_data) {
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
	                                         Common::DebugCounters::Counter::ShaderCompileNs,
	                                         Common::DebugCounters::Peak::ShaderCompileNs);
	const auto profile = SpirvProfile(m_graphics);
	options.profile    = &profile;
	Common::CompilePause::Scope       compile_pause;
	Common::ShaderCompileStatus::Work compile_work;
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
	                                               program.specialization, push_data);
	auto info = std::move(result.program).TakeCompiledInfo();

	// The same inputs give the same module id across runs.
	const auto function_code = program.FunctionCode();
	ShaderRecompiler::IR::Writer inputs;
	ShaderRecompiler::IR::IoAll(inputs, key.stage, key.hash, key.user_data_count, key.static_state,
	                            info.bindings.push_data_start_dword, program.specialization);
	inputs.Raw(function_code.data(), function_code.size_bytes()); // Last, so it needs no size.
	const auto id = XXH3_64bits(inputs.bytes.data(), inputs.bytes.size());

	const auto* stage = ShaderRecompiler::StageName(key.stage);
	DumpShader(result.spirv, fmt::format("{}_{:016x}_{:016x}.spv", stage, key.hash, id));
	ValidateShaderSpirv(stage, key.hash, result.spirv, profile.maintenance8);
	auto& module = program.modules.emplace_back(ShaderModule {
	    .id             = id,
	    .module         = CompileSPV(result.spirv, m_graphics.device),
	    .info           = std::move(info),
	    .specialization = program.specialization,
	    .function_code  = {function_code.begin(), function_code.end()},
	});
	RetainModuleSpirv(module, result.spirv);
	m_compiled[static_cast<size_t>(key.stage)]++;
	// Guest geometry shaders are compiled through the host mesh stage.
	std::printf("Shaders: VS %u | PS %u | CS %u | GS %u | LS %u | HS %u | TES %u\n",
	            m_compiled[static_cast<size_t>(ShaderType::Vertex)],
	            m_compiled[static_cast<size_t>(ShaderType::Pixel)],
	            m_compiled[static_cast<size_t>(ShaderType::Compute)],
	            m_compiled[static_cast<size_t>(ShaderType::Mesh)],
	            m_compiled[static_cast<size_t>(ShaderType::Local)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationControl)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
	// The inputs of these come from the stage's static key.
	if (key.stage == ShaderType::Pixel &&
	    options.input_info.pixel->parameter_mode == ShaderPixelParameterMode::Rectangle) {
		module.rect_parameters = RectListParameters(*options.input_info.pixel, module.info);
	} else if (key.stage == ShaderType::Mesh) {
		module.mesh_task = options.input_info.vertex->indirect_draw;
	}
	// Programs with GPU-selected calls depend on their callees' current code.
	if (program.call_source == nullptr) {
		StoreModule(key, program, module, result.spirv);
	}
	return module;
}

// Diagnostic only: writes everything a shader translation reads, for tools/shaderCheck.
template <typename InputInfo>
static void DumpShaderCapsule(const ShaderParams& params, const InputInfo& input_info,
                              const ShaderRecompiler::CompileOptions& options) {
	static const char* dir = std::getenv("KYTY_SHADER_CAPSULES");
	if (dir == nullptr) return;
	const auto path = fmt::format("{}/{}_{:016x}.cap", dir,
	                              ShaderRecompiler::StageName(options.stage), params.hash);
	auto* file = std::fopen(path.c_str(), "wbx");
	if (file == nullptr) return;
	auto copy  = input_info;
	copy.stage = {};
	const uint32_t header[] {0x5041434bu, static_cast<uint32_t>(options.stage), options.wave_size,
	                         options.user_data_base, params.user_data_count,
	                         static_cast<uint32_t>(sizeof(copy)),
	                         static_cast<uint32_t>(params.code.size()),
	                         static_cast<uint32_t>(params.back_code.size()),
	                         static_cast<uint32_t>(params.hash),
	                         static_cast<uint32_t>(params.hash >> 32u)};
	std::fwrite(header, sizeof(header), 1, file);
	std::fwrite(&copy, sizeof(copy), 1, file);
	std::fwrite(params.user_data.data(), 4, params.user_data_count, file);
	std::fwrite(params.code.data(), 4, params.code.size(), file);
	std::fwrite(params.back_code.data(), 4, params.back_code.size(), file);
	std::fclose(file);
}

// Diagnostic only (KYTY_SKIP_FAILED_SHADERS): a shader that cannot be compiled is skipped, so one
// run shows later failures too. The picture is incomplete and leaks what the failure held.
template <typename InputInfo>
const ShaderModule* PipelineCache::GetProgram(const ShaderParams& params, InputInfo& input_info,
                                              ShaderRecompiler::IR::PushData::Window& push_data) {
	static const bool skip_failed = std::getenv("KYTY_SKIP_FAILED_SHADERS") != nullptr;
	if (!skip_failed) return GetProgramImpl(params, input_info, push_data);
	std::jmp_buf  recover;
	const int64_t work_depth = Common::ShaderCompileStatus::ThreadDepth();
	if (setjmp(recover) != 0) {
		Common::ShaderCompileStatus::Unwind(work_depth);
		PipelineCacheLog("KYTY_SKIP shader=0x{:016x} stage={}", m_lookup_key.hash,
		                 ShaderRecompiler::StageName(m_lookup_key.stage));
		m_program_cache.erase(m_lookup_key);
		m_program_cache.emplace(m_lookup_key, Program {.skip_dispatch = true});
		return nullptr;
	}
	Common::g_dbg_recover = &recover;
	const auto* module    = GetProgramImpl(params, input_info, push_data);
	Common::g_dbg_recover = nullptr;
	return module;
}

template <typename InputInfo>
const ShaderModule* PipelineCache::GetProgramImpl(const ShaderParams& params, InputInfo& input_info,
                                              ShaderRecompiler::IR::PushData::Window& push_data) {
	const auto user_data = std::span(params.user_data).first(params.user_data_count);
	ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = user_data,
	    .shader_base                = params.Base(),
	    .read_specialization_memory = ReadShaderGuestMemory,
	    .image_heap                 = m_image_heap,
	    .heap_rows                  = m_heap_rows,
	};
	ShaderRecompiler::CompileOptions options {.wave_size = input_info.wave_size};
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		options.stage            = input_info.logical_stage;
		options.input_info.vertex = &input_info;
		options.user_data_base   = 8;
		if (options.stage == ShaderType::Mesh ||
		    options.stage == ShaderType::TessellationControl) {
			options.user_data_base = 0;
		}
		if (options.stage == ShaderType::Mesh) {
			options.wave_size = input_info.mesh.wave_size;
		}
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		options.stage            = ShaderType::Pixel;
		options.input_info.pixel = &input_info;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
		options.stage              = ShaderType::Compute;
		options.input_info.compute = &input_info;
		runtime.workgroup_counts   = input_info.workgroup_counts;
		runtime.indirect_args_address = input_info.indirect_args_address;
	}
	m_lookup_key.stage           = options.stage;
	m_lookup_key.hash            = params.hash;
	m_lookup_key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, m_lookup_key.static_state);
	auto entry = m_program_cache.find(m_lookup_key);
	const auto bind = [&](const ShaderModule& module) {
		input_info.stage = {.module = &module, .resources = &entry->second.resources};
		module.info.bindings.AdvancePushData(push_data, module.info.stage);
		return &module;
	};
	bool source_changed = false;
	if (entry != m_program_cache.end()) {
		auto& cached = entry->second;
		if (cached.skip_dispatch) return nullptr;
		ShaderModule* module = nullptr;
		if (cached.call_source) {
			source_changed = cached.RefreshCallSource(runtime);
			if (!source_changed) module = cached.FindModule<true>(runtime, push_data);
		} else {
			module = cached.FindModule<false>(runtime, push_data);
		}
		if (module != nullptr) return bind(*module);
	}

	options.shader_hash     = params.hash;
	options.user_data       = user_data;
	options.back_code       = params.back_code;
	DumpShaderCapsule(params, input_info, options);
	options.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
	ShaderRecompiler::ShaderSource source;
	ShaderRecompiler::TranslateResult  translated;
	ShaderRecompiler::IR::ResourcePlan ahead_plan;
	// A translation ahead has no linked calls.
	bool translated_ahead = false;
	if (entry == m_program_cache.end()) {
		DumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
		                                    ShaderRecompiler::StageName(options.stage), params.hash));
		translated_ahead = TakeTranslation(m_lookup_key, translated, ahead_plan);
	}
	if (entry == m_program_cache.end() && !translated_ahead) {
		source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (source.unsupported_call_pc != UINT32_MAX) {
			PipelineCacheLog("!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x{:016x} pc=0x{:08x} "
			                 "reason=unsupported GPU-selected callable target; rendering is incomplete",
			                 params.hash, source.unsupported_call_pc);
			m_program_cache.emplace(m_lookup_key, Program {.skip_dispatch = true});
			return nullptr;
		}
	}
	auto* current = entry == m_program_cache.end() ? &source : entry->second.call_source.get();
	if (translated_ahead) {
		// `translated` holds it.
	} else if (current != nullptr) {
		if (!current->calls.empty()) {
			if (entry == m_program_cache.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
			translated = [&] { Common::CompilePause::Scope pause; Common::ShaderCompileStatus::Work work(false); return ShaderRecompiler::TranslateProgram(current->linked->decoded, options); }();
			translated.program.source_reads = current->reads;
		} else {
			translated = [&] { Common::CompilePause::Scope pause; Common::ShaderCompileStatus::Work work(false); return ShaderRecompiler::TranslateProgram(current->decoded, options); }();
		}
	} else {
		translated = [&] { Common::CompilePause::Scope pause; Common::ShaderCompileStatus::Work work(false); return ShaderRecompiler::TranslateProgram(params.code, options); }();
	}
	if (entry == m_program_cache.end()) {
		auto retained = !source.calls.empty()
		                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
		                    : nullptr;
		entry = m_program_cache.emplace(m_lookup_key, Program {
		    .call_source   = std::move(retained),
		    .resource_plan = translated_ahead
		                         ? std::move(ahead_plan)
		                         : ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
		}).first;
		auto& program = entry->second;
		if (!ShaderRecompiler::IR::MaterializeResources(program.resource_plan, runtime,
		                                                program.resources, program.specialization)) {
			EXIT("shader 0x%016" PRIx64 " (%s): resources could not be materialized\n",
			     m_lookup_key.hash, ShaderRecompiler::StageName(m_lookup_key.stage));
		}
	} else if (source_changed) {
		entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		if (const auto* module = entry->second.FindModule<true>(runtime, push_data)) {
			return bind(*module);
		}
	}
	return bind(CompileModule(entry->second, entry->first, options, std::move(translated),
	                          push_data));
}

PipelineWorkers::PipelineWorkers(): PipelineWorkers(Common::ParallelThreads()) {}

PipelineWorkers::PipelineWorkers(size_t threads) {
	for (auto i = threads; i != 0; i--) {
		m_threads.emplace_back([this](std::stop_token stop) {
			std::unique_lock lock(m_mutex);
			// Once stop is requested the wait returns false only when the queue is empty.
			while (m_ready.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
				const auto job = std::move(m_jobs.front());
				m_jobs.pop();
				Common::ShaderCompileStatus::RemoveQueued();
				lock.unlock();
				job();
				lock.lock();
			}
		});
	}
}

void PipelineWorkers::Post(Common::UniqueFunction<void>&& job) {
	{
		std::scoped_lock lock(m_mutex);
		m_jobs.push(std::move(job));
	}
	Common::ShaderCompileStatus::AddQueued();
	m_ready.notify_one();
}

void PipelineCache::StartCompile(Pipeline& pipeline, Common::UniqueFunction<vk::Pipeline>&& create) {
	std::promise<vk::Pipeline> promise;
	pipeline.compiling = promise.get_future().share();
	m_workers.Post([this, promise = std::move(promise), create = std::move(create)]() mutable {
		const auto handle = create();
		// A save may have run meanwhile without this pipeline.
		if (m_driver_cache != nullptr) m_driver_cache_dirty = true;
		promise.set_value(handle);
	});
}

namespace {

// A quarter of the hardware threads, 1 to 6: the guest's threads and the pipeline workers need
// the rest.
size_t TranslatorThreads() {
	return std::clamp<size_t>(std::thread::hardware_concurrency() / 4, 1, 6);
}

// Translations ahead not taken yet, queued ones included: a loading frame dispatches hundreds of
// new programs. More are not started; the oldest finished one goes first.
constexpr size_t MaxAheadPending = 1024;

// What translation reads of a program's user data is their number.
constexpr std::array<uint32_t, 40> NoUserData {};

} // namespace

struct PipelineCache::AheadJob {
	enum class State : uint8_t { Queued, Running, Done, Known };
	State                             state = State::Queued;
	ShaderComputeInputInfo            input;
	ShaderParams                      params;
	// The code, copied when asked for and checked against params.hash.
	std::vector<uint32_t>             code;
	ShaderRecompiler::TranslateResult  result;
	ShaderRecompiler::IR::ResourcePlan plan;
};

void PipelineCache::TranslateComputeAhead(const HW::CsStageRegisters& regs,
                                          uint32_t dispatch_initiator, bool async_compute) {
	if (regs.data_addr == 0 || m_ahead_stop) {
		return;
	}
	// The same input as DispatchDirect or DispatchIndirect, then GetComputeProgram, prepare.
	HW::ComputeShaderInfo info {.cs_regs = regs};
	info.cs_regs.wave_size = Pm4::ComputeWaveSize(dispatch_initiator);
	auto  job   = std::make_shared<AheadJob>();
	auto& input = job->input;
	input.async_compute = async_compute;
	input.dispatch_thread_dimensions =
	    (dispatch_initiator & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input.host_subgroup_size = m_compute_subgroup_size;
	if (!TryPrepareProgram(info, input, job->params)) {
		return;
	}
	input.lds_storage = input.lds_size_dwords * 4u > m_max_compute_shared_memory;
	// The copy is made here, on the asking thread, so workers never read guest memory.
	job->code.resize(job->params.code.size());
	if (!LibKernel::Memory::CopySparseMemory(job->params.Base(), job->code.data(),
	                                         job->code.size() * 4u) ||
	    XXH3_64bits(job->code.data(), job->code.size() * 4u) != job->params.hash) {
		return;
	}
	ProgramKey key {.stage           = ShaderType::Compute,
	                .hash            = job->params.hash,
	                .user_data_count = job->params.user_data_count};
	BuildStageStaticKey(input, key.static_state);
	{
		std::scoped_lock lock(m_ahead_mutex);
		if (m_ahead.contains(key)) {
			return;
		}
		if (m_ahead_pending.size() >= MaxAheadPending) {
			std::erase_if(m_ahead_pending,
			              [](const auto& pending) { return pending->state == AheadJob::State::Known; });
		}
		if (m_ahead_pending.size() >= MaxAheadPending) {
			// Drops the oldest finished translation: its dispatch translates it again.
			auto& oldest = *m_ahead_pending.front();
			if (oldest.state != AheadJob::State::Done) {
				return;
			}
			oldest.state  = AheadJob::State::Known;
			oldest.result = {};
			oldest.plan   = {};
			m_ahead_pending.pop_front();
		}
		m_ahead.emplace(std::move(key), job);
		m_ahead_pending.push_back(job);
	}
	m_translators.Post([this, job = std::move(job)] { TranslateAhead(*job); });
}

void PipelineCache::TranslateAhead(AheadJob& job) {
	{
		std::scoped_lock lock(m_ahead_mutex);
		if (job.state != AheadJob::State::Queued || m_ahead_stop) {
			return;
		}
		job.state = AheadJob::State::Running;
	}
	Common::ShaderCompileStatus::Work work;
	const auto& code = job.code;
	// The options GetProgramImpl gives a compute program.
	ShaderRecompiler::CompileOptions options {.wave_size = job.input.wave_size};
	options.stage              = ShaderType::Compute;
	options.input_info.compute = &job.input;
	options.shader_hash        = job.params.hash;
	options.user_data          = std::span(NoUserData).first(job.params.user_data_count);
	options.dump_ir            = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.emit_debug_info    = Config::GraphicsDebugDumpEnabled();
	ShaderRecompiler::TranslateResult  translated;
	ShaderRecompiler::IR::ResourcePlan plan;
	volatile bool                      usable = false;
	// The registers came from a scan, not from the command processor: a translation error here
	// may not be the dispatch's, so it ends this job only. Like KYTY_SKIP_FAILED_SHADERS, the
	// jump leaks what the translation held.
	std::jmp_buf recover;
	if (setjmp(recover) == 0) {
		Common::g_dbg_recover = &recover;
		auto source = ShaderRecompiler::PrepareShaderSource(code, options);
		// Linked calls read guest memory at the dispatch.
		if (!source.decoded.has_swap_pc) {
			translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
			plan       = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			usable     = true;
		}
		Common::g_dbg_recover = nullptr;
	} else {
		PipelineCacheLog("Shader 0x{:016x}: translation ahead failed; the dispatch translates it",
		                 job.params.hash);
	}
	{
		std::scoped_lock lock(m_ahead_mutex);
		if (usable && job.state == AheadJob::State::Running) {
			job.result = std::move(translated);
			job.plan   = std::move(plan);
			job.state  = AheadJob::State::Done;
			m_ahead_finished++;
		} else {
			job.state = AheadJob::State::Known;
		}
	}
	m_ahead_done.notify_all();
}

bool PipelineCache::TakeTranslation(const ProgramKey& key,
                                    ShaderRecompiler::TranslateResult& translated,
                                    ShaderRecompiler::IR::ResourcePlan& plan) {
	std::unique_lock lock(m_ahead_mutex);
	auto [entry, inserted] = m_ahead.try_emplace(key);
	if (inserted) {
		entry->second        = std::make_shared<AheadJob>();
		entry->second->state = AheadJob::State::Known;
		return false;
	}
	const auto job = entry->second;
	if (job->state == AheadJob::State::Queued) {
		// Not started: translating it here waits for no other job.
		job->state = AheadJob::State::Known;
		return false;
	}
	// Bounded, in case a translation hangs: the draw then translates the program itself and the
	// translation ahead is dropped when it ends.
	if (!m_ahead_done.wait_for(lock, std::chrono::seconds(2),
	                           [&] { return job->state != AheadJob::State::Running; })) {
		job->state = AheadJob::State::Known;
		return false;
	}
	if (job->state != AheadJob::State::Done) {
		return false;
	}
	m_ahead_taken++;
	translated  = std::move(job->result);
	plan        = std::move(job->plan);
	job->result = {};
	job->plan   = {};
	job->state  = AheadJob::State::Known;
	return true;
}

void PipelineCache::MarkProgramsKnown() {
	std::scoped_lock lock(m_ahead_mutex);
	for (const auto& key: m_program_cache | std::views::keys) {
		auto& job = m_ahead[key];
		job       = std::make_shared<AheadJob>();
		job->state = AheadJob::State::Known;
	}
}

vk::Pipeline PipelineCache::WaitForPipeline(const std::shared_future<vk::Pipeline>& compiling) {
	if (compiling.wait_for(std::chrono::seconds {0}) != std::future_status::ready) {
		const Common::CompilePause::Scope compile_pause;
		compiling.wait();
	}
	return compiling.get();
}

PipelineCache::Pipeline& PipelineCache::Resolve(Pipeline& pipeline, bool wait) {
	if (pipeline.compiling.valid() &&
	    (wait || pipeline.compiling.wait_for(std::chrono::seconds {0}) == std::future_status::ready)) {
		pipeline.pipeline  = WaitForPipeline(pipeline.compiling);
		pipeline.compiling = {};
	}
	return pipeline;
}

PipelineCache::PipelineCache(GraphicContext&                         graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap,
                             const std::filesystem::path&           shader_store)
    : m_graphics(graphics), m_image_heap(image_heap),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval),
      m_compute_subgroup_size(graphics.SupportsComputeWave64() ? 64u : 32u),
      m_max_compute_shared_memory(
          graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize),
      m_translators(TranslatorThreads()) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	{
		const Common::CompilePause::Scope compile_pause;
		WarmUp(shader_store);
	}
	MarkProgramsKnown();
}

PipelineCache::~PipelineCache() {
	// Queued translations are dropped; running ones end before anything they use goes.
	m_ahead_stop = true;
	m_translators.Finish();
	m_workers.Finish();
	Save();
	auto destroy = [this](auto& pipelines) {
		for (auto& pipeline: pipelines | std::views::values) {
			Resolve(pipeline, true);
			m_graphics.device.destroyPipeline(pipeline.pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
			m_graphics.device.destroyShaderModule(module.module, nullptr);
		}
	}
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_driver_cache_dirty = true;
	if (const auto now = std::chrono::steady_clock::now(); now >= m_next_driver_cache_save) {
		m_driver_cache_dirty     = false;
		m_next_driver_cache_save = now + DriverCacheSaveInterval;
		// The cache object is internally synchronized, so pipelines may be created meanwhile.
		m_store.Post([this] { WriteDriverCache(); });
	}
}

void PipelineCache::Save() {
	m_store.Finish();
	if (m_driver_cache_dirty && m_driver_cache != nullptr) {
		m_driver_cache_dirty = false;
		WriteDriverCache();
	}
}

void PipelineCache::WriteDriverCache() const {
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
}

void PipelineCache::GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state) {
	const auto& context           = command.GetRegisters();
	const auto& user_config       = command.GetUserConfig();
	const auto& sh                = context.GetShaderRegisters();
	const auto& vertex_regs       = command.GetShaders().GetVs();
	const auto& pixel_regs        = command.GetShaders().GetPs();
	auto&       vertex_info       = state.vertex_info;
	auto&       pixel_info        = state.ps_input_info;
	const bool  object_primitive_id = context.HasObjectPrimitiveIdOutput();
	const auto  write_mask          = context.GetRenderTargetMask() & sh.m_cbShaderMask;
	uint32_t    color_output_mask   = object_primitive_id ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			color_output_mask |= 1u << slot;
		}
	}
	const auto& db = sh.db_shader_control;
	// Without color outputs, a pixel shader still runs for its depth or coverage side effects.
	const bool pixel_active =
	    pixel_regs.ps_regs.data_addr != 0 &&
	    (color_output_mask != 0 || db.shader_kill_enable || db.shader_z_export_enable ||
	     db.shader_mask_export_enable || db.shader_dual_export_enable ||
	     db.shader_execute_on_noop || sh.AlphaToMask());
	state.ps_active = pixel_active;
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		// Guest geometry shaders run as host mesh shaders, which cannot follow tessellation.
		if ((context.GetShaderStages() & 0x20u) != 0) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(
				    "Warning: dropping tessellation draws with a geometry shader.\n");
			}
			return;
		}
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
		// Reflection decodes both programs; the key also covers the VGT fields in tess.
		const uint64_t layout_key[] = {vertex_params[0].hash, vertex_params[1].hash,
		                               uint64_t {sh.m_vgtLsHsConfig & 0xfff00u} << 32u |
		                                   sh.m_vgtTfParam};
		const auto [layout, inserted] = m_tess_layouts.try_emplace(
		    XXH3_64bits(layout_key, sizeof(layout_key)), vertex_info[0].tess);
		if (inserted) {
			ShaderRecompiler::AnalyzeTessellationPrograms(vertex_params[0].code,
			                                              vertex_params[1].code, layout->second);
			StoreLayout(layout->first, layout->second);
		}
		for (auto& stage: vertex_info) {
			stage.tess = layout->second;
		}
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (!tess_active && !mesh_active) {
		vertex_info[0].indirect_offsets = state.indirect_offsets;
		vertex_info[0].indirect_draw    = state.indirect_draw;
	}
	if (object_primitive_id) {
		// PA_CL_OBJPRIM_ID_CNTL = 2: VGT_OBJECT_ID plus the pipelined 32-bit primitive ID.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || !pixel_active ||
		                     (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     (!mesh_active && user_config.GetPrimType() != Prospero::PrimitiveType::kTriList));
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.output_primitive_id = object_primitive_id;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: dropping mesh draws the host does not support (shader 0x{:016x}).\n",
				    vertex_params[0].hash));
			}
			return;
		}
		// A task shader launches each record's groups and passes its draw parameters in a
		// payload beside the LDS, the allocation pair and the Layer array; without room for
		// it, the CPU reads the records.
		// AMD GPUs hang on these task+mesh draws before any task or mesh wave starts: KytyPS5
		// PR #2302 (de49ee9186) saw it with Adrenalin on an RX 6600, and RADV
		// on an RX 7800 XT loses its gfx ring the same way in GTA VI (PPSA38500): the hang report
		// names the task dispatch of an indirect non-indexed mesh draw with no wave in flight.
		// On AMD the records go through the CPU instead.
		const bool task_shader_usable = m_graphics.physical_device_properties.vendorID != 0x1002u;
		if (state.indirect_draw && !task_shader_usable) {
			static std::atomic_bool noted = false;
			if (!noted.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(
				    "GPU: indirect mesh draws read their records on the CPU on AMD (no task "
				    "shader).\n");
			}
		}
		if (state.indirect_draw && task_shader_usable &&
		    (mesh.lds_size_dwords + 2u + mesh.max_vertices +
		     ShaderRecompiler::IR::PushData::MeshTaskPayloadDwordCount) * sizeof(uint32_t) <=
		        limits.maxMeshPayloadAndSharedMemorySize) {
			vertex_info[0].indirect_offsets = state.indirect_offsets;
			vertex_info[0].indirect_draw    = true;
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
		    target_export_mapping {};
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			const auto& rt = context.GetRenderTarget(slot);
			if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
				target_export_mapping[slot] =
				    TextureGetRenderTargetFormat(m_graphics, rt.info.format, rt.info.channel_type,
				                                 rt.info.channel_order)
				        .export_mapping;
			}
		}
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
				pixel_info.target_uint_mask |= 1u << export_index;
			} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
				pixel_info.target_sint_mask |= 1u << export_index;
			}
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0; blended targets are not integer.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
			pixel_info.target_uint_mask &= ~2u;
			pixel_info.target_sint_mask &= ~2u;
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		// CLIP_DISABLE also disables user planes. How they combine with shader clip and cull
		// distances is unverified; draw push constants only reach VS/mesh and PS, and the
		// generated RECTLIST stages do not forward clip distances.
		EXIT_NOT_IMPLEMENTED((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		                     Prospero::IsRectList(user_config.GetPrimType()));
		last_stage.user_clip_planes = clip_control.user_clip_planes;
	}
	auto push_data = ShaderRecompiler::IR::PushData::GraphicsWindow(
	    mesh_active, object_primitive_id, last_stage.user_clip_planes);
	if (pixel_active) {
		const auto* pixel_program = GetProgram(pixel_params, pixel_info, push_data);
		if (pixel_program == nullptr) {
			vertex_info[0].stage = {};
			return;
		}
		const auto& pixel_module = *pixel_program;
		for (const auto& output: pixel_module.info.info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				state.mrt_mask |= 1u << output.location;
			}
		}
		state.mrt_mask &= color_output_mask;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		if (GetProgram(vertex_params[i], vertex_info[i], push_data) == nullptr) {
			vertex_info[0].stage = {};
			return;
		}
	}
}

const ShaderModule* PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                                     ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_compute_subgroup_size;
	const auto        params      = PrepareProgram(regs, input_info);
	// TranslateComputeAhead prepares its input like this.
	input_info.lds_storage = input_info.lds_size_dwords * 4u > m_max_compute_shared_memory;
	ShaderRecompiler::IR::PushData::Window push_data;
	const auto* module = GetProgram(params, input_info, push_data);
	// HW_ID2 reads are translated with the graphics compute queue's values.
	EXIT_NOT_IMPLEMENTED(module != nullptr && input_info.async_compute &&
	                     module->info.info.reads_queue_id);
	return module;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(const CommandBuffer&   command,
                                                            const DrawRenderState& state,
                                                            const DrawTargets&     targets,
                                                            bool                   wait) {
	const auto* colors        = targets.colors;
	const auto& depth         = targets.depth;
	const auto  vertex_info   = state.VertexStages();
	const auto& ps_input_info = state.ps_input_info;
	const auto& vs_input_info = vertex_info.front();
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	const bool ps_active    = state.ps_active;
	const bool tessellation = vertex_info.size() == 3;
	std::array<const ShaderModule*, 4> stages {};
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		stages[i] = vertex_info[i].stage.module;
	}
	if (ps_active) {
		stages[3] = ps_input_info.stage.module;
	}
	const bool mesh        = stages[0]->info.stage == ShaderType::Mesh;
	const auto color_count = targets.color_count;

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < stages.size(); i++) {
		if (stages[i] != nullptr) {
			key.stage_ids[i] = stages[i]->id;
		}
	}
	auto&    static_params      = key.static_params;
	auto&    rendering          = key.rendering;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_active) {
			alpha_source = ps_input_info.alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = depth.desc.view_info.aspect;
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = state.topology;
	static_params.primitive_restart_enable = state.primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info.ps_sample_shading;
	static_params.alpha_to_coverage_enable = ps_active && ctx.GetShaderRegisters().AlphaToMask();
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);
	if (tessellation) {
		static_params.patch_control_points = vs_input_info.tess.input_control_points;
	} else if (!mesh && static_params.topology == vk::PrimitiveTopology::ePatchList) {
		// Rectangle lists expand through generated tessellation stages.
		static_params.patch_control_points = 3;
	}

	if (!mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			    .format  = vs_input_info.resources[attribute].RawFormat(),
			};
		}
	}

	auto [iter, inserted] = m_graphics_pipelines.try_emplace(key);
	auto& pipeline        = iter->second;
	if (!inserted) {
		return Resolve(pipeline, wait);
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(ps_input_info);
		}
	}

	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineLayouts(m_graphics, pipeline, stages);
		StartCompile(pipeline, [this, layout = pipeline.pipeline_layout, &key = iter->first, stages] {
			return CreateHostPipeline(m_graphics, layout, key, stages, m_driver_cache);
		});
	}
	StorePipeline(key);
	CheckpointDriverCache();

	return Resolve(pipeline, wait);
}

PipelineCache::Pipeline& PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                                           bool                          wait) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	const auto& module = *input_info.stage.module;

	auto [iter, inserted] = m_compute_pipelines.try_emplace(module.id);
	auto& pipeline        = iter->second;
	if (!inserted) {
		return Resolve(pipeline, wait);
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		const auto* stage = &module;
		CreatePipelineLayouts(m_graphics, pipeline, {&stage, 1});
		StartCompile(pipeline, [this, layout = pipeline.pipeline_layout, stage] {
			return CreateHostPipeline(m_graphics, layout, *stage, m_driver_cache);
		});
	}
	CheckpointDriverCache();

	return Resolve(pipeline, wait);
}
} // namespace Libs::Graphics
