#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreLock.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreRestamp.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct ShaderParams;
class CommandBuffer;

namespace ShaderRecompiler {
struct CompileOptions;
struct ShaderSource;
struct TranslateResult;
} // namespace ShaderRecompiler

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

namespace ShaderRecompiler::Spirv {
struct Profile;
} // namespace ShaderRecompiler::Spirv

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	// 0 without tessellation, 3 for a rectangle list, else the input control points.
	uint32_t                   patch_control_points     = 0;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
};

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride   = 0;
		uint8_t  instance = 0;
	};
	struct Attribute {
		uint32_t offset  = 0;
		uint8_t  binding = 0;
		uint8_t  format  = 0; // Guest buffer format.
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;
};

// Everything a graphics pipeline is made of besides the stage modules its ids name. Aligned so
// that rendering.color_formats can be passed to Vulkan.
struct alignas(4) GraphicsPipelineKey {
	// VS, LS or mesh; HS; TES; PS. 0 marks an absent stage.
	std::array<uint64_t, 4>  stage_ids {};
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;

	bool operator==(const GraphicsPipelineKey& other) const;
};

#pragma pack(pop)

// Keys are compared and hashed as bytes.
static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>);

// A compiled stage with what pipeline creation needs from it besides the key.
struct ShaderModule {
	uint64_t                                     id     = 0;
	vk::ShaderModule                             module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	// Rectangle-list PS: the parameters it reads.
	std::vector<RectListParameter>               rect_parameters;
	// Indirect mesh draw: a task shader reads the records and launches the groups.
	bool                                         mesh_task = false;
};

// A draw's depth target and its first color_count color targets.
struct DrawTargets {
	RenderDepthInfo depth;
	RenderColorInfo colors[RENDER_COLOR_ATTACHMENTS_MAX];
	uint32_t        color_count = 0;
};

// A draw's topology and shader stages (from PipelineCache::GetGraphicsPrograms).
struct DrawRenderState {
	// Color targets the pixel program writes.
	uint32_t                             mrt_mask = 0;
	vk::PrimitiveTopology                topology = vk::PrimitiveTopology::ePointList;
	bool                                 primitive_restart_enable = false;
	bool                                 ps_active                = false;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	// Input: the user-data slots a GPU-read indirect draw patches.
	ShaderIndirectOffsets                indirect_offsets;
	bool                                 indirect_draw = false;

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		return {vertex_info.data(), vertex_info[1].stage ? 3u : 1u};
	}
};

// Writes the shader store on its own thread, running jobs in the order they were posted.
class ShaderStoreWriter {
public:
	ShaderStoreWriter();
	// Keeps the first `kept` bytes of the store at `path`, or starts it over with `header`.
	void Open(const std::filesystem::path& path, std::span<const uint8_t> header, uint64_t kept);
	// Whether the store is open.
	explicit operator bool() const { return !m_file.IsInvalid(); }
	// Compresses `record` and appends it on the store thread, in call order.
	void Append(std::vector<uint8_t>&& record);
	void Post(Common::UniqueFunction<void>&& job);
	// Runs the queued jobs and stops the thread; jobs posted later never run. Safe to call again.
	void Finish() { m_thread = {}; }
	// Finish(), then closes the file; the lock stays held (the file may be replaced meanwhile).
	void CloseFile() {
		Finish();
		m_file.Close();
	}
	// CloseFile(), then releases the lock.
	void Close() {
		CloseFile();
		m_lock.Release();
	}

private:
	friend struct PipelineCacheTestAccess;

	Common::File                             m_file;
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	// Declared last, so it drains the queue and joins before the rest is destroyed.
	std::jthread m_thread;
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	// Without a shader store path nothing is kept across runs.
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	              const std::filesystem::path& shader_store = {});
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// Finishes the queued writes and stops the store thread, then writes the driver cache if it
	// changed.
	void Save();

	// --precompile-shaders (emulator.cpp), set before the cache exists: the recipes' modules are
	// translated on the pipeline workers at their normal priority, and nothing is decoded for a
	// game that never runs.
	static void SetPrecompileMode();
	// With SetPrecompileMode: waits until every module the recipes name is translated and stored
	// and every known pipeline is created, printing progress, then saves the caches and prints a
	// summary. SIGINT or SIGTERM ends it early with what is done saved. False if anything failed.
	bool Precompile();
	// --restamp-shader-store, set before the cache exists: a store written under another stamp
	// is loaded if it agrees in everything else, and `sample` of its modules (0: all) are
	// translated again from their recipes. `strict` (and any sample) keeps the all-or-nothing
	// check; otherwise the carry-over is partial.
	static void SetRestampMode(size_t sample, bool strict = false);
	// With SetRestampMode. Strict: gives the store this build's stamp if every module translated
	// again has the stored SPIR-V, and leaves it alone otherwise. Partial (the default): rewrites
	// the store under this build's stamp with only the module records this build translates to
	// exactly the stored record (SPIR-V, resource plan and module fields), and the pipelines whose
	// stages all stayed; the rest is dropped and comes back from the recipes. 0 done, 4 refused
	// (partial: nothing could be kept), 1 interrupted or failed.
	int Restamp();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		// The push constant range's stages, which every update names.
		vk::ShaderStageFlags    push_stages;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
	};

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	// Null skips the dispatch.
	const ShaderModule* GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                      ShaderComputeInputInfo&      input_info);

	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state,
	                              const DrawTargets& targets);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info);

private:
	friend struct PipelineCacheTestAccess;

	struct Program {
		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<ShaderModule>                    modules;
		bool                                         skip_dispatch = false;

		[[nodiscard]] std::span<const uint32_t> FunctionCode() const;
		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime);
		template <bool HasCalls>
		ShaderModule* FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                         ShaderRecompiler::IR::PushData::Window  push_data);
	};

	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	// Variants of one source share a bucket; ProgramKey equality compares the state.
	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const { return key.hash; }
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	using ModuleMap = std::unordered_map<uint64_t, const ShaderModule*>;

	GraphicContext&                        m_graphics;
	ShaderRecompiler::IR::ImageHeapSource* m_image_heap;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash> m_program_cache;
	// Modules compiled per stage.
	std::array<uint32_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> m_compiled {};
	ProgramKey                             m_lookup_key;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	bool                          m_driver_cache_dirty = false;
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_graphics_pipelines;
	std::unordered_map<uint64_t, Pipeline>                                     m_compute_pipelines;
	std::unordered_map<uint64_t, ShaderTessellationInputInfo>                  m_tess_layouts;
	// Records everything first created if a shader store is open, and writes the driver cache.
	ShaderStoreWriter m_store;
	// Host values TranslateComputeAhead needs off the command thread.
	uint32_t m_compute_subgroup_size     = 64;
	uint32_t m_max_compute_shared_memory = 0;
	// Translations ahead by program key, and the keys of programs that need none. Unlike the
	// program cache, any thread may use it.
	std::mutex                                                     m_ahead_mutex;
	std::condition_variable                                        m_ahead_done;
	std::unordered_map<ProgramKey, std::shared_ptr<AheadJob>, ProgramKeyHash> m_ahead;
	// Queued, running and finished translations not taken yet, oldest first, and their number.
	std::deque<std::shared_ptr<AheadJob>> m_ahead_pending;
	std::atomic<bool>                     m_ahead_stop = false;
	// Unoptimized pipelines replaced by their optimized builds (in-flight commands use them).
	std::vector<vk::Pipeline>             m_replaced_pipelines;
	// Set at destruction: background pipeline creations not started yet are dropped.
	std::shared_ptr<std::atomic<bool>>    m_deferred_stop = std::make_shared<std::atomic<bool>>(false);
	// Translations ahead finished and taken by the command thread.
	std::atomic<uint32_t>                 m_ahead_finished = 0;
	uint32_t                              m_ahead_taken    = 0;
	// Recipes: the file, what it holds, the translations done and the pipelines still to create.
	ShaderStoreWriter                           m_recipes;
	std::unordered_set<uint64_t>                m_recipe_codes;
	std::unordered_set<uint64_t>                m_recipe_modules;
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> m_recipe_pipeline_keys;
	std::vector<GraphicsPipelineKey>            m_recipe_pipelines;
	std::mutex                                  m_replay_mutex;
	std::vector<ReplayedModule>                 m_replayed;
	std::atomic<bool>                           m_replay_ready = false;
	std::atomic<size_t>                         m_replay_left  = 0;
	size_t                                      m_replay_merged = 0;
	size_t                                      m_replay_dropped = 0;
	// Modules this translator gives another id than their recipe; the id stands for the
	// recorded ones in the recipes' pipeline keys.
	size_t                                      m_replay_renamed = 0;
	std::unordered_map<uint64_t, uint64_t>      m_recipe_aliases;
	std::vector<std::string>                    m_replay_failures;
	size_t                                      m_replay_total   = 0;
	// Recipe pipelines whose stages no recipe gave.
	size_t                                      m_replay_lacking = 0;
	bool                                        m_replay_active  = false;
	// Restamp(): the store's bytes and modules, and what the comparison found.
	std::shared_ptr<const std::vector<uint8_t>> m_restamp_store;
	std::filesystem::path                       m_restamp_path;
	std::vector<uint8_t>                        m_restamp_stamp;
	ModuleMap                                   m_restamp_modules;
	bool                                        m_restamp_current   = false;
	bool                                        m_restamp_loaded    = false;
	size_t                                      m_restamp_stored    = 0;
	size_t                                      m_restamp_uncovered = 0;
	size_t                                      m_restamp_verified  = 0;
	size_t                                      m_restamp_differing = 0;
	size_t                                      m_restamp_failed    = 0;
	// Partial restamp: the module ids whose record this build reproduced apart from the plan, those
	// whose plan matched too, the count whose SPIR-V matched but not the module fields, the store's
	// records in order (their frames in m_restamp_store) and this build's store header.
	std::unordered_set<uint64_t>                m_restamp_identical;
	size_t                                      m_restamp_spirv_only = 0;
	std::unordered_set<uint64_t>                m_restamp_plan_identical;
	size_t                                      m_restamp_plan_differing = 0;
	std::vector<RestampRecord>                  m_restamp_records;
	std::vector<std::span<const uint8_t>>       m_restamp_frames;
	std::vector<uint8_t>                        m_restamp_header;
	std::chrono::steady_clock::time_point       m_replay_start;
	// Decode the store's records not asked for yet, in store order, at low priority.
	struct StoreDecoders;
	std::shared_ptr<StoreDecoders> m_store_decoders;
	// Declared last, so its jobs end before anything they use is destroyed.
	std::unique_ptr<PipelineWorkers> m_replayers;
	PipelineWorkers m_workers;
	PipelineWorkers m_translators;

	// Creates the host pipeline of `pipeline`, whose layouts exist, on a worker. With
	// `fast_first` (KYTY_FAST_PIPELINES) an unoptimized pipeline first, then the optimized one in
	// the background, which Upgrade swaps in.
	void StartCompile(Pipeline& pipeline, Common::UniqueFunction<vk::Pipeline, bool>&& create,
	                  bool fast_first = false);
	// Swaps in a finished optimized build; the unoptimized pipeline lives until destruction.
	void Upgrade(Pipeline& pipeline);
	// The same in the background: workers create it once nothing else waits, a draw at once.
	void StartDeferredCompile(Pipeline& pipeline, Common::UniqueFunction<vk::Pipeline>&& create);
	// A draw needs `pipeline`: a background creation not started yet goes before other work.
	void Expedite(Pipeline& pipeline);
	// Runs a translation ahead on a translator thread.
	void TranslateAhead(AheadJob& job);
	// Registers the job under `key` unless the key is known, and posts it.
	void QueueAhead(std::shared_ptr<AheadJob> job, ProgramKey key);
	// On the command thread, for a program key looked up for the first time: takes its
	// translation ahead and the resource plan extracted from it, waiting for a running one, or
	// returns false. Either way, later requests for the key are ignored.
	bool TakeTranslation(const ProgramKey& key, ShaderRecompiler::TranslateResult& translated,
	                     ShaderRecompiler::IR::ResourcePlan& plan);
	// Marks the keys of the programs in the cache as needing no translation ahead.
	void MarkProgramsKnown();

	template <typename InputInfo>
	const ShaderModule* GetProgram(const ShaderParams& params, InputInfo& input_info,
	                               ShaderRecompiler::IR::PushData::Window& push_data);
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions       options,
	                            ShaderRecompiler::TranslateResult      translated,
	                            ShaderRecompiler::IR::PushData::Window push_data);
	void InitializeDriverCache();
	// Marks the driver cache changed and, once per interval, has the store thread write it.
	void CheckpointDriverCache();
	void WriteDriverCache() const;

	// pipelineSerialization.cpp
	void WarmUp(const std::filesystem::path& path);
	bool LoadRecord(std::span<const uint8_t>& records);
	void StoreModule(const ProgramKey& key, const Program& program, const ShaderModule& module,
	                 const std::vector<uint32_t>& spirv);
	void StorePipeline(const GraphicsPipelineKey& key);
	void StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout);
	// Shader recipes (pipelineSerialization.cpp): the inputs of every stored module's translation,
	// kept across translator changes, so a start with a discarded store translates them again in
	// the background.
	struct RecipeJob;
	void OpenRecipes(const std::filesystem::path& store_path);
	// The command thread compiled `module` of a program without linked calls.
	void RecordRecipe(const ProgramKey& key, const ShaderParams& params,
	                  const ShaderRecompiler::CompileOptions& options,
	                  ShaderRecompiler::IR::PushData::Window push_data, const ShaderModule& module);
	void RecordRecipePipeline(const GraphicsPipelineKey& key);
	void ReplayRecipe(RecipeJob& job);
	bool TranslateRecipe(RecipeJob& job, ReplayedModule& replayed);
	// Restamp(): compares the translation with the stored module of the recipe's id.
	void VerifyRecipe(RecipeJob& job);
	// Restamp() in partial mode, once every comparison is done.
	int RestampPartial(std::chrono::steady_clock::time_point start, bool interrupted);
	// On the command thread: adds the modules translated from recipes, then creates the recipes'
	// pipelines whose stages all exist, in the background.
	void MergeReplayed();
	// Creates, in the background, the recipes' pipelines whose stages all exist.
	void CreateRecipePipelines();
	// Every module by id, with id 0 naming an absent stage.
	[[nodiscard]] ModuleMap MapModules() const;
	// False when a stage id of the key is not in `modules`.
	bool CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                            const ModuleMap& modules);
};

// The shader store of the running title; empty unless --shader-cache is on in a Release build
// without debug dumps, which change the SPIR-V.
std::filesystem::path ShaderStorePath();
// Host capabilities that change the emitted SPIR-V.
ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics);

// Stages are indexed like GraphicsPipelineKey::stage_ids.
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
