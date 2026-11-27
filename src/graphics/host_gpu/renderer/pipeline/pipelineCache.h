#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <future>
#include <deque>
#include <atomic>
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
#include <unordered_set>
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
class HeapRowSource;
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

// A module record of the shader store, decoded on first use or by the store's background decoders
// (pipelineSerialization.cpp): the store keeps only its key and pipeline fields decoded at start.
struct StoredModule {
	// The store's bytes, released once decoded, and the record's frame and fields in them.
	std::shared_ptr<const std::vector<uint8_t>> store;
	std::span<const uint8_t>                    frame;
	uint32_t                                    plan_offset  = 0;
	uint32_t                                    plan_size    = 0;
	uint32_t                                    spirv_offset = 0;
	uint32_t                                    spirv_words  = 0;
	vk::Device                                  device;
	// The first record of a program also gives the program's resource plan. The module (cheap)
	// and the plan (most of the time) decode separately: pipeline creation needs only the module.
	bool                                        want_plan = false;
	// A module the run does not use (KYTY_RT_OFF); its record may still give the plan.
	bool                                        module_dropped = false;
	std::once_flag                              module_once;
	std::once_flag                              plan_once;
	std::atomic<bool>                           decoded      = false; // The module.
	std::atomic<bool>                           plan_decoded = false;
	// Parts still to decode; the last one releases the store's bytes.
	std::atomic<int>                            parts_left = 1;
	vk::ShaderModule                            handle     = nullptr;
	ShaderRecompiler::IR::ResourcePlan          plan;
	bool                                        plan_ok = false;
	// With RetainModuleSpirv().
	std::vector<uint32_t>                       spirv;
};
// Safe on any thread: decodes the record's module once (a call during another thread's decode
// waits for it) and returns its handle.
vk::ShaderModule ResolveStoredModule(StoredModule& stored);
// The same for the resource plan of a program's first record; false when it cannot be decoded.
bool ResolveStoredPlan(StoredModule& stored);

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
	// The SPIR-V's size, and the SPIR-V itself with KYTY_FAILED_SPIRV_DUMP=1, for the report of a
	// pipeline the driver fails to create (after Axellito/VIKyng bcee3021).
	size_t                                       spirv_words = 0;
	std::vector<uint32_t>                        spirv;
	// A module loaded from the shader store has no handle until Handle() first asks for it.
	std::shared_ptr<StoredModule>                stored;

	[[nodiscard]] vk::ShaderModule Handle() const {
		return stored != nullptr ? ResolveStoredModule(*stored) : module;
	}
};

// KYTY_FAILED_SPIRV_DUMP=1: modules keep their SPIR-V, written to the shader log folder's failed/
// when a pipeline that uses them cannot be created.
bool RetainModuleSpirv();
void RetainModuleSpirv(ShaderModule& module, const std::vector<uint32_t>& spirv);

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

private:
	friend struct PipelineCacheTestAccess;

	Common::File                             m_file;
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	// Declared last, so it drains the queue and joins before the rest is destroyed.
	std::jthread m_thread;
};

// Runs jobs in the order they were posted; background jobs only when no other job waits.
class PipelineWorkers {
public:
	PipelineWorkers();
	explicit PipelineWorkers(size_t threads);
	~PipelineWorkers() { Finish(); }
	KYTY_CLASS_NO_COPY(PipelineWorkers);
	void Post(Common::UniqueFunction<void>&& job, bool background = false);
	// Runs the queued jobs and stops the threads.
	void Finish() { m_threads.clear(); }

private:
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	std::queue<Common::UniqueFunction<void>> m_background;
	std::vector<std::jthread>                m_threads;
};

// A pipeline creation queued in the background (the shader store's warm-up) that a draw needing
// it runs at once, on whichever thread asks first.
class DeferredPipeline {
public:
	explicit DeferredPipeline(Common::UniqueFunction<vk::Pipeline>&& create)
	    : m_create(std::move(create)), m_result(m_promise.get_future().share()) {}
	KYTY_CLASS_NO_COPY(DeferredPipeline);
	// Creates the pipeline unless another thread did or does; with `skip` a null one instead.
	void Run(bool skip = false);
	[[nodiscard]] bool Claimed() const { return m_claimed.load(std::memory_order_acquire); }
	// True the first time only.
	bool MarkExpedited() { return !m_expedited.exchange(true, std::memory_order_relaxed); }
	[[nodiscard]] const std::shared_future<vk::Pipeline>& Result() const { return m_result; }

private:
	std::atomic<bool>                    m_claimed   = false;
	std::atomic<bool>                    m_expedited = false;
	Common::UniqueFunction<vk::Pipeline> m_create;
	std::promise<vk::Pipeline>           m_promise;
	std::shared_future<vk::Pipeline>     m_result;
};

// A pipeline a recorded command waits for, from PipelineCache::Pipeline::Pending().
struct PendingPipeline {
	std::shared_future<vk::Pipeline>  compiling;
	std::shared_ptr<DeferredPipeline> deferred;

	[[nodiscard]] bool Ready() const {
		return compiling.wait_for(std::chrono::seconds {0}) == std::future_status::ready;
	}
	// Runs a queued background creation here rather than waiting for its turn.
	// Holds the guest clock while it waits or creates it.
	vk::Pipeline Get() const;
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	// Without a shader store path nothing is kept across runs.
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	              const std::filesystem::path& shader_store = {});
	~PipelineCache();
	void SetHeapRows(ShaderRecompiler::IR::HeapRowSource* rows) { m_heap_rows = rows; }
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
	// translated again from their recipes.
	static void SetRestampMode(size_t sample);
	// With SetRestampMode: gives the store this build's stamp if every module translated again
	// has the stored SPIR-V, and leaves it alone otherwise. 0 done, 4 refused, 1 interrupted or
	// failed.
	int Restamp();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		// Null while a worker still compiles it.
		vk::Pipeline            pipeline              = nullptr;
		std::shared_future<vk::Pipeline> compiling;
		// Set while `compiling` is a background creation not started yet.
		std::shared_ptr<DeferredPipeline> deferred;
		// KYTY_FAST_PIPELINES: the optimized build that replaces an unoptimized `pipeline`.
		std::shared_future<vk::Pipeline> optimized;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		// The push constant range's stages, which every update names.
		vk::ShaderStageFlags    push_stages;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
		// Per descriptor type, the descriptors one set of descriptor_set_layout holds.
		std::vector<vk::DescriptorPoolSize> descriptor_sizes;

		[[nodiscard]] PendingPipeline Pending() const { return {compiling, deferred}; }
	};

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	// Null skips the dispatch.
	const ShaderModule* GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                      ShaderComputeInputInfo&      input_info);

	// `fast_first`: the draw waits for a new pipeline, so it may be created unoptimized first.
	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state,
	                              const DrawTargets& targets, bool wait = true,
	                              bool fast_first = true);
	// KYTY_FAST_PIPELINES=1: new pipelines a command waits for are created without driver
	// optimizations first, and the optimized build replaces them when ready.
	static bool FastPipelines();
	// Takes a finished compile, or with `wait` the running one.
	static Pipeline& Resolve(Pipeline& pipeline, bool wait);
	// KYTY_ASYNC_PIPELINES: 0 (default) every draw waits for its pipeline; 1 ordinary geometry
	// whose pipeline is not ready when its commands are submitted is skipped for that frame,
	// while full-screen passes, depth or stencil writers, draws whose shaders write memory and
	// all dispatches wait; 2 also skips depth and stencil writers.
	static int AsyncPipelineMode();
	// KYTY_LAZY_SHADER_STORE=1 (frame replays): the shader store decodes a record only when a
	// lookup needs it and creates a stored pipeline only when a draw or dispatch needs it, with no
	// background decoders, pipeline creation, recipes or Vulkan pipeline cache file. Keeps the
	// memory a replay needs small.
	static bool LazyShaderStore();
	// A draw skipped because its pipeline was not ready; reported every 5 s.
	static void NoteSkippedDraw();
	// Waits for a pipeline being built on a worker; the guest clock stands still while it waits.
	static vk::Pipeline WaitForPipeline(const std::shared_future<vk::Pipeline>& compiling);
	// Without `wait` the pipeline may still be compiling.
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info, bool wait = true);

	// Safe on any thread. Starts translating, on a worker, the program of a compute dispatch with
	// these registers and DISPATCH_INITIATOR, unless that program was asked for before. The
	// dispatch adopts the translation if its program key is the same. Wrong registers only waste
	// the work. The code is copied from guest memory on the calling thread, which should be a
	// guest thread (it handles faults on GPU-written memory).
	void TranslateComputeAhead(const HW::CsStageRegisters& regs, uint32_t dispatch_initiator,
	                           bool async_compute);
	// The same for a draw's vertex (or mesh) and pixel programs, from the registers a scan of the
	// submitted commands sees at the draw (KYTY_NO_GRAPHICS_AHEAD=1 turns it off). Tessellation
	// draws are left to the command thread.
	void TranslateGraphicsAhead(const HW::Context& context, const HW::UserConfig& user_config,
	                            const HW::Shader& shaders, bool indirect);

private:
	friend struct PipelineCacheTestAccess;

	struct Program {
		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// Compiling pipelines keep references.
		std::deque<ShaderModule>                     modules;
		bool                                         skip_dispatch = false;
		// A program from the shader store whose resource plan is still in its first record.
		std::shared_ptr<StoredModule>                stored_plan;
		// Programs with calls: the resource plans of the earlier links call_source keeps, by link
		// id, so a callee set seen before needs no translation to find its module.
		std::unordered_map<uint64_t, ShaderRecompiler::IR::ResourcePlan> call_plans;

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

	// A translation started ahead, or a key whose program the command thread looked up first.
	struct AheadJob;

	// What a draw's programs are prepared from (PrepareGraphicsStages).
	struct GraphicsStageInputs {
		std::array<ShaderParams, 3>            vertex_params;
		ShaderParams                           pixel_params;
		ShaderRecompiler::IR::PushData::Window push_data;
		uint32_t                               color_output_mask = 0;
		bool                                   pixel_active      = false;
		bool                                   tess_active       = false;
	};
	// Fills the draw's stage inputs from the registers, or returns false when it draws nothing.
	// With `ahead` (any thread) it also returns false for what needs the command thread.
	bool PrepareGraphicsStages(const HW::Context& context, const HW::UserConfig& user_config,
	                           const HW::Shader& shaders, DrawRenderState& state,
	                           GraphicsStageInputs& out, bool ahead);

	// A module translated again from a recipe, merged on the command thread.
	struct ReplayedModule {
		ProgramKey                         key;
		ShaderRecompiler::IR::ResourcePlan plan;
		ShaderModule                       module;
		std::vector<uint32_t>              spirv;
		// Recorded ids the module's id stands for, and its recipe if no recipe has that id.
		std::vector<uint64_t>              old_ids;
		std::vector<uint8_t>               recipe;
	};

	GraphicContext&                        m_graphics;
	ShaderRecompiler::IR::ImageHeapSource* m_image_heap;
	ShaderRecompiler::IR::HeapRowSource*   m_heap_rows = nullptr;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash> m_program_cache;
	// Modules compiled per stage.
	std::array<uint32_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> m_compiled {};
	ProgramKey                             m_lookup_key;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	std::atomic<bool>             m_driver_cache_dirty = false;
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
	template <typename InputInfo>
	const ShaderModule* GetProgramImpl(const ShaderParams& params, InputInfo& input_info,
	                               ShaderRecompiler::IR::PushData::Window& push_data);
	std::string CompileReason(const ProgramKey& key, bool known_program,
	                      const ShaderRecompiler::IR::ResourceSpecialization& spec,
	                          ShaderRecompiler::IR::PushData::Window push_data, bool source_changed);
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions       options,
	                            ShaderRecompiler::TranslateResult      translated,
	                            ShaderRecompiler::IR::PushData::Window push_data);
	void InitializeDriverCache();
	// Marks the driver cache changed and, once per interval, has the store thread write it.
	void CheckpointDriverCache();
	void WriteDriverCache() const;

	// pipelineSerialization.cpp
	struct LoadedRecord;
	void WarmUp(const std::filesystem::path& path);
	// Safe on any thread: checks, decompresses and decodes one framed record.
	void DecodeRecord(std::span<const uint8_t> record, LoadedRecord& loaded);
	void InsertRecord(LoadedRecord&& loaded);
	// On the command thread: moves a stored program's resource plan in, decoding its record if no
	// decoder did yet; false when the plan cannot be decoded (the program is translated again).
	bool TakeStoredPlan(Program& program);
	void StartStoreDecoders(std::vector<std::shared_ptr<StoredModule>>&& records);
	void StopStoreDecoders();
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
// "" by default, else a suffix naming the code-generation switches set in the environment
// (KYTY_F16_STORE_CLAMP and the like), which the store's file name carries.
std::string ShaderStoreVariant();
// Host capabilities that change the emitted SPIR-V.
ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics);

// Creates the layouts a pipeline of these stages (null where absent) has, which
// CreateHostPipeline reads.
void CreatePipelineLayouts(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                           std::span<const ShaderModule* const> stages);
// Safe on any thread. Stages are indexed like GraphicsPipelineKey::stage_ids.
vk::Pipeline CreateHostPipeline(GraphicContext& graphics, vk::PipelineLayout layout,
                                const GraphicsPipelineKey&                key,
                                const std::array<const ShaderModule*, 4>& stages,
                                vk::PipelineCache driver_cache, bool optimize = true);
// Without `optimize` the driver skips its optimizations (VK_PIPELINE_CREATE_DISABLE_OPTIMIZATION).
vk::Pipeline CreateHostPipeline(GraphicContext& graphics, vk::PipelineLayout layout,
                                const ShaderModule& module, vk::PipelineCache driver_cache,
                                bool optimize = true);
// Both of the above.
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
