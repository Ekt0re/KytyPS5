#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/barrierTracker.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/skipCatchUp.h"
#include "graphics/host_gpu/renderer/pipeline/skipTaint.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/renderer/upscaler/upscaler.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <map>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t              address       = 0;
	uint64_t              count_address = 0;
	uint32_t              count         = 0;
	uint32_t              stride        = 0;
	ShaderIndirectOffsets offsets;
};

[[nodiscard]] inline constexpr uint64_t IndirectDrawRecordsSize(uint32_t count, uint32_t stride,
                                                               uint32_t record_size) noexcept {
	return count == 0 ? 0 : static_cast<uint64_t>(count - 1) * stride + record_size;
}

[[nodiscard]] inline constexpr uint32_t ClampIndirectDrawCount(uint32_t count,
                                                              uint32_t max_count) noexcept {
	return count < max_count ? count : max_count;
}

[[nodiscard]] inline constexpr bool CanUseGpuIndirectDrawCount(uint64_t count_address,
                                                              uint32_t max_count,
                                                              uint32_t max_device_count) noexcept {
	return count_address != 0 && (count_address & 3u) == 0 && max_count != 0 &&
	       max_count <= max_device_count;
}

struct DrawIndexArgs {
	uint32_t         index_count         = 0;
	const void*      index_addr          = nullptr;
	uint32_t         instance_count      = 0;
	uint32_t         index_type_and_size = 0;
	int32_t          base_vertex         = 0;
	uint32_t         first_instance      = 0;
	DrawOffsetSource offset_source       = DrawOffsetSource::DrawState;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count   = 0;
	uint32_t         instance_count = 0;
	uint32_t         first_vertex   = 0;
	uint32_t         first_instance = 0;
	DrawOffsetSource offset_source  = DrawOffsetSource::DrawState;
	DrawIndirectArgs indirect;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const noexcept { return m_buffer == nullptr; }

	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;

	[[nodiscard]] vk::CommandBuffer Handle() const noexcept;
	[[nodiscard]] vk::CommandBuffer StateHandle() const noexcept { return m_buffer; }
	[[nodiscard]] vk::CommandBuffer DispatchHandle() const noexcept;
	[[nodiscard]] BarrierTracker*  GetBarrierTracker() const noexcept {
		return BarrierTrackerEnabled() ? &m_barriers : nullptr;
	}
	void OrderDispatches() const noexcept;
	[[nodiscard]] GraphicContext& GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&       m_context;
	GraphicContext&      m_graphics;
	vk::CommandBuffer    m_buffer = nullptr;
	mutable BarrierTracker m_barriers;
	mutable RenderState  m_render_state;
	mutable bool         m_rendering   = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

namespace Pssr {
struct Request;
}

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x, uint32_t thread_group_y,
	                    uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
	                      bool async_compute);
	// The command processor wrote guest memory: indirect arguments there are read again.
	void ForgetIndirectArgs(uint64_t vaddr, uint64_t size);
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	// ResolveTexture in two steps, so a caller can keep the first for unchanged descriptors.
	[[nodiscard]] PreparedTexture PrepareTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                             const ShaderRecompiler::IR::DescriptorValue& value);
	[[nodiscard]] TextureBinding  FinishTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            PreparedTexture                            prepared);

private:
	// The newest read of each indirect-argument address since the guest queue's last sync point,
	// from before that window's writes reached it: the counts when the CPU wrote them, or the
	// dispatch guard's slot the dispatch read them from.
	struct IndirectArgsRead {
		bool                        known = false;
		vk::DispatchIndirectCommand counts {};
		vk::Buffer                  buffer;
		uint64_t                    offset = 0;
		// Guest memory at the arguments then: a CPU write since makes a CPU read stale.
		vk::DispatchIndirectCommand guest {};
		uint64_t                    sequence = 0; // m_indirect_sequence of the read
	};
	struct IndirectArgsWindow {
		std::map<uint64_t, IndirectArgsRead> reads;
		uint64_t                             generation = 0;
	};
	// Per guest queue (BufferCache::GuestQueue).
	std::map<uint32_t, IndirectArgsWindow> m_indirect_args;
	uint64_t                               m_indirect_sequence = 0; // indirect dispatches so far

	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args);

	// One skipped dispatch, kept so it can be replayed when its program becomes available.
	struct CapturedDispatch {
		HW::ComputeShaderInfo cs_regs {};
		uint32_t              groups[3]     = {0, 0, 0};
		uint32_t              mode          = 0;
		bool                  async_compute = false;
		uint64_t              args_addr     = 0;
		bool                  is_indirect   = false;
	};

	// KYTY_SKIP_TAINT: taint what a skipped dispatch would have written and queue its replay.
	static uint64_t DispatchKey(const CapturedDispatch& dispatch);
	void NoteSkippedDispatch(const CapturedDispatch& dispatch,
	                         const std::vector<SkipTaint::Use>& uses);
	void DrainSkippedDispatches(CommandBuffer& buffer);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		PreparedBindings                pixel;
	};

	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages, DrawTargets& targets);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slot, bool ignore_target_mask = false,
	                              bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                    const DrawEmitInfo& emit, const DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                          DrawRenderState& state);
	void               ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                       const DrawRenderState& state, const DrawEmitInfo& emit,
	                                       const DrawIndexBufferSource& index_source);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);
	[[nodiscard]] bool        Upscale(vk::CommandBuffer command, const Pssr::Request& request,
	                                  Dlss::Inputs& inputs);

	// A null feature is one the upscaler refused or failed.
	struct UpscalerContext {
		uint64_t                           context;
		Dlss::Creation                     creation;
		std::unique_ptr<Upscaler::Feature> feature;
	};

	// GET_LOD_STATS counters since their last reset. With --lod-stats a bound T# with counting
	// enabled counts as sampled at its BASE_LEVEL, clamped when its MIN_LOD lies above that level.
	struct MipStatsCounter {
		uint8_t finest_level = 15; // no samples
		bool    clamped      = false;
		bool    unclamped    = false;
	};

	RenderContext&                        m_context;
	bool                                  m_lod_stats = Config::LodStatsEnabled();
	GraphicsBindings                     m_graphics_bindings;
	// The current draw's targets; PrepareDrawRenderState resolves them again for each draw.
	DrawTargets                           m_targets;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	vk::DescriptorSetLayout               m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pssr_pipeline          = nullptr;
	// KYTY_SKIP_TAINT: outputs of skipped dispatches and the work held for catch-up replay.
	SkipTaint                                  m_skip_taint;
	SkipCatchUp::Queue<CapturedDispatch>       m_skip_catchup;
	bool                                       m_replaying_catchup = false;
	std::vector<UpscalerContext>          m_upscaler_contexts;
	std::array<MipStatsCounter, 256>      m_mip_stats {};

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
