#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

namespace Libs::Graphics {

struct ShaderStageRuntime;
class BarrierTracker;
struct ShaderAccessSet;

enum class ShaderBarrierPhase : uint8_t { AfterDispatch, BeforeCommand, BeforeBarrier, Flush };

struct ShaderBarrierOptions {
	BarrierTracker*       tracker          = nullptr;
	const ShaderAccessSet* accesses        = nullptr;
	ShaderBarrierPhase    phase            = ShaderBarrierPhase::AfterDispatch;
	vk::AccessFlags       destination_access =
	    vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
};

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage);
vk::PipelineStageFlags  ShaderPipelineStages(vk::ShaderStageFlags stages);
vk::MemoryBarrier       MakeShaderWriteDependency();
vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer);
bool HasShaderBufferWrites(const ShaderStageRuntime& runtime);
void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages,
                         bool writes_memory = true, const ShaderBarrierOptions& options = {});
void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages,
                              const ShaderBarrierOptions& options = {});
void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
