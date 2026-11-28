#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"

#include "common/assert.h"
#include "graphics/host_gpu/renderer/barrierTracker.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderBindings.h"

#include <cstring>

namespace Libs::Graphics {

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage) {
	switch (stage) {
		case ShaderType::Local:
		case ShaderType::Vertex: return vk::ShaderStageFlagBits::eVertex;
		case ShaderType::Mesh: return vk::ShaderStageFlagBits::eMeshEXT;
		case ShaderType::TessellationControl: return vk::ShaderStageFlagBits::eTessellationControl;
		case ShaderType::TessellationEvaluation:
			return vk::ShaderStageFlagBits::eTessellationEvaluation;
		case ShaderType::Pixel: return vk::ShaderStageFlagBits::eFragment;
		case ShaderType::Compute: return vk::ShaderStageFlagBits::eCompute;
		default: EXIT("unknown native shader stage\n");
	}
}

vk::PipelineStageFlags ShaderPipelineStages(vk::ShaderStageFlags stages) {
	vk::PipelineStageFlags result = {};
	if (stages & vk::ShaderStageFlagBits::eVertex) {
		result |= vk::PipelineStageFlagBits::eVertexShader;
	}
	if (stages & vk::ShaderStageFlagBits::eMeshEXT) {
		result |= vk::PipelineStageFlagBits::eMeshShaderEXT;
	}
	if (stages & vk::ShaderStageFlagBits::eTessellationControl) {
		result |= vk::PipelineStageFlagBits::eTessellationControlShader;
	}
	if (stages & vk::ShaderStageFlagBits::eTessellationEvaluation) {
		result |= vk::PipelineStageFlagBits::eTessellationEvaluationShader;
	}
	if (stages & vk::ShaderStageFlagBits::eFragment) {
		result |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (stages & vk::ShaderStageFlagBits::eCompute) {
		result |= vk::PipelineStageFlagBits::eComputeShader;
	}
	return result;
}

vk::MemoryBarrier MakeShaderWriteDependency() {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite |
	                        vk::AccessFlagBits::eVertexAttributeRead |
	                        vk::AccessFlagBits::eIndexRead | vk::AccessFlagBits::eUniformRead |
	                        vk::AccessFlagBits::eIndirectCommandRead |
	                        vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eColorAttachmentRead |
	                        vk::AccessFlagBits::eColorAttachmentWrite;
	return barrier;
}

vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer) {
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite |
	                        vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer;
	barrier.offset              = 0;
	barrier.size                = VK_WHOLE_SIZE;
	return barrier;
}

bool HasShaderBufferWrites(const ShaderStageRuntime& runtime) {
	const auto& program   = runtime.module->info;
	const auto& resources = *runtime.resources;
	EXIT_IF(resources.buffers.size() != program.info.buffers.size());
	// Address writes reach memory outside the bound buffer descriptors.
	bool has_writes = program.has_address_writes;
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		if (!program.info.buffers[i].written) {
			continue;
		}
		const auto&          value = resources.buffers[i];
		ShaderBufferResource descriptor;
		std::memcpy(descriptor.fields, value.dwords.data(), sizeof(descriptor.fields));
		// A zero stride means byte addressing. For either addressing mode a nonzero record
		// count is exactly the condition for a nonempty descriptor range.
		has_writes |= descriptor.Base48() != 0 && descriptor.NumRecords() != 0;
	}
	return has_writes;
}

namespace {

void RecordTrackedBarrier(vk::CommandBuffer vk_buffer, BarrierDecision barrier,
                          vk::PipelineStageFlags source_stages,
                          vk::PipelineStageFlags destination_stages,
                          vk::AccessFlags destination_access, bool writes_memory) {
	if (barrier == BarrierDecision::None) return;
	vk::MemoryBarrier memory {};
	switch (barrier) {
		case BarrierDecision::ComputeToCompute:
			memory.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
			memory.dstAccessMask = destination_access;
			source_stages      = vk::PipelineStageFlagBits::eComputeShader;
			destination_stages = vk::PipelineStageFlagBits::eComputeShader;
			break;
		case BarrierDecision::AllToCompute:
			memory.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			memory.dstAccessMask = destination_access;
			source_stages      = vk::PipelineStageFlagBits::eAllCommands;
			destination_stages = vk::PipelineStageFlagBits::eComputeShader;
			break;
		case BarrierDecision::ComputeToAll:
			if (!writes_memory) {
				vk_buffer.pipelineBarrier(source_stages, vk::PipelineStageFlagBits::eAllCommands,
				                          vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 0, nullptr);
				return;
			}
			memory.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite;
			memory.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			destination_stages   = vk::PipelineStageFlagBits::eAllCommands;
			break;
		case BarrierDecision::AllToAll:
			memory.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			memory.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			source_stages      = vk::PipelineStageFlagBits::eAllCommands;
			destination_stages = vk::PipelineStageFlagBits::eAllCommands;
			break;
		case BarrierDecision::None: return;
	}
	vk_buffer.pipelineBarrier(source_stages, destination_stages, vk::DependencyFlags {}, 1, &memory,
	                          0, nullptr, 0, nullptr);
}

} // namespace

void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages,
                         bool writes_memory, const ShaderBarrierOptions& options) {
	if (options.tracker == nullptr) {
		const vk::MemoryBarrier barrier {
		    .srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
		    .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite};
		vk_buffer.pipelineBarrier(source_stages, vk::PipelineStageFlagBits::eAllCommands,
		                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
		return;
	}
	switch (options.phase) {
		case ShaderBarrierPhase::AfterDispatch:
			EXIT_IF(options.accesses == nullptr);
			options.tracker->AfterDispatch(*options.accesses);
			return;
		case ShaderBarrierPhase::BeforeCommand:
			RecordTrackedBarrier(vk_buffer, options.tracker->BeforeCommand(), source_stages,
			                     vk::PipelineStageFlagBits::eAllCommands, options.destination_access,
			                     writes_memory);
			return;
		case ShaderBarrierPhase::BeforeBarrier:
			RecordTrackedBarrier(vk_buffer, options.tracker->BeforeBarrier(), source_stages,
			                     vk::PipelineStageFlagBits::eAllCommands, options.destination_access,
			                     writes_memory);
			return;
		case ShaderBarrierPhase::Flush:
			RecordTrackedBarrier(vk_buffer, options.tracker->Flush(), source_stages,
			                     vk::PipelineStageFlagBits::eAllCommands, options.destination_access,
			                     writes_memory);
			return;
	}
}

void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages,
                              const ShaderBarrierOptions& options) {
	if (options.tracker != nullptr) {
		EXIT_IF(options.accesses == nullptr);
		RecordTrackedBarrier(vk_buffer, options.tracker->BeforeDispatch(*options.accesses),
		                     vk::PipelineStageFlagBits::eAllCommands, destination_stages,
		                     options.destination_access, true);
		return;
	}
	const vk::MemoryBarrier barrier {
	    .srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	    .dstAccessMask = options.destination_access};
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, destination_stages,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages) {
	const auto barrier = MakeShaderWriteDependency();
	vk_buffer.pipelineBarrier(source_stages,
	                          vk::PipelineStageFlagBits::eComputeShader |
	                              vk::PipelineStageFlagBits::eAllGraphics |
	                              vk::PipelineStageFlagBits::eTransfer,
	                          vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
}

} // namespace Libs::Graphics
