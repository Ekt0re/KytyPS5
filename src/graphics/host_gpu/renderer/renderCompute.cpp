#include "common/alignment.h"
#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/readbackOrigin.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/frameHoles.h"
#include "common/logging/log.h"
#include "common/perfSummary.h"
#include "common/profiler.h"
#include "common/runtimeToggles.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/dispatchLimits.h"
#include "graphics/host_gpu/renderer/shaderTrace.h"
#include "gpu_tiler_shaders/dispatch_args_guard_spv.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/bufferProbe.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <span>
#include <string>
#include <chrono>
#include <cstdlib>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	const auto elements = descriptor.NumRecords();
	const bool covered = fill.bounded_elements != 0
	                         ? fill.bounded_elements == elements && invocations >= elements
	                         : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                          CommandBuffer& command, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!binding.image_id || !FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// CPU fills invalidate overlapping cached images. Preserve the native dispatch for
		// partial or ambiguous image backings, including pooled metadata aliases.
		if ((descriptor.Base48() & 3u) != 0 ||
		    !m_context.IsMapped(descriptor.Base48(), size)) {
			return false;
		}
		if (cache.IsRegionRegistered(descriptor.Base48(), size)) {
			// The native dispatch writes every byte of the range, after the GPU writes of the
			// images inside it, so those images take the filled data (after
			// middletooth-glitch/KytyPS5 98a6e4cd3e, there keyed to one Demon's Souls shader).
			cache.InvalidateMemoryFromGPU(descriptor.Base48(), size, true);
			return false;
		}
		// Metadata need not have an image backing yet. Keep proven fills in guest memory
		// when CPU-owned, so color-clear discovery can read them without a GPU download.
		m_context.GetBufferCache().FillBuffer(descriptor.Base48(), size, packed_clear, false);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.module->info.shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

// Global loads read pages without a buffer as zero. Cache the 32 KiB after each base the shader
// takes from user data or scalar loads before the dispatch reads it.
static void CacheDmaBases(RenderContext& context, const ShaderStageRuntime& stage) {
	const auto back = [&context](uint32_t low, uint32_t high) {
		const auto base = low | uint64_t {high} << 32u;
		for (uint64_t index = 0; index <= 2; ++index) {
			const auto page = base + index * BufferCache::CACHING_PAGESIZE;
			if (context.IsMapped(page, 1)) (void)context.GetBufferCache().FindBuffer(page, 1);
		}
	};
	const auto& info = stage.module->info.info;
	const auto& flat = stage.resources->flattened_srt;
	const auto& user = stage.resources->user_data;
	for (const auto [low, high]: info.address_bases) back(flat[low], flat[high]);
	for (const auto [low, high]: info.user_address_bases) back(user[low], user[high]);
}

// Recognized traversals trace the mirror of the TLAS whose header address the shader loads from
// scalar data.
static void PrepareRayQuery(RenderContext& context, const ShaderStageRuntime& stage,
                            PreparedBindings& bindings) {
	const auto& program = stage.module->info;
	const auto  dword   = program.bindings.ray_query_dword;
	if (dword == ShaderRecompiler::IR::PushData::NoStart) return;
	const auto& flat       = stage.resources->flattened_srt;
	const auto [low, high] = program.info.ray_query_tlas;
	const auto address = context.GetRayTracing()->Prepare(flat[low] | uint64_t {flat[high]} << 32u);
	bindings.shader_data[dword]     = static_cast<uint32_t>(address);
	bindings.shader_data[dword + 1] = static_cast<uint32_t>(address >> 32u);
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (input.stage.module->info.bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	if (indirect_args != 0) {
		const ReadbackOrigin::Scope origin(ReadbackOrigin::Origin::Args);
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

static bool HasStorageWrites(const ShaderComputeInputInfo& input_info,
                             const PreparedBindings&       bindings) {
	const auto& images = input_info.stage.module->info.info.images;
	return bindings.shared_memory.buffer != nullptr || HasShaderBufferWrites(input_info.stage) ||
	       std::ranges::any_of(images, [](const auto& image) {
		       return image.written &&
			          image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	       });
}

static ShaderAccessSet CollectDispatchAccesses(const ShaderComputeInputInfo& input_info,
                                                const PreparedBindings&       bindings) {
	const auto& program = input_info.stage.module->info;
	ShaderAccessSet accesses;
	const auto unknown = [&] {
		accesses.unknown_reads  = true;
		accesses.unknown_writes = true;
	};
	if (program.has_address_writes || program.info.uses_dma || program.info.uses_gds ||
	    program.info.uses_device_address || program.info.uses_heap_rows ||
	    program.info.uses_image_heap || program.info.function_scratch ||
	    bindings.gds.buffer != nullptr || bindings.shared_memory.buffer != nullptr ||
	    program.bindings.ray_query_dword != ShaderRecompiler::IR::PushData::NoStart) {
		unknown();
	}
	for (uint32_t i = 0; i < program.info.buffers.size(); ++i) {
		const auto& resource = program.info.buffers[i];
		const bool read      = resource.read || resource.atomic;
		const bool write     = resource.written || resource.atomic;
		if (!read && !write) continue;
		if (resource.descriptor_index == UINT32_MAX ||
		    resource.descriptor_index >= bindings.buffer_sources.size()) {
			if (read) accesses.unknown_reads = true;
			if (write) accesses.unknown_writes = true;
			continue;
		}
		const auto& source = bindings.buffer_sources[resource.descriptor_index];
		if (source.address == 0 || source.size == 0) continue;
		if (read) accesses.AddRead(source.address, source.size);
		if (write) accesses.AddWrite(source.address, source.size);
	}
	for (uint32_t i = 0; i < program.info.images.size(); ++i) {
		const auto& resource = program.info.images[i];
		const bool read      = resource.read || resource.atomic;
		const bool write     = resource.written || resource.atomic;
		if (!read && !write) continue;
		if (resource.heap || i >= bindings.images.size() || bindings.images[i].image_id == 0) {
			if (resource.heap || i >= bindings.images.size()) {
				if (read) accesses.unknown_reads = true;
				if (write) accesses.unknown_writes = true;
			}
			continue;
		}
		const auto& info = bindings.images[i].desc.info;
		const auto add   = [&](GuestRange range) {
			if (range.Empty()) return;
			if (read) accesses.AddRead(range.address, range.size);
			if (write) accesses.AddWrite(range.address, range.size);
		};
		add(info.data);
		add(info.stencil);
		add(info.metadata.range);
	}
	return accesses;
}

// Dispatch-argument guard (KYTY_DISPATCH_GUARD, on unless set to 0). Indirect dispatch counts a
// guest compute pass produced are copied by a one-invocation pass into a slot of a host-visible
// ring, which the dispatch reads instead: a count of 0 dispatches nothing (as on the console), and
// counts past the device limits or more than 2^24 workgroups in total, which only garbage from a
// mis-emulated writer gives, become an empty dispatch instead of an out-of-spec one that can keep
// the GPU busy until the device is lost (GTA VI, shader 0xe119767e79e995c9). The ring keeps the
// original counts, and refused dispatches are logged every 10 s.
class DispatchArgsGuard {
public:
	static constexpr uint32_t Slots     = 65536;
	static constexpr uint64_t SlotBytes = 32;
	static constexpr uint32_t MaxTotal  = 1u << 24u;

	struct Result {
		vk::Buffer buffer;
		uint64_t   offset;
		uint32_t   slot;
	};

	DispatchArgsGuard(GraphicContext& graphics, CommandScheduler& scheduler)
	    : m_graphics(graphics),
	      m_ring(graphics, scheduler, MemoryUsage::Download, 0, AllFlags, Slots * SlotBytes) {
		EXIT_IF(m_ring.Mapped().empty());
		std::memset(m_ring.Mapped().data(), 0xff, m_ring.Mapped().size());
		const vk::DescriptorSetLayoutBinding bindings[] {
		    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
		    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
		};
		vk::DescriptorSetLayoutCreateInfo layout_info {};
		layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
		layout_info.bindingCount = std::size(bindings);
		layout_info.pBindings    = bindings;
		RequireVulkanSuccess(graphics.device.createDescriptorSetLayout(&layout_info, nullptr,
		                                                               &m_set_layout),
		                     "create dispatch guard descriptor layout");
		const vk::PushConstantRange range {vk::ShaderStageFlagBits::eCompute, 0,
		                                   sizeof(PushConstants)};
		vk::PipelineLayoutCreateInfo pipeline_layout_info {};
		pipeline_layout_info.setLayoutCount         = 1;
		pipeline_layout_info.pSetLayouts            = &m_set_layout;
		pipeline_layout_info.pushConstantRangeCount = 1;
		pipeline_layout_info.pPushConstantRanges    = &range;
		RequireVulkanSuccess(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
		                                                          &m_layout),
		                     "create dispatch guard pipeline layout");
		m_pipeline = CreateComputePipeline(graphics.device, DISPATCH_ARGS_GUARD_SPV, m_layout,
		                                   "create dispatch guard pipeline");
		SetVulkanObjectNameF(graphics.device, m_pipeline, "Kyty.DispatchArgsGuard");
		const auto& limits = graphics.GetPhysicalDeviceProperties().limits;
		m_max[0]           = limits.maxComputeWorkGroupCount[0];
		m_max[1]           = limits.maxComputeWorkGroupCount[1];
		m_max[2]           = limits.maxComputeWorkGroupCount[2];
		m_alignment = std::max<uint64_t>(limits.minStorageBufferOffsetAlignment, 4);
	}

	static bool Enabled() {
		static const bool enabled = [] {
			const char* text = std::getenv("KYTY_DISPATCH_GUARD");
			return text == nullptr || text[0] != '0';
		}();
		return enabled;
	}

	// Records the guard pass for the arguments at `source` + `offset`; the dispatch reads the
	// returned slot.
	Result Record(vk::CommandBuffer command, vk::Buffer source, uint64_t offset,
	              uint64_t source_size, uint64_t shader) {
		Report();
		const uint32_t slot   = static_cast<uint32_t>(m_sequence % Slots);
		const uint64_t bound  = offset - offset % m_alignment;
		const uint64_t length = std::min<uint64_t>(source_size - bound, (offset - bound) + 12u);
		m_shaders[slot]       = shader;
		const PushConstants constants {.source_word = static_cast<uint32_t>((offset - bound) / 4u),
		                               .max_x       = m_max[0],
		                               .max_y       = m_max[1],
		                               .max_z       = m_max[2],
		                               .max_total   = MaxTotal,
		                               .sequence    = static_cast<uint32_t>(m_sequence)};
		m_sequence++;

		const vk::MemoryBarrier before {.srcAccessMask = vk::AccessFlagBits::eMemoryWrite,
		                                .dstAccessMask = vk::AccessFlagBits::eShaderRead};
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                        vk::PipelineStageFlagBits::eComputeShader, {}, 1, &before, 0,
		                        nullptr, 0, nullptr);
		const vk::DescriptorBufferInfo infos[] {
		    {source, bound, length},
		    {m_ring.Handle(), slot * SlotBytes, SlotBytes},
		};
		std::array<vk::WriteDescriptorSet, 2> writes {};
		for (uint32_t index = 0; index < writes.size(); ++index) {
			writes[index].dstBinding      = index;
			writes[index].descriptorCount = 1;
			writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
			writes[index].pBufferInfo     = &infos[index];
		}
		command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
		command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_layout, 0, writes);
		command.pushConstants(m_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(constants),
		                      &constants);
		command.dispatch(1, 1, 1);
		const vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eShaderWrite,
		                               .dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead |
		                                                vk::AccessFlagBits::eHostRead};
		command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
		                        vk::PipelineStageFlagBits::eDrawIndirect |
		                            vk::PipelineStageFlagBits::eHost,
		                        {}, 1, &after, 0, nullptr, 0, nullptr);
		return {m_ring.Handle(), slot * SlotBytes, slot};
	}

private:
	struct PushConstants {
		uint32_t source_word;
		uint32_t max_x;
		uint32_t max_y;
		uint32_t max_z;
		uint32_t max_total;
		uint32_t sequence;
	};

	[[nodiscard]] const uint32_t* SlotWords(uint32_t slot) const {
		return reinterpret_cast<const uint32_t*>(m_ring.Mapped().data() + slot * SlotBytes);
	}

	// Every 10 s: the dispatches the guard refused since the last report (from completed slots).
	void Report() {
		const auto now = std::chrono::steady_clock::now();
		if (now - m_last_report < std::chrono::seconds {10}) {
			return;
		}
		m_last_report = now;
		m_ring.Invalidate(0, m_ring.Size());
		uint64_t refused = 0;
		uint64_t empty   = 0;
		// Slots older than one ring lap were overwritten by later dispatches; skip them, or the
		// sequence check below never matches again and reporting stops for good.
		if (m_sequence - m_reported > Slots) {
			m_reported = m_sequence - Slots;
		}
		for (uint64_t sequence = m_reported; sequence < m_sequence; sequence++) {
			const auto  slot  = static_cast<uint32_t>(sequence % Slots);
			const auto* words = SlotWords(slot);
			if (words[7] != static_cast<uint32_t>(sequence)) {
				break; // not completed yet; report it next time
			}
			m_reported = sequence + 1;
			if (words[3] == 1u) {
				empty++;
			} else if (words[3] >= 2u) {
				Common::FrameHoles::NoteDispatchSkip(Common::FrameHoles::DispatchSkipReason::Refused);
				if (refused++ < 8) {
					std::printf("Dispatch guard: refused indirect dispatch of shader 0x%016llx with "
					            "GPU-written counts %ux%ux%u%s\n",
					            static_cast<unsigned long long>(m_shaders[slot]), words[4], words[5],
					            words[6], words[3] == 2u ? " (past device limits)"
					                                     : " (implausible total)");
				}
			}
		}
		if (refused != 0 || empty != 0) {
			std::printf("Dispatch guard: %llu GPU-written indirect dispatches were empty, %llu refused "
			            "in the last 10 s\n",
			            static_cast<unsigned long long>(empty),
			            static_cast<unsigned long long>(refused));
			std::fflush(stdout);
		}
	}

	GraphicContext&                       m_graphics;
	Buffer                                m_ring;
	vk::DescriptorSetLayout               m_set_layout = nullptr;
	vk::PipelineLayout                    m_layout     = nullptr;
	vk::Pipeline                          m_pipeline   = nullptr;
	uint32_t                              m_max[3] {};
	uint64_t                              m_alignment = 4;
	uint64_t                              m_sequence  = 0;
	uint64_t                              m_reported  = 0;
	std::array<uint64_t, Slots>           m_shaders {};
	std::chrono::steady_clock::time_point m_last_report = std::chrono::steady_clock::now();
};

// One per process (one device); never destroyed, as it would outlive the device at exit.
static DispatchArgsGuard& GetDispatchArgsGuard(RenderContext& context) {
	static auto* guard =
	    new DispatchArgsGuard(context.GetGraphics(), context.GetCommandScheduler());
	return *guard;
}

// KYTY_BUFFER_PROBE=1 compares the guest bytes of every bound storage range with the bytes the
// dispatch will read. The cache state is sampled before the bindings are final, and the copy is
// recorded after they are final: a difference on a range the cache believed synced is a missed
// upload. The copy is submitted and waited for before the dispatch is recorded, so findings are
// logged even when the dispatch itself hangs. The switch is off by default and the comparison is
// generic: no program identity or per-program selection is involved.
namespace {

struct BufferProbeBinding {
	uint32_t                index          = 0;
	uint64_t                guest          = 0;
	uint64_t                size           = 0;
	uint64_t                descriptor_size = 0; // What the guest descriptor declares.
	uint64_t                compared       = 0;
	uint64_t                staging_offset = 0;
	uint64_t                host_start     = 0;
	vk::Buffer              source {};
	bool                    written   = false;
	bool                    guest_ok  = false;
	BufferProbe::CacheState state     = BufferProbe::CacheState::Synced;
	std::vector<uint8_t>    guest_bytes;
};

class BufferProbeRunner {
public:
	static constexpr uint64_t MaxBytesPerBuffer   = 16ull * 1024 * 1024;
	static constexpr uint64_t MaxBytesPerDispatch = 64ull * 1024 * 1024;

	// Read once, off unless the variable is exactly "1".
	[[nodiscard]] static bool Enabled() noexcept {
		static const bool enabled = [] {
			const char* env = std::getenv("KYTY_BUFFER_PROBE");
			return env != nullptr && env[0] == '1' && env[1] == '\0';
		}();
		return enabled;
	}

	// Before the cache binds the buffers: remember what it believes about every bound range.
	static std::vector<BufferProbe::CacheState> SampleCacheState(RenderContext& context,
	                                                            const PreparedBindings& bindings) {
		auto& cache = context.GetBufferCache();
		std::vector<BufferProbe::CacheState> result(bindings.buffer_sources.size(),
		                                           BufferProbe::CacheState::Synced);
		for (size_t i = 0; i < bindings.buffer_sources.size(); i++) {
			const auto& source = bindings.buffer_sources[i];
			if (!GuestRange {source.address, source.size}.Valid()) continue;
			result[i] = BufferProbe::ClassifyCacheState(
			    cache.IsRegionCpuModified(source.address, source.size),
			    cache.IsRegionGpuModified(source.address, source.size));
		}
		return result;
	}

	void Record(RenderContext& context, CommandBuffer& command, const PreparedBindings& bindings,
	            const std::vector<BufferProbe::CacheState>& states) {
		const auto& program = bindings.runtime->module->info;
		auto&       cache   = context.GetBufferCache();
		const auto  stream  = cache.GetUtilityBuffer(MemoryUsage::Stream).Handle();

		std::vector<BufferProbeBinding> probed;
		std::vector<vk::BufferCopy>     copies;
		uint64_t                        total = 0;
		const auto&                     snapshot = *bindings.runtime->resources;
		for (size_t resource_index = 0; resource_index < program.info.buffers.size();
		     resource_index++) {
			const auto& resource = program.info.buffers[resource_index];
			const auto  index    = resource.descriptor_index;
			if (index == UINT32_MAX || index >= bindings.buffer_sources.size() ||
			    index >= bindings.buffers.size()) {
				continue;
			}
			const auto& source = bindings.buffer_sources[index];
			const auto& info   = bindings.buffers[index];
			if (!GuestRange {source.address, source.size}.Valid() || !info.buffer) continue;
			// The memory-offset adjustment the binding step packed next to the shader data.
			const auto& layout    = program.bindings;
			const auto  dword     = layout.memory_offset_dword + index / 4u;
			const auto  adjustment = dword < bindings.shader_data.size()
			                             ? (bindings.shader_data[dword] >> ((index % 4u) * 8u)) & 0xffu
			                             : 0u;

			BufferProbeBinding entry;
			entry.index           = index;
			entry.guest           = source.address;
			entry.size            = source.size;
			entry.written         = resource.written;
			entry.state           = index < states.size() ? states[index]
			                                             : BufferProbe::CacheState::Synced;
			entry.host_start      = info.offset + adjustment;
			entry.source          = info.buffer;
			entry.descriptor_size = resource_index < snapshot.buffers.size()
			                            ? DecodeNativeDescriptor<ShaderBufferResource>(
			                                  snapshot.buffers[resource_index])
			                                  .GetSize()
			                            : 0u;
			const uint64_t remaining =
			    total < MaxBytesPerDispatch ? MaxBytesPerDispatch - total : 0u;
			entry.compared = std::min({source.size, MaxBytesPerBuffer, remaining});
			if (entry.compared == 0) continue;
			entry.guest_bytes.resize(static_cast<size_t>(entry.compared));
			entry.guest_ok = LibKernel::Memory::TryReadSparseBacking(
			    entry.guest, entry.guest_bytes.data(), entry.compared);
			entry.staging_offset = total;
			copies.emplace_back(entry.host_start, total, entry.compared);
			total += Common::AlignUp(entry.compared, 16);
			probed.push_back(std::move(entry));
		}
		if (probed.empty()) return;

		auto       staging = std::make_unique<Buffer>(context.GetGraphics(),
		                                             context.GetCommandScheduler(),
		                                             MemoryUsage::Download, 0,
		                                             vk::BufferUsageFlagBits::eTransferDst, total);
		const auto native  = command.Handle();
		vk::MemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eShaderWrite |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
		                       nullptr);
		for (size_t i = 0; i < copies.size(); i++) {
			native.copyBuffer(probed[i].source, staging->Handle(), 1, &copies[i]);
		}
		vk::MemoryBarrier after {};
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands, {}, 1, &after, 0, nullptr, 0,
		                       nullptr);

		const uint32_t           dispatch = m_dispatch++;
		std::vector<std::string> lines    = FindSplitAliases(dispatch, probed, stream);
		auto&                    scheduler = context.GetCommandScheduler();
		// Debug option, correctness over speed: submit and wait before the dispatch exists.
		scheduler.Wait(scheduler.CurrentTick());
		Report(dispatch, total, probed, lines, *staging, stream);
	}

private:
	// Two bindings whose guest ranges overlap must see the same bytes. If they are not views of one
	// host range with the same offset-from-guest-address, the program's views diverge.
	static std::vector<std::string> FindSplitAliases(uint32_t                         dispatch,
	                                                 const std::vector<BufferProbeBinding>& probed,
	                                                 vk::Buffer                       stream) {
		std::vector<std::string> lines;
		for (size_t a = 0; a < probed.size(); a++) {
			for (size_t b = a + 1; b < probed.size(); b++) {
				const auto& x = probed[a];
				const auto& y = probed[b];
				if (x.guest >= y.guest + y.size || y.guest >= x.guest + x.size) continue;
				const bool  same_buffer = x.source == y.source;
				const bool  same_stream = x.source == stream && y.source == stream;
				const int64_t delta_x = static_cast<int64_t>(x.host_start) - static_cast<int64_t>(x.guest);
				const int64_t delta_y = static_cast<int64_t>(y.host_start) - static_cast<int64_t>(y.guest);
				if ((same_buffer || same_stream) && delta_x == delta_y) continue;
				char text[320];
				std::snprintf(text, sizeof(text),
				              "[buffer-probe] SPLIT-ALIAS dispatch=%u buf=%u (guest=0x%012" PRIx64
				              " size=0x%" PRIx64 " written=%d) and buf=%u (guest=0x%012" PRIx64
				              " size=0x%" PRIx64 " written=%d) overlap in guest memory but are bound to "
				              "different host ranges (%s): the program's views of the shared bytes diverge",
				              dispatch, x.index, x.guest, x.size, x.written ? 1 : 0, y.index, y.guest,
				              y.size, y.written ? 1 : 0,
				              same_buffer || same_stream ? "same buffer, different offsets"
				                                         : "different buffers");
				lines.emplace_back(text);
			}
		}
		return lines;
	}

	void Report(uint32_t dispatch, uint64_t total, const std::vector<BufferProbeBinding>& probed,
	            const std::vector<std::string>& alias_lines, Buffer& staging, vk::Buffer stream) {
		if (!staging.IsCoherent()) staging.Invalidate(0, total);
		const auto mapped = staging.Mapped();
		std::lock_guard lock(m_mutex);
		bool            mismatch   = false;
		bool            unexpected = false;
		std::vector<std::string> lines;
		for (const auto& entry: probed) {
			if (mapped.empty()) continue;
			BufferProbe::Comparison comparison;
			if (entry.guest_ok) {
				comparison = BufferProbe::Compare(entry.guest_bytes.data(),
				                                 mapped.data() + entry.staging_offset,
				                                 entry.compared);
			}
			mismatch   = mismatch || comparison.Differs();
			unexpected = unexpected || BufferProbe::IsUnexpectedDifference(comparison, entry.state);
			auto line = BufferProbe::FormatLine(dispatch, entry.index, entry.guest, entry.size,
			                                    entry.written, entry.state, entry.guest_ok, comparison);
			if (entry.source == stream) line += " src=stream";
			if (entry.compared < entry.size) line += " TRUNCATED";
			char declared[48];
			std::snprintf(declared, sizeof(declared), " descriptor_size=0x%" PRIx64 "%s",
			              entry.descriptor_size,
			              entry.descriptor_size != entry.size ? " CLAMPED" : "");
			line += declared;
			lines.push_back(std::move(line));
		}
		const bool aliased = !alias_lines.empty();
		if (m_limiter.OnDispatch(mismatch || aliased, unexpected || aliased)) {
			for (const auto& line: alias_lines) Log::WriteToConsoleAndLog((line + "\n").c_str());
			for (const auto& line: lines) Log::WriteToConsoleAndLog((line + "\n").c_str());
		}
		if (m_limiter.SummaryDue()) {
			Log::WriteToConsoleAndLog((m_limiter.Summary() + "\n").c_str());
		}
	}

	std::mutex           m_mutex;
	BufferProbe::Limiter m_limiter;
	uint32_t             m_dispatch = 0;
};

BufferProbeRunner& GetBufferProbe() {
	static BufferProbeRunner runner;
	return runner;
}

} // namespace

// Records the dispatch, at submission if its pipeline is still compiling.
template <typename Record>
static void RecordDispatch(CommandBuffer& buffer, const PipelineCache::Pipeline& pipeline,
                           const Record& record) {
	if (pipeline.pipeline) {
		record(buffer.DispatchHandle(), pipeline.pipeline);
		return;
	}
	if (buffer.GetBarrierTracker() != nullptr) {
		// A split would move the logical dispatch into another command buffer while its tracker
		// state stayed here. The opt-in tracker keeps the dispatch in this buffer instead.
		record(buffer.DispatchHandle(), pipeline.Pending().Get());
		return;
	}
	buffer.GetContext().GetCommandScheduler().Split(
	    [record, pending = pipeline.Pending()](vk::CommandBuffer command) {
		    record(command, pending.Get());
	    });
}

// Diagnostic (with KYTY_TOGGLE_FILE): names the compute programs that store through
// GPU-computed addresses, whose writes the CPU side never sees as GPU-owned. With the toggle
// log_addr=1 every such dispatch is printed with the address bases its SRT provides and the
// guest ranges of its written buffer bindings.
static void LogAddressWrites(const ShaderStageRuntime& stage, uint32_t x, uint32_t y, uint32_t z) {
	const auto& program = stage.module->info;
	if (!program.has_address_writes || !Common::RuntimeToggles::Enabled()) return;
	static std::mutex                   mutex;
	static std::map<uint64_t, uint32_t> counts;
	std::scoped_lock lock {mutex};
	const auto n   = ++counts[program.shader_hash];
	const bool all = Common::RuntimeToggles::Get("log_addr");
	if (!all && (n & (n - 1u)) != 0u) return;
	std::string text = fmt::format("Address-store dispatch: cs=0x{:016x} groups={}x{}x{} count {}",
	                               program.shader_hash, x, y, z, n);
	const auto& info = program.info;
	const auto& flat = stage.resources->flattened_srt;
	const auto& user = stage.resources->user_data;
	for (const auto [low, high]: info.address_bases) {
		if (low < flat.size() && high < flat.size())
			text += fmt::format(" base=0x{:x}", flat[low] | uint64_t {flat[high]} << 32u);
	}
	for (const auto [low, high]: info.user_address_bases) {
		if (low < user.size() && high < user.size())
			text += fmt::format(" ubase=0x{:x}", user[low] | uint64_t {user[high]} << 32u);
	}
	for (uint32_t i = 0; i < info.buffers.size() && i < stage.resources->buffers.size(); i++) {
		if (!info.buffers[i].written) continue;
		const auto v = DecodeNativeDescriptor<ShaderBufferResource>(stage.resources->buffers[i]);
		text += fmt::format(" wbuf=0x{:x}+0x{:x}", v.Base48(), v.GetSize());
	}
	std::printf("%s\n", text.c_str());
}

// KYTY_SKIP_TAINT: the guest buffers a compute program names through descriptors, with how it uses
// them. Memory reached through a pointer the shader computes itself is not listed.
static std::vector<SkipTaint::Use> ComputeBufferUses(const ShaderStageRuntime& stage) {
	std::vector<SkipTaint::Use> uses;
	if (stage.module == nullptr || stage.resources == nullptr) {
		return uses;
	}
	const auto& metadata = stage.module->info.info.buffers;
	const auto& snapshot = stage.resources->buffers;
	for (uint32_t i = 0; i < metadata.size() && i < snapshot.size(); i++) {
		if (metadata[i].descriptor_index == UINT32_MAX) continue;
		const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot[i]);
		auto       size       = ShaderRecompiler::IR::BufferBindingSize(metadata[i], descriptor);
		if (descriptor.NumRecords() == UINT32_MAX) size = std::min<uint64_t>(size, 256u << 20u);
		if (descriptor.Base48() == 0 || size == 0) continue;
		uses.push_back({descriptor.Base48(), size, metadata[i].read || !metadata[i].written,
		                metadata[i].written});
	}
	return uses;
}

static uint64_t SteadyNowMs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}

uint64_t RenderExecutor::DispatchKey(const CapturedDispatch& dispatch) {
	uint64_t h = dispatch.cs_regs.cs_regs.data_addr;
	const auto mix = [&h](uint64_t v) {
		h ^= v + 0x9e3779b9u + (h << 6u) + (h >> 2u);
	};
	mix((uint64_t {dispatch.groups[0]} << 32u) | dispatch.groups[1]);
	mix((uint64_t {dispatch.groups[2]} << 32u) | dispatch.mode);
	mix(dispatch.args_addr ^ (uint64_t {dispatch.async_compute ? 1u : 0u} << 1u) ^
	    uint64_t {dispatch.is_indirect ? 1u : 0u});
	for (uint32_t i = 0; i < 16u; ++i) {
		mix(dispatch.cs_regs.cs_user_sgpr.value[i]);
	}
	return h;
}

// Reinstates the captured compute registers on the command buffer for a catch-up replay.
static void RestoreCs(HW::Shader& shaders, const HW::ComputeShaderInfo& cs) {
	shaders.SetCsShader(cs.cs_regs);
	const auto count = std::min<size_t>(cs.cs_regs.user_sgpr, std::size(cs.cs_user_sgpr.value));
	for (size_t i = 0; i < count; i++) {
		shaders.SetCsUserSgpr(static_cast<uint32_t>(i), cs.cs_user_sgpr.value[i]);
	}
}

void RenderExecutor::NoteSkippedDispatch(const CapturedDispatch& dispatch,
                                         const std::vector<SkipTaint::Use>& uses) {
	if (!SkipTaintEnabled() || m_replaying_catchup) {
		return;
	}
	const auto now = SteadyNowMs();
	m_skip_taint.Taint(uses, now);
	m_skip_catchup.Enqueue(DispatchKey(dispatch), dispatch.cs_regs.cs_regs.data_addr, uses, dispatch,
	                       now);
}

// Probes held dispatches in submission order and replays the ones whose program can now be built.
// A ready consumer waits behind an earlier held producer that touches the same memory.
void RenderExecutor::DrainSkippedDispatches(CommandBuffer& buffer) {
	if (!SkipTaintEnabled() || m_replaying_catchup || m_skip_catchup.Empty()) {
		return;
	}
	m_replaying_catchup = true;
	const auto now = SteadyNowMs();
	auto ready = m_skip_catchup.TakeReady(now, [this](const auto& entry) {
		const auto&            cap   = entry.payload;
		ShaderComputeInputInfo probe {};
		probe.async_compute = cap.async_compute;
		if (cap.is_indirect) {
			probe.indirect_args_address = cap.args_addr;
		} else {
			const bool use_threads =
			    (cap.mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
			probe.dispatch_thread_dimensions = use_threads;
			for (uint32_t axis = 0; axis < 3u; ++axis) {
				probe.workgroup_counts[axis] = cap.groups[axis];
			}
			if (use_threads) {
				const uint32_t sizes[3] = {cap.cs_regs.cs_regs.num_thread_x,
				                          cap.cs_regs.cs_regs.num_thread_y,
				                          cap.cs_regs.cs_regs.num_thread_z};
				for (uint32_t axis = 0; axis < 3u; ++axis) {
					const auto size = std::max(sizes[axis], 1u);
					probe.workgroup_counts[axis] =
					    cap.groups[axis] / size + (cap.groups[axis] % size != 0u);
				}
			}
		}
		return m_context.GetPipelineCache().RetryComputeProgram(cap.cs_regs, probe)
		           ? SkipCatchUp::ProbeState::Ready
		           : SkipCatchUp::ProbeState::Pending;
	});
	const auto saved_cs = buffer.GetShaders().GetCs();
	for (const auto& entry: ready) {
		const auto& cap = entry.payload;
		RestoreCs(buffer.GetShaders(), cap.cs_regs);
		if (cap.is_indirect) {
			DispatchIndirect(buffer, cap.args_addr, cap.mode, cap.async_compute);
		} else {
			DispatchDirect(buffer, cap.groups[0], cap.groups[1], cap.groups[2], cap.mode,
			               cap.async_compute);
		}
		m_skip_catchup.DropByKey(entry.key);
	}
	RestoreCs(buffer.GetShaders(), saved_cs);
	m_replaying_catchup = false;
}

// A dispatch whose inputs are tainted is skipped, and its outputs become tainted in turn. A dispatch
// that runs clears the taint of what it wrote. `uses` is returned for the catch-up record.
static bool SkipOnTaintedInput(SkipTaint& taint, const ShaderStageRuntime& stage, uint64_t args_addr,
                               std::vector<SkipTaint::Use>& uses) {
	uses = ComputeBufferUses(stage);
	if (args_addr != 0) {
		uses.push_back({args_addr, sizeof(vk::DispatchIndirectCommand), true, false});
	}
	if (!SkipTaintEnabled() || taint.Empty()) {
		return false;
	}
	const auto now = SteadyNowMs();
	if (taint.ReadsTainted(uses, now)) {
		taint.Taint(uses, now);
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("[skip] compute dispatch reads a buffer a skipped dispatch was to write; "
			     "skipped too (tainted ranges=%zu)\n",
			     taint.Size());
		}
		return true;
	}
	taint.ClearWritten(uses);
	return false;
}

// A dispatch skipped for its group count would never run, so it is not queued for catch-up
// replay; with KYTY_SKIP_TAINT its outputs are tainted so the readers are skipped as well.
static void TaintLimitSkip(SkipTaint& taint, const std::vector<SkipTaint::Use>& uses) {
	if (SkipTaintEnabled()) {
		taint.Taint(uses, SteadyNowMs());
	}
}

// A dispatch whose group counts the guest chose: names it in the log when it is large and says
// whether it must be skipped because it is above the group limit. `args_addr` is the indirect
// argument address, 0 for a direct dispatch. Only counts known on the host reach this function;
// GPU-written arguments are judged by DispatchArgsGuard on the GPU.
static bool JudgeLargeDispatch(const ShaderStageRuntime& stage, const HW::ComputeShaderInfo& cs,
                               const uint32_t (&threads)[3], const uint32_t (&groups)[3],
                               uint32_t mode, uint64_t args_addr) {
	const uint64_t group_threads = uint64_t {std::max(threads[0], 1u)} *
	                               std::max(threads[1], 1u) * std::max(threads[2], 1u);
	const auto verdict = JudgeDispatch(groups[0], groups[1], groups[2], group_threads,
	                                   DispatchDefaultMaxGroups, true);
	if (verdict == DispatchVerdict::Run) return false;
	const uint64_t hash = stage.module->info.shader_hash;
	static DispatchReportLimiter limiter;
	if (limiter.Allow(hash)) {
		const auto& user  = cs.cs_user_sgpr.value;
		const int   count = FindCountDword(user, std::size(user), groups[0],
		                                   std::max(threads[0], 1u));
		char count_text[96] = "none";
		if (count >= 0) {
			std::snprintf(count_text, sizeof(count_text), "user[%d]=0x%08" PRIx32 " (as float %g)",
			              count, user[count], static_cast<double>(DwordAsFloat(user[count])));
		}
		LOGF("[dispatch] source=%s args=0x%012" PRIx64
		     " groups=0x%" PRIx32 ",0x%" PRIx32 ",0x%" PRIx32 " (%" PRIu64 " groups, %" PRIu64
		     " invocations) local=%ux%ux%u initiator=0x%08" PRIx32
		     " [thread_dims=%u partial=%u ordered_append=%u wave32=%u] count_from_dword=%s "
		     "action=%s limit=%" PRIu32 "\n",
		     args_addr != 0 ? "indirect" : "direct", args_addr, groups[0], groups[1], groups[2],
		     DispatchGroups(groups[0], groups[1], groups[2]),
		     DispatchInvocations(groups[0], groups[1], groups[2], group_threads), threads[0],
		     threads[1], threads[2], mode, (mode >> 5u) & 1u, (mode >> 1u) & 1u,
		     (mode >> 3u) & 1u, (mode >> 15u) & 1u, count_text,
		     verdict == DispatchVerdict::Skip ? "skipped" : "run", DispatchDefaultMaxGroups);
	}
	return verdict == DispatchVerdict::Skip;
}

void RenderExecutor::DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x,
                                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
                                    bool async_compute) {
	m_context.GetCommandScheduler().StartOperation();
	auto& sh_ctx = buffer.GetShaders();

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock(m_context.GetMutex());
	DrainSkippedDispatches(buffer);
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		Common::FrameHoles::NoteDispatchSkip(Common::FrameHoles::DispatchSkipReason::NullShader);
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS     = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	CapturedDispatch capture {};
	capture.cs_regs       = cs_regs;
	capture.groups[0]     = thread_group_x;
	capture.groups[1]     = thread_group_y;
	capture.groups[2]     = thread_group_z;
	capture.mode          = mode;
	capture.async_compute = async_compute;

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0]        = thread_group_x;
	input_info.workgroup_counts[1]        = thread_group_y;
	input_info.workgroup_counts[2]        = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	m_context.GetImageHeap().BeginDraw();
	if (!m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info)) {
		NoteSkippedDispatch(capture, ComputeBufferUses(input_info.stage));
		Common::FrameHoles::NoteDispatchSkip(Common::FrameHoles::DispatchSkipReason::FailedShader);
		return;
	}
	std::vector<SkipTaint::Use> uses;
	if (SkipOnTaintedInput(m_skip_taint, input_info.stage, 0, uses)) {
		NoteSkippedDispatch(capture, uses);
		Common::FrameHoles::NoteDispatchSkip(Common::FrameHoles::DispatchSkipReason::Tainted);
		return;
	}
	if (DispatchLimitsEnabled()) {
		uint32_t final_groups[3] = {thread_group_x, thread_group_y, thread_group_z};
		if (use_thread_dimensions) {
			std::copy(std::begin(input_info.workgroup_counts), std::end(input_info.workgroup_counts),
			          std::begin(final_groups));
		}
		if (JudgeLargeDispatch(input_info.stage, cs_regs, input_info.threads_num, final_groups, mode,
		                       0)) {
			TaintLimitSkip(m_skip_taint, uses);
			ResetBindings();
			return;
		}
	}
	if (!m_replaying_catchup && !m_skip_catchup.Empty()) {
		m_skip_catchup.DropByKey(DispatchKey(capture));
		m_skip_catchup.DropSuperseded(uses);
	}
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = input_info.stage.module->info;
	const auto& resources = *input_info.stage.resources;
	if (resources.specialization_reads.empty() &&
	    (TryConsumeComputeMetaClear(input_info, buffer) ||
	     TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y,
	                           thread_group_z, mode))) {
		ResetBindings();
		return;
	}

	if (use_thread_dimensions) {
		thread_group_x = input_info.workgroup_counts[0];
		thread_group_y = input_info.workgroup_counts[1];
		thread_group_z = input_info.workgroup_counts[2];
	}

	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, false);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	PreparedBindings* descriptor_stage = &bindings;
	CacheDmaBases(m_context, input_info.stage);
	FindBuffers(std::span {&descriptor_stage, 1u});
	const bool buffer_probe = BufferProbeRunner::Enabled();
	std::vector<BufferProbe::CacheState> probe_states;
	if (buffer_probe) probe_states = BufferProbeRunner::SampleCacheState(m_context, bindings);
	if (program.has_address_writes) {
		m_context.GetBufferCache().NoteAddressWrites();
	}
	if (program.info.uses_dma || program.info.uses_device_address) {
		m_context.PrepareBda();
	}
	if (program.info.uses_heap_rows) {
		m_context.RequestFaultScan();
	}
	PrepareRayQuery(m_context, input_info.stage, bindings);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings);
	RebindBuffers(bindings);
	if (buffer_probe) {
		GetBufferProbe().Record(m_context, buffer, bindings, probe_states);
	}

	const auto accesses = CollectDispatchAccesses(input_info, bindings);
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	auto vk_buffer = buffer.DispatchHandle();
	auto* tracker  = buffer.GetBarrierTracker();
	const ShaderBarrierOptions barrier_options {.tracker = tracker, .accesses = &accesses};
	if (tracker != nullptr || HasStorageWrites(input_info, bindings)) {
		// The legacy path keeps its write hazard. The tracker also orders read-only dispatches
		// after earlier writes and emits a barrier only for an actual range hazard.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader,
		                         barrier_options);
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const auto  max_mip = image.r128 ? r.LastLevel() : r.MaxMip();
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(max_mip) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	LogAddressWrites(input_info.stage, thread_group_x, thread_group_y, thread_group_z);
	if (const auto frame = m_context.GetGpu().GetFrameNum(); ShaderTrace::Active(frame)) {
		LOGF("TRACE f=%" PRIu64 " n=%u C cs=0x%016" PRIx64 " groups=%ux%ux%u local=%ux%ux%u "
		     "buffers=%zu images=%zu\n",
		     frame, ShaderTrace::NextSeq(), program.shader_hash, thread_group_x, thread_group_y,
		     thread_group_z, input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size());
	}
	auto* const    graphics = &m_context.GetGraphics();
	const uint64_t cs_hash  = program.shader_hash;
	const auto record = [=](vk::CommandBuffer command, vk::Pipeline handle) {
		command.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
		const auto crumb = GpuBreadcrumbs::Begin(*graphics, command, 'C', cs_hash);
		command.dispatch(thread_group_x, thread_group_y, thread_group_z);
		GpuBreadcrumbs::End(*graphics, command, crumb);
	};
	constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                                   std::max(input_info.threads_num[1], 1u) *
	                                   std::max(input_info.threads_num[2], 1u);
	if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
	    group_threads <= input_info.wave_size && !input_info.lds_storage) {
		vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute,
		                       PipelineCache::Resolve(pipeline, true).pipeline);
		// A single guest wave per group can run in launch order without a device-wide
		// spinlock. Keep the guest workgroup IDs when splitting the dispatch.
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
					}
					const auto crumb = GpuBreadcrumbs::Begin(*graphics, vk_buffer, 'C', cs_hash);
					vk_buffer.dispatchBase(x, y, z, 1, 1, 1);
					GpuBreadcrumbs::End(*graphics, vk_buffer, crumb);
				}
			}
		}
	} else {
		RecordDispatch(buffer, pipeline, record);
		vk_buffer = buffer.StateHandle();
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	Common::PerfSummary::Add(Common::PerfSummary::Counter::Dispatches);
	m_context.GetBufferCache().RecordPendingWrites();

	// The legacy path still orders read-only dispatches before later writers. The tracker keeps
	// the post-dispatch barrier pending until a later dispatch or command needs it.
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader, true,
	                    barrier_options);
	ResetBindings();
	m_context.GetBufferCache().FlushIfRequested();
}

// KYTY_INDIRECT_ARGS_WINDOW=0 reads indirect arguments from memory as recorded, as before.
static bool IndirectArgsWindowEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_INDIRECT_ARGS_WINDOW");
		return text == nullptr || text[0] != '0';
	}();
	return enabled;
}

// Every 10 s: how many indirect dispatches took an earlier read of their arguments.
static void NoteIndirectArgsReuse() {
	static uint64_t count       = 0;
	static auto     last_report = std::chrono::steady_clock::now();
	count++;
	const auto now = std::chrono::steady_clock::now();
	if (count == 1 || now - last_report >= std::chrono::seconds {10}) {
		std::printf("Indirect dispatch: %llu read their arguments as fetched before writes since the "
		            "last guest sync point\n",
		            static_cast<unsigned long long>(count));
		std::fflush(stdout);
		last_report = now;
	}
}

void RenderExecutor::ForgetIndirectArgs(uint64_t vaddr, uint64_t size) {
	// The write is in guest memory, which every queue reads.
	const uint64_t first = vaddr >= sizeof(vk::DispatchIndirectCommand)
	                           ? vaddr - sizeof(vk::DispatchIndirectCommand) + 1u
	                           : 0u;
	for (auto& [queue, window]: m_indirect_args) {
		auto it = window.reads.lower_bound(first);
		while (it != window.reads.end() && it->first < vaddr + size) {
			it = window.reads.erase(it);
		}
	}
}

void RenderExecutor::DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
                                      bool async_compute) {
	EXIT_IF(args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	m_context.GetCommandScheduler().StartOperation();
	Common::LockGuard lock(m_context.GetMutex());
	DrainSkippedDispatches(buffer);
	vk::DispatchIndirectCommand args {};
	auto&                       cache = m_context.GetBufferCache();
	if (m_indirect_sequence == 0) {
		// A fault reports the first write to a page only; later writes there do not fault.
		cache.SetCpuWriteListener([this](uint64_t vaddr, uint64_t size) {
			const auto begin = vaddr & ~uint64_t {0xffff};
			const auto end   = (vaddr + size + 0xffffu) & ~uint64_t {0xffff};
			ForgetIndirectArgs(begin, end - begin);
		});
	}
	const uint64_t sequence = m_indirect_sequence++;
	auto&          window   = m_indirect_args[cache.GuestQueue()];
	if (window.generation != cache.SyncGeneration() || window.reads.size() >= 1024u) {
		window.reads.clear();
		window.generation = cache.SyncGeneration();
	}
	// GPU work recorded since this queue's last sync point may write these arguments. The
	// console's command processor reads them when it reaches this packet, while that work can
	// still be running: GTA VI copies twice into one destination whose first bytes are both
	// copies' arguments (shader 0xe119767e79e995c9, async compute, no sync packet between), and
	// the second copy reads them before the first one's waves write. The newest read of the same
	// address from before the window's writes is what the console reads.
	const bool                  overwritten = cache.WrittenSinceGuestSync(args_addr, sizeof(args));
	const IndirectArgsRead*     earlier     = nullptr;
	vk::DispatchIndirectCommand guest {};
	const bool guest_known = LibKernel::Memory::TryReadBacking(args_addr, &guest, sizeof(guest));
	if (overwritten && guest_known && IndirectArgsWindowEnabled()) {
		const auto it = window.reads.find(args_addr);
		// A CPU read is stale once the CPU wrote other counts. A GPU read's guest bytes are stale
		// by nature; a CPU write over GPU-written memory faults and drops it (SetCpuWriteListener).
		// The guard's ring (65,536 slots, at most one per indirect dispatch) may reuse an old slot.
		if (it != window.reads.end() &&
		    (it->second.known ? std::memcmp(&it->second.guest, &guest, sizeof(guest)) == 0
		                      : sequence - it->second.sequence < 32768u)) {
			earlier = &it->second;
			NoteIndirectArgsReuse();
		}
	}
	// Known only when the CPU wrote the arguments; a compute pass can produce them instead.
	bool args_known = false;
	if (earlier != nullptr) {
		args_known = earlier->known;
		args       = earlier->counts;
	} else {
		args_known = !cache.HasGpuDirtyBytes(args_addr, sizeof(args)) &&
		             LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args));
	}
	if (args_known && (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		Common::FrameHoles::NoteDispatchSkip(Common::FrameHoles::DispatchSkipReason::NullShader);
		return;
	}
	CapturedDispatch capture {};
	capture.cs_regs       = cs_regs;
	capture.mode          = mode;
	capture.async_compute = async_compute;
	capture.args_addr     = args_addr;
	capture.is_indirect   = true;

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	input_info.indirect_args_address = args_addr;
	m_context.GetImageHeap().BeginDraw();
	if (!m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info)) {
		NoteSkippedDispatch(capture, ComputeBufferUses(input_info.stage));
		Common::FrameHoles::NoteDispatchSkip(Common::FrameHoles::DispatchSkipReason::FailedShader);
		return;
	}
	std::vector<SkipTaint::Use> uses;
	if (SkipOnTaintedInput(m_skip_taint, input_info.stage, args_addr, uses)) {
		NoteSkippedDispatch(capture, uses);
		Common::FrameHoles::NoteDispatchSkip(Common::FrameHoles::DispatchSkipReason::Tainted);
		return;
	}
	if (DispatchLimitsEnabled() && args_known) {
		const uint32_t final_groups[3] = {args.x, args.y, args.z};
		if (JudgeLargeDispatch(input_info.stage, cs_regs, input_info.threads_num, final_groups, mode,
		                       args_addr)) {
			TaintLimitSkip(m_skip_taint, uses);
			ResetBindings();
			return;
		}
	}
	if (!m_replaying_catchup && !m_skip_catchup.Empty()) {
		m_skip_catchup.DropByKey(DispatchKey(capture));
		m_skip_catchup.DropSuperseded(uses);
	}
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, false);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	CacheDmaBases(m_context, input_info.stage);
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = input_info.stage.module->info;
	const bool buffer_probe = BufferProbeRunner::Enabled();
	std::vector<BufferProbe::CacheState> probe_states;
	if (buffer_probe) probe_states = BufferProbeRunner::SampleCacheState(m_context, bindings);
	if (program.has_address_writes) {
		m_context.GetBufferCache().NoteAddressWrites();
	}
	if (program.info.uses_dma || program.info.uses_device_address) {
		m_context.PrepareBda();
	}
	if (program.info.uses_heap_rows) {
		m_context.RequestFaultScan();
	}
	PrepareRayQuery(m_context, input_info.stage, bindings);
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF((args_offset & 3u) != 0);
	DispatchArgsGuard::Result dispatch_args {args_buffer->Handle(), args_offset, UINT32_MAX};
	if (earlier != nullptr && !earlier->known) {
		// The guard slot the earlier dispatch read its arguments from.
		dispatch_args = {earlier->buffer, earlier->offset, UINT32_MAX};
	} else if (!args_known && DispatchArgsGuard::Enabled()) {
		buffer.EndRendering();
		dispatch_args = GetDispatchArgsGuard(m_context).Record(
		    buffer.Handle(), args_buffer->Handle(), args_offset, args_buffer->Size(),
		    program.shader_hash);
	}
	// The newest read before the window's writes reached these arguments: later dispatches in the
	// window reuse it. A GPU read is kept only when it went through the guard's slot; the guest
	// buffer itself changes under the window's writes.
	if (earlier == nullptr && !overwritten && guest_known) {
		if (args_known) {
			window.reads[args_addr] = {
			    .known = true, .counts = args, .guest = guest, .sequence = sequence};
		} else if (dispatch_args.slot != UINT32_MAX) {
			window.reads[args_addr] = {.buffer   = dispatch_args.buffer,
			                           .offset   = dispatch_args.offset,
			                           .guest    = guest,
			                           .sequence = sequence};
		} else {
			window.reads.erase(args_addr);
		}
	}
	const bool direct = earlier != nullptr && earlier->known;
	RebindBuffers(bindings);
	if (buffer_probe) {
		GetBufferProbe().Record(m_context, buffer, bindings, probe_states);
	}
	auto accesses = CollectDispatchAccesses(input_info, bindings);
	if (!direct) accesses.AddRead(args_addr, sizeof(vk::DispatchIndirectCommand));
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	auto vk_buffer = buffer.DispatchHandle();
	auto* tracker  = buffer.GetBarrierTracker();
	const ShaderBarrierOptions barrier_options {.tracker = tracker, .accesses = &accesses};
	if (tracker != nullptr || HasStorageWrites(input_info, bindings)) {
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader,
		                         barrier_options);
	}
	if (tracker != nullptr && !direct) {
		// Compute-to-compute does not cover the draw-indirect stage that fetches the counts.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eDrawIndirect,
		                         {.destination_access = vk::AccessFlagBits::eIndirectCommandRead});
	}
	LogAddressWrites(input_info.stage, 0, 0, 0);
	if (const auto frame = m_context.GetGpu().GetFrameNum(); ShaderTrace::Active(frame)) {
		LOGF("TRACE f=%" PRIu64 " n=%u C cs=0x%016" PRIx64 " indirect=%s groups=%ux%ux%u "
		     "local=%ux%ux%u buffers=%zu images=%zu\n",
		     frame, ShaderTrace::NextSeq(), program.shader_hash, args_known ? "known" : "gpu", args.x,
		     args.y, args.z, input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size());
	}
	RecordDispatch(buffer, pipeline,
	               [args = dispatch_args.buffer, offset = dispatch_args.offset, direct,
	                counts   = args, graphics = &m_context.GetGraphics(),
	                cs_hash  = program.shader_hash](vk::CommandBuffer command, vk::Pipeline handle) {
		               command.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
		               const auto crumb = GpuBreadcrumbs::Begin(*graphics, command, 'C', cs_hash);
		               if (direct) {
			               command.dispatch(counts.x, counts.y, counts.z);
		               } else {
			               command.dispatchIndirect(args, offset);
		               }
		               GpuBreadcrumbs::End(*graphics, command, crumb);
	               });
	vk_buffer = buffer.StateHandle();
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	Common::PerfSummary::Add(Common::PerfSummary::Counter::Dispatches);
	Common::PerfSummary::Add(Common::PerfSummary::Counter::IndirectDispatches);
	m_context.GetBufferCache().RecordPendingWrites();
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader, true,
	                    barrier_options);
	ResetBindings();
	m_context.GetBufferCache().FlushIfRequested();
}

} // namespace Libs::Graphics
