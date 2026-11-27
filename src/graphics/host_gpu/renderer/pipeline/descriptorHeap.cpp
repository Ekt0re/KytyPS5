#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include <algorithm>

namespace Libs::Graphics {
namespace {

constexpr uint32_t DescriptorHeapCount = 1024;
// Every descriptor type a pool can hold without extension structures. The recompiler's layouts use
// storage buffers, sampled and storage images and samplers; the rest keep a layout of any other
// basic type from needing a dedicated pool.
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eUniformTexelBuffer, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageTexelBuffer, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eCombinedImageSampler, 1024},
    vk::DescriptorPoolSize {vk::DescriptorType::eInputAttachment, 256},
    vk::DescriptorPoolSize {vk::DescriptorType::eUniformBufferDynamic, 256},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBufferDynamic, 256},
};
// A dedicated pool holds as many sets as fit this many descriptors of its largest type.
constexpr uint32_t DedicatedPoolDescriptors = 16384;

uint32_t StandardPoolCount(vk::DescriptorType type) {
	for (const auto& size: DescriptorPoolSizes) {
		if (size.type == type) return size.descriptorCount;
	}
	return 0;
}

} // namespace

DescriptorSetSizes DescriptorSetLayoutSizes(std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	DescriptorSetSizes sizes;
	for (const auto& binding: bindings) {
		if (binding.descriptorCount == 0) continue;
		auto iter = std::ranges::find(sizes, binding.descriptorType, &vk::DescriptorPoolSize::type);
		if (iter == sizes.end()) {
			sizes.push_back({binding.descriptorType, 0});
			iter = sizes.end() - 1;
		}
		iter->descriptorCount += binding.descriptorCount;
	}
	return sizes;
}

bool FitsStandardDescriptorPool(std::span<const vk::DescriptorPoolSize> sizes) {
	return std::ranges::all_of(sizes, [](const vk::DescriptorPoolSize& size) {
		return size.descriptorCount <= StandardPoolCount(size.type);
	});
}

uint32_t DedicatedDescriptorPoolSets(std::span<const vk::DescriptorPoolSize> sizes) {
	uint32_t largest = 1;
	for (const auto& size: sizes) {
		largest = std::max(largest, size.descriptorCount);
	}
	return std::clamp(DedicatedPoolDescriptors / largest, 1u, 32u);
}

std::string DescribeDescriptorSetSizes(std::span<const vk::DescriptorPoolSize> sizes) {
	std::string text;
	for (const auto& size: sizes) {
		if (!text.empty()) text += ", ";
		text += fmt::format("{} {}/{}", vk::to_string(size.type), size.descriptorCount,
		                    StandardPoolCount(size.type));
	}
	return text;
}

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
	for (auto& [layout, pools]: m_dedicated) {
		m_graphics.device.destroyDescriptorPool(pools.current, nullptr);
		RetireDedicated(pools);
	}
	DestroyRetired(true);
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout                 layout,
                                         std::span<const vk::DescriptorPoolSize> sizes) {
	KYTY_PROFILER_FUNCTION();
	if (!FitsStandardDescriptorPool(sizes)) {
		return CommitDedicated(layout, sizes);
	}
	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}

	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = m_current_pool;
	allocate.descriptorSetCount = DescriptorSetBatch;
	allocate.pSetLayouts        = layouts.data();
	const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
	if (result == vk::Result::eSuccess) {
		batch.size = DescriptorSetBatch;
		return batch.sets[--batch.size];
	}
	EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
	        result != vk::Result::eErrorFragmentedPool);

	// A failed multi-set allocation can leave a pool unable to allocate even one set until it is
	// reset, so the rest of this pool and a fresh one are filled one set per call.
	const auto fill = [&](Batch& target) {
		while (target.size < DescriptorSetBatch &&
		       m_graphics.device.allocateDescriptorSets(&allocate, &target.sets[target.size]) ==
		           vk::Result::eSuccess) {
			target.size++;
		}
		return target.size != 0;
	};
	allocate.descriptorSetCount = 1;
	if (fill(batch)) {
		return batch.sets[--batch.size];
	}

	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
	m_master_semaphore.Refresh();
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear();
	auto& fresh_batch       = m_sets[layout];
	allocate.descriptorPool = m_current_pool;
	// The layout fits a standard pool (checked above), so an empty pool always holds one set.
	if (!fill(fresh_batch)) {
		EXIT("a fresh descriptor pool cannot allocate a set of layout %s\n",
		     DescribeDescriptorSetSizes(sizes).c_str());
	}
	return fresh_batch.sets[--fresh_batch.size];
}

vk::DescriptorSet DescriptorHeap::CommitDedicated(vk::DescriptorSetLayout                 layout,
                                                  std::span<const vk::DescriptorPoolSize> sizes) {
	auto [iter, inserted] = m_dedicated.try_emplace(layout);
	auto& pools           = iter->second;
	if (!inserted && !std::ranges::equal(pools.sizes, sizes)) {
		// A destroyed layout's handle came back for another layout.
		if (pools.current != nullptr) {
			pools.pending.emplace_back(pools.current, m_master_semaphore.CurrentTick());
		}
		RetireDedicated(pools);
		pools    = {};
		inserted = true;
	}
	if (inserted) {
		pools.sizes.assign(sizes.begin(), sizes.end());
		pools.sets = DedicatedDescriptorPoolSets(sizes);
		Log::WriteToConsoleAndLog(fmt::format(
		    "Descriptor heap: a set layout exceeds the standard pool ({}); its sets come from "
		    "dedicated pools of {} sets\n",
		    DescribeDescriptorSetSizes(sizes), pools.sets));
	}
	if (pools.batch.size != 0) {
		return pools.batch.sets[--pools.batch.size];
	}

	if (pools.current != nullptr) {
		pools.pending.emplace_back(pools.current, m_master_semaphore.CurrentTick());
		pools.current = nullptr;
		m_master_semaphore.Refresh();
		if (const auto& [pool, tick] = pools.pending.front(); m_master_semaphore.IsFree(tick)) {
			pools.current = pool;
			pools.pending.pop_front();
			EXIT_IF(m_graphics.device.resetDescriptorPool(pools.current, {}) !=
			        vk::Result::eSuccess);
		}
	}
	if (pools.current == nullptr) {
		DestroyRetired(false);
		auto pool_sizes = pools.sizes;
		for (auto& size: pool_sizes) {
			size.descriptorCount *= pools.sets;
		}
		vk::DescriptorPoolCreateInfo create {};
		create.maxSets       = pools.sets;
		create.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
		create.pPoolSizes    = pool_sizes.data();
		RequireVulkanSuccess(m_graphics.device.createDescriptorPool(&create, nullptr, &pools.current),
		                     "vkCreateDescriptorPool (dedicated)");
	}

	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = pools.current;
	allocate.descriptorSetCount = pools.sets;
	allocate.pSetLayouts        = layouts.data();
	RequireVulkanSuccess(m_graphics.device.allocateDescriptorSets(&allocate, pools.batch.sets.data()),
	                     "vkAllocateDescriptorSets (dedicated pool)");
	pools.batch.size = pools.sets;
	return pools.batch.sets[--pools.batch.size];
}

void DescriptorHeap::RetireDedicated(DedicatedPools& pools) {
	for (const auto& entry: pools.pending) {
		m_retired_dedicated.push_back(entry);
	}
	pools.pending.clear();
}

void DescriptorHeap::DestroyRetired(bool wait) {
	if (!wait) m_master_semaphore.Refresh();
	while (!m_retired_dedicated.empty()) {
		const auto [pool, tick] = m_retired_dedicated.front();
		if (wait) {
			m_master_semaphore.Wait(tick);
		} else if (!m_master_semaphore.IsFree(tick)) {
			break;
		}
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
		m_retired_dedicated.pop_front();
	}
}

void DescriptorHeap::CreateDescriptorPool() {
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount;
	create.poolSizeCount = static_cast<uint32_t>(DescriptorPoolSizes.size());
	create.pPoolSizes    = DescriptorPoolSizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
}

} // namespace Libs::Graphics
