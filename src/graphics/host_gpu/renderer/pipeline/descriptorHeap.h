#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <deque>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

// The descriptors of each type one set of a layout holds, summed over its bindings (one entry per
// type, in first-binding order).
using DescriptorSetSizes = std::vector<vk::DescriptorPoolSize>;

[[nodiscard]] DescriptorSetSizes DescriptorSetLayoutSizes(
    std::span<const vk::DescriptorSetLayoutBinding> bindings);
// True when one set fits a standard pool of the heap; other layouts get a dedicated pool.
[[nodiscard]] bool FitsStandardDescriptorPool(std::span<const vk::DescriptorPoolSize> sizes);
// The sets a dedicated pool of a layout with these sizes holds (at least 1).
[[nodiscard]] uint32_t DedicatedDescriptorPoolSets(std::span<const vk::DescriptorPoolSize> sizes);
// "storage-image 2048, sampler 12" for logs.
[[nodiscard]] std::string DescribeDescriptorSetSizes(std::span<const vk::DescriptorPoolSize> sizes);

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	// `sizes` are the layout's DescriptorSetLayoutSizes. A layout that does not fit a standard
	// pool takes its sets from pools of its own, sized from `sizes`.
	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout                 layout,
	                                       std::span<const vk::DescriptorPoolSize> sizes);

private:
	static constexpr uint32_t DescriptorSetBatch = 32;

	struct Batch {
		std::array<vk::DescriptorSet, DescriptorSetBatch> sets {};
		uint32_t                                          size = 0;
	};

	// Pools of one oversized layout: each holds `sets` sets and is allocated whole at once, then
	// retired with the tick of its last set and reset once the GPU has passed it.
	struct DedicatedPools {
		DescriptorSetSizes                                  sizes;
		uint32_t                                            sets = 1;
		vk::DescriptorPool                                  current = nullptr;
		std::deque<std::pair<vk::DescriptorPool, uint64_t>> pending;
		Batch                                               batch;
	};

	void              CreateDescriptorPool();
	vk::DescriptorSet CommitDedicated(vk::DescriptorSetLayout                 layout,
	                                  std::span<const vk::DescriptorPoolSize> sizes);
	// Moves the retired pools of a layout whose handle was destroyed to m_retired_dedicated.
	void              RetireDedicated(DedicatedPools& pools);
	// Destroys the retired pools the GPU has passed (all of them, waiting, with `wait`).
	void              DestroyRetired(bool wait);

	GraphicContext&                                     m_graphics;
	MasterSemaphore&                                    m_master_semaphore;
	vk::DescriptorPool                                  m_current_pool = nullptr;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>> m_pending_pools;
	std::unordered_map<vk::DescriptorSetLayout, Batch>  m_sets;
	std::unordered_map<vk::DescriptorSetLayout, DedicatedPools> m_dedicated;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>>         m_retired_dedicated;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
