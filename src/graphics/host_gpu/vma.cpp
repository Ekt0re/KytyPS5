#include "graphics/host_gpu/renderer/bdaGuard.h"
#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/memAccount.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Graphics {

namespace {

// KYTY_ASYNC_IMAGE_DESTROY=1 sends deleted images to a worker thread that destroys them in
// batches, instead of destroying them on the calling thread: releasing the memory of a large
// render target takes 0.2 to 0.4 ms in the driver, on the thread that records the draws. Off by
// default: GTA VI deletes few images (about 8 a second in the world, under 0.1 ms/s).
bool AsyncImageDestroy() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_ASYNC_IMAGE_DESTROY");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

class ImageGraveyard final {
public:
	ImageGraveyard() = default;
	~ImageGraveyard() {
		Flush();
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_one();
		if (m_thread.joinable()) {
			m_thread.join();
		}
	}
	ImageGraveyard(const ImageGraveyard&)            = delete;
	ImageGraveyard& operator=(const ImageGraveyard&) = delete;

	void Push(VmaAllocator allocator, VkImage image, VmaAllocation allocation) {
		{
			std::lock_guard lock(m_mutex);
			if (!m_thread.joinable()) {
				m_thread = std::thread([this] { Run(); });
			}
			m_items.push_back({allocator, image, allocation});
			m_used = true;
		}
		m_wake.notify_one();
	}

	// Returns once every image pushed so far is destroyed; whether any image was ever pushed (the
	// worker may have finished one between a failed allocation and this call).
	bool Flush() {
		std::unique_lock lock(m_mutex);
		m_idle.wait(lock, [this] { return m_items.empty() && !m_busy; });
		return m_used;
	}

private:
	struct Item {
		VmaAllocator  allocator;
		VkImage       image;
		VmaAllocation allocation;
	};

	void Run() {
		std::vector<Item> batch;
		std::unique_lock  lock(m_mutex);
		for (;;) {
			m_wake.wait(lock, [this] { return m_stop || !m_items.empty(); });
			if (m_items.empty()) {
				return;
			}
			batch.swap(m_items);
			m_busy = true;
			lock.unlock();
			for (const auto& item: batch) {
				vmaDestroyImage(item.allocator, item.image, item.allocation);
			}
			batch.clear();
			lock.lock();
			m_busy = false;
			m_idle.notify_all();
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_wake;
	std::condition_variable m_idle;
	std::vector<Item>       m_items;
	bool                    m_busy = false;
	bool                    m_stop = false;
	bool                    m_used = false;
	std::thread             m_thread;
};

ImageGraveyard& Graveyard() {
	static ImageGraveyard graveyard;
	return graveyard;
}

} // namespace

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}
	// 64 MiB blocks: allocations above 32 MiB get memory of their own (AnyPS5 e8332d72).
	info.preferredLargeHeapBlockSize = 64ull * 1024 * 1024;

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	Common::MemAccount::SetExternal("vk", [this] {
		if (allocator == nullptr) return std::string("none");
		VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
		vmaGetHeapBudgets(allocator, budgets);
		uint64_t device_bytes = 0;
		uint64_t host_bytes   = 0;
		for (uint32_t i = 0; i < physical_device_memory_properties.memoryHeapCount; i++) {
			const bool dev = static_cast<bool>(
			    physical_device_memory_properties.memoryHeaps[i].flags &
			    vk::MemoryHeapFlagBits::eDeviceLocal);
			(dev ? device_bytes : host_bytes) += budgets[i].usage;
		}
		return fmt::format("device={:.0f}MiB host={:.0f}MiB",
		                   static_cast<double>(device_bytes) / 1048576.0,
		                   static_cast<double>(host_bytes) / 1048576.0);
	});
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	Common::MemAccount::ClearExternal("vk");
	BdaGuard::Shutdown(); // Buffers KYTY_BDA_GUARD=2 kept alive.
	Graveyard().Flush();
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = physical_device_memory_properties;
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	const auto        create       = [&] {
		return static_cast<vk::Result>(
		    vmaCreateImage(allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
		                   &alloc_info, &native_image, &image.allocation, nullptr));
	};
	auto result = create();
	if (result != vk::Result::eSuccess && Graveyard().Flush()) {
		// Images the worker destroyed (or still held) may have freed memory since: try again.
		result = create();
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
	}

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	if (AsyncImageDestroy()) {
		Graveyard().Push(allocator, image.image, image.allocation);
	} else {
		vmaDestroyImage(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
}

} // namespace Libs::Graphics
