// DescriptorHeap: a set layout larger than the standard pool gets its sets from dedicated pools
// (GTA VI's open world died at "!fill(fresh_batch)" on such a compute layout).
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace {
using namespace Libs::Graphics;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "DescriptorHeapTests:%d: %s\n", __LINE__, #condition);            \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

void TestSizes() {
	const std::array bindings = {
	    vk::DescriptorSetLayoutBinding {0, vk::DescriptorType::eStorageBuffer, 3,
	                                    vk::ShaderStageFlagBits::eCompute, nullptr},
	    vk::DescriptorSetLayoutBinding {1, vk::DescriptorType::eStorageImage, 1500,
	                                    vk::ShaderStageFlagBits::eCompute, nullptr},
	    vk::DescriptorSetLayoutBinding {2, vk::DescriptorType::eStorageBuffer, 2,
	                                    vk::ShaderStageFlagBits::eCompute, nullptr},
	    vk::DescriptorSetLayoutBinding {3, vk::DescriptorType::eSampler, 0,
	                                    vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	const auto sizes = DescriptorSetLayoutSizes(bindings);
	CHECK(sizes.size() == 2);
	CHECK(sizes[0].type == vk::DescriptorType::eStorageBuffer && sizes[0].descriptorCount == 5);
	CHECK(sizes[1].type == vk::DescriptorType::eStorageImage && sizes[1].descriptorCount == 1500);
	CHECK(!FitsStandardDescriptorPool(sizes));
	CHECK(DedicatedDescriptorPoolSets(sizes) == 10);
	CHECK(DescribeDescriptorSetSizes(sizes).find("StorageImage 1500/1024") != std::string::npos);

	const std::array small = {vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 1024},
	                          vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 4},
	                          vk::DescriptorPoolSize {vk::DescriptorType::eCombinedImageSampler, 8}};
	CHECK(FitsStandardDescriptorPool(small));
	const std::array huge = {vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 100000}};
	CHECK(DedicatedDescriptorPoolSets(huge) == 1);
	CHECK(!FitsStandardDescriptorPool(
	    std::array {vk::DescriptorPoolSize {vk::DescriptorType::eUniformBuffer, 2000}}));
}

struct Device {
	GraphicContext context;
	bool           ok = false;

	Device() {
		static vk::detail::DynamicLoader loader;
		const auto get = loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
		if (get == nullptr) return;
		VULKAN_HPP_DEFAULT_DISPATCHER.init(get);
		vk::ApplicationInfo app {};
		app.pApplicationName = "DescriptorHeapTests";
		app.apiVersion       = VK_API_VERSION_1_3;
		vk::InstanceCreateInfo instance_info {};
		instance_info.pApplicationInfo = &app;
		if (vk::createInstance(&instance_info, nullptr, &context.instance) != vk::Result::eSuccess) return;
		VULKAN_HPP_DEFAULT_DISPATCHER.init(context.instance);
		uint32_t count = 1;
		std::array<vk::PhysicalDevice, 1> physical {};
		const auto result = context.instance.enumeratePhysicalDevices(&count, physical.data());
		if ((result != vk::Result::eSuccess && result != vk::Result::eIncomplete) || count == 0) return;
		context.physical_device = physical[0];
		context.physical_device.getProperties(&context.physical_device_properties);

		const float           priority = 1.0f;
		vk::DeviceQueueCreateInfo queue {};
		queue.queueFamilyIndex = 0;
		queue.queueCount       = 1;
		queue.pQueuePriorities = &priority;
		vk::PhysicalDeviceVulkan12Features features12 {};
		features12.timelineSemaphore = true;
		vk::DeviceCreateInfo device_info {};
		device_info.pNext                = &features12;
		device_info.queueCreateInfoCount = 1;
		device_info.pQueueCreateInfos    = &queue;
		if (context.physical_device.createDevice(&device_info, nullptr, &context.device) !=
		    vk::Result::eSuccess) {
			return;
		}
		VULKAN_HPP_DEFAULT_DISPATCHER.init(context.device);
		ok = true;
	}
	~Device() {
		if (context.device) context.device.destroy();
		if (context.instance) context.instance.destroy();
	}
};

// One "submission": hands out a tick, marks it submitted and signals it from the host.
void Advance(MasterSemaphore& semaphore, GraphicContext& context) {
	const auto tick = semaphore.NextTick();
	semaphore.MarkSubmitted(tick);
	vk::SemaphoreSignalInfo signal {};
	signal.semaphore = semaphore.Handle();
	signal.value     = tick;
	CHECK(context.device.signalSemaphore(&signal) == vk::Result::eSuccess);
}

void TestAllocation() {
	Device device;
	if (!device.ok) {
		std::puts("DescriptorHeapTests: no Vulkan device, allocation test skipped");
		return;
	}
	auto&      context = device.context;
	const auto limits  = context.physical_device_properties.limits;
	// Above the standard pool's 1024 storage images and 8192 storage buffers, within the device
	// (24252 storage images: GTA VI CS 0xb3d5fc8e702cb6b6).
	const uint32_t images  = std::min(24252u, std::min(limits.maxDescriptorSetStorageImages,
	                                                  limits.maxPerStageDescriptorStorageImages));
	const uint32_t buffers = std::min(9000u, std::min(limits.maxDescriptorSetStorageBuffers,
	                                                  limits.maxPerStageDescriptorStorageBuffers));
	if (images <= 1024 && buffers <= 8192) {
		std::puts("DescriptorHeapTests: device limits below the standard pool, test skipped");
		return;
	}
	std::vector<vk::DescriptorSetLayoutBinding> bindings = {
	    {0, vk::DescriptorType::eStorageBuffer, buffers, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageImage, images, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eUniformBuffer, 2, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
	layout_info.pBindings    = bindings.data();
	vk::DescriptorSetLayout big = nullptr;
	CHECK(context.device.createDescriptorSetLayout(&layout_info, nullptr, &big) == vk::Result::eSuccess);
	const auto big_sizes = DescriptorSetLayoutSizes(bindings);
	CHECK(!FitsStandardDescriptorPool(big_sizes));

	const vk::DescriptorSetLayoutBinding small_binding {0, vk::DescriptorType::eStorageBuffer, 4,
	                                                    vk::ShaderStageFlagBits::eCompute, nullptr};
	layout_info.bindingCount        = 1;
	layout_info.pBindings           = &small_binding;
	vk::DescriptorSetLayout small = nullptr;
	CHECK(context.device.createDescriptorSetLayout(&layout_info, nullptr, &small) == vk::Result::eSuccess);
	const auto small_sizes = DescriptorSetLayoutSizes({&small_binding, 1});

	{
		MasterSemaphore semaphore(context);
		DescriptorHeap  heap(context, semaphore);
		// Many more sets than one dedicated pool holds: pools retire and come back by tick, and
		// the standard layout keeps allocating beside it.
		const auto per_pool = DedicatedDescriptorPoolSets(big_sizes);
		for (uint32_t frame = 0; frame < 20; frame++) {
			std::set<VkDescriptorSet> sets;
			for (uint32_t i = 0; i < per_pool * 2 + 1; i++) {
				const auto set = heap.Commit(big, big_sizes);
				CHECK(set != nullptr);
				// Sets of one frame are all in flight together: none is handed out twice.
				CHECK(sets.insert(static_cast<VkDescriptorSet>(set)).second);
				CHECK(heap.Commit(small, small_sizes) != nullptr);
			}
			Advance(semaphore, context);
		}
		// A set whose pool is still in flight is never handed out again before the GPU passes it.
		std::set<VkDescriptorSet> unsignaled;
		for (uint32_t i = 0; i < per_pool * 3; i++) {
			const auto set = heap.Commit(big, big_sizes);
			CHECK(unsignaled.insert(static_cast<VkDescriptorSet>(set)).second);
		}
		Advance(semaphore, context);
	}
	context.device.destroyDescriptorSetLayout(big, nullptr);
	context.device.destroyDescriptorSetLayout(small, nullptr);
	std::printf("DescriptorHeapTests: %u storage buffers + %u storage images per set allocated\n",
	            buffers, images);
}

} // namespace

int main() {
	TestSizes();
	TestAllocation();
	std::puts("Descriptor heap tests passed");
}
