#include "kernel/vulkanGuestBacking.h"

#include "common/emulatorConfig.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
#define KYTY_VULKAN_GUEST_BACKING 1
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <unistd.h>
#include <vector>
#endif

namespace Libs::LibKernel::Memory {

int VulkanGuestBackingMode() {
	static const int mode = [] {
		const char* text = std::getenv("KYTY_VK_GUEST_BACKING");
		return text != nullptr ? std::atoi(text) : 0;
	}();
	return mode;
}

#if defined(KYTY_VULKAN_GUEST_BACKING)

namespace {

void Log(const char* format, const char* detail) {
	std::printf("Vulkan guest backing: ");
	std::printf(format, detail);
	std::printf("\n");
	std::fflush(stdout);
}

uint64_t ReadSysValue(const std::string& path) {
	uint64_t value = 0;
	if (FILE* file = std::fopen(path.c_str(), "r"); file != nullptr) {
		if (std::fscanf(file, "%lu", &value) != 1) {
			value = 0;
		}
		std::fclose(file);
	}
	return value;
}

uint64_t MemAvailable() {
	uint64_t kb   = 0;
	FILE*    file = std::fopen("/proc/meminfo", "r");
	if (file == nullptr) {
		return UINT64_MAX;
	}
	char line[256];
	while (std::fgets(line, sizeof(line), file) != nullptr) {
		if (std::sscanf(line, "MemAvailable: %lu kB", &kb) == 1) {
			break;
		}
	}
	std::fclose(file);
	return kb != 0 ? kb * 1024u : UINT64_MAX;
}

} // namespace

struct VulkanGuestAllocator::Impl {
	void*                     library  = nullptr;
	VkInstance                instance = VK_NULL_HANDLE;
	VkDevice                  device   = VK_NULL_HANDLE;
	PFN_vkDestroyInstance     destroy_instance = nullptr;
	PFN_vkDestroyDevice       destroy_device   = nullptr;
	PFN_vkAllocateMemory      allocate_memory  = nullptr;
	PFN_vkFreeMemory          free_memory      = nullptr;
	PFN_vkGetMemoryFdKHR      get_memory_fd    = nullptr;
	Info                      info;
	uint64_t                  max_allocation = 0;
	// sysfs directory of the PCI device (amdgpu's mem_info_gtt_*), empty when unknown.
	std::string               sysfs;

	~Impl() {
		if (device != VK_NULL_HANDLE) {
			destroy_device(device, nullptr);
		}
		if (instance != VK_NULL_HANDLE) {
			destroy_instance(instance, nullptr);
		}
		if (library != nullptr) {
			dlclose(library);
		}
	}
};

VulkanGuestAllocator* VulkanGuestAllocator::Create() {
	const int mode = VulkanGuestBackingMode();
	if (mode <= 0) {
		return nullptr;
	}
	auto  impl = std::make_unique<Impl>();
	auto* lib  = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	if (lib == nullptr) {
		Log("no libvulkan.so.1 (%s); memfd backing", dlerror());
		return nullptr;
	}
	impl->library = lib;
	auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
	if (gipa == nullptr) {
		Log("%s; memfd backing", "no vkGetInstanceProcAddr");
		return nullptr;
	}
	auto create_instance =
	    reinterpret_cast<PFN_vkCreateInstance>(gipa(VK_NULL_HANDLE, "vkCreateInstance"));
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "Kyty guest backing";
	app.apiVersion       = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	if (create_instance == nullptr ||
	    create_instance(&instance_info, nullptr, &impl->instance) != VK_SUCCESS) {
		impl->instance = VK_NULL_HANDLE;
		Log("%s; memfd backing", "vkCreateInstance failed");
		return nullptr;
	}
	const auto instance = impl->instance;
#define KYTY_LOAD(type, name) auto name = reinterpret_cast<type>(gipa(instance, #name))
	KYTY_LOAD(PFN_vkDestroyInstance, vkDestroyInstance);
	KYTY_LOAD(PFN_vkEnumeratePhysicalDevices, vkEnumeratePhysicalDevices);
	KYTY_LOAD(PFN_vkGetPhysicalDeviceProperties2, vkGetPhysicalDeviceProperties2);
	KYTY_LOAD(PFN_vkGetPhysicalDeviceMemoryProperties, vkGetPhysicalDeviceMemoryProperties);
	KYTY_LOAD(PFN_vkEnumerateDeviceExtensionProperties, vkEnumerateDeviceExtensionProperties);
	KYTY_LOAD(PFN_vkCreateDevice, vkCreateDevice);
	KYTY_LOAD(PFN_vkGetDeviceProcAddr, vkGetDeviceProcAddr);
#undef KYTY_LOAD
	impl->destroy_instance = vkDestroyInstance;

	uint32_t count = 0;
	vkEnumeratePhysicalDevices(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	vkEnumeratePhysicalDevices(instance, &count, devices.data());
	devices.resize(count);
	if (const auto index = ::Config::GetGpuIndex();
	    index >= 0 && static_cast<size_t>(index) < devices.size()) {
		devices = {devices[index]};
	}

	struct Candidate {
		VkPhysicalDevice device      = VK_NULL_HANDLE;
		uint32_t         handle_type = 0;
		uint32_t         type        = 0;
		bool             discrete    = false;
		bool             pci_info    = false;
	};
	std::vector<Candidate> candidates;
	for (auto* physical: devices) {
		uint32_t extension_count = 0;
		vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, nullptr);
		std::vector<VkExtensionProperties> extensions(extension_count);
		vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count,
		                                     extensions.data());
		const auto has = [&](const char* name) {
			return std::any_of(extensions.begin(), extensions.end(), [&](const auto& e) {
				return std::strcmp(e.extensionName, name) == 0;
			});
		};
		if (!has(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) {
			continue;
		}
		VkPhysicalDeviceProperties2 properties {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
		vkGetPhysicalDeviceProperties2(physical, &properties);
		VkPhysicalDeviceMemoryProperties memory {};
		vkGetPhysicalDeviceMemoryProperties(physical, &memory);
		// System memory by default: host-visible, coherent and CPU-cached (a write-combined type
		// would make every guest CPU read uncached), outside VRAM. Mode 2: ReBAR VRAM.
		constexpr VkMemoryPropertyFlags SystemFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
		                                              VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
		constexpr VkMemoryPropertyFlags RebarFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
		                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
		                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
		const auto wanted = mode == 2 ? RebarFlags : SystemFlags;
		uint32_t   type   = UINT32_MAX;
		for (uint32_t t = 0; t < memory.memoryTypeCount && type == UINT32_MAX; t++) {
			const auto flags = memory.memoryTypes[t].propertyFlags;
			const auto heap  = memory.memoryHeaps[memory.memoryTypes[t].heapIndex].flags;
			// Plain types only (no AMD device-coherent/uncached variants, no protected memory).
			constexpr VkMemoryPropertyFlags Unwanted =
			    VK_MEMORY_PROPERTY_PROTECTED_BIT | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD |
			    VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
			const bool vram = (heap & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
			if ((flags & wanted) == wanted && (flags & Unwanted) == 0 && vram == (mode == 2) &&
			    (mode == 2 || (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0)) {
				type = t;
			}
		}
		if (type == UINT32_MAX) {
			continue;
		}
		Candidate candidate {physical,
		                     has(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME)
		                         ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT
		                         : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
		                     type,
		                     properties.properties.deviceType ==
		                         VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
		                     has(VK_EXT_PCI_BUS_INFO_EXTENSION_NAME)};
		candidates.push_back(candidate);
	}
	// The renderer prefers a discrete GPU too; it imports only when the device UUIDs match.
	std::stable_sort(candidates.begin(), candidates.end(),
	                 [](const auto& a, const auto& b) { return a.discrete && !b.discrete; });
	if (candidates.empty()) {
		Log("%s; memfd backing", "no device exports host-visible memory as an fd");
		return nullptr;
	}
	const auto& chosen = candidates.front();

	VkPhysicalDevicePCIBusInfoPropertiesEXT pci {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT};
	VkPhysicalDeviceMaintenance3Properties maintenance3 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
	VkPhysicalDeviceIDProperties id {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
	                                 &maintenance3};
	if (chosen.pci_info) {
		maintenance3.pNext = &pci;
	}
	VkPhysicalDeviceProperties2 properties {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
	vkGetPhysicalDeviceProperties2(chosen.device, &properties);
	std::memcpy(impl->info.device_uuid.data(), id.deviceUUID, VK_UUID_SIZE);
	impl->info.handle_type       = chosen.handle_type;
	impl->info.memory_type_index = chosen.type;
	impl->info.device_local      = mode == 2;
	impl->max_allocation         = maintenance3.maxMemoryAllocationSize;
	if (chosen.pci_info) {
		char path[96];
		std::snprintf(path, sizeof(path), "/sys/bus/pci/devices/%04x:%02x:%02x.%x/",
		              pci.pciDomain, pci.pciBus, pci.pciDevice, pci.pciFunction);
		impl->sysfs = path;
	}

	const float             priority = 0.0f;
	VkDeviceQueueCreateInfo queue {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue.queueFamilyIndex = 0;
	queue.queueCount       = 1;
	queue.pQueuePriorities = &priority;
	const char* extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
	                            VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue;
	device_info.enabledExtensionCount =
	    chosen.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT ? 2 : 1;
	device_info.ppEnabledExtensionNames = extensions;
	if (vkCreateDevice(chosen.device, &device_info, nullptr, &impl->device) != VK_SUCCESS) {
		impl->device = VK_NULL_HANDLE;
		Log("%s; memfd backing", "vkCreateDevice failed");
		return nullptr;
	}
	const auto device = impl->device;
	impl->destroy_device =
	    reinterpret_cast<PFN_vkDestroyDevice>(vkGetDeviceProcAddr(device, "vkDestroyDevice"));
	impl->allocate_memory =
	    reinterpret_cast<PFN_vkAllocateMemory>(vkGetDeviceProcAddr(device, "vkAllocateMemory"));
	impl->free_memory =
	    reinterpret_cast<PFN_vkFreeMemory>(vkGetDeviceProcAddr(device, "vkFreeMemory"));
	impl->get_memory_fd =
	    reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
	if (impl->allocate_memory == nullptr || impl->free_memory == nullptr ||
	    impl->get_memory_fd == nullptr) {
		Log("%s; memfd backing", "device functions missing");
		return nullptr;
	}
	std::printf("Vulkan guest backing: %s, memory type %u (%s), %s handles, allocations up to %lu "
	            "MiB, GTT info %s\n",
	            properties.properties.deviceName, chosen.type,
	            mode == 2 ? "ReBAR VRAM" : "system memory",
	            chosen.handle_type == VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT ? "dma-buf"
	                                                                                 : "opaque fd",
	            impl->max_allocation >> 20u,
	            impl->sysfs.empty() ? "unavailable" : impl->sysfs.c_str());
	std::fflush(stdout);
	return new VulkanGuestAllocator(impl.release());
}

VulkanGuestAllocator::~VulkanGuestAllocator() {
	delete m_impl;
}

const VulkanGuestAllocator::Info& VulkanGuestAllocator::GetInfo() const {
	return m_impl->info;
}

uint64_t VulkanGuestAllocator::MaxAllocationSize() const {
	return m_impl->max_allocation;
}

int VulkanGuestAllocator::Allocate(uint64_t size) {
	auto& impl = *m_impl;
	// The allocation is committed at once and lives outside the process's memory cgroup, so
	// leave room: 1 GiB of GTT for the renderer's own host-visible memory, 2 GiB of system memory.
	constexpr uint64_t GttReserve    = 1ull << 30u;
	constexpr uint64_t SystemReserve = 2ull << 30u;
	if (!impl.info.device_local && !impl.sysfs.empty()) {
		const auto total = ReadSysValue(impl.sysfs + "mem_info_gtt_total");
		const auto used  = ReadSysValue(impl.sysfs + "mem_info_gtt_used");
		if (total != 0 && (used > total || total - used < size + GttReserve)) {
			std::printf("Vulkan guest backing: %lu MiB of GTT left, not enough for %lu MiB\n",
			            (total - std::min(used, total)) >> 20u, size >> 20u);
			std::fflush(stdout);
			return -1;
		}
	}
	if (const auto available = MemAvailable(); available < size + SystemReserve) {
		std::printf("Vulkan guest backing: %lu MiB of system memory available, not enough for %lu "
		            "MiB\n",
		            available >> 20u, size >> 20u);
		std::fflush(stdout);
		return -1;
	}
	VkExportMemoryAllocateInfo export_info {VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
	export_info.handleTypes = impl.info.handle_type;
	VkMemoryAllocateInfo allocate_info {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &export_info};
	allocate_info.allocationSize  = size;
	allocate_info.memoryTypeIndex = impl.info.memory_type_index;
	VkDeviceMemory memory         = VK_NULL_HANDLE;
	if (const auto result = impl.allocate_memory(impl.device, &allocate_info, nullptr, &memory);
	    result != VK_SUCCESS) {
		std::printf("Vulkan guest backing: vkAllocateMemory(%lu MiB) failed: %d\n", size >> 20u,
		            static_cast<int>(result));
		std::fflush(stdout);
		return -1;
	}
	VkMemoryGetFdInfoKHR fd_info {VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
	fd_info.memory     = memory;
	fd_info.handleType = static_cast<VkExternalMemoryHandleTypeFlagBits>(impl.info.handle_type);
	int fd             = -1;
	const auto result  = impl.get_memory_fd(impl.device, &fd_info, &fd);
	// The fd holds its own reference to the memory object.
	impl.free_memory(impl.device, memory, nullptr);
	if (result != VK_SUCCESS || fd < 0) {
		std::printf("Vulkan guest backing: vkGetMemoryFdKHR failed: %d\n", static_cast<int>(result));
		std::fflush(stdout);
		return -1;
	}
	return fd;
}

#else

VulkanGuestAllocator* VulkanGuestAllocator::Create() {
	return nullptr;
}

VulkanGuestAllocator::~VulkanGuestAllocator() = default;

int VulkanGuestAllocator::Allocate(uint64_t /*size*/) {
	return -1;
}

const VulkanGuestAllocator::Info& VulkanGuestAllocator::GetInfo() const {
	static const Info info {};
	return info;
}

uint64_t VulkanGuestAllocator::MaxAllocationSize() const {
	return 0;
}

#endif

} // namespace Libs::LibKernel::Memory
