// Driver cost of compute pipeline creation from several threads of one process against as many
// processes (simnJS 78959548d compiles compute pipelines in helper processes; whether NVIDIA's
// driver compiles faster across processes than across threads is what this measures).
//
// pipeline_bench <spv dir> [--top K] [--skip S] [--threads N | --procs N] [--salt S]
//                [--throttle default|off|eco]
//
// Takes the K largest compute modules (.spv) of the folder after the S largest, raises each
// module's id bound by the salt (as KYTY_PIPELINE_COLD_SALT does, so no disk cache serves it) and
// creates its pipeline, with no VkPipelineCache, on N threads of this process or in N child
// processes of one thread each. Every process has a driver disk cache folder of its own
// (__GL_SHADER_DISK_CACHE_PATH). Work is taken largest first from one shared counter. The device
// enables every feature the driver offers; the layouts are reflected from the SPIR-V (set 0 as
// push descriptors when it fits, like the emulator). `--throttle off` opts the processes out of
// Windows power throttling (EcoQoS), `eco` opts them in, `default` leaves it to Windows.
// Prints one line per pipeline and a summary: the span from the first creation's start to the last
// one's end, the summed driver time, and the share of creations that started on performance cores.
// Windows only.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

PFN_vkGetInstanceProcAddr g_get_instance_proc = nullptr;
PFN_vkCreateInstance      g_create_instance   = nullptr;

#define INSTANCE_FUNCS(X)                                                                          \
	X(vkEnumeratePhysicalDevices)                                                                  \
	X(vkGetPhysicalDeviceProperties)                                                               \
	X(vkGetPhysicalDeviceProperties2)                                                              \
	X(vkGetPhysicalDeviceQueueFamilyProperties)                                                    \
	X(vkGetPhysicalDeviceFeatures2)                                                                \
	X(vkEnumerateDeviceExtensionProperties)                                                        \
	X(vkCreateDevice)                                                                              \
	X(vkGetDeviceProcAddr)

#define DEVICE_FUNCS(X)                                                                            \
	X(vkCreateShaderModule)                                                                        \
	X(vkDestroyShaderModule)                                                                       \
	X(vkCreateDescriptorSetLayout)                                                                 \
	X(vkDestroyDescriptorSetLayout)                                                                \
	X(vkCreatePipelineLayout)                                                                      \
	X(vkDestroyPipelineLayout)                                                                     \
	X(vkCreateComputePipelines)                                                                    \
	X(vkDestroyPipeline)                                                                           \
	X(vkCreatePipelineCache)

#define DECLARE(name) PFN_##name g_##name = nullptr;
INSTANCE_FUNCS(DECLARE)
DEVICE_FUNCS(DECLARE)
#undef DECLARE

struct Binding {
	uint32_t         set   = 0;
	uint32_t         index = 0;
	VkDescriptorType type  = VK_DESCRIPTOR_TYPE_MAX_ENUM;
	uint32_t         count = 1;
};

struct Module {
	std::string           name;
	std::vector<uint32_t> words;
	std::vector<Binding>  bindings;
	bool                  compute = false;
};

// The same reflection as driver_repro.cpp.
void Reflect(Module& module) {
	const auto& words = module.words;
	struct TypeInfo {
		uint32_t op = 0, a = 0, b = 0;
	};
	std::map<uint32_t, TypeInfo> types;
	std::map<uint32_t, uint32_t> constants, sets, bindings, block_kind;
	std::vector<std::pair<uint32_t, std::pair<uint32_t, uint32_t>>> variables;
	for (size_t at = 5; at < words.size();) {
		const uint32_t op = words[at] & 0xffffu, count = words[at] >> 16u;
		if (count == 0 || at + count > words.size()) break;
		const uint32_t* w = &words[at];
		switch (op) {
			case 15:
				if (count >= 3 && w[1] == 5) module.compute = true;
				break;
			case 71:
				if (count >= 4) {
					if (w[2] == 34) sets[w[1]] = w[3];
					if (w[2] == 33) bindings[w[1]] = w[3];
					if (w[2] == 2 || w[2] == 3) block_kind[w[1]] = w[2];
				}
				break;
			case 25:
				if (count >= 9) types[w[1]] = {25, w[7], 0};
				break;
			case 26: types[w[1]] = {26}; break;
			case 27: types[w[1]] = {27}; break;
			case 28:
				if (count >= 4) types[w[1]] = {28, w[2], w[3]};
				break;
			case 29:
				if (count >= 3) types[w[1]] = {29, w[2], 0};
				break;
			case 30: types[w[1]] = {30}; break;
			case 32:
				if (count >= 4) types[w[1]] = {32, w[2], w[3]};
				break;
			case 5341: types[w[1]] = {5341}; break;
			case 43:
				if (count >= 4) constants[w[2]] = w[3];
				break;
			case 59:
				if (count >= 5) variables.push_back({w[2], {w[1], w[3]}});
				break;
			default: break;
		}
		at += count;
	}
	for (const auto& [id, type_class]: variables) {
		const auto [pointer_type, storage_class] = type_class;
		if (storage_class == 9 || bindings.count(id) == 0) continue;
		Binding binding;
		binding.set   = sets.count(id) != 0 ? sets[id] : 0;
		binding.index = bindings[id];
		uint32_t type = types[pointer_type].b;
		while (types[type].op == 28 || types[type].op == 29) {
			const auto& array = types[type];
			binding.count *= array.op == 28 ? std::max(constants[array.b], 1u) : 16u;
			type = array.a;
		}
		const auto& t = types[type];
		if (storage_class == 12 || (storage_class == 2 && block_kind[type] == 3)) {
			binding.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		} else if (storage_class == 2) {
			binding.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		} else if (t.op == 25) {
			binding.type = t.a == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		} else if (t.op == 26) {
			binding.type = VK_DESCRIPTOR_TYPE_SAMPLER;
		} else if (t.op == 27) {
			binding.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		} else if (t.op == 5341) {
			binding.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		} else {
			continue;
		}
		if (std::ranges::none_of(module.bindings, [&](const Binding& b) {
			    return b.set == binding.set && b.index == binding.index;
		    })) {
			module.bindings.push_back(binding);
		}
	}
}

struct Device {
	VkDevice device                   = VK_NULL_HANDLE;
	// --cache: one VkPipelineCache every creation passes, as the emulator's driver cache.
	VkPipelineCache cache             = VK_NULL_HANDLE;
	uint32_t max_push_descriptors     = 0;
	uint32_t max_push_constants       = 128;
};

// With `robust`, robust buffer and image access as the emulator enables them (robustBufferAccess,
// robustBufferAccess2, robustImageAccess2); without, both buffer features off.
bool g_robust = false;
// --uab: sets other than 0 are update-after-bind heaps (ImageHeap).
bool g_uab = false;
// --emu-ext: the emulator's other device extensions and their features.
bool g_emu_ext = false;
// --cache empty|<file> (Device::cache).
std::string g_cache;
// --reserve all|low (ReserveAddressSpace).
std::string g_reserve;
void ReserveAddressSpace(bool low);

bool CreateDevice(Device& out) {
	if (!g_reserve.empty()) ReserveAddressSpace(g_reserve == "low");
	HMODULE library = LoadLibraryA("vulkan-1.dll");
	if (library == nullptr) return false;
	g_get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	    reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
	if (g_get_instance_proc == nullptr) return false;
	g_create_instance = reinterpret_cast<PFN_vkCreateInstance>(
	    g_get_instance_proc(VK_NULL_HANDLE, "vkCreateInstance"));
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "pipeline_bench";
	app.apiVersion       = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	VkInstance instance            = VK_NULL_HANDLE;
	if (g_create_instance(&instance_info, nullptr, &instance) != VK_SUCCESS) return false;
#define LOAD(name)                                                                                 \
	g_##name = reinterpret_cast<PFN_##name>(g_get_instance_proc(instance, #name));                 \
	if (g_##name == nullptr) return false;
	INSTANCE_FUNCS(LOAD)
#undef LOAD
	uint32_t count = 0;
	g_vkEnumeratePhysicalDevices(instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	g_vkEnumeratePhysicalDevices(instance, &count, devices.data());
	if (devices.empty()) return false;
	VkPhysicalDevice physical = devices.front();
	for (const auto candidate: devices) {
		VkPhysicalDeviceProperties properties {};
		g_vkGetPhysicalDeviceProperties(candidate, &properties);
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			physical = candidate;
			break;
		}
	}
	VkPhysicalDevicePushDescriptorPropertiesKHR push {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
	VkPhysicalDeviceProperties2 properties2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &push};
	g_vkGetPhysicalDeviceProperties2(physical, &properties2);
	out.max_push_descriptors = push.maxPushDescriptors;
	out.max_push_constants   = std::min(128u, properties2.properties.limits.maxPushConstantsSize);

	uint32_t families = 0;
	g_vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, nullptr);
	std::vector<VkQueueFamilyProperties> family_list(families);
	g_vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, family_list.data());
	uint32_t family = 0;
	for (uint32_t i = 0; i < families; i++) {
		if ((family_list[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
			family = i;
			break;
		}
	}
	uint32_t extension_count = 0;
	g_vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, nullptr);
	std::vector<VkExtensionProperties> available(extension_count);
	g_vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, available.data());
	const auto has = [&](const char* name) {
		return std::ranges::any_of(available, [&](const VkExtensionProperties& e) {
			return std::strcmp(e.extensionName, name) == 0;
		});
	};
	// Every feature the driver has, so whatever the translator emitted is allowed.
	VkPhysicalDeviceRayQueryFeaturesKHR               ray_query {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
	VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomic_float {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT};
	VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT image_atomic64 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT};
	VkPhysicalDeviceRobustness2FeaturesEXT robust {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
	VkPhysicalDeviceVulkan13Features f13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
	VkPhysicalDeviceVulkan11Features f11 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12};
	VkPhysicalDeviceFeatures2        features {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11};
	std::vector<const char*>         extensions;
	void**                           tail = &f13.pNext;
	const auto chain = [&](const char* extension, auto& feature) {
		if (!has(extension)) return;
		extensions.push_back(extension);
		*tail = &feature;
		tail  = &feature.pNext;
	};
	chain(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, robust);
	chain(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, atomic_float);
	chain(VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME, image_atomic64);
	if (has(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) && has(VK_KHR_RAY_QUERY_EXTENSION_NAME) &&
	    has(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
		extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
		chain(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, acceleration);
		chain(VK_KHR_RAY_QUERY_EXTENSION_NAME, ray_query);
	}
	for (const char* plain: {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
	                         "VK_KHR_shader_float_controls2", "VK_KHR_maintenance5",
	                         "VK_KHR_shader_subgroup_extended_types",
	                         "VK_EXT_shader_demote_to_helper_invocation"}) {
		if (has(plain)) extensions.push_back(plain);
	}
	// --emu-ext: the other device extensions the emulator enables (vulkanWindow.cpp, FSR and GPU
	// breadcrumbs off), each with its features as the driver has them.
	VkPhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_FEATURES_KHR};
	VkPhysicalDeviceMeshShaderFeaturesEXT mesh {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
	VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT feedback_layout {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT};
	VkPhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT feedback_dynamic {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_FEATURES_EXT};
	VkPhysicalDeviceProvokingVertexFeaturesEXT provoking {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT};
	VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR};
	VkPhysicalDeviceMaintenance8FeaturesKHR maintenance8 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
	VkPhysicalDeviceShaderClockFeaturesKHR clock {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
	VkPhysicalDeviceConditionalRenderingFeaturesEXT conditional {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
	VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
	VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
	VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR};
	VkPhysicalDevicePipelineBinaryFeaturesKHR binaries {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR};
	VkPhysicalDeviceDepthClipControlFeaturesEXT clip_control {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_CONTROL_FEATURES_EXT};
	VkPhysicalDeviceImageViewMinLodFeaturesEXT min_lod {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT};
	VkPhysicalDeviceDepthClipEnableFeaturesEXT clip_enable {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT};
	VkPhysicalDeviceExtendedDynamicStateFeaturesEXT dynamic_state {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_FEATURES_EXT};
	if (g_emu_ext) {
		const auto chain_once = [&](const char* extension, auto& feature) {
			if (std::ranges::none_of(extensions, [&](const char* e) { return std::strcmp(e, extension) == 0; })) {
				chain(extension, feature);
			}
		};
		chain_once("VK_KHR_workgroup_memory_explicit_layout", workgroup);
		chain_once("VK_EXT_mesh_shader", mesh);
		chain_once("VK_EXT_attachment_feedback_loop_layout", feedback_layout);
		chain_once("VK_EXT_attachment_feedback_loop_dynamic_state", feedback_dynamic);
		chain_once("VK_EXT_provoking_vertex", provoking);
		chain_once("VK_KHR_unified_image_layouts", unified);
		chain_once("VK_KHR_maintenance8", maintenance8);
		chain_once("VK_KHR_shader_clock", clock);
		chain_once("VK_EXT_conditional_rendering", conditional);
		chain_once("VK_KHR_fragment_shader_barycentric", barycentric);
		chain_once("VK_KHR_compute_shader_derivatives", derivatives);
		chain_once("VK_KHR_pipeline_binary", binaries);
		chain_once("VK_EXT_depth_clip_control", clip_control);
		chain_once("VK_EXT_image_view_min_lod", min_lod);
		chain_once("VK_EXT_depth_clip_enable", clip_enable);
		chain_once("VK_EXT_extended_dynamic_state", dynamic_state);
		if (has("VK_KHR_maintenance5")) {
			*tail = &maintenance5;
			tail  = &maintenance5.pNext;
		}
		for (const char* plain: {"VK_EXT_memory_budget", "VK_EXT_external_memory_host",
		                         "VK_EXT_depth_range_unrestricted", "VK_KHR_swapchain",
		                         "VK_KHR_maintenance1"}) {
			if (has(plain)) extensions.push_back(plain);
		}
	}
	g_vkGetPhysicalDeviceFeatures2(physical, &features);
	// Features that change how the driver compiles or need a capture/replay set-up stay off.
	features.features.robustBufferAccess          = g_robust ? features.features.robustBufferAccess : VK_FALSE;
	acceleration.accelerationStructureCaptureReplay = VK_FALSE;
	acceleration.accelerationStructureIndirectBuild = VK_FALSE;
	acceleration.accelerationStructureHostCommands  = VK_FALSE;
	f12.bufferDeviceAddressCaptureReplay          = VK_FALSE;
	f12.bufferDeviceAddressMultiDevice            = VK_FALSE;
	robust.robustBufferAccess2                    = g_robust ? robust.robustBufferAccess2 : VK_FALSE;
	if (!g_robust) robust.robustImageAccess2 = VK_FALSE;
	const float             priority = 1.0f;
	VkDeviceQueueCreateInfo queue {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue.queueFamilyIndex = family;
	queue.queueCount       = 1;
	queue.pQueuePriorities = &priority;
	VkDeviceCreateInfo info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &features};
	info.queueCreateInfoCount    = 1;
	info.pQueueCreateInfos       = &queue;
	info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	info.ppEnabledExtensionNames = extensions.data();
	if (g_vkCreateDevice(physical, &info, nullptr, &out.device) != VK_SUCCESS) return false;
#define LOAD(name)                                                                                 \
	g_##name = reinterpret_cast<PFN_##name>(g_vkGetDeviceProcAddr(out.device, #name));             \
	if (g_##name == nullptr) return false;
	DEVICE_FUNCS(LOAD)
#undef LOAD
	if (!g_cache.empty()) {
		// "empty", or a file whose VkPipelineCache data starts at its Vulkan header (the
		// emulator's PPSA38500.bin has a prefix of its own before it).
		std::vector<uint8_t> data;
		if (g_cache != "empty") {
			std::ifstream file(g_cache, std::ios::binary);
			data.assign(std::istreambuf_iterator<char>(file), {});
			size_t at = 0;
			for (; at + 12 <= data.size(); at++) {
				uint32_t header[3];
				std::memcpy(header, data.data() + at, sizeof(header));
				if (header[0] >= 32 && header[0] <= 64 && header[1] == 1 && header[2] == 0x10DE) break;
			}
			data.erase(data.begin(), data.begin() + static_cast<ptrdiff_t>(std::min(at, data.size())));
		}
		VkPipelineCacheCreateInfo cache_info {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
		cache_info.initialDataSize = data.size();
		cache_info.pInitialData    = data.empty() ? nullptr : data.data();
		const auto result = g_vkCreatePipelineCache(out.device, &cache_info, nullptr, &out.cache);
		std::printf("pipeline cache: %zu bytes, result %d\n", data.size(), static_cast<int>(result));
	}
	return true;
}

// Exceptions raised in this process (a vectored handler counts them and passes them on): the
// emulator's own vectored handler sees every one the driver raises first.
std::atomic<uint64_t> g_exceptions {0};
LONG CALLBACK CountException(EXCEPTION_POINTERS*) {
	g_exceptions.fetch_add(1, std::memory_order_relaxed);
	return EXCEPTION_CONTINUE_SEARCH;
}

int64_t Now() {
	LARGE_INTEGER value;
	QueryPerformanceCounter(&value);
	return value.QuadPart;
}

double Ms(int64_t ticks) {
	static const double frequency = [] {
		LARGE_INTEGER value;
		QueryPerformanceFrequency(&value);
		return static_cast<double>(value.QuadPart);
	}();
	return static_cast<double>(ticks) * 1000.0 / frequency;
}

// Logical processor number -> true for a performance core (highest efficiency class).
std::vector<bool> PerformanceProcessors() {
	ULONG size = 0;
	GetSystemCpuSetInformation(nullptr, 0, &size, GetCurrentProcess(), 0);
	std::vector<uint8_t> buffer(size);
	std::vector<bool>    result(256, false);
	if (GetSystemCpuSetInformation(reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data()),
	                               size, &size, GetCurrentProcess(), 0) == 0) {
		return result;
	}
	uint8_t best = 0;
	for (ULONG offset = 0; offset < size;) {
		const auto& entry = *reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(&buffer[offset]);
		offset += entry.Size;
		if (entry.Type == CpuSetInformation) best = std::max(best, entry.CpuSet.EfficiencyClass);
	}
	for (ULONG offset = 0; offset < size;) {
		const auto& entry = *reinterpret_cast<const SYSTEM_CPU_SET_INFORMATION*>(&buffer[offset]);
		offset += entry.Size;
		if (entry.Type == CpuSetInformation && entry.CpuSet.Group == 0 &&
		    entry.CpuSet.LogicalProcessorIndex < result.size()) {
			result[entry.CpuSet.LogicalProcessorIndex] = entry.CpuSet.EfficiencyClass == best;
		}
	}
	return result;
}

void SetThrottle(HANDLE process, const std::string& mode) {
	if (mode == "default") return;
	PROCESS_POWER_THROTTLING_STATE state {.Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
	                                      .ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
	                                      .StateMask   = mode == "eco"
	                                                         ? PROCESS_POWER_THROTTLING_EXECUTION_SPEED
	                                                         : 0u};
	SetProcessInformation(process, ProcessPowerThrottling, &state, sizeof(state));
}

// --reserve: before the device is created, reserves the free address space the emulator takes
// for the guest (0x40000 up to 0xfc00000000, as kernel/memoryAddressSpace.inc), so the driver
// allocates as it does inside the emulator. `low`: below 4 GiB only.
void ReserveAddressSpace(bool low) {
	const uint64_t end      = low ? 0x100000000ull : 0xfc00000000ull;
	uint64_t       next     = 0x40000ull;
	uint64_t       reserved = 0;
	while (next < end) {
		MEMORY_BASIC_INFORMATION info {};
		if (VirtualQuery(reinterpret_cast<const void*>(static_cast<uintptr_t>(next)), &info,
		                 sizeof(info)) == 0) {
			break;
		}
		const auto base       = reinterpret_cast<uint64_t>(info.BaseAddress);
		const auto region_end = std::min(end, base + info.RegionSize);
		if (region_end <= next) break;
		if (info.State == MEM_FREE) {
			const auto from = (next + 0xffffull) & ~0xffffull;
			const auto to   = region_end & ~0xffffull;
			if (to > from && VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(from)),
			                              static_cast<size_t>(to - from), MEM_RESERVE,
			                              PAGE_NOACCESS) != nullptr) {
				reserved += to - from;
			}
		}
		next = region_end;
	}
	std::printf("reserved %llu GiB of address space\n",
	            static_cast<unsigned long long>(reserved >> 30));
}


struct Options {
	std::filesystem::path dir;
	size_t                top      = 28;
	size_t                skip     = 0;
	uint32_t              threads  = 1;
	uint32_t              procs    = 0;
	uint32_t              salt     = 1;
	std::string           throttle = "default";
	// Child: index of the shared counter's mapping, and the process number.
	std::string           counter;
	uint32_t              child = UINT32_MAX;
};

std::vector<Module> LoadModules(const Options& options) {
	std::vector<std::pair<uintmax_t, std::filesystem::path>> files;
	for (const auto& entry: std::filesystem::directory_iterator(options.dir)) {
		if (entry.path().extension() == ".spv") files.emplace_back(entry.file_size(), entry.path());
	}
	std::ranges::sort(files, [](const auto& a, const auto& b) {
		return a.first != b.first ? a.first > b.first : a.second < b.second;
	});
	std::vector<Module> modules;
	for (size_t i = options.skip; i < files.size() && modules.size() < options.top; i++) {
		Module        module;
		std::ifstream file(files[i].second, std::ios::binary);
		module.words.resize(files[i].first / 4);
		file.read(reinterpret_cast<char*>(module.words.data()),
		          static_cast<std::streamsize>(module.words.size() * 4));
		if (!file || module.words.size() < 5 || module.words[0] != 0x07230203u) continue;
		Reflect(module);
		if (!module.compute) continue;
		module.name = files[i].second.filename().string();
		// The id bound, raised by the salt: other bytes, the same module.
		module.words[3] += options.salt;
		modules.push_back(std::move(module));
	}
	return modules;
}

struct Sample {
	size_t  index = 0;
	int64_t start = 0, end = 0;
	bool    performance = false;
	bool    ok          = false;
	// Raised in the process meanwhile (all threads).
	uint64_t exceptions = 0;
};

Sample Create(const Device& device, const Module& module, size_t index,
              const std::vector<bool>& performance) {
	Sample sample {.index = index};
	VkShaderModuleCreateInfo module_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	module_info.codeSize = module.words.size() * 4;
	module_info.pCode    = module.words.data();
	VkShaderModule shader = VK_NULL_HANDLE;
	if (g_vkCreateShaderModule(device.device, &module_info, nullptr, &shader) != VK_SUCCESS) {
		return sample;
	}
	uint32_t max_set = 0;
	for (const auto& b: module.bindings) max_set = std::max(max_set, b.set);
	uint64_t set0 = 0;
	for (const auto& b: module.bindings) {
		if (b.set == 0) set0 += b.count;
	}
	std::vector<VkDescriptorSetLayout> layouts(module.bindings.empty() ? 0 : max_set + 1);
	for (uint32_t set = 0; set < layouts.size(); set++) {
		std::vector<VkDescriptorSetLayoutBinding> list;
		for (const auto& b: module.bindings) {
			if (b.set == set) list.push_back({b.index, b.type, b.count, VK_SHADER_STAGE_COMPUTE_BIT});
		}
		VkDescriptorSetLayoutCreateInfo info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		if (set == 0 && set0 <= device.max_push_descriptors && device.max_push_descriptors != 0) {
			info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
		}
		// --uab: a set other than 0 is a bindless heap, like the emulator's image heap.
		std::vector<VkDescriptorBindingFlags>       flags(list.size(), 0);
		VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info {
		    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
		if (g_uab && set != 0) {
			for (auto& f: flags) {
				f = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
				    VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
			}
			flags_info.bindingCount  = static_cast<uint32_t>(flags.size());
			flags_info.pBindingFlags = flags.data();
			info.pNext               = &flags_info;
			info.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
		}
		info.bindingCount = static_cast<uint32_t>(list.size());
		info.pBindings    = list.data();
		g_vkCreateDescriptorSetLayout(device.device, &info, nullptr, &layouts[set]);
	}
	VkPushConstantRange        push {VK_SHADER_STAGE_COMPUTE_BIT, 0, device.max_push_constants};
	VkPipelineLayoutCreateInfo layout_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout_info.setLayoutCount         = static_cast<uint32_t>(layouts.size());
	layout_info.pSetLayouts            = layouts.data();
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	VkPipelineLayout layout            = VK_NULL_HANDLE;
	g_vkCreatePipelineLayout(device.device, &layout_info, nullptr, &layout);
	VkComputePipelineCreateInfo info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	info.stage        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	info.stage.module = shader;
	info.stage.pName  = "main";
	info.layout       = layout;
	VkPipeline pipeline = VK_NULL_HANDLE;
	const auto cpu      = GetCurrentProcessorNumber();
	sample.performance  = cpu < performance.size() && performance[cpu];
	const auto exceptions = g_exceptions.load();
	sample.start          = Now();
	sample.ok = g_vkCreateComputePipelines(device.device, device.cache, 1, &info, nullptr,
	                                       &pipeline) == VK_SUCCESS;
	sample.end        = Now();
	sample.exceptions = g_exceptions.load() - exceptions;
	if (pipeline != VK_NULL_HANDLE) g_vkDestroyPipeline(device.device, pipeline, nullptr);
	g_vkDestroyPipelineLayout(device.device, layout, nullptr);
	for (const auto l: layouts) {
		if (l != VK_NULL_HANDLE) g_vkDestroyDescriptorSetLayout(device.device, l, nullptr);
	}
	g_vkDestroyShaderModule(device.device, shader, nullptr);
	return sample;
}

void PrintSample(const Module& module, const Sample& s, uint32_t worker) {
	std::printf("pipe %s words %zu ms %.0f worker %u core %s start %lld end %lld exceptions %llu%s\n",
	            module.name.c_str(), module.words.size(), Ms(s.end - s.start), worker,
	            s.performance ? "P" : "E", static_cast<long long>(s.start),
	            static_cast<long long>(s.end), static_cast<unsigned long long>(s.exceptions),
	            s.ok ? "" : " FAILED");
	std::fflush(stdout);
}

// Takes work from `next` (shared with other processes or threads) until none is left.
void Work(const Device& device, const std::vector<Module>& modules, volatile LONG* next,
          uint32_t worker, const std::vector<bool>& performance, std::mutex& print) {
	while (true) {
		const auto index = static_cast<size_t>(InterlockedIncrement(next) - 1);
		if (index >= modules.size()) return;
		const auto sample = Create(device, modules[index], index, performance);
		std::scoped_lock lock(print);
		PrintSample(modules[index], sample, worker);
	}
}

int RunChild(const Options& options) {
	SetThrottle(GetCurrentProcess(), options.throttle);
	const auto modules = LoadModules(options);
	Device     device;
	if (!CreateDevice(device)) return 4;
	HANDLE mapping = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, options.counter.c_str());
	if (mapping == nullptr) return 5;
	auto* shared = static_cast<volatile LONG*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 64));
	// shared[0] the next job, shared[1] the processes ready; all start together.
	InterlockedIncrement(&shared[1]);
	while (shared[1] < static_cast<LONG>(options.procs)) Sleep(1);
	std::mutex print;
	Work(device, modules, &shared[0], options.child, PerformanceProcessors(), print);
	return 0;
}

std::filesystem::path DiskCache(uint32_t salt, uint32_t index) {
	return std::filesystem::temp_directory_path() /
	       ("pipeline_bench_cache_" + std::to_string(GetCurrentProcessId()) + "_" +
	        std::to_string(salt) + "_" + std::to_string(index));
}

int RunParent(const Options& options, int argc, char** argv) {
	const auto modules = LoadModules(options);
	if (modules.empty()) {
		std::printf("no compute modules in %s\n", options.dir.string().c_str());
		return 2;
	}
	size_t words = 0;
	for (const auto& m: modules) words += m.words.size();
	std::printf("bench: %zu modules (%zu words, largest %zu), %s %u, salt %u, throttle %s\n",
	            modules.size(), words, modules.front().words.size(),
	            options.procs != 0 ? "processes" : "threads",
	            options.procs != 0 ? options.procs : options.threads, options.salt,
	            options.throttle.c_str());
	std::fflush(stdout);
	const int64_t begin = Now();
	if (options.procs == 0) {
		const auto cache = DiskCache(options.salt, 0);
		std::filesystem::create_directories(cache);
		SetEnvironmentVariableW(L"__GL_SHADER_DISK_CACHE_PATH", cache.wstring().c_str());
		SetThrottle(GetCurrentProcess(), options.throttle);
		Device device;
		if (!CreateDevice(device)) {
			std::printf("no device\n");
			return 4;
		}
		const auto       performance = PerformanceProcessors();
		volatile LONG    next        = 0;
		std::mutex       print;
		std::vector<std::thread> threads;
		for (uint32_t i = 0; i < options.threads; i++) {
			threads.emplace_back([&, i] { Work(device, modules, &next, i, performance, print); });
		}
		for (auto& t: threads) t.join();
		std::error_code error;
		std::filesystem::remove_all(cache, error);
	} else {
		const auto name = "pipeline_bench_" + std::to_string(GetCurrentProcessId()) + "_" +
		                  std::to_string(options.salt);
		HANDLE mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, 64,
		                                    name.c_str());
		auto*  shared  = static_cast<volatile LONG*>(MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, 64));
		shared[0] = 0;
		shared[1] = 0;
		std::string arguments;
		for (int i = 1; i < argc; i++) {
			arguments += " \"";
			arguments += argv[i];
			arguments += "\"";
		}
		std::vector<HANDLE> processes;
		std::vector<std::filesystem::path> caches;
		for (uint32_t i = 0; i < options.procs; i++) {
			caches.push_back(DiskCache(options.salt, i));
			std::filesystem::create_directories(caches.back());
			SetEnvironmentVariableW(L"__GL_SHADER_DISK_CACHE_PATH", caches.back().wstring().c_str());
			std::string command = "\"" + std::string(argv[0]) + "\"" + arguments + " --counter " +
			                      name + " --child " + std::to_string(i);
			STARTUPINFOA startup {sizeof(startup)};
			// The children print into this process's output.
			startup.dwFlags    = STARTF_USESTDHANDLES;
			startup.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
			startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
			startup.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
			PROCESS_INFORMATION info {};
			if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
			                    nullptr, nullptr, &startup, &info)) {
				std::printf("could not start child %u (%lu)\n", i, GetLastError());
				return 3;
			}
			CloseHandle(info.hThread);
			processes.push_back(info.hProcess);
		}
		for (size_t i = 0; i < processes.size(); i++) {
			WaitForSingleObject(processes[i], INFINITE);
			DWORD code = 0;
			GetExitCodeProcess(processes[i], &code);
			if (code != 0) std::printf("child %zu exited with %lu\n", i, code);
			CloseHandle(processes[i]);
		}
		for (const auto& cache: caches) {
			std::error_code error;
			std::filesystem::remove_all(cache, error);
		}
	}
	std::printf("bench done: %.1f s from launch\n", Ms(Now() - begin) / 1000.0);
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	AddVectoredExceptionHandler(1, CountException);
	Options options;
	for (int i = 1; i < argc; i++) {
		const std::string arg  = argv[i];
		const char*       next = i + 1 < argc ? argv[i + 1] : "";
		if (arg == "--top") options.top = std::strtoull(next, nullptr, 10), i++;
		else if (arg == "--skip") options.skip = std::strtoull(next, nullptr, 10), i++;
		else if (arg == "--threads") options.threads = std::strtoul(next, nullptr, 10), i++;
		else if (arg == "--procs") options.procs = std::strtoul(next, nullptr, 10), i++;
		else if (arg == "--salt") options.salt = std::strtoul(next, nullptr, 10), i++;
		else if (arg == "--throttle") options.throttle = next, i++;
		else if (arg == "--robust") g_robust = true;
		else if (arg == "--uab") g_uab = true;
		else if (arg == "--emu-ext") g_emu_ext = true;
		else if (arg == "--cache") g_cache = next, i++;
		else if (arg == "--reserve") g_reserve = next, i++;
		else if (arg == "--counter") options.counter = next, i++;
		else if (arg == "--child") options.child = std::strtoul(next, nullptr, 10), i++;
		else if (options.dir.empty()) options.dir = arg;
		else {
			std::printf("unknown argument %s\n", arg.c_str());
			return 2;
		}
	}
	if (options.dir.empty() || options.threads == 0) {
		std::printf("usage: pipeline_bench <spv dir> [--top K] [--skip S] [--threads N | --procs N] "
		            "[--salt S] [--throttle default|off|eco]\n");
		return 2;
	}
	if (options.child != UINT32_MAX) return RunChild(options);
	return RunParent(options, argc, argv);
}
