// Cross-platform driver reproducer for dumped compute SPIR-V. Every (module, feature-variant)
// pair runs in a child process so a driver fault is reported as that child's exit code.
// This tool deliberately does not time pipeline creation; use spvbench for driver-cost numbers.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <ctime>
#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifdef __linux__
#include <sys/mman.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kFeatures           = 1u << 0;
constexpr uint32_t kSubgroup           = 1u << 1;
constexpr uint32_t kPushDesc           = 1u << 2;
constexpr uint32_t kCache              = 1u << 3;
constexpr uint32_t kRobust2            = 1u << 4;
constexpr uint32_t kReserve            = 1u << 5;
constexpr uint32_t kReserveLow         = 1u << 6;
constexpr uint32_t kEmulatorLike       = kFeatures | kSubgroup | kPushDesc | kCache | kRobust2;
constexpr uint32_t kHung               = 259u;
constexpr uint32_t kVariantUnsupported = 6u;

struct Variant {
	uint32_t    bits;
	const char* name;
};

constexpr Variant kVariants[] = {
    {kFeatures, "features"},
    {kFeatures | kSubgroup, "features+subgroup64"},
    {kFeatures | kPushDesc, "features+push-descriptor"},
    {kFeatures | kCache | kRobust2, "features+robustness2+cache"},
    {kEmulatorLike, "emulator-like"},
    {kEmulatorLike | kReserveLow, "emulator-like+reserve-low"},
    {kEmulatorLike | kReserve, "emulator-like+reserve"},
};

PFN_vkGetInstanceProcAddr                  g_get_instance_proc             = nullptr;
PFN_vkCreateInstance                       g_create_instance               = nullptr;
PFN_vkEnumerateInstanceExtensionProperties g_enumerate_instance_extensions = nullptr;

#define INSTANCE_FUNCS(X)                                                                          \
	X(vkDestroyInstance)                                                                           \
	X(vkEnumeratePhysicalDevices)                                                                  \
	X(vkGetPhysicalDeviceProperties)                                                               \
	X(vkGetPhysicalDeviceProperties2)                                                              \
	X(vkGetPhysicalDeviceQueueFamilyProperties)                                                    \
	X(vkGetPhysicalDeviceFeatures2)                                                                \
	X(vkEnumerateDeviceExtensionProperties)                                                        \
	X(vkCreateDevice)                                                                              \
	X(vkGetDeviceProcAddr)

#define DEVICE_FUNCS(X)                                                                            \
	X(vkDestroyDevice)                                                                             \
	X(vkDeviceWaitIdle)                                                                            \
	X(vkCreateShaderModule)                                                                        \
	X(vkDestroyShaderModule)                                                                       \
	X(vkCreateDescriptorSetLayout)                                                                 \
	X(vkDestroyDescriptorSetLayout)                                                                \
	X(vkCreatePipelineLayout)                                                                      \
	X(vkDestroyPipelineLayout)                                                                     \
	X(vkCreatePipelineCache)                                                                       \
	X(vkDestroyPipelineCache)                                                                      \
	X(vkCreateComputePipelines)                                                                    \
	X(vkDestroyPipeline)

#define DECLARE(name) PFN_##name g_##name = nullptr;
INSTANCE_FUNCS(DECLARE)
DEVICE_FUNCS(DECLARE)
#undef DECLARE

std::string g_self;

void Step(const char* text) {
	std::printf("    %s\n", text);
	std::fflush(stdout);
}

[[noreturn]] void Fail(const char* what, VkResult result) {
	std::printf("    FAILED: %s (VkResult %d)\n", what, static_cast<int>(result));
	std::fflush(stdout);
	std::_Exit(2);
}

void Check(const char* what, VkResult result) {
	if (result != VK_SUCCESS) Fail(what, result);
}

void SleepMs(uint32_t milliseconds) {
#ifdef _WIN32
	Sleep(milliseconds);
#else
	timespec delay {static_cast<time_t>(milliseconds / 1000u),
	                static_cast<long>((milliseconds % 1000u) * 1000000u)};
	while (::nanosleep(&delay, &delay) != 0 && errno == EINTR) {
	}
#endif
}

bool LoadLoader() {
#ifdef _WIN32
	HMODULE library = LoadLibraryA("vulkan-1.dll");
	if (library == nullptr) return false;
	g_get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	    reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
	const char* names[] = {"libvulkan.so.1",
	                       "libvulkan.so",
	                       "libvulkan.1.dylib",
	                       "libMoltenVK.dylib",
	                       "@executable_path/libMoltenVK.dylib",
	                       "@executable_path/../Frameworks/libMoltenVK.dylib"};
	void*       library = nullptr;
	for (const char* name: names) {
		if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL)) != nullptr) break;
	}
	if (library == nullptr) return false;
	g_get_instance_proc =
	    reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
	if (g_get_instance_proc == nullptr) return false;
	g_create_instance = reinterpret_cast<PFN_vkCreateInstance>(
	    g_get_instance_proc(VK_NULL_HANDLE, "vkCreateInstance"));
	g_enumerate_instance_extensions = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
	    g_get_instance_proc(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
	return g_create_instance != nullptr;
}

bool HasInstanceExtension(const char* name) {
	if (g_enumerate_instance_extensions == nullptr) return false;
	uint32_t count = 0;
	if (g_enumerate_instance_extensions(nullptr, &count, nullptr) != VK_SUCCESS || count == 0) {
		return false;
	}
	std::vector<VkExtensionProperties> extensions(count);
	if (g_enumerate_instance_extensions(nullptr, &count, extensions.data()) != VK_SUCCESS) {
		return false;
	}
	return std::ranges::any_of(extensions, [&](const VkExtensionProperties& extension) {
		return std::strcmp(extension.extensionName, name) == 0;
	});
}

bool LoadInstance(VkInstance instance) {
	bool ok = true;
#define LOAD(name)                                                                                 \
	g_##name = reinterpret_cast<PFN_##name>(g_get_instance_proc(instance, #name));                 \
	ok       = ok && g_##name != nullptr;
	INSTANCE_FUNCS(LOAD)
#undef LOAD
	return ok;
}

bool LoadDevice(VkDevice device) {
	bool ok = true;
#define LOAD(name)                                                                                 \
	g_##name = reinterpret_cast<PFN_##name>(g_vkGetDeviceProcAddr(device, #name));                 \
	ok       = ok && g_##name != nullptr;
	DEVICE_FUNCS(LOAD)
#undef LOAD
	return ok;
}

bool ReadSpirv(const std::filesystem::path& path, std::vector<uint32_t>& words) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) return false;
	const auto bytes = file.tellg();
	if (bytes <= 0 || static_cast<uint64_t>(bytes) % 4 != 0) return false;
	words.resize(static_cast<size_t>(bytes) / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes));
	return static_cast<bool>(file) && words.size() >= 5 && words[0] == 0x07230203u;
}

struct Binding {
	uint32_t         set   = 0;
	uint32_t         index = 0;
	VkDescriptorType type  = VK_DESCRIPTOR_TYPE_MAX_ENUM;
	uint32_t         count = 1;
};

struct ShaderInfo {
	bool                 compute = false;
	std::vector<Binding> bindings;
};

ShaderInfo ParseSpirv(const std::vector<uint32_t>& words) {
	ShaderInfo info;
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
			case 15: // OpEntryPoint
				if (count >= 3 && w[1] == 5) info.compute = true;
				break;
			case 71: // OpDecorate
				if (count >= 4) {
					if (w[2] == 34) sets[w[1]] = w[3];
					if (w[2] == 33) bindings[w[1]] = w[3];
					if (w[2] == 2 || w[2] == 3) block_kind[w[1]] = w[2];
				}
				break;
			case 25: // OpTypeImage
				if (count >= 9) types[w[1]] = {25, w[7], 0};
				break;
			case 26: // OpTypeSampler
				if (count >= 2) types[w[1]] = {26};
				break;
			case 27: // OpTypeSampledImage
				if (count >= 2) types[w[1]] = {27};
				break;
			case 28: // OpTypeArray
				if (count >= 4) types[w[1]] = {28, w[2], w[3]};
				break;
			case 29: // OpTypeRuntimeArray
				if (count >= 3) types[w[1]] = {29, w[2], 0};
				break;
			case 30: // OpTypeStruct
				if (count >= 2) types[w[1]] = {30};
				break;
			case 32: // OpTypePointer
				if (count >= 4) types[w[1]] = {32, w[2], w[3]};
				break;
			case 5341: // OpTypeAccelerationStructureKHR
				if (count >= 2) types[w[1]] = {5341};
				break;
			case 43: // OpConstant
				if (count >= 4) constants[w[2]] = w[3];
				break;
			case 59: // OpVariable
				if (count >= 5) variables.push_back({w[2], {w[1], w[3]}});
				break;
			default: break;
		}
		at += count;
	}
	for (const auto& [id, type_class]: variables) {
		const auto [pointer_type, storage_class] = type_class;
		if (storage_class == 9 || bindings.count(id) == 0)
			continue; // PushConstant or not a resource
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
			binding.type =
			    t.a == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		} else if (t.op == 26) {
			binding.type = VK_DESCRIPTOR_TYPE_SAMPLER;
		} else if (t.op == 27) {
			binding.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		} else if (t.op == 5341) {
			binding.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		} else {
			continue;
		}
		const bool seen = std::ranges::any_of(info.bindings, [&](const Binding& existing) {
			return existing.set == binding.set && existing.index == binding.index;
		});
		if (!seen) info.bindings.push_back(binding);
	}
	return info;
}

constexpr uint64_t kGranularity = 0x10000;
constexpr uint64_t kMinimumGap  = 0x1000000;

bool ReserveGap(uint64_t begin, uint64_t end, uint64_t& reserved) {
	if (begin >= end) return false;
	const auto from = (begin + kGranularity - 1) & ~(kGranularity - 1);
	const auto to   = end & ~(kGranularity - 1);
	if (to <= from || to - from <= kMinimumGap) return false;
#ifdef _WIN32
	if (VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(from)),
	                 static_cast<size_t>(to - from), MEM_RESERVE, PAGE_NOACCESS) == nullptr) {
		return false;
	}
#elif defined(__linux__) && defined(MAP_FIXED_NOREPLACE)
	const auto flags  = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED_NOREPLACE;
	auto*      mapped = mmap(reinterpret_cast<void*>(static_cast<uintptr_t>(from)),
	                         static_cast<size_t>(to - from), PROT_NONE, flags, -1, 0);
	if (mapped == MAP_FAILED) return false;
	if (reinterpret_cast<uintptr_t>(mapped) != from) {
		munmap(mapped, static_cast<size_t>(to - from));
		static bool warned = false;
		if (!warned) {
			Step("kernel did not honor MAP_FIXED_NOREPLACE; address reservation is unavailable");
			warned = true;
		}
		return false;
	}
#else
	return false;
#endif
	reserved += to - from;
	return true;
}

void ReserveAddressSpace(bool below_4gib_only) {
	const std::pair<uint64_t, uint64_t> all[] = {
	    {0x0000040000ull, 0x1000000000ull},
	    {0x1000000000ull, 0xfc00000000ull},
	    {0x080000000000ull, 0x080000000000ull + (512ull << 30)},
	};
	const std::pair<uint64_t, uint64_t> low[]    = {{0x0000040000ull, 0x100000000ull}};
	const auto*                         regions  = below_4gib_only ? low : all;
	const size_t                        count    = below_4gib_only ? 1 : 3;
	uint64_t                            reserved = 0;
#if defined(_WIN32) || (defined(__linux__) && defined(MAP_FIXED_NOREPLACE))
	for (size_t index = 0; index < count; index++) {
		const auto [start, end] = regions[index];
#ifdef _WIN32
		uint64_t next = start;
		while (next < end) {
			MEMORY_BASIC_INFORMATION info {};
			if (VirtualQuery(reinterpret_cast<const void*>(static_cast<uintptr_t>(next)), &info,
			                 sizeof(info)) == 0) {
				break;
			}
			const auto base = reinterpret_cast<uint64_t>(info.BaseAddress);
			if (info.RegionSize > std::numeric_limits<uint64_t>::max() - base) break;
			const auto region_end = std::min(end, base + info.RegionSize);
			if (region_end <= next) break;
			if (info.State == MEM_FREE) ReserveGap(next, region_end, reserved);
			next = region_end;
		}
#else
		std::vector<std::pair<uint64_t, uint64_t>> mapped;
		std::ifstream                              maps("/proc/self/maps");
		for (std::string line; std::getline(maps, line);) {
			unsigned long long first = 0, last = 0;
			if (std::sscanf(line.c_str(), "%llx-%llx", &first, &last) == 2 && first < last) {
				mapped.emplace_back(first, last);
			}
		}
		std::ranges::sort(mapped);
		uint64_t cursor = start;
		for (const auto& [mapped_begin, mapped_end]: mapped) {
			if (mapped_end <= cursor) continue;
			if (mapped_begin >= end) break;
			if (mapped_begin > cursor) ReserveGap(cursor, std::min(mapped_begin, end), reserved);
			cursor = std::max(cursor, mapped_end);
			if (cursor >= end) break;
		}
		if (cursor < end) ReserveGap(cursor, end, reserved);
#endif
	}
#else
	Step("address-space reservation is not available on this platform");
#endif
	std::printf("    reserved %llu GiB of address space\n",
	            static_cast<unsigned long long>(reserved >> 30));
	std::fflush(stdout);
}

int RunChild(const std::filesystem::path& path, uint32_t bits) {
	std::vector<uint32_t> words;
	if (!ReadSpirv(path, words)) {
		Step("not a SPIR-V module");
		return 5;
	}
	const auto shader = ParseSpirv(words);
	if (!shader.compute) {
		Step("not a compute shader, skipped");
		return 3;
	}
	if (!LoadLoader()) {
		Step("no Vulkan loader");
		return 4;
	}
	if ((bits & (kReserve | kReserveLow)) != 0) ReserveAddressSpace((bits & kReserve) == 0);

	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName   = "driver_repro";
	app.applicationVersion = 1;
	app.apiVersion         = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	std::vector<const char*> instance_extensions;
#ifdef __APPLE__
	if (HasInstanceExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
		instance_extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
		instance_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
	}
#endif
	instance_info.enabledExtensionCount   = static_cast<uint32_t>(instance_extensions.size());
	instance_info.ppEnabledExtensionNames = instance_extensions.data();
	VkInstance instance                   = VK_NULL_HANDLE;
	if (g_create_instance(&instance_info, nullptr, &instance) != VK_SUCCESS ||
	    !LoadInstance(instance)) {
		Step("vkCreateInstance or instance function loading failed");
		return 4;
	}

	uint32_t device_count = 0;
	if (g_vkEnumeratePhysicalDevices(instance, &device_count, nullptr) != VK_SUCCESS ||
	    device_count == 0) {
		Step("no Vulkan device");
		return 4;
	}
	std::vector<VkPhysicalDevice> devices(device_count);
	if (g_vkEnumeratePhysicalDevices(instance, &device_count, devices.data()) != VK_SUCCESS) {
		Step("could not enumerate Vulkan devices");
		return 4;
	}
	VkPhysicalDevice           physical = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties properties {};
	for (const auto candidate: devices) {
		g_vkGetPhysicalDeviceProperties(candidate, &properties);
		if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
			physical = candidate;
			break;
		}
	}
	if (physical == VK_NULL_HANDLE) physical = devices.front();
	g_vkGetPhysicalDeviceProperties(physical, &properties);
	VkPhysicalDeviceSubgroupSizeControlProperties subgroup_properties {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES};
	VkPhysicalDevicePushDescriptorPropertiesKHR push_properties {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
	subgroup_properties.pNext = &push_properties;
	VkPhysicalDeviceProperties2 properties2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
	                                         &subgroup_properties};
	g_vkGetPhysicalDeviceProperties2(physical, &properties2);

	uint32_t family_count = 0;
	g_vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, nullptr);
	std::vector<VkQueueFamilyProperties> families(family_count);
	g_vkGetPhysicalDeviceQueueFamilyProperties(physical, &family_count, families.data());
	uint32_t family = UINT32_MAX;
	for (uint32_t index = 0; index < family_count; index++) {
		if ((families[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
			family = index;
			break;
		}
	}
	if (family == UINT32_MAX) {
		Step("no compute queue");
		return 4;
	}

	uint32_t extension_count = 0;
	g_vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, nullptr);
	std::vector<VkExtensionProperties> available(extension_count);
	g_vkEnumerateDeviceExtensionProperties(physical, nullptr, &extension_count, available.data());
	const auto has = [&](const char* name) {
		return std::ranges::any_of(available, [&](const VkExtensionProperties& extension) {
			return std::strcmp(extension.extensionName, name) == 0;
		});
	};

	VkPhysicalDeviceRobustness2FeaturesEXT supported_robust {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
	VkPhysicalDeviceVulkan13Features supported13 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &supported_robust};
	VkPhysicalDeviceVulkan12Features supported12 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &supported13};
	VkPhysicalDeviceVulkan11Features supported11 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &supported12};
	VkPhysicalDeviceFeatures2 supported {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
	                                     &supported11};
	if (!has(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) supported13.pNext = nullptr;
	g_vkGetPhysicalDeviceFeatures2(physical, &supported);
	const bool subgroup64_capable =
	    supported13.subgroupSizeControl != VK_FALSE &&
	    (subgroup_properties.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
	    subgroup_properties.minSubgroupSize <= 64 && subgroup_properties.maxSubgroupSize >= 64;

	std::vector<const char*>               extensions;
	VkPhysicalDeviceRobustness2FeaturesEXT robust {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
	VkPhysicalDeviceVulkan13Features enabled13 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceVulkan12Features enabled12 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &enabled13};
	VkPhysicalDeviceVulkan11Features enabled11 {
	    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &enabled12};
	VkPhysicalDeviceFeatures2 enabled {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &enabled11};
	if ((bits & kFeatures) != 0) {
		auto&       f                          = enabled.features;
		const auto& s                          = supported.features;
		f.fragmentStoresAndAtomics             = s.fragmentStoresAndAtomics;
		f.samplerAnisotropy                    = s.samplerAnisotropy;
		f.robustBufferAccess                   = s.robustBufferAccess;
		f.depthBounds                          = s.depthBounds;
		f.depthClamp                           = s.depthClamp;
		f.shaderStorageImageWriteWithoutFormat = s.shaderStorageImageWriteWithoutFormat;
		f.shaderStorageImageReadWithoutFormat  = s.shaderStorageImageReadWithoutFormat;
		f.shaderStorageImageMultisample        = s.shaderStorageImageMultisample;
		f.shaderImageGatherExtended            = s.shaderImageGatherExtended;
		f.shaderResourceMinLod                 = s.shaderResourceMinLod;
		f.independentBlend                     = s.independentBlend;
		f.dualSrcBlend                         = s.dualSrcBlend;
		f.tessellationShader                   = s.tessellationShader;
		f.drawIndirectFirstInstance            = s.drawIndirectFirstInstance;
		f.multiDrawIndirect                    = s.multiDrawIndirect;
		f.sampleRateShading                    = s.sampleRateShading;
		f.depthBiasClamp                       = s.depthBiasClamp;
		f.shaderClipDistance                   = s.shaderClipDistance;
		f.shaderCullDistance                   = s.shaderCullDistance;
		f.largePoints                          = s.largePoints;
		f.multiViewport                        = s.multiViewport;
		f.fillModeNonSolid                     = s.fillModeNonSolid;
		f.vertexPipelineStoresAndAtomics       = s.vertexPipelineStoresAndAtomics;
		f.shaderInt64                          = s.shaderInt64;
		f.shaderFloat64                        = s.shaderFloat64;
		enabled11.storageBuffer16BitAccess     = supported11.storageBuffer16BitAccess;
		enabled11.shaderDrawParameters         = supported11.shaderDrawParameters;
		enabled12.samplerMirrorClampToEdge     = supported12.samplerMirrorClampToEdge;
		enabled12.storageBuffer8BitAccess      = supported12.storageBuffer8BitAccess;
		enabled12.shaderBufferInt64Atomics     = supported12.shaderBufferInt64Atomics;
		enabled12.shaderSharedInt64Atomics     = supported12.shaderSharedInt64Atomics;
		enabled12.samplerFilterMinmax          = supported12.samplerFilterMinmax;
		enabled12.shaderSampledImageArrayNonUniformIndexing =
		    supported12.shaderSampledImageArrayNonUniformIndexing;
		enabled12.shaderStorageBufferArrayNonUniformIndexing =
		    supported12.shaderStorageBufferArrayNonUniformIndexing;
		enabled12.shaderStorageImageArrayNonUniformIndexing =
		    supported12.shaderStorageImageArrayNonUniformIndexing;
		enabled12.descriptorBindingSampledImageUpdateAfterBind =
		    supported12.descriptorBindingSampledImageUpdateAfterBind;
		enabled12.descriptorBindingUpdateUnusedWhilePending =
		    supported12.descriptorBindingUpdateUnusedWhilePending;
		enabled12.descriptorBindingPartiallyBound = supported12.descriptorBindingPartiallyBound;
		enabled12.timelineSemaphore               = supported12.timelineSemaphore;
		enabled12.bufferDeviceAddress             = supported12.bufferDeviceAddress;
		enabled12.shaderOutputViewportIndex       = supported12.shaderOutputViewportIndex;
		enabled12.shaderOutputLayer               = supported12.shaderOutputLayer;
		enabled13.synchronization2                = supported13.synchronization2;
		enabled13.dynamicRendering                = supported13.dynamicRendering;
		enabled13.robustImageAccess               = supported13.robustImageAccess;
	}
	if ((bits & kSubgroup) != 0) enabled13.subgroupSizeControl = supported13.subgroupSizeControl;
	if ((bits & kRobust2) != 0 && has(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME)) {
		extensions.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);
		robust.robustBufferAccess2 = supported_robust.robustBufferAccess2;
		robust.robustImageAccess2  = supported_robust.robustImageAccess2;
		robust.nullDescriptor      = supported_robust.nullDescriptor;
		enabled13.pNext            = &robust;
	}
	const bool push_desc = (bits & kPushDesc) != 0 && has(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
	if ((bits & kSubgroup) != 0 && !subgroup64_capable) {
		Step("required subgroup size 64 is not supported for compute");
		return kVariantUnsupported;
	}
	if (push_desc) {
		uint64_t descriptors = 0;
		for (const auto& binding: shader.bindings) {
			if (binding.set == 0) descriptors += binding.count;
		}
		if (descriptors > push_properties.maxPushDescriptors) {
			Step("set 0 exceeds the device push-descriptor limit");
			return kVariantUnsupported;
		}
		extensions.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
	}
	if (has("VK_KHR_portability_subset")) {
		extensions.push_back("VK_KHR_portability_subset");
	}

	const float             priority = 1.0f;
	VkDeviceQueueCreateInfo queue_info {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &enabled};
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	device_info.ppEnabledExtensionNames = extensions.data();
	VkDevice device                     = VK_NULL_HANDLE;
	Step("creating device");
	if (g_vkCreateDevice(physical, &device_info, nullptr, &device) != VK_SUCCESS ||
	    !LoadDevice(device)) {
		Step("vkCreateDevice or device function loading failed");
		return 2;
	}

	VkShaderModuleCreateInfo module_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	module_info.codeSize  = words.size() * sizeof(uint32_t);
	module_info.pCode     = words.data();
	VkShaderModule module = VK_NULL_HANDLE;
	Step("creating shader module");
	Check("vkCreateShaderModule", g_vkCreateShaderModule(device, &module_info, nullptr, &module));
	SleepMs(1500); // A deferred driver-thread fault from module creation surfaces here.

	uint32_t max_set = 0;
	for (const auto& binding: shader.bindings)
		max_set = std::max(max_set, binding.set);
	std::vector<VkDescriptorSetLayout> set_layouts(shader.bindings.empty() ? 0 : max_set + 1);
	for (uint32_t set = 0; set < set_layouts.size(); set++) {
		std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
		for (const auto& binding: shader.bindings) {
			if (binding.set != set) continue;
			layout_bindings.push_back(
			    {binding.index, binding.type, binding.count, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
		}
		VkDescriptorSetLayoutCreateInfo layout_info {
		    VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		if (push_desc && set == 0) {
			layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
		}
		layout_info.bindingCount = static_cast<uint32_t>(layout_bindings.size());
		layout_info.pBindings    = layout_bindings.data();
		Check("vkCreateDescriptorSetLayout",
		      g_vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &set_layouts[set]));
	}

	VkPushConstantRange push_range {
	    VK_SHADER_STAGE_COMPUTE_BIT, 0,
	    std::min<uint32_t>(128, properties.limits.maxPushConstantsSize)};
	VkPipelineLayoutCreateInfo layout_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout_info.setLayoutCount         = static_cast<uint32_t>(set_layouts.size());
	layout_info.pSetLayouts            = set_layouts.data();
	layout_info.pushConstantRangeCount = push_range.size != 0 ? 1u : 0u;
	layout_info.pPushConstantRanges    = &push_range;
	VkPipelineLayout pipeline_layout   = VK_NULL_HANDLE;
	Check("vkCreatePipelineLayout",
	      g_vkCreatePipelineLayout(device, &layout_info, nullptr, &pipeline_layout));

	VkPipelineCache cache = VK_NULL_HANDLE;
	if ((bits & kCache) != 0) {
		VkPipelineCacheCreateInfo cache_info {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
		Check("vkCreatePipelineCache",
		      g_vkCreatePipelineCache(device, &cache_info, nullptr, &cache));
	}
	VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup {
	    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
	subgroup.requiredSubgroupSize = 64;
	VkComputePipelineCreateInfo pipeline_info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	pipeline_info.stage        = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
	pipeline_info.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	pipeline_info.stage.module = module;
	pipeline_info.stage.pName  = "main";
	if ((bits & kSubgroup) != 0 && subgroup64_capable) {
		pipeline_info.stage.pNext = &subgroup;
	}
	pipeline_info.layout = pipeline_layout;
	VkPipeline pipeline  = VK_NULL_HANDLE;
	Step("creating compute pipeline");
	Check("vkCreateComputePipelines",
	      g_vkCreateComputePipelines(device, cache, 1, &pipeline_info, nullptr, &pipeline));
	SleepMs(1500); // Keep the child alive for deferred driver work after pipeline creation.
	g_vkDeviceWaitIdle(device);
	if (pipeline != VK_NULL_HANDLE) g_vkDestroyPipeline(device, pipeline, nullptr);
	if (cache != VK_NULL_HANDLE) g_vkDestroyPipelineCache(device, cache, nullptr);
	g_vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	for (const auto layout: set_layouts) {
		if (layout != VK_NULL_HANDLE) g_vkDestroyDescriptorSetLayout(device, layout, nullptr);
	}
	g_vkDestroyShaderModule(device, module, nullptr);
	g_vkDestroyDevice(device, nullptr);
	g_vkDestroyInstance(instance, nullptr);
	Step("pipeline created OK");
	return 0;
}

std::string SafeText(std::string_view text) {
	std::string out;
	for (size_t index = 0; index < text.size();) {
		if (index + 16 <= text.size() && std::ranges::all_of(text.substr(index, 16), [](char c) {
			    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
		    })) {
			out += "<id>";
			index += 16;
		} else {
			out += text[index++];
		}
	}
	return out;
}

std::string Describe(uint32_t code) {
	switch (code) {
		case 0: return "OK";
		case 2: return "Vulkan error";
		case 3: return "skipped (not compute)";
		case 4: return "no Vulkan";
		case 5: return "invalid SPIR-V";
		case kVariantUnsupported: return "variant unsupported by device";
		case 127: return "could not execute child";
		case kHung: return "HUNG (killed after timeout)";
		case 0xC0000005u: return "CRASH (access violation)";
		default: break;
	}
	if (code >= 128 && code <= 192) {
		return "CRASH (signal " + std::to_string(code - 128) + ")";
	}
	char text[64];
	std::snprintf(text, sizeof(text), "exit code 0x%08X", code);
	return text;
}

#ifdef _WIN32
std::wstring Utf8ToWide(std::string_view text) {
	if (text.empty()) return {};
	const auto size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
	                                      static_cast<int>(text.size()), nullptr, 0);
	if (size == 0) return {};
	std::wstring out(static_cast<size_t>(size), L'\0');
	MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
	                    out.data(), size);
	return out;
}

std::string WideToUtf8(std::wstring_view text) {
	if (text.empty()) return {};
	const auto size =
	    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
	                        static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr);
	if (size == 0) return {};
	std::string out(static_cast<size_t>(size), '\0');
	WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
	                    out.data(), size, nullptr, nullptr);
	return out;
}

std::wstring WindowsQuote(const std::wstring& text) {
	std::wstring out     = L"\"";
	size_t       slashes = 0;
	for (wchar_t character: text) {
		if (character == L'\\') {
			slashes++;
			continue;
		}
		if (character == L'"') {
			out.append(slashes * 2 + 1, L'\\');
			out += character;
		} else {
			out.append(slashes, L'\\');
			out += character;
		}
		slashes = 0;
	}
	out.append(slashes * 2, L'\\');
	out += L'"';
	return out;
}
#endif

std::string PathString(const std::filesystem::path& path) {
#ifdef _WIN32
	return WideToUtf8(path.wstring());
#else
	return path.string();
#endif
}

std::filesystem::path PathFromUtf8(std::string_view text) {
#ifdef _WIN32
	return std::filesystem::path(Utf8ToWide(text));
#else
	return std::filesystem::path(text);
#endif
}

uint32_t RunProcess(const std::vector<std::string>& arguments, uint32_t timeout_ms) {
#ifdef _WIN32
	wchar_t    executable[32768];
	const auto length =
	    GetModuleFileNameW(nullptr, executable, static_cast<DWORD>(std::size(executable)));
	std::wstring command = WindowsQuote(length == 0 ? Utf8ToWide(arguments.front())
	                                                : std::wstring(executable, length));
	for (size_t index = 1; index < arguments.size(); index++) {
		command += L' ';
		command += WindowsQuote(Utf8ToWide(arguments[index]));
	}
	STARTUPINFOW        startup {sizeof(startup)};
	PROCESS_INFORMATION process {};
	if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
	                    nullptr, &startup, &process)) {
		return 127;
	}
	DWORD      code   = kHung;
	const auto waited = WaitForSingleObject(process.hProcess, timeout_ms);
	if (waited == WAIT_TIMEOUT) {
		TerminateProcess(process.hProcess, kHung);
		WaitForSingleObject(process.hProcess, INFINITE);
	} else if (waited == WAIT_OBJECT_0) {
		GetExitCodeProcess(process.hProcess, &code);
	} else {
		code = 126;
	}
	CloseHandle(process.hProcess);
	CloseHandle(process.hThread);
	return code;
#else
	const auto pid = ::fork();
	if (pid < 0) return 127;
	if (pid == 0) {
		std::vector<char*> argv;
		argv.reserve(arguments.size() + 1);
		for (const auto& argument: arguments)
			argv.push_back(const_cast<char*>(argument.c_str()));
		argv.push_back(nullptr);
		if (g_self.find('/') != std::string::npos) {
			::execv(g_self.c_str(), argv.data());
		} else {
			::execvp(g_self.c_str(), argv.data());
		}
		std::_Exit(127);
	}
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
	for (;;) {
		int        status = 0;
		const auto waited = ::waitpid(pid, &status, WNOHANG);
		if (waited == pid) {
			return WIFEXITED(status)     ? static_cast<uint32_t>(WEXITSTATUS(status))
			       : WIFSIGNALED(status) ? 128u + WTERMSIG(status)
			                             : 126u;
		}
		if (waited < 0 && errno != EINTR) return 126;
		if (std::chrono::steady_clock::now() >= deadline) {
			::kill(pid, SIGKILL);
			while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
			}
			return kHung;
		}
		SleepMs(5);
	}
#endif
}

bool RunJobs(const std::vector<std::filesystem::path>& shaders,
             const std::filesystem::path& results_path, uint32_t timeout_ms,
             std::vector<uint32_t>& codes) {
	std::ofstream results(results_path, std::ios::trunc);
	if (!results) return false;
	const auto report = [&](const std::string& line) {
		std::printf("%s\n", SafeText(line).c_str());
		std::fflush(stdout);
		results << SafeText(line) << '\n';
		results.flush();
	};
	report("driver_repro: " + std::to_string(shaders.size()) + " module(s) x " +
	       std::to_string(std::size(kVariants)) + " variants");
	codes.clear();
	for (size_t index = 0; index < shaders.size(); index++) {
		for (const auto& variant: kVariants) {
			const std::vector<std::string> arguments {g_self, "--child", PathString(shaders[index]),
			                                          std::to_string(variant.bits)};
			const auto                     code = RunProcess(arguments, timeout_ms);
			codes.push_back(code);
			report("shader " + std::to_string(index + 1) + " " +
			       SafeText(PathString(shaders[index].filename())) + " [" + variant.name + "] " +
			       Describe(code));
		}
	}
	report("done");
	return true;
}

constexpr uint32_t kMinimalComputeModule[] = {
    0x07230203u,
    0x00010000u,
    0,
    5,
    0,
    (2u << 16u) | 17u,
    1, // OpCapability Shader
    (3u << 16u) | 14u,
    0,
    1, // OpMemoryModel Logical GLSL450
    (5u << 16u) | 15u,
    5,
    1,
    0x6e69616d,
    0, // OpEntryPoint GLCompute %main "main"
    (6u << 16u) | 16u,
    1,
    17,
    1,
    1,
    1, // OpExecutionMode %main LocalSize 1 1 1
    (2u << 16u) | 19u,
    2, // %void = OpTypeVoid
    (3u << 16u) | 33u,
    3,
    2, // %fn = OpTypeFunction %void
    (5u << 16u) | 54u,
    2,
    1,
    0,
    3, // %main = OpFunction %void None %fn
    (2u << 16u) | 248u,
    4,                  // %label = OpLabel
    (1u << 16u) | 253u, // OpReturn
    (1u << 16u) | 56u,  // OpFunctionEnd
};

bool WriteWords(const std::filesystem::path& path, std::span<const uint32_t> words) {
	std::ofstream output(path, std::ios::binary | std::ios::trunc);
	if (!output) return false;
	output.write(reinterpret_cast<const char*>(words.data()),
	             static_cast<std::streamsize>(words.size_bytes()));
	return static_cast<bool>(output);
}

int SelfTest() {
#ifdef _WIN32
	const auto process_id = GetCurrentProcessId();
#else
	const auto process_id = ::getpid();
#endif
	const auto root =
	    std::filesystem::current_path() / ("driver-repro-test-" + std::to_string(process_id));
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(root);
	const auto module = root / "minimal.spv";
	if (!WriteWords(module, kMinimalComputeModule)) return 1;
	std::vector<uint32_t> codes;
	if (!RunJobs({module}, root / "results.txt", 30000, codes) ||
	    codes.size() != std::size(kVariants)) {
		std::filesystem::remove_all(root);
		return 1;
	}
	const bool no_vulkan = std::ranges::all_of(codes, [](uint32_t code) { return code == 4; });
	const bool passed    = no_vulkan || std::ranges::all_of(codes, [](uint32_t code) {
                            return code == 0 || code == kVariantUnsupported;
                        });
	std::filesystem::remove_all(root);
	if (!passed) {
		std::fprintf(stderr, "driver_repro self-test: one or more variants failed\n");
		return 1;
	}
	const bool skipped =
	    std::ranges::any_of(codes, [](uint32_t code) { return code == kVariantUnsupported; });
	std::puts(no_vulkan ? "driver_repro self-test: SKIP (no Vulkan device)"
	          : skipped ? "driver_repro self-test: OK (unsupported variants skipped)"
	                    : "driver_repro self-test: OK");
	return 0;
}

void Usage() {
	std::printf("driver_repro [--results <file>] [--timeout-ms N] <module.spv | folder> ...\n"
	            "driver_repro --self-test\n");
}

bool ParseUnsigned(const char* text, uint32_t& value) {
	char*      end    = nullptr;
	const auto parsed = std::strtoul(text, &end, 10);
	if (end == text || *end != '\0' || parsed > UINT32_MAX) return false;
	value = static_cast<uint32_t>(parsed);
	return true;
}

} // namespace

int RunMain(int argc, char** argv) {
	g_self = argc > 0 && argv[0] != nullptr ? argv[0] : "driver_repro";
	if (argc == 4 && std::strcmp(argv[1], "--child") == 0) {
		char*      end  = nullptr;
		const auto bits = std::strtoul(argv[3], &end, 10);
		if (end == argv[3] || *end != '\0' || bits > UINT32_MAX) return 127;
		return RunChild(PathFromUtf8(argv[2]), static_cast<uint32_t>(bits));
	}
	if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) return SelfTest();
	if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
		Usage();
		return 0;
	}
	std::filesystem::path              results_path;
	uint32_t                           timeout_ms = 60000;
	std::vector<std::filesystem::path> inputs;
	for (int index = 1; index < argc; index++) {
		const std::string_view argument = argv[index];
		if (argument == "--results" || argument == "--timeout-ms") {
			if (++index >= argc) {
				Usage();
				return 2;
			}
			if (argument == "--results")
				results_path = PathFromUtf8(argv[index]);
			else if (!ParseUnsigned(argv[index], timeout_ms) || timeout_ms == 0 ||
			         timeout_ms == UINT32_MAX) {
				Usage();
				return 2;
			}
		} else if (argument.starts_with('-')) {
			Usage();
			return 2;
		} else {
			inputs.emplace_back(PathFromUtf8(argv[index]));
		}
	}
	if (inputs.empty()) {
		Usage();
		return 2;
	}
	std::vector<std::filesystem::path> shaders;
	for (const auto& input: inputs) {
		std::error_code error;
		if (std::filesystem::is_directory(input, error)) {
			for (const auto& entry: std::filesystem::directory_iterator(input, error)) {
				if (entry.is_regular_file() && entry.path().extension() == ".spv") {
					shaders.push_back(entry.path());
				}
			}
		} else if (std::filesystem::is_regular_file(input, error) && input.extension() == ".spv") {
			shaders.push_back(input);
		}
		if (error) {
			std::fprintf(stderr, "driver_repro: could not read input: %s\n",
			             SafeText(error.message()).c_str());
			return 1;
		}
	}
	std::ranges::sort(shaders);
	shaders.erase(std::unique(shaders.begin(), shaders.end()), shaders.end());
	if (shaders.empty()) {
		std::fprintf(stderr, "driver_repro: no SPIR-V modules found\n");
		return 1;
	}
	if (results_path.empty()) {
		std::error_code error;
		const auto&     first = inputs.front();
		const auto      folder =
            std::filesystem::is_directory(first, error) ? first : first.parent_path();
		results_path = folder.empty() ? std::filesystem::current_path() : folder;
		results_path /= "driver_repro_results.txt";
	}
	std::vector<uint32_t> codes;
	return RunJobs(shaders, results_path, timeout_ms, codes) ? 0 : 1;
}

#ifdef _WIN32
int main() {
	int     argc = 0;
	LPWSTR* argv = CommandLineToArgvW(GetCommandLineW(), &argc);
	if (argv == nullptr) return 2;
	std::vector<std::string> arguments;
	arguments.reserve(static_cast<size_t>(argc));
	for (int index = 0; index < argc; index++)
		arguments.push_back(WideToUtf8(argv[index]));
	LocalFree(argv);
	std::vector<char*> utf8;
	utf8.reserve(arguments.size() + 1);
	for (auto& argument: arguments)
		utf8.push_back(argument.data());
	utf8.push_back(nullptr);
	return RunMain(argc, utf8.data());
}
#else
int main(int argc, char** argv) {
	return RunMain(argc, argv);
}
#endif
