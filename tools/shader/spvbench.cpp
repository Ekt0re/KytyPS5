// Offline cold-compile bench: times the host driver's pipeline create for SPIR-V modules, for
// example the ones shader_check writes (SHADER_CHECK_SPIRV). A word count does not predict the
// driver's cost; this measures it without starting the emulator.
//
// Usage: spvbench [options] <module.spv | folder> ...
//   --runs <n>        create each module n times (each one cold)
//   --warm            do not make modules unique, so a driver cache may serve repeats
//   --cache <file>    seed a VkPipelineCache from a file
//   --empty-layout    use no descriptor sets instead of the bindings the module declares
//   --robustness2     enable VK_EXT_robustness2 with the features the device reports
//   --validation      enable VK_LAYER_KHRONOS_validation
//   --stage <s>       cs, vs, ps or ms for every module instead of its entry point
//   --device <i>      physical device index
// One line per create: "<ms> ms words=<n> selects=<n> stage=<s> <file> ok|FAIL <reason>".
// Exits 1 if any module failed to load or create.
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <process.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace {

#define SPVBENCH_INSTANCE_FUNCS(X)                                                                 \
	X(vkDestroyInstance)                                                                           \
	X(vkEnumeratePhysicalDevices)                                                                  \
	X(vkGetPhysicalDeviceProperties)                                                               \
	X(vkGetPhysicalDeviceQueueFamilyProperties)                                                    \
	X(vkGetPhysicalDeviceFeatures2)                                                                \
	X(vkCreateDevice)                                                                              \
	X(vkGetDeviceProcAddr)

#define SPVBENCH_DEVICE_FUNCS(X)                                                                   \
	X(vkDestroyDevice)                                                                             \
	X(vkDeviceWaitIdle)                                                                            \
	X(vkCreateShaderModule)                                                                        \
	X(vkDestroyShaderModule)                                                                       \
	X(vkCreateDescriptorSetLayout)                                                                 \
	X(vkDestroyDescriptorSetLayout)                                                                \
	X(vkCreatePipelineLayout)                                                                      \
	X(vkDestroyPipelineLayout)                                                                     \
	X(vkCreateComputePipelines)                                                                    \
	X(vkCreateGraphicsPipelines)                                                                   \
	X(vkDestroyPipeline)                                                                           \
	X(vkCreatePipelineCache)                                                                       \
	X(vkDestroyPipelineCache)

PFN_vkGetInstanceProcAddr g_get_instance_proc = nullptr;
PFN_vkCreateInstance      g_create_instance   = nullptr;
#define SPVBENCH_DECLARE(name) PFN_##name g_##name = nullptr;
SPVBENCH_INSTANCE_FUNCS(SPVBENCH_DECLARE)
SPVBENCH_DEVICE_FUNCS(SPVBENCH_DECLARE)
#undef SPVBENCH_DECLARE

bool LoadLoader() {
#ifdef _WIN32
	HMODULE library = LoadLibraryA("vulkan-1.dll");
	if (library == nullptr) return false;
	g_get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
	    reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr")));
#else
	const char* names[] = {"libvulkan.so.1", "libvulkan.so", "libvulkan.1.dylib", "libMoltenVK.dylib"};
	void*       library = nullptr;
	for (const char* name: names) {
		if ((library = dlopen(name, RTLD_NOW | RTLD_LOCAL)) != nullptr) break;
	}
	if (library == nullptr) return false;
	g_get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr"));
#endif
	if (g_get_instance_proc == nullptr) return false;
	g_create_instance =
	    reinterpret_cast<PFN_vkCreateInstance>(g_get_instance_proc(VK_NULL_HANDLE, "vkCreateInstance"));
	return g_create_instance != nullptr;
}

bool LoadInstance(VkInstance instance) {
	bool ok = true;
#define SPVBENCH_LOAD(name)                                                                        \
	g_##name = reinterpret_cast<PFN_##name>(g_get_instance_proc(instance, #name));                 \
	ok       = ok && g_##name != nullptr;
	SPVBENCH_INSTANCE_FUNCS(SPVBENCH_LOAD)
#undef SPVBENCH_LOAD
	return ok;
}

bool LoadDevice(VkDevice device) {
	bool ok = true;
#define SPVBENCH_LOAD(name)                                                                        \
	g_##name = reinterpret_cast<PFN_##name>(g_vkGetDeviceProcAddr(device, #name));                 \
	ok       = ok && g_##name != nullptr;
	SPVBENCH_DEVICE_FUNCS(SPVBENCH_LOAD)
#undef SPVBENCH_LOAD
	return ok;
}

constexpr uint32_t kSpirvMagic        = 0x07230203u;
constexpr uint32_t kOpEntryPoint      = 15;
constexpr uint32_t kOpModuleProcessed = 330;
constexpr uint32_t kOpSelect          = 169;

// Instructions of the sections before the annotations: capabilities, extensions, imports, the
// memory model, entry points, execution modes and debug information.
bool BeforeAnnotations(uint32_t op) {
	switch (op) {
		case 2:   // OpSourceContinued
		case 3:   // OpSource
		case 4:   // OpSourceExtension
		case 5:   // OpName
		case 6:   // OpMemberName
		case 7:   // OpString
		case 10:  // OpExtension
		case 11:  // OpExtInstImport
		case 14:  // OpMemoryModel
		case 15:  // OpEntryPoint
		case 16:  // OpExecutionMode
		case 17:  // OpCapability
		case 330: // OpModuleProcessed
		case 331: // OpExecutionModeId
			return true;
		default: return false;
	}
}

struct Options {
	int                   runs           = 1;
	bool                  warm           = false;
	bool                  empty_layout   = false;
	bool                  robustness2    = false;
	bool                  validation     = false;
	bool                  stage_forced   = false;
	VkShaderStageFlagBits forced_stage   = VK_SHADER_STAGE_COMPUTE_BIT;
	uint32_t              device_index   = 0;
	std::string           cache_path;
};

Options g_options;

struct Module {
	std::string           path;
	std::string           name;
	std::vector<uint32_t> words;
};

bool ReadModule(const std::string& path, std::vector<uint32_t>& out) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) return false;
	const auto size = file.tellg();
	if (size <= 0 || size % 4 != 0) return false;
	out.resize(static_cast<size_t>(size) / 4);
	file.seekg(0);
	file.read(reinterpret_cast<char*>(out.data()), size);
	return file.good() && out.size() >= 5 && out[0] == kSpirvMagic;
}

template <typename F>
void ForEachInstruction(const std::vector<uint32_t>& words, F&& f) {
	for (size_t pos = 5; pos < words.size();) {
		const uint32_t count = words[pos] >> 16u;
		if (count == 0 || pos + count > words.size()) return;
		if (!f(pos, words[pos] & 0xffffu, &words[pos], count)) return;
		pos += count;
	}
}

uint32_t CountSelects(const std::vector<uint32_t>& words) {
	uint32_t selects = 0;
	ForEachInstruction(words, [&](size_t, uint32_t op, const uint32_t*, uint32_t) {
		selects += op == kOpSelect ? 1u : 0u;
		return true;
	});
	return selects;
}

// The stage of the module's first entry point; 0 when it has none this bench can build.
VkShaderStageFlagBits StageOf(const std::vector<uint32_t>& words) {
	VkShaderStageFlagBits stage = static_cast<VkShaderStageFlagBits>(0);
	ForEachInstruction(words, [&](size_t, uint32_t op, const uint32_t* w, uint32_t) {
		if (op != kOpEntryPoint) return true;
		switch (w[1]) {
			case 0: stage = VK_SHADER_STAGE_VERTEX_BIT; break;
			case 4: stage = VK_SHADER_STAGE_FRAGMENT_BIT; break;
			case 5: stage = VK_SHADER_STAGE_COMPUTE_BIT; break;
			case 5364: stage = VK_SHADER_STAGE_TASK_BIT_EXT; break;
			case 5365: stage = VK_SHADER_STAGE_MESH_BIT_EXT; break;
			default: break;
		}
		return false;
	});
	return stage;
}

const char* StageName(VkShaderStageFlagBits stage) {
	switch (stage) {
		case VK_SHADER_STAGE_VERTEX_BIT: return "vs";
		case VK_SHADER_STAGE_FRAGMENT_BIT: return "ps";
		case VK_SHADER_STAGE_COMPUTE_BIT: return "cs";
		case VK_SHADER_STAGE_TASK_BIT_EXT: return "ts";
		case VK_SHADER_STAGE_MESH_BIT_EXT: return "ms";
		default: return "?";
	}
}

// An OpModuleProcessed string unique to this process and create, placed at the end of the debug
// section, so a driver cache keyed on the module bytes cannot serve an earlier compile.
// A per-process serial alone is not enough: it repeats across runs.
void MakeUnique(std::vector<uint32_t>& words, uint64_t nonce, uint32_t serial) {
	const std::string     text  = "spvbench cold " + std::to_string(nonce) + "-" + std::to_string(serial);
	const uint32_t        count = 1u + static_cast<uint32_t>((text.size() + 4u) / 4u);
	std::vector<uint32_t> inst(count, 0u);
	inst[0] = (count << 16u) | kOpModuleProcessed;
	std::memcpy(&inst[1], text.data(), text.size());
	size_t at = words.size();
	ForEachInstruction(words, [&](size_t pos, uint32_t op, const uint32_t*, uint32_t) {
		if (BeforeAnnotations(op)) return true;
		at = pos;
		return false;
	});
	words.insert(words.begin() + static_cast<std::ptrdiff_t>(at), inst.begin(), inst.end());
}

// The descriptor bindings the module declares, per set, so the driver compiles the module's
// real resource access; an empty layout understates the cost of resource-heavy modules.
std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> ReflectBindings(const std::vector<uint32_t>& words,
                                                                              VkShaderStageFlags stages) {
	std::unordered_map<uint32_t, uint32_t> set, binding, constants, pointee, array_len, array_elem;
	std::unordered_map<uint32_t, uint32_t> opcode_of, block_kind, image_dim, image_sampled;
	struct Var {
		uint32_t id, type, storage;
	};
	std::vector<Var> vars;
	ForEachInstruction(words, [&](size_t, uint32_t op, const uint32_t* w, uint32_t) {
		switch (op) {
			case 71:
				if (w[2] == 34) set[w[1]] = w[3];
				if (w[2] == 33) binding[w[1]] = w[3];
				if (w[2] == 2 || w[2] == 3) block_kind[w[1]] = w[2];
				break;
			case 43: constants[w[2]] = w[3]; break;
			case 32: pointee[w[1]] = w[3]; break;
			case 28:
				array_elem[w[1]] = w[2];
				array_len[w[1]]  = w[3];
				opcode_of[w[1]]  = op;
				break;
			case 29:
				array_elem[w[1]] = w[2];
				opcode_of[w[1]]  = op;
				break;
			case 25:
				image_dim[w[1]]     = w[3];
				image_sampled[w[1]] = w[7];
				opcode_of[w[1]]     = op;
				break;
			case 26:
			case 27:
			case 30:
			case 5341: opcode_of[w[1]] = op; break;
			case 59: vars.push_back({w[2], w[1], w[3]}); break;
			default: break;
		}
		return true;
	});
	std::map<uint32_t, std::vector<VkDescriptorSetLayoutBinding>> result;
	for (const auto& v: vars) {
		// UniformConstant, Uniform and StorageBuffer variables with a set and a binding.
		if (v.storage != 0 && v.storage != 2 && v.storage != 12) continue;
		if (set.count(v.id) == 0 || binding.count(v.id) == 0) continue;
		uint32_t type = pointee[v.type], descriptors = 1;
		while (opcode_of[type] == 28 || opcode_of[type] == 29) {
			// A runtime array gets a fixed count large enough for the driver to compile indexing.
			descriptors *= opcode_of[type] == 28 ? constants[array_len[type]] : 1024u;
			type = array_elem[type];
		}
		VkDescriptorType kind {};
		const uint32_t   op = opcode_of[type];
		if (v.storage == 12) {
			kind = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		} else if (v.storage == 2) {
			kind = block_kind[type] == 3 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		} else if (op == 26) {
			kind = VK_DESCRIPTOR_TYPE_SAMPLER;
		} else if (op == 27) {
			kind = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		} else if (op == 5341) {
			kind = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		} else if (op == 25 && image_dim[type] == 5) {
			kind = image_sampled[type] == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER
			                                : VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
		} else if (op == 25) {
			kind = image_sampled[type] == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		} else {
			continue;
		}
		VkDescriptorSetLayoutBinding b {};
		b.binding         = binding[v.id];
		b.descriptorType  = kind;
		b.descriptorCount = descriptors;
		b.stageFlags      = stages;
		result[set[v.id]].push_back(b);
	}
	return result;
}

// A vertex stage that does nothing, for fragment modules: a graphics pipeline needs a vertex
// or mesh stage.
constexpr uint32_t kEmptyVertexModule[] = {
    kSpirvMagic, 0x00010000, 0, 5, 0,
    (2u << 16u) | 17u, 1,                       // OpCapability Shader
    (3u << 16u) | 14u, 0, 1,                    // OpMemoryModel Logical GLSL450
    (5u << 16u) | 15u, 0, 1, 0x6e69616d, 0,     // OpEntryPoint Vertex %1 "main"
    (2u << 16u) | 19u, 2,                       // %2 = OpTypeVoid
    (3u << 16u) | 33u, 3, 2,                    // %3 = OpTypeFunction %2
    (5u << 16u) | 54u, 2, 1, 0, 3,              // %1 = OpFunction %2 None %3
    (2u << 16u) | 248u, 4,                      // %4 = OpLabel
    (1u << 16u) | 253u,                         // OpReturn
    (1u << 16u) | 56u,                          // OpFunctionEnd
};

struct Device {
	VkInstance         instance     = VK_NULL_HANDLE;
	VkPhysicalDevice   physical     = VK_NULL_HANDLE;
	VkDevice           device       = VK_NULL_HANDLE;
	VkPipelineCache    cache        = VK_NULL_HANDLE;
	VkShaderModule     empty_vertex = VK_NULL_HANDLE;
	VkPhysicalDeviceProperties props {};
	bool               mesh         = false;
};

bool CreateDevice(Device& d) {
	if (!LoadLoader()) {
		std::fprintf(stderr, "spvbench: no Vulkan loader\n");
		return false;
	}
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.pApplicationName = "spvbench";
	app.apiVersion       = VK_API_VERSION_1_3;
	VkInstanceCreateInfo instance_info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	instance_info.pApplicationInfo = &app;
	const char* layer              = "VK_LAYER_KHRONOS_validation";
	if (g_options.validation) {
		instance_info.enabledLayerCount   = 1;
		instance_info.ppEnabledLayerNames = &layer;
	}
	if (g_create_instance(&instance_info, nullptr, &d.instance) != VK_SUCCESS || !LoadInstance(d.instance)) {
		std::fprintf(stderr, "spvbench: vkCreateInstance failed\n");
		return false;
	}
	uint32_t count = 0;
	g_vkEnumeratePhysicalDevices(d.instance, &count, nullptr);
	std::vector<VkPhysicalDevice> devices(count);
	g_vkEnumeratePhysicalDevices(d.instance, &count, devices.data());
	if (g_options.device_index >= count) {
		std::fprintf(stderr, "spvbench: device %u of %u does not exist\n", g_options.device_index, count);
		return false;
	}
	d.physical = devices[g_options.device_index];
	g_vkGetPhysicalDeviceProperties(d.physical, &d.props);

	uint32_t family_count = 0;
	g_vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &family_count, nullptr);
	std::vector<VkQueueFamilyProperties> families(family_count);
	g_vkGetPhysicalDeviceQueueFamilyProperties(d.physical, &family_count, families.data());
	uint32_t family = UINT32_MAX;
	for (uint32_t i = 0; i < family_count && family == UINT32_MAX; i++) {
		if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0) family = i;
	}
	if (family == UINT32_MAX) {
		std::fprintf(stderr, "spvbench: no graphics queue\n");
		return false;
	}

	// Every feature the device reports is enabled, as the emulator enables what modules use.
	VkPhysicalDeviceVulkan11Features       features11 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
	VkPhysicalDeviceVulkan12Features       features12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	VkPhysicalDeviceVulkan13Features       features13 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
	VkPhysicalDeviceMeshShaderFeaturesEXT  mesh {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
	VkPhysicalDeviceRobustness2FeaturesEXT robustness2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT};
	VkPhysicalDeviceFeatures2              features {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	features.pNext   = &features11;
	features11.pNext = &features12;
	features12.pNext = &features13;
	features13.pNext = &mesh;
	if (g_options.robustness2) mesh.pNext = &robustness2;
	g_vkGetPhysicalDeviceFeatures2(d.physical, &features);
	d.mesh = mesh.meshShader != VK_FALSE;
	// Needs the fragment shading rate extension, which this bench does not enable.
	mesh.primitiveFragmentShadingRateMeshShader = VK_FALSE;
	if (!d.mesh) features13.pNext = g_options.robustness2 ? static_cast<void*>(&robustness2) : nullptr;
	std::vector<const char*> extensions;
	if (d.mesh) extensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
	if (g_options.robustness2) extensions.push_back(VK_EXT_ROBUSTNESS_2_EXTENSION_NAME);

	const float             priority = 1.0f;
	VkDeviceQueueCreateInfo queue_info {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue_info.queueFamilyIndex = family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.pNext                   = &features;
	device_info.queueCreateInfoCount    = 1;
	device_info.pQueueCreateInfos       = &queue_info;
	device_info.enabledExtensionCount   = static_cast<uint32_t>(extensions.size());
	device_info.ppEnabledExtensionNames = extensions.data();
	if (g_vkCreateDevice(d.physical, &device_info, nullptr, &d.device) != VK_SUCCESS || !LoadDevice(d.device)) {
		std::fprintf(stderr, "spvbench: vkCreateDevice failed\n");
		return false;
	}

	if (!g_options.cache_path.empty()) {
		std::ifstream     in(g_options.cache_path, std::ios::binary);
		std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		VkPipelineCacheCreateInfo cache_info {VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
		cache_info.initialDataSize = bytes.size();
		cache_info.pInitialData    = bytes.empty() ? nullptr : bytes.data();
		if (g_vkCreatePipelineCache(d.device, &cache_info, nullptr, &d.cache) != VK_SUCCESS) {
			std::fprintf(stderr, "spvbench: the driver rejected pipeline cache %s\n", g_options.cache_path.c_str());
			return false;
		}
	}

	VkShaderModuleCreateInfo vertex_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	vertex_info.codeSize = sizeof(kEmptyVertexModule);
	vertex_info.pCode    = kEmptyVertexModule;
	if (g_vkCreateShaderModule(d.device, &vertex_info, nullptr, &d.empty_vertex) != VK_SUCCESS) {
		std::fprintf(stderr, "spvbench: the empty vertex module failed\n");
		return false;
	}
	return true;
}

void DestroyDevice(Device& d) {
	if (d.device != VK_NULL_HANDLE) {
		g_vkDeviceWaitIdle(d.device);
		if (d.empty_vertex != VK_NULL_HANDLE) g_vkDestroyShaderModule(d.device, d.empty_vertex, nullptr);
		if (d.cache != VK_NULL_HANDLE) g_vkDestroyPipelineCache(d.device, d.cache, nullptr);
		g_vkDestroyDevice(d.device, nullptr);
	}
	if (d.instance != VK_NULL_HANDLE) g_vkDestroyInstance(d.instance, nullptr);
}

VkResult CreatePipeline(const Device& d, VkShaderModule module, VkShaderStageFlagBits stage, VkPipelineLayout layout,
                        VkPipeline* pipeline) {
	VkPipelineShaderStageCreateInfo stages[2] {};
	for (auto& s: stages) {
		s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		s.pName = "main";
	}
	stages[0].stage  = stage;
	stages[0].module = module;
	if (stage == VK_SHADER_STAGE_COMPUTE_BIT) {
		VkComputePipelineCreateInfo info {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		info.stage  = stages[0];
		info.layout = layout;
		return g_vkCreateComputePipelines(d.device, d.cache, 1, &info, nullptr, pipeline);
	}
	uint32_t stage_count = 1;
	if (stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
		stages[1]        = stages[0];
		stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = d.empty_vertex;
		stage_count      = 2;
	}
	const VkFormat                color_format = VK_FORMAT_B8G8R8A8_UNORM;
	VkPipelineRenderingCreateInfo rendering {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
	rendering.colorAttachmentCount    = 1;
	rendering.pColorAttachmentFormats = &color_format;
	rendering.depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT;
	VkPipelineVertexInputStateCreateInfo   vertex_input {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	VkPipelineInputAssemblyStateCreateInfo input_assembly {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	VkPipelineViewportStateCreateInfo viewport {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	viewport.viewportCount = 1;
	viewport.scissorCount  = 1;
	VkPipelineRasterizationStateCreateInfo rasterizer {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
	rasterizer.cullMode    = VK_CULL_MODE_NONE;
	rasterizer.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rasterizer.lineWidth   = 1.0f;
	VkPipelineMultisampleStateCreateInfo multisample {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineDepthStencilStateCreateInfo depth {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
	depth.depthTestEnable  = VK_TRUE;
	depth.depthWriteEnable = VK_TRUE;
	depth.depthCompareOp   = VK_COMPARE_OP_LESS;
	VkPipelineColorBlendAttachmentState blend_attachment {};
	blend_attachment.colorWriteMask = 0xf;
	VkPipelineColorBlendStateCreateInfo blend {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	blend.attachmentCount            = 1;
	blend.pAttachments               = &blend_attachment;
	const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	VkPipelineDynamicStateCreateInfo dynamic {VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
	dynamic.dynamicStateCount = 2;
	dynamic.pDynamicStates    = dynamic_states;
	const bool                   mesh = stage == VK_SHADER_STAGE_MESH_BIT_EXT;
	VkGraphicsPipelineCreateInfo info {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	info.pNext               = &rendering;
	info.stageCount          = stage_count;
	info.pStages             = stages;
	info.pVertexInputState   = mesh ? nullptr : &vertex_input;
	info.pInputAssemblyState = mesh ? nullptr : &input_assembly;
	info.pViewportState      = &viewport;
	info.pRasterizationState = &rasterizer;
	info.pMultisampleState   = &multisample;
	info.pDepthStencilState  = &depth;
	info.pColorBlendState    = &blend;
	info.pDynamicState       = &dynamic;
	info.layout              = layout;
	return g_vkCreateGraphicsPipelines(d.device, d.cache, 1, &info, nullptr, pipeline);
}

// Returns the create time in milliseconds, or -1 with the reason in error.
double TimeCreate(const Device& d, const std::vector<uint32_t>& words, VkShaderStageFlagBits stage, std::string& error) {
	VkShaderModuleCreateInfo module_info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	module_info.codeSize = words.size() * 4u;
	module_info.pCode    = words.data();
	VkShaderModule module = VK_NULL_HANDLE;
	if (g_vkCreateShaderModule(d.device, &module_info, nullptr, &module) != VK_SUCCESS) {
		error = "vkCreateShaderModule failed";
		return -1.0;
	}
	VkShaderStageFlags layout_stages = stage;
	if (stage == VK_SHADER_STAGE_FRAGMENT_BIT) layout_stages |= VK_SHADER_STAGE_VERTEX_BIT;
	VkPushConstantRange range {};
	range.stageFlags = layout_stages;
	range.size       = d.props.limits.maxPushConstantsSize;
	std::vector<VkDescriptorSetLayout> set_layouts;
	bool                               layouts_ok = true;
	if (!g_options.empty_layout) {
		const auto     reflected = ReflectBindings(words, layout_stages);
		const uint32_t sets      = reflected.empty() ? 0 : reflected.rbegin()->first + 1;
		for (uint32_t index = 0; index < sets; index++) {
			const auto                      found = reflected.find(index);
			VkDescriptorSetLayoutCreateInfo set_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
			if (found != reflected.end()) {
				set_info.bindingCount = static_cast<uint32_t>(found->second.size());
				set_info.pBindings    = found->second.data();
			}
			VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
			layouts_ok = layouts_ok && g_vkCreateDescriptorSetLayout(d.device, &set_info, nullptr, &set_layout) == VK_SUCCESS;
			set_layouts.push_back(set_layout);
		}
	}
	VkPipelineLayoutCreateInfo layout_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &range;
	layout_info.setLayoutCount         = static_cast<uint32_t>(set_layouts.size());
	layout_info.pSetLayouts            = set_layouts.data();
	VkPipelineLayout layout            = VK_NULL_HANDLE;
	double           ms                = -1.0;
	if (!layouts_ok) {
		error = "vkCreateDescriptorSetLayout failed for the reflected bindings";
	} else if (g_vkCreatePipelineLayout(d.device, &layout_info, nullptr, &layout) != VK_SUCCESS) {
		error = "vkCreatePipelineLayout failed";
	} else {
		VkPipeline     pipeline = VK_NULL_HANDLE;
		const auto     begin    = std::chrono::steady_clock::now();
		const VkResult result   = CreatePipeline(d, module, stage, layout, &pipeline);
		const auto     end      = std::chrono::steady_clock::now();
		if (result == VK_SUCCESS) {
			ms = std::chrono::duration<double, std::milli>(end - begin).count();
		} else {
			error = "pipeline create returned " + std::to_string(static_cast<int>(result));
		}
		if (pipeline != VK_NULL_HANDLE) g_vkDestroyPipeline(d.device, pipeline, nullptr);
		g_vkDestroyPipelineLayout(d.device, layout, nullptr);
	}
	for (auto set_layout: set_layouts) {
		if (set_layout != VK_NULL_HANDLE) g_vkDestroyDescriptorSetLayout(d.device, set_layout, nullptr);
	}
	g_vkDestroyShaderModule(d.device, module, nullptr);
	return ms;
}

bool ParseStage(const char* s, VkShaderStageFlagBits& stage) {
	if (std::strcmp(s, "cs") == 0) stage = VK_SHADER_STAGE_COMPUTE_BIT;
	else if (std::strcmp(s, "vs") == 0) stage = VK_SHADER_STAGE_VERTEX_BIT;
	else if (std::strcmp(s, "ps") == 0) stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	else if (std::strcmp(s, "ms") == 0) stage = VK_SHADER_STAGE_MESH_BIT_EXT;
	else return false;
	return true;
}

} // namespace

int main(int argc, char** argv) {
	std::vector<std::string> paths;
	for (int i = 1; i < argc; i++) {
		const bool has_value = i + 1 < argc;
		if (std::strcmp(argv[i], "--runs") == 0 && has_value) {
			g_options.runs = std::max(1, std::atoi(argv[++i]));
		} else if (std::strcmp(argv[i], "--warm") == 0) {
			g_options.warm = true;
		} else if (std::strcmp(argv[i], "--empty-layout") == 0) {
			g_options.empty_layout = true;
		} else if (std::strcmp(argv[i], "--validation") == 0) {
			g_options.validation = true;
		} else if (std::strcmp(argv[i], "--robustness2") == 0) {
			g_options.robustness2 = true;
		} else if (std::strcmp(argv[i], "--cache") == 0 && has_value) {
			g_options.cache_path = argv[++i];
		} else if (std::strcmp(argv[i], "--device") == 0 && has_value) {
			g_options.device_index = static_cast<uint32_t>(std::atoi(argv[++i]));
		} else if (std::strcmp(argv[i], "--stage") == 0 && has_value) {
			if (!ParseStage(argv[++i], g_options.forced_stage)) {
				std::fprintf(stderr, "spvbench: unknown stage %s\n", argv[i]);
				return 2;
			}
			g_options.stage_forced = true;
		} else if (argv[i][0] == '-') {
			std::fprintf(stderr, "spvbench: unknown option %s\n", argv[i]);
			return 2;
		} else {
			paths.emplace_back(argv[i]);
		}
	}
	if (paths.empty()) {
		std::fprintf(stderr, "usage: spvbench [options] <module.spv | folder> ...\n");
		return 2;
	}

	bool                failed = false;
	std::vector<Module> modules;
	auto                add = [&](const std::filesystem::path& path) {
		Module m;
		m.path = path.string();
		m.name = path.filename().string();
		if (ReadModule(m.path, m.words)) {
			modules.push_back(std::move(m));
		} else {
			std::printf("FAIL %s: not a SPIR-V module\n", m.path.c_str());
			failed = true;
		}
	};
	for (const auto& path: paths) {
		std::error_code ec;
		if (std::filesystem::is_directory(path, ec)) {
			for (const auto& entry: std::filesystem::directory_iterator(path, ec)) {
				if (entry.is_regular_file() && entry.path().extension() == ".spv") add(entry.path());
			}
		} else {
			add(path);
		}
	}
	std::sort(modules.begin(), modules.end(),
	          [](const Module& a, const Module& b) { return a.words.size() > b.words.size(); });

	Device d;
	if (!CreateDevice(d)) {
		DestroyDevice(d);
		return 2;
	}
	std::printf("device=%s driver=0x%x modules=%zu runs=%d cold=%d layout=%s robustness2=%d validation=%d cache=%s\n",
	            d.props.deviceName, d.props.driverVersion, modules.size(), g_options.runs, g_options.warm ? 0 : 1,
	            g_options.empty_layout ? "empty" : "reflected", g_options.robustness2 ? 1 : 0,
	            g_options.validation ? 1 : 0, g_options.cache_path.empty() ? "none" : g_options.cache_path.c_str());
	const auto nonce = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) ^
	                   static_cast<uint64_t>(
#ifdef _WIN32
	                       _getpid()
#else
	                       getpid()
#endif
	                   );
	uint32_t serial   = 0;
	double   total_ms = 0.0;
	for (const auto& m: modules) {
		const auto     stage   = g_options.stage_forced ? g_options.forced_stage : StageOf(m.words);
		const uint32_t selects = CountSelects(m.words);
		const char*    skip    = nullptr;
		if (stage == 0) skip = "no entry point this bench can build";
		else if (stage == VK_SHADER_STAGE_TASK_BIT_EXT) skip = "a task module needs its mesh stage";
		else if (stage == VK_SHADER_STAGE_MESH_BIT_EXT && !d.mesh) skip = "the device has no mesh shaders";
		if (skip != nullptr) {
			std::printf("%10s words=%zu selects=%u stage=%s %s FAIL %s\n", "-", m.words.size(), selects,
			            StageName(stage), m.name.c_str(), skip);
			failed = true;
			continue;
		}
		for (int run = 0; run < g_options.runs; run++) {
			auto words = m.words;
			if (!g_options.warm) MakeUnique(words, nonce, ++serial);
			std::string  error;
			const double ms = TimeCreate(d, words, stage, error);
			if (ms < 0.0) {
				std::printf("%10s words=%zu selects=%u stage=%s %s FAIL %s\n", "-", m.words.size(), selects,
				            StageName(stage), m.name.c_str(), error.c_str());
				failed = true;
			} else {
				total_ms += ms;
				std::printf("%7.1f ms words=%zu selects=%u stage=%s %s ok\n", ms, m.words.size(), selects,
				            StageName(stage), m.name.c_str());
			}
			std::fflush(stdout);
		}
	}
	std::printf("total %.1f ms over %zu modules\n", total_ms, modules.size());
	DestroyDevice(d);
	return failed ? 1 : 0;
}
