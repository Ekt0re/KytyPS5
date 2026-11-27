#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEBINARYSTORE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEBINARYSTORE_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <filesystem>

namespace Libs::Graphics {

struct GraphicContext;

// Our own pipeline binaries (VK_KHR_pipeline_binary), kept in _PipelineCache/<title>.binaries.
//
// The driver's pipeline cache cannot be relied on across runs: NVIDIA keeps it per executable
// identity (a rebuilt or renamed emulator starts cold) and vkGetPipelineCacheData returns at most
// its per-application disk cache, about 430 MB under the "Driver default" shader cache size, while
// a large title needs several GB. So every new build, and every run past that size, compiled
// hundreds of pipelines again (1-18 s each for GTA VI's compute shaders).
//
// Each pipeline is looked up by the driver's own key for its create info (vkGetPipelineKeyKHR).
// Found: it is created from the stored binaries. Not found: it is created with the capture flag
// and its binaries are appended to the store. The file has no size limit (64-bit offsets, read a
// record at a time); its header holds the driver's global key, so a driver update starts it over.
// Anything that fails falls back to plain creation. KYTY_PIPELINE_BINARIES=0 turns it off.
class PipelineBinaryStore {
public:
	// The store of the running process (one device).
	static PipelineBinaryStore& Get();

	// Opens or creates the store at `path`; does nothing when the device lacks the extension.
	void Open(GraphicContext& graphics, const std::filesystem::path& path);
	void Close();
	[[nodiscard]] bool Enabled() const;
	// How this thread's last pipeline was made ("from stored binaries", "compiled, binaries
	// captured", ...), for log lines.
	[[nodiscard]] static const char* LastSource();

	// Create one pipeline like vkCreate*Pipelines with these create infos, through the store.
	[[nodiscard]] vk::Result CreateCompute(GraphicContext& graphics, vk::PipelineCache cache,
	                                       const vk::ComputePipelineCreateInfo& info,
	                                       vk::Pipeline* pipeline);
	[[nodiscard]] vk::Result CreateGraphics(GraphicContext& graphics, vk::PipelineCache cache,
	                                        const vk::GraphicsPipelineCreateInfo& info,
	                                        vk::Pipeline* pipeline);

	PipelineBinaryStore();
	~PipelineBinaryStore();
	PipelineBinaryStore(const PipelineBinaryStore&)            = delete;
	PipelineBinaryStore& operator=(const PipelineBinaryStore&) = delete;

	struct Impl; // Defined in the .cpp.

private:
	Impl* m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEBINARYSTORE_H_
