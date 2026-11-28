#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESTATS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <string_view>

// Diagnostic (after AnyPS5 APS5_PIPELINE_STATS): the driver's statistics of every pipeline the
// renderer creates, from VK_KHR_pipeline_executable_properties, to find the guest shaders that
// compile badly on the host GPU (register use, spills, instruction counts). Names, units and the
// statistics themselves are the driver's own, so compare pipelines on one driver only.
//
// KYTY_PIPELINE_STATS=1 writes kyty_pipeline_stats.csv in the working directory; any other value
// but 0 is the file to write. One row per statistic:
//   pipeline,executable,stages,subgroup_size,statistic,value
// "pipeline" is the guest shaders, "<stage>_<hash>" each, as in RenderDoc pipeline names and
// KYTY_TRACE_SHADERS; the hashes match --replay-shaders files. Statistics cost compile time, and
// pipeline binaries are off while it records (a stored binary has no statistics of its own); a
// pipeline the driver's cache returns may report the cached compile or nothing, so start from an
// empty cache (KYTY_PIPELINE_COLD_SALT) for a complete list. tools/replay/pipeline_stats.py
// ranks the result.
namespace Libs::Graphics {
struct GraphicContext;
} // namespace Libs::Graphics

namespace Libs::Graphics::PipelineStats {

[[nodiscard]] bool Requested() noexcept;

// After a pipeline created with vk::PipelineCreateFlagBits::eCaptureStatisticsKHR.
void Record(const GraphicContext& graphics, vk::Pipeline pipeline, std::string_view name);

} // namespace Libs::Graphics::PipelineStats

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESTATS_H_
