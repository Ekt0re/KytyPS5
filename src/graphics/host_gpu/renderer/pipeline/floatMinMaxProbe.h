#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FLOATMINMAXPROBE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FLOATMINMAXPROBE_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

namespace Libs::Graphics {

// Runs the min/max edge matrix on the device once and reports how the host's GLSL.std.450
// NMin/NMax compare to the guest's V_MIN/V_MAX rule.
ShaderRecompiler::Spirv::FloatMinMax ProbeFloatMinMax(const GraphicContext& graphics);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_FLOATMINMAXPROBE_H_ */
