#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

struct GraphicContext;

// Diagnostic (KYTY_GPU_BREADCRUMBS=1): every draw and dispatch writes its sequence number into a
// host-visible buffer with VK_AMD_buffer_marker before it starts (top of pipe) and after it
// completes (bottom of pipe). On device loss, the work between the last completed and the last
// started number is printed with its guest shader hashes, followed by VK_EXT_device_fault data.
namespace GpuBreadcrumbs {

// Whether to request the extensions (environment only; the device decides the rest).
[[nodiscard]] bool Requested();

// Records the start marker of one draw or dispatch; returns its number, 0 when disabled.
[[nodiscard]] uint32_t Begin(GraphicContext& graphics, vk::CommandBuffer command, char kind,
                             uint64_t first_hash, uint64_t second_hash = 0);
// Records the completion marker of the work Begin numbered.
void End(GraphicContext& graphics, vk::CommandBuffer command, uint32_t id);

// The number of the last recorded draw or dispatch, and of the last one the GPU completed (both 0
// while disabled). Completed() reads the host-visible marker, so it trails the GPU by nothing.
[[nodiscard]] uint32_t Recorded();
[[nodiscard]] uint32_t Completed();
[[nodiscard]] bool     Available();

// Prints what the GPU was running; call before exiting on VK_ERROR_DEVICE_LOST.
void ReportDeviceLost(GraphicContext& graphics, const char* where);

} // namespace GpuBreadcrumbs
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUBREADCRUMBS_H_
