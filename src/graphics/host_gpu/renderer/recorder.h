#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RECORDER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RECORDER_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstddef>
#include <cstdint>

// Vulkan command recording and submission of the GPU thread on a thread of their own (on by
// request: KYTY_RECORDER=1 turns it on). The recording part is ported from
// batatinhafcbatatinha-pixel/KytyPS5 28a3ca930 and 833cbc96b, which ported it from AnyPS5
// (glue/Recorder.cpp), both GPL-2.0. See recorder.cpp.
namespace Libs::Graphics::Recorder {

// After VULKAN_HPP_DEFAULT_DISPATCHER.init(device), before anything else fetches device functions
// through the dispatcher (the memory allocator gets the waiting destroy and free functions).
void Install();

[[nodiscard]] bool Enabled() noexcept;

// Whether commands the calling thread records into `buffer` are queued for the recorder thread
// (the buffer was begun on the GPU thread and is not ended yet, and the caller is that thread).
[[nodiscard]] bool IsDeferred(VkCommandBuffer buffer) noexcept;

// Runs on the recorder thread after every command queued before it: ends `buffers` and then calls
// `run(data, buffers, count)`. Every buffer must be deferred (IsDeferred), and the caller must be
// the GPU thread. `data` is copied (trivially copyable, alignment at most 16). A record too large
// for the queue runs on the calling thread after the queue drained, like any oversized command.
using DeferredRun = void (*)(const void* data, const VkCommandBuffer* buffers, uint32_t count);
void EndAndRun(const VkCommandBuffer* buffers, uint32_t count, DeferredRun run, const void* data,
               size_t size);

// Before code that records into `buffer` with its own Vulkan function pointers (upscaler SDKs,
// ImGui): every command queued so far has reached the real command buffer when this returns.
void Sync(VkCommandBuffer buffer);
inline void Sync(vk::CommandBuffer buffer) {
	Sync(static_cast<VkCommandBuffer>(buffer));
}

// Waits until every queued command (and deferred submission) ran.
void SyncAll();

// Commands that ran on the calling thread because their copies did not fit into one queue record.
[[nodiscard]] uint64_t OversizedCommands() noexcept;

} // namespace Libs::Graphics::Recorder

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RECORDER_H_ */
