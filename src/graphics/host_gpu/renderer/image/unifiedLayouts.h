#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_UNIFIEDLAYOUTS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_UNIFIEDLAYOUTS_H_

#include "common/runtimeToggles.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>

// VK_KHR_unified_image_layouts: with the unifiedImageLayouts feature, VK_IMAGE_LAYOUT_GENERAL
// performs as well as the specific layouts for every use except presentation (and video, which
// needs unifiedImageLayoutsVideo). Cached images then stay in GENERAL after their first
// UNDEFINED -> GENERAL transition: Image::Transit still takes the layout a use needs (the
// "logical" layout) and maps it with Host(), so its barriers only order memory accesses, and
// barriers between two reads of the same layout are dropped. Everything that hands a layout
// to Vulkan (copies, clears, attachments, descriptors) maps it the same way.
//
// On when the device enables the feature. KYTY_UNIFIED_LAYOUTS=0 keeps the per-use layouts.
// With KYTY_TOGGLE_FILE, a "unified_layouts=0|1" line switches it at run time (A/B runs); the
// value is re-read at operation boundaries on the GPU thread (Refresh), so one draw or dispatch
// and the copies it records see one mode. Images and image heap elements follow at their next
// use (the heap re-interns its elements when Epoch() changes).
namespace Libs::Graphics::UnifiedLayouts {

namespace Detail {
inline std::atomic<bool>     g_supported {false};
inline std::atomic<bool>     g_active {false};
inline std::atomic<uint32_t> g_epoch {0};
inline uint32_t              g_calls = 0; // GPU thread only.

inline bool EnvAllows() {
	static const bool allowed = [] {
		const char* env = std::getenv("KYTY_UNIFIED_LAYOUTS");
		return !(env != nullptr && env[0] == '0');
	}();
	return allowed;
}
} // namespace Detail

[[nodiscard]] inline bool Supported() noexcept {
	return Detail::g_supported.load(std::memory_order_relaxed);
}

[[nodiscard]] inline bool Active() noexcept {
	return Detail::g_active.load(std::memory_order_relaxed);
}

// Bumped on every switch between the modes.
[[nodiscard]] inline uint32_t Epoch() noexcept {
	return Detail::g_epoch.load(std::memory_order_relaxed);
}

inline void Set(bool active) noexcept {
	if (Detail::g_active.exchange(active, std::memory_order_relaxed) != active) {
		Detail::g_epoch.fetch_add(1u, std::memory_order_relaxed);
	}
}

// At device creation: whether the device enabled unifiedImageLayouts.
inline void Configure(bool supported) noexcept {
	Detail::g_supported.store(supported, std::memory_order_relaxed);
	Set(supported && Detail::EnvAllows());
}

// At an operation boundary on the GPU thread: applies the runtime toggle, if there is one.
inline void Refresh() {
	if (!Supported() || !Common::RuntimeToggles::Enabled() || (Detail::g_calls++ & 63u) != 0u) {
		return;
	}
	const int toggle = Common::RuntimeToggles::Find("unified_layouts");
	if (toggle >= 0) {
		Set(toggle == 1);
	} else {
		Set(Detail::EnvAllows());
	}
}

// The layout to use for `logical` in the current mode. Presentation, video and attachment
// feedback loop layouts keep their own (the extension does not cover the first two, and a
// GENERAL feedback loop attachment would need VkAttachmentFeedbackLoopInfoEXT).
[[nodiscard]] constexpr vk::ImageLayout Map(vk::ImageLayout logical, bool unified) noexcept {
	if (!unified) return logical;
	switch (logical) {
		case vk::ImageLayout::eUndefined:
		case vk::ImageLayout::ePreinitialized:
		case vk::ImageLayout::ePresentSrcKHR:
		case vk::ImageLayout::eSharedPresentKHR:
		case vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT:
		case vk::ImageLayout::eVideoDecodeDstKHR:
		case vk::ImageLayout::eVideoDecodeSrcKHR:
		case vk::ImageLayout::eVideoDecodeDpbKHR:
		case vk::ImageLayout::eVideoEncodeDstKHR:
		case vk::ImageLayout::eVideoEncodeSrcKHR:
		case vk::ImageLayout::eVideoEncodeDpbKHR: return logical;
		default: return vk::ImageLayout::eGeneral;
	}
}

[[nodiscard]] inline vk::ImageLayout Host(vk::ImageLayout logical) noexcept {
	return Map(logical, Active());
}

[[nodiscard]] inline vk::ImageLayout TransferSrc() noexcept {
	return Host(vk::ImageLayout::eTransferSrcOptimal);
}

[[nodiscard]] inline vk::ImageLayout TransferDst() noexcept {
	return Host(vk::ImageLayout::eTransferDstOptimal);
}

// What Image::GetBarriers does for a (sub)resource in `old_layout` last accessed with
// `old_access` that is next used in `new_layout` (already mapped) with `new_access`:
// nothing (the per-use rules: same layout and access, and no write that a repeat must wait for),
// merge the new reader into the state (unified layouts only: two reads of one layout), or emit
// a barrier.
enum class Need : uint8_t { None, Merge, Barrier };

[[nodiscard]] constexpr Need Decide(vk::ImageLayout old_layout, vk::AccessFlags2 old_access,
                                    vk::ImageLayout new_layout, vk::AccessFlags2 new_access,
                                    bool unified) noexcept {
	// Writes that repeat with a barrier on the per-use path (attachment writes do not: the
	// rasterizer orders them).
	constexpr vk::AccessFlags2 repeated_writes = vk::AccessFlagBits2::eTransferWrite |
	                                             vk::AccessFlagBits2::eShaderWrite |
	                                             vk::AccessFlagBits2::eMemoryWrite;
	constexpr vk::AccessFlags2 writes =
	    repeated_writes | vk::AccessFlagBits2::eShaderStorageWrite |
	    vk::AccessFlagBits2::eColorAttachmentWrite |
	    vk::AccessFlagBits2::eDepthStencilAttachmentWrite | vk::AccessFlagBits2::eHostWrite |
	    vk::AccessFlagBits2::eAccelerationStructureWriteKHR;
	if (old_layout != new_layout) {
		return Need::Barrier;
	}
	if (old_access == new_access && !(old_access & repeated_writes)) {
		return Need::None;
	}
	if (unified && !(old_access & writes) && !(new_access & writes)) {
		return Need::Merge;
	}
	return Need::Barrier;
}

} // namespace Libs::Graphics::UnifiedLayouts

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGE_UNIFIEDLAYOUTS_H_
