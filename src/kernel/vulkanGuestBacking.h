#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_VULKAN_GUEST_BACKING_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_VULKAN_GUEST_BACKING_H_

#include "common/common.h"

#include <array>
#include <cstdint>

namespace Libs::LibKernel::Memory {

// Linux, KYTY_VK_GUEST_BACKING=1: guest direct memory comes from host-visible Vulkan allocations
// exported as dma-buf (or opaque) file descriptors instead of a memfd. The fd can be mapped
// MAP_SHARED any number of times at any page offset and protected like the memfd, so the
// backing alias, guest views and fault handling stay as they are; and the renderer can import
// the same memory and bind zero-copy buffers to it, which RADV refuses for VK_EXT_external_
// memory_host on a memfd mapping (VK_ERROR_INVALID_EXTERNAL_HANDLE).
//
// The backing exists from kernel init, long before the window and its Vulkan device, so this owns
// a small Vulkan device of its own (no queues used) just to allocate. A dma-buf stays valid on its
// own and across devices: the renderer imports it with VkImportMemoryFdInfoKHR.
//
// Elsewhere (Windows, macOS) Create() always returns nullptr.
class VulkanGuestAllocator {
public:
	struct Info {
		std::array<uint8_t, 16> device_uuid {};
		uint32_t                handle_type       = 0; // VkExternalMemoryHandleTypeFlagBits
		uint32_t                memory_type_index = 0;
		bool                    device_local      = false;
	};

	// nullptr when KYTY_VK_GUEST_BACKING is off or no device can export host-visible memory; the
	// reason is logged once. KYTY_VK_GUEST_BACKING=2 picks ReBAR VRAM instead of system memory
	// (an experiment: it competes with the renderer for VRAM).
	static VulkanGuestAllocator* Create();
	~VulkanGuestAllocator();
	KYTY_CLASS_NO_COPY(VulkanGuestAllocator);

	// A new zero-filled allocation as a file descriptor the caller owns, or -1 when the driver
	// refuses or the GTT / system memory headroom is too small (logged).
	int Allocate(uint64_t size);

	[[nodiscard]] const Info& GetInfo() const;
	[[nodiscard]] uint64_t    MaxAllocationSize() const;

private:
	struct Impl;
	explicit VulkanGuestAllocator(Impl* impl): m_impl(impl) {}
	Impl* m_impl = nullptr;
};

[[nodiscard]] int VulkanGuestBackingMode();

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_VULKAN_GUEST_BACKING_H_ */
