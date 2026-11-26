#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics::GpuBreadcrumbs {

namespace {

struct Entry {
	uint64_t first  = 0;
	uint64_t second = 0;
	uint32_t id     = 0;
	char     kind   = 0;
};

constexpr uint32_t RingSize = 1u << 16u;

std::mutex                    g_mutex;
bool                          g_initialized = false;
VkBuffer                      g_buffer      = VK_NULL_HANDLE;
VmaAllocation                 g_allocation  = nullptr;
volatile uint32_t*            g_markers     = nullptr; // [0] started, [1] completed
std::atomic<uint32_t>         g_next {0};
std::array<Entry, RingSize>   g_ring {};

bool Initialize(GraphicContext& graphics) {
	std::scoped_lock lock {g_mutex};
	if (g_initialized) return g_markers != nullptr;
	g_initialized = true;
	if (!graphics.buffer_marker_enabled || graphics.allocator == nullptr) return false;
	VkBufferCreateInfo buffer_info {};
	buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	buffer_info.size  = 256;
	buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VmaAllocationCreateInfo allocation_info {};
	allocation_info.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
	                        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
	allocation_info.usage         = VMA_MEMORY_USAGE_AUTO;
	allocation_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
	                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
	VmaAllocationInfo result {};
	if (vmaCreateBuffer(graphics.allocator, &buffer_info, &allocation_info, &g_buffer,
	                    &g_allocation, &result) != VK_SUCCESS ||
	    result.pMappedData == nullptr) {
		Log::WriteToConsoleAndLog("GPU breadcrumbs: no host-visible marker buffer\n");
		return false;
	}
	g_markers = static_cast<volatile uint32_t*>(result.pMappedData);
	g_markers[0] = 0;
	g_markers[1] = 0;
	Log::WriteToConsoleAndLog("GPU breadcrumbs: enabled (VK_AMD_buffer_marker)\n");
	return true;
}

std::string Describe(const Entry& entry) {
	return entry.kind == 'D'
	           ? fmt::format("draw #{} vs=0x{:016x} ps=0x{:016x}", entry.id, entry.first,
	                         entry.second)
	           : fmt::format("dispatch #{} cs=0x{:016x}", entry.id, entry.first);
}

} // namespace

bool Requested() {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_GPU_BREADCRUMBS");
		return value != nullptr && value[0] != 0 && std::strcmp(value, "0") != 0;
	}();
	return requested;
}

uint32_t Begin(GraphicContext& graphics, vk::CommandBuffer command, char kind,
               uint64_t first_hash, uint64_t second_hash) {
	if (!Requested() || !Initialize(graphics)) return 0;
	const auto id = g_next.fetch_add(1, std::memory_order_relaxed) + 1u;
	auto&      entry = g_ring[id % RingSize];
	entry = {first_hash, second_hash, id, kind};
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eTopOfPipe, g_buffer, 0, id);
	if (id % 100000u == 0u) {
		// Shows the markers advancing (the GPU trails the recording).
		Log::WriteToConsoleAndLog(fmt::format("GPU breadcrumbs: recorded {}, started {}, completed {}\n",
		                                      id, g_markers[0], g_markers[1]));
	}
	return id;
}

void End(GraphicContext& graphics, vk::CommandBuffer command, uint32_t id) {
	if (id == 0 || g_markers == nullptr) return;
	(void)graphics;
	command.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, g_buffer, 4, id);
}

void ReportDeviceLost(GraphicContext& graphics, const char* where) {
	std::string text = fmt::format("GPU device lost ({})\n", where);
	if (g_markers != nullptr) {
		const uint32_t started   = g_markers[0];
		const uint32_t completed = g_markers[1];
		const uint32_t recorded  = g_next.load(std::memory_order_relaxed);
		text += fmt::format("GPU breadcrumbs: recorded {}, started {}, completed {}\n", recorded,
		                    started, completed);
		// Bottom-of-pipe markers complete in order; everything after `completed` up to the
		// last started one was in flight.
		const uint32_t first = completed + 1u;
		const uint32_t last  = std::max(started, first);
		for (uint32_t id = first; id <= last && id - first < 32u; ++id) {
			const auto& entry = g_ring[id % RingSize];
			if (entry.id == id) text += "  in flight: " + Describe(entry) + "\n";
		}
		const auto& done = g_ring[completed % RingSize];
		if (completed != 0 && done.id == completed) {
			text += "  last completed: " + Describe(done) + "\n";
		}
	} else if (Requested()) {
		text += "GPU breadcrumbs: requested but unavailable (no VK_AMD_buffer_marker)\n";
	}
	if (graphics.device_fault_enabled && graphics.device != nullptr) {
		VkDeviceFaultCountsEXT counts {};
		counts.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT;
		const auto get = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(graphics.device,
		                                                      "vkGetDeviceFaultInfoEXT"));
		if (get != nullptr && get(graphics.device, &counts, nullptr) == VK_SUCCESS) {
			std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
			std::vector<VkDeviceFaultVendorInfoEXT>  vendors(counts.vendorInfoCount);
			VkDeviceFaultInfoEXT                     info {};
			info.sType             = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT;
			info.pAddressInfos     = addresses.data();
			info.pVendorInfos      = vendors.data();
			counts.vendorBinarySize = 0;
			const auto status = get(graphics.device, &counts, &info);
			if (status == VK_SUCCESS || status == VK_INCOMPLETE) {
				text += fmt::format("Device fault: {}\n", info.description);
				for (const auto& address: addresses) {
					text += fmt::format("  address type {} 0x{:016x} (precision {})\n",
					                    static_cast<int>(address.addressType),
					                    address.reportedAddress, address.addressPrecision);
				}
				for (const auto& vendor: vendors) {
					text += fmt::format("  vendor 0x{:x} data 0x{:x}: {}\n", vendor.vendorFaultCode,
					                    vendor.vendorFaultData, vendor.description);
				}
			}
		}
	}
	Log::WriteToConsoleAndLog(text);
}

} // namespace Libs::Graphics::GpuBreadcrumbs
