// Command recorder (graphics/host_gpu/renderer/recorder.cpp) without a Vulkan device: the
// dispatcher gets fake entry points that log each call, then the recorder hooks are installed.
// Checks the order of queued and oversized commands (records larger than one queue record run on
// the calling thread after the queue drained) and the deferred end + run.

#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/recorder.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace Libs::Graphics {
namespace {
thread_local bool t_gpu_thread = false;
} // namespace

bool GuestGpu::IsGpuThread() noexcept {
	return t_gpu_thread;
}
} // namespace Libs::Graphics

namespace {

using namespace Libs::Graphics;

struct Call {
	std::string     name;
	std::thread::id thread;
	uint64_t        value = 0; // count or checksum
};

std::mutex        g_mutex;
std::vector<Call> g_calls;

void Log(const char* name, uint64_t value) {
	std::lock_guard lock(g_mutex);
	g_calls.push_back({name, std::this_thread::get_id(), value});
}

template <typename Pfn>
struct Noop;
template <typename R, typename... Args>
struct Noop<R(VKAPI_PTR*)(Args...)> {
	static R VKAPI_CALL Call(Args... /*args*/) {
		if constexpr (!std::is_void_v<R>) {
			return R {};
		}
	}
};

VKAPI_ATTR VkResult VKAPI_CALL FakeBegin(VkCommandBuffer /*buffer*/, const VkCommandBufferBeginInfo* /*info*/) {
	Log("begin", 0);
	return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeEnd(VkCommandBuffer /*buffer*/) {
	Log("end", 0);
	return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeFill(VkCommandBuffer /*buffer*/, VkBuffer /*dst*/, VkDeviceSize /*offset*/,
                                    VkDeviceSize /*size*/, uint32_t data) {
	Log("fill", data);
}

VKAPI_ATTR void VKAPI_CALL FakeCopy(VkCommandBuffer /*buffer*/, VkBuffer /*src*/, VkBuffer /*dst*/, uint32_t count,
                                    const VkBufferCopy* regions) {
	uint64_t sum = 0;
	for (uint32_t i = 0; i < count; ++i) {
		sum += regions[i].srcOffset + regions[i].dstOffset * 3 + regions[i].size * 7;
	}
	Log("copy", sum);
}

VKAPI_ATTR void VKAPI_CALL FakeBarrier(VkCommandBuffer /*buffer*/, VkPipelineStageFlags /*src*/,
                                       VkPipelineStageFlags /*dst*/, VkDependencyFlags /*flags*/,
                                       uint32_t /*memory_count*/, const VkMemoryBarrier* /*memory*/,
                                       uint32_t /*buffer_count*/, const VkBufferMemoryBarrier* /*buffers*/,
                                       uint32_t image_count, const VkImageMemoryBarrier* images) {
	uint64_t sum = 0;
	for (uint32_t i = 0; i < image_count; ++i) {
		sum += images[i].subresourceRange.baseMipLevel;
	}
	Log("barrier", sum);
}

uint64_t CopySum(const std::vector<VkBufferCopy>& regions) {
	uint64_t sum = 0;
	for (const auto& r: regions) {
		sum += r.srcOffset + r.dstOffset * 3 + r.size * 7;
	}
	return sum;
}

std::vector<VkBufferCopy> Regions(uint32_t count) {
	std::vector<VkBufferCopy> regions(count);
	for (uint32_t i = 0; i < count; ++i) {
		regions[i] = {i, i * 2u + 1u, i % 97u + 1u};
	}
	return regions;
}

void RunLogged(const void* data, const VkCommandBuffer* /*buffers*/, uint32_t count) {
	// The payload's first and last words.
	const auto* words = static_cast<const uint64_t*>(data);
	Log("run", words[0] + words[1] + count);
}

int g_failures = 0;

void Expect(bool condition, const char* what) {
	if (!condition) {
		std::fprintf(stderr, "FAILED: %s\n", what);
		g_failures++;
	}
}

void InstallFakes() {
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
#define KYTY_TEST_NOOP(name) d.vk##name = &Noop<PFN_vk##name>::Call;
	KYTY_TEST_NOOP(ResetCommandBuffer)
	KYTY_TEST_NOOP(AllocateCommandBuffers)
	KYTY_TEST_NOOP(FreeCommandBuffers)
	KYTY_TEST_NOOP(ResetCommandPool)
	KYTY_TEST_NOOP(DestroyCommandPool)
	KYTY_TEST_NOOP(DestroyBuffer)
	KYTY_TEST_NOOP(DestroyImage)
	KYTY_TEST_NOOP(DestroyImageView)
	KYTY_TEST_NOOP(DestroyBufferView)
	KYTY_TEST_NOOP(DestroySampler)
	KYTY_TEST_NOOP(DestroyPipeline)
	KYTY_TEST_NOOP(DestroyPipelineLayout)
	KYTY_TEST_NOOP(DestroyDescriptorSetLayout)
	KYTY_TEST_NOOP(DestroyDescriptorPool)
	KYTY_TEST_NOOP(ResetDescriptorPool)
	KYTY_TEST_NOOP(DestroyQueryPool)
	KYTY_TEST_NOOP(FreeMemory)
	KYTY_TEST_NOOP(GetDeviceProcAddr)
#undef KYTY_TEST_NOOP
	d.vkBeginCommandBuffer = &FakeBegin;
	d.vkEndCommandBuffer   = &FakeEnd;
	d.vkCmdFillBuffer      = &FakeFill;
	d.vkCmdCopyBuffer      = &FakeCopy;
	d.vkCmdPipelineBarrier = &FakeBarrier;
}

} // namespace

int main() {
	InstallFakes();
	setenv("KYTY_RECORDER", "1", 1);
	Recorder::Install();
	Expect(Recorder::Enabled(), "recorder installed");
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;

	const auto main_thread = std::this_thread::get_id();
	std::vector<std::string> expected;
	std::vector<uint64_t>    values;
	std::vector<bool>        on_caller; // ran on the GPU thread (oversized) instead of the recorder

	std::thread gpu([&] {
		t_gpu_thread = true;
		auto* buffer = reinterpret_cast<VkCommandBuffer>(uintptr_t {0x1000});
		auto* second = reinterpret_cast<VkCommandBuffer>(uintptr_t {0x2000});
		auto* dummy  = reinterpret_cast<VkBuffer>(uintptr_t {0x10});

		VkCommandBufferBeginInfo begin {};
		begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		Expect(d.vkBeginCommandBuffer(buffer, &begin) == VK_SUCCESS, "begin");
		Expect(Recorder::IsDeferred(buffer), "buffer begun on the GPU thread is deferred");
		expected.emplace_back("begin"), values.push_back(0), on_caller.push_back(false);

		d.vkCmdFillBuffer(buffer, dummy, 0, 4, 1);
		expected.emplace_back("fill"), values.push_back(1), on_caller.push_back(false);

		// 400,000 regions: 9.6 MB of copies, more than one queue record holds.
		const auto big = Regions(400000);
		d.vkCmdCopyBuffer(buffer, dummy, dummy, static_cast<uint32_t>(big.size()), big.data());
		expected.emplace_back("copy"), values.push_back(CopySum(big)), on_caller.push_back(true);

		const auto small = Regions(10);
		d.vkCmdCopyBuffer(buffer, dummy, dummy, static_cast<uint32_t>(small.size()), small.data());
		expected.emplace_back("copy"), values.push_back(CopySum(small)), on_caller.push_back(false);

		// 200,000 image barriers: 14 MB.
		std::vector<VkImageMemoryBarrier> barriers(200000);
		uint64_t                          barrier_sum = 0;
		for (size_t i = 0; i < barriers.size(); ++i) {
			barriers[i].sType                        = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			barriers[i].subresourceRange.baseMipLevel = static_cast<uint32_t>(i % 13);
			barrier_sum += i % 13;
		}
		d.vkCmdPipelineBarrier(buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
		                       nullptr, 0, nullptr, static_cast<uint32_t>(barriers.size()), barriers.data());
		expected.emplace_back("barrier"), values.push_back(barrier_sum), on_caller.push_back(true);

		d.vkCmdFillBuffer(buffer, dummy, 0, 4, 2);
		expected.emplace_back("fill"), values.push_back(2), on_caller.push_back(false);

		// A 4 KiB payload is queued; the buffer is ended on the recorder thread first.
		std::vector<uint64_t> payload(512, 0);
		payload[0] = 40;
		payload[1] = 2;
		Recorder::EndAndRun(&buffer, 1, &RunLogged, payload.data(), payload.size() * sizeof(uint64_t));
		Expect(!Recorder::IsDeferred(buffer), "ended buffer is no longer deferred");
		expected.emplace_back("end"), values.push_back(0), on_caller.push_back(false);
		expected.emplace_back("run"), values.push_back(43), on_caller.push_back(false);

		// A 9 MB payload does not fit: the end and the run happen on this thread, in order.
		Expect(d.vkBeginCommandBuffer(second, &begin) == VK_SUCCESS, "begin second");
		expected.emplace_back("begin"), values.push_back(0), on_caller.push_back(false);
		d.vkCmdFillBuffer(second, dummy, 0, 4, 3);
		expected.emplace_back("fill"), values.push_back(3), on_caller.push_back(false);
		std::vector<uint64_t> huge((size_t {9} << 20) / sizeof(uint64_t), 0);
		huge[0] = 7;
		huge[1] = 5;
		Recorder::EndAndRun(&second, 1, &RunLogged, huge.data(), huge.size() * sizeof(uint64_t));
		expected.emplace_back("end"), values.push_back(0), on_caller.push_back(true);
		expected.emplace_back("run"), values.push_back(13), on_caller.push_back(true);

		Recorder::SyncAll();
		Expect(Recorder::OversizedCommands() == 3, "three oversized records ran on the caller");

		std::lock_guard lock(g_mutex);
		Expect(g_calls.size() == expected.size(), "number of calls");
		const auto gpu_thread = std::this_thread::get_id();
		for (size_t i = 0; i < expected.size() && i < g_calls.size(); ++i) {
			const auto& call = g_calls[i];
			if (call.name != expected[i] || call.value != values[i]) {
				std::fprintf(stderr, "call %zu: got %s/%llu, expected %s/%llu\n", i, call.name.c_str(),
				             static_cast<unsigned long long>(call.value), expected[i].c_str(),
				             static_cast<unsigned long long>(values[i]));
				g_failures++;
			}
			const bool ran_on_caller = call.thread == gpu_thread;
			if (ran_on_caller != on_caller[i] || call.thread == main_thread) {
				std::fprintf(stderr, "call %zu (%s) ran on the wrong thread\n", i, call.name.c_str());
				g_failures++;
			}
		}
	});
	gpu.join();

	// Another thread's buffer is never deferred.
	{
		auto*                    other = reinterpret_cast<VkCommandBuffer>(uintptr_t {0x3000});
		VkCommandBufferBeginInfo begin {};
		begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		(void)d.vkBeginCommandBuffer(other, &begin);
		Expect(!Recorder::IsDeferred(other), "buffer of another thread is not deferred");
	}

	if (g_failures != 0) {
		std::fprintf(stderr, "recorder tests: %d failures\n", g_failures);
		return 1;
	}
	std::printf("recorder tests passed\n");
	return 0;
}
