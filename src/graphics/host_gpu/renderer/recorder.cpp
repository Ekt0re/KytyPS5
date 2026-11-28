// Vulkan command recording and submission on a second thread.
//
// Ported from batatinhafcbatatinha-pixel/KytyPS5 28a3ca930 and 833cbc96b (GPL-2.0), which ported
// it from AnyPS5 (glue/Recorder.cpp, GPL-2.0). Changes here: the deferred end and submission of
// the GPU thread's command buffers (EndAndRun), so the GPU thread never waits for the recorder at
// a submission; threads other than the GPU thread do not wait for the queue when they destroy
// objects (they may hold the queue mutex the deferred submission takes, and they never destroy
// objects queued commands use: those are released once their tick completed, which needs the
// submission); vkCmdResetQueryPool is copied; the KYTY_X_CUT switches are fixed to their default.
//
// The GPU thread spends a large part of every draw inside the Vulkan driver recording commands.
// Command buffers begun on the GPU thread are "deferred": their vkCmd* calls (Vulkan-Hpp calls
// them through the dynamic dispatcher, whose function pointers are replaced here) copy their
// arguments, including the arrays and structures they point to, into a ring; a recorder thread
// replays them in order into the real command buffer. vkEndCommandBuffer, command pool
// operations and the destruction of objects a pending command may use wait until the recorder
// has replayed everything queued so far. Commands with extension structures (pNext), commands
// that are not copied here (the "fallback" list) and calls from other threads on a deferred
// buffer also wait, then run on the calling thread.
//
// Code that records with its own function pointers (upscaler SDKs, ImGui) must call Sync() on
// the command buffer first.
//
// On by default; KYTY_RECORDER=0 turns it off (KYTY_RECORDER_TOGGLE then switches it at run time).

#include "graphics/host_gpu/renderer/recorder.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perfSummary.h"
#include "common/profiler.h"
#include "common/runtimeToggles.h"
#include "graphics/guest_gpu/graphicsRun.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#if defined(__linux__)
#include <pthread.h>
#endif
#include <new>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace Libs::Graphics::Recorder {
namespace {

void CpuRelax() noexcept {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
	__builtin_ia32_pause();
#elif defined(__aarch64__) || defined(__arm__)
	__asm__ __volatile__("yield");
#endif
}

void SpinOrYield() noexcept {
	for (int i = 0; i < 64; ++i) {
		CpuRelax();
	}
	std::this_thread::yield();
}

// ------------------------------------------------------------------------------------- the ring

// Single producer (the GPU thread), single consumer (the recorder thread).
class Ring final {
public:
	static constexpr size_t Bytes    = size_t {32} << 20;
	static constexpr size_t SlotSize = 192; // the replay closure
	// Larger records are not queued: the caller waits for the queue and runs the command itself
	// (Push), which keeps the order.
	static constexpr size_t MaxRecord = Bytes / 4;

	struct Header {
		void (*run)(void* closure);
		uint32_t size;   // whole record, header included; 0: wrap to the start
		uint32_t unused; // keeps the closure 16-byte aligned
	};
	static_assert(sizeof(Header) == 16);

	Ring(): m_data(static_cast<uint8_t*>(::operator new(Bytes, std::align_val_t {64}))) {
		m_thread = std::thread([this] { Run(); });
	}

	// The ring lives until the process ends (the recorder thread is never joined).
	Ring(const Ring&)            = delete;
	Ring& operator=(const Ring&) = delete;
	Ring(Ring&&)                 = delete;
	Ring& operator=(Ring&&)      = delete;
	~Ring()                      = delete;

	// Room a record keeps for a closure of this type, a multiple of 16. Sized to the closure, a
	// command of a few words fills one or two cache lines instead of four: the ring is far larger
	// than the caches, so the GPU thread writes and the recorder thread reads fewer lines that
	// miss.
	template <typename Closure>
	static size_t ClosureRoom() noexcept {
		static_assert(sizeof(Closure) <= SlotSize && alignof(Closure) <= 16);
		constexpr bool compact = true;
		return compact ? (sizeof(Closure) + 15) & ~size_t {15} : SlotSize;
	}

	// Reserves a record with `closure` bytes for the closure (ClosureRoom) and `extra` bytes
	// after it; returns its start.
	uint8_t* Reserve(size_t closure, size_t extra) {
		const size_t size = (sizeof(Header) + closure + extra + 15) & ~size_t {15};
		EXIT_IF(size > MaxRecord);
		for (;;) {
			const auto write = m_write_local;
			auto       read  = m_read_cached;
			if (Bytes - (write - read) < size * 2 + 16) {
				read = m_read_cached = m_read.load(std::memory_order_acquire);
			}
			const auto used = write - read;
			const auto at   = static_cast<size_t>(write % Bytes);
			// A record never wraps: the tail is skipped with a wrap marker.
			const size_t need = (at + size > Bytes) ? (Bytes - at) + size : size;
			if (Bytes - used >= need + 16) {
				if (at + size > Bytes) {
					auto* wrap = reinterpret_cast<Header*>(m_data + at);
					wrap->run  = nullptr;
					wrap->size = 0;
					m_write_local += Bytes - at;
				}
				m_pending_size = size;
				return m_data + (m_write_local % Bytes);
			}
			Kick();
			SpinOrYield();
		}
	}

	template <typename F>
	void Publish(uint8_t* record, F&& closure) {
		using Closure = std::decay_t<F>;
		static_assert(sizeof(Closure) <= SlotSize && alignof(Closure) <= 16);
		static_assert(std::is_trivially_destructible_v<Closure>);
		auto* header = reinterpret_cast<Header*>(record);
		new (record + sizeof(Header)) Closure(std::forward<F>(closure));
		header->run = [](void* at) {
			auto* stored = static_cast<Closure*>(at);
			(*stored)();
		};
		header->size = static_cast<uint32_t>(m_pending_size);
		m_write_local += m_pending_size;
		Common::PerfSummary::Add(Common::PerfSummary::Counter::RecRecords);
		// seq_cst store and load: either the recorder sees the new write position before it
		// sleeps, or this thread sees it sleeping and wakes it.
		m_write.store(m_write_local, std::memory_order_seq_cst);
		if (m_sleeping.load(std::memory_order_seq_cst)) {
			Kick();
		}
	}

	// Waits until every published record ran. Any thread.
	void Drain() {
		const auto target = m_write.load(std::memory_order_acquire);
		if (m_done.load(std::memory_order_acquire) >= target) {
			return;
		}
		Common::PerfSummary::Add(Common::PerfSummary::Counter::RecDrains);
		Common::PerfSummary::ScopedMicros timer(Common::PerfSummary::Counter::RecDrainUs);
		Kick();
		while (m_done.load(std::memory_order_acquire) < target) {
			SpinOrYield();
		}
	}

private:
	void Kick() {
		{
			std::lock_guard lock(m_wake_mutex);
		}
		m_wake.notify_one();
	}

	void Run() {
		KYTY_PROFILER_THREAD("Thread_Recorder");
#if defined(__linux__)
		pthread_setname_np(pthread_self(), "Thread_Recorder");
#endif
		Common::PerfSummary::RegisterRecorderThread();
		// Reading the write position after every record pulls its cache line away from the
		// producer, whose next publish then waits for it back: during a burst the position is
		// read again only once the records seen are done.
		constexpr bool batched = true;
		uint64_t          read    = 0;
		for (;;) {
			auto write = m_write.load(std::memory_order_acquire);
			if (read == write) {
				// Spin a little (the next command of a burst usually follows within microseconds),
				// then sleep until a publish wakes this thread: about 10-30 us of pause loop, so an
				// idle recorder does not take a core from the guest and translator threads.
				bool got = false;
				for (int i = 0; i < 256 && !got; ++i) {
					CpuRelax();
					got = m_write.load(std::memory_order_acquire) != read;
				}
				if (!got) {
					std::unique_lock lock(m_wake_mutex);
					m_sleeping.store(true, std::memory_order_seq_cst);
					if (m_write.load(std::memory_order_seq_cst) == read) {
						// Publish() wakes it (seq_cst handshake on m_sleeping); the timeout is a backstop.
						m_wake.wait_for(lock, std::chrono::milliseconds(50));
					}
					m_sleeping.store(false, std::memory_order_relaxed);
				}
				continue;
			}
			while (read != write) {
				auto* header = reinterpret_cast<Header*>(m_data + (read % Bytes));
				if (header->size == 0) {
					read += Bytes - (read % Bytes);
					continue;
				}
				header->run(reinterpret_cast<uint8_t*>(header) + sizeof(Header));
				read += header->size;
				if ((++m_ran & 31) == 0) {
					m_read.store(read, std::memory_order_release);
					m_done.store(read, std::memory_order_release);
				}
				if (!batched || read == write) {
					write = m_write.load(std::memory_order_acquire);
				}
			}
			m_read.store(read, std::memory_order_release);
			m_done.store(read, std::memory_order_release);
		}
	}

	uint8_t*                m_data;
	std::mutex              m_wake_mutex;
	std::condition_variable m_wake;
	// One cache line per writer: the producer's (written every record), the consumer's (every
	// 32 records) and the sleep flag (written when the consumer sleeps, read every record).
	// Sharing a line made every record of one thread wait for the line the other just wrote.
	alignas(64) std::atomic<uint64_t> m_write {0};
	uint64_t                          m_write_local  = 0; // producer only
	uint64_t                          m_read_cached  = 0; // producer only
	size_t                            m_pending_size = 0; // producer only
	alignas(64) std::atomic<uint64_t> m_read {0};
	std::atomic<uint64_t>             m_done {0};
	uint64_t                          m_ran = 0; // consumer only
	alignas(64) std::atomic<bool>     m_sleeping {false};
	alignas(64) std::thread           m_thread;
};

Ring* g_ring = nullptr;

// Commands run on the calling thread because their record was too large (tests read it).
std::atomic<uint64_t> g_oversized {0};

// The one thread that queues: the first GPU thread that begins a deferred command buffer.
std::atomic<std::thread::id> g_producer {};

bool IsProducer() noexcept {
	if (!GuestGpu::IsGpuThread()) {
		return false;
	}
	const auto self  = std::this_thread::get_id();
	auto       owner = g_producer.load(std::memory_order_acquire);
	if (owner == self) {
		return true;
	}
	if (owner == std::thread::id {}) {
		return g_producer.compare_exchange_strong(owner, self) || owner == self;
	}
	return false;
}

// Open deferred command buffers (Begin on the GPU thread .. End). A batch holds one buffer per
// split segment (a draw waiting for its pipeline) besides the current one, so while new pipelines
// compile there can be many; a buffer begun when every slot is taken records on the calling thread.
constexpr size_t                                  MaxOpen = 256;
std::array<std::atomic<VkCommandBuffer>, MaxOpen> g_open {};
std::atomic<size_t>                               g_open_high {0}; // slots in use are below this
// Grows after every Open and Close: the producer's last queued buffer (Queue) is still open
// while it has not changed.
std::atomic<uint64_t>                             g_open_epoch {0};
thread_local VkCommandBuffer                      t_queued       = VK_NULL_HANDLE;
thread_local uint64_t                             t_queued_epoch = ~uint64_t {0};

bool IsOpen(VkCommandBuffer buffer) noexcept {
	const auto high = g_open_high.load(std::memory_order_acquire);
	for (size_t i = 0; i < high; ++i) {
		if (g_open[i].load(std::memory_order_acquire) == buffer) {
			return true;
		}
	}
	return false;
}

bool Open(VkCommandBuffer buffer) {
	for (size_t i = 0; i < MaxOpen; ++i) {
		VkCommandBuffer expected = VK_NULL_HANDLE;
		if (g_open[i].compare_exchange_strong(expected, buffer)) {
			auto high = g_open_high.load();
			while (high < i + 1 && !g_open_high.compare_exchange_weak(high, i + 1)) {
			}
			g_open_epoch.fetch_add(1, std::memory_order_acq_rel);
			return true;
		}
	}
	return false;
}

void Close(VkCommandBuffer buffer) {
	for (auto& slot: g_open) {
		VkCommandBuffer expected = buffer;
		if (slot.compare_exchange_strong(expected, VK_NULL_HANDLE)) {
			g_open_epoch.fetch_add(1, std::memory_order_acq_rel);
			return;
		}
	}
}

// Buffers the producer began on its own thread because every open slot was taken (producer only).
// While there are any, nothing is queued: all of them come from the same command pools as the
// deferred ones, and a pool must not be used by two threads at once.
std::vector<VkCommandBuffer> g_direct;

void ForgetDirect(VkCommandBuffer buffer) {
	if (g_direct.empty() || !IsProducer()) {
		return;
	}
	std::erase(g_direct, buffer);
	if (g_direct.empty()) {
		g_open_epoch.fetch_add(1, std::memory_order_acq_rel);
	}
}

// KYTY_RECORDER_TOGGLE=<file>: while the file starts with '0', buffers the GPU thread begins are
// recorded and submitted on that thread (as with KYTY_RECORDER=0), so one game run can measure
// both. Read at most every 250 ms, by the GPU thread when it begins a command buffer.
// With KYTY_TOGGLE_FILE, its "recorder" line decides instead (0 = paused, 1 = on; abrun.sh).
bool Paused() {
	static const char* const path = std::getenv("KYTY_RECORDER_TOGGLE");
	if (path == nullptr && !Common::RuntimeToggles::Enabled()) {
		return false;
	}
	static bool paused = false;
	static auto next   = std::chrono::steady_clock::time_point {};
	const auto  now    = std::chrono::steady_clock::now();
	if (now >= next) {
		next = now + std::chrono::milliseconds(250);
		bool value = false;
		if (const int toggle = Common::RuntimeToggles::Find("recorder"); toggle >= 0) {
			value = toggle == 0;
		} else if (path != nullptr) {
			FILE* file = std::fopen(path, "rb");
			if (file != nullptr) {
				value = std::fgetc(file) == '0';
				std::fclose(file);
			}
		}
		if (value != paused) {
			paused = value;
			Log::WriteToConsoleAndLog(paused ? "[recorder] toggle: off\n" : "[recorder] toggle: on\n");
		}
	}
	return paused;
}

// Whether a call on `buffer` is queued. When not, it runs on the calling thread, after the queue
// drained if the buffer is a deferred one used from another thread.
bool Queue(VkCommandBuffer buffer) {
	// Every command of a batch asks again for the same buffer on the same thread: the answer
	// stands until a buffer is opened or closed.
	constexpr bool remember = true;
	const auto        epoch    = g_open_epoch.load(std::memory_order_acquire);
	if (remember && buffer == t_queued && epoch == t_queued_epoch) {
		return true;
	}
	if (g_ring == nullptr) {
		return false;
	}
	if (!IsOpen(buffer)) {
		// A buffer the GPU thread records itself (begun elsewhere, with pNext, or while paused) may
		// come from a pool the recorder is still using for queued commands: let those finish
		// first (a pool must not be used by two threads at once). Free when nothing is queued.
		if (IsProducer()) {
			g_ring->Drain();
		}
		return false;
	}
	if (IsProducer()) {
		if (!g_direct.empty()) {
			g_ring->Drain();
			return false;
		}
		// Read before IsOpen: a Close since then changes the epoch, so the next call asks again.
		t_queued       = buffer;
		t_queued_epoch = epoch;
		return true;
	}
	g_ring->Drain();
	return false;
}

// Before a call on `buffer` that runs on the calling thread.
void DrainIfOpen(VkCommandBuffer buffer) {
	if (g_ring != nullptr && IsOpen(buffer)) {
		g_ring->Drain();
	}
}

// Before destroying or resetting an object a queued command may use (any thread).
void DrainAll() {
	if (g_ring != nullptr) {
		g_ring->Drain();
	}
}

bool AnyOpen() noexcept {
	const auto high = g_open_high.load(std::memory_order_acquire);
	for (size_t i = 0; i < high; ++i) {
		if (g_open[i].load(std::memory_order_acquire) != VK_NULL_HANDLE) {
			return true;
		}
	}
	return false;
}

// Before destroying an object. Only the GPU thread waits: it is the one that records the queued
// commands, while the objects other threads destroy are either their own or released after the
// tick of the work using them completed, which needs the deferred submission to have run. Other
// threads may hold the queue mutex the deferred submission takes, so waiting there could deadlock.
void DrainForDestroy() {
	if (g_ring != nullptr && IsProducer()) {
		g_ring->Drain();
	}
}

// Before command pool and command buffer lifetime calls: also from other threads while a deferred
// buffer is open.
void DrainForPool() {
	if (g_ring != nullptr && (IsProducer() || AnyOpen())) {
		g_ring->Drain();
	}
}

// Copies of the arrays a command points to, placed after the closure in its record.
class Extra final {
public:
	explicit Extra(uint8_t* at): m_at(at) {}
	template <typename T>
	T* Copy(const T* from, size_t count) {
		if (from == nullptr || count == 0) {
			return nullptr;
		}
		m_at       = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(m_at) + alignof(T) - 1) &
		                                        ~(uintptr_t {alignof(T)} - 1));
		auto* copy = reinterpret_cast<T*>(m_at);
		std::memcpy(static_cast<void*>(copy), from, sizeof(T) * count);
		m_at += sizeof(T) * count;
		return copy;
	}

	const void* CopyBytes(const void* from, size_t size, size_t align) {
		m_at = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(m_at) + align - 1) & ~(uintptr_t {align} - 1));
		auto* copy = m_at;
		std::memcpy(copy, from, size);
		m_at += size;
		return copy;
	}

private:
	uint8_t* m_at;
};

template <typename T>
constexpr size_t Room(size_t count) {
	return count == 0 ? 0 : sizeof(T) * count + alignof(T);
}

// Queues the closure `make(fill(extra))` with `extra` bytes of copies filled by `fill(Extra&)`.
// A record that would not fit (Ring::MaxRecord: a command with very long arrays) runs on the
// calling thread instead, once every queued command ran, from copies on the heap.
template <typename Fill, typename Make>
void Push(size_t extra, Fill&& fill, Make&& make) {
	using State   = std::invoke_result_t<Fill&, Extra&>;
	using Closure = std::decay_t<std::invoke_result_t<Make&, State&>>;
	const auto room = Ring::ClosureRoom<Closure>();
	if (sizeof(Ring::Header) + room + extra + 15 > Ring::MaxRecord) {
		g_ring->Drain();
		g_oversized.fetch_add(1, std::memory_order_relaxed);
		struct alignas(16) Block {
			uint8_t bytes[16];
		};
		std::vector<Block> heap((room + extra + 15) / 16);
		auto*              base = reinterpret_cast<uint8_t*>(heap.data());
		Extra              copies(base + room);
		auto               state   = fill(copies);
		auto               closure = make(state);
		closure();
		return;
	}
	auto* record = g_ring->Reserve(room, extra);
	Extra      copies(record + sizeof(Ring::Header) + room);
	auto       state = fill(copies);
	g_ring->Publish(record, make(state));
}

template <typename F>
void PushSimple(F&& run) {
	auto* record = g_ring->Reserve(Ring::ClosureRoom<std::decay_t<F>>(), 0);
	g_ring->Publish(record, std::forward<F>(run));
}

// ------------------------------------------------------------------------- real entry points

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define KYTY_RECORDER_LIFETIME(X)                                                                  \
	X(BeginCommandBuffer)                                                                          \
	X(EndCommandBuffer)                                                                            \
	X(ResetCommandBuffer)                                                                          \
	X(AllocateCommandBuffers)                                                                      \
	X(FreeCommandBuffers)                                                                          \
	X(ResetCommandPool)                                                                            \
	X(DestroyCommandPool)                                                                          \
	X(DestroyBuffer)                                                                               \
	X(DestroyImage)                                                                                \
	X(DestroyImageView)                                                                            \
	X(DestroyBufferView)                                                                           \
	X(DestroySampler)                                                                              \
	X(DestroyPipeline)                                                                             \
	X(DestroyPipelineLayout)                                                                       \
	X(DestroyDescriptorSetLayout)                                                                  \
	X(DestroyDescriptorPool)                                                                       \
	X(ResetDescriptorPool)                                                                         \
	X(DestroyQueryPool)                                                                            \
	X(FreeMemory)                                                                                  \
	X(GetDeviceProcAddr)

// Copied into the ring (optional: hooked when the device has them).
#define KYTY_RECORDER_COMMANDS(X)                                                                  \
	X(CmdBeginConditionalRenderingEXT)                                                             \
	X(CmdEndConditionalRenderingEXT)                                                               \
	X(CmdBeginQuery)                                                                               \
	X(CmdEndQuery)                                                                                 \
	X(CmdWriteTimestamp)                                                                           \
	X(CmdSetCheckpointNV)                                                                          \
	X(CmdBeginRendering)                                                                           \
	X(CmdEndRendering)                                                                             \
	X(CmdBindDescriptorSets)                                                                       \
	X(CmdBindIndexBuffer)                                                                          \
	X(CmdBindPipeline)                                                                             \
	X(CmdBindVertexBuffers2)                                                                       \
	X(CmdBlitImage)                                                                                \
	X(CmdBuildAccelerationStructuresKHR)                                                           \
	X(CmdClearAttachments)                                                                         \
	X(CmdClearColorImage)                                                                          \
	X(CmdClearDepthStencilImage)                                                                   \
	X(CmdCopyBuffer)                                                                               \
	X(CmdCopyBufferToImage)                                                                        \
	X(CmdCopyImage)                                                                                \
	X(CmdCopyImageToBuffer)                                                                        \
	X(CmdDispatch)                                                                                 \
	X(CmdDispatchBase)                                                                             \
	X(CmdDispatchIndirect)                                                                         \
	X(CmdDraw)                                                                                     \
	X(CmdDrawIndexed)                                                                              \
	X(CmdDrawIndexedIndirect)                                                                      \
	X(CmdDrawIndirect)                                                                             \
	X(CmdDrawMeshTasksEXT)                                                                         \
	X(CmdFillBuffer)                                                                               \
	X(CmdPipelineBarrier)                                                                          \
	X(CmdPipelineBarrier2)                                                                         \
	X(CmdPushConstants)                                                                            \
	X(CmdPushDescriptorSetKHR)                                                                     \
	X(CmdResetQueryPool)                                                                           \
	X(CmdResolveImage)                                                                             \
	X(CmdSetAttachmentFeedbackLoopEnableEXT)                                                       \
	X(CmdSetBlendConstants)                                                                        \
	X(CmdSetCullMode)                                                                              \
	X(CmdSetDepthBias)                                                                             \
	X(CmdSetDepthBiasEnable)                                                                       \
	X(CmdSetDepthBounds)                                                                           \
	X(CmdSetDepthBoundsTestEnable)                                                                 \
	X(CmdSetDepthCompareOp)                                                                        \
	X(CmdSetDepthTestEnable)                                                                       \
	X(CmdSetDepthWriteEnable)                                                                      \
	X(CmdSetFrontFace)                                                                             \
	X(CmdSetLineWidth)                                                                             \
	X(CmdSetScissor)                                                                               \
	X(CmdSetScissorWithCount)                                                                      \
	X(CmdSetStencilCompareMask)                                                                    \
	X(CmdSetStencilOp)                                                                             \
	X(CmdSetStencilReference)                                                                      \
	X(CmdSetStencilTestEnable)                                                                     \
	X(CmdSetStencilWriteMask)                                                                      \
	X(CmdSetViewport)                                                                              \
	X(CmdSetViewportWithCount)                                                                     \
	X(CmdUpdateBuffer)                                                                             \
	X(CmdWriteBufferMarkerAMD)

// Aliases Vulkan-Hpp may call instead: (alias, hooked command).
#define KYTY_RECORDER_ALIASES(X)                                                                   \
	X(CmdBeginRenderingKHR, CmdBeginRendering)                                                     \
	X(CmdEndRenderingKHR, CmdEndRendering)                                                         \
	X(CmdPipelineBarrier2KHR, CmdPipelineBarrier2)                                                 \
	X(CmdBindVertexBuffers2EXT, CmdBindVertexBuffers2)                                             \
	X(CmdSetCullModeEXT, CmdSetCullMode)                                                           \
	X(CmdSetFrontFaceEXT, CmdSetFrontFace)                                                         \
	X(CmdSetDepthTestEnableEXT, CmdSetDepthTestEnable)                                             \
	X(CmdSetDepthWriteEnableEXT, CmdSetDepthWriteEnable)                                           \
	X(CmdSetDepthCompareOpEXT, CmdSetDepthCompareOp)                                               \
	X(CmdSetDepthBoundsTestEnableEXT, CmdSetDepthBoundsTestEnable)                                 \
	X(CmdSetStencilTestEnableEXT, CmdSetStencilTestEnable)                                         \
	X(CmdSetStencilOpEXT, CmdSetStencilOp)                                                         \
	X(CmdSetDepthBiasEnableEXT, CmdSetDepthBiasEnable)                                             \
	X(CmdSetViewportWithCountEXT, CmdSetViewportWithCount)                                         \
	X(CmdSetScissorWithCountEXT, CmdSetScissorWithCount)                                           \
	X(CmdDispatchBaseKHR, CmdDispatchBase)                                                         \
	X(CmdPushDescriptorSet, CmdPushDescriptorSetKHR)

// Not copied: they wait for the queue and run on the calling thread, so that a later renderer
// change calling one of them keeps the command order. Every vkCmd* of vulkan_core.h that is not
// copied above is listed here (a command missing from both lists would run on the calling thread
// while the recorder thread may be recording the same buffer).
#define KYTY_RECORDER_FALLBACKS(X)                                                                 \
	X(CmdBeginCustomResolveEXT)                                                                       \
	X(CmdBeginDebugUtilsLabelEXT)                                                                     \
	X(CmdBeginPerTileExecutionQCOM)                                                                   \
	X(CmdBeginQueryIndexedEXT)                                                                        \
	X(CmdBeginRenderPass)                                                                             \
	X(CmdBeginRenderPass2)                                                                            \
	X(CmdBeginRenderPass2KHR)                                                                         \
	X(CmdBeginTransformFeedbackEXT)                                                                   \
	X(CmdBeginVideoCodingKHR)                                                                         \
	X(CmdBindDescriptorBufferEmbeddedSamplers2EXT)                                                    \
	X(CmdBindDescriptorBufferEmbeddedSamplersEXT)                                                     \
	X(CmdBindDescriptorBuffersEXT)                                                                    \
	X(CmdBindDescriptorSets2)                                                                         \
	X(CmdBindDescriptorSets2KHR)                                                                      \
	X(CmdBindIndexBuffer2)                                                                            \
	X(CmdBindIndexBuffer2KHR)                                                                         \
	X(CmdBindInvocationMaskHUAWEI)                                                                    \
	X(CmdBindPipelineShaderGroupNV)                                                                   \
	X(CmdBindShadersEXT)                                                                              \
	X(CmdBindShadingRateImageNV)                                                                      \
	X(CmdBindTileMemoryQCOM)                                                                          \
	X(CmdBindTransformFeedbackBuffersEXT)                                                             \
	X(CmdBindVertexBuffers)                                                                           \
	X(CmdBlitImage2)                                                                                  \
	X(CmdBlitImage2KHR)                                                                               \
	X(CmdBuildAccelerationStructureNV)                                                                \
	X(CmdBuildAccelerationStructuresIndirectKHR)                                                      \
	X(CmdBuildClusterAccelerationStructureIndirectNV)                                                 \
	X(CmdBuildMicromapsEXT)                                                                           \
	X(CmdBuildPartitionedAccelerationStructuresNV)                                                    \
	X(CmdControlVideoCodingKHR)                                                                       \
	X(CmdConvertCooperativeVectorMatrixNV)                                                            \
	X(CmdCopyAccelerationStructureKHR)                                                                \
	X(CmdCopyAccelerationStructureNV)                                                                 \
	X(CmdCopyAccelerationStructureToMemoryKHR)                                                        \
	X(CmdCopyBuffer2)                                                                                 \
	X(CmdCopyBuffer2KHR)                                                                              \
	X(CmdCopyBufferToImage2)                                                                          \
	X(CmdCopyBufferToImage2KHR)                                                                       \
	X(CmdCopyImage2)                                                                                  \
	X(CmdCopyImage2KHR)                                                                               \
	X(CmdCopyImageToBuffer2)                                                                          \
	X(CmdCopyImageToBuffer2KHR)                                                                       \
	X(CmdCopyMemoryIndirectKHR)                                                                       \
	X(CmdCopyMemoryIndirectNV)                                                                        \
	X(CmdCopyMemoryToAccelerationStructureKHR)                                                        \
	X(CmdCopyMemoryToImageIndirectKHR)                                                                \
	X(CmdCopyMemoryToImageIndirectNV)                                                                 \
	X(CmdCopyMemoryToMicromapEXT)                                                                     \
	X(CmdCopyMicromapEXT)                                                                             \
	X(CmdCopyMicromapToMemoryEXT)                                                                     \
	X(CmdCopyQueryPoolResults)                                                                        \
	X(CmdCopyTensorARM)                                                                               \
	X(CmdCuLaunchKernelNVX)                                                                           \
	X(CmdDebugMarkerBeginEXT)                                                                         \
	X(CmdDebugMarkerEndEXT)                                                                           \
	X(CmdDebugMarkerInsertEXT)                                                                        \
	X(CmdDecodeVideoKHR)                                                                              \
	X(CmdDecompressMemoryEXT)                                                                         \
	X(CmdDecompressMemoryIndirectCountEXT)                                                            \
	X(CmdDecompressMemoryIndirectCountNV)                                                             \
	X(CmdDecompressMemoryNV)                                                                          \
	X(CmdDispatchDataGraphARM)                                                                        \
	X(CmdDispatchTileQCOM)                                                                            \
	X(CmdDrawClusterHUAWEI)                                                                           \
	X(CmdDrawClusterIndirectHUAWEI)                                                                   \
	X(CmdDrawIndexedIndirectCount)                                                                    \
	X(CmdDrawIndexedIndirectCountAMD)                                                                 \
	X(CmdDrawIndexedIndirectCountKHR)                                                                 \
	X(CmdDrawIndirectByteCountEXT)                                                                    \
	X(CmdDrawIndirectCount)                                                                           \
	X(CmdDrawIndirectCountAMD)                                                                        \
	X(CmdDrawIndirectCountKHR)                                                                        \
	X(CmdDrawMeshTasksIndirectCountEXT)                                                               \
	X(CmdDrawMeshTasksIndirectCountNV)                                                                \
	X(CmdDrawMeshTasksIndirectEXT)                                                                    \
	X(CmdDrawMeshTasksIndirectNV)                                                                     \
	X(CmdDrawMeshTasksNV)                                                                             \
	X(CmdDrawMultiEXT)                                                                                \
	X(CmdDrawMultiIndexedEXT)                                                                         \
	X(CmdEncodeVideoKHR)                                                                              \
	X(CmdEndDebugUtilsLabelEXT)                                                                       \
	X(CmdEndPerTileExecutionQCOM)                                                                     \
	X(CmdEndQueryIndexedEXT)                                                                          \
	X(CmdEndRenderPass)                                                                               \
	X(CmdEndRenderPass2)                                                                              \
	X(CmdEndRenderPass2KHR)                                                                           \
	X(CmdEndRendering2EXT)                                                                            \
	X(CmdEndRendering2KHR)                                                                            \
	X(CmdEndTransformFeedbackEXT)                                                                     \
	X(CmdEndVideoCodingKHR)                                                                           \
	X(CmdExecuteCommands)                                                                             \
	X(CmdExecuteGeneratedCommandsEXT)                                                                 \
	X(CmdExecuteGeneratedCommandsNV)                                                                  \
	X(CmdInsertDebugUtilsLabelEXT)                                                                    \
	X(CmdNextSubpass)                                                                                 \
	X(CmdNextSubpass2)                                                                                \
	X(CmdNextSubpass2KHR)                                                                             \
	X(CmdOpticalFlowExecuteNV)                                                                        \
	X(CmdPreprocessGeneratedCommandsEXT)                                                              \
	X(CmdPreprocessGeneratedCommandsNV)                                                               \
	X(CmdPushConstants2)                                                                              \
	X(CmdPushConstants2KHR)                                                                           \
	X(CmdPushDescriptorSet2)                                                                          \
	X(CmdPushDescriptorSet2KHR)                                                                       \
	X(CmdPushDescriptorSetWithTemplate)                                                               \
	X(CmdPushDescriptorSetWithTemplate2)                                                              \
	X(CmdPushDescriptorSetWithTemplate2KHR)                                                           \
	X(CmdPushDescriptorSetWithTemplateKHR)                                                            \
	X(CmdResetEvent)                                                                                  \
	X(CmdResetEvent2)                                                                                 \
	X(CmdResetEvent2KHR)                                                                              \
	X(CmdResolveImage2)                                                                               \
	X(CmdResolveImage2KHR)                                                                            \
	X(CmdSetAlphaToCoverageEnableEXT)                                                                 \
	X(CmdSetAlphaToOneEnableEXT)                                                                      \
	X(CmdSetCoarseSampleOrderNV)                                                                      \
	X(CmdSetColorBlendAdvancedEXT)                                                                    \
	X(CmdSetColorBlendEnableEXT)                                                                      \
	X(CmdSetColorBlendEquationEXT)                                                                    \
	X(CmdSetColorWriteEnableEXT)                                                                      \
	X(CmdSetColorWriteMaskEXT)                                                                        \
	X(CmdSetConservativeRasterizationModeEXT)                                                         \
	X(CmdSetCoverageModulationModeNV)                                                                 \
	X(CmdSetCoverageModulationTableEnableNV)                                                          \
	X(CmdSetCoverageModulationTableNV)                                                                \
	X(CmdSetCoverageReductionModeNV)                                                                  \
	X(CmdSetCoverageToColorEnableNV)                                                                  \
	X(CmdSetCoverageToColorLocationNV)                                                                \
	X(CmdSetDepthBias2EXT)                                                                            \
	X(CmdSetDepthClampEnableEXT)                                                                      \
	X(CmdSetDepthClampRangeEXT)                                                                       \
	X(CmdSetDepthClipEnableEXT)                                                                       \
	X(CmdSetDepthClipNegativeOneToOneEXT)                                                             \
	X(CmdSetDescriptorBufferOffsets2EXT)                                                              \
	X(CmdSetDescriptorBufferOffsetsEXT)                                                               \
	X(CmdSetDeviceMask)                                                                               \
	X(CmdSetDeviceMaskKHR)                                                                            \
	X(CmdSetDiscardRectangleEXT)                                                                      \
	X(CmdSetDiscardRectangleEnableEXT)                                                                \
	X(CmdSetDiscardRectangleModeEXT)                                                                  \
	X(CmdSetEvent)                                                                                    \
	X(CmdSetEvent2)                                                                                   \
	X(CmdSetEvent2KHR)                                                                                \
	X(CmdSetExclusiveScissorEnableNV)                                                                 \
	X(CmdSetExclusiveScissorNV)                                                                       \
	X(CmdSetExtraPrimitiveOverestimationSizeEXT)                                                      \
	X(CmdSetFragmentShadingRateEnumNV)                                                                \
	X(CmdSetFragmentShadingRateKHR)                                                                   \
	X(CmdSetLineRasterizationModeEXT)                                                                 \
	X(CmdSetLineStipple)                                                                              \
	X(CmdSetLineStippleEXT)                                                                           \
	X(CmdSetLineStippleEnableEXT)                                                                     \
	X(CmdSetLineStippleKHR)                                                                           \
	X(CmdSetLogicOpEXT)                                                                               \
	X(CmdSetLogicOpEnableEXT)                                                                         \
	X(CmdSetPatchControlPointsEXT)                                                                    \
	X(CmdSetPerformanceMarkerINTEL)                                                                   \
	X(CmdSetPerformanceOverrideINTEL)                                                                 \
	X(CmdSetPerformanceStreamMarkerINTEL)                                                             \
	X(CmdSetPolygonModeEXT)                                                                           \
	X(CmdSetPrimitiveRestartEnable)                                                                   \
	X(CmdSetPrimitiveRestartEnableEXT)                                                                \
	X(CmdSetPrimitiveTopology)                                                                        \
	X(CmdSetPrimitiveTopologyEXT)                                                                     \
	X(CmdSetProvokingVertexModeEXT)                                                                   \
	X(CmdSetRasterizationSamplesEXT)                                                                  \
	X(CmdSetRasterizationStreamEXT)                                                                   \
	X(CmdSetRasterizerDiscardEnable)                                                                  \
	X(CmdSetRasterizerDiscardEnableEXT)                                                               \
	X(CmdSetRayTracingPipelineStackSizeKHR)                                                           \
	X(CmdSetRenderingAttachmentLocations)                                                             \
	X(CmdSetRenderingAttachmentLocationsKHR)                                                          \
	X(CmdSetRenderingInputAttachmentIndices)                                                          \
	X(CmdSetRenderingInputAttachmentIndicesKHR)                                                       \
	X(CmdSetRepresentativeFragmentTestEnableNV)                                                       \
	X(CmdSetSampleLocationsEXT)                                                                       \
	X(CmdSetSampleLocationsEnableEXT)                                                                 \
	X(CmdSetSampleMaskEXT)                                                                            \
	X(CmdSetShadingRateImageEnableNV)                                                                 \
	X(CmdSetTessellationDomainOriginEXT)                                                              \
	X(CmdSetVertexInputEXT)                                                                           \
	X(CmdSetViewportShadingRatePaletteNV)                                                             \
	X(CmdSetViewportSwizzleNV)                                                                        \
	X(CmdSetViewportWScalingEnableNV)                                                                 \
	X(CmdSetViewportWScalingNV)                                                                       \
	X(CmdSubpassShadingHUAWEI)                                                                        \
	X(CmdTraceRaysIndirect2KHR)                                                                       \
	X(CmdTraceRaysIndirectKHR)                                                                        \
	X(CmdTraceRaysKHR)                                                                                \
	X(CmdTraceRaysNV)                                                                                 \
	X(CmdUpdatePipelineIndirectBufferNV)                                                              \
	X(CmdWaitEvents)                                                                                  \
	X(CmdWaitEvents2)                                                                                 \
	X(CmdWaitEvents2KHR)                                                                              \
	X(CmdWriteAccelerationStructuresPropertiesKHR)                                                    \
	X(CmdWriteAccelerationStructuresPropertiesNV)                                                     \
	X(CmdWriteBufferMarker2AMD)                                                                       \
	X(CmdWriteMicromapsPropertiesEXT)                                                                 \
	X(CmdWriteTimestamp2)                                                                             \
	X(CmdWriteTimestamp2KHR)

struct Real {
#define KYTY_RECORDER_REAL(name) PFN_vk##name vk##name = nullptr;
	KYTY_RECORDER_LIFETIME(KYTY_RECORDER_REAL)
	KYTY_RECORDER_COMMANDS(KYTY_RECORDER_REAL)
	KYTY_RECORDER_REAL(DestroyAccelerationStructureKHR)
#undef KYTY_RECORDER_REAL
};
Real g_real;

// Real entry points of the fallback commands, one variable each (template arguments below).
#define KYTY_RECORDER_FALLBACK_REAL(name) PFN_vk##name g_fallback_##name = nullptr;
KYTY_RECORDER_FALLBACKS(KYTY_RECORDER_FALLBACK_REAL)
#undef KYTY_RECORDER_FALLBACK_REAL

template <typename Pfn>
struct Fallback;
template <typename R, typename... Args>
struct Fallback<R(VKAPI_PTR*)(VkCommandBuffer, Args...)> {
	using Pfn = R(VKAPI_PTR*)(VkCommandBuffer, Args...);
	template <Pfn* RealFunction>
	static R VKAPI_CALL Call(VkCommandBuffer buffer, Args... args) {
		DrainIfOpen(buffer);
		return (*RealFunction)(buffer, args...);
	}
};

// ----------------------------------------------------------------------------- command buffers

VKAPI_ATTR VkResult VKAPI_CALL BeginCommandBuffer(VkCommandBuffer buffer, const VkCommandBufferBeginInfo* info) {
	if (IsOpen(buffer)) {
		// Begun again without End (an implicit reset).
		g_ring->Drain();
		Close(buffer);
	}
	if (info->pNext != nullptr || info->pInheritanceInfo != nullptr || !IsProducer()) {
		DrainForDestroy(); // the producer only: same pool rule as in Queue
		return g_real.vkBeginCommandBuffer(buffer, info);
	}
	const bool paused = Paused();
	if (paused || !g_direct.empty() || !Open(buffer)) {
		// Recorded on this thread until every such buffer ended; the cached answer of Queue() is
		// dropped through the epoch.
		g_ring->Drain();
		if (g_direct.empty() && !paused) {
			Log::WriteToConsoleAndLog("[recorder] more than 256 open command buffers: recording on the GPU thread "
			                          "until they are submitted\n");
		}
		g_direct.push_back(buffer);
		g_open_epoch.fetch_add(1, std::memory_order_acq_rel);
		return g_real.vkBeginCommandBuffer(buffer, info);
	}
	const auto flags = info->flags;
	PushSimple([buffer, flags] {
		VkCommandBufferBeginInfo begin {};
		begin.sType       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin.flags       = flags;
		const auto result = g_real.vkBeginCommandBuffer(buffer, &begin);
		EXIT_IF(result != VK_SUCCESS);
	});
	return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL EndCommandBuffer(VkCommandBuffer buffer) {
	if (IsOpen(buffer)) {
		g_ring->Drain();
		Close(buffer);
	}
	ForgetDirect(buffer);
	DrainForDestroy(); // the producer only: same pool rule as in Queue
	return g_real.vkEndCommandBuffer(buffer);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandBuffer(VkCommandBuffer buffer, VkCommandBufferResetFlags flags) {
	DrainAll();
	Close(buffer);
	ForgetDirect(buffer);
	return g_real.vkResetCommandBuffer(buffer, flags);
}

VKAPI_ATTR VkResult VKAPI_CALL AllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* info,
                                                      VkCommandBuffer* buffers) {
	DrainForPool();
	return g_real.vkAllocateCommandBuffers(device, info, buffers);
}

VKAPI_ATTR void VKAPI_CALL FreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t count,
                                              const VkCommandBuffer* buffers) {
	DrainForPool();
	for (uint32_t i = 0; i < count; ++i) {
		Close(buffers[i]);
		ForgetDirect(buffers[i]);
	}
	g_real.vkFreeCommandBuffers(device, pool, count, buffers);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetCommandPool(VkDevice device, VkCommandPool pool, VkCommandPoolResetFlags flags) {
	DrainForPool();
	return g_real.vkResetCommandPool(device, pool, flags);
}

VKAPI_ATTR VkResult VKAPI_CALL ResetDescriptorPool(VkDevice device, VkDescriptorPool pool,
                                                   VkDescriptorPoolResetFlags flags) {
	DrainForDestroy();
	return g_real.vkResetDescriptorPool(device, pool, flags);
}

VKAPI_ATTR void VKAPI_CALL DestroyCommandPool(VkDevice device, VkCommandPool pool, const VkAllocationCallbacks* a) {
	DrainForPool();
	g_real.vkDestroyCommandPool(device, pool, a);
}

#define KYTY_RECORDER_DESTROY(name, Type)                                                          \
	VKAPI_ATTR void VKAPI_CALL name(VkDevice device, Type object, const VkAllocationCallbacks* a) { \
		DrainForDestroy();                                                                         \
		g_real.vk##name(device, object, a);                                                        \
	}
KYTY_RECORDER_DESTROY(DestroyBuffer, VkBuffer)
KYTY_RECORDER_DESTROY(DestroyImage, VkImage)
KYTY_RECORDER_DESTROY(DestroyImageView, VkImageView)
KYTY_RECORDER_DESTROY(DestroyBufferView, VkBufferView)
KYTY_RECORDER_DESTROY(DestroySampler, VkSampler)
KYTY_RECORDER_DESTROY(DestroyPipeline, VkPipeline)
KYTY_RECORDER_DESTROY(DestroyPipelineLayout, VkPipelineLayout)
KYTY_RECORDER_DESTROY(DestroyDescriptorSetLayout, VkDescriptorSetLayout)
KYTY_RECORDER_DESTROY(DestroyDescriptorPool, VkDescriptorPool)
KYTY_RECORDER_DESTROY(DestroyQueryPool, VkQueryPool)
KYTY_RECORDER_DESTROY(DestroyAccelerationStructureKHR, VkAccelerationStructureKHR)
KYTY_RECORDER_DESTROY(FreeMemory, VkDeviceMemory)
#undef KYTY_RECORDER_DESTROY

// ------------------------------------------------------------------------ commands with scalars

#define KYTY_RECORDER_P(...) (__VA_ARGS__)
#define KYTY_RECORDER_C(...) __VA_ARGS__
#define KYTY_RECORDER_SCALAR(name, params, args, captures)                                         \
	VKAPI_ATTR void VKAPI_CALL name params {                                                       \
		if (!Queue(buffer)) {                                                                      \
			g_real.vk##name args;                                                                  \
			return;                                                                                \
		}                                                                                          \
		PushSimple([captures] { g_real.vk##name args; });                                          \
	}

KYTY_RECORDER_SCALAR(CmdEndConditionalRenderingEXT, KYTY_RECORDER_P(VkCommandBuffer buffer),
                     KYTY_RECORDER_P(buffer), KYTY_RECORDER_C(buffer))
KYTY_RECORDER_SCALAR(CmdBeginQuery,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkQueryPool pool, uint32_t query,
                                     VkQueryControlFlags flags),
                     KYTY_RECORDER_P(buffer, pool, query, flags), KYTY_RECORDER_C(buffer, pool, query, flags))
KYTY_RECORDER_SCALAR(CmdEndQuery, KYTY_RECORDER_P(VkCommandBuffer buffer, VkQueryPool pool, uint32_t query),
                     KYTY_RECORDER_P(buffer, pool, query), KYTY_RECORDER_C(buffer, pool, query))
KYTY_RECORDER_SCALAR(CmdResetQueryPool,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkQueryPool pool, uint32_t first, uint32_t count),
                     KYTY_RECORDER_P(buffer, pool, first, count), KYTY_RECORDER_C(buffer, pool, first, count))
KYTY_RECORDER_SCALAR(CmdWriteBufferMarkerAMD,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkPipelineStageFlagBits stage, VkBuffer dst,
                                     VkDeviceSize offset, uint32_t marker),
                     KYTY_RECORDER_P(buffer, stage, dst, offset, marker),
                     KYTY_RECORDER_C(buffer, stage, dst, offset, marker))
KYTY_RECORDER_SCALAR(CmdEndRendering, KYTY_RECORDER_P(VkCommandBuffer buffer), KYTY_RECORDER_P(buffer),
                     KYTY_RECORDER_C(buffer))
// The marker is an opaque pointer value: copied as such.
KYTY_RECORDER_SCALAR(CmdSetCheckpointNV, KYTY_RECORDER_P(VkCommandBuffer buffer, const void* marker),
                     KYTY_RECORDER_P(buffer, marker), KYTY_RECORDER_C(buffer, marker))
KYTY_RECORDER_SCALAR(CmdWriteTimestamp,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkPipelineStageFlagBits stage, VkQueryPool pool,
                                     uint32_t query),
                     KYTY_RECORDER_P(buffer, stage, pool, query), KYTY_RECORDER_C(buffer, stage, pool, query))
KYTY_RECORDER_SCALAR(CmdBindIndexBuffer,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkBuffer index, VkDeviceSize offset, VkIndexType type),
                     KYTY_RECORDER_P(buffer, index, offset, type), KYTY_RECORDER_C(buffer, index, offset, type))
KYTY_RECORDER_SCALAR(CmdBindPipeline,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkPipelineBindPoint point, VkPipeline pipeline),
                     KYTY_RECORDER_P(buffer, point, pipeline), KYTY_RECORDER_C(buffer, point, pipeline))
KYTY_RECORDER_SCALAR(CmdDispatch, KYTY_RECORDER_P(VkCommandBuffer buffer, uint32_t x, uint32_t y, uint32_t z),
                     KYTY_RECORDER_P(buffer, x, y, z), KYTY_RECORDER_C(buffer, x, y, z))
KYTY_RECORDER_SCALAR(CmdDispatchBase,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, uint32_t bx, uint32_t by, uint32_t bz, uint32_t x,
                                     uint32_t y, uint32_t z),
                     KYTY_RECORDER_P(buffer, bx, by, bz, x, y, z), KYTY_RECORDER_C(buffer, bx, by, bz, x, y, z))
KYTY_RECORDER_SCALAR(CmdDispatchIndirect, KYTY_RECORDER_P(VkCommandBuffer buffer, VkBuffer args, VkDeviceSize offset),
                     KYTY_RECORDER_P(buffer, args, offset), KYTY_RECORDER_C(buffer, args, offset))
KYTY_RECORDER_SCALAR(CmdDraw,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, uint32_t vertices, uint32_t instances,
                                     uint32_t first_vertex, uint32_t first_instance),
                     KYTY_RECORDER_P(buffer, vertices, instances, first_vertex, first_instance),
                     KYTY_RECORDER_C(buffer, vertices, instances, first_vertex, first_instance))
KYTY_RECORDER_SCALAR(CmdDrawIndexed,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, uint32_t indices, uint32_t instances, uint32_t first_index,
                                     int32_t vertex_offset, uint32_t first_instance),
                     KYTY_RECORDER_P(buffer, indices, instances, first_index, vertex_offset, first_instance),
                     KYTY_RECORDER_C(buffer, indices, instances, first_index, vertex_offset, first_instance))
KYTY_RECORDER_SCALAR(CmdDrawIndexedIndirect,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkBuffer args, VkDeviceSize offset, uint32_t count,
                                     uint32_t stride),
                     KYTY_RECORDER_P(buffer, args, offset, count, stride),
                     KYTY_RECORDER_C(buffer, args, offset, count, stride))
KYTY_RECORDER_SCALAR(CmdDrawIndirect,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkBuffer args, VkDeviceSize offset, uint32_t count,
                                     uint32_t stride),
                     KYTY_RECORDER_P(buffer, args, offset, count, stride),
                     KYTY_RECORDER_C(buffer, args, offset, count, stride))
KYTY_RECORDER_SCALAR(CmdDrawMeshTasksEXT, KYTY_RECORDER_P(VkCommandBuffer buffer, uint32_t x, uint32_t y, uint32_t z),
                     KYTY_RECORDER_P(buffer, x, y, z), KYTY_RECORDER_C(buffer, x, y, z))
KYTY_RECORDER_SCALAR(CmdFillBuffer,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                                     uint32_t data),
                     KYTY_RECORDER_P(buffer, dst, offset, size, data), KYTY_RECORDER_C(buffer, dst, offset, size, data))
KYTY_RECORDER_SCALAR(CmdSetAttachmentFeedbackLoopEnableEXT, KYTY_RECORDER_P(VkCommandBuffer buffer, VkImageAspectFlags mask),
                     KYTY_RECORDER_P(buffer, mask), KYTY_RECORDER_C(buffer, mask))
KYTY_RECORDER_SCALAR(CmdSetCullMode, KYTY_RECORDER_P(VkCommandBuffer buffer, VkCullModeFlags mode),
                     KYTY_RECORDER_P(buffer, mode), KYTY_RECORDER_C(buffer, mode))
KYTY_RECORDER_SCALAR(CmdSetDepthBias, KYTY_RECORDER_P(VkCommandBuffer buffer, float constant, float clamp, float slope),
                     KYTY_RECORDER_P(buffer, constant, clamp, slope), KYTY_RECORDER_C(buffer, constant, clamp, slope))
KYTY_RECORDER_SCALAR(CmdSetDepthBiasEnable, KYTY_RECORDER_P(VkCommandBuffer buffer, VkBool32 enable),
                     KYTY_RECORDER_P(buffer, enable), KYTY_RECORDER_C(buffer, enable))
KYTY_RECORDER_SCALAR(CmdSetDepthBounds, KYTY_RECORDER_P(VkCommandBuffer buffer, float low, float high),
                     KYTY_RECORDER_P(buffer, low, high), KYTY_RECORDER_C(buffer, low, high))
KYTY_RECORDER_SCALAR(CmdSetDepthBoundsTestEnable, KYTY_RECORDER_P(VkCommandBuffer buffer, VkBool32 enable),
                     KYTY_RECORDER_P(buffer, enable), KYTY_RECORDER_C(buffer, enable))
KYTY_RECORDER_SCALAR(CmdSetDepthCompareOp, KYTY_RECORDER_P(VkCommandBuffer buffer, VkCompareOp op),
                     KYTY_RECORDER_P(buffer, op), KYTY_RECORDER_C(buffer, op))
KYTY_RECORDER_SCALAR(CmdSetDepthTestEnable, KYTY_RECORDER_P(VkCommandBuffer buffer, VkBool32 enable),
                     KYTY_RECORDER_P(buffer, enable), KYTY_RECORDER_C(buffer, enable))
KYTY_RECORDER_SCALAR(CmdSetDepthWriteEnable, KYTY_RECORDER_P(VkCommandBuffer buffer, VkBool32 enable),
                     KYTY_RECORDER_P(buffer, enable), KYTY_RECORDER_C(buffer, enable))
KYTY_RECORDER_SCALAR(CmdSetFrontFace, KYTY_RECORDER_P(VkCommandBuffer buffer, VkFrontFace face),
                     KYTY_RECORDER_P(buffer, face), KYTY_RECORDER_C(buffer, face))
KYTY_RECORDER_SCALAR(CmdSetLineWidth, KYTY_RECORDER_P(VkCommandBuffer buffer, float width), KYTY_RECORDER_P(buffer, width),
                     KYTY_RECORDER_C(buffer, width))
KYTY_RECORDER_SCALAR(CmdSetStencilCompareMask, KYTY_RECORDER_P(VkCommandBuffer buffer, VkStencilFaceFlags faces, uint32_t mask),
                     KYTY_RECORDER_P(buffer, faces, mask), KYTY_RECORDER_C(buffer, faces, mask))
KYTY_RECORDER_SCALAR(CmdSetStencilOp,
                     KYTY_RECORDER_P(VkCommandBuffer buffer, VkStencilFaceFlags faces, VkStencilOp fail, VkStencilOp pass,
                                     VkStencilOp depth_fail, VkCompareOp op),
                     KYTY_RECORDER_P(buffer, faces, fail, pass, depth_fail, op),
                     KYTY_RECORDER_C(buffer, faces, fail, pass, depth_fail, op))
KYTY_RECORDER_SCALAR(CmdSetStencilReference, KYTY_RECORDER_P(VkCommandBuffer buffer, VkStencilFaceFlags faces, uint32_t reference),
                     KYTY_RECORDER_P(buffer, faces, reference), KYTY_RECORDER_C(buffer, faces, reference))
KYTY_RECORDER_SCALAR(CmdSetStencilTestEnable, KYTY_RECORDER_P(VkCommandBuffer buffer, VkBool32 enable),
                     KYTY_RECORDER_P(buffer, enable), KYTY_RECORDER_C(buffer, enable))
KYTY_RECORDER_SCALAR(CmdSetStencilWriteMask, KYTY_RECORDER_P(VkCommandBuffer buffer, VkStencilFaceFlags faces, uint32_t mask),
                     KYTY_RECORDER_P(buffer, faces, mask), KYTY_RECORDER_C(buffer, faces, mask))

#undef KYTY_RECORDER_SCALAR
#undef KYTY_RECORDER_P
#undef KYTY_RECORDER_C

// ------------------------------------------------------------------------- commands with arrays

VKAPI_ATTR void VKAPI_CALL CmdSetBlendConstants(VkCommandBuffer buffer, const float constants[4]) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetBlendConstants(buffer, constants);
		return;
	}
	const std::array<float, 4> copy {constants[0], constants[1], constants[2], constants[3]};
	PushSimple([buffer, copy] { g_real.vkCmdSetBlendConstants(buffer, copy.data()); });
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewport(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                          const VkViewport* viewports) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetViewport(buffer, first, count, viewports);
		return;
	}
	Push(
	    Room<VkViewport>(count), [&](Extra& extra) { return extra.Copy(viewports, count); },
	    [&](const VkViewport* copy) {
		    return [buffer, first, count, copy] { g_real.vkCmdSetViewport(buffer, first, count, copy); };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissor(VkCommandBuffer buffer, uint32_t first, uint32_t count, const VkRect2D* rects) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetScissor(buffer, first, count, rects);
		return;
	}
	Push(
	    Room<VkRect2D>(count), [&](Extra& extra) { return extra.Copy(rects, count); },
	    [&](const VkRect2D* copy) {
		    return [buffer, first, count, copy] { g_real.vkCmdSetScissor(buffer, first, count, copy); };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdSetViewportWithCount(VkCommandBuffer buffer, uint32_t count, const VkViewport* viewports) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetViewportWithCount(buffer, count, viewports);
		return;
	}
	Push(
	    Room<VkViewport>(count), [&](Extra& extra) { return extra.Copy(viewports, count); },
	    [&](const VkViewport* copy) {
		    return [buffer, count, copy] { g_real.vkCmdSetViewportWithCount(buffer, count, copy); };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdSetScissorWithCount(VkCommandBuffer buffer, uint32_t count, const VkRect2D* rects) {
	if (!Queue(buffer)) {
		g_real.vkCmdSetScissorWithCount(buffer, count, rects);
		return;
	}
	Push(
	    Room<VkRect2D>(count), [&](Extra& extra) { return extra.Copy(rects, count); },
	    [&](const VkRect2D* copy) {
		    return [buffer, count, copy] { g_real.vkCmdSetScissorWithCount(buffer, count, copy); };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdPushConstants(VkCommandBuffer buffer, VkPipelineLayout layout, VkShaderStageFlags stages,
                                            uint32_t offset, uint32_t size, const void* values) {
	if (!Queue(buffer)) {
		g_real.vkCmdPushConstants(buffer, layout, stages, offset, size, values);
		return;
	}
	Push(
	    Room<uint8_t>(size), [&](Extra& extra) { return extra.Copy(static_cast<const uint8_t*>(values), size); },
	    [&](const uint8_t* copy) {
		    return [buffer, layout, stages, offset, size, copy] {
			    g_real.vkCmdPushConstants(buffer, layout, stages, offset, size, copy);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdUpdateBuffer(VkCommandBuffer buffer, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                                           const void* data) {
	if (!Queue(buffer)) {
		g_real.vkCmdUpdateBuffer(buffer, dst, offset, size, data);
		return;
	}
	const auto bytes = static_cast<size_t>(size);
	Push(
	    Room<uint8_t>(bytes), [&](Extra& extra) { return extra.Copy(static_cast<const uint8_t*>(data), bytes); },
	    [&](const uint8_t* copy) {
		    return [buffer, dst, offset, size, copy] { g_real.vkCmdUpdateBuffer(buffer, dst, offset, size, copy); };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdBindDescriptorSets(VkCommandBuffer buffer, VkPipelineBindPoint point, VkPipelineLayout layout,
                                                 uint32_t first, uint32_t count, const VkDescriptorSet* sets,
                                                 uint32_t dynamic_count, const uint32_t* dynamic) {
	if (!Queue(buffer)) {
		g_real.vkCmdBindDescriptorSets(buffer, point, layout, first, count, sets, dynamic_count, dynamic);
		return;
	}
	struct State {
		const VkDescriptorSet* sets;
		const uint32_t*        dynamic;
	};
	Push(
	    Room<VkDescriptorSet>(count) + Room<uint32_t>(dynamic_count),
	    [&](Extra& extra) { return State {extra.Copy(sets, count), extra.Copy(dynamic, dynamic_count)}; },
	    [&](const State& s) {
		    return [buffer, point, layout, first, count, dynamic_count, s] {
			    g_real.vkCmdBindDescriptorSets(buffer, point, layout, first, count, s.sets, dynamic_count, s.dynamic);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdBindVertexBuffers2(VkCommandBuffer buffer, uint32_t first, uint32_t count,
                                                 const VkBuffer* buffers, const VkDeviceSize* offsets,
                                                 const VkDeviceSize* sizes, const VkDeviceSize* strides) {
	if (!Queue(buffer)) {
		g_real.vkCmdBindVertexBuffers2(buffer, first, count, buffers, offsets, sizes, strides);
		return;
	}
	struct State {
		const VkBuffer*     buffers;
		const VkDeviceSize* offsets;
		const VkDeviceSize* sizes;
		const VkDeviceSize* strides;
	};
	Push(
	    Room<VkBuffer>(count) + 3 * Room<VkDeviceSize>(count),
	    [&](Extra& extra) {
		    return State {extra.Copy(buffers, count), extra.Copy(offsets, count), extra.Copy(sizes, count),
		                  extra.Copy(strides, count)};
	    },
	    [&](const State& s) {
		    return [buffer, first, count, s] {
			    g_real.vkCmdBindVertexBuffers2(buffer, first, count, s.buffers, s.offsets, s.sizes, s.strides);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBuffer(VkCommandBuffer buffer, VkBuffer src, VkBuffer dst, uint32_t count,
                                         const VkBufferCopy* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdCopyBuffer(buffer, src, dst, count, regions);
		return;
	}
	Push(
	    Room<VkBufferCopy>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	    [&](const VkBufferCopy* copy) {
		    return [buffer, src, dst, count, copy] { g_real.vkCmdCopyBuffer(buffer, src, dst, count, copy); };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyBufferToImage(VkCommandBuffer buffer, VkBuffer src, VkImage dst, VkImageLayout layout,
                                                uint32_t count, const VkBufferImageCopy* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdCopyBufferToImage(buffer, src, dst, layout, count, regions);
		return;
	}
	Push(
	    Room<VkBufferImageCopy>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	    [&](const VkBufferImageCopy* copy) {
		    return [buffer, src, dst, layout, count, copy] {
			    g_real.vkCmdCopyBufferToImage(buffer, src, dst, layout, count, copy);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImageToBuffer(VkCommandBuffer buffer, VkImage src, VkImageLayout layout, VkBuffer dst,
                                                uint32_t count, const VkBufferImageCopy* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdCopyImageToBuffer(buffer, src, layout, dst, count, regions);
		return;
	}
	Push(
	    Room<VkBufferImageCopy>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	    [&](const VkBufferImageCopy* copy) {
		    return [buffer, src, layout, dst, count, copy] {
			    g_real.vkCmdCopyImageToBuffer(buffer, src, layout, dst, count, copy);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdCopyImage(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout, VkImage dst,
                                        VkImageLayout dst_layout, uint32_t count, const VkImageCopy* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdCopyImage(buffer, src, src_layout, dst, dst_layout, count, regions);
		return;
	}
	Push(
	    Room<VkImageCopy>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	    [&](const VkImageCopy* copy) {
		    return [buffer, src, src_layout, dst, dst_layout, count, copy] {
			    g_real.vkCmdCopyImage(buffer, src, src_layout, dst, dst_layout, count, copy);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdBlitImage(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout, VkImage dst,
                                        VkImageLayout dst_layout, uint32_t count, const VkImageBlit* regions,
                                        VkFilter filter) {
	if (!Queue(buffer)) {
		g_real.vkCmdBlitImage(buffer, src, src_layout, dst, dst_layout, count, regions, filter);
		return;
	}
	Push(
	    Room<VkImageBlit>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	    [&](const VkImageBlit* copy) {
		    return [buffer, src, src_layout, dst, dst_layout, count, copy, filter] {
			    g_real.vkCmdBlitImage(buffer, src, src_layout, dst, dst_layout, count, copy, filter);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdResolveImage(VkCommandBuffer buffer, VkImage src, VkImageLayout src_layout, VkImage dst,
                                           VkImageLayout dst_layout, uint32_t count, const VkImageResolve* regions) {
	if (!Queue(buffer)) {
		g_real.vkCmdResolveImage(buffer, src, src_layout, dst, dst_layout, count, regions);
		return;
	}
	Push(
	    Room<VkImageResolve>(count), [&](Extra& extra) { return extra.Copy(regions, count); },
	    [&](const VkImageResolve* copy) {
		    return [buffer, src, src_layout, dst, dst_layout, count, copy] {
			    g_real.vkCmdResolveImage(buffer, src, src_layout, dst, dst_layout, count, copy);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdClearColorImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
                                              const VkClearColorValue* color, uint32_t count,
                                              const VkImageSubresourceRange* ranges) {
	if (!Queue(buffer)) {
		g_real.vkCmdClearColorImage(buffer, image, layout, color, count, ranges);
		return;
	}
	const VkClearColorValue value = *color;
	Push(
	    Room<VkImageSubresourceRange>(count), [&](Extra& extra) { return extra.Copy(ranges, count); },
	    [&](const VkImageSubresourceRange* copy) {
		    return [buffer, image, layout, value, count, copy] {
			    g_real.vkCmdClearColorImage(buffer, image, layout, &value, count, copy);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdClearDepthStencilImage(VkCommandBuffer buffer, VkImage image, VkImageLayout layout,
                                                     const VkClearDepthStencilValue* value, uint32_t count,
                                                     const VkImageSubresourceRange* ranges) {
	if (!Queue(buffer)) {
		g_real.vkCmdClearDepthStencilImage(buffer, image, layout, value, count, ranges);
		return;
	}
	const VkClearDepthStencilValue clear = *value;
	Push(
	    Room<VkImageSubresourceRange>(count), [&](Extra& extra) { return extra.Copy(ranges, count); },
	    [&](const VkImageSubresourceRange* copy) {
		    return [buffer, image, layout, clear, count, copy] {
			    g_real.vkCmdClearDepthStencilImage(buffer, image, layout, &clear, count, copy);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdClearAttachments(VkCommandBuffer buffer, uint32_t count,
                                               const VkClearAttachment* attachments, uint32_t rect_count,
                                               const VkClearRect* rects) {
	if (!Queue(buffer)) {
		g_real.vkCmdClearAttachments(buffer, count, attachments, rect_count, rects);
		return;
	}
	struct State {
		const VkClearAttachment* attachments;
		const VkClearRect*       rects;
	};
	Push(
	    Room<VkClearAttachment>(count) + Room<VkClearRect>(rect_count),
	    [&](Extra& extra) { return State {extra.Copy(attachments, count), extra.Copy(rects, rect_count)}; },
	    [&](const State& s) {
		    return [buffer, count, rect_count, s] {
			    g_real.vkCmdClearAttachments(buffer, count, s.attachments, rect_count, s.rects);
		    };
	    });
}

// ------------------------------------------------------------------- commands with structures

VKAPI_ATTR void VKAPI_CALL CmdBeginConditionalRenderingEXT(VkCommandBuffer buffer,
                                                           const VkConditionalRenderingBeginInfoEXT* info) {
	if (info->pNext != nullptr || !Queue(buffer)) {
		DrainIfOpen(buffer);
		g_real.vkCmdBeginConditionalRenderingEXT(buffer, info);
		return;
	}
	const auto copy = *info;
	PushSimple([buffer, copy] { g_real.vkCmdBeginConditionalRenderingEXT(buffer, &copy); });
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier(VkCommandBuffer buffer, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                                              VkDependencyFlags flags, uint32_t memory_count,
                                              const VkMemoryBarrier* memory, uint32_t buffer_count,
                                              const VkBufferMemoryBarrier* buffers, uint32_t image_count,
                                              const VkImageMemoryBarrier* images) {
	bool chained = false;
	for (uint32_t i = 0; i < memory_count; ++i) {
		chained |= memory[i].pNext != nullptr;
	}
	for (uint32_t i = 0; i < buffer_count; ++i) {
		chained |= buffers[i].pNext != nullptr;
	}
	for (uint32_t i = 0; i < image_count; ++i) {
		chained |= images[i].pNext != nullptr;
	}
	if (chained || !Queue(buffer)) {
		DrainIfOpen(buffer);
		g_real.vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count,
		                            images);
		return;
	}
	struct State {
		const VkMemoryBarrier*       memory;
		const VkBufferMemoryBarrier* buffers;
		const VkImageMemoryBarrier*  images;
	};
	Push(
	    Room<VkMemoryBarrier>(memory_count) + Room<VkBufferMemoryBarrier>(buffer_count) +
	        Room<VkImageMemoryBarrier>(image_count),
	    [&](Extra& extra) {
		    return State {extra.Copy(memory, memory_count), extra.Copy(buffers, buffer_count),
		                  extra.Copy(images, image_count)};
	    },
	    [&](const State& s) {
		    return [buffer, src, dst, flags, memory_count, buffer_count, image_count, s] {
			    g_real.vkCmdPipelineBarrier(buffer, src, dst, flags, memory_count, s.memory, buffer_count, s.buffers,
			                                image_count, s.images);
		    };
	    });
}

VKAPI_ATTR void VKAPI_CALL CmdPipelineBarrier2(VkCommandBuffer buffer, const VkDependencyInfo* info) {
	bool chained = info->pNext != nullptr;
	for (uint32_t i = 0; i < info->memoryBarrierCount; ++i) {
		chained |= info->pMemoryBarriers[i].pNext != nullptr;
	}
	for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; ++i) {
		chained |= info->pBufferMemoryBarriers[i].pNext != nullptr;
	}
	for (uint32_t i = 0; i < info->imageMemoryBarrierCount; ++i) {
		chained |= info->pImageMemoryBarriers[i].pNext != nullptr;
	}
	if (chained || !Queue(buffer)) {
		DrainIfOpen(buffer);
		g_real.vkCmdPipelineBarrier2(buffer, info);
		return;
	}
	Push(
	    Room<VkDependencyInfo>(1) + Room<VkMemoryBarrier2>(info->memoryBarrierCount) +
	        Room<VkBufferMemoryBarrier2>(info->bufferMemoryBarrierCount) +
	        Room<VkImageMemoryBarrier2>(info->imageMemoryBarrierCount),
	    [&](Extra& extra) {
		    auto* copy                  = extra.Copy(info, 1);
		    copy->pMemoryBarriers       = extra.Copy(info->pMemoryBarriers, info->memoryBarrierCount);
		    copy->pBufferMemoryBarriers = extra.Copy(info->pBufferMemoryBarriers, info->bufferMemoryBarrierCount);
		    copy->pImageMemoryBarriers  = extra.Copy(info->pImageMemoryBarriers, info->imageMemoryBarrierCount);
		    return static_cast<const VkDependencyInfo*>(copy);
	    },
	    [&](const VkDependencyInfo* copy) { return [buffer, copy] { g_real.vkCmdPipelineBarrier2(buffer, copy); }; });
}

VKAPI_ATTR void VKAPI_CALL CmdBeginRendering(VkCommandBuffer buffer, const VkRenderingInfo* info) {
	bool chained = info->pNext != nullptr;
	for (uint32_t i = 0; i < info->colorAttachmentCount; ++i) {
		chained |= info->pColorAttachments[i].pNext != nullptr;
	}
	chained |= info->pDepthAttachment != nullptr && info->pDepthAttachment->pNext != nullptr;
	chained |= info->pStencilAttachment != nullptr && info->pStencilAttachment->pNext != nullptr;
	if (chained || !Queue(buffer)) {
		DrainIfOpen(buffer);
		g_real.vkCmdBeginRendering(buffer, info);
		return;
	}
	Push(
	    Room<VkRenderingInfo>(1) + Room<VkRenderingAttachmentInfo>(info->colorAttachmentCount) +
	        2 * Room<VkRenderingAttachmentInfo>(1),
	    [&](Extra& extra) {
		    auto* copy               = extra.Copy(info, 1);
		    copy->pColorAttachments  = extra.Copy(info->pColorAttachments, info->colorAttachmentCount);
		    copy->pDepthAttachment   = extra.Copy(info->pDepthAttachment, info->pDepthAttachment != nullptr ? 1 : 0);
		    copy->pStencilAttachment = extra.Copy(info->pStencilAttachment, info->pStencilAttachment != nullptr ? 1 : 0);
		    return static_cast<const VkRenderingInfo*>(copy);
	    },
	    [&](const VkRenderingInfo* copy) { return [buffer, copy] { g_real.vkCmdBeginRendering(buffer, copy); }; });
}

// Which array of a descriptor write the driver reads. Other types (inline uniform blocks,
// acceleration structures: both need pNext; mutable and the rest) are not copied.
enum class WriteArray { Image, Buffer, TexelBuffer, Unsupported };

WriteArray ArrayOf(VkDescriptorType type) {
	switch (type) {
		case VK_DESCRIPTOR_TYPE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
		case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return WriteArray::Image;
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return WriteArray::Buffer;
		case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return WriteArray::TexelBuffer;
		default: return WriteArray::Unsupported;
	}
}

VKAPI_ATTR void VKAPI_CALL CmdPushDescriptorSetKHR(VkCommandBuffer buffer, VkPipelineBindPoint point,
                                                   VkPipelineLayout layout, uint32_t set, uint32_t count,
                                                   const VkWriteDescriptorSet* writes) {
	bool   copyable = true;
	size_t extra    = Room<VkWriteDescriptorSet>(count);
	for (uint32_t i = 0; i < count && copyable; ++i) {
		const auto n = writes[i].descriptorCount;
		copyable &= writes[i].pNext == nullptr;
		switch (ArrayOf(writes[i].descriptorType)) {
			case WriteArray::Image:
				copyable &= writes[i].pImageInfo != nullptr || n == 0;
				extra += Room<VkDescriptorImageInfo>(n);
				break;
			case WriteArray::Buffer:
				copyable &= writes[i].pBufferInfo != nullptr || n == 0;
				extra += Room<VkDescriptorBufferInfo>(n);
				break;
			case WriteArray::TexelBuffer:
				copyable &= writes[i].pTexelBufferView != nullptr || n == 0;
				extra += Room<VkBufferView>(n);
				break;
			case WriteArray::Unsupported: copyable = false; break;
		}
	}
	if (!copyable || !Queue(buffer)) {
		DrainIfOpen(buffer);
		g_real.vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, writes);
		return;
	}
	Push(
	    extra,
	    [&](Extra& copies) {
		    auto* copy = copies.Copy(writes, count);
		    for (uint32_t i = 0; i < count; ++i) {
			    const auto n             = writes[i].descriptorCount;
			    const auto array         = ArrayOf(writes[i].descriptorType);
			    copy[i].pImageInfo       = array == WriteArray::Image ? copies.Copy(writes[i].pImageInfo, n) : nullptr;
			    copy[i].pBufferInfo      = array == WriteArray::Buffer ? copies.Copy(writes[i].pBufferInfo, n) : nullptr;
			    copy[i].pTexelBufferView =
			        array == WriteArray::TexelBuffer ? copies.Copy(writes[i].pTexelBufferView, n) : nullptr;
		    }
		    return static_cast<const VkWriteDescriptorSet*>(copy);
	    },
	    [&](const VkWriteDescriptorSet* copy) {
		    return [buffer, point, layout, set, count, copy] {
			    g_real.vkCmdPushDescriptorSetKHR(buffer, point, layout, set, count, copy);
		    };
	    });
}

// Geometry arrays, pointer-to-pointer ranges and device addresses: never copied.
VKAPI_ATTR void VKAPI_CALL CmdBuildAccelerationStructuresKHR(VkCommandBuffer buffer, uint32_t count,
                                                             const VkAccelerationStructureBuildGeometryInfoKHR* infos,
                                                             const VkAccelerationStructureBuildRangeInfoKHR* const* ranges) {
	DrainIfOpen(buffer);
	g_real.vkCmdBuildAccelerationStructuresKHR(buffer, count, infos, ranges);
}

// ----------------------------------------------------------------------------------- install

struct Entry {
	const char*        name;
	PFN_vkVoidFunction stub;
};

// Handed out by vkGetDeviceProcAddr (the memory allocator fetches its functions through it): only
// functions that wait. vkBeginCommandBuffer is not among them, so code with its own function
// pointers never begins a deferred command buffer.
#define KYTY_RECORDER_ENTRY(name) {"vk" #name, reinterpret_cast<PFN_vkVoidFunction>(&(name))}
const Entry g_entries[] = {
    KYTY_RECORDER_ENTRY(EndCommandBuffer),
    KYTY_RECORDER_ENTRY(ResetCommandBuffer),
    KYTY_RECORDER_ENTRY(AllocateCommandBuffers),
    KYTY_RECORDER_ENTRY(FreeCommandBuffers),
    KYTY_RECORDER_ENTRY(ResetCommandPool),
    KYTY_RECORDER_ENTRY(DestroyCommandPool),
    KYTY_RECORDER_ENTRY(DestroyBuffer),
    KYTY_RECORDER_ENTRY(DestroyImage),
    KYTY_RECORDER_ENTRY(DestroyImageView),
    KYTY_RECORDER_ENTRY(DestroyBufferView),
    KYTY_RECORDER_ENTRY(DestroySampler),
    KYTY_RECORDER_ENTRY(DestroyPipeline),
    KYTY_RECORDER_ENTRY(DestroyPipelineLayout),
    KYTY_RECORDER_ENTRY(DestroyDescriptorSetLayout),
    KYTY_RECORDER_ENTRY(DestroyDescriptorPool),
    KYTY_RECORDER_ENTRY(ResetDescriptorPool),
    KYTY_RECORDER_ENTRY(DestroyQueryPool),
    KYTY_RECORDER_ENTRY(FreeMemory),
};
#undef KYTY_RECORDER_ENTRY

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name) {
	for (const auto& entry: g_entries) {
		if (std::strcmp(entry.name, name) == 0) {
			return entry.stub;
		}
	}
	return g_real.vkGetDeviceProcAddr(device, name);
}

} // namespace

bool Enabled() noexcept {
	return g_ring != nullptr;
}

void Sync(VkCommandBuffer buffer) {
	DrainIfOpen(buffer);
}

void SyncAll() {
	DrainAll();
}

uint64_t OversizedCommands() noexcept {
	return g_oversized.load(std::memory_order_relaxed);
}

bool IsDeferred(VkCommandBuffer buffer) noexcept {
	return g_ring != nullptr && IsOpen(buffer) && IsProducer() && g_direct.empty();
}

void EndAndRun(const VkCommandBuffer* buffers, uint32_t count, DeferredRun run, const void* data, size_t size) {
	constexpr size_t Align = 16;
	EXIT_IF(g_ring == nullptr || count == 0 || run == nullptr);
	for (uint32_t i = 0; i < count; ++i) {
		EXIT_IF(!IsDeferred(buffers[i]));
		Close(buffers[i]);
	}
	struct State {
		const VkCommandBuffer* buffers;
		const void*            data;
	};
	Push(
	    Room<VkCommandBuffer>(count) + size + Align,
	    [&](Extra& extra) { return State {extra.Copy(buffers, count), extra.CopyBytes(data, size, Align)}; },
	    [&](const State& s) {
		    return [count, run, s] {
			    for (uint32_t i = 0; i < count; ++i) {
				    const auto result = g_real.vkEndCommandBuffer(s.buffers[i]);
				    EXIT_IF(result != VK_SUCCESS);
			    }
			    run(s.data, s.buffers, count);
		    };
	    });
}

void Install() {
	if (g_ring != nullptr) {
		// The dispatcher was loaded again: its entries are the real ones now.
		Log::WriteToConsoleAndLog("[recorder] device functions loaded again: recording stays on the calling thread\n");
		return;
	}
	// On by default (GTA VI world A/B rt_ab1: -26 % GPU-thread instructions per operation);
	// KYTY_RECORDER=0 turns it off.
	const char* setting = std::getenv("KYTY_RECORDER");
	if (setting != nullptr && std::strcmp(setting, "0") == 0) {
		return;
	}
	auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;

	bool complete = true;
#define KYTY_RECORDER_LOAD_REQUIRED(name)                                                          \
	g_real.vk##name = d.vk##name;                                                                  \
	if (g_real.vk##name == nullptr) {                                                              \
		Log::WriteToConsoleAndLog("[recorder] vk" #name " missing: off\n");                        \
		complete = false;                                                                          \
	}
#define KYTY_RECORDER_LOAD_OPTIONAL(name) g_real.vk##name = d.vk##name;
	KYTY_RECORDER_LIFETIME(KYTY_RECORDER_LOAD_REQUIRED)
	KYTY_RECORDER_COMMANDS(KYTY_RECORDER_LOAD_OPTIONAL)
	KYTY_RECORDER_LOAD_OPTIONAL(DestroyAccelerationStructureKHR)
#undef KYTY_RECORDER_LOAD_REQUIRED
#undef KYTY_RECORDER_LOAD_OPTIONAL
	if (!complete) {
		return;
	}

	g_ring = new Ring();

	// Fallbacks first: an alias below may replace one of them with a copying hook.
#define KYTY_RECORDER_SET_FALLBACK(name)                                                           \
	g_fallback_##name = d.vk##name;                                                                \
	if (g_fallback_##name != nullptr) {                                                            \
		d.vk##name = &Fallback<PFN_vk##name>::Call<&g_fallback_##name>;                            \
	}
	KYTY_RECORDER_FALLBACKS(KYTY_RECORDER_SET_FALLBACK)
#undef KYTY_RECORDER_SET_FALLBACK

#define KYTY_RECORDER_SET_REQUIRED(name) d.vk##name = &(name);
#define KYTY_RECORDER_SET_OPTIONAL(name)                                                           \
	if (g_real.vk##name != nullptr) {                                                              \
		d.vk##name = &(name);                                                                      \
	}
	KYTY_RECORDER_LIFETIME(KYTY_RECORDER_SET_REQUIRED)
	KYTY_RECORDER_COMMANDS(KYTY_RECORDER_SET_OPTIONAL)
	KYTY_RECORDER_SET_OPTIONAL(DestroyAccelerationStructureKHR)
#undef KYTY_RECORDER_SET_REQUIRED
#undef KYTY_RECORDER_SET_OPTIONAL

#define KYTY_RECORDER_SET_ALIAS(alias, name)                                                       \
	if (g_real.vk##name != nullptr && d.vk##alias != nullptr) {                                    \
		d.vk##alias = &(name);                                                                     \
	}
	KYTY_RECORDER_ALIASES(KYTY_RECORDER_SET_ALIAS)
#undef KYTY_RECORDER_SET_ALIAS

	Log::WriteToConsoleAndLog("[recorder] Vulkan commands of the GPU thread are recorded and submitted on their own "
	                          "thread (KYTY_RECORDER=1)\n");
}

// NOLINTEND(cppcoreguidelines-macro-usage)

} // namespace Libs::Graphics::Recorder
