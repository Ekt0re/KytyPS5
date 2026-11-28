#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/faultStats.h"
#include "common/logging/log.h"
#include "common/perfSummary.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace Libs::Graphics {

namespace {

// After a guest write fault the clean buffer pages of the aligned window around it lose their
// write watch too, so a run of guest writes faults once per window instead of once per page
// (after the Bloodborne port's write-fault window, BB_FAULT_WINDOW). KYTY_FAULT_WINDOW=n sets the
// window in KiB, rounded down to a power of two in [4, 1024]; 4 turns it off.
// KYTY_FAULT_WINDOW_TOGGLE=path rereads the size from that file twice a second, for A/B runs.
std::atomic<uint64_t> g_fault_window {0};

uint64_t FaultWindowBytes(uint64_t kib) {
	return std::bit_floor(std::clamp<uint64_t>(kib, 4, 1024)) * 1024ull;
}

void InitFaultWindow() {
	static std::once_flag once;
	std::call_once(once, [] {
		const char* value = std::getenv("KYTY_FAULT_WINDOW");
		g_fault_window.store(FaultWindowBytes(value != nullptr ? std::strtoull(value, nullptr, 10) : 64),
		                     std::memory_order_relaxed);
		const char* toggle = std::getenv("KYTY_FAULT_WINDOW_TOGGLE");
		if (toggle == nullptr) {
			return;
		}
		std::thread([path = std::string(toggle)] {
			uint64_t last = 0;
			for (;;) {
				std::this_thread::sleep_for(std::chrono::milliseconds(500));
				uint64_t kib = 0;
				if (auto* file = std::fopen(path.c_str(), "r"); file != nullptr) {
					if (std::fscanf(file, "%" SCNu64, &kib) != 1) {
						kib = 0;
					}
					std::fclose(file);
				}
				if (kib == 0 || kib == last) {
					continue;
				}
				last = kib;
				g_fault_window.store(FaultWindowBytes(kib), std::memory_order_relaxed);
				std::printf("FaultWindow %" PRIu64 " KiB\n", FaultWindowBytes(kib) / 1024);
				std::fflush(stdout);
			}
		}).detach();
	});
}

} // namespace

RenderContext::RenderContext(GraphicContext& graphics, const std::filesystem::path& shader_store)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_resolve_cache(*this), m_texture_cache(graphics, m_command_scheduler, m_page_manager,
	                                          m_buffer_cache, m_resolve_cache),
      m_image_heap(*this, m_page_manager), m_pipeline_cache(graphics, &m_image_heap, shader_store),
      m_occlusion_queries(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitFaultWindow();
	Common::FaultStats::ConfigureStorm();
	m_pipeline_cache.SetHeapRows(&m_buffer_cache.GetHeapRows());
	if (graphics.ray_query_enabled) {
		m_ray_tracing = std::make_unique<RayTracing>(*this);
	}
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// Fault count is taken at the existing PerfSummary site below; this timer and the hot-block
	// table only add the cost and the address distribution, never a second count.
	Common::PerfSummary::ScopedMicros fault_timer(Common::PerfSummary::Counter::FaultHandlerUs);
	Common::FaultStats::SetFrame(static_cast<uint32_t>(GetGpu().GetFrameNum()));
	Common::FaultStats::g_hot_blocks.Note(fault_vaddr);
	// The access retries once the image write-backs over its page land.
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		return true;
	}
	using Common::DebugCounters::Counter;
	Common::DebugCounters::ScopedTimer timer(
	    access == PageFaultAccess::Write ? Counter::SyncWriteFaults : Counter::SyncReadFaults,
	    Counter::SyncFaultNs);
	Common::PerfSummary::Add(access == PageFaultAccess::Write
	                             ? Common::PerfSummary::Counter::WriteFaults
	                             : Common::PerfSummary::Counter::ReadFaults);
	if (access == PageFaultAccess::Write) {
		{
			Common::DebugCounters::ScopedTimer buffer_timer(Counter::SyncFaultBufferNs);
			m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
			// Images and image heaps keep their own watches, so their pages still fault.
			const auto window = g_fault_window.load(std::memory_order_relaxed);
			if (window > TRACKER_PAGE_SIZE) {
				const auto begin = Common::AlignDown(fault_vaddr, window);
				if (IsMapped(begin, window)) {
					m_buffer_cache.InvalidateWriteWindow(begin, window);
				}
			}
		}
		{
			Common::DebugCounters::ScopedTimer texture_timer(Counter::SyncFaultTextureNs);
			m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		}
		Common::DebugCounters::ScopedTimer heap_timer(Counter::SyncFaultHeapNs);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

bool RenderContext::IsAnyMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Intersects(vaddr, size);
}

void RenderContext::NoteMappingChange(uint64_t vaddr, uint64_t size) noexcept {
	const auto next = m_mapping_generation.load(std::memory_order_relaxed) + 1u;
	m_mapping_changes[next % m_mapping_changes.size()] = {vaddr, size};
	m_mapping_generation.store(next, std::memory_order_release);
}

bool RenderContext::MappingChangedSince(uint64_t generation, uint64_t vaddr,
                                        uint64_t size) const noexcept {
	std::shared_lock lock(m_mapped_ranges_mutex);
	const auto       current = m_mapping_generation.load(std::memory_order_relaxed);
	if (generation > current || current - generation >= m_mapping_changes.size()) {
		return true;
	}
	for (auto change = generation + 1u; change <= current; ++change) {
		const auto& [address, bytes] = m_mapping_changes[change % m_mapping_changes.size()];
		if (vaddr >= address ? vaddr - address < bytes : address - vaddr < size) {
			return true;
		}
	}
	return false;
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	NoteMappingChange(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::DropCachedMemory(uint64_t vaddr, uint64_t size) {
	// Check cache ownership on the GPU thread. Guest-memory callbacks can still
	// access a range with no cached data, so they must finish before it is unmapped.
	if (m_command_scheduler.Active() &&
	    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
	     m_texture_cache.IsRegionRegistered(vaddr, size) ||
	     m_command_scheduler.HasPendingPriorityOperations())) {
		const auto tick = m_command_scheduler.CurrentTick();
		m_command_scheduler.Finish();
		m_command_scheduler.WaitPriorityOperations(tick);
	}
	m_buffer_cache.UnmapZeroCopy(vaddr, size);
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.UnmapMemory(vaddr, size);
	m_image_heap.UnmapMemory(vaddr, size);
}

// KYTY_SYNC_FRESH_MAPS=1: a guest thread that maps memory over a free or reserved range waits
// for the GPU thread again, as before.
static bool DeferFreshMaps() {
	static const bool defer = [] {
		const char* env = std::getenv("KYTY_SYNC_FRESH_MAPS");
		return env == nullptr || env[0] != '1';
	}();
	return defer;
}

void RenderContext::PostDropCachedMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported cache drop from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	Common::PerfSummary::Add(Common::PerfSummary::Counter::GpuUnmaps);
	if (m_gpu == nullptr || m_gpu->IsStopping() || GuestGpu::IsGpuThread()) {
		DropCachedMemory(vaddr, size);
		return;
	}
	// A sender faster than the GPU thread waits for it now and then; the commands before this
	// one run first.
	constexpr uint32_t MaxPendingCommands = 1024;
	if (m_gpu->PendingCommands() >= MaxPendingCommands) {
		m_gpu->SendCommandSync([this, vaddr, size] { DropCachedMemory(vaddr, size); });
		return;
	}
	m_gpu->SendCommand([this, vaddr, size] { DropCachedMemory(vaddr, size); });
}

void RenderContext::MapFreshMemory(uint64_t vaddr, uint64_t size) {
	// The unmap that made the range free or reserved wrote its GPU-written buffer bytes back
	// (BufferCache::InvalidateMemory downloads GPU-dirty pages and clears their marks), so the
	// buffer cache holds none for it. If it still does (GPU work that wrote the range after its
	// unmap, or a write-back that did not clear the marks), a deferred drop would write those
	// bytes into the new mapping: unmap first and wait, as before. The query takes no lock.
	if (!DeferFreshMaps() || m_buffer_cache.IsRegionGpuModified(vaddr, size)) {
		UnmapMemory(vaddr, size);
		MapMemory(vaddr, size);
		return;
	}
	// The range was free or reserved until now: no guest thread and no GPU work could reach its
	// memory, so nothing has to finish before the memory changes, unlike an unmap. The range is
	// mapped for the GPU at once (a guest write fault in it is handled from now on); what the
	// caches still hold for these addresses from an earlier mapping is dropped by an ordered
	// command. GTA VI commits a 3.3 GiB heap 64 KiB at a time while it loads (54,000 maps), and
	// waiting for the GPU thread cost each of them two thread wake-ups: 0.4 ms on an idle host,
	// 1.5 ms and more on a loaded one, which is longer than the game's 30 s hang check allows.
	MapMemory(vaddr, size);
	PostDropCachedMemory(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	Common::PerfSummary::Add(Common::PerfSummary::Counter::GpuUnmaps);
	const auto unmap = [this, vaddr, size] {
		DropCachedMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
		NoteMappingChange(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	[[maybe_unused]] static const bool logged = [] {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		return true;
	}();
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	if (m_ray_tracing) {
		m_ray_tracing->RunGarbageCollector();
	}
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	const auto ident = static_cast<uintptr_t>(event_id);
	auto*      data  = reinterpret_cast<void*>((static_cast<uintptr_t>(context_id) << 16u) | ident);
	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, ident, LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS, data);
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
