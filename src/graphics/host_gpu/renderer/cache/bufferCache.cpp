#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "graphics/host_gpu/renderer/bdaGuard.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/perfSummary.h"
#include "common/profiler.h"
#include "common/runtimeToggles.h"
#include "graphics/guest_gpu/command_processor/waitDiagnostics.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <numeric>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// GPU-modified ranges come as many small scattered islands, so a readback widens to a window.
constexpr uint64_t DownloadWindowSize = 512 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferRelease([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

// KYTY_SYNC_STATS also counts re-uploads of downloaded bytes and of unchanged pages, which hashes
// every uploaded page.
bool DetailedSyncStats() noexcept {
	static const bool enabled = std::getenv("KYTY_SYNC_STATS") != nullptr;
	return enabled && Common::DebugCounters::Enabled();
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
		// Work recorded after the fill above finds no entry for the buffer.
		BdaGuard::Retire(buffer.BufferDeviceAddress());
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferRelease([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	if (buffer.IsZeroCopy()) {
		// The bytes are in guest memory already; only the write has to finish.
		return ResolveZeroCopy(buffer, !async);
	}
	std::vector<DownloadPart> parts;
	uint64_t                  total_size = 0;
	if (!CollectDownload(buffer, vaddr, size, parts, total_size)) {
		return false;
	}
	DownloadParts<async>(parts, total_size, false);
	return true;
}

bool BufferCache::CollectDownload(Buffer& buffer, uint64_t vaddr, uint64_t size,
                                  std::vector<DownloadPart>& parts, uint64_t& total_size) {
	const auto   buffer_address = buffer.CpuAddress();
	DownloadPart part {&buffer, buffer_address, {}};
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    part.copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Start each packed range on its own 64-byte cache line.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	if (part.copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes,
	                           std::accumulate(part.copies.begin(), part.copies.end(), uint64_t {0},
	                                           [](uint64_t sum, const vk::BufferCopy& copy) {
		                                           return sum + Common::AlignUp(copy.size, 64);
	                                           }));
	Common::PerfSummary::Add(Common::PerfSummary::Counter::BufferReadbacks);
	if (DetailedSyncStats()) {
		for (const auto& copy: part.copies) {
			m_stats_downloaded.Add(buffer_address + copy.srcOffset, copy.size);
		}
	}
	parts.push_back(std::move(part));
	return true;
}

template <bool async>
void BufferCache::DownloadParts(std::vector<DownloadPart>& parts, uint64_t total_size,
                                bool fuse) {
	EXIT_IF(parts.empty() || total_size == 0);
	auto download = m_download_buffer.MapDownload(total_size, 64);
	for (auto& part: parts) {
		for (auto& copy: part.copies) {
			copy.dstOffset += download.offset;
		}
	}

	// The publication below takes `download` over (and with it a temporary buffer's owner) before
	// `record` runs: the copy targets are taken here.
	const auto download_handle = download.buffer->Handle();
	const auto download_offset = download.offset;
	const auto record = [&parts, download_handle, download_offset, total_size](vk::CommandBuffer native) {
		// One barrier for all source buffers, then their copies, then one barrier for the bytes
		// the host reads.
		std::vector<vk::BufferMemoryBarrier> before(parts.size());
		for (size_t index = 0; index < parts.size(); ++index) {
			auto& barrier               = before[index];
			barrier.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
			barrier.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
			barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			barrier.buffer              = parts[index].buffer->Handle();
			barrier.offset              = 0;
			barrier.size                = parts[index].buffer->Size();
		}
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr,
		                       static_cast<uint32_t>(before.size()), before.data(), 0, nullptr);
		for (const auto& part: parts) {
			native.copyBuffer(part.buffer->Handle(), download_handle,
			                  static_cast<uint32_t>(part.copies.size()), part.copies.data());
		}

		auto after          = before.front();
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eHostRead;
		after.buffer        = download_handle;
		after.offset        = download_offset;
		after.size          = total_size;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands |
		                           vk::PipelineStageFlagBits::eHost,
		                       {}, 0, nullptr, 1, &after, 0, nullptr);
	};
	// Decided after MapDownload, which may submit the current command buffer while it waits.
	const bool detached =
	    !async && std::all_of(parts.begin(), parts.end(), [&](const DownloadPart& part) {
		    return CanDownloadDetached(part.buffer->CpuAddress(), part.copies);
	    });
	// `record` reads `parts` by reference and may run after this point: the publication keeps
	// its own copy of the addresses and offsets.
	auto publish = [total_size, parts = parts, download = std::move(download)] {
		download.buffer->Invalidate(download.offset, total_size);
		for (const auto& part: parts) {
			const auto buffer_address = part.buffer_address;
			for (const auto& copy: part.copies) {
				Libs::LibKernel::Memory::WriteBacking(
				    buffer_address + copy.srcOffset,
				    download.mapped + (copy.dstOffset - download.offset), copy.size);
			}
		}
	};
	if constexpr (async) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		record(command.Handle());
		m_scheduler.DeferPriorityOperation(std::move(publish));
		return;
	}
	Common::PerfSummary::Add(Common::PerfSummary::Counter::RbBatches);
	if (QueueProbe()) {
		ProbeReadbackQueue(parts);
	}
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::SyncReadbacks,
	                                         Common::DebugCounters::Counter::SyncReadbackNs);
	using Common::PerfSummary::Counter;
	using Common::PerfSummary::ScopedMicros;
	const bool copy_queue = detached && CopyQueue();
	const auto path       = ReadbackBatch::ChooseReadbackPath(
        {.detached   = detached,
	           .copy_queue = copy_queue,
	           .fuse       = fuse && m_detached_readback == DetachedReadback::Submit &&
	                   m_scheduler.CanSubmitDetachedWithCurrent()});
	struct PathCounters {
		Counter count, wait, total;
	};
	const auto counters = [path]() -> PathCounters {
		switch (path) {
			case ReadbackBatch::ReadbackPath::CopyQueue:
				return {Counter::RbPathCq, Counter::RbPathCqWaitUs, Counter::RbPathCqTotalUs};
			case ReadbackBatch::ReadbackPath::Fused:
				return {Counter::RbPathFused, Counter::RbPathFusedWaitUs, Counter::RbPathFusedTotalUs};
			case ReadbackBatch::ReadbackPath::Detached:
				return {Counter::RbPathDetached, Counter::RbPathDetachedWaitUs,
				        Counter::RbPathDetachedTotalUs};
			case ReadbackBatch::ReadbackPath::Flush: break;
		}
		return {Counter::RbPathFlush, Counter::RbPathFlushWaitUs, Counter::RbPathFlushTotalUs};
	}();
	Common::PerfSummary::Add(counters.count);
	ScopedMicros path_total_timer(counters.total);
	if (path == ReadbackBatch::ReadbackPath::CopyQueue) {
		// As a detached copy, but on the copy queue: it waits on the GPU for the submitted ticks
		// only (on the universal queue it also queued behind the work submitted ahead of it),
		// and the open command buffer is not submitted.
		Common::DebugCounters::Add(Common::DebugCounters::Counter::SyncDetachedReadbacks);
		const auto tick = m_scheduler.CurrentTick();
		if (Common::PerfSummary::Enabled()) {
			auto& master = m_scheduler.GetMasterSemaphore();
			if (!master.IsFree(tick - 1)) {
				master.Refresh();
				if (!master.IsFree(tick - 1)) {
					Common::PerfSummary::Add(Counter::RbCqGpuBusy);
				}
			}
		}
		{
			ScopedMicros path_wait_timer(counters.wait);
			(void)m_scheduler.SubmitOnCopyQueue(record);
			ScopedMicros wait_timer(Counter::RbWaitUs);
			m_scheduler.WaitPriorityOperations(tick - 1);
		}
		// KYTY_COPY_QUEUE_FLUSH=1: the open command buffer is still submitted (without a wait), as
		// the detached path does.
		if (CopyQueueFlush()) {
			ScopedMicros flush_timer(Counter::RbFlushUs);
			m_scheduler.Flush();
			Common::PerfSummary::Add(Counter::RbSubmits);
		}
	} else if (detached) {
		// The barrier orders the copy after every submitted command, and no unsubmitted command
		// writes these ranges: the copy reads what a flush and wait would have read.
		Common::DebugCounters::Add(Common::DebugCounters::Counter::SyncDetachedReadbacks);
		// Nothing is deferred at the current tick (CanDownloadDetached): older publications land
		// first, as in a flush and wait.
		const auto tick = m_scheduler.CurrentTick();
		// Batched: the copy and the open command buffer go to the GPU in one submission, the copy
		// first, and only the copy is waited for. Unbatched: the copy's own submission and wait,
		// then the open command buffer's submission.
		// The fused path's wait counter includes submitting the open command buffer: its wait
		// happens inside SubmitDetachedWithCurrent.
		std::optional<ScopedMicros> fused_wait_timer;
		if (path == ReadbackBatch::ReadbackPath::Fused) {
			fused_wait_timer.emplace(counters.wait);
		}
		if (path == ReadbackBatch::ReadbackPath::Fused && m_scheduler.SubmitDetachedWithCurrent(record)) {
			Common::PerfSummary::Add(Counter::RbFused);
			ScopedMicros wait_timer(Counter::RbWaitUs);
			m_scheduler.WaitPriorityOperations(tick - 1);
		} else {
			fused_wait_timer.reset();
			{
				ScopedMicros path_wait_timer(counters.wait);
				m_scheduler.SubmitDetached(record);
				ScopedMicros wait_timer(Counter::RbWaitUs);
				m_scheduler.WaitPriorityOperations(tick - 1);
			}
			// The open command buffer is still submitted, without waiting for it, as a flush and
			// wait would have: completions the guest waits for (flips, interrupts, image
			// write-backs) keep the latency they had. Without this, GTA VI fell into a black
			// 30 fps loop in 2 of 3 runs although every detached download read the same bytes
			// as a flushed one (KYTY_DETACHED_READBACK=2 skips the submission, for comparison).
			if (m_detached_readback == DetachedReadback::Submit) {
				ScopedMicros flush_timer(Counter::RbFlushUs);
				m_scheduler.Flush();
				Common::PerfSummary::Add(Counter::RbSubmits);
			}
		}
	} else {
		// A write of these ranges is still in the open command buffer: the copy goes behind it,
		// into the same submission.
		{
			ScopedMicros submit_timer(Counter::RbSubmitUs);
			auto&        command = m_scheduler.Current();
			command.EndRendering();
			record(command.Handle());
		}
		const auto tick = m_scheduler.CurrentTick();
		{
			ScopedMicros flush_timer(Counter::RbFlushUs);
			Common::PerfSummary::Add(Counter::RbSubmits);
			m_scheduler.Flush();
		}
		ScopedMicros wait_timer(Counter::RbWaitUs);
		ScopedMicros path_wait_timer(counters.wait);
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
	}
	ScopedMicros copy_timer(Counter::RbCopyUs);
	publish();
}

bool BufferCache::CanDownloadDetached(uint64_t                        buffer_address,
                                      std::span<const vk::BufferCopy> copies) {
	const auto tick = m_scheduler.CurrentTick();
	// Address writes may reach these ranges; publications deferred at this tick go first.
	if (m_detached_readback == DetachedReadback::Off || m_address_writes_pending || m_address_write_tick == tick ||
	    m_scheduler.HasPriorityOperationsSince(tick)) {
		return false;
	}
	return std::none_of(copies.begin(), copies.end(), [&](const vk::BufferCopy& copy) {
		return HasUnsubmittedWrite(buffer_address + copy.srcOffset, copy.size);
	});
}

void BufferCache::MarkUnsubmittedWrite(uint64_t vaddr, uint64_t size) {
	const auto tick = m_scheduler.CurrentTick();
	if (m_unsubmitted_tick != tick) {
		m_unsubmitted_writes.Clear();
		m_unsubmitted_tick = tick;
	}
	m_unsubmitted_writes.Add(vaddr, size);
}

void BufferCache::RecordPendingWrites() {
	m_pending_writes.ForEach(
	    [this](uint64_t begin, uint64_t end) { MarkUnsubmittedWrite(begin, end - begin); });
	m_pending_writes.Clear();
	if (m_address_writes_pending) {
		m_address_writes_pending = false;
		m_address_write_tick     = m_scheduler.CurrentTick();
	}
}

bool BufferCache::HasUnsubmittedWrite(uint64_t vaddr, uint64_t size) const {
	return m_pending_writes.Intersects(vaddr, size) ||
	       (m_unsubmitted_tick == m_scheduler.CurrentTick() &&
	        m_unsubmitted_writes.Intersects(vaddr, size));
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_guest_imports(graphics),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	// On by default (bb_ab2, world: readback ms/f -14%, gpu_cpu_us/op -19%);
	// KYTY_BATCH_READBACK=0 turns it off.
	if (const char* value = std::getenv("KYTY_BATCH_READBACK"); value != nullptr) {
		m_batch_readback_env = value[0] != '0';
	}
	m_batch_readback = m_batch_readback_env;
	if (const char* value = std::getenv("KYTY_BATCH_READBACK_PREFETCH"); value != nullptr) {
		m_batch_prefetch = std::strtoull(value, nullptr, 10) * 1024;
	}
	// KYTY_DETACHED_READBACK=0 flushes and waits for every synchronous download, as before; 2
	// reads detached without submitting the open command buffer afterwards.
	if (const char* value = std::getenv("KYTY_DETACHED_READBACK"); value != nullptr) {
		m_detached_readback = value[0] == '0'   ? DetachedReadback::Off
		                      : value[0] == '2' ? DetachedReadback::NoSubmit
		                                        : DetachedReadback::Submit;
	}
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	WaitZeroCopyReads(vaddr, size);
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	QueueBdaSync(vaddr, size);
}

void BufferCache::InvalidateWriteWindow(uint64_t vaddr, uint64_t size) {
	WaitZeroCopyReads(vaddr, size);
	m_memory_tracker.MarkCleanAsCpuModified(vaddr, size);
	// Shader address reads see the window's CPU bytes only through these uploads.
	QueueBdaSync(vaddr, size);
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		using Common::PerfSummary::Counter;
		Common::PerfSummary::ScopedMicros handler_timer(Counter::RbHandlerUs);
		// Guest write faults on GPU-written pages come here too (InvalidateMemory).
		Common::PerfSummary::ScopedMicros write_timer(Counter::RbWriteUs, is_write);
		Common::PerfSummary::Add(Counter::RbCalls);
		if (is_write) {
			Common::PerfSummary::Add(Counter::RbWriteCalls);
		}
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		const auto buffer_id = FindBuffer(vaddr, size);
		auto&      buffer    = m_slot_buffers[buffer_id];
		bool       migrate   = false;
		const bool gpu_modified = IsRegionGpuModified(vaddr, size);
		// Diagnostic: a read fault that keeps repeating on one page without a download is a
		// livelock (the page stays read-protected); logged at 256, 4096, 65536, ... repeats.
		{
			static uint64_t last_page = ~0ull;
			static uint64_t repeats   = 0;
			const auto      page      = vaddr / TRACKER_PAGE_SIZE;
			repeats                   = page == last_page ? repeats + 1 : 0;
			last_page                 = page;
			if (repeats == 256 || (repeats >= 4096 && (repeats & (repeats - 1)) == 0)) {
				std::printf("BufferCache: read fault repeats %" PRIu64 "x at 0x%016" PRIx64
				            " (page 0x%016" PRIx64 "): gpu_modified=%d buffer 0x%016" PRIx64
				            "+0x%" PRIx64 " zero_copy=%d readback_faults=%u gpu_ranges=%d\n",
				            repeats, vaddr, page * TRACKER_PAGE_SIZE, gpu_modified ? 1 : 0,
				            buffer.CpuAddress(), buffer.Size(), buffer.IsZeroCopy() ? 1 : 0,
				            buffer.readback_faults,
				            m_gpu_modified_ranges.Intersects(vaddr, size) ? 1 : 0);
			}
		}
		if (gpu_modified) {
			const auto start =
			    std::max(Common::AlignDown(vaddr, DownloadWindowSize), buffer.CpuAddress());
			const auto end = std::min(std::max(start + DownloadWindowSize, vaddr + size),
			                          buffer.CpuAddress() + buffer.Size());
			if (Common::PerfSummary::Enabled() &&
			    m_readback_neighbours.Note(Common::AlignDown(vaddr, DownloadWindowSize),
			                               DownloadWindowSize)) {
				Common::PerfSummary::Add(Common::PerfSummary::Counter::RbNeighbour);
			}
			bool downloaded = false;
			if (!buffer.IsZeroCopy() && BatchReadback()) {
				downloaded = DownloadWindowBatch(buffer_id, vaddr, size, {start, end});
			} else {
				downloaded = DownloadBufferMemory<false>(buffer, start, end - start);
				if (downloaded) {
					m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
				}
			}
			if (downloaded) {
				if (buffer.IsZeroCopy()) {
					// Resolved in place: a VRAM buffer would have copied the window.
					m_zero_copy_fences.CountSaved(buffer.CpuAddress());
					Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcSaved);
				}
				if (WaitDiag::Enabled()) {
					WaitDiag::Global().NoteReadback(start, end - start, WaitDiag::NowMs());
				}
				// Buffers the CPU keeps faulting on may move to guest memory (zero-copy).
				buffer.readback_faults++;
				migrate = ZeroCopyEnabled() && !buffer.IsZeroCopy();
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size);
			if (m_cpu_write_listener) {
				m_cpu_write_listener(vaddr, size);
			}
		}
		if (migrate) {
			MaybeMigrateToZeroCopy(buffer_id);
		}
	});
}

bool BufferCache::QueueProbe() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_RB_QUEUE_PROBE");
		return value != nullptr && value[0] == '1';
	}();
	return enabled && Common::PerfSummary::Enabled();
}

void BufferCache::ProbeReadbackQueue(const std::vector<DownloadPart>& parts) {
	using Common::PerfSummary::Counter;
	auto& master = m_scheduler.GetMasterSemaphore();
	master.Refresh();
	const auto submitted = master.SubmittedTick();
	const auto done      = master.KnownGpuTick();
	Common::PerfSummary::Add(Counter::RbQueueTicks, submitted > done ? submitted - done : 0);
	uint64_t producer = 0;
	for (const auto& part: parts) {
		producer = std::max(producer, part.buffer->probe_write_tick);
	}
	if (producer <= done) {
		Common::PerfSummary::Add(Counter::RbProducerDone);
		return;
	}
	if (producer > submitted) {
		Common::PerfSummary::Add(Counter::RbProducerOpen);
		return;
	}
	Common::PerfSummary::Add(Counter::RbProducerPending);
	Common::PerfSummary::Add(Counter::RbAfterTicks, submitted - producer);
	// The download waits for this anyway: timing it separately splits the wait into the part
	// up to the producing work and the part for work queued behind it.
	Common::PerfSummary::ScopedMicros timer(Counter::RbProducerWaitUs);
	master.Wait(producer);
}

bool BufferCache::CopyQueue() {
	if (!m_scheduler.HasCopyQueue()) {
		return false;
	}
	if (!Common::RuntimeToggles::Enabled()) {
		return true;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now >= m_copy_queue_next) {
		m_copy_queue_next  = now + std::chrono::milliseconds(250);
		const int toggle   = Common::RuntimeToggles::Find("copy_queue");
		m_copy_queue       = toggle != 0;
	}
	return m_copy_queue;
}

bool BufferCache::CopyQueueFlush() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_COPY_QUEUE_FLUSH");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

bool BufferCache::BatchReadback() {
	if (!Common::RuntimeToggles::Enabled()) {
		return m_batch_readback_env;
	}
	const auto now = std::chrono::steady_clock::now();
	if (now >= m_batch_readback_next) {
		m_batch_readback_next = now + std::chrono::milliseconds(250);
		const int toggle      = Common::RuntimeToggles::Find("batch_readback");
		m_batch_readback      = toggle < 0 ? m_batch_readback_env : toggle == 1;
	}
	return m_batch_readback;
}

bool BufferCache::DownloadWindowBatch(BufferId first_id, uint64_t vaddr, uint64_t size,
                                      ReadbackBatch::Range first_window) {
	using ReadbackBatch::Range;
	auto& first = m_slot_buffers[first_id];
	const Range first_range {first.CpuAddress(), first.CpuAddress() + first.Size()};
	// The batch window: the aligned block around the access plus the prefetch on both sides.
	const auto window = ReadbackBatch::AccessWindow(vaddr, size, DownloadWindowSize,
	                                                m_batch_prefetch, {0, UINT64_MAX});
	// The access's own buffer keeps its usual window, widened by the prefetch.
	const auto own = window.Clamp(first_range);
	const Range first_part {std::min(first_window.begin, own.Empty() ? first_window.begin : own.begin),
	                        std::max(first_window.end, own.Empty() ? first_window.end : own.end)};
	const auto pieces = ReadbackBatch::SplitByOwner<BufferId>(
	    window, CACHING_PAGESIZE, first_id, first_part,
	    [this](uint64_t address) -> std::optional<std::pair<BufferId, Range>> {
		    if (!GuestRange {address, CACHING_PAGESIZE}.Valid()) {
			    return std::nullopt;
		    }
		    const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
		    if (owner == nullptr || IsBufferInvalid(*owner)) {
			    return std::nullopt;
		    }
		    const auto& buffer = m_slot_buffers[*owner];
		    return std::pair {*owner, Range {buffer.CpuAddress(), buffer.CpuAddress() + buffer.Size()}};
	    });
	std::vector<DownloadPart> parts;
	std::vector<Range>        windows;
	uint64_t                  total_size = 0;
	bool                      first_done = false;
	for (const auto& piece: pieces) {
		auto& buffer = m_slot_buffers[piece.id];
		const bool is_first = piece.id == first_id;
		// Zero-copy buffers resolve on their own fault; other buffers only when they hold
		// GPU-written bytes in the window and the open command buffer does not write them (that
		// would turn the whole batch into a flush and wait).
		if (!is_first &&
		    (buffer.IsZeroCopy() ||
		     !m_memory_tracker.IsRegionGpuModified(piece.range.begin, piece.range.Size()) ||
		     HasUnsubmittedWrite(piece.range.begin, piece.range.Size()))) {
			continue;
		}
		if (CollectDownload(buffer, piece.range.begin, piece.range.Size(), parts, total_size)) {
			windows.push_back(piece.range);
			first_done = first_done || is_first;
			if (!is_first) {
				Common::PerfSummary::Add(Common::PerfSummary::Counter::RbExtraBufs);
			}
		}
	}
	if (parts.empty()) {
		return false;
	}
	DownloadParts<false>(parts, total_size, true);
	for (const auto& range: windows) {
		m_memory_tracker.UnmarkRegionAsGpuModified(range.begin, range.Size());
	}
	return first_done;
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	MarkUnsubmittedWrite(overlap.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

bool BufferCache::ZeroCopyEnabled() const {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_ZERO_COPY");
		return text != nullptr && text[0] == '1';
	}();
	// GuestBackingImports::Available() settles on first use and is cheap after that.
	return enabled && !m_zero_copy_policy.off && (m_graphics.external_memory_host_enabled ||
	                   const_cast<GuestBackingImports&>(m_guest_imports).Available());
}

bool BufferCache::ResolveZeroCopy(Buffer& buffer, bool wait) {
	Common::PerfSummary::ScopedMicros timer(Common::PerfSummary::Counter::ZcResolveUs);
	const auto vaddr = buffer.CpuAddress();
	const auto size  = buffer.Size();
	bool       any   = false;
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size,
	                                             [&](uint64_t address, uint64_t bytes) noexcept {
		                                             any = true;
		                                             m_gpu_modified_ranges.Subtract(address, bytes);
		                                             ForgetColorMetadataChecks(address, bytes);
	                                             });
	if (!any || !wait) {
		return any;
	}
	if (buffer.last_write_tick != 0 && !m_scheduler.IsFree(buffer.last_write_tick)) {
		KYTY_PROFILER_BLOCK("BufferCache::ZeroCopyWait");
		m_scheduler.Wait(std::min(buffer.last_write_tick, m_scheduler.CurrentTick()));
	}
	// Every write to this buffer has finished: its GPU-written pages are readable again. They stay
	// write-watched, so a CPU write faults into the read fence (WaitZeroCopyReads) and queues the
	// fence for readers by address; the rest of the buffer keeps its protection.
	ReleaseGpuWrittenRuns(m_memory_tracker, vaddr, size, [](uint64_t, uint64_t bytes) {
		Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcStateKiB, bytes >> 10u);
	});
	return true;
}

void BufferCache::ReleaseZeroCopyWrites(BufferId id, uint64_t vaddr, uint64_t size, uint64_t tick) {
	// The writes are in guest memory once the tick completes. A newer write keeps the pages until
	// its own release; a buffer deleted or rebuilt since is not this one.
	if (IsBufferInvalid(id)) {
		return;
	}
	auto& buffer = m_slot_buffers[id];
	if (!buffer.IsZeroCopy() || buffer.CpuAddress() != vaddr || buffer.Size() != size ||
	    buffer.last_write_tick != tick) {
		return;
	}
	Common::PerfSummary::ScopedMicros timer(Common::PerfSummary::Counter::ZcReleaseUs);
	bool any = false;
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size,
	                                             [&](uint64_t address, uint64_t bytes) noexcept {
		                                             any = true;
		                                             m_gpu_modified_ranges.Subtract(address, bytes);
		                                             ForgetColorMetadataChecks(address, bytes);
	                                             });
	if (!any) {
		return;
	}
	// Only the GPU-written runs change, and they stay write-watched (see ResolveZeroCopy).
	ReleaseGpuWrittenRuns(m_memory_tracker, vaddr, size, [](uint64_t, uint64_t bytes) {
		Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcStateKiB, bytes >> 10u);
	});
	// A VRAM buffer would need a read-back for the CPU to see this write.
	m_zero_copy_fences.CountSaved(vaddr);
	Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcSaved);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::ZeroCopyReleases);
}

void BufferCache::FenceZeroCopyRead(BufferId id, Buffer& buffer, uint64_t /*vaddr*/,
                                    uint64_t /*size*/) {
	const auto tick = m_scheduler.CurrentTick();
	// One fence per buffer and tick: a CPU write after it faults and waits for this tick (which
	// submits it), so later binds in the same tick find the pages still protected.
	if (buffer.last_read_tick == tick) {
		Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcFenceSkips);
		return;
	}
	Common::PerfSummary::ScopedMicros timer(Common::PerfSummary::Counter::ZcFenceUs);
	Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcFences);
	buffer.last_read_tick = tick;
	// Clearing CPU-dirty state write-protects the pages the CPU wrote since; the others are
	// protected already. Nothing to upload: guest memory is the buffer. The whole buffer, as later
	// binds this tick may use other parts of it.
	m_memory_tracker.ForEachUploadRange(
	    buffer.CpuAddress(), buffer.Size(), false, [](uint64_t, uint64_t bytes) noexcept {
		    Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcStateKiB, bytes >> 10u);
	    },
	    []() noexcept {});
	const auto recorded =
	    m_zero_copy_fences.Record(buffer.CpuAddress(), buffer.CpuAddress() + buffer.Size(), tick);
	buffer.read_waits = recorded.waits;
	buffer.read_saved = recorded.saved;
	if (buffer.read_release_tick != tick) {
		buffer.read_release_tick = tick;
		m_scheduler.DeferOperation([this, id, address = buffer.CpuAddress(), bytes = buffer.Size(),
		                            tick] { ReleaseZeroCopyReads(id, address, bytes, tick); });
	}
}

void BufferCache::ReleaseZeroCopyReads(BufferId id, uint64_t vaddr, uint64_t size, uint64_t tick) {
	// Drops the fence once its tick completed, unless a newer read replaced it. The pages stay
	// write-protected: the first CPU write afterwards faults once and no longer waits, and its
	// fault queues the next submission's fence for readers by address (SynchronizeBda).
	m_zero_copy_fences.Release(vaddr, tick);
	(void)id;
	(void)size;
}

void BufferCache::WaitZeroCopyReads(uint64_t vaddr, uint64_t size) {
	const auto tick = m_zero_copy_fences.Find(vaddr, size);
	// A completion operation that writes guest memory is ordered after the GPU work already, and
	// waiting from it could wait for itself.
	if (tick == 0 || CommandScheduler::InDeferredOperation() || m_scheduler.IsFree(tick)) {
		return;
	}
	KYTY_PROFILER_BLOCK("BufferCache::ZeroCopyReadWait");
	Common::PerfSummary::ScopedMicros timer(Common::PerfSummary::Counter::ZcWaitUs);
	Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcReadWaits);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::ZeroCopyReadWaits);
	m_zero_copy_fences.CountWait(vaddr, size, tick);
	// Only the GPU thread submits the current tick. It only submits it: the writer waits for the
	// GPU on its own thread, and the GPU thread goes on with the next packets meanwhile.
	if (GuestGpu::IsGpuThread()) {
		m_scheduler.Wait(tick);
		return;
	}
	// The GPU thread may fence the buffer for a newer tick meanwhile (it no longer stops while this
	// thread waits); those reads come from packets the guest most likely queued before this write,
	// so they are waited for too (a few rounds: a buffer read by every tick would never settle).
	auto wait_tick = tick;
	for (int round = 0; round < 3 && wait_tick != 0 && !m_scheduler.IsFree(wait_tick);
	     round++, wait_tick = m_zero_copy_fences.Find(vaddr, size)) {
		if (wait_tick == m_scheduler.CurrentTick()) {
			Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcOpenWaits);
			m_scheduler.Context().GetGpu().SendCommandSync([this, wait_tick] {
				if (wait_tick == m_scheduler.CurrentTick()) {
					m_scheduler.Flush();
				}
			});
		}
		m_scheduler.Wait(wait_tick);
	}
}

static uint32_t EnvValue(const char* name, uint32_t fallback) {
	const char* text = std::getenv(name);
	if (text == nullptr || text[0] == 0) {
		return fallback;
	}
	return static_cast<uint32_t>(std::strtoul(text, nullptr, 10));
}

void BufferCache::RefreshZeroCopyPolicy() {
	static const bool requested = [] {
		const char* text = std::getenv("KYTY_ZERO_COPY");
		return text != nullptr && text[0] == '1';
	}();
	const auto tick = m_scheduler.CurrentTick();
	if (!requested || (m_zero_copy_policy.known && tick == m_zero_copy_policy.tick)) {
		return;
	}
	m_zero_copy_policy.tick = tick;
	const bool known        = m_zero_copy_policy.known;
	m_zero_copy_policy.known = true;
	// Read-waits per spared read-back above which a buffer goes back to VRAM
	// (KYTY_ZERO_COPY_WAIT_RATIO, default 0: never), once it has KYTY_ZERO_COPY_WAIT_MIN recent waits.
	// Runtime toggles (one run compares settings): zc_off=1 turns zero-copy off,
	// zc_ratio_b=1 uses KYTY_ZERO_COPY_WAIT_RATIO_B.
	static const uint32_t ratio_a  = EnvValue("KYTY_ZERO_COPY_WAIT_RATIO", 0);
	static const uint32_t ratio_b  = EnvValue("KYTY_ZERO_COPY_WAIT_RATIO_B", 8);
	static const uint32_t min      = std::max(EnvValue("KYTY_ZERO_COPY_WAIT_MIN", 8), 1u);
	bool                  off      = false;
	uint32_t              ratio    = ratio_a;
	if (Common::RuntimeToggles::Enabled()) {
		off   = Common::RuntimeToggles::Get("zc_off");
		ratio = Common::RuntimeToggles::Get("zc_ratio_b") ? ratio_b : ratio_a;
	}
	if (off != m_zero_copy_policy.off || ratio != m_zero_copy_policy.ratio) {
		if (known) {
			std::printf("Zero-copy: policy %s, wait ratio %u (min %u waits)\n", off ? "off" : "on",
			            ratio, min);
			std::fflush(stdout);
		}
		// Every buffer is judged afresh under the new setting.
		m_zero_copy_rejected.Clear();
		m_zero_copy_fences.ResetCounts();
	}
	m_zero_copy_policy.off       = off;
	m_zero_copy_policy.ratio     = ratio;
	m_zero_copy_policy.min_waits = min;
}

bool BufferCache::ZeroCopyKeepsWaiting(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	if (!m_zero_copy_policy.off) {
		// As of its last fence.
		const auto waits = buffer.read_waits;
		const auto ratio = m_zero_copy_policy.ratio;
		if (ratio == 0 || waits < m_zero_copy_policy.min_waits ||
		    waits <= static_cast<uint64_t>(ratio) * buffer.read_saved) {
			return false;
		}
	}
	// The CPU keeps rewriting memory the GPU is still reading, far more often than the buffer
	// spares a read-back: a VRAM copy (a snapshot per upload) is cheaper there than waiting. The
	// range is not judged again under this setting.
	const auto vaddr = buffer.CpuAddress();
	const auto size  = buffer.Size();
	if (!m_zero_copy_policy.off && m_zero_copy_rejected_logs++ < 64) {
		std::printf("Zero-copy: buffer 0x%011llx+0x%llx made the CPU wait %u times for %u spared "
		            "read-backs; back to VRAM\n",
		            static_cast<unsigned long long>(vaddr), static_cast<unsigned long long>(size),
		            buffer.read_waits, buffer.read_saved);
		std::fflush(stdout);
	}
	if (!m_zero_copy_policy.off) {
		Common::PerfSummary::Add(Common::PerfSummary::Counter::ZcWaitDrops);
		m_zero_copy_rejected.Add(vaddr, size);
	}
	DropZeroCopyBuffer(id);
	m_zero_copy_fences.Forget(vaddr);
	return true;
}

void BufferCache::JoinZeroCopy(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	if (overlap.IsZeroCopy()) {
		// Same guest memory: nothing to move, only its pending writes.
		new_buffer.last_write_tick = std::max(new_buffer.last_write_tick, overlap.last_write_tick);
	} else {
		// Only bytes the GPU wrote are newer in the old buffer; guest memory holds the rest and
		// may hold CPU writes the old buffer never received, so a whole copy would undo them.
		bool copied = false;
		m_gpu_modified_ranges.ForEachInRange(
		    overlap.CpuAddress(), overlap.Size(), [&](uint64_t start, uint64_t end) {
			    new_buffer.CopyFrom(m_scheduler.Current(), overlap, start - overlap.CpuAddress(),
			                        start - new_buffer.CpuAddress(), end - start);
			    copied = true;
		    });
		if (copied) {
			new_buffer.last_write_tick = m_scheduler.CurrentTick();
		}
	}
	DeleteBuffer(overlap_id);
}

// Read-back faults before a buffer is judged for zero-copy; KYTY_ZERO_COPY_THRESHOLD=<n> changes it.
static uint32_t ZeroCopyThreshold() {
	static const uint32_t threshold = [] {
		const char* text  = std::getenv("KYTY_ZERO_COPY_THRESHOLD");
		const auto  value = text != nullptr ? std::strtoul(text, nullptr, 10) : 0ul;
		return value != 0 ? static_cast<uint32_t>(value) : 16u;
	}();
	return threshold;
}

void BufferCache::MaybeMigrateToZeroCopy(BufferId id) {
	const uint32_t     Threshold = ZeroCopyThreshold();
	constexpr uint64_t MaxSize   = 512ull * 1024 * 1024;
	auto&              buffer    = m_slot_buffers[id];
	if (buffer.IsZeroCopy() || buffer.readback_faults < Threshold ||
	    buffer.zero_copy_judgment == 2) {
		return;
	}
	const auto vaddr = buffer.CpuAddress();
	const auto size  = buffer.Size();
	bool       images = false;
	{
		// The texture cache never holds its lock across ReadMemory (textureCache.cpp).
		std::scoped_lock lock {m_texture_cache.m_lock};
		m_texture_cache.ForEachImageInRegion(vaddr, size, [&](ImageId, Image&) {
			images = true;
			return true;
		});
	}
	if (images || size > MaxSize || m_zero_copy_rejected.Intersects(vaddr, size) ||
	    LibKernel::Memory::ContiguousBackingAlias(vaddr, size) == 0) {
		buffer.zero_copy_judgment = 2;
		return;
	}
	m_zero_copy.Add(vaddr, size);
	// CreateBuffer joins the old buffer into one over guest memory.
	(void)CreateBuffer(vaddr, size);
}

bool BufferCache::ZeroCopyBackingValid(BufferId id) {
	auto&      buffer     = m_slot_buffers[id];
	const auto generation = LibKernel::Memory::BackingGeneration();
	if (buffer.backing_generation == generation) {
		return true;
	}
	// Guest mappings change all the time; walk this buffer's pages only when a change touched it.
	if (!LibKernel::Memory::BackingChangedSince(buffer.backing_generation, buffer.CpuAddress(),
	                                            buffer.Size()) ||
	    LibKernel::Memory::ContiguousBackingAlias(buffer.CpuAddress(), buffer.Size()) ==
	        buffer.HostPointer()) {
		buffer.backing_generation = generation;
		return true;
	}
	std::printf("Zero-copy: buffer 0x%011llx+0x%llx lost its backing; back to VRAM\n",
	            static_cast<unsigned long long>(buffer.CpuAddress()),
	            static_cast<unsigned long long>(buffer.Size()));
	std::fflush(stdout);
	DropZeroCopyBuffer(id);
	return false;
}

void BufferCache::DropZeroCopyBuffer(BufferId id) {
	auto&      buffer = m_slot_buffers[id];
	const auto vaddr  = buffer.CpuAddress();
	const auto size   = buffer.Size();
	(void)ResolveZeroCopy(buffer, true);
	// Its read fence stays until its tick completes: a CPU write still waits for reads in flight.
	m_zero_copy.Subtract(vaddr, size);
	m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
	DeleteBuffer(id);
}

void BufferCache::UnmapZeroCopy(uint64_t vaddr, uint64_t size) {
	if (m_zero_copy.Empty() || !m_zero_copy.Intersects(vaddr, size)) {
		return;
	}
	std::vector<BufferId> drop;
	auto                  it = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < vaddr + size; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		if (buffer.IsZeroCopy() && buffer.CpuAddress() + buffer.Size() > vaddr) {
			drop.push_back(it->second);
		}
	}
	for (const auto id: drop) {
		DropZeroCopyBuffer(id);
	}
	m_zero_copy.Subtract(vaddr, size);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto     overlap = ResolveOverlaps(vaddr, size);
	const auto     length  = overlap.end - overlap.begin;
	constexpr auto flags   = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;

	BufferId id {};
	bool     zero_copy = false;
	if (!m_zero_copy.Empty() && !m_zero_copy_policy.off &&
	    m_zero_copy.Intersects(overlap.begin, length)) {
		constexpr uint64_t MaxZeroCopy = 512ull * 1024 * 1024;
		const auto         generation  = LibKernel::Memory::BackingGeneration();
		const auto         alias =
		    length <= MaxZeroCopy ? LibKernel::Memory::ContiguousBackingAlias(overlap.begin, length) : 0;
		if (alias != 0 && (alias & 0xfffu) == 0) {
			// The backing's own Vulkan memory where it has some (KYTY_VK_GUEST_BACKING=1),
			// otherwise a host-pointer import (VK_EXT_external_memory_host).
			bool inserted = false;
			if (const auto imported = m_guest_imports.Find(alias, length)) {
				id       = m_slot_buffers.insert(m_graphics, m_scheduler, overlap.begin,
				                                 reinterpret_cast<void*>(alias), imported->memory,
				                                 imported->offset, imported->memory_size,
				                                 imported->memory_type, imported->handle_type, flags,
				                                 length);
				inserted = true;
			} else if (m_graphics.external_memory_host_enabled) {
				id       = m_slot_buffers.insert(m_graphics, m_scheduler, overlap.begin,
				                                 reinterpret_cast<void*>(alias), flags, length);
				inserted = true;
			}
			zero_copy = inserted && m_slot_buffers[id].IsZeroCopy();
			if (zero_copy) {
				m_slot_buffers[id].backing_generation = generation;
			} else if (inserted) {
				m_slot_buffers.erase(id);
			}
		}
		if (zero_copy) {
			m_zero_copy.Add(overlap.begin, length);
		} else {
			m_zero_copy.Subtract(overlap.begin, length);
		}
		if (++m_zero_copy_created <= 64 || !zero_copy || (m_zero_copy_created & 255u) == 0) {
			std::printf("Zero-copy: #%llu buffer 0x%011llx+0x%llx %s\n",
			            static_cast<unsigned long long>(m_zero_copy_created),
			            static_cast<unsigned long long>(overlap.begin),
			            static_cast<unsigned long long>(length),
			            zero_copy    ? "imports guest memory"
			            : alias == 0 ? "stays in VRAM (backing not contiguous or too large)"
			                         : "stays in VRAM (import refused)");
			std::fflush(stdout);
		}
	}
	if (!zero_copy) {
		id = m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
		                           flags, length);
	}
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}{}]", overlap.begin, length,
	                     zero_copy ? " zero-copy" : "");
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		if (zero_copy) {
			JoinZeroCopy(id, old_id, !overlap.has_stream_leap);
		} else {
			JoinOverlap(id, old_id, !overlap.has_stream_leap);
		}
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    ImageId image) {
	if (buffer.IsZeroCopy()) {
		// Guest memory is the buffer: nothing to upload. A write still marks the range GPU-owned
		// until it finishes, which clears its CPU-dirty state first (the tracker keeps them apart).
		const bool from_image = image && SynchronizeBufferFromImage(buffer, image);
		if (is_written || from_image) {
			m_memory_tracker.ForEachUploadRange(
			    vaddr, size, true, [](uint64_t, uint64_t) noexcept {}, []() noexcept {});
			buffer.last_write_tick = m_scheduler.CurrentTick();
		}
		return from_image;
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept {
		    const BufferUpload upload {&buffer, copies.size()};
		    UploadCopies({&upload, 1}, copies, total_size);
	    });
	if (image) {
		return SynchronizeBufferFromImage(buffer, image);
	}
	return false;
}

void BufferCache::RecordUploadStats(uint64_t address, const uint8_t* data, uint64_t size) {
	m_stats_downloaded.ForEachInRange(address, size, [](uint64_t start, uint64_t end) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::SyncReuploadBytes, end - start);
	});
	m_stats_downloaded.Subtract(address, size);
	// Whole 4 KiB pages only, keyed by guest page.
	const auto end = address + size;
	for (auto page = Common::AlignUp(address, TRACKER_PAGE_SIZE); page + TRACKER_PAGE_SIZE <= end;
	     page += TRACKER_PAGE_SIZE) {
		const auto hash          = XXH3_64bits(data + (page - address), TRACKER_PAGE_SIZE);
		auto [entry, inserted]   = m_stats_upload_hashes.try_emplace(page, hash);
		if (!inserted) {
			if (entry->second == hash) {
				Common::DebugCounters::Add(
				    Common::DebugCounters::Counter::SyncUnchangedUploadBytes, TRACKER_PAGE_SIZE);
			}
			entry->second = hash;
		}
	}
}

void BufferCache::UploadCopies(std::span<const BufferUpload> uploads,
                               std::span<vk::BufferCopy> copies, uint64_t total_size) {
	if (copies.empty()) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(
	    m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		    size_t first = 0;
		    for (const auto& upload: uploads) {
			    for (auto& copy: copies.subspan(first, upload.end - first)) {
				    const auto address = upload.buffer->CpuAddress() + copy.dstOffset;
				    if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset,
					                                         copy.size)) {
					    EXIT("BufferCache: failed to read sparse guest buffer memory\n");
				    }
				    if (DetailedSyncStats()) {
					    RecordUploadStats(address, mapped + copy.srcOffset, copy.size);
				    }
				    copy.srcOffset += base_offset;
			    }
			    first = upload.end;
		    }
	    });
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	const vk::MemoryBarrier before {
	    .srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                 vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
	    .dstAccessMask = vk::AccessFlagBits::eTransferWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       1, &before, 0, nullptr, 0, nullptr);
	size_t first = 0;
	for (const auto& upload: uploads) {
		native.copyBuffer(source.first->Handle(), upload.buffer->Handle(),
		                  static_cast<uint32_t>(upload.end - first), copies.data() + first);
		first = upload.end;
	}
	const vk::MemoryBarrier after {.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
	                               .dstAccessMask = vk::AccessFlagBits::eMemoryRead |
	                                                vk::AccessFlagBits::eMemoryWrite};
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands,
	                       vk::DependencyFlagBits::eByRegion, 1, &after, 0, nullptr, 0, nullptr);
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	Common::PerfSummary::Add(Common::PerfSummary::Counter::BufferLookups);
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));

	const auto image             = is_texel_buffer && !is_written
	                                   ? m_texture_cache.FindImageFromRange(vaddr, size)
	                                   : ImageId {};
	auto       end               = vaddr + size;
	bool       written_past_read = false;
	if (image) {
		const auto image_end = [&] {
			std::scoped_lock lock {m_texture_cache.m_lock};
			const auto&      range = m_texture_cache.m_slot_images[image].info.data;
			return range.address == vaddr ? range.End() : end;
		}();
		if (image_end > end) {
			written_past_read = m_memory_tracker.IsRegionGpuModified(end, image_end - end);
			end               = written_past_read ? end : image_end;
		}
	}
	if (!is_written && size <= CACHING_PAGESIZE &&
	    (m_zero_copy.Empty() || !m_zero_copy.Intersects(vaddr, size)) &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size) && (!image || written_past_read)) {
		// Within one draw window the guest has no signal to rewrite the range. A copy made in this
		// tick at most an eighth of a lap ago is nearly a lap ahead of the ring's writes, like a
		// fresh copy, even if the draw's preparation submits the tick.
		auto& copy = m_stream_copies[(vaddr * 0x9e3779b97f4a7c15ull) >> (64u - STREAM_COPY_BITS)];
		const auto tick = m_scheduler.CurrentTick();
		if (copy.vaddr == vaddr && copy.size == size && copy.window == m_draw_window &&
		    copy.tick == tick &&
		    m_stream_buffer.Position() - copy.position <= m_stream_buffer.Size() / 8) {
			return {&m_stream_buffer, copy.offset};
		}
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			Common::DebugCounters::Add(Common::DebugCounters::Counter::SyncStreamBytes, size);
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			copy = {vaddr, size, m_draw_window, tick, offset, m_stream_buffer.Position()};
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, end - vaddr)) {
		id = FindBuffer(vaddr, end - vaddr);
	}
	RefreshZeroCopyPolicy();
	if (m_slot_buffers[id].IsZeroCopy() && (!ZeroCopyBackingValid(id) || ZeroCopyKeepsWaiting(id))) {
		id = FindBuffer(vaddr, end - vaddr);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, end - vaddr, is_written, image);
	if (is_written) {
		buffer.probe_write_tick = m_scheduler.CurrentTick();
	}
	if (!is_written && buffer.IsZeroCopy()) {
		FenceZeroCopyRead(id, buffer, vaddr, end - vaddr);
	}
	if (is_written && buffer.IsZeroCopy() && buffer.release_tick != buffer.last_write_tick) {
		static const bool release = [] {
			const char* text = std::getenv("KYTY_ZERO_COPY_RELEASE");
			return text == nullptr || text[0] != '0';
		}();
		if (release) {
			buffer.release_tick = buffer.last_write_tick;
			m_scheduler.DeferOperation([this, id, address = buffer.CpuAddress(),
			                            bytes = buffer.Size(), tick = buffer.last_write_tick] {
				ReleaseZeroCopyWrites(id, address, bytes, tick);
			});
		}
	}
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		m_pending_writes.Add(vaddr, size);
		m_sync_windows[m_guest_queue].writes.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, {});
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		(void)m_scheduler.Context().InvalidateMemory(vaddr, size);
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	// A fill writes every byte, after the GPU writes of the images it covers.
	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size, true);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	RecordPendingWrites();
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	// A copy into whole levels of a GPU-written image lands on its GPU contents, not stale guest
	// bytes, while no buffer holds the range.
	if (src_memory && dst_memory && !IsRegionRegistered(dst_vaddr, size) &&
	    m_texture_cache.CopyToImage(dst_vaddr, src_vaddr, size)) {
		return;
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size, true);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
	RecordPendingWrites();
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Contains(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::NoteGpuWrites(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid GPU write range\n");
	}
	const auto end = vaddr + size;
	m_sync_windows[m_guest_queue].writes.Add(vaddr, size);
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		const auto& buffer = m_slot_buffers[it->second];
		const auto  start  = std::max(buffer.CpuAddress(), vaddr);
		const auto  finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start >= finish) {
			continue;
		}
		TouchBuffer(buffer);
		m_memory_tracker.MarkRegionAsGpuModifiedUnlessCpuDirty(
		    start, finish - start, [&](uint64_t page, uint64_t bytes) noexcept {
			    const auto first = std::max(page, start);
			    const auto last  = std::min(page + bytes, finish);
			    if (first < last) {
				    m_gpu_modified_ranges.Add(first, last - first);
				    ForgetColorMetadataChecks(first, last - first);
				    // Images over the written bytes are stale, as for other buffer writes.
				    m_texture_cache.InvalidateMemoryFromGPU(first, last - first);
			    }
		    });
	}
}

void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	m_bda_sync.Add(begin, end - begin);
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	const auto queued = [this] {
		std::lock_guard lock(m_bda_sync_mutex);
		return std::exchange(m_bda_sync, {});
	}();
	std::vector<vk::BufferCopy> copies;
	std::vector<BufferUpload>   uploads;
	uint64_t                    total_size = 0;
	queued.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
			auto it = m_buffers.upper_bound(start);
			if (it != m_buffers.begin()) {
				--it;
			}
			for (; it != m_buffers.end() && it->first < finish; ++it) {
				auto& buffer = m_slot_buffers[it->second];
				const auto first = std::max(buffer.CpuAddress(), start);
				const auto last   = std::min(buffer.CpuAddress() + buffer.Size(), finish);
				if (buffer.IsZeroCopy()) {
					// The GPU reads guest memory itself, through addresses too: the CPU's
					// writes since the last submission are visible already, and further ones
					// wait while this submission may still read them.
					if (first < last) {
						FenceZeroCopyRead(it->second, buffer, first, last - first);
					}
					continue;
				}
				if (first < last) {
					// Reads follow every range's protection; a write in between re-queues its page.
					m_memory_tracker.ForEachUploadRange(
					    first, last - first, false,
					    [&](uint64_t address, uint64_t bytes) noexcept {
						    if (uploads.empty() || uploads.back().buffer != &buffer) {
							    uploads.push_back({&buffer, 0});
						    }
						    copies.emplace_back(total_size, buffer.Offset(address), bytes);
						    uploads.back().end = copies.size();
						    total_size += bytes;
					    },
					    []() noexcept {});
				}
			}
		});
	});
	Common::DebugCounters::Add(Common::DebugCounters::Counter::SyncBdaUploadBytes, total_size);
	UploadCopies(uploads, copies, total_size);
}

} // namespace Libs::Graphics
