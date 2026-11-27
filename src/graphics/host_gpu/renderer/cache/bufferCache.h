#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/readbackBatch.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// The fault buffer holds the fault bitmap (pages, then heap rows) and then the lock words of
	// emulated image atomics (IR::EmulatedImageAtomicBits), which the bitmap scan never touches.
	static constexpr uint64_t FAULT_BITMAP_BYTES =
	    (CACHING_NUMPAGES + ShaderRecompiler::IR::HeapRowBits) / 8;
	static constexpr uint64_t IMAGE_ATOMIC_LOCK_DWORD = FAULT_BITMAP_BYTES / 4;
	// After them, at a storage-offset-aligned position, the write bitmap: one bit per caching
	// page that a device-address store (indirect buffer store or atomic) wrote. (Port of TheCruZ's
	// write bitmap, KytyPS5 PR #2469 line 427a75fc.)
	static constexpr uint64_t WRITE_BITMAP_OFFSET =
	    (FAULT_BITMAP_BYTES + ShaderRecompiler::IR::ImageAtomicLockWords * 4 + 255) & ~uint64_t {255};
	static constexpr uint64_t WRITE_BITMAP_BYTES = ((CACHING_NUMPAGES + 127) / 128) * 16;
	static constexpr uint64_t FAULT_BUFFER_BYTES  = WRITE_BITMAP_OFFSET + WRITE_BITMAP_BYTES;
	static_assert(FAULT_BITMAP_BYTES % 16 == 0 && IMAGE_ATOMIC_LOCK_DWORD < UINT32_MAX);
	static_assert(WRITE_BITMAP_OFFSET / 4 + WRITE_BITMAP_BYTES / 4 < UINT32_MAX);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// After a guest write fault: the clean pages of the window become CPU-modified too, so the
	// guest's next writes there do not fault one page at a time.
	void                   InvalidateWriteWindow(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	// Ends the draw window, whose draws share the stream copies of CPU-written ranges.
	void                   EndDrawWindow() noexcept { m_draw_window++; }
	[[nodiscard]] uint64_t DrawWindow() const noexcept { return m_draw_window; }
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] HeapRows& GetHeapRows() noexcept { return m_fault_manager.GetHeapRows(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// Guest memory that GPU work recorded on a guest queue since that queue's last synchronisation
	// point (a partial flush, ACQUIRE_MEM, WAIT_REG_MEM, a memory semaphore, RELEASE_MEM, a flip,
	// the end of a submission) may write. On the console that work can still be running when the
	// queue's command processor reads its next packet's indirect arguments, so those reads see
	// memory as it was before it. Other queues' packets do not stall this one.
	void SetGuestQueue(uint32_t queue) noexcept { m_guest_queue = queue; }
	void NoteGuestSync() noexcept {
		auto& window = m_sync_windows[m_guest_queue];
		window.writes.Clear();
		window.generation++;
	}
	[[nodiscard]] bool WrittenSinceGuestSync(uint64_t vaddr, uint64_t size) const {
		const auto it = m_sync_windows.find(m_guest_queue);
		return it != m_sync_windows.end() && it->second.writes.Intersects(vaddr, size);
	}
	[[nodiscard]] uint32_t GuestQueue() const noexcept { return m_guest_queue; }
	[[nodiscard]] uint64_t SyncGeneration() const {
		const auto it = m_sync_windows.find(m_guest_queue);
		return it != m_sync_windows.end() ? it->second.generation : 0;
	}
	[[nodiscard]] bool HasOnlyGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               ProcessFaultBuffer();
	// Records stores a shader made through device addresses, known once it completed: the bytes
	// of [vaddr, vaddr + size) a cached buffer holds become GPU modified, except on pages the CPU
	// dirtied since, whose upload replaces them.
	void               NoteGpuWrites(uint64_t vaddr, uint64_t size);
	// Uploads the CPU-dirty bytes of the queued ranges `mapped` covers behind one barrier pair.
	void SynchronizeBda(const RangeSet& mapped);
	// Zero-copy buffers over an unmapped range are dropped before their backing is reused.
	void               UnmapZeroCopy(uint64_t vaddr, uint64_t size);
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	void               QueueBdaSync(uint64_t vaddr, uint64_t size);
	void               RunGarbageCollector();
	// After a draw, dispatch, fill or copy is recorded: the ranges bound for writing since the last
	// call, and address writes noted since, belong to the current command buffer.
	void RecordPendingWrites();
	// A program with address writes is bound; its writes can reach any range.
	void NoteAddressWrites() noexcept { m_address_writes_pending = true; }
	// Called on the GPU thread when a guest CPU write faults into GPU-written memory.
	void SetCpuWriteListener(std::function<void(uint64_t, uint64_t)> listener) {
		m_cpu_write_listener = std::move(listener);
	}

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	// Copies [previous upload's end, end) go to `buffer`.
	struct BufferUpload {
		Buffer* buffer;
		size_t  end;
	};

	// The stream copy of a CPU-written range, in a slot picked by its address.
	struct StreamCopy {
		uint64_t vaddr    = 0;
		uint64_t size     = 0;
		uint64_t window   = 0;
		uint64_t tick     = 0;
		uint64_t offset   = 0;
		uint64_t position = 0;
	};
	static constexpr uint32_t STREAM_COPY_BITS = 10;

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, Common::SlotId image);
	void UploadCopies(std::span<const BufferUpload> uploads, std::span<vk::BufferCopy> copies,
	                  uint64_t total_size);
	void RecordUploadStats(uint64_t address, const uint8_t* data, uint64_t size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, Common::SlotId id);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// One buffer's ranges of a download, packed into the download buffer at dstOffset.
	struct DownloadPart {
		Buffer*                     buffer;
		uint64_t                    buffer_address;
		std::vector<vk::BufferCopy> copies;
	};
	// Takes the GPU-modified ranges of [vaddr, vaddr + size) out of the dirty set into a new part
	// (false when there is none). Not for zero-copy buffers.
	[[nodiscard]] bool CollectDownload(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                   std::vector<DownloadPart>& parts, uint64_t& total_size);
	// Copies the parts to the host in one command sequence and publishes them into guest memory:
	// synchronously (before returning) or, `async`, at the tick's completion. `fuse`: a detached
	// copy goes into the same submission as the open command buffer.
	template <bool async>
	void DownloadParts(std::vector<DownloadPart>& parts, uint64_t total_size, bool fuse);
	// KYTY_BATCH_READBACK (default 1; 0 = off) or toggle batch_readback (re-read 4x per second): one synchronous
	// download copies every buffer of the access's window, in the open command buffer's
	// submission when it can run detached.
	[[nodiscard]] bool BatchReadback();
	// KYTY_COPY_QUEUE (default 1; the device then has a copy queue), overridden by toggle copy_queue when
	// the toggle file has that line (re-read 4x per second): detached synchronous downloads copy
	// on the copy queue (ReadbackBatch::ReadbackPath::CopyQueue).
	[[nodiscard]] bool        CopyQueue();
	// KYTY_COPY_QUEUE_FLUSH=1: a copy-queue download still submits the open command buffer.
	[[nodiscard]] static bool CopyQueueFlush();
	// KYTY_RB_QUEUE_PROBE=1 (with KYTY_PERF_SUMMARY): measures the GPU queue ahead of each
	// synchronous download and waits for its producing tick first, timed on its own.
	[[nodiscard]] static bool QueueProbe();
	void                      ProbeReadbackQueue(const std::vector<DownloadPart>& parts);
	// The batched download of a CPU access at [vaddr, vaddr + size) in buffer `first_id`, whose
	// own window is `first_window`: that window and the GPU-written bytes of every other
	// non-zero-copy buffer in the access's aligned window, in one copy sequence, then all those
	// windows are returned to the CPU. True when the first buffer had bytes to copy.
	[[nodiscard]] bool DownloadWindowBatch(BufferId first_id, uint64_t vaddr, uint64_t size,
	                                       ReadbackBatch::Range first_window);
	// GPU writes recorded in the current command buffer, or bound and not recorded yet. A
	// synchronous download of ranges without such writes reads them in a detached submission
	// behind the submitted work, without flushing the current command buffer.
	void               MarkUnsubmittedWrite(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasUnsubmittedWrite(uint64_t vaddr, uint64_t size) const;
	[[nodiscard]] bool CanDownloadDetached(uint64_t                        buffer_address,
	                                       std::span<const vk::BufferCopy> copies);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);

	// Zero-copy guest memory (KYTY_ZERO_COPY=1, VK_EXT_external_memory_host). A buffer the CPU
	// keeps faulting on is rebuilt over guest memory itself, so GPU writes land there. Its pages
	// still count as GPU-modified while a write is pending (labels reach the guest before the
	// host GPU finishes), but a fault on them only waits for the buffer's last write instead of
	// flushing all recorded work and copying bytes back.
	[[nodiscard]] bool ZeroCopyEnabled() const;
	// Settles the pending GPU writes of a zero-copy buffer: false when none are tracked. With
	// `wait`, waits for its last write and returns its pages to the CPU.
	[[nodiscard]] bool ResolveZeroCopy(Buffer& buffer, bool wait);
	// When `tick` completes and it was the buffer's last write, returns its pages to the CPU, so
	// a later CPU access does not fault (KYTY_ZERO_COPY_RELEASE=0 waits for the fault instead).
	void               ReleaseZeroCopyWrites(BufferId id, uint64_t vaddr, uint64_t size, uint64_t tick);
	void               JoinZeroCopy(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	void               MaybeMigrateToZeroCopy(BufferId id);
	// False (and the buffer dropped) when the guest range moved to other backing.
	[[nodiscard]] bool ZeroCopyBackingValid(BufferId id);
	void               DropZeroCopyBuffer(BufferId id);
	// Zero-copy reads: the GPU reads guest memory itself, and labels reach the guest before the host
	// GPU is done, so the guest may rewrite memory a submitted read still uses (a VRAM copy was a
	// snapshot). A GPU read write-protects the buffer's pages until its tick completes; a CPU write
	// in between waits for that tick (any thread), then proceeds.
	void               FenceZeroCopyRead(BufferId id, Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ReleaseZeroCopyReads(BufferId id, uint64_t vaddr, uint64_t size, uint64_t tick);
	void               WaitZeroCopyReads(uint64_t vaddr, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	// Declared before the buffers, so it outlives the zero-copy buffers bound to its memory.
	GuestBackingImports                               m_guest_imports;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	RangeSet                                          m_pending_writes;
	struct SyncWindow {
		RangeSet writes;
		uint64_t generation = 0;
	};
	std::map<uint32_t, SyncWindow>                    m_sync_windows;
	uint32_t                                          m_guest_queue = 0;
	std::function<void(uint64_t, uint64_t)>           m_cpu_write_listener;
	RangeSet                                          m_unsubmitted_writes;
	uint64_t                                          m_unsubmitted_tick       = 0;
	uint64_t                                          m_address_write_tick     = 0;
	bool                                              m_address_writes_pending = false;
	enum class DetachedReadback { Off, Submit, NoSubmit };
	DetachedReadback                                  m_detached_readback = DetachedReadback::Submit;
	// KYTY_BATCH_READBACK (default 1), overridden by the batch_readback toggle when present.
	bool                                              m_batch_readback_env = true;
	bool                                              m_batch_readback     = true;
	std::chrono::steady_clock::time_point             m_batch_readback_next {};
	bool                                              m_copy_queue = true;
	std::chrono::steady_clock::time_point             m_copy_queue_next {};
	// KYTY_BATCH_READBACK_PREFETCH=KiB: a batch also copies this much on both sides of the window.
	uint64_t                                          m_batch_prefetch = 0;
	ReadbackBatch::NeighbourHistory                   m_readback_neighbours;
	// Statistics only (KYTY_SYNC_STATS): bytes last downloaded, and hashes of uploaded pages.
	RangeSet                                          m_stats_downloaded;
	std::unordered_map<uint64_t, uint64_t>            m_stats_upload_hashes;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	MemoryTracker                                     m_memory_tracker;
	std::mutex                                        m_bda_sync_mutex;
	RangeSet                                          m_bda_sync;
	// Guest ranges whose buffers are zero-copy; a buffer created over them imports guest memory.
	RangeSet                                          m_zero_copy;
	uint64_t                                          m_zero_copy_created = 0;
	// Zero-copy read fences by buffer start: {end, tick}. Written on the GPU thread, read by
	// faulting threads.
	struct ZeroCopyFence {
		uint64_t end  = 0;
		uint64_t tick = 0;
	};
	std::mutex                                        m_zero_copy_fence_mutex;
	std::map<uint64_t, ZeroCopyFence>                 m_zero_copy_fences;
	std::atomic<bool>                                 m_any_zero_copy_fence {false};
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	std::array<StreamCopy, 1u << STREAM_COPY_BITS>    m_stream_copies {};
	uint64_t                                          m_draw_window = 0;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
