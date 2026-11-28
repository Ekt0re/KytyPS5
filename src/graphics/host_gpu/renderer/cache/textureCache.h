#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "common/smallVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/resolveStamp.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <algorithm>
#include <bitset>
#include <cstdlib>
#include <map>
#include <mutex>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
class ResolveCache;
class TextureCache;
struct TextureCacheTestAccess;

namespace BcAlias {
[[nodiscard]] bool Matches(const ImageInfo& requested, const ImageInfo& cached);
[[nodiscard]] bool Publish(TextureCache& cache, Image& image);
void Resolve(TextureCache& cache, ImageId cached, ImageId merged);
} // namespace BcAlias

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
		bool          heap = false; // PS5: resolved for a resident image heap.
		// PS5: set by FindImage.
		ResolveStamp stamp;
	};

	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache, ResolveCache& resolve_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	// full_write: the GPU write covers every byte of the range, so images inside it take the
	// written data even when they hold GPU writes of their own.
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size,
	                                           bool full_write = false);
	// Writes a copy into whole levels of one layer of a GPU-written image straight into it, and
	// returns false for any other copy.
	[[nodiscard]] bool CopyToImage(uint64_t address, uint64_t source, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	void ProcessDownloadImages();
	// With storage readback, a GL2 writeback makes GPU-written images visible to the CPU; returns
	// whether a download was recorded.
	[[nodiscard]] bool WriteBackImages();
	// The tick of the latest write-back pending over the range's pages, or 0 when none is.
	[[nodiscard]] uint64_t WriteBackTick(uint64_t address, uint64_t size);
	void RunGarbageCollector();

private:
	struct MetaDataInfo {
		// PS5 DB_DEPTH_VIEW encodes 13-bit base and last array-slice indices.
		std::bitset<8192> clear_mask;
		// The depth image that last acquired this HTile range.
		ImageId owner;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	// A recorded image download whose guest pages stay inaccessible until it lands.
	struct WriteBack {
		GuestRange range;
		uint64_t   tick;
	};

	// PS5: guest addresses span 44 bits, so an entry keeps the whole image address.
	struct BucketEntry {
		uint64_t address;
		uint32_t size;
		ImageId  id;

		[[nodiscard]] bool Overlaps(uint64_t addr, size_t size) const noexcept {
			return address < addr + size && addr < address + this->size;
		}
	};

	struct alignas(64) Bucket {
		Common::SmallVector<BucketEntry, 3, uint32_t> entries;
	};
	static_assert(sizeof(Bucket) == 64);

	using ImagePageTable = MultiLevelPageTable<Bucket, 20, 44, 14>;

	// Callers pass a nonempty, valid guest range.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		static constexpr bool RETURNS_BOOL =
		    std::is_same_v<std::invoke_result_t<Func, uint64_t>, bool>;
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			if constexpr (RETURNS_BOOL) {
				if (func(page)) {
					break;
				}
			} else {
				func(page);
			}
		}
	}

	// The walks below iterate the page entries in place: func must not insert, register,
	// unregister or free images. Caller holds m_lock.

	// Calls func once per image overlapping the region, from the first page both share; a func
	// returning true stops the walk.
	template <typename Func>
	void ForEachImageInRegion(uint64_t address, size_t size, Func&& func) {
		static constexpr bool BOOL_BREAK =
		    std::is_same_v<std::invoke_result_t<Func, ImageId, Image&>, bool>;
		ImagePageTable::PageRange pages {};
		if (!ImagePageTable::TryGetPageRange(address, size, pages)) {
			return;
		}
		const uint64_t first_page = address >> ImagePageTable::kPageBits;
		ForEachPage(address, size, [&](uint64_t page) {
			const auto* bucket = m_image_page_table.Find(page);
			if (bucket == nullptr) {
				if constexpr (BOOL_BREAK) {
					return false;
				} else {
					return;
				}
			}
			for (const auto& entry: bucket->entries) {
				const uint64_t base_page = entry.address >> ImagePageTable::kPageBits;
				if (page != std::max(first_page, base_page)) {
					continue;
				}
				if (!entry.Overlaps(address, size)) {
					continue;
				}
				Image& image = m_slot_images[entry.id];
				if constexpr (BOOL_BREAK) {
					if (func(entry.id, image)) {
						return true;
					}
				} else {
					func(entry.id, image);
				}
			}
			if constexpr (BOOL_BREAK) {
				return false;
			}
		});
	}

	// Calls func for each image registered at address.
	template <typename Func>
	void ForEachImageWithAddress(uint64_t address, Func&& func) {
		const auto* bucket = m_image_page_table.Find(address >> ImagePageTable::kPageBits);
		if (bucket == nullptr) {
			return;
		}
		for (const auto& entry: bucket->entries) {
			if (entry.address == address) {
				func(entry.id, m_slot_images[entry.id]);
			}
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	[[nodiscard]] bool        ReleaseIdleImages(GuestRange requested);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	// DeleteImage without the release of the slot: frees its stencil associations, drops its
	// download and HTile ownership and unregisters it. Caller holds m_lock.
	void                      DetachImage(ImageId id);
	void                      FreeImage(ImageId id);
	// Parked images (PS5, GTA VI's depth "repoint storm": ResolveDepthOverlap converts the same
	// depth image between D16, D16S8 and R16 dozens of times a frame). With parking on, the image
	// a depth conversion replaces is kept, detached and unregistered, instead of destroyed, and a
	// later conversion to the same shape (SameParkedShape) takes it back instead of creating an
	// image. Parked images are destroyed beyond the image or MiB cap (oldest first), after
	// m_park_max_frames suspend points, when their memory is unmapped, when parking is switched
	// off, and all at once when an image cannot be allocated. Callers hold m_lock.
	struct ParkedImage {
		ImageId  id;
		uint64_t gc_tick = 0;
		uint64_t bytes   = 0;
	};
	// Parks the image if parking is on and it qualifies, otherwise frees it.
	void                  RetireImage(ImageId id);
	[[nodiscard]] ImageId UnparkImage(const ImageInfo& info);
	void                  DiscardParkedImage(const ParkedImage& parked);
	// Discards the parked images over the range, or (size 0) those parked too long.
	void                  DiscardParkedImages(uint64_t address, uint64_t size);
	void                  DiscardAllParkedImages();
	// Destroys every parked image now (after the GPU is done with them); false if none was.
	[[nodiscard]] bool    ReleaseParkedImages();
	[[nodiscard]] static bool SameParkedShape(const ImageInfo& parked, const ImageInfo& wanted);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);
	[[nodiscard]] static bool ParkImagesDefault() {
		static const bool park = [] {
			const char* value = std::getenv("KYTY_PARK_IMAGES");
			return value == nullptr || value[0] != '0';
		}();
		return park;
	}
	[[nodiscard]] static uint64_t ParkEnvValue(const char* name, uint64_t fallback) {
		const char* value = std::getenv(name);
		if (value == nullptr || value[0] == '\0') {
			return fallback;
		}
		return std::strtoull(value, nullptr, 10);
	}
	[[nodiscard]] static bool        ShareDepthStencilDefault() {
		static const bool share = [] {
			const char* value = std::getenv("KYTY_DEPTH_STENCIL_SHARE");
			return value != nullptr && value[0] == '1';
		}();
		return share;
	}

	// single_view: the binding views one level and one layer, as a block view of a BC image must.
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged, bool exact_format,
	                                           bool single_view);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	[[nodiscard]] bool          RetireThroughGuestMemory(ImageId replacement, ImageId retired);
	void                        RefreshImage(ImageId id);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          CanDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                   uint64_t destination_size, uint32_t levels);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	// Caller holds m_lock and ends rendering. Begins rendering `layers` layers of one color mip
	// through a `format` view, clearing them or, when `clear` is null, loading them; returns the
	// render area.
	vk::Rect2D BeginColorRendering(Image& image, vk::Format format, uint32_t level,
	                               uint32_t layer, uint32_t layers, const vk::ClearValue* clear,
	                               vk::CommandBuffer command);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	// The depth image's stencil plane record covering exactly range, or none.
	[[nodiscard]] ImageId FindStencilPlane(const Image& depth, GuestRange range) const;
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;

	[[nodiscard]] bool DownloadImageMemory(ImageId id);
	[[nodiscard]] bool DownloadQueuedImages();

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	ResolveCache&                                     m_resolve_cache;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::mutex                                        m_write_back_mutex;
	std::vector<WriteBack>                            m_write_backs;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t m_critical_gc_memory      = 3ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick                 = 0;
	// KYTY_DEPTH_STENCIL_SHARE=1 (off by default until verified in game): a depth-only target
	// renders the depth aspect of the depth/stencil image at its address instead of replacing
	// it with a depth-only image. With it on, the runtime toggle depth_split=1 turns it off again.
	bool     m_share_depth_stencil     = ShareDepthStencilDefault();
	// Keep images that depth conversions replace for reuse (see ParkedImage). On by default
	// (world A/B pk_ab: depth conversion time -44 %, -6 % GPU-thread time per operation);
	// KYTY_PARK_IMAGES=0 turns it off. Runtime toggles, re-read at each suspend point:
	// park_off=1 turns it off, park_images=1 turns it on.
	// KYTY_PARK_MAX (images, default 256), KYTY_PARK_MB (default 256) and KYTY_PARK_FRAMES
	// (suspend points a parked image is kept, default 30) bound the pool.
	bool                     m_park_images      = ParkImagesDefault();
	uint64_t                 m_park_max_count   = ParkEnvValue("KYTY_PARK_MAX", 256);
	uint64_t                 m_park_max_bytes   = ParkEnvValue("KYTY_PARK_MB", 256) << 20u;
	uint64_t                 m_park_max_frames  = ParkEnvValue("KYTY_PARK_FRAMES", 30);
	uint64_t                 m_parked_bytes     = 0;
	uint64_t                 m_park_cap_hits    = 0;
	std::vector<ParkedImage> m_parked_images; // Oldest first.
	bool     m_readback_linear_images  = false;
	bool     m_readback_tiled_images   = false;
	bool     m_readback_storage_images = false;

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class ImageHeap;
	friend class ResolveCache;
	friend bool BcAlias::Matches(const ImageInfo&, const ImageInfo&);
	friend bool BcAlias::Publish(TextureCache&, Image&);
	friend void BcAlias::Resolve(TextureCache&, ImageId, ImageId);
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
