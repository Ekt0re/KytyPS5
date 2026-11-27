#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RESOLVECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RESOLVECACHE_H_

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/cache/resolveStamp.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace Libs::Graphics {

class RenderContext;

// PS5: checks kept FindImage results and keeps each render target's last resolve.
class ResolveCache {
public:
	explicit ResolveCache(RenderContext& context) noexcept: m_context(context) {}

	// The image set, a stencil plane record or the surface metadata changed.
	void Invalidate() noexcept { Invalidate(0, std::numeric_limits<uint64_t>::max()); }
	// The same, where only lookups of images that overlap the guest range can find another
	// result: an image was registered or unregistered there.
	void Invalidate(uint64_t address, uint64_t size) noexcept {
		m_epoch++;
		m_changes[m_epoch % m_changes.size()] = {address, size};
	}
	[[nodiscard]] uint64_t Epoch() const noexcept { return m_epoch; }
	// Whether an Invalidate after `epoch` concerns lookups in the guest range. True as well once
	// the changes since then are no longer on record.
	[[nodiscard]] bool ChangedSince(uint64_t epoch, uint64_t address, uint64_t size) const noexcept;
	// Whether every change since `epoch` is still on record.
	[[nodiscard]] bool OnRecord(uint64_t epoch) const noexcept {
		return epoch != 0 && epoch <= m_epoch && m_epoch - epoch < m_changes.size();
	}
	// Brings a stamp up to date if no Invalidate since it was made concerns lookups in the guest
	// range FindImage searched for it; true if that moved the stamp.
	bool Carry(ResolveStamp& stamp, uint64_t address, uint64_t size) const noexcept {
		if (stamp.epoch == 0 || stamp.epoch == m_epoch || ChangedSince(stamp.epoch, address, size)) {
			return false;
		}
		stamp.epoch = m_epoch;
		return true;
	}
	// The stamp of a FindImage whose first loop matched `found` among `candidates` images.
	[[nodiscard]] ResolveStamp Stamp(const ImageInfo& info, size_t candidates, ImageId found) const;
	// Whether FindImage, given the same inputs, would find `id` again; if so, accesses it as
	// FindImage does.
	[[nodiscard]] bool Reuse(ImageId id, const ResolveStamp& stamp);
	// Reuse without the draw window and tick of color clear keys: for a caller that knows the
	// lookup's color metadata holds no GPU-written blocks, so the keys cannot change its result.
	[[nodiscard]] bool ReuseImageSet(ImageId id, const ResolveStamp& stamp);

	// The last target resolved from the same arguments, while Reuse holds.
	[[nodiscard]] bool FindColor(uint32_t slot, const HW::RenderTarget& rt, bool exact_format,
	                             RenderColorInfo& r);
	void               SetColor(uint32_t slot, const HW::RenderTarget& rt, bool exact_format,
	                            const RenderColorInfo& r);
	[[nodiscard]] bool FindDepth(const HW::DepthRenderTarget& z, TextureCache::ImageDesc& desc,
	                             ImageId& id);
	void SetDepth(const HW::DepthRenderTarget& z, const TextureCache::ImageDesc& desc, ImageId id);

private:
	struct Color {
		HW::RenderTarget registers {};
		bool             exact_format = false;
		RenderColorInfo  target;
	};
	struct Depth {
		HW::DepthRenderTarget   registers {};
		TextureCache::ImageDesc desc;
		ImageId                 image_id;
	};

	struct Change {
		uint64_t address = 0;
		uint64_t size    = std::numeric_limits<uint64_t>::max();
	};

	RenderContext&                                  m_context;
	uint64_t                                        m_epoch = 1;
	// m_changes[epoch % size] is the change that started `epoch`, for the last `size` epochs.
	std::array<Change, 256>                         m_changes {};
	std::array<Color, RENDER_COLOR_ATTACHMENTS_MAX> m_colors;
	Depth                                           m_depth;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RESOLVECACHE_H_
