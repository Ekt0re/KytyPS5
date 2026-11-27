#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/cache/resolveStamp.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

// Length of the resident image heap array: IR::ImageHeapCapacity, or the largest power of two
// the device's update-after-bind limits allow. Shaders declare it and the heap binds it, so every
// device (the window's and test harnesses') gets it from here.
uint32_t ImageHeapCapacity(const GraphicContext& graphics);

class Buffer;
class CommandBuffer;
class PageManager;
class RenderContext;
class RenderExecutor;
struct ImageHeapTestAccess;

// Guest image tables that shaders index by GPU-computed offsets, kept resident in one bindless
// sampled-image array. A table is keyed by its base and follows its descriptor's size. Each view
// of a table owns device slot words, one per table dword, that select an array element; one
// element serves each host binding across views. CPU writes to a table fault in its
// snapshot, which each draw refreshes once; a view takes the changes when a draw first acquires
// it. Images behind elements report their changes through Image::heap_changes. A smaller table
// leaves at a suspend point if the frame did not acquire it, a larger one after
// LargeTableIdleFrames such frames.
class ImageHeap final: public ShaderRecompiler::IR::ImageHeapSource {
public:
	ImageHeap(RenderContext& context, PageManager& page_manager);
	~ImageHeap();
	KYTY_CLASS_NO_COPY(ImageHeap);

	// Starts a draw or dispatch, before its programs acquire views.
	void BeginDraw() { ++m_draw; }
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
	             ShaderRecompiler::IR::ImageHeapView&          result) override;
	// Image discovery: resolves new elements and rediscovers changed images.
	void Rebind(RenderExecutor& executor);
	// Recording: uploads slots, moves resident images to their read layout and binds the array.
	void Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point, vk::PipelineLayout layout);
	// CPU write fault, on any thread.
	void InvalidateMemory(uint64_t address, uint64_t size);
	void UnmapMemory(uint64_t address, uint64_t size);
	// At a suspend point.
	void RunGarbageCollector();

	// KYTY_BDA_GUARD: what the view whose slot words live at `slots` (the device address Acquire
	// returned) looks like now, to check it against the words a program is about to be given.
	struct GuardView {
		bool     found      = false;
		uint32_t kinds      = 0; // Arms the view's slot words select among.
		uint32_t slot_count = 0;
		uint32_t bad_word   = 0; // A slot word whose kind or element is out of range, if any.
		bool     bad        = false;
	};
	[[nodiscard]] GuardView GuardFind(uint64_t slots) const;
	// A zeroed slot buffer: every key selects the null image.
	[[nodiscard]] uint64_t GuardNullSlots();

private:
	using DescriptorValue = ShaderRecompiler::IR::DescriptorValue;
	using ImageResource   = ShaderRecompiler::IR::ImageResource;

	struct Key {
		DescriptorValue                          value;
		ShaderRecompiler::Decoder::ImageDimension dimension = ShaderRecompiler::Decoder::ImageDimension::Unknown;
		bool                                     r128 = false;
		bool                                     depth_view = false;

		bool operator==(const Key&) const = default;
	};

	struct KeyHash {
		size_t operator()(const Key& key) const noexcept;
	};

	// A stale element's slots move to a new element of its descriptor, a buried one's to null.
	enum class State : uint8_t { Live, Stale, Buried };

	struct Element {
		Key      key;
		ImageId  image;
		uint64_t serial = 0; // Image::serial of image.
		uint32_t kind   = 0;
		uint32_t refs   = 0; // Slots of all views that select it.
		State    state  = State::Live;
	};

	struct Memory;

	struct View {
		Memory&                      memory;
		uint64_t                     id      = 0; // Key of m_view_ids.
		uint64_t                     step    = 0;
		uint64_t                     first   = 0;
		uint64_t                     count   = 0;
		uint64_t                     limit   = 0; // Probes a key domain allows, if fewer.
		ImageResource                root;
		std::vector<uint32_t>        slots; // Plus a trailing null slot.
		std::vector<uint32_t>        kind_keys;
		std::vector<DescriptorValue> kinds; // One representative per kind_keys entry.
		std::unique_ptr<Buffer>      buffer;
		size_t                       dirty_begin   = 0; // Slots to upload.
		size_t                       dirty_end     = 0;
		size_t                       changed_begin = 0; // Table bytes changed since it was probed.
		size_t                       changed_end   = 0;
	};

	struct Memory {
		uint64_t                           base = 0;
		uint64_t                           size = 0;
		std::vector<uint32_t>              words;
		std::vector<uint8_t>               dirty_pages; // Unwatched, so unread since a write.
		std::unordered_map<uint32_t, uint64_t> dead;   // Buried words: their descriptor hashes.
		uint64_t                           draw = 0;    // Last draw that refreshed it.
		uint32_t                           idle = 0;    // Suspend points since a frame acquired it.
		std::vector<std::unique_ptr<View>> views;
	};

	[[nodiscard]] bool Refresh(Memory& memory);
	void               MarkDirty(Memory& memory, size_t begin, size_t end);
	template <bool track>
	void     Watch(const Memory& memory, size_t begin, size_t end);
	void     Unwatch(const Memory& memory);
	void     Drop(Memory& memory);
	void     UpdateProbes(View& view, uint64_t begin, uint64_t end);
	void     UpdateProbe(View& view, uint64_t probe);
	void     Assign(View& view, size_t slot, uint32_t entry);
	uint32_t Intern(View& view, uint32_t kind, const Key& key);
	void     Release(uint32_t entry);
	void     Retire();
	void     Mark(uint32_t index, State state);
	template <typename Fn>
	bool     ForEachChanged(Fn&& fn);
	void     AddUser(uint32_t index, const View& view, size_t slot);
	void     Repoint();
	void     TakeChanges();
	ImageId  CurrentImage(const Image::HeapChange& change);
	uint32_t Allocate();
	void     Resize(Memory& memory, uint64_t base, uint64_t size);
	void     WriteDescriptors();

	RenderContext&                       m_context;
	PageManager&                         m_page_manager;
	uint32_t                             m_capacity = 0; // Resident images the array binds.
	std::mutex                           m_mutex; // Fault-visible Memory state and m_memories.
	std::vector<std::unique_ptr<Memory>> m_memories;
	std::vector<View*>                   m_active; // Acquired since the last commit.
	uint64_t                             m_draw         = 1;
	uint64_t                             m_suspend_draw = 0; // m_draw at the last suspend point.
	std::vector<Element>                              m_elements;
	std::unordered_map<Key, uint32_t, KeyHash>        m_index;
	std::vector<uint32_t>                             m_free;
	std::vector<uint32_t>                             m_pending;  // Not yet resolved.
	std::vector<uint32_t>                             m_released; // Unselected after the upload.
	// An element that is null because another extent holds its address, with what that result
	// of its last lookup depends on: the guest mappings and the images at its guest range (the
	// mapping generation and ResolveCache epoch of the lookup).
	struct HeldEntry {
		uint32_t index      = 0;
		uint64_t generation = 0;
		uint64_t epoch      = 0;
		uint64_t address    = 0;
		uint64_t size       = 0;
	};
	std::vector<HeldEntry>                            m_held;
	// An element whose metadata clears apply per draw, with what discovery reads to keep its
	// last lookup: the state of that lookup's preparation in m_dcc_prepared.
	struct DccEntry {
		uint32_t     index    = 0;
		bool         prepared = false; // The fields below are set.
		uint64_t     generation       = 0;
		ResolveStamp stamp; // The last lookup's: while ResolveCache::Reuse holds, the lookup finds
		                    // the same image again and has nothing new to clear.
		uint64_t     data_address     = 0;
		uint64_t     data_size        = 0;
		uint64_t     metadata_address = 0;
		uint64_t     metadata_size    = 0;
	};
	std::vector<DccEntry>                             m_dcc;
	// The texture preparation of an m_dcc element, kept while its descriptor and the guest
	// mappings stay the same, so the per-draw retry only looks the texture up again.
	struct DccPrepared {
		Key             key;
		uint32_t        kind       = 0;
		uint64_t        generation = 0;
		PreparedTexture prepared;
	};
	std::unordered_map<uint32_t, DccPrepared>         m_dcc_prepared;
	// Guest ranges of images that lost a heap reference since held elements last retried.
	std::vector<std::pair<uint64_t, uint64_t>>        m_unpinned;
	// Per element, the (view id, slot word) pairs that selected it, so Repoint visits only the
	// slots of marked elements instead of every probe of every view. Entries go stale when their
	// slot moves on (checked when read) and are dropped with the element's last reference.
	std::vector<std::vector<std::pair<uint64_t, uint32_t>>> m_users;
	std::unordered_map<uint64_t, View*>               m_view_ids;
	uint64_t                                          m_next_view_id = 1;
	std::vector<uint32_t>                             m_marked; // Referenced since marked.
	std::vector<std::pair<uint64_t, ImageId>>         m_away; // Serial, image: not read layout.
	uint64_t                                          m_held_serial = 0; // Image::serials.
	std::deque<std::pair<uint64_t, uint32_t>>         m_retired; // Tick, element.
	std::vector<std::pair<size_t, size_t>>            m_runs;
	std::vector<uint32_t>                             m_scratch;
	std::vector<Image::HeapChange>                    m_changes;
	std::vector<std::pair<uint32_t, vk::DescriptorImageInfo>> m_writes;
	vk::DescriptorSetLayout                           m_layout = nullptr;
	vk::DescriptorPool                                m_pool   = nullptr;
	vk::DescriptorSet                                 m_set    = nullptr;
	std::unique_ptr<Buffer>                           m_guard_null; // KYTY_BDA_GUARD only.

	friend struct ImageHeapTestAccess;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_IMAGEHEAP_H_
