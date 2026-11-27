#include "graphics/host_gpu/renderer/pipeline/imageHeap.h"
#include "graphics/host_gpu/renderer/resolveCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/perfSummary.h"
#include "common/runtimeToggles.h"
#include "common/profiler.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/bdaGuard.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <iterator>
#include <xxhash.h>

namespace Libs::Graphics {
namespace {

namespace IR = ShaderRecompiler::IR;

// Smaller tables leave after one idle frame, larger ones after LargeTableIdleFrames.
constexpr uint64_t MinTableBytes        = 64ull * 1024ull;
constexpr uint32_t LargeTableIdleFrames = 32;
constexpr uint32_t ElementMask          = (1u << IR::ImageHeapKindShift) - 1u;

template <typename Pred, typename Fn>
void ForEachRun(size_t begin, size_t end, Pred&& pred, Fn&& fn) {
	while (begin < end) {
		if (!pred(begin)) {
			++begin;
			continue;
		}
		auto run = begin + 1u;
		while (run < end && pred(run)) ++run;
		fn(begin, run);
		begin = run;
	}
}

// Grows [begin, end) to cover [first, last).
void Widen(size_t& begin, size_t& end, size_t first, size_t last) {
	const bool empty = begin == end;
	begin            = empty ? first : std::min(begin, first);
	end              = empty ? last : std::max(end, last);
}

// Probes at first + k * step whose first dword lies within a table of size bytes.
uint64_t Probes(uint64_t size, uint64_t first, uint64_t step) {
	return size >= first + 4u ? (size - first - 4u) / step + 1u : 0u;
}

bool EnvFlag(const char* name, bool fallback) {
	const char* value = std::getenv(name);
	return value == nullptr ? fallback : value[0] != '0';
}

// KYTY_HEAP_RANGED=0, or the runtime toggle heap_full=1: discovery looks a texture up again
// after any change of the image set or of the guest mappings, as before, instead of only after
// one at the texture's own guest range. Called on the GPU thread; the toggle is read every 64th
// call.
bool RangedDiscovery() {
	static const bool enabled = EnvFlag("KYTY_HEAP_RANGED", true);
	static bool       full    = false;
	static uint32_t   calls   = 0;
	if (Common::RuntimeToggles::Enabled() && (calls++ & 63u) == 0u) {
		full = Common::RuntimeToggles::Get("heap_full");
	}
	return enabled && !full;
}

// KYTY_HEAP_VERIFY=1: every lookup the guest range saves is made anyway and must find what the
// element holds; mismatches are counted and the first ones logged.
bool VerifyDiscovery() {
	static const bool enabled = EnvFlag("KYTY_HEAP_VERIFY", false);
	return enabled;
}

// KYTY_HEAP_STATS=1: what discovery did, on stdout about every two seconds.
struct HeapStats {
	enum Count : uint32_t {
		Rebinds,
		Idle, // Rebinds without a lookup, a change or a new element.
		Changes,
		ChangedScans,
		DccElements,
		DccReuse,     // Kept by the resolve cache as before.
		DccCarried,   // Kept because the image set changed elsewhere.
		DccRemapped,  // Preparation kept because the mappings changed elsewhere.
		DccPrepare,   // Looked up: mappings or the descriptor changed.
		DccNoStamp,   // Looked up: the last lookup gave no reusable stamp.
		DccRange,     // Looked up: the image set changed at its range.
		DccUnknown,   // Looked up: the changes since the stamp are off the record.
		DccClearKeys, // Looked up: GPU-written metadata in another draw window or tick.
		DccStale,
		HeldRuns,
		HeldElements,
		HeldCarried,
		HeldLookups,
		Pending,
		Repoints,
		Mismatches,
		CountEnd
	};
	enum Time : uint32_t { TChanges, TDcc, THeld, TRepoint, TPending, TWrite, TimeEnd };

	static bool Enabled() {
		static const bool enabled = EnvFlag("KYTY_HEAP_STATS", false);
		return enabled;
	}
	void Add(Count count, uint64_t value = 1) { counts[count] += value; }
	// Ends the running part and starts `next`.
	void Lap(Time next) {
		if (!Enabled()) return;
		const auto now = std::chrono::steady_clock::now();
		ns[part] += static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
		start = now;
		part  = next;
	}
	void Begin() {
		if (!Enabled()) return;
		start = std::chrono::steady_clock::now();
		part  = TChanges;
	}
	void Report(uint64_t epoch) {
		if (!Enabled()) return;
		const auto secs = std::chrono::duration<double>(start - last).count();
		if (secs < 2.0) return;
		const auto per = [&](Count count) {
			return counts[Rebinds] == 0 ? 0.0
			                            : static_cast<double>(counts[count]) /
			                                  static_cast<double>(counts[Rebinds]);
		};
		const auto ms = [&](Time time) { return static_cast<double>(ns[time]) / 1e6 / secs; };
		std::printf("[heap-stats] rebinds/s=%.0f idle=%.0f epochs/s=%.0f | per rebind: changes=%.2f "
		            "scans=%.2f dcc=%.1f (reuse %.1f carried %.1f remapped %.1f; lookups: prepare %.2f nostamp "
		            "%.2f range %.2f unknown %.2f clearkeys %.2f; stale %.3f) held runs=%.2f "
		            "elements=%.1f (carried %.1f lookups %.2f) pending=%.2f repoints=%.3f | "
		            "mismatches=%" PRIu64 " | ms/s: changes=%.1f dcc=%.1f held=%.1f repoint=%.1f "
		            "pending=%.1f write=%.1f\n",
		            static_cast<double>(counts[Rebinds]) / secs,
		            static_cast<double>(counts[Idle]) / secs,
		            static_cast<double>(epoch - last_epoch) / secs, per(Changes), per(ChangedScans),
		            per(DccElements), per(DccReuse), per(DccCarried), per(DccRemapped), per(DccPrepare),
		            per(DccNoStamp), per(DccRange), per(DccUnknown), per(DccClearKeys),
		            per(DccStale), per(HeldRuns), per(HeldElements), per(HeldCarried),
		            per(HeldLookups), per(Pending), per(Repoints), mismatches + counts[Mismatches],
		            ms(TChanges), ms(TDcc), ms(THeld), ms(TRepoint), ms(TPending), ms(TWrite));
		std::fflush(stdout);
		mismatches += counts[Mismatches];
		counts     = {};
		ns         = {};
		last       = start;
		last_epoch = epoch;
	}

	std::array<uint64_t, CountEnd>        counts {};
	std::array<uint64_t, TimeEnd>         ns {};
	uint64_t                              mismatches = 0;
	uint64_t                              last_epoch = 0;
	Time                                  part       = TChanges;
	std::chrono::steady_clock::time_point start;
	std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
};

// Discovery runs on the GPU thread only.
HeapStats g_stats;

void LogMismatch(const char* list, uint32_t index, uint64_t address, uint64_t size) {
	static uint32_t logged = 0;
	if (logged < 16u) {
		++logged;
		LOGF("ImageHeap: KYTY_HEAP_VERIFY: %s element %u at %016" PRIx64 "+%" PRIx64
		     " was kept, but its lookup finds another image\n",
		     list, index, address, size);
	}
}

uint64_t Hash(const IR::DescriptorValue& value) {
	return XXH3_64bits(value.dwords.data(), sizeof(value.dwords));
}

// Tables grow a little on most frames, so their slot words get room to double.
std::unique_ptr<Buffer> SlotBuffer(RenderContext& context, uint64_t size) {
	return std::make_unique<Buffer>(
	    context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer |
	        vk::BufferUsageFlagBits::eShaderDeviceAddress,
	    std::bit_ceil(size + 4u));
}

// Guest bytes of a table that its tracker pages [begin, end) cover.
std::pair<uint64_t, uint64_t> PageRange(uint64_t base, uint64_t size, size_t begin, size_t end) {
	const auto origin = Common::AlignDown(base, TRACKER_PAGE_SIZE);
	return {std::max(base, origin + begin * TRACKER_PAGE_SIZE),
	        std::min(base + size, origin + end * TRACKER_PAGE_SIZE)};
}

} // namespace

size_t ImageHeap::KeyHash::operator()(const Key& key) const noexcept {
	const auto binding = (static_cast<uint32_t>(key.dimension) << 2u) | (uint32_t(key.r128) << 1u) | uint32_t(key.depth_view);
	return XXH3_64bits_withSeed(key.value.dwords.data(), sizeof(key.value.dwords), binding);
}

uint32_t ImageHeapCapacity(const GraphicContext& graphics) {
	if (graphics.image_heap_capacity == 0u && graphics.physical_device) {
		vk::PhysicalDeviceVulkan12Properties limits {};
		vk::PhysicalDeviceProperties2        properties {};
		properties.pNext = &limits;
		graphics.physical_device.getProperties2(&properties);
		const auto limit = std::min({limits.maxDescriptorSetUpdateAfterBindSampledImages,
		                             limits.maxPerStageDescriptorUpdateAfterBindSampledImages,
		                             limits.maxPerStageUpdateAfterBindResources});
		graphics.image_heap_capacity =
		    limit == 0u ? 0u : std::min(IR::ImageHeapCapacity, std::bit_floor(limit));
	}
	return graphics.image_heap_capacity;
}

ImageHeap::ImageHeap(RenderContext& context, PageManager& page_manager)
    : m_context(context), m_page_manager(page_manager) {
	auto& graphics = context.GetGraphics();
	// Shaders declare the same array length (SpirvProfile).
	m_capacity = ImageHeapCapacity(graphics);
	EXIT_NOT_IMPLEMENTED(m_capacity == 0u);

	const vk::DescriptorBindingFlags binding_flags =
	    vk::DescriptorBindingFlagBits::ePartiallyBound |
	    vk::DescriptorBindingFlagBits::eUpdateAfterBind |
	    vk::DescriptorBindingFlagBits::eUpdateUnusedWhilePending;
	vk::DescriptorSetLayoutBindingFlagsCreateInfo flags_info {};
	flags_info.bindingCount  = 1;
	flags_info.pBindingFlags = &binding_flags;
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eSampledImage,
	                                              m_capacity,
	                                              vk::ShaderStageFlagBits::eAll, nullptr};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.pNext        = &flags_info;
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool;
	layout_info.bindingCount = 1;
	layout_info.pBindings    = &binding;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_layout) !=
	        vk::Result::eSuccess);
	const vk::DescriptorPoolSize pool_size {vk::DescriptorType::eSampledImage,
	                                        m_capacity};
	vk::DescriptorPoolCreateInfo pool_info {};
	pool_info.flags         = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind;
	pool_info.maxSets       = 1;
	pool_info.poolSizeCount = 1;
	pool_info.pPoolSizes    = &pool_size;
	EXIT_IF(graphics.device.createDescriptorPool(&pool_info, nullptr, &m_pool) !=
	        vk::Result::eSuccess);
	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool     = m_pool;
	allocate.descriptorSetCount = 1;
	allocate.pSetLayouts        = &m_layout;
	EXIT_IF(graphics.device.allocateDescriptorSets(&allocate, &m_set) != vk::Result::eSuccess);
	graphics.image_heap_layout = m_layout;
	// Slot word 0 selects element 0, which stays a null image.
	m_elements.emplace_back();
	m_writes.push_back({0u, {nullptr, nullptr, vk::ImageLayout::eShaderReadOnlyOptimal}});
	WriteDescriptors();
}

ImageHeap::~ImageHeap() {
	for (const auto& memory: m_memories) Unwatch(*memory);
	auto& graphics             = m_context.GetGraphics();
	graphics.image_heap_layout = nullptr;
	graphics.device.destroyDescriptorPool(m_pool, nullptr);
	graphics.device.destroyDescriptorSetLayout(m_layout, nullptr);
}

bool ImageHeap::Acquire(const IR::ImageHeapRequest& request, IR::ImageHeapView& result) {
	KYTY_PROFILER_FUNCTION();
	const auto size = request.size & ~uint64_t {3};
	if (request.count == 0u || request.step == 0u ||
	    ((request.base | request.first | request.step) & 3u) != 0u) {
		return false;
	}
	auto memory = std::ranges::find_if(
	    m_memories, [&](const auto& candidate) { return candidate->base == request.base; });
	// A table follows its descriptor's size, as the first program of a draw acquires it.
	if (memory == m_memories.end() || ((*memory)->size != size && (*memory)->draw != m_draw)) {
		if (!m_context.IsMapped(request.base, size)) return false;
		if (memory == m_memories.end()) {
			// The guest republishes tables at new bases; a copy that idled a frame moves there,
			// unless it buried words, which a new table resolves.
			memory = std::ranges::find_if(m_memories, [&](const auto& candidate) {
				return candidate->size == size && candidate->idle != 0u &&
				       candidate->draw <= m_suspend_draw && candidate->dead.empty();
			});
		}
		if (memory == m_memories.end()) {
			std::scoped_lock lock {m_mutex};
			memory = m_memories.insert(m_memories.end(), std::make_unique<Memory>());
		}
		Resize(**memory, request.base, size);
	}
	// A draw refreshes a table and a view takes its changes once, so all programs of the draw see
	// one snapshot and one set of kinds.
	if ((*memory)->draw != m_draw) {
		if (!Refresh(**memory)) return false;
		(*memory)->draw = m_draw;
	}
	const auto root   = IR::ImageHeapResource(*request.root, 0u);
	auto&      views  = (*memory)->views;
	const auto bytes  = (*memory)->size;
	const auto probes = Probes(size, request.first, request.step);
	const auto limit  = request.count < probes ? request.count : UINT64_MAX;
	// Slots are indexed by table dword, so unbounded requests share the view of their lattice.
	const auto first = limit == UINT64_MAX ? request.first % request.step : request.first;
	auto       view  = std::ranges::find_if(views, [&](const auto& candidate) {
		return candidate->step == request.step && candidate->first == first &&
		       candidate->limit == limit && candidate->root == root;
	});
	if (view == views.end()) {
		const auto count   = std::min(limit, Probes(bytes, first, request.step));
		auto       created = std::make_unique<View>(View {.memory  = **memory,
		                                                  .id      = m_next_view_id++,
		                                                  .step    = request.step,
		                                                  .first   = first,
		                                                  .count   = count,
		                                                  .limit   = limit,
		                                                  .root    = root});
		created->slots.resize(bytes / 4u + 1u);
		created->buffer = SlotBuffer(m_context, bytes);
		m_view_ids.emplace(created->id, created.get());
		UpdateProbes(*created, 0u, bytes);
		// The first upload also clears the slots no probe selects.
		created->dirty_begin = 0;
		created->dirty_end   = created->slots.size();
		view = views.insert(views.end(), std::move(created));
	} else {
		// Empty after the first acquire in a draw, since the table refreshes once per draw.
		UpdateProbes(**view, (*view)->changed_begin, (*view)->changed_end);
		(*view)->changed_begin = (*view)->changed_end = 0;
	}
	if (std::ranges::find(m_active, view->get()) == m_active.end()) m_active.push_back(view->get());
	const auto& selected = **view;
	result = {.slots      = selected.buffer->BufferDeviceAddress(),
	          .slot_count = static_cast<uint32_t>(selected.slots.size() - 1u),
	          .kinds      = selected.kinds};
	return true;
}

bool ImageHeap::Refresh(Memory& memory) {
	// GPU writes leave no fault; reading the table downloads them.
	const bool gpu_dirty = m_context.GetBufferCache().HasGpuDirtyBytes(memory.base, memory.size);
	const auto pages     = memory.dirty_pages.size();
	m_runs.clear();
	{
		std::scoped_lock lock {m_mutex};
		// Watch before reading, so a write during the read faults the page again.
		ForEachRun(
		    0, pages, [&](size_t page) { return memory.dirty_pages[page] != 0u; },
		    [&](size_t begin, size_t end) {
			    std::fill(memory.dirty_pages.begin() + begin, memory.dirty_pages.begin() + end,
			              uint8_t {0});
			    Watch<true>(memory, begin, end);
			    m_runs.emplace_back(begin, end);
		    });
	}
	if (gpu_dirty) m_runs.assign(1u, {0u, pages});
	for (size_t run = 0; run < m_runs.size(); ++run) {
		const auto [address, end] = PageRange(memory.base, memory.size, m_runs[run].first,
		                                      m_runs[run].second);
		m_scratch.resize((end - address) / 4u);
		if (!LibKernel::Memory::TryReadBufferBacking(address, m_scratch.data(), end - address)) {
			std::scoped_lock lock {m_mutex};
			for (; run < m_runs.size(); ++run) {
				MarkDirty(memory, m_runs[run].first, m_runs[run].second);
			}
			return false;
		}
		const auto words = std::span(memory.words).subspan((address - memory.base) / 4u,
		                                                   m_scratch.size());
		size_t     first = 0;
		size_t     last  = words.size();
		while (first < last && words[first] == m_scratch[first]) ++first;
		while (first < last && words[last - 1u] == m_scratch[last - 1u]) --last;
		if (first == last) continue;
		std::copy(m_scratch.begin() + first, m_scratch.begin() + last, words.begin() + first);
		const auto offset = address - memory.base;
		for (auto& view: memory.views) {
			Widen(view->changed_begin, view->changed_end, offset + first * 4u, offset + last * 4u);
		}
	}
	return true;
}

void ImageHeap::MarkDirty(Memory& memory, size_t begin, size_t end) {
	ForEachRun(
	    begin, end, [&](size_t page) { return memory.dirty_pages[page] == 0u; },
	    [&](size_t first, size_t last) {
		    std::fill(memory.dirty_pages.begin() + first, memory.dirty_pages.begin() + last,
		              uint8_t {1});
		    Watch<false>(memory, first, last);
	    });
}

template <bool track>
void ImageHeap::Watch(const Memory& memory, size_t begin, size_t end) {
	const auto [address, limit] = PageRange(memory.base, memory.size, begin, end);
	m_page_manager.UpdatePageWatchers<track>(address, limit - address);
}

void ImageHeap::Unwatch(const Memory& memory) {
	ForEachRun(
	    0, memory.dirty_pages.size(), [&](size_t page) { return memory.dirty_pages[page] == 0u; },
	    [&](size_t begin, size_t end) { Watch<false>(memory, begin, end); });
}

void ImageHeap::InvalidateMemory(uint64_t address, uint64_t size) {
	std::scoped_lock lock {m_mutex};
	for (auto& memory: m_memories) {
		// Watches cover whole pages, so a write next to the table must unwatch its page too.
		const auto origin = Common::AlignDown(memory->base, TRACKER_PAGE_SIZE);
		const auto begin  = std::max(address, origin);
		const auto end    = std::min(address + size,
		                             origin + memory->dirty_pages.size() * TRACKER_PAGE_SIZE);
		if (begin >= end) continue;
		MarkDirty(*memory, (begin - origin) / TRACKER_PAGE_SIZE,
		          (end - 1u - origin) / TRACKER_PAGE_SIZE + 1u);
	}
}

void ImageHeap::UpdateProbes(View& view, uint64_t begin, uint64_t end) {
	// A probe reads its descriptor from first + probe * step.
	const uint64_t bytes = view.root.r128 ? 16u : 32u;
	auto probe = begin < view.first + bytes ? 0u : (begin - view.first - bytes) / view.step + 1u;
	for (; probe < view.count && view.first + probe * view.step < end; ++probe) {
		UpdateProbe(view, probe);
	}
}

void ImageHeap::UpdateProbe(View& view, uint64_t probe) {
	const auto& words   = view.memory.words;
	const auto  offset  = static_cast<uint32_t>((view.first + probe * view.step) / 4u);
	Key         key {.value = {.dword_count = 8u}, .dimension = view.root.dimension, .r128 = view.root.r128};
	std::copy_n(words.begin() + offset,
	            std::min<size_t>(key.r128 ? 4u : 8u, words.size() - offset),
	            key.value.dwords.begin());
	// A buried word stays null in every view until its descriptor changes.
	auto& dead = view.memory.dead;
	if (const auto found = dead.find(offset); found != dead.end()) {
		if (found->second == Hash(key.value)) {
			Assign(view, offset, 0u);
			return;
		}
		dead.erase(found);
	}
	const auto kind = IR::ImageHeapKind(view.root, key.value);
	if (kind == 0u) {
		Assign(view, offset, 0u);
		return;
	}
	const auto format = DecodeNativeDescriptor<ShaderTextureResource>(key.value).Format();
	key.depth_view = SampledDepthCompareFormat(format, view.root.depth_compare).has_value();
	Assign(view, offset, Intern(view, kind, key));
}

void ImageHeap::Assign(View& view, size_t slot, uint32_t entry) {
	const auto previous = std::exchange(view.slots[slot], entry);
	if (previous != entry) AddUser(entry & ElementMask, view, slot);
	Release(previous);
	if (previous != entry) Widen(view.dirty_begin, view.dirty_end, slot, slot + 1u);
}

void ImageHeap::AddUser(uint32_t index, const View& view, size_t slot) {
	if (index == 0u) return;
	if (m_users.size() <= index) m_users.resize(m_elements.size());
	auto& users = m_users[index];
	// Entries of slots that moved on pile up while the element lives; keep them near its refs.
	if (users.size() >= 64u && users.size() >= 2u * m_elements[index].refs) {
		std::erase_if(users, [&](const std::pair<uint64_t, uint32_t>& user) {
			const auto found = m_view_ids.find(user.first);
			return found == m_view_ids.end() ||
			       (found->second->slots[user.second] & ElementMask) != index;
		});
	}
	users.emplace_back(view.id, static_cast<uint32_t>(slot));
}

uint32_t ImageHeap::Intern(View& view, uint32_t kind, const Key& key) {
	const auto arm = static_cast<uint32_t>(std::ranges::find(view.kind_keys, kind) -
	                                       view.kind_keys.begin());
	if (arm == view.kind_keys.size()) {
		view.kind_keys.push_back(kind);
		view.kinds.push_back(key.value);
	}
	auto [found, inserted] = m_index.try_emplace(key, 0u);
	if (inserted) {
		found->second = Allocate();
		auto& element = m_elements[found->second];
		element.key   = key;
		element.kind  = kind;
		m_pending.push_back(found->second);
	}
	++m_elements[found->second].refs;
	return arm << IR::ImageHeapKindShift | found->second;
}

void ImageHeap::Release(uint32_t entry) {
	const auto index = entry & ElementMask;
	if (index == 0u || --m_elements[index].refs != 0u) return;
	auto& element = m_elements[index];
	if (element.image) {
		if (const auto found = m_by_serial.find(element.serial); found != m_by_serial.end()) {
			std::erase(found->second, index);
			if (found->second.empty()) m_by_serial.erase(found);
		}
	}
	Mark(index, State::Stale);
	if (index < m_users.size()) m_users[index].clear();
	auto& textures = m_context.GetTextureCache();
	if (auto* image = textures.m_slot_images.try_get(element.image);
	    image != nullptr && image->serial == element.serial) {
		image->heap_refs.fetch_sub(1u, std::memory_order_relaxed);
		image->tick_accessed_last = textures.m_gc_tick;
		// A held element at this address may resolve now (Rebind).
		if (!m_held.empty()) {
			m_unpinned.emplace_back(image->info.data.address, image->info.data.size);
		}
	}
	m_released.push_back(index);
}

// A released element leaves once the tick of the upload that unselects it ends, so no list may
// keep its index.
void ImageHeap::Retire() {
	if (m_released.empty()) return;
	const auto released = [&](uint32_t index) { return m_elements[index].refs == 0u; };
	std::erase_if(m_held, [&](const HeldEntry& entry) { return released(entry.index); });
	std::erase_if(m_dcc, [&](const DccEntry& entry) { return released(entry.index); });
	const auto tick = m_context.GetCommandScheduler().CurrentTick();
	for (const auto index: m_released) m_retired.emplace_back(tick, index);
	m_released.clear();
}

// Unlists the element, so its descriptor interns a new one.
void ImageHeap::Mark(uint32_t index, State state) {
	auto& element = m_elements[index];
	element.state = state;
	if (element.refs != 0u) m_marked.push_back(index);
	if (const auto found = m_index.find(element.key);
	    found != m_index.end() && found->second == index) {
		m_index.erase(found);
	}
}

// Calls fn with each selected element of an image in m_changes, sorted by serial; true if any.
template <typename Fn>
bool ImageHeap::ForEachChanged(Fn&& fn) {
	m_changed.clear();
	for (const auto& change: m_changes) {
		const auto found = m_by_serial.find(change.serial);
		if (found == m_by_serial.end()) continue;
		for (const auto index: found->second) {
			const auto& element = m_elements[index];
			if (element.refs != 0u && element.image && element.serial == change.serial) {
				m_changed.push_back(index);
			}
		}
	}
	std::ranges::sort(m_changed);
	m_changed.erase(std::ranges::unique(m_changed).begin(), m_changed.end());
	for (const auto index: m_changed) {
		fn(index);
	}
	return !m_changed.empty();
}

// Moves the slots of every view off marked elements; pending draws keep reading those. Only the
// slots that selected a marked element are visited, in the order of a scan of every probe of
// every view (memories, views, then slots), so elements are interned in the same order.
void ImageHeap::Repoint() {
	if (m_marked.empty()) return;
	std::ranges::sort(m_marked);
	m_marked.erase(std::ranges::unique(m_marked).begin(), m_marked.end());
	std::unordered_map<uint64_t, std::pair<uint32_t, uint32_t>> order; // Memory, view.
	for (uint32_t memory = 0; memory < m_memories.size(); ++memory) {
		const auto& views = m_memories[memory]->views;
		for (uint32_t view = 0; view < views.size(); ++view) {
			order.emplace(views[view]->id, std::pair {memory, view});
		}
	}
	struct Visit {
		uint32_t memory;
		uint32_t view;
		uint32_t slot;
		auto     operator<=>(const Visit&) const = default;
	};
	std::vector<Visit> visits;
	for (const auto index: m_marked) {
		if (index >= m_users.size() || m_elements[index].refs == 0u ||
		    m_elements[index].state == State::Live) {
			continue;
		}
		for (const auto& [id, slot]: m_users[index]) {
			const auto found = order.find(id);
			if (found == order.end()) continue;
			const auto& view = *m_memories[found->second.first]->views[found->second.second];
			if ((view.slots[slot] & ElementMask) != index || slot * 4u < view.first ||
			    (slot * 4u - view.first) % view.step != 0u ||
			    (slot * 4u - view.first) / view.step >= view.count) {
				continue;
			}
			visits.push_back({found->second.first, found->second.second, slot});
		}
	}
	m_marked.clear();
	std::ranges::sort(visits);
	visits.erase(std::ranges::unique(visits).begin(), visits.end());
	for (const auto& visit: visits) {
		auto&       view    = *m_memories[visit.memory]->views[visit.view];
		const auto  slot    = visit.slot;
		const auto& element = m_elements[view.slots[slot] & ElementMask];
		if (element.state == State::Buried) {
			view.memory.dead[slot] = Hash(element.key.value);
			Assign(view, slot, 0u);
		} else if (element.state == State::Stale) {
			const auto key = element.key; // Intern may grow m_elements.
			Assign(view, slot, Intern(view, element.kind, key));
		}
	}
}

void ImageHeap::TakeChanges() {
	m_changes.clear();
	{
		std::scoped_lock lock {Image::heap_mutex};
		std::swap(m_changes, Image::heap_changes);
	}
	std::ranges::sort(m_changes, {}, &Image::HeapChange::serial);
	const auto [first, last] = std::ranges::unique(m_changes, {}, &Image::HeapChange::serial);
	m_changes.erase(first, last);
}

// The registered, unassociated image a change came from, if it still is one.
ImageId ImageHeap::CurrentImage(const Image::HeapChange& change) {
	auto&            textures = m_context.GetTextureCache();
	std::scoped_lock lock {textures.m_lock};
	ImageId          current;
	textures.ForEachImageWithAddress(change.data.address, [&](ImageId id, const Image& image) {
		if (image.serial == change.serial && !image.depth_id) current = id;
	});
	return current;
}

uint32_t ImageHeap::Allocate() {
	auto& scheduler = m_context.GetCommandScheduler();
	while (!m_retired.empty() && scheduler.IsFree(m_retired.front().first)) {
		m_free.push_back(m_retired.front().second);
		m_retired.pop_front();
	}
	if (m_free.empty()) {
		if (m_elements.size() == m_capacity) {
			EXIT("image heaps exceed %u resident images\n", m_capacity);
		}
		m_elements.emplace_back();
		return static_cast<uint32_t>(m_elements.size() - 1u);
	}
	const auto index = m_free.back();
	m_free.pop_back();
	m_elements[index] = {};
	if (index < m_users.size()) m_users[index].clear();
	return index;
}

void ImageHeap::Resize(Memory& memory, uint64_t base, uint64_t size) {
	{
		std::scoped_lock lock {m_mutex};
		Unwatch(memory);
		memory.base = base;
		memory.size = size;
		// The next refresh rereads the whole table.
		memory.dirty_pages.assign((Common::AlignUp(base + size, TRACKER_PAGE_SIZE) -
		                           Common::AlignDown(base, TRACKER_PAGE_SIZE)) /
		                              TRACKER_PAGE_SIZE,
		                          uint8_t {1});
	}
	memory.words.resize(size / 4u);
	std::erase_if(memory.dead, [&](const auto& word) { return word.first >= size / 4u; });
	auto& scheduler = m_context.GetCommandScheduler();
	for (auto& view: memory.views) {
		const auto count = std::min(view->limit, Probes(size, view->first, view->step));
		for (auto probe = count; probe < view->count; ++probe) {
			Assign(*view, (view->first + probe * view->step) / 4u, 0u);
		}
		view->count       = count;
		const auto slots  = view->slots.size();
		view->slots.resize(size / 4u + 1u);
		view->dirty_end   = std::min(view->dirty_end, view->slots.size());
		view->dirty_begin = std::min(view->dirty_begin, view->dirty_end);
		if (view->slots.size() * sizeof(uint32_t) > view->buffer->Size()) {
			if (scheduler.Active()) {
				scheduler.DeferRelease([buffer = std::move(view->buffer)] {});
			}
			view->buffer = SlotBuffer(m_context, size);
			Widen(view->dirty_begin, view->dirty_end, 0u, view->slots.size());
		} else if (slots < view->slots.size()) {
			// Device words past the old end may hold stale slots.
			Widen(view->dirty_begin, view->dirty_end, slots, view->slots.size());
		}
	}
}

void ImageHeap::Rebind(RenderExecutor& executor) {
	KYTY_PROFILER_FUNCTION();
	namespace Perf = Common::PerfSummary;
	Perf::Add(Perf::Counter::HeapRebinds);
	auto&      textures = m_context.GetTextureCache();
	const bool ranged   = RangedDiscovery();
	const bool verify   = VerifyDiscovery();
	uint64_t   lookups  = 0;
	uint64_t   carried  = 0;
	g_stats.Begin();
	g_stats.Add(HeapStats::Rebinds);
	// A changed image refreshes; one that changed identity needs new elements.
	TakeChanges();
	g_stats.Add(HeapStats::Changes, m_changes.size());
	const bool had_work = !m_changes.empty() || !m_pending.empty();
	std::erase_if(m_changes, [&](const Image::HeapChange& change) {
		const auto id = CurrentImage(change);
		if (id) {
			textures.UpdateImage(id);
			const auto& image = textures.m_slot_images[id];
			if (const auto stencil = textures.FindStencilPlane(image, image.info.stencil)) {
				textures.UpdateImage(stencil);
			}
			m_away.emplace_back(change.serial, id);
		}
		return static_cast<bool>(id);
	});
	if (!m_changes.empty()) g_stats.Add(HeapStats::ChangedScans);
	bool stale = ForEachChanged([&](uint32_t index) { Mark(index, State::Stale); });
	// Elements hold their image's read layout, which a unified layouts switch changes.
	if (const auto epoch = UnifiedLayouts::Epoch(); epoch != m_layout_epoch) {
		m_layout_epoch = epoch;
		for (uint32_t index = 1; index < m_elements.size(); ++index) {
			const auto& element = m_elements[index];
			if (element.refs != 0u && element.image && element.state == State::Live) {
				Mark(index, State::Stale);
				stale = true;
			}
		}
	}
	g_stats.Lap(HeapStats::TDcc);
	const auto resolve = [&](const Element& element) {
		const auto& key = element.key;
		const auto resource = IR::ImageHeapResource(
		    {.dimension = key.dimension, .depth_compare = key.depth_view, .r128 = key.r128}, element.kind);
		++lookups;
		return executor.ResolveTexture(resource, element.key.value);
	};
	// A listed element whose descriptor resolves to another image is stale; true once it leaves
	// the list. A held element whose texture is unmapped stays null.
	const auto retry = [&](uint32_t index) {
		const auto& element = m_elements[index];
		if (element.state != State::Live) return true;
		const auto binding = resolve(element);
		if (binding.image_id == element.image &&
		    (!element.image || textures.m_slot_images[element.image].serial == element.serial)) {
			return !element.image && binding.desc.info.data.Empty();
		}
		Mark(index, State::Stale);
		stale = true;
		return true;
	};
	// Discovery applies the color metadata clears written since the last draw. The preparation
	// of an unchanged descriptor is kept, and the texture lookup (and so the clear) is skipped
	// while the resolve cache vouches for its last result: same image set, and for color metadata
	// the same draw window and scheduler tick, as for the textures draws bind directly.
	auto&      resolve_cache = m_context.GetResolveCache();
	auto&      buffers       = m_context.GetBufferCache();
	// The image set changes with every image created or freed, which while textures stream in
	// is every few draws; a lookup searches only its texture's guest range, so a stamp is carried
	// over changes at other ranges (ResolveCache::Carry).
	const auto same_image = [&](const Element& element, const TextureBinding& binding) {
		return binding.image_id == element.image &&
		       (!element.image || textures.m_slot_images[element.image].serial == element.serial);
	};
	// The list entry holds what the decision reads, so an element that is kept costs no look at
	// its preparation: most of the several hundred elements are kept by most draws.
	const auto retry_dcc = [&](DccEntry& entry) {
		const auto  index   = entry.index;
		const auto& element = m_elements[index];
		if (element.state != State::Live) {
			m_dcc_prepared.erase(index);
			return retry(index);
		}
		g_stats.Add(HeapStats::DccElements);
		const auto generation = m_context.MappingGeneration();
		bool       moved      = false;
		bool       counted    = false;
		// A preparation reads which guest ranges are mapped only at its texture's data.
		const bool remapped = ranged && entry.prepared && entry.generation != generation &&
		                      entry.data_size != 0u &&
		                      !m_context.MappingChangedSince(entry.generation, entry.data_address,
		                                                     entry.data_size);
		if (remapped) {
			entry.generation = generation;
			g_stats.Add(HeapStats::DccRemapped);
		}
		if (entry.prepared && entry.generation == generation && element.image) {
			if (HeapStats::Enabled() && entry.stamp.epoch != resolve_cache.Epoch()) {
				counted = true;
				g_stats.Add(entry.stamp.epoch == 0u
				                ? HeapStats::DccNoStamp
				                : !resolve_cache.ChangedSince(entry.stamp.epoch, entry.data_address,
				                                              entry.data_size)
				                      ? HeapStats::DccCarried
				                      : resolve_cache.OnRecord(entry.stamp.epoch)
				                            ? HeapStats::DccRange
				                            : HeapStats::DccUnknown);
			}
			moved = ranged &&
			        resolve_cache.Carry(entry.stamp, entry.data_address, entry.data_size);
			// A lookup applies color metadata clears only from blocks the GPU wrote
			// (MaterializeColorClear); with none in the metadata range, the clear keys cannot
			// change what it does.
			if (resolve_cache.Reuse(element.image, entry.stamp) ||
			    (!(entry.metadata_size != 0u &&
			       buffers.IsRegionGpuModified(entry.metadata_address, entry.metadata_size)) &&
			     resolve_cache.ReuseImageSet(element.image, entry.stamp))) {
				if (moved || remapped) {
					++carried;
				} else {
					g_stats.Add(HeapStats::DccReuse);
				}
				if (!(moved || remapped) || !verify) return false;
			} else if (!counted) {
				counted = true;
				g_stats.Add(HeapStats::DccClearKeys);
			}
		}
		const auto& key      = element.key;
		const auto  resource = IR::ImageHeapResource(
		    {.dimension = key.dimension, .depth_compare = key.depth_view, .r128 = key.r128}, element.kind);
		auto& cached = m_dcc_prepared[index];
		if (remapped) {
			if (verify) {
				auto fresh = executor.PrepareTexture(resource, element.key.value);
				if (fresh.resolved != cached.prepared.resolved ||
				    !(fresh.desc.info.data == cached.prepared.desc.info.data)) {
					g_stats.Add(HeapStats::Mismatches);
					LogMismatch("remapped", index, entry.data_address, entry.data_size);
				}
				cached.prepared = std::move(fresh);
			}
			cached.generation = generation;
		}
		if (cached.generation != generation || cached.kind != element.kind || !(cached.key == element.key) ||
		    cached.generation == 0u) {
			cached = {element.key, element.kind, generation,
			          executor.PrepareTexture(resource, element.key.value)};
			g_stats.Add(HeapStats::DccPrepare);
		} else if (!counted && !moved && !remapped) {
			g_stats.Add(HeapStats::DccNoStamp);
		}
		++lookups;
		const auto binding     = executor.FinishTexture(resource, cached.prepared);
		entry.prepared         = true;
		entry.generation       = generation;
		entry.stamp            = binding.desc.stamp;
		entry.data_address     = cached.prepared.desc.info.data.address;
		entry.data_size        = cached.prepared.desc.info.data.size;
		entry.metadata_address = cached.prepared.desc.info.metadata.range.address;
		entry.metadata_size    = cached.prepared.desc.info.metadata.range.size;
		if (same_image(element, binding)) {
			const bool leaves = !element.image && binding.desc.info.data.Empty();
			if (leaves) m_dcc_prepared.erase(index);
			return leaves;
		}
		if (moved || remapped) {
			g_stats.Add(HeapStats::Mismatches);
			LogMismatch("metadata", index, entry.data_address, entry.data_size);
		}
		g_stats.Add(HeapStats::DccStale);
		Mark(index, State::Stale);
		stale = true;
		m_dcc_prepared.erase(index);
		return true;
	};
	{
		std::erase_if(m_dcc, retry_dcc);
	}
	g_stats.Lap(HeapStats::THeld);
	// Held elements retry once an image is created, such as a render target of their extent:
	// those whose guest range saw an image registered or unregistered, or lost a heap reference
	// (the extent that held the address may have left the heap), since they last retried.
	// They retry at the same draws as before (after an image was created), so discovery makes
	// no lookup the full one would not make: a held element's lookup is not free of effects (it
	// can replace an image of another extent at its address), and extra ones change what later
	// draws find.
	const auto held_serial = Image::serials.load(std::memory_order_relaxed);
	const auto retry_held = [&](HeldEntry& entry) {
		const auto  index   = entry.index;
		const auto& element = m_elements[index];
		if (element.state != State::Live) return true;
		g_stats.Add(HeapStats::HeldElements);
		const auto generation = m_context.MappingGeneration();
		const bool kept =
		    ranged && entry.size != 0u &&
		    (entry.generation == generation ||
		     !m_context.MappingChangedSince(entry.generation, entry.address, entry.size)) &&
		    !resolve_cache.ChangedSince(entry.epoch, entry.address, entry.size) &&
		    std::ranges::none_of(m_unpinned, [&](const std::pair<uint64_t, uint64_t>& unpinned) {
			    return unpinned.first < entry.address + entry.size &&
			           entry.address < unpinned.first + unpinned.second;
		    });
		if (kept) {
			++carried;
			g_stats.Add(HeapStats::HeldCarried);
			if (!verify) {
				entry.generation = generation;
				entry.epoch      = resolve_cache.Epoch();
				return false;
			}
		}
		g_stats.Add(HeapStats::HeldLookups);
		const auto binding = resolve(element);
		if (same_image(element, binding)) {
			const bool  leaves = !element.image && binding.desc.info.data.Empty();
			const auto& data   = binding.desc.info.data;
			if (kept && leaves) {
				g_stats.Add(HeapStats::Mismatches);
				LogMismatch("held", index, entry.address, entry.size);
			}
			entry = {index, generation, resolve_cache.Epoch(), data.address, data.size};
			return leaves;
		}
		if (kept) {
			g_stats.Add(HeapStats::Mismatches);
			LogMismatch("held", index, entry.address, entry.size);
		}
		Mark(index, State::Stale);
		stale = true;
		return true;
	};
	if (m_held_serial != held_serial) {
		m_held_serial = held_serial;
		g_stats.Add(HeapStats::HeldRuns);
		std::erase_if(m_held, retry_held);
		m_unpinned.clear();
	}
	if (m_held.empty()) m_unpinned.clear();
	g_stats.Lap(HeapStats::TRepoint);
	if (stale) g_stats.Add(HeapStats::Repoints);
	if (stale) Repoint();
	g_stats.Lap(HeapStats::TPending);
	for (const auto index: m_pending) {
		if (m_elements[index].refs == 0u) continue;
		g_stats.Add(HeapStats::Pending);
		auto binding = resolve(m_elements[index]);
		if (binding.image_id) {
			auto& image   = textures.GetImage(binding.image_id);
			auto& element = m_elements[index];
			// Pinned before the refresh, so a CPU write after it reports a change.
			image.heap_refs.fetch_add(1u, std::memory_order_relaxed);
			// A stale element resolved again leaves its old image's list, and an element already
			// listed under this serial is not added twice (repoint storms re-resolve the same
			// elements over and over).
			if (element.image && element.serial != image.serial) {
				if (const auto old = m_by_serial.find(element.serial); old != m_by_serial.end()) {
					std::erase(old->second, index);
					if (old->second.empty()) m_by_serial.erase(old);
				}
			}
			element.image      = binding.image_id;
			element.serial     = image.serial;
			if (auto& list = m_by_serial[image.serial];
			    std::ranges::find(list, index) == list.end()) {
				list.push_back(index);
			}
			binding.layout     = image.ReadLayout();
			binding.image_view = textures.FindTexture(binding.image_id, binding.desc);
			if (binding.desc.info.metadata.kind == ImageMetadataKind::Dcc) m_dcc.push_back({index});
			m_away.emplace_back(image.serial, binding.image_id);
		} else if (!binding.desc.info.data.Empty()) {
			const auto& data = binding.desc.info.data;
			m_held.push_back({index, m_context.MappingGeneration(), resolve_cache.Epoch(),
			                  data.address, data.size});
		}
		m_writes.push_back({index, {nullptr, binding.image_view, binding.layout}});
	}
	m_pending.clear();
	g_stats.Lap(HeapStats::TWrite);
	WriteDescriptors();
	Perf::Add(Perf::Counter::HeapLookups, lookups);
	Perf::Add(Perf::Counter::HeapRangeSkips, carried);
	if (lookups == 0u && !had_work && !stale) g_stats.Add(HeapStats::Idle);
	g_stats.Lap(HeapStats::TChanges);
	g_stats.Report(resolve_cache.Epoch());
}

void ImageHeap::WriteDescriptors() {
	if (m_writes.empty()) return;
	Common::PerfSummary::Add(Common::PerfSummary::Counter::HeapDescriptorWrites, m_writes.size());
	std::vector<vk::WriteDescriptorSet> writes(m_writes.size());
	for (size_t write = 0; write < m_writes.size(); ++write) {
		writes[write].dstSet          = m_set;
		writes[write].dstArrayElement = m_writes[write].first;
		writes[write].descriptorCount = 1;
		writes[write].descriptorType  = vk::DescriptorType::eSampledImage;
		writes[write].pImageInfo      = &m_writes[write].second;
	}
	m_context.GetGraphics().device.updateDescriptorSets(static_cast<uint32_t>(writes.size()),
	                                                    writes.data(), 0, nullptr);
	m_writes.clear();
}

void ImageHeap::Commit(CommandBuffer& buffer, vk::PipelineBindPoint bind_point,
                       vk::PipelineLayout layout) {
	KYTY_PROFILER_FUNCTION();
	for (auto* view: m_active) {
		if (view->dirty_begin == view->dirty_end) continue;
		const auto slots  = std::span(view->slots).subspan(view->dirty_begin,
		                                                   view->dirty_end - view->dirty_begin);
		// The draw is being recorded: its targets, vertex buffers and bindings are in the current
		// command buffer or about to follow, so this upload must not wait for the stream ring. A
		// wait for ring space that the current tick wrote submits the command buffer and starts
		// another, which would get this draw without the state recorded so far, and with the
		// rest of its bindings recorded into the submitted one. Slots that do not fit the ring
		// now (several new views of a large table fill it within one tick) go through a buffer
		// of their own.
		auto&         stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		const Buffer* source = &stream;
		if (BdaGuard::Mode() != 0 && slots.size_bytes() >= (1u << 20u)) {
			BdaGuard::Note(fmt::format("heap slot upload of {} bytes at tick {}, stream position {} "
			                           "MiB of a {} MiB ring",
			                           slots.size_bytes(),
			                           m_context.GetCommandScheduler().CurrentTick(),
			                           stream.Position() >> 20u, stream.Size() >> 20u));
		}
		// KYTY_HEAP_UPLOAD_WAIT=1 restores the waiting upload, only to show the test failing.
		static const bool wait = [] {
			const char* value = std::getenv("KYTY_HEAP_UPLOAD_WAIT");
			return value != nullptr && value[0] == '1';
		}();
		auto [mapped, offset] = stream.Map(slots.size_bytes(), sizeof(uint32_t), wait);
		if (mapped != nullptr) {
			std::memcpy(mapped, slots.data(), slots.size_bytes());
			stream.Commit();
		} else {
			auto& scheduler = m_context.GetCommandScheduler();
			if (BdaGuard::Mode() != 0) {
				const auto tick = scheduler.CurrentTick();
				BdaGuard::Report(fmt::format(
				    "heap slot upload of {} bytes found the stream ring full at tick {} ({}); "
				    "waiting here {} the command buffer under the draw being recorded",
				    slots.size_bytes(), tick,
				    slots.size_bytes() > stream.Size() ? "larger than the ring"
				    : scheduler.IsFree(tick - 1u)      ? "its space belongs to this tick"
				                                       : "its space belongs to this or a pending tick",
				    slots.size_bytes() > stream.Size() ? "would have stopped the emulator instead of submitting"
				    : scheduler.IsFree(tick - 1u)      ? "submitted"
				                                       : "could have submitted"));
			}
			auto temporary = std::make_unique<Buffer>(
			    m_context.GetGraphics(), scheduler, MemoryUsage::Upload, 0,
			    vk::BufferUsageFlagBits::eTransferSrc, slots.size_bytes());
			std::memcpy(temporary->Mapped().data(), slots.data(), slots.size_bytes());
			temporary->Flush(0, slots.size_bytes());
			source = temporary.get();
			offset = 0;
			scheduler.DeferRelease([owner = std::move(temporary)] {});
		}
		view->buffer->CopyFrom(buffer, *source, offset, view->dirty_begin * sizeof(uint32_t),
		                       slots.size_bytes());
		view->dirty_begin = view->dirty_end = 0;
	}
	m_active.clear();
	Retire();
	// Images that left the read layout since discovery, such as by its own uploads, return to it
	// now, freed ones too; the next discovery still takes their changes.
	{
		std::scoped_lock lock {Image::heap_mutex};
		m_changes.assign(Image::heap_changes.begin(), Image::heap_changes.end());
	}
	std::ranges::sort(m_changes, {}, &Image::HeapChange::serial);
	std::erase_if(m_changes, [&](const Image::HeapChange& change) {
		const auto id = CurrentImage(change);
		if (id) m_away.emplace_back(change.serial, id);
		return static_cast<bool>(id);
	});
	ForEachChanged([&](uint32_t index) {
		m_away.emplace_back(m_elements[index].serial, m_elements[index].image);
	});
	std::ranges::sort(m_away);
	m_away.erase(std::ranges::unique(m_away).begin(), m_away.end());
	// This draw's own bindings own the layouts of its attachments and storage images. An image
	// this draw's discovery freed lives until the tick ends, and its slots still select it.
	auto& textures = m_context.GetTextureCache();
	std::erase_if(m_away, [&](const std::pair<uint64_t, ImageId>& away) {
		auto* image = textures.m_slot_images.try_get(away.second);
		if (image == nullptr || image->serial != away.first ||
		    image->heap_refs.load(std::memory_order_relaxed) == 0u) {
			return true;
		}
		if (image->binding.is_target || image->binding.shader_write ||
		    image->binding.force_general) {
			return false;
		}
		image->Transit(image->ReadLayout(), vk::AccessFlagBits2::eShaderRead, std::nullopt,
		               buffer.Handle());
		return true;
	});
	buffer.Handle().bindDescriptorSets(bind_point, layout, IR::ImageHeapSet, 1, &m_set, 0,
	                                   nullptr);
}

ImageHeap::GuardView ImageHeap::GuardFind(uint64_t slots) const {
	GuardView result;
	for (const auto& memory: m_memories) {
		for (const auto& view: memory->views) {
			if (view->buffer == nullptr || view->buffer->BufferDeviceAddress() != slots) continue;
			result.found      = true;
			result.kinds      = static_cast<uint32_t>(view->kind_keys.size());
			result.slot_count = static_cast<uint32_t>(view->slots.size() - 1u);
			// The words not uploaded yet; the others were checked before their upload.
			for (auto slot = view->dirty_begin; slot < view->dirty_end; ++slot) {
				const auto word = view->slots[slot];
				const auto arm  = word >> IR::ImageHeapKindShift;
				if ((word & ElementMask) >= m_elements.size() ||
				    (word != 0u && arm >= view->kind_keys.size())) {
					result.bad      = true;
					result.bad_word = word;
					break;
				}
			}
			return result;
		}
	}
	return result;
}

uint64_t ImageHeap::GuardNullSlots() {
	if (m_guard_null == nullptr) {
		m_guard_null = std::make_unique<Buffer>(
		    m_context.GetGraphics(), m_context.GetCommandScheduler(), MemoryUsage::Upload, 0,
		    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress,
		    4096);
		std::ranges::fill(m_guard_null->Mapped(), uint8_t {0});
		m_guard_null->Flush(0, m_guard_null->Size());
	}
	return m_guard_null->BufferDeviceAddress();
}

void ImageHeap::UnmapMemory(uint64_t address, uint64_t size) {
	// The range's images are freed. Their words stay null, even once it is mapped again, until
	// their descriptors change; other changes wait for discovery.
	TakeChanges();
	const auto other = [&](const Image::HeapChange& change) {
		return !ImageRangeOverlaps(change.data, {address, size});
	};
	{
		std::scoped_lock lock {Image::heap_mutex};
		std::ranges::copy_if(m_changes, std::back_inserter(Image::heap_changes), other);
	}
	std::erase_if(m_changes, other);
	std::scoped_lock lock {m_mutex};
	std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		if (address >= memory->base + memory->size || memory->base >= address + size) {
			return false;
		}
		Drop(*memory);
		return true;
	});
	if (ForEachChanged([&](uint32_t index) { Mark(index, State::Buried); })) Repoint();
}

void ImageHeap::RunGarbageCollector() {
	std::scoped_lock lock {m_mutex};
	std::erase_if(m_memories, [&](const std::unique_ptr<Memory>& memory) {
		memory->idle = memory->draw > m_suspend_draw ? 0u : memory->idle + 1u;
		if (memory->idle < (memory->size < MinTableBytes ? 1u : LargeTableIdleFrames)) {
			return false;
		}
		Drop(*memory);
		return true;
	});
	m_suspend_draw = m_draw;
}

void ImageHeap::Drop(Memory& memory) {
	auto& scheduler = m_context.GetCommandScheduler();
	Unwatch(memory);
	for (auto& view: memory.views) {
		std::erase(m_active, view.get());
		m_view_ids.erase(view->id);
		for (uint64_t probe = 0; probe < view->count; ++probe) {
			Release(view->slots[(view->first + probe * view->step) / 4u]);
		}
		if (scheduler.Active()) {
			scheduler.DeferRelease([buffer = std::move(view->buffer)] {});
		}
	}
}

} // namespace Libs::Graphics
