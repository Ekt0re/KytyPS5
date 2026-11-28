#include "graphics/host_gpu/memoryTracker.h"

#include "common/alignment.h"
#include "common/assert.h"

namespace Libs::Graphics {

namespace {

class MemoryStamp final: public ShaderRecompiler::IR::MemoryChangeStamp {
public:
	struct Region {
		RegionManager* manager   = nullptr;
		uint64_t       generation = 0;
	};

	explicit MemoryStamp(std::vector<Region> regions): m_regions(std::move(regions)) {}

	[[nodiscard]] bool Unchanged() const override {
		return std::ranges::all_of(m_regions, [](const Region& region) {
			return region.manager->Generation() == region.generation;
		});
	}

private:
	std::vector<Region> m_regions;
};

} // namespace

static_assert(std::atomic<void*>::is_always_lock_free);

MemoryTracker::MemoryTracker(PageManager& page_manager): m_page_manager(page_manager) {
	m_regions = std::make_unique<std::atomic<RegionManager*>[]>(REGION_COUNT);
}

MemoryTracker::~MemoryTracker() = default;

#if KYTY_BUILD == KYTY_BUILD_DEBUG
void MemoryTracker::ValidateGpuDirtyPages(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                          const char* operation) const noexcept {
	if (!GuestRange {vaddr, size}.Valid() || (vaddr & (TRACKER_PAGE_SIZE - 1)) != 0 ||
	    (size & (TRACKER_PAGE_SIZE - 1)) != 0) {
		EXIT("MemoryTracker: invalid dirty-page validation range\n");
	}
	for (auto page = vaddr; page < vaddr + size; page += TRACKER_PAGE_SIZE) {
		if (!dirty.Intersects(page, TRACKER_PAGE_SIZE)) {
			EXIT("MemoryTracker: GPU-dirty tracker page has no dirty bytes, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}

void MemoryTracker::ValidateGpuDirtyOwnership(const RangeSet& dirty, uint64_t vaddr, uint64_t size,
                                              const char* operation) {
	ValidateRange(vaddr, size);
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	for (auto page = begin; page < end; page += TRACKER_PAGE_SIZE) {
		const bool has_dirty_bytes = dirty.Intersects(page, TRACKER_PAGE_SIZE);
		if (IsRegionGpuModified(page, TRACKER_PAGE_SIZE) != has_dirty_bytes) {
			EXIT("MemoryTracker: tracker and byte ownership disagree, operation=%s "
			     "addr=0x%016" PRIx64 "\n",
			     operation, page);
		}
	}
}
#endif

void MemoryTracker::ValidateRange(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("invalid memory tracker range\n");
	}
}

RegionManager* MemoryTracker::GetOrCreateRegion(uint64_t index) {
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	std::lock_guard lock(m_region_mutex);
	if (auto* manager = m_regions[index].load(std::memory_order_acquire); manager != nullptr) {
		return manager;
	}
	auto  manager = std::make_unique<RegionManager>(m_page_manager, index * TRACKER_REGION_SIZE);
	auto* ptr     = manager.get();
	m_region_storage.push_back(std::move(manager));
	m_regions[index].store(ptr, std::memory_order_release);
	return ptr;
}

// Queries take no lock: dirty words are read atomically, and a query runs on the GPU thread, which
// alone writes GPU-dirty bits while other threads only set CPU-dirty ones, so it sees one instant.
bool MemoryTracker::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return Iterate<true>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return manager->IsModified<DirtySource::Cpu>(offset, bytes);
	});
}

bool MemoryTracker::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		return manager->IsModified<DirtySource::Gpu>(offset, bytes);
	});
}

std::shared_ptr<ShaderRecompiler::IR::MemoryChangeStamp> MemoryTracker::Track(uint64_t address,
                                                                           uint64_t size,
                                                                           uint32_t max_pages) {
	if (size == 0) return std::make_shared<MemoryStamp>(std::vector<MemoryStamp::Region> {});
	ValidateRange(address, size);
	const auto first_page = address / TRACKER_PAGE_SIZE;
	const auto last_page  = (address + size - 1u) / TRACKER_PAGE_SIZE;
	if (max_pages == 0 || last_page - first_page + 1u > max_pages) return {};

	std::vector<MemoryStamp::Region> regions;
	uint64_t remaining = size;
	uint64_t index     = address / TRACKER_REGION_SIZE;
	uint64_t offset    = address % TRACKER_REGION_SIZE;
	while (remaining != 0) {
		const auto bytes   = std::min(TRACKER_REGION_SIZE - offset, remaining);
		auto*      manager = m_regions[index].load(std::memory_order_acquire);
		if (manager == nullptr) return {};
		{
			std::scoped_lock lock(manager->lock);
			if (manager->IsModified<DirtySource::Cpu>(offset, bytes) ||
			    manager->IsModified<DirtySource::Gpu>(offset, bytes)) {
				return {};
			}
			manager->EnableStampGeneration();
			regions.push_back({manager, manager->Generation()});
		}
		remaining -= bytes;
		offset = 0;
		index++;
	}
	// Clean pages are already write-watched by the upload transition that cleared their dirty
	// state. A stamp observes that region-owned watch; taking another counted reference here would
	// outlive the stamp and keep the page protected after it became CPU-dirty.
	return std::make_shared<MemoryStamp>(std::move(regions));
}

void MemoryTracker::MarkRegionAsCpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Cpu, true>(offset, bytes);
	});
}

void MemoryTracker::MarkCleanAsCpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->MarkCleanAsCpuModified(offset, bytes);
	});
}

void MemoryTracker::UnmarkRegionAsGpuModified(uint64_t vaddr, uint64_t size) {
	Iterate<false>(vaddr, size, [](RegionManager* manager, uint64_t offset, uint64_t bytes) {
		std::scoped_lock lock(manager->lock);
		manager->ChangeState<DirtySource::Gpu, false>(offset, bytes);
	});
}

} // namespace Libs::Graphics
