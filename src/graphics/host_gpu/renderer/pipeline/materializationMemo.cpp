#include "graphics/host_gpu/renderer/pipeline/materializationMemo.h"

#include <xxhash.h>

#include <algorithm>
#include <cstring>
#include <utility>

namespace Libs::Graphics {

bool RecordingImageHeap::Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
                                  ShaderRecompiler::IR::ImageHeapView& view) {
	if (m_inner == nullptr || !m_inner->Acquire(request, view)) {
		m_declined = true;
		return false;
	}
	m_acquired.push_back({.request    = request,
	                      .slots      = view.slots,
	                      .slot_count = view.slot_count,
	                      .kinds      = {view.kinds.begin(), view.kinds.end()}});
	return true;
}

ShaderRecompiler::IR::SrtRuntime WalkReadHashes::Wrap(
    const ShaderRecompiler::IR::SrtRuntime& runtime) {
	m_seen.clear();
	m_inner = runtime;
	auto wrapped = runtime;
	if (runtime.read_specialization_memory != nullptr) {
		wrapped.userdata                   = this;
		wrapped.read_specialization_memory = &WalkReadHashes::Read;
	}
	return wrapped;
}

bool WalkReadHashes::Read(void* user, uint64_t address, std::span<uint32_t> values) {
	auto& self = *static_cast<WalkReadHashes*>(user);
	if (!self.m_inner.read_specialization_memory(self.m_inner.userdata, address, values)) return false;
	const uint64_t hash = values.empty() ? 0u : XXH3_64bits(values.data(), values.size_bytes());
	const auto [at, fresh] = self.m_seen.try_emplace(
	    std::pair<uint64_t, uint64_t> {address, values.size_bytes()}, Seen {hash, false});
	if (!fresh && at->second.hash != hash) at->second.unstable = true;
	return true;
}

WalkReadHashes::State WalkReadHashes::Find(uint64_t address, uint64_t size, uint64_t& hash) const {
	const auto found = m_seen.find(std::pair<uint64_t, uint64_t> {address, size});
	if (found == m_seen.end()) return State::NotRead;
	hash = found->second.hash;
	return found->second.unstable ? State::Unstable : State::Stable;
}

void HashStateDeleter::operator()(XXH3_state_s* state) const {
	XXH3_freeState(state);
}

void MemoizedMaterialization::Configure(uint32_t entries,
                                        ShaderRecompiler::IR::MemoryChangeSource* stamps,
                                        uint32_t stamp_pages) {
	const auto capacity = std::max(entries, 1u);
	if (m_capacity != capacity || m_stamps != stamps || m_stamp_pages != stamp_pages) Clear();
	m_capacity   = capacity;
	m_stamps     = stamps;
	m_stamp_pages = stamp_pages;
}

void MemoizedMaterialization::Clear() {
	m_entries.clear();
	current    = 0;
	store_gate = {};
}

bool MemoizedMaterialization::KeyMatches(const Entry& entry,
                                        const ShaderRecompiler::IR::SrtRuntime& runtime,
                                        MemoOutcome& why) const {
	if (entry.shader_base != runtime.shader_base) {
		why = MemoOutcome::ShaderBase;
	} else if (entry.indirect_args_address != runtime.indirect_args_address) {
		why = MemoOutcome::IndirectArgs;
	} else if (!std::ranges::equal(entry.user_data, runtime.user_data)) {
		why = MemoOutcome::UserData;
	} else if (!std::ranges::equal(entry.workgroup_counts, runtime.workgroup_counts)) {
		why = MemoOutcome::WorkgroupCounts;
	} else {
		return true;
	}
	return false;
}

bool MemoizedMaterialization::HashRange(const ShaderRecompiler::IR::SrtRuntime& runtime,
                                         uint64_t address, uint64_t size, uint64_t& hash) {
	if (size == 0u) {
		hash = 0u;
		return true;
	}
	if (runtime.read_specialization_memory == nullptr || (size & 3u) != 0u ||
	    size / sizeof(uint32_t) > m_scratch.max_size()) {
		return false;
	}
	constexpr uint64_t Chunk = 32u << 10u;
	if (size <= Chunk) {
		m_scratch.resize(static_cast<size_t>(size / sizeof(uint32_t)));
		if (!runtime.read_specialization_memory(runtime.userdata, address, m_scratch)) return false;
		hash = XXH3_64bits(m_scratch.data(), static_cast<size_t>(size));
		return true;
	}
	m_scratch.resize(static_cast<size_t>(Chunk / sizeof(uint32_t)));
	if (m_hash_state == nullptr) m_hash_state.reset(XXH3_createState());
	XXH3_state_t* state = m_hash_state.get();
	if (state == nullptr) return false;
	XXH3_64bits_reset(state);
	for (uint64_t done = 0; done < size; done += Chunk) {
		const auto piece = std::min(Chunk, size - done);
		const std::span<uint32_t> words(m_scratch.data(), static_cast<size_t>(piece / sizeof(uint32_t)));
		if (!runtime.read_specialization_memory(runtime.userdata, address + done, words)) return false;
		XXH3_64bits_update(state, words.data(), static_cast<size_t>(piece));
	}
	hash = XXH3_64bits_digest(state);
	return true;
}

bool MemoizedMaterialization::ValidateReads(Entry& entry,
                                             const ShaderRecompiler::IR::SrtRuntime& runtime,
                                             MemoOutcome& why, MemoLookupStats& stats) {
	for (auto& read: entry.reads) {
		if (read.stamp != nullptr && read.stamp->Unchanged()) {
			++stats.ranges_stamped;
			continue;
		}
		std::shared_ptr<ShaderRecompiler::IR::MemoryChangeStamp> fresh_stamp;
		if (m_stamps != nullptr && m_stamp_pages != 0u) {
			fresh_stamp = m_stamps->Track(read.address, read.size, m_stamp_pages);
		}
		bool same = false;
		if (read.has_bytes) {
			if (read.size == 0u) {
				same = true;
			} else {
				m_scratch.resize(static_cast<size_t>(read.size / sizeof(uint32_t)));
				if (!runtime.read_specialization_memory(runtime.userdata, read.address, m_scratch)) {
					why = MemoOutcome::Unreadable;
					return false;
				}
				same = std::memcmp(m_scratch.data(), entry.bytes.data() + read.byte_offset,
				                   static_cast<size_t>(read.size)) == 0;
			}
			++stats.ranges_small;
		} else {
			uint64_t hash = 0;
			if (!HashRange(runtime, read.address, read.size, hash)) {
				why = MemoOutcome::Unreadable;
				return false;
			}
			same = hash == read.hash;
			if (read.size != 0u) {
				++stats.ranges_large;
				stats.bytes_large += read.size;
			}
		}
		++stats.ranges_compared;
		stats.bytes_compared += read.size;
		if (!same) {
			why = MemoOutcome::RangeChanged;
			return false;
		}
		read.stamp = std::move(fresh_stamp);
	}
	return true;
}

bool MemoizedMaterialization::ReplayHeap(const Entry& entry,
                                        const ShaderRecompiler::IR::SrtRuntime& runtime) {
	for (const auto& acquired: entry.heap) {
		ShaderRecompiler::IR::ImageHeapView view;
		if (runtime.image_heap == nullptr || !runtime.image_heap->Acquire(acquired.request, view) ||
		    view.slots != acquired.slots || view.slot_count != acquired.slot_count ||
		    !std::ranges::equal(view.kinds, acquired.kinds)) {
			return false;
		}
	}
	return true;
}

MemoizedMaterialization::Entry* MemoizedMaterialization::Lookup(
    const ShaderRecompiler::IR::SrtRuntime& runtime, MemoOutcome& outcome, MemoLookupStats& stats) {
	if (m_entries.empty()) {
		outcome = MemoOutcome::Empty;
		return nullptr;
	}
	const Entry* newest = m_entries.front().get();
	for (const auto& entry: m_entries) {
		if (entry->used > newest->used) newest = entry.get();
	}
	MemoOutcome key_reason = MemoOutcome::Empty;
	MemoOutcome content    = MemoOutcome::Hit;
	for (const auto& candidate: m_entries) {
		auto&       entry = *candidate;
		MemoOutcome why   = MemoOutcome::Hit;
		if (!KeyMatches(entry, runtime, why)) {
			if (&entry == newest) key_reason = why;
			continue;
		}
		++stats.entries_tried;
		if (!ValidateReads(entry, runtime, why, stats)) {
			content = why;
			continue;
		}
		if (!ReplayHeap(entry, runtime)) {
			content = MemoOutcome::HeapChanged;
			continue;
		}
		entry.used = ++m_tick;
		outcome    = MemoOutcome::Hit;
		return &entry;
	}
	outcome = content != MemoOutcome::Hit ? content : key_reason;
	return nullptr;
}

MemoizedMaterialization::Entry* MemoizedMaterialization::Store(
    const ShaderRecompiler::IR::SrtRuntime& runtime,
    const ShaderRecompiler::IR::ResourceSnapshot& resources,
    const ShaderRecompiler::IR::ResourceSpecialization& specialization, RecordingImageHeap& heap,
    uint32_t& evicted, MemoLookupStats& stats, const WalkReadHashes* walk) {
	if (heap.Declined()) return nullptr;
	auto ranges = resources.specialization_reads;
	for (const auto& acquired: heap.Take()) {
		const auto table = std::ranges::find(
		    ranges, std::pair<uint64_t, uint64_t> {acquired.request.base, acquired.request.size});
		if (table != ranges.end()) ranges.erase(table);
	}
	std::ranges::sort(ranges);
	ranges.erase(std::ranges::unique(ranges).begin(), ranges.end());

	auto fresh = std::make_shared<Entry>();
	fresh->user_data.assign(runtime.user_data.begin(), runtime.user_data.end());
	fresh->workgroup_counts.assign(runtime.workgroup_counts.begin(), runtime.workgroup_counts.end());
	fresh->shader_base           = runtime.shader_base;
	fresh->indirect_args_address = runtime.indirect_args_address;
	fresh->reads.reserve(ranges.size());
	for (const auto& [address, size]: ranges) {
		Read read {.address = address, .size = size};
		if (m_stamps != nullptr && m_stamp_pages != 0u) {
			read.stamp = m_stamps->Track(address, size, m_stamp_pages);
		}
		if (!HashRange(runtime, address, size, read.hash)) return nullptr;
		if (size != 0u && size <= SmallRangeBytes) {
			read.has_bytes   = true;
			read.byte_offset = static_cast<uint32_t>(fresh->bytes.size());
			const auto* data = reinterpret_cast<const uint8_t*>(m_scratch.data());
			fresh->bytes.insert(fresh->bytes.end(), data, data + size);
		}
		if (walk != nullptr && size != 0u) {
			uint64_t seen = 0;
			const auto state = walk->Find(address, size, seen);
			if (state == WalkReadHashes::State::NotRead) {
				stats.walk_unobserved = true;
				return nullptr;
			}
			if (state == WalkReadHashes::State::Unstable ||
			    (state == WalkReadHashes::State::Stable && seen != read.hash)) {
				stats.walk_changed = true;
				return nullptr;
			}
		}
		++stats.ranges_compared;
		stats.bytes_compared += size;
		fresh->reads.push_back(std::move(read));
	}
	fresh->heap           = std::move(heap.Take());
	fresh->resources      = resources;
	fresh->specialization = specialization;
	fresh->serial         = ++m_serials;
	fresh->used           = ++m_tick;
	if (m_entries.size() >= m_capacity) {
		const auto oldest = std::ranges::min_element(
		    m_entries, [](const auto& a, const auto& b) { return a->used < b->used; });
		m_entries.erase(oldest);
		++evicted;
	}
	m_entries.push_back(std::move(fresh));
	return m_entries.back().get();
}

} // namespace Libs::Graphics
