#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MATERIALIZATIONMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MATERIALIZATIONMEMO_H_

#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <vector>

struct XXH3_state_s;

namespace Libs::Graphics {

struct HashStateDeleter {
	void operator()(XXH3_state_s* state) const;
};

// Records resident image-heap acquisitions so a memo hit can repeat their per-draw side effects.
class RecordingImageHeap final: public ShaderRecompiler::IR::ImageHeapSource {
public:
	struct Acquired {
		ShaderRecompiler::IR::ImageHeapRequest request;
		uint64_t                               slots      = 0;
		uint32_t                               slot_count = 0;
		std::vector<ShaderRecompiler::IR::DescriptorValue> kinds;
	};

	explicit RecordingImageHeap(ShaderRecompiler::IR::ImageHeapSource* inner): m_inner(inner) {}
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest& request,
	             ShaderRecompiler::IR::ImageHeapView& view) override;
	[[nodiscard]] bool Declined() const { return m_declined; }
	std::vector<Acquired>& Take() { return m_acquired; }

private:
	ShaderRecompiler::IR::ImageHeapSource* m_inner    = nullptr;
	std::vector<Acquired>                  m_acquired;
	bool                                   m_declined = false;
};

// Hashes callback reads as the walk sees them. Store refuses an entry when the bytes armed after
// the walk differ from the bytes that produced the snapshot.
class WalkReadHashes {
public:
	[[nodiscard]] ShaderRecompiler::IR::SrtRuntime Wrap(
	    const ShaderRecompiler::IR::SrtRuntime& runtime);
	enum class State : uint8_t { NotRead, Stable, Unstable };
	[[nodiscard]] State Find(uint64_t address, uint64_t size, uint64_t& hash) const;

private:
	static bool Read(void* user, uint64_t address, std::span<uint32_t> values);
	struct Seen {
		uint64_t hash     = 0;
		bool     unstable = false;
	};
	ShaderRecompiler::IR::SrtRuntime      m_inner;
	std::map<std::pair<uint64_t, uint64_t>, Seen> m_seen;
};

enum class MemoOutcome : uint8_t {
	Hit,
	Empty,
	ShaderBase,
	IndirectArgs,
	UserData,
	WorkgroupCounts,
	Unreadable,
	RangeChanged,
	HeapChanged,
	Count
};

struct MemoLookupStats {
	uint32_t ranges_stamped  = 0;
	uint32_t ranges_compared = 0;
	uint64_t bytes_compared  = 0;
	uint32_t entries_tried   = 0;
	uint32_t ranges_small    = 0;
	uint32_t ranges_large    = 0;
	uint64_t bytes_large     = 0;
	bool     walk_changed    = false;
	bool     walk_unobserved = false;
};

class MemoizedMaterialization {
public:
	struct Read {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t hash    = 0;
		uint32_t byte_offset = 0;
		bool     has_bytes   = false;
		std::shared_ptr<ShaderRecompiler::IR::MemoryChangeStamp> stamp;
	};
	struct Entry {
		std::vector<uint32_t> user_data;
		std::vector<uint32_t> workgroup_counts;
		uint64_t              shader_base           = 0;
		uint64_t              indirect_args_address = 0;
		std::vector<Read>     reads;
		std::vector<uint8_t>  bytes;
		std::vector<RecordingImageHeap::Acquired> heap;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		uint64_t                                    serial = 0;
		uint64_t                                    used   = 0;
	};

	static constexpr uint64_t SmallRangeBytes = 256;

	void Configure(uint32_t entries, ShaderRecompiler::IR::MemoryChangeSource* stamps,
	               uint32_t stamp_pages);
	[[nodiscard]] uint32_t Capacity() const { return m_capacity; }
	void Clear();

	Entry* Lookup(const ShaderRecompiler::IR::SrtRuntime& runtime, MemoOutcome& outcome,
	              MemoLookupStats& stats);
	Entry* Store(const ShaderRecompiler::IR::SrtRuntime& runtime,
	             const ShaderRecompiler::IR::ResourceSnapshot& resources,
	             const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	             RecordingImageHeap& heap, uint32_t& evicted, MemoLookupStats& stats,
	             const WalkReadHashes* walk = nullptr);

	uint64_t current = 0;
	[[nodiscard]] size_t Size() const { return m_entries.size(); }

	struct StoreGate {
		static constexpr uint32_t Window   = 64;
		static constexpr uint32_t MinHits  = 4;
		static constexpr uint32_t Cooldown = 1024;

		uint32_t lookups  = 0;
		uint32_t hits     = 0;
		uint32_t cooldown = 0;

		void Note(bool hit) {
			if (cooldown != 0) {
				cooldown--;
				return;
			}
			lookups++;
			hits += hit ? 1u : 0u;
			if (lookups >= Window) {
				if (hits < MinHits) cooldown = Cooldown;
				lookups = 0;
				hits    = 0;
			}
		}
		[[nodiscard]] bool Allows() const { return cooldown == 0; }
	};
	StoreGate store_gate;

private:
	bool KeyMatches(const Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                MemoOutcome& why) const;
	bool HashRange(const ShaderRecompiler::IR::SrtRuntime& runtime, uint64_t address, uint64_t size,
	               uint64_t& hash);
	bool ValidateReads(Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime,
	                   MemoOutcome& why, MemoLookupStats& stats);
	bool ReplayHeap(const Entry& entry, const ShaderRecompiler::IR::SrtRuntime& runtime);

	std::vector<std::shared_ptr<Entry>> m_entries;
	std::vector<uint32_t>               m_scratch;
	uint32_t                            m_capacity   = 1;
	uint32_t                            m_stamp_pages = 0;
	uint64_t                            m_tick       = 0;
	uint64_t                            m_serials    = 0;
	std::unique_ptr<XXH3_state_s, HashStateDeleter> m_hash_state;
	ShaderRecompiler::IR::MemoryChangeSource*       m_stamps = nullptr;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MATERIALIZATIONMEMO_H_
