#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTREADLOG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTREADLOG_H_

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

// Every guest read of a resource walk made ahead of the command thread (lookahead), with the
// bytes it saw (or their hash, for large reads). The command thread uses the walk's result only
// if every logged read still gives the same bytes and the GPU owns none of them (a walk on the
// command thread would read those back); otherwise it walks itself. The idea and the name come
// from TheCruZ's KytyEmu (GuestReadLog::StillValid, 11d077ed/81022ff0); this is a new, smaller
// implementation.
class GuestReadLog {
public:
	// Reads above this size keep a hash instead of their bytes.
	static constexpr uint32_t MaxBytesKept = 4096;

	// What the check reads through. The emulator's reads the backing store; tests read memory.
	class Memory {
	public:
		// A direct view of [address, address + size) or nullptr; the check then copies with Read.
		virtual const void* View(uint64_t address, uint64_t size) = 0;
		virtual bool        Read(uint64_t address, void* data, uint64_t size) = 0;
		// True when the GPU wrote bytes of the range that guest memory does not hold yet.
		virtual bool GpuOwned(uint64_t address, uint64_t size) = 0;

	protected:
		~Memory() = default;
	};

	enum class Check : uint8_t { Valid, Changed, GpuOwned, Unreadable };

	void Clear();
	// A read of words at address (a range read through the walk's memory callback).
	void AddRange(uint64_t address, std::span<const uint32_t> words);
	// The walker's own DWORD loads, (address, value) as it used them.
	void AddLoads(std::span<const std::pair<uint64_t, uint32_t>> loads);
	// Sorts and merges the loads. False when two loads of one address saw different values (the
	// memory changed during the walk): such a log must not be used.
	bool Finish();

	[[nodiscard]] Check StillValid(Memory& memory, uint64_t& bytes_compared) const;

	[[nodiscard]] size_t RangeCount() const { return m_ranges.size(); }
	[[nodiscard]] size_t LoadCount() const { return m_loads.size(); }

private:
	struct Range {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t hash    = 0;          // Ranges over MaxBytesKept.
		uint32_t offset  = UINT32_MAX; // Into m_bytes, or UINT32_MAX for a hashed range.
	};
	std::vector<Range>                         m_ranges;
	std::vector<uint8_t>                       m_bytes;
	std::vector<std::pair<uint64_t, uint32_t>> m_loads;
	bool                                       m_finished = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GUESTREADLOG_H_
