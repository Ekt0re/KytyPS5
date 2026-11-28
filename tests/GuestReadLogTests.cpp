#include "graphics/host_gpu/renderer/pipeline/guestReadLog.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using Libs::Graphics::GuestReadLog;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "GuestReadLogTests: failed: %s\n", text);
		std::abort();
	}
}

// Process memory, with a GPU-owned range and a range that cannot be read.
struct TestMemory final: GuestReadLog::Memory {
	uint64_t gpu_begin = 0, gpu_end = 0;
	uint64_t bad_begin = 0, bad_end = 0;
	bool     views     = true;
	uint32_t reads     = 0;

	const void* View(uint64_t address, uint64_t size) override {
		return views && !Bad(address, size) ? reinterpret_cast<const void*>(address) : nullptr;
	}
	bool Read(uint64_t address, void* data, uint64_t size) override {
		reads++;
		if (Bad(address, size)) return false;
		std::memcpy(data, reinterpret_cast<const void*>(address), size);
		return true;
	}
	bool GpuOwned(uint64_t address, uint64_t size) override {
		return address < gpu_end && gpu_begin < address + size;
	}
	[[nodiscard]] bool Bad(uint64_t address, uint64_t size) const {
		return address < bad_end && bad_begin < address + size;
	}
};

uint64_t Address(const void* pointer) {
	return reinterpret_cast<uint64_t>(pointer);
}

void TestLoadsAndRanges() {
	std::array<uint32_t, 8> table {1, 2, 3, 4, 5, 6, 7, 8};
	std::vector<uint32_t>   big(GuestReadLog::MaxBytesKept / 4u + 16u, 0x5au);
	GuestReadLog            log;
	// Loads out of order, one twice: adjacent DWORDs merge into one range each.
	const std::array<std::pair<uint64_t, uint32_t>, 5> loads {{{Address(&table[3]), 4u},
	                                                           {Address(&table[0]), 1u},
	                                                           {Address(&table[1]), 2u},
	                                                           {Address(&table[0]), 1u},
	                                                           {Address(&table[6]), 7u}}};
	log.AddLoads(loads);
	log.AddRange(Address(&table[4]), std::span(table).subspan(4, 2));
	log.AddRange(Address(big.data()), big);
	Check(log.Finish(), "a consistent log was refused");
	Check(log.RangeCount() == 5, "loads were not merged into ranges of adjacent DWORDs");
	TestMemory memory;
	uint64_t   bytes = 0;
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::Valid &&
	          bytes == 4u * 4u + 8u + big.size() * 4u,
	      "an unchanged log did not hold");
	memory.views = false;
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::Valid && memory.reads == 5,
	      "an unchanged log did not hold when read by copy");
	memory.views = true;
	table[2] = 99u; // Never read: no change.
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::Valid, "an unread change counted");
	table[1] = 99u;
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::Changed, "a changed load held");
	table[1] = 2u;
	table[5] = 0u;
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::Changed, "a changed range held");
	table[5] = 6u;
	big.back() = 0u;
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::Changed,
	      "a changed hashed range held");
	big.back() = 0x5au;
	memory.gpu_begin = Address(&table[6]);
	memory.gpu_end   = memory.gpu_begin + 1u;
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::GpuOwned,
	      "a range the GPU owns held");
	memory.gpu_begin = memory.gpu_end = 0;
	memory.bad_begin = Address(&table[0]);
	memory.bad_end   = memory.bad_begin + 4u;
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::Unreadable,
	      "an unreadable range held");
}

void TestTornLoads() {
	const uint32_t word = 3u;
	GuestReadLog   log;
	const std::array<std::pair<uint64_t, uint32_t>, 2> loads {{{Address(&word), 3u},
	                                                           {Address(&word), 4u}}};
	log.AddLoads(loads);
	Check(!log.Finish(), "a log whose loads of one address differ was accepted");
	log.Clear();
	Check(log.Finish() && log.RangeCount() == 0, "an empty log was refused");
	TestMemory memory;
	uint64_t   bytes = 0;
	Check(log.StillValid(memory, bytes) == GuestReadLog::Check::Valid, "an empty log did not hold");
}

} // namespace

int main() {
	TestLoadsAndRanges();
	TestTornLoads();
	std::puts("GuestReadLogTests: all cases passed");
	return 0;
}
