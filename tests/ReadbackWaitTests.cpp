#include "graphics/host_gpu/renderer/cache/readbackWait.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <set>

namespace {

using Libs::Graphics::ReadbackWait::FlushThrottle;
using Libs::Graphics::ReadbackWait::HotWriters;
using Libs::Graphics::ReadbackWait::MirrorIndex;
using Libs::Graphics::ReadbackWait::UnknownTick;
using Libs::Graphics::ReadbackWait::WriteTicks;

constexpr uint64_t Page = 4096;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ReadbackWaitTests: failed: %s\n", message);
		std::abort();
	}
}

void TestWriteTicks() {
	WriteTicks ticks;
	Check(ticks.Latest(0x1000, 0x1100) == UnknownTick, "nothing assigned is unknown");
	ticks.Assign(0x1000, 0x2000, 5);
	Check(ticks.Latest(0x1000, 0x2000) == 5, "whole range");
	Check(ticks.Latest(0x1800, 0x1900) == 5, "inner range");
	Check(ticks.Latest(0x0f00, 0x1100) == UnknownTick, "a gap before is unknown");
	Check(ticks.Latest(0x1f00, 0x2100) == UnknownTick, "a gap after is unknown");
	// A later write of the middle splits the range.
	ticks.Assign(0x1400, 0x1800, 9);
	Check(ticks.Latest(0x1000, 0x1400) == 5, "left part keeps its tick");
	Check(ticks.Latest(0x1800, 0x2000) == 5, "right part keeps its tick");
	Check(ticks.Latest(0x1400, 0x1800) == 9, "middle has the new tick");
	Check(ticks.Latest(0x1000, 0x2000) == 9, "the latest over all parts");
	Check(ticks.Size() == 3, "split into three ranges");
	// Repeated assignment of the same range in the same tick (the memo) and a new tick.
	ticks.Assign(0x1400, 0x1800, 9);
	Check(ticks.Size() == 3, "memo keeps the map");
	ticks.Assign(0x1400, 0x1800, 11);
	Check(ticks.Latest(0x1400, 0x1800) == 11, "exact range gets the newer tick");
	// A write covering everything replaces all parts.
	ticks.Assign(0x0800, 0x2800, 12);
	Check(ticks.Size() == 1 && ticks.Latest(0x0800, 0x2800) == 12, "covering write merges");
	// Adjacent ranges: contiguous coverage is known.
	ticks.Assign(0x2800, 0x3000, 3);
	Check(ticks.Latest(0x2000, 0x3000) == 12, "adjacent ranges are contiguous");
	ticks.Assign(0x0, 0x0, 99);
	Check(ticks.Latest(0x0800, 0x3000) == 12, "empty assignment ignored");
}

void TestWriteTicksRandom() {
	// Differential check against a per-unit array.
	constexpr uint64_t Units = 256;
	uint64_t           shadow[Units];
	for (auto& value: shadow) {
		value = UnknownTick;
	}
	WriteTicks ticks;
	uint32_t   seed = 12345;
	const auto next = [&seed] {
		seed = seed * 1664525u + 1013904223u;
		return seed >> 8u;
	};
	uint64_t tick = 1;
	for (int step = 0; step < 20000; ++step) {
		const uint64_t a = next() % Units;
		const uint64_t b = a + 1 + next() % 16;
		if (next() % 3 != 0) {
			tick += next() % 2;
			ticks.Assign(a * 16, std::min(b, Units) * 16, tick);
			for (uint64_t unit = a; unit < std::min(b, Units); ++unit) {
				shadow[unit] = tick;
			}
		} else {
			uint64_t expected = 0;
			for (uint64_t unit = a; unit < std::min(b, Units); ++unit) {
				expected = shadow[unit] == UnknownTick ? UnknownTick : std::max(expected, shadow[unit]);
				if (expected == UnknownTick) {
					break;
				}
			}
			Check(ticks.Latest(a * 16, std::min(b, Units) * 16) == expected, "random differential");
		}
	}
}

void TestHotWriters() {
	HotWriters hot;
	Check(hot.Empty(), "starts empty");
	// A readback served from submitted work does not make a range hot.
	hot.NoteWait(0x10010, 0x10020, false, false, Page);
	Check(!hot.Intersects(0x10000, 0x1000), "undrained readback is not hot");
	// A drained one does, page aligned.
	hot.NoteWait(0x10010, 0x10020, false, true, Page);
	Check(hot.Intersects(0x10ff0, 4), "drained readback page is hot");
	Check(!hot.Intersects(0x11000, 4), "next page is not hot");
	// A drained one next to it merges on overlap only.
	hot.NoteWait(0x10800, 0x11800, false, true, Page);
	Check(hot.Size() == 1 && hot.Intersects(0x11400, 4), "overlapping readback widens the entry");
	// Device-address waits: hot only when drained, or when already hot.
	Check(!hot.DeviceAddressHot(), "no device-address hotness yet");
	hot.NoteWait(0x900000, 0x900010, true, false, Page);
	Check(!hot.DeviceAddressHot(), "undrained device-address wait is not hot");
	hot.NoteWait(0x900000, 0x900010, true, true, Page);
	Check(hot.DeviceAddressHot(), "drained device-address wait is hot");
	Check(!hot.Intersects(0x900000, 4), "device-address waits add no range");
	// Entries expire after Lifetime readbacks without a hit.
	for (uint64_t i = 0; i < HotWriters::Lifetime; ++i) {
		hot.NoteWait(0x7000000, 0x7000010, false, false, Page);
	}
	Check(!hot.Intersects(0x10000, 0x2000), "entry expired");
	Check(!hot.DeviceAddressHot(), "device-address hotness expired");
	// At most MaxEntries entries; the least recently hit goes.
	for (uint64_t i = 0; i < HotWriters::MaxEntries + 4; ++i) {
		hot.NoteWait(0x100000000ull + i * 0x10000, 0x100000010ull + i * 0x10000, false, true, Page);
	}
	Check(hot.Size() == HotWriters::MaxEntries, "bounded entries");
	Check(!hot.Intersects(0x100000000ull, 4), "oldest entry evicted");
	Check(hot.Intersects(0x100000000ull + (HotWriters::MaxEntries + 3) * 0x10000, 4), "newest kept");
}

void TestFlushThrottle() {
	FlushThrottle throttle(1000);
	Check(!throttle.Take(1, 0), "no request, no submission");
	throttle.Request();
	Check(throttle.Take(1, 0), "first request submits");
	Check(!throttle.Requested(), "request consumed");
	// A request within the interval is put off for a later operation of the same tick.
	throttle.Request();
	Check(!throttle.Take(2, 500), "too soon");
	Check(throttle.Requested(), "request kept");
	Check(throttle.Take(2, 1200), "same tick after the interval submits");
	// A put-off request is dropped when another submission happened meanwhile.
	throttle.Request();
	Check(!throttle.Take(3, 1500), "too soon again");
	Check(!throttle.Take(4, 5000), "another submission sent the work");
	Check(!throttle.Requested(), "dropped request");
}

void TestMirrorIndex() {
	MirrorIndex mirror;
	// First read: the lines are added and queued, nothing held.
	auto found = mirror.Find(0x2010, 8, 3, 10);
	Check(!found.held, "new line not held");
	Check(mirror.Size() == 1, "line added");
	// A read across two lines adds the second.
	found = mirror.Find(0x2038, 16, 3, 10);
	Check(!found.held && mirror.Size() == 2, "second line added");
	auto copies = mirror.TakeDirty(10, [](uint64_t) { return true; });
	Check(copies.size() == 2 && copies[0].first == 0x2000 && copies[1].first == 0x2040,
	      "both lines copied once, sorted");
	Check(copies[0].second != copies[1].second, "distinct slots");
	// Copied at tick 10: a read needing tick 3 is held and waits for 10.
	found = mirror.Find(0x2038, 16, 3, 11);
	Check(found.held && found.wait == 10 && found.first == 0x2000, "held after the copy");
	Check(found.slots[0] == copies[0].second && found.slots[1] == copies[1].second, "slots");
	Check(mirror.RecentlyHit(), "a hit is recent");
	// A read needing a newer tick than the copy is not held, and queues a copy.
	found = mirror.Find(0x2000, 4, 12, 13);
	Check(!found.held && mirror.HasDirty(), "stale copy queues a new one");
	(void)mirror.TakeDirty(13, [](uint64_t) { return true; });
	// A write clears the copy and reports a recently used line.
	Check(mirror.MarkWritten(0x2040, 4, 14), "recent line written");
	Check(mirror.Get(0x2040)->tick == 0 && mirror.Get(0x2000)->tick == 13, "only the written line");
	found = mirror.Find(0x2040, 4, 0, 14);
	Check(!found.held, "written line not held until copied again");
	copies = mirror.TakeDirty(14, [](uint64_t) { return true; });
	Check(copies.size() == 1 && copies[0].first == 0x2040, "write queued one copy (deduplicated)");
	// Lines the accept function rejects keep no copy tick.
	(void)mirror.MarkWritten(0x2000, 4, 15);
	copies = mirror.TakeDirty(15, [](uint64_t) { return false; });
	Check(copies.empty() && mirror.Get(0x2000)->tick == 0, "rejected line stays uncopied");
	// Reads larger than MaxRead are refused.
	Check(!mirror.Find(0x3000, MirrorIndex::MaxRead + 1, 0, 16).held && mirror.Size() == 2,
	      "large read refused without adding lines");
	// Eviction: filling past MaxLines keeps the slots unique and bounded.
	for (uint64_t i = 0; i < MirrorIndex::MaxLines + 100; ++i) {
		(void)mirror.Find(0x10000000ull + i * MirrorIndex::LineSize, 4, 0, 20);
	}
	Check(mirror.Size() <= MirrorIndex::MaxLines, "bounded lines");
	copies = mirror.TakeDirty(20, [](uint64_t) { return true; });
	std::set<uint32_t> slots;
	for (const auto& [line, slot]: copies) {
		Check(slot < MirrorIndex::MaxLines, "slot in range");
		Check(slots.insert(slot).second, "slot unique among live lines");
	}
}

} // namespace

int main() {
	TestWriteTicks();
	TestWriteTicksRandom();
	TestHotWriters();
	TestFlushThrottle();
	TestMirrorIndex();
	std::printf("ReadbackWaitTests: all passed\n");
	return 0;
}
