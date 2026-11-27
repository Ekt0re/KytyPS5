#include "graphics/host_gpu/renderer/cache/readbackBatch.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::ReadbackBatch::AccessWindow;
using Libs::Graphics::ReadbackBatch::ChooseReadbackPath;
using Libs::Graphics::ReadbackBatch::ReadbackPath;
using Libs::Graphics::ReadbackBatch::NeighbourHistory;
using Libs::Graphics::ReadbackBatch::Range;
using Libs::Graphics::ReadbackBatch::SplitByOwner;

constexpr uint64_t KiB    = 1024;
constexpr uint64_t Window = 512 * KiB;
constexpr uint64_t Page   = 16 * KiB;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "ReadbackBatchTests: failed: %s\n", message);
		std::abort();
	}
}

void TestAccessWindow() {
	const Range all {0, UINT64_MAX};
	Check(AccessWindow(0x10'0103'4567, 1, Window, 0, all) == Range {0x10'0100'0000, 0x10'0108'0000},
	      "an access downloads its aligned window");
	Check(AccessWindow(0x10'0107'ff00, 0x200, Window, 0, all) ==
	          Range {0x10'0100'0000, 0x10'0108'0100},
	      "an access past the window's end extends it");
	Check(AccessWindow(0x10'0100'0000, 4, Window, 64 * KiB, all) ==
	          Range {0x10'00ff'0000, 0x10'0109'0000},
	      "the prefetch widens both sides");
	Check(AccessWindow(0x1000, 4, Window, Window, all) == Range {0, 2 * Window},
	      "the prefetch stops at address 0");
	Check(AccessWindow(0x10'0100'0000, 4, Window, 64 * KiB, {0x10'0100'4000, 0x10'0102'0000}) ==
	          Range {0x10'0100'4000, 0x10'0102'0000},
	      "the window is clamped to the limit");
}

struct FakeBuffer {
	int   id;
	Range range;
};

std::optional<std::pair<int, Range>> Find(const std::vector<FakeBuffer>& buffers, uint64_t address,
                                          int& lookups) {
	lookups++;
	for (const auto& buffer: buffers) {
		if (address >= buffer.range.begin && address < buffer.range.end) {
			return std::pair {buffer.id, buffer.range};
		}
	}
	return std::nullopt;
}

void TestSplitOneBuffer() {
	// The access's buffer covers the whole window: one part, the one the caller chose.
	const std::vector<FakeBuffer> buffers {{1, {0, 4 * Window}}};
	int        lookups = 0;
	const auto parts   = SplitByOwner<int>({Window, 2 * Window}, Page, 1, Range {Window, 2 * Window},
	                                       [&](uint64_t a) { return Find(buffers, a, lookups); });
	Check(parts.size() == 1 && parts[0].id == 1 && parts[0].range == Range {Window, 2 * Window},
	      "one buffer gives one part");
	Check(lookups == 1, "a buffer covering the window is looked up once");
}

void TestSplitSeveralBuffers() {
	// Three buffers and a hole in one window; the access is in buffer 2.
	const std::vector<FakeBuffer> buffers {
	    {1, {0, Window + 64 * KiB}},
	    {2, {Window + 64 * KiB, Window + 256 * KiB}},
	    {3, {Window + 320 * KiB, 4 * Window}},
	};
	int        lookups = 0;
	const auto parts =
	    SplitByOwner<int>({Window, 2 * Window}, Page, 2, Range {Window + 64 * KiB, Window + 256 * KiB},
	                      [&](uint64_t a) { return Find(buffers, a, lookups); });
	Check(parts.size() == 3, "three buffers give three parts");
	Check(parts[0].id == 2 && parts[0].range == Range {Window + 64 * KiB, Window + 256 * KiB},
	      "the access's buffer comes first with its own part");
	Check(parts[1].id == 1 && parts[1].range == Range {Window, Window + 64 * KiB},
	      "a buffer starting before the window is clamped to it");
	Check(parts[2].id == 3 && parts[2].range == Range {Window + 320 * KiB, 2 * Window},
	      "a buffer ending after the window is clamped to it");
	// Buffer 1 (once), buffer 2 (once), 4 pages of the hole, buffer 3 (once).
	Check(lookups == 7, "every buffer is looked up once and holes once per page");
}

void TestSplitUnalignedBuffers() {
	// Buffers that start inside a page are still found, and none is visited twice.
	const std::vector<FakeBuffer> buffers {
	    {1, {Window, Window + 0x1100}},
	    {2, {Window + 0x1100, Window + 0x9000}},
	};
	int        lookups = 0;
	const auto parts   = SplitByOwner<int>({Window, Window + 0x9000}, Page, 1, Range {Window, Window + 0x1100},
	                                       [&](uint64_t a) { return Find(buffers, a, lookups); });
	Check(parts.size() == 2 && parts[0].id == 1 && parts[1].id == 2 &&
	          parts[1].range == Range {Window + 0x1100, Window + 0x9000},
	      "a buffer starting inside a page is found on its next page");
	Check(lookups == 2, "each buffer is looked up once");
}

void TestSplitEmptyFirst() {
	const std::vector<FakeBuffer> buffers {{5, {Window, 2 * Window}}};
	int        lookups = 0;
	const auto parts   = SplitByOwner<int>({Window, 2 * Window}, Page, 9, Range {},
	                                       [&](uint64_t a) { return Find(buffers, a, lookups); });
	Check(parts.size() == 1 && parts[0].id == 5, "an empty first part is left out");
}

void TestNeighbourHistory() {
	NeighbourHistory history;
	Check(!history.Note(10 * Window, Window), "the first window has no neighbour");
	Check(history.Note(11 * Window, Window), "the next window is a neighbour");
	Check(history.Note(9 * Window, Window), "the previous window is a neighbour");
	Check(!history.Note(20 * Window, Window), "a distant window is not");
	Check(!history.Note(20 * Window, Window), "the same window again is not a neighbour");
	for (uint64_t i = 0; i < 8; i++) {
		(void)history.Note((100 + 2 * i) * Window, Window);
	}
	Check(history.Note(101 * Window, Window), "a window between two recent ones is a neighbour");
	Check(!history.Note(21 * Window, Window), "only the last 8 windows count");
}

} // namespace

void TestReadbackPath() {
	// detached, copy_queue, fuse
	static_assert(ChooseReadbackPath({false, false, false}) == ReadbackPath::Flush);
	Check(ChooseReadbackPath({.detached = false, .copy_queue = true, .fuse = true}) == ReadbackPath::Flush,
	      "a producer in the open command buffer flushes and waits, copy queue or not");
	Check(ChooseReadbackPath({.detached = false, .copy_queue = false, .fuse = true}) == ReadbackPath::Flush,
	      "a producer in the open command buffer is never fused");
	Check(ChooseReadbackPath({.detached = true, .copy_queue = true, .fuse = true}) == ReadbackPath::CopyQueue,
	      "a finished or submitted producer copies on the copy queue when it is on");
	Check(ChooseReadbackPath({.detached = true, .copy_queue = true, .fuse = false}) == ReadbackPath::CopyQueue,
	      "unbatched downloads use the copy queue too");
	Check(ChooseReadbackPath({.detached = true, .copy_queue = false, .fuse = true}) == ReadbackPath::Fused,
	      "without the copy queue a batched detached download is fused");
	Check(ChooseReadbackPath({.detached = true, .copy_queue = false, .fuse = false}) == ReadbackPath::Detached,
	      "without the copy queue and fusion it is a detached submission");
}

int main() {
	TestReadbackPath();
	TestAccessWindow();
	TestSplitOneBuffer();
	TestSplitSeveralBuffers();
	TestSplitUnalignedBuffers();
	TestSplitEmptyFirst();
	TestNeighbourHistory();
	std::printf("ReadbackBatchTests: all passed\n");
	return 0;
}
