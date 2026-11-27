#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_RAW_COMMAND_BUFFER_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_RAW_COMMAND_BUFFER_H

#include <algorithm>
#include <cstdint>
#include <span>

// Raw command-buffer decoding (KYTY_RAW_CMDBUF, opt-in).
//
// The PM4 parser reads the guest's command buffers. Where a writable GPU binding covers them, the
// resource tracker marks those pages GPU-written and read-protects them, so every parser read
// faults and copies a 512 KiB window back from the host GPU, although the GPU does not change
// command memory (GTA VI: 89 such readbacks per frame, 0.0% of 19 416 changed a byte).
//
// With the mode on, the parser decodes through the backing alias of the span instead (the same
// physical memory, never protected), as shadGT reads command memory without its readback path.
// Guest addresses never come from the alias: handlers only take addresses from packet values, and
// the cursor keeps the guest span for everything else. CPU stores (guest threads, WRITE_DATA,
// RELEASE_MEM labels) land in the same memory and are seen at once. A DMA_DATA or COPY_DATA that
// the host GPU executes into a live span is published first (WritesLiveCommands): the parser then
// reads what the old fault would have read back. A shader store into a live command buffer stays
// invisible, as in shadGT; the verify mode counts spans where that would matter.
namespace Libs::Graphics::RawCommandBuffer {

enum class Mode : uint8_t { Off, On, Verify };

// KYTY_RAW_CMDBUF: unset or 0 = Off, 1 = On, "verify" = On plus a precise read of every
// KYTY_RAW_CMDBUF_VERIFY_EVERY-th span (default 1) compared with the view.
[[nodiscard]] Mode EnvMode();
// The mode for the next submission: EnvMode(), and with KYTY_TOGGLE_FILE set the runtime toggle
// raw_cmdbuf switches it off (0) and on (1), for an A/B inside one run. Takes the toggle mutex:
// call it once per submission.
[[nodiscard]] Mode SubmissionMode();

// The span in the backing alias, or nullptr when the span is not one contiguous run of backing
// (the caller then decodes through the guest address as before).
[[nodiscard]] const uint32_t* View(std::span<const uint32_t> guest);

// Compares a view with a precise read of the guest span (which faults and reads back GPU-written
// pages as the old parser did). The view is copied before the precise read, because the readback
// publishes into the same memory. Returns the number of DWORDs that differ.
[[nodiscard]] uint64_t Verify(std::span<const uint32_t> guest, const uint32_t* view);

struct LiveRange {
	uint64_t begin = 0; // Guest address of the next packet the parser will decode.
	uint64_t end   = 0;
};

// Calls refresh(begin, size) for the part of [dst, dst + size) that each live range overlaps.
// Returns how many overlaps there were.
template <typename Refresh>
uint32_t ForEachLiveOverlap(std::span<const LiveRange> live, uint64_t dst, uint64_t size,
                            Refresh&& refresh) {
	uint32_t overlaps = 0;
	if (size == 0 || UINT64_MAX - dst < size) {
		return 0;
	}
	for (const auto& range: live) {
		const auto begin = std::max(range.begin, dst);
		const auto end   = std::min(range.end, dst + size);
		if (begin < end) {
			refresh(begin, end - begin);
			overlaps++;
		}
	}
	return overlaps;
}

// Statistics, printed as "RawCmdbuf ..." every 10 s while the mode is not Off in the environment.
enum class Stat : uint8_t {
	Spans,         // Cursors decoded with the mode on.
	Raw,           // ... through the backing alias.
	Fallback,      // ... through the guest address (not one backing run).
	Verified,      // Spans compared with a precise read.
	Mismatched,    // ... that differed.
	MismatchDw,    // DWORDs that differed.
	LiveRefreshes, // DMA writes into a live span published before decoding on.
	Count
};
void Add(Stat stat, uint64_t value = 1);
void MaybeReport();

} // namespace Libs::Graphics::RawCommandBuffer

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_RAW_COMMAND_BUFFER_H
