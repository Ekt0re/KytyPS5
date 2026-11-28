#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BUFFERPROBE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BUFFERPROBE_H_

// Pure helpers of the KYTY_BUFFER_PROBE diagnostic: compare guest memory bytes with the bytes a
// dispatch sees, classify the cache's dirty state and format the log lines. Kept free of Vulkan so
// that a focused unit test can run them without a device.

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace Libs::Graphics::BufferProbe {

inline constexpr uint64_t PageBytes         = 4096;
inline constexpr uint32_t PreviewWords      = 8;
inline constexpr uint32_t MaxMismatchPrints = 40;
inline constexpr uint32_t SummaryInterval   = 500;

struct Comparison {
	uint64_t compared_bytes   = 0;
	uint64_t first_diff       = UINT64_MAX; // Byte offset of the first differing byte.
	uint64_t differing_pages  = 0;          // 4 KiB pages (by offset) with at least one difference.
	uint64_t differing_bytes  = 0;
	uint64_t total_pages      = 0;
	uint32_t guest_head[PreviewWords] {};
	uint32_t host_head[PreviewWords] {};
	// Eight words starting at the dword holding the first difference.
	uint32_t guest_at_diff[PreviewWords] {};
	uint32_t host_at_diff[PreviewWords] {};

	[[nodiscard]] bool Differs() const noexcept { return first_diff != UINT64_MAX; }
};

namespace Detail {
inline void ReadWords(const uint8_t* bytes, uint64_t size, uint64_t offset, uint32_t* out) {
	for (uint32_t i = 0; i < PreviewWords; i++) {
		out[i]               = 0;
		const uint64_t begin = offset + uint64_t {i} * 4u;
		if (begin >= size) continue;
		uint8_t word[4] {};
		std::memcpy(word, bytes + begin, static_cast<size_t>(std::min<uint64_t>(4u, size - begin)));
		std::memcpy(&out[i], word, 4);
	}
}
} // namespace Detail

// Both sides hold `size` bytes. Differences are counted per 4 KiB page of the offset, not of the
// guest address.
[[nodiscard]] inline Comparison Compare(const uint8_t* guest, const uint8_t* host, uint64_t size) {
	Comparison result;
	result.compared_bytes = size;
	result.total_pages    = (size + PageBytes - 1u) / PageBytes;
	Detail::ReadWords(guest, size, 0, result.guest_head);
	Detail::ReadWords(host, size, 0, result.host_head);
	for (uint64_t page = 0; page < result.total_pages; page++) {
		const uint64_t begin = page * PageBytes;
		const uint64_t bytes = std::min<uint64_t>(PageBytes, size - begin);
		if (std::memcmp(guest + begin, host + begin, static_cast<size_t>(bytes)) == 0) continue;
		result.differing_pages++;
		for (uint64_t i = 0; i < bytes; i++) {
			if (guest[begin + i] == host[begin + i]) continue;
			result.differing_bytes++;
			result.first_diff = std::min(result.first_diff, begin + i);
		}
	}
	if (result.Differs()) {
		const uint64_t at = result.first_diff & ~uint64_t {3};
		Detail::ReadWords(guest, size, at, result.guest_at_diff);
		Detail::ReadWords(host, size, at, result.host_at_diff);
	}
	return result;
}

enum class CacheState : uint8_t {
	Synced,   // Neither side is marked: the cache believes guest memory and host copy agree.
	CpuDirty, // Guest wrote since the last upload.
	GpuDirty, // The GPU wrote since the last download: the host copy is the newer one.
	Both,
};

[[nodiscard]] constexpr CacheState ClassifyCacheState(bool cpu_dirty, bool gpu_dirty) noexcept {
	if (cpu_dirty && gpu_dirty) return CacheState::Both;
	if (cpu_dirty) return CacheState::CpuDirty;
	if (gpu_dirty) return CacheState::GpuDirty;
	return CacheState::Synced;
}

[[nodiscard]] constexpr const char* CacheStateName(CacheState state) noexcept {
	switch (state) {
		case CacheState::Synced: return "synced";
		case CacheState::CpuDirty: return "cpu-dirty";
		case CacheState::GpuDirty: return "gpu-dirty";
		case CacheState::Both: return "cpu+gpu-dirty";
	}
	return "?";
}

// A difference is expected when the GPU owns newer data; every other difference means the cache
// considered the range up to date while the two sides disagree.
[[nodiscard]] constexpr bool IsUnexpectedDifference(const Comparison& comparison,
                                                    CacheState        state) noexcept {
	return comparison.Differs() && state != CacheState::GpuDirty && state != CacheState::Both;
}

// One log line per bound buffer. `state_before` is the cache state before the dispatch's own
// synchronisation uploaded anything, so a "synced" range with a difference is a missed upload.
[[nodiscard]] inline std::string FormatLine(uint32_t dispatch, uint32_t index, uint64_t guest_base,
                                            uint64_t size, bool written, CacheState state_before,
                                            bool guest_readable, const Comparison& c) {
	char first[32] = "none";
	if (c.Differs()) std::snprintf(first, sizeof(first), "0x%" PRIx64, c.first_diff);
	char text[1024];
	const int n = std::snprintf(
	    text, sizeof(text),
	    "[buffer-probe] dispatch=%u buf=%u guest=0x%012" PRIx64 " size=0x%" PRIx64
	    " written=%d cache=%s%s compared=0x%" PRIx64 " first_diff=%s pages_differ=%" PRIu64
	    "/%" PRIu64 " bytes_differ=%" PRIu64,
	    dispatch, index, guest_base, size, written ? 1 : 0, CacheStateName(state_before),
	    guest_readable ? "" : " guest=unreadable", c.compared_bytes, first, c.differing_pages,
	    c.total_pages, c.differing_bytes);
	std::string line(text, n > 0 ? std::min<size_t>(static_cast<size_t>(n), sizeof(text) - 1) : 0u);
	const auto words = [](const uint32_t* w) {
		char out[128];
		std::snprintf(out, sizeof(out), "%x %x %x %x %x %x %x %x", w[0], w[1], w[2], w[3], w[4],
		              w[5], w[6], w[7]);
		return std::string(out);
	};
	line += " guest_head=[" + words(c.guest_head) + "] host_head=[" + words(c.host_head) + "]";
	if (c.Differs()) {
		line += " guest_at_diff=[" + words(c.guest_at_diff) + "] host_at_diff=[" +
		        words(c.host_at_diff) + "]";
	}
	line += IsUnexpectedDifference(c, state_before) ? " UNEXPECTED" : "";
	return line;
}

// The first lines make a run visible even when bytes agree; only the first MaxMismatchPrints
// mismatching dispatches are printed in full, and a summary follows every SummaryInterval probes.
class Limiter {
public:
	static constexpr uint32_t AlwaysPrintedDispatches = 5;

	// Call once per probed dispatch. Returns true when this dispatch's lines should be printed.
	[[nodiscard]] bool OnDispatch(bool mismatch, bool unexpected) {
		m_dispatches++;
		if (mismatch) m_mismatches++;
		if (unexpected) m_unexpected++;
		const bool print_mismatch = mismatch && m_printed < MaxMismatchPrints && (++m_printed, true);
		return print_mismatch || m_dispatches <= AlwaysPrintedDispatches;
	}
	[[nodiscard]] bool SummaryDue() const noexcept {
		return m_dispatches != 0 && m_dispatches % SummaryInterval == 0;
	}
	[[nodiscard]] std::string Summary() const {
		char text[160];
		std::snprintf(text, sizeof(text),
		              "[buffer-probe] summary: probed_dispatches=%u mismatching=%u unexpected=%u "
		              "printed=%u",
		              m_dispatches, m_mismatches, m_unexpected, m_printed);
		return text;
	}
	[[nodiscard]] uint32_t Dispatches() const noexcept { return m_dispatches; }
	[[nodiscard]] uint32_t Mismatches() const noexcept { return m_mismatches; }
	[[nodiscard]] uint32_t Unexpected() const noexcept { return m_unexpected; }
	[[nodiscard]] uint32_t Printed() const noexcept { return m_printed; }

private:
	uint32_t m_dispatches = 0;
	uint32_t m_mismatches = 0;
	uint32_t m_unexpected = 0;
	uint32_t m_printed    = 0;
};

} // namespace Libs::Graphics::BufferProbe

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BUFFERPROBE_H_
