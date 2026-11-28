#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERTRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERTRACE_H_

#include "common/logging/log.h"
#include "common/runtimeToggles.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// Diagnostic, log only (KYTY_TRACE_SHADERS): one line per recorded draw and dispatch with its
// guest shader hashes, to find which shaders a scene uses. It changes neither generated shaders
// nor the shader store key. Values:
//   N      every N-th GPU frame (1 = every frame)
//   A-B    every frame from A to B
//   A-B/N  every N-th frame from A to B
// The runtime toggle "trace_shaders" (KYTY_TOGGLE_FILE) also traces every frame while it reads 1.
// Lines: "TRACE f=<frame> n=<seq in frame> D <stage>=<hash>... ps=<hash> <draw arguments>" and
//        "TRACE f=<frame> n=<seq in frame> C cs=<hash> <dispatch arguments>".
namespace Libs::Graphics::ShaderTrace {

struct Range {
	uint64_t first  = 0;
	uint64_t last   = UINT64_MAX;
	uint64_t period = 0; // 0: no env trace
};

inline Range ParseRange(const char* text) {
	Range range;
	if (text == nullptr || text[0] == 0) {
		return range;
	}
	char*          end   = nullptr;
	const uint64_t first = std::strtoull(text, &end, 10);
	if (end != nullptr && *end == '-') {
		range.first = first;
		range.last  = std::strtoull(end + 1, &end, 10);
		range.period = 1;
		if (end != nullptr && *end == '/') {
			range.period = std::strtoull(end + 1, &end, 10);
		}
	} else {
		range.period = first;
	}
	if (range.period == 0 || range.last < range.first) {
		range.period = 0;
	}
	return range;
}

namespace Detail {
inline const Range& EnvRange() {
	static const Range range = ParseRange(std::getenv("KYTY_TRACE_SHADERS"));
	return range;
}
inline std::atomic<uint64_t> g_frame {UINT64_MAX};
inline std::atomic<bool>     g_active {false};
inline std::atomic<uint32_t> g_seq {0};
// Frame window of the KYTY_DUMP_CS diagnostic (renderCompute.cpp), for KYTY_TRACE_SHADERS=dump.
inline std::atomic<uint64_t> g_dump_first {0};
inline std::atomic<uint64_t> g_dump_frames {0};
inline std::atomic<uint64_t> g_dump_every {1};
inline bool TraceFollowsDump() {
	static const bool follows = [] {
		const char* text = std::getenv("KYTY_TRACE_SHADERS");
		return text != nullptr && std::strcmp(text, "dump") == 0;
	}();
	return follows;
}
} // namespace Detail

// Called by the KYTY_DUMP_CS diagnostic when it picks its first frame.
inline void SetDumpWindow(uint64_t first, uint64_t frames, uint64_t every) {
	Detail::g_dump_frames.store(frames, std::memory_order_relaxed);
	Detail::g_dump_every.store(every, std::memory_order_relaxed);
	Detail::g_dump_first.store(first, std::memory_order_relaxed);
}
inline bool InDumpWindow(uint64_t frame) {
	const uint64_t first = Detail::g_dump_first.load(std::memory_order_relaxed);
	return first != 0 && frame >= first && frame < first + Detail::g_dump_frames.load(std::memory_order_relaxed) &&
	       (frame - first) % Detail::g_dump_every.load(std::memory_order_relaxed) == 0;
}

// Diagnostic guest ranges (KYTY_DUMP_CS_ADDR=<hex addr>[+<hex size>][,...]): dispatches binding a
// buffer that overlaps one are dumped, traced draws binding one print their stages' resources.
struct WatchRange {
	uint64_t begin = 0;
	uint64_t end   = 0;
};
inline const std::vector<WatchRange>& WatchRanges() {
	static const std::vector<WatchRange> ranges = [] {
		std::vector<WatchRange> list;
		const char*             text = std::getenv("KYTY_DUMP_CS_ADDR");
		while (text != nullptr && *text != 0) {
			char*          end  = nullptr;
			const uint64_t addr = std::strtoull(text, &end, 16);
			if (end == text) break;
			uint64_t size = 1;
			if (*end == '+') {
				text = end + 1;
				size = std::strtoull(text, &end, 16);
			}
			list.push_back({addr, addr + std::max<uint64_t>(size, 1)});
			text = *end == ',' ? end + 1 : end;
		}
		return list;
	}();
	return ranges;
}
inline bool Watched(uint64_t address, uint64_t size) {
	for (const auto& r: WatchRanges()) {
		if (address < r.end && r.begin < address + size) return true;
	}
	return false;
}

// Whether this frame is traced; the toggle is read once per frame (Get() takes a mutex).
inline bool Active(uint64_t frame) {
	const auto& range = Detail::EnvRange();
	if (range.period == 0 && !Detail::TraceFollowsDump() && !Common::RuntimeToggles::Enabled()) {
		return false;
	}
	if (Detail::g_frame.load(std::memory_order_relaxed) != frame) {
		bool active = range.period != 0 && frame >= range.first && frame <= range.last &&
		              (frame - range.first) % range.period == 0;
		active      = active || (Detail::TraceFollowsDump() && InDumpWindow(frame));
		active      = active || Common::RuntimeToggles::Get("trace_shaders");
		Detail::g_active.store(active, std::memory_order_relaxed);
		Detail::g_seq.store(0, std::memory_order_relaxed);
		Detail::g_frame.store(frame, std::memory_order_relaxed);
	}
	return Detail::g_active.load(std::memory_order_relaxed);
}

inline uint32_t NextSeq() {
	return Detail::g_seq.fetch_add(1, std::memory_order_relaxed);
}

inline const char* StageName(int stage) {
	// ShaderType order: Unknown, Vertex, Pixel, Compute, Mesh, Local, Hull, Domain
	static constexpr const char* names[] = {"xs", "vs", "ps", "cs", "ms", "ls", "hs", "ds"};
	return stage >= 0 && stage < 8 ? names[stage] : "xs";
}

} // namespace Libs::Graphics::ShaderTrace

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERTRACE_H_
