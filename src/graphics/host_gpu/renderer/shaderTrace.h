#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERTRACE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERTRACE_H_

#include "common/logging/log.h"
#include "common/runtimeToggles.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

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
} // namespace Detail

// Whether this frame is traced; the toggle is read once per frame (Get() takes a mutex).
inline bool Active(uint64_t frame) {
	const auto& range = Detail::EnvRange();
	if (range.period == 0 && !Common::RuntimeToggles::Enabled()) {
		return false;
	}
	if (Detail::g_frame.load(std::memory_order_relaxed) != frame) {
		bool active = range.period != 0 && frame >= range.first && frame <= range.last &&
		              (frame - range.first) % range.period == 0;
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
