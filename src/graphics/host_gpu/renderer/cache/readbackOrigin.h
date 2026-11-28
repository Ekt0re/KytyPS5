#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKORIGIN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKORIGIN_H_

#include <cstdint>

// Who asked for a buffer readback, for the per-origin counters of KYTY_PERF_SUMMARY (rb_o_*).
// A scope on the asking thread names it; an inner scope only fills in an unnamed origin, so the
// most specific caller (an SRT walker load, indirect arguments) wins over the fault handler.
namespace Libs::Graphics::ReadbackOrigin {

enum class Origin : uint8_t {
	Other,    // Unnamed: unmaps and the rest.
	Guest,    // A guest thread: page faults and reads of GPU-written memory.
	Srt,      // SRT walker loads on the command thread.
	Pm4,      // Other command-thread read faults (the PM4 parser reads the command stream).
	CmdWrite, // Command-thread write faults (WriteData packets, labels through the mapping).
	Args,     // Indirect dispatch and draw arguments.
	Shader,   // Command-thread backing reads (ReadGuestMemory, shader constants).
	Texture,  // Color metadata and image heap reads.
	Count
};

namespace Detail {
inline thread_local Origin g_origin = Origin::Other;
} // namespace Detail

[[nodiscard]] inline Origin Current() noexcept {
	return Detail::g_origin;
}

class Scope {
public:
	// `fill`: only when no outer scope named the origin.
	explicit Scope(Origin origin, bool fill = false) noexcept: m_saved(Detail::g_origin) {
		if (!fill || m_saved == Origin::Other) {
			Detail::g_origin = origin;
		}
	}
	~Scope() { Detail::g_origin = m_saved; }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Origin m_saved;
};

} // namespace Libs::Graphics::ReadbackOrigin

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READBACKORIGIN_H_
