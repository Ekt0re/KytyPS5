#ifndef EMULATOR_SRC_GRAPHICS_FRAMECAPTURE_H_
#define EMULATOR_SRC_GRAPHICS_FRAMECAPTURE_H_

#include <cstdint>
#include <filesystem>
#include <span>

// Frame capture for offline replay (after AnyPS5 3679fa68, 33358ce8, ea03d75c; simnJS/AnyPS5,
// GPL-2.0, re-implemented for Kyty's kernel and GPU front end).
//
// Capture (KYTY_FRAME_CAPTURE=<dir>): after KYTY_FRAME_CAPTURE_AT_FLIP flips (default 1) the next
// guest GPU call snapshots every committed guest range, and the calls of the next
// KYTY_FRAME_CAPTURE_FRAMES flips (default 3) are recorded: command buffer submissions, rewind
// releases, suspend points and video out calls, each preceded by the guest memory pages the CPU
// changed since the previous call, and the GPU progress the guest had seen. Video out set-up calls
// made before the capture are kept and written first.
//
// Replay (kyty_emulator --replay <dir>): no game and no lock. The ranges are mapped again at their
// addresses, filled from the snapshot, and the calls issued in their order, each one once the GPU
// has completed as many submissions as it had when the guest made it. Frame times and the final
// display image (PNG) are written to --replay-out.
namespace Libs::Graphics {
struct ShaderMappedData;
} // namespace Libs::Graphics

namespace Libs::Graphics::FrameCapture {

enum class VideoOutCallKind : uint32_t {
	Open = 1,
	Close,
	RegisterBuffers2,
	SubmitChangeBufferAttribute2,
	UnregisterBuffers,
	SetFlipRate,
	SetFlipMaster,
	SubmitFlip,
};

[[nodiscard]] bool Enabled() noexcept;
// This process replays a capture (RunReplay was called).
[[nodiscard]] bool Replaying() noexcept;

// Guest GPU calls; queue is the AGC queue (0 graphics, 0x20.. compute).
void Submit(uint32_t queue, const uint32_t* commands, uint32_t dwords);
void ReleaseRewind(const uint32_t* rewind);
void SuspendPoint();
void VideoOutCall(VideoOutCallKind kind, std::span<const int64_t> args,
                  std::span<const uint8_t> blob = {});
// The guest created a shader (sceAgcCreateShader): the replay maps it the same way.
void ShaderMapped(uint64_t address, const ShaderMappedData& data);
// A flip was submitted by the CPU or a GPU end-of-pipe packet: a frame boundary.
void NoteFlip() noexcept;

struct ReplayOptions {
	std::filesystem::path capture;
	std::filesystem::path output;
	uint32_t              repeat = 1;
};

// Runs on a thread of its own while the main thread runs the window loop; exits the process.
void RunReplay(const ReplayOptions& options);

} // namespace Libs::Graphics::FrameCapture

#endif // EMULATOR_SRC_GRAPHICS_FRAMECAPTURE_H_
