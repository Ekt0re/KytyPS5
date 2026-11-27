#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BDAGUARD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BDAGUARD_H_

#include <cstdint>
#include <string>
#include <string_view>

namespace Libs::Graphics {

// Diagnostic (KYTY_BDA_GUARD): shaders read host buffers through raw device addresses, which no
// driver bounds check covers; an address of a destroyed buffer, or an offset past a live one,
// faults the GPU. The guard keeps the device-address range of every live buffer that has one and
// a short history of the destroyed ones, so that
//   - an address the host hands to a shader can be checked on the CPU before the draw is
//     recorded (Check), and replaced when it is dead,
//   - a buffer that is destroyed while recorded GPU work may still read it is reported, and
//   - a GPU fault address is attributed to a live or destroyed buffer (Describe).
// KYTY_BDA_GUARD=1 checks and reports; =2 also keeps every destroyed buffer alive until the GPU
// completed all work recorded before its destruction (GPU breadcrumbs give the completion point;
// without them the buffer is kept for a fixed time).
namespace BdaGuard {

// 0 (off), 1 or 2; read once from the environment.
[[nodiscard]] int Mode();

// A buffer with a device address came to life. `guest` is its guest address, 0 for host data.
void OnCreate(uint64_t address, uint64_t size, uint64_t guest);
// The buffer is about to be destroyed. True when the guard took the handle and the allocation
// over (mode 2): the caller must not destroy them.
[[nodiscard]] bool OnDestroy(uint64_t address, void* allocator, void* buffer, void* allocation);
// The host stopped handing the address out (its page table entries were cleared): work recorded
// from now on cannot read the buffer.
void Retire(uint64_t address);
// The draw or dispatch about to be recorded reads the buffer.
void Use(uint64_t address);

// Whether [address, address + bytes) lies in one live buffer. A failure is counted and (the
// first ones) printed with `what`, the shader and binding, and the buffer the range used to
// belong to with the time since its destruction.
[[nodiscard]] bool Check(uint64_t address, uint64_t bytes, std::string_view what, uint64_t shader,
                         uint32_t binding, uint64_t guest = 0);
// Counts and prints another kind of finding (`text` without a newline).
void Report(std::string_view text);

// Prints context that is not a finding (the first ones only).
void Note(std::string_view text);

// Lines that say which buffer a faulting GPU address belongs or belonged to; empty when off.
[[nodiscard]] std::string Describe(uint64_t address, uint64_t precision);

// Releases kept buffers whose readers completed; called between draws.
void Poll();
// Destroys every kept buffer (device shutdown).
void Shutdown();

// Findings so far (failed checks, early destructions, reports).
[[nodiscard]] uint64_t Findings();

} // namespace BdaGuard
} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BDAGUARD_H_
