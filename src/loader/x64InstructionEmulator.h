#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <cstddef>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool TryEmulate(void* native_context);
// Completes a faulting load into EAX with `value` and resumes after it.
void CompleteLoad(void* native_context, size_t length, uint32_t value);
// Resumes after the faulting instruction of `length` bytes without executing it.
void SkipInstruction(void* native_context, size_t length);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
