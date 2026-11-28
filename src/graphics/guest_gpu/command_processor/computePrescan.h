#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMPUTE_PRESCAN_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMPUTE_PRESCAN_H

#include "graphics/guest_gpu/hardwareContext.h"

#include <cstdint>
#include <span>

namespace Libs::Graphics {

// Applies a compute SH register write; false for a register the command processor does not know.
bool DecodeCsRegister(HW::CsStageRegisters& cs_regs, uint32_t cmd_offset, uint32_t value);

// Finds the compute dispatches of a command buffer before the command processor runs it, so that
// their programs can be translated ahead. It only reads: what lies outside `commands` is copied
// through checked reads (call it on a guest thread, which handles faults on GPU-written memory
// like the guest's own reads), and packets it does not follow (branches, predication, packets that
// change registers from GPU memory) are stepped over, so it may miss or misjudge a dispatch. It
// starts from and leaves the registers in `regs`, and tells `sink` each dispatch's registers and
// initiator.
class CommandProcessor;

class ComputePrescanSink {
public:
	virtual ~ComputePrescanSink()                                                   = default;
	virtual void Found(const HW::CsStageRegisters& regs, uint32_t dispatch_initiator) = 0;
	// The same dispatch with its arguments: DISPATCH_DIRECT's dimensions, or (groups null) the
	// address of an indirect dispatch's arguments, 0 when the scan does not know it.
	virtual void FoundDispatch(const HW::CsStageRegisters& regs, uint32_t dispatch_initiator,
	                           const uint32_t* /*groups*/, uint64_t /*indirect_args*/) {
		Found(regs, dispatch_initiator);
	}
	// With a `shadow` processor: a draw, with the graphics registers the shadow holds there.
	virtual void FoundDraw(CommandProcessor& /*shadow*/, bool /*indirect*/) {}
};
// `shadow`, if given, is a command processor that never runs commands: register packets are
// applied to its registers with the command processor's own handlers (graphics translate-ahead).
void PrescanComputeDispatches(std::span<const uint32_t> commands, HW::CsStageRegisters& regs,
                              ComputePrescanSink& sink, CommandProcessor* shadow = nullptr);

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMPUTE_PRESCAN_H
