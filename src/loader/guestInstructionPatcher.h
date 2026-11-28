// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#ifndef KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_
#define KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_

#include "common/common.h"

#include <span>
#include <utility>
#include <vector>

namespace Loader {

struct GuestInstructionHostFeatures {
	bool sse4a = false;
	bool rdpid = false;
	bool clwb  = false;
};

GuestInstructionHostFeatures GetGuestInstructionHostFeatures();

struct InstructionPatchCounts {
	uint64_t found   = 0;
	uint64_t native  = 0;
	uint64_t trapped = 0;

	uint64_t Skipped() const { return found - native - trapped; }

	InstructionPatchCounts& operator+=(const InstructionPatchCounts& other) {
		found += other.found;
		native += other.native;
		trapped += other.trapped;
		return *this;
	}
};

struct GuestInstructionPatchResult {
	uint64_t               function_count                           = 0;
	// Functions found through the calls and address loads of the listed ones.
	uint64_t               called_function_count                    = 0;
	// Of those, the ones found only through a RIP-relative LEA.
	uint64_t               lea_function_count                       = 0;
	// Found starts that lie strictly inside the range of a function .eh_frame lists: they split a
	// real function in two (or are not instruction boundaries at all).
	uint64_t               called_inside_listed_count               = 0;
	uint64_t               trampoline_bytes                         = 0;
	uint64_t               instruction_count                        = 0;
	uint64_t               red_zone_function_count                  = 0;
	uint64_t               memory_instruction_count                 = 0;
	uint64_t               short_memory_instruction_count           = 0;
	uint64_t               patched_memory_instruction_count         = 0;
	uint64_t               stack_dependent_memory_instruction_count = 0;
	uint64_t               control_flow_memory_instruction_count    = 0;
	uint64_t               unrelocatable_memory_instruction_count   = 0;
	uint64_t               indirect_red_zone_function_count         = 0;
	InstructionPatchCounts reciprocal_sqrt;
	InstructionPatchCounts extrq;
	InstructionPatchCounts insertq;
	InstructionPatchCounts rdpid;
	InstructionPatchCounts clwb;
	InstructionPatchCounts rdtsc;
};

void RegisterGuestInstructionPatchModule(void* module_ptr, uint64_t module_size,
                                         void* trampoline_area_ptr, uint64_t trampoline_area_size);
void UnregisterGuestInstructionPatchModule(void* module_ptr);

// Apply enabled instruction fixes using native trampolines or safe trap fallbacks. A nonzero
// tsc_multiplier rewrites RDTSC/RDTSCP to return host TSC * tsc_multiplier >> 32.
GuestInstructionPatchResult PatchGuestInstructions(
    uint64_t segment_addr, uint64_t segment_size, std::span<const uintptr_t> function_starts,
    bool protect_memory, bool emulate_amd,
    GuestInstructionHostFeatures host_features  = GetGuestInstructionHostFeatures(),
    uint64_t                     tsc_multiplier = 0,
    std::span<const std::pair<uintptr_t, uintptr_t>> listed_ranges = {});

bool DecodeEhFrameFunctionStarts(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 std::vector<uintptr_t>* function_starts);

// The [start, end) range of every function .eh_frame lists, sorted by start. The unwind records
// are read only inside [image_addr, image_addr + image_size); a record outside it is left out.
// PatchGuestInstructions() uses the ranges to drop discovered starts inside a listed function.
bool DecodeEhFrameFunctionRanges(uint64_t eh_frame_header_addr, uint64_t eh_frame_header_size,
                                 uint64_t image_addr, uint64_t image_size,
                                 std::vector<std::pair<uintptr_t, uintptr_t>>* ranges);

// Whether address lies strictly inside one of the sorted ranges (not at its start).
bool IsInsideListedFunction(std::span<const std::pair<uintptr_t, uintptr_t>> ranges,
                            uintptr_t                                         address);

} // namespace Loader

#endif /* KYTY_LOADER_GUEST_INSTRUCTION_PATCHER_H_ */
