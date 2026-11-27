#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORERESTAMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORERESTAMP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

// Partial carry-over of a shader store to a new stamp (--restamp-shader-store,
// PipelineCache::Restamp): which records of the old store the new one keeps. Only what this
// translator was shown to produce byte for byte stays; everything else is dropped and comes back
// from the recipes like any module a store lacks.
struct RestampRecord {
	enum class Type : uint8_t { Module, GraphicsPipeline, TessellationLayout };
	Type                    type = Type::Module;
	// Module: its id, and a hash of its program key (records of one program share it).
	uint64_t                module_id = 0;
	uint64_t                program   = 0;
	// GraphicsPipeline: its stages' module ids, 0 for an absent stage.
	std::array<uint64_t, 4> stage_ids {};
};

struct RestampSelection {
	// One per record, in store order.
	std::vector<bool> keep;
	size_t            modules_kept       = 0;
	// Module records whose translation differs, failed, or that no recipe can translate again.
	size_t            modules_dropped    = 0;
	// Module records that would give their program its resource plan (the first record of a
	// program the store keeps) whose stored plan is not the one this build extracts.
	size_t            modules_plan_dropped = 0;
	// Further records of a module id an earlier record holds (the first one is the one verified).
	size_t            modules_duplicate  = 0;
	size_t            pipelines_kept     = 0;
	// Pipelines with a stage whose module was dropped.
	size_t            pipelines_dropped  = 0;
	// Tessellation layouts come from translator code (AnalyzeTessellationPrograms) and are not
	// verified, so they are all dropped; a run analyzes and stores them again on first use.
	size_t            layouts_dropped    = 0;
};

// `identical`: the module ids whose translation by this build gave the stored module exactly (SPIR-V,
// module fields, program key). `plan_identical`: those whose stored resource plan is also the one
// this build extracts. A load takes a program's plan from its first record only (later records'
// plans are never read, and a run stores the program's current plan with them, which need not be
// what a translation extracts), so only the record that becomes its program's first one in the new
// store must have an identical plan; one that does not is dropped and the next one is checked.
RestampSelection SelectRestampRecords(std::span<const RestampRecord>      records,
                                      const std::unordered_set<uint64_t>& identical,
                                      const std::unordered_set<uint64_t>& plan_identical);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORERESTAMP_H_
