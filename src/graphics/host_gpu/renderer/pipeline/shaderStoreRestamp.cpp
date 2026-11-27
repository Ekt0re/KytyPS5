#include "graphics/host_gpu/renderer/pipeline/shaderStoreRestamp.h"

namespace Libs::Graphics {

RestampSelection SelectRestampRecords(std::span<const RestampRecord>      records,
                                      const std::unordered_set<uint64_t>& identical,
                                      const std::unordered_set<uint64_t>& plan_identical) {
	RestampSelection selection;
	selection.keep.assign(records.size(), false);
	// The modules the new store holds; a pipeline may only name those.
	std::unordered_set<uint64_t> kept;
	std::unordered_set<uint64_t> seen;
	// Programs whose plan record the new store has.
	std::unordered_set<uint64_t> planned;
	for (size_t i = 0; i < records.size(); i++) {
		const auto& record = records[i];
		if (record.type != RestampRecord::Type::Module) continue;
		if (!seen.insert(record.module_id).second) {
			selection.modules_duplicate++;
			continue;
		}
		if (record.module_id == 0 || !identical.contains(record.module_id)) {
			selection.modules_dropped++;
		} else if (!planned.contains(record.program) && !plan_identical.contains(record.module_id)) {
			selection.modules_plan_dropped++;
		} else {
			planned.insert(record.program);
			selection.keep[i] = true;
			kept.insert(record.module_id);
			selection.modules_kept++;
		}
	}
	for (size_t i = 0; i < records.size(); i++) {
		const auto& record = records[i];
		switch (record.type) {
			case RestampRecord::Type::Module: break;
			case RestampRecord::Type::GraphicsPipeline: {
				bool complete = true;
				for (const auto id: record.stage_ids) {
					if (id != 0 && !kept.contains(id)) complete = false;
				}
				selection.keep[i] = complete;
				(complete ? selection.pipelines_kept : selection.pipelines_dropped)++;
				break;
			}
			case RestampRecord::Type::TessellationLayout: selection.layouts_dropped++; break;
		}
	}
	return selection;
}

} // namespace Libs::Graphics
