#include "graphics/host_gpu/renderer/pipeline/pipelineStats.h"

#include "graphics/host_gpu/graphicContext.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <vector>

namespace Libs::Graphics::PipelineStats {

namespace {

const char* Setting() noexcept {
	const char* value = std::getenv("KYTY_PIPELINE_STATS");
	return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0') ? value
	                                                                                      : nullptr;
}

// CSV fields: quoted when they hold a separator, a quote or a line break.
std::string Field(std::string_view text) {
	if (text.find_first_of(",\"\n\r") == std::string_view::npos) return std::string(text);
	std::string quoted = "\"";
	for (const char c: text) {
		if (c == '"') quoted += '"';
		quoted += (c == '\n' || c == '\r') ? ' ' : c;
	}
	return quoted + '"';
}

std::string Value(const vk::PipelineExecutableStatisticKHR& statistic) {
	switch (statistic.format) {
		case vk::PipelineExecutableStatisticFormatKHR::eBool32:
			return statistic.value.b32 != VK_FALSE ? "1" : "0";
		case vk::PipelineExecutableStatisticFormatKHR::eInt64:
			return std::to_string(statistic.value.i64);
		case vk::PipelineExecutableStatisticFormatKHR::eUint64:
			return std::to_string(statistic.value.u64);
		case vk::PipelineExecutableStatisticFormatKHR::eFloat64:
			return fmt::format("{:g}", statistic.value.f64);
	}
	return "";
}

struct Output {
	std::mutex mutex;
	FILE*      file   = nullptr;
	bool       failed = false;
};

Output& Out() {
	static Output output;
	return output;
}

} // namespace

bool Requested() noexcept {
	return Setting() != nullptr;
}

void Record(const GraphicContext& graphics, vk::Pipeline pipeline, std::string_view name) {
	if (!graphics.pipeline_statistics_enabled || pipeline == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineExecutablePropertiesKHR == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkGetPipelineExecutableStatisticsKHR == nullptr) {
		return;
	}
	const vk::PipelineInfoKHR info {.pipeline = pipeline};
	uint32_t                  count = 0;
	if (graphics.device.getPipelineExecutablePropertiesKHR(&info, &count, nullptr) !=
	    vk::Result::eSuccess) {
		return;
	}
	std::vector<vk::PipelineExecutablePropertiesKHR> executables(count);
	if (graphics.device.getPipelineExecutablePropertiesKHR(&info, &count, executables.data()) !=
	    vk::Result::eSuccess) {
		return;
	}
	std::string rows;
	const auto  pipeline_field = Field(name.empty() ? std::string_view {"host"} : name);
	for (uint32_t index = 0; index < count; index++) {
		const vk::PipelineExecutableInfoKHR executable {.pipeline        = pipeline,
		                                                .executableIndex = index};
		uint32_t                            statistics_count = 0;
		if (graphics.device.getPipelineExecutableStatisticsKHR(&executable, &statistics_count,
		                                                       nullptr) != vk::Result::eSuccess) {
			continue;
		}
		std::vector<vk::PipelineExecutableStatisticKHR> statistics(statistics_count);
		if (graphics.device.getPipelineExecutableStatisticsKHR(
		        &executable, &statistics_count, statistics.data()) != vk::Result::eSuccess) {
			continue;
		}
		const auto& properties = executables[index];
		const auto  prefix =
		    fmt::format("{},{},0x{:x},{},", pipeline_field, Field(properties.name.data()),
		                static_cast<vk::ShaderStageFlags::MaskType>(properties.stages),
		                properties.subgroupSize);
		for (uint32_t s = 0; s < statistics_count; s++) {
			rows += prefix + Field(statistics[s].name.data()) + "," + Value(statistics[s]) + "\n";
		}
	}
	auto&           out = Out();
	std::lock_guard lock(out.mutex);
	if (out.failed) return;
	if (out.file == nullptr) {
		const std::string setting = Setting();
		const std::string path    = setting == "1" ? "kyty_pipeline_stats.csv" : setting;
		out.file                  = std::fopen(path.c_str(), "w");
		if (out.file == nullptr) {
			std::fprintf(stderr, "KYTY_PIPELINE_STATS: cannot write %s\n", path.c_str());
			out.failed = true;
			return;
		}
		std::fprintf(stderr, "KYTY_PIPELINE_STATS: writing %s\n", path.c_str());
		std::fputs("pipeline,executable,stages,subgroup_size,statistic,value\n", out.file);
	}
	std::fputs(rows.c_str(), out.file);
	std::fflush(out.file);
}

} // namespace Libs::Graphics::PipelineStats
