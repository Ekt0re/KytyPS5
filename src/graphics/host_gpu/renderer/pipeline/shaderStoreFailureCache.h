#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREFAILURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREFAILURECACHE_H_

#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

// A translator capability generation. Failed translations are stored in the persistent shader
// store as negative entries so a new run does not retry them on every boot. Bump this when a
// translator fix makes earlier rejections translatable: a failure recorded by an older generation
// no longer applies and is retried once. It travels with the negative record, not with the store
// header, so positive modules and driver pipelines keep being reused.
inline constexpr uint64_t kTranslatorCapabilityVersion = 1;

// Internal experiment switch: KYTY_SHADER_STORE_RETRY_FAILED=1 ignores negative records for one
// run. Without it the negative cache is still opt-in as a whole: the default path is unchanged.
[[nodiscard]] inline bool ShaderStoreNegativeCacheEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_SHADER_STORE_NEGATIVE_CACHE");
		return env != nullptr && env[0] == '1';
	}();
	return enabled;
}

[[nodiscard]] inline constexpr bool FailureCacheApplies(uint64_t stored_version) {
	return stored_version == kTranslatorCapabilityVersion;
}

// A failure with no message still needs a readable reason in the store and in the report.
[[nodiscard]] inline std::string NormalizeFailureReason(std::string_view reason) {
	return reason.empty() ? std::string {"unspecified failure"} : std::string {reason};
}

// The key and payload of one negative record. The key fields are the ones PipelineCache::ProgramKey
// holds, written in the same order as the positive module records.
struct FailureRecord {
	ShaderType           stage            = ShaderType::Unknown;
	uint64_t             hash             = 0;
	uint32_t             user_data_count  = 0;
	std::vector<uint32_t> static_state;
	uint64_t             capability_version = 0;
	std::string          reason;
};

template <class S> void IoFailureRecord(S& io, FailureRecord& record) {
	IoAll(io, record.stage, record.hash, record.user_data_count, record.static_state,
	      record.capability_version, record.reason);
}

[[nodiscard]] inline std::vector<uint8_t> SaveFailureRecord(const FailureRecord& failure) {
	using ShaderRecompiler::IR::Writer;
	Writer       io;
	FailureRecord copy = failure;
	IoFailureRecord(io, copy);
	return std::move(io.bytes);
}

[[nodiscard]] inline bool LoadFailureRecord(std::span<const uint8_t> bytes, FailureRecord& failure) {
	using ShaderRecompiler::IR::Reader;
	Reader io {bytes};
	IoFailureRecord(io, failure);
	return io.ok && io.bytes.empty();
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTOREFAILURECACHE_H_
