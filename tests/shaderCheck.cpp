// Offline shader translation for one capsule/register dump or a whole capsule corpus.
// Corpus mode isolates every translation in a child process and never times driver pipeline
// creation; use spvbench for that measurement.
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/frontend/cfg/ShaderCFG.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <cerrno>
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;
#endif

using namespace Libs::Graphics;

namespace {

using Json = nlohmann::ordered_json;

struct Options {
	std::filesystem::path input;
	std::filesystem::path output;
	std::filesystem::path compare_log;
	uint32_t              jobs       = 1;
	uint32_t              limit      = 0;
	uint32_t              timeout_ms = 20000;
	std::string           only;
	bool                  emit_spirv = false;
	bool                  self_test  = false;
};

struct CorpusEntry {
	std::filesystem::path capsule;
	std::string           file;
	std::string           stage;
	uint64_t              hash       = 0;
	bool                  hash_valid = false;
};

struct ProgramInput {
	ShaderType                              stage          = ShaderType::Unknown;
	uint32_t                                wave_size      = 64;
	uint32_t                                user_data_base = 0;
	uint64_t                                hash           = 0;
	bool                                    capsule        = false;
	std::string                             skip;
	std::vector<uint32_t>                   user_data;
	std::vector<uint32_t>                   code;
	std::vector<uint32_t>                   back_code;
	std::unique_ptr<ShaderVertexInputInfo>  vertex;
	std::unique_ptr<ShaderPixelInputInfo>   pixel;
	std::unique_ptr<ShaderComputeInputInfo> compute;
};

constexpr uint32_t kStillActive    = 259u;
constexpr uint32_t kFatalExit      = 321u;
constexpr uint32_t kFatalExitPosix = kFatalExit & 0xffu;

std::string g_self;

double Milliseconds(std::chrono::steady_clock::duration duration) {
	return std::chrono::duration<double, std::milli>(duration).count();
}

std::string StageSlug(ShaderType stage) {
	const auto* name = ShaderRecompiler::StageName(stage);
	std::string text = name == nullptr ? "unknown" : name;
	for (char& character: text)
		character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
	return text;
}

std::string SafeText(std::string_view text) {
	std::string out;
	for (size_t index = 0; index < text.size();) {
		if (index + 16 <= text.size() && std::ranges::all_of(text.substr(index, 16), [](char c) {
			    return std::isxdigit(static_cast<unsigned char>(c)) != 0;
		    })) {
			out += "<id>";
			index += 16;
		} else {
			out += text[index++];
		}
	}
	return out;
}

bool ReadWords(const std::filesystem::path& path, std::vector<uint32_t>& words) {
	std::ifstream input(path, std::ios::binary | std::ios::ate);
	if (!input) return false;
	const auto bytes = input.tellg();
	if (bytes <= 0 || static_cast<uint64_t>(bytes) % sizeof(uint32_t) != 0) return false;
	words.resize(static_cast<size_t>(bytes) / sizeof(uint32_t));
	input.seekg(0);
	input.read(reinterpret_cast<char*>(words.data()), static_cast<std::streamsize>(bytes));
	return static_cast<bool>(input);
}

bool LoadProgram(const std::filesystem::path& path, ProgramInput& program) {
	std::vector<uint32_t> data;
	if (!ReadWords(path, data)) return false;
	program.vertex  = std::make_unique<ShaderVertexInputInfo>();
	program.pixel   = std::make_unique<ShaderPixelInputInfo>();
	program.compute = std::make_unique<ShaderComputeInputInfo>();
	size_t     at   = 0;
	const auto take = [&](std::vector<uint32_t>& out, size_t count) {
		if (count > data.size() - at) return false;
		out.assign(data.begin() + static_cast<std::ptrdiff_t>(at),
		           data.begin() + static_cast<std::ptrdiff_t>(at + count));
		at += count;
		return true;
	};
	const auto input = [&](void* out, size_t size) {
		if (size != data[5] || at > data.size() || size / 4 > data.size() - at ||
		    (size + 3) / 4 > data.size() - at) {
			return false;
		}
		std::memcpy(out, data.data() + at, size);
		at += (size + 3) / 4;
		return true;
	};

	if (data[0] == 0x5041434bu && data.size() >= 10) {
		program.capsule        = true;
		program.stage          = static_cast<ShaderType>(data[1]);
		program.wave_size      = data[2];
		program.user_data_base = data[3];
		program.hash           = data[8] | uint64_t {data[9]} << 32u;
		at                     = 10;
		if (program.stage == ShaderType::Compute) {
			if (!input(program.compute.get(), sizeof(*program.compute))) return false;
		} else if (program.stage == ShaderType::Pixel) {
			if (!input(program.pixel.get(), sizeof(*program.pixel))) return false;
		} else if (program.stage == ShaderType::Vertex || program.stage == ShaderType::Local ||
		           program.stage == ShaderType::TessellationControl ||
		           program.stage == ShaderType::TessellationEvaluation ||
		           program.stage == ShaderType::Mesh) {
			if (!input(program.vertex.get(), sizeof(*program.vertex))) return false;
		} else {
			return false;
		}
		if (!take(program.user_data, data[4]) || !take(program.code, data[6]) ||
		    !take(program.back_code, data[7]) || at != data.size()) {
			return false;
		}
		return true;
	}

	if (data.size() < 4 || data[0] != 0x4745524bu) return false;
	std::vector<uint32_t> regs;
	const auto            type = data[1], count = data[3], bytes = data[2];
	at = 4;
	if (count > (data.size() - at) / 2 || !take(regs, size_t {count} * 2u) ||
	    !take(program.code, bytes / 4u) || bytes % 4u != 0) {
		return false;
	}
	const bool     has_scratch_size = at < data.size();
	const uint32_t dumped_scratch   = has_scratch_size ? data[at] : 0u;
	const auto     scratch_size     = [&](uint32_t rsrc2) {
        return has_scratch_size ? dumped_scratch : (rsrc2 & 1u) != 0u ? 0xffffu : 0u;
	};
	const auto reg = [&](uint32_t offset) {
		for (uint32_t index = 0; index < count; ++index) {
			if (regs[index * 2u] == offset) return regs[index * 2u + 1u];
		}
		return 0u;
	};
	if (type == 0u) {
		const auto rsrc2 = reg(0x213);
		program.stage    = ShaderType::Compute;
		const auto users = (rsrc2 >> 1u) & 31u;
		program.user_data.resize(users);
		program.compute->scratch_size_dwords = scratch_size(rsrc2);
		program.compute->group_id[0]         = (rsrc2 >> 7u) & 1u;
		program.compute->group_id[1]         = (rsrc2 >> 8u) & 1u;
		program.compute->group_id[2]         = (rsrc2 >> 9u) & 1u;
		program.compute->tg_size_en          = (rsrc2 >> 10u) & 1u;
		program.compute->thread_ids_num      = static_cast<int>((rsrc2 >> 11u) & 3u) + 1;
		program.compute->workgroup_register  = static_cast<int>(users);
		for (uint32_t index = 0; index < 3u; ++index) {
			program.compute->threads_num[index] = std::max(reg(0x207 + index) & 0xffffu, 1u);
		}
		program.compute->lds_size_dwords = ((rsrc2 >> 15u) & 0x1ffu) * 128u;
		const auto* wave                 = std::getenv("SHADER_CHECK_WAVE");
		program.compute->wave_size       = wave != nullptr && wave[0] == '3' ? 32u : 64u;
		program.wave_size                = program.compute->wave_size;
		return true;
	}
	if (type == 1u) {
		const auto rsrc2 = reg(0x0b);
		program.stage    = ShaderType::Pixel;
		program.user_data.resize(((rsrc2 >> 1u) & 31u) | ((rsrc2 >> 27u) & 1u) << 5u);
		program.pixel->scratch_size_dwords  = scratch_size(rsrc2);
		program.pixel->input_num            = 32u;
		program.pixel->ps_system_input_base = 2u;
		const auto* wave                    = std::getenv("SHADER_CHECK_WAVE");
		program.wave_size                   = program.pixel->wave_size =
            wave != nullptr && wave[0] == '3' ? 32u : 64u;
		return true;
	}
	if (type == 2u && std::getenv("SHADER_CHECK_GS") != nullptr) {
		const auto     rsrc1 = reg(0x8a), rsrc2 = reg(0x8b);
		const auto     users                = ((rsrc2 >> 1u) & 31u) | ((rsrc2 >> 27u) & 1u) << 5u;
		const auto     lds                  = (rsrc2 >> 19u) & 0xffu;
		const auto*    wave                 = std::getenv("SHADER_CHECK_WAVE");
		const uint32_t wave_size            = wave != nullptr && wave[0] == '3' ? 32u : 64u;
		program.vertex->scratch_size_dwords = scratch_size(rsrc2);
		program.vertex->float_mode          = static_cast<uint8_t>((rsrc1 >> 12u) & 0xffu);
		if (lds == 0u) {
			program.stage = program.vertex->logical_stage = ShaderType::Vertex;
			program.user_data_base                        = 8u;
			program.wave_size = program.vertex->wave_size = wave_size;
			program.user_data.resize(users);
			return true;
		}
		if (((rsrc1 >> 29u) & 3u) == 0u && ((rsrc2 >> 16u) & 3u) == 1u) {
			program.stage = program.vertex->logical_stage = ShaderType::Mesh;
			program.user_data_base                        = 0u;
			auto& mesh                                    = program.vertex->mesh;
			mesh.input_primitive       = static_cast<uint32_t>(Prospero::PrimitiveType::kPointList);
			mesh.output_primitive_size = 3u;
			mesh.fast_launch           = true;
			mesh.primitives_per_group = mesh.vertices_per_group = 1u;
			mesh.max_vertices = mesh.max_primitives = 128u;
			mesh.wave_size = program.wave_size = program.vertex->wave_size = wave_size;
			mesh.lds_size_dwords                                           = lds * 128u;
			mesh.scratch_size_dwords = program.vertex->scratch_size_dwords;
			mesh.threads_num[0]      = 128u;
			mesh.threads_num[1] = mesh.threads_num[2] = 1u;
			program.user_data.resize(8u + users);
			return true;
		}
	}
	program.skip = type == 2u ? "ngg" : "stage";
	return false;
}

void PrintVertexInfo(const ProgramInput& program, const ShaderRecompiler::CompileOptions& options) {
	if (std::getenv("SHADER_CHECK_INFO") == nullptr || options.stage == ShaderType::Compute ||
	    options.stage == ShaderType::Pixel) {
		return;
	}
	const auto& v = *program.vertex;
	const auto& m = v.mesh;
	std::printf(
	    "INFO stage=%d logical=%d wave=%u base=%u users=%zu back=%zu fetch=%d/%d/%d res=%d "
	    "bufs=%d scratch=%u lds=%u vsout=%08x ucp=%u indirect=%d | mesh in=%u out=%u ppg=%u "
	    "vpg=%u maxv=%u maxp=%u prov=%u fast=%d pid=%d/%d wave=%u lds=%u scratch=%u "
	    "threads=%u,%u,%u\n",
	    static_cast<int>(options.stage), static_cast<int>(v.logical_stage), options.wave_size,
	    options.user_data_base, program.user_data.size(), program.back_code.size(),
	    v.has_dynamic_vertex_fetch, v.fetch_attrib_reg, v.fetch_buffer_reg, v.resources_num,
	    v.buffers_num, v.scratch_size_dwords, v.lds_size_dwords, v.pa_cl_vs_out_cntl,
	    v.user_clip_planes, v.indirect_draw, m.input_primitive, m.output_primitive_size,
	    m.primitives_per_group, m.vertices_per_group, m.max_vertices, m.max_primitives,
	    m.provoking_vertex, m.fast_launch, m.input_primitive_id, m.output_primitive_id, m.wave_size,
	    m.lds_size_dwords, m.scratch_size_dwords, m.threads_num[0], m.threads_num[1],
	    m.threads_num[2]);
}

Json BaseResult(const std::filesystem::path& path, ShaderType stage) {
	return {{"file", SafeText(path.filename().string())},
	        {"stage", StageSlug(stage)},
	        {"outcome", "crashed"},
	        {"shape", ""},
	        {"phase_ms", Json::object()},
	        {"blocks", 0},
	        {"loops", 0},
	        {"spirv_words", 0},
	        {"op_words", Json::object()},
	        {"from_recipe", false},
	        {"total_ms", 0.0}};
}

bool WriteJson(const std::filesystem::path& path, const Json& value) {
	std::ofstream output(path, std::ios::trunc);
	if (!output) return false;
	output << value.dump() << '\n';
	return static_cast<bool>(output);
}

bool ReadVariant(const std::filesystem::path& capsule, const ProgramInput& program,
                 ShaderRecompiler::CompileOptions&             options,
                 ShaderRecompiler::IR::ResourceSpecialization& specialization,
                 ShaderRecompiler::IR::PushData::Window& window, std::string* reason = nullptr) {
	const auto reject = [&](std::string_view text) {
		if (reason != nullptr) *reason = text;
		return false;
	};
	std::ifstream input(capsule.string() + ".variant", std::ios::binary);
	if (!input || !program.capsule) return reject("sidecar is missing");
	std::vector<uint8_t>         raw((std::istreambuf_iterator<char>(input)),
	                                 std::istreambuf_iterator<char>());
	ShaderRecompiler::IR::Reader reader {raw};
	uint8_t                      type = 0;
	ShaderType                   stage {};
	uint64_t                     hash = 0, id = 0;
	uint32_t                     user_data_count = 0, wave = 0, base = 0;
	std::vector<uint32_t>        static_state, user_data;
	std::vector<uint8_t>         input_bytes;
	ShaderRecompiler::IR::IoAll(reader, type, stage, hash, user_data_count, static_state, wave,
	                            base, user_data, window.begin, window.end, specialization, id,
	                            input_bytes);
	if (!reader.ok) return reject("record read failed");
	if (!reader.bytes.empty()) {
		return reject("record has " + std::to_string(reader.bytes.size()) + " trailing bytes of " +
		              std::to_string(raw.size()));
	}
	if (type != 1u) return reject("record kind is not a variant");
	if (stage != program.stage || hash != program.hash) return reject("record identity differs");
	if (wave != options.wave_size || base != options.user_data_base)
		return reject("record options differ");
	if (user_data_count != options.user_data.size() ||
	    !std::ranges::equal(user_data, options.user_data)) {
		return reject("record user data differs");
	}
	return true;
}

int TranslateOne(const std::filesystem::path& path, const std::filesystem::path& result_path,
                 const std::filesystem::path& spirv_path, bool corpus_mode) {
	Json         result      = BaseResult(path, ShaderType::Unknown);
	const auto   total_begin = std::chrono::steady_clock::now();
	ProgramInput program;
	if (!LoadProgram(path, program)) {
		result["outcome"] = "invalid-input";
		result["shape"] =
		    program.skip.empty() ? "not a supported capsule or register dump" : program.skip;
		result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
		if (!result_path.empty() && !WriteJson(result_path, result)) return 1;
		if (!corpus_mode) {
			if (program.skip.empty()) return 2;
			std::printf("SKIP %s\n", SafeText(result["shape"].get<std::string>()).c_str());
			return 3;
		}
		return 2;
	}
	result["stage"] = StageSlug(program.stage);
	ShaderRecompiler::CompileOptions options;
	options.dump_ir        = std::getenv("SHADER_CHECK_DUMP") != nullptr;
	options.stage          = program.stage;
	options.wave_size      = program.wave_size;
	options.user_data_base = program.user_data_base;
	options.shader_hash    = program.hash;
	options.user_data      = program.user_data;
	options.back_code      = program.back_code;
	if (program.stage == ShaderType::Compute)
		options.input_info.compute = program.compute.get();
	else if (program.stage == ShaderType::Pixel)
		options.input_info.pixel = program.pixel.get();
	else
		options.input_info.vertex = program.vertex.get();
	PrintVertexInfo(program, options);

	const auto prepare_begin     = std::chrono::steady_clock::now();
	auto       source            = ShaderRecompiler::PrepareShaderSource(program.code, options);
	result["phase_ms"]["decode"] = Milliseconds(std::chrono::steady_clock::now() - prepare_begin);
	if (source.unsupported_call_pc != UINT32_MAX) {
		result["outcome"]  = "rejected:call-target";
		result["shape"]    = "GPU-selected call target is not present in the capsule";
		result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
		if (!result_path.empty() && !WriteJson(result_path, result)) return 1;
		if (!corpus_mode) {
			std::printf("unsupported GPU-selected call target pc=0x%08x\n",
			            source.unsupported_call_pc);
			return 4;
		}
		return 4;
	}
	if (!source.calls.empty()) {
		result["outcome"]  = "skipped:calls";
		result["shape"]    = "linked calls require guest memory that the capsule does not carry";
		result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
		if (!result_path.empty() && !WriteJson(result_path, result)) return 1;
		if (!corpus_mode) {
			std::puts("SKIP calls");
			return 3;
		}
		return 3;
	}

	const auto cfg_begin       = std::chrono::steady_clock::now();
	auto       graph           = ShaderRecompiler::CFG::BuildGraph(source.decoded);
	result["phase_ms"]["cfg"]  = Milliseconds(std::chrono::steady_clock::now() - cfg_begin);
	result["blocks"]           = graph.blocks.size();
	result["loops"]            = graph.natural_loops.size();
	const auto translate_begin = std::chrono::steady_clock::now();
	auto       translated      = ShaderRecompiler::TranslateProgram(source.decoded, options);
	result["phase_ms"]["translate"] =
	    Milliseconds(std::chrono::steady_clock::now() - translate_begin);

	const bool emit = corpus_mode || !spirv_path.empty();
	if (emit) {
		ShaderRecompiler::Spirv::Profile profile {};
		profile.robust_buffer_access = true;
		profile.ray_query            = true;
		// SHADER_CHECK_MINMAX=othernan|exact: the float min/max lowering a probed host earns.
		if (const char* minmax = std::getenv("SHADER_CHECK_MINMAX"); minmax != nullptr) {
			if (std::strcmp(minmax, "exact") == 0) {
				profile.float_min_max = ShaderRecompiler::Spirv::FloatMinMax::Exact;
			} else if (std::strcmp(minmax, "othernan") == 0) {
				profile.float_min_max = ShaderRecompiler::Spirv::FloatMinMax::OtherNanSelect;
			}
		}
		options.profile              = &profile;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::PushData::Window       window;
		const bool from_recipe = ReadVariant(path, program, options, specialization, window);
		if (!from_recipe) {
			specialization.buffers.resize(translated.program.info.buffers.size());
			specialization.images.resize(translated.program.info.images.size());
			specialization.samplers.resize(translated.program.info.samplers.size());
		}
		const auto spirv_begin = std::chrono::steady_clock::now();
		auto       compiled    = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                                          specialization, window);
		result["phase_ms"]["spirv"] = Milliseconds(std::chrono::steady_clock::now() - spirv_begin);
		const auto& spirv           = compiled.spirv;
		result["spirv_words"]       = spirv.size();
		result["from_recipe"]       = from_recipe;
		std::vector<std::pair<uint64_t, uint32_t>> words_by_op(65536);
		for (uint32_t op = 0; op < 65536; op++)
			words_by_op[op].second = op;
		uint64_t instructions = 0;
		for (size_t at = 5; at < spirv.size();) {
			const auto length = spirv[at] >> 16u;
			if (length == 0 || at + length > spirv.size()) break;
			words_by_op[spirv[at] & 0xffffu].first += length;
			instructions++;
			at += length;
		}
		std::sort(words_by_op.begin(), words_by_op.end(), std::greater<>());
		Json op_words = Json::object();
		for (size_t index = 0; index < 16 && words_by_op[index].first != 0; index++) {
			op_words[std::to_string(words_by_op[index].second)] = words_by_op[index].first;
		}
		result["op_words"] = std::move(op_words);
		result["outcome"]  = "emitted";
		if (!spirv_path.empty()) {
			std::ofstream output(spirv_path, std::ios::binary | std::ios::trunc);
			if (!output) return 1;
			output.write(reinterpret_cast<const char*>(spirv.data()),
			             static_cast<std::streamsize>(spirv.size() * sizeof(uint32_t)));
			if (!output) return 1;
		}
		if (!corpus_mode) {
			std::printf("specialization: %s\n", from_recipe ? "from the recipe" : "default");
			std::printf("SPIRV words=%zu instructions=%llu code_words=%zu\n", spirv.size(),
			            static_cast<unsigned long long>(instructions), program.code.size());
			for (size_t index = 0; index < 16 && words_by_op[index].first != 0; index++) {
				std::printf("  op %u: %llu words (%.1f%%)\n", words_by_op[index].second,
				            static_cast<unsigned long long>(words_by_op[index].first),
				            100.0 * static_cast<double>(words_by_op[index].first) /
				                static_cast<double>(spirv.size()));
			}
		}
	} else {
		result["outcome"] = "translated";
	}
	result["total_ms"] = Milliseconds(std::chrono::steady_clock::now() - total_begin);
	if (!result_path.empty() && !WriteJson(result_path, result)) return 1;
	if (!corpus_mode) {
		std::puts("OK");
		std::fflush(nullptr);
	}
	return 0;
}

bool ReadCapsuleHeader(const std::filesystem::path& path, ShaderType& stage, uint64_t& hash) {
	std::ifstream            input(path, std::ios::binary);
	std::array<uint32_t, 10> header {};
	input.read(reinterpret_cast<char*>(header.data()), sizeof(header));
	if (!input || header[0] != 0x5041434bu) return false;
	stage = static_cast<ShaderType>(header[1]);
	hash  = header[8] | uint64_t {header[9]} << 32u;
	return true;
}

Json FailureResult(const CorpusEntry& entry, std::string outcome, std::string shape) {
	auto result       = BaseResult(entry.capsule, entry.stage.empty() ? ShaderType::Unknown
	                                                                  : static_cast<ShaderType>(0));
	result["file"]    = SafeText(entry.file);
	result["stage"]   = entry.stage.empty() ? "unknown" : entry.stage;
	result["outcome"] = std::move(outcome);
	result["shape"]   = SafeText(shape);
	return result;
}

#ifdef _WIN32
std::string WindowsQuote(const std::string& text) {
	std::string out     = "\"";
	size_t      slashes = 0;
	for (char character: text) {
		if (character == '\\') {
			slashes++;
			continue;
		}
		if (character == '"') {
			out.append(slashes * 2 + 1, '\\');
			out += character;
		} else {
			out.append(slashes, '\\');
			out += character;
		}
		slashes = 0;
	}
	out.append(slashes * 2, '\\');
	out += '"';
	return out;
}
#endif

struct RunningChild {
#ifdef _WIN32
	HANDLE process = nullptr;
#else
	pid_t pid = -1;
#endif
	size_t                                index = 0;
	std::chrono::steady_clock::time_point deadline;
};

bool StartChild(const std::vector<std::string>& arguments, RunningChild& child) {
#ifdef _WIN32
	std::string command;
	for (const auto& argument: arguments) {
		if (!command.empty()) command += ' ';
		command += WindowsQuote(argument);
	}
	STARTUPINFOA        startup {sizeof(startup)};
	PROCESS_INFORMATION process {};
	if (!CreateProcessA(nullptr, command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr,
	                    nullptr, &startup, &process)) {
		return false;
	}
	child.process = process.hProcess;
	CloseHandle(process.hThread);
	return true;
#else
	std::vector<char*> argv;
	argv.reserve(arguments.size() + 1);
	for (const auto& argument: arguments)
		argv.push_back(const_cast<char*>(argument.c_str()));
	argv.push_back(nullptr);
	const auto error =
	    ::posix_spawnp(&child.pid, g_self.c_str(), nullptr, nullptr, argv.data(), environ);
	return error == 0;
#endif
}

// False means the wait itself failed. A timeout is an ordinary child result.
bool WaitChild(RunningChild& child, uint32_t& code, bool& timed_out) {
	timed_out = false;
#ifdef _WIN32
	const auto waited = WaitForSingleObject(child.process, 0);
	if (waited == WAIT_OBJECT_0) {
		DWORD exit_code = kStillActive;
		GetExitCodeProcess(child.process, &exit_code);
		code = exit_code;
		CloseHandle(child.process);
		child.process = nullptr;
		return true;
	}
	if (waited == WAIT_TIMEOUT && std::chrono::steady_clock::now() >= child.deadline) {
		TerminateProcess(child.process, kStillActive);
		WaitForSingleObject(child.process, INFINITE);
		code      = kStillActive;
		timed_out = true;
		CloseHandle(child.process);
		child.process = nullptr;
		return true;
	}
	if (waited == WAIT_TIMEOUT) {
		code = kStillActive;
		return true;
	}
	if (waited == WAIT_FAILED && child.process != nullptr) {
		CloseHandle(child.process);
		child.process = nullptr;
	}
	return false;
#else
	int        status = 0;
	const auto waited = ::waitpid(child.pid, &status, WNOHANG);
	if (waited == 0 && std::chrono::steady_clock::now() >= child.deadline) {
		::kill(child.pid, SIGKILL);
		while (::waitpid(child.pid, &status, 0) < 0 && errno == EINTR) {
		}
		code      = kStillActive;
		timed_out = true;
		return true;
	}
	if (waited == child.pid) {
		code = WIFEXITED(status)     ? static_cast<uint32_t>(WEXITSTATUS(status))
		       : WIFSIGNALED(status) ? 128u + static_cast<uint32_t>(WTERMSIG(status))
		                             : 0u;
		return true;
	}
	if (waited == 0 || (waited < 0 && errno == EINTR)) {
		code = kStillActive;
		return true;
	}
	return false;
#endif
}

bool ReadJsonLines(const std::filesystem::path& path, std::vector<Json>& records) {
	std::ifstream input(path);
	if (!input) return false;
	for (std::string line; std::getline(input, line);) {
		if (line.empty()) continue;
		auto record = Json::parse(line, nullptr, false);
		if (record.is_discarded() || !record.is_object()) return false;
		records.push_back(std::move(record));
	}
	return true;
}

bool ParseUnsigned(const char* text, uint32_t& value) {
	char*      end    = nullptr;
	const auto parsed = std::strtoul(text, &end, 10);
	if (end == text || *end != '\0' || parsed > UINT32_MAX) return false;
	value = static_cast<uint32_t>(parsed);
	return true;
}

struct LiveLine {
	std::string stage;
	uint64_t    hash    = 0;
	bool        emitted = false;
};

bool ParseHash16(std::string_view text, uint64_t& hash) {
	if (text.size() != 16) return false;
	uint64_t value = 0;
	for (char character: text) {
		uint32_t digit = 0;
		if (character >= '0' && character <= '9')
			digit = static_cast<uint32_t>(character - '0');
		else if (character >= 'a' && character <= 'f')
			digit = static_cast<uint32_t>(character - 'a' + 10);
		else if (character >= 'A' && character <= 'F')
			digit = static_cast<uint32_t>(character - 'A' + 10);
		else
			return false;
		value = (value << 4u) | digit;
	}
	hash = value;
	return true;
}

std::optional<LiveLine> ParseLiveLine(std::string_view line) {
	constexpr std::string_view EndMarker = " translation end: hash=0x";
	if (const auto end = line.find(EndMarker); end != std::string_view::npos) {
		const auto head = line.rfind("ShaderRecompiler ", end);
		if (head == std::string_view::npos) return {};
		LiveLine   parsed;
		const auto stage = line.substr(head + 17, end - head - 17);
		parsed.stage.assign(stage);
		for (char& character: parsed.stage) {
			character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
		}
		const auto hash_at = end + EndMarker.size();
		if (line.size() < hash_at + 16 || !ParseHash16(line.substr(hash_at, 16), parsed.hash))
			return {};
		if (const auto outcome = line.find(" outcome=", hash_at + 16);
		    outcome != std::string_view::npos) {
			const auto text = line.substr(outcome + 9);
			parsed.emitted  = text.starts_with("emitted") || text.starts_with("served from store");
		}
		return parsed;
	}
	constexpr std::string_view CompileMarker = "Shader compile: ";
	if (const auto compile = line.find(CompileMarker); compile != std::string_view::npos) {
		const auto rest      = line.substr(compile + CompileMarker.size());
		const auto stage_end = rest.find(' ');
		if (stage_end == std::string_view::npos || rest.size() < stage_end + 19 ||
		    rest.substr(stage_end + 1, 2) != "0x") {
			return {};
		}
		LiveLine parsed;
		parsed.stage.assign(rest.substr(0, stage_end));
		for (char& character: parsed.stage) {
			character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
		}
		if (!ParseHash16(rest.substr(stage_end + 3, 16), parsed.hash)) return {};
		parsed.emitted = true;
		return parsed;
	}
	return {};
}

struct Agreement {
	uint64_t tool_total = 0, not_in_log = 0, both = 0, only_live = 0, only_tool = 0, neither = 0;
	std::map<std::string, uint64_t> only_live_by_tool_outcome;
};

std::map<std::string, Agreement> CompareWithLiveLog(const std::vector<std::string>& lines,
                                                    const std::vector<CorpusEntry>& entries,
                                                    const std::vector<Json>&        results) {
	std::map<std::pair<std::string, uint64_t>, bool> live;
	for (const auto& line: lines) {
		if (const auto parsed = ParseLiveLine(line))
			live[{parsed->stage, parsed->hash}] |= parsed->emitted;
	}
	std::map<std::string, Agreement> stages;
	for (size_t index = 0; index < entries.size() && index < results.size(); index++) {
		const auto& entry = entries[index];
		if (!entry.hash_valid || entry.stage.empty()) continue;
		auto& agreement = stages[entry.stage];
		agreement.tool_total++;
		const auto found = live.find({entry.stage, entry.hash});
		if (found == live.end()) {
			agreement.not_in_log++;
			continue;
		}
		const auto outcome      = results[index].value("outcome", "crashed");
		const bool tool_emitted = outcome == "emitted";
		if (found->second && tool_emitted)
			agreement.both++;
		else if (found->second) {
			agreement.only_live++;
			agreement.only_live_by_tool_outcome[outcome]++;
		} else if (tool_emitted)
			agreement.only_tool++;
		else
			agreement.neither++;
	}
	return stages;
}

std::string AgreementText(const std::map<std::string, Agreement>& stages) {
	std::string text = "\n## Agreement with the live log\n\n";
	if (stages.empty()) text += "- no comparable shader was translated\n";
	for (const auto& [stage, agreement]: stages) {
		const auto live_emitted = agreement.both + agreement.only_live;
		char       line[256];
		std::snprintf(line, sizeof(line),
		              "- %s: tool %llu, not in log %llu; emitted in both %llu, only live %llu, "
		              "only tool %llu, neither %llu; fidelity %.1f%% of %llu live-emitted\n",
		              stage.c_str(), static_cast<unsigned long long>(agreement.tool_total),
		              static_cast<unsigned long long>(agreement.not_in_log),
		              static_cast<unsigned long long>(agreement.both),
		              static_cast<unsigned long long>(agreement.only_live),
		              static_cast<unsigned long long>(agreement.only_tool),
		              static_cast<unsigned long long>(agreement.neither),
		              live_emitted == 0 ? 0.0
		                                : 100.0 * static_cast<double>(agreement.both) /
		                                      static_cast<double>(live_emitted),
		              static_cast<unsigned long long>(live_emitted));
		text += line;
		for (const auto& [outcome, count]: agreement.only_live_by_tool_outcome) {
			text += "  - only live, tool said " + SafeText(outcome) + ": " + std::to_string(count) +
			        "\n";
		}
	}
	return text;
}

bool WriteSummary(const std::filesystem::path& output, const std::vector<Json>& results,
                  const std::vector<CorpusEntry>& entries,
                  const std::filesystem::path&    compare_log) {
	std::map<std::string, uint64_t>                        outcomes;
	std::map<std::string, std::map<std::string, uint64_t>> stages;
	std::map<std::string, double>                          phase_totals;
	std::map<std::string, uint64_t>                        op_totals;
	for (const auto& result: results) {
		outcomes[result.value("outcome", "crashed")]++;
		stages[result.value("stage", "unknown")][result.value("outcome", "crashed")]++;
		if (result.contains("phase_ms")) {
			for (const auto& [name, milliseconds]: result["phase_ms"].items()) {
				phase_totals[name] += milliseconds.get<double>();
			}
		}
		if (result.contains("op_words")) {
			for (const auto& [op, words]: result["op_words"].items())
				op_totals[op] += words.get<uint64_t>();
		}
	}
	std::ofstream summary(output / "summary.md", std::ios::trunc);
	if (!summary) return false;
	summary << "# Offline shader corpus\n\nShaders: " << results.size() << "\n\n## Outcomes\n\n";
	if (outcomes.empty()) summary << "- none\n";
	for (const auto& [outcome, count]: outcomes)
		summary << "- " << SafeText(outcome) << ": " << count << '\n';
	summary << "\n## Outcomes by stage\n\n";
	for (const auto& [stage, counts]: stages) {
		summary << "- " << SafeText(stage) << ':';
		for (const auto& [outcome, count]: counts)
			summary << ' ' << SafeText(outcome) << '=' << count;
		summary << '\n';
	}
	summary << "\n## Translation phases\n\n";
	if (phase_totals.empty()) summary << "- none\n";
	for (const auto& [name, milliseconds]: phase_totals)
		summary << "- " << name << ": " << milliseconds << " ms\n";
	summary << "\n## SPIR-V opcode words\n\n";
	if (op_totals.empty()) summary << "- none\n";
	std::vector<std::pair<uint64_t, std::string>> ranked_ops;
	for (const auto& [op, words]: op_totals)
		ranked_ops.emplace_back(words, op);
	std::ranges::sort(ranked_ops, std::greater<>());
	for (size_t index = 0; index < std::min(size_t {16}, ranked_ops.size()); index++) {
		summary << "- op " << ranked_ops[index].second << ": " << ranked_ops[index].first
		        << " words\n";
	}
	std::vector<size_t> largest(results.size());
	for (size_t index = 0; index < largest.size(); index++)
		largest[index] = index;
	std::ranges::sort(largest, [&](size_t left, size_t right) {
		return results[left].value("spirv_words", uint64_t {0}) >
		       results[right].value("spirv_words", uint64_t {0});
	});
	summary << "\n## Largest emitted modules\n\n";
	if (largest.empty()) summary << "- none\n";
	for (size_t index = 0; index < std::min(size_t {20}, largest.size()); index++) {
		const auto& result = results[largest[index]];
		if (result.value("spirv_words", uint64_t {0}) == 0) continue;
		summary << "- " << SafeText(result.value("file", "?")) << ": "
		        << result.value("spirv_words", uint64_t {0}) << " words\n";
	}
	summary << "\n## Driver cost\n\nPipeline creation timings are intentionally not measured here; "
	           "run spvbench on the emitted modules.\n";
	if (!compare_log.empty()) {
		std::ifstream log(compare_log);
		if (!log) return false;
		std::vector<std::string> lines;
		for (std::string line; std::getline(log, line);) {
			if (line.find(" translation end: hash=0x") != std::string::npos ||
			    line.find("Shader compile: ") != std::string::npos) {
				lines.push_back(std::move(line));
			}
		}
		summary << AgreementText(CompareWithLiveLog(lines, entries, results));
	}
	return static_cast<bool>(summary);
}

int RunCorpus(const Options& options) {
	std::error_code error;
	std::filesystem::create_directories(options.output, error);
	if (error) {
		std::fprintf(stderr, "shader_check: could not create output directory: %s\n",
		             SafeText(error.message()).c_str());
		return 1;
	}
	std::vector<CorpusEntry>            entries;
	std::filesystem::directory_iterator iterator(options.input, error);
	if (error) {
		std::fprintf(stderr, "shader_check: could not open corpus: %s\n",
		             SafeText(error.message()).c_str());
		return 1;
	}
	for (const auto& item: iterator) {
		if (!item.is_regular_file() || item.path().extension() != ".cap") continue;
		CorpusEntry entry;
		entry.capsule = item.path();
		entry.file    = item.path().filename().string();
		ShaderType stage {};
		uint64_t   hash  = 0;
		entry.hash_valid = ReadCapsuleHeader(entry.capsule, stage, hash);
		if (entry.hash_valid) {
			entry.hash  = hash;
			entry.stage = StageSlug(stage);
		}
		if (!options.only.empty() && entry.stage != options.only) continue;
		entries.push_back(std::move(entry));
	}
	std::ranges::sort(entries, {}, [](const CorpusEntry& entry) { return entry.file; });
	if (options.limit != 0 && entries.size() > options.limit) entries.resize(options.limit);

	const auto work      = options.output / ".work";
	const auto spirv_dir = options.output / "spirv";
	std::filesystem::create_directories(work, error);
	std::filesystem::create_directories(spirv_dir, error);
	if (error) {
		std::fprintf(stderr, "shader_check: could not create work directories: %s\n",
		             SafeText(error.message()).c_str());
		return 1;
	}
	std::vector<Json>         results(entries.size());
	std::vector<RunningChild> running;
	size_t                    next = 0;
	while (next < entries.size() || !running.empty()) {
		while (next < entries.size() && running.size() < options.jobs) {
			const auto result_path = work / (std::to_string(next) + ".json");
			std::filesystem::remove(result_path, error);
			std::filesystem::path spirv_path;
			if (options.emit_spirv) {
				char name[64];
				std::snprintf(name, sizeof(name), "shader_%06zu.spv", next);
				spirv_path = spirv_dir / name;
			}
			RunningChild child;
			child.index = next;
			child.deadline =
			    std::chrono::steady_clock::now() + std::chrono::milliseconds(options.timeout_ms);
			const std::vector<std::string> arguments {
			    g_self, "--child", entries[next].capsule.string(), result_path.string(),
			    spirv_path.empty() ? "-" : spirv_path.string()};
			if (!StartChild(arguments, child)) {
				results[next] =
				    FailureResult(entries[next], "spawn-failed", "could not start child process");
				++next;
				continue;
			}
			running.push_back(child);
			++next;
		}
		if (running.empty()) continue;
		bool progressed = false;
		for (size_t slot = 0; slot < running.size();) {
			uint32_t code      = 0;
			bool     timed_out = false;
			if (!WaitChild(running[slot], code, timed_out)) {
				results[running[slot].index] = FailureResult(
				    entries[running[slot].index], "crashed", "waiting for the child failed");
				running.erase(running.begin() + static_cast<std::ptrdiff_t>(slot));
				progressed = true;
				continue;
			}
			if (code == kStillActive && !timed_out) {
				slot++;
				continue;
			}
			const auto        index       = running[slot].index;
			const auto        result_path = work / (std::to_string(index) + ".json");
			std::vector<Json> parsed;
			Json              result;
			if (ReadJsonLines(result_path, parsed) && parsed.size() == 1)
				result = std::move(parsed[0]);
			if (result.is_object()) {
				results[index] = std::move(result);
			} else if (timed_out) {
				results[index] =
				    FailureResult(entries[index], "timeout", "translation exceeded timeout");
			} else if (code == kFatalExit || code == kFatalExitPosix) {
				results[index] = FailureResult(entries[index], "rejected:fatal",
				                               "translator reported a fatal error");
			} else {
				results[index] = FailureResult(entries[index], "crashed",
				                               "child exited without a result (code " +
				                                   std::to_string(code) + ")");
			}
			std::filesystem::remove(result_path, error);
			running.erase(running.begin() + static_cast<std::ptrdiff_t>(slot));
			progressed = true;
		}
		if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}
	std::filesystem::remove(work, error);
	std::ofstream jsonl(options.output / "results.jsonl", std::ios::trunc);
	if (!jsonl) return 1;
	for (const auto& result: results)
		jsonl << result.dump() << '\n';
	jsonl.close();
	if (!WriteSummary(options.output, results, entries, options.compare_log)) {
		std::fprintf(stderr, "shader_check: could not write corpus summary\n");
		return 1;
	}
	std::printf("shader_check: %zu capsule(s), %zu emitted\n", results.size(),
	            static_cast<size_t>(std::ranges::count_if(results, [](const Json& result) {
		            return result.value("outcome", "") == "emitted";
	            })));
	return 0;
}

bool WriteCapsule(const std::filesystem::path& path, uint64_t hash, std::span<const uint32_t> code,
                  bool with_variant) {
	ShaderComputeInputInfo input {};
	input.threads_num[0] = input.threads_num[1] = input.threads_num[2] = 1;
	input.wave_size                                                    = 64;
	const uint32_t header[] {0x5041434bu,
	                         static_cast<uint32_t>(ShaderType::Compute),
	                         64,
	                         0,
	                         0,
	                         static_cast<uint32_t>(sizeof(input)),
	                         static_cast<uint32_t>(code.size()),
	                         0,
	                         static_cast<uint32_t>(hash),
	                         static_cast<uint32_t>(hash >> 32u)};
	std::ofstream  output(path, std::ios::binary | std::ios::trunc);
	if (!output) return false;
	output.write(reinterpret_cast<const char*>(header), sizeof(header));
	output.write(reinterpret_cast<const char*>(&input), sizeof(input));
	output.write(reinterpret_cast<const char*>(code.data()),
	             static_cast<std::streamsize>(code.size() * sizeof(uint32_t)));
	if (!output) return false;
	output.close();
	if (!with_variant) return true;
	ShaderRecompiler::IR::Writer writer;
	uint8_t                      type = 1;
	std::vector<uint32_t>        static_state, user_data;
	std::vector<uint8_t>         input_bytes(sizeof(input));
	std::memcpy(input_bytes.data(), &input, sizeof(input));
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	ShaderRecompiler::IR::IoAll(writer, type, ShaderType::Compute, hash, 0u, static_state, 64u, 0u,
	                            user_data, 0u, 32u, specialization, uint64_t {0}, input_bytes);
	std::ofstream variant(path.string() + ".variant", std::ios::binary | std::ios::trunc);
	variant.write(reinterpret_cast<const char*>(writer.bytes.data()),
	              static_cast<std::streamsize>(writer.bytes.size()));
	return static_cast<bool>(variant);
}

std::string TestHash(uint64_t value) {
	char text[32];
	std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(value));
	return text;
}

bool Check(bool condition, const char* what) {
	if (!condition) std::fprintf(stderr, "shader_check self-test: %s\n", what);
	return condition;
}

int SelfTest() {
	bool ok = true;
	{
		const std::vector<std::string> log {
		    "ShaderRecompiler CS translation end: hash=" + TestHash(1) + " outcome=emitted (async)",
		    "Shader compile: PS " + TestHash(2) + " new-code"};
		const std::vector<CorpusEntry> entries {{{}, "one.cap", "cs", 1, true},
		                                        {{}, "two.cap", "ps", 2, true}};
		const std::vector<Json>        results {{{"stage", "cs"}, {"outcome", "emitted"}},
		                                        {{"stage", "ps"}, {"outcome", "emitted"}}};
		const auto                     agreement = CompareWithLiveLog(log, entries, results);
		ok &= Check(agreement.at("cs").both == 1 && agreement.at("ps").both == 1,
		            "live-log parsers agree on emitted shaders");
	}
#ifdef _WIN32
	const auto process_id = GetCurrentProcessId();
#else
	const auto process_id = ::getpid();
#endif
	const auto root =
	    std::filesystem::current_path() / ("kyty-shader-check-test-" + std::to_string(process_id));
	const auto input  = root / "input";
	const auto output = root / "output";
	std::filesystem::remove_all(root);
	std::filesystem::create_directories(input);
	constexpr std::array<uint32_t, 1> emitted {0xbf810000u};
	constexpr std::array<uint32_t, 2> rejected {0xb96af818u, 0xbf810000u};
	ok &= Check(WriteCapsule(input / "one.cap", 1, emitted, true), "write emitted fixture");
	ok &= Check(WriteCapsule(input / "two.cap", 2, rejected, false), "write rejected fixture");
	{
		ProgramInput                     check;
		ShaderRecompiler::CompileOptions check_options;
		check_options.stage     = ShaderType::Compute;
		check_options.wave_size = 64;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::PushData::Window       window;
		std::string                                  reason;
		const bool                                   loaded = LoadProgram(input / "one.cap", check);
		check_options.user_data                             = check.user_data;
		const bool variant = loaded && ReadVariant(input / "one.cap", check, check_options,
		                                           specialization, window, &reason);
		ok &= Check(variant, reason.empty() ? "read recipe sidecar" : reason.c_str());
	}
	if (!ok) return 1;
	const auto log_path = root / "live.log";
	{
		std::ofstream log(log_path);
		log << "ShaderRecompiler CS translation end: hash=" << TestHash(1) << " outcome=emitted\n";
	}
	Options options;
	options.input            = input;
	options.output           = output;
	options.compare_log      = log_path;
	options.jobs             = 2;
	options.timeout_ms       = 10000;
	options.emit_spirv       = true;
	const int         status = RunCorpus(options);
	std::vector<Json> results;
	const bool        read         = ReadJsonLines(output / "results.jsonl", results);
	bool              have_emitted = false, have_rejected = false, have_recipe = false;
	for (const auto& result: results) {
		const auto outcome = result.value("outcome", "");
		have_emitted |= outcome == "emitted";
		have_rejected |= outcome.starts_with("rejected:");
		have_recipe |= outcome == "emitted" && result.value("from_recipe", false);
	}
	ok &= Check(status == 0 && read && results.size() == 2, "corpus translated both fixtures");
	ok &= Check(have_emitted && have_rejected && have_recipe, "corpus outcomes and recipe replay");
	if (!have_emitted || !have_rejected || !have_recipe) {
		for (const auto& result: results)
			std::fprintf(stderr, "self-test result: %s\n", result.dump().c_str());
	}
	ok &= Check(std::filesystem::is_regular_file(output / "spirv" / "shader_000000.spv"),
	            "corpus emitted SPIR-V");
	{
		std::ifstream     summary(output / "summary.md");
		const std::string text((std::istreambuf_iterator<char>(summary)),
		                       std::istreambuf_iterator<char>());
		ok &= Check(text.find("Agreement with the live log") != std::string::npos,
		            "summary contains live-log agreement");
		ok &= Check(text.find("spvbench") != std::string::npos,
		            "summary delegates driver timing to spvbench");
	}
	std::filesystem::remove_all(root);
	if (!ok) return 1;
	std::puts("shader_check self-test: OK");
	return 0;
}

void Usage() {
	std::printf("shader_check <capsule.cap | registers.reg>\n"
	            "shader_check --corpus <capsule dir> --out <report dir> [--jobs N] [--limit N]\n"
	            "            [--timeout-ms N] [--only cs|ps|vs|ls|hs|tes|ms] [--emit-spirv]\n"
	            "            [--compare-log <run log>]\n"
	            "shader_check --self-test\n");
}

std::optional<Options> ParseOptions(int argc, char** argv) {
	Options options;
	options.jobs = std::max(1u, std::thread::hardware_concurrency());
	for (int index = 1; index < argc; index++) {
		const std::string_view argument = argv[index];
		if (argument == "--self-test") {
			options.self_test = true;
			continue;
		}
		if (argument == "--emit-spirv") {
			options.emit_spirv = true;
			continue;
		}
		if (index + 1 >= argc) return {};
		const char* value = argv[++index];
		if (argument == "--corpus" || argument == "--dir")
			options.input = value;
		else if (argument == "--out")
			options.output = value;
		else if (argument == "--compare-log")
			options.compare_log = value;
		else if (argument == "--only") {
			options.only = value;
			for (char& character: options.only) {
				character = static_cast<char>(std::tolower(static_cast<unsigned char>(character)));
			}
		} else if (argument == "--jobs") {
			if (!ParseUnsigned(value, options.jobs) || options.jobs == 0) return {};
		} else if (argument == "--limit") {
			if (!ParseUnsigned(value, options.limit)) return {};
		} else if (argument == "--timeout-ms") {
			if (!ParseUnsigned(value, options.timeout_ms) || options.timeout_ms == 0) return {};
		} else
			return {};
	}
	if (!options.self_test && (options.input.empty() || options.output.empty())) return {};
	return options;
}

} // namespace

int main(int argc, char** argv) {
	g_self = argc > 0 && argv[0] != nullptr ? argv[0] : "shader_check";
	if (argc >= 4 && std::strcmp(argv[1], "--child") == 0 && (argc == 4 || argc == 5)) {
		const std::filesystem::path spirv =
		    argc == 5 && std::strcmp(argv[4], "-") != 0 ? argv[4] : "";
		return TranslateOne(argv[2], argv[3], spirv, true);
	}
	if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 || std::strcmp(argv[1], "-h") == 0)) {
		Usage();
		return 0;
	}
	if (argc == 2 && std::strcmp(argv[1], "--self-test") == 0) return SelfTest();
	if (argc == 2 && argv[1][0] != '-') {
		const char* spirv = std::getenv("SHADER_CHECK_SPIRV");
		return TranslateOne(argv[1], {}, spirv == nullptr ? "" : spirv, false);
	}
	const auto options = ParseOptions(argc, argv);
	if (!options) {
		Usage();
		return 2;
	}
	if (options->self_test) return SelfTest();
	return RunCorpus(*options);
}
