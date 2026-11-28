#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/faultReport.h"
#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/shaderCompileStatus.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineBinaryStore.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"

#include <csetjmp>
#include <csignal>
#include <thread>

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS && !defined(__APPLE__)
#include <sys/resource.h>
#include <unistd.h>
#endif

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fmt/format.h>
#include <ranges>
#include <xxhash.h>
#include <zstd.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

// The shader store: a header, then zstd-compressed records of everything a run created first.
// Records are only appended, so a store cut short by a crash keeps every complete record.
namespace Common {
extern thread_local std::jmp_buf* g_dbg_recover;
}

namespace Libs::Graphics {

namespace {

using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

enum class Record : uint8_t { Module, GraphicsPipeline, TessellationLayout };

// Precedes each record's zstd frame.
struct Frame {
	uint32_t compressed_size = 0;
	uint32_t raw_size        = 0;
	uint64_t checksum        = 0; // XXH3 of the compressed bytes.
};

// Changes with every build, so records never meet code that encodes them differently. Zero when
// the executable cannot be found.
uint64_t BuildStamp() {
	// KYTY_SHADER_CACHE_STAMP=N replaces the executable's time: a rebuild that does not change how
	// shaders are translated keeps the cache. Change N whenever the shader recompiler changes
	// (a main update, a shader patch), or the stored programs no longer match the code.
	if (const char* env = std::getenv("KYTY_SHADER_CACHE_STAMP"); env != nullptr && env[0] != 0) {
		const auto value = std::strtoull(env, nullptr, 10);
		if (value != 0) {
			return value;
		}
	}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::array<wchar_t, 32768> name {};
	GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	const std::filesystem::path path = name.data();
#elif defined(__APPLE__)
	std::array<char, 4096> name {};
	auto size = static_cast<uint32_t>(name.size());
	_NSGetExecutablePath(name.data(), &size);
	const std::filesystem::path path = name.data();
#else
	const std::filesystem::path path = "/proc/self/exe";
#endif
	std::error_code error;
	const auto      time = std::filesystem::last_write_time(path, error).time_since_epoch().count();
	return error ? 0 : static_cast<uint64_t>(time);
}

// The store's own format: what this file and pipelineCache.cpp make of the translator's output.
// KYTY_SHADER_CACHE_STAMP names the translator alone (the sources under graphics/shader and the
// guest_gpu definitions they include), so it no longer changes when this code does. BUMP THIS
// whenever any of these change:
//   - the record framing or the fields of a record (Frame, Record, IoKey, IoModule, the store
//     header);
//   - GraphicsPipelineKey or a struct in it, which records hold as raw bytes (the asserts below
//     fail when their layout changes);
//   - the module id formula (PipelineCache::CompileModule, ReplayRecipe);
//   - what a translation is given (the CompileOptions of PipelineCache::GetProgramImpl and of the
//     translations ahead, SpirvProfile's meaning for the same bytes).
// Version 1 leaves the header as it was before the constant existed.
constexpr uint64_t StoreFormatVersion = 1;

static_assert(sizeof(PipelineStaticParameters) == 118 && sizeof(PipelineRenderingState) == 44 &&
                  sizeof(PipelineVertexInputState) == 354 && sizeof(GraphicsPipelineKey) == 548 &&
                  offsetof(GraphicsPipelineKey, rendering) == 32 &&
                  offsetof(GraphicsPipelineKey, vertex_input) == 76 &&
                  offsetof(GraphicsPipelineKey, static_params) == 430 &&
                  offsetof(PipelineStaticParameters, color_mask) == 17 &&
                  offsetof(PipelineStaticParameters, color_srcblend) == 54 &&
                  offsetof(PipelineStaticParameters, blend_enable) == 110,
              "stored pipeline keys changed their layout: bump StoreFormatVersion, then update "
              "these numbers");

// The translator's code-generation switches read from the environment that Profile does not
// hold: a store written with one of them changed must not serve a run without it. Each switch
// away from its default changes the key; the defaults leave it as it was. A new switch of this
// kind in the translator belongs here.
uint64_t CodegenSwitches() {
	const auto env = [](const char* name) {
		const char* value = std::getenv(name);
		return value != nullptr ? value : "";
	};
	uint64_t switches = 0;
	// F16StoreClampEnabled.
	if (ShaderRecompiler::IR::F16StoreClampEnabled()) switches |= 1u;
	// PixelQuadDppEnabled (spirvEmitterFlow.cpp).
	if (env("KYTY_PS_QUAD_DPP")[0] != 0 && env("KYTY_PS_QUAD_DPP")[0] != '0') switches |= 2u;
	// MovrelLaneBoundsEnabled (ConstantPropagation.cpp).
	if (env("KYTY_MOVREL_LANE_BOUNDS")[0] == '0') switches |= 4u;
	// ExecSelectsEnabled (ShaderRecompiler.cpp).
	if (env("KYTY_EXEC_SELECTS")[0] == '0' || env("KYTY_EXEC_SELECT_ELIM")[0] == '0') {
		switches |= 8u;
	}
	return switches;
}

uint64_t CodegenSwitchKey() {
	uint64_t key = CodegenSwitches() * 0x9e3779b97f4a7c15ull;
	// Ballot guards change the SPIR-V of most modules but are not part of Profile (whose bytes
	// stay those of stores written without them).
	constexpr uint64_t BallotGuardsStamp = 0x62616c6c6f740001ull;
	if (ShaderRecompiler::Spirv::BallotGuardsEnabled()) key ^= BallotGuardsStamp;
	// The loop iteration limit is a constant in every module with a loop (0: no guard at all).
	if (const auto limit = ShaderRecompiler::Spirv::LoopIterationLimit();
	    limit != ShaderRecompiler::Spirv::DefaultLoopIterationLimit) {
		constexpr uint64_t LoopLimitStamp = 0x6c6f6f706c696d00ull;
		key ^= LoopLimitStamp ^ (uint64_t {limit} * 0xff51afd7ed558ccdull);
	}
	return key;
}

// What every record depends on besides its own fields; empty when the build is unknown.
std::vector<uint8_t> StoreHeader(const GraphicContext& graphics) {
	const auto stamp = BuildStamp();
	if (stamp == 0) {
		return {};
	}
	const uint64_t keyed_stamp =
	    stamp ^ CodegenSwitchKey() ^ ((StoreFormatVersion - 1) * 0xc2b2ae3d27d4eb4full);
	Writer header;
	IoAll(header, keyed_stamp, SpirvProfile(graphics),
	      graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize);
	return std::move(header.bytes);
}

// Record fields, shared by writing and reading.
template <class S, class Key> void IoKey(S& io, Key& key) {
	auto& [stage, hash, user_data_count, static_state] = key;
	IoAll(io, stage, hash, user_data_count, static_state);
}

template <class S, class Module> void IoModule(S& io, Module& stage) {
	// The handle and the SPIR-V report fields come from the SPIR-V; stored programs have no
	// function code.
	auto& [id, handle, info, specialization, function_code, rect_parameters, mesh_task, spirv_words,
	       spirv, stored, specialization_hash, function_code_hash] = stage;
	IoAll(io, id, info, specialization, rect_parameters, mesh_task);
}

} // namespace

// A store per set of switches: with one file, runs with and without a switch would discard each
// other's store at every start.
std::string ShaderStoreVariant() {
	const auto switches = CodegenSwitches();
	auto variant = switches == 0 ? std::string {} : fmt::format("-sw{:x}", switches);
	// A loop iteration limit away from the default (KYTY_SHADER_LOOP_LIMIT) gets its own file too.
	if (const auto limit = ShaderRecompiler::Spirv::LoopIterationLimit();
	    limit != ShaderRecompiler::Spirv::DefaultLoopIterationLimit) {
		variant += fmt::format("-loop{}", limit);
	}
	return variant;
}

bool ShaderStoreWriter::Open(const std::filesystem::path& path, std::span<const uint8_t> header,
                             uint64_t kept) {
	Common::File::CreateDirectories(path.parent_path());
	if (!m_lock.Acquire(path)) {
		return false;
	}
	if (kept == 0) {
		if (m_file.Create(path)) {
			m_file.Write(header.data(), static_cast<uint32_t>(header.size()));
		}
	} else if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
		m_file.Truncate(kept);
		m_file.Seek(kept);
	}
	return true;
}

ShaderStoreWriter::ShaderStoreWriter() {
	m_thread = std::jthread([this](std::stop_token stop) {
		Common::FaultReport::SetThreadTag("store write", "store-writer");
		Common::FaultReport::RegisterThreadStack();
		std::unique_lock lock(m_mutex);
		// Once stop is requested the wait returns false only when the queue is empty.
		while (m_ready.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
			const auto job = std::move(m_jobs.front());
			m_jobs.pop();
			lock.unlock();
			job();
			lock.lock();
		}
	});
}

void ShaderStoreWriter::Append(std::vector<uint8_t>&& record) {
	Post([this, record = std::move(record)] {
		std::vector<uint8_t> bytes(sizeof(Frame) + ZSTD_compressBound(record.size()));
		const auto size = ZSTD_compress(bytes.data() + sizeof(Frame), bytes.size() - sizeof(Frame),
		                                record.data(), record.size(), 3);
		EXIT_IF(ZSTD_isError(size));
		const Frame frame {.compressed_size = static_cast<uint32_t>(size),
		                   .raw_size        = static_cast<uint32_t>(record.size()),
		                   .checksum        = XXH3_64bits(bytes.data() + sizeof(Frame), size)};
		std::memcpy(bytes.data(), &frame, sizeof(frame));
		// A failed write leaves a partial record; the next run drops it and everything after it.
		m_file.Write(bytes.data(), static_cast<uint32_t>(sizeof(Frame) + size));
	});
}

void ShaderStoreWriter::Post(Common::UniqueFunction<void>&& job) {
	{
		std::scoped_lock lock(m_mutex);
		m_jobs.push(std::move(job));
	}
	m_ready.notify_one();
}

namespace {

std::atomic<bool> g_precompile_mode {false};
// Set by SIGINT or SIGTERM during a precompile.
volatile std::sig_atomic_t g_precompile_interrupted = 0;

bool PrecompileMode() {
	return g_precompile_mode.load(std::memory_order_relaxed);
}

// --restamp-shader-store: the store of another stamp is checked against this translator.
std::atomic<bool>   g_restamp_mode {false};
std::atomic<size_t> g_restamp_sample {0};
// --restamp-strict: all or nothing, as before partial carry-over existed.
std::atomic<bool>   g_restamp_strict {false};

// Whether Restamp() keeps what it verified and drops the rest (with a sample, the modules it
// leaves out are dropped as well).
bool PartialRestamp() {
	return !g_restamp_strict.load(std::memory_order_relaxed);
}

bool RestampMode() {
	return g_restamp_mode.load(std::memory_order_relaxed);
}

// The header's first field, the keyed stamp.
constexpr size_t StampBytes = sizeof(uint64_t);

// KYTY_SYNC_WARMUP=1: the warm-up creates every pipeline before the game starts (as before),
// instead of in the background.
bool SyncWarmUp() {
	static const bool sync = [] {
		const char* env = std::getenv("KYTY_SYNC_WARMUP");
		return env != nullptr && env[0] == '1';
	}();
	return sync;
}

// Decode time by part, summed over the warm-up's threads.
std::atomic<uint64_t> g_decode_zstd_ns {0};
std::atomic<uint64_t> g_decode_plan_ns {0};
std::atomic<uint64_t> g_decode_spirv_ns {0};
// Decompression at start, for the keys.
std::atomic<uint64_t> g_index_zstd_ns {0};
// Plans the command thread found not decoded yet, and its time on them (waits included).
std::atomic<uint64_t> g_demand_plans {0};
std::atomic<uint64_t> g_demand_plan_ns {0};

// KYTY_EAGER_SHADER_STORE=1 (or KYTY_FAILED_SPIRV_DUMP=1): every record is decoded before the
// game starts, as before, instead of on first use and in the background.
bool EagerStore() {
	static const bool eager = [] {
		const char* env = std::getenv("KYTY_EAGER_SHADER_STORE");
		return (env != nullptr && env[0] == '1') || RetainModuleSpirv();
	}();
	return eager;
}

uint64_t NanosecondsSince(std::chrono::steady_clock::time_point start) {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now() - start)
	                                 .count());
}

double MillisecondsSince(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
	    .count();
}

} // namespace

// A record decoded on any thread, inserted on the warm-up's.
struct PipelineCache::LoadedRecord {
	uint8_t                            type = 0; // Record
	ProgramKey                         key;
	ShaderModule                       module;
	GraphicsPipelineKey                pipeline_key;
	uint64_t                           layout_key = 0;
	ShaderTessellationInputInfo        layout;
	bool                               ok = false;
};

void PipelineCache::WarmUp(const std::filesystem::path& path) {
	if (path.empty()) {
		return;
	}
	const auto header = StoreHeader(m_graphics);
	if (header.empty()) {
		return;
	}
	const auto start      = std::chrono::steady_clock::now();
	auto       store_data = std::make_shared<std::vector<uint8_t>>();
	auto&      bytes      = *store_data;
	if (Common::File file;
	    Common::File::IsFileExisting(path) && file.Open(path, Common::File::Mode::Read)) {
		bytes.resize(file.Size());
		uint32_t read = 0;
		file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
		bytes.resize(read);
	}
	const auto read_ms = MillisecondsSince(start);
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: compiling shaders and pipelines from {} ({:.1f} MB) with {} threads\n",
	    Common::PathToString(path), static_cast<double>(bytes.size()) / 1'000'000.0,
	    Common::ParallelThreads()));
	auto records  = std::span<const uint8_t>(bytes);
	bool reusable =
	    records.size() >= header.size() && std::ranges::equal(records.first(header.size()), header);
	if (RestampMode()) {
		// A store of another stamp that agrees in everything else (device profile, format) is
		// loaded as it is; Restamp() gives it this stamp once its modules are verified.
		m_restamp_current = reusable;
		reusable = records.size() >= header.size() &&
		           std::ranges::equal(records.subspan(StampBytes, header.size() - StampBytes),
		                              std::span(header).subspan(StampBytes));
		m_restamp_loaded = reusable && !m_restamp_current;
		if (!m_restamp_loaded) {
			// Nothing to verify, and nothing here may be overwritten.
			return;
		}
		m_restamp_store = store_data;
		m_restamp_path  = path;
		m_restamp_stamp.assign(header.begin(), header.begin() + StampBytes);
		m_restamp_header = header;
	}
	if (reusable) {
		records = records.subspan(header.size());
		// The frames' bounds first, then their contents on several threads, inserted in order.
		std::vector<std::span<const uint8_t>> frames;
		for (auto rest = records;;) {
			Frame frame;
			if (rest.size() < sizeof(frame)) break;
			std::memcpy(&frame, rest.data(), sizeof(frame));
			if (rest.size() - sizeof(frame) < frame.compressed_size) break;
			frames.push_back(rest.first(sizeof(frame) + frame.compressed_size));
			rest = rest.subspan(sizeof(frame) + frame.compressed_size);
		}
		// Only the keys and the pipeline fields here; the resource plans and shader modules, which
		// take nearly all the time, on first use and on the background decoders.
		std::vector<LoadedRecord> loaded(frames.size());
		Common::ParallelFor(frames.size(), [&](size_t i) {
			DecodeRecord(frames[i], loaded[i]);
			if (loaded[i].module.stored != nullptr) {
				loaded[i].module.stored->store = store_data;
			}
		});
		size_t kept = 0;
		std::vector<std::shared_ptr<StoredModule>> stored;

		for (; kept < loaded.size() && loaded[kept].ok; kept++) {
			if (loaded[kept].module.stored != nullptr) {
				stored.push_back(loaded[kept].module.stored);
			}
			if (RestampMode()) {
				// What a partial restamp keeps or drops, record by record.
				RestampRecord record;
				switch (static_cast<Record>(loaded[kept].type)) {
					case Record::Module: {
						record.type      = RestampRecord::Type::Module;
						record.module_id = loaded[kept].module.id;
						Writer key;
						IoKey(key, loaded[kept].key);
						record.program = XXH3_64bits(key.bytes.data(), key.bytes.size());
						break;
					}
					case Record::GraphicsPipeline:
						record.type      = RestampRecord::Type::GraphicsPipeline;
						record.stage_ids = loaded[kept].pipeline_key.stage_ids;
						break;
					case Record::TessellationLayout:
						record.type = RestampRecord::Type::TessellationLayout;
						break;
				}
				m_restamp_records.push_back(record);
				m_restamp_frames.push_back(frames[kept]);
			}
			InsertRecord(std::move(loaded[kept]));
			records = records.subspan(frames[kept].size());
		}
		loaded.clear();
		if (EagerStore()) {
			Common::ParallelFor(stored.size(), [&](size_t i) {
				if (stored[i]->want_plan) ResolveStoredPlan(*stored[i]);
				if (!stored[i]->module_dropped) ResolveStoredModule(*stored[i]);
			});
			if (RetainModuleSpirv()) {
				for (auto& program: m_program_cache | std::views::values) {
					for (auto& module: program.modules) {
						if (module.stored != nullptr) module.spirv = std::move(module.stored->spirv);
					}
				}
			}
			Log::WriteToConsoleAndLog(fmt::format(
			    "Shader cache: record time over all threads: zstd {:.0f} ms, resource plans {:.0f} "
			    "ms, shader modules {:.0f} ms\n",
			    static_cast<double>(g_decode_zstd_ns) / 1e6,
			    static_cast<double>(g_decode_plan_ns) / 1e6,
			    static_cast<double>(g_decode_spirv_ns) / 1e6));
		} else if (!stored.empty() && !PrecompileMode() && !OnDemandShaderStore()) {
			// A precompile decodes only the modules its pipelines need, on their workers.
			StartStoreDecoders(std::move(stored));
		}
	}
	const auto parse_ms = MillisecondsSince(start) - read_ms;
	// A damaged record ends the store; the records before it stay.
	if (!m_store.Open(path, header, reusable ? bytes.size() - records.size() : 0)) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader cache: {} is being written by another process (a precompile or another game of "
		    "this stamp); this run reads it but stores nothing\n",
		    Common::PathToString(path)));
	}

	const auto modules = MapModules();
	// A key with a stage missing from the store keeps no pipeline.
	std::vector<std::pair<const GraphicsPipelineKey*, std::array<const ShaderModule*, 4>>> graphics;
	graphics.reserve(m_graphics_pipelines.size());
	std::erase_if(m_graphics_pipelines, [&](const auto& entry) {
		std::array<const ShaderModule*, 4> stages {};
		for (size_t i = 0; i < stages.size(); i++) {
			const auto found = modules.find(entry.first.stage_ids[i]);
			if (found == modules.end()) return true;
			stages[i] = found->second;
		}
		graphics.emplace_back(&entry.first, stages);
		return false;
	});
	std::vector<const ShaderModule*> compute;
	for (const auto* module: modules | std::views::values) {
		if (module != nullptr && module->info.stage == ShaderType::Compute) {
			compute.push_back(module);
		}
	}
	const bool sync = SyncWarmUp();
	if (RestampMode()) {
		// Nothing is created: a restamp only compares modules.
	} else if (sync) {
		std::vector<Common::UniqueFunction<void>> jobs;
		jobs.reserve(graphics.size() + compute.size());
		for (const auto& [key, stages]: graphics) {
			jobs.emplace_back([this, key, stages] {
				CreatePipelineInternal(m_graphics, m_graphics_pipelines.at(*key), *key, stages,
				                       m_driver_cache);
			});
		}
		// A compute pipeline depends only on its module.
		for (const auto* module: compute) {
			jobs.emplace_back([this, module, &pipeline = m_compute_pipelines[module->id]] {
				CreatePipelineInternal(m_graphics, pipeline, *module, m_driver_cache);
			});
		}
		Common::ShaderCompileStatus::AddQueued(static_cast<int64_t>(jobs.size()));
		Common::ParallelFor(jobs.size(), [&](size_t i) {
			Common::ShaderCompileStatus::RemoveQueued();
			jobs[i]();
		});
		if (!m_graphics_pipelines.empty() || !m_compute_pipelines.empty()) {
			CheckpointDriverCache();
		}
	} else {
		// The layouts here, the pipelines on the workers while the game starts: a draw that
		// needs one not created yet has it created at once (Expedite, PendingPipeline::Get).
		// Compute pipelines go first, since dispatches cannot be skipped.
		auto remaining = std::make_shared<std::atomic<size_t>>(graphics.size() + compute.size());
		const auto counted = [remaining, start](vk::Pipeline pipeline) {
			if (remaining->fetch_sub(1, std::memory_order_acq_rel) == 1) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Shader cache: background pipeline creation finished {:.0f} ms after the "
				    "warm-up started\n",
				    MillisecondsSince(start)));
			}
			return pipeline;
		};
		for (const auto* module: compute) {
			auto& pipeline = m_compute_pipelines[module->id];
			CreatePipelineLayouts(m_graphics, pipeline, {&module, 1});
			StartDeferredCompile(pipeline, [this, counted, layout = pipeline.pipeline_layout, module] {
				return counted(CreateHostPipeline(m_graphics, layout, *module, m_driver_cache));
			});
		}
		for (const auto& [key, stages]: graphics) {
			auto& pipeline = m_graphics_pipelines.at(*key);
			CreatePipelineLayouts(m_graphics, pipeline, stages);
			StartDeferredCompile(pipeline,
			                     [this, counted, layout = pipeline.pipeline_layout, key, stages] {
				                     return counted(CreateHostPipeline(m_graphics, layout, *key,
				                                                       stages, m_driver_cache));
			                     });
		}
	}
	OpenRecipes(path);
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: loaded {} programs, {} modules, {} pipelines in {:.0f} ms (read {:.0f} ms, "
	    "records {:.0f} ms, pipelines {:.0f} ms{}{})\n",
	    m_program_cache.size(), modules.size() - 1,
	    m_graphics_pipelines.size() + m_compute_pipelines.size(), MillisecondsSince(start),
	    read_ms, parse_ms, MillisecondsSince(start) - read_ms - parse_ms,
	    sync ? "" : ", creation continues in the background",
	    m_store_decoders != nullptr ? ", records decode on first use and in the background" : ""));
}

void PipelineCache::DecodeRecord(std::span<const uint8_t> record, LoadedRecord& loaded) {
	Frame frame;
	std::memcpy(&frame, record.data(), sizeof(frame));
	const auto compressed = record.subspan(sizeof(frame));
	if (XXH3_64bits(compressed.data(), compressed.size()) != frame.checksum ||
	    ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != frame.raw_size) {
		return;
	}
	const auto           part = std::chrono::steady_clock::now();
	std::vector<uint8_t> raw(frame.raw_size);
	if (ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size()) != raw.size()) {
		return;
	}
	g_index_zstd_ns += NanosecondsSince(part);
	Reader     reader {raw};
	const auto complete = [&] { return reader.ok && reader.bytes.empty(); };
	const auto offset   = [&] { return static_cast<uint32_t>(raw.size() - reader.bytes.size()); };
	// Skips a stored vector of `element` bytes each, returning its count and offset.
	const auto skip = [&](size_t element, uint32_t& count) {
		IoAll(reader, count);
		const auto at = offset();
		reader.ok     = reader.ok && uint64_t {count} * element <= reader.bytes.size();
		if (reader.ok) reader.bytes = reader.bytes.subspan(count * element);
		return at;
	};
	auto type = Record::Module;
	IoAll(reader, type);
	loaded.type = static_cast<uint8_t>(type);
	switch (type) {
		case Record::Module: {
			auto stored = std::make_shared<StoredModule>();
			IoKey(reader, loaded.key);
			stored->plan_offset = skip(1, stored->plan_size);
			IoModule(reader, loaded.module);
			stored->spirv_offset = skip(sizeof(uint32_t), stored->spirv_words);
			if (!complete() || stored->spirv_words == 0) {
				return;
			}
			stored->frame               = record;
			stored->device              = m_graphics.device;
			loaded.module.spirv_words   = stored->spirv_words;
			loaded.module.stored        = std::move(stored);
			break;
		}
		case Record::GraphicsPipeline: {
			reader.Raw(&loaded.pipeline_key, sizeof(loaded.pipeline_key));
			if (!complete()) {
				return;
			}
			break;
		}
		case Record::TessellationLayout: {
			IoAll(reader, loaded.layout_key, loaded.layout);
			if (!complete()) {
				return;
			}
			break;
		}
		default: return;
	}
	loaded.ok = true;
}

namespace {

// The record's decompressed bytes; the frame was checked and decompressed once at start, so this
// cannot fail.
std::vector<uint8_t> DecompressStored(const StoredModule& stored) {
	Frame frame;
	std::memcpy(&frame, stored.frame.data(), sizeof(frame));
	const auto compressed = stored.frame.subspan(sizeof(frame));
	const auto part       = std::chrono::steady_clock::now();
	std::vector<uint8_t> raw(frame.raw_size);
	EXIT_IF(ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size()) !=
	        raw.size());
	g_decode_zstd_ns += NanosecondsSince(part);
	return raw;
}

void FinishStoredPart(StoredModule& stored) {
	if (stored.parts_left.fetch_sub(1, std::memory_order_acq_rel) == 1) {
		stored.store.reset();
	}
}

} // namespace

vk::ShaderModule ResolveStoredModule(StoredModule& stored) {
	std::call_once(stored.module_once, [&] {
		auto                  raw = DecompressStored(stored);
		std::vector<uint32_t> spirv(stored.spirv_words);
		std::memcpy(spirv.data(), raw.data() + stored.spirv_offset, spirv.size() * sizeof(uint32_t));
		raw             = {};
		const auto part = std::chrono::steady_clock::now();
		stored.handle   = CompileSPV(spirv, stored.device);
		g_decode_spirv_ns += NanosecondsSince(part);
		if (RetainModuleSpirv()) {
			stored.spirv = std::move(spirv);
		}
		stored.decoded.store(true, std::memory_order_release);
		FinishStoredPart(stored);
	});
	return stored.handle;
}

bool ResolveStoredPlan(StoredModule& stored) {
	std::call_once(stored.plan_once, [&] {
		const auto raw  = DecompressStored(stored);
		const auto part = std::chrono::steady_clock::now();
		stored.plan_ok  = ShaderRecompiler::IR::LoadResourcePlan(
            std::span(raw).subspan(stored.plan_offset, stored.plan_size), stored.plan);
		g_decode_plan_ns += NanosecondsSince(part);
		stored.plan_decoded.store(true, std::memory_order_release);
		FinishStoredPart(stored);
	});
	return stored.plan_ok;
}

bool PipelineCache::LazyShaderStore() {
	static const bool lazy = [] {
		const char* env = std::getenv("KYTY_LAZY_SHADER_STORE");
		return env != nullptr && env[0] == '1';
	}();
	return lazy;
}

bool PipelineCache::OnDemandShaderStore() {
	static const bool on_demand = [] {
		const char* env = std::getenv("KYTY_ON_DEMAND_SHADER_STORE");
		return (env != nullptr && env[0] == '1') || LazyShaderStore();
	}();
	return on_demand;
}

bool PipelineCache::TakeStoredPlan(Program& program) {
	const auto stored = std::move(program.stored_plan);
	// How much of the store a run uses: its programs are looked up here once each.
	{
		static const auto start = std::chrono::steady_clock::now();
		static uint32_t   used  = 0;
		if (++used % 100u == 0u) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Shader cache: {} stored programs used, {} of them not decoded yet, {:.0f} s after "
			    "the first ({} programs known)\n",
			    used, g_demand_plans.load(), MillisecondsSince(start) / 1000.0,
			    m_program_cache.size()));
		}
	}
	if (!stored->plan_decoded.load(std::memory_order_acquire)) {
		const auto start = std::chrono::steady_clock::now();
		ResolveStoredPlan(*stored);
		g_demand_plans++;
		g_demand_plan_ns += NanosecondsSince(start);
	}
	if (!ResolveStoredPlan(*stored)) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader cache: a stored resource plan could not be decoded; translating the program "
		    "again\n"));
		return false;
	}
	// Decoded once; no other thread touches the plan any more.
	program.materialization_memo.Clear();
	program.resource_plan = std::move(stored->plan);
	return true;
}

void PipelineCache::InsertRecord(LoadedRecord&& loaded) {
	switch (static_cast<Record>(loaded.type)) {
		case Record::Module: {
			const auto [program, inserted] = m_program_cache.try_emplace(std::move(loaded.key));
			if (inserted) {
				loaded.module.stored->want_plan  = true;
				loaded.module.stored->parts_left = 2;
				program->second.stored_plan     = loaded.module.stored;
			}
			// KYTY_RT_OFF: stored BVH modules walk the scene; they are dropped (their records give
			// only the plan), and their programs translate again with the miss.
			if (loaded.module.info.info.bvh && ShaderRecompiler::Spirv::BvhIntersectAlwaysMisses()) {
				auto& stored          = *loaded.module.stored;
				stored.module_dropped = true;
				if (stored.parts_left.fetch_sub(1) == 1) stored.store.reset();
				break;
			}
			program->second.Adopt(std::move(loaded.module));
			break;
		}
		// WarmUp creates the pipeline once every record is loaded.
		case Record::GraphicsPipeline: m_graphics_pipelines.try_emplace(loaded.pipeline_key); break;
		case Record::TessellationLayout:
			m_tess_layouts.try_emplace(loaded.layout_key, loaded.layout);
			break;
	}
}

// Background decoders: a few low-priority threads decode, in store order (the order a run first
// needed the records in), every record no lookup or pipeline creation decoded first.
struct PipelineCache::StoreDecoders {
	std::vector<std::shared_ptr<StoredModule>> records;
	std::atomic<size_t>                        next = 0;
	std::atomic<bool>                          stop = false;
	std::vector<std::jthread>                  threads;
};

namespace {

void LowerThreadPriority() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
#elif !defined(__APPLE__)
	// Linux: a nice value of one thread.
	setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 19);
#endif
}

size_t StoreDecoderThreads() {
	if (const char* env = std::getenv("KYTY_STORE_DECODE_THREADS"); env != nullptr && env[0] != 0) {
		return std::strtoull(env, nullptr, 10);
	}
	return Common::ParallelThreads();
}

} // namespace

void PipelineCache::StartStoreDecoders(std::vector<std::shared_ptr<StoredModule>>&& records) {
	auto decoders     = std::make_shared<StoreDecoders>();
	decoders->records = std::move(records);
	const auto count  = StoreDecoderThreads();
	auto       left   = std::make_shared<std::atomic<size_t>>(count);
	const auto start  = std::chrono::steady_clock::now();
	// Shown in the shader banner like other shader work.
	Common::ShaderCompileStatus::AddQueued(static_cast<int64_t>(decoders->records.size()));
	for (size_t i = 0; i < count; i++) {
		decoders->threads.emplace_back([state = decoders.get(), left, start] {
			LowerThreadPriority();
			Common::FaultReport::SetThreadTag("store decode", "store-decode");
			Common::FaultReport::RegisterThreadStack();
			while (true) {
				const auto index = state->next++;
				if (index >= state->records.size()) break;
				Common::ShaderCompileStatus::RemoveQueued();
				if (state->stop) continue;
				const Common::ShaderCompileStatus::Work work;
				auto& record = *state->records[index];
				if (record.want_plan && !record.plan_decoded.load(std::memory_order_acquire)) {
					ResolveStoredPlan(record);
				}
				if (!record.module_dropped && !record.decoded.load(std::memory_order_acquire)) {
					ResolveStoredModule(record);
				}
			}
			if (left->fetch_sub(1) == 1 && !state->stop) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Shader cache: all {} stored modules decoded {:.1f} s after decoding "
				    "started (over all threads: zstd {:.0f} ms at start and {:.0f} ms since, "
				    "resource plans {:.0f} ms, shader modules {:.0f} ms); {} plans were decoded "
				    "on first use ({:.0f} ms on the command thread)\n",
				    state->records.size(), MillisecondsSince(start) / 1000.0,
				    static_cast<double>(g_index_zstd_ns) / 1e6,
				    static_cast<double>(g_decode_zstd_ns) / 1e6,
				    static_cast<double>(g_decode_plan_ns) / 1e6,
				    static_cast<double>(g_decode_spirv_ns) / 1e6, g_demand_plans.load(),
				    static_cast<double>(g_demand_plan_ns) / 1e6));
			}
		});
	}
	m_store_decoders = std::move(decoders);
}

void PipelineCache::StopStoreDecoders() {
	if (m_store_decoders != nullptr) {
		m_store_decoders->stop = true;
		m_store_decoders->threads.clear(); // Joins.
	}
}

// Records are encoded here, since programs and modules change later.
void PipelineCache::StoreModule(const ProgramKey& key, const Program& program,
                                const ShaderModule& module, const std::vector<uint32_t>& spirv) {
	// KYTY_RT_OFF modules never enter the store, so runs without it keep the real ones.
	if (!m_store || (module.info.info.bvh && ShaderRecompiler::Spirv::BvhIntersectAlwaysMisses())) {
		return;
	}
	Writer record;
	IoAll(record, Record::Module);
	IoKey(record, key);
	IoAll(record, ShaderRecompiler::IR::SaveResourcePlan(program.resource_plan));
	IoModule(record, module);
	IoAll(record, spirv);
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StorePipeline(const GraphicsPipelineKey& key) {
	RecordRecipePipeline(key);
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::GraphicsPipeline);
	record.Raw(&key, sizeof(key));
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, Record::TessellationLayout, key, layout);
	m_store.Append(std::move(record.bytes));
}

PipelineCache::ModuleMap PipelineCache::MapModules() const {
	ModuleMap modules {{0, nullptr}};
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
			modules.emplace(module.id, &module);
		}
	}
	return modules;
}

bool PipelineCache::CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
                                           const ModuleMap& modules) {
	std::array<const ShaderModule*, 4> stages {};
	for (size_t i = 0; i < stages.size(); i++) {
		const auto found = modules.find(key.stage_ids[i]);
		if (found == modules.end()) {
			return false;
		}
		stages[i] = found->second;
	}
	CreatePipelineInternal(m_graphics, pipeline, key, stages, m_driver_cache);
	return true;
}


// Shader recipes, after the shader recipe pack of bloodborne_pc (deadinside28 / Supermedo,
// GPL-2.0) and the pipeline manifest of AnyPS5 (simnJS, GPL-2.0): `<title>.recipes` beside the
// store holds, for each module the store got, what its translation read (the guest code, the user
// data, the stage input and the specialization) and the graphics pipeline keys. Unlike the store,
// nothing in it depends on the translator, so it is kept when a new stamp discards the store. At
// start, modules the store lacks are translated again on low-priority background threads
// (KYTY_RECIPE_THREADS, default half the hardware threads), keep their ids, and so let the
// recipes' pipeline keys be created in the background as well. KYTY_SHADER_RECIPES=0 turns it off.
// The guest code makes the file private to the dump it came from.
namespace {

enum class RecipeRecord : uint8_t { Code, Variant, Pipeline };

constexpr uint64_t RecipeMagic = 0x3150435259544b59ull; // "YKTYRCP1"

bool RecipesEnabled() {
	static const bool enabled = [] {
		const char* env = std::getenv("KYTY_SHADER_RECIPES");
		return (env == nullptr || env[0] != '0') && !PipelineCache::LazyShaderStore();
	}();
	return enabled;
}

uint64_t RecipeCodeKey(ShaderType stage, uint64_t hash) {
	return hash ^ (uint64_t {static_cast<uint8_t>(stage)} * 0x9e3779b97f4a7c15ull);
}

template <class Info> std::vector<uint8_t> InputBytes(const Info* info) {
	auto copy  = *info;
	copy.stage = {};
	std::vector<uint8_t> bytes(sizeof(copy));
	std::memcpy(bytes.data(), &copy, sizeof(copy));
	return bytes;
}

} // namespace

struct PipelineCache::RecipeJob {
	ProgramKey                                   key;
	std::shared_ptr<const std::vector<uint32_t>> code;
	std::shared_ptr<const std::vector<uint32_t>> back_code;
	std::vector<uint32_t>                        user_data;
	uint32_t                                     wave_size      = 64;
	uint32_t                                     user_data_base = 0;
	ShaderRecompiler::IR::PushData::Window       window;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	uint64_t                                     id = 0;
	// The ids other recipes with the same inputs have (from other translators).
	std::vector<uint64_t>                        other_ids;
	std::vector<uint8_t>                         input;
	ShaderVertexInputInfo                        vertex {};
	ShaderPixelInputInfo                         pixel {};
	ShaderComputeInputInfo                       compute {};
};


void PipelineCache::OpenRecipes(const std::filesystem::path& store_path) {
	if (store_path.empty() || !RecipesEnabled()) {
		return;
	}
	// The recipes do not depend on the translator's switches: every store variant shares them.
	auto name = Common::PathToString(store_path.stem());
	if (const auto variant = ShaderStoreVariant(); !variant.empty() && name.ends_with(variant)) {
		name.resize(name.size() - variant.size());
	}
	const auto path = store_path.parent_path() / (name + ".recipes");
	const auto start = std::chrono::steady_clock::now();
	Writer     header;
	IoAll(header, RecipeMagic);
	std::vector<uint8_t> bytes;
	if (Common::File file;
	    Common::File::IsFileExisting(path) && file.Open(path, Common::File::Mode::Read)) {
		bytes.resize(file.Size());
		uint32_t read = 0;
		file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
		bytes.resize(read);
	}
	auto       records  = std::span<const uint8_t>(bytes);
	const bool reusable = records.size() >= header.bytes.size() &&
	                      std::ranges::equal(records.first(header.bytes.size()), header.bytes);
	std::unordered_map<uint64_t, std::pair<std::shared_ptr<const std::vector<uint32_t>>,
	                                       std::shared_ptr<const std::vector<uint32_t>>>>
	                                        codes;
	std::vector<std::unique_ptr<RecipeJob>> jobs;
	const auto                              known = MapModules();
	// The variants by what they are translated from, in file order. A module's id also depends on
	// what the translator makes of these inputs, so a file that went through several translators
	// may name one variant by several ids: the last one is translated, and whatever id this
	// translator gives it stands for the others in the recipes' pipeline keys.
	struct Variants {
		std::vector<uint64_t>      ids;
		std::unique_ptr<RecipeJob> job;
	};
	std::vector<Variants>                variants;
	std::unordered_map<uint64_t, size_t> variant_index;
	if (reusable) {
		records = records.subspan(header.bytes.size());
		while (records.size() >= sizeof(Frame)) {
			Frame frame;
			std::memcpy(&frame, records.data(), sizeof(frame));
			if (records.size() - sizeof(frame) < frame.compressed_size) break;
			const auto compressed = records.subspan(sizeof(frame), frame.compressed_size);
			if (XXH3_64bits(compressed.data(), compressed.size()) != frame.checksum ||
			    ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != frame.raw_size) {
				break;
			}
			std::vector<uint8_t> raw(frame.raw_size);
			if (ZSTD_decompress(raw.data(), raw.size(), compressed.data(), compressed.size()) !=
			    raw.size()) {
				break;
			}
			Reader reader {raw};
			auto   type = RecipeRecord::Code;
			IoAll(reader, type);
			bool ok = false;
			switch (type) {
				case RecipeRecord::Code: {
					auto                  stage = ShaderType::Unknown;
					uint64_t              hash  = 0;
					std::vector<uint32_t> code;
					std::vector<uint32_t> back_code;
					IoAll(reader, stage, hash, code, back_code);
					if ((ok = reader.ok && reader.bytes.empty())) {
						m_recipe_codes.insert(RecipeCodeKey(stage, hash));
						codes[RecipeCodeKey(stage, hash)] = {
						    std::make_shared<const std::vector<uint32_t>>(std::move(code)),
						    std::make_shared<const std::vector<uint32_t>>(std::move(back_code))};
					}
					break;
				}
				case RecipeRecord::Variant: {
					auto                 job = std::make_unique<RecipeJob>();
					std::vector<uint8_t> input;
					IoKey(reader, job->key);
					IoAll(reader, job->wave_size, job->user_data_base, job->user_data,
					      job->window.begin, job->window.end, job->specialization, job->id, input);
					const auto stage = job->key.stage;
					const auto size  = stage == ShaderType::Pixel     ? sizeof(job->pixel)
					                   : stage == ShaderType::Compute ? sizeof(job->compute)
					                                                  : sizeof(job->vertex);
					if ((ok = reader.ok && reader.bytes.empty())) {
						// A build with other stage inputs keeps the record but cannot use it.
						m_recipe_modules.insert(job->id);
						const auto code = codes.find(RecipeCodeKey(stage, job->key.hash));
						if (input.size() == size && code != codes.end()) {
							std::memcpy(stage == ShaderType::Pixel     ? static_cast<void*>(&job->pixel)
							            : stage == ShaderType::Compute ? static_cast<void*>(&job->compute)
							                                           : static_cast<void*>(&job->vertex),
							            input.data(), size);
							job->code      = code->second.first;
							job->back_code = code->second.second;
							Writer same;
							IoKey(same, job->key);
							IoAll(same, job->wave_size, job->user_data_base, job->user_data,
							      job->window.begin, job->window.end, job->specialization, input);
							job->input = std::move(input);
							const auto [index, first] = variant_index.try_emplace(
							    XXH3_64bits(same.bytes.data(), same.bytes.size()), variants.size());
							if (first) variants.emplace_back();
							auto& variant = variants[index->second];
							if (std::ranges::find(variant.ids, job->id) == variant.ids.end()) {
								variant.ids.push_back(job->id);
							}
							variant.job = std::move(job);
						}
					}
					break;
				}
				case RecipeRecord::Pipeline: {
					GraphicsPipelineKey key;
					reader.Raw(&key, sizeof(key));
					if ((ok = reader.ok && reader.bytes.empty())) {
						m_recipe_pipeline_keys.insert(key);
						if (!m_graphics_pipelines.contains(key)) m_recipe_pipelines.push_back(key);
					}
					break;
				}
				default: break;
			}
			if (!ok) break;
			records = records.subspan(sizeof(frame) + compressed.size());
		}
	}
	std::vector<std::unique_ptr<RecipeJob>> verify;
	for (auto& variant: variants) {
		// A stored id needs no translation and stands for the others.
		uint64_t stored = 0;
		for (const auto id: variant.ids) {
			if (known.contains(id)) stored = id;
		}
		if (RestampMode()) {
			if (stored != 0) {
				variant.job->id = stored;
				verify.push_back(std::move(variant.job));
			}
			continue;
		}
		if (stored != 0) {
			for (const auto id: variant.ids) {
				if (id != stored) m_recipe_aliases.insert_or_assign(id, stored);
			}
			continue;
		}
		for (const auto id: variant.ids) {
			if (id != variant.job->id) variant.job->other_ids.push_back(id);
		}
		jobs.push_back(std::move(variant.job));
	}
	if (!m_recipes.Open(path, header.bytes, reusable ? bytes.size() - records.size() : 0)) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader recipes: {} is being written by another process; this run records nothing\n",
		    Common::PathToString(path)));
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader recipes: {} programs, {} modules, {} pipelines in {} ({:.1f} MB, {:.0f} ms); "
	    "{} modules to translate again\n",
	    m_recipe_codes.size(), m_recipe_modules.size(), m_recipe_pipeline_keys.size(),
	    Common::PathToString(path), static_cast<double>(bytes.size()) / 1'000'000.0,
	    MillisecondsSince(start), jobs.size()));
	if (RestampMode()) {
		// Only the stored modules are translated, to compare; nothing is added to the store.
		m_recipe_pipelines.clear();
		m_restamp_modules = known;
		m_restamp_stored  = known.size() - 1;
		std::unordered_set<uint64_t> covered;
		for (const auto& job: verify) covered.insert(job->id);
		m_restamp_uncovered = m_restamp_stored - covered.size();
		// Every module, or a sample spread evenly over the file.
		const auto   sample = g_restamp_sample.load();
		const size_t step   = sample == 0 || sample >= verify.size() ? 1 : verify.size() / sample;
		for (size_t i = 0; i < verify.size(); i += step) {
			m_replay_left++;
			m_replay_total++;
			m_workers.Post([this, job = std::shared_ptr<RecipeJob>(std::move(verify[i]))] {
				VerifyRecipe(*job);
			});
		}
		return;
	}
	if (jobs.empty()) {
		// Pipelines an interrupted run left, or whose stages a stored module stands for.
		CreateRecipePipelines();
		m_replay_lacking = m_recipe_pipelines.size();
		m_recipe_pipelines.clear();
		return;
	}
	const auto threads = [] {
		const char* env = std::getenv("KYTY_RECIPE_THREADS");
		const auto  value = env != nullptr ? std::strtoul(env, nullptr, 10) : 0ul;
		return value != 0 ? size_t {value} : Common::ParallelThreads();
	}();
	m_replay_start  = std::chrono::steady_clock::now();
	m_replay_left   = jobs.size();
	m_replay_total  = jobs.size();
	m_replay_active = true;
	// A precompile has no game to leave threads to: the pipeline workers translate, and create
	// pipelines (their background queue) when no translation waits.
	if (!PrecompileMode()) {
		m_replayers = std::make_unique<PipelineWorkers>(threads);
	}
	auto& workers = PrecompileMode() ? m_workers : *m_replayers;
	for (auto& job: jobs) {
		workers.Post([this, job = std::shared_ptr<RecipeJob>(std::move(job))] {
			ReplayRecipe(*job);
		});
	}
}

// Translates a recipe as the command thread did when it recorded it; false if that fails.
bool PipelineCache::TranslateRecipe(RecipeJob& job, ReplayedModule& replayed) {
	const Common::ShaderCompileStatus::Work work;
	const auto                              profile = SpirvProfile(m_graphics);
	ShaderRecompiler::CompileOptions        options {.wave_size = job.wave_size};
	options.stage          = job.key.stage;
	options.user_data_base = job.user_data_base;
	options.shader_hash    = job.key.hash;
	options.dump_ir        = false;
	options.profile        = &profile;
	options.user_data      = job.user_data;
	options.back_code      = *job.back_code;
	if (job.key.stage == ShaderType::Pixel) {
		options.input_info.pixel = &job.pixel;
	} else if (job.key.stage == ShaderType::Compute) {
		options.input_info.compute = &job.compute;
	} else {
		options.input_info.vertex = &job.vertex;
	}
	volatile bool  usable = false;
	std::jmp_buf   recover;
	// Like a translation ahead, a failure ends this job only (and leaks what it held).
	if (setjmp(recover) == 0) {
		Common::g_dbg_recover = &recover;
		auto translated = ShaderRecompiler::TranslateProgram(*job.code, options);
		replayed.plan   = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		auto result     = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                                   job.specialization, job.window);
		auto info       = std::move(result.program).TakeCompiledInfo();
		// The id CompileModule gives these inputs. Another translator may have given another
		// one: the module is the one a lookup with these inputs finds all the same, and its id
		// replaces the recorded ones in the recipes' pipeline keys.
		Writer inputs;
		IoAll(inputs, job.key.stage, job.key.hash, job.key.user_data_count, job.key.static_state,
		      info.bindings.push_data_start_dword, job.specialization);
		const auto id    = XXH3_64bits(inputs.bytes.data(), inputs.bytes.size());
		replayed.old_ids = job.other_ids;
		if (id != job.id) {
			replayed.old_ids.push_back(job.id);
			Writer record;
			IoAll(record, RecipeRecord::Variant);
			IoKey(record, job.key);
			IoAll(record, job.wave_size, job.user_data_base, job.user_data, job.window.begin,
			      job.window.end, job.specialization, id, job.input);
			replayed.recipe = std::move(record.bytes);
		}
		replayed.module = ShaderModule {.id             = id,
		                                .info           = std::move(info),
		                                .specialization = job.specialization};
		if (job.key.stage == ShaderType::Pixel &&
		    job.pixel.parameter_mode == ShaderPixelParameterMode::Rectangle) {
			replayed.module.rect_parameters = RectListParameters(job.pixel, replayed.module.info);
		} else if (job.key.stage == ShaderType::Mesh) {
			replayed.module.mesh_task = job.vertex.indirect_draw;
		}
		replayed.spirv = std::move(result.spirv);
		usable         = true;
		Common::g_dbg_recover = nullptr;
	}
	return usable;
}

void PipelineCache::VerifyRecipe(RecipeJob& job) {
	if (!m_ahead_stop) {
		ReplayedModule replayed {.key = job.key};
		const bool     usable = TranslateRecipe(job, replayed);
		const auto     stored = m_restamp_modules.find(job.id);
		bool           same   = false;
		// The record as StoreModule would write it, apart from the resource plan: program key,
		// module fields and SPIR-V. And the plan on its own (SelectRestampRecords says when it
		// matters).
		bool           module_same = false;
		bool           plan_same   = false;
		if (usable && replayed.module.id == job.id && stored != m_restamp_modules.end() &&
		    stored->second->stored != nullptr &&
		    stored->second->stored->spirv_words == replayed.spirv.size()) {
			const auto& record = *stored->second->stored;
			const auto  raw    = DecompressStored(record);
			same = std::memcmp(raw.data() + record.spirv_offset, replayed.spirv.data(),
			                   replayed.spirv.size() * sizeof(uint32_t)) == 0;
			if (same) {
				Writer head;
				IoAll(head, Record::Module);
				IoKey(head, job.key);
				Writer plan;
				IoAll(plan, ShaderRecompiler::IR::SaveResourcePlan(replayed.plan));
				Writer fields;
				IoModule(fields, replayed.module);
				// The stored record's parts: [head][plan count, plan][fields][SPIR-V count, SPIR-V].
				const size_t plan_begin = record.plan_offset - sizeof(uint32_t);
				const size_t plan_end   = size_t {record.plan_offset} + record.plan_size;
				const size_t spirv_head = record.spirv_offset - sizeof(uint32_t);
				const auto   part_equal = [&](size_t begin, size_t end,
				                            const std::vector<uint8_t>& bytes) {
					return begin <= end && end <= raw.size() && end - begin == bytes.size() &&
					       std::memcmp(raw.data() + begin, bytes.data(), bytes.size()) == 0;
				};
				module_same = part_equal(0, plan_begin, head.bytes) &&
				              part_equal(plan_end, spirv_head, fields.bytes);
				plan_same   = module_same && part_equal(plan_begin, plan_end, plan.bytes);
			}
		}
		std::scoped_lock lock(m_replay_mutex);
		m_restamp_verified++;
		if (module_same) {
			m_restamp_identical.insert(job.id);
			if (plan_same) {
				m_restamp_plan_identical.insert(job.id);
			} else {
				m_restamp_plan_differing++;
			}
		} else if (same) {
			m_restamp_spirv_only++;
		}
		if (!usable) m_restamp_failed++;
		if (!same || (!module_same && PartialRestamp())) {
			if (!same) m_restamp_differing++;
			if (m_replay_failures.size() < 16) {
				m_replay_failures.push_back(fmt::format(
				    "{} 0x{:016x} module 0x{:016x}: {}", ShaderRecompiler::StageName(job.key.stage),
				    job.key.hash, job.id,
				    !usable                          ? "translation failed"
				    : replayed.module.id != job.id ? "another module id"
				    : !same                          ? "other SPIR-V"
				                                     : "same SPIR-V, other module fields"));
			}
		}
	}
	m_replay_left--;
}

void PipelineCache::ReplayRecipe(RecipeJob& job) {
	if (m_ahead_stop) {
		return;
	}
	// The game's own translations go first: a program it needs now is translated on the command
	// thread at normal priority.
	if (!PrecompileMode()) {
		static thread_local const bool lowered = (LowerThreadPriority(), true);
		(void)lowered;
	}
	ReplayedModule replayed {.key = job.key};
	const bool     usable = TranslateRecipe(job, replayed);
	{
		std::scoped_lock lock(m_replay_mutex);
		if (usable) {
			m_replayed.push_back(std::move(replayed));
		} else {
			m_replay_dropped++;
			if (m_replay_failures.size() < 16) {
				m_replay_failures.push_back(
				    fmt::format("{} 0x{:016x} module 0x{:016x}",
				                ShaderRecompiler::StageName(job.key.stage), job.key.hash, job.id));
			}
		}
	}
	m_replay_left--;
	m_replay_ready = true;
}

void PipelineCache::CreateRecipePipelines() {
	if (m_recipe_pipelines.empty()) {
		return;
	}
	// The pipelines whose stages now all exist.
	const auto modules = MapModules();
	std::erase_if(m_recipe_pipelines, [&](const GraphicsPipelineKey& recorded) {
		auto                               key = recorded;
		std::array<const ShaderModule*, 4> stages {};
		for (size_t i = 0; i < stages.size(); i++) {
			if (const auto alias = m_recipe_aliases.find(key.stage_ids[i]);
			    alias != m_recipe_aliases.end()) {
				key.stage_ids[i] = alias->second;
			}
			const auto found = modules.find(key.stage_ids[i]);
			if (found == modules.end()) return false;
			stages[i] = found->second;
		}
		auto [iter, inserted] = m_graphics_pipelines.try_emplace(key);
		if (inserted) {
			auto& pipeline = iter->second;
			CreatePipelineLayouts(m_graphics, pipeline, stages);
			StartDeferredCompile(pipeline,
			                     [this, layout = pipeline.pipeline_layout, &key = iter->first,
			                      stages] {
				                     return CreateHostPipeline(m_graphics, layout, key, stages,
				                                               m_driver_cache);
			                     });
			StorePipeline(iter->first);
		}
		return true;
	});
}

void PipelineCache::MergeReplayed() {
	if (!m_replay_ready.exchange(false)) {
		return;
	}
	// Read first: when it is zero, every result is in m_replayed already.
	const bool                  finished = m_replay_left == 0;
	std::vector<ReplayedModule> replayed;
	size_t                      dropped = 0;
	{
		std::scoped_lock lock(m_replay_mutex);
		replayed.swap(m_replayed);
		dropped = m_replay_dropped;
	}
	for (auto& done: replayed) {
		auto [entry, inserted] = m_program_cache.try_emplace(done.key);
		auto& program          = entry->second;
		if (inserted) {
			program.resource_plan = std::move(done.plan);
		} else if (program.call_source != nullptr || program.skip_dispatch) {
			continue;
		}
		for (const auto old: done.old_ids) {
			if (old != done.module.id) m_recipe_aliases.insert_or_assign(old, done.module.id);
		}
		if (!done.recipe.empty()) {
			m_replay_renamed++;
			if (m_recipes && m_recipe_modules.insert(done.module.id).second) {
				m_recipes.Append(std::move(done.recipe));
			}
		}
		if (std::ranges::any_of(program.modules, [&](const ShaderModule& module) {
			    return module.id == done.module.id;
		    })) {
			continue;
		}
		done.module.module = CompileSPV(done.spirv, m_graphics.device);
		RetainModuleSpirv(done.module, done.spirv);
		const auto& module = program.Adopt(std::move(done.module));
		StoreModule(entry->first, program, module, done.spirv);
		m_replay_merged++;
		if (module.info.stage == ShaderType::Compute && !m_compute_pipelines.contains(module.id)) {
			auto&       pipeline = m_compute_pipelines[module.id];
			const auto* stage    = &module;
			CreatePipelineLayouts(m_graphics, pipeline, {&stage, 1});
			StartDeferredCompile(pipeline, [this, layout = pipeline.pipeline_layout, stage] {
				return CreateHostPipeline(m_graphics, layout, *stage, m_driver_cache);
			});
		}
	}
	CreateRecipePipelines();
	if (finished && m_replay_active) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader recipes: translated {} modules again in {:.1f} s ({} unusable); {} pipelines "
		    "wait for stages the recipes lacked\n",
		    m_replay_merged, MillisecondsSince(m_replay_start) / 1000.0, dropped,
		    m_recipe_pipelines.size()));
		m_replay_lacking = m_recipe_pipelines.size();
		m_replay_active  = false;
		m_recipe_pipelines.clear();
		if (m_replayers != nullptr) m_replayers->Finish();
	}
}

void PipelineCache::SetPrecompileMode() {
	g_precompile_mode = true;
}

bool PipelineCache::Precompile() {
	using Clock      = std::chrono::steady_clock;
	const auto start = Clock::now();
	if (!m_store) {
		Log::WriteToConsoleAndLog(
		    "Shader precompile: no shader store (a Release build and a title id are needed, and no "
		    "other process may be writing it)\n");
		return false;
	}
	g_precompile_interrupted = 0;
	const auto interrupt     = [](int) { g_precompile_interrupted = 1; };
	std::signal(SIGINT, interrupt);
	std::signal(SIGTERM, interrupt);
	const auto ready = [](const auto& pipelines) {
		return static_cast<size_t>(std::ranges::count_if(pipelines | std::views::values,
		                                                 [](const Pipeline& pipeline) {
			return !pipeline.compiling.valid() ||
			       pipeline.compiling.wait_for(std::chrono::seconds {0}) ==
			           std::future_status::ready;
		}));
	};
	const auto stored_modules = MapModules().size() - 1;
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader precompile: {} modules to translate, {} modules in the store, {} pipelines "
	    "known, {} threads\n",
	    m_replay_total, stored_modules, m_graphics_pipelines.size() + m_compute_pipelines.size(),
	    Common::ParallelThreads()));
	double translate_s = 0.0;
	auto   next_report = start + std::chrono::seconds {5};
	bool   interrupted = false;
	while (true) {
		MergeReplayed();
		const bool translated = !m_replay_active && m_replay_left == 0;
		if (translated && translate_s == 0.0) translate_s = MillisecondsSince(start) / 1000.0;
		const auto total = m_graphics_pipelines.size() + m_compute_pipelines.size();
		const auto done  = ready(m_graphics_pipelines) + ready(m_compute_pipelines);
		if (translated && done == total) break;
		if (g_precompile_interrupted != 0) {
			interrupted = true;
			break;
		}
		if (const auto now = Clock::now(); now >= next_report) {
			next_report = now + std::chrono::seconds {5};
			Log::WriteToConsoleAndLog(fmt::format(
			    "Shader precompile: {:.0f} s: modules {}/{}, pipelines {}/{}\n",
			    MillisecondsSince(start) / 1000.0, m_replay_total - m_replay_left, m_replay_total,
			    done, total));
			std::fflush(stdout);
			// Writes the driver cache once per interval, so an interrupted run keeps most of it.
			if (done != 0) CheckpointDriverCache();
		}
		std::this_thread::sleep_for(std::chrono::milliseconds {100});
	}
	if (interrupted) {
		Log::WriteToConsoleAndLog("Shader precompile: interrupted; saving what is done\n");
		// Queued translations and pipelines are dropped, running ones end.
		m_ahead_stop     = true;
		*m_deferred_stop = true;
		m_workers.Finish();
		MergeReplayed();
	}
	size_t created = 0;
	size_t failed  = 0;
	const auto count = [&](auto& pipelines) {
		for (auto& pipeline: pipelines | std::views::values) {
			(Resolve(pipeline, !interrupted).pipeline != nullptr ? created : failed)++;
		}
	};
	count(m_graphics_pipelines);
	count(m_compute_pipelines);
	// Dropped ones are not failures.
	if (interrupted) failed = 0;
	if (m_driver_cache != nullptr) m_driver_cache_dirty = true;
	Save();
	PipelineBinaryStore::Get().Close();
	size_t                   dropped = 0;
	std::vector<std::string> failures;
	{
		std::scoped_lock lock(m_replay_mutex);
		dropped  = m_replay_dropped;
		failures = m_replay_failures;
	}
	for (const auto& failure: failures) {
		Log::WriteToConsoleAndLog(
		    fmt::format("Shader precompile: translation failed: {}\n", failure));
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader precompile: {} in {:.1f} s ({:.1f} s translating): {} modules translated ({} got "
	    "a new id), {} failed to translate, {} modules were stored before; {} pipelines created "
	    "of {} graphics and {} compute, {} failed, {} recipe pipelines lack a stage\n",
	    interrupted ? "interrupted" : "done", MillisecondsSince(start) / 1000.0, translate_s,
	    m_replay_merged, m_replay_renamed, dropped, stored_modules, created,
	    m_graphics_pipelines.size(), m_compute_pipelines.size(), failed, m_replay_lacking));
	return !interrupted && failed == 0 && dropped == 0;
}

void PipelineCache::SetRestampMode(size_t sample, bool strict) {
	g_precompile_mode = true;
	g_restamp_mode    = true;
	g_restamp_sample  = sample;
	g_restamp_strict  = strict;
}

int PipelineCache::Restamp() {
	const auto start = std::chrono::steady_clock::now();
	if (m_restamp_current) {
		Log::WriteToConsoleAndLog("Shader restamp: the store has this stamp already\n");
		return 0;
	}
	if (!m_restamp_loaded) {
		Log::WriteToConsoleAndLog(
		    "Shader restamp: refused: no store, or one written for another device profile or "
		    "store format\n");
		return 4;
	}
	if (!m_store) {
		Log::WriteToConsoleAndLog(
		    "Shader restamp: refused: another process is writing the store\n");
		return 4;
	}
	g_precompile_interrupted = 0;
	const auto interrupt     = [](int) { g_precompile_interrupted = 1; };
	std::signal(SIGINT, interrupt);
	std::signal(SIGTERM, interrupt);
	const bool partial = PartialRestamp();
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader restamp: translating {} of {} stored modules again to compare, {} threads, {}\n",
	    m_replay_total, m_restamp_stored, Common::ParallelThreads(),
	    partial ? "partial (keeps the identical ones)" : "strict (all or nothing)"));
	auto next_report = start + std::chrono::seconds {5};
	while (m_replay_left != 0 && g_precompile_interrupted == 0) {
		if (const auto now = std::chrono::steady_clock::now(); now >= next_report) {
			next_report = now + std::chrono::seconds {5};
			Log::WriteToConsoleAndLog(fmt::format("Shader restamp: {:.0f} s: modules {}/{}\n",
			                                      MillisecondsSince(start) / 1000.0,
			                                      m_replay_total - m_replay_left, m_replay_total));
			std::fflush(stdout);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds {100});
	}
	const bool interrupted = m_replay_left != 0;
	// Queued translations and pipelines are dropped, running ones end.
	m_ahead_stop     = true;
	*m_deferred_stop = true;
	m_workers.Finish();
	m_store.Finish();
	m_recipes.Finish();
	PipelineBinaryStore::Get().Close();
	for (const auto& failure: m_replay_failures) {
		Log::WriteToConsoleAndLog(fmt::format("Shader restamp: differs: {}\n", failure));
	}
	if (partial) {
		return RestampPartial(start, interrupted);
	}
	// With every module asked for, one without a recipe cannot be vouched for either.
	const bool all     = g_restamp_sample == 0;
	const bool refused = interrupted || m_restamp_differing != 0 || m_restamp_verified == 0 ||
	                     (all && m_restamp_uncovered != 0);
	bool written = false;
	if (!refused) {
		if (Common::File file; file.Open(m_restamp_path, Common::File::Mode::ReadWrite)) {
			uint32_t count = 0;
			file.Seek(0);
			file.Write(m_restamp_stamp.data(), static_cast<uint32_t>(m_restamp_stamp.size()),
			           &count);
			written = count == m_restamp_stamp.size() && file.Flush();
		}
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader restamp: {} in {:.1f} s: {} of {} stored modules translated again, {} differ, {} "
	    "have no recipe\n",
	    interrupted ? "interrupted, store left as it was"
	    : refused   ? "refused, store left as it was"
	    : written   ? "done, the store now has this stamp"
	                : "failed to write the store",
	    MillisecondsSince(start) / 1000.0, m_restamp_verified, m_restamp_stored,
	    m_restamp_differing, m_restamp_uncovered));
	return interrupted ? 1 : refused ? 4 : written ? 0 : 1;
}

int PipelineCache::RestampPartial(std::chrono::steady_clock::time_point start, bool interrupted) {
	if (interrupted) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader restamp: interrupted, store left as it was, in {:.1f} s\n",
		    MillisecondsSince(start) / 1000.0));
		return 1;
	}
	const auto selection =
	    SelectRestampRecords(m_restamp_records, m_restamp_identical, m_restamp_plan_identical);
	const auto dropped   = static_cast<size_t>(std::ranges::count(selection.keep, false));
	bool       written   = false;
	const char* outcome  = nullptr;
	if (selection.modules_kept == 0) {
		outcome = "refused (nothing to keep), store left as it was";
	} else if (dropped == 0) {
		// Everything verified: only the stamp changes, in place.
		if (Common::File file; file.Open(m_restamp_path, Common::File::Mode::ReadWrite)) {
			uint32_t count = 0;
			file.Seek(0);
			file.Write(m_restamp_stamp.data(), static_cast<uint32_t>(m_restamp_stamp.size()),
			           &count);
			written = count == m_restamp_stamp.size() && file.Flush();
		}
		outcome = written ? "done, every record kept, the store now has this stamp"
		                  : "failed to write the store";
	} else {
		// The kept records under this build's header, written beside the store and moved over it.
		// The lock stays held until the process ends, so no other writer opens the old file.
		m_store.CloseFile();
		auto temp = m_restamp_path;
		temp += ".restamp";
		if (Common::File file; file.Create(temp)) {
			const auto write = [&](std::span<const uint8_t> bytes) {
				uint32_t count = 0;
				file.Write(bytes.data(), static_cast<uint32_t>(bytes.size()), &count);
				return count == bytes.size();
			};
			written = write(m_restamp_header);
			for (size_t i = 0; written && i < m_restamp_frames.size(); i++) {
				if (selection.keep[i]) written = write(m_restamp_frames[i]);
			}
			written = written && file.Flush();
			file.Close();
		}
		std::error_code error;
		if (written) {
			std::filesystem::rename(temp, m_restamp_path, error);
			written = !error;
		}
		if (!written) std::filesystem::remove(temp, error);
		outcome = written ? "done, the store now has this stamp" : "failed to write the store";
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader restamp: {} in {:.1f} s: {} of {} stored modules translated again; modules: {} "
	    "kept, {} dropped ({} other SPIR-V, {} same SPIR-V but other module fields, {} failed to "
	    "translate, {} have no recipe, {} not translated (sample), {} duplicate records, {} give "
	    "their program a plan this build extracts differently; {} kept records carry another plan "
	    "that no load reads); pipelines: {} kept, {} dropped; {} tessellation layouts dropped\n",
	    outcome, MillisecondsSince(start) / 1000.0, m_restamp_verified, m_restamp_stored,
	    selection.modules_kept,
	    selection.modules_dropped + selection.modules_duplicate + selection.modules_plan_dropped,
	    m_restamp_differing - m_restamp_failed, m_restamp_spirv_only, m_restamp_failed,
	    m_restamp_uncovered,
	    m_restamp_stored - m_restamp_uncovered - std::min(m_restamp_verified, m_restamp_stored - m_restamp_uncovered),
	    selection.modules_duplicate, selection.modules_plan_dropped,
	    m_restamp_plan_differing - std::min(m_restamp_plan_differing, selection.modules_plan_dropped),
	    selection.pipelines_kept,
	    selection.pipelines_dropped, selection.layouts_dropped));
	return selection.modules_kept == 0 ? 4 : written ? 0 : 1;
}

void PipelineCache::RecordRecipe(const ProgramKey& key, const ShaderParams& params,
                                 const ShaderRecompiler::CompileOptions& options,
                                 ShaderRecompiler::IR::PushData::Window  push_data,
                                 const ShaderModule&                     module) {
	if (!m_recipes || !m_recipe_modules.insert(module.id).second) {
		return;
	}
	if (m_recipe_codes.insert(RecipeCodeKey(key.stage, key.hash)).second) {
		Writer record;
		IoAll(record, RecipeRecord::Code, key.stage, key.hash,
		      std::vector<uint32_t>(params.code.begin(), params.code.end()),
		      std::vector<uint32_t>(params.back_code.begin(), params.back_code.end()));
		m_recipes.Append(std::move(record.bytes));
	}
	std::vector<uint8_t> input;
	if (key.stage == ShaderType::Pixel) {
		input = InputBytes(options.input_info.pixel);
	} else if (key.stage == ShaderType::Compute) {
		input = InputBytes(options.input_info.compute);
	} else {
		input = InputBytes(options.input_info.vertex);
	}
	Writer record;
	IoAll(record, RecipeRecord::Variant);
	IoKey(record, key);
	IoAll(record, options.wave_size, options.user_data_base,
	      std::vector<uint32_t>(options.user_data.begin(), options.user_data.end()), push_data.begin,
	      push_data.end, module.specialization, module.id, input);
	m_recipes.Append(std::move(record.bytes));
}

void PipelineCache::RecordRecipePipeline(const GraphicsPipelineKey& key) {
	if (!m_recipes || !m_recipe_pipeline_keys.insert(key).second) {
		return;
	}
	Writer record;
	IoAll(record, RecipeRecord::Pipeline);
	record.Raw(&key, sizeof(key));
	m_recipes.Append(std::move(record.bytes));
}

} // namespace Libs::Graphics
