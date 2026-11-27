#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/shaderCompileStatus.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"

#include <csetjmp>
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

// What every record depends on besides its own fields; empty when the build is unknown.
std::vector<uint8_t> StoreHeader(const GraphicContext& graphics) {
	const auto stamp = BuildStamp();
	if (stamp == 0) {
		return {};
	}
	// Ballot guards change the SPIR-V of most modules but are not part of Profile (whose bytes
	// stay those of stores written without them): a different stamp keeps the two apart.
	constexpr uint64_t BallotGuardsStamp = 0x62616c6c6f740001ull;
	const uint64_t     keyed_stamp =
	    ShaderRecompiler::Spirv::BallotGuardsEnabled() ? stamp ^ BallotGuardsStamp : stamp;
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
	       spirv, stored] = stage;
	IoAll(io, id, info, specialization, rect_parameters, mesh_task);
}

} // namespace

void ShaderStoreWriter::Open(const std::filesystem::path& path, std::span<const uint8_t> header,
                             uint64_t kept) {
	Common::File::CreateDirectories(path.parent_path());
	if (kept == 0) {
		if (m_file.Create(path)) {
			m_file.Write(header.data(), static_cast<uint32_t>(header.size()));
		}
	} else if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
		m_file.Truncate(kept);
		m_file.Seek(kept);
	}
}

ShaderStoreWriter::ShaderStoreWriter() {
	m_thread = std::jthread([this](std::stop_token stop) {
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
	auto       records = std::span<const uint8_t>(bytes);
	const bool reusable =
	    records.size() >= header.size() && std::ranges::equal(records.first(header.size()), header);
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
		} else if (!stored.empty()) {
			StartStoreDecoders(std::move(stored));
		}
	}
	const auto parse_ms = MillisecondsSince(start) - read_ms;
	// A damaged record ends the store; the records before it stay.
	m_store.Open(path, header, reusable ? bytes.size() - records.size() : 0);

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
	if (sync) {
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

bool PipelineCache::TakeStoredPlan(Program& program) {
	const auto stored = std::move(program.stored_plan);
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
			program->second.modules.push_back(std::move(loaded.module));
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
		return env == nullptr || env[0] != '0';
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
	ShaderVertexInputInfo                        vertex {};
	ShaderPixelInputInfo                         pixel {};
	ShaderComputeInputInfo                       compute {};
};


void PipelineCache::OpenRecipes(const std::filesystem::path& store_path) {
	if (store_path.empty() || !RecipesEnabled()) {
		return;
	}
	auto path = store_path;
	path.replace_extension(".recipes");
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
						if (input.size() == size && code != codes.end() && !known.contains(job->id)) {
							std::memcpy(stage == ShaderType::Pixel     ? static_cast<void*>(&job->pixel)
							            : stage == ShaderType::Compute ? static_cast<void*>(&job->compute)
							                                           : static_cast<void*>(&job->vertex),
							            input.data(), size);
							job->code      = code->second.first;
							job->back_code = code->second.second;
							jobs.push_back(std::move(job));
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
	m_recipes.Open(path, header.bytes, reusable ? bytes.size() - records.size() : 0);
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader recipes: {} programs, {} modules, {} pipelines in {} ({:.1f} MB, {:.0f} ms); "
	    "{} modules to translate again\n",
	    m_recipe_codes.size(), m_recipe_modules.size(), m_recipe_pipeline_keys.size(),
	    Common::PathToString(path), static_cast<double>(bytes.size()) / 1'000'000.0,
	    MillisecondsSince(start), jobs.size()));
	if (jobs.empty()) {
		m_recipe_pipelines.clear();
		return;
	}
	const auto threads = [] {
		const char* env = std::getenv("KYTY_RECIPE_THREADS");
		const auto  value = env != nullptr ? std::strtoul(env, nullptr, 10) : 0ul;
		return value != 0 ? size_t {value} : Common::ParallelThreads();
	}();
	m_replay_start = std::chrono::steady_clock::now();
	m_replay_left  = jobs.size();
	m_replayers    = std::make_unique<PipelineWorkers>(threads);
	for (auto& job: jobs) {
		m_replayers->Post([this, job = std::shared_ptr<RecipeJob>(std::move(job))] {
			ReplayRecipe(*job);
		});
	}
}

void PipelineCache::ReplayRecipe(RecipeJob& job) {
	if (m_ahead_stop) {
		return;
	}
	// The game's own translations go first: a program it needs now is translated on the command
	// thread at normal priority.
	static thread_local const bool lowered = (LowerThreadPriority(), true);
	(void)lowered;
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
	ReplayedModule replayed {.key = job.key};
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
		// The id CompileModule gives these inputs; another one names no stored pipeline.
		Writer inputs;
		IoAll(inputs, job.key.stage, job.key.hash, job.key.user_data_count, job.key.static_state,
		      info.bindings.push_data_start_dword, job.specialization);
		if (XXH3_64bits(inputs.bytes.data(), inputs.bytes.size()) == job.id) {
			replayed.module = ShaderModule {.id             = job.id,
			                                .info           = std::move(info),
			                                .specialization = job.specialization};
			if (job.key.stage == ShaderType::Pixel &&
			    job.pixel.parameter_mode == ShaderPixelParameterMode::Rectangle) {
				replayed.module.rect_parameters =
				    RectListParameters(job.pixel, replayed.module.info);
			} else if (job.key.stage == ShaderType::Mesh) {
				replayed.module.mesh_task = job.vertex.indirect_draw;
			}
			replayed.spirv = std::move(result.spirv);
			usable         = true;
		}
		Common::g_dbg_recover = nullptr;
	}
	{
		std::scoped_lock lock(m_replay_mutex);
		if (usable) {
			m_replayed.push_back(std::move(replayed));
		} else {
			m_replay_dropped++;
		}
	}
	m_replay_left--;
	m_replay_ready = true;
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
		} else if (program.call_source != nullptr || program.skip_dispatch ||
		           std::ranges::any_of(program.modules, [&](const ShaderModule& module) {
			           return module.id == done.module.id;
		           })) {
			continue;
		}
		done.module.module = CompileSPV(done.spirv, m_graphics.device);
		RetainModuleSpirv(done.module, done.spirv);
		const auto& module = program.modules.emplace_back(std::move(done.module));
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
	// The pipelines whose stages now all exist.
	const auto modules = MapModules();
	std::erase_if(m_recipe_pipelines, [&](const GraphicsPipelineKey& key) {
		std::array<const ShaderModule*, 4> stages {};
		for (size_t i = 0; i < stages.size(); i++) {
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
	if (finished && m_replayers != nullptr) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Shader recipes: translated {} modules again in {:.1f} s ({} unusable); {} pipelines "
		    "wait for stages the recipes lacked\n",
		    m_replay_merged, MillisecondsSince(m_replay_start) / 1000.0, dropped,
		    m_recipe_pipelines.size()));
		m_recipe_pipelines.clear();
		m_replayers->Finish();
	}
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
