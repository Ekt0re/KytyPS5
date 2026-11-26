#include "graphics/frameCapture.h"

#include "common/common.h"
#include "common/compilePause.h"
#include "common/threads.h"
#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/timer.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/regionDefinitions.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/shader/shader.h"
#include "kernel/fileSystem.h"
#include "kernel/memory.h"
#include "libs/agc.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <fmt/format.h>
#include <xxhash.h>
#include <zstd.h>

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <sys/prctl.h>
#include <sys/uio.h>
#include <unistd.h>
#endif

#define STB_IMAGE_WRITE_STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

namespace Libs::Graphics::FrameCapture {

namespace {

namespace Memory = LibKernel::Memory;

constexpr uint32_t CaptureVersion = 1;
constexpr uint64_t PageBytes      = 4096;
constexpr uint64_t ChunkBytes     = uint64_t {1} << 20u;
constexpr int      ZstdLevel      = 1;
// GuestGpu queue ids: 0 graphics, 1 + n compute queue 0x20 + n.
constexpr uint32_t ComputeQueueBase = 0x20;

enum class CallKind : uint32_t {
	Memory = 1,
	Submit,
	ReleaseRewind,
	SuspendPoint,
	VideoOut,
	FrameEnd,
	Start,
	End,
};

constexpr uint32_t CallMagic = 0x4c41434b; // "KCAL"

#pragma pack(push, 1)
struct CallHeader {
	uint32_t magic;
	uint32_t kind;
	uint64_t ordinal;
	uint64_t time_ns;
	uint32_t arg_count;
	uint32_t wait_count;
	uint64_t blob_raw;
	uint64_t blob_packed;
};
struct Wait {
	uint32_t queue;
	uint32_t reserved;
	uint64_t completed;
};
struct ChunkHeader {
	uint64_t address;
	uint32_t raw;
	uint32_t packed; // 0: all zero
};
struct PageHeader {
	uint64_t address;
	uint32_t bytes;
	uint32_t reserved;
};
#pragma pack(pop)

struct Call {
	CallKind             kind = CallKind::End;
	uint64_t             blob_raw = 0; // replay: blob holds the packed bytes until used
	uint64_t             ordinal = 0;
	uint64_t             time_ns = 0;
	std::vector<int64_t> args;
	std::vector<Wait>    waits;
	std::vector<uint8_t> blob;
};

[[noreturn]] void Fatal(const std::string& reason) {
	std::fprintf(stderr, "[replay] failed: %s\n", reason.c_str());
	std::fflush(stdout);
	std::fflush(stderr);
	std::quick_exit(1);
}

std::vector<uint8_t> Pack(std::span<const uint8_t> raw) {
	std::vector<uint8_t> packed(ZSTD_compressBound(raw.size()));
	const auto           size = ZSTD_compress(packed.data(), packed.size(), raw.data(), raw.size(), ZstdLevel);
	EXIT_IF(ZSTD_isError(size) != 0u);
	packed.resize(size);
	return packed;
}

void Unpack(std::span<const uint8_t> packed, std::span<uint8_t> raw) {
	const auto size = ZSTD_decompress(raw.data(), raw.size(), packed.data(), packed.size());
	if (ZSTD_isError(size) != 0u || size != raw.size()) {
		Fatal("corrupt compressed data");
	}
}

bool AllZero(std::span<const uint8_t> bytes) {
	return std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == 0; });
}

// Guest memory as the guest wrote it, without faulting on pages the GPU caches protect.
bool ReadGuest(uint64_t address, uint8_t* data, uint64_t size) {
	if (Memory::TryReadBacking(address, data, size)) return true;
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	const iovec local {data, size};
	const iovec remote {reinterpret_cast<void*>(address), size};
	return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(size);
#else
	return false;
#endif
}

uint64_t NowNs() {
	return Common::SteadyNs();
}

std::atomic<uint64_t>& Flips() {
	static std::atomic<uint64_t> flips {0};
	return flips;
}

uint64_t EnvNumber(const char* name, uint64_t fallback) {
	const char* text = std::getenv(name);
	if (text == nullptr || *text == '\0') return fallback;
	return std::strtoull(text, nullptr, 0);
}

GuestGpu& Gpu() {
	auto* renderer = AgcRenderer();
	EXIT_IF(renderer == nullptr);
	return renderer->GetGpu();
}

uint32_t PacingQueue(uint32_t queue) {
	return queue == 0 ? 0u : 1u + (queue - ComputeQueueBase);
}

class Recorder {
public:
	Recorder(std::filesystem::path directory, uint64_t at_flip, uint64_t after_seconds,
	         uint64_t frames)
	    : m_directory(std::move(directory)), m_at_flip(at_flip),
	      m_after_ns(after_seconds * 1000000000ull), m_frames(frames), m_created(NowNs()) {}

	void Record(Call call) {
		std::lock_guard lock(m_mutex);
		if (m_state == State::Done) return;
		const auto flips = Flips().load(std::memory_order_acquire);
		if (m_state == State::Waiting) {
			// Video out set-up before the capture is replayed first; earlier flips are not.
			if (call.kind == CallKind::VideoOut &&
			    static_cast<VideoOutCallKind>(call.args[0]) != VideoOutCallKind::SubmitFlip) {
				m_prologue.push_back(call);
			}
			if (flips < m_at_flip || NowNs() - m_created < m_after_ns) return;
			if (!Start(flips)) return;
			if (call.kind == CallKind::VideoOut &&
			    static_cast<VideoOutCallKind>(call.args[0]) != VideoOutCallKind::SubmitFlip) {
				return; // written with the prologue
			}
		}
		const Common::CompilePause::Scope hold;
		while (m_marked_flips < flips) {
			Call marker;
			marker.kind = CallKind::FrameEnd;
			marker.args = {static_cast<int64_t>(++m_marked_flips)};
			if (!Write(std::move(marker))) return;
		}
		if (flips >= m_start_flips + m_frames) {
			Finish();
			return;
		}
		if (Delta()) Write(std::move(call));
	}

private:
	enum class State { Waiting, Recording, Done };

	struct Range {
		Memory::CaptureRange          info;
		bool                  tracked = false;
		std::vector<uint64_t> hashes;
	};

	bool Fail(const char* reason) {
		Stop(reason);
		return false;
	}

	bool Start(uint64_t flips) {
		const Common::CompilePause::Scope hold;
		const auto                        started = NowNs();
		std::filesystem::create_directories(m_directory);
		if (std::filesystem::exists(m_directory / "header.json")) {
			return Fail("the capture directory already holds a capture");
		}
		FlushGpuWrites();
		// The replay sets up the same memory layout from the game's parameters.
		std::error_code copy_error;
		std::filesystem::copy_file(LibKernel::FileSystem::GetRealFilename("/app0/sce_sys/param.json"),
		                           m_directory / "param.json",
		                           std::filesystem::copy_options::overwrite_existing, copy_error);
		const auto ranges   = Memory::CaptureRanges();
		const auto physical = Memory::CapturePhysicalBlocks();
		{
			std::ofstream out(m_directory / "ranges.bin", std::ios::binary);
			const auto    count  = static_cast<uint32_t>(ranges.size());
			const auto    pcount = static_cast<uint32_t>(physical.size());
			out.write(reinterpret_cast<const char*>(&count), sizeof(count));
			out.write(reinterpret_cast<const char*>(ranges.data()),
			          static_cast<std::streamsize>(sizeof(Memory::CaptureRange) * count));
			out.write(reinterpret_cast<const char*>(&pcount), sizeof(pcount));
			out.write(reinterpret_cast<const char*>(physical.data()),
			          static_cast<std::streamsize>(sizeof(Memory::CapturePhysical) * pcount));
			if (!out) return Fail("cannot write ranges.bin");
		}
		m_snapshot.open(m_directory / "snapshot.bin", std::ios::binary);
		m_calls.open(m_directory / "calls.bin", std::ios::binary);
		if (!m_snapshot || !m_calls) return Fail("cannot create the capture files");
		for (const auto& info: ranges) {
			Range range;
			range.info = info;
			// Ranges the CPU can write are hashed for deltas; the GPU's own writes are replayed.
			range.tracked = (info.protection & 0x02) != 0;
			m_ranges.push_back(std::move(range));
		}
		for (auto& range: m_ranges) {
			if (!Snapshot(range)) return false;
		}
		m_snapshot.close();
		auto& gpu = Gpu();
		for (uint32_t queue = 0; queue < GuestGpu::PacingQueues; queue++) {
			m_baseline[queue] = gpu.AcceptedSubmissions(queue);
		}
		m_state        = State::Recording;
		m_start_flips  = flips;
		m_marked_flips = flips;
		m_origin       = NowNs();
		for (auto& call: m_prologue) {
			if (!Write(std::move(call))) return false;
		}
		m_prologue.clear();
		Call start;
		start.kind = CallKind::Start;
		start.args = {static_cast<int64_t>(flips)};
		if (!Write(std::move(start))) return false;
		std::fprintf(stderr,
		             "[frame-capture] started at flip %" PRIu64 ": %zu ranges, %.1f GiB, snapshot "
		             "%.1f MiB in %.1f s\n",
		             flips, m_ranges.size(), static_cast<double>(m_total_bytes) / (1u << 30u),
		             static_cast<double>(m_snapshot_bytes) / (1u << 20u),
		             static_cast<double>(NowNs() - started) / 1e9);
		return true;
	}

	// GPU buffer writes go back to guest memory, so the snapshot holds what the GPU made.
	void FlushGpuWrites() {
		if (EnvNumber("KYTY_FRAME_CAPTURE_FLUSH", 1) == 0) return;
		auto* renderer = AgcRenderer();
		if (renderer == nullptr) return;
		const auto ranges = Memory::CaptureRanges();
		renderer->GetGpu().SendCommandSync([&] {
			auto& buffers = renderer->GetBufferCache();
			for (const auto& range: ranges) {
				if (!GuestRange {range.start, range.size}.Valid()) continue;
				for (uint64_t offset = 0; offset < range.size; offset += ChunkBytes) {
					const auto size = std::min(ChunkBytes, range.size - offset);
					if (buffers.HasGpuDirtyBytes(range.start + offset, size)) {
						buffers.ReadMemory(range.start + offset, size);
					}
				}
			}
		});
	}

	bool Snapshot(Range& range) {
		const auto chunks = (range.info.size + ChunkBytes - 1) / ChunkBytes;
		if (range.tracked) range.hashes.resize((range.info.size + PageBytes - 1) / PageBytes);
		m_total_bytes += range.info.size;
		constexpr size_t Batch = 64;
		for (uint64_t first = 0; first < chunks; first += Batch) {
			std::error_code space_error;
			if (std::filesystem::space(m_directory, space_error).available < (uint64_t {10} << 30u)) {
				return Fail("less than 10 GiB free on the capture volume");
			}
			const auto                        count = std::min<uint64_t>(Batch, chunks - first);
			std::vector<std::vector<uint8_t>> packed(count);
			std::vector<uint32_t>             raw(count);
			std::atomic<bool>                 failed {false};
			Common::ParallelFor(count, [&](size_t index) {
				const auto offset = (first + index) * ChunkBytes;
				const auto size   = std::min(ChunkBytes, range.info.size - offset);
				std::vector<uint8_t> bytes(size);
				if (!ReadGuest(range.info.start + offset, bytes.data(), size)) {
					// Unreadable pages (guest PROT_NONE, holes) are kept as zero.
					std::fill(bytes.begin(), bytes.end(), 0);
				}
				if (range.tracked) {
					for (uint64_t page = 0; page < size; page += PageBytes) {
						range.hashes[(offset + page) / PageBytes] =
						    XXH3_64bits(bytes.data() + page, std::min(PageBytes, size - page));
					}
				}
				raw[index] = static_cast<uint32_t>(size);
				if (!AllZero(bytes)) packed[index] = Pack(bytes);
			});
			if (failed) return Fail("snapshot read failed");
			for (uint64_t index = 0; index < count; index++) {
				ChunkHeader header {range.info.start + (first + index) * ChunkBytes, raw[index],
				                    static_cast<uint32_t>(packed[index].size())};
				m_snapshot.write(reinterpret_cast<const char*>(&header), sizeof(header));
				m_snapshot.write(reinterpret_cast<const char*>(packed[index].data()),
				                 static_cast<std::streamsize>(packed[index].size()));
				m_snapshot_bytes += sizeof(header) + packed[index].size();
			}
			if (!m_snapshot) return Fail("cannot write snapshot.bin");
		}
		return true;
	}

	// The pages the CPU changed since the last call.
	bool Delta() {
		if (m_work.empty()) {
			for (uint32_t index = 0; index < m_ranges.size(); index++) {
				if (!m_ranges[index].tracked) continue;
				for (uint64_t offset = 0; offset < m_ranges[index].info.size; offset += ChunkBytes) {
					m_work.push_back({index, offset});
				}
			}
		}
		std::vector<uint8_t> pages;
		std::mutex           pages_mutex;
		Common::ParallelFor(m_work.size(), [&](size_t item) {
			auto&                range  = m_ranges[m_work[item].range];
			const auto           offset = m_work[item].offset;
			const auto           size   = std::min(ChunkBytes, range.info.size - offset);
			std::vector<uint8_t> bytes(size);
			if (!ReadGuest(range.info.start + offset, bytes.data(), size)) return;
			std::vector<uint8_t> local;
			for (uint64_t page = 0; page < size; page += PageBytes) {
				const auto bytes_in_page = std::min(PageBytes, size - page);
				const auto hash          = XXH3_64bits(bytes.data() + page, bytes_in_page);
				auto&      known         = range.hashes[(offset + page) / PageBytes];
				if (hash == known) continue;
				known = hash;
				PageHeader header {range.info.start + offset + page, static_cast<uint32_t>(bytes_in_page),
				                   0};
				const auto* raw = reinterpret_cast<const uint8_t*>(&header);
				local.insert(local.end(), raw, raw + sizeof(header));
				local.insert(local.end(), bytes.data() + page, bytes.data() + page + bytes_in_page);
			}
			if (!local.empty()) {
				std::lock_guard lock(pages_mutex);
				pages.insert(pages.end(), local.begin(), local.end());
			}
		});
		if (pages.empty()) return true;
		Call call;
		call.kind = CallKind::Memory;
		call.blob = std::move(pages);
		return Write(std::move(call));
	}

	bool Write(Call call) {
		call.ordinal = ++m_ordinal;
		call.time_ns = NowNs() - m_origin;
		if (call.kind != CallKind::Memory && call.kind != CallKind::FrameEnd) {
			auto& gpu = Gpu();
			for (uint32_t queue = 0; queue < GuestGpu::PacingQueues; queue++) {
				const auto completed = gpu.CompletedSubmissions(queue);
				if (completed > m_baseline[queue]) {
					call.waits.push_back({queue, 0, completed - m_baseline[queue]});
				}
			}
		}
		std::vector<uint8_t> packed;
		if (!call.blob.empty()) packed = Pack(call.blob);
		CallHeader header {CallMagic,
		                   static_cast<uint32_t>(call.kind),
		                   call.ordinal,
		                   call.time_ns,
		                   static_cast<uint32_t>(call.args.size()),
		                   static_cast<uint32_t>(call.waits.size()),
		                   call.blob.size(),
		                   packed.size()};
		m_calls.write(reinterpret_cast<const char*>(&header), sizeof(header));
		m_calls.write(reinterpret_cast<const char*>(call.args.data()),
		              static_cast<std::streamsize>(call.args.size() * sizeof(int64_t)));
		m_calls.write(reinterpret_cast<const char*>(call.waits.data()),
		              static_cast<std::streamsize>(call.waits.size() * sizeof(Wait)));
		m_calls.write(reinterpret_cast<const char*>(packed.data()),
		              static_cast<std::streamsize>(packed.size()));
		if (!m_calls) return Fail("cannot write calls.bin");
		m_call_bytes += sizeof(header) + packed.size();
		return true;
	}

	void Finish() {
		Call end;
		end.kind = CallKind::End;
		if (!Write(std::move(end))) return;
		m_calls.close();
		std::ofstream header(m_directory / "header.json");
		header << "{\"magic\":\"kyty-frame-capture\",\"version\":" << CaptureVersion
		       << ",\"start_flip\":" << m_start_flips << ",\"frames\":" << m_frames
		       << ",\"calls\":" << m_ordinal << ",\"ranges\":" << m_ranges.size()
		       << ",\"guest_bytes\":" << m_total_bytes << ",\"snapshot_bytes\":" << m_snapshot_bytes
		       << ",\"call_bytes\":" << m_call_bytes << "}\n";
		m_state = State::Done;
		m_ranges.clear();
		std::fprintf(stderr, "[frame-capture] recorded %" PRIu64 " frames, %" PRIu64 " calls into %s\n",
		             m_frames, m_ordinal, m_directory.string().c_str());
	}

	void Stop(const char* reason) {
		std::fprintf(stderr, "[frame-capture] stopped: %s\n", reason);
		m_state = State::Done;
		m_ranges.clear();
		m_prologue.clear();
	}

	const std::filesystem::path m_directory;
	const uint64_t              m_at_flip;
	const uint64_t              m_after_ns;
	const uint64_t              m_frames;
	const uint64_t              m_created;
	std::mutex                  m_mutex;
	State                       m_state = State::Waiting;
	std::vector<Call>           m_prologue;
	std::vector<Range>          m_ranges;
	struct Work {
		uint32_t range;
		uint64_t offset;
	};
	std::vector<Work> m_work;
	std::ofstream               m_snapshot;
	std::ofstream               m_calls;
	std::array<uint64_t, GuestGpu::PacingQueues> m_baseline {};
	uint64_t                    m_start_flips    = 0;
	uint64_t                    m_marked_flips   = 0;
	uint64_t                    m_ordinal        = 0;
	uint64_t                    m_origin         = 0;
	uint64_t                    m_total_bytes    = 0;
	uint64_t                    m_snapshot_bytes = 0;
	uint64_t                    m_call_bytes     = 0;
};

Recorder* Active() {
	static Recorder* const recorder = []() -> Recorder* {
		const char* directory = std::getenv("KYTY_FRAME_CAPTURE");
		if (directory == nullptr || *directory == '\0') return nullptr;
		return new Recorder(directory, EnvNumber("KYTY_FRAME_CAPTURE_AT_FLIP", 1),
		                    EnvNumber("KYTY_FRAME_CAPTURE_AFTER_SECONDS", 0),
		                    std::max<uint64_t>(1, EnvNumber("KYTY_FRAME_CAPTURE_FRAMES", 3)));
	}();
	return recorder;
}

void RecordCall(CallKind kind, std::initializer_list<int64_t> args) {
	auto* recorder = Active();
	if (recorder == nullptr) return;
	Call call;
	call.kind = kind;
	call.args = args;
	recorder->Record(std::move(call));
}

// Replay ----------------------------------------------------------------------------------------

struct ReplayState {
	std::vector<uint64_t> flip_times;
	std::mutex            mutex;
	std::atomic<bool>     replaying {false};
};

ReplayState& Replay() {
	static ReplayState state;
	return state;
}

std::vector<uint8_t> ReadFile(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) Fatal("cannot open " + path.string());
	return {std::istreambuf_iterator<char>(in), {}};
}

template <typename T>
T Take(const std::vector<uint8_t>& bytes, size_t& at) {
	if (at + sizeof(T) > bytes.size()) Fatal("truncated capture file");
	T value;
	std::memcpy(&value, bytes.data() + at, sizeof(T));
	at += sizeof(T);
	return value;
}

std::vector<Call> ReadCalls(const std::filesystem::path& path) {
	const auto        bytes = ReadFile(path);
	std::vector<Call> calls;
	for (size_t at = 0; at < bytes.size();) {
		const auto header = Take<CallHeader>(bytes, at);
		if (header.magic != CallMagic) Fatal("calls.bin: bad record");
		Call call;
		call.kind    = static_cast<CallKind>(header.kind);
		call.ordinal = header.ordinal;
		call.time_ns = header.time_ns;
		for (uint32_t i = 0; i < header.arg_count; i++) call.args.push_back(Take<int64_t>(bytes, at));
		for (uint32_t i = 0; i < header.wait_count; i++) call.waits.push_back(Take<Wait>(bytes, at));
		if (header.blob_raw != 0) {
			if (at + header.blob_packed > bytes.size()) Fatal("truncated calls.bin");
			call.blob_raw = header.blob_raw;
			call.blob.assign(bytes.data() + at, bytes.data() + at + header.blob_packed);
			at += header.blob_packed;
		}
		calls.push_back(std::move(call));
	}
	return calls;
}

// Shaders the game created (sceAgcCreateShader) are known by their headers, which the creation
// relocates in place and points at the code: registering them again lets the replay translate
// them. The capture holds no creation calls, so the restored memory is scanned for them.
constexpr uint32_t ShaderFileHeader = 0x34333231;
constexpr uint32_t ShaderVersion    = 0x18;

std::atomic<uint64_t>& ShadersFound() {
	static std::atomic<uint64_t> found {0};
	return found;
}

bool GuestPointer(uint64_t value) {
	return value == 0 || (value >= (uint64_t {1} << 32u) && value < (uint64_t {1} << 47u));
}

void RegisterShaderAt(uint64_t address) {
	Shader header {};
	if (!ReadGuest(address, reinterpret_cast<uint8_t*>(&header), sizeof(header))) return;
	const auto code = reinterpret_cast<uint64_t>(header.code);
	if (header.file_header != ShaderFileHeader || header.version != ShaderVersion || code == 0 ||
	    (code & 0xFFFF0000000000FFull) != 0 || header.shader_size == 0 ||
	    header.shader_size > (64u << 20u) ||
	    !GuestPointer(reinterpret_cast<uint64_t>(header.user_data)) ||
	    !GuestPointer(reinterpret_cast<uint64_t>(header.input_semantics)) ||
	    header.shader_size % 4u != 0 || !GuestRange {code, header.shader_size}.Valid()) {
		return;
	}
	// Everything the registration reads must be mapped in the replay.
	uint8_t probe[16];
	const auto readable = [&](const void* pointer, uint64_t bytes) {
		return pointer == nullptr || ReadGuest(reinterpret_cast<uint64_t>(pointer), probe, std::min<uint64_t>(bytes, 16));
	};
	if (!readable(reinterpret_cast<const void*>(code), 16) || !readable(reinterpret_cast<const void*>(code + header.shader_size - 4), 4) ||
	    !readable(header.user_data, 16) || !readable(header.input_semantics, 4)) {
		return;
	}
	ShaderMappedData map;
	map.type                = static_cast<Prospero::ShaderBinaryType>(header.type);
	map.user_data           = header.user_data;
	map.input_semantics     = header.input_semantics;
	map.num_input_semantics = header.num_input_semantics;
	map.code_size_bytes     = header.shader_size;
	map.scratch_size_dwords = header.scratch_size_dw_per_thread;
	ShaderMapUserData(code, map);
	ShadersFound().fetch_add(1, std::memory_order_relaxed);
}

void ScanShaders(uint64_t address, std::span<const uint8_t> bytes) {
	for (size_t at = 0; at + 8 <= bytes.size(); at += 8) {
		uint64_t word = 0;
		std::memcpy(&word, bytes.data() + at, 8);
		if (word == (uint64_t {ShaderVersion} << 32u | ShaderFileHeader)) RegisterShaderAt(address + at);
	}
}

void WriteGuest(uint64_t address, const uint8_t* data, uint64_t size) {
	if (!Memory::TryWriteBacking(address, data, size)) {
		std::memcpy(reinterpret_cast<void*>(address), data, size);
	}
}

std::vector<uint8_t> Blob(const Call& call) {
	std::vector<uint8_t> raw(call.blob_raw);
	if (call.blob_raw != 0) Unpack(call.blob, raw);
	return raw;
}

std::atomic<const char*>& Phase() {
	static std::atomic<const char*> phase {"start"};
	return phase;
}

std::filesystem::path& GuardOutput() {
	static std::filesystem::path path;
	return path;
}

// Resident and shared-memory use every 250 ms, so a cap kill still shows where memory grew. Once
// the calls run, guest shared memory growing more than KYTY_REPLAY_SHMEM_GUARD_MB (default 2048)
// over its level after loading dumps every thread's stack (eu-stack) and ends the replay with exit
// code 3, before the memory cap kills it.
void StartMemorySampler() {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
#endif
	// The cgroup files are read without the process's memory map lock, which a growing mapping
	// may hold for a long time.
	std::string cgroup_path;
	{
		std::ifstream cgroup("/proc/self/cgroup");
		for (std::string line; std::getline(cgroup, line);) {
			if (line.rfind("0::", 0) == 0) cgroup_path = "/sys/fs/cgroup" + line.substr(3);
		}
	}
	std::thread([cgroup_path] {
		const auto start     = NowNs();
		const auto guard     = EnvNumber("KYTY_REPLAY_SHMEM_GUARD_MB", 2048) << 20u;
		uint64_t   baseline  = 0;
		uint64_t   tick      = 0;
		const auto read_file = [](const std::string& path) {
			std::ifstream file(path);
			uint64_t      value = 0;
			file >> value;
			return value;
		};
		const auto limit = read_file(cgroup_path + "/memory.max");
		for (;; tick++) {
			const auto current = read_file(cgroup_path + "/memory.current");
			const bool calls   = std::strcmp(Phase().load(), "calls") == 0;
			if (calls && baseline == 0) baseline = current;
			if (calls && (current > baseline + guard || (limit != 0 && current + (1ull << 30u) > limit))) {
				std::fprintf(stderr,
				             "[replay] memory guard: cgroup %" PRIu64 " MiB, %" PRIu64
				             " MiB after loading; thread stacks in %s\n",
				             current >> 20u, baseline >> 20u,
				             (GuardOutput() / "stacks.txt").string().c_str());
				std::fflush(stderr);
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
				const auto command =
				    fmt::format("timeout -s KILL 10 eu-stack -p {} > '{}' 2>&1", getpid(),
				                (GuardOutput() / "stacks.txt").string());
				[[maybe_unused]] const int result = std::system(command.c_str());
#endif
				std::fflush(stdout);
				std::quick_exit(3);
			}
			if (tick % 5 == 0) {
				std::string stat;
				std::ifstream memory_stat(cgroup_path + "/memory.stat");
				for (std::string entry; std::getline(memory_stat, entry);) {
					for (const char* key: {"anon ", "file ", "shmem ", "kernel "}) {
						if (entry.rfind(key, 0) == 0) {
							stat += " " + entry.substr(0, entry.find(' ')) + "=" +
							        std::to_string(std::strtoull(entry.c_str() + entry.find(' '), nullptr, 10) >> 20u) + "M";
						}
					}
				}
				std::fprintf(stderr, "[replay-mem] t=%.2f s cgroup=%" PRIu64 "M%s phase=%s\n",
				             static_cast<double>(NowNs() - start) / 1e9, current >> 20u, stat.c_str(),
				             Phase().load());
				std::fflush(stderr);
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
	}).detach();
}

double ShmemGiB() {
	std::ifstream status("/proc/self/status");
	for (std::string line; std::getline(status, line);) {
		if (line.rfind("RssShmem:", 0) == 0) return std::strtod(line.c_str() + 9, nullptr) / (1u << 20u);
	}
	return 0.0;
}

double RssGiB() {
	std::ifstream status("/proc/self/status");
	for (std::string line; std::getline(status, line);) {
		if (line.rfind("VmRSS:", 0) == 0) return std::strtod(line.c_str() + 6, nullptr) / (1u << 20u);
	}
	return 0.0;
}

// The 1 MiB chunks the replay changed, restored before a further pass.
std::mutex&                   TouchedMutex() { static std::mutex m; return m; }
std::vector<uint64_t>&        Touched() { static std::vector<uint64_t> t; return t; }

void Touch(uint64_t address, uint64_t size) {
	std::lock_guard lock(TouchedMutex());
	for (auto chunk = address & ~(ChunkBytes - 1); chunk < address + size; chunk += ChunkBytes) {
		Touched().push_back(chunk);
	}
}

// invalidate: restore only the touched chunks (zero ones too) and tell the GPU caches.
uint64_t LoadSnapshot(const std::filesystem::path& path, bool invalidate) {
	std::vector<uint64_t> touched;
	if (invalidate) {
		std::lock_guard lock(TouchedMutex());
		touched = Touched();
		std::sort(touched.begin(), touched.end());
		touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
	}
	std::ifstream in(path, std::ios::binary);
	if (!in) Fatal("cannot open snapshot.bin");
	std::vector<uint8_t> packed;
	std::vector<uint8_t> raw;
	uint64_t             chunks = 0;
	for (ChunkHeader header {}; in.read(reinterpret_cast<char*>(&header), sizeof(header));) {
		if (invalidate && !std::binary_search(touched.begin(), touched.end(), header.address)) {
			in.seekg(header.packed, std::ios::cur);
			continue;
		}
		raw.assign(header.raw, 0);
		if (header.packed != 0) {
			packed.resize(header.packed);
			in.read(reinterpret_cast<char*>(packed.data()), header.packed);
			Unpack(packed, raw);
		}
		if (header.packed != 0 || invalidate) WriteGuest(header.address, raw.data(), raw.size());
		if (header.packed != 0 && !invalidate) ScanShaders(header.address, raw);
		if (invalidate && GuestRange {header.address, raw.size()}.Valid()) {
			Memory::InvalidateMemory(header.address, raw.size());
		}
		chunks++;
	}
	return chunks;
}

void ApplyPages(const std::vector<uint8_t>& blob) {
	for (size_t at = 0; at < blob.size();) {
		Touch(Take<PageHeader>(blob, at).address, 1);
		at -= sizeof(PageHeader);
		const auto header = Take<PageHeader>(blob, at);
		if (at + header.bytes > blob.size()) Fatal("truncated memory record");
		WriteGuest(header.address, blob.data() + at, header.bytes);
		ScanShaders(header.address, {blob.data() + at, header.bytes});
		if (GuestRange {header.address, header.bytes}.Valid()) {
			Memory::InvalidateMemory(header.address, header.bytes);
		}
		at += header.bytes;
	}
}

bool WaitFor(const std::vector<Wait>& waits, const std::array<uint64_t, GuestGpu::PacingQueues>& base) {
	auto&      gpu      = Gpu();
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
	for (const auto& wait: waits) {
		while (gpu.CompletedSubmissions(wait.queue) < base[wait.queue] + wait.completed) {
			if (std::chrono::steady_clock::now() > deadline) return false;
			std::this_thread::sleep_for(std::chrono::microseconds(50));
		}
	}
	return true;
}

bool ReadDisplay(int handle, std::vector<uint8_t>& rgba, uint32_t& width, uint32_t& height) {
	ImageInfo info {};
	if (!VideoOut::VideoOutCurrentImage(handle, &info)) return false;
	auto* renderer = AgcRenderer();
	bool  ok       = false;
	renderer->GetGpu().SendCommandSync([&] {
		Common::LockGuard       lock(renderer->GetMutex());
		auto&                   cache = renderer->GetTextureCache();
		TextureCache::ImageDesc desc {};
		desc.info                  = info;
		desc.view_info.format      = info.pixel_format;
		desc.view_info.type        = vk::ImageViewType::e2D;
		desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		desc.view_info.base_level  = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer  = 0;
		desc.view_info.layer_count = 1;
		desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
		desc.type                  = TextureCache::BindingType::VideoOut;
		const auto id              = cache.FindImage(desc);
		auto&      image           = cache.GetImage(id);
		cache.UpdateImage(id);
		const auto format = image.backing.format;
		width             = info.extent.width;
		height            = info.extent.height;
		const auto bytes  = uint64_t {width} * height * 4;
		auto       download =
		    renderer->GetBufferCache().GetUtilityBuffer(MemoryUsage::Download).MapDownload(bytes, 4);
		vk::BufferImageCopy copy {};
		copy.bufferOffset     = download.offset;
		copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copy.imageExtent      = vk::Extent3D {width, height, 1};
		image.Download(std::span(&copy, 1), download.buffer->Handle(), download.offset, bytes);
		renderer->GetCommandScheduler().Finish();
		download.buffer->Invalidate(download.offset, bytes);
		rgba.resize(bytes);
		const auto* source = download.mapped;
		for (uint64_t pixel = 0; pixel < uint64_t {width} * height; pixel++) {
			const auto* texel = source + pixel * 4;
			auto*       out   = rgba.data() + pixel * 4;
			switch (format) {
				case vk::Format::eB8G8R8A8Unorm:
				case vk::Format::eB8G8R8A8Srgb:
					out[0] = texel[2], out[1] = texel[1], out[2] = texel[0], out[3] = 255;
					break;
				case vk::Format::eA2B10G10R10UnormPack32:
				case vk::Format::eA2R10G10B10UnormPack32: {
					uint32_t word = 0;
					std::memcpy(&word, texel, 4);
					const auto low  = static_cast<uint8_t>((word >> 2u) & 0xffu);
					const auto mid  = static_cast<uint8_t>((word >> 12u) & 0xffu);
					const auto high = static_cast<uint8_t>((word >> 22u) & 0xffu);
					const bool abgr = format == vk::Format::eA2B10G10R10UnormPack32;
					out[0] = abgr ? low : high, out[1] = mid, out[2] = abgr ? high : low, out[3] = 255;
					break;
				}
				default: out[0] = texel[0], out[1] = texel[1], out[2] = texel[2], out[3] = 255; break;
			}
		}
		ok = true;
	});
	return ok;
}

} // namespace

bool Enabled() noexcept {
	return Active() != nullptr;
}

void NoteFlip() noexcept {
	Flips().fetch_add(1, std::memory_order_acq_rel);
	auto& replay = Replay();
	if (replay.replaying) {
		std::lock_guard lock(replay.mutex);
		replay.flip_times.push_back(NowNs());
	}
}

void Submit(uint32_t queue, const uint32_t* commands, uint32_t dwords) {
	RecordCall(CallKind::Submit,
	           {queue, static_cast<int64_t>(reinterpret_cast<uintptr_t>(commands)), dwords});
}

void ReleaseRewind(const uint32_t* rewind) {
	RecordCall(CallKind::ReleaseRewind, {static_cast<int64_t>(reinterpret_cast<uintptr_t>(rewind))});
}

void SuspendPoint() {
	RecordCall(CallKind::SuspendPoint, {});
}

void VideoOutCall(VideoOutCallKind kind, std::span<const int64_t> args, std::span<const uint8_t> blob) {
	auto* recorder = Active();
	if (recorder == nullptr) return;
	Call call;
	call.kind = CallKind::VideoOut;
	call.args.push_back(static_cast<int64_t>(kind));
	call.args.insert(call.args.end(), args.begin(), args.end());
	call.blob.assign(blob.begin(), blob.end());
	recorder->Record(std::move(call));
}

void RunReplay(const ReplayOptions& options) {
	int exit_code = 0;
	{
		GuardOutput() = options.output;
		std::filesystem::create_directories(options.output);
		StartMemorySampler();
		Phase() = "map";
		const auto& dir = options.capture;
		if (!std::filesystem::exists(dir / "header.json")) {
			Fatal("no finished capture in " + dir.string());
		}
		std::filesystem::create_directories(options.output);
		// Ranges and physical blocks.
		const auto                   bytes = ReadFile(dir / "ranges.bin");
		size_t                       at    = 0;
		std::vector<Memory::CaptureRange>    ranges(Take<uint32_t>(bytes, at));
		for (auto& range: ranges) range = Take<Memory::CaptureRange>(bytes, at);
		std::vector<Memory::CapturePhysical> physical(Take<uint32_t>(bytes, at));
		for (auto& block: physical) block = Take<Memory::CapturePhysical>(bytes, at);
		uint32_t physical_failed = 0;
		for (const auto& block: physical) physical_failed += Memory::ReplayAllocatePhysical(block) ? 0 : 1;
		uint32_t skipped = 0;
		for (const auto& range: ranges) {
			if (!Memory::ReplayMapRange(range)) skipped++;
		}
		Phase()               = "load";
		const auto load_start = NowNs();
		const auto chunks     = LoadSnapshot(dir / "snapshot.bin", false);
		const auto calls      = ReadCalls(dir / "calls.bin");
		std::printf("[replay] %zu ranges (%u not mapped), %zu physical blocks (%u not placed), %" PRIu64
		            " snapshot chunks, %" PRIu64 " shaders, %zu calls, loaded in %.1f s\n",
		            ranges.size(), skipped, physical.size(), physical_failed, chunks,
		            ShadersFound().load(), calls.size(),
		            static_cast<double>(NowNs() - load_start) / 1e9);
		std::printf("[replay] resident after load: %.2f GiB\n", RssGiB());

		auto& gpu         = Gpu();
		int   main_handle = -1;
		std::map<int64_t, int> handles; // recorded handle -> replay handle
		std::ofstream          report(options.output / "replay.json");
		report << "{\"capture\":\"" << dir.string() << "\",\"passes\":[";
		for (uint32_t pass = 0; pass < std::max(1u, options.repeat); pass++) {
			if (pass != 0) {
				gpu.WaitForIdle();
				LoadSnapshot(dir / "snapshot.bin", true);
			}
			std::array<uint64_t, GuestGpu::PacingQueues> base {};
			for (uint32_t queue = 0; queue < GuestGpu::PacingQueues; queue++) {
				base[queue] = gpu.AcceptedSubmissions(queue);
			}
			{
				std::lock_guard lock(Replay().mutex);
				Replay().flip_times.clear();
				Replay().replaying = true;
			}
			bool       started     = false;
			uint64_t   timeouts    = 0;
			uint64_t   submissions = 0;
			const auto pass_start  = NowNs();
			Phase()                = "calls";
			uint64_t call_index    = 0;
			for (const auto& call: calls) {
				if (++call_index % 25 == 0) {
					std::fprintf(stderr, "[replay] call %" PRIu64 "/%zu, %" PRIu64 " submissions\n",
					             call_index, calls.size(), submissions);
				}
				if (!call.waits.empty() && !WaitFor(call.waits, base)) timeouts++;
				const double shmem_before = call_index <= 80 ? ShmemGiB() : 0.0;
				switch (call.kind) {
					case CallKind::Memory: ApplyPages(Blob(call)); break;
					case CallKind::Submit: {
						auto*      commands = reinterpret_cast<const uint32_t*>(call.args[1]);
						const auto queue    = static_cast<uint32_t>(call.args[0]);
						const auto dwords   = static_cast<uint32_t>(call.args[2]);
						if (queue == 0) {
							gpu.Submit({commands, dwords});
						} else {
							gpu.SubmitCompute(queue, {commands, dwords});
						}
						submissions++;
						break;
					}
					case CallKind::ReleaseRewind:
						gpu.ReleaseRewind(reinterpret_cast<const uint32_t*>(call.args[0]));
						break;
					case CallKind::SuspendPoint: gpu.SuspendPoint(); break;
					case CallKind::Start: started = true; break;
					case CallKind::VideoOut: {
						const auto kind = static_cast<VideoOutCallKind>(call.args[0]);
						const auto map  = [&](int64_t handle) {
							const auto found = handles.find(handle);
							return found != handles.end() ? found->second : static_cast<int>(handle);
						};
						const bool setup = kind != VideoOutCallKind::SubmitFlip &&
						                   kind != VideoOutCallKind::SetFlipRate;
						if (pass != 0 && setup) break;
						// Captures from before this rule kept the flips made before the start.
						if (!started && kind == VideoOutCallKind::SubmitFlip) break;
						switch (kind) {
							case VideoOutCallKind::Open: {
								const int handle = VideoOut::VideoOutOpen(
								    255, static_cast<int>(call.args[1]), static_cast<int>(call.args[2]), nullptr);
								handles[call.args[3]] = handle;
								if (handle != call.args[3]) {
									// Flip packets in the command buffers name the recorded handle.
									std::fprintf(stderr,
									             "[replay] video out handle %d differs from the recorded %" PRId64
									             "\n",
									             handle, call.args[3]);
								}
								if (main_handle < 0 && call.args[1] == 0) main_handle = handle;
								break;
							}
							case VideoOutCallKind::Close: VideoOut::VideoOutClose(map(call.args[1])); break;
							case VideoOutCallKind::RegisterBuffers2: {
								// The buffers (four pointers each), then the attribute.
								const auto  blob = Blob(call);
								const auto* buffers =
								    reinterpret_cast<const VideoOut::VideoOutBuffers*>(blob.data());
								const auto* attribute = reinterpret_cast<const VideoOut::VideoOutBufferAttribute2*>(
								    blob.data() + 32 * call.args[4]);
								VideoOut::VideoOutRegisterBuffers2(
								    map(call.args[1]), static_cast<int>(call.args[2]),
								    static_cast<int>(call.args[3]), buffers, static_cast<int>(call.args[4]),
								    attribute, static_cast<int>(call.args[5]), nullptr);
								break;
							}
							case VideoOutCallKind::SubmitChangeBufferAttribute2:
								VideoOut::VideoOutSubmitChangeBufferAttribute2(
								    map(call.args[1]), static_cast<int>(call.args[2]),
								    reinterpret_cast<const VideoOut::VideoOutBufferAttribute2*>(Blob(call).data()),
								    nullptr);
								break;
							case VideoOutCallKind::UnregisterBuffers:
								VideoOut::VideoOutUnregisterBuffers(map(call.args[1]),
								                                    static_cast<int>(call.args[2]));
								break;
							case VideoOutCallKind::SetFlipRate:
								VideoOut::VideoOutSetFlipRate(map(call.args[1]), static_cast<int>(call.args[2]));
								break;
							case VideoOutCallKind::SetFlipMaster:
								VideoOut::VideoOutSetFlipMaster(map(call.args[1]), map(call.args[2]));
								break;
							case VideoOutCallKind::SubmitFlip:
								VideoOut::VideoOutSubmitFlip(map(call.args[1]), static_cast<int>(call.args[2]),
								                             static_cast<int>(call.args[3]), call.args[4]);
								break;
						}
						break;
					}
					case CallKind::FrameEnd:
					case CallKind::End: break;
				}
				if (call_index <= 80) {
					// Which call makes the guest memory resident (diagnostic for the memory cap).
					const double shmem_after = ShmemGiB();
					std::fprintf(stderr,
					             "[replay-call] %" PRIu64 " kind=%u a0=0x%" PRIx64 " a1=0x%" PRIx64
					             " a2=0x%" PRIx64 " shmem %.2f -> %.2f GiB\n",
					             call_index, static_cast<unsigned>(call.kind),
					             call.args.size() > 0 ? static_cast<uint64_t>(call.args[0]) : 0,
					             call.args.size() > 1 ? static_cast<uint64_t>(call.args[1]) : 0,
					             call.args.size() > 2 ? static_cast<uint64_t>(call.args[2]) : 0,
					             shmem_before, shmem_after);
				}
			}
			static_cast<void>(started);
			gpu.WaitForIdle();
			// Flips complete at a vblank after the GPU is done with them.
			for (const auto& [recorded, handle]: handles) {
				const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
				while (VideoOut::VideoOutIsFlipPending(handle) > 0 &&
				       std::chrono::steady_clock::now() < deadline) {
					std::this_thread::sleep_for(std::chrono::milliseconds(1));
				}
			}
			const auto pass_ns = NowNs() - pass_start;
			std::vector<uint64_t> flips;
			{
				std::lock_guard lock(Replay().mutex);
				flips              = Replay().flip_times;
				Replay().replaying = false;
			}
			std::printf("[replay] pass %u: %.3f s, %" PRIu64 " submissions, %zu flips, %" PRIu64
			            " pacing timeouts, resident %.2f GiB\n",
			            pass + 1, static_cast<double>(pass_ns) / 1e9, submissions, flips.size(), timeouts,
			            RssGiB());
			report << (pass != 0 ? "," : "") << "{\"seconds\":" << static_cast<double>(pass_ns) / 1e9
			       << ",\"submissions\":" << submissions << ",\"timeouts\":" << timeouts
			       << ",\"flip_ms\":[";
			for (size_t i = 0; i < flips.size(); i++) {
				const auto ms = static_cast<double>(flips[i] - (i == 0 ? pass_start : flips[i - 1])) / 1e6;
				report << (i != 0 ? "," : "") << ms;
				std::printf("[replay]   frame %zu: %.1f ms\n", i + 1, ms);
			}
			report << "]}";
		}
		report << "]}\n";
		// The image the main port shows at the end.
		std::vector<uint8_t> rgba;
		uint32_t             width  = 0;
		uint32_t             height = 0;
		if (main_handle >= 0 && ReadDisplay(main_handle, rgba, width, height)) {
			const auto png = (options.output / "frame.png").string();
			stbi_write_png(png.c_str(), static_cast<int>(width), static_cast<int>(height), 4,
			               rgba.data(), static_cast<int>(width * 4));
			std::ofstream raw(options.output / "frame.rgba", std::ios::binary);
			raw.write(reinterpret_cast<const char*>(rgba.data()), static_cast<std::streamsize>(rgba.size()));
			std::printf("[replay] display %ux%u written to %s (hash %016" PRIx64 ")\n", width, height,
			            png.c_str(), XXH3_64bits(rgba.data(), rgba.size()));
		} else {
			std::printf("[replay] no display image to read\n");
			exit_code = 2;
		}
	}
	std::fflush(stdout);
	std::fflush(stderr);
	std::quick_exit(exit_code);
}

} // namespace Libs::Graphics::FrameCapture
