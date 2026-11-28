#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_LOOKAHEAD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_LOOKAHEAD_H_

#include "graphics/host_gpu/renderer/pipeline/guestReadLog.h"
#include "graphics/shader/recompiler/ir/ResourceSnapshot.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics {

// The inputs of one program's resource walk (MaterializeResources) at one draw or dispatch, as a
// scan of the submitted commands sees them before the command thread runs them.
struct LookaheadJob {
	const void*                               program = nullptr; // What the command thread matches.
	const ShaderRecompiler::IR::ResourcePlan* plan    = nullptr;
	uint64_t                                  shader_base           = 0;
	uint64_t                                  indirect_args_address = 0;
	std::array<uint32_t, 40>                  user_data {};
	uint32_t                                  user_data_count       = 0;
	std::array<uint32_t, 3>                   workgroup_counts {};
	uint32_t                                  workgroup_count_size  = 0; // 0 or 3, as SrtRuntime has.
	uint64_t                                  sequence              = 0;
	uint64_t                                  fingerprint           = 0;

	[[nodiscard]] std::span<const uint32_t> UserData() const {
		return std::span(user_data).first(user_data_count);
	}
	[[nodiscard]] std::span<const uint32_t> WorkgroupCounts() const {
		return std::span(workgroup_counts).first(workgroup_count_size);
	}
	[[nodiscard]] bool Matches(const void* other_program,
	                           const ShaderRecompiler::IR::SrtRuntime& runtime) const;
};

// Resource walks of draws and dispatches resolved ahead of the command thread (KYTY_LOOKAHEAD).
// A resolve thread takes jobs in submission order, walks each program's plan with scratch of its
// own and logs every guest read (GuestReadLog). The command thread takes the resolution whose
// inputs equal its own and uses it only if every logged read still holds and the GPU owns none of
// the bytes; otherwise it walks itself, as without lookahead. The walk ahead reads the backing
// store and never reads back GPU-written memory or handles faults, and it gives up on walks that
// need the image heap or the GPU's heap-row reports (command-thread state).
//
// Design after TheCruZ's KytyEmu resolve thread (11d077ed, 7e31e056: program lock, plan scratch
// and the reads of GPU-written memory are the races to handle); this is a new, smaller
// implementation that leaves the command processor as it is.
class Lookahead {
public:
	enum class Mode : uint8_t { Off, On, Verify };
	// KYTY_LOOKAHEAD=1 (on) or =verify; anything else is off.
	static Mode EnvMode();

	struct Hooks {
		bool (*read)(uint64_t address, void* data, uint64_t size)   = nullptr; // Any thread.
		const void* (*view)(uint64_t address, uint64_t size)        = nullptr; // Command thread.
		bool (*gpu_owned)(uint64_t address, uint64_t size)          = nullptr; // Command thread.
	};

	struct Resolution {
		LookaheadJob                                 job;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		GuestReadLog                                 log;
	};

	enum class Outcome : uint8_t { Hit, MissNone, MissChanged, MissGpu, MissUnread };

	// Without `thread` nothing resolves until ResolvePending() (tests).
	Lookahead(Mode mode, Hooks hooks, bool thread = true);
	~Lookahead();
	Lookahead(const Lookahead&)            = delete;
	Lookahead& operator=(const Lookahead&) = delete;

	[[nodiscard]] Mode GetMode() const { return m_mode; }
	// Runtime switch (toggle lookahead): off drops everything queued and resolved.
	void SetActive(bool active);
	[[nodiscard]] bool Active() const { return m_active.load(std::memory_order_relaxed); }

	static uint64_t Fingerprint(const void* program, uint64_t shader_base, uint64_t indirect_args,
	                            std::span<const uint32_t> workgroup_counts,
	                            std::span<const uint32_t> user_data);

	// Any thread. Sets the job's sequence and fingerprint.
	void Enqueue(LookaheadJob&& job);
	// Command thread: the checked resolution for these inputs, or null (outcome says why).
	std::unique_ptr<Resolution> Take(const void* program,
	                                 const ShaderRecompiler::IR::SrtRuntime& runtime,
	                                 Outcome& outcome);
	// Runs the queued jobs on the calling thread (tests, or a Lookahead without a thread).
	void ResolvePending();

	static bool SameResult(const ShaderRecompiler::IR::ResourceSnapshot&       a_resources,
	                       const ShaderRecompiler::IR::ResourceSpecialization& a_specialization,
	                       const ShaderRecompiler::IR::ResourceSnapshot&       b_resources,
	                       const ShaderRecompiler::IR::ResourceSpecialization& b_specialization);

	struct Stats {
		uint64_t queued = 0, resolved = 0, failed = 0, heap = 0, dropped = 0;
	};
	[[nodiscard]] Stats GetStats() const;

	// Bounds: jobs waiting, resolutions waiting, and how far behind the command thread's last
	// hit a job may be before it is dropped unresolved.
	static constexpr size_t   MaxQueued     = 65536;
	static constexpr size_t   MaxResolved   = 16384;
	static constexpr uint64_t MaxLag        = 32768;

private:
	enum class Result : uint8_t { Ok, Failed, Heap };
	Result Resolve(const LookaheadJob& job, Resolution& out);
	void   Run(const std::stop_token& stop);
	// Resolutions the command thread passed MaxLag sequences ago; m_table_mutex held.
	size_t PurgePassed();
	void   Store(std::unique_ptr<Resolution> resolution, const std::stop_token* stop);
	void   ResolveOne(LookaheadJob&& job, const std::stop_token* stop);

	const Mode  m_mode;
	const Hooks m_hooks;

	std::atomic<bool>     m_active {true};
	std::atomic<uint64_t> m_sequence {0};
	std::atomic<uint64_t> m_used_sequence {0}; // The newest job sequence the command thread used.

	mutable std::mutex       m_queue_mutex;
	std::condition_variable  m_queue_ready;
	std::deque<LookaheadJob> m_queue;

	std::mutex              m_table_mutex;
	std::condition_variable m_table_space;
	std::unordered_map<uint64_t, std::deque<std::unique_ptr<Resolution>>> m_table;
	std::deque<std::pair<uint64_t, uint64_t>> m_order; // (fingerprint, sequence), oldest first.
	size_t                                    m_resolved_count = 0;

	// Resolve thread only: per-plan scratch, and programs whose walks keep failing.
	struct ProgramState {
		uint32_t failures  = 0;
		uint32_t successes = 0;
	};
	std::unordered_map<const void*, ProgramState> m_programs;
	std::unordered_set<const void*>               m_skipped;
	std::unordered_map<const ShaderRecompiler::IR::ResourcePlan*,
	                   std::unique_ptr<ShaderRecompiler::IR::WalkScratch>>
	    m_scratch;

	std::atomic<uint64_t> m_queued {0}, m_resolved {0}, m_failed {0}, m_heap {0}, m_dropped {0};

	std::jthread m_thread;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_LOOKAHEAD_H_
