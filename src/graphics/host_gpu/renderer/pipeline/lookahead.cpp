#include "graphics/host_gpu/renderer/pipeline/lookahead.h"

#include "common/perfSummary.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <xxhash.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <pthread.h>
#endif

namespace Libs::Graphics {

namespace IR = ShaderRecompiler::IR;
using Common::PerfSummary::Add;
using Common::PerfSummary::Counter;

namespace {

// Logs what the walk reads through its memory callback. A read that fails fails the walk ahead:
// the command thread cannot check a failed read.
struct LoggedReader {
	Lookahead::Hooks hooks;
	GuestReadLog*    log    = nullptr;
	bool             failed = false;

	static bool Read(void* user, uint64_t address, std::span<uint32_t> values) {
		auto& self = *static_cast<LoggedReader*>(user);
		if (values.empty()) return false;
		if (!self.hooks.read(address, values.data(), values.size_bytes())) {
			self.failed = true;
			return false;
		}
		self.log->AddRange(address, values);
		return true;
	}
};

// The image heap and the heap-row reports belong to the command thread. A walk that asks for
// them is given up: declining changes what the walk produces.
struct DecliningImageHeap final: IR::ImageHeapSource {
	bool asked = false;
	bool Acquire(const IR::ImageHeapRequest&, IR::ImageHeapView&) override {
		asked = true;
		return false;
	}
};

struct DecliningHeapRows final: IR::HeapRowSource {
	bool asked = false;
	bool Acquire(uint64_t, uint32_t, uint64_t, IR::HeapRowView&) override {
		asked = true;
		return false;
	}
	void Reject(uint64_t, uint32_t, uint32_t, const IR::DescriptorValue&, std::string_view) override {}
};

// A program whose walks ahead fail this often, and never succeed, is not walked ahead again.
constexpr uint32_t MaxFailuresWithoutSuccess = 8;

template <typename T>
bool SameVectors(const std::vector<T>& a, const std::vector<T>& b) {
	return a == b;
}

} // namespace

bool LookaheadJob::Matches(const void* other_program, const IR::SrtRuntime& runtime) const {
	return program == other_program && shader_base == runtime.shader_base &&
	       indirect_args_address == runtime.indirect_args_address &&
	       std::ranges::equal(UserData(), runtime.user_data) &&
	       std::ranges::equal(WorkgroupCounts(), runtime.workgroup_counts);
}

Lookahead::Mode Lookahead::EnvMode() {
	const char* value = std::getenv("KYTY_LOOKAHEAD");
	if (value == nullptr) return Mode::Off;
	if (std::strcmp(value, "1") == 0) return Mode::On;
	if (std::strcmp(value, "verify") == 0) return Mode::Verify;
	return Mode::Off;
}

Lookahead::Lookahead(Mode mode, Hooks hooks, bool thread): m_mode(mode), m_hooks(hooks) {
	if (thread && mode != Mode::Off) {
		m_thread = std::jthread([this](const std::stop_token& stop) { Run(stop); });
	}
}

Lookahead::~Lookahead() {
	if (m_thread.joinable()) {
		m_thread.request_stop();
		{
			std::scoped_lock lock(m_queue_mutex);
		}
		m_queue_ready.notify_all();
		m_thread.join();
	}
}

void Lookahead::SetActive(bool active) {
	if (m_active.exchange(active, std::memory_order_relaxed) == active || active) return;
	size_t dropped = 0;
	{
		std::scoped_lock lock(m_queue_mutex);
		dropped += m_queue.size();
		m_queue.clear();
	}
	{
		std::scoped_lock lock(m_table_mutex);
		dropped += m_resolved_count;
		m_table.clear();
		m_order.clear();
		m_resolved_count = 0;
	}
	m_dropped += dropped;
	Add(Counter::LaDropped, dropped);
}

uint64_t Lookahead::Fingerprint(const void* program, uint64_t shader_base, uint64_t indirect_args,
                                std::span<const uint32_t> workgroup_counts,
                                std::span<const uint32_t> user_data) {
	std::array<uint32_t, 8 + 3 + 40> words {};
	const auto put = [&](size_t at, uint64_t value) {
		words[at]      = static_cast<uint32_t>(value);
		words[at + 1u] = static_cast<uint32_t>(value >> 32u);
	};
	put(0, reinterpret_cast<uint64_t>(program));
	put(2, shader_base);
	put(4, indirect_args);
	const auto counts = workgroup_counts.first(std::min<size_t>(workgroup_counts.size(), 3u));
	const auto data   = user_data.first(std::min<size_t>(user_data.size(), 40u));
	words[6] = static_cast<uint32_t>(counts.size());
	words[7] = static_cast<uint32_t>(data.size());
	std::ranges::copy(counts, words.begin() + 8);
	std::ranges::copy(data, words.begin() + 11);
	return XXH3_64bits(words.data(), (11u + data.size()) * sizeof(uint32_t));
}

void Lookahead::Enqueue(LookaheadJob&& job) {
	if (!Active() || job.plan == nullptr) return;
	job.sequence    = m_sequence.fetch_add(1, std::memory_order_relaxed) + 1;
	job.fingerprint = Fingerprint(job.program, job.shader_base, job.indirect_args_address,
	                              job.WorkgroupCounts(), job.UserData());
	size_t dropped = 0;
	{
		// Full: the newest job goes (the oldest are the ones the command thread needs next).
		std::scoped_lock lock(m_queue_mutex);
		if (m_queue.size() >= MaxQueued) {
			dropped++;
		} else {
			m_queue.push_back(std::move(job));
		}
	}
	m_queued++;
	Add(Counter::LaJobs);
	if (dropped != 0) {
		m_dropped += dropped;
		Add(Counter::LaDropped, dropped);
	}
	m_queue_ready.notify_one();
}

Lookahead::Result Lookahead::Resolve(const LookaheadJob& job, Resolution& out) {
	auto& slot = m_scratch[job.plan];
	if (slot == nullptr) {
		if (m_scratch.size() > 4096) {
			m_scratch.clear();
			return Resolve(job, out);
		}
		slot = std::make_unique<IR::WalkScratch>();
	}
	auto& scratch = *slot;
	scratch.Prepare(*job.plan);
	scratch.loads.clear();
	out.job = job;
	out.log.Clear();
	LoggedReader       reader {.hooks = m_hooks, .log = &out.log};
	DecliningImageHeap heap;
	DecliningHeapRows  rows;
	const IR::SrtRuntime runtime {
	    .user_data                  = job.UserData(),
	    .shader_base                = job.shader_base,
	    .userdata                   = &reader,
	    .read_specialization_memory = &LoggedReader::Read,
	    .workgroup_counts           = job.WorkgroupCounts(),
	    .indirect_args_address      = job.indirect_args_address,
	    .image_heap                 = &heap,
	    .heap_rows                  = &rows,
	};
	bool ok = false;
	{
		IR::ScopedWalkScratch scoped(scratch);
		ok = IR::MaterializeResources(*job.plan, runtime, out.resources, out.specialization);
	}
	if (heap.asked || rows.asked) return Result::Heap;
	if (!ok || reader.failed) return Result::Failed;
	out.log.AddLoads(scratch.loads);
	return out.log.Finish() ? Result::Ok : Result::Failed;
}

size_t Lookahead::PurgePassed() {
	const auto used    = m_used_sequence.load(std::memory_order_relaxed);
	size_t     dropped = 0;
	while (!m_order.empty() && m_order.front().second + MaxLag < used) {
		const auto [fingerprint, sequence] = m_order.front();
		m_order.pop_front();
		const auto found = m_table.find(fingerprint);
		if (found == m_table.end()) continue;
		auto&      list = found->second;
		const auto it   = std::ranges::find_if(
            list, [&](const auto& entry) { return entry->job.sequence == sequence; });
		if (it == list.end()) continue;
		list.erase(it);
		if (list.empty()) m_table.erase(found);
		m_resolved_count--;
		dropped++;
	}
	return dropped;
}

void Lookahead::Store(std::unique_ptr<Resolution> resolution, const std::stop_token* stop) {
	const auto fingerprint = resolution->job.fingerprint;
	const auto sequence    = resolution->job.sequence;
	size_t     dropped     = 0;
	{
		std::unique_lock lock(m_table_mutex);
		dropped += PurgePassed();
		// Full: the resolve thread waits for the command thread to take or pass resolutions
		// (dropping the oldest would drop exactly the ones it takes next).
		while (m_resolved_count >= MaxResolved) {
			if (stop == nullptr || stop->stop_requested() || !Active()) {
				dropped++;
				resolution.reset();
				break;
			}
			m_table_space.wait_for(lock, std::chrono::milliseconds(2));
			dropped += PurgePassed();
		}
		if (resolution != nullptr) {
			m_table[fingerprint].push_back(std::move(resolution));
			m_order.emplace_back(fingerprint, sequence);
			m_resolved_count++;
		}
	}
	if (dropped != 0) {
		m_dropped += dropped;
		Add(Counter::LaDropped, dropped);
	}
}

void Lookahead::ResolveOne(LookaheadJob&& job, const std::stop_token* stop) {
	if (job.sequence + MaxLag < m_used_sequence.load(std::memory_order_relaxed) ||
	    m_skipped.contains(job.program)) {
		m_dropped++;
		Add(Counter::LaDropped);
		return;
	}
	auto resolution = std::make_unique<Resolution>();
	const auto start = std::chrono::steady_clock::now();
	const auto result = Resolve(job, *resolution);
	Add(Counter::LaResolveUs,
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
	                              std::chrono::steady_clock::now() - start)
	                              .count()));
	auto& state = m_programs[job.program];
	if (result != Result::Ok) {
		(result == Result::Heap ? m_heap : m_failed)++;
		Add(result == Result::Heap ? Counter::LaResolveHeap : Counter::LaResolveFails);
		if (++state.failures >= MaxFailuresWithoutSuccess && state.successes == 0) {
			m_skipped.insert(job.program);
		}
		return;
	}
	state.successes++;
	m_resolved++;
	Add(Counter::LaResolved);
	if (!Active()) return;
	Store(std::move(resolution), stop);
}

void Lookahead::ResolvePending() {
	const bool speculative = IR::SrtSpeculativeThread();
	IR::SetSrtSpeculativeThread(true);
	for (;;) {
		LookaheadJob job;
		{
			std::scoped_lock lock(m_queue_mutex);
			if (m_queue.empty()) break;
			job = std::move(m_queue.front());
			m_queue.pop_front();
		}
		ResolveOne(std::move(job), nullptr);
	}
	IR::SetSrtSpeculativeThread(speculative);
}

void Lookahead::Run(const std::stop_token& stop) {
#if defined(__linux__)
	pthread_setname_np(pthread_self(), "kyty_lookahead");
#endif
	IR::SetSrtSpeculativeThread(true);
	for (;;) {
		LookaheadJob job;
		{
			std::unique_lock lock(m_queue_mutex);
			m_queue_ready.wait(lock, [&] { return stop.stop_requested() || !m_queue.empty(); });
			if (stop.stop_requested()) return;
			job = std::move(m_queue.front());
			m_queue.pop_front();
		}
		ResolveOne(std::move(job), &stop);
	}
}

std::unique_ptr<Lookahead::Resolution> Lookahead::Take(const void* program,
                                                       const IR::SrtRuntime& runtime,
                                                       Outcome& outcome) {
	outcome = Outcome::MissNone;
	const auto fingerprint = Fingerprint(program, runtime.shader_base, runtime.indirect_args_address,
	                                     runtime.workgroup_counts, runtime.user_data);
	std::unique_ptr<Resolution> resolution;
	{
		std::scoped_lock lock(m_table_mutex);
		const auto found = m_table.find(fingerprint);
		if (found == m_table.end()) return nullptr;
		auto& list = found->second;
		const auto it = std::ranges::find_if(
		    list, [&](const auto& entry) { return entry->job.Matches(program, runtime); });
		if (it == list.end()) return nullptr;
		resolution = std::move(*it);
		list.erase(it);
		if (list.empty()) m_table.erase(found);
		m_resolved_count--;
	}
	m_table_space.notify_one();
	auto used = m_used_sequence.load(std::memory_order_relaxed);
	while (used < resolution->job.sequence &&
	       !m_used_sequence.compare_exchange_weak(used, resolution->job.sequence,
	                                              std::memory_order_relaxed)) {
	}
	struct Memory final: GuestReadLog::Memory {
		const Hooks& hooks;
		explicit Memory(const Hooks& h): hooks(h) {}
		const void* View(uint64_t address, uint64_t size) override {
			return hooks.view != nullptr ? hooks.view(address, size) : nullptr;
		}
		bool Read(uint64_t address, void* data, uint64_t size) override {
			return hooks.read(address, data, size);
		}
		bool GpuOwned(uint64_t address, uint64_t size) override {
			return hooks.gpu_owned != nullptr && hooks.gpu_owned(address, size);
		}
	} memory(m_hooks);
	uint64_t bytes = 0;
	const auto check = resolution->log.StillValid(memory, bytes);
	Add(Counter::LaRevalidateBytes, bytes);
	switch (check) {
		case GuestReadLog::Check::Valid: outcome = Outcome::Hit; return resolution;
		case GuestReadLog::Check::Changed: outcome = Outcome::MissChanged; break;
		case GuestReadLog::Check::GpuOwned: outcome = Outcome::MissGpu; break;
		case GuestReadLog::Check::Unreadable: outcome = Outcome::MissUnread; break;
	}
	return nullptr;
}

bool Lookahead::SameResult(const IR::ResourceSnapshot&       a_resources,
                           const IR::ResourceSpecialization& a_specialization,
                           const IR::ResourceSnapshot&       b_resources,
                           const IR::ResourceSpecialization& b_specialization) {
	return SameVectors(a_resources.buffers, b_resources.buffers) &&
	       SameVectors(a_resources.images, b_resources.images) &&
	       SameVectors(a_resources.samplers, b_resources.samplers) &&
	       SameVectors(a_resources.flattened_srt, b_resources.flattened_srt) &&
	       SameVectors(a_resources.user_data, b_resources.user_data) &&
	       SameVectors(a_resources.specialization_reads, b_resources.specialization_reads) &&
	       a_resources.uniform_fill == b_resources.uniform_fill &&
	       a_specialization == b_specialization;
}

Lookahead::Stats Lookahead::GetStats() const {
	return {m_queued.load(), m_resolved.load(), m_failed.load(), m_heap.load(), m_dropped.load()};
}

} // namespace Libs::Graphics
