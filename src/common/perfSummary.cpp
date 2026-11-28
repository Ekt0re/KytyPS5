#include "common/perfSummary.h"

#include "common/faultStats.h"
#include "common/frameHoles.h"

#include <array>
#include <chrono>
#include <deque>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

namespace Common::PerfSummary {

namespace {

constexpr auto N = static_cast<size_t>(Counter::Count);
constexpr auto G = static_cast<size_t>(Gauge::Count);

constexpr std::array<const char*, G> g_gauge_names = {"park_mb_peak", "vram_mb_peak"};
// The highest level of each gauge since the start (for the total line).
std::array<std::atomic<uint64_t>, G> g_gauge_run_peaks {};

// Names follow the Counter order.
constexpr std::array<const char*, N> g_names = {
    "flips",         "draws",       "dispatches",  "indirect_dispatches", "write_faults",
    "read_faults",   "buf_readbacks", "tex_readbacks", "tex_lookups",     "buf_lookups",
    "desc_writes",   "heap_desc_writes", "pipe_lookups", "pipe_created",  "amm_maps",
    "amm_unmaps",    "gpu_unmaps",  "heap_rebinds", "heap_lookups",       "heap_range_skips",
    "depth_recreates", "depth_recreate_us", "img_created", "img_destroyed", "park_hits",
    "park_misses",   "park_cap_evicts", "barriers",    "layout_transitions",
     "rb_calls",  "rb_batches",  "rb_submits",          "rb_fused",
    "rb_extra_bufs", "rb_neighbour", "rb_handler_us", "rb_flush_us",   "rb_submit_us",
    "rb_wait_us",    "rb_copy_us",    "rb_write_calls", "rb_write_us",
    "rb_queue_ticks", "rb_prod_done", "rb_prod_pending", "rb_prod_open", "rb_after_ticks",
    "rb_prod_wait_us",
    "rb_path_cq", "rb_path_cq_wait_us", "rb_path_cq_total_us",
    "rb_path_fused", "rb_path_fused_wait_us", "rb_path_fused_total_us",
    "rb_path_det", "rb_path_det_wait_us", "rb_path_det_total_us",
    "rb_path_flush", "rb_path_flush_wait_us", "rb_path_flush_total_us",
    "rb_cq_submits", "rb_cq_fence_us", "rb_cq_gpu_busy",
    "zc_resolve_us", "zc_release_us", "zc_fences", "zc_fence_skips", "zc_fence_us",
    "zc_read_waits", "zc_wait_us", "zc_wait_drops", "zc_state_kib",
    "zc_open_waits", "zc_saved",
    "f_wait_gpu_us", "f_wait_guest_us", "f_materialize_us",
    "f_module_lookups", "f_module_hits", "f_module_misses",
    "f_handler_us", "f_invalidate_us", "f_protect_us", "f_lock_wait_us", "f_fast_skips",
    "f_storm_marked", "f_storm_kept", "f_storm_expired",
    "rec_records", "rec_drains", "rec_drain_us",
    "rb_o_guest_calls", "rb_o_guest_waits", "rb_o_guest_wait_us", "rb_o_srt_calls", "rb_o_srt_waits", "rb_o_srt_wait_us", "rb_o_pm4_calls", "rb_o_pm4_waits", "rb_o_pm4_wait_us", "rb_o_cmdwr_calls", "rb_o_cmdwr_waits", "rb_o_cmdwr_wait_us", "rb_o_args_calls", "rb_o_args_waits", "rb_o_args_wait_us", "rb_o_shader_calls", "rb_o_shader_waits", "rb_o_shader_wait_us", "rb_o_tex_calls", "rb_o_tex_waits", "rb_o_tex_wait_us", "rb_o_other_calls", "rb_o_other_waits", "rb_o_other_wait_us",
    "rb_tick_early", "rb_tick_skipped", "rb_hot_notes", "rb_hot_flushes",
    "rb_mirror_hits", "rb_mirror_misses", "rb_mirror_wait_us", "rb_mirror_copies",
    "la_jobs", "la_resolved", "la_resolve_fails", "la_resolve_heap", "la_resolve_us", "la_dropped",
    "la_lookups", "la_hits", "la_miss_none", "la_miss_changed", "la_miss_gpu", "la_miss_unread",
    "la_reval_us", "la_reval_bytes", "la_verified", "la_verify_mismatches",
};

struct Sample {
	std::chrono::steady_clock::time_point time;
	std::array<uint64_t, N>               counters {};
	std::array<uint64_t, G>               gauges {};
	double   gpu_cpu_ms       = 0.0;
	double   rec_cpu_ms       = 0.0; // The recording thread (KYTY_RECORDER=1).
	uint64_t gpu_instr_user   = 0;
	uint64_t gpu_instr_kernel = 0;
	uint64_t gpu_csw_vol      = 0;
	uint64_t gpu_csw_invol    = 0;
	double   proc_cpu_ms      = 0.0;
	uint64_t proc_minflt      = 0;
	uint64_t proc_majflt      = 0;
	uint64_t proc_csw_vol     = 0;
	uint64_t proc_csw_invol   = 0;
};

std::mutex        g_mutex;
bool              g_started = false;
uint32_t          g_period  = 10;
Sample            g_first;
std::atomic<bool> g_total_printed {false};

// KYTY_FRAME_COST=1: the existing reporter also prints one [frame] line per interval. The warning
// fires once after ten seconds of consecutive windows with more module misses than hits.
bool   g_frame_cost     = false;
double g_frame_bad_secs = 0.0;
bool   g_frame_warned   = false;
// KYTY_FAULT_STATS=1: the existing reporter also prints one [fault] group per interval.
bool g_fault_stats = false;
// KYTY_FRAME_HOLES=1: completed frames hand their skip counts to this queue; only the reporter
// writes the lines. Bounded so a long run with many incomplete frames cannot grow without limit.
constexpr size_t         kMaxQueuedHoleLines = 64;
std::deque<std::string>  g_hole_lines;

// The GPU thread, set by RegisterGpuThread. Its CPU time, instructions and context switches
// come from Linux interfaces; other hosts report the counters only ("na" for the rest).
std::atomic<bool> g_gpu_registered {false};
int               g_fd_user   = -1; // Instructions retired in user mode.
int               g_fd_kernel = -1; // In kernel mode; needs perf_event_paranoid <= 1.

#if defined(__linux__)

clockid_t g_gpu_clock    = 0;
clockid_t         g_rec_clock    = 0;
double            g_rec_cpu_base = 0.0;
std::atomic<bool> g_rec_registered {false};
pid_t     g_gpu_tid      = 0;
double    g_gpu_cpu_base = 0.0;

int OpenInstructionCounter(bool kernel) {
	perf_event_attr attr {};
	attr.type           = PERF_TYPE_HARDWARE;
	attr.size           = sizeof(attr);
	attr.config         = PERF_COUNT_HW_INSTRUCTIONS;
	attr.exclude_user   = kernel ? 1 : 0;
	attr.exclude_kernel = kernel ? 0 : 1;
	attr.exclude_hv     = 1;
	// The calling thread on any CPU; not inherited by threads it starts.
	return static_cast<int>(syscall(SYS_perf_event_open, &attr, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
}

uint64_t ReadCounter(int fd) {
	uint64_t value = 0;
	if (fd < 0 || read(fd, &value, sizeof(value)) != static_cast<ssize_t>(sizeof(value))) {
		return 0;
	}
	return value;
}

double ClockMs(clockid_t clock) {
	timespec ts {};
	if (clock_gettime(clock, &ts) != 0) {
		return 0.0;
	}
	return static_cast<double>(ts.tv_sec) * 1e3 + static_cast<double>(ts.tv_nsec) * 1e-6;
}

void ReadThreadSwitches(pid_t tid, uint64_t& voluntary, uint64_t& involuntary) {
	char path[64];
	std::snprintf(path, sizeof(path), "/proc/self/task/%d/status", static_cast<int>(tid));
	std::FILE* file = std::fopen(path, "r");
	if (file == nullptr) {
		return;
	}
	char line[256];
	while (std::fgets(line, sizeof(line), file) != nullptr) {
		unsigned long long value = 0;
		if (std::sscanf(line, "voluntary_ctxt_switches: %llu", &value) == 1) {
			voluntary = value;
		} else if (std::sscanf(line, "nonvoluntary_ctxt_switches: %llu", &value) == 1) {
			involuntary = value;
		}
	}
	std::fclose(file);
}

// Thread and process figures of the host.
void TakeHost(Sample& s) {
	if (g_gpu_registered.load(std::memory_order_acquire)) {
		s.gpu_cpu_ms       = ClockMs(g_gpu_clock) - g_gpu_cpu_base;
		s.gpu_instr_user   = ReadCounter(g_fd_user);
		s.gpu_instr_kernel = ReadCounter(g_fd_kernel);
		ReadThreadSwitches(g_gpu_tid, s.gpu_csw_vol, s.gpu_csw_invol);
	}
	if (g_rec_registered.load(std::memory_order_acquire)) {
		s.rec_cpu_ms = ClockMs(g_rec_clock) - g_rec_cpu_base;
	}
	rusage usage {};
	if (getrusage(RUSAGE_SELF, &usage) == 0) {
		s.proc_cpu_ms = static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1e3 +
		                static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) * 1e-3;
		s.proc_minflt    = static_cast<uint64_t>(usage.ru_minflt);
		s.proc_majflt    = static_cast<uint64_t>(usage.ru_majflt);
		s.proc_csw_vol   = static_cast<uint64_t>(usage.ru_nvcsw);
		s.proc_csw_invol = static_cast<uint64_t>(usage.ru_nivcsw);
	}
}

#else

void TakeHost(Sample& /*s*/) {}

#endif

Sample Take() {
	Sample s;
	s.time = std::chrono::steady_clock::now();
	for (size_t i = 0; i < N; i++) {
		s.counters[i] = Detail::g_counters[i].load(std::memory_order_relaxed);
	}
	// A gauge's peak restarts with each sample.
	for (size_t i = 0; i < G; i++) {
		s.gauges[i]     = Detail::g_gauge_peaks[i].exchange(0, std::memory_order_relaxed);
		auto& run_peak  = g_gauge_run_peaks[i];
		auto  seen      = run_peak.load(std::memory_order_relaxed);
		while (seen < s.gauges[i] &&
		       !run_peak.compare_exchange_weak(seen, s.gauges[i], std::memory_order_relaxed)) {
		}
	}
	TakeHost(s);
	return s;
}

void Print(const char* tag, const Sample& from, const Sample& to) {
	const auto count = [&](Counter c) {
		return to.counters[static_cast<size_t>(c)] - from.counters[static_cast<size_t>(c)];
	};
	const double secs   = std::chrono::duration<double>(to.time - from.time).count();
	const auto   flips  = count(Counter::Flips);
	const auto   draws  = count(Counter::Draws);
	const auto   ops    = draws + count(Counter::Dispatches);
	const double frames = flips != 0 ? static_cast<double>(flips) : 1.0;
	const double gpu_ms = to.gpu_cpu_ms - from.gpu_cpu_ms;
	const auto   user   = to.gpu_instr_user - from.gpu_instr_user;
	const auto   kernel = to.gpu_instr_kernel - from.gpu_instr_kernel;

	std::string line = "PerfSummary ";
	char        text[160];
	const auto  add = [&](const char* format, auto... args) {
		std::snprintf(text, sizeof(text), format, args...);
		line += text;
	};
	// Absolute values of the interval.
	add("%s secs=%.1f", tag, secs);
	for (size_t i = 0; i < N; i++) {
		add(" %s=%" PRIu64, g_names[i], to.counters[i] - from.counters[i]);
	}
	for (size_t i = 0; i < G; i++) {
		add(" %s=%" PRIu64, g_gauge_names[i], to.gauges[i]);
	}
#if defined(__linux__)
	add(" gpu_cpu_ms=%.0f", gpu_ms);
	if (g_rec_registered.load(std::memory_order_acquire)) {
		// Busy share: CPU time of the recording thread per wall second (its short spins count).
		const double rec_ms = to.rec_cpu_ms - from.rec_cpu_ms;
		add(" rec_cpu_ms=%.0f rec_busy=%.3f", rec_ms, rec_ms / (secs > 0.0 ? secs * 1e3 : 1.0));
	}
#else
	line += " gpu_cpu_ms=na";
#endif
	if (g_fd_user >= 0) {
		add(" gpu_instr_user=%" PRIu64, user);
	} else {
		line += " gpu_instr_user=na";
	}
	if (g_fd_kernel >= 0) {
		add(" gpu_instr_kernel=%" PRIu64, kernel);
	} else {
		line += " gpu_instr_kernel=na";
	}
	add(" gpu_csw_vol=%" PRIu64 " gpu_csw_invol=%" PRIu64, to.gpu_csw_vol - from.gpu_csw_vol,
	    to.gpu_csw_invol - from.gpu_csw_invol);
	add(" proc_cpu_ms=%.0f proc_minflt=%" PRIu64 " proc_majflt=%" PRIu64, to.proc_cpu_ms - from.proc_cpu_ms,
	    to.proc_minflt - from.proc_minflt, to.proc_majflt - from.proc_majflt);
	add(" proc_csw_vol=%" PRIu64 " proc_csw_invol=%" PRIu64, to.proc_csw_vol - from.proc_csw_vol,
	    to.proc_csw_invol - from.proc_csw_invol);
	// Per guest frame; an interval without a flip reports its values per one frame.
	add(" | fps=%.2f gpu_cpu_ms/f=%.2f", static_cast<double>(flips) / (secs > 0.0 ? secs : 1.0),
	    gpu_ms / frames);
	if (g_fd_user >= 0) {
		add(" gpu_Minstr_user/f=%.2f", static_cast<double>(user) / frames * 1e-6);
	}
	if (g_fd_kernel >= 0) {
		add(" gpu_Minstr_kernel/f=%.2f", static_cast<double>(kernel) / frames * 1e-6);
	}
	add(" gpu_csw_vol/f=%.1f gpu_csw_invol/f=%.1f",
	    static_cast<double>(to.gpu_csw_vol - from.gpu_csw_vol) / frames,
	    static_cast<double>(to.gpu_csw_invol - from.gpu_csw_invol) / frames);
	for (size_t i = 1; i < N; i++) {
		add(" %s/f=%.1f", g_names[i], static_cast<double>(to.counters[i] - from.counters[i]) / frames);
	}
	add(" proc_cpu_ms/f=%.1f proc_minflt/f=%.0f proc_majflt/f=%.1f",
	    (to.proc_cpu_ms - from.proc_cpu_ms) / frames,
	    static_cast<double>(to.proc_minflt - from.proc_minflt) / frames,
	    static_cast<double>(to.proc_majflt - from.proc_majflt) / frames);
	// Per synchronous download: handler time per ReadMemory request and submits per download.
	if (const auto calls = count(Counter::RbCalls); calls != 0) {
		add(" rb_per_fault_us=%.1f", static_cast<double>(count(Counter::RbHandlerUs)) /
		                                 static_cast<double>(calls));
	}
	if (const auto batches = count(Counter::RbBatches); batches != 0) {
		add(" rb_submits/batch=%.2f", static_cast<double>(count(Counter::RbSubmits)) /
		                                  static_cast<double>(batches));
	}
	// Per draw or dispatch ("op").
	if (ops != 0) {
		add(" | gpu_cpu_us/op=%.1f", gpu_ms * 1e3 / static_cast<double>(ops));
		if (g_fd_user >= 0) {
			add(" gpu_instr_user/op=%.0f", static_cast<double>(user) / static_cast<double>(ops));
		}
		if (g_fd_kernel >= 0) {
			add(" gpu_instr_kernel/op=%.0f", static_cast<double>(kernel) / static_cast<double>(ops));
		}
	}
	line += '\n';
	std::fputs(line.c_str(), stdout);
	if (g_frame_cost) {
		const auto wait_gpu     = count(Counter::FrameWaitGpuUs);
		const auto wait_guest   = count(Counter::FrameWaitGuestUs);
		const auto materialize  = count(Counter::FrameMaterializeUs);
		const auto lookups      = count(Counter::FrameModuleLookups);
		const auto hits         = count(Counter::FrameModuleHits);
		const auto misses       = count(Counter::FrameModuleMisses);
		const auto frame_line = Detail::FormatFrameLine(secs, flips, wait_gpu, wait_guest,
		                                              materialize, lookups, hits, misses);
		std::fputs(frame_line.c_str(), stdout);
		// A module miss streak is only interesting after the store has had time to settle: one
		// window of more misses than hits is normal while it loads. Ten consecutive seconds is
		// the same rule our original frame reporter used.
		if (misses > hits && (hits + misses) != 0) {
			g_frame_bad_secs += secs;
		} else {
			g_frame_bad_secs = 0.0;
		}
		const auto warning = Detail::FormatFrameWarning(hits, misses, g_frame_bad_secs);
		if (!warning.empty() && !g_frame_warned) {
			g_frame_warned = true;
			std::fputs(warning.c_str(), stdout);
		}
	}
	if (g_fault_stats) {
		const auto writes   = count(Counter::WriteFaults);
		const auto reads    = count(Counter::ReadFaults);
		const auto handler  = count(Counter::FaultHandlerUs);
		const auto invalidate = count(Counter::FaultInvalidateUs);
		const auto protect  = count(Counter::FaultProtectUs);
		const auto lock_wait = count(Counter::FaultLockWaitUs);
		const auto fast     = count(Counter::FaultFastSkips);
		const auto marked   = count(Counter::FaultStormMarked);
		const auto kept     = count(Counter::FaultStormKept);
		const auto expired  = count(Counter::FaultStormExpired);
		const auto line2    = Detail::FormatFaultLine(secs, writes, reads, handler, invalidate,
		                                          protect, lock_wait, fast, marked, kept, expired);
		std::fputs(line2.c_str(), stdout);
		// The hot-block table is drained here, never from the fault handler. Blocks are noted at
		// the existing HandleFault counting site, so the rate and the blocks describe the same
		// interval without counting faults twice.
		std::array<Detail::FaultBlock, FaultStats::FaultHotBlocks::kTop> blocks {};
		uint64_t dropped = 0;
		const auto used  = FaultStats::g_hot_blocks.TakeTop(blocks, dropped);
		const auto hot   = Detail::FormatFaultHotLine(blocks.data(), used, dropped);
		std::fputs(hot.c_str(), stdout);
	}
	if (Detail::g_frame_holes.load(std::memory_order_relaxed)) {
		std::deque<std::string> pending;
		{
			std::lock_guard lock(g_mutex);
			pending.swap(g_hole_lines);
		}
		for (const auto& hole: pending) {
			std::fputs(hole.c_str(), stdout);
			std::fputc('\n', stdout);
		}
	}
	std::fflush(stdout);
}

void PrintTotal() {
	if (g_total_printed.exchange(true)) {
		return;
	}
	auto now = Take();
	for (size_t i = 0; i < G; i++) {
		now.gauges[i] = g_gauge_run_peaks[i].load(std::memory_order_relaxed);
	}
	const auto secs = std::chrono::duration<double>(now.time - g_first.time).count();
	char       tag[48];
	std::snprintf(tag, sizeof(tag), "total t=%.0f", secs);
	Print(tag, g_first, now);
}

} // namespace

void NoteFrameHolesFrame(uint64_t frame) {
	if (!FrameHolesEnabled()) {
		return;
	}
	const auto snap = FrameHoles::Global().TakeSnapshot();
	const auto line = Detail::FormatFrameHolesLine(frame, snap);
	if (line.empty()) {
		return;
	}
	std::lock_guard lock(g_mutex);
	if (g_hole_lines.size() >= kMaxQueuedHoleLines) {
		g_hole_lines.pop_front();
	}
	g_hole_lines.push_back(line);
}

namespace Detail {

std::string FormatFrameLine(double secs, uint64_t frames, uint64_t wait_gpu_us,
                            uint64_t wait_guest_us, uint64_t materialize_us,
                            uint64_t module_lookups, uint64_t module_hits,
                            uint64_t module_misses) {
	const auto per_frame_ms = [frames](uint64_t micros) {
		return frames == 0 ? 0.0
		                   : static_cast<double>(micros) / static_cast<double>(frames) * 1e-3;
	};
	const auto total_ms = [](uint64_t micros) { return static_cast<double>(micros) * 1e-3; };
	char       text[512];
	std::snprintf(text, sizeof(text),
	    "[frame] secs=%.1f frames=%" PRIu64
	    " wait_gpu_ms=%.3f wait_guest_ms=%.3f materialize_ms=%.3f"
	    " ms/frame wait_gpu=%.3f wait_guest=%.3f materialize=%.3f"
	    " module_lookups=%" PRIu64 " module_hits=%" PRIu64 " module_misses=%" PRIu64 "\n",
	    secs, frames, total_ms(wait_gpu_us), total_ms(wait_guest_us), total_ms(materialize_us),
	    per_frame_ms(wait_gpu_us), per_frame_ms(wait_guest_us), per_frame_ms(materialize_us),
	    module_lookups, module_hits, module_misses);
	return text;
}

std::string FormatFrameWarning(uint64_t hits, uint64_t misses, double sustained_seconds) {
	if (sustained_seconds < 10.0 || misses <= hits || (hits + misses) == 0) {
		return {};
	}
	char text[512];
	std::snprintf(text, sizeof(text),
	    "[frame] WARNING module-misses: module lookups missed more often than they hit for "
	    "%.0f s (last window hits=%" PRIu64 " misses=%" PRIu64
	    "); modules that exist are not found, each use requests a new translation\n",
	    sustained_seconds, hits, misses);
	return text;
}

std::string FormatFaultLine(double secs, uint64_t writes, uint64_t reads, uint64_t handler_us,
                            uint64_t invalidate_us, uint64_t protect_us, uint64_t lock_wait_us,
                            uint64_t fast_skips, uint64_t marked, uint64_t kept,
                            uint64_t expired) {
	const auto per_s = [secs](uint64_t value) {
		return secs > 0.0 ? static_cast<double>(value) / secs : 0.0;
	};
	const auto ms_per_s = [secs](uint64_t micros) {
		return secs > 0.0 ? static_cast<double>(micros) / 1e3 / secs : 0.0;
	};
	char text[512];
	std::snprintf(text, sizeof(text),
	    "[fault] secs=%.1f write=%.0f/s read=%.0f/s handler_ms/s=%.1f invalidate_ms/s=%.1f "
	    "protect_ms/s=%.1f lock_wait_ms/s=%.1f fast_skip=%.0f/s "
	    "storm(marked=%" PRIu64 " kept=%" PRIu64 " expired=%" PRIu64 ")\n",
	    secs, per_s(writes), per_s(reads), ms_per_s(handler_us), ms_per_s(invalidate_us),
	    ms_per_s(protect_us), ms_per_s(lock_wait_us), per_s(fast_skips), marked, kept, expired);
	return text;
}

std::string FormatFaultHotLine(const FaultBlock* blocks, size_t count, uint64_t dropped) {
	std::string line = "[fault] top 64K blocks:";
	char        text[64];
	for (size_t i = 0; i < count; i++) {
		std::snprintf(text, sizeof(text), " 0x%" PRIx64 "=%" PRIu64, blocks[i].address,
		              blocks[i].count);
		line += text;
	}
	if (dropped != 0) {
		std::snprintf(text, sizeof(text), " (untracked %" PRIu64 ")", dropped);
		line += text;
	}
	line += '\n';
	return line;
}

std::string FormatFrameHolesLine(uint64_t frame, const FrameHoles::Snapshot& snapshot) {
	return FrameHoles::Tracker::FormatLine(frame, snapshot);
}

} // namespace Detail

void Start() {
	std::lock_guard lock(g_mutex);
	if (g_started) {
		return;
	}
	const char* env        = std::getenv("KYTY_PERF_SUMMARY");
	const char* frame_env  = std::getenv("KYTY_FRAME_COST");
	const bool  frame_cost = frame_env != nullptr && frame_env[0] != '\0' && frame_env[0] != '0';
	const char* fault_env  = std::getenv("KYTY_FAULT_STATS");
	const bool  fault_stats =
	    fault_env != nullptr && fault_env[0] != '\0' && fault_env[0] != '0';
	const char* holes_env  = std::getenv("KYTY_FRAME_HOLES");
	const bool  frame_holes =
	    holes_env != nullptr && holes_env[0] != '\0' && holes_env[0] != '0';
	if ((env == nullptr || env[0] == '\0' || env[0] == '0') && !frame_cost && !fault_stats &&
	    !frame_holes) {
		return;
	}
	g_started     = true;
	g_frame_cost  = frame_cost;
	g_fault_stats = fault_stats;
	Detail::g_frame_holes.store(frame_holes, std::memory_order_relaxed);
	FrameHoles::SetEnabled(frame_holes);
	bool explicit_period = false;
	if (env != nullptr && env[0] != '\0' && env[0] != '0') {
		if (const auto period = std::strtoul(env, nullptr, 10); period >= 2) {
			g_period         = static_cast<uint32_t>(period);
			explicit_period = true;
		}
	}
	// The frame, fault and hole lines are a five-second window unless the caller chose a period.
	if ((frame_cost || fault_stats || frame_holes) && !explicit_period) {
		g_period = 5;
	}
	Detail::g_enabled.store(true, std::memory_order_relaxed);
	g_first = Take();
	std::atexit(PrintTotal);
	std::at_quick_exit(PrintTotal);
	std::thread([] {
		auto last = g_first;
		for (uint64_t index = 1;; index++) {
			std::this_thread::sleep_for(std::chrono::seconds(g_period));
			if (g_total_printed.load()) {
				return;
			}
			const auto now = Take();
			char       tag[48];
			std::snprintf(tag, sizeof(tag), "t=%" PRIu64, index * g_period);
			Print(tag, last, now);
			last = now;
		}
	}).detach();
}

void RegisterRecorderThread() {
#if defined(__linux__)
	if (!Detail::g_enabled.load(std::memory_order_relaxed) || g_rec_registered.load(std::memory_order_acquire)) {
		return;
	}
	pthread_getcpuclockid(pthread_self(), &g_rec_clock);
	g_rec_cpu_base = ClockMs(g_rec_clock);
	g_rec_registered.store(true, std::memory_order_release);
#endif
}

void RegisterGpuThread() {
	if (!Detail::g_enabled.load(std::memory_order_relaxed) ||
	    g_gpu_registered.load(std::memory_order_acquire)) {
		return;
	}
#if defined(__linux__)
	pthread_getcpuclockid(pthread_self(), &g_gpu_clock);
	g_gpu_tid      = static_cast<pid_t>(syscall(SYS_gettid));
	g_gpu_cpu_base = ClockMs(g_gpu_clock);
	// perf_event_paranoid 2 allows user-mode counting only; above that (or without a PMU, as in
	// most virtual machines) the instruction fields read "na".
	g_fd_user   = OpenInstructionCounter(false);
	g_fd_kernel = OpenInstructionCounter(true);
	std::printf("PerfSummary: GPU thread tid %d, instruction counters: user %s, kernel %s\n",
	            static_cast<int>(g_gpu_tid), g_fd_user >= 0 ? "on" : "unavailable",
	            g_fd_kernel >= 0 ? "on" : "unavailable (perf_event_paranoid > 1?)");
	std::fflush(stdout);
#endif
	g_gpu_registered.store(true, std::memory_order_release);
	// Exit handlers run in reverse order of registration: registered again here, after the
	// emulator's own (which write the pipeline cache and can take seconds), the total prints first.
	std::atexit(PrintTotal);
	std::at_quick_exit(PrintTotal);
}

} // namespace Common::PerfSummary
