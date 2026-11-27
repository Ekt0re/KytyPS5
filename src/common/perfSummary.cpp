#include "common/perfSummary.h"

#include <array>
#include <chrono>
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

// Names follow the Counter order.
constexpr std::array<const char*, N> g_names = {
    "flips",         "draws",       "dispatches",  "indirect_dispatches", "write_faults",
    "read_faults",   "buf_readbacks", "tex_readbacks", "tex_lookups",     "buf_lookups",
    "desc_writes",   "heap_desc_writes", "pipe_lookups", "pipe_created",  "amm_maps",
    "amm_unmaps",    "gpu_unmaps",  "heap_rebinds", "heap_lookups",       "heap_range_skips",
    "depth_recreates", "barriers",    "layout_transitions",
     "rb_calls",  "rb_batches",  "rb_submits",          "rb_fused",
    "rb_extra_bufs", "rb_neighbour", "rb_handler_us", "rb_flush_us",   "rb_submit_us",
    "rb_wait_us",    "rb_copy_us",
};

struct Sample {
	std::chrono::steady_clock::time_point time;
	std::array<uint64_t, N>               counters {};
	double   gpu_cpu_ms       = 0.0;
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

// The GPU thread, set by RegisterGpuThread. Its CPU time, instructions and context switches
// come from Linux interfaces; other hosts report the counters only ("na" for the rest).
std::atomic<bool> g_gpu_registered {false};
int               g_fd_user   = -1; // Instructions retired in user mode.
int               g_fd_kernel = -1; // In kernel mode; needs perf_event_paranoid <= 1.

#if defined(__linux__)

clockid_t g_gpu_clock    = 0;
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
#if defined(__linux__)
	add(" gpu_cpu_ms=%.0f", gpu_ms);
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
	std::fflush(stdout);
}

void PrintTotal() {
	if (g_total_printed.exchange(true)) {
		return;
	}
	const auto now  = Take();
	const auto secs = std::chrono::duration<double>(now.time - g_first.time).count();
	char       tag[48];
	std::snprintf(tag, sizeof(tag), "total t=%.0f", secs);
	Print(tag, g_first, now);
}

} // namespace

void Start() {
	std::lock_guard lock(g_mutex);
	if (g_started) {
		return;
	}
	const char* env = std::getenv("KYTY_PERF_SUMMARY");
	if (env == nullptr || env[0] == '\0' || env[0] == '0') {
		return;
	}
	g_started = true;
	if (const auto period = std::strtoul(env, nullptr, 10); period >= 2) {
		g_period = static_cast<uint32_t>(period);
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
