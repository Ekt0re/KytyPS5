#include "common/memAccount.h"

#include "common/hostStats.h"
#include "common/subsystems.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <memory>
#include <mutex>
#include <thread>

#if defined(__GLIBC__)
#include <malloc.h>
#endif
#if defined(__linux__)
#include <csignal>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
// clang-format off
#include <psapi.h>
// clang-format on
#endif

namespace Common::MemAccount {

namespace {

uint32_t ReadEnvUint(const char* name) {
	const char* env = std::getenv(name);
	if (env == nullptr || env[0] == '\0') return 0;
	return static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
}

uint32_t ReportSeconds() {
	static const uint32_t value = ReadEnvUint("KYTY_MEM_REPORT");
	return value;
}

uint32_t LimitGb() {
	static const uint32_t value = ReadEnvUint("KYTY_MEM_LIMIT_GB");
	return value;
}

bool Enabled() {
	return ReportSeconds() != 0 || LimitGb() != 0;
}

// Never destroyed: owners may update gauges during static destruction.
std::mutex& Mutex() {
	static auto* mutex = new std::mutex;
	return *mutex;
}
std::vector<std::pair<std::string, std::unique_ptr<Gauge>>>& Owners() {
	static auto* owners = new std::vector<std::pair<std::string, std::unique_ptr<Gauge>>>;
	return *owners;
}
std::mutex& ReadMutex() {
	static auto* mutex = new std::mutex;
	return *mutex;
}
std::vector<std::pair<std::string, ExternalReader>>& Externals() {
	static auto* externals = new std::vector<std::pair<std::string, ExternalReader>>;
	return *externals;
}

std::atomic<bool>                     g_drop_requested {false};
std::mutex                            g_thread_mutex;
std::condition_variable               g_thread_wake;
bool                                  g_stop = false;
std::thread                           g_thread;
std::atomic<uint64_t>                 g_peak_seen {0};
std::atomic<uint64_t>                 g_guest_committed {0};
std::atomic<bool>                     g_exit_printed {false};
std::atomic<const char*>              g_exit_reason {nullptr};
std::atomic<bool>                     g_term_requested {false};
std::chrono::steady_clock::time_point g_start = std::chrono::steady_clock::now();

std::string Mib(uint64_t bytes) {
	return fmt::format("{:.0f}MiB", static_cast<double>(bytes) / 1048576.0);
}

Sample TakeSample(bool with_heap = true) {
	Sample sample;
	sample.process         = ReadProcessMemory();
	sample.heap            = with_heap ? ReadHeapStats() : HeapStats {};
	sample.owners          = Snapshot();
	sample.zram            = ReadZram();
	sample.guest_committed = g_guest_committed.load(std::memory_order_relaxed);
	std::scoped_lock                                    read_lock(ReadMutex());
	std::vector<std::pair<std::string, ExternalReader>> readers;
	{
		std::scoped_lock lock(Mutex());
		readers = Externals();
	}
	for (const auto& [name, reader]: readers) {
		sample.external.emplace_back(name, reader());
	}
	return sample;
}

uint64_t ReadKb(std::string_view text, std::string_view key, bool* found = nullptr) {
	const auto at = text.find(key);
	if (at == std::string_view::npos) return 0;
	auto rest = text.substr(at + key.size());
	while (!rest.empty() && (rest.front() == ' ' || rest.front() == '\t')) rest.remove_prefix(1);
	uint64_t   value = 0;
	const auto res   = std::from_chars(rest.data(), rest.data() + rest.size(), value);
	if (res.ec == std::errc() && found != nullptr) *found = true;
	return value * 1024u;
}

#if defined(__linux__)
void OnTerm(int) {
	g_term_requested.store(true, std::memory_order_relaxed);
	::alarm(8);
}
#endif

void Run() {
	const uint64_t report_ms      = static_cast<uint64_t>(ReportSeconds()) * 1000u;
	const uint64_t limit          = static_cast<uint64_t>(LimitGb()) << 30;
	LimitGuard     guard(limit);
	const auto     begin          = std::chrono::steady_clock::now();
	uint64_t       last_report_ms = 0;
	std::vector<std::pair<std::string, uint64_t>> previous_holders;
	std::unique_lock                              lock(g_thread_mutex);
	while (!g_stop) {
		g_thread_wake.wait_for(lock, std::chrono::seconds(1), [] { return g_stop; });
		if (g_stop) break;
		const auto now_ms = static_cast<uint64_t>(
		    std::chrono::duration_cast<std::chrono::milliseconds>(
		        std::chrono::steady_clock::now() - begin)
		        .count());
		lock.unlock();
#if defined(__linux__)
		if (g_term_requested.load(std::memory_order_relaxed)) {
			NoteExitReason("sigterm");
			Subsystems::EmergencyShutdownActive();
			std::fflush(nullptr);
			std::signal(SIGTERM, SIG_DFL);
			std::raise(SIGTERM);
			std::_Exit(143);
		}
#endif
		if (report_ms != 0 && now_ms - last_report_ms >= report_ms) {
			last_report_ms    = now_ms;
			const auto sample = TakeSample(true);
			std::printf("%s\n", FormatLine(sample, "").c_str());
			std::fflush(stdout);
			previous_holders = Holders(sample);
		}
		if (limit != 0) {
			// Check every second using cheap process + optional zram reads only (no mallinfo2).
			const auto process = ReadProcessMemory();
			if (process.available) {
				const auto footprint = Footprint(process, ReadZram());
				const auto action    = guard.Step(now_ms, footprint);
				if (action != LimitGuard::Action::None) {
					const auto sample = TakeSample(true);
					const auto grew   = FormatGrowth(previous_holders, Holders(sample));
					if (action == LimitGuard::Action::DropCaches) {
						std::printf("%s\n", FormatLine(sample, "limit").c_str());
						std::printf("[mem] KYTY_MEM_LIMIT_GB=%u passed (footprint %s): dropping "
						            "caches; stopping in 10 s if still over. Grew: %s\n",
						            LimitGb(), Mib(footprint).c_str(), grew.c_str());
						std::fflush(stdout);
						g_drop_requested.store(true, std::memory_order_relaxed);
#if defined(__GLIBC__)
						malloc_trim(0);
#endif
					} else if (action == LimitGuard::Action::Stop) {
						std::printf("%s\n", FormatLine(sample, "limit-stop").c_str());
						std::printf("[mem] STOPPING: footprint %s still over KYTY_MEM_LIMIT_GB=%u "
						            "after 10 s. Grew: %s\n",
						            Mib(footprint).c_str(), LimitGb(), grew.c_str());
						NoteExitReason("limit-stop");
						std::fflush(nullptr);
						Subsystems::EmergencyShutdownActive();
						std::fflush(nullptr);
						std::_Exit(75);
					}
				}
			}
		}
		lock.lock();
	}
}

} // namespace

Gauge& Owner(std::string_view name) {
	std::scoped_lock lock(Mutex());
	for (auto& [key, gauge]: Owners()) {
		if (key == name) return *gauge;
	}
	return *Owners().emplace_back(std::string(name), std::make_unique<Gauge>()).second;
}

std::vector<Entry> Snapshot() {
	std::vector<Entry> entries;
	std::scoped_lock   lock(Mutex());
	entries.reserve(Owners().size());
	for (const auto& [name, gauge]: Owners()) {
		entries.push_back({name, gauge->Bytes(), gauge->Items()});
	}
	return entries;
}

void SetExternal(std::string_view name, ExternalReader reader) {
	std::scoped_lock lock(Mutex());
	for (auto& [key, existing]: Externals()) {
		if (key == name) {
			existing = std::move(reader);
			return;
		}
	}
	Externals().emplace_back(std::string(name), std::move(reader));
}

void ClearExternal(std::string_view name) {
	std::scoped_lock read_lock(ReadMutex());
	std::scoped_lock lock(Mutex());
	std::erase_if(Externals(), [&](const auto& entry) { return entry.first == name; });
}

ProcessMemory ParseProcStatus(std::string_view text) {
	ProcessMemory memory;
	memory.rss           = ReadKb(text, "VmRSS:", &memory.available);
	memory.swap          = ReadKb(text, "VmSwap:");
	memory.peak_rss      = ReadKb(text, "VmHWM:");
	memory.rss_anon      = ReadKb(text, "RssAnon:");
	memory.rss_file      = ReadKb(text, "RssFile:");
	memory.rss_shmem     = ReadKb(text, "RssShmem:");
	memory.private_bytes = memory.rss_anon;
	return memory;
}

ProcessMemory ReadProcessMemory() {
	ProcessMemory memory;
#if defined(__linux__)
	// Direct /proc/self/status read gives VmHWM and RSS split without HostStatsSampler's
	// per-thread /proc/self/task walk on the 1 s limit tick.
	std::ifstream file("/proc/self/status");
	if (file) {
		std::ostringstream text;
		text << file.rdbuf();
		memory = ParseProcStatus(text.str());
	}
#elif defined(_WIN32)
	PROCESS_MEMORY_COUNTERS_EX pmc {};
	pmc.cb = sizeof(pmc);
	if (K32GetProcessMemoryInfo(GetCurrentProcess(),
	                            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
	                            sizeof(pmc)) != 0) {
		memory.available     = true;
		memory.rss           = static_cast<uint64_t>(pmc.WorkingSetSize);
		memory.peak_rss      = static_cast<uint64_t>(pmc.PeakWorkingSetSize);
		memory.private_bytes = static_cast<uint64_t>(pmc.PrivateUsage);
		memory.swap          = pmc.PrivateUsage > pmc.WorkingSetSize
		                           ? static_cast<uint64_t>(pmc.PrivateUsage - pmc.WorkingSetSize)
		                           : 0;
	}
	MEMORYSTATUSEX sys {};
	sys.dwLength = sizeof(sys);
	if (GlobalMemoryStatusEx(&sys) != 0) {
		memory.sys_commit_total = static_cast<uint64_t>(sys.ullTotalPageFile);
		memory.sys_commit_used  = sys.ullTotalPageFile >= sys.ullAvailPageFile
		                              ? static_cast<uint64_t>(sys.ullTotalPageFile - sys.ullAvailPageFile)
		                              : 0;
	}
#elif defined(__APPLE__)
	// Reuse HostStatsSampler's Mach task_info(TASK_VM_INFO) path.
	static HostStatsSampler sampler;
	const auto              stats = sampler.Sample();
	if (stats.process_resident.has_value()) {
		memory.available     = true;
		memory.rss           = *stats.process_resident;
		memory.private_bytes = stats.process_memory.value_or(0);
		memory.swap          = stats.process_compressed.value_or(0);
		memory.rss_file      = stats.process_file_backed.value_or(0);
	}
#endif
	if (memory.available) {
		auto seen = g_peak_seen.load(std::memory_order_relaxed);
		while (memory.rss > seen && !g_peak_seen.compare_exchange_weak(seen, memory.rss)) {
		}
		memory.peak_rss = std::max(memory.peak_rss, g_peak_seen.load(std::memory_order_relaxed));
	}
	return memory;
}

HeapStats ReadHeapStats() {
	HeapStats stats;
#if defined(__GLIBC__) && defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 33)
	const auto info = mallinfo2();
	stats.known     = true;
	stats.in_use    = static_cast<uint64_t>(info.uordblks) + static_cast<uint64_t>(info.hblkhd);
	stats.free_held = static_cast<uint64_t>(info.fordblks);
#endif
#endif
	return stats;
}

ZramStats ParseZramMmStat(std::string_view text) {
	ZramStats   stats;
	uint64_t    fields[3] = {0, 0, 0};
	size_t      count     = 0;
	const char* p         = text.data();
	const char* end       = p + text.size();
	while (count < 3 && p < end) {
		while (p < end && (*p == ' ' || *p == '\t' || *p == '\n')) p++;
		const auto result = std::from_chars(p, end, fields[count]);
		if (result.ec != std::errc()) return stats;
		p = result.ptr;
		count++;
	}
	if (count < 3) return stats;
	stats.known      = true;
	stats.orig       = fields[0];
	stats.compressed = fields[2];
	return stats;
}

ZramStats ReadZram() {
	ZramStats sum;
#if defined(__linux__)
	std::error_code error;
	if (!std::filesystem::exists("/sys/block", error)) return sum;
	for (const auto& entry: std::filesystem::directory_iterator("/sys/block", error)) {
		if (entry.path().filename().string().rfind("zram", 0) != 0) continue;
		const auto mm_stat = entry.path() / "mm_stat";
		if (!std::filesystem::exists(mm_stat, error)) continue;
		std::ifstream file(mm_stat);
		if (!file) continue;
		std::ostringstream text;
		text << file.rdbuf();
		const auto one = ParseZramMmStat(text.str());
		if (one.known) {
			sum.known = true;
			sum.orig += one.orig;
			sum.compressed += one.compressed;
		}
	}
#endif
	return sum;
}

uint64_t Footprint(const ProcessMemory& process, const ZramStats& zram) {
	if (!zram.known || zram.orig == 0 || zram.compressed >= zram.orig) {
		return process.rss + process.swap;
	}
	const auto share = static_cast<double>(zram.compressed) / static_cast<double>(zram.orig);
	return process.rss + static_cast<uint64_t>(static_cast<double>(process.swap) * share);
}

void NoteGuestBacking(uint64_t bytes) {
	g_guest_committed.fetch_add(bytes, std::memory_order_relaxed);
}

std::vector<std::pair<std::string, uint64_t>> Holders(const Sample& sample) {
	std::vector<std::pair<std::string, uint64_t>> holders;
	if (sample.process.rss_anon != 0 || sample.process.rss_file != 0 ||
	    sample.process.rss_shmem != 0) {
		holders.emplace_back("rss-anon", sample.process.rss_anon);
		holders.emplace_back("rss-file", sample.process.rss_file);
		holders.emplace_back("rss-shmem(guest)", sample.process.rss_shmem);
	} else if (sample.process.available) {
		holders.emplace_back("rss", sample.process.rss);
		if (sample.process.private_bytes != 0) {
			holders.emplace_back("private", sample.process.private_bytes);
		}
	}
	if (sample.process.swap != 0) holders.emplace_back("swap", sample.process.swap);
	if (sample.heap.known) {
		holders.emplace_back("heap-in-use", sample.heap.in_use);
		holders.emplace_back("heap-free-held", sample.heap.free_held);
	}
	for (const auto& owner: sample.owners) holders.emplace_back(owner.name, owner.bytes);
	return holders;
}

std::string FormatGrowth(const std::vector<std::pair<std::string, uint64_t>>& before,
                         const std::vector<std::pair<std::string, uint64_t>>& after, size_t top) {
	std::vector<std::pair<std::string, int64_t>> grew;
	for (const auto& [name, now]: after) {
		uint64_t was = 0;
		for (const auto& [old_name, old_value]: before) {
			if (old_name == name) {
				was = old_value;
				break;
			}
		}
		if (now > was) grew.emplace_back(name, static_cast<int64_t>(now - was));
	}
	std::sort(grew.begin(), grew.end(),
	          [](const auto& a, const auto& b) { return a.second > b.second; });
	std::string text;
	for (size_t i = 0; i < grew.size() && i < top; i++) {
		text += fmt::format("{}{} +{}", i == 0 ? "" : ", ", grew[i].first,
		                    Mib(static_cast<uint64_t>(grew[i].second)));
	}
	return text.empty() ? "nothing" : text;
}

uint64_t Counted(const std::vector<Entry>& owners) {
	uint64_t sum = 0;
	for (const auto& owner: owners) sum += owner.bytes;
	return sum;
}

int64_t Unaccounted(uint64_t rss, uint64_t counted) {
	return rss >= counted ? static_cast<int64_t>(rss - counted)
	                      : -static_cast<int64_t>(counted - rss);
}

std::string FormatLine(const Sample& sample, std::string_view tag) {
	const auto elapsed =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
	const auto  counted = Counted(sample.owners);
	std::string line;
	if (sample.process.available) {
		const auto gap = Unaccounted(sample.process.rss, counted);
		line           = fmt::format(
            "[mem] {} t={:.0f}s rss={} private={} swap={} total={} peak-rss={} counted={} "
                      "unaccounted={}{}",
            tag, elapsed, Mib(sample.process.rss), Mib(sample.process.private_bytes),
            Mib(sample.process.swap), Mib(sample.process.rss + sample.process.swap),
            Mib(sample.process.peak_rss), Mib(counted), gap < 0 ? "-" : "",
            Mib(static_cast<uint64_t>(gap < 0 ? -gap : gap)));
	} else {
		line = fmt::format("[mem] {} t={:.0f}s rss=not-available counted={}", tag, elapsed,
		                   Mib(counted));
	}
	line += " |";
	for (const auto& owner: sample.owners) {
		line += fmt::format(" {}={}", owner.name, Mib(owner.bytes));
		if (owner.items != 0) line += fmt::format("({})", owner.items);
	}
	if (sample.heap.known) {
		line += fmt::format(" | heap-in-use={} heap-free-held={}", Mib(sample.heap.in_use),
		                    Mib(sample.heap.free_held));
	}
	if (sample.process.available) {
		line += fmt::format(" | footprint={}", Mib(Footprint(sample.process, sample.zram)));
		if (sample.zram.known) {
			line +=
			    fmt::format(" zram={}->{}", Mib(sample.zram.orig), Mib(sample.zram.compressed));
		}
		if (sample.process.rss_anon != 0 || sample.process.rss_file != 0 ||
		    sample.process.rss_shmem != 0) {
			line += fmt::format(" rss-anon={} rss-file={} rss-shmem={}",
			                    Mib(sample.process.rss_anon), Mib(sample.process.rss_file),
			                    Mib(sample.process.rss_shmem));
		}
		if (sample.process.sys_commit_total != 0) {
			line += fmt::format(" sys-commit={}/{}", Mib(sample.process.sys_commit_used),
			                    Mib(sample.process.sys_commit_total));
		}
	}
	if (sample.guest_committed != 0) {
		line += fmt::format(" | guest: committed={}", Mib(sample.guest_committed));
		if (sample.process.rss_shmem != 0) {
			line += fmt::format(" touched={}", Mib(sample.process.rss_shmem));
		}
	}
	for (const auto& [name, text]: sample.external) {
		line += fmt::format(" | {}: {}", name, text);
	}
	return line;
}

bool TakeDropRequest() {
	return g_drop_requested.exchange(false, std::memory_order_relaxed);
}

void NoteExitReason(const char* reason) {
	g_exit_reason.store(reason, std::memory_order_relaxed);
}

void PrintExit(std::string_view reason) {
	if (!Enabled() || g_exit_printed.exchange(true)) return;
	const auto  sample = TakeSample(false);
	std::string ext;
	for (const auto& [name, text]: sample.external) ext += fmt::format(" {}: {}", name, text);
	const auto elapsed =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - g_start).count();
	if (sample.process.available) {
		std::printf("[exit] reason=%.*s t=%.0fs rss=%s footprint=%s peak-rss=%s "
		            "guest-backing=%s%s\n",
		            static_cast<int>(reason.size()), reason.data(), elapsed,
		            Mib(sample.process.rss).c_str(),
		            Mib(Footprint(sample.process, sample.zram)).c_str(),
		            Mib(sample.process.peak_rss).c_str(), Mib(sample.guest_committed).c_str(),
		            ext.c_str());
	} else {
		std::printf("[exit] reason=%.*s t=%.0fs rss=not-available guest-backing=%s%s\n",
		            static_cast<int>(reason.size()), reason.data(), elapsed,
		            Mib(sample.guest_committed).c_str(), ext.c_str());
	}
	std::fflush(stdout);
}

void PrintNow(std::string_view tag) {
	std::printf("%s\n", FormatLine(TakeSample(true), tag).c_str());
	std::fflush(stdout);
}

void Start() {
	if (!Enabled()) return;
	std::scoped_lock lock(g_thread_mutex);
	if (g_thread.joinable()) return;
	g_stop = false;
#if defined(__linux__)
	struct sigaction action {};
	action.sa_handler = OnTerm;
	sigemptyset(&action.sa_mask);
	::sigaction(SIGTERM, &action, nullptr);
#endif
	g_thread = std::thread(Run);
}

void Stop() {
	if (!Enabled()) return;
	{
		std::scoped_lock lock(g_thread_mutex);
		g_stop = true;
	}
	g_thread_wake.notify_all();
	if (g_thread.joinable()) g_thread.join();
	PrintNow("exit");
	PrintExit("normal");
}

void EmergencyPrint() {
	if (!Enabled()) return;
	PrintNow("emergency");
	const char* reason = g_exit_reason.load(std::memory_order_relaxed);
	PrintExit(reason != nullptr ? reason : "emergency");
}

} // namespace Common::MemAccount
