#include "FileTimeTestUtil.h"
#include "common/assert.h"
#include "common/config.h"
#include "common/emulatorConfig.h"
#include "common/faultReport.h"
#include "common/file.h"
#include "common/guestPark.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/memAccount.h"
#include "common/platform/sysDbg.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "common/virtualMemory.h"
#include "libs/libs.h"

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>

#ifdef __linux__
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace {

namespace VirtualMemory = Common::VirtualMemory;

constexpr uint64_t PAGE = 0x10000;
constexpr uint64_t BASE = 0x0000'0600'0000'0000ull;

TEST(VirtualMemory, AllocFixedRefusesMappedPagesAndFreeReleasesOnlyItsRange) {
	ASSERT_TRUE(VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::ReadWrite));
	ASSERT_TRUE(VirtualMemory::AllocFixed(BASE + PAGE, PAGE, VirtualMemory::Mode::ReadWrite));

	EXPECT_FALSE(VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::ReadWrite));
	EXPECT_FALSE(VirtualMemory::AllocFixed(BASE - PAGE, 2 * PAGE, VirtualMemory::Mode::ReadWrite));

	ASSERT_TRUE(VirtualMemory::Free(BASE, PAGE));
	EXPECT_FALSE(VirtualMemory::AllocFixed(BASE + PAGE, PAGE, VirtualMemory::Mode::ReadWrite));
	ASSERT_TRUE(VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::ReadWrite));

	EXPECT_TRUE(VirtualMemory::Free(BASE, PAGE));
	EXPECT_TRUE(VirtualMemory::Free(BASE + PAGE, PAGE));
}

volatile uint64_t g_fault_rsp = 0;

// Resolves the write fault on BASE by making the page writable.
bool MakeBaseWritable(const Common::HostException::ExceptionInfo& info) {
	g_fault_rsp = info.rsp;
	return info.type == Common::HostException::ExceptionType::AccessViolation &&
	       info.access_violation_type == Common::HostException::AccessViolationType::Write &&
	       info.access_violation_vaddr == BASE &&
	       VirtualMemory::Protect(BASE, PAGE, VirtualMemory::Mode::ReadWrite);
}

bool RefuseFault(const Common::HostException::ExceptionInfo& /*info*/) {
	return false;
}

TEST(HostException, TheFirstHandlerStaysInstalledAndResolvesFaults) {
	EXPECT_EXIT(
	    {
		    namespace HostException = Common::HostException;
		    const bool installs     = HostException::InstallHandler(MakeBaseWritable) &&
			                          HostException::InstallHandler(MakeBaseWritable) &&
			                          !HostException::InstallHandler(RefuseFault);
		    const bool mapped = VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::Read);
		    auto*      value  = reinterpret_cast<volatile uint32_t*>(BASE);
		    *value            = 7;
		    // The faulting frame's stack pointer lies just below this local.
		    const auto local  = reinterpret_cast<uint64_t>(&value);
		    const bool rsp_ok = g_fault_rsp <= local && local - g_fault_rsp < 0x10000;
		    std::_Exit(installs && mapped && *value == 7 && rsp_ok ? 0 : 1);
	    },
	    testing::ExitedWithCode(0), "");
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
volatile sig_atomic_t g_dispatched = 0;

// Refuses the fault if the guest signal it raises was delivered before the handler returned.
bool RaiseDispatchDuringFault(const Common::HostException::ExceptionInfo& info) {
	std::raise(Common::HostException::GuestSignalDispatchSignal());
	return g_dispatched == 0 && info.access_violation_vaddr == BASE &&
	       VirtualMemory::Protect(BASE, PAGE, VirtualMemory::Mode::ReadWrite);
}

TEST(HostException, GuestSignalWaitsUntilTheFaultHandlerReturns) {
	EXPECT_EXIT(
	    {
		    std::signal(Common::HostException::GuestSignalDispatchSignal(),
			            [](int) { g_dispatched = 1; });
		    const bool installs = Common::HostException::InstallHandler(RaiseDispatchDuringFault);
		    const bool mapped   = VirtualMemory::AllocFixed(BASE, PAGE, VirtualMemory::Mode::Read);
		    *reinterpret_cast<volatile uint32_t*>(BASE) = 1;
		    std::_Exit(installs && mapped && g_dispatched == 1 ? 0 : 1);
	    },
	    testing::ExitedWithCode(0), "");
}
#endif

size_t CountOf(const char* text, const char* needle) {
	size_t count = 0;
	for (const char* p = text; (p = std::strstr(p, needle)) != nullptr; p++) {
		count++;
	}
	return count;
}

#if defined(__linux__)
int g_fault_pipe = -1;

// Writes the host fault report to the pipe the parent reads, then ends the child.
bool ReportAndExit(const Common::HostException::ExceptionInfo& info) {
	Common::FaultReport::Registers regs {};
	regs.pc  = info.exception_address;
	regs.rsp = info.rsp;
	regs.rbp = info.rbp;
	Common::FaultReport::WriteHostContext(
	    regs,
	    {[](void*, const char* text) { (void)!::write(g_fault_pipe, text, std::strlen(text)); },
	     nullptr});
	std::_Exit(0);
}

extern "C" __attribute__((noinline, used)) void KytyFaultReportTestNullWrite(volatile int* p) {
	*p = 1;
	asm volatile("" ::: "memory");
}

TEST(FaultReport, WorkerThreadFaultNamesModuleFunctionAndFrames) {
	int fds[2];
	ASSERT_EQ(::pipe(fds), 0);
	const pid_t child = ::fork();
	ASSERT_GE(child, 0);
	if (child == 0) {
		::close(fds[0]);
		g_fault_pipe = fds[1];
		Common::FaultReport::RefreshModules();
		Common::HostException::InstallHandler(ReportAndExit);
		std::thread worker([] {
			Common::FaultReport::SetThreadTag("fault test tag", "fault-test-worker");
			Common::FaultReport::RegisterThreadStack();
			KytyFaultReportTestNullWrite(nullptr);
		});
		worker.join();
		std::_Exit(2);
	}
	::close(fds[1]);
	std::string text;
	char        buffer[4096];
	for (ssize_t n; (n = ::read(fds[0], buffer, sizeof(buffer))) > 0;) {
		text.append(buffer, static_cast<size_t>(n));
	}
	::close(fds[0]);
	int status = 0;
	::waitpid(child, &status, 0);
	const size_t frames = CountOf(text.c_str(), " bt[") + CountOf(text.c_str(), " fp[");
	EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << "status " << status;
	EXPECT_NE(text.find("fault-test-worker"), std::string::npos) << text;
	EXPECT_NE(text.find("doing: fault test tag"), std::string::npos) << text;
	EXPECT_NE(text.find("common_tests+0x"), std::string::npos) << text;
	EXPECT_NE(text.find("[KytyFaultReportTestNullWrite+0x"), std::string::npos) << text;
	EXPECT_GE(frames, 3u) << text;
}
#elif defined(_WIN32)
__declspec(noinline) void KytyFaultReportTestCapture(std::string* out) {
	CONTEXT ctx {};
	RtlCaptureContext(&ctx);
	Common::FaultReport::Registers regs {};
	regs.pc             = ctx.Rip;
	regs.rsp            = ctx.Rsp;
	regs.rbp            = ctx.Rbp;
	regs.native_context = &ctx;
	Common::FaultReport::WriteHostContext(
	    regs, {[](void* user, const char* text) { static_cast<std::string*>(user)->append(text); },
	           out});
}

TEST(FaultReport, WorkerThreadFaultNamesModuleFunctionAndFrames) {
	Common::FaultReport::RefreshModules();
	std::string text;
	std::thread worker([&] {
		Common::FaultReport::SetThreadTag("fault test tag", "fault-test-worker");
		KytyFaultReportTestCapture(&text);
	});
	worker.join();
	EXPECT_NE(text.find("fault-test-worker"), std::string::npos) << text;
	EXPECT_NE(text.find("doing: fault test tag"), std::string::npos) << text;
	EXPECT_NE(text.find("common_tests"), std::string::npos) << text;
	EXPECT_NE(text.find("+0x"), std::string::npos) << text;
	EXPECT_GE(CountOf(text.c_str(), " bt["), 2u) << text;
}
#elif defined(__APPLE__)
__attribute__((noinline)) void KytyFaultReportTestCapture(std::string* out) {
	Common::FaultReport::Registers regs {};
	regs.pc = reinterpret_cast<uint64_t>(&KytyFaultReportTestCapture);
	Common::FaultReport::WriteHostContext(
	    regs, {[](void* user, const char* text) { static_cast<std::string*>(user)->append(text); },
	           out});
}

TEST(FaultReport, WorkerThreadFaultNamesModuleFunctionAndFrames) {
	std::string text;
	std::thread worker([&] {
		Common::FaultReport::SetThreadTag("fault test tag", "fault-test-worker");
		KytyFaultReportTestCapture(&text);
	});
	worker.join();
	EXPECT_NE(text.find("fault-test-worker"), std::string::npos) << text;
	EXPECT_NE(text.find("doing: fault test tag"), std::string::npos) << text;
	EXPECT_NE(text.find("common_tests+0x"), std::string::npos) << text;
	EXPECT_GE(CountOf(text.c_str(), " bt["), 2u) << text;
}
#else
TEST(FaultReport, WorkerThreadFaultNamesModuleFunctionAndFrames) {
	GTEST_SKIP() << "Host fault symbolization is not available on this platform";
}
#endif

TEST(SysDbg, StackUsageReportsTheCallingThreadStackBase) {
	std::thread([] {
		const auto s = SysStackUsage();
		EXPECT_NE(s.reserved_addr, 0u);
		EXPECT_LT(s.reserved_addr, reinterpret_cast<uintptr_t>(&s));
	}).join();
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
constexpr int FATAL_EXIT_STATUS = 321;
#else
// POSIX keeps the low 8 bits of the exit status.
constexpr int FATAL_EXIT_STATUS = 321 & 0xff;
#endif

TEST(Assert, ExitMacrosEvaluateOnceAndExitWithTheFatalStatus) {
	int evaluations = 0;
	EXIT_IF(++evaluations == 0);
	EXIT_NOT_IMPLEMENTED(++evaluations == 0);
	EXPECT_EQ(evaluations, 2);

	EXPECT_EXIT(EXIT_IF(evaluations == 2), testing::ExitedWithCode(FATAL_EXIT_STATUS), "");
	EXPECT_EXIT(EXIT_NOT_IMPLEMENTED(true), testing::ExitedWithCode(FATAL_EXIT_STATUS), "");
	EXPECT_EXIT(EXIT("%d\n", evaluations), testing::ExitedWithCode(FATAL_EXIT_STATUS), "");
}

TEST(File, MissingFileStaysInvalidAndCreatedFileRoundTrips) {
	const auto path = std::filesystem::temp_directory_path() / "kyty_common_file_test.bin";
	std::filesystem::remove(path);

	Common::File missing;
	EXPECT_FALSE(missing.Open(path, Common::File::Mode::Read));
	EXPECT_TRUE(missing.IsInvalid());

	const uint32_t value = 0x12345678u;
	{
		Common::File out;
		ASSERT_TRUE(out.Create(path));
		out.Write(&value, sizeof(value));
		EXPECT_EQ(out.Size(), sizeof(value));
	}

	Common::File in;
	ASSERT_TRUE(in.Open(path, Common::File::Mode::Read));
	uint32_t read_value = 0;
	in.Read(&read_value, sizeof(read_value));
	EXPECT_EQ(read_value, value);
	EXPECT_TRUE(in.Seek(2));
	EXPECT_EQ(in.Tell(), 2u);
	in.Close();
	std::filesystem::remove(path);
}

TEST(File, NativeExistenceQueriesFollowTheEntryType) {
	const auto dir  = std::filesystem::temp_directory_path();
	const auto file = dir / "kyty_common_exists_test.bin";
	{
		Common::File out;
		ASSERT_TRUE(out.Create(file));
		const uint8_t byte = 1;
		out.Write(&byte, sizeof(byte));
	}

	EXPECT_TRUE(Common::File::IsDirectoryExisting(dir));
	EXPECT_TRUE(Common::File::IsDirectoryExisting(dir / ""));
	EXPECT_FALSE(Common::File::IsFileExisting(dir));
	EXPECT_EQ(Common::File::Size(dir), 0u);

	EXPECT_TRUE(Common::File::IsFileExisting(file));
	EXPECT_FALSE(Common::File::IsFileExisting(file / ""));
	EXPECT_FALSE(Common::File::IsDirectoryExisting(file));
	EXPECT_EQ(Common::File::Size(file), 1u);

	std::filesystem::remove(file);
	EXPECT_FALSE(Common::File::IsFileExisting(file));
}

TEST(File, GetTimesReportsExactNanoseconds) {
	const auto path = std::filesystem::temp_directory_path() / "kyty_common_time_test.bin";
	ASSERT_TRUE(Common::File().Create(path));
	// A multiple of 100 ns, the Windows file time unit.
	constexpr int64_t WRITE_NS = 1700000000123456700;
	SetUnixWriteTime(path, WRITE_NS);

	const auto times = Common::File::GetTimes(path);
	ASSERT_TRUE(times.has_value());
	EXPECT_EQ(times->write_ns, WRITE_NS);
	EXPECT_GT(times->access_ns, WRITE_NS);

	Common::File in(path, Common::File::Mode::Read);
	const auto   open_times = in.GetTimes();
	ASSERT_TRUE(open_times.has_value());
	EXPECT_EQ(open_times->write_ns, WRITE_NS);
	in.Close();
	std::filesystem::remove(path);
}

std::string ReadAll(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary);
	return {std::istreambuf_iterator<char>(file), {}};
}

TEST(Log, FileGetsPlainTextAndTheConsoleGetsEachLineOnce) {
	EXPECT_EXIT(
	    {
		    const auto dir =
		        std::filesystem::temp_directory_path() /
		        ("kyty_log_test_" +
				 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
		    std::filesystem::create_directories(dir);
		    const auto console_file = dir / "console.txt";
		    std::freopen(console_file.string().c_str(), "wb", stdout);
		    Log::Write({}, "p");
		    const auto log_file = dir / "log.txt";
		    std::ofstream(log_file) << "previous run";
		    Config::Initialize();
		    Config::ConfigOptions options;
		    options.printf_direction   = Config::LogDirection::File;
		    options.printf_output_file = log_file;
		    Config::Load(options);
		    Log::Initialize();
		    Log::Write(Log::Color::Red, "a");
		    Log::WriteToConsoleAndLog("b");
		    Log::Flush();
		    const bool file_ok = ReadAll(log_file) == "ab";

		    options.printf_direction = Config::LogDirection::Console;
		    Config::Load(options);
		    Log::Initialize();
		    Log::Write(Log::Color::Red, "c");
		    Log::WriteToConsoleAndLog("d");
		    const bool console_ok = ReadAll(console_file) == "pb\x1b[31mc\x1b[0md";

		    options.printf_direction = Config::LogDirection::Silent;
		    Config::Load(options);
		    Log::Initialize();
		    const bool      silent_ok = Log::IsSilent();
		    std::error_code ec;
		    std::filesystem::remove_all(dir, ec);
		    std::_Exit(file_ok && console_ok && silent_ok ? 0 : 1);
	    },
	    testing::ExitedWithCode(0), "");
}

TEST(StringUtils, PathRoundTripsUtf8) {
	constexpr std::string_view Utf8 = "dir/\xc3\xa9-\xe6\x97\xa5.bin";
	EXPECT_EQ(Common::PathToString(Common::PathFromUtf8(Utf8)), Utf8);
	EXPECT_EQ(Common::PathToGenericString(Common::PathFromUtf8(Utf8)), Utf8);
}

TEST(Thread, MutexIsRecursiveAndCondVarTimesOutOrWakes) {
	Common::Mutex   mutex;
	Common::CondVar cond_var;
	mutex.Lock();
	mutex.Lock();
	mutex.Unlock();
	EXPECT_FALSE(cond_var.WaitFor(&mutex, 1000));

	bool        ready = false;
	std::thread signaller([&] {
		Common::LockGuard lock(mutex);
		ready = true;
		cond_var.Signal();
	});
	while (!ready) {
		cond_var.Wait(&mutex);
	}
	mutex.Unlock();
	signaller.join();
}

TEST(Thread, SleepsAtLeastTheRequestedTimeAndIdsAreUniquePerThread) {
	Common::InitializeThreads();
	EXPECT_TRUE(Common::Thread::IsMainThread());

	const auto start = std::chrono::steady_clock::now();
	Common::Thread::SleepMicro(2000);
	EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::microseconds(2000));

	const int id = Common::Thread::GetThreadIdUnique();
	EXPECT_EQ(Common::Thread::GetThreadIdUnique(), id);
	int  other_id   = id;
	bool other_main = true;
	std::thread([&] {
		other_id   = Common::Thread::GetThreadIdUnique();
		other_main = Common::Thread::IsMainThread();
	}).join();
	EXPECT_NE(other_id, id);
	EXPECT_FALSE(other_main);
}

#ifdef __linux__
unsigned ReadCpuValue(int cpu, const char* file) {
	unsigned value = 0;
	std::ifstream("/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/" + file) >> value;
	return value;
}

TEST(Thread, ReservesTheFastestCoreAndRestoresTheProcessAffinity) {
	Common::InitializeThreads();
	int       processor = -1;
	cpu_set_t reserved {};
	std::thread([&] {
		processor = Common::Thread::ReserveCoreForCurrentThread();
		ASSERT_EQ(sched_getaffinity(0, sizeof(reserved), &reserved), 0);
	}).join();
	cpu_set_t others {};
	ASSERT_EQ(sched_getaffinity(0, sizeof(others), &others), 0);
	const auto rank = [](int cpu) {
		return std::tuple(ReadCpuValue(cpu, "acpi_cppc/highest_perf"),
		                  ReadCpuValue(cpu, "cpufreq/cpuinfo_max_freq"),
		                  ReadCpuValue(cpu, "topology/thread_siblings_list"));
	};
	if (processor < 0) {
		// No core is reserved only where fewer than eight cores report a performance or frequency.
		cpu_set_t cores {};
		for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
			if (CPU_ISSET(cpu, &others)) {
				const auto [performance, frequency, first] = rank(cpu);
				if (performance != 0 || frequency != 0) {
					CPU_SET(first, &cores);
				}
			}
		}
		EXPECT_LT(CPU_COUNT(&cores), 8);
		return;
	}
	EXPECT_TRUE(CPU_ISSET(processor, &reserved));
	for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
		if (CPU_ISSET(cpu, &reserved)) {
			EXPECT_EQ(std::get<2>(rank(cpu)), std::get<2>(rank(processor))) << cpu;
		}
		if (CPU_ISSET(cpu, &others)) {
			EXPECT_LT(rank(cpu), rank(processor)) << cpu;
		}
	}

	cpu_set_t restored {};
	std::thread([&] {
		Common::Thread::RestoreProcessAffinity();
		ASSERT_EQ(sched_getaffinity(0, sizeof(restored), &restored), 0);
	}).join();
	CPU_OR(&reserved, &reserved, &others);
	EXPECT_TRUE(CPU_EQUAL(&restored, &reserved));
}
#endif

TEST(MemAccount, ParseProcStatusAndZramFootprint) {
	const auto memory = Common::MemAccount::ParseProcStatus(
	    "VmRSS:\t 1000 kB\nVmSwap:\t 1000 kB\nVmHWM:\t 1200 kB\n"
	    "RssAnon:\t 600 kB\nRssFile:\t 300 kB\nRssShmem:\t 100 kB\n");
	EXPECT_TRUE(memory.available);
	EXPECT_EQ(memory.rss, 1000u * 1024u);
	EXPECT_EQ(memory.swap, 1000u * 1024u);
	EXPECT_EQ(memory.peak_rss, 1200u * 1024u);
	EXPECT_EQ(memory.rss_anon, 600u * 1024u);
	EXPECT_EQ(memory.rss_file, 300u * 1024u);
	EXPECT_EQ(memory.rss_shmem, 100u * 1024u);
	EXPECT_FALSE(Common::MemAccount::ParseProcStatus("no matching keys").available);

	const auto zram = Common::MemAccount::ParseZramMmStat("8000 2000 2400 0 3000 5 0 0 0\n");
	EXPECT_TRUE(zram.known);
	EXPECT_EQ(zram.orig, 8000u);
	EXPECT_EQ(zram.compressed, 2400u);
	Common::MemAccount::ProcessMemory p;
	p.available = true;
	p.rss       = 1000;
	p.swap      = 1000;
	EXPECT_EQ(Common::MemAccount::Footprint(p, zram), 1300u);
	EXPECT_EQ(Common::MemAccount::Footprint(p, {}), 2000u);
	EXPECT_FALSE(Common::MemAccount::ParseZramMmStat("garbage").known);
}

TEST(MemAccount, FormatLineAndGrowthCoverInjectedAndUnavailableFigures) {
	Common::MemAccount::Sample sample;
	sample.process.available        = true;
	sample.process.rss              = 3ull << 30;
	sample.process.private_bytes    = 2ull << 30;
	sample.process.swap             = 1ull << 30;
	sample.process.peak_rss         = 4ull << 30;
	sample.process.rss_shmem        = 1ull << 30;
	sample.process.sys_commit_used  = 10ull << 30;
	sample.process.sys_commit_total = 32ull << 30;
	sample.heap                     = {.known = true, .in_use = 512ull << 20, .free_held = 64ull << 20};
	sample.owners                   = {{"shader-store", 256ull << 20, 42}};
	sample.zram                     = {.known = true, .orig = 4ull << 30, .compressed = 1ull << 30};
	sample.guest_committed          = 16ull << 30;
	sample.external                 = {{"vk", "device=800MiB host=128MiB"}};

	const auto line = Common::MemAccount::FormatLine(sample, "now");
	EXPECT_NE(line.find("[mem] now "), std::string::npos) << line;
	EXPECT_NE(line.find("rss=3072MiB private=2048MiB swap=1024MiB"), std::string::npos) << line;
	EXPECT_NE(line.find("counted=256MiB unaccounted=2816MiB"), std::string::npos) << line;
	EXPECT_NE(line.find("shader-store=256MiB(42)"), std::string::npos) << line;
	EXPECT_NE(line.find("heap-in-use=512MiB heap-free-held=64MiB"), std::string::npos) << line;
	EXPECT_NE(line.find("footprint=3328MiB zram=4096MiB->1024MiB"), std::string::npos) << line;
	EXPECT_NE(line.find("sys-commit=10240MiB/32768MiB"), std::string::npos) << line;
	EXPECT_NE(line.find("guest: committed=16384MiB touched=1024MiB"), std::string::npos) << line;
	EXPECT_NE(line.find("vk: device=800MiB host=128MiB"), std::string::npos) << line;

	Common::MemAccount::Sample unavailable;
	const auto                 na_line = Common::MemAccount::FormatLine(unavailable, "na");
	EXPECT_NE(na_line.find("rss=not-available"), std::string::npos) << na_line;

	const std::vector<std::pair<std::string, uint64_t>> before {
	    {"heap-in-use", 100ull << 20}, {"shader-store", 10ull << 20}};
	const std::vector<std::pair<std::string, uint64_t>> after {
	    {"heap-in-use", 400ull << 20}, {"shader-store", 30ull << 20}};
	EXPECT_EQ(Common::MemAccount::FormatGrowth(before, after, 2),
	          "heap-in-use +300MiB, shader-store +20MiB");
	EXPECT_EQ(Common::MemAccount::FormatGrowth(after, before), "nothing");
}

TEST(MemAccount, LimitGuardDropsThenStopsOnlyWhileOver) {
	using Action = Common::MemAccount::LimitGuard::Action;
	Common::MemAccount::LimitGuard off(0);
	EXPECT_EQ(off.Step(0, 1000), Action::None);
	EXPECT_EQ(off.Step(20000, 1000), Action::None);

	Common::MemAccount::LimitGuard guard(100, 10000);
	EXPECT_EQ(guard.Step(1000, 100), Action::None);
	EXPECT_EQ(guard.Step(2000, 101), Action::DropCaches);
	EXPECT_EQ(guard.Step(11999, 101), Action::None);
	// Dropping back under resets the grace timer.
	EXPECT_EQ(guard.Step(12000, 90), Action::None);
	EXPECT_EQ(guard.Step(13000, 150), Action::DropCaches);
	EXPECT_EQ(guard.Step(22999, 150), Action::None);
	EXPECT_EQ(guard.Step(23000, 150), Action::Stop);
}

TEST(GuestPark, RegistryCountsThreadsAndFlagsFirstAtEachPc) {
	Common::GuestPark::Registry registry;
	EXPECT_EQ(registry.Count(), 0u);

	const auto first = registry.Add(0x1000, 0x0, 11, "worker-a");
	EXPECT_EQ(first.count, 1u);
	EXPECT_TRUE(first.first_at_pc);
	EXPECT_EQ(first.at_pc, 1u);

	const auto repeat = registry.Add(0x1000, 0x8, 12, "worker-b");
	EXPECT_EQ(repeat.count, 2u);
	EXPECT_FALSE(repeat.first_at_pc);
	EXPECT_EQ(repeat.at_pc, 2u);

	const auto other = registry.Add(0x2000, 0x10, 13, nullptr);
	EXPECT_EQ(other.count, 3u);
	EXPECT_TRUE(other.first_at_pc);
	EXPECT_EQ(other.at_pc, 1u);
	EXPECT_EQ(registry.Count(), 3u);

	for (size_t i = 0; i < Common::GuestPark::Registry::kCapacity; i++) {
		(void)registry.Add(0x3000 + i * 4, 0, static_cast<int32_t>(100 + i), "fill");
	}
	EXPECT_EQ(registry.Count(), 3u + Common::GuestPark::Registry::kCapacity);
}

TEST(Config, ParseTraceCallsFilterHandlesOffAllAndNamedLibraries) {
	const auto off = Config::ParseTraceCallsFilter("");
	EXPECT_FALSE(off.enabled);
	EXPECT_TRUE(Config::ParseTraceCallsFilter("0").libraries.empty());
	EXPECT_FALSE(Config::ParseTraceCallsFilter("0").enabled);
	EXPECT_FALSE(Config::ParseTraceCallsFilter(",,,").enabled);

	const auto all = Config::ParseTraceCallsFilter("all");
	EXPECT_TRUE(all.enabled);
	EXPECT_TRUE(all.all);
	EXPECT_TRUE(all.libraries.empty());

	const auto named = Config::ParseTraceCallsFilter("libA,,libB");
	EXPECT_TRUE(named.enabled);
	EXPECT_FALSE(named.all);
	ASSERT_EQ(named.libraries.size(), 2u);
	EXPECT_EQ(named.libraries[0], "libA");
	EXPECT_EQ(named.libraries[1], "libB");
}

TEST(Libs, PowerOfTwoRateLimitMatchesFirstFourAndPowersOfTwo) {
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(0));
	for (uint64_t n = 1; n <= 4; n++) {
		EXPECT_TRUE(Libs::ShouldLogPowerOfTwo(n));
	}
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(5));
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(7));
	EXPECT_TRUE(Libs::ShouldLogPowerOfTwo(8));
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(9));
	EXPECT_TRUE(Libs::ShouldLogPowerOfTwo(16));
	EXPECT_TRUE(Libs::ShouldLogPowerOfTwo(32));
	EXPECT_FALSE(Libs::ShouldLogPowerOfTwo(33));
}

TEST(Libs, CallRingWrapsAroundAndIteratesOldestFirst) {
	Libs::CallRing ring;
	for (uint64_t i = 0; i < Libs::kCallRingSize + 10; i++) {
		const uint64_t args[6] = {i, i + 1, 0, 0, 0, 0};
		auto&          rec     = ring.Push("library", "Func", args, 0x1000 + i);
		rec.result             = i * 2;
		rec.completed          = true;
	}
	EXPECT_EQ(ring.count, Libs::kCallRingSize);
	size_t seen = 0;
	ring.ForEachOldestFirst([&](const Libs::CallRecord& rec, size_t idx) {
		EXPECT_EQ(idx, seen);
		EXPECT_EQ(rec.args[0], 10u + seen);
		EXPECT_EQ(rec.result, (10u + seen) * 2u);
		EXPECT_EQ(rec.guest_ret_addr, 0x1000u + 10u + seen);
		seen++;
	});
	EXPECT_EQ(seen, Libs::kCallRingSize);
}

} // namespace

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
