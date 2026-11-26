#include "common/compilePause.h"
#include "common/timer.h"
#include "kernel/pthread.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>

// The guest clock stands still while a GPU queue waits for a compile (after AnyPS5 62a241e4).
namespace {

using Libs::LibKernel::KernelClockGettime;
using Libs::LibKernel::KernelTimespec;
using Libs::LibKernel::KernelUsleep;

constexpr int GuestClockMonotonic = 4;

// sceKernelReadTsc without --guest-tsc (KernelReadTsc itself reads the emulator config).
uint64_t ReadTsc() {
	return Common::CompilePause::GuestTicks();
}

uint64_t MonotonicNs() {
	KernelTimespec time {};
	EXPECT_EQ(KernelClockGettime(GuestClockMonotonic, &time), 0);
	return static_cast<uint64_t>(time.tv_sec) * 1000000000ull + static_cast<uint64_t>(time.tv_nsec);
}

TEST(GuestClockHold, ClocksStandStillWhileHeld) {
	if (!Common::CompilePause::Enabled()) GTEST_SKIP() << "KYTY_PAUSE_ON_COMPILE=0";
	uint64_t mono_before    = 0;
	uint64_t tsc_before     = 0;
	uint64_t mono_held      = 0;
	uint64_t tsc_held       = 0;
	{
		const Common::CompilePause::Scope hold;
		{
			// Nested holds share one pause.
			const Common::CompilePause::Scope nested;
			mono_before    = MonotonicNs();
			tsc_before     = ReadTsc();
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(60));
		mono_held    = MonotonicNs();
		tsc_held     = ReadTsc();
	}
	EXPECT_EQ(mono_held, mono_before);
	EXPECT_EQ(tsc_held, tsc_before);

	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	const auto mono_after = MonotonicNs();
	EXPECT_GE(mono_after, mono_held);
	// The clock runs again, without the held 60 ms.
	EXPECT_GE(mono_after - mono_held, 15000000u);
	EXPECT_LT(mono_after - mono_held, 55000000u);
	EXPECT_GT(ReadTsc(), tsc_held);
}

TEST(GuestClockHold, MonotonicAcrossConcurrentHolds) {
	if (!Common::CompilePause::Enabled()) GTEST_SKIP() << "KYTY_PAUSE_ON_COMPILE=0";
	std::atomic_bool stop {false};
	std::thread      holder([&] {
        while (!stop.load()) {
            const Common::CompilePause::Scope hold;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    });
	uint64_t last_mono = 0;
	uint64_t last_tsc  = 0;
	const auto end     = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
	while (std::chrono::steady_clock::now() < end) {
		const auto mono = MonotonicNs();
		const auto tsc  = ReadTsc();
		ASSERT_GE(mono, last_mono);
		ASSERT_GE(tsc, last_tsc);
		last_mono = mono;
		last_tsc  = tsc;
	}
	stop = true;
	holder.join();
}

TEST(GuestClockHold, GuestSleepDoesNotCountHeldTime) {
	if (!Common::CompilePause::Enabled()) GTEST_SKIP() << "KYTY_PAUSE_ON_COMPILE=0";
	// A title watchdog: ReadTsc, usleep, ReadTsc. A hold in the middle of the sleep makes the sleep
	// last longer on the host, while the guest still sees about the time it asked for.
	std::atomic_bool started {false};
	std::thread      holder([&] {
        while (!started.load()) std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        const Common::CompilePause::Scope hold;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    });
	const auto host_start = Common::SteadyNs();
	const auto mono_start = MonotonicNs();
	started               = true;
	KernelUsleep(100000);
	const auto host_slept = Common::SteadyNs() - host_start;
	const auto mono_slept = MonotonicNs() - mono_start;
	holder.join();
	EXPECT_GE(host_slept, 240000000u);
	EXPECT_GE(mono_slept, 100000000u);
	EXPECT_LT(mono_slept, 200000000u);
}

} // namespace
