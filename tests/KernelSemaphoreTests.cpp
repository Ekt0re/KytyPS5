#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "kernel/semaphore.h"
#include "libs/errno.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace {

using namespace Libs::LibKernel;
using namespace Libs::LibKernel::Semaphore;
using namespace std::chrono_literals;

constexpr uint32_t ATTR_FIFO = 0x01;

// Gives a waiter thread time to queue on the semaphore.
void LetQueue() {
	std::this_thread::sleep_for(100ms);
}

// A signal stops at the queue head when it cannot satisfy it: the later, smaller waiter keeps
// waiting, and the next signals wake the waiters in queue order (AnyPS5 #21061, measured on a PS5).
TEST(KernelSemaphore, SignalStopsAtAHeadThatNeedsMore) {
	KernelSema sem = nullptr;
	ASSERT_EQ(KernelCreateSema(&sem, "test", ATTR_FIFO, 0, 10, nullptr), OK);

	std::atomic<int> order {0};
	std::atomic<int> large_done {0};
	std::atomic<int> small_done {0};
	int              large_ret = -1;
	int              small_ret = -1;

	std::thread large([&] {
		KernelUseconds timeout = 5000000;
		large_ret              = KernelWaitSema(sem, 3, &timeout);
		large_done             = ++order;
	});
	LetQueue();
	std::thread small([&] {
		KernelUseconds timeout = 5000000;
		small_ret              = KernelWaitSema(sem, 1, &timeout);
		small_done             = ++order;
	});
	LetQueue();

	// One count does not satisfy the head (3), so the waiter of 1 behind it must not take it.
	EXPECT_EQ(KernelSignalSema(sem, 1), OK);
	LetQueue();
	EXPECT_EQ(large_done.load(), 0);
	EXPECT_EQ(small_done.load(), 0);

	// The count stays free: a poll still gets it.
	EXPECT_EQ(KernelPollSema(sem, 1), OK);
	EXPECT_EQ(KernelPollSema(sem, 1), KERNEL_ERROR_EBUSY);

	// Three counts wake the head only.
	EXPECT_EQ(KernelSignalSema(sem, 3), OK);
	LetQueue();
	EXPECT_EQ(large_done.load(), 1);
	EXPECT_EQ(small_done.load(), 0);

	// The next count wakes the small waiter.
	EXPECT_EQ(KernelSignalSema(sem, 1), OK);
	large.join();
	small.join();
	EXPECT_EQ(large_ret, OK);
	EXPECT_EQ(small_ret, OK);
	EXPECT_EQ(small_done.load(), 2);
	EXPECT_EQ(KernelPollSema(sem, 1), KERNEL_ERROR_EBUSY);
	EXPECT_EQ(KernelDeleteSema(sem), OK);
}

// One signal that covers several waiters in queue order wakes all of them.
TEST(KernelSemaphore, SignalWakesEveryHeadItCovers) {
	KernelSema sem = nullptr;
	ASSERT_EQ(KernelCreateSema(&sem, "test", ATTR_FIFO, 0, 10, nullptr), OK);

	int         first_ret  = -1;
	int         second_ret = -1;
	std::thread first([&] {
		KernelUseconds timeout = 5000000;
		first_ret              = KernelWaitSema(sem, 2, &timeout);
	});
	LetQueue();
	std::thread second([&] {
		KernelUseconds timeout = 5000000;
		second_ret             = KernelWaitSema(sem, 1, &timeout);
	});
	LetQueue();

	EXPECT_EQ(KernelSignalSema(sem, 4), OK);
	first.join();
	second.join();
	EXPECT_EQ(first_ret, OK);
	EXPECT_EQ(second_ret, OK);
	// 4 - 2 - 1 counts are left.
	EXPECT_EQ(KernelPollSema(sem, 1), OK);
	EXPECT_EQ(KernelPollSema(sem, 1), KERNEL_ERROR_EBUSY);
	EXPECT_EQ(KernelDeleteSema(sem), OK);
}

} // namespace

int main(int argc, char** argv) {
	Common::InitializeThreads();
	Common::Subsystems subsystems;
	subsystems.Initialize<Config::Lifecycle>();
	subsystems.Initialize<Log::Lifecycle>();
	subsystems.Initialize<Libs::LibKernel::Memory::Lifecycle>();
	Libs::LibKernel::Initialize();
	Libs::LibKernel::PthreadInitSelfForMainThread();
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
