#include "common/perfSummary.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

namespace {

using namespace Common::PerfSummary;

TEST(PerfSummaryFrame, LineReportsPresentedFramesAndTimeSplit) {
	const auto line = Detail::FormatFrameLine(5.0, 2, 24'000, 6'000, 4'000, 10, 7, 3);
	EXPECT_NE(line.find("[frame] secs=5.0 frames=2"), std::string::npos) << line;
	EXPECT_NE(line.find("wait_gpu_ms=24.000 wait_guest_ms=6.000 materialize_ms=4.000"),
	          std::string::npos)
	    << line;
	EXPECT_NE(line.find("ms/frame wait_gpu=12.000 wait_guest=3.000 materialize=2.000"),
	          std::string::npos)
	    << line;
	EXPECT_NE(line.find("module_lookups=10 module_hits=7 module_misses=3"), std::string::npos)
	    << line;
}

TEST(PerfSummaryFrame, ZeroFramesDoesNotDivideByZero) {
	const auto line = Detail::FormatFrameLine(5.0, 0, 1'000, 2'000, 3'000, 0, 0, 0);
	EXPECT_NE(line.find("frames=0"), std::string::npos) << line;
	EXPECT_NE(line.find("ms/frame wait_gpu=0.000 wait_guest=0.000 materialize=0.000"),
	          std::string::npos)
	    << line;
}

TEST(PerfSummaryFrame, ModuleMissWarningNeedsSustainedStreakAndMoreMissesThanHits) {
	EXPECT_TRUE(Detail::FormatFrameWarning(7, 3, 9.9).empty());
	EXPECT_TRUE(Detail::FormatFrameWarning(7, 3, 10.0).empty()); // More hits than misses.
	EXPECT_TRUE(Detail::FormatFrameWarning(3, 3, 20.0).empty()); // Equal.
	EXPECT_TRUE(Detail::FormatFrameWarning(0, 0, 20.0).empty()); // No lookups.
	const auto warning = Detail::FormatFrameWarning(3, 7, 10.0);
	EXPECT_NE(warning.find("[frame] WARNING module-misses"), std::string::npos) << warning;
	EXPECT_NE(warning.find("hits=3 misses=7"), std::string::npos) << warning;
}

TEST(PerfSummaryFrame, AddAccumulatesCountersOnlyWhileEnabled) {
	const auto index = static_cast<size_t>(Counter::FrameWaitGpuUs);
	const auto before = Detail::g_counters[index].load(std::memory_order_relaxed);
	Add(Counter::FrameWaitGpuUs, 25);
	EXPECT_EQ(Detail::g_counters[index].load(std::memory_order_relaxed), before);
	Detail::g_enabled.store(true, std::memory_order_relaxed);
	Add(Counter::FrameWaitGpuUs, 25);
	Add(Counter::FrameModuleMisses, 2);
	Detail::g_enabled.store(false, std::memory_order_relaxed);
	EXPECT_EQ(Detail::g_counters[index].load(std::memory_order_relaxed), before + 25);
	EXPECT_EQ(Detail::g_counters[static_cast<size_t>(Counter::FrameModuleMisses)].load(
	              std::memory_order_relaxed),
	          2u);
}

} // namespace

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
