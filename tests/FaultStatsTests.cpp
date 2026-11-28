#include "common/faultStats.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

namespace {

using namespace Common;
using PerfSummary::Detail::FaultBlock;

TEST(FaultStats, RateLineReportsIntervalRatesAndStormOutcomes) {
	const auto line = PerfSummary::Detail::FormatFaultLine(5.0, 100, 10, 5000, 2000, 1000, 500,
	                                                       3, 1, 2, 0);
	EXPECT_NE(line.find("[fault] secs=5.0 write=20/s read=2/s"), std::string::npos) << line;
	EXPECT_NE(line.find("handler_ms/s=1.0 invalidate_ms/s=0.4 protect_ms/s=0.2 "
	                    "lock_wait_ms/s=0.1 fast_skip=1/s"),
	          std::string::npos)
	    << line;
	EXPECT_NE(line.find("storm(marked=1 kept=2 expired=0)"), std::string::npos) << line;
}

TEST(FaultStats, ZeroLengthIntervalDoesNotDivideByZero) {
	const auto line = PerfSummary::Detail::FormatFaultLine(0.0, 100, 10, 5000, 2000, 1000, 500,
	                                                       3, 1, 2, 0);
	EXPECT_NE(line.find("write=0/s read=0/s"), std::string::npos) << line;
	EXPECT_EQ(line.find("nan"), std::string::npos) << line;
}

TEST(FaultStats, HotBlocksRankByCountAndClearOnTake) {
	FaultStats::FaultHotBlocks blocks;
	for (int i = 0; i < 5; i++) {
		blocks.Note(0x10000);
	}
	for (int i = 0; i < 3; i++) {
		blocks.Note(0x20000);
	}
	std::array<FaultBlock, FaultStats::FaultHotBlocks::kTop> top {};
	uint64_t                                                 dropped = 0;
	EXPECT_EQ(blocks.TakeTop(top, dropped), 2u);
	EXPECT_EQ(top[0].address, 0x10000u);
	EXPECT_EQ(top[0].count, 5u);
	EXPECT_EQ(top[1].address, 0x20000u);
	EXPECT_EQ(top[1].count, 3u);
	EXPECT_EQ(dropped, 0u);
	const auto hot = PerfSummary::Detail::FormatFaultHotLine(top.data(), 2, dropped);
	EXPECT_NE(hot.find("0x10000=5"), std::string::npos) << hot;
	EXPECT_NE(hot.find("0x20000=3"), std::string::npos) << hot;

	// Taking the top drains the table: a second take with no new faults is empty.
	EXPECT_EQ(blocks.TakeTop(top, dropped), 0u);
}

TEST(FaultStats, HotBlocksBoundTheTableAndReportDroppedFaults) {
	FaultStats::FaultHotBlocks blocks;
	for (uint64_t i = 1; i <= 2 * FaultStats::FaultHotBlocks::kSlots; i++) {
		blocks.Note(i << 16);
	}
	std::array<FaultBlock, FaultStats::FaultHotBlocks::kTop> top {};
	uint64_t                                                 dropped = 0;
	const auto                                               used = blocks.TakeTop(top, dropped);
	EXPECT_LE(used, FaultStats::FaultHotBlocks::kTop);
	EXPECT_GT(dropped, 0u);
	const auto hot = PerfSummary::Detail::FormatFaultHotLine(top.data(), used, dropped);
	EXPECT_NE(hot.find("untracked"), std::string::npos) << hot;
}

TEST(FaultStats, StormValuesHaveFallbacksAndCaps) {
	EXPECT_EQ(FaultStats::ParseStormValue(nullptr, 8, 30000), 8u);
	EXPECT_EQ(FaultStats::ParseStormValue("", 8, 30000), 8u);
	EXPECT_EQ(FaultStats::ParseStormValue("3", 0, 250), 3u);
	EXPECT_EQ(FaultStats::ParseStormValue("999", 0, 250), 250u);
	EXPECT_EQ(FaultStats::ParseStormValue("0", 8, 30000), 0u);
	EXPECT_FALSE(FaultStats::StormEnabled()); // Nothing set in the test environment.
}

} // namespace

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
