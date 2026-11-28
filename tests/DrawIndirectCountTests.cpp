#include "graphics/host_gpu/renderer/render.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>

namespace Libs::Graphics {
namespace {

TEST(DrawIndirectCount, ClampsCpuAndDeferredCounts) {
	EXPECT_EQ(ClampIndirectDrawCount(0, 4), 0u);
	EXPECT_EQ(ClampIndirectDrawCount(3, 4), 3u);
	EXPECT_EQ(ClampIndirectDrawCount(4, 4), 4u);
	EXPECT_EQ(ClampIndirectDrawCount(UINT32_MAX, 4), 4u);
	EXPECT_EQ(ClampIndirectDrawCount(4, 0), 0u);
}

TEST(DrawIndirectCount, ComputesRecordRangesWithoutNarrowing) {
	EXPECT_EQ(IndirectDrawRecordsSize(0, 32, 20), 0u);
	EXPECT_EQ(IndirectDrawRecordsSize(1, 32, 20), 20u);
	EXPECT_EQ(IndirectDrawRecordsSize(2, 32, 20), 52u);
	constexpr uint32_t count = std::numeric_limits<uint32_t>::max();
	EXPECT_EQ(IndirectDrawRecordsSize(count, count, 20),
	          static_cast<uint64_t>(count - 1) * count + 20u);
}

TEST(DrawIndirectCount, GatesGpuCountsOnAlignmentAndDeviceLimit) {
	EXPECT_TRUE(CanUseGpuIndirectDrawCount(0x1000, 4, 4));
	EXPECT_TRUE(CanUseGpuIndirectDrawCount(0x1000, 3, 4));
	EXPECT_FALSE(CanUseGpuIndirectDrawCount(0, 4, 4));
	EXPECT_FALSE(CanUseGpuIndirectDrawCount(0x1002, 4, 4));
	EXPECT_FALSE(CanUseGpuIndirectDrawCount(0x1000, 0, 4));
	EXPECT_FALSE(CanUseGpuIndirectDrawCount(0x1000, 5, 4));
}

TEST(DrawIndirectCount, DefaultsToTheCpuCountPath) {
	const DrawIndirectArgs args {};
	EXPECT_EQ(args.address, 0u);
	EXPECT_EQ(args.count_address, 0u);
	EXPECT_EQ(args.count, 0u);
	EXPECT_EQ(args.stride, 0u);
}

} // namespace
} // namespace Libs::Graphics

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
