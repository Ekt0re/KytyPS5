#include "graphics/host_gpu/renderer/pipeline/bufferProbe.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace Libs::Graphics;

TEST(BufferProbe, IdenticalBytesDoNotDiffer) {
	std::vector<uint8_t> guest(3 * 4096 + 100, 0);
	const auto           host = guest;
	const auto           c    = BufferProbe::Compare(guest.data(), host.data(), guest.size());
	EXPECT_FALSE(c.Differs());
	EXPECT_EQ(c.first_diff, UINT64_MAX);
	EXPECT_EQ(c.differing_pages, 0u);
	EXPECT_EQ(c.total_pages, 4u);
}

TEST(BufferProbe, FirstDifferencePagesAndPreviewWords) {
	std::vector<uint8_t> guest(3 * 4096 + 100, 0);
	auto                 host = guest;
	host[4096 + 7]          = 0xab;
	host[3 * 4096 + 5]      = 1;
	host[3 * 4096 + 99]     = 2;
	for (uint32_t i = 0; i < BufferProbe::PreviewWords; i++) {
		guest[i * 4] = static_cast<uint8_t>(i + 1);
		host[i * 4]  = static_cast<uint8_t>(i + 1);
	}
	const auto c = BufferProbe::Compare(guest.data(), host.data(), guest.size());
	EXPECT_TRUE(c.Differs());
	EXPECT_EQ(c.first_diff, 4096u + 7u);
	EXPECT_EQ(c.differing_pages, 2u);
	EXPECT_EQ(c.differing_bytes, 3u);
	EXPECT_EQ(c.guest_head[0], 1u);
	EXPECT_EQ(c.host_head[BufferProbe::PreviewWords - 1], BufferProbe::PreviewWords);
	// The words around the difference start at the dword that holds it.
	EXPECT_EQ(c.host_at_diff[0], 0xab000000u);
	EXPECT_EQ(c.guest_at_diff[0], 0u);
}

TEST(BufferProbe, ShortBufferReadsZeroPastTheEnd) {
	const uint8_t a[6] {1, 0, 0, 0, 9, 9};
	const uint8_t b[6] {1, 0, 0, 0, 9, 8};
	const auto    c = BufferProbe::Compare(a, b, 6);
	EXPECT_EQ(c.first_diff, 5u);
	EXPECT_EQ(c.host_head[2], 0u);
	EXPECT_EQ(c.guest_head[1], 0x0909u);
}

TEST(BufferProbe, CacheStateClassifiesExpectedDifferences) {
	using BufferProbe::CacheState;
	EXPECT_EQ(BufferProbe::ClassifyCacheState(false, false), CacheState::Synced);
	EXPECT_EQ(BufferProbe::ClassifyCacheState(true, false), CacheState::CpuDirty);
	EXPECT_EQ(BufferProbe::ClassifyCacheState(false, true), CacheState::GpuDirty);
	EXPECT_EQ(BufferProbe::ClassifyCacheState(true, true), CacheState::Both);
	BufferProbe::Comparison same;
	BufferProbe::Comparison different;
	different.first_diff = 4;
	EXPECT_FALSE(BufferProbe::IsUnexpectedDifference(same, CacheState::Synced));
	EXPECT_TRUE(BufferProbe::IsUnexpectedDifference(different, CacheState::Synced));
	EXPECT_TRUE(BufferProbe::IsUnexpectedDifference(different, CacheState::CpuDirty));
	EXPECT_FALSE(BufferProbe::IsUnexpectedDifference(different, CacheState::GpuDirty));
	EXPECT_FALSE(BufferProbe::IsUnexpectedDifference(different, CacheState::Both));
}

TEST(BufferProbe, LineReportsFieldsWithoutProgramIdentity) {
	BufferProbe::Comparison c;
	c.compared_bytes   = 8192;
	c.total_pages      = 2;
	c.first_diff       = 0x1004;
	c.differing_pages  = 1;
	c.differing_bytes  = 4;
	const auto line     = BufferProbe::FormatLine(3, 1, 0x200008c000ull, 8192, true,
	                                              BufferProbe::CacheState::Synced, true, c);
	EXPECT_NE(line.find("[buffer-probe] dispatch=3 buf=1"), std::string::npos) << line;
	EXPECT_NE(line.find("first_diff=0x1004"), std::string::npos) << line;
	EXPECT_NE(line.find("pages_differ=1/2"), std::string::npos) << line;
	EXPECT_NE(line.find("cache=synced"), std::string::npos) << line;
	EXPECT_NE(line.find("UNEXPECTED"), std::string::npos) << line;
	EXPECT_EQ(line.find("shader="), std::string::npos) << line;

	BufferProbe::Comparison none;
	const auto               ok = BufferProbe::FormatLine(1, 0, 0x1000, 16, false,
	                                                      BufferProbe::CacheState::GpuDirty, false,
	                                                      none);
	EXPECT_NE(ok.find("first_diff=none"), std::string::npos) << ok;
	EXPECT_NE(ok.find("guest=unreadable"), std::string::npos) << ok;
	EXPECT_EQ(ok.find("UNEXPECTED"), std::string::npos) << ok;
}

TEST(BufferProbe, LimiterCapsLinesAndSummarises) {
	BufferProbe::Limiter limiter;
	uint32_t             printed = 0;
	for (uint32_t i = 0; i < 1000; i++) {
		const bool mismatch = i % 2 == 0;
		if (limiter.OnDispatch(mismatch, mismatch)) printed++;
		EXPECT_EQ(limiter.SummaryDue(), (i + 1) % BufferProbe::SummaryInterval == 0);
	}
	// The first 40 mismatches plus the two clean dispatches among the first five probes.
	EXPECT_EQ(printed, BufferProbe::MaxMismatchPrints + 2u);
	EXPECT_EQ(limiter.Printed(), BufferProbe::MaxMismatchPrints);
	EXPECT_EQ(limiter.Mismatches(), 500u);
	EXPECT_EQ(limiter.Dispatches(), 1000u);
	EXPECT_NE(limiter.Summary().find("probed_dispatches=1000 mismatching=500 unexpected=500"),
	          std::string::npos);
}

} // namespace

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
