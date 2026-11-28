#include "graphics/host_gpu/renderer/dispatchLimits.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DispatchLimitsTests: failed: %s\n", message);
		std::abort();
	}
}

void TestCounts() {
	// The count is a full 32-bit value: nothing in the upper byte is a flag.
	Check(DispatchGroups(0x01000062u, 1, 1) == 16777314ull, "the upper byte is part of the count");
	Check(DispatchInvocations(0x01000062u, 1, 1, 64) == 16777314ull * 64u, "invocations");
	Check(DispatchGroups(UINT32_MAX, UINT32_MAX, UINT32_MAX) == UINT64_MAX, "groups saturate");
	Check(DispatchInvocations(UINT32_MAX, UINT32_MAX, UINT32_MAX, 1024) == UINT64_MAX,
	      "invocations saturate");
	Check(DispatchGroups(0, UINT32_MAX, UINT32_MAX) == 0, "a zero count stays zero");
}

void TestJudge() {
	const uint32_t limit = DispatchDefaultMaxGroups;
	Check(JudgeDispatch(480, 270, 1, 64, limit, true) == DispatchVerdict::Run, "a frame pass runs");
	Check(JudgeDispatch(1, 1, 1, 1024, limit, true) == DispatchVerdict::Run, "one group runs");
	Check(JudgeDispatch(1u << 20u, 1, 1, 64, limit, true) == DispatchVerdict::Report,
	      "large counts are reported");
	Check(JudgeDispatch(limit, 1, 1, 64, limit, true) == DispatchVerdict::Report,
	      "exactly the limit still runs");
	Check(JudgeDispatch(limit + 1, 1, 1, 64, limit, true) == DispatchVerdict::Skip,
	      "above the limit is skipped");
	Check(JudgeDispatch(4096, 4096, 1, 1, limit, true) == DispatchVerdict::Skip,
	      "the product across axes is judged");
	Check(JudgeDispatch(0x01000062u, 1, 1, 64, limit, false) == DispatchVerdict::Report,
	      "without a skip the large dispatch is only reported");
	Check(JudgeDispatch(1000, 1, 1, 64, limit, false) == DispatchVerdict::Run,
	      "small counts stay quiet");
}

void TestCountDword() {
	// The first words are user data of a dispatch: a float count divided by the group size gives
	// the x count.
	const uint32_t words[4] = {0x012e000cu, 0u, 0x40014526u, 1u};
	Check(FindCountDword(words, 4, 0x01000515u, 64) == 2, "finds the dword the count came from");
	Check(FindCountDword(words, 4, 0x01000516u, 64) == -1, "no dword matches");
	Check(FindCountDword(words, 4, 1, 64) == -1, "a count of one matches nothing");
	Check(FindCountDword(words, 4, 5, 0) == -1, "a zero group size matches nothing");
	const float two = DwordAsFloat(0x40014526u);
	Check(two > 2.0f && two < 2.01f, "the dword is a float near two");
	Check(DwordAsFloat(0xbffebae2u) < -1.99f, "the negated float");
}

void TestLimiter() {
	DispatchReportLimiter limiter(2);
	Check(limiter.Allow(7) && limiter.Allow(7) && !limiter.Allow(7), "two reports per shader");
	Check(limiter.Allow(8), "another shader has its own count");
	Check(!limiter.Allow(7), "the first shader stays limited");
}

} // namespace

int main() {
	TestCounts();
	TestJudge();
	TestCountDword();
	TestLimiter();
	std::printf("DispatchLimitsTests: ok\n");
	return 0;
}
