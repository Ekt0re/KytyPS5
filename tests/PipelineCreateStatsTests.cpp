#include "graphics/host_gpu/renderer/pipeline/createStats.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PipelineCreateStatsTests: failed: %s\n", message);
		std::abort();
	}
}

namespace CS = Libs::Graphics::CreateStats;

void TestBucketEdges() {
	Check(CS::Bucket(0) == 0 && CS::Bucket(9'999) == 0 && CS::Bucket(10'000) == 1 &&
	          CS::Bucket(99'999) == 1 && CS::Bucket(100'000) == 2 &&
	          CS::Bucket(999'999) == 2 && CS::Bucket(1'000'000) == 3 &&
	          CS::Bucket(10'000'000) == 4,
	      "bucket edges");
	Check(CS::BucketCount == 5, "five buckets");
}

void TestSyntheticEvents() {
	CS::Totals totals;
	CS::Enter(totals);
	CS::Enter(totals);
	CS::Leave(totals, totals.compute, 4'000, true, true, 3'000'000);
	CS::Leave(totals, totals.compute, 2'500'000, true, false, 2'400'000'000);
	CS::Enter(totals);
	CS::Leave(totals, totals.graphics, 500, false, false, 0);
	const auto line = CS::Line(totals);
	Check(line.find("cs n=2 wall=2504ms driver=2403ms max=2500ms cache-hit=1/2 <10ms=1 "
	                "<100ms=0 <1s=0 <10s=1 >=10s=0") != std::string::npos,
	      "compute figures");
	Check(line.find("gfx n=1") != std::string::npos &&
	          line.find("cache-hit=0/0") != std::string::npos,
	      "graphics figures");
	Check(line.find("in-driver now=0 peak=2") != std::string::npos, "concurrency");
}

void TestEmptyAndInvalidFeedback() {
	CS::Totals totals;
	Check(CS::Line(totals).find("[pipeline-create] cs n=0") != std::string::npos, "empty line");
	CS::Enter(totals);
	CS::Leave(totals, totals.compute, 10'000'000, false, true, 1'000);
	const auto line = CS::Line(totals);
	Check(line.find("cache-hit=0/0") != std::string::npos &&
	          line.find(">=10s=1") != std::string::npos,
	      "invalid feedback is not counted as a hit");
}

} // namespace

int main() {
	TestBucketEdges();
	TestSyntheticEvents();
	TestEmptyAndInvalidFeedback();
	std::puts("PipelineCreateStatsTests: ok");
	return 0;
}
