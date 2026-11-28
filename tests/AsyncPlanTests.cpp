#include "common/compilePause.h"

#include <cstdio>
#include <cstdlib>

namespace {

using Change = Common::CompilePause::FrameStallTracker::Change;
using Tracker = Common::CompilePause::FrameStallTracker;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "AsyncPlanTests: failed: %s\n", message);
		std::abort();
	}
}

void TestFrameStallTracker() {
	Tracker off;
	Change  change = Change::None;
	Check(!off.Update(5, 1'000, 0, &change) && change == Change::None,
	      "a zero threshold never starts a stall");
	Check(!off.Update(5, 100'000, 0, &change) && off.frame == ~uint64_t {0},
	      "disabled tracking leaves its frame state untouched");

	Tracker tracker;
	Check(!tracker.Update(1, 0, 2'000, &change) && change == Change::None, "first draw");
	Check(!tracker.Update(2, 1'500, 2'000, &change) && change == Change::None, "new frame");
	Check(!tracker.Update(2, 3'499, 2'000, &change) && change == Change::None,
	      "just under the threshold");
	Check(tracker.Update(2, 3'500, 2'000, &change) && change == Change::Started,
	      "stall starts at the threshold");
	Check(tracker.Update(2, 9'000, 2'000, &change) && change == Change::None,
	      "stall continues without another edge");
	Check(tracker.deferred == 0 && tracker.deferred_total == 0,
	      "seeing draws during a stall does not count them as skipped");
	tracker.NoteDeferred();
	tracker.NoteDeferred();
	Check(tracker.deferred == 2 && tracker.deferred_total == 2, "actual skips are counted");
	Check(!tracker.Update(3, 9'100, 2'000, &change) && change == Change::Ended,
	      "the next frame ends the stall");
	Check(tracker.deferred == 2, "the ended stall keeps its count for reporting");
	Check(!tracker.Update(3, 11'000, 2'000, &change) && change == Change::None,
	      "the new frame gets a new clock origin");
	Check(tracker.Update(3, 11'100, 2'000, &change) && change == Change::Started,
	      "a second stall starts independently");
	Check(tracker.deferred == 0 && tracker.deferred_total == 2,
	      "the per-stall count resets but the total remains");
	tracker.NoteDeferred();
	Check(tracker.deferred == 1 && tracker.deferred_total == 3, "the second stall is counted");

	Tracker backwards;
	Check(!backwards.Update(1, 5'000, 2'000, &change), "clock starts");
	Check(!backwards.Update(1, 1'000, 2'000, &change) && change == Change::None,
	      "a backwards clock does not start a stall");
}

} // namespace

int main() {
	TestFrameStallTracker();
	std::printf("AsyncPlanTests: all passed\n");
	return 0;
}
