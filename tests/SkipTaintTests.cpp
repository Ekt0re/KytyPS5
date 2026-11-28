#include "graphics/host_gpu/renderer/pipeline/skipCatchUp.h"
#include "graphics/host_gpu/renderer/pipeline/skipTaint.h"

#include <cstdio>
#include <cstdlib>

using namespace Libs::Graphics;
using Use = SkipTaint::Use;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SkipTaintTests: failed: %s\n", message);
		std::abort();
	}
}

void TestFollowsData() {
	SkipTaint taint;
	Check(taint.Empty() && !taint.ReadsTainted({{0x1000, 64, true, false}}, 0), "empty reads nothing");
	// The skipped producer writes B and reads A; A is not tainted.
	taint.Taint({{0x1000, 256, true, false}, {0x2000, 256, false, true}}, 100);
	Check(taint.Size() == 1, "only written ranges are tainted");
	Check(!taint.ReadsTainted({{0x1000, 256, true, false}}, 110), "the producer's input is clean");
	Check(taint.ReadsTainted({{0x2080, 4, true, false}}, 110), "a read inside the output is tainted");
	Check(taint.ReadsTainted({{0x1f00, 0x200, true, false}}, 110), "a read overlapping the edge is tainted");
	Check(!taint.ReadsTainted({{0x2100, 16, true, false}}, 110), "a read past the end is clean");
	Check(!taint.ReadsTainted({{0x2000, 256, false, true}}, 110), "a pure write does not read");
}

void TestClearAndSplit() {
	SkipTaint taint;
	taint.Taint({{0x2000, 0x300, false, true}}, 0);
	taint.ClearWritten({{0x2100, 0x100, false, true}});
	Check(taint.Size() == 2, "a write inside splits the range");
	Check(taint.ReadsTainted({{0x2000, 4, true, false}}, 1), "the head stays tainted");
	Check(!taint.ReadsTainted({{0x2100, 0x100, true, false}}, 1), "the written middle is clean");
	Check(taint.ReadsTainted({{0x22fc, 4, true, false}}, 1), "the tail stays tainted");
	taint.ClearWritten({{0x1000, 0x2000, false, true}});
	Check(taint.Empty(), "a covering write clears everything");
	taint.Taint({{0x3000, 16, false, true}}, 0);
	taint.ClearWritten({{0x3000, 16, true, false}});
	Check(taint.Size() == 1, "a read-only use clears nothing");
}

void TestTransitiveAndExpiry() {
	SkipTaint taint(1000);
	taint.Taint({{0x1000, 64, false, true}}, 0);
	const std::vector<Use> consumer {{0x1000, 64, true, false}, {0x5000, 64, false, true}};
	Check(taint.ReadsTainted(consumer, 10), "the consumer reads tainted data");
	taint.Taint(consumer, 10); // it is skipped: its output is tainted too
	Check(taint.ReadsTainted({{0x5000, 4, true, false}}, 20), "taint follows the data");
	Check(taint.ReadsTainted({{0x5000, 4, true, false}}, 1005), "the second range is younger");
	Check(!taint.ReadsTainted({{0x1000, 4, true, false}}, 1005), "the first range expired");
	Check(!taint.ReadsTainted({{0x5000, 4, true, false}}, 1011), "the second expired too");
	Check(taint.Empty(), "expired ranges are dropped");
}

void TestRepeatedSkipsStayBounded() {
	SkipTaint taint;
	// The same producers are skipped frame after frame: the set must not grow with the skips.
	for (uint64_t frame = 0; frame < 10000; frame++) {
		taint.Taint({{0x2000, 0x300, false, true}, {0x2100, 0x100, false, true}, {0x8000, 64, false, true}}, frame);
	}
	Check(taint.Size() <= 4, "repeated skips of the same ranges do not pile up");
	Check(taint.ReadsTainted({{0x2000, 4, true, false}}, 10000), "still tainted");
	taint.ClearWritten({{0x2000, 0x300, false, true}});
	Check(taint.Size() == 1 && !taint.ReadsTainted({{0x2000, 0x300, true, false}}, 10000), "one write clears all of it");
	// A newer skip refreshes the age of what it covers.
	SkipTaint aging(1000);
	aging.Taint({{0x1000, 64, false, true}}, 0);
	aging.Taint({{0x1000, 64, false, true}}, 900);
	Check(aging.ReadsTainted({{0x1000, 4, true, false}}, 1500), "the newer skip keeps the range tainted");
}

void TestCatchUpReplaysProducerBeforeConsumer() {
	SkipTaint                    taint;
	SkipCatchUp::Queue<uint32_t> queue(5000);
	const std::vector<Use>       prod_uses {{0x2000, 256, false, true}};
	const std::vector<Use>       cons_uses {{0x2000, 256, true, false}, {0x3000, 256, false, true}};

	// Producer skipped on translation; consumer skipped because it reads the producer's buffer.
	taint.Taint(prod_uses, 100);
	queue.Enqueue(1, 0xaaaa000000000001ull, prod_uses, 101u, 100);
	Check(taint.ReadsTainted(cons_uses, 105), "consumer sees producer output tainted");
	taint.Taint(cons_uses, 105);
	queue.Enqueue(2, 0xaaaa000000000002ull, cons_uses, 102u, 105);
	// A repeat skip of the same producer refreshes instead of growing the queue.
	queue.Enqueue(1, 0xaaaa000000000001ull, prod_uses, 101u, 110);
	Check(queue.Size() == 2, "duplicate key deduplicates");

	// While the producer program is still pending, its consumer stays waiting on it.
	auto none = queue.TakeReady(200, [](const auto&) { return SkipCatchUp::ProbeState::Pending; });
	Check(none.empty() && queue.Size() == 2, "pending entries remain queued");

	// Once both programs are available, they replay in submission order and clear the taint.
	auto ready = queue.TakeReady(300, [](const auto&) { return SkipCatchUp::ProbeState::Ready; });
	Check(ready.size() == 2 && ready[0].payload == 101u && ready[1].payload == 102u,
	      "producer replays before consumer in FIFO order");
	for (const auto& entry: ready) {
		taint.ClearWritten(entry.uses);
	}
	Check(taint.Empty() && queue.Empty(), "replay clears taint and drains queue");
	Check(queue.GetStats().replayed == 2, "replayed stat tracks replays");

	// A fresh dispatch that overwrote the outputs supersedes the queued producer.
	queue.Enqueue(3, 0xaaaa000000000003ull, prod_uses, 103u, 400);
	queue.DropSuperseded({{0x2000, 512, false, true}});
	Check(queue.Empty() && queue.GetStats().superseded == 1,
	      "covering write supersedes stale queued producer");

	// A permanently failed program is dropped cleanly instead of retrying forever.
	queue.Enqueue(4, 0xaaaa000000000004ull, prod_uses, 104u, 500);
	auto failed = queue.TakeReady(600, [](const auto&) { return SkipCatchUp::ProbeState::Failed; });
	Check(failed.empty() && queue.Empty() && queue.GetStats().dropped_failed == 1,
	      "permanent failure drops queued entry");
}

// A ready consumer must not replay ahead of an earlier producer that is still pending.
void TestReadyEntryWaitsBehindPendingProducer() {
	SkipCatchUp::Queue<uint32_t> queue(5000);
	const std::vector<Use>       prod {{0x2000, 256, false, true}};
	const std::vector<Use>       cons {{0x2000, 256, true, false}, {0x3000, 256, false, true}};
	const std::vector<Use>       other {{0x9000, 256, false, true}};
	queue.Enqueue(1, 1, prod, 1u, 0);
	queue.Enqueue(2, 2, cons, 2u, 0);
	queue.Enqueue(3, 3, other, 3u, 0);
	const auto probe = [](const auto& e) {
		return e.payload == 1u ? SkipCatchUp::ProbeState::Pending : SkipCatchUp::ProbeState::Ready;
	};
	auto ready = queue.TakeReady(10, probe);
	Check(ready.size() == 1 && ready[0].payload == 3u, "only the independent entry replays");
	Check(queue.Size() == 2, "producer and its consumer stay queued");
	ready = queue.TakeReady(20, [](const auto&) { return SkipCatchUp::ProbeState::Ready; });
	Check(ready.size() == 2 && ready[0].payload == 1u && ready[1].payload == 2u,
	      "producer then consumer once the producer is ready");
	// Two readers of the same buffer do not order each other.
	queue.Enqueue(4, 4, {{0x5000, 16, true, false}}, 4u, 30);
	queue.Enqueue(5, 5, {{0x5000, 16, true, false}}, 5u, 30);
	ready = queue.TakeReady(40, [](const auto& e) {
		return e.payload == 4u ? SkipCatchUp::ProbeState::Pending : SkipCatchUp::ProbeState::Ready;
	});
	Check(ready.size() == 1 && ready[0].payload == 5u, "read-read does not conflict");
	// A failed earlier producer blocks nothing.
	queue.Enqueue(6, 6, prod, 6u, 50);
	queue.Enqueue(7, 7, cons, 7u, 50);
	ready = queue.TakeReady(60, [](const auto& e) {
		return e.payload == 6u ? SkipCatchUp::ProbeState::Failed : SkipCatchUp::ProbeState::Ready;
	});
	Check(ready.size() == 2 && ready[0].payload == 4u && ready[1].payload == 7u,
	      "failed producer does not block");
}

} // namespace

int main() {
	TestRepeatedSkipsStayBounded();
	TestReadyEntryWaitsBehindPendingProducer();
	TestFollowsData();
	TestClearAndSplit();
	TestTransitiveAndExpiry();
	TestCatchUpReplaysProducerBeforeConsumer();
	std::printf("SkipTaintTests: ok\n");
	return 0;
}
