#include "graphics/guest_gpu/command_processor/waitDiagnostics.h"
#include "kernel/guestWait.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace Libs::Graphics::WaitDiag;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "WaitDiagnosticsTests: failed: %s\n", message);
		std::abort();
	}
}

bool Has(const std::vector<std::string>& lines, const std::string& part) {
	for (const auto& line: lines) {
		if (line.find(part) != std::string::npos) {
			return true;
		}
	}
	return false;
}

WaitPacket Packet(uint64_t address, uint64_t reference, uint64_t value) {
	WaitPacket packet;
	packet.address   = address;
	packet.reference = reference;
	packet.mask      = 0xffffffffu;
	packet.func      = 3;
	packet.value     = value;
	packet.buffer    = 0x1000;
	packet.offset_dw = 8;
	return packet;
}

void TestDecodeAndPollCount() {
	Registry r;
	r.NoteQueue(0, QueueState::WaitingMemory, 0, 1000);
	uint64_t polls = 0;
	for (int i = 0; i < 5; i++) {
		polls = r.NotePoll(0, Packet(0x20000, 7, 0), 1000 + i);
	}
	Check(polls == 5, "five polls of one packet count five");
	Check(r.NotePoll(0, Packet(0x20000, 8, 0), 1010) == 1, "another reference is another packet");
	const auto lines = r.Report(2000);
	Check(Has(lines, "wait_reg_mem32 object=0x20000"), "address decoded with the shared wait type");
	Check(Has(lines, "value=0 for 1.0 s"), "last value and wait age decoded");
	Check(Has(lines, "compare (value & mask) == 0x8"), "compare decoded");
	Check(Has(lines, "polled 1 times"), "poll count shown");
	Check(Has(lines, "no label write within 4 KiB"), "empty ring said so");
	Check(Has(lines, "no recent read back covers"), "no readback said so");
}

void TestLabelsNearTheWaitedAddress() {
	Registry r;
	r.NoteQueue(0, QueueState::WaitingMemory, 0, 0);
	r.NotePoll(0, Packet(0x20000, 7, 0), 100);
	// Labels carry a queue slot (2 names event id 0x21), not the event id itself.
	r.NoteLabel(0x20800, 5, 4, 2, 500); // near
	r.NoteLabel(0x30000, 6, 4, 2, 600); // far
	r.NoteLabel(0x20000, 4, 4, 2, 700); // the address itself
	r.NoteReadback(0x1f000, 0x2000, 800);
	const auto lines = r.Report(1000);
	Check(Has(lines, "0x0000000000020800 <- 0x5"), "near label shown");
	Check(!Has(lines, "0x0000000000030000"), "far label not shown");
	Check(Has(lines, "[the waited address]"), "exact address marked");
	Check(Has(lines, "read back 0x000000000001f000"), "covering readback shown");
	Check(Has(lines, "queue 0x21 (compute)"), "compute queue named");
}

void TestRingKeepsOnlyTheLast32() {
	Registry r;
	r.NoteQueue(0, QueueState::WaitingMemory, 0, 0);
	r.NotePoll(0, Packet(0x20000, 7, 0), 100);
	r.NoteLabel(0x20004, 1, 4, 0, 10);
	for (uint32_t i = 0; i < LabelRingLen; i++) {
		r.NoteLabel(0x90000 + i * 4, i, 4, 0, 20 + i);
	}
	const auto lines = r.Report(1000);
	Check(!Has(lines, "0x0000000000020004"), "the oldest write was overwritten");
	Check(Has(lines, "no label write within 4 KiB"), "and the ring says nothing is near");
}

void TestOtherQueueStates() {
	Registry r;
	r.NoteQueue(0, QueueState::WaitingMemory, 0, 0);
	r.NotePoll(0, Packet(0x20000, 7, 0), 100);
	r.NoteQueue(1, QueueState::SuspendedForShader, 0xabcdull, 100);
	r.NoteQueue(2, QueueState::Idle, 0, 100);
	r.NoteRunning(3, 100);
	const auto lines = r.Report(5100);
	Check(Has(lines, "suspended for shader 0x000000000000abcd for 5.0 s"), "shader queue shown");
	Check(Has(lines, "queue 0x21 (compute): idle"), "idle queue shown");
	Check(Has(lines, "queue 0x22 (compute): running"), "running queue shown");
}

void TestStateKeepsItsStartTime() {
	Registry r;
	r.NoteQueue(1, QueueState::SuspendedForShader, 5, 100);
	r.NoteQueue(1, QueueState::SuspendedForShader, 5, 4000);
	r.NoteRunning(1, 4500); // a retry slice must not reset a known wait
	const auto lines = r.Report(5100);
	Check(Has(lines, "for 5.0 s"), "wait start kept across retries");
}

void TestWaitWithoutPacketIsSaid() {
	Registry r;
	r.NoteQueue(0, QueueState::WaitingMemory, 0, 0);
	r.NotePoll(0, Packet(0x20000, 7, 0), 100);
	const auto lines = r.Report(9000); // last poll long ago: it waits on something else now
	Check(Has(lines, "not on a wait_reg_mem packet"), "stale packet is not shown as the cause");
}

void TestGpuFlagsSurvivePolls() {
	Registry r;
	r.NoteQueue(0, QueueState::WaitingMemory, 0, 0);
	auto first         = Packet(0x20000, 7, 0);
	first.gpu_modified = 1;
	first.gpu_dirty    = 1;
	r.NotePoll(0, first, 100);
	r.NotePoll(0, Packet(0x20000, 7, 0), 200);
	const auto lines = r.Report(300);
	Check(Has(lines, "range written by a shader: yes; written bytes not read back yet: yes"),
	      "flags kept");
}

void TestSharedGuestWaitDescription() {
	Libs::LibKernel::GuestWaitState state;
	Check(Libs::LibKernel::DescribeWait(state, 1000000000ull) == "running", "no wait means running");
	state.call.store("sceKernelWaitEventFlag", std::memory_order_release);
	state.object.store(0x1234, std::memory_order_relaxed);
	state.value.store(9, std::memory_order_relaxed);
	state.since.store(0ull, std::memory_order_relaxed);
	const auto described = Libs::LibKernel::DescribeWait(state, 2000000000ull);
	Check(described.find("waits in sceKernelWaitEventFlag object=0x1234 value=9") !=
	          std::string::npos,
	      "the crash-time description is reused for a wait");
	Libs::LibKernel::GuestWaitState from_packet;
	Packet(0x20000, 7, 0).FillGuestWaitState(from_packet, 1000);
	Check(from_packet.object.load() == 0x20000, "queue wait uses the same type");
}

} // namespace

int main() {
	TestDecodeAndPollCount();
	TestLabelsNearTheWaitedAddress();
	TestRingKeepsOnlyTheLast32();
	TestOtherQueueStates();
	TestStateKeepsItsStartTime();
	TestWaitWithoutPacketIsSaid();
	TestGpuFlagsSurvivePolls();
	TestSharedGuestWaitDescription();
	std::puts("WaitDiagnosticsTests: ok");
	return 0;
}
