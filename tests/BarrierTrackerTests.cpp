#include "graphics/host_gpu/renderer/barrierTracker.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "BarrierTrackerTests: failed: %s\n", text);
    std::abort();
  }
}

ShaderAccessSet Access(uint64_t read_address, uint64_t read_size, uint64_t write_address,
                       uint64_t write_size) {
  ShaderAccessSet result;
  result.AddRead(read_address, read_size);
  result.AddWrite(write_address, write_size);
  return result;
}

void TestHazards() {
  struct Case {
    ShaderAccessSet previous;
    ShaderAccessSet current;
    BarrierDecision expected;
    const char *text;
  };
  const ShaderAccessSet read_a = Access(0x1000, 0x100, 0, 0);
  const ShaderAccessSet read_b = Access(0x2000, 0x100, 0, 0);
  const ShaderAccessSet write_a = Access(0, 0, 0x1000, 0x100);
  const ShaderAccessSet write_b = Access(0, 0, 0x2000, 0x100);
  const ShaderAccessSet partial = Access(0x1080, 0x100, 0, 0);
  const ShaderAccessSet nested = Access(0x1040, 0x40, 0, 0);
  const Case cases[] = {
      {write_a, read_a, BarrierDecision::ComputeToCompute, "RAW"},
      {read_a, write_a, BarrierDecision::ComputeToCompute, "WAR"},
      {write_a, write_a, BarrierDecision::ComputeToCompute, "WAW"},
      {read_a, read_a, BarrierDecision::None, "RAR"},
      {write_a, read_b, BarrierDecision::None, "disjoint ranges"},
      {read_a, write_b, BarrierDecision::None, "disjoint inverse ranges"},
      {write_a, Access(0x1100, 0x100, 0, 0), BarrierDecision::None, "touching ranges"},
      {write_a, partial, BarrierDecision::ComputeToCompute, "partial overlap"},
      {write_a, nested, BarrierDecision::ComputeToCompute, "nested overlap"},
  };
  for (const auto &item : cases) {
    BarrierTracker tracker;
    tracker.AfterDispatch(item.previous);
    Check(tracker.BeforeDispatch(item.current) == item.expected, item.text);
  }

  BarrierTracker aliases;
  aliases.AfterDispatch(write_a);
  ShaderAccessSet alias_read;
  alias_read.AddRead(0x1000, 0x40);
  alias_read.AddRead(0x1080, 0x80);
  Check(aliases.BeforeDispatch(alias_read) == BarrierDecision::ComputeToCompute,
        "the same guest range reached through two resources aliases");
}

void TestUnknownFallback() {
  const auto run = [](ShaderAccessSet previous, ShaderAccessSet current, BarrierDecision expected,
                      const char *text) {
    BarrierTracker tracker;
    tracker.AfterDispatch(previous);
    Check(tracker.BeforeDispatch(current) == expected, text);
  };
  ShaderAccessSet unknown_read;
  unknown_read.unknown_reads = true;
  ShaderAccessSet unknown_write;
  unknown_write.unknown_writes = true;
  run(unknown_read, Access(0, 0, 0x1000, 0x100), BarrierDecision::ComputeToCompute,
      "unknown prior read versus known write");
  run(Access(0, 0, 0x1000, 0x100), unknown_read, BarrierDecision::ComputeToCompute,
      "known prior write versus unknown current read");
  run(Access(0x1000, 0x100, 0, 0), unknown_write, BarrierDecision::ComputeToCompute,
      "known prior read versus unknown current write");
  run(unknown_write, Access(0x1000, 0x100, 0, 0), BarrierDecision::ComputeToCompute,
      "unknown prior write versus known current read");
  run(unknown_read, Access(0x1000, 0x100, 0, 0), BarrierDecision::None,
      "unknown reads do not order read/read");

  ShaderAccessSet overflowing;
  overflowing.AddRead(UINT64_MAX, 2);
  Check(overflowing.unknown_reads, "an overflowing range becomes unknown");
}

void TestDeferredOrdering() {
  BarrierTracker tracker;
  const auto write_a = Access(0, 0, 0x1000, 0x100);
  const auto write_b = Access(0, 0, 0x2000, 0x100);
  const auto read_a = Access(0x1000, 0x100, 0, 0);
  const auto read_b = Access(0x2000, 0x100, 0, 0);
  tracker.AfterDispatch(write_a);
  Check(tracker.HasPendingDispatches() && tracker.HasPendingWrites(),
        "a write remains pending after dispatch");
  Check(tracker.BeforeDispatch(write_b) == BarrierDecision::None, "disjoint writes need no barrier");
  tracker.AfterDispatch(write_b);
  Check(tracker.BeforeDispatch(read_a) == BarrierDecision::ComputeToCompute, "union keeps the first hazard");
  tracker.AfterDispatch(read_a);
  Check(tracker.BeforeDispatch(read_b) == BarrierDecision::None, "a global barrier clears older history");
  tracker.AfterDispatch(read_b);
  Check(!tracker.HasPendingWrites(), "pending reads are not pending writes");
  Check(tracker.BeforeCommand() == BarrierDecision::ComputeToAll, "other command flushes pending reads");
  Check(tracker.BeforeDispatch(write_b) == BarrierDecision::AllToCompute,
        "write after other command needs all-to-compute");
  tracker.AfterDispatch(write_b);
  Check(tracker.BeforeBarrier() == BarrierDecision::ComputeToAll, "explicit barrier flushes dispatches");
  Check(tracker.BeforeDispatch(read_b) == BarrierDecision::None, "barrier does not mark other work");
  tracker.AfterDispatch(read_b);
  Check(tracker.Flush() == BarrierDecision::ComputeToAll, "flush orders pending dispatches");
  Check(tracker.BeforeCommand() == BarrierDecision::None, "command without dispatch");
  Check(tracker.Flush() == BarrierDecision::AllToAll, "flush orders unknown other work");
  tracker.Reset();
  Check(!tracker.HasPendingDispatches(), "reset drops pending state");

  BarrierTracker other;
  Check(other.BeforeCommand() == BarrierDecision::None, "other command without dispatch");
  Check(other.BeforeDispatch(read_a) == BarrierDecision::None, "read after other command");
  other.AfterDispatch(read_a);
  Check(other.BeforeDispatch(write_b) == BarrierDecision::None,
        "a read consumes the other-command boundary before a disjoint write");
}

struct Reference {
  std::vector<std::pair<uint64_t, uint64_t>> reads;
  std::vector<std::pair<uint64_t, uint64_t>> writes;
  bool unknown_reads = false;
  bool unknown_writes = false;
  bool pending = false;
  bool other = false;

  static bool Overlaps(const std::vector<std::pair<uint64_t, uint64_t>> &ranges,
                       uint64_t address, uint64_t size) {
    for (const auto &[begin, end] : ranges) {
      if (address < end && begin < address + size) return true;
    }
    return false;
  }
  bool Hazard(const ShaderAccessSet &current) const {
    if (unknown_writes && current.HasAccesses()) return true;
    if (current.unknown_writes &&
        (unknown_reads || unknown_writes || !reads.empty() || !writes.empty()))
      return true;
    if (unknown_reads && current.HasWrites()) return true;
    if (current.unknown_reads && (unknown_writes || !writes.empty())) return true;
    bool hazard = false;
    current.reads.ForEach([&](uint64_t begin, uint64_t end) {
      hazard |= Overlaps(writes, begin, end - begin);
    });
    current.writes.ForEach([&](uint64_t begin, uint64_t end) {
      hazard |= Overlaps(reads, begin, end - begin) || Overlaps(writes, begin, end - begin);
    });
    return hazard;
  }
  void Clear() {
    reads.clear();
    writes.clear();
    unknown_reads = unknown_writes = pending = false;
  }
  BarrierDecision Before(const ShaderAccessSet &current) {
    if (other) {
      other = false;
      if (current.HasWrites()) return BarrierDecision::AllToCompute;
    }
    if (pending && Hazard(current)) {
      Clear();
      return BarrierDecision::ComputeToCompute;
    }
    return BarrierDecision::None;
  }
  void After(const ShaderAccessSet &current) {
    current.reads.ForEach([&](uint64_t begin, uint64_t end) {
      reads.emplace_back(begin, end);
    });
    current.writes.ForEach([&](uint64_t begin, uint64_t end) {
      writes.emplace_back(begin, end);
    });
    unknown_reads |= current.unknown_reads;
    unknown_writes |= current.unknown_writes;
    pending = true;
    other = false;
  }
  BarrierDecision Command() {
    if (!pending) {
      other = true;
      return BarrierDecision::None;
    }
    Clear();
    other = true;
    return BarrierDecision::ComputeToAll;
  }
  BarrierDecision ExplicitBarrier() {
    if (!pending) return BarrierDecision::None;
    Clear();
    return BarrierDecision::ComputeToAll;
  }
  BarrierDecision Flush() {
    if (pending) {
      Clear();
      other = false;
      return BarrierDecision::ComputeToAll;
    }
    if (other) {
      other = false;
      return BarrierDecision::AllToAll;
    }
    return BarrierDecision::None;
  }
};

void TestRandomizedReference() {
  std::mt19937 random {0x5eed1234u};
  BarrierTracker tracker;
  Reference reference;
  uint32_t tracked_barriers = 0;
  uint32_t legacy_barriers = 0;
  for (uint32_t round = 0; round < 3000; ++round) {
    ShaderAccessSet current;
    const uint32_t ranges = random() % 4u;
    for (uint32_t i = 0; i < ranges; ++i) {
      const uint64_t address = 0x1000u + (random() % 8u) * 0x100u;
      const uint64_t size = 0x20u << (random() % 4u);
      if ((random() & 1u) != 0) current.AddRead(address, size);
      if ((random() & 1u) != 0) current.AddWrite(address, size);
    }
    current.unknown_reads = (random() % 97u) == 0;
    current.unknown_writes = (random() % 131u) == 0;
    const uint32_t operation = random() % 10u;
    if (operation < 6) {
      const auto expected = reference.Before(current);
      Check(tracker.BeforeDispatch(current) == expected, "random dispatch decision");
      tracked_barriers += expected == BarrierDecision::None ? 0u : 1u;
      legacy_barriers += current.HasWrites() ? 2u : 1u;
      tracker.AfterDispatch(current);
      reference.After(current);
    } else if (operation < 8) {
      const auto expected = reference.Command();
      Check(tracker.BeforeCommand() == expected, "random other command");
      tracked_barriers += expected == BarrierDecision::None ? 0u : 1u;
    } else if (operation == 8) {
      const auto expected = reference.ExplicitBarrier();
      Check(tracker.BeforeBarrier() == expected, "random explicit barrier");
      tracked_barriers += expected == BarrierDecision::None ? 0u : 1u;
    } else {
      const auto expected = reference.Flush();
      Check(tracker.Flush() == expected, "random flush");
      tracked_barriers += expected == BarrierDecision::None ? 0u : 1u;
    }
  }
  Check(tracked_barriers < legacy_barriers, "tracking emits fewer barriers than the legacy fence");
}

} // namespace

int main() {
  TestHazards();
  TestUnknownFallback();
  TestDeferredOrdering();
  TestRandomizedReference();
  std::puts("BarrierTrackerTests: ok");
  return 0;
}
