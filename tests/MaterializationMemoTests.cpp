#include "graphics/host_gpu/renderer/pipeline/materializationMemo.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>

namespace {

using namespace Libs::Graphics;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MaterializationMemoTests: failed: %s\n", text);
    std::abort();
  }
}

constexpr uint64_t PageSize = 4096;

struct Memory {
  std::map<uint64_t, std::vector<uint32_t>> pages;
  uint32_t reads = 0;
  bool readable = true;

  void Write(uint64_t address, uint32_t value) {
    auto &page = pages[address / PageSize];
    page.resize(PageSize / 4, 0u);
    page[(address % PageSize) / 4] = value;
  }
  static bool Read(void *user, uint64_t address, std::span<uint32_t> values) {
    auto &self = *static_cast<Memory *>(user);
    ++self.reads;
    if (!self.readable) return false;
    for (size_t i = 0; i < values.size(); ++i) {
      const auto at = address + i * 4;
      const auto found = self.pages.find(at / PageSize);
      values[i] = found == self.pages.end() ? 0u : found->second[(at % PageSize) / 4];
    }
    return true;
  }
};

class Stamps final : public IR::MemoryChangeSource {
public:
  struct State {
    uint64_t generation = 1;
    bool trackable = true;
  };
  std::map<std::pair<uint64_t, uint64_t>, State> states;
  std::set<uint64_t> untrackable;
  uint32_t tracks = 0;

  std::shared_ptr<IR::MemoryChangeStamp> Track(uint64_t address, uint64_t size,
                                               uint32_t max_pages) override {
    ++tracks;
    if (size == 0) return std::make_shared<AlwaysUnchanged>();
    const auto pages = (address + size - 1u) / PageSize - address / PageSize + 1u;
    if (max_pages == 0 || pages > max_pages) return {};
    for (auto page = address / PageSize; page <= (address + size - 1u) / PageSize; ++page) {
      if (untrackable.contains(page)) return {};
    }
    auto &state = states[{address, size}];
    if (!state.trackable) return {};
    return std::make_shared<Stamp>(state);
  }
  void Written(uint64_t address) {
    for (auto &[range, state] : states) {
      if (range.first <= address && address < range.first + range.second) ++state.generation;
    }
  }

private:
  class AlwaysUnchanged final : public IR::MemoryChangeStamp {
    bool Unchanged() const override { return true; }
  };
  class Stamp final : public IR::MemoryChangeStamp {
  public:
    explicit Stamp(State &state) : m_state(state), m_generation(state.generation) {}
    bool Unchanged() const override { return m_state.generation == m_generation; }
  private:
    State &m_state;
    uint64_t m_generation;
  };
};

struct Heap final : IR::ImageHeapSource {
  uint64_t slots = 0x1000;
  uint32_t slot_count = 8;
  std::vector<IR::DescriptorValue> kinds{IR::DescriptorValue{.dwords = {1}, .dword_count = 8}};
  uint32_t acquires = 0;
  bool Acquire(const IR::ImageHeapRequest &, IR::ImageHeapView &view) override {
    ++acquires;
    view = {.slots = slots, .slot_count = slot_count, .kinds = kinds};
    return true;
  }
};

struct Inputs {
  std::vector<uint32_t> user_data{1, 2, 3, 4};
  std::vector<uint32_t> workgroups;
  uint64_t base = 0x40000;
  uint64_t indirect = 0;
  IR::ImageHeapSource *heap = nullptr;
  Memory *memory = nullptr;

  IR::SrtRuntime Runtime() const {
    IR::SrtRuntime runtime;
    runtime.user_data = user_data;
    runtime.shader_base = base;
    runtime.userdata = memory;
    runtime.read_specialization_memory = Memory::Read;
    runtime.workgroup_counts = workgroups;
    runtime.indirect_args_address = indirect;
    runtime.image_heap = heap;
    return runtime;
  }
};

IR::ResourceSnapshot Snapshot(std::vector<std::pair<uint64_t, uint64_t>> reads) {
  IR::ResourceSnapshot snapshot;
  snapshot.specialization_reads = std::move(reads);
  snapshot.flattened_srt = {7, 8, 9};
  return snapshot;
}

MemoOutcome Lookup(MemoizedMaterialization &memo, const Inputs &inputs, MemoLookupStats &stats) {
  MemoOutcome outcome = MemoOutcome::Count;
  stats = {};
  const auto *entry = memo.Lookup(inputs.Runtime(), outcome, stats);
  Check((entry != nullptr) == (outcome == MemoOutcome::Hit), "a hit returns its entry, a miss none");
  return outcome;
}

const void *Store(MemoizedMaterialization &memo, const Inputs &inputs,
                  std::vector<std::pair<uint64_t, uint64_t>> reads, uint32_t *evicted = nullptr,
                  bool record_heap = false, const WalkReadHashes *walk = nullptr,
                  MemoLookupStats *stats_out = nullptr) {
  RecordingImageHeap recorder(inputs.heap);
  if (record_heap) {
    IR::ImageHeapView view;
    const IR::ImageHeapRequest request{.base = 0x9000, .size = 64, .step = 4, .count = 16};
    Check(recorder.Acquire(request, view), "heap acquire");
  }
  uint32_t dropped = 0;
  MemoLookupStats stats;
  const auto *entry = memo.Store(inputs.Runtime(), Snapshot(std::move(reads)),
                                 IR::ResourceSpecialization{}, recorder,
                                 evicted != nullptr ? *evicted : dropped, stats, walk);
  if (stats_out != nullptr) *stats_out = stats;
  return entry;
}

constexpr uint64_t Table = 0x10000;
constexpr uint64_t Block = 0x20000;
constexpr uint64_t Spans = 0x30f00;
const std::vector<std::pair<uint64_t, uint64_t>> Reads{{Table, 64}, {Block, 32}, {Spans, 0x300}};

void FillMemory(Memory &memory) {
  memory.Write(Table, 11);
  memory.Write(Block, 22);
  memory.Write(Spans, 33);
  memory.Write(Spans + 0x200, 44);
}

void TestHitAndRanges() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, nullptr, 0);
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Empty, "nothing stored yet");
  Check(Store(memo, inputs, Reads) != nullptr, "stored");
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Hit, "same inputs and bytes hit");
  Check(stats.ranges_compared == 3 && stats.ranges_stamped == 0,
        "without stamps every range is compared");
  memory.Write(Block, 23);
  Check(Lookup(memo, inputs, stats) == MemoOutcome::RangeChanged, "changed bytes miss");
}

void TestStamps() {
  Memory memory;
  FillMemory(memory);
  Stamps stamps;
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, &stamps, 4);
  Check(Store(memo, inputs, Reads) != nullptr, "stored with stamps");
  const auto after_store = memory.reads;
  MemoLookupStats stats;
  for (int i = 0; i < 5; ++i) {
    Check(Lookup(memo, inputs, stats) == MemoOutcome::Hit, "unchanged stamp hits");
    Check(stats.ranges_stamped == 3 && stats.ranges_compared == 0,
          "every range validated by its stamp");
  }
  Check(memory.reads == after_store, "stamped hits read no guest memory");
  memory.Write(Table, 11);
  stamps.Written(Table);
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Hit,
        "an equal rewrite is compared and remains a hit");
  Check(stats.ranges_compared == 1 && stats.ranges_stamped == 2,
        "only the written range is compared");
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Hit && stats.ranges_stamped == 3,
        "the equal rewrite is stamped again");
  memory.Write(Table, 12);
  stamps.Written(Table);
  Check(Lookup(memo, inputs, stats) == MemoOutcome::RangeChanged, "a changed rewrite misses");
}

void TestStampFallback() {
  Memory memory;
  FillMemory(memory);
  Stamps stamps;
  stamps.untrackable.insert(Block / PageSize);
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, &stamps, 4);
  Store(memo, inputs, Reads);
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Hit && stats.ranges_stamped == 2 &&
            stats.ranges_compared == 1,
        "an untrackable range is compared while the others are stamped");
  memory.Write(Block, 99);
  Check(Lookup(memo, inputs, stats) == MemoOutcome::RangeChanged,
        "a change in an untrackable range is found");

  stamps.untrackable.clear();
  MemoizedMaterialization capped;
  capped.Configure(1, &stamps, 1);
  Store(capped, inputs, Reads);
  Check(Lookup(capped, inputs, stats) == MemoOutcome::Hit && stats.ranges_stamped == 2 &&
            stats.ranges_compared == 1,
        "only the range over the page cap falls back to comparison");
}

void TestKeysAndEntries() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(1, nullptr, 0);
  Store(memo, inputs, Reads);
  MemoLookupStats stats;
  auto changed = inputs;
  changed.user_data[2] = 77;
  Check(Lookup(memo, changed, stats) == MemoOutcome::UserData, "user data key");
  changed = inputs;
  changed.base = 0x50000;
  Check(Lookup(memo, changed, stats) == MemoOutcome::ShaderBase, "shader base key");
  changed = inputs;
  changed.indirect = 0x8000;
  Check(Lookup(memo, changed, stats) == MemoOutcome::IndirectArgs, "indirect argument key");
  changed = inputs;
  changed.workgroups = {1, 2, 3};
  Check(Lookup(memo, changed, stats) == MemoOutcome::WorkgroupCounts, "workgroup key");
  Check(stats.ranges_compared == 0, "key misses read no ranges");

  Inputs other = inputs;
  other.user_data[0] = 100;
  MemoizedMaterialization several;
  several.Configure(4, nullptr, 0);
  Store(several, inputs, Reads);
  Store(several, other, Reads);
  for (int i = 0; i < 4; ++i) {
    Check(Lookup(several, inputs, stats) == MemoOutcome::Hit, "first entry hits");
    Check(Lookup(several, other, stats) == MemoOutcome::Hit, "second entry hits");
  }
  Check(several.Size() == 2, "both entries are retained");
  uint32_t evicted = 0;
  Store(several, other, Reads, &evicted);
  Check(evicted == 0, "a full memo does not evict below capacity");
}

void TestHeapReplay() {
  Memory memory;
  FillMemory(memory);
  Heap heap;
  Inputs inputs{.heap = &heap, .memory = &memory};
  const std::vector<std::pair<uint64_t, uint64_t>> reads{{Table, 64}, {0x9000, 64}};
  MemoizedMaterialization memo;
  memo.Configure(1, nullptr, 0);
  Check(Store(memo, inputs, reads, nullptr, true) != nullptr, "one entry can keep a heap plan");
  heap.acquires = 0;
  memory.reads = 0;
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Hit, "heap plan hits");
  Check(heap.acquires == 1, "a hit repeats the heap acquisition");
  Check(stats.ranges_compared == 1, "the heap table itself is not hashed");
  memory.Write(0x9000, 5);
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Hit,
        "heap content is validated by the heap source");
  heap.slots = 0x2000;
  Check(Lookup(memo, inputs, stats) == MemoOutcome::HeapChanged, "other slots miss");
  heap.slots = 0x1000;
  heap.kinds.push_back({});
  Check(Lookup(memo, inputs, stats) == MemoOutcome::HeapChanged, "other kinds miss");
}

void TestUnreadableAndClear() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  MemoizedMaterialization memo;
  memo.Configure(2, nullptr, 0);
  memory.readable = false;
  Check(Store(memo, inputs, Reads) == nullptr, "an unreadable range cannot be stored");
  memory.readable = true;
  Check(Store(memo, inputs, Reads) != nullptr, "stored after memory becomes readable");
  memory.readable = false;
  MemoLookupStats stats;
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Unreadable, "unreadable lookup misses");
  memory.readable = true;
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Hit,
        "a failed lookup does not replace a valid stamp");
  const std::vector<std::pair<uint64_t, uint64_t>> odd{{Table, 6}};
  Check(Store(memo, inputs, odd) == nullptr, "non-dword ranges cannot be stored");
  memo.Clear();
  Check(Lookup(memo, inputs, stats) == MemoOutcome::Empty, "clear drops every entry");
}

void TestWalkStoreRace() {
  Memory memory;
  FillMemory(memory);
  Inputs inputs{.memory = &memory};
  const std::vector<std::pair<uint64_t, uint64_t>> reads{{Table, 64}, {Block, 32}};
  const auto walk_and_store = [&](bool write_after, bool write_during, bool observe_all,
                                  bool &changed, bool &unobserved) {
    MemoizedMaterialization memo;
    memo.Configure(2, nullptr, 0);
    WalkReadHashes hashes;
    auto walk = hashes.Wrap(inputs.Runtime());
    for (const auto &[address, size] : reads) {
      if (!observe_all && address == Block) continue;
      std::vector<uint32_t> data(size / 4);
      Check(walk.read_specialization_memory(walk.userdata, address, data), "walk read");
      if (write_during) {
        memory.Write(address, 1000u);
        Check(walk.read_specialization_memory(walk.userdata, address, data), "walk read again");
      }
    }
    if (write_after) memory.Write(Block, 77);
    MemoLookupStats stats;
    const auto *entry = Store(memo, inputs, reads, nullptr, false, &hashes, &stats);
    changed = stats.walk_changed;
    unobserved = stats.walk_unobserved;
    return entry != nullptr;
  };
  bool changed = false;
  bool unobserved = false;
  Check(walk_and_store(false, false, true, changed, unobserved) && !changed && !unobserved,
        "stable walk bytes are stored");
  Check(!walk_and_store(true, false, true, changed, unobserved) && changed,
        "a write after the walk is rejected");
  FillMemory(memory);
  Check(!walk_and_store(false, true, true, changed, unobserved) && changed,
        "a range that changes during the walk is rejected");
  FillMemory(memory);
  Check(!walk_and_store(false, false, false, changed, unobserved) && unobserved,
        "an unobserved range is not stored");
}

void TestStoreGate() {
  using Gate = MemoizedMaterialization::StoreGate;
  Gate busy;
  for (uint32_t i = 0; i < Gate::Window; ++i) {
    Check(busy.Allows(), "stores while measuring");
    busy.Note(i % 8 == 0);
  }
  Check(busy.Allows(), "enough hits keep storing");
  Gate missing;
  for (uint32_t i = 0; i < Gate::Window - 1; ++i) missing.Note(i < Gate::MinHits - 1);
  missing.Note(false);
  Check(!missing.Allows(), "too few hits start a cooldown");
  for (uint32_t i = 0; i < Gate::Cooldown - 1; ++i) missing.Note(false);
  Check(!missing.Allows(), "cooldown is still active");
  missing.Note(false);
  Check(missing.Allows(), "measurement resumes after cooldown");
}

} // namespace

int main() {
  TestStoreGate();
  TestHitAndRanges();
  TestStamps();
  TestStampFallback();
  TestKeysAndEntries();
  TestHeapReplay();
  TestUnreadableAndClear();
  TestWalkStoreRace();
  std::puts("MaterializationMemoTests: ok");
  return 0;
}
