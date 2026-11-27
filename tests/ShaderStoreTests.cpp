#include "graphics/host_gpu/renderer/pipeline/shaderStoreLock.h"
#include "graphics/host_gpu/renderer/pipeline/shaderStoreRestamp.h"

#include <filesystem>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <unordered_set>
#include <vector>

// Partial restamp (PipelineCache::Restamp): which records of the old shader store stay; and the
// single-writer lock of a store (ShaderStoreLock).

namespace {

using Libs::Graphics::RestampRecord;
using Libs::Graphics::SelectRestampRecords;
using Type = RestampRecord::Type;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "ShaderStoreTests: failed: %s\n", message);
    std::abort();
  }
}

// Each module its own program unless one is given.
RestampRecord Module(uint64_t id, uint64_t program = 0) {
  return {.type = Type::Module, .module_id = id, .program = program != 0 ? program : 1000 + id};
}

// Plans identical for every module unless a test says otherwise.
Libs::Graphics::RestampSelection Select(const std::vector<RestampRecord> &records,
                                        const std::unordered_set<uint64_t> &identical) {
  return SelectRestampRecords(records, identical, identical);
}

RestampRecord Pipeline(uint64_t vs, uint64_t hs, uint64_t es, uint64_t ps) {
  return {.type = Type::GraphicsPipeline, .stage_ids = {vs, hs, es, ps}};
}

RestampRecord Layout() { return {.type = Type::TessellationLayout}; }

void TestKeepsOnlyIdenticalModules() {
  const std::vector<RestampRecord> records = {Module(1), Module(2), Module(3)};
  const auto selection = Select(records, {1, 3});
  Check(selection.keep == std::vector<bool>({true, false, true}), "identical modules kept, the other dropped");
  Check(selection.modules_kept == 2 && selection.modules_dropped == 1, "module counts");
}

void TestPipelinesFollowTheirStages() {
  const std::vector<RestampRecord> records = {
      Module(10), Module(20), Module(30),
      Pipeline(10, 0, 0, 20),   // both stages kept
      Pipeline(10, 0, 0, 30),   // pixel stage dropped
      Pipeline(10, 0, 0, 0),    // no pixel stage: depth only
      Pipeline(10, 40, 50, 20), // tessellation stages the store never had
  };
  const auto selection = Select(records, {10, 20, 40, 50});
  Check(selection.keep[3], "pipeline with kept stages stays");
  Check(!selection.keep[4], "pipeline with a dropped stage goes");
  Check(selection.keep[5], "absent stages (0) do not drop a pipeline");
  Check(!selection.keep[6], "a stage verified but absent from the store does not count as kept");
  Check(selection.pipelines_kept == 2 && selection.pipelines_dropped == 2, "pipeline counts");
}

void TestModuleOrderDoesNotMatter() {
  // A pipeline record may come before the modules it names (records are appended as created).
  const std::vector<RestampRecord> records = {Pipeline(7, 0, 0, 8), Module(7), Module(8)};
  const auto selection = Select(records, {7, 8});
  Check(selection.keep == std::vector<bool>({true, true, true}), "pipeline before its modules is kept");
}

void TestDuplicatesAndLayouts() {
  const std::vector<RestampRecord> records = {Module(5), Module(5), Layout(), Module(6), Layout()};
  const auto selection = Select(records, {5, 6});
  Check(selection.keep == std::vector<bool>({true, false, false, true, false}),
        "only the first record of an id stays; layouts are dropped");
  Check(selection.modules_kept == 2 && selection.modules_duplicate == 1 && selection.layouts_dropped == 2,
        "duplicate and layout counts");
}

void TestNothingIdentical() {
  const std::vector<RestampRecord> records = {Module(1), Pipeline(1, 0, 0, 0), Module(0)};
  const auto selection = Select(records, {0});
  Check(selection.modules_kept == 0 && selection.pipelines_kept == 0, "nothing kept");
  Check(selection.keep == std::vector<bool>({false, false, false}), "id 0 is never a module");
}

// A load reads a program's plan from its first record only: that one must have the extracted plan.
void TestPlanOfTheFirstKeptRecord() {
  const std::vector<RestampRecord> records = {
      Module(1, 7), Module(2, 7), Module(3, 7), // program 7: plans of 1 and 3 differ
      Module(4, 8), Module(5, 8),               // program 8: first plan identical, second not
      Module(6, 9), Module(9, 9),               // program 9: module 6 differs, 9's plan differs
  };
  const auto selection = SelectRestampRecords(records, {1, 2, 3, 4, 5, 9}, {2, 4});
  Check(selection.keep == std::vector<bool>({false, true, true, true, true, false, false}),
        "records before a program's first identical plan are dropped, later ones kept");
  Check(selection.modules_plan_dropped == 2 && selection.modules_dropped == 1 &&
            selection.modules_kept == 4,
        "plan counts");
}

void TestEmpty() {
  const auto selection = SelectRestampRecords({}, {}, {});
  Check(selection.keep.empty() && selection.modules_kept == 0, "empty store");
}

// A game and a precompile of one stamp must not both write its store.
void TestStoreLockSingleWriter() {
  using Libs::Graphics::ShaderStoreLock;
  const auto dir = std::filesystem::temp_directory_path() / "kyty_shader_store_lock_test";
  std::filesystem::create_directories(dir);
  const auto store = dir / "TEST.shaders";
  {
    ShaderStoreLock first;
    ShaderStoreLock second;
    Check(first.Acquire(store), "first writer gets the lock");
    Check(first.Acquire(store), "acquiring again is a no-op");
    Check(!second.Acquire(store), "a second writer is refused while the first holds it");
    Check(!second.Held(), "a refused lock is not held");
    ShaderStoreLock other_file;
    Check(other_file.Acquire(dir / "OTHER.shaders"), "another store has its own lock");
#ifndef _WIN32
    // Another process is refused too (what a game started during a precompile is).
    const pid_t child = fork();
    if (child == 0) {
      ShaderStoreLock in_child;
      _exit(in_child.Acquire(store) ? 1 : 0);
    }
    int status = 0;
    waitpid(child, &status, 0);
    Check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "another process is refused");
#endif
    first.Release();
    Check(second.Acquire(store), "the lock is free once released");
  }
  {
    ShaderStoreLock again;
    Check(again.Acquire(store), "destruction releases the lock");
  }
  std::error_code error;
  std::filesystem::remove_all(dir, error);
}

} // namespace

int main() {
  TestStoreLockSingleWriter();
  TestKeepsOnlyIdenticalModules();
  TestPipelinesFollowTheirStages();
  TestModuleOrderDoesNotMatter();
  TestDuplicatesAndLayouts();
  TestNothingIdentical();
  TestPlanOfTheFirstKeptRecord();
  TestEmpty();
  std::printf("ShaderStoreTests: all passed\n");
  return 0;
}
