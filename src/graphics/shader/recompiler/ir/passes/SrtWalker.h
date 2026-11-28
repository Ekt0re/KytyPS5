#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <memory>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR {

using SrtMemoryReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

// Image descriptors at base + first + k * step, k < count, within size bytes that a sampled
// root selects by GPU-computed byte offset. The host keeps them resident instead of
// materializing them.
struct ImageHeapRequest {
	uint64_t             base  = 0;
	uint64_t             size  = 0;
	uint64_t             step  = 0;
	uint64_t             first = 0;
	uint64_t             count = 0;
	const ImageResource* root  = nullptr;
};

struct ImageHeapView {
	uint64_t                         slots      = 0; // Device address of one slot per heap dword.
	uint32_t                         slot_count = 0; // Keys past it select a trailing null slot.
	std::span<const DescriptorValue> kinds; // One representative descriptor per kind.
};

class ImageHeapSource {
public:
	// False when the host does not keep the table resident.
	virtual bool Acquire(const ImageHeapRequest& request, ImageHeapView& view) = 0;

protected:
	~ImageHeapSource() = default;
};

struct HeapRowView {
	uint32_t              first = UINT32_MAX; // Report bit of row 0.
	std::vector<uint32_t> rows;               // Rows the GPU reported.
};

class HeapRowSource {
public:
	// The reported rows of a root that selects among count rows. False when the root cannot report.
	virtual bool Acquire(uint64_t shader, uint32_t source, uint64_t count, HeapRowView& view) = 0;
	// A reported row that cannot be bound.
	virtual void Reject(uint64_t shader, uint32_t source, uint32_t row, const DescriptorValue& value,
	                    std::string_view reason) = 0;

protected:
	~HeapRowSource() = default;
};

struct SrtRuntime {
	std::span<const uint32_t> user_data;
	uint64_t                  shader_base                = 0;
	void*                     userdata                   = nullptr;
	SrtMemoryReader           read_specialization_memory = nullptr;
	std::span<const uint32_t> workgroup_counts;
	uint64_t                  indirect_args_address = 0;
	ImageHeapSource*          image_heap            = nullptr;
	HeapRowSource*            heap_rows             = nullptr;
};

class SrtReadCapture {
public:
	SrtReadCapture(SrtRuntime source, std::vector<std::pair<uint64_t, uint64_t>>& ranges):
	    m_source(source), m_ranges(ranges) {}
	SrtRuntime ObservedRuntime();

private:
	static bool Read(void* userdata, uint64_t address, std::span<uint32_t> values);

	SrtRuntime                                  m_source;
	std::vector<std::pair<uint64_t, uint64_t>>& m_ranges;
};

// Walker loads from pages the host made inaccessible (they share a page with GPU-written bytes)
// used to fault once per load and per walk; the fault handler then served the load from the
// backing. With a guard installed the walker asks `counts` (one uint16 per page slot, nonzero
// when a page of the slot may be inaccessible; see ReadWatchFilter) before each load and calls
// `load` instead of faulting. `load` returns false when it cannot serve the address, and the
// plain load (with its fault handling) runs as before. Without a guard nothing changes.
using SrtGuardedLoad = bool (*)(uint64_t address, uint32_t& value);
void SetSrtWalkerGuard(const void* counts, uint64_t slot_count, SrtGuardedLoad load);
enum class SrtWalkerGuardMode : uint8_t { Off, On, Verify };
// Verify walks twice, guarded and plain, compares everything the walk wrote and counts
// differences; the plain walk's results are used.
void SetSrtWalkerGuardMode(SrtWalkerGuardMode mode);
struct SrtWalkerGuardStats {
	uint64_t walks      = 0;
	uint64_t served     = 0; // Loads the guard served without a fault.
	uint64_t declined   = 0; // Loads it passed on to the plain load.
	uint64_t verified   = 0; // Walks compared in verify mode.
	uint64_t mismatches = 0;
};
// Counts walks and served/passed-on loads; off by default (verified walks and mismatches are
// always counted).
void SetSrtWalkerGuardCounting(bool counting);
SrtWalkerGuardStats GetSrtWalkerGuardStats();

// Walk state a thread supplies in place of a plan's own (ResourcePlan's mutable scratch), so that
// a thread other than the command thread can walk and materialize a plan the command thread uses
// at the same time (lookahead). While a ScopedWalkScratch is alive on a thread, every walk and
// materialization on that thread uses `scratch` for every plan; Prepare() sizes it for one plan
// (call it whenever the plan changes). `loads` receives the (address, value) of every guest load
// of the walks, as the walker saw them.
struct WalkScratch {
	std::vector<uint64_t>                   srt_scratch;
	std::vector<uint32_t>                   material_keys;
	std::vector<ResourcePlan::SelectorKeys> selector_keys;
	std::vector<ScanCacheEntry>             scan_cache;
	std::vector<std::pair<uint64_t, uint32_t>> loads;
	const ResourcePlan*                     prepared = nullptr;

	void Prepare(const ResourcePlan& plan);
};

class ScopedWalkScratch {
public:
	explicit ScopedWalkScratch(WalkScratch& scratch);
	~ScopedWalkScratch();
	ScopedWalkScratch(const ScopedWalkScratch&)            = delete;
	ScopedWalkScratch& operator=(const ScopedWalkScratch&) = delete;

private:
	WalkScratch* m_previous;
};

// The scratch of plan that walks on this thread use.
std::vector<uint64_t>&                   PlanSrtScratch(const ResourcePlan& plan);
std::vector<uint32_t>&                   PlanMaterialKeys(const ResourcePlan& plan);
std::vector<ResourcePlan::SelectorKeys>& PlanSelectorKeys(const ResourcePlan& plan);
std::vector<ScanCacheEntry>&             PlanScanCache(const ResourcePlan& plan);

// A thread that walks ahead of the command thread (lookahead). Its walks must not touch host
// state the command thread owns: walker loads that would fault are completed from the backing
// (the fault handler asks SrtSpeculativeThread()), materialization failures are not logged and
// its time is not counted as the command thread's.
// Whether walkers compiled from now on capture each load's value (needed by WalkScratch::loads,
// so by lookahead). Default: on when KYTY_LOOKAHEAD is set, else off (no extra work per load).
// Set it before any plan is compiled.
void SetSrtWalkerValueCapture(bool capture);
void SetSrtSpeculativeThread(bool speculative);
bool SrtSpeculativeThread();

enum class RuntimeValueType { Any, Integer, ImmutableInteger };

bool ValidateRuntimeValue(const ResourcePlan& program, Value value,
                          RuntimeValueType type = RuntimeValueType::Any);
// ValidateRuntimeValue for many values of one unchanged program: the proofs of shared
// subexpressions are kept between calls instead of being rebuilt for every value.
class RuntimeValueValidator {
public:
	explicit RuntimeValueValidator(const ResourcePlan& program);
	~RuntimeValueValidator();
	RuntimeValueValidator(const RuntimeValueValidator&)            = delete;
	RuntimeValueValidator& operator=(const RuntimeValueValidator&) = delete;

	bool Validate(Value value, RuntimeValueType type = RuntimeValueType::Any);

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};
// Operations whose result is uniform when their operands are uniform.
bool IsRuntimeUniformOp(ValueOpcode opcode);
// Uniform across a compute wave, including values derived from its workgroup ID.
// This does not imply that the host can evaluate the value.
bool ValidateWaveUniformValue(const ResourcePlan& program, Value value);
// Compiles the plan's x86 walker. Without values it walks the plan's resources; otherwise it
// only evaluates the values, with every block reachable.
void CompileSrtWalker(ResourcePlan& plan, std::span<const Value> values = {});
// Runs the walker. flat receives the scalar reads of reachable blocks and reads, when non-null,
// one DWORD range per guest load. False when a reachable read is unknown. With a WalkScratch in
// place, its `loads` also receive each guest load's address and value.
bool WalkSrt(const ResourcePlan& plan, const SrtRuntime& runtime, std::vector<uint32_t>& flat,
             std::vector<std::pair<uint64_t, uint64_t>>* reads);
// Results of the last walk; false when unknown.
bool WalkedValue(const ResourcePlan& plan, Value value, uint32_t& result);
bool WalkedDescriptor(const ResourcePlan& plan, uint32_t source, DescriptorValue& result);
bool WalkedActive(const ResourcePlan& plan, uint32_t source);
bool WalkedImageActive(const ResourcePlan& plan, uint32_t image);
// Length of the guest load at rip if it belongs to the walker running on this thread, else 0.
// The exception handler completes such a load from the backing, else as a zero read.
uint32_t SrtWalkerLoadLength(uint64_t rip);

// Nonnegative affine coefficients: element zero is constant, remaining elements are
// caller-selected leaves. Runtime range proofs must additionally exclude U32 wrap.
template <size_t Size>
std::optional<std::array<uint64_t, Size>> AffineIndex(
    Value value, const auto& resolve, const auto& leaf, uint32_t depth = 0) {
	value = resolve(value);
	if (depth > 32u || value.GetType() != Type::U32) return {};
	std::array<uint64_t, Size> result {};
	if (value.IsImmediate()) {
		result[0] = value.U32();
		return result;
	}
	if (const auto index = leaf(value)) {
		result[*index] = 1u;
		return result;
	}
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	const auto op = inst->GetOpcode();
	if (op != ValueOpcode::IAdd32 && op != ValueOpcode::IMul32 &&
	    op != ValueOpcode::ShiftLeftLogical32) return {};
	auto left = AffineIndex<Size>(inst->Arg(0), resolve, leaf, depth + 1u);
	auto right = AffineIndex<Size>(inst->Arg(1), resolve, leaf, depth + 1u);
	if (!left || !right) return {};
	const auto constant = [](const auto& coefficients) {
		return std::ranges::all_of(std::span(coefficients).subspan(1u),
		                          [](uint64_t coefficient) { return coefficient == 0u; });
	};
	if (op == ValueOpcode::IMul32 && !constant(*right)) std::swap(left, right);
	if (op != ValueOpcode::IAdd32 && !constant(*right)) return {};
	if (op == ValueOpcode::ShiftLeftLogical32) {
		if ((*right)[0] >= 32u) return {};
		(*right)[0] = uint64_t {1} << (*right)[0];
	}
	for (size_t index = 0; index < Size; ++index) {
		(*left)[index] = op == ValueOpcode::IAdd32
		    ? (*left)[index] + (*right)[index] : (*left)[index] * (*right)[0];
		if ((*left)[index] > UINT32_MAX) return {};
	}
	return left;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTWALKER_H_ */
