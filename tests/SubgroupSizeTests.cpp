#include "graphics/host_gpu/renderer/pipeline/subgroupSize.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

constexpr uint32_t kStageVertex   = 0x00000001u;
constexpr uint32_t kStageTessCtrl = 0x00000002u;
constexpr uint32_t kStageTessEval = 0x00000004u;
constexpr uint32_t kStageFragment = 0x00000010u;
constexpr uint32_t kStageCompute  = 0x00000020u;
constexpr uint32_t kStageTask     = 0x00000040u;
constexpr uint32_t kStageMesh     = 0x00000080u;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SubgroupSizeTests: failed: %s\n", message);
		std::abort();
	}
}

void TestDualWaveCaps() {
	// Device with default=64, min=32, max=64 and requiredSubgroupSize for compute, fragment,
	// task and mesh stages.
	const SubgroupSizeCaps caps {
	    .default_size     = 64u,
	    .min_size         = 32u,
	    .max_size         = 64u,
	    .supported_stages = kStageCompute | kStageFragment | kStageTask | kStageMesh,
	};

	{
		const auto d = ResolveStageSubgroupSize(kStageFragment, 32u, false, true, caps);
		Check(d.apply && d.required_size == 32u && d.expected_size == 32u && !d.mismatch_warning,
		      "wave32 fragment requests a 32-lane subgroup");
	}
	{
		const auto d = ResolveStageSubgroupSize(kStageFragment, 32u, false, false, caps);
		Check(d.apply && d.required_size == 32u && !d.mismatch_warning,
		      "wave32 fragment without subgroup ops");
	}
	{
		const auto d = ResolveStageSubgroupSize(kStageFragment, 64u, false, true, caps);
		Check(d.apply && d.required_size == 64u && !d.mismatch_warning,
		      "wave64 fragment requests a 64-lane subgroup");
	}
	{
		const auto d32 = ResolveStageSubgroupSize(kStageMesh, 32u, false, true, caps);
		const auto d64 = ResolveStageSubgroupSize(kStageMesh, 64u, false, true, caps);
		Check(d32.apply && d32.required_size == 32u && !d32.mismatch_warning, "wave32 mesh");
		Check(d64.apply && d64.required_size == 64u && !d64.mismatch_warning, "wave64 mesh");
	}
	{
		const auto d32 = ResolveStageSubgroupSize(kStageCompute, 32u, false, true, caps);
		const auto d64 = ResolveStageSubgroupSize(kStageCompute, 64u, false, true, caps);
		Check(d32.apply && d32.required_size == 32u && !d32.mismatch_warning, "wave32 compute");
		Check(d64.apply && d64.required_size == 64u && !d64.mismatch_warning, "wave64 compute");
	}
	{
		const auto d_vs = ResolveStageSubgroupSize(kStageVertex, 32u, false, true, caps);
		Check(!d_vs.apply && d_vs.mismatch_warning && d_vs.expected_size == 32u,
		      "wave32 vertex with subgroup ops warns");
		const auto d_vs_noop = ResolveStageSubgroupSize(kStageVertex, 32u, false, false, caps);
		Check(!d_vs_noop.apply && !d_vs_noop.mismatch_warning,
		      "wave32 vertex without subgroup ops stays quiet");
		const auto d_tcs64 = ResolveStageSubgroupSize(kStageTessCtrl, 64u, false, true, caps);
		Check(!d_tcs64.apply && !d_tcs64.mismatch_warning,
		      "wave64 tessellation control matches the host default");
		const auto d_tes32 = ResolveStageSubgroupSize(kStageTessEval, 32u, false, true, caps);
		Check(!d_tes32.apply && d_tes32.mismatch_warning,
		      "wave32 tessellation evaluation with subgroup ops warns");
	}
}

void TestWave32OnlyCaps() {
	const SubgroupSizeCaps caps {
	    .default_size     = 32u,
	    .min_size         = 32u,
	    .max_size         = 32u,
	    .supported_stages = kStageCompute | kStageFragment,
	};

	{
		const auto d_cs = ResolveStageSubgroupSize(kStageCompute, 64u, true, true, caps);
		Check(d_cs.apply && d_cs.required_size == 32u && !d_cs.mismatch_warning,
		      "emulated wave64 compute expects 32");
		const auto d_mesh = ResolveStageSubgroupSize(kStageMesh, 64u, true, true, caps);
		Check(!d_mesh.apply && d_mesh.expected_size == 32u && !d_mesh.mismatch_warning,
		      "emulated wave64 mesh matches the host default");
	}
	{
		const auto d_fs = ResolveStageSubgroupSize(kStageFragment, 64u, false, true, caps);
		Check(!d_fs.apply && d_fs.mismatch_warning && d_fs.expected_size == 64u,
		      "wave64 fragment with subgroup ops warns on a wave32-only device");
		const auto d_fs_noop =
		    ResolveStageSubgroupSize(kStageFragment, 64u, false, false, caps);
		Check(!d_fs_noop.apply && !d_fs_noop.mismatch_warning,
		      "wave64 fragment without subgroup ops stays quiet");
	}
}

void TestNoControlAndInvalidWave() {
	const SubgroupSizeCaps compute_only {
	    .default_size     = 64u,
	    .min_size         = 32u,
	    .max_size         = 64u,
	    .supported_stages = kStageCompute,
	};
	{
		const auto d = ResolveStageSubgroupSize(kStageFragment, 32u, false, true, compute_only);
		Check(!d.apply && d.mismatch_warning && d.expected_size == 32u,
		      "a compute-only device warns on a wave32 fragment with subgroup ops");
	}

	const SubgroupSizeCaps no_control {
	    .default_size     = 64u,
	    .min_size         = 0u,
	    .max_size         = 0u,
	    .supported_stages = 0u,
	};
	{
		const auto d32 = ResolveStageSubgroupSize(kStageFragment, 32u, false, true, no_control);
		Check(!d32.apply && d32.mismatch_warning, "no size control warns on a mismatch");
		const auto d64 = ResolveStageSubgroupSize(kStageFragment, 64u, false, true, no_control);
		Check(!d64.apply && !d64.mismatch_warning, "no size control matches the default");
	}

	{
		const auto d0 = ResolveStageSubgroupSize(kStageFragment, 0u, false, true, compute_only);
		Check(!d0.apply && !d0.mismatch_warning && d0.expected_size == 0u,
		      "an unknown guest wave size is ignored");
	}
}

} // namespace

int main() {
	TestDualWaveCaps();
	TestWave32OnlyCaps();
	TestNoControlAndInvalidWave();
	std::puts("SubgroupSizeTests: ok");
	return 0;
}
