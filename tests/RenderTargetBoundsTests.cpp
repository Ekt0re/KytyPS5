#include "graphics/host_gpu/renderer/renderTarget.h"

#include <gtest/gtest.h>

namespace Libs::Graphics {
namespace {

TEST(RenderTargetBounds, ClampsAreaToAllocatedMipExtent) {
	EXPECT_EQ(ClampRenderArea({512, 512}, {400, 512, 1}, 0), (vk::Extent2D {400, 512}));
	EXPECT_EQ(ClampRenderArea({512, 256}, {1024, 512, 1}, 1), (vk::Extent2D {512, 256}));
	EXPECT_EQ(ClampRenderArea({512, 256}, {1024, 512, 1}, 2), (vk::Extent2D {256, 128}));
}

TEST(RenderTargetBounds, KeepsAreaInsideAllocationUnchanged) {
	EXPECT_EQ(ClampRenderArea({320, 240}, {400, 512, 1}, 0), (vk::Extent2D {320, 240}));
	EXPECT_EQ(ClampRenderArea({1, 1}, {1, 1, 1}, 8), (vk::Extent2D {1, 1}));
}

TEST(RenderTargetBounds, RejectsInvalidMipLevel) {
	EXPECT_EQ(ClampRenderArea({320, 240}, {400, 512, 1}, 32), (vk::Extent2D {0, 0}));
}

TEST(RenderTargetBounds, ClampsLayersToRemainingAllocation) {
	EXPECT_EQ(ClampRenderLayers(1, 4, 4), 3u);
	EXPECT_EQ(ClampRenderLayers(1, 3, 4), 3u);
	EXPECT_EQ(ClampRenderLayers(0, 4, 4), 4u);
}

TEST(RenderTargetBounds, RejectsExhaustedLayerRange) {
	EXPECT_EQ(ClampRenderLayers(4, 1, 4), 0u);
	EXPECT_EQ(ClampRenderLayers(5, 1, 4), 0u);
}

} // namespace
} // namespace Libs::Graphics

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
