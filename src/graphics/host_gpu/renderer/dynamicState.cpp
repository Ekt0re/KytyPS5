#include "graphics/host_gpu/renderer/dynamicState.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/renderFix.h"

namespace Libs::Graphics {

// KYTY_DYNAMIC_STATE_ALWAYS (default 1, toggle dynamic_state_always): set the depth compare op,
// depth bounds, depth bias and stencil state of a draw whether or not its test or bias is
// enabled. Pipelines declare them all dynamic, and Vulkan requires every dynamic state of the
// bound pipeline to be set in the command buffer before a draw (VUID-vkCmdDraw-None-07848);
// with 0 they are only set while the test or bias that uses them is on, as before.
static RenderFix DynamicStateAlways {"dynamic state of disabled tests",
                                     "KYTY_DYNAMIC_STATE_ALWAYS", "dynamic_state_always",
                                     true};

void DynamicState::Commit(const GraphicContext& graphics, vk::CommandBuffer cmdbuf) {
	// Invalidate() marks every state dirty at the start of a command buffer, so a dirty state of
	// a disabled test is one this command buffer has not set yet, or one whose value changed.
	const bool unset = (!depth_test_enabled && dirty_state.depth_compare_op) ||
	                   (!depth_bounds_test_enabled && dirty_state.depth_bounds) ||
	                   (!depth_bias_enabled && dirty_state.depth_bias) ||
	                   (!stencil_test_enabled &&
	                    (dirty_state.stencil_front_ops || dirty_state.stencil_back_ops ||
	                     dirty_state.stencil_front_reference || dirty_state.stencil_back_reference ||
	                     dirty_state.stencil_front_write_mask || dirty_state.stencil_back_write_mask ||
	                     dirty_state.stencil_front_compare_mask ||
	                     dirty_state.stencil_back_compare_mask));
	const bool always = unset && DynamicStateAlways.Enabled();
	if (unset) {
		DynamicStateAlways.Hit("depth test %d, depth bounds %d, depth bias %d, stencil test %d",
		                       depth_test_enabled ? 1 : 0, depth_bounds_test_enabled ? 1 : 0,
		                       depth_bias_enabled ? 1 : 0, stencil_test_enabled ? 1 : 0);
	}
	if (dirty_state.viewports) {
		dirty_state.viewports = false;
		cmdbuf.setViewportWithCount(num_viewports, viewports.data());
	}
	if (dirty_state.scissors) {
		dirty_state.scissors = false;
		cmdbuf.setScissorWithCount(num_scissors, scissors.data());
	}
	if (dirty_state.depth_test_enabled) {
		dirty_state.depth_test_enabled = false;
		cmdbuf.setDepthTestEnable(depth_test_enabled);
	}
	if (dirty_state.depth_write_enabled) {
		dirty_state.depth_write_enabled = false;
		// Note that this must be set in a command buffer even if depth test is disabled.
		cmdbuf.setDepthWriteEnable(depth_write_enabled);
	}
	if ((depth_test_enabled || always) && dirty_state.depth_compare_op) {
		dirty_state.depth_compare_op = false;
		cmdbuf.setDepthCompareOp(depth_compare_op);
	}
	if (dirty_state.depth_bounds_test_enabled) {
		dirty_state.depth_bounds_test_enabled = false;
#if !defined(__APPLE__)
		cmdbuf.setDepthBoundsTestEnable(depth_bounds_test_enabled);
#endif
	}
	if ((depth_bounds_test_enabled || always) && dirty_state.depth_bounds) {
		dirty_state.depth_bounds = false;
#if !defined(__APPLE__)
		cmdbuf.setDepthBounds(depth_bounds_min, depth_bounds_max);
#endif
	}
	if (dirty_state.depth_bias_enabled) {
		dirty_state.depth_bias_enabled = false;
		cmdbuf.setDepthBiasEnable(depth_bias_enabled);
	}
	if ((depth_bias_enabled || always) && dirty_state.depth_bias) {
		dirty_state.depth_bias = false;
		cmdbuf.setDepthBias(depth_bias_constant, depth_bias_clamp, depth_bias_slope);
	}
	if (dirty_state.stencil_test_enabled) {
		dirty_state.stencil_test_enabled = false;
		cmdbuf.setStencilTestEnable(stencil_test_enabled);
	}
	if (stencil_test_enabled || always) {
		if (dirty_state.stencil_front_ops && dirty_state.stencil_back_ops &&
		    stencil_front_ops == stencil_back_ops) {
			dirty_state.stencil_front_ops = false;
			dirty_state.stencil_back_ops  = false;
			cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFrontAndBack, stencil_front_ops.fail_op,
			                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
			                    stencil_front_ops.compare_op);
		} else {
			if (dirty_state.stencil_front_ops) {
				dirty_state.stencil_front_ops = false;
				cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eFront, stencil_front_ops.fail_op,
				                    stencil_front_ops.pass_op, stencil_front_ops.depth_fail_op,
				                    stencil_front_ops.compare_op);
			}
			if (dirty_state.stencil_back_ops) {
				dirty_state.stencil_back_ops = false;
				cmdbuf.setStencilOp(vk::StencilFaceFlagBits::eBack, stencil_back_ops.fail_op,
				                    stencil_back_ops.pass_op, stencil_back_ops.depth_fail_op,
				                    stencil_back_ops.compare_op);
			}
		}
		if (dirty_state.stencil_front_reference && dirty_state.stencil_back_reference &&
		    stencil_front_reference == stencil_back_reference) {
			dirty_state.stencil_front_reference = false;
			dirty_state.stencil_back_reference  = false;
			cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack,
			                           stencil_front_reference);
		} else {
			if (dirty_state.stencil_front_reference) {
				dirty_state.stencil_front_reference = false;
				cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eFront,
				                           stencil_front_reference);
			}
			if (dirty_state.stencil_back_reference) {
				dirty_state.stencil_back_reference = false;
				cmdbuf.setStencilReference(vk::StencilFaceFlagBits::eBack, stencil_back_reference);
			}
		}
		if (dirty_state.stencil_front_write_mask && dirty_state.stencil_back_write_mask &&
		    stencil_front_write_mask == stencil_back_write_mask) {
			dirty_state.stencil_front_write_mask = false;
			dirty_state.stencil_back_write_mask  = false;
			cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack,
			                           stencil_front_write_mask);
		} else {
			if (dirty_state.stencil_front_write_mask) {
				dirty_state.stencil_front_write_mask = false;
				cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eFront,
				                           stencil_front_write_mask);
			}
			if (dirty_state.stencil_back_write_mask) {
				dirty_state.stencil_back_write_mask = false;
				cmdbuf.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, stencil_back_write_mask);
			}
		}
		if (dirty_state.stencil_front_compare_mask && dirty_state.stencil_back_compare_mask &&
		    stencil_front_compare_mask == stencil_back_compare_mask) {
			dirty_state.stencil_front_compare_mask = false;
			dirty_state.stencil_back_compare_mask  = false;
			cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack,
			                             stencil_front_compare_mask);
		} else {
			if (dirty_state.stencil_front_compare_mask) {
				dirty_state.stencil_front_compare_mask = false;
				cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eFront,
				                             stencil_front_compare_mask);
			}
			if (dirty_state.stencil_back_compare_mask) {
				dirty_state.stencil_back_compare_mask = false;
				cmdbuf.setStencilCompareMask(vk::StencilFaceFlagBits::eBack,
				                             stencil_back_compare_mask);
			}
		}
	}
	if (dirty_state.cull_mode) {
		dirty_state.cull_mode = false;
		cmdbuf.setCullMode(cull_mode);
	}
	if (dirty_state.front_face) {
		dirty_state.front_face = false;
		cmdbuf.setFrontFace(front_face);
	}
	if (dirty_state.blend_constants) {
		dirty_state.blend_constants = false;
		cmdbuf.setBlendConstants(blend_constants.data());
	}
	if (dirty_state.line_width) {
		dirty_state.line_width = false;
		cmdbuf.setLineWidth(line_width);
	}
	if (dirty_state.feedback_loop_aspects && graphics.attachment_feedback_loop_enabled) {
		dirty_state.feedback_loop_aspects = false;
		cmdbuf.setAttachmentFeedbackLoopEnableEXT(feedback_loop_aspects);
	}
}

} // namespace Libs::Graphics
