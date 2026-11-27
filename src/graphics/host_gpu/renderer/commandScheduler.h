#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           EndRendering() { m_command.EndRendering(); }
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Records a small command buffer, submits it behind all submitted work and waits for it. The
	// current command buffer stays open and the tick timeline does not advance. Its pool is not
	// shared: call it only from the thread that records the current command buffer.
	template <typename Record>
	void SubmitDetached(Record&& record) {
		record(BeginDetached());
		EndDetached();
	}
	// Like SubmitDetached followed by Flush, in ONE queue submission: the detached commands go
	// first in it, the open command buffer after them, and only the detached commands are waited
	// for (a timeline semaphore of their own). A new command buffer begins. False, with nothing
	// recorded, when that cannot be done (the command recorder submits on its own thread).
	template <typename Record>
	bool SubmitDetachedWithCurrent(Record&& record) {
		if (!CanSubmitDetachedWithCurrent()) {
			return false;
		}
		record(BeginDetached());
		EndDetachedWithCurrent();
		return true;
	}
	[[nodiscard]] bool CanSubmitDetachedWithCurrent() const;
	// KYTY_COPY_QUEUE: records a small command buffer for the copy queue (another queue family,
	// GraphicContext::copy_queue), submits it waiting on the GPU for every tick handed out before
	// the current one (the master timeline, not the open command buffer, which stays open and is
	// not submitted), and waits for it on the host. False, with nothing recorded, without a copy
	// queue. Only from the thread that records the current command buffer.
	template <typename Record>
	bool SubmitOnCopyQueue(Record&& record) {
		if (!HasCopyQueue()) {
			return false;
		}
		record(BeginCopyQueue());
		EndCopyQueue();
		return true;
	}
	[[nodiscard]] bool HasCopyQueue() const noexcept;
	// Moves recording to a new command buffer. `tail` records what is left of the current one when
	// the commands are submitted, so a draw can wait there for its pipeline while this thread goes
	// on. The tail ends the rendering if it is open.
	void           Split(Common::UniqueFunction<void, vk::CommandBuffer>&& tail);
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      StartOperation();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferRelease(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	// True while an operation deferred at `tick` or later has not completed.
	[[nodiscard]] bool        HasPriorityOperationsSince(uint64_t tick);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	// Detached commands submitted in the same vkQueueSubmit as the open command buffer, ahead of
	// it, signalling `semaphore` to `value`.
	struct DetachedPrefix {
		vk::CommandBuffer buffer;
		vk::Semaphore     semaphore;
		uint64_t          value;
	};

	struct Segment {
		vk::CommandBuffer                               buffer;
		Common::UniqueFunction<void, vk::CommandBuffer> tail;
	};

	void              BeginNext();
	vk::CommandBuffer BeginDetached();
	void              EndDetached();
	void              EndDetachedWithCurrent();
	vk::CommandBuffer BeginCopyQueue();
	void              EndCopyQueue();
	uint64_t          SubmitWith(SubmitInfo submit, const DetachedPrefix* prefix);
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void SealReleasesLocked();
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	vk::CommandPool              m_detached_pool    = nullptr;
	vk::CommandBuffer            m_detached_command = nullptr;
	vk::Fence                    m_detached_fence   = nullptr;
	vk::Semaphore                m_detached_timeline = nullptr;
	uint64_t                     m_detached_value    = 0;
	vk::CommandPool              m_copy_pool         = nullptr;
	vk::CommandBuffer            m_copy_command      = nullptr;
	vk::Fence                    m_copy_fence        = nullptr;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	// Recorded before m_command and submitted with it.
	std::vector<Segment>         m_segments;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::vector<Common::UniqueFunction<void>> m_unsealed_releases;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	uint64_t                     m_priority_active_tick = 0; // 0 while none runs
	OperationState               m_operation_state      = OperationState::Open;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
