#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/recorder.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <optional>
#include <type_traits>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
	if (m_detached_pool != nullptr) {
		m_graphics.device.destroyFence(m_detached_fence, nullptr);
		m_graphics.device.destroyCommandPool(m_detached_pool, nullptr);
	}
}

// Detached submissions follow KytyPS5 PR #1254 by NeySlim ("gpu: read back submitted buffer
// data without a flush").
vk::CommandBuffer CommandScheduler::BeginDetached() {
	CheckActive();
	auto& device = m_graphics.device;
	if (m_detached_pool == nullptr) {
		vk::CommandPoolCreateInfo create {};
		create.queueFamilyIndex = m_graphics.queue_family;
		create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
		                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
		RequireVulkanSuccess(device.createCommandPool(&create, nullptr, &m_detached_pool),
		                     "create detached command pool");
		vk::CommandBufferAllocateInfo allocate {};
		allocate.commandPool        = m_detached_pool;
		allocate.level              = vk::CommandBufferLevel::ePrimary;
		allocate.commandBufferCount = 1;
		RequireVulkanSuccess(device.allocateCommandBuffers(&allocate, &m_detached_command),
		                     "allocate detached command buffer");
		const vk::FenceCreateInfo fence {};
		RequireVulkanSuccess(device.createFence(&fence, nullptr, &m_detached_fence),
		                     "create detached fence");
	}
	// With the command recorder, earlier ticks may still wait on its thread for their submission:
	// the copy must follow them on the queue.
	if (Recorder::Enabled() && m_master.SubmittedTick() + 1 < m_master.CurrentTick()) {
		Recorder::SyncAll();
	}
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	RequireVulkanSuccess(m_detached_command.begin(&begin), "begin detached command buffer");
	return m_detached_command;
}

void CommandScheduler::EndDetached() {
	// Commands the recorder queued for this buffer reach it before it ends.
	if (Recorder::IsDeferred(static_cast<VkCommandBuffer>(m_detached_command))) {
		Recorder::Sync(m_detached_command);
	}
	RequireVulkanSuccess(m_detached_command.end(), "end detached command buffer");
	vk::Result result;
	{
		Common::LockGuard lock(m_graphics.queue_mutex);
		vk::SubmitInfo    submit_info {};
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers    = &m_detached_command;
		result = m_graphics.queue.submit(1, &submit_info, m_detached_fence);
	}
	RequireVulkanSuccess(result, "submit detached command buffer");
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);
	// The signal of a queue-submission fence also covers every batch submitted earlier on this
	// queue.
	RequireVulkanSuccess(
	    m_graphics.device.waitForFences(1, &m_detached_fence, VK_TRUE, UINT64_MAX),
	    "wait for detached fence");
	RequireVulkanSuccess(m_graphics.device.resetFences(1, &m_detached_fence),
	                     "reset detached fence");
	RequireVulkanSuccess(m_detached_command.reset({}), "reset detached command buffer");
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		SealReleasesLocked();
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        !m_unsealed_releases.empty() || m_priority_active_tick != 0);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	{
		std::lock_guard lock(m_operation_mutex);
		SealReleasesLocked();
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() || !IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::StartOperation() {
	{
		std::lock_guard lock(m_operation_mutex);
		SealReleasesLocked();
	}
	PopPendingOperations();
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferRelease(Common::UniqueFunction<void>&& operation) {
	CheckActive();
	EXIT_IF(!operation);
	{
		std::lock_guard lock(m_operation_mutex);
		if (m_operation_state == OperationState::Open) {
			m_unsealed_releases.push_back(std::move(operation));
			return;
		}
	}
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::SealReleasesLocked() {
	const auto last_recorded_tick = m_command.IsInvalid() ? CurrentTick() - 1 : CurrentTick();
	for (auto& release: m_unsealed_releases) {
		m_pending_operations.push({std::move(release), last_recorded_tick});
	}
	m_unsealed_releases.clear();
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPriorityOperationsSince(uint64_t tick) {
	std::lock_guard lock(m_operation_mutex);
	// Operations are queued in tick order, so the newest one is at the back.
	return (!m_priority_operations.empty() && m_priority_operations.back().tick >= tick) ||
	       (m_priority_active_tick != 0 && m_priority_active_tick >= tick);
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active_tick != 0;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && m_priority_active_tick == 0; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at =
		    m_priority_active_tick != 0 && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	m_dynamic_state.Invalidate();
	return m_command;
}

void CommandScheduler::Split(Common::UniqueFunction<void, vk::CommandBuffer>&& tail) {
	EXIT_IF(m_command.IsInvalid());
	m_segments.push_back({m_command.m_buffer, std::move(tail)});
	m_command.m_rendering = false;
	m_command.m_buffer    = nullptr;
	BeginCommand();
}

namespace {

// What a deferred submission needs on the recorder thread (copied into its record).
struct DeferredSubmit {
	SubmitInfo       submit;
	vk::Queue        queue;
	Common::Mutex*   queue_mutex;
	MasterSemaphore* master;
	uint64_t         tick;
};
static_assert(std::is_trivially_copyable_v<DeferredSubmit>);

vk::Result QueueSubmit(vk::Queue queue, const SubmitInfo& submit, const vk::CommandBuffer* buffers,
                       uint32_t count) {
	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = submit.wait_stages.data();
	submit_info.commandBufferCount   = count;
	submit_info.pCommandBuffers      = buffers;
	submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

	return queue.submit(1, &submit_info, nullptr);
}

// Runs on the command recorder's thread once it ended the buffers.
void RunDeferredSubmit(const void* data, const VkCommandBuffer* buffers, uint32_t count) {
	const auto& deferred = *static_cast<const DeferredSubmit*>(data);
	static_assert(sizeof(vk::CommandBuffer) == sizeof(VkCommandBuffer));
	vk::Result result;
	{
		Common::LockGuard lock(*deferred.queue_mutex);
		result = QueueSubmit(deferred.queue, deferred.submit,
		                     reinterpret_cast<const vk::CommandBuffer*>(buffers), count);
		deferred.master->MarkSubmitted(deferred.tick);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);
	if (result != vk::Result::eSuccess) {
		EXIT("vkQueueSubmit failed: %s, tick=%" PRIu64 "\n", vk::to_string(result).c_str(),
		     deferred.tick);
	}
}

} // namespace

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());

	m_command.EndRendering();
	std::vector<vk::CommandBuffer> buffers;
	buffers.reserve(m_segments.size() + 1);
	for (const auto& segment: m_segments) {
		segment.tail(segment.buffer);
		buffers.push_back(segment.buffer);
	}
	m_segments.clear();
	buffers.push_back(m_command.m_buffer);
	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	// Buffers the GPU thread recorded through the command recorder are ended and submitted on the
	// recorder's thread, after the commands queued into them. Ticks are still handed out here, in
	// the order of the submissions, which the recorder keeps.
	const bool deferred =
	    Recorder::Enabled() && std::all_of(buffers.begin(), buffers.end(), [](vk::CommandBuffer b) {
		    return Recorder::IsDeferred(static_cast<VkCommandBuffer>(b));
	    });
	if (deferred) {
		DeferredSubmit run {};
		run.submit      = submit;
		run.queue       = graphics.queue;
		run.queue_mutex = &graphics.queue_mutex;
		run.master      = &m_master;
		run.tick        = m_master.NextTick();
		run.submit.AddSignal(m_master.Handle(), run.tick);
		static_assert(sizeof(vk::CommandBuffer) == sizeof(VkCommandBuffer));
		Recorder::EndAndRun(reinterpret_cast<const VkCommandBuffer*>(buffers.data()),
		                    static_cast<uint32_t>(buffers.size()), &RunDeferredSubmit, &run,
		                    sizeof(run));
		m_command.m_buffer = nullptr;
		return run.tick;
	}

	// Submissions still queued on the recorder thread signal earlier ticks: they go first.
	// So do the commands queued for buffers of this batch, and the recorder must not use the
	// command pool while this thread ends them.
	if (m_master.SubmittedTick() + 1 < m_master.CurrentTick() ||
	    std::any_of(buffers.begin(), buffers.end(), [](vk::CommandBuffer b) {
		    return Recorder::IsDeferred(static_cast<VkCommandBuffer>(b));
	    })) {
		Recorder::SyncAll();
	}
	for (const auto buffer: buffers) {
		EXIT_IF(buffer.end() != vk::Result::eSuccess);
	}

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);
		result = QueueSubmit(graphics.queue, submit, buffers.data(),
		                     static_cast<uint32_t>(buffers.size()));
		m_master.MarkSubmitted(tick);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		if (result == vk::Result::eErrorDeviceLost) {
			GpuBreadcrumbs::ReportDeviceLost(graphics, "vkQueueSubmit");
		}
		EXIT("vkQueueSubmit failed: %s, tick=%" PRIu64 "\n", vk::to_string(result).c_str(), tick);
	}

	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
