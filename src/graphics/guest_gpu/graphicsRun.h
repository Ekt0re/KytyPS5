#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <string>
#include <thread>

namespace Libs::Graphics {

class RenderContext;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	// Commands sent to the GPU thread that it has not started yet.
	[[nodiscard]] uint32_t PendingCommands() const noexcept {
		return m_pending_commands.load(std::memory_order_acquire);
	}
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// The guest released the REWIND packet at `rewind`: the commands it held back, usually a
	// chained buffer, are written now and are scanned for compute dispatches (Prescan).
	void ReleaseRewind(const uint32_t* rewind);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;
	// Submissions accepted and completed per queue (0 graphics, 1 + n compute queue 0x20 + n),
	// for frame capture and replay pacing (graphics/frameCapture.cpp).
	static constexpr uint32_t PacingQueues = 1 + 7 * 8;
	[[nodiscard]] uint64_t    AcceptedSubmissions(uint32_t queue_id) const noexcept {
		return m_accepted[queue_id].load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t CompletedSubmissions(uint32_t queue_id) const noexcept {
		return m_completed[queue_id].load(std::memory_order_acquire);
	}

	// Finer than submissions: the PM4 packets each queue has executed, and how often the GPU
	// thread found nothing it could run (every queue empty or parked at a REWIND, a wait or a
	// semaphore). A frame of GTA VI is one graphics submission chained through about two hundred
	// REWINDs, so the replay paces the guest's memory writes by packets.
	[[nodiscard]] uint64_t ExecutedPackets(uint32_t queue_id) const noexcept {
		return m_packets[queue_id].load(std::memory_order_acquire);
	}
	[[nodiscard]] uint64_t Parks() const noexcept { return m_parks.load(std::memory_order_acquire); }
	// The queue executes packets only while it has executed fewer than `limit` (replay): its
	// commands then see guest memory change at the same packets as in the recorded run.
	// Packets are counted for submissions made from here on only: commands still running when a
	// capture starts are not part of it.
	// Called on the GPU thread (SendCommandSync), between two packets: the submission it may be
	// in the middle of is an earlier one, too.
	void MarkPacketEpoch() noexcept {
		m_epoch.fetch_add(1, std::memory_order_acq_rel);
		m_counting = false;
	}
	void SetPacketLimit(uint32_t queue_id, uint64_t limit) noexcept {
		m_packet_limit[queue_id].store(limit, std::memory_order_release);
	}
	// One line per queue with unfinished commands: the packet it stands at. For diagnostics.
	[[nodiscard]] std::string DescribeUnfinished();

	[[nodiscard]] static bool IsGpuThread() noexcept;

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

	enum class SubmissionType { Commands, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Commands;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		Pm4Execution              execution;
		bool                      blocked         = false;
		uint32_t                  epoch           = 0;
		uint64_t                  flip_request_id = 0;
	};

	void              Enqueue(Submission submission);
	// Has the compute programs that `commands` dispatches translated ahead.
	void              Prescan(uint32_t queue_id, std::span<const uint32_t> commands);
	void              ProcessCommands();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	std::binary_semaphore                          m_suspend_point_ready {1};
	// The compute registers each queue's submitted commands leave, as Prescan sees them.
	std::mutex                                     m_prescan_mutex;
	std::array<HW::CsStageRegisters, QueueCount>   m_prescan_regs {};
	// The graphics queue's registers as Prescan sees them, for graphics translate-ahead.
	std::unique_ptr<CommandProcessor>              m_prescan_shadow;

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	std::atomic_int m_done_num = 0;
	std::array<std::atomic<uint64_t>, QueueCount> m_accepted {};
	std::array<std::atomic<uint64_t>, QueueCount> m_completed {};
	std::array<std::atomic<uint64_t>, QueueCount> m_packets {};
	std::array<std::atomic<uint64_t>, QueueCount> m_packet_limit;
	std::atomic<uint64_t>                         m_parks {0};
	std::atomic<uint32_t>                         m_epoch {0};
	bool                                          m_counting = true; // GPU thread only
	std::jthread    m_thread;

	friend class CommandProcessor;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
