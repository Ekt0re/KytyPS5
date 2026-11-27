#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/command_processor/rawCommandBuffer.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

enum class Pm4ProcessResult { Complete, Blocked };

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

struct IndirectDrawRegisters {
	uint32_t vertex_offset;
	uint32_t instance_offset;
	uint32_t index_offset;
	uint32_t draw_index;
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }
	// The packet the execution stands at (null when it has none), for diagnostics.
	[[nodiscard]] const uint32_t* Position() const noexcept {
		if (m_buffer_stack.empty()) return nullptr;
		const auto& cursor = m_buffer_stack.back();
		return cursor.offset_dw < cursor.commands.size() ? cursor.commands.data() + cursor.offset_dw
		                                                 : nullptr;
	}
	// Where the packet under the cursor sits, for the wait reporter (note: guest addresses).
	void DiagnosticPacketLocation(uint64_t& buffer, uint32_t& offset_dw) const noexcept {
		buffer    = 0;
		offset_dw = 0;
		if (m_buffer_stack.empty()) {
			return;
		}
		const auto& cursor = m_buffer_stack.back();
		buffer             = reinterpret_cast<uint64_t>(cursor.commands.data());
		offset_dw          = cursor.offset_dw;
	}

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands; // Guest addresses.
		uint32_t                  offset_dw = 0;
		// KYTY_RAW_CMDBUF: the same DWORDs in the backing alias (nullptr: decode through
		// commands). Resolved when the cursor is first decoded in a Process() call.
		const uint32_t* view       = nullptr;
		bool            view_ready = false;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
};

class CommandProcessor {
public:
	CommandProcessor(RenderContext& renderer, int interrupt_event_id)
	    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}
	~CommandProcessor() = default;

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ResetQueue(uint32_t op);
	void ApplyContextStateOperation(ContextStateOperation operation);

	void            BufferInit();
	void            BufferFlush();
	void            BufferWait();
	HW::Context&    GetCtx() { return m_ctx; }
	HW::UserConfig& GetUcfg() { return m_ucfg; }
	HW::Shader&     GetShCtx() { return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return m_dispatch_indirect_args_base_addr;
	}
	void SetNumInstances(uint32_t num_instances);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       IndirectDrawRegisters registers, uint32_t draw_initiator, bool indexed);
	void ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address, uint64_t data,
	                uint32_t interrupt_context_id, bool gl2_writeback);
	void WriteBackImages();
	// A partial flush, ACQUIRE_MEM, WAIT_REG_MEM, memory semaphore, RELEASE_MEM, flip or the end of
	// a submission: this queue's earlier GPU work is done as far as later indirect argument reads
	// go.
	void NoteGuestSync();
	void Flip(int handle, int index, int flip_mode, int64_t flip_arg);
	void PrepareCpuFlip(uint64_t request_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(const uint64_t* label);
	void ProcessPssr(const Pssr::Request& request);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);

	void WaitForRewind(bool valid);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t wait_op);
	void MemSemaphore(uint64_t* counter, bool signal, bool write);
	// For the stall reporter (waitDiagnostics.h); the queue slot follows the event id.
	[[nodiscard]] uint32_t DiagnosticsSlot() const;
	void NoteWaitPoll(const void* address, bool is64, uint32_t func, uint64_t ref, uint64_t mask,
	                  uint64_t value);
	void NoteLabelWrite(const void* address, uint64_t value, uint32_t size);
	void WriteData(uint64_t dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void GetLodStats(volatile uint32_t* dst, uint32_t size, bool reset);
	void DmaData(uint8_t dst_sel, uint64_t dst_address_or_offset, uint8_t src_sel,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	[[nodiscard]] bool IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

private:
	void ProcessPm4(Pm4Execution& execution);
	void ResolveCommandView(Pm4Execution::BufferCursor& cursor);
	// A host GPU write into a live command span is published before the parser reads on.
	void RefreshLiveCommands(uint64_t dst, uint64_t size);
	void SuspendPm4();
	void                CountDraw();
	uint32_t            NumInstances();
	const void*         IndexAddress(uint32_t first_index) const;
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	RenderContext& m_renderer;
	HW::Context    m_ctx;
	std::optional<HW::Context> m_saved_ctx;
	HW::UserConfig m_ucfg;
	HW::Shader     m_sh_ctx;
	uint32_t       m_index_type_and_size              = 0;
	uint32_t       m_index_buffer_size                = 0;
	uint64_t       m_index_base_addr                  = 0;
	uint64_t       m_draw_indirect_args_base_addr     = 0;
	uint64_t       m_dispatch_indirect_args_base_addr = 0;
	// Persistent draw state. Indirect draws leave the address of their last instance count,
	// read only when a later draw inherits it.
	uint32_t m_num_instances      = 1;
	uint64_t m_num_instances_addr = 0;
	// A GPU-counted multi-draw defers the count read until NUM_INSTANCES is inherited.
	uint64_t m_num_instances_count_addr = 0;
	uint64_t m_num_instances_args_addr  = 0;
	uint32_t m_num_instances_max_count  = 0;
	uint32_t m_num_instances_stride     = 0;

	const int      m_interrupt_event_id;
	const uint32_t m_flush_draws                 = Config::GetFlushDraws();
	uint32_t       m_draws                       = 0; // Since the buffer start or CountDraw's flush
	uint64_t       m_synthetic_occlusion_counter = 0;
	bool           m_predicate_skip              = false;
	RawCommandBuffer::Mode m_raw_mode            = RawCommandBuffer::Mode::Off;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
