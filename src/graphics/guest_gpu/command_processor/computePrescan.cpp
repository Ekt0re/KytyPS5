#include "graphics/guest_gpu/command_processor/computePrescan.h"

#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"
#include "kernel/memory.h"

#include <vector>

namespace Libs::Graphics {

namespace {

// Indirect buffers nest this deep at most; deeper ones are not followed.
constexpr uint32_t MaxDepth = 4;
// Bounds the dwords one scan copies from indirect buffers and register lists.
constexpr uint64_t CopyBudgetDw = uint64_t {1} << 22u;

class Prescan {
public:
	Prescan(HW::CsStageRegisters& regs, ComputePrescanSink& sink, CommandProcessor* shadow)
	    : m_regs(regs), m_sink(sink), m_shadow(shadow) {}

	void Walk(std::span<const uint32_t> commands, uint32_t depth) {
		// A chained buffer replaces the current one, like in the command processor.
		std::vector<uint32_t> chained;
		for (size_t position = 0; position < commands.size();) {
			const uint32_t header = commands[position];
			if (header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u) {
				position++;
				continue;
			}
			// Only type 3 packets follow; anything else means the stream is not understood.
			if ((header >> 30u) != 3u) return;
			const uint32_t length = KYTY_PM4_LEN(header);
			if (length > commands.size() - position) return;
			const auto body = commands.subspan(position + 1, length - 1);
			position += length;
			const bool indirect = ((header >> 8u) & 0xffu) == Pm4::IT_INDIRECT_BUFFER;
			uint64_t   chain_address = 0;
			uint32_t   chain_dwords  = 0;
			if (indirect && body.size() == 3 && (body[2] & (1u << 20u)) != 0) {
				chain_address = IndirectAddress(body);
				chain_dwords  = body[2] & 0xfffffu;
			} else if (indirect && body.size() == 13) {
				// A branch chains to one of two buffers, depending on memory the GPU may still
				// write: both are scanned, the else buffer first.
				if ((body[0] & 0x3u) == 2u && depth < MaxDepth) {
					std::vector<uint32_t> other;
					if (Copy(IndirectAddress(body.subspan(10)), body[12] & 0xfffffu, other)) {
						Walk(other, depth + 1);
					}
				}
				chain_address = IndirectAddress(body.subspan(7));
				chain_dwords  = body[9] & 0xfffffu;
			}
			if (chain_address != 0) {
				std::vector<uint32_t> next;
				if (!Copy(chain_address, chain_dwords, next)) return;
				chained  = std::move(next);
				commands = chained;
				position = 0;
				continue;
			}
			Packet(header, body, depth);
		}
	}

private:
	static uint64_t IndirectAddress(std::span<const uint32_t> body) {
		return (body[0] & ~0x3u) | (static_cast<uint64_t>(body[1]) << 32u);
	}

	void SetShRegister(uint32_t offset, uint32_t value) {
		// Registers the command processor rejects stay as they were.
		DecodeCsRegister(m_regs, offset & ~0x70000000u, value);
	}

	bool Copy(uint64_t address, uint32_t dwords, std::vector<uint32_t>& data) {
		if (address == 0 || dwords == 0 || dwords > m_budget) return false;
		m_budget -= dwords;
		data.resize(dwords);
		// Private mappings too; on a guest thread, a read of GPU-written memory is handled like
		// the guest's own.
		return LibKernel::Memory::CopySparseMemory(address, data.data(), uint64_t {dwords} * 4u);
	}

	// Graphics state for translate-ahead: register packets go through the command processor's
	// own handlers on the shadow, which only change its registers. Packets that would reach the
	// host or guest memory (flips, user-data tags) are left out.
	void Shadow(uint32_t header, std::span<const uint32_t> body) {
		const uint32_t opcode = (header >> 8u) & 0xffu;
		const auto     call   = [&](cp_op_parser_func_t handler) {
            const auto size = static_cast<uint32_t>(body.size() + 1);
            handler(*m_shadow, header & ~1u, body.data(), size, size);
		};
		switch (opcode) {
			case Pm4::IT_SET_CONTEXT_REG:
			case Pm4::IT_SET_SH_REG:
			case Pm4::IT_SET_CONTEXT_REG_INDIRECT:
			case Pm4::IT_SET_SH_REG_INDIRECT:
			case Pm4::IT_SET_UCONFIG_REG_INDIRECT:
			case Pm4::IT_CLEAR_STATE: call(g_cp_op_func[opcode]); break;
			case Pm4::IT_SET_UCONFIG_REG:
			case Pm4::IT_SET_UCONFIG_REG_INDEX:
				if (KYTY_PM4_R(header) == 0u && !body.empty() &&
				    body[0] != Pm4::SQ_THREAD_TRACE_USERDATA_2) {
					call(g_cp_op_func[opcode]);
				}
				break;
			case Pm4::IT_NOP:
				if (KYTY_PM4_R(header) == Pm4::R_CONTEXT_STATE) call(g_cp_op_custom_func[Pm4::R_CONTEXT_STATE]);
				break;
			case Pm4::IT_DRAW_INDEX_2:
			case Pm4::IT_DRAW_INDEX_OFFSET_2:
			case Pm4::IT_DRAW_INDEX_MULTI_INST:
			case Pm4::IT_DRAW_INDEX_AUTO: m_sink.FoundDraw(*m_shadow, false); break;
			case Pm4::IT_DRAW_INDIRECT:
			case Pm4::IT_DRAW_INDEX_INDIRECT:
			case Pm4::IT_DRAW_INDIRECT_MULTI:
			case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: m_sink.FoundDraw(*m_shadow, true); break;
			default: break;
		}
	}

	void Packet(uint32_t header, std::span<const uint32_t> body, uint32_t depth) {
		const uint32_t opcode = (header >> 8u) & 0xffu;
		if (m_shadow != nullptr && !body.empty()) {
			Shadow(header, body);
		}
		switch (opcode) {
			case Pm4::IT_SET_SH_REG:
				for (size_t i = 1; i < body.size(); i++) {
					SetShRegister(body[0] + static_cast<uint32_t>(i - 1), body[i]);
				}
				break;
			case Pm4::IT_SET_SH_REG_INDIRECT: {
				if (body.size() != 4) break;
				const uint64_t address =
				    (body[0] & 0xfffffffcu) | (static_cast<uint64_t>(body[1]) << 32u);
				std::vector<uint32_t> pairs;
				if (!Copy(address, (body[3] & 0x3fffu) * 2u, pairs)) break;
				for (size_t i = 0; i + 1 < pairs.size(); i += 2) {
					if (pairs[i] != 0xffffffffu) SetShRegister(pairs[i], pairs[i + 1]);
				}
				break;
			}
			case Pm4::IT_DISPATCH_DIRECT:
				if (body.size() == 4) m_sink.FoundDispatch(m_regs, body[3], body.data(), 0);
				break;
			case Pm4::IT_DISPATCH_INDIRECT:
				if (header == 0xc0021600u) {
					m_sink.FoundDispatch(m_regs, body[2], nullptr,
					                     body[0] | (static_cast<uint64_t>(body[1]) << 32u));
				} else if (header == 0xc0011600u) {
					m_sink.FoundDispatch(m_regs, body[1], nullptr,
					                     m_dispatch_base != 0 ? m_dispatch_base + body[0] : 0);
				}
				break;
			case Pm4::IT_SET_BASE:
				// As CpOpSetBase: base 1 of the compute shader type is the indirect dispatch base.
				if (KYTY_PM4_LEN(header) == 4u && body.size() == 3 && (body[0] & 0xfu) == 1u &&
				    ((header >> 1u) & 0x3u) == 1u) {
					m_dispatch_base = (body[1] & ~0x7ull) | (static_cast<uint64_t>(body[2] & 0xffffu) << 32u);
				}
				break;
			case Pm4::IT_INDIRECT_BUFFER: {
				// A call. A 14-dword packet is a branch, which is not followed.
				if (body.size() != 3 || depth >= MaxDepth) break;
				std::vector<uint32_t> commands;
				if (Copy(IndirectAddress(body), body[2] & 0xfffffu, commands)) {
					Walk(commands, depth + 1);
				}
				break;
			}
			default: break;
		}
	}

	HW::CsStageRegisters& m_regs;
	ComputePrescanSink&   m_sink;
	CommandProcessor*     m_shadow;
	uint64_t              m_budget = CopyBudgetDw;
	uint64_t              m_dispatch_base = 0;
};

} // namespace

void PrescanComputeDispatches(std::span<const uint32_t> commands, HW::CsStageRegisters& regs,
                              ComputePrescanSink& sink, CommandProcessor* shadow) {
	Prescan(regs, sink, shadow).Walk(commands, 0);
}

} // namespace Libs::Graphics
