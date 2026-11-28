#include "graphics/shader/recompiler/ir/passes/DeadCodeElimination.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <utility>

namespace Libs::Graphics::ShaderRecompiler::IR {

void RemoveIdentities(Program& program) {
	// Resolve metadata before deleting its identity nodes; retain only folded native constants.
	for (auto& write: program.scalar_writes) {
		const auto value = write.value.Resolve();
		write.value = value.IsImmediate() && value.GetType() == Type::U32 ? value : Value {};
	}
	for (auto* block: program.blocks) {
		block->condition       = block->condition.Resolve();
		block->indirect_target = block->indirect_target.Resolve();
	}
	bool has_identities = false;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::Identity) {
				continue;
			}
			has_identities = true;
			if (inst.uses.empty()) {
				continue;
			}
			// Forward any remaining non-Identity uses to the fully resolved target in order.
			const auto replacement = inst.Arg(0).Resolve();
			auto*      target      = replacement.TryInstruction();
			const auto old_uses    = std::exchange(inst.uses, {});
			for (const auto& use: old_uses) {
				if (use.user->Parent() != nullptr && use.user->GetOpcode() == ValueOpcode::Identity) {
					continue;
				}
				Inst::ArgSlot(*use.user, use.operand) = replacement;
				if (target != nullptr) {
					target->uses.push_back(use);
				}
			}
		}
	}
	if (!has_identities) {
		return;
	}
	// Strip Identity entries from every live use list in one linear pass (preserving order),
	// avoiding O(U^2) per-node RemoveUse scans when many folded instructions share a target.
	const auto is_identity_use = [](const Use& use) {
		return use.user->Parent() != nullptr && use.user->GetOpcode() == ValueOpcode::Identity;
	};
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (inst.GetOpcode() != ValueOpcode::Identity && !inst.uses.empty()) {
				std::erase_if(inst.uses, is_identity_use);
			}
		}
	}
	for (auto& inst: program.value_storage) {
		if (!inst.uses.empty()) {
			std::erase_if(inst.uses, is_identity_use);
		}
	}
	for (auto* block: program.blocks) {
		auto& instructions = block->Instructions();
		for (auto& inst: instructions) {
			if (inst.GetOpcode() == ValueOpcode::Identity) {
				inst.DropArgs();
			}
		}
		std::erase_if(instructions, [](const Inst& inst) {
			return inst.GetOpcode() == ValueOpcode::Identity;
		});
	}
}

void EliminateDeadCode(const BlockList& blocks) {
	bool changed;
	do {
		std::vector<Inst*> pending;
		for (auto* block: blocks) {
			for (auto& inst: *block) {
				// Retained resource-planning instructions live outside the block list.
				inst.live = inst.MayHaveSideEffects() || std::ranges::any_of(inst.Uses(),
				    [](const Use& use) { return use.user->Parent() == nullptr; });
				if (inst.live) pending.push_back(&inst);
			}
		}
		while (!pending.empty()) {
			const auto* inst = pending.back();
			pending.pop_back();
			for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
				auto* input = inst->Arg(arg).TryInstruction();
				if (input != nullptr && input->Parent() != nullptr && !input->live) {
					input->live = true;
					pending.push_back(input);
				}
			}
		}
		// Compact live instructions' use lists in a single order-preserving pass instead of
		// calling O(U) RemoveUse per dead operand.
		const auto is_dead_use = [](const Use& use) {
			return use.user->Parent() != nullptr && !use.user->live;
		};
		for (auto* block: blocks) {
			for (auto& inst: *block) {
				if (inst.live && !inst.uses.empty()) {
					std::erase_if(inst.uses, is_dead_use);
				}
			}
		}
		// Detach any remaining external-storage operands and drop dead args before erasure.
		for (auto* block: blocks) {
			for (auto& inst: *block) {
				if (inst.live) continue;
				for (size_t arg = 0; arg < inst.NumArgs(); ++arg) {
					auto* input = inst.Arg(arg).TryInstruction();
					if (input != nullptr && input->Parent() == nullptr) {
						inst.RemoveUse(input, arg);
					}
				}
				inst.DropArgs();
			}
		}
		for (auto* block: blocks)
			std::erase_if(block->Instructions(), [](const Inst& inst) { return !inst.live; });

		changed = false;
		for (auto* block: blocks) {
			const auto& term = block->terminator;
			if (term.kind != CFG::TerminatorKind::ConditionalBranch || term.loop_header ||
			    block->condition.IsImmediate()) continue;
			for (const bool take_true: {false, true}) {
				const auto* empty = take_true ? term.false_block : term.true_block;
				const auto* merge = take_true ? term.true_block : term.false_block;
				if (empty == nullptr || merge == nullptr || !empty->empty() ||
				    empty->terminator.loop_header || merge->terminator.loop_header ||
				    empty->terminator.kind != CFG::TerminatorKind::Branch ||
				    empty->terminator.true_block != merge ||
				    std::ranges::any_of(*merge, [](const Inst& inst) {
					    return inst.GetOpcode() == ValueOpcode::Phi;
				    })) continue;
				// Keep the CFG and SSA edges intact; only the empty detour's predicate is dead.
				const auto resolved_condition = block->condition.Resolve();
				for (auto& inst: *block) {
					if (inst.GetOpcode() == ValueOpcode::Reference &&
					    inst.Arg(0).Resolve() == resolved_condition)
						inst.SetArg(0, Value(take_true));
				}
				block->condition = Value(take_true);
				changed = true;
				break;
			}
		}
	} while (changed);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
