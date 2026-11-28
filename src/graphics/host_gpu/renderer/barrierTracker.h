#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERTRACKER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERTRACKER_H_

#include "graphics/host_gpu/rangeSet.h"

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics {

// Guest memory a dispatch can read or write. Unknown accesses retain conservative ordering for
// device-address stores, utility buffers and dynamic heaps that cannot be enumerated as ranges.
struct ShaderAccessSet {
	RangeSet reads;
	RangeSet writes;

	bool unknown_reads  = false;
	bool unknown_writes = false;

	void AddRead(uint64_t address, uint64_t size) {
		if (size == 0) return;
		if (size > UINT64_MAX - address) {
			unknown_reads = true;
			return;
		}
		reads.Add(address, size);
	}
	void AddWrite(uint64_t address, uint64_t size) {
		if (size == 0) return;
		if (size > UINT64_MAX - address) {
			unknown_writes = true;
			return;
		}
		writes.Add(address, size);
	}

	[[nodiscard]] bool HasReads() const { return unknown_reads || !reads.Empty(); }
	[[nodiscard]] bool HasWrites() const { return unknown_writes || !writes.Empty(); }
	[[nodiscard]] bool HasAccesses() const { return HasReads() || HasWrites(); }
};

enum class BarrierDecision : uint8_t { None, ComputeToCompute, AllToCompute, ComputeToAll, AllToAll };

// Orders guest compute dispatches and the non-dispatch commands recorded between them. Range
// history is unioned until a global barrier is emitted; at that point the barrier covers all of it.
class BarrierTracker {
public:
	[[nodiscard]] BarrierDecision BeforeDispatch(const ShaderAccessSet& current) {
		if (m_other_command) {
			m_other_command = false;
			if (current.HasWrites()) return BarrierDecision::AllToCompute;
		}
		if (m_pending && Hazard(current)) {
			ClearDispatches();
			return BarrierDecision::ComputeToCompute;
		}
		return BarrierDecision::None;
	}

	void AfterDispatch(const ShaderAccessSet& current) noexcept {
		Merge(current);
		m_pending       = true;
		m_other_command = false;
	}

	[[nodiscard]] BarrierDecision BeforeCommand() noexcept {
		if (!m_pending) {
			m_other_command = true;
			return BarrierDecision::None;
		}
		const auto result = BarrierDecision::ComputeToAll;
		ClearDispatches();
		m_other_command = true;
		return result;
	}

	[[nodiscard]] BarrierDecision BeforeBarrier() noexcept {
		if (!m_pending) return BarrierDecision::None;
		const auto result = BarrierDecision::ComputeToAll;
		ClearDispatches();
		return result;
	}

	[[nodiscard]] BarrierDecision Flush() noexcept {
		if (m_pending) {
			const auto result = BarrierDecision::ComputeToAll;
			ClearDispatches();
			m_other_command = false;
			return result;
		}
		if (m_other_command) {
			m_other_command = false;
			return BarrierDecision::AllToAll;
		}
		return BarrierDecision::None;
	}

	void Reset() noexcept {
		m_reads.Clear();
		m_writes.Clear();
		m_unknown_reads  = false;
		m_unknown_writes = false;
		m_pending        = false;
		m_other_command  = false;
	}
	[[nodiscard]] bool HasPendingDispatches() const noexcept { return m_pending; }
	[[nodiscard]] bool HasPendingWrites() const noexcept {
		return m_pending && (m_unknown_writes || !m_writes.Empty());
	}

private:
	void Merge(const ShaderAccessSet& current) {
		current.reads.ForEach([this](uint64_t address, uint64_t end) {
			m_reads.Add(address, end - address);
		});
		current.writes.ForEach([this](uint64_t address, uint64_t end) {
			m_writes.Add(address, end - address);
		});
		m_unknown_reads  |= current.unknown_reads;
		m_unknown_writes |= current.unknown_writes;
	}
	void ClearDispatches() noexcept {
		m_reads.Clear();
		m_writes.Clear();
		m_unknown_reads  = false;
		m_unknown_writes = false;
		m_pending        = false;
	}
	[[nodiscard]] bool Hazard(const ShaderAccessSet& current) const {
		if ((m_unknown_writes && current.HasAccesses()) ||
		    (current.unknown_writes && (m_unknown_reads || m_unknown_writes || !m_reads.Empty() ||
		                               !m_writes.Empty())) ||
		    (m_unknown_reads && current.HasWrites()) ||
		    (current.unknown_reads && (m_unknown_writes || !m_writes.Empty()))) {
			return true;
		}
		bool hazard = false;
		current.reads.ForEach([&](uint64_t address, uint64_t end) {
			hazard |= m_writes.Intersects(address, end - address);
		});
		current.writes.ForEach([&](uint64_t address, uint64_t end) {
			hazard |= m_reads.Intersects(address, end - address) ||
			          m_writes.Intersects(address, end - address);
		});
		return hazard;
	}

	RangeSet m_reads;
	RangeSet m_writes;
	bool     m_unknown_reads  = false;
	bool     m_unknown_writes = false;
	bool     m_pending        = false;
	bool     m_other_command  = false;
};

[[nodiscard]] inline bool BarrierTrackerEnabled() {
	static const bool enabled = [] {
		const char* value    = std::getenv("KYTY_BARRIER_TRACKER");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BARRIERTRACKER_H_
