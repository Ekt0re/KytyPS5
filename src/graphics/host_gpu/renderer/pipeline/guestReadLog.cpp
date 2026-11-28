#include "graphics/host_gpu/renderer/pipeline/guestReadLog.h"

#include <xxhash.h>

#include <algorithm>
#include <cstring>

namespace Libs::Graphics {

void GuestReadLog::Clear() {
	m_ranges.clear();
	m_bytes.clear();
	m_loads.clear();
	m_finished = false;
}

void GuestReadLog::AddRange(uint64_t address, std::span<const uint32_t> words) {
	if (words.empty()) return;
	Range range {.address = address, .size = words.size_bytes()};
	if (range.size <= MaxBytesKept) {
		range.offset = static_cast<uint32_t>(m_bytes.size());
		const auto* bytes = reinterpret_cast<const uint8_t*>(words.data());
		m_bytes.insert(m_bytes.end(), bytes, bytes + range.size);
	} else {
		range.hash = XXH3_64bits(words.data(), words.size_bytes());
	}
	m_ranges.push_back(range);
}

void GuestReadLog::AddLoads(std::span<const std::pair<uint64_t, uint32_t>> loads) {
	m_loads.insert(m_loads.end(), loads.begin(), loads.end());
}

bool GuestReadLog::Finish() {
	if (m_finished) return true;
	m_finished = true;
	std::ranges::sort(m_loads);
	// Equal pairs are one load; equal addresses with other values saw the memory change.
	m_loads.erase(std::unique(m_loads.begin(), m_loads.end()), m_loads.end());
	for (size_t i = 1; i < m_loads.size(); ++i) {
		if (m_loads[i].first == m_loads[i - 1].first) return false;
	}
	// Adjacent DWORDs become one range with its bytes.
	for (size_t i = 0; i < m_loads.size();) {
		size_t end = i + 1;
		while (end < m_loads.size() && m_loads[end].first == m_loads[end - 1].first + 4u) ++end;
		Range range {.address = m_loads[i].first,
		             .size    = (end - i) * sizeof(uint32_t),
		             .offset  = static_cast<uint32_t>(m_bytes.size())};
		for (size_t k = i; k < end; ++k) {
			const auto value = m_loads[k].second;
			const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
			m_bytes.insert(m_bytes.end(), bytes, bytes + sizeof(value));
		}
		m_ranges.push_back(range);
		i = end;
	}
	return true;
}

GuestReadLog::Check GuestReadLog::StillValid(Memory& memory, uint64_t& bytes_compared) const {
	std::vector<uint8_t> copy;
	for (const auto& range: m_ranges) {
		if (memory.GpuOwned(range.address, range.size)) return Check::GpuOwned;
		const auto* current = static_cast<const uint8_t*>(memory.View(range.address, range.size));
		if (current == nullptr) {
			copy.resize(range.size);
			if (!memory.Read(range.address, copy.data(), range.size)) return Check::Unreadable;
			current = copy.data();
		}
		bytes_compared += range.size;
		const bool same = range.offset != UINT32_MAX
		                      ? std::memcmp(current, m_bytes.data() + range.offset, range.size) == 0
		                      : XXH3_64bits(current, range.size) == range.hash;
		if (!same) return Check::Changed;
	}
	return Check::Valid;
}

} // namespace Libs::Graphics
