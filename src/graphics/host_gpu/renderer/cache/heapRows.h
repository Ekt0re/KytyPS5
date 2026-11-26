#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HEAPROWS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HEAPROWS_H_

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <fmt/format.h>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace Libs::Graphics {

// Rows of buffer V# tables that shaders selected and the host had not bound. A shader sets the
// row's bit past the BDA fault bitmap; the fault scan reports it here, and the next
// materialization of that root binds the row.
class HeapRows final: public ShaderRecompiler::IR::HeapRowSource {
public:
	bool Acquire(uint64_t shader, uint32_t source, uint64_t count,
	             ShaderRecompiler::IR::HeapRowView& view) override {
		std::lock_guard lock(m_mutex);
		const auto [it, inserted] = m_roots.try_emplace({shader, source, count});
		auto& root = it->second;
		if (inserted) {
			root.shader = shader;
			root.source = source;
			if (count <= ShaderRecompiler::IR::HeapRowBits - m_next) {
				root.first = m_next;
				m_bases.emplace(m_next, &root);
				m_next += static_cast<uint32_t>(count);
			} else {
				Print(fmt::format("GPU-selected buffer rows of shader 0x{:016x} source {} are not "
				                  "reported: no space left\n", shader, source));
			}
		}
		view.first = root.first;
		view.rows.assign(root.rows.begin(), root.rows.end());
		return root.first != UINT32_MAX;
	}

	void Reject(uint64_t shader, uint32_t source, uint32_t row,
	            const ShaderRecompiler::IR::DescriptorValue& value, std::string_view reason) override {
		std::lock_guard lock(m_mutex);
		const std::array words {value.dwords[0], value.dwords[1], value.dwords[2], value.dwords[3]};
		if (m_rejected.size() >= MaxLogged || !m_rejected.emplace(shader, source, row, words).second)
			return;
		Print(fmt::format("shader 0x{:016x} source {} selected buffer row {}, which reads as null "
		                  "({}): {:08x},{:08x},{:08x},{:08x}\n",
		                  shader, source, row, reason, words[0], words[1], words[2], words[3]));
	}

	// bit: index past the last page of the fault bitmap.
	void Report(uint64_t bit) {
		std::lock_guard lock(m_mutex);
		auto next = m_bases.upper_bound(static_cast<uint32_t>(bit));
		if (next == m_bases.begin()) return;
		auto& root = *std::prev(next)->second;
		const auto row = static_cast<uint32_t>(bit - root.first);
		// A root binds a bounded number of rows; the rows reported longest ago make way.
		if (std::ranges::find(root.rows, row) != root.rows.end()) return;
		if (root.rows.size() >= ShaderRecompiler::IR::MaxHeapRows) {
			if (!root.full) {
				root.full = true;
				Print(fmt::format("shader 0x{:016x} source {} selected more than {} buffer rows; "
				                  "accesses to evicted rows are lost until they are bound again\n",
				                  root.shader, root.source, ShaderRecompiler::IR::MaxHeapRows));
			}
			root.rows.erase(root.rows.begin());
		}
		root.rows.push_back(row);
		if (m_logged < MaxLogged) {
			++m_logged;
			Print(fmt::format("shader 0x{:016x} source {} selected unbound buffer row {}; accesses "
			                  "before it is bound were lost\n", root.shader, root.source, row));
		}
	}

private:
	static constexpr uint32_t MaxLogged = 4096u;

	struct Root {
		uint64_t              shader = 0;
		uint32_t              source = 0;
		uint32_t              first  = UINT32_MAX;
		bool                  full   = false;
		std::vector<uint32_t> rows; // Oldest first.
	};

	static void Print(const std::string& message) { Log::WriteToConsoleAndLog(message); }

	std::mutex                                                             m_mutex;
	std::map<std::tuple<uint64_t, uint32_t, uint64_t>, Root>               m_roots;
	std::map<uint32_t, Root*>                                              m_bases;
	std::set<std::tuple<uint64_t, uint32_t, uint32_t, std::array<uint32_t, 4>>> m_rejected;
	uint32_t m_next   = 0;
	uint32_t m_logged = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_HEAPROWS_H_
