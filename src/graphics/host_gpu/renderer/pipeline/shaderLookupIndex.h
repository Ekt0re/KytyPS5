#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLOOKUPINDEX_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLOOKUPINDEX_H_

#include "graphics/shader/recompiler/ir/Serialize.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics {

// Runtime-only lookup keys. They are never written to the shader store: a stored module's
// specialization is decoded with the record that already names its program.
template <class Value> uint64_t HashShaderValue(const Value& value) {
	ShaderRecompiler::IR::Writer writer;
	auto                         copy = value;
	ShaderRecompiler::IR::IoAll(writer, copy);
	return XXH3_64bits(writer.bytes.data(), writer.bytes.size());
}

inline uint64_t HashShaderWords(std::span<const uint32_t> words) {
	return XXH3_64bits(words.data(), words.size_bytes());
}

// Hashes only select candidates. The caller still compares the complete specialization, push-data
// placement and callee code, so a collision cannot change which module a lookup returns.
template <class Module> class ShaderLookupIndex {
public:
	void Add(uint64_t hash, Module* module) { m_entries.emplace(hash, module); }

	template <class Match> Module* Find(uint64_t hash, Match&& match) const {
		const auto [first, last] = m_entries.equal_range(hash);
		for (auto entry = first; entry != last; ++entry) {
			if (match(*entry->second)) return entry->second;
		}
		return nullptr;
	}

	[[nodiscard]] bool   Empty() const { return m_entries.empty(); }
	[[nodiscard]] size_t Size() const { return m_entries.size(); }

private:
	std::unordered_multimap<uint64_t, Module*> m_entries;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERLOOKUPINDEX_H_
