#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

// Opaque proof that guest bytes covered by a materialization have not changed. Implementations
// are owned by the renderer's existing memory and backing trackers.
class MemoryChangeStamp {
public:
	virtual ~MemoryChangeStamp() = default;
	[[nodiscard]] virtual bool Unchanged() const = 0;
};

class MemoryChangeSource {
public:
	virtual ~MemoryChangeSource() = default;
	// Enables any tracking that must be active before a resource walk starts.
	virtual void BeginTracking() {}
	// A stamp over [address, address + size), or null when the range cannot be proven stable.
	[[nodiscard]] virtual std::shared_ptr<MemoryChangeStamp> Track(uint64_t address, uint64_t size,
	                                                              uint32_t max_pages) = 0;
};

enum class UniformFillKind { None, Buffer, Image };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;
	uint32_t                 bounded_elements = 0;

	bool operator==(const UniformFill&) const = default;
};

struct ResourceSnapshot {
	struct IndirectBufferTable {
		uint32_t                     root_resource = 0;
		uint32_t                     header_word   = 0;
		std::vector<DescriptorValue> entries;
		std::vector<uint32_t>        class_index;
	};

	std::vector<DescriptorValue>     buffers;
	std::vector<DescriptorValue>     images;
	std::vector<DescriptorValue>     samplers;
	std::vector<uint32_t>            flattened_srt;
	std::vector<uint32_t>            user_data;
	std::vector<IndirectBufferTable> indirect_buffer_tables;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
