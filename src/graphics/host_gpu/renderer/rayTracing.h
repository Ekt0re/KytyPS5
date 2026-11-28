#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RAYTRACING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RAYTRACING_H_

#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <span>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics {

class BufferCache;
class CommandScheduler;
class RenderContext;
struct GraphicContext;
struct RayTracingTestAccess;

// Mirrors guest BVHs (header formats 1, 3, 6 and 10) into host acceleration structures. The guest
// builds and refits them on the GPU, so a GPU job reads them there; the host reads back only what
// sizes the builds. A guest TLAS instance names a (BLAS, subtree root) pair, mirrored as one BLAS
// over the subtree's prim-record range.
class RayTracing {
public:
	explicit RayTracing(RenderContext& context);
	~RayTracing();
	KYTY_CLASS_NO_COPY(RayTracing);

	// Records the mirror of the guest TLAS header at `tlas`. Returns the device address of its
	// context {u32 valid, u32 guest instance count, u32, u64 TLAS address at +16, u64 guest
	// instance array at +24}, valid once every visible instance has a mirror of its pair as the
	// guest holds it at the dispatch. New or changed BVHs get mirrors a few readbacks later; a
	// call made while every readback area is in flight waits for one if none holds its TLAS, and
	// otherwise reads nothing back. Instance i has custom index i and SBT offset equal to the first
	// prim record of its range, so a hit's record is that plus the primitive. The context and the
	// mirrors serve the commands recorded before the next call; the context, those recorded before
	// the 16th call after it. The guest page table must be
	// synchronized first (PrepareBda). `shader_hash`: the dispatch's program, for the "[RT]"
	// summaries.
	[[nodiscard]] vk::DeviceAddress Prepare(uint64_t tlas, uint64_t shader_hash = 0);

	// A dispatch of a program that walks a guest BVH without a ray query: a walk the recognizer
	// does not drive (see the "[RT]" summaries). `user_data`: its user SGPRs. With
	// KYTY_RT_HEADER_PROBE=1 the first dispatch of each such program (64 at most) also logs the
	// "PSR_BVHL" headers that the 64-bit values at user SGPR pointers + 0..0xf8 point to: the
	// header format the program walks. The probe reads those bytes back from the GPU. On the
	// execution thread, before the dispatch.
	void NoteGuestDispatch(uint64_t shader_hash, std::span<const uint32_t> user_data);

	// Writes the "[RT]" summary of the last 10 s to stdout, when a program walked a guest BVH or a
	// dispatch traced without a valid TLAS context: the dispatches that traced with a ray query,
	// those that walked a guest BVH (with the last program), the TLAS contexts the readbacks found
	// valid and not valid (a header the mirror cannot read, or instances without a mirror yet),
	// and the programs compiled so far (ShaderRecompiler::GetRayTracingPrograms()). Costs a clock
	// read; called at every dispatch that walks a BVH and at every suspend point.
	void ReportCoverage();

	// Called at each suspend point. Every EvictFrames of them, the mirrors no call used since the
	// last pass go: TLASes no call prepared, pairs no visible instance of a prepared TLAS named,
	// and BLASes left without pairs. A BVH used again gets its mirror a few readbacks later.
	void RunGarbageCollector();

	// At each guest flip. The first call over a TLAS after one mirrors it; later calls before the
	// next take that call's context: the guest refits its BVHs once a frame.
	void NoteFlip() noexcept { ++m_flip; }

private:
	friend struct RayTracingTestAccess;

	static constexpr uint32_t Areas       = 8;
	static constexpr uint32_t Contexts    = 16;
	static constexpr uint32_t EvictFrames = 64;

	struct Structure {
		std::unique_ptr<Buffer>      buffer;
		vk::AccelerationStructureKHR handle         = nullptr;
		vk::DeviceAddress            address        = 0;
		vk::DeviceSize               build_scratch  = 0;
		vk::DeviceSize               update_scratch = 0;
		uint32_t                     capacity       = 0; // Primitives or instances.
	};

	struct Blas {
		uint64_t                address;
		std::unique_ptr<Buffer> triangles;
		uint32_t                capacity = 0; // Records `triangles` holds.
		uint32_t                begin    = 0; // Records and node count of the last readback.
		uint32_t                end      = 0;
		uint32_t                nodes    = 0;
		uint32_t                topology = 0; // Hashes of the last complete readback.
		uint32_t                geometry = 0;
		bool                    known    = false;
		bool                    dynamic  = false; // Its geometry changed: updated every call.
		bool                    reshaped = false; // Its records changed: rebuilt every call.
	};

	struct Pair {
		uint32_t                blas;
		uint32_t                root;
		Structure               structure;
		std::unique_ptr<Buffer> boxes; // Box nodes the last walk visited.
		uint32_t                box_capacity = 0;
		uint32_t                box_count    = 0;
		uint32_t                links        = 0; // Their link hash.
		// Range of the last walk, or the header's records at the BLAS root, under BLAS topology
		// `topology`.
		uint32_t count          = 0;
		uint32_t first          = 0;
		uint32_t topology       = 0;
		uint32_t geometry       = 0; // BLAS geometry at the last build or update.
		uint64_t update_serial  = 0;
		bool     built          = false; // The structure holds the range.
		bool     walked         = false;
		bool     walking        = false;
		bool     geometry_known = false;
		bool     used           = true; // Since the last eviction pass.
	};

	struct Tlas {
		Structure               structure;
		std::unique_ptr<Buffer> instances;
		uint32_t                instance_slots = 64;
		bool                    used           = true; // Since the last eviction pass.
	};

	// A dispatch's readback.
	struct Area {
		std::unique_ptr<Buffer> buffer;
		uint64_t                tick           = 0;
		uint64_t                serial         = 0;
		uint64_t                tlas           = 0;
		uint32_t                instance_slots = 0;
		size_t                  pairs          = 0;
		std::vector<uint32_t>   walks;    // Pair per walk result.
		std::vector<uint32_t>   verified; // Pairs whose boxes were hashed again.
		std::vector<uint32_t>   covered;  // Records hashed per BLAS.
		uint64_t                shader_hash = 0; // The program of the dispatch.
	};

	// Counters of the "[RT]" summaries.
	struct Coverage {
		uint64_t traced       = 0; // Dispatches with a ray query.
		uint64_t guest        = 0; // Dispatches that walk a guest BVH.
		uint64_t valid        = 0; // Readbacks with a valid TLAS context.
		uint64_t invalid      = 0; // Readbacks without one.
		uint64_t unreadable   = 0; // Of these, those whose TLAS header the mirror cannot read.
		uint64_t last_guest   = 0;
		uint64_t last_invalid = 0;
	};

	void ProbeHeaders(uint64_t shader_hash, std::span<const uint32_t> user_data);

	void Collect();
	void Apply(Area& area);
	void AddPair(uint64_t blas, uint32_t root);
	void Retire(std::unique_ptr<Buffer>& buffer);
	void Retire(Structure& structure);
	void Create(Structure& structure, const vk::AccelerationStructureGeometryKHR& geometry,
	            vk::AccelerationStructureTypeKHR type, vk::BuildAccelerationStructureFlagsKHR flags,
	            uint32_t capacity);
	// A buffer of at least `size` bytes; one that grows takes the next power of two unless `exact`
	// (acceleration structures and triangle buffers, which keep their size, and there are thousands).
	void Reserve(std::unique_ptr<Buffer>& buffer, vk::BufferUsageFlags usage, uint64_t size,
	             MemoryUsage memory = MemoryUsage::DeviceLocal, bool exact = false);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	BufferCache&                                      m_buffer_cache;
	RenderContext&                                    m_context;
	vk::DescriptorSetLayout                           m_descriptor_layout = nullptr;
	vk::PipelineLayout                                m_layout            = nullptr;
	vk::Pipeline                                      m_walk_pipeline     = nullptr;
	vk::Pipeline                                      m_pipeline          = nullptr;
	uint64_t                                          m_scratch_alignment = 0;
	uint64_t                                          m_max_instances     = 0;
	std::vector<Blas>                                 m_blases;
	std::map<uint64_t, uint32_t>                      m_blas_index;
	std::vector<Pair>                                 m_pairs;
	std::map<std::pair<uint64_t, uint32_t>, uint32_t> m_pair_index;
	std::vector<std::array<uint32_t, 4>>              m_keys; // {BLAS low, high, root, pair}.
	std::map<uint64_t, Tlas>                          m_tlases;
	std::unique_ptr<Buffer>                           m_state;
	// Copies of the context each call returns: a dispatch tracing several TLASes reads one per
	// call, and the next call rewrites m_state. A TLAS keeps a slot of its own while there are
	// free ones; the rest share the slots after them in turn.
	std::unique_ptr<Buffer>                           m_contexts;
	uint32_t                                          m_next_context = 0;
	struct Prepared {
		uint64_t flip = 0;
		uint32_t slot = 0;
	};
	std::map<uint64_t, Prepared>                      m_prepared; // By TLAS header.
	uint64_t                                          m_flip = 1;
	std::unique_ptr<Buffer>                           m_scratch;
	std::array<Area, Areas>                           m_areas;
	uint32_t                                          m_next_area = 0;
	uint64_t                                          m_serial    = 0;
	uint64_t                                          m_applied = 0; // Serial of the last readback.
	uint32_t                                          m_frame   = 0; // Suspend points seen.
	Coverage                                          m_coverage;
	Coverage                                          m_reported; // At the last summary.
	uint64_t                                          m_reported_programs = 0;
	std::chrono::steady_clock::time_point             m_report_time = std::chrono::steady_clock::now();
	std::unordered_set<uint64_t>                      m_probed; // Programs whose headers were probed.
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RAYTRACING_H_
