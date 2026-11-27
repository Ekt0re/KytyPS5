#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORELOCK_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORELOCK_H_

#include <cstdint>
#include <filesystem>

namespace Libs::Graphics {

// One writer per shader store or recipes file. Both are append-only files that a run truncates
// to the records it could read and then appends to at its own offset: a second process doing the
// same (a game started while a precompile of its stamp runs) overwrites the first one's records or
// leaves a hole of zeros, and the next run's load stops there and truncates everything after it
// (2026-11-27: a 1,122 s precompile of 4,825 modules was cut back to 2,885 that way). The lock is
// <file>.lock, held from Acquire() until Release() or the end of the process (an OS lock: a
// crashed writer never leaves it behind). Without it a run reads the file but writes nothing.
class ShaderStoreLock {
public:
	ShaderStoreLock() = default;
	~ShaderStoreLock() { Release(); }
	ShaderStoreLock(const ShaderStoreLock&)            = delete;
	ShaderStoreLock& operator=(const ShaderStoreLock&) = delete;

	// False when another writer (any process, or another lock in this one) holds it. Does not
	// wait. Acquiring a lock already held by this object returns true.
	bool Acquire(const std::filesystem::path& file);
	void Release();
	[[nodiscard]] bool Held() const { return m_handle != Invalid; }

private:
	static constexpr intptr_t Invalid = -1;
	intptr_t                  m_handle = Invalid;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_SHADERSTORELOCK_H_
