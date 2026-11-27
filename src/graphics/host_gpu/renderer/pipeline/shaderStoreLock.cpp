#include "graphics/host_gpu/renderer/pipeline/shaderStoreLock.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

bool ShaderStoreLock::Acquire(const std::filesystem::path& file) {
	if (Held()) return true;
	auto path = file;
	path += ".lock";
#ifdef _WIN32
	// No sharing: a second open of the file, in any process, fails while this handle lives.
	const HANDLE handle = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0,
	                                  nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
	if (handle == INVALID_HANDLE_VALUE) return false;
	m_handle = reinterpret_cast<intptr_t>(handle);
#else
	const int fd = open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
	if (fd < 0) return false;
	// flock locks belong to the open file description: a second open in this process conflicts
	// as one in another process does.
	if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
		close(fd);
		return false;
	}
	m_handle = fd;
#endif
	return true;
}

void ShaderStoreLock::Release() {
	if (!Held()) return;
#ifdef _WIN32
	CloseHandle(reinterpret_cast<HANDLE>(m_handle));
#else
	close(static_cast<int>(m_handle));
#endif
	m_handle = Invalid;
}

} // namespace Libs::Graphics
