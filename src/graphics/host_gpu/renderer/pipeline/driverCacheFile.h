#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEFILE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEFILE_H_

#include "common/common.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

// A read-only view of the driver's pipeline cache file. Mapping keeps the bytes in the kernel's
// page cache instead of making a second full copy on the heap, which the driver copies again
// into its cache object. On a platform (or a file) that cannot be mapped, the original
// read-into-owned-memory behaviour is used, so a failure is never fatal.
namespace Libs::Graphics {

class DriverCacheFile {
public:
	const uint8_t* data = nullptr;
	size_t         size = 0;

	DriverCacheFile() = default;
	DriverCacheFile(const DriverCacheFile&)            = delete;
	DriverCacheFile& operator=(const DriverCacheFile&) = delete;

	DriverCacheFile(DriverCacheFile&& other) noexcept {
		*this = std::move(other);
	}

	DriverCacheFile& operator=(DriverCacheFile&& other) noexcept {
		if (this == &other) {
			return *this;
		}
		Release();
		owned    = std::move(other.owned);
		data     = std::exchange(other.data, nullptr);
		size     = std::exchange(other.size, 0);
		mapped   = std::exchange(other.mapped, nullptr);
		map_size = std::exchange(other.map_size, 0);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		file_handle = std::exchange(other.file_handle, INVALID_HANDLE_VALUE);
		map_handle  = std::exchange(other.map_handle, nullptr);
#endif
		return *this;
	}

	~DriverCacheFile() {
		Release();
	}

	void Release() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (mapped != nullptr) {
			UnmapViewOfFile(mapped);
		}
		if (map_handle != nullptr) {
			CloseHandle(map_handle);
		}
		if (file_handle != INVALID_HANDLE_VALUE) {
			CloseHandle(file_handle);
		}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		if (mapped != nullptr) {
			munmap(mapped, map_size);
		}
#endif
		owned.clear();
		owned.shrink_to_fit();
		mapped   = nullptr;
		map_size = 0;
		data     = nullptr;
		size     = 0;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		file_handle = INVALID_HANDLE_VALUE;
		map_handle  = nullptr;
#endif
	}

	static DriverCacheFile Map(const std::filesystem::path& path) {
		DriverCacheFile result;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		result.file_handle = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
		                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
		if (result.file_handle == INVALID_HANDLE_VALUE) {
			return ReadIntoOwned(path);
		}
		LARGE_INTEGER length {};
		if (!GetFileSizeEx(result.file_handle, &length) || length.QuadPart <= 0 ||
		    static_cast<uint64_t>(length.QuadPart) > SIZE_MAX) {
			result.Release();
			return ReadIntoOwned(path);
		}
		result.map_handle = CreateFileMappingW(result.file_handle, nullptr, PAGE_READONLY, 0, 0,
		                                       nullptr);
		if (result.map_handle == nullptr) {
			result.Release();
			return ReadIntoOwned(path);
		}
		result.mapped = MapViewOfFile(result.map_handle, FILE_MAP_READ, 0, 0, 0);
		if (result.mapped == nullptr) {
			result.Release();
			return ReadIntoOwned(path);
		}
		result.map_size = static_cast<size_t>(length.QuadPart);
		result.data     = static_cast<const uint8_t*>(result.mapped);
		result.size     = result.map_size;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
		if (fd < 0) {
			return ReadIntoOwned(path);
		}
		struct stat info {};
		if (fstat(fd, &info) != 0 || info.st_size <= 0) {
			::close(fd);
			return ReadIntoOwned(path);
		}
		void* map =
		    mmap(nullptr, static_cast<size_t>(info.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
		::close(fd);
		if (map == MAP_FAILED) {
			return ReadIntoOwned(path);
		}
		result.mapped   = map;
		result.map_size = static_cast<size_t>(info.st_size);
		result.data     = static_cast<const uint8_t*>(map);
		result.size     = result.map_size;
#else
		// No mapping API on this platform: explicit fallback, never a silent no-op.
		return ReadIntoOwned(path);
#endif
		return result;
	}

private:
	static DriverCacheFile ReadIntoOwned(const std::filesystem::path& path) {
		DriverCacheFile result;
		std::ifstream   file(path, std::ios::binary | std::ios::ate);
		if (!file) {
			return result;
		}
		const auto length = static_cast<size_t>(file.tellg());
		result.owned.resize(length);
		file.seekg(0);
		file.read(reinterpret_cast<char*>(result.owned.data()), static_cast<std::streamsize>(length));
		if (!file) {
			result.owned.clear();
			return result;
		}
		result.data = result.owned.data();
		result.size = length;
		return result;
	}

	void*  mapped   = nullptr;
	size_t map_size = 0;
	std::vector<uint8_t> owned;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE file_handle = INVALID_HANDLE_VALUE;
	HANDLE map_handle  = nullptr;
#endif
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DRIVERCACHEFILE_H_
