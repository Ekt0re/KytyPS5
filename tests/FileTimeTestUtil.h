#ifndef KYTY_TESTS_FILE_TIME_TEST_UTIL_H_
#define KYTY_TESTS_FILE_TIME_TEST_UTIL_H_

#include <chrono>
#include <cstdint>
#include <filesystem>

// Sets a host file's write time to `unix_ns` nanoseconds since the Unix epoch.
inline void SetUnixWriteTime(const std::filesystem::path& path, int64_t unix_ns) {
	// Round to the file clock's unit first: MSVC's file_clock counts from 1601, and a nanosecond
	// count from there overflows int64_t for present-day times (then Windows rejects the time).
	const auto time = std::chrono::floor<std::filesystem::file_time_type::duration>(
	    std::chrono::sys_time<std::chrono::nanoseconds> {std::chrono::nanoseconds(unix_ns)});
#ifdef _MSVC_STL_VERSION // no file_clock::from_sys
	const auto file_time = std::chrono::clock_cast<std::chrono::file_clock>(time);
#else
	const auto file_time = std::chrono::file_clock::from_sys(time);
#endif
	std::filesystem::last_write_time(
	    path, std::chrono::floor<std::filesystem::file_time_type::duration>(file_time));
}

#endif // KYTY_TESTS_FILE_TIME_TEST_UTIL_H_
