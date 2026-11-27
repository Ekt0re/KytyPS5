#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_RUNTIMETOGGLES_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_RUNTIMETOGGLES_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// Diagnostic (KYTY_TOGGLE_FILE=path): switches read from a file of "name=0|1" lines, re-read at
// most every 250 ms, so one game run can bisect a rendering problem. Unknown or missing names
// read as 0. Known names:
//   exact     - GPU buffer writes refresh only images starting at the write (overlap refresh off)
//   pred_off  - SET_PREDICATION never skips draws or dispatches
//   trace_shaders - log draws and dispatches with their shader hashes (see shaderTrace.h)
//   walker_guard  - SRT walker loads from read-watched pages skip the page fault (default on;
//                   only a walker_guard line in the file changes it)
namespace Common::RuntimeToggles {

namespace Detail {
inline const char* Path() {
	static const char* path = std::getenv("KYTY_TOGGLE_FILE");
	return path;
}
inline std::mutex                       g_mutex;
inline std::string                      g_text;
inline std::chrono::steady_clock::time_point g_next {};
} // namespace Detail

inline bool Enabled() {
	return Detail::Path() != nullptr && Detail::Path()[0] != 0;
}

// 1 or 0 as the file says, -1 when the file has no line for `name` (or there is no file).
inline int Find(const char* name) {
	if (!Enabled()) return -1;
	std::scoped_lock lock {Detail::g_mutex};
	const auto now = std::chrono::steady_clock::now();
	if (now >= Detail::g_next) {
		Detail::g_next = now + std::chrono::milliseconds(250);
		std::string text;
		if (FILE* file = std::fopen(Detail::Path(), "r")) {
			std::array<char, 512> chunk {};
			size_t                read = 0;
			while ((read = std::fread(chunk.data(), 1, chunk.size(), file)) > 0) text.append(chunk.data(), read);
			std::fclose(file);
		}
		if (text != Detail::g_text) {
			Detail::g_text = text;
			std::printf("Runtime toggles: %s\n", text.c_str());
		}
	}
	const std::string key = std::string(name) + "=";
	size_t            pos = 0;
	while ((pos = Detail::g_text.find(key, pos)) != std::string::npos) {
		if (pos == 0 || Detail::g_text[pos - 1] == '\n') {
			return Detail::g_text.compare(pos + key.size(), 1, "1") == 0 ? 1 : 0;
		}
		pos += key.size();
	}
	return -1;
}

inline bool Get(const char* name) {
	return Find(name) == 1;
}

} // namespace Common::RuntimeToggles

#endif
