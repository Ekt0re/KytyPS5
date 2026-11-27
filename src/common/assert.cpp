#include "common/assert.h"

#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <csetjmp>
#include <fmt/format.h>
#include <utility>

namespace Common {

thread_local std::jmp_buf* g_dbg_recover = nullptr;

[[noreturn]] static void DbgReport(const char* title, std::string_view text, const char* file,
                                   int line) {
	Log::WriteToConsoleAndLog(fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL,
	                                      title, text, file, line));
	if (auto* recover = std::exchange(g_dbg_recover, nullptr); recover != nullptr) {
		std::longjmp(*recover, 1);
	}
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(321);
}

void DbgExitIfHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr), file,
	          line);
}

void DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

void DbgExitHandler(const char* file, int line, std::string_view text) {
	DbgReport("--- Error ---", text, file, line);
}

} // namespace Common
