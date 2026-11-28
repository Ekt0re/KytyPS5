#include "common/assert.h"

#include "common/memAccount.h"
#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <algorithm>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <utility>

namespace Common {

thread_local std::jmp_buf* g_dbg_recover = nullptr;

namespace {

thread_local char            t_fatal_text[1024] = {};
thread_local FatalDiagnostic t_fatal_diagnostic {};

} // namespace

const FatalDiagnostic& LastFatalDiagnostic() {
	return t_fatal_diagnostic;
}

[[noreturn]] static void DbgReport(const char* title, std::string_view text, const char* file,
                                   int line) {
	Log::WriteToConsoleAndLog(fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL,
	                                      title, text, file, line));
	const auto size = std::min(text.size(), sizeof(t_fatal_text) - 1);
	std::memcpy(t_fatal_text, text.data(), size);
	t_fatal_text[size]     = 0;
	t_fatal_diagnostic     = {t_fatal_text, file != nullptr ? file : "", line};
	if (auto* recover = std::exchange(g_dbg_recover, nullptr); recover != nullptr) {
		std::longjmp(*recover, 1);
	}
	MemAccount::NoteExitReason("fatal");
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
