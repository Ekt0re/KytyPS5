#ifndef KYTY_COMMON_FAULT_REPORT_H_
#define KYTY_COMMON_FAULT_REPORT_H_

#include "common/common.h"

#include <cstddef>
#include <cstdint>

// Host-side context for a fatal fault: module, symbol and offset for pc and the return-address
// chain, plus the thread's name and tag. On Linux module and symbol tables are built outside the
// handler so the handler allocates nothing and takes no lock of its own.
namespace Common::FaultReport {

// Notes what the calling host thread does, for the fault report. `tag` must be a string literal.
// With `name` the host thread is also given that name (at most 15 characters on Linux).
void SetThreadTag(const char* tag, const char* name = nullptr) noexcept;

// Reads loaded modules (and, once, main executable symbols on Linux or DbgHelp setup on Windows).
void RefreshModules();

// Remembers the calling thread's stack limits so the frame-pointer walk stays inside them.
void RegisterThreadStack();

struct Sink {
	void (*write)(void* user, const char* text) = nullptr;
	void* user                                  = nullptr;
};

struct Registers {
	uint64_t pc             = 0;
	uint64_t rsp            = 0;
	uint64_t rbp            = 0;
	void*    native_context = nullptr;
};

// "module+0xoffset [symbol+0xoffset]", "module+0xoffset", or "not in a loaded module".
void FormatAddress(uint64_t address, char* out, size_t size);

// Prints the thread name and tag, pc, and return-address chain through `sink`.
void WriteHostContext(const Registers& regs, const Sink& sink);

} // namespace Common::FaultReport

#endif /* KYTY_COMMON_FAULT_REPORT_H_ */
