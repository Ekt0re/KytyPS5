#ifndef KYTY_COMMON_FP_ENV_H_
#define KYTY_COMMON_FP_ENV_H_

#include "common/common.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>


namespace Common {

struct FpEnv {
	uint32_t mxcsr;
	uint16_t control;
};

// What every guest thread starts with: MXCSR flush-to-zero, denormals-are-zero, round to nearest,
// all exceptions masked; x87 64-bit precision, round to nearest, all exceptions masked.
inline constexpr FpEnv GuestFpEnv {0x9fc0, 0x037f};

// x86-64: the same GNU inline assembly as before on every host compiler the tree supports
// (clang, gcc, clang-cl). The MSVC CRT's _controlfp_s cannot stand in on x64 Windows: it sets the
// SSE control state and does not support the x87 precision field.
#if defined(__x86_64__) || defined(_M_X64)
inline uint32_t ReadMxcsr() {
	uint32_t value {};
	asm volatile("stmxcsr %0" : "=m"(value) : : "memory");
	return value;
}
inline void WriteMxcsr(uint32_t value) {
	asm volatile("ldmxcsr %0" : : "m"(value) : "memory");
}
inline uint16_t ReadX87Control() {
	uint16_t control {};
	asm volatile("fnstcw %0" : "=m"(control) : : "memory");
	return control;
}
inline void WriteX87Control(uint16_t control) {
	asm volatile("fldcw %0" : : "m"(control) : "memory");
}
#else
// Explicitly unavailable: the guest is x86-64 only, so there is no FP control state to check
// here. Return and accept the guest default instead of pretending to read hardware.
inline uint32_t ReadMxcsr() { return GuestFpEnv.mxcsr; }
inline void WriteMxcsr(uint32_t value) { (void)value; }
inline uint16_t ReadX87Control() { return GuestFpEnv.control; }
inline void WriteX87Control(uint16_t control) { (void)control; }
#endif

inline FpEnv ReadFpEnv() {
	return {ReadMxcsr(), ReadX87Control()};
}

inline void WriteFpEnv(FpEnv env) {
	WriteMxcsr(env.mxcsr);
	WriteX87Control(env.control);
}

// KYTY_FP_ENV_CHECK=1: report the control state at thread start, library calls and faults. The
// guest signal-handler fix (setting the state before a guest handler) is always active.
inline bool FpEnvCheckEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_FP_ENV_CHECK");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return enabled;
}

// The chosen part of the state, without what computation leaves behind: MXCSR without its six
// sticky exception flags (bits 0-5); the x87 control word's masks (0-5), precision (8-9) and
// rounding (10-11), without the reserved and infinity-control bits.
[[nodiscard]] constexpr uint32_t MxcsrControl(uint32_t mxcsr) {
	return mxcsr & 0xffc0u;
}
[[nodiscard]] constexpr uint16_t X87Control(uint16_t control) {
	return static_cast<uint16_t>(control & 0x0f3fu);
}
[[nodiscard]] constexpr bool SameFpControl(FpEnv a, FpEnv b) {
	return MxcsrControl(a.mxcsr) == MxcsrControl(b.mxcsr) &&
	       X87Control(a.control) == X87Control(b.control);
}

// Places the FP control state is checked at (one report per site and thread when enabled).
enum class FpCheckSite : uint32_t {
	ThreadStart    = 0,
	ThreadCreator  = 1,
	LibraryEntry   = 2,
	LibraryChanged = 3,
	FaultReturn    = 4,
	GuestFault     = 5,
	SignalHandler  = 6,
	Count          = 7,
};

inline thread_local uint32_t t_fp_check_reported = 0;

// True the first time on this thread that `site` sees `env` differ from `expected` in its chosen
// part; the caller then logs. Costs one compare while the state is as expected.
[[nodiscard]] inline bool FpControlFirstDeviation(FpCheckSite site, FpEnv env,
                                                  FpEnv expected = GuestFpEnv) {
	if (!FpEnvCheckEnabled() || SameFpControl(env, expected)) {
		return false;
	}
	const uint32_t bit = 1u << static_cast<uint32_t>(site);
	if ((t_fp_check_reported & bit) != 0) {
		return false;
	}
	t_fp_check_reported |= bit;
	return true;
}

[[nodiscard]] constexpr const char* FpCheckSiteName(FpCheckSite site) {
	switch (site) {
		case FpCheckSite::ThreadStart: return "thread-start";
		case FpCheckSite::ThreadCreator: return "thread-creator";
		case FpCheckSite::LibraryEntry: return "library-entry";
		case FpCheckSite::LibraryChanged: return "library-changed";
		case FpCheckSite::FaultReturn: return "fault-return";
		case FpCheckSite::GuestFault: return "guest-fault";
		case FpCheckSite::SignalHandler: return "signal-handler";
		default: return "?";
	}
}

} // namespace Common

#endif /* KYTY_COMMON_FP_ENV_H_ */
