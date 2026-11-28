#ifndef KYTY_COMMON_FP_ENV_H_
#define KYTY_COMMON_FP_ENV_H_

#include "common/common.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <float.h>
#include <immintrin.h>
#endif

namespace Common {

struct FpEnv {
	uint32_t mxcsr;
	uint16_t control;
};

// What every guest thread starts with: MXCSR flush-to-zero, denormals-are-zero, round to nearest,
// all exceptions masked; x87 64-bit precision, round to nearest, all exceptions masked.
inline constexpr FpEnv GuestFpEnv {0x9fc0, 0x037f};

// x86-64 only: the MXCSR intrinsics work with clang-cl and gcc/clang; the x87 control word uses
// the CRT's word on Windows, where inline AT&T assembly is not available.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
inline uint32_t ReadMxcsr() {
	return _mm_getcsr();
}
inline void WriteMxcsr(uint32_t value) {
	_mm_setcsr(value);
}
// MSVC's control word scrambles the six exception masks and keeps precision and rounding in
// different bits from the x87 word. Convert only the fields GuestFpEnv specifies.
inline uint16_t MsvcToX87(unsigned word) {
	uint16_t control = 0;
	if ((word & 0x10u) != 0) control |= 1u << 0; // invalid
	if ((word & 0x02u) != 0) control |= 1u << 1; // denormal
	if ((word & 0x08u) != 0) control |= 1u << 2; // divide by zero
	if ((word & 0x04u) != 0) control |= 1u << 3; // overflow
	if ((word & 0x01u) != 0) control |= 1u << 4; // underflow
	if ((word & 0x20u) != 0) control |= 1u << 5; // precision
	if ((word & 0x30000u) == 0x10000u) control |= 0x200u;  // double precision
	if ((word & 0x30000u) != 0x10000u && (word & 0x30000u) != 0x20000u) control |= 0x300u;
	control |= static_cast<uint16_t>(((word >> 8u) & 3u) << 10u); // rounding
	return control;
}
inline unsigned X87ToMsvc(uint16_t control) {
	unsigned word = 0;
	if ((control & (1u << 0)) != 0) word |= 0x10u;
	if ((control & (1u << 1)) != 0) word |= 0x02u;
	if ((control & (1u << 2)) != 0) word |= 0x08u;
	if ((control & (1u << 3)) != 0) word |= 0x04u;
	if ((control & (1u << 4)) != 0) word |= 0x01u;
	if ((control & (1u << 5)) != 0) word |= 0x20u;
	if ((control & 0x300u) == 0u) word |= 0x20000u;   // single precision
	else if ((control & 0x300u) == 0x200u) word |= 0x10000u; // double
	word |= ((control >> 10u) & 3u) << 8u; // rounding
	return word;
}
inline uint16_t ReadX87Control() {
	unsigned word = 0;
	_controlfp_s(&word, 0, 0);
	return MsvcToX87(word);
}
inline void WriteX87Control(uint16_t control) {
	const unsigned mask = _MCW_EM | _MCW_RC | _MCW_PC;
	_controlfp_s(nullptr, X87ToMsvc(control), mask);
}
#elif defined(__x86_64__) || defined(_M_X64)
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
