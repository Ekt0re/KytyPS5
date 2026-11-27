#ifndef EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_
#define EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_

#include "common/abi.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "loader/timer.h" // IWYU pragma: keep

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLED g_print_name

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME_ENABLE(flag) PRINT_NAME_ENABLED = flag;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_DEFINE(name) void name(Loader::SymbolDatabase* s)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_NAME(l, m)                                                                             \
	[[maybe_unused]] static thread_local bool PRINT_NAME_ENABLED = false;                          \
	static constexpr char                     g_library[]        = l;                              \
	static constexpr char                     g_module[]         = m;
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_ADD(l, n, f, t) s->Add({n, l, g_module, t}, reinterpret_cast<uint64_t>(f), "" #f)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_OBJECT(n, f) LIB_ADD(g_library, n, f, Loader::SymbolType::Object)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC(n, f)                                                                             \
	s->Add({n, g_library, g_module, Loader::SymbolType::Func},                                     \
	       Libs::MakeLibThunk<f, Libs::FixedString(g_library), Libs::FixedString("" #f)>(),       \
	       "" #f)
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define LIB_FUNC_IN(l, n, f)                                                                       \
	s->Add({n, l, g_module, Loader::SymbolType::Func},                                             \
	       Libs::MakeLibThunk<f, Libs::FixedString(l), Libs::FixedString("" #f)>(), "" #f)

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define PRINT_NAME()                                                                               \
	if (PRINT_NAME_ENABLED) {                                                                      \
		Libs::PrintName(g_library, g_module, __func__);                                              \
	}

namespace Loader {
class SymbolDatabase;
} // namespace Loader

namespace Libs {

// Set once from KYTY_TRACE_CALLS at InitAll. Off leaves each library export pointing at the
// original function, so a normal run pays nothing for the trace.
inline bool                     g_trace_calls_enabled = false;
inline bool                     g_trace_all           = false;
inline std::vector<std::string> g_trace_libraries;

constexpr size_t kCallRingSize = 256;

struct CallRecord {
	const char* library        = nullptr;
	const char* function       = nullptr;
	uint64_t    args[6]        = {};
	uint64_t    result         = 0;
	uint64_t    guest_ret_addr = 0;
	bool        completed      = false;
};

// Per-thread ring of the most recent calls. The crash report prints it without touching guest
// memory: every field is a host value copied when the call entered and left.
struct CallRing {
	std::array<CallRecord, kCallRingSize> entries {};
	size_t                                next  = 0;
	size_t                                count = 0;

	CallRecord& Push(const char* library, const char* function, const uint64_t (&args)[6],
	                 uint64_t guest_ret_addr) {
		auto& entry          = entries[next];
		entry.library        = library;
		entry.function       = function;
		std::copy_n(args, 6, entry.args);
		entry.result         = 0;
		entry.guest_ret_addr = guest_ret_addr;
		entry.completed      = false;
		next                 = (next + 1) % kCallRingSize;
		if (count < kCallRingSize) {
			count++;
		}
		return entry;
	}

	template <typename Fn>
	void ForEachOldestFirst(Fn&& fn) const {
		const size_t start = (count < kCallRingSize) ? 0 : next;
		for (size_t i = 0; i < count; i++) {
			fn(entries[(start + i) % kCallRingSize], i);
		}
	}
};

constexpr bool ShouldLogPowerOfTwo(uint64_t count) {
	return count != 0 && (count <= 4 || (count & (count - 1)) == 0);
}

const CallRing& GetThreadCallRing();
[[gnu::noinline]] CallRecord& RecordThreadCallEntry(const char* library, const char* function,
                                                    const uint64_t (&args)[6],
                                                    uint64_t guest_ret_addr);
[[gnu::noinline]] bool MatchesTraceFilter(const char* library);
[[gnu::noinline]] bool ShouldLogFailedCall(const void* fn_key, uint64_t result);
[[gnu::noinline]] void LogFilteredCall(const char* library, const char* function,
                                       const uint64_t (&args)[6], uint64_t result);
[[gnu::noinline]] void LogFailedCall(const char* library, const char* function,
                                     const uint64_t (&args)[6], uint64_t result);

template <size_t N>
struct FixedString {
	char data[N] {};
	consteval FixedString(const char (&str)[N]) {
		for (size_t i = 0; i < N; i++) {
			data[i] = str[i];
		}
	}
};

template <typename T>
void PushSysvIntArg(uint64_t (&out)[6], size_t& idx, const T& arg) {
	if (idx >= 6) {
		return;
	}
	if constexpr (std::is_pointer_v<T>) {
		out[idx++] = reinterpret_cast<uint64_t>(arg);
	} else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
		out[idx++] = static_cast<uint64_t>(arg);
	}
}

template <typename Ret>
bool IsFailureReturn(const Ret& value, uint64_t& raw_result) {
	if constexpr (std::is_pointer_v<Ret>) {
		raw_result = reinterpret_cast<uint64_t>(value);
		return value == nullptr;
	} else if constexpr (std::is_integral_v<Ret> && !std::is_same_v<Ret, bool>) {
		if constexpr (sizeof(Ret) <= 4) {
			const auto u32 = static_cast<uint32_t>(value);
			raw_result     = static_cast<uint64_t>(u32);
			return u32 >= 0x80000000u;
		} else {
			raw_result = static_cast<uint64_t>(value);
			return static_cast<int64_t>(raw_result) < 0 ||
			       (raw_result >= 0x80000000ull && raw_result <= 0xffffffffull);
		}
	} else {
		raw_result = 0;
		return false;
	}
}

template <auto Fn, FixedString Library, FixedString Name, typename Sig>
struct LibThunkHelper;

template <auto Fn, FixedString Library, FixedString Name, typename Ret, typename... Args>
struct LibThunkHelper<Fn, Library, Name, Ret (KYTY_SYSV_ABI*)(Args...)> {
	static KYTY_SYSV_ABI Ret Call(Args... args) {
		if (!g_trace_calls_enabled) {
			return Fn(args...);
		}
		uint64_t raw_args[6] {};
		size_t   arg_idx = 0;
		(PushSysvIntArg(raw_args, arg_idx, args), ...);
		const auto guest_ret = reinterpret_cast<uint64_t>(__builtin_return_address(0));
		auto&      record    = RecordThreadCallEntry(Library.data, Name.data, raw_args, guest_ret);
		const bool trace_all = g_trace_all || MatchesTraceFilter(Library.data);
		if constexpr (std::is_void_v<Ret>) {
			Fn(args...);
			record.completed = true;
			if (trace_all) {
				LogFilteredCall(Library.data, Name.data, raw_args, 0);
			}
		} else {
			Ret      ret        = Fn(args...);
			uint64_t raw_result = 0;
			const bool failed   = IsFailureReturn(ret, raw_result);
			record.result       = raw_result;
			record.completed    = true;
			if (trace_all) {
				LogFilteredCall(Library.data, Name.data, raw_args, raw_result);
			}
			if (failed && ShouldLogFailedCall(reinterpret_cast<const void*>(&Call), raw_result)) {
				LogFailedCall(Library.data, Name.data, raw_args, raw_result);
			}
			return ret;
		}
	}

	// Read during InitAll, after g_trace_calls_enabled is set: off, the export is the function itself.
	static uint64_t Addr() {
		return g_trace_calls_enabled ? reinterpret_cast<uint64_t>(&Call) : reinterpret_cast<uint64_t>(Fn);
	}
};

// Variadic guest exports keep their original address: their arguments cannot be captured by the
// fixed-width thunk.
template <auto Fn, FixedString Library, FixedString Name, typename Ret, typename... Args>
struct LibThunkHelper<Fn, Library, Name, Ret (KYTY_SYSV_ABI*)(Args..., ...)> {
	static uint64_t Addr() { return reinterpret_cast<uint64_t>(Fn); }
};

template <auto Fn, FixedString Library, FixedString Name>
uint64_t MakeLibThunk() {
	return LibThunkHelper<Fn, Library, Name, decltype(Fn)>::Addr();
}

// Keep the formatting path from inflating fiber functions' stack frames under LTO.
[[gnu::noinline]] inline void PrintName(const char* library, const char* module, const char* function) {
	if (!Log::IsSilent()) {
		const auto elapsed_ms      = static_cast<uint64_t>(Loader::Timer::GetTimeMs());
		const auto print_name_time = fmt::format(
		    "{:02}:{:02}:{:02}.{:03}", elapsed_ms / 3600000, (elapsed_ms / 60000) % 60,
		    (elapsed_ms / 1000) % 60, elapsed_ms % 1000);
		LOGF_COLOR(Log::Color::Cyan, "[%d][%s] %s::%s::%s()\n",
		           Common::Thread::GetThreadIdUnique(), print_name_time.c_str(), library, module,
		           function);
	}
}

void InitAll(Loader::SymbolDatabase* s);

} // namespace Libs
#endif /* EMULATOR_INCLUDE_EMULATOR_LIBS_LIBS_H_ */
