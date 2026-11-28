#include "libs/libs.h"

#include "common/logging/log.h"

#include <atomic>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string_view>
#include <unordered_map>

// The out-of-line half of the library-call thunk (libs.h, LibThunkHelper): the KYTY_TRACE_CALLS
// ring, filter and logs and the KYTY_FP_ENV_CHECK reports. Kept apart from libs.cpp, which
// registers every library, so the thunk can be linked into unit tests.

namespace Libs {

namespace {

thread_local CallRing t_call_ring;

struct FailedCallKey {
	const void* fn     = nullptr;
	uint64_t    result = 0;

	bool operator==(const FailedCallKey&) const = default;
};

struct FailedCallKeyHash {
	size_t operator()(const FailedCallKey& k) const {
		const auto a = reinterpret_cast<uintptr_t>(k.fn);
		return static_cast<size_t>(a ^ (k.result * 0x9e3779b97f4a7c15ull));
	}
};

std::mutex                                                   g_fail_mutex;
std::unordered_map<FailedCallKey, uint64_t, FailedCallKeyHash> g_fail_counts;

bool AsciiEqualInsensitive(std::string_view a, std::string_view b) {
	if (a.size() != b.size()) {
		return false;
	}
	for (size_t i = 0; i < a.size(); i++) {
		if (std::tolower(static_cast<unsigned char>(a[i])) !=
		    std::tolower(static_cast<unsigned char>(b[i]))) {
			return false;
		}
	}
	return true;
}

} // namespace

const CallRing& GetThreadCallRing() {
	return t_call_ring;
}

CallRecord& RecordThreadCallEntry(const char* library, const char* function,
                                  const uint64_t (&args)[6], uint64_t guest_ret_addr) {
	return t_call_ring.Push(library, function, args, guest_ret_addr);
}

bool MatchesTraceFilter(const char* library) {
	if (!g_trace_calls_enabled || library == nullptr || g_trace_all) {
		return false;
	}
	const std::string_view lib(library);
	for (const auto& entry: g_trace_libraries) {
		if (AsciiEqualInsensitive(lib, entry)) {
			return true;
		}
	}
	return false;
}

bool ShouldLogFailedCall(const void* fn_key, uint64_t result) {
	std::scoped_lock lock(g_fail_mutex);
	const auto       count = ++g_fail_counts[FailedCallKey {fn_key, result}];
	return ShouldLogPowerOfTwo(count);
}

void LogFilteredCall(const char* library, const char* function, const uint64_t (&args)[6],
                     uint64_t result) {
	std::printf("[%d][call] %s::%s(0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx) = 0x%llx\n",
	            Common::Thread::GetThreadIdUnique(), library, function,
	            static_cast<unsigned long long>(args[0]), static_cast<unsigned long long>(args[1]),
	            static_cast<unsigned long long>(args[2]), static_cast<unsigned long long>(args[3]),
	            static_cast<unsigned long long>(args[4]), static_cast<unsigned long long>(args[5]),
	            static_cast<unsigned long long>(result));
}

void LogFailedCall(const char* library, const char* function, const uint64_t (&args)[6],
                   uint64_t result) {
	std::printf("[%d][fail] %s::%s(0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx, 0x%llx) = 0x%llx\n",
	            Common::Thread::GetThreadIdUnique(), library, function,
	            static_cast<unsigned long long>(args[0]), static_cast<unsigned long long>(args[1]),
	            static_cast<unsigned long long>(args[2]), static_cast<unsigned long long>(args[3]),
	            static_cast<unsigned long long>(args[4]), static_cast<unsigned long long>(args[5]),
	            static_cast<unsigned long long>(result));
}

namespace {
std::atomic<uint64_t>                       g_float_return_calls {0};
std::mutex                                  g_float_return_mutex;
std::unordered_map<const void*, uint64_t>   g_float_return_logged;

bool RateLimited(uint64_t count) {
	return count <= 4 || (count & (count - 1)) == 0;
}
} // namespace

uint64_t FloatReturnCalls() {
	return g_float_return_calls.load(std::memory_order_relaxed);
}

void NoteFloatReturn(const void* fn_key, const char* library, const char* function, double value) {
	const auto calls = g_float_return_calls.fetch_add(1, std::memory_order_relaxed) + 1;
	if (value == 0.0 || !std::isfinite(value)) {
		uint64_t count = 0;
		{
			std::scoped_lock lock(g_float_return_mutex);
			count = ++g_float_return_logged[fn_key];
		}
		if (RateLimited(count) && count <= 64) {
			LOGF("[fp-return] thread=%d %s::%s returned %g (this result #%" PRIu64
			     ", float-returning library calls so far %" PRIu64 ")\n",
			     Common::Thread::GetThreadIdUnique(), library, function, value, count, calls);
		}
	}
}

void ReportCallFpEnv(Common::FpCheckSite site, const char* library, const char* function,
                     Common::FpEnv env, Common::FpEnv expected) {
	LOGF("[fpenv] site=%s thread=%d mxcsr=0x%04x x87=0x%04x expected mxcsr=0x%04x x87=0x%04x "
	     "call=%s::%s (first on this thread)\n",
	     Common::FpCheckSiteName(site), Common::Thread::GetThreadIdUnique(), env.mxcsr,
	     env.control, expected.mxcsr, expected.control, library, function);
}

} // namespace Libs
