#include "libs/libs.h"

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "libs/errno.h"
#include "loader/symbolDatabase.h"

#include <atomic>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace Libs {

namespace LibContentDelete {
LIB_DEFINE(InitContentDelete_1);
} // namespace LibContentDelete

namespace LibContentExport {
LIB_DEFINE(InitContentExport_1);
} // namespace LibContentExport

namespace LibContentSearch {
LIB_DEFINE(InitContentSearch_1);
} // namespace LibContentSearch

namespace VideoDec2 {
LIB_DEFINE(InitVideoDec2_1);
} // namespace VideoDec2

namespace Vdecsw {
LIB_DEFINE(InitVdecsw_1);
} // namespace Vdecsw

namespace LibMouse {
LIB_DEFINE(InitMouse_1);
} // namespace LibMouse

namespace LibKeyboard {
LIB_DEFINE(InitKeyboard_1);
} // namespace LibKeyboard

namespace Ime {
LIB_DEFINE(InitPlatform_1_Ime);
} // namespace Ime

namespace LibUlt {
LIB_DEFINE(InitUlt_1);
} // namespace LibUlt

namespace LibPsml {
LIB_DEFINE(InitPsml_1);
} // namespace LibPsml

namespace LibCes {
LIB_DEFINE(InitCes_1);
} // namespace LibCes

namespace LibC {
LIB_DEFINE(InitLibC_1);
} // namespace LibC

namespace LibAmpr {
LIB_DEFINE(InitAmpr_1);
} // namespace LibAmpr

namespace Coredump {
LIB_DEFINE(InitCoredump_1);
} // namespace Coredump

namespace LibRazorCpu {
LIB_DEFINE(InitRazorCpu_1);
} // namespace LibRazorCpu

namespace Fiber {
LIB_DEFINE(InitFiber_1);
} // namespace Fiber

namespace LibRtc {
LIB_DEFINE(InitRtc_1);
} // namespace LibRtc

namespace LibGen5 {
LIB_DEFINE(InitVideoOut_1);
namespace VrrStatus {
LIB_DEFINE(InitVideoOutVrrStatus_1);
} // namespace VrrStatus
} // namespace LibGen5

LIB_DEFINE(InitAppContent_1);
LIB_DEFINE(InitAudio_1);
LIB_DEFINE(InitBluetoothHid_1);
LIB_DEFINE(InitConvertKeycode_1);
LIB_DEFINE(InitDbgAddressSanitizer_1);
LIB_DEFINE(InitDeviceService_1);
LIB_DEFINE(InitDialog_1);
LIB_DEFINE(InitFont_1);
LIB_DEFINE(InitFontFt_1);
LIB_DEFINE(InitAgcDriver_1);
LIB_DEFINE(InitHmd2_1);
LIB_DEFINE(InitJpegDec_1);
LIB_DEFINE(InitJpegEnc_1);
LIB_DEFINE(InitVrTracker2_1);
LIB_DEFINE(InitLibKernel_1);
LIB_DEFINE(InitNet_1);
LIB_DEFINE(InitPad_1);
LIB_DEFINE(InitPlayGo_1);
LIB_DEFINE(InitPngDec_1);
LIB_DEFINE(InitPngEnc_1);
LIB_DEFINE(InitPlatform_1);
LIB_DEFINE(InitRudp_1);
LIB_DEFINE(InitSaveData_1);
LIB_DEFINE(InitShare_1);
LIB_DEFINE(InitSysmodule_1);
LIB_DEFINE(InitSystemService_1);
LIB_DEFINE(InitTextToSpeech2_1);
LIB_DEFINE(InitUsbd_1);
LIB_DEFINE(InitUserService_1);
LIB_DEFINE(InitWebBrowserDialog_1);

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

void InitAll(Loader::SymbolDatabase* s) {
	const auto& filter  = Config::GetTraceCallsFilter();
	g_trace_calls_enabled = filter.enabled;
	g_trace_all           = filter.all;
	g_trace_libraries     = filter.libraries;
	InitAudio_1(s);
	InitConvertKeycode_1(s);
	LibAmpr::InitAmpr_1(s);
	InitAppContent_1(s);
	Coredump::InitCoredump_1(s);
	LibContentDelete::InitContentDelete_1(s);
	LibContentExport::InitContentExport_1(s);
	LibContentSearch::InitContentSearch_1(s);
	LibC::InitLibC_1(s);
	LibCes::InitCes_1(s);
	InitDbgAddressSanitizer_1(s);
	LibRazorCpu::InitRazorCpu_1(s);
	InitDialog_1(s);
	Fiber::InitFiber_1(s);
	InitFont_1(s);
	InitFontFt_1(s);
	InitAgcDriver_1(s);
	InitBluetoothHid_1(s);
	InitDeviceService_1(s);
	InitUsbd_1(s);
	InitHmd2_1(s);
	InitJpegDec_1(s);
	InitJpegEnc_1(s);
	InitVrTracker2_1(s);
	InitLibKernel_1(s);
	LibMouse::InitMouse_1(s);
	LibKeyboard::InitKeyboard_1(s);
	Ime::InitPlatform_1_Ime(s);
	InitNet_1(s);
	InitPad_1(s);
	InitPlayGo_1(s);
	LibPsml::InitPsml_1(s);
	InitPngDec_1(s);
	InitPngEnc_1(s);
	InitPlatform_1(s);
	InitRudp_1(s);
	LibRtc::InitRtc_1(s);
	InitSaveData_1(s);
	InitShare_1(s);
	InitSysmodule_1(s);
	InitSystemService_1(s);
	InitTextToSpeech2_1(s);
	LibUlt::InitUlt_1(s);
	InitUserService_1(s);
	VideoDec2::InitVideoDec2_1(s);
	Vdecsw::InitVdecsw_1(s);
	LibGen5::InitVideoOut_1(s);
	LibGen5::VrrStatus::InitVideoOutVrrStatus_1(s);
	InitWebBrowserDialog_1(s);
}

namespace LibContentExport {

LIB_NAME("ContentExport", "ContentExport");

namespace ContentExport {

constexpr int CONTENT_EXPORT_ERROR_INVALID_PARAM = -2137182186; /* 0x809D3016 */

struct ContentExportInitParam2 {
	void*   malloc_func;
	void*   free_func;
	void*   user_data;
	size_t  buffer_size;
	int64_t reserved0;
	int64_t reserved1;
};

static bool g_initialized = false;

static int KYTY_SYSV_ABI ContentExportInit2(const ContentExportInitParam2* init_param) {
	PRINT_NAME();

	LOGF("\t init_param  = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(init_param));

	if (init_param == nullptr) {
		return CONTENT_EXPORT_ERROR_INVALID_PARAM;
	}

	LOGF("\t malloc_func = 0x%016" PRIx64 "\n"
	     "\t free_func   = 0x%016" PRIx64 "\n"
	     "\t user_data   = 0x%016" PRIx64 "\n"
	     "\t buffer_size = %" PRIu64 "\n",
	     reinterpret_cast<uint64_t>(init_param->malloc_func),
	     reinterpret_cast<uint64_t>(init_param->free_func),
	     reinterpret_cast<uint64_t>(init_param->user_data),
	     static_cast<uint64_t>(init_param->buffer_size));

	g_initialized = true;

	return OK;
}

} // namespace ContentExport

LIB_DEFINE(InitContentExport_1) {
	LIB_FUNC("0GnN4QCgIfs", ContentExport::ContentExportInit2);
}

} // namespace LibContentExport

namespace LibContentSearch {

LIB_NAME("ContentSearch", "ContentSearch");

namespace ContentSearch {

constexpr int CONTENT_SEARCH_ERROR_INVALID_PARAM = -2137190397; /* 0x809D1003 */

struct ContentSearchInitParam {
	size_t memory_size;
};

static bool g_initialized = false;

static int KYTY_SYSV_ABI ContentSearchInit(const ContentSearchInitParam* init_param) {
	PRINT_NAME();

	LOGF("\t init_param  = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(init_param));

	if (init_param == nullptr) {
		return CONTENT_SEARCH_ERROR_INVALID_PARAM;
	}

	LOGF("\t memory_size = %" PRIu64 "\n", static_cast<uint64_t>(init_param->memory_size));

	g_initialized = true;

	return OK;
}

} // namespace ContentSearch

LIB_DEFINE(InitContentSearch_1) {
	LIB_FUNC("dPj4ZtRcIWk", ContentSearch::ContentSearchInit);
}

} // namespace LibContentSearch

namespace LibContentDelete {

LIB_NAME("ContentDelete", "ContentDelete");

namespace ContentDelete {

constexpr int CONTENT_DELETE_ERROR_INVALID_PARAM = -2137174015; /* 0x809D5001 */

struct ContentDeleteInitParam {
	char   reserved1[4];
	size_t heap_size;
	char   reserved2[32];
};

static bool g_initialized = false;

static int KYTY_SYSV_ABI ContentDeleteInitialize(const ContentDeleteInitParam* init_param) {
	PRINT_NAME();

	LOGF("\t init_param = 0x%016" PRIx64 "\n", reinterpret_cast<uint64_t>(init_param));

	if (init_param == nullptr) {
		return CONTENT_DELETE_ERROR_INVALID_PARAM;
	}

	LOGF("\t heap_size  = %" PRIu64 "\n", static_cast<uint64_t>(init_param->heap_size));

	g_initialized = true;

	return OK;
}

} // namespace ContentDelete

LIB_DEFINE(InitContentDelete_1) {
	LIB_FUNC("zoxb0wEChEM", ContentDelete::ContentDeleteInitialize);
}

} // namespace LibContentDelete

} // namespace Libs
