#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING

#include "common/abi.h"
#include "libs/errno.h"
#include "libs/network.h"
#include "loader/symbolDatabase.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

namespace Libs::LibNet {
void InitNet_1_Net(Loader::SymbolDatabase* symbols);
}

namespace {

int failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "NetResolverTests:%d: %s\n", __LINE__, #condition);               \
			failures++;                                                                            \
		}                                                                                          \
	} while (false)

using NetResolverStartNtoaMultipleRecordsFn = int(KYTY_SYSV_ABI*)(
    int rid, const char* hostname, void* info, int timeout, int retry, int flags);

template <typename T>
T GetNetFunction(const Loader::SymbolDatabase& symbols, const char* nid) {
	const auto* record = symbols.FindByNid(nid, Loader::SymbolType::Func);
	CHECK(record != nullptr);
	return record != nullptr ? reinterpret_cast<T>(record->vaddr) : nullptr;
}

void TestRegistration(const Loader::SymbolDatabase& symbols) {
	const auto fn = GetNetFunction<NetResolverStartNtoaMultipleRecordsFn>(symbols, "RCCY01Xd+58");
	CHECK(fn != nullptr);
}

void TestParameterValidation(NetResolverStartNtoaMultipleRecordsFn fn) {
	using namespace Libs::Network;
	Initialize();

	const int pool = Net::NetPoolCreate("test_pool", 0x10000, 0);
	CHECK(pool >= 0);
	const int rid = Net::NetResolverCreate("test_res", pool, 0);
	CHECK(rid >= 0);

	Net::NetResolverInfo info {};

	// 1. Invalid resolver ID
	CHECK(fn(-1, "localhost", &info, 0, 0, 0) == NET_ERROR_EBADF);
	CHECK(fn(999, "localhost", &info, 0, 0, 0) == NET_ERROR_EBADF);

	// 2. Null hostname
	CHECK(fn(rid, nullptr, &info, 0, 0, 0) == NET_ERROR_EINVAL);

	// 3. Null info
	CHECK(fn(rid, "localhost", nullptr, 0, 0, 0) == NET_ERROR_EINVAL);

	// 4. Empty hostname
	CHECK(fn(rid, "", &info, 0, 0, 0) == NET_ERROR_EINVAL);

	// 5. Hostname > 255 chars
	std::string too_long_host(256, 'a');
	CHECK(fn(rid, too_long_host.c_str(), &info, 0, 0, 0) == NET_ERROR_EINVAL);

	CHECK(Net::NetResolverDestroy(rid) == 0);
	CHECK(Net::NetPoolDestroy(pool) == 0);
	Shutdown();
}

void TestLocalhostAndLiteralIp(NetResolverStartNtoaMultipleRecordsFn fn) {
	using namespace Libs::Network;
	Initialize();

	const int pool = Net::NetPoolCreate("test_pool2", 0x10000, 0);
	CHECK(pool >= 0);
	const int rid = Net::NetResolverCreate("test_res2", pool, 0);
	CHECK(rid >= 0);

	Net::NetResolverInfo info {};
	uint32_t             expected_loopback = 0;
	CHECK(Net::NetInetPton(2, "127.0.0.1", &expected_loopback) == 1);

	// Test localhost (with trailing dot as well)
	CHECK(fn(rid, "localhost.", &info, 0, 0, 0) == 0);
	CHECK(info.records == 1);
	CHECK(info.recordsv4 == 1);
	CHECK(info.addrs[0].af == 2); // AF_INET
	CHECK(info.addrs[0].u.addr.inaddr_addr == expected_loopback);

	// Test literal IP address
	std::memset(&info, 0, sizeof(info));
	CHECK(fn(rid, "192.168.1.50", &info, 0, 0, 0) == 0);
	CHECK(info.records == 1);
	CHECK(info.recordsv4 == 1);
	CHECK(info.addrs[0].af == 2);
	uint32_t expected_ip = 0;
	CHECK(Net::NetInetPton(2, "192.168.1.50", &expected_ip) == 1);
	CHECK(info.addrs[0].u.addr.inaddr_addr == expected_ip);

	// Test literal IP address with DISABLE_IPADDRESS flag (0x00010000)
	std::memset(&info, 0, sizeof(info));
	int ret_dis = fn(rid, "192.168.1.50", &info, 0, 0, 0x00010000);
	CHECK(ret_dis == 0);
	CHECK(info.records == 1);
	CHECK(info.addrs[0].u.addr.inaddr_addr == expected_ip);

	// Test Abort and queued abort
	CHECK(Net::NetResolverAbort(rid, 0) == 0);
	// Once aborted, next lookup should consume abort and return NET_ERROR_EINTR
	CHECK(fn(rid, "localhost", &info, 0, 0, 0) == NET_ERROR_EINTR);
	// Subsequent lookup should succeed again
	CHECK(fn(rid, "localhost", &info, 0, 0, 0) == 0);

	// Test Async lookup (flag 0x00000001)
	// Use static storage to outlive the detached worker thread
	static Net::NetResolverInfo async_info;
	std::memset(&async_info, 0, sizeof(async_info));
	CHECK(fn(rid, "localhost", &async_info, 0, 0, 0x00000001) == 0);
	// Wait a moment for background thread
	for (int i = 0; i < 50 && async_info.records == 0; ++i) {
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	CHECK(async_info.records == 1);
	CHECK(async_info.addrs[0].u.addr.inaddr_addr == expected_loopback);

	CHECK(Net::NetResolverDestroy(rid) == 0);
	CHECK(Net::NetPoolDestroy(pool) == 0);
	Shutdown();
}

void TestResolveGoogle(NetResolverStartNtoaMultipleRecordsFn fn) {
	using namespace Libs::Network;
	Initialize();

	const int pool = Net::NetPoolCreate("test_pool_google", 0x10000, 0);
	CHECK(pool >= 0);
	const int rid = Net::NetResolverCreate("test_res_google", pool, 0);
	CHECK(rid >= 0);

	// Use static storage to outlive the detached worker thread (if it runs async)
	static Net::NetResolverInfo google_info;
	std::memset(&google_info, 0, sizeof(google_info));
	const int ret = fn(rid, "google.com", &google_info, 0, 0, 0);
	CHECK(ret == 0);
	CHECK(google_info.records > 0);
	CHECK(google_info.recordsv4 > 0);
	CHECK(google_info.records <= 10);
	for (uint32_t i = 0; i < google_info.records; ++i) {
		CHECK(google_info.addrs[i].af == 2); // AF_INET
		CHECK(google_info.addrs[i].u.addr.inaddr_addr != 0);
		char ip_str[32] = {0};
		Net::NetInetNtop(2, &google_info.addrs[i].u.addr.inaddr_addr, ip_str, sizeof(ip_str));
		std::printf("Resolved google.com record %u: %s\n", i, ip_str);
	}

	CHECK(Net::NetResolverDestroy(rid) == 0);
	CHECK(Net::NetPoolDestroy(pool) == 0);
	Shutdown();
}

} // namespace

int main() {
	Loader::SymbolDatabase symbols;
	Libs::LibNet::InitNet_1_Net(&symbols);

	TestRegistration(symbols);

	const auto fn = GetNetFunction<NetResolverStartNtoaMultipleRecordsFn>(symbols, "RCCY01Xd+58");
	if (fn == nullptr) {
		return 1;
	}

	TestParameterValidation(fn);
	TestLocalhostAndLiteralIp(fn);
	TestResolveGoogle(fn);

	if (failures == 0) {
		std::printf("All NetResolverTests passed successfully!\n");
	}

	return failures == 0 ? 0 : 1;
}
