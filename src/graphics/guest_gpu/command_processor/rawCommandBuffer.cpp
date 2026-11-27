#include "graphics/guest_gpu/command_processor/rawCommandBuffer.h"

#include "common/runtimeToggles.h"
#include "kernel/memory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::RawCommandBuffer {

namespace {

std::array<std::atomic<uint64_t>, static_cast<size_t>(Stat::Count)> g_stats {};
std::atomic<uint64_t> g_verify_sequence {0};

uint64_t VerifyEvery() {
	static const uint64_t every = [] {
		const char* value = std::getenv("KYTY_RAW_CMDBUF_VERIFY_EVERY");
		const auto  n     = value != nullptr ? std::strtoull(value, nullptr, 10) : 0;
		return n == 0 ? uint64_t {1} : static_cast<uint64_t>(n);
	}();
	return every;
}

} // namespace

Mode EnvMode() {
	static const Mode mode = [] {
		const char* value = std::getenv("KYTY_RAW_CMDBUF");
		if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
			return Mode::Off;
		}
		if (std::strcmp(value, "verify") == 0) {
			return Mode::Verify;
		}
		return Mode::On;
	}();
	return mode;
}

Mode SubmissionMode() {
	const auto mode = EnvMode();
	if (mode == Mode::Off || !Common::RuntimeToggles::Enabled()) {
		return mode;
	}
	return Common::RuntimeToggles::Get("raw_cmdbuf") ? mode : Mode::Off;
}

const uint32_t* View(std::span<const uint32_t> guest) {
	if (guest.empty()) {
		return nullptr;
	}
	return static_cast<const uint32_t*>(LibKernel::Memory::TryGetBackingView(
	    reinterpret_cast<uint64_t>(guest.data()), guest.size_bytes()));
}

uint64_t Verify(std::span<const uint32_t> guest, const uint32_t* view) {
	if (view == nullptr || guest.empty() ||
	    g_verify_sequence.fetch_add(1, std::memory_order_relaxed) % VerifyEvery() != 0) {
		return 0;
	}
	std::vector<uint32_t> raw(view, view + guest.size());
	std::vector<uint32_t> precise(guest.size());
	// The old parser's read: faults on read-protected pages and publishes GPU-written bytes.
	std::memcpy(precise.data(), guest.data(), guest.size_bytes());
	uint64_t differ = 0;
	for (size_t i = 0; i < raw.size(); i++) {
		differ += raw[i] != precise[i] ? 1u : 0u;
	}
	Add(Stat::Verified);
	if (differ != 0) {
		Add(Stat::Mismatched);
		Add(Stat::MismatchDw, differ);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			std::printf("RawCmdbuf verify: span 0x%016" PRIx64 " +%zu dw: %" PRIu64
			            " dw differ from a precise read\n",
			            reinterpret_cast<uint64_t>(guest.data()), guest.size(), differ);
		}
	}
	return differ;
}

void Add(Stat stat, uint64_t value) {
	g_stats[static_cast<size_t>(stat)].fetch_add(value, std::memory_order_relaxed);
}

void MaybeReport() {
	if (EnvMode() == Mode::Off) {
		return;
	}
	using Clock = std::chrono::steady_clock;
	static Clock::time_point next = Clock::now() + std::chrono::seconds(10);
	const auto               now  = Clock::now();
	if (now < next) {
		return;
	}
	next = now + std::chrono::seconds(10);
	const auto get = [](Stat stat) {
		return g_stats[static_cast<size_t>(stat)].load(std::memory_order_relaxed);
	};
	std::printf("RawCmdbuf spans=%" PRIu64 " raw=%" PRIu64 " fallback=%" PRIu64
	            " verified=%" PRIu64 " mismatched=%" PRIu64 " mismatch_dw=%" PRIu64
	            " live_refreshes=%" PRIu64 "\n",
	            get(Stat::Spans), get(Stat::Raw), get(Stat::Fallback), get(Stat::Verified),
	            get(Stat::Mismatched), get(Stat::MismatchDw), get(Stat::LiveRefreshes));
	std::fflush(stdout);
}

} // namespace Libs::Graphics::RawCommandBuffer
