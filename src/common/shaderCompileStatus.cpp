#include "common/shaderCompileStatus.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <stop_token>
#include <thread>

namespace Common::ShaderCompileStatus {

namespace {

using Clock = std::chrono::steady_clock;

constexpr auto TICK           = std::chrono::milliseconds(250);
constexpr auto HOLD           = std::chrono::seconds(1);
constexpr auto PRINT_INTERVAL = std::chrono::seconds(2);

std::atomic<bool>     g_banner_visible {false};
std::atomic<uint64_t> g_banner_revision {0};

uint64_t Clamp(int64_t value) {
	return value > 0 ? static_cast<uint64_t>(value) : 0;
}

void Monitor(const std::stop_token& stop) {
	std::mutex                  mutex;
	std::condition_variable_any wake;
	bool                        in_burst = false;
	Clock::time_point           burst_start {};
	Clock::time_point           last_busy {};
	Clock::time_point           last_print {};
	uint64_t                    burst_start_done = 0;
	uint64_t                    idle_done        = 0;
	Counts                      shown {};
	std::unique_lock            lock(mutex);
	while (!stop.stop_requested()) {
		wake.wait_for(lock, stop, TICK, [] { return false; });
		if (stop.stop_requested()) {
			break;
		}
		const auto now    = Clock::now();
		const auto counts = Read();
		const bool busy   = counts.queued + counts.active > 0;
		if (busy) {
			last_busy = now;
			if (!in_burst) {
				in_burst         = true;
				burst_start      = now - TICK / 2;
				burst_start_done = idle_done;
				last_print       = {};
			}
		} else if (!in_burst) {
			idle_done = counts.done;
		}
		if (in_burst && busy && now - last_print >= PRINT_INTERVAL) {
			last_print = now;
			std::printf("[shaders] compiling: %llu queued, %llu active, %llu done\n",
			            static_cast<unsigned long long>(counts.queued),
			            static_cast<unsigned long long>(counts.active),
			            static_cast<unsigned long long>(counts.done));
			std::fflush(stdout);
		}
		bool visible = in_burst;
		if (in_burst && !busy && now - last_busy >= HOLD) {
			in_burst  = false;
			visible   = false;
			idle_done = counts.done;
			std::printf("[shaders] burst done: %llu shaders in %.1f s\n",
			            static_cast<unsigned long long>(counts.done - burst_start_done),
			            std::chrono::duration<double>(last_busy - burst_start).count());
			std::fflush(stdout);
		}
		const bool changed = visible != g_banner_visible.load(std::memory_order_relaxed) ||
		                     (visible && (counts.queued != shown.queued ||
		                                  counts.active != shown.active || counts.done != shown.done));
		if (changed) {
			shown = counts;
			g_banner_visible.store(visible, std::memory_order_release);
			g_banner_revision.fetch_add(1, std::memory_order_acq_rel);
		}
	}
}

} // namespace

namespace Detail {

void StartMonitor() noexcept {
	static std::jthread monitor(Monitor);
}

} // namespace Detail

Counts Read() noexcept {
	return {Clamp(Detail::g_queued.load(std::memory_order_relaxed)),
	        Clamp(Detail::g_active.load(std::memory_order_relaxed)),
	        Detail::g_done.load(std::memory_order_relaxed)};
}

bool BannerVisible() noexcept {
	return g_banner_visible.load(std::memory_order_acquire);
}

uint64_t BannerRevision() noexcept {
	return g_banner_revision.load(std::memory_order_acquire);
}

} // namespace Common::ShaderCompileStatus
