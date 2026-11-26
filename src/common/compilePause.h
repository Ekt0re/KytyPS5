#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_COMPILEPAUSE_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_COMPILEPAUSE_H_

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

// Diagnostic (KYTY_PAUSE_ON_COMPILE): while the host translates or compiles shaders, long guest
// sleeps do not end, so the guest sees the compile as a pause rather than as missing progress.
namespace Common::CompilePause {

inline std::atomic<int> g_active {0};

inline bool Enabled() {
	static const bool enabled = std::getenv("KYTY_PAUSE_ON_COMPILE") != nullptr;
	return enabled;
}

struct Scope {
	Scope() { g_active.fetch_add(1, std::memory_order_relaxed); }
	~Scope() { g_active.fetch_sub(1, std::memory_order_relaxed); }
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;
};

// Called after a guest sleep of at least half a second.
inline void HoldWhileCompiling() {
	if (!Enabled()) return;
	while (g_active.load(std::memory_order_relaxed) > 0)
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
}

} // namespace Common::CompilePause

#endif
