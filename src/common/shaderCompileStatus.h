#ifndef KYTY_COMMON_SHADERCOMPILESTATUS_H_
#define KYTY_COMMON_SHADERCOMPILESTATUS_H_

#include <atomic>
#include <cstdint>

// Always-on counts of host shader work (translations, SPIR-V compiles, pipeline creation), so a
// compile burst can be told apart from a hang or slow emulation. Updates are relaxed atomics; a
// monitor thread, started by the first update, prints a console line every 2 s while work is in
// flight and one when a burst ends, and tells the presentation when the banner changes.
namespace Common::ShaderCompileStatus {

namespace Detail {
inline std::atomic<int64_t>  g_queued {0};
inline std::atomic<int64_t>  g_active {0};
inline std::atomic<uint64_t> g_done {0};
inline thread_local int64_t  t_depth = 0;
void                         StartMonitor() noexcept;
inline void                  EnsureMonitor() noexcept {
	static std::atomic<bool> started {false};
	if (!started.load(std::memory_order_relaxed) && !started.exchange(true)) {
		StartMonitor();
	}
}
} // namespace Detail

// Jobs posted to a worker queue and not started yet.
inline void AddQueued(int64_t count = 1) noexcept {
	Detail::g_queued.fetch_add(count, std::memory_order_relaxed);
	Detail::EnsureMonitor();
}
inline void RemoveQueued(int64_t count = 1) noexcept {
	Detail::g_queued.fetch_sub(count, std::memory_order_relaxed);
}

// One unit of shader work running on this thread. With `counted` its end adds one to done.
class Work {
public:
	explicit Work(bool counted = true) noexcept: m_counted(counted) {
		Detail::g_active.fetch_add(1, std::memory_order_relaxed);
		Detail::t_depth++;
		Detail::EnsureMonitor();
	}
	~Work() {
		if (m_counted) Detail::g_done.fetch_add(1, std::memory_order_relaxed);
		Detail::t_depth--;
		Detail::g_active.fetch_sub(1, std::memory_order_relaxed);
	}
	Work(const Work&)            = delete;
	Work& operator=(const Work&) = delete;

private:
	bool m_counted;
};

// Work scopes open on this thread. A longjmp past Work scopes (KYTY_SKIP_FAILED_SHADERS) skips
// their destructors: the recovery passes the depth taken before setjmp to Unwind.
[[nodiscard]] inline int64_t ThreadDepth() noexcept {
	return Detail::t_depth;
}
inline void Unwind(int64_t depth) noexcept {
	if (Detail::t_depth > depth) {
		Detail::g_active.fetch_sub(Detail::t_depth - depth, std::memory_order_relaxed);
		Detail::t_depth = depth;
	}
}

struct Counts {
	uint64_t queued = 0;
	uint64_t active = 0;
	uint64_t done   = 0;
};

[[nodiscard]] Counts Read() noexcept;
// True while work is queued or running, and for about a second after it stops.
[[nodiscard]] bool BannerVisible() noexcept;
// Changes whenever the banner appears, disappears or its counts change (at most 4 times a
// second), so the host can repaint it while the guest does not flip.
[[nodiscard]] uint64_t BannerRevision() noexcept;

} // namespace Common::ShaderCompileStatus

#endif // KYTY_COMMON_SHADERCOMPILESTATUS_H_
