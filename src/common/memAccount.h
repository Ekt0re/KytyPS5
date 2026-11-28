#ifndef KYTY_COMMON_MEM_ACCOUNT_H_
#define KYTY_COMMON_MEM_ACCOUNT_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Periodic process memory reporting (KYTY_MEM_REPORT=<seconds>, 0/unset = off), memory-limit
// guard (KYTY_MEM_LIMIT_GB=<n>, 0/unset = off), and one [exit] summary line. When both switches
// are off/unset, the subsystem is inactive and does not start a thread or touch signal handlers.
namespace Common::MemAccount {

// Named memory owner gauge updated from any thread and read by the reporter.
class Gauge {
public:
	void Set(uint64_t bytes, uint64_t items = 0) {
		m_bytes.store(bytes, std::memory_order_relaxed);
		m_items.store(items, std::memory_order_relaxed);
	}
	void Add(int64_t bytes, int64_t items = 0) {
		Apply(m_bytes, bytes);
		Apply(m_items, items);
	}
	[[nodiscard]] uint64_t Bytes() const { return m_bytes.load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t Items() const { return m_items.load(std::memory_order_relaxed); }

private:
	static void Apply(std::atomic<uint64_t>& value, int64_t delta) {
		auto current = value.load(std::memory_order_relaxed);
		for (;;) {
			const auto next = delta >= 0 ? current + static_cast<uint64_t>(delta)
			                  : current > static_cast<uint64_t>(-delta)
			                      ? current - static_cast<uint64_t>(-delta)
			                      : uint64_t {0};
			if (value.compare_exchange_weak(current, next, std::memory_order_relaxed)) return;
		}
	}
	std::atomic<uint64_t> m_bytes {0};
	std::atomic<uint64_t> m_items {0};
};

// Returns (or creates) the gauge for a named owner. Valid for the lifetime of the process.
Gauge& Owner(std::string_view name);

struct Entry {
	std::string name;
	uint64_t    bytes = 0;
	uint64_t    items = 0;
};
[[nodiscard]] std::vector<Entry> Snapshot();

using ExternalReader = std::function<std::string()>;
void SetExternal(std::string_view name, ExternalReader reader);
void ClearExternal(std::string_view name);

struct ProcessMemory {
	bool     available        = false;
	uint64_t rss              = 0; // Resident set / working set.
	uint64_t private_bytes    = 0; // Private committed/anonymous bytes where reported.
	uint64_t swap             = 0; // Swapped / compressed / paged-out commit.
	uint64_t peak_rss         = 0; // Peak resident set (or max observed).
	uint64_t rss_anon         = 0;
	uint64_t rss_file         = 0;
	uint64_t rss_shmem        = 0;
	uint64_t sys_commit_used  = 0; // Optional system commit figure (Windows).
	uint64_t sys_commit_total = 0;
};
[[nodiscard]] ProcessMemory ParseProcStatus(std::string_view text);
[[nodiscard]] ProcessMemory ReadProcessMemory();

struct HeapStats {
	bool     known     = false;
	uint64_t in_use    = 0;
	uint64_t free_held = 0;
};
[[nodiscard]] HeapStats ReadHeapStats();

// Optional Linux zram swap stats (/sys/block/zram*/mm_stat).
struct ZramStats {
	bool     known      = false;
	uint64_t orig       = 0;
	uint64_t compressed = 0;
};
[[nodiscard]] ZramStats ParseZramMmStat(std::string_view text);
[[nodiscard]] ZramStats ReadZram();
[[nodiscard]] uint64_t  Footprint(const ProcessMemory& process, const ZramStats& zram);

void NoteGuestBacking(uint64_t bytes);

struct Sample {
	ProcessMemory                                    process;
	HeapStats                                        heap;
	std::vector<Entry>                               owners;
	std::vector<std::pair<std::string, std::string>> external;
	ZramStats                                        zram;
	uint64_t                                         guest_committed = 0;
};

[[nodiscard]] std::vector<std::pair<std::string, uint64_t>> Holders(const Sample& sample);
[[nodiscard]] std::string FormatGrowth(const std::vector<std::pair<std::string, uint64_t>>& before,
                                       const std::vector<std::pair<std::string, uint64_t>>& after,
                                       size_t top = 4);
[[nodiscard]] uint64_t    Counted(const std::vector<Entry>& owners);
[[nodiscard]] int64_t     Unaccounted(uint64_t rss, uint64_t counted);
[[nodiscard]] std::string FormatLine(const Sample& sample, std::string_view tag);

// Limit decision for KYTY_MEM_LIMIT_GB: first time over -> DropCaches, still over after grace_ms
// -> Stop, dropping back under resets the timer.
class LimitGuard {
public:
	enum class Action : uint8_t { None, DropCaches, Stop };
	explicit LimitGuard(uint64_t limit_bytes, uint64_t grace_ms = 10000)
	    : m_limit(limit_bytes), m_grace_ms(grace_ms) {}
	[[nodiscard]] Action Step(uint64_t now_ms, uint64_t footprint) {
		if (m_limit == 0 || footprint <= m_limit) {
			m_over = false;
			return Action::None;
		}
		if (!m_over) {
			m_over       = true;
			m_over_since = now_ms;
			return Action::DropCaches;
		}
		return now_ms - m_over_since >= m_grace_ms ? Action::Stop : Action::None;
	}

private:
	uint64_t m_limit;
	uint64_t m_grace_ms;
	bool     m_over       = false;
	uint64_t m_over_since = 0;
};

[[nodiscard]] bool TakeDropRequest();
void               NoteExitReason(const char* reason);
void               PrintExit(std::string_view reason);
void               PrintNow(std::string_view tag);
void               Start();
void               Stop();
void               EmergencyPrint();

struct Lifecycle {
	static constexpr const char* name               = "MemAccount";
	static constexpr auto        initialize         = Start;
	static constexpr auto        shutdown           = Stop;
	static constexpr auto        emergency_shutdown = EmergencyPrint;
};

} // namespace Common::MemAccount

#endif /* KYTY_COMMON_MEM_ACCOUNT_H_ */
