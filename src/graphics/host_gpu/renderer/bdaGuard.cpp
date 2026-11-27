#include "graphics/host_gpu/renderer/bdaGuard.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/gpuBreadcrumbs.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fmt/format.h>
#include <map>
#include <mutex>
#include <vk_mem_alloc.h>

namespace Libs::Graphics::BdaGuard {

namespace {

using Clock = std::chrono::steady_clock;

struct Live {
	uint64_t size     = 0;
	uint64_t guest    = 0;
	uint64_t serial   = 0;
	uint32_t last_use = 0;     // Breadcrumb number of the last work that may read it.
	bool     tracked  = false; // last_use is meaningful (Retire or Use was called).
};

struct Dead {
	uint64_t          address  = 0;
	uint64_t          size     = 0;
	uint64_t          guest    = 0;
	uint64_t          serial   = 0;
	uint32_t          recorded = 0; // Breadcrumbs at destruction.
	uint32_t          completed = 0;
	Clock::time_point when;
};

struct Kept {
	VmaAllocator      allocator  = nullptr;
	VkBuffer          buffer     = VK_NULL_HANDLE;
	VmaAllocation     allocation = nullptr;
	uint64_t          address    = 0;
	uint64_t          size       = 0;
	uint32_t          recorded   = 0;
	Clock::time_point when;
};

constexpr size_t   MaxDead     = 4096;
constexpr uint32_t MaxPrinted  = 200;
constexpr auto     KeepWithout = std::chrono::seconds(5); // Without breadcrumbs.

std::mutex               g_mutex;
std::map<uint64_t, Live> g_live; // By device address.
std::deque<Dead>         g_dead;
std::deque<Kept>         g_kept;
uint64_t                 g_serial = 0;
std::atomic<uint64_t>    g_findings {0};
std::atomic<uint32_t>    g_printed {0};
uint64_t                 g_kept_bytes = 0;
uint64_t                 g_kept_total = 0;

void Print(const std::string& text) {
	if (g_printed.fetch_add(1, std::memory_order_relaxed) < MaxPrinted) {
		Log::WriteToConsoleAndLog("[bda-guard] " + text + "\n");
	}
}

// The live buffer that holds address, or g_live.end().
auto FindLive(uint64_t address) {
	auto next = g_live.upper_bound(address);
	if (next == g_live.begin()) return g_live.end();
	const auto found = std::prev(next);
	return address - found->first < found->second.size ? found : g_live.end();
}

std::string DescribeLocked(uint64_t address, uint64_t bytes) {
	const auto now = Clock::now();
	if (const auto live = FindLive(address); live != g_live.end()) {
		return fmt::format("live buffer #{} 0x{:x}+0x{:x} guest 0x{:x}{}", live->second.serial,
		                   live->first, live->second.size, live->second.guest,
		                   address + bytes > live->first + live->second.size
		                       ? fmt::format(": the range ends 0x{:x} bytes past it",
		                                     address + bytes - live->first - live->second.size)
		                       : std::string());
	}
	// The most recently destroyed buffer that held it.
	for (auto dead = g_dead.rbegin(); dead != g_dead.rend(); ++dead) {
		if (address >= dead->address && address - dead->address < dead->size) {
			return fmt::format(
			    "destroyed buffer #{} 0x{:x}+0x{:x} guest 0x{:x}, destroyed {} ms ago at breadcrumb "
			    "{} (GPU had completed {})",
			    dead->serial, dead->address, dead->size, dead->guest,
			    std::chrono::duration_cast<std::chrono::milliseconds>(now - dead->when).count(),
			    dead->recorded, dead->completed);
		}
	}
	// Otherwise the nearest live neighbours, to tell an overrun from a stray value.
	std::string text = "no live or recently destroyed buffer";
	auto        next = g_live.upper_bound(address);
	if (next != g_live.begin()) {
		const auto before = std::prev(next);
		text += fmt::format("; 0x{:x} past the end of live #{} 0x{:x}+0x{:x} guest 0x{:x}",
		                    address - before->first - before->second.size, before->second.serial,
		                    before->first, before->second.size, before->second.guest);
	}
	if (next != g_live.end()) {
		text += fmt::format("; 0x{:x} before live #{} 0x{:x}+0x{:x}", next->first - address,
		                    next->second.serial, next->first, next->second.size);
	}
	return text;
}

void PollLocked() {
	const bool crumbs    = GpuBreadcrumbs::Available();
	const auto completed = GpuBreadcrumbs::Completed();
	const auto now       = Clock::now();
	while (!g_kept.empty()) {
		const auto& kept = g_kept.front();
		const bool  done = crumbs ? completed >= kept.recorded && now - kept.when > std::chrono::milliseconds(50)
		                          : now - kept.when > KeepWithout;
		if (!done) break;
		vmaDestroyBuffer(kept.allocator, kept.buffer, kept.allocation);
		g_kept_bytes -= kept.size;
		g_kept.pop_front();
	}
}

} // namespace

int Mode() {
	static const int mode = [] {
		const char* value = std::getenv("KYTY_BDA_GUARD");
		const int   mode  = value == nullptr ? 0 : std::clamp(std::atoi(value), 0, 2);
		if (mode != 0) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "[bda-guard] enabled: device addresses are checked before use{}\n",
			    mode == 2 ? "; destroyed buffers are kept until the GPU completed prior work" : ""));
		}
		return mode;
	}();
	return mode;
}

void OnCreate(uint64_t address, uint64_t size, uint64_t guest) {
	if (Mode() == 0 || address == 0) return;
	std::scoped_lock lock {g_mutex};
	// A kept buffer still owns its range, so a new one cannot overlap anything known.
	if (const auto live = FindLive(address); live != g_live.end()) {
		g_findings.fetch_add(1, std::memory_order_relaxed);
		Print(fmt::format("new buffer 0x{:x}+0x{:x} overlaps {}", address, size,
		                  DescribeLocked(address, size)));
	}
	g_live[address] = {.size = size, .guest = guest, .serial = ++g_serial};
}

bool OnDestroy(uint64_t address, void* allocator, void* buffer, void* allocation) {
	if (Mode() == 0 || address == 0) return false;
	std::scoped_lock lock {g_mutex};
	const auto recorded  = GpuBreadcrumbs::Recorded();
	const auto completed = GpuBreadcrumbs::Completed();
	const auto found     = g_live.find(address);
	Live       live;
	if (found != g_live.end()) {
		live = found->second;
		g_live.erase(found);
	}
	// Work the GPU has not completed may still read a buffer whose last reader is among it.
	if (live.tracked && GpuBreadcrumbs::Available() && completed < live.last_use) {
		g_findings.fetch_add(1, std::memory_order_relaxed);
		Print(fmt::format("buffer #{} 0x{:x}+0x{:x} guest 0x{:x} is destroyed while work up to "
		                  "breadcrumb {} may read it and the GPU completed only {} (recorded {})",
		                  live.serial, address, live.size, live.guest, live.last_use, completed,
		                  recorded));
	}
	g_dead.push_back({address, live.size, live.guest, live.serial, recorded, completed, Clock::now()});
	if (g_dead.size() > MaxDead) g_dead.pop_front();
	PollLocked();
	if (Mode() < 2) return false;
	g_kept.push_back({static_cast<VmaAllocator>(allocator), static_cast<VkBuffer>(buffer),
	                  static_cast<VmaAllocation>(allocation), address, live.size, recorded,
	                  Clock::now()});
	g_kept_bytes += live.size;
	g_kept_total++;
	return true;
}

void Retire(uint64_t address) {
	if (Mode() == 0) return;
	std::scoped_lock lock {g_mutex};
	if (const auto found = g_live.find(address); found != g_live.end()) {
		found->second.tracked  = true;
		found->second.last_use = std::max(found->second.last_use, GpuBreadcrumbs::Recorded());
	}
}

void Use(uint64_t address) {
	if (Mode() == 0) return;
	std::scoped_lock lock {g_mutex};
	if (const auto found = FindLive(address); found != g_live.end()) {
		found->second.tracked  = true;
		found->second.last_use = std::max(found->second.last_use, GpuBreadcrumbs::Recorded() + 1u);
	}
}

bool Check(uint64_t address, uint64_t bytes, std::string_view what, uint64_t shader,
           uint32_t binding, uint64_t guest) {
	if (Mode() == 0) return true;
	std::scoped_lock lock {g_mutex};
	const auto live = FindLive(address);
	if (live != g_live.end() && bytes <= live->second.size &&
	    address - live->first <= live->second.size - bytes) {
		return true;
	}
	g_findings.fetch_add(1, std::memory_order_relaxed);
	Print(fmt::format("{}: shader 0x{:016x} binding {} guest 0x{:x} gets device address "
	                  "0x{:x}+0x{:x}: {} (breadcrumb {}, completed {})",
	                  what, shader, binding, guest, address, bytes, DescribeLocked(address, bytes),
	                  GpuBreadcrumbs::Recorded(), GpuBreadcrumbs::Completed()));
	return false;
}

void Report(std::string_view text) {
	if (Mode() == 0) return;
	g_findings.fetch_add(1, std::memory_order_relaxed);
	Print(std::string(text));
}

void Note(std::string_view text) {
	if (Mode() == 0) return;
	static std::atomic<uint32_t> notes {0};
	if (notes.fetch_add(1, std::memory_order_relaxed) < 100u) {
		Log::WriteToConsoleAndLog("[bda-guard] note: " + std::string(text) + "\n");
	}
}

std::string Describe(uint64_t address, uint64_t precision) {
	if (Mode() == 0) return {};
	std::scoped_lock lock {g_mutex};
	return fmt::format("  [bda-guard] 0x{:x}: {}\n  [bda-guard] {} live buffers, {} kept ({} MiB), {} "
	                   "findings\n",
	                   address, DescribeLocked(address, std::max<uint64_t>(precision, 1u)),
	                   g_live.size(), g_kept.size(), g_kept_bytes >> 20u,
	                   g_findings.load(std::memory_order_relaxed));
}

void Poll() {
	if (Mode() < 2) return;
	std::scoped_lock lock {g_mutex};
	PollLocked();
}

void Shutdown() {
	if (Mode() == 0) return;
	std::scoped_lock lock {g_mutex};
	for (const auto& kept: g_kept) vmaDestroyBuffer(kept.allocator, kept.buffer, kept.allocation);
	g_kept.clear();
	g_kept_bytes = 0;
	Log::WriteToConsoleAndLog(fmt::format("[bda-guard] {} findings, {} buffers kept in total\n",
	                                      g_findings.load(std::memory_order_relaxed), g_kept_total));
}

uint64_t Findings() {
	return g_findings.load(std::memory_order_relaxed);
}

} // namespace Libs::Graphics::BdaGuard
