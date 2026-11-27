#include "common/faultReport.h"

#include <atomic>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstring>

#if defined(__linux__)
#include <algorithm>
#include <dlfcn.h>
#include <elf.h>
#include <execinfo.h>
#include <fcntl.h>
#include <link.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>
#elif defined(_WIN32)
#include <windows.h> // IWYU pragma: keep
// clang-format off
#include <dbghelp.h>
// clang-format on
#include <mutex>
#elif defined(__APPLE__)
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#endif

namespace Common::FaultReport {

namespace {

thread_local const char* t_tag = nullptr;
thread_local char        t_name[32] {};

const char* BaseName(const char* path) {
	const char* slash = std::strrchr(path, '/');
#if defined(_WIN32)
	const char* back = std::strrchr(path, '\\');
	if (back != nullptr && (slash == nullptr || back > slash)) slash = back;
#endif
	return slash != nullptr ? slash + 1 : path;
}

#if defined(__GNUC__) || defined(__clang__)
void Out(const Sink& sink, const char* fmt, ...) __attribute__((format(printf, 2, 3)));
#endif
void Out(const Sink& sink, const char* fmt, ...) {
	if (sink.write == nullptr) return;
	char    text[768];
	va_list args;
	va_start(args, fmt);
	std::vsnprintf(text, sizeof(text), fmt, args);
	va_end(args);
	sink.write(sink.user, text);
}

void WriteThreadHeader(const Sink& sink, const char* name, unsigned long tid) {
	Out(sink, "--- Host fault context ---\n");
	Out(sink, "host thread: %s tid=%lu doing: %s\n",
	    name != nullptr && name[0] != '\0' ? name : "(unnamed)", tid,
	    t_tag != nullptr ? t_tag : "(no tag)");
}

#if defined(__linux__)

constexpr size_t MAX_MODULES = 512;
constexpr size_t NAME_SIZE   = 256;

struct Module {
	uint64_t start = 0; // Mapped range of one loadable segment.
	uint64_t end   = 0;
	uint64_t base  = 0; // Load bias: address - base is the offset in the shared object.
	char     name[NAME_SIZE] {};
};

// Published lock-free: count is stored last, so a handler sees a valid prefix during refresh.
Module              g_modules[MAX_MODULES];
std::atomic<size_t> g_module_count {0};
std::atomic<bool>   g_refreshing {false};

struct Symbol {
	uint64_t    address;
	uint64_t    size;
	const char* name;
};
std::vector<Symbol> g_symbols; // Main executable functions, sorted by link-time address.
uint64_t            g_exe_base  = 0;
bool                g_exe_known = false;

thread_local uintptr_t t_stack_low  = 0;
thread_local uintptr_t t_stack_high = 0;

void LoadExecutableSymbols(const char* path) {
	const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
	if (fd < 0) return;
	struct stat st {};
	if (::fstat(fd, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(Elf64_Ehdr))) {
		::close(fd);
		return;
	}
	void* map = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
	::close(fd);
	if (map == MAP_FAILED) return;
	// Keep mapping: symbol names point into it; untouched pages cost no memory.
	const auto* bytes = static_cast<const uint8_t*>(map);
	const auto  size  = static_cast<uint64_t>(st.st_size);
	const auto* eh    = reinterpret_cast<const Elf64_Ehdr*>(bytes);
	if (std::memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0 || eh->e_ident[EI_CLASS] != ELFCLASS64 ||
	    eh->e_shoff == 0 ||
	    eh->e_shoff + static_cast<uint64_t>(eh->e_shnum) * sizeof(Elf64_Shdr) > size) {
		return;
	}
	const auto* sh = reinterpret_cast<const Elf64_Shdr*>(bytes + eh->e_shoff);
	for (unsigned i = 0; i < eh->e_shnum; i++) {
		if ((sh[i].sh_type != SHT_SYMTAB && sh[i].sh_type != SHT_DYNSYM) ||
		    sh[i].sh_link >= eh->e_shnum || sh[i].sh_offset + sh[i].sh_size > size) {
			continue;
		}
		const auto& str = sh[sh[i].sh_link];
		if (str.sh_offset + str.sh_size > size) continue;
		const auto* syms  = reinterpret_cast<const Elf64_Sym*>(bytes + sh[i].sh_offset);
		const auto  count = sh[i].sh_size / sizeof(Elf64_Sym);
		const auto* names = reinterpret_cast<const char*>(bytes + str.sh_offset);
		for (uint64_t s = 0; s < count; s++) {
			if (ELF64_ST_TYPE(syms[s].st_info) != STT_FUNC || syms[s].st_value == 0 ||
			    syms[s].st_name >= str.sh_size) {
				continue;
			}
			g_symbols.push_back({syms[s].st_value, syms[s].st_size, names + syms[s].st_name});
		}
	}
	std::sort(g_symbols.begin(), g_symbols.end(),
	          [](const Symbol& a, const Symbol& b) { return a.address < b.address; });
}

const Symbol* FindExeSymbol(uint64_t link_address) {
	if (g_symbols.empty()) return nullptr;
	auto it = std::upper_bound(g_symbols.begin(), g_symbols.end(), link_address,
	                           [](uint64_t a, const Symbol& s) { return a < s.address; });
	while (it != g_symbols.begin()) {
		--it;
		if (link_address < it->address + std::max<uint64_t>(it->size, 1)) return &*it;
		if (it->size != 0) break;
	}
	return nullptr;
}

bool FindModule(uint64_t address, const Module** out) {
	const auto count = std::min(g_module_count.load(std::memory_order_acquire), MAX_MODULES);
	for (size_t i = 0; i < count; i++) {
		if (address >= g_modules[i].start && address < g_modules[i].end) {
			*out = &g_modules[i];
			return true;
		}
	}
	return false;
}

template <class Fn>
bool ScanProcMaps(Fn&& fn) {
	const int fd = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
	if (fd < 0) return false;
	char   buffer[4096];
	char   line[NAME_SIZE + 128];
	size_t line_length = 0;
	for (;;) {
		const auto n = ::read(fd, buffer, sizeof(buffer));
		if (n <= 0) break;
		for (ssize_t i = 0; i < n; i++) {
			if (buffer[i] == '\n') {
				line[line_length] = '\0';
				uint64_t start = 0, end = 0;
				int      consumed = 0;
				if (std::sscanf(line, "%" SCNx64 "-%" SCNx64 " %*s %*s %*s %*s %n", &start, &end,
				                &consumed) >= 2 &&
				    consumed != 0) {
					fn(start, end, line + consumed);
				}
				line_length = 0;
			} else if (line_length + 1 < sizeof(line)) {
				line[line_length++] = buffer[i];
			}
		}
	}
	::close(fd);
	return true;
}

// Fallback for modules loaded after RefreshModules (e.g. Vulkan drivers), using read(2) only.
bool FindInMaps(uint64_t address, Module* out) {
	bool     found       = false;
	uint64_t found_start = 0, found_end = 0, lowest = 0;
	char     found_name[NAME_SIZE] {};
	if (!ScanProcMaps([&](uint64_t start, uint64_t end, const char* path) {
		    if (!found && address >= start && address < end) {
			    found = true;
			    found_start = start;
			    found_end = end;
			    lowest = start;
			    std::snprintf(found_name, sizeof(found_name), "%s", path);
		    } else if (found && std::strcmp(path, found_name) == 0 && start < lowest) {
			    lowest = start;
		    }
	    }) ||
	    !found) {
		return false;
	}
	if (found_name[0] != '\0') {
		ScanProcMaps([&](uint64_t start, uint64_t, const char* path) {
			if (std::strcmp(path, found_name) == 0 && start < lowest) lowest = start;
		});
	}
	out->start = found_start;
	out->end   = found_end;
	out->base  = lowest;
	std::snprintf(out->name, sizeof(out->name), "%s",
	              found_name[0] != '\0' ? found_name : "[anonymous]");
	return true;
}

bool IsReadable(uint64_t address, size_t size) {
	if (address == 0 || address + size < address) return false;
	uint8_t probe = 0;
	iovec   local {&probe, 1};
	iovec   remote {reinterpret_cast<void*>(address), 1};
	if (::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0) != 1) return false;
	if (size > 1) {
		remote.iov_base = reinterpret_cast<void*>(address + size - 1);
		return ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0) == 1;
	}
	return true;
}

int PhdrCallback(dl_phdr_info* info, size_t, void* user) {
	const auto* exe_name = static_cast<const char*>(user);
	const char* name =
	    info->dlpi_name != nullptr && info->dlpi_name[0] != '\0' ? info->dlpi_name : exe_name;
	for (unsigned i = 0; i < info->dlpi_phnum; i++) {
		const auto& ph = info->dlpi_phdr[i];
		if (ph.p_type != PT_LOAD) continue;
		const auto index = g_module_count.load(std::memory_order_relaxed);
		if (index >= MAX_MODULES) return 1;
		auto& m = g_modules[index];
		m.start = info->dlpi_addr + ph.p_vaddr;
		m.end   = m.start + ph.p_memsz;
		m.base  = info->dlpi_addr;
		std::snprintf(m.name, sizeof(m.name), "%s", name);
		g_module_count.store(index + 1, std::memory_order_release);
	}
	return 0;
}

bool InLoader(uint64_t address) {
	const Module* m = nullptr;
	if (!FindModule(address, &m)) return false;
	const char* base = BaseName(m->name);
	return std::strncmp(base, "ld-linux", 8) == 0 || std::strncmp(base, "ld-", 3) == 0;
}

#elif defined(_WIN32)

std::mutex g_dbghelp_mutex;
bool       g_sym_initialized = false;

bool EnsureSymInitializedLocked(HANDLE process) {
	if (g_sym_initialized) return true;
	SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
	g_sym_initialized = SymInitialize(process, nullptr, TRUE) != FALSE;
	return g_sym_initialized;
}

void FormatAddressLocked(HANDLE process, uint64_t address, char* out, size_t size) {
	if (!EnsureSymInitializedLocked(process)) {
		std::snprintf(out, size, "not in a loaded module");
		return;
	}
	const DWORD64 mod_base = SymGetModuleBase64(process, static_cast<DWORD64>(address));
	char          mod_path[MAX_PATH] {};
	const char*   mod_name = nullptr;
	if (mod_base != 0) {
		if (GetModuleFileNameA(reinterpret_cast<HMODULE>(static_cast<uintptr_t>(mod_base)),
		                       mod_path, MAX_PATH) != 0) {
			mod_name = BaseName(mod_path);
		} else {
			IMAGEHLP_MODULE64 mod_info {};
			mod_info.SizeOfStruct = sizeof(mod_info);
			if (SymGetModuleInfo64(process, static_cast<DWORD64>(address), &mod_info) != FALSE &&
			    mod_info.ModuleName[0] != '\0') {
				mod_name = mod_info.ModuleName;
			}
		}
	}
	if (mod_base == 0 || mod_name == nullptr) {
		std::snprintf(out, size, "not in a loaded module");
		return;
	}
	const uint64_t            offset       = address - static_cast<uint64_t>(mod_base);
	constexpr size_t          SYM_NAME_CAP = 256;
	alignas(SYMBOL_INFO) char sym_buf[sizeof(SYMBOL_INFO) + SYM_NAME_CAP] {};
	auto*                     sym = reinterpret_cast<SYMBOL_INFO*>(sym_buf);
	sym->SizeOfStruct             = sizeof(SYMBOL_INFO);
	sym->MaxNameLen               = SYM_NAME_CAP;
	DWORD64 sym_disp              = 0;
	if (SymFromAddr(process, static_cast<DWORD64>(address), &sym_disp, sym) != FALSE &&
	    sym->Name[0] != '\0') {
		std::snprintf(out, size, "%s+0x%" PRIx64 " [%s+0x%" PRIx64 "]", mod_name, offset, sym->Name,
		              static_cast<uint64_t>(sym_disp));
	} else {
		std::snprintf(out, size, "%s+0x%" PRIx64, mod_name, offset);
	}
}

#endif

} // namespace

void SetThreadTag(const char* tag, const char* name) noexcept {
	t_tag = tag;
	if (name != nullptr) {
		std::snprintf(t_name, sizeof(t_name), "%s", name);
#if defined(__linux__)
		char os_name[16] {};
		std::snprintf(os_name, sizeof(os_name), "%.15s", name);
		::pthread_setname_np(::pthread_self(), os_name);
#elif defined(__APPLE__)
		::pthread_setname_np(name);
#endif
	}
}

void RefreshModules() {
#if defined(__linux__)
	if (g_refreshing.exchange(true)) return;
	char       exe[NAME_SIZE] {};
	const auto n = ::readlink("/proc/self/exe", exe, sizeof(exe) - 1);
	if (n > 0) exe[n] = '\0';
	g_module_count.store(0, std::memory_order_release);
	::dl_iterate_phdr(PhdrCallback, exe);
	if (g_symbols.empty() && exe[0] != '\0') {
		if (g_module_count.load() > 0) {
			g_exe_base  = g_modules[0].base;
			g_exe_known = true;
		}
		LoadExecutableSymbols(exe);
	}
	void* warm[4];
	(void)::backtrace(warm, 4);
	g_refreshing.store(false);
#elif defined(_WIN32)
	std::lock_guard lock(g_dbghelp_mutex);
	const HANDLE    process = GetCurrentProcess();
	if (g_sym_initialized) {
		SymRefreshModuleList(process);
	} else {
		EnsureSymInitializedLocked(process);
	}
#endif
}

void RegisterThreadStack() {
#if defined(__linux__)
	pthread_attr_t attr;
	if (::pthread_getattr_np(::pthread_self(), &attr) != 0) return;
	void*  addr = nullptr;
	size_t size = 0;
	if (::pthread_attr_getstack(&attr, &addr, &size) == 0) {
		t_stack_low  = reinterpret_cast<uintptr_t>(addr);
		t_stack_high = t_stack_low + size;
	}
	::pthread_attr_destroy(&attr);
#endif
}

void FormatAddress(uint64_t address, char* out, size_t size) {
	if (out == nullptr || size == 0) return;
#if defined(__linux__)
	const Module* module = nullptr;
	Module        from_maps;
	if (!FindModule(address, &module) && FindInMaps(address, &from_maps)) module = &from_maps;
	if (module == nullptr) {
		std::snprintf(out, size, "not in a loaded module");
		return;
	}
	const uint64_t offset     = address - module->base;
	const char*    symbol     = nullptr;
	uint64_t       sym_offset = 0;
	if (const auto* s =
	        (g_exe_known && module->base == g_exe_base) ? FindExeSymbol(offset) : nullptr;
	    s != nullptr) {
		symbol     = s->name;
		sym_offset = offset - s->address;
	} else if (!InLoader(address)) {
		Dl_info dl {};
		if (::dladdr(reinterpret_cast<void*>(address), &dl) != 0 && dl.dli_sname != nullptr) {
			symbol     = dl.dli_sname;
			sym_offset = address - reinterpret_cast<uint64_t>(dl.dli_saddr);
		}
	}
	if (symbol != nullptr) {
		std::snprintf(out, size, "%s+0x%" PRIx64 " [%s+0x%" PRIx64 "]", BaseName(module->name),
		              offset, symbol, sym_offset);
	} else {
		std::snprintf(out, size, "%s+0x%" PRIx64, BaseName(module->name), offset);
	}
#elif defined(_WIN32)
	if (!g_dbghelp_mutex.try_lock()) {
		std::snprintf(out, size, "symbol table busy");
		return;
	}
	std::lock_guard lock(g_dbghelp_mutex, std::adopt_lock);
	FormatAddressLocked(GetCurrentProcess(), address, out, size);
#elif defined(__APPLE__)
	Dl_info dl {};
	if (::dladdr(reinterpret_cast<void*>(address), &dl) != 0 && dl.dli_fname != nullptr) {
		const uint64_t offset = address - reinterpret_cast<uint64_t>(dl.dli_fbase);
		std::snprintf(out, size, "%s+0x%" PRIx64, BaseName(dl.dli_fname), offset);
	} else {
		std::snprintf(out, size, "not in a loaded module");
	}
#else
	(void)address;
	std::snprintf(out, size, "not available on this platform");
#endif
}

void WriteHostContext(const Registers& regs, const Sink& sink) {
#if defined(__linux__)
	char name[32] {};
	::pthread_getname_np(::pthread_self(), name, sizeof(name));
	WriteThreadHeader(sink, t_name[0] != '\0' ? t_name : name, static_cast<unsigned long>(::gettid()));
	char where[NAME_SIZE + 160];
	FormatAddress(regs.pc, where, sizeof(where));
	Out(sink, " pc       %016" PRIx64 " = %s\n", regs.pc, where);

	uintptr_t low  = regs.rsp;
	uintptr_t high = regs.rsp + 1024u * 1024u;
	if (t_stack_low <= regs.rsp && regs.rsp < t_stack_high) high = t_stack_high;
	Out(sink, "frame-pointer chain (code built without frame pointers may skip frames):\n");
	uint64_t frame = regs.rbp;
	int      depth = 0;
	while (depth < 24 && frame >= low && frame + 16 <= high && (frame & 7u) == 0 &&
	       IsReadable(frame, 16)) {
		const auto* words = reinterpret_cast<const uint64_t*>(frame);
		FormatAddress(words[1], where, sizeof(where));
		Out(sink, " fp[%2d]   %016" PRIx64 " = %s\n", depth, words[1], where);
		depth++;
		if (words[0] <= frame) break;
		frame = words[0];
	}
	if (depth == 0) Out(sink, " (none)\n");

	void*      frames[48];
	const auto count = ::backtrace(frames, 48);
	Out(sink, "backtrace (unwinder, %d frames, handler frames first):\n", count);
	for (int i = 0; i < count; i++) {
		FormatAddress(reinterpret_cast<uint64_t>(frames[i]), where, sizeof(where));
		Out(sink, " bt[%2d]   %016" PRIx64 " = %s\n", i, reinterpret_cast<uint64_t>(frames[i]),
		    where);
	}

	Out(sink, "stack words that point into modules (rsp upward):\n");
	int named = 0;
	for (int i = 0; i < 128 && named < 24; i++) {
		const uint64_t addr = regs.rsp + static_cast<uint64_t>(i) * 8u;
		if (addr + 8 > high || !IsReadable(addr, 8)) break;
		const uint64_t value = *reinterpret_cast<const uint64_t*>(addr);
		const Module*  m     = nullptr;
		if (value > 0x10000 && FindModule(value, &m)) {
			FormatAddress(value, where, sizeof(where));
			Out(sink, " rsp+0x%03x %016" PRIx64 " = %s\n", i * 8, value, where);
			named++;
		}
	}
#elif defined(_WIN32)
	WriteThreadHeader(sink, t_name, static_cast<unsigned long>(GetCurrentThreadId()));
	if (!g_dbghelp_mutex.try_lock()) {
		Out(sink, " pc       %016" PRIx64 " = symbol table busy\n", regs.pc);
		return;
	}
	std::lock_guard lock(g_dbghelp_mutex, std::adopt_lock);
	const HANDLE    process = GetCurrentProcess();
	char            where[512];
	FormatAddressLocked(process, regs.pc, where, sizeof(where));
	Out(sink, " pc       %016" PRIx64 " = %s\n", regs.pc, where);

	if (regs.native_context != nullptr && EnsureSymInitializedLocked(process)) {
		CONTEXT      ctx = *static_cast<const CONTEXT*>(regs.native_context);
		STACKFRAME64 frame {};
		frame.AddrPC.Offset    = ctx.Rip;
		frame.AddrPC.Mode      = AddrModeFlat;
		frame.AddrFrame.Offset = ctx.Rbp;
		frame.AddrFrame.Mode   = AddrModeFlat;
		frame.AddrStack.Offset = ctx.Rsp;
		frame.AddrStack.Mode   = AddrModeFlat;
		Out(sink, "backtrace (StackWalk64):\n");
		int depth = 0;
		while (depth < 48 &&
		       StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, GetCurrentThread(), &frame, &ctx,
		                   nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr) !=
		           FALSE &&
		       frame.AddrPC.Offset != 0) {
			FormatAddressLocked(process, static_cast<uint64_t>(frame.AddrPC.Offset), where,
			                    sizeof(where));
			Out(sink, " bt[%2d]   %016" PRIx64 " = %s\n", depth,
			    static_cast<uint64_t>(frame.AddrPC.Offset), where);
			depth++;
		}
		if (depth == 0) Out(sink, " (none)\n");
	} else {
		void*        frames[48];
		const USHORT count = RtlCaptureStackBackTrace(0, 48, frames, nullptr);
		Out(sink, "backtrace (RtlCaptureStackBackTrace, %u frames):\n",
		    static_cast<unsigned>(count));
		for (USHORT i = 0; i < count; i++) {
			const auto addr = reinterpret_cast<uint64_t>(frames[i]);
			FormatAddressLocked(process, addr, where, sizeof(where));
			Out(sink, " bt[%2d]   %016" PRIx64 " = %s\n", static_cast<int>(i), addr, where);
		}
	}
#elif defined(__APPLE__)
	char name[64] {};
	::pthread_getname_np(::pthread_self(), name, sizeof(name));
	uint64_t tid = 0;
	::pthread_threadid_np(nullptr, &tid);
	WriteThreadHeader(sink, t_name[0] != '\0' ? t_name : name, static_cast<unsigned long>(tid));
	char where[384];
	FormatAddress(regs.pc, where, sizeof(where));
	Out(sink, " pc       %016" PRIx64 " = %s\n", regs.pc, where);
	void*      frames[48];
	const auto count = ::backtrace(frames, 48);
	Out(sink, "backtrace (%d frames, module+offset only):\n", count);
	for (int i = 0; i < count; i++) {
		const auto addr = reinterpret_cast<uint64_t>(frames[i]);
		FormatAddress(addr, where, sizeof(where));
		Out(sink, " bt[%2d]   %016" PRIx64 " = %s\n", i, addr, where);
	}
#else
	WriteThreadHeader(sink, t_name, 0);
	Out(sink, " pc       %016" PRIx64 " = not available on this platform\n", regs.pc);
#endif
}

} // namespace Common::FaultReport
