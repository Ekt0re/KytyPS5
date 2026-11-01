// Internal memory address-space helpers included by memory.cpp.
// Kept in the same translation unit because they depend on memory.cpp-local helpers.

static constexpr uint64_t HOST_SYSTEM_MANAGED_MIN  = 0x0000040000ull;
static constexpr uint64_t HOST_SYSTEM_MANAGED_MAX  = 0x07ffffbfffull;
#if defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fbfffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x7000000000ull;
#else
static constexpr uint64_t HOST_SYSTEM_RESERVED_MAX = 0x0fffffffffull;
static constexpr uint64_t HOST_USER_MIN            = 0x1000000000ull;
#endif
static constexpr uint64_t HOST_USER_MAX = 0xfbffffffffull;

static bool GuestFreeRangeContains(uint64_t base, uint64_t range_size, uint64_t candidate,
                                   uint64_t size) {
	const auto range_end = base + range_size;
	return candidate <= range_end && size <= range_end - candidate;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
static bool CoalescePlaceholders(uint64_t vaddr, uint64_t size) {
	return VirtualFree(reinterpret_cast<void*>(vaddr), size,
	                   MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS) != 0;
}
#endif

// Ranges whose guest mapping is briefly absent while the emulator replaces it: a guest mmap with
// MAP_FIXED over an existing mapping (unmapped, then mapped again), and, on Windows, the parts of a
// file view that stay mapped when part of it is unmapped (Windows splits a view only by unmapping
// all of it). On the console these replacements are atomic, so guest threads keep using the
// memory: GTA V's streamer decompresses into buffers whose 64 KiB pages are mapped again or
// neighbour an unmap. A fault inside such a range waits until the mapping is back and is retried
// instead of being reported as an access violation.
struct TransientUnmapWindow {
	std::atomic<uint64_t> id {0};
	std::atomic<uint64_t> start {0};
	std::atomic<uint64_t> end {0};
	std::atomic<int64_t>  closed_ns {0}; // 0 while the range is being remapped
	std::atomic<bool>     owned {false}; // an open Scope holds this entry
};

class TransientUnmapWindows {
public:
	using Window = TransientUnmapWindow;

	class Scope {
	public:
		Scope(uint64_t start, uint64_t end) noexcept: m_window(Open(start, end)) {}
		~Scope() {
			m_window->closed_ns.store(Now(), std::memory_order_release);
			m_window->owned.store(false, std::memory_order_release);
		}
		KYTY_CLASS_NO_COPY(Scope);

	private:
		Window* m_window;
	};

	// The newest range being remapped recently that holds vaddr (0: none). Waits while it still
	// is being remapped.
	static uint64_t Wait(uint64_t vaddr) noexcept {
		constexpr int64_t Second  = 1000000000;
		uint64_t          matched = 0;
		for (auto& window: s_windows) {
			// The id names one opening of this entry (Open's seqlock). Read it with the range until
			// both belong to the same opening: the entry may be reopening for a range that holds
			// vaddr. id 0 on an owned entry means Open is writing it.
			uint64_t id    = 0;
			uint64_t start = 0;
			uint64_t end   = 0;
			for (;;) {
				id    = window.id.load(std::memory_order_acquire);
				start = window.start.load(std::memory_order_relaxed);
				end   = window.end.load(std::memory_order_relaxed);
				std::atomic_thread_fence(std::memory_order_acquire);
				const bool stable = window.id.load(std::memory_order_relaxed) == id;
				if (stable && (id != 0 || !window.owned.load(std::memory_order_acquire))) {
					break;
				}
#if defined(_M_X64) || defined(__x86_64__)
				_mm_pause();
#endif
			}
			if (id == 0 || vaddr < start || vaddr >= end) {
				continue;
			}
			const auto begin  = Now();
			int64_t    closed = 0;
			for (uint32_t spins = 0;; spins++) {
				closed = window.closed_ns.load(std::memory_order_acquire);
				if (closed != 0) {
					break;
				}
				// Reuse needs the original Scope to have closed first.
				if (window.id.load(std::memory_order_acquire) != id) {
					closed = Now();
					break;
				}
				if (Now() - begin > Second * 10) {
					return 0; // never closed: report the fault
				}
				if (spins < 4096) {
#if defined(_M_X64) || defined(__x86_64__)
					_mm_pause();
#endif
				} else {
					std::this_thread::yield();
				}
			}
			// The fault may be handled after the mapping is already back.
			if (Now() - closed < Second) {
				matched = std::max(matched, id);
			}
		}
		return matched;
	}

private:
	// Claims an entry no open Scope holds, so an open range is never overwritten. Scopes only
	// span one remap, so when every entry is open one frees up shortly.
	static Window* Claim() noexcept {
		for (;;) {
			for (size_t i = 0; i < s_windows.size(); i++) {
				auto& window =
				    s_windows[s_next.fetch_add(1, std::memory_order_relaxed) % s_windows.size()];
				bool expected = false;
				if (window.owned.compare_exchange_strong(expected, true,
				                                         std::memory_order_acquire)) {
					return &window;
				}
			}
			std::this_thread::yield();
		}
	}

	static Window* Open(uint64_t start, uint64_t end) noexcept {
		auto& window = *Claim();
		// Seqlock: id 0 while the range changes, then the new id publishes the range.
		window.id.store(0, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		window.closed_ns.store(0, std::memory_order_relaxed);
		window.start.store(start, std::memory_order_relaxed);
		window.end.store(end, std::memory_order_relaxed);
		window.id.store(s_opened.fetch_add(1, std::memory_order_relaxed) + 1,
		                std::memory_order_seq_cst);
		return &window;
	}

	static int64_t Now() noexcept {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(
		           std::chrono::steady_clock::now().time_since_epoch())
		    .count();
	}

	inline static std::array<Window, 64> s_windows {};
	inline static std::atomic<uint32_t>  s_next {0};
	inline static std::atomic<uint64_t>  s_opened {0};
};

class GuestBackingStore {
public:
	enum class FailureReason {
		None,
		FixedMapViewFailed,
		PlaceholderMapFailed,
		ProtectFailed,
		AddressRangeUnavailable,
	};

	explicit GuestBackingStore(uint64_t size): m_size(size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		m_handle = CreateFileMappingA(
		    INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_COMMIT,
		    static_cast<DWORD>(m_size >> 32u), static_cast<DWORD>(m_size & 0xffffffffu), nullptr);
		if (m_handle == nullptr) {
			EXIT("could not reserve %" PRIu64 " MB for guest direct memory: 0x%08" PRIx32
			     ". Windows commits\n"
			     "this up front, so the paging file is usually what needs to be larger.\n",
			     m_size / (1024U * 1024U), static_cast<uint32_t>(GetLastError()));
		}
		m_backing_base = static_cast<uint8_t*>(
		    MapViewOfFile(m_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, m_size));
		if (m_backing_base == nullptr) {
			EXIT("direct-memory backing: writable alias failed: 0x%08" PRIx32 "\n",
			     static_cast<uint32_t>(GetLastError()));
		}
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#if defined(__APPLE__)
		{
			char shm_name[64];
			snprintf(shm_name, sizeof(shm_name), "/kyty-dm-%d-%llx", static_cast<int>(getpid()),
			         static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this)));
			m_fd = shm_open(shm_name, O_RDWR | O_CREAT | O_EXCL, 0600);
			if (m_fd >= 0) {
				shm_unlink(shm_name);
			}
		}
#else
		if (InitVulkanChunks()) {
			ExcludeFromCoreDumps();
			return;
		}
		m_fd = static_cast<int>(syscall(SYS_memfd_create, "KytyDirectMemory", 0));
#endif
		if (m_fd < 0) {
			EXIT("direct-memory backing: shared fd creation failed: %d\n", errno);
		}
		if (ftruncate(m_fd, static_cast<off_t>(m_size)) != 0) {
			EXIT("direct-memory backing: ftruncate to %" PRIu64 " MB failed: %d\n",
			     m_size / (1024U * 1024U), errno);
		}
		auto* backing = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
		if (backing == MAP_FAILED) {
			EXIT("direct-memory backing: writable alias failed: %d\n", errno);
		}
		m_backing_base = static_cast<uint8_t*>(backing);
#if !defined(__APPLE__)
		ExcludeFromCoreDumps();
#endif
#endif
		Common::MemAccount::NoteGuestBacking(m_size);
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	// A core dump reads every dumpable mapping, and reading shared memory that was never written
	// allocates it: a crash with 12 GB of committed guest memory first grew by all of it (through
	// the alias and the guest views), ran into the memory limit or the machine's memory, and
	// wrote a core of several GB. Shared anonymous mappings (bit 1 of the filter) are left out of
	// the dump; the emulator's own memory stays in. KYTY_COREDUMP_GUEST_MEMORY=1 keeps the default.
	static void ExcludeFromCoreDumps() {
		if (const char* keep = std::getenv("KYTY_COREDUMP_GUEST_MEMORY");
		    keep != nullptr && *keep != '\0' && *keep != '0') {
			return;
		}
		unsigned filter = 0;
		FILE*    file   = std::fopen("/proc/self/coredump_filter", "r+");
		if (file == nullptr) {
			return;
		}
		if (std::fscanf(file, "%x", &filter) == 1 && (filter & 0x2u) != 0) {
			std::rewind(file);
			std::fprintf(file, "0x%x", filter & ~0x2u);
		}
		std::fclose(file);
	}
#endif

	// Guest views outlive the alias and the handle; the address-space owner releases them.
	~GuestBackingStore() {
		for (auto& bucket: m_pages) {
			delete[] bucket.load(std::memory_order_relaxed);
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		UnmapViewOfFile(m_backing_base);
		CloseHandle(m_handle);
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		munmap(m_backing_base, m_size);
		if (m_fd >= 0) {
			close(m_fd);
		}
#if !defined(__APPLE__)
		for (uint32_t i = 0; i < m_chunk_count; i++) {
			if (const int fd = m_chunks[i].fd.load(std::memory_order_relaxed); fd >= 0) {
				close(fd);
			}
		}
		delete[] m_chunks;
		delete m_vulkan;
#endif
#endif
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
	// KYTY_VK_GUEST_BACKING=1: see vulkanGuestBacking.h. The backing is cut into chunks (one
	// Vulkan allocation is limited, 4 GiB on RADV) that are allocated when a guest page in them is
	// first mapped: a Vulkan allocation is committed at once, where the memfd commits only what is
	// touched (GTA VI touches 7-8 GB of its 12.7 GB). Until then the chunk's part of the alias is
	// inaccessible; nothing reads it, since only mapped pages are read through the alias. A chunk
	// Vulkan can't provide (driver refusal, too little GTT or system memory) gets a memfd of its
	// own, so a failure never stops the game.
	static constexpr uint64_t VulkanChunkSize = 1ull << 30u;

	bool InitVulkanChunks() {
		m_vulkan = VulkanGuestAllocator::Create();
		if (m_vulkan == nullptr) {
			return false;
		}
		if (m_vulkan->MaxAllocationSize() != 0 && m_vulkan->MaxAllocationSize() < VulkanChunkSize) {
			std::printf("Vulkan guest backing: allocations are limited to %" PRIu64
			            " MiB, below the %" PRIu64 " MiB chunk; memfd backing\n",
			            m_vulkan->MaxAllocationSize() >> 20u, VulkanChunkSize >> 20u);
			std::fflush(stdout);
			delete m_vulkan;
			m_vulkan = nullptr;
			return false;
		}
		auto* alias = mmap(nullptr, m_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
		                   -1, 0);
		if (alias == MAP_FAILED) {
			delete m_vulkan;
			m_vulkan = nullptr;
			return false;
		}
		m_backing_base = static_cast<uint8_t*>(alias);
		m_chunk_count  = static_cast<uint32_t>((m_size + VulkanChunkSize - 1u) / VulkanChunkSize);
		m_chunks       = new Chunk[m_chunk_count];
		std::printf("Vulkan guest backing: %" PRIu64 " MiB in %u chunks of up to %" PRIu64
		            " MiB, allocated on first use\n",
		            m_size >> 20u, m_chunk_count, VulkanChunkSize >> 20u);
		std::fflush(stdout);
		return true;
	}

	[[nodiscard]] uint64_t ChunkBytes(uint32_t index) const {
		return std::min(VulkanChunkSize, m_size - uint64_t {index} * VulkanChunkSize);
	}

	// Allocates the chunk and maps it into the alias. Callers that map guest views hold the state
	// lock exclusively; the mutex also orders it against ZeroRange and the renderer's lookups.
	bool EnsureChunk(uint32_t index) {
		auto& chunk = m_chunks[index];
		if (chunk.fd.load(std::memory_order_acquire) >= 0) {
			return true;
		}
		std::lock_guard lock(m_chunk_mutex);
		if (chunk.fd.load(std::memory_order_relaxed) >= 0) {
			return true;
		}
		const auto bytes  = ChunkBytes(index);
		// A Vulkan chunk is committed whole where the memfd commits what is touched, so only the
		// first KYTY_VK_GUEST_BACKING_MAX_MB (default 4096) of mapped chunks are Vulkan memory:
		// GTA VI maps its heavy chunks first, and zero-copy buffers came from chunks 0-3, 6, 7.
		static const uint64_t cap = [] {
			const char* text  = std::getenv("KYTY_VK_GUEST_BACKING_MAX_MB");
			const auto  value = text != nullptr ? std::strtoull(text, nullptr, 10) : 4096ull;
			return value << 20u;
		}();
		const bool over_cap = m_vulkan_bytes + bytes > cap;
		int        fd       = over_cap ? -1 : m_vulkan->Allocate(bytes);
		bool       vulkan   = fd >= 0;
		if (over_cap && !m_cap_logged) {
			m_cap_logged = true;
			std::printf("Vulkan guest backing: %" PRIu64 " MiB cap reached, further chunks are "
			            "memfds (KYTY_VK_GUEST_BACKING_MAX_MB)\n",
			            cap >> 20u);
			std::fflush(stdout);
		}
		if (!vulkan) {
			fd = static_cast<int>(syscall(SYS_memfd_create, "KytyDirectMemory", 0));
			if (fd < 0 || ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
				EXIT("direct-memory backing: memfd for chunk %u failed: %d\n", index, errno);
			}
			if (m_memfd_chunks++ == 0 && !over_cap) {
				std::printf("Vulkan guest backing: chunk %u falls back to a memfd (no zero-copy "
				            "there)\n",
				            index);
				std::fflush(stdout);
			}
		}
		if (mmap(m_backing_base + uint64_t {index} * VulkanChunkSize, bytes, PROT_READ | PROT_WRITE,
		         MAP_SHARED | MAP_FIXED, fd, 0) == MAP_FAILED) {
			EXIT("direct-memory backing: alias of chunk %u failed: %d\n", index, errno);
		}
		chunk.vulkan = vulkan;
		chunk.fd.store(fd, std::memory_order_release);
		if (vulkan) {
			m_vulkan_bytes += bytes;
			std::printf("Vulkan guest backing: chunk %u (%" PRIu64 " MiB) allocated, %" PRIu64
			            " MiB in Vulkan memory\n",
			            index, bytes >> 20u, m_vulkan_bytes >> 20u);
			std::fflush(stdout);
		}
		return true;
	}

	// The file and file offset that hold backing byte `offset`, and how many bytes from there
	// lie in that file. -1 for a chunk that was never allocated.
	[[nodiscard]] int FileAt(uint64_t offset, uint64_t* file_offset, uint64_t* run) const {
		if (m_chunks == nullptr) {
			*file_offset = offset;
			*run         = m_size - offset;
			return m_fd;
		}
		const auto index = static_cast<uint32_t>(offset / VulkanChunkSize);
		*file_offset     = offset % VulkanChunkSize;
		*run             = ChunkBytes(index) - *file_offset;
		return m_chunks[index].fd.load(std::memory_order_acquire);
	}

public:
	[[nodiscard]] bool IsVulkanBacked() const { return m_vulkan != nullptr; }

	[[nodiscard]] const VulkanGuestAllocator* GetVulkanAllocator() const { return m_vulkan; }

	// The Vulkan chunk that holds all of [backing_offset, backing_offset + size), with the offset
	// in it. False when the range spans chunks or its chunk is a memfd.
	bool FindVulkanChunk(uint64_t backing_offset, uint64_t size, uint32_t* index,
	                     uint64_t* chunk_offset, uint64_t* chunk_size) const {
		if (m_chunks == nullptr || size == 0 || backing_offset >= m_size ||
		    m_size - backing_offset < size) {
			return false;
		}
		const auto i = static_cast<uint32_t>(backing_offset / VulkanChunkSize);
		if ((backing_offset + size - 1u) / VulkanChunkSize != i ||
		    m_chunks[i].fd.load(std::memory_order_acquire) < 0 || !m_chunks[i].vulkan) {
			return false;
		}
		*index        = i;
		*chunk_offset = backing_offset % VulkanChunkSize;
		*chunk_size   = ChunkBytes(i);
		return true;
	}

	// A new descriptor of a Vulkan chunk for the renderer's import, which takes ownership.
	[[nodiscard]] int DupVulkanChunk(uint32_t index) const {
		if (m_chunks == nullptr || index >= m_chunk_count || !m_chunks[index].vulkan) {
			return -1;
		}
		const int fd = m_chunks[index].fd.load(std::memory_order_acquire);
		return fd >= 0 ? fcntl(fd, F_DUPFD_CLOEXEC, 0) : -1;
	}

	[[nodiscard]] uint32_t TestChunkCount() const { return m_chunk_count; }
#endif

	KYTY_CLASS_NO_COPY(GuestBackingStore);

	static const char* GetFailureReasonName(FailureReason reason) {
		switch (reason) {
			case FailureReason::None: return "n/a";
			case FailureReason::FixedMapViewFailed: return "fixed-map-view-failed";
			case FailureReason::PlaceholderMapFailed: return "placeholder-map-failed";
			case FailureReason::ProtectFailed: return "protect-failed";
			case FailureReason::AddressRangeUnavailable: return "address-range-unavailable";
			default: return "unknown";
		}
	}

	[[nodiscard]] uint64_t GetBackingBase() const {
		return reinterpret_cast<uint64_t>(m_backing_base);
	}

	[[nodiscard]] uint64_t GetBackingSize() const { return m_size; }

	void ZeroRange(uint64_t backing_offset, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
		if (m_chunks != nullptr) {
			// A chunk that was never allocated is zero when it is.
			for (auto offset = backing_offset; offset < backing_offset + size;) {
				uint64_t   file_offset = 0;
				uint64_t   run         = 0;
				const auto fd          = FileAt(offset, &file_offset, &run);
				const auto bytes       = std::min(run, backing_offset + size - offset);
				if (fd >= 0) {
					std::memset(m_backing_base + offset, 0, bytes);
				}
				offset += bytes;
			}
		} else
#endif
		std::memset(m_backing_base + backing_offset, 0, size);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		g_test_backing_bytes_zeroed += size;
#endif
	}

	// Callers hold the state lock, so no view changes under the write. Unbacked gaps are skipped,
	// mirroring sparse reads that zero-fill them.
	void WriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		const auto* source = static_cast<const uint8_t*>(data);
		ForEachRun(
		    vaddr, size,
		    [&](uint64_t offset, uint64_t start, uint64_t bytes) {
			    std::memcpy(m_backing_base + offset, source + (start - vaddr), bytes);
		    },
		    [](uint64_t, uint64_t) {});
	}

	// Needs no lock: page entries are atomic and buckets live as long as the store.
	[[nodiscard]] bool IsBacked(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		for (auto page = vaddr >> PageBits; page <= (vaddr + size - 1u) >> PageBits; ++page) {
			if (LoadPage(page) == 0) {
				return false;
			}
		}
		return true;
	}

	// Changes whenever a guest page is bound to or unbound from the backing, so holders of a
	// backing address (zero-copy GPU buffers) know to check it again.
	[[nodiscard]] uint64_t Generation() const noexcept {
		return m_generation.load(std::memory_order_acquire);
	}

	// Whether a page change after `generation` touched [vaddr, vaddr + size). True when the
	// history no longer reaches back that far.
	[[nodiscard]] bool ChangedSince(uint64_t generation, uint64_t vaddr, uint64_t size) const {
		std::lock_guard lock(m_changes_mutex);
		const auto      now = m_generation.load(std::memory_order_acquire);
		if (now - generation >= m_changes.size()) {
			return true;
		}
		for (auto g = generation + 1; g <= now; ++g) {
			const auto& change = m_changes[g % m_changes.size()];
			if (change.generation != g ||
			    (change.start < vaddr + size && vaddr < change.end)) {
				return true;
			}
		}
		return false;
	}

	// The alias address of [vaddr, vaddr + size) when one contiguous backing run holds all of it,
	// 0 otherwise. Callers hold the state lock.
	[[nodiscard]] uint64_t ContiguousAlias(uint64_t vaddr, uint64_t size) const {
		uint64_t alias  = 0;
		uint64_t runs   = 0;
		bool     gapped = false;
		ForEachRun(
		    vaddr, size,
		    [&](uint64_t offset, uint64_t start, uint64_t bytes) {
			    runs++;
			    if (start == vaddr && bytes == size) {
				    alias = reinterpret_cast<uint64_t>(m_backing_base) + offset;
			    }
		    },
		    [&](uint64_t, uint64_t) { gapped = true; });
		return runs == 1 && !gapped ? alias : 0;
	}

	// Where a guest page lives in the backing, or UINT64_MAX for a page that has none.
	[[nodiscard]] uint64_t BackingOffset(uint64_t vaddr) const {
		const auto entry = LoadPage(vaddr >> PageBits);
		return entry != 0 ? (uint64_t {entry - 1u} << PageBits) + (vaddr & PageMask) : UINT64_MAX;
	}

	// Needs no lock either, as the alias stays mapped. A read racing an unmap copies the backing's
	// current bytes, or zeros for a page unmapped between the check and the copy.
	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) const {
		if (data == nullptr || !IsBacked(vaddr, size)) {
			return false;
		}
		ReadSparseBacking(vaddr, data, size, [](uint64_t, uint8_t* destination, uint64_t bytes) {
			std::memset(destination, 0, bytes);
		});
		return true;
	}

	// The alias address of a range that one contiguous run of backing holds, or nullptr. Needs
	// no lock, like TryReadBacking; the alias stays mapped for the store's lifetime.
	[[nodiscard]] const uint8_t* TryGetBackingView(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return nullptr;
		}
		const auto first = LoadPage(vaddr >> PageBits);
		if (first == 0) {
			return nullptr;
		}
		auto expected = first;
		for (auto page = (vaddr >> PageBits) + 1u; page <= (vaddr + size - 1u) >> PageBits; ++page) {
			if (LoadPage(page) != ++expected) {
				return nullptr;
			}
		}
		return m_backing_base + ((uint64_t {first - 1u} << PageBits) + (vaddr & PageMask));
	}

	// The same bytes as TryReadBacking, read through the file where the host can: a file read
	// of shared memory that was never written returns zeros without allocating it, where a copy
	// from the mapping commits every page it touches. For readers of whole address spaces (frame
	// capture); small hot reads keep TryReadBacking, which needs no system call.
	bool TryReadBackingUncommitted(uint64_t vaddr, void* data, uint64_t size) const {
		if (data == nullptr || !IsBacked(vaddr, size)) {
			return false;
		}
		auto* destination = static_cast<uint8_t*>(data);
		ForEachRun(
		    vaddr, size,
		    [&](uint64_t offset, uint64_t start, uint64_t bytes) {
			    auto* out = destination + (start - vaddr);
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
			    // A dma-buf has no read(): its pages are all committed anyway, so the copy below
			    // costs nothing extra. Runs may span chunks; each piece reads its own file.
			    while (bytes != 0) {
				    uint64_t   file_offset = 0;
				    uint64_t   file_run    = 0;
				    const auto fd          = FileAt(offset, &file_offset, &file_run);
				    if (fd < 0 || (m_chunks != nullptr && m_chunks[offset / VulkanChunkSize].vulkan)) {
					    break;
				    }
				    const auto read = pread(fd, out, std::min(bytes, file_run),
				                            static_cast<off_t>(file_offset));
				    if (read < 0 && errno == EINTR) {
					    continue;
				    }
				    if (read <= 0) {
					    break;
				    }
				    out += read;
				    offset += static_cast<uint64_t>(read);
				    bytes -= static_cast<uint64_t>(read);
			    }
#endif
			    std::memcpy(out, m_backing_base + offset, bytes);
		    },
		    [&](uint64_t start, uint64_t bytes) {
			    std::memset(destination + (start - vaddr), 0, bytes);
		    });
		return true;
	}

	template <typename Gap>
	void ReadSparseBacking(uint64_t vaddr, void* data, uint64_t size, Gap&& gap) const {
		auto* destination = static_cast<uint8_t*>(data);
		ForEachRun(
		    vaddr, size,
		    [&](uint64_t offset, uint64_t start, uint64_t bytes) {
			    std::memcpy(destination + (start - vaddr), m_backing_base + offset, bytes);
		    },
		    [&](uint64_t start, uint64_t bytes) {
			    gap(start, destination + (start - vaddr), bytes);
		    });
	}

	// Mapping and unmapping hold the state lock exclusively.
	bool MapFixed(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	              VirtualMemory::Mode mode, FailureReason* failure_reason = nullptr) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (const auto callback = g_test_before_backing_map) {
			g_test_before_backing_map = nullptr;
			callback(vaddr, size);
		}
#endif
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		if (!ReplacePlaceholderView(vaddr, size, backing_offset, mode, failure_reason)) {
			return false;
		}
#else
		if (!MapView(vaddr, size, backing_offset, mode, failure_reason)) {
			return false;
		}
#endif
		StorePages(vaddr, size, backing_offset);
		return true;
	}

	// The caller validated the span; failing here is an emulator fault.
	void Unmap(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const auto end   = vaddr + size;
		uint64_t   views = 0;
		for (auto current = vaddr; current < end; ++views) {
			const auto view     = QueryView(current);
			const auto part_end = std::min(end, view.base + view.size);
			UnmapOne(view, current, part_end - current);
			current = part_end;
		}
		// Each unmapped file view becomes a separate Windows placeholder. The address-space owner
		// records this contiguous guest span as one free range, so normalize fragmented backing
		// views into one placeholder before publishing it there.
		EXIT_IF(views > 1 && !CoalescePlaceholders(vaddr, size));
#else
		// Replace only the requested range so concurrent accesses and GPU watch
		// protections on the surviving mappings remain intact.
		EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(vaddr), size));
		ClearPages(vaddr, size);
#endif
	}

private:
	static constexpr uint64_t PageBits    = 14;
	static constexpr uint64_t PageMask    = (uint64_t {1} << PageBits) - 1u;
	static constexpr uint64_t BucketPages = uint64_t {1} << 16u;
	static constexpr uint64_t BucketCount =
	    ((kExtendedMemoryBase + kExtendedMemorySize) >> PageBits) / BucketPages;

	// Zero for an unbacked page, otherwise the backing page index plus one.
	[[nodiscard]] uint32_t LoadPage(uint64_t page) const {
		if (page / BucketPages >= BucketCount) {
			return 0;
		}
		const auto* entries = m_pages[page / BucketPages].load(std::memory_order_acquire);
		return entries != nullptr ? entries[page % BucketPages].load(std::memory_order_relaxed) : 0;
	}

	// Walks the span in runs of contiguous backing, as (backing offset, start, bytes), and runs of
	// unbacked pages, as (start, bytes).
	template <typename Region, typename Gap>
	void ForEachRun(uint64_t vaddr, uint64_t size, Region&& region, Gap&& gap) const {
		const auto end = vaddr + size;
		for (auto start = vaddr; start < end;) {
			const auto     entry    = LoadPage(start >> PageBits);
			const uint32_t step     = entry != 0 ? 1 : 0;
			auto           expected = entry + step;
			auto           stop     = (start | PageMask) + 1u;
			while (stop < end && LoadPage(stop >> PageBits) == expected) {
				stop += PageMask + 1u;
				expected += step;
			}
			stop = std::min(stop, end);
			if (entry != 0) {
				region((uint64_t {entry - 1u} << PageBits) + (start & PageMask), start,
				       stop - start);
			} else {
				gap(start, stop - start);
			}
			start = stop;
		}
	}

	void NoteChange(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_changes_mutex);
		const auto      generation = m_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
		m_changes[generation % m_changes.size()] = {generation, vaddr, vaddr + size};
	}

	void StorePages(uint64_t vaddr, uint64_t size, uint64_t backing_offset) {
		NoteChange(vaddr, size);
		const auto end = vaddr + size;
		EXIT_IF(((vaddr | size | backing_offset) & PageMask) != 0 ||
		        (end >> PageBits) > BucketCount * BucketPages || backing_offset + size > m_size);
		auto entry = static_cast<uint32_t>(backing_offset >> PageBits) + 1u;
		for (auto page = vaddr >> PageBits; page < end >> PageBits; ++page, ++entry) {
			auto& bucket  = m_pages[page / BucketPages];
			auto* entries = bucket.load(std::memory_order_relaxed);
			if (entries == nullptr) {
				entries = new std::atomic<uint32_t>[BucketPages] {};
				bucket.store(entries, std::memory_order_release);
			}
			entries[page % BucketPages].store(entry, std::memory_order_relaxed);
		}
	}

	void ClearPages(uint64_t vaddr, uint64_t size) {
		NoteChange(vaddr, size);
		for (auto page = vaddr >> PageBits; page < (vaddr + size) >> PageBits; ++page) {
			m_pages[page / BucketPages].load(std::memory_order_relaxed)[page % BucketPages].store(
			    0, std::memory_order_relaxed);
		}
	}

	static void SetFailure(FailureReason* out, FailureReason reason) {
		if (out != nullptr) {
			*out = reason;
		}
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}

	// Collects the page runs of the span whose protection differs from the view's mode.
	static void QueryProtections(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode,
	                             std::vector<MEMORY_BASIC_INFORMATION>* out) {
		const auto end = vaddr + size;
		for (auto addr = vaddr; addr < end;) {
			MEMORY_BASIC_INFORMATION info {};
			EXIT_IF(VirtualQuery(reinterpret_cast<void*>(addr), &info, sizeof(info)) == 0);
			info.RegionSize = std::min<SIZE_T>(info.RegionSize, end - addr);
			if (info.Protect != GetProtect(mode)) {
				out->push_back(info);
			}
			addr += info.RegionSize;
		}
	}

	static bool ApplyProtection(void* ptr, uint64_t size, DWORD protect) {
		DWORD old_protect = 0;
		return VirtualProtect(ptr, size, protect, &old_protect) != 0;
	}

	static bool SplitPlaceholder(uint64_t vaddr, uint64_t size) {
		return VirtualFree(reinterpret_cast<void*>(vaddr), size,
		                   MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	struct View {
		uint64_t base    = 0;
		uint64_t size    = 0;
		DWORD    protect = 0;
	};

	// The OS records a view as the regions sharing its allocation base; guest and tracker
	// protections split it into several.
	static View QueryView(uint64_t vaddr) {
		MEMORY_BASIC_INFORMATION info {};
		EXIT_IF(VirtualQuery(reinterpret_cast<void*>(vaddr), &info, sizeof(info)) == 0 ||
		        info.Type != MEM_MAPPED);
		View view {reinterpret_cast<uint64_t>(info.AllocationBase), 0, info.AllocationProtect};
		auto end = reinterpret_cast<uint64_t>(info.BaseAddress) + info.RegionSize;
		while (VirtualQuery(reinterpret_cast<void*>(end), &info, sizeof(info)) != 0 &&
		       reinterpret_cast<uint64_t>(info.AllocationBase) == view.base) {
			end += info.RegionSize;
		}
		view.size = end - view.base;
		return view;
	}

	void UnmapOne(const View& view, uint64_t vaddr, uint64_t size) {
		const uint64_t offset     = uint64_t {LoadPage(view.base >> PageBits) - 1u} << PageBits;
		const uint64_t left_size  = vaddr - view.base;
		const uint64_t right_addr = vaddr + size;
		const uint64_t right_size = view.base + view.size - right_addr;
		// Survivors are re-mapped read-write, executable if the view was, and the protections guest
		// mprotect and write tracking set on them are re-applied afterwards.
		const auto mode =
		    (view.protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE)) != 0
		        ? VirtualMemory::Mode::ExecuteReadWrite
		        : VirtualMemory::Mode::ReadWrite;
		std::vector<MEMORY_BASIC_INFORMATION> protections;
		QueryProtections(view.base, left_size, mode, &protections);
		QueryProtections(right_addr, right_size, mode, &protections);

		// The survivors are briefly unmapped too: their faults wait and are retried until they are
		// back with their protections.
		std::optional<TransientUnmapWindows::Scope> left_window;
		std::optional<TransientUnmapWindows::Scope> right_window;
		if (left_size != 0) {
			left_window.emplace(view.base, vaddr);
		}
		if (right_size != 0) {
			right_window.emplace(right_addr, right_addr + right_size);
		}
		EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(view.base), view.size));
		ClearPages(vaddr, size);
		EXIT_IF(left_size != 0 && !SplitPlaceholder(view.base, left_size));
		EXIT_IF(right_size != 0 && !SplitPlaceholder(vaddr, size));
		EXIT_IF(left_size != 0 && !MapFixed(view.base, left_size, offset, mode));
		EXIT_IF(right_size != 0 &&
		        !MapFixed(right_addr, right_size, offset + (right_addr - view.base), mode));
		for (const auto& info: protections) {
			EXIT_IF(!ApplyProtection(info.BaseAddress, info.RegionSize, info.Protect));
		}
	}

	bool ReplacePlaceholderView(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                            VirtualMemory::Mode mode, FailureReason* failure_reason) {
		const DWORD map_protect =
		    mode == VirtualMemory::Mode::NoAccess ? PAGE_READWRITE : GetProtect(mode);
		void* ptr =
		    MapViewOfFile3(m_handle, GetCurrentProcess(), reinterpret_cast<void*>(vaddr),
		                   backing_offset, size, MEM_REPLACE_PLACEHOLDER, map_protect, nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			}
			SetFailure(failure_reason, FailureReason::PlaceholderMapFailed);
			return false;
		}

		if (mode == VirtualMemory::Mode::NoAccess && !ApplyProtection(ptr, size, PAGE_NOACCESS)) {
			UnmapViewOfFile2(GetCurrentProcess(), ptr, MEM_PRESERVE_PLACEHOLDER);
			SetFailure(failure_reason, FailureReason::ProtectFailed);
			return false;
		}
		return true;
	}

	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t /*size*/) {
		return UnmapViewOfFile2(GetCurrentProcess(), vaddr, MEM_PRESERVE_PLACEHOLDER) != 0;
	}

	HANDLE m_handle = nullptr;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	static int GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PROT_READ;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::Execute: return PROT_EXEC;
			case VirtualMemory::Mode::ExecuteRead: return PROT_EXEC | PROT_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PROT_EXEC | PROT_READ | PROT_WRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PROT_NONE;
		}
	}

	bool MapView(uint64_t vaddr, uint64_t size, uint64_t backing_offset, VirtualMemory::Mode mode,
	             FailureReason* failure_reason) {
#if !defined(__APPLE__)
		if (m_chunks != nullptr) {
			// One mapping per chunk the view spans.
			for (uint64_t done = 0; done < size;) {
				const auto offset = backing_offset + done;
				EXIT_IF(offset >= m_size);
				const auto index = static_cast<uint32_t>(offset / VulkanChunkSize);
				EnsureChunk(index);
				uint64_t   file_offset = 0;
				uint64_t   run         = 0;
				const auto fd          = FileAt(offset, &file_offset, &run);
				const auto bytes       = std::min(run, size - done);
				if (mmap(reinterpret_cast<void*>(vaddr + done), bytes, GetProtect(mode),
				         MAP_SHARED | MAP_FIXED, fd, static_cast<off_t>(file_offset)) ==
				    MAP_FAILED) {
					if (done != 0) {
						EXIT_IF(!UnmapViewPreservePlaceholder(reinterpret_cast<void*>(vaddr), done));
					}
					SetFailure(failure_reason, FailureReason::FixedMapViewFailed);
					return false;
				}
				done += bytes;
			}
			return true;
		}
#endif
		if (mmap(reinterpret_cast<void*>(vaddr), size, GetProtect(mode), MAP_SHARED | MAP_FIXED,
		         m_fd, static_cast<off_t>(backing_offset)) == MAP_FAILED) {
			SetFailure(failure_reason, FailureReason::FixedMapViewFailed);
			return false;
		}
		return true;
	}

	// Replace the file view atomically instead of exposing an unmapped hole in the
	// process-lifetime guest reservation.
	static bool UnmapViewPreservePlaceholder(void* vaddr, uint64_t size) {
		return mmap(vaddr, size, PROT_NONE, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0) !=
		       MAP_FAILED;
	}

	int m_fd = -1;
#if !defined(__APPLE__)
	struct Chunk {
		std::atomic<int> fd {-1};
		bool             vulkan = false; // written before fd is published
	};
	VulkanGuestAllocator* m_vulkan       = nullptr;
	Chunk*                m_chunks       = nullptr;
	uint32_t              m_chunk_count  = 0;
	uint32_t              m_memfd_chunks = 0;
	uint64_t              m_vulkan_bytes = 0;
	bool                  m_cap_logged   = false;
	std::mutex            m_chunk_mutex;
#endif
#endif

	uint8_t*                                                     m_backing_base = nullptr;
	uint64_t                                                     m_size         = 0;
	std::array<std::atomic<std::atomic<uint32_t>*>, BucketCount> m_pages {};
	std::atomic<uint64_t>                                        m_generation {1};
	struct PageChange {
		uint64_t generation = 0;
		uint64_t start      = 0;
		uint64_t end        = 0;
	};
	mutable std::mutex                                           m_changes_mutex;
	std::array<PageChange, 1024>                                 m_changes {};
};

// Callers change it with the state lock held exclusively and query it under the operation lock or
// the state lock. TryWriteBacking, CopySparseMemory and ProtectTransient take the state lock shared
// themselves.
class GuestAddressSpace {
public:
	explicit GuestAddressSpace(uint64_t backing_size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		SYSTEM_INFO info {};
		GetSystemInfo(&info);
		m_granularity = info.dwAllocationGranularity;
#endif
		ReserveGuestRegions();
		m_backing = new GuestBackingStore(backing_size);
	}

	~GuestAddressSpace() {
		delete m_backing;
		m_backing = nullptr;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [start, size]: m_owned) {
			const auto end     = start + size;
			auto       current = start;
			while (current < end) {
				MEMORY_BASIC_INFORMATION info {};
				EXIT_IF(VirtualQuery(reinterpret_cast<const void*>(current), &info, sizeof(info)) ==
				        0);
				const auto region_start = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end   = region_start + static_cast<uint64_t>(info.RegionSize);
				if (info.Type == MEM_MAPPED) {
					EXIT_IF(UnmapViewOfFile2(GetCurrentProcess(), info.AllocationBase,
					                         MEM_PRESERVE_PLACEHOLDER) == 0);
					continue;
				}
				if (info.State == MEM_COMMIT) {
					EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_start), info.RegionSize,
					                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
					continue;
				}
				if (info.State == MEM_RESERVE) {
					EXIT_IF(VirtualFree(info.AllocationBase, 0, MEM_RELEASE) == 0);
				}
				current = region_end;
			}
		}
#else
		for (const auto& [start, size]: m_owned) {
			munmap(reinterpret_cast<void*>(start), size);
		}
#endif
	}

	KYTY_CLASS_NO_COPY(GuestAddressSpace);

	[[nodiscard]] uint64_t GetGranularity() const { return m_granularity; }
	[[nodiscard]] uint64_t GetBackingBase() const { return m_backing->GetBackingBase(); }
	[[nodiscard]] uint64_t GetBackingSize() const { return m_backing->GetBackingSize(); }
	[[nodiscard]] GuestBackingStore& GetBacking() const { return *m_backing; }

	void ZeroBacking(uint64_t backing_offset, uint64_t size) {
		m_backing->ZeroRange(backing_offset, size);
	}

	// The owned ranges are fixed at construction.
	[[nodiscard]] bool Owns(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || UINT64_MAX - vaddr < size) {
			return false;
		}
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return vaddr >= range.first && vaddr + size <= range.first + range.second;
		});
	}

#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
	[[nodiscard]] bool OverlapsOwned(uint64_t vaddr, uint64_t size) const {
		return std::any_of(m_owned.begin(), m_owned.end(), [vaddr, size](const auto& range) {
			return VirtualRangesOverlap(vaddr, size, range.first, range.second);
		});
	}
#endif

	// Writes the backed bytes of a range that holds no private mappings.
	[[nodiscard]] uint64_t BackingGeneration() const { return m_backing->Generation(); }
	[[nodiscard]] bool     BackingChangedSince(uint64_t generation, uint64_t vaddr,
	                                           uint64_t size) const {
		return m_backing->ChangedSince(generation, vaddr, size);
	}

	// See GuestBackingStore::ContiguousAlias; 0 also when a private mapping overlaps the range.
	[[nodiscard]] uint64_t ContiguousBackingAlias(uint64_t vaddr, uint64_t size) {
		if (!Owns(vaddr, size)) {
			return 0;
		}
		std::shared_lock state(g_memory_state_mutex);
		if (OverlapsPrivateUnlocked(vaddr, size)) {
			return 0;
		}
		return m_backing->ContiguousAlias(vaddr, size);
	}

	bool TryWriteBacking(uint64_t vaddr, const void* data, uint64_t size) {
		if (!Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock state(g_memory_state_mutex);
		if (OverlapsPrivateUnlocked(vaddr, size)) {
			return false;
		}
		m_backing->WriteBacking(vaddr, data, size);
		return true;
	}

	bool TryReadBacking(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBacking(vaddr, data, size);
	}

	bool TryReadBackingUncommitted(uint64_t vaddr, void* data, uint64_t size) {
		return m_backing->TryReadBackingUncommitted(vaddr, data, size);
	}

	[[nodiscard]] const void* TryGetBackingView(uint64_t vaddr, uint64_t size) const {
		return m_backing->TryGetBackingView(vaddr, size);
	}

	[[nodiscard]] uint64_t BackingOffset(uint64_t vaddr) const {
		return m_backing->BackingOffset(vaddr);
	}

	[[nodiscard]] bool IsBacked(uint64_t vaddr, uint64_t size) const {
		return m_backing->IsBacked(vaddr, size);
	}

	template <bool copy_private>
	bool CopySparseMemory(uint64_t vaddr, void* data, uint64_t size) {
		if (m_backing->TryReadBacking(vaddr, data, size)) {
			return true;
		}
		if (data == nullptr || !Owns(vaddr, size)) {
			return false;
		}
		std::shared_lock state(g_memory_state_mutex);
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
		if (g_test_backing_read != nullptr) {
			g_test_backing_read(vaddr, size);
		}
#endif
		if constexpr (!copy_private) {
			if (OverlapsPrivateUnlocked(vaddr, size)) {
				return false;
			}
		}
		m_backing->ReadSparseBacking(
		    vaddr, data, size, [&](uint64_t start, uint8_t* destination, uint64_t bytes) {
			    if constexpr (copy_private) {
				    // Backing gaps can contain private mappings or reservations only.
				    CopyPrivateUnlocked(start, destination, bytes);
			    } else {
				    std::memset(destination, 0, bytes);
			    }
		    });
		return true;
	}

	bool Protect(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	// Tracker protections only read m_mapped; the page tracker orders them per page.
	bool ProtectTransient(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		std::shared_lock state(g_memory_state_mutex);
		return ProtectMappedUnlocked(vaddr, size, mode);
	}

	bool MapBacking(uint64_t vaddr, uint64_t size, uint64_t backing_offset,
	                VirtualMemory::Mode               mode,
	                GuestBackingStore::FailureReason* failure_reason = nullptr) {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
		const bool replacing = ContainsMappedUnlocked(vaddr, size, MappingType::Backing);
#else
		constexpr bool replacing = false;
#endif
		if (!replacing && !ConsumeUnlocked(vaddr, size)) {
			if (failure_reason != nullptr) {
				*failure_reason = GuestBackingStore::FailureReason::AddressRangeUnavailable;
			}
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;
		if (m_backing->MapFixed(vaddr, size, backing_offset, host_mode, failure_reason)) {
			if (!replacing) {
				AddMappedUnlocked(vaddr, size, MappingType::Backing);
			}
			return true;
		}
		if (!replacing) {
			AddFreeUnlocked(vaddr, size);
		}
		return false;
	}

	bool UnmapBacking(uint64_t vaddr, uint64_t size) {
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Backing)) {
			return false;
		}
		m_backing->Unmap(vaddr, size);
		RemoveMappedUnlocked(vaddr, size);
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	uint64_t FindFreeAligned(uint64_t search_start, uint64_t search_end, uint64_t size,
	                         uint64_t alignment) {
		alignment = std::max<uint64_t>(alignment, PageSize());
		return FindFreeAlignedUnlocked(search_start, search_end, size, alignment);
	}

	bool ContainsFree(uint64_t vaddr, uint64_t size) {
		return FindContaining(vaddr, size) != m_free.end();
	}

	bool Commit(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		if (!ConsumeUnlocked(vaddr, size)) {
			return false;
		}
		const auto host_mode = VirtualMemory::IsExecute(mode)
		                           ? VirtualMemory::Mode::ExecuteReadWrite
		                           : VirtualMemory::Mode::ReadWrite;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		auto* ptr = VirtualAlloc2(GetCurrentProcess(), reinterpret_cast<void*>(vaddr), size,
		                          MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
		                          GetProtect(host_mode), nullptr, 0);
		if (ptr == nullptr || reinterpret_cast<uint64_t>(ptr) != vaddr) {
			if (ptr != nullptr) {
				VirtualFree(ptr, 0, MEM_RELEASE);
			}
			AddFreeUnlocked(vaddr, size);
			return false;
		}
		AddMappedUnlocked(vaddr, size, MappingType::Private);
		return true;
#else
		if (VirtualMemory::Protect(vaddr, size, host_mode)) {
			AddMappedUnlocked(vaddr, size, MappingType::Private);
			return true;
		}
		AddFreeUnlocked(vaddr, size);
		return false;
#endif
	}

	bool ReleaseCommitted(uint64_t vaddr, uint64_t size) {
		if (!IsPageAligned(vaddr) || !IsPageAligned(size)) {
			return false;
		}
		if (!ContainsMappedUnlocked(vaddr, size, MappingType::Private)) {
			return false;
		}
		const auto end     = vaddr + size;
		auto       current = vaddr;
		std::vector<std::pair<uint64_t, uint64_t>> regions;
		while (current < end) {
			const auto it       = FindMappedContainingUnlocked(current);
			const auto part_end = std::min(end, it->second.vaddr + it->second.size);
			regions.emplace_back(current, part_end - current);
			current = part_end;
		}
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		for (const auto& [region_vaddr, region_size]: regions) {
			EXIT_IF(VirtualFree(reinterpret_cast<void*>(region_vaddr), region_size,
			                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0);
		}
		// ReleaseCommitted also publishes the aggregate span as one logical free range.
		EXIT_IF(regions.size() > 1 && !CoalescePlaceholders(vaddr, size));
#else
		for (const auto& [region_vaddr, region_size]: regions) {
			RestorePlaceholder(region_vaddr, region_size);
		}
#endif
		RemoveMappedUnlocked(vaddr, size);
		AddFreeUnlocked(vaddr, size);
		return true;
	}

	bool SelfTest() {
		auto addr = FindFreeAligned(HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_MANAGED_MAX + 1u,
		                            PageSize(), PageSize());
		if (addr == 0) {
			addr = FindFreeAligned(HOST_USER_MIN, HOST_USER_MAX + 1u, PageSize(), PageSize());
		}
		if (addr == 0) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: 16 KiB range unavailable\n");
			return false;
		}
		if (!Commit(addr, PageSize(), VirtualMemory::Mode::ReadWrite)) {
			LOGF_COLOR(Log::Color::Red, "\t guest address-space self-test: commit failed\n");
			return false;
		}

		auto* ptr = reinterpret_cast<uint64_t*>(addr);
		*ptr      = 0x4b59545950484f4cull; // "KYTYPHOL"

		if (!ReleaseCommitted(addr, PageSize())) {
			LOGF_COLOR(Log::Color::Red,
			           "\t guest address-space self-test: release-committed failed\n");
			return false;
		}

		LOGF_COLOR(Log::Color::Green, "\t guest address-space self-test: ok\n");
		return true;
	}

private:
	enum class MappingType { Backing, Private };

	struct MappedRegion {
		uint64_t    vaddr = 0;
		uint64_t    size  = 0;
		MappingType type  = MappingType::Private;
	};

	static constexpr uint64_t PageSize() { return 0x4000; }

	static uint64_t AlignUp(uint64_t value, uint64_t alignment) {
		const auto remainder = value % alignment;
		return remainder != 0 ? value + (alignment - remainder) : value;
	}

	static uint64_t AlignDown(uint64_t value, uint64_t alignment) {
		return value & ~(alignment - 1u);
	}

	[[nodiscard]] static bool IsPageAligned(uint64_t value) {
		return (value & (PageSize() - 1u)) == 0;
	}

	bool ProtectMappedUnlocked(uint64_t vaddr, uint64_t size, VirtualMemory::Mode mode) {
		const auto end = vaddr + size;
		auto       it  = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto& region      = it->second;
			const auto  range_start = std::max(vaddr, region.vaddr);
			const auto  range_end   = std::min(end, region.vaddr + region.size);
			if (range_start >= range_end) {
				continue;
			}
			if (!VirtualMemory::Protect(range_start, range_end - range_start, mode)) {
				return false;
			}
		}
		return true;
	}

	bool ConsumeUnlocked(uint64_t vaddr, uint64_t size) {
		const auto it = FindContaining(vaddr, size);
		if (it == m_free.end() || !SplitUnlocked(it, vaddr, size)) {
			return false;
		}
		m_free.erase(vaddr);
		return true;
	}

	std::map<uint64_t, MappedRegion>::iterator FindMappedContainingUnlocked(uint64_t vaddr) {
		auto next = m_mapped.upper_bound(vaddr);
		if (next == m_mapped.begin()) {
			return m_mapped.end();
		}
		auto it = std::prev(next);
		return vaddr < it->second.vaddr + it->second.size ? it : m_mapped.end();
	}

	bool ContainsMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			const auto it = FindMappedContainingUnlocked(current);
			if (it == m_mapped.end() || it->second.type != type) {
				return false;
			}
			current = std::min(end, it->second.vaddr + it->second.size);
		}
		return true;
	}

	[[nodiscard]] bool OverlapsPrivateUnlocked(uint64_t vaddr, uint64_t size) const {
		// Backing-store writes and reads ask this for a few bytes at a time, mostly within the
		// mapping they asked for last (labels, query results, WRITE_DATA): remember, per thread,
		// the last shared mapping that held the whole range until the mappings change.
		struct LastShared {
			const void* space      = nullptr;
			uint64_t    generation = 0;
			uint64_t    begin      = 0;
			uint64_t    end        = 0;
		};
		thread_local LastShared last;
		const auto generation = m_mapped_generation.load(std::memory_order_relaxed);
		if (last.space == this && last.generation == generation && vaddr >= last.begin &&
		    vaddr + size <= last.end) {
			return false;
		}
		auto mapping = m_mapped.upper_bound(vaddr);
		if (mapping != m_mapped.begin()) {
			--mapping;
		}
		for (; mapping != m_mapped.end() && mapping->second.vaddr < vaddr + size; ++mapping) {
			const auto& region = mapping->second;
			if (region.vaddr + region.size <= vaddr) {
				continue;
			}
			if (region.type == MappingType::Private) {
				return true;
			}
			if (region.vaddr <= vaddr && vaddr + size <= region.vaddr + region.size) {
				last = {this, generation, region.vaddr, region.vaddr + region.size};
			}
		}
		return false;
	}

	// Copies the private mappings of a span and zero-fills the rest.
	void CopyPrivateUnlocked(uint64_t vaddr, uint8_t* destination, uint64_t size) const {
		const auto end    = vaddr + size;
		auto       cursor = vaddr;
		auto       it     = m_mapped.upper_bound(vaddr);
		if (it != m_mapped.begin()) {
			--it;
		}
		for (; it != m_mapped.end() && it->second.vaddr < end; ++it) {
			const auto start = std::max(cursor, it->second.vaddr);
			const auto stop  = std::min(end, it->second.vaddr + it->second.size);
			if (start < stop) {
				std::memset(destination + (cursor - vaddr), 0, start - cursor);
				std::memcpy(destination + (start - vaddr), reinterpret_cast<const void*>(start),
				            stop - start);
				cursor = stop;
			}
		}
		std::memset(destination + (cursor - vaddr), 0, end - cursor);
	}

	void AddMappedUnlocked(uint64_t vaddr, uint64_t size, MappingType type) {
		m_mapped_generation.fetch_add(1, std::memory_order_relaxed);
		m_mapped.emplace(vaddr, MappedRegion {vaddr, size, type});
	}

	void RemoveMappedUnlocked(uint64_t vaddr, uint64_t size) {
		const auto end     = vaddr + size;
		auto       current = vaddr;
		while (current < end) {
			const auto it           = FindMappedContainingUnlocked(current);
			const auto original     = it->second;
			const auto original_end = original.vaddr + original.size;
			const auto part_end     = std::min(end, original_end);
			m_mapped.erase(it);
			m_mapped_generation.fetch_add(1, std::memory_order_relaxed);
			if (original.vaddr < current) {
				AddMappedUnlocked(original.vaddr, current - original.vaddr, original.type);
			}
			if (part_end < original_end) {
				AddMappedUnlocked(part_end, original_end - part_end, original.type);
			}
			current = part_end;
		}
	}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	static DWORD GetProtect(VirtualMemory::Mode mode) {
		switch (mode) {
			case VirtualMemory::Mode::Read: return PAGE_READONLY;
			case VirtualMemory::Mode::Write:
			case VirtualMemory::Mode::ReadWrite: return PAGE_READWRITE;
			case VirtualMemory::Mode::Execute: return PAGE_EXECUTE;
			case VirtualMemory::Mode::ExecuteRead: return PAGE_EXECUTE_READ;
			case VirtualMemory::Mode::ExecuteWrite:
			case VirtualMemory::Mode::ExecuteReadWrite: return PAGE_EXECUTE_READWRITE;
			case VirtualMemory::Mode::NoAccess:
			default: return PAGE_NOACCESS;
		}
	}
#endif

	void ReserveGuestRegions() {
		const std::array<std::pair<uint64_t, uint64_t>, 3> regions {{
		    {HOST_SYSTEM_MANAGED_MIN, HOST_SYSTEM_RESERVED_MAX + 1u},
		    {HOST_USER_MIN, HOST_USER_MAX + 1u},
		    {kExtendedMemoryBase, kExtendedMemoryBase + kExtendedMemorySize},
		}};

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		constexpr uint64_t MinimumRegionSize = 0x1000000;
		const auto         process           = GetCurrentProcess();
		for (const auto& [guest_start, guest_end]: regions) {
			auto next_addr = guest_start;
			while (next_addr < guest_end) {
				MEMORY_BASIC_INFORMATION info {};
				if (VirtualQuery(reinterpret_cast<const void*>(next_addr), &info, sizeof(info)) == 0) {
					EXIT("failed to inspect guest address space at 0x%016" PRIx64 "\n", next_addr);
				}

				const auto region_base = reinterpret_cast<uint64_t>(info.BaseAddress);
				const auto region_end = std::min<uint64_t>(
				    guest_end, region_base + static_cast<uint64_t>(info.RegionSize));
				const auto reserve_start =
				    AlignUp(std::max(region_base, guest_start), m_granularity);
				const auto reserve_end = AlignDown(region_end, m_granularity);

				if (info.State == MEM_FREE && reserve_end > reserve_start &&
				    reserve_end - reserve_start > MinimumRegionSize) {
					auto* ptr = VirtualAlloc2(
					    process, reinterpret_cast<void*>(reserve_start), reserve_end - reserve_start,
					    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0);
					if (reinterpret_cast<uint64_t>(ptr) != reserve_start) {
						EXIT("failed to reserve guest address space at 0x%016" PRIx64
						     ", size 0x%016" PRIx64 "\n",
						     reserve_start, reserve_end - reserve_start);
					}
					AddFreeUnlocked(reserve_start, reserve_end - reserve_start);
					m_owned.emplace_back(reserve_start, reserve_end - reserve_start);
				}

				next_addr = region_end;
			}
		}
#else
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX && !defined(__APPLE__)
		std::vector<std::pair<uint64_t, uint64_t>> occupied;
		FILE*                                      maps = fopen("/proc/self/maps", "r");
		EXIT_IF(maps == nullptr);

		char line[512];
		while (fgets(line, sizeof(line), maps) != nullptr) {
			unsigned long long mapping_start = 0;
			unsigned long long mapping_end   = 0;
			if (sscanf(line, "%llx-%llx", &mapping_start, &mapping_end) == 2) {
				occupied.emplace_back(static_cast<uint64_t>(mapping_start),
				                      static_cast<uint64_t>(mapping_end));
			}
		}
		fclose(maps);

		auto reserve_range = [this](uint64_t start, uint64_t end) {
			start = AlignUp(start, PageSize());
			end   = AlignDown(end, PageSize());
			if (end <= start) {
				return;
			}

			const auto size = end - start;
			int        flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif

			void* ptr = mmap(reinterpret_cast<void*>(start), size, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, size);
				}
				return;
			}

			AddFreeUnlocked(start, size);
			// Free ranges merge across region boundaries, so owned ranges do as well.
			if (!m_owned.empty() && m_owned.back().first + m_owned.back().second == start) {
				m_owned.back().second += size;
			} else {
				m_owned.emplace_back(start, size);
			}
		};

		for (const auto& [region_start, region_end]: regions) {
			auto current = region_start;

			for (const auto& [mapping_start, mapping_end]: occupied) {
				if (mapping_end <= current) {
					continue;
				}
				if (mapping_start >= region_end) {
					break;
				}

				if (mapping_start > current) {
					reserve_range(current, std::min(mapping_start, region_end));
				}

				current = std::max(current, mapping_end);
				if (current >= region_end) {
					break;
				}
			}

			if (current < region_end) {
				reserve_range(current, region_end);
			}
		}
#else
		for (const auto& [start, end]: regions) {
			int flags = MAP_PRIVATE | MAP_ANON | MAP_NORESERVE;
#if defined(KYTY_LINKED_GUEST_ADDRESS_SPACE)
			flags |= MAP_FIXED;
#elif defined(MAP_FIXED_NOREPLACE)
			flags |= MAP_FIXED_NOREPLACE;
#endif
			void* ptr = mmap(reinterpret_cast<void*>(start), end - start, PROT_NONE, flags, -1, 0);
			if (ptr == MAP_FAILED || reinterpret_cast<uint64_t>(ptr) != start) {
				if (ptr != MAP_FAILED) {
					munmap(ptr, end - start);
				}
				EXIT("failed to reserve guest address space at 0x%016" PRIx64
				     ", size 0x%016" PRIx64 "\n",
				     start, end - start);
			}
			AddFreeUnlocked(start, end - start);
			m_owned.emplace_back(start, end - start);
		}
#endif
#endif
	}

	uint64_t FindFreeAlignedUnlocked(uint64_t search_start, uint64_t search_end, uint64_t size,
	                                 uint64_t alignment) const {
		// Include the predecessor because search_start can lie inside a free range.
		auto it = m_free.upper_bound(search_start);
		if (it != m_free.begin()) {
			--it;
		}
		for (; it != m_free.end() && it->first < search_end; ++it) {
#if defined(KYTY_VIRTUAL_MEMORY_ALLOCATION_TESTS)
			g_test_free_range_visits.fetch_add(1, std::memory_order_relaxed);
#endif
			const auto& [base, range_size] = *it;
			const auto candidate           = AlignUp(std::max(base, search_start), alignment);
			if (candidate < search_end && size <= search_end - candidate &&
			    GuestFreeRangeContains(base, range_size, candidate, size)) {
				return candidate;
			}
		}
		return 0;
	}

	std::map<uint64_t, uint64_t>::iterator FindContaining(uint64_t vaddr, uint64_t size) {
		auto next = m_free.upper_bound(vaddr);
		if (next == m_free.begin()) {
			return m_free.end();
		}
		auto it = std::prev(next);
		return GuestFreeRangeContains(it->first, it->second, vaddr, size) ? it : m_free.end();
	}

	bool SplitUnlocked(std::map<uint64_t, uint64_t>::iterator it, uint64_t vaddr, uint64_t size) {
		const auto base = it->first;
		const auto end  = it->first + it->second;
		if (base != vaddr) {
			const auto left_size = vaddr - base;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(base), left_size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           base, end - base, vaddr, size, static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			it->second = left_size;
			it         = m_free.emplace(vaddr, end - vaddr).first;
		}

		if (it->second != size) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (VirtualFree(reinterpret_cast<void*>(it->first), size,
			                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) == 0) {
				LOGF_COLOR(Log::Color::Red,
				           "\t guest placeholder split failed: base=0x%016" PRIx64
				           " size=0x%016" PRIx64 " requested=0x%016" PRIx64
				           " request_size=0x%016" PRIx64 " error=0x%08" PRIx32 "\n",
				           it->first, it->second, vaddr, size,
				           static_cast<uint32_t>(GetLastError()));
				return false;
			}
#endif
			const auto right_start = it->first + size;
			const auto right_size  = it->second - size;
			it->second             = size;
			m_free.emplace(right_start, right_size);
		}

		return true;
	}

	void AddFreeUnlocked(uint64_t vaddr, uint64_t size) {
		auto [it, inserted] = m_free.emplace(vaddr, size);
		if (!inserted) {
			it->second = size;
		}

		auto base = it->first;
		auto end  = it->first + it->second;

		auto first = it;
		while (first != m_free.begin()) {
			auto prev = std::prev(first);
			if (prev->first + prev->second != base) {
				break;
			}
			base  = prev->first;
			first = prev;
		}

		auto last = std::next(it);
		while (last != m_free.end() && end == last->first) {
			end  = last->first + last->second;
			last = std::next(last);
		}

		if (base != it->first || end != it->first + it->second) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
			if (CoalescePlaceholders(base, end - base)) {
				m_free.erase(first, last);
				m_free.emplace(base, end - base);
			}
#else
			m_free.erase(first, last);
			m_free.emplace(base, end - base);
#endif
		}
	}

	void RestorePlaceholder(uint64_t vaddr, uint64_t size) {
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
		// The address remains owned for the process lifetime; MAP_FIXED only replaces one of our
		// mappings and never acquires a new guest address.
		void* ptr = mmap(reinterpret_cast<void*>(vaddr), size, PROT_NONE,
		                 MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
		EXIT_IF(ptr == MAP_FAILED);
#else
		(void)vaddr;
		(void)size;
#endif
	}

	std::map<uint64_t, uint64_t>               m_free;
	std::map<uint64_t, MappedRegion>            m_mapped;
	// Advances on every change of m_mapped (under the exclusive lock; read under either).
	std::atomic<uint64_t>                       m_mapped_generation {1};
	std::vector<std::pair<uint64_t, uint64_t>> m_owned;
	uint64_t                                   m_granularity = 0x10000;
	GuestBackingStore*                         m_backing     = nullptr;
};
