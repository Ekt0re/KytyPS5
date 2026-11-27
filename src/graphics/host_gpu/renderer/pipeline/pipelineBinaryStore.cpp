#include "graphics/host_gpu/renderer/pipeline/pipelineBinaryStore.h"

#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fmt/format.h>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// File layout, all little-endian:
//   header:  "KYTYPBN1" (8 bytes), u32 global key size, 32 bytes global key
//   records: u32 'PBR1', u32 pipeline key size, 32 bytes pipeline key, u32 binary count,
//            per binary {u32 key size, 32 bytes key, u64 data size, data}, u64 XXH3 of the record
//            from its pipeline key size up to the last binary's data.
// The last byte tells whether the driver's internal cache was off when the binaries were captured
// ('2'; complete binaries) or on ('1'; NVIDIA's then depend on that cache): the two never mix.
constexpr char     FileMagicCached[8]   = {'K', 'Y', 'T', 'Y', 'P', 'B', 'N', '1'};
constexpr char     FileMagicComplete[8] = {'K', 'Y', 'T', 'Y', 'P', 'B', 'N', '2'};
constexpr uint32_t RecordMagic   = 0x31524250u; // "PBR1"
constexpr size_t   KeyBytes      = VK_MAX_PIPELINE_BINARY_KEY_SIZE_KHR;
constexpr size_t   HeaderBytes   = sizeof(FileMagicCached) + sizeof(uint32_t) + KeyBytes;
// A record past this is damaged (a pipeline's binaries are far smaller); stops a bad scan early.
constexpr uint64_t MaxRecordData = uint64_t {1} << 32u;

struct Key {
	uint32_t                      size = 0;
	std::array<uint8_t, KeyBytes> bytes {};

	bool operator==(const Key&) const = default;
};

struct KeyHash {
	size_t operator()(const Key& key) const {
		return static_cast<size_t>(XXH3_64bits(key.bytes.data(), key.size));
	}
};

Key ToKey(const VkPipelineBinaryKeyKHR& key) {
	Key result;
	result.size = std::min<uint32_t>(key.keySize, KeyBytes);
	std::memcpy(result.bytes.data(), key.key, result.size);
	return result;
}

VkPipelineBinaryKeyKHR ToVk(const Key& key) {
	VkPipelineBinaryKeyKHR result {};
	result.sType   = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
	result.keySize = key.size;
	std::memcpy(result.key, key.bytes.data(), key.size);
	return result;
}

bool SeekFile(std::FILE* file, uint64_t offset) {
#if defined(_WIN32)
	return _fseeki64(file, static_cast<__int64>(offset), SEEK_SET) == 0;
#else
	return fseeko(file, static_cast<off_t>(offset), SEEK_SET) == 0;
#endif
}

uint64_t TellFile(std::FILE* file) {
#if defined(_WIN32)
	return static_cast<uint64_t>(_ftelli64(file));
#else
	return static_cast<uint64_t>(ftello(file));
#endif
}

bool ReadExact(std::FILE* file, void* data, size_t size) {
	return std::fread(data, 1, size, file) == size;
}

bool SwitchEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PIPELINE_BINARIES");
		return value == nullptr || value[0] != '0';
	}();
	return enabled;
}

} // namespace

struct PipelineBinaryStore::Impl {
	struct Binary {
		Key                  key;
		std::vector<uint8_t> data;
	};
	struct Record {
		uint64_t offset = 0; // Of the pipeline key size: the hashed part starts there.
		uint64_t size   = 0; // To the end of the last binary's data (the hash follows).
	};

	std::mutex                               mutex; // The file and the index.
	std::FILE*                               file = nullptr;
	std::filesystem::path                    path;
	std::unordered_map<Key, Record, KeyHash> index;
	uint64_t                                 file_size = 0;
	bool                                     enabled   = false;

	std::atomic<uint64_t> hits {0};
	std::atomic<uint64_t> stored {0};
	std::atomic<uint64_t> fallbacks {0};
	std::atomic<uint64_t> stored_bytes {0};

	// Indexes the records after the header without reading their data (a store may hold many GB):
	// each record's sizes must fit in the file, its hash is checked when it is loaded. Returns where
	// the last whole record ends, so a record a crash cut short is written over.
	uint64_t Scan() {
		uint64_t length = 0;
		if (SeekFile(file, 0) && std::fseek(file, 0, SEEK_END) == 0) length = TellFile(file);
		uint64_t at = HeaderBytes;
		while (at + 4 + 4 + KeyBytes + 4 + 8 <= length && SeekFile(file, at)) {
			uint32_t magic = 0, key_size = 0, count = 0;
			Key      key;
			if (!ReadExact(file, &magic, 4) || magic != RecordMagic || !ReadExact(file, &key_size, 4) ||
			    key_size > KeyBytes || !ReadExact(file, key.bytes.data(), KeyBytes) ||
			    !ReadExact(file, &count, 4) || count == 0 || count > 64) {
				break;
			}
			key.size = key_size;
			bool     ok  = true;
			uint64_t end = at + 4 + 4 + KeyBytes + 4;
			for (uint32_t i = 0; ok && i < count; i++) {
				uint32_t binary_key_size = 0;
				uint64_t data_size       = 0;
				ok = SeekFile(file, end) && ReadExact(file, &binary_key_size, 4) &&
				     binary_key_size <= KeyBytes && SeekFile(file, end + 4 + KeyBytes) &&
				     ReadExact(file, &data_size, 8) && data_size <= MaxRecordData;
				end += 4 + KeyBytes + 8 + data_size;
				ok = ok && end + 8 <= length;
			}
			if (!ok) break;
			// The hashed part runs from the key size to the end of the last binary's data.
			index[key] = {.offset = at + 4, .size = end - (at + 4)};
			at         = end + 8;
		}
		return at;
	}

	// Drops a record whose binaries could not be used, so the pipeline is captured and appended
	// again (the later record wins on the next scan).
	void Forget(const Key& key) {
		std::lock_guard lock(mutex);
		index.erase(key);
	}

	// The binaries of `key`, or none (also when the record does not match its hash).
	bool Load(const Key& key, std::vector<Binary>& binaries) {
		std::vector<uint8_t> record;
		uint64_t             hash = 0;
		{
			std::lock_guard lock(mutex);
			const auto      found = index.find(key);
			if (found == index.end() || file == nullptr || !SeekFile(file, found->second.offset)) {
				return false;
			}
			record.resize(static_cast<size_t>(found->second.size));
			if (!ReadExact(file, record.data(), record.size()) || !ReadExact(file, &hash, 8)) {
				return false;
			}
		}
		if (XXH3_64bits(record.data(), record.size()) != hash) {
			Forget(key);
			return false;
		}
		size_t     at   = 4 + KeyBytes; // Past the pipeline key.
		const auto take = [&](void* data, size_t size) {
			if (at + size > record.size()) return false;
			std::memcpy(data, record.data() + at, size);
			at += size;
			return true;
		};
		uint32_t count = 0;
		if (!take(&count, 4)) return false;
		binaries.resize(count);
		for (auto& binary: binaries) {
			uint32_t binary_key_size = 0;
			uint64_t data_size       = 0;
			if (!take(&binary_key_size, 4) || binary_key_size > KeyBytes ||
			    !take(binary.key.bytes.data(), KeyBytes) || !take(&data_size, 8) ||
			    data_size > record.size() - at) {
				return false;
			}
			binary.key.size = binary_key_size;
			binary.data.assign(record.begin() + static_cast<ptrdiff_t>(at),
			                   record.begin() + static_cast<ptrdiff_t>(at + data_size));
			at += static_cast<size_t>(data_size);
		}
		return true;
	}

	void Append(const Key& key, const std::vector<Binary>& binaries) {
		std::vector<uint8_t> record;
		const auto           put = [&](const void* data, size_t size) {
            const auto* bytes = static_cast<const uint8_t*>(data);
            record.insert(record.end(), bytes, bytes + size);
		};
		const uint32_t count = static_cast<uint32_t>(binaries.size());
		put(&key.size, 4);
		put(key.bytes.data(), KeyBytes);
		put(&count, 4);
		for (const auto& binary: binaries) {
			const uint64_t data_size = binary.data.size();
			put(&binary.key.size, 4);
			put(binary.key.bytes.data(), KeyBytes);
			put(&data_size, 8);
			put(binary.data.data(), binary.data.size());
		}
		const uint64_t hash = XXH3_64bits(record.data(), record.size());
		std::lock_guard lock(mutex);
		if (file == nullptr || index.contains(key) || !SeekFile(file, file_size)) {
			return;
		}
		const bool written = std::fwrite(&RecordMagic, 4, 1, file) == 1 &&
		                     std::fwrite(record.data(), 1, record.size(), file) == record.size() &&
		                     std::fwrite(&hash, 8, 1, file) == 1 && std::fflush(file) == 0;
		if (!written) {
			// A partial record is cut off by the next scan; stop writing to a failing file.
			Log::WriteToConsoleAndLog(fmt::format("Pipeline binaries: writing {} failed; the store "
			                                      "is read-only for this run\n",
			                                      path.string()));
			std::fclose(file);
			file    = nullptr;
			enabled = false;
			return;
		}
		// Offsets of the binary count and the record's end (before its hash).
		index[key] = {.offset = file_size + 4, .size = record.size()};
		file_size += 4 + record.size() + 8;
		stored_bytes += record.size();
	}
};

PipelineBinaryStore& PipelineBinaryStore::Get() {
	static PipelineBinaryStore store;
	return store;
}

PipelineBinaryStore::PipelineBinaryStore(): m_impl(new Impl) {}

PipelineBinaryStore::~PipelineBinaryStore() {
	Close();
	delete m_impl;
}

bool PipelineBinaryStore::Enabled() const {
	return m_impl->enabled;
}

void PipelineBinaryStore::Open(GraphicContext& graphics, const std::filesystem::path& path) {
	auto&           impl = *m_impl;
	std::lock_guard lock(impl.mutex);
	if (impl.file != nullptr || !SwitchEnabled() || !graphics.pipeline_binary_enabled) {
		if (!graphics.pipeline_binary_enabled && SwitchEnabled()) {
			Log::WriteToConsoleAndLog("Pipeline binaries: unavailable (no VK_KHR_pipeline_binary)\n");
		}
		return;
	}
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	if (d.vkGetPipelineKeyKHR == nullptr || d.vkCreatePipelineBinariesKHR == nullptr ||
	    d.vkGetPipelineBinaryDataKHR == nullptr || d.vkReleaseCapturedPipelineDataKHR == nullptr ||
	    d.vkDestroyPipelineBinaryKHR == nullptr) {
		Log::WriteToConsoleAndLog("Pipeline binaries: unavailable (functions not loaded)\n");
		return;
	}
	VkPipelineBinaryKeyKHR global {};
	global.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
	if (d.vkGetPipelineKeyKHR(graphics.device, nullptr, &global) != VK_SUCCESS) {
		Log::WriteToConsoleAndLog("Pipeline binaries: unavailable (no global key)\n");
		return;
	}
	const Key global_key = ToKey(global);
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	impl.path = path;
	const char* file_magic =
	    graphics.pipeline_binary_internal_cache_disabled ? FileMagicComplete : FileMagicCached;
	// An existing store must carry this driver's global key and this cache mode; other binaries
	// are unusable here, so that file is set aside (never overwritten) and a new one started.
	bool reuse = false;
	if (std::FILE* existing = std::fopen(path.string().c_str(), "rb"); existing != nullptr) {
		char     magic[8] {};
		uint32_t size = 0;
		Key      stored;
		reuse = ReadExact(existing, magic, 8) && std::memcmp(magic, file_magic, 8) == 0 &&
		        ReadExact(existing, &size, 4) && size <= KeyBytes &&
		        ReadExact(existing, stored.bytes.data(), KeyBytes);
		stored.size = size;
		std::fclose(existing);
		if (!reuse || !(stored == global_key)) {
			auto old = path;
			old += ".old";
			std::filesystem::remove(old, error);
			std::filesystem::rename(path, old, error);
			Log::WriteToConsoleAndLog(fmt::format(
			    "Pipeline binaries: {} was made by another driver or cache mode; moved to {}\n",
			    path.string(), old.string()));
			reuse = false;
		}
	}
	impl.file = std::fopen(path.string().c_str(), reuse ? "r+b" : "w+b");
	if (impl.file == nullptr) {
		Log::WriteToConsoleAndLog(fmt::format("Pipeline binaries: cannot open {}\n", path.string()));
		return;
	}
	const auto start = std::chrono::steady_clock::now();
	if (reuse) {
		impl.file_size = impl.Scan();
	} else {
		const uint32_t size = global_key.size;
		const bool written  = std::fwrite(file_magic, 8, 1, impl.file) == 1 &&
		                     std::fwrite(&size, 4, 1, impl.file) == 1 &&
		                     std::fwrite(global_key.bytes.data(), KeyBytes, 1, impl.file) == 1 &&
		                     std::fflush(impl.file) == 0;
		if (!written) {
			std::fclose(impl.file);
			impl.file = nullptr;
			return;
		}
		impl.file_size = HeaderBytes;
	}
	impl.enabled = true;
	Log::WriteToConsoleAndLog(fmt::format(
	    "Pipeline binaries: {} holds {} pipelines ({:.1f} MB), indexed in {} ms\n", path.string(),
	    impl.index.size(), static_cast<double>(impl.file_size) / 1'000'000.0,
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start)
	        .count()));
}

void PipelineBinaryStore::Close() {
	auto&           impl = *m_impl;
	std::lock_guard lock(impl.mutex);
	if (impl.file != nullptr) {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Pipeline binaries: this run created {} pipelines from the store, stored {} ({:.1f} MB), "
		    "fell back {} times; {} pipelines ({:.1f} MB) in {}\n",
		    impl.hits.load(), impl.stored.load(),
		    static_cast<double>(impl.stored_bytes.load()) / 1'000'000.0, impl.fallbacks.load(),
		    impl.index.size(), static_cast<double>(impl.file_size) / 1'000'000.0, impl.path.string()));
		std::fclose(impl.file);
		impl.file = nullptr;
	}
	impl.enabled = false;
}

namespace {

thread_local const char* g_last_source = "";

// Store counters every 20 s while pipelines are created.
void PrintStats(PipelineBinaryStore::Impl& impl) {
	static std::atomic<int64_t> last {0};
	const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
	auto       seen = last.load(std::memory_order_relaxed);
	if (now - seen < 20'000'000'000 || !last.compare_exchange_strong(seen, now)) return;
	std::printf("Pipeline binaries: %llu created from the store, %llu stored (%.1f MB), %llu fell "
	            "back\n",
	            static_cast<unsigned long long>(impl.hits.load()),
	            static_cast<unsigned long long>(impl.stored.load()),
	            static_cast<double>(impl.stored_bytes.load()) / 1'000'000.0,
	            static_cast<unsigned long long>(impl.fallbacks.load()));
}

// Creates one pipeline through the store. `create(next, flags2)` creates it with `next` prepended to
// the create info's pNext chain; `info_next` is the create info itself (for vkGetPipelineKeyKHR).
template <typename Create>
vk::Result CreateThroughStore(PipelineBinaryStore::Impl& impl, vk::Device device,
                              void* create_info, uint64_t legacy_flags, Create&& create,
                              vk::Pipeline* pipeline) {
	const auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
	VkPipelineCreateInfoKHR key_info {};
	key_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_INFO_KHR;
	key_info.pNext = create_info;
	VkPipelineBinaryKeyKHR vk_key {};
	vk_key.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
	PrintStats(impl);
	if (d.vkGetPipelineKeyKHR(device, &key_info, &vk_key) != VK_SUCCESS) {
		g_last_source = "no key";
		impl.fallbacks++;
		return create(nullptr);
	}
	const Key key = ToKey(vk_key);

	// Stored: create the binaries from their data and the pipeline from those.
	std::vector<PipelineBinaryStore::Impl::Binary> stored;
	if (impl.Load(key, stored) && !stored.empty()) {
		std::vector<VkPipelineBinaryKeyKHR>  keys;
		std::vector<VkPipelineBinaryDataKHR> datas;
		for (auto& binary: stored) {
			keys.push_back(ToVk(binary.key));
			datas.push_back({.dataSize = binary.data.size(), .pData = binary.data.data()});
		}
		const VkPipelineBinaryKeysAndDataKHR keys_and_data {
		    .binaryCount         = static_cast<uint32_t>(stored.size()),
		    .pPipelineBinaryKeys = keys.data(),
		    .pPipelineBinaryData = datas.data()};
		VkPipelineBinaryCreateInfoKHR binary_create {};
		binary_create.sType            = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR;
		binary_create.pKeysAndDataInfo = &keys_and_data;
		std::vector<VkPipelineBinaryKHR> handles(stored.size());
		VkPipelineBinaryHandlesInfoKHR   handles_info {};
		handles_info.sType               = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR;
		handles_info.pipelineBinaryCount = static_cast<uint32_t>(handles.size());
		handles_info.pPipelineBinaries   = handles.data();
		const auto binaries_result =
		    d.vkCreatePipelineBinariesKHR(device, &binary_create, nullptr, &handles_info);
		auto pipeline_result = vk::Result::eErrorUnknown;
		if (binaries_result == VK_SUCCESS) {
			VkPipelineBinaryInfoKHR binary_info {};
			binary_info.sType             = VK_STRUCTURE_TYPE_PIPELINE_BINARY_INFO_KHR;
			binary_info.binaryCount       = handles_info.pipelineBinaryCount;
			binary_info.pPipelineBinaries = handles.data();
			pipeline_result               = create(&binary_info);
			for (uint32_t i = 0; i < handles_info.pipelineBinaryCount; i++) {
				d.vkDestroyPipelineBinaryKHR(device, handles[i], nullptr);
			}
			if (pipeline_result == vk::Result::eSuccess) {
				g_last_source = "from stored binaries";
				impl.hits++;
				return pipeline_result;
			}
		}
		thread_local char reason[160];
		std::snprintf(reason, sizeof(reason),
		              "stored binaries rejected: %u binaries, create binaries %s, create pipeline %s",
		              static_cast<unsigned>(stored.size()),
		              vk::to_string(static_cast<vk::Result>(binaries_result)).c_str(),
		              binaries_result == VK_SUCCESS ? vk::to_string(pipeline_result).c_str() : "-");
		g_last_source = reason;
		impl.fallbacks++;
		// Captured again below; the rejected record is replaced on the next scan.
		impl.Forget(key);
	}

	// Not stored: create it with its data captured, store the binaries, release the data.
	VkPipelineCreateFlags2CreateInfo flags2 {};
	flags2.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
	flags2.flags = legacy_flags | VK_PIPELINE_CREATE_2_CAPTURE_DATA_BIT_KHR;
	const auto result = create(&flags2);
	g_last_source = "compiled, binaries captured";
	if (result != vk::Result::eSuccess) {
		g_last_source = "capture failed";
		impl.fallbacks++;
		return create(nullptr);
	}
	VkPipelineBinaryCreateInfoKHR binary_create {};
	binary_create.sType    = VK_STRUCTURE_TYPE_PIPELINE_BINARY_CREATE_INFO_KHR;
	binary_create.pipeline = static_cast<VkPipeline>(*pipeline);
	VkPipelineBinaryHandlesInfoKHR handles_info {};
	handles_info.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_HANDLES_INFO_KHR;
	std::vector<PipelineBinaryStore::Impl::Binary> captured;
	if (d.vkCreatePipelineBinariesKHR(device, &binary_create, nullptr, &handles_info) ==
	        VK_SUCCESS &&
	    handles_info.pipelineBinaryCount != 0) {
		std::vector<VkPipelineBinaryKHR> handles(handles_info.pipelineBinaryCount);
		handles_info.pPipelineBinaries = handles.data();
		if (d.vkCreatePipelineBinariesKHR(device, &binary_create, nullptr, &handles_info) ==
		    VK_SUCCESS) {
			bool ok = true;
			for (uint32_t i = 0; i < handles_info.pipelineBinaryCount; i++) {
				VkPipelineBinaryDataInfoKHR data_info {};
				data_info.sType          = VK_STRUCTURE_TYPE_PIPELINE_BINARY_DATA_INFO_KHR;
				data_info.pipelineBinary = handles[i];
				VkPipelineBinaryKeyKHR binary_key {};
				binary_key.sType = VK_STRUCTURE_TYPE_PIPELINE_BINARY_KEY_KHR;
				size_t size      = 0;
				ok = ok && d.vkGetPipelineBinaryDataKHR(device, &data_info, &binary_key, &size,
				                                        nullptr) == VK_SUCCESS;
				PipelineBinaryStore::Impl::Binary binary;
				binary.data.resize(size);
				ok = ok && d.vkGetPipelineBinaryDataKHR(device, &data_info, &binary_key, &size,
				                                        binary.data.data()) == VK_SUCCESS;
				binary.key = ToKey(binary_key);
				if (ok) captured.push_back(std::move(binary));
				d.vkDestroyPipelineBinaryKHR(device, handles[i], nullptr);
			}
			if (!ok) captured.clear();
		}
	}
	VkReleaseCapturedPipelineDataInfoKHR release {};
	release.sType    = VK_STRUCTURE_TYPE_RELEASE_CAPTURED_PIPELINE_DATA_INFO_KHR;
	release.pipeline = static_cast<VkPipeline>(*pipeline);
	d.vkReleaseCapturedPipelineDataKHR(device, &release, nullptr);
	if (!captured.empty()) {
		impl.Append(key, captured);
		impl.stored++;
	}
	return result;
}

} // namespace

const char* PipelineBinaryStore::LastSource() {
	return g_last_source;
}

vk::Result PipelineBinaryStore::CreateCompute(GraphicContext& graphics, vk::PipelineCache cache,
                                              const vk::ComputePipelineCreateInfo& info,
                                              vk::Pipeline* pipeline) {
	if (!m_impl->enabled) {
		return graphics.device.createComputePipelines(cache, 1, &info, nullptr, pipeline);
	}
	VkComputePipelineCreateInfo c_info = info;
	const auto create                  = [&](void* next) {
        VkComputePipelineCreateInfo chained = c_info;
        if (next != nullptr) {
            static_cast<VkBaseOutStructure*>(next)->pNext =
                const_cast<VkBaseOutStructure*>(static_cast<const VkBaseOutStructure*>(c_info.pNext));
            chained.pNext = next;
        }
        VkPipeline handle = VK_NULL_HANDLE;
        // VUID-vkCreateComputePipelines-pNext-09616/09617: no cache with binaries or capture.
        const auto result = static_cast<vk::Result>(VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateComputePipelines(
            graphics.device, next != nullptr ? VK_NULL_HANDLE : static_cast<VkPipelineCache>(cache), 1,
            &chained, nullptr, &handle));
        *pipeline = handle;
        return result;
	};
	return CreateThroughStore(*m_impl, graphics.device, &c_info,
	                          static_cast<uint64_t>(c_info.flags), create, pipeline);
}

vk::Result PipelineBinaryStore::CreateGraphics(GraphicContext& graphics, vk::PipelineCache cache,
                                               const vk::GraphicsPipelineCreateInfo& info,
                                               vk::Pipeline* pipeline) {
	if (!m_impl->enabled) {
		return graphics.device.createGraphicsPipelines(cache, 1, &info, nullptr, pipeline);
	}
	VkGraphicsPipelineCreateInfo c_info = info;
	const auto create                   = [&](void* next) {
        VkGraphicsPipelineCreateInfo chained = c_info;
        if (next != nullptr) {
            static_cast<VkBaseOutStructure*>(next)->pNext =
                const_cast<VkBaseOutStructure*>(static_cast<const VkBaseOutStructure*>(c_info.pNext));
            chained.pNext = next;
        }
        VkPipeline handle = VK_NULL_HANDLE;
        // VUID-vkCreateGraphicsPipelines-pNext-09616/09617: no cache with binaries or capture.
        const auto result = static_cast<vk::Result>(VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateGraphicsPipelines(
            graphics.device, next != nullptr ? VK_NULL_HANDLE : static_cast<VkPipelineCache>(cache), 1,
            &chained, nullptr, &handle));
        *pipeline = handle;
        return result;
	};
	return CreateThroughStore(*m_impl, graphics.device, &c_info,
	                          static_cast<uint64_t>(c_info.flags), create, pipeline);
}

} // namespace Libs::Graphics
