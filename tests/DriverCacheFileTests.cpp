#include "graphics/host_gpu/renderer/pipeline/driverCacheFile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <utility>
#include <vector>

using Libs::Graphics::DriverCacheFile;

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DriverCacheFileTests: failed: %s\n", message);
		std::abort();
	}
}

void TestRoundTripAndMove() {
	const auto dir = std::filesystem::temp_directory_path() / "kyty_driver_cache_map_test";
	std::filesystem::create_directories(dir);
	const auto path = dir / "TEST.bin";
	const std::vector<uint8_t> expected {0x53, 0x49, 0x47, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	                                      1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	{
		std::FILE* file = std::fopen(path.string().c_str(), "wb");
		Check(file != nullptr, "test cache file was created");
		std::fwrite(expected.data(), 1, expected.size(), file);
		std::fclose(file);
	}

	DriverCacheFile mapped = DriverCacheFile::Map(path);
	Check(mapped.data != nullptr, "the cache file was mapped or read");
	Check(mapped.size == expected.size(), "the whole file is visible");
	Check(std::memcmp(mapped.data, expected.data(), expected.size()) == 0,
	      "the mapped bytes are the file's bytes");
	const auto* base = mapped.data;

	DriverCacheFile moved = std::move(mapped);
	Check(mapped.data == nullptr && mapped.size == 0, "the moved-from view is empty");
	Check(moved.data == base && moved.size == expected.size(), "the move keeps the same view");
	Check(std::memcmp(moved.data, expected.data(), expected.size()) == 0,
	      "bytes survive the move");
	moved.Release();
	Check(moved.data == nullptr && moved.size == 0, "release drops the view");

	DriverCacheFile missing = DriverCacheFile::Map(dir / "MISSING.bin");
	Check(missing.data == nullptr && missing.size == 0, "a missing file is an empty view");

	const auto empty_path = dir / "EMPTY.bin";
	{
		std::FILE* file = std::fopen(empty_path.string().c_str(), "wb");
		Check(file != nullptr, "empty test file was created");
		std::fclose(file);
	}
	DriverCacheFile empty = DriverCacheFile::Map(empty_path);
	Check(empty.data == nullptr && empty.size == 0, "an empty file is an empty view");

	std::error_code error;
	std::filesystem::remove_all(dir, error);
}

} // namespace

int main() {
	TestRoundTripAndMove();
	std::puts("DriverCacheFileTests: ok");
	return 0;
}
