#ifndef EMULATOR_INCLUDE_EMULATOR_EMULATOR_H_
#define EMULATOR_INCLUDE_EMULATOR_EMULATOR_H_

#include "common/additionalContent.h"
#include "common/emulatorConfig.h"
#include "common/stringUtils.h"

#include <filesystem>

namespace Emulator {

struct RunOptions {
	Config::ConfigOptions                config;
	std::filesystem::path                app0_dir;
	std::filesystem::path                elf;
	std::filesystem::path                game_patch;
	std::filesystem::path                addcont_dir;
	std::vector<Common::AddContentEntry> addcont;
	// Frame capture replay (graphics/frameCapture.h): no game is loaded.
	std::filesystem::path replay;
	std::filesystem::path replay_out;
	uint32_t              replay_repeat = 1;
	std::filesystem::path replay_shaders; // --replay-shaders: dump the shaders, replay nothing
};

void Run(const RunOptions& options);
void RunReplay(const RunOptions& options);
// --precompile-shaders, --restamp-shader-store: fills or re-stamps the title's shader cache
// without loading the game; the exit code.
int  RunPrecompile(const RunOptions& options);
int LoadExec(const char* path, const char* const argv[]);

} // namespace Emulator

#endif /* EMULATOR_INCLUDE_EMULATOR_EMULATOR_H_ */
