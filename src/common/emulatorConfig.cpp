#include "common/emulatorConfig.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/perfSummary.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

namespace Config {

static std::unique_ptr<ConfigOptions> g_config;

// KYTY_SYNC_STATS=n prints the memory synchronisation counters every n seconds to stdout.
static uint32_t SyncStatsPeriod() {
	static const uint32_t period = [] {
		const char* env = std::getenv("KYTY_SYNC_STATS");
		return env == nullptr ? 0u : static_cast<uint32_t>(std::strtoul(env, nullptr, 10));
	}();
	return period;
}

TraceCallsFilter ParseTraceCallsFilter(std::string_view value) {
	TraceCallsFilter result;
	if (value.empty() || value == "0") {
		return result;
	}
	result.enabled = true;
	if (value == "all") {
		result.all = true;
		return result;
	}
	size_t pos = 0;
	while (pos < value.size()) {
		const size_t comma = value.find(',', pos);
		const auto token =
		    comma == std::string_view::npos ? value.substr(pos) : value.substr(pos, comma - pos);
		if (!token.empty()) {
			result.libraries.emplace_back(token);
		}
		if (comma == std::string_view::npos) {
			break;
		}
		pos = comma + 1;
	}
	if (result.libraries.empty()) {
		result.enabled = false;
	}
	return result;
}

const TraceCallsFilter& GetTraceCallsFilter() {
	static const TraceCallsFilter filter = [] {
		const char* env = std::getenv("KYTY_TRACE_CALLS");
		return ParseTraceCallsFilter(env == nullptr ? std::string_view {} : std::string_view {env});
	}();
	return filter;
}

bool TraceCallsEnabled() {
	return GetTraceCallsFilter().enabled;
}

static void StartSyncStatsReporter() {
	static bool started = false;
	const auto  period  = SyncStatsPeriod();
	if (started || period == 0) {
		return;
	}
	started = true;
	std::thread([period] {
		using Common::DebugCounters::Counter;
		using Common::DebugCounters::Get;
		constexpr auto                                  N = static_cast<size_t>(Counter::Count);
		std::array<uint64_t, N>                         last {};
		auto                                            last_time = std::chrono::steady_clock::now();
		for (uint64_t index = 0;; index++) {
			std::this_thread::sleep_for(std::chrono::seconds(period));
			const auto now  = std::chrono::steady_clock::now();
			const auto secs = std::chrono::duration<double>(now - last_time).count();
			last_time       = now;
			std::array<uint64_t, N> cur {};
			for (size_t i = 0; i < N; i++) {
				cur[i] = Get(static_cast<Counter>(i));
			}
			const auto d = [&](Counter c) {
				return static_cast<double>(cur[static_cast<size_t>(c)] - last[static_cast<size_t>(c)]) /
				       secs;
			};
			constexpr double MB = 1024.0 * 1024.0;
			std::printf("SyncStats t=%" PRIu64 " flips/s=%.1f subs/s=%.0f draws/s=%.0f | "
			            "faults r/s=%.0f w/s=%.0f all/s=%.0f fault_ms/s=%.1f (buf %.1f tex %.1f heap %.1f) | "
			            "readbacks/s=%.0f detached/s=%.0f zc_releases/s=%.0f zc_read_waits/s=%.0f readback_ms/s=%.1f bufdown_MB/s=%.1f texdown_MB/s=%.1f | "
			            "bufup_MB/s=%.1f reup_MB/s=%.1f unchanged_MB/s=%.1f stream_MB/s=%.1f "
			            "bda_MB/s=%.1f texup_MB/s=%.1f | protect/s=%.0f protect_MB/s=%.1f "
			            "protect_ms/s=%.1f | gpuwait_ms/s=%.0f gputhread_ms/s=%.0f\n",
			            index * period, d(Counter::FlipsRequested), d(Counter::QueueSubmits),
			            d(Counter::Draws), d(Counter::SyncReadFaults), d(Counter::SyncWriteFaults),
			            d(Counter::GpuFaults), d(Counter::SyncFaultNs) / 1e6, d(Counter::SyncFaultBufferNs) / 1e6,
			            d(Counter::SyncFaultTextureNs) / 1e6, d(Counter::SyncFaultHeapNs) / 1e6, d(Counter::SyncReadbacks), d(Counter::SyncDetachedReadbacks),
			            d(Counter::ZeroCopyReleases), d(Counter::ZeroCopyReadWaits), d(Counter::SyncReadbackNs) / 1e6, d(Counter::BufferDownloadBytes) / MB,
			            d(Counter::TextureDownloadBytes) / MB, d(Counter::BufferUploadBytes) / MB,
			            d(Counter::SyncReuploadBytes) / MB, d(Counter::SyncUnchangedUploadBytes) / MB,
			            d(Counter::SyncStreamBytes) / MB, d(Counter::SyncBdaUploadBytes) / MB,
			            d(Counter::TextureUploadBytes) / MB, d(Counter::SyncProtectCalls),
			            d(Counter::SyncProtectBytes) / MB, d(Counter::SyncProtectNs) / 1e6,
			            d(Counter::GpuWaitNs) / 1e6, d(Counter::GpuThreadBusyNs) / 1e6);
			std::fflush(stdout);
			last = cur;
		}
	}).detach();
}

void Initialize() {
	EXIT_IF(g_config != nullptr);

	g_config = std::make_unique<ConfigOptions>();
}

void Shutdown() {
	Common::DebugCounters::SetEnabled(false);
	g_config.reset();
}

void Load(const ConfigOptions& cfg) {
	EXIT_IF(g_config == nullptr);
	EXIT_IF(cfg.user_name.empty() || cfg.user_name.size() > MAX_USER_NAME_LENGTH);
	EXIT_IF(!IsConfiguredUserIdValid(cfg.user_id));
	EXIT_IF(cfg.controller_speaker_volume > 100 || cfg.controller_vibration_intensity > 100);
	EXIT_IF(cfg.keyboard_player > static_cast<uint32_t>(MAX_PLAYERS));

	*g_config = cfg;
	Common::DebugCounters::SetEnabled(cfg.perf_overlay_enabled || SyncStatsPeriod() != 0);
	StartSyncStatsReporter();
	Common::PerfSummary::Start();
}

uint32_t GetScreenWidth() {
	return g_config->screen_width;
}

uint32_t GetScreenHeight() {
	return g_config->screen_height;
}

const std::string& GetUserName() {
	return g_config->user_name;
}

int32_t GetUserId() {
	return g_config->user_id;
}

const std::string& GetAudioInputDevice() {
	return g_config->audio_input_device;
}

AudioOutput GetAudioOutput() {
	return g_config->audio_output;
}

bool Audio3dEnabled() {
	return g_config->audio_3d_enabled;
}

const std::optional<ControllerColor>& GetControllerColor() {
	return g_config->controller_color;
}

uint32_t GetControllerSpeakerVolume() {
	return g_config->controller_speaker_volume;
}

uint32_t GetControllerVibrationIntensity() {
	return g_config->controller_vibration_intensity;
}

uint32_t GetKeyboardPlayer() {
	return g_config->keyboard_player;
}

int32_t GetPlayerUserId(int player) {
	return g_config->user_id + player * PLAYER_USER_ID_STEP;
}

int GetPlayerFromUserId(int32_t user_id) {
	for (int player = 0; player < MAX_PLAYERS; player++) {
		if (user_id == GetPlayerUserId(player)) {
			return player;
		}
	}
	return -1;
}

bool IsGuestUserId(int32_t user_id) {
	return GetPlayerFromUserId(user_id) > 0;
}

PresentMode GetPresentMode() {
	return g_config->present_mode;
}

int32_t GetGpuIndex() {
	return g_config->gpu_index;
}

bool FullscreenEnabled() {
	return g_config->fullscreen_enabled;
}

bool HdrEnabled() {
	return g_config->hdr_enabled;
}

bool HideCursorEnabled() {
	return g_config->hide_cursor_enabled;
}

bool StretchEnabled() {
	return g_config->stretch_enabled;
}

bool VrEnabled() {
	return g_config->vr_enabled;
}

bool Ps5ProEnabled() {
	return g_config->pro_enabled;
}

bool Output1080pEnabled() {
	return g_config->output_1080p;
}

PssrUpscaler GetPssrUpscaler() {
	return g_config->pssr_upscaler;
}

const std::filesystem::path& GetDlssFolder() {
	return g_config->dlss_folder;
}

uint32_t GetDlssPreset() {
	return g_config->dlss_preset;
}

const std::filesystem::path& GetFsrFolder() {
	return g_config->fsr_folder;
}

bool AmdCpuEnabled() {
	return g_config->amd_cpu_enabled;
}

bool GuestTscEnabled() {
	return g_config->guest_tsc_enabled;
}

uint32_t GetVblankFrequency() {
	return std::clamp(g_config->vblank_frequency, 30u, 360u);
}

uint32_t GetConsoleLanguage() {
	return g_config->console_language;
}

bool VulkanValidationEnabled() {
	return g_config->vulkan_validation_enabled;
}

bool IsNetworkConnected() {
	return g_config->network_connected;
}

bool PsnEmulationEnabled() {
	return g_config->psn_emulation_enabled;
}

bool ShaderValidationEnabled() {
	return g_config->shader_validation_enabled;
}

bool ShaderCacheEnabled() {
	return g_config->shader_cache_enabled;
}

bool PrecompileShadersEnabled() {
	return g_config->precompile_shaders || g_config->restamp_shader_store;
}

LogDirection GetShaderLogDirection() {
	return g_config->shader_log_direction;
}

std::filesystem::path GetShaderLogFolder() {
	return g_config->shader_log_folder;
}

bool CommandBufferDumpEnabled() {
	return g_config->command_buffer_dump_enabled;
}

std::filesystem::path GetCommandBufferDumpFolder() {
	return g_config->command_buffer_dump_folder;
}

bool GraphicsDebugDumpEnabled() {
	return g_config->graphics_debug_dump_enabled;
}

LogDirection GetPrintfDirection() {
	return g_config->printf_direction;
}

std::filesystem::path GetPrintfOutputFile() {
	return g_config->printf_output_file;
}

bool ProfilerEnabled() {
	return g_config->profiler_enabled;
}

bool SpirvDebugPrintfEnabled() {
	return g_config->spirv_debug_printf_enabled;
}

bool GpuAssistedValidationEnabled() {
	return g_config->gpu_assisted_validation_enabled && g_config->vulkan_validation_enabled;
}

bool RenderDocEnabled() {
	return g_config->renderdoc_enabled;
}

bool ReadbackLinearImagesEnabled() {
	return g_config->readback_linear_images;
}

bool ReadbackTiledImagesEnabled() {
	return g_config->readback_tiled_images;
}

bool ReadbackStorageImagesEnabled() {
	return g_config->readback_storage_images;
}

bool SyncRawImageBuffersEnabled() {
	return g_config->sync_raw_image_buffers;
}

uint32_t GetFlushDraws() {
	return g_config->flush_draws;
}

bool TessellationEnabled() {
	return g_config->tessellation_enabled;
}

bool OcclusionQueriesEnabled() {
	return g_config->occlusion_queries_enabled;
}

bool RelaxedShaderFloatEnabled() {
	return g_config->relaxed_shader_float;
}

bool NativeRtEnabled() {
	return g_config->native_rt;
}

bool LodStatsEnabled() {
	return g_config->lod_stats_enabled;
}

bool TrophyEnabled() {
	return g_config->trophy_enabled;
}

uint32_t GetTrophyDurationSeconds() {
	return std::clamp(g_config->trophy_duration_seconds, 5u, 10u);
}

bool PlayGoHackEnabled() {
	return g_config->playgo_hack_enabled;
}

bool SkipNoticeScreen() {
	return g_config->skip_notice_screen;
}

const std::string& GetLaunchActivity() {
	return g_config->launch_activity;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
bool RedZoneProtectionEnabled() {
	return g_config->red_zone_protection_enabled;
}
#endif

const Keymap& GetKeymap() {
	return g_config->keymap;
}

const std::vector<std::string>& GetGuestArguments() {
	return g_config->guest_arguments;
}

} // namespace Config
