#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERFIX_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERFIX_H_

#include "common/runtimeToggles.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

// A rendering-correctness change that can be turned off for an A/B: the environment variable
// `env` (0 or 1) sets it for the run, `fallback` when it is unset, and with KYTY_TOGGLE_FILE a
// `toggle`=0|1 line in that file overrides it while the game runs (both arms of an A/B in one
// run). Hit() counts the cases the change decides differently, whatever its state, and prints
// the first few and every power of two, so a run shows whether the game reaches the change.
class RenderFix {
public:
	constexpr RenderFix(const char* name, const char* env, const char* toggle, bool fallback)
	    : m_name(name), m_env(env), m_toggle(toggle), m_fallback(fallback) {}

	[[nodiscard]] bool Enabled() const {
		if (Common::RuntimeToggles::Enabled()) {
			if (const int value = Common::RuntimeToggles::Find(m_toggle); value >= 0) {
				return value == 1;
			}
		}
		return Default();
	}

	[[nodiscard]] bool Default() const {
		int value = m_default.load(std::memory_order_relaxed);
		if (value < 0) {
			const char* text = std::getenv(m_env);
			value = text == nullptr || text[0] == '\0' ? (m_fallback ? 1 : 0) : (text[0] != '0');
			m_default.store(value, std::memory_order_relaxed);
		}
		return value == 1;
	}

	// detail: printf format and arguments describing this case (a shader hash, an address).
	template <typename... Args>
	void Hit(const char* format, Args... args) const {
		const uint64_t count = m_hits.fetch_add(1, std::memory_order_relaxed) + 1;
		if (count > 8 && (count & (count - 1)) != 0) {
			return;
		}
		char detail[256];
		std::snprintf(detail, sizeof(detail), format, args...);
		std::printf("Render fix %s (%s): case %llu, %s: %s\n", m_name, m_env,
		            static_cast<unsigned long long>(count), Enabled() ? "applied" : "off", detail);
	}

	void Hit() const { Hit("%s", "-"); }

private:
	const char*                  m_name;
	const char*                  m_env;
	const char*                  m_toggle;
	bool                         m_fallback;
	mutable std::atomic<int>     m_default {-1};
	mutable std::atomic<uint64_t> m_hits {0};
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERFIX_H_ */
