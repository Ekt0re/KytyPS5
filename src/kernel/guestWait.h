#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_GUEST_WAIT_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_GUEST_WAIT_H_

#include <atomic>
#include <cstdint>

namespace Libs::LibKernel {

// The blocking call a guest thread is in, read by the thread dump of the crash report. Lives in
// the thread's PthreadPrivate; call == nullptr means the thread is not waiting.
struct GuestWaitState {
	std::atomic<const char*> call {nullptr};
	std::atomic<uint64_t>    object {0};
	std::atomic<int64_t>     value {0};
	std::atomic<uint64_t>    since {0};
};

// The state GuestWaitScope writes on the calling host thread; nullptr (the default, and what
// tests without a guest thread have) turns the scopes into no-ops. The pthread code sets it
// wherever it sets the current guest thread.
void SetCurrentGuestWaitState(GuestWaitState* state) noexcept;

// The blocking call the current guest thread is in (call name, wait object, a call-specific
// value), for the thread dump of the crash report. Nested scopes restore the outer one.
class GuestWaitScope {
public:
	GuestWaitScope(const char* call, uint64_t object, int64_t value = 0) noexcept;
	~GuestWaitScope();
	GuestWaitScope(const GuestWaitScope&)            = delete;
	GuestWaitScope& operator=(const GuestWaitScope&) = delete;

private:
	const char* m_call;
	uint64_t    m_object;
	int64_t     m_value;
	uint64_t    m_since;
};

} // namespace Libs::LibKernel

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_GUEST_WAIT_H_ */
