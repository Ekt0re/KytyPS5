#include "kernel/guestWait.h"

#include "common/timer.h"

// Kept apart from pthread.cpp so that the small kernel test targets, which compile single
// kernel files that mark their waits, link without the whole pthread implementation.
namespace Libs::LibKernel {

static thread_local GuestWaitState* g_guest_wait_state = nullptr;

void SetCurrentGuestWaitState(GuestWaitState* state) noexcept {
	g_guest_wait_state = state;
}

GuestWaitScope::GuestWaitScope(const char* call, uint64_t object, int64_t value) noexcept {
	auto* self = g_guest_wait_state;
	m_call     = self != nullptr ? self->call.load(std::memory_order_relaxed) : nullptr;
	m_object   = self != nullptr ? self->object.load(std::memory_order_relaxed) : 0;
	m_value    = self != nullptr ? self->value.load(std::memory_order_relaxed) : 0;
	m_since    = self != nullptr ? self->since.load(std::memory_order_relaxed) : 0;
	if (self == nullptr) return;
	self->object.store(object, std::memory_order_relaxed);
	self->value.store(value, std::memory_order_relaxed);
	self->since.store(Common::SteadyNs(), std::memory_order_relaxed);
	self->call.store(call, std::memory_order_release);
}

GuestWaitScope::~GuestWaitScope() {
	auto* self = g_guest_wait_state;
	if (self == nullptr) return;
	self->object.store(m_object, std::memory_order_relaxed);
	self->value.store(m_value, std::memory_order_relaxed);
	self->since.store(m_since, std::memory_order_relaxed);
	self->call.store(m_call, std::memory_order_release);
}

} // namespace Libs::LibKernel
