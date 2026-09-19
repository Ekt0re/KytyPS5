#include "common/asyncWriter.h"

#include "common/stringUtils.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fmt/format.h>
#include <fstream>
#include <mutex>
#include <thread>

namespace Common {

namespace {

struct WriterState {
	std::mutex                    mutex;
	std::condition_variable       cv_work;
	std::condition_variable       cv_flush;
	std::deque<AsyncWriter::Task> queue;
	std::thread                   worker_thread;
	std::atomic_bool              running {false};
	std::atomic_bool              is_working {false};
	std::atomic_bool              in_emergency {false};
	std::atomic_size_t            dropped_count {0};
	std::atomic_size_t            queue_bytes {0};
	AsyncWriter::QueueConfig      queue_config;
	AsyncWriter::RetryPolicy      retry_policy;
};

static WriterState*             g_writer = nullptr;
static std::mutex               g_init_mutex;
static AsyncWriter::QueueConfig s_queue_config;
static AsyncWriter::RetryPolicy s_retry_policy;
static bool                     s_config_set = false;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#include <share.h>
#endif

static FILE* OpenFile(const std::filesystem::path& path, const char* mode) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	wchar_t wmode[8] = {};
	for (size_t i = 0; mode[i] != '\0' && i < 7; ++i) {
		wmode[i] = static_cast<wchar_t>(mode[i]);
	}
	return _wfsopen(path.c_str(), wmode, _SH_DENYNO);
#else
	return std::fopen(path.c_str(), mode);
#endif
}

static void ProcessTask(const AsyncWriter::Task& task) noexcept {
	try {
		switch (task.type) {
			case AsyncWriter::TaskType::FileWrite: {
				if (!task.path.empty()) {
					std::error_code ec;
					const auto      parent = task.path.parent_path();
					if (!parent.empty()) {
						std::filesystem::create_directories(parent, ec);
					}

					int retry_count = 0;
					int max_retries = g_writer ? g_writer->retry_policy.max_retries : 3;
					int backoff_ms  = g_writer ? g_writer->retry_policy.backoff_ms : 100;

					while (retry_count <= max_retries) {
						FILE* fp = OpenFile(task.path, "wb");
						if (fp != nullptr) {
							if (!task.data.empty()) {
								std::fwrite(task.data.data(), 1, task.data.size(), fp);
							}
							std::fclose(fp);
							break; // Success
						}

						retry_count++;
						if (retry_count <= max_retries) {
							std::this_thread::sleep_for(
							    std::chrono::milliseconds(backoff_ms * retry_count));
						}
					}

					if (retry_count > max_retries && g_writer) {
						g_writer->dropped_count.fetch_add(1, std::memory_order_relaxed);
					}
				}
				break;
			}
			case AsyncWriter::TaskType::FileAppend: {
				if (!task.path.empty()) {
					std::error_code ec;
					const auto      parent = task.path.parent_path();
					if (!parent.empty()) {
						std::filesystem::create_directories(parent, ec);
					}

					int retry_count = 0;
					int max_retries = g_writer ? g_writer->retry_policy.max_retries : 3;
					int backoff_ms  = g_writer ? g_writer->retry_policy.backoff_ms : 100;

					while (retry_count <= max_retries) {
						FILE* fp = OpenFile(task.path, "ab");
						if (fp != nullptr) {
							if (!task.data.empty()) {
								std::fwrite(task.data.data(), 1, task.data.size(), fp);
							}
							std::fclose(fp);
							break; // Success
						}

						retry_count++;
						if (retry_count <= max_retries) {
							std::this_thread::sleep_for(
							    std::chrono::milliseconds(backoff_ms * retry_count));
						}
					}

					if (retry_count > max_retries && g_writer) {
						g_writer->dropped_count.fetch_add(1, std::memory_order_relaxed);
					}
				}
				break;
			}
			case AsyncWriter::TaskType::CustomTask: {
				if (task.custom_fn) {
					task.custom_fn();
				}
				break;
			}
			case AsyncWriter::TaskType::DualOutput: {
				// Write to file
				if (!task.path.empty()) {
					std::error_code ec;
					const auto      parent = task.path.parent_path();
					if (!parent.empty()) {
						std::filesystem::create_directories(parent, ec);
					}

					int retry_count = 0;
					int max_retries = g_writer ? g_writer->retry_policy.max_retries : 3;
					int backoff_ms  = g_writer ? g_writer->retry_policy.backoff_ms : 100;

					while (retry_count <= max_retries) {
						FILE* fp = OpenFile(task.path, "ab");
						if (fp != nullptr) {
							if (!task.data.empty()) {
								std::fwrite(task.data.data(), 1, task.data.size(), fp);
							}
							std::fclose(fp);
							break; // Success
						}

						retry_count++;
						if (retry_count <= max_retries) {
							std::this_thread::sleep_for(
							    std::chrono::milliseconds(backoff_ms * retry_count));
						}
					}

					if (retry_count > max_retries && g_writer) {
						g_writer->dropped_count.fetch_add(1, std::memory_order_relaxed);
					}
				}

				// Write to console
				if (!task.console_text.empty()) {
					if (task.console_style != fmt::text_style {}) {
						fmt::print(stdout, task.console_style, "{}", task.console_text);
					} else {
						std::fwrite(task.console_text.data(), 1, task.console_text.size(), stdout);
					}
					std::fflush(stdout);
				}
				break;
			}
		}
	} catch (...) {
		// Ignore exceptions in worker thread to prevent crashing
		if (g_writer) {
			g_writer->dropped_count.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

static void WorkerLoop(WriterState* state) {
	while (true) {
		std::deque<AsyncWriter::Task> batch;
		{
			std::unique_lock lock(state->mutex);
			state->cv_work.wait(lock, [state] {
				return !state->running.load(std::memory_order_acquire) || !state->queue.empty();
			});

			if (!state->running.load(std::memory_order_acquire) && state->queue.empty()) {
				break;
			}

			if (state->queue.empty()) {
				continue;
			}

			batch.swap(state->queue);
			state->queue_bytes.store(0, std::memory_order_relaxed);
			state->is_working.store(true, std::memory_order_release);
		}

		for (const auto& task: batch) {
			ProcessTask(task);
		}

		{
			std::lock_guard lock(state->mutex);
			state->is_working.store(false, std::memory_order_release);
			if (state->queue.empty()) {
				state->cv_flush.notify_all();
			}
		}
	}
}

} // namespace

void AsyncWriter::SetQueueConfig(const QueueConfig& config) {
	std::lock_guard init_lock(g_init_mutex);
	if (g_writer != nullptr) {
		return; // Cannot change config after initialization
	}
	s_queue_config = config;
	s_config_set   = true;
}

void AsyncWriter::SetRetryPolicy(const RetryPolicy& policy) {
	std::lock_guard init_lock(g_init_mutex);
	if (g_writer != nullptr) {
		return; // Cannot change policy after initialization
	}
	s_retry_policy = policy;
}

void AsyncWriter::Initialize() {
	std::lock_guard init_lock(g_init_mutex);
	if (g_writer != nullptr) {
		return;
	}

	auto* state = new WriterState();
	state->running.store(true, std::memory_order_release);
	state->queue_config  = s_config_set ? s_queue_config : QueueConfig {};
	state->retry_policy  = s_retry_policy;
	state->worker_thread = std::thread(WorkerLoop, state);
	g_writer             = state;
}

void AsyncWriter::Shutdown() {
	WriterState* state = nullptr;
	{
		std::lock_guard init_lock(g_init_mutex);
		state    = g_writer;
		g_writer = nullptr;
	}

	if (state == nullptr) {
		return;
	}

	{
		std::lock_guard lock(state->mutex);
		state->running.store(false, std::memory_order_release);
		state->cv_work.notify_one();
	}

	if (state->worker_thread.joinable()) {
		state->worker_thread.join();
	}

	// Drain any remaining tasks
	while (!state->queue.empty()) {
		ProcessTask(state->queue.front());
		state->queue.pop_front();
	}

	delete state;
}

void AsyncWriter::EnqueueFileWrite(const std::filesystem::path& path, std::vector<uint8_t> data,
                                   bool append) {
	if (g_writer == nullptr) {
		Initialize();
	}

	auto* state = g_writer;
	if (state == nullptr) {
		return;
	}

	size_t data_size = data.size();
	Task   task {
	    .type      = append ? TaskType::FileAppend : TaskType::FileWrite,
	    .path      = path,
	    .data      = std::move(data),
	    .custom_fn = nullptr,
	};

	{
		std::lock_guard lock(state->mutex);

		// Check queue limits and drop oldest if needed
		while (state->queue.size() >= state->queue_config.max_messages ||
		       state->queue_bytes.load(std::memory_order_relaxed) + data_size >
		           state->queue_config.max_bytes) {
			// Drop oldest message
			if (!state->queue.empty()) {
				size_t oldest_size = state->queue.front().data.size();
				state->queue.pop_front();
				state->dropped_count.fetch_add(1, std::memory_order_relaxed);
				state->queue_bytes.fetch_sub(oldest_size, std::memory_order_relaxed);
			} else {
				break;
			}
		}

		state->queue.push_back(std::move(task));
		state->queue_bytes.fetch_add(data_size, std::memory_order_relaxed);
		state->cv_work.notify_one();
	}
}

void AsyncWriter::EnqueueFileWrite(const std::filesystem::path& path, std::string_view text,
                                   bool append) {
	std::vector<uint8_t> bytes(text.begin(), text.end());
	EnqueueFileWrite(path, std::move(bytes), append);
}

void AsyncWriter::EnqueueDualOutput(const std::filesystem::path& file_path, std::string_view text,
                                    const std::string& console_text, fmt::text_style style,
                                    bool bypass_queue_limit) {
	if (g_writer == nullptr) {
		Initialize();
	}

	auto* state = g_writer;
	if (state == nullptr) {
		return;
	}

	size_t data_size = text.size();
	Task   task {
	    .type          = TaskType::DualOutput,
	    .path          = file_path,
	    .data          = std::vector<uint8_t>(text.begin(), text.end()),
	    .custom_fn     = nullptr,
	    .console_text  = console_text,
	    .console_style = style,
	    .timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                           std::chrono::steady_clock::now().time_since_epoch())
	                                           .count()),
	};

	{
		std::lock_guard lock(state->mutex);

		if (!bypass_queue_limit) {
			// Check queue limits and drop oldest if needed
			while (state->queue.size() >= state->queue_config.max_messages ||
			       state->queue_bytes.load(std::memory_order_relaxed) + data_size >
			           state->queue_config.max_bytes) {
				// Drop oldest message
				if (!state->queue.empty()) {
					size_t oldest_size = state->queue.front().data.size();
					state->queue.pop_front();
					state->dropped_count.fetch_add(1, std::memory_order_relaxed);
					state->queue_bytes.fetch_sub(oldest_size, std::memory_order_relaxed);
				} else {
					break;
				}
			}
		}

		state->queue.push_back(std::move(task));
		state->queue_bytes.fetch_add(data_size, std::memory_order_relaxed);
		state->cv_work.notify_one();
	}
}

void AsyncWriter::EnqueueTask(std::function<void()> task) {
	if (g_writer == nullptr) {
		Initialize();
	}

	auto* state = g_writer;
	if (state == nullptr) {
		return;
	}

	Task t {
	    .type      = TaskType::CustomTask,
	    .path      = {},
	    .data      = {},
	    .custom_fn = std::move(task),
	};

	{
		std::lock_guard lock(state->mutex);
		state->queue.push_back(std::move(t));
		state->cv_work.notify_one();
	}
}

void AsyncWriter::Flush() {
	auto* state = g_writer;
	if (state == nullptr) {
		return;
	}

	std::unique_lock lock(state->mutex);
	state->cv_flush.wait(lock, [state] {
		return state->queue.empty() && !state->is_working.load(std::memory_order_acquire);
	});
}

void AsyncWriter::EmergencyFlush() noexcept {
	auto* state = g_writer;
	if (state == nullptr) {
		std::fflush(nullptr);
		return;
	}

	// Prevent re-entry or recursive deadlocks during crash handling
	bool expected = false;
	if (!state->in_emergency.compare_exchange_strong(expected, true)) {
		std::fflush(nullptr);
		return;
	}

	// Drain all pending tasks synchronously
	std::deque<Task> pending;
	{
		std::unique_lock lock(state->mutex, std::defer_lock);
		// Try to lock with a short timeout / immediate attempt; if failed due to crash in locked
		// state, force stealing the queue
		if (lock.try_lock()) {
			pending.swap(state->queue);
		} else {
			// Emergency scenario: thread holding lock may be dead
			pending.swap(state->queue);
		}
	}

	while (!pending.empty()) {
		ProcessTask(pending.front());
		pending.pop_front();
	}

	std::fflush(nullptr);
}

size_t AsyncWriter::GetPendingCount() {
	auto* state = g_writer;
	if (state == nullptr) {
		return 0;
	}
	std::lock_guard lock(state->mutex);
	return state->queue.size() + (state->is_working.load(std::memory_order_acquire) ? 1 : 0);
}

size_t AsyncWriter::GetDroppedCount() {
	auto* state = g_writer;
	if (state == nullptr) {
		return 0;
	}
	return state->dropped_count.load(std::memory_order_relaxed);
}

} // namespace Common
