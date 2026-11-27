#include "graphics/host_gpu/renderer/pipeline/shaderStoreLock.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

// The single-writer lock of a shader store (ShaderStoreLock).

namespace {

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "ShaderStoreTests: failed: %s\n", message);
    std::abort();
  }
}

// A game and a precompile of one stamp must not both write its store.
void TestStoreLockSingleWriter() {
  using Libs::Graphics::ShaderStoreLock;
  const auto dir = std::filesystem::temp_directory_path() / "kyty_shader_store_lock_test";
  std::filesystem::create_directories(dir);
  const auto store = dir / "TEST.shaders";
  {
    ShaderStoreLock first;
    ShaderStoreLock second;
    Check(first.Acquire(store), "first writer gets the lock");
    Check(first.Acquire(store), "acquiring again is a no-op");
    Check(!second.Acquire(store), "a second writer is refused while the first holds it");
    Check(!second.Held(), "a refused lock is not held");
    ShaderStoreLock other_file;
    Check(other_file.Acquire(dir / "OTHER.shaders"), "another store has its own lock");
#ifndef _WIN32
    // Another process is refused too (what a game started during a precompile is).
    const pid_t child = fork();
    if (child == 0) {
      ShaderStoreLock in_child;
      _exit(in_child.Acquire(store) ? 1 : 0);
    }
    int status = 0;
    waitpid(child, &status, 0);
    Check(WIFEXITED(status) && WEXITSTATUS(status) == 0, "another process is refused");
#endif
    first.Release();
    Check(second.Acquire(store), "the lock is free once released");
  }
  {
    ShaderStoreLock again;
    Check(again.Acquire(store), "destruction releases the lock");
  }
  std::error_code error;
  std::filesystem::remove_all(dir, error);
}

} // namespace

int main() {
  TestStoreLockSingleWriter();
  std::printf("ShaderStoreTests: all passed\n");
  return 0;
}
