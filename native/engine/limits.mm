// MLX's two command-buffer limits, read and set while the process runs.
//
// MLX closes its open command buffer when the buffer holds more than max_ops
// operations, or when the arrays it has bound add up to more than max_mb
// (counted in units of 2^20 elements).  At the pinned MLX both numbers are read
// from the environment once, when the Metal device is made, and kept in two
// private members of the device object; the command encoder reads them through
// the device's getter at every check.
//
// A decode step that runs on MLX's graph binds whole weight tensors of
// gigabytes, so the size rule closes a buffer after nearly every launch (plain
// decode on Flash-Next: 389 command buffers a token for 828 launches).  A
// prefill needs the rule: it is what bounds the temporaries one buffer can pin.
// So the size limit has to differ between two phases of one process, and MLX
// at the pin has no setter.
//
// The members' place inside the object is found, never assumed: the object's
// own getter reports the two values, the object's bytes are searched for that
// adjacent pair, exactly one place may hold it, and a trial write there must
// come back through the getter.  Anything else and the limits are left alone
// for the life of the process.
#include <cstring>
#include <mutex>

#include "mlx/backend/metal/device.h"

#include "engine.h"

namespace mtplx_engine {

namespace {

std::mutex g_mutex;
bool g_searched = false;
int* g_limits = nullptr;  // {max_ops, max_mb} inside MLX's device object

int* find_limits(mlx::core::metal::Device& device) {
  const auto [ops, mb] = device.get_max_ops_mb_per_buffer();
  char* base = reinterpret_cast<char*>(&device);
  const size_t size = sizeof(mlx::core::metal::Device);
  int* found = nullptr;
  int matches = 0;
  for (size_t at = 0; at + 2 * sizeof(int) <= size; at += alignof(int)) {
    int pair[2];
    std::memcpy(pair, base + at, sizeof(pair));
    if (pair[0] == ops && pair[1] == mb) {
      found = reinterpret_cast<int*>(base + at);
      matches++;
    }
  }
  if (matches != 1) {
    return nullptr;
  }
  // The true pair always matches, so a single match is the true pair; the
  // getter still has to say so.
  found[0] = ops + 1;
  found[1] = mb + 1;
  const auto [ops_seen, mb_seen] = device.get_max_ops_mb_per_buffer();
  found[0] = ops;
  found[1] = mb;
  if (ops_seen != ops + 1 || mb_seen != mb + 1) {
    return nullptr;
  }
  return found;
}

}  // namespace

bool command_buffer_limits(int* ops, int* mb, int set_ops, int set_mb) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto& device = mlx::core::metal::device(mlx::core::Device::gpu);
  if (!g_searched) {
    g_searched = true;
    g_limits = find_limits(device);
  }
  const auto [now_ops, now_mb] = device.get_max_ops_mb_per_buffer();
  *ops = now_ops;
  *mb = now_mb;
  if (g_limits == nullptr) {
    return false;
  }
  if (set_ops > 0) {
    g_limits[0] = set_ops;
  }
  if (set_mb > 0) {
    g_limits[1] = set_mb;
  }
  return true;
}

}  // namespace mtplx_engine
