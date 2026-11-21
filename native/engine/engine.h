// The native executor: shared declarations.
//
// Three parts live behind this header:
//   kernels.mm   which pipeline is which kernel, and how each of its buffer
//                arguments is accessed (from Metal's reflection)
//   recorder.mm  hooks on the compute encoder MLX uses; a recording is the
//                list of launches one thread encoded between begin and end
//   executor.mm  a sealed program and its replay on a queue of our own
//
// Objective-C++ with ARC.  Nothing here throws.

#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mtplx_engine {

// Metal allows 31 buffer arguments per kernel.
constexpr int kMaxSlots = 31;

inline bool host_readable(id<MTLBuffer> buffer) {
  const MTLStorageMode mode = buffer.storageMode;
  return mode == MTLStorageModeShared || mode == MTLStorageModeManaged;
}

// A buffer no larger than this is copied when a recording first sees it, so a
// small host-written array that MLX later recycles can be restored.
constexpr uint64_t kSnapshotBytes = 64 * 1024;

// ------------------------------------------------------------------ kernels

struct KernelInfo {
  int handle = -1;  // position in the kernel table
  std::string name;
  id<MTLComputePipelineState> pso = nil;  // kept for the life of the process
  bool reflected = false;  // false: created before the hooks, arguments unknown
  std::string problem;  // non-empty: the kernel uses something we cannot replay
  uint8_t kind[kMaxSlots] = {};  // 0 unused, 1 buffer, 2 threadgroup memory
  char access[kMaxSlots] = {};  // 'r' or 'w' for kind 1
};

// Installs the hooks on the classes this device hands out.  Returns "" or the
// reason they could not be installed.  Safe to call more than once.
std::string install_hooks(id<MTLDevice> device);

// MLX's own command queue and ours are not ordered against each other.  Every
// command buffer MLX commits is counted, and this waits until all of them have
// completed (or the bound passes: false).  It costs one comparison when MLX
// has nothing in flight.
bool wait_for_mlx(uint64_t timeout_ns);
// A command queue of ours: its buffers are not MLX's work.
void register_own_queue(void* queue);
bool hooks_installed();
std::string encoder_class_name();
std::string device_class_name();

// Never null.  A pipeline created before the hooks comes back with
// reflected == false (macOS 26 can still describe some of them).
KernelInfo* kernel_for_pipeline(void* pso);
KernelInfo* kernel_by_id(int kernel_id);  // null when unknown
int kernel_count();
uint64_t pipelines_seen();

// ----------------------------------------------------------------- recorder

struct TraceSlot {
  uint8_t index = 0;
  uint8_t kind = 0;  // 1 buffer, 2 inline bytes
  uint32_t buffer = 0;  // index into Trace::buffers
  uint64_t offset = 0;
  uint32_t bytes_pos = 0;
  uint32_t bytes_len = 0;
};

struct TraceStep {
  void* pso = nullptr;
  uint8_t threads = 0;  // 0: dispatchThreadgroups, 1: dispatchThreads
  int indirect_buffer = -1;  // the grid argument is a read, separate from shader bindings
  uint64_t indirect_offset = 0;
  uint64_t grid[3] = {0, 0, 0};
  uint64_t group[3] = {0, 0, 0};
  bool barrier_before = false;
  int32_t mark = -1;
  uint32_t slots_pos = 0;
  uint32_t slots_len = 0;
  uint32_t tg_pos = 0;
  uint32_t tg_len = 0;
};

struct TraceBuffer {
  id<MTLBuffer> buffer = nil;  // held so the address cannot be reused
  uint64_t length = 0;
  bool has_snapshot = false;
  std::vector<uint8_t> snapshot;
  // The bytes at the end of the recording differ from the snapshot taken at
  // the first bind.  For a buffer no recorded launch writes, that means the
  // host or work outside the recording wrote it: not a constant.
  bool changed = false;
};

enum Call : int {
  kSetPipeline = 0,
  kSetBuffer,
  kSetBufferOffset,
  kSetBuffers,
  kSetBytes,
  kSetThreadgroupMemory,
  kDispatchThreadgroups,
  kDispatchThreads,
  kBarrierScope,
  kBarrierResources,
  kEndEncoding,
  kCommit,
  kDispatchIndirect,
  kCallCount,
};

struct Trace {
  std::string label;
  std::vector<TraceStep> steps;
  std::vector<TraceSlot> slots;
  std::vector<std::pair<uint8_t, uint64_t>> tg_memory;  // (index, length)
  std::vector<uint8_t> bytes;
  std::vector<TraceBuffer> buffers;
  std::vector<std::string> marks;
  uint64_t mlx_encoders = 0;
  uint64_t mlx_command_buffers = 0;
  std::string unsupported;  // first call seen that a replay cannot reproduce
  // How often MLX made each encoder call during the recording (kCallNames).
  uint64_t calls[kCallCount] = {};
};

extern const char* const kCallNames[kCallCount];

// One recording per thread.  Each returns "" or the reason it failed.
std::string record_begin(const std::string& label);
std::string record_mark(const std::string& label);
std::string record_end(std::unique_ptr<Trace>* out);
bool recording_on_this_thread();

// The original (unhooked) encoder methods for the class we hooked, so the
// replay does not pay for its own hooks.
struct EncoderImps {
  Class cls = Nil;
  IMP set_pipeline = nullptr;
  IMP set_buffer = nullptr;
  IMP set_buffer_offset = nullptr;
  IMP set_bytes = nullptr;
  IMP set_tg_memory = nullptr;
  IMP dispatch_groups = nullptr;
  IMP dispatch_threads = nullptr;
  IMP barrier_scope = nullptr;
  IMP barrier_resources = nullptr;
};
const EncoderImps& original_encoder_imps();

// MLX's two command-buffer limits (operations, and 2^20 elements, a buffer) as
// they are now; a positive set_ops or set_mb then replaces that one.  False
// when their place in MLX's device object could not be proven (limits.mm):
// the current values are still reported and nothing is changed.
bool command_buffer_limits(int* ops, int* mb, int set_ops, int set_mb);

}  // namespace mtplx_engine
