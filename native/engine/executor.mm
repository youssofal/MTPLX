// Sealing a program and replaying it.

#import "executor.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <atomic>
#include <dlfcn.h>
#include <pthread.h>
#include <map>
#include <thread>

// An encoder that ignores every call.  Encoding into it runs the engine's own
// loop and nothing else, which is how that loop's cost and its allocations are
// measured apart from Metal's.
@interface MTPLXNullEncoder : NSObject
@end

@implementation MTPLXNullEncoder
- (void)setComputePipelineState:(id)state {
}
- (void)setBuffer:(id)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
}
- (void)setBufferOffset:(NSUInteger)offset atIndex:(NSUInteger)index {
}
- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
}
- (void)setThreadgroupMemoryLength:(NSUInteger)length atIndex:(NSUInteger)index {
}
- (void)dispatchThreadgroups:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group {
}
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)buffer
                         indirectBufferOffset:(NSUInteger)offset
                        threadsPerThreadgroup:(MTLSize)group {
}
- (void)dispatchThreads:(MTLSize)grid threadsPerThreadgroup:(MTLSize)group {
}
- (void)memoryBarrierWithScope:(MTLBarrierScope)scope {
}
- (void)memoryBarrierWithResources:(const id<MTLResource> __unsafe_unretained[])resources
                             count:(NSUInteger)count {
}
- (MTLDispatchType)dispatchType {
  return MTLDispatchTypeConcurrent;
}
@end

namespace mtplx_engine {

namespace {

inline uint64_t now_ns() {
  return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

inline uint64_t thread_cpu_ns() {
  return clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID);
}

// Always inlined: a call that returned the buffer would hand it back
// autoreleased whenever the return handshake fails.
__attribute__((always_inline)) inline id<MTLBuffer> metal_buffer(
    const mx::allocator::Buffer& buffer) {
  return (__bridge id<MTLBuffer>)const_cast<void*>(buffer.ptr());
}

// The CPU address of a buffer's bytes, read through MLX's accessor and never
// with -contents on an id: -contents returns an inner pointer, so ARC retains
// and autoreleases the receiver to keep it alive while the pointer is in use.
// The engine is called from Python threads, which have no autorelease pool,
// so that reference was never given back: the buffer stayed allocated after
// MLX freed it, for the life of the thread, and outside MLX's count.
inline uint8_t* host_bytes(const mx::allocator::Buffer& buffer) {
  return static_cast<uint8_t*>(mx::allocator::Buffer(const_cast<void*>(buffer.ptr())).raw_ptr());
}

inline uint8_t* host_bytes(__unsafe_unretained id<MTLBuffer> buffer) {
  return static_cast<uint8_t*>(mx::allocator::Buffer((__bridge void*)buffer).raw_ptr());
}

inline mx::array::Flags unpack_flags(uint8_t bits) {
  mx::array::Flags flags;
  flags.contiguous = (bits & 1) != 0;
  flags.row_contiguous = (bits & 2) != 0;
  flags.col_contiguous = (bits & 4) != 0;
  return flags;
}

typedef void (*SetPipelineFn)(id, SEL, id<MTLComputePipelineState>);
typedef void (*SetBufferFn)(id, SEL, id<MTLBuffer>, NSUInteger, NSUInteger);
typedef void (*SetBufferOffsetFn)(id, SEL, NSUInteger, NSUInteger);
typedef void (*SetBytesFn)(id, SEL, const void*, NSUInteger, NSUInteger);
typedef void (*SetTgMemoryFn)(id, SEL, NSUInteger, NSUInteger);
typedef void (*DispatchFn)(id, SEL, MTLSize, MTLSize);
typedef void (*BarrierScopeFn)(id, SEL, MTLBarrierScope);
typedef void (*BarrierResourcesFn)(
    id, SEL, const id<MTLResource> __unsafe_unretained*, NSUInteger);

struct DtypeName {
  const char* name;
  mx::Dtype dtype;
};

const DtypeName kDtypes[] = {
    {"bool", mx::bool_},       {"uint8", mx::uint8},     {"uint16", mx::uint16},
    {"uint32", mx::uint32},    {"uint64", mx::uint64},   {"int8", mx::int8},
    {"int16", mx::int16},      {"int32", mx::int32},     {"int64", mx::int64},
    {"float16", mx::float16},  {"float32", mx::float32}, {"float64", mx::float64},
    {"bfloat16", mx::bfloat16}, {"complex64", mx::complex64},
};

// The allocation probe.  libmalloc calls a logger, when one is set, for every
// allocation in the process; ours counts those made by the one thread that is
// probing.  The logger runs inside malloc, so it must not touch anything that
// can allocate: no thread-local storage (its first use on a thread allocates),
// only the thread's own id and two plain globals.
typedef void (MallocLogger)(uint32_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uint32_t);
std::atomic<uint64_t> g_alloc_count{0};
std::atomic<pthread_t> g_alloc_thread{nullptr};
MallocLogger** g_logger_slot = nullptr;
bool g_alloc_probe = false;

void count_allocation(uint32_t type, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uint32_t) {
  // type bit 1: an allocation (bit 2 is a free; a realloc carries both)
  if ((type & 2u) != 0 && g_alloc_thread.load(std::memory_order_relaxed) == pthread_self()) {
    g_alloc_count.fetch_add(1, std::memory_order_relaxed);
  }
}

struct AllocScope {
  uint64_t start = 0;
  pthread_t outer = nullptr;
  AllocScope() {
    if (g_alloc_probe) {
      outer = g_alloc_thread.exchange(pthread_self(), std::memory_order_relaxed);
      start = g_alloc_count.load(std::memory_order_relaxed);
    }
  }
  uint64_t stop() {
    if (!g_alloc_probe) {
      return 0;
    }
    const uint64_t counted = g_alloc_count.load(std::memory_order_relaxed) - start;
    g_alloc_thread.store(outer, std::memory_order_relaxed);
    return counted;
  }
};

}  // namespace

bool alloc_probe_enable(bool on) {
  if (g_logger_slot == nullptr) {
    // Looked up by name at run time: the module must load on a system that
    // does not have it.
    g_logger_slot = (MallocLogger**)dlsym(RTLD_DEFAULT, "malloc_logger");
  }
  if (g_logger_slot == nullptr) {
    return false;
  }
  if (on) {
    if (*g_logger_slot != nullptr && *g_logger_slot != count_allocation) {
      return false;  // someone else is logging (a malloc debugging tool)
    }
    *g_logger_slot = count_allocation;
  } else if (*g_logger_slot == count_allocation) {
    *g_logger_slot = nullptr;
  }
  g_alloc_probe = on;
  return true;
}

const char* dtype_name(mx::Dtype dtype) {
  for (const auto& entry : kDtypes) {
    if (entry.dtype.val() == dtype.val()) {
      return entry.name;
    }
  }
  return "unknown";
}

bool dtype_from_name(const std::string& name, mx::Dtype* out) {
  for (const auto& entry : kDtypes) {
    if (name == entry.name) {
      *out = entry.dtype;
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------ Program

Program::~Program() {
  for (auto& buffer : owned) {
    if (buffer.ptr() != nullptr) {
      mx::allocator::free(buffer);
    }
  }
}

// ------------------------------------------------------------------- Engine

Engine::Engine() = default;

Engine::~Engine() {
  // Releasing the queue makes Metal purge its resource pool, which
  // autoreleases.  An engine can end on a thread with no pool of its own (the
  // module's self-check, the interpreter's exit), so the Metal objects are
  // let go of inside one.
  @autoreleasepool {
    for (auto& slot : pool_) {
      if (slot.buffer.ptr() != nullptr) {
        mx::allocator::free(slot.buffer);
      }
    }
    flight_ = Flight();
    stuck_buffer_ = nil;
    fences_.clear();
    null_encoder_ = nil;
    queue_ = nil;
  }
}

std::string Engine::init() {
  @autoreleasepool {
    // MLX's device, taken from a buffer of MLX's own allocator: the engine
    // must bind MLX's buffers, so it has to be the same device object.
    try {
      mx::allocator::Buffer probe = mx::allocator::malloc(64);
      if (probe.ptr() == nullptr) {
        return "no_metal_buffer";
      }
      device_ = metal_buffer(probe).device;
      mx::allocator::free(probe);
    } catch (const std::exception& error) {
      return std::string("mlx_allocator_failed: ") + error.what();
    }
    if (device_ == nil) {
      return "no_metal_device";
    }
    std::string problem = install_hooks(device_);
    if (!problem.empty()) {
      return problem;
    }
    queue_ = [device_ newCommandQueue];
    if (queue_ == nil) {
      return "no_command_queue";
    }
    queue_.label = @"mtplx.engine";
    register_own_queue((__bridge void*)queue_);
    gpu_name = device_.name.UTF8String ?: "";
    if (@available(macOS 14.0, *)) {
      gpu_arch = device_.architecture.name.UTF8String ?: "";
    }
  }
  return "";
}

int Engine::pin(const mx::array& array) {
  std::lock_guard lock(mutex_);
  const void* key = array.buffer().ptr();
  auto found = pin_by_buffer_.find(key);
  if (found != pin_by_buffer_.end()) {
    return found->second;
  }
  int slot = (int)pins_.size();
  pins_.push_back(array);
  pin_by_buffer_.emplace(key, slot);
  return slot;
}

int Engine::pin_slot_for(const void* buffer) const {
  auto found = pin_by_buffer_.find(buffer);
  return found == pin_by_buffer_.end() ? -1 : found->second;
}

bool Engine::reserve_temp(size_t index, uint64_t bytes) {
  if (index >= pool_.size()) {
    pool_.resize(index + 1);
    pool_slots_.store(pool_.size(), std::memory_order_relaxed);
  }
  PoolSlot& slot = pool_[index];
  if (bytes == 0 || (slot.buffer.ptr() != nullptr && slot.length >= bytes)) {
    return true;
  }
  // Invalidate every sealed table before replacing its allocation. A
  // failed allocation must never leave a table pointing into a freed slot.
  ++pool_generation_;
  if (slot.buffer.ptr() != nullptr) {
    pool_bytes_.fetch_sub(slot.length, std::memory_order_relaxed);
    mx::allocator::free(slot.buffer);
    slot.buffer = mx::allocator::Buffer{nullptr};
    slot.length = 0;
  }
  slot.buffer = mx::allocator::malloc(bytes);
  if (slot.buffer.ptr() == nullptr) {
    return false;
  }
  slot.length = metal_buffer(slot.buffer).length;
  pool_bytes_.fetch_add(slot.length, std::memory_order_relaxed);
  return true;
}

uint64_t Engine::release_idle_temporaries() {
  std::lock_guard lock(mutex_);
  // Even a completed submission owns its buffers until it is collected.
  if (flight_.active || stuck_ || pool_.empty()) {
    return 0;
  }
  const uint64_t bytes = pool_bytes();
  ++pool_generation_;
  for (auto& slot : pool_) {
    if (slot.buffer.ptr() != nullptr) {
      mx::allocator::free(slot.buffer);
    }
  }
  pool_.clear();
  pool_bytes_.store(0, std::memory_order_relaxed);
  pool_slots_.store(0, std::memory_order_relaxed);
  return bytes;
}

void Engine::count_refusal(const std::string& reason) {
  refusals_[reason] += 1;
}

std::shared_ptr<Program> Engine::seal(ProgramSpec&& spec, std::string* error) {
  std::lock_guard lock(mutex_);
  // Sealing can replace working memory, which a replay on the GPU is using.
  if (stuck_ && !recovered()) {
    *error = "engine_stuck";
    return nullptr;
  }
  if (!flight_left_gpu(wait_floor_ns_)) {
    *error = "engine_busy";
    return nullptr;
  }
  auto program = std::make_shared<Program>();
  program->engine = this;
  program->name = std::move(spec.name);
  program->buffers = std::move(spec.buffers);
  program->inputs = std::move(spec.inputs);
  program->outputs = std::move(spec.outputs);

  const size_t n_buffers = program->buffers.size();
  program->table.assign(n_buffers, nil);
  program->delta.assign(n_buffers, 0);
  program->required.assign(n_buffers, 0);
  for (size_t b = 0; b < n_buffers; ++b) {
    program->required[b] = program->buffers[b].nbytes;
  }
  program->owned_of.assign(n_buffers, -1);
  program->temp_slot.assign(n_buffers, -1);
  program->members.assign(n_buffers, {});

  for (size_t i = 0; i < program->inputs.size(); ++i) {
    int b = program->inputs[i].buffer;
    if (b < 0) {
      continue;
    }
    if ((size_t)b >= n_buffers || program->buffers[(size_t)b].role != Role::Input) {
      *error = "input_buffer_role";
      return nullptr;
    }
    program->members[(size_t)b].push_back((int)i);
  }
  for (auto& out : program->outputs) {
    if (out.alias_input >= (int)program->inputs.size()) {
      *error = "output_alias_range";
      return nullptr;
    }
    if (out.alias_input < 0 && out.buffer >= 0) {
      if ((size_t)out.buffer >= n_buffers ||
          program->buffers[(size_t)out.buffer].role != Role::Output) {
        *error = "output_buffer_role";
        return nullptr;
      }
    }
  }

  // Working memory.  Temporaries hold nothing between runs, so every program
  // of the engine shares one pool: the k-th largest temporary of any program
  // uses pool slot k, and a slot is as large as the largest that uses it.
  std::vector<std::pair<uint64_t, int>> temps;
  try {
    for (size_t b = 0; b < n_buffers; ++b) {
      BufferSpec& buffer = program->buffers[b];
      switch (buffer.role) {
        case Role::Input:
          if (program->members[b].empty()) {
            *error = "input_buffer_without_input";
            return nullptr;
          }
          program->input_buffers.push_back((int)b);
          if (buffer.clobbered) {
            for (int member : program->members[b]) {
              program->clobbered_inputs.push_back(member);
            }
          }
          break;
        case Role::Output:
          if (buffer.data.size() > buffer.nbytes) {
            *error = "bytes_longer_than_buffer";
            return nullptr;
          }
          program->output_buffers.push_back((int)b);
          if (!buffer.data.empty()) {
            program->seeded_outputs.push_back((int)b);
          }
          break;
        case Role::Pinned: {
          if (buffer.index < 0 || (size_t)buffer.index >= pins_.size()) {
            *error = "pin_slot_unknown";
            return nullptr;
          }
          const mx::array& pinned = pins_[(size_t)buffer.index];
          program->pinned.push_back(pinned);
          program->table[b] = metal_buffer(pinned.buffer());
          break;
        }
        case Role::Const:
        case Role::Seed: {
          if (buffer.data.size() > buffer.nbytes) {
            *error = "bytes_longer_than_buffer";
            return nullptr;
          }
          mx::allocator::Buffer made = mx::allocator::malloc(buffer.nbytes);
          if (made.ptr() == nullptr) {
            *error = "allocation_failed";
            return nullptr;
          }
          program->owned_of[b] = (int)program->owned.size();
          program->owned.push_back(made);
          id<MTLBuffer> metal = metal_buffer(made);
          program->table[b] = metal;
          std::memcpy(host_bytes(made), buffer.data.data(), buffer.data.size());
          program->const_bytes += metal.length;
          if (buffer.role == Role::Seed) {
            program->seed_buffers.push_back((int)b);
          } else {
            buffer.data.clear();
            buffer.data.shrink_to_fit();
          }
          break;
        }
        case Role::Temp:
          temps.emplace_back(buffer.nbytes, (int)b);
          break;
      }
    }
    std::sort(temps.begin(), temps.end(), [](const auto& a, const auto& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    for (size_t k = 0; k < temps.size(); ++k) {
      if (!reserve_temp(k, temps[k].first)) {
        *error = "allocation_failed";
        return nullptr;
      }
      program->temp_slot[(size_t)temps[k].second] = (int)k;
      program->temp_bytes += temps[k].first;
    }
  } catch (const std::exception& failure) {
    *error = std::string("allocation_failed: ") + failure.what();
    return nullptr;
  }

  // The length of each fresh output buffer.  A state leaf the program
  // rewrites into a fresh buffer every run comes back as an input the next
  // run, and that input's buffer must be as long as it was in the recording
  // (the program uses all of it: can_run).  MLX's allocator hands out what
  // its cache holds, so a recorded output buffer is often shorter than the
  // recorded buffer of the input it feeds; allocated at its own recorded
  // length, the leaf would arrive too short every second run.
  program->fresh_bytes.reserve(program->output_buffers.size());
  for (int b : program->output_buffers) {
    uint64_t bytes = program->buffers[(size_t)b].nbytes;
    for (const ArraySpec& out : program->outputs) {
      if (out.buffer != b || out.alias_input >= 0) {
        continue;
      }
      for (int member : program->clobbered_inputs) {
        const ArraySpec& in = program->inputs[(size_t)member];
        if (in.dtype.val() == out.dtype.val() && in.shape == out.shape) {
          bytes = std::max(bytes, program->buffers[(size_t)in.buffer].nbytes);
        }
      }
    }
    program->fresh_bytes.push_back(bytes);
  }

  // Flatten the steps.
  // Equal inline bytes are stored once and share an id.
  std::map<std::string, std::pair<uint32_t, uint32_t>> interned;
  program->steps.reserve(spec.steps.size());
  for (const StepSpec& step : spec.steps) {
    KernelInfo* kernel = kernel_by_id(step.kernel);
    if (kernel == nullptr || kernel->pso == nil) {
      *error = "unknown_kernel";
      return nullptr;
    }
    ExecStep exec;
    exec.pso = kernel->pso;
    exec.kernel = step.kernel;
    exec.threads = step.threads ? 1 : 0;
    exec.indirect_buffer = step.indirect_buffer;
    exec.indirect_offset = step.indirect_offset;
    if (step.indirect_buffer >= 0) {
      const size_t b = (size_t)step.indirect_buffer;
      const uint64_t offset = step.indirect_offset;
      if (step.threads || b >= n_buffers || offset % alignof(uint32_t)) {
        *error = "invalid_indirect_grid";
        return nullptr;
      }
      const auto& buffer = program->buffers[b];
      const uint64_t size = buffer.role == Role::Pinned ? program->table[b].length : buffer.nbytes;
      if (offset > size || size - offset < sizeof(MTLDispatchThreadgroupsIndirectArguments)) {
        *error = "indirect_grid_out_of_bounds";
        return nullptr;
      }
      if (buffer.role == Role::Input) {
        program->indirect_inputs.emplace_back((int)b, offset);
      }
    }
    exec.barrier = step.barrier_before ? 1 : 0;
    exec.barrier_min = 0;
    exec.barrier_kind = 0;
    exec.res_pos = 0;
    exec.res_len = 0;
    exec.chain_run = 0;
    exec.grid = MTLSizeMake(step.grid[0], step.grid[1], step.grid[2]);
    exec.group = MTLSizeMake(step.group[0], step.group[1], step.group[2]);
    exec.bind_pos = (uint32_t)program->binds.size();
    for (const auto& [slot, buffer, offset, can_write] : step.bindings) {
      (void)can_write;
      if (slot < 0 || slot >= kMaxSlots || buffer < 0 || (size_t)buffer >= n_buffers) {
        *error = "binding_out_of_range";
        return nullptr;
      }
      program->binds.push_back(ExecBind{(uint8_t)slot, (uint32_t)buffer, offset});
    }
    exec.bind_len = (uint16_t)(program->binds.size() - exec.bind_pos);
    exec.inline_pos = (uint32_t)program->inlines.size();
    for (const auto& [slot, data] : step.inline_bytes) {
      if (slot < 0 || slot >= kMaxSlots) {
        *error = "binding_out_of_range";
        return nullptr;
      }
      std::string key((const char*)data.data(), data.size());
      auto found = interned.find(key);
      ExecInline entry;
      entry.slot = (uint8_t)slot;
      entry.len = (uint32_t)data.size();
      if (found == interned.end()) {
        entry.intern = (uint32_t)interned.size() + 1;
        entry.pos = (uint32_t)program->bytes.size();
        program->bytes.insert(program->bytes.end(), data.begin(), data.end());
        while (program->bytes.size() % kInlineAlign != 0) {
          program->bytes.push_back(0);
        }
        interned.emplace(std::move(key), std::make_pair(entry.intern, entry.pos));
      } else {
        entry.intern = found->second.first;
        entry.pos = found->second.second;
      }
      program->inlines.push_back(entry);
    }
    exec.inline_len = (uint16_t)(program->inlines.size() - exec.inline_pos);
    exec.tg_pos = (uint32_t)program->tg.size();
    for (const auto& [slot, length] : step.tg_memory) {
      if (slot < 0 || slot >= kMaxSlots) {
        *error = "binding_out_of_range";
        return nullptr;
      }
      program->tg.emplace_back((uint8_t)slot, length);
    }
    exec.tg_len = (uint16_t)(program->tg.size() - exec.tg_pos);
    program->steps.push_back(exec);
  }
  // The inline constants as one buffer.
  if (!program->bytes.empty()) {
    try {
      mx::allocator::Buffer made = mx::allocator::malloc(program->bytes.size());
      if (made.ptr() == nullptr) {
        *error = "allocation_failed";
        return nullptr;
      }
      program->owned.push_back(made);
      id<MTLBuffer> metal = metal_buffer(made);
      std::memcpy(host_bytes(made), program->bytes.data(), program->bytes.size());
      program->constants = metal;
      program->const_bytes += metal.length;
    } catch (const std::exception& failure) {
      *error = std::string("allocation_failed: ") + failure.what();
      return nullptr;
    }
  }

  // Where barriers are needed, from what each launch reads and writes.
  //
  // A launch must not start before an earlier launch has finished when it
  // touches a buffer that one wrote, or writes a buffer that one read.  Two
  // answers are kept.  `barrier_min`: with whole-pass barriers, which clear
  // every pending hazard at once.  `barrier_kind`: with barriers that name
  // buffers, which the documentation defines for writes becoming visible;
  // a write after a read is given a whole-pass barrier to stay safe.
  {
    std::vector<uint8_t> epoch_w(n_buffers, 0), epoch_r(n_buffers, 0);
    std::vector<uint8_t> pend_w(n_buffers, 0), pend_r(n_buffers, 0);
    std::vector<uint32_t> touched, named;
    for (size_t index = 0; index < spec.steps.size(); ++index) {
      const StepSpec& step = spec.steps[index];
      ExecStep& exec = program->steps[index];
      bool need = false;
      bool write_after_read = false;
      named.clear();
      auto each_access = [&](auto visit) {
        for (const auto& [slot, buffer, offset, writes] : step.bindings) {
          visit((size_t)buffer, writes);
        }
        if (step.indirect_buffer >= 0) {
          visit((size_t)step.indirect_buffer, false);
        }
      };
      each_access([&](size_t b, bool can_write) {
        if (epoch_w[b] || (can_write && epoch_r[b])) {
          need = true;
        }
        if (pend_w[b] && std::find(named.begin(), named.end(), (uint32_t)b) == named.end()) {
          named.push_back((uint32_t)b);
        }
        if (can_write && pend_r[b] && !pend_w[b]) {
          write_after_read = true;
        }
      });
      if (need) {
        exec.barrier_min = 1;
        program->barriers_min += 1;
        for (uint32_t b : touched) {
          epoch_w[b] = 0;
          epoch_r[b] = 0;
        }
        touched.clear();
      }
      if (write_after_read || named.size() > kMaxSlots + 1) {
        exec.barrier_kind = 2;
        program->barriers_whole += 1;
        std::fill(pend_w.begin(), pend_w.end(), 0);
        std::fill(pend_r.begin(), pend_r.end(), 0);
      } else if (!named.empty()) {
        exec.barrier_kind = 1;
        exec.res_pos = (uint32_t)program->barrier_buffers.size();
        exec.res_len = (uint16_t)named.size();
        program->barrier_buffers.insert(program->barrier_buffers.end(), named.begin(), named.end());
        program->barriers_named += 1;
        for (uint32_t b : named) {
          pend_w[b] = 0;
        }
      }
      each_access([&](size_t b, bool can_write) {
        if (can_write) {
          epoch_w[b] = 1;
          pend_w[b] = 1;
        } else {
          epoch_r[b] = 1;
          pend_r[b] = 1;
        }
        touched.push_back((uint32_t)b);
      });
      if (exec.barrier) {
        program->barriers_recorded += 1;
      }
    }
    // Runs of launches that each wait for the one before: a serial pass
    // orders those with no barrier at all.
    const size_t n = program->steps.size();
    for (size_t index = n; index-- > 0;) {
      ExecStep& exec = program->steps[index];
      const bool alone = exec.barrier_min && (index + 1 == n || program->steps[index + 1].barrier_min);
      exec.chain_run = alone ? 1 + (index + 1 < n ? program->steps[index + 1].chain_run : 0) : 0;
    }
  }

  // Resolve the temporaries now; a later pool growth bumps the generation and
  // the next run resolves them again.
  for (size_t b = 0; b < n_buffers; ++b) {
    if (program->temp_slot[b] >= 0) {
      program->table[b] = metal_buffer(pool_[(size_t)program->temp_slot[b]].buffer);
    }
  }
  program->pool_generation = pool_generation_;
  if (held_.capacity() < program->inputs.size()) {
    held_.reserve(program->inputs.size());
  }
  if (fresh_.capacity() < program->output_buffers.size()) {
    fresh_.reserve(program->output_buffers.size());
  }
  return program;
}

bool Engine::settle(const std::vector<const mx::array*>& arrays, uint64_t timeout_us) {
  const uint64_t deadline = now_ns() + timeout_us * 1000;
  for (;;) {
    bool single = true;
    for (const mx::array* array : arrays) {
      if (array->status() != mx::array::Status::unscheduled &&
          array->data_shared_ptr().use_count() > 1) {
        single = false;
        break;
      }
    }
    if (single) {
      return true;
    }
    if (now_ns() >= deadline) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
}

std::string Engine::can_run(
    Program& program,
    const std::vector<const mx::array*>& inputs,
    bool settle_inputs,
    bool drain,
    bool holders,
    std::vector<int>* grow) {
  if (inputs.size() != program.inputs.size()) {
    return "input_count";
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const mx::array& array = *inputs[i];
    const ArraySpec& spec = program.inputs[i];
    if (array.status() == mx::array::Status::unscheduled) {
      return "input_not_evaluated";
    }
    // Whatever MLX still has in flight that produces an input must be
    // complete before our launches read it.  An array MLX evaluated without
    // waiting carries the event of its batch; waiting on that event is the
    // cheapest drain that is always enough, and it costs one status check
    // when there is nothing to wait for.
    if (drain && !array.is_available()) {
      try {
        const_cast<mx::array&>(array).wait();
      } catch (const std::exception&) {
        return "input_failed";
      }
    }
    if (array.dtype().val() != spec.dtype.val()) {
      return "input_dtype";
    }
    if (array.shape() != spec.shape) {
      return "input_shape";
    }
    // The stride of a dimension of extent 1 is never used to address an
    // element, and MLX leaves whatever the array's creation route put there,
    // so the same layout can carry different values in it.
    const auto& strides = array.strides();
    if (strides.size() != spec.strides.size()) {
      return "input_strides";
    }
    for (size_t d = 0; d < strides.size(); ++d) {
      if (spec.shape[d] > 1 && strides[d] != spec.strides[d]) {
        return "input_strides";
      }
    }
  }
  for (int b : program.input_buffers) {
    const std::vector<int>& members = program.members[(size_t)b];
    const mx::array& first = *inputs[(size_t)members[0]];
    const void* buffer = first.buffer().ptr();
    if (buffer == nullptr) {
      return "input_empty";
    }
    int64_t shift = first.offset() - program.inputs[(size_t)members[0]].offset;
    for (size_t k = 1; k < members.size(); ++k) {
      const mx::array& other = *inputs[(size_t)members[k]];
      if (other.buffer().ptr() != buffer ||
          other.offset() - program.inputs[(size_t)members[k]].offset != shift) {
        return "input_alias";
      }
    }
    if (program.buffers[(size_t)b].clobbered) {
      // The program writes into this input's buffer, and not only where the
      // array is.  MLX frees an input the moment its last use is evaluated
      // and hands the buffer to a later launch of the same evaluation as
      // working memory; a view of a longer buffer (a request's first round)
      // is then written far past the array's own end.  So every byte of the
      // recorded buffer must exist in the buffer handed in now: a smaller one
      // (a plain copy of the array, say) would be written past its end.
      // What a run needs of it is the recorded length, or less once a run
      // has shown the program leaves the room past the array alone
      // (Program::required, relax_input).
      const int64_t recorded = (int64_t)program.required[(size_t)b];
      const int64_t length = (int64_t)metal_buffer(first.buffer()).length;
      if (shift < 0 || recorded + shift > length) {
        // A buffer of the recorded length can be made for it: the run copies
        // the array there and writes into the copy, which nobody else holds
        // (so neither test below applies to it).  A state leaf lands in a
        // shorter buffer whenever MLX's path wrote it last (the allocator
        // hands that path whatever fits), and a round is not given back to
        // MLX for the price of copying a few kilobytes.
        const int64_t pad = shift < 0 ? -shift : 0;
        const int64_t need = std::max(recorded + (shift > 0 ? shift : 0), pad + length);
        if (grow == nullptr || members.size() != 1 || (uint64_t)need > kGrowLimit) {
          const int member = members[0];
          last_short_detail_ = "input " + std::to_string(member) + ": buffer " +
              std::to_string(length) + " bytes, the program needs " +
              std::to_string(recorded) + (shift != 0 ? " at shift " + std::to_string(shift) : "") +
              " (recorded " + std::to_string(program.buffers[(size_t)b].nbytes) +
              ", the array ends at " +
              std::to_string(first.offset() + (int64_t)first.data_size() * (int64_t)first.itemsize()) +
              ")";
          return "input_buffer_small";
        }
        grow->push_back(b);
        continue;
      }
      // That is also only safe when the caller's array is the last holder of
      // its data: the same test MLX applies before it reuses an input's
      // buffer for an output.
      if (members.size() != 1) {
        return "state_shared";
      }
      if (holders) {
        std::string shared = sole_holder(program, first, settle_inputs);
        if (!shared.empty()) {
          return shared;
        }
      }
    }
  }
  for (const auto& [buffer, offset] : program.indirect_inputs) {
    const int member = program.members[(size_t)buffer][0];
    const mx::array& array = *inputs[(size_t)member];
    const int64_t shifted = (int64_t)offset + array.offset() - program.inputs[(size_t)member].offset;
    const uint64_t length = metal_buffer(array.buffer()).length;
    if (shifted < 0 || shifted % alignof(uint32_t) || (uint64_t)shifted > length ||
        length - (uint64_t)shifted < sizeof(MTLDispatchThreadgroupsIndirectArguments)) {
      return "indirect_grid_out_of_bounds";
    }
  }
  // Independently bound read-only inputs may be views of one allocation:
  // each binding retains its own offset and neither changes any bytes.
  // A program with a written input still requires all its buffers to be
  // distinct. Recorded aliases must preserve their relative offsets above.
  const size_t n = program.input_buffers.size();
  if (n > 1 && !program.clobbered_inputs.empty()) {
    static thread_local std::vector<const void*> seen;
    seen.clear();
    for (int b : program.input_buffers) {
      seen.push_back(inputs[(size_t)program.members[(size_t)b][0]]->buffer().ptr());
    }
    std::sort(seen.begin(), seen.end());
    for (size_t k = 1; k < seen.size(); ++k) {
      if (seen[k] == seen[k - 1]) {
        return "input_alias";
      }
    }
  }
  return "";
}

std::string Engine::sole_holder(Program& program, const mx::array& first, bool settle_inputs) {
  if (first.is_donatable()) {
    return "";
  }
  if (!settle_inputs || program.settle_was_futile) {
    return "state_shared";
  }
  const uint64_t waited_from = now_ns();
  static thread_local std::vector<const mx::array*> one(1);
  one[0] = &first;
  settle(one, 1000);
  settle_waits_ += 1;
  settle_wait_ns_ += now_ns() - waited_from;
  if (!first.is_donatable()) {
    program.settle_was_futile = true;
    return "state_shared";
  }
  return "";
}

std::string Engine::settle_run(Program& program) {
  // The second half of the legality check when the launches were encoded
  // before MLX's queue was empty: every input complete, and every input the
  // program writes into held by the caller alone.
  for (const mx::array& array : held_) {
    if (!array.is_available()) {
      try {
        const_cast<mx::array&>(array).wait();
      } catch (const std::exception&) {
        return "input_failed";
      }
    }
  }
  for (int b : program.input_buffers) {
    if (!program.buffers[(size_t)b].clobbered) {
      continue;
    }
    // held_ adds one holder of the array object, not of its data.
    const mx::array& first = held_[(size_t)program.members[(size_t)b][0]];
    std::string shared = sole_holder(program, first, true);
    if (!shared.empty()) {
      return shared;
    }
  }
  return "";
}

void Engine::encode(
    Program& program,
    id<MTLComputeCommandEncoder> encoder,
    size_t first,
    size_t last,
    const RunOptions& options,
    RunStats* stats) {
  const int barriers = options.barriers;
  static const SEL sel_pipeline = @selector(setComputePipelineState:);
  static const SEL sel_buffer = @selector(setBuffer:offset:atIndex:);
  static const SEL sel_offset = @selector(setBufferOffset:atIndex:);
  static const SEL sel_bytes = @selector(setBytes:length:atIndex:);
  static const SEL sel_tg = @selector(setThreadgroupMemoryLength:atIndex:);
  static const SEL sel_groups = @selector(dispatchThreadgroups:threadsPerThreadgroup:);
  static const SEL sel_threads = @selector(dispatchThreads:threadsPerThreadgroup:);
  static const SEL sel_barrier = @selector(memoryBarrierWithScope:);
  static const SEL sel_barrier_named = @selector(memoryBarrierWithResources:count:);

  // The methods themselves, looked up once per encoder.  For the class the
  // recorder wrapped these are the originals, so a replay skips its own hooks.
  const EncoderImps& known = original_encoder_imps();
  Class cls = object_getClass(encoder);
  const bool direct = cls == known.cls;
  auto set_pipeline = (SetPipelineFn)(
      direct ? known.set_pipeline : class_getMethodImplementation(cls, sel_pipeline));
  auto set_buffer =
      (SetBufferFn)(direct ? known.set_buffer : class_getMethodImplementation(cls, sel_buffer));
  auto set_offset = (SetBufferOffsetFn)(
      direct ? known.set_buffer_offset : class_getMethodImplementation(cls, sel_offset));
  auto set_bytes =
      (SetBytesFn)(direct ? known.set_bytes : class_getMethodImplementation(cls, sel_bytes));
  auto set_tg =
      (SetTgMemoryFn)(direct ? known.set_tg_memory : class_getMethodImplementation(cls, sel_tg));
  auto dispatch_groups = (DispatchFn)(
      direct ? known.dispatch_groups : class_getMethodImplementation(cls, sel_groups));
  auto dispatch_threads = (DispatchFn)(
      direct ? known.dispatch_threads : class_getMethodImplementation(cls, sel_threads));
  auto barrier = (BarrierScopeFn)(
      direct ? known.barrier_scope : class_getMethodImplementation(cls, sel_barrier));
  auto barrier_named = (BarrierResourcesFn)(
      direct ? known.barrier_resources : class_getMethodImplementation(cls, sel_barrier_named));

  // What the encoder currently has bound, so a binding already in place is
  // not sent again.
  __unsafe_unretained id<MTLBuffer> bound[kMaxSlots] = {};
  uint64_t bound_offset[kMaxSlots] = {};
  uint32_t bound_inline[kMaxSlots] = {};
  uint64_t bound_tg[kMaxSlots] = {};
  bool tg_set[kMaxSlots] = {};
  __unsafe_unretained id<MTLComputePipelineState> pipeline = nil;

  const ExecBind* binds = program.binds.data();
  const ExecInline* inlines = program.inlines.data();
  const uint8_t* bytes = program.bytes.data();
  __unsafe_unretained id<MTLBuffer>* table = program.table.data();
  const int64_t* delta = program.delta.data();
  const bool concurrent = encoder.dispatchType == MTLDispatchTypeConcurrent;
  __unsafe_unretained id<MTLBuffer> constants = options.copy_inline ? nil : program.constants;
  uint64_t sent = 0;
  uint64_t skipped = 0;
  uint64_t barriers_sent = 0;
  const uint32_t* barrier_buffers = program.barrier_buffers.data();

  for (size_t s = first; s < last; ++s) {
    const ExecStep& step = program.steps[s];
    // A pass starts with nothing pending, and a serial pass orders its
    // launches by itself.
    if (concurrent && s != first) {
      uint8_t kind = 0;  // 0 none, 1 named buffers, 2 whole pass
      switch (barriers) {
        case kRecorded:
          kind = step.barrier ? 2 : 0;
          break;
        case kMinimal:
          kind = step.barrier_min ? 2 : 0;
          break;
        case kNamed:
          kind = step.barrier_kind;
          break;
        default:
          break;
      }
      if (kind == 2) {
        barrier(encoder, sel_barrier, MTLBarrierScopeBuffers);
        ++barriers_sent;
      } else if (kind == 1) {
        // Up to one descriptor in addition to all shader binding slots.
        __unsafe_unretained id<MTLResource> named[kMaxSlots + 1];
        const uint16_t count = step.res_len;
        for (uint16_t k = 0; k < count; ++k) {
          named[k] = table[barrier_buffers[step.res_pos + k]];
        }
        barrier_named(encoder, sel_barrier_named, named, count);
        ++barriers_sent;
      }
    }
    if (step.pso != pipeline) {
      pipeline = step.pso;
      set_pipeline(encoder, sel_pipeline, pipeline);
    }
    const ExecBind* bind = binds + step.bind_pos;
    for (uint16_t k = 0; k < step.bind_len; ++k, ++bind) {
      __unsafe_unretained id<MTLBuffer> buffer = table[bind->buffer];
      const uint64_t offset = (uint64_t)((int64_t)bind->offset + delta[bind->buffer]);
      const uint8_t slot = bind->slot;
      if (bound[slot] == buffer) {
        if (bound_offset[slot] == offset) {
          ++skipped;
          continue;
        }
        set_offset(encoder, sel_offset, offset, slot);
      } else {
        set_buffer(encoder, sel_buffer, buffer, offset, slot);
        bound[slot] = buffer;
        bound_inline[slot] = 0;
      }
      bound_offset[slot] = offset;
      ++sent;
    }
    const ExecInline* inl = inlines + step.inline_pos;
    if (constants != nil) {
      // The constant is already on the GPU side: bind its place in the buffer.
      for (uint16_t k = 0; k < step.inline_len; ++k, ++inl) {
        const uint8_t slot = inl->slot;
        if (bound[slot] == constants) {
          if (bound_offset[slot] == inl->pos) {
            ++skipped;
            continue;
          }
          set_offset(encoder, sel_offset, inl->pos, slot);
        } else {
          set_buffer(encoder, sel_buffer, constants, inl->pos, slot);
          bound[slot] = constants;
          bound_inline[slot] = 0;
        }
        bound_offset[slot] = inl->pos;
        ++sent;
      }
    } else {
      for (uint16_t k = 0; k < step.inline_len; ++k, ++inl) {
        const uint8_t slot = inl->slot;
        if (bound_inline[slot] == inl->intern) {
          ++skipped;
          continue;
        }
        set_bytes(encoder, sel_bytes, bytes + inl->pos, inl->len, slot);
        bound_inline[slot] = inl->intern;
        bound[slot] = nil;
        ++sent;
      }
    }
    for (uint16_t k = 0; k < step.tg_len; ++k) {
      const auto& entry = program.tg[step.tg_pos + k];
      if (!tg_set[entry.first] || bound_tg[entry.first] != entry.second) {
        set_tg(encoder, sel_tg, entry.second, entry.first);
        tg_set[entry.first] = true;
        bound_tg[entry.first] = entry.second;
      }
    }
    if (step.indirect_buffer >= 0) {
      const int b = step.indirect_buffer;
      [encoder dispatchThreadgroupsWithIndirectBuffer:table[b]
                                indirectBufferOffset:(NSUInteger)((int64_t)step.indirect_offset + delta[b])
                               threadsPerThreadgroup:step.group];
    } else if (step.threads) {
      dispatch_threads(encoder, sel_threads, step.grid, step.group);
    } else {
      dispatch_groups(encoder, sel_groups, step.grid, step.group);
    }
  }
  stats->binds += sent;
  stats->binds_skipped += skipped;
  stats->barriers += barriers_sent;
  stats->launches += last - first;
}

void Engine::encode_passes(
    Program& program,
    id<MTLCommandBuffer> buffer,
    size_t first,
    size_t last,
    const RunOptions& options,
    size_t* fence_index,
    RunStats* stats) {
  size_t s = first;
  while (s < last) {
    bool serial = options.dispatch == kSerial;
    size_t end = last;
    const bool cut_at_barriers = options.barriers == kPasses && options.dispatch != kSerial;
    if (options.dispatch == kHybrid) {
      const uint32_t min_run = options.min_run > 1 ? (uint32_t)options.min_run : 2;
      if (program.steps[s].chain_run >= min_run) {
        serial = true;
        end = std::min(last, s + program.steps[s].chain_run);
      } else {
        end = s + 1;
        while (end < last && program.steps[end].chain_run < min_run &&
               !(cut_at_barriers && program.steps[end].barrier_min)) {
          ++end;
        }
      }
    } else if (cut_at_barriers) {
      end = s + 1;
      while (end < last && !program.steps[end].barrier_min) {
        ++end;
      }
    }
    id<MTLComputeCommandEncoder> encoder = [buffer
        computeCommandEncoderWithDispatchType:serial ? MTLDispatchTypeSerial
                                                     : MTLDispatchTypeConcurrent];
    // Metal does not order two passes over buffers it does not track, in one
    // command buffer or across two, so every pass after the first of a run
    // waits on the fence of the pass before it.  MLX does the same.
    const bool chained = end < last || *fence_index > 0 || options.split > 0;
    if (chained) {
      if (*fence_index >= fences_.size()) {
        fences_.push_back([device_ newFence]);
      }
      if (*fence_index > 0) {
        [encoder waitForFence:fences_[*fence_index - 1]];
      }
    }
    {
      AllocScope encode_allocations;
      encode(program, encoder, s, end, options, stats);
      stats->mallocs_encode += encode_allocations.stop();
    }
    if (chained) {
      [encoder updateFence:fences_[*fence_index]];
      *fence_index += 1;
    }
    [encoder endEncoding];
    stats->passes += 1;
    s = end;
  }
}

std::string Engine::bind_run(
    Program& program,
    const std::vector<const mx::array*>& inputs,
    const RunOptions& options) {
  if (flight_.active) {
    // One replay at a time: programs share the working memory.
    count_refusal("engine_busy");
    return "engine_busy";
  }
  if (stuck_ && !recovered()) {
    count_refusal("engine_stuck");
    return "engine_stuck";
  }
  // MLX's queue and ours are not ordered against each other, and a buffer an
  // input views can still be written by MLX work that carries another
  // array's event.  So everything MLX has committed must have run before our
  // launches read anything: before they are COMMITTED.  With encode_early
  // they are encoded first, under whatever MLX still has on the GPU, and the
  // drain and the sole-holder test follow (finish_early) before the commit.
  const bool early = options.encode_early && !options.skip_drain;
  if (!early && !options.skip_drain && !wait_for_mlx(wait_floor_ns_)) {
    count_refusal("mlx_busy");
    return "mlx_busy";
  }
  grow_.clear();
  grown_now_ = 0;
  grown_now_bytes_ = 0;
  std::string reason =
      can_run(program, inputs, true, !options.skip_drain && !early, !early, &grow_);
  if (!reason.empty()) {
    count_refusal(reason);
    return reason;
  }
  if (!grow_.empty() && early && !wait_for_mlx(wait_floor_ns_)) {
    // The copy below reads the input's bytes: MLX must have written them.
    count_refusal("mlx_busy");
    return "mlx_busy";
  }

  // Hold every input for the length of the run.  This is taken after the
  // sole-holder test above, which must see only the caller's references.
  held_.clear();
  for (const mx::array* array : inputs) {
    held_.push_back(*array);
  }
  fresh_.clear();
  try {
    for (int b : grow_) {
      // The input arrived in a buffer shorter than the recorded one: this run
      // works on a copy in a buffer of the recorded length.  The caller's
      // array is not written; the outputs that live in this input's buffer
      // live in the copy.
      const size_t member = (size_t)program.members[(size_t)b][0];
      const mx::array& source = held_[member];
      if (!source.is_available()) {
        const_cast<mx::array&>(source).wait();
      }
      const int64_t shift = source.offset() - program.inputs[member].offset;
      const uint64_t pad = shift < 0 ? (uint64_t)(-shift) : 0;
      const uint64_t need = program.required[(size_t)b] + (shift > 0 ? (uint64_t)shift : 0);
      grown_now_bytes_ += metal_buffer(source.buffer()).length;
      held_[member] = copy_layout(source, need, pad);
      grown_now_ += 1;
    }
    grown_inputs_ += grown_now_;
    grown_bytes_ += grown_now_bytes_;
    for (int b : program.input_buffers) {
      const int member = program.members[(size_t)b][0];
      const mx::array& array = held_[(size_t)member];
      program.table[(size_t)b] = metal_buffer(array.buffer());
      program.delta[(size_t)b] = array.offset() - program.inputs[(size_t)member].offset;
    }
    for (size_t k = 0; k < program.output_buffers.size(); ++k) {
      const int b = program.output_buffers[k];
      mx::allocator::Buffer made = mx::allocator::malloc(program.fresh_bytes[k]);
      fresh_.push_back(made);
      program.table[(size_t)b] = metal_buffer(made);
    }
    if (program.pool_generation != pool_generation_) {
      for (size_t b = 0; b < program.temp_slot.size(); ++b) {
        const int slot = program.temp_slot[b];
        if (slot >= 0) {
          if (!reserve_temp((size_t)slot, program.buffers[b].nbytes)) {
            return "failed:prepare: allocation_failed";
          }
          program.table[b] = metal_buffer(pool_[(size_t)slot].buffer);
        }
      }
      program.pool_generation = pool_generation_;
    }
    if (options.fill >= 0) {
      for (size_t b = 0; b < program.temp_slot.size(); ++b) {
        if (program.temp_slot[b] >= 0) {
          std::memset(host_bytes(program.table[b]), options.fill, program.table[b].length);
        }
      }
      for (int b : program.output_buffers) {
        std::memset(
            host_bytes(program.table[(size_t)b]), options.fill, program.table[(size_t)b].length);
      }
    }
    for (int b : program.seed_buffers) {
      const BufferSpec& seed = program.buffers[(size_t)b];
      std::memcpy(host_bytes(program.table[(size_t)b]), seed.data.data(), seed.data.size());
    }
    // An output whose buffer still held a recycled constant when the recording
    // first wrote it starts each run with that constant's bytes: MLX's donated
    // in-place updates (slice update, scatter) write only some elements and
    // leave the rest as the constant had them.  After the fill, like a seed.
    for (int b : program.seeded_outputs) {
      const BufferSpec& seed = program.buffers[(size_t)b];
      std::memcpy(host_bytes(program.table[(size_t)b]), seed.data.data(), seed.data.size());
    }
  } catch (const std::exception& error) {
    return std::string("failed:prepare: ") + error.what();
  }
  return "";
}

void Engine::commit_run(
    Program& program, const RunOptions& options, Flight* flight, RunStats* stats) {
  const size_t n_steps = program.steps.size();
  const size_t chunk = options.split > 0 ? (size_t)options.split : n_steps;
  const uint64_t encode_start = now_ns();
  size_t fence_index = 0;
  const bool hold_back = options.encode_early && !options.skip_drain;
  flight->all = options.split > 0 || hold_back ? [NSMutableArray array] : nil;
  flight->done = dispatch_semaphore_create(0);
  flight->handled = std::make_shared<std::atomic<uint64_t>>(0);
  dispatch_semaphore_t done = flight->done;
  std::shared_ptr<std::atomic<uint64_t>> handled = flight->handled;
  for (size_t s = 0; s < n_steps; s += chunk) {
    const size_t end = std::min(n_steps, s + chunk);
    id<MTLCommandBuffer> buffer = [queue_ commandBufferWithUnretainedReferences];
    encode_passes(program, buffer, s, end, options, &fence_index, stats);
    if (end == n_steps) {
      [buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
        handled->store(now_ns(), std::memory_order_release);
        dispatch_semaphore_signal(done);
      }];
    }
    if (!hold_back) {
      [buffer commit];
      if (flight->first_committed_at == 0) {
        flight->first_committed_at = now_ns();
      }
    }
    stats->command_buffers += 1;
    [flight->all addObject:buffer];
    if (flight->first == nil) {
      flight->first = buffer;
    }
    flight->last = buffer;
  }
  flight->committed_at = now_ns();
  stats->encode_ns = flight->committed_at - encode_start;
}

std::string Engine::finish_early(Program& program, Flight* flight) {
  // The launches are encoded; nothing is on the GPU yet.
  if (!wait_for_mlx(wait_floor_ns_)) {
    return "mlx_busy";
  }
  std::string reason = settle_run(program);
  if (!reason.empty()) {
    return reason;
  }
  for (id<MTLCommandBuffer> buffer in flight->all) {
    [buffer commit];
    if (flight->first_committed_at == 0) {
      flight->first_committed_at = now_ns();
    }
  }
  flight->committed_at = now_ns();
  return "";
}

std::string Engine::flight_verdict(const Flight& flight, RunStats* stats) {
  id<MTLCommandBuffer> bad = nil;
  if (flight.last.status != MTLCommandBufferStatusCompleted) {
    bad = flight.last;
  }
  for (id<MTLCommandBuffer> buffer in flight.all) {
    if (buffer.status != MTLCommandBufferStatusCompleted) {
      bad = buffer;
      break;
    }
  }
  if (bad != nil) {
    // collect() calls this on a Python thread, outside any pool: the error's
    // strings come back autoreleased.
    @autoreleasepool {
      return std::string("failed:command_buffer: ") +
          (bad.error.localizedDescription.UTF8String ?: "unknown");
    }
  }
  stats->gpu_ns = (uint64_t)((flight.last.GPUEndTime - flight.first.GPUStartTime) * 1e9);
  // The command buffer's times are seconds on the clock now_ns() reads.
  const uint64_t gpu_start = (uint64_t)(flight.first.GPUStartTime * 1e9);
  const uint64_t gpu_end = (uint64_t)(flight.last.GPUEndTime * 1e9);
  const uint64_t handled = flight.handled ? flight.handled->load(std::memory_order_acquire) : 0;
  stats->start_ns = gpu_start > flight.first_committed_at ? gpu_start - flight.first_committed_at : 0;
  stats->notify_ns = handled > gpu_end ? handled - gpu_end : 0;
  stats->wake_ns = flight.woke_at > handled && handled != 0 ? flight.woke_at - handled : 0;
  return "";
}

std::string Engine::build_outputs(Program& program, std::vector<mx::array>* outputs) {
  try {
    const size_t base = outputs->size();
    size_t next_fresh = 0;
    static thread_local std::vector<int> first_on;
    static thread_local std::vector<int> fresh_of;
    first_on.assign(program.buffers.size(), -1);
    fresh_of.assign(program.buffers.size(), -1);
    for (int b : program.output_buffers) {
      fresh_of[(size_t)b] = (int)next_fresh++;
    }
    for (size_t j = 0; j < program.outputs.size(); ++j) {
      const ArraySpec& spec = program.outputs[j];
      mx::array out(spec.shape, spec.dtype, nullptr, {});
      const int64_t itemsize = (int64_t)out.itemsize();
      if (spec.alias_input >= 0) {
        const mx::array& in = held_[(size_t)spec.alias_input];
        const int64_t relative =
            spec.offset - program.inputs[(size_t)spec.alias_input].offset;
        out.copy_shared_buffer(
            in, spec.strides, unpack_flags(spec.flags), spec.data_size, relative / itemsize);
      } else if (spec.buffer < 0) {
        out.set_data(mx::allocator::malloc(0));
      } else if (first_on[(size_t)spec.buffer] < 0) {
        const int which = fresh_of[(size_t)spec.buffer];
        out.set_data(
            fresh_[(size_t)which],
            spec.data_size,
            spec.strides,
            unpack_flags(spec.flags),
            spec.offset,
            mx::allocator::free);
        fresh_[(size_t)which] = mx::allocator::Buffer{nullptr};
        first_on[(size_t)spec.buffer] = (int)j;
      } else {
        const mx::array& owner = (*outputs)[base + (size_t)first_on[(size_t)spec.buffer]];
        out.copy_shared_buffer(
            owner,
            spec.strides,
            unpack_flags(spec.flags),
            spec.data_size,
            (spec.offset - owner.offset()) / itemsize);
      }
      out.set_status(mx::array::Status::available);
      outputs->push_back(std::move(out));
    }
  } catch (const std::exception& error) {
    return std::string("failed:outputs: ") + error.what();
  }
  return "";
}

void Engine::release_run(Program& program) {
  // Anything allocated for an output that no array took over goes back.
  for (auto& buffer : fresh_) {
    if (buffer.ptr() != nullptr) {
      mx::allocator::free(buffer);
    }
  }
  fresh_.clear();
  held_.clear();
  for (int b : program.input_buffers) {
    program.table[(size_t)b] = nil;
  }
  for (int b : program.output_buffers) {
    program.table[(size_t)b] = nil;
  }
}

void Engine::account_run(Program& program, const RunStats& stats, const std::string& failure) {
  last_ = stats;
  if (failure == "null_run") {
    return;  // measured, nothing ran: no outputs, no totals
  }
  if (failure.empty()) {
    runs_ += 1;
    program.runs += 1;
    if (stats.gpu_ns > 0 && (program.best_gpu_ns == 0 || stats.gpu_ns < program.best_gpu_ns)) {
      program.best_gpu_ns = stats.gpu_ns;
    }
    program.settle_was_futile = false;
    const uint64_t to_handler = stats.start_ns + stats.gpu_ns + stats.notify_ns;
    if (to_handler > 0) {
      program.wait_ema_ns =
          program.wait_ema_ns == 0 ? to_handler : (program.wait_ema_ns * 7 + to_handler) / 8;
    }
    total_.launches += stats.launches;
    total_.command_buffers += stats.command_buffers;
    total_.encode_ns += stats.encode_ns;
    total_.gpu_ns += stats.gpu_ns;
    total_.wall_ns += stats.wall_ns;
    total_.cpu_ns += stats.cpu_ns;
    total_.wait_ns += stats.wait_ns;
    total_.prepare_ns += stats.prepare_ns;
    total_.outputs_ns += stats.outputs_ns;
    total_.start_ns += stats.start_ns;
    total_.notify_ns += stats.notify_ns;
    total_.wake_ns += stats.wake_ns;
    total_.binds += stats.binds;
    total_.binds_skipped += stats.binds_skipped;
    total_.grown += stats.grown;
    total_.grown_bytes += stats.grown_bytes;
  } else {
    count_refusal("failed");
  }
}

std::string Engine::run(
    Program& program,
    const std::vector<const mx::array*>& inputs,
    std::vector<mx::array>* outputs,
    const RunOptions& options) {
  std::lock_guard lock(mutex_);
  const uint64_t wall_start = now_ns();
  const uint64_t cpu_start = thread_cpu_ns();
  RunStats stats;
  AllocScope all_allocations;

  std::string failure = bind_run(program, inputs, options);
  if (!failure.empty() && failure.rfind("failed:", 0) != 0) {
    return failure;  // nothing was touched
  }
  const uint64_t bound = wait_bound(program);
  id<MTLCommandBuffer> timed_out = nil;
  stats.prepare_ns = now_ns() - wall_start;
  stats.grown = grown_now_;
  stats.grown_bytes = grown_now_bytes_;

  if (failure.empty()) {
    @autoreleasepool {
      const size_t n_steps = program.steps.size();
      size_t fence_index = 0;
      if (options.null_encoder) {
        if (null_encoder_ == nil) {
          null_encoder_ = [[MTPLXNullEncoder alloc] init];
        }
        const uint64_t encode_start = now_ns();
        AllocScope encode_allocations;
        encode(program, (id<MTLComputeCommandEncoder>)null_encoder_, 0, n_steps, options, &stats);
        stats.mallocs_encode += encode_allocations.stop();
        stats.encode_ns = now_ns() - encode_start;
        failure = "null_run";
      } else if (options.profile) {
        // One command buffer per launch: the GPU's own clock per step.
        profile_.clear();
        profile_.reserve(n_steps);
        RunOptions single = options;
        single.dispatch = options.dispatch == kHybrid ? (int)kConcurrent : options.dispatch;
        single.split = 0;
        for (size_t s = 0; s < n_steps && failure.empty(); ++s) {
          const uint64_t step_start = now_ns();
          id<MTLCommandBuffer> buffer = [queue_ commandBufferWithUnretainedReferences];
          fence_index = 0;
          encode_passes(program, buffer, s, s + 1, single, &fence_index, &stats);
          stats.command_buffers += 1;
          if (!commit_and_wait(buffer, bound)) {
            timed_out = buffer;
            failure = "failed:timeout: launch " + std::to_string(s) + " did not complete";
            break;
          }
          if (buffer.status != MTLCommandBufferStatusCompleted) {
            failure = std::string("failed:command_buffer: ") +
                (buffer.error.localizedDescription.UTF8String ?: "unknown");
          }
          StepTime time;
          time.kernel = program.steps[s].kernel;
          time.gpu_ns = (uint64_t)((buffer.GPUEndTime - buffer.GPUStartTime) * 1e9);
          time.wall_ns = now_ns() - step_start;
          stats.gpu_ns += time.gpu_ns;
          profile_.push_back(time);
          if (failure.empty() && options.after_step) {
            options.after_step(s);
          }
        }
      } else if (n_steps > 0) {
        Flight flight;
        commit_run(program, options, &flight, &stats);
        if (options.encode_early && !options.skip_drain) {
          const uint64_t drain_start = now_ns();
          std::string declined = finish_early(program, &flight);
          stats.prepare_ns += now_ns() - drain_start;
          if (!declined.empty()) {
            // Encoded and never committed: nothing was touched.
            release_run(program);
            count_refusal(declined);
            return declined;
          }
        }
        // The thread sleeps here until the GPU is done: no polling.  The wait
        // is bounded: a launch that never ends must not hold the caller.
        const bool completed = wait_flight(flight, program, bound);
        flight.woke_at = now_ns();
        stats.wait_ns = flight.woke_at - flight.committed_at;
        if (!completed) {
          // Nothing of this run may be read.
          timed_out = flight.last;
          failure = "failed:timeout: no completion within " +
              std::to_string(bound / 1'000'000ull) + " ms";
        } else {
          failure = flight_verdict(flight, &stats);
        }
      }
    }
  }

  const uint64_t outputs_start = now_ns();
  if (failure.empty()) {
    failure = build_outputs(program, outputs);
  }
  if (timed_out != nil) {
    // The GPU may still be writing this run's buffers: nothing is freed or
    // reused until the command buffer has left it.
    abandon(timed_out, program);
  }
  release_run(program);

  const uint64_t end = now_ns();
  stats.outputs_ns = end - outputs_start;
  stats.wall_ns = end - wall_start;
  stats.cpu_ns = thread_cpu_ns() - cpu_start;
  stats.mallocs_total = all_allocations.stop();
  account_run(program, stats, failure);
  if (failure == "null_run") {
    return "";
  }
  return failure;
}

std::string Engine::submit(
    std::shared_ptr<Program> program,
    const std::vector<const mx::array*>& inputs,
    const RunOptions& options) {
  std::lock_guard lock(mutex_);
  const uint64_t wall_start = now_ns();
  const uint64_t cpu_start = thread_cpu_ns();
  AllocScope all_allocations;

  std::string failure = bind_run(*program, inputs, options);
  if (!failure.empty()) {
    if (failure.rfind("failed:", 0) == 0) {
      release_run(*program);
      count_refusal("failed");
    }
    return failure;
  }
  flight_ = Flight();
  flight_.bound = wait_bound(*program);
  flight_.stats.prepare_ns = now_ns() - wall_start;
  flight_.stats.grown = grown_now_;
  flight_.stats.grown_bytes = grown_now_bytes_;
  @autoreleasepool {
    if (!program->steps.empty()) {
      commit_run(*program, options, &flight_, &flight_.stats);
      if (options.encode_early && !options.skip_drain) {
        const uint64_t drain_start = now_ns();
        std::string declined = finish_early(*program, &flight_);
        flight_.stats.prepare_ns += now_ns() - drain_start;
        if (!declined.empty()) {
          release_run(*program);
          flight_ = Flight();
          count_refusal(declined);
          return declined;
        }
      }
    } else {
      flight_.committed_at = now_ns();
    }
  }
  flight_.program = std::move(program);
  flight_.active = true;
  flight_.stats.wall_ns = now_ns() - wall_start;
  flight_.stats.cpu_ns = thread_cpu_ns() - cpu_start;
  flight_.stats.mallocs_total = all_allocations.stop();
  return "";
}

std::string Engine::collect(std::vector<mx::array>* outputs, uint64_t timeout_ns) {
  std::lock_guard lock(mutex_);
  if (!flight_.active) {
    return "no_flight";
  }
  const uint64_t wall_start = now_ns();
  const uint64_t cpu_start = thread_cpu_ns();
  AllocScope all_allocations;
  const bool completed =
      flight_.done == nil || wait_flight(flight_, *flight_.program, timeout_ns);
  if (!completed) {
    flight_.stats.wall_ns += now_ns() - wall_start;
    flight_.stats.cpu_ns += thread_cpu_ns() - cpu_start;
    flight_.stats.mallocs_total += all_allocations.stop();
    return "pending";
  }
  // What the GPU took is known from the command buffers; what the caller
  // waited is only the time it spent in here.
  flight_.woke_at = now_ns();
  flight_.stats.wait_ns += flight_.woke_at - wall_start;
  Program& program = *flight_.program;
  std::string failure = flight_.done == nil ? "" : flight_verdict(flight_, &flight_.stats);
  const uint64_t outputs_start = now_ns();
  if (failure.empty()) {
    failure = build_outputs(program, outputs);
  }
  release_run(program);
  const uint64_t end = now_ns();
  flight_.stats.outputs_ns = end - outputs_start;
  flight_.stats.wall_ns += end - wall_start;
  flight_.stats.cpu_ns += thread_cpu_ns() - cpu_start;
  flight_.stats.mallocs_total += all_allocations.stop();
  const RunStats stats = flight_.stats;
  std::shared_ptr<Program> keep = std::move(flight_.program);
  flight_ = Flight();
  account_run(*keep, stats, failure);
  return failure;
}

bool Engine::wait_flight(Flight& flight, const Program& program, uint64_t timeout_ns) {
  const uint64_t entered = now_ns();
  if (spin_lead_ns_ > 0 && program.wait_ema_ns > spin_lead_ns_ && flight.handled) {
    // Sleep until shortly before the replay is due...
    const uint64_t due = flight.committed_at + program.wait_ema_ns;
    if (due > entered + spin_lead_ns_) {
      const uint64_t nap = std::min(timeout_ns, due - spin_lead_ns_ - entered);
      if (dispatch_semaphore_wait(
              flight.done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)nap)) == 0) {
        return true;
      }
    }
    // ...then watch for the completion handler's mark without sleeping, for a
    // bounded time: a sleeping thread is woken some tens of microseconds
    // after the handler has run, a watching one sees it at once.
    const uint64_t give_up = std::max(now_ns(), due) + spin_lead_ns_;
    while (now_ns() < give_up) {
      if (flight.handled->load(std::memory_order_acquire) != 0) {
        // The handler signals right after its mark; the signal is taken so
        // the semaphore stays balanced.
        while (dispatch_semaphore_wait(flight.done, DISPATCH_TIME_NOW) != 0) {
        }
        return true;
      }
#if defined(__arm64__) || defined(__aarch64__)
      __asm__ volatile("yield");
#endif
    }
  }
  const uint64_t spent = now_ns() - entered;
  const uint64_t left = timeout_ns > spent ? timeout_ns - spent : 0;
  return dispatch_semaphore_wait(
             flight.done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)left)) == 0;
}

bool Engine::flight_left_gpu(uint64_t timeout_ns) {
  if (!flight_.active || flight_.done == nil) {
    return true;
  }
  if (dispatch_semaphore_wait(
          flight_.done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)timeout_ns)) != 0) {
    return false;
  }
  dispatch_semaphore_signal(flight_.done);  // still to be collected
  return true;
}

void Engine::give_up() {
  std::lock_guard lock(mutex_);
  if (!flight_.active) {
    return;
  }
  std::shared_ptr<Program> keep = std::move(flight_.program);
  if (flight_.last != nil) {
    // As a run that passed its bound: its buffers stay parked until the
    // command buffer has left the GPU.
    abandon(flight_.last, *keep);
  }
  release_run(*keep);
  flight_ = Flight();
  count_refusal("failed");
}

Engine::Peek Engine::peek(const Program& program, int index) const {
  Peek peek;
  if (index < 0 || (size_t)index >= program.table.size()) {
    return peek;
  }
  id<MTLBuffer> buffer = program.table[(size_t)index];
  if (buffer == nil) {
    return peek;
  }
  peek.bytes = host_bytes(buffer);
  peek.length = (size_t)buffer.length;
  peek.shift = program.delta[(size_t)index];
  return peek;
}

bool Engine::commit_and_wait(id<MTLCommandBuffer> buffer, uint64_t bound_ns) {
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  [buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
    dispatch_semaphore_signal(done);
  }];
  [buffer commit];
  return dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)bound_ns)) == 0;
}

uint64_t Engine::wait_bound(const Program& program) const {
  return std::max(wait_floor_ns_, 50 * program.best_gpu_ns);
}

void Engine::abandon(id<MTLCommandBuffer> buffer, Program& program) {
  stuck_ = true;
  stuck_buffer_ = buffer;
  parked_programs_.push_back(program.shared_from_this());
  for (auto& array : held_) {
    parked_held_.push_back(std::move(array));
  }
  held_.clear();
  for (auto& made : fresh_) {
    if (made.ptr() != nullptr) {
      parked_fresh_.push_back(made);
      made = mx::allocator::Buffer{nullptr};
    }
  }
  count_refusal("timeout");
}

bool Engine::recovered() {
  if (stuck_buffer_ != nil && stuck_buffer_.status < MTLCommandBufferStatusCompleted) {
    return false;
  }
  for (auto& made : parked_fresh_) {
    mx::allocator::free(made);
  }
  parked_fresh_.clear();
  parked_held_.clear();
  parked_programs_.clear();
  stuck_buffer_ = nil;
  stuck_ = false;
  return true;
}

bool Engine::stuck() {
  std::lock_guard lock(mutex_);
  return stuck_ && !recovered();
}

mx::array Engine::clone_layout(const mx::array& source, uint64_t min_length, uint64_t pad) {
  wait_for_mlx(wait_floor_ns_);
  if (!source.is_available()) {
    const_cast<mx::array&>(source).wait();
  }
  return copy_layout(source, min_length, pad);
}

mx::array Engine::copy_layout(const mx::array& source, uint64_t min_length, uint64_t pad) {
  // The caller has made sure nothing is still writing the source.
  id<MTLBuffer> from = metal_buffer(source.buffer());
  const size_t length = (size_t)from.length;
  mx::allocator::Buffer made =
      mx::allocator::malloc(std::max<size_t>((size_t)min_length, (size_t)pad + length));
  if (made.ptr() == nullptr) {
    throw std::runtime_error("allocation_failed");
  }
  std::memcpy(host_bytes(made) + pad, host_bytes(source.buffer()), length);
  mx::array out(source.shape(), source.dtype(), nullptr, {});
  out.set_data(
      made, source.data_size(), source.strides(), source.flags(),
      source.offset() + (int64_t)pad, mx::allocator::free);
  out.set_status(mx::array::Status::available);
  return out;
}

namespace {

// The pattern of a marked tail: one 64-bit word per 8 bytes, from a counter.
inline uint64_t tail_word(uint64_t seed, uint64_t index) {
  uint64_t z = seed + 0x9E3779B97F4A7C15ull * (index + 1);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// [end, length) of the array's buffer, or an empty range when that room is
// not the array's alone to mark.
bool tail_range(const mx::array& array, uint64_t* end, uint64_t* length) {
  if (array.status() != mx::array::Status::available || array.buffer().ptr() == nullptr) {
    return false;
  }
  // Another holder of the data is a view or a parent: the bytes past this
  // array may be its.
  if (array.data_shared_ptr().use_count() > 1 || array.offset() < 0) {
    return false;
  }
  const uint64_t stop =
      (uint64_t)array.offset() + (uint64_t)array.data_size() * (uint64_t)array.itemsize();
  const uint64_t size = (uint64_t)metal_buffer(array.buffer()).length;
  if (stop >= size) {
    return false;
  }
  *end = stop;
  *length = size;
  return true;
}

}  // namespace

uint64_t Engine::mark_tail(const mx::array& array, uint64_t seed) {
  uint64_t end = 0;
  uint64_t length = 0;
  if (!tail_range(array, &end, &length)) {
    return 0;
  }
  // Nothing may be writing the buffer while its tail is written from here:
  // MLX's queue first (the array is complete, but the queue may still hold
  // work that uses the buffer as working memory).
  wait_for_mlx(wait_floor_ns_);
  uint8_t* bytes = host_bytes(array.buffer());
  for (uint64_t at = end, index = 0; at < length; at += 8, ++index) {
    const uint64_t word = tail_word(seed, index);
    std::memcpy(bytes + at, &word, (size_t)std::min<uint64_t>(8, length - at));
  }
  return length - end;
}

bool Engine::tail_intact(const mx::array& array, uint64_t seed, uint64_t marked) {
  if (marked == 0 || array.buffer().ptr() == nullptr) {
    return false;
  }
  const uint64_t end =
      (uint64_t)array.offset() + (uint64_t)array.data_size() * (uint64_t)array.itemsize();
  const uint64_t length = (uint64_t)metal_buffer(array.buffer()).length;
  if (end + marked != length) {
    return false;
  }
  const uint8_t* bytes = host_bytes(array.buffer());
  for (uint64_t at = end, index = 0; at < length; at += 8, ++index) {
    const uint64_t word = tail_word(seed, index);
    if (std::memcmp(bytes + at, &word, (size_t)std::min<uint64_t>(8, length - at)) != 0) {
      return false;
    }
  }
  return true;
}

bool Engine::relax_input(Program& program, int index) {
  if (index < 0 || (size_t)index >= program.inputs.size()) {
    return false;
  }
  const ArraySpec& spec = program.inputs[(size_t)index];
  const int b = spec.buffer;
  if (b < 0 || program.members[(size_t)b].size() != 1 || spec.offset < 0) {
    return false;
  }
  const uint64_t itemsize = (uint64_t)spec.dtype.size();
  const uint64_t end = (uint64_t)spec.offset + spec.data_size * itemsize;
  if (end >= program.required[(size_t)b]) {
    return end == program.required[(size_t)b];
  }
  // A launch bound past the array's end works in room that is not the
  // array's: that program keeps the recorded length.
  for (const ExecStep& step : program.steps) {
    for (uint32_t k = 0; k < step.bind_len; ++k) {
      const ExecBind& bind = program.binds[step.bind_pos + k];
      if ((int)bind.buffer == b && bind.offset >= end) {
        return false;
      }
    }
  }
  program.required[(size_t)b] = end;
  return true;
}

void Engine::synchronize() {
  std::lock_guard lock(mutex_);
  @autoreleasepool {
    id<MTLCommandBuffer> buffer = [queue_ commandBuffer];
    commit_and_wait(buffer, wait_floor_ns_);
  }
}

}  // namespace mtplx_engine
