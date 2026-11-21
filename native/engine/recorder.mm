// The recorder: what one thread encoded between begin and end.
//
// MLX reaches Metal through one concrete compute-encoder class per GPU.  Every
// method of that class that carries state into a launch is wrapped here.  With
// no recording open a wrapper costs one load of a global and a jump; with one
// open, only the recording thread is followed.
//
// Metal keeps bindings from one launch to the next inside an encoder, so the
// recorder mirrors the encoder's binding table and files the whole table at
// every launch.  Which of those slots the kernel really uses is decided later,
// from the kernel's reflection.

#import "engine.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace mtplx_engine {

std::string install_device_hooks(id<MTLDevice> device);  // kernels.mm

const char* const kCallNames[kCallCount] = {
    "setComputePipelineState:",
    "setBuffer:offset:atIndex:",
    "setBufferOffset:atIndex:",
    "setBuffers:offsets:withRange:",
    "setBytes:length:atIndex:",
    "setThreadgroupMemoryLength:atIndex:",
    "dispatchThreadgroups:threadsPerThreadgroup:",
    "dispatchThreads:threadsPerThreadgroup:",
    "memoryBarrierWithScope:",
    "memoryBarrierWithResources:count:",
    "endEncoding",
    "commit",
    "dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:",
};

namespace {

struct SlotState {
  uint8_t kind = 0;  // 0 nothing, 1 buffer, 2 inline bytes
  uint32_t buffer = 0;
  uint64_t offset = 0;
  uint32_t bytes_pos = 0;
  uint32_t bytes_len = 0;
};

struct Recorder {
  std::unique_ptr<Trace> trace;
  void* encoder = nullptr;  // the open encoder being followed
  SlotState slots[kMaxSlots];
  uint64_t tg_memory[kMaxSlots] = {};
  bool tg_set[kMaxSlots] = {};
  int high_slot = -1;
  void* pso = nullptr;
  bool pending_barrier = false;
  int32_t mark = -1;
  std::unordered_map<void*, uint32_t> buffer_ids;

  void reset_encoder_state() {
    for (auto& slot : slots) {
      slot = SlotState();
    }
    std::memset(tg_memory, 0, sizeof(tg_memory));
    std::memset(tg_set, 0, sizeof(tg_set));
    high_slot = -1;
    pso = nullptr;
  }

  // Follow `obj` if it is the encoder of this recording.  A second encoder
  // open at the same time cannot be ordered against the first from here.
  bool follow(void* obj) {
    if (encoder == obj) {
      return true;
    }
    if (encoder != nullptr) {
      if (trace->unsupported.empty()) {
        trace->unsupported = "two_encoders_open";
      }
      return false;
    }
    encoder = obj;
    reset_encoder_state();
    trace->mlx_encoders++;
    if (!trace->steps.empty()) {
      pending_barrier = true;
    }
    return true;
  }

  uint32_t buffer_id(id<MTLBuffer> buffer) {
    void* key = (__bridge void*)buffer;
    auto found = buffer_ids.find(key);
    if (found != buffer_ids.end()) {
      return found->second;
    }
    TraceBuffer entry;
    entry.buffer = buffer;
    entry.length = buffer.length;
    if (entry.length <= kSnapshotBytes && host_readable(buffer)) {
      const void* contents = buffer.contents;
      if (contents != nullptr) {
        entry.snapshot.resize((size_t)entry.length);
        std::memcpy(entry.snapshot.data(), contents, (size_t)entry.length);
        entry.has_snapshot = true;
      }
    }
    uint32_t id = (uint32_t)trace->buffers.size();
    trace->buffers.push_back(std::move(entry));
    buffer_ids.emplace(key, id);
    return id;
  }

  void set_buffer(id<MTLBuffer> buffer, uint64_t offset, uint64_t index) {
    if (index >= (uint64_t)kMaxSlots) {
      note_unsupported("buffer_index_out_of_range");
      return;
    }
    SlotState& slot = slots[index];
    if (buffer == nil) {
      slot = SlotState();
      return;
    }
    slot.kind = 1;
    slot.buffer = buffer_id(buffer);
    slot.offset = offset;
    if ((int)index > high_slot) {
      high_slot = (int)index;
    }
  }

  void set_buffer_offset(uint64_t offset, uint64_t index) {
    if (index < (uint64_t)kMaxSlots && slots[index].kind == 1) {
      slots[index].offset = offset;
    } else {
      note_unsupported("offset_without_buffer");
    }
  }

  void set_bytes(const void* bytes, uint64_t length, uint64_t index) {
    if (index >= (uint64_t)kMaxSlots) {
      note_unsupported("buffer_index_out_of_range");
      return;
    }
    SlotState& slot = slots[index];
    slot.kind = 2;
    slot.bytes_pos = (uint32_t)trace->bytes.size();
    slot.bytes_len = (uint32_t)length;
    const auto* src = static_cast<const uint8_t*>(bytes);
    trace->bytes.insert(trace->bytes.end(), src, src + length);
    if ((int)index > high_slot) {
      high_slot = (int)index;
    }
  }

  void dispatch(bool threads, MTLSize grid, MTLSize group, int indirect = -1, uint64_t offset = 0) {
    TraceStep step;
    step.pso = pso;
    step.indirect_buffer = indirect;
    step.indirect_offset = offset;
    step.threads = threads ? 1 : 0;
    step.grid[0] = grid.width;
    step.grid[1] = grid.height;
    step.grid[2] = grid.depth;
    step.group[0] = group.width;
    step.group[1] = group.height;
    step.group[2] = group.depth;
    step.barrier_before = pending_barrier;
    pending_barrier = false;
    step.mark = mark;
    step.slots_pos = (uint32_t)trace->slots.size();
    for (int index = 0; index <= high_slot; ++index) {
      const SlotState& slot = slots[index];
      if (slot.kind == 0) {
        continue;
      }
      TraceSlot filed;
      filed.index = (uint8_t)index;
      filed.kind = slot.kind;
      filed.buffer = slot.buffer;
      filed.offset = slot.offset;
      filed.bytes_pos = slot.bytes_pos;
      filed.bytes_len = slot.bytes_len;
      trace->slots.push_back(filed);
    }
    step.slots_len = (uint32_t)trace->slots.size() - step.slots_pos;
    step.tg_pos = (uint32_t)trace->tg_memory.size();
    for (int index = 0; index < kMaxSlots; ++index) {
      if (tg_set[index]) {
        trace->tg_memory.emplace_back((uint8_t)index, tg_memory[index]);
      }
    }
    step.tg_len = (uint32_t)trace->tg_memory.size() - step.tg_pos;
    if (pso == nullptr) {
      note_unsupported("dispatch_without_pipeline");
    }
    trace->steps.push_back(step);
  }

  void note_unsupported(const char* what) {
    if (trace->unsupported.empty()) {
      trace->unsupported = what;
    }
  }
};

// Number of recordings open in the process.  The wrappers read this first, so
// a process that is not recording pays one relaxed load per encoder call.
std::atomic<int> g_open{0};
thread_local Recorder* t_recorder = nullptr;

inline Recorder* active(void* obj) {
  if (g_open.load(std::memory_order_relaxed) == 0) {
    return nullptr;
  }
  Recorder* recorder = t_recorder;
  if (recorder == nullptr || !recorder->follow(obj)) {
    return nullptr;
  }
  return recorder;
}

inline Recorder* active_any_thread_state() {
  if (g_open.load(std::memory_order_relaxed) == 0) {
    return nullptr;
  }
  return t_recorder;
}

// ------------------------------------------------------------ the wrappers

typedef void (*SetPipelineFn)(id, SEL, id<MTLComputePipelineState>);
typedef void (*SetBufferFn)(id, SEL, id<MTLBuffer>, NSUInteger, NSUInteger);
typedef void (*SetBufferOffsetFn)(id, SEL, NSUInteger, NSUInteger);
typedef void (*SetBuffersFn)(
    id, SEL, const id<MTLBuffer> __unsafe_unretained*, const NSUInteger*, NSRange);
typedef void (*SetBytesFn)(id, SEL, const void*, NSUInteger, NSUInteger);
typedef void (*SetTgMemoryFn)(id, SEL, NSUInteger, NSUInteger);
typedef void (*DispatchFn)(id, SEL, MTLSize, MTLSize);
typedef void (*BarrierScopeFn)(id, SEL, MTLBarrierScope);
typedef void (*BarrierResourcesFn)(
    id, SEL, const id<MTLResource> __unsafe_unretained*, NSUInteger);
typedef void (*VoidFn)(id, SEL);

SetPipelineFn o_set_pipeline = nullptr;
SetBufferFn o_set_buffer = nullptr;
SetBufferOffsetFn o_set_buffer_offset = nullptr;
SetBuffersFn o_set_buffers = nullptr;
SetBytesFn o_set_bytes = nullptr;
SetTgMemoryFn o_set_tg_memory = nullptr;
DispatchFn o_dispatch_groups = nullptr;
DispatchFn o_dispatch_threads = nullptr;
BarrierScopeFn o_barrier_scope = nullptr;
BarrierResourcesFn o_barrier_resources = nullptr;
VoidFn o_end_encoding = nullptr;
VoidFn o_commit = nullptr;

void h_set_pipeline(id obj, SEL sel, id<MTLComputePipelineState> state) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kSetPipeline]++;
    r->pso = (__bridge void*)state;
  }
  o_set_pipeline(obj, sel, state);
}

void h_set_buffer(id obj, SEL sel, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger index) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kSetBuffer]++;
    r->set_buffer(buffer, offset, index);
  }
  o_set_buffer(obj, sel, buffer, offset, index);
}

void h_set_buffer_offset(id obj, SEL sel, NSUInteger offset, NSUInteger index) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kSetBufferOffset]++;
    r->set_buffer_offset(offset, index);
  }
  o_set_buffer_offset(obj, sel, offset, index);
}

void h_set_buffers(
    id obj,
    SEL sel,
    const id<MTLBuffer> __unsafe_unretained* buffers,
    const NSUInteger* offsets,
    NSRange range) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kSetBuffers]++;
    for (NSUInteger i = 0; i < range.length; ++i) {
      r->set_buffer(buffers[i], offsets[i], range.location + i);
    }
  }
  o_set_buffers(obj, sel, buffers, offsets, range);
}

void h_set_bytes(id obj, SEL sel, const void* bytes, NSUInteger length, NSUInteger index) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kSetBytes]++;
    r->set_bytes(bytes, length, index);
  }
  o_set_bytes(obj, sel, bytes, length, index);
}

void h_set_tg_memory(id obj, SEL sel, NSUInteger length, NSUInteger index) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kSetThreadgroupMemory]++;
    if (index < (NSUInteger)kMaxSlots) {
      r->tg_memory[index] = length;
      r->tg_set[index] = true;
    } else {
      r->note_unsupported("threadgroup_index_out_of_range");
    }
  }
  o_set_tg_memory(obj, sel, length, index);
}

void h_dispatch_groups(id obj, SEL sel, MTLSize grid, MTLSize group) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kDispatchThreadgroups]++;
    r->dispatch(false, grid, group);
  }
  o_dispatch_groups(obj, sel, grid, group);
}

void h_dispatch_threads(id obj, SEL sel, MTLSize grid, MTLSize group) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kDispatchThreads]++;
    r->dispatch(true, grid, group);
  }
  o_dispatch_threads(obj, sel, grid, group);
}

void h_barrier_scope(id obj, SEL sel, MTLBarrierScope scope) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kBarrierScope]++;
    r->pending_barrier = true;
  }
  o_barrier_scope(obj, sel, scope);
}

void h_barrier_resources(
    id obj,
    SEL sel,
    const id<MTLResource> __unsafe_unretained* resources,
    NSUInteger count) {
  if (Recorder* r = active((__bridge void*)obj)) {
    // Replayed as a barrier over all buffers: never weaker than this one.
    r->trace->calls[kBarrierResources]++;
    r->pending_barrier = true;
  }
  o_barrier_resources(obj, sel, resources, count);
}

void h_end_encoding(id obj, SEL sel) {
  if (Recorder* r = active_any_thread_state()) {
    if (r->encoder == (__bridge void*)obj) {
      r->trace->calls[kEndEncoding]++;
      r->encoder = nullptr;
      r->pending_barrier = !r->trace->steps.empty();
    }
  }
  o_end_encoding(obj, sel);
}

// MLX's committed and completed command buffers, counted.  Never destroyed: a
// completion handler may run while the process exits.
struct MlxFlight {
  std::mutex mutex;
  std::condition_variable done;
  uint64_t committed = 0;
  uint64_t completed = 0;
  std::vector<void*> own_queues;
};

MlxFlight& mlx_flight() {
  static MlxFlight* flight = new MlxFlight();
  return *flight;
}

void h_commit(id obj, SEL sel) {
  if (Recorder* r = active_any_thread_state()) {
    r->trace->calls[kCommit]++;
    r->trace->mlx_command_buffers++;
  }
  id<MTLCommandBuffer> buffer = (id<MTLCommandBuffer>)obj;
  void* queue = (__bridge void*)buffer.commandQueue;
  MlxFlight& flight = mlx_flight();
  bool own = false;
  uint64_t serial = 0;
  {
    std::lock_guard lock(flight.mutex);
    for (void* mine : flight.own_queues) {
      own = own || mine == queue;
    }
    if (!own) {
      serial = ++flight.committed;
    }
  }
  if (!own) {
    // A queue completes its buffers in the order they were committed.
    [buffer addCompletedHandler:^(id<MTLCommandBuffer>) {
      MlxFlight& state = mlx_flight();
      {
        std::lock_guard lock(state.mutex);
        if (serial > state.completed) {
          state.completed = serial;
        }
      }
      state.done.notify_all();
    }];
  }
  o_commit(obj, sel);
}

// The forms that also carry an attribute stride.  With the static stride they
// are the plain calls under another name (Metal's validation layer routes the
// plain calls through them), so they are recorded as the plain calls; with a
// real stride a replay cannot reproduce them.
typedef void (*SetBufferStrideFn)(id, SEL, id<MTLBuffer>, NSUInteger, NSUInteger, NSUInteger);
typedef void (*SetBytesStrideFn)(id, SEL, const void*, NSUInteger, NSUInteger, NSUInteger);
typedef void (*SetOffsetStrideFn)(id, SEL, NSUInteger, NSUInteger, NSUInteger);
SetBufferStrideFn o_set_buffer_stride = nullptr;
SetBytesStrideFn o_set_bytes_stride = nullptr;
SetOffsetStrideFn o_set_offset_stride = nullptr;

void h_set_buffer_stride(
    id obj, SEL sel, id<MTLBuffer> buffer, NSUInteger offset, NSUInteger stride, NSUInteger index) {
  if (Recorder* r = active((__bridge void*)obj)) {
    if (stride == MTLAttributeStrideStatic) {
      r->set_buffer(buffer, offset, index);
    } else {
      r->note_unsupported(sel_getName(sel));
    }
  }
  o_set_buffer_stride(obj, sel, buffer, offset, stride, index);
}

void h_set_bytes_stride(
    id obj, SEL sel, const void* bytes, NSUInteger length, NSUInteger stride, NSUInteger index) {
  if (Recorder* r = active((__bridge void*)obj)) {
    if (stride == MTLAttributeStrideStatic) {
      r->set_bytes(bytes, length, index);
    } else {
      r->note_unsupported(sel_getName(sel));
    }
  }
  o_set_bytes_stride(obj, sel, bytes, length, stride, index);
}

void h_set_offset_stride(id obj, SEL sel, NSUInteger offset, NSUInteger stride, NSUInteger index) {
  if (Recorder* r = active((__bridge void*)obj)) {
    if (stride == MTLAttributeStrideStatic) {
      r->set_buffer_offset(offset, index);
    } else {
      r->note_unsupported(sel_getName(sel));
    }
  }
  o_set_offset_stride(obj, sel, offset, stride, index);
}

// Calls a replay cannot reproduce.  Seeing one during a recording marks the
// trace, and the build refuses it; the call itself goes through untouched.
#define MTPLX_UNSUPPORTED(NAME, ORIG, SIGNATURE, ARGS)                       \
  IMP ORIG = nullptr;                                                        \
  void NAME SIGNATURE {                                                      \
    if (Recorder* r = active((__bridge void*)obj)) {                         \
      r->note_unsupported(sel_getName(sel));                                 \
    }                                                                        \
    ((void (*) SIGNATURE)ORIG) ARGS;                                         \
  }

using IndirectFn = void (*)(id, SEL, id<MTLBuffer>, NSUInteger, MTLSize);
IndirectFn o_indirect_dispatch = nullptr;
void h_indirect_dispatch(id obj, SEL sel, id<MTLBuffer> buffer, NSUInteger offset, MTLSize group) {
  if (Recorder* r = active((__bridge void*)obj)) {
    r->trace->calls[kDispatchIndirect]++;
    if (buffer == nil) {
      r->note_unsupported("indirect_without_buffer");
    } else {
      r->dispatch(false, MTLSizeMake(0, 0, 0), group, (int)r->buffer_id(buffer), offset);
    }
  }
  o_indirect_dispatch(obj, sel, buffer, offset, group);
}

MTPLX_UNSUPPORTED(
    h_execute_range,
    o_execute_range,
    (id obj, SEL sel, id<MTLIndirectCommandBuffer> commands, NSRange range),
    (obj, sel, commands, range))
MTPLX_UNSUPPORTED(
    h_execute_indirect,
    o_execute_indirect,
    (id obj, SEL sel, id<MTLIndirectCommandBuffer> commands, id<MTLBuffer> buffer, NSUInteger offset),
    (obj, sel, commands, buffer, offset))
MTPLX_UNSUPPORTED(
    h_set_texture,
    o_set_texture,
    (id obj, SEL sel, id<MTLTexture> texture, NSUInteger index),
    (obj, sel, texture, index))

#undef MTPLX_UNSUPPORTED

// Stores the original first, then swaps, so a wrapper never runs with a null
// original even when another thread is encoding during the install.
template <typename Fn>
bool wrap(Class cls, SEL sel, Fn* original, IMP replacement) {
  Method method = class_getInstanceMethod(cls, sel);
  if (method == nullptr) {
    return false;
  }
  *original = (Fn)method_getImplementation(method);
  class_replaceMethod(cls, sel, replacement, method_getTypeEncoding(method));
  return true;
}

std::mutex g_install_mutex;
bool g_installed = false;
Class g_encoder_class = Nil;
Class g_command_buffer_class = Nil;
EncoderImps g_imps;

std::string install_encoder_hooks(Class cls) {
  bool ok = true;
  ok &= wrap(cls, @selector(setComputePipelineState:), &o_set_pipeline, (IMP)h_set_pipeline);
  ok &= wrap(cls, @selector(setBuffer:offset:atIndex:), &o_set_buffer, (IMP)h_set_buffer);
  ok &= wrap(
      cls, @selector(setBufferOffset:atIndex:), &o_set_buffer_offset, (IMP)h_set_buffer_offset);
  ok &= wrap(cls, @selector(setBuffers:offsets:withRange:), &o_set_buffers, (IMP)h_set_buffers);
  ok &= wrap(cls, @selector(setBytes:length:atIndex:), &o_set_bytes, (IMP)h_set_bytes);
  ok &= wrap(
      cls, @selector(setThreadgroupMemoryLength:atIndex:), &o_set_tg_memory, (IMP)h_set_tg_memory);
  ok &= wrap(
      cls,
      @selector(dispatchThreadgroups:threadsPerThreadgroup:),
      &o_dispatch_groups,
      (IMP)h_dispatch_groups);
  ok &= wrap(
      cls,
      @selector(dispatchThreads:threadsPerThreadgroup:),
      &o_dispatch_threads,
      (IMP)h_dispatch_threads);
  ok &= wrap(cls, @selector(memoryBarrierWithScope:), &o_barrier_scope, (IMP)h_barrier_scope);
  ok &= wrap(
      cls,
      @selector(memoryBarrierWithResources:count:),
      &o_barrier_resources,
      (IMP)h_barrier_resources);
  ok &= wrap(cls, @selector(endEncoding), &o_end_encoding, (IMP)h_end_encoding);
  if (!ok) {
    return "encoder_class_lacks_methods";
  }
  // Optional: not every system has every one of these.
  wrap(
      cls,
      @selector(dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:),
      &o_indirect_dispatch,
      (IMP)h_indirect_dispatch);
  wrap(
      cls,
      @selector(setBuffer:offset:attributeStride:atIndex:),
      &o_set_buffer_stride,
      (IMP)h_set_buffer_stride);
  wrap(
      cls,
      @selector(setBytes:length:attributeStride:atIndex:),
      &o_set_bytes_stride,
      (IMP)h_set_bytes_stride);
  wrap(
      cls,
      @selector(setBufferOffset:attributeStride:atIndex:),
      &o_set_offset_stride,
      (IMP)h_set_offset_stride);
  wrap(cls, @selector(executeCommandsInBuffer:withRange:), &o_execute_range, (IMP)h_execute_range);
  wrap(
      cls,
      @selector(executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:),
      &o_execute_indirect,
      (IMP)h_execute_indirect);
  wrap(cls, @selector(setTexture:atIndex:), &o_set_texture, (IMP)h_set_texture);

  g_imps.cls = cls;
  g_imps.set_pipeline = (IMP)o_set_pipeline;
  g_imps.set_buffer = (IMP)o_set_buffer;
  g_imps.set_buffer_offset = (IMP)o_set_buffer_offset;
  g_imps.set_bytes = (IMP)o_set_bytes;
  g_imps.set_tg_memory = (IMP)o_set_tg_memory;
  g_imps.dispatch_groups = (IMP)o_dispatch_groups;
  g_imps.dispatch_threads = (IMP)o_dispatch_threads;
  g_imps.barrier_scope = (IMP)o_barrier_scope;
  g_imps.barrier_resources = (IMP)o_barrier_resources;
  return "";
}

}  // namespace

std::string install_hooks(id<MTLDevice> device) {
  std::lock_guard lock(g_install_mutex);
  if (g_installed) {
    return "";
  }
  if (device == nil) {
    return "no_metal_device";
  }
  @autoreleasepool {
    // Ask the device for an encoder exactly as MLX does and take its class.
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (queue == nil) {
      return "no_command_queue";
    }
    id<MTLCommandBuffer> buffer = [queue commandBufferWithUnretainedReferences];
    id<MTLComputeCommandEncoder> concurrent =
        [buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    Class concurrent_class = object_getClass(concurrent);
    [concurrent endEncoding];
    id<MTLComputeCommandEncoder> serial =
        [buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    Class serial_class = object_getClass(serial);
    [serial endEncoding];
    Class buffer_class = object_getClass(buffer);
    [buffer commit];
    if (concurrent_class == Nil || buffer_class == Nil) {
      return "no_encoder_class";
    }
    if (serial_class != concurrent_class) {
      return "encoder_class_differs_by_dispatch_type";
    }
    std::string problem = install_device_hooks(device);
    if (!problem.empty()) {
      return problem;
    }
    problem = install_encoder_hooks(concurrent_class);
    if (!problem.empty()) {
      return problem;
    }
    if (!wrap(buffer_class, @selector(commit), &o_commit, (IMP)h_commit)) {
      return "command_buffer_class_lacks_commit";
    }
    g_encoder_class = concurrent_class;
    g_command_buffer_class = buffer_class;
    g_installed = true;
  }
  return "";
}

bool hooks_installed() {
  std::lock_guard lock(g_install_mutex);
  return g_installed;
}

std::string encoder_class_name() {
  return g_encoder_class ? class_getName(g_encoder_class) : "";
}

const EncoderImps& original_encoder_imps() {
  return g_imps;
}

std::string record_begin(const std::string& label) {
  if (!hooks_installed()) {
    return "hooks_not_installed";
  }
  if (t_recorder != nullptr) {
    return "already_recording";
  }
  auto* recorder = new Recorder();
  recorder->trace = std::make_unique<Trace>();
  recorder->trace->label = label;
  recorder->trace->steps.reserve(4096);
  recorder->trace->slots.reserve(32768);
  recorder->trace->bytes.reserve(65536);
  t_recorder = recorder;
  g_open.fetch_add(1, std::memory_order_release);
  return "";
}

std::string record_mark(const std::string& label) {
  Recorder* recorder = t_recorder;
  if (recorder == nullptr) {
    return "not_recording";
  }
  recorder->mark = (int32_t)recorder->trace->marks.size();
  recorder->trace->marks.push_back(label);
  return "";
}

std::string record_end(std::unique_ptr<Trace>* out) {
  Recorder* recorder = t_recorder;
  if (recorder == nullptr) {
    return "not_recording";
  }
  g_open.fetch_sub(1, std::memory_order_release);
  t_recorder = nullptr;
  // Once everything the recording launched has run, a buffer nothing in the
  // recording writes must still hold what it held at its first bind.
  if (wait_for_mlx(2'000'000'000ull)) {
    for (TraceBuffer& buffer : recorder->trace->buffers) {
      if (buffer.has_snapshot && buffer.buffer.contents != nullptr) {
        buffer.changed = std::memcmp(
            buffer.snapshot.data(), buffer.buffer.contents, buffer.snapshot.size()) != 0;
      }
    }
  }
  *out = std::move(recorder->trace);
  delete recorder;
  return "";
}

bool wait_for_mlx(uint64_t timeout_ns) {
  MlxFlight& flight = mlx_flight();
  std::unique_lock lock(flight.mutex);
  const uint64_t target = flight.committed;
  if (flight.completed >= target) {
    return true;
  }
  return flight.done.wait_for(lock, std::chrono::nanoseconds(timeout_ns), [&flight, target] {
    return flight.completed >= target;
  });
}

void register_own_queue(void* queue) {
  MlxFlight& flight = mlx_flight();
  std::lock_guard lock(flight.mutex);
  flight.own_queues.push_back(queue);
}

bool recording_on_this_thread() {
  return t_recorder != nullptr;
}

}  // namespace mtplx_engine
