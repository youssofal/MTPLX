// Which pipeline is which kernel.
//
// MLX builds a compute pipeline from a Metal function and keeps only the
// pipeline.  A replay needs two more facts per pipeline: the function's name
// (for the census and the profile) and, for every buffer argument, whether the
// kernel can write through it (to tell an input from an output).  Both come
// from the moment the pipeline is created, so the device's pipeline-creation
// methods are wrapped: the wrapper asks Metal for the same pipeline with
// reflection switched on and files what it learns.  The pipeline MLX receives
// is the one Metal built; nothing about the kernel changes.

#import "engine.h"

#import <objc/message.h>
#import <objc/runtime.h>

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace mtplx_engine {

namespace {

std::shared_mutex g_mutex;
std::unordered_map<void*, KernelInfo*> g_by_pso;  // entries are never freed
std::vector<KernelInfo*> g_by_id;
std::atomic<uint64_t> g_pipelines_seen{0};

const MTLPipelineOption kReflect =
    MTLPipelineOptionBindingInfo | MTLPipelineOptionBufferTypeInfo;

typedef id (*NewWithFunctionFn)(id, SEL, id<MTLFunction>, NSError* __autoreleasing*)
    __attribute__((ns_returns_retained));
typedef id (*NewWithFunctionOptionsFn)(
    id,
    SEL,
    id<MTLFunction>,
    MTLPipelineOption,
    MTLComputePipelineReflection* __autoreleasing*,
    NSError* __autoreleasing*) __attribute__((ns_returns_retained));
typedef id (*NewWithDescriptorFn)(
    id,
    SEL,
    MTLComputePipelineDescriptor*,
    MTLPipelineOption,
    MTLComputePipelineReflection* __autoreleasing*,
    NSError* __autoreleasing*) __attribute__((ns_returns_retained));

NewWithFunctionFn g_orig_with_function = nullptr;
NewWithFunctionOptionsFn g_orig_with_function_options = nullptr;
NewWithDescriptorFn g_orig_with_descriptor = nullptr;
Class g_device_class = Nil;

void read_reflection(KernelInfo* info, MTLComputePipelineReflection* reflection) {
  if (reflection == nil) {
    return;
  }
  for (id<MTLBinding> binding in reflection.bindings) {
    if (!binding.isUsed) {
      continue;
    }
    NSUInteger index = binding.index;
    switch (binding.type) {
      case MTLBindingTypeBuffer:
        if (index >= (NSUInteger)kMaxSlots) {
          info->problem = "buffer_index_out_of_range";
          break;
        }
        info->kind[index] = 1;
        info->access[index] = binding.access == MTLBindingAccessReadOnly ? 'r' : 'w';
        break;
      case MTLBindingTypeThreadgroupMemory:
        if (index >= (NSUInteger)kMaxSlots) {
          info->problem = "threadgroup_index_out_of_range";
          break;
        }
        info->kind[index] = 2;
        break;
      default:
        // Textures, samplers, function tables, acceleration structures: MLX's
        // kernels use none, and a replay does not carry them.
        info->problem = "non_buffer_argument";
        break;
    }
  }
  info->reflected = true;
}

KernelInfo* file_pipeline(
    id<MTLComputePipelineState> pso,
    NSString* name,
    MTLComputePipelineReflection* reflection) {
  void* key = (__bridge void*)pso;
  std::unique_lock lock(g_mutex);
  auto found = g_by_pso.find(key);
  if (found != g_by_pso.end()) {
    KernelInfo* info = found->second;
    if (!info->reflected && reflection != nil) {
      read_reflection(info, reflection);
    }
    if (info->name.empty() && name != nil) {
      info->name = name.UTF8String;
    }
    return info;
  }
  auto* info = new KernelInfo();
  info->handle = (int)g_by_id.size();
  info->pso = pso;
  if (name != nil) {
    info->name = name.UTF8String;
  }
  read_reflection(info, reflection);
  g_by_pso.emplace(key, info);
  g_by_id.push_back(info);
  return info;
}

__attribute__((ns_returns_retained)) id hook_with_function(
    id device,
    SEL,
    id<MTLFunction> function,
    NSError* __autoreleasing* error) {
  MTLComputePipelineReflection* reflection = nil;
  id pso = g_orig_with_function_options(
      device,
      @selector(newComputePipelineStateWithFunction:options:reflection:error:),
      function,
      kReflect,
      &reflection,
      error);
  if (pso != nil) {
    g_pipelines_seen.fetch_add(1, std::memory_order_relaxed);
    file_pipeline(pso, function.name, reflection);
  }
  return pso;
}

__attribute__((ns_returns_retained)) id hook_with_function_options(
    id device,
    SEL sel,
    id<MTLFunction> function,
    MTLPipelineOption options,
    MTLComputePipelineReflection* __autoreleasing* reflection_out,
    NSError* __autoreleasing* error) {
  MTLComputePipelineReflection* reflection = nil;
  id pso = g_orig_with_function_options(
      device, sel, function, options | kReflect, &reflection, error);
  if (pso != nil) {
    g_pipelines_seen.fetch_add(1, std::memory_order_relaxed);
    file_pipeline(pso, function.name, reflection);
  }
  if (reflection_out != nullptr) {
    *reflection_out = reflection;
  }
  return pso;
}

__attribute__((ns_returns_retained)) id hook_with_descriptor(
    id device,
    SEL sel,
    MTLComputePipelineDescriptor* descriptor,
    MTLPipelineOption options,
    MTLComputePipelineReflection* __autoreleasing* reflection_out,
    NSError* __autoreleasing* error) {
  MTLComputePipelineReflection* reflection = nil;
  id pso = g_orig_with_descriptor(
      device, sel, descriptor, options | kReflect, &reflection, error);
  if (pso != nil) {
    g_pipelines_seen.fetch_add(1, std::memory_order_relaxed);
    file_pipeline(pso, descriptor.computeFunction.name, reflection);
  }
  if (reflection_out != nullptr) {
    *reflection_out = reflection;
  }
  return pso;
}

IMP swap(Class cls, SEL sel, IMP replacement) {
  Method method = class_getInstanceMethod(cls, sel);
  if (method == nullptr) {
    return nullptr;
  }
  IMP original = method_getImplementation(method);
  class_replaceMethod(cls, sel, replacement, method_getTypeEncoding(method));
  return original;
}

}  // namespace

std::string install_device_hooks(id<MTLDevice> device) {
  static std::mutex once;
  std::lock_guard lock(once);
  Class cls = object_getClass(device);
  if (g_device_class == cls) {
    return "";
  }
  if (g_device_class != Nil) {
    return "second_device_class";
  }
  SEL with_function = @selector(newComputePipelineStateWithFunction:error:);
  SEL with_function_options =
      @selector(newComputePipelineStateWithFunction:options:reflection:error:);
  SEL with_descriptor =
      @selector(newComputePipelineStateWithDescriptor:options:reflection:error:);
  if (!class_getInstanceMethod(cls, with_function) ||
      !class_getInstanceMethod(cls, with_function_options) ||
      !class_getInstanceMethod(cls, with_descriptor)) {
    return "device_class_lacks_pipeline_methods";
  }
  // The originals must be in place before any wrapper can run.
  g_orig_with_function_options = (NewWithFunctionOptionsFn)method_getImplementation(
      class_getInstanceMethod(cls, with_function_options));
  g_orig_with_descriptor = (NewWithDescriptorFn)method_getImplementation(
      class_getInstanceMethod(cls, with_descriptor));
  g_orig_with_function = (NewWithFunctionFn)method_getImplementation(
      class_getInstanceMethod(cls, with_function));
  swap(cls, with_function_options, (IMP)hook_with_function_options);
  swap(cls, with_descriptor, (IMP)hook_with_descriptor);
  swap(cls, with_function, (IMP)hook_with_function);
  g_device_class = cls;
  return "";
}

std::string device_class_name() {
  return g_device_class ? class_getName(g_device_class) : "";
}

KernelInfo* kernel_for_pipeline(void* pso) {
  {
    std::shared_lock lock(g_mutex);
    auto found = g_by_pso.find(pso);
    if (found != g_by_pso.end()) {
      return found->second;
    }
  }
  // Created before the hooks.  Newer systems keep reflection on the pipeline
  // when it was built with it; otherwise the arguments stay unknown and a
  // program that launches this kernel is refused.  Reached from Python (a
  // trace's export), which has no autorelease pool: the label and the
  // reflection come back autoreleased.
  @autoreleasepool {
    id<MTLComputePipelineState> state = (__bridge id<MTLComputePipelineState>)pso;
    MTLComputePipelineReflection* reflection = nil;
    if (@available(macOS 26.0, *)) {
#if defined(__MAC_26_0) && __MAC_OS_X_VERSION_MAX_ALLOWED >= __MAC_26_0
      reflection = state.reflection;
#else
      // Built with an SDK older than macOS 26 (a source install with older
      // Xcode tools): the property is not declared there, so the same getter
      // is called by name, and only where the running system has it.
      SEL getter = NSSelectorFromString(@"reflection");
      if ([(id)state respondsToSelector:getter]) {
        reflection = ((MTLComputePipelineReflection * (*)(id, SEL)) objc_msgSend)((id)state, getter);
      }
#endif
    }
    return file_pipeline(state, state.label, reflection);
  }
}

KernelInfo* kernel_by_id(int kernel_id) {
  std::shared_lock lock(g_mutex);
  if (kernel_id < 0 || kernel_id >= (int)g_by_id.size()) {
    return nullptr;
  }
  return g_by_id[(size_t)kernel_id];
}

int kernel_count() {
  std::shared_lock lock(g_mutex);
  return (int)g_by_id.size();
}

uint64_t pipelines_seen() {
  return g_pipelines_seen.load(std::memory_order_relaxed);
}

}  // namespace mtplx_engine
