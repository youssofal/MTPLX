// The Python module mtplx._engine.
//
// Thin by design: Python owns the policy (what to record, how a trace becomes
// a program, when to fall back) and this file moves data across.  The module
// shares MLX's array type with mlx.core, so it has to be built with the same
// nanobind MLX was built with and in MLX's type domain.

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#import "../engine/executor.h"
#include "../engine/history_rows.h"

#include <mach-o/dyld.h>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

#include "mlx/memory.h"
#include "mlx/transforms.h"

namespace nb = nanobind;
using namespace nb::literals;
using namespace mtplx_engine;

#ifndef MTPLX_ENGINE_VERSION
#define MTPLX_ENGINE_VERSION "0.0.0"
#endif
#ifndef MTPLX_ENGINE_MLX_VERSION
#define MTPLX_ENGINE_MLX_VERSION "unknown"
#endif
#ifndef MTPLX_ENGINE_NANOBIND_VERSION
#define MTPLX_ENGINE_NANOBIND_VERSION "unknown"
#endif

namespace {

// Bumped whenever the Python side has to change with this file.
constexpr int kNativeApi = 1;

struct PyTrace {
  std::unique_ptr<Trace> trace;
};

struct PyProgram {
  std::shared_ptr<Program> program;
};

// Keeps the Metal buffers of described arrays alive (not the arrays), so a
// buffer address taken before a recording cannot be handed to another buffer.
struct PyHeld {
  std::vector<id<MTLBuffer>> buffers;
};

struct PyEngine {
  std::unique_ptr<Engine> engine;
  std::string label;
  bool validate = false;
};

const mx::array* as_array(nb::handle handle) {
  if (!nb::isinstance<mx::array>(handle)) {
    throw nb::type_error("expected an mlx.core.array");
  }
  return nb::inst_ptr<mx::array>(handle);
}

uint8_t pack_flags(const mx::array::Flags& flags) {
  return (uint8_t)((flags.contiguous ? 1 : 0) | (flags.row_contiguous ? 2 : 0) |
      (flags.col_contiguous ? 4 : 0));
}

template <typename Vec>
nb::tuple to_tuple(const Vec& values) {
  nb::list items;
  for (auto value : values) {
    items.append((int64_t)value);
  }
  return nb::tuple(items);
}

nb::tuple describe_one(const mx::array& array, PyHeld* held) {
  if (array.status() == mx::array::Status::unscheduled) {
    throw nb::value_error("describe: the array has not been evaluated");
  }
  const void* pointer = array.buffer().ptr();
  uint64_t length = 0;
  if (pointer != nullptr) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)const_cast<void*>(pointer);
    length = buffer.length;
    if (held != nullptr) {
      held->buffers.push_back(buffer);
    }
  }
  return nb::make_tuple(
      (uint64_t)(uintptr_t)pointer,
      (int64_t)array.offset(),
      to_tuple(array.shape()),
      dtype_name(array.dtype()),
      to_tuple(array.strides()),
      (uint64_t)array.data_size(),
      (int)pack_flags(array.flags()),
      (uint64_t)array.nbytes(),
      (int)array.itemsize(),
      length);
}

std::string os_version() {
  NSOperatingSystemVersion v;
  @autoreleasepool {  // a Python thread has none of its own
    v = NSProcessInfo.processInfo.operatingSystemVersion;
  }
  return std::to_string(v.majorVersion) + "." + std::to_string(v.minorVersion) + "." +
      std::to_string(v.patchVersion);
}

nb::dict stats_dict(const RunStats& stats) {
  nb::dict out;
  out["launches"] = stats.launches;
  out["command_buffers"] = stats.command_buffers;
  out["encode_ns"] = stats.encode_ns;
  out["gpu_ns"] = stats.gpu_ns;
  out["wall_ns"] = stats.wall_ns;
  out["cpu_ns"] = stats.cpu_ns;
  out["wait_ns"] = stats.wait_ns;
  out["prepare_ns"] = stats.prepare_ns;
  out["outputs_ns"] = stats.outputs_ns;
  out["start_ns"] = stats.start_ns;
  out["notify_ns"] = stats.notify_ns;
  out["wake_ns"] = stats.wake_ns;
  out["binds"] = stats.binds;
  out["binds_skipped"] = stats.binds_skipped;
  out["passes"] = stats.passes;
  out["barriers"] = stats.barriers;
  out["grown"] = stats.grown;
  out["grown_bytes"] = stats.grown_bytes;
  out["mallocs_encode"] = stats.mallocs_encode;
  out["mallocs_total"] = stats.mallocs_total;
  return out;
}

nb::dict export_trace(const Trace& trace, bool include_indirect) {
  nb::list steps;
  for (const TraceStep& step : trace.steps) {
    KernelInfo* kernel = kernel_for_pipeline(step.pso);
    const TraceSlot* by_index[kMaxSlots] = {};
    for (uint32_t k = 0; k < step.slots_len; ++k) {
      const TraceSlot& slot = trace.slots[step.slots_pos + k];
      by_index[slot.index] = &slot;
    }
    nb::list bindings;
    nb::list inline_bytes;
    nb::list missing;
    for (int index = 0; index < kMaxSlots; ++index) {
      const TraceSlot* slot = by_index[index];
      if (kernel->reflected) {
        if (kernel->kind[index] != 1) {
          continue;
        }
        if (slot == nullptr) {
          missing.append(index);
          continue;
        }
      } else if (slot == nullptr) {
        continue;
      }
      if (slot->kind == 1) {
        const char access[2] = {kernel->reflected ? kernel->access[index] : '?', 0};
        bindings.append(nb::make_tuple(index, slot->buffer, slot->offset, access));
      } else {
        inline_bytes.append(nb::make_tuple(
            index,
            nb::bytes((const char*)trace.bytes.data() + slot->bytes_pos, slot->bytes_len)));
      }
    }
    nb::list tg;
    for (uint32_t k = 0; k < step.tg_len; ++k) {
      const auto& entry = trace.tg_memory[step.tg_pos + k];
      if (!kernel->reflected || kernel->kind[entry.first] == 2) {
        tg.append(nb::make_tuple((int)entry.first, entry.second));
      }
    }
    std::string name = kernel->name.empty() ? "pipeline_" + std::to_string(kernel->handle) : kernel->name;
    nb::object indirect = nb::none();
    if (step.indirect_buffer >= 0) {
      indirect = nb::make_tuple(step.indirect_buffer, step.indirect_offset);
    }
    steps.append(nb::make_tuple(
        kernel->handle,
        name,
        kernel->reflected,
        kernel->problem,
        bindings,
        inline_bytes,
        tg,
        step.threads != 0,
        nb::make_tuple(step.grid[0], step.grid[1], step.grid[2]),
        nb::make_tuple(step.group[0], step.group[1], step.group[2]),
        step.barrier_before,
        step.mark >= 0 ? trace.marks[(size_t)step.mark] : std::string(),
        missing, indirect));
  }
  nb::list buffers;
  for (size_t b = 0; b < trace.buffers.size(); ++b) {
    const TraceBuffer& buffer = trace.buffers[b];
    nb::object snapshot = nb::none();
    if (buffer.has_snapshot) {
      snapshot = nb::bytes((const char*)buffer.snapshot.data(), buffer.snapshot.size());
    }
    buffers.append(nb::make_tuple(
        (int)b,
        (uint64_t)(uintptr_t)(__bridge void*)buffer.buffer,
        buffer.length,
        snapshot,
        buffer.changed));
  }
  nb::dict out;
  out["label"] = trace.label;
  out["steps"] = steps;
  out["buffers"] = buffers;
  out["mlx_encoders"] = trace.mlx_encoders;
  out["mlx_command_buffers"] = trace.mlx_command_buffers;
  // An older reader understands only direct grids. Make it decline this recording.
  out["unsupported"] = !include_indirect && trace.unsupported.empty() && trace.calls[kDispatchIndirect]
      ? std::string("indirect_dispatch_unsupported") : trace.unsupported;
  nb::dict calls;
  for (int k = 0; k < kCallCount; ++k) {
    calls[kCallNames[k]] = trace.calls[k];
  }
  out["calls"] = calls;
  return out;
}

// (name, buffers, steps, inputs, outputs) as plain Python data; see
// mtplx/engine/builder.py for the writer.
ProgramSpec parse_spec(nb::tuple spec) {
  ProgramSpec out;
  out.name = nb::cast<std::string>(spec[0]);
  for (nb::handle item : nb::cast<nb::list>(spec[1])) {
    nb::tuple t = nb::cast<nb::tuple>(item);
    BufferSpec buffer;
    int role = nb::cast<int>(t[0]);
    if (role < 0 || role > 5) {
      throw nb::value_error("seal: unknown buffer role");
    }
    buffer.role = (Role)role;
    buffer.nbytes = nb::cast<uint64_t>(t[1]);
    buffer.index = nb::cast<int>(t[2]);
    buffer.clobbered = nb::cast<bool>(t[3]);
    if (!t[4].is_none()) {
      nb::bytes data = nb::cast<nb::bytes>(t[4]);
      const auto* raw = (const uint8_t*)data.c_str();
      buffer.data.assign(raw, raw + data.size());
    }
    out.buffers.push_back(std::move(buffer));
  }
  for (nb::handle item : nb::cast<nb::list>(spec[2])) {
    nb::tuple t = nb::cast<nb::tuple>(item);
    StepSpec step;
    step.kernel = nb::cast<int>(t[0]);
    for (nb::handle b : nb::cast<nb::list>(t[1])) {
      nb::tuple bt = nb::cast<nb::tuple>(b);
      // A binding without the access flag is taken as one that can write.
      const bool can_write = nb::len(bt) > 3 ? nb::cast<bool>(bt[3]) : true;
      step.bindings.emplace_back(
          nb::cast<int>(bt[0]), nb::cast<int>(bt[1]), nb::cast<uint64_t>(bt[2]), can_write);
    }
    for (nb::handle b : nb::cast<nb::list>(t[2])) {
      nb::tuple bt = nb::cast<nb::tuple>(b);
      nb::bytes data = nb::cast<nb::bytes>(bt[1]);
      const auto* raw = (const uint8_t*)data.c_str();
      step.inline_bytes.emplace_back(
          nb::cast<int>(bt[0]), std::vector<uint8_t>(raw, raw + data.size()));
    }
    for (nb::handle b : nb::cast<nb::list>(t[3])) {
      nb::tuple bt = nb::cast<nb::tuple>(b);
      step.tg_memory.emplace_back(nb::cast<int>(bt[0]), nb::cast<uint64_t>(bt[1]));
    }
    step.threads = nb::cast<bool>(t[4]);
    nb::tuple grid = nb::cast<nb::tuple>(t[5]);
    nb::tuple group = nb::cast<nb::tuple>(t[6]);
    for (int k = 0; k < 3; ++k) {
      step.grid[k] = nb::cast<uint64_t>(grid[k]);
      step.group[k] = nb::cast<uint64_t>(group[k]);
    }
    step.barrier_before = nb::cast<bool>(t[7]);
    if (nb::len(t) > 8 && !t[8].is_none()) {
      nb::tuple address = nb::cast<nb::tuple>(t[8]);
      step.indirect_buffer = nb::cast<int>(address[0]);
      step.indirect_offset = nb::cast<uint64_t>(address[1]);
      if (step.indirect_buffer < 0) {
        throw nb::value_error("seal: invalid_indirect_grid");
      }
    }
    out.steps.push_back(std::move(step));
  }
  auto parse_array = [](nb::tuple t, bool is_output) {
    ArraySpec array;
    array.buffer = nb::cast<int>(t[0]);
    array.offset = nb::cast<int64_t>(t[1]);
    for (nb::handle dim : nb::cast<nb::tuple>(t[2])) {
      array.shape.push_back(nb::cast<int32_t>(dim));
    }
    std::string dtype = nb::cast<std::string>(t[3]);
    if (!dtype_from_name(dtype, &array.dtype)) {
      throw nb::value_error("seal: unknown dtype");
    }
    for (nb::handle stride : nb::cast<nb::tuple>(t[4])) {
      array.strides.push_back(nb::cast<int64_t>(stride));
    }
    array.data_size = nb::cast<uint64_t>(t[5]);
    array.flags = (uint8_t)nb::cast<int>(t[6]);
    if (is_output) {
      array.alias_input = nb::cast<int>(t[7]);
    }
    return array;
  };
  for (nb::handle item : nb::cast<nb::list>(spec[3])) {
    out.inputs.push_back(parse_array(nb::cast<nb::tuple>(item), false));
  }
  for (nb::handle item : nb::cast<nb::list>(spec[4])) {
    out.outputs.push_back(parse_array(nb::cast<nb::tuple>(item), true));
  }
  return out;
}

// The C++ test for a sole holder counts MLX's own handles.  Two Python names
// for one array object share a single handle, so Python's count is checked as
// well: an input the program writes into may be referenced by the list it was
// passed in, by what this call holds, and by `allowed` minus those.
bool python_holders_exceed(PyProgram& program, nb::handle sequence, Py_ssize_t allowed) {
  for (int index : program.program->clobbered_inputs) {
    PyObject* item = PySequence_Fast_GET_ITEM(sequence.ptr(), (Py_ssize_t)index);
    if (Py_REFCNT(item) > allowed) {
      return true;
    }
  }
  return false;
}

// ------------------------------------------------ MLX's buffer protocol
//
// numpy and memoryview read an mlx.core.array on the host through its buffer
// slot, and MLX's slot evaluates the array first.  When that evaluation
// throws (a tracer read inside mx.compile, an allocation MLX refuses), the
// C++ exception leaves the slot.  CPython calls the slot from C, so the
// exception unwinds through the interpreter's own C frames without any of
// their cleanup.  With no C++ handler above, the process aborts.  Under a
// nanobind call that does catch it (a compiled function being traced), the
// Python frames in between are never popped: the thread's current frame and
// data-stack top still point at frames whose memory is then reused, and the
// heap is corrupted from there (the SIGSEGV in the garbage collector and
// SIGBUS past a data-stack chunk that ended full pytest runs on 20 November).
//
// This wrapper is the C boundary MLX's slot does not have: the exception
// becomes the Python error nanobind raises for it, and the slot fails with -1
// as the buffer protocol requires.  numpy then reads the array through
// __array__, a nanobind method, which raises the same error cleanly.
getbufferproc g_mlx_getbuffer = nullptr;

PyObject* builtin_error_type(nb::exception_type type) {
  switch (type) {
    case nb::exception_type::stop_iteration: return PyExc_StopIteration;
    case nb::exception_type::index_error: return PyExc_IndexError;
    case nb::exception_type::key_error: return PyExc_KeyError;
    case nb::exception_type::value_error: return PyExc_ValueError;
    case nb::exception_type::type_error: return PyExc_TypeError;
    case nb::exception_type::buffer_error: return PyExc_BufferError;
    case nb::exception_type::import_error: return PyExc_ImportError;
    case nb::exception_type::attribute_error: return PyExc_AttributeError;
    default: return PyExc_RuntimeError;
  }
}

int guarded_getbuffer(PyObject* obj, Py_buffer* view, int flags) noexcept {
  try {
    return g_mlx_getbuffer(obj, view, flags);
  } catch (nb::python_error& error) {
    error.restore();
  } catch (const nb::builtin_exception& error) {
    PyErr_SetString(builtin_error_type(error.type()), error.what());
  } catch (const std::bad_alloc& error) {
    PyErr_SetString(PyExc_MemoryError, error.what());
  } catch (const std::domain_error& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
  } catch (const std::invalid_argument& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
  } catch (const std::length_error& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
  } catch (const std::out_of_range& error) {
    PyErr_SetString(PyExc_IndexError, error.what());
  } catch (const std::range_error& error) {
    PyErr_SetString(PyExc_ValueError, error.what());
  } catch (const std::overflow_error& error) {
    PyErr_SetString(PyExc_OverflowError, error.what());
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
  } catch (...) {
    PyErr_SetString(PyExc_SystemError, "mlx.core.array buffer: an exception of unknown type");
  }
  // MLX fills the view only after everything that can throw; a failed slot
  // hands back no object.
  view->obj = nullptr;
  return -1;
}

// Puts the boundary on mlx.core.array and on any Python subclass of it made
// before this module loaded (a subclass copies its base's slot when made).
// "" once installed (also when it already was), otherwise why it is not.
std::string guard_mlx_buffer_protocol() {
  nb::handle type = nb::type<mx::array>();
  if (!type.is_valid()) {
    return "mlx_array_type_not_found";
  }
  PyBufferProcs* procs = ((PyTypeObject*)type.ptr())->tp_as_buffer;
  if (procs == nullptr || procs->bf_getbuffer == nullptr) {
    return "no_buffer_slot";
  }
  if (procs->bf_getbuffer == &guarded_getbuffer) {
    return "";
  }
  const getbufferproc original = procs->bf_getbuffer;
  // Every type that carries MLX's slot, found before anything changes and
  // held until the slots are replaced.
  std::vector<nb::object> types;
  try {
    types.push_back(nb::borrow(type));
    for (size_t next = 0; next < types.size(); ++next) {
      for (nb::handle sub : types[next].attr("__subclasses__")()) {
        types.push_back(nb::borrow(sub));
      }
    }
  } catch (nb::python_error& error) {
    return std::string("subclasses_unreadable: ") + error.what();
  }
  g_mlx_getbuffer = original;
  for (const nb::object& each : types) {
    PyBufferProcs* slot = ((PyTypeObject*)each.ptr())->tp_as_buffer;
    if (slot != nullptr && slot->bf_getbuffer == original) {
      slot->bf_getbuffer = &guarded_getbuffer;
    }
  }
  return "";
}

std::string g_self_check_result;
bool g_self_check_done = false;

std::string self_check() {
  if (!g_self_check_done) {
    Engine probe;
    g_self_check_result = probe.init();
    g_self_check_done = true;
  }
  return g_self_check_result;
}

// Two copies of libmlx in one process would give this module its own MLX,
// with its own allocator and its own idea of an array.
int count_loaded_mlx_images() {
  int count = 0;
  const uint32_t n = _dyld_image_count();
  for (uint32_t i = 0; i < n; ++i) {
    const char* name = _dyld_get_image_name(i);
    if (name == nullptr) {
      continue;
    }
    const char* leaf = strrchr(name, '/');
    leaf = leaf ? leaf + 1 : name;
    if (strcmp(leaf, "libmlx.dylib") == 0) {
      ++count;
    }
  }
  return count;
}

}  // namespace

NB_MODULE(_engine, m) {
  m.doc() = "MTPLX native executor: records MLX's Metal launches and replays them";
  m.attr("NATIVE_API") = kNativeApi;
  m.attr("SUPPORTS_INDIRECT_DISPATCH") = true;
  // An output buffer given bytes in the program spec starts every run with them.
  m.attr("SUPPORTS_OUTPUT_SEEDS") = true;
  m.attr("__version__") = MTPLX_ENGINE_VERSION;
  m.attr("BUILT_AGAINST_MLX") = MTPLX_ENGINE_MLX_VERSION;
  m.attr("BUILT_AGAINST_NANOBIND") = MTPLX_ENGINE_NANOBIND_VERSION;
  m.attr("SNAPSHOT_BYTES") = (uint64_t)kSnapshotBytes;
  // The boundary is for the test suite, where tracer reads fail on purpose
  // (see guarded_getbuffer).  A serving process keeps MLX's own slot: the
  // module installs it when it loads only with MTPLX_MLX_BUFFER_GUARD=1 (the
  // suite's conftest sets it), and install_mlx_buffer_guard() installs it on
  // request.  MLX_BUFFER_GUARD says which: "installed", "off", or why it is
  // unavailable.
  const char* guard_env = std::getenv("MTPLX_MLX_BUFFER_GUARD");
  if (guard_env != nullptr && std::string(guard_env) == "1") {
    const std::string buffer_guard = guard_mlx_buffer_protocol();
    m.attr("MLX_BUFFER_GUARD") =
        buffer_guard.empty() ? std::string("installed") : "unavailable: " + buffer_guard;
  } else {
    m.attr("MLX_BUFFER_GUARD") = std::string("off");
  }
  // "installed" (also when it already was), or why it is unavailable.
  // MLX_BUFFER_GUARD keeps the state the module loaded with.
  m.def("install_mlx_buffer_guard", []() {
    const std::string buffer_guard = guard_mlx_buffer_protocol();
    return buffer_guard.empty() ? std::string("installed") : "unavailable: " + buffer_guard;
  });

  m.def("self_check", []() {
    int images = count_loaded_mlx_images();
    if (images != 1) {
      return std::string("libmlx_images_loaded=") + std::to_string(images);
    }
    return self_check();
  });
  // MLX's command-buffer limits as (ops, mb, settable).  A positive argument
  // replaces that limit when their place in MLX's device object was proven
  // (engine/limits.mm); otherwise nothing changes and settable is False.
  m.def(
      "command_buffer_limits",
      [](int ops, int mb) {
        int now_ops = 0;
        int now_mb = 0;
        const bool settable = command_buffer_limits(&now_ops, &now_mb, ops, mb);
        return nb::make_tuple(now_ops, now_mb, settable);
      },
      "ops"_a = 0,
      "mb"_a = 0);
  // Fails with a TypeError when this module and mlx.core do not share types.
  m.def("abi_probe", [](const mx::array& a) { return (int64_t)a.size(); }, "a"_a);
  m.def("write_history_rows", &write_history_rows, "history"_a, "rows"_a, "offset"_a);
  // Makes `target` (an existing array object) be `source`, in place.
  m.def(
      "overwrite_descriptor",
      [](nb::handle target, nb::handle source) {
        mx::array* dst = const_cast<mx::array*>(as_array(target));
        dst->overwrite_descriptor(*as_array(source));
      },
      "target"_a,
      "source"_a);
  m.def("is_donatable", [](nb::handle a) { return as_array(a)->is_donatable(); }, "a"_a);
  // (holders of the array's data, MLX's own verdict, Python references)
  m.def(
      "holders",
      [](nb::handle a) {
        const mx::array* array = as_array(a);
        long data = array->status() == mx::array::Status::unscheduled
            ? 0
            : array->data_shared_ptr().use_count();
        return nb::make_tuple(data, array->is_donatable(), (long)Py_REFCNT(a.ptr()));
      },
      "a"_a);
  m.def("alloc_probe", [](bool on) { return alloc_probe_enable(on); }, "on"_a);
  m.def("hooks_installed", []() { return hooks_installed(); });
  m.def("kernel_count", []() { return kernel_count(); });
  m.def("pipelines_seen", []() { return pipelines_seen(); });
  m.def("kernel", [](int kernel_id) -> nb::object {
    KernelInfo* kernel = kernel_by_id(kernel_id);
    if (kernel == nullptr) {
      return nb::none();
    }
    nb::list args;
    for (int index = 0; index < kMaxSlots; ++index) {
      if (kernel->kind[index] == 1) {
        const char access[2] = {kernel->access[index], 0};
        args.append(nb::make_tuple(index, "buffer", access));
      } else if (kernel->kind[index] == 2) {
        args.append(nb::make_tuple(index, "threadgroup", ""));
      }
    }
    nb::dict out;
    out["id"] = kernel->handle;
    out["name"] = kernel->name;
    out["reflected"] = kernel->reflected;
    out["problem"] = kernel->problem;
    out["arguments"] = args;
    return out;
  });

  nb::class_<PyHeld>(m, "Held");

  nb::class_<PyTrace>(m, "Trace")
      .def("export", [](PyTrace& self, bool include_indirect) {
        return export_trace(*self.trace, include_indirect);
      }, "include_indirect"_a = false)
      .def("launches", [](PyTrace& self) { return self.trace->steps.size(); })
      .def(
          "read_buffer",
          [](PyTrace& self, int buffer_id, uint64_t limit) -> nb::object {
            if (buffer_id < 0 || (size_t)buffer_id >= self.trace->buffers.size()) {
              throw nb::index_error("read_buffer: unknown buffer");
            }
            const TraceBuffer& buffer = self.trace->buffers[(size_t)buffer_id];
            if (buffer.length > limit || !host_readable(buffer.buffer) ||
                buffer.buffer.contents == nullptr) {
              return nb::none();
            }
            return nb::bytes((const char*)buffer.buffer.contents, buffer.length);
          },
          "buffer_id"_a,
          "limit"_a);

  nb::class_<PyProgram>(m, "Program")
      .def_prop_ro("name", [](PyProgram& self) { return self.program->name; })
      .def_prop_ro("launches", [](PyProgram& self) { return self.program->steps.size(); })
      .def_prop_ro("runs", [](PyProgram& self) { return self.program->runs; })
      .def_prop_ro("temp_bytes", [](PyProgram& self) { return self.program->temp_bytes; })
      .def_prop_ro("const_bytes", [](PyProgram& self) { return self.program->const_bytes; })
      .def_prop_ro(
          "barriers",
          [](PyProgram& self) {
            nb::dict out;
            out["recorded"] = self.program->barriers_recorded;
            out["minimal"] = self.program->barriers_min;
            out["named"] = self.program->barriers_named;
            out["whole_with_named"] = self.program->barriers_whole;
            uint32_t longest = 0;
            uint32_t in_runs = 0;
            for (size_t s = 0; s < self.program->steps.size();) {
              uint32_t run = self.program->steps[s].chain_run;
              if (run >= 2) {
                in_runs += run;
                longest = run > longest ? run : longest;
                s += run;
              } else {
                s += 1;
              }
            }
            out["launches_in_dependent_runs"] = in_runs;
            out["longest_dependent_run"] = longest;
            return out;
          })
      .def_prop_ro("n_inputs", [](PyProgram& self) { return self.program->inputs.size(); })
      .def_prop_ro("n_outputs", [](PyProgram& self) { return self.program->outputs.size(); })
      .def_prop_ro(
          "required",
          [](PyProgram& self) {
            // Per buffer: the bytes a run needs to exist in an input's buffer.
            nb::list out;
            for (uint64_t bytes : self.program->required) {
              out.append(bytes);
            }
            return out;
          });

  nb::class_<PyEngine>(m, "Engine")
      .def(
          "__init__",
          [](PyEngine* self, const std::string& label, bool validate) {
            auto engine = std::make_unique<Engine>();
            std::string problem = engine->init();
            if (!problem.empty()) {
              throw std::runtime_error("mtplx engine: " + problem);
            }
            new (self) PyEngine();
            self->label = label;
            self->validate = validate;
            self->engine = std::move(engine);
          },
          "label"_a = "",
          "validate"_a = false)
      .def(
          "info",
          [](PyEngine& self) {
            nb::dict out;
            out["label"] = self.label;
            out["module_version"] = MTPLX_ENGINE_VERSION;
            out["native_api"] = kNativeApi;
            out["built_against_mlx"] = MTPLX_ENGINE_MLX_VERSION;
            out["built_against_nanobind"] = MTPLX_ENGINE_NANOBIND_VERSION;
            out["gpu_name"] = self.engine->gpu_name;
            out["gpu_arch"] = self.engine->gpu_arch;
            out["macos"] = os_version();
            out["encoder_class"] = encoder_class_name();
            out["device_class"] = device_class_name();
            out["direct_calls"] = original_encoder_imps().cls != Nil;
            return out;
          })
      .def(
          "describe",
          [](PyEngine&, nb::list arrays) {
            auto* held = new PyHeld();
            nb::object keeper = nb::cast(held, nb::rv_policy::take_ownership);
            nb::list out;
            for (nb::handle item : arrays) {
              out.append(describe_one(*as_array(item), held));
            }
            return nb::make_tuple(keeper, out);
          },
          "arrays"_a)
      .def(
          "record_begin",
          [](PyEngine&, const std::string& label) {
            // What MLX still has in flight must have run before the first
            // launch is recorded: a buffer's bytes are read at its first
            // bind, and work in flight could still be writing them.
            bool idle;
            {
              nb::gil_scoped_release release;
              idle = wait_for_mlx(2'000'000'000ull);
            }
            if (!idle) {
              throw std::runtime_error("record_begin: mlx_busy");
            }
            std::string problem = record_begin(label);
            if (!problem.empty()) {
              throw std::runtime_error("record_begin: " + problem);
            }
          },
          "label"_a = "")
      .def(
          "clone_layout",
          [](PyEngine& self, nb::handle array, uint64_t min_length, uint64_t pad) {
            return nb::cast(self.engine->clone_layout(*as_array(array), min_length, pad));
          },
          "array"_a,
          "min_length"_a = 0,
          "pad"_a = 0)
      .def(
          "mark_tail",
          [](PyEngine& self, nb::handle array, uint64_t seed) {
            const mx::array* a = as_array(array);
            nb::gil_scoped_release release;
            return self.engine->mark_tail(*a, seed);
          },
          "array"_a,
          "seed"_a)
      .def(
          "tail_intact",
          [](PyEngine& self, nb::handle array, uint64_t seed, uint64_t marked) {
            const mx::array* a = as_array(array);
            nb::gil_scoped_release release;
            return self.engine->tail_intact(*a, seed, marked);
          },
          "array"_a,
          "seed"_a,
          "marked"_a)
      .def(
          "relax_input",
          [](PyEngine& self, PyProgram& program, int index) {
            return self.engine->relax_input(*program.program, index);
          },
          "program"_a,
          "index"_a)
      .def("last_short_detail", [](PyEngine& self) { return self.engine->last_short_detail(); })
      .def(
          "set_wait_floor_ms",
          [](PyEngine& self, uint64_t ms) { self.engine->set_wait_floor_ms(ms); },
          "ms"_a)
      .def("stuck", [](PyEngine& self) { return self.engine->stuck(); })
      .def(
          "wait_for_mlx",
          [](PyEngine&, uint64_t timeout_ms) {
            nb::gil_scoped_release release;
            return wait_for_mlx(timeout_ms * 1'000'000ull);
          },
          "timeout_ms"_a = 2000)
      .def(
          "record_mark",
          [](PyEngine&, const std::string& label) {
            std::string problem = record_mark(label);
            if (!problem.empty()) {
              throw std::runtime_error("record_mark: " + problem);
            }
          },
          "label"_a)
      .def(
          "record_end",
          [](PyEngine&) {
            auto* out = new PyTrace();
            std::string problem = record_end(&out->trace);
            if (!problem.empty()) {
              delete out;
              throw std::runtime_error("record_end: " + problem);
            }
            return nb::cast(out, nb::rv_policy::take_ownership);
          })
      .def("recording", [](PyEngine&) { return recording_on_this_thread(); })
      .def(
          "pin",
          [](PyEngine& self, nb::list arrays) {
            nb::list slots;
            for (nb::handle item : arrays) {
              const mx::array* array = as_array(item);
              if (array->status() == mx::array::Status::unscheduled) {
                slots.append(-2);  // not evaluated yet: skipped, the caller is told
                continue;
              }
              if (array->buffer().ptr() == nullptr) {
                slots.append(-1);  // empty
                continue;
              }
              slots.append(self.engine->pin(*array));
            }
            return slots;
          },
          "arrays"_a)
      .def(
          "pin_slot",
          [](PyEngine& self, uint64_t pointer) {
            return self.engine->pin_slot_for((const void*)(uintptr_t)pointer);
          },
          "pointer"_a)
      .def("pin_count", [](PyEngine& self) { return self.engine->pin_count(); })
      .def(
          "seal",
          [](PyEngine& self, nb::tuple spec) {
            ProgramSpec parsed = parse_spec(spec);
            std::string error;
            auto* out = new PyProgram();
            out->program = self.engine->seal(std::move(parsed), &error);
            if (out->program == nullptr) {
              delete out;
              throw std::runtime_error("seal: " + error);
            }
            return nb::cast(out, nb::rv_policy::take_ownership);
          },
          "spec"_a)
      .def(
          "can_run",
          [](PyEngine& self, PyProgram& program, nb::list inputs, int extra_refs) {
            nb::tuple kept = nb::tuple(inputs);
            std::vector<const mx::array*> arrays;
            arrays.reserve(nb::len(kept));
            for (nb::handle item : kept) {
              arrays.push_back(as_array(item));
            }
            if (arrays.size() == program.program->inputs.size() &&
                python_holders_exceed(program, kept, 2 + extra_refs)) {
              return std::string("state_shared");
            }
            // As a run would answer: an input in a buffer shorter than the
            // recorded one is copied into a longer one by the run itself.
            std::vector<int> grow;
            nb::gil_scoped_release release;
            return self.engine->can_run(*program.program, arrays, true, true, true, &grow);
          },
          "program"_a,
          "inputs"_a,
          "extra_refs"_a = 1)
      .def(
          "run",
          [](PyEngine& self,
             PyProgram& program,
             nb::list inputs,
             bool serial,
             bool profile,
             bool skip_drain,
             int split,
             int extra_refs,
             int dispatch,
             int barriers,
             int min_run,
             bool copy_inline,
             bool null_encoder,
             int fill,
             nb::object after_step) {
            // A tuple of our own keeps every input object alive while the
            // interpreter lock is released, without adding a holder to the
            // arrays themselves.
            nb::tuple kept = nb::tuple(inputs);
            std::vector<const mx::array*> arrays;
            arrays.reserve(nb::len(kept));
            for (nb::handle item : kept) {
              arrays.push_back(as_array(item));
            }
            if (arrays.size() == program.program->inputs.size() &&
                python_holders_exceed(program, kept, 2 + extra_refs)) {
              self.engine->count_refusal("state_shared");
              return nb::make_tuple("state_shared", nb::list());
            }
            RunOptions options;
            options.dispatch = serial ? (int)kSerial : dispatch;
            options.barriers = barriers;
            options.min_run = min_run;
            options.copy_inline = copy_inline;
            options.null_encoder = null_encoder;
            options.profile = profile;
            options.skip_drain = skip_drain;
            options.split = split;
            options.fill = fill;
            // A Python error inside the observer ends the observing, not the
            // run: the GPU work is already committed launch by launch.
            std::string observer_error;
            if (!after_step.is_none()) {
              options.after_step = [&after_step, &observer_error](size_t step) {
                if (!observer_error.empty()) {
                  return;
                }
                nb::gil_scoped_acquire lock;
                try {
                  after_step(step);
                } catch (nb::python_error& error) {
                  observer_error = error.what();
                }
              };
            }
            std::vector<mx::array> outputs;
            outputs.reserve(program.program->outputs.size());
            std::string reason;
            {
              nb::gil_scoped_release release;
              reason = self.engine->run(*program.program, arrays, &outputs, options);
            }
            if (!observer_error.empty()) {
              throw std::runtime_error("after_step: " + observer_error);
            }
            nb::list result;
            if (reason.empty()) {
              for (mx::array& out : outputs) {
                result.append(nb::cast(std::move(out)));
              }
            }
            return nb::make_tuple(reason, result);
          },
          "program"_a,
          "inputs"_a,
          "serial"_a = false,
          "profile"_a = false,
          "skip_drain"_a = false,
          "split"_a = 0,
          "extra_refs"_a = 1,
          "dispatch"_a = 0,
          "barriers"_a = 0,
          "min_run"_a = 4,
          "copy_inline"_a = false,
          "null_encoder"_a = false,
          "fill"_a = -1,
          "after_step"_a = nb::none())
      .def(
          "submit",
          [](PyEngine& self,
             PyProgram& program,
             nb::list inputs,
             int split,
             int extra_refs,
             int dispatch,
             int barriers,
             int min_run,
             bool encode_early) {
            // The first half of a run: returns once the launches are
            // committed.  "" means in flight; any other answer is a reason
            // exactly as `run` gives one.
            nb::tuple kept = nb::tuple(inputs);
            std::vector<const mx::array*> arrays;
            arrays.reserve(nb::len(kept));
            for (nb::handle item : kept) {
              arrays.push_back(as_array(item));
            }
            if (arrays.size() == program.program->inputs.size() &&
                python_holders_exceed(program, kept, 2 + extra_refs)) {
              self.engine->count_refusal("state_shared");
              return std::string("state_shared");
            }
            RunOptions options;
            options.dispatch = dispatch;
            options.barriers = barriers;
            options.min_run = min_run;
            options.split = split;
            options.encode_early = encode_early;
            nb::gil_scoped_release release;
            return self.engine->submit(program.program, arrays, options);
          },
          "program"_a,
          "inputs"_a,
          "split"_a = 0,
          "extra_refs"_a = 1,
          "dispatch"_a = 0,
          "barriers"_a = 0,
          "min_run"_a = 4,
          "encode_early"_a = false)
      .def(
          "collect",
          [](PyEngine& self, uint64_t timeout_ms) {
            // The second half: (reason, outputs).  "pending" means the GPU
            // has not finished; nothing was touched and the call can be
            // made again.
            std::vector<mx::array> outputs;
            std::string reason;
            {
              nb::gil_scoped_release release;
              reason = self.engine->collect(&outputs, timeout_ms * 1'000'000ull);
            }
            nb::list result;
            if (reason.empty()) {
              for (mx::array& out : outputs) {
                result.append(nb::cast(std::move(out)));
              }
            }
            return nb::make_tuple(reason, result);
          },
          "timeout_ms"_a)
      .def(
          "set_spin_lead_us",
          [](PyEngine& self, uint64_t us) { self.engine->set_spin_lead_us(us); },
          "us"_a)
      .def("give_up", [](PyEngine& self) {
        nb::gil_scoped_release release;
        self.engine->give_up();
      })
      .def("in_flight", [](PyEngine& self) { return self.engine->in_flight(); })
      .def(
          "peek",
          [](PyEngine& self, PyProgram& program, int buffer, int64_t offset, int64_t length) {
            // Only meaningful inside after_step: the bytes of one program
            // buffer as they are now.  The offset is in the recorded buffer;
            // an input's buffer is read where this call's array sits.
            Engine::Peek peek = self.engine->peek(*program.program, buffer);
            if (peek.bytes == nullptr) {
              throw std::runtime_error("peek: the buffer is not bound (call from after_step)");
            }
            int64_t start = offset + peek.shift;
            if (start < 0 || (size_t)start > peek.length) {
              throw std::runtime_error("peek: offset outside the buffer");
            }
            size_t count = peek.length - (size_t)start;
            if (length >= 0 && (size_t)length < count) {
              count = (size_t)length;
            }
            return nb::bytes((const char*)peek.bytes + start, count);
          },
          "program"_a,
          "buffer"_a,
          "offset"_a = 0,
          "length"_a = -1)
      .def(
          "peek_digest",
          [](PyEngine& self, PyProgram& program, int buffer, int64_t offset, int64_t length) {
            // The same region as peek, as (bytes read, 64-bit digest): for
            // buffers too large to copy at every launch.
            Engine::Peek peek = self.engine->peek(*program.program, buffer);
            if (peek.bytes == nullptr) {
              throw std::runtime_error("peek: the buffer is not bound (call from after_step)");
            }
            int64_t start = offset + peek.shift;
            if (start < 0 || (size_t)start > peek.length) {
              throw std::runtime_error("peek: offset outside the buffer");
            }
            size_t count = peek.length - (size_t)start;
            if (length >= 0 && (size_t)length < count) {
              count = (size_t)length;
            }
            const uint8_t* data = peek.bytes + start;
            uint64_t lanes[4] = {
                0x9E3779B97F4A7C15ull, 0xC2B2AE3D27D4EB4Full, 0x165667B19E3779F9ull,
                0x27D4EB2F165667C5ull};
            size_t i = 0;
            {
              nb::gil_scoped_release release;
              for (; i + 32 <= count; i += 32) {
                for (int lane = 0; lane < 4; ++lane) {
                  uint64_t word;
                  std::memcpy(&word, data + i + 8 * (size_t)lane, 8);
                  uint64_t v = (lanes[lane] ^ word) * 0x9E3779B97F4A7C15ull;
                  lanes[lane] = (v << 31) | (v >> 33);
                }
              }
            }
            uint64_t digest = lanes[0] ^ (lanes[1] * 3) ^ (lanes[2] * 5) ^ (lanes[3] * 7);
            for (; i < count; ++i) {
              digest = (digest ^ data[i]) * 0x100000001B3ull;
            }
            digest ^= (uint64_t)count * 0x9E3779B97F4A7C15ull;
            return nb::make_tuple((uint64_t)count, digest);
          },
          "program"_a,
          "buffer"_a,
          "offset"_a = 0,
          "length"_a = -1)
      .def(
          "settle",
          [](PyEngine&, nb::list arrays, uint64_t timeout_us) {
            nb::tuple kept = nb::tuple(arrays);
            std::vector<const mx::array*> pointers;
            for (nb::handle item : kept) {
              pointers.push_back(as_array(item));
            }
            nb::gil_scoped_release release;
            return Engine::settle(pointers, timeout_us);
          },
          "arrays"_a,
          "timeout_us"_a = 5000)
      .def(
          "layout_key",
          [](PyEngine&, nb::list arrays) {
            // One number for what a recording depends on in its inputs: each
            // one's type, shape and the strides that address elements.  FNV-1a.
            uint64_t hash = 1469598103934665603ull;
            auto mix = [&hash](uint64_t value) {
              for (int k = 0; k < 8; ++k) {
                hash ^= (value >> (8 * k)) & 0xff;
                hash *= 1099511628211ull;
              }
            };
            for (nb::handle item : arrays) {
              const mx::array* array = as_array(item);
              mix((uint64_t)array->dtype().val());
              const auto& shape = array->shape();
              mix(shape.size());
              if (array->status() == mx::array::Status::unscheduled) {
                for (auto extent : shape) {
                  mix((uint64_t)extent);
                }
                mix(0x6c617a79ull);  // not evaluated: its strides are not known yet
                continue;
              }
              const auto& strides = array->strides();
              for (size_t d = 0; d < shape.size(); ++d) {
                mix((uint64_t)shape[d]);
                mix(shape[d] > 1 ? (uint64_t)strides[d] : 0);
              }
            }
            return hash;
          },
          "arrays"_a)
      .def("last_run", [](PyEngine& self) { return stats_dict(self.engine->last()); })
      .def(
          "last_profile",
          [](PyEngine& self) {
            nb::list out;
            for (const StepTime& time : self.engine->last_profile()) {
              KernelInfo* kernel = kernel_by_id(time.kernel);
              out.append(nb::make_tuple(
                  time.kernel, kernel ? kernel->name : std::string(), time.gpu_ns, time.wall_ns));
            }
            return out;
          })
      .def("release_idle_temporaries", [](PyEngine& self) { return self.engine->release_idle_temporaries(); })
      .def(
          "counters",
          [](PyEngine& self) {
            nb::dict out = stats_dict(self.engine->total());
            out["runs"] = self.engine->runs();
            out["settle_waits"] = self.engine->settle_waits();
            out["settle_wait_ns"] = self.engine->settle_wait_ns();
            out["pool_bytes"] = self.engine->pool_bytes();
            out["pool_slots"] = self.engine->pool_slots();
            out["pins"] = self.engine->pin_count();
            out["kernels"] = kernel_count();
            out["pipelines_seen"] = pipelines_seen();
            nb::dict refused;
            for (const auto& [reason, count] : self.engine->refusals()) {
              refused[reason.c_str()] = count;
            }
            out["refused"] = refused;
            return out;
          })
      .def("synchronize", [](PyEngine& self) {
        nb::gil_scoped_release release;
        self.engine->synchronize();
      });
}
