#include "history_rows.h"

#include <algorithm>
#include <stdexcept>
#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

namespace mtplx_engine {
namespace mx = mlx::core;
namespace {

constexpr auto kName = "custom_kernel_mtplx_write_history_rows";
constexpr auto kSource = R"(
    #include <metal_stdlib>
    using namespace metal;
    kernel void custom_kernel_mtplx_write_history_rows(
        device const ushort* history [[buffer(0)]], device const ushort* patch [[buffer(1)]],
        device const int* offset [[buffer(2)]], device ushort* out [[buffer(3)]],
        constant uint4& shape [[buffer(4)]], constant uint& workers [[buffer(5)]],
        constant ulong* patch_stride [[buffer(6)]], constant ulong* history_stride [[buffer(7)]],
        constant uint& aligned_history [[buffer(8)]], uint3 index [[thread_position_in_grid]]) {
        const uint head = index.z % shape.x, batch = index.z / shape.x;
        const uint feature = index.x * 4;
        const ulong bank_base = ulong(index.z) * shape.y * shape.w + feature;
        const long start = offset[0];
        const bool valid = start >= 0 && start + shape.z <= shape.y;
        if (index.y < shape.z && valid) {
            const ulong src = batch*patch_stride[0] + head*patch_stride[1] +
                              index.y*patch_stride[2] + feature*patch_stride[3];
            const ushort4 incoming(patch[src], patch[src+patch_stride[3]],
                                   patch[src+2*patch_stride[3]], patch[src+3*patch_stride[3]]);
            const ulong dst = bank_base + (ulong(start) + index.y)*shape.w;
            *reinterpret_cast<device ushort4*>(out + dst) = incoming;
        }
        // A sole owner donated its storage. A retained history needs its
        // unchanged rows in the new buffer. Both use exactly the same grid.
        if (history == out) return;
        for (ulong row = index.y; row < shape.y; row += workers) {
            if (valid && row >= ulong(start) && row < ulong(start) + shape.z) continue;
            const ulong src = batch*history_stride[0] + head*history_stride[1] +
                              row*history_stride[2] + feature*history_stride[3];
            const ushort4 old = aligned_history ? *reinterpret_cast<const device ushort4*>(history + src) :
                ushort4(history[src], history[src+history_stride[3]],
                        history[src+2*history_stride[3]], history[src+3*history_stride[3]]);
            *reinterpret_cast<device ushort4*>(out + bank_base + row*shape.w) = old;
        }
    })";

class WriteHistoryRows final : public mx::UnaryPrimitive {
 public:
  using UnaryPrimitive::UnaryPrimitive;
  const char* name() const override { return kName; }
  bool is_equivalent(const mx::Primitive&) const override { return true; }
  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {inputs[0].shape()};
  }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::invalid_argument("write_history_rows requires a Metal stream");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const auto& history = inputs[0];
    const auto& rows = inputs[1];
    const uint32_t aligned = history.flags().row_contiguous && history.offset() % 8 == 0;
    if (aligned && history.is_donatable()) {
      out.copy_shared_buffer(history);
    } else {
      out.set_data(mx::allocator::malloc(out.nbytes()));
    }
    auto& device = mx::metal::device(stream().device);
    auto* library = device.get_library(kName, [] { return std::string(kSource); });
    auto* kernel = device.get_kernel(kName, library);
    auto& encoder = mx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(history, 0);
    encoder.set_input_array(rows, 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_output_array(out, 3);
    const uint32_t shape[] = {uint32_t(history.shape(1)), uint32_t(history.shape(2)),
                              uint32_t(rows.shape(2)), uint32_t(history.shape(3))};
    const uint32_t workers = std::min(shape[1], std::max(shape[2], 256u));
    encoder.set_bytes(shape, 4);
    encoder.set_bytes(workers, 5);
    encoder.set_vector_bytes(rows.strides(), 6);
    encoder.set_vector_bytes(history.strides(), 7);
    encoder.set_bytes(aligned, 8);
    MTL::Size grid(shape[3] / 4, workers, size_t(history.shape(0)) * shape[0]);
    encoder.dispatch_threads(grid, MTL::Size(std::min(grid.width, kernel->maxTotalThreadsPerThreadgroup()), 1, 1));
  }
};
} // namespace

mx::array write_history_rows(const mx::array& history, const mx::array& rows, const mx::array& offset) {
  if (history.ndim() != 4 || rows.ndim() != 4 || history.shape(0) != rows.shape(0) ||
      history.shape(1) != rows.shape(1) || history.shape(3) != rows.shape(3) || history.shape(3) % 4 ||
      !rows.size() || history.shape(2) < rows.shape(2) || history.dtype() != rows.dtype() ||
      (history.dtype() != mx::float16 && history.dtype() != mx::bfloat16) ||
      offset.dtype() != mx::int32 || offset.size() != 1 || offset.ndim() > 1) {
    throw std::invalid_argument("incompatible history, incoming rows or offset");
  }
  return mx::array(history.shape(), history.dtype(), std::make_shared<WriteHistoryRows>(mx::to_stream({})),
                   {history, rows, offset});
}
} // namespace mtplx_engine
