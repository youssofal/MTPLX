// SPDX-License-Identifier: Apache-2.0

#include "ple_cpu_rows.h"

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

namespace mtplx_native::ple_cpu_rows {

namespace {

using mx::array;

// Installation creates this one CPU stream and one permit pool.  A window
// only captures the already-installed objects; it never creates a stream or a
// worker of its own.
struct Runtime final {
  Runtime()
      : cpu_stream(mx::new_thread_unsafe_stream(mx::Device::cpu)),
        permits(std::make_shared<PermitPool>()) {}

  mx::Stream cpu_stream;
  std::shared_ptr<PermitPool> permits;
};

Runtime& runtime() {
  static Runtime runtime;
  return runtime;
}

// The provider primitive is compiled in a separate translation unit so the
// established synthetic source-level contract remains independently
// reviewable.  Both factories still resolve these accessors to the same
// process-lifetime stream and permit pool.
}  // namespace

mx::Stream installed_cpu_stream() { return runtime().cpu_stream; }

std::shared_ptr<PermitPool> installed_permits() { return runtime().permits; }

namespace {

class PleCpuRowsPrimitive final : public mx::Primitive {
 public:
  PleCpuRowsPrimitive(mx::Stream cpu_stream,
                      std::shared_ptr<RequestState> request)
      : mx::Primitive(cpu_stream), request_(std::move(request)) {}

  ~PleCpuRowsPrimitive() override = default;

  void eval_cpu(const std::vector<array>&,
                std::vector<array>& outputs) override {
    auto& weight_output = outputs[0];
    auto& scales_output = outputs[1];
    auto& bias_output = outputs[2];

    // These are MLX allocator buffers.  The copied array descriptors below
    // are captured by the queued lambda and therefore keep each buffer alive
    // until the delayed task has copied or zeroed it.
    weight_output.set_data(mx::allocator::malloc(kWeightBytes));
    scales_output.set_data(mx::allocator::malloc(kMetadataBytes));
    bias_output.set_data(mx::allocator::malloc(kMetadataBytes));

    auto weight = weight_output;
    auto scales = scales_output;
    auto bias = bias_output;
    auto state = std::move(request_);

    auto task = [state,
                 weight = std::move(weight),
                 scales = std::move(scales),
                 bias = std::move(bias)]() mutable {
      try {
        auto* weight_dst = weight.data<std::uint32_t>();
        auto* scales_dst = scales.data<std::uint16_t>();
        auto* bias_dst = bias.data<std::uint16_t>();
        std::memset(weight_dst, 0, kWeightBytes);
        std::memset(scales_dst, 0, kMetadataBytes);
        std::memset(bias_dst, 0, kMetadataBytes);

        if (state->cancelled()) {
          throw std::runtime_error(
              "[mtplx_native_ple_cpu_rows] request cancelled");
        }
        if (state->delay_ms() != 0) {
          std::this_thread::sleep_for(
              std::chrono::milliseconds(state->delay_ms()));
        }
        if (state->force_fail()) {
          throw std::runtime_error(
              "[mtplx_native_ple_cpu_rows] request failed");
        }

        const auto& payload = state->payload();
        for (std::size_t row = 0; row < kRows; ++row) {
          const auto* packed = payload.data() + row * 100;
          std::memcpy(weight_dst + row * kWeightValuesPerRow, packed, 80);
          std::memcpy(scales_dst + row * kMetadataValuesPerRow,
                      packed + 80,
                      10);
          std::memcpy(bias_dst + row * kMetadataValuesPerRow,
                      packed + 90,
                      10);
        }
      } catch (...) {
        // The scheduler observes this exception on its ordinary CPU stream;
        // there is no custom error pointer or hidden signal to maintain.
        state->release_permit();
        throw;
      }
      state->release_permit();
    };

    auto& encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_output_array(weight_output);
    encoder.set_output_array(scales_output);
    encoder.set_output_array(bias_output);
    // Do not release here if dispatch itself reports an enqueue error: the
    // encoder may already own the task (notably while adding its completion
    // task at the dispatch-group boundary).  The last shared RequestState
    // owner, either queued task or local lambda, releases exactly once.
    encoder.dispatch(std::move(task));
  }

  void eval_gpu(const std::vector<array>&,
                std::vector<array>&) override {
    throw std::runtime_error(
        "[mtplx_native_ple_cpu_rows] primitive is CPU-stream-only");
  }

  const char* name() const override { return "MtplxPleCpuRows"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<array>&) override {
    return {mx::Shape{64, 20}, mx::Shape{64, 5}, mx::Shape{64, 5}};
  }

  bool is_equivalent(const mx::Primitive&) const override {
    // Every request owns a distinct permit/payload and must remain a distinct
    // graph node even if two payloads contain equal bytes.
    return false;
  }

 private:
  std::shared_ptr<RequestState> request_;
};

// The real sidecar lane has its own explicit primitive and job state.  It
// shares the installed CPU stream and permit pool above with the synthetic
// control lane, but it never falls back to that lane when provider work is
// selected.
}  // namespace

CpuRowsArrays make_cpu_rows(const PackedPayload& payload,
                            int delay_ms,
                            bool force_fail,
                            bool cancelled) {
  auto& installed = runtime();
  auto request = RequestState::admit(
      installed.permits, payload, delay_ms, force_fail, cancelled);
  auto primitive = std::make_shared<PleCpuRowsPrimitive>(
      installed.cpu_stream, std::move(request));
  auto outputs = mx::array::make_arrays(
      {mx::Shape{64, 20}, mx::Shape{64, 5}, mx::Shape{64, 5}},
      {mx::uint32, mx::bfloat16, mx::bfloat16},
      primitive,
      std::vector<array>{});
  return {outputs.at(0), outputs.at(1), outputs.at(2)};
}

}  // namespace mtplx_native::ple_cpu_rows
