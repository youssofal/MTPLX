// SPDX-License-Identifier: Apache-2.0

#include "ple_cpu_rows.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"

namespace mtplx_native::ple_cpu_rows {

namespace mx = mlx::core;

// Defined by ple_cpu_rows.cpp.  Keeping the installation boundary there makes
// both factories use the same process-lifetime stream and permit pool without
// duplicating the stream constructor in this provider translation unit.
mx::Stream installed_cpu_stream();
std::shared_ptr<PermitPool> installed_permits();

namespace {

using mx::array;

class PleSidecarRowsPrimitive final : public mx::Primitive {
 public:
  PleSidecarRowsPrimitive(mx::Stream cpu_stream,
                          std::shared_ptr<SidecarJob> job)
      : mx::Primitive(cpu_stream), job_(std::move(job)) {}

  ~PleSidecarRowsPrimitive() override = default;

  void eval_cpu(const std::vector<array>&,
                std::vector<array>& outputs) override {
    auto& weight_output = outputs[0];
    auto& scales_output = outputs[1];
    auto& bias_output = outputs[2];

    // The output descriptors are captured by the queued task so their MLX
    // allocator buffers stay alive through all 192 bounded preads and the
    // packed-to-plane split.
    weight_output.set_data(mx::allocator::malloc(kWeightBytes));
    scales_output.set_data(mx::allocator::malloc(kMetadataBytes));
    bias_output.set_data(mx::allocator::malloc(kMetadataBytes));

    auto weight = weight_output;
    auto scales = scales_output;
    auto bias = bias_output;
    auto job = std::move(job_);
    auto task = [job = std::move(job),
                 weight = std::move(weight),
                 scales = std::move(scales),
                 bias = std::move(bias)]() mutable {
      try {
        auto* weight_dst = weight.data<std::uint32_t>();
        auto* scales_dst = scales.data<std::uint16_t>();
        auto* bias_dst = bias.data<std::uint16_t>();
        // A failed sidecar read must never expose partially populated planes.
        std::memset(weight_dst, 0, kWeightBytes);
        std::memset(scales_dst, 0, kMetadataBytes);
        std::memset(bias_dst, 0, kMetadataBytes);

        const auto packed = job->run();
        for (std::size_t row = 0; row < kRows; ++row) {
          const auto* source = packed.data() + row * 100;
          std::memcpy(weight_dst + row * kWeightValuesPerRow, source, 80);
          std::memcpy(scales_dst + row * kMetadataValuesPerRow,
                      source + 80,
                      10);
          std::memcpy(bias_dst + row * kMetadataValuesPerRow,
                      source + 90,
                      10);
        }
      } catch (...) {
        // The ordinary MLX CPU scheduler owns exception propagation.  The
        // job only releases its permit here; no custom error/event channel is
        // introduced by the provider lane.
        job->release_permit();
        throw;
      }
      job->release_permit();
    };

    auto& encoder = mx::cpu::get_command_encoder(stream());
    encoder.set_output_array(weight_output);
    encoder.set_output_array(scales_output);
    encoder.set_output_array(bias_output);
    encoder.dispatch(std::move(task));
  }

  void eval_gpu(const std::vector<array>&,
                std::vector<array>&) override {
    throw std::runtime_error(
        "[mtplx_native_ple_cpu_rows] sidecar primitive is CPU-stream-only");
  }

  const char* name() const override { return "MtplxPleSidecarRows"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<array>&) override {
    return {mx::Shape{64, 20}, mx::Shape{64, 5}, mx::Shape{64, 5}};
  }

  bool is_equivalent(const mx::Primitive&) const override {
    // Each job has an independent snapshot, reader work, and permit.
    return false;
  }

 private:
  std::shared_ptr<SidecarJob> job_;
};

}  // namespace

CpuRowsArrays make_sidecar_rows(
    const std::shared_ptr<const SidecarProducer>& producer,
    const std::array<std::int64_t, host_provider::kHistoryRows>& previous,
    const std::array<std::int64_t, host_provider::kInputRows>& ids) {
  if (producer == nullptr) {
    throw std::invalid_argument(
        "[mtplx_native_ple_cpu_rows] sidecar producer is null");
  }
  SidecarJobInput input{previous, ids};
  auto job = producer->make_job(input, installed_permits());
  auto primitive = std::make_shared<PleSidecarRowsPrimitive>(
      installed_cpu_stream(), std::move(job));
  auto outputs = mx::array::make_arrays(
      {mx::Shape{64, 20}, mx::Shape{64, 5}, mx::Shape{64, 5}},
      {mx::uint32, mx::bfloat16, mx::bfloat16},
      primitive,
      std::vector<array>{});
  return {outputs.at(0), outputs.at(1), outputs.at(2)};
}

}  // namespace mtplx_native::ple_cpu_rows
