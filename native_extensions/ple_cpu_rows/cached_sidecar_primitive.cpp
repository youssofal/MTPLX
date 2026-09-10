// SPDX-License-Identifier: Apache-2.0

#include "cached_sidecar_primitive.h"

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

// Defined by ple_cpu_rows.cpp.  The synthetic and cached provider primitives
// deliberately share this one process-lifetime CPU stream and permit pool.
mx::Stream installed_cpu_stream();
std::shared_ptr<PermitPool> installed_permits();

namespace {

using mx::array;

void copy_packed_to_planes(const host::PackedRows& packed,
                           array& weight_output,
                           array& scales_output,
                           array& bias_output) {
  auto* weight_dst = weight_output.data<std::uint32_t>();
  auto* scales_dst = scales_output.data<std::uint16_t>();
  auto* bias_dst = bias_output.data<std::uint16_t>();
  for (std::size_t row = 0; row < kRows; ++row) {
    const auto* source = packed.data() + row * 100;
    std::memcpy(weight_dst + row * kWeightValuesPerRow, source, 80);
    std::memcpy(scales_dst + row * kMetadataValuesPerRow, source + 80, 10);
    std::memcpy(bias_dst + row * kMetadataValuesPerRow, source + 90, 10);
  }
}

class PleCachedSidecarRowsPrimitive final : public mx::Primitive {
 public:
  PleCachedSidecarRowsPrimitive(mx::Stream cpu_stream,
                                std::shared_ptr<CachedSidecarJob> job)
      : mx::Primitive(cpu_stream), job_(std::move(job)) {}

  ~PleCachedSidecarRowsPrimitive() override = default;

  void eval_cpu(const std::vector<array>&,
                std::vector<array>& outputs) override {
    auto& weight_output = outputs[0];
    auto& scales_output = outputs[1];
    auto& bias_output = outputs[2];

    // These are MLX allocator buffers.  The queued lambda captures array
    // descriptors, retaining all three buffers through row reads and copies.
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
        // A failed read must not expose partially populated planes.  The
        // scheduler propagates the exception; zeroing is not success.
        std::memset(weight.data<std::uint32_t>(), 0, kWeightBytes);
        std::memset(scales.data<std::uint16_t>(), 0, kMetadataBytes);
        std::memset(bias.data<std::uint16_t>(), 0, kMetadataBytes);
        const auto packed = job->run();
        copy_packed_to_planes(packed, weight, scales, bias);
      } catch (...) {
        // The ordinary MLX CPU scheduler owns propagation.  The job's state
        // makes permit release idempotent on both success and failure.
        job->release_permit();
        throw;
      }
      // This is the outer publication boundary: all three MLX copies have
      // completed before the job's permit can be returned.
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
        "[mtplx_native_ple_cpu_rows] cached sidecar primitive is CPU-stream-only");
  }

  const char* name() const override { return "MtplxCachedSidecarRows"; }

  std::vector<mx::Shape> output_shapes(
      const std::vector<array>&) override {
    return {mx::Shape{64, 20}, mx::Shape{64, 5}, mx::Shape{64, 5}};
  }

  bool is_equivalent(const mx::Primitive&) const override {
    // Each handoff owns an independent ticket, snapshot, and permit.
    return false;
  }

 private:
  std::shared_ptr<CachedSidecarJob> job_;
};

}  // namespace

CachedRowsSubmission make_cached_sidecar_rows(
    const std::shared_ptr<CachedSidecarProducer>& producer,
    const CachedRowHandoff& handoff) {
  if (producer == nullptr) {
    throw std::invalid_argument(
        "[mtplx_native_ple_cpu_rows] cached sidecar producer is null");
  }
  auto job = producer->make_job(handoff, installed_permits());
  std::optional<std::uint64_t> ticket;
  if (handoff.miss_count != 0) ticket = job->ticket();

  auto primitive = std::make_shared<PleCachedSidecarRowsPrimitive>(
      installed_cpu_stream(), std::move(job));
  auto outputs = mx::array::make_arrays(
      {mx::Shape{64, 20}, mx::Shape{64, 5}, mx::Shape{64, 5}},
      {mx::uint32, mx::bfloat16, mx::bfloat16},
      primitive,
      std::vector<array>{});
  return CachedRowsSubmission{
      std::move(ticket),
      CachedRowsArrays{outputs.at(0), outputs.at(1), outputs.at(2)}};
}

std::vector<CachedCompletion> drain_cached_completions(
    const std::shared_ptr<CachedSidecarProducer>& producer) {
  if (producer == nullptr) {
    throw std::invalid_argument(
        "[mtplx_native_ple_cpu_rows] cached sidecar producer is null");
  }
  return producer->drain_completed();
}

}  // namespace mtplx_native::ple_cpu_rows
