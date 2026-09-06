// SPDX-License-Identifier: Apache-2.0
//
// MLX-free construction and job ownership for the real sidecar CPU-row
// producer.  The implementation deliberately reuses the standalone
// host_provider NgramPlan and RawSidecarBatchReader; this header contains no
// model, Python, or MLX dependency.

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "host_provider.h"
#include "request_state.h"

namespace mtplx_native::ple_cpu_rows {

namespace host = mtplx_native::host_provider;

using SidecarPackedRows = host::PackedRows;
using SidecarRowIds = host::RowIds;

// The only values that vary for an M4 sidecar job.  The arrays are copied into
// SidecarJob at admission; callers may freely reuse or mutate their originals
// once make_job returns.
struct SidecarJobInput {
  std::array<std::int64_t, host::kHistoryRows> previous{};
  std::array<std::int64_t, host::kInputRows> ids{};
};

// A construction-bound, immutable sidecar route.  The plan and reader are
// made once and retained by every job.  In particular, no fstat/layout check,
// plan construction, or descriptor duplication occurs on a window job.
class SidecarProducer final
    : public std::enable_shared_from_this<SidecarProducer> {
 public:
  static std::shared_ptr<SidecarProducer> install(
      int descriptor,
      host::SidecarLayout layout,
      const std::array<std::int64_t, host::kNgramSize>& multipliers,
      const std::array<std::int64_t, host::kNgramHeads>& sizes,
      const std::array<std::int64_t, host::kNgramHeads>& offsets,
      std::int64_t eos);

  SidecarProducer(const SidecarProducer&) = delete;
  SidecarProducer& operator=(const SidecarProducer&) = delete;

  const host::NgramPlan& plan() const noexcept { return plan_; }
  const host::RawSidecarBatchReader& reader() const noexcept {
    return *reader_;
  }
  std::uint64_t row_count() const noexcept { return row_count_; }

  // Admission is nonblocking.  The returned job owns one permit until its
  // run() task completes or the job is abandoned before dispatch.
  std::shared_ptr<class SidecarJob> make_job(
      const SidecarJobInput& input,
      const std::shared_ptr<PermitPool>& permits) const;

 private:
  SidecarProducer(
      host::SidecarLayout layout,
      std::array<std::int64_t, host::kNgramSize> multipliers,
      std::array<std::int64_t, host::kNgramHeads> sizes,
      std::array<std::int64_t, host::kNgramHeads> offsets,
      std::int64_t eos,
      std::shared_ptr<const host::RawSidecarBatchReader> reader);

  static void validate_row_ranges(
      const std::array<std::int64_t, host::kNgramHeads>& sizes,
      const std::array<std::int64_t, host::kNgramHeads>& offsets,
      std::uint64_t row_count);

  const std::uint64_t row_count_;
  const host::NgramPlan plan_;
  const std::shared_ptr<const host::RawSidecarBatchReader> reader_;
};

// One immutable M4 sidecar request.  run() computes exactly 64 row IDs using
// the installed plan and delegates the fixed 192 preads to the installed
// reader; it returns packed bytes only and never publishes mutable history.
class SidecarJob final {
 public:
  static std::shared_ptr<SidecarJob> admit(
      std::shared_ptr<const SidecarProducer> producer,
      SidecarJobInput input,
      const std::shared_ptr<PermitPool>& permits);

  ~SidecarJob();

  SidecarJob(const SidecarJob&) = delete;
  SidecarJob& operator=(const SidecarJob&) = delete;

  SidecarPackedRows run() const;
  void release_permit() const noexcept;

 private:
  SidecarJob(std::shared_ptr<const SidecarProducer> producer,
             SidecarJobInput input,
             std::shared_ptr<PermitPool> permits) noexcept
      : producer_(std::move(producer)),
        input_(std::move(input)),
        permits_(std::move(permits)) {}

  const std::shared_ptr<const SidecarProducer> producer_;
  const SidecarJobInput input_;
  const std::shared_ptr<PermitPool> permits_;
  mutable std::atomic_bool permit_held_{true};
};

}  // namespace mtplx_native::ple_cpu_rows
