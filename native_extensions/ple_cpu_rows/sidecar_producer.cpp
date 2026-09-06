// SPDX-License-Identifier: Apache-2.0

#include "sidecar_producer.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mtplx_native::ple_cpu_rows {

namespace {

using I64 = std::int64_t;
using U64 = std::uint64_t;

void require_sidecar_range(I64 size,
                           I64 offset,
                           std::size_t head,
                           U64 row_count) {
  if (size <= 0) {
    throw std::invalid_argument("sidecar ngram head size must be positive");
  }
  if (offset < 0) {
    throw std::out_of_range(
        "sidecar ngram head offset must be nonnegative");
  }

  // size is positive, so this subtraction cannot underflow.  Check the
  // signed addition before evaluating it; this keeps the constructor proof
  // defined even for a deliberately adversarial INT64_MAX fixture.
  const I64 last = size - 1;
  if (offset > std::numeric_limits<I64>::max() - last) {
    throw std::overflow_error("sidecar ngram head range overflows int64");
  }
  const I64 end = offset + last;
  if (static_cast<U64>(end) >
      static_cast<U64>(std::numeric_limits<std::uint32_t>::max())) {
    throw std::out_of_range(
        "sidecar ngram head range exceeds uint32 row IDs");
  }
  if (static_cast<U64>(end) >= row_count) {
    throw std::out_of_range(
        "sidecar ngram head range exceeds sidecar row count");
  }
  (void)head;
}

}  // namespace

std::shared_ptr<SidecarProducer> SidecarProducer::install(
    int descriptor,
    host::SidecarLayout layout,
    const std::array<I64, host::kNgramSize>& multipliers,
    const std::array<I64, host::kNgramHeads>& sizes,
    const std::array<I64, host::kNgramHeads>& offsets,
    I64 eos) {
  // RawSidecarBatchReader duplicates and validates the descriptor before its
  // shared owner is published.  Closing the caller's fd immediately after
  // this call therefore cannot invalidate a queued job.
  auto reader = std::make_shared<const host::RawSidecarBatchReader>(
      descriptor, layout);
  validate_row_ranges(sizes, offsets, layout.row_count);
  return std::shared_ptr<SidecarProducer>(new SidecarProducer(
      layout, multipliers, sizes, offsets, eos, std::move(reader)));
}

SidecarProducer::SidecarProducer(
    host::SidecarLayout layout,
    std::array<I64, host::kNgramSize> multipliers,
    std::array<I64, host::kNgramHeads> sizes,
    std::array<I64, host::kNgramHeads> offsets,
    I64 eos,
    std::shared_ptr<const host::RawSidecarBatchReader> reader)
    : row_count_(layout.row_count),
      plan_(multipliers, sizes, offsets, eos),
      reader_(std::move(reader)) {
  if (reader_ == nullptr) {
    throw std::invalid_argument("sidecar producer reader is null");
  }
}

void SidecarProducer::validate_row_ranges(
    const std::array<I64, host::kNgramHeads>& sizes,
    const std::array<I64, host::kNgramHeads>& offsets,
    U64 row_count) {
  if (row_count == 0) {
    throw std::invalid_argument("sidecar row count must be positive");
  }
  for (std::size_t head = 0; head < host::kNgramHeads; ++head) {
    require_sidecar_range(sizes[head], offsets[head], head, row_count);
  }
}

std::shared_ptr<SidecarJob> SidecarProducer::make_job(
    const SidecarJobInput& input,
    const std::shared_ptr<PermitPool>& permits) const {
  // shared_from_this() is intentional: a job keeps the complete installed
  // producer (plan plus duplicated reader) alive until its queued callable
  // finishes, rather than retaining naked references into a caller object.
  return SidecarJob::admit(shared_from_this(), input, permits);
}

std::shared_ptr<SidecarJob> SidecarJob::admit(
    std::shared_ptr<const SidecarProducer> producer,
    SidecarJobInput input,
    const std::shared_ptr<PermitPool>& permits) {
  if (producer == nullptr) {
    throw std::invalid_argument("sidecar job producer is null");
  }
  if (permits == nullptr) {
    throw std::invalid_argument("sidecar job permit pool is null");
  }
  if (!permits->try_acquire()) {
    throw std::runtime_error(
        "[mtplx_native_ple_cpu_rows] bounded two-request queue is full");
  }

  // As with the synthetic RequestState, hold a move-only lease across every
  // allocation needed to publish the shared job.  An allocation exception
  // releases the permit once, while the published job owns it thereafter.
  PermitLease lease(permits);
  auto owned = std::unique_ptr<SidecarJob>(
      new SidecarJob(std::move(producer), std::move(input), permits));
  lease.disarm();
  return std::shared_ptr<SidecarJob>(std::move(owned));
}

SidecarJob::~SidecarJob() { release_permit(); }

SidecarPackedRows SidecarJob::run() const {
  try {
    const host::NgramRowsResult result =
        producer_->plan().compute(input_.previous, input_.ids);

    // SidecarProducer::validate_row_ranges proves every result is in the
    // uint32/reader domain for every possible modulo result.  This conversion
    // is consequently an invariant-preserving cast, not a per-window check.
    SidecarRowIds row_ids{};
    for (std::size_t index = 0; index < row_ids.size(); ++index) {
      row_ids[index] = static_cast<std::uint32_t>(result.rows[index]);
    }
    // On an I/O/hash exception no output publication can follow, so release
    // the slot here before propagating the ordinary CPU exception.  Success
    // intentionally leaves the slot held until the outer primitive copies all
    // three MLX planes and calls release_permit().
    return producer_->reader().read_rows(row_ids);
  } catch (...) {
    release_permit();
    throw;
  }
}

void SidecarJob::release_permit() const noexcept {
  if (permit_held_.exchange(false, std::memory_order_acq_rel)) {
    permits_->release();
  }
}

}  // namespace mtplx_native::ple_cpu_rows
