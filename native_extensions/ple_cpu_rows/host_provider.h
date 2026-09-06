// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mtplx_native::host_provider {

constexpr std::size_t kHistoryRows = 2;
constexpr std::size_t kInputRows = 4;
constexpr std::size_t kNgramSize = 3;
constexpr std::size_t kHeadsPerNgram = 8;
constexpr std::size_t kNgramHeads = 16;
constexpr std::size_t kNgramRowsPerWindow = kInputRows * kNgramHeads;

constexpr std::size_t kRowsPerWindow = 64;
constexpr std::size_t kWeightBytesPerRow = 80;
constexpr std::size_t kScaleBytesPerRow = 10;
constexpr std::size_t kBiasBytesPerRow = 10;
constexpr std::size_t kPackedBytesPerRow =
    kWeightBytesPerRow + kScaleBytesPerRow + kBiasBytesPerRow;
constexpr std::size_t kPackedWindowBytes =
    kRowsPerWindow * kPackedBytesPerRow;

static_assert(kNgramRowsPerWindow == kRowsPerWindow);
static_assert(kPackedBytesPerRow == 100);
static_assert(kPackedWindowBytes == 6400);

struct NgramRowsResult {
  std::array<std::int64_t, kNgramRowsPerWindow> rows{};
  std::array<std::int64_t, kHistoryRows> history{};
};

// Construction binds the invariant hash parameters once.  compute() accepts
// only the values that change for a window and performs no plan validation.
class NgramPlan final {
 public:
  NgramPlan(const std::array<std::int64_t, kNgramSize>& multipliers,
            const std::array<std::int64_t, kNgramHeads>& sizes,
            const std::array<std::int64_t, kNgramHeads>& offsets,
            std::int64_t eos);

  NgramRowsResult compute(
      const std::array<std::int64_t, kHistoryRows>& previous,
      const std::array<std::int64_t, kInputRows>& ids) const;

 private:
  std::array<std::uint64_t, kNgramSize> multiplier_bits_{};
  std::array<std::uint64_t, kNgramHeads> sizes_{};
  std::array<std::uint64_t, kNgramHeads> offset_bits_{};
  std::int64_t eos_ = 0;
};

// One fixed-M4 call: two history IDs plus four new IDs produce 64 row IDs,
// followed by the last two IDs needed by the next window.  This convenience
// wrapper constructs a plan; an installed route should retain NgramPlan.
NgramRowsResult ngram_rows_fixed_m4(
    const std::array<std::int64_t, kHistoryRows>& previous,
    const std::array<std::int64_t, kInputRows>& ids,
    const std::array<std::int64_t, kNgramSize>& multipliers,
    const std::array<std::int64_t, kNgramHeads>& sizes,
    const std::array<std::int64_t, kNgramHeads>& offsets,
    std::int64_t eos);

struct PlaneSpec {
  std::uint64_t offset = 0;
  std::uint64_t length = 0;
  std::uint64_t stride = 0;
};

struct SidecarLayout {
  std::uint64_t row_count = 0;
  PlaneSpec weights{};
  PlaneSpec scales{};
  PlaneSpec biases{};
};

// Pure layout checks used by construction and by the bounded CPU harness.
// file_size is supplied separately so overflow and range cases need no large
// physical fixture.
void check_sidecar_layout(const SidecarLayout& layout,
                          std::uint64_t file_size);

std::uint64_t checked_plane_row_offset(const PlaneSpec& plane,
                                       std::uint64_t row_id,
                                       std::uint64_t row_count);

using PackedRows = std::array<std::uint8_t, kPackedWindowBytes>;
using PackedRow = std::array<std::uint8_t, kPackedBytesPerRow>;
using RowIds = std::array<std::uint32_t, kRowsPerWindow>;

// Synchronously reads one bounded 64-row batch from three regular-file
// planes. The constructor duplicates the caller's descriptor and owns that
// duplicate; the original may be closed before read_rows is called.
class RawSidecarBatchReader final {
 public:
  RawSidecarBatchReader(int descriptor, SidecarLayout layout);
  ~RawSidecarBatchReader();

  RawSidecarBatchReader(const RawSidecarBatchReader&) = delete;
  RawSidecarBatchReader& operator=(const RawSidecarBatchReader&) = delete;

  RawSidecarBatchReader(RawSidecarBatchReader&& other) noexcept;
  RawSidecarBatchReader& operator=(RawSidecarBatchReader&& other) noexcept;

  // Read one complete packed row or an ordered bounded subset.  The caller
  // supplies unique IDs when it wants deduplicated I/O; this API preserves
  // the supplied order and does not add reuse policy.
  PackedRow read_one(std::uint32_t row_id) const;
  std::vector<PackedRow> read_subset(
      const std::vector<std::uint32_t>& row_ids) const;

  PackedRows read_rows(const RowIds& row_ids) const;

 private:
  void read_one_into(std::uint32_t row_id,
                     std::uint8_t* destination) const;

  int descriptor_ = -1;
  SidecarLayout layout_{};
  std::uint64_t file_size_ = 0;
};

}  // namespace mtplx_native::host_provider
