// SPDX-License-Identifier: Apache-2.0

#include "host_provider.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace mtplx_native::host_provider {

namespace {

using U64 = std::uint64_t;
using I64 = std::int64_t;

class ScopedFd final {
 public:
  explicit ScopedFd(int descriptor) noexcept : descriptor_(descriptor) {}
  ~ScopedFd() {
    if (descriptor_ >= 0) {
      ::close(descriptor_);
    }
  }

  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;

  int get() const noexcept { return descriptor_; }

  int release() noexcept {
    const int descriptor = descriptor_;
    descriptor_ = -1;
    return descriptor;
  }

 private:
  int descriptor_ = -1;
};

U64 bits_of(I64 value) noexcept {
  U64 bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

I64 value_of(U64 bits) noexcept {
  I64 value = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

U64 checked_add(U64 left, U64 right, const char* what) {
  if (right > std::numeric_limits<U64>::max() - left) {
    throw std::overflow_error(std::string(what) + " addition overflow");
  }
  return left + right;
}

U64 checked_mul(U64 left, U64 right, const char* what) {
  if (left != 0 && right > std::numeric_limits<U64>::max() / left) {
    throw std::overflow_error(std::string(what) + " multiplication overflow");
  }
  return left * right;
}

U64 nonnegative_mod(U64 signed_bits, U64 divisor) noexcept {
  if ((signed_bits >> 63) == 0) {
    return signed_bits % divisor;
  }
  // Compute abs(INT64_MIN) in unsigned arithmetic; signed negation would be
  // undefined at the exact input that the parity harness exercises.
  const U64 magnitude = (~signed_bits) + U64{1};
  const U64 remainder = magnitude % divisor;
  return remainder == 0 ? U64{0} : divisor - remainder;
}

struct Range {
  U64 begin;
  U64 end;
};

Range plane_range(const PlaneSpec& plane, U64 row_count, const char* label) {
  const U64 bytes = checked_mul(row_count, plane.stride, label);
  if (bytes != plane.length) {
    throw std::invalid_argument(std::string(label) +
                                " length does not match row stride");
  }
  return {plane.offset, checked_add(plane.offset, plane.length, label)};
}

void require_nonoverlap(const Range& left,
                        const char* left_label,
                        const Range& right,
                        const char* right_label) {
  if (left.begin < right.end && right.begin < left.end) {
    throw std::invalid_argument(std::string(left_label) + " overlaps " +
                                right_label);
  }
}

void read_exact(int descriptor,
                std::uint8_t* destination,
                std::size_t length,
                U64 offset) {
  std::size_t completed = 0;
  while (completed < length) {
    // The construction-time plane range check proves offset + length is at
    // most off_t::max(), so progress within this fixed read is bounded too.
    const U64 current = offset + static_cast<U64>(completed);
    const ssize_t count = ::pread(
        descriptor,
        destination + completed,
        length - completed,
        static_cast<off_t>(current));
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::system_error(errno, std::generic_category(), "pread");
    }
    if (count == 0) {
      throw std::runtime_error("sidecar short read");
    }
    completed += static_cast<std::size_t>(count);
  }
}

}  // namespace

NgramPlan::NgramPlan(
    const std::array<I64, kNgramSize>& multipliers,
    const std::array<I64, kNgramHeads>& sizes,
    const std::array<I64, kNgramHeads>& offsets,
    I64 eos)
    : eos_(eos) {
  for (std::size_t index = 0; index < kNgramSize; ++index) {
    multiplier_bits_[index] = bits_of(multipliers[index]);
  }
  for (I64 size : sizes) {
    if (size <= 0) {
      throw std::invalid_argument("ngram head size must be positive");
    }
  }
  for (std::size_t index = 0; index < kNgramHeads; ++index) {
    sizes_[index] = static_cast<U64>(sizes[index]);
    offset_bits_[index] = bits_of(offsets[index]);
  }
}

NgramRowsResult NgramPlan::compute(
    const std::array<I64, kHistoryRows>& previous,
    const std::array<I64, kInputRows>& ids) const {
  std::array<I64, kHistoryRows + kInputRows> history{};
  std::copy(previous.begin(), previous.end(), history.begin());
  std::copy(ids.begin(), ids.end(), history.begin() + kHistoryRows);

  // Record the preceding EOS before inspecting the current position. This is
  // the NumPy prev_incl[:, :-1] rule and keeps a current EOS out of its own
  // segment-start scan.
  std::array<I64, kHistoryRows + kInputRows> previous_eos{};
  I64 last_eos = -1;
  for (std::size_t position = 0; position < history.size(); ++position) {
    previous_eos[position] = last_eos;
    if (history[position] == eos_) {
      last_eos = static_cast<I64>(position);
    }
  }

  std::array<std::array<U64, kHistoryRows + kInputRows>, kNgramSize> shifted{};
  for (std::size_t position = 0; position < history.size(); ++position) {
    shifted[0][position] = bits_of(history[position]);
    const I64 position_in_segment =
        static_cast<I64>(position) - (previous_eos[position] + 1);
    for (std::size_t shift = 1; shift < kNgramSize; ++shift) {
      const bool has_source = position >= shift;
      const bool is_valid =
          has_source && position_in_segment >= static_cast<I64>(shift);
      const std::size_t source = has_source ? position - shift : 0;
      shifted[shift][position] =
          is_valid ? bits_of(history[source]) : bits_of(eos_);
    }
  }

  NgramRowsResult result{};
  for (std::size_t input_position = 0; input_position < kInputRows;
       ++input_position) {
    const std::size_t history_position = kHistoryRows + input_position;
    const std::size_t output_base = input_position * kNgramHeads;
    for (std::size_t ngram = 2; ngram <= kNgramSize; ++ngram) {
      const std::size_t head_base = (ngram - 2) * kHeadsPerNgram;
      U64 mixed = multiplier_bits_[0] * shifted[0][history_position];
      for (std::size_t part = 1; part < ngram; ++part) {
        mixed ^= multiplier_bits_[part] * shifted[part][history_position];
      }
      for (std::size_t head = 0; head < kHeadsPerNgram; ++head) {
        const std::size_t head_index = head_base + head;
        const U64 remainder = nonnegative_mod(mixed, sizes_[head_index]);
        const U64 row_bits = remainder + offset_bits_[head_index];
        result.rows[output_base + head_index] = value_of(row_bits);
      }
    }
  }

  result.history[0] = ids[kInputRows - kHistoryRows];
  result.history[1] = ids[kInputRows - 1];
  return result;
}

NgramRowsResult ngram_rows_fixed_m4(
    const std::array<I64, kHistoryRows>& previous,
    const std::array<I64, kInputRows>& ids,
    const std::array<I64, kNgramSize>& multipliers,
    const std::array<I64, kNgramHeads>& sizes,
    const std::array<I64, kNgramHeads>& offsets,
    I64 eos) {
  const NgramPlan plan(multipliers, sizes, offsets, eos);
  return plan.compute(previous, ids);
}

void check_sidecar_layout(const SidecarLayout& layout, U64 file_size) {
  const U64 off_t_max = static_cast<U64>(std::numeric_limits<off_t>::max());
  if (file_size > off_t_max) {
    throw std::overflow_error("sidecar file size exceeds off_t");
  }
  if (layout.row_count == 0) {
    throw std::invalid_argument("sidecar row count must be positive");
  }
  if (layout.weights.stride != kWeightBytesPerRow) {
    throw std::invalid_argument("weights stride must be 80 bytes");
  }
  if (layout.scales.stride != kScaleBytesPerRow) {
    throw std::invalid_argument("scales stride must be 10 bytes");
  }
  if (layout.biases.stride != kBiasBytesPerRow) {
    throw std::invalid_argument("biases stride must be 10 bytes");
  }

  const Range weights = plane_range(layout.weights, layout.row_count, "weights");
  const Range scales = plane_range(layout.scales, layout.row_count, "scales");
  const Range biases = plane_range(layout.biases, layout.row_count, "biases");
  for (const auto& range : {weights, scales, biases}) {
    if (range.end > off_t_max) {
      throw std::overflow_error("sidecar plane exceeds off_t");
    }
    if (range.end > file_size) {
      throw std::out_of_range("sidecar plane exceeds file size");
    }
  }
  require_nonoverlap(weights, "weights", scales, "scales");
  require_nonoverlap(weights, "weights", biases, "biases");
  require_nonoverlap(scales, "scales", biases, "biases");
}

U64 checked_plane_row_offset(const PlaneSpec& plane,
                             U64 row_id,
                             U64 row_count) {
  if (row_count == 0 || row_id >= row_count) {
    throw std::out_of_range("sidecar row ID is outside row count");
  }
  return checked_add(
      plane.offset,
      checked_mul(row_id, plane.stride, "row offset"),
      "row offset");
}

RawSidecarBatchReader::RawSidecarBatchReader(int descriptor,
                                             SidecarLayout layout)
    : layout_(layout) {
  if (descriptor < 0) {
    throw std::invalid_argument("sidecar descriptor must be nonnegative");
  }
  ScopedFd owned(::dup(descriptor));
  if (owned.get() < 0) {
    throw std::system_error(errno, std::generic_category(), "dup");
  }
  const int flags = ::fcntl(owned.get(), F_GETFD);
  if (flags < 0) {
    throw std::system_error(errno, std::generic_category(), "fcntl(F_GETFD)");
  }
  if (::fcntl(owned.get(), F_SETFD, flags | FD_CLOEXEC) != 0) {
    throw std::system_error(
        errno, std::generic_category(), "fcntl(FD_CLOEXEC)");
  }
  struct stat info {};
  if (::fstat(owned.get(), &info) != 0) {
    throw std::system_error(errno, std::generic_category(), "fstat");
  }
  if (!S_ISREG(info.st_mode)) {
    throw std::invalid_argument("sidecar descriptor must refer to a regular file");
  }
  if (info.st_size < 0) {
    throw std::runtime_error("sidecar file size is negative");
  }
  file_size_ = static_cast<U64>(info.st_size);
  if (file_size_ > static_cast<U64>(std::numeric_limits<off_t>::max())) {
    throw std::overflow_error("sidecar file size exceeds off_t");
  }
  check_sidecar_layout(layout_, file_size_);
  descriptor_ = owned.release();
}

RawSidecarBatchReader::~RawSidecarBatchReader() {
  if (descriptor_ >= 0) {
    ::close(descriptor_);
  }
}

RawSidecarBatchReader::RawSidecarBatchReader(
    RawSidecarBatchReader&& other) noexcept
    : descriptor_(other.descriptor_),
      layout_(other.layout_),
      file_size_(other.file_size_) {
  other.descriptor_ = -1;
  other.file_size_ = 0;
}

RawSidecarBatchReader& RawSidecarBatchReader::operator=(
    RawSidecarBatchReader&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (descriptor_ >= 0) {
    ::close(descriptor_);
  }
  descriptor_ = other.descriptor_;
  layout_ = other.layout_;
  file_size_ = other.file_size_;
  other.descriptor_ = -1;
  other.file_size_ = 0;
  return *this;
}

void RawSidecarBatchReader::read_one_into(
    std::uint32_t row_id,
    std::uint8_t* destination) const {
  const U64 row = row_id;
  if (row >= layout_.row_count) {
    throw std::out_of_range("sidecar row ID is outside row count");
  }
  // check_sidecar_layout proved each complete plane range fits in both the
  // file and off_t, so these direct operations cannot overflow for a valid
  // row.  Keep the single variable row-ID check above outside the three
  // fixed plane reads.
  const U64 weight_offset =
      layout_.weights.offset + row * layout_.weights.stride;
  const U64 scale_offset =
      layout_.scales.offset + row * layout_.scales.stride;
  const U64 bias_offset =
      layout_.biases.offset + row * layout_.biases.stride;
  read_exact(descriptor_, destination, kWeightBytesPerRow, weight_offset);
  read_exact(
      descriptor_, destination + kWeightBytesPerRow, kScaleBytesPerRow,
      scale_offset);
  read_exact(
      descriptor_,
      destination + kWeightBytesPerRow + kScaleBytesPerRow,
      kBiasBytesPerRow,
      bias_offset);
}

PackedRow RawSidecarBatchReader::read_one(std::uint32_t row_id) const {
  PackedRow output{};
  read_one_into(row_id, output.data());
  return output;
}

std::vector<PackedRow> RawSidecarBatchReader::read_subset(
    const std::vector<std::uint32_t>& row_ids) const {
  if (row_ids.size() > kRowsPerWindow) {
    throw std::invalid_argument("sidecar row subset exceeds one M4 window");
  }
  std::vector<PackedRow> output(row_ids.size());
  for (std::size_t index = 0; index < row_ids.size(); ++index) {
    read_one_into(row_ids[index], output[index].data());
  }
  return output;
}

PackedRows RawSidecarBatchReader::read_rows(const RowIds& row_ids) const {
  PackedRows output{};
  for (std::size_t index = 0; index < row_ids.size(); ++index) {
    read_one_into(
        row_ids[index], output.data() + index * kPackedBytesPerRow);
  }
  return output;
}

}  // namespace mtplx_native::host_provider
