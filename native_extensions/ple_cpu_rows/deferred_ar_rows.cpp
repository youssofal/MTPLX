// SPDX-License-Identifier: Apache-2.0

#include "deferred_ar_rows.h"

#include <cstring>

#include "mlx/allocator.h"

namespace mtplx_native::ple_cpu_rows {

namespace {

constexpr std::size_t kArWeightBytes = kArWeightValues * sizeof(std::uint32_t);
constexpr std::size_t kArMetadataBytes =
    kArMetadataValues * sizeof(std::uint16_t);
constexpr std::size_t kArTokenBytes = sizeof(std::int64_t);

}  // namespace

DeferredArToken::DeferredArToken()
    : token_(mx::allocator::malloc(kArTokenBytes), mx::Shape{1}, mx::int64) {
  std::memset(token_.data<std::int64_t>(), 0, kArTokenBytes);
}

std::shared_ptr<DeferredArToken> DeferredArToken::make() {
  return std::shared_ptr<DeferredArToken>(new DeferredArToken());
}

mx::array DeferredArToken::array() const {
  return token_;
}

void DeferredArToken::fill(std::int64_t token) {
  *token_.data<std::int64_t>() = token;
}

DeferredArRows::DeferredArRows()
    : weights_(mx::allocator::malloc(kArWeightBytes),
               mx::Shape{16, 20},
               mx::uint32),
      scales_(mx::allocator::malloc(kArMetadataBytes),
              mx::Shape{16, 5},
              mx::bfloat16),
      biases_(mx::allocator::malloc(kArMetadataBytes),
              mx::Shape{16, 5},
              mx::bfloat16) {
  std::memset(weights_.data<std::uint32_t>(), 0, kArWeightBytes);
  std::memset(scales_.data<std::uint16_t>(), 0, kArMetadataBytes);
  std::memset(biases_.data<std::uint16_t>(), 0, kArMetadataBytes);
}

std::shared_ptr<DeferredArRows> DeferredArRows::make() {
  return std::shared_ptr<DeferredArRows>(new DeferredArRows());
}

DeferredArPlanes DeferredArRows::planes() const {
  return {weights_, scales_, biases_};
}

void DeferredArRows::fill(const std::uint32_t* weights,
                          const std::uint16_t* scales,
                          const std::uint16_t* biases) {
  std::memcpy(weights_.data<std::uint32_t>(), weights, kArWeightBytes);
  std::memcpy(scales_.data<std::uint16_t>(), scales, kArMetadataBytes);
  std::memcpy(biases_.data<std::uint16_t>(), biases, kArMetadataBytes);
}

}  // namespace mtplx_native::ple_cpu_rows
