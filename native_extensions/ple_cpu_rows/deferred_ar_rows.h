// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <tuple>

#include "mlx/array.h"

namespace mtplx_native::ple_cpu_rows {

namespace mx = mlx::core;

constexpr std::size_t kArRows = 16;
constexpr std::size_t kArWeightValues = kArRows * 20;
constexpr std::size_t kArMetadataValues = kArRows * 5;

using DeferredArPlanes = std::tuple<mx::array, mx::array, mx::array>;

class DeferredArToken final {
 public:
  static std::shared_ptr<DeferredArToken> make();

  DeferredArToken(const DeferredArToken&) = delete;
  DeferredArToken& operator=(const DeferredArToken&) = delete;

  mx::array array() const;
  void fill(std::int64_t token);

 private:
  DeferredArToken();

  mx::array token_;
};

class DeferredArRows final {
 public:
  static std::shared_ptr<DeferredArRows> make();

  DeferredArRows(const DeferredArRows&) = delete;
  DeferredArRows& operator=(const DeferredArRows&) = delete;

  DeferredArPlanes planes() const;
  void fill(const std::uint32_t* weights,
            const std::uint16_t* scales,
            const std::uint16_t* biases);

 private:
  DeferredArRows();

  mx::array weights_;
  mx::array scales_;
  mx::array biases_;
};

}  // namespace mtplx_native::ple_cpu_rows
