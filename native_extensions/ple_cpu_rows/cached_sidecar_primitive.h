// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "cached_sidecar_producer.h"
#include "mlx/array.h"
#include "mlx/stream.h"

namespace mtplx_native::ple_cpu_rows {

namespace mx = mlx::core;

using CachedRowsArrays = std::tuple<mx::array, mx::array, mx::array>;

// The ticket is present only when misses were admitted.  The arrays are
// ordinary MLX-owned U32/BF16 planes; the primitive copies the native packed
// rows into them on the installed CPU stream before the caller dequantizes.
struct CachedRowsSubmission final {
  std::optional<std::uint64_t> ticket;
  CachedRowsArrays arrays;
};

CachedRowsSubmission make_cached_sidecar_rows(
    const std::shared_ptr<CachedSidecarProducer>& producer,
    const CachedRowHandoff& handoff);

std::vector<CachedCompletion> drain_cached_completions(
    const std::shared_ptr<CachedSidecarProducer>& producer);

}  // namespace mtplx_native::ple_cpu_rows
