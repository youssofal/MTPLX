// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "mlx/array.h"
#include "mlx/stream.h"

#include "request_state.h"
#include "sidecar_producer.h"

namespace mtplx_native::ple_cpu_rows {

namespace mx = mlx::core;

using CpuRowsArrays = std::tuple<mx::array, mx::array, mx::array>;

// The payload is already in the exact packed row layout consumed by the
// ordinary model dequantizer: 64 rows of 80-byte U32 weights, 10-byte BF16
// scales, and 10-byte BF16 biases.  This adapter performs no dequantization.
CpuRowsArrays make_cpu_rows(const PackedPayload& payload,
                            int delay_ms = 0,
                            bool force_fail = false,
                            bool cancelled = false);

// Explicit real-sidecar factory.  The synthetic make_cpu_rows() control lane
// above remains independent; this factory captures an installed immutable
// SidecarProducer and a copied M4 token snapshot for one CPU-stream job.
CpuRowsArrays make_sidecar_rows(
    const std::shared_ptr<const SidecarProducer>& producer,
    const std::array<std::int64_t, host_provider::kHistoryRows>& previous,
    const std::array<std::int64_t, host_provider::kInputRows>& ids);

}  // namespace mtplx_native::ple_cpu_rows
