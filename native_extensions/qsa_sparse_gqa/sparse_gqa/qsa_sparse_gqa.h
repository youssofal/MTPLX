// SPDX-License-Identifier: Apache-2.0
//
// Host entry point for MTPLX's port of oMLX's direct-index sparse-GQA
// attention (see steel_qsa_sparse_gqa.h for provenance and algorithm).

#pragma once

#include <string>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace mtplx_native {

/// Empty when the call is on the supported contract; otherwise a precise,
/// caller-facing reason.  Exposed so the Python lane can gate without
/// catching an exception (mirrors qsa_prefill_flash's _unsupported_reason).
std::string qsa_sparse_gqa_unsupported_reason(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_blocks, float scale, int q_offset,
    int key_length, int key_tile, int dimension_tile, mx::StreamOrDevice s = {});

/// queries      [1, 24, M, 256]      fp16/bf16, last dim contiguous
/// keys/values  [1,  2, cap, 256]    the FULL cache backing, same dtype
/// selected     [1,  1, M, 512]      uint32 or int32, chronological block ids
/// key_length   logical tokens in the cache (<= cap); -1 means keys.shape(2)
/// returns      [1, 24, M, 256]      same dtype as queries
mx::array qsa_sparse_gqa_attention(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_blocks, float scale, int q_offset,
    int key_length = -1, int key_tile = 128, int dimension_tile = 32,
    mx::StreamOrDevice s = {});

} // namespace mtplx_native
