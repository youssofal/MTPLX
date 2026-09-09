// SPDX-License-Identifier: Apache-2.0
//
// Host entry point for the SPLIT-K decode variant of MTPLX's direct-index
// sparse-GQA attention (see steel_qsa_sparse_gqa_decode.h for provenance,
// the algorithm, and why decode needs a KV split at all).

#pragma once

#include <string>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace mtplx_native {

/// Empty when the call is on the supported contract; otherwise a precise,
/// caller-facing reason.  Exposed so the Python lane can gate without
/// catching an exception.
std::string qsa_sparse_gqa_decode_unsupported_reason(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_blocks, const mx::array &query_offset,
    float scale, int key_length, int key_tile, int dimension_tile,
    int key_splits, mx::StreamOrDevice s = {});

/// queries       [1, 24, M, 256]   fp16/bf16, last dim contiguous
/// keys/values   [1,  2, cap, 256] the FULL cache backing, same dtype
/// selected      [1,  1, M, 512]   uint32 or int32, the argpartition output
///                                 in ITS OWN order (never re-sorted)
/// query_offset  [1] int32         absolute position of query row 0; a device
///                                 buffer so a tensor-valued cache offset
///                                 never has to be read on the host
/// key_length    logical tokens in the cache (<= cap); -1 means keys.shape(2)
/// key_splits    target number of KV splits; clamped to the tile count and
///                                 then rounded so no split is empty
/// returns       [1, 24, M, 256]   same dtype as queries
mx::array qsa_sparse_gqa_decode(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_blocks, const mx::array &query_offset,
    float scale, int key_length = -1, int key_tile = 128,
    int dimension_tile = 32, int key_splits = 8, mx::StreamOrDevice s = {});

/// The host half of the split geometry, exported so the Python lane, the
/// tests and the harness can size the partial buffer without duplicating the
/// arithmetic.  Writes ``n_tiles``, ``tiles_per_split`` and ``n_splits``.
void qsa_sparse_gqa_decode_split_geometry(int selected_tokens, int key_tile,
                                          int key_splits, int *n_tiles,
                                          int *tiles_per_split, int *n_splits);

} // namespace mtplx_native
