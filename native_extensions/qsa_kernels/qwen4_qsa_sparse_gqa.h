// SPDX-License-Identifier: Apache-2.0
//
// Vendored into MTPLX from oMLX (https://github.com/jundot/omlx), PR #3244,
// revision dc312e6e905e03d21ef0c4a86289cbfa2cf857cc. The oMLX namespace
// ``omlx::glm_kernels`` is renamed to ``mtplx::qsa_kernels`` and the tile
// defaults are pinned to the one measured production specialization so a
// co-installed oMLX cannot collide with this module's Metal library name.

#pragma once

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mx = mlx::core;

namespace mtplx::qsa_kernels {

// Exact Qwen4 QSA main attention over query-specific selected four-token
// blocks. ``selected_blocks`` is ``[1, 1, qL, 512]`` uint32 holding a
// chronological VALID PREFIX of block ids; validity is positional, derived
// in-kernel as ``min(512, (q_offset + row + 1) / 4)``. Callers own that
// invariant (see mtplx/kernels/qsa_prefill_direct.py).
mx::array qwen4_qsa_sparse_gqa_attention(
    const mx::array &queries, const mx::array &keys, const mx::array &values,
    const mx::array &selected_blocks, float scale, int q_offset,
    int key_tile = 64, int dimension_tile = 64, mx::StreamOrDevice s = {});

} // namespace mtplx::qsa_kernels
