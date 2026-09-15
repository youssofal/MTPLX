// SPDX-License-Identifier: Apache-2.0
//
// Shared host/device parameter blocks for the SPLIT-K (KV-split) variant of
// the QSA direct-index sparse-GQA kernel -- the decode geometries, M=4
// (fixed-M4 verify) and M=1 (single-row draft/decode).
//
// Two blocks, because the lane is two dispatches: the split pass writes one
// unnormalised online-softmax state per (query row, KV head, KV split), and
// the merge pass reduces those states into the attention output.
//
// Both are included by the C++ encoder AND by the MSL kernel, so the layout
// can never drift between the two sides.  A silent drift here is wrong
// attention, not a crash, which is why the static_asserts below are compiled
// by both.

#pragma once

#ifndef __METAL_VERSION__
#include <cstdint>
#endif

/// Pass 1: one threadgroup per (query row, KV head, KV split).
struct MtplxQsaSparseGqaDecodeParams {
  int q_heads;    ///< 24
  int kv_heads;   ///< 2
  int qL;         ///< query rows in this call (M): 4 for verify, 1 for draft
  int kL;         ///< logical key length (NOT the cache backing capacity)
  int topk;       ///< selected blocks per row (512)
  int gqa_factor; ///< 12

  /// Split geometry, all host-computed from ``topk``, ``compress_ratio`` and
  /// the compiled-in ``BK``.  ``n_splits * tiles_per_split >= n_tiles`` and
  /// ``(n_splits - 1) * tiles_per_split < n_tiles``: no split is ever empty,
  /// so every threadgroup in the grid does real work.
  int n_tiles;         ///< ceil((topk*ratio + ratio-1) / BK)
  int tiles_per_split; ///< >= 1
  int n_splits;        ///< ceil(n_tiles / tiles_per_split), == grid.z
  int partial_ld;      ///< head_dim + 2: the row carries [O(256) | m | l]

  float scale; ///< 1/sqrt(head_dim); the kernel folds M_LOG2E in itself
  int _pad;    ///< explicit: keeps the int64 block 8-byte aligned on both sides

  int64_t Q_strides[3];    ///< (B, H, L); last dim is unit stride
  int64_t K_strides[3];    ///< (B, H_kv, L) into the FULL cache backing
  int64_t V_strides[3];    ///< (B, H_kv, L) into the FULL cache backing
  int64_t Topk_strides[3]; ///< (B, 1, M); last dim is unit stride
};

// 11 ints (``_pad`` included) + 1 float = 48, already 8-byte aligned, then
// 4 x 3 x int64 = 96.  Pin it: host and device MUST agree byte for byte.
static_assert(sizeof(MtplxQsaSparseGqaDecodeParams) == 144,
              "MtplxQsaSparseGqaDecodeParams layout drifted host vs device");
static_assert(alignof(MtplxQsaSparseGqaDecodeParams) == 8,
              "MtplxQsaSparseGqaDecodeParams alignment drifted");

/// Pass 2: one threadgroup per (query head, query row); head_dim threads.
struct MtplxQsaSparseGqaMergeParams {
  int q_heads;    ///< 24
  int qL;         ///< query rows (M)
  int head_dim;   ///< 256
  int n_splits;   ///< how many partial states to merge
  int partial_ld; ///< head_dim + 2
  int _pad;       ///< explicit alignment

  int64_t O_strides[3]; ///< (B, H, L) of the [1, 24, M, 256] output
};

static_assert(sizeof(MtplxQsaSparseGqaMergeParams) == 48,
              "MtplxQsaSparseGqaMergeParams layout drifted host vs device");
static_assert(alignof(MtplxQsaSparseGqaMergeParams) == 8,
              "MtplxQsaSparseGqaMergeParams alignment drifted");
