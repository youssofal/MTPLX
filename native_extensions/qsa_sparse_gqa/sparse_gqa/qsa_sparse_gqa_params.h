// SPDX-License-Identifier: Apache-2.0
//
// Shared host/device parameter block for the QSA direct-index sparse-GQA
// kernel.  Both qsa_sparse_gqa.cpp (C++) and qsa_sparse_gqa.metal (MSL)
// include this file, so the layout can never drift between the encoder and
// the kernel -- the failure mode oMLX's duplicated struct leaves open.

#pragma once

#ifndef __METAL_VERSION__
#include <cstdint>
#endif

struct MtplxQsaSparseGqaParams {
  int B;          ///< batch (always 1 on the supported geometry)
  int q_heads;    ///< 24
  int kv_heads;   ///< 2
  int qL;         ///< query rows in this call (M)
  int kL;         ///< logical key length (NOT the cache backing capacity)
  int topk;       ///< selected blocks per row (512)
  int gqa_factor; ///< 12
  int q_offset;   ///< absolute position of query row 0

  float scale; ///< 1/sqrt(head_dim), pre-M_LOG2E in the kernel
  int _pad;    ///< explicit: keeps the int64 block 8-byte aligned on both sides

  int64_t Q_strides[3];    ///< (B, H, L); last dim is unit stride
  int64_t K_strides[3];    ///< (B, H_kv, L) into the FULL cache backing
  int64_t V_strides[3];    ///< (B, H_kv, L) into the FULL cache backing
  int64_t Topk_strides[3]; ///< (B, 1, M); last dim is unit stride
  int64_t O_strides[3];    ///< (B, H, L)
};

// The encoder writes this struct with set_bytes and the kernel reads it as a
// `constant` pointer, so host and device MUST agree on the layout byte for
// byte.  A mismatch is silent wrong attention, not a crash, so pin it here:
// this header is compiled by BOTH sides, and the assert fires on whichever
// one drifts.  8 ints + float + pad = 40 B, then 5 x 3 x int64 = 120 B.
static_assert(sizeof(MtplxQsaSparseGqaParams) == 160,
              "MtplxQsaSparseGqaParams layout drifted between host and device");
static_assert(alignof(MtplxQsaSparseGqaParams) == 8,
              "MtplxQsaSparseGqaParams alignment drifted");
