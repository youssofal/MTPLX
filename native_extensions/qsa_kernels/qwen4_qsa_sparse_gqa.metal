// SPDX-License-Identifier: Apache-2.0
//
// Vendored into MTPLX from oMLX (https://github.com/jundot/omlx), PR #3244,
// revision dc312e6e905e03d21ef0c4a86289cbfa2cf857cc.
//
// MTPLX ships only the one measured production specialization
// (BK=64, DC=64) in fp16 and bf16. The oMLX file also instantiates
// (128,32), (256,32) and (128,64); packaging kernels that no MTPLX caller
// can request only widens the Metal-compilation surface that must be
// qualified on the first M3 run.

// Include order is load-bearing: Steel's attention header provides Limits
// used by the specialized Qwen kernel. The params struct is defined here
// (Metal has no access to the C++ translation unit) and MUST match the
// layout in qwen4_qsa_sparse_gqa.cpp byte-for-byte.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h"

struct Qwen4QSASparseGQAParams {
  int B;
  int q_heads;
  int kv_heads;
  int qL;
  int kL;
  int topk;
  int gqa_factor;
  int q_offset;

  float scale;

  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t Topk_strides[3];
  int64_t O_strides[3];
};

#include "kernels/steel_qwen4_qsa_sparse_gqa.h"
// clang-format on

#define instantiate_qwen4_sparse_gqa(tname, dtype, bk, dc)                     \
  instantiate_kernel("qwen4_qsa_sparse_gqa_" #tname "_bk" #bk "_dc" #dc        \
                     "_gqa12_hp16_d256_wm2",                                   \
                     qwen4_qsa_sparse_gqa_attention, dtype, bk, dc, 12, 16,    \
                     256, 2, uint, float)

instantiate_qwen4_sparse_gqa(float16, half, 64, 64);
instantiate_qwen4_sparse_gqa(bfloat16, bfloat16_t, 64, 64);
