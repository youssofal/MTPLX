// SPDX-License-Identifier: Apache-2.0

// Include order is load-bearing: mlx's utils.h provides Limits and the
// instantiate_kernel macro used by the specialized QSA kernel.
// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "sparse_gqa/steel_qsa_sparse_gqa.h"
// clang-format on

// MTPLX's `_select_eager` emits int32 block ids; oMLX's ABI is uint32.  Both
// are instantiated so the lane never pays an 8 MB astype per layer per chunk.
#define instantiate_qsa_sparse_gqa(tname, dtype, iname, itype, bk, dc)          \
  instantiate_kernel("mtplx_qsa_sparse_gqa_" #tname "_" #iname "_bk" #bk "_dc" #dc \
                     "_gqa12_hp16_d256_wm2",                                    \
                     mtplx_qsa_sparse_gqa_attention, dtype, bk, dc, 12, 16,     \
                     256, 2, itype, float)

#define instantiate_qsa_sparse_gqa_tiles(tname, dtype, iname, itype)           \
  instantiate_qsa_sparse_gqa(tname, dtype, iname, itype, 128, 32);             \
  instantiate_qsa_sparse_gqa(tname, dtype, iname, itype, 256, 32);             \
  instantiate_qsa_sparse_gqa(tname, dtype, iname, itype, 64, 64);              \
  instantiate_qsa_sparse_gqa(tname, dtype, iname, itype, 128, 64)

#define instantiate_qsa_sparse_gqa_dtype(tname, dtype)                         \
  instantiate_qsa_sparse_gqa_tiles(tname, dtype, uint32, uint);                \
  instantiate_qsa_sparse_gqa_tiles(tname, dtype, int32, int)

instantiate_qsa_sparse_gqa_dtype(float16, half);
instantiate_qsa_sparse_gqa_dtype(bfloat16, bfloat16_t);

// ---------------------------------------------------------------------------
// Split-K (KV-split) DECODE variant: M=4 verify and M=1 draft.  Same four
// (BK, DC) tiles as the prefill kernel so one harness sweep serves both, and
// the same {fp16, bf16} x {uint32, int32} matrix so no lane pays an astype.
// See steel_qsa_sparse_gqa_decode.h for why decode has to be split-K.
// ---------------------------------------------------------------------------
#include "sparse_gqa/steel_qsa_sparse_gqa_decode.h"

#define instantiate_qsa_sparse_gqa_decode(tname, dtype, iname, itype, bk, dc)  \
  instantiate_kernel("mtplx_qsa_sparse_gqa_decode_split_" #tname "_" #iname    \
                     "_bk" #bk "_dc" #dc "_gqa12_hp16_d256_wm2",               \
                     mtplx_qsa_sparse_gqa_decode_split, dtype, bk, dc, 12, 16, \
                     256, 2, itype, float)

#define instantiate_qsa_sparse_gqa_decode_tiles(tname, dtype, iname, itype)    \
  instantiate_qsa_sparse_gqa_decode(tname, dtype, iname, itype, 128, 32);      \
  instantiate_qsa_sparse_gqa_decode(tname, dtype, iname, itype, 256, 32);      \
  instantiate_qsa_sparse_gqa_decode(tname, dtype, iname, itype, 64, 64);       \
  instantiate_qsa_sparse_gqa_decode(tname, dtype, iname, itype, 128, 64)

#define instantiate_qsa_sparse_gqa_decode_dtype(tname, dtype)                  \
  instantiate_qsa_sparse_gqa_decode_tiles(tname, dtype, uint32, uint);         \
  instantiate_qsa_sparse_gqa_decode_tiles(tname, dtype, int32, int);           \
  instantiate_kernel("mtplx_qsa_sparse_gqa_decode_merge_" #tname "_d256",      \
                     mtplx_qsa_sparse_gqa_decode_merge, dtype, 256, float)

instantiate_qsa_sparse_gqa_decode_dtype(float16, half);
instantiate_qsa_sparse_gqa_decode_dtype(bfloat16, bfloat16_t);
