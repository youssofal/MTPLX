"""TensorOps (NAX) flash-decoding SDPA for the MTP verify window (hyper K2, 2026-09-01).

Successor to :mod:`mtplx.kernels.sdpa_nax_tile`. Same contract as
:func:`mtplx.kernels.sdpa_gqa_packed.sdpa_gqa_packed_tail` (tail-causal, whole-capacity
buffers, bf16/fp16 in, same dtype out, split-KV partials merged by the shared reduce
kernel), same NAXFrag register layout (cider / mlx steel convention). What changes:

* **No V staging.** PV runs the ``transpose_right=false`` descriptor and loads V
  fragments straight from device memory the way mlx's ``attention_nax`` does; the
  16 KB threadgroup transpose scatter and its two barriers per tile are gone.
* **Key-split inside the threadgroup.** A threadgroup is ``NSGM x KS`` simdgroups: KS
  simdgroups interleave 32-key tiles of one contiguous KV chunk and merge their
  online-softmax state through threadgroup memory at the end, so one partial per
  threadgroup reaches DRAM instead of one per simdgroup. At the tile kernel's block
  count that was 50 MB of partials per layer plus the reduce read-back; at
  KS=4 / 128 blocks it is 6 MB.
* **Q optionally staged once** in threadgroup memory (``MTPLX_NAX_FLASH_QTG=1``)
  instead of re-read from device per tile.

Gated: returns None on any contract miss; env kill-switch MTPLX_NAX_FLASH=0.
Tunables (cells only): MTPLX_NAX_FLASH_KS (default 1), MTPLX_NAX_FLASH_BLOCKS,
MTPLX_NAX_FLASH_QTG.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from ..nax_verify import nax_available
from .sdpa_gqa_packed import _paged_reduce_kernel

nax_flash_bail_counts: dict[str, int] = {}
# Successful dispatches, so /health can show the route engaged (not only
# its bails) — the one-line receipt the #459 reports needed.
nax_flash_dispatch_counts: dict[str, int] = {}

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr short kElemsPerFrag = 8;
constant constexpr short kElemCols = 4;
constant constexpr short kElemRowsJump = 8;

// NAXFrag lane -> (col, row) coordinate map (cider / mlx steel convention).
inline short2 nax_get_coord(ushort lid) {
  short qid = short(lid >> 2);
  short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
  short fn = ((qid & 2) | (short(lid) & 1)) * 4;
  return short2{fn, fm};
}
"""

# Template params (mx.fast.metal_kernel): InT, D, QL, GQA_F, KS, QTG.
# Inputs: queries [HQ*QL, D] flat, keys/values [HK, kcap, D], offset(i32[1]), kcap,
#         scale(f32), blocks(i32[1]).
# Outputs: partials [1, HQ, QL, blocks, D] InT; sums/maxs [1, HQ, QL, blocks] f32.
_SOURCE = r"""
    constexpr int TK = 32;                       // keys per tile (N of the NT matmul)
    constexpr int MROWS = 16;                    // M rows per simdgroup
    constexpr int LIVE = GQA_F * QL;             // live M rows per KV head
    constexpr int NSGM = (LIVE + MROWS - 1) / MROWS;
    constexpr int NSGS = NSGM * KS;
    constexpr int NTHREADS = NSGS * 32;
    constexpr int DFRAGS = D / 16;
    constexpr int NGROUPS = D / 32;              // PV output column groups
    constexpr int PHG = 2;                       // merge phase width: 2 groups = 64 dims
    constexpr int NPHASES = NGROUPS / PHG;

    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = lid >> 5;
    const ushort lane = ushort(lid & 31);
    const int mg = int(sg) / KS;                 // M simdgroup index
    const int ks = int(sg) % KS;                 // key-split index
    const int kv_head = int(threadgroup_position_in_grid.x);
    const int block_idx = int(threadgroup_position_in_grid.z);
    const int n_blocks = int(blocks[0]);

    const int n_kv = static_cast<int>(offset[0]);
    const int tail_lo = n_kv - QL;

    const int chunk = ((n_kv + n_blocks - 1) / n_blocks + TK - 1) / TK * TK;
    const int kv_start = block_idx * chunk;
    const int kv_end = metal::min(kv_start + chunk, n_kv);

    const short2 sc = nax_get_coord(lane);
    const int m_base = mg * MROWS;

    threadgroup float tg_m[NSGS * MROWS];
    threadgroup float tg_l[NSGS * MROWS];
    threadgroup float tg_o[NSGM * MROWS * (PHG * 32)];
    threadgroup InT tg_q[QTG ? (NSGM * MROWS * D) : 4];

    const device InT* k_head = keys + (size_t)kv_head * kcap * D;
    const device InT* v_head = values + (size_t)kv_head * kcap * D;
    const device InT* q_head = queries + (size_t)kv_head * LIVE * D;

    if (QTG) {
      for (int idx4 = int(lid); idx4 < NSGM * MROWS * (D / 4); idx4 += NTHREADS) {
        const int r = idx4 / (D / 4);
        const int c4 = idx4 % (D / 4);
        vec<InT, 4> q4 = vec<InT, 4>(InT(0));
        if (r < LIVE) {
          q4 = reinterpret_cast<const device vec<InT, 4>*>(q_head + (size_t)r * D + c4 * 4)[0];
        }
        reinterpret_cast<threadgroup vec<InT, 4>*>(tg_q + (size_t)r * D + c4 * 4)[0] = q4;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    constexpr auto desc_nt = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto desc_nn = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, false, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc_nt, metal::execution_simdgroup> mm_qk;
    mpp::tensor_ops::matmul2d<desc_nn, metal::execution_simdgroup> mm_pv;

    auto ct_qa = mm_qk.get_left_input_cooperative_tensor<InT, InT, float>();
    auto ct_kb = mm_qk.get_right_input_cooperative_tensor<InT, InT, float>();
    auto ct_s = mm_qk.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_qa)>, metal::remove_addrspace_t<decltype(ct_kb)>, float>();
    auto ct_pa = mm_pv.get_left_input_cooperative_tensor<InT, InT, float>();
    auto ct_vb = mm_pv.get_right_input_cooperative_tensor<InT, InT, float>();
    auto ct_o = mm_pv.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(ct_pa)>, metal::remove_addrspace_t<decltype(ct_vb)>, float>();

    float row_m[2] = {-1e38f, -1e38f};
    float row_l[2] = {0.0f, 0.0f};
    float o_frag[NGROUPS][2][kElemsPerFrag];
    for (int g = 0; g < NGROUPS; g++)
      for (int h = 0; h < 2; h++)
        for (int i = 0; i < kElemsPerFrag; i++) o_frag[g][h][i] = 0.0f;

    for (int t0 = kv_start + ks * TK; t0 < kv_end; t0 += KS * TK) {
      const bool tile_full = (t0 + TK) <= kv_end;

      // ── QK: S[16,32] = Q[16,256] x Ktile[32,256]^T ──
      for (short i = 0; i < 2 * kElemsPerFrag; i++) ct_s[i] = 0.0f;
      for (int kk = 0; kk < DFRAGS; kk++) {
        for (short i = 0; i < 2; i++) {
          const int mr = m_base + sc.y + i * kElemRowsJump;
          vec<InT, 4> qq = vec<InT, 4>(InT(0));
          if (QTG) {
            qq = reinterpret_cast<const threadgroup vec<InT, 4>*>(
                tg_q + (size_t)mr * D + kk * 16 + sc.x)[0];
          } else if (mr < LIVE) {
            qq = reinterpret_cast<const device vec<InT, 4>*>(
                q_head + (size_t)mr * D + kk * 16 + sc.x)[0];
          }
          for (short j = 0; j < kElemCols; j++) ct_qa[i * kElemCols + j] = qq[j];
        }
        for (short hh = 0; hh < 2; hh++) {
          for (short i = 0; i < 2; i++) {
            const int kr = t0 + hh * 16 + sc.y + i * kElemRowsJump;
            vec<InT, 4> kk4 = vec<InT, 4>(InT(0));
            if (tile_full || kr < kv_end) {
              kk4 = reinterpret_cast<const device vec<InT, 4>*>(
                  k_head + (size_t)kr * D + kk * 16 + sc.x)[0];
            }
            for (short j = 0; j < kElemCols; j++)
              ct_kb[hh * kElemsPerFrag + i * kElemCols + j] = kk4[j];
          }
        }
        mm_qk.run(ct_qa, ct_kb, ct_s);
      }

      // ── in-register masked online softmax ──
      float s_p[2][kElemsPerFrag];
      float tile_f[2];
      {
        float tmax[2] = {-1e38f, -1e38f};
        for (short i = 0; i < 2; i++) {
          const int m_local = m_base + sc.y + i * kElemRowsJump;
          const int row_limit = tail_lo + (m_local % QL);
          const bool live = m_local < LIVE;
          for (short hh = 0; hh < 2; hh++) {
            for (short j = 0; j < kElemCols; j++) {
              const int gp = t0 + hh * 16 + sc.x + j;
              const float raw = ct_s[hh * kElemsPerFrag + i * kElemCols + j];
              const bool vis = live && gp < kv_end && gp <= row_limit;
              const float s = vis ? raw * scale : -1e38f;
              s_p[hh][i * kElemCols + j] = s;
              tmax[i] = metal::max(tmax[i], s);
            }
          }
        }
        for (short i = 0; i < 2; i++) {
          tmax[i] = metal::max(tmax[i], simd_shuffle_xor(tmax[i], ushort(1)));
          tmax[i] = metal::max(tmax[i], simd_shuffle_xor(tmax[i], ushort(8)));
          const float new_m = metal::max(row_m[i], tmax[i]);
          tile_f[i] = metal::exp(row_m[i] - new_m);
          float tsum = 0.0f;
          for (short hh = 0; hh < 2; hh++) {
            for (short j = 0; j < kElemCols; j++) {
              const float s = s_p[hh][i * kElemCols + j];
              const float p = (s > -1e37f) ? metal::exp(s - new_m) : 0.0f;
              s_p[hh][i * kElemCols + j] = p;
              tsum += p;
            }
          }
          tsum += simd_shuffle_xor(tsum, ushort(1));
          tsum += simd_shuffle_xor(tsum, ushort(8));
          row_m[i] = new_m;
          row_l[i] = row_l[i] * tile_f[i] + tsum;
        }
      }
      for (short i = 0; i < 2; i++) {
        const float f = tile_f[i];
        for (int g = 0; g < NGROUPS; g++)
          for (short hh = 0; hh < 2; hh++)
            for (short j = 0; j < kElemCols; j++)
              o_frag[g][hh][i * kElemCols + j] *= f;
      }

      // ── PV: O[16, 32g..] += P[16,32] x V[32 keys, 32 dims] (NN, V straight from device) ──
      for (int g = 0; g < NGROUPS; g++) {
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_o[i] = o_frag[g][0][i];
          ct_o[kElemsPerFrag + i] = o_frag[g][1][i];
        }
        for (int kk = 0; kk < 2; kk++) {
          for (short i = 0; i < 2; i++)
            for (short j = 0; j < kElemCols; j++)
              ct_pa[i * kElemCols + j] = InT(s_p[kk][i * kElemCols + j]);
          for (short hh = 0; hh < 2; hh++) {
            for (short i = 0; i < 2; i++) {
              const int kr = t0 + kk * 16 + sc.y + i * kElemRowsJump;
              vec<InT, 4> v4 = vec<InT, 4>(InT(0));
              if (tile_full || kr < kv_end) {
                v4 = reinterpret_cast<const device vec<InT, 4>*>(
                    v_head + (size_t)kr * D + g * 32 + hh * 16 + sc.x)[0];
              }
              for (short j = 0; j < kElemCols; j++)
                ct_vb[hh * kElemsPerFrag + i * kElemCols + j] = v4[j];
            }
          }
          mm_pv.run(ct_pa, ct_vb, ct_o);
        }
        for (short i = 0; i < kElemsPerFrag; i++) {
          o_frag[g][0][i] = ct_o[i];
          o_frag[g][1][i] = ct_o[kElemsPerFrag + i];
        }
      }
    }

    // ── merge the KS key-split simdgroups of each M group through threadgroup memory ──
    if (KS > 1) {
      if ((lane & 0x9) == 0) {
        for (short i = 0; i < 2; i++) {
          const int r = sc.y + i * kElemRowsJump;
          tg_m[(mg * KS + ks) * MROWS + r] = row_m[i];
          tg_l[(mg * KS + ks) * MROWS + r] = row_l[i];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float f_self[2];
      for (short i = 0; i < 2; i++) {
        const int r = sc.y + i * kElemRowsJump;
        float M = -1e38f;
        for (int s = 0; s < KS; s++) M = metal::max(M, tg_m[(mg * KS + s) * MROWS + r]);
        float L = 0.0f;
        for (int s = 0; s < KS; s++) {
          const float ms = tg_m[(mg * KS + s) * MROWS + r];
          L += tg_l[(mg * KS + s) * MROWS + r] * metal::exp(ms - M);
        }
        f_self[i] = metal::exp(row_m[i] - M);
        row_m[i] = M;
        row_l[i] = L;
      }
      for (int g = 0; g < NGROUPS; g++)
        for (short hh = 0; hh < 2; hh++)
          for (short i = 0; i < 2; i++)
            for (short j = 0; j < kElemCols; j++)
              o_frag[g][hh][i * kElemCols + j] *= f_self[i];
      for (int p = 0; p < NPHASES; p++) {
        for (int s = 1; s < KS; s++) {
          if (ks == s) {
            for (int gg = 0; gg < PHG; gg++)
              for (short hh = 0; hh < 2; hh++)
                for (short i = 0; i < 2; i++) {
                  const int r = sc.y + i * kElemRowsJump;
                  threadgroup float* dst = tg_o + (size_t)(mg * MROWS + r) * (PHG * 32)
                      + gg * 32 + hh * 16 + sc.x;
                  for (short j = 0; j < kElemCols; j++)
                    dst[j] = o_frag[p * PHG + gg][hh][i * kElemCols + j];
                }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (ks == 0) {
            for (int gg = 0; gg < PHG; gg++)
              for (short hh = 0; hh < 2; hh++)
                for (short i = 0; i < 2; i++) {
                  const int r = sc.y + i * kElemRowsJump;
                  const threadgroup float* src = tg_o + (size_t)(mg * MROWS + r) * (PHG * 32)
                      + gg * 32 + hh * 16 + sc.x;
                  for (short j = 0; j < kElemCols; j++)
                    o_frag[p * PHG + gg][hh][i * kElemCols + j] += src[j];
                }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }

    if (ks == 0) {
      for (short i = 0; i < 2; i++) {
        const int m_local = m_base + sc.y + i * kElemRowsJump;
        if (m_local >= LIVE) continue;
        const int hq_row = kv_head * LIVE + m_local;
        device InT* prow = partials + ((size_t)hq_row * n_blocks + block_idx) * D;
        for (int g = 0; g < NGROUPS; g++)
          for (short hh = 0; hh < 2; hh++)
            for (short j = 0; j < kElemCols; j++)
              prow[g * 32 + hh * 16 + sc.x + j] = InT(o_frag[g][hh][i * kElemCols + j]);
      }
      if ((lane & 0x9) == 0) {
        for (short i = 0; i < 2; i++) {
          const int m_local = m_base + sc.y + i * kElemRowsJump;
          if (m_local < LIVE) {
            const int hq_row = kv_head * LIVE + m_local;
            sums[hq_row * n_blocks + block_idx] = row_l[i];
            maxs[hq_row * n_blocks + block_idx] = (row_l[i] > 0.0f) ? row_m[i] : -1e38f;
          }
        }
      }
    }
"""


@lru_cache(maxsize=4)
def _nax_flash_kernel():
    try:
        return mx.fast.metal_kernel(
            name="mtplx_nax_flash_partials",
            input_names=["queries", "keys", "values", "offset", "kcap",
                         "scale", "blocks"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER,
            source=_SOURCE,
        )
    except Exception:  # noqa: BLE001 — toolchain without Metal4/MPP support
        return None


def _bail(reason: str):
    nax_flash_bail_counts[reason] = nax_flash_bail_counts.get(reason, 0) + 1
    if os.environ.get("MTPLX_NAX_FLASH_DEBUG"):
        print(f"[nax-flash bail] {reason}")
    return None


def _default_blocks(capacity: int) -> int:
    # 2026-09-01 walk bench, 72.7k QL4: KS=1/512 blocks 1.015 ms/layer beat KS=4/128 (1.080),
    # KS=2/256 (1.069), KS=4/64 (1.055), KS=8/64 (1.195). Shorter contexts: 16k/128k sweep pins.
    if capacity >= 65536:
        return 512
    if capacity >= 16384:
        return 256
    return 128


def sdpa_nax_flash(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    offset: int | mx.array,
    scale: float,
) -> mx.array | None:
    """TensorOps flash-decoding tail-causal SDPA. Same contract as the packed kernel."""
    if os.environ.get("MTPLX_NAX_FLASH", "1") == "0":
        return _bail("env_disabled")
    if not mx.metal.is_available():
        return _bail("metal_unavailable")
    if not nax_available():
        return _bail("gpu_family_or_os")
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return _bail("ndim")
    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1 or d != 256:
        return _bail("shape_gate")
    hk = int(keys.shape[1])
    capacity = int(keys.shape[2])
    if hk <= 0 or hq % hk:
        return _bail("gqa_heads")
    gqa_factor = hq // hk
    if gqa_factor * q_len > 64:
        return _bail("m_rows_gt_64")
    if q_len < 1 or q_len > 10:
        return _bail("q_len")
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return _bail("query_dtype")
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return _bail("kv_dtype_mismatch")
    if int(values.shape[1]) != hk or int(values.shape[2]) != capacity \
            or int(keys.shape[3]) != d or int(values.shape[3]) != d:
        return _bail("kv_layout_mismatch")

    if isinstance(offset, mx.array):
        if offset.size != 1:
            return _bail("offset_shape")
        offset_arr = offset.astype(mx.int32).reshape(1)
    else:
        offset_int = int(offset)
        if offset_int <= 0 or offset_int > capacity:
            return _bail("offset_range")
        offset_arr = mx.array([offset_int], dtype=mx.int32)

    ks = int(os.environ.get("MTPLX_NAX_FLASH_KS", "1") or 1)
    ks = max(1, min(8, ks))
    qtg = 1 if os.environ.get("MTPLX_NAX_FLASH_QTG", "0") == "1" else 0
    blocks = int(os.environ.get("MTPLX_NAX_FLASH_BLOCKS", "0") or 0) or _default_blocks(capacity)
    blocks = max(32, (blocks // 32) * 32)
    blocks_arr = mx.array([blocks], dtype=mx.int32)

    kernel = _nax_flash_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return _bail("kernel_unavailable")

    nsgm = (gqa_factor * q_len + 15) // 16
    nthreads = 32 * nsgm * ks
    partial_shape = (bsz, hq, q_len, blocks, d)
    stats_shape = (bsz, hq, q_len, blocks)
    try:
        partials, sums, maxs = kernel(
            inputs=[queries, keys, values, offset_arr, capacity,
                    float(scale), blocks_arr],
            template=[
                ("InT", queries.dtype),
                ("D", d),
                ("QL", q_len),
                ("GQA_F", gqa_factor),
                ("KS", ks),
                ("QTG", qtg),
            ],
            grid=(hk * nthreads, 1, blocks),
            threadgroup=(nthreads, 1, 1),
            output_shapes=[partial_shape, stats_shape, stats_shape],
            output_dtypes=[queries.dtype, mx.float32, mx.float32],
        )
    except Exception as exc:  # noqa: BLE001 — dispatch/compile failure => stock fallback
        return _bail(f"dispatch_failed: {type(exc).__name__}: {str(exc)[:2000]}")

    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", queries.dtype),
            ("V", d),
        ],
        grid=(bsz * hq * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(bsz, hq, q_len, d)],
        output_dtypes=[queries.dtype],
    )
    nax_flash_dispatch_counts["dispatched"] = nax_flash_dispatch_counts.get("dispatched", 0) + 1
    return out
