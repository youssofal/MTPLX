"""TensorOps (NAX) wide-M tail-causal SDPA — Phase 2.2 of the hyper campaign.

The whole verify walk for one KV head is a matrix problem: GQA_F(6) x QL(4)
query rows = M=24 rows sharing one K/V stream.  This kernel feeds that M block
through the M5 per-core Neural Accelerators via Metal 4 TensorOps
(``mpp::tensor_ops::matmul2d``), reading each K/V tile ONCE per simdgroup pair
instead of once per scalar row-chain.

Contract: identical to :func:`mtplx.kernels.sdpa_gqa_packed.sdpa_gqa_packed_tail`
(tail-causal, ``row j`` attends to ``n <= offset - q_len + j``; whole-capacity
buffers; bf16/fp16 in, same dtype out).  Split-KV partials are emitted in the
packed kernel's exact ``[1, HQ, QL, BLOCKS, D]`` layout so its reduce kernel is
reused verbatim — the fp32 LSE merge contract is shared, not reimplemented.

Fragment/coordinate mapping follows the proven M5 pattern from
Mininglamp-AI/cider (w8a8_matmul.metal, Apache-2.0): descriptor
``matmul2d_descriptor(16, 32, 16, false, true, true, multiply_accumulate)``
with the NAXFrag 8-elements-per-thread register layout.  PV reuses the SAME
transposed-right descriptor by staging V transposed in threadgroup memory.

Requires macOS >= 26.2 (bf16 TensorOps).  Gated: returns None on any contract
miss; env kill-switch MTPLX_NAX_TILE=0.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from ..nax_verify import nax_available
from .sdpa_gqa_packed import _blocks_for_capacity, _paged_reduce_kernel

nax_tile_bail_counts: dict[str, int] = {}

_HEADER = r"""
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr short kElemsPerFrag = 8;
constant constexpr short kElemCols = 4;
constant constexpr short kElemRowsJump = 8;

// NAXFrag lane -> (col, row) coordinate map (cider convention, M5 G17G).
inline short2 nax_get_coord(ushort lid) {
  short qid = short(lid >> 2);
  short fm = ((qid & 4) | ((short(lid) >> 1) & 3));
  short fn = ((qid & 2) | (short(lid) & 1)) * 4;
  return short2{fn, fm};
}
"""

# Template params injected by mx.fast.metal_kernel: InT, D, QL, GQA_F.
# Inputs: queries, keys, values, offset(i32[1]), kcap(int), scale(f32), blocks(int)
# Outputs: partials [1, HQ, QL, blocks, D] InT, sums/maxs [1, HQ, QL, blocks] f32
_SOURCE = r"""
    constexpr int TK = 32;               // keys per tile
    constexpr int MROWS = 16;            // M rows per simdgroup
    constexpr int LIVE = GQA_F * QL;     // live M rows per KV head
    constexpr int NSGS = (LIVE + MROWS - 1) / MROWS;  // simdgroups (M pad /16)
    constexpr int NTHREADS = NSGS * 32;
    constexpr int DFRAGS = D / 16;       // 16 K-frags along head dim
    constexpr int NGROUPS = D / 32;      // 8 PV output column groups

    const uint lid64 = thread_position_in_threadgroup.x;      // 0..NTHREADS-1
    const uint sg = lid64 >> 5;                               // simdgroup id
    const ushort lane = ushort(lid64 & 31);
    const int kv_head = int(threadgroup_position_in_grid.x);
    const int block_idx = int(threadgroup_position_in_grid.z);
    const int n_blocks = int(blocks[0]);

    const int n_kv = static_cast<int>(offset[0]);
    const int tail_lo = n_kv - QL;

    // Contiguous chunk per block, tile-aligned (32).
    const int chunk = ((n_kv + n_blocks - 1) / n_blocks + TK - 1) / TK * TK;
    const int kv_start = block_idx * chunk;
    const int kv_end = metal::min(kv_start + chunk, n_kv);

    const short2 sc = nax_get_coord(lane);
    const int m_base = int(sg) * MROWS;   // rows [m_base, m_base+16) of the 32-pad

    // Threadgroup staging: only the transposed V tile. S and P never leave
    // registers — the fragment coord map makes the exp'd S fragment the PV
    // left operand directly, and row max/sum reduce over the 4 lanes sharing
    // a row (lane bits 0 and 3) via two simd shuffles.
    threadgroup InT tg_vT[D * TK];              // 16 KB at D=256

    const device InT* k_head = keys + (size_t)kv_head * kcap * D;
    const device InT* v_head = values + (size_t)kv_head * kcap * D;
    // Q rows for this KV head start at global q-row kv_head*LIVE (row-major
    // [HQ*QL, D] flatten of [1, HQ, QL, D]).
    const device InT* q_head = queries + (size_t)kv_head * LIVE * D;

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16, 32, 16, false, true, true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> mm;

    auto ct_a = mm.get_left_input_cooperative_tensor<InT, InT, float>();
    auto ct_b = mm.get_right_input_cooperative_tensor<InT, InT, float>();
    // macOS 27 MPP SDK rejects address-space-qualified template operands
    // (issue #404); strip them like mlx's steel/gemm/nax.h does.
    auto ct_c = mm.get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>, float>();

    // Per-thread state: each thread carries TWO rows (i=0: sc.y, i=1: sc.y+8)
    // of online-softmax state, replicated across the 4 lanes sharing the row.
    float row_m[2] = {-1e38f, -1e38f};
    float row_l[2] = {0.0f, 0.0f};
    float o_frag[NGROUPS][2][kElemsPerFrag];
    for (int g = 0; g < NGROUPS; g++)
      for (int h = 0; h < 2; h++)
        for (int i = 0; i < kElemsPerFrag; i++) o_frag[g][h][i] = 0.0f;

    for (int t0 = kv_start; t0 < kv_end; t0 += TK) {
      // ── stage V^T tile (all threads, vectorized coalesced read;
      //     transpose scatter lands in threadgroup memory, which is cheap).
      //     DIRECTV=1 skips staging: PV loads V straight from device via the
      //     transposed coordinate map (scattered 2B loads, L1-served across
      //     the g-groups) and drops one barrier per tile. ──
      if (!DIRECTV) {
        for (int idx4 = int(lid64); idx4 < TK * (D / 4); idx4 += NTHREADS) {
          const int p = idx4 / (D / 4);    // key row in tile
          const int d4 = idx4 % (D / 4);
          const int gp = t0 + p;
          vec<InT, 4> v4 = vec<InT, 4>(InT(0));
          if (gp < kv_end) {
            const device vec<InT, 4>* src = reinterpret_cast<const device vec<InT, 4>*>(
                v_head + (size_t)gp * D + d4 * 4);
            v4 = src[0];
          }
          for (short e = 0; e < 4; e++)
            tg_vT[(d4 * 4 + e) * TK + p] = v4[e];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }

      // ── QK: S[16,32] += Q[16,256] x Ktile[32,256]^T ──
      // ct_c accumulates in place across all DFRAGS runs (single S tile).
      const bool tile_full = (t0 + TK) <= kv_end;
      for (short i = 0; i < 2 * kElemsPerFrag; i++) ct_c[i] = 0.0f;

      for (int kk = 0; kk < DFRAGS; kk++) {
        // left: Q fragment [16,16] at rows m_base+, dims kk*16 (zero-pad dead
        // rows). 4 contiguous bf16 per (i) = one 8-byte vector load.
        for (short i = 0; i < 2; i++) {
          const int mr = m_base + sc.y + i * kElemRowsJump;
          if (mr < LIVE) {
            const device vec<InT, 4>* qv = reinterpret_cast<const device vec<InT, 4>*>(
                q_head + (size_t)mr * D + kk * 16 + sc.x);
            const vec<InT, 4> qq = qv[0];
            for (short j = 0; j < kElemCols; j++)
              ct_a[i * kElemCols + j] = qq[j];
          } else {
            for (short j = 0; j < kElemCols; j++)
              ct_a[i * kElemCols + j] = InT(0);
          }
        }
        // right: K tile halves [16,16] rows t0+, dims kk*16 (transpose_b)
        for (short hh = 0; hh < 2; hh++) {
          for (short i = 0; i < 2; i++) {
            const int kr = t0 + hh * 16 + sc.y + i * kElemRowsJump;
            if (tile_full || kr < kv_end) {
              const device vec<InT, 4>* kvv = reinterpret_cast<const device vec<InT, 4>*>(
                  k_head + (size_t)kr * D + kk * 16 + sc.x);
              const vec<InT, 4> kk4 = kvv[0];
              for (short j = 0; j < kElemCols; j++)
                ct_b[hh * kElemsPerFrag + i * kElemCols + j] = kk4[j];
            } else {
              for (short j = 0; j < kElemCols; j++)
                ct_b[hh * kElemsPerFrag + i * kElemCols + j] = InT(0);
            }
          }
        }
        mm.run(ct_a, ct_b, ct_c);
      }
      // ── in-register masked softmax over the S fragments ──
      // s_p[hh][i*4+j]: row = m_base + sc.y + i*8, col = t0 + hh*16 + sc.x + j.
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
              const float raw = ct_c[hh * kElemsPerFrag + i * kElemCols + j];
              const bool vis = live && gp < kv_end && gp <= row_limit;
              const float s = vis ? raw * scale : -1e38f;
              s_p[hh][i * kElemCols + j] = s;
              tmax[i] = metal::max(tmax[i], s);
            }
          }
        }
        // reduce max across the 4 lanes sharing each row (lane bits 0, 3)
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

      // ── O rescale by this tile's factors ──
      for (short i = 0; i < 2; i++) {
        const float f = tile_f[i];
        for (int g = 0; g < NGROUPS; g++)
          for (short hh = 0; hh < 2; hh++)
            for (short j = 0; j < kElemCols; j++)
              o_frag[g][hh][i * kElemCols + j] *= f;
      }

      // ── PV: O[16, 32g..] += P[16,32] x (V^T)[32dims,32keys]^T ──
      // ct_c holds one O group across both K-frags (init from o_frag,
      // write back once); threadgroup loads vectorized 4-wide.
      for (int g = 0; g < NGROUPS; g++) {
        for (short i = 0; i < kElemsPerFrag; i++) {
          ct_c[i] = o_frag[g][0][i];
          ct_c[kElemsPerFrag + i] = o_frag[g][1][i];
        }
        for (int kk = 0; kk < 2; kk++) {          // 32 keys = 2 K-frags
          // P fragment IS the exp'd S fragment for this key half — registers.
          for (short i = 0; i < 2; i++)
            for (short j = 0; j < kElemCols; j++)
              ct_a[i * kElemCols + j] = InT(s_p[kk][i * kElemCols + j]);
          for (short hh = 0; hh < 2; hh++) {
            for (short i = 0; i < 2; i++) {
              const int dcol = g * 32 + hh * 16 + sc.y + i * kElemRowsJump;
              if (DIRECTV) {
                for (short j = 0; j < kElemCols; j++) {
                  const int kr = t0 + kk * 16 + sc.x + j;
                  ct_b[hh * kElemsPerFrag + i * kElemCols + j] = (kr < kv_end)
                      ? v_head[(size_t)kr * D + dcol] : InT(0);
                }
              } else {
                const threadgroup vec<InT, 4>* vv = reinterpret_cast<const threadgroup vec<InT, 4>*>(
                    tg_vT + dcol * TK + kk * 16 + sc.x);
                const vec<InT, 4> v4 = vv[0];
                for (short j = 0; j < kElemCols; j++)
                  ct_b[hh * kElemsPerFrag + i * kElemCols + j] = v4[j];
              }
            }
          }
          mm.run(ct_a, ct_b, ct_c);
        }
        for (short i = 0; i < kElemsPerFrag; i++) {
          o_frag[g][0][i] = ct_c[i];
          o_frag[g][1][i] = ct_c[kElemsPerFrag + i];
        }
      }
      if (!DIRECTV) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    // ── store partials [1, HQ, QL, blocks, D] + stats [1, HQ, QL, blocks] ──
    for (short i = 0; i < 2; i++) {
      const int m_local = m_base + sc.y + i * kElemRowsJump;
      if (m_local >= LIVE) continue;
      const int hq_row = kv_head * LIVE + m_local;     // == h*QL + j global
      device InT* prow = partials
          + ((size_t)hq_row * n_blocks + block_idx) * D;
      for (int g = 0; g < NGROUPS; g++)
        for (short hh = 0; hh < 2; hh++)
          for (short j = 0; j < kElemCols; j++)
            prow[g * 32 + hh * 16 + sc.x + j] = InT(o_frag[g][hh][i * kElemCols + j]);
    }
    // Stats: one designated lane per row (lane bits 0 and 3 clear) writes.
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
"""


@lru_cache(maxsize=4)
def _nax_tile_kernel():
    try:
        return mx.fast.metal_kernel(
            name="mtplx_nax_tile_partials",
            input_names=["queries", "keys", "values", "offset", "kcap",
                         "scale", "blocks"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER,
            source=_SOURCE,
        )
    except Exception:  # noqa: BLE001 — toolchain without Metal4/MPP support
        return None


def _bail(reason: str):
    nax_tile_bail_counts[reason] = nax_tile_bail_counts.get(reason, 0) + 1
    if os.environ.get("MTPLX_NAX_TILE_DEBUG"):
        print(f"[nax-tile bail] {reason}")
    return None


def sdpa_nax_tile(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    offset: int | mx.array,
    scale: float,
) -> mx.array | None:
    """TensorOps wide-M tail-causal SDPA. Same contract as the packed kernel."""
    if os.environ.get("MTPLX_NAX_TILE", "1") == "0":
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

    blocks = _blocks_for_capacity(capacity)
    # Partial-traffic cap (2026-08-26 Pulse receipt: 210 GB/s of partial
    # writes at the scalar kernel's block count = 3x read amplification).
    # The tile kernel fills the GPU with (hk x NSGS-simdgroup) threadgroups;
    # far fewer, fatter blocks cut partials linearly. Env-tunable for cells.
    cap_blocks = int(os.environ.get("MTPLX_NAX_TILE_BLOCKS", "0") or 0)
    if cap_blocks > 0:
        blocks = min(blocks, max(32, (cap_blocks // 32) * 32))
    if blocks <= 0 or blocks % 32:
        return _bail("blocks_geometry")
    blocks_arr = mx.array([blocks], dtype=mx.int32)

    kernel = _nax_tile_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return _bail("kernel_unavailable")

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
                ("DIRECTV", 1 if os.environ.get("MTPLX_NAX_TILE_DIRECTV") == "1" else 0),
            ],
            grid=(hk * 32 * ((gqa_factor * q_len + 15) // 16), 1, blocks),
            threadgroup=(32 * ((gqa_factor * q_len + 15) // 16), 1, 1),
            output_shapes=[partial_shape, stats_shape, stats_shape],
            output_dtypes=[queries.dtype, mx.float32, mx.float32],
        )
    except Exception:  # noqa: BLE001 — dispatch/compile failure => stock fallback
        return _bail("dispatch_failed")

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
    return out
