"""TensorOps flash-decoding SDPA, head-dim-split variant (hyper K2 variant B, 2026-09-01).

Same contract as :func:`mtplx.kernels.sdpa_nax_flash.sdpa_nax_flash`. The difference is
register pressure: variant A keeps a full [16 x 256] fp32 output tile per simdgroup (128
accumulators per thread) and the 72.7k walk bench showed 60-70 GB/s of device writes
beyond its partials, i.e. register spills. Here a threadgroup is ``NSGM x 2`` simdgroups
and each simdgroup owns HALF of the head dimension for one 16-row M block:

* QK: each simdgroup computes the partial score tile over its 128 dims (loading only its
  half of every K row), the two halves are summed through a double-buffered threadgroup
  buffer (one barrier per 32-key tile), both simdgroups then run the same online softmax.
* PV: each simdgroup accumulates only its 128 output dims (64 fp32 accumulators per
  thread) from its half of every V row.

Bytes read from DRAM are unchanged; the kernel just splits them across twice the
simdgroups with half the live state each. Gated like the sibling kernel; env kill-switch
MTPLX_NAX_FLASH_DSPLIT=0; block count MTPLX_NAX_FLASH_DSPLIT_BLOCKS.
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

from ..nax_verify import nax_available
from .sdpa_gqa_packed import _paged_reduce_kernel
from .sdpa_nax_flash import _HEADER

nax_flash_dsplit_bail_counts: dict[str, int] = {}
# Successful dispatches, so /health can show the route engaged (not only
# its bails) — the one-line receipt the #459 reports needed.
nax_flash_dsplit_dispatch_counts: dict[str, int] = {}

# Template params: InT, D, QL, GQA_F.
_SOURCE = r"""
    constexpr int TK = 32;
    constexpr int MROWS = 16;
    constexpr int LIVE = GQA_F * QL;
    constexpr int NSGM = (LIVE + MROWS - 1) / MROWS;
    constexpr int NDH = 2;                       // head-dim halves
    constexpr int DH = D / NDH;                  // dims per simdgroup
    constexpr int NSGS = NSGM * NDH;
    constexpr int DFRAGS = DH / 16;
    constexpr int NGROUPS = DH / 32;
    constexpr int SBUF = MROWS * TK;             // one partial S tile (floats)

    const uint lid = thread_position_in_threadgroup.x;
    const uint sg = lid >> 5;
    const ushort lane = ushort(lid & 31);
    const int dh = int(sg) % NDH;
    const int mg = int(sg) / NDH;
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
    const int db = dh * DH;

    threadgroup float tg_s[2 * NSGS * SBUF];

    const device InT* k_head = keys + (size_t)kv_head * kcap * D;
    const device InT* v_head = values + (size_t)kv_head * kcap * D;
    const device InT* q_head = queries + (size_t)kv_head * LIVE * D;

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

    const int my_slot = mg * NDH + dh;
    const int other_slot = mg * NDH + (1 - dh);
    int tile_par = 0;

    for (int t0 = kv_start; t0 < kv_end; t0 += TK) {
      const bool tile_full = (t0 + TK) <= kv_end;

      // ── partial QK over this simdgroup's 128 dims ──
      for (short i = 0; i < 2 * kElemsPerFrag; i++) ct_s[i] = 0.0f;
      for (int kk = 0; kk < DFRAGS; kk++) {
        for (short i = 0; i < 2; i++) {
          const int mr = m_base + sc.y + i * kElemRowsJump;
          vec<InT, 4> qq = vec<InT, 4>(InT(0));
          if (mr < LIVE) {
            qq = reinterpret_cast<const device vec<InT, 4>*>(
                q_head + (size_t)mr * D + db + kk * 16 + sc.x)[0];
          }
          for (short j = 0; j < kElemCols; j++) ct_qa[i * kElemCols + j] = qq[j];
        }
        for (short hh = 0; hh < 2; hh++) {
          for (short i = 0; i < 2; i++) {
            const int kr = t0 + hh * 16 + sc.y + i * kElemRowsJump;
            vec<InT, 4> kk4 = vec<InT, 4>(InT(0));
            if (tile_full || kr < kv_end) {
              kk4 = reinterpret_cast<const device vec<InT, 4>*>(
                  k_head + (size_t)kr * D + db + kk * 16 + sc.x)[0];
            }
            for (short j = 0; j < kElemCols; j++)
              ct_kb[hh * kElemsPerFrag + i * kElemCols + j] = kk4[j];
          }
        }
        mm_qk.run(ct_qa, ct_kb, ct_s);
      }

      // ── exchange the partial S with the partner half (one barrier per tile) ──
      {
        threadgroup float* mine = tg_s + (size_t)(tile_par * NSGS + my_slot) * SBUF;
        for (short hh = 0; hh < 2; hh++)
          for (short i = 0; i < 2; i++) {
            const int idx = (sc.y + i * kElemRowsJump) * TK + hh * 16 + sc.x;
            float4 v4;
            v4.x = ct_s[hh * kElemsPerFrag + i * kElemCols + 0];
            v4.y = ct_s[hh * kElemsPerFrag + i * kElemCols + 1];
            v4.z = ct_s[hh * kElemsPerFrag + i * kElemCols + 2];
            v4.w = ct_s[hh * kElemsPerFrag + i * kElemCols + 3];
            reinterpret_cast<threadgroup float4*>(mine + idx)[0] = v4;
          }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float s_raw[2][kElemsPerFrag];
      {
        const threadgroup float* other = tg_s + (size_t)(tile_par * NSGS + other_slot) * SBUF;
        for (short hh = 0; hh < 2; hh++)
          for (short i = 0; i < 2; i++) {
            const int idx = (sc.y + i * kElemRowsJump) * TK + hh * 16 + sc.x;
            const float4 v4 = reinterpret_cast<const threadgroup float4*>(other + idx)[0];
            s_raw[hh][i * kElemCols + 0] = ct_s[hh * kElemsPerFrag + i * kElemCols + 0] + v4.x;
            s_raw[hh][i * kElemCols + 1] = ct_s[hh * kElemsPerFrag + i * kElemCols + 1] + v4.y;
            s_raw[hh][i * kElemCols + 2] = ct_s[hh * kElemsPerFrag + i * kElemCols + 2] + v4.z;
            s_raw[hh][i * kElemCols + 3] = ct_s[hh * kElemsPerFrag + i * kElemCols + 3] + v4.w;
          }
      }
      tile_par ^= 1;

      // ── in-register masked online softmax (identical in both halves) ──
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
              const float raw = s_raw[hh][i * kElemCols + j];
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

      // ── PV over this simdgroup's 128 output dims ──
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
                    v_head + (size_t)kr * D + db + g * 32 + hh * 16 + sc.x)[0];
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

    // ── store this half of the partial + (dh == 0) the row stats ──
    for (short i = 0; i < 2; i++) {
      const int m_local = m_base + sc.y + i * kElemRowsJump;
      if (m_local >= LIVE) continue;
      const int hq_row = kv_head * LIVE + m_local;
      device InT* prow = partials + ((size_t)hq_row * n_blocks + block_idx) * D + db;
      for (int g = 0; g < NGROUPS; g++)
        for (short hh = 0; hh < 2; hh++)
          for (short j = 0; j < kElemCols; j++)
            prow[g * 32 + hh * 16 + sc.x + j] = InT(o_frag[g][hh][i * kElemCols + j]);
    }
    if (dh == 0 && (lane & 0x9) == 0) {
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
def _nax_flash_dsplit_kernel():
    try:
        return mx.fast.metal_kernel(
            name="mtplx_nax_flash_dsplit_partials",
            input_names=["queries", "keys", "values", "offset", "kcap",
                         "scale", "blocks"],
            output_names=["partials", "sums", "maxs"],
            header=_HEADER,
            source=_SOURCE,
        )
    except Exception:  # noqa: BLE001 — toolchain without Metal4/MPP support
        return None


def _bail(reason: str):
    nax_flash_dsplit_bail_counts[reason] = nax_flash_dsplit_bail_counts.get(reason, 0) + 1
    if os.environ.get("MTPLX_NAX_FLASH_DEBUG"):
        print(f"[nax-flash-dsplit bail] {reason}")
    return None


def _default_dsplit_blocks(capacity: int) -> int:
    # 2026-09-01 W2 walk sweep (ms/layer): 16k b128 0.257 / b256 0.267;
    # 72.7k b256 0.898 / b512 0.917 / b1024 1.011; 128k b512 1.504 / b1024 1.552.
    if capacity >= 98304:
        return 512
    if capacity >= 65536:
        return 256
    if capacity >= 16384:
        return 128
    return 64


def sdpa_nax_flash_dsplit(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    offset: int | mx.array,
    scale: float,
) -> mx.array | None:
    """Head-dim-split TensorOps flash SDPA. Same contract as the packed kernel."""
    if os.environ.get("MTPLX_NAX_FLASH_DSPLIT", "1") == "0":
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
    if gqa_factor * q_len > 32:
        return _bail("m_rows_gt_32")
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

    blocks = int(os.environ.get("MTPLX_NAX_FLASH_DSPLIT_BLOCKS", "0") or 0) or _default_dsplit_blocks(capacity)
    blocks = max(32, (blocks // 32) * 32)
    blocks_arr = mx.array([blocks], dtype=mx.int32)

    kernel = _nax_flash_dsplit_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return _bail("kernel_unavailable")

    nsgm = (gqa_factor * q_len + 15) // 16
    nthreads = 32 * nsgm * 2
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
    nax_flash_dsplit_dispatch_counts["dispatched"] = nax_flash_dsplit_dispatch_counts.get("dispatched", 0) + 1
    return out
