// SPDX-License-Identifier: Apache-2.0
//
// SPLIT-K (KV-split) direct-index sparse GQA attention for Qwen3.8
// Flash-Next QSA DECODE -- M=4 (fixed-M4 verify) and M=1 (single-row draft).
//
// Derived from MTPLX's phase-1 port of oMLX's single-pass kernel
// (sparse_gqa/steel_qsa_sparse_gqa.h; jundot/omlx 7467dce8, Jonathan
// Spangler, Apache-2.0; the 128-bit global K/V staging is in turn adapted
// from mlx-serve's MIT `msv_attn_p256`, Copyright 2026 David Dalcu).  The
// per-tile body below -- the direct-index K/V staging, the Steel MMA
// score/PV pair, the fp32 online softmax -- is that kernel's, unchanged.
//
// WHY A SPLIT-K VARIANT EXISTS AT ALL
// -----------------------------------
// The single-pass kernel parallelises over QUERY ROWS: its grid is
// ``(qL, kv_heads, 1)`` threadgroups of 64 threads.  At prefill (4,096 rows)
// that is 8,192 threadgroups.  At M=4 it is EIGHT threadgroups on a 40-core
// M5 Max, and at M=1 it is TWO -- and each of those few threadgroups still
// walks all 2,051 selected keys x 256 dims.  Phase 1's own design note
// (docs/perf/qsa-sparse-gqa-phase2-wiring.md, section 4) called this out and
// priced the fix as its own item: split the selected keys across several
// threadgroups per (row, KV head) and combine the partial online-softmax
// states in a second pass.  That is this file.
//
// It is also why the decode lane MUST be split-K rather than single-pass:
// MTPLX's own history has a hand-written mx.fast.metal_kernel SDPA losing to
// stock at long N precisely because MLX's production SDPA switches to a
// KV-split two-pass path there.  A single-pass sparse kernel would repeat
// that mistake with a shorter (2,051-key) but equally serial walk.
//
// WHAT CHANGES FROM THE PREFILL KERNEL, AND WHY
// ---------------------------------------------
//  1. ``tid.z`` is the KV SPLIT, not the batch.  The contract already
//     refuses ``B != 1``, so the batch axis was carrying no information.
//  2. The per-tile loop runs over ``[t0, t1)`` instead of ``[0, n_tiles)``.
//  3. The final ``Otile / sum`` divide MOVES to the merge pass; pass 1
//     writes the UNNORMALISED accumulator plus the running ``(m, l)`` pair.
//  4. The query offset arrives as a one-element int32 DEVICE buffer rather
//     than a host scalar in the params block, because the fixed-M4 verify
//     may carry a tensor-valued cache offset and reading it on the host
//     would synchronize the graph.
//  5. Validity is decided PER SLOT (``block_id < complete_blocks``), not by
//     a leading-prefix cut.  This is the load-bearing difference and it is
//     not cosmetic:
//
//     The prefill branch of ``_select_eager`` SORTS its top-k ascending, so
//     there the valid entries really are a leading prefix and the prefix cut
//     is exact.  The DECODE branch does not sort: it hands
//     ``mx.argpartition``'s raw output straight to the rows-gather token
//     list, whose predicate is ``block < visible_blocks`` evaluated on
//     EVERY slot.  Applying the prefix cut to an unsorted row
//     would drop visible blocks and admit invisible ones.  Below the
//     dense/sparse crossover the argpartition genuinely returns
//     -inf-scored ids (there are fewer than 512 complete blocks to choose
//     from), so this is a real case, not a hypothetical one.
//
//     Note also what is NOT used as the predicate: ``candidate <= q_abs``.
//     That is implied by ``block < complete_blocks`` but is NOT equivalent
//     to it -- for ``block == complete_blocks`` it would admit the tokens of
//     the incomplete block, which the tail slots already cover, and
//     double-count them in the softmax denominator.
//
// The visible set is therefore IDENTICAL to the shipped rows-gather lane's,
// slot for slot.  The arithmetic is not: fp32 online softmax (exp2) and an
// fp32 P@V against the shipped path's fp32 softmax, bf16 probability cast
// and bf16 P@V, plus the split-K rescale.  That is a ROUNDING-CLASS change
// to attention output on the same terms as kernels/qwen4_m4_hyper_read.py --
// adopt it on greedy-token agreement and a HumanEval gate, never on a digest.

#pragma once

#include "mlx/backend/metal/kernels/steel/attn/attn.h"
#include "mlx/backend/metal/kernels/steel/attn/params.h"

#include "sparse_gqa/qsa_sparse_gqa_decode_params.h"

using namespace mlx::steel;

struct MtplxQsaDecMaxOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct MtplxQsaDecSumOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MtplxQsaDecMulOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct MtplxQsaDecExpSubOp {
  template <typename T> METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::fast::exp2(x - y);
  }
};

// clang-format off
template <
    typename T,
    int BK,
    int DC,
    int GQA,
    int H_PAD,
    int D,
    int WM,
    typename IndexT,
    typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(WM * 32)]] void
mtplx_qsa_sparse_gqa_decode_split(
    const device T* Q [[buffer(0)]],
    const device T* K [[buffer(1)]],
    const device T* V [[buffer(2)]],
    const device IndexT* Topk [[buffer(3)]],
    const device int* QOffset [[buffer(4)]],
    device AccumType* Partial [[buffer(5)]],
    const constant MtplxQsaSparseGqaDecodeParams* params [[buffer(6)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]]) { // clang-format on

  constexpr short kFragSize = 8;
  constexpr short padQ = 16 / sizeof(T);
  constexpr short padK = 16 / sizeof(T);
  constexpr short padV = 16 / sizeof(T);

  constexpr short LDQ = DC + padQ;
  constexpr short LDK = BK + padK;
  constexpr short LDV = DC + padV;

  constexpr int kNWarps = WM;
  constexpr int TQ = H_PAD / (kNWarps * kFragSize);
  constexpr int TK = BK / kFragSize;
  constexpr int TDC = DC / kFragSize;
  constexpr int D_CHUNKS = D / DC;

  static_assert(GQA <= H_PAD, "Qwen GQA heads must fit the padded MMA tile.");
  static_assert(TQ == 1, "Qwen sparse GQA expects one query-head tile.");
  static_assert(H_PAD % (kNWarps * kFragSize) == 0,
                "Padded query heads must divide evenly across simdgroups.");
  static_assert(BK % kFragSize == 0, "BK must be a multiple of eight.");
  static_assert(DC % kFragSize == 0, "DC must be a multiple of eight.");
  static_assert(D % DC == 0, "Head dimension must divide DC.");

  constexpr int tgp_size = WM * 32;
  const int lane = int(simd_group_id * 32 + simd_lane_id);
  const int q_pos = int(tid.x);
  const int kv_head = int(tid.y);
  const int split = int(tid.z);

  threadgroup T Qs[H_PAD * LDQ];
  threadgroup T KVs[(BK * LDV > DC * LDK) ? BK * LDV : DC * LDK];
  threadgroup int selected[BK];

  using MMAFragAcc = BaseMMAFrag<AccumType, kFragSize, kFragSize>;
  MMATile<AccumType, TQ, 1, MMAFragAcc> Qtile;
  MMATile<AccumType, 1, TK, MMAFragAcc> Ktile;
  MMATile<AccumType, TQ, TK, MMAFragAcc> Stile;
  MMATile<AccumType, 1, 1, MMAFragAcc> Vtile;
  MMATile<AccumType, TQ, D_CHUNKS * TDC, MMAFragAcc> Otile;
  Otile.clear();

  const short2 simd_coord = MMAFragAcc::get_coord(simd_lane_id);
  const short sm = simd_coord.y;
  const short sn = simd_coord.x;
  const short tm = kFragSize * TQ * simd_group_id;
  const short Qs_offset = (tm + sm) * LDQ + sn;
  const short Ks_offset = sm * LDK + sn;
  const short Vs_offset = sm * LDV + sn;

  const AccumType scale = AccumType(params->scale * M_LOG2E_F);
  constexpr short rows_per_thread = decltype(Stile)::kRowsPerThread;
  static_assert(rows_per_thread == 1,
                "One query-head tile means one accumulator row per thread; "
                "the (m, l) store below assumes it.");
  AccumType max_score[rows_per_thread];
  AccumType sum_score[rows_per_thread] = {0};
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < rows_per_thread; ++i) {
    // finite_min, NOT -infinity: an all-masked tile must produce
    // exp2(-inf - finite_min) == 0 rather than exp2(-inf + inf) == NaN.
    max_score[i] = Limits<AccumType>::finite_min;
  }

  const int query_head_base = kv_head * GQA;
  const device T *q_base = Q + size_t(query_head_base) * params->Q_strides[1] +
                           size_t(q_pos) * params->Q_strides[2];
  const device T *k_base = K + size_t(kv_head) * params->K_strides[1];
  const device T *v_base = V + size_t(kv_head) * params->V_strides[1];
  const device IndexT *topk_base = Topk + size_t(q_pos) * params->Topk_strides[2];

  const int q_abs = QOffset[0] + q_pos;
  constexpr int kCompressRatio = 4;
  constexpr int kTail = kCompressRatio - 1;
  const int selected_tokens = params->topk * kCompressRatio + kTail;
  // Identical to the shipped lane's ``visible_blocks = (qpos + 1) / RATIO``.
  const int complete_blocks = (q_abs + 1) / kCompressRatio;

  const int t0 = split * params->tiles_per_split;
  const int t1 = metal::min(params->n_tiles, t0 + params->tiles_per_split);

  for (int ktile = t0; ktile < t1; ++ktile) {
    const int topk_off = ktile * BK;
    for (int k = lane; k < BK; k += tgp_size) {
      const int slot = topk_off + k;
      int k_pos = -1;
      if (slot < params->topk * kCompressRatio) {
        const int block_slot = slot / kCompressRatio;
        // PER SLOT, not a prefix cut: the decode selector does not sort.
        const long raw_block = long(topk_base[block_slot]);
        if (raw_block >= 0 && raw_block < long(complete_blocks)) {
          const long candidate =
              raw_block * long(kCompressRatio) + long(slot % kCompressRatio);
          if (candidate >= 0 && candidate < long(params->kL)) {
            k_pos = int(candidate);
          }
        }
      } else if (slot < selected_tokens) {
        const int tail_offset = slot - params->topk * kCompressRatio;
        const long candidate =
            long(complete_blocks) * long(kCompressRatio) + long(tail_offset);
        if (candidate >= 0 && candidate < long(params->kL) &&
            candidate <= long(q_abs)) {
          k_pos = int(candidate);
        }
      }
      selected[k] = k_pos;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    Stile.clear();
    STEEL_PRAGMA_UNROLL
    for (short dchunk = 0; dchunk < D_CHUNKS; ++dchunk) {
      const int dbase = int(dchunk) * DC;
      for (int elem = lane; elem < H_PAD * (DC / 8); elem += tgp_size) {
        const int h = elem / (DC / 8);
        const int d8 = elem - h * (DC / 8);
        uint4 word = uint4(0);
        if (h < GQA) {
          word = *((const device uint4 *)(q_base +
                                          size_t(h) * params->Q_strides[1] +
                                          dbase) +
                   d8);
        }
        *((threadgroup uint4 *)(Qs + h * LDQ) + d8) = word;
      }
      for (int elem = lane; elem < BK * (DC / 8); elem += tgp_size) {
        const int k = elem / (DC / 8);
        const int d8 = elem - k * (DC / 8);
        const int k_pos = selected[k];
        uint4 word = uint4(0);
        if (k_pos >= 0) {
          word = *((const device uint4 *)(k_base +
                                          size_t(k_pos) * params->K_strides[2] +
                                          dbase) +
                   d8);
        }
        thread T *values = (thread T *)&word;
        const int d = d8 * 8;
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < 8; ++e) {
          KVs[k + (d + e) * LDK] = values[e];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short dd = 0; dd < TDC; ++dd) {
        simdgroup_barrier(mem_flags::mem_none);
        Qtile.template load<T, 1, 1, LDQ, 1>(&Qs[Qs_offset + dd * kFragSize]);
        Ktile.template load<T, 1, 1, LDK, 1>(
            &KVs[Ks_offset + dd * kFragSize * LDK]);
        simdgroup_barrier(mem_flags::mem_none);
        tile_matmad(Stile, Qtile, Ktile, Stile);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(Stile)::kElemsPerTile; ++i) {
      Stile.elems()[i] *= scale;
    }
    {
      using stile_t = decltype(Stile);
      using selem_t = typename stile_t::elem_type;
      constexpr auto neg_inf = selem_t(-INFINITY);
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < stile_t::kTileRows; ++i) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < stile_t::kTileCols; ++j) {
          const short col_pos = sn + j * stile_t::kFragCols;
          STEEL_PRAGMA_UNROLL
          for (short e = 0; e < stile_t::MMAFrag_t::kElemCols; ++e) {
            if (selected[col_pos + e] < 0) {
              Stile.frag_at(i, j)[e] = neg_inf;
            }
          }
        }
      }
    }

    AccumType new_max[rows_per_thread];
    AccumType factor[rows_per_thread];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      new_max[i] = max_score[i];
    }
    Stile.template row_reduce<MtplxQsaDecMaxOp>(new_max);
    Stile.template row_bin_op<MtplxQsaDecExpSubOp>(new_max);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      factor[i] = metal::fast::exp2(max_score[i] - new_max[i]);
      max_score[i] = new_max[i];
    }
    AccumType sum_score_tmp[rows_per_thread] = {0};
    Stile.template row_reduce<MtplxQsaDecSumOp>(sum_score_tmp);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < rows_per_thread; ++i) {
      sum_score[i] = sum_score[i] * factor[i] + sum_score_tmp[i];
    }
    Otile.template row_bin_op<MtplxQsaDecMulOp>(factor);

    STEEL_PRAGMA_UNROLL
    for (short vchunk = 0; vchunk < D_CHUNKS; ++vchunk) {
      const int dbase = int(vchunk) * DC;
      for (int elem = lane; elem < BK * (DC / 8); elem += tgp_size) {
        const int k = elem / (DC / 8);
        const int d8 = elem - k * (DC / 8);
        const int k_pos = selected[k];
        uint4 word = uint4(0);
        if (k_pos >= 0) {
          word = *((const device uint4 *)(v_base +
                                          size_t(k_pos) * params->V_strides[2] +
                                          dbase) +
                   d8);
        }
        *((threadgroup uint4 *)(KVs + k * LDV) + d8) = word;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      STEEL_PRAGMA_UNROLL
      for (short iq = 0; iq < TQ; ++iq) {
        STEEL_PRAGMA_UNROLL
        for (short id = 0; id < TDC; ++id) {
          STEEL_PRAGMA_UNROLL
          for (short ik = 0; ik < TK; ++ik) {
            const short kk = ik * kFragSize;
            const short dd = id * kFragSize;
            Vtile.template load<T, 1, 1, LDV, 1>(
                &KVs[Vs_offset + kk * LDV + dd]);
            MMAFragAcc::mma(Otile.frag_at(iq, vchunk * TDC + id),
                            Stile.frag_at(iq, ik), Vtile.frag_at(0, 0),
                            Otile.frag_at(iq, vchunk * TDC + id));
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // NO divide here: the merge pass owns the normalisation, because the
  // denominator is only known once every split has reported.
  const short rows_left = short(GQA - (tm + sm));
  if (rows_left > 0) {
    const size_t ld = size_t(params->partial_ld);
    const size_t split_stride =
        size_t(params->q_heads) * size_t(params->qL) * ld;
    device AccumType *prow = Partial + size_t(split) * split_stride +
                             size_t(query_head_base + tm + sm) *
                                 size_t(params->qL) * ld +
                             size_t(q_pos) * ld;
    Otile.template store_safe<AccumType, 1, 1>(prow + sn, params->partial_ld,
                                               short2(D - sn, rows_left));
    // The four lanes that share a row all hold the same reduced (m, l) after
    // ``row_reduce``; sn == 0 elects one of them to write.
    if (sn == 0) {
      prow[D] = max_score[0];
      prow[D + 1] = sum_score[0];
    }
  }
}

// clang-format off
template <typename T, int D, typename AccumType = float>
[[kernel, max_total_threads_per_threadgroup(D)]] void
mtplx_qsa_sparse_gqa_decode_merge(
    const device AccumType* Partial [[buffer(0)]],
    device T* O [[buffer(1)]],
    const constant MtplxQsaSparseGqaMergeParams* params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]]) { // clang-format on

  // One threadgroup per (query head, query row); one thread per head dim.
  const int row = int(tid.x);
  const int d = int(lid.x);
  const size_t ld = size_t(params->partial_ld);
  const int n_splits = params->n_splits;
  const size_t split_stride = size_t(params->q_heads) * size_t(params->qL) * ld;
  const device AccumType *base = Partial + size_t(row) * ld;

  // Pass 1 never writes -infinity into m (it initialises to finite_min), so
  // this max is always replaced by a real value and the rescale below can
  // never evaluate inf - inf.
  AccumType m = Limits<AccumType>::finite_min;
  for (int s = 0; s < n_splits; ++s) {
    m = metal::max(m, base[size_t(s) * split_stride + D]);
  }

  AccumType denom = AccumType(0);
  AccumType acc = AccumType(0);
  for (int s = 0; s < n_splits; ++s) {
    const device AccumType *p = base + size_t(s) * split_stride;
    const AccumType alpha = metal::fast::exp2(p[D] - m);
    denom += alpha * p[D + 1];
    acc += alpha * p[d];
  }
  // denom == 0 is only reachable if EVERY slot of the row was masked, which
  // the lane's contract excludes (complete_blocks >= 1 guarantees at least
  // one visible block is selected).  Emit zero rather than a NaN.
  const AccumType inv =
      (denom > AccumType(0)) ? (AccumType(1) / denom) : AccumType(0);

  const int head = row / params->qL;
  const int q_pos = row - head * params->qL;
  O[size_t(head) * size_t(params->O_strides[1]) +
    size_t(q_pos) * size_t(params->O_strides[2]) + size_t(d)] =
      static_cast<T>(acc * inv);
}
