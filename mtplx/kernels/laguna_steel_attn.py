"""Tiled (flash) prefill attention for Laguna S-2.1, ported from the mlx.fast
**Laguna XS2.1** challenge's P2 *steel* attention path (``steel_attention`` /
its ``_nax`` M5 twin under ``Vendor/mlx-swift/.../steel/attn/``).

## What is being ported (and what is NOT)

The donor is MLX's fused *steel* flash-attention kernel: it tiles Q into blocks
of ``BQ`` rows, streams K/V in ``BK``-row tiles staged in threadgroup memory,
and accumulates ``softmax(QK^T * scale + mask) @ V`` with the online-softmax
(running max / running sum) recurrence, so the full ``[qL, kL]`` score matrix is
never materialized.  Masking is positional and done *inside* the kernel: a
``do_causal`` function-constant path (``row_pos < col_pos -> -inf``) and, for
the sliding layers, an added window bound.  See
``steel/attn/kernels/steel_attention.h`` (the ``attention`` kernel) and
``params.h`` in the challenge repo.

The donor does its two matmuls (``Q@K^T`` and ``P@V``) with the ``mlx::steel``
simdgroup-matrix (MMA) fragment machinery (``MMATile`` / ``BaseMMAFrag`` /
``BlockLoaderT`` from ``steel/attn/{attn,mma,loader,transforms}.h``).  Those
headers are **not** reachable from ``mx.fast.metal_kernel`` (it compiles a
standalone Metal snippet with only MLX's injected preamble), so this is an
**algorithmic** port, not a byte port: the tiling, the KV staging, the
online-softmax recurrence, the positional causal + sliding-window masking, and
the GQA head mapping are reproduced faithfully; the two inner GEMMs are done
with a cooperative simdgroup layout (one simdgroup per query row, 32 lanes over
head_dim, ``simd_sum`` for the QK dot) instead of simdgroup-matrix fragments.
The steel MMA path is the reason MLX's fused SDPA is fast, so this port is
expected to be *slower* than stock ``mx.fast.scaled_dot_product_attention`` at
prefill — the deliverable is a correct, S-2.1-shaped reproduction that is then
**measured** against stock, not assumed to beat it.

## S-2.1 shape

Two interleaved head families, both causal, head_dim 128:

    full_attention   : 48 q-heads / 8 kv-heads -> gqa_factor 6, causal (no window)
    sliding_attention: 72 q-heads / 8 kv-heads -> gqa_factor 9, causal + window 512

head_dim 128 is what makes the fused steel attention dispatchable at prefill
(the challenge notes head dim 128 as the enabling condition).  YaRN partial
rotary and the YaRN ``attention_factor`` are applied to Q/K *before* attention
(inside RoPE), so this kernel receives already-roped Q/K and the only attention
scalar it needs is ``scale = head_dim ** -0.5`` — identical to what the stock
path receives (``mtplx/models/laguna.py`` ``Attention``).

## Eligibility / fallback

``steel_attention_prefill`` runs the kernel only for the shapes it covers
(4-D, head_dim 128, ``hq % hk == 0``, fp16/bf16, causal, ``kL >= qL``) and
returns ``None`` otherwise so callers fall back to stock SDPA.
``steel_attention_or_sdpa`` wires that fallback in and asserts the output shape.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import mlx.core as mx


_HEAD_DIM = 128
_BD = 32          # lanes per simdgroup (cooperate over head_dim)
_BQ = 8           # query rows (== simdgroups) per threadgroup
_BK = 32          # KV rows staged per tile
_ELEMS = _HEAD_DIM // _BD  # 4 head-dim elements owned per lane


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------
def is_steel_attention_eligible(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    causal: bool = True,
) -> bool:
    """Whether the tiled steel-attention prefill kernel covers this shape.

    The kernel implements *positional* causal (+ optional sliding-window)
    masking only; an arbitrary/additive mask or a non-causal request is not
    covered and must fall back to stock SDPA.
    """

    if not mx.metal.is_available():
        return False
    if not causal:
        return False
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return False
    b, hq, ql, d = (int(v) for v in queries.shape)
    if d != _HEAD_DIM or ql < 1:
        return False
    bk, hk, n, dk = (int(v) for v in keys.shape)
    if bk != b or dk != d:
        return False
    if tuple(values.shape) != (b, hk, n, d):
        return False
    if hk <= 0 or hq % hk != 0:
        return False
    if n < ql:  # causal prefill needs kL >= qL
        return False
    # bf16/fp16 only: the fp32 KV tiles would need 2*BK*D*4 = 32 KiB of
    # threadgroup memory (exactly the Apple limit, no headroom), so fp32 falls
    # back to stock like the sibling SDPA kernels.
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return False
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return False
    return True


# ---------------------------------------------------------------------------
# The Metal kernel
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _steel_attention_kernel(hq: int, hk: int, gqa: int, window: int):
    header = f"""
        using namespace metal;
        constant constexpr int D = {_HEAD_DIM};
        constant constexpr int BD = {_BD};
        constant constexpr int ELEMS = {_ELEMS};
        constant constexpr int BQ = {_BQ};
        constant constexpr int BK = {_BK};
        constant constexpr int HQ = {hq};
        constant constexpr int HK = {hk};
        constant constexpr int GQA = {gqa};
        constant constexpr int WINDOW = {window};   // 0 => full causal (no window)
        constant constexpr int TG = BQ * BD;        // threads per threadgroup
    """

    source = """
        typedef float U;
        uint tg = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;   // 0..BQ-1  -> query row in block
        uint simd_lid = thread_index_in_simdgroup;         // 0..BD-1  -> head-dim lane

        int qLi = qL;
        int kLi = kL;
        int offi = qL_off;

        int NQ = (qLi + BQ - 1) / BQ;
        uint per_batch = uint(HQ) * uint(NQ);
        uint b  = tg / per_batch;
        uint rem = tg - b * per_batch;
        uint h  = rem / uint(NQ);
        uint qb = rem - h * uint(NQ);

        uint kv_head = h / uint(GQA);

        int q_row = int(qb) * BQ + int(simd_gid);     // query index within [0, qL)
        bool row_active = (q_row < qLi);
        int q_pos = offi + q_row;                      // absolute query position

        // KV staging tiles (row-major [BK][D]); shared by all BQ query rows.
        threadgroup T Ks[BK * D];
        threadgroup T Vs[BK * D];

        // This simdgroup's query row: each lane owns ELEMS head-dim elements.
        U qreg[ELEMS];
        U acc[ELEMS];
        for (int e = 0; e < ELEMS; ++e) { acc[e] = 0; }
        U m = Limits<U>::finite_min;    // running max
        U l = 0;                         // running denominator

        if (row_active) {
            size_t qbase =
                ((size_t)(b * uint(HQ) + h) * (size_t)qLi + (size_t)q_row) * (size_t)D
                + (size_t)(simd_lid * ELEMS);
            const device T* qp = queries + qbase;
            for (int e = 0; e < ELEMS; ++e) {
                qreg[e] = static_cast<U>(scale) * static_cast<U>(qp[e]);
            }
        }

        // Block-level KV bounds (uniform across the whole threadgroup so every
        // thread runs the same number of tile iterations and hits every barrier).
        int block_q0 = int(qb) * BQ;
        int last_row = block_q0 + BQ - 1;
        if (last_row > qLi - 1) last_row = qLi - 1;
        int block_q_min = offi + block_q0;
        int block_q_max = offi + last_row;
        int kv_hi = block_q_max;                 // last key any row in block can see
        if (kv_hi > kLi - 1) kv_hi = kLi - 1;
        int kv_lo = 0;
        if (WINDOW > 0) {
            kv_lo = block_q_min - WINDOW + 1;
            if (kv_lo < 0) kv_lo = 0;
        }

        int tid_in_tg = int(simd_gid) * BD + int(simd_lid);   // 0..TG-1

        for (int tile_start = (kv_lo / BK) * BK;
             tile_start <= kv_hi;
             tile_start += BK) {

            // Cooperatively stage K/V tile [BK][D] into threadgroup memory.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int idx = tid_in_tg; idx < BK * D; idx += TG) {
                int kk = idx / D;
                int dd = idx - kk * D;
                int kpos = tile_start + kk;
                if (kpos < kLi) {
                    size_t kvbase =
                        ((size_t)(b * uint(HK) + kv_head) * (size_t)kLi
                         + (size_t)kpos) * (size_t)D + (size_t)dd;
                    Ks[idx] = keys[kvbase];
                    Vs[idx] = values[kvbase];
                } else {
                    Ks[idx] = T(0);
                    Vs[idx] = T(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (row_active) {
                for (int kk = 0; kk < BK; ++kk) {
                    int kpos = tile_start + kk;

                    // Positional mask (uniform across the 32 lanes of this row's
                    // simdgroup, so simd_sum stays in uniform control flow).
                    bool valid = (kpos < kLi) && (kpos <= q_pos);
                    if (WINDOW > 0) {
                        valid = valid && ((q_pos - kpos) < WINDOW);
                    }

                    threadgroup const T* kp = Ks + kk * D + simd_lid * ELEMS;
                    U partial = 0;
                    for (int e = 0; e < ELEMS; ++e) {
                        partial += qreg[e] * static_cast<U>(kp[e]);
                    }
                    U score = simd_sum(partial);   // full QK dot over head_dim

                    if (valid) {
                        U new_m = max(m, score);
                        U corr  = fast::exp(m - new_m);
                        U p     = fast::exp(score - new_m);
                        m = new_m;
                        l = l * corr + p;
                        threadgroup const T* vp = Vs + kk * D + simd_lid * ELEMS;
                        for (int e = 0; e < ELEMS; ++e) {
                            acc[e] = acc[e] * corr + p * static_cast<U>(vp[e]);
                        }
                    }
                }
            }
        }

        if (row_active) {
            U inv = (l > 0) ? (U(1) / l) : U(0);
            size_t obase =
                ((size_t)(b * uint(HQ) + h) * (size_t)qLi + (size_t)q_row) * (size_t)D
                + (size_t)(simd_lid * ELEMS);
            device T* op = out + obase;
            for (int e = 0; e < ELEMS; ++e) {
                op[e] = static_cast<T>(acc[e] * inv);
            }
        }
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_steel_attn_hq{hq}_hk{hk}_w{window}",
        input_names=["queries", "keys", "values", "scale", "qL", "kL", "qL_off"],
        output_names=["out"],
        header=header,
        source=source,
    )


def steel_attention_prefill(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    causal: bool = True,
    window: int = 0,
) -> Optional[mx.array]:
    """Run the tiled steel-attention prefill kernel.

    ``queries`` is ``[B, HQ, qL, D]``; ``keys``/``values`` are ``[B, HK, kL, D]``
    (``kL >= qL``).  Applies causal masking, plus a sliding window of ``window``
    positions when ``window > 0`` (S-2.1 sliding layers pass ``window=512``; full
    layers pass ``window=0``).  The query at row ``i`` is treated as absolute
    position ``(kL - qL) + i`` (standard causal offset).

    Returns the ``[B, HQ, qL, D]`` output, or ``None`` if the shape/dtype is not
    covered (caller should fall back to stock SDPA).
    """

    if not is_steel_attention_eligible(queries, keys, values, causal=causal):
        return None

    b, hq, ql, d = (int(v) for v in queries.shape)
    hk = int(keys.shape[1])
    kl = int(keys.shape[2])
    gqa = hq // hk
    window = int(window) if window and window > 0 else 0
    q_off = kl - ql

    nq = (ql + _BQ - 1) // _BQ
    num_tg = b * hq * nq
    tg_threads = _BQ * _BD

    kernel = _steel_attention_kernel(hq, hk, gqa, window)
    (out,) = kernel(
        inputs=[queries, keys, values, float(scale), int(ql), int(kl), int(q_off)],
        template=[("T", queries.dtype)],
        grid=(num_tg * tg_threads, 1, 1),
        threadgroup=(tg_threads, 1, 1),
        output_shapes=[(b, hq, ql, d)],
        output_dtypes=[queries.dtype],
    )
    assert tuple(out.shape) == (b, hq, ql, d), (out.shape, (b, hq, ql, d))
    return out


def steel_attention_or_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask=None,
    causal: bool = True,
    window: int = 0,
) -> mx.array:
    """Steel-attention prefill kernel when eligible, else stock SDPA.

    ``mask`` is what stock ``mx.fast.scaled_dot_product_attention`` should
    receive on the fallback path (``"causal"``, a boolean/additive array, or
    ``None``).  When the kernel is eligible it does its own positional masking
    from ``causal`` / ``window`` and ``mask`` is ignored.
    """

    out = steel_attention_prefill(
        queries, keys, values, scale=scale, causal=causal, window=window
    )
    if out is None:
        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale, mask=mask
        )
    b, hq, ql, d = (int(v) for v in queries.shape)
    assert tuple(out.shape) == (b, hq, ql, d), (out.shape, (b, hq, ql, d))
    return out


# ---------------------------------------------------------------------------
# Pure-mx numeric references (CPU-validatable; no Metal)
# ---------------------------------------------------------------------------
def attention_mask_bool(
    q_len: int,
    k_len: int,
    *,
    causal: bool = True,
    window: int = 0,
) -> mx.array:
    """Boolean keep-mask ``[q_len, k_len]`` matching mlx-lm ``create_causal_mask``.

    Query row ``i`` is absolute position ``(k_len - q_len) + i``.  Keep key ``j``
    iff ``j <= i_abs`` (causal) and, when ``window > 0``, ``i_abs - j < window``
    (sliding).  This is exactly the mask ``create_attention_mask`` materializes
    for the S-2.1 sliding layers at ``N > window`` and equivalent to the
    ``"causal"`` string for the full layers.
    """

    off = k_len - q_len
    i = mx.arange(off, off + q_len)[:, None]
    j = mx.arange(k_len)[None]
    keep = i >= j if causal else mx.ones((q_len, k_len), dtype=mx.bool_)
    if window and window > 0:
        keep = keep & (i < j + window)
    return keep


def reference_masked_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    causal: bool = True,
    window: int = 0,
) -> mx.array:
    """Naive masked ``softmax(QK^T * scale + mask) @ V`` reference (the truth).

    Fully materializes the score matrix; GQA handled by repeating KV heads.
    Computed in float32 for a stable oracle regardless of input dtype.
    """

    b, hq, ql, d = queries.shape
    hk = keys.shape[1]
    kl = keys.shape[2]
    rep = hq // hk

    q = queries.astype(mx.float32)
    k = keys.astype(mx.float32)
    v = values.astype(mx.float32)
    if rep > 1:
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)

    scores = (q @ k.swapaxes(-1, -2)) * scale             # [B, HQ, qL, kL]
    keep = attention_mask_bool(ql, kl, causal=causal, window=window)  # [qL, kL]
    neg = mx.array(-1e30, dtype=mx.float32)
    scores = mx.where(keep[None, None], scores, neg)
    weights = mx.softmax(scores, axis=-1, precise=True)
    out = weights @ v                                     # [B, HQ, qL, D]
    return out.astype(queries.dtype)


def reference_online_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    causal: bool = True,
    window: int = 0,
    bk: int = _BK,
) -> mx.array:
    """Online-softmax (flash) reference mirroring the kernel's accumulation.

    Streams the K/V sequence in ``bk``-row tiles maintaining a running max /
    running denominator / running weighted-V accumulator, exactly as the Metal
    kernel does (natural ``exp``, scale folded into Q).  Proves the flash
    recurrence is numerically equivalent to :func:`reference_masked_sdpa`
    *without any GPU* — a real CPU check of the ported algorithm.
    """

    b, hq, ql, d = queries.shape
    hk = keys.shape[1]
    kl = keys.shape[2]
    rep = hq // hk

    q = queries.astype(mx.float32) * scale
    k = keys.astype(mx.float32)
    v = values.astype(mx.float32)
    if rep > 1:
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)

    off = kl - ql
    q_abs = mx.arange(off, off + ql)[None, None, :, None]  # [1,1,qL,1]
    ninf = float("-inf")

    m = mx.full((b, hq, ql), ninf, dtype=mx.float32)
    l = mx.zeros((b, hq, ql), dtype=mx.float32)
    acc = mx.zeros((b, hq, ql, d), dtype=mx.float32)

    for t0 in range(0, kl, bk):
        t1 = min(t0 + bk, kl)
        kt = k[:, :, t0:t1, :]                              # [B,HQ,tk,D]
        vt = v[:, :, t0:t1, :]
        st = q @ kt.swapaxes(-1, -2)                        # [B,HQ,qL,tk]

        j = mx.arange(t0, t1)[None, None, None, :]          # [1,1,1,tk]
        keep = j <= q_abs                                    # causal
        if window and window > 0:
            keep = keep & (q_abs - j < window)
        st = mx.where(keep, st, mx.array(ninf, mx.float32))

        tmax = st.max(axis=-1)                               # [B,HQ,qL]; -inf if none
        new_m = mx.maximum(m, tmax)
        # Guard the all-masked-so-far rows (new_m == -inf) against inf-inf NaNs.
        safe_m = mx.where(mx.isinf(new_m), mx.array(0.0, mx.float32), new_m)
        corr = mx.exp(mx.where(mx.isinf(m), mx.array(0.0, mx.float32), m) - safe_m)
        p = mx.exp(st - safe_m[..., None])                   # masked -> 0

        l = l * corr + p.sum(axis=-1)
        acc = acc * corr[..., None] + (p @ vt)
        m = new_m

    out = acc / mx.where(l > 0, l, mx.array(1.0, mx.float32))[..., None]
    return out.astype(queries.dtype)
