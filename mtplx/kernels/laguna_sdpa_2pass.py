"""Two-pass (KV-split / flash-decode) GQA decode attention, parameterized by
the KV-reuse GROUP factor.

## Why this file exists (the fair-vehicle re-test)

An earlier probe put GQA KV-reuse inside a *single-pass* decode SDPA
(``laguna_sdpa_pair.py``): one threadgroup per query group scans the ENTIRE KV
sequence.  It lost to stock ``mx.fast.scaled_dot_product_attention`` at long
context.  But that was an unfair vehicle: MLX's production SDPA switches to a
**2-pass KV-split** (flash-decode) path at long N -- many threadgroups each own
a KV chunk, then a combine pass reduces them.  The single-pass kernel can only
ever launch ``HQ`` (48-72) threadgroups, so it is GPU-starved at long N no
matter how good its KV-reuse is.  The old "no-go" therefore confounded two
different things:

    (a) "KV-reuse doesn't help"                 <- the theory under test
    (b) "single-pass under-parallelizes at long N"  <- a vehicle artifact

This file removes (b).  It puts KV-reuse INSIDE a 2-pass kernel so both arms use
the *same tiling*; only the GROUP knob changes.  Then the delta isolates
KV-reuse cleanly.

## The two passes

Pass 1 (partial): split the N keys into ``S`` chunks.  Launch
``(B * HQ/GROUP * S)`` threadgroups.  Each threadgroup owns GROUP query heads
that share one KV head AND one KV chunk.  It reads that KV chunk ONCE and
produces GROUP online-softmax partial states ``(m, l, acc)`` for its chunk --
exactly the per-key math of ``laguna_sdpa_pair.py``, just restricted to a chunk
and left UN-normalized.

Pass 2 (combine): for each (b, query-head), reduce the ``S`` partials with the
standard flash-decode log-sum-exp combine into the final output.

## The knob

``GROUP`` is the only thing that changes between arms:

    GROUP == 1     : per-head 2-pass control.  Each query head re-reads its KV
                     chunk (this is MLX's approach).  Max threadgroups, max KV
                     bandwidth.
    GROUP == gqa   : read each KV chunk once per KV head.  Max reuse, fewest
                     threadgroups.

Same 2-pass tiling across arms, so ``GROUP>1`` vs ``GROUP==1`` at fixed ``S`` is
a clean isolation of the KV-reuse term.

GROUP must divide gqa so every group stays inside one KV head.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_HEAD_DIM = 128


def is_two_pass_eligible(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    group: int,
    mask=None,
) -> bool:
    if not mx.metal.is_available():
        return False
    if mask is not None:
        return False
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return False
    b, hq, ql, d = (int(v) for v in queries.shape)
    if ql != 1 or d != _HEAD_DIM:
        return False
    bk, hk, n, dk = (int(v) for v in keys.shape)
    if bk != b or dk != d:
        return False
    if tuple(values.shape) != (b, hk, n, d):
        return False
    if hk <= 0 or hq % hk != 0:
        return False
    gqa = hq // hk
    if group <= 0 or gqa % group != 0:
        return False
    if n <= 0:
        return False
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return False
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return False
    return True


@lru_cache(maxsize=None)
def _pass1_kernel(d: int, group: int, gqa: int, hq: int, hk: int):
    header = f"""
        using namespace metal;
        constant constexpr int D = {d};
        constant constexpr int GROUP = {group};
        constant constexpr int GQA = {gqa};
        constant constexpr int HQ = {hq};
        constant constexpr int HK = {hk};
        constant constexpr int BN = 32;
        constant constexpr int BD = 32;
        constant constexpr int QK_PER_THREAD = D / BD;
        constant constexpr int V_PER_THREAD = D / BD;
        constant constexpr int GROUPS_PER_BATCH = HQ / GROUP;
    """

    source = """
        typedef float U;
        uint tg = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;   // 0..31, strides KV chunk
        uint simd_lid = thread_index_in_simdgroup;         // 0..31, over head_dim

        threadgroup U outputs[BN * BD];
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];

        // Decode threadgroup id -> (batch, query-group, chunk)
        uint Su = uint(S);
        uint per_batch = uint(GROUPS_PER_BATCH) * Su;
        uint b = tg / per_batch;
        uint rem = tg - b * per_batch;
        uint hg = rem / Su;
        uint s = rem - hg * Su;

        uint q_head_base = hg * uint(GROUP);
        uint kv_head = q_head_base / uint(GQA);

        // Chunk boundaries over the N keys.
        int chunk_len = (N + int(S) - 1) / int(S);
        int chunk_start = int(s) * chunk_len;
        int chunk_end = chunk_start + chunk_len;
        if (chunk_end > N) chunk_end = N;

        size_t k_base = ((size_t)(b * uint(HK) + kv_head) * (size_t)N) * (size_t)D;
        const device T* kptr = keys + k_base
            + (size_t)(chunk_start + int(simd_gid)) * (size_t)D + simd_lid * QK_PER_THREAD;
        const device T* vptr = values + k_base
            + (size_t)(chunk_start + int(simd_gid)) * (size_t)D + simd_lid * V_PER_THREAD;
        int inner_stride = BN * D;

        // Load GROUP query heads (scaled), zero GROUP output accumulators.
        thread U q[GROUP][QK_PER_THREAD];
        thread U o[GROUP][V_PER_THREAD];
        U maxs[GROUP];
        U sums[GROUP];
        for (int g = 0; g < GROUP; ++g) {
            const device T* qp = queries
                + (size_t)(b * uint(HQ) + q_head_base + uint(g)) * (size_t)D
                + simd_lid * QK_PER_THREAD;
            for (int j = 0; j < QK_PER_THREAD; ++j) {
                q[g][j] = static_cast<U>(scale) * static_cast<U>(qp[j]);
            }
            for (int j = 0; j < V_PER_THREAD; ++j) {
                o[g][j] = 0;
            }
            maxs[g] = Limits<U>::finite_min;
            sums[g] = 0;
        }

        // Scan THIS chunk only: read k and v ONCE per key, reuse across GROUP heads.
        for (int i = chunk_start + int(simd_gid); i < chunk_end; i += BN) {
            U k[QK_PER_THREAD];
            U vv[V_PER_THREAD];
            for (int j = 0; j < QK_PER_THREAD; ++j) {
                k[j] = static_cast<U>(kptr[j]);
            }
            for (int j = 0; j < V_PER_THREAD; ++j) {
                vv[j] = static_cast<U>(vptr[j]);
            }
            for (int g = 0; g < GROUP; ++g) {
                U score = 0;
                for (int j = 0; j < QK_PER_THREAD; ++j) {
                    score += q[g][j] * k[j];
                }
                score = simd_sum(score);
                U new_max = max(maxs[g], score);
                U factor = fast::exp(maxs[g] - new_max);
                U exp_score = fast::exp(score - new_max);
                maxs[g] = new_max;
                sums[g] = sums[g] * factor + exp_score;
                for (int j = 0; j < V_PER_THREAD; ++j) {
                    o[g][j] = o[g][j] * factor + exp_score * vv[j];
                }
            }
            kptr += inner_stride;
            vptr += inner_stride;
        }

        // Combine the 32 simdgroup partials per head; store UN-normalized (m,l,acc).
        for (int g = 0; g < GROUP; ++g) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_lid == 0) {
                max_scores[simd_gid] = maxs[g];
                sum_exp_scores[simd_gid] = sums[g];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            U m = max_scores[simd_lid];
            U gmax = simd_max(m);
            U factor = fast::exp(m - gmax);
            U gsum = simd_sum(sum_exp_scores[simd_lid] * factor);

            uint qh = q_head_base + uint(g);
            size_t ml_idx = (size_t)(b * uint(HQ) + qh) * (size_t)S + (size_t)s;
            if (simd_gid == 0 && simd_lid == 0) {
                partial_m[ml_idx] = gmax;
                partial_l[ml_idx] = gsum;
            }
            size_t out_base =
                ((size_t)(b * uint(HQ) + qh) * (size_t)S + (size_t)s) * (size_t)D;
            for (int i = 0; i < V_PER_THREAD; ++i) {
                outputs[simd_lid * BD + simd_gid] = o[g][i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                U red = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
                if (simd_lid == 0) {
                    partial_out[out_base + (size_t)(simd_gid * V_PER_THREAD + i)] = red;
                }
                if (i + 1 < V_PER_THREAD) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_2pass_p1_gqa{gqa}_group{group}_d{d}_hq{hq}",
        input_names=["queries", "keys", "values", "scale", "N", "S"],
        output_names=["partial_out", "partial_m", "partial_l"],
        header=header,
        source=source,
    )


@lru_cache(maxsize=None)
def _pass2_kernel(d: int, hq: int):
    header = f"""
        using namespace metal;
        constant constexpr int D = {d};
        constant constexpr int HQ = {hq};
    """
    source = """
        typedef float U;
        uint tg = threadgroup_position_in_grid.x;         // which (b, query-head)
        uint t = thread_position_in_threadgroup.x;         // 0..D-1, one output component

        uint b = tg / uint(HQ);
        uint qh = tg - b * uint(HQ);
        uint Su = uint(S);

        size_t ml_base = (size_t)(b * uint(HQ) + qh) * (size_t)S;
        size_t po_base = ml_base * (size_t)D;

        // Global max over the S chunk-maxima.
        U gmax = Limits<U>::finite_min;
        for (uint c = 0; c < Su; ++c) {
            gmax = max(gmax, partial_m[ml_base + c]);
        }

        // Log-sum-exp combine of the S partials for this component.
        U L = 0;
        U acc = 0;
        for (uint c = 0; c < Su; ++c) {
            U m = partial_m[ml_base + c];
            U l = partial_l[ml_base + c];
            U f = fast::exp(m - gmax);
            L += l * f;
            acc += partial_out[po_base + (size_t)c * (size_t)D + t] * f;
        }
        U res = (L == 0) ? 0 : acc / L;
        out[(size_t)(b * uint(HQ) + qh) * (size_t)D + t] = static_cast<T>(res);
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_2pass_p2_d{d}_hq{hq}",
        input_names=["partial_out", "partial_m", "partial_l", "S"],
        output_names=["out"],
        header=header,
        source=source,
    )


def two_pass_gqa_sdpa_decode(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    group: int,
    chunks: int,
    mask=None,
) -> mx.array | None:
    """Two-pass KV-split decode SDPA with KV-reuse factor ``group`` and ``chunks``
    KV chunks. Returns ``None`` on unsupported shapes.

    ``queries`` is ``[B, HQ, 1, D]``; ``keys``/``values`` are ``[B, HK, N, D]``.
    Output matches ``scaled_dot_product_attention``'s ``[B, HQ, 1, D]``.
    """

    if not is_two_pass_eligible(queries, keys, values, group=group, mask=mask):
        return None

    b, hq, _, d = (int(v) for v in queries.shape)
    hk = int(keys.shape[1])
    n = int(keys.shape[2])
    gqa = hq // hk
    s = int(chunks)
    if s <= 0:
        return None

    groups_per_batch = hq // group

    k1 = _pass1_kernel(d, group, gqa, hq, hk)
    partial_out, partial_m, partial_l = k1(
        inputs=[queries, keys, values, float(scale), int(n), int(s)],
        template=[("T", queries.dtype)],
        grid=(b * groups_per_batch * s * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(b, hq, s, d), (b, hq, s), (b, hq, s)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )

    k2 = _pass2_kernel(d, hq)
    (out,) = k2(
        inputs=[partial_out, partial_m, partial_l, int(s)],
        template=[("T", queries.dtype)],
        grid=(b * hq * d, 1, 1),
        threadgroup=(d, 1, 1),
        output_shapes=[(b, hq, 1, d)],
        output_dtypes=[queries.dtype],
    )
    return out
