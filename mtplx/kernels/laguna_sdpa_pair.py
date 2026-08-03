"""Group-3 GQA decode attention for Laguna S-2.1.

Ported from the mlx.fast **Laguna XS2.1** challenge kernel ``sdpa_vector`` (its
``DARKBLOOM_GQA_PAIR_HEADS`` fast path), re-expressed as a Python
``mx.fast.metal_kernel`` and *adapted* to Laguna S-2.1's head geometry.

## The adaptation that matters (why this is not a verbatim port)

The challenge kernel shares each K/V device read across **two** adjacent query
heads that map to the same KV head, then keeps independent online-softmax state
per head.  Its own predicate is ``gqa_factor == 8 || gqa_factor == 6`` and it
documents that *both factors must be even* so no adjacent pair straddles a
KV-head ownership boundary.

Laguna S-2.1 has TWO attention geometries:

    full-attention layers  : 48 q-heads / 8 kv-heads -> gqa_factor 6  (even)
    sliding-attention layers: 72 q-heads / 8 kv-heads -> gqa_factor 9  (ODD)

The challenge's pair-of-2 is *silently wrong* on the sliding layers: with
``gqa_factor == 9`` the adjacent pair ``(8, 9)`` spans query heads owned by KV
heads 0 and 1, so a verbatim port would read the wrong KV rows for one head of
every ninth pair.  A brain-dead port would corrupt 36 of the 48 layers.

The correct, unified adaptation is **group-of-3**: ``3`` divides both ``6`` and
``9``, so one kernel serves both layer families, every group stays inside one
KV head (``GQA % 3 == 0``), and each K/V row is now shared across **three**
query heads instead of two — a larger decode KV-bandwidth reduction than the
donor's, which is exactly the term that dominates long-context decode attention
(the stock vector kernel launches one threadgroup per query head and re-reads
the KV rows once per head).

## Topology (mirrors the stock sdpa_vector)

1024-thread threadgroup = 32 simdgroups x 32 lanes.  ``simd_gid`` strides over
the KV sequence; the 32 lanes of a simdgroup cooperate on one 128-wide QK dot
(``qk_per_thread == 4``) via ``simd_sum``.  Each threadgroup owns GROUP=3 query
heads sharing one KV head, reads each K and V row once, and updates three
independent online-softmax accumulators from the shared registers.  The
cross-simdgroup combine is the stock single-plane reduction, run once per head.

Only the KV-reuse is ported here; the challenge's exchange-plane barrier trick
(measured +0.60% there) is intentionally left out of this first cut — it is a
separate, smaller optimisation and the win here is the 3x KV read reduction.

Eligibility is narrow and matches the decode fast path: single query token, no
mask (the sliding RotatingKVCache returns no mask at decode; full layers are
non-causal at q_len==1), head_dim 128, ``gqa_factor`` a multiple of 3.  The
public helper returns ``None`` on anything else so callers fall back to stock
``scaled_dot_product_attention``.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_GROUP = 3
_HEAD_DIM = 128


def is_grouped_gqa_sdpa_eligible(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    mask=None,
) -> bool:
    """Whether the group-3 decode SDPA kernel covers this exact shape."""

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
    if gqa % _GROUP != 0:
        return False
    if n <= 0:
        return False
    if queries.dtype not in (mx.bfloat16, mx.float16):
        return False
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return False
    return True


@lru_cache(maxsize=None)
def _grouped_gqa_sdpa_kernel(d: int, group: int, gqa: int, hq: int, hk: int):
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
        uint simd_gid = simdgroup_index_in_threadgroup;   // 0..31, strides KV
        uint simd_lid = thread_index_in_simdgroup;         // 0..31, over head_dim

        threadgroup U outputs[BN * BD];
        threadgroup U max_scores[BN];
        threadgroup U sum_exp_scores[BN];

        uint b = tg / uint(GROUPS_PER_BATCH);
        uint hg = tg - b * uint(GROUPS_PER_BATCH);
        uint q_head_base = hg * uint(GROUP);
        uint kv_head = q_head_base / uint(GQA);

        size_t k_base = ((size_t)(b * uint(HK) + kv_head) * (size_t)N) * (size_t)D;
        const device T* kptr = keys + k_base
            + (size_t)simd_gid * (size_t)D + simd_lid * QK_PER_THREAD;
        const device T* vptr = values + k_base
            + (size_t)simd_gid * (size_t)D + simd_lid * V_PER_THREAD;
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

        // Scan KV: read k and v ONCE per key, reuse across all GROUP heads.
        for (int i = simd_gid; i < N; i += BN) {
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

        // Combine the 32 simdgroup partials, once per head (stock reduction).
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
            for (int i = 0; i < V_PER_THREAD; ++i) {
                outputs[simd_lid * BD + simd_gid] = o[g][i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                U red = simd_sum(outputs[simd_gid * BD + simd_lid] * factor);
                o[g][i] = gsum == 0 ? red : red / gsum;
                if (i + 1 < V_PER_THREAD) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            device T* outp = out
                + (size_t)(b * uint(HQ) + q_head_base + uint(g)) * (size_t)D
                + simd_gid * V_PER_THREAD;
            if (simd_lid == 0) {
                for (int i = 0; i < V_PER_THREAD; ++i) {
                    outp[i] = static_cast<T>(o[g][i]);
                }
            }
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_gqa{gqa}_group{group}_sdpa_d{d}_hq{hq}",
        input_names=["queries", "keys", "values", "scale", "N"],
        output_names=["out"],
        header=header,
        source=source,
    )


def grouped_gqa_sdpa_decode(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask=None,
) -> mx.array | None:
    """Group-3 GQA decode attention. Returns ``None`` on unsupported shapes.

    ``queries`` is ``[B, HQ, 1, D]``; ``keys``/``values`` are the contiguous
    ``[B, HK, N, D]`` cache buffers.  Output matches
    ``scaled_dot_product_attention``'s ``[B, HQ, 1, D]``.
    """

    if not is_grouped_gqa_sdpa_eligible(queries, keys, values, mask=mask):
        return None

    b, hq, _, d = (int(v) for v in queries.shape)
    hk = int(keys.shape[1])
    n = int(keys.shape[2])
    gqa = hq // hk

    # metal_kernel's ensure_row_contiguous (default) copies non-contiguous
    # inputs itself; an explicit mx.contiguous here only adds graph nodes to
    # the per-step decode path, so pass the buffers straight through.
    num_groups = b * (hq // _GROUP)
    kernel = _grouped_gqa_sdpa_kernel(d, _GROUP, gqa, hq, hk)
    (out,) = kernel(
        inputs=[queries, keys, values, float(scale), int(n)],
        template=[("T", queries.dtype)],
        grid=(num_groups * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(b, hq, 1, d)],
        output_dtypes=[queries.dtype],
    )
    return out
