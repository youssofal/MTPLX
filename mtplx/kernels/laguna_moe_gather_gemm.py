"""Hand grouped gather-GEMM for the Laguna S-2.1 MoE PREFILL expert path.

Ported from the mlx.fast **Laguna XS2.1** challenge prefill MoE kernel
``fp_gather_qmm_rhs_nax`` (Vendor/mlx-swift/.../fp_quantized_nax.cpp),
re-expressed for **Laguna S-2.1**'s affine oQ4e bank (affine 4-bit, group_size
128) instead of the challenge's NVFP4 (group-16).

## What the challenge kernel actually is

``fp_gather_qmm_rhs_nax`` is NOT a from-scratch GEMM: it is a *fork of MLX's own
steel tiled gather-GEMM* — the exact kernel ``mx.gather_qmm(sorted_indices=True)``
dispatches (BM64/BN64/BK32, WM2/WN2, 16x16 ``simdgroup_matrix`` MMA fragments).
The fork adds function constants that all default OFF, and the header states:
"when false the kernel is byte-for-byte the upstream algorithm" (fn-const 203).
The added levers are:

    RUNSKIP (fn-const 203)  -- elide MMAs for simdgroups whose output rows in a
                               sorted run are empty (dead work store_slice would
                               discard anyway); bit-exact, helps only when tiles
                               are fragmented / experts are empty.
    STAGE_WIDEST/WIDELD     -- wider threadgroup weight loads (byte-identical).
    STAGE2_GATHER           -- register-staged double-buffer (prefetch tile k+1).

The challenge's own measured verdict (their notes): the prefill NVFP4 GEMM is
the MLX builtin with "NO HEADROOM", the gather-staging levers were
"shelved-regressed", RUNSKIP is "inert by default, prefill-only M>=64", and a
split-K tile regroup was "+0.18%" (below their ~0.5-2.9% noise). Prefill also
weights only 0.25 of their score.

## What this module implements and tests

A genuine HAND grouped gather-GEMM for affine 4-bit gs128, structured as one
threadgroup per active expert that batches the expert's sorted token rows so the
quantized weight row is dequantized ONCE and reused across ``ROW_TILE`` tokens
(the grouped-GEMM weight-reuse win), with a RUNSKIP-style skip of empty experts.
It deliberately does NOT use ``simdgroup_matrix`` — it is the honest "can a hand
affine gather-GEMM beat stock ``mx.gather_qmm``" probe. Stock gather_qmm at
prefill is compute-bound and rides the hardware MMA + double-buffered steel
pipeline, so this is expected to LOSE; the companion check measures by how much
and confirms the "no prefill MoE lever" verdict for affine oQ4e.

Transpose=True convention (matches ``mx.gather_qmm(..., transpose=True)`` and
the S-2.1 gate/up/down banks): ``w`` is ``[E, N, K]``; output row m (expert
``e_m``) is ``y[m, n] = sum_k x[m, k] * dequant(w[e_m, n, k])``.

Use :func:`grouped_gather_gemm` (guarded) or the raw :func:`grouped_gather_gemm_t`.
Falls back to stock ``mx.gather_qmm(sorted_indices=True)`` on any unsupported
shape/dtype/quant.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


_BITS = 4
_GROUP_SIZE = 128
_PACK = 8
_WORDS_PER_GROUP = _GROUP_SIZE // _PACK  # 16


def _on_metal_device() -> bool:
    try:
        return mx.metal.is_available() and mx.default_device() == mx.Device(mx.gpu)
    except Exception:
        return False


def sorted_run_layout(row_expert: mx.array, num_experts: int):
    """Sort rows by expert and return (order, row_expert_sorted, start, count).

    ``start[e]`` / ``count[e]`` are the contiguous run of expert ``e`` inside the
    sorted stream (the layout the grouped kernel and stock's sorted path both
    consume). Computed with a stable sort so ties keep input order.
    """

    order = mx.argsort(row_expert.astype(mx.uint32))
    re_sorted = row_expert.astype(mx.uint32)[order]
    ar = mx.arange(num_experts, dtype=mx.uint32)
    # count[e] = number of sorted rows equal to e; start[e] = exclusive prefix.
    eq = (re_sorted[None, :] == ar[:, None]).astype(mx.int32)  # [E, M]
    count = eq.sum(axis=1).astype(mx.int32)                    # [E]
    start = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(count)[:-1]])
    return order, re_sorted, start.astype(mx.int32), count


def is_grouped_gather_eligible(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    row_start: mx.array,
    row_count: mx.array,
) -> bool:
    if not _on_metal_device():
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 2:
        return False
    if w.dtype != mx.uint32 or w.ndim != 3:
        return False
    if scales.ndim != 3 or biases.ndim != 3:
        return False
    if scales.dtype != mx.float32 or biases.dtype != mx.float32:
        return False
    E, N, KP = (int(v) for v in w.shape)
    K = KP * _PACK
    if K % _GROUP_SIZE != 0:
        return False
    KG = K // _GROUP_SIZE
    if tuple(scales.shape) != (E, N, KG) or tuple(biases.shape) != (E, N, KG):
        return False
    if int(x.shape[1]) != K:
        return False
    if row_start.ndim != 1 or row_count.ndim != 1:
        return False
    if int(row_start.shape[0]) != E or int(row_count.shape[0]) != E:
        return False
    return True


@lru_cache(maxsize=None)
def _grouped_gather_kernel(N: int, K: int, threads: int, row_tile: int):
    KP = K // _PACK
    KG = K // _GROUP_SIZE

    header = f"""
        using namespace metal;
        constant constexpr uint N_OUT = {N};
        constant constexpr uint K_IN = {K};
        constant constexpr uint KP = {KP};
        constant constexpr uint KG = {KG};
        constant constexpr uint TG = {threads};
        constant constexpr uint RT = {row_tile};
        constant constexpr uint WPG = {_WORDS_PER_GROUP};
    """

    # One threadgroup per expert (tg == expert id). RUNSKIP: empty experts return
    # immediately. Each thread owns a strided set of output columns n; for each
    # column it dequantizes the expert's weight row ONCE per ROW_TILE-sized chunk
    # of the expert's sorted token rows and accumulates into RT row-accumulators,
    # so the weight read is amortized across RT tokens (the grouped-GEMM win).
    source = """
        uint e = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;

        int cnt = row_count[e];
        if (cnt == 0) {           // RUNSKIP: skip empty expert tile
            return;
        }
        int rs = row_start[e];

        size_t w_e = (size_t)e * N_OUT * KP;
        size_t s_e = (size_t)e * N_OUT * KG;

        for (uint n = lid; n < N_OUT; n += TG) {
            const device uint*  wrow = w      + w_e + (size_t)n * KP;
            const device float* srow = scales + s_e + (size_t)n * KG;
            const device float* brow = biases + s_e + (size_t)n * KG;

            for (int r0 = 0; r0 < cnt; r0 += RT) {
                int rt = min((int)RT, cnt - r0);
                float acc[RT];
                for (uint i = 0; i < RT; ++i) acc[i] = 0.0f;

                for (uint g = 0; g < KG; ++g) {
                    float sc = srow[g];
                    float bi = brow[g];
                    uint kbase = g * WPG * 8u;
                    uint wbase = g * WPG;
                    for (uint wi = 0; wi < WPG; ++wi) {
                        uint word = wrow[wbase + wi];
                        uint k = kbase + wi * 8u;
                        for (uint t = 0; t < 8u; ++t) {
                            float wv = float((word >> (4u * t)) & 0xFu) * sc + bi;
                            uint kk = k + t;
                            for (int i = 0; i < rt; ++i) {
                                float xv = float(x[(size_t)(rs + r0 + i) * K_IN + kk]);
                                acc[i] += wv * xv;
                            }
                        }
                    }
                }
                for (int i = 0; i < rt; ++i) {
                    y[(size_t)(rs + r0 + i) * N_OUT + n] = acc[i];
                }
            }
        }
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_grouped_gather_gemm_n{N}_k{K}_t{threads}_r{row_tile}",
        input_names=["x", "w", "scales", "biases", "row_start", "row_count"],
        output_names=["y"],
        header=header,
        source=source,
    )


def grouped_gather_gemm_t(
    x: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    row_start: mx.array,
    row_count: mx.array,
    *,
    threads: int = 256,
    row_tile: int = 8,
) -> mx.array:
    """Grouped gather-GEMM ``y[M, N]`` (float32); x rows sorted by expert.

    Eligibility is the caller's responsibility here (raw entry). ``row_start`` /
    ``row_count`` describe each expert's contiguous run in the sorted stream
    (see :func:`sorted_run_layout`).
    """

    M = int(x.shape[0])
    K = int(x.shape[1])
    E, N, _ = (int(v) for v in w.shape)
    threads = int(threads)
    if threads <= 0 or threads > 1024:
        threads = 256

    kernel = _grouped_gather_kernel(N, K, threads, int(row_tile))
    (y,) = kernel(
        inputs=[x, w, scales, biases, row_start, row_count],
        grid=(threads * E, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
    )
    # Fake-speedup guard: exactly one output row per input token row, N wide.
    assert tuple(y.shape) == (M, N), (
        f"grouped_gather_gemm_t produced {tuple(y.shape)}, expected {(M, N)}"
    )
    return y


def grouped_gather_gemm(
    x_sorted: mx.array,
    w: mx.array,
    scales: mx.array,
    biases: mx.array,
    row_expert_sorted: mx.array,
    num_experts: int,
    *,
    threads: int = 256,
    row_tile: int = 8,
) -> mx.array:
    """Guarded drop-in: grouped gather-GEMM over sorted rows -> ``[M, N]`` f32.

    Falls back to stock ``mx.gather_qmm(sorted_indices=True)`` on any shape/dtype
    the hand kernel does not cover.
    """

    _, _, start, count = sorted_run_layout(row_expert_sorted, num_experts)
    if not is_grouped_gather_eligible(x_sorted, w, scales, biases, start, count):
        y = mx.gather_qmm(
            x_sorted.reshape(int(x_sorted.shape[0]), 1, int(x_sorted.shape[1])),
            w, scales, biases,
            rhs_indices=row_expert_sorted,
            transpose=True, group_size=_GROUP_SIZE, bits=_BITS, mode="affine",
            sorted_indices=True,
        )
        return y.reshape(int(x_sorted.shape[0]), int(w.shape[1]))
    return grouped_gather_gemm_t(
        x_sorted, w, scales, biases, start, count,
        threads=threads, row_tile=row_tile,
    )
