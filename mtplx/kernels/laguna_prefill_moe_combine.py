"""Fused MoE combine tail for the Laguna S-2.1 PREFILL sparse block.

Prefill twin of the decode ``fused_moe_combine`` in ``laguna_decode.py``, run
over ``M = T`` tokens.  After the grouped gather-GEMM
(``laguna_moe_gather_gemm.py``) has produced per-token expert outputs
``[M, top_k, hidden]``, this kernel does the whole combine tail in one dispatch:

    weighted reduce over top_k  ->  x routed_scaling (2.5)  ->  + shared expert
    ->  + residual

Ported from the challenge's own decode routed-down + combine fusion (the
"exact router reduction, routed scale, and BF16 residual add" path in
``Sources/MLXFastModel/LagunaRuntimeModel.swift`` -- prefill there stayed on the
stock separate ops), re-expressed prefill-shaped and for S-2.1's top-10 /
routed_scaling 2.5 / BF16 residual.

Why the scale and residual live HERE (they are split off the stock router/decoder
in the S-2.1 model).  The stock ``LagunaSparseMoeBlock`` folds routed_scaling
into the routing weights (``(w * 2.5).astype(x.dtype)``) before the combine, and
the ``DecoderLayer`` adds the residual after the block returns.  This tail takes
the NORMALIZED, UNSCALED float32 router weights (P3's output) and reproduces
both roundings in order: ``bfloat(w_f32 * 2.5)`` is the exact value the stock
weight-scale astype produces, and ``(reduce + shared) + residual`` is the exact
pair of BF16 adds the stock block-then-decoder does (float add is commutative,
so operand order within an add does not matter).

Bit-exactness of the reduction matches MLX's own ``col_reduce_small`` order for a
K-deep BF16 column reduction: ``TY = min(8, K)`` partial accumulators, partial y
summing rows ``{y, y+TY, ...}`` in ascending order, partials combined in
ascending y, everything in the tensor dtype -- identical to the decode kernel.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


def _on_metal_device() -> bool:
    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def is_moe_combine_prefill_eligible(
    expert_out: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array,
) -> bool:
    if not _on_metal_device():
        return False
    if expert_out.ndim != 3 or weights.ndim != 2:
        return False
    if shared.ndim != 2 or residual.ndim != 2:
        return False
    if expert_out.dtype not in (mx.bfloat16, mx.float16):
        return False
    if shared.dtype != expert_out.dtype or residual.dtype != expert_out.dtype:
        return False
    if weights.dtype != mx.float32:
        return False
    rows, top_k, hidden = (int(dim) for dim in expert_out.shape)
    if top_k <= 0 or top_k > 32:
        return False
    if (int(weights.shape[0]), int(weights.shape[1])) != (rows, top_k):
        return False
    if (int(shared.shape[0]), int(shared.shape[1])) != (rows, hidden):
        return False
    return (int(residual.shape[0]), int(residual.shape[1])) == (rows, hidden)


@lru_cache(maxsize=None)
def _moe_combine_prefill_kernel(top_k: int, hidden: int):
    ty = min(8, top_k)
    header = f"""
        using namespace metal;
        constant constexpr int TOP_K = {top_k};
        constant constexpr int HIDDEN = {hidden};
        constant constexpr int TY = {ty};
    """
    # One thread per output element (row, hidden column).  The routing weight for
    # expert r is shared across all hidden columns, so routed_scaling folds into
    # it once: wv = T(w_f32[r] * routed_scaling), exactly the stock weight-scale
    # astype.  The reduction reproduces col_reduce_small; the two trailing adds
    # reproduce (block combine + shared) then (decoder residual).
    source = """
        uint idx = thread_position_in_grid.x;
        uint row = idx / uint(HIDDEN);
        uint c = idx - row * uint(HIDDEN);

        const device T* base_ptr =
            expert_out + (size_t)row * (size_t)(TOP_K * HIDDEN) + c;

        T totals[TY];
        for (int y = 0; y < TY; ++y) {
            totals[y] = T(0);
        }
        for (int r = 0; r < TOP_K; ++r) {
            float wf = weights[(size_t)row * TOP_K + r] * routed_scaling;
            T wv = static_cast<T>(wf);
            T prod = base_ptr[(size_t)r * HIDDEN] * wv;
            totals[r % TY] = prod + totals[r % TY];
        }
        T total = totals[0];
        for (int y = 1; y < TY; ++y) {
            total = totals[y] + total;
        }
        combined[idx] = (total + shared_in[idx]) + residual_in[idx];
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_prefill_moe_combine_k{top_k}_h{hidden}",
        input_names=["expert_out", "weights", "shared_in", "residual_in", "routed_scaling"],
        output_names=["combined"],
        header=header,
        source=source,
    )


def fused_moe_combine_prefill(
    expert_out: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array,
    routed_scaling: float,
) -> mx.array:
    """Weighted combine + routed_scaling + shared-expert add + residual, one pass.

    ``expert_out`` is ``[M, top_k, hidden]`` (the gather-GEMM output), ``weights``
    is ``[M, top_k]`` float32 (P3's normalized, UNSCALED router weights),
    ``shared`` and ``residual`` are ``[M, hidden]``.  Returns ``[M, hidden]`` in
    the expert dtype.

    Falls back to the stock op chain on any shape the kernel does not cover.
    """

    if not is_moe_combine_prefill_eligible(expert_out, weights, shared, residual):
        w = (weights * routed_scaling).astype(expert_out.dtype)
        combined = (expert_out * w[..., None]).sum(axis=-2)
        return (combined + shared) + residual

    rows, top_k, hidden = (int(dim) for dim in expert_out.shape)
    kernel = _moe_combine_prefill_kernel(top_k, hidden)
    total = rows * hidden
    (combined,) = kernel(
        inputs=[expert_out, weights, shared, residual, float(routed_scaling)],
        template=[("T", expert_out.dtype)],
        grid=(total, 1, 1),
        threadgroup=(256 if total >= 256 else 32, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[expert_out.dtype],
    )
    # Fake-speedup guard: exactly one combined row per token, hidden wide.
    assert tuple(combined.shape) == (rows, hidden), (
        f"moe combine {tuple(combined.shape)} != {(rows, hidden)}"
    )
    return combined


def moe_combine_prefill_reference(
    expert_out: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array,
    routed_scaling: float,
) -> mx.array:
    """Pure-mx reference: the stock combine + scale + shared + residual.

    Identical to the stock fallback expression, kept separate so the CPU check
    reads as reference-vs-stock and to document the exact op order the kernel
    reproduces.
    """

    w = (weights * routed_scaling).astype(expert_out.dtype)  # bf16(w_f32 * 2.5)
    combined = (expert_out * w[..., None]).sum(axis=-2)  # bf16 col reduction
    return (combined + shared) + residual  # two bf16 adds
