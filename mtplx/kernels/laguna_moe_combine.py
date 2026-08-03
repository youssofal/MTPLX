"""D10 -- Laguna S-2.1 fused MoE combine: weighted expert reduce + routed
scale (2.5) + shared-expert add + residual add, in one dispatch.

Ported from the mlx.fast *Laguna XS2.1* challenge MoE-tail fusion
``lagunaPrefillMoETailKernel`` / ``lagunaSharedDownResidualKernel``
(Sources/MLXFastModel/LagunaRuntimeModel.swift, ~L9328-9360, L6560-6622),
reshaped from the challenge's 8-experts / hidden-2048 to Laguna **S-2.1**'s
**top-10 experts / hidden-3072**.

## What the challenge kernel fuses

After the routed experts produce their per-expert outputs, the challenge tail
collapses the whole combine into one dispatch, one thread per output column:

    total  = sum_k( bf16( expert_out[k] * weight[k] ) )    # bf16 accum from 0
    scaled = bf16( total * bf16(2.5) )                      # routed scale
    r2     = bf16( scaled + shared )                        # shared-expert add
    out    = bf16( residual + r2 )                          # residual add

replacing the stock chain of a broadcast-multiply, an axis reduce, a shared add
and a separate residual add.

## S-2.1 numerics

S-2.1's stock combine (``laguna._stock_moe_combine``) is
``(expert_out * weights[..., None]).sum(-2) + shared`` in bf16, and its residual
add lives one level up in the decoder layer (``hidden + mlp(...)``). Two S-2.1
shape facts drive the port:

* **The routed 2.5 scale is pre-baked into ``weights`` upstream** (the router
  epilogue multiplies by ``moe_routed_scaling_factor``), so the default combine
  needs no in-kernel scale. The challenge applies 2.5 in-kernel to raw weights;
  by distributivity ``sum(x*w)*2.5 == sum(x*(w*2.5))`` up to bf16 rounding, so a
  ``scale`` param is offered for the challenge's literal raw-weight form.
* **top_k = 10, not 8.** MLX's bf16 reduction over the top-k axis is
  ``col_reduce_small`` with ``threadgroup_y = min(8, top_k)`` partials combined
  in ascending order -- NOT a strict left fold. For k=8 those coincide (the
  challenge's in-order sum is bit-exact); for k=10 they do not. This port
  reproduces MLX's ``TY = min(8, top_k)`` partial-accumulation order (the same
  order ``laguna_decode.fused_moe_combine`` already uses) so the reduce stays
  **bit-exact with the stock combine at k=10**. The challenge's literal
  strict-in-order sum would reassociate 2 of the 10 terms; this is the one
  deliberate, documented deviation, made for digest-parity on S-2.1.

## Entry points

* ``fused_moe_combine(expert_out, weights, shared)`` -- residual-free drop-in
  for ``laguna.MOE_COMBINE_IMPL`` / ``laguna_decode.fused_moe_combine``. No
  scale multiply (weights pre-scaled). Bit-exact with the stock combine.
* ``fused_moe_combine_residual(expert_out, weights, shared, residual, scale=1.0)``
  -- the challenge's full tail: reduce (+ optional scale) + shared + residual in
  one dispatch. Default ``scale=1.0`` assumes S-2.1's pre-scaled weights; pass
  raw weights + ``scale=2.5`` for the challenge's literal in-kernel scale.

Both fall back to the exact stock op chain on any unsupported
shape/dtype/device (Metal is GPU-only), so callers never own a correctness
branch.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

# S-2.1 real combine shape (documentation).
LAGUNA_S21_TOP_K = 10
LAGUNA_S21_HIDDEN = 3072
LAGUNA_S21_ROUTED_SCALING = 2.5


def _on_metal_device() -> bool:
    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def _combine_shapes_ok(
    expert_out: mx.array, weights: mx.array, shared: mx.array
) -> bool:
    if expert_out.ndim != 3 or weights.ndim != 2 or shared.ndim != 2:
        return False
    if expert_out.dtype not in (mx.bfloat16, mx.float16):
        return False
    if shared.dtype != expert_out.dtype:
        return False
    if weights.dtype not in (mx.float32, expert_out.dtype):
        return False
    rows, top_k, hidden = (int(dim) for dim in expert_out.shape)
    if top_k <= 0 or top_k > 32:
        return False
    if (int(weights.shape[0]), int(weights.shape[1])) != (rows, top_k):
        return False
    return (int(shared.shape[0]), int(shared.shape[1])) == (rows, hidden)


def is_moe_combine_eligible(
    expert_out: mx.array, weights: mx.array, shared: mx.array
) -> bool:
    if not _on_metal_device():
        return False
    return _combine_shapes_ok(expert_out, weights, shared)


def is_moe_combine_residual_eligible(
    expert_out: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array,
) -> bool:
    if not _on_metal_device():
        return False
    if not _combine_shapes_ok(expert_out, weights, shared):
        return False
    if residual.dtype != expert_out.dtype:
        return False
    return tuple(int(d) for d in residual.shape) == tuple(int(d) for d in shared.shape)


@lru_cache(maxsize=None)
def _moe_combine_kernel(top_k: int, hidden: int, with_residual: bool, with_scale: bool):
    ty = min(8, top_k)
    header = f"""
        using namespace metal;
        constant constexpr int TOP_K = {top_k};
        constant constexpr int HIDDEN = {hidden};
        constant constexpr int TY = {ty};
    """

    scale_line = (
        "        total = static_cast<T>(total * static_cast<T>(scale));\n"
        if with_scale
        else ""
    )
    residual_line = (
        "        out = static_cast<T>(residual_in[idx] + out);\n"
        if with_residual
        else ""
    )

    # One thread per output column. Reproduce col_reduce_small's threadgroup_y=TY
    # accumulation exactly: partial y takes rows {y, y+TY, ...} in order, then
    # partials combine in ascending y as op(partial, running). Scale (optional)
    # is applied to the reduced total, then shared add, then residual add --
    # matching the challenge tail's `scaled -> +shared -> residual + r2` operand
    # order (bf16 add is round-symmetric, so `residual + r2 == r2 + residual`).
    source = (
        """
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
            T wv = static_cast<T>(weights[(size_t)row * TOP_K + r]);
            T prod = base_ptr[(size_t)r * HIDDEN] * wv;
            totals[r % TY] = prod + totals[r % TY];
        }
        T total = totals[0];
        for (int y = 1; y < TY; ++y) {
            total = totals[y] + total;
        }
"""
        + scale_line
        + "        T out = static_cast<T>(total + shared_in[idx]);\n"
        + residual_line
        + "        combined[idx] = out;\n"
    )

    input_names = ["expert_out", "weights", "shared_in"]
    if with_residual:
        input_names.append("residual_in")
    if with_scale:
        input_names.append("scale")

    tag = f"k{top_k}_h{hidden}"
    if with_residual:
        tag += "_res"
    if with_scale:
        tag += "_scale"
    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_moe_combine_{tag}",
        input_names=input_names,
        output_names=["combined"],
        header=header,
        source=source,
    )


def _stock_combine(
    expert_out: mx.array, weights: mx.array, shared: mx.array
) -> mx.array:
    """Exact stock combine (== ``laguna._stock_moe_combine``)."""

    combined = (expert_out * weights.astype(expert_out.dtype)[..., None]).sum(axis=-2)
    return combined + shared


def moe_combine_reference(
    expert_out: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array | None = None,
    *,
    scale: float = 1.0,
) -> mx.array:
    """Pure-mx mirror of the fused kernel (any device). Scale is applied to the
    reduced total (kernel operand order), then shared, then residual."""

    combined = (expert_out * weights.astype(expert_out.dtype)[..., None]).sum(axis=-2)
    if scale != 1.0:
        # Cast scale to T first, then T*T -- matches the kernel's
        # `static_cast<T>(total * static_cast<T>(scale))` operand order.
        combined = combined * mx.array(scale, dtype=expert_out.dtype)
    out = combined + shared
    if residual is not None:
        out = residual + out
    return out


def fused_moe_combine(
    expert_out: mx.array, weights: mx.array, shared: mx.array
) -> mx.array:
    """Weighted expert combine + shared-expert add, one dispatch. Drop-in for
    ``laguna.MOE_COMBINE_IMPL``. Falls back to the stock chain when ineligible.
    """

    if not is_moe_combine_eligible(expert_out, weights, shared):
        return _stock_combine(expert_out, weights, shared)

    rows, top_k, hidden = (int(dim) for dim in expert_out.shape)
    kernel = _moe_combine_kernel(top_k, hidden, with_residual=False, with_scale=False)
    total = rows * hidden
    (combined,) = kernel(
        inputs=[expert_out, weights, shared],
        template=[("T", expert_out.dtype)],
        grid=(total, 1, 1),
        threadgroup=(256 if total >= 256 else 32, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[expert_out.dtype],
    )
    # Fake-speedup / miswire guard: one output row per token row, HIDDEN wide.
    assert tuple(combined.shape) == (rows, hidden), (
        f"moe_combine produced {tuple(combined.shape)}, expected {(rows, hidden)}"
    )
    return combined


def fused_moe_combine_residual(
    expert_out: mx.array,
    weights: mx.array,
    shared: mx.array,
    residual: mx.array,
    *,
    scale: float = 1.0,
) -> mx.array:
    """The challenge's full MoE tail: weighted reduce (+ optional routed scale)
    + shared add + residual add, one dispatch.

    ``scale=1.0`` (default) assumes S-2.1's pre-scaled weights and is bit-exact
    with ``_stock_moe_combine(...) + residual``. Pass raw (normalized-only)
    weights + ``scale=2.5`` for the challenge's literal in-kernel routed scale.
    Falls back to the stock chain when ineligible.
    """

    if not is_moe_combine_residual_eligible(expert_out, weights, shared, residual):
        return moe_combine_reference(
            expert_out, weights, shared, residual, scale=scale
        )

    rows, top_k, hidden = (int(dim) for dim in expert_out.shape)
    with_scale = scale != 1.0
    kernel = _moe_combine_kernel(
        top_k, hidden, with_residual=True, with_scale=with_scale
    )
    total = rows * hidden
    inputs = [expert_out, weights, shared, residual]
    if with_scale:
        inputs.append(float(scale))
    (combined,) = kernel(
        inputs=inputs,
        template=[("T", expert_out.dtype)],
        grid=(total, 1, 1),
        threadgroup=(256 if total >= 256 else 32, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[expert_out.dtype],
    )
    assert tuple(combined.shape) == (rows, hidden), (
        f"moe_combine_residual produced {tuple(combined.shape)}, "
        f"expected {(rows, hidden)}"
    )
    return combined
