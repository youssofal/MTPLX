"""D12 -- Laguna S-2.1 fused MoE router: sigmoid + e_score_correction_bias +
top-k, selected by the challenge's **bitonic sort network** (not MTPLX's
tree-reduction selector).

Ported from the mlx.fast *Laguna XS2.1* challenge decode router
``lagunaDecodeRouterTop8Kernel`` (Sources/MLXFastModel/LagunaRuntimeModel.swift,
~L8192-8331), reshaped from the challenge's 256-expert / **top-8** selection to
Laguna **S-2.1**'s 256-expert / **top-10** selection.

## What the challenge kernel is

One threadgroup per token row, ``NUM_EXPERTS`` lanes (one lane per expert). Each
lane:

    x       = float(logits[e])
    score   = numerically-stable sigmoid(x)     # y=1/(1+exp(|x|)); x<0 ? y : 1-y
    key     = -(score + correction_bias[e])     # negate so "smaller key = better"
    index   = e

then the whole ``NUM_EXPERTS``-element vector is sorted by a **full Batcher
bitonic network** carrying ``(key, index, score)``, with a total-order
comparator (``key`` ascending, ties broken by original expert index; NaN sorts
last). The lower half of every merged sequence keeps the better entries, so
after the sort ranks ``0..NUM_EXPERTS`` live in lanes ``0..NUM_EXPERTS`` and the
first ``TOP_K`` lanes hold exactly the top-k experts, already in choice order.

Sorting all 256 then reading the first ``TOP_K`` lanes is the same SET as
``argpartition(-(sigmoid+bias), top_k-1)[:top_k]`` -- it just also orders them
and gives a stable, index-based tie-break the unordered argpartition does not
promise. That set-identity is the correctness contract this port is validated to
(``router_topk_reference`` below + the CPU check).

Intra-simdgroup stages (``stride < 32``) exchange operands through
``simd_shuffle_xor`` in registers (no threadgroup memory, no barrier); the
``stride >= 32`` stages cross a simdgroup boundary and go through threadgroup
scratch with barriers -- byte-for-byte the challenge's schedule.

## S-2.1 shaping / drop-in contract

This is a drop-in for ``mtplx.kernels.laguna_decode.fused_router_topk`` -- same
signature, same output contract ``(indices[rows, top_k] uint32,
weights[rows, top_k] float32)`` -- so it can be swapped in wherever the MTPLX
router is installed. The *selection* is the challenge's bitonic network; the
*output epilogue* keeps MTPLX's contract: ``normalize`` folds in the top-k
renormalization and ``scale`` folds in ``moe_routed_scaling_factor`` (2.5 for
S-2.1), so the returned weights are already ``score/sum * scale``. (The
challenge decode router leaves normalize/scale to downstream kernels; S-2.1's
stock path folds them into the weights before the combine, so this port folds
them here to stay a true drop-in -- documented deviation, same numbers.)

Falls back to the exact stock op chain (``sigmoid -> +bias -> argpartition ->
gather -> normalize -> scale``) on any shape/dtype/device the kernel does not
cover, so callers never own a correctness branch. On a CPU device the fallback
is always taken (``mx.fast.metal_kernel`` is GPU-only).
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

# S-2.1 real router shape (documentation / default guard bounds).
LAGUNA_S21_NUM_EXPERTS = 256
LAGUNA_S21_TOP_K = 10

DEFAULT_ROUTER_MAX_ROWS = 512


def _router_max_rows() -> int:
    raw = os.environ.get("MTPLX_LAGUNA_BITONIC_ROUTER_MAX_ROWS")
    if raw is None:
        return DEFAULT_ROUTER_MAX_ROWS
    try:
        return int(raw)
    except ValueError:
        return DEFAULT_ROUTER_MAX_ROWS


def _on_metal_device() -> bool:
    """Metal AVAILABLE is not Metal being the device in use.

    ``mx.fast.metal_kernel`` raises "Only supports the GPU" when the default
    device is the CPU, so eligibility must fail closed to the stock chain there.
    """

    if not mx.metal.is_available():
        return False
    try:
        return mx.default_device() == mx.gpu
    except Exception:
        return False


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def is_router_bitonic_eligible(
    logits: mx.array, correction_bias: mx.array, top_k: int
) -> bool:
    """The bitonic network requires a power-of-two expert count and a top_k that
    fits inside the first simdgroup (so the normalizing ``simd_shuffle`` sum over
    lanes ``0..top_k-1`` stays intra-simdgroup)."""

    if not _on_metal_device():
        return False
    if logits.ndim != 2 or correction_bias.ndim != 1:
        return False
    if logits.dtype != mx.float32 or correction_bias.dtype != mx.float32:
        return False
    experts = int(logits.shape[1])
    if experts != int(correction_bias.shape[0]):
        return False
    if not _is_pow2(experts) or not (32 <= experts <= 1024):
        return False
    if not (0 < top_k <= 32) or top_k > experts:
        return False
    if int(logits.shape[0]) > _router_max_rows():
        return False
    return True


_BITONIC_HEADER = """
    using namespace metal;

    inline bool laguna_router_key_before(
        float a, uint a_index, float b, uint b_index) {
        bool a_nan = metal::isnan(a);
        bool b_nan = metal::isnan(b);
        if (a_nan || b_nan) {
            if (a_nan != b_nan) {
                return !a_nan;   // a non-NaN sorts before a NaN
            }
            return a_index < b_index;
        }
        if (a < b) { return true; }
        if (b < a) { return false; }
        return a_index < b_index;
    }
"""


@lru_cache(maxsize=None)
def _router_bitonic_kernel(experts: int, top_k: int):
    header = _BITONIC_HEADER + f"""
        constant constexpr uint NUM_EXPERTS = {experts};
        constant constexpr uint TOP_K = {top_k};
    """

    # One threadgroup per row, NUM_EXPERTS lanes. Carries (key, index, score)
    # through the full bitonic network -- lane e ends holding rank-e, so lane
    # e's score is rank-e's score directly (no recompute needed in the
    # epilogue). `scale` / `normalize` are scalar inputs (MTPLX contract).
    source = """
        uint row  = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;

        threadgroup float xchg_keys[NUM_EXPERTS];
        threadgroup uint  xchg_indices[NUM_EXPERTS];
        threadgroup float xchg_scores[NUM_EXPERTS];

        float x = float(logits[row * NUM_EXPERTS + lane]);
        float y = 1.0f / (1.0f + metal::exp(metal::abs(x)));
        float my_score = x < 0.0f ? y : 1.0f - y;
        float my_key   = -(my_score + correction_bias[lane]);
        uint  my_index = lane;

        for (uint sequence = 2; sequence <= NUM_EXPERTS; sequence <<= 1) {
            for (uint stride = sequence >> 1; stride > 0; stride >>= 1) {
                float other_key;
                uint  other_index;
                float other_score;
                if (stride < 32) {
                    other_key   = simd_shuffle_xor(my_key,   ushort(stride));
                    other_index = simd_shuffle_xor(my_index, ushort(stride));
                    other_score = simd_shuffle_xor(my_score, ushort(stride));
                } else {
                    xchg_keys[lane]    = my_key;
                    xchg_indices[lane] = my_index;
                    xchg_scores[lane]  = my_score;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    uint partner = lane ^ stride;
                    other_key   = xchg_keys[partner];
                    other_index = xchg_indices[partner];
                    other_score = xchg_scores[partner];
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }

                bool is_lower = (lane & stride) == 0;
                float a_key   = is_lower ? my_key   : other_key;
                uint  a_index = is_lower ? my_index : other_index;
                float a_score = is_lower ? my_score : other_score;
                float b_key   = is_lower ? other_key   : my_key;
                uint  b_index = is_lower ? other_index : my_index;
                float b_score = is_lower ? other_score : my_score;

                bool lower_wants_better = (lane & sequence) == 0;
                bool b_before_a = laguna_router_key_before(
                    b_key, b_index, a_key, a_index);
                bool a_before_b = laguna_router_key_before(
                    a_key, a_index, b_key, b_index);
                bool swap = lower_wants_better ? b_before_a : a_before_b;
                if (swap) {
                    my_key   = is_lower ? b_key   : a_key;
                    my_index = is_lower ? b_index : a_index;
                    my_score = is_lower ? b_score : a_score;
                }
            }
        }

        // Ranks 0..TOP_K-1 live in lanes 0..TOP_K-1 of simdgroup 0. Run the
        // normalizing sum UNGUARDED so every simd_shuffle source lane is active;
        // only lanes < TOP_K write. The left-fold operand order reproduces the
        // stock `total = scores[i] + total` reduction.
        float total = 0.0f;
        if (normalize) {
            for (uint i = 0; i < TOP_K; ++i) {
                total = simd_shuffle(my_score, ushort(i)) + total;
            }
        }
        if (lane < TOP_K) {
            router_indices[row * TOP_K + lane] = my_index;
            float w = normalize ? (my_score / total) : my_score;
            router_scores[row * TOP_K + lane] = w * scale;
        }
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_router_bitonic_e{experts}_k{top_k}",
        input_names=["logits", "correction_bias", "scale", "normalize"],
        output_names=["router_indices", "router_scores"],
        header=header,
        source=source,
    )


def _stable_sigmoid(logits: mx.array) -> mx.array:
    """The kernel's exact numerically-stable sigmoid form (y=1/(1+exp(|x|)))."""

    y = 1.0 / (1.0 + mx.exp(mx.abs(logits)))
    return mx.where(logits < 0, y, 1.0 - y)


def router_topk_reference(
    logits: mx.array,
    correction_bias: mx.array,
    top_k: int,
    *,
    normalize: bool,
    scale: float,
) -> tuple[mx.array, mx.array]:
    """Pure-mx mirror of the bitonic kernel's SELECTION + epilogue.

    Sorts all experts by choice score descending (tie-break: lower expert index,
    which is what the bitonic total order gives), returns the sorted top-k
    indices (uint32) and their (optionally normalized, scaled) sigmoid weights.
    Used to prove the kernel's selection equals argpartition's set. Runs on any
    device (no Metal), so it doubles as the CPU oracle.
    """

    scores = _stable_sigmoid(logits)
    choice = scores + correction_bias
    # Full sort by choice descending (argsort of -choice is ascending). Ties on
    # real-valued logits do not occur; the kernel's total order breaks any tie by
    # lower expert index, and set-identity is what this reference is checked to.
    order = mx.argsort(-choice, axis=-1)[..., :top_k]
    indices = order.astype(mx.uint32)
    weights = mx.take_along_axis(scores, order, axis=-1)
    if normalize:
        weights = weights / weights.sum(axis=-1, keepdims=True)
    weights = weights * scale
    return indices, weights


def _stock_router_topk(
    logits: mx.array,
    correction_bias: mx.array,
    top_k: int,
    *,
    normalize: bool,
    scale: float,
) -> tuple[mx.array, mx.array]:
    """The exact stock op chain -- also the guarded fallback."""

    scores = mx.sigmoid(logits)
    choice = scores + correction_bias
    indices = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / weights.sum(axis=-1, keepdims=True)
    return indices, weights * scale


def fused_router_topk_bitonic(
    logits: mx.array,
    correction_bias: mx.array,
    top_k: int,
    *,
    normalize: bool,
    scale: float,
) -> tuple[mx.array, mx.array]:
    """Bitonic-selected ``(indices[rows, top_k] uint32, weights[rows, top_k]
    float32)`` for one MoE routing decision per row.

    Drop-in for ``laguna_decode.fused_router_topk``. Falls back to the stock
    chain on any unsupported shape/dtype/device.
    """

    if not is_router_bitonic_eligible(logits, correction_bias, top_k):
        return _stock_router_topk(
            logits, correction_bias, top_k, normalize=normalize, scale=scale
        )

    rows, experts = int(logits.shape[0]), int(logits.shape[1])
    kernel = _router_bitonic_kernel(experts, top_k)
    indices, weights = kernel(
        inputs=[logits, correction_bias, float(scale), bool(normalize)],
        grid=(experts * rows, 1, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    # Fake-speedup / miswire guard: exactly one (index, weight) pair per
    # (row, selected-slot).
    assert tuple(indices.shape) == (rows, top_k), (
        f"router bitonic produced indices {tuple(indices.shape)}, "
        f"expected {(rows, top_k)}"
    )
    assert tuple(weights.shape) == (rows, top_k), (
        f"router bitonic produced weights {tuple(weights.shape)}, "
        f"expected {(rows, top_k)}"
    )
    return indices, weights
