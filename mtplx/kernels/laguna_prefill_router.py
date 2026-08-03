"""Fused MoE router (sigmoid + bias + top-k) for the Laguna S-2.1 PREFILL path.

Prefill twin of the decode ``fused_router_topk`` in ``laguna_decode.py``.  It is
the SAME per-row selection epilogue -- one threadgroup per token, one thread per
expert, sigmoid -> add correction bias -> top-k -> gather the unbiased scores ->
normalize -> scale -- run over ``M = T`` tokens instead of the 1..4 decode rows.
The only change from decode is the row gate: decode caps rows at 4 because the
stock op chain barely grows with rows while the kernel's serial reduction rounds
do, so the fused kernel loses at batch; prefill runs many rows deliberately, so
this variant drops the cap and lets the caller decide.

Ported from the challenge's own fused router-selection epilogue
(``lagunaResidualRMSNormRouterSource`` in
``Sources/MLXFastModel/LagunaRuntimeModel.swift``), re-expressed for S-2.1's
256 experts / top-10 / norm_topk_prob routing.

S-2.1 routing (``LagunaSparseMoeBlock``): logits come from the BF16 router gate
widened to float32; ``moe_router_logit_softcapping`` is 0.0 so there is no
softcap; selection is ``argpartition(-(sigmoid+bias))[:10]``; the weights are
the UNBIASED sigmoid scores at the selected experts, normalized (norm_topk_prob
is True).  The routed scaling 2.5 is applied downstream in the combine tail (P5,
``laguna_prefill_moe_combine``), so this kernel's ``scale`` defaults to 1.0.

Selection cannot be bit-identical to ``argpartition`` by construction: argpart
leaves the selected order unspecified and the normalizing sum accumulates in an
order this kernel cannot reproduce.  Ties break toward the lower expert index
here.  The check measures selection PARITY (the set of chosen experts per token)
and the normalized weights, not a byte match.
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


def _router_selection_shape_ok(experts: int, top_k: int) -> bool:
    """The expert/top-k bounds the selection epilogue is compiled for.

    One thread per expert, selection scratch sized at compile time, and the
    ``stride = experts/2; stride >>= 1`` tree reduction, which only folds every
    element into lane 0 when ``experts`` is a power of two.  S-2.1 is 256; the
    power-of-two guard keeps a non-power-of-two config (which the tree would
    silently mis-reduce) on the stock path.
    """

    return (
        0 < top_k <= 32
        and 32 <= experts <= 1024
        and (experts % 32) == 0
        and (experts & (experts - 1)) == 0
    )


def is_router_prefill_eligible(logits: mx.array, bias: mx.array, top_k: int) -> bool:
    """Whether the prefill router covers this shape.

    Unlike decode there is NO row cap: prefill drives M = T rows on purpose.
    """

    if not _on_metal_device():
        return False
    if logits.ndim != 2 or bias.ndim != 1:
        return False
    if logits.dtype != mx.float32 or bias.dtype != mx.float32:
        return False
    experts = int(logits.shape[1])
    if experts != int(bias.shape[0]):
        return False
    if int(logits.shape[0]) <= 0:
        return False
    return _router_selection_shape_ok(experts, top_k)


_ROUTER_SELECT_DECLS = """
        threadgroup float tg_score[NUM_EXPERTS];
        threadgroup float tg_choice[NUM_EXPERTS];
        threadgroup float red_val[NUM_EXPERTS];
        threadgroup uint  red_idx[NUM_EXPERTS];
        threadgroup uint  sel_idx[TOP_K];
        threadgroup float sel_score[TOP_K];
"""

# Entered with `score` (the sigmoid of this thread's logit) already in hand.
# Identical to the decode selection epilogue so the two can never disagree about
# tie-break or accumulation order.
_ROUTER_SELECT_EPILOGUE = """
        tg_score[lid] = score;
        tg_choice[lid] = score + correction_bias[lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TOP_K; ++k) {
            red_val[lid] = tg_choice[lid];
            red_idx[lid] = lid;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint stride = NUM_EXPERTS / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    float mine = red_val[lid];
                    float theirs = red_val[lid + stride];
                    uint mine_idx = red_idx[lid];
                    uint their_idx = red_idx[lid + stride];
                    // Ties resolve toward the lower expert index.
                    if (theirs > mine || (theirs == mine && their_idx < mine_idx)) {
                        red_val[lid] = theirs;
                        red_idx[lid] = their_idx;
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (lid == 0) {
                sel_idx[k] = red_idx[0];
                sel_score[k] = tg_score[red_idx[0]];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (lid == sel_idx[k]) {
                tg_choice[lid] = -metal::numeric_limits<float>::infinity();
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid == 0) {
            float total = 0.0f;
            for (uint k = 0; k < TOP_K; ++k) {
                total += sel_score[k];
            }
            float invsum = (total == 0.0f) ? 0.0f : (1.0f / total);
            for (uint k = 0; k < TOP_K; ++k) {
                indices[row * TOP_K + k] = sel_idx[k];
                weights[row * TOP_K + k] =
                    normalize ? (sel_score[k] * invsum * scale)
                              : (sel_score[k] * scale);
            }
        }
"""


@lru_cache(maxsize=None)
def _router_prefill_kernel(experts: int, top_k: int):
    header = f"""
        using namespace metal;
        constant constexpr int NUM_EXPERTS = {experts};
        constant constexpr int TOP_K = {top_k};
    """
    source = (
        """
        uint row = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
"""
        + _ROUTER_SELECT_DECLS
        + """
        float logit = logits[row * NUM_EXPERTS + lid];
        float score = 1.0f / (1.0f + metal::exp(-logit));
"""
        + _ROUTER_SELECT_EPILOGUE
    )
    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_prefill_router_e{experts}_k{top_k}",
        input_names=["logits", "correction_bias", "scale", "normalize"],
        output_names=["indices", "weights"],
        header=header,
        source=source,
    )


def fused_router_prefill(
    logits: mx.array,
    correction_bias: mx.array,
    top_k: int,
    *,
    normalize: bool = True,
    scale: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Return ``(indices, weights)`` for the routing decision over M tokens.

    ``logits`` is ``[M, experts]`` float32 (the widened router gate output),
    ``correction_bias`` is ``[experts]`` float32.  Returns ``indices``
    ``[M, top_k]`` uint32 and ``weights`` ``[M, top_k]`` float32.  With the
    defaults the weights are the normalized unbiased sigmoid scores (the routed
    scaling 2.5 is applied by the P5 combine tail).

    Falls back to the stock op chain on any shape the kernel does not cover.
    """

    if not is_router_prefill_eligible(logits, correction_bias, top_k):
        scores = mx.sigmoid(logits)
        choice = scores + correction_bias
        indices = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if normalize:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return indices.astype(mx.uint32), weights * scale

    rows, experts = int(logits.shape[0]), int(logits.shape[1])
    kernel = _router_prefill_kernel(experts, top_k)
    indices, weights = kernel(
        inputs=[logits, correction_bias, float(scale), bool(normalize)],
        grid=(experts * rows, 1, 1),
        threadgroup=(experts, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    # Fake-speedup guard: exactly one (index,weight) row of width top_k per token.
    assert tuple(indices.shape) == (rows, top_k), (
        f"router indices {tuple(indices.shape)} != {(rows, top_k)}"
    )
    assert tuple(weights.shape) == (rows, top_k), (
        f"router weights {tuple(weights.shape)} != {(rows, top_k)}"
    )
    return indices, weights


def router_prefill_reference(
    logits: mx.array,
    correction_bias: mx.array,
    top_k: int,
    *,
    normalize: bool = True,
    scale: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Pure-mx reference: the stock ``LagunaSparseMoeBlock`` routing selection.

    Selection uses ``argpartition(-(sigmoid+bias))``; weights are the unbiased
    sigmoid scores at the selected experts, normalized then scaled.  Returned
    indices are sorted ascending so a set-parity comparison against the kernel is
    order-independent (argpartition's order is unspecified).
    """

    scores = mx.sigmoid(logits)
    choice = scores + correction_bias
    indices = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if normalize:
        weights = weights / weights.sum(axis=-1, keepdims=True)
    weights = weights * scale
    order = mx.argsort(indices, axis=-1)
    return (
        mx.take_along_axis(indices, order, axis=-1).astype(mx.uint32),
        mx.take_along_axis(weights, order, axis=-1),
    )
