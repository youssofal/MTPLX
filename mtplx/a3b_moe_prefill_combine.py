"""One-kernel MoE combine in the A3B prefill (opt-in, on the invariant lane).

After the routed experts, mlx-lm's ``Qwen3NextSparseMoeBlock`` (the MoE block
of Qwen3.5/3.6, A3B) materializes three ``[rows, top_k, hidden]`` BF16
tensors: the unsorted expert outputs, their product with the routing scores
and, before the column sum, nothing else reads them. At 2,048 rows, top-8 of
2,048 that is ~0.2 GB of traffic per layer for an 8 MB result.

On the batch-invariant prefill lane every prefill forward from 128 tokens
already runs the experts expert-sorted (``BatchInvariantSwitchGLU``). This
route asks the lane for the sorted expert output and its inverse permutation
(``sorted_experts``, the stock ``SwitchGLU`` chain without the unsort) and
hands both, with the scores and the gated shared expert, to Flash-Next's
combine kernel (``kernels/qwen4_moe_prefill_combine.py``), which is generic in
``top_k`` and ``hidden`` and rounds exactly like the stock tail. The routing,
the experts and the shared expert are the stock calls; only the tail changes.

Forwards narrower than the lane's sorted minimum (128 tokens on A3B), decode,
verify and the lone final prompt token keep the stock block. The load-time
self-check (``kernel_selfcheck``, lane ``a3b_moe_prefill_combine``) compares
the kernel with the stock tail at the A3B geometry on this GPU and turns the
route off for the process on any difference or failure.

``MTPLX_A3B_MOE_PREFILL_COMBINE=1`` turns it on (default off); installed only
together with the invariant lane (``MTPLX_BATCH_INVARIANT_PREFILL=1``).
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

from .batch_invariant_prefill import BatchInvariantSwitchGLU
from .kernel_selfcheck import lane_disabled
from .kernels.qwen4_moe_prefill_combine import moe_prefill_combine

ENV = "MTPLX_A3B_MOE_PREFILL_COMBINE"
LANE = "a3b_moe_prefill_combine"
COUNTERS: dict[str, int] = {"combined_forwards": 0}


def switched_on() -> bool:
    """The switch alone (default off)."""

    return os.environ.get(ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    """Switched on, and not turned off by the load-time self-check."""

    return switched_on() and not lane_disabled(LANE)


def _routing(block: Qwen3NextSparseMoeBlock, x: mx.array) -> tuple[mx.array, mx.array]:
    """``(expert ids, scores)`` exactly as the stock block computes them."""

    gates = mx.softmax(block.gate(x), axis=-1, precise=True)
    k = block.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if block.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores


def _shared(block: Qwen3NextSparseMoeBlock, x: mx.array) -> mx.array:
    return mx.sigmoid(block.shared_expert_gate(x)) * block.shared_expert(x)


class CombinedPrefillSparseMoeBlock(Qwen3NextSparseMoeBlock):
    """The stock MoE block whose prefill tail is one combine kernel."""

    def __call__(self, x: mx.array) -> mx.array:
        switch_mlp = self.switch_mlp
        if (
            self.sharding_group is not None
            or not enabled()
            or not isinstance(switch_mlp, BatchInvariantSwitchGLU)
            or not switch_mlp.sorted_route_applies(x, self.top_k)
        ):
            return super().__call__(x)
        inds, scores = _routing(self, x)
        y_sorted, inv_order = switch_mlp.sorted_experts(x, inds)
        hidden = int(x.shape[-1])
        rows = int(y_sorted.shape[0]) // self.top_k
        COUNTERS["combined_forwards"] += 1
        out = moe_prefill_combine(
            y_sorted,
            inv_order,
            scores.reshape(rows, self.top_k),
            _shared(self, x).reshape(rows, hidden),
        )
        return out.reshape(x.shape)


def install_a3b_moe_prefill_combine(model: Any) -> int:
    """Swap every stock ``Qwen3NextSparseMoeBlock`` whose experts run on the
    invariant lane for the combining subclass. Class swaps only: the
    parameter tree stays as loaded. Returns the number of blocks swapped."""

    swapped = 0
    for _name, module in model.named_modules():
        if type(module) is Qwen3NextSparseMoeBlock and isinstance(
            module.switch_mlp, BatchInvariantSwitchGLU
        ):
            module.__class__ = CombinedPrefillSparseMoeBlock
            swapped += 1
    return swapped
