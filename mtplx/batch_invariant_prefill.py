"""Batch-invariant prefill matmuls (opt-in, MTPLX_BATCH_INVARIANT_PREFILL=1).

Problem: MLX picks its quantized-matmul kernel on the row count M of each
forward, and those kernels round differently. On MLX 0.32 (quantized.cpp):

- ``quantized_matmul`` with 2-D weights runs ``qmm_splitk`` whenever the
  32x32 output tiles number fewer than ~512; the split count follows
  ``512 / (m_tiles * n_tiles)``, so for narrow outputs (MoE router N=256,
  shared expert N=512, GDN a/b N=32, k/v N=512) a row's result depends on how
  many rows share the forward, up to M=1024 and beyond. Batched calls skip
  split-K and run the NAX ``qmm`` whose rows are independent once each batch
  holds more than 32 rows (``bm`` 64).
- ``gather_qmm`` (routed experts) runs the tiled ``gather_qmm_rhs`` only with
  at least 4 token-expert rows per expert; smaller forwards fall back to the
  per-row ``gather_qmv``, which rounds differently.
- ``scaled_dot_product_attention`` with head dim 256 runs the fused causal
  kernel from 1024 query rows, an unfused matmul route from 9 to 1023 and the
  vector kernel below 9.

On Qwen3.6-35B-A3B the 8th and 9th expert often score within one bf16 step,
so those rounding differences change the routing and move scores ~0.2 nats
per token between block layouts (256 against 2048 rows, token by token, a
64-row tail forward).

Fix, prefill phase only (decode and verify keep the stock kernels): every
affine ``QuantizedLinear`` runs as a two-batch matmul of at least 33 rows
per batch (zero padded), every ``SwitchGLU`` pads its tokens so the sorted
tiled kernel always runs, and Qwen3-Next full attention always runs the fused
causal kernel (zero query rows in front below 9 rows). All three reuse stock
kernels; padding costs work only on forwards narrower than the minimums. The
GDN layers needed nothing. The switch is read once at construction, like the
other MoE knobs.

The lone final prompt token runs on the stock kernels (``stock_prefill_kernels``):
padding one row to 128 expert tokens costs more than the prompt's own prefill
on short prompts, and no caller compares that row against a wider forward.

Only MoE models get the lane by default. On a dense model (no ``SwitchGLU``)
there is no routing to stabilize and the padding only costs: on Qwen3.5-9B
~4% on prompt scoring and ~15% on greedy time to first token, against a ~21%
scoring gain on Qwen3.6-35B-A3B. MTPLX_BATCH_INVARIANT_PREFILL_DENSE=1 admits
dense models anyway, for callers that need invariant rows there.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU, _gather_sort

from .attention_context import current_attention_phase

BATCH_INVARIANT_PREFILL_ENV = "MTPLX_BATCH_INVARIANT_PREFILL"
BATCH_INVARIANT_PREFILL_DENSE_ENV = "MTPLX_BATCH_INVARIANT_PREFILL_DENSE"

# Two batches of more than 32 rows each: MLX's batched NAX qmm then uses the
# same 64-row tiles and no split-K at every row count.
_LINEAR_BATCHES = 2
_MIN_LINEAR_ROWS = 2 * 33
# gather_qmm_rhs needs at least this many token-expert rows per expert.
_MIN_ROWS_PER_EXPERT = 4
# Fused SDPA: more than 8 query rows selects the full (not the vector) kernel.
_MIN_FUSED_QUERY_ROWS = 9
_FUSED_HEAD_DIMS = frozenset({64, 72, 80, 96, 128, 192, 256})
_STOCK_SDPA: dict[str, Any] = {"sdpa": None}
# SwitchGLU padding keeps expert rows invariant from 16 experts up (16/32/256
# measured); with 8 experts, top-2, MLX picks another expert kernel below 33
# tokens even after padding (M5 Pro, MLX 0.32.2, synthetic tensors).
_MIN_INVARIANT_EXPERTS = 16
_STATE: dict[str, Any] = {"installed": False, "report": {}, "refusal": None}
_STOCK_KERNELS: ContextVar[bool] = ContextVar(
    "mtplx_batch_invariant_stock_kernels", default=False
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def batch_invariant_prefill_enabled() -> bool:
    """Whether construction installs the batch-invariant prefill lane."""

    return _env_flag(BATCH_INVARIANT_PREFILL_ENV)


def batch_invariant_prefill_dense_enabled() -> bool:
    """Whether the lane may also install on a dense model (no ``SwitchGLU``)."""

    return _env_flag(BATCH_INVARIANT_PREFILL_DENSE_ENV)


def batch_invariant_prefill_installed() -> bool:
    """Whether this process runs its prefill through the invariant lane, so
    callers may choose forward widths freely without changing results."""

    return _STATE["installed"]


def batch_invariant_prefill_status() -> dict[str, Any]:
    """The lane's state for /health: the install report, or why it is off."""

    if _STATE["installed"]:
        return {"installed": True, **_STATE["report"]}
    if _STATE["refusal"] is not None:
        return {"installed": False, "reason": _STATE["refusal"]}
    reason = "not_reached" if batch_invariant_prefill_enabled() else "disabled"
    return {"installed": False, "reason": reason}


def _uncovered_module_reason(module: Any, under_switch_glu: bool) -> str | None:
    if type(module) is SwitchGLU:
        experts = int(module.gate_proj["weight"].shape[0])
        if experts < _MIN_INVARIANT_EXPERTS:
            return f"switch_glu_experts:{experts}"
        return None
    if "scales" not in module:
        return None  # not a quantized projection
    if type(module) is nn.QuantizedLinear:
        if module.mode == "affine":
            return None
        return f"unsupported_linear:QuantizedLinear[{module.mode}]"
    if hasattr(module, "as_linear") or (
        under_switch_glu and type(module) is QuantizedSwitchLinear
    ):
        return None  # embedding lookup, or an expert projection of a SwitchGLU
    return f"unsupported_linear:{type(module).__name__}"


def batch_invariant_prefill_refusal(model: Any) -> str | None:
    """Why the lane cannot make ``model``'s prefill rows invariant, or None.

    The lane only swaps affine ``nn.QuantizedLinear`` and stock ``SwitchGLU``
    classes; any other quantized projection (e.g. Prism's
    ``HadamardQuantizedLinear``) keeps its row-count-dependent kernels, and
    a SwitchGLU with few experts stays row dependent despite the padding.
    Half a lane would let the prefill-sized scoring trunk and the dropped
    tail grid change results, so the caller installs nothing when this
    returns a reason.

    A dense model (no ``SwitchGLU``) is refused with ``dense_model`` unless
    MTPLX_BATCH_INVARIANT_PREFILL_DENSE is set: without routing the lane
    only costs prefill time. Coverage reasons come first, so a model the lane
    cannot cover keeps that reason even with the override."""

    modules = model.named_modules()
    switch_glus = {name for name, module in modules if type(module) is SwitchGLU}
    for name, module in modules:
        under_switch_glu = name.rpartition(".")[0] in switch_glus
        reason = _uncovered_module_reason(module, under_switch_glu)
        if reason is not None:
            return reason
    if not switch_glus and not batch_invariant_prefill_dense_enabled():
        return "dense_model"
    return None


def refuse_batch_invariant_prefill(reason: str) -> None:
    """Record why this process runs its prefill on the stock kernels."""

    _STATE["refusal"] = reason


@contextmanager
def stock_prefill_kernels() -> Iterator[None]:
    """Run the enclosed prefill forward on the stock kernels, for forwards
    whose rows no wider forward is ever compared against (the lone final
    prompt token)."""

    token = _STOCK_KERNELS.set(True)
    try:
        yield
    finally:
        _STOCK_KERNELS.reset(token)


def _in_prefill() -> bool:
    return current_attention_phase() == "prefill" and not _STOCK_KERNELS.get()


def _pad_rows(rows: mx.array, target: int) -> mx.array:
    missing = target - int(rows.shape[0])
    if missing <= 0:
        return rows
    padding = mx.zeros((missing,) + tuple(rows.shape[1:]), dtype=rows.dtype)
    return mx.concatenate([rows, padding], axis=0)


def _broadcast_batches(array: mx.array | None) -> mx.array | None:
    if array is None:
        return None
    return mx.broadcast_to(array, (_LINEAR_BATCHES,) + tuple(array.shape))


def split_k_free_quantized_matmul(
    x: mx.array,
    weight: mx.array,
    scales: mx.array,
    biases: mx.array | None,
    *,
    group_size: int,
    bits: int,
    mode: str = "affine",
) -> mx.array:
    """``x @ dequantize(weight).T`` with a result per row that does not depend
    on how many rows ``x`` holds."""

    in_features = int(x.shape[-1])
    rows = x.reshape(-1, in_features)
    row_count = int(rows.shape[0])
    padded = max(_MIN_LINEAR_ROWS, row_count + row_count % _LINEAR_BATCHES)
    batched = _pad_rows(rows, padded).reshape(
        _LINEAR_BATCHES, padded // _LINEAR_BATCHES, in_features
    )
    out = mx.quantized_matmul(
        batched,
        _broadcast_batches(weight),
        _broadcast_batches(scales),
        _broadcast_batches(biases),
        transpose=True,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    out = out.reshape(padded, -1)[:row_count]
    return out.reshape(*x.shape[:-1], int(out.shape[-1]))


class BatchInvariantQuantizedLinear(nn.QuantizedLinear):
    """``QuantizedLinear`` whose prefill rows are independent of the row count."""

    def __call__(self, x: mx.array) -> mx.array:
        if not _in_prefill():
            return super().__call__(x)
        out = split_k_free_quantized_matmul(
            x,
            self["weight"],
            self["scales"],
            self.get("biases"),
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
        )
        if "bias" in self:
            out = out + self["bias"]
        return out


class BatchInvariantSwitchGLU(SwitchGLU):
    """``SwitchGLU`` that always runs the sorted tiled expert kernel in prefill."""

    def _min_sorted_tokens(self, top_k: int) -> int:
        experts = int(self.gate_proj["weight"].shape[0])
        return -(-_MIN_ROWS_PER_EXPERT * experts // int(top_k))

    def sorted_route_applies(self, x: mx.array, top_k: int) -> bool:
        """Whether this prefill forward runs the sorted kernel without padding,
        so ``sorted_experts`` may hand out its expert-sorted output."""

        tokens = 1
        for dim in x.shape[:-1]:
            tokens *= int(dim)
        return _in_prefill() and tokens >= self._min_sorted_tokens(top_k)

    def sorted_experts(self, x: mx.array, indices: mx.array) -> tuple[mx.array, mx.array]:
        """The stock sorted chain without its final unsort, for a forward that
        ``sorted_route_applies`` to: ``([tokens * top_k, dims]`` expert outputs
        in expert-sorted order, the inverse permutation to token order)."""

        x = mx.expand_dims(x, (-2, -3))
        x, idx, inv_order = _gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=True)
        x_gate = self.gate_proj(x, idx, sorted_indices=True)
        y = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=True)
        return y.reshape(-1, int(y.shape[-1])), inv_order

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        if not _in_prefill():
            return super().__call__(x, indices)
        experts = int(self.gate_proj["weight"].shape[0])
        top_k = int(indices.shape[-1])
        min_tokens = self._min_sorted_tokens(top_k)
        tokens = x.reshape(-1, int(x.shape[-1]))
        token_count = int(tokens.shape[0])
        if token_count >= min_tokens:
            return super().__call__(x, indices)
        # Padding tokens use the last top_k experts: valid ids, and their
        # rows only extend the tail of the sorted expert list.
        pad_ids = mx.broadcast_to(
            mx.arange(experts - top_k, experts, dtype=indices.dtype),
            (min_tokens - token_count, top_k),
        )
        routes = mx.concatenate([indices.reshape(-1, top_k), pad_ids], axis=0)
        out = super().__call__(_pad_rows(tokens, min_tokens), routes)
        return out[:token_count].reshape(*indices.shape, int(out.shape[-1]))


def _fused_attention_rows(
    queries: mx.array, keys: mx.array, mask: Any, cache: Any
) -> int | None:
    """Padded query rows for the fused causal kernel, or None when this call
    must keep the stock route (array mask, quantized KV, too few keys)."""

    if hasattr(cache, "bits") or not (mask is None or mask == "causal"):
        return None
    query_rows = int(queries.shape[2])
    padded = max(query_rows, _MIN_FUSED_QUERY_ROWS)
    if padded > int(keys.shape[2]) or int(queries.shape[-1]) not in _FUSED_HEAD_DIMS:
        return None
    return padded


def batch_invariant_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    cache: Any,
    scale: float,
    mask: Any,
    sinks: mx.array | None = None,
) -> mx.array:
    """mlx-lm's ``scaled_dot_product_attention`` with, in prefill, one fused
    causal kernel for every query count.

    Stock MLX runs the fused kernel only from 1024 query rows (head dim 256)
    and an unfused matmul-softmax-matmul below that, so a row's attention
    depends on the forward's width. Fewer than 9 query rows would pick the
    vector kernel; zero query rows in front of them keep the causal diagonal
    of the real rows and are dropped after the call."""

    padded = _fused_attention_rows(queries, keys, mask, cache) if _in_prefill() else None
    if padded is None:
        return _STOCK_SDPA["sdpa"](
            queries, keys, values, cache=cache, scale=scale, mask=mask, sinks=sinks
        )
    lead = padded - int(queries.shape[2])
    if lead:
        shape = list(queries.shape)
        shape[2] = lead
        queries = mx.concatenate([mx.zeros(shape, dtype=queries.dtype), queries], axis=2)
    out = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale=scale,
        mask="causal",
        sinks=sinks,
        force_fused=True,
    )
    return out[:, :, lead:]


def _install_attention_route() -> bool:
    """Hook mlx-lm's SDPA where Qwen3-Next attention finds it: the module
    global of ``qwen3_next`` (stock forward) and ``base`` (MTPLX's attention
    routes, e.g. ``attention_split``, import it from there per call)."""

    from mlx_lm.models import base, qwen3_next

    if qwen3_next.scaled_dot_product_attention is batch_invariant_sdpa:
        return False
    _STOCK_SDPA["sdpa"] = base.scaled_dot_product_attention
    qwen3_next.scaled_dot_product_attention = batch_invariant_sdpa
    base.scaled_dot_product_attention = batch_invariant_sdpa
    return True


def _swap_class(module: Any, new_class: type) -> None:
    module.__class__ = new_class


def install_batch_invariant_prefill(model: Any) -> dict[str, int]:
    """Route every affine ``QuantizedLinear`` and stock ``SwitchGLU`` of
    ``model``, and the Qwen3-Next attention, through the batch-invariant
    prefill lane. Class swaps and one function hook: the parameter tree and
    the decode path stay exactly as loaded."""

    report = {
        "linears": 0,
        "switch_glus": 0,
        "skipped_linears": 0,
        "attention_hooked": int(_install_attention_route()),
    }
    for _name, module in model.named_modules():
        if type(module) is nn.QuantizedLinear:
            if module.mode == "affine":
                _swap_class(module, BatchInvariantQuantizedLinear)
                report["linears"] += 1
            else:
                report["skipped_linears"] += 1
        elif type(module) is SwitchGLU:
            _swap_class(module, BatchInvariantSwitchGLU)
            report["switch_glus"] += 1
    _STATE["installed"] = True
    _STATE["report"] = report
    return report
