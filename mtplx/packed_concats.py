"""Packed projection concats at small S (Speed War 2 row 27).

Mechanism (mlxfast arena crown overlay, 2026-08-19 re-scrape §3.4.5): several
projections of one layer read the SAME input activation; concatenating their
weight rows at load time turns N quantized-matmul launches into one, then the
output is split. Row-concat of per-output-row quantized triples is
element-identical to separate launches (each output row's groups, scales and
accumulation order are unchanged). Arena receipts: FA QKV concat +1.94%
promoted; MLP gate+up (N=34816) gated S<=16 in the 3.249 crown; the DFlash2
port measured the family at -9.8% leg time.

Sites (this module): Qwen3Next Attention q|k|v (N=14336) and MLP gate|up
(N=34816). Fused path fires only when S <= MTPLX_PACKED_PROJ_MAX_S
(default 16) — at prefill widths the unpacked kernels win (arena S-gates).

Off by default: MTPLX_PACKED_PROJ_CONCATS=1 installs. Class-level wrappers
with instance-attr payloads (unfused instances take the original path) and
engagement counters (mistakes/: verify a monkeypatch engaged with a counter
before reading any A/B).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

COUNTERS: dict[str, int] = {
    "attention_fused_modules": 0,
    "mlp_fused_modules": 0,
    "skipped_modules": 0,
    "fused_attention_calls": 0,
    "fused_mlp_calls": 0,
}


def enabled() -> bool:
    return str(os.environ.get("MTPLX_PACKED_PROJ_CONCATS", "") or "").strip() in {
        "1",
        "true",
        "on",
        "yes",
    }


def _max_s() -> int:
    raw = os.environ.get("MTPLX_PACKED_PROJ_MAX_S", "16")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 16


def _linear_kind(module: Any) -> str | None:
    if isinstance(module, nn.QuantizedLinear):
        if str(getattr(module, "mode", "affine")) != "affine":
            return None
        return "quantized"
    if isinstance(module, nn.Linear):
        return "linear"
    return None


def _pack(members: list[Any]) -> dict[str, Any] | None:
    """Row-concat compatible projections into one fused payload (or None)."""

    kinds = {_linear_kind(m) for m in members}
    if len(kinds) != 1 or None in kinds:
        return None
    kind = kinds.pop()
    if any("bias" in m for m in members) != all("bias" in m for m in members):
        return None
    has_bias = all("bias" in m for m in members)
    splits: list[int] = []
    total = 0
    if kind == "quantized":
        group_sizes = {int(m.group_size) for m in members}
        bits = {int(m.bits) for m in members}
        if len(group_sizes) != 1 or len(bits) != 1:
            return None
        in_cols = {m.weight.shape[-1] for m in members}
        if len(in_cols) != 1:
            return None
        for m in members[:-1]:
            total += int(m.scales.shape[0])
            splits.append(total)
        payload: dict[str, Any] = {
            "kind": kind,
            "weight": mx.concatenate([m.weight for m in members], axis=0),
            "scales": mx.concatenate([m.scales for m in members], axis=0),
            "biases": mx.concatenate([m.biases for m in members], axis=0),
            "group_size": group_sizes.pop(),
            "bits": bits.pop(),
            "splits": splits,
        }
    else:
        in_cols = {m.weight.shape[-1] for m in members}
        if len(in_cols) != 1:
            return None
        for m in members[:-1]:
            total += int(m.weight.shape[0])
            splits.append(total)
        payload = {
            "kind": kind,
            "weight": mx.concatenate([m.weight for m in members], axis=0),
            "splits": splits,
        }
    if has_bias:
        payload["bias"] = mx.concatenate([m.bias for m in members], axis=0)
    mx.eval([v for v in payload.values() if isinstance(v, mx.array)])
    return payload


def _fused_forward(payload: dict[str, Any], x: mx.array) -> list[mx.array]:
    if payload["kind"] == "quantized":
        out = mx.quantized_matmul(
            x,
            payload["weight"],
            scales=payload["scales"],
            biases=payload["biases"],
            transpose=True,
            group_size=payload["group_size"],
            bits=payload["bits"],
        )
    else:
        out = x @ payload["weight"].T
    if "bias" in payload:
        out = out + payload["bias"]
    # Contiguous copies: stock projections emit contiguous tensors, and a
    # strided split view can route a downstream kernel (norm/rope/sdpa) onto a
    # different reduction variant — measured as a rare 1-ulp logit flip that
    # broke sampled-trajectory identity (seed-825 token 233, 2026-08-19).
    return [mx.contiguous(t) for t in mx.split(out, payload["splits"], axis=-1)]


def install_qwen3_next_packed_concats(model: Any) -> dict[str, int] | None:
    """Fuse attention q|k|v and MLP gate|up on every stock decoder layer."""

    if not enabled():
        return None
    from mlx_lm.models import qwen3_next as qn

    max_s = _max_s()

    attention_class = qn.Qwen3NextAttention
    mlp_class = qn.Qwen3NextMLP

    if not getattr(qn, "_mtplx_packed_concats_installed", False):
        attention_call = attention_class.__call__
        mlp_call = mlp_class.__call__

        def attention_call_packed(self, x, mask=None, cache=None):
            payload = getattr(self, "_mtplx_fused_qkv", None)
            if payload is None or x.shape[1] > max_s:
                return attention_call(self, x, mask=mask, cache=cache)
            COUNTERS["fused_attention_calls"] += 1
            # Tail replicated verbatim from the stock forward (qwen3_next
            # Qwen3NextAttention.__call__) with the three projections replaced
            # by one fused launch. Re-audit on any mlx-lm pin bump.
            B, L, _ = x.shape
            q_proj_output, keys, values = _fused_forward(payload, x)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, self.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(B, L, -1)
            queries = self.q_norm(queries).transpose(0, 2, 1, 3)
            keys = self.k_norm(
                keys.reshape(B, L, self.num_key_value_heads, -1)
            ).transpose(0, 2, 1, 3)
            values = values.reshape(B, L, self.num_key_value_heads, -1).transpose(
                0, 2, 1, 3
            )
            if cache is not None:
                queries = self.rope(queries, offset=cache.offset)
                keys = self.rope(keys, offset=cache.offset)
                keys, values = cache.update_and_fetch(keys, values)
            else:
                queries = self.rope(queries)
                keys = self.rope(keys)
            output = qn.scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            return self.o_proj(output * mx.sigmoid(gate))

        def mlp_call_packed(self, x):
            payload = getattr(self, "_mtplx_fused_gate_up", None)
            if payload is None or (x.ndim > 1 and x.shape[1] > max_s):
                return mlp_call(self, x)
            COUNTERS["fused_mlp_calls"] += 1
            gate, up = _fused_forward(payload, x)
            return self.down_proj(qn.swiglu(gate, up))

        attention_class.__call__ = attention_call_packed
        mlp_class.__call__ = mlp_call_packed
        qn._mtplx_packed_concats_installed = True

    # Attach fused payloads per instance.
    text_model = getattr(model, "language_model", model)
    inner = getattr(text_model, "model", text_model)
    layers = getattr(inner, "layers", None) or []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if isinstance(attn, attention_class) and not hasattr(
            attn, "_mtplx_fused_qkv"
        ):
            payload = _pack([attn.q_proj, attn.k_proj, attn.v_proj])
            if payload is not None:
                attn._mtplx_fused_qkv = payload
                COUNTERS["attention_fused_modules"] += 1
            else:
                COUNTERS["skipped_modules"] += 1
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, mlp_class) and not hasattr(mlp, "_mtplx_fused_gate_up"):
            payload = _pack([mlp.gate_proj, mlp.up_proj])
            if payload is not None:
                mlp._mtplx_fused_gate_up = payload
                COUNTERS["mlp_fused_modules"] += 1
            else:
                COUNTERS["skipped_modules"] += 1

    logger.info("[packed-concats] %s", COUNTERS)
    import atexit
    import sys

    if not getattr(install_qwen3_next_packed_concats, "_receipt_registered", False):

        def _dump() -> None:
            sys.stderr.write(f"[packed-concats] exit receipt: {COUNTERS}\n")

        atexit.register(_dump)
        install_qwen3_next_packed_concats._receipt_registered = True
    return dict(COUNTERS)
