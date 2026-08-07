"""Load EschaLabs ``Qwen3.6-35B-A3B-Escha-W2`` into the mtplx runtime.

Escha-W2 is a Qwen3.6-A3B (``qwen3_5_moe`` hybrid GDN + full-attn) checkpoint whose MoE experts
are the ``eschamoe`` 2/3-bit trellis format and whose non-expert weights are per-out-channel
symmetric int8. This module builds the standard mlx-lm ``qwen3_5_moe`` trunk and swaps in:

  * MoE experts  -> :class:`EschaSwitchGLU` (2-bit codes resident, decoded on the fly, no cache)
  * non-expert Linears + lm_head -> :class:`mtplx.int8_linear.Int8Linear` (int8 resident, fused matvec)

The result is an ordinary ``qwen3_5_moe`` model object, so mtplx's existing qwen3-next runtime
(``forward_ar`` / ragged+recurrent caches / ``batched_decode`` MTP spec-decode) drives it
unchanged. ``_load_base_model`` dispatches here when :func:`is_escha_checkpoint` matches.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .eschamoe import escha_qmv, fused_moe_matmul, t128, COMPUTE_DTYPE
from .int8_linear import Int8Linear


def is_escha_checkpoint(path: Path, config: dict[str, Any]) -> bool:
    """True for a qwen3_5_moe checkpoint whose experts are eschamoe-packed (``escha_code``)."""
    tcfg = config.get("text_config", config)
    if str(config.get("model_type") or tcfg.get("model_type") or "").lower() != "qwen3_5_moe":
        return False
    idx = Path(path) / "model.safetensors.index.json"
    if idx.exists():
        try:
            wm = json.loads(idx.read_text(encoding="utf-8")).get("weight_map", {})
            return any(".escha_code" in k for k in wm)
        except Exception:
            pass
    # single-file / no index: peek a shard's keys
    from safetensors import safe_open
    for shard in sorted(glob.glob(os.path.join(str(path), "*.safetensors")))[:1]:
        with safe_open(shard, "numpy") as f:
            return any(".escha_code" in k for k in f.keys())
    return False


# ── Compiled decode path ──────────────────────────────────────────────────────────────────────
# At batch=1 decode the eschamoe MoE compute is a FIXED-shape chain of many tiny Metal launches
# (per token: 2 gathers, 3 t128 Hadamard kernels, 2 escha_qmv matvec kernels, silu, casts). That
# chain is host-encode-bound — the CPU can't hand kernels to the GPU fast enough to keep it busy.
# Wrapping the pure compute in ``mx.compile`` lets MLX fuse/plan the launch chain once (traced on
# the first token) and re-issue it as a compact captured graph every subsequent token, which cuts
# the per-token Python + kernel-encode overhead. Weights are passed as ARGUMENTS (not closed over),
# so the SAME compiled graph is reused across all 40 layers and every token — one trace per shape.
# ``mx.fast.metal_kernel`` composes with ``mx.compile`` (see memory: metal-kernel-compiles-in-031),
# so escha_qmv's custom primitive is embedded in the traced graph rather than re-encoded per call.
#
# ESCHA_COMPILE=0 restores the exact eager path (A/B + numerical-debug escape hatch). Default on.
_ESCHA_COMPILE = os.environ.get("ESCHA_COMPILE", "1").strip().lower() not in ("0", "", "false", "no", "off")
_DECODE_FN: dict = {}


def _escha_decode_compute(x2, ind2, gu_code, gu_rin, gu_rout, dn_code, dn_rin, dn_rout, I, H):
    """Shape-stable eschamoe decode MoE compute (the S<=256 on-device path, no host sync).

    x2 [Tt, H], ind2 [Tt, top_k] -> y [Tt*top_k, H] bf16.  ``I``/``H`` are Python ints (model
    globals, baked when compiled).  This is byte-for-byte the body of the original
    ``_forward_ondevice`` minus the trailing reshape (kept in the caller so ``lead`` never enters
    the traced graph).  Called eagerly when ESCHA_COMPILE=0, or wrapped by ``mx.compile`` otherwise.
    """
    Tt = x2.shape[0]
    top_k = ind2.shape[-1]
    flat_e = ind2.reshape(-1)
    flat_tok = mx.repeat(mx.arange(Tt), top_k)
    xh = t128(x2[flat_tok], pre=gu_rin[flat_e])
    y_gu = escha_qmv(xh, flat_e, gu_code, 2, 2 * I)
    y_gu = t128(y_gu, post=gu_rout[flat_e])
    gated = nn.silu(y_gu[:, :I]) * y_gu[:, I:]
    xhd = t128(gated, pre=dn_rin[flat_e])
    y = escha_qmv(xhd, flat_e, dn_code, 3, H)
    y = t128(y, post=dn_rout[flat_e]).astype(mx.bfloat16)
    return y


def _get_decode_fn(I, H):
    """One ``mx.compile``d graph per (I, H) — i.e. one for the whole model. Cached so the trace
    happens once; same-shape calls (all layers, every token) reuse it. Distinct S (e.g. spec-verify
    with S>8) trace their own graph on first use — a small bounded set, each reused thereafter."""
    key = (I, H)
    fn = _DECODE_FN.get(key)
    if fn is None:
        fn = mx.compile(
            lambda x2, ind2, guc, gur, guo, dnc, dnr, dno:
                _escha_decode_compute(x2, ind2, guc, gur, guo, dnc, dnr, dno, I, H)
        )
        _DECODE_FN[key] = fn
    return fn


def _group_layout(ind_np, top_k, E):
    """Sort routed slots by expert, pad each expert block to 16 rows (prefill path)."""
    T = ind_np.shape[0]
    flat_e = ind_np.reshape(-1).astype(np.int64)
    S = flat_e.size
    slot_tok = np.repeat(np.arange(T, dtype=np.int64), top_k)
    order = np.argsort(flat_e, kind="stable")
    se = flat_e[order]
    counts = np.bincount(flat_e, minlength=E)
    ptiles = (counts + 15) // 16
    pstart = np.zeros(E, np.int64)
    if E > 1:
        pstart[1:] = np.cumsum(ptiles[:-1]) * 16
    estart = np.zeros(E, np.int64)
    if E > 1:
        estart[1:] = np.cumsum(counts[:-1])
    ppos = pstart[se] + (np.arange(S, dtype=np.int64) - estart[se])
    Mpad = int(ptiles.sum()) * 16
    padded_tok = np.zeros(Mpad, np.int64); padded_tok[ppos] = slot_tok[order]
    erow = np.zeros(Mpad, np.int64); erow[ppos] = se
    tile_expert = np.repeat(np.arange(E, dtype=np.int32), ptiles)
    return tile_expert, padded_tok, erow, order, ppos, S


class EschaSwitchGLU(nn.Module):
    """Drop-in for qwen3_5_moe's ``switch_mlp``: 2-bit eschamoe experts decoded on the fly
    (dense W never formed, weights never cached). Small S (decode / spec-verify) takes an
    on-device routed path (no host sync); large S (prefill) groups by expert into 16-row tiles."""

    def __init__(self, gu_code, gu_rin, gu_rout, dn_code, dn_rin, dn_rout, H, I):
        super().__init__()
        self.gu_code, self.dn_code = gu_code, dn_code
        cast = lambda a: a.astype(COMPUTE_DTYPE)      # hoist the rin/rout cast to load
        self.gu_rin, self.gu_rout = cast(gu_rin), cast(gu_rout)
        self.dn_rin, self.dn_rout = cast(dn_rin), cast(dn_rout)
        self.H, self.I = H, I

    def __call__(self, x, indices):
        H, I = self.H, self.I
        lead = tuple(indices.shape[:-1]); top_k = indices.shape[-1]
        Tt = 1
        for d in lead:
            Tt *= d
        ind2 = indices.reshape(Tt, top_k)
        x2 = x.reshape(Tt, H)
        S = Tt * top_k
        # Decode / spec-verify (small S): on-device per-row path, no host sync.
        # Prefill (large S): the grouped 16-row-tile path. It reuses each decoded 2-bit weight
        # across 16 tokens, which is ~1.6x faster than the per-row path here (444 vs 276 tok/s @16k)
        # despite a per-layer host grouping sync — the decode-reuse dominates. (The ideal is
        # on-device grouping WITH tile reuse; that's a follow-up kernel change.)
        if S <= 256:
            return self._forward_ondevice(x2, ind2, Tt, top_k, lead)
        ind_np = np.array(ind2)
        tile_expert, padded_tok, erow, dst_slot, valid_prow, S = _group_layout(ind_np, top_k, self.gu_code.shape[0])
        te = mx.array(tile_expert); tok = mx.array(padded_tok); er = mx.array(erow)
        xg = x2[tok]
        xh = t128(xg * self.gu_rin[er]).astype(mx.float16)
        y_gu = fused_moe_matmul(xh, te, self.gu_code, 2, 2 * I)
        y_gu = t128(y_gu) * self.gu_rout[er]
        gated = (nn.silu(y_gu[:, :I]) * y_gu[:, I:]).astype(mx.float16)
        xhd = t128(gated * self.dn_rin[er]).astype(mx.float16)
        y = fused_moe_matmul(xhd, te, self.dn_code, 3, H)
        # Keep the scatter buffer in the model dtype (bf16): the result is returned as x.dtype
        # anyway, so the f32 round-trip only doubled the prefill activation footprint for nothing.
        y = (t128(y) * self.dn_rout[er]).astype(x.dtype)
        out = mx.zeros((S, H), x.dtype)
        out[mx.array(dst_slot)] = y[mx.array(valid_prow)]
        return out.reshape(*lead, top_k, H)

    def _forward_ondevice(self, x2, ind2, Tt, top_k, lead):
        args = (x2, ind2, self.gu_code, self.gu_rin, self.gu_rout,
                self.dn_code, self.dn_rin, self.dn_rout)
        if _ESCHA_COMPILE:
            y = _get_decode_fn(self.I, self.H)(*args)
        else:
            y = _escha_decode_compute(*args, self.I, self.H)
        return y.reshape(*lead, top_k, self.H)


def load_escha_model(path: Path, config: dict[str, Any]):
    """Build the qwen3_5_moe trunk with eschamoe experts + int8 non-experts. -> (model, tokenizer)."""
    from mlx_lm.models import qwen3_5_moe
    from safetensors import safe_open

    path = Path(path)
    args = qwen3_5_moe.ModelArgs.from_dict(config)
    model = qwen3_5_moe.Model(args)
    tcfg = config["text_config"]
    H, I, E = tcfg["hidden_size"], tcfg["moe_intermediate_size"], tcfg["num_experts"]
    nlayers = tcfg["num_hidden_layers"]

    weights: dict[str, Any] = {}
    escha: dict[str, Any] = {}
    int8_raw: dict[str, tuple] = {}
    for shard in sorted(glob.glob(os.path.join(str(path), "*.safetensors"))):
        with safe_open(shard, "numpy") as f:
            for k in f.keys():
                if k.startswith("mtp.") or k.endswith(".weight_scale"):
                    continue
                if k.endswith(".weight_int8"):
                    base = k[: -len(".weight_int8")]
                    w = mx.array(f.get_tensor(k))
                    s = mx.array(f.get_tensor(base + ".weight_scale"))
                    if "embed_tokens" not in base and "conv1d" not in base:
                        int8_raw[base] = (w, s)
                    else:
                        weights[base + ".weight"] = w.astype(mx.bfloat16) * s[:, None].astype(mx.bfloat16)
                elif ".experts." in k and ".escha_" in k:
                    escha[k] = mx.array(f.get_tensor(k))
                else:
                    weights[k] = mx.array(f.get_tensor(k))

    def g(layer, proj, suf):
        return escha[f"model.language_model.layers.{layer}.mlp.experts.{proj}.escha_{suf}"]

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)

    # eschamoe experts, on the fly (no dense, no cache)
    for l in range(nlayers):
        model.language_model.model.layers[l].mlp.switch_mlp = EschaSwitchGLU(
            g(l, "gate_up_proj", "code"), g(l, "gate_up_proj", "rin"), g(l, "gate_up_proj", "rout"),
            g(l, "down_proj", "code"), g(l, "down_proj", "rin"), g(l, "down_proj", "rout"), H, I)

    # int8 non-expert Linears (incl lm_head; key "lm_head.weight_int8" has no lm prefix)
    lm = model.language_model
    for base, (w, s) in int8_raw.items():
        key = base
        for pre in ("model.language_model.", "language_model.model.", "language_model."):
            if key.startswith(pre):
                key = key[len(pre):]; break
        parts = key.split(".")
        if parts[0] == "lm_head":
            parent, attr = lm, "lm_head"
        elif parts[0] == "layers":
            parent = lm.model.layers[int(parts[1])]
            for p in parts[2:-1]:
                parent = getattr(parent, p)
            attr = parts[-1]
        else:
            continue
        if isinstance(getattr(parent, attr, None), nn.Linear):
            setattr(parent, attr, Int8Linear(w, s))

    mx.eval(model.parameters())

    from mlx_lm.utils import load_tokenizer
    tokenizer = load_tokenizer(path)
    return model, tokenizer
