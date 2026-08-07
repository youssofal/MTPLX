"""Runtime MTP injection for Qwen3.5/3.6 MoE (``qwen3_5_mtp``).

The Qwen3.5-MoE MTP export ships a single appended NextN predictor in a
``mtp.*`` namespace (typically a separate ``model-mtp-head.safetensors``):

    mtp.pre_fc_norm_embedding   RMSNorm on the next-token embedding   (enorm)
    mtp.pre_fc_norm_hidden      RMSNorm on the trunk hidden           (hnorm)
    mtp.fc                      Linear concat[e, h] (2*H -> H)         (eh_proj)
    mtp.layers.0                one FULL-attention Qwen3.5 MoE block   (mtp_block)
    mtp.norm                    RMSNorm before the shared lm_head
    (lm_head is shared with the trunk)

Two facts make this simpler than it looks:

1. **The trunk is a plain ``qwen3_5_moe``.** The MTP checkpoint differs from the
   AR export only in the top-level ``model_type`` string (``qwen3_5_mtp`` vs
   ``qwen3_5_moe``) — the ``text_config`` is identical. mlx-lm has no
   ``qwen3_5_mtp`` module, so ``install_qwen3_5_mtp_trunk_shim`` registers a
   ``sys.modules`` alias that points ``mlx_lm.models.qwen3_5_mtp`` at
   ``qwen3_5_moe`` before load. The trunk then loads exactly like the AR export
   and the extra ``mtp.*`` weights are simply ignored by the trunk.

2. **The head is one full-attention block.** ``mtp.layers.0`` has ``self_attn``
   (not the trunk's hybrid ``linear_attn``) + a 256-expert MoE, i.e. exactly the
   layer ``qwen3_5.DecoderLayer`` builds when ``(layer_idx+1) %
   full_attention_interval == 0``. We reuse that class so the module tree and
   math match natively; keys map 1:1 after stripping the ``mtp.`` prefix.

Status: loads + drafts, hardware-verified on Qwen3.6-35B-A3B (M5 Max 128 GB).
Weight coverage is exact (strict load of all 46 mtp.* tensors); the trunk loads
coherently (the double-shift note below) and the head reaches ~90% 1-step greedy
draft acceptance on a structured smoke — well clear of the ~3% "donkey band",
confirming the default wiring (pre-norm trunk hidden, concat order
[embedding, hidden]). A full acceptance sweep across diverse text is the
remaining production-tuning step; a contract override can flip the variant.

Note (trunk double-shift): mlx-lm's qwen3_5 sanitize adds +1.0 to trunk norm
weights when mtp.* keys are present (a raw zero-centered-norm convention). This
export already stores final-convention norms, so the trunk-load shim strips
mtp.* before sanitize to avoid a double shift (which otherwise corrupts the
trunk into gibberish).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .artifacts import expected_mtp_file, text_config

logger = logging.getLogger(__name__)

QWEN3_5_MTP_MODEL_TYPES = {"qwen3_5_mtp"}

# Which trunk layer's eschamoe experts the Escha MTP draft head borrows (its own predictor ships
# no routed experts). Layer-robust in the acceptance sweep; 20 was best. Env-overridable for A/B.
ESCHA_MTP_SHARE_LAYER = int(os.environ.get("ESCHA_MTP_SHARE_LAYER", "20"))


def is_escha_qwen3_5_mtp(config: dict[str, Any], model_path: Path | str) -> bool:
    """True for an Escha-W2 checkpoint (qwen3_5_moe 2-bit trunk) that ships an MTP predictor.

    Distinct from :func:`is_qwen3_5_mtp_config` (model_type ``qwen3_5_mtp``): Escha keeps
    model_type ``qwen3_5_moe`` and its MTP head omits routed experts, so it takes the borrow-
    trunk-experts path in :func:`inject_qwen3_5_mtp_support`.
    """
    if int(text_config(config).get("mtp_num_hidden_layers", 0) or 0) <= 0:
        return False
    from .escha_load import is_escha_checkpoint

    return is_escha_checkpoint(Path(model_path), config)


def _model_type(config: dict[str, Any]) -> str:
    return str(config.get("model_type") or "").lower()


def _num_mtp_layers(config: dict[str, Any]) -> int:
    tcfg = text_config(config)
    return int(
        config.get("num_nextn_predict_layers")
        or tcfg.get("num_nextn_predict_layers")
        or 0
    )


def is_qwen3_5_mtp_config(config: dict[str, Any]) -> bool:
    """True for Qwen3.5-MoE configs that declare an appended MTP predictor."""
    return _model_type(config) in QWEN3_5_MTP_MODEL_TYPES and _num_mtp_layers(config) > 0


def install_qwen3_5_mtp_trunk_shim() -> None:
    """Alias ``mlx_lm.models.qwen3_5_mtp`` -> ``qwen3_5_moe`` so the trunk loads.

    The MTP export's top-level ``model_type`` is ``qwen3_5_mtp``, for which
    mlx-lm has no module; the trunk itself is a vanilla ``qwen3_5_moe``. Making
    the module importable (as an alias) lets ``mlx_lm.utils.load`` build the
    trunk with the correct classes; the ``mtp.*`` tensors it doesn't recognise
    are loaded onto the head separately by ``inject_qwen3_5_mtp_support``.
    Idempotent.
    """
    name = "mlx_lm.models.qwen3_5_mtp"
    if name in sys.modules:
        return
    import types

    import mlx_lm.models.qwen3_5_moe as base

    class _TrunkModel(base.Model):
        def sanitize(self, weights):
            # mlx-lm's qwen3_5 sanitize shifts trunk norm weights by +1.0 when
            # any ``mtp.*`` key is present (a raw-checkpoint zero-centered-norm
            # convention). This export stores trunk norms already in final
            # convention (conv1d is sanitized), so that shift would double-apply
            # and corrupt the trunk. Drop ``mtp.*`` here — the head is loaded
            # separately by ``inject_qwen3_5_mtp_support`` — so the shift is
            # driven only by the (correct) unsanitized-conv1d signal.
            weights = {k: v for k, v in weights.items() if "mtp." not in str(k)}
            return super().sanitize(weights)

    shim = types.ModuleType(name)
    shim.Model = _TrunkModel
    shim.ModelArgs = base.ModelArgs
    sys.modules[name] = shim


def _strip_mtp_prefix(key: str) -> str | None:
    """``mtp.<rest>`` -> ``<rest>`` (the local head module tree); else None."""
    k = str(key)
    for outer in ("language_model.", "model.model.", "model."):
        if k.startswith(outer) and "mtp." in k:
            k = k[k.index("mtp."):]
            break
    if k.startswith("mtp."):
        return k[len("mtp."):]
    return None


def _candidate_weight_files(model_path: Path, config: dict[str, Any]) -> list[Path]:
    mtp_file = expected_mtp_file(model_path, config)
    if mtp_file.exists():
        return [mtp_file]
    head = model_path / "model-mtp-head.safetensors"
    if head.exists():
        return [head]
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            weight_map = {}
        selected = {
            model_path / rel
            for key, rel in weight_map.items()
            if _strip_mtp_prefix(key) is not None
        }
        if selected:
            return sorted(selected)
    return sorted(model_path.glob("model*.safetensors"))


def _load_mtp_weights(paths: list[Path]) -> dict[str, Any]:
    import mlx.core as mx

    mapped: dict[str, Any] = {}
    for path in paths:
        if path.suffix != ".safetensors":
            continue
        for key, value in mx.load(str(path)).items():
            local = _strip_mtp_prefix(key)
            if local is not None:
                mapped[local] = value
    return mapped


def _full_attention_layer_idx(args: Any) -> int:
    """A layer_idx that qwen3_5.DecoderLayer builds as full-attention."""
    interval = int(getattr(args, "full_attention_interval", 4) or 4)
    return interval - 1  # (idx+1) % interval == 0 -> self_attn branch


def _make_qwen3_5_mtp_module(args: Any):
    import mlx.nn as nn
    from mlx_lm.models.qwen3_5 import DecoderLayer

    class _Qwen35MTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            # one FULL-attention Qwen3.5 MoE decoder block
            self.layers = [DecoderLayer(args=args, layer_idx=_full_attention_layer_idx(args))]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    return _Qwen35MTP()


def _quantize_like_trunk(mtp: Any, config: dict[str, Any], contract: Any | None) -> None:
    """Quantise the head to match the checkpoint's quant config (weights are
    stored quantized: fc/self_attn/mlp carry .scales/.biases)."""
    q = config.get("quantization") or text_config(config).get("quantization")
    if not q:
        return
    import mlx.nn as nn

    nn.quantize(
        mtp,
        group_size=int(q.get("group_size", 64)),
        bits=int(q.get("bits", 4)),
        mode=str(q.get("mode", "affine")),
    )


def _validate_load_coverage(mtp: Any, weights: dict[str, Any]) -> None:
    from mlx.utils import tree_flatten

    current = tree_flatten(mtp.parameters(), destination={})
    supplied = dict(weights)
    extra = sorted(set(supplied) - set(current))
    missing = sorted(set(current) - set(supplied))
    mismatched = [
        (k, tuple(current[k].shape), tuple(supplied[k].shape))
        for k in sorted(set(current) & set(supplied))
        if tuple(current[k].shape) != tuple(supplied[k].shape)
    ]
    if not extra and not missing and not mismatched:
        return
    parts = []
    if missing:
        parts.append(f"missing={missing[:12]}" + (" ..." if len(missing) > 12 else ""))
    if extra:
        parts.append(f"extra={extra[:12]}" + (" ..." if len(extra) > 12 else ""))
    if mismatched:
        parts.append("shape_mismatch=" + str([f"{k}: want {w}, got {g}" for k, w, g in mismatched[:6]]))
    raise ValueError("Qwen3.5 MTP overlay does not match runtime module tree: " + "; ".join(parts))


def inject_qwen3_5_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: Any | None = None,
) -> bool:
    """Attach Qwen3.5-MoE native MTP support to a loaded ``qwen3_5_moe`` trunk."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_5 import TextModelArgs

    from .mtp_patch import _text_model

    model_path = Path(model_path)
    escha = is_escha_qwen3_5_mtp(config, model_path)
    if not is_qwen3_5_mtp_config(config) and not escha:
        return False

    tcfg = text_config(config)
    args = TextModelArgs.from_dict(tcfg)

    weights = _load_mtp_weights(_candidate_weight_files(model_path, config))
    if not weights:
        logger.warning("[Qwen3.5 MTP inject] no mtp.* weights found in %s", model_path)
        return False

    if escha:
        # Escha stores RMSNorm weights as (w-1); shift every head norm +1 (the trunk gets this
        # from mlx-lm sanitize, but the raw mtp.* norms loaded here do not).
        weights = {
            k: (v + 1.0 if ("norm" in k and getattr(v, "ndim", 0) == 1) else v)
            for k, v in weights.items()
        }

    # Operate at the qwen3_5 *TextModel* level (``.model`` inner trunk + ``.lm_head``), which is
    # what ``_text_model`` returns and what ``validate_mtp_support`` inspects. For the standard
    # mlx-lm outer ``qwen3_5_moe.Model`` this is ``model.language_model``; for a bare TextModel it
    # is ``model`` itself. The MTP surface lives here so ``.mtp`` sits where validate looks, and a
    # thin delegating wrapper (below) re-exposes it on the outer model the runtime actually holds.
    text_model = _text_model(model)

    mtp = _make_qwen3_5_mtp_module(args)
    if escha:
        # Escha's MTP predictor ships NO routed experts (17 fp16 tensors): load only the shipped
        # non-expert weights and BORROW the trunk's eschamoe experts for the head's 256-way router.
        mtp.load_weights(list(weights.items()), strict=False)
        mtp.layers[0].mlp.switch_mlp = (
            text_model.model.layers[ESCHA_MTP_SHARE_LAYER].mlp.switch_mlp
        )
        logger.info("[Qwen3.5 MTP inject] escha head: borrow trunk layer %d experts, %d tensors",
                    ESCHA_MTP_SHARE_LAYER, len(weights))
    else:
        _quantize_like_trunk(mtp, config, contract)
        _validate_load_coverage(mtp, weights)
        mtp.load_weights(list(weights.items()), strict=True)
    mx.eval(mtp.parameters())

    text_model.mtp = mtp
    text_model._mtplx_hidden_variant = "pre_norm"
    text_model._mtplx_concat_order = "embedding_hidden"

    original_text_class = text_model.__class__

    class _MTPLXQwen35TextModel(original_text_class):
        def _lm_logits(self, h):
            lm = getattr(self, "lm_head", None)
            if lm is not None:
                return lm(h)
            return self.model.embed_tokens.as_linear(h)

        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            **kwargs,
        ):
            if input_embeddings is not None:
                raise ValueError("Qwen3.5 MTP backend does not support input_embeddings")
            if not return_hidden:
                return super().__call__(inputs, cache=cache)
            # Expose the pre-final-norm residual stream: Qwen3_5TextModel applies
            # ``self.norm`` before returning, so temporarily swap it for identity
            # (avoids re-running the hybrid linear/full attention layer loop).
            inner = self.model  # Qwen3_5TextModel (the inner trunk)
            real_norm = inner.norm
            try:
                inner.norm = lambda x: x
                pre_norm = inner(inputs, cache=cache)
            finally:
                inner.norm = real_norm
            post_norm = real_norm(pre_norm)
            logits = self._lm_logits(post_norm)
            variant = hidden_variant or getattr(self, "_mtplx_hidden_variant", "pre_norm")
            hidden = pre_norm if variant == "pre_norm" else post_norm
            return logits, hidden

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str = "pre_norm",
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            layer_cache = mtp_cache if mtp_cache is not None else cache
            if isinstance(layer_cache, list):
                layer_cache = layer_cache[0] if layer_cache else None
            e = self.mtp.pre_fc_norm_embedding(self.model.embed_tokens(next_token_ids))
            h = self.mtp.pre_fc_norm_hidden(hidden_states)
            # vLLM/DeepSeek reference concat order is [embedding, hidden].
            mixed = self.mtp.fc(mx.concatenate([e, h], axis=-1))
            mask = create_attention_mask(mixed, layer_cache)
            hidden = self.mtp.layers[0](mixed, mask=mask, cache=layer_cache)
            logits = self._lm_logits(self.mtp.norm(hidden))
            if not return_hidden:
                return logits
            return logits, hidden

        def mtp_update_cache(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            _logits, hidden = self.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                return_hidden=True,
                mtp_depth=mtp_depth,
            )
            return hidden

        def make_mtp_cache(self):
            return [KVCache()]

    text_model.__class__ = _MTPLXQwen35TextModel

    # The runtime holds the outer model (``self.model`` in MTPLXRuntime). When that is the mlx-lm
    # ``qwen3_5_moe.Model`` wrapper, re-expose the MTP surface on it by delegating to the patched
    # TextModel — same pattern as the generic ``inject_mtp_support``.
    if getattr(model, "language_model", None) is text_model:
        model.mtp = mtp
        original_outer_class = model.__class__

        class _MTPLXQwen35OuterModel(original_outer_class):
            def __call__(
                self,
                inputs,
                cache=None,
                return_hidden: bool = False,
                input_embeddings=None,
                hidden_variant: str | None = None,
                **kwargs,
            ):
                return self.language_model(
                    inputs,
                    cache=cache,
                    return_hidden=return_hidden,
                    input_embeddings=input_embeddings,
                    hidden_variant=hidden_variant,
                    **kwargs,
                )

            def mtp_forward(self, *args, **kwargs):
                return self.language_model.mtp_forward(*args, **kwargs)

            def mtp_update_cache(self, *args, **kwargs):
                return self.language_model.mtp_update_cache(*args, **kwargs)

            def make_mtp_cache(self):
                return self.language_model.make_mtp_cache()

        model.__class__ = _MTPLXQwen35OuterModel

    logger.info(
        "[Qwen3.5 MTP inject] native head bound (depth 1, %d tensors) for %s",
        len(weights),
        model_path,
    )
    return True


def validate_qwen3_5_mtp_support(model: Any) -> bool:
    if getattr(model, "mtp", None) is None:
        return False
    if not getattr(model.mtp, "layers", None):
        return False
    return callable(getattr(model, "mtp_forward", None)) and callable(
        getattr(model, "make_mtp_cache", None)
    )
