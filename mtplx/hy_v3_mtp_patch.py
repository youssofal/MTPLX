"""Hy3 (hy_v3) MTP support injection.

Real hy_v3 exports ship the appended NextN layer in tencent-native layout —
canonical ``model.layers.{num_hidden_layers}.*`` tensors inside the standard
sharded checkpoint. The released mlx-lm hy_v3 model class (the #1211 line)
loads the trunk only and sanitizes those tensors away, so injection grafts the
draft head from the checkpoint itself: build the module (enorm/hnorm/eh_proj +
one DecoderLayer + final_layernorm), load and (per config) quantize its
weights, and install the MTPLX runtime surface (``mtp_forward`` /
``mtp_update_cache`` / ``make_mtp_cache``, plus ``return_hidden`` on the trunk
forward) as a subclass wrapper.

When a future mlx-lm loads the head natively (``model.mtp`` present), the
graft is skipped and the same wrapper binds the native module unchanged.

Architecture contract (must match the checkpoint):
- one appended NextN layer (depth 1) with its own MoE MLP;
- draft input is concat[enorm(next-token embedding), hnorm(trunk hidden)]
  with the trunk hidden taken POST-final-norm ("embedding_hidden" order).
  Measured, not assumed: teacher-forced draft/target argmax agreement on a
  1024-token real-code prompt is 0.773 with the post-norm hidden vs 0.387
  pre-norm (and 0.000 with the concat order flipped) — the checkpoint's head
  was trained on the normed trunk output;
- shared embeddings and lm_head.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TOP_LEVEL_MTP_PARTS = ("enorm", "hnorm", "eh_proj", "final_layernorm")


def is_hy_v3_config(config: dict[str, Any]) -> bool:
    """True for any hy_v3 checkpoint (with or without the MTP head)."""
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(a) for a in config.get("architectures") or []]
    return model_type == "hy_v3" or any(
        a in ("HyV3ForCausalLM", "HYV3ForCausalLM") for a in architectures
    )


def install_hy_v3_model_shim() -> None:
    """Make ``mlx_lm.models.hy_v3`` importable on released mlx-lm.

    No released mlx-lm ships a hy_v3 model class (ml-explore/mlx-lm#1211 is
    open; the MTP surface lives only in the #1485 stack). Register the vendored
    class so ``mlx_lm.utils.load`` can build the model. If a future upstream
    module exists AND exposes the MTP surface (``predict_next_tokens``), prefer
    it; a base-only upstream (which strips MTP in sanitize) is overridden, as
    it would silently drop the head this backend exists to use. Idempotent.
    """
    name = "mlx_lm.models.hy_v3"
    mod = sys.modules.get(name)
    if mod is not None and hasattr(getattr(mod, "Model", None), "predict_next_tokens"):
        return
    if mod is None:
        try:
            import importlib

            mod = importlib.import_module(name)
        except ImportError:
            mod = None
    if mod is not None and hasattr(getattr(mod, "Model", None), "predict_next_tokens"):
        return
    from . import vendored_hy_v3

    sys.modules[name] = vendored_hy_v3
    logger.info("[Hy3] vendored hy_v3 model class registered as %s", name)


def is_hy_v3_mtp_config(config: dict[str, Any]) -> bool:
    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(a) for a in config.get("architectures") or []]
    if model_type != "hy_v3" and "HyV3ForCausalLM" not in architectures:
        return False
    return int(config.get("num_nextn_predict_layers") or 0) > 0


def _num_trunk_layers(config: dict[str, Any], model: Any = None) -> int:
    declared = config.get("num_hidden_layers")
    if declared is not None:
        return int(declared)
    return len(model.model.layers)


def _spec_layer_prefix(config: dict[str, Any], model: Any = None) -> str:
    return f"model.layers.{_num_trunk_layers(config, model)}."


def _load_appended_layer_weights(
    model_path: Path, config: dict[str, Any], model: Any = None
) -> dict[str, Any]:
    """Read the NextN layer's tensors from the standard sharded checkpoint.

    Returns them renamed for the grafted module: ``enorm/hnorm/eh_proj/
    final_layernorm`` stay top-level, everything else (attention, MoE, layer
    norms) moves under ``layer.``. Empty dict when the checkpoint carries no
    appended layer (AR-only export).
    """
    import mlx.core as mx

    prefix = _spec_layer_prefix(config, model)
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
        candidates = [model_path / s for s in shards]
    else:
        candidates = sorted(model_path.glob("model*.safetensors"))

    raw: dict[str, Any] = {}
    for file in candidates:
        for key, value in mx.load(str(file)).items():
            if key.startswith(prefix):
                raw[key] = value

    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        suffix = key.removeprefix(prefix)
        if suffix.split(".", 1)[0] in _TOP_LEVEL_MTP_PARTS:
            mapped[suffix] = value
        else:
            mapped[f"layer.{suffix}"] = value
    return mapped


def _make_grafted_mtp(args: Any, spec_idx: int):
    import mlx.nn as nn
    from mlx_lm.models import hy_v3 as impl

    class _HyV3GraftedMTP(nn.Module):
        """Checkpoint-grafted NextN head; call-compatible with the native block."""

        def __init__(self):
            super().__init__()
            self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.eh_proj = nn.Linear(
                args.hidden_size * 2, args.hidden_size, bias=False
            )
            self.layer = impl.DecoderLayer(args, layer_idx=spec_idx)
            self.final_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )

        def __call__(self, hidden_states, e_next, mask, cache=None):
            import mlx.core as mx

            mixed = self.eh_proj(
                mx.concatenate(
                    [self.enorm(e_next), self.hnorm(hidden_states)], axis=-1
                )
            )
            # pre-final-norm draft hidden; the head path applies
            # final_layernorm before the shared lm_head.
            return self.layer(mixed, mask, cache=cache)

    return _HyV3GraftedMTP()


def _quantize_grafted_mtp(
    mtp: Any, config: dict[str, Any], weights: dict[str, Any], spec_prefix: str
) -> None:
    """Quantize grafted modules whose checkpoint tensors are quantized.

    Per-module entries in ``config["quantization"]`` (keyed by the original
    ``model.layers.{N}.<module>`` path) take precedence over the global
    group_size/bits — real exports quantize the draft head at a different
    precision than the 2-bit expert body.
    """
    import mlx.nn as nn

    quantization = config.get("quantization") or config.get("quantization_config")
    if not quantization:
        return
    prefix = spec_prefix

    def class_predicate(path: str, module: Any):
        if not hasattr(module, "to_quantized"):
            return False
        if f"{path}.scales" not in weights:
            return False
        original = prefix + (
            path.removeprefix("layer.") if path.startswith("layer.") else path
        )
        spec = quantization.get(original) or quantization
        try:
            return {
                "group_size": int(spec["group_size"]),
                "bits": int(spec["bits"]),
                "mode": spec.get("mode", "affine"),
            }
        except (KeyError, TypeError, ValueError):
            return False

    if "group_size" not in quantization or "bits" not in quantization:
        return
    nn.quantize(
        mtp,
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def _checkpoint_carries_mtp_tensors(
    model_path: Path, config: dict[str, Any], model: Any = None
) -> bool:
    """True when the checkpoint ships draft tensors in either hy_v3 layout.

    Key-name scan only (index json when present, safetensors headers
    otherwise) — never materializes tensors, so it is cheap even on the
    295B artifact.
    """
    prefixes = (_spec_layer_prefix(config, model), "model.mtp.", "mtp.")
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        except (OSError, ValueError):
            weight_map = {}
        if weight_map:
            return any(str(key).startswith(prefixes) for key in weight_map)
    try:
        from safetensors import safe_open

        for file in sorted(model_path.glob("model*.safetensors")):
            with safe_open(str(file), framework="np") as handle:
                if any(str(key).startswith(prefixes) for key in handle.keys()):
                    return True
        return False
    except Exception:
        # Header scan unavailable: fall back to the graft loader's reader.
        return bool(_load_appended_layer_weights(model_path, config, model))


def inject_hy_v3_mtp_support(
    model: Any,
    path: Path,
    config: dict[str, Any],
    contract: Any,
) -> bool:
    """Install the MTPLX draft surface on an already-loaded hy_v3 model.

    Returns True when the model exposes a usable MTP head. Raises if the
    config promises MTP but the checkpoint carries no draft tensors (an
    AR-only export) — including when the loaded model class constructs a
    native ``model.mtp`` submodule unconditionally (the vendored class
    does), where a randomly initialized head must never pass as usable.
    """
    if not is_hy_v3_mtp_config(config):
        return False

    import inspect

    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    from mlx_lm.models.cache import KVCache

    path = Path(path)
    native = getattr(model, "mtp", None) is not None
    if native and not _checkpoint_carries_mtp_tensors(path, config, model):
        raise RuntimeError(
            f"{path}: config declares num_nextn_predict_layers="
            f"{config.get('num_nextn_predict_layers')} but the checkpoint "
            f"carries no {_spec_layer_prefix(config, model)}* or model.mtp.* "
            "tensors — an AR-only export; the model class's constructed MTP "
            "submodule is uninitialized and must not serve as a draft head."
        )
    grafted = None
    if not native:
        weights = _load_appended_layer_weights(path, config, model)
        if not weights:
            raise RuntimeError(
                f"{path}: config declares num_nextn_predict_layers="
                f"{config.get('num_nextn_predict_layers')} but the checkpoint "
                f"carries no {_spec_layer_prefix(config, model)}* tensors and the "
                "loaded model has no native MTP submodule — an AR-only export."
            )
        args = getattr(model, "args", None)
        if args is None:
            from mlx_lm.models import hy_v3 as impl

            args = impl.ModelArgs.from_dict(config)
        grafted = _make_grafted_mtp(args, _num_trunk_layers(config, model))
        _quantize_grafted_mtp(
            grafted, config, weights, _spec_layer_prefix(config, model)
        )
        grafted.load_weights(list(weights.items()), strict=True)
        mx.eval(grafted.parameters())

    # validate_mtp_support (mtp_patch.py) requires model.mtp.layers to be a
    # truthy container; both the native MTPBlock and the graft expose their
    # single decoder as `.layer`. The list assignment registers an alias (a
    # child-module container), not a weight copy.
    if grafted is not None:
        model.mtp = grafted
    if getattr(model.mtp, "layers", None) is None:
        model.mtp.layers = [model.mtp.layer]

    native_return_hidden = "return_hidden_states" in inspect.signature(
        type(model).__call__
    ).parameters
    original_outer_class = model.__class__

    # Hy3 has exactly one native draft wiring (pre-final-norm trunk hidden,
    # [enorm(embedding), hnorm(hidden)] concat). Like the sibling appended-layer
    # backends (glm_mtp defaults post_norm, step3p5 pre_norm), we ACCEPT and
    # IGNORE whatever hidden_variant / concat_order the runtime resolves from
    # the contract default — raising would break every draft call, since
    # runtime.draft_mtp resolves None/auto/contract to contract.hidden_variant
    # (== 'post_norm' for a bare MTPContract).
    class _MTPLXHyV3Model(original_outer_class):
        def make_cache(self):
            # hy_v3 (plain causal MoE attention) does not define a native
            # make_cache like the hybrid-attention backends (e.g. qwen3_5) do;
            # mlx_lm.server falls back to a default per-layer KVCache list in
            # this situation (mlx_lm.models.cache.make_prompt_cache's `else`
            # branch). Replicate that directly here -- calling
            # make_prompt_cache(self) would recurse, since it dispatches back
            # to this very method once it sees the model has a make_cache.
            return [KVCache() for _ in self.model.layers]

        def _head_logits(self, hidden):
            if getattr(self.args, "enable_lm_head_fp32", False):
                hidden = hidden.astype(mx.float32)
            if getattr(self.args, "tie_word_embeddings", False):
                return self.model.embed_tokens.as_linear(hidden)
            return self.lm_head(hidden)

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
                raise ValueError("Hy3 MTP backend does not support input_embeddings")
            if not return_hidden:
                return super().__call__(inputs, cache=cache)
            if native_return_hidden:
                # A native forward returns the raw (pre-norm) trunk hidden;
                # normalize it to match the measured draft contract.
                logits, h = super().__call__(
                    inputs, cache=cache, return_hidden_states=True
                )
                return logits, self.model.norm(h)
            # Released hy_v3 has no return_hidden_states: walk the trunk and
            # hand back the POST-final-norm hidden the draft layer consumes
            # (measured contract — see module docstring).
            inner = self.model
            if getattr(inner, "pipeline_size", 1) > 1:
                raise ValueError(
                    "Hy3 MTP return_hidden is single-rank only (pipeline off)"
                )
            h = inner.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(inner.layers)
            mask = create_attention_mask(h, cache[0])
            for layer, c in zip(inner.layers, cache):
                h = layer(h, mask, cache=c)
            normed = inner.norm(h)
            return self._head_logits(normed), normed

        def _hy3_mtp_logits(self, h_mtp):
            native_logits = getattr(self, "_logits", None)
            if callable(native_logits):
                return native_logits(h_mtp)
            return self._head_logits(self.mtp.final_layernorm(h_mtp))

        def mtp_forward(
            self,
            hidden_states,
            next_token_ids,
            cache=None,
            mtp_cache=None,
            concat_order=None,
            return_hidden: bool = False,
            mtp_hidden_variant: str = "post_norm",
            position_offset: int | None = None,
            mtp_depth: int | None = None,
        ):
            # depth is informational for a single-layer head — the one NextN
            # layer is reused regardless (mirrors GLM's modulo-into-layers).
            layer_cache = mtp_cache if mtp_cache is not None else cache
            if isinstance(layer_cache, list):
                layer_cache = layer_cache[0] if layer_cache else None
            e_next = self.model.embed_tokens(next_token_ids)
            mask = create_attention_mask(e_next, layer_cache)
            h_mtp = self.mtp(hidden_states, e_next, mask, layer_cache)
            logits = self._hy3_mtp_logits(h_mtp)
            if not return_hidden:
                return logits
            return logits, h_mtp

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

    model.__class__ = _MTPLXHyV3Model
    logger.info(
        "[Hy3 MTP inject] %s head bound (depth 1) for %s",
        "native" if native else "checkpoint-grafted",
        path,
    )
    return True
