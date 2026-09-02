"""Runtime MTP injection for MiMo MLX models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .artifacts import expected_mtp_file, text_config

logger = logging.getLogger(__name__)


def _num_mtp_layers(config: dict[str, Any]) -> int:
    tcfg = text_config(config)
    return int(
        tcfg.get("num_nextn_predict_layers")
        or tcfg.get("mtp_num_hidden_layers")
        or config.get("num_nextn_predict_layers")
        or 0
    )


def _model_type(config: dict[str, Any]) -> str:
    tcfg = text_config(config)
    return str(tcfg.get("model_type") or config.get("model_type") or "").lower()


def is_mimo_mtp_config(config: dict[str, Any]) -> bool:
    return _model_type(config) == "mimo" and _num_mtp_layers(config) > 0


def _load_weight_file(path: Path) -> dict[str, Any]:
    import mlx.core as mx

    if path.suffix == ".json":
        return {}
    return dict(mx.load(str(path)))


def _candidate_weight_files(model_path: Path, config: dict[str, Any]) -> list[Path]:
    mtp_file = expected_mtp_file(model_path, config)
    if mtp_file.exists():
        return [mtp_file]

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            weight_map = {}
        start = int(text_config(config).get("num_hidden_layers") or config.get("num_hidden_layers") or 0)
        count = _num_mtp_layers(config)
        wanted_prefixes = tuple(
            [f"model.layers.{start + i}." for i in range(count)]
            + [f"model.mtp_layers.{i}." for i in range(count)]
        )
        selected = {
            model_path / rel
            for key, rel in weight_map.items()
            if str(key).startswith(wanted_prefixes) or str(key).startswith("lm_head.")
        }
        if selected:
            return sorted(selected)

    return sorted(model_path.glob("model*.safetensors"))


def _has_complete_mimo_mtp_payload(
    weights: dict[str, Any],
    *,
    num_mtp_layers: int,
) -> bool:
    """Return true only when every declared MiMo MTP layer has real weights.

    A prefix-mismatched checkpoint still yields a non-empty ``mapped`` from
    stray keys, and ``strict=False`` then loads nothing -- injection would
    report success while the draft head stays randomly initialized.
    """

    for local_idx in range(num_mtp_layers):
        prefix = f"layers.{local_idx}."
        required = (
            f"{prefix}token_layernorm.weight",
            f"{prefix}hidden_layernorm.weight",
            f"{prefix}input_proj.weight",
            f"{prefix}final_layernorm.weight",
        )
        if not all(key in weights for key in required):
            return False
        if not any(key.startswith(f"{prefix}mtp_block.") for key in weights):
            return False
    return True


def _rewrite_mimo_mtp_weights(
    raw: dict[str, Any],
    *,
    start_layer: int,
    num_mtp_layers: int,
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    special = ("token_layernorm.", "hidden_layernorm.", "input_proj.", "final_layernorm.")
    for key, value in raw.items():
        if "rotary_emb.inv_freq" in key:
            continue
        if key == "lm_head.weight":
            mapped["lm_head.weight"] = value
            continue
        if key.startswith("mtp."):
            mapped[key.removeprefix("mtp.")] = value
            continue
        if key.startswith("layers.") or key.startswith("lm_head."):
            mapped[key] = value
            continue
        for local_idx in range(num_mtp_layers):
            raw_prefixes = (
                f"model.mtp_layers.{local_idx}.",
                f"model.mtp_layers.{start_layer + local_idx}.",
                f"model.layers.{start_layer + local_idx}.",
            )
            matched_prefix = next((prefix for prefix in raw_prefixes if key.startswith(prefix)), None)
            if matched_prefix is None:
                continue
            suffix = key.removeprefix(matched_prefix)
            local_prefix = f"layers.{local_idx}"
            if suffix.startswith(special):
                mapped[f"{local_prefix}.{suffix}"] = value
            elif suffix.startswith("embed_tokens."):
                pass
            else:
                mapped[f"{local_prefix}.mtp_block.{suffix}"] = value
            break
    return mapped


def _quantize_for_loaded_weights(mtp: Any, config: dict[str, Any], weights: dict[str, Any]) -> None:
    import mlx.nn as nn

    quantization = config.get("quantization") or config.get("quantization_config") or {}
    if not quantization:
        return
    if "group_size" not in quantization or "bits" not in quantization:
        return

    def class_predicate(path: str, module: Any):
        if not hasattr(module, "to_quantized"):
            return False
        if f"{path}.scales" in weights:
            return {
                "group_size": int(quantization["group_size"]),
                "bits": int(quantization["bits"]),
                "mode": quantization.get("mode", "affine"),
            }
        return False

    nn.quantize(
        mtp,
        group_size=int(quantization["group_size"]),
        bits=int(quantization["bits"]),
        mode=quantization.get("mode", "affine"),
        class_predicate=class_predicate,
    )


def _make_mimo_mtp_module(config: dict[str, Any], args: Any):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import mimo
    from mlx_lm.models.base import create_attention_mask

    start_layer = int(getattr(args, "num_hidden_layers"))
    num_mtp_layers = _num_mtp_layers(config)

    class _MiMOMTPLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.hidden_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.input_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.mtp_block = mimo.TransformerBlock(args)
            self.final_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, input_ids, previous_hidden_states, *, embed_tokens, cache=None):
            inputs_embeds = self.token_layernorm(embed_tokens(input_ids))
            previous_hidden_states = self.hidden_layernorm(previous_hidden_states)
            hidden = self.input_proj(
                mx.concatenate([previous_hidden_states, inputs_embeds], axis=-1)
            )
            mask = create_attention_mask(hidden, cache)
            hidden = self.mtp_block(hidden, mask=mask, cache=cache)
            return self.final_layernorm(hidden)

    class _MiMOMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [_MiMOMTPLayer() for _ in range(num_mtp_layers)]
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
            self.start_layer = start_layer
            self.num_mtp_layers = num_mtp_layers

    return _MiMOMTP()


def inject_mimo_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: Any | None = None,
) -> bool:
    """Attach MiMo native MTP support to a loaded mlx-lm model."""
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

    if not is_mimo_mtp_config(config):
        return False

    model_path = Path(model_path)
    tcfg = text_config(config)
    from mlx_lm.models import mimo

    args = getattr(model, "args", None)
    if args is None:
        args = mimo.ModelArgs.from_dict(tcfg)

    mtp = _make_mimo_mtp_module(config, args)
    raw_weights: dict[str, Any] = {}
    for file in _candidate_weight_files(model_path, config):
        raw_weights.update(_load_weight_file(file))

    mapped = _rewrite_mimo_mtp_weights(
        raw_weights,
        start_layer=int(getattr(args, "num_hidden_layers")),
        num_mtp_layers=_num_mtp_layers(config),
    )
    if not mapped or not _has_complete_mimo_mtp_payload(
        mapped, num_mtp_layers=_num_mtp_layers(config)
    ):
        logger.warning("[MiMo MTP inject] No MiMo MTP weights found in %s", model_path)
        return False

    _quantize_for_loaded_weights(mtp, config, mapped)
    mtp.load_weights(list(mapped.items()), strict=False)
    if "lm_head.weight" not in mapped:
        # MiMo shares the trunk output head with the MTP layer, matching
        # vLLM's mimo_mtp. _MiMOMTP still declares its own lm_head, which is
        # only ever filled by the trunk-shard path in _candidate_weight_files.
        # A forged artifact stores the head in mtp.safetensors, and that path
        # returns the sidecar alone, so lm_head stayed randomly initialised
        # and load_weights(strict=False) bound it without complaint. Every
        # contract candidate then scored zero agreement.
        trunk_head = getattr(model, "lm_head", None)
        if trunk_head is not None:
            mtp.lm_head = trunk_head
    mx.eval(mtp.parameters())

    original_outer_class = model.__class__

    class _MTPLXMiMoModel(original_outer_class):
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
                raise ValueError("MiMo MTP backend does not support input_embeddings")
            hidden = self.model(inputs, cache)
            logits = self.model.embed_tokens.as_linear(hidden) if self.args.tie_word_embeddings else self.lm_head(hidden)
            if not return_hidden:
                return logits
            return logits, hidden

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
            if concat_order not in {None, "hidden_embedding"}:
                raise ValueError("MiMo MTP backend supports hidden_embedding concat order only")
            if mtp_depth is not None and int(mtp_depth) > 1:
                raise ValueError("MiMo MTP backend currently supports mtp_depth=1 only")
            depth = 0
            layer_cache = None
            if mtp_cache is not None:
                layer_cache = mtp_cache[depth] if isinstance(mtp_cache, list) else mtp_cache
            hidden = self.mtp.layers[depth](
                next_token_ids,
                hidden_states,
                embed_tokens=self.model.embed_tokens,
                cache=layer_cache,
            )
            if self.args.tie_word_embeddings:
                logits = self.model.embed_tokens.as_linear(hidden)
            else:
                logits = self.mtp.lm_head(hidden)
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
            return [KVCache() for _ in self.mtp.layers]

        def make_cache(self):
            make_cache = getattr(super(), "make_cache", None)
            if callable(make_cache):
                return make_cache()
            return [KVCache() for _ in getattr(self.model, "layers", ())]

    model.mtp = mtp
    model.__class__ = _MTPLXMiMoModel
    logger.info("[MiMo MTP inject] Loaded %d tensors from %s", len(mapped), model_path)
    return True
