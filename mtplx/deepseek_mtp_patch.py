"""Runtime MTP injection for DeepSeek V3/V3.2 MLX models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .artifacts import expected_mtp_file, text_config

logger = logging.getLogger(__name__)

DEEPSEEK_MTP_MODEL_TYPES = {
    "deepseek_v3",
    "deepseek_v32",
    "glm_moe_dsa",
}


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


def is_deepseek_mtp_config(config: dict[str, Any]) -> bool:
    return _model_type(config) in DEEPSEEK_MTP_MODEL_TYPES and _num_mtp_layers(config) > 0


def _deepseek_impl(config: dict[str, Any]):
    model_type = _model_type(config)
    if model_type == "glm_moe_dsa":
        from mlx_lm.models import deepseek_v32
        from mlx_lm.models import glm_moe_dsa as impl
        from mlx_lm.models.cache import CacheList, KVCache

        return {
            "module": impl,
            "args_cls": impl.ModelArgs,
            "layer_cls": deepseek_v32.DeepseekV32DecoderLayer,
            "cache_factory": lambda: CacheList(KVCache(), KVCache()),
            "cache_for_mask": lambda cache: cache[0] if cache is not None else None,
        }

    if model_type == "deepseek_v32":
        from mlx_lm.models import deepseek_v32 as impl
        from mlx_lm.models.cache import CacheList, KVCache

        return {
            "module": impl,
            "args_cls": impl.ModelArgs,
            "layer_cls": impl.DeepseekV32DecoderLayer,
            "cache_factory": lambda: CacheList(KVCache(), KVCache()),
            "cache_for_mask": lambda cache: cache[0] if cache is not None else None,
        }

    from mlx_lm.models import deepseek_v3 as impl
    from mlx_lm.models.cache import KVCache

    return {
        "module": impl,
        "args_cls": impl.ModelArgs,
        "layer_cls": impl.DeepseekV3DecoderLayer,
        "cache_factory": KVCache,
        "cache_for_mask": lambda cache: cache,
    }


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
        wanted_prefixes = tuple(f"model.layers.{start + i}." for i in range(count))
        selected = {
            model_path / rel
            for key, rel in weight_map.items()
            if str(key).startswith(wanted_prefixes)
        }
        if selected:
            return sorted(selected)

    return sorted(model_path.glob("model*.safetensors"))


def _dense_weight_dims(weight: Any) -> tuple[int, int]:
    out_dims = int(weight.shape[0])
    in_dims = int(weight.shape[1])
    return out_dims, in_dims


def _rewrite_kv_b_projection(weights: dict[str, Any], prefix: str, args: Any) -> None:
    import mlx.core as mx

    weight_key = f"{prefix}.self_attn.kv_b_proj.weight"
    if weight_key not in weights:
        return

    quantized = f"{prefix}.self_attn.kv_b_proj.scales" in weights
    v = weights.pop(weight_key)
    head_dim = int(args.qk_nope_head_dim) + int(args.v_head_dim)

    bits = None
    group_size = None
    if quantized:
        dims = int(args.kv_lora_rank)
        scales = weights.pop(f"{prefix}.self_attn.kv_b_proj.scales")
        biases = weights.pop(f"{prefix}.self_attn.kv_b_proj.biases")
        bits = (int(v.shape[-1]) * 32) // dims
        group_size = dims // int(scales.shape[-1])
        v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

    num_heads = int(args.num_attention_heads)
    v = v.reshape(num_heads, head_dim, -1)
    wk = mx.contiguous(v[:, : int(args.qk_nope_head_dim), :].swapaxes(-1, -2))
    wv = mx.contiguous(v[:, int(args.qk_nope_head_dim) :, :])

    if quantized:
        wk, wk_scales, wk_biases = mx.quantize(wk, bits=bits, group_size=group_size)
        wv, wv_scales, wv_biases = mx.quantize(wv, bits=bits, group_size=group_size)
        weights[f"{prefix}.self_attn.embed_q.scales"] = wk_scales
        weights[f"{prefix}.self_attn.embed_q.biases"] = wk_biases
        weights[f"{prefix}.self_attn.unembed_out.scales"] = wv_scales
        weights[f"{prefix}.self_attn.unembed_out.biases"] = wv_biases

    weights[f"{prefix}.self_attn.embed_q.weight"] = wk
    weights[f"{prefix}.self_attn.unembed_out.weight"] = wv


def _stack_moe_experts(weights: dict[str, Any], prefix: str, args: Any) -> None:
    import mlx.core as mx

    n_routed = int(getattr(args, "n_routed_experts", 0) or 0)
    if n_routed <= 0:
        return
    for module in ("gate_proj", "down_proj", "up_proj"):
        for leaf in ("weight", "scales", "biases"):
            first = f"{prefix}.mlp.experts.0.{module}.{leaf}"
            if first not in weights:
                continue
            values = [
                weights.pop(f"{prefix}.mlp.experts.{idx}.{module}.{leaf}")
                for idx in range(n_routed)
            ]
            weights[f"{prefix}.mlp.switch_mlp.{module}.{leaf}"] = mx.stack(values)


def _has_complete_deepseek_mtp_payload(
    weights: dict[str, Any],
    *,
    num_mtp_layers: int,
) -> bool:
    """Return true only when every declared DeepSeek MTP layer has real weights.

    A prefix-mismatched checkpoint still yields a non-empty ``mapped`` from
    stray keys, and ``strict=False`` then loads nothing -- injection would
    report success while the draft head stays randomly initialized.
    """

    for local_idx in range(num_mtp_layers):
        prefix = f"layers.{local_idx}."
        required = (
            f"{prefix}enorm.weight",
            f"{prefix}hnorm.weight",
            f"{prefix}eh_proj.weight",
        )
        if not all(key in weights for key in required):
            return False
        if not any(key.startswith(f"{prefix}mtp_block.") for key in weights):
            return False
    return True


def _rewrite_deepseek_mtp_weights(
    raw: dict[str, Any],
    *,
    args: Any,
    start_layer: int,
    num_mtp_layers: int,
) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("mtp."):
            mapped[key.removeprefix("mtp.")] = value
            continue
        if key.startswith("layers."):
            mapped[key] = value
            continue
        for local_idx in range(num_mtp_layers):
            spec_idx = start_layer + local_idx
            prefix = f"model.layers.{spec_idx}."
            if not key.startswith(prefix):
                continue
            suffix = key.removeprefix(prefix)
            local_prefix = f"layers.{local_idx}"
            if suffix.startswith("shared_head.norm."):
                mapped[f"{local_prefix}.shared_head_norm.{suffix.removeprefix('shared_head.norm.')}"] = value
            elif suffix.startswith("shared_head.head."):
                mapped[f"{local_prefix}.shared_head_head.{suffix.removeprefix('shared_head.head.')}"] = value
            elif suffix.startswith(("enorm.", "hnorm.", "eh_proj.")):
                mapped[f"{local_prefix}.{suffix}"] = value
            elif suffix.startswith("embed_tokens."):
                # DeepSeek MTP shares the target embedding; the runtime reuses it.
                pass
            else:
                mapped[f"{local_prefix}.mtp_block.{suffix}"] = value
            break

    for local_idx in range(num_mtp_layers):
        block_prefix = f"layers.{local_idx}.mtp_block"
        _rewrite_kv_b_projection(mapped, block_prefix, args)
        _stack_moe_experts(mapped, block_prefix, args)

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


def _make_deepseek_mtp_module(config: dict[str, Any], args: Any):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask

    impl = _deepseek_impl(config)
    layer_cls = impl["layer_cls"]
    cache_for_mask = impl["cache_for_mask"]
    start_layer = int(getattr(args, "num_hidden_layers"))
    num_mtp_layers = _num_mtp_layers(config)

    class _DeepSeekMTPLayer(nn.Module):
        def __init__(self, layer_idx: int):
            super().__init__()
            self.enorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.hnorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.eh_proj = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.mtp_block = layer_cls(args, layer_idx=layer_idx)
            self.shared_head_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.shared_head_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        def __call__(self, input_ids, previous_hidden_states, *, embed_tokens, cache=None):
            inputs_embeds = embed_tokens(input_ids)
            mixed = self.eh_proj(
                mx.concatenate(
                    [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)],
                    axis=-1,
                )
            )
            mask_cache = cache_for_mask(cache)
            mask = create_attention_mask(mixed, mask_cache, return_array=True)
            hidden = self.mtp_block(mixed, mask=mask, cache=cache)
            logits = self.shared_head_head(self.shared_head_norm(hidden))
            return logits, hidden

    class _DeepSeekMTP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [
                _DeepSeekMTPLayer(start_layer + idx)
                for idx in range(num_mtp_layers)
            ]
            self.start_layer = start_layer
            self.num_mtp_layers = num_mtp_layers

    return _DeepSeekMTP()


def inject_deepseek_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: Any | None = None,
) -> bool:
    """Attach DeepSeek V3/V3.2 native MTP support to a loaded mlx-lm model."""
    import mlx.core as mx

    if not is_deepseek_mtp_config(config):
        return False

    model_path = Path(model_path)
    tcfg = text_config(config)
    impl = _deepseek_impl(config)
    args = getattr(model, "args", None)
    if args is None:
        args = impl["args_cls"].from_dict(tcfg)

    mtp = _make_deepseek_mtp_module(config, args)
    raw_weights: dict[str, Any] = {}
    for file in _candidate_weight_files(model_path, config):
        raw_weights.update(_load_weight_file(file))

    mapped = _rewrite_deepseek_mtp_weights(
        raw_weights,
        args=args,
        start_layer=int(getattr(args, "num_hidden_layers")),
        num_mtp_layers=_num_mtp_layers(config),
    )
    if not mapped or not _has_complete_deepseek_mtp_payload(
        mapped, num_mtp_layers=_num_mtp_layers(config)
    ):
        logger.warning("[DeepSeek MTP inject] No DeepSeek MTP weights found in %s", model_path)
        return False

    _quantize_for_loaded_weights(mtp, config, mapped)
    mtp.load_weights(list(mapped.items()), strict=False)
    mx.eval(mtp.parameters())

    cache_factory = impl["cache_factory"]
    original_outer_class = model.__class__

    class _MTPLXDeepSeekModel(original_outer_class):
        def __call__(
            self,
            inputs,
            cache=None,
            return_hidden: bool = False,
            input_embeddings=None,
            hidden_variant: str | None = None,
            emit_logits: bool = True,
            logits_keep: int | None = None,
            **kwargs,
        ):
            if input_embeddings is not None:
                raise ValueError("DeepSeek MTP backend does not support input_embeddings")
            hidden = self.model(inputs, cache)
            logits = None
            if emit_logits:
                logits_source = hidden
                if logits_keep is not None:
                    logits_source = logits_source[:, -max(1, int(logits_keep)):, :]
                logits = self.lm_head(logits_source)
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
            if concat_order not in {None, "embedding_hidden"}:
                raise ValueError("DeepSeek MTP backend supports embedding_hidden concat order only")
            depth = 0 if mtp_depth is None else max(int(mtp_depth) - 1, 0)
            depth %= len(self.mtp.layers)
            layer_cache = None
            if mtp_cache is not None:
                layer_cache = mtp_cache[depth] if isinstance(mtp_cache, list) else mtp_cache
            logits, hidden = self.mtp.layers[depth](
                next_token_ids,
                hidden_states,
                embed_tokens=self.model.embed_tokens,
                cache=layer_cache,
            )
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
            return [cache_factory() for _ in self.mtp.layers]

        def make_cache(self):
            make_cache = getattr(super(), "make_cache", None)
            if callable(make_cache):
                return make_cache()
            return [cache_factory() for _ in self.layers]

    model.mtp = mtp
    model.__class__ = _MTPLXDeepSeekModel
    logger.info("[DeepSeek MTP inject] Loaded %d tensors from %s", len(mapped), model_path)
    return True
