"""Runtime MTP injection for Qwen3.6/Qwen3.5 MLX models."""

from __future__ import annotations

import inspect
import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .artifacts import expected_mtp_file, is_mtp_key, normalize_mtp_key, text_config
from .constants import EXPECTED_ALL_PREQUANTIZED_MTP_KEYS, EXPECTED_PREQUANTIZED_MTP_KEYS

logger = logging.getLogger(__name__)

_RMSNORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
    "pre_fc_norm_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "norm.weight",
)


@dataclass(frozen=True)
class MTPContract:
    hidden_variant: str = "post_norm"
    concat_order: str = "embedding_hidden"
    mtp_quant_bits: int | None = None
    mtp_quant_group_size: int = 64
    mtp_quant_mode: str = "affine"
    mtp_quant_policy: str | None = None
    mtp_prequantized: bool = False

    def validate(self) -> None:
        if self.hidden_variant not in {"pre_norm", "post_norm"}:
            raise ValueError("hidden_variant must be 'pre_norm' or 'post_norm'")
        if self.concat_order not in {"embedding_hidden", "hidden_embedding"}:
            raise ValueError("concat_order must be 'embedding_hidden' or 'hidden_embedding'")
        if self.mtp_quant_bits is not None and self.mtp_quant_bits <= 0:
            raise ValueError("mtp_quant_bits must be positive when set")
        if self.mtp_quant_group_size <= 0:
            raise ValueError("mtp_quant_group_size must be positive")
        if self.mtp_quant_policy not in {None, "all", "cyankiwi"}:
            raise ValueError("mtp_quant_policy must be None, 'all', or 'cyankiwi'")

    def with_config_defaults(self, config: dict[str, Any]) -> "MTPContract":
        mtp_quant = config.get("mtplx_mtp_quantization")
        if not isinstance(mtp_quant, dict) or not mtp_quant:
            return self
        updates: dict[str, Any] = {}
        if self.mtp_quant_bits is None and mtp_quant.get("bits") is not None:
            updates["mtp_quant_bits"] = int(mtp_quant["bits"])
        if self.mtp_quant_group_size == 64 and mtp_quant.get("group_size") is not None:
            updates["mtp_quant_group_size"] = int(mtp_quant["group_size"])
        if self.mtp_quant_mode == "affine" and mtp_quant.get("mode") is not None:
            updates["mtp_quant_mode"] = str(mtp_quant["mode"])
        if self.mtp_quant_policy is None and mtp_quant.get("policy") is not None:
            updates["mtp_quant_policy"] = str(mtp_quant["policy"])
        if not self.mtp_prequantized and mtp_quant.get("prequantized") is not None:
            updates["mtp_prequantized"] = bool(mtp_quant["prequantized"])
        return replace(self, **updates) if updates else self


def _num_mtp_layers(config: dict[str, Any]) -> int:
    tcfg = text_config(config)
    return int(
        tcfg.get("mtp_num_hidden_layers")
        or tcfg.get("num_nextn_predict_layers")
        or config.get("num_nextn_predict_layers")
        or 0
    )


def _text_model(model: Any) -> Any:
    return getattr(model, "language_model", model)


def _quantize_mtp_module(mtp: Any, contract: MTPContract) -> None:
    import mlx.nn as nn

    if contract.mtp_quant_bits is None:
        return

    policy = contract.mtp_quant_policy or "all"
    if policy == "all":
        nn.quantize(
            mtp,
            group_size=contract.mtp_quant_group_size,
            bits=contract.mtp_quant_bits,
            mode=contract.mtp_quant_mode,
        )
        return

    if policy != "cyankiwi":
        raise ValueError(f"Unsupported MTP quantization policy: {policy}")

    def predicate(path: str, module: Any):
        if path == "fc" or path.startswith("pre_fc_norm") or path == "norm":
            return False
        if path.startswith("layers.") and hasattr(module, "to_quantized"):
            return {
                "group_size": contract.mtp_quant_group_size,
                "bits": contract.mtp_quant_bits,
                "mode": contract.mtp_quant_mode,
            }
        return False

    nn.quantize(mtp, class_predicate=predicate)


def _stack_mtp_moe_experts(
    weights: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Stack per-expert MTP MLP weights into SwitchGLU (``switch_mlp``) layout.

    Qwen3.5-MoE ships an MoE MTP head whose experts are stored one tensor per
    expert (``layers.{i}.mlp.experts.{e}.{proj}.{leaf}``), but mlx-lm's
    ``SparseMoeBlock`` expects a single stacked tensor per projection
    (``layers.{i}.mlp.switch_mlp.{proj}.{leaf}`` of shape ``[num_experts, ...]``).
    This mirrors the stacking mlx-lm performs for the main decoder layers.

    No-op for dense MTP heads (``num_experts <= 0`` or no per-expert keys), so
    the existing Qwen3-Next dense path is unaffected.
    """
    tcfg = text_config(config)
    num_experts = int(tcfg.get("num_experts") or 0)
    if num_experts <= 0:
        return weights

    import mlx.core as mx

    n_layers = max(_num_mtp_layers(config), 1)
    for li in range(n_layers):
        prefix = f"layers.{li}.mlp"
        if f"{prefix}.experts.0.gate_proj.weight" not in weights:
            continue
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for leaf in ("weight", "scales", "biases"):
                if f"{prefix}.experts.0.{proj}.{leaf}" not in weights:
                    continue
                stacked = [
                    weights.pop(f"{prefix}.experts.{e}.{proj}.{leaf}")
                    for e in range(num_experts)
                ]
                weights[f"{prefix}.switch_mlp.{proj}.{leaf}"] = mx.stack(stacked, axis=0)
    return weights


def _finalize_mtp_weights(
    raw_mtp: dict[str, Any],
    config: dict[str, Any],
    *,
    prequantized: bool = False,
) -> dict[str, Any]:
    try:
        import mlx.core as mx
    except Exception:
        mx = None

    tcfg = text_config(config)
    quant_config = tcfg.get("quantization") or tcfg.get("quantization_config") or {}
    if not quant_config:
        quant_config = config.get("quantization") or config.get("quantization_config") or {}
    bits = int(quant_config.get("bits", 4)) if quant_config else 4
    group_size = int(quant_config.get("group_size", 64)) if quant_config else 64

    if prequantized:
        weights = dict(raw_mtp)
        for key, value in list(weights.items()):
            if value.ndim == 1 and any(key.endswith(suffix) for suffix in _RMSNORM_SUFFIXES):
                if float(value.mean().item()) < 0.5:
                    weights[key] = value + 1.0
        return _stack_mtp_moe_experts(weights, config)

    weights: dict[str, Any] = {}
    processed: set[str] = set()
    for key in sorted(raw_mtp):
        if key in processed or key.endswith((".scales", ".biases")):
            continue
        scales_key = key.replace(".weight", ".scales")
        biases_key = key.replace(".weight", ".biases")
        if scales_key != key and scales_key in raw_mtp and biases_key in raw_mtp:
            if mx is None:
                raise RuntimeError("MLX is required to dequantize embedded MTP weights")
            weights[key] = mx.dequantize(
                raw_mtp[key],
                raw_mtp[scales_key],
                raw_mtp[biases_key],
                group_size=group_size,
                bits=bits,
            )
            processed.update({key, scales_key, biases_key})
        else:
            weights[key] = raw_mtp[key]
            processed.add(key)

    for key, value in list(weights.items()):
        if value.ndim == 1 and any(key.endswith(suffix) for suffix in _RMSNORM_SUFFIXES):
            if float(value.mean().item()) < 0.5:
                weights[key] = value + 1.0
    return _stack_mtp_moe_experts(weights, config)


def _strip_mtp_namespace(key: str) -> str:
    return normalize_mtp_key(key).removeprefix("mtp.")


def _mtp_contract_for_weight_keys(
    contract: MTPContract,
    keys: tuple[str, ...],
    config: dict[str, Any],
) -> MTPContract:
    normalized = {normalize_mtp_key(key) for key in keys}
    if contract.mtp_prequantized:
        return contract
    if normalized == set(EXPECTED_ALL_PREQUANTIZED_MTP_KEYS):
        policy = "all"
    elif normalized == set(EXPECTED_PREQUANTIZED_MTP_KEYS):
        policy = "cyankiwi"
    else:
        return contract

    tcfg = text_config(config)
    quant_config = tcfg.get("quantization") or tcfg.get("quantization_config") or {}
    if not quant_config:
        quant_config = config.get("quantization") or config.get("quantization_config") or {}
    updates: dict[str, Any] = {
        "mtp_prequantized": True,
        "mtp_quant_policy": contract.mtp_quant_policy or policy,
    }
    if contract.mtp_quant_bits is None:
        updates["mtp_quant_bits"] = int((quant_config or {}).get("bits", 4))
    if contract.mtp_quant_group_size == 64:
        updates["mtp_quant_group_size"] = int((quant_config or {}).get("group_size", 64))
    if contract.mtp_quant_mode == "affine":
        updates["mtp_quant_mode"] = str((quant_config or {}).get("mode", "affine"))
    return replace(contract, **updates)


def _load_mtp_weights(
    mtp_file: Path,
    config: dict[str, Any],
    *,
    prequantized: bool = False,
) -> dict[str, Any]:
    import mlx.core as mx

    raw = mx.load(str(mtp_file))
    raw_mtp = {_strip_mtp_namespace(k): v for k, v in raw.items() if is_mtp_key(k)}
    del raw
    return _finalize_mtp_weights(raw_mtp, config, prequantized=prequantized)


def _safetensors_runtime_framework() -> str:
    try:
        import mlx.core  # noqa: F401 - registers mlx.core for safetensors.

        return "mlx"
    except Exception:
        return "np"


def _mtp_file_keys(mtp_file: Path) -> tuple[str, ...]:
    try:
        from safetensors import safe_open

        with safe_open(str(mtp_file), framework=_safetensors_runtime_framework()) as handle:
            return tuple(sorted(str(key) for key in handle.keys()))
    except Exception:
        try:
            import mlx.core as mx

            raw = mx.load(str(mtp_file))
            keys = tuple(sorted(str(key) for key in raw.keys()))
            del raw
            return keys
        except Exception:
            return ()


def _embedded_mtp_weight_map(model_path: Path) -> dict[str, Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map", {})
        except Exception:
            weight_map = {}
        return {
            str(key): model_path / str(rel)
            for key, rel in weight_map.items()
            if is_mtp_key(str(key))
        }

    from safetensors import safe_open

    result: dict[str, Path] = {}
    framework = _safetensors_runtime_framework()
    for shard in sorted(model_path.glob("model*.safetensors")):
        with safe_open(str(shard), framework=framework) as handle:
            for key in handle.offset_keys():
                if is_mtp_key(str(key)):
                    result[str(key)] = shard
    return result


def _load_embedded_mtp_weights(
    model_path: Path,
    config: dict[str, Any],
    *,
    prequantized: bool = False,
) -> dict[str, Any]:
    key_to_file = _embedded_mtp_weight_map(model_path)
    if not key_to_file:
        return {}

    raw_mtp: dict[str, Any] = {}
    files: dict[Path, list[str]] = {}
    for key, shard in key_to_file.items():
        files.setdefault(shard, []).append(key)

    for shard, keys in files.items():
        try:
            import mlx.core as mx

            shard_tensors = mx.load(str(shard))
            for key in sorted(keys):
                raw_mtp[_strip_mtp_namespace(key)] = shard_tensors[key]
            del shard_tensors
        except Exception:
            from safetensors import safe_open

            with safe_open(str(shard), framework=_safetensors_runtime_framework()) as handle:
                for key in sorted(keys):
                    raw_mtp[_strip_mtp_namespace(key)] = handle.get_tensor(key)

    return _finalize_mtp_weights(raw_mtp, config, prequantized=prequantized)


def inject_mtp_support(
    model: Any,
    model_path: Path | str,
    config: dict[str, Any],
    contract: MTPContract | None = None,
) -> bool:
    """Attach Qwen native MTP support to a loaded mlx-lm model instance."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask, scaled_dot_product_attention
    from mlx_lm.models.cache import KVCache
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

    contract = contract or MTPContract()
    contract.validate()
    n_layers = _num_mtp_layers(config)
    if n_layers <= 0:
        logger.info("[MTP inject] Model config has no MTP layers")
        return False

    model_path = Path(model_path)
    mtp_file = expected_mtp_file(model_path, config)
    mtp_weights: dict[str, Any] | None = None
    mtp_source = mtp_file
    if mtp_file.exists():
        contract = _mtp_contract_for_weight_keys(contract, _mtp_file_keys(mtp_file), config)
        mtp_weights = _load_mtp_weights(
            mtp_file,
            config,
            prequantized=contract.mtp_prequantized,
        )
    else:
        embedded_weight_map = _embedded_mtp_weight_map(model_path)
        contract = _mtp_contract_for_weight_keys(
            contract,
            tuple(embedded_weight_map),
            config,
        )
        mtp_weights = _load_embedded_mtp_weights(
            model_path,
            config,
            prequantized=contract.mtp_prequantized,
        )
        mtp_source = model_path / "model*.safetensors::embedded-mtp"
        if not mtp_weights:
            logger.warning("[MTP inject] MTP weights not found: %s", mtp_file)
            return False

    tcfg = text_config(config)
    text_model = _text_model(model)
    args = getattr(text_model, "args", None)
    if not isinstance(args, TextModelArgs):
        args = TextModelArgs.from_dict(tcfg)

    fa_idx = args.full_attention_interval - 1

    class _MTPModule(nn.Module):
        def __init__(self, args: TextModelArgs, n_layers: int):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.pre_fc_norm_embedding = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [DecoderLayer(args, layer_idx=fa_idx) for _ in range(n_layers)]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    mtp = _MTPModule(args, n_layers)
    if contract.mtp_prequantized:
        _quantize_mtp_module(mtp, contract)
    mtp.load_weights(list(mtp_weights.items()), strict=False)
    if not contract.mtp_prequantized:
        _quantize_mtp_module(mtp, contract)
    mx.eval(mtp.parameters())

    text_model.mtp = mtp
    text_model._mtplx_hidden_variant = contract.hidden_variant
    text_model._mtplx_concat_order = contract.concat_order
    text_model._mtplx_mtp_quant_policy = contract.mtp_quant_policy

    original_text_class = text_model.__class__

    class _MTPLXTextModel(original_text_class):
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
            inner = self.model
            hidden_states = input_embeddings if input_embeddings is not None else inner.embed_tokens(inputs)
            if cache is None:
                cache = [None] * len(inner.layers)

            fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
            ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
            for layer, layer_cache in zip(inner.layers, cache):
                mask = ssm_mask if layer.is_linear else fa_mask
                hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)

            pre_norm = hidden_states
            variant = hidden_variant or getattr(self, "_mtplx_hidden_variant", "post_norm")
            needs_post_norm = emit_logits or (return_hidden and variant != "pre_norm")
            post_norm = inner.norm(hidden_states) if needs_post_norm else None
            logits = None
            if emit_logits:
                logits_source = post_norm
                if logits_keep is not None:
                    keep = max(1, int(logits_keep))
                    logits_source = logits_source[:, -keep:, :]
                logits = (
                    inner.embed_tokens.as_linear(logits_source)
                    if self.args.tie_word_embeddings
                    else self.lm_head(logits_source)
                )
            if not return_hidden:
                return logits
            hidden = pre_norm if variant == "pre_norm" else post_norm
            return logits, hidden

        def _mixed_hidden(self, variant: str, *, previous, fc_hidden, pre_norm, post_norm, input_embeds):
            aliases = {
                "fc": fc_hidden,
                "pre_norm": pre_norm,
                "post_norm": post_norm,
                "embedding": input_embeds,
                "prev": previous,
            }
            if variant in aliases:
                return aliases[variant]

            # Experimental hidden repair syntax:
            #   mix:<left>:<right>:<alpha>
            # returns alpha * left + (1 - alpha) * right.
            # Alpha accepts decimal points or "p" as the decimal separator,
            # e.g. mix:pre_norm:prev:0p75.
            if variant.startswith("mix:"):
                parts = variant.split(":")
                if len(parts) != 4:
                    raise ValueError("mix variant must be mix:<left>:<right>:<alpha>")
                left_name, right_name, alpha_raw = parts[1], parts[2], parts[3]
                if left_name not in aliases or right_name not in aliases:
                    raise ValueError(
                        "mix variant sources must be one of "
                        "'fc', 'pre_norm', 'post_norm', 'embedding', or 'prev'"
                    )
                alpha = float(alpha_raw.replace("p", "."))
                if not 0.0 <= alpha <= 1.0:
                    raise ValueError("mix variant alpha must be in [0, 1]")
                return aliases[left_name] * alpha + aliases[right_name] * (1.0 - alpha)

            raise ValueError(
                "mtp_hidden_variant must be 'fc', 'pre_norm', 'post_norm', "
                "'embedding', 'prev', or mix:<left>:<right>:<alpha>"
            )

        def _mtp_full_attention_layer(self, layer, x, *, mask=None, cache=None, position_offset: int | None = None):
            if position_offset is None:
                return layer(x, mask=mask, cache=cache)
            if layer.is_linear:
                raise ValueError("explicit MTP position offsets require a full-attention MTP layer")

            attn = layer.self_attn
            normed = layer.input_layernorm(x)
            B, L, _ = normed.shape

            q_proj_output = attn.q_proj(normed)
            queries, gate = mx.split(
                q_proj_output.reshape(B, L, attn.num_attention_heads, -1),
                2,
                axis=-1,
            )
            gate = gate.reshape(B, L, -1)

            keys, values = attn.k_proj(normed), attn.v_proj(normed)
            queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
            keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(
                0,
                2,
                1,
                3,
            )
            values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(
                0,
                2,
                1,
                3,
            )

            queries = attn.rope(queries, offset=int(position_offset))
            keys = attn.rope(keys, offset=int(position_offset))
            paged_mtp_enabled = (
                os.environ.get("MTPLX_VLLM_METAL_PAGED_MTP_ATTN", "")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"}
            )
            use_paged_mtp = bool(
                paged_mtp_enabled
                and cache is not None
                and int(L) == 1
                and hasattr(cache, "update_without_fetch")
                and hasattr(cache, "paged_attention")
            )
            if use_paged_mtp:
                # The paged primitive is causal-safe only for single-token MTP
                # draft/update calls. Multi-token committed-history appends keep
                # the stock SDPA path so each query cannot see future keys from
                # the same append chunk.
                cache.update_without_fetch(keys, values)
                output = cache.paged_attention(queries, scale=attn.scale)
                if output is None:
                    keys, values = cache.state
                    output = scaled_dot_product_attention(
                        queries,
                        keys,
                        values,
                        cache=cache,
                        scale=attn.scale,
                        mask=mask,
                    )
            else:
                if cache is not None:
                    keys, values = cache.update_and_fetch(keys, values)
                output = scaled_dot_product_attention(
                    queries,
                    keys,
                    values,
                    cache=cache,
                    scale=attn.scale,
                    mask=mask,
                )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            h = x + attn.o_proj(output * mx.sigmoid(gate))
            return h + layer.mlp(layer.post_attention_layernorm(h))

        def _mtp_core(
            self,
            hidden_states,
            next_token_ids,
            mtp_cache=None,
            concat_order=None,
            mtp_hidden_variant: str = "post_norm",
            position_offset: int | None = None,
            emit_logits: bool = True,
        ):
            input_embeds = self.model.embed_tokens(next_token_ids)
            e = self.mtp.pre_fc_norm_embedding(input_embeds)
            h = self.mtp.pre_fc_norm_hidden(hidden_states)
            order = concat_order or getattr(self, "_mtplx_concat_order", "embedding_hidden")
            parts = [e, h] if order == "embedding_hidden" else [h, e]
            x = self.mtp.fc(mx.concatenate(parts, axis=-1))
            fc_hidden = x
            layer_cache = mtp_cache[0] if mtp_cache else None
            mask = create_attention_mask(x, layer_cache)
            x = self._mtp_full_attention_layer(
                self.mtp.layers[0],
                x,
                mask=mask,
                cache=layer_cache,
                position_offset=position_offset,
            )
            pre_norm = x
            post_norm = self.mtp.norm(x)
            hidden = self._mixed_hidden(
                mtp_hidden_variant,
                previous=hidden_states,
                fc_hidden=fc_hidden,
                pre_norm=pre_norm,
                post_norm=post_norm,
                input_embeds=input_embeds,
            )
            if not emit_logits:
                return None, hidden
            draft_lm_head = getattr(self, "_mtplx_draft_lm_head", None)
            logits = (
                draft_lm_head(post_norm)
                if draft_lm_head is not None
                else (
                    self.model.embed_tokens.as_linear(post_norm)
                    if self.args.tie_word_embeddings
                    else self.lm_head(post_norm)
                )
            )
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
        ):
            logits, hidden = self._mtp_core(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                mtp_hidden_variant=mtp_hidden_variant,
                position_offset=position_offset,
                emit_logits=True,
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
        ):
            _logits, hidden = self._mtp_core(
                hidden_states,
                next_token_ids,
                mtp_cache=mtp_cache,
                concat_order=concat_order,
                mtp_hidden_variant="post_norm",
                position_offset=position_offset,
                emit_logits=False,
            )
            return hidden

        def make_mtp_cache(self):
            return [KVCache() for _ in self.mtp.layers]

    text_model.__class__ = _MTPLXTextModel

    if hasattr(model, "language_model") and model.language_model is text_model:
        model.mtp = mtp
        original_outer_class = model.__class__

        class _MTPLXOuterModel(original_outer_class):
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
                return self.language_model(
                    inputs,
                    cache=cache,
                    return_hidden=return_hidden,
                    input_embeddings=input_embeddings,
                    hidden_variant=hidden_variant,
                    emit_logits=emit_logits,
                    logits_keep=logits_keep,
                    **kwargs,
                )

            def mtp_forward(self, *args, **kwargs):
                return self.language_model.mtp_forward(*args, **kwargs)

            def mtp_update_cache(self, *args, **kwargs):
                return self.language_model.mtp_update_cache(*args, **kwargs)

            def make_mtp_cache(self):
                return self.language_model.make_mtp_cache()

        model.__class__ = _MTPLXOuterModel

    logger.info("[MTP inject] Loaded %d tensors from %s", len(mtp_weights), mtp_source)
    return True


def validate_mtp_support(model: Any) -> bool:
    text_model = _text_model(model)
    if getattr(text_model, "mtp", None) is None:
        return False
    if not getattr(text_model.mtp, "layers", None):
        return False
    try:
        call_sig = inspect.signature(type(text_model).__call__)
    except Exception:
        return False
    return (
        "return_hidden" in call_sig.parameters
        and callable(getattr(text_model, "mtp_forward", None))
        and callable(getattr(text_model, "make_mtp_cache", None))
    )
