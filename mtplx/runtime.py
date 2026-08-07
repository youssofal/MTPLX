"""High-level MTPLX runtime loading primitives."""

from __future__ import annotations

import hashlib
import inspect as py_inspect
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifacts import (
    inspect_model,
    load_config,
    mtp_weights_present_on_disk,
    text_config,
)
from .mtp_adapters import (
    install_saved_mtp_lora_adapter,
    merge_installed_mtp_lora_adapters,
    mtp_adapter_depth,
)
from .mtp_patch import MTPContract, inject_mtp_support, validate_mtp_support

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .a3b_compiled_target_prefix import A3BCompiledTargetPrefixFactory


def _detect_total_system_memory_bytes() -> int | None:
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            total = int(
                subprocess.check_output(
                    ["sysctl", "-n", "hw.memsize"],
                    text=True,
                ).strip()
            )
            if total > 0:
                return total
        except Exception:
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
        total = page_size * pages
        return total if total > 0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _preflight_laguna_system_memory(config: dict[str, Any]) -> None:
    if not _is_laguna_s_2_1_mlx_4bit_config(config):
        return
    from .models.laguna_config import LAGUNA_S_2_1_MIN_RESIDENT_BYTES

    system_reserve = 16 * 1024**3
    total = _detect_total_system_memory_bytes()
    if total is None or total >= LAGUNA_S_2_1_MIN_RESIDENT_BYTES + system_reserve:
        return
    required = LAGUNA_S_2_1_MIN_RESIDENT_BYTES + system_reserve
    raise RuntimeError(
        "Laguna-S-2.1 requires at least "
        f"{required / 1024**3:.1f} GiB unified memory "
        "for weights, runtime headroom, and the system reserve"
    )


@dataclass
class MTPLXRuntime:
    model: Any
    tokenizer: Any
    model_path: Path
    mtp_enabled: bool
    contract: MTPContract
    mtp_adapter_path: Path | None = None
    mtp_adapter_metadata: dict[str, Any] | None = None
    mtp_adapter_merge_report: dict[str, Any] | None = None
    deepseek_v4_o_lora_report: dict[str, Any] | None = None
    deepseek_v4_attn_proj_wide_m3_report: dict[str, Any] | None = None
    deepseek_v4_attention_island_report: dict[str, Any] | None = None
    a3b_compiled_target_prefix_factory: A3BCompiledTargetPrefixFactory | None = None
    a3b_whole_moe_installed: bool = False
    _a3b_whole_moe_request_preflights: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _a3b_whole_moe_request_geometry_keys: dict[
        tuple[int, str, str], str
    ] = field(default_factory=dict, init=False, repr=False)
    diagnostic_counters: dict[str, int] = field(default_factory=dict)
    _forward_ar_supports_emit_logits: bool | None = field(default=None, init=False, repr=False)
    _forward_ar_supports_logits_keep: bool | None = field(default=None, init=False, repr=False)

    def _count(self, key: str, amount: int = 1) -> None:
        self.diagnostic_counters[key] = int(self.diagnostic_counters.get(key, 0)) + int(amount)

    @staticmethod
    def _sequence_len(input_ids: Any) -> int:
        shape = getattr(input_ids, "shape", ())
        if len(shape) >= 2:
            return int(shape[1])
        if shape:
            return int(shape[0])
        return 1

    def _forward_ar_capabilities(self) -> tuple[bool, bool]:
        if (
            self._forward_ar_supports_emit_logits is None
            or self._forward_ar_supports_logits_keep is None
        ):
            try:
                params = py_inspect.signature(self.model.__call__).parameters
            except Exception:
                params = {}
            accepts_kwargs = any(
                param.kind == py_inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            patched_kwargs = bool(self.mtp_enabled and accepts_kwargs)
            self._forward_ar_supports_emit_logits = (
                "emit_logits" in params or patched_kwargs
            )
            self._forward_ar_supports_logits_keep = (
                "logits_keep" in params or patched_kwargs
            )
        return (
            bool(self._forward_ar_supports_emit_logits),
            bool(self._forward_ar_supports_logits_keep),
        )

    def embed_tokens(self, input_ids):
        """Embed token ids with the text model's embedding table."""

        text_model = getattr(self.model, "language_model", self.model)
        return text_model.model.embed_tokens(input_ids)

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        self._count("forward_ar_hidden_calls" if return_hidden else "forward_ar_plain_calls")
        if not self.mtp_enabled and return_hidden:
            raise RuntimeError("return_hidden requires an MTP-patched runtime")
        if input_embeddings is not None and not self.mtp_enabled:
            raise RuntimeError("vision splice requires the MTP-patched runtime")
        kwargs = {}
        if hidden_variant is not None:
            kwargs["hidden_variant"] = hidden_variant
        if input_embeddings is not None:
            # Vision splice path: the patched text model takes the rows
            # directly; ids still travel for mask construction.
            kwargs["input_embeddings"] = input_embeddings
        supports_emit_logits, supports_logits_keep = self._forward_ar_capabilities()
        if supports_emit_logits:
            kwargs["emit_logits"] = bool(emit_logits)
        elif not emit_logits:
            self._count("forward_ar_emit_logits_unsupported")
        if logits_keep is not None and supports_logits_keep:
            kwargs["logits_keep"] = int(logits_keep)
        elif logits_keep is not None:
            self._count("forward_ar_logits_keep_unsupported")
        sequence_len = self._sequence_len(input_ids)
        if bool(emit_logits) or not supports_emit_logits:
            if logits_keep is not None and supports_logits_keep:
                emitted = min(sequence_len, max(1, int(logits_keep)))
            else:
                emitted = sequence_len
            self._count("logits_tokens_emitted", emitted)
            if emitted == 1:
                self._count("final_logits_tokens_emitted", 1)
            else:
                self._count("full_logits_tokens_emitted", emitted)
        # kwargs == {"emit_logits": True} is semantically the plain call —
        # MTP-patched wrappers advertise emit_logits via **kwargs, so on MTP
        # runtimes the bare-kwargs case never occurs and the compiled hook
        # must accept the default-emit form too.
        plain_call = not kwargs or (
            set(kwargs) == {"emit_logits"} and kwargs["emit_logits"] is True
        )
        if not return_hidden and hidden_variant is None and plain_call:
            # Decode-only (seq_len == 1). Prefill is multi-token over an
            # unprimed cache: seeding the compiled graph from its None KV
            # leaves throws, and its shape differs from a single-token decode
            # step, forcing a retrace. Prefill stays eager.
            compiled = (
                self._compiled_ar_forward(cache) if sequence_len == 1 else None
            )
            if compiled is not None:
                # Engagement proof: arm A (flag off) must report 0 here,
                # arm B (on) > 0 — the A/B credits nothing without it.
                self._count("compiled_forward_calls")
                return compiled(input_ids, cache)
            if not kwargs:
                return self.model(input_ids, cache=cache)
        return self.model(
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            **kwargs,
        )

    def _compiled_ar_forward(self, cache):
        """Compiled target forward (MTPLX_COMPILE_AR_FORWARD).

        Kills the per-token Python graph rebuild by tracing the full trunk
        forward once (CompiledARForward, KV state threaded). Applies to
        fully-resident loads with a standard per-layer KV cache; a host-sync
        buried in the model forward surfaces as an error on the first traced
        call rather than silently degrading. Rebuilds per cache identity so a
        new generation gets fresh threaded state. Returns None (the eager
        path) otherwise.
        """
        from .compiled_forward import CompiledARForward, compile_forward_enabled

        if not compile_forward_enabled() or not cache:
            return None
        # An unprimed cache (empty context / first token) has None KV leaves
        # that would crash the compiled graph. Only compile once the cache
        # holds real keys, and only for the plain growable KVCache shape the
        # fixed-buffer conversion understands.
        first = cache[0]
        if getattr(first, "keys", None) is None:
            return None
        if any(
            not hasattr(entry, "keys")
            or not hasattr(entry, "values")
            or not hasattr(entry, "offset")
            for entry in cache
        ):
            return None
        cache_key = id(first)
        if (
            getattr(self, "_compiled_ar", None) is None
            or getattr(self, "_compiled_ar_key", None) != cache_key
        ):
            import os as _os

            reserve = int(_os.environ.get("MTPLX_COMPILE_AR_RESERVE_TOKENS", "4096"))
            self._compiled_ar = CompiledARForward(self.model, reserve_tokens=reserve)
            self._compiled_ar_key = cache_key
        return self._compiled_ar

    def forward_ar_capture(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        capture_backend: str | None = None,
    ):
        text_model = getattr(self.model, "language_model", self.model)
        inner = getattr(text_model, "model", None)
        if not (hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx")):
            # Uniform full-attention model (e.g. hy_v3): every layer is plain
            # causal attention, so there is no GDN/recurrent state to capture
            # and forward_with_gdn_capture's hybrid layout (fa_idx/ssm_idx,
            # layer.is_linear) does not exist. The verify forward is just the
            # plain AR forward; commit_captured_prefix with empty captures
            # commits by trimming the standard (trimmable) KV caches, which is
            # the correct prefix commit for pure-attention layers.
            self._count("forward_ar_capture_plain_attention_calls")
            if return_hidden:
                logits, hidden = self.forward_ar(
                    input_ids,
                    cache=cache,
                    return_hidden=True,
                    hidden_variant=hidden_variant,
                )
                return logits, hidden, {}
            logits = self.forward_ar(input_ids, cache=cache)
            return logits, {}

        from .gdn_capture import forward_with_gdn_capture

        return forward_with_gdn_capture(
            self.model,
            input_ids,
            cache=cache,
            return_hidden=return_hidden,
            hidden_variant=hidden_variant,
            capture_backend=capture_backend,
        )

    def _forward_ar_capture_a3b_postconv(
        self,
        input_ids,
        *,
        cache,
        hidden_variant: str | None,
        postconv_implementations: tuple[Callable[..., Any], ...],
    ):
        from .gdn_capture import forward_with_a3b_gdn_postconv_capture

        return forward_with_a3b_gdn_postconv_capture(
            self.model,
            input_ids,
            cache=cache,
            hidden_variant=hidden_variant,
            postconv_implementations=postconv_implementations,
        )

    def draft_mtp(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        mtp_depth: int | None = None,
        position_offset: int | None = None,
    ):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("draft_mtp_calls")
        resolved_hidden_variant = (
            self.contract.hidden_variant
            if mtp_hidden_variant in {None, "auto", "contract"}
            else str(mtp_hidden_variant)
        )
        resolved_concat_order = (
            self.contract.concat_order if concat_order in {None, "auto", "contract"} else concat_order
        )
        with mtp_adapter_depth(self.model, mtp_depth):
            kwargs = {
                "mtp_cache": mtp_cache,
                "concat_order": resolved_concat_order,
                "return_hidden": return_hidden,
                "mtp_hidden_variant": resolved_hidden_variant,
                "position_offset": position_offset,
            }
            try:
                params = py_inspect.signature(self.model.mtp_forward).parameters
            except Exception:
                params = {}
            if "mtp_depth" in params:
                kwargs["mtp_depth"] = mtp_depth
            return self.model.mtp_forward(hidden_states, next_token_ids, **kwargs)

    def update_mtp_cache(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache=None,
        concat_order: str | None = None,
        mtp_hidden_variant: str | None = None,
        position_offset: int | None = None,
        input_embeddings=None,
    ):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("update_mtp_cache_calls")
        resolved_hidden_variant = (
            self.contract.hidden_variant
            if mtp_hidden_variant in {None, "auto", "contract"}
            else str(mtp_hidden_variant)
        )
        resolved_concat_order = (
            self.contract.concat_order if concat_order in {None, "auto", "contract"} else concat_order
        )
        update = getattr(self.model, "mtp_update_cache", None)
        if update is not None:
            try:
                params = py_inspect.signature(update).parameters
            except Exception:
                params = {}
            accepts_kwargs = any(
                param.kind == py_inspect.Parameter.VAR_KEYWORD
                for param in params.values()
            )
            candidates = {
                "mtp_cache": mtp_cache,
                "concat_order": resolved_concat_order,
                "mtp_hidden_variant": resolved_hidden_variant,
                "position_offset": position_offset,
                "input_embeddings": input_embeddings,
            }
            kwargs = {
                key: value
                for key, value in candidates.items()
                if accepts_kwargs or key in params
            }
            if input_embeddings is not None and "input_embeddings" not in kwargs:
                # Silently dropping the spliced vision rows would rebuild the
                # exact draft-history corruption this parameter fixes (#103).
                raise RuntimeError(
                    "this MTP backend does not accept input_embeddings; "
                    "vision history append is unsupported for it"
                )
            if "mtp_depth" in params:
                kwargs["mtp_depth"] = None
            return update(hidden_states, next_token_ids, **kwargs)
        if input_embeddings is not None:
            raise RuntimeError(
                "mtp_forward fallback does not accept input_embeddings; "
                "vision history append is unsupported for it"
            )
        _logits, hidden = self.model.mtp_forward(
            hidden_states,
            next_token_ids,
            mtp_cache=mtp_cache,
            concat_order=resolved_concat_order,
            return_hidden=True,
            mtp_hidden_variant=resolved_hidden_variant,
            position_offset=position_offset,
        )
        return hidden

    def make_cache(self):
        inner = getattr(self.model, "language_model", self.model)
        cache = inner.make_cache()
        from .cache_state import (
            configure_owned_recurrent_state_cache,
            configure_tail_owned_attention_kv_cache,
        )

        configure_owned_recurrent_state_cache(cache)
        configure_tail_owned_attention_kv_cache(cache)
        return cache

    def repage_target_prefill_cache(self, cache: Any) -> bool:
        """Install the runtime's decode cache layout after contiguous prefill."""

        from .cache_state import configure_tail_owned_attention_kv_cache

        configure_tail_owned_attention_kv_cache(cache)
        return True

    def make_mtp_cache(self):
        if not self.mtp_enabled:
            raise RuntimeError("MTP is not enabled for this runtime")
        self._count("make_mtp_cache_calls")
        cache = self.model.make_mtp_cache()
        from .cache_state import configure_mtp_attention_kv_cache

        configure_mtp_attention_kv_cache(cache)
        return cache


class LagunaARRuntime(MTPLXRuntime):
    """Target-only runtime that preserves Laguna's native cache ownership."""

    def forward_ar(
        self,
        input_ids,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        del return_hidden, hidden_variant
        return self.model(
            input_ids,
            cache=cache,
            input_embeddings=input_embeddings,
            emit_logits=emit_logits,
            logits_keep=logits_keep,
        )

    def make_cache(self):
        inner = getattr(self.model, "language_model", self.model)
        return inner.make_cache()

    def repage_target_prefill_cache(self, cache: Any) -> bool:
        del cache
        return False


# HF class name (as declared in config ``architectures``) -> mlx-lm module
# implementing it. Extend this table only with verified schema-compatible
# pairs; an architecture absent here keeps the fail-loud unknown-model_type
# behavior.
_ARCHITECTURE_DECLARED_MODULES = {
    "Qwen3_5ForConditionalGeneration": "qwen3_5",
    "Qwen3_5ForCausalLM": "qwen3_5",
    "Qwen3_5TextForCausalLM": "qwen3_5",
    "Qwen3_5MoeForConditionalGeneration": "qwen3_5_moe",
    "Qwen3_5MoeForCausalLM": "qwen3_5_moe",
    "Qwen3_5MoeTextForCausalLM": "qwen3_5_moe",
}


def _install_architectures_declared_module_alias(config: dict[str, Any]) -> bool:
    """Alias ``mlx_lm.models.<model_type>`` to the module implementing the
    checkpoint's declared ``architectures`` class, when mlx-lm has no module
    for the model_type itself.

    ``mlx_lm.utils.load`` resolves the model class from ``model_type`` alone,
    so a schema-compatible checkpoint under a fresh model_type string (the
    Qwen3.6 -> "qwen3_5" precedent, expected again for Qwen3.8) would
    hard-fail even though the checkpoint itself names the implementing class.
    This honors that declaration — transformers' own class resolution works
    the same way — and logs loudly so an alias load is never silent.
    Returns True when an alias was installed.
    """
    import importlib
    import importlib.util

    tcfg = text_config(config)
    model_type = str(config.get("model_type") or tcfg.get("model_type") or "").strip()
    if not model_type:
        return False
    alias_name = f"mlx_lm.models.{model_type}"
    if alias_name in sys.modules:
        return False
    try:
        if importlib.util.find_spec(alias_name) is not None:
            return False  # mlx-lm knows this model_type natively
    except (ImportError, ValueError):
        return False
    architectures: list[str] = []
    for source in (config, tcfg):
        raw = source.get("architectures")
        if isinstance(raw, list):
            architectures.extend(str(item) for item in raw)
    for arch in architectures:
        target = _ARCHITECTURE_DECLARED_MODULES.get(arch)
        if target is None:
            continue
        try:
            module = importlib.import_module(f"mlx_lm.models.{target}")
        except ImportError:
            continue
        sys.modules[alias_name] = module
        logger.warning(
            "[model-alias] model_type %r has no mlx-lm module; loading via the "
            "checkpoint's declared architecture %s (mlx_lm.models.%s)",
            model_type,
            arch,
            target,
        )
        return True
    return False


def load(
    model_path: Path | str,
    *,
    mtp: bool = True,
    contract: MTPContract | None = None,
    mtp_adapter: Path | str | None = None,
    merge_mtp_adapter: bool = False,
    gemma4_draft_block_size: int | None = None,
    gemma4_target_distribution_mode: str | None = None,
    proj_quant: str | None = None,
    proj_requant: str | None = None,
) -> MTPLXRuntime:
    """Load an MLX model and optionally inject native MTP support.

    ``proj_quant`` / ``proj_requant`` (or the ``MTPLX_PROJ_QUANT`` /
    ``MTPLX_PROJ_REQUANT`` environment variables) quantize the trunk
    ``*_proj`` Linears at load time — see :mod:`mtplx.proj_quant`. Applied
    to the trunk only, before MTP injection, so a draft head's precision is
    never reduced.
    """
    path = Path(model_path)
    from .gemma4_pair import resolve_gemma4_pair_paths

    gemma4_pair = resolve_gemma4_pair_paths(path)
    if gemma4_pair is not None:
        if mtp:
            from .backends.gemma4_assistant import (
                DEFAULT_DRAFT_BLOCK_SIZE,
                Gemma4AssistantRuntimeConfig,
                load_gemma4_assistant_pair,
            )

            metadata = gemma4_pair["metadata"]
            benchmark = (
                metadata.get("benchmark") if isinstance(metadata, dict) else {}
            )
            draft_block_size = DEFAULT_DRAFT_BLOCK_SIZE
            if isinstance(benchmark, dict):
                try:
                    draft_block_size = int(
                        benchmark.get("best_block_size") or draft_block_size
                    )
                except (TypeError, ValueError):
                    draft_block_size = DEFAULT_DRAFT_BLOCK_SIZE
            if gemma4_draft_block_size is not None:
                draft_block_size = int(gemma4_draft_block_size)
            runtime = load_gemma4_assistant_pair(
                Gemma4AssistantRuntimeConfig.from_paths(
                    target_model_path=gemma4_pair["target_model"],
                    assistant_model_path=gemma4_pair["assistant_model"],
                    draft_block_size=draft_block_size,
                    target_distribution_mode=gemma4_target_distribution_mode,
                )
            )
            runtime.model_path = path
            runtime.path = path
            runtime.bundle_path = path
            return runtime
        path = Path(gemma4_pair["target_model"])
    config = load_config(path)
    from .a3b_whole_moe import validate_a3b_whole_moe_load_options

    validate_a3b_whole_moe_load_options(
        mtp_adapter=mtp_adapter,
        merge_mtp_adapter=merge_mtp_adapter,
    )
    if mtp and _is_laguna_s_2_1_mlx_4bit_config(config):
        raise ValueError(
            "Laguna-S-2.1 has no native MTP head; "
            "load it with mtp=False (CLI: --no-mtp)."
        )
    _preflight_laguna_system_memory(config)
    from .step3p5_mtp_patch import is_step3p5_mtp_config
    from .qwen3_5_mtp_patch import (
        install_qwen3_5_mtp_trunk_shim,
        is_qwen3_5_mtp_config,
    )

    # Qwen3.5-MoE MTP exports carry model_type ``qwen3_5_mtp`` (no mlx-lm module);
    # the trunk is a vanilla ``qwen3_5_moe``. Alias it so the trunk loads.
    if is_qwen3_5_mtp_config(config):
        install_qwen3_5_mtp_trunk_shim()

    # hy_v3 has no model class in any released mlx-lm; register the vendored
    # one (kept MTP head) before mlx_lm.utils.load resolves the model type.
    from .hy_v3_mtp_patch import install_hy_v3_model_shim, is_hy_v3_config

    if is_hy_v3_config(config):
        install_hy_v3_model_shim()

    # A checkpoint whose model_type has no mlx-lm module may still declare the
    # implementing class in ``architectures`` — new Qwen generations reuse the
    # qwen3_5 schema under fresh model_type strings (Qwen3.6 shipped as
    # qwen3_5; vLLM loads Qwen3.8-Max FP8 through the same classes). Honor the
    # checkpoint's own declaration instead of hard-failing the load.
    _install_architectures_declared_module_alias(config)

    if is_step3p5_mtp_config(config):
        from mlx_lm.utils import load_model

        tokenizer = _load_tokenizer_resilient(path, config)
        model, _loaded_config = load_model(path)
    else:
        model, tokenizer = _load_base_model(path, config)
    import os as _os

    proj_quant = proj_quant or _os.environ.get("MTPLX_PROJ_QUANT") or None
    proj_requant = proj_requant or _os.environ.get("MTPLX_PROJ_REQUANT") or None
    if proj_quant or proj_requant:
        from .proj_quant import quantize_projections, requantize_projections

        if proj_quant:
            touched = quantize_projections(model, proj_quant)
            logger.info(
                "[proj-quant] quantized %d trunk *_proj modules to %s",
                len(touched), proj_quant,
            )
        if proj_requant:
            touched = requantize_projections(model, proj_requant)
            logger.info(
                "[proj-quant] requantized %d trunk *_proj modules to %s",
                len(touched), proj_requant,
            )
    deepseek_v4_attn_proj_wide_m3_report = None
    if str((config or {}).get("model_type") or "").lower() == "deepseek_v4":
        from .models.deepseek_v4 import configure_deepseek_v4_moe_tail

        configure_deepseek_v4_moe_tail(model, config)
        from .deepseek_v4_attn_proj_wide_m3 import (
            deepseek_v4_attn_proj_wide_m3_enabled,
        )

        if deepseek_v4_attn_proj_wide_m3_enabled():
            from .deepseek_v4_attn_proj_wide_m3 import (
                install_deepseek_v4_attn_proj_wide_m3,
            )

            deepseek_v4_attn_proj_wide_m3_report = (
                install_deepseek_v4_attn_proj_wide_m3(model, config)
            )
            logger.info(
                "[deepseek-v4-attn-proj-wide-m3] %s",
                deepseek_v4_attn_proj_wide_m3_report,
            )
    runtime_metadata = _load_runtime_metadata(path)
    contract = (
        (contract or MTPContract())
        .with_runtime_metadata(runtime_metadata, preserve_explicit=True)
        .with_config_defaults(config)
    )
    mtp_enabled = False
    if mtp:
        from .deepseek_mtp_patch import inject_deepseek_mtp_support, is_deepseek_mtp_config
        from .glm_mtp_patch import inject_glm_mtp_support, is_glm_mtp_config
        from .mimo_mtp_patch import inject_mimo_mtp_support, is_mimo_mtp_config
        from .nemotron_h_mtp_patch import inject_nemotron_h_mtp_support, is_nemotron_h_mtp_config
        from .step3p5_mtp_patch import inject_step3p5_mtp_support
        from .hy_v3_mtp_patch import inject_hy_v3_mtp_support, is_hy_v3_mtp_config
        from .models.deepseek_v4 import (
            inject_deepseek_v4_mtp_support,
            is_deepseek_v4_mtp_config,
        )
        from .qwen3_5_mtp_patch import (
            inject_qwen3_5_mtp_support,
            is_escha_qwen3_5_mtp,
        )

        if is_deepseek_v4_mtp_config(config):
            # Native draft head: the block binds through the ordinary load path
            # and the model already carries the runtime surface, so this only
            # publishes it. Placed ahead of is_deepseek_mtp_config defensively --
            # that predicate keys on model_type in {deepseek_v3, deepseek_v32,
            # glm_moe_dsa}, so it cannot match a deepseek_v4 config today, but it
            # is the arm that would build a V3 head if the sets ever overlap.
            mtp_enabled = inject_deepseek_v4_mtp_support(model, path, config, contract)
        elif is_nemotron_h_mtp_config(config):
            mtp_enabled = inject_nemotron_h_mtp_support(model, path, config, contract)
        elif is_mimo_mtp_config(config):
            mtp_enabled = inject_mimo_mtp_support(model, path, config, contract)
        elif is_glm_mtp_config(config):
            mtp_enabled = inject_glm_mtp_support(model, path, config, contract)
        elif is_step3p5_mtp_config(config):
            mtp_enabled = inject_step3p5_mtp_support(model, path, config, contract)
        elif is_hy_v3_mtp_config(config):
            mtp_enabled = inject_hy_v3_mtp_support(model, path, config, contract)
        elif is_qwen3_5_mtp_config(config) or is_escha_qwen3_5_mtp(config, path):
            # qwen3_5_mtp head, or Escha-W2 (qwen3_5_moe 2-bit trunk whose MTP head borrows the
            # trunk's eschamoe experts) — same head, injected by the same function.
            mtp_enabled = inject_qwen3_5_mtp_support(model, path, config, contract)
        elif is_deepseek_mtp_config(config):
            mtp_enabled = inject_deepseek_mtp_support(model, path, config, contract)
        else:
            mtp_enabled = inject_mtp_support(model, path, config, contract)
        if mtp_enabled:
            if not validate_mtp_support(model):
                raise RuntimeError(f"MTP injection failed for {path}")
        elif mtp_weights_present_on_disk(path, config):
            # MTP weights ship with the model but injection could not use
            # them: a genuine failure the operator should see.
            raise RuntimeError(f"MTP injection failed for {path}")
        else:
            # The config declares MTP layers but no MTP weights are present on
            # disk (e.g. a quant conversion that dropped the draft head).
            # Degrade to autoregressive rather than failing the load.
            logger.warning(
                "[MTP] %s declares MTP layer(s) but ships no MTP weights; "
                "serving autoregressive (no speculative draft head).",
                path,
            )
    compiled_target_factory = None
    whole_moe_plan = None
    selfcheck_report = None
    from .escha_load import is_escha_checkpoint

    escha = is_escha_checkpoint(path, config)
    if escha:
        # Escha's eschamoe (2-bit) experts + int8 non-experts ARE the optimized MoE/MLP path;
        # the standard-A3B serving opts below (native_mlp / moe_packed_projections / whole_moe /
        # row_owned_router / compiled_target_prefix) all assume dense-affine SwitchGLU + Linear
        # and do not apply. The plain qwen3_5_moe forward + mtplx caches + batched_decode still
        # drive it (forward_ar / make_cache come from the runtime class, not this block).
        logger.info("[escha] serving via eschamoe/int8 path; skipping standard-A3B MoE/MLP opts")
    # Laguna skips the qwen3-next kernel stack entirely; its own env-gated
    # fused lanes install right before runtime construction below.
    if not _is_laguna_s_2_1_mlx_4bit_config(config) and not escha:
        from .attention_split import configure_split_full_attention
        from .moe_packed_projections import (
            configure_moe_packed_projections,
            moe_pack_gate_up_enabled,
        )
        from .native_mlp import configure_native_mlp

        configure_split_full_attention(model)
        configure_native_mlp(model)
        # Construction-time only: replaces the MoE gate/up projections with one
        # packed matmul each. Must run after MTP injection so the draft block's
        # MoE layer is packed too, and after load-coverage validation so the
        # packed parameter tree is never compared against checkpoint keys.
        if moe_pack_gate_up_enabled():
            pack_report = configure_moe_packed_projections(model)
            logger.info("[moe-pack] %s", pack_report)
        from .nax_verify import install_nax_qlinear_patch, nax_env_enabled

        if nax_env_enabled():
            nax_report = install_nax_qlinear_patch()
            logger.info("[nax-verify] %s", nax_report)
        from .kernels.gdn_blocked_prefill import (
            blocked_prefill_env_enabled,
            install_gdn_blocked_prefill_patch,
        )

        if blocked_prefill_env_enabled():
            gdn_prefill_report = install_gdn_blocked_prefill_patch()
            logger.info("[gdn-blocked-prefill] %s", gdn_prefill_report)
        from .qwen_row_owned_router import (
            install_qwen_row_owned_routers,
            prepare_qwen_row_owned_routers,
        )
        from .a3b_whole_moe import (
            install_a3b_whole_moe,
            prepare_a3b_whole_moe,
            run_a3b_whole_moe_selfcheck,
        )

        from .gdn_capture import (
            install_a3b_gdn_postconv,
            prepare_a3b_gdn_postconv,
        )
        from .a3b_compiled_target_prefix import (
            preflight_a3b_k1_target_prefix_load_graph,
            prepare_a3b_compiled_target_prefix,
        )

        whole_moe_plan = prepare_a3b_whole_moe(model, config=config)
        router_plan = prepare_qwen_row_owned_routers(model, config=config)
        postconv_plan = prepare_a3b_gdn_postconv(model, config=config)
        postconv_factory = None
        from .kernel_selfcheck import maybe_run_model_selfcheck

        selfcheck_report = maybe_run_model_selfcheck(model)
        if whole_moe_plan is not None and router_plan is None:
            from .a3b_whole_moe import A3BWholeMoeConfigError

            raise A3BWholeMoeConfigError(
                "whole-MoE target M2 requires the accepted row-owned router/combine route"
            )
        if router_plan is not None:
            router_report = install_qwen_row_owned_routers(router_plan, selfcheck_report)
            logger.info("[qwen-row-owned-router] %s", router_report)
        if whole_moe_plan is not None:
            selfcheck_report = run_a3b_whole_moe_selfcheck(
                whole_moe_plan,
                selfcheck_report,
            )
        if postconv_plan is not None:
            postconv_factory = install_a3b_gdn_postconv(
                postconv_plan, selfcheck_report
            )
            from .gdn_capture import gdn_postconv_stats

            logger.info("[a3b-gdn-postconv] %s", gdn_postconv_stats())
        compiled_target_factory = prepare_a3b_compiled_target_prefix(
            model,
            config=config,
            gdn_postconv_factory=postconv_factory,
        )
    adapter_path = Path(mtp_adapter) if mtp_adapter is not None else None
    adapter_metadata = None
    adapter_merge_report = None
    if adapter_path is not None:
        if not mtp_enabled:
            raise RuntimeError("MTP adapter requires mtp=True")
        adapter_metadata = install_saved_mtp_lora_adapter(model, adapter_path)
        if merge_mtp_adapter:
            adapter_merge_report = merge_installed_mtp_lora_adapters(model)
    elif merge_mtp_adapter:
        raise RuntimeError("merge_mtp_adapter requires mtp_adapter")
    deepseek_v4_o_lora_report = None
    deepseek_v4_attention_island_report = None
    if str(config.get("model_type") or "").lower() == "deepseek_v4":
        from .models.deepseek_v4 import (
            _o_lora_mode_from_env,
            install_deepseek_v4_o_lora_routes,
        )

        selected_o_lora_mode = _o_lora_mode_from_env()
        # The canonical mixed route hard-validates the exact DeepSeek-V4-Flash
        # topology (43 body layers, rank-1024 Q4/g64 wo_a/wo_b, one dense-BF16
        # MTP block) and refuses anything else. That strictness is correct for
        # the explicit gather_qmm opt-in, but the default "cached" mode must
        # keep loading every DSV4 MTP artifact (8-bit/bf16 user conversions,
        # other group sizes) exactly as v2.4.2 did via the per-module dense
        # route — which is bit-identical on the canonical artifact anyway
        # (test_cached_dequant_is_bit_identical).
        canonical_mixed_route = bool(
            mtp_enabled and selected_o_lora_mode == "gather_qmm"
        )
        if not mtp_enabled:
            # An artifact that declared but did not ship MTP weights already
            # degraded to AR above. It has no dense MTP module to validate or
            # route, so bind the trunk's explicit stock/cached construction.
            selected_o_lora_mode = "cached"
        deepseek_v4_o_lora_report = install_deepseek_v4_o_lora_routes(
            model,
            mode=selected_o_lora_mode,
            canonical_mixed_route=canonical_mixed_route,
        )
        logger.info("[deepseek-v4-o-lora] %s", deepseek_v4_o_lora_report)
        from .deepseek_v4_attention_island import (
            deepseek_v4_attention_island_enabled,
            install_deepseek_v4_attention_island,
        )

        if deepseek_v4_attention_island_enabled():
            deepseek_v4_attention_island_report = (
                install_deepseek_v4_attention_island(model, config)
            )
            logger.info(
                "[deepseek-v4-attention-island] %s",
                deepseek_v4_attention_island_report,
            )
    fused_report: list[dict[str, Any]] = []
    if _is_laguna_s_2_1_mlx_4bit_config(config):
        # Env-gated fused decode paths (MTPLX_LAGUNA_*): with no switches set
        # this returns an empty report and changes nothing, so default serving
        # behavior is untouched; a serving wrapper that exports the measured
        # stack gets it engaged at load.
        from .models.laguna_fused import install_from_env as _laguna_install_fused

        fused_report = _laguna_install_fused(model)
        if fused_report:
            logger.info("[laguna-fused] %s", fused_report)
    runtime_class = (
        LagunaARRuntime
        if _is_laguna_s_2_1_mlx_4bit_config(config)
        else MTPLXRuntime
    )
    runtime = runtime_class(
        model,
        tokenizer,
        path,
        mtp_enabled,
        contract,
        mtp_adapter_path=adapter_path,
        mtp_adapter_metadata=adapter_metadata,
        mtp_adapter_merge_report=adapter_merge_report,
        deepseek_v4_o_lora_report=deepseek_v4_o_lora_report,
        deepseek_v4_attn_proj_wide_m3_report=deepseek_v4_attn_proj_wide_m3_report,
        deepseek_v4_attention_island_report=deepseek_v4_attention_island_report,
        a3b_compiled_target_prefix_factory=compiled_target_factory,
        a3b_whole_moe_installed=False,
    )
    if whole_moe_plan is not None:
        if compiled_target_factory is None:
            from .a3b_whole_moe import A3BWholeMoeConfigError

            raise A3BWholeMoeConfigError(
                "whole-MoE requires exact compiled target-prefix construction"
            )
        whole_moe_report = install_a3b_whole_moe(
            whole_moe_plan,
            selfcheck_report,
            compiled_preflight=lambda: preflight_a3b_k1_target_prefix_load_graph(
                runtime, compiled_target_factory
            ),
        )
        runtime.a3b_whole_moe_installed = True
        logger.info("[a3b-whole-moe] %s", whole_moe_report)
    # The server prints this as its startup engagement receipt; logger.info
    # alone is invisible under `python -m mtplx.server.openai` (no handler).
    runtime.laguna_fused_report = fused_report
    return runtime


def inspect(path: Path | str):
    return inspect_model(path)


def _is_laguna_s_2_1_mlx_4bit_config(config: dict[str, Any]) -> bool:
    from .models.laguna_config import is_laguna_s_2_1_mlx_4bit_config

    return is_laguna_s_2_1_mlx_4bit_config(config)


def _model_classes_for_config(config: dict[str, Any]) -> tuple[type, type] | None:
    """Return MTPLX-owned model classes for architectures missing in mlx-lm."""

    if str(config.get("model_type") or "").lower() == "deepseek_v4":
        from .models.deepseek_v4 import Model, ModelArgs

        return Model, ModelArgs
    if not _is_laguna_s_2_1_mlx_4bit_config(config):
        return None
    from .models.laguna import Model, ModelArgs

    return Model, ModelArgs


def _load_base_model(path: Path, config: dict[str, Any]) -> tuple[Any, Any]:
    from .escha_load import is_escha_checkpoint, load_escha_model

    if is_escha_checkpoint(path, config):
        # EschaLabs Qwen3.6-A3B-Escha-W2: qwen3_5_moe trunk with eschamoe 2-bit experts +
        # int8 non-experts. Returns an ordinary qwen3_5_moe model the runtime drives unchanged.
        return load_escha_model(path, config)
    if (
        config.get("architectures") == ["LagunaForCausalLM"]
        and str(config.get("model_type") or "").lower() == "laguna"
        and "model_file" in config
    ):
        raise ValueError("Laguna model_file execution is not permitted")
    model_classes = _model_classes_for_config(config)
    if model_classes is not None:
        from mlx_lm.utils import load_model

        from .models.laguna_config import laguna_module_quantization

        tokenizer = _load_tokenizer_resilient(path, config)
        load_kwargs: dict[str, Any] = {
            "get_model_classes": lambda config: model_classes,
        }
        module_quantization = laguna_module_quantization(config)
        if module_quantization is not None:
            # The pinned oQ4e checkpoint keys its mixed-precision quantization
            # dict by the ``language_model.``-prefixed export path. Strip the
            # prefix so mlx-lm's config-driven quantizer addresses each module
            # by its tree path (the BF16 routers carry no entry and stay
            # unquantized). mlx-lm reads this from config["quantization"], not
            # from any model-level predicate.
            load_kwargs["model_config"] = {
                "quantization": module_quantization,
                "quantization_config": module_quantization,
            }
        model, _loaded_config = load_model(path, **load_kwargs)
        return model, tokenizer

    from mlx_lm.utils import load as mlx_lm_load

    return mlx_lm_load(str(_mtp_alias_load_path(path, config)))


# A chat_template that is nothing but a Jinja ``{% include %}`` redirect to a
# sidecar file. The pinned Laguna-S-2.1 oQ4e checkpoint ships the 35-char stub
# ``{% include 'chat_template.jinja' %}`` in tokenizer_config.json. transformers
# compiles embedded chat templates in a loader-less Jinja Environment, so any
# apply_chat_template on such a stub raises
# ``TypeError('no loader for this environment specified')`` — the failure the
# 2026-07-22 laguna serving window hit on both the one-shot and server paths.
_JINJA_INCLUDE_CHAT_TEMPLATE_RE = re.compile(r"\{%-?\s*include\b")


def _is_jinja_include_chat_template(chat_template: Any) -> bool:
    """True when ``chat_template`` is a string carrying a Jinja include."""

    return isinstance(chat_template, str) and bool(
        _JINJA_INCLUDE_CHAT_TEMPLATE_RE.search(chat_template)
    )


def _pinned_chat_template_text(model_path: Path) -> str | None:
    """Contents of the sidecar chat_template.jinja pinned next to the model.

    Returns None when the file is absent or empty. The file is only read, never
    mutated — its sha256 is load-bearing for artifact-integrity checks.
    """

    jinja = model_path / "chat_template.jinja"
    if not jinja.exists():
        return None
    text = jinja.read_text(encoding="utf-8")
    return text if text.strip() else None


def _repair_included_chat_template(tokenizer: Any, model_path: Path) -> None:
    """Swap an include-stub chat_template for the pinned sidecar contents.

    The oQ4e tokenizer_config.json redirects its chat_template to a sidecar via
    ``{% include 'chat_template.jinja' %}``. transformers cannot resolve the
    include (no Jinja loader), so apply_chat_template raises the moment it runs.
    The real template — self-contained, no include/import/extends — lives in
    chat_template.jinja beside the weights; substitute its contents in memory.
    Setting ``tokenizer.chat_template`` on the mlx-lm TokenizerWrapper forwards
    to the underlying HF tokenizer, which is what apply_chat_template renders.
    """

    current = getattr(tokenizer, "chat_template", None)
    if not _is_jinja_include_chat_template(current):
        return
    replacement = _pinned_chat_template_text(model_path)
    if replacement is None:
        return
    tokenizer.chat_template = replacement


def _load_tokenizer_resilient(model_path: Path, config: dict[str, Any]) -> Any:
    from mlx_lm.utils import load_tokenizer

    try:
        tokenizer = load_tokenizer(model_path)
    except Exception as exc:  # noqa: BLE001 - transformers raises several strict-config errors
        logger.warning(
            "[tokenizer] AutoTokenizer parse failed (%s); using tokenizer.json fallback",
            exc,
        )
    else:
        _repair_included_chat_template(tokenizer, model_path)
        return tokenizer

    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from transformers import PreTrainedTokenizerFast

    tcfg_path = model_path / "tokenizer_config.json"
    tcfg = json.loads(tcfg_path.read_text(encoding="utf-8")) if tcfg_path.exists() else {}
    passthrough = {
        key: tcfg[key]
        for key in ("bos_token", "eos_token", "pad_token", "unk_token", "additional_special_tokens")
        if key in tcfg
    }
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(model_path / "tokenizer.json"),
        **passthrough,
    )
    chat_template = tcfg.get("chat_template")
    # An include-stub is not a usable template (transformers has no loader for
    # it); treat it as absent so the pinned sidecar below supplies the real one.
    if _is_jinja_include_chat_template(chat_template):
        chat_template = None
    if not chat_template:
        replacement = _pinned_chat_template_text(model_path)
        if replacement is not None:
            chat_template = replacement
    if chat_template:
        hf_tokenizer.chat_template = chat_template
    eos = config.get("eos_token_id")
    if eos is None:
        eos = (config.get("text_config") or {}).get("eos_token_id")
    if isinstance(eos, int):
        eos_ids = [eos]
    elif isinstance(eos, (list, tuple)):
        eos_ids = list(eos)
    else:
        eos_ids = None
    return TokenizerWrapper(
        hf_tokenizer,
        eos_token_ids=eos_ids,
        chat_template=None,
    )


def _mtp_alias_load_path(path: Path, config: dict[str, Any] | None) -> Path:
    """Loadable path for `*_mtp`-typed checkpoints (issue #147).

    vLLM-convention MTP checkpoints ship config.json with model_type like
    ``qwen3_5_mtp``: the trunk is the plain base architecture plus an
    embedded MTP head. mlx_lm's class table has no ``*_mtp`` modules, so
    handing it the raw dir fails with "Model type ... not supported" even
    though the forge probe correctly reports the family as supported. When
    the stripped base module exists in mlx_lm and the full name does not,
    build a symlink wrapper with a patched config.json (model_type=base)
    and load through it; MTP injection later picks the head up from the
    original weights. Everything else returns the path untouched.
    """

    model_type = str((config or {}).get("model_type") or "")
    if not model_type.endswith("_mtp"):
        return path
    base_type = model_type[: -len("_mtp")]
    import importlib.util

    def _mlx_lm_has(model_type_name: str) -> bool:
        return (
            importlib.util.find_spec(f"mlx_lm.models.{model_type_name}")
            is not None
        )

    if _mlx_lm_has(model_type) or not _mlx_lm_has(base_type):
        return path
    try:
        wrapper_root = Path.home() / ".mtplx" / "build-cache" / "mtp-alias-load"
        digest = hashlib.sha256(
            f"{path.resolve()}::{base_type}".encode("utf-8")
        ).hexdigest()[:16]
        wrapper = wrapper_root / f"{path.name}-{base_type}-{digest}"
        patched_config = dict(config or {})
        patched_config["model_type"] = base_type
        marker = wrapper / ".mtplx-alias-source"
        if not marker.exists() or marker.read_text(encoding="utf-8") != str(
            path.resolve()
        ):
            wrapper.mkdir(parents=True, exist_ok=True)
            for item in path.iterdir():
                if item.name in {"config.json", ".mtplx-alias-source"}:
                    continue
                link = wrapper / item.name
                if link.is_symlink() or link.exists():
                    continue
                link.symlink_to(item)
            marker.write_text(str(path.resolve()), encoding="utf-8")
        # Rewrite the config every time: the source config may have changed.
        (wrapper / "config.json").write_text(
            json.dumps(patched_config, indent=2), encoding="utf-8"
        )
        return wrapper
    except Exception:
        # Wrapper construction is best-effort; the raw path preserves the
        # original (informative) mlx_lm error.
        return path


def _load_runtime_metadata(path: Path) -> dict[str, Any] | None:
    runtime_path = path / "mtplx_runtime.json"
    if not runtime_path.exists():
        return None
    try:
        data = json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None
