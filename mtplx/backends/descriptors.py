"""Backend-owned product semantics for MTPLX runtime surfaces.

This module is deliberately import-light: CLI, onboarding, artifact inspection,
and the HTTP server can use it without importing MLX.  Architecture-specific
runtime code still lives in each backend module; this file only describes the
public contract that every surface needs to render, validate, and route a model
without inheriting Qwen-specific assumptions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any


ASSISTANT_MODEL_ATTRS = ("assistant_model", "gemma_assistant_model")
TARGET_DISTRIBUTION_MODE_ATTRS = (
    "target_distribution_mode",
    "gemma_target_distribution_mode",
)


@dataclass(frozen=True)
class SamplerDefaults:
    temperature: float
    top_p: float
    top_k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "top_k": int(self.top_k),
        }


@dataclass(frozen=True)
class DraftSemantics:
    """How a backend exposes speculative depth/block controls to users."""

    request_field: str
    display_label: str
    default: int
    minimum: int
    maximum: int
    unit: str

    def clamp(self, value: int | None) -> int:
        raw = self.default if value is None else int(value)
        return max(int(self.minimum), min(int(self.maximum), int(raw)))

    def label_for_stats(self, value: int | None, *, generation_mode: str = "mtp") -> str:
        if str(generation_mode or "").lower() == "ar":
            return "AR"
        clamped = self.clamp(value)
        if self.unit == "block":
            return f"MTP block {clamped}"
        return f"MTP depth {clamped}"

    def to_dict(self) -> dict[str, Any]:
        if self.unit == "block":
            labels = [f"Block {value}" for value in range(self.minimum, self.maximum + 1)]
        else:
            labels = [f"D{value}" for value in range(self.minimum, self.maximum + 1)]
        return {
            "supported": True,
            "request_field": self.request_field,
            "display_label": self.display_label,
            "default": int(self.default),
            "minimum": int(self.minimum),
            "maximum": int(self.maximum),
            "unit": self.unit,
            "value_labels": labels,
        }


@dataclass(frozen=True)
class ReasoningCodec:
    parser: str
    display_name: str
    default_mode: str = "off"
    supported: bool = True
    modes: tuple[str, ...] = ("auto", "on", "off")
    history_policy: str = "preserve_when_enabled"
    # Drawn from mtplx.reasoning_effort.REASONING_EFFORT_LEVELS: the app
    # renders these verbatim, so a level outside that vocabulary is one the
    # user can pick and no writing surface will accept
    # (test_reasoning_effort_vocabulary_covers_every_family pins it).
    effort_levels: tuple[str, ...] = ()
    default_effort: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": bool(self.supported),
            "parser": self.parser,
            "display_name": self.display_name,
            "modes": list(self.modes if self.supported else ()),
            "default_mode": self.default_mode,
            "default": self.default_mode,
            "history_policy": self.history_policy,
            "effort_levels": list(self.effort_levels if self.supported else ()),
            "default_effort": self.default_effort if self.supported else None,
        }


@dataclass(frozen=True)
class TunePolicy:
    supported: bool
    control_field: str = "depth"
    candidates: tuple[str, ...] = ("AR", "D1", "D2", "D3")
    supported_families: tuple[str, ...] = ("qwen3_5", "qwen3_6", "qwen3_8", "gemma4")
    unsupported_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": bool(self.supported),
            "supported_families": list(self.supported_families),
            "control_field": self.control_field,
            "candidates": list(self.candidates if self.supported else ()),
            "unsupported_reason": self.unsupported_reason,
        }


@dataclass(frozen=True)
class KVQuantPolicy:
    supported: bool
    modes: tuple[str, ...] = ("off",)
    restart_required: bool = True
    proof_level: str = "not_validated"
    disabled_reason: str | None = (
        "KV quantization is not supported for this model."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": bool(self.supported),
            "modes": list(self.modes if self.supported else ("off",)),
            "restart_required": bool(self.restart_required),
            "proof_level": self.proof_level,
            "disabled_reason": None if self.supported else self.disabled_reason,
        }


@dataclass(frozen=True)
class ContextWindowPolicy:
    """Model-facing context-window bounds for app and client launch UX."""

    supported: bool = True
    minimum: int = 4_096
    maximum: int = 262_144
    default: int = 262_144
    step: int = 1_024
    source: str = "model_config"

    def clamp(self, value: int | None) -> int:
        raw = self.default if value is None else int(value)
        return max(int(self.minimum), min(int(self.maximum), int(raw)))

    def with_resolved_max(
        self,
        value: int | None,
        *,
        source: str = "runtime",
    ) -> "ContextWindowPolicy":
        if value is None:
            return self
        resolved = int(value)
        if resolved <= 0 or resolved > 1_048_576:
            return self
        maximum = max(int(self.minimum), resolved)
        default = min(maximum, max(int(self.minimum), int(self.default)))
        return ContextWindowPolicy(
            supported=self.supported,
            minimum=self.minimum,
            maximum=maximum,
            default=default,
            step=self.step,
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "supported": bool(self.supported),
            "minimum": int(self.minimum),
            "maximum": int(self.maximum),
            "default": int(self.default),
            "step": int(self.step),
            "source": self.source,
            "unit": "tokens",
        }


@dataclass(frozen=True)
class TargetDistributionMode:
    """Exactness metadata for one backend-owned verification mode."""

    name: str
    exact: bool
    product: bool
    status: str = "debug"
    notes: str = ""
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "exact": bool(self.exact),
            "product": bool(self.product),
            "status": self.status,
            "notes": self.notes,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True)
class TargetDistributionPolicy:
    """How an external-drafter backend verifies assistant proposals."""

    modes: tuple[str, ...] = ("backend_default",)
    default_mode: str = "backend_default"
    default_window_size: int | None = None
    exact: bool | None = True
    mode_metadata: tuple[TargetDistributionMode, ...] = ()
    status: str = "backend_default"
    telemetry_fields: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        exact: bool | None = self.exact
        if self.mode_metadata:
            exact_values = {bool(mode.exact) for mode in self.mode_metadata}
            exact = exact_values.pop() if len(exact_values) == 1 else None
        return {
            "modes": list(self.modes),
            "default_mode": self.default_mode,
            "default_window_size": (
                None if self.default_window_size is None else int(self.default_window_size)
            ),
            "exact": exact if exact is None else bool(exact),
            "mode_metadata": [mode.to_dict() for mode in self.mode_metadata],
            "status": self.status,
            "telemetry_fields": list(self.telemetry_fields),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BackendDescriptor:
    backend_id: str
    architecture_id: str
    model_family: str
    display_name: str
    artifact_layout: str
    runtime_capabilities: tuple[str, ...]
    sampler_defaults: SamplerDefaults
    reasoning_codec: ReasoningCodec
    draft_semantics: DraftSemantics
    uses_external_assistant: bool = False
    uses_draft_lm_head: bool = True
    hidden_variant: str = "post_norm"
    mtp_history_policy: str = "committed"
    target_distribution_modes: tuple[str, ...] = ("backend_default",)
    default_target_distribution_mode: str = "backend_default"
    target_distribution_policy: TargetDistributionPolicy = field(
        default_factory=TargetDistributionPolicy
    )
    tune_policy: TunePolicy = field(
        default_factory=lambda: TunePolicy(
            supported=False,
            unsupported_reason="Tune is not supported for this model family.",
        )
    )
    kv_quant_policy: KVQuantPolicy = field(default_factory=KVQuantPolicy)
    context_window_policy: ContextWindowPolicy = field(
        default_factory=ContextWindowPolicy
    )
    default_max_response_tokens: int | None = None
    default_tool_prompt_mode: str = "hybrid"
    required_tool_prompt_mode: str | None = None
    required_chat_template_profile: str | None = None
    allows_chat_template_path: bool = True
    validation_status: str = "qa_verified"
    app_ui_policy: str = "descriptor_owned"
    status: str = "qa_verified"
    profile_policy: str = "profile-owned"
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        policy = self.target_distribution_policy
        if (
            policy.modes != ("backend_default",)
            or policy.default_mode != "backend_default"
        ):
            object.__setattr__(self, "target_distribution_modes", tuple(policy.modes))
            object.__setattr__(
                self,
                "default_target_distribution_mode",
                str(policy.default_mode),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "architecture_id": self.architecture_id,
            "model_family": self.model_family,
            "display_name": self.display_name,
            "artifact_layout": self.artifact_layout,
            "runtime_capabilities": list(self.runtime_capabilities),
            "sampler_defaults": self.sampler_defaults.to_dict(),
            "reasoning_codec": self.reasoning_codec.to_dict(),
            "draft_semantics": self.draft_semantics.to_dict(),
            "uses_external_assistant": bool(self.uses_external_assistant),
            "uses_draft_lm_head": bool(self.uses_draft_lm_head),
            "hidden_variant": self.hidden_variant,
            "mtp_history_policy": self.mtp_history_policy,
            "target_distribution_modes": list(self.target_distribution_modes),
            "default_target_distribution_mode": self.default_target_distribution_mode,
            "target_distribution_policy": self.target_distribution_policy.to_dict(),
            "tune_policy": self.tune_policy.to_dict(),
            "kv_quant_policy": self.kv_quant_policy.to_dict(),
            "context_window_policy": self.context_window_policy.to_dict(),
            "default_max_response_tokens": self.default_max_response_tokens,
            "default_tool_prompt_mode": self.default_tool_prompt_mode,
            "required_tool_prompt_mode": self.required_tool_prompt_mode,
            "required_chat_template_profile": self.required_chat_template_profile,
            "allows_chat_template_path": self.allows_chat_template_path,
            "validation_status": self.validation_status,
            "app_ui_policy": self.app_ui_policy,
            "status": self.status,
            "profile_policy": self.profile_policy,
            "notes": list(self.notes),
        }

    def supports(self, capability: str) -> bool:
        return str(capability) in set(self.runtime_capabilities)


QWEN3_NEXT_DESCRIPTOR = BackendDescriptor(
    backend_id="qwen3_next",
    architecture_id="qwen3-next-mtp",
    model_family="qwen",
    display_name="Qwen native MTP",
    artifact_layout="single_mlx_folder_native_mtp",
    runtime_capabilities=(
        "target_logits",
        "native_draft_head",
        "exact_speculative_sampling",
        "sessionbank_committed_mtp_history",
        "native_adaptive_depth_policy",
        "async_session_postcommit",
    ),
    sampler_defaults=SamplerDefaults(temperature=0.6, top_p=0.95, top_k=20),
    reasoning_codec=ReasoningCodec(
        parser="qwen3",
        display_name="Qwen think tags",
        default_mode="auto",
    ),
    draft_semantics=DraftSemantics(
        request_field="depth",
        display_label="Draft depth",
        default=3,
        minimum=1,
        maximum=3,
        unit="depth",
    ),
    uses_external_assistant=False,
    uses_draft_lm_head=True,
    hidden_variant="post_norm",
    tune_policy=TunePolicy(supported=True),
    kv_quant_policy=KVQuantPolicy(
        supported=True,
        modes=("off", "q8", "q4"),
        proof_level="unit_and_runtime_cache_validated",
        disabled_reason=None,
    ),
    context_window_policy=ContextWindowPolicy(
        maximum=262_144,
        default=262_144,
        source="qwen3_next_config",
    ),
    status="qa_verified",
)


# Qwen3.8 family overrides. Qwen3.8-27B shares the Qwen3.6 trunk geometry and
# therefore the qwen3_next backend, but ships its own inference contract: the
# official thinking-mode sampler (temperature 1.0, top_p 0.95, top_k 20),
# official reasoning_effort levels (xhigh / medium / low), preserve_thinking
# retained-history rendering, and a multi-step-trained MTP head. Upstream's
# generic default is xhigh; MTPLX defaults coding sessions to medium after a
# strict max-fan live A/B completed the same correct uncapped Aphanes task in
# 51.52s versus 314.91s at xhigh (2026-08-14). Users can still select xhigh.
QWEN3_8_SAMPLER_DEFAULTS = SamplerDefaults(temperature=1.0, top_p=0.95, top_k=20)
# Strict max-fan A/B on drop day (2026-08-14, Bare-Speed Q4, alternating
# 2,000-token xhigh arms) kept the official target sampler for the draft:
# draft 1.0 averaged 46.05 tok/s versus 42.79 at 0.6, with higher D2/D3
# acceptance. The earlier 0.6 result was thermally uncontrolled and is not a
# product receipt.
QWEN3_8_DRAFT_TEMPERATURE = 1.0
# Per-family dynamic draft-temperature curves: (target_temperature,
# draft_temperature) points, piecewise-linear, flat extrapolation
# (draft_sampling.resolve_draft_temperature). An absent family means the
# identity policy — the static per-family draft temperature above. Curves
# are stamped ONLY from a measured max-fan ABBA calibration campaign (see
# MEASUREMENTS.md); never invent offsets. Target temp 0 is handled by
# greedy draft coupling in the server resolver, not by these curves.
DRAFT_TEMPERATURE_CURVES: dict[str, tuple[tuple[float, float], ...]] = {}


def draft_temperature_curve_for_model(
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
    descriptor: "BackendDescriptor | None" = None,
) -> tuple[tuple[float, float], ...] | None:
    """The measured draft-temperature curve for the model's family, or None
    (identity) when no calibration has been stamped."""

    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=descriptor,
    )
    return DRAFT_TEMPERATURE_CURVES.get(family)
QWEN3_8_REASONING_CODEC = ReasoningCodec(
    parser="qwen3",
    display_name="Qwen think tags",
    default_mode="auto",
    effort_levels=("xhigh", "medium", "low"),
    default_effort="medium",
)
QWEN3_8_DRAFT_SEMANTICS = DraftSemantics(
    request_field="depth",
    display_label="Draft depth",
    default=3,
    minimum=1,
    # The multi-step head trains for deeper drafts, but the live depth-4 serve
    # lane killed the daemon on drop day (silent death mid-request, no crash
    # report — memory-kill signature; receipt: scratchpad serve-d4-t0.6.log,
    # warmup healthy at 88-92 tok/s then nothing). Cap at 3 until the deep
    # lane is root-caused; reopen with the QL5-7 packed-verify extension.
    maximum=3,
    unit="depth",
)


def sampler_defaults_for_model(
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
    descriptor: BackendDescriptor | None = None,
) -> SamplerDefaults:
    resolved = descriptor or descriptor_from_inspection(inspection)
    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=resolved,
    )
    if family == "qwen3_8":
        return QWEN3_8_SAMPLER_DEFAULTS
    return resolved.sampler_defaults


def draft_semantics_for_model(
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
    descriptor: BackendDescriptor | None = None,
) -> DraftSemantics:
    resolved = descriptor or descriptor_from_inspection(inspection)
    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=resolved,
    )
    if family == "qwen3_8":
        return QWEN3_8_DRAFT_SEMANTICS
    return resolved.draft_semantics


LAGUNA_AR_DESCRIPTOR = BackendDescriptor(
    backend_id="laguna_ar",
    architecture_id="laguna-s-2.1-ar",
    model_family="laguna",
    display_name="Laguna-S-2.1 target-only AR",
    artifact_layout="single_mlx_folder_target_only_ar",
    runtime_capabilities=("target_logits", "target_only_ar"),
    sampler_defaults=SamplerDefaults(temperature=1.0, top_p=1.0, top_k=20),
    reasoning_codec=ReasoningCodec(
        parser="poolside_v1",
        display_name="Poolside v1 think tags",
        default_mode="on",
        supported=True,
        modes=("auto", "on", "off"),
        history_policy="preserve_when_enabled",
    ),
    draft_semantics=DraftSemantics(
        request_field="depth",
        display_label="Draft depth",
        default=1,
        minimum=1,
        maximum=1,
        unit="depth",
    ),
    uses_external_assistant=False,
    uses_draft_lm_head=False,
    tune_policy=TunePolicy(
        supported=False,
        supported_families=(),
        unsupported_reason="Laguna-S-2.1 is installed as target-only AR.",
    ),
    kv_quant_policy=KVQuantPolicy(supported=False),
    context_window_policy=ContextWindowPolicy(
        maximum=1_048_576,
        default=32_768,
        source="laguna_s_2_1_config",
    ),
    default_max_response_tokens=32_768,
    default_tool_prompt_mode="native",
    required_tool_prompt_mode="native",
    required_chat_template_profile="tokenizer",
    allows_chat_template_path=False,
    validation_status="target_exact_ar",
    status="target_exact_ar",
    notes=(
        "The checkpoint has no native MTP head.",
        "The bundled loader and native MLX cache path are pinned to Laguna-S-2.1 4-bit geometry.",
    ),
)


MLX_LM_AR_DESCRIPTOR = BackendDescriptor(
    backend_id="mlx_lm_ar",
    architecture_id="mlx-lm-ar-family",
    model_family="mlx-lm",
    display_name="mlx-lm target-only AR",
    artifact_layout="single_mlx_folder_target_only_ar",
    runtime_capabilities=("target_logits", "target_only_ar"),
    sampler_defaults=SamplerDefaults(temperature=0.6, top_p=0.95, top_k=20),
    reasoning_codec=ReasoningCodec(
        parser="none",
        display_name="No verified reasoning parser",
        default_mode="off",
        supported=False,
        modes=(),
        history_policy="visible_content_only",
    ),
    draft_semantics=DraftSemantics(
        request_field="depth",
        display_label="Draft depth",
        default=1,
        minimum=1,
        maximum=1,
        unit="depth",
    ),
    uses_external_assistant=False,
    uses_draft_lm_head=False,
    tune_policy=TunePolicy(
        supported=False,
        supported_families=(),
        unsupported_reason=(
            "mlx-lm AR-only checkpoints have no MTPLX tune path."
        ),
    ),
    kv_quant_policy=KVQuantPolicy(supported=False),
    context_window_policy=ContextWindowPolicy(
        maximum=1_048_576,
        default=131_072,
        source="model_config",
    ),
    default_max_response_tokens=32_768,
    default_tool_prompt_mode="native",
    required_chat_template_profile="tokenizer",
    validation_status="experimental_mlx_lm_ar",
    status="experimental_mlx_lm_ar",
    notes=(
        "Recognized no-MTP architectures served through the bundled mlx-lm "
        "loader in target-only AR mode.",
        "No exactness baseline: runs report as unverified until a "
        "per-artifact contract is recorded.",
    ),
)


# Same lane and backend_id as MLX_LM_AR_DESCRIPTOR, but with the LFM wire
# format verified: literal <think> tags without a prefilled open tag, and
# pythonic <|tool_call_start|> envelopes. Deliberately NOT registered in
# DESCRIPTORS_BY_BACKEND_ID (the generic entry owns the backend_id); only
# descriptor_from_runtime returns it, after sniffing the loaded model family.
MLX_LM_AR_LFM2_DESCRIPTOR = replace(
    MLX_LM_AR_DESCRIPTOR,
    display_name="mlx-lm target-only AR (LFM grammar)",
    reasoning_codec=ReasoningCodec(
        parser="lfm2",
        display_name="LFM think tags",
        default_mode="auto",
    ),
    # The LFM chat template owns tool formatting (List of tools + pythonic
    # <|tool_call_start|> envelope). The hybrid launch default would inject the
    # legacy <tool_call> XML contract on top and the model obeys the injected
    # format over its native one — with malformed results.
    required_tool_prompt_mode="native",
)


NATIVE_CONTRACT_DESCRIPTOR = BackendDescriptor(
    backend_id="native_mtp",
    architecture_id="native-contract-mtp",
    model_family="native-mtp",
    display_name="Native MTP",
    artifact_layout="single_mlx_folder_native_mtp",
    runtime_capabilities=(
        "target_logits",
        "native_draft_head",
        "exact_speculative_sampling",
        "sessionbank_committed_mtp_history",
        "native_adaptive_depth_policy",
        "async_session_postcommit",
    ),
    sampler_defaults=SamplerDefaults(temperature=0.6, top_p=0.95, top_k=20),
    reasoning_codec=ReasoningCodec(
        parser="none",
        display_name="No verified reasoning parser",
        default_mode="off",
        supported=False,
        modes=(),
        history_policy="visible_content_only",
    ),
    draft_semantics=DraftSemantics(
        request_field="depth",
        display_label="Draft depth",
        default=3,
        minimum=1,
        maximum=3,
        unit="depth",
    ),
    uses_external_assistant=False,
    uses_draft_lm_head=True,
    hidden_variant="post_norm",
    tune_policy=TunePolicy(
        supported=False,
        unsupported_reason="Tune is supported for Qwen 3.5, Qwen 3.6, and Gemma 4 MTPLX models only.",
    ),
    kv_quant_policy=KVQuantPolicy(supported=False),
    status="experimental_contract_gated",
)


STEP3P5_MTP_DESCRIPTOR = BackendDescriptor(
    backend_id="step3p5_mtp",
    architecture_id="step3p5-mtp",
    model_family="step",
    display_name="Step native MTP",
    artifact_layout="single_mlx_folder_native_mtp",
    runtime_capabilities=(
        "target_logits",
        "native_draft_head",
        "exact_speculative_sampling",
        "sessionbank_committed_mtp_history",
    ),
    sampler_defaults=SamplerDefaults(temperature=0.6, top_p=0.95, top_k=20),
    reasoning_codec=ReasoningCodec(
        parser="step3p5",
        display_name="Step reasoning",
        default_mode="auto",
        supported=True,
        modes=("auto", "on", "off"),
        history_policy="preserve_when_enabled",
        effort_levels=("low", "medium", "high"),
        default_effort="low",
    ),
    draft_semantics=DraftSemantics(
        request_field="depth",
        display_label="Draft depth",
        default=1,
        minimum=1,
        maximum=3,
        unit="depth",
    ),
    uses_external_assistant=False,
    uses_draft_lm_head=True,
    hidden_variant="pre_norm",
    tune_policy=TunePolicy(
        supported=False,
        unsupported_reason="Tune is supported for Qwen 3.5, Qwen 3.6, and Gemma 4 MTPLX models only.",
    ),
    kv_quant_policy=KVQuantPolicy(
        supported=False,
        disabled_reason="KV quantization is not supported for Step.",
    ),
    validation_status="experimental_contract_gated",
    status="experimental_contract_gated",
    notes=(
        "Step uses appended NextN layers and remains contract-gated for v1 UX.",
        "Step reasoning uses the step3p5 parser with low/medium/high effort controls.",
        "Do not inherit Qwen tuning or KV quantization controls.",
    ),
)


DEEPSEEK_MTP_DESCRIPTOR = BackendDescriptor(
    backend_id="deepseek_mtp",
    architecture_id="deepseek-v3-mtp",
    model_family="deepseek",
    display_name="DeepSeek native MTP",
    artifact_layout="single_mlx_folder_native_mtp",
    runtime_capabilities=NATIVE_CONTRACT_DESCRIPTOR.runtime_capabilities,
    sampler_defaults=NATIVE_CONTRACT_DESCRIPTOR.sampler_defaults,
    reasoning_codec=NATIVE_CONTRACT_DESCRIPTOR.reasoning_codec,
    draft_semantics=NATIVE_CONTRACT_DESCRIPTOR.draft_semantics,
    uses_external_assistant=False,
    uses_draft_lm_head=True,
    hidden_variant="post_norm",
    tune_policy=NATIVE_CONTRACT_DESCRIPTOR.tune_policy,
    kv_quant_policy=KVQuantPolicy(
        supported=False,
        disabled_reason="KV quantization is not supported for DeepSeek.",
    ),
    validation_status="native_contract_gated",
    status="experimental_contract_gated",
)


DEEPSEEK_V4_DSPARK_DESCRIPTOR = replace(
    DEEPSEEK_MTP_DESCRIPTOR,
    backend_id="deepseek_v4_dspark",
    architecture_id="deepseek-v4-dspark-dflash2",
    display_name="DeepSeek V4 DSpark through DFlash2",
    artifact_layout="split_mia_tp1_target_plus_k64_draft",
    runtime_capabilities=(
        "target_logits",
        "dspark_k5",
        "dflash2_scheduler",
        "mia_nvfp4_stock432_target_kv",
        "mia_nvfp4_stock432_dspark_kv",
        "fp32_exact_speculative_sampling",
        "bf16_wide_numerics_reported",
    ),
    sampler_defaults=SamplerDefaults(temperature=0.0, top_p=1.0, top_k=0),
    reasoning_codec=ReasoningCodec(
        parser="none",
        display_name="No verified reasoning parser",
        default_mode="off",
        supported=False,
        modes=(),
        history_policy="visible_content_only",
    ),
    draft_semantics=DraftSemantics(
        request_field="depth",
        display_label="DSpark future tokens",
        default=5,
        minimum=5,
        maximum=5,
        unit="depth",
    ),
    uses_draft_lm_head=False,
    hidden_variant="target_taps_40_41_42",
    mtp_history_policy="cycle",
    tune_policy=TunePolicy(
        supported=False,
        unsupported_reason="DSpark Phase 1 owns one fixed DFlash2 K5 lane.",
    ),
    kv_quant_policy=KVQuantPolicy(
        supported=False,
        disabled_reason=(
            "DSpark owns Mia stock432 NVFP4 target and draft K/V from offset zero."
        ),
    ),
    context_window_policy=ContextWindowPolicy(
        maximum=384_000,
        default=384_000,
        source="sealed_mia_engine_plan",
    ),
    required_chat_template_profile="tokenizer",
    allows_chat_template_path=False,
    validation_status="real_checkpoint_guarded",
    status="real_checkpoint_guarded",
    profile_policy="backend-aware-sustained",
    notes=(
        "The existing DFlash2 scheduler owns verification and acceptance.",
        "The artifact is the pinned split Mia/Sero TP1 target plus packaged K64 draft.",
        "Target and all three DSpark caches use stock432 NVFP4 records from offset zero.",
        "Default bf16 wide-forward divergence is reported; the fp32 causality gate is exact.",
    ),
)


GLM_MTP_DESCRIPTOR = BackendDescriptor(
    backend_id="glm_mtp",
    architecture_id="glm4-moe-mtp",
    model_family="glm",
    display_name="GLM native MTP",
    artifact_layout="single_mlx_folder_native_mtp",
    runtime_capabilities=NATIVE_CONTRACT_DESCRIPTOR.runtime_capabilities,
    sampler_defaults=NATIVE_CONTRACT_DESCRIPTOR.sampler_defaults,
    reasoning_codec=NATIVE_CONTRACT_DESCRIPTOR.reasoning_codec,
    draft_semantics=NATIVE_CONTRACT_DESCRIPTOR.draft_semantics,
    uses_external_assistant=False,
    uses_draft_lm_head=True,
    hidden_variant="post_norm",
    tune_policy=NATIVE_CONTRACT_DESCRIPTOR.tune_policy,
    kv_quant_policy=KVQuantPolicy(
        supported=False,
        disabled_reason="KV quantization is not supported for GLM.",
    ),
    validation_status="native_contract_gated",
    status="experimental_contract_gated",
)


HY_V3_MTP_DESCRIPTOR = BackendDescriptor(
    backend_id="hy_v3_mtp",
    architecture_id="hy-v3-mtp",
    model_family="hy",
    display_name="Hy3 native MTP",
    artifact_layout="single_mlx_folder_native_mtp",
    runtime_capabilities=NATIVE_CONTRACT_DESCRIPTOR.runtime_capabilities,
    # Official Tencent inference settings (tencent/Hy3 generation_config.json):
    # temperature 0.9, top_p 1.0, top_k disabled. Do NOT inherit the Qwen
    # 0.6/0.95/20 coding sampler; Hy3's CoT measurably degrades at greedy/cold
    # settings (community reports on the release thread).
    sampler_defaults=SamplerDefaults(temperature=0.9, top_p=1.0, top_k=0),
    # Hy3 emits <think:opensource>...</think:opensource> (suffixed single
    # tokens). The qwen3-style splitter handles suffixed spellings.
    reasoning_codec=ReasoningCodec(
        parser="qwen3",
        display_name="Hy3 think tags",
        default_mode="auto",
    ),
    draft_semantics=NATIVE_CONTRACT_DESCRIPTOR.draft_semantics,
    uses_external_assistant=False,
    uses_draft_lm_head=True,
    hidden_variant="pre_norm",
    tune_policy=TunePolicy(
        supported=False,
        unsupported_reason="Tune is supported for Qwen 3.5, Qwen 3.6, and Gemma 4 MTPLX models only.",
    ),
    kv_quant_policy=KVQuantPolicy(
        supported=False,
        disabled_reason="KV quantization is not supported for Hy3.",
    ),
    validation_status="experimental_contract_gated",
    status="experimental_contract_gated",
    notes=(
        "Single appended NextN MoE layer (depth 1); 192-expert sigmoid top-8 "
        "routing with expert bias; draft input eh_proj(concat[enorm(embedding), "
        "hnorm(pre-final-norm hidden)]).",
        "Official sampler: temperature 0.9, top_p 1.0, top_k off.",
    ),
)


GEMMA4_TARGET_DISTRIBUTION_POLICY = TargetDistributionPolicy(
    modes=("gemma4_target_prefix_exact",),
    default_mode="gemma4_target_prefix_exact",
    default_window_size=None,
    exact=True,
    mode_metadata=(
        TargetDistributionMode(
            name="gemma4_target_prefix_exact",
            exact=True,
            product=True,
            status="product_candidate",
            notes=(
                "MLX-VLM-style target-sampled prefix verification. The target "
                "model samples the verifier rows; assistant tokens are accepted "
                "only while they match those target samples."
            ),
            aliases=(
                "target_prefix",
                "prefix_walk",
                "sampled_prefix",
                "mlx_vlm_prefix_walk",
                "gemma4_sparse_head",
                "sparse_head",
                "row_lazy_logits",
                "row_lazy_hidden",
                "dense_logits_topk_debug",
                "fused_logits_topk",
                "certified_topk",
                "batched_logits_debug",
            ),
        ),
    ),
    status="runtime_runnable_qa_pending",
    telemetry_fields=(
        "verify_target_distribution_time_s",
        "target_distribution_mode",
        "target_distribution_materialized_rows",
        "target_distribution_materialized_windows",
        "target_distribution_share",
        "next_hidden_eval_time_s",
    ),
    notes=(
        "Gemma uses target-sampled prefix verification; the old p/q sparse-head oracle is not the product path.",
    ),
)


GEMMA4_ASSISTANT_DESCRIPTOR = BackendDescriptor(
    backend_id="gemma4_assistant",
    architecture_id="gemma4-assistant-mtp",
    model_family="gemma4",
    display_name="Gemma 4 assistant MTP",
    artifact_layout="assistant_pair_bundle",
    runtime_capabilities=(
        "target_logits",
        "target_pre_norm_hidden",
        "external_assistant_shared_kv",
        "dense_tied_assistant_lm_head",
        "exact_speculative_sampling",
        "backend_target_distribution_policy",
        "gemma4_channel_reasoning",
        "requires_generation_thread_affinity",
    ),
    sampler_defaults=SamplerDefaults(temperature=1.0, top_p=0.95, top_k=64),
    reasoning_codec=ReasoningCodec(
        parser="gemma4",
        display_name="Gemma channel thinking",
        default_mode="auto",
    ),
    draft_semantics=DraftSemantics(
        request_field="draft_block_size",
        display_label="Draft block",
        default=4,
        minimum=2,
        maximum=8,
        unit="block",
    ),
    uses_external_assistant=True,
    uses_draft_lm_head=False,
    hidden_variant="gemma4_pre_norm",
    mtp_history_policy="assistant_shared_kv",
    target_distribution_policy=GEMMA4_TARGET_DISTRIBUTION_POLICY,
    tune_policy=TunePolicy(
        supported=True,
        control_field="draft_block_size",
        candidates=(
            "AR",
            "Block 2",
            "Block 3",
            "Block 4",
            "Block 5",
            "Block 6",
            "Block 7",
            "Block 8",
        ),
        supported_families=("qwen3_5", "qwen3_6", "gemma4"),
    ),
    kv_quant_policy=KVQuantPolicy(
        supported=False,
        disabled_reason="KV quantization is not supported for Gemma.",
    ),
    context_window_policy=ContextWindowPolicy(
        maximum=262_144,
        default=262_144,
        source="gemma4_config",
    ),
    validation_status="runtime_runnable_qa_pending",
    status="runtime_runnable_qa_pending",
    profile_policy="backend-aware-sustained",
    notes=(
        "Gemma uses an official external assistant and target shared KV.",
        "Gemma's verifier policy is backend-owned because its large tied LM head has different batching tradeoffs than native Qwen MTP heads.",
        "The public speed claim is QA-gated; metadata alone is not verification.",
    ),
)


DESCRIPTORS_BY_BACKEND_ID: dict[str, BackendDescriptor] = {
    QWEN3_NEXT_DESCRIPTOR.backend_id: QWEN3_NEXT_DESCRIPTOR,
    LAGUNA_AR_DESCRIPTOR.backend_id: LAGUNA_AR_DESCRIPTOR,
    MLX_LM_AR_DESCRIPTOR.backend_id: MLX_LM_AR_DESCRIPTOR,
    NATIVE_CONTRACT_DESCRIPTOR.backend_id: NATIVE_CONTRACT_DESCRIPTOR,
    GEMMA4_ASSISTANT_DESCRIPTOR.backend_id: GEMMA4_ASSISTANT_DESCRIPTOR,
    STEP3P5_MTP_DESCRIPTOR.backend_id: STEP3P5_MTP_DESCRIPTOR,
    DEEPSEEK_MTP_DESCRIPTOR.backend_id: DEEPSEEK_MTP_DESCRIPTOR,
    DEEPSEEK_V4_DSPARK_DESCRIPTOR.backend_id: DEEPSEEK_V4_DSPARK_DESCRIPTOR,
    GLM_MTP_DESCRIPTOR.backend_id: GLM_MTP_DESCRIPTOR,
    HY_V3_MTP_DESCRIPTOR.backend_id: HY_V3_MTP_DESCRIPTOR,
    "mimo_mtp": NATIVE_CONTRACT_DESCRIPTOR,
    "nemotron_h_mtp": NATIVE_CONTRACT_DESCRIPTOR,
}


def backend_descriptors() -> tuple[BackendDescriptor, ...]:
    """Return the unique backend descriptors used by public surfaces."""

    out: list[BackendDescriptor] = []
    seen: set[str] = set()
    for descriptor in DESCRIPTORS_BY_BACKEND_ID.values():
        if descriptor.backend_id in seen:
            continue
        seen.add(descriptor.backend_id)
        out.append(descriptor)
    return tuple(out)


def descriptor_for_architecture_id(value: str | None) -> BackendDescriptor | None:
    arch_id = str(value or "").strip()
    if not arch_id:
        return None
    if arch_id in {"glm-moe-dsa-mtp", "glm4-moe-lite-mtp"}:
        return GLM_MTP_DESCRIPTOR
    for descriptor in backend_descriptors():
        if descriptor.architecture_id == arch_id:
            return descriptor
    return None


def _inspection_dict(inspection: dict[str, Any] | None) -> dict[str, Any]:
    return inspection if isinstance(inspection, dict) else {}


def _compatibility_dict(inspection: dict[str, Any] | None) -> dict[str, Any]:
    data = _inspection_dict(inspection)
    value = data.get("compatibility")
    return value if isinstance(value, dict) else {}


def _text_markers(model_ref: str | None, inspection: dict[str, Any] | None) -> str:
    data = _inspection_dict(inspection)
    compatibility = _compatibility_dict(inspection)
    parts = [
        model_ref,
        data.get("model_dir"),
        data.get("runtime_model"),
        data.get("architecture"),
        data.get("model_type"),
        data.get("recommended_backend"),
        data.get("mtp_arch"),
        compatibility.get("recommended_backend"),
        compatibility.get("arch_id"),
    ]
    return " ".join(str(part or "") for part in parts).lower()


# Stock Qwen3 sizes collide with the 3.8 version token: in "qwen3-8b" or
# "qwen3-80b" the digit-run ending in "b" right after the token is a
# parameter count, not a version, and must not claim the qwen3_8 family.
_QWEN3_8_MARKER = re.compile(r"qwen3[._-]?8(?!\d*b)")


def _explicit_qwen_family_marker(text: str) -> str | None:
    if _QWEN3_8_MARKER.search(text):
        return "qwen3_8"
    if "qwen3.6" in text or "qwen3_6" in text or "qwen36" in text or "qwen3-6" in text:
        return "qwen3_6"
    if "qwen3.5" in text or "qwen3_5" in text or "qwen3-5" in text:
        return "qwen3_5"
    return None


@lru_cache(maxsize=64)
def _artifact_family_texts(model_ref: str) -> tuple[str, str]:
    """Family markers the artifact carries about itself (issue #268).

    A model served from a renamed or symlinked directory has no family
    marker in its ref, and the shared qwen3_next descriptor cannot split
    3.5/3.6/3.8 — config.json says qwen3_5 for all of them. The artifact
    still knows what it is: the forge provenance in mtplx_runtime.json
    (source trunk, published repo) and the symlink-resolved path name the
    family, returned as ``(provenance, resolved_path)`` in that order of
    authority — what the artifact says outranks what the folder is called.
    Family is a behavior contract, not a first-party identity claim, so
    provenance matching is safe here — unlike the served-model-id lane,
    where fuzzy inference was deliberately removed (July 2026, issue #57).
    """
    resolved = ""
    parts: list[str] = []
    try:
        path = Path(model_ref).expanduser()
        if not path.exists():
            return "", ""
        resolved = str(path.resolve())
        runtime_json = path / "mtplx_runtime.json"
        if runtime_json.is_file():
            data = json.loads(runtime_json.read_text())
            if isinstance(data, dict):
                provenance = data.get("forge_provenance")
                provenance = provenance if isinstance(provenance, dict) else {}
                inputs = provenance.get("forge_inputs")
                inputs = inputs if isinstance(inputs, dict) else {}
                parts.extend(
                    str(value or "")
                    for value in (
                        data.get("public_model_id"),
                        data.get("served_model_id"),
                        data.get("model_id"),
                        data.get("published_to_hf"),
                        data.get("base_trunk"),
                        data.get("artifact_role"),
                        inputs.get("trunk_path"),
                        inputs.get("mtp_source_path"),
                    )
                )
    except Exception:
        pass
    return " ".join(part for part in parts if part).lower(), resolved.lower()


def model_family_from_inspection(
    inspection: dict[str, Any] | None = None,
    *,
    model_ref: str | None = None,
    descriptor: BackendDescriptor | None = None,
) -> str:
    text = _text_markers(model_ref, inspection)
    if model_ref:
        # The artifact outranks its folder name: forge provenance first,
        # then the symlink-resolved location, then the ref as spelled.
        provenance_text, resolved_path = _artifact_family_texts(str(model_ref))
        artifact_family = _explicit_qwen_family_marker(
            provenance_text
        ) or _explicit_qwen_family_marker(resolved_path)
        if artifact_family is not None:
            return artifact_family
    ref_family = _explicit_qwen_family_marker(str(model_ref or "").lower())
    if ref_family is not None:
        return ref_family
    backend_id = (
        str(getattr(descriptor, "backend_id", "") or "")
        if descriptor is not None
        else backend_id_from_inspection(inspection)
    )
    if backend_id == GEMMA4_ASSISTANT_DESCRIPTOR.backend_id or "gemma4" in text or "gemma-4" in text:
        return "gemma4"
    if backend_id == STEP3P5_MTP_DESCRIPTOR.backend_id or "step3p5" in text or "step3p7" in text or "step-3.7" in text:
        return "step"
    if backend_id == DEEPSEEK_MTP_DESCRIPTOR.backend_id or "deepseek" in text:
        return "deepseek"
    if backend_id == GLM_MTP_DESCRIPTOR.backend_id or "glm" in text:
        return "glm"
    if "lfm2" in text:
        return "lfm2"
    family = _explicit_qwen_family_marker(text)
    if family is not None:
        return family
    descriptor_family = getattr(descriptor, "model_family", None)
    if descriptor_family == "qwen":
        return "qwen3_6"
    if descriptor_family is not None and descriptor_family not in {"native-mtp", "qwen"}:
        return str(descriptor_family)
    return "unknown"


def tune_policy_for_model(
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
    descriptor: BackendDescriptor | None = None,
) -> TunePolicy:
    descriptor = descriptor or descriptor_from_inspection(inspection)
    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=descriptor,
    )
    if descriptor.backend_id == DEEPSEEK_V4_DSPARK_DESCRIPTOR.backend_id:
        return descriptor.tune_policy
    if family in {"qwen3_5", "qwen3_6"}:
        return TunePolicy(supported=True)
    if family == "qwen3_8":
        # Multi-step-trained MTP head wants deeper candidates, but the live
        # depth-4 lane killed the daemon on drop day (see
        # QWEN3_8_DRAFT_SEMANTICS). Cap tune at D3 so an app Tune run cannot
        # crash the daemon; restore AR..D6 with the deep-lane fix.
        return TunePolicy(
            supported=True,
            candidates=("AR", "D1", "D2", "D3"),
        )
    if family == "gemma4":
        return GEMMA4_ASSISTANT_DESCRIPTOR.tune_policy
    if family == "step":
        return STEP3P5_MTP_DESCRIPTOR.tune_policy
    return TunePolicy(
        supported=False,
        unsupported_reason="Tune is supported for Qwen 3.5, Qwen 3.6, Qwen 3.8, and Gemma 4 MTPLX models only.",
    )


def kv_quant_policy_for_model(
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
    descriptor: BackendDescriptor | None = None,
) -> KVQuantPolicy:
    descriptor = descriptor or descriptor_from_inspection(inspection)
    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=descriptor,
    )
    if descriptor.backend_id == DEEPSEEK_V4_DSPARK_DESCRIPTOR.backend_id:
        return descriptor.kv_quant_policy
    if family in {"qwen3_5", "qwen3_6", "qwen3_8"}:
        return QWEN3_NEXT_DESCRIPTOR.kv_quant_policy
    if family == "gemma4":
        return GEMMA4_ASSISTANT_DESCRIPTOR.kv_quant_policy
    if family == "step":
        return STEP3P5_MTP_DESCRIPTOR.kv_quant_policy
    if family == "glm":
        return GLM_MTP_DESCRIPTOR.kv_quant_policy
    if family == "deepseek":
        return DEEPSEEK_MTP_DESCRIPTOR.kv_quant_policy
    return KVQuantPolicy(supported=False)


def _context_window_from_inspection(inspection: dict[str, Any] | None) -> int | None:
    data = _inspection_dict(inspection)
    compatibility = _compatibility_dict(inspection)
    candidates: list[int] = []
    for source in (data, compatibility):
        for key in (
            "model_context_window",
            "max_context_window",
            "max_model_len",
            "context_window",
            "context_length",
            "model_max_length",
        ):
            value = source.get(key)
            if isinstance(value, int):
                candidates.append(value)
    sane = [value for value in candidates if 0 < value <= 1_048_576]
    return max(sane) if sane else None


def context_window_policy_for_model(
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
    descriptor: BackendDescriptor | None = None,
) -> ContextWindowPolicy:
    descriptor = descriptor or descriptor_from_inspection(inspection)
    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=descriptor,
    )
    if descriptor.backend_id == DEEPSEEK_V4_DSPARK_DESCRIPTOR.backend_id:
        return descriptor.context_window_policy
    if family in {"qwen3_5", "qwen3_6", "qwen3_8"}:
        base = QWEN3_NEXT_DESCRIPTOR.context_window_policy
    elif family == "gemma4":
        base = GEMMA4_ASSISTANT_DESCRIPTOR.context_window_policy
    elif family == "step":
        base = STEP3P5_MTP_DESCRIPTOR.context_window_policy
    elif family == "glm":
        base = GLM_MTP_DESCRIPTOR.context_window_policy
    elif family == "deepseek":
        base = DEEPSEEK_MTP_DESCRIPTOR.context_window_policy
    else:
        base = descriptor.context_window_policy
    return base.with_resolved_max(_context_window_from_inspection(inspection))


def reasoning_policy_for_model(
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
    descriptor: BackendDescriptor | None = None,
) -> ReasoningCodec:
    descriptor = descriptor or descriptor_from_inspection(inspection)
    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=descriptor,
    )
    if descriptor.backend_id == DEEPSEEK_V4_DSPARK_DESCRIPTOR.backend_id:
        return descriptor.reasoning_codec
    if family == "qwen3_8":
        return QWEN3_8_REASONING_CODEC
    if family in {"qwen3_5", "qwen3_6"}:
        return QWEN3_NEXT_DESCRIPTOR.reasoning_codec
    if family == "gemma4":
        return GEMMA4_ASSISTANT_DESCRIPTOR.reasoning_codec
    if family == "step":
        return STEP3P5_MTP_DESCRIPTOR.reasoning_codec
    if family == "glm":
        return GLM_MTP_DESCRIPTOR.reasoning_codec
    if family == "deepseek":
        return DEEPSEEK_MTP_DESCRIPTOR.reasoning_codec
    if family == "laguna":
        return LAGUNA_AR_DESCRIPTOR.reasoning_codec
    if family == "lfm2":
        return MLX_LM_AR_LFM2_DESCRIPTOR.reasoning_codec
    return ReasoningCodec(
        parser="none",
        display_name="No verified reasoning parser",
        default_mode="off",
        supported=False,
        modes=(),
        history_policy="visible_content_only",
    )


def model_controls_for_descriptor(
    descriptor: BackendDescriptor,
    *,
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=descriptor,
    )
    tune_policy = tune_policy_for_model(model_ref, inspection, descriptor)
    kv_policy = kv_quant_policy_for_model(model_ref, inspection, descriptor)
    reasoning_policy = reasoning_policy_for_model(model_ref, inspection, descriptor)
    context_policy = context_window_policy_for_model(model_ref, inspection, descriptor)
    sampler = sampler_defaults_for_model(model_ref, inspection, descriptor).to_dict()
    return {
        "schema_version": 1,
        "model_ref": model_ref,
        "model_family": family,
        "backend_id": descriptor.backend_id,
        "architecture_id": descriptor.architecture_id,
        "support_level": descriptor.status,
        "display_name": descriptor.display_name,
        "draft_control": draft_semantics_for_model(
            model_ref, inspection, descriptor
        ).to_dict(),
        "sampling": {
            **sampler,
            "family_default_reason": (
                "Gemma assistant sampler"
                if family == "gemma4"
                else (
                    "Qwen3.8 official thinking sampler"
                    if family == "qwen3_8"
                    else (
                        "Qwen coding sampler"
                        if family in {"qwen3_5", "qwen3_6"}
                        else f"{descriptor.display_name} sampler"
                    )
                )
            ),
        },
        "reasoning": reasoning_policy.to_dict(),
        "tune": tune_policy.to_dict(),
        "kv_quant": kv_policy.to_dict(),
        "context_window": context_policy.to_dict(),
    }


def descriptor_for_model(
    descriptor: BackendDescriptor,
    *,
    model_ref: str | None = None,
    inspection: dict[str, Any] | None = None,
) -> BackendDescriptor:
    """Resolve family policy for a model sharing a backend lane.

    Qwen3.8 deliberately reuses the qwen3_next runtime, but its official
    sampler, reasoning controls, and multi-step MTP range differ from the
    Qwen3.5/3.6 lane defaults.  Returning a descriptor view keeps server
    validation and health telemetry on the same contract as model_controls.
    """

    family = model_family_from_inspection(
        inspection,
        model_ref=model_ref,
        descriptor=descriptor,
    )
    if family != "qwen3_8":
        return descriptor
    return replace(
        descriptor,
        sampler_defaults=sampler_defaults_for_model(
            model_ref, inspection, descriptor
        ),
        reasoning_codec=reasoning_policy_for_model(
            model_ref, inspection, descriptor
        ),
        draft_semantics=draft_semantics_for_model(
            model_ref, inspection, descriptor
        ),
        tune_policy=tune_policy_for_model(model_ref, inspection, descriptor),
    )


def assistant_target_distribution_choices() -> tuple[str, ...]:
    """Return all descriptor-declared target-distribution modes for CLI parsers."""

    choices: list[str] = []
    seen: set[str] = set()
    for descriptor in backend_descriptors():
        if not descriptor.uses_external_assistant:
            continue
        for mode in descriptor.target_distribution_modes:
            if mode == "backend_default" or mode in seen:
                continue
            seen.add(mode)
            choices.append(mode)
        for mode in descriptor.target_distribution_policy.mode_metadata:
            for alias in mode.aliases:
                normalized = alias.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                choices.append(normalized)
    return tuple(choices)


def profile_payload_for_descriptor(
    descriptor: BackendDescriptor,
    profile_payload: dict[str, Any],
    *,
    profile_name: str,
    model_id: str | None = None,
    sampler: dict[str, Any] | None = None,
    draft_default: int | None = None,
) -> dict[str, Any]:
    """Apply backend-owned profile semantics without family branches.

    Public surfaces start from a product profile such as sustained or
    performance-cold.  Backends can then declare whether that profile is used
    unchanged or whether runtime policy is owned by the backend because the
    architecture has different draft/cache machinery.
    """

    payload = dict(profile_payload)
    payload["backend_id"] = descriptor.backend_id
    payload["architecture_id"] = descriptor.architecture_id
    if descriptor.profile_policy == "profile-owned":
        return payload

    if model_id:
        payload["model_id"] = str(model_id)
    payload["runtime_profile"] = f"{descriptor.backend_id}_{profile_name}"
    payload["summary"] = (
        f"{profile_name.replace('-', ' ').title()} profile through "
        f"{descriptor.display_name}: backend-owned sampler, draft control, "
        "draft machinery, and cache policy."
    )
    payload["caveats"] = list(payload.get("caveats") or []) + [
        "Backend policy is declared by the selected architecture, not inherited from another model family.",
    ]
    if not descriptor.uses_draft_lm_head:
        payload["draft_lm_head"] = None
    payload["draft_control"] = descriptor.draft_semantics.request_field
    payload["draft_unit"] = descriptor.draft_semantics.unit
    payload["draft_default"] = (
        descriptor.draft_semantics.clamp(draft_default)
        if draft_default is not None
        else descriptor.draft_semantics.default
    )
    if sampler is not None:
        payload["sampler"] = dict(sampler)
    if descriptor.target_distribution_modes != ("backend_default",):
        payload["target_distribution_policy"] = (
            descriptor.target_distribution_policy.to_dict()
        )
    return payload


def descriptor_for_backend_id(value: str | None) -> BackendDescriptor:
    backend_id = str(value or "").strip()
    if not backend_id:
        return QWEN3_NEXT_DESCRIPTOR
    return DESCRIPTORS_BY_BACKEND_ID.get(backend_id, NATIVE_CONTRACT_DESCRIPTOR)


def backend_id_from_inspection(inspection: dict[str, Any] | None) -> str:
    data = inspection or {}
    compatibility = data.get("compatibility") if isinstance(data.get("compatibility"), dict) else {}
    backend = data.get("recommended_backend") or compatibility.get("recommended_backend")
    if backend:
        return str(backend)
    arch_id = data.get("mtp_arch") or compatibility.get("arch_id")
    descriptor = descriptor_for_architecture_id(arch_id)
    if descriptor is not None:
        return descriptor.backend_id
    return QWEN3_NEXT_DESCRIPTOR.backend_id


def descriptor_from_inspection(inspection: dict[str, Any] | None) -> BackendDescriptor:
    return descriptor_for_backend_id(backend_id_from_inspection(inspection))


def _runtime_is_lfm2(runtime: Any) -> bool:
    model_args = getattr(getattr(runtime, "model", None), "args", None)
    model_type = str(getattr(model_args, "model_type", "") or "").lower()
    if model_type.startswith("lfm2"):
        return True
    return "lfm2" in str(getattr(runtime, "model_path", "") or "").lower()


def descriptor_from_runtime(runtime: Any, args: Any | None = None) -> BackendDescriptor:
    runtime_backend = getattr(runtime, "backend_id", None)
    requested_backend = getattr(args, "backend_id", None) if args is not None else None
    if runtime_backend == DEEPSEEK_V4_DSPARK_DESCRIPTOR.backend_id:
        from mtplx.deepseek_v4_mia_engine import (
            MIA_CONTEXT_CAPACITY,
            MIA_TARGET_PHYSICAL_CAPACITY,
            MiaDeepseekV4EnginePlan,
        )

        plan = getattr(getattr(runtime, "model", None), "_mia_engine_plan", None)
        if (
            not isinstance(plan, MiaDeepseekV4EnginePlan)
            or int(plan.context_capacity_tokens) != MIA_CONTEXT_CAPACITY
            or int(plan.target_physical_capacity_tokens)
            != MIA_TARGET_PHYSICAL_CAPACITY
        ):
            raise RuntimeError(
                "deepseek_v4_dspark runtime has no sealed Mia engine plan"
            )
        descriptor = DEEPSEEK_V4_DSPARK_DESCRIPTOR
    elif requested_backend == DEEPSEEK_V4_DSPARK_DESCRIPTOR.backend_id:
        raise RuntimeError(
            "sealed DSpark backend requires runtime.backend_id=deepseek_v4_dspark "
            "and its installed Mia engine plan"
        )
    elif runtime_backend:
        descriptor = descriptor_for_backend_id(str(runtime_backend))
    elif bool(getattr(runtime, "gemma4_external_assistant", False)):
        return GEMMA4_ASSISTANT_DESCRIPTOR
    else:
        descriptor = descriptor_for_backend_id(
            str(requested_backend) if requested_backend else None
        )
    if (
        descriptor.backend_id == MLX_LM_AR_DESCRIPTOR.backend_id
        and _runtime_is_lfm2(runtime)
    ):
        return MLX_LM_AR_LFM2_DESCRIPTOR
    return descriptor


def _arg_value(args: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                return value
    return default


def assistant_model_from_args(args: Any, default: str | None = None) -> str | None:
    value = _arg_value(args, ASSISTANT_MODEL_ATTRS, default)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def target_distribution_mode_from_args(
    args: Any,
    descriptor: BackendDescriptor | None = None,
) -> str | None:
    value = _arg_value(args, TARGET_DISTRIBUTION_MODE_ATTRS)
    if value is not None:
        text = str(value)
        if descriptor is not None:
            normalized = text.strip().lower().replace("-", "_")
            for mode in descriptor.target_distribution_policy.mode_metadata:
                names = (mode.name, *mode.aliases)
                if normalized in {item.strip().lower().replace("-", "_") for item in names}:
                    return mode.name
        return text
    if descriptor is not None and descriptor.default_target_distribution_mode != "backend_default":
        return descriptor.default_target_distribution_mode
    return None


def draft_control_from_args(args: Any, descriptor: BackendDescriptor) -> int:
    value = getattr(args, descriptor.draft_semantics.request_field, None)
    if value is None and descriptor.draft_semantics.unit == "block":
        value = getattr(args, "gemma_draft_block_size", None)
    if value is None and descriptor.draft_semantics.request_field == "depth":
        value = getattr(args, "depth", descriptor.draft_semantics.default)
    if value is None:
        value = descriptor.draft_semantics.default
    return int(value)


def set_assistant_model_arg(args: Any, value: str | None) -> None:
    for name in ASSISTANT_MODEL_ATTRS:
        setattr(args, name, value)


def set_target_distribution_mode_arg(args: Any, value: str | None) -> None:
    for name in TARGET_DISTRIBUTION_MODE_ATTRS:
        setattr(args, name, value)


def set_draft_control_arg(args: Any, descriptor: BackendDescriptor, value: int) -> None:
    value = int(value)
    setattr(args, descriptor.draft_semantics.request_field, value)
    if descriptor.draft_semantics.unit == "block":
        setattr(args, "gemma_draft_block_size", value)
    args.depth = value


def sync_backend_arg_aliases(args: Any) -> None:
    """Keep old Gemma-only option names as aliases for generic backend knobs."""

    assistant = assistant_model_from_args(args)
    if assistant is not None:
        set_assistant_model_arg(args, assistant)
    draft_block = _arg_value(args, ("draft_block_size", "gemma_draft_block_size"))
    if draft_block is not None:
        setattr(args, "draft_block_size", int(draft_block))
        setattr(args, "gemma_draft_block_size", int(draft_block))
    mode = _arg_value(args, TARGET_DISTRIBUTION_MODE_ATTRS)
    if mode is not None:
        set_target_distribution_mode_arg(args, str(mode))


def sampler_defaults_from_inspection(
    inspection: dict[str, Any] | None,
) -> dict[str, Any]:
    data = inspection or {}
    sampler = data.get("recommended_sampler")
    if not isinstance(sampler, dict):
        pair = data.get("gemma4_pair")
        if isinstance(pair, dict):
            sampler = pair.get("sampler")
    if isinstance(sampler, dict):
        try:
            return {
                "temperature": float(sampler["temperature"]),
                "top_p": float(sampler["top_p"]),
                "top_k": int(sampler["top_k"]),
            }
        except (KeyError, TypeError, ValueError):
            pass
    return descriptor_from_inspection(data).sampler_defaults.to_dict()


def draft_default_from_inspection(inspection: dict[str, Any] | None) -> int:
    descriptor = descriptor_from_inspection(inspection)
    data = inspection or {}
    pair = data.get("gemma4_pair")
    benchmark = pair.get("benchmark") if isinstance(pair, dict) else {}
    if isinstance(benchmark, dict):
        benchmark_surface = str(
            benchmark.get("prompt_encoding")
            or benchmark.get("surface")
            or benchmark.get("benchmark_surface")
            or ""
        ).lower()
        if descriptor.backend_id == "gemma4_assistant" and benchmark_surface not in {
            "chat",
            "server_chat",
            "web_chat",
            "openai_chat",
        }:
            return descriptor.draft_semantics.default
        try:
            return descriptor.draft_semantics.clamp(int(benchmark["best_block_size"]))
        except (KeyError, TypeError, ValueError):
            pass
    compatibility = data.get("compatibility") if isinstance(data.get("compatibility"), dict) else {}
    contract = compatibility.get("runtime_contract") if isinstance(compatibility, dict) else None
    if isinstance(contract, dict):
        try:
            if descriptor.backend_id != STEP3P5_MTP_DESCRIPTOR.backend_id:
                return descriptor.draft_semantics.clamp(int(contract.get("mtp_depth_max")))
        except (TypeError, ValueError):
            pass
    return descriptor.draft_semantics.default
