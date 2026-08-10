"""Architecture compatibility registry and runtime-contract checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mtplx.profiles import DEFAULT_PROFILE_NAME, PROFILE_CHOICES, resolve_profile_name


RUNTIME_CONTRACT_FILE = "mtplx_runtime.json"
SUPPORTED_ARCH_IDS = {
    "laguna-s-2.1-ar",
    "deepseek-v4",
    "qwen3-next-mtp",
    "deepseek-v3-mtp",
    "glm-moe-dsa-mtp",
    "glm4-moe-mtp",
    "glm4-moe-lite-mtp",
    "mimo-mtp",
    "nemotron-h-mtp",
    "gemma4-assistant-mtp",
    "step3p5-mtp",
    "hy-v3-mtp",
}

TIER_VERIFIED = "verified"
TIER_FAMILY_COMPATIBLE_UNVERIFIED = "family-compatible-unverified"
TIER_ARCH_COMPATIBLE_UNVERIFIED = "architecture-compatible-but-unverified"
TIER_INCOMPATIBLE_ARCHITECTURE = "incompatible-architecture"
TIER_NO_MTP = "no-MTP"
TIER_AR_ONLY = "AR-only"

EXIT_VERIFIED = 0
EXIT_NO_MTP = 2
EXIT_UNVERIFIED = 3
EXIT_INCOMPATIBLE_ARCHITECTURE = 4
BLOCKING_RUNTIME_STATUSES = {
    "candidate",
    "candidate_build_only_benchmark_pending",
    "candidate_build_only",
    "failed",
    "fail",
    "failing",
    "pending",
    "blocked",
    "blocker",
    "needs_repair",
    "needs_reverify",
    "unverified",
}
BLOCKING_RUNTIME_STATUS_PREFIXES = (
    "candidate",
    "fail",
    "pending",
    "blocked",
    "blocker",
    "needs",
    "unverified",
)
BLOCKING_SPEED_VERDICTS = {
    "mtp_acceptance_collapsed",
    "no_mtp_depth_beat_ar",
    "no_quality_passed_mtp_depth_beat_ar",
}


class ModelCompatibilityError(RuntimeError):
    exit_code = 1


class UnverifiedArchitectureError(ModelCompatibilityError):
    exit_code = EXIT_UNVERIFIED


class IncompatibleArchitectureError(ModelCompatibilityError):
    exit_code = EXIT_INCOMPATIBLE_ARCHITECTURE


class NoMTPError(ModelCompatibilityError):
    exit_code = EXIT_NO_MTP


@dataclass(frozen=True)
class ArchitectureSupport:
    arch_id: str
    display_name: str
    family: str
    backend: str | None
    support_level: str
    runtime_compatibility: str
    can_run_verified: bool = False
    aliases: tuple[str, ...] = ()
    config_markers: tuple[str, ...] = ("mtp_num_hidden_layers", "num_nextn_predict_layers")
    family_gate: str = "none"
    references: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "display_name": self.display_name,
            "family": self.family,
            "backend": self.backend,
            "support_level": self.support_level,
            "runtime_compatibility": self.runtime_compatibility,
            "can_run_verified": self.can_run_verified,
            "aliases": list(self.aliases),
            "config_markers": list(self.config_markers),
            "family_gate": self.family_gate,
            "references": list(self.references),
            "notes": self.notes,
        }


ARCHITECTURE_CATALOG: dict[str, ArchitectureSupport] = {
    "laguna-s-2.1-ar": ArchitectureSupport(
        arch_id="laguna-s-2.1-ar",
        display_name="Laguna-S-2.1 oQ4e (MLX)",
        family="laguna",
        backend="laguna_ar",
        support_level="verified-native-ar-only",
        runtime_compatibility="native-ar-only",
        can_run_verified=True,
        aliases=("laguna", "LagunaForCausalLM"),
        config_markers=(),
        family_gate="laguna-s-2.1-mlx-4bit-geometry",
        references=(
            "https://huggingface.co/mlx-community/Laguna-S-2.1-oQ4e",
            "https://huggingface.co/pipenetwork/Laguna-S-2.1-MLX-4bit/blob/5544297f819d50330bc3616dd15cbc7edb598b2f/laguna.py",
        ),
        notes=(
            "Target-only AR runtime for the exact mlx-community Laguna-S-2.1-oQ4e "
            "checkpoint; the artifact has no native MTP head and must be loaded "
            "with mtp=False."
        ),
    ),
    "muse-glimmer-ar": ArchitectureSupport(
        arch_id="muse-glimmer-ar",
        display_name="Muse-Glimmer-30B text tower (MLX)",
        family="muse_glimmer",
        backend="muse_glimmer_ar",
        support_level="experimental-native-ar-only",
        runtime_compatibility="native-ar-only",
        can_run_verified=True,
        aliases=(
            "muse_glimmer",
            "muse_glimmer_text",
            "MuseGlimmerForConditionalGeneration",
        ),
        config_markers=(),
        family_gate="none",
        references=(
            "https://huggingface.co/meta-models/Muse-Glimmer-30B",
            "https://github.com/ggml-org/llama.cpp/blob/master/src/models/muse-glimmer.cpp",
        ),
        notes=(
            "Target-only AR runtime for the Muse-Glimmer text tower (Gemma-family: "
            "sigmoid gated attention, parameter-free QK-norm with qk_scale_factor, "
            "NoPE on the global layers, sandwich norms, final-logit softcap). Loaded "
            "via the vendored mlx_lm model class registered by muse_glimmer_patch; "
            "no native MTP head, so it runs mtp=False like any AR checkpoint."
        ),
    ),
    "qwen3-next-mtp": ArchitectureSupport(
        arch_id="qwen3-next-mtp",
        display_name="Qwen3.6 / Qwen3-Next / Qwen3.5 MTP",
        family="qwen",
        backend="qwen3_next",
        support_level="verified-native",
        runtime_compatibility="native",
        can_run_verified=True,
        aliases=("qwen3_5_mtp", "qwen3_6_mtp", "qwen3-next", "qwen3_5"),
        family_gate="qwen-mtp-sidecar-or-embedded",
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/qwen3_next_mtp.py",
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/qwen3_5_mtp.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/qwen3_next.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/qwen3_5.py",
        ),
        notes="Product-verified default backend; this remains the only promoted shipping runtime.",
    ),
    "deepseek-v3-mtp": ArchitectureSupport(
        arch_id="deepseek-v3-mtp",
        display_name="DeepSeek V3 / V3.2 MTP",
        family="deepseek",
        backend="deepseek_mtp",
        support_level="experimental-native-contract-gated",
        runtime_compatibility="native-contract-gated",
        can_run_verified=True,
        aliases=("deepseek_mtp", "deepseek_v3", "deepseek_v32"),
        family_gate="appended-layer-mtp-markers",
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/config/speculative.py",
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/deepseek_mtp.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/deepseek_v3.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/deepseek_v32.py",
        ),
        notes="Experimental native backend is present for verified-contract models; exactness and performance still need per-model QA before promotion.",
    ),
    "glm-moe-dsa-mtp": ArchitectureSupport(
        arch_id="glm-moe-dsa-mtp",
        display_name="GLM MoE DSA MTP",
        family="glm",
        backend="deepseek_mtp",
        support_level="experimental-native-contract-gated",
        runtime_compatibility="native-contract-gated",
        can_run_verified=True,
        aliases=("glm_moe_dsa", "glm_moe_dsa_mtp"),
        family_gate="appended-layer-mtp-markers",
        references=(
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/glm_moe_dsa.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/deepseek_v32.py",
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/deepseek_mtp.py",
        ),
        notes=(
            "GLM MoE DSA is an mlx-lm DeepSeek V3.2-derived architecture; "
            "MTPLX routes verified-contract artifacts through the DeepSeek MTP backend."
        ),
    ),
    "deepseek-v4": ArchitectureSupport(
        arch_id="deepseek-v4",
        display_name="DeepSeek-V4-Flash (MLX)",
        family="deepseek",
        backend="deepseek_v4",
        support_level="experimental-native-ar-only",
        runtime_compatibility="native-ar-only",
        can_run_verified=True,
        # Keep aliases minimal: "deepseek_v4" is a substring of "deepseek_v4_mtp",
        # so a longer alias here would out-sort (and wrongly capture) the
        # deepseek-v4-mtp split config. Detection of the AR checkpoint works via
        # this alias / the model_type; the MTP-split entry keeps priority for its
        # own longer "deepseek_v4_mtp" marker.
        aliases=("deepseek_v4",),
        config_markers=(),
        family_gate="deepseek-v4-mlx",
        references=(
            "https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash",
            "https://huggingface.co/mlx-community/DeepSeek-V4-Flash-4bit",
            "REFERENCES:TOOLS/DeepSeek-V4-Flash/inference/model.py",
        ),
        notes=(
            "Native MLX loader (mtplx.models.deepseek_v4) for DeepSeek-V4-Flash. "
            "V4 adds Hyper-Connections, Compressed-Sparse-Attention, grouped "
            "output-LoRA, and hash layers over V3.2. This is also the runnable "
            "V4 MTP arch: the draft block (mtp.0.*) binds through the ordinary "
            "load path when the checkpoint ships it, and the speculative lane "
            "drives it through mtplx.generation like every other native MTP "
            "backend. The published mlx-community conversions drop the block "
            "while still declaring num_nextn_predict_layers, which is the case "
            "the runtime's degrade-to-autoregressive branch covers -- those keep "
            "running target-only (mtp=False). The runtime_compatibility field "
            "stays 'native-ar-only' because it is what routes a checkpoint with "
            "no draft head to the AR-only verdict; an MTP-bearing artifact is "
            "resolved dynamically by the family gate instead."
        ),
    ),
    "deepseek-v4-mtp": ArchitectureSupport(
        arch_id="deepseek-v4-mtp",
        display_name="DeepSeek V4 MTP (split checkpoint)",
        family="deepseek",
        backend="deepseek_v4_mtp",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("deepseek_v4_mtp",),
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/deepseek_v4_mtp.py",
        ),
        notes=(
            "vLLM's SPLIT V4 MTP layout: a standalone checkpoint carrying only "
            "the draft module (model_type deepseek_v4_mtp / "
            "DeepseekV4MTPForCausalLM), which vLLM separated out from DeepSeek "
            "V3. MTPLX now has a real V4 draft-head runtime, but it is not this "
            "artifact shape -- it loads a MERGED directory whose ordinary shards "
            "carry mtp.0.* beside the trunk, which detects as arch_id "
            "'deepseek-v4'. This entry stays pending because MTPLX has no loader "
            "that assembles a target from two separate repos, not because the "
            "backend is missing."
        ),
    ),
    "glm4-moe-mtp": ArchitectureSupport(
        arch_id="glm4-moe-mtp",
        display_name="GLM-4 MoE MTP",
        family="glm",
        backend="glm_mtp",
        support_level="experimental-native-contract-gated",
        runtime_compatibility="native-contract-gated",
        can_run_verified=True,
        aliases=("glm4_moe_mtp", "glm4_moe"),
        family_gate="appended-layer-mtp-markers",
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/glm4_moe_mtp.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/glm4_moe.py",
        ),
        notes="Experimental native backend is present for verified-contract GLM-4 MoE MTP artifacts; real-checkpoint exactness/performance QA is still required before promotion.",
    ),
    "glm4-moe-lite-mtp": ArchitectureSupport(
        arch_id="glm4-moe-lite-mtp",
        display_name="GLM-4 MoE Lite MTP",
        family="glm",
        backend="glm_mtp",
        support_level="experimental-native-contract-gated",
        runtime_compatibility="native-contract-gated",
        can_run_verified=True,
        aliases=("glm4_moe_lite_mtp", "glm4_moe_lite"),
        family_gate="appended-layer-mtp-markers",
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/glm4_moe_lite_mtp.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/glm4_moe_lite.py",
        ),
        notes="Experimental native backend is present for verified-contract GLM-4 MoE Lite MTP artifacts; the Lite MLA cache/key rewrite is handled separately from plain GLM-4 MoE.",
    ),
    "glm-ocr-mtp": ArchitectureSupport(
        arch_id="glm-ocr-mtp",
        display_name="GLM OCR MTP",
        family="glm",
        backend="glm_ocr_mtp",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("glm_ocr_mtp", "glm_ocr"),
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/glm_ocr_mtp.py",
        ),
        notes="Recognized for compatibility reporting; not a target runtime backend yet.",
    ),
    "minimax-m2-mtp": ArchitectureSupport(
        arch_id="minimax-m2-mtp",
        display_name="MiniMax M2 MTP",
        family="minimax",
        backend="minimax_m2",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=(
            "minimax_m2",
            "minimax_m2_5",
            "minimax_m25",
            "minimax_m2_6",
            "minimax_m26",
            "MiniMaxM2ForCausalLM",
            "MiniMaxM25ForCausalLM",
            "MiniMaxM26ForCausalLM",
        ),
        config_markers=("num_mtp_modules", "num_nextn_predict_layers", "mtp_num_hidden_layers"),
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/minimax_m2.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/minimax.py",
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/llama_eagle3.py",
        ),
        notes=(
            "MiniMax M2-family speculative support in vLLM is EAGLE3-style "
            "auxiliary-hidden drafting, not the native MTP proposer contract MTPLX "
            "uses for Qwen/DeepSeek/GLM/MiMo/Nemotron-H."
        ),
    ),
    "mimo-mtp": ArchitectureSupport(
        arch_id="mimo-mtp",
        display_name="MiMo MTP",
        family="mimo",
        backend="mimo_mtp",
        support_level="experimental-native-contract-gated",
        runtime_compatibility="native-contract-gated",
        can_run_verified=True,
        aliases=("mimo_mtp", "MiMoForCausalLM", "mimo"),
        family_gate="mimo-layer0-mtp-markers",
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/mimo_mtp.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/mimo.py",
        ),
        notes="Experimental native backend is present for verified-contract MiMo artifacts; vLLM's proposer only supports one-token draft depth today.",
    ),
    "gemma-mtp": ArchitectureSupport(
        arch_id="gemma-mtp",
        display_name="Gemma MTP marker variant",
        family="gemma",
        backend="gemma_mtp",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("gemma3", "gemma4", "gemma_mtp"),
        references=(
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/gemma4.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/gemma3.py",
        ),
        notes="Mainline Gemma configs are no-MTP unless an explicit MTP marker is present.",
    ),
    "gemma4-assistant-mtp": ArchitectureSupport(
        arch_id="gemma4-assistant-mtp",
        display_name="Gemma 4 assistant-backed MTP",
        family="gemma",
        backend="gemma4_assistant",
        support_level="runtime-runnable-qa-pending",
        runtime_compatibility="assistant-pair-native",
        can_run_verified=True,
        aliases=("gemma4_assistant", "Gemma4AssistantPair", "gemma4-assistant-mtp"),
        config_markers=("mtplx_pair.json", "assistant_pair_bundle"),
        family_gate="target-assistant-bundle",
        references=(
            "https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/",
            "https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/speculative/drafters/gemma4_assistant",
            "https://huggingface.co/google/gemma-4-31B-it-assistant",
        ),
        notes=(
            "Gemma 4 uses an external assistant drafter sharing target KV. "
            "The runnable artifact is the bundle root with mtplx_pair.json, "
            "target/, and assistant/; target-only folders are not runnable."
        ),
    ),
    "ernie-mtp": ArchitectureSupport(
        arch_id="ernie-mtp",
        display_name="ERNIE MoE MTP",
        family="ernie",
        backend="ernie_mtp",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("ernie_mtp", "ernie4_5_moe"),
        references=("REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/ernie_mtp.py",),
    ),
    "nemotron-h-mtp": ArchitectureSupport(
        arch_id="nemotron-h-mtp",
        display_name="Nemotron-H MTP",
        family="nemotron",
        backend="nemotron_h_mtp",
        support_level="experimental-native-contract-gated",
        runtime_compatibility="native-contract-gated",
        can_run_verified=True,
        aliases=("nemotron_h_mtp", "nemotron_h", "nemotron_h_puzzle"),
        family_gate="nemotron-h-pattern-mtp-markers",
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/nemotron_h_mtp.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/nemotron_h.py",
        ),
        notes=(
            "Experimental native backend for vLLM-style Nemotron-H MTP predictor "
            "artifacts. Supports the one-step MTP path whose pattern contains "
            "attention/MoE blocks only."
        ),
    ),
    "exaone-moe-mtp": ArchitectureSupport(
        arch_id="exaone-moe-mtp",
        display_name="EXAONE MoE MTP",
        family="exaone",
        backend="exaone_moe_mtp",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("exaone_moe_mtp", "exaone_moe"),
        references=("REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/exaone_moe_mtp.py",),
    ),
    "exaone4-5-mtp": ArchitectureSupport(
        arch_id="exaone4-5-mtp",
        display_name="EXAONE 4.5 MTP",
        family="exaone",
        backend="exaone4_5_mtp",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("exaone4_5_mtp", "exaone4_5"),
        references=("REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/exaone4_5_mtp.py",),
    ),
    "longcat-flash-mtp": ArchitectureSupport(
        arch_id="longcat-flash-mtp",
        display_name="LongCat Flash MTP",
        family="longcat",
        backend="longcat_flash_mtp",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("longcat_flash_mtp", "longcat_flash"),
        references=("REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/longcat_flash_mtp.py",),
    ),
    "pangu-ultra-moe-mtp": ArchitectureSupport(
        arch_id="pangu-ultra-moe-mtp",
        display_name="Pangu Ultra MoE MTP",
        family="pangu",
        backend="pangu_ultra_moe_mtp",
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("pangu_ultra_moe_mtp", "pangu_ultra_moe", "openpangu_mtp", "openpangu"),
        references=("REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/openpangu_mtp.py",),
    ),
    "step3p5-mtp": ArchitectureSupport(
        arch_id="step3p5-mtp",
        display_name="Step-3.5 / Step-3.7-Flash MTP",
        family="step",
        backend="step3p5_mtp",
        support_level="experimental-native-contract-gated",
        runtime_compatibility="native-contract-gated",
        can_run_verified=True,
        aliases=(
            "step3p5_mtp",
            "step3p5",
            "step3p7",
            "step3p7_mtp",
            "step3p7forconditionalgeneration",
            "step3p5forcausallm",
        ),
        family_gate="appended-layer-mtp-markers",
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/step3p5_mtp.py",
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/step3p5.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/step3p5.py",
        ),
        notes=(
            "Step ships 3 distinct appended NextN layers with dense MLP, GQA, "
            "zero-centered norms, and shared heads. MTPLX routes verified-contract "
            "artifacts through the dedicated Step MTP injector, which feeds the "
            "draft layers the trunk pre-final-norm hidden state."
        ),
    ),
    "hy-v3-mtp": ArchitectureSupport(
        arch_id="hy-v3-mtp",
        display_name="HY V3 MTP",
        family="hy",
        backend="hy_v3_mtp",
        support_level="experimental-native-contract-gated",
        runtime_compatibility="native-contract-gated",
        can_run_verified=True,
        aliases=("hy_v3_mtp", "hy_v3"),
        family_gate="appended-layer-mtp-markers",
        references=(
            "REFERENCES:TOOLS/vllm-official-main/vllm/model_executor/models/hy_v3_mtp.py",
            "REFERENCES:TOOLS/mlx-lm/mlx_lm/models/hy_v3.py",
        ),
        notes=(
            "Hy3 ships one appended NextN layer with its own 192-expert MoE, "
            "eh_proj over concat[enorm(embedding), hnorm(hidden)], and shared "
            "embeddings/head. The draft consumes the POST-final-norm trunk "
            "hidden (measured: teacher-forced agreement 0.773 post vs 0.387 "
            "pre on real code). Injection grafts the head from the standard "
            "appended-layer checkpoint; a native mlx-lm surface, when it "
            "lands, is bound instead."
        ),
    ),
    "generic-mtp": ArchitectureSupport(
        arch_id="generic-mtp",
        display_name="Generic MTP marker",
        family="generic",
        backend=None,
        support_level="recognized-backend-pending",
        runtime_compatibility="recognized-backend-pending",
        aliases=("mtp", "nextn"),
        references=("REFERENCES:TOOLS/vllm-official-main/vllm/config/speculative.py",),
        notes="Fallback for explicit MTP/nextn configs whose family is not mapped yet.",
    ),
}


def architecture_catalog() -> list[dict[str, Any]]:
    return [support.to_dict() for support in ARCHITECTURE_CATALOG.values()]


def architecture_support_for(arch_id: str | None) -> ArchitectureSupport | None:
    if not arch_id:
        return None
    key = str(arch_id).strip().lower()
    if key in ARCHITECTURE_CATALOG:
        return ARCHITECTURE_CATALOG[key]
    normalized = key.replace("_", "-")
    if normalized in ARCHITECTURE_CATALOG:
        return ARCHITECTURE_CATALOG[normalized]
    for support in ARCHITECTURE_CATALOG.values():
        aliases = {alias.lower().replace("_", "-") for alias in support.aliases}
        if normalized in aliases:
            return support
    return None


@dataclass(frozen=True)
class RuntimeContract:
    mtplx_version: str
    arch_id: str
    mtp_depth_max: int
    recommended_profile: str
    exactness_baseline: dict[str, Any]
    verified_on: dict[str, Any]
    recommended_draft_lm_head: dict[str, Any] | None = None
    recommended_draft_sampler: dict[str, Any] | None = None
    mtp_contract: dict[str, Any] | None = None
    runtime_env_overrides: dict[str, str] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeContract":
        missing = [
            key
            for key in (
                "mtplx_version",
                "arch_id",
                "mtp_depth_max",
                "recommended_profile",
                "exactness_baseline",
                "verified_on",
            )
            if key not in data
        ]
        if missing:
            raise ValueError(f"runtime contract missing required keys: {', '.join(missing)}")
        raw_profile = str(data["recommended_profile"])
        profile = resolve_profile_name(raw_profile)
        if profile not in PROFILE_CHOICES:
            raise ValueError(f"runtime contract has invalid recommended_profile: {profile}")
        depth = int(data["mtp_depth_max"])
        if depth <= 0:
            raise ValueError("runtime contract mtp_depth_max must be positive")
        recommended_draft_lm_head = None
        if data.get("recommended_draft_lm_head") is not None:
            from mtplx.draft_lm_head import normalize_draft_lm_head_spec

            recommended_draft_lm_head = normalize_draft_lm_head_spec(
                data.get("recommended_draft_lm_head")
            )
        recommended_draft_sampler = None
        if data.get("recommended_draft_sampler") is not None:
            from mtplx.draft_sampling import normalize_draft_sampler_spec

            recommended_draft_sampler = normalize_draft_sampler_spec(
                data.get("recommended_draft_sampler")
            )
        mtp_contract = None
        if data.get("mtp_contract") is not None:
            from mtplx.mtp_patch import MTPContract

            mtp_contract = MTPContract().with_metadata(
                data.get("mtp_contract"),
                preserve_explicit=False,
            ).to_dict()
        runtime_env_overrides = None
        if data.get("runtime_env_overrides") is not None:
            from mtplx.profiles import normalize_runtime_env_overrides

            runtime_env_overrides = normalize_runtime_env_overrides(
                data.get("runtime_env_overrides")
            )
        return cls(
            mtplx_version=str(data["mtplx_version"]),
            arch_id=str(data["arch_id"]),
            mtp_depth_max=depth,
            recommended_profile=profile,
            exactness_baseline=dict(data["exactness_baseline"]),
            verified_on=dict(data["verified_on"]),
            recommended_draft_lm_head=recommended_draft_lm_head,
            recommended_draft_sampler=recommended_draft_sampler,
            mtp_contract=mtp_contract,
            runtime_env_overrides=runtime_env_overrides,
            raw=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        out = {
            "mtplx_version": self.mtplx_version,
            "arch_id": self.arch_id,
            "mtp_depth_max": self.mtp_depth_max,
            "recommended_profile": self.recommended_profile,
            "exactness_baseline": self.exactness_baseline,
            "verified_on": self.verified_on,
        }
        if self.recommended_draft_lm_head is not None:
            out["recommended_draft_lm_head"] = dict(self.recommended_draft_lm_head)
        if self.recommended_draft_sampler is not None:
            out["recommended_draft_sampler"] = dict(self.recommended_draft_sampler)
        if self.mtp_contract is not None:
            out["mtp_contract"] = dict(self.mtp_contract)
        if self.runtime_env_overrides is not None:
            out["runtime_env_overrides"] = dict(self.runtime_env_overrides)
        return out


@dataclass(frozen=True)
class CompatibilityVerdict:
    tier: str
    arch_id: str | None
    supported: bool
    recognized: bool
    can_run: bool
    exit_code: int
    message: str
    recommended_backend: str | None = None
    recommended_profile: str | None = None
    runtime_contract: RuntimeContract | None = None
    runtime_contract_path: str | None = None
    runtime_contract_error: str | None = None
    unsafe_force_required: bool = False
    unverified_model: bool = False
    mtp_supported: str = "no"
    runtime_compatibility: str = "unsupported"
    support_level: str = "unsupported"
    support_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "arch_id": self.arch_id,
            "supported": self.supported,
            "recognized": self.recognized,
            "can_run": self.can_run,
            "exit_code": self.exit_code,
            "message": self.message,
            "recommended_backend": self.recommended_backend,
            "recommended_profile": self.recommended_profile,
            "runtime_contract": (
                self.runtime_contract.to_dict() if self.runtime_contract else None
            ),
            "runtime_contract_path": self.runtime_contract_path,
            "runtime_contract_error": self.runtime_contract_error,
            "unsafe_force_required": self.unsafe_force_required,
            "unverified_model": self.unverified_model,
            "mtp_supported": self.mtp_supported,
            "runtime_compatibility": self.runtime_compatibility,
            "support_level": self.support_level,
            "support_notes": self.support_notes,
        }


def _contract_path(model_dir: Path) -> Path:
    return model_dir / RUNTIME_CONTRACT_FILE


def load_runtime_contract(model_dir: Path | str) -> tuple[RuntimeContract | None, str | None]:
    path = _contract_path(Path(model_dir))
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RuntimeContract.from_dict(data), None
    except Exception as exc:
        return None, str(exc)


def _text(value: Any) -> str:
    return str(value or "").lower().replace("-", "_")


def _compact(value: str) -> str:
    return _text(value).replace("_", "").replace(" ", "")


def _alias_matches(combined: str, alias: str) -> bool:
    alias_text = _text(alias)
    if not alias_text:
        return False
    return alias_text in combined or _compact(alias_text) in _compact(combined)


def _support_alias_matches(support: ArchitectureSupport, combined: str) -> bool:
    aliases = (support.arch_id, *support.aliases)
    return any(_alias_matches(combined, alias) for alias in aliases)


def _detect_arch_id(inspection: Any) -> str | None:
    architecture = _text(getattr(inspection, "architecture", None))
    model_type = _text(getattr(inspection, "model_type", None))
    combined = f"{architecture} {model_type}"
    has_config_mtp = int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0) > 0
    has_explicit_mtp = has_config_mtp or "mtp" in combined or "nextn" in combined

    qwen_support = ARCHITECTURE_CATALOG["qwen3-next-mtp"]
    if _support_alias_matches(qwen_support, combined):
        return "qwen3-next-mtp"

    supports = [
        support
        for support in ARCHITECTURE_CATALOG.values()
        if support.arch_id not in {"qwen3-next-mtp", "generic-mtp"}
    ]
    supports.sort(
        key=lambda row: max(
            (len(_compact(alias)) for alias in (row.arch_id, *row.aliases)),
            default=0,
        ),
        reverse=True,
    )
    for support in supports:
        if _support_alias_matches(support, combined) and (
            has_explicit_mtp or support.runtime_compatibility == "native-ar-only"
        ):
            return support.arch_id
    if "mtp" in combined or "nextn" in combined:
        return "generic-mtp"
    return None


def _has_mtp_markers(inspection: Any) -> bool:
    mtp = getattr(inspection, "mtp", None)
    return bool(
        int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0) > 0
        or (mtp is not None and bool(getattr(mtp, "exists", False)))
    )


def _passes_verified_runtime_gate(arch_id: str, inspection: Any, tensor_gate: bool) -> bool:
    return _passes_family_runtime_gate(arch_id, inspection, tensor_gate)


def _runtime_contract_blocker(contract: RuntimeContract) -> str | None:
    exactness_blocker = _runtime_evidence_blocker(
        contract.exactness_baseline,
        section_name="exactness_baseline",
    )
    if exactness_blocker:
        return exactness_blocker
    raw = contract.raw if isinstance(contract.raw, dict) else {}
    speed_evidence = raw.get("speed_evidence")
    speed_blocker = _runtime_evidence_blocker(
        speed_evidence,
        section_name="speed_evidence",
    )
    if speed_blocker:
        return speed_blocker
    if isinstance(speed_evidence, dict):
        verdict = _text(speed_evidence.get("verdict"))
        if verdict in BLOCKING_SPEED_VERDICTS:
            return f"speed_evidence verdict is {speed_evidence.get('verdict')}"
    return None


def _runtime_evidence_blocker(value: Any, *, section_name: str) -> str | None:
    if not isinstance(value, dict):
        return None
    if value.get("public_release_blocker") is True:
        return f"{section_name} is marked public_release_blocker"
    status = _text(value.get("status"))
    if status and (
        status in BLOCKING_RUNTIME_STATUSES
        or status.startswith(tuple(f"{prefix}_" for prefix in BLOCKING_RUNTIME_STATUS_PREFIXES))
    ):
        return f"{section_name} status is {value.get('status')}"
    return None


def _weight_keys(inspection: Any) -> tuple[str, ...]:
    return tuple(str(key) for key in (getattr(inspection, "weight_keys", ()) or ()))


def _is_sidecar_mtp_key(key: str) -> bool:
    return key.startswith(("mtp.", "language_model.mtp."))


def _has_numbered_qwen_moe_expert_key(key: str) -> bool:
    marker = ".mlp.experts."
    if marker not in key:
        return False
    suffix = key.split(marker, 1)[1]
    expert_id = suffix.split(".", 1)[0]
    return expert_id.isdigit()


def _qwen_moe_body_layout_blocker(inspection: Any) -> str | None:
    model_type = _text(getattr(inspection, "model_type", None))
    architecture = _text(getattr(inspection, "architecture", None))
    descriptor = f"{model_type} {architecture}"
    if "qwen3_5_moe" not in descriptor and "qwen3_5moe" not in descriptor:
        return None
    body_keys = tuple(
        key for key in _weight_keys(inspection) if not _is_sidecar_mtp_key(key)
    )
    if not body_keys:
        return None
    has_numbered_experts = any(
        _has_numbered_qwen_moe_expert_key(key) for key in body_keys
    )
    has_switch_mlp = any(".mlp.switch_mlp." in key for key in body_keys)
    if has_numbered_experts and not has_switch_mlp:
        return (
            "Qwen MoE body uses numbered mlp.experts.* tensors, but the current "
            "native MLX backend expects the packed switch_mlp layout. Re-run "
            "Forge repair/conversion before marking this artifact runnable."
        )
    return None


def _runtime_body_layout_blocker(arch_id: str | None, inspection: Any) -> str | None:
    if arch_id == "qwen3-next-mtp":
        return _qwen_moe_body_layout_blocker(inspection)
    return None


_APPENDED_LAYER_MARKER_SUFFIXES = (
    "enorm.weight",
    "hnorm.weight",
    "eh_proj.weight",
    "shared_head.norm.weight",
    "shared_head.head.weight",
    "shared_head.output.weight",
)


def _has_marker_under_prefixes(
    keys: tuple[str, ...],
    prefixes: tuple[str, ...],
    suffixes: tuple[str, ...],
    substrings: tuple[str, ...] = (),
) -> bool:
    for key in keys:
        for prefix in prefixes:
            if not key.startswith(prefix):
                continue
            suffix = key.removeprefix(prefix)
            if suffix.endswith(suffixes) or any(marker in suffix for marker in substrings):
                return True
    return False


def _has_all_suffixes_under_prefixes(
    keys: tuple[str, ...],
    prefixes: tuple[str, ...],
    suffixes: tuple[str, ...],
) -> bool:
    for suffix in suffixes:
        if not any(key.startswith(prefix) and key.removeprefix(prefix).endswith(suffix) for key in keys for prefix in prefixes):
            return False
    return True


_HY_V3_MTP_MARKER_SUFFIXES = (
    "enorm.weight",
    "hnorm.weight",
    "eh_proj.weight",
    "final_layernorm.weight",
)


def _passes_hy_v3_gate(inspection: Any) -> bool:
    """Hy3's appended MTP block ships in one of two layouts: repacked
    checkpoints put it directly under an ``mtp.`` prefix (``mtp.enorm.weight``,
    ``mtp.eh_proj.weight``, ``mtp.layer.*`` — verified against the shipped
    `hy3-demolition-mlx-*-mtp` indexes), while tencent-native exports keep the
    canonical appended-layer form ``model.layers.{num_hidden_layers}.*``
    (``...enorm.weight``, ``...self_attn.*``, ``...mlp.*``). Neither uses the
    ``mtp.layers.{idx}.`` nesting DeepSeek/GLM/Step share, so this stays a
    dedicated gate instead of `_passes_appended_layer_gate`."""
    keys = _weight_keys(inspection)
    if not keys:
        return False
    count = int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0)
    if count <= 0:
        return False
    if _has_marker_under_prefixes(
        keys,
        ("mtp.",),
        _HY_V3_MTP_MARKER_SUFFIXES,
        ("mtp.layer.",),
    ):
        return True
    start = int(getattr(inspection, "num_hidden_layers", 0) or 0)
    if start <= 0:
        return False
    return _has_marker_under_prefixes(
        keys,
        (f"model.layers.{start}.",),
        _HY_V3_MTP_MARKER_SUFFIXES,
        ("self_attn.", "mlp."),
    )


def _passes_appended_layer_gate(inspection: Any) -> bool:
    keys = _weight_keys(inspection)
    if not keys:
        return False
    start = int(getattr(inspection, "num_hidden_layers", 0) or 0)
    count = int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0)
    if start <= 0 or count <= 0:
        return False
    for local_idx in range(count):
        layer_idx = start + local_idx
        prefixes = (
            f"model.layers.{layer_idx}.",
            f"mtp.layers.{local_idx}.",
            f"layers.{local_idx}.",
        )
        if not _has_marker_under_prefixes(
            keys,
            prefixes,
            _APPENDED_LAYER_MARKER_SUFFIXES,
            ("mtp_block.",),
        ):
            return False
    return True


def _passes_mimo_layer_gate(inspection: Any) -> bool:
    keys = _weight_keys(inspection)
    if not keys:
        return False
    start = int(getattr(inspection, "num_hidden_layers", 0) or 0)
    count = int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0)
    if start <= 0 or count <= 0:
        return False
    # The current MiMo proposer path is one-token MTP; gate the layer it can
    # actually execute while keeping deeper configured layers unpromoted.
    prefixes = (
        "model.mtp_layers.0.",
        f"model.mtp_layers.{start}.",
        f"model.layers.{start}.",
        "mtp.layers.0.",
        "layers.0.",
    )
    return _has_marker_under_prefixes(
        keys,
        prefixes,
        (
            "token_layernorm.weight",
            "hidden_layernorm.weight",
            "input_proj.weight",
            "final_layernorm.weight",
        ),
        ("mtp_block.",),
    )


def _passes_nemotron_h_gate(inspection: Any) -> bool:
    keys = _weight_keys(inspection)
    if not keys:
        return False
    if int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0) != 1:
        return False
    pattern = str(getattr(inspection, "mtp_pattern", None) or "")
    if not pattern or not set(pattern).issubset({"*", "E"}):
        return False
    start = int(getattr(inspection, "num_hidden_layers", 0) or 0)
    if start <= 0:
        return False
    physical_layers = len(pattern)
    for local_idx in range(physical_layers):
        prefixes = (
            f"mtp.layers.{local_idx}.",
            f"layers.{local_idx}.",
            f"model.layers.{start + local_idx}.",
            f"backbone.layers.{start + local_idx}.",
        )
        has_layer_body = False
        for key in keys:
            for prefix in prefixes:
                if not key.startswith(prefix):
                    continue
                suffix = key.removeprefix(prefix)
                if suffix == "norm.weight" or suffix.startswith("mixer."):
                    has_layer_body = True
                    break
            if has_layer_body:
                break
        if not has_layer_body:
            return False
    first_prefixes = (
        "mtp.layers.0.",
        "layers.0.",
        f"model.layers.{start}.",
        f"backbone.layers.{start}.",
    )
    if not _has_all_suffixes_under_prefixes(
        keys,
        first_prefixes,
        ("enorm.weight", "hnorm.weight", "eh_proj.weight"),
    ):
        return False
    last_idx = physical_layers - 1
    last_prefixes = (
        f"mtp.layers.{last_idx}.",
        f"layers.{last_idx}.",
        f"model.layers.{start + last_idx}.",
        f"backbone.layers.{start + last_idx}.",
    )
    return _has_marker_under_prefixes(keys, last_prefixes, ("final_layernorm.weight",))


def _passes_deepseek_v4_gate(inspection: Any) -> bool:
    """DeepSeek-V4-Flash MLX artifact: model_type deepseek_v4 (or the
    DeepseekV4ForCausalLM architecture).  The mlx-community conversion drops the
    MTP block, so this is a target-only AR gate with no MTP-marker requirement."""
    model_type = _text(getattr(inspection, "model_type", None))
    architecture = _compact(_text(getattr(inspection, "architecture", None)))
    return model_type == "deepseek_v4" or "deepseekv4forcausallm" in architecture


def _passes_family_runtime_gate(arch_id: str, inspection: Any, tensor_gate: bool) -> bool:
    if arch_id == "deepseek-v4":
        return _passes_deepseek_v4_gate(inspection)
    if arch_id == "laguna-s-2.1-ar":
        return bool(
            getattr(inspection, "laguna_s_2_1_mlx_4bit_match", False)
            and getattr(
                inspection,
                "laguna_s_2_1_artifacts_complete",
                False,
            )
        )
    if arch_id == "qwen3-next-mtp":
        return bool(
            tensor_gate
            and int(getattr(inspection, "mtp_num_hidden_layers", 0) or 0) > 0
        )
    if arch_id == "hy-v3-mtp":
        return _passes_hy_v3_gate(inspection)
    if arch_id in {
        "deepseek-v3-mtp",
        "glm-moe-dsa-mtp",
        "glm4-moe-mtp",
        "glm4-moe-lite-mtp",
        "step3p5-mtp",
    }:
        return _passes_appended_layer_gate(inspection)
    if arch_id == "mimo-mtp":
        return _passes_mimo_layer_gate(inspection)
    if arch_id == "nemotron-h-mtp":
        return _passes_nemotron_h_gate(inspection)
    if arch_id == "gemma4-assistant-mtp":
        return (
            _text(getattr(inspection, "model_type", None)) == "gemma4_pair"
            and isinstance(getattr(inspection, "gemma4_pair", None), dict)
        )
    if arch_id == "muse-glimmer-ar":
        # Plain target-only AR text tower; the vendored mlx_lm loader handles it
        # like any dense AR checkpoint, so recognition is sufficient.
        return True
    return False


def compatibility_for_inspection(inspection: Any) -> CompatibilityVerdict:
    model_dir = Path(getattr(inspection, "model_dir", "."))
    contract_data = getattr(inspection, "runtime_contract_data", None)
    contract_error = getattr(inspection, "runtime_contract_error", None)
    if contract_data is not None:
        try:
            contract = RuntimeContract.from_dict(dict(contract_data))
            contract_error = None
        except Exception as exc:
            contract = None
            contract_error = str(exc)
    else:
        contract, local_contract_error = load_runtime_contract(model_dir)
        contract_error = contract_error or local_contract_error
    detected_arch_id = _detect_arch_id(inspection)
    has_mtp = _has_mtp_markers(inspection)
    mtp_artifact = getattr(inspection, "mtp", None)
    tensor_gate = bool(getattr(mtp_artifact, "passes_tensor_gate", False))
    mtp_artifact_exists = bool(getattr(mtp_artifact, "exists", False))
    contract_path = getattr(inspection, "runtime_contract_path", None)
    if not contract_path:
        contract_path = str(_contract_path(model_dir)) if _contract_path(model_dir).exists() else None
    body_blocker = _runtime_body_layout_blocker(detected_arch_id, inspection)
    if body_blocker:
        support = architecture_support_for(detected_arch_id)
        return CompatibilityVerdict(
            tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
            arch_id=detected_arch_id,
            supported=False,
            recognized=support is not None,
            can_run=False,
            exit_code=EXIT_UNVERIFIED,
            message=body_blocker,
            recommended_backend=(support.backend if support else None),
            recommended_profile=(
                contract.recommended_profile if contract is not None else DEFAULT_PROFILE_NAME
            ),
            runtime_contract=contract,
            runtime_contract_path=contract_path,
            runtime_contract_error=contract_error,
            unsafe_force_required=False,
            unverified_model=True,
            mtp_supported="partial" if has_mtp else "no",
            runtime_compatibility="invalid-base-tensor-layout",
            support_level="native-backend-invalid-base-tensors",
            support_notes=(support.notes if support else None),
        )

    if contract is not None:
        arch_id = contract.arch_id
        support = architecture_support_for(arch_id)
        blocker = _runtime_contract_blocker(contract)
        if blocker:
            # Contract evidence (exactness baseline, speed verdicts) is a
            # label, never a load gate: the model is architecturally
            # runnable, so it runs as unverified and the verdict explains
            # why the verified badge is withheld. Refusals are reserved
            # for artifacts that physically cannot execute.
            return CompatibilityVerdict(
                tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
                arch_id=arch_id,
                supported=False,
                recognized=support is not None,
                can_run=True,
                exit_code=EXIT_UNVERIFIED,
                message=(
                    "Runtime contract is not verified: "
                    f"{blocker}. The model runs as unverified; regenerate "
                    "the contract with Forge to restore the verified badge."
                ),
                recommended_backend=(support.backend if support else None),
                recommended_profile=contract.recommended_profile,
                runtime_contract=contract,
                runtime_contract_path=contract_path,
                runtime_contract_error=contract_error,
                unsafe_force_required=False,
                unverified_model=True,
                mtp_supported="partial" if has_mtp else "no",
                runtime_compatibility="runtime-contract-unverified",
                support_level=(
                    "native-backend-needs-contract-repair"
                    if support is not None
                    else "unsupported"
                ),
                support_notes=(support.notes if support else None),
            )
        if (
            arch_id in SUPPORTED_ARCH_IDS
            and has_mtp
            and _passes_verified_runtime_gate(arch_id, inspection, tensor_gate)
        ):
            return CompatibilityVerdict(
                tier=TIER_VERIFIED,
                arch_id=arch_id,
                supported=True,
                recognized=True,
                can_run=True,
                exit_code=EXIT_VERIFIED,
                message="Verified MTPLX runtime contract found.",
                recommended_backend=(support.backend if support else None),
                recommended_profile=contract.recommended_profile,
                runtime_contract=contract,
                runtime_contract_path=contract_path,
                mtp_supported="yes",
                runtime_compatibility=(support.runtime_compatibility if support else "native"),
                support_level=(support.support_level if support else "verified-native"),
                support_notes=(support.notes if support else None),
            )
        if arch_id not in SUPPORTED_ARCH_IDS:
            if support is not None:
                return CompatibilityVerdict(
                    tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
                    arch_id=support.arch_id,
                    supported=False,
                    recognized=True,
                    can_run=False,
                    exit_code=EXIT_UNVERIFIED,
                    message=(
                        f"{support.display_name} runtime contract detected and "
                        "recognized, but MTPLX does not yet have a native MLX "
                        "runtime backend for this family."
                    ),
                    recommended_backend=support.backend,
                    runtime_contract=contract,
                    runtime_contract_path=contract_path,
                    mtp_supported="recognized" if has_mtp else "partial",
                    runtime_compatibility=support.runtime_compatibility,
                    support_level=support.support_level,
                    support_notes=support.notes,
                    unverified_model=True,
                )
            return CompatibilityVerdict(
                tier=TIER_INCOMPATIBLE_ARCHITECTURE,
                arch_id=arch_id,
                supported=False,
                recognized=False,
                can_run=False,
                exit_code=EXIT_INCOMPATIBLE_ARCHITECTURE,
                message=(
                    f"{arch_id} runtime contract detected; not supported in "
                    "v0.2.0. Planned for a later backend."
                ),
                runtime_contract=contract,
                runtime_contract_path=contract_path,
                mtp_supported="partial" if has_mtp else "no",
                runtime_compatibility="unsupported",
            )
        return CompatibilityVerdict(
            tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
            arch_id=arch_id,
            supported=False,
            recognized=True,
            can_run=False,
            exit_code=EXIT_UNVERIFIED,
            message=(
                "Runtime contract exists but local MTP artifact inspection did not "
                "pass; refusing to run without repair."
            ),
            recommended_backend=(support.backend if support else "qwen3_next"),
            recommended_profile=contract.recommended_profile,
            runtime_contract=contract,
            runtime_contract_path=contract_path,
            runtime_contract_error=contract_error,
            unsafe_force_required=True,
            unverified_model=True,
            mtp_supported="partial",
            runtime_compatibility="needs-grafting",
            support_level="native-backend-needs-contract-repair",
            support_notes=(support.notes if support else None),
        )

    if contract_error:
        support = architecture_support_for(detected_arch_id)
        return CompatibilityVerdict(
            tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
            arch_id=detected_arch_id,
            supported=False,
            recognized=support is not None,
            can_run=False,
            exit_code=EXIT_UNVERIFIED,
            message=f"Invalid {RUNTIME_CONTRACT_FILE}: {contract_error}",
            recommended_backend=(support.backend if support else None),
            runtime_contract_path=contract_path,
            runtime_contract_error=contract_error,
            unsafe_force_required=detected_arch_id == "qwen3-next-mtp",
            unverified_model=True,
            mtp_supported="partial" if has_mtp else "no",
            runtime_compatibility=(
                "needs-grafting"
                if detected_arch_id == "qwen3-next-mtp"
                else (support.runtime_compatibility if support else "unsupported")
            ),
            support_level=(support.support_level if support else "unsupported"),
            support_notes=(support.notes if support else None),
        )

    if detected_arch_id == "qwen3-next-mtp":
        support = architecture_support_for(detected_arch_id)
        marker_text = (
            "Qwen3-Next MTP markers detected"
            if has_mtp
            else "Qwen3-Next architecture detected"
        )
        if support is not None and _passes_family_runtime_gate(detected_arch_id, inspection, tensor_gate):
            return CompatibilityVerdict(
                tier=TIER_FAMILY_COMPATIBLE_UNVERIFIED,
                arch_id=detected_arch_id,
                supported=True,
                recognized=True,
                can_run=True,
                exit_code=EXIT_VERIFIED,
                message=(
                    f"{marker_text}; native MTP tensors match the supported "
                    "Qwen family layout. No mtplx_runtime.json exactness "
                    "baseline is present, so runs are marked unverified until "
                    "a first-load smoke baseline is recorded."
                ),
                recommended_backend="qwen3_next",
                recommended_profile=DEFAULT_PROFILE_NAME,
                unsafe_force_required=False,
                unverified_model=True,
                mtp_supported="yes",
                runtime_compatibility="native-family-gated",
                support_level="native-family-auto-smoke",
                support_notes=(support.notes if support else None),
            )
        if not mtp_artifact_exists:
            return CompatibilityVerdict(
                tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
                arch_id=detected_arch_id,
                supported=False,
                recognized=True,
                can_run=False,
                exit_code=EXIT_UNVERIFIED,
                message=(
                    f"{marker_text}, but this folder does not contain runnable "
                    "Qwen MTP tensors. mtplx_runtime.json is optional metadata; "
                    "the blocker is missing MTP weights. Use a complete model with "
                    "mtp.safetensors or embedded mtp.* / language_model.mtp.* "
                    "weights, or build and verify one from its original source with "
                    "Forge. MTPLX cannot safely attach an arbitrary sidecar: matching "
                    "tensor shapes do not prove it was trained for this trunk."
                ),
                recommended_backend="qwen3_next",
                recommended_profile=DEFAULT_PROFILE_NAME,
                unsafe_force_required=False,
                unverified_model=True,
                mtp_supported="no",
                runtime_compatibility="missing-mtp-weights",
                support_level="native-backend-missing-mtp-weights",
                support_notes=(support.notes if support else None),
            )
        return CompatibilityVerdict(
            tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
            arch_id=detected_arch_id,
            supported=False,
            recognized=True,
            can_run=False,
            exit_code=EXIT_UNVERIFIED,
            message=(
                f"{marker_text}, and an MTP artifact is present, but its tensor "
                "layout does not match the Qwen native MTP runtime gate. "
                "mtplx_runtime.json is optional metadata; repair or regenerate "
                "the MTP sidecar/embedded weights so the tensor gate passes."
            ),
            recommended_backend="qwen3_next",
            recommended_profile=DEFAULT_PROFILE_NAME,
            unsafe_force_required=False,
            unverified_model=True,
            mtp_supported="partial",
            runtime_compatibility="invalid-mtp-tensor-layout",
            support_level="native-backend-invalid-mtp-tensors",
            support_notes=(support.notes if support else None),
        )

    support = architecture_support_for(detected_arch_id)
    if support is not None and has_mtp:
        family_gate = _passes_family_runtime_gate(
            support.arch_id,
            inspection,
            tensor_gate,
        )
        if support.can_run_verified and family_gate:
            return CompatibilityVerdict(
                tier=TIER_FAMILY_COMPATIBLE_UNVERIFIED,
                arch_id=support.arch_id,
                supported=True,
                recognized=True,
                can_run=True,
                exit_code=EXIT_VERIFIED,
                message=(
                    f"{support.display_name} MTP markers and tensor layout "
                    "match a supported native backend. No mtplx_runtime.json "
                    "exactness baseline is present, so runs are marked "
                    "unverified until a first-load smoke baseline is recorded."
                ),
                recommended_backend=support.backend,
                recommended_profile=DEFAULT_PROFILE_NAME,
                unsafe_force_required=False,
                unverified_model=True,
                mtp_supported="yes",
                runtime_compatibility="native-family-gated",
                support_level="native-family-auto-smoke",
                support_notes=support.notes,
            )
        if support.can_run_verified:
            return CompatibilityVerdict(
                tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
                arch_id=support.arch_id,
                supported=False,
                recognized=True,
                can_run=False,
                exit_code=EXIT_UNVERIFIED,
                message=(
                    f"{support.display_name} markers recognized and a native "
                    "backend exists, but no verified mtplx_runtime.json contract "
                    "is present for this artifact."
                ),
                recommended_backend=support.backend,
                recommended_profile=DEFAULT_PROFILE_NAME,
                unverified_model=True,
                mtp_supported="recognized",
                runtime_compatibility="needs-contract",
                support_level=support.support_level,
                support_notes=support.notes,
            )
        return CompatibilityVerdict(
            tier=TIER_ARCH_COMPATIBLE_UNVERIFIED,
            arch_id=support.arch_id,
            supported=False,
            recognized=True,
            can_run=False,
            exit_code=EXIT_UNVERIFIED,
            message=(
                f"{support.display_name} MTP markers recognized, but MTPLX does "
                "not yet have a native MLX runtime backend for this family."
            ),
            recommended_backend=support.backend,
            unverified_model=True,
            mtp_supported="recognized",
            runtime_compatibility=support.runtime_compatibility,
            support_level=support.support_level,
            support_notes=support.notes,
        )

    architecture_text = _text(getattr(inspection, "architecture", None))
    model_type_text = _text(getattr(inspection, "model_type", None))
    compact_architecture = _compact(architecture_text)
    is_gemma4_pair_subfolder = (
        "gemma4forconditionalgeneration" in compact_architecture
        or "gemma4assistantforcausallm" in compact_architecture
        or model_type_text == "gemma4_assistant"
    )
    if not has_mtp and is_gemma4_pair_subfolder:
        # A complete assistant-pair bundle has no MTP tensors of its
        # own: drafting comes from the paired assistant model. The
        # bundle root is recognizable by mtplx_pair.json (recorded as a
        # sidecar by local inspection) or, for remote repos, by weights
        # under both target/ and assistant/. Without this, the Hugging
        # Face preflight rejected the official Gemma 4 repos that the
        # app runs fine (issue #16).
        pair_model_files = tuple(getattr(inspection, "model_files", ()) or ())
        pair_sidecars = getattr(inspection, "sidecars", {}) or {}
        has_pair_root = bool(pair_sidecars.get("mtplx_pair.json")) or (
            any(str(name).startswith("target/") for name in pair_model_files)
            and any(str(name).startswith("assistant/") for name in pair_model_files)
        )
        if has_pair_root:
            return CompatibilityVerdict(
                tier=TIER_VERIFIED,
                arch_id="gemma4-assistant-mtp",
                supported=True,
                recognized=True,
                can_run=True,
                exit_code=EXIT_VERIFIED,
                message=(
                    "Gemma 4 assistant-pair bundle: mtplx_pair.json with "
                    "target/ and assistant/ weights. Runs on the "
                    "gemma4_assistant backend."
                ),
                recommended_backend="gemma4_assistant",
                recommended_profile=DEFAULT_PROFILE_NAME,
                mtp_supported="yes",
                runtime_compatibility="assistant-pair-native",
                support_level="gemma4-pair-bundle",
                support_notes=(
                    "Drafting comes from the paired assistant model; the "
                    "bundle root is the runnable artifact."
                ),
            )
        is_assistant = (
            "assistant" in compact_architecture
            or model_type_text == "gemma4_assistant"
        )
        folder_kind = "assistant" if is_assistant else "target"
        return CompatibilityVerdict(
            tier=TIER_NO_MTP,
            arch_id="gemma4-assistant-mtp",
            supported=False,
            recognized=True,
            can_run=False,
            exit_code=EXIT_NO_MTP,
            message=(
                f"Gemma 4 {folder_kind} folder detected, but MTPLX Gemma "
                "requires the assistant-pair bundle root containing "
                "mtplx_pair.json, target/, and assistant/. Inspect or start "
                "the bundle root instead of this subfolder."
            ),
            recommended_backend="gemma4_assistant",
            recommended_profile=DEFAULT_PROFILE_NAME,
            mtp_supported="no",
            runtime_compatibility="incomplete-assistant-pair",
            support_level="gemma4-pair-bundle-required",
            support_notes=(
                "Gemma 4 support is an external assistant-pair backend; "
                "target-only and assistant-only MLX folders are not runnable."
            ),
        )

    if not has_mtp:
        if (
            support is not None
            and support.runtime_compatibility == "native-ar-only"
            and _passes_family_runtime_gate(support.arch_id, inspection, tensor_gate)
        ):
            return CompatibilityVerdict(
                tier=TIER_AR_ONLY,
                arch_id=support.arch_id,
                supported=True,
                recognized=True,
                can_run=True,
                exit_code=EXIT_VERIFIED,
                message=(
                    f"{support.display_name} matches the bundled MLX loader; "
                    "run in target-only AR mode because the checkpoint has no "
                    "native MTP head."
                ),
                recommended_backend=support.backend,
                recommended_profile=DEFAULT_PROFILE_NAME,
                mtp_supported="no",
                runtime_compatibility=support.runtime_compatibility,
                support_level=support.support_level,
                support_notes=support.notes,
            )
        return CompatibilityVerdict(
            tier=TIER_NO_MTP,
            arch_id=detected_arch_id,
            supported=False,
            recognized=support is not None,
            can_run=False,
            exit_code=EXIT_NO_MTP,
            message=(
                "Model has no MTP head. MTPLX requires an MTP-equipped model."
            ),
            mtp_supported="no",
            runtime_compatibility="unsupported",
            support_level=(support.support_level if support else "unsupported"),
            support_notes=(support.notes if support else None),
        )

    return CompatibilityVerdict(
        tier=TIER_INCOMPATIBLE_ARCHITECTURE,
        arch_id=detected_arch_id or "generic-mtp",
        supported=False,
        recognized=False,
        can_run=False,
        exit_code=EXIT_INCOMPATIBLE_ARCHITECTURE,
        message=(
            f"{detected_arch_id or 'generic MTP'} detected; not supported in "
            "v0.2.0 because no supported native MLX runtime family "
            "matched this artifact."
        ),
        mtp_supported="partial",
        runtime_compatibility="unsupported",
    )


def require_verified_or_raise(
    inspection: Any,
    *,
    unsafe_force_unverified: bool = False,
    yes: bool = False,
) -> CompatibilityVerdict:
    verdict = compatibility_for_inspection(inspection)
    if verdict.can_run:
        return verdict
    if (
        unsafe_force_unverified
        and yes
        and verdict.tier == TIER_ARCH_COMPATIBLE_UNVERIFIED
        and verdict.unsafe_force_required
    ):
        return verdict
    if verdict.tier == TIER_NO_MTP:
        raise NoMTPError(verdict.message)
    if verdict.tier == TIER_INCOMPATIBLE_ARCHITECTURE:
        raise IncompatibleArchitectureError(verdict.message)
    raise UnverifiedArchitectureError(verdict.message)
