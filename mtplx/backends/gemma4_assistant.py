"""Gemma 4 31B assistant-backed MTP scaffold.

This module is intentionally local to MTPLX and keeps MLX imports lazy.  The
implementation is based on the official Google Gemma 4 assistant config shape
and the MLX drafter mechanics demonstrated in Blaizzy/mlx-vlm PR #1112. Gemma
uses target-sampled prefix verification rather than Qwen-style p/q residual
correction.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from mtplx.backends import DraftTokens, ModelState, MTPBackend, VerifyOutput
from mtplx.backends.descriptors import GEMMA4_ASSISTANT_DESCRIPTOR
from mtplx.profiles import DEFAULT_PROFILE_NAME
from mtplx.sampling import SamplerConfig


ARCH_ID = GEMMA4_ASSISTANT_DESCRIPTOR.architecture_id
BACKEND_NAME = GEMMA4_ASSISTANT_DESCRIPTOR.backend_id

OFFICIAL_TARGET_REPO = "google/gemma-4-31B-it"
OFFICIAL_ASSISTANT_REPO = "google/gemma-4-31B-it-assistant"
OFFICIAL_TARGET_REVISION = "145dc2508c480a64b47242f160d286cff94a2343"
OFFICIAL_ASSISTANT_REVISION = "cffbbd2cea41ea56a0fa5b0487e0d445121fd204"

CONSUMER_TARGET_QUANTIZATION = {
    "bits": 4,
    "group_size": 64,
    "mode": "affine",
    "format": "mlx-flat4-g64",
}
PRIMARY_ASSISTANT_DTYPE = "bfloat16"

GEMMA4_31B_BACKBONE_HIDDEN_SIZE = 5376
GEMMA4_31B_ASSISTANT_HIDDEN_SIZE = 1024
GEMMA4_31B_ASSISTANT_LAYERS = 4
GEMMA4_31B_TARGET_LAYERS = 60
GEMMA4_31B_VOCAB_SIZE = 262144
DEFAULT_DRAFT_BLOCK_SIZE = GEMMA4_ASSISTANT_DESCRIPTOR.draft_semantics.default
DEFAULT_TARGET_DISTRIBUTION_MODE = (
    GEMMA4_ASSISTANT_DESCRIPTOR.default_target_distribution_mode
)
DEFAULT_TARGET_DISTRIBUTION_WINDOW_SIZE = int(
    GEMMA4_ASSISTANT_DESCRIPTOR.target_distribution_policy.default_window_size or 1
)
GEMMA4_TARGET_DISTRIBUTION_MODES = set(
    GEMMA4_ASSISTANT_DESCRIPTOR.target_distribution_modes
)
GEMMA4_SESSION_STATE_POLICY = "assistant_shared_kv"
MIN_ARTIFACT_FREE_GIB = 220.0
_THREAD_LOCAL = threading.local()

GEMMA4_LAYER_TYPES_31B_ASSISTANT = (
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
)
GEMMA4_ATTENTION_PHASES = (
    "prefill",
    "decode_verify",
    "ar_decode",
    "postcommit",
    "unknown",
)
GEMMA4_PAGED_BAILOUT_REASONS = (
    "empty_cache",
    "batch_not_1",
    "q_len_invalid",
    "q_len_gt_max",
    "unsupported_mask",
    "offset_invalid",
    "block_size_mismatch",
    "head_dim_unsupported",
    "dtype_unsupported",
    "blocks_invalid",
    "kernel_unavailable",
    "partitioned_unavailable",
    "partitioned_invalid_output",
    "turboquant_unsupported",
    "unknown",
)
GEMMA4_CACHE_COUNTERS = (
    "dense_fallback_calls",
    "paged_active_array_calls",
    "partitioned_paged_calls",
    "large_q_split_sdpa_fallback_calls",
)


class Gemma4AssistantUnsupported(RuntimeError):
    """Raised when a Gemma 4 pair is outside this 31B-only scaffold."""


def _normalize_attention_phase(phase: str | None) -> str:
    value = str(phase or "unknown")
    return value if value in GEMMA4_ATTENTION_PHASES else "unknown"


def _normalize_bailout_reason(reason: str | None) -> str:
    value = str(reason or "unknown")
    return value if value in GEMMA4_PAGED_BAILOUT_REASONS else "unknown"


def _env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _gemma4_draft_position(kv_offset: Any) -> Any:
    """Return the target position that produced the primary token.

    Gemma's assistant receives the sampled primary token plus the previous
    target hidden state. If target KV length is N, that hidden state belongs to
    position N - 1, while N remains the valid KV length for masks.
    """

    if isinstance(kv_offset, int):
        return max(int(kv_offset) - 1, 0)
    if isinstance(kv_offset, np.integer):
        return max(int(kv_offset) - 1, 0)
    mx = _require_mlx_core()
    return mx.maximum(kv_offset.astype(mx.int32) - 1, 0)


def _sample_token_ids_from_logits_mlx(logits: Any, sampler: SamplerConfig):
    """Sample token ids on-device from target/drafter logits.

    Gemma's MTP path follows the MLX-VLM target-prefix verifier: both drafter
    and target verifier rows are sampled directly, and proposal tokens are
    accepted only while they match the target samples. No p/q or residual
    distribution is needed for this backend.
    """

    mx = _require_mlx_core()
    if sampler.temperature <= 0:
        return mx.argmax(logits, axis=-1)
    if int(sampler.top_k) > 0:
        return _sample_token_ids_topk_nucleus_from_logits_mlx(logits, sampler)

    logits = logits.astype(mx.float32)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mlx_sampler = _mlx_sampler_for_config(
        temp=float(sampler.temperature),
        top_p=float(sampler.top_p),
        top_k=int(sampler.top_k),
    )
    return mlx_sampler(logprobs)


def _sample_token_ids_topk_nucleus_from_logits_mlx(logits: Any, sampler: SamplerConfig):
    """Sample with MTPLX/MLX-LM top-p then top-k semantics without full sorting.

    MLX-LM applies top-p to full-vocab log-probs, then top-k, then samples with
    temperature. For the final top-k support, a token survives top-p iff the
    probability mass of strictly higher-probability tokens is below `top_p`.
    That condition only needs the top-k logits plus the full-vocab normalizer,
    avoiding an argsort over Gemma's 262k-token vocabulary for every verifier row.
    """

    mx = _require_mlx_core()
    logits = logits.astype(mx.float32)
    vocab_size = int(logits.shape[-1])
    top_k = max(1, min(int(sampler.top_k), vocab_size))
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    top_indices = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., :top_k]
    top_logprobs = mx.take_along_axis(logprobs, top_indices, axis=-1)
    order = mx.argsort(-top_logprobs, axis=-1)
    top_indices = mx.take_along_axis(top_indices, order, axis=-1)
    top_logprobs = mx.take_along_axis(top_logprobs, order, axis=-1)

    top_p = float(sampler.top_p)
    if 0.0 < top_p < 1.0:
        top_probs = mx.exp(top_logprobs)
        higher_mass = mx.cumsum(top_probs, axis=-1) - top_probs
        keep = higher_mass < top_p
        first = mx.arange(top_k) == 0
        keep = keep | first
        top_logprobs = mx.where(keep, top_logprobs, -float("inf"))

    sampled_offsets = mx.random.categorical(
        top_logprobs * (1.0 / float(sampler.temperature))
    )
    return mx.take_along_axis(top_indices, sampled_offsets[..., None], axis=-1)[..., 0]


@lru_cache(maxsize=32)
def _mlx_sampler_for_config(*, temp: float, top_p: float, top_k: int):
    from mlx_lm.sample_utils import make_sampler

    return make_sampler(temp=float(temp), top_p=float(top_p), top_k=int(top_k))


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def _top_model_type(config: dict[str, Any]) -> str:
    return str(config.get("model_type") or "")


def _architecture_text(config: dict[str, Any]) -> str:
    return " ".join(str(item) for item in config.get("architectures", ()) or ())


def is_gemma4_assistant_config(config: dict[str, Any]) -> bool:
    arch = _architecture_text(config)
    return _top_model_type(config) == "gemma4_assistant" or "Gemma4Assistant" in arch


def is_gemma4_31b_target_config(config: dict[str, Any]) -> bool:
    text = _text_config(config)
    arch = _architecture_text(config)
    model_type = str(text.get("model_type") or config.get("model_type") or "")
    return (
        ("Gemma4" in arch or model_type.startswith("gemma4"))
        and int(text.get("hidden_size") or 0) == GEMMA4_31B_BACKBONE_HIDDEN_SIZE
        and int(text.get("num_hidden_layers") or 0) == GEMMA4_31B_TARGET_LAYERS
        and int(text.get("hidden_size_per_layer_input") or 0) == 0
        and not bool(text.get("enable_moe_block"))
    )


def is_gemma4_31b_assistant_config(config: dict[str, Any]) -> bool:
    text = _text_config(config)
    layer_types = tuple(str(item) for item in text.get("layer_types", ()) or ())
    return (
        is_gemma4_assistant_config(config)
        and int(config.get("backbone_hidden_size") or 0)
        == GEMMA4_31B_BACKBONE_HIDDEN_SIZE
        and not bool(config.get("use_ordered_embeddings"))
        and int(text.get("hidden_size") or 0) == GEMMA4_31B_ASSISTANT_HIDDEN_SIZE
        and int(text.get("num_hidden_layers") or 0) == GEMMA4_31B_ASSISTANT_LAYERS
        and int(text.get("num_kv_shared_layers") or 0) == GEMMA4_31B_ASSISTANT_LAYERS
        and layer_types == GEMMA4_LAYER_TYPES_31B_ASSISTANT
    )


def validate_gemma4_31b_pair_configs(
    target_config: dict[str, Any],
    assistant_config: dict[str, Any],
) -> None:
    if not is_gemma4_31b_target_config(target_config):
        raise Gemma4AssistantUnsupported(
            "Gemma MTP target must be dense Gemma 4 31B text config "
            "(hidden_size=5376, num_hidden_layers=60, no MoE, no ordered/per-layer inputs)."
        )
    if not is_gemma4_assistant_config(assistant_config):
        raise Gemma4AssistantUnsupported(
            "Gemma MTP assistant must have model_type='gemma4_assistant'."
        )
    if bool(assistant_config.get("use_ordered_embeddings")):
        raise Gemma4AssistantUnsupported(
            "Gemma ordered/centroid assistant heads are out of scope for this phase "
            "(E2B/E4B support is deferred)."
        )
    if not is_gemma4_31b_assistant_config(assistant_config):
        raise Gemma4AssistantUnsupported(
            "Only the official dense Gemma 4 31B assistant is supported in this scaffold "
            "(backbone_hidden_size=5376, hidden_size=1024, four shared-KV layers)."
        )


@dataclass(frozen=True)
class Gemma4CacheCounters:
    dense_fallback_calls: int = 0
    paged_active_array_calls: int = 0
    partitioned_paged_calls: int = 0
    large_q_split_sdpa_fallback_calls: int = 0

    def delta_from(self, before: "Gemma4CacheCounters") -> dict[str, int]:
        return {
            name: int(getattr(self, name)) - int(getattr(before, name))
            for name in GEMMA4_CACHE_COUNTERS
        }

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in GEMMA4_CACHE_COUNTERS}


@dataclass(frozen=True)
class Gemma4LongContextPolicy:
    """Sustained-mode guardrails for the future Gemma 31B QA pass."""

    partition_threshold: int = 2048
    allow_dense_fallback_after_threshold: bool = False
    assert_no_paged_active_arrays: bool = False
    trace_paged_attention: bool = False

    @classmethod
    def from_env(cls) -> "Gemma4LongContextPolicy":
        import os

        return cls(
            partition_threshold=int(
                os.environ.get("MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD", "2048")
            ),
            allow_dense_fallback_after_threshold=_env_truthy(
                os.environ.get("MTPLX_ALLOW_LONG_CONTEXT_DENSE_FALLBACK")
            ),
            assert_no_paged_active_arrays=_env_truthy(
                os.environ.get("MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS")
            ),
            trace_paged_attention=_env_truthy(os.environ.get("MTPLX_PAGED_ATTENTION_TRACE")),
        )

    def validate(self) -> None:
        if int(self.partition_threshold) < 0:
            raise ValueError("Gemma 4 partition_threshold must be >= 0")

    def guard_cache_delta(
        self,
        *,
        phase: str,
        before: Gemma4CacheCounters,
        after: Gemma4CacheCounters,
        offset: int,
        q_len: int,
    ) -> None:
        self.validate()
        if int(offset) < int(self.partition_threshold):
            return
        delta = after.delta_from(before)
        dense_delta = int(delta["dense_fallback_calls"])
        if dense_delta > 0 and not self.allow_dense_fallback_after_threshold:
            raise Gemma4AssistantUnsupported(
                "Gemma 4 long-context dense K/V materialization is blocked "
                f"(phase={_normalize_attention_phase(phase)}, offset={offset}, "
                f"q_len={q_len}, dense_fallback_delta={dense_delta}). "
                "Route large-query attention through partitioned paged attention "
                "or a bounded split-SDPA fallback before Gemma runtime QA."
            )
        active_delta = int(delta["paged_active_array_calls"])
        if active_delta > 0 and self.assert_no_paged_active_arrays:
            raise Gemma4AssistantUnsupported(
                "Gemma 4 paged active-array materialization is blocked by "
                f"MTPLX_ASSERT_NO_PAGED_ACTIVE_ARRAYS (phase={phase}, "
                f"offset={offset}, q_len={q_len}, delta={active_delta})."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_threshold": int(self.partition_threshold),
            "allow_dense_fallback_after_threshold": bool(
                self.allow_dense_fallback_after_threshold
            ),
            "assert_no_paged_active_arrays": bool(self.assert_no_paged_active_arrays),
            "trace_paged_attention": bool(self.trace_paged_attention),
        }


@dataclass
class Gemma4RuntimeTelemetry:
    trace_events: bool = False
    dense_fallback_calls_by_phase: dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in GEMMA4_ATTENTION_PHASES}
    )
    paged_active_array_calls_by_phase: dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in GEMMA4_ATTENTION_PHASES}
    )
    paged_attention_bailouts_by_phase_reason: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            phase: {reason: 0 for reason in GEMMA4_PAGED_BAILOUT_REASONS}
            for phase in GEMMA4_ATTENTION_PHASES
        }
    )
    paged_attention_large_q_path: dict[str, int] = field(
        default_factory=lambda: {
            "tail_paged": 0,
            "partitioned_paged": 0,
            "large_q_split_sdpa_fallback": 0,
            "dense_forbidden": 0,
            "unknown": 0,
        }
    )
    events: list[dict[str, Any]] = field(default_factory=list)

    def record_cache_delta(
        self,
        *,
        phase: str,
        before: Gemma4CacheCounters,
        after: Gemma4CacheCounters,
        offset: int,
        q_len: int,
        path: str = "unknown",
    ) -> dict[str, Any]:
        normalized_phase = _normalize_attention_phase(phase)
        delta = after.delta_from(before)
        dense_delta = max(0, int(delta["dense_fallback_calls"]))
        active_delta = max(0, int(delta["paged_active_array_calls"]))
        split_delta = max(0, int(delta["large_q_split_sdpa_fallback_calls"]))
        partitioned_delta = max(0, int(delta["partitioned_paged_calls"]))

        self.dense_fallback_calls_by_phase[normalized_phase] += dense_delta
        self.paged_active_array_calls_by_phase[normalized_phase] += active_delta
        if split_delta:
            self.paged_attention_large_q_path["large_q_split_sdpa_fallback"] += split_delta
        if partitioned_delta:
            self.paged_attention_large_q_path["partitioned_paged"] += partitioned_delta
        if dense_delta:
            self.paged_attention_large_q_path["dense_forbidden"] += dense_delta

        event = {
            "phase": normalized_phase,
            "path": str(path),
            "offset": int(offset),
            "q_len": int(q_len),
            "delta": dict(delta),
        }
        if self.trace_events or any(int(value) > 0 for value in delta.values()):
            self.events.append(event)
        return event

    def record_bailout(self, *, phase: str, reason: str, **details: Any) -> None:
        normalized_phase = _normalize_attention_phase(phase)
        normalized_reason = _normalize_bailout_reason(reason)
        self.paged_attention_bailouts_by_phase_reason[normalized_phase][normalized_reason] += 1
        event = {
            "phase": normalized_phase,
            "reason": normalized_reason,
            "kind": "paged_attention_bailout",
        }
        event.update(details)
        self.events.append(event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_events": bool(self.trace_events),
            "prefill_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase["prefill"]
            ),
            "decode_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase["decode_verify"]
            ),
            "ar_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase["ar_decode"]
            ),
            "postcommit_dense_fallback_calls": int(
                self.dense_fallback_calls_by_phase["postcommit"]
            ),
            "dense_fallback_calls_by_phase": dict(self.dense_fallback_calls_by_phase),
            "paged_active_array_calls_by_phase": dict(
                self.paged_active_array_calls_by_phase
            ),
            "paged_attention_bailouts_by_phase_reason": {
                phase: dict(reasons)
                for phase, reasons in self.paged_attention_bailouts_by_phase_reason.items()
            },
            "paged_attention_large_q_path": dict(self.paged_attention_large_q_path),
            "events": list(self.events),
        }


@dataclass
class Gemma4PromptState:
    cache: Any
    logits: Any
    hidden: Any
    shared_kv_states: dict[str, Any]
    kv_offset: int
    prompt_eval_time_s: float
    cached_tokens: int = 0
    suffix_tokens: int = 0
    cache_hit: bool = False
    cache_miss_reason: str | None = None
    restore_mode: str = "cold"


@dataclass(frozen=True)
class Gemma4AssistantRuntimeConfig:
    target_model_path: Path
    assistant_model_path: Path
    draft_block_size: int = DEFAULT_DRAFT_BLOCK_SIZE
    target_distribution_mode: str = DEFAULT_TARGET_DISTRIBUTION_MODE
    backend: str = BACKEND_NAME
    long_context_policy: Gemma4LongContextPolicy = field(
        default_factory=Gemma4LongContextPolicy
    )

    @classmethod
    def from_paths(
        cls,
        *,
        target_model_path: Path | str,
        assistant_model_path: Path | str,
        draft_block_size: int = DEFAULT_DRAFT_BLOCK_SIZE,
        target_distribution_mode: str | None = None,
    ) -> "Gemma4AssistantRuntimeConfig":
        return cls(
            target_model_path=Path(target_model_path),
            assistant_model_path=Path(assistant_model_path),
            draft_block_size=int(draft_block_size),
            target_distribution_mode=_target_distribution_mode(
                target_distribution_mode or DEFAULT_TARGET_DISTRIBUTION_MODE
            ),
        )

    def validate_static(self) -> None:
        if self.backend != BACKEND_NAME:
            raise ValueError(f"Gemma 4 assistant backend must be {BACKEND_NAME!r}")
        if self.draft_block_size < 2:
            raise ValueError("Gemma 4 assistant draft_block_size must be >= 2")
        target_distribution_mode = _target_distribution_mode(self.target_distribution_mode)
        if target_distribution_mode not in GEMMA4_TARGET_DISTRIBUTION_MODES:
            raise ValueError(
                "Gemma 4 target_distribution_mode must be one of "
                f"{sorted(GEMMA4_TARGET_DISTRIBUTION_MODES)}"
            )
        self.long_context_policy.validate()
        if not self.target_model_path.exists():
            raise FileNotFoundError(
                f"Gemma 4 target path must be local for this scaffold: {self.target_model_path}"
            )
        if not self.assistant_model_path.exists():
            raise FileNotFoundError(
                "Gemma 4 assistant path must be local for this scaffold: "
                f"{self.assistant_model_path}"
            )
        validate_gemma4_31b_pair_configs(
            _load_json(self.target_model_path / "config.json"),
            _load_json(self.assistant_model_path / "config.json"),
        )


@dataclass
class Gemma4AssistantArgs:
    model_type: str = "gemma4_assistant"
    backbone_hidden_size: int = GEMMA4_31B_BACKBONE_HIDDEN_SIZE
    use_ordered_embeddings: bool = False
    num_centroids: int = 2048
    centroid_intermediate_top_k: int = 32
    tie_word_embeddings: bool = True
    block_size: int = DEFAULT_DRAFT_BLOCK_SIZE
    target_layer_ids: list[int] = field(default_factory=list)
    target_layer_types: list[str] = field(default_factory=list)
    text_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "Gemma4AssistantArgs":
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in dict(params).items() if key in fields})

    from_hf_dict = from_dict

    def validate_31b_dense(self) -> None:
        validate_gemma4_31b_pair_configs(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": {
                    "model_type": "gemma4_text",
                    "hidden_size": GEMMA4_31B_BACKBONE_HIDDEN_SIZE,
                    "num_hidden_layers": GEMMA4_31B_TARGET_LAYERS,
                    "hidden_size_per_layer_input": 0,
                    "enable_moe_block": False,
                },
            },
            {
                "model_type": self.model_type,
                "architectures": ["Gemma4AssistantForCausalLM"],
                "backbone_hidden_size": self.backbone_hidden_size,
                "use_ordered_embeddings": self.use_ordered_embeddings,
                "text_config": self.text_config,
            },
        )


@dataclass(frozen=True)
class Gemma4TargetOutput:
    logits: Any
    hidden: Any
    shared_kv_states: dict[str, Any]
    cache_offset: int
    attention_phase: str = "unknown"
    cache_counters: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Gemma4DraftStep:
    token_id: int
    logits: Any
    q_distribution: Any
    hidden: Any


@dataclass(frozen=True)
class Gemma4DraftBlock:
    steps: tuple[Gemma4DraftStep, ...]

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(step.token_id for step in self.steps)


@dataclass(frozen=True)
class Gemma4VerifyBlock:
    input_token_ids: tuple[int, ...]
    output: Gemma4TargetOutput
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Gemma4ExactRoundResult:
    accepted_token_ids: tuple[int, ...]
    corrected_token_id: int | None
    bonus_token_id: int | None
    next_primary_token_id: int | None
    accepted_count: int
    metadata: dict[str, Any]
    next_hidden: Any | None = field(default=None, repr=False)
    next_shared_kv_states: dict[str, Any] = field(default_factory=dict, repr=False)
    next_kv_offset: int | None = None


def make_drafter_masks(
    shared_kv_states: dict[str, Any],
    *,
    query_len: int,
    query_offset: Any,
    sliding_window: int,
    dtype: Any,
    kv_valid_len: Any = None,
) -> dict[str, Any]:
    if kv_valid_len is None:
        kv_valid_len = query_offset
    masks: dict[str, Any] = {}
    for layer_type, kv in shared_kv_states.items():
        kv_len = _kv_len(kv)
        if layer_type == "sliding_attention":
            local_query_offset = _local_window_offset(query_offset, kv_len)
            local_valid_len = _local_window_offset(kv_valid_len, kv_len)
            masks[layer_type] = _bidirectional_swa_mask(
                query_len=query_len,
                query_offset=local_query_offset,
                kv_len=kv_len,
                window=sliding_window,
                kv_valid_len=local_valid_len,
                key_offset=0,
                dtype=dtype,
            )
        else:
            mx = _require_mlx_core()
            key_offset = (
                max(int(kv_valid_len) - kv_len, 0)
                if isinstance(kv_valid_len, int)
                else mx.maximum(kv_valid_len - kv_len, 0)
            )
            masks[layer_type] = _bidirectional_full_mask(
                query_len=query_len,
                kv_len=kv_len,
                kv_valid_len=kv_valid_len,
                key_offset=key_offset,
                dtype=dtype,
            )
    return masks


def _local_window_offset(query_offset: Any, kv_len: int) -> Any:
    """Map absolute decode positions onto the local rotating/shared-KV window."""

    if isinstance(query_offset, int):
        return min(int(query_offset), int(kv_len))
    mx = _require_mlx_core()
    limit = mx.array(int(kv_len))
    dtype = getattr(query_offset, "dtype", None)
    if dtype is not None:
        limit = limit.astype(dtype)
    return mx.minimum(query_offset, limit)


def normalize_batched_shared_kv_states(
    shared_kv_states: dict[str, Any],
    *,
    kv_valid_len: Any,
    left_padding: Any = None,
) -> dict[str, Any]:
    if left_padding is None:
        return shared_kv_states
    out: dict[str, Any] = {}
    for layer_type, (keys, values) in shared_kv_states.items():
        out[layer_type] = (
            _normalize_shared_kv_tensor(keys, kv_valid_len, left_padding),
            _normalize_shared_kv_tensor(values, kv_valid_len, left_padding),
        )
    return out


def _normalize_shared_kv_tensor(tensor: Any, kv_valid_len: Any, left_padding: Any) -> Any:
    if getattr(tensor, "ndim", 0) != 4:
        return tensor
    mx = _require_mlx_core()
    from mlx_lm.models.cache import dynamic_roll

    batch = int(tensor.shape[0])
    seq_len = int(tensor.shape[-2])
    valid = _broadcast_batch_vector(kv_valid_len, batch, seq_len)
    left = _broadcast_batch_vector(left_padding, batch, seq_len)
    if batch == 1 and int(left[0].item()) == 0 and int(valid[0].item()) >= seq_len:
        return tensor
    rolled = dynamic_roll(tensor, -left[:, None], axis=2)
    keep = mx.arange(seq_len)[None, :] < valid[:, None]
    keep = keep.astype(tensor.dtype)[:, None, :, None]
    return rolled * keep


def _broadcast_batch_vector(value: Any, batch: int, limit: int) -> Any:
    mx = _require_mlx_core()
    if isinstance(value, int):
        vector = mx.array([value], dtype=mx.int32)
    elif hasattr(value, "astype"):
        vector = value.astype(mx.int32)
    else:
        vector = mx.array(value, dtype=mx.int32)
    if vector.ndim == 0:
        vector = vector[None]
    elif vector.ndim > 1:
        vector = vector.reshape(-1)
    if vector.shape[0] == 1 and batch != 1:
        vector = mx.repeat(vector, batch, axis=0)
    if vector.shape[0] != batch:
        raise ValueError(
            f"Expected batch metadata of length {batch}, got {vector.shape[0]}"
        )
    return mx.clip(vector, 0, limit)


class Gemma4TargetAdapter:
    """Wrap the loaded MLX-LM Gemma 4 target and expose shared-KV hooks."""

    def __init__(self, target_model: Any):
        self.target_model = target_model
        self.language_model = getattr(target_model, "language_model", target_model)
        self.text_model = getattr(self.language_model, "model", self.language_model)
        self._validate_31b_dense()

    def _validate_31b_dense(self) -> None:
        config = getattr(self.text_model, "config", None)
        if config is None:
            raise Gemma4AssistantUnsupported("Gemma 4 target has no text config")
        if int(getattr(config, "hidden_size", 0) or 0) != GEMMA4_31B_BACKBONE_HIDDEN_SIZE:
            raise Gemma4AssistantUnsupported("only dense Gemma 4 31B target is scaffolded")
        if int(getattr(config, "hidden_size_per_layer_input", 0) or 0):
            raise Gemma4AssistantUnsupported("Gemma E2B/E4B per-layer inputs are out of scope")
        if bool(getattr(config, "enable_moe_block", False)):
            raise Gemma4AssistantUnsupported("Gemma 26B-A4B MoE target is out of scope")

    def make_cache(self):
        config = getattr(self.text_model, "config", None)
        layers = list(getattr(self.text_model, "layers"))
        first_kv_shared = len(layers) - int(getattr(config, "num_kv_shared_layers", 0) or 0)
        caches = []
        try:
            from mlx_lm.models.cache import KVCache
        except Exception:
            return self.language_model.make_cache()

        for layer in layers[:first_kv_shared]:
            if str(getattr(layer, "layer_type", "")) == "full_attention":
                caches.append(KVCache())
            else:
                caches.append(
                    Gemma4RollbackRotatingKVCache(
                        max_size=int(getattr(config, "sliding_window", 512) or 512),
                        keep=0,
                    )
                )
        return caches

    def cache_offset(self, cache: Any) -> int:
        for item in cache or ():
            if item is None:
                continue
            if hasattr(item, "offset"):
                return _offset_to_int(item.offset)
            if hasattr(item, "_idx"):
                return _offset_to_int(item._idx)
        return 0

    def cache_position(self, cache: Any) -> Any:
        for item in cache or ():
            if item is None:
                continue
            if hasattr(item, "offset"):
                return item.offset
            if hasattr(item, "_idx"):
                return item._idx
        return 0

    def forward_with_state(
        self,
        inputs: Any,
        *,
        cache: Any = None,
        phase: str = "unknown",
        telemetry: Gemma4RuntimeTelemetry | None = None,
        long_context_policy: Gemma4LongContextPolicy | None = None,
        compute_logits: bool = True,
        include_cache_offset: bool = True,
    ) -> Gemma4TargetOutput:
        phase = _normalize_attention_phase(phase)
        q_len = _shape_dim(inputs, 1, fallback=1)
        before_counters = _cache_counters(cache)
        before_offset = (
            self.cache_offset(cache)
            if telemetry is not None or long_context_policy is not None
            else 0
        )

        input_embeddings = self.text_model.embed_tokens(inputs)
        h = input_embeddings * float(getattr(self.text_model, "embed_scale", 1.0))
        if getattr(self.text_model, "hidden_size_per_layer_input", 0):
            raise Gemma4AssistantUnsupported("per-layer Gemma inputs are not scaffolded")

        layers = list(getattr(self.text_model, "layers"))
        previous_kvs = list(getattr(self.text_model, "previous_kvs"))
        if cache is None:
            cache = [None] * len(layers)
        else:
            cache = list(cache) + [None] * (len(layers) - len(cache))

        masks = self.text_model._make_masks(h, cache)
        shared_kv_states: dict[str, Any] = {}
        intermediates = [(None, None)] * len(layers)
        for idx, (layer, layer_cache, mask, previous_idx) in enumerate(
            zip(layers, cache, masks, previous_kvs)
        ):
            kvs, offset = intermediates[previous_idx]
            h, kvs, offset = layer(
                h,
                mask,
                layer_cache,
                per_layer_input=None,
                shared_kv=kvs,
                offset=offset,
            )
            intermediates[idx] = (kvs, offset)
            if kvs is not None:
                shared_kv_states[str(layer.layer_type)] = kvs

        pre_norm_hidden = h
        logits = self.logits_from_hidden(pre_norm_hidden) if compute_logits else None

        after_counters = _cache_counters(cache)
        if telemetry is not None:
            telemetry.record_cache_delta(
                phase=phase,
                before=before_counters,
                after=after_counters,
                offset=before_offset,
                q_len=q_len,
                path="gemma4_target_forward",
            )
        if long_context_policy is not None:
            long_context_policy.guard_cache_delta(
                phase=phase,
                before=before_counters,
                after=after_counters,
                offset=before_offset,
                q_len=q_len,
            )

        return Gemma4TargetOutput(
            logits=logits,
            hidden=pre_norm_hidden,
            shared_kv_states=shared_kv_states,
            cache_offset=self.cache_offset(cache) if include_cache_offset else 0,
            attention_phase=phase,
            cache_counters=after_counters.to_dict(),
        )

    def logits_from_hidden(self, hidden: Any) -> Any:
        from mlx_lm.models.gemma4_text import logit_softcap

        normed = self.text_model.norm(hidden)
        if getattr(self.language_model, "tie_word_embeddings", True):
            logits = self.text_model.embed_tokens.as_linear(normed)
        else:
            logits = self.language_model.lm_head(normed)
        softcap = getattr(self.language_model, "final_logit_softcapping", None)
        if softcap is not None:
            logits = logit_softcap(softcap, logits)
        return logits

    def draft_hidden_from_target_hidden(self, hidden: Any) -> Any:
        return self.text_model.norm(hidden)

    def rollback_speculative_cache(
        self,
        cache: Any,
        *,
        verified_tokens: int,
        committed_tokens: int,
    ) -> int:
        trim = int(verified_tokens) - int(committed_tokens)
        if trim <= 0:
            return 0
        for item in cache or ():
            if item is None:
                continue
            if hasattr(item, "is_trimmable") and not item.is_trimmable():
                continue
            if not hasattr(item, "trim"):
                raise Gemma4AssistantUnsupported(
                    f"Gemma 4 cache item {type(item).__name__} cannot be rolled back"
                )
            item.trim(trim)
        return trim


class Gemma4TensorOffsetRotatingKVCache:
    """Short-context tensor-offset wrapper for Gemma sliding-window caches."""

    def __init__(
        self,
        keys: Any,
        values: Any,
        offset: Any,
        *,
        max_size: int,
        step: int = 256,
    ) -> None:
        mx = _require_mlx_core()
        offset_array = offset if hasattr(offset, "shape") else mx.array(offset, dtype=mx.int32)
        self.cache = [keys, values, offset_array]
        self.max_size = int(max_size)
        self.step = int(step)

    @classmethod
    def from_rotating_cache(
        cls,
        entry: Any,
        *,
        reserve_tokens: int,
    ) -> "Gemma4TensorOffsetRotatingKVCache":
        cache = cls(
            entry.keys,
            entry.values,
            entry.offset,
            max_size=int(entry.max_size),
            step=int(getattr(entry, "step", 256)),
        )
        cache.ensure_capacity(_offset_to_int(entry.offset) + int(reserve_tokens))
        return cache

    @property
    def keys(self):
        return self.cache[0]

    @keys.setter
    def keys(self, value):
        self.cache[0] = value

    @property
    def values(self):
        return self.cache[1]

    @values.setter
    def values(self, value):
        self.cache[1] = value

    @property
    def offset(self):
        return self.cache[2]

    @offset.setter
    def offset(self, value):
        mx = _require_mlx_core()
        self.cache[2] = value if hasattr(value, "shape") else mx.array(value, dtype=mx.int32)

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, value):
        self.cache = value

    @property
    def compile_state(self):
        return self.cache

    def ensure_capacity(self, needed: int) -> None:
        if self.keys is None or self.values is None:
            return
        needed = min(int(needed), self.max_size)
        capacity = int(self.keys.shape[2])
        if needed <= capacity:
            return
        mx = _require_mlx_core()
        new_capacity = min(
            self.max_size,
            ((needed + self.step - 1) // self.step) * self.step,
        )
        extra = new_capacity - capacity
        k_shape = (*self.keys.shape[:2], extra, self.keys.shape[3])
        v_shape = (*self.values.shape[:2], extra, self.values.shape[3])
        self.keys = mx.concatenate(
            [self.keys, mx.zeros(k_shape, dtype=self.keys.dtype)],
            axis=2,
        )
        self.values = mx.concatenate(
            [self.values, mx.zeros(v_shape, dtype=self.values.dtype)],
            axis=2,
        )

    def update_and_fetch(self, keys, values):
        mx = _require_mlx_core()
        steps = int(keys.shape[2])
        self.cache[0] = mx.slice_update(self.cache[0], keys, self.cache[2], axes=(2,))
        self.cache[1] = mx.slice_update(self.cache[1], values, self.cache[2], axes=(2,))
        self.cache[2] = self.cache[2] + steps
        return self.cache[0], self.cache[1]

    def make_mask(self, N: int, window_size=None, return_array: bool = False):
        del return_array
        mx = _require_mlx_core()
        capacity = int(self.keys.shape[2])
        rinds = mx.arange(capacity)
        linds = self.cache[2] + mx.arange(int(N))
        mask = linds[:, None] >= rinds[None, :]
        if window_size is not None:
            mask = mask & (linds[:, None] < rinds[None, :] + int(window_size))
        return mask

    def size(self):
        return _offset_to_int(self.cache[2])

    def is_trimmable(self):
        return True

    def trim(self, n):
        mx = _require_mlx_core()
        n = int(n)
        self.cache[2] = mx.maximum(
            self.cache[2] - n,
            mx.array(0, dtype=self.cache[2].dtype),
        )
        return n

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes + self.cache[2].nbytes


class Gemma4RollbackRotatingKVCache:
    """Gemma sliding-window cache with exact speculative prefix rollback.

    Stock MLX ``RotatingKVCache`` deliberately reports non-trimmable once the
    sliding window is full.  That is fine for autoregressive decode, but exact
    speculative decoding must verify a block, keep only the accepted prefix,
    and discard the rejected suffix.  This local cache mirrors MLX-LM's
    temporal-order semantics while retaining enough last-update state to undo
    or partially commit the most recent verify block.
    """

    step = 256

    def __init__(self, max_size: int, keep: int = 0) -> None:
        self.keep = int(keep)
        self.keys = None
        self.values = None
        self.offset = 0
        self.max_size = int(max_size)
        self._idx = 0
        self._last_update: dict[str, Any] | None = None
        self._replaying_prefix = False

    def _clear_last_update(self) -> None:
        if not self._replaying_prefix:
            self._last_update = None

    def _save_concat_update(self, keys: Any, values: Any) -> None:
        if self._replaying_prefix:
            return
        self._last_update = {
            "kind": "concat",
            "keys": self.keys,
            "values": self.values,
            "offset": int(self.offset),
            "idx": int(self._idx),
            "update_keys": keys,
            "update_values": values,
            "length": int(keys.shape[2]),
        }

    def _save_in_place_update(self, keys: Any, values: Any, *, start: int) -> None:
        if self._replaying_prefix:
            return
        mx = _require_mlx_core()
        length = int(keys.shape[2])
        start_index = mx.array(int(start), dtype=mx.int32)
        old_keys = mx.slice(self.keys, start_index, axes=(2,), slice_size=keys.shape)
        old_values = mx.slice(
            self.values,
            start_index,
            axes=(2,),
            slice_size=values.shape,
        )
        self._last_update = {
            "kind": "in_place",
            "offset": int(self.offset),
            "idx": int(self._idx),
            "start": int(start),
            "old_keys": old_keys,
            "old_values": old_values,
            "update_keys": keys,
            "update_values": values,
            "length": length,
        }

    def _trim(self, trim_size: int, value: Any, append: Any = None):
        mx = _require_mlx_core()
        to_cat = []
        if int(trim_size) > 0:
            to_cat = [
                value[..., : self.keep, :],
                value[..., int(trim_size) + self.keep :, :],
            ]
        else:
            to_cat = [value]
        if append is not None:
            to_cat.append(append)
        return mx.concatenate(to_cat, axis=2)

    def _temporal_order(self, value: Any):
        mx = _require_mlx_core()
        if self._idx == value.shape[2]:
            return value
        if self._idx < self.offset:
            return mx.concatenate(
                [
                    value[..., : self.keep, :],
                    value[..., self._idx :, :],
                    value[..., self.keep : self._idx, :],
                ],
                axis=2,
            )
        return value[..., : self._idx, :]

    def _update_concat(self, keys: Any, values: Any):
        self._save_concat_update(keys, values)
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = self._temporal_order(self.keys)
            self.values = self._temporal_order(self.values)
            self._idx = self.keys.shape[2]
            trim_size = self._idx - self.max_size + 1
            self.keys = self._trim(trim_size, self.keys, keys)
            self.values = self._trim(trim_size, self.values, values)
        self.offset += int(keys.shape[2])
        self._idx = int(self.keys.shape[2])
        return self.keys, self.values

    def _update_in_place(self, keys: Any, values: Any):
        mx = _require_mlx_core()
        batch, n_kv_heads, steps, k_head_dim = keys.shape
        v_head_dim = values.shape[3]
        prev = int(self.offset)
        if self.keys is None or (
            prev >= self.keys.shape[2] and self.keys.shape[2] < self.max_size
        ):
            new_size = min(self.step, self.max_size - prev)
            k_shape = (batch, n_kv_heads, new_size, k_head_dim)
            v_shape = (batch, n_kv_heads, new_size, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v
            self._idx = prev

        trim_size = int(self.keys.shape[2]) - self.max_size
        if trim_size > 0:
            self.keys = self._trim(trim_size, self.keys)
            self.values = self._trim(trim_size, self.values)
            self._idx = self.max_size

        if self._idx == self.max_size:
            self._idx = self.keep

        write_start = int(self._idx)
        self._save_in_place_update(keys, values, start=write_start)
        self.keys[..., write_start : write_start + steps, :] = keys
        self.values[..., write_start : write_start + steps, :] = values
        self.offset += int(steps)
        self._idx += int(steps)

        if self.offset < self.max_size:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    def update_and_fetch(self, keys: Any, values: Any):
        self._clear_last_update()
        if int(keys.shape[2]) == 1:
            return self._update_in_place(keys, values)
        return self._update_concat(keys, values)

    def _replay_committed_prefix(self, keys: Any, values: Any, committed: int) -> None:
        if int(committed) <= 0:
            return
        self._replaying_prefix = True
        try:
            self.update_and_fetch(
                keys[..., : int(committed), :],
                values[..., : int(committed), :],
            )
        finally:
            self._replaying_prefix = False
            self._last_update = None

    def make_mask(self, n_tokens: int, window_size=None, return_array: bool = False):
        mx = _require_mlx_core()
        n_tokens = int(n_tokens)
        if n_tokens > 1:
            from mlx_lm.models.base import create_causal_mask

            window_size = window_size or self.max_size
            offset = min(self.max_size - 1, int(self.offset))
            if offset + n_tokens > window_size or bool(return_array):
                return create_causal_mask(
                    n_tokens,
                    offset,
                    window_size=window_size,
                )
            return "causal"
        if window_size is None:
            return None
        if self.offset >= window_size and self.max_size > window_size:
            idx = int(self._idx)
            if idx >= self.max_size:
                idx = 0
            mask_size = self.offset + 1 if self.offset < self.max_size else self.max_size
            mask = mx.arange(mask_size) >= (mask_size - int(window_size))
            return mx.roll(mask, shift=idx + 1)
        return None

    def size(self):
        return min(int(self.offset), int(self.max_size))

    @property
    def state(self):
        if self.keys is None:
            return None
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        return self.keys, self.values

    @state.setter
    def state(self, value) -> None:
        if value is None:
            self.keys = None
            self.values = None
            self.offset = 0
            self._idx = 0
            self._last_update = None
            return
        self.keys, self.values = value

    @property
    def meta_state(self):
        return tuple(map(str, (self.keep, self.max_size, self.offset, self._idx)))

    @meta_state.setter
    def meta_state(self, value) -> None:
        self.keep, self.max_size, self.offset, self._idx = map(int, value)
        self._last_update = None

    def is_trimmable(self):
        return True

    def trim(self, n_tokens: int):
        n_tokens = int(n_tokens)
        if n_tokens <= 0:
            return 0
        last = self._last_update
        if last is not None:
            length = int(last.get("length") or 0)
            if n_tokens <= length:
                committed = length - n_tokens
                update_keys = last["update_keys"]
                update_values = last["update_values"]
                if last.get("kind") == "concat":
                    self.keys = last["keys"]
                    self.values = last["values"]
                    self.offset = int(last["offset"])
                    self._idx = int(last["idx"])
                    self._last_update = None
                    self._replay_committed_prefix(update_keys, update_values, committed)
                    return n_tokens
                if last.get("kind") == "in_place":
                    mx = _require_mlx_core()
                    self.keys = mx.slice_update(
                        self.keys,
                        last["old_keys"],
                        int(last["start"]),
                        axes=(2,),
                    )
                    self.values = mx.slice_update(
                        self.values,
                        last["old_values"],
                        int(last["start"]),
                        axes=(2,),
                    )
                    self.offset = int(last["offset"])
                    self._idx = int(last["idx"])
                    self._last_update = None
                    self._replay_committed_prefix(update_keys, update_values, committed)
                    return n_tokens

        n_tokens = min(int(self.offset), n_tokens)
        self.offset -= n_tokens
        self._idx = max(self.keep, int(self._idx) - n_tokens)
        self._last_update = None
        return n_tokens

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


def promote_gemma4_cache_for_compiled_verify(
    cache: Any,
    *,
    reserve_tokens: int,
) -> dict[str, int]:
    """Promote Gemma target cache offsets to tensor state for compiled verify."""

    from mtplx.graphbank import TensorOffsetKVCache

    try:
        from mlx_lm.models.cache import KVCache, RotatingKVCache
    except Exception as exc:  # pragma: no cover - import guard for minimal envs
        raise Gemma4AssistantUnsupported(
            f"cannot import MLX cache classes for Gemma compile path: {exc}"
        ) from exc

    stats = {"full": 0, "sliding": 0, "already_promoted": 0, "skipped": 0}
    for idx, entry in enumerate(cache or ()):
        if entry is None:
            stats["skipped"] += 1
            continue
        if isinstance(entry, (TensorOffsetKVCache, Gemma4TensorOffsetRotatingKVCache)):
            if hasattr(entry, "ensure_capacity"):
                entry.ensure_capacity(entry.size() + int(reserve_tokens))
            stats["already_promoted"] += 1
            continue
        if isinstance(entry, KVCache):
            if entry.keys is None or entry.values is None:
                stats["skipped"] += 1
                continue
            cache[idx] = TensorOffsetKVCache.from_kv_cache(
                entry,
                reserve_tokens=int(reserve_tokens),
            )
            stats["full"] += 1
            continue
        if isinstance(entry, RotatingKVCache):
            if entry.keys is None or entry.values is None:
                stats["skipped"] += 1
                continue
            if int(entry.keep) != 0:
                raise Gemma4AssistantUnsupported(
                    "Gemma compiled verify only supports RotatingKVCache keep=0"
                )
            if _offset_to_int(entry.offset) + int(reserve_tokens) > int(entry.max_size):
                raise Gemma4AssistantUnsupported(
                    "Gemma compiled verify is limited to prompts that fit inside "
                    "the sliding-window cache for this phase"
                )
            cache[idx] = Gemma4TensorOffsetRotatingKVCache.from_rotating_cache(
                entry,
                reserve_tokens=int(reserve_tokens),
            )
            stats["sliding"] += 1
            continue
        stats["skipped"] += 1
    return stats


class Gemma4AssistantRuntime:
    def __init__(
        self,
        *,
        target_model: Any,
        tokenizer: Any,
        assistant_model: Any,
        config: Gemma4AssistantRuntimeConfig,
    ):
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.target = Gemma4TargetAdapter(target_model)
        self.assistant = assistant_model.bind(self.target)
        self.config = config
        self.model_path = config.target_model_path
        self.path = config.target_model_path
        self.mtp_enabled = True
        self.backend_id = BACKEND_NAME
        self.gemma4_external_assistant = True
        self.telemetry = Gemma4RuntimeTelemetry(
            trace_events=bool(config.long_context_policy.trace_paged_attention)
        )
        self._compiled_verify: dict[tuple[int, bool], Any] = {}
        self._compiled_distribution_windows: dict[tuple[int, float, int], Any] = {}
        self.compile_stats: dict[str, Any] = {
            "enabled": False,
            "compiled_calls": 0,
            "fallback_calls": 0,
            "compile_errors": {},
            "promotions": {},
        }
        self.distribution_compile_stats: dict[str, Any] = {
            "enabled": False,
            "compiled_calls": 0,
            "fallback_calls": 0,
            "compile_errors": {},
        }

    def make_cache(self):
        return self.target.make_cache()

    def forward_target(
        self,
        input_ids: Any,
        *,
        cache: Any = None,
        phase: str = "unknown",
        compute_logits: bool = True,
    ) -> Gemma4TargetOutput:
        return self.target.forward_with_state(
            input_ids,
            cache=cache,
            phase=phase,
            telemetry=self.telemetry,
            long_context_policy=self.config.long_context_policy,
            compute_logits=compute_logits,
        )

    def propose_block(
        self,
        *,
        last_token_id: int,
        hidden: Any,
        shared_kv_states: dict[str, Any],
        kv_offset: int,
        sampler: SamplerConfig,
        rng: np.random.Generator,
        draft_block_size: int | None = None,
    ) -> Gemma4DraftBlock:
        mx = _require_mlx_core()

        del rng

        self.assistant.set_shared_kv(
            shared_kv_states,
            kv_offset,
            position=_gemma4_draft_position(kv_offset),
            kv_valid_len=kv_offset,
        )
        token = mx.array([[int(last_token_id)]], dtype=mx.int32)
        hidden_step = self.target.draft_hidden_from_target_hidden(hidden)
        steps: list[Gemma4DraftStep] = []
        block_size = int(draft_block_size or self.config.draft_block_size)
        for _ in range(max(0, block_size - 1)):
            hidden_step, logits = self.assistant.draft_step(token, hidden_step)
            draft_token_arr = _sample_token_ids_from_logits_mlx(
                logits[:, -1, :][0],
                sampler,
            )
            mx.eval(draft_token_arr)
            draft_token = int(draft_token_arr.item())
            token = mx.array([[int(draft_token)]], dtype=mx.int32)
            steps.append(
                Gemma4DraftStep(
                    token_id=int(draft_token),
                    logits=logits,
                    q_distribution=None,
                    hidden=hidden_step,
                )
            )
        return Gemma4DraftBlock(tuple(steps))

    def verify_block(
        self,
        *,
        primary_token_id: int,
        draft_token_ids: tuple[int, ...],
        cache: Any,
        phase: str = "decode_verify",
        compute_logits: bool = True,
    ) -> Gemma4VerifyBlock:
        if compute_logits and _gemma4_compile_verify_enabled():
            try:
                return self._verify_block_compiled(
                    primary_token_id=primary_token_id,
                    draft_token_ids=draft_token_ids,
                    cache=cache,
                    phase=phase,
                    compute_logits=compute_logits,
                )
            except Exception as exc:
                key = type(exc).__name__
                errors = self.compile_stats.setdefault("compile_errors", {})
                errors[key] = int(errors.get(key, 0)) + 1
                self.compile_stats["fallback_calls"] = (
                    int(self.compile_stats.get("fallback_calls", 0)) + 1
                )
        mx = _require_mlx_core()
        verify_input = (int(primary_token_id), *tuple(int(token) for token in draft_token_ids))
        output = self.target.forward_with_state(
            mx.array([list(verify_input)], dtype=mx.int32),
            cache=cache,
            phase=phase,
            telemetry=self.telemetry,
            long_context_policy=self.config.long_context_policy,
            compute_logits=compute_logits,
        )
        return Gemma4VerifyBlock(
            input_token_ids=verify_input,
            output=output,
            telemetry=self.telemetry.to_dict(),
        )

    def prepare_compiled_verify(self, cache: Any, *, reserve_tokens: int) -> dict[str, int]:
        promotions = promote_gemma4_cache_for_compiled_verify(
            cache,
            reserve_tokens=int(reserve_tokens),
        )
        self.compile_stats["enabled"] = True
        self.compile_stats["promotions"] = dict(promotions)
        return promotions

    def _verify_block_compiled(
        self,
        *,
        primary_token_id: int,
        draft_token_ids: tuple[int, ...],
        cache: Any,
        phase: str,
        compute_logits: bool,
    ) -> Gemma4VerifyBlock:
        mx = _require_mlx_core()
        from mtplx.graphbank import cache_array_tree

        verify_input = (int(primary_token_id), *tuple(int(token) for token in draft_token_ids))
        input_ids = mx.array([list(verify_input)], dtype=mx.int32)
        length = len(verify_input)
        key = (length, bool(compute_logits))
        fn = self._compiled_verify.get(key)
        if fn is None:
            if compute_logits:

                def verify_fn(compiled_input_ids):
                    output = self.target.forward_with_state(
                        compiled_input_ids,
                        cache=cache,
                        phase=phase,
                        telemetry=None,
                        long_context_policy=None,
                        compute_logits=True,
                        include_cache_offset=False,
                    )
                    sliding = output.shared_kv_states.get("sliding_attention")
                    full = output.shared_kv_states.get("full_attention")
                    if sliding is None or full is None:
                        raise Gemma4AssistantUnsupported(
                            "Gemma compiled verify expected full and sliding shared KV"
                        )
                    return (
                        output.logits,
                        output.hidden,
                        sliding[0],
                        sliding[1],
                        full[0],
                        full[1],
                    )
            else:

                def verify_fn(compiled_input_ids):
                    output = self.target.forward_with_state(
                        compiled_input_ids,
                        cache=cache,
                        phase=phase,
                        telemetry=None,
                        long_context_policy=None,
                        compute_logits=False,
                        include_cache_offset=False,
                    )
                    sliding = output.shared_kv_states.get("sliding_attention")
                    full = output.shared_kv_states.get("full_attention")
                    if sliding is None or full is None:
                        raise Gemma4AssistantUnsupported(
                            "Gemma compiled verify expected full and sliding shared KV"
                        )
                    return (
                        output.hidden,
                        sliding[0],
                        sliding[1],
                        full[0],
                        full[1],
                    )

            fn = mx.compile(
                verify_fn,
                inputs=cache_array_tree(cache),
                outputs=cache_array_tree(cache),
            )
            self._compiled_verify[key] = fn
        if compute_logits:
            logits, hidden, sliding_k, sliding_v, full_k, full_v = fn(input_ids)
        else:
            hidden, sliding_k, sliding_v, full_k, full_v = fn(input_ids)
            logits = None
        eval_targets = [hidden, sliding_k, sliding_v, full_k, full_v]
        if logits is not None:
            eval_targets.append(logits)
        mx.eval(*eval_targets)
        self.compile_stats["compiled_calls"] = (
            int(self.compile_stats.get("compiled_calls", 0)) + 1
        )
        output = Gemma4TargetOutput(
            logits=logits,
            hidden=hidden,
            shared_kv_states={
                "sliding_attention": (sliding_k, sliding_v),
                "full_attention": (full_k, full_v),
            },
            cache_offset=self.target.cache_offset(cache),
            attention_phase=_normalize_attention_phase(phase),
            cache_counters=_cache_counters(cache).to_dict(),
        )
        return Gemma4VerifyBlock(
            input_token_ids=verify_input,
            output=output,
            telemetry=self.telemetry.to_dict(),
        )

class Gemma4AssistantBackend(MTPBackend):
    arch_id = ARCH_ID

    def load(self, model_path: Path) -> ModelState:
        raise NotImplementedError(
            "Gemma4AssistantBackend.load requires target+assistant paths; "
            "use load_pair(Gemma4AssistantRuntimeConfig(...))."
        )

    def load_pair(self, config: Gemma4AssistantRuntimeConfig) -> ModelState:
        runtime = load_gemma4_assistant_pair(config)
        return ModelState(
            model_path=config.target_model_path,
            runtime=runtime,
            metadata={
                "arch_id": self.arch_id,
                "backend": BACKEND_NAME,
                "assistant_model_path": str(config.assistant_model_path),
                "draft_block_size": int(config.draft_block_size),
                "qa_pending": True,
                "long_context_policy": config.long_context_policy.to_dict(),
            },
        )

    def verify(self, state: ModelState, draft_tokens: DraftTokens, hidden: Any) -> VerifyOutput:
        raise NotImplementedError("Gemma 4 assistant verify is scaffolded but QA-pending")

    def propose(self, state: ModelState, hidden: Any) -> DraftTokens:
        raise NotImplementedError("Gemma 4 assistant propose is scaffolded but QA-pending")

    def recommended_profile(self) -> str:
        return DEFAULT_PROFILE_NAME

    def health(self) -> dict[str, Any]:
        return {
            "arch_id": self.arch_id,
            "backend": BACKEND_NAME,
            "support_level": "runtime-runnable-qa-pending",
            "can_run_public": True,
            "target_model": OFFICIAL_TARGET_REPO,
            "target_revision": OFFICIAL_TARGET_REVISION,
            "target_quantization": dict(CONSUMER_TARGET_QUANTIZATION),
            "assistant_model": OFFICIAL_ASSISTANT_REPO,
            "assistant_revision": OFFICIAL_ASSISTANT_REVISION,
            "assistant_dtype": PRIMARY_ASSISTANT_DTYPE,
            "draft_block_size": DEFAULT_DRAFT_BLOCK_SIZE,
            "upstream_draft_block_size_hints": {
                "single_request": 6,
                "batched_generation": 3,
                "mtplx_scaffold_default": DEFAULT_DRAFT_BLOCK_SIZE,
            },
            "exact_sampling_required": True,
            "batch_scope": "B=1 only until 160-token QA passes",
            "unsupported": ["ordered_embeddings", "E2B/E4B", "26B-A4B MoE", "multimodal generation"],
        }


def load_gemma4_assistant_pair(config: Gemma4AssistantRuntimeConfig) -> Gemma4AssistantRuntime:
    config.validate_static()
    from mlx_lm.utils import load as mlx_lm_load

    target_model, tokenizer = mlx_lm_load(str(config.target_model_path), lazy=True)
    assistant_model = load_gemma4_assistant_model(config.assistant_model_path)
    return Gemma4AssistantRuntime(
        target_model=target_model,
        tokenizer=tokenizer,
        assistant_model=assistant_model,
        config=config,
    )


def load_gemma4_assistant_model(assistant_path: Path | str):
    from mlx_lm.utils import load_model

    model, _config = load_model(
        Path(assistant_path),
        lazy=True,
        get_model_classes=_assistant_model_classes,
    )
    return model


def gemma4_exact_speculative_round(
    runtime: Gemma4AssistantRuntime,
    *,
    primary_token_id: int,
    hidden: Any,
    shared_kv_states: dict[str, Any],
    kv_offset: int,
    cache: Any,
    sampler: SamplerConfig,
    rng: np.random.Generator,
    allow_unverified: bool = False,
    draft_block_size: int | None = None,
    draft_sampler: SamplerConfig | None = None,
) -> Gemma4ExactRoundResult:
    """Run one B=1 exact Gemma assistant-backed speculative round.

    Gemma uses target-sampled prefix verification: the target samples one token
    from every verifier row, assistant drafts are accepted while they match that
    target-sampled prefix, and the first mismatch falls back to the target
    sample. This is exact for Gemma without p/q residual correction.
    """
    del allow_unverified

    mx = _require_mlx_core()

    distribution_mode = _target_distribution_mode(runtime.config.target_distribution_mode)
    proposal_sampler = draft_sampler or sampler
    draft_started = time.perf_counter()
    draft_block = runtime.propose_block(
        last_token_id=primary_token_id,
        hidden=hidden,
        shared_kv_states=shared_kv_states,
        kv_offset=kv_offset,
        sampler=proposal_sampler,
        rng=rng,
        draft_block_size=draft_block_size,
    )
    draft_time_s = time.perf_counter() - draft_started
    verify_started = time.perf_counter()
    lazy_target_sampling = float(getattr(sampler, "temperature", 0.0) or 0.0) > 0.0
    verify = runtime.verify_block(
        primary_token_id=primary_token_id,
        draft_token_ids=draft_block.token_ids,
        cache=cache,
        compute_logits=not lazy_target_sampling,
    )
    verify_time_s = time.perf_counter() - verify_started

    accepted: list[int] = []
    correction: int | None = None
    bonus: int | None = None
    target_distribution_time_s = 0.0
    target_hidden_time_s = 0.0
    row_distribution_evals = 0
    target_distribution_metadata: dict[str, Any] = {
        "oracle": "gemma4_target_prefix_exact",
        "exact": True,
        "p_q_residual": False,
        "materialized_windows": 0,
        "materialized_rows": 0,
        "sampling_strategy": (
            "row_lazy_logits" if lazy_target_sampling else "all_rows_logits"
        ),
        "target_sample_rows": int(len(draft_block.steps) + 1),
        "top_k": int(getattr(sampler, "top_k", 0) or 0),
        "batch_size": 1,
        "oracle_window_size": _gemma4_target_distribution_window_size(),
    }

    target_sample_started = time.perf_counter()
    if lazy_target_sampling:
        for depth in range(len(draft_block.steps) + 1):
            row_logits = runtime.target.logits_from_hidden(
                verify.output.hidden[:, depth : depth + 1, :]
            )
            target_token_arr = _sample_token_ids_from_logits_mlx(row_logits, sampler)
            mx.eval(target_token_arr)
            row_distribution_evals += 1
            target_token = int(target_token_arr.reshape(-1).tolist()[0])
            if depth < len(draft_block.steps):
                step = draft_block.steps[depth]
                if int(step.token_id) == target_token:
                    accepted.append(step.token_id)
                    continue
                correction = target_token
                break
            bonus = target_token
    else:
        target_token_arr = _sample_token_ids_from_logits_mlx(
            verify.output.logits[:, : len(draft_block.steps) + 1, :],
            sampler,
        )
        mx.eval(target_token_arr)
        target_tokens = [int(token) for token in target_token_arr.reshape(-1).tolist()]
        row_distribution_evals = len(target_tokens)
        for depth, step in enumerate(draft_block.steps):
            target_token = int(target_tokens[depth])
            if int(step.token_id) == target_token:
                accepted.append(step.token_id)
                continue
            correction = target_token
            break
        if correction is None:
            bonus = int(target_tokens[len(draft_block.steps)])
    target_distribution_time_s = time.perf_counter() - target_sample_started
    target_distribution_metadata["materialized_rows"] = int(row_distribution_evals)
    target_distribution_metadata["materialized_windows"] = 1 if row_distribution_evals else 0
    accept_started = time.perf_counter()
    accept_time_s = time.perf_counter() - accept_started

    rejected = len(verify.input_token_ids) - (1 + len(accepted))
    next_hidden = verify.output.hidden[:, len(accepted) : len(accepted) + 1, :]
    next_shared_kv_states = _slice_shared_kv_states(
        verify.output.shared_kv_states,
        rejected_tokens=rejected,
    )
    rollback_started = time.perf_counter()
    runtime.target.rollback_speculative_cache(
        cache,
        verified_tokens=len(verify.input_token_ids),
        committed_tokens=1 + len(accepted),
    )
    rollback_time_s = time.perf_counter() - rollback_started
    next_primary = bonus if correction is None else correction
    return Gemma4ExactRoundResult(
        accepted_token_ids=tuple(accepted),
        corrected_token_id=None if correction is None else int(correction),
        bonus_token_id=None if bonus is None else int(bonus),
        next_primary_token_id=None if next_primary is None else int(next_primary),
        accepted_count=len(accepted),
        metadata={
            "arch_id": ARCH_ID,
            "backend": BACKEND_NAME,
            "qa_status": "runtime_integrated_app_qa_pending",
            "draft_block_size": int(runtime.config.draft_block_size),
            "all_drafts_accepted": correction is None,
            "rejected_verified_tokens": int(rejected),
            "timing_s": {
                "draft": float(draft_time_s),
                "verify": float(verify_time_s),
                "target_hidden": float(target_hidden_time_s),
                "target_distribution": float(target_distribution_time_s),
                "accept": float(accept_time_s),
                "rollback": float(rollback_time_s),
            },
            "target_distribution_mode": distribution_mode,
            "row_distribution_evals": int(row_distribution_evals),
            "target_distribution_metadata": dict(target_distribution_metadata),
            "cache_counters": dict(verify.output.cache_counters),
            "cache_offset": int(verify.output.cache_offset),
            "telemetry": runtime.telemetry.to_dict(),
            "compiled_verify": dict(runtime.compile_stats),
            "compiled_distribution": dict(runtime.distribution_compile_stats),
            "long_context_policy": runtime.config.long_context_policy.to_dict(),
        },
        next_hidden=next_hidden,
        next_shared_kv_states=next_shared_kv_states,
        next_kv_offset=runtime.target.cache_offset(cache),
    )


def _assistant_model_classes(config: dict[str, Any]):
    from mlx_lm.models import gemma4_text

    import mlx.core as mx
    import mlx.nn as nn

    def _shared_kv_decoder_layer(text_config: Any, index: int):
        try:
            layer = gemma4_text.DecoderLayer(
                text_config,
                layer_idx=index,
                kv_shared_only=True,
            )
        except TypeError:
            layer = gemma4_text.DecoderLayer(text_config, layer_idx=index)
        attention = getattr(layer, "self_attn", None)
        if getattr(attention, "is_kv_shared_layer", True) is False:
            raise Gemma4AssistantUnsupported(
                "Gemma assistant decoder layer is not shared-KV-only"
            )
        return layer

    class _DraftInner(nn.Module):
        def __init__(self, text_config: Any):
            super().__init__()
            self.config = text_config
            self.embed_tokens = nn.Embedding(text_config.vocab_size, text_config.hidden_size)
            self.layers = [
                _shared_kv_decoder_layer(text_config, index)
                for index in range(text_config.num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)

    class Gemma4AssistantDraftModel(nn.Module):
        def __init__(self, args: Gemma4AssistantArgs):
            super().__init__()
            args.validate_31b_dense()
            text_config = gemma4_text.ModelArgs.from_dict(dict(args.text_config or {}))
            text_config.num_kv_shared_layers = text_config.num_hidden_layers
            self.args = args
            self.config = args
            self.text_config = text_config
            self.model = _DraftInner(text_config)
            self.pre_projection = nn.Linear(
                2 * args.backbone_hidden_size,
                text_config.hidden_size,
                bias=False,
            )
            self.post_projection = nn.Linear(
                text_config.hidden_size,
                args.backbone_hidden_size,
                bias=False,
            )
            if not args.tie_word_embeddings:
                self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False)
            self._lm_head_fn = None
            self._input_embed = None
            self._input_embed_scale = 1.0
            self._shared_kv = None
            self._position = 0
            self._kv_valid_len = 0

        def bind(self, target: Any):
            if self.config.use_ordered_embeddings:
                raise Gemma4AssistantUnsupported("centroid-routed assistant heads are out of scope")
            if self.config.tie_word_embeddings:
                self._lm_head_fn = self.model.embed_tokens.as_linear
            else:
                self._lm_head_fn = self.lm_head
            text_model = getattr(target, "text_model", None)
            if text_model is None:
                language_model = getattr(target, "language_model", target)
                text_model = getattr(language_model, "model", language_model)
            self._input_embed = text_model.embed_tokens
            self._input_embed_scale = float(getattr(text_model, "embed_scale", 1.0))
            target_config = getattr(text_model, "config", None)
            if target_config is not None:
                self.config.target_layer_types[:] = list(
                    getattr(target_config, "layer_types", []) or []
                )
            return self

        def make_cache(self):
            return []

        def reset(self, target: Any):
            self.bind(target)
            self._shared_kv = None
            self._position = 0
            self._kv_valid_len = 0
            return self.make_cache()

        def set_shared_kv(
            self,
            shared_kv_states: dict[str, Any],
            kv_offset: Any,
            position: Any = None,
            kv_valid_len: Any = None,
            left_padding: Any = None,
        ) -> None:
            if kv_valid_len is None:
                kv_valid_len = kv_offset
            if left_padding is not None:
                shared_kv_states = normalize_batched_shared_kv_states(
                    shared_kv_states,
                    kv_valid_len=kv_valid_len,
                    left_padding=left_padding,
                )
            self._shared_kv = shared_kv_states
            self._position = kv_offset if position is None else position
            self._kv_valid_len = kv_valid_len

        def draft_step(self, token: Any, previous_hidden: Any) -> tuple[Any, Any]:
            if self._shared_kv is None:
                raise RuntimeError("set_shared_kv() must be called before draft_step()")
            if self._input_embed is None:
                raise RuntimeError("bind(target) must be called before draft_step()")
            token_embed = self._input_embed(token) * self._input_embed_scale
            inputs_embeds = mx.concatenate([token_embed, previous_hidden], axis=-1)
            return self(inputs_embeds)

        def __call__(self, inputs_embeds: Any) -> tuple[Any, Any]:
            position_ids = _position_ids(self._position)
            h = self.pre_projection(inputs_embeds)
            query_len = h.shape[1]
            query_offset = (
                int(position_ids[0, 0].item())
                if position_ids.shape[0] == 1
                else position_ids[:, 0]
            )
            masks = make_drafter_masks(
                self._shared_kv,
                query_len=query_len,
                query_offset=query_offset,
                sliding_window=self.text_config.sliding_window,
                dtype=h.dtype,
                kv_valid_len=self._kv_valid_len,
            )
            offset = mx.array(query_offset) if isinstance(query_offset, int) else query_offset
            for layer in self.model.layers:
                kv = self._shared_kv[layer.layer_type]
                h, _, _ = layer(
                    h,
                    masks.get(layer.layer_type),
                    None,
                    per_layer_input=None,
                    shared_kv=kv,
                    offset=offset,
                )
            h = self.model.norm(h)
            next_hidden = self.post_projection(h)
            logits = self._lm_head_fn(h)
            return next_hidden, logits

        def sanitize(self, weights: dict[str, Any]) -> dict[str, Any]:
            sanitized = {}
            for key, value in weights.items():
                if key == "masked_embedding.token_ordering":
                    continue
                if key == "lm_head.weight" and self.config.tie_word_embeddings:
                    continue
                sanitized[key] = value
            return sanitized

    return Gemma4AssistantDraftModel, Gemma4AssistantArgs


def _position_ids(position: Any):
    mx = _require_mlx_core()
    if isinstance(position, int):
        return mx.array([[position]])
    if hasattr(position, "shape"):
        if getattr(position, "ndim", 0) == 0:
            return position.reshape((1, 1))
        if getattr(position, "ndim", 0) == 1:
            return position[:, None]
        return position
    return mx.array(position)


def _slice_shared_kv_states(
    shared_kv_states: dict[str, Any],
    *,
    rejected_tokens: int,
) -> dict[str, Any]:
    if int(rejected_tokens) <= 0:
        return dict(shared_kv_states)
    out: dict[str, Any] = {}
    for layer_type, kv in shared_kv_states.items():
        keys, values = kv
        valid = max(1, int(keys.shape[-2]) - int(rejected_tokens))
        out[layer_type] = (keys[..., :valid, :], values[..., :valid, :])
    return out


def _bidirectional_swa_mask(
    *,
    query_len: int,
    query_offset: Any,
    kv_len: int,
    window: int,
    kv_valid_len: Any = None,
    key_offset: Any = 0,
    dtype: Any,
):
    mx = _require_mlx_core()
    if (
        isinstance(query_offset, int)
        and (kv_valid_len is None or isinstance(kv_valid_len, int))
        and isinstance(key_offset, int)
        and kv_len <= window
        and query_offset - key_offset < window
        and key_offset + kv_len - (query_offset + query_len) < window
    ):
        return None
    if isinstance(query_offset, int):
        if hasattr(key_offset, "size") and getattr(key_offset, "size", 0) == 1:
            key_offset = int(key_offset.item())
        if hasattr(kv_valid_len, "size") and getattr(kv_valid_len, "size", 0) == 1:
            kv_valid_len = int(kv_valid_len.item())
        q_idx = mx.arange(query_offset, query_offset + query_len)[:, None]
        k_idx_2d = mx.arange(key_offset, key_offset + kv_len)[None, :]
        dist = q_idx - k_idx_2d
        inside = (dist > -window) & (dist < window)
        if kv_valid_len is not None:
            inside = inside & (k_idx_2d < int(kv_valid_len))
        bias = mx.where(
            inside,
            mx.array(0.0, dtype=dtype),
            mx.array(-mx.inf, dtype=dtype),
        )
        return bias[None, None, :, :]

    offsets = query_offset
    q_idx = offsets[:, None] + mx.arange(query_len)[None, :]
    if hasattr(key_offset, "shape"):
        k_idx_3d = key_offset[:, None, None] + mx.arange(kv_len)[None, None, :]
    else:
        k_idx_3d = mx.arange(key_offset, key_offset + kv_len)[None, None, :]
    dist = q_idx[:, :, None] - k_idx_3d
    inside = (dist > -window) & (dist < window)
    if kv_valid_len is not None:
        valid = kv_valid_len if not isinstance(kv_valid_len, int) else mx.array(kv_valid_len)
        inside = inside & (k_idx_3d < valid[:, None, None])
    bias = mx.where(
        inside,
        mx.array(0.0, dtype=dtype),
        mx.array(-mx.inf, dtype=dtype),
    )
    return bias[:, None, :, :]


def _bidirectional_full_mask(
    *,
    query_len: int,
    kv_len: int,
    kv_valid_len: Any = None,
    key_offset: Any = 0,
    dtype: Any,
):
    del query_len
    mx = _require_mlx_core()
    if kv_valid_len is None:
        return None
    if isinstance(kv_valid_len, int):
        if isinstance(key_offset, int) and key_offset + kv_len <= kv_valid_len:
            return None
        k_idx = mx.arange(key_offset, key_offset + kv_len)
        inside = k_idx < int(kv_valid_len)
        bias = mx.where(
            inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype)
        )
        return bias[None, None, None, :]
    if hasattr(key_offset, "shape"):
        k_idx = key_offset[:, None] + mx.arange(kv_len)[None, :]
    else:
        k_idx = mx.arange(key_offset, key_offset + kv_len)[None, :]
    inside = k_idx < kv_valid_len[:, None]
    bias = mx.where(inside, mx.array(0.0, dtype=dtype), mx.array(-mx.inf, dtype=dtype))
    return bias[:, None, None, :]


def _kv_len(kv: Any) -> int:
    keys, _values = kv
    return int(keys.shape[-2])


def _shape_dim(value: Any, axis: int, *, fallback: int) -> int:
    shape = getattr(value, "shape", None)
    if shape is None:
        return int(fallback)
    try:
        return int(shape[axis])
    except Exception:
        return int(fallback)


def _offset_to_int(value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    if hasattr(value, "item"):
        mx = _require_mlx_core()
        mx.eval(value)
        try:
            return int(value.item())
        except ValueError:
            return int(value.max().item())
    return int(value)


def _cache_counters(cache: Any) -> Gemma4CacheCounters:
    totals = {name: 0 for name in GEMMA4_CACHE_COUNTERS}
    for item in _walk_cache_items(cache):
        stats = item if isinstance(item, dict) else None
        if stats is None:
            raw_stats = getattr(item, "stats", None)
            stats = raw_stats if isinstance(raw_stats, dict) else {}
        for name in GEMMA4_CACHE_COUNTERS:
            raw_value = stats.get(name) if isinstance(stats, dict) else None
            if raw_value is None:
                raw_value = getattr(item, name, 0)
            try:
                totals[name] += int(raw_value)
            except Exception:
                continue
    return Gemma4CacheCounters(**totals)


def _walk_cache_items(cache: Any):
    if cache is None:
        return
    if isinstance(cache, dict):
        yield cache
        for value in cache.values():
            yield from _walk_cache_items(value)
        return
    if isinstance(cache, (list, tuple)):
        for value in cache:
            yield from _walk_cache_items(value)
        return
    yield cache


def _target_distribution_mode(configured: str | None = None) -> str:
    import os

    value = os.environ.get(
        "MTPLX_GEMMA4_TARGET_DISTRIBUTIONS",
        str(configured or DEFAULT_TARGET_DISTRIBUTION_MODE),
    )
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {
        "gemma4_target_prefix_exact",
        "target_prefix",
        "target_prefix_exact",
        "prefix_walk",
        "prefix_walk_debug",
        "sampled_prefix",
        "mlx_vlm_prefix_walk",
        # Backward-compatible names from the abandoned Gemma p/q oracle lane.
        "gemma4_sparse_head",
        "sparse_head",
        "sparse",
        "progressive",
        "progressive_hidden",
        "windowed",
        "windowed_hidden",
        "adaptive_hidden",
        "row_lazy_hidden",
        "hidden",
        "batched",
        "batch",
        "batched_logits",
        "batched_logits_debug",
        "all_rows",
        "bonus_lazy",
        "bonus_lazy_logits",
        "lazy_bonus",
        "draft_batched_bonus_lazy",
        "dense_logits_topk_debug",
        "fused",
        "fused_logits",
        "fused_logits_topk",
        "metal_logits_topk",
        "logits_topk",
        "certified",
        "certified_topk",
        "certified_top_k",
        "topk_certified",
        "row_logits",
        "lazy_logits",
        "row_lazy_logits",
        "row_lazy_logits_debug",
        "logits",
    }:
        return "gemma4_target_prefix_exact"
    raise ValueError("MTPLX_GEMMA4_TARGET_DISTRIBUTIONS must be 'gemma4_target_prefix_exact'")


def _gemma4_target_distribution_window_size() -> int:
    import os

    try:
        return max(
            1,
            int(
                os.environ.get(
                    "MTPLX_GEMMA4_TARGET_DISTRIBUTION_WINDOW",
                    str(DEFAULT_TARGET_DISTRIBUTION_WINDOW_SIZE),
                )
            ),
        )
    except (TypeError, ValueError):
        return DEFAULT_TARGET_DISTRIBUTION_WINDOW_SIZE


def _gemma4_compile_verify_enabled() -> bool:
    import os

    return _env_truthy(os.environ.get("MTPLX_GEMMA4_COMPILE_VERIFY"))


def _gemma4_compile_distributions_enabled() -> bool:
    import os

    return _env_truthy(os.environ.get("MTPLX_GEMMA4_COMPILE_DISTRIBUTIONS", "1"))


def _gemma4_force_hidden_eval_before_distribution() -> bool:
    import os

    return _env_truthy(
        os.environ.get("MTPLX_GEMMA4_EVAL_TARGET_HIDDEN_BEFORE_DISTRIBUTION")
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_mlx_core():
    import mlx.core as mx

    return mx


def _ensure_thread_streams() -> None:
    if getattr(_THREAD_LOCAL, "gemma4_streams_ready", False):
        return
    mx = _require_mlx_core()
    try:
        # The Gemma target/assistant are loaded on the server thread, then
        # generation runs in MTPLX's serialized worker.  MLX stream ids are
        # thread-local, so create the first non-default GPU stream in the worker
        # before evaluating lazy arrays that were queued on Stream(gpu, 1).
        mx.new_stream(mx.gpu)
    except Exception:
        pass
    _THREAD_LOCAL.gemma4_streams_ready = True


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def _stop_token_ids(tokenizer: Any) -> set[int]:
    values: set[int] = set()
    for name in ("eos_token_ids", "eos_token_id", "pad_token_id"):
        raw = getattr(tokenizer, name, None)
        if isinstance(raw, (list, tuple, set)):
            values.update(int(item) for item in raw if item is not None)
        elif raw is not None:
            try:
                values.add(int(raw))
            except (TypeError, ValueError):
                pass
    return values


def _strip_terminal_stop(tokens: list[int], stop_ids: set[int]) -> list[int]:
    while tokens and int(tokens[-1]) in stop_ids:
        tokens = tokens[:-1]
    return tokens


def _emit_gemma_token(
    token_id: int,
    *,
    stop_ids: set[int],
    token_callback: Any | None,
) -> None:
    if token_callback is not None and int(token_id) not in stop_ids:
        token_callback([int(token_id)])


class _Gemma4RepetitionAwareWire:
    """Armed-stream holdback for the repetition stop (F35), gemma4 loops.

    Same contract as the serial-lane fix in :mod:`mtplx.generation`, whose
    ``_repetition_stream_holdback_tokens`` / ``_repetition_stream_emit_limit``
    are imported (not duplicated): while the uncapped repetition stop is
    armed, the wire must never outrun a future trim — both gemma4 loops used
    to emit every token BEFORE ``_trim_repeated_suffix`` ran, so a fired trim
    left already-streamed garbage on the wire (stream != non-stream, wire >
    usage). A detector-window tail is held back and reconciled by ``flush``
    once the trim decision is known. Disarmed requests keep the historical
    per-token ``_emit_gemma_token`` pattern byte for byte.
    """

    def __init__(
        self,
        tokens: list[int],
        *,
        config: Any,
        stop_ids: set[int],
        token_callback: Any | None,
    ) -> None:
        from mtplx.generation import (
            _repetition_stream_emit_limit,
            _repetition_stream_holdback_tokens,
        )

        self._tokens = tokens
        self._config = config
        self._stop_ids = stop_ids
        self._callback = token_callback
        self._emit_limit = _repetition_stream_emit_limit
        self._holdback = (
            _repetition_stream_holdback_tokens(config)
            if token_callback is not None
            else 0
        )
        self._streamed = 0

    def emit(self) -> None:
        """Release whatever is wire-safe after a token was appended."""

        if self._callback is None:
            return
        if self._holdback <= 0:
            _emit_gemma_token(
                self._tokens[-1],
                stop_ids=self._stop_ids,
                token_callback=self._callback,
            )
            return
        limit = self._emit_limit(len(self._tokens), self._config, self._holdback)
        if limit > self._streamed:
            released = [
                int(token)
                for token in self._tokens[self._streamed : limit]
                if int(token) not in self._stop_ids
            ]
            self._streamed = limit
            if released:
                self._callback(released)

    def flush(self) -> None:
        """Post-loop reconcile: the trim decision is known here — release
        the held tail in full (no trim) or the post-trim remainder."""

        if self._callback is None or self._holdback <= 0:
            return
        self._streamed = min(self._streamed, len(self._tokens))
        held = [
            int(token)
            for token in self._tokens[self._streamed :]
            if int(token) not in self._stop_ids
        ]
        self._streamed = len(self._tokens)
        if held:
            self._callback(held)


def _gemma4_session_extra_state(
    *,
    shared_kv_states: dict[str, Any],
    kv_offset: int,
) -> dict[str, Any]:
    return {
        "gemma4_shared_kv_states": dict(shared_kv_states),
        "gemma4_kv_offset": int(kv_offset),
        "gemma4_session_state_policy": GEMMA4_SESSION_STATE_POLICY,
    }


def _gemma4_prefill_prompt(
    runtime: Gemma4AssistantRuntime,
    prompt_ids: list[int],
    *,
    cache: Any,
    phase: str,
) -> tuple[Gemma4TargetOutput, float]:
    mx = _require_mlx_core()
    started = time.perf_counter()
    output = runtime.forward_target(
        mx.array([prompt_ids], dtype=mx.int32),
        cache=cache,
        phase=phase,
        compute_logits=False,
    )
    last_hidden = output.hidden[:, -1:, :]
    last_logits = runtime.target.logits_from_hidden(last_hidden)
    mx.eval(last_logits, last_hidden)
    return (
        Gemma4TargetOutput(
            logits=last_logits,
            hidden=last_hidden,
            shared_kv_states=output.shared_kv_states,
            cache_offset=output.cache_offset,
            attention_phase=output.attention_phase,
            cache_counters=dict(output.cache_counters),
        ),
        time.perf_counter() - started,
    )


def _clone_gemma4_prompt_cache(
    runtime: Gemma4AssistantRuntime,
    cache: list[Any],
) -> list[Any]:
    """Retain the pre-decode cache before sliding layers evict its history."""

    from mtplx.cache_state import restore_cache, snapshot_cache_lazy_hybrid

    clone = runtime.make_cache()
    restore_cache(
        clone,
        snapshot_cache_lazy_hybrid(cache),
        clone_states=False,
    )
    return clone


def _restore_or_prefill_gemma4_prompt(
    runtime: Gemma4AssistantRuntime,
    prompt_ids: list[int],
    *,
    session_bank: Any | None = None,
    session_restore_mode: str = "clone",
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    require_shared_kv: bool,
) -> Gemma4PromptState:
    if not prompt_ids:
        raise ValueError("prompt_ids must not be empty for Gemma 4 generation")

    def cold_prefill(reason: str | None = None) -> Gemma4PromptState:
        cache = runtime.make_cache()
        output, elapsed = _gemma4_prefill_prompt(
            runtime,
            prompt_ids,
            cache=cache,
            phase="prefill",
        )
        return Gemma4PromptState(
            cache=cache,
            logits=output.logits[:, -1, :],
            hidden=output.hidden[:, -1:, :],
            shared_kv_states=output.shared_kv_states,
            kv_offset=int(output.cache_offset),
            prompt_eval_time_s=float(elapsed),
            cached_tokens=0,
            suffix_tokens=len(prompt_ids),
            cache_hit=False,
            cache_miss_reason=reason
            or getattr(session_bank, "last_miss_reason", None),
            restore_mode="cold",
        )

    if session_bank is None:
        return cold_prefill()

    started = time.perf_counter()
    restored = session_bank.restore(
        runtime,
        prompt_ids,
        mode=session_restore_mode,
        hidden_variant="gemma4_pre_norm",
        template_hash=session_template_hash,
        mtp_history_policy=GEMMA4_SESSION_STATE_POLICY,
        draft_head_identity=session_draft_head_identity,
        policy_fingerprint=session_policy_fingerprint,
    )
    if restored is None:
        candidates = getattr(session_bank, "near_prefix_candidates", None)
        restore_prefix = getattr(session_bank, "restore_entry_prefix_cache", None)
        if callable(candidates) and callable(restore_prefix):
            try:
                near_matches = candidates(
                    prompt_ids,
                    model_path=str(runtime.model_path),
                    mtp_enabled=bool(runtime.mtp_enabled),
                    hidden_variant="gemma4_pre_norm",
                    template_hash=session_template_hash,
                    mtp_history_policy=GEMMA4_SESSION_STATE_POLICY,
                    draft_head_identity=session_draft_head_identity,
                    policy_fingerprint=session_policy_fingerprint,
                )
            except Exception:
                near_matches = []
            for entry, matched in near_matches:
                matched = int(matched)
                if (
                    matched < 2
                    or matched >= int(getattr(entry, "prefix_len", 0) or 0)
                    or matched >= len(prompt_ids)
                    or str(getattr(entry, "model_path", ""))
                    != str(runtime.model_path)
                    or bool(getattr(entry, "mtp_enabled", False))
                    != bool(runtime.mtp_enabled)
                    or getattr(entry, "hidden_variant", None) != "gemma4_pre_norm"
                    or (
                        session_template_hash is not None
                        and getattr(entry, "template_hash", None)
                        != session_template_hash
                    )
                    or getattr(entry, "mtp_history_policy", None)
                    != GEMMA4_SESSION_STATE_POLICY
                    or (
                        session_draft_head_identity is not None
                        and getattr(entry, "draft_head_identity", None)
                        != session_draft_head_identity
                    )
                    or (
                        session_policy_fingerprint is not None
                        and getattr(entry, "policy_fingerprint", None)
                        != session_policy_fingerprint
                    )
                ):
                    continue
                try:
                    prefix_restore = restore_prefix(
                        runtime,
                        entry,
                        matched,
                        mode=session_restore_mode,
                        full_boundary=True,
                    )
                except Exception:
                    prefix_restore = None
                if prefix_restore is None:
                    continue
                boundary_hidden = None
                if len(prefix_restore) == 5:
                    cache, _history, storage_mode, restore_point, boundary_hidden = (
                        prefix_restore
                    )
                elif len(prefix_restore) == 4:
                    cache, _history, storage_mode, restore_point = prefix_restore
                else:
                    cache, _history, storage_mode = prefix_restore
                    restore_point = matched
                restore_point = int(restore_point)
                prefill_start = restore_point
                if prefill_start < 0 or prefill_start >= len(prompt_ids):
                    continue
                output, _suffix_elapsed = _gemma4_prefill_prompt(
                    runtime,
                    list(prompt_ids[prefill_start:]),
                    cache=cache,
                    phase="prefill",
                )
                entry.hits += 1
                entry.last_access_s = time.time()
                return Gemma4PromptState(
                    cache=cache,
                    logits=output.logits[:, -1, :],
                    hidden=output.hidden[:, -1:, :],
                    shared_kv_states=output.shared_kv_states,
                    kv_offset=int(output.cache_offset),
                    prompt_eval_time_s=time.perf_counter() - started,
                    cached_tokens=restore_point,
                    suffix_tokens=len(prompt_ids) - restore_point,
                    cache_hit=True,
                    cache_miss_reason=None,
                    restore_mode=f"block_prefix_{storage_mode}",
                )
        return cold_prefill(getattr(session_bank, "last_miss_reason", None))

    suffix = list(prompt_ids[restored.entry.prefix_len :])
    if suffix:
        output, suffix_elapsed = _gemma4_prefill_prompt(
            runtime,
            suffix,
            cache=restored.cache,
            phase="prefill",
        )
        return Gemma4PromptState(
            cache=restored.cache,
            logits=output.logits[:, -1, :],
            hidden=output.hidden[:, -1:, :],
            shared_kv_states=output.shared_kv_states,
            kv_offset=int(output.cache_offset),
            prompt_eval_time_s=time.perf_counter() - started,
            cached_tokens=int(restored.entry.prefix_len),
            suffix_tokens=len(suffix),
            cache_hit=True,
            cache_miss_reason=None,
            restore_mode=str(restored.restore_mode or session_restore_mode),
        )

    extra_state = restored.extra_state if isinstance(restored.extra_state, dict) else {}
    shared_kv_states = extra_state.get("gemma4_shared_kv_states")
    if require_shared_kv and not isinstance(shared_kv_states, dict):
        return cold_prefill("gemma4_shared_kv_missing")
    if not isinstance(shared_kv_states, dict):
        shared_kv_states = {}
    kv_offset = extra_state.get("gemma4_kv_offset")
    if kv_offset is None:
        kv_offset = runtime.target.cache_offset(restored.cache)
    return Gemma4PromptState(
        cache=restored.cache,
        logits=restored.logits,
        hidden=restored.hidden,
        shared_kv_states=dict(shared_kv_states),
        kv_offset=int(kv_offset),
        prompt_eval_time_s=time.perf_counter() - started,
        cached_tokens=int(restored.entry.prefix_len),
        suffix_tokens=0,
        cache_hit=True,
        cache_miss_reason=None,
        restore_mode=str(restored.restore_mode or session_restore_mode),
    )


def generate_gemma4_ar(
    runtime: Gemma4AssistantRuntime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: SamplerConfig,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    token_callback: Any | None = None,
    trace_label: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    prefill_callback: Any | None = None,
    repetition_stop: bool = False,
    session_bank: Any | None = None,
    session_restore_mode: str = "clone",
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
):
    """Generate with the Gemma target only, using the local target adapter."""

    _ensure_thread_streams()
    mx = _require_mlx_core()
    from mtplx.generation import (
        GenerationFinalState,
        GenerationOutput,
        GenerationStats,
        _generation_rate_fields,
        _repetition_stop_config,
        _sample_from_logits,
        _trim_repeated_suffix,
    )

    del trace_label, trace_metadata
    rng = np.random.default_rng(int(seed))
    try:
        mx.random.seed(int(seed))
    except Exception:
        pass
    stop_ids = set(stop_token_ids) if stop_token_ids is not None else _stop_token_ids(runtime.tokenizer)
    started_all = time.perf_counter()
    prefill_started_s = time.perf_counter()
    if prefill_callback is not None:
        try:
            prefill_callback(
                {
                    "phase": "started",
                    "tokens_done": 0,
                    "tokens_total": int(len(prompt_ids)),
                    "cached_tokens": 0,
                    "new_prefill_tokens": int(len(prompt_ids)),
                    "elapsed_s": 0.0,
                    "started_s": prefill_started_s,
                }
            )
        except Exception:
            pass
    prompt_state = _restore_or_prefill_gemma4_prompt(
        runtime,
        prompt_ids,
        session_bank=session_bank,
        session_restore_mode=session_restore_mode,
        session_template_hash=session_template_hash,
        session_draft_head_identity=session_draft_head_identity,
        session_policy_fingerprint=session_policy_fingerprint,
        require_shared_kv=False,
    )
    if prefill_callback is not None:
        try:
            elapsed = max(0.0, time.perf_counter() - prefill_started_s)
            new_tokens = int(prompt_state.suffix_tokens)
            tok_s = (new_tokens / elapsed) if elapsed > 0.0 and new_tokens else None
            prefill_callback(
                {
                    "phase": "completed",
                    "tokens_total": int(len(prompt_ids)),
                    "tokens_done": int(len(prompt_ids)),
                    "cached_tokens": int(prompt_state.cached_tokens),
                    "new_prefill_tokens": new_tokens,
                    "elapsed_s": elapsed,
                    "prompt_eval_time_s": float(prompt_state.prompt_eval_time_s),
                    "prefill_tok_s": tok_s,
                    "prefill_compute_tok_s": tok_s,
                    "prefill_wall_tok_s": tok_s,
                    "cache_hit": bool(prompt_state.cache_hit),
                    "cache_miss_reason": prompt_state.cache_miss_reason,
                }
            )
        except Exception:
            pass
    cache = prompt_state.cache
    prompt_boundary_cache = (
        _clone_gemma4_prompt_cache(runtime, cache) if capture_final_state else None
    )
    prompt_boundary_extra_state = _gemma4_session_extra_state(
        shared_kv_states=prompt_state.shared_kv_states,
        kv_offset=int(prompt_state.kv_offset),
    )
    logits = prompt_state.logits
    hidden = prompt_state.hidden
    shared_kv_states = prompt_state.shared_kv_states
    kv_offset = int(prompt_state.kv_offset)
    tokens: list[int] = []
    target_decode_time = 0.0
    verify_calls = 0
    events: list[dict[str, Any]] = []
    pending_token_needs_commit = False
    repetition_config = _repetition_stop_config(bool(repetition_stop))
    repetition_result = None
    wire = _Gemma4RepetitionAwareWire(
        tokens,
        config=repetition_config,
        stop_ids=stop_ids,
        token_callback=token_callback,
    )

    for step in range(int(max_tokens)):
        token, _dist = _sample_from_logits(logits[0], sampler, rng)
        token = int(token)
        tokens.append(token)
        pending_token_needs_commit = True
        events.append({"step": int(step), "token": token})
        wire.emit()
        repetition_result = _trim_repeated_suffix(tokens, repetition_config)
        if repetition_result is not None:
            events.append(
                {
                    "step": int(step),
                    "repetition_stop": {
                        "reason": "exact_repeated_token_suffix",
                        "block_tokens": repetition_result.block_tokens,
                        "repeats": repetition_result.repeats,
                        "trimmed_tokens": repetition_result.repeated_tokens,
                    },
                }
            )
            break
        if step + 1 >= int(max_tokens) or token in stop_ids:
            break

        target_started = time.perf_counter()
        output = runtime.forward_target(
            mx.array([[token]], dtype=mx.int32),
            cache=cache,
            phase="ar_decode",
        )
        mx.eval(output.logits, output.hidden)
        target_decode_time += time.perf_counter() - target_started
        verify_calls += 1
        logits = output.logits[:, -1, :]
        hidden = output.hidden[:, -1:, :]
        shared_kv_states = output.shared_kv_states
        kv_offset = int(output.cache_offset)
        pending_token_needs_commit = False

    wire.flush()
    final_state = None
    finish_reason = "stop" if any(token in stop_ids for token in tokens) else "length"
    if capture_final_state and pending_token_needs_commit and tokens:
        commit_started = time.perf_counter()
        output = runtime.forward_target(
            mx.array([[int(tokens[-1])]], dtype=mx.int32),
            cache=cache,
            phase="postcommit",
        )
        mx.eval(output.logits, output.hidden)
        target_decode_time += time.perf_counter() - commit_started
        verify_calls += 1
        logits = output.logits[:, -1, :]
        hidden = output.hidden[:, -1:, :]
        shared_kv_states = output.shared_kv_states
        kv_offset = int(output.cache_offset)
        pending_token_needs_commit = False

    elapsed = time.perf_counter() - started_all
    extra_state = _gemma4_session_extra_state(
        shared_kv_states=shared_kv_states,
        kv_offset=kv_offset,
    )
    if capture_final_state:
        final_state = GenerationFinalState(
            final_trunk_cache=cache,
            final_logits=logits,
            final_hidden=hidden,
            final_committed_mtp_cache=None,
            generated_token_ids=tuple(int(token) for token in tokens),
            safe_to_commit=not pending_token_needs_commit,
            finish_reason=finish_reason,
            extra_state=extra_state,
            prompt_boundary_cache=prompt_boundary_cache,
            prompt_boundary_logits=prompt_state.logits,
            prompt_boundary_hidden=prompt_state.hidden,
            prompt_boundary_extra_state=prompt_boundary_extra_state,
        )
    stats = GenerationStats(
        mode="ar",
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        **_generation_rate_fields(
            generated_tokens=len(tokens),
            elapsed_s=elapsed,
            prompt_eval_time_s=prompt_state.prompt_eval_time_s,
        ),
        runtime_mtp_enabled=False,
        prompt_eval_time_s=prompt_state.prompt_eval_time_s,
        cached_tokens=int(prompt_state.cached_tokens),
        new_prefill_tokens=int(prompt_state.suffix_tokens),
        session_cache_hit=bool(prompt_state.cache_hit),
        cache_miss_reason=prompt_state.cache_miss_reason,
        session_restore_mode=prompt_state.restore_mode,
        target_forward_time_s=prompt_state.prompt_eval_time_s + target_decode_time,
        verify_time_s=target_decode_time,
        verify_forward_time_s=target_decode_time,
        verify_eval_time_s=target_decode_time,
        verify_joint_eval_time_s=target_decode_time,
        verify_calls=verify_calls,
        peak_memory_bytes=int(mx.get_peak_memory()),
        repetition_stop_triggered=repetition_result is not None,
        repetition_stop_reason=(
            "exact_repeated_token_suffix" if repetition_result is not None else None
        ),
        repetition_stop_block_tokens=(
            0 if repetition_result is None else repetition_result.block_tokens
        ),
        repetition_stop_repeats=(
            0 if repetition_result is None else repetition_result.repeats
        ),
        repetition_stop_trimmed_tokens=(
            0 if repetition_result is None else repetition_result.repeated_tokens
        ),
        repetition_stop_raw_tokens=(
            0 if repetition_result is None else len(tokens) + repetition_result.repeated_tokens
        ),
        events=events,
        owned_attn_kv=runtime.telemetry.to_dict(),
    )
    return GenerationOutput(
        tokens=tokens,
        text=_decode_tokens(runtime.tokenizer, _strip_terminal_stop(tokens, stop_ids)),
        stats=stats,
        final_state=final_state,
    )


def generate_gemma4_assistant(
    runtime: Gemma4AssistantRuntime,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    sampler: SamplerConfig,
    draft_sampler: SamplerConfig | None = None,
    speculative_depth: int = DEFAULT_DRAFT_BLOCK_SIZE,
    adaptive_draft: bool = False,
    seed: int = 0,
    stop_token_ids: set[int] | None = None,
    token_callback: Any | None = None,
    session_bank: Any | None = None,
    session_restore_mode: str = "clone",
    session_template_hash: str | None = None,
    session_draft_head_identity: str | None = None,
    session_policy_fingerprint: str | None = None,
    capture_final_state: bool = False,
    trace_label: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    prefill_callback: Any | None = None,
    repetition_stop: bool = False,
    requested_speculative_depth: int | None = None,
):
    """Generate with the external Gemma assistant using target-prefix exactness."""

    _ensure_thread_streams()
    mx = _require_mlx_core()
    from mtplx.generation import (
        GenerationFinalState,
        GenerationOutput,
        GenerationStats,
        _generation_rate_fields,
        _repetition_stop_config,
        _sample_from_logits,
        _trim_repeated_suffix,
    )

    del trace_label, trace_metadata
    rng = np.random.default_rng(int(seed))
    try:
        mx.random.seed(int(seed))
    except Exception:
        pass
    stop_ids = set(stop_token_ids) if stop_token_ids is not None else _stop_token_ids(runtime.tokenizer)
    initial_block_size = max(2, int(speculative_depth or runtime.config.draft_block_size))
    min_block_size = int(GEMMA4_ASSISTANT_DESCRIPTOR.draft_semantics.minimum)
    max_block_size = int(GEMMA4_ASSISTANT_DESCRIPTOR.draft_semantics.maximum)
    block_size = max(min_block_size, min(max_block_size, initial_block_size))
    current_block_size = int(block_size)
    proposal_sampler = draft_sampler or sampler
    started_all = time.perf_counter()
    prefill_started_s = time.perf_counter()
    if prefill_callback is not None:
        try:
            prefill_callback(
                {
                    "phase": "started",
                    "tokens_done": 0,
                    "tokens_total": int(len(prompt_ids)),
                    "cached_tokens": 0,
                    "new_prefill_tokens": int(len(prompt_ids)),
                    "elapsed_s": 0.0,
                    "started_s": prefill_started_s,
                }
            )
        except Exception:
            pass
    prompt_state = _restore_or_prefill_gemma4_prompt(
        runtime,
        prompt_ids,
        session_bank=session_bank,
        session_restore_mode=session_restore_mode,
        session_template_hash=session_template_hash,
        session_draft_head_identity=session_draft_head_identity,
        session_policy_fingerprint=session_policy_fingerprint,
        require_shared_kv=True,
    )
    if prefill_callback is not None:
        try:
            elapsed = max(0.0, time.perf_counter() - prefill_started_s)
            new_tokens = int(prompt_state.suffix_tokens)
            tok_s = (new_tokens / elapsed) if elapsed > 0.0 and new_tokens else None
            prefill_callback(
                {
                    "phase": "completed",
                    "tokens_total": int(len(prompt_ids)),
                    "tokens_done": int(len(prompt_ids)),
                    "cached_tokens": int(prompt_state.cached_tokens),
                    "new_prefill_tokens": new_tokens,
                    "elapsed_s": elapsed,
                    "prompt_eval_time_s": float(prompt_state.prompt_eval_time_s),
                    "prefill_tok_s": tok_s,
                    "prefill_compute_tok_s": tok_s,
                    "prefill_wall_tok_s": tok_s,
                    "cache_hit": bool(prompt_state.cache_hit),
                    "cache_miss_reason": prompt_state.cache_miss_reason,
                }
            )
        except Exception:
            pass
    cache = prompt_state.cache
    prompt_boundary_cache = (
        _clone_gemma4_prompt_cache(runtime, cache) if capture_final_state else None
    )
    prompt_boundary_extra_state = _gemma4_session_extra_state(
        shared_kv_states=prompt_state.shared_kv_states,
        kv_offset=int(prompt_state.kv_offset),
    )

    tokens: list[int] = []
    stats_depth = max_block_size if adaptive_draft else block_size
    accepted_by_depth = [0 for _ in range(stats_depth - 1)]
    drafted_by_depth = [0 for _ in range(stats_depth - 1)]
    timing_totals = {
        "draft": 0.0,
        "verify": 0.0,
        "target_hidden": 0.0,
        "target_distribution": 0.0,
        "accept": 0.0,
        "rollback": 0.0,
        "next_hidden_eval": 0.0,
    }
    events: list[dict[str, Any]] = []
    accepted_drafts = 0
    drafted_tokens = 0
    correction_tokens = 0
    bonus_tokens = 0
    verify_calls = 0
    row_distribution_evals = 0
    target_distribution_certified_rows = 0
    target_distribution_fallback_rows = 0
    target_distribution_materialized_rows = 0
    target_distribution_materialized_windows = 0
    target_distribution_top_k = 0
    target_distribution_batch_size = 1
    target_distribution_modes: set[str] = set()
    target_distribution_window_sizes: set[int] = set()
    draft_block_sizes_used: list[int] = []
    interval_100: dict[int, dict[str, Any]] = {}

    primary, _dist = _sample_from_logits(prompt_state.logits[0], sampler, rng)
    hidden = prompt_state.hidden
    shared_kv_states = prompt_state.shared_kv_states
    kv_offset = int(prompt_state.kv_offset)
    pending_primary_needs_commit = False
    safe_to_commit = True
    repetition_config = _repetition_stop_config(bool(repetition_stop))
    repetition_result = None
    wire = _Gemma4RepetitionAwareWire(
        tokens,
        config=repetition_config,
        stop_ids=stop_ids,
        token_callback=token_callback,
    )

    decode_started = time.perf_counter()
    while len(tokens) < int(max_tokens):
        primary = int(primary)
        tokens.append(primary)
        pending_primary_needs_commit = True
        events.append({"step": len(tokens) - 1, "token": primary, "source": "target"})
        wire.emit()
        repetition_result = _trim_repeated_suffix(tokens, repetition_config)
        if repetition_result is not None:
            events.append(
                {
                    "step": len(tokens) - 1,
                    "repetition_stop": {
                        "reason": "exact_repeated_token_suffix",
                        "block_tokens": repetition_result.block_tokens,
                        "repeats": repetition_result.repeats,
                        "trimmed_tokens": repetition_result.repeated_tokens,
                    },
                }
            )
            safe_to_commit = False
            break
        if len(tokens) >= int(max_tokens) or primary in stop_ids:
            break

        remaining = int(max_tokens) - len(tokens) + 1
        round_block_size = max(min_block_size, min(current_block_size, remaining))
        draft_block_sizes_used.append(int(round_block_size))
        round_bucket = (max(0, len(tokens) - 1) // 100) * 100
        round_started = time.perf_counter()
        result = gemma4_exact_speculative_round(
            runtime,
            primary_token_id=primary,
            hidden=hidden,
            shared_kv_states=shared_kv_states,
            kv_offset=kv_offset,
            cache=cache,
            sampler=sampler,
            draft_sampler=proposal_sampler,
            rng=rng,
            draft_block_size=round_block_size,
        )
        verify_calls += 1
        drafted_tokens += round_block_size - 1
        for depth in range(round_block_size - 1):
            if depth < len(drafted_by_depth):
                drafted_by_depth[depth] += 1
        for key, value in (result.metadata.get("timing_s") or {}).items():
            if key in timing_totals:
                timing_totals[key] += float(value)
        target_mode = result.metadata.get("target_distribution_mode")
        if target_mode:
            target_distribution_modes.add(str(target_mode))
        row_distribution_evals += int(result.metadata.get("row_distribution_evals") or 0)
        distribution_metadata = result.metadata.get("target_distribution_metadata")
        if isinstance(distribution_metadata, dict):
            target_distribution_certified_rows += int(
                distribution_metadata.get("certified_rows") or 0
            )
            target_distribution_fallback_rows += int(
                distribution_metadata.get("fallback_rows") or 0
            )
            target_distribution_materialized_rows += int(
                distribution_metadata.get("materialized_rows") or 0
            )
            target_distribution_materialized_windows += int(
                distribution_metadata.get("materialized_windows") or 0
            )
            target_distribution_top_k = max(
                target_distribution_top_k,
                int(distribution_metadata.get("top_k") or 0),
            )
            target_distribution_batch_size = max(
                target_distribution_batch_size,
                int(distribution_metadata.get("batch_size") or 1),
            )
            try:
                target_distribution_window_sizes.add(
                    int(distribution_metadata.get("oracle_window_size"))
                )
            except (TypeError, ValueError):
                pass

        emitted_accepted = 0
        for depth, token in enumerate(result.accepted_token_ids):
            if len(tokens) >= int(max_tokens):
                break
            token = int(token)
            tokens.append(token)
            emitted_accepted += 1
            accepted_drafts += 1
            if depth < len(accepted_by_depth):
                accepted_by_depth[depth] += 1
            events.append({"step": len(tokens) - 1, "token": token, "source": "assistant"})
            wire.emit()
            repetition_result = _trim_repeated_suffix(tokens, repetition_config)
            if repetition_result is not None:
                events.append(
                    {
                        "step": len(tokens) - 1,
                        "repetition_stop": {
                            "reason": "exact_repeated_token_suffix",
                            "block_tokens": repetition_result.block_tokens,
                            "repeats": repetition_result.repeats,
                            "trimmed_tokens": repetition_result.repeated_tokens,
                        },
                    }
                )
                safe_to_commit = False
                break
            if token in stop_ids:
                break
        if repetition_result is not None:
            break
        if emitted_accepted < len(result.accepted_token_ids):
            safe_to_commit = False
        if tokens and int(tokens[-1]) in stop_ids:
            break

        if result.corrected_token_id is not None:
            correction_tokens += 1
        if result.bonus_token_id is not None:
            bonus_tokens += 1
        bucket = interval_100.setdefault(
            int(round_bucket),
            {
                "start_token": int(round_bucket),
                "end_token": int(round_bucket + 99),
                "verify_calls": 0,
                "drafted": 0,
                "accepted": 0,
                "target_distribution_time_s": 0.0,
                "verify_time_s": 0.0,
                "round_time_s": 0.0,
                "cache_events": {},
            },
        )
        bucket["verify_calls"] = int(bucket["verify_calls"]) + 1
        bucket["drafted"] = int(bucket["drafted"]) + max(0, int(round_block_size) - 1)
        bucket["accepted"] = int(bucket["accepted"]) + int(result.accepted_count)
        bucket["target_distribution_time_s"] = float(
            bucket["target_distribution_time_s"]
        ) + float((result.metadata.get("timing_s") or {}).get("target_distribution") or 0.0)
        bucket["verify_time_s"] = float(bucket["verify_time_s"]) + float(
            (result.metadata.get("timing_s") or {}).get("verify") or 0.0
        )
        bucket["round_time_s"] = float(bucket["round_time_s"]) + (
            time.perf_counter() - round_started
        )
        cache_events = bucket["cache_events"]
        if isinstance(cache_events, dict):
            for key, value in (result.metadata.get("cache_counters") or {}).items():
                try:
                    cache_events[str(key)] = int(cache_events.get(str(key), 0)) + int(value)
                except (TypeError, ValueError):
                    continue
        if result.next_primary_token_id is None:
            break
        if adaptive_draft:
            if result.corrected_token_id is None:
                current_block_size = min(max_block_size, int(current_block_size) + 2)
            else:
                current_block_size = max(min_block_size, int(current_block_size) - 1)
        primary = int(result.next_primary_token_id)
        hidden = result.next_hidden
        shared_kv_states = result.next_shared_kv_states
        kv_offset = int(result.next_kv_offset or 0)
        hidden_eval_started = time.perf_counter()
        mx.eval(hidden)
        timing_totals["next_hidden_eval"] += time.perf_counter() - hidden_eval_started
        pending_primary_needs_commit = False

    wire.flush()
    final_state = None
    finish_reason = "stop" if any(token in stop_ids for token in tokens) else "length"
    if capture_final_state and pending_primary_needs_commit and tokens:
        commit_started = time.perf_counter()
        output = runtime.forward_target(
            mx.array([[int(tokens[-1])]], dtype=mx.int32),
            cache=cache,
            phase="postcommit",
        )
        mx.eval(output.logits, output.hidden)
        elapsed_commit = time.perf_counter() - commit_started
        timing_totals["verify"] += elapsed_commit
        verify_calls += 1
        hidden = output.hidden[:, -1:, :]
        shared_kv_states = output.shared_kv_states
        kv_offset = int(output.cache_offset)
        pending_primary_needs_commit = False

    decode_s = time.perf_counter() - decode_started
    elapsed = time.perf_counter() - started_all
    interval_rows = []
    for key in sorted(interval_100):
        row = dict(interval_100[key])
        drafted = int(row.get("drafted") or 0)
        row["acceptance"] = (
            float(int(row.get("accepted") or 0) / drafted) if drafted else 0.0
        )
        row["target_distribution_share"] = (
            float(row.get("target_distribution_time_s") or 0.0)
            / float(row.get("round_time_s") or 1.0)
        )
        interval_rows.append(row)
    mean_accept_probability_by_depth = [
        (accepted / drafted if drafted else None)
        for accepted, drafted in zip(accepted_by_depth, drafted_by_depth)
    ]
    stats = GenerationStats(
        mode="mtpk",
        generated_tokens=len(tokens),
        elapsed_s=elapsed,
        **_generation_rate_fields(
            generated_tokens=len(tokens),
            elapsed_s=elapsed,
            prompt_eval_time_s=prompt_state.prompt_eval_time_s,
        ),
        runtime_mtp_enabled=True,
        prompt_eval_time_s=prompt_state.prompt_eval_time_s,
        cached_tokens=int(prompt_state.cached_tokens),
        new_prefill_tokens=int(prompt_state.suffix_tokens),
        session_cache_hit=bool(prompt_state.cache_hit),
        cache_miss_reason=prompt_state.cache_miss_reason,
        session_restore_mode=prompt_state.restore_mode,
        target_forward_time_s=prompt_state.prompt_eval_time_s + float(timing_totals["verify"]),
        verify_time_s=float(timing_totals["verify"]),
        verify_forward_time_s=float(timing_totals["verify"]),
        verify_target_distribution_time_s=float(timing_totals["target_distribution"]),
        verify_hidden_eval_time_s=float(timing_totals["target_hidden"]),
        draft_time_s=float(timing_totals["draft"]),
        accept_time_s=float(timing_totals["accept"]),
        rollback_time_s=float(timing_totals["rollback"]),
        accepted_drafts=int(accepted_drafts),
        rejected_drafts=max(0, int(drafted_tokens) - int(accepted_drafts)),
        drafted_tokens=int(drafted_tokens),
        verify_calls=int(verify_calls),
        correction_tokens=int(correction_tokens),
        bonus_tokens=int(bonus_tokens),
        speculative_depth=int(block_size - 1),
        requested_speculative_depth=int(
            requested_speculative_depth
            if requested_speculative_depth is not None
            else block_size
        ),
        accepted_by_depth=accepted_by_depth,
        drafted_by_depth=drafted_by_depth,
        mean_accept_probability_by_depth=mean_accept_probability_by_depth,
        peak_memory_bytes=int(mx.get_peak_memory()),
        repetition_stop_triggered=repetition_result is not None,
        repetition_stop_reason=(
            "exact_repeated_token_suffix" if repetition_result is not None else None
        ),
        repetition_stop_block_tokens=(
            0 if repetition_result is None else repetition_result.block_tokens
        ),
        repetition_stop_repeats=(
            0 if repetition_result is None else repetition_result.repeats
        ),
        repetition_stop_trimmed_tokens=(
            0 if repetition_result is None else repetition_result.repeated_tokens
        ),
        repetition_stop_raw_tokens=(
            0 if repetition_result is None else len(tokens) + repetition_result.repeated_tokens
        ),
        events=events,
        draft_core={
            "backend": BACKEND_NAME,
            "draft_block_size": int(block_size),
            "draft_schedule": "heuristic" if adaptive_draft else "constant",
            "draft_block_sizes_used": draft_block_sizes_used,
            "draft_block_size_final": int(current_block_size),
            "assistant_model_path": str(runtime.config.assistant_model_path),
            "decode_s": float(decode_s),
            "target_distribution_mode": (
                sorted(target_distribution_modes)[0]
                if len(target_distribution_modes) == 1
                else runtime.config.target_distribution_mode
            ),
            "target_distribution_window_size": (
                sorted(target_distribution_window_sizes)[0]
                if len(target_distribution_window_sizes) == 1
                else _gemma4_target_distribution_window_size()
            ),
            "row_distribution_evals": int(row_distribution_evals),
            "target_distribution_certified_rows": int(target_distribution_certified_rows),
            "target_distribution_fallback_rows": int(target_distribution_fallback_rows),
            "target_distribution_materialized_rows": int(
                target_distribution_materialized_rows
            ),
            "target_distribution_materialized_windows": int(
                target_distribution_materialized_windows
            ),
            "target_distribution_top_k": int(target_distribution_top_k),
            "target_distribution_batch_size": int(target_distribution_batch_size),
            "target_distribution_share": (
                float(timing_totals["target_distribution"]) / float(decode_s)
                if decode_s > 0
                else 0.0
            ),
            "interval_100": interval_rows,
            "compiled_distribution": dict(runtime.distribution_compile_stats),
            "next_hidden_eval_time_s": float(timing_totals["next_hidden_eval"]),
            "acceptance": (
                float(accepted_drafts / drafted_tokens) if drafted_tokens else 0.0
            ),
            "telemetry": runtime.telemetry.to_dict(),
            "qa_source": "gemma4_pair_runtime",
        },
        owned_attn_kv=runtime.telemetry.to_dict(),
    )
    if capture_final_state:
        final_logits = runtime.target.logits_from_hidden(hidden)[:, -1, :]
        mx.eval(final_logits)
        final_state = GenerationFinalState(
            final_trunk_cache=cache,
            final_logits=final_logits,
            final_hidden=hidden,
            final_committed_mtp_cache=None,
            generated_token_ids=tuple(int(token) for token in tokens),
            safe_to_commit=bool(safe_to_commit and not pending_primary_needs_commit),
            finish_reason=finish_reason,
            extra_state=_gemma4_session_extra_state(
                shared_kv_states=shared_kv_states,
                kv_offset=int(kv_offset),
            ),
            prompt_boundary_cache=prompt_boundary_cache,
            prompt_boundary_logits=prompt_state.logits,
            prompt_boundary_hidden=prompt_state.hidden,
            prompt_boundary_extra_state=prompt_boundary_extra_state,
        )
    return GenerationOutput(
        tokens=tokens,
        text=_decode_tokens(runtime.tokenizer, _strip_terminal_stop(tokens, stop_ids)),
        stats=stats,
        final_state=final_state,
    )
