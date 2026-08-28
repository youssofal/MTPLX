"""Measured Qwen 3.8 27B optimization route and artifact contract."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .draft_lm_head import configure_qwen38_row10_compact_head
from .gdn_capture import (
    configure_qwen38_row18_gdn_decay_memo,
    configure_qwen38_row48_capture,
)
from .mtp_patch import install_qwen38_kv_only_history_append
from .qwen38_challenge_kernels import (
    configure_qwen38_row21_qk_rms_rope,
    configure_qwen38_row24_qk_length_limit,
)
from .qwen38_mtp_block_artifacts import configure_qwen38_mtp_block

QWEN38_Q8_LINEAR_ATTN_LAYERS = (
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22,
    24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45,
    46, 48, 49, 50, 52, 53, 54, 56, 57, 58, 60, 61, 62,
)
QWEN38_PACKING = "mlx_affine_u32_le"
DEFAULT_QWEN38_CACHE_ROUTE = "control"
QWEN38_KV_ONLY_MIN_CONTEXT = 16_384
QWEN38_EXPECTED_GDN_MODULES = len(QWEN38_Q8_LINEAR_ATTN_LAYERS)
QWEN38_EXPECTED_FULL_ATTENTION_MODULES = 64 - QWEN38_EXPECTED_GDN_MODULES
QWEN38_FINAL_ROUTE: Mapping[str, Any] = MappingProxyType(
    {
        "cache_route": DEFAULT_QWEN38_CACHE_ROUTE,
        "dual_norm": False,
    }
)
QWEN38_LOW_FIXED_ROUTE = (
    "r20_kv_only_history+r53_command_buffers+r08_device_draft+"
    "r10_compact_vocab+r21_qk_rms_rope+r24_eval_ladder+"
    "r26_prefill_ladder_3"
)
QWEN38_LOW_ADAPTIVE_ROUTE = (
    QWEN38_LOW_FIXED_ROUTE + "+r11_position_ema+r17_q4_mtp_block"
)
QWEN38_XHIGH_FIXED_ROUTE = (
    "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3+"
    "r50_wired_residency+r53_command_buffers"
)
QWEN38_XHIGH_ADAPTIVE_ROUTE = QWEN38_XHIGH_FIXED_ROUTE + "+r11_position_ema"
QWEN38_LOW_BF16_INSTALLED_ROUTE = (
    "kv_only_history+r21_qk_rms_rope+r24_eval_ladder+"
    "r26_prefill_ladder_3+r10_compact_vocab"
)
QWEN38_LOW_Q4_INSTALLED_ROUTE = (
    "r17_q4_mtp_block+" + QWEN38_LOW_BF16_INSTALLED_ROUTE
)
QWEN38_XHIGH_BF16_INSTALLED_ROUTE = (
    "kv_only_history+r24_eval_ladder+r26_prefill_ladder_3+"
    "r50_wired_residency"
)
QWEN38_LOW_BF16_KERNEL_IDS = (
    "qwen38_mtp_kv_only_history_ge16384_v1",
    "qwen38_qk_rms_rope_bf16_h256_r64_v1",
    "qwen38_row24_qk_rms_rope_l_le16_v1",
    "qwen38_row24_target_eval_ladder_v1",
    "qwen38_row26_prefill_eval_every3_v1",
    "qwen38_row26_qk_rms_rope_l_le32_v1",
    "qwen38_row10_compact_q4_g64_vocab_v1",
)
QWEN38_XHIGH_BF16_KERNEL_IDS = (
    "qwen38_mtp_kv_only_history_ge16384_v1",
    "qwen38_row24_target_eval_ladder_v1",
    "qwen38_row26_prefill_eval_every3_v1",
    "qwen38_row50_post_warm_wired_residency_v1",
)
QWEN38_LOW_BF16_FEATURE_KEYS = (
    "r10_compact_vocab",
    "r20_kv_only_history",
    "r21_qk_rms_rope",
    "r24_eval_ladder",
    "r24_qk_length_limit",
    "r26_prefill_ladder_3",
    "r26_qk_length_limit",
    "r53_command_buffers",
)
QWEN38_XHIGH_BF16_FEATURE_KEYS = (
    "r20_kv_only_history",
    "r24_eval_ladder",
    "r26_prefill_ladder_3",
    "r50_wired_residency",
    "r53_command_buffers",
)


def qwen38_final_route() -> dict[str, Any]:
    """Return the explicit unchanged-control route for ordinary runtime loads."""

    return dict(QWEN38_FINAL_ROUTE)


class Qwen38ContractError(RuntimeError):
    """The requested Qwen 3.8 route does not match its measured contract."""


@dataclass(frozen=True)
class Qwen38ModelContract:
    contract_id: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    dtype: str
    trunk_bits: int
    trunk_group_size: int
    trunk_mode: str
    packing: str


@dataclass(frozen=True)
class Qwen38RouteBindings:
    mtp_cache_append: Callable[..., Any]
    history_route_id: str = "stock_history"


@dataclass(frozen=True)
class Qwen38RouteSpec:
    route_id: str
    contract: Qwen38ModelContract
    bindings: Qwen38RouteBindings
    history_route_id: str = "stock_history"
    kernel_ids: tuple[str, ...] = ()
    min_context_tokens: int = 0
    policy_id: str = "current_mtplx"
    selfcheck_status: str = "unchecked"
    selfcheck_passed: bool = False
    performance_profile: str | None = None
    requested_route_id: str | None = None
    draft_core: str = "stock"
    mtp_block_identity: str = "bf16"

    @property
    def fingerprint(self) -> str:
        payload = {
            "contract_id": self.contract.contract_id,
            "kernel_ids": list(self.kernel_ids),
            "history_route_id": self.history_route_id,
            "min_context_tokens": self.min_context_tokens,
            "policy_id": self.policy_id,
            "performance_profile": self.performance_profile,
            "requested_route_id": self.requested_route_id,
            "draft_core": self.draft_core,
            "mtp_block_identity": self.mtp_block_identity,
            "route_id": self.route_id,
            "selfcheck_passed": self.selfcheck_passed,
            "selfcheck_status": self.selfcheck_status,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Qwen38PerformanceProfileConfig:
    """Construction-only inputs for one prevalidated request route."""

    requested_route_id: str
    install_options: Mapping[str, Any]
    draft_core: str = "stock"
    row53_command_buffers: bool = False
    installed_route_id: str | None = None
    kernel_ids: tuple[str, ...] = ()
    feature_keys: tuple[str, ...] = ()


_MISSING_PROFILE_ATTRIBUTE = object()


@dataclass(frozen=True)
class Qwen38ExecutionBindings:
    """Already-validated object bindings installed at a request boundary."""

    forward_layers: Any
    prefill_forward_layers: Any
    prepare_mtp_inputs: Any
    mtp_block: Any
    model_mtp_block: Any
    draft_lm_head: Any
    draft_token_id_map: Any
    draft_target_vocab_size: Any
    row21_bindings: tuple[tuple[Any, type, Any], ...]
    wired_limit_setter: Callable[[int], Any] | None = None
    wired_limit_bytes: int | None = None


@dataclass(frozen=True)
class Qwen38PerformanceProfile:
    profile_id: str
    requested_route_id: str
    route: Qwen38RouteSpec
    bindings: Qwen38ExecutionBindings
    feature_receipt: Mapping[str, Any]
    draft_core: str
    mtp_block_identity: str


def qwen38_measured_performance_profile_configs(
    *,
    adaptive_policy: str,
    q4_mtp_block: Path | None,
) -> dict[str, Qwen38PerformanceProfileConfig]:
    """Return the independently measured low/xhigh construction profiles."""

    policy = str(adaptive_policy or "none").strip().lower()
    if policy not in {"none", "position_ema"}:
        raise Qwen38ContractError(
            "measured Qwen 3.8 profiles support only adaptive-policy none or "
            "position_ema"
        )
    low_options: dict[str, Any] = {
        "cache_route": "kv_only_history",
        "row10_compact_vocab": True,
        "row21_qk_rms_rope": True,
        "row24_eval_ladder": True,
        "row26_prefill_ladder_3": True,
    }
    low_route = QWEN38_LOW_FIXED_ROUTE
    xhigh_route = QWEN38_XHIGH_FIXED_ROUTE
    if policy == "position_ema":
        if q4_mtp_block is None:
            raise Qwen38ContractError(
                "position_ema low profile requires a Q4 MTP block artifact"
            )
        try:
            artifact = Path(q4_mtp_block).expanduser().resolve(strict=True)
        except OSError as exc:
            raise Qwen38ContractError(
                "position_ema low profile Q4 MTP block artifact is unavailable: "
                f"{q4_mtp_block}"
            ) from exc
        low_options.update(
            {
                "mtp_block_variant": "r17",
                "mtp_block_artifact_path": artifact,
            }
        )
        low_route = QWEN38_LOW_ADAPTIVE_ROUTE
        xhigh_route = QWEN38_XHIGH_ADAPTIVE_ROUTE
    return {
        "stock": Qwen38PerformanceProfileConfig(
            requested_route_id="control",
            install_options={"cache_route": "control"},
            draft_core="stock",
            installed_route_id="control",
        ),
        "low": Qwen38PerformanceProfileConfig(
            requested_route_id=low_route,
            install_options=low_options,
            draft_core="device",
            row53_command_buffers=True,
            installed_route_id=(
                QWEN38_LOW_Q4_INSTALLED_ROUTE
                if policy == "position_ema"
                else QWEN38_LOW_BF16_INSTALLED_ROUTE
            ),
            kernel_ids=(
                ("qwen38_row17_q4_g64_mtp_block_v1",)
                + QWEN38_LOW_BF16_KERNEL_IDS
                if policy == "position_ema"
                else QWEN38_LOW_BF16_KERNEL_IDS
            ),
            feature_keys=(
                ("r17_q4_mtp_block",) + QWEN38_LOW_BF16_FEATURE_KEYS
                if policy == "position_ema"
                else QWEN38_LOW_BF16_FEATURE_KEYS
            ),
        ),
        "xhigh": Qwen38PerformanceProfileConfig(
            requested_route_id=xhigh_route,
            install_options={
                "cache_route": "kv_only_history",
                "row24_eval_ladder": True,
                "row26_prefill_ladder_3": True,
                "row50_wired_residency": True,
            },
            draft_core="stock",
            row53_command_buffers=True,
            installed_route_id=QWEN38_XHIGH_BF16_INSTALLED_ROUTE,
            kernel_ids=QWEN38_XHIGH_BF16_KERNEL_IDS,
            feature_keys=QWEN38_XHIGH_BF16_FEATURE_KEYS,
        ),
    }


def _qwen38_identity(config: Mapping[str, Any], model_path: Path) -> bool:
    runtime = config.get("mtplx_runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    sources = (
        str(model_path),
        str(runtime.get("base_trunk") or ""),
        str(runtime.get("source_repo") or ""),
        str(runtime.get("public_model_id") or ""),
    )
    return any(
        re.search(r"qwen(?:3)?[.\-_]?8[^/]*27b", source, re.IGNORECASE)
        for source in sources
    )


def is_qwen38_27b_candidate(config: Mapping[str, Any], model_path: Path) -> bool:
    """Return whether this is the one measured Optimized-Speed artifact."""

    runtime = config.get("mtplx_runtime")
    runtime = runtime if isinstance(runtime, Mapping) else {}
    sources = (
        str(model_path),
        str(model_path.resolve()),
        str(runtime.get("public_model_id") or ""),
    )
    return any(
        re.search(
            r"(?:qwen3[.]8-27b-mtplx-optimized-speed|"
            r"mtplx-qwen38-27b-optimized-speed)$",
            source,
            re.IGNORECASE,
        )
        for source in sources
    )


def _expected_q8_overrides() -> set[str]:
    names = {
        "language_model.model.embed_tokens",
        "language_model.lm_head",
    }
    names.update(
        f"language_model.model.layers.{layer}.linear_attn.out_proj"
        for layer in QWEN38_Q8_LINEAR_ATTN_LAYERS
    )
    names.update(
        f"language_model.model.layers.{layer}.mlp.{projection}"
        for layer in range(56, 64)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    return names


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise Qwen38ContractError(
            f"Qwen 3.8 contract {label} mismatch: {actual!r} != {expected!r}"
        )


def validate_qwen38_27b_contract(
    config: Mapping[str, Any],
    model_path: Path,
    *,
    packing: str = QWEN38_PACKING,
) -> Qwen38ModelContract:
    """Validate the exact artifact used for the retained measurements."""

    if not _qwen38_identity(config, model_path):
        raise Qwen38ContractError("expected exact Qwen 3.8 27B identity")
    _require_equal(
        "architectures",
        config.get("architectures"),
        ["Qwen3_5ForConditionalGeneration"],
    )
    _require_equal("model_type", config.get("model_type"), "qwen3_5")
    text = config.get("text_config")
    if not isinstance(text, Mapping):
        raise Qwen38ContractError("Qwen 3.8 contract text_config is missing")
    expected_text = {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1,
    }
    for field, expected in expected_text.items():
        _require_equal(field, text.get(field), expected)
    extra = config.get("mlx_lm_extra_tensors")
    extra = extra if isinstance(extra, Mapping) else {}
    _require_equal("MTP sidecar", extra.get("mtp_file"), "mtp.safetensors")

    quantization = config.get("quantization")
    if not isinstance(quantization, Mapping):
        raise Qwen38ContractError("Qwen 3.8 trunk quantization is missing")
    trunk = (
        quantization.get("bits"),
        quantization.get("group_size"),
        quantization.get("mode"),
    )
    if trunk != (4, 32, "affine"):
        raise Qwen38ContractError(
            f"Qwen 3.8 trunk quantization mismatch: {trunk!r} != (4, 32, 'affine')"
        )
    overrides = {
        key for key, value in quantization.items() if isinstance(value, Mapping)
    }
    expected_overrides = _expected_q8_overrides()
    if overrides != expected_overrides:
        raise Qwen38ContractError(
            "Qwen 3.8 quantization override map mismatch: "
            f"missing={sorted(expected_overrides - overrides)}, "
            f"extra={sorted(overrides - expected_overrides)}"
        )
    for name in sorted(expected_overrides):
        spec = quantization[name]
        observed = (spec.get("bits"), spec.get("group_size"), spec.get("mode"))
        if observed != (8, 64, "affine"):
            raise Qwen38ContractError(
                f"Qwen 3.8 override {name} mismatch: {observed!r}"
            )
    if packing != QWEN38_PACKING:
        raise Qwen38ContractError(
            f"Qwen 3.8 packing mismatch: {packing!r} != {QWEN38_PACKING!r}"
        )

    contract_payload = {
        **expected_text,
        "packing": packing,
        "trunk_bits": 4,
        "trunk_group_size": 32,
        "trunk_mode": "affine",
    }
    contract_id = hashlib.sha256(
        json.dumps(contract_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return Qwen38ModelContract(
        contract_id=contract_id,
        hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        num_hidden_layers=int(text["num_hidden_layers"]),
        num_attention_heads=int(text["num_attention_heads"]),
        num_key_value_heads=int(text["num_key_value_heads"]),
        head_dim=int(text["head_dim"]),
        vocab_size=int(text["vocab_size"]),
        dtype=str(text["dtype"]),
        trunk_bits=4,
        trunk_group_size=32,
        trunk_mode="affine",
        packing=packing,
    )


def build_qwen38_route(
    config: Mapping[str, Any],
    model_path: Path,
    *,
    bindings: Qwen38RouteBindings,
    route_id: str,
    kernel_ids: tuple[str, ...] = (),
    min_context_tokens: int = 0,
    policy_id: str = "current_mtplx",
    selfcheck_status: str | None = None,
    selfcheck_passed: bool | None = None,
) -> Qwen38RouteSpec:
    contract = validate_qwen38_27b_contract(config, model_path)
    if not callable(bindings.mtp_cache_append):
        raise Qwen38ContractError(
            f"route {route_id!r} is missing callable mtp_cache_append"
        )
    if bindings.history_route_id not in {"stock_history", "kv_only_history"}:
        raise Qwen38ContractError(
            f"route {route_id!r} has unknown history route "
            f"{bindings.history_route_id!r}"
        )
    if selfcheck_passed is None:
        selfcheck_passed = route_id == "control"
    if selfcheck_status is None:
        selfcheck_status = "control" if route_id == "control" else "unchecked"
    return Qwen38RouteSpec(
        route_id=route_id,
        contract=contract,
        bindings=bindings,
        history_route_id=bindings.history_route_id,
        kernel_ids=tuple(kernel_ids),
        min_context_tokens=max(0, int(min_context_tokens)),
        policy_id=policy_id,
        selfcheck_status=selfcheck_status,
        selfcheck_passed=bool(selfcheck_passed),
    )


def control_bindings(runtime: Any) -> Qwen38RouteBindings:
    return Qwen38RouteBindings(
        mtp_cache_append=getattr(
            runtime.model,
            "mtp_update_cache",
            runtime.update_mtp_cache,
        )
    )


def _validate_qwen38_dual_norm_install(text: Any, *, q8_embedding: bool) -> None:
    """Validate the fixed dual-norm kernel operands before enabling the lane."""

    mtp = getattr(text, "mtp", None)
    embedding_norm = getattr(mtp, "pre_fc_norm_embedding", None)
    hidden_norm = getattr(mtp, "pre_fc_norm_hidden", None)
    operands = (embedding_norm, hidden_norm)
    if any(value is None for value in operands):
        raise Qwen38ContractError("Qwen 3.8 dual RMSNorm modules are unavailable")
    if float(embedding_norm.eps) != float(hidden_norm.eps):
        raise Qwen38ContractError("Qwen 3.8 dual RMSNorm eps values differ")
    for name, norm in (("embedding", embedding_norm), ("hidden", hidden_norm)):
        weight = getattr(norm, "weight", None)
        if (
            tuple(getattr(weight, "shape", ())) != (5120,)
            or str(getattr(weight, "dtype", "")) != "mlx.core.bfloat16"
        ):
            raise Qwen38ContractError(
                f"Qwen 3.8 {name} RMSNorm requires BF16 hidden-5120 weight"
            )
    if not q8_embedding:
        return
    embedding = getattr(getattr(text, "model", None), "embed_tokens", None)
    if (
        int(getattr(embedding, "bits", 0)) != 8
        or int(getattr(embedding, "group_size", 0)) != 64
        or str(getattr(embedding, "mode", "")).lower() != "affine"
        or tuple(getattr(getattr(embedding, "weight", None), "shape", ())[-1:])
        != (1280,)
        or tuple(getattr(getattr(embedding, "scales", None), "shape", ())[-1:])
        != (80,)
        or tuple(getattr(getattr(embedding, "biases", None), "shape", ())[-1:])
        != (80,)
        or str(getattr(getattr(embedding, "weight", None), "dtype", ""))
        != "mlx.core.uint32"
        or str(getattr(getattr(embedding, "scales", None), "dtype", ""))
        != "mlx.core.bfloat16"
        or str(getattr(getattr(embedding, "biases", None), "dtype", ""))
        != "mlx.core.bfloat16"
    ):
        raise Qwen38ContractError(
            "Qwen 3.8 row 63 requires affine Q8/group-64 embedding geometry"
        )


def install_qwen38_control_route(
    runtime: Any,
    config: Mapping[str, Any],
    model_path: Path,
) -> Qwen38RouteSpec | None:
    return install_qwen38_route(runtime, config, model_path, cache_route="control")


def configure_qwen38_row50_wired_residency(
    runtime: Any,
    *,
    active: bool,
    mx_module: Any | None = None,
) -> dict[str, Any]:
    """Apply row 50's post-warm resident-weight budget and restore controls."""

    if mx_module is None:
        import mlx.core as mx
    else:
        mx = mx_module

    state = getattr(runtime, "_qwen38_row50_wired_state", None)
    if not active:
        if isinstance(state, dict) and state.get("installed"):
            baseline = int(state["baseline_limit_bytes"])
            mx.set_wired_limit(baseline)
            return {
                **state,
                "active": False,
                "restored_limit_bytes": baseline,
            }
        return {"installed": False, "active": False}

    if isinstance(state, dict) and state.get("installed"):
        runtime._qwen38_row50_set_wired_limit = mx.set_wired_limit
        mx.set_wired_limit(int(state["target_limit_bytes"]))
        return {**state, "active": True}

    info = dict(mx.device_info())
    physical = int(info.get("memory_size") or 0)
    if physical and physical < 96 * 2**30:
        return {
            "installed": False,
            "active": False,
            "reason": "physical_memory_below_96gib",
            "physical_memory_bytes": physical,
        }

    # Row 50 sizes residency only after temporary warm graphs leave scope.
    mx.clear_cache()
    active_bytes = int(mx.get_active_memory())
    if active_bytes <= 0:
        return {"installed": False, "active": False, "reason": "no_active_memory"}
    target = active_bytes + 64 * 2**20
    recommended = int(info.get("max_recommended_working_set_size") or 0)
    if recommended > 0:
        target = min(target, max(0, recommended - 256 * 2**20))
    if target <= 0:
        return {"installed": False, "active": False, "reason": "invalid_target"}
    baseline = int(mx.set_wired_limit(target))
    state = {
        "installed": True,
        "active": True,
        "active_memory_bytes": active_bytes,
        "target_limit_bytes": target,
        "baseline_limit_bytes": baseline,
        "max_recommended_working_set_bytes": recommended,
        "slack_bytes": 64 * 2**20,
    }
    runtime._qwen38_row50_wired_state = state
    runtime._qwen38_row50_set_wired_limit = mx.set_wired_limit
    return dict(state)


def install_qwen38_route(
    runtime: Any,
    config: Mapping[str, Any],
    model_path: Path,
    *,
    cache_route: str = DEFAULT_QWEN38_CACHE_ROUTE,
    dual_norm: bool = False,
    row10_compact_vocab: bool = False,
    mtp_block_variant: str | None = None,
    mtp_block_artifact_path: Path | None = None,
    row18_gdn_decay_memo: bool = False,
    row21_qk_rms_rope: bool = False,
    row24_eval_ladder: bool = False,
    row26_prefill_ladder_3: bool = False,
    row48_boundary_fused: bool = False,
    row50_wired_residency: bool = False,
    row63_q8_embedding_dual_norm: bool = False,
) -> Qwen38RouteSpec | None:
    if not is_qwen38_27b_candidate(config, model_path):
        return None
    if not bool(getattr(runtime, "mtp_enabled", False)):
        return None
    cache_route_id = str(cache_route or "control").strip().lower()
    bindings = control_bindings(runtime)
    kernel_ids: list[str] = []
    route_features: list[str] = []
    feature_receipt: dict[str, dict[str, int]] = {}

    text = getattr(runtime.model, "language_model", runtime.model)
    if mtp_block_variant is not None or hasattr(
        text, "_mtplx_qwen38_control_mtp_block"
    ):
        mtp_block_report = configure_qwen38_mtp_block(
            runtime,
            variant=mtp_block_variant,
            artifact_path=mtp_block_artifact_path,
        )
    else:
        mtp_block_report = {"installed": False, "active": False, "variant": None}
    if mtp_block_variant is not None:
        if not bool(mtp_block_report.get("installed")):
            raise Qwen38ContractError(
                f"Qwen 3.8 {mtp_block_variant} MTP block was not installed"
            )
        if mtp_block_variant == "r17":
            route_features.append("r17_q4_mtp_block")
            kernel_ids.append("qwen38_row17_q4_g64_mtp_block_v1")
            feature_receipt["r17_q4_mtp_block"] = mtp_block_report
        elif mtp_block_variant == "r28":
            route_features.extend(("r17_q4_mtp_block", "r28_q4_mtp_block"))
            kernel_ids.append("qwen38_row28_q4_g64_mtp_block_v1")
            feature_receipt["r28_q4_mtp_block"] = mtp_block_report
        elif mtp_block_variant == "r36":
            route_features.extend(("r17_q4_mtp_block", "r36_qkv_islands"))
            kernel_ids.append("qwen38_row36_q4_g64_bf16_qkv_islands_v1")
            feature_receipt["r36_qkv_islands"] = mtp_block_report
        else:
            raise Qwen38ContractError(
                f"unknown Qwen 3.8 MTP block variant: {mtp_block_variant!r}"
            )

    min_context_tokens = 0
    if cache_route_id == "kv_only_history":
        implementation = getattr(
            runtime.model,
            "mtp_update_cache_kv_only_history",
            None,
        )
        if hasattr(text, "mtp"):
            implementation = install_qwen38_kv_only_history_append(runtime.model)
        if not callable(implementation):
            raise Qwen38ContractError(
                "Qwen 3.8 K/V-only history route is unavailable on the loaded model"
            )
        bindings = replace(
            bindings,
            mtp_cache_append=implementation,
            history_route_id="kv_only_history",
        )
        route_features.append("kv_only_history")
        kernel_ids.append("qwen38_mtp_kv_only_history_ge16384_v1")
        min_context_tokens = QWEN38_KV_ONLY_MIN_CONTEXT
        feature_receipt["r20_kv_only_history"] = {
            "installed": True,
            "min_context_tokens": QWEN38_KV_ONLY_MIN_CONTEXT,
        }
    elif cache_route_id != "control":
        raise Qwen38ContractError(
            f"unknown Qwen 3.8 cache route: {cache_route!r}"
        )

    row18_gdn_report = configure_qwen38_row18_gdn_decay_memo(
        runtime.model,
        active=bool(row18_gdn_decay_memo),
    )
    if row18_gdn_decay_memo:
        configured_modules = int(row18_gdn_report.get("configured_modules", 0))
        active_modules = int(row18_gdn_report.get("active_modules", 0))
        if (
            configured_modules != QWEN38_EXPECTED_GDN_MODULES
            or active_modules != QWEN38_EXPECTED_GDN_MODULES
        ):
            raise Qwen38ContractError(
                "Qwen 3.8 row 18 GDN coverage mismatch: "
                f"configured={configured_modules}, active={active_modules}, "
                f"expected={QWEN38_EXPECTED_GDN_MODULES}"
            )
        route_features.append("r18_gdn_decay_memo")
        kernel_ids.append("qwen38_row18_gdn_neg_exp_a_log_memo_v1")
        feature_receipt["r18_gdn_decay_memo"] = row18_gdn_report

    row21_report = configure_qwen38_row21_qk_rms_rope(
        runtime.model,
        active=bool(row21_qk_rms_rope),
    )
    if row21_qk_rms_rope:
        eligible_modules = int(row21_report.get("eligible_modules", 0))
        active_modules = int(row21_report.get("active_modules", 0))
        if (
            eligible_modules != QWEN38_EXPECTED_FULL_ATTENTION_MODULES
            or active_modules != QWEN38_EXPECTED_FULL_ATTENTION_MODULES
        ):
            raise Qwen38ContractError(
                "Qwen 3.8 row 21 full-attention coverage mismatch: "
                f"eligible={eligible_modules}, active={active_modules}, "
                f"expected={QWEN38_EXPECTED_FULL_ATTENTION_MODULES}"
            )
        route_features.append("r21_qk_rms_rope")
        kernel_ids.append("qwen38_qk_rms_rope_bf16_h256_r64_v1")
        feature_receipt["r21_qk_rms_rope"] = row21_report

    row24_qk_report = configure_qwen38_row24_qk_length_limit(
        runtime.model,
        active=bool(row24_eval_ladder and row21_qk_rms_rope),
        max_length=32 if row26_prefill_ladder_3 else 16,
    )
    if row24_eval_ladder and row21_qk_rms_rope:
        if int(row24_qk_report.get("active_modules", 0)) <= 0:
            raise Qwen38ContractError(
                "Qwen 3.8 row 24 Q/K length limit configured no modules"
            )
        kernel_ids.append("qwen38_row24_qk_rms_rope_l_le16_v1")
        feature_receipt["r24_qk_length_limit"] = row24_qk_report

    target_layer_route = (
        "_mtplx_forward_layers_row26"
        if row26_prefill_ladder_3
        else "_mtplx_forward_layers_row24"
        if row24_eval_ladder
        else "_mtplx_forward_layers_stock"
    )
    target_layer_impl = getattr(text, target_layer_route, None)
    if (row24_eval_ladder or row26_prefill_ladder_3) and not callable(
        target_layer_impl
    ):
        raise Qwen38ContractError(
            f"Qwen 3.8 target layer route {target_layer_route!r} is unavailable"
        )
    prefill_only_target_ladder = bool(
        row26_prefill_ladder_3 and not row21_qk_rms_rope
    )
    if prefill_only_target_ladder:
        stock_target_layer_impl = getattr(
            text, "_mtplx_forward_layers_stock", None
        )
        if not callable(stock_target_layer_impl):
            raise Qwen38ContractError(
                "Qwen 3.8 prefill-only target ladder requires the stock decode route"
            )
        text._mtplx_qwen38_prefill_forward_layers = target_layer_impl
        text._mtplx_forward_layers = stock_target_layer_impl
    elif callable(target_layer_impl):
        text._mtplx_qwen38_prefill_forward_layers = None
        text._mtplx_forward_layers = target_layer_impl
    if row24_eval_ladder:
        route_features.append("r24_eval_ladder")
        kernel_ids.append("qwen38_row24_target_eval_ladder_v1")
        feature_receipt["r24_eval_ladder"] = {"active": 1}
    if row26_prefill_ladder_3:
        if not row24_eval_ladder:
            raise Qwen38ContractError(
                "Qwen 3.8 row 26 prefill cadence requires retained row 24"
            )
        route_features.append("r26_prefill_ladder_3")
        kernel_ids.append("qwen38_row26_prefill_eval_every3_v1")
        if row21_qk_rms_rope:
            kernel_ids.append("qwen38_row26_qk_rms_rope_l_le32_v1")
            feature_receipt["r26_qk_length_limit"] = row24_qk_report
        feature_receipt["r26_prefill_ladder_3"] = (
            {
                "active": 1,
                "phase_scope": "prefill",
                "decode_route": "stock",
            }
            if prefill_only_target_ladder
            else {"active": 1}
        )
    row48_report = configure_qwen38_row48_capture(
        runtime,
        active=bool(row48_boundary_fused),
    )
    if row48_boundary_fused:
        route_features.append("r48_boundary_fused")
        kernel_ids.append("qwen38_row48_boundary_fused_residual_rmsnorm_v1")
        feature_receipt["r48_boundary_fused"] = row48_report
    row50_report = configure_qwen38_row50_wired_residency(
        runtime,
        active=bool(row50_wired_residency),
    )
    if row50_wired_residency:
        if not bool(row50_report.get("installed")):
            raise Qwen38ContractError(
                "Qwen 3.8 row 50 wired residency could not be installed"
            )
        route_features.append("r50_wired_residency")
        kernel_ids.append("qwen38_row50_post_warm_wired_residency_v1")
        feature_receipt["r50_wired_residency"] = row50_report
    if dual_norm:
        _validate_qwen38_dual_norm_install(text, q8_embedding=False)
    if dual_norm:
        route_features.append("dual_norm")
        kernel_ids.append("qwen38_dual_rms_norm_concat_bf16_v1")
        feature_receipt["dual_norm"] = {"active": 1}
    if row63_q8_embedding_dual_norm:
        _validate_qwen38_dual_norm_install(text, q8_embedding=True)
    mtp_input_route = (
        "_mtplx_prepare_mtp_inputs_row63_dual"
        if row63_q8_embedding_dual_norm and dual_norm
        else "_mtplx_prepare_mtp_inputs_row63"
        if row63_q8_embedding_dual_norm
        else "_mtplx_prepare_mtp_inputs_dual"
        if dual_norm
        else "_mtplx_prepare_mtp_inputs_stock"
    )
    mtp_input_impl = getattr(text, mtp_input_route, None)
    if (row63_q8_embedding_dual_norm or dual_norm) and not callable(
        mtp_input_impl
    ):
        raise Qwen38ContractError(
            f"Qwen 3.8 MTP input route {mtp_input_route!r} is unavailable"
        )
    if callable(mtp_input_impl):
        text._mtplx_prepare_mtp_inputs = mtp_input_impl
    if row63_q8_embedding_dual_norm:
        route_features.append("r63_q8_embedding_dual_norm")
        kernel_ids.append("qwen38_row63_q8_g64_embedding_dual_rmsnorm_concat_v1")
        feature_receipt["r63_q8_embedding_dual_norm"] = {"active": 1}
    row10_report = configure_qwen38_row10_compact_head(
        runtime,
        active=bool(row10_compact_vocab),
    )
    if row10_compact_vocab:
        if not bool(row10_report.get("installed")):
            raise Qwen38ContractError("Qwen 3.8 row 10 compact head was not installed")
        route_features.append("r10_compact_vocab")
        kernel_ids.append("qwen38_row10_compact_q4_g64_vocab_v1")
        feature_receipt["r10_compact_vocab"] = row10_report

    route_id = "+".join(route_features) if route_features else "control"
    route = build_qwen38_route(
        config,
        model_path,
        bindings=bindings,
        route_id=route_id,
        kernel_ids=tuple(kernel_ids),
        min_context_tokens=min_context_tokens,
    )
    runtime.qwen38_route = route
    runtime.qwen38_feature_receipt = feature_receipt
    return route


def _profile_attribute(value: Any, name: str) -> Any:
    return getattr(value, name, _MISSING_PROFILE_ATTRIBUTE)


def _capture_qwen38_execution_bindings(runtime: Any) -> Qwen38ExecutionBindings:
    text = getattr(runtime.model, "language_model", runtime.model)
    row21 = []
    for binding in tuple(getattr(text, "_mtplx_row21_bindings", ()) or ()):
        attention = binding[0]
        row21.append(
            (
                attention,
                attention.__class__,
                _profile_attribute(attention, "_mtplx_prepare_explicit_qk"),
            )
        )
    return Qwen38ExecutionBindings(
        forward_layers=_profile_attribute(text, "_mtplx_forward_layers"),
        prefill_forward_layers=_profile_attribute(
            text, "_mtplx_qwen38_prefill_forward_layers"
        ),
        prepare_mtp_inputs=_profile_attribute(text, "_mtplx_prepare_mtp_inputs"),
        mtp_block=_profile_attribute(text, "mtp"),
        model_mtp_block=_profile_attribute(runtime.model, "mtp"),
        draft_lm_head=_profile_attribute(text, "_mtplx_draft_lm_head"),
        draft_token_id_map=_profile_attribute(text, "_mtplx_draft_token_id_map"),
        draft_target_vocab_size=_profile_attribute(
            text, "_mtplx_draft_target_vocab_size"
        ),
        row21_bindings=tuple(row21),
    )


def _apply_profile_attribute(value: Any, name: str, binding: Any) -> None:
    if binding is _MISSING_PROFILE_ATTRIBUTE:
        if hasattr(value, name):
            delattr(value, name)
        return
    setattr(value, name, binding)


def _apply_qwen38_execution_bindings(
    runtime: Any,
    bindings: Qwen38ExecutionBindings,
) -> None:
    text = getattr(runtime.model, "language_model", runtime.model)
    _apply_profile_attribute(text, "_mtplx_forward_layers", bindings.forward_layers)
    _apply_profile_attribute(
        text,
        "_mtplx_qwen38_prefill_forward_layers",
        bindings.prefill_forward_layers,
    )
    _apply_profile_attribute(
        text, "_mtplx_prepare_mtp_inputs", bindings.prepare_mtp_inputs
    )
    _apply_profile_attribute(text, "mtp", bindings.mtp_block)
    _apply_profile_attribute(runtime.model, "mtp", bindings.model_mtp_block)
    _apply_profile_attribute(text, "_mtplx_draft_lm_head", bindings.draft_lm_head)
    _apply_profile_attribute(
        text, "_mtplx_draft_token_id_map", bindings.draft_token_id_map
    )
    _apply_profile_attribute(
        text,
        "_mtplx_draft_target_vocab_size",
        bindings.draft_target_vocab_size,
    )
    for attention, selected_class, explicit_qk in bindings.row21_bindings:
        attention.__class__ = selected_class
        _apply_profile_attribute(
            attention, "_mtplx_prepare_explicit_qk", explicit_qk
        )
    if (
        bindings.wired_limit_setter is not None
        and bindings.wired_limit_bytes is not None
    ):
        bindings.wired_limit_setter(bindings.wired_limit_bytes)


def _mtp_block_identity(options: Mapping[str, Any]) -> str:
    variant = options.get("mtp_block_variant")
    return "bf16" if variant is None else f"q4-{variant}"


def install_qwen38_performance_profiles(
    runtime: Any,
    config: Mapping[str, Any],
    model_path: Path,
    *,
    stock: Qwen38PerformanceProfileConfig,
    low: Qwen38PerformanceProfileConfig,
    xhigh: Qwen38PerformanceProfileConfig,
    environment: Mapping[str, str] | None = None,
) -> Mapping[str, Qwen38PerformanceProfile]:
    """Validate and bind stock/low/xhigh routes before accepting requests."""

    row53_receipt: dict[str, Any] | None = None
    if any(profile.row53_command_buffers for profile in (stock, low, xhigh)):
        if environment is None:
            import os

            environment = os.environ
        try:
            max_mb = int(environment.get("MLX_MAX_MB_PER_BUFFER", "0") or "0")
            max_ops = int(environment.get("MLX_MAX_OPS_PER_BUFFER", "0") or "0")
        except (TypeError, ValueError) as exc:
            raise Qwen38ContractError(
                "Qwen 3.8 command-buffer contract has non-integer settings"
            ) from exc
        if (max_mb, max_ops) != (512, 50):
            raise Qwen38ContractError(
                "Qwen 3.8 command-buffer contract requires "
                "MLX_MAX_MB_PER_BUFFER=512 and MLX_MAX_OPS_PER_BUFFER=50"
            )
        row53_receipt = {
            "installed": True,
            "active": True,
            "max_mb_per_buffer": max_mb,
            "max_ops_per_buffer": max_ops,
            "process_latched": True,
        }

    installed: dict[str, Qwen38PerformanceProfile] = {}
    for profile_id, profile_config in (
        ("stock", stock),
        ("low", low),
        ("xhigh", xhigh),
    ):
        if profile_config.draft_core not in {"stock", "device"}:
            raise Qwen38ContractError(
                f"{profile_id} performance profile has invalid draft core "
                f"{profile_config.draft_core!r}"
            )
        route = install_qwen38_route(
            runtime,
            config,
            model_path,
            **dict(profile_config.install_options),
        )
        if route is None:
            raise Qwen38ContractError(
                f"{profile_id} performance profile did not install a bound route"
            )
        identity = _mtp_block_identity(profile_config.install_options)
        selected_route = replace(
            route,
            performance_profile=profile_id,
            requested_route_id=profile_config.requested_route_id,
            draft_core=profile_config.draft_core,
            mtp_block_identity=identity,
        )
        feature_receipt = dict(
            getattr(runtime, "qwen38_feature_receipt", {}) or {}
        )
        if profile_config.row53_command_buffers:
            if row53_receipt is None:  # pragma: no cover - guarded above
                raise Qwen38ContractError(
                    "Qwen 3.8 command-buffer contract was not validated"
                )
            feature_receipt["r53_command_buffers"] = dict(row53_receipt)
        if profile_config.installed_route_id is not None:
            if route.route_id != profile_config.installed_route_id:
                raise Qwen38ContractError(
                    f"{profile_id} performance profile installed route mismatch: "
                    f"{route.route_id!r} != {profile_config.installed_route_id!r}"
                )
            if tuple(route.kernel_ids) != tuple(profile_config.kernel_ids):
                raise Qwen38ContractError(
                    f"{profile_id} performance profile kernel contract mismatch: "
                    f"{tuple(route.kernel_ids)!r} != {profile_config.kernel_ids!r}"
                )
            actual_feature_keys = tuple(sorted(feature_receipt))
            expected_feature_keys = tuple(sorted(profile_config.feature_keys))
            if actual_feature_keys != expected_feature_keys:
                raise Qwen38ContractError(
                    f"{profile_id} performance profile feature contract mismatch: "
                    f"{actual_feature_keys!r} != {expected_feature_keys!r}"
                )
        installed[profile_id] = Qwen38PerformanceProfile(
            profile_id=profile_id,
            requested_route_id=profile_config.requested_route_id,
            route=selected_route,
            bindings=_capture_qwen38_execution_bindings(runtime),
            feature_receipt=MappingProxyType(feature_receipt),
            draft_core=profile_config.draft_core,
            mtp_block_identity=identity,
        )
    row50_state = getattr(runtime, "_qwen38_row50_wired_state", None)
    row50_setter = getattr(runtime, "_qwen38_row50_set_wired_limit", None)
    if isinstance(row50_state, Mapping) and row50_state.get("installed"):
        if not callable(row50_setter):
            raise Qwen38ContractError(
                "Qwen 3.8 row 50 route has no prebound wired-limit setter"
            )
        for profile_id, profile_config in (
            ("stock", stock),
            ("low", low),
            ("xhigh", xhigh),
        ):
            limit_key = (
                "target_limit_bytes"
                if profile_config.install_options.get("row50_wired_residency")
                else "baseline_limit_bytes"
            )
            wired_limit_bytes = int(row50_state[limit_key])
            profile = installed[profile_id]
            installed[profile_id] = replace(
                profile,
                bindings=replace(
                    profile.bindings,
                    wired_limit_setter=row50_setter,
                    wired_limit_bytes=wired_limit_bytes,
                ),
            )
    runtime.qwen38_performance_profiles = MappingProxyType(installed)
    select_qwen38_performance_profile(runtime, None)
    return runtime.qwen38_performance_profiles


def select_qwen38_performance_profile(
    runtime: Any,
    reasoning_effort: str | None,
) -> Qwen38PerformanceProfile:
    """Select one already-bound route once at the serialized request boundary."""

    profiles = getattr(runtime, "qwen38_performance_profiles", None)
    if not isinstance(profiles, Mapping):
        raise Qwen38ContractError("Qwen 3.8 performance profiles are not installed")
    profile_id = (
        "xhigh"
        if str(reasoning_effort or "").strip().lower() == "xhigh"
        else "low"
        if str(reasoning_effort or "").strip().lower() == "low"
        else "stock"
    )
    profile = profiles.get(profile_id)
    if not isinstance(profile, Qwen38PerformanceProfile):
        raise Qwen38ContractError(
            f"Qwen 3.8 performance profile {profile_id!r} is unavailable"
        )
    _apply_qwen38_execution_bindings(runtime, profile.bindings)
    runtime.qwen38_route = profile.route
    runtime.qwen38_feature_receipt = dict(profile.feature_receipt)
    runtime.qwen38_selected_performance_profile = profile
    return profile


def policy_fingerprint_with_qwen38_route(
    fingerprint: str,
    route: Qwen38RouteSpec | None,
) -> str:
    if route is None:
        return fingerprint
    return f"{fingerprint};qwen38_route={route.fingerprint}"


def qwen38_route_receipt(route: Qwen38RouteSpec | None) -> dict[str, Any] | None:
    if route is None:
        return None
    receipt = {
        "route_id": route.route_id,
        "fingerprint": route.fingerprint,
        "contract_id": route.contract.contract_id,
        "kernel_ids": list(route.kernel_ids),
        "history_route_id": route.history_route_id,
        "min_context_tokens": route.min_context_tokens,
        "policy_id": route.policy_id,
        "selfcheck": {
            "passed": bool(route.selfcheck_passed),
            "status": route.selfcheck_status,
        },
    }
    if route.performance_profile is not None:
        receipt.update(
            {
                "performance_profile": route.performance_profile,
                "requested_route_id": route.requested_route_id,
                "installed_route_id": route.route_id,
                "draft_core": route.draft_core,
                "mtp_block_identity": route.mtp_block_identity,
            }
        )
    return receipt
