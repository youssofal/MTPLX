#!/usr/bin/env python3
"""Matched real-model gate for Qwen 3.8 challenge-port candidates."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from scripts.qwen38_native_mtp_candidates import (
        FROZEN_TARGET_SUBSTRATE,
        NATIVE_MTP_CANDIDATES,
        NativeMTPRouteError,
        canonicalize_native_mtp_route,
        validate_native_mtp_route_delta,
    )
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from qwen38_native_mtp_candidates import (  # type: ignore[no-redef]
        FROZEN_TARGET_SUBSTRATE,
        NATIVE_MTP_CANDIDATES,
        NativeMTPRouteError,
        canonicalize_native_mtp_route,
        validate_native_mtp_route_delta,
    )

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = Path.home() / (
    ".mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
)
DEFAULT_PROMPT = ROOT / "mtplx/benchmarks/prompts/python_modules_long.jsonl"
DEFAULT_CONTEXT = ROOT / "mtplx/generation.py"
DEFAULT_LOCK = Path("/tmp/mtplx-gpu-exclusive.lock")
PROMOTION_THRESHOLD_PCT = 0.05
REQUIRED_MLX_VERSION = "0.32.2"
REQUIRED_MLX_METAL_VERSION = "0.32.2"
MODEL_ARTIFACT_HASHES_ENV = "MTPLX_QWEN38_MODEL_ARTIFACT_HASHES"
FIXED_NATIVE_ROUTE = "control"
ADAPTIVE_NATIVE_ROUTE = "r11_position_ema"
LOW_ADAPTIVE_SHARED_ROUTE = (
    "r20_kv_only_history+r53_command_buffers+r08_device_draft+"
    "r10_compact_vocab+r21_qk_rms_rope+r24_eval_ladder+"
    "r26_prefill_ladder_3"
)
LOW_FIXED_NATIVE_ROUTE = LOW_ADAPTIVE_SHARED_ROUTE
LOW_ADAPTIVE_NATIVE_ROUTE = LOW_ADAPTIVE_SHARED_ROUTE + "+r11_position_ema"
LOW_Q4_ADAPTIVE_NATIVE_ROUTE = (
    LOW_ADAPTIVE_SHARED_ROUTE + "+r11_position_ema+r17_q4_mtp_block"
)
XHIGH_ADAPTIVE_SHARED_ROUTE = (
    "r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3+"
    "r50_wired_residency+r53_command_buffers"
)
XHIGH_FIXED_NATIVE_ROUTE = XHIGH_ADAPTIVE_SHARED_ROUTE
XHIGH_ADAPTIVE_NATIVE_ROUTE = XHIGH_ADAPTIVE_SHARED_ROUTE + "+r11_position_ema"
XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE = (
    XHIGH_ADAPTIVE_SHARED_ROUTE + "+r11_position_ema+r17_q4_mtp_block"
)
GREEDY_ADAPTIVE_SHARED_ROUTE = (
    "r18_gdn_decay_memo+r20_kv_only_history+r21_qk_rms_rope+"
    "r24_eval_ladder+r26_prefill_ladder_3+r48_boundary_fused+"
    "r50_wired_residency+r61_dual_norm_concat"
)
GREEDY_ADAPTIVE_NATIVE_ROUTE = GREEDY_ADAPTIVE_SHARED_ROUTE + "+r11_position_ema"
GREEDY_Q4_ADAPTIVE_NATIVE_ROUTE = (
    GREEDY_ADAPTIVE_SHARED_ROUTE + "+r28_q4_mtp_block+r11_position_ema"
)
ALLOWED_ROUTE_FEATURES = frozenset(
    {
        "control",
        "kv_only_history",
        "dual_norm",
        "r08_device_draft",
        "r10_compact_vocab",
        "r11_position_ema",
        "r17_q4_mtp_block",
        "r28_q4_mtp_block",
        "r36_qkv_islands",
        "r18_gdn_decay_memo",
        "r20_kv_only_history",
        "r21_qk_rms_rope",
        "r24_eval_ladder",
        "r26_prefill_ladder_3",
        "r48_boundary_fused",
        "r50_wired_residency",
        "r53_command_buffers",
        "r61_dual_norm_concat",
        "r63_q8_embedding_dual_norm",
    }
)
OPTIMIZED_MAIN_BASE = {
    "id": "upstream_main_qwen38_optimized_speed",
    "commit": "bd4421567f9e16ce957c6ef97708b072dcd73937",
    "internal_control_route": "control",
}


def _read_prompt(path: Path) -> tuple[str, str]:
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    return str(row["id"]), str(row["prompt"])


def _expand_prompt_to_token_count(
    tokenizer: Any,
    seed_prompt: str,
    target_tokens: int,
) -> tuple[str, list[int]]:
    """Repeat a fixed seed and truncate its token IDs to an exact cold-prefill size."""

    if target_tokens <= 0:
        raise ValueError("prompt token target must be positive")
    unit = seed_prompt.rstrip() + "\n"
    repeats = 1
    token_ids = list(tokenizer.encode(unit))
    while len(token_ids) < target_tokens:
        repeats *= 2
        token_ids = list(tokenizer.encode(unit * repeats))
    token_ids = token_ids[:target_tokens]
    return str(tokenizer.decode(token_ids)), token_ids


def _context_prompt_to_token_count(
    tokenizer: Any,
    *,
    context: str,
    instruction: str,
    target_tokens: int,
) -> tuple[str, list[int]]:
    """Fill an exact prompt budget with context and one intact tail instruction."""

    if target_tokens <= 0:
        raise ValueError("prompt token target must be positive")
    tail_ids = list(tokenizer.encode("\n\n" + instruction.strip()))
    if len(tail_ids) >= target_tokens:
        raise ValueError("tail instruction does not fit inside prompt token target")
    context_unit = context.rstrip() + "\n"
    context_ids = list(tokenizer.encode(context_unit))
    if not context_ids:
        raise ValueError("context must encode to at least one token")
    context_budget = target_tokens - len(tail_ids)
    repeats = (context_budget + len(context_ids) - 1) // len(context_ids)
    token_ids = (context_ids * repeats)[:context_budget] + tail_ids
    return str(tokenizer.decode(token_ids)), token_ids


def _token_hash(tokens: list[int]) -> str:
    payload = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _generation_metrics(
    stats: Any,
    *,
    verify_strategy: str,
    verify_core: str,
) -> dict[str, Any]:
    peak = int(stats.peak_memory_bytes)
    target_prefill_time = _finite_positive(
        stats.prompt_target_prefill_time_s, "target prefill time"
    )
    target_prefill_rate = _finite_nonnegative(
        stats.prompt_target_prefill_tok_s, "target prefill throughput"
    )
    history_time = _finite_positive(
        stats.prompt_mtp_history_time_s, "MTP history time"
    )
    history_rate = _finite_nonnegative(
        stats.prompt_mtp_history_tok_s, "MTP history throughput"
    )
    proposer_time = _finite_positive(stats.draft_time_s, "MTP proposer time")
    decode_elapsed = _finite_positive(stats.decode_elapsed_s, "decode elapsed time")
    decode_rate = _finite_nonnegative(stats.decode_tok_s, "decode throughput")
    proposer_rate = _finite_nonnegative(
        float(stats.drafted_tokens) / proposer_time,
        "MTP proposer throughput",
    )
    capture_commit_events = sum(
        str(event.get("capture_repair") or "").startswith("captured_")
        for event in stats.events
    )
    return {
        "prefill_tokens": int(stats.new_prefill_tokens),
        "prefill_time_s": target_prefill_time,
        "prefill_tok_s": target_prefill_rate,
        "target_prefill_time_s": target_prefill_time,
        "target_prefill_tok_s": target_prefill_rate,
        "mtp_history_tokens": int(stats.new_prefill_tokens),
        "mtp_history_time_s": history_time,
        "mtp_history_tok_s": history_rate,
        "mtp_decode_tokens": int(stats.drafted_tokens),
        "mtp_decode_time_s": proposer_time,
        "mtp_decode_tok_s": proposer_rate,
        "decode_elapsed_s": decode_elapsed,
        "decode_tok_s": decode_rate,
        "peak_memory_bytes": peak,
        "peak_memory_gib": peak / 2**30,
        "capture_commit_time_s": float(stats.capture_commit_time_s),
        "capture_commit_events": int(capture_commit_events),
        "verify_strategy": str(verify_strategy),
        "verify_core": str(verify_core),
        "speculative_depth": int(stats.speculative_depth),
        "requested_speculative_depth": int(stats.requested_speculative_depth),
        "verify_calls": int(stats.verify_calls),
        "bonus_tokens": int(stats.bonus_tokens),
        "correction_tokens": int(stats.correction_tokens),
        "context_copy_active": bool(stats.context_copy_active),
        "context_copy_rounds": int(stats.context_copy_rounds),
        "context_copy_drafted_tokens": int(stats.context_copy_drafted_tokens),
        "context_copy_accepted_tokens": int(stats.context_copy_accepted_tokens),
    }


def _finite_positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _finite_nonnegative(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return number


def _validated_software_versions(
    *, version_fn: Any = importlib.metadata.version
) -> dict[str, str]:
    """Fail before model load unless both MLX distributions are the pinned build."""

    versions = {
        "mlx": str(version_fn("mlx")),
        "mlx_metal": str(version_fn("mlx-metal")),
    }
    expected = {
        "mlx": REQUIRED_MLX_VERSION,
        "mlx_metal": REQUIRED_MLX_METAL_VERSION,
    }
    errors = [
        f"{name.replace('_', '-')}=={required} required, found {versions[name]}"
        for name, required in expected.items()
        if versions[name] != required
    ]
    if errors:
        raise RuntimeError("; ".join(errors))
    return versions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_hashes(model_path: Path) -> dict[str, str]:
    """Hash every authoritative target and MTP tensor artifact before load."""

    names = (
        "config.json",
        "mtplx_runtime.json",
        "MTPLX_RUNTIME.json",
        "MTPLX_PUBLISH_MANIFEST.json",
        "model.safetensors.index.json",
        "mtp.safetensors",
    )
    paths = {
        name: path
        for name in names
        if (path := model_path / name).is_file()
    }
    for required in ("config.json", "model.safetensors.index.json", "mtp.safetensors"):
        if required not in paths:
            raise RuntimeError(f"model artifact {required} is missing before load")
    try:
        index = json.loads(paths["model.safetensors.index.json"].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid model shard index: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model shard index has no weight_map")
    root = model_path.resolve()
    for raw_name in sorted(set(weight_map.values())):
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("model shard index contains an invalid shard path")
        shard = (model_path / raw_name).resolve()
        if not shard.is_relative_to(root):
            raise RuntimeError(f"model shard escapes artifact root: {raw_name}")
        if not shard.is_file():
            raise RuntimeError(f"referenced model shard is missing: {raw_name}")
        paths[raw_name] = shard
    return {name: _sha256_file(path) for name, path in sorted(paths.items())}


def _validate_model_artifact_hash_receipt(
    model_path: Path,
    hashes: dict[str, str],
) -> None:
    required = {
        "config.json",
        "model.safetensors.index.json",
        "mtp.safetensors",
    }
    runtime_manifest = model_path / "mtplx_runtime.json"
    if runtime_manifest.is_file():
        required.add("mtplx_runtime.json")
    index_path = model_path / "model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid model shard index: {exc}") from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model shard index has no weight_map")
    required.update(str(name) for name in weight_map.values())
    for name in sorted(required):
        if name not in hashes:
            raise RuntimeError(f"attested model artifact hashes missing {name}")
        digest = hashes[name]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"attested model artifact hash is invalid for {name}")
        if not (model_path / name).is_file():
            raise RuntimeError(f"attested model artifact is missing: {name}")


def _attested_model_artifact_hashes(
    model_path: Path,
    *,
    guarded_by_parent: bool,
    environment: Any = os.environ,
) -> dict[str, str]:
    """Reuse the lock-owning parent's byte hashes; direct owners hash once."""

    encoded = environment.get(MODEL_ARTIFACT_HASHES_ENV)
    if guarded_by_parent:
        if not encoded:
            raise RuntimeError("attested parent did not provide model artifact hashes")
        try:
            hashes = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise RuntimeError("attested model artifact hashes are invalid JSON") from exc
        if not isinstance(hashes, dict) or not hashes:
            raise RuntimeError("attested model artifact hashes are empty")
        normalized = {str(name): str(digest) for name, digest in hashes.items()}
        _validate_model_artifact_hash_receipt(model_path, normalized)
        return normalized
    return _model_artifact_hashes(model_path)


def _frozen_substrate_fingerprint(
    *,
    model_path: Path,
    model_artifact_hashes: dict[str, str],
    route_ids: list[str],
) -> str:
    frozen_features = {spec.feature for spec in FROZEN_TARGET_SUBSTRATE.values()}
    selected_frozen_features = sorted(
        set().union(*(_validate_route_id(route_id) for route_id in route_ids))
        & frozen_features
    )
    payload = {
        "base": OPTIMIZED_MAIN_BASE,
        "model_path": str(model_path),
        "model_artifact_hashes": model_artifact_hashes,
        "selected_frozen_features": selected_frozen_features,
        "frozen_target_substrate": {
            str(row): {
                "feature": spec.feature,
                "source_commit": spec.source_commit,
                "owned_surface": spec.owned_surface,
            }
            for row, spec in sorted(FROZEN_TARGET_SUBSTRATE.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _candidate_artifact_hashes(
    feature_receipt: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    for feature, report in feature_receipt.items():
        if not isinstance(report, dict) or not report.get("file_sha256"):
            continue
        hashes[feature] = {
            key: report[key]
            for key in (
                "manifest_sha256",
                "file_sha256",
                "artifact_bytes",
            )
            if key in report
        }
    return hashes


def _expected_candidate_artifact_hashes(feature: str) -> dict[str, Any] | None:
    spec = NATIVE_MTP_CANDIDATES.get(feature)
    if spec is None or spec.ownership != "artifact":
        return None
    return {
        "manifest_sha256": spec.artifact_manifest_sha256,
        "file_sha256": spec.artifact_file_sha256,
        "artifact_bytes": spec.artifact_bytes,
    }


def _phase_summary(
    arms: list[dict[str, Any]],
    *,
    control_id: str,
    candidate_id: str,
) -> dict[str, Any]:
    """Aggregate existing external stats into sortable phase measurements."""

    phase_fields = (
        ("wall", "wall_s", None),
        ("target_prefill", "target_prefill_time_s", "target_prefill_tok_s"),
        ("mtp_history", "mtp_history_time_s", "mtp_history_tok_s"),
        ("mtp_decode", "mtp_decode_time_s", "mtp_decode_tok_s"),
        ("decode", "decode_elapsed_s", "decode_tok_s"),
    )
    routes = (control_id, candidate_id)
    mean_time_s: dict[str, dict[str, float]] = {}
    mean_throughput: dict[str, dict[str, float]] = {}
    for route in routes:
        route_arms = [arm for arm in arms if arm.get("route_id") == route]
        if not route_arms:
            raise ValueError(f"phase summary has no arms for route {route!r}")
        mean_time_s[route] = {
            phase: _finite_positive(
                math.fsum(
                    _finite_positive(arm[time_key], f"{route} {phase} time")
                    for arm in route_arms
                )
                / len(route_arms),
                f"{route} mean {phase} time",
            )
            for phase, time_key, _ in phase_fields
        }
        mean_throughput[route] = {
            phase: _finite_nonnegative(
                math.fsum(
                    _finite_nonnegative(
                        arm[rate_key], f"{route} {phase} throughput"
                    )
                    for arm in route_arms
                )
                / len(route_arms),
                f"{route} mean {phase} throughput",
            )
            for phase, _, rate_key in phase_fields
            if rate_key is not None
        }
    time_improvement_pct = {
        phase: (
            mean_time_s[control_id][phase] / mean_time_s[candidate_id][phase]
            - 1.0
        )
        * 100.0
        if mean_time_s[candidate_id][phase] > 0.0
        else 0.0
        for phase, _, _ in phase_fields
    }
    throughput_improvement_pct: dict[str, float] = {}
    for phase, _, rate_key in phase_fields:
        if rate_key is None:
            continue
        denominator = _finite_positive(
            mean_throughput[control_id][phase],
            f"{control_id} mean {phase} throughput denominator",
        )
        throughput_improvement_pct[phase] = (
            mean_throughput[candidate_id][phase] / denominator - 1.0
        ) * 100.0
    for label, values in (
        ("time improvement", time_improvement_pct),
        ("throughput improvement", throughput_improvement_pct),
    ):
        for phase, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"{phase} {label} must be finite")
    return {
        "phase_order": [phase for phase, _, _ in phase_fields],
        "mean_time_s": mean_time_s,
        "mean_throughput_tok_s": mean_throughput,
        "time_improvement_pct": time_improvement_pct,
        "throughput_improvement_pct": throughput_improvement_pct,
    }


def _correctness_summary(
    arms: list[dict[str, Any]],
    *,
    route_ids: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    """Require exact output and schedule replay for the retained route."""

    cross_route_token_exact = len({arm["token_hash"] for arm in arms}) == 1

    def schedule_fingerprint(arm: dict[str, Any]) -> tuple[Any, ...] | None:
        attempted = tuple(arm.get("attempted_depth_schedule") or ())
        accepted = tuple(arm.get("accepted_depth_schedule") or ())
        if attempted or accepted:
            return ("events", attempted, accepted)
        drafted_by_depth = tuple(arm.get("drafted_by_depth") or ())
        accepted_by_depth = tuple(arm.get("accepted_by_depth") or ())
        if drafted_by_depth or accepted_by_depth:
            return ("depth_histograms", drafted_by_depth, accepted_by_depth)
        return None

    schedule_fingerprints = [schedule_fingerprint(arm) for arm in arms]
    cross_route_schedule_exact = bool(
        schedule_fingerprints
        and all(value is not None for value in schedule_fingerprints)
        and len(set(schedule_fingerprints)) == 1
    )
    per_route_deterministic = {
        route_id: len(
            {
                (arm["token_hash"], schedule_fingerprint(arm))
                for arm in arms
                if arm["route_id"] == route_id
            }
        )
        == 1
        and all(
            schedule_fingerprint(arm) is not None
            for arm in arms
            if arm["route_id"] == route_id
        )
        for route_id in route_ids
    }
    full_output = all(int(arm["generated_tokens"]) == max_tokens for arm in arms)
    deterministic = all(per_route_deterministic.values())
    passed = bool(full_output and deterministic)
    exact = bool(cross_route_token_exact and cross_route_schedule_exact)
    return {
        "passed": passed,
        "mode": "exact" if exact else ("deterministic_drift" if passed else "rejected"),
        "full_output": full_output,
        "cross_route_token_exact": cross_route_token_exact,
        "cross_route_schedule_exact": cross_route_schedule_exact,
        "schedule_capture": (
            "events"
            if schedule_fingerprints
            and all(value is not None and value[0] == "events" for value in schedule_fingerprints)
            else "depth_histograms"
            if schedule_fingerprints
            and all(
                value is not None and value[0] == "depth_histograms"
                for value in schedule_fingerprints
            )
            else "missing"
        ),
        "per_route_deterministic": per_route_deterministic,
    }


def _validate_route_id(route_id: str) -> set[str]:
    return set(canonicalize_native_mtp_route(route_id))


def _route_execution_options(route_id: str) -> dict[str, Any]:
    """Translate chronological proposal features into one cumulative run."""

    features = _validate_route_id(route_id)
    source_rows: list[int] = []
    if "r08_device_draft" in features:
        source_rows.append(8)
    if "r10_compact_vocab" in features:
        source_rows.append(10)
    if "r17_q4_mtp_block" in features:
        source_rows.append(17)
    if "r18_gdn_decay_memo" in features:
        source_rows.append(18)
    if "r20_kv_only_history" in features:
        source_rows.append(20)
    if "r21_qk_rms_rope" in features:
        source_rows.append(21)
    if "r24_eval_ladder" in features:
        source_rows.append(24)
    if "r26_prefill_ladder_3" in features:
        source_rows.append(26)
    if "r28_q4_mtp_block" in features:
        source_rows.append(28)
    if "r36_qkv_islands" in features:
        source_rows.append(36)
    if "r48_boundary_fused" in features:
        source_rows.append(48)
    if "r50_wired_residency" in features:
        source_rows.append(50)
    if "r53_command_buffers" in features:
        source_rows.append(53)
    if "r61_dual_norm_concat" in features:
        source_rows.append(61)
    if "r63_q8_embedding_dual_norm" in features:
        source_rows.append(63)
    if "r11_position_ema" in features:
        source_rows.append(11)
    return {
        "cache_route": (
            "kv_only_history"
            if {"kv_only_history", "r20_kv_only_history"} & features
            else "control"
        ),
        "dual_norm": bool({"dual_norm", "r61_dual_norm_concat"} & features),
        "row63_q8_embedding_dual_norm": "r63_q8_embedding_dual_norm" in features,
        "row10_compact_vocab": "r10_compact_vocab" in features,
        "adaptive_policy": (
            "position_ema" if "r11_position_ema" in features else "none"
        ),
        "speculative_depth": 3,
        "adaptive_depth_cap": 3 if "r11_position_ema" in features else 0,
        "mtp_block_variant": (
            "r36"
            if "r36_qkv_islands" in features
            else "r28"
            if "r28_q4_mtp_block" in features
            else "r17"
            if "r17_q4_mtp_block" in features
            else None
        ),
        "row18_gdn_decay_memo": "r18_gdn_decay_memo" in features,
        "row21_qk_rms_rope": "r21_qk_rms_rope" in features,
        "row24_eval_ladder": "r24_eval_ladder" in features,
        "row26_prefill_ladder_3": "r26_prefill_ladder_3" in features,
        "row48_boundary_fused": "r48_boundary_fused" in features,
        "row50_wired_residency": "r50_wired_residency" in features,
        "row53_command_buffers": "r53_command_buffers" in features,
        "draft_core": "device" if "r08_device_draft" in features else "stock",
        "source_rows": tuple(source_rows),
    }


def _validate_process_latched_route(
    options: dict[str, Any],
    *,
    environment: Any = os.environ,
) -> None:
    if not bool(options["row53_command_buffers"]):
        return
    max_mb = int(environment.get("MLX_MAX_MB_PER_BUFFER", "0") or "0")
    max_ops = int(environment.get("MLX_MAX_OPS_PER_BUFFER", "0") or "0")
    if (max_mb, max_ops) != (512, 50):
        raise RuntimeError(
            "Qwen 3.8 row 53 process contract requires "
            "MLX_MAX_MB_PER_BUFFER=512 and MLX_MAX_OPS_PER_BUFFER=50"
        )


def _conditioning_order(
    order: list[str],
    *,
    candidate_id: str | None,
) -> list[str]:
    """Condition each route once without letting row 50 erase control warmup."""

    unique_routes = list(dict.fromkeys(order))
    inferred_candidate = candidate_id
    if inferred_candidate is None and len(unique_routes) == 2:
        inferred_candidate = unique_routes[1]
    if (
        inferred_candidate is not None
        and inferred_candidate in unique_routes
        and "r50_wired_residency" in _validate_route_id(inferred_candidate)
    ):
        return [
            inferred_candidate,
            *[route for route in unique_routes if route != inferred_candidate],
        ]
    return unique_routes


def _promotion_decision(
    *,
    order: list[str],
    control_id: str | None,
    candidate_id: str | None,
    improvement_pct: float | None,
    correctness: dict[str, Any],
    source_status: list[str],
    engagement_errors: list[str] | None = None,
    allow_frozen_candidate: bool = False,
    validated_route_delta: Any | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_order = (
        [control_id, candidate_id, candidate_id, control_id]
        if control_id is not None and candidate_id is not None
        else []
    )
    if order != expected_order:
        errors.append("gate requires exactly four timed ABBA arms")
    if control_id is None or candidate_id is None:
        errors.append("gate requires explicit control and candidate routes")
    elif validated_route_delta is not None:
        validated_control = set(validated_route_delta.control_features)
        validated_candidate_features = getattr(
            validated_route_delta,
            "candidate_features_set",
            None,
        )
        if validated_candidate_features is None:
            validated_candidate_features = validated_route_delta.candidate_features
        validated_candidate = set(validated_candidate_features)
        if (
            validated_control != (_validate_route_id(control_id) - {"control"})
            or validated_candidate
            != (_validate_route_id(candidate_id) - {"control"})
        ):
            errors.append("prevalidated native-MTP route delta does not match gate routes")
    else:
        try:
            validate_native_mtp_route_delta(
                control_id,
                candidate_id,
                allow_frozen_candidate=allow_frozen_candidate,
            )
        except NativeMTPRouteError as exc:
            errors.append(f"native-MTP route delta rejected: {exc}")
    if improvement_pct is None or not math.isfinite(float(improvement_pct)):
        errors.append("candidate improvement must be finite")
    elif improvement_pct <= PROMOTION_THRESHOLD_PCT:
        errors.append(
            "candidate improvement must be strictly greater than "
            f"{PROMOTION_THRESHOLD_PCT:.2f}%"
        )
    if not bool(correctness.get("passed")):
        errors.append("correctness/determinism gate did not pass")
    if source_status:
        errors.append("promotion receipt requires a clean source tree")
    errors.extend(engagement_errors or [])
    return {
        "passed": not errors,
        "threshold_pct": PROMOTION_THRESHOLD_PCT,
        "errors": errors,
    }


def _candidate_engagement_errors(
    candidate_route: str | None,
    warmups: list[dict[str, Any]],
    arms: list[dict[str, Any]],
) -> list[str]:
    """Reject a candidate whose route was configured but never traced."""

    if candidate_route is None:
        return []
    features = _validate_route_id(candidate_route)
    del warmups
    candidate_runs = [
        run
        for run in arms
        if run.get("route_id") == candidate_route
    ]
    errors: list[str] = []
    if not candidate_runs:
        return ["candidate timed arms are missing"]
    if "r08_device_draft" in features and not all(
        run.get("draft_core") == "device"
        and sum(int(value) for value in run.get("drafted_by_depth") or ()) > 0
        for run in candidate_runs
    ):
        errors.append("row 8 device draft route did not execute")
    if "r11_position_ema" in features and not all(
        any(
            event.get("kind") == "position_ema"
            for event in (run.get("adaptive_policy_events") or [])
        )
        or (
            (receipt := run.get("adaptive_policy_receipt")) is not None
            and receipt.get("kind") == "position_ema"
            and bool(receipt.get("executed"))
        )
        for run in candidate_runs
    ):
        errors.append("row 11 position-EMA adaptive policy did not execute")

    receipt_features = {
        "r10_compact_vocab": "r10_compact_vocab",
        "r17_q4_mtp_block": "r17_q4_mtp_block",
        "r18_gdn_decay_memo": "r18_gdn_decay_memo",
        "r21_qk_rms_rope": "r21_qk_rms_rope",
        "r24_eval_ladder": "r24_eval_ladder",
        "r26_prefill_ladder_3": "r26_prefill_ladder_3",
        "r28_q4_mtp_block": "r28_q4_mtp_block",
        "r36_qkv_islands": "r36_qkv_islands",
        "r48_boundary_fused": "r48_boundary_fused",
        "r50_wired_residency": "r50_wired_residency",
        "r53_command_buffers": "r53_command_buffers",
        "r61_dual_norm_concat": "dual_norm",
        "r63_q8_embedding_dual_norm": "r63_q8_embedding_dual_norm",
    }

    def installed(report: dict[str, Any]) -> bool:
        return bool(
            report.get("active")
            or report.get("installed") and report.get("active") is not False
            or int(report.get("active_modules", 0)) > 0
        )

    for feature, receipt_key in receipt_features.items():
        if feature not in features:
            continue
        if feature in {"r17_q4_mtp_block", "r53_command_buffers"}:
            continue
        if not all(
            installed(((run.get("feature_receipt") or {}).get(receipt_key) or {}))
            for run in candidate_runs
        ):
            errors.append(f"{feature} construction receipt is not active")
    if "r17_q4_mtp_block" in features and not all(
        bool((report := ((run.get("feature_receipt") or {}).get("r17_q4_mtp_block") or {})).get("active"))
        and report.get("variant") == "r17"
        and int(report.get("bits", 0)) == 4
        and int(report.get("group_size", 0)) == 64
        for run in candidate_runs
    ):
        errors.append("row 17 pinned Q4/group-64 MTP block was not active")
    if "r53_command_buffers" in features and not all(
        bool(
            (
                report := (
                    (run.get("feature_receipt") or {}).get(
                        "r53_command_buffers"
                    )
                    or {}
                )
            ).get("active")
        )
        and int(report.get("max_mb_per_buffer", 0)) == 512
        and int(report.get("max_ops_per_buffer", 0)) == 50
        for run in candidate_runs
    ):
        errors.append("row 53 process-latched command-buffer profile was not active")
    if "r20_kv_only_history" in features:
        route_receipts = [
            run.get("history_route_receipt") or {} for run in candidate_runs
        ]
        if not route_receipts:
            errors.append("row 20 request route receipt is missing")
        for receipt in route_receipts:
            prompt_tokens = int(receipt.get("prompt_tokens", 0))
            expected = prompt_tokens >= 16_384
            expected_route = "kv_only_history" if expected else "stock_history"
            if (
                bool(receipt.get("row20_engaged")) != expected
                or receipt.get("route_id") != expected_route
            ):
                errors.append("row 20 request route receipt contradicts prompt phase")
                break
    return errors


def _load_optimized_speed_stack(
    model_path: Path,
    runtime_contract: dict[str, Any],
    *,
    apply_profile_env_fn: Any = None,
    load_runtime_fn: Any = None,
    install_draft_head_fn: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Construct the same Turbo/Q4 draft stack used by Optimized-Speed serving."""

    from mtplx.draft_lm_head import draft_lm_head_spec_from_runtime_contract
    from mtplx.draft_sampling import draft_sampler_spec_from_runtime_contract
    from mtplx.profiles import (
        get_profile,
        runtime_env_overrides_from_contract,
    )

    profile = get_profile("turbo")
    fallback_head = {
        "bits": int(profile.draft_lm_head.bits),
        "group_size": int(profile.draft_lm_head.group_size),
        "mode": str(profile.draft_lm_head.mode),
    }
    draft_head = draft_lm_head_spec_from_runtime_contract(
        runtime_contract,
        fallback=fallback_head,
    )
    if draft_head is None:  # pragma: no cover - Turbo always has this requirement
        raise RuntimeError("Turbo profile requires a draft-only LM head")
    draft_sampler = draft_sampler_spec_from_runtime_contract(runtime_contract)
    runtime_env_overrides = runtime_env_overrides_from_contract(runtime_contract)

    if apply_profile_env_fn is None:
        from mtplx.profiles import apply_profile_env

        apply_profile_env_fn = apply_profile_env
    apply_profile_env_fn(
        profile.name,
        runtime_env_overrides=runtime_env_overrides,
    )

    # Runtime modules that bind env-gated kernels are deliberately imported
    # only after the production profile has populated the process environment.
    if load_runtime_fn is None:
        from mtplx.runtime import load

        load_runtime_fn = load
    runtime = load_runtime_fn(model_path, mtp=True)
    if install_draft_head_fn is None:
        from mtplx.draft_lm_head import _install_draft_lm_head

        install_draft_head_fn = _install_draft_lm_head
    draft_head_report = install_draft_head_fn(
        runtime,
        bits=int(draft_head["bits"]),
        group_size=int(draft_head["group_size"]),
        mode=str(draft_head["mode"]),
    )
    return runtime, {
        "base_stack": dict(OPTIMIZED_MAIN_BASE),
        "profile": profile.name,
        "runtime_profile": profile.runtime_profile,
        "runtime_env": {**profile.env_dict(), **runtime_env_overrides},
        "draft_lm_head": draft_head,
        "draft_lm_head_report": draft_head_report,
        "draft_sampler": draft_sampler,
        "mtp_hidden_variant": "post_norm",
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
    }


def _run_arm(
    runtime: Any,
    config: dict[str, Any],
    model_path: Path,
    prompt_ids: list[int],
    *,
    route_id: str,
    max_tokens: int,
    seed: int,
    target_temperature: float,
    draft_temperature: float,
    top_p: float,
    top_k: int,
    row17_artifact_path: Path | None,
    row28_artifact_path: Path | None,
    row36_artifact_path: Path | None,
    performance_profile: str,
    stop_token_ids: set[int] | None = None,
) -> dict[str, Any]:
    options = _route_execution_options(route_id)
    _validate_process_latched_route(options)

    import mlx.core as mx

    from mtplx.generation import generate_mtpk
    from mtplx.adaptive import PositionEMADepthPolicy
    from mtplx.qwen38_challenge import install_qwen38_route
    from mtplx.sampling import SamplerConfig
    route = install_qwen38_route(
        runtime,
        config,
        model_path,
        cache_route=str(options["cache_route"]),
        dual_norm=bool(options["dual_norm"]),
        row10_compact_vocab=bool(options["row10_compact_vocab"]),
        mtp_block_variant=options["mtp_block_variant"],
        mtp_block_artifact_path=(
            row36_artifact_path
            if options["mtp_block_variant"] == "r36"
            else row28_artifact_path
            if options["mtp_block_variant"] == "r28"
            else row17_artifact_path
        ),
        row18_gdn_decay_memo=bool(options["row18_gdn_decay_memo"]),
        row21_qk_rms_rope=bool(options["row21_qk_rms_rope"]),
        row24_eval_ladder=bool(options["row24_eval_ladder"]),
        row26_prefill_ladder_3=bool(options["row26_prefill_ladder_3"]),
        row48_boundary_fused=bool(options["row48_boundary_fused"]),
        row50_wired_residency=bool(options["row50_wired_residency"]),
        row63_q8_embedding_dual_norm=bool(
            options["row63_q8_embedding_dual_norm"]
        ),
    )
    target_sampler = SamplerConfig(
        temperature=target_temperature,
        top_p=top_p,
        top_k=top_k,
    )
    draft_sampler = SamplerConfig(
        temperature=draft_temperature,
        top_p=top_p,
        top_k=top_k,
    )
    history_route_receipt = runtime.bind_mtp_history_append_route(
        len(prompt_ids)
    ).receipt
    adaptive_policy = (
        PositionEMADepthPolicy(
            max_depth=int(options["speculative_depth"]),
            depth_cap=int(options["adaptive_depth_cap"]),
        )
        if options["adaptive_policy"] == "position_ema"
        else None
    )
    adaptive_policy_initial_state = (
        {
            "accept_ema": tuple(
                float(value) for value in adaptive_policy.position_accept_ema
            ),
            "depth": int(adaptive_policy.current_depth),
        }
        if adaptive_policy is not None
        else None
    )
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=target_sampler,
        draft_sampler=draft_sampler,
        speculative_depth=int(options["speculative_depth"]),
        adaptive_policy=adaptive_policy,
        seed=seed,
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        draft_core=str(options["draft_core"]),
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        mtp_history_policy="committed",
        stop_token_ids=stop_token_ids,
    )
    wall_s = time.perf_counter() - started
    stats = output.stats
    compiled_verify_receipt = dict(
        (dict(getattr(stats, "graphbank", {}) or {}).get("compiled_verify") or {})
    )
    device_core_receipt = dict(getattr(stats, "draft_core", {}) or {})
    adaptive_policy_receipt = None
    if adaptive_policy is not None and adaptive_policy_initial_state is not None:
        final_accept_ema = tuple(
            float(value) for value in adaptive_policy.position_accept_ema
        )
        adaptive_policy_receipt = {
            "kind": "position_ema",
            "executed": final_accept_ema
            != adaptive_policy_initial_state["accept_ema"],
            "initial_accept_ema": list(adaptive_policy_initial_state["accept_ema"]),
            "final_accept_ema": list(final_accept_ema),
            "initial_depth": int(adaptive_policy_initial_state["depth"]),
            "final_depth": int(adaptive_policy.current_depth),
            "max_depth": int(options["speculative_depth"]),
            "depth_cap": int(options["adaptive_depth_cap"]),
        }
    feature_receipt = dict(
        getattr(runtime, "qwen38_feature_receipt", {}) or {}
    )
    if options["row53_command_buffers"]:
        max_mb = int(os.environ.get("MLX_MAX_MB_PER_BUFFER", "0") or "0")
        max_ops = int(os.environ.get("MLX_MAX_OPS_PER_BUFFER", "0") or "0")
        feature_receipt["r53_command_buffers"] = {
            "installed": True,
            "active": max_mb == 512 and max_ops == 50,
            "max_mb_per_buffer": max_mb,
            "max_ops_per_buffer": max_ops,
            "process_latched": True,
        }
    return {
        **_generation_metrics(
            stats,
            verify_strategy="capture_commit",
            verify_core="linear-gdn-from-conv-tape",
        ),
        "route_id": route_id,
        "performance_profile": performance_profile,
        "requested_route_id": route_id,
        "installed_route_id": route.route_id,
        "route_fingerprint": hashlib.sha256(
            f"{route.fingerprint}:{route_id}".encode()
        ).hexdigest(),
        "kernel_ids": list(route.kernel_ids),
        "feature_receipt": feature_receipt,
        "candidate_artifact_hashes": _candidate_artifact_hashes(feature_receipt),
        "process_environment": {
            name: os.environ.get(name)
            for name in ("MLX_MAX_MB_PER_BUFFER", "MLX_MAX_OPS_PER_BUFFER")
        },
        "source_rows": list(options["source_rows"]),
        "draft_core": str(options["draft_core"]),
        "adaptive_policy_state": str(options["adaptive_policy"]),
        "mtp_block_identity": (
            "bf16"
            if options["mtp_block_variant"] is None
            else f"q4-{options['mtp_block_variant']}"
        ),
        "history_route_receipt": history_route_receipt,
        "wall_s": wall_s,
        "generated_tokens": int(stats.generated_tokens),
        "prompt_mtp_history_time_s": float(stats.prompt_mtp_history_time_s),
        "draft_time_s": float(stats.draft_time_s),
        "accepted_by_depth": list(stats.accepted_by_depth),
        "drafted_by_depth": list(stats.drafted_by_depth),
        "attempted_depth_schedule": [
            int(event.get("depth", 0)) for event in stats.events
        ],
        "accepted_depth_schedule": [
            int(event.get("accepted_depths", 0)) for event in stats.events
        ],
        "adaptive_policy_events": [
            dict(event["policy"])
            for event in stats.events
            if isinstance(event.get("policy"), dict)
        ],
        "adaptive_policy_receipt": adaptive_policy_receipt,
        "compiled_verify_receipt": compiled_verify_receipt,
        "device_core_receipt": device_core_receipt,
        "token_hash": _token_hash(list(output.tokens)),
        "tokens": list(output.tokens),
        "finish_reason": output.finish_reason,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--draft-temperature", type=float)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "xhigh"),
        default="low",
    )
    parser.add_argument(
        "--order",
        default="control,kv_only_history,kv_only_history,control",
    )
    parser.add_argument("--control-route")
    parser.add_argument("--candidate-route")
    parser.add_argument("--row17-artifact", type=Path)
    parser.add_argument("--row28-artifact", type=Path)
    parser.add_argument("--row36-artifact", type=Path)
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=1024,
        help="Full-output conditioning tokens per route before timed arms.",
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_path = args.model.expanduser().resolve()
    from scripts.qwen35b_mtp_batch_numerics_attribution import (
        _verify_parent_guard_attestation,
    )

    guarded_by_parent = _verify_parent_guard_attestation(args.lock)
    lock_handle = None
    if not guarded_by_parent:
        lock_handle = args.lock.open("a+")
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"GPU lock is busy: {args.lock}") from exc

    source_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    if source_status:
        raise RuntimeError("exact campaign requires a clean source tree")
    software_versions = _validated_software_versions()
    model_artifact_hashes = _attested_model_artifact_hashes(
        model_path,
        guarded_by_parent=guarded_by_parent,
    )
    order = [item.strip() for item in args.order.split(",") if item.strip()]
    if not order:
        raise ValueError("order must contain at least one route")
    for item in order:
        _validate_route_id(item)
        _validate_process_latched_route(_route_execution_options(item))
    frozen_substrate_fingerprint = _frozen_substrate_fingerprint(
        model_path=model_path,
        model_artifact_hashes=model_artifact_hashes,
        route_ids=order,
    )

    from mtplx.backends.registry import load_runtime_contract

    contract, contract_error = load_runtime_contract(model_path)
    if contract_error is not None:
        raise RuntimeError(f"invalid runtime contract: {contract_error}")
    runtime_contract = {} if contract is None else contract.raw
    runtime, optimized_stack = _load_optimized_speed_stack(
        model_path,
        runtime_contract,
    )
    from mtplx.artifacts import load_config

    config = load_config(model_path)
    draft_temperature = (
        float(args.draft_temperature)
        if args.draft_temperature is not None
        else float((optimized_stack.get("draft_sampler") or {}).get("temperature", 1.0))
    )
    prompt_id, prompt = _read_prompt(args.prompt_file)
    if args.prompt_tokens is None:
        prompt_ids = list(runtime.tokenizer.encode(prompt))
    elif args.context_file is not None:
        from scripts.qwen38_native_mtp_matrix_arm import build_prompt

        prompt, prompt_ids = build_prompt(
            runtime.tokenizer,
            workload=args.reasoning_effort,
            context=args.context_file.read_text(encoding="utf-8"),
            instruction=prompt,
            target_tokens=args.prompt_tokens,
        )
    else:
        prompt, prompt_ids = _expand_prompt_to_token_count(
            runtime.tokenizer,
            prompt,
            args.prompt_tokens,
        )
    unique_routes = list(dict.fromkeys(order))
    conditioning_routes = _conditioning_order(
        order,
        candidate_id=args.candidate_route,
    )

    warmups = [
        _run_arm(
            runtime,
            config,
            model_path,
            prompt_ids,
            route_id=route_id,
            max_tokens=args.warmup_tokens,
            seed=args.seed,
            target_temperature=args.target_temperature,
            draft_temperature=draft_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            row17_artifact_path=args.row17_artifact,
            row28_artifact_path=args.row28_artifact,
            row36_artifact_path=args.row36_artifact,
            performance_profile=args.reasoning_effort,
        )
        for route_id in conditioning_routes
    ]
    arms = [
        _run_arm(
            runtime,
            config,
            model_path,
            prompt_ids,
            route_id=route_id,
            max_tokens=args.max_tokens,
            seed=args.seed,
            target_temperature=args.target_temperature,
            draft_temperature=draft_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            row17_artifact_path=args.row17_artifact,
            row28_artifact_path=args.row28_artifact,
            row36_artifact_path=args.row36_artifact,
            performance_profile=args.reasoning_effort,
        )
        for route_id in order
    ]
    correctness = _correctness_summary(
        arms,
        route_ids=unique_routes,
        max_tokens=args.max_tokens,
    )
    exact = bool(
        correctness["cross_route_token_exact"]
        and correctness["cross_route_schedule_exact"]
    )
    by_route = {
        route_id: [arm["wall_s"] for arm in arms if arm["route_id"] == route_id]
        for route_id in unique_routes
    }
    means = {
        route_id: _finite_positive(
            math.fsum(
                _finite_positive(value, f"{route_id} wall time")
                for value in values
            )
            / len(values),
            f"{route_id} mean wall time",
        )
        for route_id, values in by_route.items()
        if values
    }
    control_id = args.control_route
    candidate_id = args.candidate_route
    if control_id is None and candidate_id is None and len(unique_routes) == 2:
        control_id, candidate_id = unique_routes
    if (control_id is None) != (candidate_id is None):
        raise ValueError("control-route and candidate-route must be supplied together")
    if control_id is not None and (
        control_id not in unique_routes or candidate_id not in unique_routes
    ):
        raise ValueError("control-route and candidate-route must occur in order")
    improvement_pct = None
    if control_id is not None and candidate_id is not None:
        improvement_pct = (means[control_id] / means[candidate_id] - 1.0) * 100.0
    promotion = _promotion_decision(
        order=order,
        control_id=control_id,
        candidate_id=candidate_id,
        improvement_pct=improvement_pct,
        correctness=correctness,
        source_status=source_status,
        engagement_errors=_candidate_engagement_errors(
            candidate_id,
            warmups,
            arms,
        ),
    )
    receipt = {
        "kind": "qwen38_challenge_port_gate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(model_path),
        "prompt_file": str(args.prompt_file.resolve()),
        "context_file": (
            None if args.context_file is None else str(args.context_file.resolve())
        ),
        "context_sha256": (
            None
            if args.context_file is None
            else hashlib.sha256(args.context_file.read_bytes()).hexdigest()
        ),
        "prompt_id": prompt_id,
        "prompt_tokens": len(prompt_ids),
        "prompt_token_sha256": _token_hash(prompt_ids),
        "prompt_token_target": args.prompt_tokens,
        "enable_thinking": True,
        "reasoning_effort": args.reasoning_effort,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "target_temperature": args.target_temperature,
        "draft_temperature": draft_temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "optimized_speed_stack": optimized_stack,
        "order": order,
        "gpu_lock_scope": "attested_parent" if guarded_by_parent else "direct",
        "gpu_lock_path": str(args.lock.resolve()),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "mlx_version": software_versions["mlx"],
        "mlx_metal_version": software_versions["mlx_metal"],
        "model_artifact_hashes": model_artifact_hashes,
        "frozen_substrate_fingerprint": frozen_substrate_fingerprint,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "source_status": source_status,
        "exact": exact,
        "token_exact": correctness["cross_route_token_exact"],
        "schedule_exact": correctness["cross_route_schedule_exact"],
        "correctness": correctness,
        "control_route_id": control_id,
        "candidate_route_id": candidate_id,
        "mean_wall_s": means,
        "candidate_improvement_pct": improvement_pct,
        "candidate_engagement_errors": _candidate_engagement_errors(
            candidate_id,
            warmups,
            arms,
        ),
        "promotion": promotion,
        "warmups": warmups,
        "arms": arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "exact": exact,
                "candidate_improvement_pct": improvement_pct,
                "output": str(args.output),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if promotion["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
