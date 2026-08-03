#!/usr/bin/env python3
"""Validate the guarded DeepSeek-V4 MoE-tail primer/C0/B/C1 K3 bracket.

The verdict is persisted before a loss returns nonzero.  Correct-but-slower
results therefore remain auditable and can never be mistaken for a promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


_MODEL_PATH = "/Users/davidtai/models/DeepSeek-V4-Flash-2bit-DQ-mtp"
_PROMPT_PATH = (
    "/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4/"
    "smoke-2bitdq-20260731-prompt2.txt"
)
_PROMPT_SHA256 = "ee94397faa812c91d5f1a0ee17c5bb6ca6032883653591dd33d4cfddb737ac33"
_CONFIG_SHA256 = "c8ff87fd5ee5c9587d0c937e9bfd3193e1a1621141aa367848a9610b3291fa6f"
_INDEX_SHA256 = "c84d2b369f5d5023d0f2d183fc36a935a3981751414996243b65f069983e43d8"
_MLX_CORE_SHA256 = "d7bd29fc20b4a08318d21161c3dfb340889cc9454c5e554ad749eb0127cfa2d6"
_MLX_LIB_SHA256 = "2ee6fbd32ff22e22e1301ebe3c3bece95584104ff9cbc900513d41a095211bbd"
_LOCK_REQUESTED = "/tmp/mtplx-gpu-exclusive.lock"
_LOCK_RESOLVED = str(Path(_LOCK_REQUESTED).resolve())
_CONTRACT = {
    "prompt_tokens": 328,
    "max_tokens": 256,
    "depths": [3],
    "verify_strategy": "capture_commit",
    "verify_core": "stock",
    "mtp_history_policy": "committed",
}
_ARTIFACT = {
    "config_sha256": _CONFIG_SHA256,
    "index_sha256": _INDEX_SHA256,
    "model_type": "deepseek_v4",
    "num_hidden_layers": 43,
    "num_nextn_predict_layers": 1,
    "body_q2_routed_projections": 129,
    "body_q2_manifest_tensors": 387,
    "mtp_manifest_tensors": 35,
    "index_weight_count": 2645,
}
_LOADED = {
    "runtime_mtp_enabled": True,
    "body_layers_loaded": 43,
    "mtp_blocks_bound": 1,
    "body_q2_routed_projections": 129,
    "body_q2_weight_dtype": "uint32",
    "mtp_mxfp4_routed_projections": 3,
    "mtp_routed_weight_dtype": "uint32",
}
_TAIL_REPORT = {
    "route": "decode_verify_m4",
    "body_layers_installed": 43,
    "mtp_layers_stock": 1,
    "verify_rows": 4,
    "repair_rows": 1,
    "topk": 6,
    "hidden_size": 4096,
    "kernel_selfcheck_exact": True,
}
_STOCK_ROUTE_CENSUS = {
    "body_candidate": 0,
    "body_stock": 43,
    "body_other": 0,
    "mtp_stock": 1,
    "mtp_other": 0,
}
_CANDIDATE_ROUTE_CENSUS = {
    "body_candidate": 43,
    "body_stock": 0,
    "body_other": 0,
    "mtp_stock": 1,
    "mtp_other": 0,
}
_COUNTERS = (
    "accepted_by_depth",
    "drafted_by_depth",
    "accepted_drafts",
    "rejected_drafts",
    "drafted_tokens",
    "skipped_drafts",
    "bonus_tokens",
    "correction_tokens",
    "verify_calls",
    "mtp_forward_calls",
    "make_mtp_cache_calls",
    "update_mtp_cache_calls",
    "mtp_history_append_calls",
    "forward_ar_hidden_calls",
    "forward_ar_plain_calls",
)
_WINDOW_KEYS = {
    "schema_version",
    "kind",
    "verified",
    "verified_monotonic_ns",
    "window_id",
    "attestation",
    "lock_identity",
}


def _stage4_env(_candidate: bool) -> dict[str, str]:
    return {
        "MTPLX_COMPILED_VERIFY": "off",
        "MTPLX_DSV4_ATTN": "fused",
        "MTPLX_DSV4_FP32_ACTIVATIONS": "0",
        "MTPLX_DSV4_HC_COMPILE": "1",
        "MTPLX_DSV4_MOE_TAIL": "1",
        "MTPLX_DSV4_O_LORA": "cached",
        "MTPLX_DSV4_SINKHORN_KERNEL": "1",
    }


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _guard_errors(window: Any, label: str) -> list[str]:
    prefix = f"{label}.guard_window"
    if not isinstance(window, dict):
        return [f"{prefix} is absent or not an object"]
    if set(window) != _WINDOW_KEYS | {
        "receipt_path",
        "receipt_sha256",
        "consumer_verification",
    }:
        return [f"{prefix} has an unexpected shape"]
    document = {key: window[key] for key in _WINDOW_KEYS}
    attestation = document.get("attestation")
    if not isinstance(attestation, dict):
        return [f"{prefix}.attestation is absent or not an object"]
    lock_identity = document.get("lock_identity")
    consumer = window.get("consumer_verification")
    if not isinstance(lock_identity, dict):
        return [f"{prefix}.lock_identity is absent or not an object"]
    if not isinstance(consumer, dict):
        return [f"{prefix}.consumer_verification is absent or not an object"]
    errors = []
    integers = (
        "guard_pid",
        "child_pid",
        "issued_monotonic_ns",
        "expires_monotonic_ns",
        "lock_device",
        "lock_inode",
    )
    if document.get("schema_version") != 1:
        errors.append(f"{prefix}.schema_version is not 1")
    if document.get("kind") != "mtplx_verified_guard_window":
        errors.append(f"{prefix}.kind is invalid")
    if document.get("verified") is not True:
        errors.append(f"{prefix} is not verified")
    if attestation.get("schema_version") != 1:
        errors.append(f"{prefix}.attestation schema is invalid")
    if any(
        isinstance(attestation.get(key), bool)
        or not isinstance(attestation.get(key), int)
        for key in integers
    ):
        errors.append(f"{prefix}.attestation integer identity is malformed")
    else:
        issued = attestation["issued_monotonic_ns"]
        expires = attestation["expires_monotonic_ns"]
        verified = document.get("verified_monotonic_ns")
        if (
            isinstance(verified, bool)
            or not isinstance(verified, int)
            or not issued <= verified <= expires
            or expires - issued > 60_000_000_000
        ):
            errors.append(f"{prefix} verification is outside the attestation expiry")
    attested_path = attestation.get("lock_path")
    try:
        attested_resolved = str(Path(attested_path).resolve())
    except TypeError:
        attested_resolved = None
    if attested_resolved != _LOCK_RESOLVED:
        errors.append(f"{prefix} did not attest the canonical GPU lock realpath")
    expected_lock_identity = {
        "requested_path": _LOCK_REQUESTED,
        "resolved_path": _LOCK_RESOLVED,
        "device": attestation.get("lock_device"),
        "inode": attestation.get("lock_inode"),
    }
    if lock_identity != expected_lock_identity:
        errors.append(f"{prefix} lock requested/resolved path or device/inode is invalid")
    if not _valid_sha256(attestation.get("nonce_sha256")):
        errors.append(f"{prefix} nonce digest is malformed")
    if document.get("window_id") != _canonical_digest(attestation):
        errors.append(f"{prefix}.window_id does not bind the attestation")
    receipt_path = window.get("receipt_path")
    if not isinstance(receipt_path, str) or not Path(receipt_path).is_absolute():
        errors.append(f"{prefix}.receipt_path is not absolute")
    if (
        not _valid_sha256(window.get("receipt_sha256"))
        or window.get("receipt_sha256") != _canonical_digest(document)
    ):
        errors.append(f"{prefix}.receipt_sha256 does not bind the document")
    ancestry = consumer.get("ancestry")
    child_pid = attestation.get("child_pid")
    guard_pid = attestation.get("guard_pid")
    consumer_pid = consumer.get("consumer_pid")
    if (
        not isinstance(ancestry, list)
        or not ancestry
        or any(isinstance(pid, bool) or not isinstance(pid, int) for pid in ancestry)
        or isinstance(consumer_pid, bool)
        or not isinstance(consumer_pid, int)
        or ancestry[0] != consumer_pid
        or child_pid not in ancestry
        or guard_pid not in ancestry
    ):
        errors.append(f"{prefix} consumer ancestry is invalid")
    else:
        child_index = ancestry.index(child_pid)
        guard_index = ancestry.index(guard_pid)
        if (
            child_index != consumer.get("child_pid_index")
            or guard_index != consumer.get("guard_pid_index")
            or guard_index <= child_index
        ):
            errors.append(f"{prefix} consumer ancestry ordering is invalid")
    if consumer.get("lock_held") is not True:
        errors.append(f"{prefix} did not observe the GPU lock held")
    if (
        consumer.get("observed_lock_device") != attestation.get("lock_device")
        or consumer.get("observed_lock_inode") != attestation.get("lock_inode")
    ):
        errors.append(f"{prefix} observed lock device/inode differs from attestation")
    return errors


def _identity_errors(actual: Any, expected: dict, prefix: str) -> list[str]:
    if not isinstance(actual, dict):
        return [f"{prefix} is absent or not an object"]
    return [
        f"{prefix}.{key}={actual.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if actual.get(key) != value
    ]


def _receipt_errors(
    receipt: dict[str, Any], label: str, *, candidate: bool, role: str
) -> list[str]:
    errors = []
    for key, expected in _CONTRACT.items():
        if receipt.get(key) != expected:
            errors.append(f"{label}.{key}={receipt.get(key)!r}, expected {expected!r}")
    for key, expected in (
        ("status", 0),
        ("model_path", _MODEL_PATH),
        ("model_type", "deepseek_v4"),
        ("num_hidden_layers", 43),
        ("num_nextn_predict_layers", 1),
        ("receipt_role", role),
        ("performance_eligible", role == "measurement"),
    ):
        if receipt.get(key) != expected:
            errors.append(f"{label}.{key}={receipt.get(key)!r}, expected {expected!r}")
    source_commit = receipt.get("source_commit")
    if not (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    ):
        errors.append(f"{label}.source_commit is absent or malformed")
    host = receipt.get("host") or {}
    if host.get("mlx_version") != "0.31.2":
        errors.append(f"{label}.host.mlx_version is not official 0.31.2")
    errors.extend(
        _identity_errors(
            receipt.get("mlx_identity"),
            {
                "version": "0.31.2",
                "core_sha256": _MLX_CORE_SHA256,
                "lib_sha256": _MLX_LIB_SHA256,
            },
            f"{label}.mlx_identity",
        )
    )
    errors.extend(
        _identity_errors(
            receipt.get("artifact_identity"), _ARTIFACT, f"{label}.artifact_identity"
        )
    )
    errors.extend(
        _identity_errors(
            receipt.get("loaded_runtime_identity"),
            _LOADED,
            f"{label}.loaded_runtime_identity",
        )
    )
    if receipt.get("prompt_file") != _PROMPT_PATH:
        errors.append(f"{label}.prompt_file is not canonical")
    errors.extend(
        _identity_errors(
            receipt.get("prompt"),
            {"path": _PROMPT_PATH, "sha256": _PROMPT_SHA256, "tokens": 328},
            f"{label}.prompt",
        )
    )
    expected_env = _stage4_env(candidate)
    if receipt.get("launch_mtplx_env") != expected_env:
        errors.append(
            f"{label}.launch_mtplx_env={receipt.get('launch_mtplx_env')!r}, "
            f"expected {expected_env!r}"
        )
    return errors


def _measured_arm(
    receipt: dict[str, Any], label: str, depth: int | None
) -> tuple[dict[str, Any] | None, list[str]]:
    arm_name = "AR" if depth is None else f"K{depth}"
    raw_arms = receipt.get("arms")
    if not isinstance(raw_arms, list) or not all(
        isinstance(arm, dict) for arm in raw_arms
    ):
        return None, [f"{label} arms are absent or malformed"]
    arms = [
        arm
        for arm in raw_arms
        if arm.get("speculative_depth") == depth
    ]
    if len(arms) != 1:
        return None, [f"{label} must contain exactly one {arm_name} arm; found {len(arms)}"]
    arm = arms[0]
    errors = []
    if arm.get("error"):
        errors.append(f"{label}.{arm_name} reported error: {arm['error']}")
    tokens = arm.get("tokens")
    if (
        arm.get("generated_tokens") != 256
        or not isinstance(tokens, list)
        or len(tokens) != 256
        or not all(isinstance(token, int) and not isinstance(token, bool) for token in tokens)
    ):
        errors.append(f"{label}.{arm_name} did not persist exactly 256 integer tokens")
    stats = arm.get("stats")
    if not isinstance(stats, dict):
        errors.append(f"{label}.{arm_name} stats are absent")
    else:
        missing = [key for key in _COUNTERS if key not in stats]
        if missing:
            errors.append(f"{label}.{arm_name} stats missing counters {missing}")
        drafted = stats.get("drafted_by_depth")
        if depth == 3 and (
            not isinstance(drafted, list)
            or len(drafted) < 3
            or drafted[2] <= 0
            or stats.get("verify_calls", 0) <= 0
        ):
            errors.append(f"{label}.K3 did not execute the physical M4 target workload")
    return arm, errors


def validate_moe_tail_k3_bracket(
    primer: dict[str, Any],
    before: dict[str, Any],
    candidate: dict[str, Any],
    after: dict[str, Any],
    *,
    peak_ceiling_gib: float,
    live_guard_window: dict[str, Any] | None = None,
) -> dict[str, Any]:
    receipts = {
        "primer": primer,
        "C0": before,
        "candidate": candidate,
        "C1": after,
    }
    errors: list[str] = []
    for label, receipt in receipts.items():
        errors.extend(
            _receipt_errors(
                receipt,
                label,
                candidate=label == "candidate",
                role=(
                    "discarded_control_primer"
                    if label == "primer"
                    else "measurement"
                ),
            )
        )
        errors.extend(_guard_errors(receipt.get("guard_window"), label))
        require_exact = receipt.get("require_exact")
        fp32_activations = receipt.get("fp32_activations")
        reported_enforcement = receipt.get("spec_equals_ar_enforced")
        if not isinstance(require_exact, bool) or not isinstance(fp32_activations, bool):
            errors.append(f"{label} exactness configuration is absent or malformed")
        exact_enforced = require_exact is True or fp32_activations is True
        if reported_enforcement is not exact_enforced:
            errors.append(f"{label} exactness enforcement receipt is inconsistent")
        if exact_enforced:
            raw_arms = receipt.get("arms")
            k3_arms = (
                [
                    arm
                    for arm in raw_arms
                    if isinstance(arm, dict) and arm.get("speculative_depth") == 3
                ]
                if isinstance(raw_arms, list)
                else []
            )
            exact_gate = (
                k3_arms[0].get("spec_equals_ar") if len(k3_arms) == 1 else None
            )
            if (
                not isinstance(exact_gate, dict)
                or exact_gate.get("enforced") is not True
                or exact_gate.get("pass") is not True
            ):
                errors.append(f"{label}.K3 enforced exactness gate failed or is missing")
    windows = [receipt.get("guard_window") for receipt in receipts.values()]
    same_guard = all(window == windows[0] for window in windows[1:])
    if not same_guard:
        errors.append("guard window differs across primer/C0/candidate/C1")
    live_guard_errors: list[str] = []
    if live_guard_window is not None:
        live_guard_errors.extend(_guard_errors(live_guard_window, "validator_live"))
        static_keys = _WINDOW_KEYS | {"receipt_path", "receipt_sha256"}
        reference_window = windows[0] if isinstance(windows[0], dict) else {}
        if any(
            live_guard_window.get(key) != reference_window.get(key)
            for key in static_keys
        ):
            live_guard_errors.append(
                "validator live guard differs from measured guard window"
            )
        errors.extend(live_guard_errors)

    expected_indices = {"primer": 0, "C0": 1, "candidate": 2, "C1": 3}
    process_identities = set()
    for label, receipt in receipts.items():
        single = receipt.get("single_process_bracket")
        if not isinstance(single, dict):
            errors.append(f"{label} has no single process bracket identity")
            continue
        if single.get("model_load_count") != 1:
            errors.append(f"{label} single process bracket did not load exactly once")
        if single.get("execution_order") != ["primer", "C0", "candidate", "C1"]:
            errors.append(f"{label} single process bracket execution order is invalid")
        if single.get("arm_index") != expected_indices[label]:
            errors.append(f"{label} single process bracket arm index is invalid")
        bracket_id = single.get("bracket_id")
        process_pid = single.get("process_pid")
        model_object_id = single.get("model_object_id")
        if (
            not _valid_sha256(bracket_id)
            or isinstance(process_pid, bool)
            or not isinstance(process_pid, int)
            or process_pid <= 0
            or isinstance(model_object_id, bool)
            or not isinstance(model_object_id, int)
            or model_object_id <= 0
        ):
            errors.append(f"{label} single process/model identity is malformed")
        else:
            process_identities.add((bracket_id, process_pid, model_object_id))
        guard = receipt.get("guard_window")
        consumer = guard.get("consumer_verification") if isinstance(guard, dict) else {}
        if not isinstance(consumer, dict):
            consumer = {}
        if consumer.get("consumer_pid") != single.get("process_pid"):
            errors.append(f"{label} single process pid differs from guard consumer")
    if len(process_identities) != 1:
        errors.append("single process/model identity differs across bracket")

    expected_bindings = {
        "primer": {"ar": "stock", "k3": "stock", "post": "stock"},
        "C0": {"ar": "stock", "k3": "stock", "post": "stock"},
        "candidate": {"ar": "stock", "k3": "candidate", "post": "stock"},
        "C1": {"ar": "stock", "k3": "stock", "post": "stock"},
    }
    for label, receipt in receipts.items():
        if receipt.get("route_binding") != expected_bindings[label]:
            errors.append(f"{label} callable route was not reset exactly")
        expected_census = {
            "ar": _STOCK_ROUTE_CENSUS,
            "k3": (
                _CANDIDATE_ROUTE_CENSUS
                if label == "candidate"
                else _STOCK_ROUTE_CENSUS
            ),
            "post": _STOCK_ROUTE_CENSUS,
        }
        if receipt.get("route_census") != expected_census:
            errors.append(f"{label} callable route census is invalid")

    lane_data = {
        "ar": {"tokens": {}, "counters": {}, "peaks": {}, "tps": {}},
        "k3": {"tokens": {}, "counters": {}, "peaks": {}, "tps": {}},
    }
    for label, receipt in receipts.items():
        for lane, depth in (("ar", None), ("k3", 3)):
            arm, arm_errors = _measured_arm(receipt, label, depth)
            errors.extend(arm_errors)
            if arm is None:
                continue
            persisted = arm.get("tokens")
            if isinstance(persisted, list):
                lane_data[lane]["tokens"][label] = hashlib.sha256(
                    json.dumps(persisted, separators=(",", ":")).encode()
                ).hexdigest()
            stats = arm.get("stats")
            if isinstance(stats, dict) and all(key in stats for key in _COUNTERS):
                lane_data[lane]["counters"][label] = {
                    key: stats[key] for key in _COUNTERS
                }
            arm_name = "AR" if lane == "ar" else "K3"
            try:
                peak = float(arm["peak_gib"])
                lane_data[lane]["peaks"][label] = peak
                if not 0.0 < peak < peak_ceiling_gib:
                    errors.append(
                        f"{label}.{arm_name} peak_gib={peak:g} is outside "
                        f"(0, {peak_ceiling_gib:g})"
                    )
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label}.{arm_name} peak_gib is invalid")
            try:
                tps = float(arm["decode_tokens_per_second"])
                if tps <= 0:
                    raise ValueError
                lane_data[lane]["tps"][label] = tps
            except (KeyError, TypeError, ValueError):
                errors.append(f"{label}.{arm_name} decode_tokens_per_second is invalid")

    equality = {}
    for lane, shown in (("ar", "AR"), ("k3", "K3")):
        digests = lane_data[lane]["tokens"]
        counters = lane_data[lane]["counters"]
        token_equal = len(digests) == 4 and len(set(digests.values())) == 1
        counter_values = list(counters.values())
        counter_equal = len(counter_values) == 4 and all(
            value == counter_values[0] for value in counter_values[1:]
        )
        equality[lane] = {"tokens": token_equal, "counters": counter_equal}
        if not token_equal:
            errors.append(f"{shown} token digest differs across primer/C0/candidate/C1")
        if not counter_equal:
            errors.append(f"{shown} counters differ across primer/C0/candidate/C1")

    if candidate.get("deepseek_v4_moe_tail") != _TAIL_REPORT:
        errors.append("candidate has no valid MoE-tail installation report")
    for label in ("primer", "C0", "C1"):
        if receipts[label].get("deepseek_v4_moe_tail") is not None:
            errors.append(f"{label} control is not stock: MoE-tail report is present")
    commits = {receipt.get("source_commit") for receipt in receipts.values()}
    if len(commits) != 1 or None in commits:
        errors.append("source_commit differs across primer/C0/candidate/C1")

    def comparison(values: dict[str, float]) -> tuple[float | None, float | None, float | None]:
        if not all(label in values for label in ("C0", "candidate", "C1")):
            return None, None, None
        mean = (values["C0"] + values["C1"]) / 2.0
        if mean <= 0:
            return None, None, None
        drift = abs(values["C1"] - values["C0"]) / mean
        delta = (values["candidate"] - mean) / mean
        return mean, drift, delta

    k3_mean, k3_drift, k3_delta = comparison(lane_data["k3"]["tps"])
    ar_mean, ar_drift, ar_delta = comparison(lane_data["ar"]["tps"])
    k3_pass = (
        k3_drift is not None and k3_delta is not None and k3_delta > k3_drift
    )
    ar_regression = None if ar_delta is None else max(0.0, -ar_delta)
    ar_pass = (
        ar_drift is not None
        and ar_regression is not None
        and ar_regression <= ar_drift
    )
    performance_pass = k3_pass and ar_pass
    integrity_pass = not errors
    status = (
        "INVALID_BRACKET"
        if not integrity_pass
        else "PASS"
        if performance_pass
        else "LOSS"
    )
    return {
        "schema_version": 1,
        "kind": "deepseek_v4_moe_tail_k3_bracket",
        "status": status,
        "integrity_pass": integrity_pass,
        "performance_pass": performance_pass if integrity_pass else False,
        "errors": errors,
        "peak_ceiling_gib": peak_ceiling_gib,
        "tokens": {
            "digests": lane_data["k3"]["tokens"],
            "all_equal": equality["ar"]["tokens"] and equality["k3"]["tokens"],
            "ar": lane_data["ar"]["tokens"],
            "k3": lane_data["k3"]["tokens"],
        },
        "counters": {
            "values": lane_data["k3"]["counters"],
            "all_equal": equality["ar"]["counters"]
            and equality["k3"]["counters"],
            "ar": lane_data["ar"]["counters"],
            "k3": lane_data["k3"]["counters"],
        },
        "peak_gib": {
            "ar": lane_data["ar"]["peaks"],
            "k3": lane_data["k3"]["peaks"],
        },
        "guard_window": {
            "window_id": (
                primer.get("guard_window", {}).get("window_id")
                if isinstance(primer.get("guard_window"), dict)
                else None
            ),
            "all_equal_and_valid": same_guard
            and not any("guard_window" in error for error in errors),
            "validator_live_recheck": live_guard_window is not None
            and not live_guard_errors,
        },
        "primer": {
            "receipt_role": primer.get("receipt_role"),
            "performance_data_used": False,
        },
        "ar_tps": lane_data["ar"]["tps"],
        "k3_tps": lane_data["k3"]["tps"],
        "ar_negative_control": {
            "pass": ar_pass,
            "mean_tps": ar_mean,
            "drift_fraction": ar_drift,
            "candidate_delta_fraction": ar_delta,
            "candidate_regression_fraction": ar_regression,
        },
        "k3_performance": {
            "pass": k3_pass,
            "mean_tps": k3_mean,
            "drift_fraction": k3_drift,
            "candidate_delta_fraction": k3_delta,
        },
        "control": {
            "mean_tps": k3_mean,
            "drift_fraction": k3_drift,
            "candidate_delta_fraction": k3_delta,
        },
        "source_commit": next(iter(commits)) if len(commits) == 1 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primer", required=True, type=Path)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--peak-ceiling-gib", type=float, default=108.0)
    parser.add_argument("--require-live-guard", action="store_true")
    parser.add_argument("--benchmark-exit-code", type=int, default=0)
    args = parser.parse_args()
    if args.peak_ceiling_gib <= 0:
        parser.error("--peak-ceiling-gib must be positive")
    if args.benchmark_exit_code < 0:
        parser.error("--benchmark-exit-code must be nonnegative")
    receipt_paths = {
        "primer": args.primer,
        "before": args.before,
        "candidate": args.candidate,
        "after": args.after,
    }
    errors: list[str] = []
    live_guard_window = None
    if args.require_live_guard:
        from deepseek_v4_guard_window import load_verified_guard_window

        try:
            live_guard_window = load_verified_guard_window()
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"validator live guard verification failed: {error}")
    receipts: dict[str, dict[str, Any]] = {}
    for label, path in receipt_paths.items():
        if not path.is_file():
            continue
        try:
            receipt = json.loads(path.read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            errors.append(f"{label} benchmark receipt is unreadable: {error}")
            continue
        if not isinstance(receipt, dict):
            errors.append(f"{label} benchmark receipt is not an object")
            continue
        receipts[label] = receipt
    missing = [label for label in receipt_paths if label not in receipts]
    if missing:
        errors.append(f"missing benchmark receipts: {missing}")
        result: dict[str, Any] = {
            "schema_version": 1,
            "kind": "deepseek_v4_moe_tail_k3_bracket",
            "status": "INVALID_BRACKET",
            "integrity_pass": False,
            "performance_pass": False,
            "errors": errors,
            "benchmark_exit_code": args.benchmark_exit_code,
            "receipt_paths": {
                label: str(path) for label, path in receipt_paths.items()
            },
            "receipts_present": sorted(receipts),
            "guard_window": {
                "validator_live_recheck": live_guard_window is not None,
                "window_id": (
                    live_guard_window.get("window_id")
                    if isinstance(live_guard_window, dict)
                    else None
                ),
            },
        }
    else:
        result = validate_moe_tail_k3_bracket(
            receipts["primer"],
            receipts["before"],
            receipts["candidate"],
            receipts["after"],
            peak_ceiling_gib=args.peak_ceiling_gib,
            live_guard_window=live_guard_window,
        )
        result["benchmark_exit_code"] = args.benchmark_exit_code
        if errors:
            result["errors"].extend(errors)
            result["status"] = "INVALID_BRACKET"
            result["integrity_pass"] = False
            result["performance_pass"] = False
    if args.benchmark_exit_code:
        error = f"benchmark aborted with exit code {args.benchmark_exit_code}"
        if error not in result["errors"]:
            result["errors"].append(error)
        result["status"] = "INVALID_BRACKET"
        result["integrity_pass"] = False
        result["performance_pass"] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2 if result["status"] == "INVALID_BRACKET" else 1


if __name__ == "__main__":
    raise SystemExit(main())
