#!/usr/bin/env python3
"""Run the four-lane Qwen3.8 native-MTP comparison matrix."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any, NamedTuple

from mtplx.qwen38_challenge import (
    QWEN38_LOW_BF16_FEATURE_KEYS,
    QWEN38_LOW_BF16_INSTALLED_ROUTE,
    QWEN38_LOW_BF16_KERNEL_IDS,
    QWEN38_LOW_Q4_INSTALLED_ROUTE,
    QWEN38_XHIGH_BF16_FEATURE_KEYS,
    QWEN38_XHIGH_BF16_INSTALLED_ROUTE,
    QWEN38_XHIGH_BF16_KERNEL_IDS,
)

try:
    from scripts import qwen38_challenge_port_gate as gate
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    import qwen38_challenge_port_gate as gate  # type: ignore[no-redef]


V292_COMMIT = "bbc67427e88288001e4b90ecb44708dc0222154c"
MODEL_ID = "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed"
LOW_FIXED_NATIVE_ROUTE = gate.LOW_FIXED_NATIVE_ROUTE
LOW_ADAPTIVE_NATIVE_ROUTE = gate.LOW_ADAPTIVE_NATIVE_ROUTE
LOW_Q4_ADAPTIVE_NATIVE_ROUTE = gate.LOW_Q4_ADAPTIVE_NATIVE_ROUTE
XHIGH_FIXED_NATIVE_ROUTE = gate.XHIGH_FIXED_NATIVE_ROUTE
XHIGH_ADAPTIVE_NATIVE_ROUTE = gate.XHIGH_ADAPTIVE_NATIVE_ROUTE
XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE = gate.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE
GREEDY_ADAPTIVE_NATIVE_ROUTE = gate.GREEDY_ADAPTIVE_NATIVE_ROUTE
GREEDY_Q4_ADAPTIVE_NATIVE_ROUTE = gate.GREEDY_Q4_ADAPTIVE_NATIVE_ROUTE

LANE_IDS = (
    "v2.9.2-mlx0322",
    "full-fixed-k3",
    "full-adaptive",
    "full-q4-adaptive",
)
PAIRED_ORDER = (
    "v2.9.2-mlx0322",
    "full-fixed-k3",
    "full-adaptive",
    "full-q4-adaptive",
    "full-q4-adaptive",
    "full-adaptive",
    "full-fixed-k3",
    "v2.9.2-mlx0322",
)
LOW_BF16_OPTIMIZED_KERNEL_IDS = QWEN38_LOW_BF16_KERNEL_IDS
LOW_BF16_OPTIMIZED_FEATURE_KEYS = QWEN38_LOW_BF16_FEATURE_KEYS
LOW_BF16_OPTIMIZED_INSTALLED_ROUTE_ID = QWEN38_LOW_BF16_INSTALLED_ROUTE
LOW_Q4_OPTIMIZED_KERNEL_IDS = (
    "qwen38_row17_q4_g64_mtp_block_v1",
    *LOW_BF16_OPTIMIZED_KERNEL_IDS,
)
LOW_Q4_OPTIMIZED_INSTALLED_ROUTE_ID = QWEN38_LOW_Q4_INSTALLED_ROUTE
XHIGH_BF16_OPTIMIZED_KERNEL_IDS = QWEN38_XHIGH_BF16_KERNEL_IDS
XHIGH_BF16_OPTIMIZED_FEATURE_KEYS = QWEN38_XHIGH_BF16_FEATURE_KEYS
XHIGH_BF16_OPTIMIZED_INSTALLED_ROUTE_ID = QWEN38_XHIGH_BF16_INSTALLED_ROUTE
XHIGH_Q4_OPTIMIZED_KERNEL_IDS = (
    "qwen38_row17_q4_g64_mtp_block_v1",
    *XHIGH_BF16_OPTIMIZED_KERNEL_IDS,
)
XHIGH_Q4_OPTIMIZED_INSTALLED_ROUTE_ID = (
    "r17_q4_mtp_block+" + XHIGH_BF16_OPTIMIZED_INSTALLED_ROUTE_ID
)
ONE_PASS_ORDER = LANE_IDS
CONTEXT_TOKENS = (1_024, 16_384, 65_536, 131_072)
CONDITIONER_OUTPUT_TOKENS = 1_024
LOW_OUTPUT_TOKENS = 1_024
XHIGH_OUTPUT_TOKENS = 1_024
VANITY_PROMPT_TOKENS = 100
VANITY_TEMPERATURE = 0.0
REQUIRED_MLX_VERSION = "0.32.2"
REQUIRED_MLX_METAL_VERSION = "0.32.2"
MODEL_HASHES_ENV = gate.MODEL_ARTIFACT_HASHES_ENV
ROOT = Path(__file__).resolve().parents[1]
ARM_SCRIPT = ROOT / "scripts/qwen38_native_mtp_matrix_arm.py"
ISOLATED_SCRIPT = ROOT / "scripts/qwen38_challenge_port_isolated_gate.py"
VANITY_PROMPT_FILE = ROOT / "mtplx/benchmarks/prompts/qwen38_palindrome_vanity.jsonl"
PYTHON_PROMPT_FILE = ROOT / "mtplx/benchmarks/prompts/python_modules_long.jsonl"
PYTHON_CONTEXT_MANIFEST = (
    ROOT / "benchmarks/fixtures/qwen38-pr335-python-context.json"
)
PROMPT_ARTIFACT_SHA256 = {
    "vanity": "878a98fe36e5d62566b093b77d11d11bd502fb31e6d2caf7309ea71a9a79bb02",
    "python": "ca2054913c5c27c24c983ed27e3ee4eff1d01d456a73e71377fdaea3cbf8c140",
}
PYTHON_CONTEXT_SHA256 = "dfa72b4d7b161ef6f6105b0a635cff3bf3d112f37bdb4f0d1a4eb092ccbf5771"
ROW17_ARTIFACT_SHA256 = "0e267a482e74c2664ce41dc4c4326f480020d015372fc9f7654ea3a136d62815"
PROMPT_TOKEN_SHA256 = {
    "vanity": {
        100: "94a188b7cacc378c60a6503feea97429c59f6dab3980635eaa5e35da1e6b767b",
    },
    "low": {
        1_024: "5e19fc124e7743301a25b1f70de3f7010fd5ecab8236a680b1d73c86652de696",
        16_384: "c453849d0e60ca98d2f326c0cc5594c601630e17e3b2e31b830bc83182a2ff59",
        65_536: "779e065a4d932f882c34ac513cdca9212ee6c2ebf61ab2e4283b3f5ab22fa1cf",
        131_072: "f34bdb262905ae262cd4f044a0cafa5222411ded79c6615cbf9be0266429dc2c",
    },
    "xhigh": {
        1_024: "806039fe48f994aad16eda98fee9e9b64388697797c7fd620bd55eb28c2a53e9",
        16_384: "d78595989ba63e067242f446c47f7b09682c00843376def66015add88d50b18e",
        65_536: "9b67a17554da6586120595d724b87b83b0cfad45214c19140d9bb499f5261a4a",
        131_072: "a9fb510a6d62511650f9dc5c0a98d72120131760644cab7835ad53cfcc73a07a",
    },
}


class LaneSpec(NamedTuple):
    lane_id: str
    source_root: Path
    source_commit: str
    route_id: str


def lane_specs(
    *,
    baseline_root: Path,
    baseline_commit: str,
    candidate_root: Path,
    candidate_commit: str,
    workload: str,
) -> dict[str, LaneSpec]:
    if workload not in {"vanity", "low", "xhigh"}:
        raise ValueError(f"unknown workload: {workload}")
    if workload in {"vanity", "xhigh"}:
        fixed_route = XHIGH_FIXED_NATIVE_ROUTE
        adaptive_route = XHIGH_ADAPTIVE_NATIVE_ROUTE
        q4_route = XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE
    else:
        fixed_route = LOW_FIXED_NATIVE_ROUTE
        adaptive_route = LOW_ADAPTIVE_NATIVE_ROUTE
        q4_route = LOW_Q4_ADAPTIVE_NATIVE_ROUTE
    return {
        "v2.9.2-mlx0322": LaneSpec(
            "v2.9.2-mlx0322", baseline_root, baseline_commit, "control"
        ),
        "full-fixed-k3": LaneSpec(
            "full-fixed-k3",
            candidate_root,
            candidate_commit,
            fixed_route,
        ),
        "full-adaptive": LaneSpec(
            "full-adaptive",
            candidate_root,
            candidate_commit,
            adaptive_route,
        ),
        "full-q4-adaptive": LaneSpec(
            "full-q4-adaptive",
            candidate_root,
            candidate_commit,
            q4_route,
        ),
    }


def _selected_lanes(lane_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not lane_ids:
        raise ValueError("at least one benchmark lane is required")
    if len(set(lane_ids)) != len(lane_ids):
        raise ValueError("benchmark lanes must be unique")
    unknown = tuple(lane_id for lane_id in lane_ids if lane_id not in LANE_IDS)
    if unknown:
        raise ValueError(f"unknown benchmark lanes: {unknown}")
    return lane_ids


def _paired_order(lane_ids: tuple[str, ...]) -> tuple[str, ...]:
    lanes = _selected_lanes(lane_ids)
    return (*lanes, *reversed(lanes))


def order_for_context(
    context_tokens: int,
    lane_ids: tuple[str, ...] = LANE_IDS,
) -> tuple[str, ...]:
    if context_tokens not in CONTEXT_TOKENS:
        raise ValueError(f"unsupported context size: {context_tokens}")
    lanes = _selected_lanes(lane_ids)
    return lanes if context_tokens == 131_072 else _paired_order(lanes)


def _workload_values(workload: str) -> tuple[int, float, float, int]:
    if workload == "low":
        return LOW_OUTPUT_TOKENS, 1.0, 0.95, 20
    if workload == "xhigh":
        return XHIGH_OUTPUT_TOKENS, 1.0, 0.95, 20
    if workload == "vanity":
        return LOW_OUTPUT_TOKENS, VANITY_TEMPERATURE, 1.0, 0
    raise ValueError(f"unknown workload: {workload}")


def child_command(
    *,
    lane: LaneSpec,
    workload: str,
    context_tokens: int,
    output: Path,
    model: Path,
    prompt_file: Path,
    context_file: Path,
    row17_artifact: Path,
    python: Path,
    lock: Path,
) -> list[str]:
    output_tokens, temperature, top_p, top_k = _workload_values(workload)
    conditioner_tokens = 0 if workload == "vanity" else CONDITIONER_OUTPUT_TOKENS
    command = [
        str(python.absolute()),
        str(ARM_SCRIPT),
        "--source-root", str(lane.source_root.resolve()),
        "--source-commit", lane.source_commit,
        "--lane-id", lane.lane_id,
        "--route", lane.route_id,
        "--workload", workload,
        "--model", str(model.resolve()),
        "--prompt-file", str(prompt_file.resolve()),
        "--context-file", str(context_file.resolve()),
        "--prompt-tokens", str(context_tokens),
        "--max-tokens", str(output_tokens),
        "--warmup-tokens", str(conditioner_tokens),
        "--seed", "42",
        "--target-temperature", str(temperature),
        "--draft-temperature", str(temperature),
        "--top-p", str(top_p),
        "--top-k", str(top_k),
        "--row17-artifact", str(row17_artifact.resolve()),
        "--lock", str(lock.resolve()),
        "--output", str(output.resolve()),
    ]
    route_features = gate._validate_route_id(lane.route_id)
    if workload != "vanity":
        command.append("--force-exact-output")
    if context_tokens == 131_072 and "r11_position_ema" in route_features:
        command.append("--record-depth-usage")
    return command


def _feature_is_active(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    active = value.get("active")
    if (
        active is True
        or (
            isinstance(active, (int, float))
            and not isinstance(active, bool)
            and active > 0
        )
        or value.get("installed") is True
    ):
        return True
    return any(
        int(value.get(key, 0)) > 0
        for key in ("active_modules", "configured_modules", "construction_bound")
    )


def _optimized_route_contract(expected_route: str) -> dict[str, Any]:
    if expected_route in {
        LOW_FIXED_NATIVE_ROUTE,
        LOW_ADAPTIVE_NATIVE_ROUTE,
        LOW_Q4_ADAPTIVE_NATIVE_ROUTE,
    }:
        profile = "low"
        fixed_route = LOW_FIXED_NATIVE_ROUTE
        bf16_kernels = LOW_BF16_OPTIMIZED_KERNEL_IDS
        bf16_features = LOW_BF16_OPTIMIZED_FEATURE_KEYS
        bf16_installed = LOW_BF16_OPTIMIZED_INSTALLED_ROUTE_ID
        q4_kernels = LOW_Q4_OPTIMIZED_KERNEL_IDS
        q4_installed = LOW_Q4_OPTIMIZED_INSTALLED_ROUTE_ID
        prefill_only = False
    elif expected_route in {
        XHIGH_FIXED_NATIVE_ROUTE,
        XHIGH_ADAPTIVE_NATIVE_ROUTE,
        XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE,
    }:
        profile = "xhigh"
        fixed_route = XHIGH_FIXED_NATIVE_ROUTE
        bf16_kernels = XHIGH_BF16_OPTIMIZED_KERNEL_IDS
        bf16_features = XHIGH_BF16_OPTIMIZED_FEATURE_KEYS
        bf16_installed = XHIGH_BF16_OPTIMIZED_INSTALLED_ROUTE_ID
        q4_kernels = XHIGH_Q4_OPTIMIZED_KERNEL_IDS
        q4_installed = XHIGH_Q4_OPTIMIZED_INSTALLED_ROUTE_ID
        prefill_only = True
    else:
        raise ValueError(f"route is not a final optimized profile: {expected_route}")
    uses_q4 = "r17_q4_mtp_block" in gate._validate_route_id(expected_route)
    return {
        "profile": profile,
        "fixed_route": fixed_route,
        "uses_q4": uses_q4,
        "kernel_ids": q4_kernels if uses_q4 else bf16_kernels,
        "feature_keys": (
            (*bf16_features, "r17_q4_mtp_block")
            if uses_q4
            else bf16_features
        ),
        "installed_route_id": q4_installed if uses_q4 else bf16_installed,
        "prefill_only": prefill_only,
    }


def full_fixed_receipt_errors(
    receipt: dict[str, Any], *, expected_route: str
) -> list[str]:
    errors: list[str] = []
    contract = _optimized_route_contract(expected_route)
    if expected_route != contract["fixed_route"]:
        errors.append("optimized fixed K3 route is not the profile's fixed route")
    if receipt.get("route_id") != expected_route or expected_route == "control":
        errors.append("optimized fixed K3 route is not the requested optimized route")
    installed_route = receipt.get("installed_route_id")
    if installed_route != contract["installed_route_id"]:
        errors.append("optimized fixed K3 installed route mismatch")
    if receipt.get("performance_profile") != contract["profile"]:
        errors.append("optimized fixed K3 performance profile mismatch")
    if int(receipt.get("speculative_depth", -1)) != 3 or int(
        receipt.get("requested_speculative_depth", -1)
    ) != 3:
        errors.append("optimized fixed K3 speculative depth is not fixed at 3")
    if receipt.get("adaptive_policy_receipt") is not None or receipt.get(
        "adaptive_policy_events"
    ):
        errors.append("optimized fixed K3 executed an adaptive policy")

    kernel_ids = tuple(receipt.get("kernel_ids") or ())
    if kernel_ids != contract["kernel_ids"]:
        errors.append("optimized fixed K3 BF16 kernel stack mismatch")
    features = receipt.get("feature_receipt") or {}
    if set(features) != set(contract["feature_keys"]):
        errors.append("optimized fixed K3 feature receipt mismatch")
    if any(
        not _feature_is_active(features.get(key))
        for key in contract["feature_keys"]
    ):
        errors.append("optimized fixed K3 feature stack is incomplete")
    row26 = features.get("r26_prefill_ladder_3") or {}
    if contract["prefill_only"] and (
        row26.get("phase_scope") != "prefill" or row26.get("decode_route") != "stock"
    ):
        errors.append("optimized fixed K3 target phase route mismatch")
    if any(key in features for key in ("r17_q4_mtp_block", "r28_q4_mtp_block", "r36_qkv_islands")):
        errors.append("optimized fixed K3 installed a Q4 MTP block")

    device_core = receipt.get("device_core_receipt") or {}
    if not (
        receipt.get("draft_core")
        == gate._route_execution_options(expected_route)["draft_core"]
        and device_core.get("requested")
        == gate._route_execution_options(expected_route)["draft_core"]
        and (
            int(device_core.get("device_calls", 0)) > 0
            if receipt.get("draft_core") == "device"
            else int(device_core.get("device_calls", -1)) == 0
        )
    ):
        errors.append("optimized fixed K3 draft core did not remain selected")
    if int(device_core.get("device_fallbacks", -1)) != 0:
        errors.append("optimized fixed K3 stock draft fallback occurred")
    if receipt.get("fallback_ar") is True:
        errors.append("optimized fixed K3 fell back to autoregressive decode")

    compiled_verify = receipt.get("compiled_verify_receipt") or {}
    exception_fallbacks = sum(
        int(value)
        for reason, value in (compiled_verify.get("fallback_reasons") or {}).items()
        if str(reason).startswith("exception:")
    )
    if not (
        compiled_verify.get("mode") == "on"
        and int(compiled_verify.get("compiled_calls", 0)) > 0
        and compiled_verify.get("permanent_eager") is False
        and not compiled_verify.get("permanent_eager_reason")
        and exception_fallbacks == 0
    ):
        errors.append("optimized fixed K3 compiled verification did not engage cleanly")
    return errors


def adaptive_optimized_receipt_errors(
    receipt: dict[str, Any], *, expected_route: str
) -> list[str]:
    errors: list[str] = []
    features = gate._validate_route_id(expected_route)
    if "r11_position_ema" not in features or receipt.get("route_id") != expected_route:
        errors.append("adaptive route is not the requested optimized route")
        return errors

    contract = _optimized_route_contract(expected_route)
    uses_q4 = bool(contract["uses_q4"])
    if receipt.get("performance_profile") != contract["profile"]:
        errors.append("adaptive performance profile mismatch")
    if receipt.get("installed_route_id") != contract["installed_route_id"]:
        errors.append(
            f"adaptive {'Q4' if uses_q4 else 'BF16'} installed route mismatch"
        )

    if tuple(receipt.get("kernel_ids") or ()) != contract["kernel_ids"]:
        errors.append(f"adaptive {'Q4' if uses_q4 else 'BF16'} kernel stack mismatch")

    feature_receipt = receipt.get("feature_receipt") or {}
    if set(feature_receipt) != set(contract["feature_keys"]):
        errors.append("adaptive feature receipt mismatch")
    if any(
        not _feature_is_active(feature_receipt.get(key))
        for key in contract["feature_keys"]
    ):
        errors.append("adaptive shared feature stack is incomplete")
    row26 = feature_receipt.get("r26_prefill_ladder_3") or {}
    if contract["prefill_only"] and (
        row26.get("phase_scope") != "prefill" or row26.get("decode_route") != "stock"
    ):
        errors.append("adaptive target phase route mismatch")
    if uses_q4:
        if not _feature_is_active(feature_receipt.get("r17_q4_mtp_block")):
            errors.append("adaptive Q4 MTP block is inactive")
    elif any(
        key in feature_receipt
        for key in ("r17_q4_mtp_block", "r28_q4_mtp_block", "r36_qkv_islands")
    ):
        errors.append("adaptive BF16 unexpectedly installed a Q4 MTP block")
    return errors


def _output_contract_errors(
    receipt: dict[str, Any],
    *,
    output_tokens: int,
) -> list[str]:
    errors: list[str] = []
    workload = receipt.get("workload")
    if workload == "vanity":
        if receipt.get("stop_token_policy") != "tokenizer_default":
            errors.append("vanity stop-token policy is not tokenizer default")
        if receipt.get("finish_reason") != "stop":
            errors.append("vanity arm did not stop naturally")
        generated = int(receipt.get("generated_tokens", -1))
        if not 0 < generated <= output_tokens:
            errors.append("vanity output token count is invalid")
        if int(receipt.get("conditioner_generated_tokens", -1)) != 0:
            errors.append("vanity unexpectedly generated conditioner tokens")
        if receipt.get("conditioner_finish_reason") is not None:
            errors.append("vanity unexpectedly recorded a conditioner finish reason")
        return errors

    if receipt.get("stop_token_policy") != "disabled_for_exact_output":
        errors.append("thinking stop-token policy did not force exact output")
    if int(receipt.get("generated_tokens", -1)) != output_tokens:
        errors.append("timed output token count is not exact")
    if receipt.get("finish_reason") != "length":
        errors.append("exact-output arm did not finish at the requested length")
    if int(receipt.get("conditioner_generated_tokens", -1)) != CONDITIONER_OUTPUT_TOKENS:
        errors.append("conditioner output token count is not exact")
    if receipt.get("conditioner_finish_reason") != "length":
        errors.append("conditioner did not finish at the requested length")
    return errors


def receipt_errors(
    receipt: dict[str, Any],
    *,
    lane: LaneSpec,
    context_tokens: int,
    output_tokens: int,
) -> list[str]:
    errors: list[str] = []
    exact = {
        "lane_id": lane.lane_id,
        "source_commit": lane.source_commit,
        "route_id": lane.route_id,
        "prompt_tokens": context_tokens,
        "conditioner_output_tokens": (
            0 if receipt.get("workload") == "vanity" else CONDITIONER_OUTPUT_TOKENS
        ),
        "max_tokens": output_tokens,
        "mlx_version": REQUIRED_MLX_VERSION,
        "mlx_metal_version": REQUIRED_MLX_METAL_VERSION,
        "gpu_lock_scope": "attested_parent",
        "source_import_attested": True,
        "model_id": MODEL_ID,
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "speculative_depth": 3,
        "requested_speculative_depth": 3,
        "draft_core": str(gate._route_execution_options(lane.route_id)["draft_core"]),
    }
    for key, expected in exact.items():
        if receipt.get(key) != expected:
            errors.append(f"{key} mismatch: {receipt.get(key)!r} != {expected!r}")
    if receipt.get("source_status"):
        errors.append("source tree is not clean")
    model_hashes = receipt.get("model_artifact_hashes") or {}
    if not (
        isinstance(model_hashes, dict)
        and {"config.json", "mtp.safetensors"} <= set(model_hashes)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in model_hashes.values()
        )
    ):
        errors.append("model artifact attestation is missing")
    workload = str(receipt.get("workload") or "")
    expected_prompt_hash = PROMPT_TOKEN_SHA256.get(workload, {}).get(context_tokens)
    if receipt.get("prompt_token_sha256") != expected_prompt_hash:
        errors.append("prompt token hash does not match the frozen workload")
    expected_artifact_hash = PROMPT_ARTIFACT_SHA256[
        "vanity" if workload == "vanity" else "python"
    ]
    if receipt.get("prompt_artifact_sha256") != expected_artifact_hash:
        errors.append("prompt artifact hash does not match the frozen workload")
    if receipt.get("context_artifact_sha256") != PYTHON_CONTEXT_SHA256:
        errors.append("Python context artifact hash does not match the frozen workload")
    if receipt.get("row17_artifact_sha256") != ROW17_ARTIFACT_SHA256:
        errors.append("row17 artifact hash does not match the frozen custom head")
    expected_sampler = {
        "target_temperature": _workload_values(receipt.get("workload", ""))[1],
        "draft_temperature": _workload_values(receipt.get("workload", ""))[1],
        "top_p": _workload_values(receipt.get("workload", ""))[2],
        "top_k": _workload_values(receipt.get("workload", ""))[3],
    }
    if receipt.get("sampler") != expected_sampler:
        errors.append("sampler receipt does not match the frozen workload")
    optimized_stack = receipt.get("optimized_stack") or {}
    expected_stack = {
        "profile": "turbo",
        "runtime_profile": "native_mtp_turbo",
        "draft_lm_head": {"bits": 4, "group_size": 64, "mode": "affine"},
        "draft_sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        "mtp_hidden_variant": "post_norm",
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
    }
    for key, expected in expected_stack.items():
        if optimized_stack.get(key) != expected:
            errors.append(f"optimized stack {key} mismatch")
    route_features = gate._validate_route_id(lane.route_id)
    is_adaptive = "r11_position_ema" in route_features
    is_full_fixed = lane.route_id in {
        LOW_FIXED_NATIVE_ROUTE,
        XHIGH_FIXED_NATIVE_ROUTE,
    }
    records_depth = context_tokens == 131_072 and is_adaptive
    required_runtime_env = {
        "MTPLX_COMPILED_VERIFY": "1",
        "MTPLX_DROP_EVENTS": "0" if records_depth else "1",
        "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
        "MTPLX_MTP_HISTORY_POLICY": "committed",
    }
    runtime_env = optimized_stack.get("runtime_env") or {}
    for key, expected in required_runtime_env.items():
        if runtime_env.get(key) != expected:
            errors.append(f"optimized stack runtime env {key} mismatch")
    errors.extend(_output_contract_errors(receipt, output_tokens=output_tokens))
    if lane.route_id != "control":
        expected_options = gate._route_execution_options(lane.route_id)
        if tuple(receipt.get("source_rows") or ()) != tuple(
            expected_options["source_rows"]
        ):
            errors.append("optimized source-row receipt mismatch")
        if not receipt.get("kernel_ids"):
            errors.append("optimized route reported no installed kernels")
        errors.extend(gate._candidate_engagement_errors(lane.route_id, [], [receipt]))
        expected_draft_core = str(
            gate._route_execution_options(lane.route_id)["draft_core"]
        )
        if is_full_fixed:
            errors.extend(
                full_fixed_receipt_errors(receipt, expected_route=lane.route_id)
            )
        elif is_adaptive:
            errors.extend(
                adaptive_optimized_receipt_errors(
                    receipt, expected_route=lane.route_id
                )
            )
        if records_depth:
            try:
                depth_usage_from_schedules(
                    attempted_depth_schedule=list(
                        receipt["attempted_depth_schedule"]
                    ),
                    accepted_depth_schedule=list(receipt["accepted_depth_schedule"]),
                    verify_calls=int(receipt["verify_calls"]),
                    drafted_by_depth=list(receipt["drafted_by_depth"]),
                    accepted_by_depth=list(receipt["accepted_by_depth"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"adaptive depth usage is invalid: {exc}")
        elif is_adaptive and receipt.get("depth_usage") is not None:
            try:
                expected_usage = depth_usage(
                    decode_cycles=len(receipt["attempted_depth_schedule"]),
                    verify_calls=int(receipt["verify_calls"]),
                    drafted_by_depth=list(receipt["drafted_by_depth"]),
                    accepted_by_depth=list(receipt["accepted_by_depth"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"adaptive depth usage is invalid: {exc}")
            else:
                if receipt.get("depth_usage") != expected_usage:
                    errors.append("adaptive depth usage does not match raw histograms")
        if is_adaptive:
            policy = receipt.get("adaptive_policy_receipt") or {}
            if not (
                policy.get("kind") == "position_ema"
                and policy.get("executed") is True
                and len(policy.get("initial_accept_ema") or ()) == 3
                and len(policy.get("final_accept_ema") or ()) == 3
                and 0 <= int(policy.get("initial_depth", -1)) <= 3
                and 0 <= int(policy.get("final_depth", -1)) <= 3
                and int(policy.get("max_depth", -1)) == 3
                and int(policy.get("depth_cap", -1)) == 3
            ):
                errors.append("adaptive policy state receipt is incomplete")
        device_core = receipt.get("device_core_receipt") or {}
        if expected_draft_core == "device" and not (
            device_core.get("requested") == "device"
            and int(device_core.get("device_calls", 0)) > 0
            and int(device_core.get("device_fallbacks", -1)) == 0
        ):
            errors.append("adaptive device draft core did not engage without fallback")
        if expected_draft_core == "stock" and not (
            device_core.get("requested") == "stock"
            and int(device_core.get("device_calls", -1)) == 0
            and int(device_core.get("device_fallbacks", -1)) == 0
        ):
            errors.append("adaptive stock draft core did not remain selected")
        compiled_verify = receipt.get("compiled_verify_receipt") or {}
        exception_fallbacks = sum(
            int(value)
            for reason, value in (compiled_verify.get("fallback_reasons") or {}).items()
            if str(reason).startswith("exception:")
        )
        if not (
            compiled_verify.get("mode") == "on"
            and int(compiled_verify.get("compiled_calls", 0)) > 0
            and compiled_verify.get("permanent_eager") is False
            and not compiled_verify.get("permanent_eager_reason")
            and exception_fallbacks == 0
        ):
            errors.append("adaptive compiled verification did not engage cleanly")
    elif receipt.get("source_rows") or receipt.get("kernel_ids"):
        errors.append("control route unexpectedly installed candidate features")
    return errors


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"invalid {key} values")
    return statistics.fmean(values)


def depth_usage(
    *,
    decode_cycles: int,
    verify_calls: int,
    drafted_by_depth: list[int],
    accepted_by_depth: list[int],
) -> dict[str, Any]:
    drafted = ([int(value) for value in drafted_by_depth] + [0, 0, 0])[:3]
    accepted = ([int(value) for value in accepted_by_depth] + [0, 0, 0])[:3]
    cycles = int(decode_cycles)
    verified = int(verify_calls)
    if verified != drafted[0]:
        raise ValueError("verify calls contradict attempted MTP depth")
    if not (cycles >= verified >= drafted[0] >= drafted[1] >= drafted[2] >= 0):
        raise ValueError("drafted-depth histogram contradicts generated work")
    if not (
        cycles >= accepted[0] >= accepted[1] >= accepted[2] >= 0
        and all(left <= right for left, right in zip(accepted, drafted))
    ):
        raise ValueError("accepted-depth histogram contradicts drafted work")
    attempted_exact = (
        cycles - verified,
        drafted[0] - drafted[1],
        drafted[1] - drafted[2],
        drafted[2],
    )
    accepted_exact = (
        cycles - accepted[0],
        accepted[0] - accepted[1],
        accepted[1] - accepted[2],
        accepted[2],
    )

    def keyed(values: tuple[int, int, int, int]) -> dict[str, int]:
        return {f"D{depth}": value for depth, value in enumerate(values)}

    def shares(values: tuple[int, int, int, int]) -> dict[str, float]:
        return {
            f"D{depth}": value / cycles * 100.0 if cycles else 0.0
            for depth, value in enumerate(values)
        }

    return {
        "unit": "speculative_decode_cycles",
        "decode_cycles": cycles,
        "attempted_tokens_by_position": {
            f"D{depth + 1}": drafted[depth] for depth in range(3)
        },
        "accepted_tokens_by_position": {
            f"D{depth + 1}": accepted[depth] for depth in range(3)
        },
        "acceptance_rate_pct_by_position": {
            f"D{depth + 1}": (
                accepted[depth] / drafted[depth] * 100.0 if drafted[depth] else 0.0
            )
            for depth in range(3)
        },
        "attempted_counts": keyed(attempted_exact),
        "attempted_share_pct": shares(attempted_exact),
        "accepted_counts": keyed(accepted_exact),
        "accepted_share_pct": shares(accepted_exact),
        "mean_attempted_depth": (
            sum(depth * value for depth, value in enumerate(attempted_exact)) / cycles
            if cycles else 0.0
        ),
        "mean_accepted_depth": (
            sum(depth * value for depth, value in enumerate(accepted_exact)) / cycles
            if cycles else 0.0
        ),
    }


def depth_usage_from_schedules(
    *,
    attempted_depth_schedule: list[int],
    accepted_depth_schedule: list[int],
    verify_calls: int,
    drafted_by_depth: list[int],
    accepted_by_depth: list[int],
) -> dict[str, Any]:
    attempted = [int(value) for value in attempted_depth_schedule]
    accepted = [int(value) for value in accepted_depth_schedule]
    cycles = len(attempted)
    if not cycles or len(accepted) != cycles:
        raise ValueError("attempted and accepted depth schedules must align")
    if int(verify_calls) != cycles:
        raise ValueError("verify calls contradict recorded depth schedules")
    if any(depth not in range(4) for depth in (*attempted, *accepted)):
        raise ValueError("recorded speculative depth is outside D0-D3")
    if any(accepted_depth > attempted_depth for attempted_depth, accepted_depth in zip(attempted, accepted)):
        raise ValueError("accepted depth exceeds attempted depth")

    drafted = ([int(value) for value in drafted_by_depth] + [0, 0, 0])[:3]
    accepted_tokens = ([int(value) for value in accepted_by_depth] + [0, 0, 0])[:3]
    if not (drafted[0] >= drafted[1] >= drafted[2] >= 0):
        raise ValueError("drafted-depth histogram contradicts generated work")
    if not (
        accepted_tokens[0] >= accepted_tokens[1] >= accepted_tokens[2] >= 0
        and all(left <= right for left, right in zip(accepted_tokens, drafted))
    ):
        raise ValueError("accepted-depth histogram contradicts drafted work")

    attempted_counts = Counter(attempted)
    accepted_counts = Counter(accepted)

    def keyed(counts: Counter[int]) -> dict[str, int]:
        return {f"D{depth}": counts.get(depth, 0) for depth in range(4)}

    def shares(counts: Counter[int]) -> dict[str, float]:
        return {
            f"D{depth}": counts.get(depth, 0) / cycles * 100.0
            for depth in range(4)
        }

    return {
        "unit": "speculative_decode_cycles",
        "decode_cycles": cycles,
        "attempted_tokens_by_position": {
            f"D{depth + 1}": drafted[depth] for depth in range(3)
        },
        "accepted_tokens_by_position": {
            f"D{depth + 1}": accepted_tokens[depth] for depth in range(3)
        },
        "acceptance_rate_pct_by_position": {
            f"D{depth + 1}": (
                accepted_tokens[depth] / drafted[depth] * 100.0
                if drafted[depth]
                else 0.0
            )
            for depth in range(3)
        },
        "attempted_counts": keyed(attempted_counts),
        "attempted_share_pct": shares(attempted_counts),
        "accepted_counts": keyed(accepted_counts),
        "accepted_share_pct": shares(accepted_counts),
        "mean_attempted_depth": statistics.fmean(attempted),
        "mean_accepted_depth": statistics.fmean(accepted),
    }


def _aggregate_depth_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    drafted = [0, 0, 0]
    accepted = [0, 0, 0]
    attempted_schedule: list[int] = []
    accepted_schedule: list[int] = []
    for row in rows:
        attempted_schedule.extend(row.get("attempted_depth_schedule") or ())
        accepted_schedule.extend(row.get("accepted_depth_schedule") or ())
        for index, value in enumerate((row.get("drafted_by_depth") or ())[:3]):
            drafted[index] += int(value)
        for index, value in enumerate((row.get("accepted_by_depth") or ())[:3]):
            accepted[index] += int(value)
    return depth_usage_from_schedules(
        attempted_depth_schedule=attempted_schedule,
        accepted_depth_schedule=accepted_schedule,
        verify_calls=sum(int(row["verify_calls"]) for row in rows),
        drafted_by_depth=drafted,
        accepted_by_depth=accepted,
    )


def aggregate(
    *,
    workload: str,
    context_tokens: int,
    order: tuple[str, ...],
    receipts: list[dict[str, Any]],
    specs: dict[str, LaneSpec],
) -> dict[str, Any]:
    output_tokens, temperature, top_p, top_k = _workload_values(workload)
    errors: list[str] = []
    observed_lane_ids = tuple(dict.fromkeys(order))
    try:
        selected_lane_ids = _selected_lanes(observed_lane_ids)
    except ValueError as exc:
        errors.append(str(exc))
        selected_lane_ids = tuple(
            lane_id for lane_id in observed_lane_ids if lane_id in specs
        )
    expected_order = (
        selected_lane_ids
        if context_tokens == 131_072
        else _paired_order(selected_lane_ids)
        if selected_lane_ids
        else ()
    )
    if order != expected_order:
        errors.append(
            f"noncanonical lane order: {order} != {expected_order}"
        )
    if len(receipts) != len(order):
        errors.append(f"expected {len(order)} arms, found {len(receipts)}")
    for index, (lane_id, receipt) in enumerate(zip(order, receipts)):
        if lane_id not in specs:
            errors.append(f"arm {index}: unknown benchmark lane {lane_id}")
            continue
        errors.extend(
            f"arm {index}: {error}"
            for error in receipt_errors(
                receipt,
                lane=specs[lane_id],
                context_tokens=context_tokens,
                output_tokens=output_tokens,
            )
        )
    for key in (
        "prompt_token_sha256",
        "prompt_artifact_sha256",
        "context_artifact_sha256",
        "model_artifact_hashes",
        "row17_artifact_sha256",
    ):
        values = {json.dumps(receipt.get(key), sort_keys=True) for receipt in receipts}
        if len(values) != 1:
            errors.append(f"{key} changed across arms")

    summary: dict[str, dict[str, Any]] = {}
    fixed_rows = [
        row for row in receipts if row.get("lane_id") == "full-fixed-k3"
    ]
    fixed_wall = _mean(fixed_rows, "wall_s") if fixed_rows else None
    for lane_id in selected_lane_ids:
        rows = [row for row in receipts if row.get("lane_id") == lane_id]
        expected_arms = order.count(lane_id)
        if len(rows) != expected_arms:
            errors.append(f"{lane_id} has {len(rows)} arms, expected {expected_arms}")
            continue
        wall = _mean(rows, "wall_s")
        records_depth = (
            context_tokens == 131_072
            and "r11_position_ema" in gate._validate_route_id(specs[lane_id].route_id)
        )
        usage = None
        if rows and records_depth:
            try:
                usage = _aggregate_depth_usage(rows)
            except (KeyError, TypeError, ValueError) as exc:
                errors.append(f"{lane_id} depth usage is invalid: {exc}")
        summary[lane_id] = {
            "arms": len(rows),
            "source_commit": specs[lane_id].source_commit,
            "route_id": specs[lane_id].route_id,
            "prefill_tok_s_mean": _mean(rows, "prefill_tok_s"),
            "decode_tok_s_mean": _mean(rows, "decode_tok_s"),
            "wall_s_mean": wall,
            "wall_faster_vs_fixed_k3_pct": (
                (fixed_wall / wall - 1.0) * 100.0
                if fixed_wall is not None
                else None
            ),
            "peak_memory_gib_max": max(float(row["peak_memory_gib"]) for row in rows),
            "per_lane_token_deterministic": (
                len({row["token_hash"] for row in rows}) == 1
                if len(rows) > 1
                else None
            ),
            "depth_usage": usage,
        }
    return {
        "schema_version": 1,
        "kind": (
            "qwen38_native_mtp_four_lane_matrix"
            if selected_lane_ids == LANE_IDS
            else "qwen38_native_mtp_selected_lane_matrix"
        ),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": workload,
        "context_tokens": context_tokens,
        "conditioner_output_tokens": (
            0 if workload == "vanity" else CONDITIONER_OUTPUT_TOKENS
        ),
        "timed_output_tokens": output_tokens,
        "order": list(order),
        "lanes": list(selected_lane_ids),
        "sampler": {
            "temperature": temperature,
            "draft_temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": 42,
        },
        "software": {"mlx": REQUIRED_MLX_VERSION, "mlx_metal": REQUIRED_MLX_METAL_VERSION},
        "invariant_errors": errors,
        "summary": summary,
        "arms": receipts,
    }


def _load_isolated() -> Any:
    spec = importlib.util.spec_from_file_location("qwen38_matrix_isolated", ISOLATED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ISOLATED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validated_parent_guard_scope(scope: str) -> str:
    if scope not in {"direct", "attested_parent"}:
        raise RuntimeError(f"matrix parent has invalid execution guard scope: {scope}")
    return scope


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def _git_status(root: Path) -> list[str]:
    return subprocess.check_output(
        ["git", "status", "--short"], cwd=root, text=True
    ).splitlines()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _interpreter_versions(python: Path) -> dict[str, str]:
    program = (
        "import importlib.metadata,json;"
        "print(json.dumps({'mlx':importlib.metadata.version('mlx'),"
        "'mlx_metal':importlib.metadata.version('mlx-metal')},sort_keys=True))"
    )
    output = subprocess.check_output(
        [str(python.absolute()), "-c", program], text=True
    ).strip()
    return {str(key): str(value) for key, value in json.loads(output).items()}


def _assert_campaign_inputs(args: argparse.Namespace) -> None:
    for label, root in (
        ("baseline", args.baseline_root),
        ("candidate", args.candidate_root),
    ):
        if _git_status(root):
            raise RuntimeError(f"{label} source tree must be clean before the campaign")
    output_root = args.output_root.resolve()
    for label, root in (
        ("baseline", args.baseline_root.resolve()),
        ("candidate", args.candidate_root.resolve()),
    ):
        if output_root == root or output_root.is_relative_to(root):
            raise RuntimeError(
                f"output root must be outside the {label} source tree: {output_root}"
            )
    expected_prompt = VANITY_PROMPT_FILE if args.workload == "vanity" else PYTHON_PROMPT_FILE
    if args.prompt_file.resolve() != expected_prompt.resolve():
        raise RuntimeError(
            f"{args.workload} requires frozen prompt artifact {expected_prompt}"
        )
    expected_hashes = {
        args.prompt_file: PROMPT_ARTIFACT_SHA256[
            "vanity" if args.workload == "vanity" else "python"
        ],
        args.context_file: PYTHON_CONTEXT_SHA256,
        args.row17_artifact: ROW17_ARTIFACT_SHA256,
    }
    for path, expected_hash in expected_hashes.items():
        observed_hash = _sha256(path)
        if observed_hash != expected_hash:
            raise RuntimeError(
                f"frozen artifact hash mismatch for {path.name}: "
                f"{observed_hash} != {expected_hash}"
            )


def run(args: argparse.Namespace) -> int:
    _assert_campaign_inputs(args)
    observed_versions = _interpreter_versions(args.python)
    required_versions = {
        "mlx": REQUIRED_MLX_VERSION,
        "mlx_metal": REQUIRED_MLX_METAL_VERSION,
    }
    if observed_versions != required_versions:
        raise RuntimeError(
            f"benchmark interpreter versions mismatch: {observed_versions} "
            f"!= {required_versions}"
        )
    candidate_commit = _git_commit(args.candidate_root)
    baseline_commit = _git_commit(args.baseline_root)
    if baseline_commit != V292_COMMIT:
        raise RuntimeError(
            f"baseline must be exact v2.9.2: {baseline_commit} != {V292_COMMIT}"
        )
    specs = lane_specs(
        baseline_root=args.baseline_root,
        baseline_commit=baseline_commit,
        candidate_root=args.candidate_root,
        candidate_commit=candidate_commit,
        workload=args.workload,
    )
    contexts = (
        (VANITY_PROMPT_TOKENS,)
        if args.workload == "vanity"
        else tuple(args.contexts)
    )
    isolated = _load_isolated()
    args.output_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    with isolated._gpu_lock_scope(args.lock) as lock_scope:
        _validated_parent_guard_scope(lock_scope)
        model_hashes = gate._model_artifact_hashes(args.model)
        for context_tokens in contexts:
            order = (
                _paired_order(tuple(args.lanes))
                if args.workload == "vanity"
                else order_for_context(context_tokens, tuple(args.lanes))
            )
            context_root = args.output_root / f"{args.workload}-{context_tokens}"
            context_root.mkdir(parents=True, exist_ok=True)
            receipts: list[dict[str, Any]] = []
            for index, lane_id in enumerate(order):
                lane = specs[lane_id]
                output = context_root / f"arm-{index}-{lane_id}.json"
                command = child_command(
                    lane=lane,
                    workload=args.workload,
                    context_tokens=context_tokens,
                    output=output,
                    model=args.model,
                    prompt_file=args.prompt_file,
                    context_file=args.context_file,
                    row17_artifact=args.row17_artifact,
                    python=args.python,
                    lock=args.lock,
                )
                environment = dict(os.environ)
                environment["PYTHONPATH"] = str(lane.source_root)
                environment[MODEL_HASHES_ENV] = json.dumps(
                    model_hashes, sort_keys=True, separators=(",", ":")
                )
                result = isolated._run_attested_child(
                    command,
                    environment=isolated._environment_for_route(lane.route_id, environment),
                    lock_path=args.lock,
                    owns_process_group=True,
                )
                log = output.with_suffix(".log")
                log.write_text(result.stdout or "", encoding="utf-8")
                if result.returncode != 0 or not output.is_file():
                    raise RuntimeError(
                        f"{args.workload} {context_tokens} arm {index} failed; see {log}"
                    )
                receipt = json.loads(output.read_text(encoding="utf-8"))
                receipts.append(receipt)
                print(json.dumps({
                    "event": "arm_complete",
                    "workload": args.workload,
                    "context_tokens": context_tokens,
                    "arm": index + 1,
                    "lane": lane_id,
                    "wall_s": receipt["wall_s"],
                }), flush=True)
            combined = aggregate(
                workload=args.workload,
                context_tokens=context_tokens,
                order=order,
                receipts=receipts,
                specs=specs,
            )
            combined_path = context_root / "combined.json"
            combined_path.write_text(
                json.dumps(combined, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            if combined["invariant_errors"]:
                raise RuntimeError(
                    f"{args.workload} {context_tokens} invariant errors: "
                    f"{combined['invariant_errors']}"
                )
            completed.append({
                "workload": args.workload,
                "context_tokens": context_tokens,
                "receipt_sha256": hashlib.sha256(combined_path.read_bytes()).hexdigest(),
                "summary": combined["summary"],
            })
            (args.output_root / "index.json").write_text(
                json.dumps({
                    "kind": (
                        "qwen38_native_mtp_four_lane_campaign"
                        if tuple(args.lanes) == LANE_IDS
                        else "qwen38_native_mtp_selected_lane_campaign"
                    ),
                    "lanes": list(args.lanes),
                    "completed": completed,
                }, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("vanity", "low", "xhigh"), required=True)
    parser.add_argument("--contexts", nargs="+", type=int, default=list(CONTEXT_TOKENS))
    parser.add_argument("--lanes", nargs="+", choices=LANE_IDS, default=list(LANE_IDS))
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--row17-artifact", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.workload != "vanity" and any(value not in CONTEXT_TOKENS for value in args.contexts):
        raise ValueError(f"contexts must be selected from {CONTEXT_TOKENS}")
    _selected_lanes(tuple(args.lanes))
    return args


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
