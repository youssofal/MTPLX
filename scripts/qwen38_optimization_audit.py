#!/usr/bin/env python3
"""Canonical 54-row Qwen 3.8 optimization rebenchmark inventory.

The historical Yukon rows are provenance inputs, not 54 independent runtime
switches.  This module binds every qualifying row either to one of the current
construction-time A/B cases or to an explicit source-causal disposition.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import qwen38_challenge_inventory as inventory  # noqa: E402
from scripts import qwen38_challenge_port_gate as gate  # noqa: E402
from scripts import qwen38_challenge_port_isolated_gate as isolated  # noqa: E402
from scripts import qwen38_native_mtp_campaign as campaign  # noqa: E402
from scripts.qwen38_challenge_inventory import InventoryRow  # noqa: E402
from scripts.qwen38_native_mtp_campaign import (  # noqa: E402
    DEFAULT_CONTEXT,
    DEFAULT_LOCK,
    DEFAULT_PROMPT,
    EXACT_CONDITIONER_TOKENS,
    EXACT_CONTEXT_TOKENS,
    EXACT_OUTPUT_TOKENS,
    EXACT_SEED,
    EXACT_TEMPERATURE,
    EXACT_TOP_K,
    EXACT_TOP_P,
)

EXPECTED_PROMPT_TOKEN_SHA256 = {
    1_024: "3015401ec3e421502b1a23f18d0a6e5d53004b189fdbab0e2e3ba27802fcd7e6",
    16_384: "af141694261c1d3c4d8aa6e36e903fa55fae08e2fc3ad21ad78ebcde213f6954",
}


@dataclass(frozen=True)
class DirectCase:
    case_id: str
    feature: str
    control_route: str
    candidate_route: str
    source_rows: tuple[int, ...]
    phase: str
    allow_frozen_candidate: bool = False


@dataclass(frozen=True)
class AuditRow:
    ordinal: int
    pr_number: int
    source_commit: str
    relative_percent: float
    mechanism: str
    disposition_kind: str
    reason: str
    direct_case_id: str | None


@dataclass(frozen=True)
class ExecutionItem:
    case_id: str
    feature: str
    control_route: str
    candidate_route: str
    context_tokens: int
    allow_frozen_candidate: bool


@dataclass(frozen=True)
class RouteContract:
    requested_features: frozenset[str]
    installed_route_id: str
    kernel_ids: tuple[str, ...]
    feature_receipt_keys: frozenset[str]
    draft_core: str
    adaptive: bool


DIRECT_CASES: tuple[DirectCase, ...] = (
    DirectCase("r08-device-draft", "r08_device_draft", "control", "r08_device_draft", (8,), "decode"),
    DirectCase("r10-compact-vocab", "r10_compact_vocab", "r08_device_draft", "r08_device_draft+r10_compact_vocab", (10,), "decode"),
    DirectCase("r11-position-ema", "r11_position_ema", "control", "r11_position_ema", (11,), "adaptive-decode"),
    DirectCase("r18-gdn-decay-memo", "r18_gdn_decay_memo", "control", "r18_gdn_decay_memo", (18,), "prefill+decode", True),
    DirectCase("r20-kv-only-history", "r20_kv_only_history", "control", "r20_kv_only_history", (20,), "prefill-history"),
    DirectCase("r21-qk-rms-rope", "r21_qk_rms_rope", "control", "r21_qk_rms_rope", (21,), "prefill+decode", True),
    DirectCase("r24-eval-ladder", "r24_eval_ladder", "r21_qk_rms_rope", "r21_qk_rms_rope+r24_eval_ladder", (24,), "prefill+decode", True),
    DirectCase("r26-prefill-ladder-3", "r26_prefill_ladder_3", "r21_qk_rms_rope+r24_eval_ladder", "r21_qk_rms_rope+r24_eval_ladder+r26_prefill_ladder_3", (26,), "prefill", True),
    DirectCase("r48-boundary-fused", "r48_boundary_fused", "control", "r48_boundary_fused", (45, 48), "prefill+decode", True),
    DirectCase("r50-wired-residency", "r50_wired_residency", "control", "r50_wired_residency", (50,), "whole-run", True),
    DirectCase("r53-command-buffers", "r53_command_buffers", "control", "r53_command_buffers", (53,), "prefill+decode", True),
    DirectCase("r61-dual-norm-concat", "r61_dual_norm_concat", "control", "r61_dual_norm_concat", (60, 61), "history+decode"),
    DirectCase("r63-q8-embedding-dual-norm", "r63_q8_embedding_dual_norm", "r61_dual_norm_concat", "r63_q8_embedding_dual_norm", (63,), "history+decode"),
    DirectCase("r17-q4-mtp-block", "r17_q4_mtp_block", "control", "r17_q4_mtp_block", (17,), "history+decode"),
    DirectCase("r28-q4-mtp-block", "r28_q4_mtp_block", "r17_q4_mtp_block", "r28_q4_mtp_block", (28,), "history+decode"),
    DirectCase("r36-qkv-islands", "r36_qkv_islands", "r17_q4_mtp_block", "r36_qkv_islands", (33, 36), "history+decode"),
)


_NON_DIRECT: dict[int, tuple[str, str]] = {
    2: ("already-base", "checkpoint and rejection fast paths are already part of the unchanged MTPLX control"),
    3: ("removed-rejected", "the packed target Q/K/V adaptation was measured slower and removed; no current switch remains"),
    4: ("already-base", "the generalized lazy verify boundary already subsumes this K=1 mechanism"),
    5: ("already-base", "the exact device top-k20 sampler supersedes the challenge-only target top-2 ledger"),
    6: ("no-callsite", "the stochastic route has no duplicate target argmax consumer"),
    7: ("already-base", "persistent committed MTP history is part of the unchanged control"),
    9: ("removed-rejected", "the paired group-32 M4 QMV adaptation regressed and was removed"),
    12: ("superseded", "the lazy prefix replay snapshot was removed by the following row"),
    13: ("removed-rejected", "the live S=4 GDN input-projection fusion regressed and was removed"),
    14: ("already-base", "the broader capture/commit recurrent boundary is already active"),
    15: ("no-callsite", "widths 6 through 9 are unreachable for native fixed K3 verification"),
    16: ("already-base", "the full route-specific compiled target graph already encloses these GDN expressions"),
    19: ("correctness-ineligible", "the argmax-only proposal readout cannot preserve temperature-1 speculative acceptance"),
    23: ("dependency-absent", "the compact selector depends on the absent argmax-only proposal path"),
    25: ("superseded", "the interim adaptive streak constant is replaced by later policy state"),
    30: ("already-base", "post-norm verify reuse and the output gate are already enclosed by the compiled target route"),
    32: ("superseded", "the interim adaptive policy revision is not the installed native policy"),
    34: ("no-callsite", "M6 and M9 direct-nibble lanes are unreachable under native K3"),
    37: ("conditioner-covered", "the warm-only restack is paid before timing and its M8 toggle is unreachable"),
    38: ("superseded", "the interim M8 toggle is unreachable under native K3 and later replaced"),
    39: ("superseded", "the temporary M4 ownership change is restored by the next row"),
    40: ("no-callsite", "the surviving source change targets M7, which native K3 never dispatches"),
    41: ("already-base", "the stronger group-32 M4 direct-nibble target lane is already in the control"),
    42: ("correctness-ineligible", "the affine-2 shortlist is argmax-only and cannot produce the full proposal distribution"),
    47: ("correctness-ineligible", "the selector is argmax-only and the M8 QMV lane is unreachable under native K3"),
    59: ("no-callsite", "the depth-6 adaptive floor is outside the native D0-D3 policy"),
    66: ("no-op", "the source change edits only a human-readable artifact note"),
    67: ("correctness-ineligible", "the selected-row rerank depends on the absent argmax shortlist"),
    69: ("correctness-ineligible", "the clustered shortlist cannot preserve the full stochastic proposal distribution"),
    70: ("removed-incompatible", "the QMV family cannot cross the retained lazy-history cache mutation boundary without changing the control"),
    71: ("correctness-ineligible", "the cluster QMV belongs only to the absent argmax shortlist"),
    78: ("dependency-absent", "the active-group launch depends on the removed row-70 QMV family"),
    79: ("dependency-absent", "the probe fraction belongs only to the absent clustered selector"),
    80: ("dependency-absent", "the M2 launch depends on the removed row-70/78 QMV family"),
    82: ("conditioner-covered", "the warm-shape change is untimed and the probe-sort factory belongs to the absent selector"),
}


def build_audit_rows(rows: Sequence[InventoryRow]) -> tuple[AuditRow, ...]:
    direct_by_row = {
        max(case.source_rows): case
        for case in DIRECT_CASES
    }
    consolidated_by_row = {
        row: case
        for case in DIRECT_CASES
        for row in case.source_rows[:-1]
    }
    known = set(direct_by_row) | set(consolidated_by_row) | set(_NON_DIRECT)
    actual = {row.ordinal for row in rows}
    if actual != known:
        missing = sorted(actual - known)
        extra = sorted(known - actual)
        raise ValueError(
            f"54-row audit mapping drifted: unmapped={missing}, nonqualifying={extra}"
        )
    result: list[AuditRow] = []
    for row in rows:
        direct = direct_by_row.get(row.ordinal)
        consolidated = consolidated_by_row.get(row.ordinal)
        if direct is None and consolidated is None:
            kind, reason = _NON_DIRECT[row.ordinal]
            direct_case_id = None
        elif consolidated is not None:
            kind = "consolidated-to"
            reason = (
                f"superseded precursor; fresh evidence is owned by current "
                f"descendant switch {consolidated.feature}"
            )
            direct_case_id = consolidated.case_id
        else:
            kind = "direct-abba"
            reason = (
                f"fresh matched 1K and 16K ABBA through current switch "
                f"{direct.feature}"
            )
            direct_case_id = direct.case_id
        result.append(
            AuditRow(
                ordinal=row.ordinal,
                pr_number=row.pr_number,
                source_commit=row.source_commit,
                relative_percent=float(row.relative_percent),
                mechanism=row.mechanism,
                disposition_kind=kind,
                reason=reason,
                direct_case_id=direct_case_id,
            )
        )
    return tuple(result)


def build_execution_plan(
    cases: Sequence[DirectCase],
) -> tuple[ExecutionItem, ...]:
    return tuple(
        ExecutionItem(
            case_id=case.case_id,
            feature=case.feature,
            control_route=case.control_route,
            candidate_route=case.candidate_route,
            context_tokens=context_tokens,
            allow_frozen_candidate=case.allow_frozen_candidate,
        )
        for case in cases
        for context_tokens in EXACT_CONTEXT_TOKENS
    )


def expected_route_contract(route_id: str) -> RouteContract:
    features = frozenset(gate._validate_route_id(route_id)) - {"control"}
    installed: list[str] = []
    kernels: list[str] = []
    receipt_keys: set[str] = set()

    if "r36_qkv_islands" in features:
        installed.extend(("r17_q4_mtp_block", "r36_qkv_islands"))
        kernels.append("qwen38_row36_q4_g64_bf16_qkv_islands_v1")
        receipt_keys.add("r36_qkv_islands")
    elif "r28_q4_mtp_block" in features:
        installed.extend(("r17_q4_mtp_block", "r28_q4_mtp_block"))
        kernels.append("qwen38_row28_q4_g64_mtp_block_v1")
        receipt_keys.add("r28_q4_mtp_block")
    elif "r17_q4_mtp_block" in features:
        installed.append("r17_q4_mtp_block")
        kernels.append("qwen38_row17_q4_g64_mtp_block_v1")
        receipt_keys.add("r17_q4_mtp_block")

    feature_contracts = (
        (
            "r20_kv_only_history",
            "kv_only_history",
            "qwen38_mtp_kv_only_history_ge16384_v1",
            "r20_kv_only_history",
        ),
        (
            "r18_gdn_decay_memo",
            "r18_gdn_decay_memo",
            "qwen38_row18_gdn_neg_exp_a_log_memo_v1",
            "r18_gdn_decay_memo",
        ),
        (
            "r21_qk_rms_rope",
            "r21_qk_rms_rope",
            "qwen38_qk_rms_rope_bf16_h256_r64_v1",
            "r21_qk_rms_rope",
        ),
    )
    for feature, installed_id, kernel_id, receipt_key in feature_contracts:
        if feature in features:
            installed.append(installed_id)
            kernels.append(kernel_id)
            receipt_keys.add(receipt_key)

    if "r24_eval_ladder" in features and "r21_qk_rms_rope" in features:
        kernels.append("qwen38_row24_qk_rms_rope_l_le16_v1")
        receipt_keys.add("r24_qk_length_limit")
    if "r24_eval_ladder" in features:
        installed.append("r24_eval_ladder")
        kernels.append("qwen38_row24_target_eval_ladder_v1")
        receipt_keys.add("r24_eval_ladder")
    if "r26_prefill_ladder_3" in features:
        installed.append("r26_prefill_ladder_3")
        kernels.append("qwen38_row26_prefill_eval_every3_v1")
        receipt_keys.add("r26_prefill_ladder_3")
        if "r21_qk_rms_rope" in features:
            kernels.append("qwen38_row26_qk_rms_rope_l_le32_v1")
            receipt_keys.add("r26_qk_length_limit")

    tail_contracts = (
        (
            "r48_boundary_fused",
            "r48_boundary_fused",
            "qwen38_row48_boundary_fused_residual_rmsnorm_v1",
            "r48_boundary_fused",
        ),
        (
            "r50_wired_residency",
            "r50_wired_residency",
            "qwen38_row50_post_warm_wired_residency_v1",
            "r50_wired_residency",
        ),
        (
            "r61_dual_norm_concat",
            "dual_norm",
            "qwen38_dual_rms_norm_concat_bf16_v1",
            "dual_norm",
        ),
        (
            "r63_q8_embedding_dual_norm",
            "r63_q8_embedding_dual_norm",
            "qwen38_row63_q8_g64_embedding_dual_rmsnorm_concat_v1",
            "r63_q8_embedding_dual_norm",
        ),
        (
            "r10_compact_vocab",
            "r10_compact_vocab",
            "qwen38_row10_compact_q4_g64_vocab_v1",
            "r10_compact_vocab",
        ),
    )
    for feature, installed_id, kernel_id, receipt_key in tail_contracts:
        if feature in features:
            installed.append(installed_id)
            kernels.append(kernel_id)
            receipt_keys.add(receipt_key)
    if "r53_command_buffers" in features:
        receipt_keys.add("r53_command_buffers")
    return RouteContract(
        requested_features=features,
        installed_route_id="+".join(installed) if installed else "control",
        kernel_ids=tuple(kernels),
        feature_receipt_keys=frozenset(receipt_keys),
        draft_core="device" if "r08_device_draft" in features else "stock",
        adaptive="r11_position_ema" in features,
    )


def _isolated_command(
    args: object,
    item: ExecutionItem,
    output: Path,
) -> list[str]:
    order = ",".join(
        (
            item.control_route,
            item.candidate_route,
            item.candidate_route,
            item.control_route,
        )
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/qwen38_challenge_port_isolated_gate.py"),
        "--model",
        str(args.model),
        "--prompt-file",
        str(args.prompt_file),
        "--prompt-tokens",
        str(item.context_tokens),
        "--context-file",
        str(args.context_file),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--seed",
        str(args.seed),
        "--target-temperature",
        str(args.temperature),
        "--draft-temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--order",
        order,
        "--control-route",
        item.control_route,
        "--candidate-route",
        item.candidate_route,
        "--lock",
        str(args.lock),
        "--output",
        str(output),
    ]
    if item.allow_frozen_candidate:
        command.append("--allow-frozen-candidate")
    for flag, value in (
        ("--row17-artifact", args.row17_artifact),
        ("--row28-artifact", args.row28_artifact),
        ("--row36-artifact", args.row36_artifact),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return command


def _receipt_errors(
    item: ExecutionItem,
    receipt: dict[str, object],
    *,
    expected_source_commit: str | None = None,
    expected_model_artifact_hashes: dict[str, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    expected_order = [
        item.control_route,
        item.candidate_route,
        item.candidate_route,
        item.control_route,
    ]
    expected_scalars = (
        ("prompt_token_target", item.context_tokens),
        ("max_tokens", EXACT_OUTPUT_TOKENS),
        ("seed", EXACT_SEED),
        ("target_temperature", EXACT_TEMPERATURE),
        ("draft_temperature", EXACT_TEMPERATURE),
        ("top_p", EXACT_TOP_P),
        ("top_k", EXACT_TOP_K),
        ("control_route_id", item.control_route),
        ("candidate_route_id", item.candidate_route),
        ("candidate_feature", item.feature),
        ("timed_arm_count", 4),
    )
    for key, expected in expected_scalars:
        if receipt.get(key) != expected:
            errors.append(f"{key} must equal {expected!r}")
    if receipt.get("order") != expected_order:
        errors.append("route order must be exact ABBA")
    if (
        receipt.get("prompt_token_sha256")
        != EXPECTED_PROMPT_TOKEN_SHA256[item.context_tokens]
    ):
        errors.append("prompt token hash does not match the exact workload")
    if receipt.get("source_status"):
        errors.append("source tree must be clean")
    if (
        expected_source_commit is not None
        and receipt.get("source_commit") != expected_source_commit
    ):
        errors.append("source commit does not match the audit parent")
    errors.extend(
        str(error) for error in receipt.get("receipt_invariant_errors") or ()
    )
    errors.extend(
        str(error) for error in receipt.get("candidate_engagement_errors") or ()
    )
    correctness = receipt.get("correctness") or {}
    if not isinstance(correctness, dict) or not correctness.get("passed"):
        errors.append("correctness and determinism must pass")
    if receipt.get("mlx_version") != "0.32.2":
        errors.append("MLX must be 0.32.2")
    if receipt.get("mlx_metal_version") != "0.32.2":
        errors.append("MLX Metal must be 0.32.2")
    if not receipt.get("model_artifact_hashes"):
        errors.append("model artifact hashes must be present")
    elif (
        expected_model_artifact_hashes is not None
        and receipt.get("model_artifact_hashes")
        != expected_model_artifact_hashes
    ):
        errors.append("model artifact hashes do not match the audit parent")
    if not receipt.get("phase_summary"):
        errors.append("phase summary must be present")
    arms = receipt.get("arms") or ()
    if len(arms) != 4 or any(
        not isinstance(arm, dict)
        or int(arm.get("generated_tokens", -1)) != EXACT_OUTPUT_TOKENS
        for arm in arms
    ):
        errors.append("all four arms must generate exactly 1024 tokens")
    if len(arms) == 4:
        for index, (arm, expected_route) in enumerate(
            zip(arms, expected_order, strict=True)
        ):
            if not isinstance(arm, dict):
                continue
            contract = expected_route_contract(expected_route)
            if arm.get("route_id") != expected_route:
                errors.append(f"arm {index} requested route does not match ABBA")
            if arm.get("installed_route_id") != contract.installed_route_id:
                errors.append(f"arm {index} installed route does not match request")
            if tuple(arm.get("kernel_ids") or ()) != contract.kernel_ids:
                errors.append(f"arm {index} kernel IDs do not match request")
            if set((arm.get("feature_receipt") or {}).keys()) != set(
                contract.feature_receipt_keys
            ):
                errors.append(
                    f"arm {index} feature receipt does not match requested state"
                )
            if arm.get("draft_core") != contract.draft_core:
                errors.append(f"arm {index} draft core does not match request")
            adaptive_receipt = arm.get("adaptive_policy_receipt")
            adaptive_events = arm.get("adaptive_policy_events") or ()
            if contract.adaptive:
                if not (
                    isinstance(adaptive_receipt, dict)
                    and adaptive_receipt.get("kind") == "position_ema"
                    and adaptive_receipt.get("executed") is True
                ) and not adaptive_events:
                    errors.append(
                        f"arm {index} adaptive policy did not execute"
                    )
            elif adaptive_receipt is not None or adaptive_events:
                errors.append(
                    f"arm {index} unexpectedly executed an adaptive policy"
                )
    return errors


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=gate.DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--context-file", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=EXACT_OUTPUT_TOKENS)
    parser.add_argument(
        "--warmup-tokens", type=int, default=EXACT_CONDITIONER_TOKENS
    )
    parser.add_argument("--seed", type=int, default=EXACT_SEED)
    parser.add_argument("--temperature", type=float, default=EXACT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=EXACT_TOP_P)
    parser.add_argument("--top-k", type=int, default=EXACT_TOP_K)
    parser.add_argument("--row17-artifact", type=Path, required=True)
    parser.add_argument("--row28-artifact", type=Path, required=True)
    parser.add_argument("--row36-artifact", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_errors(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []
    expected = {
        "row17_artifact": (
            238_934_093,
            "0e267a482e74c2664ce41dc4c4326f480020d015372fc9f7654ea3a136d62815",
        ),
        "row28_artifact": (
            238_934_129,
            "c934b40f1254858425cc0b5fdfe62b6ae13d1a4aff74da9d81606e92fdcf41ee",
        ),
        "row36_artifact": (
            270_404_624,
            "517bb133d7ca6e228a5129710b3cb2c25aa9944753b9f9a225fa1e8135df5e65",
        ),
    }
    for name, (expected_bytes, expected_sha256) in expected.items():
        path = Path(getattr(args, name)).expanduser()
        if not path.is_file():
            errors.append(f"{name} is missing: {path}")
            continue
        if path.stat().st_size != expected_bytes:
            errors.append(f"{name} byte count does not match pinned artifact")
            continue
        digest = gate._sha256_file(path)
        if digest != expected_sha256:
            errors.append(f"{name} SHA-256 does not match pinned artifact")
    return errors


def _parent_model_artifact_hashes(model_path: Path) -> dict[str, str]:
    """Hash once in the lock-owning audit parent, independent of outer guard."""

    return gate._model_artifact_hashes(model_path)


def _result_entry(
    item: ExecutionItem,
    receipt: dict[str, object],
    raw_path: Path,
) -> dict[str, object]:
    return {
        **asdict(item),
        "raw_receipt": str(raw_path.resolve()),
        "candidate_improvement_pct": receipt["candidate_improvement_pct"],
        "phase_summary": receipt["phase_summary"],
        "promotion": receipt["promotion"],
    }


def _campaign_payload(
    *,
    audit_rows: Sequence[AuditRow],
    plan: Sequence[ExecutionItem],
    results: Sequence[dict[str, object]],
    source_commit: str,
    model_artifact_hashes: dict[str, str],
    lock_scope: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "qwen38_54_optimization_rebench",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": len(results) == len(plan),
        "source_commit": source_commit,
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "model_artifact_hashes": model_artifact_hashes,
        "gpu_lock_scope": lock_scope,
        "protocol": {
            "contexts": list(EXACT_CONTEXT_TOKENS),
            "conditioner_tokens": EXACT_CONDITIONER_TOKENS,
            "output_tokens": EXACT_OUTPUT_TOKENS,
            "temperature": EXACT_TEMPERATURE,
            "top_p": EXACT_TOP_P,
            "top_k": EXACT_TOP_K,
            "seed": EXACT_SEED,
            "order": "ABBA",
            "cold_timed_prompt": True,
            "prefix_or_session_restore": False,
        },
        "historical_rows": [asdict(row) for row in audit_rows],
        "execution_plan": [asdict(item) for item in plan],
        "results": list(results),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    workload_errors = campaign._exact_workload_errors(args)
    artifact_errors = _artifact_errors(args)
    if workload_errors or artifact_errors:
        raise ValueError("; ".join((*workload_errors, *artifact_errors)))
    source_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    if source_status:
        raise RuntimeError("exact campaign requires a clean source tree")
    existing = list(args.output_dir.glob("*.json"))
    if args.output.exists() or existing:
        raise RuntimeError("fresh audit output paths must be empty")

    source = inventory.load_inventory(
        inventory.DEFAULT_RECEIPT, inventory.DEFAULT_DESIGN
    )
    report = inventory.validate_inventory(source)
    if report.errors:
        raise RuntimeError("; ".join(report.errors))
    audit_rows = build_audit_rows(report.qualifying_rows)
    plan = build_execution_plan(DIRECT_CASES)
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    results: list[dict[str, object]] = []

    with isolated._gpu_lock_scope(args.lock) as lock_scope:
        model_path = args.model.expanduser().resolve()
        model_artifact_hashes = _parent_model_artifact_hashes(model_path)
        environment = dict(os.environ)
        environment[gate.MODEL_ARTIFACT_HASHES_ENV] = json.dumps(
            model_artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for index, item in enumerate(plan, start=1):
            raw_path = args.output_dir / (
                f"{index:02d}-{item.case_id}-{item.context_tokens}.json"
            )
            print(
                f"START {index}/{len(plan)} {item.case_id} "
                f"context={item.context_tokens}",
                flush=True,
            )
            process = isolated._run_attested_child(
                _isolated_command(args, item, raw_path),
                environment=environment,
                lock_path=args.lock,
                owns_process_group=lock_scope == "direct",
            )
            if process.returncode not in (0, 2) or not raw_path.is_file():
                raise RuntimeError(
                    f"{item.case_id} context {item.context_tokens} failed "
                    f"({process.returncode}):\n{process.stdout}"
                )
            receipt = json.loads(raw_path.read_text(encoding="utf-8"))
            receipt_errors = _receipt_errors(
                item,
                receipt,
                expected_source_commit=source_commit,
                expected_model_artifact_hashes=model_artifact_hashes,
            )
            if receipt_errors:
                raise RuntimeError(
                    f"{item.case_id} context {item.context_tokens} receipt "
                    f"rejected: {'; '.join(receipt_errors)}"
                )
            result = _result_entry(item, receipt, raw_path)
            results.append(result)
            _write_json(
                args.output,
                _campaign_payload(
                    audit_rows=audit_rows,
                    plan=plan,
                    results=results,
                    source_commit=source_commit,
                    model_artifact_hashes=model_artifact_hashes,
                    lock_scope=lock_scope,
                ),
            )
            print(
                f"DONE {index}/{len(plan)} {item.case_id} "
                f"context={item.context_tokens} "
                f"wall_delta={float(receipt['candidate_improvement_pct']):+.4f}%",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
