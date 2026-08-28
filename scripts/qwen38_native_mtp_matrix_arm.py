#!/usr/bin/env python3
"""Run one source-attested arm of the Qwen3.8 native-MTP matrix."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REQUIRED_MLX_VERSION = "0.32.2"
REQUIRED_MLX_METAL_VERSION = "0.32.2"
MODEL_ARTIFACT_HASHES_ENV = "MTPLX_QWEN38_MODEL_ARTIFACT_HASHES"
CONTROL_ROUTE = "control"
VERIFY_STRATEGY = "capture_commit"
VERIFY_CORE = "linear-gdn-from-conv-tape"
SPECULATIVE_DEPTH = 3


def _activate_source_root(root: Path, expected_commit: str) -> tuple[Path, str, list[str]]:
    """Pin imports to one clean, exact source tree before importing MTPLX."""

    source_root = root.resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if commit != expected_commit:
        raise RuntimeError(f"source commit mismatch: {commit} != {expected_commit}")
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=source_root, text=True
    ).splitlines()
    if status:
        raise RuntimeError(f"source tree is not clean: {status}")
    source_text = str(source_root)
    sys.path[:] = [item for item in sys.path if Path(item or ".").resolve() != source_root]
    sys.path.insert(0, source_text)
    os.chdir(source_root)
    return source_root, commit, status


def _split_template(rendered: str, sentinel: str) -> tuple[str, str]:
    if rendered.count(sentinel) != 1:
        raise RuntimeError("chat template did not preserve the prompt sentinel exactly once")
    return tuple(rendered.split(sentinel, 1))  # type: ignore[return-value]


def _filled_context_ids(tokenizer: Any, context: str, budget: int) -> list[int]:
    if budget < 0:
        raise ValueError("prompt framing exceeds the requested token budget")
    context_ids = list(tokenizer.encode(context.rstrip() + "\n"))
    if not context_ids:
        raise ValueError("context must encode to at least one token")
    repeats = (budget + len(context_ids) - 1) // len(context_ids)
    return (context_ids * repeats)[:budget]


def build_prompt(
    tokenizer: Any,
    *,
    workload: str,
    instruction: str,
    context: str,
    target_tokens: int,
) -> tuple[str, list[int]]:
    """Build the exact benchmark prompt without importing an alternate tokenizer."""

    if target_tokens <= 0:
        raise ValueError("prompt token target must be positive")
    if workload == "vanity":
        token_ids = list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": instruction}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
        if len(token_ids) != target_tokens:
            raise ValueError(
                f"vanity prompt must encode to exactly {target_tokens} tokens; "
                f"found {len(token_ids)}"
            )
        return str(tokenizer.decode(token_ids)), token_ids
    if workload in {"low", "xhigh"}:
        sentinel = "__MTPLX_QWEN38_CONTEXT_SENTINEL_7A6E7D0C__"
        rendered = str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": sentinel}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
                reasoning_effort=workload,
            )
        )
        prefix, suffix = _split_template(rendered, sentinel)
        prefix_ids = list(tokenizer.encode(prefix))
        suffix_ids = list(tokenizer.encode(suffix))
        tail_ids = list(tokenizer.encode("\n\n" + instruction.strip()))
        context_budget = target_tokens - len(prefix_ids) - len(tail_ids) - len(suffix_ids)
        if context_budget <= 0:
            raise ValueError(
                f"instruction and {workload} framing do not fit the prompt budget"
            )
        token_ids = (
            prefix_ids
            + _filled_context_ids(tokenizer, context, context_budget)
            + tail_ids
            + suffix_ids
        )
        if len(token_ids) != target_tokens:
            raise RuntimeError(
                f"{workload} prompt construction missed its exact token budget"
            )
        return str(tokenizer.decode(token_ids)), token_ids
    raise ValueError(f"unknown workload: {workload}")


def _token_hash(tokens: list[int]) -> str:
    encoded = json.dumps(tokens, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_versions() -> dict[str, str]:
    versions = {
        "mlx": str(importlib.metadata.version("mlx")),
        "mlx_metal": str(importlib.metadata.version("mlx-metal")),
    }
    expected = {
        "mlx": REQUIRED_MLX_VERSION,
        "mlx_metal": REQUIRED_MLX_METAL_VERSION,
    }
    errors = [
        f"{name.replace('_', '-')}=={expected[name]} required, found {version}"
        for name, version in versions.items()
        if version != expected[name]
    ]
    if errors:
        raise RuntimeError("; ".join(errors))
    return versions


def _assert_imported_from_source(module_name: str, source_root: Path) -> None:
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise RuntimeError(f"selected source did not load module {module_name}")
    resolved = Path(module_file).resolve()
    if not resolved.is_relative_to(source_root):
        raise RuntimeError(f"module {module_name} was not imported from selected source")


def _attested_model_hashes(model: Path) -> dict[str, str]:
    encoded = os.environ.get(MODEL_ARTIFACT_HASHES_ENV)
    if not encoded:
        raise RuntimeError("attested parent did not provide model artifact hashes")
    try:
        raw = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise RuntimeError("attested model artifact hashes are invalid JSON") from exc
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError("attested model artifact hashes are empty")
    hashes = {str(name): str(digest) for name, digest in raw.items()}
    for name, digest in hashes.items():
        if Path(name).name != name or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise RuntimeError(f"invalid attested model artifact entry: {name!r}")
        if not (model / name).is_file():
            raise RuntimeError(f"attested model artifact is missing: {name}")
    return hashes


def _read_prompt(path: Path) -> tuple[str, str]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError("prompt file is empty")
    row = json.loads(rows[0])
    return str(row["id"]), str(row["prompt"])


def _finite_positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _generation_metrics(stats: Any) -> dict[str, Any]:
    peak = int(stats.peak_memory_bytes)
    target_prefill_time = _finite_positive(
        stats.prompt_target_prefill_time_s, "target prefill time"
    )
    target_prefill_rate = _finite_nonnegative(
        stats.prompt_target_prefill_tok_s, "target prefill throughput"
    )
    history_time = _finite_positive(stats.prompt_mtp_history_time_s, "MTP history time")
    history_rate = _finite_nonnegative(
        stats.prompt_mtp_history_tok_s, "MTP history throughput"
    )
    proposer_time = _finite_positive(stats.draft_time_s, "MTP proposer time")
    decode_elapsed = _finite_positive(stats.decode_elapsed_s, "decode elapsed time")
    decode_rate = _finite_nonnegative(stats.decode_tok_s, "decode throughput")
    proposer_rate = _finite_nonnegative(
        float(stats.drafted_tokens) / proposer_time, "MTP proposer throughput"
    )
    events = list(getattr(stats, "events", ()) or ())
    capture_commit_events = sum(
        str(event.get("capture_repair") or "").startswith("captured_")
        for event in events
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
        "verify_strategy": VERIFY_STRATEGY,
        "verify_core": VERIFY_CORE,
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


def _history_route_receipt(runtime: Any, prompt_tokens: int) -> dict[str, Any]:
    bind = getattr(runtime, "bind_mtp_history_append_route", None)
    if callable(bind):
        return dict(bind(prompt_tokens).receipt)
    return {
        "route_id": "native_internal_committed_history",
        "prompt_tokens": prompt_tokens,
        "row20_engaged": False,
        "construction_binding_available": False,
    }


def depth_usage(
    *,
    decode_cycles: int,
    verify_calls: int,
    drafted_by_depth: list[int],
    accepted_by_depth: list[int],
) -> dict[str, Any]:
    """Convert cumulative D1-D3 counters to exact D0-D3 cycle usage."""

    drafted = ([int(value) for value in drafted_by_depth] + [0, 0, 0])[:3]
    accepted = ([int(value) for value in accepted_by_depth] + [0, 0, 0])[:3]
    decode_cycles = int(decode_cycles)
    verified = int(verify_calls)
    if verified != drafted[0]:
        raise ValueError("verify calls contradict attempted MTP depth")
    if not (
        decode_cycles >= verified >= drafted[0] >= drafted[1] >= drafted[2] >= 0
    ):
        raise ValueError("drafted-depth histogram contradicts generated work")
    if not (
        decode_cycles >= accepted[0] >= accepted[1] >= accepted[2] >= 0
        and all(left <= right for left, right in zip(accepted, drafted))
    ):
        raise ValueError("accepted-depth histogram contradicts drafted work")
    attempted_exact = (
        decode_cycles - verified,
        drafted[0] - drafted[1],
        drafted[1] - drafted[2],
        drafted[2],
    )
    accepted_exact = (
        decode_cycles - accepted[0],
        accepted[0] - accepted[1],
        accepted[1] - accepted[2],
        accepted[2],
    )

    def counts(values: tuple[int, int, int, int]) -> dict[str, int]:
        return {f"D{depth}": value for depth, value in enumerate(values)}

    def shares(values: tuple[int, int, int, int]) -> dict[str, float]:
        return {
            f"D{depth}": value / decode_cycles * 100.0 if decode_cycles else 0.0
            for depth, value in enumerate(values)
        }

    return {
        "unit": "speculative_decode_cycles",
        "decode_cycles": decode_cycles,
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
        "attempted_counts": counts(attempted_exact),
        "attempted_share_pct": shares(attempted_exact),
        "accepted_counts": counts(accepted_exact),
        "accepted_share_pct": shares(accepted_exact),
        "mean_attempted_depth": (
            sum(depth * value for depth, value in enumerate(attempted_exact))
            / decode_cycles
            if decode_cycles
            else 0.0
        ),
        "mean_accepted_depth": (
            sum(depth * value for depth, value in enumerate(accepted_exact))
            / decode_cycles
            if decode_cycles
            else 0.0
        ),
    }


def _load_optimized_speed_stack(
    model: Path, *, record_depth_usage: bool
) -> tuple[Any, dict[str, Any]]:
    from mtplx.backends.registry import load_runtime_contract
    from mtplx.draft_lm_head import (
        _install_draft_lm_head,
        draft_lm_head_spec_from_runtime_contract,
    )
    from mtplx.draft_sampling import draft_sampler_spec_from_runtime_contract
    from mtplx.profiles import (
        apply_profile_env,
        get_profile,
        runtime_env_overrides_from_contract,
    )

    contract, contract_error = load_runtime_contract(model)
    if contract_error is not None:
        raise RuntimeError(f"invalid runtime contract: {contract_error}")
    runtime_contract = {} if contract is None else contract.raw
    profile = get_profile("turbo")
    fallback_head = {
        "bits": int(profile.draft_lm_head.bits),
        "group_size": int(profile.draft_lm_head.group_size),
        "mode": str(profile.draft_lm_head.mode),
    }
    draft_head = draft_lm_head_spec_from_runtime_contract(
        runtime_contract, fallback=fallback_head
    )
    if draft_head is None:
        raise RuntimeError("Turbo profile requires a draft-only LM head")
    draft_sampler = draft_sampler_spec_from_runtime_contract(runtime_contract)
    runtime_env_overrides = runtime_env_overrides_from_contract(runtime_contract)
    apply_profile_env(profile.name, runtime_env_overrides=runtime_env_overrides)
    if record_depth_usage:
        os.environ["MTPLX_DROP_EVENTS"] = "0"

    # Env-gated runtime modules are imported only after the production profile.
    from mtplx.runtime import load

    runtime = load(model, mtp=True)
    draft_head_report = _install_draft_lm_head(
        runtime,
        bits=int(draft_head["bits"]),
        group_size=int(draft_head["group_size"]),
        mode=str(draft_head["mode"]),
    )
    return runtime, {
        "profile": profile.name,
        "runtime_profile": profile.runtime_profile,
        "runtime_env": {
            **profile.env_dict(),
            **runtime_env_overrides,
            "MTPLX_DROP_EVENTS": os.environ["MTPLX_DROP_EVENTS"],
        },
        "draft_lm_head": draft_head,
        "draft_lm_head_report": draft_head_report,
        "draft_sampler": draft_sampler,
        "mtp_hidden_variant": "post_norm",
        "mtp_cache_policy": "persistent",
        "mtp_history_policy": "committed",
        "verify_strategy": VERIFY_STRATEGY,
        "verify_core": VERIFY_CORE,
    }


def _run_control_arm(
    runtime: Any,
    prompt_ids: list[int],
    *,
    max_tokens: int,
    seed: int,
    target_temperature: float,
    draft_temperature: float,
    top_p: float,
    top_k: int,
    force_exact_output: bool,
) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx.generation import generate_mtpk
    from mtplx.sampling import SamplerConfig

    target_sampler = SamplerConfig(
        temperature=target_temperature, top_p=top_p, top_k=top_k
    )
    draft_sampler = SamplerConfig(
        temperature=draft_temperature, top_p=top_p, top_k=top_k
    )
    history_route_receipt = _history_route_receipt(runtime, len(prompt_ids))
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = generate_mtpk(
        runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=target_sampler,
        draft_sampler=draft_sampler,
        speculative_depth=SPECULATIVE_DEPTH,
        adaptive_policy=None,
        seed=seed,
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        draft_core="stock",
        verify_strategy=VERIFY_STRATEGY,
        verify_core=VERIFY_CORE,
        mtp_history_policy="committed",
        stop_token_ids=set() if force_exact_output else None,
    )
    wall_s = time.perf_counter() - started
    stats = output.stats
    drafted_by_depth = list(stats.drafted_by_depth)
    accepted_by_depth = list(stats.accepted_by_depth)
    generated_tokens = int(stats.generated_tokens)
    return {
        **_generation_metrics(stats),
        "route_id": CONTROL_ROUTE,
        "installed_route_id": CONTROL_ROUTE,
        "route_fingerprint": hashlib.sha256(CONTROL_ROUTE.encode()).hexdigest(),
        "kernel_ids": [],
        "feature_receipt": {},
        "candidate_artifact_hashes": {},
        "source_rows": [],
        "draft_core": "stock",
        "history_route_receipt": history_route_receipt,
        "wall_s": wall_s,
        "generated_tokens": generated_tokens,
        "prompt_mtp_history_time_s": float(stats.prompt_mtp_history_time_s),
        "draft_time_s": float(stats.draft_time_s),
        "accepted_by_depth": accepted_by_depth,
        "drafted_by_depth": drafted_by_depth,
        "attempted_depth_schedule": [],
        "accepted_depth_schedule": [],
        "adaptive_policy_events": [],
        "adaptive_policy_receipt": None,
        "token_hash": _token_hash(list(output.tokens)),
        "tokens": list(output.tokens),
        "finish_reason": output.finish_reason,
    }


def _run_one(
    runtime: Any,
    config: dict[str, Any],
    model: Path,
    prompt_ids: list[int],
    *,
    route: str,
    max_tokens: int,
    seed: int,
    target_temperature: float,
    draft_temperature: float,
    top_p: float,
    top_k: int,
    row17_artifact: Path,
    record_depth_usage: bool,
    force_exact_output: bool,
    performance_profile: str = "low",
    allow_fixed_diagnostic_route: bool = False,
) -> dict[str, Any]:
    if route == CONTROL_ROUTE:
        return _run_control_arm(
            runtime,
            prompt_ids,
            max_tokens=max_tokens,
            seed=seed,
            target_temperature=target_temperature,
            draft_temperature=draft_temperature,
            top_p=top_p,
            top_k=top_k,
            force_exact_output=force_exact_output,
        )

    from scripts import qwen38_challenge_port_gate as source_gate

    allowed = {
        source_gate.LOW_FIXED_NATIVE_ROUTE,
        source_gate.LOW_ADAPTIVE_NATIVE_ROUTE,
        source_gate.LOW_Q4_ADAPTIVE_NATIVE_ROUTE,
        source_gate.XHIGH_FIXED_NATIVE_ROUTE,
        source_gate.XHIGH_ADAPTIVE_NATIVE_ROUTE,
        source_gate.XHIGH_Q4_ADAPTIVE_NATIVE_ROUTE,
        source_gate.GREEDY_ADAPTIVE_NATIVE_ROUTE,
        source_gate.GREEDY_Q4_ADAPTIVE_NATIVE_ROUTE,
    }
    if route not in allowed:
        if not allow_fixed_diagnostic_route:
            raise RuntimeError(
                f"matrix arm rejected non-final optimized route: {route}"
            )
        route_features = source_gate._validate_route_id(route)
        forbidden = route_features & {
            "r11_position_ema",
            "r17_q4_mtp_block",
            "r28_q4_mtp_block",
            "r36_qkv_islands",
        }
        if forbidden:
            raise RuntimeError(
                "fixed BF16 diagnostic route contains adaptive or Q4 features: "
                + ", ".join(sorted(forbidden))
            )
    result = source_gate._run_arm(
        runtime,
        config,
        model,
        prompt_ids,
        route_id=route,
        max_tokens=max_tokens,
        seed=seed,
        target_temperature=target_temperature,
        draft_temperature=draft_temperature,
        top_p=top_p,
        top_k=top_k,
        row17_artifact_path=row17_artifact,
        row28_artifact_path=None,
        row36_artifact_path=None,
        performance_profile=performance_profile,
        stop_token_ids=set() if force_exact_output else None,
    )
    if record_depth_usage and result.get("draft_core") == "device":
        result["depth_usage"] = depth_usage(
            decode_cycles=len(result["attempted_depth_schedule"]),
            verify_calls=int(result["verify_calls"]),
            drafted_by_depth=list(result["drafted_by_depth"]),
            accepted_by_depth=list(result["accepted_by_depth"]),
        )
    return result


def _assert_route_policy_contract(route: str, arm: dict[str, Any]) -> None:
    from scripts import qwen38_challenge_port_gate as source_gate

    route_features = source_gate._validate_route_id(route)
    policy = arm.get("adaptive_policy_receipt")
    if "r11_position_ema" in route_features:
        policy = policy or {}
        if policy.get("kind") != "position_ema" or not policy.get("executed"):
            raise RuntimeError("adaptive policy did not execute in timed arm")
        return
    if policy is not None or arm.get("adaptive_policy_events"):
        raise RuntimeError("fixed optimized route executed an adaptive policy")


def _performance_profile_for_workload(workload: str) -> str:
    return "xhigh" if workload == "vanity" else workload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--workload", choices=("vanity", "low", "xhigh"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--context-file", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--warmup-tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-temperature", type=float, required=True)
    parser.add_argument("--draft-temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--row17-artifact", type=Path, required=True)
    parser.add_argument("--record-depth-usage", action="store_true")
    parser.add_argument("--force-exact-output", action="store_true")
    parser.add_argument("--allow-fixed-diagnostic-route", action="store_true")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    source_root, source_commit, source_status = _activate_source_root(
        args.source_root, args.source_commit
    )

    # Import the attestation verifier from the selected source only after pinning it.
    from scripts.qwen35b_mtp_batch_numerics_attribution import (
        _verify_parent_guard_attestation,
    )

    if not _verify_parent_guard_attestation(args.lock):
        raise RuntimeError("matrix arm requires an attested lock-owning parent")
    versions = _validated_versions()
    model = args.model.resolve(strict=True)
    model_hashes = _attested_model_hashes(model)
    row17_artifact = args.row17_artifact.resolve(strict=True)
    row17_hash = _sha256(row17_artifact)

    runtime, optimized_stack = _load_optimized_speed_stack(
        model, record_depth_usage=args.record_depth_usage
    )
    for module_name in ("mtplx", "mtplx.runtime"):
        _assert_imported_from_source(module_name, source_root)
    from mtplx.artifacts import load_config

    config = load_config(model)
    prompt_id, instruction = _read_prompt(args.prompt_file)
    context = args.context_file.read_text(encoding="utf-8")
    prompt, prompt_ids = build_prompt(
        runtime.tokenizer,
        workload=args.workload,
        instruction=instruction,
        context=context,
        target_tokens=args.prompt_tokens,
    )
    if len(prompt_ids) != args.prompt_tokens:
        raise RuntimeError("prompt token count changed after construction")
    prompt_token_sha256 = _token_hash(prompt_ids)
    performance_profile = _performance_profile_for_workload(args.workload)

    conditioner = None
    if args.warmup_tokens > 0:
        conditioner = _run_one(
            runtime,
            config,
            model,
            prompt_ids,
            route=args.route,
            max_tokens=args.warmup_tokens,
            seed=args.seed,
            target_temperature=args.target_temperature,
            draft_temperature=args.draft_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            row17_artifact=row17_artifact,
            record_depth_usage=False,
            force_exact_output=args.force_exact_output,
            performance_profile=performance_profile,
            allow_fixed_diagnostic_route=args.allow_fixed_diagnostic_route,
        )
    arm = _run_one(
        runtime,
        config,
        model,
        prompt_ids,
        route=args.route,
        max_tokens=args.max_tokens,
        seed=args.seed,
        target_temperature=args.target_temperature,
        draft_temperature=args.draft_temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        row17_artifact=row17_artifact,
        record_depth_usage=args.record_depth_usage,
        force_exact_output=args.force_exact_output,
        performance_profile=performance_profile,
        allow_fixed_diagnostic_route=args.allow_fixed_diagnostic_route,
    )
    _assert_imported_from_source("mtplx.generation", source_root)
    if args.route != CONTROL_ROUTE:
        _assert_imported_from_source("scripts.qwen38_challenge_port_gate", source_root)
    receipt = {
        "schema_version": 1,
        "kind": "qwen38_native_mtp_matrix_arm",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lane_id": args.lane_id,
        "source_commit": source_commit,
        "source_status": source_status,
        "source_import_attested": True,
        "route_id": args.route,
        "workload": args.workload,
        "prompt_id": prompt_id,
        "prompt_tokens": len(prompt_ids),
        "prompt_token_sha256": prompt_token_sha256,
        "prompt_text_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_artifact_sha256": _sha256(args.prompt_file),
        "context_artifact_sha256": _sha256(args.context_file),
        "conditioner_output_tokens": args.warmup_tokens,
        "conditioner_generated_tokens": (
            0 if conditioner is None else int(conditioner["generated_tokens"])
        ),
        "conditioner_finish_reason": (
            None if conditioner is None else conditioner["finish_reason"]
        ),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "sampler": {
            "target_temperature": args.target_temperature,
            "draft_temperature": args.draft_temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        },
        "mlx_version": versions["mlx"],
        "mlx_metal_version": versions["mlx_metal"],
        "gpu_lock_scope": "attested_parent",
        "model_id": model.name,
        "model_artifact_hashes": model_hashes,
        "row17_artifact_sha256": row17_hash,
        "stop_token_policy": (
            "disabled_for_exact_output"
            if args.force_exact_output
            else "tokenizer_default"
        ),
        "optimized_stack": optimized_stack,
        **arm,
    }
    if receipt["route_id"] != arm["route_id"]:
        raise RuntimeError("timed arm route receipt changed")
    if args.route != CONTROL_ROUTE:
        _assert_route_policy_contract(args.route, arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    del source_root
    return 0


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
