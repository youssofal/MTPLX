"""Greedy Qwen3.8 DFlash2 depth-sweep adapters and orchestration."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from mtplx.benchmarks.dflash2_contract import (
    DepthBracket,
    parse_dflash2_widths,
    select_stock_depth,
)
from mtplx.sampling import SamplerConfig


GREEDY = SamplerConfig(temperature=1.0, top_p=1.0, top_k=1)
MTP_DEPTH = 3
QWEN38_OPTIMIZED_SPEED = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
QWEN38_OPTIMIZED_SPEED_DIRNAMES = (
    "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed",
    "Qwen3.8-27B-MTPLX-Optimized-Speed",
)
QWEN38_OPTIMIZED_QUALITY = "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality"
QWEN38_OPTIMIZED_QUALITY_DIRNAMES = (
    "Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality",
    "Qwen3.8-27B-MTPLX-Optimized-Quality",
)
QWEN38_DFLASH2 = "z-lab/Qwen3.8-27B-DFlash2"
QWEN38_DFLASH2_REVISION = "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"
QWEN38_DFLASH2_LAYERS = (5, 19, 33, 47, 61)


def _target_model_id(model_ref: str) -> str:
    path_name = Path(model_ref).expanduser().name
    for model_id, directory_names in (
        (QWEN38_OPTIMIZED_SPEED, QWEN38_OPTIMIZED_SPEED_DIRNAMES),
        (QWEN38_OPTIMIZED_QUALITY, QWEN38_OPTIMIZED_QUALITY_DIRNAMES),
    ):
        if model_ref == model_id or path_name in directory_names:
            return model_id
    raise ValueError(
        "DFlash2 benchmark requires Qwen3.8 Optimized Speed or Optimized Quality"
    )


def _generate_ar(*args, **kwargs):
    from mtplx.generation import generate_ar

    return generate_ar(*args, **kwargs)


def _generate_mtpk(*args, **kwargs):
    from mtplx.generation import generate_mtpk

    return generate_mtpk(*args, **kwargs)


def _build_offline_runtime_context(**kwargs):
    from dflash_mlx.runtime.context import build_offline_runtime_context

    return build_offline_runtime_context(**kwargs)


def _stream_dflash_generate(**kwargs):
    from dflash_mlx.runtime import stream_dflash_generate

    return stream_dflash_generate(**kwargs)


def _resolve_model_path(model_ref: str):
    from mtplx.hf_loader import resolve_model_path

    return resolve_model_path(model_ref)


def _resolve_draft_snapshot(repo_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_files_only=True,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.py",
                "*.txt",
                "tokenizer*",
            ],
        )
    ).resolve()
    if path.name != revision:
        raise ValueError("DFlash2 draft did not resolve to the pinned revision")
    return str(path)


def _inspect_model(model_path: str):
    from mtplx.artifacts import inspect_model

    return inspect_model(model_path)


def _runtime_env_overrides_from_contract(contract):
    from mtplx.profiles import runtime_env_overrides_from_contract

    return runtime_env_overrides_from_contract(contract)


def _apply_profile_env(profile: str, **kwargs):
    from mtplx.profiles import apply_profile_env

    return apply_profile_env(profile, **kwargs)


def _profile_env_overridden():
    from mtplx.profiles import profile_env_overridden

    return tuple(dict(row) for row in profile_env_overridden)


def _load_mtplx_dflash2_bundle(model_path: str, draft_ref: str):
    from mtplx.benchmarks.dflash2_runtime import load_mtplx_dflash2_bundle

    return load_mtplx_dflash2_bundle(model_path, draft_ref)


def _install_draft_lm_head(runtime: Any, **kwargs):
    from mtplx.draft_lm_head import _install_draft_lm_head

    return _install_draft_lm_head(runtime, **kwargs)


def _build_exact_python_prompt_ids(tokenizer: Any, **kwargs):
    from mtplx.benchmarks.dflash2_contract import build_exact_python_prompt_ids

    return build_exact_python_prompt_ids(tokenizer, **kwargs)


def _validated_runtime_contract(
    inspection: dict[str, Any],
    *,
    model_id: str = QWEN38_OPTIMIZED_SPEED,
) -> dict[str, Any]:
    """Fail before load unless this is an exact supported Qwen3.8 artifact."""

    artifact_contracts = {
        QWEN38_OPTIMIZED_SPEED: {
            "bits": 4,
            "group_size": 32,
            "lm_head": {"bits": 8, "group_size": 64, "mode": "affine"},
            "mtp_sidecar": "bf16",
            "verified_model": "Qwen3.8-27B-MTPLX-Optimized-Speed",
        },
        QWEN38_OPTIMIZED_QUALITY: {
            "bits": 8,
            "group_size": 64,
            "lm_head": None,
            "mtp_sidecar": "prequantized-mlx-affine",
            "verified_model": "Qwen3.8-27B-MTPLX-Optimized-Quality",
        },
    }
    try:
        artifact = artifact_contracts[model_id]
    except KeyError as error:
        raise ValueError("unsupported Qwen3.8 DFlash2 target artifact") from error

    quantization = inspection.get("quantization") or {}
    lm_head = quantization.get("language_model.lm_head") or {}
    mtp = inspection.get("mtp") or {}
    compatibility = inspection.get("compatibility") or {}
    runtime_contract = compatibility.get("runtime_contract")
    valid = (
        inspection.get("passes_primary_gate") is True
        and inspection.get("model_type") == "qwen3_5_text"
        and inspection.get("architecture") == "Qwen3_5ForConditionalGeneration"
        and inspection.get("num_hidden_layers") == 64
        and inspection.get("hidden_size") == 5120
        and quantization.get("bits") == artifact["bits"]
        and quantization.get("group_size") == artifact["group_size"]
        and (
            (artifact["lm_head"] is None and not lm_head)
            or lm_head == artifact["lm_head"]
        )
        and mtp.get("passes_tensor_gate") is True
        and mtp.get("sidecar_format") == artifact["mtp_sidecar"]
        and compatibility.get("tier") == "verified"
        and compatibility.get("arch_id") == "qwen3-next-mtp"
        and compatibility.get("support_level") == "verified-native"
        and isinstance(runtime_contract, dict)
    )
    if not valid:
        raise ValueError(
            "DFlash2 benchmark target does not match its verified Qwen3.8 "
            "artifact contract"
        )
    mtp_contract = runtime_contract.get("mtp_contract") or {}
    if (
        runtime_contract.get("recommended_profile") != "turbo"
        or runtime_contract.get("mtp_depth_max") != MTP_DEPTH
        or mtp_contract.get("mtp_quant_group_size") != 64
        or mtp_contract.get("mtp_quant_mode") != "affine"
        or (runtime_contract.get("verified_on") or {}).get("model")
        != artifact["verified_model"]
    ):
        raise ValueError(
            "Qwen3.8 runtime contract does not match the "
            "promoted turbo depth-3 MTP control"
        )
    return runtime_contract


def _validated_draft_meta(bundle: Any, pinned_path: str) -> dict[str, int]:
    meta = getattr(bundle, "draft_meta", None)
    if not isinstance(meta, dict):
        raise ValueError("DFlash2 bundle has no draft metadata")
    resolved_ref = meta.get("resolved_model_ref")
    if not isinstance(resolved_ref, str) or Path(resolved_ref).resolve() != Path(
        pinned_path
    ).resolve():
        raise ValueError("DFlash2 bundle did not load the pinned snapshot")
    config = meta.get("config") or {}
    dflash_config = config.get("dflash_config") or {}
    quant = meta.get("draft_quant")
    expected_quant = {"weight_bits": 4, "group_size": 64, "act_bits": 16}
    if (
        getattr(bundle, "checkpoint_block_size", None) != 8
        or tuple(getattr(bundle, "target_layer_ids", ())) != QWEN38_DFLASH2_LAYERS
        or dflash_config.get("block_size") != 8
        or tuple(dflash_config.get("target_layer_ids") or ())
        != QWEN38_DFLASH2_LAYERS
        or quant != expected_quant
    ):
        raise ValueError(
            "DFlash2 pinned snapshot metadata does not match block-8 q4/group-64"
        )
    return dict(expected_quant)


def _exact_tokens(tokens: Iterable[int], *, expected_tokens: int, engine: str) -> tuple[int, ...]:
    token_ids = tuple(int(token) for token in tokens)
    if len(token_ids) != expected_tokens:
        raise RuntimeError(
            f"{engine} did not produce the forced token count "
            f"{expected_tokens}: got {len(token_ids)}"
        )
    return token_ids


def run_target_oracle(
    bundle: Any,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 1024,
) -> tuple[int, ...]:
    output = _generate_ar(
        bundle.runtime,
        list(prompt_ids),
        max_tokens=max_tokens,
        sampler=GREEDY,
        seed=0,
        stop_token_ids=set(),
    )
    return _exact_tokens(
        output.tokens,
        expected_tokens=max_tokens,
        engine="target-only oracle",
    )


def arm_receipt_from_mtplx(
    output: Any,
    *,
    prompt_tokens: int,
) -> dict[str, Any]:
    stats = output.stats
    generated_tokens = int(stats.generated_tokens)
    elapsed_s = float(stats.elapsed_s)
    decode_elapsed_s = float(stats.decode_elapsed_s)
    prefill_s = elapsed_s - decode_elapsed_s
    if not math.isfinite(prefill_s) or prefill_s <= 0.0:
        raise RuntimeError("MTPLX MTP control must report a positive prefill duration")
    accepted_by_depth = [int(value) for value in stats.accepted_by_depth]
    accepted_from_draft = sum(accepted_by_depth)
    if not 0 <= accepted_from_draft <= generated_tokens:
        raise RuntimeError("MTPLX MTP control reported an invalid acceptance count")
    return {
        "tokens": tuple(int(token) for token in output.tokens),
        "generated_tokens": generated_tokens,
        "decode_tps": float(stats.decode_tok_s),
        "elapsed_s": elapsed_s,
        "decode_elapsed_s": decode_elapsed_s,
        "prefill_s": prefill_s,
        "prefill_tps": int(prompt_tokens) / prefill_s,
        "peak_memory_gb": float(stats.peak_memory_bytes) / (1024**3),
        "verify_calls": int(stats.verify_calls),
        "accepted_by_depth": accepted_by_depth,
        "accepted_from_draft": accepted_from_draft,
        "spec_decode_hit_rate": accepted_from_draft / generated_tokens,
        "engine": "mtplx_mtp",
    }


def run_mtp_control(
    bundle: Any,
    prompt_ids: Sequence[int],
    *,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    output = _generate_mtpk(
        bundle.runtime,
        list(prompt_ids),
        max_tokens=max_tokens,
        sampler=GREEDY,
        speculative_depth=MTP_DEPTH,
        seed=0,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        mtp_cache_policy="persistent",
        mtp_history_policy="cycle",
    )
    receipt = arm_receipt_from_mtplx(output, prompt_tokens=len(prompt_ids))
    receipt["tokens"] = _exact_tokens(
        receipt["tokens"],
        expected_tokens=max_tokens,
        engine="MTPLX MTP control",
    )
    if receipt["generated_tokens"] != max_tokens:
        raise RuntimeError(
            "MTPLX MTP control stats did not report the forced token count "
            f"{max_tokens}: got {receipt['generated_tokens']}"
        )
    return receipt


def build_fixed_dflash_runtime_context():
    return _build_offline_runtime_context(
        quantize_kv_cache=False,
        verify_mode="dflash",
        copyspec_mode="off",
    )


def arm_receipt_from_dflash_events(
    events: Iterable[Any],
    *,
    requested_width: int,
    expected_tokens: int,
) -> dict[str, Any]:
    from dflash_mlx.engine.events import SummaryEvent

    summaries = [event for event in events if isinstance(event, SummaryEvent)]
    if len(summaries) != 1:
        raise RuntimeError("DFlash2 stream ended without exactly one summary")
    summary = summaries[0]

    effective_width = int(summary.block_tokens or 0)
    if effective_width != requested_width:
        raise RuntimeError(
            f"DFlash2 requested width {requested_width} became {effective_width}"
        )
    if summary.fallback_ar:
        raise RuntimeError(
            "DFlash2 reported fallback AR: "
            f"{summary.fallback_reason or 'unspecified reason'}"
        )
    generated_tokens = int(summary.generation_tokens)
    if generated_tokens != expected_tokens:
        raise RuntimeError(
            "DFlash2 did not produce the forced token count "
            f"{expected_tokens}: got {generated_tokens}"
        )
    token_ids = _exact_tokens(
        summary.generated_token_ids,
        expected_tokens=expected_tokens,
        engine="DFlash2 token ID count",
    )
    prefill_us = float(summary.phase_timings_us.get("prefill", 0.0))
    prompt_token_count = int(summary.prompt_token_count)
    if (
        not math.isfinite(prefill_us)
        or prefill_us <= 0.0
        or prompt_token_count <= 0
    ):
        raise RuntimeError("DFlash2 summary must report a positive prefill duration")
    elapsed_us = float(summary.elapsed_us)
    decode_us = elapsed_us - prefill_us
    if not math.isfinite(decode_us) or decode_us <= 0.0:
        raise RuntimeError("DFlash2 summary must report a positive decode duration")

    accepted_from_draft = int(summary.accepted_from_draft)
    if not 0 <= accepted_from_draft <= generated_tokens:
        raise RuntimeError("DFlash2 summary reported an invalid acceptance count")

    return {
        "tokens": token_ids,
        "generated_tokens": generated_tokens,
        "decode_tps": generated_tokens / (decode_us / 1_000_000.0),
        "elapsed_s": elapsed_us / 1_000_000.0,
        "prefill_s": prefill_us / 1_000_000.0,
        "prefill_tps": prompt_token_count / (prefill_us / 1_000_000.0),
        "decode_elapsed_s": decode_us / 1_000_000.0,
        "peak_memory_gb": float(summary.peak_memory_gb or 0.0),
        "cycles_completed": int(summary.cycles_completed),
        "accepted_from_draft": accepted_from_draft,
        "acceptance_ratio": float(summary.acceptance_ratio),
        "spec_decode_hit_rate": accepted_from_draft / generated_tokens,
        "acceptance_history": [int(value) for value in summary.acceptance_history],
        "requested_width": int(requested_width),
        "effective_width": effective_width,
        "fallback_ar": bool(summary.fallback_ar),
        "fallback_reason": summary.fallback_reason,
        "engine": "dflash_mlx_0_1_10",
    }


def run_dflash2_candidate(
    bundle: Any,
    prompt_ids: Sequence[int],
    width: int,
    runtime_context: Any,
    *,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    events = _stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt_tokens_override=list(prompt_ids),
        prompt="",
        use_chat_template=False,
        max_new_tokens=max_tokens,
        block_tokens=int(width),
        stop_token_ids=[],
        runtime_context=runtime_context,
    )
    return arm_receipt_from_dflash_events(
        events,
        requested_width=int(width),
        expected_tokens=max_tokens,
    )


def _token_sha256(tokens: Sequence[int]) -> str:
    payload = ",".join(str(int(token)) for token in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def enrich_depth_sweep_metrics(receipt: dict[str, Any]) -> dict[str, Any]:
    """Add derived prefill and speculative-hit metrics to an existing receipt."""

    prompt_tokens = int(receipt["workload"]["prompt_tokens"])
    if prompt_tokens <= 0:
        raise ValueError("receipt prompt token count must be positive")
    for bracket in receipt["brackets"]:
        for arm_name in ("control_before", "candidate", "control_after"):
            arm = bracket[arm_name]
            generated_tokens = int(arm["generated_tokens"])
            if generated_tokens <= 0:
                raise ValueError("receipt generated token count must be positive")
            if "prefill_s" not in arm:
                arm["prefill_s"] = float(arm["elapsed_s"]) - float(
                    arm["decode_elapsed_s"]
                )
            prefill_s = float(arm["prefill_s"])
            if not math.isfinite(prefill_s) or prefill_s <= 0.0:
                raise ValueError("receipt prefill duration must be finite and positive")
            arm["prefill_tps"] = prompt_tokens / prefill_s

            if "accepted_from_draft" not in arm:
                arm["accepted_from_draft"] = sum(
                    int(value) for value in arm["accepted_by_depth"]
                )
            accepted_from_draft = int(arm["accepted_from_draft"])
            if not 0 <= accepted_from_draft <= generated_tokens:
                raise ValueError("receipt acceptance count is out of range")
            arm["spec_decode_hit_rate"] = (
                accepted_from_draft / generated_tokens
            )
    return receipt


def _receipt_without_tokens(arm: dict[str, Any]) -> dict[str, Any]:
    public = dict(arm)
    try:
        tokens = tuple(int(token) for token in public.pop("tokens"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("benchmark arm receipt must contain integer tokens") from error
    public["token_sha256"] = _token_sha256(tokens)
    return public


def _oracle_comparison(
    arm: dict[str, Any],
    oracle_tokens: tuple[int, ...],
) -> dict[str, Any]:
    tokens = tuple(int(token) for token in arm["tokens"])
    matching_prefix = 0
    for arm_token, oracle_token in zip(tokens, oracle_tokens):
        if arm_token != oracle_token:
            break
        matching_prefix += 1
    exact_match = tokens == oracle_tokens
    first_mismatch = None
    if not exact_match:
        first_mismatch = {
            "index": matching_prefix,
            "oracle_token": (
                oracle_tokens[matching_prefix]
                if matching_prefix < len(oracle_tokens)
                else None
            ),
            "arm_token": (
                tokens[matching_prefix]
                if matching_prefix < len(tokens)
                else None
            ),
        }
    return {
        "exact_match": exact_match,
        "matching_prefix_tokens": matching_prefix,
        "first_mismatch": first_mismatch,
    }


def _public_arm_receipt(
    arm: dict[str, Any],
    oracle_tokens: tuple[int, ...],
) -> dict[str, Any]:
    public = _receipt_without_tokens(arm)
    public["oracle_comparison"] = _oracle_comparison(arm, oracle_tokens)
    return public


def _arm_matches_oracle(
    arm: dict[str, Any],
    oracle_tokens: tuple[int, ...],
    *,
    expected_tokens: int,
) -> bool:
    try:
        tokens = tuple(int(token) for token in arm["tokens"])
        generated_tokens = int(arm["generated_tokens"])
    except (KeyError, TypeError, ValueError):
        return False
    return tokens == oracle_tokens and generated_tokens == expected_tokens


def _arm_has_expected_output(
    arm: dict[str, Any],
    *,
    expected_tokens: int,
) -> bool:
    try:
        tokens = tuple(int(token) for token in arm["tokens"])
        generated_tokens = int(arm["generated_tokens"])
    except (KeyError, TypeError, ValueError):
        return False
    return len(tokens) == expected_tokens and generated_tokens == expected_tokens


def run_dflash2_depth_sweep(
    *,
    bundle: Any,
    prompt_ids: Sequence[int],
    widths: Sequence[int],
    repetitions: int,
    max_tokens: int = 1024,
    oracle_tokens: Sequence[int] | None = None,
    arm_runner: Any | None = None,
) -> dict[str, Any]:
    width_tuple = tuple(widths)
    if not width_tuple:
        raise ValueError("widths must not be empty")
    if any(type(width) is not int or not 1 <= width <= 8 for width in width_tuple):
        raise ValueError("widths must be integers between 1 and 8")
    if len(width_tuple) != len(set(width_tuple)):
        raise ValueError("widths must be unique")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    prompt_tuple = tuple(int(token) for token in prompt_ids)
    if not prompt_tuple:
        raise ValueError("prompt_ids must not be empty")

    if oracle_tokens is None:
        oracle_tuple = run_target_oracle(
            bundle,
            prompt_tuple,
            max_tokens=max_tokens,
        )
    else:
        oracle_tuple = _exact_tokens(
            oracle_tokens,
            expected_tokens=max_tokens,
            engine="injected target-only oracle",
        )

    production_runner = arm_runner is None
    runtime_context = None
    if production_runner:
        runtime_context = build_fixed_dflash_runtime_context()

        def resolved_arm_runner(kind: str, width: int) -> dict[str, Any]:
            if kind == "mtp":
                return run_mtp_control(
                    bundle,
                    prompt_tuple,
                    max_tokens=max_tokens,
                )
            return run_dflash2_candidate(
                bundle,
                prompt_tuple,
                width,
                runtime_context,
                max_tokens=max_tokens,
            )

    else:
        resolved_arm_runner = arm_runner

    brackets: list[dict[str, Any]] = []
    selection_rows: list[DepthBracket] = []
    warmed_widths: set[int] = set()
    for repetition in range(repetitions):
        offset = repetition % len(width_tuple)
        rotated_widths = width_tuple[offset:] + width_tuple[:offset]
        for width in rotated_widths:
            if production_runner and width not in warmed_widths:
                run_dflash2_candidate(
                    bundle,
                    prompt_tuple,
                    width,
                    runtime_context,
                    max_tokens=32,
                )
                warmed_widths.add(width)

            control_before = resolved_arm_runner("mtp", MTP_DEPTH)
            candidate = resolved_arm_runner("dflash2", width)
            control_after = resolved_arm_runner("mtp", MTP_DEPTH)
            token_parity_passed = all(
                _arm_matches_oracle(
                    arm,
                    oracle_tuple,
                    expected_tokens=max_tokens,
                )
                for arm in (control_before, candidate, control_after)
            )
            validation_passed = all(
                _arm_has_expected_output(arm, expected_tokens=max_tokens)
                for arm in (control_before, candidate, control_after)
            ) and (
                candidate.get("requested_width") == width
                and candidate.get("effective_width") == width
                and candidate.get("fallback_ar") is False
            )

            selection_rows.append(
                DepthBracket(
                    width=width,
                    candidate_decode_tps=float(candidate["decode_tps"]),
                    control_before_tps=float(control_before["decode_tps"]),
                    control_after_tps=float(control_after["decode_tps"]),
                    validation_passed=validation_passed,
                )
            )
            brackets.append(
                {
                    "repetition": repetition,
                    "width": width,
                    "control_before": _public_arm_receipt(
                        control_before,
                        oracle_tuple,
                    ),
                    "candidate": _public_arm_receipt(candidate, oracle_tuple),
                    "control_after": _public_arm_receipt(
                        control_after,
                        oracle_tuple,
                    ),
                    "validation_passed": validation_passed,
                    "token_parity_passed": token_parity_passed,
                }
            )

    control_hashes = {
        arm["token_sha256"]
        for bracket in brackets
        for arm in (bracket["control_before"], bracket["control_after"])
    }
    candidate_repeats_checked = repetitions >= 2
    candidate_stable_by_width: dict[str, bool | None] = {}
    for width in width_tuple:
        hashes = {
            bracket["candidate"]["token_sha256"]
            for bracket in brackets
            if bracket["width"] == width
        }
        candidate_stable_by_width[str(width)] = (
            len(hashes) == 1 if candidate_repeats_checked else None
        )
    determinism_passed = len(control_hashes) == 1 and (
        not candidate_repeats_checked
        or all(value is True for value in candidate_stable_by_width.values())
    )
    determinism = {
        "control_stable": len(control_hashes) == 1,
        "candidate_repeats_checked": candidate_repeats_checked,
        "candidate_stable_by_width": candidate_stable_by_width,
        "passed": determinism_passed,
    }

    selection = None
    if determinism_passed and all(row.validation_passed for row in selection_rows):
        selection = asdict(select_stock_depth(selection_rows))
    return {
        "workload": {
            "prompt_tokens": len(prompt_tuple),
            "generated_tokens": max_tokens,
            "greedy": True,
            "temperature": GREEDY.temperature,
            "top_p": GREEDY.top_p,
            "top_k": GREEDY.top_k,
        },
        "widths": list(width_tuple),
        "repetitions": repetitions,
        "oracle_token_sha256": _token_sha256(oracle_tuple),
        "brackets": brackets,
        "determinism": determinism,
        "selection": selection,
    }


def run_cli_sweep(args: Any, *, token_count: int = 1024) -> dict[str, Any]:
    """Load one verified Qwen3.8 target artifact and run one closed sweep."""

    if type(token_count) is not int or token_count not in {32, 1024}:
        raise ValueError("DFlash2 benchmark token count must be 32 or 1024")
    requested_model = str(args.model)
    target_model_id = _target_model_id(requested_model)
    if str(args.draft_model) != QWEN38_DFLASH2:
        raise ValueError("DFlash2 benchmark requires the Qwen3.8 DFlash2 checkpoint")
    widths = parse_dflash2_widths(args.widths)
    if type(args.repetitions) is not int or args.repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")

    resolved_model = str(_resolve_model_path(requested_model))
    if _target_model_id(resolved_model) != target_model_id:
        raise ValueError(
            "Qwen3.8 target did not resolve to its verified local artifact"
        )
    inspection = _inspect_model(resolved_model).to_dict()
    runtime_contract = _validated_runtime_contract(
        inspection,
        model_id=target_model_id,
    )
    pinned_draft = _resolve_draft_snapshot(
        QWEN38_DFLASH2,
        QWEN38_DFLASH2_REVISION,
    )
    runtime_overrides = _runtime_env_overrides_from_contract(runtime_contract)
    _apply_profile_env("turbo", runtime_env_overrides=runtime_overrides)
    overridden = _profile_env_overridden()
    if overridden:
        names = ", ".join(str(row.get("var")) for row in overridden)
        raise ValueError(
            "operator environment overrides the optimized turbo control: " + names
        )

    bundle = _load_mtplx_dflash2_bundle(resolved_model, pinned_draft)
    draft_quant = _validated_draft_meta(bundle, pinned_draft)
    _install_draft_lm_head(
        bundle.runtime,
        bits=4,
        group_size=64,
        mode="affine",
    )
    prompt = _build_exact_python_prompt_ids(
        bundle.tokenizer,
        token_count=token_count,
    )
    receipt = run_dflash2_depth_sweep(
        bundle=bundle,
        prompt_ids=prompt.token_ids,
        widths=widths,
        repetitions=args.repetitions,
        max_tokens=token_count,
    )
    receipt["model"] = {
        "requested": requested_model,
        "resolved": resolved_model,
        "draft": {
            "requested": str(args.draft_model),
            "revision": QWEN38_DFLASH2_REVISION,
            "resolved": pinned_draft,
            "quant": draft_quant,
        },
        "profile": "turbo",
        "mtp_depth": MTP_DEPTH,
        "draft_lm_head": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    return receipt


def write_depth_sweep_result(path: Path | str, receipt: dict[str, Any]) -> None:
    """Atomically publish one JSON-safe depth-sweep receipt."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
