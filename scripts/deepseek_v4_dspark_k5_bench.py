#!/usr/bin/env python3
"""Guarded real-checkpoint gates for DeepSeek DSpark through DFlash2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


DEFAULT_MODEL = Path(
    "/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1"
)
DEFAULT_PROMPT = "Write a Python function that returns the first n Fibonacci numbers."
MIA_SOURCE_REVISION = "d4ba142bc1d971eb73a911e207e3e963bbb3c455"
MIA_MODEL_REVISION = "22f28d32b9b29b4352eaa380ff8c2c170b2847ab"
MIA_SOURCE_CONFIG_SHA256 = (
    "b001ec8308044aa11daa0e624f5aea5e5362a63c05879a83a7be046b00eada82"
)
MIA_SOURCE_INDEX_SHA256 = (
    "61af5c0782a8651ef893004e84369d2281a0fc316c8bcefc0bd8f76244224649"
)
MIA_IMAGE_DIGEST = (
    "sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4"
)


def _require_clean_source(repo: Path) -> str:
    """Bind the measurement to one committed source tree before MLX import."""

    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"],
        text=True,
    )
    if status.strip():
        raise RuntimeError("worktree is dirty; refusing an unrepeatable benchmark")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError(f"source commit is malformed: {commit!r}")
    return commit


def _guard_before_mlx() -> dict:
    from deepseek_v4_guard_window import (
        WINDOW_PATH_ENV,
        WINDOW_SHA256_ENV,
        issue_guard_window,
        load_verified_guard_window,
    )

    path, digest = issue_guard_window()
    os.environ[WINDOW_PATH_ENV] = str(path)
    os.environ[WINDOW_SHA256_ENV] = digest
    return load_verified_guard_window()


def _memory(getter: str) -> int:
    import mlx.core as mx

    fn = getattr(mx, getter, None)
    if fn is None:
        fn = getattr(getattr(mx, "metal", None), getter, None)
    return int(fn()) if callable(fn) else 0


def _process_peak_rss_bytes() -> int:
    """Return the process-lifetime RSS high-water mark in bytes."""

    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1_024


def _finish_memory_receipt(
    *,
    mlx_active_after_load_bytes: int,
    mlx_peak_after_load_bytes: int,
    process_peak_rss_after_load_bytes: int,
    mlx_peak_reset_before_arm: bool,
    mlx_active_before_arm_bytes: int,
    mlx_peak_before_arm_bytes: int,
) -> dict:
    return {
        "mlx": {
            "after_load": {
                "active_bytes": int(mlx_active_after_load_bytes),
                "peak_bytes": int(mlx_peak_after_load_bytes),
                "peak_scope": "process_lifetime_through_load",
            },
            "arm": {
                "active_bytes_before": int(mlx_active_before_arm_bytes),
                "peak_bytes_before": int(mlx_peak_before_arm_bytes),
                "active_bytes_after": _memory("get_active_memory"),
                "peak_bytes": _memory("get_peak_memory"),
                "peak_reset_before_arm": bool(mlx_peak_reset_before_arm),
                "peak_scope": (
                    "since_explicit_arm_reset"
                    if mlx_peak_reset_before_arm
                    else "process_lifetime"
                ),
            },
        },
        "process": {
            "peak_rss_bytes_after_load": int(process_peak_rss_after_load_bytes),
            "peak_rss_bytes_after_arm": _process_peak_rss_bytes(),
            "peak_rss_scope": "process_lifetime_since_exec",
        },
    }


def _reset_peak_before_arm() -> bool:
    """Drop probe temporaries, then measure the cold generation arm itself."""

    import mlx.core as mx

    gc.collect()
    clear = getattr(mx, "clear_cache", None)
    if callable(clear):
        clear()
    reset = getattr(mx, "reset_peak_memory", None)
    if not callable(reset):
        return False
    reset()
    return True


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(tokens, separators=(",", ":")).encode()
    ).hexdigest()


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cache_contract(bundle, *, requested_span_tokens: int) -> dict:
    """Report the installed fixed arena without relabeling it per request."""

    requested_span_tokens = int(requested_span_tokens)
    plan = bundle.target_model._mia_engine_plan
    context_capacity = int(plan.context_capacity_tokens)
    target_physical_capacity = int(plan.target_physical_capacity_tokens)
    if requested_span_tokens > context_capacity:
        raise ValueError(
            f"requested span {requested_span_tokens} exceeds the installed "
            f"{context_capacity}-token cache plan"
        )

    target = []
    draft = []
    try:
        target = bundle.target_ops.make_cache(
            bundle.target_model,
            enable_speculative_linear_cache=True,
            quantize_kv_cache=False,
            cache_capacity_tokens=requested_span_tokens,
        )
        draft = bundle.draft_backend.make_cache(
            draft_model=bundle.draft_model,
            sink_size=0,
            window_size=int(bundle.draft_model.args.sliding_window),
            allow_full_context_layers=False,
        )

        page_geometry = tuple(plan.page_geometry)
        if len(target) != len(page_geometry):
            raise RuntimeError("Mia target cache count changed from its engine plan")

        window_capacities: set[int] = set()
        compressed_pages: dict[int, list[int]] = {}
        indexer_capacities: set[int] = set()
        for layer_id, (cache, geometry) in enumerate(
            zip(target, page_geometry, strict=True)
        ):
            ratio = int(cache.compress_ratio)
            if (
                int(geometry.layer_id) != layer_id
                or int(geometry.compress_ratio) != ratio
                or cache.window.mode != "nvfp4_stock432_fixed_window"
                or int(cache.window.record_bytes) != 432
            ):
                raise RuntimeError("Mia target fixed-window cache contract changed")
            window_capacities.add(int(cache.window.capacity))
            if ratio == 0:
                if int(geometry.compressed_capacity) != 0:
                    raise RuntimeError("Mia uncompressed page geometry changed")
                continue

            expected_capacity = (target_physical_capacity + ratio - 1) // ratio
            observed_capacity = int(cache.compressed.capacity)
            if (
                cache.compressed.mode != "nvfp4_stock432_paged"
                or int(cache.compressed.record_bytes) != 432
                or int(geometry.compressed_capacity) != expected_capacity
                or observed_capacity != expected_capacity
            ):
                raise RuntimeError("Mia compressed page geometry changed")
            tier = compressed_pages.setdefault(ratio, [observed_capacity, 0])
            if tier[0] != observed_capacity:
                raise RuntimeError("Mia compressed page tier has mixed capacities")
            tier[1] += 1

            if ratio == 4:
                indexer = cache.index_compressed
                if (
                    indexer.mode != "fp8_e4m3_ue8m0_scale132_paged"
                    or int(indexer.record_bytes) != 132
                    or int(indexer.capacity) != expected_capacity
                ):
                    raise RuntimeError("Mia indexer page geometry changed")
                indexer_capacities.add(int(indexer.capacity))

        if len(window_capacities) != 1 or len(indexer_capacities) != 1:
            raise RuntimeError("Mia fixed target cache capacities are inconsistent")

        draft_ring_capacities: set[int] = set()
        for cache in draft:
            ring = cache.ring
            if (
                ring.mode != "nvfp4_stock432_fixed_ring"
                or int(ring.record_bytes) != 432
                or int(ring.nbytes) % int(ring.record_bytes)
            ):
                raise RuntimeError("Mia DSpark fixed-ring cache contract changed")
            draft_ring_capacities.add(
                int(ring.nbytes) // int(ring.record_bytes)
            )
        if len(draft_ring_capacities) != 1:
            raise RuntimeError("Mia DSpark ring capacities are inconsistent")

        return {
            "request": {"span_tokens": requested_span_tokens},
            "installed_cache_plan": {
                "context_capacity_tokens": context_capacity,
                "target_physical_capacity_tokens": target_physical_capacity,
                "max_batch_tokens": int(plan.max_batch_tokens),
            },
            "target_kv": {
                "mode": "nvfp4_stock432",
                "record_bytes": 432,
                "start": 0,
                "layers": len(target),
                "window_mode": "nvfp4_stock432_fixed_window",
                "window_capacity_records": next(iter(window_capacities)),
                "compressed_pages": [
                    {
                        "compress_ratio": ratio,
                        "capacity_records": values[0],
                        "layers": values[1],
                    }
                    for ratio, values in sorted(compressed_pages.items())
                ],
                "paged_compressed_layers": sum(
                    values[1] for values in compressed_pages.values()
                ),
                "paged_indexer_layers": sum(
                    int(cache.compress_ratio == 4) for cache in target
                ),
                "indexer_mode": "fp8_e4m3_ue8m0_scale132_paged",
                "indexer_record_bytes": 132,
                "indexer_capacity_records": next(iter(indexer_capacities)),
            },
            "dspark_kv": {
                "mode": "nvfp4_stock432",
                "record_bytes": 432,
                "start": 0,
                "stages": len(draft),
                "ring_mode": "nvfp4_stock432_fixed_ring",
                "ring_capacity_records": next(iter(draft_ring_capacities)),
            },
        }
    finally:
        bundle.target_ops.cleanup_generation_caches(target, draft)


def _prompt_ids(bundle, text: str, target_tokens: int | None = None) -> list[int]:
    encoded = [int(value) for value in bundle.tokenizer.encode(text)]
    if target_tokens is None:
        return encoded
    if target_tokens <= 0 or not encoded:
        raise ValueError("prompt token target requires positive size and non-empty text")
    return (encoded * ((target_tokens + len(encoded) - 1) // len(encoded)))[
        :target_tokens
    ]


def _python_vocabulary_prompt_ids(
    tokenizer,
    *,
    context_tokens: int,
    python_prompt_tokens: int = 1_024,
) -> tuple[list[int], dict]:
    """Build a near-one-pass vocabulary prefix plus a coherent Python tail."""

    from mtplx.benchmarks.programming_prompts import (
        build_unique_programming_context,
    )

    context_tokens = int(context_tokens)
    python_prompt_tokens = int(python_prompt_tokens)
    if python_prompt_tokens <= 0 or context_tokens < python_prompt_tokens:
        raise ValueError(
            "Python vocabulary prompt requires context >= positive tail size"
        )
    python_text = build_unique_programming_context()
    encoded_python = [int(value) for value in tokenizer.encode(python_text)]
    if len(encoded_python) < python_prompt_tokens:
        raise ValueError(
            "unique Python repository prompt is shorter than the requested tail"
        )
    python_tail = encoded_python[-python_prompt_tokens:]

    base_vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    get_vocab = getattr(tokenizer, "get_vocab", None)
    full_vocab = get_vocab() if callable(get_vocab) else {}
    vocab_size = max(
        base_vocab_size,
        max((int(value) for value in full_vocab.values()), default=-1) + 1,
    )
    if vocab_size <= 0:
        raise ValueError("tokenizer does not expose a positive vocabulary size")
    special_ids = {
        int(value)
        for value in (getattr(tokenizer, "all_special_ids", ()) or ())
        if 0 <= int(value) < vocab_size
    }
    if len(special_ids) >= vocab_size:
        raise ValueError("tokenizer vocabulary contains no normal token ids")

    digest = hashlib.sha256(
        json.dumps(python_tail, separators=(",", ":")).encode()
    ).digest()
    start = int.from_bytes(digest[:8], "little") % vocab_size
    step = 65_537 % vocab_size
    if step == 0:
        step = 1
    while math.gcd(step, vocab_size) != 1:
        step += 1

    filler_target = context_tokens - python_prompt_tokens
    filler: list[int] = []
    cursor = 0
    while len(filler) < filler_target:
        token_id = (start + (cursor % vocab_size) * step) % vocab_size
        cursor += 1
        if token_id not in special_ids:
            filler.append(token_id)
    token_ids = filler + python_tail
    filler_unique = len(set(filler))
    return token_ids, {
        "prompt_policy": "python_vocab_tail_v1",
        "prompt_context_tokens": context_tokens,
        "prompt_actual_tokens": len(token_ids),
        "python_prompt_tokens": python_prompt_tokens,
        "python_prompt_source_tokens": len(encoded_python),
        "python_prompt_sha256": hashlib.sha256(python_text.encode()).hexdigest(),
        "vocabulary_size": vocab_size,
        "vocabulary_base_size": base_vocab_size,
        "vocabulary_special_ids_excluded": len(special_ids),
        "vocabulary_permutation_step": step,
        "vocabulary_filler_tokens": len(filler),
        "vocabulary_unique_ids": filler_unique,
        "vocabulary_duplicate_ids": len(filler) - filler_unique,
        "prompt_token_sha256": _token_digest(token_ids),
    }


def _arm_payload(
    output,
    *,
    total_prompt_tokens: int,
    new_prefill_tokens: int | None = None,
    ttft_s: float | None = None,
) -> dict:
    stats = output.stats.to_dict()
    total_prompt_tokens = int(total_prompt_tokens)
    if new_prefill_tokens is None:
        new_prefill_tokens = int(stats["new_prefill_tokens"])
    else:
        new_prefill_tokens = int(new_prefill_tokens)
    prompt_time = stats["prompt_eval_time_s"]
    return {
        "tokens": list(output.tokens),
        "token_digest": _token_digest(list(output.tokens)),
        "generated_tokens": len(output.tokens),
        "prompt_tokens": total_prompt_tokens,
        "new_prefill_tokens": new_prefill_tokens,
        "prompt_time_s": prompt_time,
        "prefill_tok_s": (
            new_prefill_tokens / prompt_time if prompt_time > 0 else 0.0
        ),
        "prefill_tok_s_token_basis": "new_prefill_tokens",
        "ttft_s": None if ttft_s is None else float(ttft_s),
        "ttft_scope": "request_start_through_first_emitted_token",
        "decode_time_s": stats["decode_elapsed_s"],
        "elapsed_s": stats["elapsed_s"],
        "decode_tok_s": stats["decode_tok_s"],
        "milliseconds_per_token": (
            1000.0 * stats["decode_elapsed_s"] / len(output.tokens)
            if output.tokens
            else 0.0
        ),
        "peak_memory_bytes": stats["peak_memory_bytes"],
        "peak_memory_kind": "mlx_allocator_peak",
        "peak_memory_scope": "since_last_mlx_peak_reset",
        "accepted_future_tokens": stats["accepted_drafts"],
        "drafted_future_tokens": stats["drafted_tokens"],
        "events": stats["events"],
    }


def _target_ar(bundle, prompt_ids: list[int], max_tokens: int) -> dict:
    from mtplx.generation import generate_sealed_target_ar
    from mtplx.sampling import SamplerConfig

    output = generate_sealed_target_ar(
        bundle.runtime,
        prompt_ids,
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        seed=0,
        stop_token_ids=set(),
    )
    return _arm_payload(output, total_prompt_tokens=len(prompt_ids))


def _dspark(bundle, prompt_ids: list[int], max_tokens: int, context) -> dict:
    from mtplx.deepseek_v4_dflash2 import generate_deepseek_v4_dflash2

    request_started = time.perf_counter()
    first_token_s = None

    def record_first_token(_token_ids: list[int]) -> None:
        nonlocal first_token_s
        if first_token_s is None:
            first_token_s = time.perf_counter()

    output = generate_deepseek_v4_dflash2(
        bundle,
        prompt_ids,
        max_tokens=max_tokens,
        stop_token_ids=[],
        runtime_context=context,
        token_callback=record_first_token,
    )
    return _arm_payload(
        output,
        total_prompt_tokens=len(prompt_ids),
        new_prefill_tokens=len(prompt_ids),
        ttft_s=(
            None if first_token_s is None else first_token_s - request_started
        ),
    )


def _first_epoch(bundle, prompt_ids: list[int], context) -> dict:
    from dflash_mlx.engine.config import verify_token_count_for_block
    from dflash_mlx.engine.events import SummaryEvent
    from dflash_mlx.runtime import stream_dflash_generate

    summary = None
    for event in stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt_tokens_override=prompt_ids,
        prompt="",
        use_chat_template=False,
        max_new_tokens=6,
        block_tokens=6,
        stop_token_ids=[],
        quantize_kv_cache=False,
        runtime_context=context,
    ):
        if isinstance(event, SummaryEvent):
            summary = event
    if summary is None:
        raise RuntimeError("fixed-linear DFlash2 did not emit a summary")
    if (
        summary.cycles_completed < 1
        or not summary.acceptance_history
        or summary.block_tokens is None
        or summary.verify_len_cap is None
    ):
        raise RuntimeError("fixed-linear DFlash2 summary has no completed first cycle")

    block_len = max(1, min(int(summary.block_tokens), 6))
    physical_verify_width = verify_token_count_for_block(
        block_len,
        int(summary.verify_len_cap),
    )
    acceptance_len = int(summary.acceptance_history[0])
    commit_count = 1 + acceptance_len
    committed_output_ids = list(summary.generated_token_ids[:commit_count])
    if commit_count > physical_verify_width or len(committed_output_ids) != commit_count:
        raise RuntimeError("fixed-linear DFlash2 first-cycle accounting is inconsistent")
    return {
        "cycle": 1,
        "block_len": block_len,
        "proposed_token_count": block_len,
        "future_draft_count": max(0, block_len - 1),
        "physical_verify_width": physical_verify_width,
        "acceptance_len": acceptance_len,
        "commit_count": commit_count,
        "committed_output_ids": committed_output_ids,
        "committed_output_relation": "summary_generated_prefix",
        "summary": summary.to_payload(),
    }


def _write(payload: dict, output: Path | None) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(encoded)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
    print(output)


def _deepseek_quality_gate(candidate_tokens: list[int], control_tokens: list[int]) -> dict:
    """Require exact committed tokens for the sealed Mia DSpark lane."""

    from deepseek_v4_mtpk_bench import _divergence

    gate = _divergence(candidate_tokens, control_tokens)
    gate["enforced"] = True
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--arm",
        choices=("construct", "one-cycle", "dspark", "exact-stream", "bracket"),
        required=True,
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument(
        "--prompt-mode",
        choices=("repeat", "python-vocab"),
        default="repeat",
    )
    parser.add_argument("--python-prompt-tokens", type=int, default=1_024)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    source_commit = _require_clean_source(repo)
    guard = _guard_before_mlx()
    from mtplx.dflash_identity import require_pinned_dflash_install

    dflash_identity = require_pinned_dflash_install()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mtplx.benchmarks.dflash2_runtime import (
        load_mtplx_deepseek_v4_dflash2_bundle,
    )
    from mtplx.deepseek_v4_dspark_artifact import open_verified_dspark_artifact

    artifact = open_verified_dspark_artifact(args.model)
    load_started = time.perf_counter()
    bundle = load_mtplx_deepseek_v4_dflash2_bundle(
        str(args.model),
        dflash_identity=dflash_identity,
    )
    load_time = time.perf_counter() - load_started
    context = bundle.runtime_context
    if context.dflash_identity is not dflash_identity:
        raise RuntimeError("DFlash bundle did not preserve its preflight receipt")
    parameters = tree_flatten(bundle.target_model.parameters())
    resident_bytes = sum(int(value.nbytes) for _name, value in parameters)
    mlx_active_after_load = _memory("get_active_memory")
    mlx_peak_after_load = _memory("get_peak_memory")
    process_peak_rss_after_load = _process_peak_rss_bytes()
    if args.prompt_mode == "python-vocab":
        if args.prompt_tokens is None:
            raise ValueError("--prompt-mode python-vocab requires --prompt-tokens")
        prompt_ids, prompt_metadata = _python_vocabulary_prompt_ids(
            bundle.tokenizer,
            context_tokens=args.prompt_tokens,
            python_prompt_tokens=args.python_prompt_tokens,
        )
    else:
        prompt_ids = _prompt_ids(bundle, args.prompt, args.prompt_tokens)
        prompt_metadata = {
            "prompt_policy": "repeat_hard_truncate",
            "prompt_context_tokens": len(prompt_ids),
            "prompt_actual_tokens": len(prompt_ids),
        }
    requested_span_tokens = len(prompt_ids) + max(0, int(args.max_tokens))
    os.environ["MTPLX_CONTEXT_WINDOW_TOKENS"] = str(requested_span_tokens)
    common = {
        "schema_version": 3,
        "kind": "deepseek_v4_dspark_dflash2_k5",
        "arm": args.arm,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": str(args.model),
        "mia_source": "MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark",
        "mia_source_revision": MIA_SOURCE_REVISION,
        "mia_model": "0xSero/deepseek-v4-flash-0731-spark",
        "mia_model_revision": MIA_MODEL_REVISION,
        "mia_source_config_sha256": MIA_SOURCE_CONFIG_SHA256,
        "mia_source_index_sha256": MIA_SOURCE_INDEX_SHA256,
        "mia_runtime_image_digest": MIA_IMAGE_DIGEST,
        "config_sha256": artifact.config_sha256,
        "index_sha256": artifact.index_sha256,
        "target_artifact_index_sha256": _file_digest(
            args.model / "model.safetensors.index.json"
        ),
        "draft_artifact_index_sha256": artifact.index_sha256,
        "engine": "dflash_mlx_0_1_10",
        "dflash_vcs": context.dflash_identity.vcs,
        "dflash_url": context.dflash_identity.url,
        "dflash_revision": context.dflash_identity.commit_id,
        "dflash_requested_revision": context.dflash_identity.requested_revision,
        "physical_verify_width": bundle.checkpoint_block_size,
        "future_draft_count": bundle.checkpoint_block_size - 1,
        "dspark_stages": len(bundle.target_model.dspark.stages),
        "target_taps": list(bundle.target_layer_ids),
        "load_time_s": load_time,
        "resident_parameter_bytes": resident_bytes,
        "mlx_version": mx.__version__,
        "fp32_activations": (
            (os.environ.get("MTPLX_DSV4_FP32_ACTIVATIONS") or "").strip().lower()
            in {"1", "true", "yes", "on"}
        ),
        "guard_window_id": guard["window_id"],
        "source_head": source_commit,
        "prompt": prompt_metadata,
        **_cache_contract(
            bundle,
            requested_span_tokens=requested_span_tokens,
        ),
    }
    mlx_peak_reset_before_arm = _reset_peak_before_arm()
    mlx_active_before_arm = _memory("get_active_memory")
    mlx_peak_before_arm = _memory("get_peak_memory")
    status = 0
    if args.arm == "construct":
        payload = common
    elif args.arm == "one-cycle":
        payload = {
            **common,
            "prompt_tokens": len(prompt_ids),
            "cycle": _first_epoch(bundle, prompt_ids, context),
        }
    elif args.arm == "dspark":
        payload = {
            **common,
            "prompt_tokens": len(prompt_ids),
            "dspark": _dspark(bundle, prompt_ids, args.max_tokens, context),
        }
    elif args.arm == "exact-stream":
        ar = _target_ar(bundle, prompt_ids, args.max_tokens)
        dspark = _dspark(bundle, prompt_ids, args.max_tokens, context)
        quality = _deepseek_quality_gate(dspark["tokens"], ar["tokens"])
        status = 0 if quality["pass"] or not quality["enforced"] else 2
        payload = {
            **common,
            "exact": quality["pass"],
            "quality_gate": quality,
            "ar": ar,
            "dspark": dspark,
        }
    else:
        control_0 = _target_ar(bundle, prompt_ids, args.max_tokens)
        candidate = _dspark(bundle, prompt_ids, args.max_tokens, context)
        control_1 = _target_ar(bundle, prompt_ids, args.max_tokens)
        controls_exact = control_0["tokens"] == control_1["tokens"]
        quality = _deepseek_quality_gate(candidate["tokens"], control_0["tokens"])
        status = (
            0
            if controls_exact and (quality["pass"] or not quality["enforced"])
            else 2
        )
        controls = (control_0["decode_tok_s"] + control_1["decode_tok_s"]) / 2
        payload = {
            **common,
            "exact": controls_exact and quality["pass"],
            "controls_exact": controls_exact,
            "quality_gate": quality,
            "controls_mean_decode_tok_s": controls,
            "candidate_delta_percent": (
                100.0 * (candidate["decode_tok_s"] / controls - 1.0)
                if controls > 0
                else 0.0
            ),
            "arms": {"C0": control_0, "K5": candidate, "C1": control_1},
        }
    payload["memory"] = _finish_memory_receipt(
        mlx_active_after_load_bytes=mlx_active_after_load,
        mlx_peak_after_load_bytes=mlx_peak_after_load,
        process_peak_rss_after_load_bytes=process_peak_rss_after_load,
        mlx_peak_reset_before_arm=mlx_peak_reset_before_arm,
        mlx_active_before_arm_bytes=mlx_active_before_arm,
        mlx_peak_before_arm_bytes=mlx_peak_before_arm,
    )
    _write(payload, args.out)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
