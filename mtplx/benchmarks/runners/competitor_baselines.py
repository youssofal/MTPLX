"""Same-harness runners for external speculative baselines."""

from __future__ import annotations

import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mtplx.benchmarks.schema import load_prompt_suite
from mtplx.benchmarks.validators.basic import (
    validate_json_text,
    validate_no_degenerate_loop,
)


def run_dflash_mlx_baseline(
    model_path: Path | str,
    draft_model: str,
    *,
    max_tokens: int = 1024,
    block_size: int = 8,
) -> dict[str, Any]:
    from types import SimpleNamespace

    from mtplx.benchmarks.runners.dflash2_depth_sweep import run_cli_sweep

    return run_cli_sweep(
        SimpleNamespace(
            model=str(model_path),
            draft_model=str(draft_model),
            widths=str(block_size),
            repetitions=1,
        ),
        token_count=max_tokens,
    )


def run_ddtree_mlx_baseline(
    model_path: Path | str,
    draft_model: str,
    prompt_suite: Path | str,
    *,
    ddtree_source: Path | str = "REFERENCES:TOOLS/ddtree-mlx",
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    max_tokens: int = 96,
    tree_budget: int = 4,
    limit: int | None = None,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    _add_source_path(ddtree_source)

    try:
        from dflash_mlx.generate import get_stop_token_ids, load_runtime_components
        from ddtree_mlx.runtime import generate_ddtree_once
    except Exception as exc:  # pragma: no cover - environment/reporting path
        return _error_result(
            "ddtree_mlx",
            model_path,
            draft_model,
            prompt_suite,
            "import_failed",
            exc,
        )

    try:
        target_model, tokenizer, draft, draft_ref = load_runtime_components(
            model_ref=str(model_path),
            draft_ref=str(draft_model),
        )
        if draft is None:
            raise RuntimeError(
                "DDTree load_runtime_components returned no draft model; "
                "the DFlash drafter is likely unavailable or gated."
            )
        stop_ids = get_stop_token_ids(tokenizer)
    except Exception as exc:
        return _error_result(
            "ddtree_mlx",
            model_path,
            draft_model,
            prompt_suite,
            "load_failed",
            exc,
        )

    rows = []
    prompts = load_prompt_suite(prompt_suite)
    if limit is not None:
        prompts = prompts[:limit]

    for case in prompts:
        messages = case.messages or [{"role": "user", "content": case.prompt}]
        kwargs: dict[str, Any] = {}
        if enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        prompt_tokens = list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **kwargs,
            )
        )
        try:
            result = generate_ddtree_once(
                target_model=target_model,
                draft_model=draft,
                tokenizer=tokenizer,
                prompt_tokens=prompt_tokens,
                max_new_tokens=min(max_tokens, case.max_tokens),
                tree_budget=tree_budget,
                stop_token_ids=stop_ids,
            )
        except Exception as exc:
            rows.append(
                {
                    "prompt_id": case.id,
                    "category": case.category,
                    "prompt_sha256": case.prompt_sha256,
                    "error": repr(exc),
                    "validations": [],
                }
            )
            continue

        text = tokenizer.decode(result.get("generated_token_ids", []))
        validations = [asdict(validate_no_degenerate_loop(text))]
        if case.category == "json_tool":
            validations.append(asdict(validate_json_text(text.strip())))
        rows.append(
            {
                "prompt_id": case.id,
                "category": case.category,
                "prompt_sha256": case.prompt_sha256,
                "generated_tokens": result.get("generation_tokens", 0),
                "tok_s": result.get("tokens_per_second", 0.0),
                "avg_acceptance": result.get("avg_acceptance"),
                "fast_path_ratio": result.get("fast_path_ratio"),
                "tree_aware_linear": result.get("tree_aware_linear", False),
                "validations": validations,
                "text": text,
                "raw": result,
            }
        )

    validations = [v for row in rows for v in row.get("validations", [])]
    successful = [row for row in rows if row.get("tok_s") is not None]
    return {
        "backend": "ddtree_mlx",
        "model_path": str(model_path),
        "draft_model": str(draft_ref),
        "prompt_suite": str(prompt_suite),
        "sampler": {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "note": "DDTree baseline depends on dflash_mlx runtime semantics.",
        },
        "max_tokens": max_tokens,
        "tree_budget": tree_budget,
        "enable_thinking": enable_thinking,
        "rows": rows,
        "summary": {
            "prompts": len(rows),
            "successful_prompts": len(successful),
            "generated_tokens": sum(int(row.get("generated_tokens") or 0) for row in rows),
            "mean_tok_s": (
                statistics.mean([float(row["tok_s"]) for row in successful])
                if successful
                else 0.0
            ),
            "mean_acceptance": _mean_present(row.get("avg_acceptance") for row in successful),
            "validations_passed": sum(1 for v in validations if v["passed"]),
            "validations_total": len(validations),
        },
    }


def write_competitor_result(path: Path | str, result: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True))


def _add_source_path(source: Path | str) -> None:
    path = str(Path(source).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _mean_present(values) -> float | None:
    present = [float(v) for v in values if v is not None]
    return statistics.mean(present) if present else None


def _error_result(
    backend: str,
    model_path: Path | str,
    draft_model: str,
    prompt_suite: Path | str,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "model_path": str(model_path),
        "draft_model": str(draft_model),
        "prompt_suite": str(prompt_suite),
        "error": {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "repr": repr(exc),
        },
        "rows": [],
        "summary": {
            "prompts": 0,
            "successful_prompts": 0,
            "generated_tokens": 0,
            "mean_tok_s": 0.0,
            "validations_passed": 0,
            "validations_total": 0,
        },
    }
