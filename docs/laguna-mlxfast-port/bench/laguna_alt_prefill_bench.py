#!/usr/bin/env python3
"""Prefill A/B: the alt prefill forward vs the reference eager prefill, B=1.

Reference = `runtime.model(prompts, cache, logits_keep=1)` (the eager LagunaModel
forward + head, with install_from_env fusions). Alt = `alt_prefill_forward(...)` +
head, under an AltConfig. Same prompt (ctx tokens), same last-token argmax digest,
prefill tok/s = context_tokens / mean forward seconds. One window, all cells on the
same loaded model, so the numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _alt_config(flags: str) -> Any:
    from mtplx.laguna_alt_step import STOCK, AltConfig

    names = [n.strip() for n in flags.split(",") if n.strip()]
    if not names:
        return STOCK
    valid = {f.name for f in __import__("dataclasses").fields(AltConfig)}
    unknown = [n for n in names if n not in valid]
    if unknown:
        raise SystemExit(f"unknown AltConfig flags {unknown}")
    return AltConfig(**{n: True for n in names})


def _first_token(logits):
    import mlx.core as mx

    return int(mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32).item())


def run_reference(runtime, prompts, arguments) -> dict[str, Any]:
    import mlx.core as mx

    ctx = arguments.context_tokens
    # warmup
    cache = runtime.make_cache()
    logits = runtime.model(prompts, cache=cache, logits_keep=1)
    mx.eval(logits)
    mx.synchronize()
    del cache
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

    started = time.perf_counter()
    for _ in range(arguments.reps):
        cache = runtime.make_cache()
        logits = runtime.model(prompts, cache=cache, logits_keep=1)
        mx.eval(logits)
        del cache
    mx.synchronize()
    elapsed = (time.perf_counter() - started) / arguments.reps
    tok = _first_token(logits)
    peak = int(mx.get_peak_memory())
    gc.collect()
    mx.clear_cache()
    return {
        "lane": "reference",
        "prefill_seconds": round(elapsed, 4),
        "prefill_tokps": round(ctx / elapsed, 1),
        "peak_gib": round(peak / 1024**3, 2),
        "first_token": tok,
    }


def run_alt(runtime, prompts, arguments, *, config_str, lane_name) -> dict[str, Any]:
    import mlx.core as mx

    from mtplx.laguna_alt_step import alt_prefill_forward

    config = _alt_config(config_str)
    ctx = arguments.context_tokens
    model = runtime.model

    def _forward():
        cache = runtime.make_cache()
        hidden = alt_prefill_forward(model, prompts, cache, config=config)
        logits = model.lm_head(hidden[:, -1:, :])
        return logits

    # warmup
    logits = _forward()
    mx.eval(logits)
    mx.synchronize()
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()

    started = time.perf_counter()
    for _ in range(arguments.reps):
        logits = _forward()
        mx.eval(logits)
    mx.synchronize()
    elapsed = (time.perf_counter() - started) / arguments.reps
    tok = _first_token(logits)
    peak = int(mx.get_peak_memory())
    gc.collect()
    mx.clear_cache()
    return {
        "lane": lane_name,
        "alt_config": config_str or "stock",
        "prefill_seconds": round(elapsed, 4),
        "prefill_tokps": round(ctx / elapsed, 1),
        "peak_gib": round(peak / 1024**3, 2),
        "first_token": tok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=96)  # accepted, unused
    parser.add_argument("--warmup-tokens", type=int, default=8)  # accepted, unused
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--alt-configs", default="")
    parser.add_argument("--alt-scheds", default="")  # accepted, unused (prefill = one pass)
    arguments = parser.parse_args()

    alt_configs = [c.strip() for c in arguments.alt_configs.split(";")]

    import mlx.core as mx
    from laguna_lane import MODEL_REPO, build_prompts, guard_memory, resolve_model_dir

    print(f"[prefill:{arguments.label}] alt_configs={alt_configs!r} reps={arguments.reps}", flush=True)
    model_dir = resolve_model_dir()
    from mtplx.runtime import load as runtime_load

    runtime = runtime_load(model_dir, mtp=False)
    mx.eval(runtime.model.parameters())
    from mtplx.models import laguna_fused

    report = laguna_fused.install_from_env(runtime.model)
    print(f"[prefill:{arguments.label}] install_from_env: {report}", flush=True)

    guard = guard_memory(1, arguments.context_tokens, arguments.decode_tokens)
    if guard["refused"]:
        print(f"[prefill:{arguments.label}] REFUSED {guard}", flush=True)
        return 1

    prompts = build_prompts(runtime.tokenizer, 1, arguments.context_tokens)

    cells: list[dict[str, Any]] = []

    def _run(name, fn):
        try:
            cell = fn()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            cells.append({"lane": name, "error": repr(exc)})
            return
        cells.append(cell)
        print(
            f"[prefill:{arguments.label}] {name}: {cell['prefill_tokps']} tok/s "
            f"({cell['prefill_seconds']}s) peak {cell['peak_gib']} GiB "
            f"first_tok={cell['first_token']}",
            flush=True,
        )

    _run("reference", lambda: run_reference(runtime, prompts, arguments))
    for config_str in alt_configs:
        name = f"alt[{config_str or 'stock'}]"
        _run(name, lambda name=name, cs=config_str: run_alt(runtime, prompts, arguments, config_str=cs, lane_name=name))

    ref = next((c for c in cells if c.get("lane") == "reference"), None)
    for cell in cells:
        if cell.get("lane") == "reference" or "first_token" not in cell:
            continue
        tok_match = ref is not None and cell["first_token"] == ref["first_token"]
        speedup = ref["prefill_tokps"] and cell["prefill_tokps"] / ref["prefill_tokps"]
        print(
            f"[prefill:{arguments.label}] {cell['lane']}: TOKEN "
            f"{'MATCH' if tok_match else 'MISMATCH!!'} | {cell['prefill_tokps']} vs ref "
            f"{ref['prefill_tokps']} tok/s ({speedup:.3f}x)",
            flush=True,
        )

    out = {
        "label": arguments.label,
        "mode": "prefill",
        "alt_configs": alt_configs,
        "context_tokens": arguments.context_tokens,
        "reps": arguments.reps,
        "env": {k: v for k, v in os.environ.items() if k.startswith("MTPLX_LAGUNA_")},
        "cells": cells,
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(__file__).resolve().parent / f"laguna-alt-prefill-{arguments.label}-{stamp}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"[prefill:{arguments.label}] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
