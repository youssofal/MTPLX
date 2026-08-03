#!/usr/bin/env python3
"""Whole-runtime A/B: the alternative Laguna runtime vs the reference lane, B=1.

The reference arm is the 67.4 tok/s path: ``install_from_env`` fusions on top of
``LagunaCompiledLane`` (compiled). The alt arm is ``LagunaAltLane`` under an
``AltConfig`` — the standalone port from ``PORT_LEDGER.md``. Both run in ONE
window off the SAME loaded weights, because cross-window comparison on a loaded
box drifts ~4% (see laguna_lane.py) and only an in-window pairing is honest.

Shape is the canonical compiled-lane shape (ctx 1024 / decode 96 / warmup 8, B=1,
cap 2048) — the shape every ``laguna-compiled-lane-*`` receipt and the 67.4
number were measured at.

Digest equality is the correctness gate: the alt lane is arithmetic-equivalent to
the reference by construction (with no AltConfig flags on) and must stay
token-for-token identical as ported kernels are turned on — a mismatch is a
kernel bug, never a benchmarking artifact. tok/s + peak GiB are the perf axes.

Baseline usage (F0.3), no kernels ported yet::

    run_guarded.py -- python bench/laguna/laguna_alt_ab_bench.py --label baseline

As kernels land, enable them and re-run in one window::

    ... --label d1 --alt-config d1_residual_router
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

# Reuse the reference-lane machinery verbatim so the two arms share load, prompt
# build, prefill shape, timing and digest code — the only new thing here is the
# alt lane.
from laguna_compiled_lane_bench import (  # noqa: E402
    _argmax_next,
    _digest_tokens,
    run_compiled_step_lane,
)


def _alt_config(flags: str) -> Any:
    """Build an AltConfig from a comma-separated flag list (see PORT_LEDGER)."""

    from mtplx.laguna_alt_step import STOCK, AltConfig

    names = [name.strip() for name in flags.split(",") if name.strip()]
    if not names:
        return STOCK
    valid = {f.name for f in __import__("dataclasses").fields(AltConfig)}
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise SystemExit(f"unknown AltConfig flags {unknown}; valid: {sorted(valid)}")
    return AltConfig(**{name: True for name in names})


def run_alt_lane(
    runtime,
    prompts,
    arguments,
    *,
    compiled: bool,
    lane_name: str,
    config_str: str,
    sched: str = "sync",
) -> dict[str, Any]:
    """Eager prefill -> snapshot -> decode via LagunaAltLane under an AltConfig.

    Mirrors ``run_compiled_step_lane`` exactly (same prefill shape, same warmup /
    measured-window split, same peak-memory + digest accounting) so the alt and
    reference numbers are directly comparable — only the lane class, its config,
    and the decode SCHEDULING differ.

    ``sched`` is the LEDGER S1 lever (async-eval decode ladder):
      * ``"sync"``  — ``mx.eval(token)`` every step; the host blocks on the GPU
        each token (the reference behaviour, host-exposed).
      * ``"async"`` — ``mx.async_eval`` the FULL step state every token so the
        host races ahead building step N+1's graph while the GPU runs step N.
        Evaluating the whole state (token + leaves), not just the token, is what
        keeps the graph from growing unbounded. Value-preserving: async_eval only
        changes WHEN the host blocks, so the digest must be identical to sync.
    """

    import mlx.core as mx

    from mtplx.laguna_alt_step import LagunaAltLane

    config = _alt_config(config_str)
    packed = os.environ.get("MTPLX_LAGUNA_PACKED_KV", "0").strip() == "1"

    tight = (
        arguments.context_tokens
        + arguments.decode_tokens
        + arguments.warmup_tokens
        + 16
    )
    cap_env = os.environ.get("MTPLX_LAGUNA_CAP", "").strip()
    cap = int(cap_env) if cap_env else max(2048, tight)
    if cap < tight:
        raise SystemExit(f"MTPLX_LAGUNA_CAP={cap} < required {tight}")

    mx.clear_cache()
    mx.reset_peak_memory()

    cache = runtime.make_cache()
    mx.synchronize()
    logits = runtime.forward_ar(prompts, cache=cache, logits_keep=1)
    token = _argmax_next(logits)
    mx.eval(token)
    mx.synchronize()

    lane = LagunaAltLane(runtime.model, cap, compiled=compiled, config=config, packed_kv=packed)
    lane.seed(cache, token)

    tokens: list[Any] = [token]
    for _ in range(arguments.warmup_tokens):
        token = lane.advance()
        mx.eval(token)
        tokens.append(token)
    mx.synchronize()

    started = time.perf_counter()
    for i in range(arguments.decode_tokens):
        token = lane.advance()
        if sched == "async":
            # Race the host ahead: submit the whole step state non-blocking so
            # the next advance() encodes while this step runs on the GPU.
            mx.async_eval(lane.token, lane.offset, lane.ring_idx, *lane.leaves)
        elif sched == "ladder":
            # The challenge's interval staging (1,7,15,23,31,39 ~= every 8):
            # let several steps' graphs accumulate, submit non-blocking at the
            # interval, so the host has fewer sync points than async-every-step.
            if (i + 1) % 8 == 0 or i == arguments.decode_tokens - 1:
                mx.async_eval(lane.token, lane.offset, lane.ring_idx, *lane.leaves)
        else:  # sync — mx.eval every step (the reference behaviour)
            mx.eval(token)
        tokens.append(token)
    mx.synchronize()
    elapsed = time.perf_counter() - started

    ms_per_step = 1000.0 * elapsed / arguments.decode_tokens
    peak_bytes = int(mx.get_peak_memory())
    digest, _rows = _digest_tokens(tokens)

    result = {
        "lane": lane_name,
        "compiled": bool(compiled),
        "cap": cap,
        "sched": sched,
        "packed_kv": packed,
        "alt_config": [
            f.name
            for f in __import__("dataclasses").fields(config)
            if getattr(config, f.name)
        ],
        "decode_tokens": arguments.decode_tokens,
        "warmup_tokens": arguments.warmup_tokens,
        "ms_per_step": round(ms_per_step, 3),
        "per_request_tokps": round(1000.0 / ms_per_step, 2),
        "peak_bytes": peak_bytes,
        "peak_gib": round(peak_bytes / 1024**3, 2),
        "token_digest": digest,
    }

    del cache, lane
    gc.collect()
    mx.clear_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--context-tokens", type=int, default=1024)
    parser.add_argument("--decode-tokens", type=int, default=96)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument(
        "--alt-configs",
        default="",
        help="semicolon-separated list of alt cells to run against the reference, "
        "each a comma-separated AltConfig flag set (empty = all-stock alt). "
        "e.g. ';d1_residual_router' runs alt-stock AND alt-D1 in ONE window so "
        "the kernel's delta is drift-free.",
    )
    parser.add_argument(
        "--alt-scheds",
        default="sync",
        help="comma-separated decode schedulings for the alt cells (LEDGER S1): "
        "'sync' (mx.eval each step, the reference behaviour) and/or 'async' "
        "(mx.async_eval the full state each step, host races ahead). "
        "'sync,async' A/Bs the async ladder in one window.",
    )
    arguments = parser.parse_args()

    alt_configs = [c.strip() for c in arguments.alt_configs.split(";")]
    alt_scheds = [s.strip() for s in arguments.alt_scheds.split(",") if s.strip()]

    import mlx.core as mx

    from laguna_lane import MODEL_REPO, build_prompts, guard_memory, resolve_model_dir

    print(f"[alt-ab:{arguments.label}] alt_configs={alt_configs!r}", flush=True)

    model_dir = resolve_model_dir()
    start = time.perf_counter()
    from mtplx.runtime import load as runtime_load

    runtime = runtime_load(model_dir, mtp=False)
    mx.eval(runtime.model.parameters())
    print(
        f"[alt-ab:{arguments.label}] loaded {MODEL_REPO} in "
        f"{time.perf_counter() - start:.1f}s",
        flush=True,
    )

    # The reference arm's fusions. This is what makes the reference the 67.4
    # path; the alt lane reads the same installed hooks as its stock fallback for
    # any span it has not yet ported, so an all-stock alt arm must match it.
    from mtplx.models import laguna_fused

    report = laguna_fused.install_from_env(runtime.model)
    print(f"[alt-ab:{arguments.label}] install_from_env: {report}", flush=True)

    guard = guard_memory(1, arguments.context_tokens, arguments.decode_tokens)
    if guard["refused"]:
        print(f"[alt-ab:{arguments.label}] B=1 REFUSED {guard}", flush=True)
        return 1

    prompts = build_prompts(runtime.tokenizer, 1, arguments.context_tokens)

    cells: list[dict[str, Any]] = []
    digests: dict[str, str] = {}

    def _run(name, fn):
        try:
            cell = fn()
        except Exception as exc:  # keep the window alive if one cell errors
            import traceback

            traceback.print_exc()
            cells.append({"lane": name, "error": repr(exc)})
            return
        digests[name] = cell["token_digest"]
        cells.append(cell)
        print(
            f"[alt-ab:{arguments.label}] lane={name} "
            f"{cell['ms_per_step']:.2f} ms/step "
            f"{cell['per_request_tokps']:.1f} tok/s "
            f"peak {cell['peak_gib']:.2f} GiB "
            f"digest={cell['token_digest']}",
            flush=True,
        )

    # Reference: the 67.x install_from_env compiled lane.
    _run(
        "reference",
        lambda: run_compiled_step_lane(
            runtime, prompts, arguments, compiled=True, lane_name="reference"
        ),
    )
    # One alt cell per (config, scheduling), all on the SAME loaded model.
    for config_str in alt_configs:
        for sched in alt_scheds:
            name = f"alt[{config_str or 'stock'}|{sched}]"
            _run(
                name,
                lambda name=name, config_str=config_str, sched=sched: run_alt_lane(
                    runtime,
                    prompts,
                    arguments,
                    compiled=True,
                    lane_name=name,
                    config_str=config_str,
                    sched=sched,
                ),
            )

    # Correctness gate + per-cell delta vs the reference (drift-free, one window).
    ref_digest = digests.get("reference")
    ref_cell = next((c for c in cells if c.get("lane") == "reference"), None)
    digest_match: dict[str, bool] = {}
    for cell in cells:
        name = cell.get("lane")
        if name == "reference" or "token_digest" not in cell:
            continue
        match = cell["token_digest"] == ref_digest
        digest_match[name] = match
        if ref_cell and cell.get("ms_per_step"):
            speedup = ref_cell["ms_per_step"] / cell["ms_per_step"]
            print(
                f"[alt-ab:{arguments.label}] {name}: DIGEST "
                f"{'MATCH' if match else 'MISMATCH!!'} | "
                f"{cell['per_request_tokps']:.1f} vs ref "
                f"{ref_cell['per_request_tokps']:.1f} tok/s ({speedup:.3f}x ms/step)",
                flush=True,
            )

    out = {
        "label": arguments.label,
        "alt_configs": alt_configs,
        "alt_scheds": alt_scheds,
        "env": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("MTPLX_LAGUNA_")
        },
        "context_tokens": arguments.context_tokens,
        "decode_tokens": arguments.decode_tokens,
        "warmup_tokens": arguments.warmup_tokens,
        "cells": cells,
        "digests": digests,
        "digest_match": digest_match,
    }
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = Path(__file__).resolve().parent / (
        f"laguna-alt-ab-{arguments.label}-{stamp}.json"
    )
    path.write_text(json.dumps(out, indent=1))
    print(f"[alt-ab:{arguments.label}] wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
