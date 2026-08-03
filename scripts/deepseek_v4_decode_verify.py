"""Root-cause harness for a degenerate deepseek_v4 generation.

The smoke run (scripts/deepseek_v4_smoke_generate.py) produced a repetition loop.
Two explanations are possible and they need different owners:

  H1  the streaming decode state machine diverges from the one-shot forward at
      real dims (a backend bug in mtplx/models/deepseek_v4.py), or
  H2  the streaming decode is exact and the loop is the model's own greedy
      behaviour on that prompt at this quantisation (not a backend bug).

This script separates them with one measurement and one control, off a single
load:

  GATE A (consistency)  Re-run the smoke run's full token sequence through the
      *one-shot* path (``cache=None``) — the path that is parity-gated against the
      reference in tests/test_deepseek_v4_parity.py — and compare its argmax at
      every generated position against what streaming actually emitted.  Full
      agreement falsifies H1: the cache path reproduces the gated path.

  GATE B (control prompts)  Greedy-generate on prompts whose continuation is
      determinate, so "does this model produce coherent text at all" is answered
      independently of the smoke prompt, which ran out of file to write.

Both gates need the real checkpoint, so this runs in the guarded MLX window.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx


CONTROL_PROMPTS = {
    "factual": "The capital of France is",
    "code_docstring": '''import bisect
from typing import List


def merge_intervals(intervals: List[tuple[int, int]]) -> List[tuple[int, int]]:
    """Merge overlapping half-open intervals and return them sorted by start.

    Intervals that merely touch (``(1, 3)`` and ``(3, 5)``) are merged, because
    the ranges are half-open. The input is not mutated.
    """
''',
}


def _default_model() -> str | None:
    hits = sorted(
        glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/"
                "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/snapshots/*/"
            )
        )
    )
    return hits[0] if hits else None


def _greedy(model, tokenizer, prompt: str, max_tokens: int):
    cache = model.make_cache()
    ids = tokenizer.encode(prompt)
    logits = model(mx.array(ids)[None], cache=cache)
    token = mx.argmax(logits[:, -1], axis=-1)
    mx.eval(token)
    out = [int(token.item())]
    token = token[:, None]
    for _ in range(max_tokens - 1):
        logits = model(token, cache=cache)
        token = mx.argmax(logits[:, -1], axis=-1)
        mx.eval(token)
        out.append(int(token.item()))
        token = token[:, None]
    return ids, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_default_model())
    ap.add_argument(
        "--run-json",
        required=True,
        help="receipt from deepseek_v4_smoke_generate.py to re-check",
    )
    ap.add_argument("--control-tokens", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    model_path = Path(os.path.expanduser(args.model)).resolve()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_lm.utils import load_config

    from mtplx.runtime import _load_base_model

    config = load_config(model_path)
    t0 = time.perf_counter()
    model, tokenizer = _load_base_model(model_path, config)
    mx.eval(model.parameters())
    print(f"[verify] loaded in {time.perf_counter() - t0:.1f}s")
    sys.stdout.flush()

    receipt = json.loads(Path(args.run_json).read_text())
    prompt_ids = tokenizer.encode(receipt["prompt"])
    streamed = receipt["generated_token_ids"]
    assert len(prompt_ids) == receipt["prompt_tokens"], "tokenizer drift"

    # ---- GATE A: one-shot forward over prompt + generated[:-1] --------------
    sequence = list(prompt_ids) + list(streamed[:-1])
    print(f"[verify] GATE A: one-shot forward over {len(sequence)} tokens "
          f"(cache=None, the parity-gated path)")
    sys.stdout.flush()
    base = len(prompt_ids) - 1
    t0 = time.perf_counter()
    logits = model(mx.array(sequence)[None])
    predicted = mx.argmax(logits[0], axis=-1)
    mx.eval(predicted)
    one_shot_seconds = time.perf_counter() - t0
    predicted = [int(v) for v in predicted.tolist()]

    # How saturated was the loop?  Same forward, no second pass.
    row = logits[0, base:].astype(mx.float32)
    top2 = mx.sort(mx.topk(row, 2, axis=-1), axis=-1)  # ascending: [second, top]
    mx.eval(top2)
    margins = [float(r[1] - r[0]) for r in top2.tolist()][:16]
    del logits, row, top2

    one_shot_next = predicted[base : base + len(streamed)]
    agree = [a == b for a, b in zip(one_shot_next, streamed)]
    n_agree = sum(agree)
    first_divergence = agree.index(False) if not all(agree) else None
    print(f"[verify] one-shot forward {one_shot_seconds:.2f}s")
    print(f"[verify] GATE A: {n_agree}/{len(streamed)} streamed tokens match "
          f"the one-shot argmax")
    if first_divergence is not None:
        i = first_divergence
        print(f"[verify]   first divergence at generated index {i}: "
              f"streamed={streamed[i]} one_shot={one_shot_next[i]}")
    print(f"[verify] GATE A: {'PASS (decode == one-shot)' if n_agree == len(streamed) else 'FAIL (decode diverges)'}")
    print(f"[verify] top1-top2 logit margin, first 16 generated positions: "
          f"{[round(m, 2) for m in margins]}")
    sys.stdout.flush()

    # ---- GATE B: control prompts -------------------------------------------
    controls = {}
    for name, prompt in CONTROL_PROMPTS.items():
        print(f"\n[verify] GATE B control {name!r} "
              f"({len(tokenizer.encode(prompt))} prompt tokens)")
        ids, out = _greedy(model, tokenizer, prompt, args.control_tokens)
        text = tokenizer.decode(out)
        unique = len(set(out))
        print(f"--- generated ({len(out)} tokens, {unique} unique) ---")
        print(text)
        print("--- end ---")
        sys.stdout.flush()
        controls[name] = {
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "generated_token_ids": out,
            "generated_text": text,
            "unique_tokens": unique,
        }

    result = {
        "harness": "scripts/deepseek_v4_decode_verify.py",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": ["python", *sys.argv],
        "model_path": str(model_path),
        "source_run": str(args.run_json),
        "gate_a": {
            "description": "streaming decode argmax vs one-shot (cache=None) argmax",
            "tokens_compared": len(streamed),
            "tokens_agreeing": n_agree,
            "first_divergence_index": first_divergence,
            "one_shot_seconds": one_shot_seconds,
            "pass": n_agree == len(streamed),
            "top1_minus_top2_margins_first16": margins,
        },
        "gate_b": controls,
    }
    if args.out:
        stem = Path(args.out)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
        print(f"\nreceipt: {stem.with_suffix('.json')}")

    return 0 if n_agree == len(streamed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
