"""Real-weights generation smoke test for the MTPLX deepseek_v4 backend.

Loads a real ``mlx-community/DeepSeek-V4-Flash-*`` checkpoint through the *mtplx*
load path (``mtplx.runtime._load_base_model`` -> ``mlx_lm.utils.load_model`` with
``get_model_classes`` resolving to ``mtplx.models.deepseek_v4``), runs one greedy
completion off ``Model.make_cache()``, and reports load time, prefill tok/s,
decode tok/s and peak memory.

This is a smoke harness, not a benchmark suite: one prompt per run, temp 0.
Several ``--prompt-file``/``--out`` pairs may be passed, in which case every run
executes off a single load — the 2-bit checkpoint is ~90 GiB, so a second load
is minutes of wall clock and another full pass over the wired-memory budget.

Context budget: the ratio-4 :class:`Indexer` top-k filter is wired, so the
backend is correct past ``index_topk * ratio`` (~2048) tokens of context, which
is where dense-over-compressed stops being the reference's computation.  What
still bounds a run is memory, not correctness: the indexer forms a
``[b, s, index_n_heads, n_comp]`` fp32 score tensor, and attention a
``[b, n_heads, s, n_win + n_comp]`` score tensor, both quadratic in context.
``--max-context`` (default 8192) is that memory ceiling, not a math one; the
long-context score-tensor cost is a known open item.  ``--prefill-chunk`` keeps
the prefill half of it bounded by feeding the prompt to the live cache in
chunks (gated in tests/test_deepseek_v4_indexer.py::test_sparse_chunked_prefill_
then_decode), which is also how the serve path feeds long prompts.

``--gate-a`` replays prompt + generated tokens through the *one-shot*
(``cache=None``) forward — the path parity-gated against the reference in
tests/test_deepseek_v4_parity.py — and counts argmax agreement at every
generated position.  Full agreement gates the streaming state machine at real
dims.  The one-shot forward is a single call by definition, so it cannot be
chunked; it is the memory high-water mark of a long run.

Reading GATE A past ``index_topk * ratio``: below that threshold the two paths
agree exactly (dense regime, 128/128 measured).  Above it the selection is
*discrete*, and the two paths reduce over different shapes — so a compressed
row sitting within float noise of the top-k cut can be selected by one path and
not the other.  Measured at 3.4K context (bench/deepseek-v4/sparse-2bitdq-
20260731-probe.json): the rank-512-vs-513 score gap is 2e-4..1e-2 on all 21
ratio-4 layers while the inter-path score noise is 4e-3..2.9, so 18/21 layers
select a slightly different row set and the logit rows spread by mean 0.11.
A generated position whose top1-top2 margin is under that spread can therefore
flip without anything being wrong.  Note the one-shot forward's own length
changes ``n_comp`` too, so it is not a fixed oracle in this regime — two
one-shot forwards over different totals disagree at the same position.  Judge a
sparse GATE A by *where* it diverges (an isolated near-tie that does not
propagate) rather than by an exact count.

MUST run inside the box's serialized MLX window (bench/laguna/run_guarded.py) —
the 2-bit checkpoint is ~90 GiB and does not fit beside the served model.

Usage:
  python scripts/deepseek_v4_smoke_generate.py \
      --model ~/.cache/huggingface/hub/models--mlx-community--DeepSeek-V4-Flash-2bit-DQ/snapshots/<rev> \
      --max-tokens 128 --out bench/deepseek-v4/smoke-2bitdq-YYYYMMDD
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import sys
import time
from pathlib import Path

import mlx.core as mx


DEFAULT_PROMPT = '''"""Rolling-window rate limiter used by the ingest workers."""

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Allow at most ``max_events`` in any ``window_seconds`` sliding window.

    The limiter is intentionally not thread-safe; each ingest worker owns one.
    Timestamps are monotonic so a clock adjustment cannot open the gate early.
    """

    max_events: int
    window_seconds: float
    _events: deque = field(default_factory=deque)

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0] <= cutoff:
            self._events.popleft()

    def allow(self) -> bool:
        """Record and admit one event, or return False if the window is full."""
        now = time.monotonic()
        self._evict(now)
        if len(self._events) >= self.max_events:
            return False
        self._events.append(now)
        return True

    def retry_after(self) -> float:
        """Seconds until the next event would be admitted (0.0 if admissible)."""
        now = time.monotonic()
        self._evict(now)
        if len(self._events) < self.max_events:
            return 0.0
        return self._events[0] + self.window_seconds - now

    def reset(self) -> None:
'''

# Memory ceiling, not a math one: the indexer/attention score tensors are
# quadratic in context (see the module docstring).  The pre-indexer harness
# capped at 2048 because the ratio-4 filter did not exist and dense-over-
# compressed silently stopped being the reference computation past that point;
# that reason is gone, this one is not.
_CONTEXT_GUARD_TOKENS = 8192


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


def _peak_bytes() -> int:
    for getter in ("get_peak_memory",):
        fn = getattr(mx, getter, None)
        if callable(fn):
            return int(fn())
    fn = getattr(getattr(mx, "metal", None), "get_peak_memory", None)
    return int(fn()) if callable(fn) else -1


def _active_bytes() -> int:
    fn = getattr(mx, "get_active_memory", None)
    if callable(fn):
        return int(fn())
    fn = getattr(getattr(mx, "metal", None), "get_active_memory", None)
    return int(fn()) if callable(fn) else -1


def _reset_peak() -> None:
    fn = getattr(mx, "reset_peak_memory", None)
    if callable(fn):
        fn()


def _gib(n: int) -> float:
    return n / (1024**3)


def _eval_cache(cache, *extra) -> None:
    """Force the graph of a chunked prefill so it does not span every chunk."""
    live = [a for layer in cache for a in layer.state if a is not None]
    mx.eval(*extra, *live)


def _run_one(
    *,
    model,
    tokenizer,
    config,
    quant,
    overrides,
    model_path: Path,
    prompt: str,
    max_tokens: int,
    prefill_chunk: int,
    max_context: int,
    gate_a: bool,
    out_stem: str | None,
    label: str,
    load_seconds: float,
    after_load_active: int,
) -> dict:
    prompt_ids = tokenizer.encode(prompt)
    n_prompt = len(prompt_ids)
    total_context = n_prompt + max_tokens
    print(f"[smoke] {label}prompt tokens: {n_prompt}  "
          f"new tokens: {max_tokens}  total context: {total_context}")
    if total_context > max_context:
        sys.exit(
            f"total context {total_context} exceeds --max-context "
            f"({max_context}); raise it deliberately, after checking the "
            f"quadratic score-tensor cost against the wired-memory budget"
        )
    sys.stdout.flush()

    _reset_peak()
    cache = model.make_cache()
    ids = mx.array(prompt_ids)[None]

    # ---- prefill ----------------------------------------------------------
    t0 = time.perf_counter()
    if prefill_chunk and n_prompt > prefill_chunk:
        n_chunks = 0
        for start in range(0, n_prompt, prefill_chunk):
            logits = model(ids[:, start : start + prefill_chunk], cache=cache)
            n_chunks += 1
            if start + prefill_chunk < n_prompt:
                _eval_cache(cache, logits)
                del logits
    else:
        n_chunks = 1
        logits = model(ids, cache=cache)
    last = logits[:, -1]
    token = mx.argmax(last, axis=-1)
    mx.eval(token)
    prefill_seconds = time.perf_counter() - t0
    first_id = int(token.item())
    print(f"[smoke] prefill {n_prompt} tok in {prefill_seconds:.2f}s = "
          f"{n_prompt / prefill_seconds:.1f} tok/s   first token id={first_id}"
          f"   chunks={n_chunks}")
    print(f"[smoke] after prefill: active={_gib(_active_bytes()):.2f} GiB  "
          f"peak={_gib(_peak_bytes()):.2f} GiB")
    sys.stdout.flush()

    ln = logits[0, -1].astype(mx.float32)
    mx.eval(ln)
    finite = bool(mx.all(mx.isfinite(ln)).item())
    spread = float(mx.std(ln).item())
    print(f"[smoke] first-token logits finite={finite} std={spread:.4f}")
    del logits, last, ln

    eos_ids = set()
    for attribute in ("eos_token_ids", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int):
            eos_ids.add(value)
        elif isinstance(value, (list, tuple, set)):
            eos_ids |= {int(v) for v in value}

    # ---- decode -----------------------------------------------------------
    generated = [first_id]
    step_seconds: list[float] = []
    printed = 0
    text = ""
    stopped_on_eos = first_id in eos_ids
    token = token[:, None]
    if not stopped_on_eos:
        for _ in range(max_tokens - 1):
            t0 = time.perf_counter()
            logits = model(token, cache=cache)
            token = mx.argmax(logits[:, -1], axis=-1)
            mx.eval(token)
            step_seconds.append(time.perf_counter() - t0)
            next_id = int(token.item())
            token = token[:, None]
            generated.append(next_id)
            if next_id in eos_ids:
                stopped_on_eos = True
                break
            # streamed print, deliberately outside the timed region
            text = tokenizer.decode(generated)
            if len(text) > printed:
                sys.stdout.write(text[printed:])
                sys.stdout.flush()
                printed = len(text)

    text = tokenizer.decode(generated)
    if len(text) > printed:
        sys.stdout.write(text[printed:])
    sys.stdout.write("\n")
    sys.stdout.flush()

    decode_seconds = sum(step_seconds)
    decode_tps = (len(step_seconds) / decode_seconds) if decode_seconds else 0.0
    peak = _peak_bytes()

    print("\n=== SMOKE SUMMARY ===")
    print(f"label            : {label.strip() or '(single run)'}")
    print(f"load             : {load_seconds:.1f} s")
    print(f"prefill          : {n_prompt} tok / {prefill_seconds:.2f} s = "
          f"{n_prompt / prefill_seconds:.2f} tok/s  (chunk={prefill_chunk or 0})")
    print(f"decode           : {len(step_seconds)} tok / {decode_seconds:.2f} s = "
          f"{decode_tps:.3f} tok/s  ({decode_seconds / max(len(step_seconds), 1):.3f} s/tok)")
    print(f"tokens generated : {len(generated)} (eos hit: {stopped_on_eos})")
    print(f"peak memory      : {_gib(peak):.2f} GiB")
    print(f"active memory    : {_gib(_active_bytes()):.2f} GiB")
    print(f"logits finite    : {finite}   std={spread:.4f}")
    sys.stdout.flush()

    # ---- GATE A: streaming decode vs the one-shot (parity-gated) forward ---
    gate = None
    if gate_a:
        del cache
        clear = getattr(mx, "clear_cache", None)
        if callable(clear):
            clear()
        _reset_peak()
        sequence = list(prompt_ids) + list(generated[:-1])
        print(f"\n[smoke] GATE A: one-shot forward over {len(sequence)} tokens "
              f"(cache=None, the parity-gated path)")
        sys.stdout.flush()
        base = n_prompt - 1
        t0 = time.perf_counter()
        one_shot_logits = model(mx.array(sequence)[None])
        predicted = mx.argmax(one_shot_logits[0], axis=-1)
        mx.eval(predicted)
        one_shot_seconds = time.perf_counter() - t0
        gate_peak = _peak_bytes()
        del one_shot_logits
        predicted = [int(v) for v in predicted.tolist()]
        one_shot_next = predicted[base : base + len(generated)]
        agree = [a == b for a, b in zip(one_shot_next, generated)]
        n_agree = sum(agree)
        first_divergence = agree.index(False) if not all(agree) else None
        print(f"[smoke] one-shot forward {one_shot_seconds:.2f}s  "
              f"peak={_gib(gate_peak):.2f} GiB")
        print(f"[smoke] GATE A: {n_agree}/{len(generated)} streamed tokens match "
              f"the one-shot argmax")
        if first_divergence is not None:
            i = first_divergence
            print(f"[smoke]   first divergence at generated index {i}: "
                  f"streamed={generated[i]} one_shot={one_shot_next[i]}")
        print(f"[smoke] GATE A: "
              f"{'PASS (decode == one-shot)' if n_agree == len(generated) else 'FAIL (decode diverges)'}")
        sys.stdout.flush()
        gate = {
            "description": "streaming decode argmax vs one-shot (cache=None) argmax",
            "tokens_compared": len(generated),
            "tokens_agreeing": n_agree,
            "first_divergence_index": first_divergence,
            "one_shot_tokens": len(sequence),
            "one_shot_seconds": one_shot_seconds,
            "one_shot_peak_gib": _gib(gate_peak),
            "pass": n_agree == len(generated),
        }
        peak = max(peak, gate_peak)

    receipt = {
        "harness": "scripts/deepseek_v4_smoke_generate.py",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": ["python", *sys.argv],
        "host": {
            "platform": platform.platform(),
            "mlx_version": mx.__version__,
            "python": sys.version.split()[0],
        },
        "model_path": str(model_path),
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "quantization": {
            "default_bits": quant.get("bits"),
            "default_group_size": quant.get("group_size"),
            "default_mode": quant.get("mode"),
            "per_path_overrides": len(overrides),
        },
        "sampling": {"greedy": True, "temperature": 0.0},
        "prompt": prompt,
        "prompt_tokens": n_prompt,
        "max_tokens": max_tokens,
        "prefill_chunk": prefill_chunk or 0,
        "prefill_chunks": n_chunks,
        "generated_token_ids": generated,
        "generated_text": text,
        "stopped_on_eos": stopped_on_eos,
        "first_token_logits": {"finite": finite, "std": spread},
        "gate_a": gate,
        "timings": {
            "load_seconds": load_seconds,
            "prefill_seconds": prefill_seconds,
            "prefill_tokens_per_second": n_prompt / prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens": len(step_seconds),
            "decode_tokens_per_second": decode_tps,
            "decode_step_seconds": step_seconds,
        },
        "memory": {
            "peak_bytes": peak,
            "peak_gib": _gib(peak),
            "active_after_load_bytes": after_load_active,
            "active_after_load_gib": _gib(after_load_active),
            "active_end_gib": _gib(_active_bytes()),
        },
    }

    if out_stem:
        stem = Path(out_stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        stem.with_suffix(".json").write_text(json.dumps(receipt, indent=2))
        stem.with_suffix(".txt").write_text(
            f"PROMPT ({n_prompt} tokens)\n{'=' * 72}\n{prompt}\n"
            f"{'=' * 72}\nGENERATED ({len(generated)} tokens, greedy)\n"
            f"{'=' * 72}\n{text}\n"
        )
        print(f"receipts         : {stem.with_suffix('.json')}")
        print(f"                   {stem.with_suffix('.txt')}")
        sys.stdout.flush()

    return receipt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=_default_model())
    ap.add_argument(
        "--prompt-file",
        nargs="+",
        default=None,
        help="one or more prompt files; every run executes off a single load",
    )
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument(
        "--prefill-chunk",
        nargs="+",
        type=int,
        default=[0],
        help="feed the prompt to the cache in chunks of this many tokens "
             "(0 = one call, the shortest path); one value, or one per prompt",
    )
    ap.add_argument(
        "--max-context",
        type=int,
        default=_CONTEXT_GUARD_TOKENS,
        help="refuse prompt+new tokens above this; a memory ceiling, not a "
             "correctness one (see the module docstring)",
    )
    ap.add_argument(
        "--gate-a",
        action="store_true",
        help="replay each run through the one-shot (cache=None) forward and "
             "count argmax agreement over the generated positions",
    )
    ap.add_argument(
        "--out",
        nargs="+",
        default=None,
        help="receipt path stem(s); writes <stem>.json and <stem>.txt",
    )
    args = ap.parse_args()
    if not args.model:
        sys.exit("no model path; pass --model")
    model_path = Path(os.path.expanduser(args.model)).resolve()

    prompts: list[str]
    if args.prompt_file:
        prompts = [Path(p).read_text() for p in args.prompt_file]
    else:
        prompts = [DEFAULT_PROMPT]

    outs: list[str | None]
    if args.out:
        if len(args.out) != len(prompts):
            sys.exit(
                f"--out has {len(args.out)} stems but there are "
                f"{len(prompts)} prompts; pass one stem per prompt"
            )
        outs = list(args.out)
    else:
        outs = [None] * len(prompts)

    chunks = list(args.prefill_chunk)
    if len(chunks) == 1:
        chunks = chunks * len(prompts)
    if len(chunks) != len(prompts):
        sys.exit(
            f"--prefill-chunk has {len(chunks)} values but there are "
            f"{len(prompts)} prompts; pass one value, or one per prompt"
        )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mlx_lm.utils import load_config

    from mtplx.runtime import _load_base_model

    config = load_config(model_path)
    print(f"[smoke] model      : {model_path}")
    print(f"[smoke] model_type : {config.get('model_type')}  "
          f"layers={config.get('num_hidden_layers')}")
    quant = config.get("quantization") or {}
    overrides = [k for k in quant if k not in ("group_size", "bits", "mode")]
    print(f"[smoke] quantization: default bits={quant.get('bits')} "
          f"group_size={quant.get('group_size')} mode={quant.get('mode')} "
          f"per-path overrides={len(overrides)}")
    print(f"[smoke] runs       : {len(prompts)}  max_context={args.max_context}  "
          f"gate_a={args.gate_a}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    model, tokenizer = _load_base_model(model_path, config)
    mx.eval(model.parameters())
    load_seconds = time.perf_counter() - t0
    after_load_active = _active_bytes()
    print(f"[smoke] loaded in {load_seconds:.1f}s  "
          f"active={_gib(after_load_active):.2f} GiB  "
          f"peak={_gib(_peak_bytes()):.2f} GiB")
    sys.stdout.flush()

    status = 0
    for i, (prompt, out_stem, chunk) in enumerate(zip(prompts, outs, chunks)):
        label = f"[run {i + 1}/{len(prompts)}] " if len(prompts) > 1 else ""
        if label:
            print(f"\n{'#' * 72}\n# {label.strip()}  out={out_stem}\n{'#' * 72}")
            sys.stdout.flush()
        receipt = _run_one(
            model=model,
            tokenizer=tokenizer,
            config=config,
            quant=quant,
            overrides=overrides,
            model_path=model_path,
            prompt=prompt,
            max_tokens=args.max_tokens,
            prefill_chunk=chunk,
            max_context=args.max_context,
            gate_a=args.gate_a,
            out_stem=out_stem,
            label=label,
            load_seconds=load_seconds,
            after_load_active=after_load_active,
        )
        if not receipt["first_token_logits"]["finite"]:
            status = 1
        if receipt["gate_a"] is not None and not receipt["gate_a"]["pass"]:
            status = 1
        clear = getattr(mx, "clear_cache", None)
        if callable(clear):
            clear()

    return status


if __name__ == "__main__":
    raise SystemExit(main())
