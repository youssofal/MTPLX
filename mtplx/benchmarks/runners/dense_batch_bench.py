"""GPU bench + parity gate for the dense batched-MTP cohort driver.

Runs ``generate_dense_mtp_batch`` on a real runtime at a grid of (batch, depth)
points, reporting aggregate decode tok/s, tokens/cycle, acceptance-by-depth,
and (optionally) the per-stream sha parity gate.

WHAT THE PARITY GATE TESTS, AND WHAT IT DOES NOT
-------------------------------------------------
The gate pins ``cohort_slots``, so it compares a row batched among real prompts
against that row alone **in a cohort of the same slot count, padded with dummy
rows**. Both arms therefore run at IDENTICAL tensor geometry and differ only in
the other rows' CONTENTS.

**It tests per-row forward independence at fixed geometry** -- that a row's
output does not depend on who its neighbours are. That property is real, it is
what the fixed-shape discipline below is for, and it has since been confirmed
on real weights.

**It does NOT test shape-invariance**, and must not be cited for it. A row
decoded in a cohort is not byte-identical to that row decoded in a genuinely
one-row run, because float reduction order changes with matmul shape. That was
measured directly (Qwen3.8-27B, identical prompt lengths, identical KV
capacity, varying only the row count: all four rows diverged from their solo
runs). This gate passing says nothing about that case, in either direction.

Usage (from the repo venv on the GPU box):

    .venv/bin/python -m mtplx.benchmarks.runners.dense_batch_bench \
        --model ~/.mtplx/models/Qwen3.8-27B-... \
        --prompt-file /path/to/agentic-24k.txt \
        --batches 1,2,4 --depths 3 --max-new 512 \
        --out results/dense-batch/b1-24k.json

``--prompt-file`` is raw text, encoded with the model's chat template
(``enable_thinking`` honoured via --thinking). ``--prompt-tokens-file`` takes a
JSON list of token ids instead (exact-prompt reuse across arms).
``--vary-suffix`` appends one distinct extra user token per row so cohort rows
are genuinely different streams (required for a meaningful parity gate; without
it greedy rows are identical by construction).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _encode_prompt(
    tokenizer: Any, text: str, thinking: str | None, effort: str | None = None
) -> list[int]:
    messages = [{"role": "user", "content": text}]
    kwargs: dict[str, Any] = {"add_generation_prompt": True}
    if thinking is not None:
        kwargs["enable_thinking"] = thinking != "off"
    if effort:
        kwargs["reasoning_effort"] = effort
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(t) for t in ids]


def _stop_ids(tokenizer: Any) -> set[int]:
    stop: set[int] = set()
    eos = getattr(tokenizer, "eos_token_ids", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    if isinstance(eos, int):
        stop.add(int(eos))
    elif eos:
        stop.update(int(t) for t in eos)
    return stop


def _dequantize_trunk_bf16(rt: Any) -> int:
    """Replace every trunk QuantizedLinear with an exact bf16 Linear."""
    import mlx.core as mx
    import mlx.nn as nn

    def _walk(module: Any):
        for name, child in module.children().items():
            if isinstance(child, dict):
                for k, v in child.items():
                    if isinstance(v, nn.Module):
                        yield from _walk_named(module, f"{name}.{k}", v)
            elif isinstance(child, list):
                for i, v in enumerate(child):
                    if isinstance(v, nn.Module):
                        yield from _walk_named(module, f"{name}.{i}", v)
            elif isinstance(child, nn.Module):
                yield from _walk_named(module, name, child)

    def _walk_named(parent: Any, name: str, module: Any):
        if isinstance(module, nn.QuantizedLinear):
            yield parent, name, module
            return
        yield from _walk(module)

    inner = getattr(rt.model, "language_model", rt.model).model
    swapped = 0
    for layer in inner.layers:
        for parent, name, mod in list(_walk(layer)):
            w = mx.dequantize(
                mod.weight,
                mod.scales,
                mod.biases,
                group_size=mod.group_size,
                bits=mod.bits,
            ).astype(mx.bfloat16)
            lin = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False)
            lin.weight = w
            bias = getattr(mod, "bias", None)
            if bias is not None and not isinstance(bias, (int, float)):
                lin.bias = bias
            # setattr through dotted path (list/dict-nested modules)
            obj = parent
            parts = name.split(".")
            for part in parts[:-1]:
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
            last = parts[-1]
            if last.isdigit():
                obj[int(last)] = lin
            else:
                setattr(obj, last, lin)
            mx.eval(lin.weight)
            swapped += 1
    return swapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file")
    parser.add_argument("--prompt-tokens-file")
    parser.add_argument(
        "--prompt-suite",
        help="name of a bundled prompt suite in mtplx/benchmarks/prompts "
        "(e.g. python_modules_long), or a path to a .jsonl in that format. "
        "Use --prompt-suite-id to pick a row; the first row is used otherwise.",
    )
    parser.add_argument(
        "--prompt-suite-id",
        default=None,
        help="id of the row to use from --prompt-suite",
    )
    parser.add_argument(
        "--prompt-json",
        help="JSON bundle with a 'messages' list (chat-template prompt bundle)",
    )
    parser.add_argument("--batches", default="1,2,4")
    parser.add_argument("--depths", default="3")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--capture-backend", default="stock")
    parser.add_argument("--head-history", default="committed")
    parser.add_argument("--loop-mode", default="pipelined")
    parser.add_argument("--draft-core", default="eager")
    parser.add_argument("--ragged-2pass", action="store_true")
    parser.add_argument("--mtp-adapter", default=None,
                        help="path to a trained MTP LoRA adapter npz")
    parser.add_argument(
        "--dequant-trunk",
        action="store_true",
        help="F1 experiment: dequantize trunk Linears to bf16 in place (exact "
        "affine dequant, same function; ~54 GB resident). Head/lm_head stay "
        "quantized (they win at small M).",
    )
    parser.add_argument("--history-window", type=int, default=8192)
    parser.add_argument("--thinking", default="low", help="'off' disables thinking")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help=(
            "reasoning effort passed to the chat template (low/medium/high/"
            "xhigh on families that support it). Distinct from --thinking, "
            "which only toggles thinking on/off: without this, every 'on' "
            "level encodes identically."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy (exactness regime); >0 = speculative sampling")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--decode",
        choices=("mtp", "ar"),
        default="mtp",
        help="mtp runs the batched MTP driver. ar runs plain autoregressive "
        "greedy decode over the same prompts and grid, as the baseline that "
        "says what speculation is actually worth at each width.",
    )
    parser.add_argument(
        "--session-bank",
        action="store_true",
        help="attach a SessionBank so prefix caching is live. Each row gets its "
        "own session id and every prompt is distinct, so nothing is ever reused: "
        "this arm measures what the cache COSTS when it cannot help.",
    )
    parser.add_argument(
        "--ignore-stop",
        action="store_true",
        help="ignore EOS so every stream runs to --max-new (fixed-length arms; "
        "required for batch-scaling comparisons)",
    )
    parser.add_argument("--vary-suffix", action="store_true")
    parser.add_argument("--parity", action="store_true", help="run the B=1 sha gate")
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--truncate-prompt", type=int, default=None,
                        help="keep only the LAST N tokens of the encoded prompt")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from mtplx.dense_mtp_batch import generate_dense_mtp_batch
    from mtplx.runtime import load

    print(f"[dense-batch-bench] loading {args.model}", flush=True)
    rt = load(args.model, mtp=True, mtp_adapter=args.mtp_adapter)
    tokenizer = rt.tokenizer
    if args.dequant_trunk:
        n = _dequantize_trunk_bf16(rt)
        print(f"[dense-batch-bench] dequantized {n} trunk Linears to bf16", flush=True)

    if args.prompt_suite:
        suite = Path(args.prompt_suite)
        if not suite.exists():
            suite = (
                Path(__file__).resolve().parents[1] / "prompts" / f"{args.prompt_suite}.jsonl"
            )
        if not suite.exists():
            raise SystemExit(f"prompt suite not found: {args.prompt_suite}")
        rows = [json.loads(l) for l in suite.read_text().splitlines() if l.strip()]
        if args.prompt_suite_id:
            rows = [r for r in rows if r.get("id") == args.prompt_suite_id]
            if not rows:
                raise SystemExit(f"no row with id {args.prompt_suite_id} in {suite}")
        row = rows[0]
        print(
            f"[dense-batch-bench] prompt suite {suite.name} row "
            f"{row.get('id')!r} (suite max_tokens={row.get('max_tokens')})",
            flush=True,
        )
        base_prompt = _encode_prompt(
            tokenizer, row["prompt"], args.thinking, args.reasoning_effort
        )
    elif args.prompt_tokens_file:
        base_prompt = [
            int(t) for t in json.loads(Path(args.prompt_tokens_file).read_text())
        ]
    elif args.prompt_json:
        bundle = json.loads(Path(args.prompt_json).read_text())
        kwargs: dict[str, Any] = {"add_generation_prompt": True}
        if args.thinking is not None:
            kwargs["enable_thinking"] = args.thinking != "off"
        if args.reasoning_effort:
            kwargs["reasoning_effort"] = args.reasoning_effort
        try:
            ids = tokenizer.apply_chat_template(bundle["messages"], **kwargs)
        except TypeError:
            ids = tokenizer.apply_chat_template(
                bundle["messages"], add_generation_prompt=True
            )
        base_prompt = [int(t) for t in (ids.tolist() if hasattr(ids, "tolist") else ids)]
    elif args.prompt_file:
        base_prompt = _encode_prompt(
            tokenizer,
            Path(args.prompt_file).read_text(),
            args.thinking,
            args.reasoning_effort,
        )
    else:
        raise SystemExit("one of --prompt-file / --prompt-tokens-file is required")
    if args.truncate_prompt:
        base_prompt = base_prompt[-int(args.truncate_prompt):]
    stop_ids = _stop_ids(tokenizer)
    if args.ignore_stop:
        # Throughput arms must be comparable across batch sizes: a stream that
        # hits EOS early leaves a dead row that still costs full batch width
        # every cycle, so aggregate tok/s ends up measuring how soon streams
        # happened to stop rather than how fast the cohort decodes. Ignoring
        # stop tokens pins every arm at exactly batch * max_new tokens.
        stop_ids = set()
    print(
        f"[dense-batch-bench] prompt_tokens={len(base_prompt)} "
        f"stop_ids={sorted(stop_ids) if stop_ids else 'IGNORED (fixed-length arms)'}",
        flush=True,
    )

    def _prompts(batch: int) -> list[list[int]]:
        if not args.vary_suffix:
            return [list(base_prompt) for _ in range(batch)]
        # One distinct printable token per row keeps rows genuinely different
        # while preserving the shared length (replace the last token).
        rows = []
        for b in range(batch):
            row = list(base_prompt)
            row.append(int(tokenizer.encode(f" {b}")[-1]))
            rows.append(row)
        width = max(len(r) for r in rows)
        pad = rows[0][0]
        return [[pad] * (width - len(r)) + r for r in rows]

    batches = [int(x) for x in str(args.batches).split(",") if x.strip()]
    depths = [int(x) for x in str(args.depths).split(",") if x.strip()]

    # Warmup: tiny run to trigger weight residency / kernel compilation.
    print("[dense-batch-bench] warmup", flush=True)
    generate_dense_mtp_batch(
        rt,
        [_prompts(1)[0][-256:]],
        max_new_tokens=max(4, args.warmup_tokens),
        depth=depths[0],
        stop_token_ids=stop_ids,
        capture_backend=args.capture_backend,
        head_history=args.head_history,
        history_window=args.history_window,
        loop_mode=args.loop_mode,
        draft_core=args.draft_core,
        ragged_attention=args.ragged_2pass,
        collect_stats=False,
    )

    # r16 lesson: back-to-back 24k arms in one process died silently (no
    # traceback, no crash report, SIGKILL under memory pressure) with the
    # prior arm's Metal cache pool still resident. Drop the pool between
    # generates; prefer one arm per invocation for big-B 24k runs regardless.
    import mlx.core as mx

    mx.clear_cache()

    receipt: dict[str, Any] = {
        "bench": "dense_mtp_batch",
        "model": str(args.model),
        "prompt_tokens": len(base_prompt),
        "capture_backend": args.capture_backend,
        "prompt_suite": args.prompt_suite,
        "prompt_suite_id": args.prompt_suite_id,
        "thinking": args.thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "ignore_stop": bool(args.ignore_stop),
        "reasoning_effort": args.reasoning_effort,
        "head_history": args.head_history,
        "loop_mode": args.loop_mode,
        "draft_core": args.draft_core,
        "ragged_2pass": bool(args.ragged_2pass),
        "mtp_adapter": args.mtp_adapter,
        "dequant_trunk": bool(args.dequant_trunk),
        "history_window": int(args.history_window),
        "max_new": int(args.max_new),
        "vary_suffix": bool(args.vary_suffix),
        "session_bank": bool(args.session_bank),
        "decode": args.decode,
        "arms": [],
    }

    bank = None
    if args.session_bank:
        from mtplx.session_bank import SessionBank

        bank = SessionBank()
        print("[dense-batch-bench] SessionBank attached (prefix caching live)", flush=True)

    for depth in depths:
        for batch in batches:
            prompts = _prompts(batch)
            label = f"B={batch} depth={depth}"
            print(f"[dense-batch-bench] arm {label}", flush=True)
            mx.clear_cache()
            started = time.perf_counter()
            if args.decode == "ar":
                from mtplx.batched_decode import generate_greedy_batched

                assert len({len(p) for p in prompts}) == 1, (
                    "AR baseline needs equal-length prompts; left-padding a hybrid "
                    "GDN trunk is silently incorrect, so this refuses rather than pads"
                )
                ar = generate_greedy_batched(
                    rt,
                    prompts,
                    max_new_tokens=args.max_new,
                    stop_token_ids=stop_ids,
                    decode_mode="ar",
                    collect_stats=True,
                )
                arm = {
                    "batch": batch,
                    "depth": 0,
                    "decode": "ar",
                    "prefill_s": ar.prefill_s,
                    "decode_s": ar.decode_s,
                    "cycles": ar.cycles,
                    "generated_tokens": ar.generated_tokens,
                    "aggregate_decode_tokps": ar.aggregate_decode_tokps,
                    "tokens_per_cycle": (
                        ar.generated_tokens / ar.cycles if ar.cycles else 0.0
                    ),
                    "accepted_by_depth": [],
                    "drafted_by_depth": [],
                    "acceptance_by_depth": [],
                    "finish_reasons": [s.finish_reason for s in ar.streams],
                    "tokens_per_stream": [len(s.tokens) for s in ar.streams],
                    "shas": ar.shas,
                    "wall_s": time.perf_counter() - started,
                    "meta": ar.meta,
                }
                receipt["arms"].append(arm)
                print(
                    f"[dense-batch-bench]   {label}: "
                    f"{ar.aggregate_decode_tokps:.2f} tok/s aggregate (AR baseline)",
                    flush=True,
                )
                if args.out:
                    out = Path(args.out)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(receipt, indent=2))
                continue

            res = generate_dense_mtp_batch(
                rt,
                prompts,
                max_new_tokens=args.max_new,
                depth=depth,
                stop_token_ids=stop_ids,
                capture_backend=args.capture_backend,
                head_history=args.head_history,
                history_window=args.history_window,
                loop_mode=args.loop_mode,
                draft_core=args.draft_core,
                ragged_attention=args.ragged_2pass,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                sampling_seed=args.sampling_seed,
                session_bank=bank,
                session_ids=(
                    [f"bench-d{depth}-b{batch}-r{i}" for i in range(batch)]
                    if bank is not None
                    else None
                ),
            )
            arm: dict[str, Any] = {
                "batch": batch,
                "depth": depth,
                "prefill_s": res.prefill_s,
                "decode_s": res.decode_s,
                "cycles": res.cycles,
                "generated_tokens": res.generated_tokens,
                "aggregate_decode_tokps": res.aggregate_decode_tokps,
                "tokens_per_cycle": res.tokens_per_cycle,
                "accepted_by_depth": res.accepted_by_depth,
                "drafted_by_depth": res.drafted_by_depth,
                "acceptance_by_depth": [
                    (a / d if d else 0.0)
                    for a, d in zip(res.accepted_by_depth, res.drafted_by_depth)
                ],
                "finish_reasons": [s.finish_reason for s in res.streams],
                "tokens_per_stream": [len(s.tokens) for s in res.streams],
                "shas": res.shas,
                "wall_s": time.perf_counter() - started,
                "meta": res.meta,
            }
            print(
                f"[dense-batch-bench]   {label}: {res.aggregate_decode_tokps:.2f} tok/s "
                f"aggregate, {res.tokens_per_cycle:.3f} tok/cycle, "
                f"accept={arm['acceptance_by_depth']}",
                flush=True,
            )
            if args.parity and batch > 1:
                # Fixed-shape gate discipline (as the A3B lane's): the reference
                # decodes each stream ALONE in a cohort of the SAME slot count
                # (dummy rows fill the rest), so every kernel dispatch has
                # identical shapes and a mismatch isolates per-row forward
                # independence, not batch-size numerics.
                mismatches = []
                for b, prompt in enumerate(prompts):
                    # Each reference decode is a fresh long-prompt prefill in
                    # the same process; without dropping the Metal buffer
                    # cache between them the pool grows per prefill until the
                    # OS kills the process silently (no traceback, no crash
                    # report) on large-context arms.
                    mx.clear_cache()
                    ref = generate_dense_mtp_batch(
                        rt,
                        [prompt],
                        max_new_tokens=args.max_new,
                        depth=depth,
                        stop_token_ids=stop_ids,
                        capture_backend=args.capture_backend,
                        head_history=args.head_history,
                        history_window=args.history_window,
                        loop_mode=args.loop_mode,
                        draft_core=args.draft_core,
                        ragged_attention=args.ragged_2pass,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                sampling_seed=args.sampling_seed,
                        cohort_slots=batch,
                        pad_id=prompt[0],
                        collect_stats=False,
                    )
                    if ref.streams[0].sha != res.streams[b].sha:
                        first = next(
                            (
                                i
                                for i, (x, y) in enumerate(
                                    zip(ref.streams[0].tokens, res.streams[b].tokens)
                                )
                                if x != y
                            ),
                            min(
                                len(ref.streams[0].tokens),
                                len(res.streams[b].tokens),
                            ),
                        )
                        mismatches.append({"stream": b, "first_divergence": first})
                arm["parity"] = {
                    "passed": not mismatches,
                    "mismatches": mismatches,
                }
                print(
                    f"[dense-batch-bench]   parity: "
                    f"{'PASS' if not mismatches else mismatches}",
                    flush=True,
                )
            receipt["arms"].append(arm)
            if args.out:
                # Incremental receipt: a killed run keeps its completed arms.
                out = Path(args.out)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(receipt, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2))
        print(f"[dense-batch-bench] receipt -> {out}", flush=True)
    else:
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
