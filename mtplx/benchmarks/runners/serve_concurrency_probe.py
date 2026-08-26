"""Concurrent-request driver for a running ``mtplx serve``.

``dense_batch_bench`` calls ``generate_dense_mtp_batch`` directly, so it proves
the DRIVER works and proves nothing at all about the SERVER. Nothing in the
tree exercises the serving path: requests arriving separately, over HTTP, and
being gathered into a cohort by the model-owner thread. This probe is that
missing piece, and every later item in the serving plan is measured through it.

WHAT IT MEASURES, AND WHY THE FIRST ONE MATTERS MOST
----------------------------------------------------
1. **The observed cohort width.** Read from the server's own counters, not
   inferred from throughput. This is the whole point. A concurrency run that
   never batched and a concurrency run that batched without helping produce the
   SAME disappointing tokens per second, and the first is a configuration bug
   while the second is a real result. Reporting throughput without the width
   histogram cannot tell them apart, and that confusion has already cost this
   campaign a day. The histogram comes from
   ``scheduler.dense_mtp_batch.batch_histogram`` in ``/v1/mtplx/snapshot``,
   diffed across the run, so it counts the cohorts THIS probe caused.

2. **Aggregate and per-stream throughput.** Aggregate is the serving number;
   per-stream is what each caller experiences. Batching trades the second for
   the first and both belong in the receipt.

3. **Per-caller corroboration of the width.** Each response carries its own
   ``active_batch_size``, so the histogram is checked against one record per
   request rather than trusted alone. Note that the response envelope filters
   unrecognised stat keys, so the dense lane's own ``dense_mtp_batch_*`` fields
   do not reach the client even though the service sets them; read
   ``active_batch_size`` and ``scheduler_policy``, which do survive.

Usage (needs a server already running; this probe never starts one):

    python -m mtplx.benchmarks.runners.serve_concurrency_probe \\
        --base-url http://127.0.0.1:8080 --model local \\
        --n 8 --max-tokens 128 --out results/serve-concurrency/n8.json

Stdlib only, on purpose: it must run from anywhere that can reach the port,
including a box with no project venv.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Per-request keys the server stamps that this probe reports verbatim. Anything
# not found is reported as missing rather than as zero: a zero would read as a
# measurement and these are receipts.
_STAT_KEYS = (
    "active_batch_size",
    "dense_mtp_batch_real_width",
    "dense_mtp_batch_left_pad_tokens",
    "dense_mtp_batch_prompt_tokens_true",
    "dense_mtp_batch_cohort_aggregate_tok_s",
    "scheduler_lane",
    "scheduler_policy",
    "decode_tok_s",
    "decode_elapsed_s",
    "queue_wait_s",
    "generated_tokens",
    "target_verify_cycles",
)


def _post_sse(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST a streaming request and time every chunk as it arrives.

    Returns arrival timestamps rather than just the text, because the timing is
    the measurement: a lane that buffers and flushes at the end produces the
    same tokens as one that streams, and only the arrival spread tells them
    apart.
    """

    body = json.dumps({**payload, "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    started = time.perf_counter()
    arrivals: list[float] = []
    content_pieces: list[str] = []
    final_chunk: dict[str, Any] = {}
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            final_chunk = chunk
            choices = chunk.get("choices") or [{}]
            delta = (choices[0] or {}).get("delta") or {}
            piece = delta.get("content")
            if piece:
                arrivals.append(time.perf_counter() - started)
                content_pieces.append(piece)
    return {
        "arrivals_s": arrivals,
        "text": "".join(content_pieces),
        "final_chunk": final_chunk,
        "total_s": time.perf_counter() - started,
    }


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _find_stats(payload: Any) -> dict[str, Any]:
    """Dig the per-request stats block out of the response envelope.

    Searched for rather than addressed by path, because the envelope shape is
    the server's business and this probe should keep working when it changes. A
    dict carrying any of the marker keys is the stats block.
    """

    markers = {"scheduler_lane", "decode_tok_s", "active_batch_size"}
    found: dict[str, Any] = {}

    def walk(node: Any) -> None:
        nonlocal found
        if found:
            return
        if isinstance(node, dict):
            if markers & set(node.keys()):
                found = node
                return
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(payload)
    return found


# Server counters that answer "did continuous batching happen". Cumulative on
# the server, so the receipt records before AND after: the delta is this run's
# contribution, the after-value is the server's lifetime tally, and conflating
# them is what makes a receipt look like it contradicts a log.
_CONTINUOUS_KEYS = (
    "refill_admitted_total",
    "refill_requeued_total",
    "requests_served_total",
    "max_requests_in_one_cohort",
    "continuous_batching_observed",
)


def _continuous(snapshot: dict[str, Any], lane_key: str) -> dict[str, Any]:
    scheduler = (snapshot or {}).get("scheduler") or {}
    lane = scheduler.get(lane_key) or {}
    return {key: lane.get(key) for key in _CONTINUOUS_KEYS if key in lane}


def _histogram(snapshot: dict[str, Any], lane_key: str) -> dict[str, int]:
    scheduler = (snapshot or {}).get("scheduler") or {}
    lane = scheduler.get(lane_key) or {}
    raw = lane.get("batch_histogram") or {}
    return {str(k): int(v) for k, v in raw.items()}


def _histogram_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    widths = set(before) | set(after)
    delta = {w: after.get(w, 0) - before.get(w, 0) for w in widths}
    return {w: c for w, c in sorted(delta.items(), key=lambda kv: int(kv[0])) if c}


def _build_prompt(index: int, words: int, spread: int) -> str:
    """One prompt per row, distinct in content and optionally in length.

    Distinct content matters: identical prompts at temperature 0 produce
    identical streams, which would make a cohort look correct even if rows were
    leaking into each other. ``spread`` grows each successive prompt so a run
    can exercise the mixed-length admission path on purpose.
    """

    length = max(1, words + index * spread)
    filler = " ".join(f"item{index}-{i}" for i in range(length))
    return (
        f"You are given a list labelled L{index}. Repeat the list back, then "
        f"count how many entries it has and state the number.\n\nL{index}: {filler}"
    )


def _run_one(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout: float,
    index: int,
    barrier: threading.Barrier | None,
    results: list[dict[str, Any]],
    lock: threading.Lock,
    stream: bool = False,
) -> None:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
    }
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if top_k:
        payload["top_k"] = int(top_k)
    record: dict[str, Any] = {"index": index, "prompt_chars": len(prompt)}
    if stream:
        if barrier is not None:
            try:
                barrier.wait(timeout=timeout)
            except threading.BrokenBarrierError:
                record["error"] = "barrier_broken"
        started = time.perf_counter()
        try:
            sse = _post_sse(
                f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout
            )
            arrivals = sse["arrivals_s"]
            record.update(
                {
                    "ok": True,
                    "streamed": True,
                    "latency_s": sse["total_s"],
                    "chunks": len(arrivals),
                    "time_to_first_token_s": arrivals[0] if arrivals else None,
                    "time_to_last_token_s": arrivals[-1] if arrivals else None,
                    # THE discriminator. Buffered-then-flushed output has every
                    # chunk landing in one burst at the end, so the spread is a
                    # sliver of total latency. Genuine streaming spreads across
                    # the run.
                    "arrival_spread_s": (
                        arrivals[-1] - arrivals[0] if len(arrivals) > 1 else 0.0
                    ),
                    "arrival_spread_fraction": (
                        (arrivals[-1] - arrivals[0]) / sse["total_s"]
                        if len(arrivals) > 1 and sse["total_s"] > 0
                        else 0.0
                    ),
                    # The decode window is the time in which chunks COULD have
                    # arrived: everything after the first token. Spread over
                    # total latency has prefill in its denominator, so its
                    # ceiling is set by how long prefill took and a perfect
                    # streamer can score arbitrarily low on it.
                    "decode_window_s": (
                        sse["total_s"] - arrivals[0] if arrivals else 0.0
                    ),
                    "spread_over_decode_window": (
                        (arrivals[-1] - arrivals[0]) / (sse["total_s"] - arrivals[0])
                        if len(arrivals) > 1 and sse["total_s"] > arrivals[0]
                        else 0.0
                    ),
                    "completion_tokens": len(arrivals),
                    "per_stream_tok_s": (
                        len(arrivals) / sse["total_s"] if sse["total_s"] > 0 else 0.0
                    ),
                    "stats": {},
                    "stats_missing": [],
                }
            )
        except Exception as exc:  # noqa: BLE001
            record.update(
                {
                    "ok": False,
                    "streamed": True,
                    "latency_s": time.perf_counter() - started,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        with lock:
            results.append(record)
        return
    if barrier is not None:
        # Release every thread at the same instant. Without this the requests
        # trickle in and the server correctly seals cohorts of one, which would
        # look like "batching does not work".
        try:
            barrier.wait(timeout=timeout)
        except threading.BrokenBarrierError:
            record["error"] = "barrier_broken"
    started = time.perf_counter()
    try:
        response = _post_json(
            f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout
        )
        elapsed = time.perf_counter() - started
        stats = _find_stats(response)
        usage = response.get("usage") or {}
        completion_tokens = int(
            usage.get("completion_tokens") or stats.get("generated_tokens") or 0
        )
        record.update(
            {
                "ok": True,
                "latency_s": elapsed,
                "completion_tokens": completion_tokens,
                "per_stream_tok_s": (
                    completion_tokens / elapsed if elapsed > 0 else 0.0
                ),
                "stats": {k: stats[k] for k in _STAT_KEYS if k in stats},
                "stats_missing": [k for k in _STAT_KEYS if k not in stats],
            }
        )
    except urllib.error.HTTPError as exc:
        record.update(
            {
                "ok": False,
                "latency_s": time.perf_counter() - started,
                "error": f"HTTP {exc.code}",
                "body": exc.read().decode("utf-8", "replace")[:2000],
            }
        )
    except Exception as exc:  # noqa: BLE001 - the probe reports, it does not raise
        record.update(
            {
                "ok": False,
                "latency_s": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    with lock:
        results.append(record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fire concurrent requests at a running mtplx serve and "
        "report the OBSERVED cohort width alongside throughput."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="local")
    parser.add_argument("--n", type=int, default=8, help="concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--prompt-words",
        type=int,
        default=60,
        help="approximate length of the first prompt, in filler words",
    )
    parser.add_argument(
        "--length-spread",
        type=int,
        default=0,
        help="extra filler words per successive request; non-zero exercises "
        "the mixed-length admission path and the left-pad cost",
    )
    parser.add_argument(
        "--mixed-sampling",
        action="store_true",
        help="give every other request a different temperature, which MUST "
        "split the cohort under item 1's uniform-sampling constraint",
    )
    parser.add_argument(
        "--no-barrier",
        action="store_true",
        help="do not release the requests simultaneously (measures the "
        "arrival-gathering window instead of best-case batching)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="use SSE streaming and measure chunk arrival times; the arrival "
        "spread is what distinguishes real streaming from buffer-then-flush",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--lane-key",
        default="dense_mtp_batch",
        help="which scheduler lane's counters to diff (dense_mtp_batch or "
        "mtp_batch)",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    snapshot_url = f"{base}/v1/mtplx/snapshot"

    before_snapshot: dict[str, Any] = {}
    snapshot_error: str | None = None
    try:
        before_snapshot = _get_json(snapshot_url, args.timeout)
    except Exception as exc:  # noqa: BLE001
        snapshot_error = f"{type(exc).__name__}: {exc}"

    prompts = [
        _build_prompt(i, args.prompt_words, args.length_spread)
        for i in range(args.n)
    ]
    temperatures = [
        (args.temperature + 0.7 if (args.mixed_sampling and i % 2) else args.temperature)
        for i in range(args.n)
    ]

    results: list[dict[str, Any]] = []
    lock = threading.Lock()
    barrier = None if args.no_barrier else threading.Barrier(args.n)
    threads = [
        threading.Thread(
            target=_run_one,
            kwargs={
                "base_url": base,
                "model": args.model,
                "prompt": prompts[i],
                "max_tokens": args.max_tokens,
                "temperature": temperatures[i],
                "top_p": args.top_p,
                "top_k": args.top_k,
                "timeout": args.timeout,
                "index": i,
                "barrier": barrier,
                "results": results,
                "lock": lock,
                "stream": args.stream,
            },
            daemon=True,
        )
        for i in range(args.n)
    ]

    wall_started = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    wall_s = time.perf_counter() - wall_started

    after_snapshot: dict[str, Any] = {}
    try:
        after_snapshot = _get_json(snapshot_url, args.timeout)
    except Exception as exc:  # noqa: BLE001
        snapshot_error = snapshot_error or f"{type(exc).__name__}: {exc}"

    width_histogram = _histogram_delta(
        _histogram(before_snapshot, args.lane_key),
        _histogram(after_snapshot, args.lane_key),
    )
    continuous_before = _continuous(before_snapshot, args.lane_key)
    continuous_after = _continuous(after_snapshot, args.lane_key)
    continuous_delta = {
        key: (continuous_after.get(key, 0) - continuous_before.get(key, 0))
        for key in ("refill_admitted_total", "refill_requeued_total",
                    "requests_served_total")
        if isinstance(continuous_after.get(key), int)
        and isinstance(continuous_before.get(key), int)
    }

    results.sort(key=lambda r: r["index"])
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    total_tokens = sum(int(r["completion_tokens"]) for r in ok)
    latencies = [float(r["latency_s"]) for r in ok]
    # Per-caller corroboration of the cohort width, independent of the
    # server-side histogram. Read `active_batch_size` as well as the dense
    # lane's own key: the response envelope filters unrecognised stat keys, so
    # `dense_mtp_batch_real_width` does not survive to the client even though
    # the service sets it. Relying on that key alone left this array EMPTY in
    # every receipt, which reads as "the corroboration is missing" rather than
    # "it is under a different name" — the same shape of silent-looking-wrong
    # this probe exists to prevent.
    def _row_width(stats: dict[str, Any]) -> int | None:
        for key in ("dense_mtp_batch_real_width", "active_batch_size"):
            value = stats.get(key)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
        return None

    per_request_widths = [_row_width(r.get("stats", {})) for r in ok]
    observed_widths = sorted({w for w in per_request_widths if w is not None})
    # A solo-served request reports no cohort width at all, which is correct
    # rather than missing: a cohort of one takes the ordinary solo path.
    solo_served = sum(1 for w in per_request_widths if w is None)

    receipt = {
        "probe": "serve_concurrency_probe",
        "base_url": base,
        "requested_concurrency": args.n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "mixed_sampling": bool(args.mixed_sampling),
        "length_spread_words": args.length_spread,
        "simultaneous_release": not args.no_barrier,
        "wall_s": wall_s,
        "requests_ok": len(ok),
        "requests_failed": len(failed),
        "total_completion_tokens": total_tokens,
        "aggregate_tok_s": (total_tokens / wall_s if wall_s > 0 else 0.0),
        "per_stream_tok_s_median": (
            statistics.median([float(r["per_stream_tok_s"]) for r in ok]) if ok else 0.0
        ),
        "latency_s_min": min(latencies) if latencies else None,
        "latency_s_median": statistics.median(latencies) if latencies else None,
        "latency_s_max": max(latencies) if latencies else None,
        # THE receipt: server-side cohort widths caused by this run.
        "server_batch_histogram_delta": width_histogram,
        # Continuous-batching evidence, so this receipt stands alone.
        "continuous_batching_before": continuous_before,
        "continuous_batching_after": continuous_after,
        "continuous_batching_delta": continuous_delta,
        "counter_scope_note": (
            "server_batch_histogram_delta and continuous_batching_delta are "
            "scoped to THIS run. continuous_batching_after is the server's "
            "cumulative lifetime tally across every run against it, so it will "
            "exceed this run's numbers whenever the server served earlier "
            "requests. A larger cumulative figure is not a contradiction."
        ),
        "continuous_batching_verdict": (
            "NOT APPLICABLE: no refill counters exposed by this lane"
            if not continuous_after
            else "CONFIRMED: a cohort served more requests than its width"
            if continuous_delta.get("refill_admitted_total", 0) > 0
            else "NOT OBSERVED: every request fit without reusing a slot"
        ),
        "per_request_observed_widths": observed_widths,
        "per_request_widths": per_request_widths,
        "requests_served_solo": solo_served,
        "streaming": bool(args.stream),
        "streaming_summary": (
            None
            if not args.stream
            else {
                "time_to_first_token_s": [
                    r.get("time_to_first_token_s") for r in ok
                ],
                "arrival_spread_s": [r.get("arrival_spread_s") for r in ok],
                "arrival_spread_fraction": [
                    r.get("arrival_spread_fraction") for r in ok
                ],
                "spread_over_decode_window": [
                    r.get("spread_over_decode_window") for r in ok
                ],
                "chunks": [r.get("chunks") for r in ok],
                "verdict_metric": (
                    "spread_over_decode_window: fraction of the post-first-token "
                    "window over which chunks arrived. Buffer-then-flush puts "
                    "every chunk at one instant and scores ~0; streaming scores "
                    "near 1. NOT measured against total latency, whose ceiling "
                    "is set by prefill cost rather than by streaming."
                ),
                "verdict": (
                    "no streamed requests succeeded"
                    if not ok
                    else "SINGLE CHUNK: nothing to infer from one arrival"
                    if max((r.get("chunks") or 0) for r in ok) <= 1
                    else "STREAMING: chunks arrived throughout decoding"
                    if min(
                        (r.get("spread_over_decode_window") or 0.0) for r in ok
                    )
                    > 0.5
                    else "BUFFERED: chunks arrived in a burst rather than "
                    "throughout decoding"
                ),
            }
        ),
        "snapshot_error": snapshot_error,
        "requests": results,
    }

    print(f"[serve-probe] concurrency={args.n} wall={wall_s:.2f}s "
          f"ok={len(ok)} failed={len(failed)}", flush=True)
    print(f"[serve-probe] aggregate={receipt['aggregate_tok_s']:.2f} tok/s  "
          f"per-stream median={receipt['per_stream_tok_s_median']:.2f} tok/s",
          flush=True)
    if width_histogram:
        widths = ", ".join(f"width {w} x{c}" for w, c in width_histogram.items())
        print(f"[serve-probe] server sealed cohorts: {widths}", flush=True)
        if observed_widths:
            print(
                f"[serve-probe] per-request widths agree: {observed_widths}"
                + (f" ({solo_served} served solo)" if solo_served else ""),
                flush=True,
            )
        elif args.stream:
            print(
                "[serve-probe] per-request widths not collected in stream mode; "
                "the histogram above is the width evidence",
                flush=True,
            )
        elif solo_served == len(ok):
            print("[serve-probe] every request was served solo", flush=True)
        else:
            print(
                "[serve-probe] WARNING: no per-request width reported, so the "
                "histogram is uncorroborated",
                flush=True,
            )
    elif snapshot_error:
        print(f"[serve-probe] cohort widths UNKNOWN: snapshot unreadable "
              f"({snapshot_error}). Throughput alone cannot tell 'never "
              f"batched' from 'batched and did not help'.", flush=True)
    else:
        print("[serve-probe] server sealed NO cohorts on this lane. Either the "
              "lane is not installed or every request was served solo.",
              flush=True)
    if args.stream and ok:
        ttft = [r.get("time_to_first_token_s") or 0.0 for r in ok]
        frac = [r.get("arrival_spread_fraction") or 0.0 for r in ok]
        over_decode = [r.get("spread_over_decode_window") or 0.0 for r in ok]
        chunks = [r.get("chunks") or 0 for r in ok]
        print(
            f"[serve-probe] streaming: {min(chunks)}-{max(chunks)} chunks, "
            f"first token at {min(ttft):.2f}-{max(ttft):.2f}s, arriving over "
            f"{min(over_decode)*100:.0f}-{max(over_decode)*100:.0f}% of the "
            f"decode window ({min(frac)*100:.0f}-{max(frac)*100:.0f}% of total "
            f"latency, which includes prefill)",
            flush=True,
        )
        print(f"[serve-probe] {receipt['streaming_summary']['verdict']}", flush=True)
    if continuous_after:
        admitted = continuous_delta.get("refill_admitted_total", 0)
        print(
            f"[serve-probe] continuous batching: {admitted} request(s) admitted "
            f"into freed slots this run; server lifetime max requests in one "
            f"cohort = {continuous_after.get('max_requests_in_one_cohort')}",
            flush=True,
        )
    if failed:
        print(f"[serve-probe] {len(failed)} failed: "
              f"{failed[0].get('error')}", flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(receipt, indent=2))
        print(f"[serve-probe] receipt -> {out}", flush=True)
    else:
        print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
