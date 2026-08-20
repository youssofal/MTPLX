#!/usr/bin/env python3
"""StreamScope — the streaming-smoothness measurement harness (2026-08-19).

Why this exists: every prior streaming regression shipped while the usual
numbers were green, because each layer was measured alone and upstream of
where the user's eye looks. StreamScope stamps the SAME stream at every
layer and merges the timelines on wall clock:

  engine visible-emit census   (server, MTPLX_STREAM_CENSUS=1, post-decoder)
  SSE client arrival           (this script, one localhost hop later)
  app document flushes         (UIStreamPerfProbe uistream-*.jsonl)
  app render layer + paint     (renderTimed sites + CADisplayLink watchdog)
  CPU/GPU/temps                (macmon pipe)

Subcommands:

  api          Run the prompt battery against a serving daemon over HTTP,
               stamping every SSE event client-side. Produces per-prompt
               scorecard.json + timeline.jsonl + a battery summary.

  app-collect  Harvest app-side diagnostics (aime-*.jsonl, uistream-*.jsonl)
               written since --since, merge with engine/client artifacts if
               given, and score the app render pipeline.

Examples:
  python scripts/streamscope_run.py api \
      --base-url http://127.0.0.1:52415 --model Qwen3.8-27B-...-Quality \
      --label baseline-oq8 --out outputs/streamscope-20260819
  python scripts/streamscope_run.py app-collect \
      --since 2026-08-19T15:00:00 --label baseline-oq8-app \
      --engine-run outputs/streamscope-20260819/baseline-oq8/flappy \
      --out outputs/streamscope-20260819

Thermal rule: `api` refuses to run without a verified fanmax receipt unless
--no-thermal-gate is passed explicitly (and says so loudly in the summary).

Stdlib only. No new dependencies.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

FANMAX_GATE = "/Users/youssof/Projects/MTPLX/scripts/fanmax_gate_20260819.py"
THERMALFORGE = str(Path.home() / ".mtplx/bin/thermalforge")
APP_DIAG_DIR = Path.home() / "Library/Application Support/MTPLX/Diagnostics"
DEFAULT_CENSUS_DIR = "/tmp/mtplx-stream-census"

# The battery. flappy is the founder's literal repro; the other four are the
# content classes the offline replay ranked by decoder-holdback severity
# (minified worst: 879 ms gaps at 33 tok/s on the shipped decoder).
PROMPTS: dict[str, dict] = {
    "flappy": {
        "prompt": "make the ultimate flappy bird game, gorgeous overkill beautiful, in HTML",
        "reasoning_effort": "medium",
        "max_tokens": 6144,
    },
    # reasoning_effort medium everywhere: it is the founder's real request
    # shape, and the default effort burned entire token budgets in the think
    # channel (finish_reason=length, answer_tokens=0) on the 2026-08-19
    # baseline attempt. Never cap thinking to force an answer — bound the
    # budget generously and shape the request like the product does.
    "dense_python": {
        "prompt": (
            "Write one Python file implementing an LRU cache class and a trie-based "
            "autocomplete class with insert/search/complete methods. Dense code, no "
            "comments, no blank lines, no explanation outside one code fence."
        ),
        "reasoning_effort": "medium",
        "max_tokens": 4096,
    },
    "markdown_table": {
        "prompt": (
            "Produce a markdown table with 40 rows comparing sorting algorithms. "
            "Columns: name | best | average | worst | space | stable | in-place. "
            "Output the table only, no prose."
        ),
        "reasoning_effort": "medium",
        # 40 comparison rows invite long thinking even at medium effort; give
        # the answer room (a 4096 budget hit finish=length on 2026-08-19).
        "max_tokens": 6144,
    },
    "minified_json": {
        "prompt": (
            "Output a single line of minified JSON (no code fence, no whitespace "
            "anywhere): an array of 80 objects with keys id, name, email, tags "
            "(array of 3 short strings), active."
        ),
        "reasoning_effort": "medium",
        "max_tokens": 5120,
    },
    "prose": {
        "prompt": (
            "Explain in flowing prose, with no lists and no code, how a hot air "
            "balloon works. Around 400 words."
        ),
        "reasoning_effort": "medium",
        "max_tokens": 3072,
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, int(len(ordered) * p / 100)))
    return round(ordered[rank], 2)


def gap_stats(gaps_ms: list[float]) -> dict:
    return {
        "count": len(gaps_ms),
        "p50": pct(gaps_ms, 50),
        "p90": pct(gaps_ms, 90),
        "p95": pct(gaps_ms, 95),
        "max": round(max(gaps_ms), 2) if gaps_ms else None,
        "over_100ms": sum(1 for g in gaps_ms if g > 100),
        "over_150ms": sum(1 for g in gaps_ms if g > 150),
        "over_250ms": sum(1 for g in gaps_ms if g > 250),
    }


def classify_delta(payload: dict) -> tuple[str, int]:
    """Mirror of the server census classification, applied client-side."""
    progress = payload.get("mtplx_progress")
    if isinstance(progress, dict):
        return ("heartbeat" if progress.get("heartbeat") else "progress"), 0
    choices = payload.get("choices") or []
    delta = (choices[0].get("delta") or {}) if choices else {}
    if delta.get("content"):
        return "content", len(delta["content"])
    if delta.get("reasoning_content"):
        return "reasoning", len(delta["reasoning_content"])
    if delta.get("tool_calls"):
        chars = sum(
            len(((call.get("function") or {}).get("arguments")) or "")
            for call in delta["tool_calls"]
            if isinstance(call, dict)
        )
        return "tool_calls", chars
    if delta.get("role"):
        return "role", 0
    if choices and choices[0].get("finish_reason"):
        return "finish", 0
    return "other", 0


# ---------------------------------------------------------------- thermal


def thermal_mode_is_max() -> bool:
    try:
        out = subprocess.run(
            [THERMALFORGE, "status"], capture_output=True, text=True, timeout=30
        )
        fans = json.loads(out.stdout)["fans"]
        return bool(fans) and all(
            str(f.get("mode", "")).lower() not in {"auto", ""} for f in fans
        )
    except Exception:
        return False


def run_fanmax_gate(out_dir: Path) -> dict:
    receipt = out_dir / f"fanmax_receipt_{int(time.time())}.json"
    proc = subprocess.run(
        [sys.executable, FANMAX_GATE, str(receipt)],
        capture_output=True,
        text=True,
        timeout=200,
    )
    ok = proc.returncode == 0
    return {"compliant": ok, "receipt": str(receipt), "stdout": proc.stdout.strip()[-400:]}


# ---------------------------------------------------------------- macmon


class MacmonSampler:
    """`macmon pipe` JSONL subprocess. One line per interval; the `timestamp`
    field is wall clock, which is what the merged timeline joins on."""

    def __init__(self, path: Path, interval_ms: int = 500):
        self.path = path
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen | None = None
        self.sink = None

    def start(self) -> bool:
        exe = shutil.which("macmon")
        if not exe:
            return False
        self.sink = open(self.path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            [exe, "pipe", "-i", str(self.interval_ms)],
            stdout=self.sink,
            stderr=subprocess.DEVNULL,
        )
        return True

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
        if self.sink is not None:
            self.sink.close()
            self.sink = None


def macmon_window_stats(path: Path, t0_wall: float, t1_wall: float) -> dict | None:
    if not path.exists():
        return None
    cpu, gpu, cpu_t, gpu_t = [], [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            sample = json.loads(line)
            ts = datetime.fromisoformat(sample["timestamp"]).timestamp()
        except Exception:
            continue
        if not (t0_wall <= ts <= t1_wall):
            continue
        if isinstance(sample.get("cpu_usage_pct"), (int, float)):
            cpu.append(sample["cpu_usage_pct"] * 100)
        gpu_usage = sample.get("gpu_usage")
        if isinstance(gpu_usage, list) and len(gpu_usage) == 2:
            gpu.append(gpu_usage[1] * 100)
        temp = sample.get("temp") or {}
        if isinstance(temp.get("cpu_temp_avg"), (int, float)):
            cpu_t.append(temp["cpu_temp_avg"])
        if isinstance(temp.get("gpu_temp_avg"), (int, float)):
            gpu_t.append(temp["gpu_temp_avg"])
    if not cpu and not gpu:
        return None
    return {
        "samples": max(len(cpu), len(gpu)),
        "cpu_pct_mean": round(statistics.fmean(cpu), 1) if cpu else None,
        "cpu_pct_max": round(max(cpu), 1) if cpu else None,
        "gpu_pct_mean": round(statistics.fmean(gpu), 1) if gpu else None,
        "gpu_pct_max": round(max(gpu), 1) if gpu else None,
        "cpu_temp_max": round(max(cpu_t), 1) if cpu_t else None,
        "gpu_temp_max": round(max(gpu_t), 1) if gpu_t else None,
    }


# ---------------------------------------------------------------- SSE run


def run_sse_prompt(
    base_url: str,
    model: str,
    spec: dict,
    records_path: Path,
) -> dict:
    """POST one streaming chat completion; stamp every SSE event on arrival.

    Timestamps happen at readline return — one buffered localhost hop after
    the server's own census stamp, so the pair also measures transport skew.
    """
    parsed = urlparse(base_url)
    conn_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = conn_cls(parsed.hostname, parsed.port or 80, timeout=600)
    body = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": spec["prompt"]}],
        "max_tokens": spec.get("max_tokens", 4096),
    }
    if spec.get("reasoning_effort"):
        body["reasoning_effort"] = spec["reasoning_effort"]

    t_request_mono = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(body),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read(400)!r}")

    records: list[dict] = []
    response_id = None
    usage = None
    mtplx_stats = None
    content_parts: list[str] = []
    reasoning_chars = 0
    while True:
        raw = resp.readline()
        if not raw:
            break
        t_mono = time.perf_counter()
        t_wall = time.time()
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: "):
            continue
        payload_text = line[6:]
        if payload_text == "[DONE]":
            records.append(
                {"t_mono": t_mono, "t_wall": t_wall, "channel": "done", "chars": 0,
                 "bytes": len(raw)}
            )
            break
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        response_id = payload.get("id") or response_id
        channel, chars = classify_delta(payload)
        if channel == "content":
            content_parts.append(payload["choices"][0]["delta"]["content"])
        elif channel == "reasoning":
            reasoning_chars += chars
        if payload.get("usage"):
            usage = payload["usage"]
        if payload.get("mtplx_stats"):
            mtplx_stats = payload["mtplx_stats"]
        records.append(
            {"t_mono": t_mono, "t_wall": t_wall, "channel": channel, "chars": chars,
             "bytes": len(raw)}
        )
    conn.close()

    with open(records_path, "w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, separators=(",", ":")) + "\n")

    return {
        "response_id": response_id,
        "t_request_mono": t_request_mono,
        "records": records,
        "usage": usage,
        "mtplx_stats": mtplx_stats,
        "content_text": "".join(content_parts),
        "reasoning_chars": reasoning_chars,
    }


def score_emit_timeline(records: list[dict], t_request_mono: float) -> dict:
    """Scorecard math shared by client-arrival and server-census timelines."""
    content = [r for r in records if r["channel"] == "content"]
    reasoning = [r for r in records if r["channel"] == "reasoning"]
    if not content:
        return {"content_emits": 0}
    gaps = [
        (b["t_mono"] - a["t_mono"]) * 1000
        for a, b in zip(content, content[1:])
    ]
    reasoning_gaps = [
        (b["t_mono"] - a["t_mono"]) * 1000
        for a, b in zip(reasoning, reasoning[1:])
    ]
    bursts = [float(r["chars"]) for r in content]
    window_s = content[-1]["t_mono"] - content[0]["t_mono"]
    total_chars = int(sum(bursts))
    emit = gap_stats(gaps)
    # SHIP BAR operationalization: "round gap" = p50 emit gap (the median
    # emit is one verify round on a clean stream); "one round's text" =
    # 2x median burst with a 48-char floor to absorb tokenizer jitter.
    p50 = emit["p50"] or 0
    burst_p50 = pct(bursts, 50) or 0
    burst_p95 = pct(bursts, 95) or 0
    return {
        "content_emits": len(content),
        "content_chars": total_chars,
        "content_window_s": round(window_s, 3),
        "content_chars_per_s": round(total_chars / window_s, 1) if window_s > 0 else None,
        "ttfc_ms": round((content[0]["t_mono"] - t_request_mono) * 1000, 1),
        "emit_gap_ms": emit,
        "burst_chars": {"p50": burst_p50, "p95": burst_p95,
                        "max": max(bursts) if bursts else None},
        "reasoning_gap_ms": gap_stats(reasoning_gaps),
        "ship_bar": {
            "stalls_over_150ms": emit["over_150ms"],
            "stalls_ok": emit["over_150ms"] == 0,
            "emit_gap_p95_ok": (emit["p95"] or 0) <= p50 * 1.2 if p50 else None,
            "burst_p95_ok": burst_p95 <= max(2 * burst_p50, 48),
        },
    }


def load_census_records(census_dir: Path, response_id: str) -> list[dict] | None:
    path = census_dir / f"{response_id.replace(':', '_')}.jsonl"
    if not path.exists():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records or None


def cmd_api(args: argparse.Namespace) -> int:
    out_root = Path(args.out) / args.label
    out_root.mkdir(parents=True, exist_ok=True)
    census_dir = Path(args.census_dir)

    thermal: dict = {"gated": not args.no_thermal_gate}
    if args.no_thermal_gate:
        print("[streamscope] WARNING: thermal gate SKIPPED by flag — numbers "
              "from this run are not comparable receipts.")
    else:
        thermal.update(run_fanmax_gate(out_root))
        if not thermal.get("compliant"):
            print("[streamscope] FATAL: fanmax gate not compliant; refusing to "
                  "run a model-loaded battery. See receipt:", thermal.get("receipt"))
            return 2

    prompt_keys = (
        [k.strip() for k in args.prompts.split(",") if k.strip()]
        if args.prompts
        else list(PROMPTS)
    )
    unknown = [k for k in prompt_keys if k not in PROMPTS]
    if unknown:
        print(f"[streamscope] unknown prompts: {unknown}; known: {list(PROMPTS)}")
        return 2

    summary: dict = {
        "created_at": now_iso(),
        "base_url": args.base_url,
        "model": args.model,
        "label": args.label,
        "thermal": thermal,
        "prompts": {},
    }

    for repeat in range(args.repeat):
        for key in prompt_keys:
            arm = key if args.repeat == 1 else f"{key}-r{repeat + 1}"
            arm_dir = out_root / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            if not args.no_thermal_gate and not thermal_mode_is_max():
                print("[streamscope] fan mode drifted off max — re-gating")
                regate = run_fanmax_gate(out_root)
                if not regate.get("compliant"):
                    print("[streamscope] FATAL: re-gate failed mid-battery.")
                    return 2
            macmon = MacmonSampler(arm_dir / "macmon.jsonl")
            macmon_ok = macmon.start()
            print(f"[streamscope] {arm}: streaming …", flush=True)
            t0_wall = time.time()
            try:
                run = run_sse_prompt(
                    args.base_url, args.model, PROMPTS[key], arm_dir / "sse_client.jsonl"
                )
            finally:
                time.sleep(1.0)
                macmon.stop()
            t1_wall = time.time()

            card = {
                "arm": arm,
                "prompt_key": key,
                "response_id": run["response_id"],
                "client": score_emit_timeline(run["records"], run["t_request_mono"]),
                "census": None,
                "transport_skew_ms_p95": None,
                "usage": run["usage"],
                "mtplx_stats_subset": {
                    k: run["mtplx_stats"].get(k)
                    for k in (
                        "decode_tok_s", "prefill_tok_s", "generated_tokens",
                        "reasoning_tokens", "answer_tokens",
                        "producer_gap_ms_p95", "producer_gap_ms_max",
                        "producer_gaps_over_200ms",
                    )
                } if run["mtplx_stats"] else None,
                "reasoning_chars": run["reasoning_chars"],
                "macmon": macmon_window_stats(arm_dir / "macmon.jsonl", t0_wall, t1_wall)
                if macmon_ok else None,
            }
            if run["response_id"]:
                census_records = load_census_records(census_dir, run["response_id"])
                if census_records:
                    shutil.copy(
                        census_dir / f"{run['response_id'].replace(':', '_')}.jsonl",
                        arm_dir / "census.jsonl",
                    )
                    card["census"] = score_emit_timeline(
                        census_records, census_records[0]["t_mono"]
                    )
                    # Transport skew: census wall stamp vs client wall stamp,
                    # matched pairwise on content records in order.
                    census_content = [r for r in census_records if r["channel"] == "content"]
                    client_content = [r for r in run["records"] if r["channel"] == "content"]
                    skews = [
                        (c2["t_wall"] - c1["t_wall"]) * 1000
                        for c1, c2 in zip(census_content, client_content)
                    ]
                    card["transport_skew_ms_p95"] = pct(skews, 95)
            # Engine-vs-eye headline: engine mean rate over the content
            # window vs what actually crossed the wire per second.
            stats = card["mtplx_stats_subset"] or {}
            client = card["client"]
            if stats.get("decode_tok_s") and client.get("content_window_s"):
                gen = stats.get("generated_tokens") or 0
                window_tok_s = (
                    round(gen / client["content_window_s"], 2)
                    if gen and client["content_window_s"] > 0 else None
                )
                card["engine_decode_tok_s"] = stats["decode_tok_s"]
                card["window_tok_s"] = window_tok_s

            (arm_dir / "scorecard.json").write_text(
                json.dumps(card, indent=2) + "\n", encoding="utf-8"
            )
            (arm_dir / "response.json").write_text(
                json.dumps(
                    {
                        "content_chars": len(run["content_text"]),
                        "content_head": run["content_text"][:400],
                        "content_tail": run["content_text"][-400:],
                        "usage": run["usage"],
                        "mtplx_stats": run["mtplx_stats"],
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            summary["prompts"][arm] = {
                "emit_gap_ms_p95": client.get("emit_gap_ms", {}).get("p95"),
                "emit_gap_ms_max": client.get("emit_gap_ms", {}).get("max"),
                "stalls_over_150ms": client.get("ship_bar", {}).get("stalls_over_150ms"),
                "burst_p95": client.get("burst_chars", {}).get("p95"),
                "chars_per_s": client.get("content_chars_per_s"),
                "ship_bar_ok": (
                    client.get("ship_bar", {}).get("stalls_ok"),
                    client.get("ship_bar", {}).get("emit_gap_p95_ok"),
                    client.get("ship_bar", {}).get("burst_p95_ok"),
                ),
            }
            gap = client.get("emit_gap_ms", {})
            print(
                f"[streamscope] {arm}: emits={client.get('content_emits')} "
                f"gap p50/p95/max = {gap.get('p50')}/{gap.get('p95')}/{gap.get('max')} ms "
                f">150ms={gap.get('over_150ms')} burst_p95={client.get('burst_chars', {}).get('p95')}",
                flush=True,
            )
            if args.cooldown_s and (repeat, key) != (args.repeat - 1, prompt_keys[-1]):
                time.sleep(args.cooldown_s)

    (out_root / "battery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[streamscope] battery summary -> {out_root / 'battery_summary.json'}")
    return 0


# ------------------------------------------------------------ app-collect


def parse_since(text: str) -> float:
    return datetime.fromisoformat(text).astimezone().timestamp()


def cmd_app_collect(args: argparse.Namespace) -> int:
    since = parse_since(args.since)
    out_dir = Path(args.out) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    aime_events: list[dict] = []
    traces: list[tuple[Path, list[dict]]] = []
    for path in sorted(APP_DIAG_DIR.glob("*.jsonl")):
        if path.stat().st_mtime < since:
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if path.name.startswith("aime-"):
            aime_events.extend(rows)
        elif path.name.startswith("uistream-"):
            traces.append((path, rows))
        shutil.copy(path, out_dir / path.name)

    summaries = [e for e in aime_events if e.get("name") == "ui_turn_render_summary"]
    card: dict = {
        "created_at": now_iso(),
        "label": args.label,
        "since": args.since,
        "aime_files_events": len(aime_events),
        "uistream_traces": [p.name for p, _ in traces],
        "turn_summaries": [e.get("fields") for e in summaries],
    }

    # Merge every source we have into one wall-clock timeline.
    timeline: list[dict] = []
    for trace_path, rows in traces:
        anchor = next((r for r in rows if r.get("kind") == "turn"), None)
        if not anchor or "t_wall" not in anchor:
            continue
        offset = anchor["t_wall"] - anchor["t_uptime"]
        for row in rows:
            if "t" not in row:
                continue
            entry = dict(row)
            entry["t_wall"] = round(row["t"] + offset, 4)
            entry["src"] = f"app:{row.get('kind')}"
            timeline.append(entry)
    engine_run = Path(args.engine_run) if args.engine_run else None
    if engine_run:
        for name, src in (("census.jsonl", "census"), ("sse_client.jsonl", "sse_client")):
            path = engine_run / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["src"] = src
                timeline.append(row)
        macmon_path = engine_run / "macmon.jsonl"
        if macmon_path.exists():
            for line in macmon_path.read_text(encoding="utf-8").splitlines():
                try:
                    sample = json.loads(line)
                    ts = datetime.fromisoformat(sample["timestamp"]).timestamp()
                except Exception:
                    continue
                timeline.append({
                    "t_wall": ts, "src": "macmon",
                    "cpu_pct": round(sample.get("cpu_usage_pct", 0) * 100, 1),
                    "gpu_pct": round((sample.get("gpu_usage") or [0, 0])[1] * 100, 1),
                })
    timeline.sort(key=lambda r: r.get("t_wall", 0))
    with open(out_dir / "timeline.jsonl", "w", encoding="utf-8") as sink:
        for row in timeline:
            sink.write(json.dumps(row, separators=(",", ":")) + "\n")

    # Perceived-TPS ratio: painted chars/s (app flushes) over engine visible
    # chars/s (census if present, else client arrivals).
    flushes = [r for r in timeline if r.get("src") == "app:flush"]
    engine_rows = [
        r for r in timeline
        if r.get("src") in ("census", "sse_client") and r.get("channel") == "content"
    ]
    if flushes and engine_rows:
        painted = sum(r.get("drained_bytes", 0) for r in flushes)
        painted_window = flushes[-1]["t_wall"] - flushes[0]["t_wall"]
        emitted = sum(r.get("chars", 0) for r in engine_rows)
        emitted_window = engine_rows[-1]["t_wall"] - engine_rows[0]["t_wall"]
        if painted_window > 0 and emitted_window > 0 and emitted:
            painted_rate = painted / painted_window
            emitted_rate = emitted / emitted_window
            card["perceived_tps_ratio"] = round(painted_rate / emitted_rate, 3)
            card["painted_chars_per_s"] = round(painted_rate, 1)
            card["emitted_chars_per_s"] = round(emitted_rate, 1)

    (out_dir / "app_scorecard.json").write_text(
        json.dumps(card, indent=2) + "\n", encoding="utf-8"
    )
    latest = summaries[-1]["fields"] if summaries and summaries[-1].get("fields") else {}
    print(f"[streamscope] app-collect: {len(summaries)} turn summaries, "
          f"{len(timeline)} timeline rows -> {out_dir}")
    if latest:
        print("[streamscope] latest turn:",
              json.dumps({k: latest.get(k) for k in (
                  "flush_gap_ms_p95", "apply_ms_p95", "draw_ms_p95",
                  "paint_gap_ms_p95", "paint_gaps_over_100ms",
                  "stalls_over_50ms") if k in latest}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    api = sub.add_parser("api", help="run the SSE prompt battery")
    api.add_argument("--base-url", required=True)
    api.add_argument("--model", required=True)
    api.add_argument("--label", required=True)
    api.add_argument("--out", default="outputs/streamscope-20260819")
    api.add_argument("--prompts", default="", help="comma list; default all")
    api.add_argument("--repeat", type=int, default=1)
    api.add_argument("--cooldown-s", type=float, default=15.0)
    api.add_argument("--census-dir", default=DEFAULT_CENSUS_DIR)
    api.add_argument("--no-thermal-gate", action="store_true")
    api.set_defaults(func=cmd_api)

    collect = sub.add_parser("app-collect", help="harvest + score app diagnostics")
    collect.add_argument("--since", required=True, help="ISO timestamp (local)")
    collect.add_argument("--label", required=True)
    collect.add_argument("--out", default="outputs/streamscope-20260819")
    collect.add_argument("--engine-run", default="",
                         help="api-run arm dir to merge census/client/macmon from")
    collect.set_defaults(func=cmd_app_collect)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
