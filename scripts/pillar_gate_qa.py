#!/usr/bin/env python3
"""Pillar gate: pre-release pass/fail checks for the regressions users hit.

Every check here is scar tissue from a real founder-visible failure:

1. vision_cache   — one image in an agent history must NOT disable prompt
                    caching for the rest of the session (2026-07-09: one
                    screenshot -> every turn re-prefilled ~30k tokens, e2e
                    25-31 -> ~10 tok/s, ~100 GB system pressure). Also
                    asserts the correctness sentinel: a DIFFERENT image at
                    the same position must not restore past the image.
2. memory_ceiling — daemon MLX active memory during the run must stay
                    within weights + bank budget + working margin
                    (pillar 3: "no memory bloat" — the 50% RAM promise).
3. long_output_decay — decode tok/s over one long generation must not
                    collapse (pillar 1; founder: q8 at 13 tok/s @ 20k out).

Usage:
  pillar_gate_qa.py --base-url http://127.0.0.1:PORT   # existing daemon
  (the release script boots a scratch daemon and passes its URL)

Exit code 0 = all gates pass; 1 = any gate failed. JSON report on stdout.
Thermal rule: the caller must run this under verified max fans; pass
--fan-rpm-verified with the measured RPM (recorded into the report).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import struct
import sys
import time
import urllib.request
import zlib
from typing import Any


def make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    """Minimal solid-color PNG (no PIL dependency)."""

    def chunk(kind: bytes, payload: bytes) -> bytes:
        block = kind + payload
        return (
            struct.pack(">I", len(payload))
            + block
            + struct.pack(">I", zlib.crc32(block) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    body = zlib.compress(row * height, 6)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", body)
        + chunk(b"IEND", b"")
    )


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


CODE_BLOCK = (
    "def evaluate(board, depth, alpha, beta):\n"
    "    for mv in order_moves(board):\n"
    "        score = -evaluate(apply(board, mv), depth - 1, -beta, -alpha)\n"
    "        alpha = max(alpha, score)\n"
    "        if alpha >= beta: break\n"
    "    return alpha\n\n"
)


def build_context(target_tokens_approx: int) -> list[dict[str, Any]]:
    repeats = max(4, target_tokens_approx // 60)
    return [
        {"role": "system", "content": "You are a coding agent. Keep working."},
        {"role": "user", "content": "execute the plan: build chess.\n" + CODE_BLOCK * repeats},
        {"role": "assistant", "content": "Initial files created.\n" + CODE_BLOCK * (repeats // 2)},
        {"role": "user", "content": "Now write the styles."},
    ]


class Client:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def chat(self, messages, *, max_tokens: int, timeout: float = 1800):
        body = {
            "model": "default",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        ttft = None
        # (timestamp, cumulative_chars): chunk cadence is cadence-limited by
        # the stream interval, so decay must be measured in content
        # throughput, never chunk rate.
        progress: list[tuple[float, int]] = []
        chars = 0
        usage = None
        text = io.StringIO()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                try:
                    payload = json.loads(line[6:])
                except Exception:
                    continue
                if payload == "[DONE]":
                    break
                if isinstance(payload, dict) and payload.get("usage"):
                    usage = payload["usage"]
                for choice in payload.get("choices", []) if isinstance(payload, dict) else []:
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        now = time.time()
                        if ttft is None:
                            ttft = now - t0
                        chars += len(delta["content"])
                        progress.append((now, chars))
                        text.write(delta["content"])
        return {
            "wall_s": time.time() - t0,
            "ttft_s": ttft,
            "usage": usage or {},
            "progress": progress,
            "text": text.getvalue(),
        }

    def snapshot(self) -> dict[str, Any]:
        with urllib.request.urlopen(self.base_url + "/v1/mtplx/snapshot", timeout=15) as r:
            return json.loads(r.read())


def gate_vision_cache(client: Client, report: dict[str, Any]) -> bool:
    msgs = build_context(9000)
    r1 = client.chat(msgs, max_tokens=250)
    msgs.append({"role": "assistant", "content": r1["text"] or "(styles)"})
    msgs.append({"role": "user", "content": [
        {"type": "text", "text": "Look, the screen is blank "},
        {"type": "image_url", "image_url": {"url": data_url(make_png(1200, 800, (30, 60, 120)))}},
    ]})
    r2 = client.chat(msgs, max_tokens=250)
    snap2 = client.snapshot()
    msgs.append({"role": "assistant", "content": r2["text"] or "(diagnosis)"})
    msgs.append({"role": "user", "content": "Apply the fix."})
    r3 = client.chat(msgs, max_tokens=250)
    snap3 = client.snapshot()
    cached3 = int((snap3.get("latest") or {}).get("cached_tokens") or 0)
    prompt3 = int((snap3.get("latest") or {}).get("prompt_tokens") or 0)

    # Correctness sentinel: different pixels, same position -> the bank must
    # not restore at or past the image.
    diff = list(msgs[:-2])
    diff[-1] = {"role": "user", "content": [
        {"type": "text", "text": "Look, the screen is blank "},
        {"type": "image_url", "image_url": {"url": data_url(make_png(1200, 800, (200, 40, 40)))}},
    ]}
    r4 = client.chat(diff, max_tokens=120)
    snap4 = client.snapshot()
    cached4 = int((snap4.get("latest") or {}).get("cached_tokens") or 0)
    prompt2 = int((snap2.get("latest") or {}).get("prompt_tokens") or 0)
    image_tokens = max(1, prompt2 - int((r1["usage"] or {}).get("prompt_tokens") or 0))

    post_image_cache_ok = cached3 >= prompt3 - 4096  # follow-up mostly warm
    alias_blocked = cached4 <= (prompt2 - image_tokens)  # never past the image
    report["vision_cache"] = {
        "post_image_followup": {
            "prompt_tokens": prompt3,
            "cached_tokens": cached3,
            "wall_s": round(r3["wall_s"], 2),
            "pass": post_image_cache_ok,
        },
        "different_image_alias_blocked": {
            "prompt_tokens": prompt2,
            "cached_tokens": cached4,
            "image_tokens_approx": image_tokens,
            "pass": alias_blocked,
        },
    }
    return post_image_cache_ok and alias_blocked


def gate_memory_ceiling(client: Client, report: dict[str, Any]) -> bool:
    snap = client.snapshot()
    mem = snap.get("mem") or {}
    active = int(mem.get("active_memory_bytes") or 0)
    weights = int(mem.get("model_weights_bytes") or 0)
    bank = snap.get("session_bank") or {}
    bank_budget = int(bank.get("max_bytes") or 0)
    # Working margin: prefill transients + compiled buffers. 16 GiB is
    # generous for a 27B at 32k; the founder's complaint was 3-5x this.
    margin = 16 << 30
    ceiling = weights + bank_budget + margin
    ok = weights > 0 and active <= ceiling
    report["memory_ceiling"] = {
        "active_bytes": active,
        "weights_bytes": weights,
        "bank_budget_bytes": bank_budget,
        "working_margin_bytes": margin,
        "ceiling_bytes": ceiling,
        "pass": ok,
    }
    return ok


def gate_long_output_decay(
    client: Client, report: dict[str, Any], *, max_tokens: int
) -> bool:
    msgs = [
        {"role": "system", "content": "You are a meticulous engineer."},
        {
            "role": "user",
            "content": (
                "Write an extremely detailed, file-by-file implementation of a "
                "browser chess game with an AI opponent. Do not stop early; "
                "include full code for every file."
            ),
        },
    ]
    # The model occasionally answers this prompt with a short "I am ready to
    # proceed" preamble and a clean stop (temperature variance, seen 2026-07-31
    # at healthy 52 tok/s). That is insufficient DATA for a decay measurement,
    # not a decay failure — retry once with a fresh request before failing so
    # a one-in-N conversational flake cannot abort a release run.
    attempts = 0
    while True:
        attempts += 1
        result = client.chat(msgs, max_tokens=max_tokens)
        progress = result["progress"]
        total_chars = progress[-1][1] if progress else 0
        if len(progress) >= 100 and total_chars >= 4000:
            break
        if attempts >= 2:
            report["long_output_decay"] = {
                "pass": False,
                "reason": (
                    f"too little streamed content ({len(progress)} chunks, "
                    f"{total_chars} chars) in {attempts} attempts"
                ),
            }
            return False
        report.setdefault("long_output_decay_retries", []).append(
            {"chunks": len(progress), "chars": total_chars}
        )
    # Content throughput (chars/s) per output quintile: SSE chunk cadence is
    # pinned by the stream interval, so chunk rate is blind to decode decay —
    # a slowing decoder produces the same chunk rate with thinner chunks.
    quintile_chars = total_chars / 5
    boundaries: list[float] = []
    target = quintile_chars
    for ts, cum in progress:
        if cum >= target:
            boundaries.append(ts)
            target += quintile_chars
    if len(boundaries) < 5:
        boundaries.append(progress[-1][0])
    start_ts = progress[0][0]
    first_rate = quintile_chars / max(1e-6, boundaries[0] - start_ts)
    last_rate = quintile_chars / max(1e-6, boundaries[4] - boundaries[3])
    ratio = last_rate / max(1e-6, first_rate)
    completion_tokens = (result["usage"] or {}).get("completion_tokens")
    decode_window_s = progress[-1][0] - start_ts
    ok = ratio >= 0.65
    report["long_output_decay"] = {
        "chunks": len(progress),
        "completion_tokens": completion_tokens,
        "total_chars": total_chars,
        "mean_decode_tok_s": (
            round((completion_tokens - 1) / decode_window_s, 2)
            if completion_tokens and decode_window_s > 0
            else None
        ),
        "first_quintile_chars_s": round(first_rate, 1),
        "last_quintile_chars_s": round(last_rate, 1),
        "ratio": round(ratio, 3),
        "threshold": 0.65,
        "pass": ok,
    }
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--long-output-tokens", type=int, default=6000)
    parser.add_argument("--fan-rpm-verified", type=int, default=0)
    parser.add_argument(
        "--skip", action="append", default=[],
        choices=["vision_cache", "memory_ceiling", "long_output_decay"],
    )
    args = parser.parse_args(argv)

    client = Client(args.base_url)
    report: dict[str, Any] = {
        "base_url": args.base_url,
        "fan_rpm_verified": args.fan_rpm_verified,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    results: dict[str, bool] = {}
    if "vision_cache" not in args.skip:
        results["vision_cache"] = gate_vision_cache(client, report)
    if "memory_ceiling" not in args.skip:
        results["memory_ceiling"] = gate_memory_ceiling(client, report)
    if "long_output_decay" not in args.skip:
        results["long_output_decay"] = gate_long_output_decay(
            client, report, max_tokens=args.long_output_tokens
        )
    report["results"] = results
    report["pass"] = all(results.values()) if results else False
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
