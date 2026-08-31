#!/usr/bin/env python3
"""Test decode throughput across prefill context rungs up to 546k on Bare-Speed."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_niah_benchmarks import _generate_haystack, _query_server

RUNGS = [16384, 32768, 65536, 131072, 262144, 524288, 546000]


def main():
    parser = argparse.ArgumentParser(description="Test decode tok/s across prefill depth rungs up to 546k")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="mtplx-flash-next-bare-speed")
    parser.add_argument("--depths", type=int, nargs="+", default=RUNGS)
    parser.add_argument("--output", default="benchmarks/results/decode_speed_546k_receipts.json")
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("MTPLX Decode Throughput Across Prefill Depth Rungs (up to 546k)")
    print(f"Model: {args.model}")
    print(f"Context Rungs: {args.depths}")
    print(f"Max Output Tokens: {args.max_tokens}")
    print("=" * 90)
    print()

    results = []

    for idx, depth in enumerate(args.depths, 1):
        pos = 0.50
        key_suffix = uuid.uuid4().hex[:8].upper()
        needle = f"PASSKEY-{key_suffix}"

        haystack, num_blocks = _generate_haystack(depth, needle, pos)
        print(f"[{idx}/{len(args.depths)}] Testing Prefill Depth ~{depth} tokens (Needle @ 50%) ...", flush=True)

        t0 = time.time()
        try:
            resp = _query_server(args.base_url, args.model, haystack, max_tokens=args.max_tokens)
            content = resp["choices"][0]["message"]["content"].strip()
            stats = resp.get("mtplx_stats", {})
            usage = resp.get("usage", {})
            timings = resp.get("timings", {})

            success = needle in content
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            ttft_s = stats.get("ttft_s", resp.get("wall_time_s", 0))
            prefill_tok_s = stats.get("prefill_tok_s", (prompt_tokens / max(0.001, ttft_s)))
            decode_tok_s = stats.get("decode_tok_s", timings.get("predicted_per_second", 0.0))
            peak_mem_gb = stats.get("peak_memory_bytes", 0) / (1024**3)
            decode_flash_skip = stats.get("decode_flash_skip", 0)
            decode_dense_mask = stats.get("decode_dense_mask", 0)

            status_str = "MATCH" if success else "MISSED"
            print(
                f"   Result: [{status_str}] Prompt: {prompt_tokens:,} tok | "
                f"TTFT: {ttft_s:.2f}s | Prefill: {prefill_tok_s:.1f} tok/s | "
                f"Decode: {decode_tok_s:.2f} tok/s | Peak RAM: {peak_mem_gb:.2f} GB | "
                f"Flash-Skip: {decode_flash_skip}"
            )
            if not success:
                print(f"   Expected: {needle}")
                print(f"   Received: {content[:120]}")

            record = {
                "context_target": depth,
                "needle": needle,
                "retrieved_content": content,
                "success": success,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "ttft_s": ttft_s,
                "wall_time_s": resp.get("wall_time_s", 0),
                "prefill_tok_s": prefill_tok_s,
                "decode_tok_s": decode_tok_s,
                "peak_memory_gb": peak_mem_gb,
                "decode_flash_skip": decode_flash_skip,
                "decode_dense_mask": decode_dense_mask,
                "prefill_route": stats.get("prefill_route", ""),
                "mtp_depth": stats.get("mtp_depth", 0),
                "qsa_prefill_engagement": stats.get("qsa_prefill_engagement", {}),
            }
            results.append(record)

        except Exception as exc:
            print(f"   [ERROR] {exc}")
            results.append({
                "context_target": depth,
                "error": str(exc),
                "success": False,
            })

        out_path.write_text(json.dumps(results, indent=2))
        print()

    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Target':<10} | {'Prompt Tokens':<14} | {'Prefill (tok/s)':<16} | {'Decode (tok/s)':<15} | {'Peak RAM (GB)':<13} | {'Recall'}")
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['context_target']:<10} | ERROR: {r['error']}")
        else:
            recall_str = "100% (MATCH)" if r["success"] else "FAIL (MISSED)"
            print(
                f"{r['context_target']:<10} | {r['prompt_tokens']:<14,d} | "
                f"{r['prefill_tok_s']:<16.1f} | {r['decode_tok_s']:<15.2f} | "
                f"{r['peak_memory_gb']:<13.2f} | {recall_str}"
            )
    print("=" * 90)


if __name__ == "__main__":
    main()
