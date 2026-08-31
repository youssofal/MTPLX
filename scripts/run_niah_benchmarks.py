#!/usr/bin/env python3
"""Needle-In-A-Haystack (NIAH) / Passkey Retrieval Quality Benchmark for MTPLX.

Evaluates multi-depth factual recall accuracy, prefill throughput, and peak memory
across context rungs (16k up to 262k / 1M).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# Reusable paragraphs for haystack generation
ESSAY_CORPUS = [
    (
        "Computer architecture has seen fundamental shifts with the advent of unified memory "
        "and wide-vector single-instruction-multiple-data operations on Apple Silicon. By eliminating "
        "explicit host-to-device PCI Express copies, heterogeneous execution units can operate on shared "
        "backing allocations with zero serialization overhead. However, maintaining strict temporal locality "
        "and minimizing cache eviction across shared L2 buffers remains a vital optimization."
    ),
    (
        "In modern compiler design, graph fusion transforms an acyclic dataflow representation by combining "
        "contiguous elementwise and reduction operators into a single fused compute kernel. This reduces "
        "intermediate DRAM traffic from O(N) to O(1) per layer pass, amortizing threadgroup launch latency "
        "and preventing register spills across SIMD boundaries."
    ),
    (
        "Rotary Position Embeddings (RoPE) encode relative position information directly into query and key "
        "projections via complex rotation matrices. When extending context beyond pre-training boundaries, "
        "YaRN (Yet another RoPE extensioN) applies progressive frequency interpolation across wave modes, "
        "preserving high-frequency positional discrimination while smoothly scaling low-frequency components."
    ),
    (
        "Linear attention mechanisms and state space models replace quadratic self-attention matrices with "
        "recurrent associative state updates. The Gated DeltaNet architecture introduces an input-dependent "
        "decay and fast update rule that allows streaming context digestion with constant memory per step, "
        "blending the throughput of linear recurrences with full attention fidelity at critical layers."
    ),
    (
        "Block-sparse attention restricts computation to top-K most relevant key-value blocks identified by "
        "a lightweight query-selected indexer. Rather than materializing dense boolean masks of size S x T, "
        "the attention consumer directly streams selected four-token memory blocks from backing tensors into "
        "threadgroup memory, achieving near-linear scaling at multi-million token sequences."
    ),
]


def _generate_haystack(
    target_tokens: int,
    needle: str,
    needle_depth_fraction: float,
) -> tuple[str, int]:
    """Generate a haystack of target_tokens with needle at depth_fraction."""
    # 5 paragraphs in ESSAY_CORPUS = 319 tokens (~63.8 tokens/paragraph).
    # Task suffix adds ~30 tokens, needle adds ~20 tokens.
    n_paras = max(1, int(math.ceil((target_tokens - 50) / 63.8)))
    blocks = [ESSAY_CORPUS[i % len(ESSAY_CORPUS)] for i in range(n_paras)]

    # Insert needle at the requested fraction
    insertion_idx = max(0, min(len(blocks) - 1, int(len(blocks) * needle_depth_fraction)))
    needle_sentence = f"\n\nNOTICE: The secret passkey for retrieval verification is {needle}. Please retain this key.\n\n"
    blocks.insert(insertion_idx, needle_sentence)
    
    haystack = " ".join(blocks)
    return haystack, len(blocks)


def _query_server(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{prompt}\n\nTask: What is the secret passkey for retrieval verification mentioned in the text? State the exact passkey.",
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-MTPLX-Cache-Mode": "bypass",
        },
        method="POST",
    )
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=2400) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    wall_time = time.perf_counter() - t0
    
    body["wall_time_s"] = wall_time
    return body


def main():
    parser = argparse.ArgumentParser(description="Run NIAH passkey retrieval benchmarks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="mtplx-flash-next-bare-speed")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[16384, 32768, 65536, 131072, 262144],
        help="Context length rungs in tokens",
    )
    parser.add_argument(
        "--positions",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.75, 0.90],
        help="Needle depth fractions (0.0 to 1.0)",
    )
    parser.add_argument("--output", default="benchmarks/results/niah_bare_speed_receipts.json")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MTPLX Passkey Retrieval / Needle-In-A-Haystack Benchmark")
    print(f"Model: {args.model}")
    print(f"Context Rungs: {args.depths}")
    print(f"Needle Positions: {[f'{int(p*100)}%' for p in args.positions]}")
    print("=" * 80)
    print()

    results = []
    total_runs = len(args.depths) * len(args.positions)
    run_idx = 0

    for depth in args.depths:
        for pos in args.positions:
            run_idx += 1
            key_suffix = uuid.uuid4().hex[:8].upper()
            needle = f"PASSKEY-{key_suffix}"

            haystack, num_blocks = _generate_haystack(depth, needle, pos)
            pos_pct = f"{int(pos * 100)}%"
            print(f"[{run_idx}/{total_runs}] Testing Depth={depth} tokens, Needle @ {pos_pct} ...", end=" ", flush=True)

            try:
                resp = _query_server(args.base_url, args.model, haystack)
                content = resp["choices"][0]["message"]["content"].strip()
                stats = resp.get("mtplx_stats", {})
                usage = resp.get("usage", {})

                success = needle in content
                prompt_tokens = usage.get("prompt_tokens", 0)
                ttft_s = stats.get("ttft_s", resp["wall_time_s"])
                prefill_tok_s = stats.get("prefill_tok_s", (prompt_tokens / max(0.001, ttft_s)))
                decode_tok_s = stats.get("decode_tok_s", 0.0)
                peak_mem_gb = stats.get("peak_memory_bytes", 0) / (1024**3)

                status_str = "MATCH" if success else "MISSED"
                print(f"[{status_str}] (Prompt: {prompt_tokens} tok, TTFT: {ttft_s:.2f}s, Prefill: {prefill_tok_s:.1f} tok/s, Decode: {decode_tok_s:.1f} tok/s, Peak: {peak_mem_gb:.2f} GB)")
                if not success:
                    print(f"   Expected: {needle}")
                    print(f"   Received: {content[:120]}")

                record = {
                    "context_target": depth,
                    "needle_position_fraction": pos,
                    "needle_position_percent": pos_pct,
                    "needle": needle,
                    "retrieved_content": content,
                    "success": success,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "ttft_s": ttft_s,
                    "wall_time_s": resp["wall_time_s"],
                    "prefill_tok_s": prefill_tok_s,
                    "decode_tok_s": decode_tok_s,
                    "peak_memory_gb": peak_mem_gb,
                    "prefill_route": stats.get("prefill_route", ""),
                }
                results.append(record)

            except Exception as exc:
                print(f"[ERROR] {exc}")
                results.append({
                    "context_target": depth,
                    "needle_position_fraction": pos,
                    "error": str(exc),
                    "success": False,
                })

            # Save incrementally
            out_path.write_text(json.dumps(results, indent=2))

    print()
    print("=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    passed_count = sum(1 for r in results if r.get("success"))
    print(f"Total Tests: {len(results)}, Passed: {passed_count}/{len(results)} ({passed_count/max(1, len(results))*100:.1f}%)")
    print(f"Receipts saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
