#!/usr/bin/env python3
"""Multi-turn chat against a running server, to measure prefix reuse.

Makes the caching claim reproducible the same way `dense_batch_bench`
makes the throughput claims reproducible. Point it at a server with the
prefix cache enabled and it reports how much of each prompt was NOT
re-read, straight off the wire.

Original note:

Does prefix caching help on the traffic it exists for: multi-turn chat?

The overhead arm (2026-08-25) priced the cache when it CANNOT help: distinct
sessions, zero restores, no reuse available. It found no consistent penalty.
That is a necessary result and not a sufficient one, because it never measured
the gain.

This measures the gain, on the shape of traffic the feature targets. Several
conversations run concurrently; each turn resends the whole history plus a new
message, so turn N+1's prompt begins with turn N's prompt. The bank is keyed by
content (`longest_prefix`), so reuse should follow without any session header.

Reads `usage.prompt_tokens_details.cached_tokens` off the wire rather than
server internals: that is the number a caller can actually see, and a gain the
caller cannot observe is not a gain worth claiming.

    cache_multiturn_bench.py --base-url URL --model M [--conversations 8]
                             [--turns 6] [--out FILE]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics as st
import time
import urllib.error
import urllib.request

# Each turn adds a substantial user message, so the shared prefix grows
# turn over turn the way a real conversation's does.
# A long system prompt, identical across every conversation. This is the
# realistic dominant source of prefix reuse: chat apps and agent harnesses
# resend the same large preamble on every single request, and avoiding that
# re-read is most of what prefix caching is for. It also guarantees every
# prompt clears MTPLX_DENSE_BATCH_PREFIX_MIN_TOKENS from turn zero, which the
# first version of this benchmark did not -- its prompts topped out at 107
# tokens against a 256-token floor, so nothing could ever be reused and the
# run measured 0.0% while proving nothing.
SYSTEM_PROMPT = (
    "You are a senior systems engineer advising on storage and database "
    "internals. Answer precisely and concretely, with concrete numbers where "
    "they matter. Prefer mechanisms over analogies. When a tradeoff exists, "
    "name both sides and say which you would take and why. "
) * 12

# Per-conversation topics. The first version of this benchmark gave every
# conversation the SAME question sequence, so four concurrent conversations sent
# byte-identical prompts and collided on one bank entry -- which measured
# contention, not caching. Real traffic is many DISTINCT conversations that
# share only a system preamble, so that is what this sends.
TOPICS = [
    ("B-trees", [
        "Explain how a B-tree keeps its balance during insertion, in detail.",
        "Now contrast that with a red-black tree. Where does each win?",
        "Given 90 percent reads over 400 GB, which would you pick and why?",
        "What changes if the storage is NVMe rather than spinning disk?",
        "Write out the split logic for a B-tree node in pseudocode.",
        "Walk through what happens when three splits cascade to the root.",
    ]),
    ("write-ahead logging", [
        "Explain how write-ahead logging guarantees durability, in detail.",
        "What exactly must be fsynced, and when, for that guarantee to hold?",
        "How does group commit change the latency and throughput picture?",
        "Describe recovery after a crash mid-checkpoint.",
        "Write out the redo loop in pseudocode.",
        "What breaks if the log and the data live on different devices?",
    ]),
    ("MVCC", [
        "Explain multi-version concurrency control and what it costs, in detail.",
        "How are old versions reclaimed, and when does that go wrong?",
        "Contrast snapshot isolation with serializable. Where do they differ?",
        "Describe a write-skew anomaly concretely.",
        "Write out visibility-check pseudocode for a tuple version.",
        "How would you size the version store for a heavy update workload?",
    ]),
    ("LSM trees", [
        "Explain how an LSM tree turns random writes into sequential ones.",
        "Describe leveled versus tiered compaction and their tradeoffs.",
        "What is write amplification here and how do you measure it?",
        "How do bloom filters change the read path?",
        "Write out the merge step of a compaction in pseudocode.",
        "When does an LSM lose to a B-tree, concretely?",
    ]),
    ("query planning", [
        "Explain how a cost-based optimizer picks a join order, in detail.",
        "Where do cardinality estimates come from and how do they go wrong?",
        "Contrast hash join, merge join and nested loop by workload.",
        "How would you detect a plan regression in production?",
        "Write out the dynamic-programming join enumeration in pseudocode.",
        "What would you change if statistics were always stale?",
    ]),
    ("replication", [
        "Explain synchronous versus asynchronous replication and the tradeoff.",
        "What does a quorum buy you, and what does it cost in latency?",
        "Describe split-brain and how fencing prevents it.",
        "How do you measure replication lag honestly?",
        "Write out leader-election pseudocode at a high level.",
        "What fails first when a follower falls permanently behind?",
    ]),
    ("buffer pools", [
        "Explain how a database buffer pool decides what to evict, in detail.",
        "Contrast LRU, clock and ARC for a mixed workload.",
        "How does prefetching interact with eviction policy?",
        "What happens under memory pressure from another process?",
        "Write out the page-pin and unpin protocol in pseudocode.",
        "How would you size a buffer pool for 400 GB of hot data?",
    ]),
    ("checksums and corruption", [
        "Explain how page checksums detect silent corruption, in detail.",
        "Where can corruption slip past a checksum entirely?",
        "Contrast CRC32C with a cryptographic hash for this job.",
        "How would you recover a single corrupted page in production?",
        "Write out the verify-on-read path in pseudocode.",
        "What monitoring would catch a failing drive before it lies to you?",
    ]),
]



def _post(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data, time.perf_counter() - t0


def run_conversation(
    base_url: str, model: str, conv_id: int, turns: int, max_tokens: int, timeout: float
) -> list[dict]:
    """One conversation, turns strictly sequential (that is what makes a prefix)."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    rows: list[dict] = []
    for t in range(turns):
        _topic, _questions = TOPICS[conv_id % len(TOPICS)]
        messages.append(
            {"role": "user", "content": _questions[t % len(_questions)]}
        )
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "temperature": 0.0,
        }
        try:
            data, wall = _post(url, payload, timeout)
        except urllib.error.HTTPError as e:
            rows.append({"conv": conv_id, "turn": t, "error": f"HTTP {e.code}"})
            break
        except Exception as e:  # noqa: BLE001 - record and continue
            rows.append({"conv": conv_id, "turn": t, "error": type(e).__name__})
            break

        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
        # A thinking model splits `content` from `reasoning_content`. At a low
        # max_tokens the whole budget can go to reasoning and `content` comes
        # back empty -- which silently stops the transcript growing, which is
        # exactly how the first run of this benchmark reported 0.0% reuse and
        # measured nothing. Fall back so a turn always contributes.
        reply = msg.get("content") or ""
        if not reply.strip():
            reply = msg.get("reasoning_content") or ""
        messages.append({"role": "assistant", "content": reply})

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        cached = int(details.get("cached_tokens") or 0)
        rows.append(
            {
                "conv": conv_id,
                "turn": t,
                "wall_s": wall,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached,
                "reuse_frac": (cached / prompt_tokens) if prompt_tokens else 0.0,
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--conversations", type=int, default=8)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print(
        f"[cache-bench] {args.label or 'run'}: {args.conversations} conversations "
        f"x {args.turns} turns, concurrent",
        flush=True,
    )
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.conversations
    ) as ex:
        futures = [
            ex.submit(
                run_conversation,
                args.base_url,
                args.model,
                c,
                args.turns,
                args.max_tokens,
                args.timeout,
            )
            for c in range(args.conversations)
        ]
        rows = [r for f in futures for r in f.result()]
    wall = time.perf_counter() - t0

    ok = [r for r in rows if "error" not in r]
    errs = [r for r in rows if "error" in r]
    # Turn 0 can never reuse anything: it is the first prompt of its
    # conversation. Reporting it in the mean would understate the effect.
    later = [r for r in ok if r["turn"] > 0]

    summary = {
        "label": args.label,
        "conversations": args.conversations,
        "turns": args.turns,
        "wall_s": wall,
        "requests_ok": len(ok),
        "requests_error": len(errs),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in ok),
        "total_cached_tokens": sum(r["cached_tokens"] for r in ok),
        "overall_reuse_frac": (
            sum(r["cached_tokens"] for r in ok) / sum(r["prompt_tokens"] for r in ok)
            if sum(r["prompt_tokens"] for r in ok)
            else 0.0
        ),
        "reuse_frac_turns_after_first": (
            st.mean([r["reuse_frac"] for r in later]) if later else 0.0
        ),
        "median_wall_s_turn0": (
            st.median([r["wall_s"] for r in ok if r["turn"] == 0])
            if any(r["turn"] == 0 for r in ok)
            else None
        ),
        "median_wall_s_later": (
            st.median([r["wall_s"] for r in later]) if later else None
        ),
        "rows": rows,
    }

    print(
        f"[cache-bench] {len(ok)} ok, {len(errs)} errors, {wall:.1f}s wall\n"
        f"[cache-bench] reuse: {summary['overall_reuse_frac']*100:.1f}% overall, "
        f"{summary['reuse_frac_turns_after_first']*100:.1f}% on turns after the first\n"
        f"[cache-bench] median wall: turn0 {summary['median_wall_s_turn0']}, "
        f"later {summary['median_wall_s_later']}",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[cache-bench] receipt -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
