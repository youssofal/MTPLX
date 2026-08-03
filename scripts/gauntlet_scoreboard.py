#!/usr/bin/env python3
"""Per-session scoreboard over the engine request-log JSONL.

Summarizes a live coding session against the product bars: decode floor
(default 40 tok/s), re-prefill hygiene (full re-prefills, salvage sizes),
TTFT distribution, restore-mode mix, and postcommit effectiveness.

Usage:
  python3 scripts/gauntlet_scoreboard.py [--log PATH] [--session SUBSTR]
      [--floor 40] [--min-out 30]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.expanduser(
        "~/.mtplx/logs/request-log-8001.jsonl"))
    ap.add_argument("--session", default="ses_")
    ap.add_argument("--floor", type=float, default=40.0)
    ap.add_argument("--min-out", type=int, default=30,
                    help="ignore decode readings on tiny outputs")
    args = ap.parse_args()

    rows = []
    with open(args.log, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            sid = str(r.get("session_id") or "")
            if args.session and args.session not in sid:
                continue
            rows.append(r)
    if not rows:
        print("no matching rows")
        return

    decs = [r["decode_tok_s"] for r in rows
            if (r.get("completion_tokens") or 0) >= args.min_out
            and r.get("decode_tok_s")]
    ttfts = [r.get("ttft_s") or 0 for r in rows]
    newpfs = [r.get("new_prefill_tokens") or 0 for r in rows]
    viol = [r for r in rows
            if (r.get("completion_tokens") or 0) >= args.min_out
            and (r.get("decode_tok_s") or 99) < args.floor]
    full_reprefill = [r for r in rows
                      if (r.get("new_prefill_tokens") or 0) > 4000
                      and (r.get("cached_tokens") or 0) < 512
                      and (r.get("prompt_tokens") or 0) > 4000]
    modes: dict[str, int] = {}
    for r in rows:
        m = str(r.get("session_restore_mode"))
        modes[m] = modes.get(m, 0) + 1

    def dist(vals, unit=""):
        if not vals:
            return "n/a"
        vs = sorted(vals)
        return (f"min {vs[0]:.1f} p50 {vs[len(vs)//2]:.1f} "
                f"p90 {vs[int(len(vs)*0.9)]:.1f} max {vs[-1]:.1f}{unit}")

    print(f"rows={len(rows)} sessions={len({r.get('session_id') for r in rows})}")
    print(f"decode (out>={args.min_out}): {dist(decs)} tok/s  "
          f"mean {statistics.mean(decs):.1f}" if decs else "decode: n/a")
    print(f"FLOOR<{args.floor}: {len(viol)}/{len(decs) or 1} real turns")
    for r in viol:
        print(f"  viol: dec={r['decode_tok_s']:.1f} out={r.get('completion_tokens')} "
              f"prompt={r.get('prompt_tokens')} newpf={r.get('new_prefill_tokens')} "
              f"ttft={r.get('ttft_s'):.1f}")
    print(f"ttft: {dist(ttfts, 's')}")
    print(f"newpf: {dist([float(n) for n in newpfs])}")
    print(f"full re-prefills (>4k cold): {len(full_reprefill)}")
    print(f"restore modes: {modes}")


if __name__ == "__main__":
    main()
