#!/usr/bin/env python3
"""Pooled flat-or-better gate over the head sweep battery.

Single seeds are EOS-length-variable trajectories: per-seed tokens/cycle
swings ~±1.0 and one short outlier moves a 3-seed mean by 7%. The stable
head-quality signal is per-position acceptance POOLED across every paired
seed-row (2 cases x 3 seeds = 6 pairs per pack; if a pre-committed extension
run exists as sweep-<pack>-<arm>-ext.json its rows pool in too, giving 12
pairs). Gate per pack: pooled RC
acceptance within TOLERANCE of pooled base at every depth position. Also
reports the fleet-wide mean delta (directional flat-or-better) and counts
identical-trajectory seeds (quantized head agreeing with the base head
token-for-token — fidelity evidence, not an anomaly).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SWEEPS = Path(__file__).resolve().parent.parent / "outputs" / "head-restamp-20260820" / "sweeps"
TOLERANCE = 0.02
PACKS = [
    "Qwen3.8-27B-MTPLX-Optimized-Speed",
    "Qwen3.8-27B-MTPLX-Bare-Speed",
    "Qwen3.8-27B-MTPLX-Optimized-Quality",
    "Qwen3.8-27B-MTPLX-Optimized-Speed-FP16",
    "Qwen3.8-27B-MTPLX-Bare-Speed-FP16",
    "Qwen3.8-27B-MTPLX-Optimized-Quality-FP16",
]


def pooled(rows: list[dict]) -> list[float]:
    n_pos = max(len(r["acceptance_by_depth"]) for r in rows)
    return [
        sum(float(r["acceptance_by_depth"][i] or 0.0) for r in rows) / len(rows)
        for i in range(n_pos)
    ]


def main() -> None:
    failures = []
    fleet_deltas: list[list[float]] = []
    results = {}
    for pack in PACKS:
        base_path = SWEEPS / f"sweep-{pack}-base.json"
        rc_path = SWEEPS / f"sweep-{pack}-rc.json"
        if not (base_path.exists() and rc_path.exists()):
            print(f"PENDING  {pack}")
            continue
        base_rows = json.loads(base_path.read_text())["rows"]
        rc_rows = json.loads(rc_path.read_text())["rows"]
        for arm, rows in (("base", base_rows), ("rc", rc_rows)):
            ext_path = SWEEPS / f"sweep-{pack}-{arm}-ext.json"
            if ext_path.exists():
                rows.extend(json.loads(ext_path.read_text())["rows"])
        base_acc = pooled(base_rows)
        rc_acc = pooled(rc_rows)
        deltas = [rc - b for rc, b in zip(rc_acc, base_acc)]
        fleet_deltas.append(deltas)
        identical = sum(
            1
            for b, r in zip(
                sorted(base_rows, key=lambda x: (x["case"], x["seed"])),
                sorted(rc_rows, key=lambda x: (x["case"], x["seed"])),
            )
            if b["acceptance_by_depth"] == r["acceptance_by_depth"]
            and b["tokens_per_cycle"] == r["tokens_per_cycle"]
        )
        verdict = "PASS"
        for i, d in enumerate(deltas):
            if d < -TOLERANCE:
                verdict = f"FAIL pos{i + 1} {d:+.4f}"
                failures.append((pack, i + 1, d))
        results[pack] = {
            "base_pooled": [round(x, 4) for x in base_acc],
            "rc_pooled": [round(x, 4) for x in rc_acc],
            "deltas": [round(x, 4) for x in deltas],
            "identical_trajectory_seeds": identical,
            "paired_rows": len(base_rows),
            "verdict": verdict,
        }
        print(
            f"{verdict:24} {pack}\n"
            f"    base {results[pack]['base_pooled']}  rc {results[pack]['rc_pooled']}"
            f"  delta {results[pack]['deltas']}"
            f"  identical-seeds {identical}/{len(base_rows)}"
        )
    if fleet_deltas:
        n_pos = max(len(d) for d in fleet_deltas)
        fleet_mean = [
            round(sum(d[i] for d in fleet_deltas if len(d) > i) / len(fleet_deltas), 4)
            for i in range(n_pos)
        ]
        print(f"\nfleet mean delta by position: {fleet_mean} over {len(fleet_deltas)} packs")
    (SWEEPS / "gate-summary.json").write_text(json.dumps(results, indent=2) + "\n")
    if failures:
        print(f"\nGATE FAILED: {failures}")
        raise SystemExit(1)
    if len(fleet_deltas) == len(PACKS):
        print("\nGATE PASSED: all packs flat-or-better (pooled, tolerance -0.02)")


if __name__ == "__main__":
    main()
