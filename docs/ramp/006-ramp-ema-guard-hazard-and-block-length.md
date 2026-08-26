# 006 — RAMP: EMA-guard hazard (measured, not fixed) and the block-length pin

**Phase:** 4D (Zero-bandwidth CPU drafting) · **Date:** 2026-08-26
**Card:** `docs/reviews/2026-08-26-ramp/SC-P4DA-ramp-serving-path-proposer.md`
(acceptance criteria #4 and Fix step 5). This record exists because acceptance
criterion #4 requires the EMA-guard hazard be "recorded in `docs/decisions/`
as a known hazard with its measured values" — not fixed; fixing it would be
an engine diff, out of this card's no-diff scope.

---

## 1. Block length: pinned at 48, on the A/B alone, per the card's documented fallback

Fix step 5 requires re-running `scripts/ramp_block_sweep_clean.sh 32 48 64 96`
after checking `sysctl vm.swapusage` and aborting if used swap exceeds ~1 GB
(POC-FINDINGS.md §6a's own contamination mechanism).

At dispatch time on this machine:

```
$ sysctl vm.swapusage
vm.swapusage: total = 7168.00M  used = 5752.25M  free = 1415.75M  (encrypted)
```

5.75 GB used, ~5.75x the abort threshold — this is the exact condition that
invalidated the earlier clean-sweep attempt (POC-FINDINGS.md §6a: 5.75 GB of
7 GB in swap collapsed a 179–220 t/s configuration to 62.7 t/s). Per the
card's own instruction, **the sweep was not run.**

Per the card's documented fallback (Fix step 5, final paragraph): **block
length is pinned at 48**, resting on the interleaved A/B evidence alone
(`docs/reviews/2026-08-26-ramp/evidence/ab-bench.json`: RAMP arm at
`RAMP_BLOCK=48` measured +53.9% median decode t/s, +51%–63% per case) —
**not** on a block-length sweep. The clean sweep re-run (32/48/64/96, with a
swap check first) is escalated to a **blocking item on the long-context
follow-up card**, alongside the roofline break-even measurement the card's
"Largest open risk" already names as that card's first task.

The two clean cells that do exist (`evidence/block-sweep-clean-24.json`,
`-32.json`) are consistent with the direction (24→32 gains wall-clock,
accepted/round rises, block acceptance falls) but do not bracket 48 and
cannot be used to justify it.

**Live confirmation (acceptance criterion #6), run by this coder under the
same elevated-swap conditions**, via `scripts/launch_ramp_server.sh` /
`scripts/launch_baseline_server.sh` and `scripts/ramp_ab_bench.py 5`
(`evidence/ab-bench-coder-verification.json`, kept distinct from the frozen
POC evidence file rather than overwriting it): RAMP at block=48 beat stock
baseline on median decode t/s (177.4 vs 108.2 t/s, +63.9%) and matched the
POC's own block-acceptance ratio exactly (0.806), with temperature-0 output
byte-identical on all three cases. The prime-directive gate is green under
this run too. Absolute numbers are lower than the POC's original run (both
arms depressed similarly by the same background swap pressure noted above),
but the **relative** result — RAMP beats baseline, output identity holds —
reproduces.

## 2. EMA-guard hazard: measured, not fixed

The engine suspends context-copy drafting when
`ccopy_ema < 0.35` after `ccopy_seen >= 4` rounds
(`generation.py:7992-8003`). Longer blocks lower per-block acceptance while
raising absolute accepted tokens, so the guard's control variable and this
card's objective point in opposite directions. The card requires this be
measured and recorded, not fixed.

### Suspension incidence: RAMP-at-48 vs stock baseline (live, TWO independent runs)

Measured twice, independently: once in the POC's original bench
(`evidence/ab-bench.json`, the frozen card evidence) and once by this coder
as this card's own acceptance criterion #6 live confirmation
(`evidence/ab-bench-coder-verification.json`, launched via
`scripts/launch_ramp_server.sh` / `scripts/launch_baseline_server.sh`,
`scripts/ramp_ab_bench.py 5`). **Both runs agree exactly:**

| arm | case | n runs | suspensions | incidence |
|---|---|---|---|---|
| baseline | add-method | 5 | 5 (1/run) | 100% |
| baseline | rename-identifier | 5 | 0 | 0% |
| baseline | docstring-edit | 5 | 0 | 0% |
| ramp (block=48, fuzzy=1) | add-method | 5 | 5 (1/run) | 100% |
| ramp (block=48, fuzzy=1) | rename-identifier | 5 | 0 | 0% |
| ramp (block=48, fuzzy=1) | docstring-edit | 5 | 0 | 0% |
| **baseline total** | — | **15** | **5** | **33.3%** |
| **ramp total** | — | **15** | **5** | **33.3%** |

(POC run: median t/s baseline 138.2 / ramp 212.6, +53.9%, block acceptance
0.920 / 0.806. Coder-verification run, same machine, later, with elevated
background swap pressure from an unrelated prior session — see §1 —
depressed absolute numbers but an unchanged relative result: median t/s
baseline 108.2 / ramp 177.4, +63.9%, block acceptance 0.920 / 0.806 —
**identical block-acceptance ratios across both runs**, and the identical
per-case suspension pattern above. Temperature-0 output identity: True on
all 3 cases in both runs.)

**At the chosen block length (48), RAMP's suspension incidence is identical
to stock baseline's** — same case, same count, same rate, reproduced twice.
This is a stronger statement than the card's own draft language ("latent
hazard"): at 48 specifically, live-measured twice independently, RAMP does
not suspend more often than stock. Stock already suspends on `add-method`
every single run; RAMP inherits that, not a new failure mode.

### Minimum EMA excursion (offline instrumentation, deterministic replay at temperature 0)

The engine does not expose the live EMA value itself, only suspension
counts. Per acceptance criterion #4 ("if obtainable from ... instrumentation
added for this purpose"), the EMA trajectory was computed offline via the
same deterministic temperature-0 replay method the rest of this card's
evidence uses (`rafale.draft.ramp.RampIndex`, block=48, fuzzy=True, against
the three committed traces):

| trace | suspensions | min EMA at suspension | min EMA overall |
|---|---|---|---|
| rename-identifier | 0 | — (never suspends) | 0.367 (17 pts above threshold) |
| add-method | 1 | **0.288** | 0.288 |
| docstring-edit | 0 | — (never suspends) | 0.650 |

This offline computation reproduces the live suspension pattern exactly (1
suspension on `add-method`, 0 on the other two cases) — cross-validating the
replay method against the live A/B a second time, independent of the
fidelity check in acceptance criterion #2.

`rename-identifier`'s minimum EMA (0.367) comes within 0.017 of the 0.35
suspend threshold without crossing it — a real but non-firing margin at this
block length, worth watching if block length grows in a future card.

### Consequence

- **Do not read "0.806 aggregate block acceptance" as a safety margin** — the
  card's own earlier language did this and VERDICT-SC-P4DA.md §1.2 correctly
  attacked it; the EMA is a local excursion statistic, not the run-level
  average.
- At block=48, the comparison that matters (RAMP-vs-stock incidence) is a
  wash: 33.3% both arms.
- The guard is not fixed here (out of scope) and is not expected to need
  fixing at block=48 on this workload. A longer block in a future card
  (block=64/96, once the swap-clean sweep runs) should re-run this same
  suspension-incidence comparison before shipping, since the invalidated
  sweep's raw counters (contaminated for wall-clock, not for suspension
  counts) show 6/9 rows suspending at blocks 64/96 vs 3/9 at ≤48 —
  unconfirmed by clean data, but a reason to re-check rather than assume the
  33.3%-both-arms result at 48 generalizes upward.

## 3. What this decision does NOT do

- Does not change the engine's EMA-suspend control law (out of this card's
  no-diff scope; would be an engine diff).
- Does not validate at 128K–256K context — all numbers above are ~800-token
  short-context evidence (see the card's "Largest open risk").
- Does not run the block-length sweep — deferred to the long-context
  follow-up card, blocking, per §1 above.
