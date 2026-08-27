# SC-P4DA — Install the RAMP proposer on the serving path via the no-diff context_copy seam

**Status: FROZEN for dispatch** (not yet dispatched; amended once before
dispatch — see *Revision history* at the end). Written by the Opus card writer,
2026-08-26.
Evidence: `docs/reviews/2026-08-26-ramp/evidence/` (POC code in `scripts/ramp_*`,
results committed alongside this card).

> **Read `POC-FINDINGS.md` in this directory before the card.** Two of the four
> mechanisms originally proposed for RAMP were **measured and disconfirmed**, and
> this card deliberately does not build them. A worker who "helpfully" adds them
> back is reintroducing a measured 52% regression.

## Source

Plan `docs/plans/ane-optimization-plan.md` **Phase 4D** ("Zero-bandwidth CPU
drafting: prompt-lookup fused with MTP"), read together with
`docs/decisions/005-engine-fork-decision.md` (seam tiers, adapter 2) and
`docs/decisions/004-gate-05-no-go-ane-prefill.md` (ANE is dead; speculation and
quant are the only decode levers left).

**This is a GREENFIELD addition, not a fix.** Confirmed by repo-wide grep:

```
$ grep -rn "ramp\|Ramp\|RAMP" rafale/ tests/
(no hits)
$ ls rafale/draft/
__init__.py  fusion.py  ngram.py
```

`rafale/draft/ngram.py` and `rafale/draft/fusion.py` exist but are unwired
scaffolds (`fusion.py` is a docstring and a `TODO(phase-4d)`, nothing else) that
re-implement what the engine already ships. They are **fenced out of scope**
below; deciding their fate is a separate card.

Target files are all new. No existing file is modified by this card.

## Defect / gap

The engine already ships an exact-n-gram prompt-lookup drafter. It works, and it
is live-verified. The gap is **not** that drafting is missing — it is that the
shipped drafter leaves a large, measured amount of throughput on the table for
two specific, independently-verified reasons.

### Gap 1 — the block-length ladder is far too short for a bus-bound machine

```python
# mtplx/context_copy.py:92-100
# Confidence ladder: block length by backward match extension (0..ng_max-ng_min).
# A longer suffix match earns a longer copy block, so a weak match only ever
# risks a short, cheap verify while a strong match copies a full window.
_BLOCK_LADDER = (8, 12, 16, 24, 32)


def block_for_ext(ext: int, k_cap: int) -> int:
    idx = max(0, min(int(ext), len(_BLOCK_LADDER) - 1))
    return min(_BLOCK_LADDER[idx], max(4, k_cap))
```

`ext` is capped at `ng_max - ng_min` = 10 - 6 = 4 (`context_copy.py:75-79`,
`:135`), so the reachable block lengths are exactly 8, 12, 16, 24, 32, and the
observed live mean is far below the cap: the baseline arm commits **20.52
accepted tokens per copy round** over 15 real runs.

The ladder's stated rationale — "a weak match only ever risks a short, cheap
verify" — optimizes the wrong quantity. The block is verified in **one** forward
pass of `T = 1 + len(block)` rows:

```python
# mtplx/generation.py:7845-7852
_cc_T = 1 + len(_cc_block)
...
_cc_logits, _cc_hidden, _cc_captures = rt.forward_ar_capture(
    mx.array([[primary] + _cc_block]),
    cache=cache, return_hidden=True,
    hidden_variant=base_hidden_variant,
    capture_backend=verify_core_backend,
)
```

On a machine where decode is bus-bound (CLAUDE.md rule 9: ~300 GB/s, weight
reads dominate), the marginal cost of extra rows in that single pass is small
relative to the weight read it amortizes. Shortening the block to protect
per-block acceptance therefore **buys a metric nobody is paid for and sells the
one that matters**. Measured, offline, over real traces: forcing acceptance up
to 0.970 by shrinking blocks costs **-52.4%** accepted-tokens-per-verify-pass.

### Gap 2 — the exact n-gram key goes dark across every edit divergence

```python
# mtplx/context_copy.py:131-133
cands = self.grams.get(tuple(history[-self.ng_min:]))
if not cands:
    return None, -1
```

The lookup key is an **exact** `ng_min`-gram (default 6) dict hit. When a coding
agent re-emits a file with a change — the target workload — the generated suffix
diverges from the prompt at every edit site, and the exact key cannot fire again
until `ng_min` freshly-aligned tokens have accumulated. Measured over the real
traces, the shipped drafter is dark on **34.3%** of decode cycles.

### Both gaps sit behind a seam decision 005 mis-tiers

Decision 005's seam table puts "External draft-token injection" at **Tier 3 —
`Yes`, a real patch** (`005-engine-fork-decision.md:122`). That is correct for
arbitrary tokens and **wrong for this card's needs**, because the drafter
symbols are imported *function-locally, on every call*:

```python
# mtplx/generation.py:7479-7482  (inside generate_mtpk, 4-space indent)
    from .context_copy import (NgramIndex, block_for_ext, context_copy_block_k,
                               context_copy_enabled, context_copy_min_ext,
                               context_copy_ng_max, context_copy_ng_min,
                               context_copy_target_prefix_enabled)
```

Rebinding those attributes on `mtplx.context_copy` before the server starts
replaces the proposer with **zero lines of engine source changed**. Proven live,
not inferred: the POC ran a fully patched server on :8002 for 15 real requests.

The envelope this buys is exactly *which contiguous prompt span is proposed, and
how long it is* — because the block is still sliced from `prompt_ids` by the
engine:

```python
# mtplx/generation.py:7835-7837
_cc_klen = block_for_ext(_cc_ext, ccopy_k)
_cc_block = [int(t) for t in prompt_ids[_cc_pos:_cc_pos + _cc_klen]]
```

**Both gaps above fall inside that envelope.** The POC measured every mechanism
that would have required Tier 3 and found no win (see *Pre-empting the obvious
wrong fix*), so this card needs no engine patch at all.

### Second defect site

There is one, and it is the reason acceptance criterion #4 exists. The engine's
EMA guard controls on **per-block acceptance ratio**:

```python
# mtplx/generation.py:7992-8003
ccopy_ema = 0.7 * ccopy_ema + 0.3 * (_cc_nacc / len(_cc_block))
ccopy_seen += 1
if _cc_nacc / len(_cc_block) >= 0.5:
    ccopy_backoff = 64          # copy is paying again: full retry rate
if ccopy_seen >= 4 and ccopy_ema < 0.35:
    # acceptance collapsed (novel region with incidental repeats):
    # suspend copy rounds and let the MTP head work; retry with
    # exponential backoff so recurring probes stay cheap
    ccopy_suspend_until = len(tokens) + ccopy_backoff
    ccopy_backoff = min(ccopy_backoff * 2, 4096)
```

Longer blocks *necessarily* lower this ratio — measured live, 0.920 → 0.806 —
while raising absolute accepted tokens (20.52 → 38.70 per round). The guard's
control variable and this card's objective therefore point in **opposite
directions**. At the measured 0.806 the guard does not fire (0.806 ≫ 0.35), so
this is a *latent* hazard, not a live bug: the margin narrows as block length
grows, and a longer default would eventually trip a guard that is suspending
drafting precisely when drafting is paying best. The card requires this be
**measured and recorded**, not fixed — fixing it means changing engine control
flow, which is out of this card's no-diff scope.

### Test coverage confirmed absent

`grep -rn "context_copy\|ramp\|block_for_ext\|NgramIndex" tests/` returns no
hits. `tests/test_ngram.py` (6 test functions) covers only the unwired
`rafale/draft/ngram.py` scaffold, not the engine seam and not RAMP.

## POC evidence — what was actually run

Everything below is measured. No number in this card is estimated or projected.

### (i) The offline replay is faithful to the engine

`scripts/ramp_replay.py` replays the decode cycle over real captured
(prompt_ids, output_ids) traces, importing the **real** `mtplx.context_copy`
rather than a transcription. At temperature 0 the target's acceptance is exactly
"longest common prefix of the proposal with the stream the target actually
produced", so acceptance is computable offline with no model in the loop.

Simulated vs. the live engine's own `context_copy_*` counters for the same three
requests (`evidence/replay-results.json`):

| trace | | rounds | drafted | accepted | blocks | suspensions |
|---|---|---|---|---|---|---|
| rename-identifier | engine | 30 | 652 | 580 | 30 | 0 |
| | sim (mtp_advance=3) | 32 | 674 | 602 | 31 | 0 |
| add-method | engine | 31 | 688 | 648 | 28 | 1 |
| | sim (mtp_advance=3) | 32 | 708 | 675 | 29 | 1 |
| docstring-edit | engine | 22 | 512 | 475 | 22 | 0 |
| | sim (mtp_advance=3) | 23 | 519 | 495 | 23 | 0 |

Within a few percent on every counter, at the MTP advance the engine's depth-2
configuration implies. Two things follow: the simulator is a trustworthy
stand-in, and the completion re-tokenization in `ramp_capture_traces.py` is
faithful.

### (ii) Ablation, one variable at a time (CLAUDE.md rule 3)

Accepted tokens per verify pass, aggregated over the three traces,
`mtp_advance=3`. `A1` is the control proving the RAMP proposer reduces exactly
to the engine's algorithm with every knob off.

| variant | tok/pass | vs V0 | blk acc | dark% |
|---|---|---|---|---|
| V0-engine-exact (baseline) | 14.044 | +0.0% | 0.932 | 34.3% |
| A1-control-ladder | 14.044 | **+0.0%** | 0.932 | 34.3% |
| A3-block-fixed32 | 16.586 | +18.1% | 0.866 | 40.5% |
| A4-block-fixed48 | 20.042 | +42.7% | 0.856 | 49.0% |
| A5-block-fixed64 | 22.115 | +57.5% | 0.815 | 54.0% |
| A6-block-fixed96 | 20.253 | +44.2% | 0.750 | 68.4% |
| B1-fuzzy-ladder | 15.031 | +7.0% | 0.934 | 25.8% |
| **B3-fuzzy-fixed48** | **22.905** | **+63.1%** | 0.824 | 39.3% |
| B4-fuzzy-fixed64 | 21.143 | +50.5% | 0.777 | 54.9% |
| C1-wide-ladder | 13.942 | **-0.7%** | 0.920 | 34.1% |
| D1-consensus-rank | 14.044 | **+0.0%** | 0.932 | 34.3% |
| E1-consensus-block | 6.681 | **-52.4%** | 0.968 | 16.3% |

### (iii) On metal: a real patched server, 15 runs per arm

`scripts/ramp_ab_bench.py`, alternating A/B/A/B so thermal drift hits both arms
equally, 5 reps × 3 cases, temperature 0 (`evidence/ab-bench.json`):

| arm | n | median t/s | p95 t/s | accepted/round | block acc |
|---|---|---|---|---|---|
| baseline (:8001, stock) | 15 | 138.2 | 143.9 | 20.52 | 0.920 |
| RAMP (:8002, patched) | 15 | **212.6** | 228.5 | **38.70** | 0.806 |

**+53.9% median wall-clock decode throughput.**

**Prime-directive gate: GREEN.** Temperature-0 output is byte-identical
(sha256) between arms on all three cases — `identical=True` for
`rename-identifier`, `add-method`, `docstring-edit`. This is structurally
expected (the copy block is a point-mass proposal verified by argmax match, so
the draft *source* cannot move the output law) and it was confirmed empirically
rather than assumed.

### (iv) Live block-length sweep

`scripts/ramp_block_sweep.sh` restarts the patched server per cell
(`evidence/block-sweep*.json`). Wall-clock, not the offline proxy:

| RAMP_BLOCK | n | median t/s | rounds | accepted/round | blk acc | output identical |
|---|---|---|---|---|---|---|
| 16 | 9 | 69.6 | 351 | 14.62 | 0.913 | True |
| 32 | 9 | 150.2 | 195 | 27.09 | 0.847 | True |
| 48 | 9 | 184.3 | 138 | 45.63 | 0.716 | True |
| 64 | 9 | 76.1 | 114 | 51.55 | 0.716 | True |
| 96 | 9 | 248.4 | 84 | 62.50 | 0.669 | True |

**These wall-clock figures are contaminated and must not be used to pick a block
length** — the sweep ran with the stock baseline server still resident, no
cool-down, and n=3 per case. The non-monotone 48 → 64 → 96 column is the tell.
`POC-FINDINGS.md` §6 has the diagnosis; §6a has the clean partial re-run
(only cells 24 and 32 completed) and acceptance criterion step 5 is written
against that reality. The **deterministic** columns (rounds, accepted/round,
block acceptance, output identity) are trustworthy — they are computed, not
timed, and were byte-stable across every repetition.

The `block=16` cell is the load-bearing negative: short blocks **combined with**
fuzzy re-anchoring are far *worse* than the stock baseline (69.6 vs 138.2 t/s),
because fuzzy firing more often at short block length multiplies verify passes.
Fuzzy retrieval is only safe **with** long blocks. A worker who ships fuzzy
matching on the stock ladder ships a regression.

### Pre-empting the obvious wrong fix

Four mechanisms were proposed for RAMP. Two were **measured and disconfirmed**;
build neither.

- **"Index the model's own generated output as an extra copy source"** (the
  Tier-3-requiring mechanism, and the one whose name RAMP is built on) —
  **measured -0.7%**, and the full combination `RAMP+wide` is *worse* than RAMP
  without it (+3.0% vs +31.7%). The engine's own comment at
  `generation.py:7536-7539` ("matches into the model's own generated text …
  tend to have weak continuation predictiveness") is **empirically correct on
  this workload**. Do not build it.
- **"Multi-candidate consensus ranking"** — **measured +0.0%**, byte-for-byte
  identical to baseline on every counter. With an exact 6-gram key the candidate
  set is effectively degenerate; there is nothing to rank. Do not build it.
- **"Adaptive block length from candidate agreement"** — **measured -52.4%**. It
  produces the best per-block acceptance of any variant (0.968) and the worst
  throughput, which is the roofline lesson in one row. Do not build it.
- **"Backward-match extension is a good confidence signal for block length"** —
  this is the ladder's premise and the data contradicts it: `B1-fuzzy-ladder`
  gains only +7.0% while a *fixed* length gains +42.7%. Do not build an adaptive
  policy keyed on `ext`.

If a worker is tempted to implement `wide` / `consensus` / `agreement-sizing`
because "RAMP means retrieval-augmented and this is the retrieval part":
**stop**. The retrieval augmentation that pays is the *fuzzy re-anchor inside
the prompt*, plus block length. That is the whole finding.

## Fix

1. **Create `rafale/draft/ramp.py`.** Two layers, strictly separated:

   a. **Pure-Python core, no engine import at module top level.** CLAUDE.md
      *Environment split* is binding: `mtplx` is macOS-only, so a module that
      imports it at import time is not importable in CI. The retrieval logic —
      the exact index, the short-anchor index, the mismatch-tolerant backward
      similarity score, candidate selection, and the block-length policy — must
      be plain Python over `list[int]` with **zero** engine dependency, so it is
      unit-testable off-Mac.

      The POC (`scripts/ramp_patch.py`) does **not** satisfy this — it imports
      `mtplx.context_copy` at module scope. That is acceptable in a throwaway
      probe and is not acceptable in `rafale/`. Restructuring it is part of this
      card, not a refactor to skip.

   b. **`install()` — the engine binding, importing `mtplx` lazily inside the
      function.** It subclasses the engine's real `NgramIndex` (so every exact
      hit keeps the engine's exact behaviour and only the miss path changes),
      wraps it around the pure core, and rebinds three module attributes:
      `NgramIndex`, `block_for_ext`, and — only when a fixed block length is
      configured — `context_copy_block_k`. Rebinding `context_copy_block_k` is
      required, not optional: `block_for_ext(ext, ccopy_k)` receives
      `ccopy_k = context_copy_block_k()` (`generation.py:7532`), and the stock
      value 24 would re-clamp a longer block. Confirmed by reading
      `context_copy.py:61-65` and `:98-100`.

   c. **Preserve the engine's `find()` contract exactly**: signature
      `find(self, history, *, max_pos=None) -> tuple[int | None, int]`, returning
      a position that indexes `prompt_ids` and honouring `max_pos` as an
      exclusive upper bound. A position `>= max_pos` yields an empty slice at
      `generation.py:7836` and silently disables drafting — no exception, no
      log. Return `(None, -1)` on miss.

   d. **Counters.** Count probes, exact hits, fuzzy hits, and misses, exposed
      via a module-level accessor. The engine's `context_copy_*` telemetry
      cannot distinguish an exact hit from a fuzzy one, and per-source
      accounting is what the plan's Phase 4D asks for ("Track acceptance per
      source").

2. **Create `tests/test_ramp.py`** — pure Python, no `mtplx` import, runs
   anywhere. Covers the Gherkin scenarios below.

3. **Create `tests/test_engine_seam.py`** — the seam guard decision 005's
   *Missing evidence* item 6 calls "the first Phase 1.5 follow-up task", and
   which its kill criterion names as the detection mechanism that does not yet
   exist. It must:
   - skip cleanly (`pytest.importorskip`) when `mtplx` is unavailable, so CI
     stays green off-Mac;
   - assert the function-local import line still exists in
     `mtplx/generation.py` by reading the source and matching
     `from .context_copy import` **inside** a `def`-indented block — the whole
     no-diff mechanism dies silently if that import is hoisted to module scope;
   - assert `NgramIndex`, `block_for_ext`, `context_copy_block_k` are still
     module attributes of `mtplx.context_copy` with the expected call
     signatures;
   - assert `_BLOCK_LADDER == (8, 12, 16, 24, 32)`, so a ladder change upstream
     is detected rather than silently re-baselining this card's numbers;
   - assert and **record** the observed `mtplx` / `mlx-lm` / `mlx` versions and
     fail on any drift from the pin in decision 005's Environment table
     (`mtplx` 2.7.1, `mlx-lm` 0.31.3, `mlx` 0.32.0).

4. **Create `scripts/launch_ramp_server.sh`** — decision 001's launch line
   verbatim, differing only by routing through the RAMP installer and by port.
   Any other difference from `scripts/launch_baseline_server.sh` is a bug: later
   phases compare against numbers produced by exactly that configuration.

5. **Default block length — the committed sweep does NOT determine it. Re-run it.**
   *(Amended 2026-08-26 after the clean sweep came back partial; see
   `POC-FINDINGS.md` §6a. The original step told the worker to read the value off
   `evidence/block-sweep-*.json`, which is unsatisfiable: the only clean cells are
   24 and 32, and the offline table's peak at 48–64 measures a different quantity
   than wall-clock.)*

   The worker must:

   a. Re-run `scripts/ramp_block_sweep_clean.sh 32 48 64 96` on a machine that has
      **not** just been through a long benchmarking session. Check
      `sysctl vm.swapusage` first and abort if used swap exceeds ~1 GB — the
      partial sweep died because ~8 sequential 27B loads pushed 5.75 GB of 7 GB
      into swap and collapsed throughput to 62.7 t/s on a configuration that had
      served at 179–220 t/s an hour before.
   b. Take the cell with the highest **median** wall-clock t/s (not mean — every
      cell can carry outliers). **Do not use median-vs-p95 agreement as the
      contamination test — it is structurally blind to swap-pressure
      contamination**, which lands in the *lower* tail while p95 only probes
      the upper tail (cell 24 in the partial clean sweep passed a 2%
      median/p95 test while 3 of its 8 runs — 37.5% — were swap-contaminated
      outliers). Instead require **`(max − min) / median ≤ 0.15`** across all
      `n` runs in the cell, or equivalently report the count of runs below
      `0.9 × median` and require it to be zero. A cell failing this is
      contaminated and must be re-run, not used.
   c. If two cells are within 5% of each other, choose the **shorter**: both the
      EMA-guard margin (second defect site) and the long-context risk worsen with
      length.
   d. Record the chosen value, the sweep row that justifies it, and the observed
      `vm.swapusage` in a comment.

   If the re-run cannot be completed, pinning **48** is acceptable **only** with
   an explicit note that it rests on the interleaved A/B (+53.9%, `evidence/ab-bench.json`)
   rather than on a block-length sweep — and the follow-up sweep becomes a
   blocking item on the long-context card. Do **not** silently substitute a
   number from the offline table.

6. Where genuinely ambiguous, implement the **minimal** version: this card's
   objective is a proposer that is provably at least as good as stock and never
   changes temperature-0 output. Anything cleverer is a later card with its own
   evidence.

7. **`install()` must default to OFF.** This is not optional prose — it is a
   Fix requirement. `install()` must take an explicit `enabled: bool` parameter
   (or equivalent), defaulting to `False`, and `scripts/launch_ramp_server.sh`
   must set it explicitly `True` to demonstrate the live path. Any environment
   variable equivalent (e.g. `RAMP_ENABLED`) must also default unset/off. This
   directly contradicts the POC's own defaults (`ramp_patch.py`'s `RAMP_BLOCK`
   and `RAMP_FUZZY` env vars default to on) — do not carry that default
   forward into `rafale/draft/ramp.py`. The reason is §"Largest open risk"
   below: every number in this card is short-context evidence, and shipping it
   on by default in a 128K+ configuration is exactly the regression that
   section warns against.

### Do NOT touch (scope fences)

- **Do NOT modify any file under the MTPLX site-packages tree.** The entire
  value of this card is that it needs no engine diff. A worker who edits
  `generation.py` or `context_copy.py` has failed the card even if the numbers
  improve.
- **Do NOT touch `rafale/draft/ngram.py` or `rafale/draft/fusion.py`**, and do
  not delete them. They are unwired scaffolds that duplicate engine
  functionality; their fate is a separate card. `tests/test_ngram.py`'s 6 tests
  depend on `ngram.py` and must keep passing unchanged.
- **Do NOT touch `scripts/launch_baseline_server.sh`.** Decision 001 pins it and
  forbids edits without a new decision record; every committed baseline number
  in `results/` was produced by it.
- **Do NOT change the engine's EMA-suspend control law** — that means an engine
  diff. This card *measures and records* the hazard (acceptance criterion #4);
  it does not fix it.
- **Do NOT implement `wide` corpus indexing, `consensus` ranking, or
  agreement-based block sizing.** Measured -0.7%, +0.0%, and -52.4%
  respectively. If tempted, note it in the report and do not do it.
- **Do NOT touch anything under `rafale/ane/`.** Gate 0.5 is a recorded NO-GO
  (decision 004); RAMP is CPU-retrieval plus GPU-verify only.
- **Do NOT add a `type: ignore` to make `mtplx` imports resolve.**
  `pyrightconfig.json` already resolves them via `extraPaths`.
- **Do NOT commit raw server logs or `.mlpackage`/compile-cache artifacts.**

## Touch List (only these files)

- `rafale/draft/ramp.py` (new)
- `tests/test_ramp.py` (new)
- `tests/test_engine_seam.py` (new)
- `scripts/launch_ramp_server.sh` (new)

Four files, all new; zero existing files modified. The count exceeds the
3-file guideline, and the justification is that the blast radius is *lower*
than a 3-file card that edits existing code: no existing behaviour can regress
because no existing file changes. If the dispatcher still wants a split, the
clean seam is (1)+(2) as one card and (3)+(4) as a second.

> Enforcement note: this list is intent. The actual boundary is the dedicated
> git worktree and the pre-merge `git diff --stat` / `git status --porcelain`
> review (RULES §3).

## Worktree provisioning (RULES §3.7 — mandatory)

Two resources are gitignored and **will be silently absent** in a fresh
worktree. Both must be provisioned by the operator BEFORE dispatch, not by the
worker:

1. **`.venv`** — the project interpreter, needed by `make lint` / `make test`.
   `ln -s ~/rafale-project-worktree/.venv <worktree>/.venv`
   **RULE 3.7.4 applies:** a symlinked venv's editable install resolves to
   whichever source was last `pip install -e`'d into it — almost always the main
   repo, NOT this worktree. The worker must run
   `PYTHONPATH="<worktree>" .venv/bin/python -m pytest ...` and must **never**
   run `pip install -e` / `uv pip install -e` against the shared `.venv`.

2. **The MTPLX interpreter** — not a worktree resource at all; it is a fixed
   absolute path (`/opt/homebrew/var/mtplx/venv-2.7.1/bin/python`) that every
   RAMP script already hardcodes and that exists machine-wide. Nothing to link.
   **But `tests/test_engine_seam.py` must not assume it**: the test runs under
   `.venv`, where `mtplx` is not importable, and must `pytest.importorskip`.

3. **The model checkpoint** (`~/.mtplx/models/...`) is machine-wide, ~30 GB, and
   is needed only by acceptance criterion #6 (the live check). Nothing to link.

**Before dispatch, the operator must run acceptance §5's command in a throwaway
worktree from this card's baseline commit.** If it fails for a missing-resource
reason, the provisioning above is wrong — fix the card, not the worktree.

### Blocker — RESOLVED before dispatch

A Strong Card first-fail lock was armed against this worktree, blocking every
`pytest` invocation. **Cleared 2026-08-26** per judge verdict
`VERDICT-SC-P4DA.md` §6: the lock's recorded `command` was a `sed` of
`RULES.md` (a documentation read, not a test run) that fell through to a
pytest-summary-text fallback and matched the string `"567 passed,"` inside
`RULES.md`'s own prose — no test was ever executed. Root cause and general
lesson (missing `sed`/`awk` in the guard's read-only allowlist) are recorded
in the verdict for the hooks maintainer. **Baseline suite count established:
85 passed** (`uv run pytest`, clean run after the lock was cleared) — see
acceptance criterion #5.

## Non-goals

- **Not building Tier 3 draft injection.** No engine source diff, no foreign
  tokens, no fused copy+MTP blocks, no tree verification (Phase 8B). Every
  measured win fits the no-diff envelope.
- **Not fixing the engine's EMA-suspend control law.** Measured and recorded
  only (see second defect site).
- **Not validating at 128K–256K context.** All POC traces are ~800-token
  prompts. See *Largest open risk*.
- **Not deciding the fate of `rafale/draft/ngram.py` / `fusion.py`.**
- **Not building an adaptive block-length controller.** A fixed length read off
  the live sweep, with an explicit follow-up card for adaptivity.
- **Not touching quant, KV, or prefill.** One variable at a time.

## Behavioral spec (Gherkin)

```gherkin
Scenario: the proposer reduces exactly to the engine's algorithm when disabled
  Given a RAMP proposer configured with fuzzy matching off and the ladder block policy
  And the three captured traces in docs/reviews/2026-08-26-ramp/evidence/traces.json
  When the decode-cycle replay is run for both it and the engine's own proposer
  Then every counter (rounds, drafted, accepted, accepted_blocks, probes, suspensions) is identical

Scenario: an exact n-gram hit is never altered by RAMP
  Given a token stream whose ng_min-gram suffix occurs exactly in the prompt
  When RAMP's find() is called
  Then it returns the same (position, extension) the engine's NgramIndex returns
  And the fuzzy fallback is not consulted

Scenario: the fuzzy fallback fires only where the exact key goes dark
  Given a token stream whose ng_min-gram suffix does NOT occur in the prompt
  And whose shorter anchor suffix does occur
  When RAMP's find() is called
  Then it returns a position inside the prompt with the highest backward similarity
  And the fuzzy-hit counter increments while the exact-hit counter does not

Scenario: the max_pos contract is honoured
  Given a candidate position at or beyond max_pos
  When RAMP's find() is called with that max_pos
  Then that candidate is never returned
  And the result is either an earlier valid position or (None, -1)

Scenario: a miss returns the engine's miss sentinel
  Given a token stream with neither an exact nor an anchor match in the prompt
  When RAMP's find() is called
  Then it returns (None, -1) so the engine falls through to a normal MTP round

Scenario: the configured block length survives the engine's cap
  Given RAMP is configured with a fixed block length longer than the stock ladder maximum
  When install() has run and the engine computes block_for_ext(ext, context_copy_block_k())
  Then the returned length equals the configured length for every ext in 0..4

Scenario: the seam guard fails when the function-local import is gone
  Given a copy of mtplx/generation.py whose context_copy import has been hoisted to module scope
  When the seam guard's import-locality check runs against it
  Then the check fails

Scenario: the seam guard fails on an engine version bump
  Given an installed mtplx version different from the pin in decision 005
  When the seam guard runs
  Then it fails and its message names both the observed and the pinned version

Scenario: RAMP is off by default
  Given install() is called with no explicit enabled argument
  Then the stock engine proposer remains in effect
  And no RAMP module attribute rebinding occurs

Scenario: temperature-0 output is unchanged on the live serving path
  Given the stock baseline server and a RAMP-patched server on the same checkpoint
  When the same coding-agent prompt is sent to both at temperature 0
  Then the two completions are byte-identical
```

## Acceptance criteria — FAIL-FIRST MANDATORY

1. **Write the failing test FIRST and capture its output before the fix exists.**
   This is a GREENFIELD card: the fail-first evidence is the `ImportError` /
   `ModuleNotFoundError` from `rafale.draft.ramp` not existing. That IS the gap —
   say so explicitly.
   ```
   1. Run the fail-first command with output redirected to the exact required filename:
      PYTHONPATH="$WT" .venv/bin/python -m pytest -q tests/test_ramp.py > SC-P4DA-failfirst-before-fix.txt 2>&1
   2. Immediately `cat SC-P4DA-failfirst-before-fix.txt` (or `wc -l`) to confirm the file
      exists on disk and is non-empty, and paste that confirmation in your report.
   3. Only THEN proceed to the fix. Do not treat step 1's on-screen output as sufficient —
      the operator will `ls` the file directly (RULE 5.6) and a missing file fails
      acceptance regardless of how correct the underlying fix is.
   ```
   **A "pre-existing / environment issue" claim requires literal pasted proof**
   (the actual command and its actual output), not a narrated diagnosis. An
   unproven claim is treated as false and investigated as a real regression
   (RULES §5.2).

2. **The control scenario passes**: with fuzzy off and the ladder policy, RAMP's
   replay counters are *identical* — not close — to **`V0-engine-exact`** (the
   replay's own reimplementation of the engine's policy) on all three committed
   traces. Assert equality on every counter. A near-miss here means the
   reimplementation drifted from `V0` and every other ablation number is void.
   Separately, and non-blocking, report the replay's fidelity against the
   **live engine's** `context_copy_*` telemetry as a one-sided bound: replay
   counters may exceed the live engine's by up to **7%** (all three traces show
   a systematic positive bias — the replay never under-counts). This is a
   weaker, informational check, not a pass/fail gate — do not conflate it with
   the `V0` equality assertion above.

3. **Every other Gherkin scenario has a 1:1 test**, named after the scenario,
   including the negative cases: `max_pos` exclusion, the `(None, -1)` miss
   sentinel, the hoisted-import seam failure, and the version-drift seam failure.
   The last two must be tested against a *synthetic* source string and a
   *monkeypatched* version respectively — do not mutate the installed engine.

4. **Measure and record the EMA-guard's actual firing behavior — not the
   aggregate acceptance ratio, which does not predict it.** POC evidence shows
   the guard fires routinely, including on the stock baseline (5 of 15 runs),
   and at block-acceptance ratios as high as 0.880 and 0.862 — well above any
   aggregate threshold that would look "safe". Aggregate block acceptance is
   the wrong statistic: the guard's EMA is a local excursion measure, not
   summarized by the run-level average. With the chosen block length:
   - Report the **suspension count per run** (RAMP arm vs stock baseline arm),
     not just the aggregate ratio.
   - Report the **minimum observed EMA excursion**, if obtainable from engine
     telemetry or instrumentation added for this purpose.
   - Compare RAMP's suspension *incidence* against the stock baseline's — the
     question is whether RAMP suspends **more often** than stock, not whether
     it suspends at all (stock already does).
   - Do **not** use a bare 0.45 aggregate-ratio threshold as a stop/go gate —
     it is not derived and the evidence shows it doesn't track the hazard.
   Record all of this in `docs/decisions/` as a known hazard with its measured
   values. Do **not** change the guard.

5. **Run the full suite twice in a row**:
   ```
   cd <worktree> && PYTHONPATH="<worktree>" timeout 240 .venv/bin/python -m pytest -q
   ```
   Both runs green, at or above the baseline count of **85 passed** (established
   2026-08-26 after the first-fail lock was judged spurious and cleared per
   `VERDICT-SC-P4DA.md` §6 — `uv run pytest` → `85 passed`). Investigate if
   materially different; do not report a lower number as "fine". **Do not add
   new `--ignore` / `-k` / skip flags.** Narrowing the suite to make it green
   is a card failure (RULES §5.3).

6. **Live confirmation on the serving path** (macOS only). Launch
   `scripts/launch_ramp_server.sh`, run `scripts/ramp_ab_bench.py` against it and
   the stock baseline, and confirm **both**:
   (a) median decode t/s is above the stock baseline arm, and
   (b) temperature-0 output is byte-identical between arms on all three cases.
   (b) is the prime-directive gate and is **binding**: a throughput win with any
   output difference does not merge, no matter how much faster it is.

7. `make lint` clean. Do not silence a type error with `type: ignore`;
   `pyrightconfig.json` resolves the engine imports via `extraPaths`.

8. Keep your own context small: redirect test output to files and `tail`/`grep`
   the summary back.

9. **`install()` defaults to off, verified by test** (Fix step 7 / Gherkin
   "RAMP is off by default"). A call to `install()` with no explicit
   `enabled=True` must leave the stock engine proposer in effect. This is a
   pass/fail criterion, not documentation.

## State explicitly in your final report

(a) the fail-first evidence — the test output from BEFORE the fix, and
    confirmation the file exists on disk;
(b) the control scenario's (#2) pass status, with the counter-equality output;
(c) each remaining Gherkin scenario's test name and pass status, negative cases
    named individually;
(d) the block length you chose, the exact sweep row you read it off, and whether
    the two-cells-within-5% tiebreak applied;
(e) the measured EMA-guard margin (#4) and where you recorded it;
(f) the two full-suite run results, with pass counts, verbatim from the summary
    line, and the baseline number you compared against;
(g) the live A/B result (#6): median t/s per arm and the per-case output-identity
    verdict;
(h) the complete list of files you created, modified, or deleted — including
    anything outside the Touch List, and why;
(i) anything you were tempted to change and did not — in particular whether you
    were tempted to add `wide`, `consensus`, agreement-sizing, or an engine diff.

## Largest open risk (state this to the breaker)

**Every POC number comes from ~800-token prompts. The plan targets
128K–256K.** This is not merely "unmeasured" — it is *estimable*, and the
estimate is not reassuring.

**Corrected mechanism.** The block is verified in a **single batched attention
call** (`generation.py:7845-7852`) — the KV cache is read **once per verify
pass, regardless of block length**. The cost that grows with block length `T`
is **attention compute (FLOPs ∝ T × C)**, not KV bandwidth per proposed row.
(An earlier draft of this analysis, and the GPT-5.6 consult it quoted, stated
this as a KV-bandwidth effect — that framing is wrong and matters: a
bandwidth story would imply Phase 6 KV quantization could rescue RAMP at long
context; the compute story says it would not.)

**Roofline break-even**, from committed project constants (decision 003: 64
KB/token KV; plan: ~300 GB/s, ~28 GB Q8 weights; `config.json`: 24 heads × 256
head_dim, 16 attention layers): at `T = 49` (block=48), attention compute is
**0.015 TFLOP at 800 tokens** and **5.05 TFLOP at 256K** — a 330× swing. Which
block length wins depends on one measurable scalar, **achieved attention
FLOPS at long context**: under ~30 TFLOPS achieved, the optimum **collapses
to the shortest block at both 128K and 256K** — i.e. to the stock ladder, this
card's own fourth kill-criterion bullet. Under ~60 TFLOPS it likely survives.
**This is not a footnote — it is the single scalar the long-context follow-up
card must measure first, before re-running any block sweep at length.**

The GPT-5.6 consult (`evidence/consult-gpt56.txt`) also makes a sharper
methodological point that the short-context ablation in this card glosses
over: *"there is no defensible universal crossover T from the supplied
numbers — optimize accepted tokens per verify **time**, not accepted tokens
per **pass**."* At 800 tokens `tok/pass ≈ tok/time` because pass time is
`T`-independent at that scale; that equivalence is exactly what breaks down
at 128K, where pass time grows with `T`. The long-context follow-up must
measure `tok/pass` **and** wall-clock verify time separately, not assume the
short-context proxy still holds.

The whole case for long blocks in this card rests on extra verify rows being
nearly free, which is a *short-context* observation. The `block=16` sweep
cell already shows this regime is not monotone even at short context — a
badly chosen length is *worse than doing nothing* (69.6 vs 138.2 t/s).

Two secondary risks:

- **CPU cost of the fuzzy probe at long context.** The short-anchor index over
  128K tokens has far more candidates per key, and the backward-similarity scan
  runs per decode cycle on the CPU. The POC caps candidates, but the cap was
  never exercised at scale.
- **The A/B ran two 27B servers resident simultaneously.** Interleaved A/B/A/B
  makes the *comparison* fair, but both arms may be depressed in absolute terms
  by memory pressure. The relative +53.9% is the defensible claim; the absolute
  t/s figures are not clean baseline numbers.

**Consequence for the card:** a long-context re-run of the block sweep is a
mandatory follow-up card before RAMP is enabled by default in any 128K
configuration. This card ships it behind an explicit, off-by-default flag with
the measured short-context evidence attached — not as a new default.

## Kill criterion

Abandon RAMP if **any** holds:

- The live A/B (#6) fails to beat the stock baseline arm on median decode t/s.
- Temperature-0 output differs between arms on any case (prime-directive gate).
- The seam guard cannot be made to detect a hoisted import or a version bump —
  the seam would then be defended by nothing and would break silently at the
  next dependency refresh (decision 005's own words).
- The long-context follow-up shows the optimum block length collapsing to the
  stock ladder, at which point the remaining win is the fuzzy re-anchor alone
  (+7.0% measured on the ladder), which does not justify carrying a live
  monkeypatch over engine internals.

## Failure protocol

If blocked, ambiguous, or the code does not match this card's `Defect` section:
STOP and say so in your report. `INVALID_CARD` is honorable. Do not improvise a
different fix, do not widen scope to make something pass, do not delete a
failing test, and do not patch engine source to make the numbers work.

## Environment

Every `file:line` reference to engine source is relative to
`/opt/homebrew/var/mtplx/venv-2.7.1/lib/python3.13/site-packages/`.

| Item | Value |
|---|---|
| Interpreter (engine) | `/opt/homebrew/var/mtplx/venv-2.7.1/bin/python` |
| Interpreter (project) | `.venv/bin/python` |
| `mtplx` | 2.7.1 |
| `mlx-lm` | 0.31.3 |
| `mlx` | 0.32.0 |
| Checkpoint | `~/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality` |
| Baseline server | `scripts/launch_baseline_server.sh` (decision 001), port 8001 |
| Card baseline commit | the commit that adds this card |

## Revision history

**v3 (2026-08-26, before dispatch) — five blocking edits from the breaker's
adversarial verdict** (`VERDICT-SC-P4DA.md`, ACCEPT-WITH-EDITS): (1) acceptance
criterion #2 reworded to name `V0-engine-exact` explicitly, since the original
wording asserted equality against the live engine, which fails by construction
(+7% systematic bias) — added a separate non-blocking fidelity bound instead;
(2) Fix step 5b's contamination test replaced — median-vs-p95 agreement is
structurally blind to swap-pressure contamination, which lands in the lower
tail; replaced with a max-min/median band test; (3) acceptance criterion #4
rewritten — aggregate block acceptance does not predict EMA-guard firing
(cells at 0.880/0.862 suspend; the guard already fires on stock baseline);
now requires suspension-count-per-run and RAMP-vs-stock incidence instead of
a bare aggregate-ratio threshold; (4) the off-by-default requirement, previously
only prose in "Largest open risk", is now a Fix step, a Gherkin scenario, and
an acceptance criterion; (5) "Largest open risk" corrected — the mechanism was
misstated as KV-bandwidth-per-row when it is attention-FLOPs-per-pass (KV is
read once per verify pass); added a roofline break-even computation showing the
outcome hinges on one measurable scalar (achieved attention FLOPS at long
context), half of whose plausible range trips this card's own kill criterion.
Also: the first-fail pytest lock was judged spurious and cleared, and the
baseline suite count (85 passed) is now filled into acceptance criterion #5.

**v2 (2026-08-26, before dispatch) — Fix step 5, which was unsatisfiable.**
v1 instructed the worker to read the default block length off
`evidence/block-sweep-*.json`. After v1 was written, the clean re-run of that
sweep completed only two of five cells (24 and 32) before machine memory
pressure — 5.75 GB of 7 GB in swap after ~8 sequential 27B model loads —
collapsed throughput and made the remaining cells untrustworthy. The run was
stopped rather than finished with bad data.

That left v1's step 5 pointing at evidence that does not determine the answer:
the only clean cells bracket the offline peak rather than locating it. Step 5 now
tells the worker to re-run the sweep on a machine checked for swap pressure
first, states the median/p95 agreement test a cell must pass to be used, and
gives an explicit fallback (pin 48 on the interleaved A/B evidence, with the
sweep escalated to a blocking item on the long-context card) rather than leaving
the worker to improvise.

The contaminated first sweep's wall-clock figures are also now labelled as such
inside the POC evidence table, so a reader cannot lift 96 → 248.4 t/s out of it
as if it were a result.

**Nothing else changed.** The defect analysis, the three disconfirmed mechanisms,
the no-diff seam finding, the +53.9% A/B, and the temperature-0 identity gate are
all v1 material and all still stand — none of them depend on the block sweep.
