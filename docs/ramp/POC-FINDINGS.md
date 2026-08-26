# RAMP POC — what was built, what it proved, what it killed

**Date:** 2026-08-26 · **Author:** Opus card writer · **Card:** `SC-P4DA-ramp-serving-path-proposer.md`

RAMP was proposed as "Retrieval-Augmented Multi-token Prediction": CPU-side
speculative drafting that goes beyond MTPLX's exact-n-gram, prompt-only
prompt-lookup drafter. This POC built it, ran it against the real engine, and
measured every proposed mechanism separately.

**Headline: the premise half-holds, and the half that holds is not the half the
name is about.** The retrieval-augmentation that pays is a mismatch-tolerant
re-anchor *inside the prompt* plus a much longer proposal block. The mechanisms
that motivated the "retrieval-augmented" framing — indexing generated text,
multi-candidate consensus — are measured at **-0.7%** and **+0.0%**. A third
proposed mechanism, agreement-driven adaptive block sizing, is measured at
**-52.4%**.

Nothing here is estimated. Every number is from a run whose output is committed
under `evidence/`.

---

## 1. What was built

| File | What it is |
|---|---|
| `scripts/ramp_capture_traces.py` | Captures real `(prompt_ids, output_ids, engine telemetry)` traces from the live server on coding-agent file-edit prompts built from real repo files. |
| `scripts/ramp_replay.py` | Offline decode-cycle replay. Imports the **real** `mtplx.context_copy` (not a transcription) so the baseline arm cannot drift from the engine. Runs the ablation. |
| `scripts/ramp_patch.py` | The RAMP proposer, injected into a live server by rebinding `mtplx.context_copy` module attributes. **Zero engine source lines changed.** |
| `scripts/ramp_launch_patched_server.py` | Decision 001's launch line verbatim, plus the patch, on port 8002. |
| `scripts/ramp_ab_bench.py` | Interleaved A/B/A/B live benchmark, baseline :8001 vs RAMP :8002, with a temperature-0 output-identity gate. |
| `scripts/ramp_block_sweep*.{sh,py}` | Live block-length sweeps (see §6 — the first one is invalid and kept as evidence of why). |

### The method that makes offline measurement legitimate

At temperature 0 the engine accepts a copy block by argmax match:

```python
# mtplx/generation.py:7870-7876
_cc_nacc = 0
for _cc_d, _cc_t in zip(_cc_block, _cc_g):
    if _cc_d == _cc_t:
        _cc_nacc += 1
    else:
        break
```

So acceptance is exactly *longest common prefix of the proposal with the token
stream the target actually produced*. Given a real captured trace, that is
computable on the CPU with no model in the loop — which is why a full ablation
was affordable, and why the offline numbers are not a simulation of acceptance
but a calculation of it.

---

## 2. Fidelity: the replay reproduces the live engine

Simulated counters vs. the engine's own `context_copy_*` telemetry for the same
three requests, at `mtp_advance=3` (what depth-2 MTP implies):

| trace | source | rounds | drafted | accepted | blocks | susp |
|---|---|---|---|---|---|---|
| rename-identifier | engine | 30 | 652 | 580 | 30 | 0 |
| | replay | 32 | 674 | 602 | 31 | 0 |
| add-method | engine | 31 | 688 | 648 | 28 | 1 |
| | replay | 32 | 708 | 675 | 29 | 1 |
| docstring-edit | engine | 22 | 512 | 475 | 22 | 0 |
| | replay | 23 | 519 | 495 | 23 | 0 |

Within a few percent on every counter, including the suspension event. This also
validates the completion re-tokenization in the capture script — if the token
stream were wrong, the counters would not line up.

**Residual uncertainty, stated:** how many tokens a fallback MTP round commits is
not observable offline, so it is a parameter. Every variant is replayed under the
same value and the sweep reports both ends of the plausible range (`mtp_advance`
1 and 3). The comparison between variants is therefore apples-to-apples even
where the absolute number carries that assumption.

---

## 3. The ablation — one variable at a time

Accepted tokens per verify pass, aggregated over the three traces,
`mtp_advance=3`. Full data: `evidence/replay-results.json`.

| variant | tok/pass | vs baseline | blk acc | dark% | verdict |
|---|---|---|---|---|---|
| V0-engine-exact | 14.044 | +0.0% | 0.932 | 34.3% | reference |
| **A1-control-ladder** | **14.044** | **+0.0%** | 0.932 | 34.3% | **control — RAMP reduces exactly to the engine** |
| A3-block-fixed32 | 16.586 | +18.1% | 0.866 | 40.5% | |
| A4-block-fixed48 | 20.042 | +42.7% | 0.856 | 49.0% | |
| A5-block-fixed64 | 22.115 | +57.5% | 0.815 | 54.0% | |
| A6-block-fixed96 | 20.253 | +44.2% | 0.750 | 68.4% | falls off |
| B1-fuzzy-ladder | 15.031 | +7.0% | 0.934 | 25.8% | fuzzy alone: small |
| **B3-fuzzy-fixed48** | **22.905** | **+63.1%** | 0.824 | 39.3% | **best** |
| B4-fuzzy-fixed64 | 21.143 | +50.5% | 0.777 | 54.9% | |
| C1-wide-ladder | 13.942 | **-0.7%** | 0.920 | 34.1% | **KILLED** |
| D1-consensus-rank | 14.044 | **+0.0%** | 0.932 | 34.3% | **KILLED** |
| E1-consensus-block | 6.681 | **-52.4%** | 0.968 | 16.3% | **KILLED** |

`A1` is the control that matters: with every RAMP knob off, the reimplemented
proposer produces byte-identical counters to the engine's own. Without that row,
none of the others mean anything.

### What died, and why it deserved to

**"Index the model's own generated output" (-0.7%).** This was the mechanism the
name "retrieval-augmented" pointed at, and the only one that would have needed
decision 005's Tier 3 engine patch. The engine's own source comment turns out to
be empirically right on this workload:

```python
# mtplx/generation.py:7536-7539
# Prompt-lookup semantics: the index covers the PROMPT only. Matches into
# the model's own generated text (self-repetition) tend to have weak
# continuation predictiveness and can cost more to verify than they commit,
# while grounded re-emission matches into the prompt (see the PR benchmarks).
```

Combined with the other mechanisms it is actively harmful: `RAMP+wide` scores
+3.0% where `RAMP` without it scores +31.7%. **Do not build it.**

**"Multi-candidate consensus ranking" (+0.0%).** Identical to baseline on every
counter, not merely close. With an exact 6-gram key the candidate set is
effectively degenerate — there is nothing to rank. **Do not build it.**

**"Adaptive block length from candidate agreement" (-52.4%).** This variant
achieves the *best* per-block acceptance of anything measured (0.968) and the
worst throughput. It is the roofline lesson in a single row: on a bus-bound
machine, optimizing acceptance *rate* while shrinking blocks trades the metric
that pays for the metric that flatters. **Do not build it.**

**Corollary — the engine's confidence ladder is keyed on a weak signal.**
`B1-fuzzy-ladder` (better retrieval, ladder block lengths) gains +7.0%.
`A4-block-fixed48` (stock retrieval, fixed length) gains +42.7%. Backward-match
extension is not a useful predictor of how long a block should be.

---

## 4. The seam: decision 005 mis-tiers this

Decision 005 puts external draft injection at **Tier 3 — needs a real engine
patch**. That is right for arbitrary tokens and wrong for what RAMP actually
needs, because `generate_mtpk` imports its drafter symbols **function-locally,
on every call**:

```python
# mtplx/generation.py:7479-7482  (inside generate_mtpk)
    from .context_copy import (NgramIndex, block_for_ext, context_copy_block_k,
                               ...)
```

Rebinding those attributes before the server starts replaces the proposer with
**zero engine source lines changed**. This was not inferred from reading — a
fully patched server ran 15 real requests on :8002.

The envelope: *which contiguous prompt span, and how long*. Everything that
measured a win fits inside it. **RAMP as measured needs no Tier 3 patch.**

One implementation trap, found by execution: `block_for_ext(ext, ccopy_k)`
receives `ccopy_k = context_copy_block_k()`, and `block_for_ext` returns
`min(_BLOCK_LADDER[idx], max(4, k_cap))`. Patching `block_for_ext` alone is not
enough — the stock cap of 24 silently re-clamps a longer block.
`context_copy_block_k` must be rebound too.

---

## 5. On metal: +53.9%, output byte-identical

`scripts/ramp_ab_bench.py`, interleaved A/B/A/B, 5 reps × 3 cases,
temperature 0, `RAMP_BLOCK=48`, `RAMP_FUZZY=1`
(`evidence/ab-bench.json`):

| arm | n | median t/s | p95 t/s | accepted/round | block acc |
|---|---|---|---|---|---|
| baseline (stock) | 15 | 138.2 | 143.9 | 20.52 | 0.920 |
| RAMP (patched) | 15 | **212.6** | 228.5 | **38.70** | 0.806 |

**+53.9% median wall-clock decode throughput.**

**Prime-directive gate: GREEN.** Temperature-0 completions are byte-identical
(sha256) between arms on all three cases. Structurally expected — the copy block
is a point-mass proposal verified by argmax, so the draft *source* cannot move
the output law — and confirmed empirically rather than assumed.

---

## 6. The first block-length sweep is INVALID — kept as evidence

`evidence/block-sweep.txt` reports a non-monotone curve: 16 → 69.6 t/s, 32 →
150.2, 48 → 184.3, **64 → 76.1**, 96 → 248.4. A dip at 64 sandwiched between 184
and 248 is not a block-length effect.

The diagnosis is in the data. Within every cell the **deterministic** counters
are perfectly stable across all repetitions — e.g. `block=48, add-method` gives
`rounds=17, drafted=816, accepted=669` on all three runs — while wall-clock t/s
on that byte-identical work reads **179.5 / 87.9 / 220.0**, a 2.5× spread.

Causes, all hygiene violations of CLAUDE.md rule 4 / plan Phase 0:

1. The stock baseline server stayed resident on :8001 for the whole sweep, so a
   second 27B process was competing for bandwidth and memory.
2. Cells ran back-to-back with no thermal cool-down.
3. n=3 per case, below the ≥5 the protocol requires.

This is committed rather than deleted because it is the concrete demonstration
of why the protocol exists, and because a reader who finds only the clean sweep
should know a contaminated one was run first.

**The A/B in §5 is not affected**: it interleaves A/B/A/B, so drift and
contention hit both arms equally. The relative +53.9% stands; the absolute t/s
figures in it are not clean baseline numbers.

`scripts/ramp_block_sweep_clean.sh` re-runs the sweep with exactly one engine
process alive at a time, warm-up runs discarded, a cool-down between runs, and a
reported per-cell spread so the reader can judge each cell's trustworthiness.

### 6a. The clean re-run is PARTIAL — two cells, and why the rest are missing

`evidence/block-sweep-clean.txt`, case `add-method`, n=8 measured runs per cell
after 2 discarded warm-ups:

| RAMP_BLOCK | n | median t/s | p95 | min | max | rounds | acc/round | blk acc | susp | output |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 8 | 149.4 | 149.8 | 48.0 | 150.2 | 31 | 21.13 | 0.880 | 1 | identical |
| 32 | 8 | **181.3** | 181.8 | 98.4 | 182.4 | 24 | 27.58 | 0.862 | 1 | identical |
| 48 | — | **not measured** | | | | | | | | |
| 64 | — | **not measured** | | | | | | | | |
| 96 | — | **not measured** | | | | | | | | |
| stock baseline | — | **not measured** | | | | | | | | |

The two cells that did complete are trustworthy in a way the first sweep's were
not: median and p95 agree to within 0.5 t/s, so the top of each distribution is
tight and the reported median is real. Each cell still carries one slow outlier
(the `min` column), which is why median, not mean, is the statistic used.

**Why the sweep was stopped rather than finished.** During the `block=48` cell
throughput collapsed to 62.7 t/s on a request that the *same configuration* had
served at 179–220 t/s an hour earlier, and the cell stalled with one generation
completed. No engine error, no thermal warning — the cause was memory pressure
accumulated across roughly eight sequential 27B model loads over the session:

```
$ sysctl vm.swapusage
vm.swapusage: total = 7168.00M  used = 5752.25M  free = 1415.75M
$ pmset -g therm
Note: No thermal warning level has been recorded
```

5.75 GB of 7 GB swap in use. Continuing would have produced exactly the class of
contaminated numbers that invalidated the first sweep, so the run was stopped and
the incomplete cells are reported as missing rather than filled with bad data.

**Consequences, stated plainly:**

- The two clean cells confirm the *direction* — 24 → 32 gains +21.4% wall-clock
  (149.4 → 181.3), with accepted-tokens-per-round rising 21.13 → 27.58 and
  per-block acceptance falling 0.880 → 0.862, exactly the trade the offline
  ablation predicts.
- They do **not** locate the optimum. The offline table puts the peak between 48
  and 64, and the live A/B at 48 measured +53.9% — but no *clean* wall-clock
  comparison of 48 against 64 and 96 exists.
- **The card's step 5 therefore cannot be satisfied from this evidence.** Card
  `SC-P4DA` instructs the worker to read the default block length off the live
  sweep; with 48/64/96 missing, that instruction is unsatisfiable as written. The
  worker must re-run `scripts/ramp_block_sweep_clean.sh` on a machine that has
  not just been through a long benchmarking session, or the card must be amended
  to pin 48 on the A/B evidence alone. **This is a known gap in the card, flagged
  here rather than papered over.**
- **A new hygiene requirement falls out of this:** the plan's Phase 0 protocol
  covers thermal cool-down but says nothing about memory-pressure accumulation
  across repeated model loads. On this machine that is a real and unmonitored
  contamination source. Any benchmark script that reloads the model per cell must
  record `vm.swapusage` per cell and refuse to report a cell measured under swap
  pressure.

One timing-independent result *is* complete and survives all of this: output
identity. `sha_unique=1` in every cell measured, at every block length tried
across both sweeps (16, 24, 32, 48, 64, 96) — the temperature-0 completion never
changed.

---

## 7. Open risks the breaker should attack

1. **Every number is from ~800-token prompts; the plan targets 128K–256K.** The
   entire case for long blocks is that extra verify rows are nearly free — a
   short-context observation. The GPT-5.6 consult
   (`evidence/consult-gpt56.txt`) names the mechanism: at 128K, verification
   attention reads a large KV cache for every proposed row, so KV bandwidth may
   dominate even when weight reads stay amortized. The 800-token optimum is not
   portable. **This is the single largest risk in the POC.**
2. **The EMA guard's control variable now opposes the objective.** The engine
   suspends drafting when per-block acceptance EMA drops below 0.35; longer
   blocks lower that ratio (0.920 → 0.806 measured) while raising absolute
   accepted tokens (20.52 → 38.70). At 0.806 the guard does not fire, so this is
   latent, not live — but the margin shrinks as block length grows.
3. **Fuzzy retrieval is only safe with long blocks.** The `block=16` cell scored
   69.6 t/s against a 138.2 baseline: fuzzy firing more often at short block
   length multiplies verify passes and is *worse than doing nothing*. Shipping
   fuzzy matching on the stock ladder would ship a regression.
4. **CPU cost of the fuzzy probe was never exercised at scale.** The short-anchor
   index over 128K tokens has far more candidates per key, and the
   backward-similarity scan runs per decode cycle. The candidate cap exists but
   was never stressed.
5. **`mtp_advance` is an offline parameter, not a measurement** (see §2).

---

## 8. Honest accounting of what this POC did not do

- Did not test above ~800 tokens of context.
- Did not measure the CPU time of the proposer itself, only end-to-end t/s.
- Did not run the full quality suite — only the temperature-0 byte-identity gate,
  which is the correct gate for a draft-source change but is not the full suite.
- Did not test with tools, streaming, or concurrent requests.
- Ran three cases from one repository; the workload is real but narrow.
- The GPT-5.6 consult stalled twice in stdin mode before succeeding on the third
  attempt; the successful response is committed verbatim at
  `evidence/consult-gpt56.txt` with its prompt at `evidence/consult-prompt.md`.
