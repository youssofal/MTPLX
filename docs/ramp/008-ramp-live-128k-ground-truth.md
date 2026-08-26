# 008 — RAMP survives at 128K on live measurement: +45.9 %, byte-identical, zero suspensions

**Date:** 2026-08-26
**Phase:** 4D (RAMP)
**Gate:** SC-P4DA kill criterion, bullet 4 — *"the long-context optimum collapses
to the stock ladder"*
**Outcome:** **NOT TRIPPED, on direct measurement.** At 128K, RAMP at its shipped
`block=48` beats the stock ladder by **+45.9 %** on median decode t/s (+44.2 %
cold, +46.2 % warm), temperature-0 output is **byte-identical**, and the EMA guard
**never suspends**. **Default remains off** — see *What this does not license*.

**Supersedes** the projected-evidence basis of
`007-ramp-long-context-block-length.md` item 1 and discharges its hedged outcome
line. Closes the modelling thread opened by
`docs/reviews/2026-08-26-ramp/VERDICT-SC-P4DA.md` §1.1.

**Evidence:**
- `results/ramp_128k_ab_block48.json` — the clean round, the result of record
- `results/ramp_128k_ab_block48_round0_invalidated.json` — the contaminated round, committed deliberately
- `docs/reviews/2026-08-26-ramp-longcontext/evidence/live-128k-ab-rows.jsonl` —
  all 21 per-request rows (`results/raw/` is gitignored by convention, so the
  rows live with this thread's other evidence; the summaries above embed them too)
- `docs/reviews/2026-08-26-ramp-longcontext/evidence/live-32k-validation-rows.jsonl`
- `docs/reviews/2026-08-26-ramp-longcontext/evidence/live-128k-corpus-manifest.json` —
  the 29 files and their token counts, so the prompt is rebuildable (the 500 KB
  body itself is not committed; `scripts/ramp_longcontext_corpus.py` regenerates
  it byte-for-byte from commit `b04b6e3`)

**Harness:** `scripts/ramp_longcontext_corpus.py`, `scripts/ramp_longcontext_ab.py`,
`scripts/ramp_longcontext_ab.sh`, `scripts/ramp_longcontext_summarize.py`.

---

## Why this record exists

`VERDICT-SC-P4DC.md` §1.6 rejected a fourth round of modelling and ordered the
measurement instead:

> Three rounds of modelling have each found and corrected a real error in the
> previous round's model, and this round's model is wrong again. That is not a run
> of bad luck; it is what an unanchored modelling chain looks like.

Every round agreed the decisive quantity was **block acceptance at long context**,
and every round declared it unmeasured. It is now measured.

## What was run

A real 128K coding-agent prompt built from **29 tracked files of this repository**
(`scripts/ramp_longcontext_corpus.py`) — not the synthetic repeating padding
`rafale/bench/prompts.py` uses, because RAMP is a prompt-lookup proposer and its
acceptance is a function of the corpus's real repetition structure. The task is
the one the ~800-token A/B already ran (`evidence/traces.json`,
`rename-identifier`): reproduce `rafale/draft/ngram.py`, which is present in the
context, with a rename applied. **Only the context length changed** (CLAUDE.md
rule 3), which is what makes the short-vs-long comparison below legitimate.

- Corpus: **127 972 tokens**, `sha256 ab823825a673d299…`, built at commit `b04b6e3`.
- Served prompt: **127 984 tokens** after the chat template.
- Both arms: `temperature = 0`, `max_tokens = 2048`, decision 001's canonical
  launch line; the RAMP arm differs only by `rafale.draft.ramp.install()` with
  `block=48, fuzzy=True` and `--port`.
- **21 requests total**, two alternation rounds, 6 per arm per round (round 0
  baseline has 3 — see *Hygiene*).

**Alternation is at the server level, not the request level.** A single 128K
server peaks at 98.4 GB (`results/baseline_128k_ane-off_mtp2.json`) on a 128 GB
machine, so the two arms cannot be co-resident the way `scripts/ramp_ab_bench.py`
runs them at ~800 tokens. Arms alternate per server launch instead. Each launch
serves one cold request (fresh 128K prefill) then warm ones
(`cached_tokens = 127 983`).

## The numbers — round 1, the clean round

| arm | median t/s | p95 | warm rel. spread | ccopy rounds | drafted | accepted | block acceptance | accepted/round | suspensions |
|---|---|---|---|---|---|---|---|---|---|
| stock ladder | **57.40** | 57.81 | 3.4 % | 30 | 652 | 580 | **0.890** | **19.33** | **0** |
| RAMP block=48 | **83.72** | 84.39 | 1.7 % | 19 | 912 | 618 | **0.678** | **32.53** | **0** |

| comparison | RAMP vs stock |
|---|---|
| cold (n = 1 per arm: 80.79 vs 56.04) | **+44.2 %** |
| warm (n = 5 per arm: 84.14 vs 57.55) | **+46.2 %** |
| **pooled median (n = 6 per arm)** | **+45.9 %** |

Temperature-0 output identity: **`sha256 b5e79a0198e8d394…` in every one of the 21
requests, both arms, both rounds.**

## The controlled comparison — same task, only context changes

The ~800-token A/B ran this exact case five times per arm
(`docs/reviews/2026-08-26-ramp/evidence/ab-bench.json`). Against it:

| quantity | 800 tokens | 128 K | change |
|---|---|---|---|
| stock decode t/s (median) | 115.93 | 57.40 | −50.5 % |
| RAMP decode t/s (median) | 175.44 | 83.72 | −52.3 % |
| **RAMP advantage** | **+51.3 %** | **+45.9 %** | −5.4 pts |
| stock block acceptance | 0.8896 | 0.8896 | **identical** |
| stock accepted/round | 19.33 | 19.33 | **identical** |
| RAMP block acceptance | 0.7164 | 0.6776 | −3.9 pts |
| RAMP accepted/round | 34.39 | 32.53 | **−5.4 %** |
| suspensions | 0 | 0 | unchanged |

Two facts in that table do the work:

1. **The stock ladder's acceptance telemetry is byte-identical across a 160×
   context range** — 30 rounds, 652 drafted, 580 accepted, at 800 tokens, at 32K
   and at 128K. Not a coincidence: at temperature 0 the emitted token sequence is
   identical and the copy source is identical, so the proposer's decisions are
   identical. Context length does not enter the retrieval decision at all. This
   reproduced on **every one of the 9 baseline requests** and every 32K validation
   request. The same holds for RAMP within a context (19 / 912 / 618 in all 12
   RAMP requests).
2. **RAMP's acceptance degrades by 5.4 % in tokens-per-pass while the stock
   ladder's is exactly fixed.** Decision 007's stated reversal condition is
   *"acceptance degrading at long context by more than ~30 % of block-48's
   tokens-per-pass, with the stock ladder's acceptance held fixed."* Measured
   differential degradation: **−5.4 %**, roughly a sixth of the margin — and the
   stock ladder's side is not merely *assumed* fixed, it is *measured* fixed.

**Verdict on the four modelling rounds' qualitative conclusion: substantially
confirmed, with one correction.** They concluded RAMP still wins at long context,
*possibly more so*. It still wins, decisively and with a large margin — but the
advantage **mildly shrinks** with context (+51.3 % → +45.9 %) rather than growing.
The "grows at long context" reading is **not** supported; the "does not collapse"
reading is confirmed. The gate turns on the second, so the gate is decided.

## The prime-directive gate

**PASS.** RAMP changes the execution path and not one byte of temperature-0
output, across the 21 requests measured here at 128K. The same `sha256` also
appears in the 2 32K validation requests and in all 10 `rename-identifier` rows of
the prior ~800-token A/B (`evidence/ab-bench.json`), both arms. This is the
engine's verifier doing its job: RAMP only *proposes*; the Qwen target *verifies*
(CLAUDE.md rule 7). No unverified speculative token reached output.

## Acceptance criterion #4's open question, answered

Open since SC-P4DA, unanswerable by any microbenchmark:

- **Block acceptance at 128K, block=48: 0.678** (618 accepted of 912 drafted),
  against the stock ladder's 0.890. RAMP's *per-block* acceptance is lower, as the
  POC predicted, and that is not the operative quantity.
- **Accepted tokens per context-copy round: 32.53 vs 19.33.** RAMP lands 6.6 %
  more total accepted tokens per response (618 vs 580) while spending **37 %
  fewer verify passes** (19 rounds vs 30). That trade is the whole mechanism, and
  it is what buys the +45.9 %.
- **EMA-guard suspensions at 128K: 0**, in both arms, in all 21 requests, in both
  rounds, and 0 in the 32K validation too. The suspend control law that decision
  006 and SC-P4DA flagged as a long-context hazard **never fires on this
  workload.** The hazard remains real in the code and unfixed, but no
  long-context claim in this project should continue to be hedged on it without
  new evidence.

## Decision

1. **Kill criterion bullet 4 is NOT TRIPPED, on measurement.** Decision 007's
   item 1 conclusion stands and its *"this is a projection, not a measurement"*
   caveat is discharged. The long-context optimum does not collapse to the stock
   ladder.
2. **The RAMP long-context modelling thread is closed.** Rounds 1–4 reached the
   right qualitative answer through four successively-corrected wrong mechanisms.
   The measurement supersedes all of them. **Nothing further should be inferred
   from `ramp_longcontext_model.py`, `ramp_kernel_regimes.py`, or the derived
   `6.13×` / `504 GB/s` scalars.** Only one thing was validated here: the ranking
   of `block=48` against the ladder at 128K. The projection's magnitudes were
   never validated, its 16-cell block sweep is still unmeasured, and its
   prediction of a *growing* advantage is contradicted above.
3. **RAMP's default stays off.** The gate cleared here is the *long-context
   collapse* gate, not the *flip the default* gate.
4. **Decision 007 item 4's mechanism and its bandwidth hand-off remain
   uncorrected and must not be cited.** `VERDICT-SC-P4DC.md` §§1.1–1.3 refuted
   them; this record makes them **moot for the gate, not correct.** That
   correction is still owed.

## What this does not license

- **One task, one prompt.** The ~800-token A/B ran three cases; this ran the one
  that transfers cleanly. A single coding-agent shape is not the workload.
- **One cold sample per arm per round.** A cold 128K prefill costs ~6–9 minutes of
  engine time inside a first-request wall that reached ~75–95 min on a
  freshly-compiled server (see *Hygiene*). The cold figure (+44.2 %) is an n = 1
  pair; the warm figure (+46.2 %, n = 5 per arm, spreads 1.7 % / 3.4 %) is the
  statistically stronger one, and they agree within 2 points.
- **The agent-harness shape is untested.** The acceptance invariance observed here
  follows from temperature-0 determinism on a *fixed* prompt. An append-only
  growing context (CLAUDE.md rule 6) is the real target workload and was not
  measured.
- **No power or bandwidth counter data.** See below.
- **Nothing about 256K**, and nothing about block lengths other than 48.

## Powermetrics — NOT obtained

`VERDICT-SC-P4DC.md` §1.6 item 1 and `FAILRUN-SC-P4DB.md` §6 item 3 both call for
a `powermetrics` counter run as the other half of the decisive measurement. **It
was not obtained.** `powermetrics` requires root and this session has no
passwordless `sudo` (`sudo -n true` → *a password is required*); it cannot run
non-interactively. `scripts/powermetrics_capture.sh` already exists and is the
right tool — it needs an operator at the keyboard:

```bash
sudo scripts/powermetrics_capture.sh ramp-128k-verify 500
```

Partial substitute captured instead: `ioreg -r -c IOAccelerator` GPU device
utilisation, which read **96–98 %** throughout both arms' prefill. That confirms
the GPU was saturated and no run was stalled. **It is not a bandwidth or power
counter and settles none of the §1.1/§1.2 questions**, which remain open along
with decision 007's *Handed to the operator* section.

## Hygiene

- **High Power Mode confirmed** (`pmset -g` → `powermode 2`), AC power, no thermal
  or performance warnings (`pmset -g therm`).
- **Warm and cold never pooled.** Warm decode is measurably *different* from cold
  on this engine — slower in the contaminated round (31.5 vs 43.1 at 128K; 66.7 vs
  94.8 at 32K), faster in the clean one (57.6 vs 56.0) — so they are not
  interchangeable samples in either direction.
- **Round 0 is invalidated and committed anyway.** Decision 007's hygiene note
  complains that two invalidated runs were described but never committed, so the
  diagnosis "cannot be checked by anyone." This record does not repeat that:
  `results/ramp_128k_ab_block48_round0_invalidated.json` is committed.

  | round | arm | cold t/s | warm median | **warm rel. spread** |
  |---|---|---|---|---|
  | 0 | baseline | 43.10 | 34.72 | **18.7 %** |
  | 0 | ramp | 73.86 | 51.24 | **85.4 %** |
  | 1 | baseline | 56.04 | 57.55 | **3.4 %** |
  | 1 | ramp | 80.79 | 84.14 | **1.7 %** |

  Round 0's absolutes are unusable. **Its ratio, however, survived:** +47.6 % warm
  in round 0 against +46.2 % in round 1. The A/B design did its job — whatever
  depressed round 0 hit both arms nearly proportionally, which is exactly what
  per-round alternation is for. The result of record is still round 1, on spread
  grounds alone.
- **Swap: occupied but inert — and this is the useful hygiene finding.**
  `vm.swapusage` read **7.98–8.53 GB of 9.22 GB (~90 %) throughout**, which is the
  signature that contaminated the earlier block sweep. Checked properly rather
  than by occupancy: over a 60 s sample during the RAMP arm, **`Pageouts` held at
  33 809 and `Swapouts` held at 1 201 172 — zero change** — with flat occupancy.
  The swap was stale residue from earlier work, not live thrashing. Both arms were
  measured at the same occupancy with the same zero paging rate.

  > **Recommended for the harness generally: swap *occupancy* is not the
  > contamination test; `Swapouts`/`Pageouts` *rate* is.** An occupancy check
  > alone would have wrongly aborted this entire measurement; a rate check alone
  > would still have caught the sweep that actually died. Note also that occupancy
  > did **not** distinguish round 0 from round 1 — both ran at ~90 % — so whatever
  > contaminated round 0, swap was not it.
- **Memory preflight per request**, recorded in every row: reclaimable pages
  **plus the server's own RSS**, against the measured 98.4 GB peak. An earlier
  version of this guard compared free-memory-*after*-launch against a peak that
  *includes* the 31 GB of resident weights, and aborted a run the machine could
  comfortably afford. Double-counting the weights is the trap.
- **`make lint && make test` clean:** `ruff check` clean, `ruff format --check`
  clean, `102 passed, 1 skipped` — the baseline count `VERDICT-SC-P4DC.md`
  establishes.

### The first-request anomaly, recorded because it is not explained

On a **freshly-launched** server the first 128K request took **~75–95 minutes of
wall clock** while the engine's own timer reported only **~520 s of prefill** for
that same request. On the *second* launch of each arm the same first request took
**363–380 s of wall**, matching the engine timer. Warm requests took 8–23 s
throughout. The gap sits outside the engine's instrumentation, with the GPU at
96–98 % and zero paging.

The most plausible explanation is one-time Metal kernel compilation / graph
specialisation for the 128K shapes, warmed on disk by the first launch — which is
also the most likely cause of round 0's contamination. **This was not verified and
is a hypothesis.** It matters twice: it is why round 0 is invalidated, and it
means `results/baseline_128k_ane-off_mtp2.json`'s `ttft_s` median of 390 s does
not describe a first-request-after-a-cold-compile-cache.

## What would reverse this

- A replicated cold-vs-cold pair at 128K in which `block=48` loses to the stock
  ladder. n = 1 cold per arm per round is this record's weakest joint, though the
  n = 5 warm pairs agree with it.
- A workload whose stock-ladder acceptance is *not* context-invariant — in
  particular the append-only agent-harness shape, which is the real target and was
  not measured.
- Differential acceptance degradation exceeding ~30 % of block-48's
  tokens-per-pass at 256K. Measured −5.4 % at 128K; 256K is unmeasured, and the
  128K → 256K step is where decision 007's projection was least constrained.
- An MLX or MTPLX release changing the verify-pass kernel or the context-copy
  proposer contract.

## Handed forward

1. **The default-flip card**, if anyone wants it, needs: multi-case coverage,
   replicated cold samples, and the append-only agent-harness shape. Not this
   record's scope.
2. **Decision 007's amendment 2 is still owed** — item 4's refuted GQA mechanism
   and the contaminated bandwidth hand-off (`VERDICT-SC-P4DC.md` §§1.1–1.3).
   Moot for the gate, still wrong on the record.
3. **The operator `powermetrics` run** above, which also settles 007's
   *Handed to the operator* section and the CLAUDE.md rule 9 bandwidth constant.
4. **The EMA-guard hazard** (decision 006) is confirmed dormant at 128K on this
   workload — 0 suspensions in all 23 requests measured here (21 at 128K, 2 at
   32K), and 0 in the prior ~800-token `rename-identifier` rows.
5. **Add a paging-*rate* check to `rafale/bench/hygiene.py`.** This measurement
   would have been aborted by an occupancy check and was correctly cleared by a
   rate check; decision 007's hygiene note asks for physical-consistency checks
   alongside spread checks, and this is a concrete one.
