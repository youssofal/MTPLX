# VERDICT — SC-P4DA (RAMP serving-path proposer)

**Role:** Strong Card BREAKER (JUDGE-PROTOCOL, `coder ≠ grader ≠ breaker`)
**Card:** `SC-P4DA-ramp-serving-path-proposer.md` (frozen, v2, undispatched)
**Baseline commit reviewed:** `bb90485` · worktree clean (`git status --porcelain` empty), matching the card's stated baseline
**Date:** 2026-08-26

---

## VERDICT: **ACCEPT-WITH-EDITS**

The card is one of the better ones this project has produced. Its central
technical claim — the no-diff `context_copy` seam — **survives adversarial
attack completely**; I re-read the installed engine and every load-bearing
citation is byte-accurate. Its three "measured and disconfirmed" mechanism
kills reproduce exactly from the raw JSON. Its honesty about the invalidated
sweep is exemplary and is the reason several of the findings below were
findable at all.

It is not ACCEPT because the card contains **four factual errors, one
unsatisfiable acceptance criterion, one acceptance criterion that provably
cannot detect the failure it exists to catch, and a load-bearing safety
property that appears only in risk prose and in no instruction a coder would
execute.** It is not REJECT because every one of those is fixable by editing
the card; no new POC work is required except what the card already schedules.

**A hard blocker is also confirmed live and must be cleared by the operator
before dispatch** (§6).

**Langfuse:** no worker trace was pulled and none exists — this card has not
been dispatched, so there is no dispatch window to query (JUDGE-PROTOCOL
§1.3a requires this be stated rather than silently skipped).

---

## 1. Attacks that LANDED

### 1.1 — The long-context risk is *estimable*, and the estimate points at the card's own kill criterion (STRONGEST)

The card calls this "the single largest risk" and stops at "unmeasured."
That understates what is knowable. The project's own committed constants make
a break-even calculable in ten lines:

- decision 003: KV is **64 KB/token** across the 16 full-attention layers
- plan: **~300 GB/s** effective, **~28 GB** of Q8 weights per pass
- `config.json`: `num_attention_heads=24`, `head_dim=256`, 16 attention layers

Per verify pass of `T = 1 + L` rows at context `C`:

```
bytes   = 28 GB + 64 KB × C                     (weights + KV, both read ONCE)
flops   = 4 · T · C · 24 · 256 · 16             (QK^T + AV)
t_pass  = bytes/300e9 + flops/FLOPS_achieved
```

The `T`-dependent term is **attention compute**, and it is negligible at short
context and dominant at long: at `T=49` it is **0.015 TFLOP at 800 tokens** and
**5.05 TFLOP at 256K** — a 330× swing. Throughput `= tok_per_pass(L) / t_pass`,
using the card's own offline `tok/pass` column:

| context | achieved FLOPS | blk8 | blk32 | blk48 | blk64 | blk96 | best |
|---|---|---|---|---|---|---|---|
| 800 tok | 30 TF | 150.0 | 176.8 | 213.1 | 234.8 | 214.2 | **64** |
| 800 tok | 60 TF | 150.1 | 177.1 | 213.7 | 235.7 | 215.4 | **64** |
| 128K | 30 TF | **102.2** | 92.9 | 97.2 | 94.7 | 70.2 | **8** |
| 128K | 60 TF | 108.3 | 110.4 | 122.2 | 124.4 | 98.6 | **64** |
| 256K | 30 TF | **77.3** | 62.8 | 62.8 | 59.2 | 41.8 | **8** |
| 256K | 60 TF | 84.5 | 80.0 | 85.4 | 84.3 | 63.8 | **48** |

At short context the model reproduces the measured optimum (~48–64), which is
the sanity check that it is not nonsense. In the **target 128K–256K band the
optimum collapses to the shortest block under half the plausible range** — i.e.
to the stock ladder, which is verbatim the card's fourth kill-criterion bullet.

So the honest statement is not "unmeasured." It is: **the outcome hinges on one
measurable scalar (achieved attention FLOPS at long context), and roughly half
the plausible range of that scalar kills the card.** That is a far more
actionable framing, and it converts an open-ended risk into a single
pre-dispatch measurement.

**The mechanism the card quotes is also wrong.** The card and POC both cite the
GPT-5.6 consult verbatim: *"verification attention may read a large KV cache
**for every proposed row**."* It does not. The `T` rows go through the engine as
a single batched attention call —
`forward_ar_capture(mx.array([[primary] + _cc_block]))`, `generation.py:7845-7852`
— so the KV cache is read **once per pass regardless of `L`**. What grows with
`T` is FLOPs, not KV bandwidth.

This distinction is not academic. A KV-bandwidth story implies **Phase 6 KV
quant would rescue RAMP at long context**; the compute story says it would not,
and would send the long-context follow-up card down the wrong mitigation path.
The card imports the consult's mechanism uncritically.

**Also unquoted: the consult's most damaging point.** `consult-gpt56.txt` Q2:
*"There is no defensible universal crossover `T` from the supplied numbers … **optimize accepted tokens per verify time, not accepted tokens per pass.**"*
That is a direct methodological rejection of the metric the entire §3 ablation
is built on, and it is the one consult answer the card never surfaces, while
quoting Q4 and Q5 at length. At 800 tokens `tok/pass ≈ tok/time` because pass
time is `L`-independent; that equivalence is exactly what fails at 128K.

### 1.2 — The EMA-guard safety argument is refuted by the card's own committed evidence

Card (second defect site): *"At the measured 0.806 the guard does not fire
(0.806 ≫ 0.35), so this is a **latent** hazard, not a live bug."*

The evidence says otherwise:

| source | blk acc | suspensions |
|---|---|---|
| `block-sweep-clean-24.json` | 0.880 | **1** |
| `block-sweep-clean-32.json` | 0.862 | **1** |
| `ab-bench.json` RAMP arm | 0.806 | **5** (1 per add-method rep) |
| `ab-bench.json` baseline arm | 0.920 | **5** |
| `traces.json` add-method (stock engine) | 0.942 | **1** |
| invalidated sweep, blocks 16/32/48 | — | 3 of 9 rows |
| invalidated sweep, blocks **64/96** | — | **6 of 9 rows** |

Three things follow, none of them in the card:

1. **The guard fires routinely — including in stock baseline.** Suspension is
   normal operation, not an edge case.
2. **Aggregate block acceptance does not predict firing.** The EMA is a local
   excursion statistic (`0.7·ema + 0.3·ratio`, 4-sample arming); the aggregate
   is not. Cells at 0.880 and 0.862 suspend. "0.806 ≫ 0.35 therefore latent" is
   invalid reasoning about the wrong statistic.
3. **Longer blocks measurably recruit new traces into suspending** — 3/9 rows at
   blocks ≤48 versus **6/9 at 64 and 96**, where `rename-identifier` starts
   suspending too. The hazard is not merely "narrowing margin"; its incidence is
   already observably growing with `L`.

**Consequently acceptance criterion #4 is keyed on the wrong quantity.** It
instructs the worker to report the aggregate per-block ratio and stop if it is
below 0.45 — a statistic already demonstrated not to predict the event, and a
threshold (0.45) with no stated derivation. A worker will measure ~0.80,
conclude "safe," and record a hazard that is firing five times per bench run.

The consult's Q4 (quoted in the card only for its reassuring half) actually
prescribes the fix: control on *accepted tokens per unit verification time*, not
on raw block acceptance.

**Compounding this:** the replay's acceptance ratio is biased **high** against
the live engine in all three traces (+0.36, +1.15, **+2.61** points), so every
offline margin in the ablation is optimistic on precisely the quantity this
argument rests on.

### 1.3 — The "clean" sweep is not clean, and criterion 5b provably cannot detect that

POC §6a: *"Each cell still carries one slow outlier (the `min` column)."*

Raw `block-sweep-clean-24.json`, the eight measured runs:

```
149.49, 149.38, 100.11, 97.38, 48.02, 149.84, 150.15, 149.52
```

**Three of eight runs are gross outliers — 37.5% of the cell is contaminated**,
not one. The file's own `relative_spread` is `0.683`, i.e. **68.3%**, rendered in
`block-sweep-clean.txt` as `spread= 0.68` immediately beside `blk_acc=0.880`,
which invites reading it as "0.68%". Cell 32 carries one outlier and the POC's
description is correct there; cell 24's is not.

Now the structural problem. Card Fix step 5b: *"Require median and p95 to agree
within ~2%; a cell where they diverge is contaminated and must be re-run."*

Cell 24 **passes that test** — median 149.44 vs p95 149.84, a 0.27% divergence —
while being 37.5% contaminated. It passes because `p95 > median` always, so the
test probes only the **upper** tail, and swap-pressure contamination lands
entirely in the **lower** tail. The card's contamination test is structurally
blind to the exact contamination that produced the data it was written from.

A test that would have caught it: require `(max−min)/median` below a threshold,
or require all `n` runs within a band of the median, or report the count of runs
below 0.9×median. None of these is in the card.

### 1.4 — Acceptance criterion #2 is unsatisfiable as literally worded

Card #2: *"RAMP's replay counters are **identical** — not close — to **the engine
proposer's** on all three committed traces. Assert equality on every counter. A
near-miss here means the reimplementation drifted and every other number is
void."*

POC §3 makes the same claim: A1 *"produces byte-identical counters to the
engine's own."*

Both are false as written. `A1-control-ladder` is byte-identical to
**`V0-engine-exact`**, which is itself a *replay* of the engine's policy. The
replay differs from the **live engine** by up to **+6.67%** (rename-identifier
rounds, 30→32), with all 15 non-suspension deltas ≥ 0 — a systematic positive
bias, not noise. Asserting counter-*equality* against the live engine's
`context_copy_*` telemetry, as #2 literally instructs, **fails by construction**.

A coder who reads "the engine proposer" as the live engine will burn the card on
an impossible criterion and, per the card's own wording, conclude "every other
number is void." Must be reworded to name `V0` explicitly.

### 1.5 — The A/B is not counterbalanced, and its headline is a case-mix artifact

`ab-bench.json` row order is strictly alternating, but **baseline runs first in
all 15 pairs**. Any monotone drift is therefore fully confounded with arm, and
there is a visible warm-up signature: baseline `rename-identifier` *declines*
across the first three reps (121.0 → 117.0 → 111.9) while RAMP *rises*
(172.3 → 175.4 → 175.2 → 179.3 → 179.6). ABBA counterbalancing costs nothing and
was not done.

Separately, the pooled **+53.9%** is a mixture of three case distributions with
different means. Per case the speedups are **+51.3% / +51.2% / +63.3%**. The
defensible claim is a **+51%–63% range**; 53.9% is an artifact of case mix and
should not be quoted to three significant figures.

To the card's credit, within-case run-to-run spread is ≤7.9% with no outliers —
**this is the cleanest data in the evidence set**, and the direction and rough
magnitude of the win are solid. Only the precision and the drift control are
overstated.

### 1.6 — The "off-by-default flag" exists only in prose

The single sentence that makes this card safe to merge given §1.1 —
*"This card ships it behind an explicit, off-by-default flag"* — appears **once**,
at line 679, inside the *Largest open risk* narrative. It appears in **no** Fix
step, **no** Touch List entry, **no** Gherkin scenario, and **no** acceptance
criterion. Nothing a coder executes mentions it.

Meanwhile the POC's own defaults are **on**: `RAMP_BLOCK = _env_int("RAMP_BLOCK", 48)`
and `RAMP_FUZZY` defaults to `1` (`ramp_patch.py:58-61`). A worker following the
Fix section and modelling it on the POC ships RAMP enabled by default at block 48
— the exact configuration §1.1 says may be a long-context regression.

### 1.7 — Provenance gaps in the single most load-bearing evidence file

`ab-bench.json` contains keys `['rows','by_arm','identity']` — **no environment
block, no config, no version stamp, no timestamp**. This violates CLAUDE.md's own
results convention ("environment block: macOS build, power mode, thermal notes,
run count").

- `RAMP_BLOCK=48` **is** recoverable arithmetically — every RAMP row has
  `drafted/rounds` exactly 48 (864/18, 816/17, 528/11). Good.
- `RAMP_FUZZY=1` is **not recoverable from any committed artifact.** The
  `[ramp] installed: block=… fuzzy=…` line goes to an uncommitted `/tmp` log.

So the headline is attested for its block length and **unattested for its fuzzy
setting** — which matters, because the card's whole retrieval story ("the fuzzy
re-anchor is what pays") rests on it, and because block-only at 48 scores +42.7%
offline versus +63.1% with fuzzy.

Worse, **fuzzy hits are unmeasurable from any live artifact by construction**:
`RampNgramIndex.find` reports `ext=0` for fuzzy hits (`ramp_patch.py:136-139`),
which the engine records verbatim, making them indistinguishable from weak exact
hits in every trace. The `STATS` dict at `ramp_patch.py:63` that would
disambiguate is **dead instrumentation** — never read, never exported, never
printed. The card's Fix 1d correctly requires real per-source counters; that is
the right fix and it should be flagged as *closing a hole the POC has*, not as a
nice-to-have.

### 1.8 — The patched launcher diverges from decision 001, by its own stated standard

`ramp_launch_patched_server.py` docstring: *"Anything that differs from the
baseline launch line is a bug in this file."*

It copies the 40-odd CLI flags faithfully. It reproduces **none** of the
environment half:

```
launch_baseline_server.sh:16   unset MTPLX_PREFILL_CHUNK_SIZE MTPLX_PREFILL_CHUNK_SIZE_DENSE MTPLX_PREFILL_CHUNK_SIZE_REPAGE
launch_baseline_server.sh:20   export MTPLX_CONTEXT_COPY=1
ramp_launch_patched_server.py  (neither — no MTPLX_* handling at all)
```

The two A/B arms are therefore configured by **different mechanisms**: baseline
hermetically from a committed script, RAMP inherited from an unrecorded shell.
Fix step 4 tells the worker to build `scripts/launch_ramp_server.sh` to exactly
the right standard — but a worker will reach for the POC launcher as the
template, and it is the thing that is wrong.

Related, undocumented, and sharper than it looks: `ramp_ab_bench.py:54` rebuilds
its prompts by reading `rafale/draft/ngram.py` **live from the working tree**,
rather than using the frozen `prompt_ids` sitting in the same `traces.json` it
already opens. Any commit touching that file silently changes the benchmark
workload with no error. (`ngram.py` is scope-fenced, so this card is safe — but
the coupling is unstated and will bite a later card.)

### 1.9 — System-coherence: two upstream documents are corrected in prose and left unamended

- The card's heading asserts decision 005 *"mis-tiers"* the seam. Its own body
  then concedes Tier 3 *"is correct for arbitrary tokens and wrong for this
  card's needs."* Both are true and the heading is the misleading one: RAMP is
  **proposer substitution**, a seam decision 005 never tiered at all — not
  draft-token injection, which decision 005 tiers correctly. Nothing in the card
  amends decision 005, so the next reader still sees an incomplete seam table,
  and a future worker may read "005 was wrong about Tier 3" and attempt real
  injection expecting it to be easy.
- Plan §4D specifies indexing matches *"anywhere in the 256K context"* — i.e.
  including the model's own generated text. The POC measured exactly that
  (`C1-wide`) at **-0.7%** and killed it. The plan is not updated.

Per Principle 1 (system coherence outranks local correctness) and the project's
audit-trace rule, both need a one-line amendment, not just a mention in a review
document.

### 1.10 — Minor but real

- **`tok/pass` ranking is not as `mtp_advance`-stable as implied.** The top-3
  block comparison the card actually pins on (B3/48 > A5/64 > A4/48) is stable
  across adv=1 and adv=3 — that part holds. But 10 of 16 variants change rank;
  **A6/96 falls from rank 4 to rank 10**, and **C1-wide's "-0.7% KILLED" verdict
  is adv=3-dependent** (exactly tied with V0 at adv=1). The C1 kill is correct in
  direction — it is never a win — but "KILLED, -0.7%" is stated more firmly than
  the data supports.
- **Fidelity holds only at adv=3.** Max delta is +6.67% at adv=3 but **+70.91%
  at adv=1**, and at adv=2 the add-method suspension event *disappears*. The POC
  frames adv as an uncertain parameter "reported at both ends"; in fact only
  adv=3 is validated, which is a stronger position than claimed — the card
  undersells itself here.
- **`install()` has no runtime guard.** If upstream stops importing
  `block_for_ext`, the patch installs, prints `[ramp] installed: block=48 …`, and
  **silently does nothing**. The card puts the guard in a *test*
  (`tests/test_engine_seam.py`), which is correct and necessary — but a test is
  not run on the serving machine before a benchmark. This is CLAUDE.md rule 1's
  failure class ("never trust an 'enabled' flag") and wants a runtime
  verify-or-raise inside `install()` too.
- **Line numbers rot fast.** Same import sits at `generation.py:6510` in mtplx
  2.3.0 vs `7479` in 2.7.1 — 969 lines of drift across two minor releases.
  Signatures are stable; positions are not. The seam guard should assert on
  *symbols and signatures* (as the card says) and treat every `file:line` in the
  card as a soft reference.
- **`context_copy_block_k` rebind is dead code in the POC.**
  `ramp_block_for_ext` returns `RAMP_BLOCK` and never reads `k_cap`
  (`ramp_patch.py:142-145`), so the "stock cap of 24 silently re-clamps" trap
  that POC §4 reports as *"found by execution"* is unreachable in the shipped
  code. Keep the instruction — it is correct defensive design for a production
  module that *does* honour `k_cap` — but fix the provenance; it is a reasoned
  precaution, not an observed trap.
- **One token per trace is missing from capture**: engine `completion_tokens`
  sums to 1927, `len(output_ids)` to 1924.

---

## 2. Attacks that did NOT land (reported per breaker duty)

### 2.1 — "Zero engine source changes" — **fully survives.** My hardest attack, cleanly refuted.

I tried to break this and could not. Verified against the installed engine:

- The import at `generation.py:7479-7482` **is** function-local, inside
  `def generate_mtpk(` (`:6136`), at indent 1, executed per call.
- **Every** rebound symbol is reached only through it. Package-wide grep gives
  exactly three consumers — `NgramIndex` at `:7535`, `block_for_ext` at `:7835`,
  `context_copy_block_k` at `:7532`. The only other `from .context_copy import`
  in the package (`:6316`) imports a symbol RAMP does not rebind. `mtplx/__init__.py`
  re-exports nothing from `context_copy`. **No stale binding path exists.**
- Signatures match exactly, including the keyword-only `max_pos` on `find`.
- Ordering is *more* robust than claimed: `install()` runs before
  `mtplx.generation` is imported at all, so the patch would survive even if
  upstream hoisted the import to module scope.
- All three cited line ranges (`7479-7482`, `7536-7539`, `7870-7876`) and the
  ladder at `context_copy.py:92-100` are **byte-accurate** in the installed
  source. Installed versions are **exactly** the card's pin: mtplx 2.7.1,
  mlx-lm 0.31.3, mlx 0.32.0.

The seam finding is correct, well-evidenced, and the card's headline claim is
earned.

### 2.2 — Prime-directive attack on temperature > 0 — **refuted, but it left a real residue**

I attacked the claim that "the draft source cannot move the output law,"
hypothesising that copy-drafting commits argmax tokens unconditionally, which
would mean RAMP roughly doubles greedy-forcing at temperature > 0 while the
card's temp-0 gate stays blind to it.

**Refuted by the source.** `generation.py:7870` guards the argmax path on
`if sampler.temperature <= 0`, and the `else` branch does proper per-token
rejection sampling against the residual, with the comment: *"the emitted stream
follows the target sampling distribution exactly at any temperature."* The
engine is distribution-preserving. The card's structural argument is sound and
my attack fails.

**But the residue is real and unstated.** Those are two *structurally different*
acceptance algorithms: longest-common-prefix argmax at temp 0, per-token
stochastic rejection at temp > 0. Under rejection sampling, acceptance decays
multiplicatively along the block, so **the optimal block length at temp > 0 is
almost certainly shorter and the +53.9% does not transfer.** Every number in this
POC — traces, A/B, both sweeps — was captured at `temperature: 0`
(`ramp_capture_traces.py:114`, `ramp_ab_bench.py:62`). The card treats
temperature 0 solely as a *quality-gate* condition (CLAUDE.md rule 2) and never
states that it is also a *performance-regime* condition. That belongs in
Non-goals.

### 2.3 — "Acceptance #6 references a script that does not exist" — **does not land**

`scripts/launch_ramp_server.sh` is indeed absent, but Fix step 4 creates it.
Criterion #6 is executable in sequence. (Flagged because a parallel audit pass
raised it; on cross-check against the card it is not a defect.)

### 2.4 — Block=48 provenance — **the card is already honest, and correct**

I attacked the 2-cell sweep as insufficient basis for block=48. It is
insufficient — and the card **already says so**, in v2, and handles it properly.
Confirmed independently: no clean wall-clock evidence for 48 exists; the A/B at
48 is clean evidence for *48-vs-stock* and for nothing else, since it carries no
24/32/64/96 arm. The card's amended step 5 (re-run with a swap check, take the
highest median, prefer the shorter on a 5% tie, explicit documented fallback to
48 on the A/B alone) is the right instruction. **No edit required beyond fixing
5b's contamination test (§1.3).**

One correction to the POC's invalidation narrative, for the record: §6 blames the
64-cell dip on timing noise, but per case the block=64 cell is *internally tight*
(78.93/78.18/76.06, 3.8% spread) while block=48 is the wild one
(179.55/87.92/220.03). The sweep is genuinely invalid — pooled medians across
three cases plus contention — but the specific diagnosis offered does not explain
the 64 cell.

---

## 3. Required edits before dispatch

**Blocking — the card cannot go to a coder until these are fixed:**

1. **Reword acceptance criterion #2** to assert counter-equality against
   **`V0-engine-exact`**, naming it explicitly, and state the separate, weaker
   fidelity check against live engine telemetry as "within 7%, one-sided." Fix
   the same overstatement in POC §3 ("byte-identical to the engine's own").
2. **Replace criterion 5b's contamination test.** Median-vs-p95 is blind to the
   lower tail. Require `(max−min)/median ≤ X`, or a count of runs below
   0.9×median, or all-`n`-within-band. Correct POC §6a's "one slow outlier" to
   three for cell 24, and print `relative_spread` as a percentage so `0.683`
   cannot read as 0.68%.
3. **Rewrite acceptance criterion #4.** Aggregate block acceptance does not
   predict guard firing — cells at 0.880 and 0.862 suspend. Require the worker
   to record the **suspension count per run** and, if obtainable, the **minimum
   EMA excursion**; drop or derive the unexplained 0.45 threshold; note that
   stock baseline also suspends so the comparison is RAMP-vs-stock incidence,
   not "does it fire at all."
4. **Put the off-by-default flag in the Fix section, the Touch List, a Gherkin
   scenario, and an acceptance criterion.** Prose at line 679 is not an
   instruction. Explicitly warn that the POC's defaults are on.
5. **Add the roofline break-even from §1.1 to the *Largest open risk* section**,
   correct the mechanism from "KV bandwidth per proposed row" to "attention FLOPs
   ∝ T×C" (KV is read once per pass, `generation.py:7845-7852`), and quote the
   consult's Q2 (`optimize accepted tokens per verify time, not per pass`)
   alongside Q4 and Q5. Name **achieved attention FLOPS at long context** as the
   single scalar that decides the kill criterion, and make measuring it the first
   task of the long-context follow-up card.

**Non-blocking but should be done in the same pass:**

6. Require a **runtime verify-or-raise inside `install()`**, not only in
   `tests/test_engine_seam.py` (CLAUDE.md rule 1).
7. Warn the worker that `ramp_launch_patched_server.py` is **not** a valid
   template for Fix step 4 — it reproduces decision 001's CLI flags but none of
   its environment (`unset MTPLX_PREFILL_CHUNK_SIZE*`, `export MTPLX_CONTEXT_COPY=1`).
8. Require the new bench/launch scripts to **write an environment + config block
   into their JSON output** (`RAMP_*`, engine versions, swap, timestamp), per
   CLAUDE.md's results convention. `ab-bench.json` has none.
9. Soften "the A/B measured +53.9%" to the per-case range **+51%–63%**, and note
   the arms were not counterbalanced (baseline first in all 15 pairs).
10. Add to **Non-goals**: "Not validated at temperature > 0" — the engine takes a
    structurally different acceptance path there (§2.2) and no POC number applies.
11. Fix the provenance of the `context_copy_block_k` claim (reasoned precaution,
    not a trap hit in execution); soften C1's "KILLED" to note the -0.7% is
    adv=3-dependent; note fidelity is validated only at adv=3.
12. Reframe the decision-005 heading from "mis-tiers" to "adds a seam 005 does not
    tier," and **amend decision 005 and plan §4D** rather than only correcting
    them here.

---

## 4. What survives untouched

Stated plainly, because the edit list above is long and the card deserves this:

- **The no-diff seam.** Verified against the installed engine; every citation
  byte-accurate; no stale binding path; signatures exact. §2.1.
- **The three mechanism kills.** `C1-wide` (-0.7%), `D1-consensus` (+0.0%),
  `E1-consensus-block` (-52.4%) all reproduce **exactly** from
  `replay-results.json`. All 12 quoted `tok/pass` values reproduce to full float
  precision — **zero numeric discrepancies** in the ablation table.
- **The A1≡V0 control.** Identical on every counter, per-trace and aggregate.
  This is the row that makes the rest meaningful and it is real.
- **Temperature-0 output identity.** 1 unique sha per case per arm, identical
  across arms on all 3 cases, and `sha_unique=1` in every sweep cell at every
  block length tried. The prime-directive gate is genuinely GREEN.
- **The direction and rough size of the win.** +51%–63% per case, ≤7.9%
  within-case spread, no outliers. Solid.
- **The v2 amendment of step 5.** Correct, honest, and properly fenced. §2.4.
- **The decision to commit the invalidated sweep.** Every one of findings 1.2,
  1.3 and 2.4 was findable *because* the writer committed contaminated data
  rather than deleting it. That is the protocol working.

---

## 5. Assessment of the card against its own kill criterion

Nothing here trips it. Bullets 1 and 2 (A/B beats baseline; temp-0 identity) are
measured and green. Bullet 3 (seam guard) is unbuilt but demonstrably buildable.
Bullet 4 (long-context optimum collapses to the stock ladder) is **not tripped,
but §1.1 shows it is closer to tripping than the card conveys** — and it is
testable before, not after, the implementation work.

---

## 6. BLOCKER — the first-fail pytest lock is real, and independently confirmed spurious

**It is real.** Verified by execution, not by reading the card:

```
$ printf '%s' "/Users/misterj/src/super-duper-disco/.claude/worktrees/phase-0-bench-harness" | shasum -a 256
eb8c8df358735596d2b95f5b9a1d40beebdf745426a7a804a91b70c439f9650e
```

— which is exactly the lock filename. The lock keys to **this** worktree. Both
invocation styles are hard-blocked (exit 2, `PreToolUse` deny):

- `PYTHONPATH=… .venv/bin/python -m pytest -q --collect-only` → **BLOCKED**
- `uv run pytest -q --collect-only` (i.e. **`make test`**) → **BLOCKED**
- `uv run ruff check .` (i.e. `make lint`) → **passes clean**

So acceptance criteria **#1, #4 and #5** cannot run, the baseline suite count
cannot be established, and the project's Definition of Done item 4
(`make lint && make test` clean) is **unsatisfiable in this worktree**. The card
is right to call this a pre-dispatch blocker.

**It is spurious, and I can name the mechanism.** The lock's recorded command is
`sed -n '1610,1720p' …/RULES.md` — a documentation read, not a test run. Root
cause is in `~/.claude/hooks/sc-firstfail-lib.sh`:
`sc_ff_is_readonly_inspection_command()` allowlists
`(ls|cat|head|tail|less|more|grep|egrep|fgrep|rg|find|wc|file|stat|pwd|echo|printf|jq)`
— **`sed` is not in that list** (nor is `awk`). So the `sed` output fell through
to the `sc_ff_has_pytest_summary()` fallback, matched the string
`"567 passed,"` inside RULES.md's own prose, and armed the lock with no test
tool ever invoked.

This is the **identical failure class** the hook's own comment documents
(RULES §14.1, nukegraph 2026-08-16), where `cat`/`tail` on a file quoting a
pytest summary set the lock. The remedy was applied to `cat`/`tail` and the
allowlist was never generalised — it is a denylist-by-omission, and any
read-only tool not enumerated re-opens the same hole.

**General lesson (JUDGE-PROTOCOL §1.5):** add `sed`, `awk`, and `diff` to
`sc_ff_is_readonly_inspection_command()`, and better, invert the test — the
summary-text fallback should fire only when the command is *known* to execute
something, rather than whenever it is not recognised as read-only.

**I am not clearing the lock.** It belongs to a different card, and a breaker
altering another card's lock exceeds this role. The operator should clear it,
citing this verdict as the judge evidence:

```
"$HOME/.claude/hooks/sc-clear-firstfail-lock.sh" \
  --judge-verdict 'Spurious lock: armed by a sed of RULES.md, not a test run. Root cause: sc_ff_is_readonly_inspection_command() omits sed/awk, so documentation text containing "567 passed," matched the sc_ff_has_pytest_summary() fallback. No test was ever executed. Clear and generalise the allowlist.' \
  --judge-evidence 'docs/reviews/2026-08-26-ramp/VERDICT-SC-P4DA.md §6' \
  '/Users/misterj/src/super-duper-disco/.claude/worktrees/phase-0-bench-harness'
```

**Pre-dispatch sequence:** clear the lock → establish the baseline suite count
from the first green run → write it into acceptance criterion #5 → apply edits
1–5 → dispatch.

---

## 7. Verification of the card's own open-items list

Each POC §7 item, checked rather than taken on trust:

| # | Item | Still open? |
|---|---|---|
| 1 | 800-token evidence vs 128K–256K target | **Open, and understated** — §1.1: estimable, and the estimate points at the kill criterion. Mechanism also mis-stated. |
| 2 | EMA guard margin narrowing | **Open, and mis-analysed** — §1.2: the guard already fires, aggregate acceptance is the wrong statistic. |
| 3 | Fuzzy only safe with long blocks | **Open and correctly stated.** `block=16` at 69.6 t/s vs 138.2 baseline confirms it. |
| 4 | Fuzzy CPU cost never stressed | **Open and correctly stated.** Candidate cap `RAMP_CANDS=8`, scan window `cands[-64:]`, never exercised at scale. |
| 5 | `mtp_advance` is a parameter | **Open, and the card undersells itself** — only adv=3 is fidelity-validated (§1.10), which is a *stronger* position than "reported at both ends." |

**Risks the writer missed entirely** (none of these appear in card or POC):

- **Temperature > 0 is a different performance regime**, not just a different
  gate condition (§2.2).
- **A/B arms not counterbalanced**; drift confounded with arm (§1.5).
- **The RAMP arm's server environment diverges from decision 001** (§1.8).
- **The benchmark workload is coupled to mutable repo files**, not to the frozen
  `prompt_ids` in the same file the script opens (§1.8).
- **`STATS` is dead instrumentation**; exact-vs-fuzzy attribution is unmeasurable
  from any committed live artifact (§1.7).
- **The replay's control arm is a reimplementation, not the engine.** Only `V0`
  uses the imported `NgramIndex`; `A1` and every RAMP row use a from-scratch
  index whose agreement with the engine is *empirical, not structural* — it scans
  a 64-candidate window where the engine scans 32 (`context_copy.py:136`), and
  ranks by 24-token backward similarity where the engine ranks by longest
  backward extension. Identical on these three ~800-token traces; not guaranteed
  on longer or more repetitive ones. The card's acceptance #2 re-asserting this
  as a test is the right mitigation — one more reason to fix its wording (§1.4).

---

**Breaker:** Claude Opus 5, adversarial pass over frozen card SC-P4DA at `bb90485`.
**Method:** independent re-derivation from raw evidence JSON; direct read of the
installed MTPLX 2.7.1 source; live execution of the blocked pytest paths;
first-principles roofline modelling from committed project constants. The card's
summary tables were not treated as evidence.
