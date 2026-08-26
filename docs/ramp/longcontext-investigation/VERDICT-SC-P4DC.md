# VERDICT — SC-P4DC (long-context record correction + dead-zone grid)

**Role:** Strong Card BREAKER (JUDGE-PROTOCOL, `coder ≠ grader ≠ breaker`)
**Card:** `SC-P4DC-longcontext-record-correction-and-deadzone-grid.md` (frozen, undispatched)
**Commit reviewed:** `176012c` · worktree `phase-0-bench-harness`, branch `worktree-phase-0-bench-harness`, clean
**Also reviewed critically:** `FAILRUN-SC-P4DB.md`, `VERDICT-SC-P4DB.md` §1.2,
`docs/decisions/007-ramp-long-context-block-length.md` (amended)
**Round:** 3 of the RAMP long-context investigation
**Date:** 2026-08-26

---

## VERDICT: **REJECT**

Not because the card is sloppy — it is the best-scoped card in this project's
history, its fences are correct, and acceptance criterion 2 is exactly the
mechanical check the fail-run promised. It is REJECT because of what it would
have a coder do:

> **Fix step 3 instructs the coder to write a mechanism into four committed
> artifacts as established fact, and that mechanism is not supported by the data
> it cites.** The `6.13× = num_attention_heads / num_key_value_heads = 24 / 4`
> match is an artifact of dividing by a quantity that the same evidence file
> proves is **not** one KV read. Under the physically defensible denominator the
> ratio is **8.6–10.1**, it drifts with context, and it matches no head-group
> arithmetic at all.

This is the third consecutive round in which a mechanism claim was carried
forward without re-deriving its denominator. Round 1's writer asserted "reads the
KV once." Round 2's breaker refuted it and asserted "reads it 6× per query head
group." Round 3's fail-run analyst wrote *"Re-derived here and confirmed exactly"*
(FAILRUN §1, Link 1) — and reproduced the arithmetic, not the inference. I
reproduced the arithmetic too. It is right. What it means is not.

The card cannot be repaired from inside itself, which is the second reason for
REJECT rather than ACCEPT-WITH-EDITS: **decision 007 already carries the wrong
mechanism as a Decision item (007 item 4), and this card's scope fence explicitly
forbids touching it.** The artifact most damaged by the error is the one the card
puts out of reach. Fixing that changes the card's Touch List, its Defect site 1,
its Fix step 3, and its Non-goals. That is a re-issue, not an edit.

`make lint && make test` **verified live at `176012c`**: `ruff check` clean,
`ruff format --check` clean over 50 files, `102 passed, 1 skipped` in 0.05s.
This is the baseline count acceptance criterion 1 must be measured against.

---

## 1. Attacks that LANDED

### 1.1 — The `6.13× ≈ 24/4` mechanism is an artifact of the denominator (STRONGEST)

`ramp_kernel_regimes.py:57` names the T=1 cell `one_full_kv_read_ms`. Every
downstream claim — VERDICT-SC-P4DB §1.2, FAILRUN §1 Link 1, SC-P4DC Defect
site 1, decision 007 item 4 — treats that label as true. **It is false, and the
committed data proves it in the same file.**

Fit the vector regime (`T = 1..5`) properly. It is not a proportionality; it is
an affine law with an excellent fit and a large, non-zero intercept:

| context | vector intercept `a` | vector slope `b` (one KV pass) | R² | `a` as % of the `T=1` cell |
|---|---|---|---|---|
| 32K  | **1.78 ms** | 2.728 ms/row | 0.9988 | **39.1 %** |
| 128K | **6.54 ms** | 10.353 ms/row | 0.9995 | **38.3 %** |
| 256K | **12.02 ms** | 22.163 ms/row | 0.9980 | **34.9 %** |

The intercept is not launch overhead: it scales with context (0.054 / 0.050 /
0.046 µs per 1000 tokens — flat to ±8% across a 8× context range). It is a real,
replicated, **context-proportional, `T`-independent** term, and it is 35–39 % of
the number three documents call "one full KV read."

So the correct unit for "how many KV passes does the tiled kernel's fixed cost
represent" is `b`, not `ms[1]`. Redo the division three ways, all from the same
committed cells and the same lowest-spread-per-cell rule the writer used:

| context | tiled intercept `I` (OLS `T≥6`) | `I / ms[1]` *(the committed claim)* | `I / b` | `(I − a) / b` |
|---|---|---|---|---|
| 32K  | 23.48 ms  | **5.16** | **8.61** | **7.96** |
| 128K | 104.59 ms | **6.13** | **10.10** | **9.47** |
| 256K | 210.44 ms | **6.11** | **9.50** | **8.95** |

`I / b` and `(I − a) / b` are the estimators that do not depend on the T=1 cell.
They land at **8–10 KV-read-equivalents**, they are not constant across context,
and they are nowhere near 6. Only the estimator that divides by a provably
inflated denominator lands near 6 — and even that one gives **5.16 at 32K**, a
14 % miss from an integer ratio that is supposed to be structural.

**The prior breaker's excuse for the 32K miss is itself refuted here.**
VERDICT-SC-P4DB §1.2 writes off 5.16 as *"slightly under at 32K (where the fixed
cost is smaller relative to launch overhead)."* If launch overhead inflated the
T=1 denominator, it would inflate it at every context. The table above shows the
inflating term is 35–39 % **at all three contexts**, not a 32K peculiarity — and
that term is exactly what kills the 6× reading everywhere, not just at 32K.

**The "self-consistency cross-check" is circular.** SC-P4DC Defect site 1 and
decision 007 item 4 both offer: *"8.59 GB in 17.05 ms = 504 GB/s; 6 × 8.59 GB /
104.6 ms = 493 GB/s. Self-consistent."* Those two expressions are algebraically
the same statement. `6 × kv / I` ≈ `kv / ms[1]` **iff** `I / ms[1]` ≈ 6. It is
the ratio restated, not an independent corroboration of it. Presenting it as a
cross-check is the same class of error the card convicts its predecessor of in
Defect site 2.

**The estimator itself was also selected.** The card and 007 describe `104.6` as
an "OLS intercept over `T ≥ 6`." The R² of that fit is 0.938–0.963 on a curve the
same documents describe as convex and rising (0.27 ms/row at `T=6→12`, 1.72 at
`T=48→96`). Fitting a line to a convex curve drags the intercept **down**.
Three defensible estimators of the same fixed cost at 128K:

| estimator | intercept | `/ ms[1]` |
|---|---|---|
| 2-point chord (`T=6, 12`) — the one `ramp_kernel_regimes.py:39` actually computes | 117.79 ms | **6.91** |
| OLS over `T ∈ [6,16]` | 112.82 ms | **6.62** |
| OLS over all `T ≥ 6` — the one quoted | 104.59 ms | **6.13** |

The quoted estimator is the one that lands closest to 6.00. Across all three
contexts the 2-point estimator gives **5.41 / 6.91 / 6.92**. A genuine head-group
multiplier is 6 exactly, at every context, under every estimator. This one is not.

**Blocking, and it is the reason for REJECT.** Fix step 3 says *"Correct the
mechanism in every place it appears … with the 6.13× table and the GQA
arithmetic."* Executing it replaces a wrong label with a differently wrong label
in four artifacts, and 007 already has it. What the data supports is bounded, not
mechanistic: *the tiled kernel's fixed cost is equivalent to **8–10** full KV
passes at the vector kernel's measured per-pass rate, rising slightly with
context; the once-per-pass label is refuted, and no head-count arithmetic
reproduces the measured multiplier.*

---

### 1.2 — There is a third, cleaner bandwidth datum in the same directory, uncited, and it changes the operator hand-off

FAILRUN §6 item 1 and 007's *Handed to the operator* build the case against
CLAUDE.md rule 9's ~300 GB/s entirely out of attention `T=1` cells (472 / 504 /
499 GB/s) and the tiled intercept (493 GB/s). Every one of those is a number
§1.1 just showed to be contaminated by a large non-KV term — the "504 GB/s" is
`kv / (b + a)`, not `kv / b`.

`evidence/dense-microbench.json`, committed in the same POC and cited by 007
line 46, contains a straight weight-streaming measurement with no KV cache, no
mask, and no attention:

> `T=1`: **26.66 GB of Q8 weights in 45.80 ms = 582.2 GB/s**, `rel_spread = 1.30 %`.

That is the tightest cell in the entire evidence set and the only clean
bandwidth measurement available. **Nobody in three rounds cited it for the
bandwidth question.** It puts the CLAUDE.md discrepancy at ≥ 1.94× on evidence
that needs no interpretation, and it independently corroborates §1.1's reading
that the vector kernel's 776–830 GB/s marginal slope is closer to the truth than
its 472–504 GB/s `T=1` rate.

Consequences the successor must carry:

- FAILRUN §6 items 1 and 2 are **one anomaly, not two**, and the natural
  resolution ("`T=1` carries a context-scaling overhead that is not launch cost"
  — FAILRUN's own second alternative, filed as unresolved) is the one the data
  supports. Resolving it that way **destroys the 6× mechanism**. The record
  currently states the mechanism as decided and files the question it depends on
  as open. That is backwards.
- 007's *Handed to the operator* section must be rewritten before an operator
  acts on it: cite the 582 GB/s dense cell as the headline, state the band as
  **582–830 GB/s** on clean estimators, and drop the `T=1` attention figures as
  the primary evidence.
- **In the project's favour, and not noted anywhere:** correcting the rate makes
  007 item 5's reversal *stronger*, not weaker. Bytes = time × rate, and the
  times are measured. At 830 GB/s a 128K `T=48` verify pass moves **130 GB** of
  KV, not 79 GB — 4.6× the ~28 GB of Q8 weights, not 2.8×. The direction of the
  KV-quant reversal survives every reading of the bandwidth question. Only its
  magnitudes are wrong, and they are wrong low.

**Blocking edit.** The bandwidth hand-off cannot be shipped to an operator built
on the contaminated cells while the clean one sits uncited in the same folder.

---

### 1.3 — The dense proxy is cited as corroboration where it supports and dismissed as noise where it does not

007 line 45–46 (post-amendment) keeps: *"corroborated by an identical step in a
dense `quantized_matmul` proxy where no KV cache exists at all."* The card's own
Defect site 4 simultaneously downgrades inferences from that file as
noise-dominated (`rel_spread` 2.75–7.17).

Both cannot stand, and the clean cells in that file cut against the card:
`T=5` = 50.25 ms (spread 1.5 %) → `T=8` = 89.04 ms (spread 3.9 %) is a **+77 %
step across the same `T=6` boundary, in a kernel with no KV cache at all.**

If a KV-free kernel shows the same discontinuity, then "the tiled attention
kernel re-reads the KV six times via GQA head groups" is not needed to explain
the step, and the step is not evidence for it. The step is real (I agree with
both prior rounds on that); the *mechanism* attributed to it is not what the
corroborating evidence corroborates. This is a direct structural counter-example
to Defect site 1 and it is sitting in the committed evidence.

**Blocking edit.** Either the dense proxy is admissible (and then it refutes the
GQA attribution) or it is not (and then 007 line 46 must drop it as
corroboration). Pick one, in writing.

---

### 1.4 — The dead-zone grid is under-specified for the question it exists to answer

Priority-3 attack: partially landed. The grid is not chord-picked — the card is
genuinely honest about the 32× band and acceptance criteria 6 and 7 permit the
new measurement to contradict the committed constants, which is the right shape.
Three specification defects remain:

**(a) The grid does not bound the onset to a uniform ratio.** `1024 → 4096 →
8192 → 32768` is spaced 4× / 2× / 4×. If the crossover falls in `[1024, 4096]`
or `[8192, 32768]` — two of the three intervals — the card ships a 4× bound
while Gherkin scenario 3 only promises "better than 32×." Adding `2048` and
`16384` bounds the whole band at 2× everywhere. These are the cheapest cells in
the matrix (32K takes ~4.5 ms/rep at `T=1`; 1K takes ~1 ms). There is no reason
to leave a 4× interval standing to save seconds of GPU time.

**(b) Acceptance criterion 4's "≥ 5 reps" is far too weak where the answer
lives.** `attn-microbench-c1024.json` ran **20 reps** and still reports
`rel_spread = 0.787` at `T=1`, `0.323` at `T=4`, `0.226` at `T=3`. The
short-context cells are the noisiest in the entire evidence set, and the
dead-zone onset is decided by a `T=5` vs `T=6` difference that at 1024 is
**−0.11 ms** on a ~1 ms baseline. A criterion inherited from 256K runs is not fit
for 1K–8K runs. Specify reps per context (e.g. ≥ 200 below 16K) and require the
`T=5→6` step to exceed its own cell spread before a dead zone is declared
present or absent.

**(c) The `analyse()` `KeyError` hazard is flagged but not resolved.** Fix
step 2 says *"either keep the grids aligned or make the failure explicit"* and
leaves the choice to the coder. `analyse()` hardcodes `rows[1]`, `rows[5]`,
`rows[6]`, `rows[12]`; the low-`T` confirmation file Fix step 1 adds to the
driver stops at `T=12`. A coder who wires the confirmation run for the new
contexts and not the full sweep gets a `KeyError` on `rows[16]`-free data or, in
the other order, silently mixes grids. Decide it in the card.

**(d) Not blocking, but state it:** `_DEAD_ZONE_MIN_CONTEXT` and
`_MIN_PROPOSAL_BLOCK` **do not exist in this repository.** Grepped `rafale/`,
`tests/`, `scripts/` at `176012c`: the only hit is
`ramp_kernel_regimes.py:55`'s `min_proposal_block` JSON key. The label
*"Measured, not chosen"* that Defect site 5(b) attacks appears only in SC-P4DB's
own text. Defect site 5(b) is therefore a critique of a withdrawn card's unwritten
code, presented in the register of a defect in the tree. Rewrite it against what
actually exists: `kernel-regimes.json` lacks the `1024` control, and the
32×-unsampled band is a gap in the artifact, not a bad constant in the codebase.

---

### 1.5 — Acceptance criterion 9 contradicts Fix step 2, and its test does not test what it says

Priority-5 attack: scope discipline is otherwise **sound**. Acceptance criterion
2 exists, is correctly specified (`git diff --stat <baseline> -- rafale/` is
empty), the Touch List contains no path under `rafale/`, and the reviewer
checklist re-checks it. That fence holds. But:

Criterion 9 reads *"No committed `evidence/` file is overwritten or deleted"* and
the scope fence repeats it — **in the same bullet that says "`kernel-regimes.json`
is regenerated."** Fix step 2 and the Touch List both require regenerating it.
A coder following criterion 9 literally cannot execute Fix step 2.

Worse, the criterion's own test does not test its words: `git log
--diff-filter=D` catches deletions (`D`), never overwrites (`M`). The check
would pass on a run that silently clobbered every committed sweep.

**Blocking edit.** State the carve-out: `kernel-regimes.json` is a derived
artifact and is the sole permitted in-place regeneration; every
`attn-microbench-*.json` is append-only. Change the test to
`git log --diff-filter=DM -- <evidence>/attn-microbench-*.json` (must be empty)
plus `git diff --stat` showing `kernel-regimes.json` as the only modified
evidence file.

---

### 1.6 — The one measurement that would settle this is excluded, and a live harness that reaches the real unknown already exists

Priority-6 attack: **lands hard.**

FAILRUN §6 item 3 states it plainly: *"Resolving it needs one
`powermetrics`/counter run, not more modelling, and it decides how much Phase 6
KV quant actually buys."* SC-P4DC does not include that run. It spends metal time
on a 4096/8192 grid whose output binds **no shipped configuration** (007 item 3,
FAILRUN Link 4 — the floor exists nowhere in code) while omitting the cheap
measurement that decides (a) whether the tiled kernel is bandwidth-bound above
`T=6`, (b) the true multiplier in §1.1, (c) the bandwidth constant in §1.2, and
(d) the magnitude of the only Phase-6-steering claim in decision 007.

And the deeper problem: **the decisive quantity is not reachable by any
microbenchmark.** Every round of this thread has agreed that whether RAMP wins at
128K depends on **acceptance at long context**, and every round has declared it
unmeasured. `scripts/ramp_ab_bench.py` already exists, already runs alternating
A/B at temperature 0, and already emits `context_copy_drafted_tokens` /
`context_copy_accepted_tokens` / `context_copy_rounds` per request. The missing
input is a long prompt corpus, not a harness.

Three rounds of modelling have each found and corrected a real error in the
previous round's model, and this round's model is wrong again (§1.1). That is not
a run of bad luck; it is what an unanchored modelling chain looks like. The
project should stop extending it. Ordering that the evidence supports:

1. **One counter/`powermetrics` run** at 128K, `T ∈ {1, 5, 6, 12, 48}` — settles
   §1.1, §1.2 and 007 item 5's magnitude in a single sitting.
2. **The live 128K A/B**, which measures acceptance directly and is the only
   thing that can clear the gate 007 is hedged on.
3. **Then**, and only if either of those leaves the dead-zone onset
   decision-relevant, the 1K–32K grid.

SC-P4DC has this ordering inverted, and its Non-goals fence off (1) and (2)
explicitly.

---

## 2. Attacks that did NOT land

- **Decision 007's amendment is substantive, not hedging** (priority 4). Checked
  against FAILRUN's specific items: item 5 is genuinely **REVERSED**, not
  softened, with a bytes-per-pass table and a stated assumption; the outcome line
  is hedged to *"NOT TRIPPED on projected cost evidence; confirmation pending the
  live 128K A/B"*; all four unquoted chords are published including 64→96 at
  23.7, with the explicit admission that the projection's own winner sits **below**
  the 30-TFLOPS line; item 3 is rewritten to say the floor binds in zero shipped
  configurations; the ~30 % headroom is restated as *differential* degradation; a
  read-the-amendment-log warning sits above the fold. This is the most honest
  decision record in the project. Its defect is §1.1/§1.2's content, not its
  candour.
- **The regime change at `T=6` is real.** I attacked it a fifth way — the
  non-monotone `T=6 → T=7` dip at 32K and 128K, the replication across two
  independently launched 128K processes, the step's presence in the KV-free dense
  proxy, and the near-exact 2.02× scaling of 256K:128K at `T=1`. It holds.
  Nothing in this verdict touches it.
- **Scope discipline holds.** Touch List clean of `rafale/`, criterion 2 correct
  and mechanically checkable, `_installed_block_for_ext` untouched, fences on
  `ngram.py` and MTPLX source correct and well-reasoned.
- **`ramp_kernel_regimes.py`'s lowest-spread-per-cell rule is not cherry-picking.**
  I re-ran it: it is deterministic, source-attributed via `cell_sources`, and
  reproduces `min_proposal_block = 9 / 11 / 10` exactly on the committed evidence.
- **The projection is genuinely independent of the chord table.** Confirmed for
  the third time: `ramp_longcontext_model.py:280` interpolates the measured curve
  and touches no FLOPS scalar. Everything in §1.1 leaves the block-64 ranking
  intact — it damages the *record*, not the projection.
- **`make lint && make test` clean at `176012c`**, run live, not assumed.

---

## 3. Re-issue spec — what SC-P4DD must change

The card's scope, fences, failure protocol and reporting section are good and
should be carried over nearly verbatim. Seven changes:

1. **Defect site 1 restated as a bound, not a mechanism.** Publish the three
   estimators of §1.1 side by side. Assert only: the tiled fixed cost is
   **8–10 vector-kernel KV passes**, drifting with context; `one_full_kv_read_ms`
   in `ramp_kernel_regimes.py:57` is misnamed and must be renamed
   (`t_at_T1_ms`); the vector regime is affine with a context-proportional
   intercept of 35–39 % of the `T=1` cell, unexplained. Delete the `24/4 = 6`
   claim and the circular 504/493 cross-check everywhere they appear.
2. **Re-open `docs/decisions/007`** — add it to the Touch List. Amendment 2 must
   correct item 4's mechanism, rewrite *Handed to the operator* around the
   582 GB/s dense cell (§1.2), restate item 5's bytes table under the corrected
   rate (noting the reversal gets stronger, not weaker), resolve the dense-proxy
   contradiction (§1.3), and add to *What would reverse this*: *"a counter run
   showing the tiled kernel's fixed cost is not ~8–10 KV passes."*
3. **Add the counter/`powermetrics` run** at 128K, `T ∈ {1, 5, 6, 12, 48}`, as
   Fix step 1. It is the cheapest decisive measurement available and it currently
   sits in Non-goals.
4. **Fix the grid** (§1.4): add `2048` and `16384`; specify reps per context with
   a floor of ~200 below 16K; require the `T=5→6` step to exceed its own cell
   spread before declaring the dead zone present or absent; decide the
   `analyse()` `KeyError` question in the card rather than delegating it.
5. **Fix criterion 9** (§1.5): name `kernel-regimes.json` as the sole permitted
   regeneration and change the git test to one that actually detects overwrites.
6. **Rewrite Defect site 5(b)** against the tree (§1.4d) — the constants it
   critiques do not exist outside a withdrawn card.
7. **Add to *Handed forward*:** the live 128K A/B is the next card and it is now
   the *blocking* one. `ramp_ab_bench.py` already emits the acceptance counters;
   what is missing is a long-context corpus. Every conclusion in this thread is
   conditional on a number no microbenchmark can produce, and three rounds of
   modelling have each been corrected by the next.

---

## 4. Standing note for the controller

Rounds 1, 2 and 3 each produced a confident mechanism claim, and each was refuted
by the next reader using **data already committed at the time the claim was
made**. In all three cases the refutation cost one afternoon of arithmetic on
files that were already in the repository.

The common failure is not carelessness — every round's prose is careful. It is
that each round accepted the previous round's *derived* quantity
(`one_full_kv_read_ms`, `504 GB/s`, `6.13×`) as a measurement and re-derived only
the arithmetic on top of it. FAILRUN §1 Link 1's *"Re-derived here and confirmed
exactly"* is literally true and was not enough.

Proposed rule, for `RULES.md` or this project's CLAUDE.md: **a derived scalar
inherited from an upstream document is an assumption until the current reader
re-fits it from the raw cells.** Reproducing the division is not re-derivation.
Name the denominator and check it against a second estimator, or state it as
inherited and unverified.
