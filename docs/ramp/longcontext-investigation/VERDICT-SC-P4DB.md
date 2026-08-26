# VERDICT — SC-P4DB (RAMP proposal floor + live long-context gate)

**Role:** Strong Card BREAKER (JUDGE-PROTOCOL, `coder ≠ grader ≠ breaker`)
**Card:** `SC-P4DB-ramp-proposal-floor-and-live-longcontext-gate.md` (frozen, undispatched)
**Commit reviewed:** `74e24d0` · worktree `phase-0-bench-harness`, branch `worktree-phase-0-bench-harness`, clean
**POC reviewed:** `POC-FINDINGS.md`, `docs/decisions/007-ramp-long-context-block-length.md`
**Predecessor reviewed critically:** `docs/reviews/2026-08-26-ramp/VERDICT-SC-P4DA.md` §1.1
**Date:** 2026-08-26

---

## VERDICT: **ACCEPT-WITH-EDITS**

The load-bearing empirical discovery is **real and survives every attack I could
mount**. There is a genuine kernel-selection regime change at `T=6` in mlx
0.32.0: it reproduces at three contexts, in two independent 128K sweeps, and — a
corroboration the writer did not notice and did not need — in the *dense
quantized-matmul* proxy at the same `T`, where no KV cache exists at all. It is a
100%-magnitude step against 3–13% cell spread. This is not curve-fitting. My
predecessor's roofline had no term for it and was, on that point, wrong.

I also independently re-ran the projection under the coefficient the writer
*didn't* sweep (`b`, the live per-row slope, whose committed cross-case spread is
**67.4%**) and the ranking held across its entire 2× range: block-64 beats the
stock ladder by +31.9% to +48.7% at 128K and +36.5% to +52.8% at 256K. **The
card's central conclusion is more robust than its own evidence section argues.**

It is **not ACCEPT** because:

1. The headline scalar (**56.8 / 46.7 TFLOPS marginal**) is **chord-selected**.
   The very JSON it comes from also computes 30.0 and 27.4 for a different chord,
   and neither number appears in the POC, the card, or the committed decision
   record. Honest full-range estimators put 256K **at or below** my predecessor's
   30-TFLOPS kill line (§1.1).
2. The stated *mechanism* ("the tiled kernel reads the KV once") is **refuted by
   the writer's own numbers by a factor of 6.13×** — and the correct mechanism
   inverts a forward-binding claim already committed in decision 007 about
   Phase 6 KV quantization (§1.2).
3. **The prescribed Fix is not implementable at the seam it names**, points the
   coder at a mechanism that cannot satisfy the card's own load-bearing
   requirement, and selects the one action the evidence does *not* license while
   explicitly forbidding the one it does (§1.3, §1.4 — **strongest attack**).
4. The `+26.1%` holdout error's direction is claimed to be "against RAMP." **The
   sensitivity sweep printed twelve lines below that claim proves the opposite**
   (§1.5).
5. Three hygiene claims in POC §5 are **false against the repository** (§1.6).
6. The card's declared baseline commit **does not contain the evidence key its
   own acceptance criterion 3 requires a test to read** (§1.7).

Every one of these is fixable by editing the card and the two committed
documents. No new POC work is required except a decision the writer must make in
prose (§1.4). Hence ACCEPT-WITH-EDITS, not REJECT.

**`make lint && make test` verified green at `74e24d0`:** ruff check + format
clean over 50 files; `102 passed, 1 skipped`. No spurious first-fail lock this
time (contrast VERDICT-SC-P4DA §6).

---

## 1. Attacks that LANDED

### 1.3 + 1.4 are the strongest; they are stated at §1.3–1.4 rather than first because they depend on the mechanism correction in §1.2.

---

### 1.1 — The headline marginal TFLOPS is chord-selected, and the unquoted chords sit on the kill line (STRONG)

`ramp_kernel_regimes.py:65` computes **four** marginal chords per context and
commits all four. POC-FINDINGS §3, the card's *Source* table, and decision 007
all quote exactly one of them — `T=8→64`, the most favourable — and none of the
three documents mentions that the others exist.

Re-derived by me from the same committed cells (lowest-spread-per-cell
selection, matching the writer's own rule):

| chord | 32K | 128K | 256K |
|---|---|---|---|
| **8 → 64 (the one quoted)** | **55.7** | **56.8** | **46.7** |
| 12 → 48 (committed, unquoted) | 53.5 | 51.9 | 40.2 |
| 6 → 96 (full tiled regime) | 40.0 | 38.7 | **33.8** |
| **48 → 96 (committed, unquoted)** | 33.1 | **30.0** | **27.4** |
| 64 → 96 | 26.3 | **23.7** | **22.2** |
| OLS over all `T ≥ 6` (R² 0.94–0.96) | 43.2 | 42.1 | 36.6 |

The marginal cost per verify row is **not** "nearly flat in T" — that phrase is
true only over `T = 6→12`, the six-cell window the writer measured the tiled
slope on (`ramp_kernel_regimes.py:38`). Over the tiled regime as a whole it rises
monotonically: 0.27 ms/row at `T=6→12`, 1.72 ms/row at `T=48→96`, 2.17 ms/row at
`T=64→96` (128K). POC §2's table reports only the `T=6→12` column and calls the
kernel "nearly flat in T" on that basis.

**Why this matters and not merely a presentation quibble.** The card's own
projection names **block-64 (`T=65`) the winner at both 128K and 256K**, and the
sweep includes block-96. The relevant marginal for a decision at that operating
point is the local slope *there* — 23.7 TFLOPS at 128K, 22.2 at 256K — which is
**below** the 30-TFLOPS figure my predecessor named as fatal, not "in the upper
half of the 30–60 band." At 256K, every estimator in the table except the quoted
chord lands at or under 30.

**It does not kill the card**, and I checked rather than assumed: the projection
in `ramp_longcontext_model.py` does not use any flat TFLOPS scalar — `t_pass`
interpolates the *measured* attention curve directly (`:280`), so the lumpiness
is already inside it. Block-64 wins on the measured curve. But the narrative laid
over that projection — "56.8, the upper half, RAMP comfortably survives" — is
built on a chord the writer chose from four, and the three documents that carry
that narrative forward are the ones a future reader will act on.

**Blocking edit A.** POC §3, the card's *Source* table, and decision 007 must
publish the full chord table above, state that the marginal is regime-dependent
and rising, and replace "in the upper half of the verdict's 30–60 band" with the
honest statement: *the marginal cost of a verify row is 42 TFLOPS-equivalent
averaged over the tiled regime at 128K and 37 at 256K, falling to ~24/~22 at the
block-64→96 margin; the projection does not depend on any single value because it
uses the measured curve.*

---

### 1.2 — "The tiled kernel reads the KV once" is refuted by the writer's own data by 6.13×, and the correct mechanism inverts a committed Phase 6 decision (STRONG)

POC §2, `ramp_kernel_regimes.py:6-7`, the card's Defect site 1, and decision 007
all state the tiled kernel "reads the KV once and is then nearly flat in T."

If that were true, the tiled kernel's fixed cost would be close to one measured
KV read. It is not:

| context | one full KV read (`T=1`, measured) | OLS intercept over `T ≥ 6` | ratio |
|---|---|---|---|
| 32K | 4.55 ms | 23.5 ms | **5.16×** |
| 128K | 17.05 ms | 104.6 ms | **6.13×** |
| 256K | 34.44 ms | 210.4 ms | **6.11×** |

`config.json`: `num_attention_heads = 24`, `num_key_value_heads = 4`.
**24 / 4 = 6.** The intercept is one KV read per *query head group* — the tiled
kernel does not exploit GQA head sharing; the vector kernel does. The fit is
quantitatively exact at both long contexts and slightly under at 32K (where the
fixed cost is smaller relative to launch overhead).

Cross-check on the same data: `T=1` at 128K reads 8.59 GB in 17.05 ms =
**504 GB/s**. 6 × 8.59 GB / 104.6 ms = **493 GB/s**. Self-consistent.

Two consequences, both of which the card must carry:

**(a) Decision 007 item 5 is backwards.** It states, as a committed decision:
> *"Phase 6 KV quantization is not the long-context rescue path … KV quant
> reduces bytes, which moves the aggregate cost; the marginal cost of an added
> verify row is compute, and would barely move."*

On the corrected mechanism, **104.6 of the 121 ms that a `T=12` verify pass costs
at 128K is KV bytes** — six times over. Q8 KV would cut roughly half of the
dominant term of every verify pass at long context. Decision 007 is a
forward-binding record that would steer Phase 6 away from the single largest
measured cost in this data. This is a Principle 1 (grand-ensemble) failure, not a
footnote.

**(b) The plan's bandwidth constant is contradicted and nobody noticed.**
CLAUDE.md rule 9 and the plan fix effective DRAM bandwidth at **~300 GB/s**. This
POC measures a 504 GB/s KV read directly, on the target machine, and neither the
POC nor the card remarks on it. Every roofline in this project — including my
predecessor's §1.1, which started this whole thread — is built on the 300 GB/s
number. Either it is stale or this cell is; it must be reconciled, not left as
two committed constants that disagree by 1.7×.

**Blocking edit B.** Correct the mechanism in POC §2, `ramp_kernel_regimes.py`'s
docstring, the card's Defect site 1, the proposed `_MIN_PROPOSAL_BLOCK` code
comment, and decision 007 items 4 and 5. Add the 300 vs 504 GB/s discrepancy to
the card's handed-forward items with a pointer to CLAUDE.md rule 9.

*(In fairness to the writer: the *regime change* is unaffected by this. Only the
label on it, and what it implies for Phase 6, are wrong.)*

---

### 1.3 — The Fix's chosen action is the one the evidence does not license, and the action it does license is explicitly forbidden (STRONGEST)

Fix step 2, second bullet:
> *"When the resolved block would fall in `1 .. _MIN_PROPOSAL_BLOCK - 1` and
> context ≥ `_DEAD_ZONE_MIN_CONTEXT`, **propose nothing** … rather than rounding
> up. Rounding up changes which tokens are proposed and therefore can change
> acceptance; declining to propose cannot. Declining is the conservative choice
> and the one this POC's evidence supports."*

**The last sentence is false, and the reasoning in the sentence before it is
self-refuting.**

The POC's model-free result is a *dominance* argument, and it has exactly one
form: block 8 and block 11 **both pay the tiled kernel's fixed cost**, so block
11 buys ~38% more proposed tokens for 1.8% more time. That argument compares two
actions on the same side of the kernel switch. It says nothing whatsoever about
declining.

Declining is on the **other** side of the switch, and the cost gap is enormous:

| action at 128K | rows | attention/pass | measured or predicted full pass |
|---|---|---|---|
| propose nothing (MTP fallback) | `T=3` | 37.58 ms | **123.2 ms** (measured, held-out cell) |
| block 8 | `T=9` | ~119.4 ms | ~252 ms (model) |
| block 11 | `T=12` | 121.06 ms | ~260 ms (model) |

Declining is **~2.1× cheaper per pass** but yields ~2.4 tokens instead of up to
9. Whether it wins depends entirely on **acceptance at long context** — the one
input the card's own Non-goals and POC §4 say is unmeasured and must not be
assumed. So the card's production change rests on precisely the assumption the
card forbids.

And the stated rationale inverts: "rounding up changes which tokens are proposed
and therefore can change acceptance; declining cannot." Declining changes the
proposal from 8 tokens to **zero tokens**. That is a strictly larger perturbation
of acceptance behaviour than 8 → 11. The rationale argues for the option the card
rejects.

POC §3 compounds this by phrasing the finding as *"propose ≥ 11 tokens **or
propose nothing**"* — treating the two as interchangeable. On this data they
differ by 137 ms per pass at 128K.

**Blocking edit C.** Either (i) change Fix step 2 to **raise to the floor**,
which is what the dominance argument licenses, and delete the "declining is
conservative" rationale; or (ii) keep declining and state plainly that it is an
acceptance-dependent bet the POC does not settle, move it behind the live A/B of
Fix step 5, and remove it from the set of changes the card claims are licensed on
model-free evidence. I recommend (i): the card's own §"What this does and does
not license" says the floor "is the one production change this POC does license
on its own evidence" — that sentence is true of raise-to-floor and false of
decline.

---

### 1.4 — The named seam cannot express "propose nothing," and the prescribed context source cannot satisfy the card's own requirement (STRONGEST, blocking)

Two independent implementation defects, both of which force a coder to guess.

**(a) `block_for_ext` has no no-proposal return channel.** The seam is
`cc.block_for_ext`, rebound at `rafale/draft/ramp.py:378-381` as
`_installed_block_for_ext(ext, k_cap) -> int`. It returns an **integer block
length**. By the time it is called, `NgramIndex.find` has already returned a
match position; the decision to propose has been taken upstream. There is no
sentinel documented anywhere in the module for "do not propose," and the
surrounding code shows the engine floors block lengths at 4 in two places
(`:76` `min(..., max(4, k_cap))`, `:390` `max(4, block)`), which is evidence that
0 or negative is *not* a value the engine expects. The card's instruction —
*"return the engine's no-proposal path"* — names a path that does not exist at
this seam. The real no-proposal channel is `NgramIndex.find` returning
`(None, -1)` (`:369`, `:375`), a **different function** the card's Fix does not
mention and its Touch List does not anticipate.

**(b) The prescribed `context_hint` is static and therefore wrong.** The card
makes "the floor is conditional on context, not unconditional" load-bearing —
it is Defect site 1's *"Pre-empt the obvious wrong fix"*, it is Gherkin scenario
2, and it is a reviewer-checklist item. It then directs the coder:
> *"…or, if that is not reachable without an engine change, from an explicit
> `context_hint` the launcher sets. **If genuinely ambiguous, implement the
> `context_hint` version**…"*

A launcher-set hint is a **constant for the whole process**. Context is not: an
agent session grows from a few hundred tokens to 128K. A static hint of 131072
applies the floor from the first token of the session — which is exactly the
unconditional clamp the card names as the wrong fix, arrived at by following the
card's own instruction. The Gherkin's two scenarios (131072 vs 800 tokens) would
be satisfiable only by two different launches, so the acceptance criteria cannot
catch it either.

**And the dynamic value is already in scope.** `_InstalledRampIndex.sync`
(`:361-365`) stores `self._ramp_corpus = list(history)`; `find` (`:367`) receives
`history` on every probe. `len(history)` **is** the current context length, it is
inside the same closure as `_installed_block_for_ext`, and reading it requires no
engine change and no engine-internals reach. The card's escape hatch (INVALID_CARD
if context is unreachable) is not needed, and the fallback it points at is worse
than the solution sitting three lines away.

**Blocking edit D.** Rewrite Fix step 2 to (i) name `len(history)`, captured in
`sync`/`find` into the closure, as the context source, deleting the `context_hint`
fallback and the associated INVALID_CARD route; and (ii) specify the exact
mechanism for whichever action blocking edit C selects — a concrete integer for
raise-to-floor, or an explicit `find`-side decline (with `NgramIndex.find` added
to the Touch List's implied surface) if declining is kept.

---

### 1.5 — The `+26.1%` holdout error's direction is claimed to favour caution; the sweep printed in the same artifact proves it favours RAMP

Claimed in three places — POC §4 *Limits* (*"The error runs against RAMP —
over-charging attention penalises long blocks — so the projection is
conservative"*), `ramp_longcontext_model.py:22-23` and `:288-291`, and decision
007 item 2 (*"conservative in both known directions"*).

The claim is testable against the artifact it sits next to.
`sensitivity.attn_scale_sweep` in `longcontext-projection.json` scales the
attention delta and reports the winner's margin over the stock ladder:

| attention delta × | 128K, block-64 vs stock | 256K, block-64 vs stock |
|---|---|---|
| 0.5 | +40.3% | +44.5% |
| 1.0 | +44.9% | +49.8% |
| 2.0 | **+50.6%** | **+54.9%** |

**More attention cost ⇒ a larger RAMP win, monotonically, at both contexts.** The
algebra is elementary: in the tiled regime the attention term is near-constant in
`T`, so inflating it adds roughly the same Δ to both arms' pass time, and
`(tok_L/(t+Δ)) / (tok_S/(t+Δ))` rises toward `tok_L/tok_S ≈ 22.9/8` as Δ grows.
So a model that **over-charges** attention **over-states** RAMP's advantage. The
conservative direction is ×0.5, not ×1.0.

Two further points the writer owes the reader:

- The holdout cell is at **`T=3`** — inside the *vector* regime, on the far side
  of the kernel switch from every operating point RAMP's long blocks use. The
  writer says so (`holdout.note`: *"constrains the T=3 end of the curve at 128K
  and nothing else"*) and then treats the +26.1% as if it characterised the model
  generally. The tiled-regime term is validated by **nothing**.
- "Robust to a 4× sweep" is a hollow robustness claim. The parameter swept is the
  one whose effect on the ranking is monotone and favourable; it **cannot** flip
  the answer, by construction.

**Landed but not fatal — I ran the sweep the writer should have.** The
coefficient that *does* oppose RAMP is `b`, the live per-row slope, whose
committed cross-case spread is **67.4%** (0.00168 → 0.00326 s/row,
`longcontext-projection.json` `fit.b_spread`). Substituting each per-case value:

| `b` (ms/row) | 128K stock | 128K block-64 | 256K block-64 | winner |
|---|---|---|---|---|
| 3.26 (add-method) | 55.0 | 77.0 (**+39.9%**) | +45.9% | block-64 |
| 2.34 (mean, shipped) | 57.2 | 82.9 (+44.9%) | +49.8% | block-64 |
| 1.68 (rename-identifier) | 60.0 | 89.3 (+48.7%) | +52.8% | block-64 |

**The ranking holds across the full 2× range.** This attack does not land against
the conclusion. It lands against the *reporting*: POC §4 gives quality statistics
for both **rejected** models (R²=0.648, +59%; α=11.6, 266% spread) and **none for
the accepted one**, whose worst statistic — a 67% spread on its only
`T`-dependent coefficient — is computed, printed by the script, committed to
JSON, and omitted from every narrative table. That asymmetry is the finding.

For the record, since the dispatch brief asked: the accepted model **has no R²**
and cannot have one as structured. It is two parameters solved exactly from two
numbers per case over three cases (`fit_live_short_context:226-249`); per-case
residuals are zero by construction. Cross-case spread is its only quality signal.
The card and POC never claim otherwise, so this is not a false claim — but a
reader comparing "R²=0.648, rejected" against a silently unquantified survivor
will draw the wrong inference.

**Blocking edit E.** Correct the error-direction claim in all three places
(including committed decision 007). Report `b_spread = 67.4%` in POC §4 *Limits*
alongside the rejected models' statistics, and report the per-case `b` sweep above
as the robustness check that actually tests the conclusion. State that the holdout
validates the vector regime only.

---

### 1.6 — Three hygiene claims in POC §5 are false against the repository

POC §5 is the section the card leans on for credibility, and decision 007 quotes
it as a lesson for the harness generally. Checked against `git log --all
--diff-filter=A` over `evidence/`:

| claim | status |
|---|---|
| *"CLAUDE.md rule 4 earned its keep twice, and **both artifacts are committed**"* | **False.** Eleven files have ever existed in `evidence/`. **None** is an invalidated run. Neither the swap-storm sweep nor the 9.6 TB/s dense run was ever committed. |
| *"Both sides of swap and pagein counters are **now written into every artifact**"* | **False.** `environment_after`/`hygiene` exist only in the two `c131072` files. `c1024`, `c32768`, and `c262144` have no post-run counters at all. |
| *"Re-run **after swap settled**"* | **Unsupported.** Swap never settled. `vm.swapusage` reads 8.53–8.59 GB used of a 9216 MB backing store — **93% full, ~630 MB free** — in *every* committed file, clean ones included. Whatever changed, it was not swap pressure abating. |

The consequence is that **the swap-storm diagnosis cannot be checked by anyone**,
including me. The diagnostic the writer credits — *"256K cells measured faster
than 128K cells at the same T, which is physically impossible"* — is a good
diagnostic and I would like to have confirmed it; the artifact needed to do so
does not exist. This directly contravenes the project's audit-trace posture (keep
the invalidated artifact, never overwrite) which the writer explicitly invokes.

**Did a similar contamination survive into the clean runs?** I checked as far as
the committed data allows:

- **Cross-context linearity holds.** 256K/128K time ratio at fixed `T`: 2.02 at
  `T=1`, 2.04 at `T=6`, 2.13 at `T=48`. The physical-consistency test the writer
  invented passes on the final data. Good.
- **One cell is visibly contaminated and was kept.** `c262144`, `T=40`:
  `t_max = 591 ms` against `t_min = 292 ms`, `rel_spread = 1.006` on 7 reps. It
  passes the writer's own committed rule (median/min = 1.02, under 15%) only
  because the rule is median-based. It is not used in any quoted quantity, but it
  is a 2× stall inside the 256K sweep — and that sweep has **no post-run pagein
  counter** to say whether anything else drifted with it.
- **`c262144` runs 7 reps.** CLAUDE.md rule 4 requires ≥ 5, so it is compliant,
  but 7 reps at 93% swap occupancy with a visible 2× outlier is thin for a cell
  that anchors the 256K half of the entire conclusion.
- **`dense-microbench.json` is badly contaminated and is quoted anyway.**
  `rel_spread`: **7.17 at T=12, 4.47 at T=6, 2.87 at T=40, 2.75 at T=96** — 287%
  to 717%. POC §4's "dense proxy is nearly flat from T=12 to T=64 (98.2 → 102.7,
  +4.5 ms)" compares a 717%-spread cell to a 5%-spread cell and reads a 4.5 ms
  difference out of it. That comparison is the entire basis of the *"roughly
  1 ms/row is neither weight streaming nor full attention — almost certainly the
  48 GatedDeltaNet layers"* finding, which POC §4 presents as a discovery and
  uses to justify the accepted model's structure. **It is noise-dominated.** I
  verified the dense numbers do not enter `t_pass` (`:280` uses `attn` only), so
  the projection is unaffected — but the inference drawn from them is not
  supported, and `kernel_time()` (`:114`) which would have used them is dead code.

**Blocking edit F.** Correct or delete the three false statements in POC §5.
Commit the invalidated artifacts, or state plainly that they were not retained
and that the diagnoses are therefore unverifiable. Add `environment_after` to the
three sweeps that lack it (a re-run, or an explicit note that they predate the
hygiene block). Downgrade the GatedDeltaNet inference in POC §4 to a hypothesis
and label `dense-microbench.json`'s contaminated cells in the artifact itself.

---

### 1.7 — The card's declared baseline commit does not contain the evidence its own acceptance criterion requires

Card header: *"Baseline commit: `ca6e739`."* Also *"branch
`claude/plan-review-upgrade-9epmpr`"* — the worktree is actually on
`worktree-phase-0-bench-harness`.

At `ca6e739`, `evidence/kernel-regimes.json` has **no `min_proposal_block` key
and no `cell_sources` key**. Both were added at `74e24d0`. So:

- Fix step 1's citation (*"`min_proposal_block`: 9 @32K, 11 @128K, 10 @256K"*)
  and Defect site 1's provenance note about `cell_sources` describe a file that
  does not exist at the declared baseline.
- **Acceptance criterion 3** requires a test that *"must read
  `evidence/kernel-regimes.json` and compare"* `_MIN_PROPOSAL_BLOCK` against
  `max(min_proposal_block)`. A coder starting from `ca6e739` gets a `KeyError`.

Worse, the constant **moved between the two commits**. At `ca6e739` the 128K
`min_proposal_block` was **10**; at `74e24d0` it is **11**. The cause is the
lowest-spread-per-cell selection rule picking up the `-lowT-confirm` file, which
shifted the 128K vector slope from 11.147 to 10.243 ms/row (8%) and the crossover
from 10.28 to 11.13. **A single re-measurement moved the production floor
constant by one token.** The card presents `_MIN_PROPOSAL_BLOCK = 11` as
"measured, not chosen"; it is measured *and* fragile at the ±1 level that the
whole floor operates on.

**Blocking edit G.** Correct the baseline commit to `74e24d0` and the branch name.
Note in Fix step 1 that the constant moved 10 → 11 within this POC and state the
tolerance the test should enforce (exact equality will break on any re-measurement
of a `T ∈ {1,5,6,12}` cell, which is arguably the intent — but the card must say
so deliberately rather than by accident).

---

### 1.8 — `_DEAD_ZONE_MIN_CONTEXT = 32768` is chosen, not measured, and the card labels it measured

Fix step 1 presents both constants under *"Measured, not chosen"*, and comments
`_DEAD_ZONE_MIN_CONTEXT = 32768  # shortest context where the penalty is measured`.

That comment is honest about what it is and dishonest about what it does. The
grid is `{1024, 32768, 131072, 262144}`. The penalty is **absent at 1K** (`T=5` =
1.095 ms, `T=6` = **0.985 ms** — the tiled kernel is immediately *cheaper*) and
**present at 32K** (+8.2 ms worst penalty on a ~26 ms pass — a **31%** overhead,
not a rounding error). The crossover therefore lies somewhere in `(1K, 32K)`, a
**32× range that was never sampled**, and the shipped threshold sits at the top of
it. Every context from ~2K to 32K gets no floor, and the dead zone is plausibly
real across much of that band.

This is not academic for a 128K-target coding agent: sessions *pass through*
2K–32K on the way up, and the plan's own harness is append-only, so every session
spends its early life in the unmeasured band.

Compounding it: `ramp_kernel_regimes.py:79` loops over `(32768, 131072, 262144)`
only. **The 1K control — the single cell proving the floor must be conditional —
is not in `kernel-regimes.json` at all.** Acceptance criterion 3's test reads only
that file, so the test can verify the floor's *value* but nothing can verify its
*condition*. Gherkin scenario 2 asserts the 800-token behaviour against no
committed evidence in the artifact the card designates.

**Blocking edit H.** Relabel `_DEAD_ZONE_MIN_CONTEXT` as a **chosen, conservative
threshold pending measurement**, not a measured constant. Add `4096` and `8192`
(or at minimum `8192`) to the microbench grid and to `ramp_kernel_regimes.py`'s
context loop as a handed-forward item, and add `1024` to the regimes output so the
"no dead zone at short context" control is committed where the test can read it.

---

### 1.9 — The `~30%` acceptance headroom measures a degradation mode that is not the likely one

Re-derived from `sensitivity()` (`:319-340`):
`breakeven = stock_tok_s × mean_pass(variant)`;
`headroom = 1 − breakeven / tokens_per_pass`.
Confirmed: block-48 fuzzy at 128K → break-even 16.08 vs 22.90 measured →
**29.8%**; block-64 → 30.97%; 256K → 30.97% / 33.24%. The arithmetic is right.

Three problems with what it means:

1. **It degrades RAMP's acceptance while holding the stock ladder's fixed.**
   Both arms' acceptance statistics come from the same ~800-token traces and are
   *equally* unmeasured at 128K. If both degrade by the same factor, the ranking
   is **exactly invariant** (`tok/s` scales by the same factor in both numerators).
   So the quantity being reported is headroom against *differential* degradation —
   long blocks degrading while short ones do not. That is a plausible failure mode
   (longer contiguous copy runs are harder to match), and it is the right thing to
   bound; but the card and decision 007 both state it as *"acceptance degrading
   more than ~30% at long context"*, which is the uniform mode, and the uniform
   mode cannot reverse anything.
2. **Two distinct quantities are conflated.** The computed headroom is in
   *tokens per pass*; decision 007's *"What would reverse this"* restates it as
   *acceptance* degrading. They are related but not equal, and the card cites the
   number as a safety margin without saying which.
3. **Hit rate is frozen too.** `variant_shapes` (`:379-397`) reuses
   `verify_passes`, `rounds`, and `drafted` unchanged. The copy-pass *fraction* at
   128K is as unmeasured as acceptance and moves the answer independently — likely
   in RAMP's favour (more history to match against), which the card does not claim
   and should not, but which shows the single-scalar headroom is under-specified.

**Blocking edit I.** Restate the headroom in both documents as *"~30% of
block-48's tokens-per-pass, against a stock ladder whose acceptance is held
fixed — i.e. headroom against **differential** degradation; uniform degradation
cannot reverse the ranking."* Add hit-rate drift to the live A/B's reporting list
in Fix step 5.

---

### 1.10 — The floor constant is not reproducible from the committed driver

`run_attn_flops_microbench.sh` sweeps `1024 32768 131072 262144` with `--warmup
3/6` and the script's default reps. It does **not** produce
`attn-microbench-c131072-lowT-confirm.json`, which has 15 reps and a truncated
`T` grid (`1,2,3,4,5,6,8,12`). No committed script or shell record produces that
file.

Because `ramp_kernel_regimes.py`'s lowest-spread rule picks `T ∈ {1,2,3,4,5,8}`
from the confirm file (`cell_sources`), and because those cells set the vector
slope, **a skeptical reader who reruns the committed driver gets
`min_proposal_block = 10` at 128K, not 11** (§1.7). The card's headline production
constant is not reproducible from the committed harness.

Other reproducibility notes, none blocking on their own: the driver hardcodes
`/opt/homebrew/var/mtplx/venv-2.7.1/bin/python` and the card's *Worktree
provisioning* section documents that path only for the Fix-step-5 A/B, not for the
microbench; and `analyse()` hardcodes `rows[1]`, `rows[5]`, `rows[6]`, `rows[12]`,
so any grid change silently `KeyError`s rather than degrading.

**Blocking edit J.** Add the confirmation run to `run_attn_flops_microbench.sh`
(or commit the exact invocation), so `kernel-regimes.json` regenerates from a
single documented command.

---

### 1.11 — Fix step 6 and acceptance criterion 9 instruct the coder to write a file that is already committed

Fix step 6: *"Record the gate outcome in
`docs/decisions/007-ramp-long-context-block-length.md`."* Touch List marks it
`(new)`. Acceptance criterion 9: *"`docs/decisions/007-*.md` written."*

**It already exists**, committed at `74e24d0`, with the outcome recorded as
**NOT TRIPPED**. The coder will find the deliverable done and either skip it or
overwrite it. Either way the card is stale against its own repository.

This also matters for §2.1 below: the gate was recorded *before* the card that
gathers the evidence for it was dispatched.

**Blocking edit K.** Change Fix step 6 and acceptance criterion 9 to *amend* 007
with the live A/B result, mark it in the Touch List as an amendment, and reword
007's outcome per §2.1.

---

## 2. Attacks that did NOT land (stated, per JUDGE-PROTOCOL)

### 2.1 — Is this a negative result spun positive? Mostly no — but the gate record is premature

The brief asked whether SC-P4DA's kill-criterion bullet 4 (*"the long-context
optimum collapses to the stock ladder"*) genuinely clears. I attacked this hard
and it holds on the evidence available:

- The projection uses the **measured** attention curve, not a flat scalar, so
  §1.1's chord problem does not propagate into it.
- Block-64 wins at 128K and 256K across a 4× attention sweep **and** across the
  full 2× range of `b`, the coefficient the writer did not sweep (§1.5). Eight
  independent parameterisations, one winner.
- The optimum does **not** collapse to the stock ladder in any of them; the stock
  ladder ranks 13th of 16 in every cell of the sensitivity sweep.
- The mechanism is coherent and now better-founded than the writer's version: at
  long context the tiled kernel carries a ~105 ms (128K) / ~210 ms (256K) fixed
  cost per pass, so *fewer, longer passes* is straightforwardly the right shape.
  The stock ladder pays that fixed cost more often. The claim that RAMP's
  advantage *grows* with context follows directly.

**What does not hold is recording the gate as a binary "NOT TRIPPED" now.**
Decision 007's headline is flat; the qualifier ("every number is projected") is
body item 2. Meanwhile the card's own acceptance criteria 7–8 require a live 128K
A/B, criterion 8 explicitly permits it to contradict the projection, and CLAUDE.md
rule 4 and the phase Definition-of-Done both require hygiene-protocol benchmark
cells before a gate is recorded. **The decision record was committed before the
measurement its own card mandates.**

**Blocking edit L.** Reword decision 007's `**Outcome:**` line to
*"NOT TRIPPED on projected cost evidence; confirmation pending the live 128K A/B
(SC-P4DB acceptance 7–8)."* The body already says this; the headline must not
outrun it.

### 2.2 — Is the `T=6` regime change a curve-fitting artifact, noise, or a cache effect? No.

I tried to break it four ways and could not:

- **Magnitude vs noise.** 128K: `T=5` = 58.03 ms → `T=6` = **117.43 ms**. A
  +102% step against per-cell spreads of 2.9% and 13.0%. 256K: 121.38 → 239.86
  (+98%). 32K: 15.37 → 26.25 (+71%). Not noise.
- **Independent replication.** Two separately-launched 128K processes, ~30 min
  apart, agree on `T=6` to 1.7% (117.43 vs 119.43) and on `T=5` to 6.3%.
- **Non-monotonicity rules out drift.** `T=7` is *cheaper* than `T=6` at 32K
  (25.33 vs 26.25) and at 128K (117.40 vs 119.43). Thermal or swap drift is
  monotone in wall-clock; this is a step function in `T`, and the sweep runs `T`
  ascending.
- **Cache effects are excluded by construction.** `bench_attention` allocates
  **16 distinct KV buffers** per context (`:157-164`) precisely so later layers
  cannot hit a warm cache, and calls `mx.clear_cache()` between contexts (`:210`).
  KV is 8.6 GB at 128K — far beyond any on-chip cache.
- **The `mask="causal"` geometry is correct.** If MLX aligned the causal mask
  top-left, `T=1` would attend to one key; it measures 8.59 GB at 504 GB/s, so
  the mask is bottom-right aligned as decoding requires. The benchmark measures
  what it claims to.
- **Corroborated by an unrelated kernel family.** The *dense* proxy —
  `mx.quantized_matmul`, **no KV cache anywhere** — steps at the same `T`:
  50.25 ms at `T=5` → 83.19 ms at `T=7` (`rel_spread` 4.6%). This is strong
  evidence for a general MLX shape-dispatch threshold at `T=6` in 0.32.0, and it
  is a corroboration the writer measured but did not notice.

I could not read MLX 0.32.0's source in this environment, so the **kernel names**
("vector path" / "tiled path") remain the writer's inference. §1.2 shows the
attached *mechanism* is wrong. **The regime change itself is solid, and my
predecessor's model genuinely lacked a term for it.** On this point the writer is
right and the previous verdict was wrong.

### 2.3 — Does the dense proxy's contamination corrupt the projection? No.

I traced it: `t_pass` (`:280-281`) uses `attn` only; `kernel_time()` (`:114-117`),
the sole consumer of `dense`, is dead code; `_arm_summary`'s `dense_s_per_pass`
(`:158-162`) is computed and never read by `fit_live_short_context`. The writer's
claim that *"the dense proxy contributes nothing to the projection"* is **true**
and I verified it rather than accepting it. Only the *inference drawn from* the
dense numbers is unsupported (§1.6).

### 2.4 — Does the 9.6 TB/s lazy-eval diagnosis hold? Yes, in code; no, in evidence.

The fix is visible and correct at `attn_flops_microbench.py:251-269`: every
`quantized_matmul` result is appended to `outs` and the list is returned, so
`mx.eval` forces all of them. The failure mode described (rebinding one name makes
all but the last dead under MLX's lazy evaluation) is real MLX behaviour and the
comment at `:252-255` documents it properly. The **artifact** showing 9.6 TB/s was
never committed (§1.6), so the diagnosis is credible on the code and unverifiable
on the evidence.

### 2.5 — Does the fix break the shipped SC-P4DA code? I found no interaction hazard.

Checked against `rafale/draft/ramp.py` at `74e24d0` and the tests:
`test_ramp_is_off_by_default` is at `:388` and
`test_ramp_install_disabled_never_imports_mtplx` at `:402` — both citations exact.
`tests/test_ramp.py` has **13** `def test_` — exact. `tests/test_engine_seam.py`
asserts `_BLOCK_LADDER == (8, 12, 16, 24, 32)` (`:156`) and the `block_for_ext`
signature (`:148`), so Fix step 4's "leave `block_for_ext` byte-faithful, apply
the floor in the installed wrapper only" is the correct seam and the scope fences
around it are right. The off-by-default property is correctly triple-anchored
(scope fence, acceptance criterion 4, Gherkin scenario 5) — the parent card's
§1.6 failure is not repeated. The `RampCounters` addition is additive and safe.
**The card's scope fences are its strongest section** and I could not find a
Touch-List file whose modification would silently break a shipped property.

---

## 3. Required edits before dispatch (all blocking)

| # | §  | Edit |
|---|---|---|
| A | 1.1 | Publish all four marginal chords + the OLS fit; drop "upper half of the 30–60 band"; state the marginal is regime-dependent and rising. Applies to POC §3, card *Source*, decision 007. |
| B | 1.2 | Correct "reads the KV once" → ~6× (GQA head-group amplification, 6.13× measured) in five places. **Reverse decision 007 item 5 on KV quantization.** Add the 300 vs 504 GB/s discrepancy against CLAUDE.md rule 9 to handed-forward items. |
| C | 1.3 | Choose: raise-to-floor (licensed by the dominance argument) **or** decline (an acceptance-dependent bet). If decline, move it behind the live A/B and delete the "declining is conservative" rationale. Fix POC §3's "≥ 11 or nothing" equivalence either way. |
| D | 1.4 | Name `len(history)` from the `sync`/`find` closure as the context source; delete the static `context_hint` fallback and its INVALID_CARD route. Specify the exact return mechanism for the action chosen in C. |
| E | 1.5 | Correct the holdout error direction in POC §4, `ramp_longcontext_model.py:22-23` and `:288-291`, and decision 007 item 2. Report `b_spread = 67.4%` and the per-case `b` sweep. State that the holdout validates the vector regime only. |
| F | 1.6 | Correct or delete the three false hygiene claims in POC §5. Commit the invalidated artifacts or state they were not retained. Add `environment_after` to the three sweeps lacking it. Downgrade the GatedDeltaNet inference to a hypothesis; label `dense-microbench.json`'s 287–717%-spread cells. |
| G | 1.7 | Baseline commit → `74e24d0`; correct the branch name. Note the 10 → 11 constant move and state the intended test tolerance. |
| H | 1.8 | Relabel `_DEAD_ZONE_MIN_CONTEXT` as chosen-pending-measurement. Add ≥ one context in (1K, 32K) to the microbench grid and `ramp_kernel_regimes.py`'s loop as a handed-forward item; add `1024` to the regimes output so the conditional-floor control is testable. |
| I | 1.9 | Restate the ~30% headroom as headroom against *differential* degradation with the stock arm held fixed; note the tokens-per-pass vs acceptance-rate conflation; add hit-rate drift to Fix step 5's reporting list. |
| J | 1.10 | Add the low-`T` confirmation run to `run_attn_flops_microbench.sh` so `kernel-regimes.json` regenerates from one documented command. |
| K | 1.11 | Decision 007 already exists — change Fix step 6 / acceptance criterion 9 to an amendment and update the Touch List. |
| L | 2.1 | Decision 007 `**Outcome:**` → *"NOT TRIPPED on projected cost evidence; confirmation pending the live 128K A/B."* |

**Non-blocking, recommended:** the lowest-spread-per-cell selection rule
(`ramp_kernel_regimes.py:88-95`, `ramp_longcontext_model.py:84-92`) is defended as
"a hygiene rule, not cherry-picking." It is still selection on an outcome
statistic, and it biases toward the luckiest sample; §1.7 shows it moved a
production constant by one token. Pooling reps across files, or reporting both
values with the delta, would be more defensible. Not blocking because the writer
records `cell_sources` and the effect is bounded and disclosed.

---

## 4. What the writer got right, and should keep

Stated because a breaker who only attacks is not calibrated.

1. **The `T=6` discovery is a genuine correction to my predecessor's verdict.**
   Building the microbenchmark at the engine's exact shapes rather than arguing
   from a roofline was the right call, and it found something no amount of further
   reasoning would have.
2. **The aggregate-vs-marginal distinction is the correct frame**, even though
   the specific chord was cherry-picked. My predecessor did apply a ratio where a
   derivative was needed.
3. **16 distinct KV buffers** (`:157-164`) with an explicit comment about why
   sharing one would be wrong. That single design choice is what makes §2.2's
   cache-effect attack fail.
4. **The `mask="causal"` geometry, `mx.eval` + `mx.synchronize` placement, and
   the lazy-eval fix** are all correct. The measurement harness is sound.
5. **The scope fences** are the best-written section of any card in this project.
   Six fences, each naming the specific evidence or test that makes it binding.
6. **Acceptance criterion 8** explicitly permits the live result to contradict the
   projection and forbids tuning it away. That is the criterion my predecessor's
   §1.2 complaint was reaching for, and it is correctly written here.
7. **The dominance argument for the floor is genuinely model-free and correct** —
   block 8 vs block 11 at 128K, 1.8% more time for ~38% more tokens. It is the
   real finding. §1.3 is not an attack on that argument; it is an attack on the
   card acting on a *different* one.
8. **The writer disbelieved a 9.6 TB/s result and found the bug.** That instinct
   is worth more than the number it discarded.

---

## 5. Verdict restated

**ACCEPT-WITH-EDITS.** Twelve blocking edits, all to documents; no new POC work
required except the writer's choice in edit C. The card's empirical core is sound
and correct my predecessor on a real point. Its production Fix, as currently
written, cannot be implemented at the seam it names and prescribes the action its
evidence does not support — **that pair (§1.3, §1.4) must be resolved before any
coder is dispatched**, or the worker will guess, and both guesses available to
them are wrong.

**Langfuse:** no worker trace exists — the card is undispatched, so there is no
dispatch window to query (JUDGE-PROTOCOL §1.3a requires this be stated rather
than silently skipped).

**Verified live at `74e24d0`:** `make lint` clean (ruff check + 50 files
formatted); `make test` → **102 passed, 1 skipped**. All `file:line` citations in
the card's Defect section re-read against the working tree and found accurate;
the only stale reference is the baseline commit itself (§1.7).
