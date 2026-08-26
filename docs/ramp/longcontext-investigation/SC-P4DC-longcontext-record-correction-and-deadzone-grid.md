# SC-P4DC — Correct the long-context record to the measured mechanism, make the floor constant reproducible, and fill the unmeasured dead-zone band

**Status:** frozen, undispatched
**Baseline commit:** `172bcc9` · worktree `phase-0-bench-harness`, branch `worktree-phase-0-bench-harness`, clean
**Supersedes:** `SC-P4DB-ramp-proposal-floor-and-live-longcontext-gate.md` (INVALID_CARD, production fix withdrawn)
**Answers:** `VERDICT-SC-P4DB.md` blocking edits A, B, E, F, H, I, J, K, L; `FAILRUN-SC-P4DB.md` §§1–6
**POC:** `POC-FINDINGS.md`, `evidence/`

---

## Source

SC-P4DB received ACCEPT-WITH-EDITS with twelve blocking edits. `FAILRUN-SC-P4DB.md`
ruled its production fix `INVALID_CARD` on four independent grounds, the decisive
one being that the proposed floor binds in **zero shipped configurations**
(FAILRUN §1, Link 4). This card carries every edit that survives that ruling.

**This card changes no code in `rafale/`.** It corrects committed records to the
measured mechanism, makes one production-candidate constant reproducible from the
committed harness, and measures the one band the POC never sampled. It is a
measurement-and-record card; there is nothing to turn on and nothing to turn off.

Blocking edits **C, D, G** are answered by withdrawal rather than by edit: C and D
prescribe how to implement a fix that is no longer being made, and G's baseline
error is moot against this card's declared baseline (`172bcc9` contains both
`min_proposal_block` and `cell_sources`; `ca6e739` contained neither).

Decision 007 was amended in the same commit that adds this card — i.e. it is
already correct in the tree the coder starts from — and is **not** in this card's
Touch List (see *Do NOT touch*).

---

## Defect / gap

### Defect site 1 — four committed artifacts state a mechanism their own data refutes by 6.13×

`POC-FINDINGS.md` §2, `scripts/ramp_kernel_regimes.py:6-7`, and (before its
amendment) decision 007 all state that the tiled kernel "reads the KV once and is
then nearly flat in `T`."

| context | one full KV read (`T=1`, measured) | OLS intercept over `T ≥ 6` | ratio |
|---|---|---|---|
| 32K | 4.55 ms | 23.5 ms | **5.16×** |
| 128K | 17.05 ms | 104.6 ms | **6.13×** |
| 256K | 34.44 ms | 210.4 ms | **6.11×** |

`config.json`: `num_attention_heads = 24`, `num_key_value_heads = 4`. **24 / 4 = 6.**
The tiled kernel reads the KV once **per query head group** — it does not exploit
GQA head sharing; the vector kernel does. Cross-check on the same data: `T=1` at
128K reads 8.59 GB in 17.05 ms = **504 GB/s**; `6 × 8.59 GB / 104.6 ms` = **493 GB/s**.
Self-consistent.

The regime change itself is unaffected (VERDICT §2.2 attacked it four ways and it
held). Only the label, and everything the label implies, are wrong — and what it
implies is a forward-binding steer away from Phase 6 KV quantization, which the
corrected accounting shows is the largest measured long-context lever in this data
(FAILRUN §3).

### Defect site 2 — the headline scalar was chord-selected, and the framing that produced it does not apply to a bandwidth-bound kernel

`scripts/ramp_kernel_regimes.py:65` computes **four** marginal chords per context
and commits all four. `POC-FINDINGS.md` §3 and the SC-P4DB *Source* table quote
exactly one — `T=8→64`, the most favourable — and neither mentions the others exist.

| chord | 32K | 128K | 256K |
|---|---|---|---|
| **8 → 64 (the one quoted)** | 55.7 | **56.8** | **46.7** |
| 12 → 48 (committed, unquoted) | 53.5 | 51.9 | 40.2 |
| 6 → 96 (full tiled regime) | 40.0 | 38.7 | 33.8 |
| **48 → 96 (committed, unquoted)** | 33.1 | **30.0** | **27.4** |
| 64 → 96 | 26.3 | **23.7** | **22.2** |
| OLS over all `T ≥ 6` (R² 0.94–0.96) | 43.2 | 42.1 | 36.6 |

The projection's own winner is block-64, whose local slope is 23.7 / 22.2 — *below*
the predecessor's 30-TFLOPS kill line, not "in the upper half of the 30–60 band."

**The fix is not to pick a better chord.** A TFLOPS-equivalent divides an imputed
FLOP count by a duration that Defect site 1 shows is set by **bytes**. Every chord
in the table moves with the byte:FLOP ratio, not with any achieved compute rate,
and comparing one against a compute-roofline kill line compares two different
physical constraints (FAILRUN §4). Publish all of them, label them as a byte:FLOP
diagnostic, and state that the projection never used any of them —
`ramp_longcontext_model.py:280` interpolates the measured curve directly, which is
why the conclusion survives the correction.

### Defect site 3 — the holdout error's direction is claimed backwards, in three places

Claimed in `POC-FINDINGS.md` §4 *Limits* (*"The error runs against RAMP … so the
projection is conservative"*), `ramp_longcontext_model.py:22-23` and `:288-291`.
The sweep printed twelve lines below the claim, in the same artifact, refutes it —
`sensitivity.attn_scale_sweep` in `longcontext-projection.json`:

| attention delta × | 128K, block-64 vs stock | 256K, block-64 vs stock |
|---|---|---|
| 0.5 | +40.3% | +44.5% |
| 1.0 | +44.9% | +49.8% |
| 2.0 | **+50.6%** | **+54.9%** |

More attention cost ⇒ a **larger** RAMP win, monotonically, at both contexts. In
the tiled regime the attention term is near-constant in `T`, so inflating it adds
roughly the same Δ to both arms and the ratio rises toward `tok_L/tok_S`. A model
that over-charges attention **over-states** RAMP's advantage. The conservative
direction is ×0.5.

Two further omissions the record owes the reader: the holdout cell is at **`T=3`**,
inside the *vector* regime — the tiled-regime term is validated by nothing — and
`POC-FINDINGS.md` §4 publishes quality statistics for both **rejected** models
(R²=0.648; α spread 266%) and **none for the accepted one**, whose worst statistic
(`fit.b_spread` = **67.4%**, the cross-case spread of its only `T`-dependent
coefficient) is computed, printed, and committed to JSON. The breaker ran the sweep
the writer should have and the ranking held across the full 2× range of `b`
(block-64 wins by +39.9% to +48.7% at 128K) — so this is a reporting defect, not a
conclusion defect, and the successor record must say both halves.

### Defect site 4 — three hygiene claims in POC §5 are false against the repository

Checked by the breaker against `git log --all --diff-filter=A` over `evidence/`;
re-confirmed at `172bcc9` (eleven files, none an invalidated run):

| claim | status |
|---|---|
| *"CLAUDE.md rule 4 earned its keep twice, and **both artifacts are committed**"* | **False.** Neither invalidated run was ever committed. |
| *"Both sides of swap and pagein counters are **now written into every artifact**"* | **False.** `environment_after`/`hygiene` exist only in the two `c131072` files. |
| *"Re-run **after swap settled**"* | **Unsupported.** `vm.swapusage` reads 8.53–8.59 GB of a 9216 MB store — 93% full — in *every* committed file, clean ones included. |

The consequence is that the swap-storm diagnosis — which decision 007's hygiene
note holds up as a lesson for the harness generally — **cannot be checked by
anyone**. This contravenes the audit-trace posture the writer explicitly invokes.

Related, same section: `dense-microbench.json` carries `rel_spread` of **2.75–7.17**
(275–717%) at `T ∈ {6, 12, 40, 96}`, and `POC-FINDINGS.md` §4 reads a 4.5 ms
difference between a 717%-spread cell and a 5%-spread cell to conclude *"roughly
1 ms/row … almost certainly the 48 GatedDeltaNet layers."* That inference is
noise-dominated. It does not reach the projection (`t_pass` uses `attn` only;
`kernel_time()` at `:114` is dead code — verified by the breaker) but it is
presented as a discovery.

### Defect site 5 — the floor constant is not reproducible from the committed harness, and its condition is untestable

Two defects that compound.

**(a) Not reproducible.** `run_attn_flops_microbench.sh` sweeps `1024 32768 131072
262144` at the script's default reps. It does **not** produce
`attn-microbench-c131072-lowT-confirm.json` (15 reps, `T ∈ {1,2,3,4,5,6,8,12}`), and
no committed script or shell record does. Because `ramp_kernel_regimes.py`'s
lowest-spread rule takes `T ∈ {1,2,3,4,5,8}` at 128K from that file (`cell_sources`),
**a reader who reruns the committed driver gets `min_proposal_block = 10` at 128K,
not 11.** The constant moved 10 → 11 between `ca6e739` and `74e24d0` on exactly this
mechanism. A production-candidate constant that cannot be regenerated from one
documented command is not "measured, not chosen."

**(b) Its condition is untestable.** `ramp_kernel_regimes.py:79` loops over
`(32768, 131072, 262144)` only. The `1024` control — the single cell proving the
dead zone is *absent* at short context, and therefore that any floor must be
conditional — is **not in `kernel-regimes.json` at all**. And the grid jumps
`1024 → 32768`: the penalty is absent at 1K (`T=5` = 1.095 ms, `T=6` = **0.985 ms**,
the tiled kernel is immediately cheaper) and present at 32K (+8.2 ms on a ~26 ms
pass — **31%**). The crossover lies somewhere in a **32× unsampled range**, and
`_DEAD_ZONE_MIN_CONTEXT = 32768` sits at the top of it while being labelled
*"Measured, not chosen."* For an append-only 128K-target harness this is not
academic: every session spends its early life inside the unmeasured band.

---

## Fix

1. **Make `kernel-regimes.json` regenerate from one documented command.** Add the
   low-`T` confirmation run to `scripts/run_attn_flops_microbench.sh` — same reps
   (15) and same truncated `T` grid (`1,2,3,4,5,6,8,12`) that produced the committed
   `attn-microbench-c131072-lowT-confirm.json` — so the driver emits every file
   `ramp_kernel_regimes.py` consumes. Read the committed file's own metadata block
   for the exact parameters; do not guess them.

2. **Add `4096` and `8192` to the microbench grid and to
   `ramp_kernel_regimes.py`'s context loop, and add `1024` to the regimes output.**
   Run them under the hygiene protocol (Fix step 6). This turns
   `_DEAD_ZONE_MIN_CONTEXT` from a guess at the top of a 32× band into a bounded
   interval, and puts the no-dead-zone-at-1K control into the artifact where a test
   can read it. `analyse()` hardcodes `rows[1]`, `rows[5]`, `rows[6]`, `rows[12]` —
   if a new context's grid lacks any of them it will `KeyError` rather than degrade;
   either keep the grids aligned or make the failure explicit.

3. **Correct the mechanism in every place it appears**, as Defect site 1 states it,
   with the 6.13× table and the GQA arithmetic: `POC-FINDINGS.md` §2 (including the
   "settles the mechanism dispute" subsection — the boundary finding is right, the
   label on the `T ≥ 6` side is not), `scripts/ramp_kernel_regimes.py`'s module
   docstring, and `docs/reviews/2026-08-26-ramp/SC-P4DA-ramp-serving-path-proposer.md`'s
   consult quotation and *Largest open risk* section.

4. **Publish all four chords plus the OLS fit** in `POC-FINDINGS.md` §3, delete
   *"in the upper half of the verdict's 30–60 band"*, and add the framing correction
   of Defect site 2: the marginal is regime-dependent and rising (0.27 ms/row at
   `T=6→12`, 1.72 at `T=48→96` at 128K); a TFLOPS-equivalent is a byte:FLOP
   diagnostic for a bandwidth-bound kernel, not an achieved compute rate; the
   projection uses the measured curve and depends on none of them.
   Add the bytes-per-pass accounting from `FAILRUN-SC-P4DB.md` §3 as the quantity
   that replaces it, **with its stated assumption** (that the kernel stays
   bandwidth-limited above `T=6`) marked as unverified.

5. **Correct the holdout-error direction** in `POC-FINDINGS.md` §4 *Limits*,
   `ramp_longcontext_model.py:22-23` and `:288-291`. Report `b_spread = 67.4%`
   alongside the rejected models' statistics, report the per-case `b` sweep as the
   robustness check that actually tests the conclusion, and state that the holdout
   validates the vector regime only. Note that the accepted model **has no R²** and
   cannot have one as structured (two parameters solved exactly from two numbers per
   case, `fit_live_short_context:226-249`) — cross-case spread is its only quality
   signal. Restate the *~30% headroom* as headroom against **differential**
   degradation with the stock arm's acceptance held fixed, and note the
   tokens-per-pass vs acceptance-rate conflation: uniform degradation scales both
   numerators and cannot reverse the ranking.

6. **Correct or delete the three false hygiene claims** in `POC-FINDINGS.md` §5.
   If the invalidated artifacts were not retained, say so plainly and mark both
   diagnoses unverifiable rather than leaving them as credited catches. Add
   `environment_after` to the runs produced under Fix steps 1–2 and note explicitly
   which committed sweeps predate the hygiene block. Downgrade the GatedDeltaNet
   inference in §4 to a labelled hypothesis and mark `dense-microbench.json`'s
   275–717%-spread cells in the artifact itself. Every run under Fix steps 1–2
   follows CLAUDE.md rule 4: High Power Mode, serial (the driver already refuses to
   run while an MTPLX server holds the GPU), ≥ 5 reps, swap counters before **and**
   after each context, median and p95, and the cross-context physical-consistency
   check POC §5 invented — a longer context measuring faster than a shorter one at
   the same `T` invalidates the run.

7. **Add the dead zone to the plan** — `docs/plans/ane-optimization-plan.md` §4D —
   as a constraint on any future drafting work, since it binds every proposer and
   not only RAMP: at long context, a proposal of ~5–10 tokens is strictly dominated;
   the two viable operating points are below the kernel switch or well above it.
   State the floor as a **measured constraint, not a shipped behaviour**, and point
   at `FAILRUN-SC-P4DB.md` §1 for why no code implements it.

8. **Report the resulting `min_proposal_block` values, whatever they are.** Fix
   steps 1–2 regenerate `kernel-regimes.json` from a fuller grid and a documented
   driver. If 128K comes back 10 rather than 11, that is the answer and it is
   reported as such (FAILRUN §5). Do not select the run that reproduces the
   committed value.

### Do NOT touch (scope fences)

- Do **NOT** modify anything under `rafale/`. No floor, no ceiling, no
  `_MIN_PROPOSAL_BLOCK`, no `_DEAD_ZONE_MIN_CONTEXT`, no `RampCounters` field, no
  context plumbing. `FAILRUN-SC-P4DB.md` §1 rules the production fix INVALID_CARD;
  re-introducing it here is the exact failure that ruling exists to prevent. If the
  new grid makes the floor look more attractive, that is a *finding for the next
  card*, not a licence to widen this one (Principle 8).
- Do **NOT** change `install()`'s `enabled=False` (`rafale/draft/ramp.py:285`) or
  `RAMP_ENABLED`/`RAMP_BLOCK`/`RAMP_FUZZY` in `scripts/launch_ramp_server.sh:41-43`.
- Do **NOT** edit `docs/decisions/007-ramp-long-context-block-length.md`. It was
  amended in the commit that adds this card, with the corrected mechanism, the reversed
  KV-quant item, and the hedged outcome. Re-amending it from inside this card would
  double-apply blocking edits B, E, I, K and L.
- Do **NOT** edit `CLAUDE.md`. The 300 GB/s vs 472–840 GB/s discrepancy
  (`FAILRUN-SC-P4DB.md` §6 items 1–2) is real, project-wide, and needs an operator
  decision and its own decision record. A card does not amend the project's
  governing constants.
- Do **NOT** touch any MTPLX source, vendor a kernel, add a custom SDPA, or "fix"
  MLX's premature switch point. Upstream observation, not this project's scope.
- Do **NOT** overwrite or delete any committed `evidence/` file. New runs land as
  new files; `kernel-regimes.json` is regenerated and its `cell_sources` must show
  which file every cell came from. If a run is invalidated, **commit it** — Defect
  site 4 exists because that did not happen last time.
- Do **NOT** change `rafale/draft/ngram.py`. `scripts/ramp_ab_bench.py:54` rebuilds
  its prompts by reading it live from the working tree, so editing it silently
  changes the benchmark workload with no error.
- Do **NOT** run the live 128K A/B or act on the projected throughput table as if
  measured. That is the next card and it gates a default flip.

## Touch List (only these files)

- `scripts/run_attn_flops_microbench.sh`
- `scripts/ramp_kernel_regimes.py`
- `scripts/ramp_longcontext_model.py` ← comment/doc correction only, Fix step 5; no model change
- `docs/reviews/2026-08-26-ramp-longcontext/POC-FINDINGS.md`
- `docs/reviews/2026-08-26-ramp-longcontext/evidence/` ← new microbench JSON + regenerated `kernel-regimes.json`
- `docs/reviews/2026-08-26-ramp/SC-P4DA-ramp-serving-path-proposer.md` ← amendment only, Fix step 3
- `docs/plans/ane-optimization-plan.md` ← §4D constraint only, Fix step 7

## Worktree provisioning (RULES §3.7)

Fix steps 3–7's document work needs only what git tracks.

Fix steps 1–2 are **macOS-only** and need the engine venv at
`/opt/homebrew/var/mtplx/venv-2.7.1/bin/python`, which
`run_attn_flops_microbench.sh` already hardcodes and which is not in git. The
microbench needs the GPU exclusively and allocates 8.6 GB of KV at 128K; the
committed artifacts were taken at **93% swap occupancy**, so check `vm.swapusage`
before starting and abort rather than produce another unverifiable sweep. No model
weights are required — the microbench synthesises tensors at the engine's shapes.

RULE 3.7.4 applies to any subprocess the tests spawn: pass
`env={**os.environ, "PYTHONPATH": str(<worktree root>)}` explicitly.

## Non-goals

- **Not** implementing a proposal floor, a ceiling, or any block-length change.
  Withdrawn; see `FAILRUN-SC-P4DB.md` §1.
- **Not** flipping any default, at any context.
- **Not** running the live 128K A/B. Next card.
- **Not** amending CLAUDE.md rule 9 or the plan's bandwidth constant.
- **Not** explaining the +26.1% holdout error, or the vector-path 1.6× bandwidth
  anomaly (`FAILRUN-SC-P4DB.md` §6 item 2). Both handed forward.
- **Not** re-running the live A/B corpus or touching `scripts/ramp_ab_bench.py`.

## Behavioral spec (Gherkin)

```gherkin
Scenario: the regimes artifact regenerates from one documented command
  Given a clean checkout at this card's baseline
  When scripts/run_attn_flops_microbench.sh is run once
  And scripts/ramp_kernel_regimes.py is run over its output
  Then every file named in kernel-regimes.json's cell_sources was produced by that run
  And no hand-run invocation is required to reproduce any committed cell

Scenario: the conditional-floor control is in the artifact the record cites
  Given the regenerated kernel-regimes.json
  When the 1024-token context is looked up
  Then it is present
  And its dead-zone band is empty or its worst penalty is under 0.2 ms

Scenario: the dead zone's onset is bounded to better than 32x
  Given the regenerated kernel-regimes.json
  When the contexts are listed
  Then 4096 and 8192 are present with measured dead-zone bands
  And the shortest context with a non-empty band is reported explicitly

Scenario: no runtime behaviour changes
  Given this card's diff
  When rafale/ is inspected
  Then it is unmodified
  And make test reports the same test count as at the baseline commit
```

## Acceptance criteria

1. **`make lint && make test` clean**, with the **exact** test count reported before
   and after. It must be **unchanged** — this card adds no tests because it changes
   no behaviour. A changed count means `rafale/` was touched; stop and report.

2. **`git diff --stat <baseline> -- rafale/` is empty.** Paste it. This is the
   card's central scope fence and the one a reviewer should check first.

3. **The regimes artifact regenerates from a single documented command**, per
   Gherkin scenario 1. Paste the command and the resulting `cell_sources`.

4. **`kernel-regimes.json` contains 1024, 4096, 8192, 32768, 131072, 262144**, each
   with `environment_before`/`environment_after` swap and pagein counters in its
   source artifact, ≥ 5 reps, and median + p95 reported.

5. **The cross-context physical-consistency check passes** on the new grid — no
   longer context measuring faster than a shorter one at the same `T`. Report the
   ratios, as POC §5 did (2.02 / 2.04 / 2.13 at `T` = 1 / 6 / 48 for 256K:128K).

6. **The regenerated `min_proposal_block` per context is reported, and any change
   from the committed 9 / 11 / 10 is stated, not reconciled.** A different value is
   a valid and valuable outcome (FAILRUN §5) and must be reported as such.

7. **The shortest context with a measured dead zone is named**, and
   `POC-FINDINGS.md` states the remaining unmeasured interval explicitly rather than
   asserting a threshold.

8. **Every one of the eight statements in Defect sites 1–5 is corrected at its
   cited location as an edit**, not restated in a new document. List file:line for
   each. VERDICT-SC-P4DB §1.9 records the standing complaint that upstream documents
   get corrected in prose and left unamended.

9. **No committed `evidence/` file is overwritten or deleted.** `git log
   --diff-filter=D -- docs/reviews/2026-08-26-ramp-longcontext/evidence/` must be
   empty. Any invalidated run from this card's own measurement is **committed**, with
   a note saying why it was rejected.

## State explicitly in your final report

- The exact test count before and after, and the `git diff --stat -- rafale/` output.
- Whether the new microbench cells ran, and if not, why. An honest "did not run"
  beats a projected number reported as measured, and Fix steps 3–7 stand alone.
- The regenerated `min_proposal_block` at every context, next to the committed values.
- The shortest context at which a dead zone was measured, and the interval still
  unmeasured below it.
- Whether any invalidated run occurred and where it is committed.
- Anything you were tempted to change outside the Touch List — especially in
  `rafale/` — and did not.

## Failure protocol

`INVALID_CARD` is honorable here and there are three live routes to it:

- **The machine cannot produce clean cells.** `vm.swapusage` at 93%, a resident
  server holding the GPU, or thermal drift that fails the cross-context consistency
  check of acceptance criterion 5. Stop after Fix steps 3–7, commit the invalidated
  run, and report. POC §5's own precedent — a documented refusal to produce
  contaminated data is a deliverable — is precisely what Defect site 4 shows was not
  actually followed.
- **The confirmation run's parameters cannot be recovered** from
  `attn-microbench-c131072-lowT-confirm.json`'s metadata. Do not guess them into the
  driver; a driver that produces a *different* file while claiming to reproduce the
  committed one is worse than no driver. Report and stop.
- **The new grid contradicts the dead-zone finding itself** — e.g. no dead zone at
  4096 *or* 8192 *or* 32768 on a clean re-run. That would put the whole POC's
  model-free result in question and needs a decision record and a fresh breaker,
  not an edit.

## Reviewer checklist before dispatch

- [ ] Every `file:line` in Defect re-read against `172bcc9`; engine line numbers
      treated as soft and asserted on symbols (VERDICT-SC-P4DB §1.10 records 969
      lines of drift across two MTPLX minor releases).
- [ ] No Touch List entry is under `rafale/`, and acceptance criterion 2 makes that
      checkable mechanically.
- [ ] Decision 007 is absent from the Touch List and named in a scope fence
      (already amended in the commit that adds this card).
- [ ] CLAUDE.md is absent from the Touch List and the bandwidth discrepancy is
      handed to the operator, not edited.
- [ ] Acceptance criteria 6 and 7 permit the new measurement to contradict the
      committed constants.
- [ ] Fix step 8 forbids selecting the run that reproduces the committed value.
