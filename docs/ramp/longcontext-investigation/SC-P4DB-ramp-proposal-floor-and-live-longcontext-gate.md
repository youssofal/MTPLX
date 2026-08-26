# SC-P4DB — Refuse to propose a block inside the attention kernel's dead zone, and gate RAMP's default on a live long-context A/B

**Status:** **INVALID_CARD — WITHDRAWN, NEVER DISPATCHED. Do not implement.**
**Superseded by:** `SC-P4DC-longcontext-record-correction-and-deadzone-grid.md`
**Why:** `VERDICT-SC-P4DB.md` (ACCEPT-WITH-EDITS, 12 blocking edits) and
`FAILRUN-SC-P4DB.md` (the fail-run ruling). The empirical POC stands; the
production fix below does not. Four independent grounds, the decisive one being
that the proposed floor binds in **zero shipped configurations**
(`FAILRUN-SC-P4DB.md` §1, Link 4). The stated mechanism under it — "the tiled
kernel reads the KV once" — is refuted by its own evidence by 6.13×; the chosen
action (decline to propose) is an acceptance-dependent bet the POC forbids
assuming; and the prescribed `context_hint` is a process constant that cannot be
conditional on live context. Kept unedited below as the audit trace.
**Baseline commit (as declared, and wrong):** `ca6e739` · worktree
`phase-0-bench-harness`, branch `claude/plan-review-upgrade-9epmpr` — the branch is
actually `worktree-phase-0-bench-harness`, and `ca6e739` contains neither
`min_proposal_block` nor `cell_sources`, so acceptance criterion 3's test would
have `KeyError`d (VERDICT §1.7).
**Parent:** `docs/reviews/2026-08-26-ramp/SC-P4DA-ramp-serving-path-proposer.md` (shipped at `5eaceb3`)
**Answers:** `docs/reviews/2026-08-26-ramp/VERDICT-SC-P4DA.md` §1.1 blocking edit 5
**POC:** `docs/reviews/2026-08-26-ramp-longcontext/POC-FINDINGS.md`

---

## Source

The breaker's verdict on SC-P4DA made one edit blocking (§3, item 5): name
**achieved attention FLOPS at long context** as the single scalar that decides
RAMP's fourth kill-criterion bullet, and make measuring it the first task of the
long-context follow-up card. This is that card, written after doing that
measurement.

**This is a FIX plus a small GREENFIELD addition to `rafale/draft/ramp.py`.**
Confirmed by reading the file at `ca6e739`: `block_for_ext` (`:66-76`),
`fixed_block_for_ext` (`:79-91`) and `install` (`:284-400`) all exist and all
treat block length as a context-free scalar. There is **no** minimum-length
concept anywhere in the module — `grep -n "min_block\|floor\|dead" rafale/draft/ramp.py`
returns nothing. The floor is new; the ladder's behaviour at short rungs is the fix.

### What the measurement returned

The verdict's framing was: under ~30 TFLOPS the optimum collapses to the stock
ladder and RAMP dies; under ~60 TFLOPS it survives. The measured answer is that
**both figures are true of the same machine**, depending on which quantity you
take:

| context | aggregate TFLOPS at T=48 | marginal TFLOPS-equiv (T=8→64) |
|---|---|---|
| 128K | **15.8** | **56.8** |
| 256K | **14.8** | **46.7** |

Block-length decisions move only the *marginal* figure, which lands in the
"survives" half. **SC-P4DA's fourth kill-criterion bullet is not tripped.**
Full derivation, held-out validation and sensitivity: POC-FINDINGS §3–4.

---

## Defect / gap

### Defect site 1 — the stock ladder's first rung sits inside a measured dead zone

```python
# rafale/draft/ramp.py:60-76
# Mirrors mtplx.context_copy._BLOCK_LADDER (mtplx 2.7.1). ...
_BLOCK_LADDER = (8, 12, 16, 24, 32)


def block_for_ext(ext: int, k_cap: int) -> int:
    ...
    idx = max(0, min(int(ext), len(_BLOCK_LADDER) - 1))
    return min(_BLOCK_LADDER[idx], max(4, k_cap))
```

`mx.fast.scaled_dot_product_attention` is served by two kernels. Below `T=6` a
vector path costs roughly one KV pass per query row; at `T=6` MLX switches to a
tiled path that reads the KV once and is then nearly flat in `T`. The switch is
not free, and the tiled line does not overtake the vector line again until
`T ≈ 11`:

| context | vector path | switch T=5→6 | tiled path (T=6→12) | dead zone | worst penalty | `min_proposal_block` |
|---|---|---|---|---|---|---|
| 32K | 2.70 ms/row | +10.9 ms | 0.275 ms/row | block 5–8 | 8 ms/pass | 9 |
| 128K | 10.24 ms/row | **+61.4 ms** | 0.273 ms/row | block 5–10 | **51 ms/pass** | **11** |
| 256K | 21.74 ms/row | **+118.5 ms** | 0.230 ms/row | block 5–9 | **97 ms/pass** | 10 |

Evidence: `evidence/kernel-regimes.json`, from
`evidence/attn-microbench-c{32768,131072,262144}*.json`. Each cell is taken from
whichever sweep measured it with the lowest relative spread; `cell_sources` in
that JSON records which file each number came from.

**Observable consequence.** With `block=None` (the module's default — the engine
ladder is kept), a backward-extension of 0 selects `_BLOCK_LADDER[0] = 8`, i.e.
`T = 9`, which is inside the dead zone at every long context measured. That pass
pays the full tiled-kernel step to verify 8 proposed tokens. At 128K it burns
~51 ms/pass more than the same proposal would have cost on the vector path, and
it is **strictly dominated** by both alternatives available at that moment:
propose nothing (`T=3`, **37.58 ms** of attention) or propose more (measured
`T=8` = 118.90 ms vs `T=12` = 121.06 ms — **2.2 ms, 1.8%, for four more proposed
tokens**).

The dominance is what makes this a defect rather than a tuning preference: there
is no acceptance rate at which block 8 is the right choice at 128K, because
block 11 costs under 2% more and proposes ~38% more.

**Pre-empt the obvious wrong fix.** Clamping the ladder *upward* to the dead
zone's ceiling for all contexts — `max(block, 11)` unconditionally — is WRONG.
At ~800 tokens the dead-zone penalty is under 0.1 ms (`attn-microbench-c1024.json`:
T=8 is 0.99 ms, T=12 is 1.02 ms) and the short rungs exist to keep *acceptance*
high when the backward-extension signal is weak. Forcing block 11 at short
context changes a regime this POC measured nothing about. The floor must be
conditional on context.

Equally wrong: making the floor a bare literal. MLX's switch point is an
implementation detail of `mx.fast.scaled_dot_product_attention` and by MLX's own
numbers it is premature — between `T=6` and `T≈11` the tiled kernel is slower
than the vector kernel would have been (POC-FINDINGS §7 item 5). A future MLX
release can move or erase this. The constant must carry its provenance and a
test that re-derives it from committed evidence.

### Defect site 2 — nothing stops a long-context default flip on projected evidence

`scripts/launch_ramp_server.sh:41-43` sets `RAMP_BLOCK="${RAMP_BLOCK:-48}"`, and
`rafale/draft/ramp.py:284` correctly defaults `install(enabled=False)`. That is
right and this card does not change it.

What is missing is any recorded statement of **what would license changing it**.
SC-P4DA's kill criterion bullet 4 is now answerable, and the answer is
favourable, which is exactly the condition under which a later reader flips a
default on a projection. Every long-context throughput number in POC-FINDINGS §4
is *projected*, carries a **+26.1%** known error on its one held-out validation
cell, and rests on an acceptance assumption measured only at ~800 tokens.

**Test coverage confirmed absent.** `grep -n "128\|131072\|long_context\|dead_zone" tests/test_ramp.py`
returns nothing; `tests/test_ramp.py` has 13 tests
(`test_ramp_is_off_by_default` at `:388` is the closest) and none concerns block
length as a function of context. `tests/test_engine_seam.py` asserts the ladder
matches the installed engine's, which this card must not break.

---

## Fix

1. **Add the measured dead-zone constants to `rafale/draft/ramp.py`**, as data
   with provenance, not as literals in a branch:

   ```python
   # Measured, not chosen: docs/reviews/2026-08-26-ramp-longcontext/evidence/
   # kernel-regimes.json. MLX switches scaled_dot_product_attention from a
   # per-row vector kernel to a tiled kernel at T=6; the tiled kernel's fixed
   # cost is not amortised until T~11, so a proposal of 5..10 tokens is
   # strictly dominated by proposing nothing or by proposing >= 11.
   _MIN_PROPOSAL_BLOCK = 11        # max over contexts of kernel-regimes.json's
                                   # min_proposal_block: 9 @32K, 11 @128K, 10 @256K
   _DEAD_ZONE_MIN_CONTEXT = 32768  # shortest context where the penalty is measured
   ```

   Take the **maximum** `min_proposal_block` across the measured contexts, not
   the value at any one context, so the floor is safe everywhere. Note the band
   is not monotone in context (11 at 128K, 10 at 256K) — that is real and comes
   from the vector-path slope growing faster with `C` than the tiled path's
   fixed cost does; do not "fix" it into a monotone formula.

2. **Add a context-aware block policy.** `install()` currently takes
   `block: int | None` and `_installed_block_for_ext` (`:378-381`) ignores
   context entirely. Add a keyword-only `min_context_for_floor: int = _DEAD_ZONE_MIN_CONTEXT`
   and apply the floor inside `_installed_block_for_ext`:
   - Determine the current context from the `k_cap`/history the engine already
     passes, **or**, if that is not reachable without an engine change, from an
     explicit `context_hint` the launcher sets. **If genuinely ambiguous,
     implement the `context_hint` version**, because this card's scope fence
     forbids touching engine source and a wrong reach into engine internals is a
     correctness bug, not a tuning miss.
   - When the resolved block would fall in `1 .. _MIN_PROPOSAL_BLOCK - 1` and
     context ≥ `_DEAD_ZONE_MIN_CONTEXT`, **propose nothing** (return the engine's
     no-proposal path) rather than rounding up. Rounding up changes which tokens
     are proposed and therefore can change acceptance; declining to propose
     cannot. Declining is the conservative choice and the one this POC's evidence
     supports.

3. **Count it.** Add `floor_declines` to `RampCounters` (`:95`) and increment on
   every decline, so the effect is observable in live telemetry. VERDICT §1.7
   flagged `STATS` in the POC as dead instrumentation that made fuzzy hits
   unmeasurable — do not repeat that: this counter must be exported wherever the
   existing counters are.

4. **Leave `block_for_ext` (`:66-76`) byte-faithful to the engine.** It exists so
   the "reduces exactly to the engine's algorithm" control means something, and
   `tests/test_engine_seam.py` asserts it. Apply the floor in the *installed*
   wrapper only.

5. **Write the live long-context A/B script** `scripts/ramp_longcontext_ab.sh`,
   modelled on `scripts/launch_baseline_server.sh` (decision 001) — **not** on
   `scripts/ramp_launch_patched_server.py`, which VERDICT §1.8 showed reproduces
   decision 001's CLI flags but none of its environment. It must:
   - run **ABBA-counterbalanced**, not baseline-first (VERDICT §1.5);
   - run ≥ 5 reps per cell at 128K under the hygiene protocol, with thermal
     cool-down and a **swap check before and after each cell** — POC-FINDINGS §5
     documents a swap-driven contamination that produced internally *tight* but
     2.7× wrong cells, invisible to any spread-based test;
   - assert temperature-0 output identity per case, per arm;
   - write an environment + config block into its JSON (VERDICT §3 item 8);
   - cover arms: stock ladder, block 11 (the floor), block 48, block 64.

6. **Record the gate outcome** in `docs/decisions/007-ramp-long-context-block-length.md`:
   SC-P4DA kill-criterion bullet 4 evaluated, **not tripped**, with the measured
   marginal-vs-aggregate distinction as the reason, and the default left off.

7. **Amend the parent card and the plan** (Principle 1, and VERDICT §1.9's
   standing complaint that upstream documents get corrected in prose and left
   unamended):
   - `SC-P4DA-ramp-serving-path-proposer.md`: its *Largest open risk* section
     says the long-context question is open. Replace with a pointer to this card
     and POC-FINDINGS, and correct the consult quotation — "reads KV for every
     proposed row" is true only for T ≤ 5.
   - `docs/plans/ane-optimization-plan.md` §4D: add the dead zone as a constraint
     on any future drafting work, since it binds every proposer, not just RAMP.

### Do NOT touch (scope fences)

- Do **NOT** change the default `enabled=False` in `install()` (`:285`), or
  `RAMP_ENABLED`/`RAMP_BLOCK`/`RAMP_FUZZY` defaults in
  `scripts/launch_ramp_server.sh:41-43`. This card measures; it does not flip a
  default. That is SC-P4DA's Fix step 7 and acceptance criterion #9 and it stays.
- Do **NOT** modify `block_for_ext` (`:66-76`) or `_BLOCK_LADDER` (`:63`) —
  `tests/test_engine_seam.py` asserts they mirror the installed engine.
- Do **NOT** touch any MTPLX source. The whole value of SC-P4DA is the zero-diff
  seam (VERDICT §2.1 verified it survives adversarial attack); a patch here
  destroys it.
- Do **NOT** "fix" MLX's premature kernel switch, vendor a kernel, or add a
  custom SDPA. Upstream observation, not this project's scope.
- Do **NOT** touch the fuzzy anchor (`_FuzzyAnchor`, `:158-208`), consensus
  ranking, or wide-corpus indexing — measured at +0.0% and −0.7% and killed in
  SC-P4DA.
- Do **NOT** change `rafale/draft/ngram.py`. `scripts/ramp_ab_bench.py:54`
  rebuilds its prompts by reading it live from the working tree (VERDICT §1.8),
  so editing it silently changes the benchmark workload with no error.
- Do **NOT** act on the projected 128K/256K throughput table as if measured.

## Touch List (only these files)

- `rafale/draft/ramp.py`
- `tests/test_ramp.py`
- `scripts/ramp_longcontext_ab.sh` (new)
- `docs/decisions/007-ramp-long-context-block-length.md` (new)
- `docs/reviews/2026-08-26-ramp/SC-P4DA-ramp-serving-path-proposer.md` ← amendment only, Fix step 7
- `docs/plans/ane-optimization-plan.md` ← §4D constraint only, Fix step 7

## Worktree provisioning (RULES §3.7)

`<none extra for the pure-Python work>` — Fix steps 1–4 and their tests need only
what git tracks; `make lint && make test` run off the project's own `uv`
environment.

Fix step 5's **live** A/B is macOS-only and needs the engine venv at
`/opt/homebrew/var/mtplx/venv-2.7.1/bin/python` and the model at
`/Users/misterj/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Quality`,
neither of which is in git. Both are absolute paths already used by
`scripts/launch_baseline_server.sh:11-12`, so no symlink is required — but the
script must **fail loudly if either is absent** rather than silently benchmarking
nothing.

RULE 3.7.4 applies to any subprocess the worker's tests spawn: pass
`env={**os.environ, "PYTHONPATH": str(<worktree root>)}` explicitly, or the child
resolves `rafale` against whatever is installed rather than this worktree.

## Non-goals

- **Not** turning RAMP on by default, at any context. Explicitly out of scope.
- **Not** re-tuning the block length. 48 stays the shipped value unless Fix step
  5's live sweep says otherwise; this card adds a floor, it does not move the
  ceiling.
- **Not** validating at temperature > 0. The engine takes a structurally
  different acceptance path there (VERDICT §2.2) and no number in this POC or
  the parent applies.
- **Not** explaining the +26.1% holdout error. Handed forward (POC-FINDINGS §7
  item 1).
- **Not** implementing KV quantization. Phase 6. Note that the POC's mechanism
  finding says KV quant would help the *aggregate* attention cost but barely move
  the *marginal* cost, so it is not the rescue path the parent card's imported
  mechanism implied.

## Behavioral spec (Gherkin)

```gherkin
Scenario: a dead-zone block is declined at long context
  Given RAMP is installed with the engine ladder (block=None) and a 131072-token context
  When the proposer would select block 8 from _BLOCK_LADDER for a weak backward extension
  Then no block is proposed
  And RampCounters.floor_declines is incremented by 1

Scenario: the same dead-zone block is kept at short context
  Given RAMP is installed with the engine ladder and an 800-token context
  When the proposer would select block 8
  Then block 8 is proposed unchanged
  And RampCounters.floor_declines is 0

Scenario: a block above the floor is never altered
  Given RAMP is installed with block=48 and a 131072-token context
  When the proposer selects a block
  Then it proposes exactly 48 tokens
  And RampCounters.floor_declines is 0

Scenario: the floor constant is the one the evidence supports
  Given evidence/kernel-regimes.json
  When _MIN_PROPOSAL_BLOCK is compared against max(min_proposal_block) over contexts
  Then they are equal

Scenario: RAMP is still off by default
  Given no explicit enable
  When install() is called with no arguments
  Then it returns None and mtplx is never imported
```

## Acceptance criteria — FAIL-FIRST MANDATORY

1. **Write the failing test first and capture its output to disk before the fix
   exists.** This is greenfield for the floor, so the fail-first evidence is the
   `AttributeError` / `ImportError` on `_MIN_PROPOSAL_BLOCK` and
   `RampCounters.floor_declines` — that IS the gap. Run:
   ```
   <pytest command> > SC-P4DB-failfirst-before-fix.txt 2>&1
   ```
   then `cat` it to confirm it exists and is non-empty, and paste that
   confirmation. On-screen output is not the deliverable; the file is.

2. **Five tests, one per Gherkin scenario, named after the scenario.** The
   dead-zone-decline and short-context-preserved pair must be a matched pair over
   the same block value — a test that only shows the decline cannot distinguish
   the intended fix from an unconditional clamp, which Defect site 1 names as the
   wrong fix.

3. **The floor constant is re-derived from committed evidence, not asserted.**
   The "floor constant" test must read
   `docs/reviews/2026-08-26-ramp-longcontext/evidence/kernel-regimes.json` and
   compare, so that a future MLX release moving the switch point breaks the test
   loudly instead of leaving a stale magic number. Cross-platform: this test must
   not import `mlx`.

4. **`test_ramp_is_off_by_default` (`tests/test_ramp.py:388`) and
   `test_ramp_install_disabled_never_imports_mtplx` (`:402`) still pass
   unmodified.** If either needs editing, stop — the change has broken the
   parent card's safety property and that is a card failure, not a test to fix.

5. **`tests/test_engine_seam.py` passes unmodified**, proving `block_for_ext` is
   still byte-faithful to the installed engine (Fix step 4).

6. **`make lint && make test` clean**, and report the **exact** test count before
   and after. Note: VERDICT §6 documented a spurious first-fail lock on this
   worktree hash; if `make test` is blocked, do not work around it — report it
   and stop.

7. **Live 128K A/B executed** per Fix step 5, with its JSON committed under
   `docs/reviews/2026-08-26-ramp-longcontext/evidence/`. Report per arm: median
   and p95 tok/s, **suspension count per run** (not aggregate block acceptance —
   VERDICT §1.2 showed the aggregate does not predict guard firing), swap before
   and after each cell, and the temperature-0 sha per case.

8. **The live result is compared against POC-FINDINGS §4's projection and the
   delta is reported**, whatever it is. The projection predicts block 48 at
   +42.5% and block 64 at +44.9% over the stock ladder at 128K, and is known to be
   conservative in two directions. **A live result that contradicts it is a
   valid and valuable outcome and must be reported as such, not tuned away.**
   If the live sweep shows the long block losing at 128K, say so and stop — that
   trips SC-P4DA's kill criterion bullet 4 after all and needs a decision record,
   not a fix.

9. **`docs/decisions/007-*.md` written** per Fix step 6, and the two upstream
   amendments of Fix step 7 made as edits, not as prose in a review document.

## State explicitly in your final report

- The exact test count before and after, and the fail-first file's path and size.
- Whether the live 128K A/B ran, and if not, why — an honest "did not run"
  beats a projected number reported as measured.
- Every place the live numbers disagreed with POC-FINDINGS §4.
- Whether `_MIN_PROPOSAL_BLOCK` ended up at the value Fix step 1 derives, and
  the arithmetic if not.
- Anything you were tempted to change outside the Touch List, and did not.

## Failure protocol

`INVALID_CARD` is honorable here and there are two live routes to it:

- If the current context is **not reachable** inside `_installed_block_for_ext`
  without touching engine source, and the `context_hint` fallback of Fix step 2
  also proves unworkable, stop and say so. Do not reach into engine internals to
  make the floor work — the zero-diff seam is worth more than the floor.
- If the live A/B at 128K cannot be run on this machine within the hygiene
  protocol (swap pressure, thermal, a resident server that will not release the
  GPU), stop after Fix steps 1–4 and report. POC-FINDINGS §5 is the precedent:
  a documented refusal to produce contaminated data is a deliverable.

## Reviewer checklist before dispatch

- [ ] Every `file:line` in Defect re-read against `ca6e739` — VERDICT §1.10
      records 969 lines of drift across two MTPLX minor releases, so treat
      engine line numbers as soft and assert on symbols.
- [ ] The floor is conditional on context, not unconditional (Defect site 1).
- [ ] The off-by-default property appears in a scope fence, an acceptance
      criterion, and a Gherkin scenario — VERDICT §1.6 burned the parent card
      for leaving it in prose only.
- [ ] Acceptance criterion 8 permits a contradicting live result.
