# FAIL-RUN ANALYSIS — SC-P4DB, and why its production fix is INVALID_CARD

**Role:** fail-run analyst (JUDGE-PROTOCOL — not the writer of SC-P4DB, not its breaker)
**Card analysed:** `SC-P4DB-ramp-proposal-floor-and-live-longcontext-gate.md` (frozen, undispatched)
**Verdict analysed:** `VERDICT-SC-P4DB.md` (ACCEPT-WITH-EDITS, 12 blocking edits)
**Tree:** `172bcc9` · worktree `phase-0-bench-harness`, branch `worktree-phase-0-bench-harness`, clean
**Date:** 2026-08-26

---

## Ruling

**The empirical POC stands. The card's production fix is `INVALID_CARD` and is not
re-issued.** SC-P4DB is superseded by `SC-P4DC-longcontext-record-correction-and-deadzone-grid.md`,
which carries every one of the breaker's twelve blocking edits and the two
corrections found here, and which touches **no code in `rafale/`**.

The short form: the `T=6` kernel regime change is real, the dominance argument for
a floor is real — and the floor, implemented exactly as the card asks, **would not
execute in any configuration this project ships**. A production change with zero
production blast radius, resting on a constant that moved by ±1 on a single
re-measurement, guarded by a threshold that was never measured across a 32× band,
is not a fix. It is ceremony. Principle 7: never add to look thorough.

What replaces it is the finding the corrected mechanism actually produces, which
is larger than the floor and points at a different phase entirely (§3).

---

## 1. Why the card failed — the four-link chain

The breaker filed twelve blocking edits. Ten are corrections to prose and
artifacts. Two (§1.3, §1.4) are structural, and they compound with two further
defects found here into a chain where every link independently blocks dispatch.

### Link 1 — the mechanism under the fix was wrong (breaker §1.2, confirmed)

The card, the POC, `ramp_kernel_regimes.py`'s docstring and decision 007 all say
the tiled kernel "reads the KV once and is then nearly flat in `T`." The tiled
kernel's OLS intercept is **6.13× one measured KV read** at 128K and 6.11× at
256K, against `num_attention_heads / num_key_value_heads = 24 / 4 = 6`. The tiled
kernel does not exploit GQA head sharing. Re-derived here and confirmed exactly.

This is not a label change. "Reads KV once, then flat in `T`" says the verify
pass's `T`-dependence is compute, and compute is what a block-length knob buys.
"Reads KV six times per pass" says the pass is dominated by a **fixed byte cost
that no block length moves**. The card's entire Fix is a block-length knob, and it
was justified on the first reading.

### Link 2 — the chosen action was the one the evidence does not license (breaker §1.3, confirmed)

The POC's model-free result is a dominance argument with exactly one form: block 8
and block 11 both pay the tiled kernel's fixed cost, so block 11 buys ~38% more
proposed tokens for 1.8% more time. It compares two actions **on the same side of
the kernel switch**.

The card's Fix instructs the coder to *decline to propose* — the other side of the
switch, where a `T=3` MTP-only pass costs 37.58 ms of attention against 121.06 ms
at `T=12`. Whether declining wins is decided entirely by acceptance at long
context, which the card's own Non-goals declare unmeasured and forbid assuming.
The card's stated rationale ("declining cannot change acceptance") is
self-refuting: declining changes the proposal from 8 tokens to zero, a strictly
larger perturbation than 8 → 11.

### Link 3 — the prescribed context source could not be conditional (breaker §1.4b, confirmed and sharpened)

The card makes "the floor is conditional on context, not unconditional"
load-bearing in three places, then routes the coder to a launcher-set
`context_hint` — a **process constant**. A hint of 131072 applies the floor from
the session's first token, which is verbatim the unconditional clamp the card's
own "Pre-empt the obvious wrong fix" paragraph forbids. A coder following the
card's instruction arrives at the card's named wrong answer.

### Link 4 (new, and decisive) — the floor binds in zero shipped configurations

Not raised by the breaker. Read `rafale/draft/ramp.py:378-381`:

```python
def _installed_block_for_ext(ext: int, k_cap: int) -> int:
    if block is None:
        return base_block_for_ext(ext, k_cap)
    return block
```

The floor can only bind where the ladder is consulted, i.e. `block is None`. Enumerate
every configuration that exists:

| configuration | ladder consulted? | floor binds? |
|---|---|---|
| `install(enabled=False)` — the shipped default (`:285`) | RAMP not installed at all; the engine's own `block_for_ext` runs | **no** |
| `scripts/launch_ramp_server.sh:41-43` — `RAMP_BLOCK=48` | `block=48`, ladder bypassed, `48 > 11` | **no** |
| `install(enabled=True, block=None)` | yes | yes — but this is the A/B **control arm**, not a shipped path |

So the "one production change this POC licenses" changes the behaviour of one
experimental control arm and nothing else. Worse, applying a floor to the control
arm *contaminates the control*: acceptance criterion 5 and `tests/test_engine_seam.py`
exist precisely to keep "reduces exactly to the engine's algorithm" meaningful, and
Fix step 4's "leave `block_for_ext` byte-faithful, apply the floor in the installed
wrapper only" preserves the letter of that while breaking its purpose — the arm
labelled *stock ladder* in Fix step 5's A/B would no longer be the stock ladder.

**The only configuration in which the floor is a genuine production fix is a
floor-only install** — `install(enabled=True, block=None, fuzzy=False)`, a minimal
rebind whose entire effect is making the engine's own ladder skip the dead zone.
That is a *default flip*, which SC-P4DB fences off as out of scope and decision 007
gates behind the live 128K A/B. It is the right eventual shape and it is not
licensed today. §5 records it.

---

## 2. Two seam facts, verified on metal against the installed engine

Both matter for whoever writes the successor card, and one overturns the breaker.

Source: `/opt/homebrew/var/mtplx/venv-2.7.1/lib/python3.13/site-packages/mtplx/generation.py`,
mtplx 2.7.1 — the engine `install()` actually rebinds.

### 2a — live context length is reachable, and the breaker's design is confirmed

```
7832:  _cc_pos, _cc_ext = ccopy_index.find(_cc_hist, max_pos=len(prompt_ids))
7835:      _cc_klen = block_for_ext(_cc_ext, ccopy_k)
```

with `_cc_hist = prompt_ids + tokens` (`:7830`). `find` is called immediately
before `block_for_ext`, in the same block, on the same iteration. So
`len(history)` captured into the closure by `_InstalledRampIndex.find`
(`rafale/draft/ramp.py:362`) and read by `_installed_block_for_ext` (`:378`) is
**exactly the current context length at the moment the block is chosen**. The
breaker's blocking edit D is correct and implementable with no engine change and
no engine-internals reach. `context_hint` was never needed.

Caveat for the successor card: a closure variable written by one function and read
by another is **per-process mutable state**, correct only while decode is serial.
It must be documented as such, and the successor must state whether MTPLX can have
two generations in flight in one process.

### 2b — the seam CAN express "decline", contra VERDICT §1.4(a)

The breaker states `block_for_ext` "has no no-proposal return channel" and that
the real channel is `NgramIndex.find` returning `(None, -1)`. The second half is
right; the first is not:

```
7836:  _cc_block = [int(t) for t in prompt_ids[_cc_pos:_cc_pos + _cc_klen]]
...
7843:  if _cc_block:
```

`_cc_klen = 0` produces an empty slice, `if _cc_block:` is false, and control falls
through to the normal MTP round — the engine's own comment at `:7840-7842` names
this fallthrough explicitly for the grammar-truncation case. **Returning 0 from
`block_for_ext` is a working decline**, landing on the identical path as a `find`
miss.

This does not rescue SC-P4DB — Link 2 kills declining on evidence grounds, not
mechanism grounds — but the successor must record it accurately. It is also
**undocumented engine behaviour**: `context_copy.py` floors at 4 in two places, so
0 is outside the range the engine's own code suggests, and VERDICT-SC-P4DB §1.10
records 969 lines of drift across two MTPLX minor releases. If a future card wants
decline, route it through `find` returning `(None, -1)` — the documented, tested
contract — not through a 0 that works by slice truthiness.

---

## 3. What the corrected mechanism actually says the lever is

Once the tiled kernel is understood as reading KV **six times per pass**, the
right accounting is bytes, not FLOPs. Using the project's own committed constants
— decision 003's **64 KB/token** KV and the plan's **~28 GB** of Q8 weights per
pass — and converting measured attention time at the KV-read rate this POC
measured directly (504 GB/s at 128K, `T=1`):

| context | KV bytes moved by one verify pass | vs the ~28 GB of weights |
|---|---|---|
| 128K, `T=3` (MTP only, no proposal) | 19 GB | 0.7× |
| 128K, `T=12` (block 11) | 61 GB | 2.2× |
| 128K, `T=49` (block 48) | **79 GB** | **2.8×** |
| 256K, `T=49` (block 48) | **168 GB** | **6.0×** |

*(`T=48` at 128K → 156.8 ms from `aggregate_tflops`, which reproduces the 156.8 ms
cell POC §5 quotes independently. The conversion assumes the kernel stays
bandwidth-limited at the measured rate above `T=6`; §6 item 3 records where that
assumption is soft.)*

**At the target contexts, a verify pass moves 2.8–6.0× more KV bytes than weight
bytes.** The project's framing — CLAUDE.md rule 9, "decode is bus-bound, weights
are the traffic" — is correct at short context and inverted at 128K+. This is the
finding, and it is bigger than the floor:

1. **Decision 007 item 5 is backwards, and the reversal is quantified.** Q8 KV
   quantization halves the dominant term of every long-context verify pass:
   ~40 GB/pass saved at 128K block-48, ~84 GB/pass at 256K. Phase 6 KV quant is
   not "not the rescue path" — on this data it is **the single largest measured
   lever at long context**, larger than any block-length choice. Decision 007 is a
   forward-binding record that would have steered Phase 6 away from it.
2. **The floor is a rounding error against it.** The dead-zone penalty is 51 ms /
   ~26 GB per pass at 128K, and only on passes that land in a five-token band, in
   a configuration nothing ships (§1, Link 4).
3. **There is no ceiling to add.** Within the tiled regime, more rows amortise the
   6× fixed read over more tokens; the projection's optimum (block 64) is the
   measured curve saying so. A ceiling would be a tuning move, and tuning is what
   the live A/B is for.

**Answer to "does raising short blocks to a floor still make sense?"** — the
dominance argument survives the mechanism correction intact (it is model-free and
compares two points on one measured curve), so the floor is not *wrong*. It is
*inert*: correct, unshippable today, and pointed at the wrong term. Ship the
mechanism correction; let Phase 6 have the lever.

---

## 4. The TFLOPS chord question, resolved by dissolving it

The breaker's §1.1 found the headline `56.8 TFLOPS marginal` was one of four
committed chords, with `48→96` at **30.0** and `64→96` at **23.7** — at or below
the predecessor's ~30 TFLOPS kill line, and at the block-64 operating point the
projection actually names as the winner.

**The corrected mechanism does not select a better chord. It invalidates the
framing.** A TFLOPS-equivalent is `flops / time` for a kernel whose time is set by
bytes. Every chord in that table is an artefact of dividing an imputed FLOP count
by a bandwidth-limited duration; the numbers rise and fall with `T` because the
byte:FLOP ratio does, not because any compute rate changed. Comparing any of them
against a compute-roofline kill line — which is what VERDICT-SC-P4DA §1.1's
`flops / FLOPS_achieved` term is — compares two quantities that do not measure the
same physical constraint.

So the gate as originally posed is **mis-specified**, and answering it either way
answers the wrong question. Three consequences, all carried into the successor
card and the 007 amendment:

- Publish the full chord table (blocking edit A) **and** label it as what it is:
  a diagnostic of the byte:FLOP ratio, not an achieved compute rate. Do not pick
  one. Do not defend one.
- Replace it as the decision quantity with the bytes-per-pass accounting of §3.
- The projection is unaffected and this is the load-bearing point: `t_pass`
  interpolates the **measured** attention curve (`ramp_longcontext_model.py:280`)
  and never touches a FLOPS scalar. The breaker verified this independently and so
  did I. The conclusion "block 64 beats the stock ladder at 128K/256K" rests on the
  measured curve and survives; only the narrative laid over it was chord-picked.

---

## 5. The constants question (blocking edits G, H, and the 10 → 11 move)

`_MIN_PROPOSAL_BLOCK` is not re-issued, because no code change is. For the record,
so the successor does not relitigate it:

- **`11`, not `10`, is the defensible value** — max over measured contexts
  (9 @32K, 11 @128K, 10 @256K), which is the only choice that is safe everywhere.
  But it is defensible **only after** blocking edit J lands: at `ca6e739` the 128K
  figure was 10, and it became 11 at `74e24d0` when the lowest-spread-per-cell rule
  picked up `attn-microbench-c131072-lowT-confirm.json` — a file **no committed
  script produces**. A production constant that a skeptical reader cannot regenerate
  from the committed harness is not "measured, not chosen." Fix the driver first;
  then the constant is real. This is why the successor card's Fix step 1 is the
  driver, not the constant.
- **`_DEAD_ZONE_MIN_CONTEXT = 32768` is chosen, not measured**, and the unmeasured
  band `(1K, 32K)` is 32× wide. This is not academic for an append-only
  128K-target harness: every session spends its early life inside it. The successor
  measures 4096 and 8192 and emits 1024 into `kernel-regimes.json`, so the
  *condition* on the floor becomes testable — today, acceptance criterion 3's test
  can verify the floor's value and **nothing** can verify its condition, because
  the 1K control the whole conditionality rests on is not in the file the test
  reads.
- **Acceptance criterion 3 would `KeyError` as written** (breaker §1.7): `ca6e739`
  has neither `min_proposal_block` nor `cell_sources`. Both exist at `74e24d0` and
  at `172bcc9`. Moot here — the successor declares `172bcc9` and ships no such test —
  but recorded so the error is not repeated.

---

## 6. Handed forward

1. **PROJECT-WIDE, CONTROLLER ACTION — CLAUDE.md rule 9's ~300 GB/s is contradicted
   by direct measurement on the target machine, by at least 1.7×.** `T=1` KV reads
   measure **472 / 504 / 499 GB/s** at 32K / 128K / 256K, self-consistently, and the
   tiled intercept independently reproduces 493 GB/s. This constant is load-bearing
   in CLAUDE.md rule 9, the plan's roofline, Gate 0.5 (decision 004), and
   VERDICT-SC-P4DA §1.1 — the verdict that started this whole thread. Deliberately
   **not** edited here: a card does not amend the project's governing constants. It
   needs an operator decision and its own decision record.
2. **A second, unexplained bandwidth anomaly, found here and not previously
   reported.** The *vector* path's marginal slope implies **794 / 839 / 790 GB/s**
   at the three contexts if it read the full KV per query row — a stable **1.58–1.68×**
   above the same file's `T=1` rate. So either the vector path does not read the
   whole KV per row (the "one KV pass per query row" label is quantitatively wrong,
   as "reads KV once" was for the tiled path), or `T=1` carries a context-scaling
   overhead that is not launch cost. Unresolved. It bounds the true effective
   bandwidth somewhere in **504–840 GB/s**, which makes item 1 a 1.7–2.8×
   discrepancy, not 1.7×.
3. **The bytes-per-pass table in §3 is a conversion, not a measurement.** It assumes
   the kernel remains bandwidth-limited at the measured rate for `T > 6`. The rising
   marginal (0.27 ms/row at `T=6→12`, 1.72 at `T=48→96`) is consistent with more
   KV re-reads at larger tiles, but it is equally consistent with a compute term
   taking over. Resolving it needs one `powermetrics`/counter run, not more modelling,
   and it decides how much Phase 6 KV quant actually buys.
4. **The floor-only install shape** (§1, Link 4) is the successor-successor card, and
   it is a default flip gated behind the live 128K A/B — not a code-only change.
5. Everything SC-P4DB itself handed forward (POC §7 items 1–5) is unchanged and
   still open.

---

## 7. What SC-P4DB got right and is preserved

Stated because a fail-run that only condemns is not calibrated, and because all of
this survives into the successor:

1. **The `T=6` regime change is a genuine correction to VERDICT-SC-P4DA's
   roofline.** Building the microbenchmark at the engine's exact shapes instead of
   arguing from a roofline found something no further reasoning would have. The
   breaker attacked it four ways and it held.
2. **The aggregate-vs-marginal distinction is the correct frame**, even though the
   chord was picked and the units are wrong: the predecessor did apply a ratio where
   a derivative was needed.
3. **The dominance argument for the floor is model-free and correct.** It is not
   re-issued because it is inert, not because it is wrong.
4. **The scope fences** are the best-written section of any card in this project and
   are carried into SC-P4DC nearly verbatim.
5. **Acceptance criterion 8** — permitting the live result to contradict the
   projection and forbidding tuning it away — is the criterion the whole process
   wants and is carried forward.
6. **The writer disbelieved a 9.6 TB/s result and found the bug.** That instinct is
   worth more than the number it discarded.
