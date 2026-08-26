# 007 — RAMP's long block survives at 128K/256K on projected evidence; the default stays off

**Date:** 2026-08-26 · **Amended:** 2026-08-26 (see *Amendment log*)
**Phase:** 4D (RAMP)
**Gate:** SC-P4DA kill criterion, bullet 4 — *"the long-context optimum collapses
to the stock ladder"*
**Outcome:** **NOT TRIPPED on projected cost evidence; confirmation pending the
live 128K A/B.** Default remains off.

> **SUPERSEDED IN PART by `008-ramp-live-128k-ground-truth.md` (2026-08-26).** The
> live 128K A/B this record hedges on has now run: RAMP `block=48` beats the stock
> ladder by **+45.9 %** at 128K, output is byte-identical, suspensions are zero,
> and acceptance degrades **−5.4 %** in tokens-per-pass against a stock ladder
> measured exactly fixed. Item 1's conclusion holds and its "projection, not a
> measurement" caveat is discharged. One correction: this record's expectation of
> an advantage that *grows* at long context is **not** supported — it mildly
> shrinks (+51.3 % at 800 tokens → +45.9 % at 128K). **Items 4 and 5 and the
> *Handed to the operator* section remain refuted by `VERDICT-SC-P4DC.md`
> §§1.1–1.3 and uncorrected; decision 008 makes them moot for the gate, not
> correct. Do not cite them.**
**Evidence:** `docs/reviews/2026-08-26-ramp-longcontext/` (POC-FINDINGS.md,
`evidence/`), breaker verdict `VERDICT-SC-P4DB.md`, fail-run analysis
`FAILRUN-SC-P4DB.md`
**Supersedes nothing.** Answers `docs/reviews/2026-08-26-ramp/VERDICT-SC-P4DA.md`
§1.1 and its blocking edit 5.

> **Read the amendment log before citing this record.** The version committed at
> `74e24d0` recorded a flat "NOT TRIPPED", asserted the tiled attention kernel
> reads the KV once per pass, and concluded from that assertion that Phase 6 KV
> quantization is *not* the long-context rescue path. The mechanism was refuted by
> its own evidence and **item 5 has been reversed**. Anything downstream that cites
> the original item 5 must be rechecked.

---

## The question

RAMP shipped (`5eaceb3`) measured entirely on ~800-token prompts, against a plan
targeting 128K–256K. The breaker derived a roofline showing the outcome hinged on
one measurable scalar — achieved attention throughput at long context — and that
roughly half the plausible range killed the card. Under ~30 TFLOPS the optimal
block length collapses to the shortest rung of the stock ladder; under ~60 TFLOPS
it survives.

**That framing is now known to be mis-specified**, and the correction is item 4
below: the kernel whose cost decides this is bandwidth-bound, not compute-bound, so
no achieved-FLOPS scalar is the right quantity to compare against a compute
roofline. The gate is decided instead by the measured cost curve, which the
projection consumes directly.

## What was measured

`scripts/attn_flops_microbench.py`, at the engine's exact shapes, on the target
machine. `mx.fast.scaled_dot_product_attention` is served by two kernels: a
per-row vector kernel below `T=6` and a tiled kernel at and above it. The step is
real — +102% at 128K against 2.9–13.0% cell spread, replicated in two independently
launched processes, non-monotone in `T` (ruling out drift), and corroborated by an
identical step in a dense `quantized_matmul` proxy where no KV cache exists at all.
The breaker attacked it four ways and it held.

**Marginal TFLOPS-equivalent, all four committed chords** (the original record
quoted only `8→64`):

| chord | 32K | 128K | 256K |
|---|---|---|---|
| 8 → 64 | 55.7 | 56.8 | 46.7 |
| 12 → 48 | 53.5 | 51.9 | 40.2 |
| 6 → 96 | 40.0 | 38.7 | 33.8 |
| 48 → 96 | 33.1 | 30.0 | 27.4 |
| 64 → 96 | 26.3 | 23.7 | 22.2 |
| OLS over all `T ≥ 6` (R² 0.94–0.96) | 43.2 | 42.1 | 36.6 |

Aggregate TFLOPS at `T=48` is 15.8 (128K) and 14.8 (256K). The marginal is
**regime-dependent and rising**, not "nearly flat in `T`" — 0.27 ms/row at
`T=6→12`, 1.72 ms/row at `T=48→96` (128K). At the projection's own winner
(block-64) the local slope is 23.7 / 22.2, **below** the predecessor's 30-TFLOPS
line, not in the upper half of a 30–60 band.

**None of these numbers decides the gate**, and that is the point of item 4: they
are byte:FLOP diagnostics for a bandwidth-bound kernel. The projection
(`ramp_longcontext_model.py:280`) interpolates the **measured** attention curve and
uses no FLOPS scalar at all, which is why its ranking survives the correction.

## Decision

1. **Bullet 4 is not tripped on projected cost evidence.** Block-64 beats the stock
   ladder at 128K and 256K across a 4× sweep of the attention term **and** across
   the full 2× range of `b`, the live per-row slope the POC did not sweep (+39.9% to
   +48.7% at 128K). The stock ladder ranks 13th of 16 in every cell of the
   sensitivity sweep. Eight independent parameterisations, one winner. **This is a
   projection, not a measurement**; confirmation requires the live 128K A/B, and
   this record must not be cited as a cleared gate until that runs.
2. **RAMP's default stays off**, at every context. The projection carries a
   **+26.1%** error on its one held-out validation cell — a cell at `T=3`, inside
   the *vector* regime, which constrains nothing about the tiled regime every long
   block operates in — and assumes acceptance behaviour measured only at ~800
   tokens. Its accepted model has **no R²** and cannot have one as structured; its
   only quality signal is a **67.4%** cross-case spread on its sole `T`-dependent
   coefficient. Good enough to keep building, not good enough to change a default.
3. **No proposal floor is implemented, and none is licensed today.** The dead-zone
   dominance argument is real and model-free — at 128K, block 11 costs 1.8% more
   than block 8 and proposes ~38% more tokens, and the engine's stock ladder
   (`_BLOCK_LADDER = (8, 12, 16, 24, 32)`) has its first rung inside that band. But
   a floor applied in `_installed_block_for_ext` binds in **zero shipped
   configurations** (`FAILRUN-SC-P4DB.md` §1, Link 4): RAMP-off does not install it,
   RAMP-on ships `block=48` which bypasses the ladder, and the only configuration
   that consults the ladder is the A/B's *control arm* — where a floor would
   contaminate the control. The one shape in which the floor is a genuine production
   fix is a floor-only install (`enabled=True, block=None, fuzzy=False`), which is a
   **default flip** and is gated behind the same live A/B as item 2. Recorded as a
   measured constraint on all future drafting work, implemented nowhere.
4. **The mechanism recorded in SC-P4DA is corrected — twice.** The consult's *"reads
   a large KV cache for every proposed row"* and the verdict's *"read once per pass"*
   are each right on one side of a boundary neither named, and the boundary is `T=6`.
   But the `T ≥ 6` label is also wrong: the tiled kernel's cost intercept is **6.13×
   one measured KV read** at 128K (6.11× at 256K, 5.16× at 32K), against
   `num_attention_heads / num_key_value_heads = 24 / 4 = 6`. **The tiled kernel reads
   the KV once per query head group** — it does not exploit GQA head sharing; the
   vector kernel does. Cross-checked: 8.59 GB in 17.05 ms = 504 GB/s at `T=1`;
   6 × 8.59 GB / 104.6 ms = 493 GB/s. The verify pass at long context is therefore
   dominated by a **fixed byte cost that no block-length choice moves**, and the
   TFLOPS framing of the original gate does not apply to it.
5. **REVERSED — Phase 6 KV quantization is the largest measured long-context lever
   in this data.** The original item 5 stated the opposite, deriving it from the
   refuted once-per-pass mechanism. Converting measured attention time at the
   measured KV-read rate, against decision 003's 64 KB/token and the plan's ~28 GB
   of Q8 weights per pass:

   | context | KV bytes per verify pass | vs weight bytes |
   |---|---|---|
   | 128K, `T=3` (no proposal) | 19 GB | 0.7× |
   | 128K, `T=49` (block 48) | 79 GB | 2.8× |
   | 256K, `T=49` (block 48) | 168 GB | 6.0× |

   At the target contexts a verify pass moves **2.8–6.0× more KV bytes than weight
   bytes**, inverting this project's short-context "decode is weight-streaming
   bound" framing. Q8 KV halves the dominant term: ~40 GB/pass saved at 128K
   block-48, ~84 GB/pass at 256K. *Caveat, stated because the conversion is not a
   measurement:* it assumes the kernel stays bandwidth-limited above `T=6`. The
   rising marginal is consistent with more KV re-reads at larger tiles and equally
   consistent with a compute term taking over. One counter run decides it, and it
   decides how much Phase 6 actually buys.
6. **The gate record was committed before the measurement its own card mandates.**
   The original was written at `74e24d0`, with SC-P4DB's acceptance criteria 7–8
   still requiring a live 128K A/B and criterion 8 explicitly permitting it to
   contradict the projection. CLAUDE.md rule 4 and the phase Definition-of-Done both
   require hygiene-protocol cells before a gate is recorded. The outcome line is
   hedged accordingly and stays hedged until the A/B runs.

## What would reverse this

- A live 128K A/B showing the long block losing to the stock ladder. That trips
  bullet 4 after all and needs its own decision record.
- Acceptance degrading at long context by more than ~30% of block-48's
  **tokens-per-pass**, *with the stock ladder's acceptance held fixed*. Stated
  precisely because the original wording ("acceptance degrading more than ~30%")
  described uniform degradation, which scales both arms' numerators identically and
  **cannot reverse the ranking**. The margin is against *differential* degradation —
  long blocks degrading while short ones do not — which is the plausible mode but a
  different quantity from the one the number was computed on.
- An MLX release moving or removing the `T=6` kernel switch, or a release in which
  the tiled kernel exploits GQA head sharing. Either would invalidate the dead-zone
  constants and much of item 5's magnitude.
- A counter run showing the tiled kernel is not bandwidth-limited above `T=6`,
  which would shrink item 5's estimate of what KV quant buys.

## Handed to the operator — outside this decision's scope

**CLAUDE.md rule 9 and the plan fix effective DRAM bandwidth at ~300 GB/s. This
POC measures 472 / 504 / 499 GB/s directly, at 32K / 128K / 256K, on the target
machine**, self-consistently and corroborated by the tiled intercept at 493 GB/s.
The vector path's marginal slope implies 790–839 GB/s if it read the full KV per
row — a stable 1.58–1.68× above the same file's `T=1` rate, unexplained
(`FAILRUN-SC-P4DB.md` §6 item 2). So the true effective figure is somewhere in
**504–840 GB/s** and the committed constant is low by **1.7–2.8×**.

That constant is load-bearing in CLAUDE.md rule 9, the plan's roofline, Gate 0.5
(decision 004), and VERDICT-SC-P4DA §1.1 — the verdict that began this thread.
Deliberately not amended here: a phase decision record does not rewrite the
project's governing constants. It needs an operator decision and its own record.

## Hygiene note

Two measurement runs were invalidated during this POC. The second is the one worth
remembering: cells that were internally *tight* (median/min = 1.03) but 2.7× wrong,
contaminated by a swap storm. No spread-based test detects that. What caught it was
cross-context consistency — 256K cells measuring faster than 128K cells at the same
`T`, which is physically impossible. This is the same failure class as the
invalidated block sweep in decision 006's lineage, and it argues for
physical-consistency checks alongside spread checks in the benchmark harness
generally.

**But the lesson is recorded on artifacts that do not exist.** POC-FINDINGS §5
claims both invalidated runs are committed; **neither is**. Eleven files have ever
existed in `evidence/` and none is an invalidated run, so the swap-storm diagnosis
**cannot be checked by anyone**, including the breaker who tried. §5 further claims
swap counters are "now written into every artifact" (they exist only in the two
`c131072` files) and that the re-run happened "after swap settled" (`vm.swapusage`
reads 93% full in *every* committed file, clean ones included). The diagnostic
instinct was right; the audit trail behind it was not kept. SC-P4DC corrects the
record and requires any invalidated run of its own to be committed.

## Amendment log

**2026-08-26 — amendment 1**, applying `VERDICT-SC-P4DB.md` blocking edits A, B, E,
I, K, L and `FAILRUN-SC-P4DB.md` §§1, 3, 4, 6:

| # | change |
|---|---|
| A | Outcome line hedged to "on projected cost evidence; confirmation pending the live 128K A/B." Full four-chord table published; "upper half of the 30–60 band" removed; marginal restated as regime-dependent and rising. |
| B | Item 4's mechanism corrected to 6.13× GQA head-group amplification. **Item 5 reversed**: Phase 6 KV quant goes from "not the rescue path" to the largest measured long-context lever, with a bytes-per-pass table and its stated assumption. |
| C | Item 3 rewritten: the proposal floor is **not** implemented and not licensed — it binds in zero shipped configurations. SC-P4DB's production fix is withdrawn as INVALID_CARD. |
| D | Item 2 gains the holdout cell's vector-regime limitation, the accepted model's absent R², and `b_spread = 67.4%`. |
| E | *What would reverse this* restates the ~30% headroom as **differential** degradation in tokens-per-pass, and notes uniform degradation cannot reverse the ranking. |
| F | Hygiene note extended: three of its own claims are false against the repository, and the diagnosis it teaches is unverifiable. |
| G | New *Handed to the operator* section: the 300 GB/s vs 504–840 GB/s discrepancy against CLAUDE.md rule 9. |
| H | Item 6 added: this record was committed before the measurement its own card mandates. |

Superseding card: `docs/reviews/2026-08-26-ramp-longcontext/SC-P4DC-longcontext-record-correction-and-deadzone-grid.md`.
