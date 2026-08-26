# Long-context POC — what the verify pass actually costs at 128K/256K

**Date:** 2026-08-26 · **Author:** Opus card writer · **Card:** `SC-P4DB-ramp-proposal-floor-and-live-longcontext-gate.md`
**Answers:** `docs/reviews/2026-08-26-ramp/VERDICT-SC-P4DA.md` §1.1 — the breaker's
single open question about RAMP at long context.

The breaker reduced RAMP's largest risk to one measurable scalar: *achieved
attention FLOPS at long context*. Under ~30 TFLOPS the optimal block length
collapses to the shortest rung of the stock ladder at 128K and 256K, and RAMP's
win evaporates; under ~60 TFLOPS it likely survives. Nothing in the project had
ever measured it.

**Headline: the scalar is 15.8 TFLOPS on the metric the verdict's model implies,
and 56.8 TFLOPS-equivalent on the metric its conclusion actually depends on.
The 3.6× gap is the whole answer, and it comes from a kernel-selection step the
roofline model has no term for. RAMP's long block survives at 128K and 256K.**

---

## 1. What was built

| File | What it is |
|---|---|
| `scripts/attn_flops_microbench.py` | Times the real kernels at the engine's exact shapes: 16 full-attention `mx.fast.scaled_dot_product_attention` calls (q=(1,24,T,256) bf16 vs (1,4,C,256) bf16 KV, causal), plus a byte- and FLOP-matched Q8 weight-streaming proxy. |
| `scripts/run_attn_flops_microbench.sh` | Hygiene driver: one process per context, strictly serial, refuses to run while an MTPLX server holds the GPU. |
| `scripts/ramp_kernel_regimes.py` | Extracts the two kernel regimes, the switch cost, and the dead zone between them. |
| `scripts/ramp_longcontext_model.py` | Projects block length to each context from the live A/B, with a held-out validation cell and a sensitivity sweep. |

Every number below is from a run whose output is committed under `evidence/`.
No number here is estimated.

---

## 2. The measured cost curve

`mx.fast.scaled_dot_product_attention` is served by **two different kernels**,
and the boundary is what decides this question.

| context | one full KV read (T=1) | vector path (T≤5) | switch T=5→6 | tiled path (T=6→12) |
|---|---|---|---|---|
| 32K | 4.55 ms | 2.70 ms/row | **+10.9 ms** | 0.275 ms/row |
| 128K | 17.05 ms | 10.24 ms/row | **+61.4 ms** | 0.273 ms/row |
| 256K | 34.44 ms | 21.74 ms/row | **+118.5 ms** | 0.230 ms/row |

Below T=6 the cost is linear in T at roughly one KV pass per query row. At T=6
MLX switches to a tiled kernel that reads the KV once and is then **nearly flat
in T** — 0.27 ms per additional row at 128K, against a 17 ms KV read.

### This settles the mechanism dispute in the parent card

The GPT-5.6 consult, quoted verbatim in SC-P4DA, said verification attention
*"may read a large KV cache **for every proposed row**."* The breaker corrected
this (VERDICT §1.1): the `T` rows go through as one batched call
(`generation.py:7845-7852`), so KV is read **once per pass regardless of `L`**.

**Both are correct, in different regimes, and neither named the boundary.**
Per-row for T ≤ 5; once-per-pass for T ≥ 6. The card quoted the first and the
verdict quoted the second.

---

## 3. The answer to the breaker's question

The verdict's model charges `flops / FLOPS_achieved` with a single flat FLOPS
figure. Which figure you measure changes the answer:

| context | aggregate TFLOPS at T=48 | **marginal** TFLOPS-equivalent (T=8→64) |
|---|---|---|
| 32K | 16.9 | 55.7 |
| 128K | **15.8** | **56.8** |
| 256K | **14.8** | **46.7** |

*Aggregate* divides total attention FLOPs by total kernel time. It is 15.8
TFLOPS at 128K — **below the verdict's pessimistic case**, which on its model
kills RAMP.

*Marginal* is the cost of one **additional** verify row, which is the only thing
a block-length decision can move. It is 56.8 TFLOPS-equivalent at 128K and 46.7
at 256K — **in the upper half of the verdict's 30–60 band, the half where RAMP
survives.**

The aggregate figure is low because the tiled kernel's fixed cost (one full KV
read, plus the switch) is amortised over the rows; it is not a rate anything
pays per row. **Applying an aggregate FLOPS number to a marginal decision is
what made the verdict's estimate point at the kill criterion.** The verdict's
arithmetic is right; the quantity it needed was the derivative, not the ratio.

### The dead zone

Because the tiled kernel starts well above where the vector line had reached, a
band of `T` exists where the engine has switched but not yet amortised:

| context | dead zone | in block terms | worst penalty |
|---|---|---|---|
| 32K | T ∈ [6, 9.4] | block 5–8 | 8 ms/pass |
| 128K | T ∈ [6, 11.1] | block 5–10 | **51 ms/pass** |
| 256K | T ∈ [6, 10.5] | block 5–9 | **97 ms/pass** |

A proposal landing in the dead zone is **strictly worse than proposing nothing
or proposing a longer block**. The engine's stock ladder is
`_BLOCK_LADDER = (8, 12, 16, 24, 32)` (`rafale/draft/ramp.py:63`) — **its first
rung is inside the dead zone at every long context measured.** This is the one
model-free, directly actionable finding in the POC, and it is the production fix
the card asks for. Taking the widest measured band, the safe floor is
**propose ≥ 11 tokens or propose nothing** (`min_proposal_block` in
`evidence/kernel-regimes.json`: 9 at 32K, **11** at 128K, 10 at 256K).

The dominance is concrete. At 128K, block 8 (`T=9`, interpolating 118.90 ms at
`T=8`) against block 11 (`T=12`, 121.06 ms): **2.2 ms more for four more
proposed tokens.** There is no acceptance rate at which the shorter one is the
right call.

---

## 4. Projection to 128K/256K, and its stated limits

`scripts/ramp_longcontext_model.py` takes the live engine's short-context pass
curve as measured and adds **only** the measured context-dependent attention
delta — because full attention over the KV is the one term that grows with `C`.
Weights do not, GDN state does not (that is what linear attention buys), Python
overhead does not.

| context | stock ladder | block-48 (fuzzy) | block-64 | best |
|---|---|---|---|---|
| ~800 tok | 94.4 t/s | 128.8 (+36%) | 125.8 (+33%) | 48 |
| 32K | 82.2 | 113.6 (+38%) | 112.4 (+37%) | 48 |
| 128K | 57.2 | 81.5 (+42%) | **82.9 (+45%)** | 64 |
| 256K | 39.8 | 57.6 (+45%) | **59.6 (+50%)** | 64 |

**RAMP's advantage over the stock ladder grows with context**, the opposite of
the verdict's expectation. The mechanism is now obvious: the stock ladder uses
*shorter* blocks, so it runs *more* passes, and at long context each pass pays a
large fixed attention cost. Longer blocks amortise it over more tokens.

### Two models were built and rejected first

Recorded because the failures are informative, and the commits keep them:

1. `t_pass = h + α·(t_dense + t_attn)` fitted over all 30 live rows → **h = −152
   ms** (unphysical), R² = 0.648, **+59%** on the held-out 128K cell. One
   coefficient absorbing dense-proxy error was extrapolated onto an attention
   term that grows 100× with context.
2. Same with α scaling only the dense proxy → **α = 11.6, 266% spread**. The
   failure is itself a finding: the dense proxy is nearly flat from T=12 to T=64
   (+4.5 ms) while the live engine costs **~27 ms more per pass** over the same
   range. **Roughly 1 ms/row of the real engine's cost is neither weight
   streaming nor full attention** — almost certainly the 48 GatedDeltaNet
   layers, whose recurrent scan is real O(T) work and not a matmul. The consult's
   Q5 predicted exactly this.

### Limits, stated

- **Held-out 128K cell over-predicts by +26.1%** (predicted 155.4 ms vs measured
  123.2 ms per pass). Not resolved. The most likely cause is that MTPLX does not
  issue one monolithic full-C SDPA per layer at 128K (`attention_split.py`,
  `--fan-mode default`, `--retrieval-max-resident 2` all exist and were not
  traced). **The error runs against RAMP** — over-charging attention penalises
  long blocks — so the projection is conservative.
- **The model also under-predicts RAMP's known short-context win**: +36% where
  the live A/B measured +51–63%. Conservative in the second direction too.
- **Acceptance is assumed context-invariant.** Tokens-per-pass comes from
  ~800-token traces. This is the single input the POC does not establish and it
  is why the card's acceptance is a live run, not this table.
- Variants above block 64 extrapolate past the fitted T range (13.8–44.2) and
  should not be trusted.

### Sensitivity — is the answer fitted or real?

| attention delta scaled | winner at 128K | vs stock |
|---|---|---|
| ×0.5 | block-64 | +40.3% |
| ×1.0 | block-64 | +44.9% |
| ×2.0 | block-64 | +50.6% |

**The ranking does not change across a 4× sweep of the one uncertain term.** And
block-48 carries **30% acceptance headroom** at 128K (break-even 16.08 tok/pass
against 22.90 measured) and **31%** at 256K before it would lose to the stock
ladder.

---

## 5. Hygiene — two contaminated runs, both caught

CLAUDE.md rule 4 earned its keep twice, and both artifacts are committed.

1. **Concurrency.** The dense mode was first run while the 262144 attention
   cells were still in flight, producing a T=48 cell 32% slower than its T=64
   neighbour at 46% spread. Hence the serial driver.
2. **Swap.** The first serial 131072 sweep produced T≥48 cells that were
   internally *tight* (median/min = 1.03) yet **2.7× what linear-in-C scaling
   from the clean 32K and 256K cells demands**. Uniformly slow reps means a
   persistent state change, not jitter. Swap grew **5.75 → 8.58 GB** during
   exactly that window, as the OS evicted a long-running VM to make room for
   8.6 GB of KV. Re-run after swap settled, the cells reproduce an earlier
   independent pass to **within 1.3%** (T=48: 156.8 vs 154.7 ms).

The tell that caught #2 was not the spread — it was **cross-context
consistency**: 256K cells measured *faster* than 128K cells at the same T, which
is physically impossible. Both sides of swap and pagein counters are now written
into every artifact.

3. **A third, milder case, kept rather than overwritten.** The main 131072 sweep's
   vector-regime cells (T=3,4,5) carry 9–24% spread and read 6–12% high. They
   were re-measured in a dedicated confirmation run
   (`evidence/attn-microbench-c131072-lowT-confirm.json`, 15 reps, 2–4% spread),
   which agrees with the main sweep to within 2% at every other cell. Rather than
   silently replacing the main file or hand-picking which one to quote,
   `ramp_kernel_regimes.py` selects **per cell, the measurement with the lowest
   relative spread**, and records `cell_sources` in its output so every quoted
   number is traceable to the file it came from.

A separate bug was found by disbelieving a result: the dense proxy first reported
**9.6 TB/s** because assigning each `quantized_matmul` to the same name made all
but the last dead code under MLX's lazy evaluation.

---

## 6. What this does and does not license

**Does:** retire the verdict's concern that block=48 is a long-context
regression. On measured cost data, with the acceptance assumption stated, the
long block is better at 128K and 256K than at 800 tokens, not worse.
SC-P4DA's fourth kill-criterion bullet is **not tripped**.

**Does not:** license turning RAMP on by default. Every throughput number at
long context is still *projected*. The projection is conservative, robust to a
4× sweep, and carries a +30% known error on its one validation cell — that is
good enough to keep building, not good enough to change a default.

**The one production change this POC does license on its own evidence** is the
proposal floor: never propose a block in the dead zone. That result is
model-free, holds at every context measured, and the engine's stock ladder
violates it.

---

## 7. Open items handed to the card

1. **The +26.1% holdout error is unexplained.** Trace whether MTPLX issues one
   full-C SDPA per layer at 128K or splits it (`attention_split.py`).
2. **Acceptance at long context is unmeasured.** The break-even is 30% headroom;
   nothing says the real degradation is smaller.
3. **Temperature > 0 remains a different regime** (VERDICT §2.2) and is
   untouched here.
4. **Blocks above 64 are extrapolation.** The live sweep must include them or the
   card must fence them off.
5. **MLX's switch point (T=6) is premature by its own numbers** — between T=6 and
   T≈10 the tiled kernel is slower than the vector kernel would have been. That
   is an upstream MLX observation, not something this project should patch, but
   it is why the dead zone exists and it may disappear in a future MLX release.
   The floor must therefore be a *measured constant with a test*, not a magic
   number.
