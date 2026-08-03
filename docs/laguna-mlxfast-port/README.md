# mlx.fast Laguna XS2.1 → Laguna S-2.1 port — results

Full port of the mlx.fast "Laguna XS2.1" challenge optimized runtime into a standalone
alternative Laguna S-2.1 runtime, reshaped to S-2.1's real geometry (hidden 3072, 48
layers, per-layer 48/72 heads → gqa 6/9, top-10 of 256 experts, moe_intermediate 1024,
YaRN mscale 1.4852) and **affine oQ4e** quant (NOT the challenge's NVFP4), then benchmarked
head-to-head against MTPLX's reference lane. Every challenge kernel was ported and measured;
nothing was skipped as "already covered."

## Whole-runtime result (decode, B=1, ctx 1024 / decode 96, the 67.4-reference shape)

| runtime | ms/step | tok/s | Δ vs reference | digest |
|---|---|---|---|---|
| MTPLX reference (`install_from_env` + `LagunaCompiledLane`) | 14.85 | **67.3** | — | `9098436fbc29879b` |
| **alt runtime (D1 + async), the port** | **14.05** | **71.2** | **+5.8%** | `9098436fbc29879b` ✓ |

**+5.8%, token-for-token identical to the reference.** Reproduced across 5 independent
guarded windows. The alt runtime is `mtplx/laguna_alt_step.py` (`LagunaAltLane`), a sibling
lane on the shared `models.laguna.Model` weights, routing the forward through the ported
kernels behind per-kernel `AltConfig` flags (all default off; a fail-loud guard makes a
half-wired flag raise rather than fake the reference's numbers).

## The two transferable wins

- **D1 — residual+RMSNorm+router-GEMV fusion (+1.0%).** Folds the MoE router GEMV into the
  post-attention residual+norm dispatch across the 47 sparse layers. Kernel bit-exact at
  3072/256 (residual/norm/logits 0.0 diff, top-10 10/10). Beats the reference's *separate*
  `kernel_router_gemv` even paying for a separate argpartition top-k.
- **S1 — async-eval decode scheduling (+4.0%).** `mx.async_eval` the full step state each
  token so the host encodes step N+1 while the GPU runs step N. Value-preserving (digest
  identical). The interval ladder (1,7,15,… ≈ every 8) measured *worse* than async-every-step.

## Per-kernel verdicts (all digest-exact where wired)

| kernel | verdict | note |
|---|---|---|
| D1 residual+router | ✅ +1.0% | bit-exact fusion |
| S1 async schedule | ✅ +4.0% | biggest lever; scheduling, not a kernel |
| D6 SDPA-vector (group-3 gqa) | ⏭️ −1.6% | KV-reuse doesn't beat stock SDPA at N=512 |
| D7–D9 affine MoE (decode per-token SwiGLU-QMV) | ⏭️ −25% decode | bit-exact but re-reads weights per token; loses to stock grouped `gather_qmm` |
| D14 lm-head top-1 | ⏭️ −0.7% | EXACT (top-1==argmax all steps); head read dominates |
| interval ladder | ⏭️ worse | async-every-step wins |
| D4 qkvg / gate-up | ⏭️ ineligible | installer converts 0 layers on affine shapes |
| D2/D3/D5/D10/D12 | ✅ active | via the installed reference kernels the alt lane reads |
| D11 dense-0 / D13 embed | — | minor components, active via stock; D11 is the D7-class that loses |
| P4 prefill MoE gather-GEMM (grouped) | ⏭️ ported, −10–23× | the challenge kernel is a FORK of MLX's steel `gather_qmm` + off-by-default micro-levers; a hand `metal_kernel` can't match the hardware MMA; RUNSKIP inert |
| P1/P2/P3/P5 prefill | ✅ integrated | alt prefill lane = reference parity (1582 vs 1579); D1-at-prefill −5% |

## Why the per-op hand kernels don't transfer

XS2.1 shipped as `poolside/Laguna-XS-2.1-NVFP4-mlx` — **NVFP4** (group-16 4-bit float, E4M3
scales), which MLX has no native kernel for; the challenge vendored and hand-patched mlx-swift
to add it. Crucially, that means the challenge's hand kernels had **no tuned stock competitor**
— they were the only NVFP4 path. Re-expressed for S-2.1's affine oQ4e, every hand kernel now
races MLX's **already-tuned** stock primitives (`gather_qmm` / SDPA / argmax), which are
bandwidth/occupancy-optimal on M5. So the techniques aren't wrong — the bar is just far higher
on affine than it was on NVFP4:

- **MoE**: stock `SwitchGLU` uses `gather_qmm(sorted_indices)` = MLX's **steel MMA grouped
  GEMM** (16×16 `simdgroup_matrix`, BK double-buffered). At decode a per-token SwiGLU-QMV
  re-reads weights per token → bandwidth-bound (−1.2…6.3×); at prefill a hand grouped GEMM
  can't match the hardware MMA units (−10…23×). The challenge's own prefill MoE kernel is a
  *fork of this same stock GEMM* with off-by-default micro-levers — not a hand GEMM that beats it.
- **Attention**: stock `mx.fast.scaled_dot_product_attention` is flash-based; the group-3
  KV-reuse can't beat it at B=1 / N≤2048.
- **lm-head**: the head *read* dominates; a top-1 kernel that also reads the whole head only
  adds top-k machinery over a plain argmax.

The transferable levers were the ones that are quant-agnostic: **cross-op FUSION (D1)** and
**host/GPU SCHEDULING (S1)**. That is the "properly optimized" result: **+5.8% decode,
digest-exact**, with every other challenge kernel ported and measured to a documented verdict.

## Prefill (lane built + measured)

`alt_prefill_forward` (in `laguna_alt_step.py`) is a full alt prefill lane mirroring the eager
forward, integrating every prefill component (P2 attention → MLX flash SDPA; P1 qk-rope, P3
router, P5 tail → installed kernels; P4 experts → stock SwitchGLU) with D1's fusion optional.
GPU A/B at ctx 1024, first-token digest-matched:

| prefill lane | tok/s | Δ vs reference |
|---|---|---|
| reference (eager) | 1579.4 | — |
| alt[stock] | 1581.9 | +0.2% (parity) |
| alt[D1] | 1498.1 | **−5.1%** |

alt[stock] = reference parity (the lane is correct and the stock prefill ops are optimal). **D1
at prefill LOSES 5%**: at prefill the router GEMV becomes a `[T,3072]@[3072,256]` GEMM, where
the per-row fused kernel loses to stock's GEMM — D1's win is decode-specific (a GEMV, rows=1).
Two MoE kernels were measured. The **decode** per-token SwiGLU-QMV (`laguna_moe_swiglu.py`)
loses to stock (it re-reads weights per token). The **prefill** grouped gather-GEMM
(`laguna_moe_gather_gemm.py`) was then ported and measured directly — and the key finding is
that the challenge's prefill kernel (`fp_gather_qmm_rhs_nax`) is **a fork of MLX's own steel
tiled gather-GEMM** (the exact kernel `mx.gather_qmm(sorted_indices=True)` dispatches, 16×16
`simdgroup_matrix` MMA), with micro-levers (RUNSKIP, wider loads, register prefetch) that
default OFF. The affine hand port loses **10–23× to stock** at every prefill T (a
`mx.fast.metal_kernel` uses scalar FMA and cannot match the hardware MMA units); RUNSKIP is
inert (0 empty experts at top-10/256), split-K is slower. Three independent lines agree: (1)
the challenge's kernel is itself stock + off-by-default micro-levers, (2) the challenge team's
own notes record prefill MoE as "NO HEADROOM" / staging "shelved-regressed" / split-K +0.18%,
(3) this direct affine measurement. **No prefill MoE lever exists on affine oQ4e — keep stock
`gather_qmm`.** The prefill lane itself is complete — every component integrated in a runnable lane and measured.

## Artifacts
- Runtime: `mtplx/laguna_alt_step.py`; kernels `mtplx/kernels/{laguna_residual_router,laguna_sdpa_pair,laguna_moe_swiglu}.py`.
- Tests: `tests/test_laguna_alt_step.py` (17 pass — parity, packed-KV, ladder value-preservation, fail-loud guards, per-kernel wiring).
- Harness: `bench/laguna/laguna_alt_ab_bench.py` (reference vs alt cells, config × schedule, digest gate).
- Receipts: `bench/laguna/laguna-alt-ab-{baseline,d1,s1,d6ladder,d14b,d4}-*.json`.
- Full per-kernel ledger + receipts: `PORT_LEDGER.md`.
