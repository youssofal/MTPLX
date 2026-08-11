# Muse-Glimmer-30B (q4) text-model optimization ledger

Measure-first, A/B each candidate against stock decode (same weights, same
shapes, in-window back-to-back so thermal drift doesn't confound), keep wins,
reject losers with evidence. GPU work under the serialized MLX window.

## Profile (q4, M5 Max, B=1 decode)

- decode **26 tok/s (38.5 ms/tok) = 82% of the 36.6 tok/s bandwidth roofline** (16.75 GB read/token).
- per-component census: **MLP 77%**, o_proj 7%, lm_head 6%, attn gate_proj 3%, q+kv 5%, norms 1%.
- isolated GEMM sum ≈ 28.3 ms; the remaining **~10 ms/tok (26%)** is SDPA + per-layer glue (cache/rope/gate-mul/softcap) + B=1 host dispatch — the schedulable headroom.

## Verdicts

| # | candidate | verdict | evidence |
|---|-----------|---------|----------|
| 1 | **async scheduling** | ✅ already in stock | `mx.async_eval` throughout `batched_decode.py`; MTPLX stock decode already overlaps dispatch (the Laguna S1 +4% lever is already captured) |
| 2 | **QKVG fusion** | ✅ **WIN +4.8%, integrated** | fuse q/k/v/gate → one `quantized_matmul`; **bit-exact** (max\|Δ\|=0), 208→52 launches/token; in-window A/B **26.10 → 27.35 tok/s**. Now the default path in `vendored_muse_glimmer_text.Attention` (id-cached lazy fuse, off the param tree). NB: this *mlx-level concat* beats stock, unlike the Laguna hand-kernel qkvg which was ineligible on affine. |
| 3a | **MLP gate/up fusion (naive mlx-concat)** | ❌ −0.6% | *Not a valid rejection* — mlx-level concat, not a shape-optimized kernel. Superseded by 3b. |
| 3b | **MLP dense-SwiGLU fused KERNEL (shape-optimized)** | ✅ **+5.2%**, quality-parity | Laguna `dense_swiglu_qmv` (in-kernel affine dequant, gate+up+silu+mult fused, per-output-element row-owned) at Glimmer's exact gs32/4-bit/6656/19968. Decode 26.58 → 27.96. **Not** bit-exact (5.86e-3, FP accum-order), but HumanEval **28/40 vs 27/40 stock = parity**. Env-gated `MG_MLP_KERNEL`. Proves a shape-optimized kernel beats stock at M=1 where the naive concat lost. |
| 4 | **qk-norm+rope kernel (row-owned)** | ✅ **+4.8%, bit-exact** | Laguna `fused_qk_rope_sliding` at Glimmer's `SlidingRopeSpec` (32q/2kv, hd128, θ=500000, param-free norm via q_weight=3.87·ones/k_weight=ones, eps1e-5), sliding layers only (globals NoPE). max\|diff\|=0 on q and k. Decode 26.59 → 27.86. Win is **dispatch** (4 kernels × 39 layers → 1 each; host-encode lag), not GPU-exec. Earlier "reject by analogy" was invalid. |
| 5 | **gated-o_proj kernel (row-owned)** | ✅ **+5.1%** isolated | Custom kernel (MLP-`down` pattern + sigmoid-gate fused into the input read), Glimmer 4/5-bit gs32. Decode 26.51 → 27.85. Non-bit-exact (3.12e-2; in-kernel sigmoid) — would need a HumanEval gate, but ~0 stacked so not pursued. |

## Stacking: the wins SATURATE (do not add)

All four are **dispatch / host-encode** wins competing for one fixed budget. Stacked (QKVG + MLP-kernel + qk-rope), decode = **27.86 tok/s ≈ any single kernel** (MLP-alone was 27.96). So the bankable win is **~+7% over raw stock (~26.0 → ~27.9)** from **QKVG (bit-exact, default) + the MLP kernel (quality-parity, env-gated)**; qk-rope and gated-o_proj are genuine +4.8–5.1% *isolated* wins but ~0 marginal on top.

**Method lesson (David's correction, 4/4 vindicated):** you cannot reject a fusion by a naive mlx-concat or by analogy — only by benchmarking a kernel **tiled+fused for the model's exact shapes**. Every candidate rejected the wrong way became a real +5%-class isolated win, because at M=1 decode the bottleneck is stock `qmm`'s small-T ramp + per-op host-encode lag, which an in-kernel-dequant row-owned kernel beats. The *new* structural lesson is that these wins saturate against a fixed dispatch budget rather than compounding.

## Salvage benchmarks (2026-08-10, in-window guarded)

| test | result | verdict |
|------|--------|---------|
| **gated-o_proj stacked on QKVG+MLP** | QKVG 26.72 → +MLP 28.02 → +MLP+gated-oproj **27.95 (−0.3%)** | ❌ no salvage — confirms the budget is fully saturated by QKVG+MLP; gated-o_proj's isolated +5.1% was pure dispatch, zero marginal here. Dead. |
| **MLP kernel at prefill (L=512)** | stock qmm **932.4** → MLP kernel **896.3 (−3.9%)** | ⚠️ the MLP kernel is a **decode-only** win. At L=512 the regime is compute-bound (large-T qmm is optimal), the M=1-tuned row-owned kernel loses. **Gate `MG_MLP_KERNEL` to L==1** so prefill keeps stock qmm. |

Net after salvage: the bankable win stands at **QKVG (bit-exact, always-on) + MLP kernel (decode-only, L==1-gated) ≈ +7% decode over raw stock**. gated-o_proj and qk-rope add nothing on top (saturated). The MLP-kernel gate is now **`MG_MLP_KERNEL` AND L==1**, not unconditional.
| 5 | **lm_head 5→4-bit** | ⚠️ not worth | ~1% byte win; lm_head is deliberately 5-bit for output fidelity (quality-gated). |

## Net

One integrated win: **QKVG fusion, +4.8%, bit-exact.** MLP and SDPA are already
stock-`qmm`/flash-optimal and are intentionally **not** hand-kerneled. The
remaining decode gap is bandwidth (MLP, quant-gated) + B=1 dispatch (already
async-overlapped in stock).
