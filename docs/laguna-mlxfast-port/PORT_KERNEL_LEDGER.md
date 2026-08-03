# Laguna S-2.1 port — completeness matrix (every challenge kernel ported + tested)

This replaces the earlier "active via installed reference kernels" hand-wave. Every
challenge optimization is now ported as the **challenge's own implementation, reshaped to
S-2.1's geometry** (not MTPLX's installed kernel, not "argued-equivalent"), CPU-validated
bit-exact/allclose against a pure-mx reference and the stock module, and measured under the
GPU flock. Δ column filled from the flock window (`f04_batch_check_runner.py`, one guarded
window, 13 checks).

## S-2.1 shape corrections the ports had to make (vs the challenge's XS2.1 geometry)

These are the concrete "customized to the Laguna S shape" changes — each was a real
divergence from the challenge Swift, caught during the port:

| kernel | XS2.1 (challenge) | S-2.1 (this port) |
|---|---|---|
| D2 YaRN mscale | 1.3465… (= factor 32) | **1.4852030263919618** (0.1·ln(128)+1, = attention_factor) |
| D2 rot dims | — | 64 of 128 (partial-rotary 0.5), θ=500000, factor 128 |
| D3 sliding heads | 64 q | **72 q** / 8 kv, full-rotary 128, θ=10000 |
| D5 o_proj quant | group-32 INT8 only | **affine gs64, 5- and 8-bit** (all S-2.1 layers), inline gate (occupancy) |
| D9 merged slots | top-8 → 9 slots | **top-10 → 11 slots** (10 routed 4-bit + 1 shared 8-bit) |
| D11 dense layer-0 | plain BF16, 2048/8192 | **mixed affine** (gate/up 5-bit, down 6-bit, gs64), 3072/12288, 48 layers |
| D12 router sort | top-8 bitonic | **top-10** Batcher bitonic over 256, +pow2-experts guard (latent bug fixed) |
| D10 combine reduce | in-order bf16 (exact at k=8) | **TY=8 col_reduce order** to stay bit-exact at k=10; scale 2.5 pre-baked upstream |
| P1 batched rope | — | length-T positions vector (batched-rope offset trap), mscale 1.4852 |

## Port + test status — MEASURED (flock window `bd7tgynzj`, 2026-08-02)

Decision lane = **chained** (B=1 serial decode link) for decode, **queued** for prefill.
For the `port | mtplx | stock` triples the numbers are µs/call → **lower is faster**.

| id | kernel | GPU correctness | isolation speed (decision lane) | verdict |
|---|---|---|---|---|
| D2 | qk-norm + YaRN rope (full) | **bit-exact** vs stock | chained 25.82 vs stock 19.57 | ❌ LOSS 0.76× |
| D3 | qk-norm + rope (sliding) | **bit-exact** vs stock | chained 17.01 vs stock 5.73 | ❌ LOSS 0.34× |
| D4 | input-norm + QKV + gate | 8-bit allclose ✓; **5-bit diverges** vs stock (0.11) | queued 0.534× | ❌ LOSS + 5-bit gap |
| D5 | gated o-proj | 8-bit allclose ✓; **5-bit allclose FAIL** (0.09) | chained 252–391 vs stock 40 | ❌ LOSS ~6–10× |
| D7 | shared-expert SwiGLU-QMV | ALL PASS | queued x0.48–0.70 | ❌ LOSS |
| D9 | 11-slot merged SwiGLU-QMV | ALL PASS | queued x0.48–0.65 | ❌ LOSS |
| D10 | MoE combine tail (+residual) | **bit-exact** | chained rows=1: 16.21 vs mtplx 11.26 vs stock 14.25 | ❌ LOSS at decode (µs epilogue) |
| D11 | dense layer-0 MLP (5/6-bit) | pass | stock/fused 0.206× | ❌ LOSS ~5× (~2% share) |
| D12 | router top-10 (bitonic) | **0 selection flips** | chained rows=1: 30.53 vs mtplx 21.20 vs stock 27.22 | ❌ LOSS vs both (shape-dependent) |
| D13 | embed + rope-atlas | n/a (pure memcpy) | challenge's own −0.23…−0.7% | ⊘ NO LEVER (no distinct decode op) |
| **P1** | **prefill qk-norm + rope** | compile bug (`T` shadow) → **FIXED** → **bit-exact** (max\|d\|=0.0, rows distinct) | **queued 1.48× (FULL)** / 0.96× (sliding); per-call ~1.0–1.08× | ✅ **mild WIN** (FULL) / neutral (sliding) |
| P2 | steel flash-attention (prefill) | ctx1024 PASS; **ctx8192 FAIL** | ratio x0.05–0.16 (full), x0.99 (sliding 8k) | ❌ LOSS (no `mlx::steel` MMA) |
| P3 | prefill router top-10 | **0 flips** | queued 0.266× @M1024, 1.021× @M10240 | ❌ LOSS @1024 / neutral @10240 |
| **P5** | **prefill MoE combine tail** | **bit-exact** (max\|d\|=0.0) | **queued 4.745× / per-call 1.603×** | ✅ **WIN** (isolation) |

**Scoreboard:** **2 new bit-exact isolation WINS** (P5 MoE-combine, P1 qk-norm+rope), 0
decode wins, 11 losses/neutral, 1 no-lever. Both wins are **prefill fusions** and both are
**bit-exact** (digest-safe). This sharpens the D1 thesis: cross-op **fusion** transfers to
S-2.1's affine geometry (D1 decode, P1+P5 prefill); hand **GEMV/attention** kernels do not,
because they race MLX's `mlx::steel` MMA (`gather_qmm`, flash-SDPA) which `mx.fast.metal_kernel`
cannot reach. The prefill combine/rope wins exist precisely where stock has **no MMA-backed
path** to beat. End-to-end ceilings are bounded by prefill's share (0.25 of score) and each
op's slice of prefill, so they are validated op-level wins layered on the shipped D1+S1
(+5.8% decode), not headline replacements.

## Prefill runtime A/B — P5 wired end-to-end (2026-08-02)

The isolation "5.1× queued" for P5 was measured vs the **naive** combine. Wired into
`alt_prefill_forward` and A/B'd against the shipped **reference** (which already installs
`kernel_moe_combine` on 47 layers), P5 is a different story — measured across a context sweep,
D1-free, all **digest-exact**:

| ctx | ref tok/s | P5 (D1-free) | P5/ref | note |
|---|---|---|---|---|
| 1024 | 1564 | 1563 | 1.000× | overhead ≈ gain |
| 8192 | 1276 | 1298 | 1.018× | crossover |
| 16384 | 904 | 906 | 1.002× | within noise |
| 32768 | 492 | 511 | 1.039× | small win |

**Verdict: bit-exact and never breaks (through ctx 65536, peak 84.5 GiB), but NOT a robust
prefill win** — vs the already-fused reference the gain is ~1–4% at ctx ≥ 8k and within the
box's cross-run variance (ref@32k swung 582→492 between sweeps under contention). The combine
is too small a share and the reference already fuses it (same lesson as D10 at decode). P5
*does* beat a D1-coupled baseline (1.04–1.09× vs D1), but D1 itself penalizes prefill, so
`d1+p5` only ties/edges reference. **A real size crossover exists** (P5 loses below ~8k where
the `[M,top_k,hidden]` intermediate it removes is cheap), so P5 is wired behind
`p5_prefill_moe_tail` + a `prefill_min_tokens` gate (recommend 8192) and left **default-off**
as a validated, digest-safe, at-worst-neutral option — not a shipped speedup.
Receipts: `benchmarks/laguna-p5-prefill-sweep-{d1free,d1coupled}-20260802.txt`.

**Benchmark tracking:** all bench/laguna benches (batched/`run_cell`, compiled-lane, alt-lane,
AB comparison) now report **`prefill_tokps`** (warmup-excluded) alongside decode, in both the
JSON receipts and printed summaries.

Already shipped (unchanged): **D1** residual+RMSNorm+router-GEMV fusion (+1.0%), **S1**
async-eval scheduling (+4.0%), best **+5.8%** (71.2 vs 67.3, digest-exact).
Already measured losers (prior fair-vehicle runs): **D6** SDPA-vector −1.6%, **D8** routed
per-token SwiGLU-QMV −25%, **D14** lm-head top-1 −0.7%, **P4** prefill gather-GEMM (stock
handles empties free).

## Recurring root cause (why the hand kernels are expected to lose on affine)
`mx.fast.metal_kernel` compiles a **standalone snippet** and cannot reach MLX's `mlx::steel`
16×16 simdgroup-matrix (MMA) headers — the exact machinery that makes stock `gather_qmm` /
flash-SDPA fast. On XS2.1's NVFP4 (no native MLX kernel) the challenge's hand kernels had no
tuned rival; re-expressed on S-2.1's **affine oQ4e** they race MLX's tuned stock and the only
transferable wins are quant-agnostic **fusion (D1)** + **scheduling (S1)**. The per-kernel Δ
below either confirms this or finds an exception — measured, not assumed.
