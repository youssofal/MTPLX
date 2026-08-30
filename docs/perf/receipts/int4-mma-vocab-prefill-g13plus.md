# INT4 simdgroup-MMA lanes: vocab + prefill retile (G13+ portable)

Date: 2026-08-25
Hardware: Mac mini M4 base (applegpu_g16g, family 9), 16 GB unified, macOS 26.2
Module: `mtplx/kernels/int4_simd_mma.py` (new)
Status: exactness-gated, env-gated off by default (`MTPLX_INT4_MMA`)

## What

Classic `simdgroup_matrix<T,8,8>` lowering of the Metal 4.1 packed-INT4 recipe
(BM=128/BN=32/BK=64/WM=4/WN=1 prefill tile; small-M vocab lane), so the same
source runs on M4 dev hardware and G17/M5 deployment. The MPP `matmul2d`
native-int4 path in `nax_verify.py` remains the G17 fast path; this module
covers every other Apple GPU.

Lanes:
- **vocab** (decode/verify M 8..16): BM∈{8,16} padded tile, BN=32, BK=64,
  one 128-thread TG per 32 output columns. Stock steel dispatch collapses on
  wide-N shapes at M≥8; this lane keeps the weight stream at bandwidth.
- **prefill** (M ≥ 128, M % 128 == 0): BM=128, BN=32, BK=64, WM=4/WN=1,
  double-buffered threadgroup weight tiles (next tile dequantized during
  current MMAs), one barrier per inner iteration, GROUP_M=8 grouped swizzle.

Dispatch: `install_int4_mma_qlinear_patch()` mirrors
`nax_verify.install_nax_qlinear_patch` (idempotent, `lane_disabled`
kill switches `int4_mma_vocab` / `int4_mma_prefill`, counters via
`mma_dispatch_counter_snapshot()`). Stock GEMV wins below M=8 (~98.6 GB/s at
m=1 vs our fixed-overhead MMA tile), so the vocab lane starts at M=8.

## Exactness receipts

Random-weight gate vs stock `mx.quantized_matmul`:
- vocab lane, k=512 n=6400 g{32,64}, m ∈ {1..16}: dmax ≤ 0.0625 bf16 — equal
  to stock's own distance from an fp32 dequant ground truth (both ~0.3–0.44%
  at |y|max ≈ 9–12.5); the deltas are accumulation-order rounding only.
- prefill lane, real layer shapes (5120→34816 / 5120→8192 / 6144→5120 /
  17408→5120) m ∈ {128,512}: **bit-exact (dmax = 0.000000)**.

Real production weights (DFlash2 drafter `mtp.safetensors`, W4/G64 affine):
all four hot shapes bit-exact at m ∈ {128, 256}.

## Perf receipts (base M4, median of 20–30 iters)

Real W4/G64 drafter weights, prefill lane:

| shape            | m   | stock ms | mma ms | speedup |
|------------------|-----|----------|--------|---------|
| q_proj 12288x5120 | 128 | 5.28    | 4.98   | 1.06x |
| o_proj  5120x6144 | 128 | 2.81    | 2.71   | 1.04x |
| up_proj 17408x5120| 128 | 7.32    | 6.95   | 1.05x |
| down_proj 5120x17408 | 128 | 7.40 | 7.00   | 1.06x |
| (same four)       | 256 | —       | —      | 1.07x avg |

Vocab lane, synthetic N=248320 K=5120 g64 (stock collapses with M):
m=8: 15.97 → 12.43 ms (**1.28x**); m=16: 25.12 → 20.63 ms (**1.22x**).
Vocab lane on real up_proj weights: m=8 **1.47x**, m=16 **1.25x**.

Isolated prefill-lane speedups are 1.04–1.10x across all trunk shapes —
same order as the upstream G17 isolated measurement (+1.66%).

## Debugging note (worth keeping)

MLX `mx.fast.metal_kernel(grid=...)` semantics: **grid is total threads**
(dispatchThreads), not threadgroups. A grid of `(m_tiles, n_tiles)` launches
one thread per tile — every kernel appears to "run" but only tid 0 executes,
and results look like deterministic corruption. Correct pattern used by every
kernel in this repo: `grid=(threads_per_tg * tg_count_x, tg_count_y, 1)`.
Inside MSL, derive tile counts from scalar size args, never from `grid_size`
(which is also threads).

## End-to-end context

keXjos/Qwen3.8-27B-mlx-2Bit (2-bit g64) loads and generates on the 16 GB M4:
10.62 tok/s decode (vs 6.4–7.4 tok/s for the GGUF llama.cpp attempts in the
prior session). The 2-bit model does not exercise these 4-bit lanes; lanes
fire for the MTPLX optimized 4-bit checkpoints (trunk W4/G32, drafter W4/G64)
and any 4-bit QuantizedLinear with eligible geometry.

## Context-size sweep (2026-08-25 addendum)

Prefill lane vs stock across chunk size M on real W4/G64 drafter weights,
all bit-exact (`mx.array_equal`), median-of-N timing:

| shape | m=128 | m=512 | m=1024 | m=2048 | m=4096 | m=8192 |
|-------|-------|-------|--------|--------|--------|--------|
| q_proj 12288x5120  | 1.06x | 1.07x | 1.08x | 1.08x | 1.08x | 1.08x |
| up_proj 17408x5120 | 1.06x | 1.08x | 1.08x | 1.08x | 1.08x | 1.08x |
| down_proj 5120x17408 | 1.06x | 1.08x | 1.09x | 1.08x | 1.08x | 1.08x |

Flat 1.06–1.09x from 128 through 8192 rows — no crossover or cliff; the
speedup does not decay with context. Typical single-image token counts
(1–2K embedding rows) land inside the swept range.

Machine context ceiling (keXjos 2-bit g64 end-to-end, naive whole-sequence
forward, 16 GB):

| ctx | prefill | decode | peak mem |
|-----|---------|--------|----------|
| 1024 | 61 tok/s | 10.69 tok/s | 10.7 GB |
| 4096 | 55 tok/s | 1.25 tok/s | 16.4 GB (RAM ceiling) |
| 16384 | FAILED | — | 12.9 GB single alloc > Metal 9.5 GB buffer cap |

The 16K failure is an artifact of materializing logits for every position
(1 x ctx x 248320): serving stacks slice hidden states before lm_head and
would clear it; the practical comfortable limit here is ~2K ctx.

Vision projector (majentik/Qwen3.8-27B-MLX-2bit, 2-bit g32 + bf16 vision
tower): loads at 11.0 GB active but generation OOMs even text-only on this
16 GB machine (mlx-lm wired-limit warning at 11957 MB of 12124 MB max).
The vision tower itself runs bf16 matmuls (INT4 recipe not applicable);
image tokens enter the language trunk as one large-M prefill batch — i.e.
the exact M>=128 regime swept above, so the lane applies unchanged once run
on a >=24 GB host like the PR #335 benchmark rig.

## Native MPP packed-INT4 lane (2026-08-25 addendum 2)

`mpp_vocab_qmm` + `_build_mpp_vocab_kernel` implement the upstream Metal 4
TensorOps recipe from the MSL 4.1 specification read:

- Weights fed DIRECTLY as `tensor<device uint4b_format>` (spec Table 7.3:
  bfloat/half x uint4b -> float, Metal 4 + OS 26.4) — no dequant pass.
- Exact affine semantics via per-group decomposition (the spec has no fused
  bf16-affine path; `tensor_blockwise` scales are ue8m0-only): per k-group,
  `C += s_g * dot(A_g, W_g)` from `matmul2d_descriptor(BM,16,64,
  multiply_accumulate)` at `execution_simdgroup` scope, plus a banked
  `rs[MAX_NG][BM]` activation row-sum pass and an epilogue adding
  `sum_g b_g[n] * rs[g][r]` — the "scale x dot_product + bias x
  activation_sum" form.
- Tile: 8x16x64 per simdgroup op (upstream's measured geometry); TG of 128
  threads covers 64 columns; transposed-view `tensor_inline` operands with
  unit-stride K axis exactly as nax_verify's working G17 kernels.

Gating: `applegpu_g17*` AND macOS >= 26.4 (int4b tensors), env lane 'mpp'
(opt-in on top of vocab/prefill), `lane_disabled('int4_mma_mpp')`, plus a
one-time eager compile-and-run probe (`_mpp_runtime_probe`) because MLX
compiles custom kernels lazily — without it the first failure would surface
at the caller's mx.eval outside any guard. Probe verdict cached for the
process life.

Verified control flow on this M4 (g16g / macOS 26.2): gate off -> portable
vocab lane; forced gate -> probe fails (local SDK lacks packed-numeric
format types pre-26.4), cached, seamless vocab fallback, counters clean.
The kernel body itself follows nax_verify's field-proven G17 conventions but
its numerics remain UNVERIFIED until run on G17-class hardware — treat as
experimental until a probe+exactness receipt exists from an M5 host.

## Long-context on 16 GB: >=10 tok/s at 128K (2026-08-25 addendum 3)

Target: keXjos 27B 2-bit g64 on the base M4, decode >= 10 tok/s at maximum
context. Three changes, no model edits:

1. **Sliced lm_head (the critical fix).** mlx_lm's whole-sequence forward
   materializes logits for every position: (ctx x 248320) — 12.9 GB single
   alloc at 16K, over Metal's 9.5 GB per-buffer cap, plus RAM thrash below
   that (decode collapsed to 1.25 tok/s at 4K). Fix: call the backbone
   (`model.language_model.model`) per chunk, slice `[:, -1:, :]`, apply
   `lm_head` to the last position only.
2. **Chunked prefill** (2048-token chunks) bounds transient activations.
3. **Rotating window on the 16 full-attention layers only**
   (`RotatingKVCache(max_size=2048, keep=4)`); linear-attention/GDN layers
   keep their constant-size global state via `ArraysCache`.

Measured (greedy decode, median step):

| ctx | config | ms/step | decode | peak GB |
|------|--------|---------|--------|---------|
| 1024 | naive baseline | 91 | 10.96 | 10.3 |
| 4096 | naive (before fix) | 802 | **1.25** | 16.4 |
| 4096 | sliced lm_head | 98 | **10.24** | 11.9 |
| 8192 | sliced, fp16 KV | ~110 | 9.1→5.5* | 12.5 |
| 16384 | sliced, fp16 KV | 108 | 9.29 | 13.9 |
| 32768 | + window 2048 | 94 | **10.67** | 11.9 |
| 65536 | + window 2048 | 97 | **10.35** | 11.9 |
| 131072 | + window 2048 | 90 | **11.11** | 11.9 |

(*first 8K run prediced chunk tuning; later runs flat.) Step time vs ctx is
FLAT with the window — attention KV reads were the marginal cost (+1.1 ms
per 1K tokens), and the window caps them.

Negative results worth keeping:
- QuantizedKVCache (8-bit and 4-bit) is SLOWER than fp16 KV at every size
  tested — eager dequant overhead exceeds bandwidth saved (mlx_lm 0.31.3).
- Needle-recall controls: in-window needle (600 back) = HIT; needles 2600
  and 6000 back MISS EVEN WITH FULL UNWINDOWED KV. The long-range recall
  gap is intrinsic to this 2-bit checkpoint (or GDN hybrid retrieval), not
  the window — so the window costs nothing measurable in quality here.

Remaining limits: prefill throughput is ~58 tok/s regardless of chunk size
(GDN chunked kernel-bound) — 128K prompt takes ~37 min to ingest; and the
9.5 GB Metal single-buffer cap still forbids any op materializing
full-sequence vocab logits.

## Follow-ups

- MLX host-side dispatch reductions from the upstream recipe require patching
  MLX's C++ backend (arg-vector reserve, config-by-reference, regex→char-scan
  sanitization); needs a source build of MLX, not doable against the pip wheel.
- Wire the vocab lane into the PR #335 serve path behind the existing profile
  plumbing so the M8/M16 verify rounds pick it up automatically.
