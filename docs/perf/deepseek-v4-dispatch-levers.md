# DeepSeek-V4 decode: dispatch-structure levers — bench note

Two decode dispatch-structure levers for the DeepSeek-V4 backend, both measured on
the real 2-bit-DQ checkpoint. The dispatch census counts the stream; the measured
window is the wall-clock A/B.

## Why these two

The measured cycle is **84.8 ms fixed + 8.9 ms/K**, with the target GPU forward
71–81% of every cycle — so the wall is the structure of the dispatch stream, not
bytes. `scripts/deepseek_v4_dispatch_census.py` measures that structure off the
Metal dispatch stream itself (instrumented MLX in `mlx-profiler`), differencing a
9-decode-step run against a 1-step one so load, prefill and compile tracing
cancel. At DeepSeek-V4-Flash's *structure* with shrunk widths, one bf16 `s == 1`
step was **19,809 dispatches in 384 command buffers**, and the `cb` rows put
**host encode at 56–59 ms against 32–35 ms of GPU execution**: the encode is
exposed, not hidden. ~2.9 µs of host encode per dispatch, size-independent.

Two thirds of those dispatches came from the Hyper-Connection Sinkhorn chain,
running 20 alternating row/column normalisations on a 4×4 tensor, 87 times per
token.

## What changed

| lever | knob | default |
|---|---|---|
| Hyper-Connection chain collapse (derive fp32 weights once, one fused affine, `mx.compile` one shared tape) | `MTPLX_DSV4_HC_COMPILE` | `1` (on) |
| Attention sink as one extra KV column + single `mx.softmax(precise=True)` | `MTPLX_DSV4_ATTN` | `fused` |
| Sinkhorn's fixed 4×4, 20-iteration fp32 recurrence as one Metal dispatch | `MTPLX_DSV4_SINKHORN_KERNEL` | `0` (off) |

`MTPLX_DSV4_HC_COMPILE=0` and `MTPLX_DSV4_ATTN=dense` restore the branch-point
behaviour exactly; `MTPLX_DSV4_ATTN=sdpa` is the third arm (see below).

## Census (bf16, per decode step, 43 layers, reproduces exactly run to run)

| arm | dispatches | command buffers |
|---|---|---|
| BEFORE (`a4c2a9c`) | 19,809 | 384 |
| `dense` + no hc compile | 18,946 | 372 |
| `fused` + no hc compile | 18,774 | 370 |
| `dense` + hc compile | 14,811 | 292 |
| **`fused` + hc compile (default)** | **14,639** | **288** |
| `fused` + hc compile + Sinkhorn kernel (opt-in) | 7,845 | 155 |
| `sdpa` + hc compile | 14,424 | 283 |

**−26.1% dispatches, −25.0% command buffers.** At fp32: 17,733 → 13,039 (−26.5%).
Per-kernel: `vs_Add` 3570→43 and `g2_Divide` 3408→11, replaced by 3354 fused
dispatches; the attention softmax chain (`vv_Maximum`, `row_reduce_max`, `v_Exp`,
the two fp32 casts) −172.

At ~2.9 µs of host encode per dispatch, ~5,200 removed dispatches is **~15 ms per
token of host encode** — against an 84.8 ms fixed term. The window below realised
13.7 ms of it.

## Measured window

`bench/deepseek-v4/kernel-a-ab-20260801`: real 2-bit-DQ checkpoint, B=1 greedy,
328-token prompt, 256 decode tokens, drift-bracketed, `MTPLX_DSV4_O_LORA=cached`
held fixed throughout (the other banked decode-byte lever — mixing them makes the
attribution unreadable). One arm per run, everything else fixed.

| arm | AR tok/s | K=3 spec tok/s |
|---|---|---|
| BEFORE (`dense`, `hc_compile=0`) | 17.37 | 26.31 |
| **AFTER (`fused`, `hc_compile=1`, default)** | **22.80 (+31.3%)** | **30.88 (+17.4%)** |

The +5.43 AR tok/s is **13.7 ms/token** of wall clock removed — 91% of the ~15 ms
of host encode the census attributed to the ~5,200 removed dispatches. So the
"host encode is exposed" reading holds on the real model: here dispatch removal is
wall-clock, near one for one. No regression — every arm's decode text stayed
coherent.

### Stage 4: Sinkhorn recurrence kernel

The final Sinkhorn floor is now an opt-in, shape-specific Metal lane.  It replaces
the 4×4 fp32 schedule (row softmax, then 20 column and 19 row normalisations) in
each HC `pre` call.  The shrunk bf16 decode census shows the structural Sinkhorn
stream collapsing from **6,794 dispatches to 86**.  This is not a general
small-matrix kernel: the forced GPU lane accepts only `hc=4`, `iters=20`,
`eps=1e-6`; CPU/no-Metal installs the stock oracle explicitly, and an unsupported
GPU configuration fails at construction before generation.

The independent E2E source is the local
`bench/deepseek-v4/stacked-ab-20260801` receipt; its immutable, scrubbed extraction
is the tracked [stage-4 receipt](receipts/deepseek-v4-sinkhorn-stage4.md).  It is a
stacked-window A/B on the real DeepSeek-V4-Flash **2-bit-DQ + mxfp4 MTP**
checkpoint: B=1 greedy, 328 prompt tokens, 256 decode tokens, cached o-LoRA, HC
compile on, and fused attention in both arms.  Only the Sinkhorn flag changes:

| stacked-window arm | AR tok/s | K=3 spec tok/s |
|---|---:|---:|
| stock Sinkhorn control | 22.318 | 30.440 |
| Sinkhorn Metal kernel | **28.858 (+29.3%)** | **32.497** |

So AR clears the requested **27 tok/s** bar, and the best K=3 is approximately
**32.5 tok/s**.  Both arms produced coherent output.  The bf16 K=3 receipt also
records the documented near-tie / `spec != AR` diagnostic; the harness reports
that divergence as data rather than turning one prompt into a quality verdict
(the behavior specified by the
[`spec gate`](../../tests/test_deepseek_v4_spec.py)).

The raw receipts are local benchmark artifacts and are intentionally not added to
this code branch; the upstream PR comment carries the supporting table.  The
[`deepseek_v4_mtpk_bench.py`](../../scripts/deepseek_v4_mtpk_bench.py) contract
defines the receipt fields.  These throughput numbers come from real generation,
not the profiler: the census above establishes dispatch structure only and is not
presented as E2E timing.  Fused attention remains the default, and the Sinkhorn
lane stays default-off until the requested serving decision changes that policy.

**The mlx 0.32 arm is a clean null.** Re-running the default arm under mlx 0.32
moved nothing past drift: once the host encode is gone, DeepSeek-V4 decode is
GPU-forward bound, not qmm-ALU bound, so 0.32's matmul / `mx.compile` changes have
no exposed host work left to bite on. It is **not** run for SDPA fusion (see
below).

**Scope — what this does not do.** The lever trims host encode. The remaining gap
to the ≥40 tok/s goal (K=3 30.9 → 40 is 1.29×) is GPU-forward and verify-width,
not dispatch; that is a different lever.

### `mx.fast.scaled_dot_product_attention` does not fuse this, on any MLX here

The `sdpa` arm is kept and gated exact, but it is **not** the default. MLX takes
`sinks=` natively, but its fused Metal kernels are only instantiated for head dims
64/96/128/256 (0.31.2) and 64/96/128/192/256 (0.32.0 and the 0.32.1.dev profiler
build) — verified by reading the symbol table of each shipped `mlx.metallib`.
DeepSeek-V4's MLA latent is **512** wide, so `sdpa` takes MLX's own unfused
fallback on every version available here; `fused` is that same fallback minus two
copies over the block, and wins.

## Gates already green

`tests/test_deepseek_v4_*.py`: **163 passed** in all six arms
(`{fused,dense,sdpa}` × `hc_compile={0,1}`), 139 pre-existing + 24 new in
`tests/test_deepseek_v4_kernel_paths.py`.

- attention arms vs the `dense` oracle, fp32: max_rel **1.9–2.4e-6**, argmax
  exact — one-shot and streaming (21-token prompt + 204 single-token decode
  steps), dense and sparse (`index_topk` crossed) regimes, and under
  `MTPLX_DSV4_FP32_ACTIVATIONS=1`.
- bf16 lane: **bit-identical** at this config, argmax 48/48.
- `_hc_pre_impl` **bit-identical** to the reference transcription it replaces
  (`_mixes` + `hc_split_sinkhorn` + weighted sum).
- compiled vs eager: **bit-identical** to 4 rows (B=1 decode, short verify
  batches); ≤2e-7 relative per call from 8 rows up, ≤3e-6 end-to-end over the
  layer stack — an order inside the 5e-5 the streaming-decode gate already
  spends. Decode-shape logits are bit-identical end to end.

## Deferred, with sizes

- **Sinkhorn is no longer a stock-op floor in the opt-in lane.** The hand kernel
  is confined to the measured 4×4/20/1e-6 fp32 geometry and the stock loop remains
  the default oracle.  It does not change the fused-attention default or make a
  broader claim about other HC shapes.
- **Precompute is off, definitively.** The reference computes
  `mixes = F.linear(x, hc_fn) * rsqrt` from the layer's own hidden state
  (`Block.hc_pre`, fetched read-only from the reference), so `comb` depends on
  activations and cannot be derived at load. `hc_fn` is `[24, hc*dim]` in the
  shipped checkpoint, which is the same fact in the weight shapes.
- **Fewer than 20 Sinkhorn iterations** would be the other way to cut the
  reductions. The reference fixes 20; convergence on the real weights is
  unmeasured and it would not be exact. Needs a checkpoint and a quality run.
- **Rope is the next structural lever.** `_apply_interleaved_rope` runs 3× per
  attention call and is pure elementwise + stack/reshape — the same
  `mx.compile` treatment applies, and `mx.fast.rope` must stay off (the batched
  T=1 row-0 bug on 0.31.2). Not attempted: out of scope for this pass.
- **Micro:** `fused` adds 86 `pad` dispatches per step (one for the KV zero row,
  one for the sink column) and 86 fill-constant copies. Caching the sink column
  would take ~43 of those. 0.6% — listed for completeness, not proposed.
