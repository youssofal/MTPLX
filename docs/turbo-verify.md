# Turbo verify kernels

Status: shipped — the default profile for the quantized 27B flagships since
2.0.0 and the 9B tier since 2.0.1; `mtplx serve` selects turbo for those
models automatically. `MTPLX_NAX_VERIFY` remains available as an explicit
A/B control, and `MTPLX_KERNEL_SELFCHECK` as the load-time diagnostic gate.

```bash
MTPLX_NAX_VERIFY=1 mtplx serve ...   # explicit A/B override on a non-default model
```

When enabled at model load, 4-bit affine projections route through
verify-specialized Metal kernels (ported from bstnxbt/dflash-mlx, Apache-2.0)
for batches of 4..16 rows — the shape of native-MTP speculative verification.
Single-token decode, drafting, and prompt prefill are untouched and remain
bit-identical to stock MLX.

- 4-row K-split kernel: any Apple Silicon.
- 16-row tile via Metal 4 tensor ops: Apple M5-class GPUs (G17) on
  macOS 26.2+, used for depths above 3.

Measured on M5 Max / Qwen3.6-27B Optimized-Speed, reasoning on, 2026-06-12:
1k-token decode 48.3 -> 65.5 tok/s mean over four matched seeds; official
flappy envelope 55.7 -> 64.5; live server completion 55.0 -> 66.7; 10k-token
generation 59.5 tok/s sustained. 4-bit, 6-bit, and 8-bit affine layouts are
supported; the 9B Optimized-Speed 6-bit lane ships via the split-K hexpack
kernels. MoE (35B-A3B) routes only dense projections: ~neutral.

Numerics: not bit-exact versus stock kernels (different accumulation order).
Argmax-identical on all probed positions; at the product sampler
(temp 0.6 / top_p 0.95 / top_k 20) the live D3 verify path measured total
variation 0.0 and sample agreement 1.0 on every probed cell
(the gate is `scripts/nax_distribution_gate_expanded` in the internal
research workspace; it is not part of the shipped package). Speculative acceptance remains mathematically exact with respect to
the verify-computed target distribution. Do not use for bit-exactness QA
(`mtplx qa exactness` reference runs, batch-equivalence gates).
