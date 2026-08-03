# DeepSeek-V4 Sinkhorn stage-4 receipt

This is the scrubbed, tracked receipt for the independently measured Sinkhorn
stage-4 window. The raw benchmark artifacts remain local; their hashes are
recorded below so the result can be audited without committing bulky generation
logs or model-derived output.

## Code provenance

- Benchmarked tree: `3f0aa06d2fcc4bd94b9aa43039f8560eb6c81ba3`
  (`perf/deepseek-v4-kernels-combined`).
- Sinkhorn implementation in that tree: `5d89f42a1f2b336f5a4cda42896f3fd00a359968`,
  code-equivalent to source revision
  `57f54dd020a581ee7e69670cc9973a1130ac17ac`.
- The later device-guard fix is
  `8067f9bfa2d879ce299aa5aa7d35b4c6db2ab125`. It was not part of the benchmarked
  tree; it changes CPU/no-Metal eligibility, not the measured GPU arithmetic.
- PR port provenance: implementation `38beed8231255a8d7095f245e188d31d2ca8fdf4`,
  guard `c8c3ea36d5d1c76833e3ed583d6b9646de9e1209`, and construction-time route
  installation `0779bf14b2dd2b7bcc3cfc2124545309381cd352`.

## Machine, model, and fixed conditions

- Machine: Apple M5 Max MacBook Pro, 128 GB.
- Runtime: macOS 26.5.2 arm64, MLX 0.31.2, Python 3.12.13.
- Model: DeepSeek-V4-Flash 2-bit-DQ trunk with the mxfp4/bf16 MTP bank; 43 body
  layers and one next-token-prediction layer.
- Shape: B=1 greedy (`temperature=0`, stop tokens disabled), 328 prompt tokens,
  256 decode tokens, AR control plus K=3.
- Speculative path: `capture_commit` verification, stock verify core, committed
  MTP history.
- Each arm had an unrecorded 8-token AR warmup.
- Held fixed: offline model access, `MTPLX_DSV4_HC_COMPILE=1`,
  `MTPLX_DSV4_O_LORA=cached`, and `MTPLX_DSV4_ATTN=fused`.
- Varied: only `MTPLX_DSV4_SINKHORN_KERNEL`, from `0` in the control and drift
  repeat to `1` in the Sinkhorn arm.

Path-neutral reproduction shape (the five-arm source window also measured MLA
and combined arms; this receipt reports the control, Sinkhorn, and control-repeat
cells):

```sh
<python> <repo>/bench/laguna/run_guarded.py <site-specific-guard-args> -- \
  env HF_HUB_OFFLINE=1 \
      MTPLX_DSV4_HC_COMPILE=1 \
      MTPLX_DSV4_O_LORA=cached \
      MTPLX_DSV4_ATTN=fused \
      MTPLX_DSV4_SINKHORN_KERNEL=<0-or-1> \
  <python> -u <worktree>/scripts/deepseek_v4_mtpk_bench.py \
      --model <model-dir> \
      --prompt-file <328-token-prompt> \
      --max-tokens 256 --depths 3 \
      --verify-strategy capture_commit --verify-core stock \
      --mtp-history-policy committed --warmup-tokens 8 \
      --out <local-receipt-stem>
```

The original driver ran all arms sequentially inside one guarded window and
repeated the control last to expose drift.

## Results

| arm | AR tok/s | K=3 tok/s | K=3 committed/verify | peak GiB | coherence | K=3 spec vs AR |
|---|---:|---:|---:|---:|---|---|
| control, Sinkhorn off | 22.318 | 30.440 | 2.844 | 96.89 | unique-line ratio 1.00; max run 1 | pass |
| Sinkhorn on | **28.858** | **32.497** | 2.723 | 97.12 | unique-line ratio 1.00; max run 1 | near-tie divergence at index 6, AR 603 vs spec 305 |
| control repeat, Sinkhorn off | 22.881 | 30.903 | 2.844 | 96.89 | unique-line ratio 1.00; max run 1 | pass |

The requested framing is Sinkhorn versus the first in-window control:
`22.318 -> 28.858 AR tok/s`, or **+29.3%**. The control repeat was 2.5% faster
than the first AR control, so the Sinkhorn gain remains well outside drift; it is
+27.7% against the two-control mean. AR clears 27 tok/s, and the best K=3 result
is 32.497 tok/s (approximately 32.5).

K=3 per-depth acceptance was `0.897 / 0.609 / 0.174` for the control and
`0.911 / 0.567 / 0.189` for Sinkhorn. All reported outputs completed coherently.
The Sinkhorn K=3 `spec != AR` result is the documented bf16 near-tie diagnostic,
not an omitted correctness claim: the receipt records the first index and both
tokens, while task quality requires a task evaluation rather than a byte-identity
verdict from one prompt.

## Profiler structure is separate

The shrunk bf16 profiler census showed the Sinkhorn schedule collapsing from
6,794 dispatches to 86. That establishes engagement and dispatch structure; it is
not used as E2E timing. The throughput table above comes only from the real-model
generation window.

## Local raw-artifact manifest

SHA-256 values are over the original local files. Basenames are retained; home,
temporary-worktree, process, service, and attestation details are deliberately
excluded from this tracked receipt.

| local artifact | SHA-256 |
|---|---|
| `stacked-ab-20260801-SUMMARY.txt` | `6e51fc4f75218730d31f7ab8b93044af4d83fdc8e77510716c2a58756a399335` |
| `stacked-ab-20260801-VERDICT.txt` | `d6a3a73e55b1a51c3a0a4b692966ac8bf28425e2ae17fa4558db8cd9eb7d5fa9` |
| `stacked-ab-20260801-before.json` | `365799a6a0be1e22b4d080b1e05026f55b45e843f83f711e2c0bc3a31d8c8bd9` |
| `stacked-ab-20260801-sink.json` | `50f784486e76077ff5b2c920534fabf605891e301bd20011a74558b1cc69b73b` |
| `stacked-ab-20260801-before2.json` | `66e1040c0f4d37cd92892d800f097ce4d2fc2aec48d66187a76a8b8576531695` |
| `stacked-ab-20260801-arms.sh` | `6b51ef1ddc4846a64d922b4fe12469f313fb995c9e9e68d8c327f68184dc15a2` |
| `stacked-ab-20260801-run.sh` | `61d45439ade50c659c432f066c75a53ba4dda5c22fb52e632c2d4f705f9d9fff` |
| `stacked-ab-20260801-window.log` | `5c6acfd157e48ddd9344fe51f48aeee3805c16ba3a31d8ccb53c587e2bbfb655` |

The upstream PR comment links this tracked receipt and reproduces the result
table. No raw receipt, generated text, model path, PID, service detail, or secret
is stored here.
