# DeepSeek-V4 Flash 0731 DSpark receipt

This is the scrubbed, tracked performance receipt for the construction-bound
DeepSeek-V4 Flash 0731 DSpark physical-M3 K2 lane. This route prioritizes the
measured throughput win and is intentionally not token-exact against serial
greedy AR. Raw generation artifacts remain local; their hashes are listed below
without model paths, generated text, service details, or machine-local process
data.

## Fixed conditions

- Machine: Apple M5 Max MacBook Pro, 128 GB, macOS 26.5.2.
- Runtime: Python 3.12.13, MLX 0.32.0, mlx-lm 0.31.3.
- Model: `mlx-community/DeepSeek-V4-Flash-0731-2.4bit-mixed` at source revision
  `10001e0065f8394e03e968e652cbbe7cd2ca122c`.
- Model identity: config SHA-256
  `44735712733fcf8f299bdf1faa1d87fac88f1917efe1d3876d6d4c582f79a68f`;
  index SHA-256
  `f1332b2b209769c2db335954c2651652a8048e7d7dbf60296c2f2c0198715861`.
- Sampler: greedy, `temperature=0`, `top_p=1`, `top_k=0`, seed 0.
- Prompt: `Explain why speculative decoding can preserve greedy output.`
  through the model chat template, 14 prompt tokens.
- Output: forced 128-token budget, two identical cases in one model load. The
  second case is the warmed comparison; the first exposes one-time compilation.
- PR lane: explicit `deepseek_v4_0731_k2=True`, fixed proposal width K2,
  persistent cache, cycle history, one physical target-M3 call per full verify
  cycle, stock verify and draft cores.
- Benchmarked commit: `51873de47ff076c95cf9938be0aca56aabe3cebb`.

## Current PR bracket

Memory growth is measured from the post-load active-memory baseline of
86.4561 GiB. Peak memory is the process-wide MLX peak and therefore includes
load-time allocation; it is identical across these in-load arms.

| case | depth | target prefill tok/s | decode tok/s | end-to-end tok/s | active GiB | growth MiB | peak GiB | accepted / drafted | exact vs K0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cold compile | K0 | 0.170 | 24.195 | 1.462 | 86.4561 | 0.0181 | 139.7061 | - | reference |
| cold compile | K2 | 103.514 | **33.925** | **32.736** | 86.4561 | 0.0192 | 139.7061 | 68 / 119 | no |
| warmed | K0 | 103.791 | 32.434 | 31.362 | 86.4561 | 0.0181 | 139.7061 | - | reference |
| warmed | K2 | 103.509 | **35.700** | **34.393** | 86.4561 | 0.0192 | 139.7061 | 68 / 119 | no |

The warmed physical-M3 K2 lane beats warmed AR: 35.700 versus 32.434 decode
tok/s, a 10.1% win; end-to-end throughput improves 9.7%. First- and
second-position acceptance were 68.3% and 45.8%. The K2 stream is deterministic
across both cases but diverges from serial greedy AR at generated-token index 44,
so this is an explicit throughput-over-exactness contract rather than an exact
speculative-decoding claim. The cold K0 prefill result is compilation time, not
model prefill throughput, so it is disclosed rather than used as a speedup claim.

## Historical K-depth diagnostic

Before the K2-only construction contract was pinned, the native DSpark harness
ran one simple 9-prompt-token, 64-output-token K0-K3 sweep on MLX 0.31.2. It did
not record prefill TPS or memory growth, so those fields are unavailable. Active
and peak memory remain useful as measured.

| depth | decode tok/s | end-to-end tok/s | active GiB | peak GiB | accepted / drafted | exact vs K0 |
|---|---:|---:|---:|---:|---:|---|
| K0 | **24.565** | **23.312** | 86.4561 | 86.5079 | - | reference |
| K1 | 19.193 | 18.584 | 86.4561 | 86.5175 | 27 / 36 | yes |
| K2 | 19.640 | 19.010 | 86.4561 | 86.5240 | 34 / 58 | yes |
| K3 | 21.413 | 20.609 | 86.4561 | 86.5392 | 37 / 76 | **no** |

This older chart is diagnostic, not a promotion result: K1 and K2 were exact but
slower than AR, while K3 was faster than K1/K2 but diverged from greedy AR. The
current public lane therefore stays construction-pinned to K2 rather than
silently widening to an unqualified K1/K3 route.

## Raw-artifact manifest

| local artifact | SHA-256 |
|---|---|
| `0731-pr-physical-m3-nonexact-k2-128-20260812.json` | `c290cfb5b0afde6eb83be79d8e5701682e4593f01bf5e1954667daa346e2f982` |
| `0731-pr-optimized-k2-128-20260812.json` | `e3e8ab454a5a6860578eb022e85297de9143b5bd5588229bb795e472ba5395c2` |
| `0731-dspark-width123-64tok-20260809.json` | `1f60e529e4c172642fa461c41f5cd5dd11f28048c571f875cff04ee73cae9a3f` |

Profiler dispatch censuses are not used as TPS proof here. The current physical-
M3 chart is an uninstrumented generation timing under the exclusive GPU lock;
its nonexactness is part of the published result, not hidden by the receipt.
