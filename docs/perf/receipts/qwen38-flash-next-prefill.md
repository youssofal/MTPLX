# Qwen3.8 Flash-Next long-context prefill receipt (qwen4_exp)

This is the scrubbed, tracked receipt for the measured long-context prefill behaviour of the two
Qwen3.8 Flash-Next packs. The raw probe artifacts remain local (`~/.mtplx/bench/`, also mirrored to
the excluded `agent-output/long-context-prefill/`); their hashes are recorded at the bottom so the
result can be audited without committing bulky per-request logs.

Plan of record for the fixes: `docs/plans/2026-09-06-long-context-prefill-handoff.md`.

## Code and machine provenance

- Measured against the **installed** runtime: `mtplx 2.11.0` (`~/.mtplx/bin/mtplx --version`),
  MLX 0.32.2, Python 3.14.5, macOS 26.4.1.
- Benchmarked tree at measurement time: this checkout at `d0518c9` on `issue-308`, version
  **2.9.0** — which did **not** contain the `qwen4_exp` descriptor, the guard at
  `mtplx/generation.py:1205`, or the KV-geometry constant. The numbers below therefore describe
  the installed 2.11.0, not that tree.
- After the 2026-09-06 rebase the tree is `main` = `406b5f768e984e036d16aca1edaddaa29fe8519e`
  (2.11.1) with the DFlash2 commit replayed as
  `d6e4a5a4a63b6a3f7b4bb179d09fc0c13585a67c`. Re-baseline before judging any fix.
- Machine: Apple M3 Ultra, 32 CPU (24P/8E), 80-core GPU, 512 GB unified
  (`hw.memsize` = 549755813888). `mtplx hardware` warns this is a high-bandwidth GPU path, not an
  M5 TensorOps path. Fan policy: Apple default (`--fan-mode default`); no ThermalForge max-fan run
  was used, so per `CONTRIBUTING.md` none of these numbers is a product headline claim.
- A second, unrelated Metal server (`ds4-server`, GLM-5.3-Flash-Q4_K on 127.0.0.1:9000,
  161.4 GiB RSS) was resident throughout. It is not part of the measurement and is not excluded
  from it — a caveat that matters for the arm-to-arm comparisons below.

## Method

`~/.mtplx/scripts/prefill-probe.py` drives the **serve path** over HTTP: for each context rung it
sends one salted prompt (cold) and then one ~600-token turn appended to the same prefix
(follow-up), with `max_tokens=1` so the measurement is prefill plus first token. Results come from
the engine's own per-request JSONL (`prompt_eval_time_s`, `new_prefill_tokens`, `cached_tokens`,
`peak_memory_bytes`), read from a recorded byte offset — not from client-side timing.

Row `kind` is derived from the engine's `cached_tokens`, never from the probe's intent. An earlier
revision of the instrument mislabelled growth rungs as cold and understated the 103 k rate by 2×
(it reported 191 tok/s where the artifact says 383); that correction is why this section exists.

**Second correction, recorded because it is the more instructive mistake:** the first version of
this receipt's 9002 table transcribed the probe's *client-side wall times* (printed to stdout as
`wall=…s`, which include HTTP, tokenisation, SSE teardown) instead of the engine's
`prompt_eval_time_s`. The two differ by up to 18 % (the 103 k follow-up is 2.397 s in the engine
and 2.83 s at the client), which silently inflated the headline decay from 2.50× to 2.9× and moved
the derived KV geometry from 249,526 to 244,249 B/token. Every number here now comes from the
JSONL engine rows, and the tables state `prompt_eval_s`, not wall.

`--tag` seeds the prompt salt, so a new tag is a genuinely cold prompt set. Throughput below is
`new_prefill_tokens / prompt_eval_time_s`.

## Baseline 9002 — `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`

Profile `turbo`, `--depth 3`, `MTPLX_NGRAM_RESIDENT=1`, `--context-window 262144`,
`--prefill-chunk-tokens 2048`, `--scheduler-mode serial`, `--paged-kv-quantization off`.

| kind | prompt | new | cached | prompt_eval_s | new tok/s | peak |
|---|---|---|---|---|---|---|
| cold | 6,500 | 6,500 | 0 | 8.012 | 811 | 112.6 GiB |
| prefix_hit | 7,049 | 556 | 6,493 | 0.960 | 579 | 112.6 GiB |
| cold | 25,745 | 25,745 | 0 | 27.965 | 921 | 117.7 GiB |
| prefix_hit | 26,294 | 556 | 25,738 | 1.220 | 456 | 117.7 GiB |
| cold | 51,405 | 51,405 | 0 | 69.276 | 742 | 124.1 GiB |
| prefix_hit | 51,954 | 556 | 51,398 | 1.615 | 344 | 124.1 GiB |
| cold | 102,731 | 102,731 | 0 | **267.920** | **383** | 135.0 GiB |
| prefix_hit | 103,280 | 556 | 102,724 | **2.397** | **232** | 135.0 GiB |

## Baseline 9001 — `grant-ai/Qwen3.8-Flash-Next-Abliterated-MTPLX-4bit`

Profile `sustained`, `--depth 1`, `MTPLX_NGRAM_RESIDENT=0` (sidecar streamed), same window/chunk.

| kind | prompt | new | cached | prompt_eval_s | new tok/s | peak |
|---|---|---|---|---|---|---|
| cold | 7,478 | 7,478 | 0 | 7.579 | 987 | 85.4 GiB |
| prefix_hit | 8,150 | 0 | 8,150 | 0.030 | 0 | 85.4 GiB |
| cold | 26,727 | 26,727 | 0 | 29.358 | 910 | 90.3 GiB |
| prefix_hit | 27,399 | 679 | 26,720 | 1.496 | 454 | 90.3 GiB |
| cold | 52,389 | 52,389 | 0 | 71.039 | 737 | 95.7 GiB |
| prefix_hit | 53,055 | 673 | 52,382 | 1.897 | 355 | 95.7 GiB |
| cold | 103,714 | 103,714 | 0 | **268.637** | **386** | 104.5 GiB |
| prefix_hit | 104,382 | 675 | 103,707 | **2.835** | **238** | 104.5 GiB |

The `8,150` follow-up row reports `new_prefill_tokens: 0` — a 672-token turn that the engine
charged as a pure prefix hit at 0.030 s. It is left in the table because it is in the artifact and
because it shows the cache path is capable of returning in tens of milliseconds; the other
follow-up rows are the ones that carry cost.

## Arms

| arm | single variable | cold eval_s | cold tok/s | follow-up eval_s | peak |
|---|---|---|---|---|---|
| baseline | — | 267.920 | 383 | 2.397 | 135.0 GiB |
| A | `--context-window 131072` | 268.162 | 383 | 2.853 | 123.3 GiB |
| C | `--prefill-chunk-tokens 8192` | **315.868** | **325** | 2.757 | **157.7 GiB** |
| D | `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=32768` | 267.003 | 385 | 2.860 | 123.3 GiB |
| B | `--paged-kv-quantization q8` | refused at load, downgraded to `off` — no artifact | | | |

Arm D is a real null result, not a dead setting: the
`dense-decode ceiling auto: … tokens` line is present in the baseline serve log and **absent** when
the ceiling is supplied explicitly.

### Repetition noise — read before comparing any two follow-up numbers

The four runs above all measured the same nominal work: one ~556–679-token follow-up at ~103 k
context. Their `prompt_eval_time_s` values are 2.397 (baseline), 2.853 (A), 2.757 (C), 2.860 (D):
**median 2.805 s, stdev 0.218 s, spread 17.0 %.** The baseline's 2.397 s is the low outlier, not the
other three being slow.

Cold prefill at the same context, excluding arm C which was deliberately a different setting:
267.920 / 268.162 / 267.003 s → **stdev 0.611 s, spread 0.43 %.**

Consequence, and it changes how the defect list should be attacked:

- Cold-prefill comparisons are conclusive at the ±0.5 % level. Everything claimed about
  profile / window / chunk / ceiling on the cold path stands.
- **Follow-up comparisons are not** resolvable below ~20 % from one run each. "Follow-up unchanged"
  is justified for A/C/D against each other (2.757–2.860 s) and *not* against the baseline's
  2.397 s. The honest range for the follow-up decay versus the 0.960 s at 6.5 k is
  **2.50×–2.98×**, not a single 2.5×.
- Any acceptance test on the follow-up path needs **≥3 repetitions**, judged on the median. A
  claimed 2× win (1.40 s from the median) clears the measured stdev comfortably; a claimed 10 % win
  would be indistinguishable from this noise.

## Counters and cross-checks

- `paged_gqa_sdpa_calls`, `prefill_partitioned_paged_calls`, `prefill_dense_fallback_calls`,
  `attention_dense_fallback_calls`: **0 at every rung on both packs**.
  `prefill_attention_impl` and `prefill_layout` are absent from the rows entirely.
- On the slow long-context rows of the historical 27B log, `target_forward_time_s` is 99.7 % of
  `prompt_eval_time_s` (32.47 of 32.57 s), with `state_rebase_count: 0`,
  `trunk_cache_materialize_time_s: 0.00`, `cache_restore_time_s: 0.00` — the cost is the forward
  pass, not restore or paging bookkeeping.
- Marginal KV from `peak_memory_bytes`: 9002 **249,526 B/token** ((135.0−112.6) GiB /
  (102,731−6,500) tokens, = 244 KiB, **3.81×** the assumption); 9001 **213,309 B/token** (208 KiB,
  3.25×). The engine's dense-decode policy assumes **65,536 B/token** (`generation.py:1499`). The
  engine's own KV-quantization refusal message states "KV on 12 of 48 layers (~24 KB/token)"
  ≈ 288 KB/token, which agrees with the measurement and contradicts the assumption.
- Projection at 262,144: **172.0 GiB** (9002) / **136.0 GiB** (9001) against the 192.0 GiB engine
  budget — reachable. An earlier draft of this receipt claimed 262 k was over budget ("278.2 GiB")
  and a later one said 173.5 / 169.4; both were wrong, the first by double-counting the 107.1 GiB of
  weights already inside `peak_memory_bytes`, the second by carrying the client-wall-time figures
  described below.
- Persona tax: the identical one-line prompt is **62 prompt_tokens on 9002 and 1,058 on 9001**
  (+996); probe-body deltas were +982 and +984 at two rungs. Cause: 9001's `chat_template.jinja`
  is the Blackfrost build while its `tokenizer_config.json` is stock Qwen.
- `vm.swapusage` read `used = 0.00M` at session start and through every single-server arm; with
  both Flash-Next ports plus `ds4-server` resident it read `used = 81.56M`.

## Raw artifact hashes

First 16 hex chars of SHA-256, paths relative to `~/.mtplx/`:

```
aa95fd34d7163110  bench/prefill-probe-9001-baseline-20260905-222454.json
f015896cfc2b30e3  bench/prefill-probe-9001-gate-20260905-222412.json
055085e7d25bf7f0  bench/prefill-probe-9002-armA-ctx131k-20260905-223642.json
8433d4d265fd3f54  bench/prefill-probe-9002-armC-chunk8192-20260905-224845.json
08597a69bffcb9c8  bench/prefill-probe-9002-armD-dense32k-20260905-230359.json
f8f2ddc2b737f25c  bench/prefill-probe-9002-baseline-20260905-220920.json
0ed036bb9ba33ac4  bench/prefill-probe-9002-smoke-20260905-220855.json
004ecc3c12877bbf  bench/identity-verify.txt
a5b93d39e0526701  scripts/prefill-probe.py
94e941128509f319  scripts/serve-flash-next-9002.sh
72a3706da4e6ccb1  scripts/serve-flash-next-uncensored-9001.sh
```
