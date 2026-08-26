# Mia DSpark benchmark receipts

## Cold Python vocabulary ladder

These measurements use the exact local Mia/Sero package at
`/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1` and pinned
DFlash revision `54644e991039110f30140006c892c57734b9311e`. The 1K, 16K,
and 64K rows were refreshed on MTPLX source
`8f47da2e61c42b4a2bea81f0b0e586762d4d481c`. The 128K row is the latest
valid receipt at that context, but it remains on source
`cc6e1b1986a946fc7907b292a827e6d80e1aa316`; it is labeled explicitly rather
than presented as a current-head measurement.

Source lineage is documented against MiaAI's DGX Spark launcher and Sero's
packaged artifact, with the RTX6K Discord community
(`https://discord.gg/X54jjmcxWJ`) included in the references. Its related
RTX PRO 6000 / SM120 public wiki is pinned at
`local-inference-lab/rtx6kpro@3633c2c6028056729a6612126e9afe05c2e3cf08`.
These receipts are Apple Metal measurements and do not claim RTX PRO 6000
validation.

Every row is a separate process with an empty request cache. Model load is
reported separately and is not included in TTFT. The request prompt ends with
the same coherent 1,024-token Python repository task. Its prefix walks a
deterministic permutation of the tokenizer vocabulary, excludes special token
IDs, and avoids repeated filler IDs until the usable vocabulary is exhausted.
The 1K row is the coherent Python task without a vocabulary prefix. The 16K
and 64K fillers contain no duplicate IDs. The 128K filler covers all 129,278
usable IDs before the 770 repeats that are mathematically required to fill its
remaining positions.

Each request generates exactly 1,024 tokens with physical M6 DSpark. TTFT is
wall time from request start through the first emitted token. The MLX peak is
the allocator high-water mark after an explicit reset immediately before the
request; it includes the already installed fixed physical cache arena.

The historical `mia-015153d9-piecewise-1024x6-full-accept.json` receipt emits
only six tokens in one verification cycle. Its 43.523 tok/s value is a
micro-gate cycle-cost calculation, not sustained decode throughput, and is
explicitly excluded from this chart and from PR performance claims.

| Cold prompt | Source | Load | TTFT | Prefill | Prefill tok/s | Decode | Decode tok/s | Request | MLX peak | K5 accept | Cycles |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | `8f47da2e` | 82.19 s | 5.90 s | 5.87 s | 174.53 | 27.87 s | 36.74 | 33.74 s | 103.832 GB / 96.701 GiB | 770/1,270 (60.63%) | 254 |
| 16,384 | `8f47da2e` | 82.24 s | 88.48 s | 88.45 s | 185.23 | 44.34 s | 23.09 | 132.79 s | 103.931 GB / 96.794 GiB | 833/955 (87.23%) | 191 |
| 65,536 | `8f47da2e` | 85.04 s | 375.76 s | 375.72 s | 174.43 | 48.44 s | 21.14 | 424.16 s | 103.932 GB / 96.794 GiB | 818/1,030 (79.42%) | 206 |
| 131,072 | `cc6e1b19` | 81.72 s | 808.21 s | 808.17 s | 162.18 | 62.70 s | 16.33 | 870.88 s | 103.915 GB / 96.778 GiB | 835/945 (88.36%) | 189 |

Raw receipts:

- [`mia-8f47da2e-python-vocab-cold-1024x1024.json`](../../../bench/deepseek-v4-mia/mia-8f47da2e-python-vocab-cold-1024x1024.json)
- [`mia-8f47da2e-python-vocab-cold-16384x1024.json`](../../../bench/deepseek-v4-mia/mia-8f47da2e-python-vocab-cold-16384x1024.json)
- [`mia-8f47da2e-python-vocab-cold-65536x1024.json`](../../../bench/deepseek-v4-mia/mia-8f47da2e-python-vocab-cold-65536x1024.json)
- [`mia-cc6e1b19-python-vocab-cold-131072x1024.json`](../../../bench/deepseek-v4-mia/mia-cc6e1b19-python-vocab-cold-131072x1024.json)

The command shape for each independent arm was:

```bash
python3 /Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py \
  --plist /Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist \
  --child-timeout-seconds 1800 -- \
  .venv/bin/python scripts/deepseek_v4_dspark_k5_bench.py \
  --arm dspark --max-tokens 1024 \
  --prompt-tokens <1024|16384|65536|131072> \
  --prompt-mode python-vocab --python-prompt-tokens 1024 \
  --out <receipt.json>
```

## What the ladder says

On current head, prefill measures 174.53 tok/s at 1K, 185.23 tok/s at 16K,
and 174.43 tok/s at 64K. The fixed 384K target arena keeps the current-head
allocator peak essentially flat: the 16K and 64K arms differ by only 196,608
bytes. This is the intended vLLM-style ownership result—request length advances
logical page frontiers and block tables instead of growing or materializing a
contiguous cache. It also means the roughly 103.9 GB peak is paid at
installation rather than scaled to the individual request. The older 128K row
is not used as current-head proof for that claim.

The refreshed sustained decode results are 36.74 tok/s at 1K, 23.09 tok/s at
16K, and 21.14 tok/s at 64K. Relative to the prior `cc6e1b19` receipts, those
are improvements of 17.4%, 9.1%, and 36.0%, respectively. Decode is not an
attention-only context-length curve because useful tokens per physical M6
cycle vary with K5 acceptance: the refreshed rows accept 60.63%, 87.23%, and
79.42% of drafted future tokens. The 16.33 tok/s 128K result remains useful
historical evidence, but it must not be attributed to the current head.

## Remaining performance headroom

The retained 16-byte Trellis staging change preserves exact final bits. The
construction-bound piecewise target route compiles only cache-free regions and
keeps all 43 cache-owning attention calls eager. The full 1,024-output rows
above remain the only published throughput evidence. Current-head coverage
reaches 64K; the valid 128K receipt predates the final performance-path commits.
The long-context receipts still expose meaningful work in paged MLA attention
and in the number of physical verification cycles.

Further optimization should start with a fresh profile of this post-staging
stack, then change only the largest measured bucket. Promising categories are
source-faithful long-context MLA/page scheduling and improvements that preserve
K5 acceptance while reducing verify work. MoE changes should be reconsidered
only if the new profile still shows a sufficient ceiling. Any follow-up must
retain Mia's arithmetic, stock432 NVFP4 records, Mia132 index layout, physical
M6 ownership, and construction-bound routing; no silent enabled-path fallback
or per-token eligibility checks belong in that work.
