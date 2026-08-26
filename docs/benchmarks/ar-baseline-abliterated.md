# Autoregressive baseline: Abliterated build

Private notes, not part of the upstream PR.

`generate_greedy_batched(decode_mode="ar")` on
`grant-ai/Qwen3.8-27B-Abliterated-MTPLX-4bit`, matched to the
Optimized-Speed ladder: bundled `long_code` prompt, thinking on at effort
low, greedy, 512 tokens per stream, same process and timing path.

| n | AR Optimized-Speed | MTP Optimized-Speed | speedup | AR Abliterated | MTP Abliterated | speedup |
|---|---|---|---|---|---|---|
| 1 | 31.1 | 66.8 | 2.15x | 35.4 | 63.9 | 1.81x |
| 2 | 58.0 | 81.5 | 1.40x | 65.1 | 77.7 | 1.19x |
| 4 | 97.3 | 104.4 | 1.07x | 104.9 | 90.7 | 0.87x |
| 6 | 108.1 | 137.5 | 1.27x | 114.9 | 130.9 | 1.14x |
| 8 | 119.2 | 170.2 | 1.43x | 127.7 | 164.9 | 1.29x |
| 10 | 127.9 | 127.4 | 1.00x | 134.2 | 120.8 | 0.90x |
| 12 | 110.5 | 139.5 | 1.26x | 111.0 | 134.2 | 1.21x |
| 16 | 143.9 | 192.8 | 1.34x | 143.9 | 183.1 | 1.27x |
| 24 | 203.5 | 210.4 | 1.03x | 201.9 | 201.6 | 1.00x |
| 32 | 272.0 | 222.9 | 0.82x | 273.8 | 212.6 | 0.78x |

Findings:

* Abliterated is FASTER at plain autoregressive decoding below 12
  streams: 35.4 against 31.1 at one stream, 13.6% ahead. Its 4-bit
  group-64 trunk streams less scale and bias metadata per weight block
  than Optimized-Speed's group-32. The gap closes as concurrency rises
  and both converge by n = 16, where the matmul rather than weight
  streaming is the constraint.
* Abliterated is SLOWER with speculation at every cohort size, so its
  speedup ratio is worse on both counts: a faster denominator and a
  slower numerator. 1.81x at one stream against 2.15x.
* The crossover lands in the same place on both models. Parity at 24,
  AR ahead at 32. That is consistent with the crossover being set by
  `tau / (depth + 1)` rather than by anything model-specific.
* Practical read for this build: the LoRA-tuned head is not earning its
  keep relative to what the trunk quantization already gives. The
  head-quality work is worth more here than on Optimized-Speed, since
  the same trunk advantage would compound with better acceptance.
