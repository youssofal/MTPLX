# Escha-W2 serving — evidence

`EschaLabs/Qwen3.6-35B-A3B-Escha-W2` served through `mtplx.runtime.load(path, mtp=True)` on an
M5 Max (128 GB). Recommended path: autoregressive (lossless). Measured 2026-08-05.

## Improvements (before → after)

| change | before | after | notes |
| --- | --- | --- | --- |
| int8 non-experts, fused matvec (no dequant-at-load) | 25.1 tok/s | 59.6 tok/s | 2.37×, −2 GiB, quality-preserved |
| eschamoe compute fp32 → bf16 (native dtype) | — | — | identical greedy output, drops the cast storm |
| `mx.compile` decode chain (`ESCHA_COMPILE`) | 56.5 | 58.5 tok/s | bit-identical (same output SHA) |
| async-eval AR decode (`MTPLX_AR_ASYNC_PIPELINE`) | 49.4 | 56.5 tok/s | bit-identical; overlaps host encode w/ GPU |
| chunked prefill (bounded activation memory) | 125 GiB @32k | 22.4 GiB @32k | prefill no longer O(ctx) in memory |
| base `qwen3_5_mtp` MTP injector object level | broken (AttributeError) | fixed | also repairs the non-escha A3B path |

Decode @1024, cumulative: 49.4 (sync/eager) → 56.5 (async) → **58.5 tok/s** (async + compile), all lossless.

## Benchmark — prefill tok/s, decode tok/s, peak memory (1024 → 128k)

| context | prefill tok/s | decode tok/s | peak memory |
| ---: | ---: | ---: | ---: |
| 1,024 | 428 | 58.1 | 16.2 GiB |
| 16,384 | 446 | 53.6 | 20.8 GiB |
| 32,768 | 425 | 49.5 | 22.4 GiB |
| 49,152 | 414 | 46.5 | 25.8 GiB |
| 65,536 | 399 | 43.2 | 29.1 GiB |
| 81,920 | 384 | 40.8 | 32.4 GiB |
| 98,304 | 371 | 38.7 | 35.8 GiB |
| 114,688 | 358 | 36.7 | 39.0 GiB |
| 131,072 | 345 | 34.8 | 42.4 GiB |

Peak memory grows with the KV cache (16.2 GiB @1k → 42.4 GiB @128k), not with prefill activations —
chunked prefill keeps the activation working set bounded. Decode tok/s declines as the KV grows
(bandwidth-bound); prefill holds ~345–446 tok/s.

## Example generation

Prompt (chat template): *Write a Python function `is_palindrome(s)` that returns True if the string
is a palindrome, ignoring case, spaces, and punctuation. Include a docstring and two example asserts.*

Output (94 tokens, clean EOS):

```python
def is_palindrome(s):
    """Returns True if the string is a palindrome, ignoring case, spaces, and punctuation."""
    cleaned = ''.join(char.lower() for char in s if char.isalnum())
    return cleaned == cleaned[::-1]

assert is_palindrome("A man, a plan, a canal: Panama") == True
assert is_palindrome("racecar") == True
```

Both assertions hold.

## MTP

The native MTP draft head binds and drafts (short-context spec-decode is bit-exact). Long-context
spec-decode is experimental — its multi-token verify uses a different attention kernel than
single-token decode, which can flip a near-tie greedy argmax. Serve AR until that is addressed.
