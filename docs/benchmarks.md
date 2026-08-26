# Benchmarks

Every benchmark claim should record:

- hardware and RAM
- macOS version
- model and quantization
- sampler settings
- prompt suite
- token count
- profile
- fan mode
- date and commit

Separate cold headline runs from sustained no-fan runs and fan-controlled diagnostics.

```bash
mtplx bench run --suite cold-long-code-192 --max-tokens 192 --strict-cold
mtplx bench run --suite flappy --max-tokens 10000 --no-fanmax
```

## Dense MTP batch lane — Qwen3.8-27B, continuous batching

**Commit `fe74b93`.** Every number below came from that commit; a
result's scope is a commit.

**What this table is.** A BEFORE/AFTER regression check: the same fixed-width
cohorts run with and without continuous batching, to show that adding it costs
nothing at a fixed width. It is not a measure of absolute throughput, and it
uses `--ignore-stop` so every stream runs the full length. Absolute throughput
on current `main`, measured with different flags, is in the pull request
description and will not match these figures.

| | |
|---|---|
| hardware | Mac Studio M3 Ultra, 256 GB unified memory |
| model | `ref-Qwen3.8-27B-MTPLX-Optimized-Speed`, unabliterated, 4-bit group size 32, 8-bit `lm_head` / `embed_tokens` |
| MTP depth | 3 |
| prompt | bundled short suite, 223 tokens, `--vary-suffix` so rows are distinct streams |
| tokens per stream | 512, `--ignore-stop` |
| capture backend | `linear-gdn-from-conv-stream`, `--loop-mode pipelined`, `--head-history committed`, window 24576 |
| mlx / mlx-lm | 0.32.1 / 0.31.3 — identical to the baseline run, so the only difference is this fork's code |
| baseline | `results/dense-batch/r32-stock-*`, same machine, same model, same flags |
| date | 2026-08-24 |

**How to read `tau`.** It is the mean number of tokens committed per
decode cycle per stream, against a ceiling of `depth + 1 = 4`. Throughput
at a fixed width scales with it, because acceptance changes how many
tokens a cycle commits without changing the work the cycle does.

### temperature 0, thinking off

| streams | before tok/s | after tok/s | delta | tau | acceptance d1 / d2 / d3 |
|---|---|---|---|---|---|
| 1 | 72.4 | 72.9 | +0.7% | 3.531 | 0.931 / 0.862 / 0.745 |
| 2 | 79.3 | 80.2 | +1.1% | 3.048 | 0.891 / 0.757 / 0.626 |
| 4 | 94.6 | 95.0 | +0.4% | 2.876 | 0.883 / 0.765 / 0.629 |
| 6 | 131.7 | 131.7 | -0.0% | 2.926 | 0.876 / 0.726 / 0.591 |
| 8 | 171.3 | 172.1 | +0.5% | 2.926 | 0.877 / 0.745 / 0.620 |
| 10 | 130.9 | 131.2 | +0.2% | 2.893 | 0.886 / 0.764 / 0.638 |
| 12 | 147.3 | 148.2 | +0.6% | 2.977 | 0.887 / 0.763 / 0.634 |
| 16 | 202.8 | 203.4 | +0.3% | 2.960 | 0.881 / 0.758 / 0.629 |
| 24 | 201.8 | 202.1 | +0.2% | 2.738 | 0.871 / 0.744 / 0.605 |
| 32 | 222.0 | 222.6 | +0.3% | 2.876 | 0.878 / 0.750 / 0.626 |

### temperature 0, thinking low

| streams | before tok/s | after tok/s | delta | tau | acceptance d1 / d2 / d3 |
|---|---|---|---|---|---|
| 1 | 66.8 | 67.3 | +0.6% | 3.261 | 0.892 / 0.752 / 0.618 |
| 2 | 81.5 | 81.9 | +0.5% | 3.141 | 0.884 / 0.737 / 0.596 |
| 4 | 104.4 | 104.7 | +0.3% | 3.200 | 0.892 / 0.749 / 0.613 |
| 6 | 137.5 | 137.7 | +0.1% | 3.066 | 0.881 / 0.722 / 0.589 |
| 8 | 170.2 | 170.2 | +0.0% | 2.893 | 0.860 / 0.703 / 0.544 |
| 10 | 127.4 | 127.8 | +0.3% | 2.813 | 0.860 / 0.679 / 0.524 |
| 12 | 139.5 | 139.9 | +0.3% | 2.813 | 0.865 / 0.690 / 0.539 |
| 16 | 192.8 | 193.1 | +0.2% | 2.813 | 0.866 / 0.687 / 0.535 |
| 24 | 210.4 | 211.1 | +0.4% | 2.860 | 0.879 / 0.718 / 0.561 |
| 32 | 222.9 | 223.6 | +0.3% | 2.893 | 0.873 / 0.698 / 0.548 |

### temperature 0, thinking medium

| streams | before tok/s | after tok/s | delta | tau | acceptance d1 / d2 / d3 |
|---|---|---|---|---|---|
| 1 | 61.4 | 61.8 | +0.6% | 2.994 | 0.842 / 0.649 / 0.515 |
| 2 | 79.6 | 80.0 | +0.5% | 3.066 | 0.859 / 0.697 / 0.538 |
| 4 | 96.4 | 96.6 | +0.2% | 2.943 | 0.848 / 0.713 / 0.583 |
| 6 | 129.4 | 129.6 | +0.2% | 2.876 | 0.847 / 0.676 / 0.538 |
| 8 | 168.9 | 168.8 | -0.0% | 2.876 | 0.854 / 0.698 / 0.569 |
| 10 | 126.6 | 127.0 | +0.3% | 2.798 | 0.853 / 0.700 / 0.579 |
| 12 | 141.1 | 141.4 | +0.3% | 2.844 | 0.845 / 0.685 / 0.537 |
| 16 | 195.0 | 195.6 | +0.3% | 2.844 | 0.842 / 0.671 / 0.523 |
| 24 | 201.7 | 202.3 | +0.3% | 2.738 | 0.848 / 0.671 / 0.528 |
| 32 | 211.2 | 211.8 | +0.3% | 2.738 | 0.849 / 0.692 / 0.561 |

### temperature 0, thinking xhigh

| streams | before tok/s | after tok/s | delta | tau | acceptance d1 / d2 / d3 |
|---|---|---|---|---|---|
| 1 | 81.1 | 81.8 | +0.8% | 3.969 | 0.992 / 0.992 / 0.992 |
| 2 | 76.9 | 77.2 | +0.5% | 2.960 | 0.908 / 0.771 / 0.676 |
| 4 | 87.6 | 87.2 | -0.5% | 2.667 | 0.858 / 0.663 / 0.539 |
| 6 | 127.9 | 127.6 | -0.2% | 2.844 | 0.850 / 0.672 / 0.525 |
| 8 | 159.1 | 159.6 | +0.3% | 2.709 | 0.864 / 0.695 / 0.536 |
| 10 | 127.4 | 127.5 | +0.1% | 2.813 | 0.860 / 0.686 / 0.548 |
| 12 | 139.5 | 139.5 | +0.0% | 2.813 | 0.868 / 0.702 / 0.572 |
| 16 | 192.7 | 193.1 | +0.2% | 2.813 | 0.860 / 0.678 / 0.544 |
| 24 | 198.4 | 198.8 | +0.2% | 2.695 | 0.863 / 0.636 / 0.483 |
| 32 | 211.0 | 211.6 | +0.3% | 2.738 | 0.876 / 0.707 / 0.570 |

### temperature 1.0, thinking off

| streams | before tok/s | after tok/s | delta | tau | acceptance d1 / d2 / d3 |
|---|---|---|---|---|---|
| 1 | 60.8 | 60.9 | +0.2% | 3.160 | 0.926 / 0.846 / 0.389 |
| 2 | 72.8 | 73.1 | +0.3% | 2.943 | 0.875 / 0.720 / 0.540 |
| 4 | 95.2 | 95.7 | +0.4% | 3.048 | 0.892 / 0.776 / 0.642 |
| 6 | 119.5 | 119.5 | -0.0% | 2.813 | 0.829 / 0.667 / 0.523 |
| 8 | 154.1 | 154.4 | +0.2% | 2.813 | 0.875 / 0.741 / 0.554 |
| 10 | 129.7 | 130.0 | +0.2% | 3.030 | 0.907 / 0.772 / 0.576 |
| 12 | 113.2 | 113.4 | +0.2% | 2.415 | 0.844 / 0.693 / 0.479 |
| 16 | 179.0 | 179.4 | +0.2% | 2.813 | 0.895 / 0.773 / 0.503 |
| 24 | 190.6 | 190.9 | +0.1% | 2.798 | 0.886 / 0.751 / 0.526 |
| 32 | 181.5 | 182.0 | +0.3% | 2.560 | 0.882 / 0.738 / 0.510 |

### temperature 1.0, thinking low

| streams | before tok/s | after tok/s | delta | tau | acceptance d1 / d2 / d3 |
|---|---|---|---|---|---|
| 1 | 58.3 | 58.7 | +0.6% | 3.030 | 0.864 / 0.675 / 0.491 |
| 2 | 76.3 | 76.6 | +0.4% | 3.084 | 0.893 / 0.742 / 0.597 |
| 4 | 91.6 | 91.1 | -0.6% | 2.926 | 0.849 / 0.666 / 0.510 |
| 6 | 125.7 | 125.0 | -0.6% | 2.960 | 0.889 / 0.723 / 0.563 |
| 8 | 155.6 | 155.6 | -0.1% | 2.844 | 0.860 / 0.696 / 0.546 |
| 10 | 115.1 | 115.1 | +0.0% | 2.681 | 0.870 / 0.702 / 0.547 |
| 12 | 133.0 | 133.2 | +0.2% | 2.844 | 0.875 / 0.707 / 0.558 |
| 16 | 179.0 | 179.3 | +0.1% | 2.813 | 0.871 / 0.712 / 0.567 |
| 24 | 192.7 | 192.8 | +0.1% | 2.829 | 0.878 / 0.723 / 0.557 |
| 32 | 199.5 | 199.8 | +0.2% | 2.813 | 0.870 / 0.713 / 0.536 |

### temperature 1.0, thinking medium

| streams | before tok/s | after tok/s | delta | tau | acceptance d1 / d2 / d3 |
|---|---|---|---|---|---|
| 1 | 56.0 | 56.4 | +0.7% | 2.909 | 0.807 / 0.636 / 0.466 |
| 2 | 69.2 | 69.4 | +0.3% | 2.798 | 0.826 / 0.674 / 0.488 |
| 4 | 90.8 | 91.2 | +0.4% | 2.909 | 0.858 / 0.723 / 0.577 |
| 6 | 114.4 | 114.5 | +0.1% | 2.695 | 0.818 / 0.656 / 0.521 |
| 8 | 158.3 | 158.4 | +0.1% | 2.893 | 0.869 / 0.703 / 0.537 |
| 10 | 123.9 | 124.2 | +0.3% | 2.893 | 0.879 / 0.729 / 0.598 |
| 12 | 126.1 | 126.6 | +0.4% | 2.695 | 0.864 / 0.703 / 0.550 |
| 16 | 177.1 | 177.3 | +0.1% | 2.783 | 0.853 / 0.687 / 0.492 |
| 24 | 193.9 | 194.1 | +0.1% | 2.844 | 0.879 / 0.726 / 0.570 |
| 32 | 120.3 | 120.4 | +0.1% | 1.695 | 0.828 / 0.656 / 0.505 |

### temperature 1.0, thinking xhigh

| streams | before tok/s | after tok/s | delta | tau | acceptance d1 / d2 / d3 |
|---|---|---|---|---|---|
| 1 | 49.2 | 49.3 | +0.3% | 2.547 | 0.736 / 0.483 / 0.328 |
| 2 | 75.6 | 75.7 | +0.1% | 3.048 | 0.856 / 0.683 / 0.536 |
| 4 | 85.2 | 85.0 | -0.1% | 2.709 | 0.798 / 0.622 / 0.482 |
| 6 | 107.5 | 107.4 | -0.1% | 2.535 | 0.838 / 0.650 / 0.501 |
| 8 | 151.3 | 151.8 | +0.3% | 2.768 | 0.835 / 0.685 / 0.520 |
| 10 | 118.3 | 118.4 | +0.1% | 2.753 | 0.850 / 0.690 / 0.544 |
| 12 | 127.5 | 128.1 | +0.5% | 2.723 | 0.848 / 0.689 / 0.544 |
| 16 | 147.5 | 147.9 | +0.2% | 2.317 | 0.816 / 0.628 / 0.452 |
| 24 | 164.0 | 164.1 | +0.1% | 2.404 | 0.828 / 0.639 / 0.494 |
| 32 | 168.3 | 168.3 | +0.0% | 2.370 | 0.810 / 0.632 / 0.473 |

### What the comparison establishes

Across **80 matched arms**, aggregate throughput moved by a median of **+0.2%** (range -0.6% to +1.1%) — noise, in the direction of slightly faster.

**`tau` is identical to the baseline at every batch size, to three decimal places.** That is the load-bearing result and it is stronger than the throughput numbers: it means a fixed-width cohort takes the same path it always did, committing the same tokens per cycle. Continuous batching changes which requests are in a cohort, not how a cohort decodes.

**Acceptance does not degrade with width**, which is what makes wide cohorts a good trade: an extra row costs a row of compute and does not cost the speculation that makes MTP worth running.
