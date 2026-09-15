# Qwen3.8 Flash-Next serving battery

This directory holds the final battery outputs for the Qwen3.8 Flash-Next
serving optimizations. Every number comes from the real `mtplx serve` path.

## The cell

The canonical served cell is a 16,384 token templated prompt with 1,024 output
tokens, temperature 1, top-p 0.95, top-k 20, reasoning effort `xhigh`, native MTP
depth 3, and a pre-warmed n-gram table. Cross-request prefix restore is off, so
each seed prefills cold. The host holds a 40 degree Celsius thermal floor with
the fans at maximum. The context sweep uses 1,024, 8,192, 16,384, 32,768, 65,536 and
131,072 tokens at the same output length.

## The fastest-of rule

The reported rate is the fastest seed, shown with the slow-to-fast range. TTFT
and wall time are the fastest. Peak memory is the highest. The 16,384-token cell is
each arm's fastest ABAB record over its nine windows. The paired 95% interval in
Table 3 is a t-interval on the mean of the per-seed deltas; the delta itself is
the mean delta. A positive delta means the candidate is faster.

## The arms

| arm | engine | serve SHA |
| --- | --- | --- |
| A | release 2.11.2 (control), typical off | `21be78b3` |
| C | release + the #475 aux optimizations, typical off | `27d5ff6b` |
| D | C + #478 typical acceptance at 0.2 | `264e0835` |
| E | C + #478 typical acceptance at 0.09 | `264e0835` |

Arm A is the out-of-the-box release control. Arm C adds this branch's two exact
decode optimizations on top of the PR 391 remainder optimizations already in the
release line. Arms D and E add the #478 typical-acceptance optimization on top of
C at two thresholds.

"Exact" there is the optimization class: an exact optimization gives output that
is byte-for-byte the stock path, against a rounding-class optimization, whose
output differs from stock only by floating-point rounding. It is a different
sense from the exact acceptance law of #478 and #485, and it is never used to
label an arm: arms A and C are rounding-class builds overall, and their rows say
which acceptance mode is off rather than calling the arm exact.

## Contents

- `charts/` — the per-metric SVG charts, the 16K per-window interleave chart
  (`decode_16k_windows.svg`), and `manifest.json` (each chart's key, label,
  caption, alt text, and the engines and seed counts behind it). The chart inputs
  are filtered before plotting: superseded, failed, warm, and over-knob records
  are excluded, and each cell's remaining record count is asserted.
- `tables/` — the result tables (see `tables/README.md`): decode by context, the
  full per-arm metric ladder, the 16K ABAB paired deltas, the arm-C knock-outs,
  and the 255K long-context verdict.
- `abab/reports/` — the paired ABAB reports per arm pair.
