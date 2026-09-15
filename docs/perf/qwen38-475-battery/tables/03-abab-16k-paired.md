# Table 3 — 16K ABAB (fastest record) and paired deltas (mean)

The schedule is A, C, D, E, three rounds. Each arm has nine windows. The
decode below is each arm's fastest record, with the slow-to-fast range over
the nine windows. The paired delta is the mean of the nine per-seed deltas.
The interval is a 95% t-interval on that mean. The delta math is the mean
math; it is unchanged. A positive delta means the candidate is faster.

## Per-arm 16K decode (fastest of 9 windows)

| arm | decode | range | MTP accept | peak GB |
| --- | ---: | --- | ---: | ---: |
| A: release 2.11.2 | 71.97 | 70.98–71.97 | 0.478 | 93.41 |
| C: +#475 aux | 82.84 | 72.02–82.84 | 0.473 | 93.36 |
| D: +#478 @0.2 | 99.16 | 95.30–99.16 | 0.651 | 92.28 |
| E: +#478 @0.09 | 104.65 | 99.49–104.65 | 0.703 | 92.28 |

## Paired deltas (mean of 9 per-seed cells)

| pair | mean Δ tok/s | Δ% | 95% CI (tok/s) | verdict |
| --- | ---: | ---: | --- | --- |
| C − A (#475 vs release) | +5.325 | +7.44% | [+1.762, +8.889] | candidate FASTER (CI excludes 0) |
| D − A (#478 @0.2 vs release) | +25.207 | +35.21% | [+23.996, +26.418] | candidate FASTER (CI excludes 0) |
| E − A (#478 @0.09 vs release) | +30.629 | +42.79% | [+29.050, +32.207] | candidate FASTER (CI excludes 0) |
| D − C (#478 headline @0.2) | +19.882 | +26.28% | [+15.692, +24.072] | candidate FASTER (CI excludes 0) |
| E − C (#478 headline @0.09) | +25.303 | +33.19% | [+23.148, +27.459] | candidate FASTER (CI excludes 0) |

