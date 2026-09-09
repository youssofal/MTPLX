# Table 1 — Decode rate by context size (fastest of 3 seeds)

Each value is the fastest seed decode rate in tokens per second. The range
is the slowest seed to the fastest seed. The path is served. The profile is
cli-resolved Turbo. Each cell is cold. The 16K cell is each arm's fastest
ABAB record. The 255K cell is above the memory knob on every arm; see Table 5.

| context | A decode (range) | C decode (range) | D decode (range) | E decode (range) |
| --- | --- | --- | --- | --- |
| 1K | 87.32 (73.66–87.32) | 88.62 (85.88–88.62) | 103.30 (97.46–103.30) | 111.02 (99.79–111.02) |
| 8K | 72.29 (62.44–72.29) | 85.95 (75.93–85.95) | 96.11 (87.63–96.11) | 98.75 (93.98–98.75) |
| 16K | 71.97 (70.98–71.97) | 82.84 (72.02–82.84) | 99.16 (95.30–99.16) | 104.65 (99.49–104.65) |
| 32K | 76.47 (69.65–76.47) | 81.30 (71.72–81.30) | 95.08 (91.79–95.08) | 102.63 (94.56–102.63) |
| 64K | 69.26 (65.68–69.26) | 84.43 (70.82–84.43) | 94.68 (81.68–94.68) | 102.50 (88.45–102.50) |
| 128K | 64.26 (59.70–64.26) | 68.62 (65.67–68.62) | 84.83 (80.02–84.83) | 87.97 (85.56–87.97) |

A is the control. C is #475. D and E are #478 at two thresholds.
