# Profiles

| Profile | Purpose |
|---|---|
| `turbo` | Default for the quantized 27B and 9B flagships (Optimized-Speed, Optimized-Quality, the legacy Optimized hybrid, and their FP16 siblings, plus the 9B Speed pair): Sustained plus the NAX verify kernels and context-routed compiled verify. Fastest decode profile; matches the macOS app's launch presets. |
| `sustained` | Default `mtplx start` mode for every other model: native-MTP long-context path with chunked prefill, final-token logits, request-sized paged KV, and the normal Apple fan controller. |
| `sustained` + `--max` | Sustained Max: the same long-context path with ThermalForge/TG Pro fans pinned while MTPLX runs. |
| `performance-cold` + `--max` | Burst: old max-fan headline lane, not recommended beyond 8K context. |
| `performance-cold` | Legacy burst path without fan boost. Kept for explicit flags and compatibility; not shown in first-run onboarding. |
| `stable` | Hidden conservative alias for the exact/staged long-reply path and compatibility fallback. |
| `exact` | QA and release exactness checks. |
| `max-diagnostic` | Fan-control diagnostics only. Product modes are Sustained, Sustained Max, and Burst. |

`--max` is separate from profiles. It is opt-in and must restore fan state on exit when supported.
