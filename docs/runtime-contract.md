# Runtime Contract

Verified models include `mtplx_runtime.json`.

```json
{
  "mtplx_version": "2.4.2",
  "arch_id": "qwen3-next-mtp",
  "mtp_depth_max": 3,
  "recommended_profile": "stable",
  "exactness_baseline": {
    "context": 2048,
    "max_abs_diff": 0.0
  },
  "verified_on": {
    "timestamp": "2026-05-02T00:00:00Z",
    "hardware": "Apple Silicon",
    "macos": "macOS"
  }
}
```

`mtplx_version` is stamped with the runtime's real version at build time, and
`recommended_profile` above is illustrative — `turbo` is the shipped default
profile for the quantized flagships.

Architecture-compatible models without this contract are not supported by default.
