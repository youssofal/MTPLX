# MTP Forge — Backend CLI + Provenance Contract

This document is the single source of truth between the MTPLX macOS app (frontend, lives under `apps/MTPLXApp/`) and the Python implementation of the `mtplx forge` CLI subcommand family (backend). It is the spec the frontend was written against — originally authored ahead of the backend; the `mtplx forge` subcommand family now ships (probe, build, discover, publish, inspect, verify, cancel), and this document remains the contract reference both sides are held to.

If the spec changes, this file is the place to amend it. The frontend's Swift wrappers (`ForgeBuilder.swift`, `HFPublisher.swift`, `ForgeDiscoveryService.swift`, `HuggingFaceProbe.forgeProbe`) all parse against the shapes documented below.

---

## 1. CLI Surface

All commands are subcommands of `mtplx forge`. The frontend resolves the `mtplx` executable via the same search-path logic the daemon launcher uses: explicit app setting first, installed `mtplx` on `$PATH` / Homebrew search dirs next, and the source checkout wrapper only when developer QA opts in with `MTPLX_APP_ALLOW_SOURCE_WRAPPER=1`.

| Subcommand | Purpose | Required | Frontend caller |
|------------|---------|----------|------------------|
| `forge --help` | Existence probe. Exit 0 ⇒ Forge enabled. | yes | `ForgeBuilder.backendAvailable()` |
| `forge probe <repo> --json` | Single-shot HF probe with source-format classification. | yes | currently unused (frontend probes directly via the public HF endpoints); reserved for the future when MTPLX wants a server-side classifier |
| `forge build` | The main pipeline (download → convert → calibrate → verify → brand). | yes | `ForgeBuilder.stream(_:)` |
| `forge brand` | Standalone brand stamp (called by `build`'s last phase; standalone for re-stamping). | optional V1 | reserved |
| `forge verify <path>` | Run only the verify pass on an existing local model (alias around `mtplx tune`). | optional V1 | reserved |
| `forge publish` | Upload a local forged artifact to Hugging Face. | yes | `HFPublisher.stream(_:token:)` |
| `forge discover [--query …] [--limit N] [--offset N] --json` | List MTPLX-branded HF models sorted by downloads desc. | yes | `ForgeDiscoveryService.discover(_:)` |
| `forge inspect <path> --json` | Dump a local artifact's `mtplx_runtime.json`. | optional V1 | reserved |
| `forge cancel <run_id>` | Best-effort SIGTERM of an in-flight `build` or `publish`. | optional | belt-and-braces for crashed-frontend recovery |

### Shared flags (per-subcommand, not universal)

- `--out <output-dir>` — root directory; required on `build` and `publish`, optional on `verify`. Frontend always passes `$TMPDIR/mtplx-forge` (build) or `$TMPDIR/mtplx-forge-publish` (publish).
- `--run-id <id>` — required on `build` and `publish`, optional on `verify`. Frontend-generated UUID prefix (`mtplx-forge-` / `mtplx-forge-publish-` + 8 hex chars). The combined run dir is `<output-dir>/<run-id>/`.
- `--max` — pin fans at max via the existing ThermalForge integration; `build` and `verify` only (`publish` takes no `--max`). Build always passes this when the user opted in.

`probe`, `inspect`, and `cancel` take none of these flags.

### Argv-only secrets

Hugging Face tokens are never passed on argv. The frontend passes `--token stdin` and writes the token to the subprocess's stdin (one line, newline-terminated, then EOF). The backend must read at most one line and never log it.

---

## 2. Progress reporting — files, not NDJSON

All long-running subcommands report progress by **writing per-phase JSON files to disk** under the run dir. **No NDJSON streaming, no progress on stdout.** Stderr is reserved for human-friendly text (the existing `mtplx tune` / `mtplx pull` convention). Stdout carries a single final JSON object on completion (only when `--json` is passed).

Frontend polls each known file every 500 ms via `FileManager.fileExists` + a `JSONSerialization.jsonObject` decode; if the file isn't there yet the poll iteration skips it. Backend should write atomically (write-then-rename) so the frontend never reads a half-written file.

### `forge build` run-dir layout

```text
<output-dir>/<run-id>/
├── download.json     # bytes_on_disk, total_bytes?, mb_per_s, eta_s?, label?, finished
├── convert.json      # progress (0..1), label? ("to_mlx" | "quantize_body"), finished
├── calibrate.json    # progress (0..1), label? ("extract_mtp" | "requantize_mtp" | "pack_sidecar"), finished, loss?, ppl?
├── verify.json       # { rows: [ { depth, tok_s, multiplier_vs_ar, acceptance_by_position, verify_time_s } ] }
├── build_outcome.json # phase, verdict, failure_reasons, message, diagnostic?, verify_rows, speed_evidence, ar_tok_s?, best_mtp_* , architecture_id?
├── brand.json        # { branded_name, runtime_metadata: { …full mtplx_runtime.json shape… } }
└── forge.json        # { local_path, runtime_metadata: { …final mtplx_runtime.json… } }
```

#### `download.json`

```jsonc
{
  "bytes_on_disk": 1_234_567_890,
  "total_bytes": 17_179_869_184,        // optional; omit if unknown
  "mb_per_s": 12.4,
  "eta_s": 1183.0,                       // optional
  "label": "downloading shard 3 of 9",   // optional, shown in the bar header
  "finished": false
}
```

Updates as frequently as you like; frontend de-dups on `bytes_on_disk` change so writing the same value repeatedly is fine.

#### `convert.json`, `calibrate.json`, `publish.json`

All share the same shape:

```jsonc
{
  "progress": 0.42,                      // 0..1, best-effort
  "label": "quantize_body",              // optional, free-form
  "finished": false,
  "loss": 0.12,                          // optional, calibrate.json only
  "ppl": 6.84                            // optional, calibrate.json only
}
```

The frontend uses `label` to derive which sub-phase to mark in-progress on the Convert / Calibrate checklists (`"to_mlx"`, `"quantize_body"`, `"extract_mtp"`, `"requantize_mtp"`, `"pack_sidecar"`). Stick to those tokens — adding new ones is fine, mis-spelling them will hide the row state from the user.

#### `verify.json`

```jsonc
{
  "rows": [
    {
      "depth": 0,                        // 0 == AR baseline
      "tok_s": 22.3,
      "multiplier_vs_ar": 1.0,
      "acceptance_by_position": [],      // empty for AR
      "verify_time_s": 41.2
    },
    {
      "depth": 1,
      "tok_s": 34.5,
      "multiplier_vs_ar": 1.55,
      "acceptance_by_position": [0.91],
      "verify_time_s": 18.4
    },
    { "depth": 2, "tok_s": 47.1, "multiplier_vs_ar": 2.11, "acceptance_by_position": [0.88, 0.71], "verify_time_s": 13.5 },
    { "depth": 3, "tok_s": 49.6, "multiplier_vs_ar": 2.22, "acceptance_by_position": [0.86, 0.69, 0.42], "verify_time_s": 12.0 }
  ]
}
```

Frontend de-dups on `depth`. Backend can append rows incrementally — the file is read fresh on every poll, so appending and re-writing the array is fine.

#### `brand.json`

```jsonc
{
  "branded_name": "Qwen3.6-35B-A3B-MTPLX-Speed",
  "runtime_metadata": { /* see section 3 */ }
}
```

Frontend uses this to surface the metadata-preview card on the Brand stage before the build actually completes.

#### `forge.json` (terminal, success only)

```jsonc
{
  "local_path": "/Users/me/Documents/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Speed",
  "runtime_metadata": { /* see section 3 */ }
}
```

Written exactly once, on successful completion, just before the process exits 0.

### `forge publish` run-dir layout

```text
<output-dir>/<run-id>/
├── publish.json     # bytes_uploaded, total_bytes?, mb_per_s, repo_created?, revision?, finished?
└── README.md         # optional; frontend writes this here if the user supplied a custom README
```

`publish.json` shape:

```jsonc
{
  "bytes_uploaded": 12345678,
  "total_bytes": 17000000000,
  "mb_per_s": 18.6,
  "repo_created": "youssofal/Qwen3.6-35B-A3B-MTPLX-Speed",  // optional; emit when repo creation succeeds
  "revision": "abc1234",                                       // optional
  "finished": false,
  "repo": "youssofal/Qwen3.6-35B-A3B-MTPLX-Speed"             // backend can include the final repo here too
}
```

### Cancellation

On SIGINT / SIGTERM the backend should:

1. Cancel any in-flight network IO.
2. Leave partial files on disk for resume.
3. Exit non-zero (the frontend treats `terminationReason == .uncaughtSignal` as `.cancelled`).

### Backend-not-available detection

When the user is on a pre-Forge MTPLX install, argparse exits with code 2 and prints `argument: invalid choice 'forge'`. The frontend matches this exact pattern in stderr and surfaces a clean "Forge backend not available" empty state. Don't change the exit code or the error string without updating the invalid-choice matchers in `ForgeBuilder.swift` / `HFPublisher.swift` / `ForgeDiscoveryService.swift`.

---

## 3. `mtplx_runtime.json` schema

The runtime metadata schema is **additive**. Every existing field stays in place (verified verbatim against `<build-machine>/MTPLX/models/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP/mtplx_runtime.json` and `<build-machine>/MTPLX/models/Qwen3.6-27B-MTPLX-Flat4-CyanKiwiMTP/mtplx_runtime.json` — historical build-machine references kept as the verification record, not shipped paths). Forge **adds** a `forge_provenance` block and reuses the existing `speed_evidence` / `sampler` / `verified_on` blocks for verification numbers.

### Verified existing spine (do not rename)

```jsonc
{
  "mtplx_version": "2.4.2",
  "arch_id": "qwen3-next-mtp",
  "mtp_depth_max": 3,
  "recommended_profile": "performance-cold",   // or "stable"
  "sampler": { "temperature": 0.6, "top_p": 0.95, "top_k": 20 },
  "verified_on": {
    "timestamp": "2026-05-25T22:45:00+0100",
    "hardware": "Apple M5 Max, 128 GB unified memory",
    "machine_arch": "arm64",
    "macos": "26.3.1",
    "model": "<branded-name>"
  },
  "exactness_baseline": { /* free-form per build */ },
  "speed_evidence": { /* free-form per build */ },
  "mtp_sidecar": "<descriptive-sidecar-id>",
  "base_trunk": "<base-trunk-hf-repo-or-local>",
  "artifact_role": "<short tag>"
}
```

### New `forge_provenance` block (additive)

```jsonc
{
  "forge_provenance": {
    "source_repo": "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit",
    "source_sha": "7a1c0c26c56ee56f98bfdb77124acf5b239eabf3",
    "source_format": "compressed_tensors_awq",  // one of: bf16_native | mlx_affine | mlx_affine_with_mtp | compressed_tensors_awq | hf_vllm | unknown
    "forge_recipe": {
      "body_bits": 4,
      "body_group_size": 64,
      "body_mode": "affine",
      "mtp_policy": "extract_from_sidecar"      // one of: keep_bf16 | extract_from_sidecar | requantize
    },
    "forge_inputs": {
      "trunk_path": "/...",
      "mtp_source_path": "/..."
    },
    "forged_at": "2026-05-25T22:45:00+0100",     // ISO 8601 string (matches verified_on.timestamp convention)
    "mtplx_version": "2.4.2",
    "forged_locally": true,
    "published_to_hf": null                      // or the nested object below
  }
}
```

After a successful publish:

```jsonc
"published_to_hf": {
  "repo": "youssofal/Qwen3.6-35B-A3B-MTPLX-Speed",
  "revision": "abc1234",
  "visibility": "public",                  // public | private
  "license_spdx": "apache-2.0",
  "uploaded_at": "2026-05-25T23:00:00+0100"
}
```

Frontend's `MTPLXForgeProvenance` Codable (`apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXForgeProvenance.swift`) enforces snake_case keys verbatim and round-trip tests guard against drift.

### Storing verification numbers

The frontend's `ForgeOrchestrator.extractForgeVerification` reads `speed_evidence` for:

- `acceptance_by_depth: [float, ...]` — per-depth acceptance fractions for the winning candidate
- `tok_s: [float, ...]` — measured tok/s for the winning candidate (max value wins)
- `depth: int` — winning depth
- `greedy_diagnostic.tok_s: float` — AR baseline tok/s

These mirror the existing Flat4 / GDN8-Speed4 fixtures exactly, so the Python agent can re-use the existing build pipelines' speed_evidence emit code. **Do NOT** put verification numbers under `forge_provenance` — they have a home in the spine.

---

## 4. `forge discover` JSON

```jsonc
[
  {
    "repo": "youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
    "owner": "youssofal",
    "branded_name": "Qwen3.6-27B-MTPLX-Optimized-Speed",
    "downloads": 12450,
    "size_bytes": 16106127360,
    "depth": 3,                            // optional; from mtplx_runtime.json.mtp_depth_max if discoverable
    "multiplier_vs_ar": 2.54,              // optional; from speed_evidence if discoverable
    "license": "apache-2.0",               // optional
    "last_updated": "2026-04-29T14:37:00Z" // optional
  }
]
```

Backend behaviour:

- Query is `huggingface_hub.HfApi.list_models(search="MTPLX", filter="*-MTPLX-*", sort="downloads", direction=-1, limit=…, offset=…)`.
- No curated allow-list. The repo-name pattern is what keeps the wall clean.
- On HF unreachable (DNS failure, connection refused, timeout) exit non-zero AND include one of `hf_unreachable`, `name resolution failed`, `connection refused` in stderr so the frontend's `ForgeDiscoveryService.discover(_:)` raises `.hfUnreachable` cleanly.
- Pagination is offset-based (`--offset`); frontend may issue multiple calls for infinite scroll.

---

## 5. Reference pipelines (don't start from scratch)

For the build pipeline, generalise the existing one-off scripts (the paths
below are historical build-machine references, not shipped package paths):

- `<build-machine>/MTPLX/scripts/build_flat4_cyankiwi_mtp_requant.py` — the 27B requant path that built `Qwen3.6-27B-MTPLX-Optimized-Speed`. Handles the MLX-affine → MLX-affine requantisation case (body bits picker, MTP sidecar repack, runtime_metadata write, trunk symlink). Forge's bf16Native / mlxAffine / mlxAffineWithMtp source-format paths all collapse to variations of this script.
- (To be written) **35B compressed-tensors AWQ → MLX-affine** — genuinely new work. The existing 35B artifacts (`<build-machine>/MTPLX/models/Qwen3.6-35B-A3B-MTPLX-Official-4bit-*`) already pass the MoE MTP gate (commits `939b537`, `9f5b7be`, `739415b`) so the runtime side is ready; only the conversion is missing.

---

## 6. Hard rule from mlx-lm PR #990 reviewer evidence

Quantising MTP weights collapses MoE acceptance to **5-11%** (vs 79-85% with BF16 MTP). The frontend's PlanStage defaults `mtp_policy: keep_bf16` and surfaces a loud warning chip + checkbox if the user overrides to `requantize`. The backend MUST refuse a build whose recipe has `mtp_policy: requantize` UNLESS `--allow-degraded-mtp` is passed:

```bash
mtplx forge build --repo cyankiwi/X --out "$TMPDIR/mtplx-forge" --run-id mtplx-forge-01234567 \
  --branded-name X-MTPLX-Speed --recipe '{"mtp_policy":"requantize",...}'
# exits non-zero with: "MTP policy 'requantize' degrades acceptance; pass --allow-degraded-mtp to confirm"
```

The frontend already passes `--allow-degraded-mtp` when `state.recipe.degradesMtp && state.hasAcknowledgedDegradedMTP` (see `ForgeOrchestrator.startBuild`). Don't ship a build mode that silently quantises MTP — the resulting artifact reads as MTPLX-branded but performs worse than autoregressive baseline.

---

## 7. Frontend touch points (for cross-reference)

- `apps/MTPLXApp/Sources/MTPLXAppCore/Forge/ForgeBuilder.swift` — argv + file-polling for `forge build`
- `apps/MTPLXApp/Sources/MTPLXAppCore/Forge/HFPublisher.swift` — argv + file-polling for `forge publish`
- `apps/MTPLXApp/Sources/MTPLXAppCore/Forge/ForgeDiscoveryService.swift` — argv + JSON-array decode for `forge discover`
- `apps/MTPLXApp/Sources/MTPLXAppCore/Onboarding/HuggingFaceProbe.swift` — currently does its own HF probing; will call `forge probe` later if the backend wants to centralise the classifier
- `apps/MTPLXApp/Sources/MTPLXAppCore/Models/MTPLXForgeProvenance.swift` — Codable for the `forge_provenance` block + a generic `MTPLXRuntimeMetadata.parse(_:)` for the rest of the file
- `apps/MTPLXApp/Sources/MTPLXAppCore/Forge/ForgeLocalIndex.swift` — local scan that picks up any model dir whose `mtplx_runtime.json` has `forge_provenance.forged_locally == true`

When you ship the Python side, run the existing macOS app against it; the wizard should walk from Source → Plan → Convert → Calibrate → Verify → Brand → Registered with live progress for every phase and surface the published `mtplx_runtime.json` in the My Models browser.
