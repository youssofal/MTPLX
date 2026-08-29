# MTPLXApp Native Backend Foundation

This package is the native macOS foundation for MTPLX V1. It deliberately keeps
Swift as supervisor/client/state code: inference stays in the existing `mtplx`
daemon, and the future UI should bind to `MTPLXBackendStore` instead of parsing
raw backend JSON.

## Shape

- `MTPLXAppCore` contains daemon supervision, typed HTTP/SSE clients, settings,
  bounded logs, dashboard DTOs, and observable state.
- `MTPLXApp` is a minimal SwiftUI host target for debug launching only. Visual
  design, animation, and production layout are out of scope for this pass.

## Core Services

- `DaemonSupervisor` launches hidden `mtplx serve` with `Process`, captures
  stdout/stderr, probes `/health`, and stops gracefully before force kill.
- `MTPLXCommandBuilder` resolves the executable and builds safe serve arguments.
  It never passes browser/dashboard launch flags.
- `MTPLXAPIClient` owns typed calls for health, capabilities, snapshot,
  sessions, settings, cancel, cache clear, and session clear.
- `MetricsStreamClient` parses `/v1/mtplx/metrics/stream` SSE events with
  reconnect/backoff.
- `MTPLXBackendStore` is the future UI binding surface.
- `MTPLXSettingsStore` persists app config in
  `~/Library/Application Support/MTPLX/settings.json`.
- `BoundedLogStore` keeps daemon logs bounded.

Restart-required app settings live in `MTPLXAppConfiguration`: executable path,
model, model download directory, profile, host, port, generation mode, MTP loading, context window, API
key, thermal polling, stream cadence, Performance Lock, and launch behavior.
`MTPLXBackendStore.applyConfiguration(..., restartIfRunning:)` saves those
settings and restarts the daemon when the current process is running.

## Backend Contract

The native app uses the same backend truth as the web dashboard:

- `GET /health`
- `GET /metrics`
- `GET /admin/sessions`
- `POST /admin/cache/clear`
- `POST /admin/sessions/{session_id}/clear`
- `GET /v1/mtplx/snapshot`
- `GET /v1/mtplx/metrics/stream?snapshot_interval_ms=500`
- `GET /v1/mtplx/prefill_history`
- `POST /v1/mtplx/settings`
- `POST /v1/mtplx/cancel/{request_id}`
- `GET /v1/mtplx/app/capabilities`

Normal native cadence is 500 ms. Performance Lock should use 1000 ms. The
backend still defaults to 200 ms for the browser dashboard.

## Commands

```sh
swift test
./script/build_and_run.sh --verify
```

The run script builds a small `.app` bundle under `dist/` and launches that
bundle, rather than running the SwiftUI executable as a raw command-line
process.
