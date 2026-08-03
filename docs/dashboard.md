# MTPLX Live Dashboard

A gorgeous, dopamine-tuned web UI for watching an active MTPLX inference
server in real time. Built into the same `mtplx serve` process — no extra
ports, no extra binaries, no extra setup.

## Quickstart

```bash
# Easiest: pick "Live Dashboard" at step 3 of `mtplx start`.
mtplx start

# Or, if a server is already running:
mtplx dashboard

# Or, with full control:
mtplx serve --port 8000  # any client (Web UI, OpenAI SDK, hippo) drives load
mtplx dashboard          # opens http://127.0.0.1:8000/dashboard in your browser
```

Default URL: <http://127.0.0.1:8000/dashboard>.

## What you see

- **Overview** — live decode TPS gauge (canvas, 270° spring-tuned),
  5-minute TPS time-series with min/max markers, tokens-served lifetime
  tile, in-flight count, context-window utilization, last-request
  snapshot, session info, plus a sticky bottom one-liner (`N tok ·
  ttft Xs · Y tok/s · Z lifetime req`).
- **Speculative** — per-depth acceptance bars (with mean P(accept)
  overlay), verify-cycle waterfall decomposing `verify_time_s` into
  forward / logits / hidden / target-distribution / unattributed /
  accept / repair / snapshot / capture-commit / rollback, drafted/verify
  ratio, correction-vs-bonus tile, decode-vs-request tok/s, and the
  vs-vLLM oracle panel (only renders when Qwen3.6-27B is loaded).
- **Cache** — 24-slot SessionBank grid with per-slot session-id, prefix
  len, hits, bytes, age, and a one-click evict button; eviction-reason
  histogram with `CacheMissReason`-aware tooltips (POLICY_MISMATCH,
  TEMPLATE_MISMATCH, etc.); cumulative cached tokens; cache hit rate;
  prefill tok/s sparkline; TTFT distribution; context utilization bar.
- **Memory** — hardware banner (chip from `sysctl machdep.cpu.brand_string`,
  machine model from `hw.model`, unified memory bytes, profile, context
  window); MLX active + cache + peak +
  headroom stacked bar with peak tick.
- **Thermal** — twin fan rings (only when `--enable-thermal-poll` is
  on); Universal Thermal Rule banner when a request is in flight and
  fans aren't verified up; honest placeholder for GPU MHz (powermetrics
  integration is a v2 add).
- **Requests** — live in-flight list with per-request Cancel button
  (best-effort: flips the worker's `cancel_event`), and a paginated
  recent-requests table with expandable JSON rows.
- **Settings** — mutable defaults (depth, temperature, top_p, top_k,
  stream_interval, enable_thinking, reasoning_parser) with debounced
  writes; restart-required dialog with copy-CLI affordance for the
  immutable knobs (profile, model, MTP, host, port, verify_core);
  admin actions (Clear all SessionBank entries).

## Dopamine layer

- New all-time-max TPS triggers a centered `canvas-confetti` burst plus
  a Tremor toast in the top-right corner.
- Optional record-break chime via Web Audio (toggle in the top bar; off
  by default).
- TPS gauge uses `motion/react` springs so the number flips smoothly
  rather than jumping.
- Hot SessionBank slots pulse mint; cold slots fade cool.

## Themes

Four themes baked in: **hippo** (default, dark mint), **river** (cool
blue), **light** (bright for projector demos), **mono** (paranoid
contrast). Cycle with `t`. Persists in `localStorage["mtplx.dashboard.theme"]`.

## Keyboard shortcuts

| Key     | Action                                  |
| ------- | --------------------------------------- |
| `?`     | Show / hide the shortcuts overlay       |
| `t`     | Cycle theme (hippo → river → light → mono) |
| `g`     | Go to next tab                          |
| `space` | Pause / resume live updates             |
| `s`     | Toggle the new-max chime                |
| `Esc`   | Close overlays                          |

## Architecture (one paragraph)

The MTPLX server (FastAPI) gets three new in-process primitives on
`ServerState`: **MetricsBus** (asyncio pub/sub, drop-on-full to keep
the hot generation path uninterrupted), **InFlightRegistry** (request_id
→ handle with `cancel_event`/`started_s`/`prompt_preview`/`last_progress`
for external cancellation and the live in-flight panel), and
**RollingMetrics** (5-minute deque + per-session max + sticky all-time
max). Generation workers register/deregister handles around
`_run_generation`, publish `progress`/`completed`/`new_max_tps` events
to the bus, and append to RollingMetrics + the prefill history on
completion. The dashboard SPA (Vite + React 19 + TypeScript + Tailwind
v4 + shadcn primitives + Tremor for analytics tiles + uPlot for time
series + Recharts for richer charts + motion/react + canvas-confetti)
subscribes via `GET /v1/mtplx/metrics/stream` (SSE, 200 ms snapshot
cadence interleaved with bus events) and polls `/metrics`,
`/admin/sessions`, and `/v1/mtplx/prefill_history` for tabular state.
Same origin, same port, same process. Built bundle ships inside the
wheel via `[tool.setuptools.package-data]` so `pip install mtplx` is enough.

## New HTTP endpoints

| Endpoint                              | Method | Purpose                                                                                 |
| ------------------------------------- | ------ | --------------------------------------------------------------------------------------- |
| `/dashboard/`                         | GET    | SPA index (static mount; HTML fallback when the bundle is missing).                     |
| `/v1/mtplx/metrics/stream`            | GET    | Server-Sent Events: snapshots every 200 ms plus pushed bus events.                      |
| `/v1/mtplx/snapshot`                  | GET    | One-shot dashboard snapshot (same shape as the SSE snapshot event).                      |
| `/v1/mtplx/prefill_history`           | GET    | Bounded ring (cap 100) of recent prefill rows.                                          |
| `/v1/mtplx/settings`                  | POST   | Mutate the small whitelisted surface of `state.args`; rejects restart-required keys.    |
| `/v1/mtplx/cancel/{request_id}`       | POST   | Sets the in-flight handle's `cancel_event` (best-effort, one-token-batch worst case).   |

`/health` gains three fields: `chip` (`sysctl machdep.cpu.brand_string`),
`machine_model` (`sysctl hw.model`), and `unified_memory_bytes`
(`sysctl hw.memsize`), all cached after first lookup.

## What is *not* mutable from the dashboard

`profile`, `model`, `host`, `port`, `load_mtp`, `verify_core`,
`verify_strategy`, `context_window`, `api_key`.
These require a model/runtime reload. The Settings tab shows a
"restart required" card with the exact CLI command and a copy button
instead of pretending to hot-swap.

## Security notes

- `mtplx serve` defaults to `--host 127.0.0.1`. The dashboard inherits
  that bind.
- If you pass `--host 0.0.0.0`, MTPLX requires `--api-key` for chat
  completions; the dashboard control endpoints (`/v1/mtplx/settings`,
  `/v1/mtplx/cancel/{id}`) inherit that same middleware.
- CORS is wide-open (`allow_origins=["*"]`) for browser-driven
  third-party tooling. Treat this as the same trust boundary as
  `/v1/chat/completions`.

## Building from source

```bash
cd dashboard
bun install
bun run build      # outputs into ../mtplx/dashboard/_static/
```

The bundle is ~280 KB gzipped and ships in the wheel via
the `pyproject.toml` `[tool.setuptools.package-data]` table so end users do not need bun.
