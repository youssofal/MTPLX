# The Splash engine

MTPLX can serve its API and dashboard over [Inco's Splash][splash] instead of
its own MLX runtime:

```bash
mtplx serve --engine splash --model incoai/Qwen3.8-27B-Splash
```

Everything above the engine is unchanged. The same OpenAI Chat Completions,
OpenAI Responses and Anthropic Messages endpoints answer on the same port, the
native app and the web dashboard bind to the same contract, and OpenCode and
any other client keep working. What changes is which kernels run.

## Why you would

Splash is built the opposite way round from a general engine: it supports a
small set of models and rebuilds itself around each one — fused Metal kernels
compiled for that model's exact shapes, a DFlash 2 draft trained for it, and a
memory plan computed at startup. On Inco's M5 Pro measurements that buys about
2× the decode of the next-fastest engine on Qwen3.8-27B, and a 32K prefix
replays its first token in 282 ms.

The cost is reach. Splash loads only its own packages, and several things
MTPLX exposes for the MLX runtime do not exist inside it.

## What differs, concretely

| | MLX engine (default) | Splash engine |
| --- | --- | --- |
| Models | many architectures, any MLX checkpoint | Splash packages only |
| Speculation | native MTP, tunable depth | DFlash 2 draft, fixed |
| **KV cache** | **4-bit, 8-bit, or unquantized** | **8-bit, fixed** |
| Context | `--context-window` | engine-planned, up to 256K |
| Sampling | temperature, top_p, top_k (0 = off), presence penalty | temperature 0–2, top_p above 0, top_k 1–32, no penalties |
| Runtime settings | scheduler, batching, adaptive depth | sampling defaults and reasoning only |
| Prefill history | per-chunk timings | one row per request |

The KV row is the one that catches people out. Splash's Metal kernels read
8-bit KV directly rather than dequantizing, so the bit width is a property of
the compiled kernels, not a setting — there is no flag, no environment
variable, and no engine argument for it. The bridge reports this through the
same `kv_quant_policy` the app already reads:

```json
{ "supported": false, "modes": ["q8"],
  "disabled_reason": "Splash reads 8-bit KV directly in its precompiled Metal kernels..." }
```

so the app disables the KV control and explains why, instead of offering a
choice the engine cannot honor. **If you want 4-bit or unquantized KV, use the
MLX engine**, which supports all three through `--kv-quant`.

The bridge applies the same rule to everything else it cannot do: MTP depth
and per-depth acceptance stay absent, and `POST /admin/cache/clear` refuses
with a reason rather than pretending.

## The dashboard

The bridge drives the same `DashboardState` the MLX server does, so the Live
tab works unchanged: `progress` events move the decode gauge during a stream,
`completed` carries each request's envelope, `new_max_tps` fires the record
toast, and the min / max / mean / p95 row comes from the same
`RollingMetrics`.

A finished request's speeds are the engine's own. Splash's `/status` counters
are cumulative, so the bridge reads them either side of a request and reports
the difference: decode tokens over decode wall time, prefill tokens over
prefill wall time. (Requests that overlap share those counters, so each sees
the aggregate for the overlap.) Only the live gauge mid-stream is measured by
the bridge, from frame arrival, because Splash publishes counters rather than
a live rate. Memory is `memory_actual` — the process footprint, not the KV
pages alone.

Splash speculates on every decode, so draft acceptance is real and is
reported per request, labelled `draft: dflash2` to keep it distinct from
MTPLX's MTP. It varies a lot with content: on an M3 Max, a code prompt ran at
72 tok/s with 84% of drafts accepted, and a prose prompt at 36 tok/s with 34%.
On Splash 1.0.2 a short code prompt decoded at 96 tok/s (62% accepted) on the
same machine; a single sample, not a benchmark.

The app's chat reads the same stream fields from both engines. The bridge adds
what MLX sends and Splash does not: an `mtplx_progress` frame about every
200 ms for the live tok/s chip, and `usage` plus `mtplx_stats` on the finish
frame for each reply's footer (tok/s, out, in, cached, TTFT). A Stop sent with
Splash's `chatcmpl-…` id cancels the request it belongs to.

## The browser chat

`http://127.0.0.1:8000/` serves the same MTPLX chat page on both engines — one
function renders it (`mtplx/server/chat_page.py`), so the two cannot drift.
On Splash the page fills a few engine slots differently: the Speculative
section reads **DFlash 2 · on** with the package's draft length, both fixed;
the presence-penalty slider is fixed at 0; and the Top P and Top K sliders
span only what Splash accepts, so no setting on the page can fail a request.
Replies carry the same stats line (`DFlash 2 73% accepted · 105.6 tok/s ·
129 tokens · ttft 0.94s`).

The page, the app's parameter panel and the app's chat share one set of live
settings through `/v1/mtplx/settings`, as on MLX. The bridge fills them into
any request that leaves a field out — the app's chat sends none — and a write
outside Splash's range comes back moved to the nearest value it runs, with the
reason, so every panel shows what the engine will actually use. MTPLX's
`top_k: 0` (no filter) becomes 32, Splash's widest, not a greedy 1. "Hide
thinking" reaches Splash as `reasoning_effort: "none"`, the switch it reads.

The chat bar has a **Thinking** selector beside Send, on both engines, bound
to the same reasoning setting as the sidebar. Its choices come from the
loaded model's reasoning policy: Qwen 3.8 lists Auto, XHigh, Medium, Low and
Off, a model without effort levels Auto, On and Off, and a model without
thinking hides it. On Splash the levels are read from the package's chat
template, whose default (Auto) is XHigh; the app's parameter panel gets the
same effort picker from the same policy.

With an API key set, the MLX server's browser sign-in (`/mtplx/browser-auth`)
is not bridged yet, so open the page on a keyless local server.

## Requirements

Splash is a separate install:

```bash
brew install incoai/tap/splash
```

It needs an M3 or newer Mac on macOS 26.4 or later with at least 36 GB of
unified memory (48 GB recommended). MLX is *not* required: `--engine splash`
imports no MLX, so a machine that only ever runs Splash does not need it.

Packages download on first use, into the Hugging Face cache, and are verified
against their manifest by Splash's own installer:

| Package | Download |
| --- | ---: |
| `incoai/Qwen3.8-27B-Splash` | 17.4 GB |
| `incoai/Qwen3.6-35B-A3B-Splash` | 20.9 GB |

## Ports

Splash 1.0's CLI hardcodes port 8000 and takes an exclusive lock on it, which
would collide with the port MTPLX serves on. The bridge therefore starts
Splash's inner server on a private loopback port and keeps the public one for
itself. Override it with `--splash-port` if 0 is not acceptable; point
`--splash-prefix` at a non-Homebrew install.

## Loading and unloading

Splash has no runtime model-swap API — a server serves the package it was
started with — so loading a model is process lifecycle, and the bridge exposes
it as such:

```bash
curl localhost:8000/v1/mtplx/engine                      # phase, pid, packages, logs
curl -XPOST localhost:8000/v1/mtplx/engine/unload        # stop, free unified memory
curl -XPOST localhost:8000/v1/mtplx/engine/load \
  -d '{"model": "incoai/Qwen3.6-35B-A3B-Splash"}'        # swap package
```

`load` downloads the package first if it is missing. Unloading stops the
engine process, which is what actually returns its weights and KV cache to the
system.

## In the app

`Settings → engine` selects `mlx` or `splash`; the two keep separate model
choices (`model` and `splash_model`), so switching engines never overwrites
the other side's selection. Changing the engine restarts the daemon.

[splash]: https://inco.ai/blog/splash/
