# Runtime verification — Model Supervisor (2026-09-22)

Machine: M5 Max 128 GB, macOS 27.0, worktree `feat/model-supervisor` @ 0c61a59, real packs.

Command:

```
mtplx supervise --models Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed,Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed \
  --preload Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed --port 8790 --idle-ttl-s 120
```

| Check | Result |
|---|---|
| Boot, `/health` | ok, default = 9B, budget total 90.0 GiB (0.75 x RAM - 6 GiB), 9B `ready`, 27B `installed` |
| `/v1/models` | both packs listed, `loaded` flag correct |
| Chat to loaded 9B | 200, answered |
| Chat with unknown `model` | 200 via default, header `x-mtplx-routed-model` set |
| `/mtplx/admin/models` without key | 401 |
| Chat to 27B (not loaded) | JIT load, 200, answered; `/v1/models` then shows both `ready` |
| SSE stream through the proxy | frames arrive unbuffered, engine bytes intact |
| `kill -9` the 9B engine | health sweep saw it within 3 s ("crash: restart scheduled"), new pid `ready` at 6 s, next request answered; no orphaned server process, no stale listener on the old port |
| SIGTERM to the supervisor | both engines gone within 2 s, uvicorn shutdown clean, exit 143 |

Findings to fold into the fix backlog:

1. Engines answer with their own served id (`mtplx-qwen35-9b-optimized-speed`) while the
   supervisor routes by pack directory name. Register the engine's `/v1/models` id as an
   alias once READY so clients can send either, and list both in the front-door `/v1/models`.
2. Exit code on SIGTERM is 143 (uvicorn's default), not the 0 the design stated. Either
   document 143 or map a clean drain to 0.
3. The restart came back `ready` at 6 s while the weights were still loading (engine
   `/health` reports ok before warmup completes). Routing during warmup works but is slow;
   consider gating READY on the engine health's warmup field if one exists.

## Second pass after the review fixes (65e74d1)

| Check | Result |
|---|---|
| `/health` without key | redacted to model + state per engine; `fetch_daemon_health` fields intact |
| 27B JIT load | 200; `/v1/models` shows both packs `ready` with the engine-reported ids as `aliases` |
| Route by alias `mtplx-qwen38-27b-optimized-speed` | 200 |
| 34 MB body | 413 `request_too_large` |
| `kill -9` the 27B engine | `loading` at 3 s, `ready` at 9 s |
| SIGTERM | engines and listener gone, exit 143 (uvicorn's signal exit; documented) |
| Full test suite in the worktree | green, about 7090 tests, after installing the `server` extra (`llguidance`) that one pre-existing JSON-schema test needs |

Findings 1 and 3 from the first pass are fixed; finding 2 is documented rather than changed.
