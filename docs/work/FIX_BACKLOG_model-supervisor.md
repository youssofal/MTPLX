# Fix backlog — Model Supervisor (consolidated 2026-09-22)

Sources: docs/reviews/CODE_REVIEW_model-supervisor.md, SECURITY_REVIEW_model-supervisor.md, RUNTIME_model-supervisor_2026-09-22.md.

## Agent F1 scope: service.py, process.py, registry.py, budget.py + their tests
| id | sev | fix |
|---|---|---|
| C1 | CRITICAL | unload/drain vs JIT-load race: take the per-model load lock in `_unload`, `restart` and the idle/health reaps; `resolve()` must treat DRAINING as not-loadable (return ("draining", 503 Retry-After) rather than "installed"); `_unload` must capture the process object before waiting and only terminate that object |
| H2 | HIGH | `EngineProcess.spawn` must pass a scrubbed env: copy os.environ minus any key whose name contains API_KEY, TOKEN, SECRET (case-insensitive), plus explicit `env` overrides |
| H3 | HIGH | orphan protection: spawn engines in their own process group (`start_new_session=True`) and record pgid; on supervisor exit paths (stop(), atexit, SIGTERM) terminate by process group; also pass the supervisor pid to children via env `MTPLX_SUPERVISOR_PID` and have `EngineProcess` document that engines already self-exit when the parent watchdog pid vanishes if such a flag exists on `mtplx serve` (check `--app-parent-pid` or similar in public.py; use it if present) |
| H4 | HIGH | `restart()` takes the load lock; `RestartTracker.reset()` bumps `generation` |
| M3 | MEDIUM | `estimate_resident_bytes`: recursive `rglob("*.safetensors")`, still excluding `ngram-table.safetensors` |
| M4 | MEDIUM | `_free_port` TOCTOU: retry spawn on "address in use" up to 3 times with a fresh port |
| R1 | RUNTIME | aliases: `EngineRegistry.add_alias(model_id, alias)`; `resolve()` matches aliases; service registers the engine's own reported id (GET /v1/models of the engine, first entry) as an alias when the engine becomes READY |
| R3 | RUNTIME | READY gating: if the engine `/health` payload has a warmup/ready field that is false while loading (inspect `mtplx/server/openai.py` around line 16925 for `warmup`), keep LOADING until it is true |
| L1 | LOW | GC terminal records? No: keep STOPPED/FAILED records (they are the installed list). Instead ensure `set_state(READY)` clears `failure_reason`. |
| L2 | LOW | health sweep probes engines concurrently (`asyncio.gather` over `to_thread` probes) |

## Agent F2 scope: proxy.py, admin.py, cli.py (supervise parser only), public.py (supervise block only), docs/server.md + tests for proxy/admin/cli
| id | sev | fix |
|---|---|---|
| C2 | CRITICAL | mid-stream engine death after `http.response.start`: do not send a second start; close the body (send `more_body: False`) and log; add test with the fake engine `--exit-after-s` mid-stream |
| H1 | HIGH | body cap BEFORE buffering and only after auth: read at most `max_body_bytes` (config, default 32 MiB) and reply 413 `{"error":{"code":"request_too_large"}}`; run the auth check before reading the body |
| M1 | MEDIUM | rate limit: a small token bucket per client host for admin routes (e.g. 30/min) and for failed auth on any route (10/min then 429); keep it simple and in-process |
| M2 | MEDIUM | `/health` without a valid key returns only `ok`, `model`, `startup`, and `supervisor: {engines: [{model, state}]}`; pids, ports, pins, failure_reason and budget only with the key |
| L1s | LOW | path dispatch: lowercase-compare and strip duplicate slashes before matching `/health`, `/v1/models`, `/mtplx/admin/*` |
| S1 | STYLE | replace every em-dash in the new cli/public code and docs/server.md section with a hyphen or a period |
| R2 | RUNTIME | document exit code 143 on SIGTERM in docs/server.md; the CLI maps a clean shutdown (`run_supervisor` returning after lifespan shutdown completes) to 0 if uvicorn exposes that; otherwise document |
| R1b | RUNTIME | front-door `/v1/models` lists aliases: each entry gains `"aliases": [...]` (read from registry; F1 adds `EngineRecord.aliases: list[str]`) |
| T | TESTS | add: mid-stream death, body cap 413, admin rate limit 429, redacted /health, path normalization, alias routing (once F1 lands `add_alias`; if not yet present, write the test against the frozen name and mark xfail with reason) |
