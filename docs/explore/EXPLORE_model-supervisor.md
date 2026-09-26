# Explore: model supervisor (multi-model JIT routing in front of the single-model daemon)

Repo: MTPLX @ upstream main 7c2205a (v2.11.3). Line numbers are as of 2026-09-22; re-check before editing (`mtplx/server/openai.py` is ~37.7k lines).

## 1. How `mtplx serve` boots

| file:line | What it does |
| --- | --- |
| `mtplx/cli.py:3568-3865` | `serve` parser: every serve flag (`--model`, `--host`, `--port`, `--api-key`, `--api-key-file`, `--profile`, batching/MTP/thermal flags). `set_defaults(func=cmd_serve_public)`. |
| `mtplx/commands/public.py:9417` | `cmd_serve_public(args)`: resolves runtime options and fan mode, validates the API-key requirement for non-localhost binds (`_is_localhost_bind`), onboarding wizard, `args.model_id` via `_public_model_id_for_args`, per-model default profile, `_port_is_busy` short-circuit. |
| `mtplx/commands/public.py:10282` | `_run_server_child_with_app_parent_watchdog(cmd, env, cwd, app_parent_pid, poll_seconds, shutdown_grace_s)`: `mtplx serve` ALREADY runs the real server as a subprocess (`subprocess.Popen`) with a watchdog thread that kills the child on signal or when the macOS app parent disappears. The supervisor generalizes this to N engine children. |
| `mtplx/commands/public.py:10246` | `_terminate_server_child(proc, grace_s)`: SIGTERM, wait, SIGINT, wait, SIGKILL. Reuse for drains and restarts. |
| `mtplx/commands/public.py:10224` | `_pid_is_alive(pid)` via `os.kill(pid, 0)`. |
| `mtplx/server/openai.py:2901` | `class ServerState.__init__(args)`: one in-process server's state; single-model by construction (`self.model_id`, `self.runtime`). |
| `mtplx/server/openai.py:29443` | `create_app(state) -> FastAPI`: wires three ASGI middlewares (fan lease, auth+rate-limit, origin policy) and all routes. |
| `mtplx/server/openai.py:37703` | `main(argv)`: standalone entry (`python -m mtplx.server.openai`), builds `ServerState`, `create_app`, uvicorn. |
| `mtplx/server/openai.py:61` | Framework is FastAPI on Starlette; uvicorn serves. |

## 2. Model load/unload and memory accounting

| file:line | What it does |
| --- | --- |
| `mtplx/runtime.py:86`, `:566` | `class MTPLXRuntime`; `load(...)` is the single load path. There is NO unload/dispose for the chat runtime anywhere. |
| `mtplx/model_scheduler.py:113` | `ModelWorkScheduler`: single owner thread per process (foreground vs idle-persistence admission). Per-model subprocesses sidestep this. |
| `mtplx/retrieval.py:489`, `:784`, `:797`, `:833` | `RetrievalRegistry`: the only existing LRU + idle-unload implementation (embedding/reranker models). LRU eviction over `max_resident`, `unload_idle(older_than_s)` skipping pinned in-flight backends, `unload_all()`. Mirror this shape for chat engines. |
| `mtplx/memory_plan.py:388` | `plan_memory(...) -> MemoryPlan`: admission math (weights + KV + aux + prefill transient vs engine budget), `model_fits`, `context_window_fit`, human-readable `notes`. Call before spawning an engine. |
| `mtplx/memory_plan.py:273`, `:45-47` | `usable_engine_bytes(total_ram)` = `min(total, max(8 GiB, total*0.75), 192 GiB)`. No hardcoded 96 GiB constant in Python. |
| `mtplx/server/openai.py:2719-2726`, `:3316` | `_allow_swap_enabled(args)` reads `--allow-swap` / `MTPLX_ALLOW_SWAP=1`. |
| `mtplx/server/openai.py:18060-18104`, `:22767-22818` | `_shed_and_507`, `_refuse_or_admit`: request-time admission refusal (400/507) pattern; mirror the error shape for pack-level refusal. |
| `mtplx/model_catalog.py:40-56`, `:655`, `:666` | `CatalogModel(size_bytes, peak_memory_gib)`, `FeasibilityVerdict`, `evaluate_feasibility(...)`: existing per-machine "will this fit" verdict. |
| `mtplx/model_catalog.py:688-781` | `InstalledModel`, `scan_installed_models(cache_dir)`, `installed_catalog_ids`: what is on disk under `~/.mtplx/models`. |
| `mtplx/default_models.py:521` | reads `mtplx_runtime.json` from a pack. `phys_footprint` appears nowhere in Python. Whether the pack JSON carries a memory figure must be verified against a sample pack (catalog-side `peak_memory_gib` is confirmed, pack-side is not). |

## 3. Existing daemon control surface

| file:line | What it does |
| --- | --- |
| `mtplx/commands/public.py:2839` | `cmd_stop_public`: stops via the health-reported pid. No pid or port files exist anywhere; discovery is by probing a fixed port list and `/health`. |
| `mtplx/daemon_client.py:29`, `:44-67`, `:82` | `DAEMON_PROBE_PORTS = (8000, 18083, 18084, 18085)`; `RunningDaemon(host, port, model, model_path, launch_id, pid, api_key_required, health)`, `owned_by_app` keys off `launch_id`; `fetch_daemon_health` requires `ok: true`. |
| `mtplx/commands/public.py:3034` | `cmd_settings_public`: `GET/POST /v1/mtplx/settings` on one host:port. |
| `mtplx/server/openai.py:29990-29996`, `:29674`, `:30908` | settings endpoints (also `/mtplx/settings`); `/health` (includes `model`, `startup.pid`, `startup.launch_id`); `/v1/models` lists exactly one chat entry plus retrieval entries. |
| `mtplx/server/openai.py:24103-24141`, `:4768` | `_AuthRateLimitMiddleware` (pure ASGI) using `_request_is_authorized`: the pattern to reuse for the management API's key gate. |
| env | `MTPLX_ALLOW_SWAP`, `MTPLX_STREAM_STALL_DEADLINE_S`, `MTPLX_MEMORY_LIMIT_BYTES`, `MTPLX_MEMORY_BUDGET`, `MTPLX_CORS_ORIGINS`. No `MTPLX_PORT`/`MTPLX_HOST`; host/port are argv only (defaults 127.0.0.1:8000). |

## 4. macOS app's supervisor (Swift)

| file:line | What it does |
| --- | --- |
| `apps/MTPLXApp/Sources/MTPLXAppCore/Services/DaemonSupervisor.swift:58-76` | `DaemonRestartPolicy(maximumAttempts=3, initialDelaySeconds=1, maximumDelaySeconds=30, crashWindowSeconds=120)`: the restart shape to match. |
| `DaemonSupervisor.swift:131`, `:254`, `:328` | `DaemonSupervisor` with a `restartGeneration` counter to cancel stale restarts; note at 254: an OOM crash must not spin the backoff loop; `processIsAlive(pid)`. |
| `DaemonLivenessPolicy.swift:1-70` | Issue #487 fix: process alive + port accepting is never reaped on probe timeouts alone; only gone process, closed port, or long unbroken silence reaps. The Python watchdog must replicate this busy-vs-dead distinction. |
| `MTPLXCommandBuilder.swift:183-370` | `buildServeCommand`: exact argv the app launches (`serve --host --port --model` plus profile, generation-mode, MTP, batching, retrieval, ssd-session-cache, context-window, `--api-key-file`, thermal, `--app-launch-id`, `--yes`, sampling). Line 320: the key never rides on argv (`ps` visibility). Per-engine argv must stay a superset of this; never reinvent. |

## 5. Request routing today

| file:line | What it does |
| --- | --- |
| `mtplx/server/openai.py:31091`, `:31113`, `:31139`, `:31676`, `:31813` | `chat_completions`: `request.model` is only checked against the retrieval registry (400 for an embedder id); otherwise `model = state.model_id`. The request is always served by whatever is loaded; a mismatch is recorded in observability (`request_model_matches_served_model`), never rejected. This is the gap the supervisor closes. |
| `:35701`, `:35764`, `:35796`, `:35834` | `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/completions`. |
| `:15270, 30774, 30829, 35353, 35736, 35776, 36279` | `StreamingResponse(media_type="text/event-stream")`: byte-stream passthrough is enough; never buffer whole bodies. |
| `:29509-29517`, `:24103-24107` | Middleware order note; auth/origin are pure ASGI because `BaseHTTPMiddleware` taxes every SSE frame. Any supervisor middleware must be pure ASGI. |

## 6. Config / catalog / pack metadata

| file:line | What it does |
| --- | --- |
| `mtplx/config.py:18`, `:65`, `:113`, `:141`, `:222` | `~/.mtplx/config.toml`, `UserConfig`, `load_user_config`, `apply_user_config(args)`. |
| `mtplx/profiles.py` | runtime profiles (`PROFILE_CHOICES`, `get_profile`, `resolve_profile_name`). |
| `mtplx/default_models.py:13`, `:132-178`, `:521` | default model dir and fallback search; `DefaultModelSelection`; reads `mtplx_runtime.json`. |
| `mtplx/model_catalog.py:496-509`, `:531-645`, `:655-666`, `:688-781` | id resolution, hardware-tier recommendations, feasibility verdict, installed scan (`mtplx models` at `cli.py:2883`, `list` at `:3458`). |

## 7. Tests

| file | Coverage / pattern |
| --- | --- |
| `tests/test_public_cli.py` | `test_serve_wrapper_signal_stops_child_daemon` (165) exercises `_run_server_child_with_app_parent_watchdog` and `_pid_is_alive` with a real wrapper+child (SIGTERM contract: `wrapper.returncode == 128 + SIGTERM`). `test_serve_forwards_retrieval_flags_to_the_server_command` (1677) pins the public-CLI to server-subprocess argv forwarding; `public.py:9866` "rebuilt argv, so anything not forwarded here never reaches it". Also `test_serve_no_auth_*` (1715, 1748), `test_serve_refuses_to_start_when_a_retrieval_model_is_missing` (1778), `test_serve_require_max_fans_fails_closed_before_child_launch` (2510). |
| `tests/test_daemon_client.py` | `fetch_daemon_health`, `probe_running_daemons`, `stop_daemon`, `DAEMON_PROBE_PORTS`; single-daemon-per-port assumptions. |
| `tests/test_server_openai.py` | `_fake_state(...)` (~1975-2027) and `_fake_streaming_session_state()` (2029); `TestClient(create_app(state))` pattern (2058). Keep `create_app(state)` importable per engine. |
| `tests/test_dashboard_endpoints.py` | dashboard mount (`_mount_dashboard`, `openai.py:36526`); check route-prefix changes. |

## 8. Existing design notes

`docs/superpowers/specs/2026-07-29-retrieval-idle-standby-design.md` is the closest precedent and explicitly scopes chat-model unloading OUT ("touches ServerState, warmup, the MTP contract and session prefixes... deserves its own spec"). No other multi-model, supervisor, router or JIT notes exist.

## Blast radius

- New: a process supervisor package (owns N engine children generalizing `public.py:10282/10246/10224`; proxies HTTP/SSE by `model`; admission via `model_catalog.evaluate_feasibility` / `memory_plan.plan_memory`).
- `mtplx/cli.py:3568` and `public.py:9417`: a `supervise` entry (or `serve --supervisor`).
- `mtplx/server/openai.py:30908` (`/v1/models`): the supervisor synthesizes a merged list; no handler change needed if routing lives in the supervisor.
- `mtplx/daemon_client.py`: `stop`/`status`/`settings` need to reach the supervisor's port or gain a `--model` selector.
- Swift app: lowest risk is the app supervising the Python supervisor via the same `serve` argv plus a flag, keeping `/health` semantics (`launch_id`, `pid`, `model`) at the front door.
- Tests: `test_public_cli.py` argv pins, `test_daemon_client.py`, `test_server_openai.py` fixtures, `test_dashboard_endpoints.py`.

## Risks

1. Streaming passthrough: no body buffering; pure ASGI only.
2. Session bank on SSD: archive/flush before unload or conversations lose their cache prefix.
3. In-flight draining: pin in-flight requests like `RetrievalRegistry`; never kill mid-stream (issue #487's lesson).
4. OOM restarts must not spin: distinguish memory-caused death from transient crash.
5. The Swift app discovers the daemon by `/health` + pid + port only; keep that contract at the front door.
6. API key never on argv; use `--api-key-file`.
7. Argv forwarding is pinned by tests; new flags must be added to the forwarding list.
8. Making `model` a real routing key is a user-visible behavior change; decide hard-fail vs fallback-with-warning explicitly.
