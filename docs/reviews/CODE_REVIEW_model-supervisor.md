# Code Review — Model Supervisor (commits 43cb65c, 0c61a59)

Scope: `mtplx/supervisor/{registry,budget,process,proxy,admin,service}.py`, the
`supervise` CLI parser in `mtplx/cli.py`, `cmd_supervise_public` in
`mtplx/commands/public.py`, `tests/test_supervisor_*.py`. Read-only review
against `docs/features/model-supervisor/{DESIGN,FEATURE_CONTEXT}.md`.

## CRITICAL

**C1. No lock coordination between `_unload`/`_reap` and `ensure_loaded` for the
same model id — draining/dead engines can be JIT-reloaded underneath the
unload, leaking the old process and corrupting registry state.**
`mtplx/supervisor/service.py:277-292` (`_unload`) and `:219-272` (`ensure_loaded`).
`registry.resolve()` (`mtplx/supervisor/registry.py:87-112`) explicitly treats
a `DRAINING` record as `"installed"`, so a request arriving while an engine is
being idle-unloaded, evicted, or admin-unloaded takes the `ensure_loaded` path
again. `ensure_loaded` only checks `state == READY` before acquiring
`self._load_locks[model_id]`; `_unload` never acquires that lock. Interleaving:
`_unload` sets `DRAINING` and awaits its pin-drain loop (`asyncio.sleep(0.05)`,
a real yield point) → a concurrent request resolves `"installed"`, enters
`ensure_loaded`, takes the (uncontended) load lock, admits, spawns a **new**
`EngineProcess` on a new port, and overwrites `self._processes[model_id]` and
the registry state (`LOADING`, then `READY`) → `_unload`'s wait loop finally
exits and does `proc = self._processes.get(model_id)` (fetched *after* the
wait, `service.py:289`), which is now the **new** process, not the one it
meant to drain. It terminates the new engine and then unconditionally sets
`STOPPED` (`service.py:292`), clobbering whatever state the concurrent load
just reached. The original engine is never terminated — a live, untracked
child holding its port and RAM. This hits every category the review asked
about: JIT-lock/idle-sweep race, zombie child on the resource-hygiene axis,
and a state-machine corruption bug no test currently exercises.
Fix: give `_unload`/`_reap`/`restart` the same per-model `asyncio.Lock` that
`ensure_loaded` uses (acquire it for the whole drain+terminate sequence), and
capture `proc = self._processes.get(model_id)` **before** the drain wait, not
after.

**C2. Mid-stream engine death sends a second `http.response.start`, breaking
the ASGI connection instead of the clean 503 DESIGN.md promises.**
`mtplx/supervisor/proxy.py:244-264` (`_forward`). The `try` wraps the entire
`async with client.stream(...)` block, including the `async for chunk in
response.aiter_raw()` loop that runs *after* `await send({"type":
"http.response.start", ...})` has already gone out. If the engine dies or the
connection drops mid-stream, `aiter_raw()` raises `httpx.ReadError` (or
similar), and the `except` handler calls `_send_json(...)`, which builds a
`JSONResponse` and sends its own `http.response.start` — a second start
message on a scope that has already started its response. ASGI servers
(uvicorn) reject this (`AssertionError: Unexpected ASGI message
'http.response.start'...` or a torn connection), so the client sees a broken
connection or a 500, not the clean error implied by DESIGN.md's success
criterion 2 ("Killing an engine process mid-stream: in-flight request gets a
clean error"). The except tuple is also incomplete: `httpx.RemoteProtocolError`,
`httpx.PoolTimeout`, `httpx.WriteError`, and `anyio` cancellation from a client
disconnect are not caught at all and propagate as unhandled exceptions.
Fix: track whether `http.response.start` has been sent; on a stream failure
after that point, stop iterating and send a terminating empty body chunk (or
close the connection) rather than attempting a second JSON response. Only
attempt `_send_json` in the pre-start branch. Widen the caught exception set
or catch `httpx.HTTPError` plus a generic disconnect guard.
`tests/test_supervisor_proxy.py::test_engine_connection_error_maps_to_503`
only kills the engine before any request is sent (pre-`stream()`-open), so it
never exercises this path — see Tests section.

## HIGH

**H1. Request body is fully buffered in memory with no size limit before
routing.** `mtplx/supervisor/proxy.py:141` (`body = await request.body()`)
reads the entire request body for every request, including large
multipart/audio/image payloads, before the model is even resolved. There is
no `Content-Length` cap and no streaming-to-disk fallback, so a large or
malicious upload is held entirely in the supervisor's memory (on top of the
per-engine budget the feature is designed to protect). This also
contradicts the proxy module docstring's claim that it forwards "without
ever being buffered in full" — that is true only for the response side.
Fix: cap accepted body size (413 over cap) before buffering, or peek only
the first N KiB for the `model` field, falling back to a duplex passthrough
for arbitrarily large bodies (matches how `mtplx/server/openai.py` already
handles large request bodies, if it does — worth checking for parity).

**H2. No child-process reaping on supervisor crash or `SIGKILL`.**
`mtplx/supervisor/process.py:154` spawns children with plain
`subprocess.Popen`; nothing sets a process group or a `PR_SET_PDEATHSIG`-style
guard. `Supervisor.stop()` (`service.py:188`) only runs on a clean ASGI
lifespan shutdown; if the supervisor is `SIGKILL`ed or OOM-killed, every live
engine child is orphaned — still bound to its port and its share of the
memory budget, invisible to the next supervisor start. This is the
zombie-children risk FEATURE_CONTEXT's R5 calls out, and DESIGN.md's shutdown
section (behavior 8) doesn't mention it as an accepted gap.
Fix: put each child in its own process group and add a signal handler on the
supervisor that walks `_processes` and terminates them before exit (uvicorn's
default handlers currently bypass `Supervisor.stop()` entirely).

**H3. `restart()` does not go through the load lock, and its `RestartTracker`
generation is not bumped by `reset()`.** `mtplx/supervisor/service.py:294-305`.
`Supervisor.restart()` terminates the current process and flips state to
`INSTALLED` without holding `self._load_locks[model_id]`, so a client request
racing the admin restart can observe `READY` right up until the terminate
call, get pinned, and then have its engine killed out from under it (falls
into the C2 mid-stream bug). Separately, `RestartTracker.reset()`
(`process.py:96-97`) clears `_crash_times` but leaves `generation` unchanged;
a scheduled `_restart_after_delay` task from a crash that predates the admin
restart is currently saved only by the `record.state != INSTALLED` check in
`service.py:362`, which is incidental protection, not a designed invariant —
a future refactor that changes what `restart()` sets state to before calling
`ensure_loaded` would silently reintroduce a duplicate-restart race.
Fix: acquire the per-model load lock for the whole `restart()` body, and have
`reset()` increment `generation` too so staleness is enforced by the counter
itself, not by an incidental state check.

## MEDIUM

**M1. `estimate_resident_bytes` globs only the pack root, not subdirectories.**
`mtplx/supervisor/budget.py:100` uses `path.glob("*.safetensors")`
(non-recursive). If any installed pack shards its weights under a
subdirectory, the fallback estimate silently undercounts, which feeds
directly into `admit()`'s admission math — the one thing R4 requires to be
conservative. Worth confirming against how packs are actually laid out on
disk (check `mtplx/model_catalog.py` / the installer) and switching to
`path.rglob(...)` if any pack can nest weights.

**M2. `_free_port()` has a classic bind-then-close TOCTOU.**
`mtplx/supervisor/service.py:63-66`. The port is chosen by binding to `:0`,
reading it back, and closing the socket before the real child ever binds it.
Under concurrent JIT loads (two different models loading at once, or a
restart racing a fresh load — see H3) two `_free_port()` calls can return the
same port before either child has bound it, and the second `mtplx serve`
child will fail to start with an address-in-use error that surfaces only as
an opaque `load_timeout`/`GONE` from the caller's point of view, not a
diagnosable "port collision." Low probability in practice (OS ephemeral
allocation is usually sequential) but worth a comment acknowledging the
tradeoff, or holding the probe socket open until immediately before
`Popen()`.

**M3. No read timeout on the forwarding client.**
`mtplx/supervisor/service.py:173` sets `read=None` for the proxy's `httpx`
client. Intentional for long SSE streams, but combined with the busy-vs-dead
policy (a wedged completion where `/health` still answers `ok: true`) a
client can hang indefinitely with no ceiling at the proxy layer. Confirm this
is the intended contract and say so explicitly in DESIGN.md's liveness
section.

**M4. Style: em-dashes in new code.** FEATURE_CONTEXT.md's constraints
section says "hyphens not em-dashes." Two lines in the new
`cmd_supervise_public` block use em-dashes: `mtplx/commands/public.py:16225`
(`# \`mtplx supervise\` — multi-model JIT-loading front door.`) and
`mtplx/commands/public.py:16308` (`# public \`serve\` argv (DESIGN.md key
behavior 7) — never the lower-level`), plus the `--host` help string shared
across `serve`/`supervise` parsers in `mtplx/cli.py:3921` (pre-existing
pattern, but the `supervise` parser at `cli.py:3920` copies it verbatim
rather than fixing it). The `mtplx/supervisor/*.py` module docstrings
themselves are clean (no em-dashes found).

## LOW

**L1. `EngineRegistry` has no `remove()`.** A `STOPPED`/`FAILED` engine stays
in `records()` forever (no "forgotten" terminal state), so a long-running
supervisor with many load/unload cycles accumulates dead records in
`snapshot()`/`/mtplx/admin/models`. Probably fine given a small fixed
install-set, but worth a one-line DESIGN.md note that this is intentional.

**L2. `admit()`'s `reason` string formatting duplicates itself** at
`budget.py:159-168` and `178-186`. Factor into one helper; not urgent.

**L3. `RestartPolicy` docstring says it "mirrors the Swift app's
`DaemonRestartPolicy` defaults"** but cites no line numbers, unlike the
module header's citation of `DaemonLivenessPolicy.swift`. Add the same kind
of citation so the "3 attempts / 1-30s / 120s window" defaults are
independently checkable.

## DESIGN.md conformance (behaviors 1-8)

| # | Behavior | Status |
|---|---|---|
| 1 | Routing (loaded/installed/fallback/strict 404) | Matches |
| 2 | Budget + eviction, 507 shape | Matches (fields: needed_bytes, available_bytes, would_free, reason) |
| 3 | Idle unload, default exempt unless `--unload-default` | Matches |
| 4 | Liveness sweep (2s), busy-vs-dead | Matches |
| 5 | Restart backoff, OOM no-restart, "pins get 503 once the socket closes" | **Diverges** — see C1/C2: the 503 promise only holds pre-stream-start, and unload/restart aren't mutually exclusive with a concurrent JIT load |
| 6 | Auth: admin always keyed, `--insecure-lan` inference only | Matches |
| 7 | Engine argv shape | Matches (`process.py:128-145`) |
| 8 | Shutdown: drain then terminate | Matches for the clean-shutdown path; **diverges** for SIGKILL/crash (H2) |

## Resource hygiene

- Zombie children on supervisor crash: not handled (H2).
- Stderr reader thread: bounded (`deque(maxlen=200)`, `process.py:31,122`),
  daemonized, and exits cleanly when the pipe closes — good.
- `httpx.AsyncClient` lifecycle: opened in `start()`, closed in `stop()` —
  correct, but only reached on a clean lifespan shutdown (see H2).
- fds: stdout is `DEVNULL` (never leaks a pipe buffer), stderr is drained
  continuously — good.

## Repo conventions

- Docstrings read as plain language and match the register of
  `mtplx/retrieval.py`/`mtplx/daemon_client.py` (explain the "why", cite the
  Swift files being mirrored). Good.
- Error JSON shape `{"error": {"code": ..., "message": ...}}` matches the
  server's existing convention (`mtplx/server/openai.py` builds
  `{"error": error}` too) — consistent.
- Logging uses the stdlib `logging` module under a shared `"mtplx.supervisor"`
  logger name, consistent with the rest of the codebase's `logging.getLogger`
  usage.
- Em-dashes: see M4 — two spots in the new `public.py` section, one shared
  help string.

## Tests: missing coverage

Concrete gaps, as test names that don't exist yet:

1. `test_supervisor_proxy.py::test_engine_dies_mid_stream_after_headers_sent`
   — kill the engine *while* `aiter_raw()` is mid-loop (after the first SSE
   chunk); assert a clean close, not a second `http.response.start` (C2).
2. `test_supervisor_service.py::test_concurrent_unload_and_jit_load_same_model`
   (no `test_supervisor_service.py` exists — `ensure_loaded`/`_unload`/`_reap`
   have no direct unit test outside admin/proxy integration tests) — start a
   drain and fire a concurrent request for the same model id; assert exactly
   one live process survives (C1).
3. `test_supervisor_admin.py::test_restart_races_inflight_request` — pin an
   engine, call admin `/restart` concurrently, assert a clean 503 or
   completion against the old engine, never a crash (H3).
4. `test_supervisor_proxy.py::test_request_body_size_limit_rejected` — no
   test exercises a large body; add one for the size cap, or document there
   is none (H1).
5. `test_supervisor_process.py::test_child_survives_parent_sigkill` — nothing
   simulates the supervisor dying uncleanly (H2).
6. `test_supervisor_budget.py::test_estimate_resident_bytes_recurses_shards`
   — no test covers safetensors nested under a subdirectory (M1).
7. `test_supervisor_registry.py` fully covers `resolve`/`pin`/`lru`/`idle` in
   isolation but nothing exercises `DRAINING` → `resolve()` returning
   `"installed"` end-to-end, the seam C1 lives in.
8. No CLI test for `--models all` with zero installed packs, nor for
   `--preload` naming a model that fails to admit at startup beyond the
   existing log-and-continue path (`service.py:181-184`).

## FIX_BACKLOG

| id | severity | file | one-line fix |
|---|---|---|---|
| C1 | CRITICAL | mtplx/supervisor/service.py:277-292 | Have `_unload`/`_reap`/`restart` acquire the same per-model load lock as `ensure_loaded`, and capture `proc` before the drain-wait loop, not after |
| C2 | CRITICAL | mtplx/supervisor/proxy.py:244-264 | Stop sending a second `http.response.start` after streaming has begun; close the connection cleanly on a mid-stream error instead |
| H1 | HIGH | mtplx/supervisor/proxy.py:141 | Cap request body size before buffering (413 over cap) or peek-only the `model` field |
| H2 | HIGH | mtplx/supervisor/process.py:154, service.py | Put children in their own process group and add a supervisor signal handler that terminates all children before exiting on an unclean shutdown |
| H3 | HIGH | mtplx/supervisor/service.py:294-305, process.py:96-97 | Take the load lock in `restart()`; bump `RestartTracker.generation` in `reset()` |
| M1 | MEDIUM | mtplx/supervisor/budget.py:100 | Use `path.rglob("*.safetensors")` if packs can shard into subdirectories |
| M2 | MEDIUM | mtplx/supervisor/service.py:63-66 | Hold the probe socket open until just before `Popen()`, or document the TOCTOU as accepted |
| M3 | MEDIUM | mtplx/supervisor/service.py:173 | Document the no-read-timeout contract for inference forwarding in DESIGN.md |
| M4 | MEDIUM | mtplx/commands/public.py:16225,16308; mtplx/cli.py:3920 | Replace em-dashes with hyphens per FEATURE_CONTEXT.md's style constraint |
| L1 | LOW | mtplx/supervisor/registry.py | Note in DESIGN.md that terminal records are never garbage-collected |
| L2 | LOW | mtplx/supervisor/budget.py:159-186 | Factor the repeated reason-string formatting into one helper |
| L3 | LOW | mtplx/supervisor/process.py:58-67 | Cite the specific Swift `DaemonRestartPolicy` constants being mirrored |
