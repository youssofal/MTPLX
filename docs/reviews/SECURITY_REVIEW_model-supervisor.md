# Security Review — Model Supervisor

Scope: `mtplx/supervisor/{proxy,admin,service,process}.py`, `cmd_supervise_public` in
`mtplx/commands/public.py`, the `supervise` parser in `mtplx/cli.py`. Branch
`feat/model-supervisor` in `/Users/bmatthews/Code/MTPLX-supervisor`. Read-only review;
no files modified except this one.

Overall: no full auth bypass into the admin API or an engine was found. The proxy's
`is_authorized`/`_admin_authorized` gates mirror the existing server's
`_request_is_authorized` correctly, including `secrets.compare_digest`. The two HIGH
findings are both new attack surface introduced by this feature relative to the
existing single-engine server, not auth-bypass bugs in the gate logic itself.

## Findings

### H1 — Full request body buffered before the auth check (memory-exhaustion DoS)
`mtplx/supervisor/proxy.py:137-152` — `ProxyApp.__call__` does
`body = await request.body()` (line 141) *before* `self._is_authorized(request)` is
checked (line 143), and `Request.body()` has no size cap. Any client that can reach the
listening port — authenticated or not — can send an arbitrarily large or slow-trickled
body and have it fully buffered in memory ahead of any credential check. The existing
engine server's equivalent gate, `_AuthRateLimitMiddleware` (`mtplx/server/openai.py`
~24095), was deliberately built to "touch only the request head... downstream receives
an untouched stream" for exactly this reason; the supervisor's proxy does not follow
that pattern. Exploit sketch: an unauthenticated (or rate-unlimited authenticated)
client opens N connections and streams multi-GB/never-ending bodies to `/v1/...`,
exhausting supervisor RAM well before request handling or auth denial occurs.
Fix: check auth from the ASGI head first (Starlette can construct headers/query without
consuming the body, as `_AuthRateLimitMiddleware` does), then read the body with an
explicit byte ceiling (reject with 413 past e.g. `MTPLX_MAX_BODY_BYTES`), consistent
with the JIT-load-only need to peek `model`.

### H2 — Engine children inherit the supervisor's full environment, including the API key when set via `MTPLX_API_KEY`
`mtplx/supervisor/process.py:103-121` (`EngineProcess.__init__`, `env: dict|None = None`)
and `mtplx/supervisor/service.py:248-252` (`EngineProcess(spec.path, port,
self.config.engine_extra_args)` — no `env=` override ever passed). `subprocess.Popen(env=None)`
inherits the parent's *entire* environment. `resolve_api_key` (`mtplx/runtime_options.py:415-435`)
treats `MTPLX_API_KEY` as a valid key source, and `cmd_supervise_public` calls it via
`_resolve_runtime_options_on_args`. DESIGN.md behavior 7 correctly states "the supervisor
never puts a key on any argv" — true for argv, but the key still rides along in `os.environ`
into every `--no-auth` engine child when the operator sourced it from the environment
rather than `--api-key-file`. Any other local user (or a process that later compromises
one engine child) can read it from that child's `/proc/<pid>/environ`. Fix: build an
explicit `env` for `EngineProcess` that is the parent environment minus `MTPLX_API_KEY`
(and any other secret-shaped var), and pass it through from `Supervisor.ensure_loaded`.

### M1 — No rate limiting on any supervisor route (admin or proxy)
`mtplx/supervisor/admin.py` and `proxy.py` have no equivalent of the engine server's
`_AuthRateLimitMiddleware` rate limiter (`mtplx/server/openai.py:24095-24150`, which
gates *every* route, keyed and unkeyed, after the auth check). As shipped, an attacker
who can reach the port gets unlimited attempts at guessing the API key (the compare is
constant-time per attempt, but nothing slows repeated attempts), and an authenticated-or-
localhost caller can hammer `/mtplx/admin/restart`, `/unload`, or trigger repeated JIT
loads for many distinct model ids, thrashing engine processes / the budget admission
path with no backpressure. Fix: reuse the existing rate limiter keyed the same way
(`_rate_limit_key`) in front of both `AdminApp` and `ProxyApp`.

### M2 — `/health` discloses full engine topology (pid, port, pins, failure_reason) with no admin credential
`mtplx/supervisor/admin.py:90-103`, gated by `_inference_authorized` (line 80-86), which
is a no-op whenever `config.insecure_lan` or the bind is localhost — i.e. exactly the
cases DESIGN.md's "Auth" behavior 6 says should stay open for convenience. The payload
includes `supervisor.snapshot()` (`service.py:386-412`): every engine's `port`, `pid`,
`pins`, `last_used`, `failure_reason`. Under `--insecure-lan`, any host on the LAN gets
this for free with zero credential — including the ephemeral port of every `--no-auth`
engine, which is otherwise the only thing standing between "on this box" and "can send
inference/health requests directly to an engine." On a localhost bind it's handed to any
co-resident process too, which slightly widens what a co-resident attacker needs to
enumerate manually. This is a real increase in reconnaissance surface, though it does
not by itself cross a boundary the design didn't already accept (see Accepted Risks).
Fix: keep `ok`/`model`/`startup` unauthenticated (matches today's daemon health contract)
but require the admin key for the `supervisor` block, or omit `pid`/`port` from the
unauthenticated shape and only return them from `/mtplx/admin/health`.

### L1 — Route dispatch is prefix/exact-match, not normalized; malformed paths fall through to the proxy instead of 404
`mtplx/supervisor/service.py:139` (`path.startswith("/mtplx/admin")`, no trailing-slash
boundary) and `admin.py:50-65` (exact string `==` against fixed paths, no case-fold, no
dot-segment handling). Verified fail-closed: `/mtplx/admin/../v1/x` still starts with
`/mtplx/admin`, routes to `AdminApp`, matches no fixed route, returns 404 — it does *not*
reach the proxy or an engine, so this is not an auth bypass. But a case-differing or
malformed variant of `/health` (e.g. `/HEALTH`) does *not* match either branch of the
`service.py:139` dispatch and silently falls through to `ProxyApp`, which still enforces
its own auth gate — so no credential is skipped — but the request is now forwarded
verbatim to the *default engine's* own path instead of the supervisor-synthesized
`/health`, a behavior change from what the design describes. Fix: normalize path
(lower-case, collapse `//`, resolve `.`/`..`) once at the top of `Supervisor.__call__`
before the admin/proxy split, and add a regression test for it.

### L2 — Health sweep probes engines sequentially, not concurrently
`mtplx/supervisor/service.py:317-334` (`_health_sweep_once`) awaits `proc.probe` (each up
to 1.5s, via `to_thread`) one engine at a time in a `for` loop. This does not block the
asyncio event loop (each probe runs on its own thread), so other requests keep being
served, but detection latency for a wedged/dead engine grows linearly with the number of
supervised engines — worth a `asyncio.gather` if large `--models` counts become common.
Informational; not a security bug.

## Accepted risks

- **Engines run with `--no-auth` on `127.0.0.1`.** By design (DESIGN.md "Shape"), reused
  from today's single-engine `mtplx serve` model. Any other local process/user on the same
  host can connect directly to an engine's ephemeral port once it knows it (and, per M2,
  the supervisor currently hands that port out cheaply). This is no worse than running
  `mtplx serve --no-auth` today, but the supervisor's multi-engine shape makes "which port"
  answerable at scale via `/health`/`/mtplx/admin/models`. Recommended (not required)
  hardening: a per-engine shared secret generated at spawn time and passed to the child
  via `--api-key` (engine-local, never the supervisor's own key), checked only by the
  supervisor's proxy client — closes the local-multi-tenant gap without touching the
  engine's own auth model.
- **`--insecure-lan` without `--api-key-file`.** `cmd_supervise_public`
  (`public.py:16440`) only requires a key when `not api_key and not insecure_lan`, so an
  operator can run `--host 0.0.0.0 --insecure-lan --yes` with no key configured at all.
  Verified this does not create an admin bypass: `_admin_authorized` requires
  `config.api_key` unconditionally, so with no key configured the admin API 401s for
  *everyone*, including the operator — a fail-closed footgun, not a vulnerability.
- **Process spawning is argv-from-config only.** Confirmed `EngineProcess.argv`
  (`process.py:127-145`) and `_engine_flag_pairs` (`public.py:16337-16361`) build argv
  entirely from CLI-parsed operator config; no remote request data reaches
  `subprocess.Popen`, and it is never invoked with `shell=True`.
- **Terminate/restart operate on the held `Popen` object**, not a re-looked-up pid, so
  there is no PID-reuse race in `EngineProcess.terminate` (`process.py:230-262`).

## FIX_BACKLOG

| id | severity | file | fix |
|---|---|---|---|
| H1 | High | `mtplx/supervisor/proxy.py:137-152` | Check auth from headers before reading the body; cap body read size (413 past limit) |
| H2 | High | `mtplx/supervisor/process.py:103-121`, `service.py:248-252` | Pass an explicit `env` to `EngineProcess` that strips `MTPLX_API_KEY` (and other secret-shaped vars) from the child |
| M1 | Medium | `mtplx/supervisor/admin.py`, `proxy.py` | Add the existing `_AuthRateLimitMiddleware`-style rate limiter in front of both apps |
| M2 | Medium | `mtplx/supervisor/admin.py:90-103` | Require admin key for the `supervisor` block of `/health`, or drop `pid`/`port` from the unauthenticated shape |
| L1 | Low | `mtplx/supervisor/service.py:139`, `admin.py:50-65` | Normalize path (case, `//`, dot-segments) once before admin/proxy dispatch; add regression test |
| L2 | Low | `mtplx/supervisor/service.py:317-334` | `asyncio.gather` the per-engine probes in `_health_sweep_once` |
