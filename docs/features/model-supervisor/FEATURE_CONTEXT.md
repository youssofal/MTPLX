# Feature Context — Model Supervisor

Date: 2026-09-22. Branch: `feat/model-supervisor` (worktree `~/Code/MTPLX-supervisor`, cut from upstream `main` @ 7c2205a, v2.11.3).

## Problem

MTPLX runs one model per daemon, chosen at start. Everything around that boundary is where
users hit friction: switching models means a restart (#402, #253, #366), a wedged or
memory-crept engine needs a human to restart it (#456, #438, #504), a pack that does not fit
crashes or refuses with the wrong number (#510, #400), and headless operators have no
lifecycle API (#253) and fight the API-key rule on LAN (#491). LM Studio and Ollama solve
all of this with one component: a supervisor that owns engine processes.

## Scope (in)

| ID | Requirement | Source |
|---|---|---|
| R1 | Route `/v1/*` and `/v1/messages` requests to a per-model engine by the request `model` field; unknown/omitted model falls back to the default (first loaded) engine | #402 |
| R2 | JIT load: a request for an installed-but-unloaded model loads it, holding the request until healthy (bounded by a configurable timeout) | #402, #366 |
| R3 | Memory budget with LRU eviction and idle-TTL unload; budget derived from total RAM minus a system reserve, overridable | #402, #456 |
| R4 | Admission control: refuse to load a pack whose resident estimate exceeds the remaining budget, with an error that names the pack size, the budget, and what would have to be unloaded | #510, #400 |
| R5 | Self-healing: detect engine crash (process exit), health-check failure, or footprint over threshold; drain in-flight requests, restart with exponential backoff, cap restarts per window | #456, #438, #504 |
| R6 | Management API under `/mtplx/admin/*`: list (loaded + installed + memory), load, unload, restart, and a supervisor health endpoint | #253 |
| R7 | Auth: admin endpoints always require the API key; inference endpoints follow existing rules; an explicit `--insecure-lan` opt-in disables the LAN key requirement with a loud warning | #491, #253 |
| R8 | Backward compatible: `mtplx serve --model X` behaves exactly as today (single engine, same argv, same port); the supervisor is `mtplx supervise` / `mtplx serve --supervisor` and the Swift app is unaffected until it opts in | all |
| R9 | Streaming passthrough with no added buffering; per-request logs carry the engine id | all |

## Scope (out)

- Batching across engines, shared KV, or any engine-internal change.
- Running models larger than RAM (#510's swap request) — admission control says no instead.
- GUI changes in the macOS app (a later feature can point the app at the supervisor).
- Downloads on demand: JIT loads only installed packs.

## Success criteria

1. Two packs (Qwen3.5-9B and Qwen3.8-27B) served by one supervisor on one port; requests
   alternate by `model` field with no restart; measured per-model tok/s within 3% of the
   standalone daemon.
2. Killing an engine process mid-stream: in-flight request gets a clean error, the engine is
   back and answering within 60 s, no human action.
3. Requesting Flash-Next on a 64 GB machine with the 27B loaded: refused with a message that
   states sizes and the eviction that would free enough, or auto-evicts when `--evict-to-fit`.
4. All existing tests pass unchanged; new tests cover router, budget/LRU, admission math,
   restart policy, and admin auth with fake engines (no model needed).

## Constraints

- Python 3.12+, the existing dependency set (check `pyproject.toml`; prefer stdlib
  `asyncio` + the HTTP framework already used by the server).
- No engine code changes in this feature; the engine is driven only through its argv, its
  health endpoint, and process signals.
- Upstream style: plain-language docs, hyphens not em-dashes, no ASCII art.
