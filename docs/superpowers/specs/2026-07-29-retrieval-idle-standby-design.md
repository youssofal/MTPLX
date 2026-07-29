# Retrieval idle standby

## Problem

Retrieval models load on first request and then stay resident until the
`--retrieval-max-resident` cap is exceeded. With two configured models and a cap
of two, the cap can never be exceeded, so nothing is ever unloaded: several GB
stay held for the lifetime of the daemon even when no retrieval request has
arrived for hours.

The session bank (KV cache) has the same shape — it grows with use and is only
shed under system memory pressure.

Neither is a defect in the eviction logic. Idle release simply does not exist:
there is no idle timeout anywhere in the runtime.

## Non-goals

Unloading the **chat model** is out of scope. No unload path exists for it
anywhere in the core; adding one touches `ServerState`, warmup, the MTP contract
and session prefixes, and makes every subsequent request pay a cold start. It
deserves its own spec.

"Partially unloading" weights is not a thing — a transformer is resident or it
is not. What can be released separately is the MLX buffer pool, the session
bank, and whole model weights.

## Design

A watcher task runs beside the existing memory-pressure loop, every 30 s, and is
started only when an idle timeout is configured. Without configuration the
daemon behaves exactly as before.

Two triggers:

1. **Idle timeout** — retrieval backends whose last use is older than the
   threshold are unloaded; when every retrieval model is idle the session bank
   is archived to the existing SSD cold tier rather than dropped, so a resumed
   conversation restores its prefix from disk instead of re-prefilling.
2. **Memory pressure** — the existing guard gains a hook that unloads idle
   retrieval backends on the rising edge into CRITICAL.

Pinned backends — those inside an in-flight request — are never touched. The
watcher reuses the pin counting already used by the resident cap.

## Components

| Unit | Responsibility |
| --- | --- |
| `RetrievalRegistry.unload_idle(older_than_s)` | Unload unpinned backends past the threshold; return freed bytes and the ids affected. Pure registry logic, unit-testable without weights. |
| `_retrieval_idle_loop(state, interval_s)` | Call `unload_idle`, then archive the session bank when nothing is resident. Never raises into the server. |
| `--retrieval-idle-timeout` | Seconds; `0` disables. Threaded through the `serve`/`quickstart` parsers, the child argv builder, and the server parser — the three points the retrieval flags already traverse. |
| `config.toml: retrieval_idle_timeout` | Persisted default. |
| Snapshot `retrieval.idle_timeout_s` + per-model `idleSeconds` | Lets the app show "unloads in N min" rather than only "loaded". |
| App setting + Activity display | Timeout field beside the resident cap; countdown in the retrieval card. |

## Data flow

```
watcher (30 s)
   ├─ registry.unload_idle(timeout)  → frees weights, updates residency
   └─ sessions.archive_cold_tier()   → session bank to SSD, when idle
                                     ↓
                          next snapshot reports freed memory
```

## Error handling

The watcher must never take the server down: exceptions are logged and
swallowed, and a failed archive does not prevent the weight unload. Unloading a
backend that is concurrently acquired is impossible by construction — the pin
count is checked under the registry lock.

## Testing

- `unload_idle` respects the threshold, skips pinned backends, reports freed
  bytes, and is a no-op when nothing is stale.
- The watcher is not started when no timeout is configured.
- Idle seconds appear in the snapshot and decode in the app, including the
  absent-field case for older daemons.

## Measured baseline

On the machine this was designed against: chat weights 18.4 GB, session bank
6.9 GB, retrieval 6.1 GB when both models are resident. The scope above targets
the latter two, roughly 13 GB.
