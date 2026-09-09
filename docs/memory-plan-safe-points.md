# Memory-plan safe points

Routine enforcement of the SessionBank dynamic ceiling now waits for a safe
point. This implements the synchronization requested in PR #357 within the
existing memory-plan governor. `bank_dynamic_ceiling` and
`SessionBank.effective_max_bytes()` remain the budget policy; no additional
controller, environment switch, utilization thresholds, or growth policy is
introduced.

The pressure watcher samples the existing effective ceiling before acquiring
any model lock. If the bank exceeds that ceiling by the existing 256 MiB slack,
it tries the model lock without waiting, then tries the scheduler's idle gate.
Under both gates it rechecks foreground and in-flight request counts. Missing
or failed activity probes defer the operation. A bank replaced since sampling
also defers. A deferred tick samples again on the next normal watcher tick.

The scheduler gate excludes queued or running foreground, idle postcommit,
persistence, and keepalive work. Claiming an item and removing it from its queue
happen under the same scheduler condition, so the dequeue-to-start interval
cannot appear idle. Holding that condition through maintenance prevents a new
owner item from starting after the idle check. The model lock excludes restore,
commit, and MTP transactions, while the request count covers gaps between owner
items in a single foreground request.

At a safe point, the watcher trims with `protect_active=True`, retaining the
existing recent-session policy. It never changes `max_bytes` or per-session
caps, clears the allocator cache, or executes model work. Deferred attempts are
recorded as `dynamic_ceiling_deferred` with a reason in the existing bounded
memory-guard event history. Successful evictions retain the `dynamic_ceiling`
receipt.

## Pressure behavior and qualification limits

Routine dynamic-ceiling eviction can now defer throughout a long prefill; the
old watcher could trim idle entries during that prefill. This is the deliberate
tradeoff of requiring every model stage to be idle for routine maintenance.
The existing admission projection, structured 507 refusal, SSD spill, and
allocation-failure/sustained-critical safeguards remain in charge during active
requests. The separate emergency pressure responder is unchanged: WARNING
retains its bounded defer interval and CRITICAL still sheds immediately even
when the model lock is held. Those emergency operations are not represented as
safe-point maintenance.

Focused tests cover lock contention, foreground arrival during sampling, failed
probes, queued and active scheduler lanes, the dequeue gap, keepalive,
bank replacement, exception cleanup, and critical shedding during a foreground
request. Existing scheduler, plan, admission, SessionBank and SSD regressions
are run alongside them. These tests do not establish live model throughput or
protection against a physical memory-pressure event. Earlier synthetic numbers
for the removed standalone controller do not measure this integration.
