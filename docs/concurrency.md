# Concurrency modes

Concurrency is a server scheduling capability. It is not tied to one model,
one batch width, or one kernel geometry. Each model backend declares which
scheduler routes it can install and validates its own shapes and limits when
the model loads.

MTPLX keeps model execution on one owner thread. A scheduler decides whether
requests run alone or share model work. Even when requests share a batched
allocation or forward pass, they keep separate prompt history, sampler state,
random-number state, stop state, and logical KV ownership.

## Scheduler modes

| Value | Behavior |
|---|---|
| `serial` | Run one request at a time on the backend's normal generation route; this is the default |
| `cooperative` | Interleave independently owned request work where the backend supports it |
| `ar_batch` | Batch target-only autoregressive decode on a compatible backend |
| `mtp_batch` | Batch native-MTP decode using a model-specific installed MTP lane. On a dense `qwen3_5` model this selects the [dense MTP batch lane](concurrency/dense-mtp-batch.md), which admits requests continuously rather than fixing a batch's membership when it starts. |
| `mtp_cohort_experimental` | Opt into experimental native-MTP cohort scheduling |

The mode name is generic. It does not define a batch width, speculative depth,
context limit, tensor shape, or numerical policy. Those are properties of the
installed model/backend implementation.

## Ownership contract

Every admitted request owns its own:

- prompt and generated tokens;
- logical KV and recurrent state;
- target and draft sampler settings;
- seeded random-number stream;
- token budget, stop state, cancellation event, and output stream.

A backend may place those values in shared batched buffers. Row-specific masks,
offsets, commits, and rewinds must still prevent one request from reading or
changing another request's context. Fixed-shape lanes may run in lockstep; that
is shared scheduling, not shared context.

An optimized lane is installed only after its backend validates the model,
geometry, dtype, cache layout, kernels, and construction self-checks. An
unsupported combination fails clearly instead of silently changing to AR or a
different kernel.

## Select a mode

Choose the mode when the server starts. Backend-specific modes may require
additional flags:

```bash
mtplx serve \
  --model <model-or-path> \
  --scheduler-mode <serial|cooperative|ar_batch|mtp_batch|mtp_cohort_experimental>
```

Or save the scheduler choice:

```bash
mtplx config set scheduler_mode mtp_batch
mtplx config show --json
```

Scheduler and kernel routes are construction-time settings. Stop and restart
the server after changing them. A CLI value overrides the saved value for that
launch. Other required flags depend on the selected backend; use its linked
implementation guide instead of copying geometry from another model.

## Backend implementations

Model-specific guides record the exact supported features, required launch
contract, batch geometry, numerical choices, limitations, and benchmark
receipts for each installed lane:

- [Qwen3.6 35B A3B fixed B8/K1 MTP lane](concurrency/qwen35b-mtp-batch.md)
- [Dense MTP batch lane, Qwen3.5 / Qwen3.8 family](concurrency/dense-mtp-batch.md)
  — continuous admission: requests join a batch already running, and width
  follows demand in steps of one. Caller-facing guarantees are in
  [what this lane promises](dense-mtp-batch-contract.md).

This list describes available implementations. It does not redefine the
generic concurrency modes or imply that future MTP lanes must use the same
width, depth, context limit, or kernels.

## Confirm concurrent behavior

The health payload reports the selected scheduler and observed execution:

```bash
curl -s http://127.0.0.1:8000/health | jq '.scheduler | {
  mode,
  active_lane,
  config,
  telemetry
}'
```

The configured mode proves only what was selected. To prove concurrent work
actually ran, use the backend guide's behavioral receipt, such as an observed
multi-row width or batch histogram after sending simultaneous requests.
