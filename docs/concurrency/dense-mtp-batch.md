# Dense MTP batch lane (Qwen3.5 / Qwen3.8 family)

Batched multi-token-prediction decode for **dense** `qwen3_5` models, with
continuous admission: requests join a batch that is already running, and the
batch width follows demand.

This page is the operator guide — how to turn it on, what to set, and how to
tell whether it is working. **The caller-facing guarantees are a separate
document and you should read it before depending on any of them:**
[Dense batched MTP serving — what this lane promises](../dense-mtp-batch-contract.md).

**In plain terms.** Normally the server answers one request at a time. This lane
lets several requests share one pass through the model, so each of them costs
less. Unlike a fixed batch, the group is not decided in advance: a request that
arrives while the group is running joins it, and a request that finishes leaves
it immediately.

---

## Quickstart

```bash
mtplx serve \
  --model /path/to/Qwen3.8-27B-MTPLX-Optimized-Speed \
  --scheduler-mode mtp_batch \
  --decode-batch-max 8 \
  --max-active-requests 8
```

That is the whole of it. `--generation-mode mtp` and `--load-mtp` are already
the defaults, so **`--scheduler-mode mtp_batch` is the only flag that turns the
lane on**. Send ordinary `POST /v1/chat/completions` requests; concurrency is
what triggers batching and nothing per-request opts in.

**`--decode-batch-max` is the batch width.** The name says "decode batch max"
and means "how many requests may share a pass" — worth knowing, because nothing
in `--help` says so today.

### What the model has to be

The lane checks all of this at startup and **refuses loudly** rather than
falling back:

| requirement | why |
|---|---|
| `model_type` is `qwen3_5` | the lane is built on this trunk's cache layout |
| dense, not mixture-of-experts | MoE has its own lane, with its own router receipts |
| an MTP sidecar is present and loads | the decode cycle is draft-then-verify; there is no autoregressive path here |
| `--decode-batch-max` >= 2 | a width below two is not a batch |

If your server starts and the lane did not install, the error says which of
these failed. See §10 of the contract.

---

## Settings that matter

| setting | default | what it does |
|---|---|---|
| `--decode-batch-max N` | 8 | **Maximum** batch width. Not a target: three callers run as a batch of three. |
| `--batch-wait-ms` | 20 | How long a forming batch waits for company before starting. Lower means lower latency for the first caller; higher means more callers in the opening batch. With continuous admission this matters much less than it used to, because late arrivals are no longer locked out. |
| `MTPLX_DENSE_BATCH_MAX_QUEUE_DEPTH` | `8 x width` | Admission queue bound. Past it, new requests get **HTTP 503 with `Retry-After`** instead of being queued for a long time. |
| `MTPLX_DENSE_BATCH_DEADLINE_S` | unset | Wall-clock bound on one batch. Off by default on purpose — a lane that truncates because nobody chose a timeout is worse than one that visibly runs long. |
| `MTPLX_DENSE_BATCH_RESIZE_DEBUG=1` | off | Prints the row set at every admission boundary. Reach for this when you see a shape error: the failure surfaces several frames away from the resize that caused it, and the row set names the cause immediately. |
| `MTPLX_DENSE_BATCH_MEMLOG=1` | off | Per-cycle memory trace. |

**`memory_headroom`** (default `0.85`, `0` disables) is the fraction of the
GPU's recommended working set a **growing** batch may occupy. Before adding
rows the lane projects what they cost — each row's own KV over its lifetime,
plus the transient of rebuilding the batch buffer one row wider — and admits
only as many as fit. A row that does not fit waits for capacity instead of
taking the batch down with an out-of-memory.

It **throttles, it does not strand**: every request is still served, in
narrower groups. And it is an estimate over an allocator it does not control —
MLX's pool, the prefill intermediates and the weights are all outside it — so
it is a budget rather than a guarantee, which is why it can be switched off and
why the fail-loud path is still there.

It bounds **growth only**. A batch that was *sealed* too wide for the machine
could always run out of memory, and that is `--decode-batch-max` to choose.

Watch `rows_blocked_by_memory` in the snapshot: a lane running narrow for want
of headroom looks exactly like a lane nobody is using, and that difference is
the whole reason you would change a setting.

Two service-level settings have no CLI flag yet and are constructor arguments:

- **`continuous`** (default on). Turns continuous admission off, restoring the
  older behaviour where a batch's membership is fixed when it starts. The one
  reason to want this: with it on, a request that is genuinely alone goes
  through the batch driver rather than the solo path, because the server cannot
  tell at start-of-batch whether company is coming. Measured on a 4B, a lone
  request was *faster* through the batch driver, so this is unlikely to be the
  setting you want — but it exists.
- **`max_requests_per_cohort`** (default `8 x width`). How many requests one
  batch will serve before winding down and letting a fresh one form. Bounds
  per-batch bookkeeping; it is not a correctness limit.

---

## Is it actually batching?

Configured mode proves what was selected, not what happened. Read the lane's own
counters:

```bash
curl -s http://127.0.0.1:8000/v1/mtplx/snapshot \
  | jq '.scheduler.dense_mtp_batch | {
      batch_histogram, last_real_width, rows_peak,
      cohort_resizes_total, requests_served_total,
      max_requests_in_one_cohort, continuous_batching_observed,
      queue_wait_s, live_cohort
    }'
```

| field | what it tells you |
|---|---|
| `batch_histogram` | how many batches started at each width |
| `last_real_width` | the width the most recent batch **started** at |
| `rows_peak` | the widest any batch actually **ran**. Under continuous admission this exceeds `last_real_width`, and the gap between them is the feature working. |
| `active` / `active_sealed` / `active_joined` | requests this batch has **taken on** — the ones it started with plus everything it has pulled since. The gap between sealed and joined is continuous batching, visible directly. **Not the number of rows decoding right now**: a batch at four slots legitimately reports eight here once four of its requests have finished and been replaced. |
| `live_cohort.driver.live_rows` / `max_rows` | **rows decoding right now**, against the ceiling. This is the pair to watch if you want to know whether the lane is running as wide as it could — `active` cannot answer that and will mislead you if you ask it. |
| `live_cohort.growth_blocked` | why the batch is not wider, when it is not: `at_width_cap`, `cohort_request_cap` (winding down), `no_compatible_work`, `queue_empty`, or `null`. **`null` beside a non-empty queue and rows below capacity is the one that means something is wrong.** |
| `unservable_total` | requests refused because one of them could not fit in memory even alone. Distinct from `rejected_total`, which is the queue bound: this says the machine is too small for the request, not too busy for it. |
| `rows_blocked_by_memory` | rows the working-set budget kept out. Non-zero means the lane is running narrower than demand because it is out of GPU headroom. |
| `last_error_context` | on an out-of-memory: the width, request count, longest prompt and token budget the batch was carrying when it failed, with the advice that KV scales with the LONGEST concurrent prompt rather than the average one. |
| `cohort_resizes_total` | rebuilds of the row set. **Zero on a busy server means the queue is never being pulled from** — that is what "present but not working" looks like from outside. |
| `max_requests_in_one_cohort` | one batch serving more requests than its width is the single fact that distinguishes continuous batching from ordinary batching |
| `continuous_batching_observed` | the above, as a boolean |
| `queue_wait_s` | p50/p90/p99 of how long callers waited before work started for them. This is the number that moves most. |
| `live_cohort.health` | `healthy` / `starting` / `STALLED`. **Read this, not `seconds_since_progress` alone** — see §8 of the contract for why a healthy batch can go quiet. |

Every response also carries `mtplx_stats.dense_mtp_batch_cohort_width`, so a
caller can see the width it decoded at — which matters, because output depends
on it (§1 of the contract).

---

## What batching costs you

**Output depends on batch geometry.** A request decoded beside seven others does
not produce the same tokens it would have produced alone, and under continuous
admission the width changes *during* a request as neighbours come and go. This is
the ordinary property of every batched LLM server, it is measured rather than
assumed, and §1 of the contract has the detail. What it is *not*: dependent on
who else is in the batch. Neighbour content was measured and makes no
difference; the neighbour *count* does.

**A request that is genuinely alone pays a little.** See `continuous` above.

**KV memory is per row.** At roughly 64 KiB per token per row on the 27B, eight
rows at 128k context is 64 GB of working memory. Choose width and maximum
context together, not separately. §9 of the contract has the table.

---

## Reusing work across conversation turns

A chat client resends the whole conversation every turn, so turn four otherwise
pays to read turns one through three again. This lane can skip that work by
reusing what it computed for an earlier turn.

**It is off by default.** It holds gigabytes, and no long soak has exercised it
yet. Turn it on deliberately, with a byte budget:

```bash
MTPLX_DENSE_BATCH_PREFIX_CACHE_BYTES=$((6 * 1024 * 1024 * 1024)) \
mtplx serve --model /path/to/model --scheduler-mode mtp_batch \
            --decode-batch-max 4 --max-active-requests 4
```

Zero, the default, disables it entirely.

### What it does for you

Measured on a real 4B, a four-turn conversation:

| turn | prompt | reused | prefill |
|---|---|---|---|
| 1 | 626 | 0 (nothing stored yet) | 2.20 s |
| 2 | 674 | 610 (91%) | 0.36 s |
| 3 | 722 | 658 (91%) | 0.34 s |
| 4 | 770 | 706 (92%) | 0.35 s |

Reuse HOLDS as the conversation grows rather than decaying, which is the point:
a cache that helps turn two and not turn ten is not a chat cache.

**It also helps requests that merely share a preamble.** Four different
conversations sharing one system prompt produced three reuses on the very first
round, before any turn repeated. Agent and evaluation traffic -- many prompts
behind one long instruction block -- is this shape.

### Does it change the answers? Yes, sometimes — here is the honest number

**About three rows in eight** get a different answer with reuse on, at
temperature 0 as well as at 1.0. An earlier version of this page claimed the
answers do not change, on the strength of one conversation that matched
perfectly. That was one favourable case reported as a rule.

What that does and does not mean:

* The changed answers are **coherent answers**, not corruption. The one read in
  full was more accurate than the uncached one.
* Reuse is **deterministic** — the same cache gives the same answer every time.
* **This is the size of effect `prefill_chunk` already has.** With no cache
  anywhere, changing that setting from 256 to 512 changes 4 rows in 8. If you
  have ever tuned it, you have already accepted this.

**Why.** On this trunk, where a prefill pauses changes the arithmetic — the
running state is accumulated chunk by chunk, and a different layout adds it up
differently. Reuse resumes from a pause point, so it inherits that. A
pure-attention model would show none of it.

**What IS promised:** the lane verifies that restored state belongs to this
prompt and falls back to a full prefill if it cannot confirm that. Resuming from
the wrong point would give a fluent, confident, wrong answer with nothing
reporting an error, so it is checked rather than trusted.

### What it costs

The cache holds two things: the addressable working memory for each stored
prompt, and a few snapshots of the model's *running* memory, which this
architecture cannot rewind and therefore has to photograph.

**Those snapshots are not cheap.** One is **49.1 MB on the 4B**, against 65.1 MB
for a full 512-token cache -- the running state is a fixed size that does not
shrink just because you took it early. On the 27B, with twice the layers and
wider heads, expect several times that. The lane therefore caps how many it
keeps (four per entry by default) rather than letting the count grow with prompt
length.

Budget accordingly: the byte limit you set is a real ceiling for the entries
themselves, and the cache evicts least-recently-used to stay under it.

**Read the budget as a floor on what is held, not a ceiling.** The limit governs
the stored conversation caches. The recurrent snapshots attached to them are
accounted separately, so total process memory sits above the number you set. On
the 4B with a 2 GB budget, expect the server to hold roughly 3.5 GB once warm.
Size the machine for that, not for the budget alone.

This is worth stating because getting it wrong was a real defect: an earlier
version carried every snapshot a conversation had ever taken forward from turn
to turn. Nothing was copied -- the references were shared -- but nothing could
be freed either, and the budget never noticed, because the budget does not count
them. The cache reported a flat 1.7 GB while real memory climbed by about
49 MB per turn. Fixed by carrying only the most recent few forward, which took
per-conversation growth from accelerating to constant and halved it outright
(6.56 GB -> 3.10 GB over twelve turns), with reuse unchanged.

### Is it earning its memory?

`GET /health` -> `scheduler.dense_mtp_batch.prefix_cache`:

```
entries               how many prompts are stored
restores              how many requests reused work
restore_failures      how many tried and fell back (a healthy server: 0)
prompt_tokens_skipped total prompt tokens not re-read
hit_rate              restores / requests served
total_nbytes          what it currently holds
```

**`hit_rate` is the number that decides whether to keep it on.** High on chat or
agent traffic is the point. Zero on a busy server means it is holding gigabytes
for nothing, which is worse than not having it -- turn it off.

A miss is not a failure. `restore_failures` counts attempts that fell back to a
normal prefill; a cold start produces misses and no failures.

### Tuning

Defaults are reasonable; these exist for when they are not.

| variable | default | what it changes |
|---|---|---|
| `MTPLX_DENSE_BATCH_PREFIX_CACHE_BYTES` | 0 (off) | the whole feature, and its ceiling |
| `MTPLX_DENSE_PREFIX_MAX_BOUNDARIES` | 4 | snapshots CAPTURED per prefill -- raise for more reuse, at ~49 MB each on a 4B. An entry may list more than this: boundaries inherited from a shorter entry are shared references and cost nothing extra. |
| `MTPLX_DENSE_PREFIX_LADDER_STRIDE` | 1024 | how far apart the coarse snapshots sit |
| `MTPLX_DENSE_PREFIX_TAIL_GUARD_NEAR` / `_MID` | 16 / 64 | how far below a prompt's end to snapshot -- these carry the chat case |

The tail guards matter more than they look. Consecutive turns of a conversation
do not diverge in the middle; they diverge a few tokens before the end, where
the chat template's generation marker begins and the next turn instead continues
with the assistant's reply. Snapshots placed just below that point took reuse
from a decaying 76% / 71% / 66% to a steady 91% / 91% / 92%. If you use a
template with an unusually long generation suffix and reuse looks poor, raise
the guards.

### If reuse reads as zero

Check the lane actually ran before suspecting the cache:
`scheduler.dense_mtp_batch.requests_served_total` must be climbing. Sequential
single requests never form a cohort and go through the solo path instead, and
`--scheduler-mode mtp_cohort_experimental` falls back to `ar_batch` with
speculation disabled. Use `mtp_batch`, set `--decode-batch-max`, and send
genuinely concurrent requests. A wall-clock improvement is NOT evidence this
lane did anything -- the solo path has its own reuse and will happily produce
one.

## Reading the results of a benchmark

Numbers for this lane are in [benchmarks.md](../benchmarks.md). Two cautions
that have already cost time:

- **Warm the server before timing anything.** The first request after an idle
  period pays a one-off cost — measured at 2.3 s to first token against 0.15 s
  warm, on the same build. A measurement that lands on it reports a real number
  about the wrong thing.
- **A thinking model with no effort cap can look like a broken server.** A
  caller asking for 200 tokens can get 200 tokens of reasoning and an empty
  `content`. Check
  `usage.completion_tokens_details.reasoning_tokens` before concluding the lane
  returned nothing.
