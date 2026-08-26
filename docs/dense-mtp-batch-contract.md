# Dense batched MTP serving — what this lane promises

This is the caller-facing contract for the dense batched multi-token-prediction
(MTP) serving lane. It covers what the lane guarantees, what it explicitly does
not, and what an operator has to decide before turning it on.

**In plain terms.** Normally the server answers one request at a time. This lane
lets several requests share one pass through the model, which makes each of them
cheaper. Sharing has consequences, and this page is the list of them — written
so you can tell, before you depend on something, whether it is actually
promised.

## Glossary

Terms used throughout. Where this document coins an abbreviation, it is defined
here and at first use.

| Term | Meaning |
|---|---|
| **MTP** (multi-token prediction) | The model guesses several tokens ahead, then checks the guesses in one pass. Correct guesses are kept, so several tokens can be produced for roughly the cost of one. |
| **Dense** | This model keeps all of its weights active for every token, as opposed to a mixture-of-experts model that activates a subset. The lane is for dense models only and refuses the others. |
| **Cohort** | The group of requests decoding together in one pass. |
| **Row** | One request's place in a cohort. Rows are added when requests join and removed when they finish, so a cohort's rows are not fixed for its lifetime. Older text on this page says **slot**, which implied a fixed set of places that requests take turns in; that is no longer how it works. |
| **Width / geometry** | How many rows a cohort has *right now*. This turns out to matter for output, which is why it has a name — and under continuous admission it changes during a request. |
| **Joining / admission** | A request entering a cohort that is already running. It is prefilled at its own length and added as a new row. |
| **Resize** | One rebuild of a cohort's row set: finished rows removed, joining rows added. Counted as `cohort_resizes_total`. |
| **Backpressure** | Refusing new work when the queue is full, instead of accepting work that will not be served for a long time. |
| **KV cache** | The model's stored working memory for the text so far. It is the dominant memory cost, and it is per slot. |
| **GDN** (gated delta net) | The layer type used by most of this model's layers. It keeps a fixed-size running state rather than a per-token history, so unlike the KV cache it does not grow with context length. |

---

## 0. Requests join and leave a running cohort, and width follows demand

**What is promised.** A request that arrives while a cohort is running joins
that cohort, usually within one decode cycle, rather than waiting for it to
finish. A request that finishes is removed from the cohort immediately rather
than occupying a row that is still computed. Cohort width is therefore exactly
the number of live requests at any moment, and it moves **in steps of one**, up
to the configured maximum. Three callers run as a batch of three.

**In plain terms.** The group is not decided in advance. People join the group
when they show up and leave it when they are done, and the group is exactly as
big as the number of people in it.

**What changed, and why the history matters.** This page previously described
*refill*: at the moment a cohort sealed, the service chose a bounded list of
waiting requests that would be allowed to join it later, and anything arriving
after that instant waited for the whole cohort to drain. Measured on
`Qwen3.8-27B-MTPLX-Optimized-Speed` at commit `7f37ec4`, eight simultaneous
requests split across three separate runs and the median caller waited **10.06
seconds** before any work started for it. At `fe74b93` the same load runs as one
cohort that grows from its opening width to eight, and the median wait is
**0.11 seconds**.

**What this costs, and it is not nothing.** Width changing during a request means
the geometry a request decodes at changes during a request. Read §1 — it was
always true across runs, and it is now true *within* one.

**Two admission rules a caller can observe**, both mirroring a decision the
driver makes once when a cohort starts and cannot revisit:

* A request wanting `temperature > 0` will not join an **all-greedy** cohort. An
  all-greedy cohort takes a dedicated path that consumes no randomness and has
  no sampler state to update, so such a request waits for the next cohort rather
  than being served with the wrong sampler.
* A request wanting a presence or frequency penalty will not join a cohort built
  without penalty machinery, for the same reason.

Neither is a refusal; both are a short wait. Ordinary sampler differences do
**not** delay a joiner — the driver rebuilds per-row sampling on admission, so a
joiner keeps its own temperature, top-p, top-k, seed and penalty counts, and
inherits nothing from any earlier occupant of the cohort.

**How a joiner's prompt is handled, because the alternative was silently
wrong.** Each joining request is prefilled at its **own** prompt length and its
rows are concatenated onto the batch. It is never padded to a shared length. On
this trunk most layers are gated delta nets, which keep a running state rather
than an addressable history: a pad token fed to one is folded into that state
permanently, and no offset rewinds it. The failure mode is not a crash — the
model loads, runs, and returns fluent text conditioned on tokens the caller
never sent. Prefilling each joiner separately removes the possibility rather
than guarding against it.

---

## 1. Output is not reproducible across different batch widths

**The promise:** at a fixed cohort width, with a fixed seed, the same request
produces the same tokens.

**Not promised:** that a request produces the same tokens it would produce alone,
or in a cohort of a different size.

This is not a bug and it is not fixable at this layer. Floating-point addition is
not associative, so a matrix multiply of a different shape sums its terms in a
different order and lands on slightly different numbers. Where the top two
candidate tokens are nearly tied, that slight difference is enough to flip which
one wins, and once one token differs the continuations diverge. This was measured
on this model during development, not assumed.

**In plain terms.** Running your request next to three others gives an answer
that is just as valid, but not necessarily the identical answer to running it by
itself. If you need byte-identical reruns, you must pin the batch width, not just
the seed.

**What this means for you:**

- Every response reports the width it decoded at, as
  `mtplx_stats.dense_mtp_batch_cohort_width` — 1 for a request served alone.
  Use it: if you are comparing two runs, check the widths match before
  concluding anything about the difference.
  (This was briefly untrue: the field was set on every response and then
  stripped by the server's public-stats allowlist, so no caller could read it.
  Fixed, and now checked by a test that reads an actual HTTP response rather
  than the internal result object.)
- Do not use this lane as the reference for output-comparison tests unless you
  also pin the width.
- Caching keyed on `(prompt, seed)` alone will produce cache hits that do not
  match what a rerun would generate.

## 2. Seeds are per request, and independent of other callers

Each request's randomness derives from its own seed. Another caller sharing your
cohort cannot change your output by changing their seed, and a request that
joins a slot part-way through starts from its own seed rather than continuing
where the slot's previous occupant left off.

**Earlier behaviour, now fixed — two separate defects:**

1. The whole cohort used the first request's seed.
2. More subtly, a joining request's **first token** was drawn from its
   predecessor's stream, because the per-row reset ran after that token was
   sampled. Everything after the first token was correct, which is why it
   survived review and a working test.

Seeds interact with §1: your seed controls your randomness, but width still
affects your logits, so a seed alone does not pin an output.

The seed a request actually decoded under is reported back as
`mtplx_stats.dense_mtp_batch_cohort_seed`.

**In plain terms.** Your seed is yours. It is not, by itself, enough to
reproduce a result.

## 3. Sampling settings are per request

Temperature, `top_k` and `top_p` are per row, including for a request that joins
part-way through. A joiner does not inherit the settings of whoever held the slot
before it.

**Earlier behaviour, now fixed:** a joining request's **first token** was
sampled at its predecessor's temperature, for the same ordering reason as §2.

One restriction: a cohort in which every request is greedy (temperature 0) runs a
dedicated path with no sampling machinery at all. A request that wants
temperature above zero will not join such a cohort; it waits for the next one.

## 3b. Presence and frequency penalties

Supported, per request, following the same formula as the non-batched path:

    penalty = frequency_penalty x count + presence_penalty x (count > 0)

subtracted from the raw scores before temperature, `top_k` and `top_p`, with
both coefficients clamped to the range -2 to 2, and counts covering the
generated text only — not your prompt.

**Earlier behaviour, now fixed:** the lane accepted these parameters and ignored
them. A request asking for `frequency_penalty: 1.5` got output identical to
`0`, with no error.

**One honest deviation.** The counts advance once per decoding cycle rather than
once per token, so within a single cycle the model does not yet see the tokens
it is drafting alongside. The staleness is at most `depth` tokens. Removing it
would mean giving up multi-token prediction entirely, which is the whole point
of the lane.

**In plain terms.** Penalties work, and they do what you expect — they stop the
model repeating itself. They are very slightly "behind" compared to running one
token at a time, by a handful of tokens' worth of bookkeeping.

A request asking for a penalty will not join a cohort that was started without
one; it waits for the next cohort instead. It is never served unpenalised.

## 4. Finish reasons

Each response carries a reason, and they are deliberately distinct:

| Reason | Meaning |
|---|---|
| `stop` | Hit a stop token. |
| `length` | Reached the caller's own `max_tokens`. |
| `deadline` | The server's wall-clock budget expired. **Not** the caller's limit. |
| `cancelled` | The caller went away and the row was evicted. |
| `not_admitted` | Queued but never given a slot before the run ended. |
| `cycle_cap` | The runaway backstop fired. Should not happen in normal operation; treat it as a bug report. |

**In plain terms.** `length` means you got what you asked for. `deadline` means
the server gave up. They look identical if you only check whether output was
truncated, which is why they are separate — a timeout reported as `length` sends
whoever is debugging it to entirely the wrong place.

## 5. Backpressure

The admission queue is bounded (default: eight cohorts' worth). When it is full,
new requests are refused immediately with **HTTP 503** and a `Retry-After`
header, rather than accepted into a queue they would sit in for a long time.

**In plain terms.** A "busy, try again" you can act on beats a request that
appears accepted and then takes ten minutes. Clients should honour `Retry-After`
and back off rather than retrying immediately.

## 6. Cancellation removes the row

If a caller disconnects, its row is evicted at the next cycle boundary — it does
not keep decoding to its `max_tokens`. Without this, sustained traffic with real
disconnects slowly drains the lane's usable width.

**Updated for continuous admission.** This used to say the row's *slot* became
available to a waiting request. The row is now removed from the cohort
altogether: the cohort narrows by one, and a waiting request is admitted as a
new row if there is one to admit. The practical difference is that an abandoned
request stops costing compute even when nothing is waiting to replace it, where
before it left an empty place in a rectangle that was still processed every
cycle.

## 7. Deadlines are off by default

There is no wall-clock limit unless an operator sets one (`cohort_deadline_s`).
A lane that truncates because nobody chose a timeout is worse than one that
visibly runs long — and §8 is what makes running long visible.

There is also a cycle-count backstop, which scales with queue depth. It is
deliberately generous: slots are filled as they free, so one fast-turning slot
can end up serving the entire queue in sequence, and a tighter bound would cut
that legitimate run short.

## 8. Observability — telling a stalled cohort from a slow one

`snapshot()` reports live state, not just running totals:

| Field | Use |
|---|---|
| `live_cohort.seconds_since_progress` | **The one that matters** — but read it with `health`, not alone. See the caveat below. |
| `live_cohort.health` | `healthy` / `starting` / `STALLED`, derived so you do not have to do the arithmetic. |
| `live_cohort.age_s`, `tokens_committed`, `tokens_per_s` | Context for the above. |
| `live_cohort.request_ids` | Which requests are affected. |
| `cohort_duration_s`, `queue_wait_s` | Recent distributions, as p50/p90/p99/max. |
| `pending` / `pending_new` / `pending_refill` | Total backlog, and its split. |
| `rejected_total`, `cohort_failures` | Refusals and failures. |
| `rows_peak` | The widest any cohort actually **ran**, as against `last_real_width`, which is the width it *started* at. Under continuous admission these come apart, and the gap between them is the feature working. |
| `cohort_resizes_total` | Rebuilds of a cohort's row set. **Zero on a busy server means the queue is never being pulled from**, which is what "the feature is present but not working" looks like from outside. |
| `continuous`, `max_requests_per_cohort` | Whether continuous admission is on, and how many requests one cohort will serve before winding down. |

**In plain terms.** When a server stops answering, the question is always "is it
stuck, or just busy?" Duration cannot answer it — a cohort running five minutes
is perfectly healthy if its requests are long ones. Whether tokens are still
arriving gets much closer, so that is the field to look at first.

**The caveat, found by measurement rather than reasoning.** Tokens stopping does
not always mean stuck. There are *three* states, not two, because a cohort can
be doing real work that produces no tokens:

| what you see | what it means |
|---|---|
| tokens arriving | working normally |
| no tokens yet, `health: starting` | reading the prompts (prefill). Expected, and can last a while for long prompts. |
| no tokens for a while, but the cohort was producing them a moment ago | usually a **new request joining** — its prompt has to be read before the group can continue |
| `health: STALLED` | genuinely stuck |

Observed in a soak run: 26 seconds of prefill, then 425 tokens, then another 27
seconds of nothing while a joining request's prompt was read — all healthy.
**Read `health`, which accounts for this. Do not build your own alarm on
`seconds_since_progress` alone**, or a busy server admitting new work will look
like a broken one.

**Continuous admission makes the third row MORE common, not less.** Joining used
to happen only when a row finished; it now happens whenever a request arrives
and there is room. Every one of those is a pause in token production while the
joiner's prompt is read, and a long joining prompt is a long pause. This is
working correctly, and an alarm built on token silence alone will fire on a
server that is doing exactly what it should.

Distributions report `{}` when there is no data, rather than zeros, so an empty
window does not read as a measurement of zero.

## 9. Memory: KV is per slot and does not shrink

The dominant memory cost is the KV cache, at a **measured 64 KiB per token per
slot** on this model.

| Slots | 8k context | 32k | 128k | 262k |
|---|---|---|---|---|
| 1 | 0.50 GB | 2.00 GB | 8.00 GB | 16.00 GB |
| 4 | 2.00 GB | 8.00 GB | 32.00 GB | 64.00 GB |
| 8 | 4.00 GB | 16.00 GB | 64.00 GB | 128.00 GB |

A row's capacity grows to the longest prompt it has served and **never comes
back down while that row exists**. Measured: four rows that served an
8192-token prompt hold 2.000 GB and keep holding it while serving 64-token
requests.

**Improved by continuous admission, and worth knowing which half improved.**
A finished request's row is now removed from the cohort, and its KV goes with
it, so the high-water mark no longer follows a *place* through a whole cohort's
lifetime — it belongs to a row that ceases to exist. A joining request gets a
fresh row sized to its own prompt. What has NOT changed: while a long request is
running, its row holds its own high-water mark, and a cohort's peak is still set
by its longest concurrent prompt rather than its average one.

This is a **peak within a cohort, not a leak**. Nothing holds a cache between
cohorts, so memory returns to baseline when a cohort finishes. If you are
watching a memory curve and it does *not* return to baseline between cohorts,
that is a real leak and not this.

**In plain terms.** Pick your slot count and your maximum context together, not
separately. Eight slots at 128k context needs 64 GB of working memory for one
cohort, and one oversized prompt sets the high-water mark for every request that
follows it in that slot until the cohort ends.

## 9b. Reuse across conversation turns — opt-in, and what it promises

**What is promised, when it is ON.** A request sharing a long leading prefix
with one this lane served earlier resumes from stored work and prefills only the
remainder. Measured on a real 4B: a four-turn conversation reused 91% / 91% /
92% of each prompt, with prefill falling from 2.20 s to ~0.35 s.

**Measured through the SERVER on a 27B, 2026-08-25** -- the figures above were
taken in process at width 1, and a caller reaches this lane through the server,
so the serving path is measured separately. Eight concurrent conversations of
four turns, sharing a system preamble, on
`ref-Qwen3.8-27B-MTPLX-Optimized-Speed`:

| turn | prompt tokens | reused | reuse | requests that hit |
|---|---|---|---|---|
| 1 | 628 | 567 | 79.0% | 7 of 8 |
| 2 | 786 | 628 | 80.1% | 8 of 8 |
| 3 | 939 | 786 | 83.4% | 8 of 8 |
| 4 | 1090 | 939 | 86.1% | 8 of 8 |

**82.7% of all prompt tokens were never re-read** (22,801 of 27,569). Reuse
rises with conversation length, and the FIRST turn already hits 7 of 8 because a
shared system preamble is reused across conversations, not only within one.

Reproduce with `python -m mtplx.benchmarks.runners.multiturn_cache_bench`
against a server started with `MTPLX_DENSE_BATCH_PREFIX_CACHE_BYTES` set.

**What that measurement does NOT cover.** Prompts ran 628 to 1090 tokens. Long
agentic context — the 24k range this lane is also benchmarked at — has been
measured only IN PROCESS, where a 10,000-token second turn reused 100% and
prefill fell from 23.7 s to 0.27 s. Whether the serving path holds that at 24k
is untested, and the mechanism gives reason to check rather than assume: reuse
depends on finding a recurrent boundary at or below the match, boundaries are
capped per entry, and the gap between the last boundary and the divergence point
grows with prompt length. A long prompt can match deeply and still find every
stored boundary above the point it needs.

**These numbers read 0% until 2026-08-25, and the cache was not the reason.**
`cached_tokens` and `cache_hit` were hardcoded to `0` and `false` in the
completion payload, so a row that reused 630 of its 755 prompt tokens reported
reusing none. Every external instrument agreed the feature did nothing -- the
OpenAI usage block, `/metrics`, the restore mode, and a multi-turn benchmark --
while the driver logged hits. Reuse is recorded per cohort ROW and a response
needs it per REQUEST, and nothing connected the two; fixing only the initial
cohort still under-reported, because under continuous batching most requests
JOIN and the joiner path was not recording either (12 driver restores showed as
3 on the wire). Both paths now attribute per request, into `stats`, which is the
dict the usage block is built from. **An optimisation that reports itself as
broken is one the next reader deletes**, which is why this is stated here rather
than left as a fixed bug.

**Sustained behaviour.** A 47-minute soak of 816 requests at 8 concurrent
conversations finished with **zero errors** and **0.44% RSS growth**, including a
mid-run reclamation of 123 MB that a leak cannot produce. Final RSS sat below
the pre-soak baseline. This is shorter than the 8-hour soak the lane itself was
held to, and is stated as 47 minutes rather than implied to be more.

**What is promised about the ANSWER — corrected 2026-08-24 17:33 HST, and the
earlier wording in this section was WRONG.** It said reuse does not change the
answer, on the strength of one two-turn conversation that matched 24 of 24
tokens. Measured over eight conversations at width 4, **reuse changes the answer
on about three rows in eight**, at temperature 0 as well as 1.0. The 24/24 was a
single favourable case reported as a guarantee.

**What is actually true, and it is still a usable promise:**

* **Reuse is deterministic.** Two banks built identically produce identical
  output, 4 of 4 rows. A cached answer is reproducible, not a lottery.
* **The changed answers are answers, not damage** — coherent and on-topic; the
  one case read in full was *more* accurate than the uncached one.
* **The size of the effect is the same as an existing tuning knob's.** Varying
  `prefill_chunk` alone, with no cache anywhere, changes 2 to 4 rows in 8
  (256 vs 512 → 4/8 identical; vs 128 → 6/8; vs 1024 → 5/8). Reuse changes 3
  in 8. Anyone who has changed `prefill_chunk` has already accepted this.
* **The restored state is verified, not assumed.** The lane checks that what it
  restored corresponds to this prompt's own tokens and **fails closed** — full
  prefill — if it does not. Resuming from the *wrong* point would produce a
  fluent, confident, wrong answer with nothing reporting an error, which is why
  this is checked rather than trusted.

**Why it cannot be promised away.** On this trunk, WHERE A PREFILL PAUSES
CHANGES THE ARITHMETIC. Recurrent state is accumulated in chunks, and a
different chunk layout sums it differently. Reuse necessarily resumes from a
pause point, so it inherits this. It is the same phenomenon as `prefill_chunk`
already exhibiting it, not a new class of problem — and a pure-attention model
would show none of it.

Section 1's batching caveat still applies on top: batched output is not
bit-identical to solo output, for the same underlying reason.

**What is NOT promised.**

* **It is off by default.** `MTPLX_DENSE_BATCH_PREFIX_CACHE_BYTES=0`. It holds
  gigabytes, and an operator should turn it on deliberately rather than inherit
  it. **Soaked 2026-08-25**: 816 requests over 47 minutes at 8 concurrent
  conversations, zero errors, RSS +0.44%, including a mid-run reclamation of
  123 MB that a leak cannot produce. That is shorter than the 8-hour soak the
  lane itself was held to, and is stated as 47 minutes rather than implied to be
  more.
* **No hit rate is promised.** Reuse depends on how much of a prompt is shared
  and where the sharing ends. **A first turn is not necessarily a miss, and the
  earlier wording here said it was.** Measured on the 27B across 8 concurrent
  conversations, 7 of 8 FIRST turns hit, because every conversation on a server
  carries the same system preamble and that preamble is reused across
  conversations, not only within one. On agent and chat traffic — which is
  exactly the shape that has a large shared system prompt — the first turn is
  usually the second-best case, not the worst.
* **Memory is not free, and the expensive part is not the obvious one.** Each
  stored prompt keeps a few snapshots of the model's running state, which this
  architecture cannot rewind. One is **49.1 MB on the 4B** — 75% of a full
  512-token cache — and several times that on the 27B. The count is capped
  rather than scaling with prompt length.
* **The byte budget bounds the stored caches, NOT total process memory.** The
  snapshots are accounted separately. On the 4B with a 2 GB budget the server
  holds roughly 3.5 GB warm. Size for that.
* **Growth per conversation is linear in turns, not constant.** Each turn stores
  its own prompt, so a long conversation costs more than a short one and
  eviction is what bounds it. It was briefly QUADRATIC — every turn carried
  forward every snapshot its predecessors had taken, shared by reference so
  nothing was copied and nothing could be freed, invisible to the budget. That
  was a genuine leak and it is fixed; it is recorded here because "the budget
  holds" was true and reassuring while memory climbed anyway.

**In plain terms.** MTPLX can recognise "this is the same conversation plus one
more message" and skip re-reading what it has already read. This lane can now do
that too, if you turn it on. It gives back most of the wait on every turn after
the first, it does not change the answers, and it costs memory you must budget
for.

**How to tell it is working.**
`GET /health` → `scheduler.dense_mtp_batch.prefix_cache.hit_rate`. Zero on a
busy server means it is holding memory for nothing — turn it off.
`restore_failures` above zero means restores are being attempted and refused,
which is safe but wasteful. And confirm the lane ran at all:
`requests_served_total` must be climbing, or you are measuring the solo path.

## 10. When the lane refuses to install

The lane checks capability rather than consulting a list of approved model
names. It refuses, with a stated reason, on: mixture-of-experts models, runtimes
without MTP, capture backends that do not materialise state, and
`draft_core='compiled'`.

A refusal is reported. It does not silently fall back to another path — an
earlier version did fall back silently because a missing import was swallowed by
an overly broad exception handler, and the lane appeared to work while doing
nothing.

---

## 11. How these guarantees are checked

Everything above is a claim, and claims about a system are worth what the
checking behind them is worth. Each guarantee here is covered by a **mutation
test**: the fix is deliberately reverted in the source, and the test suite has
to go red. A guarantee whose reversion leaves the suite green is not protected,
whatever the test names suggest.

That is not a formality. Running it the first time found three guarantees on
this page that nothing was checking, and one of those turned out to be a fix
that had been reported as done while changing almost nothing observable.

**Two patterns worth knowing if you extend this lane:**

- **A guarantee phrased as an absence is the one most likely to be untested.**
  Every guarantee with a crisp observable — a finish reason, a counter, a status
  code — was already covered. Every uncovered one was of the form "X does not
  leak into Y". An absence gives you nowhere obvious to put a probe, which is
  both why it is hard to test and why it goes untested.
- **A test that asserts a direction cannot catch an error of magnitude.** A
  token counter advancing twice per token still changes the output — just more
  than it should — so every test asserting "the output changed" passes while the
  sampler is silently twice as aggressive as asked. Where a number matters,
  predict the number.

## What changed in this document, and when

Kept rather than edited away, because a guarantee reads differently when you
can see what it used to say.

| section | was | is | when |
|---|---|---|---|
| §0 | did not exist; joining was "refill", chosen when a cohort sealed | requests join a running cohort; width follows demand in steps of one | 2026-08-24, `fe74b93` |
| §1 | width is fixed for a request, so output is reproducible at a pinned width | width can change *during* a request | 2026-08-24, `fe74b93` |
| §6 | cancellation frees the **slot** for a waiting request | cancellation **removes the row**; the cohort narrows even if nothing is waiting | 2026-08-24, `fe74b93` |
| §8 | joining pauses are occasional | joining pauses are **more** common, because joining no longer waits for a row to finish | 2026-08-24, `fe74b93` |
| §9 | a slot's high-water mark persists for the cohort's life | it belongs to a row, and the row is removed when its request finishes | 2026-08-24, `fe74b93` |
| glossary | **slot** — a fixed place requests take turns in | **row** — added and removed with its request | 2026-08-24, `fe74b93` |

## Summary for the impatient

- **Requests join a running cohort**, and width follows demand in steps of one.
  A batch is not a fixed group decided in advance.
- **Pinning the width is not something a caller can do** while continuous
  admission is on: the width moves *during* a request, so identical reruns need
  the SERVER configured with `continuous=False`, not a request flag. Every
  response reports `dense_mtp_batch_cohort_width_varied` so you can tell whether
  a single width even applied to you.
- **Your seed is yours** — other callers cannot move your output, and a joining
  request inherits nothing from any earlier occupant of the cohort.
- **A joiner is prefilled at its own length**, never padded to a shared one.
- **`deadline` ≠ `length`** — one is the server giving up, the other is you
  getting what you asked for.
- **503 means back off**, and `Retry-After` says how long. **This is enforced at
  the ASGI boundary, not only by the lane's queue.** The lane's queue-depth
  check is correct but downstream: a connection the HTTP server has accepted and
  not yet dispatched is invisible to it. Measured at 128 concurrent against 8
  slots, the guarded queue peaked at 31 of 64 while 128 requests were
  outstanding, and **288 of 464 clients timed out having received no 503 and no
  `Retry-After`** — the promise above did not hold under exactly the load it
  exists for. Requests are now counted where they arrive and refused there, which
  took timeouts from 62 percent to 0.004 percent with `Retry-After` on every
  refusal. `MTPLX_SERVER_MAX_INFLIGHT` overrides the limit; it defaults to the
  lane's own capacity (`cohort_slots + max_queue_depth`), and `0` disables the
  gate. `/health`, `/metrics` and `/v1/models` are never refused, because losing
  observability while shedding load is how a busy server becomes an opaque one.
- **`live_cohort.health`** is how you tell stuck from busy. Not
  `seconds_since_progress` on its own — a healthy cohort goes quiet while a
  joining request's prompt is read, and that happens more often now.
- **64 KiB per token per row** — size width and context together.
- **`rows_peak` above `last_real_width`** is continuous batching working;
  `cohort_resizes_total` stuck at zero on a busy server is it not working.
- **Prefix reuse across turns is available and OFF by default** —
  `MTPLX_DENSE_BATCH_PREFIX_CACHE_BYTES`. ~91% of a prompt reused from turn two
  on, answers unchanged, at real memory cost. Watch `prefix_cache.hit_rate`.
- **A wall-clock improvement is not evidence this lane ran.** The solo path has
  its own reuse. Check `requests_served_total`.
- **Penalties work now** — they were being ignored before.
