# 009 — RAMP is a copy accelerator, not an agent accelerator: no benefit on open-ended work

**Date:** 2026-08-26
**Phase:** 4D (RAMP)
**Gate:** none tripped — this record answers the *scope* question decision 008
left open under *What this does not license*: **"One task, one prompt. A single
coding-agent shape is not the workload."**
**Outcome:** **RAMP delivers no benefit on open-ended agentic work.** On a real
code review and a real multi-file explanation, RAMP's exact n-gram index goes
**completely dark — zero exact hits in 1 544 probes** — and the fuzzy fallback
that replaces it drafts **6.7× more tokens to accept fewer**. Measured decode
t/s lands between **−2.0 % and +1.1 %** of the stock ladder in the clean round
(**−6.2 % / −4.5 %** in the noisier one), against **+13.2 %** on the mechanical
control cell measured on the same machine in the same round.

**RAMP's off-by-default status is doing real work and must stay.** Its value is
scoped to editing/repetitive tasks, not to coding-agent sessions generally.

**Second finding, unplanned and more consequential than the first:** decision
008's prime-directive result — *"RAMP changes the execution path and not one
byte of temperature-0 output"* — **does not generalise off the mechanical task
shape.** On both open-ended cells the two arms produce **different
temperature-0 output**, deterministically, reproducibly, in two independent
rounds. Chasing that down turned up a broader problem: **engine session history
moves the output too, within the stock arm alone.** Temperature-0 output
identity is only a meaningful gate on copy-shaped tasks. See *The
prime-directive gate*.

**Evidence:**
- `results/ramp_openended_ab_round2.json` — **the result of record** (clean round: zero pageouts, every cell spread ≤ 1.8 %)
- `results/ramp_openended_ab_round0.json` — the first complete round, noisier, committed deliberately
- `docs/reviews/2026-08-26-ramp-openended/evidence/openended-ab-rows.jsonl` — every per-request row, all rounds, including the excluded one
- `docs/reviews/2026-08-26-ramp-openended/evidence/prompt-manifest.json` — the 11 files and token counts, so the prompt is rebuildable
- `docs/reviews/2026-08-26-ramp-openended/evidence/output-*.txt` — full generated text per cell per arm, for the divergence analysis

**Harness:** `scripts/ramp_openended_corpus.py`, `scripts/ramp_openended_ab.py`,
`scripts/ramp_openended_ab.sh`, `scripts/ramp_openended_summarize.py`,
`scripts/ramp_openended_capture.py`.

---

## Why this record exists

Every RAMP number in this project — the ~800-token POC
(`docs/reviews/2026-08-26-ramp/evidence/ab-bench.json`), the block sweep,
decision 007's projections, and decision 008's 128K ground truth — was measured
on **one task shape**: *reproduce a file that is present in the context, with a
mechanical edit applied*. Rename an identifier, add a method, edit a docstring.

That is precisely the shape a prompt-lookup proposer is built to exploit:
almost every emitted token already exists verbatim in the prompt, so a copy
mechanism can find it. Decision 008 measured RAMP's copy path supplying **84.7 %
of all emitted tokens** on that task. Whatever such a benchmark measures, it is
not a coding-agent session — most agent turns are reviews, explanations,
investigations, and new code, where correct output must be *generated*.

Decision 008 flagged the gap in its own words and did not close it. This closes
it.

## What was run

**One variable changed: task shape** (CLAUDE.md rule 3). Three cells share a
**byte-identical 23 843-token context prefix** built from 11 real tracked
repository files (`prefix sha256 06ae381ebe64dbd1…`, commit `6b294fd`), and
differ *only* in the instruction appended after it:

| cell | shape | the ask |
|---|---|---|
| `mechanical-control` | mechanical | reproduce `rafale/draft/ngram.py` with a rename applied — the shape every prior RAMP number was measured on |
| `code-review` | open-ended | review `rafale/bench/` as a senior engineer on a PR: real findings, prose, specific suggestions |
| `explain-synthesis` | open-ended | explain how `rafale/draft/ramp.py` substitutes the engine's proposer with no engine diff, synthesising across several files |

The open-ended cells were checked for degenerate output before the matrix was
trusted: both produce substantive, on-topic technical prose that appears nowhere
in the prompt (`evidence/output-*.txt`). They are real work, not echo.

**The control cell is the load-bearing part of the design.** A null result on
the open-ended cells is uninterpretable on its own — a broken rig looks exactly
like "RAMP does not help". The control is what separates the two, and it is
asserted in the summariser rather than checked by eye.

- Both arms: `temperature = 0`, `max_tokens = 1536`, decision 001's canonical
  launch line; the RAMP arm differs only by `rafale.draft.ramp.install()` with
  `block=48, fuzzy=True`, `--port`, and the read-only counter sidecar.
- **~24K context, not 128K.** Decision 008 already settled context length. This
  run varies shape and holds context at a realistic agent-turn size, which is
  what makes a 3-cell × 2-arm × multi-round matrix affordable.
- Server-level alternation, launch order flipped per round, as decision 008.
- **Two complete rounds, 24 requests each, 48 in total.** (A third, round 1, is
  excluded — see *Hygiene*.)

### Why temperature 0

The task brief allowed nonzero temperature for realism. Temperature 0 was chosen
anyway, for three reasons, and the choice turned out to matter:

1. It is the only setting under which "RAMP changed the output" is a *fact*
   rather than sampling noise (CLAUDE.md rule 2) — and the output did change.
2. It removes sampling variance from a comparison whose effect size (~2 %) is
   small enough for noise to swamp entirely.
3. It keeps this record commensurable with every prior RAMP measurement.

The cost is stated in *What this does not license*.

## The numbers — round 2, the clean round

| cell | arm | n | median t/s | spread | ccopy rounds | drafted | accepted | block acc. | copy share of output | susp |
|---|---|---|---|---|---|---|---|---|---|---|
| `mechanical-control` | stock | 4 | **102.70** | 1.2 % | 120 | 2 608 | 2 320 | 0.890 | **84.7 %** | 0 |
| `mechanical-control` | RAMP | 4 | **116.30** | 0.9 % | 80 | 3 840 | 2 276 | 0.593 | **83.1 %** | 4 |
| `code-review` | stock | 4 | **31.95** | 1.8 % | 52 | 528 | 124 | 0.235 | **2.0 %** | 12 |
| `code-review` | RAMP | 4 | **31.31** | 0.7 % | 76 | 3 564 | 88 | **0.025** | **1.4 %** | 16 |
| `explain-synthesis` | stock | 4 | **31.23** | 0.1 % | 28 | 352 | 72 | 0.205 | **1.3 %** | 4 |
| `explain-synthesis` | RAMP | 4 | **31.59** | 0.2 % | 80 | 3 840 | 172 | **0.045** | **3.1 %** | 20 |

| cell | shape | round 2 (clean) | round 0 |
|---|---|---|---|
| `mechanical-control` | mechanical | **+13.2 %** | +11.2 % |
| `code-review` | open-ended | **−2.0 %** | −6.2 % |
| `explain-synthesis` | open-ended | **+1.1 %** | −4.5 % |
| | | **shape gap: 13.7 pts** | 16.5 pts |

**Read this honestly: the open-ended result is *no benefit*, not *reliable
harm*.** The two rounds disagree on sign (−6.2 %/−4.5 % vs −2.0 %/+1.1 %) and
agree on magnitude class: RAMP's effect on open-ended work is somewhere between
a few percent down and one percent up. What both rounds agree on decisively is
the **contrast** — the same RAMP build, on the same machine, in the same round,
against the same stock server, is worth +11 % to +13 % on the mechanical cell
and nothing at all on the open-ended ones.

Round 2 is the result of record on the project's own criterion (decision 008
chose its round on spread): round 2 has **zero pageouts and every cell spread
≤ 1.8 %**; round 0's RAMP arm carried 592 pageouts and a **10.1 %** spread on
`code-review`.

## The dark-fraction finding

This is the mechanism, and unlike the throughput deltas it is **bit-identical in
both rounds** — temperature-0 retrieval is fully deterministic, so these
counters are not a sample, they are the behaviour.

The engine's own `context_copy_*` telemetry counts rounds in which a proposal
*was made*; a probe that finds nothing leaves no engine trace at all. So the
launcher gained an opt-in read-only counter sidecar (`RAMP_COUNTERS_PORT`, unset
by default) exposing RAMP's own `probes / exact_hits / fuzzy_hits / misses`.

| cell | probes | **exact hits** | fuzzy hits | misses | **dark fraction** |
|---|---|---|---|---|---|
| `mechanical-control` | 124 | **40** | 40 | 44 | **0.355** |
| `code-review` | 988 | **0** | 76 | 912 | **0.923** |
| `explain-synthesis` | 556 | **0** | 80 | 476 | **0.856** |

**Zero exact n-gram hits in 1 544 probes across both open-ended cells.** Not
"few" — none. The exact `ng_min`-gram key, which *is* the engine's actual
context-copy mechanism and the part RAMP inherits unchanged by subclassing the
engine's `NgramIndex`, never once fired on genuinely new prose. Every proposal
RAMP made on open-ended work came from its own fuzzy short-anchor fallback, and
that fallback lands at **2.5 %–4.5 % block acceptance**.

The cost side follows directly:

- On `code-review`, RAMP drafted **3 564 tokens to accept 88**. The stock ladder
  drafted **528 to accept 124**. RAMP spent **6.7× the draft width to accept 29 %
  fewer tokens.** Every rejected token is verify-pass width paid for and thrown
  away — the exact trade decision 008 credited for the +45.9 %, running
  backwards.
- The copy path supplies **1.3 %–3.1 % of emitted tokens** on open-ended work, in
  *either* arm. **Even a perfect proposer could only touch ~2 % of the output.**
  There is no version of this mechanism that matters on this workload — which is
  also why the measured effect is ≈ 0 rather than strongly negative: the whole
  mechanism, cost and benefit together, is operating on a rounding error.
- The reason the effect is not *more* negative is the engine's own guard, below.

**The EMA-suspend guard fires constantly here.** Decision 008 reported *0
suspensions in all 23 requests* at 128K and called the decision-006 hazard
"confirmed dormant". It is not dormant; it was dormant *on the mechanical task
shape*. On open-ended work the stock ladder suspends 4–12 times per 4 requests
and RAMP 16–20. The guard is detecting exactly the low-acceptance condition
documented above and shutting the proposer off, which is what limits the damage.
Decision 008's "dormant" reading must be re-scoped to the mechanical shape.

## The prime-directive gate

**MIXED — and this is the finding that should worry a reader most.**

| cell | stock sha256 | RAMP sha256 | identical? |
|---|---|---|---|
| `mechanical-control` | `b5e79a01…` | `b5e79a01…` | **yes** |
| `code-review` | `f05d60ed…` | `4fb15d3f…` | **NO** |
| `explain-synthesis` | `dc598100…` | `f7183ba1…` | **NO** |

Each arm is **perfectly deterministic in itself** — the same sha256 in all
requests, across separate server launches and two independent rounds. This is
not flakiness. The two arms deterministically produce *different* temperature-0
output on open-ended prompts, and identical output on the mechanical one.

The mechanical control reproduces decision 008's `b5e79a01…` exactly, which is
what proves the rig is faithful. So decision 008's byte-identity claim is not
wrong — it is **narrower than it reads**. It is a property of a task whose
output is 85 % copied text, not a property of RAMP.

### How different, measured

Both arms were re-captured in full on freshly-launched servers replaying the
matrix's exact request sequence. **All six sha256s reproduced the matrix
exactly** — `b5e79a01…` / `b5e79a01…`, `f05d60ed…` / `4fb15d3f…`, `dc598100…` /
`f7183ba1…` — so the divergence is reproducible on demand, not an artefact of
one run.

| cell | identical prefix | word-level similarity | length (stock → RAMP) |
|---|---|---|---|
| `mechanical-control` | **whole file** | **1.000** | 2 757 → 2 757 chars |
| `code-review` | 1 131 chars (19.8 %) | **0.292** | 6 178 → 5 707 chars |
| `explain-synthesis` | 751 chars (12.2 %) | **0.391** | 6 148 → 6 189 chars |

**The divergence is substantive, not cosmetic.** The two arms agree for roughly
the first kilobyte, then produce materially different reviews — different
findings in a different order, ~30–39 % word-level overlap. Both remain coherent,
on-topic and plausible; **no quality judgement is made here**, because judging
which review is better is exactly what the `rafale/quality/` suite exists for and
it was not run. What is established is that the texts are *different documents*,
not the same document with a comma moved.

### The divergence is not RAMP's alone — and that is the important qualifier

The full-text capture (`scripts/ramp_openended_capture.py`) turned up a second
instability while chasing the first, and it changes how the table above should
be read.

Capturing `code-review` from a **stock** server whose request history differed
from the matrix's — same binary, same launch line, same prompt, same
`temperature = 0` — produced `77970fee…`, **not** the matrix's `f05d60ed…`. It
reproduced `77970fee…` again on a warm repeat, and again with the matrix's cell
order replayed onto that already-used server. Only when a **fresh** server
replayed the matrix's exact sequence — `mechanical-control` cold first, then
`code-review` — did `f05d60ed…` return, along with `dc598100…` and `b5e79a01…`.
All three matrix shas then reproduced exactly.

Two conclusions, and they pull in opposite directions:

1. **The matrix's arm-vs-arm comparison is sound.** Both arms ran identical
   request sequences on freshly-launched servers, each arm self-consistent
   across 2–3 independent launches, and the sequence is reproducible on demand.
   The stock-vs-RAMP divergence is a real, controlled result.
2. **But RAMP is not the only perturbation that moves this output.** Engine
   session/cache history moves it too, within the stock arm alone. So the honest
   framing is not *"RAMP corrupts output"* — it is that **this engine's
   temperature-0 output on open-ended prompts is not stable against
   execution-path perturbation of any kind.** RAMP is one such perturbation;
   request history is another. On the mechanical task, none of them move it.

That is a weaker claim about RAMP and a stronger, more uncomfortable claim about
the measurement regime: **temperature-0 output identity is only a meaningful
gate on copy-shaped tasks.** On open-ended prompts it is not a stable property to
gate on at all, and CLAUDE.md rule 2's "pinned seeds at temperature 0" hygiene
does not, by itself, deliver reproducibility here.

The most plausible mechanism, stated as a hypothesis and **not verified here**:
the launch line runs `--mtp-batch-numerics throughput`, and RAMP changes the
verify batch's *width* (block 48 vs the 8–32 ladder). Different reduction shapes
give different floating-point rounding, which flips the argmax on near-ties. A
near-verbatim copy has almost no near-ties; open prose has many. This does not
violate CLAUDE.md rule 7 — the target still verifies every token, and no
unverified speculative token reached output — but it does mean **RAMP is not
output-neutral in general**, and any future claim that it is must be re-measured
per task shape.

## Decision

1. **RAMP's default stays off, and the reason is now positive rather than
   precautionary.** Prior records kept it off for want of evidence. This one
   shows measured absence of benefit on the workload class the project targets.
2. **RAMP's claimed scope is narrowed to copy-shaped work.** Any future citation
   of +45.9 % (decision 008) or +51.3 % (the POC) must carry the qualifier
   *"on mechanical single-file edit tasks"*. Those numbers do not describe a
   coding-agent session.
3. **A default-flip card is out of reach on general agent work.** Decision 008's
   *Handed forward* item 1 asked for multi-case coverage before such a card;
   this is that coverage, and it answers no. A *task-shape-gated* RAMP — on for
   bulk edit/refactor turns, off otherwise — is the only form worth proposing,
   and it is not proposed here.
4. **Decision 008's "EMA guard confirmed dormant" is scoped to the mechanical
   shape.** The decision-006 hazard is live on open-ended work.
5. **Decision 008's prime-directive result is scoped to the mechanical shape.**
   This is the correction that should propagate furthest.
6. **Temperature-0 byte identity is retired as a general quality gate for
   open-ended prompts on this engine.** It remains valid and useful on
   copy-shaped tasks, where it is stable against both RAMP and session history.
   On open-ended prompts it is not stable against either, so a future quality
   gate for agentic workloads needs a distributional measure (the KL / top-k
   comparison `rafale/quality/` is already scoped for) rather than a hash
   comparison. Any gate that hashes open-ended output will fail for reasons
   unrelated to the change under test.

## What this does not license

- **"RAMP hurts" is not supported.** Two rounds, opposite signs, small
  magnitudes. The supported claim is *no benefit*.
- **Two open-ended cells is not "open-ended work".** Both are read-and-explain
  tasks over a fixed context. Not tested: multi-turn tool-calling loops, writing
  substantially new code, or long agentic sessions.
- **Nothing about the append-only agent-harness shape.** Still the real target
  workload, still unmeasured (decision 008 says the same). Every prompt here is
  fixed and re-sent; a growing context could plausibly restore exact hits, since
  the model's *own* recent output would enter the corpus. Note the POC already
  measured "wide" corpus indexing at **−0.7 %**, which argues against optimism.
- **Nothing about block lengths other than 48**, or about `fuzzy=False`.
- **Temperature 0 only.** At realistic sampling temperatures the output
  divergence above would be invisible (both arms would differ anyway) and the
  throughput gap would need far more samples to resolve.
- **~24K only.** The control's **+11.2 % / +13.2 %** here is far below the
  **+51.3 %** measured at ~800 tokens and the **+45.9 %** at 128K. RAMP's
  advantage on its *own best task* is therefore **not monotonic in context
  length**, which no model in this project predicts. That is an unexplained
  observation, recorded and not theorised — decision 008 item 2 forbids
  inferring anything further from `ramp_kernel_regimes.py` or
  `ramp_longcontext_model.py`, and this record does not.
- **No power or bandwidth counters.** `powermetrics` still needs root and this
  session still has no passwordless `sudo`. The operator run decision 008 asks
  for is still owed.

## Hygiene

- **High Power Mode confirmed** (`pmset -g` → `powermode 2`), AC power.
- **Paging rate, not occupancy** — decision 008's *Recommended for the harness
  generally*, now implemented: `ramp_openended_ab.py` records `Pageouts`,
  `Swapouts`, and `Pageins` before and after **every** request, and the
  summariser differences them. Swap **occupancy** sat at 7 839–7 879 MB of
  9 216 MB (~85 %) throughout — the signature that would have wrongly aborted
  this run on an occupancy check. Measured **rate**: **zero swapouts in every
  request of every round.** Pageouts: **592 total in round 0's RAMP arm**
  (≈ 10 MB across 12 requests) and **zero everywhere else, including all of
  round 2**. That asymmetry is one of the two reasons round 2 is the result of
  record; the other is spread.
- **The control validates the rig against prior records.** The stock ladder's
  telemetry on `mechanical-control` is **30 rounds / 652 drafted / 580 accepted
  per request**, byte-identical to the figures decision 008 records at 800
  tokens, 32K and 128K, in **all 12 baseline control requests of all three
  rounds**; the output sha256 `b5e79a01…` matches decision 008 too. The rig
  reproduces three prior context scales before it is asked to say anything new.
- **Warm and cold never pooled.** The first request of each cell is a cold ~24K
  prefill (~40 s wall); the rest hit prefix reuse (`cached_tokens = 23 808/23 945`,
  ~7 s wall). The summariser separates them and the reported figures pool only
  like with like.
- **Round 1 is excluded, and why is recorded rather than hidden.** Its RAMP
  server was terminated mid-arm during `code-review` rep 2 — the harness saw
  `http.client.RemoteDisconnected` and the server log ends with
  `Cancel 1 running task(s), timeout graceful shutdown exceeded`, i.e. it
  received a shutdown signal from outside the harness. **The cause was not
  identified.** Round 1's baseline arm completed and matches round 0's and round
  2's baselines closely, and the partial RAMP rows it did produce are
  bit-identical in telemetry to round 0's and round 2's; nothing about it
  contradicts the result. It is excluded because it is incomplete, not because
  it disagreed, and its rows remain in the committed JSONL so the exclusion can
  be checked by anyone.
- **`make lint && make test` clean.**

## What would reverse this

- Exact hits appearing at a material rate on an **append-only** agent context,
  where the model's own prior output is in the corpus. This is the single most
  plausible reversal and the honest gap in this record.
- A real agent trace whose emitted tokens are copy-heavy for reasons other than
  file reproduction (large diffs, bulk refactors, code moves). RAMP would help
  there, and this record does not claim otherwise.
- A task shape where the copy path carries materially more than ~3 % of emitted
  tokens while still being "open-ended". None was found here, and the 1.3–3.1 %
  figure is measured in *both* arms, so it is a property of the workload rather
  than of RAMP.
- An MLX or MTPLX release changing the verify-pass kernel or the context-copy
  proposer contract.

## Handed forward

1. **The `--mtp-batch-numerics` divergence hypothesis is unverified.** Testing it
   is cheap — run the RAMP arm at `block=32` (inside the stock ladder's width)
   and see whether byte-identity returns. Worth doing before anyone relies on
   RAMP being output-neutral. The session-history result above makes the same
   hypothesis (batch/reduction shape drives argmax flips on near-ties) explain
   both instabilities, which is a point in its favour and still not evidence.
2. **The session-history instability deserves its own bisect.** Which state —
   prefix-cache residency, first-request-cold-vs-warm, `--retrieval-max-resident`
   eviction — moves the output? This record establishes *that* it moves and
   pins the reproducible sequence; it does not isolate the cause.
3. **The append-only agent-harness shape remains the unmeasured real target**,
   now for the second decision record running.
4. **The paging-rate check asked for in decision 008 item 5 is implemented in
   this harness** (`ramp_openended_ab.py`) but is **not yet in
   `rafale/bench/hygiene.py`**, which is where decision 008 asked for it.
5. **The operator `powermetrics` run** is still owed, from decisions 007 and 008.
