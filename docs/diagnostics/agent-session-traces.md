# Diagnose an agent session

Use the engine's request receipts, flight recorder and the actual client history together. A successful request is not proof that a coding task finished correctly: run the generated artifact and exercise follow-up edits.

## Capture an isolated run

Give each baseline and candidate its own runtime, port and SSD-cache directory. Keep model files and native sampler settings identical. Record the commit, imported package path, model revision, memory limits and hardware. Verify thermal conditions before performance measurements; run one model at a time and wait for its memory to drain before switching versions.

```sh
mtplx serve --model /path/to/model --port 8211 \
  --ssd-session-cache-dir /path/to/run/ssd-bank \
  --request-log-jsonl /path/to/run/requests.jsonl \
  --flight-recorder /path/to/run/flight.jsonl
```

Point OpenCode or Pi at that port using the normal MTPLX integration. Let coding tasks stop naturally. Preserve both the client transcript and the generated project. To capture exact encoded token IDs for a local reproduction, set `MTPLX_REQUEST_CAPTURE_DIR` before starting the server. Captures contain prompt content; keep them local unless you intend to share that content.

## Open the joined view

```sh
mtplx trace report SESSION_ID --port 8211 \
  --db /path/to/opencode.db \
  --request-log /path/to/run/requests.jsonl \
  --flight-log /path/to/run/flight.jsonl \
  --out /path/to/run/report.html

mtplx trace report --pi-session /path/to/pi-session.jsonl \
  --request-log /path/to/run/requests.jsonl \
  --flight-log /path/to/run/flight.jsonl \
  --out /path/to/run/report.html
```

The report includes cache reuse, TTFT, decode curves, draft-position acceptance, tools, a request selector, an interval scrubber, and exportable evidence JSON. `mtplx trace request REQUEST_ID --json` exposes the same counters for scripts; use the same explicit log paths. Rotation files are read oldest first.

Identity matters. Flight events join to receipts by exact server request ID. Updated Pi integration supplies the preceding transcript entry ID; a unique matching receipt links directly to its response. OpenCode supplies the user-turn ID, which can contain several engine/tool steps. Older transcripts and ambiguous retries still require time/token matching, and are labelled accordingly. Unmatched receipts are retained in the evidence export. Pi readers follow the active branch rather than counting abandoned history as completed work.

## Decide whether MTP pays

Measure AR on the same model, hardware, context and workload. Pass that measured rate using `--ar-tok-s`; the tool never substitutes the CLI's historical 40 TPS default.

Let `G` be tokens actually delivered, `N` completed speculative rounds, `T` full decode wall time and `t_AR` seconds per AR token. The observed speedup is:

`speedup = (G / T) * t_AR`

The acceptance threshold holds the recorded cycle cost fixed. Re-measure when context, hardware, route or acceptance changes: rejected drafts also change repair cost.

MTP helps when `T/N < (G/N) * t_AR`. Include drafting, verification, commit/repair, host overhead and stalls in the full cycle; verify time alone is insufficient.

For fixed depth `D`, away from start/end boundaries and copy routes, aggregate acceptance `q = accepted draft tokens / all proposed draft tokens` gives approximately `G/N = 1 + D*q`. Thus `q_break_even = (cycle_time/t_AR - 1) / D`. At D3, acceptance of 20%, 30%, 40% and 50% buys approximately 1.6, 1.9, 2.2 and 2.5 tokens per round. Each is worthwhile only when its measured round costs less than that many AR steps.

Conditional per-position acceptance is different: expected output is `1 + p1 + p1*p2 + p1*p2*p3`. Do not multiply the already-unconditional per-depth receipt rates again. Mixed depths and context-copy routes invalidate the fixed-depth shortcut; use actual delivered tokens and total time.

## Interpret gaps and memory

Flight samples are progress-driven at approximately one-second intervals. Missing samples are missing observations, not fabricated zero-throughput intervals. Counter resets produce unknown deltas. Some recorded GPU timing spans overlap; summing them does not necessarily recover exclusive wall time.

Allocator active bytes, free allocator-cache bytes, bank logical bytes, process RSS and physical system pressure measure different things. Shared cache references must not be counted twice. Releasing a logical bank entry is not proof that physical memory was freed: re-read allocator counters. `cache_cleared` records the allocator cache, not deletion of the useful prompt prefix.

Test anonymous clients both with and without reasoning history, multiple distinct conversations, SSD partial restores, long outputs, edited prefixes, cancellation/retry and normal background apps. Keep the historical baseline and candidate evidence; never infer cross-chip speedups from an M5 fallback run.

## Repeat the same request at AR, D1, D2 and D3

`mtplx tune` measures its selected calibration suite. Its effective per-case
budget is the smaller of `--max-tokens` and the suite's `max_tokens`; the
bundled warm-coding case uses 512. Raw rows record `prompt_tokens` and
`token_budget`. To test a larger calibration window, supply a custom suite
with that budget and raise the command's bound. Use the uncapped request
replay below and real clients to evaluate long coding sessions.

```sh
python scripts/mtp_cost_probe.py request.json \
  --request-log /path/to/run/requests.jsonl --out /path/to/new/probe \
  --base-url http://127.0.0.1:8211 --repeats 2
```

Supply a normal chat-completion request JSON with the actual coding task and native sampler. This tool reverses depth order on alternate rounds, verifies actual fan ramp before each request, checks that the server honored the mode/depth, retains full timestamped streams and exact receipts, and computes economics against measured AR. It introduces no output or reasoning cap. Tool calls are recorded as model output; use the real client to execute and validate a whole task.

For a candidate replay, `--seed-map seeds.json` accepts the baseline's resolved seeds keyed by run label, for example `{"r1-d0": 123, "r1-d3": 456}`. Exact request captures record the resolved seed in `outcome.resolved_seed`. Preserve the request hash and every sampler setting alongside the seeds. Matching seeds reduce one source of variation; different speculative routes can still consume randomness differently.

A repetition-guard stop can arrive with client `finish_reason: "stop"`. The probe retains that sample but exits unsuccessfully and excludes its throughput from MTP economics. A guard pass is not a code-quality pass: run the produced code, its assertions and the actual client follow-up before accepting a performance result.

Proposal cycles come from the first `drafted_by_depth` counter. Verifier forwards
also include copy and bonus work and are not a cycle count. `mtp_pays` compares
actual delivered throughput with the supplied matched AR measurement. The
acceptance threshold uses `(decode_time * AR_TPS - non_draft_output) / proposals`;
it holds the observed full cost, non-draft output and depth mix constant. Copy
and adaptive runs remain usable, but this is a conditional estimate, not a
measurement of performance at another acceptance rate.
