# Changelog

All notable user-facing changes to MTPLX. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [2.5.2] - 2026-08-04

Hotfix for a long-response slowdown that shipped in 2.5.1 with Optimized
Speed V2 as the recommended coding model.

### Fixed

- Long single responses no longer slow down and stutter partway through. The
  compiled verifier hands generation to the eager path after its growth
  reserve is exhausted, and that handoff carried unfinished GPU work into the
  rest of the response. Every later token paid for the old work, so decode
  throughput fell steadily on responses past a few thousand tokens. The
  reported 12,000-token response opened at 59 tokens per second and
  ended near 30. With the fix, the same seeded generation no longer
  decays: the closing windows now run as fast as or faster than the
  early ones.
- The handoff now settles all cache and recurrent state exactly once at the
  ownership boundary and releases the compiled references before eager
  decoding continues. Handoff telemetry is exposed in the compiled verifier
  stats.

### Changed

- The minimum MLX version is now 0.32. Fresh installs already resolved MLX
  0.32.0; existing runtime environments could stay on 0.31.2 indefinitely
  because dependency upgrades only run when the declared floor requires
  them. Long-generation runs consistently read faster on the 0.32 stack,
  and existing installs now converge to what new installs already run.

### Compatibility

- Model catalogs, defaults, sampler settings, memory policy, and the 2.5.1
  V2 recommendation are unchanged.

## [2.5.1] - 2026-08-03

Optimized Speed V2 is now the recommended Qwen 3.6 27B model for coding on
modern Macs with enough unified memory. It gives users a much higher-quality
coding model while keeping the original, smaller Optimized Speed model one row
below it.

### Added

- Optimized Speed V2 is a first-class model in onboarding, the app catalog,
  CLI defaults, quickstart, downloads, served model identity, and OpenCode
  setup.
- The model is published as a dynamic 4-bit hybrid quant with hand-tuned
  sensitive parts kept at up to 16-bit. It is faster on longer agent tasks,
  slightly larger, and a little slower for short chats.
- RAM-aware defaults recommend V2 first on modern Macs with at least 32 GiB of
  detected memory. Smaller Macs keep the existing 9B and 4B recommendations.

### Compatibility

- The original Optimized Speed model remains fully supported and appears
  directly below V2 wherever the machine can run both.
- Runtime kernels, sampler defaults, cache behavior, and speculative depth are
  unchanged from 2.5.0.

## [2.5.0] - 2026-08-03

The next-model release: MTPLX can load new architecture aliases without a
hard-coded model-type wait, multi-layer MTP drafts are supported for the
upcoming Qwen generation, HY3 becomes a first-class serving target, and the
coding-agent bridge is materially more reliable. Experimental DeepSeek V4,
Laguna, and GDN speed lanes from David Tai land behind explicit opt-ins; the
shipping defaults remain unchanged.

### Added

- Upcoming-Qwen architecture readiness: the native draft path now supports
  `mtp_num_hidden_layers = N`, and unknown `model_type` values can resolve
  through their declared architecture class. A synthetic `qwen3_8` alias
  drill exercised load, generate, CLI, and app discovery before public weights
  existed; this is compatibility preparation, not a claim that unreleased
  weights were benchmarked.
- HY3 first-class serving: a vendored MTP-capable model class, official
  defaults, suffixed think-tag handling, model discovery, and native OpenCode
  tool calls. AR-only exports fail safely to the AR path rather than touching
  an uninitialized draft head.
- Experimental DeepSeek-V4 shape-specialized verification lanes: prebound
  output-LoRA routes, adaptive speculative width, exact M3 attention
  projection, sinkhorn and attention-island kernels, and compiled
  post-attention verifier islands. On the 128 GB M5 Max release machine, the
  exact 2-bit DQ model plus official MTP shard measured about 31 AR tok/s and
  36 MTP tok/s versus about 4/6 tok/s on the conservative path. These routes
  remain opt-in while broader model-quality validation continues. Contributed
  by David Tai (@davidtai, #223).
- Experimental Laguna S-2.1 `mlx.fast` ports covering decode and prefill
  kernels, including size-gated prefill MoE combine; fail-loud guards and
  portable scratch space keep unsupported shapes on the safe route.
  Contributed by David Tai (@davidtai, #222).
- Experimental GDN headquarter execution layout for verify tape capture,
  env-gated with bit-exact coverage and loud fallback. Contributed by David
  Tai (@davidtai, #209).

### Fixed

- Coding-agent JSON tool calls can carry large write bodies without the hidden
  tool guard aborting them, and common near-miss argument keys are repaired at
  the protocol boundary. This was verified through real OpenCode CLI and
  Desktop sessions, including a multi-file edit with all generated tests
  passing.
- App and CLI launches now share the same coding-agent engine environment, so
  a workflow does not silently change behavior depending on which Start button
  launched it.
- `start` and `quickstart --dry-run` report the resolved profile instead of the
  parser's placeholder default.
- Smart-fan mode holds through the post-generation heat-soak window before
  restoring automatic control, avoiding the early restore that could distort
  back-to-back performance runs (#227).

### Compatibility

- `transformers` 5.14 is allowed after tokenizer and tool-template parity were
  verified; the incompatible 5.13.0 release remains excluded (#175).
- No speculative-depth, cache, sampler, or model-speed default changed in this
  release. The opt-in model kernels fail closed to established implementations.

## [2.4.2] - 2026-08-02

The agentic-cache release: the session cache stops losing warm state
mid-run, tool-turn commits stop being ghosts, every serve keeps a durable
per-request trail by default, an experimental DeepSeek-V4-Flash backend
lands, and the documentation now matches the code everywhere it was
audited.

### Added

- DeepSeek-V4-Flash: experimental native AR backend
  (`model_type: deepseek_v4`) — Hyper-Connections, compressed sparse
  attention, hash-routed MoE, grouped output-LoRA — loading the
  mlx-community checkpoints directly, with an optional single-block MTP
  speculative lane when the checkpoint carries `mtp.0.*` weights
  (spec == AR gated; K=1-3 measured up to 2.28x on the 2bit-DQ build).
  MTP-declaring checkpoints that ship no draft weights degrade to AR
  with a clear message instead of failing at bind. Thanks @davidtai
  (#216).
- Request log, default on: every serve writes numeric/hash per-request
  telemetry to `~/.mtplx/logs/request-log-<port>.jsonl` (64 MB x 4
  rotation; no prompt or completion content; disable with
  `MTPLX_REQUEST_LOG_JSONL=off`). Pairs with 2.4.1's opt-in bit-exact
  request capture to make agent-session incidents diagnosable after the
  fact (#196/#197). New helpers: `scripts/gauntlet_scoreboard.py`
  per-session summarizer, `scripts/oc_tap.py` recording proxy and
  `scripts/oc_tap_diff.py` request-mutation analyzer for content-level
  wire truth.
- Session bank, active-session eviction protection: sessions that
  touched the bank within `MTPLX_SESSION_BANK_ACTIVE_PIN_TTL_S`
  (default 600 s) are eviction-last, so cross-session pressure evicts
  idle victims instead of the session that is mid-run.
- Session bank, newest-K per-session snapshot retention
  (`MTPLX_SESSION_BANK_PER_SESSION_MAX_ENTRIES`, default 3): divergent
  per-turn sibling snapshots no longer accumulate unreclaimed.
  `/health` now reports active sessions, the pin TTL, and recent
  evictions.
- Postcommit foreground grace (`MTPLX_POSTCOMMIT_FOREGROUND_GRACE_S`,
  default 2 s): a nearly-finished background cache commit lands instead
  of being preempted by the next fast agent-loop request.
- Session identity honors `x-session-affinity` / `x-session-id` request
  headers (OpenCode sends these per request), ending cross-request
  identity churn on that client.

### Fixed

- Tool-turn "ghost re-prefills": the tool-rewrite async commit rendered
  a canonical history that matched neither the generation nor the next
  prompt, burning full-history re-forwards (26.8 s observed) without
  ever storing. It is disabled pending a byte-proven canonical render
  (`MTPLX_IDLE_POSTCOMMIT_TOOL_REWRITE` re-enables);
  store-on-prefill and block salvage cover the lane.
- The bridge's convergence guard now states explicitly that editing and
  verification tools remain allowed and that its restriction covers
  only the current reply — a model read the old wording as a
  session-wide tool ban and stalled an entire session.
- `mtplx profile thermal`, `profile eval-attribution`,
  `profile dispatch --trace`, and `thermal fanmax-run` invoked
  research-workspace scripts that are not part of the distribution, and
  `--dry-run` printed those phantom paths as runnable commands. They
  now report availability honestly (exit 2, machine-readable
  `available: false`) and run the real script when present.
- `mtplx doctor`: Python floor corrected to 3.11 (matching
  `requires-python`); remediation texts no longer tell end users to
  edit source constants or to move a healthy server off its port;
  `--port` is documented and, when passed explicitly, aims the server
  connectivity checks.
- Session-bank near-prefix restores on backends with bounded rollback
  (DeepSeek-V4) pre-check `max_rollback` and fall back to a cold
  prefill instead of raising (#216).
- Help surfaces match their own parsers: the onboarding help no longer
  promises a Turbo wizard choice that does not exist (Turbo
  auto-selects for the quantized flagships), `--strict-cold` names the
  enforced 59 tok/s gate, `--open-dashboard` opens alongside the chosen
  client (as it always did), and the command reference teaches
  `mtplx <command> --help`, which also works for multi-word commands.

### Documentation

- Full truth sweep: ~450 documentation claims reconciled against the
  code across 27 files. Highlights: INSTALL.md no longer references an
  MLX fork removed in 2.0.0; turbo-verify.md no longer calls the
  shipped default "experimental, off by default" nor excludes the
  6-bit lane that ships; the Anthropic base-URL instruction (docs and
  the canonical example) no longer 404s; `/metrics` no longer claims a
  Prometheus mode that never existed; the README modes table shows
  Turbo as the default for the quantized 27B/9B flagships; the Laguna
  memory requirement states the real ~85.3 GiB preflight gate;
  version-era staleness ("v0.1", "preview", v0.3.x runbook pins) is
  cleared; historical release notes gain bracketed corrections where
  they documented commands that never worked. Thanks
  @PhilipJohnBasile for #218 (removed the unsupported MTP-sidecar
  graft guidance; seeded by #215).
- Dependency-record correction: the transformers pin has been
  `<5.14,!=5.13.0` since shortly after 2.0.0; the changelog never
  recorded the relaxation from `<5.13`.

### Dependencies

- pypa/gh-action-pypi-publish 1.14.1 -> 1.14.2 (#217).

## [2.4.1] - 2026-08-01

The smooth-streaming release: the app's chat render path is overhauled
(no more freeze-then-catch-up stutter, scroll bounce, or plain-text code
blocks — real syntax coloring, live code cards, tables, and actual math
notation), and the 2.4.0 short-turn regression is fixed.

### Added

- Live syntax coloring for code blocks (12 languages + generic) from a
  freeze-time lexer that colors each line exactly once; streaming cost
  is O(new text), never O(document).
- Streaming code card: an open fence renders as a live card with
  colored lines and flips once to its settled form at close.
- Pipe tables render as real tables; math renders as real notation
  (Unicode super/subscripts, stacked matrices and fractions, inline
  conversion instead of dollar-sign leaks).
- Typewriter pacing for streamed text with geometric catch-up and a
  hard drain bound (`MTPLX_STREAM_TYPEWRITER=0` to disable), and a live
  tok/s chip computed over a sliding ~5 s window.
- Performance mode is a true kill switch: plain text only, through both
  the streaming and settled render paths.
- Opt-in per-request capture for bit-exact failure replay:
  `MTPLX_REQUEST_CAPTURE_DIR=<dir>` persists each request's
  reproduction envelope at dispatch time (#196/#197, third layer).
- Opt-in frontend stream-performance probe (`MTPLX_UI_PERF=1`, HUD via
  `MTPLX_UI_PERF_HUD=1`) with a per-turn JSONL trace joinable to engine
  stats by request id.
- Experimental: cost-model speculative-depth policy
  (`--adaptive-policy cost`) and blocked-sequential GDN prefill
  (`MTPLX_GDN_BLOCKED_PREFILL=1`). Defaults unchanged.

### Fixed

- 2.4.0 short-turn regression: the compiled-verify path could reserve
  KV budget above the configured ceiling, taxing short requests with
  setup work they never used; the reserve is now clamped.
- Warming prefills yield to real traffic within one small chunk instead
  of delaying a freshly arrived request.
- Derivative model artifacts whose names extend a first-party model
  name are served under their own id, not the flagship's — the health
  payload, OpenAI `model` field, and app model chip now report the
  artifact actually loaded.
- Streaming render: line-segment coalescing keeps realized view count
  bounded on long answers; the bottom-pin scroll correction runs in the
  same display cycle as layout so the streaming bubble can no longer
  visibly bounce; a per-display-cycle window-sizing walk that floored
  every update at ~50 ms is removed (`MTPLX_APP_SIZING_TUNER=0`
  restores it).

## [2.4.0] - 2026-07-31

The 35B speed release: the 35B-A3B MoE gets a compiled decode stack and
continuous batched serving, the 2.3.0 fan regression is root-caused and
fixed, tool calling gets another round of contract hardening, and
structured output can no longer be eaten by an unbounded reasoning
prelude. Four community contributors landed code in this release.

### Added

- 35B-A3B compiled decode stack: target-prefix compiled route, whole-MoE
  fusion, GDN post-conv fusion, and a row-owned router (David Tai, #174).
- Continuous batched serving for the A3B lane: fixed-shape cohorts,
  ragged KV, fold-in repair, and AR row-packing (David Tai, #200).
- Laguna S-2.1 support (exact-pin oQ4e, AR-only) with an app catalog
  entry, plus a Poolside `arg_key`/`arg_value` tool-call dialect parser
  (David Tai, #195).
- Hy3 295B full-residency lane and generic MTP draft-contract hardening
  whose loud recurrent-cache failure also caught a real bug on the Qwen
  lane (David Tai, #208).
- `/health` now reports smart-fan restore state (`restore_verified`,
  `restore_failures`, `stale_leases_reconciled`) so stuck-fan reports
  are diagnosable from the field (#201).

### Fixed

- Fans no longer stay pinned at max after a request ends (#201). A
  failed fan restore was logged once and then treated as restored, so
  the hardware stayed ramped while the server believed it was clean;
  restores now verify the fan rows are back on the Apple auto curve and
  retry with backoff until they are. The ThermalForge daemon-socket
  restore path no longer trusts the daemon's "ok" reply without
  verifying, and falls back to the CLI in the same call. A stale-lease
  watchdog drops any fan lease held while the engine has been
  continuously idle (default 120s, `MTPLX_SMART_FAN_STALE_LEASE_S`).
- A generation cut by `max_tokens` mid-tool-call now reports
  `finish_reason: "length"` instead of `"tool_calls"`, so agent clients
  continue the turn instead of executing a truncated call (#196, #197
  layers one and two).
- The think-splitter no longer leaks reasoning into visible content when
  the text contains bare `function=` or `parameter=` strings (#196/#197
  companion fix).
- Streaming tool-call parsing handles bracket-style dialects with a
  balanced string/escape-aware scanner, buffers incomplete calls instead
  of double-delivering them, and passes through calls to undeclared
  tools per the OpenAI contract instead of dropping them (David Tai,
  #195).
- Constrained generation bounds the `<think>` prelude at 4000 characters
  (`MTPLX_THINK_PRELUDE_MAX_CHARS`, 0 restores unbounded), so an
  unclosed think block can no longer consume the entire token budget and
  return no document (Jozef Kristek, #213).
- Forge model probes recover from slow Hugging Face config responses:
  30s timeout, pinned-SHA retry, positive-MTP-only indexed acceptance,
  and revision-string validation (Philip John Basile, #210).

### Changed

- Dependency bumps: pillow 12.3.0, actions/checkout 7.0.1,
  actions/setup-python 7.0.0, pypa/gh-action-pypi-publish 1.14.1.

## [2.3.0] - 2026-07-21

The agent reliability release: the #170 tool-argument collapse is
root-caused and fixed, structured output ships with full speculative
speed (community-contributed), agent sessions keep their warm cache
through client history rewrites, and eight findings from an independent
source review of v2.2.0 are fixed, each with a regression test that
fails on the pre-fix code.

### Fixed

- Agent tool calls with nested arguments no longer collapse to empty `{}`
  arguments (#170). The streaming Qwen-XML tool parser silently discarded
  function-body text that was not wrapped in `<parameter=>` blocks, so a
  model that wrote its arguments as a JSON object inside the
  `<function=...>` envelope — the common slip on nested `edits` arrays —
  produced a schema-valid call with empty arguments on requiredless tools
  and a silently vanished call otherwise. Both the streaming and the final
  parser now accept a pure JSON-object function body as the arguments
  payload, and every unwrapped, lead, or trailing text lane is a loud
  protocol fallback with identical contracts in the two parsers.
- The injected tool contract no longer shows degenerate `[]` examples for
  array parameters: exemplars are populated from the item schema's own
  keys (an `edits` array renders as
  `[{"search": "ARGUMENT_VALUE", "replace": "ARGUMENT_VALUE"}]`), removing
  an in-prompt template for empty-array emissions.
- The final tool-call extraction lane (non-stream parsing and the stream
  fallback) no longer fabricates `{}`-argument calls out of function bodies
  it could not read — live-reproduced on v2.2.0 as a `grep` call arriving
  with empty arguments. All extraction dialects now share the strict
  parsers' contract: a pure JSON-object function body is the arguments
  payload, an unreadable body stays visible content, blank-body no-argument
  calls still parse, and partial arguments are delivered exactly as the
  model wrote them (schema validation remains the client's job, per the
  OpenAI protocol).
- Context-copy speculative rounds now stop accepting at the first
  accepted stop token, exactly like the MTP acceptance loop. Previously
  the copy lane could accept an entire block past a stop, leaving the
  target cache, logits/hidden selection, and MTP history advanced beyond
  the emitted response while the final state was still marked safe to
  commit — a session-cache poisoning risk on recurrent (GDN) models.
- Context-copy is prompt-only again at the boundary: proposal blocks are
  sliced from the prompt and capped at its edge instead of running into
  the model's own generated output (the self-repetition case the feature
  contract excludes). Boundary-less candidates are skipped inside the
  n-gram index so the best *valid* match still fires.
- Non-divisor packed MTP quantization widths (5-bit, 3-bit, 6-bit) now
  infer the correct group size via total-bit arithmetic (#182; fix by
  @Jonathangadeaharder in #183). A 5-bit group-64 head previously
  inferred group 60, overwrote the artifact's correct declared contract,
  and made the model unloadable.
- Long-running daemons no longer accumulate unbounded telemetry: the
  SessionBank eviction log is a bounded ring, the scheduler's
  started-by-batch-key counter collapses per-session key suffixes to
  stable classes, the dashboard per-session TPS map is LRU-bounded, and
  the OpenCode title fast path trims request metrics like every other
  lane (#145-adjacent slow-creep hygiene).
- The desktop Hugging Face probe now recognizes rootless assistant-pair
  bundles (`mtplx_pair.json` + `target/config.json`) the same way the
  Python runtime does, so the official Gemma4 pair repos classify as
  ready to download instead of unreadable (#107); Forge routes them to
  install-instead-of-rebuild.
- `mtplx doctor --json` stdout is now guaranteed machine-parseable: the
  report is built with stdout routed to stderr, so third-party lazy
  importers that print() their errors (huggingface_hub does, whenever its
  HTTP dependencies are broken or split across install locations) can no
  longer prefix the JSON document with prose and break `--json` consumers.
- Agent tool-loop sessions no longer lose their warm cache when the
  client rewrites history (transcript compaction, retroactive tool-result
  digests): a common-prefix identity fallback (≥4096 shared tokens and
  ≥25% of the prompt) keeps the session id instead of minting an
  anonymous one, ending the cold full-context re-prefills those
  rotations caused mid-session.
- Fan restore goes through the ThermalForge daemon socket instead of the
  app-killing CLI path, and `mtplx pull` detects interrupted downloads
  as incomplete instead of treating them as ready (thanks @titan550,
  #178/#179/#180/#181).

### Changed

- `parallel_tool_calls` is now honored: a request that declares it gets
  the declared behavior in both directions (false → at most one tool
  call per turn), with the previous client-profile heuristic kept only
  as the fallback when the field is absent (thanks @PhilipJohnBasile,
  #190 — Android Studio declares false today and was being ignored).
- Agent tool contract v13: instructs whole-file reads (the old
  "smallest read range" clause provoked storms of 1-to-5-line
  micro-reads) and forbids echoing file contents into visible text
  (double-emitted file bodies inflated agent turns by tens of
  thousands of characters). Measured on live agent sessions: thinking
  share of generated tokens 75% → 6-29%, double emissions eliminated.

### Added

- Structured output: `response_format` `json_object` and `json_schema`
  are now enforced with llguidance token masks instead of silently
  ignored — and the grammar composes with the MTP verify loop, so
  constrained requests keep full speculative speed (measured: decode
  parity within noise, under 5ms total mask cost per request). On
  thinking templates a prelude grammar lets reasoning finish before the
  document is forced. Opt-in strict tool-call grammars
  (`MTPLX_TOOL_CALL_STRICT=1`) force every tool call to a declared tool
  name with schema-valid arguments — a real OpenCode session built a
  complete project through the strict lane with zero malformed calls.
  Thanks @PhilipJohnBasile (#186, #187, #188). Requires the `[server]`
  extra, which the desktop app installs by default; bare pip installs
  get a clear 400 with an install hint.
- Durable per-request telemetry: `--request-log-jsonl` (env
  `MTPLX_REQUEST_LOG_JSONL`) appends every request record as one JSON
  line, and `scripts/session_forensics.py` correlates that log with an
  OpenCode database into a single timeline with detectors for re-prefill
  rewinds, TTFT stalls, thinking marathons, double emissions,
  session-identity rotations, and usage mismatches.
- An **opt-in** agent-lane reasoning budget (`--agent-thinking-budget`,
  env `MTPLX_THINKING_BUDGET`; OFF by default): at the budget the
  reasoning segment is force-closed with a visible bridge so the turn
  proceeds to its answer or tool call, and every engagement is surfaced
  per-request in telemetry (`thinking_guard`). Below the budget decoding
  is bit-exact; plain chat is never touched.
- Stream stall watchdog (#86 containment): if a stream receives nothing
  while the model owner's progress heartbeat is frozen for
  `MTPLX_STREAM_STALL_DEADLINE_S` (default 300s, 0 disables), the
  request fails with a structured, diagnosable error and releases its
  slot instead of hanging forever. Healthy long prefills and model loads
  tick the heartbeat and are never affected; the daemon is never killed.
- `pip install mtplx` now includes Pillow, matching the advertised image
  support in the app and server (#103); previously vision failed at
  import unless the `[server]` extra was installed.

## [2.2.0] - 2026-07-19

The copy-drafting and small-Mac release. Decoding: context-copy
(prompt-lookup) drafting lands on by default (PR #151 by lBroth) with an
exact temperature path — copied blocks are accepted with the target's own
shaped probability, so the output distribution is unchanged at any
temperature; measured +53% on edit-heavy agent turns at temp 0.6, parity
on novel text (disable with MTPLX_CONTEXT_COPY=0). Models: the 4B
zero-acceptance defect (#176) is root-caused and fixed — the engine heals
raw delta-encoded MTP sidecars at load so existing downloads recover
without re-downloading, the 4B Speed artifact is rebuilt (227.8 tok/s D3
on M5 Max, 1.71x AR), and a new 4B Quality artifact ships (191.7 tok/s
D3, 2.19x — the largest MTP multiplier in the fleet); sub-16GB Macs get
first-class catalog recommendations. Tune (#177): a 0.0-acceptance depth
can never win, be saved, or be replayed, and poisoned records are
quarantined at load. Forge: FP16 precision option with M1/M2 auto-select
(#166); the fp16 cast can no longer corrupt a sidecar in place;
degenerate sidecars are quarantined on re-forge. Server: SSD session
writer crash fixed — encode at enqueue (#169); live requests preempt idle
cache maintenance. MoE: mtp_depth_max is a ceiling, not the default; the
35B-A3B launches at its measured D2 (#174 part 1 by davidtai). Also: the
Qwen 3.6 27B AR decode-trace crash fix (#167 by davidtai), truthful
per-model profile display, hybrid-model boundary retention across append
churn, and an experimental --draft-core device (opt-in). Full details in
docs/releases/v2.2.0.md.

## [2.1.0] - 2026-07-17

The community-fixes release. Memory: the v2.x reports are root-caused and
closed (MLX allocator cache bounded by default, per-session admission
re-clamped on sub-96GB machines, paged pool bounded by the context
window, pressure responder redesigned, q4 kv-quant crash fixed, new
`--memory-budget` knob). Agent sessions: warm prefix reuse survives every
tool turn (#121), hybrid-model near-prefix restores no longer collapse to
the oldest boundary (measured 0.4s instead of 33.8s on a 22k follow-up),
restart-warm sessions keep their boundary records across SSD generations
(#159, #144), and cache hits are reported in standard `usage` fields.
Sampling: presence and frequency penalties fixed in the batched AR lane
(#156). App: startup and update hang fixed plus a full subprocess
watchdog sweep (#158), the Hermes tile launches Hermes Desktop, raw
tool-call XML no longer leaks into no-tools chats (#160). CLI: `start
opencode` serves the same lane the app serves. Backends: qwen3_5_mtp and
hy_v3 land (#142, #147). Performance: the model-owner thread is
QoS-pinned for 8 to 10% faster decode under real multitasking load.
Operators: `MTPLX_COMPILED_VERIFY_MAX_CONTEXT` is env-overridable. Full
details in docs/releases/v2.1.0.md.

## [2.0.2] - 2026-07-09

The agent-reliability release: multi-turn agent sessions now render
reasoning history on the model's trained contract (the fix for the
plan-execution repetition marathons), LAN serving works from the app,
and every agent client gets warm prefix reuse.

### Changed

- Reasoning history is now scoped to the active agent round by default
  on Qwen3.6/3.5 models, matching Qwen's trained multi-turn contract.
  The Qwen chat template keeps `<think>` blocks only for assistant
  messages after the last real user query (its built-in "rolling
  checkpoint"); MTPLX used to override that with `preserve_thinking`,
  which rendered an off-contract empty `<think>` scaffold on every
  completed assistant turn and replayed stale inline reasoning across
  turns, while silently dropping the structured `reasoning_content`
  fields agent clients such as OpenCode send. Scoped mode lets the
  template's own checkpoint govern: completed turns render with no
  think scaffold, and the active round (assistant -> tool -> assistant
  chains) keeps its reasoning, now including structured
  `reasoning_content` - strictly better in-round continuity than
  before. `--preserve-thinking on` restores the previous behavior
  byte-for-byte (including cache identity), `off` still strips
  everything, and the new explicit `scoped` value pins the scoped mode.
  Templates without the rolling checkpoint (Gemma 4, custom templates)
  keep the previous preserve-all behavior. The resolved policy is shown
  at startup ("Reasoning history: scoped (active round only)") and as
  `reasoning_history_mode` in `/v1/mtplx/settings` and the snapshot.

### Added

- Loop Guard (opt-in): `MTPLX_LOOP_GUARD=1` enables a loop-armed
  anti-repetition steering mode for models prone to verbatim repetition
  marathons (for example repetition-damaged third-party quants). Unlike
  a static presence penalty, the guard is completely inert until a real
  loop is detected mid-response (bit-exact sampling otherwise, MTP
  acceptance math unchanged), then penalizes only the tokens that would
  extend a verbatim repeat and disarms once the loop is broken. Content
  inside tool calls is never steered (code legitimately repeats short
  token runs, so tool-call spans are masked token-exactly). The default
  is OFF: no synthetic steering touches sampling unless you ask for it,
  and the repetition fix that matters for MTPLX's own models is the
  scoped reasoning history change above. Detector/steering knobs are
  documented in `mtplx/loop_guard.py`; per-request guard activity is
  visible under `loop_guard` in `/v1/mtplx/snapshot`.

### Fixed

- Hidden reasoning no longer leaks into visible chat content when a
  long reasoning tag (e.g. `</reasoning>`) splits across a streaming
  chunk boundary; the splitter held back too few bytes for tags longer
  than `</think>` (PR #149 by @Osamaali313).
- `mtplx` quickstart onboarding screens now display the requested
  `--host`/`--port` in the Web UI and dashboard URLs instead of a
  hardcoded `http://127.0.0.1:8000` (PR #148 by @hasegaw).
- Ctrl-C now returns control to the terminal within a bounded delay
  even when a browser tab holds an open chat/dashboard stream. The
  server previously waited forever for infinite SSE generators to
  finish; shutdown now drains in-flight requests with a 5-second
  deadline, and thermal/fan cleanup still runs (#124).
- Serving on any host and port now works from the app. Setting host
  0.0.0.0 (LAN serving) used to misreport free ports as occupied, bump
  the port, and kill the healthy daemon after a health-wait timeout,
  because the app probed the bind address verbatim; all app-side
  connections now resolve wildcard binds to a connectable loopback
  address (#109). The app also surfaces the "LAN serving requires an
  API key" rule before launch instead of a generic Degraded state, an
  API-key mismatch reads as a live-but-unauthorized daemon instead of
  a lost one, and the port-in-use preflight tests the address family
  the daemon will actually bind.
- The app now passes the SSD session cache setting to the daemon
  explicitly, including "Off". Since 2.0.0 flipped the serve default to
  on, an explicit Settings "Off" was silently re-enabled with default
  limits on app-launched daemons, and `~/.mtplx/session-bank` kept
  growing (#140; also the "session-bank came back after I deleted it"
  half of #138). Generated `mtplx start` server commands carry the
  explicit `--ssd-session-cache off` for the same reason.
- The app's runtime venv now self-heals after app updates. A venv whose
  base-interpreter symlink pointed into a replaced app bundle made every
  reinstall fail with "[Errno 2] No such file or directory:
  .../runtime-venv/bin/python3" and no DMG reinstall could fix it; venv
  creation now rebuilds with `--clear` when the existing venv python is
  broken or creation fails (#139).
- Warm prefix reuse no longer freezes on the oldest short prefix for
  agent harnesses outside OpenCode (Pi, little-coder, Hermes, custom
  clients). The block-prefix restore lane was still gated to OpenCode's
  compact tool contract even though kvcache-v2's boundary-true restores
  made it exact for every client, so transcripts whose turns diverge
  more than a few tokens before the stored end re-prefilled a growing
  suffix every turn (#138). The lane now engages for all clients while
  boundary-true restore is on; `MTPLX_SESSION_BLOCK_PREFIX_RESTORE=0`
  still disables it.

## [2.0.1] - 2026-07-07

Turbo for every Mac. The v2 turbo default now covers every dense catalog
model on every Apple Silicon generation, with a load-time kernel
self-validation safety net.

### Added

- 6-bit affine verify kernels (split-K hexpack family): the 6-bit 9B tier
  gains 33-62% decode and 43% 2k-prefill under turbo (M5 Max, verified
  arms). Qwen 3.5 9B and 9B FP16 now default to turbo.
- New model: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality-FP16`, the
  missing M1/M2 quality artifact. Wired into the app picker, CLI catalog,
  and chip-aware routing (a Quality pick on M1/M2 resolves the FP16
  sibling, mirroring the speed lane). Validated against the bf16 parent
  and measured at 2.5x over true AR under turbo.
- Load-time kernel self-validation: every turbo lane checks itself against
  stock MLX in the model's exact dtype/quantization at boot; a mismatching
  lane falls back to the stock path for the session and the verdicts are
  surfaced as `kernel_selfcheck` in `/health`. Worst case is v2.0.0 speed,
  never wrong output.
- `MTPLX_FORCE_GPU_FAMILY_FALLBACK=1` rehearses the exact M1-M4 code path
  on newer machines; `MTPLX_KERNEL_SELFCHECK=0` disables the probe.
- CI kernel matrix on real M1 runners: kernel exactness across
  {bf16, fp16} x {4, 6, 8}-bit plus a live 4B turbo boot smoke.

### Changed

- 27B Optimized-Speed-FP16 (the M1/M2 routing target) defaults to turbo:
  19-31% faster decode than the v2.0.0 sustained default across 0.5k-32k
  context; true-AR multiplier 1.34x to ~2x. First-ever e2e measurement of
  this artifact; exactness gated (30/30 hot-shape logit-diff cases on real
  weights, greedy match turbo vs stock).
- The published Speed-FP16 model card now recommends the turbo profile.

### Unchanged on purpose

- 35B A3B MoE and Gemma 4 keep sustained (expert layers / assistant-pair
  architecture bypass these kernels; named 2.0.2 lanes). The 4B keeps
  sustained (turbo measured slightly slower at matched depth). Compiled
  verify stays off for 6-bit models.

## [2.0.0] - 2026-07-06

MTPLX v2: the coding-agent release. Session-cache v2 (RAM + SSD), the
turbo profile with NAX verify kernels and compiled verify, a new verify
attention kernel wave for long context, and a long campaign of
OpenCode/agent-bridge fixes measured on real sessions.

### Added

- Session-cache v2: boundary-true GDN restores, O(1) RAM restores, SSD
  cold tier (default on) that survives daemon restarts — a 100k-token
  session restores in ~2s after a restart instead of a five-minute cold
  prefill. Prompt-cache reuse now chains across agent tool rounds.
- Turbo profile: verify-specialized quantized-matmul kernels
  (`MTPLX_NAX_VERIFY`, vk_k/vk-q8 families) plus context-routed compiled
  verify with a per-model quantization gate. Measured on M5 Max: 27B
  Optimized-Speed 44.7 -> 58-60 tok/s chat lane; Optimized-Quality (q8)
  31-36 -> 43-44 tok/s.
- The quantized 27B flagships (Optimized-Speed, Optimized-Quality, and
  the legacy Optimized hybrid) now default to the turbo profile on the
  CLI and the bare OpenAI API — the same launch rule the macOS app
  applies. Explicit `--profile` flags and wizard picks still win; other
  models keep the sustained default.
- `POST /admin/cache/clear` resets the MLX peak-memory counter after
  dropping the session bank, so per-request `peak_memory_bytes` reports
  the current phase instead of a process-lifetime ratchet (benchmark
  harnesses that clear between context rows now chart honest per-row
  peaks).
- `--scheduler-mode ar_batch` now genuinely admits anonymous OpenAI
  clients into the concurrency-adaptive batch lane (lone requests keep
  solo MTP; real concurrency shares the batched AR decode lane), and
  the batched lane samples on the GPU (decode-heavy batch-8 aggregate
  70.9 -> 79.2 tok/s). Serial remains the default: measured end to end,
  serialized solo-MTP still beats batched AR on prefill-heavy
  concurrent loads because MTP decode is ~4x faster per stream.
- Long-context decode wave: a packed-GQA verify attention kernel plus
  commit-first KV donation in compiled verify. Measured on M5 Max
  (Optimized-Speed): 64k decode +12%, 128k decode 17 -> 20+ tok/s, and
  peak memory down 8 GB at 64k / 16 GB at 128k.
- Startup warming without the wait: the daemon is ready in ~2s and the
  deeper kernel/shape warmup continues silently in the background,
  yielding instantly to real requests. First messages hit warm kernels
  without a slow boot.
- RAM session-cache budget now scales to the machine (roughly half the
  RAM headroom above the model) instead of a flat cap, and the app's
  Settings tab exposes explicit RAM and SSD cache limits.
- Per-request presence/frequency penalties end-to-end (server, CLI
  flags, dashboard slider, app dial), with MLX on-device penalty math.
- App chat: markdown renders live during streaming at zero per-token
  cost; each turn gets one compact activity strip with grouped tool
  rounds and a sources footer for web results; turbo is a first-class
  mode in Settings.
- Tool contracts are date-anchored (web-search answers stop regressing
  to the training cutoff) and post-search answers are no longer
  clipped to one sentence.
- Vision: images flow through the OpenAI API into MTP decode; MoE
  multimodal checkpoints that store the tower under `model.visual.*`
  (for example Ornith 1.0) are now recognised (community PR #134).
- Gemma 4 assistant-pair models default to their measured-best MTP
  depth; explicit `--depth` still wins.

### Fixed

- Fresh installs no longer crash at model load: transformers 5.13.0
  broke mlx-lm's import (`AutoTokenizer.register` string key), which
  killed every new install and DMG first run. mtplx now pins
  `transformers<5.13` (#135, #136, community PR #137).
- The app no longer kills a healthy engine. The health watchdog
  treated a response it could not parse the same as a dead server and
  terminated the daemon mid-session — the main driver of "Stream
  offline" / "server dies mid-session during agent workloads" reports
  (#105). Liveness is now transport truth: if the daemon answers, it
  lives.
- SSD session-cache restores are boundary-true for recurrent (GDN)
  layers, fixing corrupted agent output after a prefix restore (prompt
  recitation, phantom tool calls, argument leakage) (#130).
- Vision + MTP: the draft head's committed history now consumes the
  spliced vision rows instead of image-pad embeddings, fixing
  fabricated visual differences between similar screenshots (#103).
- Smart fan control ramps at request arrival, verifies actual RPM, and
  holds through the post-response cache work instead of dropping to
  auto while the GPU is still pinned (#127).
- The app no longer rewrites the Hermes profile config on every
  launch; user sections (memory/providers/delegation/auxiliary) are
  merge-preserved (#131).
- Served model identity is contract-match-only: third-party builds no
  longer get coerced onto official `mtplx-*` model ids (#57).
- OpenCode plan -> build mode switch no longer breaks the prompt cache
  or hides file tools. The bridge misread OpenCode's build-mode
  reminder ("no longer in read-only mode") as a read-only instruction,
  hid write/edit exactly when the user said "execute the plan", and the
  model spiralled re-planning files it could not create. OpenCode
  toolsets now pass through byte-stable; the negation is parsed
  correctly for other clients.
- Agent transcripts render prefix-stable across rounds (historical
  bytes never rewrite), force-answer and Pi-convergence contracts ride
  as pure suffixes, and warm prefills inherit recurrent boundaries —
  together these take mid-session tool rounds from multi-second cold
  re-prefills to sub-2s warm restores.
- One busy OpenCode conversation no longer evicts every other project
  from the RAM session cache (prefix-superseded entries + a wider
  high-memory entry budget); multitasking across projects keeps each
  project's cache warm.
- `mtplx start`'s live-dashboard handoff no longer crashes on an
  ImportError (community PR #133).
- The batch scheduler no longer accumulates every finished request
  forever (community PR #132).
- Prefill disconnect-cancel: closing an agent client mid-prefill frees
  the engine immediately instead of finishing a 48k-token orphan.
- CJK and dead-key input no longer drops composed characters in the
  app's chat composer (community PR #119).
- Dense-layout prefill chunk cache-cleanup cadence relaxed 1 -> 4:
  5-21% prefill TPS, memory byte-identical.

### Changed

- Bumped to 2.0.0. The default OpenCode/agent daemon profile is turbo
  with the compiled-verify per-model gate; q8 Quality stays on the
  eager verify path it measures best on.
- Removed the vestigial "required MLX fork" metadata from all profiles.
  MTPLX runs on stock PyPI MLX and always has in the shipped product;
  the speed stack (NAX verify kernels, packed-GQA verify attention,
  compiled verify) ships as in-package Metal kernels, not a patched
  MLX/qmm build. Profile payloads no longer carry
  `required_mlx_fork_commit`/`required_mlx_fork_fragment`, `/health`
  now reports a plain `mlx_runtime` diagnostic instead of a fork
  expectation, and `--strict-fast-path` /
  `--strict-mlx-fork-assert` are accepted as deprecated no-ops.

## [1.0.4] - 2026-06-12

Same-day hotfix: 1.0.3 broke coding agents on their first tool turn,
and pasting a GGUF repo got a wrong answer.

### Fixed

- Coding agents crashed on 1.0.3. Any tool-using client (Pi, Hermes,
  OpenCode, or anything speaking the OpenAI tools protocol) hit
  "unexpected keyword argument 'vision_splice'" on its first tool
  turn after a cache miss: non-streaming callers got a 500, streaming
  agents lost the stream, and the app reported "Stream offline." One
  stray argument left behind by the vision work, in a diagnostics
  call only agent tool turns reach. Removed, with tests that drive
  the exact path and an audit test that fails if any call ever passes
  an argument its target does not accept (#99, #100).
- Pasting a GGUF repo into "Add a model from Hugging Face" claimed
  the repository did not exist. The app now says what is actually
  going on: GGUF is llama.cpp's format and MTPLX runs MLX models. It
  names the source repo the GGUF was made from so Forge can convert
  it, and genuine typos get "check the name" instead.
- The "Add a model" repo check now follows the configured Hugging
  Face download mirror instead of always probing huggingface.co, so
  it works on networks where huggingface.co is blocked.

## [1.0.3] - 2026-06-12

The app can see. Vision lands across the Qwen models, and the
compatibility gate stops blocking models that run fine.

### Added

- Vision support in chat and the API. Attach PNG, JPEG, or WebP images
  in the app composer, or send OpenAI image_url content parts to
  /v1/chat/completions, and the model describes what it sees with MTP
  speculative decoding still running on top. Works on Qwen 3.6 27B
  (Speed and Quality), Qwen 3.6 35B, and Qwen 3.5 9B. The 9B repo on
  Hugging Face regained its vision weights; an explicit mtplx pull now
  syncs such repo updates into existing local copies automatically.
- /health reports whether the loaded model supports vision, and the
  composer adapts to it.

### Fixed

- Models that run fine are no longer refused for paperwork. The
  compatibility gate treated unverified runtime contracts (including
  the official Optimized Quality build) and even "slower than AR"
  speed evidence as reasons not to load. Verification is now a label:
  unverified models load with an honest note, and refusals are
  reserved for models that genuinely cannot execute (#98).
- The gate's explanation message crashed with a traceback instead of
  printing since 1.0.0. It prints again, including the hint that was
  supposed to unblock you (#98).
- Image attachments preview their actual pixels in the composer and
  the transcript instead of a "Could not read" placeholder.

## [1.0.2] - 2026-06-11

Bug-fix release with one small feature.

### Fixed

- Choosing the Auto or Sustained Max profile in the app's Settings left
  the engine unable to start, showing Degraded on every launch until
  the profile was changed back. Both values now resolve to real
  profiles (Sustained Max keeps its pinned-fans intent as the fan mode
  setting), existing configurations heal themselves on load, and the
  picker only offers values the engine accepts. `mtplx serve --profile
  auto` works from the command line too.
- Parallel requests from agent tools that do not send session ids could
  fail with "session anon-... is already in flight" when they shared a
  prompt prefix. Busy sessions now fork to a fresh session instead of
  erroring, and anonymous session ids are random rather than clock
  derived. Reported and fixed by Frank Denis (@jedisct1) in #95.
- A daemon launch that lost its port (another server bound it between
  checks, or a listener invisible to the local probe held it) now
  remediates and retries once before reporting a failure, and the
  failure message names the occupant when it can.

### Added

- Optional Hugging Face download mirror for networks where
  huggingface.co is blocked (requested from mainland China in #96). Set
  it inline in the onboarding download step or later in Settings under
  Advanced; downloads and the engine then use the mirror endpoint. The
  stored HF token is never sent to a mirror, so gated repos stay on the
  official endpoint.

## [1.0.1] - 2026-06-11

Bug-fix release.

### Fixed

- First-run tuning no longer fails on Macs where fan control cannot
  verify a max ramp (for example when the passwordless helper grant is
  not in place yet). Tuning now runs with fans on automatic, the
  results are labeled accordingly, and `--require-max-fans` keeps the
  strict behavior for benchmarking.
- The `mtplx` CLI accepts the official Gemma 4 assistant-pair repos
  directly from Hugging Face. The app already ran them; the CLI's
  preflight now reaches the same verdict.

## [1.0.0] - 2026-06-10

The first full release: the native macOS app and the `mtplx` command line
working as one product. Full notes:
[mtplx.com/releases/notes/v1.0.0](https://mtplx.com/releases/notes/v1.0.0.html).

### Added

- Native macOS app with onboarding (hardware check, model pick, guided
  setup, tuning), a live speed dashboard (decode gauge, acceptance by
  depth, verify waterfall, activity), native chat with attachments and
  web search, Forge, the AIME benchmark, and agent launchers for
  OpenCode, Pi, Hermes, and Open WebUI.
- New models: Gemma 4 (assistant-pair drafting tuned by draft block
  size) and Qwen 3.6 MoE 35B-A3B (prequantized expert sidecars,
  normalized expert layouts, hard blocks on unrunnable layouts),
  alongside Qwen 3.5 4B and 9B for smaller machines.
- KV cache reuse on two layers: warm-prefix reuse in RAM across turns
  and requests (multi-turn chats and agents like OpenCode hit the cache
  instead of re-processing the conversation), and an SSD session cache
  that persists KV state to disk with enforced size caps and restores
  near-instantly across server restarts.
- Concurrency: continuous batching with presets, a scheduler mode, and
  explicit caps (`--max-active-requests`, `--decode-batch-max`,
  `--batch-wait-ms`).
- Smart fan mode across the app, CLI, and server API: ramps while the
  model works, restores on idle, survives client handoffs, and keeps the
  crash-safe restore watchdog.
- Forge: convert any Hugging Face repo to MLX (AWQ, compressed-tensors,
  NVFP4, BF16 sources), calibrate and train the MTP adapter, verify with
  quality gates that reject speed wins that degrade output, and publish
  with provenance. Vision towers are preserved through conversion. In
  the app and as `mtplx forge`.
- Agent-grade serving: hardened tool contracts and dedicated lanes for
  OpenCode, Pi, and Hermes; long-context depth policy; client identity
  tagging; a live server-sent metrics stream plus snapshot, thermal, and
  prefill-history endpoints; honest cancellation that stops decode.
- Automatic runtime setup during onboarding: the app installs its own
  Python engine, fan control (ThermalForge), and the `mtplx` terminal
  command without requiring Homebrew. Release builds bundle a pinned
  CPython interpreter, the engine environment ignores user pip
  configuration, and the interpreter is signed so installed packages
  load on macOS 14 and 15. A stale `mtplx` on PATH is updated
  automatically; a newer one is left alone.
- Official Apple Silicon model catalog (Qwen 3.5/3.6, Gemma 4 in speed,
  balance, and quality builds) with device-aware defaults shared by the
  app and the CLI: chip generation picks precision and machines under
  32 GiB route to the 9B model automatically.
- App-aware `mtplx start`: detects a running MTPLX server and attaches
  instead of loading a second copy, lists installed models first, and
  adds a "Same as the MTPLX app" option. `mtplx stop` knows the app's
  persisted port.
- New commands: `mtplx stop`, `mtplx settings get/set`, and
  `mtplx bench aime` for running the app's AIME benchmark from the
  terminal.
- Sparkle automatic app updates with signed appcasts; the app verifies
  the installed engine against the shipped wheel and refreshes it after
  each update.

### Changed

- Busy ports are now handled gracefully everywhere: the app moves to the
  next free port with a banner (and persists it), and the CLI explains
  exactly who owns a busy port and how to stop it.
- The OpenAI-compatible server honors `stop` sequences (chat,
  completions, and Anthropic `stop_sequences`) and `/v1/completions`
  streams tokens as they are generated with real finish reasons.

[2.5.1]: https://github.com/youssofal/MTPLX/releases/tag/v2.5.1
[2.5.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.5.0
[2.4.2]: https://github.com/youssofal/MTPLX/releases/tag/v2.4.2
[2.4.1]: https://github.com/youssofal/MTPLX/releases/tag/v2.4.1
[2.4.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.4.0
[2.3.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.3.0
[2.2.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.2.0
[2.1.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.1.0
[2.0.2]: https://github.com/youssofal/MTPLX/releases/tag/v2.0.2
[2.0.1]: https://github.com/youssofal/MTPLX/releases/tag/v2.0.1
[2.0.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.0.0
[1.0.4]: https://github.com/youssofal/MTPLX/releases/tag/v1.0.4
[1.0.3]: https://github.com/youssofal/MTPLX/releases/tag/v1.0.3
[1.0.2]: https://github.com/youssofal/MTPLX/releases/tag/v1.0.2
[1.0.1]: https://github.com/youssofal/MTPLX/releases/tag/v1.0.1
[1.0.0]: https://github.com/youssofal/MTPLX/releases/tag/v1.0.0
