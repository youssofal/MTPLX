# Changelog

All notable user-facing changes to MTPLX. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`q6` paged KV cache quantization.** Adds an opt-in intermediate 6-bit
  storage mode between `q8` and `q4`, packing four codes into three bytes.
  It is available through config, CLI, environment variables, and the app on
  supported Qwen families. Defaults remain unchanged.

## [2.9.2] - 2026-08-25

### Changed

- **The serving endpoints are passthrough by default** (#282). No
  tool-result compaction, no read trimming, no injected steering text
  unless a rewrite feature is explicitly enabled.
  `MTPLX_AGENT_REWRITES` is the master switch; each feature arms only
  via its own environment variable. The macOS app stopped exporting
  the legacy compaction settings when launching coding agents, and the
  request log records what was and was not rewritten per request.
- Managed client configs respect user edits: `mtplx start` and the app
  only update files they wrote themselves (#282).
- **Chained greedy drafting is on by default for temperature 0
  requests below 12,288 prompt tokens** (#313, #315, #318). Gated A/B
  runs on an M5 Max measured +2.5 to +9.8 percent decode on 0.5k to 8k
  prompts and -2.9/-2.7 percent at 16k/32k, so a context fence keeps
  it off at and above 12,288. Tune with
  `MTPLX_GREEDY_TRIO_MAX_CONTEXT`, disable with
  `MTPLX_GREEDY_DRAFT_CHAIN=off`. Sampled requests are untouched.

### Fixed

- The forge decides the MTP norm convention once per tensor set
  instead of blind-shifting three norm tensors by +1.0 (#301); packs
  from absolute-encoded sources no longer ship with draft acceptance
  collapsed to 0 to 2 percent. The runtime refuses to load a
  double-shifted trunk with a clear error (#306).
- Images survive user-message canonicalization on consecutive or
  retried turns (#327); vision rows survive near-prefix cache
  restores (#296).
- The literal-repetition stop covers width 2+ batched MTP cohorts
  (#311).
- The default request JSONL is content-free as documented (#326).
- The installer and app write the PATH line through a symlinked
  `~/.zshrc` instead of replacing the symlink (#292).
- The dashboard Hardware card reports the real chip (#329).
- `bench --harness depth-sweep` honors `--depths`, `--seed`, and
  `--generation-mode`, and refuses `--stock-ar` loudly (#285).
- The fp16 fused add+rmsnorm kernel uses the exact 1024-lane
  dispatch (#319); the packed-concats exactness gate tests its claim
  honestly (#320).
- The NAX turbo verify path stops using padded M=5 lanes that measured
  slower than stock.
- The flight recorder samples non-streaming requests.
- The app renders `<br>` variants inside markdown table cells
  (PR #273 by @El-Patronum).
- `quantize: false` module overrides are honored (PR #281 by
  @shiftedx).
- Capture tooling persists exact completion token ids on all three
  lanes (PR #330 by @CharliePetch).

### Added

- Experimental, off by default: `MTPLX_FUSE_PROJ` load-time projection
  fusion (port of PR #316 by @grzracz), `MTPLX_VK_CROSSROW` crossrow
  wide-verify kernel, draft-confidence tracing with confidence-gated
  draft width, and marathon postcommit protection.
- `HISTORY.md`: the dated record of putting native MTP on Apple
  Silicon.

## [2.9.1] - 2026-08-22

### Fixed

- **Agent sessions could silently truncate and then crash near 19,000
  tokens** (#310). Paged-KV capacity now derives from the pages actually
  allocated; long coding sessions run to the model's full advertised
  context.
- **Shutdown segfault** (#303). The daemon parks its model-owner thread
  and clears MLX streams at exit; quit and restart are clean.
- **Turbo applied its full configuration.** One turbo fast-path flag
  shipped runtime-dead in 2.9.0. The fast-path env is now a single
  shared block, `/health` reports exactly what the profile set, and a
  per-lane kernel selfcheck runs at startup.
- **Multi-turn cache reuse on agent lanes.** One tokenization policy
  across all encode paths (no more cache walls at reasoning
  boundaries), tool-call turns bank their generated output directly
  from live KV, and interrupted background commits retry.
- **Long sessions stop re-deriving prior reasoning.** Client-echoed
  reasoning is rendered for turns the committed cache has not covered,
  ending marathon re-thinks of already-derived plans.
- Stamped pack draft-sampler settings win over stale client-side pins.
- Model pack updates resolve the exact installed pack directory, so a
  stale legacy bare-name cache no longer shadows the canonical copy
  into a zero-byte no-op, and update progress reports cumulative bytes
  (first pip/brew release with the fixes app users received in the
  2.9.0 updater hotfix, build 2009001).

### Changed

- **OpenCode** runs uncapped by default (the managed plugin strips
  exactly the injected 32,000 ceiling; explicit caps pass through),
  with reasoning, effort selection, reasoning round-trip, and session
  cache identity honored end to end.
- **Pi** gets a working reasoning-effort picker, a real advertised
  output ceiling (instead of Pi's silent 16,384 default), and a managed
  extension for cap hygiene and session identity — written identically
  by the app and `mtplx start pi`.
- **Hermes** requests carry client identity and configured reasoning
  effort, and the server strips Hermes's injected 65,536 default cap.
- `mtplx doctor` reports advertised output ceilings and the actual
  configured port for agent lanes.

### Added

- **Flight recorder**: per-second per-request telemetry (tok/s, context,
  speculative acceptance by depth, verify/draft time split, outcome —
  cancelled and disconnected requests included) as local JSONL under
  `~/.mtplx/metrics`, rotation-capped at 256 MB. Disable with
  `MTPLX_FLIGHT_RECORDER=off`.
- `GET /v1/mtplx/flight`: live phase, tok/s, acceptance, stall age, and
  generated-text tail for the request in flight.
- `mtplx trace`: session timelines joined to OpenCode history,
  cache-reuse analysis, automatic pathology flags, repetition
  autopsies, and per-session HTML reports.

## [2.9.0] - 2026-08-20

See the release notes: <https://mtplx.com/releases/notes/v2.9.0.html>.

## [2.8.3] - 2026-08-18

### Fixed

- **Streaming stays smooth while you actually touch the app.** The
  freeze-then-burst stutter that only appeared when a human was
  scrolling or moving the mouse — and never under hands-off testing —
  is fixed. The window-measurement guard ran in a run-loop phase that
  macOS skips while input events keep arriving, so precisely when you
  interacted, every layout pass re-measured the whole conversation and
  screen updates coalesced into bursts. The guard now runs on every
  run-loop turn, input storms included. Measured on the same machine,
  build, and prompt with synthesized human input: 40 s of continuous
  wheel-scrolling went from 70 UI stalls (18.7 s frozen, worst 1.3 s)
  to one 197 ms stall; 30 s of mouse movement over the transcript went
  from 91 stalls (26.7 s frozen) to zero.
- **Scrolling up mid-generation no longer fights you.** The
  auto-follow used to yank the view back to the bottom against your
  fingers: its user-scroll signal was set asynchronously (the
  synchronous bottom-pin raced it), trackpad momentum ran unguarded,
  and classic wheel mice never registered as scrolling at all. User
  scrolling now wins immediately and in every form — pin attempts are
  refused while you scroll, momentum is covered, and scrolling back to
  the bottom re-engages following, matching how it already behaved for
  slow trackpad drags.
- **Cancelled generations now leave full telemetry.** Stopping a reply
  mid-stream previously logged a stub record with no stream-smoothness
  census — the exact runs users complain about were the ones with no
  data. Cancelled requests now record the producer gap census, sliding
  throughput windows, and true decode rate for the streamed portion.
- **Fixed a quadratic decode-cost path on whitespace-free content.**
  The incremental detokenizer's force-flush escape (tables, URLs,
  minified code) could never trim its token cache, so every new token
  re-decoded a growing buffer. The cache now stays bounded on
  arbitrarily long runs.

- **The transcript can no longer go blank mid-generation.** The first
  2.8.3 candidate swapped the chat transcript to a lazy stack for a
  layout-cost fix; under fast streaming with the app's own scroll
  driver, the lazy container intermittently culled every visible row —
  flicker escalating to an entirely empty chat while the engine kept
  streaming. The transcript and both streaming-card stacks are eager
  again (row count stays bounded by the earlier-history slicer), and
  live-streamed markdown tables — including wrapping cells — render
  correctly while they arrive.
- **Thinking is plain text now.** The reasoning well no longer runs
  the model's thoughts through the markdown renderer, and the live
  three-line ticker anchors its window at line breaks — rendered
  thought lines never re-wrap or visibly rewrite themselves as new
  tokens land.
- **The stream no longer freezes and catches up in bursts.** Three
  server-side delivery fixes: the incremental decoder force-flushes
  whitespace-free runs (table separator rows, URLs, minified code)
  instead of holding them for their full length; freshly committed
  tokens are emitted before cache housekeeping barriers instead of
  after; and the auth gate was rewritten as pure ASGI so it no longer
  relays every stream frame through a buffered middleware channel.
  Measured on the same prompt and settings as the field report:
  sub-second delivery silences per answer dropped from ~30 to single
  digits, and generator-side gaps over 200 ms dropped to zero. Every
  request record now logs a producer gap census
  (`producer_gap_ms_p95`/`_max`, `producer_gaps_over_200ms`) so
  stream smoothness is auditable, not vibes.
- **Finished replies no longer make the app progressively hotter.**
  Each settled turn leaked an invisible repeat-forever pulse animation
  (the thinking-indicator dots), and every leaked pulse drove display
  cycles that re-measured the window against the whole transcript —
  CPU at rest climbed with every completed reply, up to ~half a core.
  The dots now stop by construction when they leave the screen, and
  the window-measurement guard re-asserts itself after each render
  phase. With no engine running the app sits at 0.0% CPU regardless
  of transcript size; the remaining background cost while a daemon is
  running is the dashboard feed, tracked for the next release.

- **The desktop app no longer renders the whole transcript per frame —
  long chats stay smooth on screen, not just on the wire.** Founder
  testing on a heavy multi-turn conversation at temperature 1.0 caught
  the other half of the freeze-and-burst reports: the app's window
  re-measured every realized message on every layout invalidation (a
  third of the main thread at idle, up to 62×/s while streaming — the
  guard meant to prevent this had silently never applied), markdown
  fence classification re-walked every character of every block per
  frame, per-delta paths copied the entire answer to test emptiness,
  and scroll pacing state invalidated the full view tree per tick. All
  of it is now O(new content): fence counts and syntax-lex state are
  computed once per block, the min-size walk is dead, and the 10 Hz
  metrics chip no longer re-evaluates every bubble or parses its stream
  byte-by-byte. Catch-up after any hiccup is rate-limited (max 256
  chars/frame) so it reads as fast typing, never a paste.
- **Markdown tables no longer draw rows on top of each other.** Table
  cells measured one line tall (the horizontal scroller proposes no
  width) but drew wrapped, so long cells bled over the rows below.
  Cells now measure at their placement width.
- **Chat streaming no longer freezes mid-response and then dumps the
  backlog in one burst.** 2.8.0 added a wire safeguard for the uncapped
  repetition stop that held a fixed ~448-token tail off the stream on
  every uncapped request — which is every desktop, web, and agent chat.
  At chat speeds that silenced the stream from roughly token 320 to
  token 768: the visible symptom was reasoning freezing for 6-11
  seconds while the speed readout collapsed, then a flood of text at
  once, plus a final end-of-response burst. Capped requests (every
  benchmark and QA row) never arm the safeguard, which is how three
  releases shipped it unnoticed. The holdback is now engaged only while
  the output actually shows a forming loop — healthy responses stream
  live, byte for byte, exactly like 2.7.1 — and a real runaway loop
  still gets trimmed before most of it reaches the wire.
  (`MTPLX_REPETITION_STREAM_HOLDBACK=candidate|strict|off` selects the
  new default, the 2.8.0-2.8.2 behavior, or the pre-2.8 wire.)
- **Fresh daemons no longer burn 30-60+ seconds of full-throttle GPU
  "warming" contexts chats never reach.** The turbo profile's 2.8.0
  background warm ladder walked prefills up to 32,768 tokens after
  every boot so deep-context *benchmark rows* would start warm — at the
  cost of every real user's Mac spinning up after launch (the "idle GPU
  burn" field reports), warm rungs re-firing between chat turns, and a
  chat sent mid-rung seeing multi-second time-to-first-token. The
  product ladder is back to the two rungs interactive chat actually
  touches (boot cost ~4.5 s on the 27B, matching 2.7.1); benchmark
  harnesses opt into the deep ladder with `MTPLX_WARMUP_LADDER`.
  Background warm steps now also wait for 90 seconds of request quiet
  (`MTPLX_WARMUP_IDLE_GRACE_S`) before touching the model, so warming
  never competes with an active conversation.
- **The release pillar gate now fails on streaming freezes.** Every
  existing gate capped `max_tokens`, so the entire uncapped code path —
  the one every chat client uses — was invisible to release QA. A new
  `uncapped_stream_cadence` gate sends the real uncapped streamed chat
  and fails the release on any delivered-content gap over 2 seconds.

## [2.8.2] - 2026-08-17

### Fixed

- **`mtplx start` no longer re-runs tuning on every launch.** The tune
  record is keyed by a hash of the exact tune settings, and 2.8.0's
  per-model profile work updated the settings the tuner *saves* under
  without updating the hand-built dict the start wizard *looks up* with —
  the two hashes could never match again, so every start re-offered
  "Run tuning [recommended]" and accepting it meant minutes of maxed-out
  GPU with the API port still closed. That combination is the "loads the
  model, machine heats up, requests never arrive" experience reported
  within hours of 2.8.1 (#280). Save and lookup now derive the key
  through one shared constructor, with a regression test that fails if
  they ever drift apart again.
- **The wizard tunes the model you actually picked.** A wizard pick was
  not marked as an explicit model selection, so the tuner re-resolved
  the hardware default: picking the 4B on a machine whose default is the
  27B tuned the 27B — minutes of the wrong benchmark — and then saved a
  record the launched model could never use. Wizard picks now carry the
  same explicitness markers as CLI flags.
- **A picked local model can no longer be silently swapped for the
  canonical default.** A local folder whose name matches the verified
  default (an LM Studio-style `Youssofal/Qwen…` layout, #279) was
  re-routed through default-model selection, replacing the on-disk path
  with the Hugging Face repo id — and start then demanded a download for
  a model that was already installed.
- **Idle daemons no longer hammer `session-bank/manifest.sqlite`.**
  Every `/health` and dashboard poll opened the SSD session-cache
  manifest and ran a full-table aggregate (about eight sqlite opens per
  health check, continuously visible in Activity Monitor, #280). The
  aggregate is now cached against the store's mutation generation with a
  5-second staleness bound: steady-state polling costs at most one
  manifest read per 5 s instead of dozens per second.
- **`mtplx --version` reports the right version again.** The 2.8.1
  hotfix bumped the package version but missed the CLI banner constant,
  so the published 2.8.1 wheel identified itself as `2.8.0 (2.8.1)` —
  exactly the line users check to confirm they escaped the 2.8.0 vision
  defect.

## [2.8.1] - 2026-08-17

### Fixed

- **A cached vision conversation can no longer serve another image's
  context.** Session frontiers are keyed by token ids, and every image
  placeholder shares one id, so after 2.8.0 taught vision histories to
  commit (the long-session cache fix), a request that repeated a
  transcript with a different image could restore the previous image's
  KV wholesale. Our own release gate's correctness sentinel caught it
  before the DMG went out. Vision turns now keep their full warm-cache
  reuse through the content-keyed store (identical pixels restore
  everything, different pixels never read past the image) and simply
  stop advancing the raw-id session frontier. PyPI 2.8.0 shipped with
  the defect for about an hour; 2.8.1 is the same release plus this
  fix, and it is what the desktop build carries.

## [2.8.0] - 2026-08-17

### Added

- **Sharing the API over your network is now a one-liner.** `mtplx serve
  --host 0.0.0.0 --api-key-file ~/.mtplx/api-key` creates the key file with
  a fresh key when it doesn't exist (0600, printed once) instead of dying on
  the exact recovery command our own non-localhost refusal suggests. Wildcard
  binds print a dialable `Network OpenAI API Base URL` (your Mac's LAN
  address — what Parallels/VM guests and other devices should use), `--host`
  finally has help text, and docs/server.md gained a sharing section.
  Keyless non-localhost binds still refuse: that guard is the product.
- **`/health` tells you when the engine is degraded.** A new additive
  `degradation` block reports compiled-verify state with the reason for any
  permanent-eager fallback, profile env keys an operator override beat, and
  NAX/GQA kernel bail counters. "Looks like turbo, runs slow" is no longer
  possible to miss from a harness.
- **`mtplx doctor` prints the compiled-verify fence** — mode, threshold, and
  which profile or env supplied it (#255).
- **Richer per-response `mtplx_stats`:** `finish_reason`,
  `draft_sampler_policy`/ownership/`greedy_coupled`,
  `repetition_stop_triggered` (+reason), `content_empty_reason` on capped
  thinking rows, and visible clamp stats (`context_cap_applied`,
  `effective_max_tokens`). All quiet-envelope: stamped only when they apply.
- **Streaming golden coverage and measurement guidance.** The request
  matrix gains SSE arms, and the benchmarking guide now documents response
  caps, `enable_thinking:false` for capped harnesses, streaming measurement
  rules (TTFT at first *content* delta; progress frames are spec-valid
  empty deltas), and the exact prompt-scoring array contract.
- **Warmup knobs:** the turbo profile ships `MTPLX_WARMUP_LADDER` crossing
  every compiled-verify bucket class up to its fence, and
  `MTPLX_WARMUP_PREFILL_CHUNK` is now env-tunable (default unchanged).
- **AR mode joins the session bank (#246).** `--generation-mode ar` (and
  `--no-mtp`) now restores warm prefixes through the same path MTP uses and
  reports real cache stats (`cached_tokens`, `cache_source`,
  `cache_miss_reason`, true `prompt_tps`), so an AR control arm measures
  decode alone instead of paying a full re-prefill every turn.
- **Prompt scoring on `/v1/completions`.** `echo: true` with `logprobs` and
  `max_tokens: 0` scores an entire prompt in one call: per-token logprobs,
  top-K alternatives, byte-exact `text_offset`, and a `token_ids` identity
  array. This is the lane KL-divergence quality harnesses need.
- **Per-request draft-sampler resolver.** The draft temperature resolves per
  request from the artifact stamp, the request's own sampling, and greedy
  coupling at temperature 0, with the resolved value stamped in
  `mtplx_stats` so telemetry always equals engine reality.
- **KV quantization is a real decode feature.** Paged q8 KV runs through a
  dedicated kernel on the decode path (with a dequant memo below the kernel
  threshold), and `mtplx_stats` reports the memo and kernel counters.
- **Faster streaming under load.** The SSE hot path moved to a loop-fed
  queue with a constant envelope and cheaper per-chunk encoding, lowering
  per-token server overhead at high decode speeds.
- **`/v1/messages` protocol conformance.** Parallel tool use, streamed
  usage accounting, and strict rejection of fields that were previously
  ignored silently.
- **Bridge hygiene.** The OpenCode and Pi plugins no longer delete a
  client's explicitly configured capabilities, and the no-tools stream
  filter stops eating turns (code fences are exempt from tool-markup
  suppression).

### Changed

- **Quickstart leads with "Auto (recommended)".** Auto pins nothing — the
  engine resolves the launch profile per model (Turbo for the quantized
  flagships). A previously saved default "sustained" state migrates to Auto
  once; deliberate picks (including Sustained Max) keep pinning. The macOS
  app's "auto" likewise emits no `--profile` flag anymore, so legacy hybrid
  and renamed model directories stop launching pinned to sustained.
- **Discover shows every MTPLX build.** Any repo with "mtplx" in its name
  appears (case-insensitive, any position), results are collected until the
  page fills *after* filtering instead of slicing the top-30-by-downloads
  first, and the wall/CLI default page is 100.
- **Config values are real pins.** A profile or sampler value from
  `config.toml` is honored as explicit in both directions (config
  "sustained" stays sustained; values equal to a family default are no
  longer treated as unset) and startup prints one line saying so. Injected
  launch defaults carry provenance and never override an artifact's stamp.
- **KV quantization actually saves memory now.** The q8 dequant mirror is
  offset-sized with geometric growth and is released once a request latches
  onto the kernel path; q4 never allocates a mirror. Numerics route once
  per request (a q8 request starting below the kernel threshold stays on
  dequant math for its whole life instead of switching mid-generation), the
  byte stat counts the mirror, and the CLI help states the honest contract.
- **Prompt-scoring (`/v1/completions` echo+logprobs) follows OpenAI array
  semantics:** all arrays length n with `null` at index 0, the scored token
  always present in its own top-K map, and a `token_ids` array for stable
  identity. `logprobs` without `echo` returns a 400 that names the exact
  requirement; `logprobs: 0` is a valid request.
- **`reasoning_effort: "high"` maps up the engine's real effort ladder**
  (xhigh on Qwen 3.8) instead of silently falling back to the default;
  unknown values return 400.
- **`MTPLX_CLIENT` is an observability label, not an owner.** Managed-client
  policy requires per-request evidence (headers/UA/body), so an anonymous
  benchmarker's sampling settings are honored against app- or
  hermes-launched daemons. Claude Code's `claude-cli/*` user agent is now
  recognized as a client hint.
- **Draft-sampler provenance is explicit end-to-end:** a daemon launched
  with a bare `--draft-temperature` pins it as operator-explicit, while
  family-default launches ship unpinned values that keep the temperature
  curve live.

### Fixed

- **Streamed text now concatenates to exactly the non-stream text.** The
  model separates `</think>` from its answer with a blank line, and the
  stream lane leaked that separator as content deltas while the non-stream
  lane stripped it, so diffing the two transports at temperature 0 showed a
  whitespace mismatch on every thinking response. Streamed content is
  edge-trimmed on the wire the same way the non-stream cleaner strips it;
  interior whitespace (markdown structure) is untouched, and transcript
  canonicalization stays byte-exact. One cosmetic shape remains: a stream
  that ends in a tool call can still carry a trailing blank line ahead of
  the tool call in some chunkings.
- **Batched-AR responses no longer carry draft-sampler stamps.** Under
  concurrency, requests that ride the batched AR lane merged request-policy
  observability wholesale, so an AR response could report a draft sampler
  policy plus `draft_time_s: 0.0`, the exact fabricated-stats shape the
  serial lane already scrubbed. Both batched sites now scrub the same way.
- **The launch environment can no longer steer request behavior.** Three
  branches (the OpenCode default-sampler override, the single-tool-call
  stream policy, and the post-request cache clear) keyed on a client hint
  that fell back to the daemon's `MTPLX_CLIENT` label, so an operator-set
  env var could change anonymous API callers' sampling. Behavior now keys
  on per-request evidence only; the env-inclusive hint remains as an
  observability label.
- **Prompt scoring's `top_logprobs` is correct under string collisions and
  safe for harness parsers.** When two token ids decode to the same display
  string, the scored token's entry now always carries its true logprob
  (previously a higher-ranked collision kept the other token's value, which
  inflated string-keyed KL readings), and index 0 is an empty dict instead
  of null so parsers that iterate entries with `.items()` don't crash on a
  spec-shaped response. `token_logprobs[0]` stays null.
- **`mtplx quickstart`/`serve` find branded local builds.** The canonical
  model id resolver only checked the Hub snapshot layout, so a forge-built
  pack living under its bare name (which `mtplx models` lists and bench
  model selection happily uses) was reported "not cached" with a 20 GB
  re-download suggestion. The resolver now falls back to the branded
  directory under the same contract validation.
- **`mtplx doctor --summary` prints the compiled-verify fence** like the
  full report already did.
- **The MTP lane's final session-bank commit is timed and guarded like
  AR's.** The post-response commit forward was billed into measured decode
  elapsed (understating MTP tok/s) and an allocation failure there could
  destroy a completed response; it is now outside the measured window and
  a failure downgrades to "no session commit this turn".
- **The macOS app's model-family detection matches the engine's.** The app
  still classified stock `Qwen/Qwen3-8B` as the 3.8 family (temperature 1.0
  defaults on the wrong model) and let folder names outrank forge
  provenance; it now uses the engine's boundary guard and provenance-first
  order.
- **The web chat UI no longer labels this launch's context cap as "the
  model's" context.** The max-tokens help fed `/health`'s `context_window` —
  which memory sizing can cap far below native — into a string attributing
  it to the model, so a 32GB machine serving a 256k-native model read
  "the model's 16.4k context". It now says whose number it is: this
  server's active context window.
- **Stock `Qwen/Qwen3-8B` is no longer misclassified as the Qwen 3.8
  family** (a regression in the artifact-identity fix gave Alibaba's most
  popular size the wrong sampler defaults and reasoning codec), and family
  resolution now trusts forge provenance over the folder name, so renamed
  or symlinked model directories keep their true family (#268).
- **Agentic sessions commit again.** The post-turn session commit now builds
  its prefix with the same committed-reasoning canonicalization the next
  request will send, so commits byte-extend instead of failing every turn
  ("retokenized_prefix_not_extending_session", 3–4% reuse — #269). The
  canonicalization gate also refuses on tool-call changes and dropped
  turns instead of substituting the wrong turn's reasoning, works for
  OpenCode's stripped preambles, and repair re-encodes preserve committed
  reasoning. Two system contracts became suffix contracts, ending
  full-context re-prefill when they flip.
- **A prompt that fills the context window returns a clear 400**
  (`context_length_exceeded`) instead of silently generating one token,
  and a fitting prompt whose `max_tokens` exceeds the remainder is clamped
  visibly — no more phantom 0.4 tok/s rows at 256k.
- **`finish_reason` is truthful everywhere:** a length cap beats
  `tool_calls` in non-streaming chat (matching streaming), `/v1/messages`
  maps priority `max_tokens` → `stop_sequence` → `tool_use`, the
  completions stream trims stop strings like non-stream, and empty content
  on a capped thinking row recovers or is stamped with the reason on both
  paths.
- **AR mode reports honest numbers:** no fabricated draft temperature, and
  elapsed time no longer includes the post-response session-bank forward
  pass (which also can no longer destroy a finished response on failure).
- **Draft-sampler telemetry equals engine reality.** The resolver no longer
  reports policy "none" while drafting at target temperature; the
  temperature-scale env applies before the stamp; artifact draft stamps
  survive launching under a different profile; OpenCode's server-side
  defaults are an honest `launch_default` tier with the coupling curve
  live; batch cohorts key on the target sampling triple.
- **Repetition-guard stops stay off the wire:** the streaming paths (AR,
  MTP-K, and both Gemma-4 loops) hold back a detector-window tail while
  the guard is armed, so trimmed loop garbage never reaches clients, and
  the stop is visible in public stats.
- **Benchmark rows stop paying hidden costs:** compiled-verify prewarm is
  no longer spent by the boot walk and warms the exact shared traces real
  rows dispatch; Metal memory caps and over-context refusal now apply to
  every bench, ladder, one-shot, and quickstart entry (#261) with
  per-row flushes.
- **Streams end honestly:** the post-content commit wait is bounded with
  live heartbeats and a stall watchdog, explicit cancels emit a terminal
  frame + `[DONE]`, client disconnects are tagged as such, and
  `GeneratorExit` no longer logs runtime errors.
- **No surface stamps "sustained" for a flagship anymore:** forge's
  artifact stamp, cached-model listings, doctor's support matrix, bench
  suites, tune, and the quickstart download branch all report what serve
  resolution actually picks.
- **Sizing and accounting truths:** nested MTP-sidecar layouts are sized
  correctly (restoring the RAM-aware auto budget on small Macs), the
  dashboard no longer double-counts the sidecar, the NAX install report
  reflects the real probe, the batch histogram stops counting phantom
  width-1 units, and prompt scoring decodes each unique token once instead
  of ~65k times at long context.

## [2.7.2] - 2026-08-16

An emergency fix for `mtplx pull`: on 2.7.1 and older, re-pulling a repo that
changed upstream — such as the Qwen 3.8 repos re-published on 2026-08-15 with
their vision towers restored — can corrupt the local copy. Upgrade before
pulling.

### Fixed

- **The Qwen 3.8 Hugging Face artifacts can see again (#263).** All six
  published 3.8 repos (Bare Speed, Optimized Speed, Optimized Quality and
  their FP16 siblings) shipped without their vision towers: the forge
  `mlx_lm.convert` lane serializes only the text model, so the 333
  `model.visual.*` tensors, `vision_config`, and the preprocessor sidecars
  were silently dropped. The repos were re-published on 2026-08-15 with the
  official bf16 tower grafted back as an index-registered
  `model-vision.safetensors` (byte-for-byte from `Qwen/Qwen3.8-27B`; language
  and MTP tensors untouched). Existing installs pick the delta up with
  `mtplx pull`; images, `/health` `vision.enabled`, prompt caching with
  images, and MTP decode were verified on all six builds. Thanks to
  @kjellix for the report and the proven graft procedure.
- **Forge keeps vision towers from now on.** `mtplx forge build` grafts the
  source's vision tower, `vision_config`, and preprocessor sidecars into the
  converted artifact on every lane (`mtplx/vision_graft.py`), and fails
  closed when a source that declares `vision_config` would produce a blind
  artifact. A repair script for already-forged artifacts ships as
  `scripts/graft_vision_tower.py`, and the pillar gate now asserts
  `vision.enabled` before the vision-cache check so a blind build fails
  loudly with the real cause.
- **`mtplx pull` no longer corrupts files that changed upstream (#258,
  #234).** The
  progress downloader treated a size-mismatched *complete* local file as a
  resumable partial and byte-range-appended the remote tail onto the old
  content — updating a repo in place (for example the restored 3.8 vision
  indexes) corrupted `config.json` and `model.safetensors.index.json` and
  left the local copy unloadable until re-downloaded. Stale files are now
  discarded and re-fetched whole; genuine `*.incomplete` partials still
  resume. **Users on ≤2.7.1 should upgrade before pulling repaired repos**;
  a failed pull from an older build is recovered by deleting the corrupt
  `config.json` + `model.safetensors.index.json` and pulling again.
## [2.7.1] - 2026-08-15

Clears the 2.7.0 known-issues list: `xhigh` is now selectable everywhere it is
offered, and the app's KV cache quantization toggle reaches Qwen 3.8.

### Fixed

- **`xhigh` no longer bounces back to `medium` in the app.** Choosing it in
  Inference settings while the model was running, or running
  `mtplx config set reasoning_effort xhigh`, was rejected: those two writing
  surfaces carried their own hardcoded effort list that stopped at `high`,
  while the request path already knew `xhigh`. Because the live-settings POST
  is all-or-nothing, one unknown level threw away the whole payload and the
  picker snapped back to the family default. All four writing surfaces
  (serve/CLI `--reasoning-effort`, `mtplx config set`, the live-settings POST)
  now read one shared vocabulary in `mtplx/reasoning_effort.py`, and narrowing
  to what the loaded model actually supports stays where it belongs — the
  family's own `effort_levels`.
- **The app's KV cache quantization toggle now applies to Qwen 3.8.** The
  launch gate listed only `qwen3_5`/`qwen3_6`, so a 3.8 launch exported
  nothing while Settings showed `q8`.
- **`mtplx doctor` names the model it actually checks** instead of hardcoding
  the previous default, and turbo's profile note reports the real compiled
  verify fence (32,768, raised from 12,288 in 2.7.0) rather than the stale one.
- **A built app can no longer rank below the release it supersedes.** Sparkle
  orders updates by `CFBundleVersion` alone, and the number derived from the
  version was narrow enough that 2.7.1 computed lower than the 2.7.0 that
  shipped — so a fresh build offered 2.7.0 to itself as an update. The
  derivation now has room for the whole version and the appcast reads its
  number back off the built bundle instead of being told one separately.

### Still open from 2.7.0

- With reasoning switched off in a plain chat with no tools, Qwen 3.8 can
  still emit a stray tool call and cut the turn short. Keep thinking on.

## [2.7.0] - 2026-08-15

Qwen3.8 support 🎉. Qwen3.8-27B came out on 2026-08-14; this release runs it
the way the model card says, with three tuned MTPLX builds, FP16 versions for
M1 and M2, and the compiled verify window extended to 32k. The release notes
at docs/releases/v2.7.0.md carry the full story and every measurement (all on
one M5 Max; nothing was measured on M1 or M2).

### Added

- **Qwen 3.8 model family** (`qwen3_8`) with the official inference contract:
  temperature 1.0 / top-p 0.95 / top-k 20, reasoning effort `xhigh`, `medium`
  and `low` (coding sessions default to `medium`: 51.5 s against 314.9 s at
  xhigh on the same correct agent task), thinking preserved in history by
  default, `chat_template_kwargs.enable_thinking` honored. Live serving is
  capped at depth 3 for now (depth 4 killed the daemon on drop day).
- **Three Qwen 3.8 builds.** Bare Speed (16.0 GB, flat 4-bit: quickest burst
  chat speeds, lower quality and slower on long coding tasks), Optimized
  Speed (20.4 GB, 4-bit dynamic quant: great coding speeds and good quality,
  recommended), Optimized Quality (29.4 GB, 8-bit dynamic quant: good coding
  speeds and perfect quality; KL to the bf16 teacher 0.00105). Each build
  states its recommended draft sampler, tuned depth (3) and measured peak in
  its own metadata; the runtime reads that ahead of profile fallbacks, and
  the app and CLI launch every build identically.
- **FP16 builds for M1 and M2.** Every 3.8 build has an FP16 sibling on the
  Hub (`-FP16`): the same quantized packs byte for byte, every 16-bit tensor
  cast bf16 to fp16 (99.992% exact, none overflow). The M1/M2 tier of the CLI
  and the app routes to them automatically, same picks, order and text.
- **`mtplx pull` mirror hint (#259).** A network-shaped download failure with
  no `HF_ENDPOINT` set now names the mirror knob (CLI env, or Settings,
  Advanced, HF download mirror in the app). Troubleshooting docs cover both.
- **`mtplx tune --require-max-fans`** and a Forge `module_overrides` recipe
  lane (per-module quantization in one conversion pass; it built Optimized
  Speed).

### Changed

- **Default model.** Fresh installs on M3/M4/M5 with 32 GB or more default to
  Qwen 3.8 Optimized Speed; M1/M2 get its FP16 sibling; under 32 GB still
  routes to the 9B. `mtplx quickstart`, `mtplx start` and the app's first-run
  picker offer the whole 3.8 line-up. `mtplx start` says once when the
  recommended default moved. Qwen 3.6 Optimized Speed V2 keeps turbo.
- **Compiled verify to 32k.** The compiled verify graph stopped at 12,288
  tokens of context since July; that fence's reason is gone, so turbo now
  compiles verify to 32,768 tokens (interleaved A/B on Qwen 3.8 Bare Speed:
  +6.9% at 20k, peak memory flat at 20k and lower at 30k).
  `MTPLX_COMPILED_VERIFY` can be set by hand.
- **Coding agents.** OpenCode and Pi no longer send an output cap of any kind
  for MTPLX models; Pi sessions carry their real session id so banked
  prefixes restore from RAM; the session bank's background re-render uses the
  effort the request ran with; reasoning cut off before the closing think tag
  is routed as reasoning.
- **App.** Qwen 3.8 launch family (turbo, official sampler preset, reasoning
  effort control with xhigh, Tune AR to D3, exact sizes and measured peaks);
  the first-run picker shows the trio (FP16 on M1/M2); the Qwen 3.6 Optimized
  Quality row on M1/M2 resolves to its FP16 build.
- **Thermal honesty.** Max-fan sessions hold an ownership token, so a daemon
  shutting down behind its replacement cannot switch fans back to Auto under
  it; Forge refuses to benchmark when verified max-fan mode cannot start.
- **`mtplx doctor`** judges memory against the model this Mac would actually
  default to; M5 Max listed in the support matrix.
- **Attribution is now required, not preferred.** MTPLX stays Apache-2.0, and
  the NOTICE file (which Apache-2.0 section 4(d) carries with every copy) now
  requires products built on MTPLX to show "Powered by MTPLX" inside the
  product, where a user can see it. A mention in a repo or on a website does
  not cover it.

### Fixed

- **SSD session cache CPU drain.** The cache walked its whole store on every
  write and every `/health` poll (816,220 files, 89.9 GB, 41.7 s per walk).
  Reconciliation is now maintenance: only when the store changed, at most 5%
  of the time, off the writer lock, yielding to live traffic. Idle CPU with a
  health poller 35% down to 0.2%; per-write cap gate 71 to 159 s down to
  3 to 6 s.
- **macOS 27 crash in the inference settings overlay (#256, #257).** SwiftUI 8
  traps on a slider with no distinct values; the depth and context-window
  sliders are now built only when there is something to slide. Reported and
  fixed by @joshlacal.
- Hardware detection uses absolute `/usr/sbin/sysctl` and
  `/usr/sbin/system_profiler` paths; `doctor` and `tune` no longer run `git`
  outside a repository (no Xcode Command Line Tools dialog on a clean Mac).
- Depth-default resolution honors artifact metadata across profile
  mismatches; the degrade pin and the no-metadata path both survive. The
  public depth ceiling follows the artifact reference, not the served-name
  alias; `tune` validates depths against what the model supports and takes
  its sampler from the same family contract as `serve`.
- `MTPLX_REQUEST_LOG_JSONL=1` logs to the default file instead of a file
  named `1`. xhigh boot no longer trips strict warmup.

### Known issues (fixed in 2.7.1)

- Choosing `xhigh` in the app's Inference settings while the model is running
  is rejected by the server, as is `mtplx config set reasoning_effort xhigh`;
  set it before starting the model or pass `--reasoning-effort xhigh`.
- The app's KV cache quantization toggle is not applied to Qwen 3.8 models.
- With reasoning off in a plain chat with no tools, Qwen 3.8 emitted a stray
  tool call and cut the turn short on about half of our coding prompts; keep
  thinking on (the default).

## [2.6.0] - 2026-08-11

### Added

- **Concurrent MTP serving.** Until now speculative decoding was a
  single-request feature: under concurrency the scheduler fell back to the
  autoregressive batch lane and every simultaneous caller lost the MTP
  speedup. `--scheduler-mode mtp_batch` serves independent requests through
  fixed-width MTP cohorts — each row owns its state, drafts are verified in
  one batched target forward, and rows join and leave mid-flight without
  disturbing their neighbours. Two native cohort widths install side by side
  (three-wide and eight-wide) and the scheduler seals each cohort at the
  narrowest width that fits, so two concurrent agents no longer pay the
  padding of an eight-lane launch. Measured on `Qwen3.6-35B-A3B` on an M5 Max,
  the three-wide lane holds 1.7–1.8× the per-request decode of the padded
  eight-lane shape, and the composite lane below adds warm-prefix restores on
  top; against the previous production `ar_batch` route the same concurrent
  agent workloads decode at 1.6–2.25× per lane, sampled at the model's
  shipped settings.

  The lane is exact where exactness is claimed and honest where it is not:
  `--mtp-batch-numerics` selects between `throughput`, `balanced`, and
  `b1-exact` profiles (documented trade-offs, install-time self-check), and
  per-request stats now report each row's own truth — its own accepted-depth
  histogram, its own cohort width, its own restore provenance — rather than
  cohort-averaged numbers.

  The session bank composes with the cohorts: a request whose prefix is
  banked restores it at cohort admission (skipping the shared prefill for
  the covered span), prefills only its uncovered suffix, and commits its own
  prompt boundary before the merge, so agent fleets with shared system
  prompts keep warm TTFT under concurrency. The plain `ar_batch` lane learned
  the same trick: finished rows and prompt-only boundaries commit to the bank
  from batch mode too.

- **LiquidAI LFM2 / LFM2.5 support.** The LFM2 family serves natively with a
  bit-exact ShortConv decode fast-path, a verified think/tool grammar
  (parser stamp, native tool prompt, and the pythonic streaming dialect the
  family emits), and a catalog lane for the `llama-ar` shape. IQuest-Coder
  checkpoints serve target-only AR through the same registry honesty:
  recognized, served without MTP claims, refused cleanly when a quantization
  the runtime cannot execute is detected.

- **Embedding and reranking endpoints.** `POST /v1/embeddings` (OpenAI shape)
  and `POST /v1/rerank` (Cohere/Jina shape) are now served by the same daemon
  as chat, so a retrieval-backed setup no longer needs a second inference
  server beside MTPLX. Both are opt-in through repeatable `--embedding-model`
  and `--reranker-model` flags on `serve` and `quickstart`, accepting a Hugging
  Face id or a local path with an optional `REF=served-id` alias. With no flag
  the endpoints answer 404 and the chat runtime is untouched.

  Retrieval models deliberately bypass the MTP generation path: multi-token
  prediction makes next-token decoding cheaper, which is meaningless for a
  model that emits a vector instead of a token stream. They run the transformer
  stack directly — last-token pooling with L2 normalisation for embeddings, a
  softmax over the `yes`/`no` logits for reranking, both padded on the right so
  causal attention leaves every real position bit-identical to an unpadded run.

  Backends are cached by resolved path, so listing one reference as both an
  embedder and a reranker loads a single copy of the weights and serves both
  roles from it. `--retrieval-max-resident` caps how many retrieval models stay
  in memory; beyond it the least recently used one is unloaded. Models load on
  first request, so an unused endpoint costs nothing.

  `/v1/models` keeps its default listing chat-only, so clients that enumerate
  models to build a chat picker (OpenCode, Cline, Continue, ...) are never
  offered an embedder as a conversation target. Retrieval models are listed
  via `?capability=embedding` or `?capability=rerank`, every entry carries a
  `capability` field, and a chat completion that requests a retrieval-only id
  is rejected with a clear 400 instead of being silently answered by the
  loaded chat model. The settings are configurable from the macOS app and
  persist in `~/.mtplx/config.toml` as `embedding_models`, `reranker_models`,
  and `retrieval_max_resident`.

  Checkpoints that ship their own Python inference code (the jina MLX
  releases bundle `model.py`/`rerank.py`) are only executed after an explicit
  opt-in: `--retrieval-trust-remote-code` on `serve`/`quickstart`, or
  `retrieval_trust_remote_code = true` in the config file. Without it the
  request is refused with a clear 403 naming the flag; models served by
  MTPLX's own loaders are unaffected.

  `/v1/embeddings` honours OpenAI's `dimensions` parameter with the
  Matryoshka recipe — truncate to the leading dimensions and re-normalise —
  matching how MRL-trained models like jina-embeddings-v5 (32→1024) are
  meant to be shortened. A `dimensions` beyond the model's native width is a
  clear 400 rather than a silently full-width vector that no longer fits the
  index the client sized.

### Fixed

- **Temperature-0 speculative output matches plain decoding again.** The
  MTP lane's cold prefill fed the whole prompt through the model in one
  window, while plain autoregressive decoding splits it into body plus a
  final single-token step. The two shapes round differently at the last
  bit, so the speculative lane started from a cache one ulp apart from the
  AR lane's — enough to flip greedy argmax at a near-tie and break the
  "temperature 0 output is token-identical" contract on the promotion gate.
  All cold-prefill paths now partition the prompt identically; the Optimized
  Speed V2 artifact, which surfaced the flip, passes its greedy exactness
  gate at every depth again.

- **Single-request decode speed under sampling recovered.** The eight-way
  sampling unification routed the single-request lane's per-token sampling
  through a full-vocabulary host reference (a NumPy partition and float64
  softmax over 248k logits per sampled token), which measured as a 15-19%
  decode regression against 2.5.4 on the serving lane. The single-request
  lane now selects its top-k support on device with the same deterministic
  tie-breaking as the batched route and moves only those k tokens to the
  host. The batched cohort lanes keep their float64 host reference
  unchanged.

- **Prefix restores no longer corrupt the session bank (#247).** Since the
  zero-copy cache work, restoring a banked prefix installed the entry's own
  KV state objects into the borrowing request's cache; the borrower's suffix
  prefill and decode then wrote in place into pages the entry still
  referenced. An interleaved near-prefix request could silently rewrite a
  neighbour's banked span, and every later match served the poisoned pages,
  so a byte-identical repeat could answer under another request's system
  prompt. Restores now install fresh zero-copy views: the entry's stored
  state can no longer be written through, the borrower's first divergent
  write pays the single deferred copy the design always intended, and
  commits stay zero-copy. Reproduced and fixed from the reporter's
  reproducer, which ships as a regression test.

- **Streaming tool calls no longer duplicate argument values into `content`
  (#249).** When streaming with a prior tool call and result in history, the
  initial-fragment guard stripped tool markup and forwarded the remainder,
  which is the tool call's argument values, into `delta.content`, while the
  parser separately emitted the correct `tool_calls`. Clients then echoed the
  duplicated text back into context on every turn. Orphan remainders are now
  held until tool extraction resolves: a parser-claimed span is dropped, and
  only genuinely visible prose wrapped in stray markup still surfaces.

- **Solo requests with penalties answer on the composite scheduler.** A solo
  request carrying `presence_penalty` or `frequency_penalty` on the
  `mtp_batch` daemon hit the compiled lane's sampler validator and returned
  HTTP 500, while penalty cohorts already fell back to the host route. Solo
  penalty requests now steer to the same host lane up front.

- **`mtplx serve --no-auth` actually parses (#235 follow-through).** The
  2.5.4 notes promised that exact spelling, but the flag existed only on the
  internal server module's parser; the public `mtplx serve` command rejected
  it with an argparse error before anything ran. The flag now parses on
  `serve` and is forwarded to the server. Non-localhost binds still require
  a key.

- **`qwen3_5_mtp` checkpoints validate and serve again.** The injector
  attached the MTP surface to the outer model wrapper while validation
  inspects the inner TextModel, so validation failed and the forward pass
  crashed. The surface now lives on the TextModel and is re-exposed on the
  outer wrapper by delegation. Cherry-picked from PR #242 (thanks @davidtai)
  together with its hermetic regression test.

- **Artifacts launch at their measured depth again.** The typed runtime
  contract kept only its schema fields, which silently dropped the
  measured-depth map and every artifact's declared depth default — so
  `quickstart`/`start` launched the 35B-A3B at its D3 *ceiling*, a measured
  ~22% decode loss against its fastest depth. Identity and depth defaults now
  resolve from the artifact's `mtplx_runtime.json` when the contract lacks
  them: the 35B Speed artifact launches at its measured D2, and artifacts
  declaring `recommended_mtp_depth` (Balance) are honored.

- **`--reasoning-parser` is authoritative.** A set parser was silently
  replaced by the backend's codec whenever they disagreed, which made the
  flag a no-op on shared lanes and left thinking impossible to enable for
  models whose templates fully support it. A parser resolved from family
  policy or typed by the operator now wins; the codec fills in only when no
  parser is set. Explicit `none` is still honored.

- **Metal buffer-object leak in long decodes.** mlx-lm's `ArraysCache`
  regrew Metal buffer objects on every `advance()`; over a long session that
  is unbounded growth. The fix is vendored in-tree and installed on both the
  server path and the batch lane, so pip installs against stock mlx-lm 0.31.x
  get it too.

- **Missing MTP heads degrade instead of refusing.** A checkpoint without
  `mtp_heads` now serves target-only AR (with the degrade reason surfaced,
  and propagated to the spawned server process) instead of being turned away.
  Checkpoints that ship custom Python (`auto_map`) are refused with the
  policy stated plainly — MTPLX never executes repository code — and
  quantizations MLX cannot run are refused with the offending bit-width named
  rather than failing later at load.

- **AR batch hardening.** Cache-removal errors in the batch lane now fail
  closed instead of corrupting the row; completed streams no longer starve
  behind still-running neighbours; and the vendored Metal shader cache is
  keyed by MLX ABI so an MLX upgrade cannot serve stale compiled kernels.

## [2.5.4] - 2026-08-07

Agent sessions got the attention this cycle. If you drive MTPLX from Pi,
OpenCode, or any tool-calling client, warm turns now stay warm. (This section
was backfilled from the GitHub release notes, which carry the full narrative.)

### Fixed

- **Warm-turn cache reuse in agent sessions.** Tool rounds carried a short
  transient hint that shifted the cached prefix by ~200 tokens per turn; the
  engine now records the stable boundary and restores from it directly. A
  postcommit within a bounded 0.6s of finishing is now awaited instead of
  discarded (a measured 1,449-token re-prefill became 436 tokens, cutting
  that turn's time-to-first-token from 2.7s to 1.1s). Background SSD cache
  maintenance no longer drains ahead of a starting turn (the unexplained
  ~0.8s stall at turn start), and the SSD tier skips hydrating candidates
  that cannot beat a fresher in-RAM match.
- **The session cache says what it is doing (#229, and the observability half
  of #230).** The daemon prints the resolved cache budget at startup with the
  override variables; outgrowing the per-session cap warns once with the
  numbers and the setting that raises the ceiling; `MTPLX_SESSION_BANK_MAX_BYTES`
  parses "8G"/"8GB"/"8GiB" and warns on unparseable values; the app no longer
  drops explicit cache sizes set in Settings under the "target default"
  policy.
- **Long-context decode on 32k+ sessions (#228).** The app forced a
  paged-attention route at 32k with launch-day thresholds that were never
  re-measured; reporters measured 4-7x slower decode at 43k. The app now
  defers to the engine's measured thresholds (64k on current kernels).
- **Vision sessions.** The near-prefix restore lane is capped at the first
  image token, so it can never resurrect cache computed from a different
  image's pixels.
- **Adaptive depth engages when it should.** The expected-value policy's cost
  constants now reflect measured reality on current kernels; depth-3 drafting
  fired on 13% of eligible rounds despite 65% acceptance.

### Added

- **`mtplx serve --no-auth` (#235).** Explicit auth off-switch for localhost
  binds; non-localhost binds still require a key.
- **llama.cpp-style `timings` object (#237).** Chat completion responses can
  include prompt/decode throughput for clients that read it from the response
  body (contributed).

## [2.5.3] - 2026-08-06

Small release. A day of head-to-head benchmarking against another engine
turned up a set of real latency bugs in our agent lane and a few places
where the API surface misled external tools. All of them are fixed here.

### Fixed

- Back-to-back agent requests no longer stall behind another session's
  background cache maintenance. Between requests the server commits session
  state to the reuse bank; since the 2.4 line that work could only be
  interrupted by its own session, so a request from any other session (a
  second chat, a subagent, an editor tool call) could wait out a
  multi-gigabyte commit and then decode slower on top of it. Worst measured
  hit on tight request cadences was a 44% slower follow-up turn and about
  0.75s of added first-token latency. Commits now yield the moment any
  request is admitted, whoever it belongs to. A session's own follow-up
  keeps the short grace it always had, so streaming tool-call turns still
  resolve their prefix instead of re-prefilling.
  (`MTPLX_POSTCOMMIT_CROSS_SESSION_YIELD=0` restores the old behavior.)
- Repeat requests skip prompt re-encoding. Rendering and tokenizing a long
  chat transcript costs 77-92ms per request; an exact-match cache now
  returns it in under a millisecond. Combined with the yield fix above,
  warm follow-up latency in our gate runs went from a 194-961ms band to a
  steady 65-74ms, and clean-room warm restores measure 2-3ms server-side.
  (A related opt-in knob,
  `MTPLX_COMPILED_VERIFY_POST_RESTORE_EAGER_ROUNDS`, can route the first
  verify round after a very large restore through the eager path; it ships
  off by default.)
- API responses no longer end with the "MTPLX TPS" stats footer. External
  tools counted it as model output, which added ~430ms to their timing
  windows, made token counts disagree with `usage`, and broke output
  equality checks at temperature 0. The MTPLX app and browser chat keep
  the footer. (`MTPLX_STATS_FOOTER_SCOPE=all` restores it everywhere.)
- `usage` now reports `completion_tokens_details.reasoning_tokens`, so
  clients can separate thinking tokens from visible output instead of
  inferring it from stream timing.
- Probing unknown endpoints or posting malformed bodies returns clean JSON
  errors with the right status codes instead of Python exception text.

### Changed

- Anonymous API clients now get standard OpenAI semantics for explicit
  request parameters: temperature, top_p, top_k, the thinking toggle,
  penalties, and generation mode in the request body are applied instead
  of being treated as hints. Requests that leave a field unset keep the
  server's launch and live settings, and clients MTPLX manages itself (the
  app, browser chat, configured OpenCode and editor lanes) stay
  server-owned exactly as before, so curated agent sampling is untouched.
  This closes the "temperature is ignored" class of report (#241).
  (`MTPLX_CLIENT_CONTROLS_DEFAULT=hints` restores the old policy.)
- Temperature-0 requests now run the draft sampler greedy as well, so the
  speculative window matches the target's argmax choices more often:
  depth-2 acceptance rose from .526 to .590 in our runs.

### Compatibility

- Model catalogs, model defaults, memory policy, and every managed-client
  behavior are unchanged. Each behavior change above has an environment
  switch that restores the previous policy.

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

[2.8.1]: https://github.com/youssofal/MTPLX/releases/tag/v2.8.1
[2.8.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.8.0
[2.7.2]: https://github.com/youssofal/MTPLX/releases/tag/v2.7.2
[2.7.1]: https://github.com/youssofal/MTPLX/releases/tag/v2.7.1
[2.7.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.7.0
[2.6.0]: https://github.com/youssofal/MTPLX/releases/tag/v2.6.0
[2.5.4]: https://github.com/youssofal/MTPLX/releases/tag/v2.5.4
[2.5.3]: https://github.com/youssofal/MTPLX/releases/tag/v2.5.3
[2.5.2]: https://github.com/youssofal/MTPLX/releases/tag/v2.5.2
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
