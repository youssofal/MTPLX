# Changelog

All notable user-facing changes to MTPLX. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [2.11.2] - 2026-09-06

A correctness release for every Mac that is not an M5, seven session-bank
and memory fixes for long agent sessions, native vision for Flash-Next in
Hermes, OpenCode and Pi, a verifier-depth fix for Flash-Next agent turns,
and app and CLI repairs.

### Fixed

- **Flash-Next multimodal requests keep QSA sparse attention.** Image-bearing
  requests ran dense causal attention on the belief that the reference does;
  Qwen's reference applies the sparse indexer to multimodal input with
  image-aware M-RoPE positions on the indexer's queries and pooled
  block-start keys. The engine follows it, checked against an independent
  position oracle. A 54k image conversation went from about 19 tok/s and a
  128 GiB allocator peak to a 92 GiB peak with turns completing at 43k–74k.
  Dense-path vision cache entries are not reused. Image-bearing turns still
  decode on the eager verifier (33–50 tok/s here versus 54–61 text-only);
  `MTPLX_QWEN4_VISION_QSA=0` is a diagnostic rollback.
- **Hermes, OpenCode and Pi advertise image input from the pack's metadata.**
  All three registered the model as text-only, so Hermes routed screenshots
  through its auxiliary Analyze Image tool as separate requests and OpenCode
  and Pi could not attach an image. The app and `mtplx` probe `config.json`
  and the weight index (`vision_config` plus a vision-tower weight), never
  the model name; existing text-only entries are upgraded; `/v1/models`
  carries `supports_vision` and `modalities`.
- **A warm vision turn is admitted as warm.** The pre-prefill guard compared
  raw image-pad tokens against content-keyed snapshots, projected a warm turn
  as a full miss, shed the snapshot it needed and could refuse with a 507.
  Admission uses the content-keyed identity restore uses; an anonymous image
  turn rejoins the session owning the matching snapshot
  (`session_source: vision_bank_prefix`); different pixels never adopt an
  older image's state.
- **A longer bank entry no longer erases a shorter exact prefix it cannot
  restore.** Flash-Next recurrent checkpoints sit on 2,048-token boundaries
  and the exact prompt end is not one, so the generation-final entry
  superseded the exact prompt entry and the next tool turn restored 2,048
  tokens and re-prefilled 31k. A longer entry replaces a shorter one only
  when it carries every restore point the shorter one supplies.
- **Hermes history matches the committed stream.** The profile enables
  `reasoning_echo`, so Hermes sends its reasoning back (the next tool turn
  reused 15,754 of 16,005 tokens); the cache producer, canonicalization and
  next-turn comparison strip the tool-call preamble the way Hermes' wire
  does, an intentionally empty visible answer stays empty, and a response's
  own reasoning survives an older interrupted turn. The Hermes profile also
  requests `compression.tool_image_retention: until_compaction`, which needs
  a Hermes change that is not upstream yet; current Hermes versions ignore
  the key.
- **The Flash-Next expected-value depth policy measures draft and verify
  cost per depth** when the compiled fixed-M4 verifier is engaged, instead of
  assuming a shorter eager draft is cheaper than compiled depth 3; it leaves
  one-time compilation out of the estimate and re-probes both depths. The
  rehearsal's 15k coding turn went from about 11 % to 90–96 % of cycles on
  the compiled route and from 41–47 to 56–59 tok/s (machine in use, not a
  quiet A/B). `MTPLX_ADAPTIVE_VERIFY_COST_FEEDBACK=0` restores the prior.
- **The prefill gauge and the Avg Prefill card measure the same chunk work.**
  The card averaged completed-request rates (a cached follow-up counted like
  a full prompt; setup counted as prefill: 422 shown for a 1,040 tok/s
  prompt) and the gauge preferred a setup-diluted cumulative rate (296 for a
  1,613 tok/s chunk). Both read measured chunks now; the card averages the
  last 100 and shows their peak, recorded once per chunk server-side.
- **Latency receipts are complete**: `ttft_s` includes admission and
  pending-history waits (a 39 s history-rebuild wait had vanished), decode
  tok/s uses the generator's decode time instead of charging prompt setup to
  decode twice, the context tile grows with the answer, and the verify
  waterfall shows the first live request.
- **App and Python test suites leave the user's files alone**: the
  daemon-supervisor tests overwrote the saved model and onboarding choices
  through the real settings file; synthetic Python requests entered the real
  request log and flight recorder.
- **27B on M1–M4: the flash-decoding verify route is gated by hardware
  (#459, #464, #467, #461).** 2.11 turned the route on in turbo without
  a GPU-family gate. It engages once the KV buffer reaches 8,192 tokens and
  uses the M5 GPU's tensor units; on M1–M4 GPUs it returned wrong attention
  (unrelated reasoning, imaginary tasks, mixed languages, no tool calls) and
  on macOS 15 the kernel failed to build (`MetalPerformancePrimitives.h`
  not found). The route now runs only on an M5-class GPU on macOS 26.2 or
  newer; everywhere else the packed-GQA verify kernel 2.10.2 shipped serves,
  validated at startup. `/health degradation.nax` carries `available`, the
  `gpu_family_or_os` bail counts and, new, `flash_dispatch_counters` for an
  engaged route. Rehearsed on an M5 Max with the reporters' 14k and 32k
  diff-summary prompts: the forced M1–M4 path and the native route both
  answer correctly; the route counts 66 dispatches with no bails. Item 3
  of #455 (incoherent output with MTP on, through Pi on an M3 Ultra)
  matches the route's engagement point and is not yet confirmed from that
  machine; #455's AR slowdown is not explained by this change.
- **AR-only sessions restore again (#465).** A target-only AR runtime
  (`--no-load-mtp`) banked its postcommit prefix under the `cycle` history
  policy while every lookup, the prefill store and the cache fingerprint
  said `committed`, so the bank refused the longest entry with
  `policy_mismatch` and Hermes or Pi re-prefilled the whole prompt on every
  top-level turn (14.5k tokens, about two minutes on an M1 Max). One policy
  per runtime is now derived in one place. The 90–116 s first turn after
  idle in #455 runs the same configuration on an M3 Ultra: the same
  mechanism on paper, unconfirmed from that machine.
- **The pre-prefill memory guard refuses a prompt that still projects over
  the memory limit after reclamation (#450).** It had admitted a 136k
  prompt at 105.4 GB against a 103.1 GB limit after clearing the allocator
  cache; the reporter's 128 GB Mac kernel-panicked four times. The guard
  re-projects after every reclamation step and answers with a structured
  507 before prefill, naming the projection, the limit and the uncached
  tokens; the engine keeps its sessions. `--allow-swap` keeps the operator's
  explicit choice; between the warning line and the limit nothing changes.
- **Deep conversations retain their session at recognized turn boundaries (#446).** A live
  session's committed stream carries the reasoning it streamed and clients
  resend the history without it, so from turn 2 on every request's raw
  shared prefix with its session ends where turn 1 started generating. The
  resolver accepted that match only as a fraction of the new prompt: 22,437
  shared tokens passed at 89k (25.06 %) and failed at 112k (20.1 %), the
  request minted a new anonymous session and block-restored 2,048 tokens of
  turn 1's snapshot, 105 s of prefill for a turn that had been 24 s (the
  reporter's 14,336 and 18,432-token remnants at turn 4 or 5, five chains of
  five; the same line crossed one chain in three on 2.10.x). Sessions now
  record the prompt length of every turn they generated from, and a shared
  prefix on one of those boundaries keeps the session whatever fraction of
  the prompt it is; edited histories keep the fraction rule, and unrelated
  conversations that share only a long system prompt still diverge before
  any boundary. The request log's prefix diagnostic names the rule
  (`reuse_rule`, `turn_boundary`).
- **The sparse-prefill native wheel is signed for notarization.** Its QSA
  kernel library and extension module are signed with the Developer ID,
  hardened runtime and a secure timestamp before they enter the runtime
  wheel (the app's signing pass cannot reach inside the archive), and the
  release script verifies every Mach-O in that wheel before submitting the
  app. The first 2.11.2 submission was rejected on exactly those two files.
- **Later conversations are admitted to the session bank again (#454).**
  The background-task heuristic (short answer + different system prompt)
  classified every conversation-continuing short turn as a title job and
  served it sessionless, so only the first conversation after a restart was
  banked and every later one re-prefilled at 0 % cache. Only the task shape
  (system prompt + one user turn) infers a background task now. Reproduced
  with the reporter's script: sessions two to four went from 0 % to 100 %
  cached on their third turn.
- **`mtplx run`, `mtplx ask` and one-shot `mtplx chat` work on Flash-Next
  (#463).** The one-shot path applied only the profile defaults, never the family lanes serve
  stamps, and crashed in the legacy capture walker
  (`AttributeError: 'DecoderLayer' object has no attribute
  'input_layernorm'`, then `KeyError: 'conv_states'`). It resolves the same
  runtime contract as serve and tune, and the legacy capture commit declines
  family-native captures instead of raising. Receipt: 52 tok/s, MTP depth 3.
- **The interactive terminal chat resolves the same contract.** The REPL
  (`mtplx chat` with no prompt, `mtplx start cli`) still applied only the
  profile defaults; it now runs the serve environment before the model loads.
- **The in-process generators use the family's verifier.** `mtplx run` and
  the terminal chat hardcoded the legacy capture-commit verifier over the
  batched verifier the Flash-Next contract selects; a two-turn terminal
  session on Flash-Next degenerated into repetition and a later run ended
  in a Metal GPU address fault. The resolved strategy and core now reach
  generation, as they always did in serve.
- **The terminal chat keeps reasoning in its own channel.** It stored
  `thought</think>answer` as the assistant's content, so Qwen 3.8's
  template nested that after an empty thinking block and the next turn's
  history was malformed. Reasoning and content are stored separately.
- **`--expect-python` validates the final answer.** It compiled the
  reasoning and the Markdown fence as Python and failed a valid program;
  it now splits off the reasoning with the model's codec, unwraps one
  enclosing fence, and still rejects malformed or missing programs.
- **The app's Hermes profile `.env` keeps a configured reasoning effort on
  its own line.** With Performance › Reasoning effort set, the app wrote
  `TERMINAL_CWD="…"HERMES_MTPLX_REASONING_EFFORT="xhigh"` as one statement;
  Hermes' dotenv parser rejected it ("could not parse statement") on every
  launch and silently lost both the working directory and the effort. Found
  by the release harness run through Hermes v0.21.0.
- **The Hermes profile no longer stamps `terminal.backend: local` (#460).**
  MTPLX wrote it into `~/.hermes/profiles/mtplx/config.yaml` on every
  launch, overriding a Docker sandbox the user had configured; the merge
  keeps what the user set.
- **A root Hermes sandbox choice reaches the MTPLX profile.** Hermes
  profiles do not inherit `~/.hermes/config.yaml`, so a profile with no
  `terminal.backend` of its own now receives the root `terminal` section;
  an explicit backend in the profile always wins and nothing else from the
  root config is copied. Both the app and `mtplx` write it, idempotently.
- **The setup wizard measures free space on the model store's volume
  (#466).** With `~/.mtplx/models` on an external drive (a symlink or
  `MTPLX_MODEL_DIR`) it read the home volume and refused every catalog
  model as "insufficient space". The download step and Forge share the fix.
- **A foreign-looking occupant of the daemon port is re-probed for five
  seconds before the app moves ports (#409).**
- **Installation survives a failing native-wheel selector.** When the
  optional step that picks the native sparse-prefill wheel fails, the app
  installs the bundled pure wheel and records that choice instead of
  stopping; installation health and the wheel fingerprint checks are
  unchanged.
- **`build_and_run.sh --no-launch` no longer terminates a running app.**
  It refuses to overwrite the exact running bundle and leaves every other
  MTPLX instance alone; the launch path is unchanged.
- **SSD prefix restore reads only the needed prefix and slices the committed
  MTP history to it (PR #444 by @softpudding).** Fixed verifier capacity is
  renewed for adaptive depths and copy windows; shared verifier programs are
  released when a model unloads; completed request banks are released while
  the shared programs stay.
- **The memory guard asks the live sessions and the bank for the reusable
  prefix before projecting, walks chain snapshots before giving up, and
  reclaims allocator storage before evicting useful snapshots (#447,
  reported and measured by @nomishbhardwaj).**
- The expected-value depth policy measures conditional acceptance
  correctly; prefill pipeline resolution stays out of decode and
  ineligible chunks; a stale A3B target-prefix test literal.

### Added

- **`--stream-stall-deadline-s` on `mtplx serve` and Performance › Advanced ›
  Stall watchdog in the app (#448)**, 0 disables; the Flash-Next
  sparse-prefill loops tick the owner heartbeat so a long page-in is not
  mistaken for a stall.
- **An MTP on/off verdict in the trace economics** for any depth policy:
  `mtp_pays` compares the tokens a run delivered per second with the
  matched AR rate the caller supplies, and `break_even_acceptance` is the
  acceptance at which the run would only have matched AR, under its
  observed cycle cost, non-draft output and depth mix. Proposal cycles are
  counted from the first draft position, not from verifier calls; `mtplx
  trace` reports `proposal_cycles`, `cycle_cost_ar_steps`,
  `drafts_per_cycle`, `acceptance_margin`, `break_even_basis` and `mtp_pays`.
- **Opt-in interleaved n-gram rows (#449, David Tai's layout).**
  `python -m mtplx.ngram_row_layout <table> --out <cache>` writes a derived
  cache with one 100-byte record per row, checked bit-for-bit, and
  `MTPLX_NGRAM_ROW_FILE` serves from it. Off by default; no speed claim
  until the cold-row measurement exists. `docs/diagnostics/ngram-row-cache.md`.
- **`mtplx trace --hermes-db <state.db> [--hermes-log <agent.log>]`** joins a
  Hermes session to engine receipts by token counts and completion clock;
  ambiguous joins stay unmatched. Trace charts leave missing samples and
  observation gaps out of the curve.
- **`scripts/run_harness_check.py`** runs an agent CLI with the real exit
  code recorded and a timeout counted as a failure.
- **Traces inspectable across harnesses**: exact Pi joins, retained tool
  results, prefix diagnostics kept with their request, verifier route costs,
  new tool content distinguished from reduced prefix reuse.
- **Forge extracts MTP heads stored outside the `mtp.` prefix (PR #442 by
  @stooit)**: GLM-4 MoE, GLM-5.3-Flash, DeepSeek-V3.2 and MiMo layouts;
  MiMo's output head bound instead of random; MiMo reaches tune; forge takes
  verification depths from the tune policy.
- **Flash-Next tuning uses the serving family contract (PR #457 by
  @stooit)** with real draft-cycle means and explicit budgets.
- **Sparse prefill packaged for the bundled Python with a compatible fallback
  (#423 by @humanrouter)**, validated through the real app installer.
- **`--compare-static` fixed-depth baselines for `mtp-adaptive` (PR #276 by
  @rinaldofesta)**; composer view lookup isolated to the main actor in the
  app tests (PR #372 by @PhilipJohnBasile).

## [2.11.1] - 2026-09-03

MTPLX 2.11. The artifact number is 2.11.1 because 2.11.0 was consumed by a
mis-stamped upload to PyPI on 2026-09-01 that was retracted; it contained
nothing beyond 2.10.2.

### Added

- **Agent-session release gate (`scripts/agent_session_gate.py`).** One
  command drives the coding-agent turn loop every harness ends up sending
  (OpenCode, Pi, Hermes, Claude Code, Cline share the OpenAI-compatible
  shape): a long real-code prompt, short same-session turns, an auto tool
  round and a forced-choice round with their tool results, judged from the
  engine's own per-request receipts. It fails on warm-turn dead time over
  1 s, a warm TTFT over 1.5 s on a small suffix, a bank restore under 90% of
  the prompt, a postcommit wait over 2 s, a tool-call turn whose
  generation-final snapshot was not banked in O(1), a decode drop past 20%
  of the cold turn, or any stream error — every symptom of the 2026-09-03
  OpenCode regressions, none of which a unit test or a single-request
  benchmark could see. `release_macos_v1.sh` runs it after the pillar gate
  against the same daemon (`MTPLX_RELEASE_AGENT_GATE_CONTEXT_TOKENS`,
  default 40000). The daemon's final stream chunk already carries every
  field it reads, so it works against any install with no log access.
- **Cache-miss receipts.** The flight recorder records the outcome of every
  generation-final snapshot attempt (mode, reason, both stream lengths, the
  divergence token); the committed-reasoning canonicalizer records why it
  stood aside; `MTPLX_DEBUG_POSTCOMMIT_MISMATCH_DIR=<dir>` dumps the
  generated and re-rendered token windows around a refused snapshot.
- **Light appearance for the app (#428).** A curated cream palette
  (cream ground, warm ink type, graphite chrome, matching code colors),
  not an inversion; every pair is gated on WCAG AA and the dark palette
  is byte-identical to 2.10. Settings > Behavior > Appearance: System,
  Dark, Light. The default stays Dark.
- **The app in twelve languages.** Onboarding opens with a searchable
  language step with flags (English, Simplified Chinese, Spanish, Hindi,
  Arabic, Brazilian Portuguese, French, Russian, Japanese, German, Korean,
  Indonesian); the same picker in Settings switches the app live, and
  Arabic lays the app out right-to-left. Every string goes through the
  localization layer with English as the fallback. Installs that finished
  onboarding on an earlier version are asked once, on the first launch
  after updating, so they learn the app speaks their language and where
  to change it; the pick applies live and the prompt never returns.
- **Paste images and files into chat.** ⌘V in the composer attaches
  what is on the clipboard: a screenshot or an image copied from a
  browser or Preview becomes an image attachment, a file copied in
  Finder becomes a document attachment (PDF, docx, md, txt, or an image
  file), exactly like the paperclip and drag-and-drop. Text still pastes
  as text. On a model without vision, a pasted or dropped image shows a
  card that says the model can't see images instead of riding along
  silently.
- **Choose a model folder.** The model picker's add row takes a local
  folder as well as a Hugging Face `org/repo`: a native folder chooser
  (or a typed path) is checked for a complete MTPLX install, remembered
  as a row in the picker, and selected in one step, so switching between
  models never means typing the directory again. The onboarding
  local-folder step gets the same chooser, and a folder that carries a
  catalog model's name is recognised as that model at a different
  location.
- **`--allow-swap` serves past the machine fit again (#427).** 2.10
  capped the default window at the memory plan's fit and refused prompts
  past it with a 507; operators who ran 2.9.x past the fit on purpose
  (32 GB Macs, swap accepted) lost that option. With the flag the
  default window is the model's own maximum, prompts past the fit are
  admitted, the banner and `/health` say so, and the plan still reports
  the overcommit. `MTPLX_ALLOW_SWAP=1` is the env form for the app and
  `mtplx start`.
- **Flash-Next speed lane on by default (PR #391 by @davidtai).** The
  compiled fixed-M4 verifier, batched target distributions, compiled MTP
  prepare, relaxed draft ties and the fused K/V gather are the family
  defaults for the Flash-Next geometry, with a 32-row rows-gather fence
  and FR-Spec drafting on packs whose lm_head has the Q8 group-64 layout
  it needs. Same-hour pairs against 2.10.2 with the copy lane on: round
  time -12% at 16k, -18.5% at 100k, -14% at 206k. A per-request memory
  gate hands prompts whose bank promotion would not fit back to the
  plain verify (`MTPLX_QWEN4_FIXED_M4_MAX_CONTEXT` is the operator belt).
- **Flash-Next: David's decode and prefill stack, on by default (PR #391 by
  @davidtai).** The second half of the PR, ported commit by commit under his
  name and measured on the coding cells: the two-kernel MoE routing head, the
  exact op diet, block verification in the accept loop, the fused QSA rope
  glue inside the compiled verify body, the M4 kernel trio, the PLE prefill
  lookahead (chunk k+1's n-gram rows gathered under chunk k, also on the
  restored-suffix prefill of warm agent turns), the first chunk's gather at
  request arrival, the n-gram table pre-read at model load
  (`--ngram-prewarm auto|all|off|<GiB>`), and the session bank's boundary
  shedding and protected terminal. The pre-scatter draft read serves greedy
  requests. Every item is exact (token-identical at temperature 0) and every
  key yields to an explicit export, `=0` included. 16k: 41.8 to 39.2 ms per
  round (-6.2%, -18.6% against 2.10.2), 63.2 to 68.4 tok/s; 100k: 46.7 to
  43.4 ms (-7.2%, -20.4% against 2.10.2), 54.0 to 60.9 tok/s, TTFT 117.0 to
  113.2 s; 206k: not re-measured after the release-night harness crash
  (2.11 as built: 56.2 ms, 44.8 tok/s); peak memory within 2.3 GB of 2.11 as
  built at 100k, flat at 16k.
- **27B flash-decoding verify route on in turbo** (`MTPLX_NAX_FLASH_ROUTE=1`,
  dim-split block defaults from the 72.7k and 128k sweeps).
- **Opt-in Steel sparse-GQA prefill consumer for M3** (PR #423 by
  @humanrouter), shipped as a native extension, not yet in the app bundle.
- **StreamScope two-turn copy-lane arm** so the streaming gate covers
  block-sized emits.
- **The n-gram pre-read reserves the engine's growth to its budget.** The
  automatic pre-read at model load subtracts max(KV estimate, engine budget
  minus the weights on disk) from free memory instead of the KV estimate
  alone; on a 128 GB Mac it drops from 23.4 GiB to 14.2 GiB, the amount the
  page cache can keep once a long prefill has grown the engine to its
  envelope. `tests/test_ngram_prewarm_reservation.py`.
- **Settings > Memory card (#431 @Journey0723, #427 @localbylocal).** A memory
  limit in GB and an allow-swap switch, carried into the daemon as
  `MTPLX_MEMORY_LIMIT_BYTES` / `MTPLX_ALLOW_SWAP`; the card shows the plan the
  engine computed for this Mac.
- **`finish_stop_origin` in the public stats and the request log (#414, PR
  #426 by @atirna).** A stop's commit path is diagnosable from the log alone.
- **`/health` carries `qwen4_install_reports`.** The stage-3 kernel report,
  the rope glue's per-item verdicts and the n-gram pre-read plan, so a
  default's engagement is readable without the serve log.
- **`--cors-origin` / `MTPLX_CORS_ORIGINS`.** Browser pages MTPLX serves
  itself are always allowed; a browser front-end on another origin now
  needs its origin listed here (repeatable flag, or a comma-separated
  env var the daemon inherits from `mtplx start`). `/admin` and the
  sign-in routes stay same-origin only regardless. See the CORS entry
  under Fixed.
- **`mtplx doctor` reports the Hugging Face token `mtplx pull` will use**
  (`hugging face token: from HF_TOKEN` / `` from `hf auth login` `` /
  `none`), and the JSON report carries `token_source`, `token_used_by_pull`
  and `token_policy`.

### Fixed

- **The pre-prefill memory guard no longer evicts a session's own
  restorable entry.** The 2.10.2 guard (#415) estimated a prompt's
  reusable prefix by exact match only. A follow-up turn the session bank
  serves by block prefix, such as the turn after a forced tool call, read
  as a full miss near the memory line and the guard cleared the session as
  superseded before the restore ran: on a 128 GB Mac serving Flash-Next, a
  41,901-token turn re-prefilled cold in 54 s with a 41,391-token restore
  available. The guard now asks the bank the same question the restore
  does (`SessionBank.longest_shared_prefix_tokens`, block-aligned) and pins
  such entries; its receipt carries `reusable_prefix_mode`. Found by the
  release script's agent-session gate.

- **Agent sessions on Flash-Next slowed to ~20 tok/s and stalled 5-8 s before
  every tool turn.** A 43k-token OpenCode session measured on the live daemon
  lost 146 s of a 14-minute task to three engine defects, none of them decode
  (the decode rounds ran at 65-70 tok/s throughout; the "21 tok/s" was dead
  time divided into 200-token tool turns):
  - Every request started the n-gram first-chunk gather at arrival
    (`MTPLX_QWEN4_PLE_FIRST_GATHER_EARLY`, on by default since the #391 port)
    and chained a page-warm of the *rest of the prompt's* rows — on warm
    bank turns whose prefill is 20-400 tokens, all of it waste (650k rows at
    43k), and the owner thread then blocked on the gather at scope exit:
    5-7.5 s per turn once memory pressure had evicted the table, charged to
    nothing in the receipt. The gather is declined when a RAM bank entry can
    already serve past the first chunk, and an unconsumed gather is never
    waited for. Warm-turn dead time 5-7.5 s → 0.01 s; the memory-pressure
    notices went with the page-warm storm.
  - The bank hydrated its own SSD twin on every warm turn (0.6 s each): the
    exact-prefix entry was excluded from the "RAM already serves this" bar,
    so the bar read 0 and the cold row was decoded unread.
  - The generation-final snapshot of a tool-call turn was refused on
    every OpenCode turn in the daemon's life (83/83 went to the retokenizing
    GPU re-prefill): tool arguments are not byte-stable through parse →
    client → re-render (a file ending in `\n` came back one token short),
    and a turn ending in `tool_calls` never advanced the session's committed
    stream, so the committed-think substitution could not apply to the
    turns that needed it. The committed post-think body (text + tool-call
    markup as generated) is now substituted into the re-render when the
    turn's calls match, and `tool_calls` finishes commit like `stop`. Tool
    rounds bank in O(1) and the next turn restores the whole turn (measured:
    write-turn follow-up 3,535 tokens re-prefilled / 3.7 s → 20 tokens /
    0.12 s).
  - A pending postcommit that was still prefilling when the 30 s bound
    expired was aborted and the request re-prefilled the same tokens from
    scratch (30 s waited + 38 s re-prefill after a 25k-token turn). The
    wait now extends while the job heartbeats at chunk boundaries and
    abandons only a silent job (`MTPLX_POSTCOMMIT_WAIT_STALL_S`, 15 s;
    ceiling `MTPLX_POSTCOMMIT_WAIT_CEILING_S`, 600 s).
  - A forced `tool_choice` rewrote the system contract and the bank
    identity, so one forced round re-prefilled a 41k session cold twice
    (41 s + 44 s). The clause now rides a transient trailing user turn like
    every other per-request steering text, and the bank fingerprint no
    longer carries tail-only contracts (forced choice, post-tool answer,
    read-only force-answer, Pi convergence): a transition round keeps the
    session's prefix. Forced round at 41k: 0.58 s.
  - A cold prompt of ~30k+ tokens could end with `finish_reason: "error"`
    after its whole prefill: the PLE lookahead's engagement verdict, a
    benchmark-arm assertion, raised inside a user request when the sidecar
    declined a low-entropy first chunk. Serving records the verdict
    (counter, `last_scope_status`, one warning); `MTPLX_QWEN4_PLE_PREFILL_
    LOOKAHEAD_STRICT=1` keeps the raise for measurement arms.
  Receipts: `scripts/agent_session_gate.py` (new release gate, below) at 40k
  passes end to end — warm turns 0.15 s TTFT / 0.01 s dead time / 66-84
  tok/s, auto tool round banked in O(1), forced round warm. OpenCode CLI,
  same task, bank on vs off: aggregate decode 87.1 vs 83.8 tok/s (the bank
  does not touch decode), total TTFT 4.5 s vs 14.0 s over three turns.
- **A second of dead air before every reply after a short pause.** macOS
  drops an idle process's GPU residency about 2.5 s after its last Metal
  command and rebuilds it on the next one at ~9 ms per GiB, so on
  Flash-Next (77 GiB of weights) the first prefill after any pause of a
  few seconds — every chat turn, every agent tool-call round trip — paid
  ~0.75-1.0 s before the first token (measured: a "hi" 1.17 s from the
  app, 0.08 s back-to-back). The engine now keeps its working set
  resident with a sub-millisecond kernel on the model queue once per
  second while it is attentive (a request completed in the last 10
  minutes, `MTPLX_GPU_KEEPALIVE_ATTENTIVE_S`), then parks. Same "hi"
  after 3-90 s of quiet: server time-to-first-token 0.083 s, first text
  on screen in the app 0.26 s (was 0.87 s). `/health` reports
  `gpu_keepalive`; every request record carries whether it started warm.
  `MTPLX_GPU_KEEPALIVE=0` disables it.
- **Replies typed in lurches on Flash-Next follow-ups.** The engine writes
  one stream frame per committed token, so a decode round lands as two
  frames a few milliseconds apart every ~40 ms and a context-copy block
  as a burst. The app's typewriter estimated the round gap from every
  frame, the 3 ms intra-round gaps pulled it to ~19 ms, and each round
  was then revealed on the next display tick and followed by idle frames
  — text updated on 42% of frames, in chunks, at the raw round cadence.
  Frames within one display frame now count as one arrival, the estimate
  tracks the real round gap, and a round types across it (a 100-character
  copy block over four frames instead of one).
- **Any web page could drive the local model.** The server ran CORS with a
  wildcard origin and credentials, so on the default keyless localhost bind
  a page in any tab could POST to `/v1/*`, read the answers, list and clear
  sessions under `/admin/*`, and keep the GPU busy. Browser requests are
  now same-origin by default (the page's `Origin` must match the `Host` it
  used), allowlisted origins reach the API but never `/admin` or sign-in,
  every other origin gets a 403 with no CORS headers, and requests without
  an `Origin` header (the app, OpenCode, Pi, Claude Code, Cline, curl) are
  untouched. The browser-auth cookie is `Secure` over https.
- **The API key stays out of URLs, argv and the Logs pane.** Opening the
  browser dashboard from the app used `/mtplx/browser-auth?mtplx_api_key=…`,
  so the key landed in browser history; the app now asks the daemon for a
  single-use 60 s ticket (`POST /mtplx/browser-auth/ticket`) and opens the
  URL it returns, falling back to the old form only when a daemon predates
  the route. The dashboard's own sign-in posts the key in a same-origin
  body (`POST /mtplx/browser-auth`) and a 401 shows a sign-in prompt instead
  of "Connection to MTPLX lost". The daemon reads its key from a user-only
  (0600) file under Application Support instead of `--api-key <key>` on
  argv, and the launched command line the app logs masks every `*-key`,
  `*-token`, `*-secret` and `*-password` value.
- **Play, Restart and first launch no longer wait on mtplx.com.** Every
  daemon launch awaited the release manifest on a 60 s timeout; offline,
  firewalled and captive-portal Macs sat in "Launching" for a minute per
  Play. The runtime decision is local now, the manifest fetch is bounded to
  3 s and refreshes the About card in the background, and first-run
  onboarding is decided from the saved settings alone.
- **The model-pack Update button no longer freezes the window** while it
  resolves the runtime, walks the pack and spawns the CLI; that work runs
  off the main actor.
- **A settings file with one bad value keeps every other setting.** One
  wrong-typed field in `settings.json` (a hand edit, a downgrade) used to
  make the whole file undecodable, re-run onboarding, and overwrite custom
  models, the API key and tuned records with defaults. Bad fields now fall
  back individually (logged with their path), a malformed custom model or
  tune record is skipped while its siblings load, and a file that cannot
  be read at all is kept beside itself as `settings.json.unreadable-<stamp>`
  with a banner and Reveal in Finder.
- **An unopenable chat store is kept, not silently swapped for memory.**
  `chats.store` and its `-wal`/`-shm` sidecars are renamed beside
  themselves, a fresh store starts, and the sidebar says so with Reveal in
  Finder; only if no store can be created does the session run in memory,
  and then the sidebar says that too. The chat models carry a versioned
  schema so the next model change has a migration path.
- **Unsaved Settings edits survive switching tabs.** The draft lives above
  the tab now; an "Unsaved changes" row offers Save / Apply + Restart and
  Revert. Mode still saves on pick.
- **A failed reply shows the server's error and offers Retry.** The
  daemon's `finish_reason: "error"` frame (memory guard, context overflow,
  tool-loop exception) was decoded as an ordinary finish, so the turn read
  "Interrupted reply" or "No visible answer generated." with no message and
  no Retry. The message is shown, persisted with the turn, and labels the
  settled bubble "Failed: <message>".
- **A reply the daemon never finished is filed as incomplete.** A stream
  that ended without its terminal chunk (process death, dropped
  connection) was persisted as a complete answer with `finish_reason:
  "stop"`. It is now `"incomplete"`, shown as interrupted, with Retry.
- **Chats are auto-titled in every language.** The title guard compared
  against the English literal "New Chat", so non-English users kept the
  placeholder forever; rows the old guard left untitled are named at
  launch from their first message.
- **A failed web search or fetch is recorded as a failure.** Offline or
  blocked providers came back as an empty result set marked success, the
  model was told "No results", and a failed fetch still added a source.
  Failures are marked on the live strip and the persisted trace, the model
  is told the tool failed, and no phantom source is added.
- **Esc on the chat surface does one thing.** It stops a streaming reply,
  otherwise closes the chat; Stop Generating in the menu moves to ⌘⇧. so
  two Esc bindings no longer race.
- **Attaching a file no longer freezes the app.** Extraction (PDF page
  walk, docx unzip, image decode) runs off the main actor with a per-card
  spinner; a file that cannot be read stays on the strip with the reason
  instead of vanishing. New caps: 500 PDF pages and 200,000 characters per
  attachment, noted on the card and in the text the model sees.
- **A badly typed value in `~/.mtplx/config.toml` no longer bricks every
  command.** `context_window = "64k"` made `status`, `list`, `config show`
  and everything else exit with a traceback; that one key now falls back
  with a one-line warning naming it, and `mtplx config set` refuses a bad
  value plainly without writing.
- **Ctrl-C at a prompt exits quietly** with status 130 instead of a
  traceback.
- **`mtplx bench` works from any directory.** Prompt suites resolve inside
  the installed package; a bare `mtplx bench` lists its actions.
- **Prose that quotes `[Calling tool:` no longer swallows the answer.** The
  streaming filter held everything after the marker until a `]` arrived and
  then dropped it at finish; the hold is now bounded by the call's own
  grammar, so real calls (including multi-line JSON arguments) are still
  hidden and ordinary text streams through.
- **The attached terminal chat gives up on a daemon that stops
  responding** (5 s to connect, 120 s with nothing on the wire) with a
  plain error and exit 1 instead of hanging forever.
- **One Hugging Face token policy.** `mtplx pull`, update checks, `inspect`
  and `doctor` agree: `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`, then the token
  stored by `hf auth login`, else anonymous; a stored token the Hub refuses
  is retried anonymously so a stale login never breaks a public pull. Pulls
  previously ignored the login token that `doctor` reported as present.
- **Passwordless-sudo setup validates the rule before installing it.** The
  rule is character-checked (an unescaped space silently turned the rest of
  a path into an argument, and visudo accepted it), parsed by `visudo -c -f`
  on a temp file, and installed by one privileged script that names the
  `thermalforge` binary fan control actually runs (`~/.mtplx/bin`), not a
  PATH copy; a missing binary is reported plainly; every sudo call is
  bounded and a no-tty session says to run `mtplx max --grant-sudo` in a
  terminal.
- **The fan-restore sidecar clears its marker only after the fans verifiably
  return to auto**, after the socket path and the sudo fallback alike, and
  writes what it did to `~/.mtplx/logs/thermal-sidecar.log`.
- **`mtplx models --update` never removes a live file before the new pack
  is complete.** Changed files are set aside as `.stale`, removed only after
  a complete download, and restored on any failure or interruption; a kill
  mid-update is recovered on the next run.
- **The curl installer no longer writes into Homebrew's bin.** It used to
  `cp` over the `mtplx` symlink there, replacing the Cellar program. The
  global launcher is opt-in (`MTPLX_GLOBAL_BIN`), a launcher this installer
  did not create is never replaced, and the same rule covers the legacy
  preview installer.
- **An `opencode.json` or Pi `models.json` MTPLX cannot read is left
  alone.** Both apps accept JSONC (comments, trailing commas), and MTPLX
  now reads them the same way and merges; a file that still does not parse
  is left untouched with a message naming the file and position instead of
  being moved aside and replaced. A rewrite keeps the previous file as
  `<name>.before-mtplx-<stamp>.bak` and `connect` prints where it wrote.
- **First-run routing refuses what cannot run.** An Intel Mac was told
  "selected because this is not Apple Silicon" and handed a 27B download;
  a Mac whose memory could not be read got the 27B; every Mac under 32 GiB
  got the 9B. Intel and too-small Macs now get one plain sentence and exit
  1 before any download, unreadable memory selects the smallest pack and
  says why, and under 32 GiB the CLI names the same pack the app's picker
  lists first (the 9B from 16 GiB, the 4B below that).
- **The browser dashboard:** the Speculative tab's hard-coded vLLM
  comparison is gone; settings sliders send only the keys you moved and
  follow the server otherwise; the live TPS gauge goes idle when the server
  does instead of pinning the last request's speed.
- **A resumed download can no longer splice a new commit's tail onto a
  stale partial.** `mtplx pull`, and the app's downloader behind it,
  range-resumed any `*.incomplete` partial next to the target whichever
  commit had written it and accepted the result on size alone, so a pack
  repaired in place on the Hub could come back as a corrupt file that
  loaded. A transfer marker now records the blob every file is fetched
  from: a partial whose blob changed, or one nothing vouches for, is
  discarded; a landed file whose recorded blob changed is refetched; and
  every LFS file is hashed as it lands against the sha256 the Hub
  publishes, so a mismatch discards the partial with a plain message
  instead of installing it. The progress figures are unchanged.
- **Published packs no longer name the machine that forged them.**
  `mtplx forge` stamped the local trunk directory it was pointed at into
  `mtplx_runtime.json` (`base_trunk`, `forge_provenance.source_repo`, the
  forge inputs), and the scrubber written to prevent that had no caller,
  so the flagship 27B packs carried a home directory. The stamper names a
  local trunk by its Hub identity (the pull marker's repo id and commit,
  or the `owner--name` cache layout), `forge publish` uploads scrubbed
  copies of every top-level JSON document that carries a local path, and
  `mtplx model publish-check` gains a `no_local_paths` gate with
  `--scrub` to rewrite staged documents in place. The scrubber itself no
  longer mistakes every string that starts with a slash for a path.
- **The browser dashboard's Thermal tab no longer shows an internal
  benchmarking rule.** On a default install (thermal polling off) every
  generation raised a banner saying that per the project's Universal
  Thermal Rule model work should run under verified max-fan mode. The
  banner is gone; the fan panel keeps its note about
  `--enable-thermal-poll`.
- **macOS 27 no longer answers long prompts with a 500 (#404, #405, #407).**
  The macOS 27 Metal Performance Primitives header rejects the
  address-space-qualified cooperative-tensor operands the QSA sparse
  prefill kernels and the 27B flash-decoding route used, so 2.10.x on the
  27 betas failed every prompt past the ~32K sparse-prefill crossover
  mid-request ("Unable to build metal library from source"). The seven
  kernel sites use the address-space-neutral operand types, and a startup
  probe dispatches the real sparse-prefill pipeline once, degrading to
  dense prefill with a diagnostic on an SDK that refuses it. Credit
  @mrmurphy (first patch), @sunnybluesea (root cause, three-site sweep,
  macOS 27 receipts), reporters @DigiJoe79 and @rameshn007.
- **An explicit `MTPLX_MEMORY_LIMIT_BYTES` is the engine budget both ways
  (#443 @yermakovm).** The planner clamped the configured Metal limit under
  its own 75% envelope, so a 96 GB Mac serving the 69.2 GiB Flash-Next
  Bare-Speed pack was refused as "does not fit" (72 GiB budget against a
  73.3 GiB minimum) even under a limit of 80G that runs it. An operator-set
  limit now plans the engine budget up to the machine's RAM, the banner
  says "(Metal limit)", and the "does not fit" note names the lever. A
  `--memory-budget` below the machine still wins, so a simulated smaller
  seat stays small.
- **Flash-Next no longer 500s under concurrency in the `ar_batch` lane (#420).**
  mlx-lm's batch generator merges every batched prompt's caches and the
  family's QSA cache has no merge, so two concurrent requests under
  `--scheduler-mode ar_batch` (the app's lane) raised an unhandled
  `ValueError` while sequential requests were fine. The lane now probes
  the model's cache family at startup, says so in the banner and in
  `/health` (`scheduler.ar_batch_unavailable_reason`), and serves
  concurrent requests one at a time on the solo lane instead. The 27B
  keeps batching. `MTPLX_AR_BATCH_CACHE_PROBE=0` skips the probe.
- **A client-capped one-token answer is diagnosable from the serve log
  (#436).** Pi caps `max_tokens` at its model `contextWindow` minus a
  chars/4 estimate of the transcript, and sends `max_tokens=1` when the
  estimate overflows; the answer then stops after one token, often inside
  a tool call, and Pi reports a truncated response. The server now logs
  one WARNING naming the client's cap, and the `mtplx_openai_generation`
  trace line carries `max_tokens`, `effective_max_tokens` and
  `finish_reason`.
- **Anthropic `/v1/messages` image blocks reach the vision tower (#441).**
  `image` blocks (base64 or URL, including ones nested inside a
  `tool_result`) become image parts instead of base64 prose, so Claude
  Code's pasted screenshots and its Read tool on image files are seen
  instead of hallucinated at 13x to 52x the token cost. Text-only
  requests render byte-identically; `count_tokens` counts the vision
  placeholder the way the chat path does.
- **Copy-lane streaming no longer pastes and freezes.** The server
  releases block-sized token batches piece by piece across the round
  (`MTPLX_STREAM_PACER`, default on; StreamScope copy-lane arm burst p95
  58 to 9 characters), and the app's typewriter rates arrivals over a
  wall-clock window and types each block across its round
  (95th-percentile flush 65 to 25 characters, longest on-screen pause
  767 to 233 ms on the same flow).
- **Fully accepted copy blocks skip the recurrent-state replay** in the
  Flash-Next family commit; the Route Tape records every copy-lane round.
- **`reasoning_effort` accepts the off-ish values some clients send**
  (PR #433 by @sypsyp97).
- **App CPU:** the decode chip publishes only on change, metrics
  snapshots slow to 500 ms while a turn streams, and the redundant
  auto-scroll task is gone.
- **The chat composer scrolls past ten lines instead of jumping to the top
  (#424, PR #437 by @MohammedThowfiq).**
- **The Performance mode survives a model restart and the launch logs the
  effective scheduling (#398 @variablefate).** The picker is a saved global;
  Settings shows "Running now: ..." and the log pane carries the
  `--scheduler-mode` / `--batching-preset` the daemon launched with.
- **A small request from another session no longer kills a marathon
  postcommit (#432 @nomishbhardwaj).** `MTPLX_POSTCOMMIT_MARATHON_PROTECT_TOKENS`
  reaches the cross-session abort with a bounded grace (one 30 s window per
  landed commit, keyed to the session), and the pending record is seeded
  from the committed frontier at arm time.
- **A draining MTPLX is not mistaken for a foreign process on its port, and a
  configured port never moves to +1 on a transient occupant (#409 @kmei3560;
  CLI lane; the app's own port handling is unchanged in this release).**
- **A streamed response closes within 30 s of its last token when another
  request's prefill is queued ahead of its session snapshot (#425
  @66duke66).** `MTPLX_STREAM_COMMIT_WAIT_MAX_S` bounds the post-generation
  commit wait; the snapshot lands in the background and the next turn waits
  for it through the pending-postcommit path.
- **The compiled fixed-M4 lane no longer skips long prompts because the
  allocator cache looked like live memory.** After a 100k prefill on a 128 GB
  Mac the gate read the freed prefill scratch held by the allocator as live
  and fell back to the plain verify; it now releases that cache when it
  stands between the request and the lane and re-reads.
- **The download panel counts only the files the repo ships.** Progress
  was the byte count of the whole model folder, so shards from a
  superseded revision or staging leftovers from an interrupted Hugging
  Face transfer counted as downloaded: the panel read 47.38 GB of
  18.52 GB at 100 percent while still downloading. The daemon's progress
  events, the resume decision and the disk headroom check now use the
  repo's manifest, and the app never prints past the repo size. The
  leftovers are reported instead (`stale_bytes`, `stale_files` on the
  events and in `mtplx pull`), and a stray `*.incomplete` next to landed
  weights no longer keeps a byte-complete folder "partial" through every
  Retry: only a needed file's partial with no final copy is a transfer.

## [2.10.2] - 2026-09-01

### Fixed

- **Memory refusals are now honest and proactive (#415).** Before a large
  prefill is admitted, the server projects its footprint, proactively
  clears superseded SessionBank entries, and, when the request genuinely
  cannot fit, answers a structured HTTP 507 upfront instead of letting
  the stream die mid-flight. Streamed requests that fail now emit an
  honest error receipt (`stream_error`, `error_kind`) instead of being
  logged as client cancellations.
- **Anthropic `/v1/messages` usage no longer double-counts cached
  prefixes.** `input_tokens` now excludes `cache_read_input_tokens`
  (Anthropic's fields are disjoint; OpenAI's are cumulative), so Claude
  Code and Pi context/auto-compaction math sees true totals on session
  cache hits. Adopted from PR #417 by @amichaelblock-lgtm.
- **Claude Code no longer times out on long prefills.** Claude Code's
  stream watchdog resets only on real message events and ignores SSE
  comments and protocol pings, so a first turn past its 300 s idle
  window (large MCP toolsets reach 137k to 165k tokens) died with
  "Stream idle timeout". The Anthropic bridge now emits prefill
  keep-alives as empty `thinking_delta` events; a measured 165k-token
  first turn on the 27B now survives a multi-minute prefill and
  completes.
- **Compile kill-switch precedence (from PR #395 by @maceip).** An
  explicit `MTPLX_COMPILED_GDN=0` / `MTPLX_QWEN4EXP_COMPILE=0` now wins
  over profile auto-arming everywhere, including `set_ar_pipeline_mode`
  re-arming; the two flags are aliases with explicit-off precedence.
- **Stop-cause telemetry (#414).** Generation stats now record
  `finish_stop_origin` distinguishing model EOS, stop sequences, length
  caps, and repetition stops, so early-stop reports are diagnosable from
  request logs.
- **Source-checkout onboarding leaves the global install alone.** Running
  the app from a source checkout no longer tries to upgrade or repoint
  the user's terminal `mtplx` installation, and QA builds prefer their
  explicitly allowed source wrapper over stale app-managed or Homebrew
  runtimes (production bundle precedence unchanged). The `--kv-quant`
  help text now describes the shipped packed-quant q4 kernel.

### Added

- **Dark lanes for measured techniques** (off by default, receipts in-tree):
  double-buffered AR decode via `mx.async_eval` (`MTPLX_ASYNC_AR=1`,
  ported from PR #396 by @maceip; measured flat at product cells) and
  M-batched fused MoE GLU verify kernels (`MTPLX_FUSED_MOE_VERIFY=1`;
  bit-identical per token, measured slower at verify widths, kept as
  wiring platform).

## [2.10.1] - 2026-08-30

### Added

- **Sparse prefill lane for Flash-Next.** Adapts the fused Metal kernels
  from PR #397 by @maceip: the model's own block scores drive a
  block-sparse FlashAttention prefill instead of full attention masks.
  On an M5 Max against 2.10.0: a 98k-token prompt processes in 114.5 s
  instead of 175.7 s with peak memory down from 91.4 to 83.0 GB, a
  131k-token prompt measures 810 tok/s, and a 262,144-token cold prompt
  completes in 355 s at 87.4 GB peak where 2.10.0 climbed to 119 GB and
  produced nothing (#393, reported by @blackjose007-stack). Auto-enabled
  on M4/M5 generation GPUs for prompts past 32k tokens; other machines
  keep the dense path; `MTPLX_QSA_PREFILL=0` disables. The kernels also
  handle YaRN rope scaling.
- **Image input for Flash-Next.** The packs always shipped their vision
  weights; the runtime now serves them. Native multimodal position
  encoding (M-RoPE) lands for the family, so images work in app chat,
  over the API, and in Pi (#328, reported by @nmqanh). Pi model configs
  written by an older MTPLX upgrade in place and user edits survive.

### Fixed

- **Flash-Next loads on 96 GB Macs.** The preload memory check used
  reserve constants tuned on 128 GB machines and refused packs that
  actually fit; the reserve now scales with total RAM (#400, reported by
  @JordiPosthumus). 128 GB and larger machines are unchanged.
- **M2/M3 GPU crash at launch.** Three Metal kernels requested
  1,024-thread threadgroups; M2/M3 generation GPUs cap these kernels at
  896 threads and refused them. A one-shot startup probe now routes
  those machines to the standard kernels instead of crashing (#400).
- **Honest Flash-Next memory admission.** The plan now prices the
  family's QSA caches and prefill transients, over-window requests
  answer a structured HTTP 507 instead of wedging the machine under
  memory pressure, and the QSA cache grows geometrically instead of by
  quadratic reallocation (#393).
- **Empty first answers with tools declared.** A first turn that ended
  entirely inside the reasoning channel returned an empty message when
  tools were present (agent clients, chat with web search). The server
  now continues the turn to a visible answer, and a failed continuation
  returns the first pass instead of an error.
- **Greedy Turbo exactness on the 27B.** Temperature-0 verification
  routes to stock kernels, restoring token-exact agreement between
  Turbo and plain decode.
- **App runtime self-heal.** The app proves its Python runtime imports
  before trusting it and rebuilds it from the bundled wheel on failure,
  ending the crash loop that survived app reinstalls because the broken
  runtime folder was kept.
- **Decode speed receipts.** Session-restore time was attributed to
  decode, understating decode speed on warm long-context turns.
  Attribution is corrected; generation behavior is unchanged.

## [2.10.0] - 2026-08-28

### Added

- **Long generations hold their decode speed.** A 34k-token uncapped chat
  answer decayed from 86 to 25 tok/s inside one request because every
  long-generation guard keyed off prompt length: the draft head's committed
  history cache grew unbounded during decode, and the allocator clear-cache
  cadence never armed for short prompts (8.6 GB of MLX allocator cache in a
  single request). The history cache now resets and regrows after 16,384
  live-appended tokens (`MTPLX_MTP_HISTORY_LIVE_RESET_THRESHOLD`, keyed on
  appends so restored session seeds are never dropped; output stream
  unchanged by the verify contract), and the clear-cache cadence arms
  mid-generation when live context crosses the threshold (0.6 GB on the
  same workload with it armed).
- **Context-copy block rounds on the batched verify lane.** Flash-Next's
  lane never had the copy mechanic, so file rewrites decoded at plain MTP
  depth even with every output token already in the prompt. A prompt n-gram
  match now proposes up to a 24-token block through the lane's normal
  verify forward with the identical probability-ratio acceptance, so
  sampling behavior is unchanged. Two-turn rewrite receipt: rewrite turn
  87.6 tok/s vs 73.8 for the fresh build, 2,400 of 3,963 tokens from 177
  copy rounds, zero copy rounds on novel text.
  `MTPLX_CONTEXT_COPY_BATCHED=0` disables.
- **Decay observability.** Receipts carry the context-copy counters and the
  live-reset fields; `MTPLX_DROP_EVENTS=0` launches also get a per-round
  `round_timing_ms` series. The growth-lever envs (clear-cache cadence,
  history window and live reset, verify snapshot, family capture-commit,
  drop-events) join the operator-beats-profile list.
- **Reasoning effort works on Flash-Next, everywhere.** The `qwen4_exp`
  family now declares the same reasoning codec as the dense 27B (Qwen
  think-tag parser; effort levels `xhigh` / `medium` / `low`; modes
  auto/on/off), so `reasoning_effort` on a request is honored instead of
  silently dropped — previously the family resolved "no levels", the
  field was discarded before it was read, and the chat template's own
  `xhigh` fallback burned thousands of thinking tokens with no way to
  turn it down. The family default is `xhigh`; the 27B keeps its
  measured `medium` coding default. Full surface parity in the same
  pass: the app's effort picker renders and persists for the family,
  `mtplx run` and `mtplx chat` gain `--reasoning-effort`, and
  `/v1/messages` (Anthropic-bridge) requests now forward the flat
  `reasoning_effort` field for every family instead of dropping it.
- **The n-gram sidecar stops costing 30 GB of RAM on paper.** The 32 GB
  Flash-Next n-gram table streams from SSD by default, but the memory
  plan, session-bank budget, Metal floor, and the app's memory card all
  still counted it as wired weights — a 128 GB Mac printed a false
  "MODEL DOES NOT FIT", resolved a 30 GB-pessimistic context window, and
  auto-budgeted the session bank 30 GB too small. One policy now drives
  gather behavior and every accounting surface; the serve banner and the
  app's Memory Detail card say `n-gram table 29.8G streamed from SSD
  (not wired)`. A new hot-row LRU (`MTPLX_NGRAM_HOT_MB`, default 1024)
  keeps decode-sized gathers in RAM, byte-identical by construction and
  by test — measured against the previously shipping streamed default:
  +5-10% AR and +7.5-16% MTP decode, at identical memory. In the
  product default config the MTP register now meets or beats the 30
  GB-wired resident pin (+2.4%), so the wired mode remains only a
  bench pin for 160 GB+ machines (`MTPLX_NGRAM_RESIDENT=1`).
- **Both Flash-Next packs are public on Hugging Face** —
  `Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed` and
  `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed` — and first-run
  onboarding now offers them on Macs that fit them, right behind the
  27B trio, resolving straight to the published repos.

- **Flash-Next turbo, first-class.** The two Qwen 3.8 Flash-Next serve
  packs now resolve the **turbo** launch profile by default across
  `mtplx start` / `serve` / `quickstart` / the app — the same measured
  launch rule the quantized 27B/9B flagships follow. On this family turbo
  rides the qwen4_exp fast lane (pipelined AR decode, compiled GDN,
  layer-owned capture-commit, and the fused hyper-read/GDN kernels below)
  rather than the 27B NAX verify patch, which stays off for the family
  until it earns its own measured win. Both packs' canonical ids
  (`mtplx-flash-next-bare-speed`, `mtplx-flash-next-optimized-speed`)
  now resolve from HF ids, folder names, and `mtplx pull` aliases;
  derivative or renamed packs deliberately do not inherit the ids.
- **Verify-width GDN conv kernel.** The conv + SiLU + L2-norm chain that
  speculative verify blocks previously ran as an eager op sequence now runs
  as one Metal dispatch for blocks of up to six rows, with a sliding
  in-block conv window, and stays fully compatible with the family's
  capture-commit rollback (it emits the exact rows the stash retains).
  Default-on for the family after two boot-triple A/B batteries in both
  orders (+3.1% and +2.3% MTP decode; the fused arm posted the campaign's
  best arm mean).
- **One-dispatch GDN decode step.** The whole GatedDeltaNet decode
  step — causal conv + SiLU + L2 norm, decay/beta gates, the fp32 delta
  recurrence, and the gated-RMS output epilogue — now runs as a single
  Metal dispatch between the in/out projections, cutting a GDN layer
  from ~6 GPU sends to 3. Default-on for the family after two
  boot-triple A/B batteries in both arm orders (+1.7% and +2.2% decode,
  the confirm triple's fused arm holding the six fastest rows).
- **Qwen 3.8 Flash-Next, day-0 native.** A new first-class model family
  (`qwen4_exp`): the 125B-A6B Qwen4-generation preview with GDN hybrid
  MoE, Qwen Sparse Attention, and the 32 GB n-gram memory sidecar —
  served by an in-tree MLX backend that is parity-exact against the
  reference implementation, with the native MTP draft head running
  through MTPLX's standard speculative lane (measured 1.6-1.7× over AR
  through the real server). Two packs: **Bare Speed** (flat 4-bit,
  fastest) and **Optimized Speed** (dynamic quant with 8-bit attention,
  higher quality). The n-gram table streams from SSD by default, so the
  packs fit 96 GB+ Macs with headroom; both appear in the app and CLI
  model pickers on machines that fit them, with the family's own serving
  contract (temperature 1.0, adaptive draft depth) applied end to end.
- `mtplx forge verify --stamp`: records a first-load smoke baseline and
  writes the pack's `mtplx_runtime.json` in place — the step that turns a
  "family-compatible-unverified" model into a verified one — without
  rebuilding or copying the artifact. Families the tune instrument cannot
  measure (Flash-Next today) take their rows through a locally booted
  `mtplx serve`, the lane that actually applies their family contract.
- **The CLI's first 90 seconds behave like 2026.** The terminal chat
  gets readline line editing with a persistent history
  (`~/.mtplx/history`, 1000 entries; arrow keys used to print raw
  escape sequences into the prompt); prompts pipe in
  (`echo "..." | mtplx run`, and a piped prompt into the chat entry
  answers through the same path `--prompt` uses, while an empty pipe
  keeps the non-tty refusal); `serve`/`run`/`chat` join the main help
  and `list`/`remove`/`config`/`env`/`dashboard`/`integrate` get a
  "Server and scripting" help group, with a one-line difflib
  "Did you mean" on any typo across all registered subcommands (exit
  code still 2); `mtplx ask` joins `run`/`chat` on
  `--reasoning-effort` and the chat REPL gains `/effort
  <level|status>`, live per turn like `/reasoning`; `mtplx --version`
  drops the redundant parenthetical when display and package versions
  match; and `mtplx hardware` stops printing "hardware acceleration
  confirmed: false" at humans for a field that means "not profiled"
  (JSON output unchanged everywhere).
- Streaming endpoints (`/v1/chat/completions`, `/v1/completions`,
  `/v1/messages`) emit a `: keep-alive` SSE comment every 5 seconds
  while a stream is still silent before its first token (#358). Long
  prefills — minutes at 32k+ prompts — previously put zero liveness
  bytes on the wire, so strict client/proxy read-timeouts (Claude
  Code, Cursor, Open WebUI, nginx, cloudflared) dropped the
  connection mid-compute. SSE comments are ignored by every compliant
  parser; once tokens flow the comments stop. Disable with
  `MTPLX_SSE_HEARTBEAT=0`; tune the cadence with
  `MTPLX_SSE_HEARTBEAT_INTERVAL_S` (minimum 1s).
- **Machine memory governor** (issue #305). MTPLX now plans its memory
  against the Mac it is actually on instead of assuming a 128 GB studio
  machine. At startup the serve banner prints the machine plan (engine
  budget, weights, resolved context window, session-bank budget); the
  default context window is the largest one whose full-window KV really
  fits (a 48 GB Mac serving the Speed model defaults to 196,608 tokens
  instead of a physically impossible 262,144 — 128 GB machines are
  unchanged), and an explicit `--context-window` above the fit still wins
  but is flagged loudly. The session bank stays idle-aggressive and
  yields dynamically as a long-context request's KV materializes, ahead
  of any swap; macOS pressure and an earlier allocator-relative signal
  drive the existing shedding guard. `/health`, the dashboard snapshot
  and the app carry `memory_plan`, guard events, and a pressure banner.
  `MTPLX_MEMORY_BUDGET=48G` reproduces a real 48 GB seat exactly (test-pinned).
- **Streaming SSD spill for large sessions** (issues #305, #323). Sessions
  above the per-session RAM cap — exactly the 100k+-token coding-agent
  sessions whose re-prefill costs minutes — now persist to the SSD tier
  through a tensor-by-tensor streaming writer (bounded memory, same
  on-disk format), instead of silently losing durability. Live-ref-only
  sessions reach the SSD tier for the first time; every remaining skip is
  recorded, never silent — including the disk-headroom size cap
  (`min(configured cap, free_disk/4)`), which now prints one console line
  per session naming the entry size, the effective cap, and free disk
  when it refuses a spill (found live on a 4 TB disk at 28 GiB free,
  which caps the lane at ~7 GiB and mutely excluded every 100k+-token
  session — the same silence class as #278, in a brand-new lane). A
  request arriving mid-write makes the streaming encode abort cleanly
  and re-dispatch for the next idle window (counted in
  `encode_yields_foreground`), so a spill in progress can never make a
  request wait; the dedicated writer thread keeps its own 600 s
  foreground pause, where waiting is free because nothing queues behind
  that thread.

- **RAMP: opt-in long-block and fuzzy re-anchor policy for context copy**
  (adapting community PR #375 by @johninthewinter). A fixed 48-token copy
  block replaces the confidence ladder and an exact n-gram miss falls
  through to a mismatch-tolerant short-anchor re-match, for edit-shaped
  temperature-0 agent workloads where the author measured +45.9 to +53.9
  percent decode with byte-identical output. Our own paired temperature-1
  chat-rewrite arms measured it a net loss (copy supply fell 504 to 123
  tokens on seed-identical streams), so it ships off by default
  (`MTPLX_RAMP_ENABLED`); off is byte-for-byte the prior proposer.
- **One-sync greedy draft read on confidence lanes** (adapting community
  PR #288 by @ArthurOstapenko). Margin-gate and adaptive-depth lanes now
  read the greedy draft token and its confidence metrics in one GPU
  synchronization instead of two; the author's paired receipt is +0.559
  percent geomean on the ExpectedValue depth-3 lane. The default greedy
  chain and every sampled-draft lane are untouched, proven by an
  engagement counter that reads zero there.
- **QSA rows-gather lane for verify widths** (adapting the per-query
  gather and GQA head-group broadcast from community PR #380 by @maceip).
  Multi-row QSA forwards previously staged a dense [rows, context] mask
  and read the full KV in all 12 QSA layers every verify round, a cost
  that grows with the generation. The opt-in lane gathers each row's
  selected blocks plus its visible tail at a constant width instead, with
  context-length and row-count routing fences so short contexts keep the
  fused dense path. Family default ON for Flash-Next (self-fenced to
  2..8 rows at 16384+ tokens of context, so shorter contexts are
  bit-identical dense; `MTPLX_QSA_GATHER=0` is the kill switch). Paired
  16,384-token receipt: the dense path's verify cost grew 37.0 to 46.9 ms
  per round across the run while the gather arm held flat 45.3 to 45.9,
  finishing its last window at 64.5 tok/s against 36.0 dense.
  Parity-tested against the dense path on the real layer.

### Fixed

- **The context-copy lane earns its block size before spending it.** Copy
  rounds verify 16-24-token candidate blocks — a ~4x-cost forward versus a
  normal round — and the acceptance gate needed four sampled rounds per
  generation before it could suspend, so short coding-agent turns re-paid
  the full misfire cost every turn (measured: whole-turn verify 60-88
  ms/round at 8k context with 21/96 copy tokens accepted, decode down to
  ~27 tok/s). Blocks now stay at 8 tokens
  (`MTPLX_CONTEXT_COPY_PROBATION_K`) until the turn's acceptance EMA
  proves the content pays, and the suspension arms one round earlier.
  Long-context re-emission — where the lane is a measured +16.7% — opens
  to full blocks by its third round and keeps its win. The batched verify
  lane (Flash-Next's copy mechanic, above) carries the identical
  probation contract.
- **Greedy draft coupling engages on launch-default-greedy servers.** The
  draft-sampler resolver received the raw request `temperature` — `None`
  when a client omits the field — so a server launched with
  `--temperature 0` serving such a client decoded greedily while drafts
  stayed at the family default (1.0): the silent sampled-draft acceptance
  collapse ([79/65/42]% by depth vs [96/87/76]% coupled), with no
  `draft_sampler_greedy_coupled` stamp to show for it. Both serve lanes
  now hand the resolver the effective sampler temperature, matching the
  resolver's documented contract; explicit-temperature requests were
  never affected.
- **SSD spills no longer fire mid-turn: the writer's foreground pause now
  outlasts a long coding turn.** The writer already stood down while a
  request was in flight, but its liveness bound was 60 s — shorter than a
  typical agent turn (60–620 s) — so every multi-GB session spill fired
  under the live decode: a ~1 GB/min unified-memory drumbeat that tripped
  macOS memory pressure (the Live-tab banner) and stole decode (measured
  −30% when a write overlapped a turn). The bound is now 600 s
  (`MTPLX_SSD_WRITER_FOREGROUND_PAUSE_MAX_S`); writes drain in the gaps
  between turns, and a bound that still expires into live traffic is
  counted in `/admin/cache/ssd` as `writer_pause_expired_busy`. The cold
  tier is a cache — waiting out a turn costs delayed durability, never
  correctness.
- **The session bank yields ahead of the prefill spike, so deep turns stop
  tripping the memory banner.** The dynamic ceiling reserved a static 3 GiB
  for generation transients, but a deep chunked prefill measures up to
  12.4 GiB of peak-over-active — so on long coding sessions the bank kept
  entries while the allocator peak kissed 99%+ of the Metal limit, firing
  the warning banner on every deep turn. The ceiling now reserves the
  spike this process has actually observed (clamped between 3 GiB and
  half the post-weights memory, at most 16 GiB), demoting idle entries
  to SSD before the next spike can slam the ceiling.
- **The memory banner now names the culprit.** The daemon reports which
  signal produced the pressure level (`memory_pressure_source`: system-wide
  macOS pressure vs this engine's allocator near its Metal limit, plus the
  live `allocator_fraction`), and the app's warning banner distinguishes
  "System memory pressure" (another process allocating; decode can dip
  until it passes — nothing was evicted) from the engine's own
  "Memory running high". Receipt for the split: an external 26 GB
  allocation storm dropped decode 65→22 tok/s with the engine's guard
  correctly doing nothing at all — the old copy blamed the engine for
  weather it didn't make. Attribution only claims what it can prove:
  when macOS and the allocator report pressure in the same tick the
  banner names the allocator (the "another process" copy asserts the
  engine's footprint is steady, which is false at a tie), and when the
  allocator probe cannot read at all the source reports `unknown` and
  the banner stays neutral.
- **Long sessions persist to SSD: the writer's backlog budget no longer
  rejects a snapshot bigger than itself** (#384, thanks @sapiens77 for a
  forensic-grade report). The SSD writer bounds queued bytes at 4 GiB by
  default, but a single snapshot larger than the whole budget failed
  admission on every attempt even with an empty queue — at roughly 84
  KB/token of 27B KV that turned the SSD tier silently off past ~50k
  tokens, with only a counter as the trace. An empty queue is not backlog
  pressure: a lone oversized entry is now admitted, logged by name, and
  counted in `/admin/cache/ssd` as `admitted_oversized_alone`; a genuinely
  backlogged writer still rejects. The reporter measured 41.7x TTFT across
  a restart (282 s to 6.8 s) once writes could land. Also from the same
  thread: `request_session_source` now reports the header source instead
  of null for header-identified sessions.
- **The memory governor's ceiling never evicts the live session's cache.**
  The dynamic bank ceiling subtracts an instantaneous working-set reading,
  so a deep prefill's transient allocator spike read as a standing
  commitment: on a 93k-token coding session the ceiling walked the session
  bank to zero bytes mid-request, evicting the in-flight session's own
  prefix entries, and every following agent turn re-prefilled from scratch
  (TTFT 54 to 57 s, prefill 909 down to 175 tok/s). The ceiling now
  squeezes idle sessions only; the active session's prefix chain survives
  even when the bank stays above target. Real macOS or allocator pressure
  keeps its take-anything eviction semantics, so the 48 GB swap-death
  protection is unchanged.
- **Flash-Next serves coding agents at its official sampler.** The app's
  OpenCode and Hermes launch presets pre-fill the Qwen3.6-era coding
  sampler (temperature 0.6), and the Flash-Next model defaults left those
  slots untouched, so the daemon booted with an explicit `--temperature
  0.6` that suppressed the pack-stamp injection — OpenCode requests were
  normalized to a 0.6 target against the family's official 1.0, with the
  draft still at the stamp's 1.0 (a mismatched verify pair). Flash-Next
  now clears the target-preset sampler slots on every launch target, so
  the zero-flag boot path injects the artifact's stamped 1.0/0.95/20 for
  target and draft alike, identical to a bare `mtplx serve`. The dense
  27B was already correct (its preset pins the official triple); 3.6-era
  models keep their measured 0.6 lane.
- **OpenCode's effort picker shows the whole Flash-Next dial.** The
  generated config only declared effort variants for tiers outside
  OpenCode's built-in list, trusting the client to surface the rest — but
  OpenCode Desktop 1.18.21 does not offer its full built-in list for
  custom openai-compatible providers, so xhigh (the Flash-Next chat
  default) was missing from the picker entirely. Every family tier is now
  written as an explicit variant, which renders on every OpenCode version
  and still merges over same-named built-ins; out-of-family tiers stay
  disabled.
- **The KV-quantization control explains itself on Flash-Next.** The
  setting showed the anonymous "not supported for this model" line because
  the family had no entry in the KV-quant policy table. The policy is
  unchanged — the validated q8/q4 paged lane is wired to the dense-27B
  attention call sites, not to Flash-Next's QSA layers — but the app now
  states the architecture truth: the hybrid design keeps KV on 12 of 48
  layers (~24 KB/token), and a quantized QSA lane has no validation
  receipts yet. Qwen 3.5/3.6/3.8 keep the full q8/q4 control.
- **The warm ladder yields to live traffic and no longer stamps the
  traffic clock** (adapting community PR #300 by @Blakeolson21). A request
  that had arrived but not yet completed read as an idle daemon, so the
  first request of a serve could share the GPU with a warm rung; and warm
  rungs stamped `last_request_at`, making /health report user traffic on
  an untouched daemon while the ladder deferred against its own output.
  Rung admission now checks live and queued foreground work as its own
  branch (safe at zero idle grace), and warm generations no longer move
  the request clock or counters anywhere.
- **The Live tab's acceptance panel no longer goes blank after a finished
  request.** Two stacked causes: the daemon's idle warmup ladder published
  its rungs into the dashboard's `latest` slot, replacing the finished
  request's receipt (a warmup row with a null request id sat where the
  user's acceptance counters belonged), and the app's Live-tab gates only
  ever ticked on `.completed` stream frames, which have no replay, so one
  frame missed during a reconnect kept the panel on its placeholder
  forever. Warmup rows now stay out of the dashboard ring server side, the
  app refuses to merge a warmup row over a real receipt, and a finished
  request observed through the snapshot poller counts as completion
  evidence (deduplicated per request), so the panel lights up within one
  poll of a request finishing regardless of stream health.
- **The compact tool contract no longer drops trailing tools** (#376,
  adapting community PR #379 by @ArctifoxNL). When the "Declared tools
  and schemas" line exceeded its 1200-character budget it was raw
  byte-cut, deleting whole tool names at the tail (`task` first, in
  Claude Code-shaped toolsets) — and the contract's own "never invent an
  undeclared tool" clause then made the model treat every dropped tool
  as nonexistent, killing subagents. Over budget, every declared tool
  name is now kept and only per-tool signature detail is shed.
- **Cancellation errors name their real cause** (#381). One shared
  per-request cancel flag is tripped by several unrelated paths — the
  `POST /v1/mtplx/cancel` endpoint, client disconnect, stop-sequence
  completion, tool-call finalization, the stall watchdog, stream
  teardown — and the terminal frame blamed every non-disconnect trip on
  the POST endpoint, framing an endpoint nobody called. The first
  origin is now recorded when the flag trips, terminal frames and the
  cancellation metric report it, and an unattributed trip says so
  instead of inventing a caller.
- `mtplx connect opencode` now actually writes `~/.config/opencode/opencode.json`
  (merge-preserving: other providers and plugins survive). It previously built
  the config, printed the config path, and wrote nothing — so a `--model-id`
  for a newly served model never reached OpenCode's provider models map and
  runs failed with `ProviderModelNotFoundError` surfaced as "Unexpected
  server error". Found wiring Flash-Next day-0.
- The serve daemon and tune/bench children now start Python with `-P`,
  so the directory you launch from can never shadow the installed
  runtime. Previously, running `mtplx serve` from any folder containing
  an `mtplx/` package (a source checkout, a vendored copy) silently
  served that folder's code instead of the installed release.
- **Unexecuted tool calls are no longer silently swallowed** (#349). A
  fresh-install user asking the built-in chat about their files saw the
  model "invoke" tools (`ls`, `find`, `search_files`, `read_file`) and
  get nothing back — no output, no error — because the server deleted
  dead tool-call markup from no-tools responses (#160) and from
  undeclared-tool fallbacks without telling anyone. Suppressed calls
  now leave a short, visible notice naming the tool and stating that
  nothing ran, this chat has no file/terminal access, and a coding
  agent (Claude Code, OpenCode, Hermes) connected to MTPLX provides
  it. The model reads the same notice in its history and stops
  claiming it ran tools; the reply is never blank. Code-fenced tool
  syntax examples are untouched, and `mtplx_stats` gains
  `unexecuted_tool_call_notice` for triage. The macOS app also
  persists a truthful `tool_not_executed` result for any tool call
  that finishes past the chat's tool-round budget, so replayed
  transcripts never show the model an unanswered tool call.
- **SSD session-cache writes no longer starve on an idle server, and
  shutdown flushes them** (issue #290). The scheduler's durability lane
  was only reachable while the idle band was completely empty, so any
  self-chaining background occupant (the background warm ladder) could
  hold SSD writes off forever — entries sat in RAM with every cold-tier
  counter at zero and vanished on restart, costing a full re-prefill.
  The server now pumps the durability lane within seconds of going
  request-idle (foreground work still always wins, and the pump disarms
  the moment a request arrives), and a plain SIGTERM/Ctrl-C gives
  pending writes a bounded best-effort flush (default 10 s,
  `MTPLX_SHUTDOWN_SSD_FLUSH_S` overrides, `0` disables) with one honest
  console line — including the write the SSD writer thread already has
  in flight, which the old shutdown killed mid-file.
- Metal allocation failures are answered as structured
  `insufficient_memory` (HTTP 507) errors with actionable advice, after
  the engine sheds its caches — instead of anonymous `internal_error`
  500s that left the next request to hit the same wall (issue #348 class).
- `--memory-budget 48G` (the bare-suffix spelling MTPLX's own messages
  advertise) crashed serve startup with a raw ValueError; single-letter
  and terabyte size suffixes parse now.
- The dashboard "RAM session cache" settings no longer invent `8G/4G`
  when nothing is configured — they report the budgets the engine
  actually resolved.
- The update dialog's release notes are readable in dark mode (#367).
  The generated notes page now declares `color-scheme: light dark` and
  pairs each appearance with readable text colors instead of shipping a
  light-only stylesheet that Sparkle's dark update window rendered as
  near-black-on-dark. The release script and the Sparkle rehearsal kit
  render through one shared template
  (`scripts/render_release_notes.py`), so the rehearsal now shows the
  exact page users get and the two can no longer drift apart.
  Notes pages already published under mtplx.com/releases/notes/ need a
  one-time re-render and re-upload to pick this up.
- **Reasoning history preserves by default on Flash-Next.** The
  `qwen4_exp` family ships the byte-identical Qwen 3.8 chat template and
  the same preserve-by-default trained contract, but the auto policy
  only recognized `qwen3_8` and dropped the family onto the scoped
  fallback. Every agent round's session-cache postcommit then aborted
  with `reasoning_history_scoping_mismatch` and re-prefilled the whole
  assistant turn. With preserve on, mid-session agent turns cost about
  20 new prefill tokens at 0.11 s first token (scoped paid 300 to 1500
  tokens at 1.8 to 2.2 s, measured on the release wall-clock rig).
- **Coding-agent lanes default to medium reasoning effort on
  Flash-Next.** OpenCode and Pi config writers resolve the family's new
  agent-lane default (codec `default_agent_effort`) instead of the chat
  default. On the identical multi-file coding task, xhigh measured
  150.2 s wall clock against 44.2 s at medium with the same correct
  output. Chat surfaces keep xhigh; both clients' effort pickers still
  offer every level per request.
- **The Pi provider merge owns the transport contract.** The
  user-preserving config merge kept an older MTPLX's
  `supportsReasoningEffort: false` alive across every re-sync, which
  silently killed Pi's effort dial after an upgrade. MTPLX's own
  compatibility keys now update on sync; user-added keys still survive.
- **The app's offline settings fallback shows family truth.** With the
  engine stopped, the inference settings panel fell back to a generic
  temperature 0.6 and the label "Custom model" for Flash-Next (and could
  persist that 0.6 over the engine's 1.0). The fallback table now
  carries the family's native 1.0 / 0.95 / 20 and the proper family
  name; live-daemon state was always correct.
- **Onboarding verifies the terminal command through the login shell.**
  The setup step graded whatever executable the app's own process PATH
  found, and a Finder-launched app never inherits the shell rc's
  `/opt/homebrew/bin` ordering, so setup could certify "up to date"
  while the user's actual terminal still ran an older Homebrew install.
  The check now asks the user's login shell which executable wins and
  grades that one.
- **Each conversation streams on its own turn stream** (#324). Switching
  conversations mid-generation no longer cross-wires or blanks either
  turn.
- **Explicit performance settings are honored over client-target
  defaults** (#325), and the native Chat launch target stops silently
  ignoring "Handle multiple at once".
- **The app accepts custom Hugging Face models the engine reports as
  runnable** (#359). Install completeness is judged by the source repo's
  own manifest instead of requiring an `mtplx_runtime.json`.
- **Forge routes official NVIDIA Nemotron-H configs** by deriving the
  MTP pattern from `mtp_layers_block_type` (#341); load no longer
  crashes with an AttributeError.
- **`mtplx remove` is fenced to the models cache and asks first.** The
  removal path ran `rmtree` on whatever the ref resolved to: a bare
  `.` or `/` resolved to the models cache directory itself and `..` to
  the whole `~/.mtplx` home (bin, config, session bank, every model),
  deleted without a word and exit 0. A ref must now resolve to a direct
  child of the models cache or the command refuses, a tty gets a
  confirmation naming the resolved path and size, `--yes` skips it for
  scripts, and a non-tty run without `--yes` refuses with the hint.
- **A malformed `config.toml` no longer bricks every command.** A
  truncated or hand-edited config raised a raw TOMLDecodeError through
  `status`, `doctor`, even `stop`, so the one file meant to hold
  preferences could lock the user out of the CLI entirely. The loader
  now degrades to defaults with one stderr line naming the file, the
  parse error, and `mtplx config show`; a bad value for a single key
  degrades that key only.
- **The daemonless CLI generate lanes couple the draft greedy under a
  greedy target.** One-shot `run`, the terminal chat, and `tune` call
  the engine directly and never pass the server's draft-sampler
  resolver, so `--temperature 0` kept the pack's stamped sampled draft
  (temperature 1.0) and paid the sampled-draft acceptance collapse
  ([79/65/42]% by depth vs [96/87/76]% coupled) on exactly the lane
  outside benchmarks run. The lanes now share one coupling helper; a
  user-typed `--draft-temperature` still wins, and a spec-less lane
  already mirrored the target and is unchanged.

## [2.9.3] - 2026-08-26 (internal build — never published; ships as part of 2.10.0)

### Fixed

- **Agent turns died with a fabricated
  `request cancelled via POST /v1/mtplx/cancel`** (issues #332, #343). After
  a complete streamed tool call the server cancels its own generation to end
  the turn; when the worker's acknowledgement outlived the stream loop's
  250 ms poll, the loop misread its own cancel as a foreign
  `POST /v1/mtplx/cancel` and killed the healthy turn with that error — no
  one ever called the endpoint. Tool-calling streams now drain until the
  worker acknowledges and end with the `tool_calls` terminal frame; a real
  cancel and a real disconnect behave as before. Deterministic regression
  test included. Only tool-calling turns were affected, which is why plain
  streaming never reproduced it.
- **Long-context decode past 128k, improved.** Past 131,072 prompt tokens dense decode
  silently repaged into a cache layout that structurally excluded the packed
  verify kernel, collapsing speculative decode to plain-AR speed (12.0 tok/s
  at 147k on an M5 Max 128 GB). The dense-decode ceiling is now memory-aware
  (`MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=auto`, floored at the old
  131,072 literal so smaller machines cannot regress), keeping the packed
  lane engaged (16.3 tok/s at 147k; 18.4 with copy speculation off). The
  resolver announces itself in the serve log, and an exported ceiling env
  beats the profile instead of being silently stomped back.
- **KV-cache quantization was effectively broken and is now usable.**
  Enabling q8/q4 crashed serving at warmup; q4 additionally re-dequantized
  the entire prefix every round, and quantized caches were refused by the
  compiled verify bank. All three fixed: the crash is gone, q4 routes
  through the exact packed-quant kernel via a persistent quantized bank,
  and the verify bank promotes quantized paged caches. Measured cost at 16k
  on an M5 Max vs off: q8 ≈ −4% decode, q4 ≈ −19% (down from crash / −50%).
  Still opt-in — the remaining gap to zero-loss, and the slow quantized
  lane past the dense ceiling, are known and being worked. A pre-existing
  q4 numerics defect at head_dim 128 is fenced fail-closed (the shipped
  head_dim 256 family is exact).
- The dynamic-offset paged verify kernel had never compiled since its
  introduction (a pointer-cast Metal bug hidden behind its own mask gate)
  and crashed at q_len > 5 once compiled; both fixed. Still opt-in
  (`MTPLX_PAGED_TAILMASK_ELIDE`) pending its serve verdict, but the crash
  class is removed.
- Warmup failures log their full traceback instead of a one-line skip.

### Changed

- **mlx floor raised to 0.32.2.** A clean same-wheel A/B on an M5 Max
  measured 0.32.0 → 0.32.2 at +29% decode / +41% prefill at 88k context
  (+31% at 16k); the floor converges existing installs onto that stack.
- Every request row now records packed-route bail counters and
  paged-adapter engagement, and `MTPLX_ROUTE_DEBUG=1` prints one line per
  layer naming the attention branch taken and every gate input — the
  decode cliff hid for months because fast lanes declined silently.

### Added

- Release-pin regression tests: the shipped profiles' fast-lane values
  (compiled-verify ceiling, packed verify kernel, dense-decode ceiling),
  operator-env-beats-profile precedence, and the mlx floor are now pinned
  by tests so a silent lane loss fails CI instead of surfacing in a
  benchmark weeks later.
- Experimental, off by default: `--scheduler-mode hyper` (single-user
  manufactured-concurrency chassis, trajectory-sha-exact vs serial),
  `MTPLX_NAX_TILE_ROUTE` (first M5 tensor-unit attention kernel at decode
  shapes), `MTPLX_ADAPTIVE_DTEMP`, `MTPLX_CCOPY_BANK_ROUTE`,
  `MTPLX_FORKEV_TELEMETRY`.

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
