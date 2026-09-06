# Long-Context Prefill Handoff — Qwen3.8 Flash-Next (qwen4_exp)

> **For agentic workers:** REQUIRED SUB-SKILL: use
> superpowers-optimized:subagent-driven-development to execute this document task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking. Every claim below carries the artifact or
> command that produced it; re-measure before you believe any absolute number.

**Goal:** Fix the four engine-side defects that made long-context prefill unusable on the
`qwen4_exp` (Qwen3.8 Flash-Next) architecture, and make them measurable in CI without a live
115 GB model load.

**Architecture:** Prefill on this arch tracks tokens already in context instead of tokens newly
processed (a 556-token turn costs 0.96 s at 6.5 k context and 2.40–2.86 s at 103 k — see
"Repetition noise" below), and no fast attention lane ever engages. The work is: (1) correct the KV
geometry constant the memory policy reasons from, (2) make the paged/QSA lane reachable and
provable-reachable through counters, (3) replace the hard 16,384-token refusal with a routed
degrade, and (4) give the prefill benchmark a serve-path mode so measurement stops being
in-process-only.

**Tech Stack:** Python 3.13, MLX/Metal, pytest, uv, the guarded `/tmp/mtplx-gpu-exclusive.lock`
convention from `docs/plans/2026-08-09-qwen35b-mtp-batch-numerics-profiles.md`.

**Assumptions:**

- Assumes the two packs named below are the only `qwen4_exp` targets that matter for this work.
- Assumes the M3 Ultra (512 GB, no M5 TensorOps path) is the reference machine; every number here
  is that machine and must be re-measured before being quoted elsewhere.
- Assumes a serve-path benchmark may be added to `prefill_bench.py`; without it, none of the
  acceptance criteria below can be tested without a 115 GB in-process load.
- Assumes `paged_gqa_sdpa_calls == 0` means "lane not used". It is read from the owned-attention
  instrument (`mtplx/generation.py:1069`), so a refactor that moves that plumbing changes the
  meaning of the counter — re-verify the plumbing before trusting the number as a test oracle.

## Version provenance — read this first

| | value |
|---|---|
| Everything measured against | installed **mtplx 2.11.0** (`~/.mtplx/bin/mtplx --version`) |
| This checkout at measurement time | **2.9.0** (`pyproject.toml`), branch `issue-308`, no `qwen4_exp` descriptor |
| After the 2026-09-06 rebase | `main` = upstream `406b5f7` (**2.11.1**), `issue-308` = `d6e4a5a` (DFlash2) rebased on it |
| Line numbers in this document | valid on the **rebased** tree, verified by grep after the rebase |

The rebase mattered: at 2.9.0 none of the symbols below existed, so an earlier draft of this
document was unusable. Re-run the anchors (`grep -n`) before patching if the tree moves again.
Residual skew: measured on 2.11.0, tree is 2.11.1 — so **Task 0 re-baselines before any fix is
judged.**

## Measured baseline (the thing to beat)

Tracked receipt: **`docs/perf/receipts/qwen38-flash-next-prefill.md`** — provenance, method, all
tables, cross-checks, and SHA-256 of every raw artifact. The raw JSONs stay local
(`~/.mtplx/bench/`; `agent-output/` is git-excluded via `.git/info/exclude`, so nothing there
reaches a clone). Reproduced below so the plan is readable without opening the receipt.

`kind` is derived from the engine's `cached_tokens` (`0` → `cold`); `prefix_hit` is one ~600-token
turn appended to the same prefix; throughput is `new_prefill_tokens / prompt_eval_time_s`.

9002 = `Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed`, profile turbo, depth 3, sidecar
resident (raw: `bench/prefill-probe-9002-baseline-20260905-220920.json`):

| kind | prompt | new | prompt_eval_s | new tok/s | peak |
|---|---|---|---|---|---|
| cold | 6,500 | 6,500 | 8.012 | 811 | 112.6 GiB |
| prefix_hit | 7,049 | 556 | 0.960 | 579 | 112.6 GiB |
| cold | 25,745 | 25,745 | 27.965 | 921 | 117.7 GiB |
| prefix_hit | 26,294 | 556 | 1.220 | 456 | 117.7 GiB |
| cold | 51,405 | 51,405 | 69.276 | 742 | 124.1 GiB |
| prefix_hit | 51,954 | 556 | 1.615 | 344 | 124.1 GiB |
| cold | 102,731 | 102,731 | **267.920** | **383** | 135.0 GiB |
| prefix_hit | 103,280 | 556 | **2.397** | **232** | 135.0 GiB |

9001 = `grant-ai/Qwen3.8-Flash-Next-Abliterated-MTPLX-4bit`, profile sustained, depth 1, sidecar
streamed (raw: `bench/prefill-probe-9001-baseline-20260905-222454.json`): cold 103,714 → 268.637 s
at 386 tok/s; 675-token follow-up at 104,382 → 2.835 s at 238 tok/s.

**Instrument warning, learned the hard way.** `prompt_eval_time_s` (engine) and the probe's terminal
`wall=` line (client — includes HTTP, tokenisation and teardown) differ by up to 18 %: the first
version of the table above carried the client numbers, which inflated the follow-up decay from 2.5×
to 2.9× and shifted the derived geometry. Quote the engine field, and diff any transcribed table
against the JSON before it leaves the editor.

### Repetition noise — what these numbers can and cannot prove

Four measurements of the *same* nominal work — one ~556–679-token follow-up at ~103 k context —
taken across the baseline and arms A/C/D: 2.397, 2.853, 2.757, 2.860 s → **median 2.805 s,
stdev 0.218 s, spread 17.0 %**, and the baseline's 2.397 s is the **low outlier**. Cold prefill at
the same context across three of those runs: 267.920 / 268.162 / 267.003 s → **stdev 0.611 s,
spread 0.43 %.**

Consequences that bind everything below:

- Cold-path claims are solid at ±0.5 %: the profile, window, chunk and ceiling results all stand.
- Follow-up claims are **not** resolvable below ~20 % from single runs. "Follow-up unchanged" is
  true for A/C/D against each other (2.757–2.860 s) and false against the baseline's 2.397 s. The
  honest decay against the 0.960 s at 6.5 k is **2.50×–2.98×**, and the single 2.5× figure that
  earlier drafts of this document (and its receipt) quoted was taken from the fastest of four.
- Acceptance criterion 2 therefore requires ≥3 repetitions judged on the median.

The two reported symptoms are both in that table: a 103 k cold start costs 268 s, and the same
556-token turn costs 0.96 s at 6.5 k context and 2.40 s at 103 k (2.5×). Every `prompt_eval_s` here
is the engine field, never the client wall time.

## Defects to fix

### D1 — the 16,384-token guard returns HTTP 500 on any non-sustained profile

- `mtplx/generation.py:1205` `_unsafe_long_context_prefill_guard_tokens()` defaults to **16384**;
  `:1219` `_assert_safe_long_context_prefill()` raises `RuntimeError("Blocked unsafe long-context
  MTP prefill path: …")`; called at `:4957`.
- `mtplx/profiles.py`: `SUSTAINED_PREFILL_ENV` at `:517`; `stable` at `:734` and
  `performance-cold` do **not** merge it, `sustained` and `turbo` (`:787`) do.
- Observed: 9001 on its own contract's `recommended_profile: stable` answered 500 at **26,727
  prompt tokens**; after switching to `sustained`, the identical request measured 915.5 tok/s.
- Consequence: a model pack can ship a runtime contract whose recommended profile cannot serve
  agent-sized prompts at all. The guard should route to the sustained prefill path, or the
  contract validator should reject the profile at startup — not fail per request.

### D2 — prefill tracks retained context; no fast lane ever engages

- `paged_gqa_sdpa_calls`, `prefill_partitioned_paged_calls`, `prefill_dense_fallback_calls`,
  `attention_dense_fallback_calls` all read **0** at every rung on both packs.
- `prefill_attention_impl` and `prefill_layout` are absent from the request-log rows entirely.
- 262,144 is advertised by the packs' `mtplx_runtime.json` and by
  `context_window_policy` in `/mtplx/settings`, so users will hit the tail.

### D3 — the dense-decode ceiling reasons from a KV constant 3.25–3.81× low

- `mtplx/generation.py:1499` hard-codes the default `MTPLX_DENSE_KV_BYTES_PER_TOKEN` to **65536**;
  `:1502` applies `MTPLX_DENSE_DECODE_RAM_PERCENT` default **15**; `:1533` emits the serve-log
  line `dense-decode ceiling auto: 262144 tokens (15% RAM over 65536 B/token — MODEL DEFAULT, set
  MTPLX_DENSE_KV_BYTES_PER_TOKEN for non-Qwen3.8 geometry)`. Other sites: `memory_plan.py:396`,
  `server/openai.py:3313`, `:3414`, `:3422`.
- Measured marginal from `peak_memory_bytes` across rungs: **249,526 B/token (244 KiB)** on the
  resident-sidecar port and **213,309 B/token (208 KiB)** on the streamed one.
- The engine's own refusal message agrees on magnitude: KV on **12 of 48 layers at ~24 KB/token**
  (≈288 KB/token).
- Corrected arithmetic: 262,144 needs **172.0 GiB** (9002) / **136.0 GiB** (9001) against the
  192.0 GiB engine budget — reachable, ~20.0 GiB of headroom on 9002. And at the corrected constant
  15 % of 512 GiB still permits **330,480 tokens** (9001's geometry: 386,591), above the 262,144
  window, so **correcting the constant does not by itself make the paged lane fire** — that is
  D2's job.

### D4 — `bench prefill-ladder` cannot run this arch, and has no serve-path mode

- Crash: `prefill_bench.py:850 run_prefill_ladder` → `generate_mtpk` → `graphbank`
  `forward_ar_capture` → `_fallback` → `AttributeError: 'DecoderLayer' object has no attribute
  'input_layernorm'`. Preceded by `compiled-verify prewarm {"skipped":
  ["promotion_failure:empty_kv_cache"], "complete": false}`, produced at
  `mtplx/graphbank.py:957`.
- The unguarded attribute appears at `mtplx/gdn_capture.py:2892`, `:2948`, `:3033` (also
  `mtp_patch.py:905`, `laguna_compiled_step.py:677`, `benchmarks/runners/verify_profile.py:206`).
  `qwen4_exp` layers do not carry that attribute name.
- `prefill_bench.py` has no url/port/harness handling, so the ladder can only load in-process;
  `rt = load(getattr(args, "model"), mtp=True)` at `prefill_bench.py:1108`. Result: measuring the
  serve path needed an external script (`~/.mtplx/scripts/prefill-probe.py`), which is a symptom.

### D5 — KV quantization is refused for this family

- `mtplx/backends/descriptors.py:481` carries the refusal text ("…attention has no validated
  quantized-cache lane yet."); the server printed the Flash-Next-specific refusal and downgraded
  `q8` → `off` silently at load. `:147` is the generic `kv_quant_unsupported_reason` string.
- Worth deciding: silent downgrade is fine as a default, but the served `/mtplx/settings`
  response should report the *effective* mode, not the requested one.

### D6 — test isolation: `test_public_cli.py` leaks into `test_generation_sustained.py`

- Reproduced: `pytest tests/test_generation_sustained.py` alone → green. Same file **after**
  `tests/test_public_cli.py` → **6 failures**:
  `test_lazy_bonus_verify_shortens_full_accept_verify_input`,
  `test_lazy_target_distributions_inline_bonus_avoids_bonus_reforward`,
  `test_lazy_target_distributions_stop_after_first_rejection`,
  `test_lazy_bonus_verify_skips_d1_by_default`,
  `test_omit_speculative_bonus_skips_bonus_distribution_row`,
  `test_trim_commit_keeps_rejected_verify_prefix_without_reforward` — with
  `assert [1, 4] == [1, 3, 1]` at `tests/test_generation_sustained.py:716` and IndexErrors at
  `:747`, `:777`, `:845`.
- **Verified pre-existing**: identical failures from an isolated `main` worktree at `406b5f7`, so
  it is not a rebase artifact.
- Also: tests asserting `DEFAULT_HF_MODEL_ID` (`test_public_cli.py::test_tune_default_dry_run_is_not_legacy_models_path`,
  `::test_quickstart_default_missing_cache_is_not_legacy_models_path`) read the **real**
  `~/.mtplx/config.toml` and fail on any machine whose default model differs. They pass under
  `HOME=$tmp MTPLX_HOME=$tmp/.mtplx`. Pin the environment in those tests.

## Levers already eliminated — do not re-test

| lever | result | evidence |
|---|---|---|
| profile turbo vs sustained | no effect on prefill (267.92 s vs 268.64 s at 103 k) | the two baselines above |
| `--context-window` 262,144 → 131,072 | 0.09 % on cold (268.162 s), follow-up within its 17 % noise band; saves 11.7 GiB | arm A |
| `--paged-kv-quantization q8` | refused by the engine, downgraded to `off` | D5 |
| `--prefill-chunk-tokens` 8192 | **17.9 % slower** (315.868 s) and +22.7 GiB | arm C |
| `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT=32768` | −0.34 % (267.003 s); the auto ceiling line disappears from the log when set explicitly, so the override was read | arm D |

MTPLX's own comment at `mtplx/profiles.py:525` records dense decode as the *faster* side of that
fence ("decode cliff: 12.0 -> 18.44 tok/s once dense decode holds"), and `:530` sets
`MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT: "auto"`. That is consistent with arm D being flat rather
than better: anything that forces repaging low is expected to lose.

## File structure

- Modify `mtplx/generation.py` — guard at `:1205-1219`, call at `:4957`, counter plumbing at
  `:1069`, KV geometry at `:1499` and `:1502`, serve-log line at `:1533`.
- Modify `mtplx/profiles.py` — `SUSTAINED_PREFILL_ENV` (`:517`), the ceiling default (`:530`), and
  the profile table (`:734` stable, `:787` turbo).
- Modify `mtplx/memory_plan.py` (`:396`) and `mtplx/server/openai.py` (`:3313`, `:3414`, `:3422`) —
  the ceiling consumers and the `Memory plan` line.
- Modify `mtplx/gdn_capture.py` — unguarded `layer.input_layernorm` at `:2892`, `:2948`, `:3033`.
- Modify `mtplx/backends/descriptors.py` — `qwen4_exp` family at `:499`, `:565`, `:585`, `:1249-1254`,
  `:1306`, `:1343`; KV-quant refusal text at `:481`.
- Modify `mtplx/graphbank.py` — `failures["empty_kv_cache"]` at `:957`, the promotion failure that
  preceded the ladder crash.
- Modify `mtplx/prefill_bench.py` — add a serve-path harness (`run_prefill_ladder` at `:850`).
- Modify `tests/test_generation_sustained.py`, `tests/test_public_cli.py` — environment isolation
  (D6).
- Create `tests/test_qwen4_prefill_lane.py` — the counter oracle for D2.
- Create `tests/test_long_context_profile_guard.py` — D1's startup-validation behaviour.

## Tasks

### Task 0 — Re-baseline on this tree before touching anything

- [ ] **Step 1:** acquire `/tmp/mtplx-gpu-exclusive.lock`; stop the user's live 9001/9002 servers
      first (`mtplx stop --port 9001`, `--port 9002`) — they hold ~115 GB each and the ladder loads
      a second copy in-process.
- [ ] **Step 2:** install this checkout into a scratch venv (`python -m pip install -e ".[dev,server]"`)
      and confirm `mtplx --version` reports 2.11.1, not the 2.11.0 the numbers above came from.
- [ ] **Step 3:** run the 8-rung probe on both packs, **repeating the 103 k rung three times**
      (cold spread is 0.43 %, but follow-up spread is 17 % and the single-shot baseline turned out
      to be the fast outlier). Store results next to the 2.11.0 baseline. A median cold `> 400 s` or
      median follow-up `> 3.3 s` at 103 k means the tree regressed, not that a fix failed.
- [ ] **Step 4:** commit nothing; append the two new artifact hashes to
      `docs/perf/receipts/qwen38-flash-next-prefill.md` and record there whether 2.11.1 moved the
      103 k numbers. If it did, every target in "Acceptance criteria" must be restated from the new
      baseline, not from the 2.11.0 figures.

### Task 1 — D6 first (it unblocks honest CI signal)

- [ ] **Step 1:** write the failing isolation test: run `test_public_cli.py` then
      `test_generation_sustained.py` in one process and assert green.
- [ ] **Step 2:** capture the red state (expected: the 5 `test_lazy_*` failures + IndexError at
      `:747`).
- [ ] **Step 3:** find the leak (suspect: process-global env or a module-level cache written by
      the CLI path and never restored). Fix with fixture-scoped restoration, not by reordering
      files.
- [ ] **Step 4:** pin `HOME`/`MTPLX_HOME` to `tmp_path` in the two `DEFAULT_HF_MODEL_ID` tests so
      they stop reading the operator's `~/.mtplx/config.toml`.
- [ ] **Step 5:** `pytest tests/test_public_cli.py tests/test_generation_sustained.py -q` green;
      commit.

### Task 2 — D1 guard: degrade instead of 500

- [ ] **Step 1:** failing test — `stable` profile + 17,000-token prompt must produce a completion,
      not `RuntimeError`; and the response/telemetry must say which lane served it.
- [ ] **Step 2:** red.
- [ ] **Step 3:** implement: at `:4957` route to the sustained prefill lane instead of raising, or
      validate `recommended_profile` against the guard at load and refuse *startup* with the same
      actionable text. Pick one; do not keep both behaviours.
- [ ] **Step 4:** add `tests/test_long_context_profile_guard.py`; also assert the pack-contract
      validator rejects `recommended_profile` values that cannot serve their own advertised
      `context_length`.
- [ ] **Step 5:** commit.

### Task 3 — D2 paged lane: make it fire, then make it provable

- [ ] **Step 1:** instrument first — emit `prefill_attention_impl` and `prefill_layout` on every
      request row, not just chunked ones (they are absent today, which made this hunt slow).
- [ ] **Step 2:** new `tests/test_qwen4_prefill_lane.py`: with a fake `qwen4_exp` layer set, assert
      the counter at `generation.py:1069` becomes non-zero past the ceiling. If no unit-level seam
      exists, that absence is itself the finding — add the seam.
- [ ] **Step 3:** on the real model, sweep `MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT` downward in
      powers of two and record where `paged_gqa_sdpa_calls` first goes non-zero, with tok/s per
      setting. Arm D already showed 32768 leaves 103 k prefill unchanged (267.003 s vs 267.920 s).
      Expected from `profiles.py:525`: repaging loses decode. If nothing ever makes the counter
      non-zero, the lane is unreachable for `qwen4_exp` and that is the answer — write it into
      `docs/perf/` rather than leaving it implied.
- [ ] **Step 4:** commit the sweep table even if the result is negative.

### Task 4 — D3 geometry constant per architecture

- [ ] **Step 1:** derive `bytes/token` from the model config (KV-bearing layers × heads × dims ×
      dtype) instead of the 65,536 MODEL DEFAULT; `descriptors.py` already knows `qwen4_exp` is
      12 KV layers of 48.
- [ ] **Step 2:** failing test asserting the announced ceiling for a `qwen4_exp`-shaped config
      matches the derived value, and that an unset override still logs the derived number.
- [ ] **Step 3:** implement; keep `MTPLX_DENSE_KV_BYTES_PER_TOKEN` as the escape hatch and log
      which of the two won.
- [ ] **Step 4:** re-run the 8-rung probe; expect **no** throughput change from this task alone
      (see D2 note) — if it does change, something else moved and must be explained.
- [ ] **Step 5:** commit.

### Task 5 — D4 benchmark: serve-path harness + the norm crash

- [ ] **Step 1:** fix the crash independent of the harness: guard the `layer.input_layernorm`
      sites in `gdn_capture.py` (`:2892`, `:2948`, `:3033`) behind the arch's real attribute
      names, with a failing test using a `qwen4_exp`-shaped fake whose layers lack
      `input_layernorm`. This is the one defect with a clean unit seam, so do it first.
- [ ] **Step 2:** add `--url`/`--port` (or `--harness serve`) to `prefill_bench.py` so the ladder
      drives a live server; the ladder's numbers must then match a probe run within noise.
- [ ] **Step 3:** delete the external `prefill-probe.py` dependency by porting its row-kind logic
      (`kind` derived from `cached_tokens`, never assumed) into the ladder's JSON.
- [ ] **Step 4:** commit.

### Task 6 — D5 honest reporting

- [ ] **Step 1:** failing test: request `q8` on a family that refuses it → `/mtplx/settings`
      reports effective `off` plus the refusal reason, not the requested value.
- [ ] **Step 2:** implement in `descriptors.py` / the settings projection; commit.

## Acceptance criteria

1. 103 k cold prefill ≤ **134 s** (baseline 267.920 s / 383 tok/s → the 2.0× line). One run is
   enough: cold-path spread measured 0.43 %.
2. 556-token follow-up at 103 k ≤ **1.40 s**, as the **median of ≥3 runs** (baseline median
   2.805 s → the 2.0× line). A single run cannot satisfy this: follow-up spread is 17.0 %, and the
   2.397 s published in the baseline table is the fast outlier of four measurements.
3. `paged_gqa_sdpa_calls` non-zero at ≥100 k **or** a written conclusion in `docs/perf/` that the
   lane is unreachable for `qwen4_exp`. No silent third option.
4. A pack whose `recommended_profile` cannot serve its own advertised `context_length` fails at
   startup, not at request 3.
5. `pytest tests/test_public_cli.py tests/test_generation_sustained.py -q` green in one process.
6. Every number claimed here reproduced on 2.11.1 by Task 0, with both JSON paths recorded in
   this document.
7. `vm.swapusage` stays `used = 0` during the run. Caveat from the field: with both Flash-Next
   ports plus an unrelated 161 GiB Metal server resident, swap went non-zero (81.56 MiB) — so this
   criterion is only meaningful under the exclusive GPU lock with one model loaded.

## Repo-workflow notes learned the hard way

- `upstream` is a **blobless partial clone** (`partialCloneFilter = blob:none`). A plain
  `git rebase main` on this box issued hundreds of one-object promisor fetches (visible with
  `GIT_TRACE=1` as `fetch upstream … --filter=blob:none --stdin`) and never finished in 15 min.
  Two fixes, both verified: hydrate the trees involved
  (`git ls-tree -r <rev> | awk '$2=="blob"{print $3}' | git cat-file --batch-check`) and use the
  explicit `git rebase --onto main <old-parent> issue-308` form, which skips the patch-id
  symmetric-difference that triggers the storm.
- Diagnose with `GIT_NO_LAZY_FETCH=1` — it turns the silent stall into an instant
  `unable to read <oid>`.
- `origin/main` does not exist locally; `main` tracks `upstream/main` (youssofal/MTPLX) while
  feature branches track `origin` (Deviad/MTPLX). `git pull --ff-only` on `main` fails by design
  because upstream rewrote history (440 local-only vs 967 upstream-only at the time of the
  rebase); `main` was fast-forwarded by `git reset --hard upstream/main` with
  `backup/pre-update-main` kept at the old tip, and `backup/pre-rebase-issue-308` at the
  pre-rebase `d0518c9`. **Nothing has been pushed.**
- Run tests with a pinned empty `HOME`/`MTPLX_HOME` or they will read the operator's live config
  (this is what produced two spurious failures during the rebase verification).
