# Semantic-anchor real-model evidence — September 7, 2026

**The six-pair measurement campaign is complete; the acceptance gate fails.** Three fresh OFF/ON pairs were measured for each workload. No additional cached tokens were recovered in any of the 24 warm-turn comparisons. Output-parity failures occurred in pairs **long 1, long 2, long 3**. These results do not support promoting this feature on the tested model and workloads.

The initial report at evidence commit `de31b270dff1f586c215dd104b2f11f878c2b1cb` contained one isolated pair per workload. This report incorporates all required repetitions and preserves those original results. An additional long ON arm rejected by the host isolation guard remains under `excluded/`; it is not counted as one of the six pairs.

## Pinned configuration and isolation

- Measured feature source: `dfe28bf94fb62d35ba305d3b764c044c880f0905`, including upstream correctness baseline `21be78b3f51820eecef020e5e4855c0715eaf9a5`. All arms use that same unchanged runtime and replay harness.
- Model: `Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed`, immutable revision `766cec2bb474544381139cc036ae1c2267759b33`. Every installed model/tokenizer file was checked against the original measured SHA-256 inventory again before the resumed campaign.
- MLX 0.32.2, mlx-lm 0.31.3, Transformers 5.8.0 and tokenizers 0.22.2; versions were rechecked before the resumed run. Apple M5 Max, 128 GiB, macOS 27.0 build 26A5425a.
- Turbo profile, target-only AR, MTP sidecar not loaded, 131,072-token context, greedy sampling, request thinking disabled, native tokenizer template, preserved prior reasoning, no agent rewrites, one-token streaming, no stats footer.
- Every arm used a new server, initially empty SessionBank and fresh private SSD directory. Bank cap 32 GiB, 24 entries, three per session; SSD cap 20 GiB; postcommit wait timeout 30 seconds. These are controlled settings, not automatic defaults.
- Identical 32-token initial warmup and blocking extended warmup completed before replay. N-gram prewarming was disabled. AC power and Apple's default fan policy were used throughout every included arm.
- The first two pairs passed before/after host process checks. For the remaining four pairs, the user authorized coordination with the other MLX task, which explicitly stopped and held its GPU/model jobs and controllers. A separate CPU validation task using approximately one core was known and was not interrupted. There is no continuous system-wide GPU trace or claim of an otherwise idle computer.

The only controlled feature difference is `MTPLX_SEMANTIC_ANCHORS=0` versus `1`. Original launch records, effective health settings and per-request flag telemetry accompany the operator-attested manifests. `PYTHONPATH` explicitly selected the pinned checkout for the CLI's safe-path server child. The summary checks environment manifests, frozen input hashes, prompt counts, flags, usage and host records across all repetitions.

## Frozen real-model workloads

Both conversations contain public synthetic reference records, actual model-generated `lookup_record` calls, deterministic tool replies, a plain append and repeated tool use. Complete requests were captured once and frozen. Newly generated OFF/ON answers were never appended to later replay requests.

Actual replay prompt counts are **4,097–4,281** for short and **126,982–127,166** for long. Capture requests counted one token fewer; both replay arms have identical canonical request hashes and measured prompt counts. The capture/replay difference was not separately diagnosed, and capture timings are excluded.

Arm order was short ON/OFF, OFF/ON, ON/OFF and long OFF/ON, ON/OFF, OFF/ON. Cold initial turns are uncached in all twelve included arms. The excluded earlier long ON attempt was restarted from scratch for the valid second pair.

## Correctness and cache recovery

Each output signature covers generated content/reasoning/tool function names and arguments, completion-token count and finish reason; random tool-call IDs are excluded. The exact within-arm repetition results are:

- Short OFF: unique output signatures per turn [1, 1, 1, 1, 1] across three repetitions.
- Short ON: unique output signatures per turn [1, 1, 1, 1, 1] across three repetitions.
- Long OFF: unique output signatures per turn [1, 1, 1, 1, 1] across three repetitions.
- Long ON: unique output signatures per turn [1, 1, 1, 1, 1] across three repetitions.

All three long plain-append comparisons return `Marker: amber-17` with OFF versus `amber-17` with ON: seven completion tokens versus five, with normal finishes. The other four long turns and all five short turns match in every pair.

A value of one means that turn produced the same output signature in every repetition of that arm. The failed pair list above compares OFF with ON; no normalization or weakened parity check was used. The per-turn receipts retain full hashes and counts, and generation logs retain the public fixture's text previews. Cause is not established by these output observations alone.

The full per-turn cache counts are in `per-turn.csv`. Existing RAM clone and block-boundary restoration must not be attributed to this feature. The initial long pair already recovered 127,010 tokens on the first two warm turns and 126,976 on the last two, identically with OFF and ON. The complete summary records every paired cached-token delta, including zero gains.

ON telemetry admits the initial user boundary (4,090 short / 126,975 long), while later candidates fail strict `not_exact_prefix` checks. The extra long boundary is one token before the 126,976-token block boundary used later. Source inspection shows that mandatory edges can split prefill spans. A resulting numerical effect is an investigation lead, not a diagnosed cause of the output mismatch. Strict prefix validation remains intact.

## Timing observations

Seconds to the first generated delta. Positive deltas mean ON was slower. Each median uses three repetitions; the paired delta is calculated within each pair before taking its median. Ranges show all three paired deltas. Failed-parity timings are diagnostic and are not eligible speed claims. Three repetitions and the recorded host limits do not establish broad statistical or model-quality guarantees.

| Workload / turn | Parity pairs | OFF median | ON median | Median paired delta | Paired delta range |
|---|---:|---:|---:|---:|---:|
| Short / tool result | 3/3 | 0.578 | 0.742 | +0.147 | +0.147 to +0.166 |
| Short / plain append | 3/3 | 0.613 | 0.780 | +0.155 | +0.147 to +0.170 |
| Short / repeated tool call | 3/3 | 1.712 | 1.867 | +0.158 | +0.154 to +0.164 |
| Short / repeated tool result | 3/3 | 0.708 | 0.887 | +0.174 | +0.168 to +0.192 |
| Long / tool result | 3/3 | 11.036 | 10.409 | -0.627 | -0.674 to -0.352 |
| Long / plain append | 0/3 | 1.309 | 1.683 | +0.373 | +0.372 to +0.414 |
| Long / repeated tool call | 3/3 | 2.974 | 3.377 | +0.403 | +0.390 to +0.430 |
| Long / repeated tool result | 3/3 | 1.653 | 2.265 | +0.595 | +0.590 to +0.634 |

Cold generated-TTFT medians, reported separately: short: OFF 6.123 s / ON 6.211 s; long: OFF 260.223 s / ON 249.267 s. For these text-only answers, visible-content TTFT equals generated TTFT; tool-only answers have null content TTFT. Both metrics and every unaggregated observation are retained.

Postcommit waiting is part of end-to-end client TTFT. In the initial long tool-result pair it accounted for 10.507 seconds OFF / 9.536 seconds ON while retokenized-history jobs completed. The records expose that waiting separately so variation in this existing cache work is not presented as semantic-anchor benefit.

## Reproduction and evidence

Use the [protocol](../semantic-anchor-ab.md) and `scripts/bench_semantic_anchor_replay.py` at the measured source commit. The unchanged harness's 15 contract tests passed before measurement; those tests validate measurement behavior rather than model exactness.

1. Decompress `short-transcript.json.gz` and `long-transcript.json.gz`. The decompressed bytes exactly match the frozen inputs; original and export hashes are in `inventory.json`.
2. Substitute local paths for `<SOURCE_DIR>`, `<MODEL_DIR>`, `<PYTHON>` and `<CAMPAIGN_DIR>` in launch records. Use the pinned source and weights, a new SSD directory for each arm and the same settings. Preserve the explicit `PYTHONPATH` for the safe-path child.
3. Finish the same warmup, inspect health, run the frozen sequence, stop that arm and start the next fresh process. Compare with the unchanged harness. Keep failed comparisons and excluded attempts visible.

The bundle includes all twelve included arm receipts, six comparisons, frozen transcripts, actual capture responses, manifests, effective health records, selected restore/commit telemetry, public generation events, `per-turn.csv`, `summary.json`, the original excluded attempt and a hash inventory. Original local files are retained. Export changes are documented path substitutions, CSV line-ending normalization, selected log fields and lossless transcript compression.

The remaining engineering gate is to diagnose the exact-output difference and demonstrate useful incremental restoration with repeatable latency evidence. This campaign does not validate other models, MTP mode, other context sizes, model quality, package signing or release readiness. The requested repetitions are finished; their outcome does not clear the feature's acceptance gate.
