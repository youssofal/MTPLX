# Exact semantic anchors: real-model A/B gate

This protocol addresses PR #350's incremental-benefit question. Planner timing is not evidence of a cache-hit or TTFT improvement. No real-model A/B receipt is claimed by adding the replay harness.

## Hold the build and workload constant

Use the same feature-capable commit for both arms, with only `MTPLX_SEMANTIC_ANCHORS=0` versus `1` changing. First bring the feature onto the current correctness baseline, including the turn-boundary behavior from #446. Comparing against stock 2.10.0 would confound that fix with semantic anchors.

Record the exact commit, model weight/quantization revision, tokenizer/template revision, MLX version, hardware, context window, MTP/AR mode and depth, memory plan, SessionBank limits, SSD policy, postcommit policy, streaming interval, model/table prewarming, and thermal/fan policy. Do not compare different binaries or run two large-model servers concurrently on one machine.

The key workloads are a plain append and a tool-result turn over the same long base prefix, including the reported roughly 128K case. Include shorter multi-turn sequences and repeated tool calls. Report actual `prompt_tokens`, not a character-count estimate. Avoid private repositories, credentials, or customer transcripts in a public fixture.

## Freeze complete requests once

Capture the complete request bodies from one controlled real-model baseline conversation, including actual assistant tool calls and tool results. Use deterministic sampling and freeze all subsequent turn bodies. The replay must not append newly generated OFF or ON answers to the next request; that would silently benchmark different transcripts.

The input is a JSON object with a stable `session_id` and a `requests` array of complete OpenAI chat request bodies. Each body names the served `model`, has `temperature: 0`, `n: 1` (or omitted), bounded `max_tokens`, and the full `messages`, tools, and settings for that turn. The harness adds streaming and final usage identically to both arms. It sends the same session header within a sequence and does not execute returned tools.

A schema-only smoke could use:

```json
{
  "session_id": "semantic-anchor-sequence-1",
  "requests": [
    {"model": "served-model-id", "temperature": 0, "max_tokens": 32,
     "messages": [{"role": "user", "content": "Reply OK."}]},
    {"model": "served-model-id", "temperature": 0, "max_tokens": 32,
     "messages": [{"role": "user", "content": "Reply OK."},
                  {"role": "assistant", "content": "OK"},
                  {"role": "user", "content": "Reply OK again."}]}
  ]
}
```

This hand-authored example is not the real-model tool-turn benchmark and must not be published as the requested performance receipt.

## Isolate each arm

Use a fresh server process and a separate newly created SSD session-cache directory for every arm and repetition. Preserve identical in-memory bank limits. Do not clear a user's existing cache or merely change the HTTP session ID while leaving another arm's bank alive. Allow identical configured model warmup and table pre-read to finish before replay. Disable the visible stats footer. Preserve server debug logs for restore, commit, token-boundary, and postcommit analysis.

Set `MTPLX_SEMANTIC_ANCHORS=0` for OFF and `1` for ON. Use `--ssd-session-cache-dir` for each new private directory and `--no-stats-footer` to keep timing prose out of output hashes. Stop the previous process before launching the next arm.

Each arm needs an operator-supplied manifest:

```json
{
  "server_commit": "FULL_FEATURE_COMMIT_SHA",
  "model": "served-model-id",
  "model_revision": "EXACT_WEIGHTS_AND_QUANTIZATION_REVISION",
  "tokenizer_revision": "EXACT_TOKENIZER_AND_TEMPLATE_REVISION",
  "mlx_version": "EXACT_INSTALLED_VERSION",
  "hardware": "M4 Max, 128 GB, exact macOS build",
  "server_settings": {
    "context_window": 131072,
    "generation_mode": "COPY_THE_ACTUAL_MODE",
    "other_settings": "RECORD_THE_COMPLETE_CONTROLLED_LAUNCH_CONFIGURATION"
  },
  "anchors_enabled": false
}
```

Create the ON manifest from OFF by changing only `anchors_enabled`. These fields are attestations, not measurements: the script cannot verify the running process's environment flag through `/health`. Check the launch and server logs. Do not include API keys in a manifest.

```bash
# With the isolated OFF server ready:
python scripts/bench_semantic_anchor_replay.py run \
  --transcript frozen-transcript.json --manifest off-manifest.json \
  --base-url http://127.0.0.1:8000 --output off-1.json

# Stop it and start the isolated ON server with otherwise identical settings:
python scripts/bench_semantic_anchor_replay.py run \
  --transcript frozen-transcript.json --manifest on-manifest.json \
  --base-url http://127.0.0.1:8000 --output on-1.json

python scripts/bench_semantic_anchor_replay.py compare \
  --off off-1.json --on on-1.json --output pair-1.json
```

For authenticated loopback serving, pass `--api-key-env MTPLX_API_KEY`. The secret is read from the environment and is not written into receipts. The default socket timeout is 1,800 seconds, allowing for the reported long re-prefill. The script performs no cache deletion or process management.

## Interpret the receipt

The harness requires an uncached first turn, explicit final cached-token usage, a successful stream terminator, and generated output. It measures client time to the first reasoning/content/tool-function delta, not a role-only event or response headers. Visible-content TTFT is separate and may be null for a tool-only answer. Missing usage, truncated streams, error finishes, and contaminated first turns fail rather than becoming zero-valued observations.

Receipts contain request/output hashes, token counts, finish reasons, and timings, not raw messages, tool arguments, or generated text. Tool-call IDs are excluded from output hashes; names and arguments remain covered. Receipts are published atomically and existing files are never overwritten.

Comparison rejects differences in the transcript, request hashes, prompt-token counts, or declared environment beyond the anchor flag. Output, finish-reason, or completion-token divergence produces a failed parity result. This does not independently establish model quality or prove the attested server configuration. A `length` finish remains visible and must not be represented as a completed agent task.

Run multiple fresh pairs with balanced OFF/ON and ON/OFF order, at least three pairs per workload. Retain every pair, including zero gains and regressions. Report each warm turn's cached tokens and generated/content TTFT, then summarize paired deltas across repetitions. Keep cold startup separate. Attach the frozen public transcript, manifests, receipts, and sanitized restore/commit logs for maintainer reproduction.

The merge gate is incremental useful restoration beyond the current anchor set, without output/correctness regressions and with repeatable latency evidence. If existing turn/block anchors already recover the prefix, report that result rather than attributing their benefit to this feature.

## No-model validation

```bash
python -m unittest discover -s tests -p test_semantic_anchor_replay.py -v
```

The 15 tests cover stream parsing, first-token attribution, missing/invalid usage, tool-only output, hash stability, comparison drift/parity, atomic non-overwriting receipts, and a loopback fake-server request replay. They establish the measurement contract only. The real 27B/128K run remains an explicit outstanding gate until hardware receipts are attached.
