# Dense batched-MTP cohort lane, design (2026-08-22)

**Goal:** in-process batched speculative decoding for DENSE hybrid targets
(model_type `qwen3_5`, e.g. Qwen3.8-27B), so B concurrent streams share ONE
weight read per verify cycle. Multi-process serving is bandwidth-walled at
~80-89 tok/s aggregate on M3 Ultra (each process re-streams ~14.4 GB of weights
per cycle); batching is the only lever past it. Physics at measured accept
(2.83 tok/cycle, depth 3, 24k ctx, ~540 GB/s effective): B=4 ≈ 250 tok/s,
B=8 ≈ 388 bandwidth-bound.

**The v2.8.0 release notes call this lane "on the roadmap"; this is that lane,
scoped like the A3B one was: driver + CPU contract tests + bench first, serving
wiring after.**

## What already exists (all shipped for the A3B MoE lane, all model-agnostic)

- `RaggedBatchKVCache` (`mtplx/ragged_kv_cache.py`): per-row `int32[B]` offsets,
  scatter append, per-row causal mask via `create_attention_mask ->
  cache.make_mask`, per-row commit via `new_offsets` / offset rewrite.
- `OwnedRecurrentStateCache` (`mtplx/cache_state.py`): batch-major GDN
  conv+matrix state with per-row masked restore.
- `commit_captured_rows` (`mtplx/gdn_capture.py:3146`): per-row ragged commit ,
  per-row KV offset rewrite + per-row `take_along_axis` selection of captured
  per-step GDN states. Requires a capture with per-step `"states"`.
- `gdn_forward_with_capture` backends `stock` / `linear_gdn*` /
  `linear_gdn_from_conv_stream`: batch-shaped dense capture that MATERIALIZES
  per-step states (the `tape` backend does not, it is the B=1 fast path and is
  NOT used here; no new Metal kernel is needed).
- `forward_with_gdn_capture` (`gdn_capture.py:2959`): the dense hybrid verify
  forward, batch-shaped, capture-carrying.
- The split-attention hook (`attention_split.py`): array cache offsets fail
  every custom fast path closed -> stock SDPA + per-row rope offsets, unchanged.
- `to_foldin_cache` (`mtplx/batched_decode.py:851`): scalar->ragged cache
  conversion after prefill.
- CPU fake-runtime test pattern (`tests/test_batched_decode.py`): per-stream
  sha parity, batched vs alone, on tiny deterministic fakes.

## What this build adds

1. **`mtplx/dense_mtp_batch.py`, `generate_dense_mtp_batch`**: a greedy,
   fixed-cohort, depth-K (default 3) dense driver. Per cycle:
   - `x0 = argmax(logits_last)` per row (device).
   - K chained MTP drafts: `d_{j+1}` from `d_j`'s head hidden, fresh head cache
     per cycle (`mtp_position_mode: local`), each a `[B,1]` head call.
   - ONE `[B, K+1]` `forward_ar_capture` (states-materializing backend), the
     amortized weight read.
   - Device decision: `k[b] =` length of the matched draft prefix
     (`cumprod` over `d_j == argmax(v_logits[:, j-1])`), keeps `= 1 + k`.
   - ONE host sync: the `[K+2, B]` bundle (input tokens + k), commit
     bookkeeping, stop detection.
   - `commit_captured_rows(cache, captures, keeps, verified=K+1)`: per-row KV
     offset rewrite + per-row GDN state selection. Fail-loud if it returns
     False (no silent fallback).
   - Next `logits_last` / `hidden_last`: per-row `take_along_axis` at `k`.
   Committed tokens per row per cycle = `x0` + the k accepted drafts ,
   identical to the solo MTP semantics, so a greedy batched stream must be
   byte-identical to the same stream decoded alone (the correctness contract,
   same as Phase-1 A3B).
2. **`tests/test_dense_mtp_batch.py`**: CPU fake that routes through the REAL
   `commit_captured_rows` + `RaggedBatchKVCache`, per-row histories are
   encoded in the fake's capture arrays so per-row selection is exercised
   end-to-end. Sha parity across accept patterns (rows broken at different
   depths), offset audits, stop handling.
3. **`mtplx/benchmarks/runners/dense_batch_bench.py`**: GPU bench, loads a
   real runtime, decodes B in {1,2,4,8} at depth {2,3} on the 24k agentic
   prompt, reports aggregate decode tok/s + acceptance-by-depth + per-stream
   sha parity vs the same driver at B=1.

## Deliberate v1 scope cuts (mirror the A3B lane's Phase-1 discipline)

- Greedy only (`temperature <= 0`); p/q sampled accept is a follow-up.
- Equal-length (or left-padded) prompts; continuous admission reuses the
  refill machinery later.
- Fresh MTP-head cache per cycle (no committed-history head cache). If GPU
  acceptance measurably drops vs solo (~0.68 at 24k), a ragged head-history
  cache is the follow-up, the head is one layer, so the cost is small.
- No scheduler/serving wiring yet: the driver is the `hooks.decode_step` body a
  dense MTP_BATCH service will call; the scheduler gate
  (`batching/scheduler.py:292`) is untouched by v1.
- Per-cycle transient capture memory: `(K+1) x 151 MB x B` of per-step GDN
  states (~2.4 GB at B=4, K=3), accepted for v1; the stream backend keeps it
  device-side and it dies each cycle.

## Known risks (each with a measurement gate before hardening)

- Acceptance without head history (above).
- `stock` capture backend throughput at S=4 (python-level stepping); the
  `linear_gdn_from_conv_stream` kernel backend is the fallback lever.
- Compute crossover at B≈4 (verify matmuls at 4B positions): measured by the
  bench ladder, decides whether B=4 or B=8 is the operating point.
- Array-offset rope on dense `qwen3_5` full-attention layers: validated for
  A3B (same rope family) but must be parity-gated on dense (the B=1-equivalent
  cohort sha test).
