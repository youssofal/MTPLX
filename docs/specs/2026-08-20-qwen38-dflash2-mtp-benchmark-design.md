# Qwen3.8 27B DFlash2 against Optimized Speed MTP

Status: approved benchmark contract; implementation and measurements pending

## Goal

Add the `z-lab/Qwen3.8-27B-DFlash2` drafter to MTPLX's DFlash competitor
channel and determine whether it can beat the unchanged Qwen3.8 27B
Optimized Speed MTP runtime on this machine.

The benchmark is greedy, uses one tokenizer-normalized 1,024-token Python
coding prompt, and forces 1,024 generated tokens. The control is the complete
MTPLX Optimized Speed `turbo` profile at its artifact-recommended MTP depth 3.
The DFlash2 arm reuses the same loaded target object, weights, tokenizer, and
resolved profile, but runs them through the DFlash2 speculative engine using
the checkpoint's native eight-token block. It does not pretend that DFlash2 is
merely a drop-in MTP draft head:
the two engines retain their own cache, proposal, and verification mechanics.

The work has two sequential phases:

1. **Phase A — depth benchmark:** run Qwen3.8 DFlash2 at its checkpoint-valid
   widths 1-8 and establish the fastest correct width.
2. **Phase B — optimization:** profile only the Phase A winner, then design and
   measure a fit-for-purpose optimization against the unchanged winner and the
   MTP reference.

MTP is the final reference to beat. Phase A is complete when the optimal stock
DFlash2 width is established honestly, even if it is slower than MTP. No custom
kernel work begins before that result exists.

## Pinned inputs

- MTPLX base: upstream `main`, initially checked out at
  `2b0360ca1af5c383a797a9d96999540f3197f182` on 2026-08-20. The final receipt
  records the actual tested commit.
- Target artifact:
  `/Users/davidtai/.mtplx/models/Youssofal--Qwen3.8-27B-MTPLX-Optimized-Speed`.
  Its files, quantization layout, runtime metadata, and tokenizer are used
  unchanged.
- Target profile: MTPLX `turbo`, including model-local runtime overrides. Both
  arms record the resolved environment and installed target routes.
- MTP control depth: 3, matching the artifact's current default and maximum.
- DFlash2 checkpoint: `z-lab/Qwen3.8-27B-DFlash2`, pinned in the receipt to Hub
  revision `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`.
- DFlash2 width authority: that checkpoint's `dflash_config.block_size=8` and
  published seven-draft-token configuration. Unrelated generic DFlash runtime
  defaults or synthetic capability tests must not clamp this model to five.
- DFlash2 reference implementation: `bstnxbt/dflash-mlx`, pinned in the
  receipt to commit `60803233af4589e18588b9bacbb03880801c828a` unless a newer
  head is deliberately selected and recorded before implementation begins.

Downloads and dependency setup happen before the GPU lock is acquired. Model
execution, warmup, profiling, and timed rows happen only while holding
`/tmp/mtplx-gpu-exclusive.lock`.

## Non-goals

- Do not run or report DFlash2 depths 9-16. The released checkpoint has a
  trained block size of 8; silently clamping larger requests to 8 would create
  duplicate, mislabeled rows.
- Do not replace the Optimized Speed target with a stock mlx-lm or community
  Qwen quantization to make DFlash2 load.
- Do not port the older MTPLX DFlash backend by name. It implements the earlier
  DFlash topology and does not preserve DFlash2's selector, dynamic
  convolution, layer layout, or checkpoint contract.
- Do not modify target weights, quantization, cache layout, target arithmetic,
  or the promoted Optimized Speed profile.
- Do not add, port, tune, or benchmark a custom DFlash2 kernel during Phase A.
  That phase changes only compatibility and benchmark plumbing required to run
  the checkpoint's native width 1-8 contract fairly.
- Do not switch the persistent Qwen service to DFlash2 as part of this work.
- Do not publish a speed claim from a microbenchmark, profiler trace, or
  unmatched runtime comparison.

## Execution architecture

### One target runtime

Both arms use the exact local Optimized Speed artifact loaded once through
MTPLX after applying the same `turbo` profile and model-local runtime
overrides. The target model object, weights, tokenizer, token embedding, and
lm-head are shared.

Each speculative engine creates its own fresh cache for each arm. The MTP
control uses MTPLX's promoted cache and verification path. The `dflash-mlx`
candidate uses its existing Qwen GDN cache, rollback, and
target-prefix verification machinery against the already-loaded MTPLX target
object. This unavoidable engine-level difference is recorded in the receipt;
Phase A does not rewrite it in an attempt to isolate only proposal cost.

The MTP arm uses the existing native MTP head and `generate_mtpk` at depth 3.

The DFlash2 arm uses the existing `dflash-mlx` engine, `QwenGdnTargetOps`, and
native `DFlash2DraftModel`; this project does not create a second
`dflash2-mlx` package, duplicate the DFlash2 implementation, or add a custom
target adapter during Phase A. Construction gives that engine the already
loaded MTPLX target model and tokenizer instead of a target model reference.

The `dflash-mlx` engine loads and validates only the released DFlash2 draft
checkpoint, captures the required target residual streams at layers 5, 19, 33,
47, and 61 through its existing Qwen target operations, maintains DFlash2's
context cache, emits one greedy candidate block, and drives target-prefix
verification. It must not load a second stock target model or replace the
already-loaded MTPLX target object.

Residual-stream capture is necessary DFlash2 proposal input and is charged to
the candidate's end-to-end time. Capture must not change target arithmetic.
The installer proves that the captures have the expected shape, dtype, layer
ownership, and position mapping for the real target before generation begins.

### DFlash2 depth semantics

The sweep parameter is DFlash2 block width, including the already-known first
target token. Width `N` therefore contains `N - 1` speculative DFlash2 tokens
and one target verification call over at most `N` positions. Width 1 is the
no-draft boundary and is retained as a diagnostic, not as a speculative win.

The requested and effective widths must be identical and recorded for every
row. The accepted sweep is exactly 1, 2, 3, 4, 5, 6, 7, and 8.

### Construction boundary

Before any measured generation, installation validates once:

- exact target artifact identity and tokenizer vocabulary;
- target architecture, 64-layer hybrid GDN/full-attention topology, hidden
  width, target tap indices, dtype, and per-layer quantization layout;
- DFlash2 architecture, block size 8, five sliding-attention layers, two-tap
  dynamic convolution, group size 16, selector rank 256, selector top-k 16,
  mask token, vocabulary, and target layer IDs;
- installed DFlash2 runtime capabilities expose the checkpoint's exact
  default/maximum block width 8, never a generic five-token clamp;
- target embedding and lm-head ownership shared with the DFlash2 proposer;
- unchanged DFlash cache rollback/commit behavior at every verify width 1-8;
- the installed MTPLX `turbo` route and its resolved model runtime overrides;
- a small exact greedy parity self-check against the unchanged target stream.

The constructor returns an immutable bundle containing the MTPLX target,
existing Qwen target operations, DFlash2 model/backend, tokenizer, and runtime
context. An invariant failure aborts before warmup. The enabled DFlash2 hot
path has no `eligible-or-stock`, exception fallback, per-cycle metadata
validation, environment reads, or proof counters.

## Greedy correctness contract

The only sampler is greedy argmax: temperature 0 with no stochastic top-p or
top-k behavior. Both arms use identical encoded prompt IDs and force the same
1,024-token generation budget without early EOS termination.

The authoritative token oracle is greedy target-only generation through the
same MTPLX target runtime and profile. Before performance is considered:

1. The MTP depth-3 output token IDs must exactly match the oracle.
2. Every DFlash2 width 1-8 must exactly match the same 1,024 oracle token IDs.
3. Prompt IDs, generated IDs, text hashes, finish reason, and token counts are
   written to the receipt.

Any token mismatch, shortened run, hidden fallback, or requested/effective
width mismatch rejects that arm.

## Benchmark workload

The repository gains one deterministic Python coding prompt fixture long
enough to exceed 1,024 encoded tokens. The runner applies the target
tokenizer's production chat template once and takes the first exactly 1,024
token IDs as the benchmark prompt. It asserts that length before model-load
timing begins and records the source prompt, exact encoded IDs and their hash,
tokenizer identity, chat-template hash, and reasoning/thinking setting. Both
arms receive that same immutable token-ID tuple directly.

Each measured row generates exactly 1,024 tokens. Warmup uses the same arm and
width but is excluded from timing. The benchmark order uses matched brackets:

1. Optimized Speed MTP depth-3 control (`C0`).
2. One DFlash2 width (`B`).
3. Optimized Speed MTP depth-3 control (`C1`).

Run at least three complete brackets per width. If a single bracket exceeds two
minutes, two are acceptable only when the two control values differ by at most
5 percent. Width order is rotated between repetitions to avoid assigning
thermal drift to depth.

The primary metric is decode tokens per second over the same post-prefill
boundary. Receipts also contain end-to-end wall time, prefill time and rate,
time to first token, generated tokens, target verify calls, drafted and
accepted tokens by position, mean accepted speculative tail, peak memory,
requested/effective width, profile identity, and thermal/quiet-machine state.

## Depth-selection and final promotion gates

The MTP reference is the mean of the two controls bracketing each candidate,
not an older published number. Observed bracket drift is the absolute
difference between `C0` and `C1`.

A Phase A DFlash2 width is ranked by its decode-throughput ratio to the mean of
its bracketing MTP controls. The optimal width is the highest correct median
ratio across accepted brackets. If the leading widths differ by no more than
the full observed control drift, run direct alternating tie-break brackets; if
they remain indistinguishable, report the tie band instead of inventing a
single winner.

Only after the winner or tie band is frozen may Phase B profile and optimize
it. A final DFlash2 configuration beats MTP only if all correctness gates pass
and its decode throughput exceeds the bracketed MTP mean by more than the full
observed control drift. A practical win must repeat in the same direction
across the accepted brackets. Peak memory and end-to-end latency are reported
even when decode throughput wins.

If no width clears this gate, the honest result is that Optimized Speed MTP
remains the winner. The runner and receipts may still land, but DFlash2 is not
promoted as a speed improvement.

## Phase B profiling and optimization loop

Phase B is blocked until Phase A has committed immutable stock-DFlash2 receipts
and selected the fastest correct width or a measured tie band. Then profile
only that selected configuration outside the scored timing window and identify
the top three wall-time components using real 1,024/1,024 shapes.

The custom optimization is specified in a separate follow-on implementation
plan written from that profiler evidence. The Phase A plan must not guess its
kernel, geometry, or target operation.

Optimization proceeds one hypothesis at a time:

1. preserve DFlash2 arithmetic, selector behavior, cache ownership, tensor
   layout, dtype, and target tap mapping;
2. add a failing parity or construction test before production code;
3. measure the isolated candidate against unchanged DFlash2 and the bracketed
   MTP reference;
4. retain only token-exact improvements that exceed observed drift;
5. revert neutral or losing candidates immediately.

Likely investigation areas are determined by the profile, not assumed in
advance. Candidate examples include DFlash2 block construction, grouped
dynamic convolution, context-cache append, selector synchronization,
target tap materialization, and verify-width routing. Improvements to
generic DFlash2 arithmetic belong upstream in `dflash-mlx`; MTPLX keeps only
the construction bundle and benchmark integration. No optimization is accepted
solely because a similarly named kernel won on the older DFlash model.

Stop when a repeatable DFlash2 configuration clears the MTP gate or when the
profiled candidates fail the first matched A/B gates. The final ledger records
each attempted change, commit, exact command, correctness result, performance
result, and retain/reject decision.

## Guarded machine operation

The guarded runner:

- resolves and acquires `/tmp/mtplx-gpu-exclusive.lock` without stealing it;
- refuses to measure while unrelated CPU/GPU-heavy processes violate the quiet
  gate;
- records the exact live `com.tea.qwen` service state and command;
- stops only that service while holding the lock;
- restores that exact service in a `finally` path after success, failure, or
  interruption;
- proves the restored model, profile, MTP depth, reasoning default, health, and
  DeepSeek-disabled state before releasing the lock.

Downloads, source checkout, test discovery, and CPU-only fixture generation do
not require the GPU lock.

## Deliverables

### Phase A

- Updated MTPLX DFlash channel using the current `dflash_mlx` package/API and
  the Qwen3.8 DFlash2 checkpoint.
- Construction-time bundle connecting the existing `dflash-mlx` engine to the
  already-loaded MTPLX target, with no second target load, custom kernel, or
  measured-path fallback.
- Exact greedy parity tests for widths 1-8.
- Deterministic 1,024-token Python prompt fixture and 1,024-token forced-output
  guarded runner.
- Bracketed MTP depth-3 reference and unchanged DFlash2 width 1-8 benchmark
  receipts.
- A frozen optimal stock DFlash2 width or an explicitly measured tie band.

### Phase B

- Profiler evidence for the Phase A winner at the real 1,024/1,024 workload.
- A separate evidence-derived custom-optimization plan.
- An optimization ledger including rejected candidates and matched A/B rows.
- A final conclusion stating whether optimized DFlash2 actually beats the
  Optimized Speed MTP control.

## Failure handling

- Optimized Speed cannot supply DFlash2's required taps without changing target
  arithmetic: stop and report the incompatibility; do not substitute a target.
- DFlash2 checkpoint or package contract differs from the pinned geometry:
  fail installation before generation.
- Width is clamped or rewritten: reject the row.
- Greedy tokens differ: reject the candidate before timing it further.
- MTP control drifts excessively: discard the bracket and restore a quiet,
  thermally stable machine before retrying.
- Live Qwen restoration fails: keep the GPU lock held while attempting bounded
  recovery, then report the exact service state rather than claiming success.
- DFlash2 loses: preserve the evidence and leave the production MTP launcher
  unchanged.

## Adversarial review

1. **Critical: DFlash2 silently runs through a separately loaded stock target.**
   The measured `dflash-mlx` engine receives the target model from an already
   constructed `MTPLXRuntime`; the runner never gives it a second target model
   reference. Receipts assert target object, target weights, profile, and the
   distinct cache/engine identities for both arms.
2. **Critical: capturing target taps disables the promoted target path.**
   Construction records the installed route and a matched target-forward probe.
   Any unavoidable route change is reported as candidate overhead and cannot be
   described as the unchanged target path.
3. **Critical: widths above the trained limit appear as successful duplicates.**
   The CLI accepts only 1-8 and asserts requested width equals effective width.
4. **Critical: speculative output differs while throughput looks good.**
   All 1,024 generated token IDs must equal the greedy target oracle before a
   row enters performance aggregation.
5. **High: DFlash2 wins only because the MTP control uses weaker settings.**
   Both arms share target bytes, tokenizer, prompt IDs, output budget, profile,
   runtime overrides, timing boundary, and machine window. The control is
   bracketed around every candidate.
6. **High: instrumentation creates the apparent bottleneck.**
   Detailed profiling is outside scored runs. Timed runs retain only existing
   cycle summaries and post-run aggregation.
7. **High: benchmark work leaves the production server down or altered.**
   The guard captures and restores the exact service configuration and verifies
   it before lock release.
