# System Paged Cache Implementation Plan

> Execute serially.  The generic ownership layer, the standard paged adapter,
> DeepSeek hybrid specs, and DFlash2 chunk lifetime share one evolving storage
> contract.  Stop for the first real correctness or memory failure; do not add
> hypothetical features or tests.

**Goal:** Replace append/concatenate ownership on the exact Mia long-context
route with a reusable fixed-capacity paged cache system, then finish the exact
model benchmarks and PR.

**Design:** `docs/specs/2026-08-21-system-paged-cache-design.md`

## Task 1: Reusable fixed-capacity page ownership

**Files:**

- Create `mtplx/paged_cache.py`.
- Modify `mtplx/cache_state.py`.
- Modify only directly relevant cases in `tests/test_cache_state.py`.

Implement immutable cache specs/plans, a fixed physical pool, request leases,
block tables, slot mappings, paged views, bounded trim, and state handoff.  Parse
configuration and validate geometry when the plan is installed.  The fixed
route must have no geometric growth, concatenation, hot environment reads, or
silent fallback.

Gate: make the existing fixed paged-cache construction/update/trim tests pass
through the shared owner.  Do not add concurrency, eviction, or prefix-sharing
tests.

Commit: `feat: add reusable fixed paged cache ownership`

## Task 2: Move the existing system paged cache onto the owner

**Files:**

- Modify `mtplx/cache_state.py`.
- Modify the runtime construction boundary that currently installs paged cache
  settings.
- Modify existing paged-cache tests only where the delegated owner changes the
  required contract.

Make `VllmMetalPagedKVCache` delegate storage, capacity, slot writes, trim, and
state transfer to the shared layer while preserving its attention interface.
Bind attention and quantization routes once at installation.  Keep the old
non-paged owners explicit; do not fall back to them from an enabled paged lane.

Gate: run the existing cache-state and GraphBank paged subsets that exercise
construction, writes, compiled offsets, trim, and state transfer.

Commit: `refactor: share paged cache ownership system-wide`

## Task 3: Install DeepSeek V4 hybrid page specs

**Files:**

- Modify `mtplx/deepseek_v4_nvfp4_kv.py`.
- Modify `mtplx/models/deepseek_v4.py`.
- Modify `mtplx/models/deepseek_v4_dspark.py`.
- Modify existing DeepSeek NVFP4, target, and DSpark checks.

Keep the proven `stock432` codec and arithmetic.  Replace growing record arrays
with paged views sized from the installed context capacity and real logical-to-
stored ratios.  Page the growing ratio-4/ratio-128 target compressed lanes and
ratio-4 indexer lane; keep target and draft windows bounded.  Preserve current
compressor frontier arithmetic and rollback semantics.

Gate: the current exact record/arithmetic/trim/DSpark checks pass, and a direct
16K cache exercise no longer replaces growing arrays.

Commit: `feat: page DeepSeek V4 native NVFP4 caches`

## Task 4: Stream DFlash2 context into draft pages

**Repositories:** MTPLX and the already-pinned DFlash2 dependency.

Add a construction-selected streaming context-consumer interface.  On the
DeepSeek adapter, project only the scheduled target chunk, write the three
draft-layer context K/V records into their persistent slots, and release chunk
features.  Do not allocate `TargetFeatureStore[prompt_length]` for this route.
Keep the existing retained-feature route for adapters that require it.

Gate: the existing DeepSeek DFlash2 adapter checks pass and the encountered
full-prompt feature-store allocation is absent.  Run one exact-model chunked
prefill/epoch before proceeding.

Commit DFlash2 first, pin that commit in MTPLX, then commit MTPLX as
`feat: stream DeepSeek draft context into paged cache`.

## Task 5: Close the full Mia/SparkInfer/vLLM execution inventory

No test, model invocation, or benchmark is permitted in Tasks 5-9.  Source
inspection, implementation, and static route review are the only allowed work.
The earlier attention-only interpretation of this task was incomplete.

Pin every serving-critical component in the Mia launcher to its concrete source
implementation and one MTPLX disposition: existing exact implementation,
source-derived Metal port, or proven TP1/Metal non-applicability.  The required
inventory is:

1. B12X compressed sparse MLA for target and DSpark;
2. B12X sparse indexer scoring and exact top-512 selection;
3. fused target/indexer compressor, normalization, RoPE, quantization, and
   cache insertion;
4. B12X mHC pre, post-pre, final post, and head execution;
5. EXL3 Trellis MoE routing, route packing, W4A16 expert execution, and output
   reduction for decode and prefill;
6. B12X WO inverse-RoPE and the two FP8 projections;
7. general non-expert FP8 linear execution;
8. fixed-K5, K64 DSpark target taps, three draft stages, DFlash2 verification,
   and Markov head;
9. fixed-capacity page ownership and construction-owned scratch; and
10. eager warmup/compile ownership corresponding to Mia's full and piecewise
    graph capture.

CUDA graph padding fixes, TP/DCP collectives, and A8 MoE kernels are not copied
onto TP1 Metal merely because they appear in the source.  They may be marked
non-applicable only with a direct source/runtime reason in `SOURCE_PINS.md`.

Exit condition: every item has a closed disposition, including the exact file,
arithmetic/layout contract, construction owner, installed runtime callable,
and phase route.  A generic MLX expression is not an existing implementation
when Mia selected a fused component specifically to remove its intermediate or
dispatch chain.

## Task 6: Port fused compressor and cache insertion

**Files:**

- Modify `mtplx/deepseek_v4_nvfp4_kv.py`.
- Modify `mtplx/deepseek_v4_paged_indexer.py`.
- Modify `mtplx/models/deepseek_v4.py`.
- Add a focused Metal kernel module only if the existing cache modules cannot
  own the fixed Mia shapes cleanly.

Port the pinned `fused_compress_quant_cache` dataflow for ratio-4 target,
ratio-128 target, and ratio-4 indexer rows.  Preserve per-dimension window
softmax, gated reduction, RMS normalization, RoPE placement, post-RoPE
quantization, exact `stock432` NVFP4 records, and Mia132 FP8 index records.
Write already-packed records into the fixed page owner; do not materialize a
second dense compressed history and do not repack records in `append`.
Construction fixes the three geometries and owns incremental window state.

Exit condition: both full-prefill and incremental paths call the installed
fused packers directly and the old generic post-GEMM compressor chain is not
reachable on the Mia route.

## Task 7: Port sparse selection and compressed MLA

**Files:**

- Modify `mtplx/deepseek_v4_paged_indexer.py`.
- Modify `mtplx/deepseek_v4_nvfp4_kv.py`.
- Modify `mtplx/models/deepseek_v4.py`.
- Modify only the DeepSeek NVFP4 checks that gate these encountered
  contracts.

The failed long-context run proved that paging storage is necessary but not
sufficient.  The exact Mia path must not turn its fixed top-k index buffer into
a full `[query, compressed-context]` boolean matrix.  Port SparkInfer's bounded
K-tile/candidate-carry interchange as compact indices plus lengths.  Port the
exact selector itself; generic `topk`, `argpartition`, and `argsort` are not the
SparkInfer selector engine.  Bind candidate and carry storage once from the
installed maximum batch/context plan.
Do not run the generic DwarfStar Hadamard on Mia indexer Q/K: the pinned fused
Q kernel quantizes post-RoPE BF16 directly and the pinned compressor never
consumes its stored `rotate` flag.

SparkInfer's multi-head-group CUDA topology is translated through the existing
M5 NAX tensor primitive rather than copied by name.  Preserve the source's
16-head ownership, BF16 QK and P.V operands, FP32 QK/softmax/P.V accumulators,
native E2M1/E4M3 dequantization, and single softmax across the SWA/indexed-cache
union.  Metal's M16xN32xK16 tile determines a 32-candidate tile, four QK
K-splits, and eight PV SIMD groups covering the 512-value row.  Remove the
rejected scalar, half-MMA, and PV-only candidates.  Install the NAX engine at
construction for prefill and the measured direct kernel for decode; phase is
the only hot route and neither enabled lane has an eligibility check or silent
fallback.

Exit condition: decode uses the fixed fused score/select route, prefill uses the
bounded tiled score/candidate-carry route, and both feed compact top-512 indices
directly into the already installed decode/prefill MLA callables.  No generic
sort, full score history, boolean selection matrix, eligibility branch, or
silent fallback remains on the Mia route.

Commit: `feat: bound Mia NVFP4 sparse prefill`

## Task 8: Port mHC and EXL3 Trellis MoE execution

**Files:**

- Add a reusable DeepSeek V4 mHC Metal execution module.
- Modify `mtplx/deepseek_v4_exl3.py`.
- Modify `mtplx/models/deepseek_v4.py`.
- Modify `mtplx/models/deepseek_v4_dspark.py` only where the same installed mHC
  boundary is shared by the draft stages.

Port SparkInfer's planned mHC execution: initial pre, fused post-pre between
layers, final post, and the final head.  Preserve residual-stream ownership,
FP32 routing matrices, Gram-trick norm arithmetic, fixed hc=4 layout, and fused
RMS normalization.  Carry the installed mHC state through target taps and the
three DSpark layers without re-materializing separate generic post/pre chains.
For M>=384 post-pre boundaries, preserve SparkInfer's large-prefill separation
of BF16 POST/Gram, BF16 matrix projection with FP32 accumulation, and compact
Gram finalize.  The Metal translation uses the existing BM64/BK32
simdgroup-matrix geometry and construction-owned BF16 views of the FP32 block
routing matrices.  Initial pre and final head retain the source FP32 route;
M<384 post-pre retains split-32 FP32 for decode/verify/draft shapes.

Port the pinned EXL3 W4A16 Trellis route rather than keeping MTPLX's temporary
direct-QMV/sorted-MMA substitute.  Preserve six-expert routing, source route
histogram/prefix/packing ownership, M=1..32 decode geometry, the separate large-
M prefill plan, fused gate/up activation, down projection, route weighting, and
shared-expert addition.  Bind route scratch and expert workspaces at model
construction.  Do not sort or search routes with generic MLX operations in the
enabled path.

Exit condition: the target and draft layers use the installed mHC state machine
and installed Trellis MoE callables directly for their fixed phase routes.

## Task 9: Finish WO, FP8, DSpark, and engine installation

Port Mia's fused inverse-RoPE/output-projection boundary while reusing MLX's
native MXFP8 GEMM only where it preserves the pinned FP8 block-scale arithmetic
and layout.  Confirm all other non-expert FP8 linears are converted once at load
and never dequantized in the request path.

Load the 106 GB target through its pinned five carried shards and 43 complete
layer-local EXL3 shards, evaluating and releasing one source shard at a time.
The exact loader must not first build an all-shards source dictionary.  Seal a
bounded-loader receipt into the engine plan before DFlash can bind.

Install one immutable Mia engine plan after exact model/artifact validation. It
owns page geometry, compressor state, selector carries, MLA scratch, mHC state,
EXL3 route/work scratch, fixed phase callables, and the finite prewarm signature
set.  Preserve the pinned K64, fixed-K5 DSpark route, target layers 40/41/42,
three context-KV consumers, DFlash2 verification, and Markov bias.  Confidence,
capacity verification, dynamic depth, draft-head FP8, TP/DCP collectives, and
CUDA padding repairs stay absent because the exact launcher disables them or
TP1 Metal has no corresponding state.

Perform a source-to-installed-route audit only.  Reject the construction if an
invariant is not satisfied; the enabled hot path has no environment reads,
model metadata checks, proof counters, eligibility checks, or fallback.

Exit condition: Tasks 5-9 have no open disposition and every exact-model
forward component is installed.  Only then may Task 10 begin.

### Static implementation closure for Tasks 5-9

Tasks 5-9 are source-route complete.  This closure records implementation
state only; no Python, MLX, Metal, model, test, or benchmark execution was used
to reach it.

- The pinned Mia/vLLM/SparkInfer inventory has one closed MTPLX disposition per
  component in `docs/ports/deepseek-v4-dspark/SOURCE_PINS.md`, including the
  direct reasons for the TP1/CUDA-only omissions.
- Ratio-4 target, ratio-128 target, and ratio-4 index compression use installed
  fused record finalizers, separate previous/current projected-window views,
  fixed absolute-position compressor state rings, and direct persistent-page
  insertion.  The exact incremental route has no retained-array journal or
  previous-window concatenation.
- Sparse selection uses bounded score tiles, compact candidate carries, and the
  exact radix fold; decode and prefill pass compact indices/lengths directly to
  the installed direct/NAX stock432 MLA callables.
- Target and draft execute the installed carried mHC state machines.  Target
  K216 MoE uses the installed top-6 router, Trellis route packing, W4A16 expert
  kernels, activation/down boundary, and weighted/shared reduction; draft K64
  uses the installed native group-32 MXFP4 route.
- Inverse RoPE is fused at the grouped output boundary, and every other
  scale-bearing non-expert FP8 module is installed once as native group-32
  MXFP8 by the bounded one-source-shard loader.
- The immutable engine plan admits 384,000 logical tokens and owns 384,005
  physical target positions for terminal M6 verification, plus 8,416-row target
  window arenas, 128-row DSpark rings, fixed compressor state,
  selector/MLA/mHC/Trellis geometry, exact phase callables, and finite prewarm
  signatures.
- The pinned DFlash dependency is
  `54644e991039110f30140006c892c57734b9311e`.  Its fixed-linear lane has direct
  chunked prefill and M6 decode/verification, streaming structured taps,
  asynchronous next proposal, direct unarmed acceptance restore, and no
  adaptive, CopySpec, snapshot, diagnostic, cache-clear, or fallback work in
  the selected loop.

Task 10 is therefore the first permitted execution boundary.  It must begin
with the shortest construction/arithmetic and tiny exact-epoch gate; the long
benchmark ladder remains prohibited until that gate passes.

## Task 10: Post-port correctness and shortest performance gate

This is the first task allowed to invoke Python, MLX, Metal, the exact model, or
any test runner.  Under the GPU guard, first run only the focused required
construction/arithmetic checks and one tiny exact-model DSpark epoch.  Then run
the shortest sparse-path gate with committed-token parity and peak memory.
Stop on the first concrete failure and repair only that encountered failure.

Do not begin the requested ladder until this task passes.

## Task 11: Safe exact-model performance ladder

Do not begin this task until the complete Mia engine in Task 5 is implemented
and audited.  Then use the GPU service-restoration guard and
`/tmp/mtplx-gpu-exclusive.lock` for every Metal run.  Do not start a long-context
run merely because correctness passes.  Run only:

1. a tiny exact-model DSpark epoch with committed-token parity;
2. cold 1,024 prefill plus a short decode, with peak memory;
3. because Mia's fixed `index_topk=512` leaves a 1K prompt below the ratio-4
   sparse boundary, one cold 4,096 prefill plus six-token decode gate that is
   the smallest safe exercise of bounded sparse prefill;
4. only if the 4K sparse gate is correct and memory remains bounded, the
   requested roughly 100-token Python prompt and 1,024 decode, followed by cold
   1,024 prefill plus 1,024 decode;
5. cold 16,384 prefill plus 1,024 decode, with peak memory; and
6. only if the 16K memory headroom makes it safe, cold 65,536 prefill plus 1,024
   decode.

Each receipt records source/model revisions, prefill and decode tok/s, DSpark
acceptance, active and peak memory, generated-token digest, and cache-plan
identity.  If a run fails, diagnose that concrete failure and repair only what
blocks the next requested gate.

## Task 12: Review and publish

Run focused lint and only the cache, DeepSeek NVFP4, DSpark, DFlash2, runtime,
and directly affected GraphBank checks.  Inspect the implementation against the
immutable-plan/no-fallback/no-hot-validation constraints.  Push DFlash2 and
MTPLX commits, correct draft PR #312 to name the Mia/Sero artifact and exact
revisions, attach only valid receipts, and make it ready once every requested
gate passes.

## Task 13: Close the live serving audit

The final server audit found two request-lifecycle defects that the standalone
exact-model receipts did not exercise.  Close them without changing target or
draft arithmetic:

1. Bind DSpark to an explicit no-retokenized-cache-postcommit route.  The
   fixed-linear DFlash lane does not publish restorable prefix snapshots, so a
   generic SessionBank rebuild would create unusable references into Mia's
   reset-on-release singleton arena.  Preserve the logical EngineSession, but
   skip every direct, async, recursive, nonstream, and streaming-inline model
   postcommit before tokenizer or model work begins.
2. Extend the shared DFlash request contract with an immutable per-request
   prefill chunk size and a cancellation callback.  Poll cancellation only
   after settled prefill chunks and before speculative decode cycles; do not
   add per-token checks or change the selected fixed-linear loop.
3. Make DFlash session construction exception-safe after target-cache
   acquisition, including draft construction and generic sparse-RoPE setup.
   Adapter-owned Mia target and draft acquisitions must also unwind if their
   returned layout validation fails.
4. Pin and install the resulting DFlash commit before MTPLX can pass its real
   imported-runtime signature and source-identity gates.

Verification is limited to the directly affected DFlash runtime, adapter,
dependency-identity, and OpenAI route tests plus focused lint and diff checks.
The published long-model receipts remain tied to the earlier inference-core
pin; do not rerun them because these changes affect serving control and failure
cleanup, not model arithmetic, cache layout, or dispatch geometry.
