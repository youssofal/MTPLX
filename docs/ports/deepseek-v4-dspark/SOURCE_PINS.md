# DeepSeek V4 DSpark Port Source Pins

The current exact-Mia cold 16K/64K/128K Python vocabulary ladder and its raw
receipts are documented in [`BENCHMARKS.md`](BENCHMARKS.md).

## Clean implementation base

- MTPLX repository: `https://github.com/youssofal/MTPLX.git`
- Base: `upstream/main@2b0360ca1af5c383a797a9d96999540f3197f182`
- Worktree: `/Users/davidtai/projects/OpenSourceWTF/.worktrees/mtplx-deepseek-v4-dspark-k5`
- Branch: `feat/deepseek-v4-dspark-k5`

## Behavior references

- MiaAI-Lab launcher and patches: `MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark@d4ba142bc1d971eb73a911e207e3e963bbb3c455`
- MiaAI target artifact: `0xSero/deepseek-v4-flash-0731-spark@22f28d32b9b29b4352eaa380ff8c2c170b2847ab`
- MiaAI runtime image: `sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`
- RTX6K Discord community reference: `https://discord.gg/X54jjmcxWJ`, with
  its related RTX PRO 6000 / SM120 public implementation wiki pinned at
  `local-inference-lab/rtx6kpro@3633c2c6028056729a6612126e9afe05c2e3cf08`,
  especially `models/deepseek-v4-flash.md`. This is a cross-hardware runbook
  reference, not evidence that the pinned Mia/Sero DGX Spark launcher or this
  Metal port was validated on RTX PRO 6000.
- Image vLLM tree: `local-inference-lab/vllm@30038602b71395f481ef4a6edfe4fcf8551d9c15`
- Image-applied vLLM patch: `/tmp/vllm.patch` in immutable image layer
  `sha256:520582d536ce8491792f637f563699bc4139760a5f97e506ef1118d4cfb0a658`
- Image SparkInfer tree: `local-inference-lab/sparkinfer@272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`
- DeepSpec: `MiaAI-Lab/DeepSpec@005e03b81cec38b7da6399833d609ee89a2587f2`
- Official DSpark inference source: `DeepSeek-V4-Flash-DSpark@aa22cb07426656189b2573b8e77a9b7333b8ae0f`

## Reused DFlash2 runtime

- Existing MTPLX DFlash2 branch: `perf/qwen38-dflash2@c3487dc56de6c734c71508c1e293a44731ff025f`
- DFlash2 dependency: `davidtai/dflash-mlx@54644e991039110f30140006c892c57734b9311e`
- Fixed-linear ownership boundary: target physical pages, compressor journals,
  live frontiers, and DSpark rings are scheduled once per installed prefill or
  verify chunk; no logical cache rows are gathered or materialized.
- Imported MTPLX bridge commit: `4d3d03aa`
- Runtime authority: `dflash_mlx.engine.spec_epoch.SpeculativeSession`
- Generation authority: `dflash_mlx.runtime.stream_dflash_generate`

DeepSeek adds target and draft protocol adapters plus a construction-qualified
fixed-linear capability.  DFlash2 continues to own prefill orchestration,
speculative epochs, physical verification, acceptance, target rollback calls,
next-primary selection, events, and cleanup.  The pinned dependency selects a
direct fixed-linear subset once per request for this capability: snapshots,
sparse prompt positions, adaptive depth, diagnostics, cache clearing, generic
AR fallback, and CopySpec history are absent, while the existing verification,
greedy acceptance, accepted-feature commit, rollback, asynchronous next-draft
launch, stop, and event ordering are unchanged.
The fixed capability additionally certifies that acceptance restore trims the
installed target pages directly, so the shared loop does not run a disabled
per-cycle rollback-arm call or repeat the staged-primary proof.
The pinned runtime also accepts a per-request prefill chunk size and a
cancellation callback.  The callback is polled only after settled prefill
chunks and before speculative decode cycles; session construction and
sparse-RoPE installation unwind acquired target and draft caches on failure.
These serving-control additions do not change fixed-linear arithmetic, layout,
verification width, or event ordering.

The official `DSparkAttention` source establishes that persistent stage K/V is
context K/V projected from accepted target taps. The five neural draft rows are
combined with that context only for the proposal attention call and are not
retained in the stage cache. The MLX cycle therefore trims rejected target-M6
rows and inserts only the retained target-tap prefix into each stage's Mia stock432
context ring.

## Exact local Mia artifact

- Pinned source directory: `/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI`
- Source `config.json` SHA-256: `b001ec8308044aa11daa0e624f5aea5e5362a63c05879a83a7be046b00eada82`
- Source `model.safetensors.index.json` SHA-256: `61af5c0782a8651ef893004e84369d2281a0fc316c8bcefc0bd8f76244224649`
- Those two hashes match the files served by the pinned `22f28d32...`
  Hugging Face revision.
- TP1 rank-sliced directory: `/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-tp1`
- TP1 `config.json` SHA-256: `39f3a9e158019dc34dd943b64f874cfc43e9e392e6ce9215a56f2e183d661d90`
- TP1 `model.safetensors.index.json` SHA-256: `b7a450f88c99aee7f6d44ecb127e91e45ab5ccb1a0dad49ca9eabb90b400c304`
- TP1 canonical artifact seal: `c05e8ecb1d387cc351d9c5733689343ccca9d92c2be663954ce154bd43befd7d`.
  It pins the normalized manifest, the exact Mia/Sero source revision and small
  files, and the canonical safetensors header plus payload of every one of the
  48 target shards. Raw manifest and shard SHA-256 values remain truthful load
  receipts, but are not artifact identity: the pinned Mia image and a current
  local PyTorch materialization can emit byte-different safetensors JSON object
  order for the same tensor metadata and payload.
- Separately derived K64 DSpark directory: `/Users/davidtai/models/DeepSeek-V4-Flash-0731-spark-MiaAI-dspark-k64`
- K64 draft weight index SHA-256: `c0d0e18e8c84fe6f1b7dc6991a4ba5765d1965f21f8892887aa01169fc2ba2b3`
- K64 draft plan SHA-256: `d7a45cc065363ec79516593d8910d0be36e6e589d093ad6ab4a3603dbf92b426`
- K64 canonical draft-plan seal: `b371a65a0f452040109648551eb362bebfa802a15d92659a4cf0b90f23a57cd9`.
  The relocatable plan seal excludes only source path spelling and raw shard
  receipt hashes; the draft shard's canonical header-plus-payload SHA-256 is
  `5d510d98e9a744aa78724b54250c8f55e319fc4ab8d44db8b68aafc1cbfe6b15`.
- DSpark: block size 5, Markov rank 256, noise token 128799, target taps 40/41/42, stage namespaces 0/1/2.
- Persistent target and draft K/V use Mia-compatible 432-byte `stock432`
  NVFP4 records. Ratio-4 indexer records use Mia's 132-byte E4M3 plus FP32
  power-of-two scale layout.
- Mia's fused FP8 indexer quantizes post-RoPE BF16 Q/K directly.  The pinned
  compressor stores a `rotate` member but never consumes it, so the exact lane
  does not apply the generic DwarfStar Hadamard before FP8 quantization.
- Mia consumes compact top-k indices/lengths and never materializes a full
  query-by-context boolean selection matrix.  The immutable runtime image is
  `sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4`.
  Its compressed layer
  `sha256:13df22ffc5bb3e52c9c4e084dcd297ba17fd4a571ef95b8708b43711ab2509c8`
  contains `/tmp/sparkinfer.patch`; that patch adds the DSV4 NVFP4 traits
  H16, 288 block threads, and 256 math threads, and disables both native DSV4
  H8 and native DSV4 H16 for NVFP4.  Stock432 therefore uses the generic
  H16/288 BF16 arm, not the native H8 arm.

- MTPLX preserves the generic H16 owner in one 288-thread Metal threadgroup:
  eight tensor-math SIMD groups plus one coordination group.  A logical tile64
  is evaluated as two sequential native NAX N32 panels because the literal CUDA
  shared-memory panel does not fit Metal. During QK the 30 KiB arena aliases
  query `[0,16384)`, first-panel FP32 scores `[16384,18432)`, BF16 K operands
  `[18432,22528)`, and FP32 partials `[22528,30720)`; after panel two completes,
  its FP32 scores reuse the dead query range `[0,2048)`. Both score ranges stay
  read-only while the corrected BF16 probabilities reuse the dead QK operand
  range `[18432,20480)`; P.V value panels then use `[2048,10240)`. The 512-byte
  row/kind/softmax metadata brings total
  threadgroup allocation to 31,232 bytes, below 32 KiB. M1 launches four groups,
  physical M6 launches 24, and DSpark K5 launches 20.  Q, K, V, and
  unnormalized P cross BF16 boundaries; completed QK is scaled by log2(e),
  correction/probability use fast base-2 exponentiation, and QK, P.V, and the
  online-softmax state accumulate in FP32.  Physical M6 remains
  decode/verification.

- The mounted Mia prefill patch requests H32, tile64, and 384 CUDA threads,
  with about 92 KiB shared memory in the selected source.  That topology cannot
  be represented by a Metal threadgroup with the 32 KiB limit: H32xD512 BF16
  query storage alone consumes 32 KiB before KV, scores, or softmax state.  The
  installed Metal prefill engine is therefore an explicitly bounded arithmetic
  mapping, H16/tile32/256 threads in 28 KiB, not a claim of topology parity.
  The image-owned patched trait revision is not present in the local source
  extraction; the available upstream trait rejects DSV4 plus NVFP4 while the
  mounted prefill patch requests it.  Thus the packaged runtime's exact prefill
  trait enablement remains an evidence gap rather than an inferred fact.

## Historical implementation boundary

- Worktree: `/Users/davidtai/projects/OpenSourceWTF/.worktrees/deepseek-v4-0731-dspark`
- HEAD: `ea0c9d3968f8cac8dfc58805d92965c46943b45e`
- State at Phase 1 start: 141 porcelain-status rows; preserved without modification.

The historical implementation is evidence only. It conditioned DSpark on the previous target token, forced the authoritative primary into the first proposal row, and never implemented useful K5 with physical-M6 verification. Reuse only arithmetic and weight-layout facts independently re-established from the pinned sources above.

## Phase 1 route boundary

- Explicit daemon route: `mtplx/server/openai.py --generation-mode dspark`
- Fixed contract: `--depth 5 --temperature 0 --load-mtp`
- Runtime loader: `mtplx.runtime.load(..., dspark=True)` validates the artifact
  before model construction and does not invoke generic MTP injection,
  projection requantization, packed-MoE/NAX patching, or the unrelated generic
  post-load installer stack after the engine has been sealed.
- Worktree Python: `.venv/bin/python`, with the exact DFlash dependency installed
  from the pinned Git revision above.

## Closed source-to-installed-route inventory

This inventory is the implementation gate for the Mia lane.  Paths under
`vllm/` refer to the pinned image vLLM tree; paths under `sparkinfer/` refer to
the pinned SparkInfer tree.  Every MTPLX route below is selected and checked by
`load_mia_exl3_dspark_model` and `build_mia_engine_plan` before a request can be
created.  The enabled callables do not probe eligibility or fall back.

### 1. B12X compressed sparse MLA

- **Source:** `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`,
  `sparkinfer/attention/_shared/mla/prefill.py`, `prefill_mg.py`, `kernel.py`,
  `kv_cache.py`, and Mia's patched
  `image-patch/sparkinfer/moe/_shared/kernels/tiny_decode.py`.
- **Arithmetic and layout:** 64 query heads, 512-wide latent rows, 64 RoPE
  dimensions, a 128-token SWA unioned with selected compressed rows, stock432
  E2M1 plus E4M3 scales, BF16 QK/P.V operands, and one FP32 online softmax with
  the learned sink.  The selected generic decode geometry is H16, 288 block
  threads, 256 math threads, and one context split for physical M1/M6 and
  DSpark K5.  The 432-byte record stores
  all 512 post-RoPE compressed latent values in bytes `[0,256)`, their 32 E4M3
  group scales in `[256,288)`, zero
  padding in `[288,304)`, and a separate 64-value BF16 RoPE tail in
  `[304,432)`; that BF16 field duplicates the rotated tail while P.V uses its
  NVFP4-quantized copy from the full post-RoPE row.  Attention scaling is
  applied to the completed FP32 QK dot, not to each
  BF16 query element.  The mounted patched CUDA prefill source requests
  H32/tile64/384 threads; MTPLX's H16/tile32/256-thread Metal prefill mapping is
  the bounded substitute documented above, not exact source topology.
- **MTPLX implementation:**
  `mtplx/kernels/deepseek_v4_nvfp4_mla.py`.
- **Construction owner / installed callable:**
  `DeepseekV4Attention.install_mia_nvfp4_attention` installs
  `_mia_cached_forward_uncompressed`, `_mia_cached_forward_ratio4`, or
  `_mia_cached_forward_ratio128` from the layer's immutable compression ratio;
  the enabled target lane never enters the generic cache/no-cache branch.  It
  also installs ratio-specific prebound callables:
  `_run_installed_window_nvfp4_{prefill,sparse}_mla`,
  `_run_installed_indexed_paged_nvfp4_{prefill,sparse}_mla`, or
  `_run_installed_sequential_paged_nvfp4_{prefill,sparse}_mla`. DSpark installs
  the separately prebound ring/K5 callable. No installed hot launcher performs
  a kernel-cache lookup or receives route, selected-width, or block geometry.
  `MiaMLAWorkspace` owns invariant empty operands once.  Every target callable
  receives the cache owner's persistent physical window pages and block table;
  no target prefill or decode gathers `window.slice()` into a contiguous
  staging array.  Candidate addresses wrap through the descriptor's logical
  8,416-row capacity before block-table translation.  They never use the
  allocation's padded 8,448 physical-row count as the circular modulus.  The
  exact QKV prologue, MLA input, MLA output, and B12X boundary remain contiguous
  token-major `[B,M,H,D]`; the checked public oracle retains its separate
  `[B,H,M,D]` kernel variant.  Thus neither side of exact MLA materializes the
  64 MiB M1024 transpose that `ensure_row_contiguous` would otherwise force.
  `DeepseekV4TargetOps` explicitly labels every DFlash prompt chunk as prefill
  and every physical M6 target call as decode/verification, so the external
  DFlash engine cannot leave the phase at `unknown`.
- **Disposition:** source-derived Metal port.  Phase is the only runtime route.

### 1a. Fused learned Q/KV prologue and stock432 insertion

- **Source:**
  `vllm/csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu`
  from the pinned image vLLM tree, plus Mia's
  `image-patch/selftest_padded_stride.py` stock432 oracle.
- **Arithmetic and layout:** the row-adjacent MXFP8 Q-rank/WKV weights produce
  one BF16 `[M,1536]` owner buffer.  One learned-RMSNorm dispatch reads the
  1024-wide Q-rank prefix and 512-wide KV suffix without materializing either
  slice.  Post-`wq_b`, every 512-wide head is RMS-normalized in FP32; only its
  final 64 values receive interleaved RoPE in FP32, and the completed Q crosses
  one BF16 boundary.  For K/V, RoPE is applied to the final 64 values of the
  learned-normalized latent in FP32, the complete post-RoPE 512-wide row crosses
  one BF16 boundary, and all 512 values are NVFP4-quantized for V.  Its first
  448 values are also K-NoPE, while the same rotated BF16 tail is duplicated in
  bytes `[304,432)` for K.  Bytes `[288,304)` remain zero.  The CUDA launch uses
  the full per-slot grid when
  `M < 1024` and the reduced one-CTA-per-row grid when `M >= 1024`; bounded
  DFlash chunks are at most 1024, so exactly M1024 selects the reduced route.
- **MTPLX implementation:**
  `mtplx/kernels/deepseek_v4_qkv_prologue.py`, finalized-record owners in
  `mtplx/deepseek_v4_nvfp4_kv.py`, and the attention bindings in
  `mtplx/models/deepseek_v4.py` / `deepseek_v4_dspark.py`.
- **Construction owner / installed callable:** after weights, B12X WO, and the
  stacked MXFP8 projection owners are installed, the loader binds 46 distinct
  `MiaBoundQKVPrologue` plans.  Each plan retains its projection owner and
  query/KV learned-norm weights by identity and closes over epsilon.  Target
  prefill/decode returns Q plus finalized records; the fixed target cache owns
  visibility, retention, and frontier changes.  DSpark context uses the
  KV-only learned-norm entrypoint over offset 1024, while K5 proposal records
  are temporary and never mutate the three persistent rings.  Initial context
  zero-fills and marks the full physical 128-row ring; later 1..128 prompt
  increments and 1..6 generation commits use the separate modulo writer.
  M1024 reduced-grid finalization is prewarmed explicitly.
- **Functional-output adaptation:** `mx.fast.metal_kernel` exposes immutable
  inputs and newly allocated outputs, so it cannot alias the persistent cache
  buffer as CUDA's fused slot-mapping kernel does.  MTPLX therefore emits one
  bounded record output and performs one cache-owned scatter.  Physical block
  and offset maps are construction-precomputed (and rebuilt at state restore),
  so the writer does not rebuild `arange`/modulo/block mappings in the hot path.
- **Disposition:** source-derived Metal arithmetic with one documented MLX
  functional-output insertion dispatch.  Legacy latent-plus-RoPE packers are
  poisoned on the fixed target and exact DSpark routes.

### 2. B12X sparse indexer

- **Source:** `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`,
  `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py`,
  `sparkinfer/attention/nsa_indexer/fused_indexer.py`, `tiled_topk.py`, and
  `paged.py`.
- **Arithmetic and layout:** post-RoPE BF16 Q/K is quantized directly to E4M3;
  each 128-wide row is a 132-byte record with a FP32 power-of-two scale.  Scores
  are `sum_h(max(q_h dot k, 0) * w_h)`: tiled prefill decodes the raw E4M3
  operands into exactly representable FP16 values and completes each dot in
  FP32.  The CuTe Q finalizer folds weights in the exact FP32 order
  `(BF16 weight * combined weight scale) * Q row scale` and stores unit scale in
  the installed Q record.  Prefill and decode then accumulate raw
  ReLU-weighted heads and apply the positive K row scale once to the completed
  head sum.  For the packaged H64 long-prefill shape, SparkInfer gathers one
  paged 32K K supertile into contiguous records and selects a Q32xK512,
  256-thread scorer.  It batches seven Q heads in shared memory, keeps the FP32
  ReLU-weighted head reduction in registers, writes a bounded FP32 tiled-logit
  slab, and invokes the radix selector separately.  Metal preserves Q32 and
  256-thread ownership but uses K256: eight SIMD groups each own Q8xK128, K/Q
  are decoded through eight-dimension staging panels, and the completed
  source-order head sum remains in matrix fragments.  K512 would require 128
  live FP32 score/dot fragment values per Metal thread before input state;
  K256 requires 64 and 6,784 bytes of threadgroup scratch (6,912 for the
  ordinary-record oracle).  At K96K this gives 12,000 scorer threadgroups for
  Q1024 and 96,375 across the Q8224/512-MiB query splits, versus the prior K40
  path's 76,864 and 617,314.  The FP32 logit slabs remain capped at 128 MiB for
  Q1024 and 512 MiB per Q8224 subchunk.  Ordinary raw-record oracle scorers are
  separately compiled and consume their stored Q scales.  Selection is exact
  top-512 over only
  causal ratio-4 rows, represented as compact indices plus lengths.  The
  bounded prefill route carries candidates across score tiles; decode produces
  candidates in one fused pass and both use the same four-pass radix fold.
- **MTPLX implementation:** `mtplx/deepseek_v4_paged_indexer.py` and the
  `Indexer` installation seam in `mtplx/models/deepseek_v4.py`.
- **Construction owner / installed callable:** `MiaIndexerWorkspace` is sized
  for 8,224 query rows and top-512 by `build_mia_engine_plan`.
  `Indexer.install_mia_paged_topk` binds
  `_run_installed_indexer_query_records` and
  `_run_installed_paged_indexer_phase_topk`; the installed query finalizer
  accepts the construction-qualified record shape and a prebound FP32 weight
  scalar directly, and the selector routes only on prefill versus
  decode/verification.
- **Disposition:** source-derived Metal port.  Generic `argsort`,
  `argpartition`, a full score history, Hadamard rotation, and a
  query-by-context boolean mask are not reachable from the Mia route.

### 3. Fused compressor and cache insertion

- **Source:**
  The image-applied `/tmp/vllm.patch` over
  `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py`, plus the
  compressor/cache wiring in `vllm/models/deepseek_v4/attention.py`.  The
  immutable patch layer above is authoritative because the image applies it
  after checking out the pinned vLLM tree and before building the runtime op.
- **Arithmetic and layout:** FP32 projection outputs and gate logits are folded
  with per-dimension window softmax; ratio-4 includes the preceding half-window
  and ratio-128 does not.  Pooling, RMSNorm, and compressor RoPE remain FP32.
  The complete post-RoPE latent crosses one BF16 boundary before direct record
  quantization: stock432 packs all 512 rotated-row values as E2M1/E4M3 and
  duplicates its rotated tail64 as the BF16 K-RoPE field.  The indexer copy
  retains its separately documented post-RoPE Mia132
  E4M3 contract.  Full and incremental paths preserve the same completed-window
  frontier.
- **MTPLX implementation:** `mtplx/kernels/deepseek_v4_compressor.py`,
  `Compressor.mia_records`, `Compressor.step_records`, and paged
  `append_records` owners.
- **Construction owner / installed callable:** every compressed target
  attention installs `_nvfp4_record_impl`; every ratio-4 indexer installs
  `_indexer_record_impl`.  `DeepseekV4NVFP4Cache` owns the frontier and writes
  already-finalized records into its fixed pages.  The exact single-sequence
  route replaces vLLM's full request state pages with an arithmetic-equivalent
  absolute-position circular state cache sized only to the live compressor
  window plus the installed 64-row rollback allowance: `[21,72,2,1024]` for
  ratio-4 target state, `[21,72,2,256]` for its indexer state, and
  `[20,192,2,512]` for ratio-128.
  Incremental compression gathers its unfinished window from these fixed rows;
  it does not concatenate a retained Python journal or prepend the previous
  ratio-sized window to the current projected batch.  The fused finalizer takes
  the previous and current windows as separate device views and selects the
  previous source only for output row zero.
- **Disposition:** source-derived Metal port.  The generic pool/norm/RoPE and
  repack chain is not reachable with a Mia cache.

### 4. B12X mHC

- **Source:** `vllm/models/deepseek_v4/nvidia/model.py` and
  `sparkinfer/norm/mhc/_kernels.py`.
- **Arithmetic and layout:** hidden size 4,096, four residual streams, 20
  Sinkhorn iterations, FP32 routing matrices, the source Gram-trick norm, BF16
  carried values, fused RMSNorm at the following branch, final post, and head
  collapse.  The head collapse is stored at the source BF16 boundary before the
  model-owned RMSNorm; the norm scale is therefore computed from the rounded
  collapse rather than an analytical pre-rounding Gram.  Target taps are
  reconstructed after layers 40, 41, and 42 without
  breaking carried residual ownership; the layer-42 reconstruction is reused
  as the final trunk state exactly as in the pinned vLLM path.  The large-M
  repeated `post_pre` route follows SparkInfer's prefill split: one BF16
  POST/Gram producer, a BF16 matrix projection with FP32 accumulators, and the
  compact Gram finalize.  Initial pre and final head retain the source FP32
  split reduction.  The SM120 default selects the sibling TF32 projection at
  this boundary; Metal has no TF32 matrix operand mode, so the port selects
  SparkInfer's co-shipped BF16 projection and the same construction-time
  FP32-to-BF16 routing-weight conversion used by the pinned vLLM model.
  Metal uses its existing 8x8 simdgroup matrix primitive with BM64, BK32, and
  padded N32 for the source N24 projection; M below 384 retains the source
  split-32 FP32 route used by M6 verification and K5 drafting.  Target and
  draft construction also reproduce `finalize_mhc_broadcast_weights`: the
  first attention `fn` is summed over its four identical input streams once,
  and initial pre consumes the resulting FP32 `[24,4096]` matrix directly.
- **MTPLX implementation:** `mtplx/kernels/deepseek_v4_mhc.py` and the fixed
  `_run_mia_hc_target_tail_taps` / `_mia_propose_k5` state machines.
- **Construction owner / installed callable:** one `MiaMHCPlan` for the
  43-layer target and one for the three DSpark stages.  The target binds
  `_mia_hc_hidden` plus `_mia_collapse`; the draft binds `_mia_propose_k5`.
  Construction materializes each block's source FP32 `fn` matrix's BF16 MMA
  view once, seals M384/BM64, and owns compact `[M,11]` Gram and `[M,24]`
  projection outputs at repeated large-M post-pre boundaries.  Initial/head
  FP32 partials and the M<384 post-pre partials remain explicit source routes;
  the initial route owns the precollapsed FP32 broadcast matrix.  Target and
  draft heads bind `head_bf16_then_rmsnorm`, preserving the separate norm call
  without a hot-path eligibility branch.
- **Disposition:** source-derived Metal port for target prefill, target M6
  verification, and draft K5.

### 5. EXL3 Trellis target MoE and routing

- **Source:** `vllm/models/deepseek_v4/nvidia/model.py`,
  `sparkinfer/gemm/trellis_linear/_small_m.py`, and
  `sparkinfer/moe/fused_moe/_impl.py`; the weight format is pinned by the target
  `EXL3_MANIFEST.json` and its ExLlamaV3 revision.
- **Arithmetic and layout:** K216, top-6 sqrt-softplus routing with correction
  bias except for the first three token-hash layers.  Softplus retains the
  source's exact `x > 20 ? x : log1p(exp(x))` arithmetic before the square
  root; the correction bias affects selection only, and unbiased selected
  scores are normalized then scaled by 1.5.  Source histogram, prefix, and
  expert-major route packing feed EXL3 MCG trellis payloads with H128 input
  transforms; fused clamped gate/up activation; down projection; route weight
  reduction and shared-expert addition.  Target prefill uses Trellis BM8 through
  M127 and BM64 from M128, while the exact physical M6 verification phase uses
  the source-backed four-row direct-QMV projection sequence and stock
  weighted tail.
  Each direct bank owns an N256 output slab: one 128-thread Metal group reuses
  its input H128 across two independently accumulated and independently
  FP16-rounded N128 panels.  This preserves the artifact's H128/K128 packing
  while mapping SparkInfer's pinned direct-top-k N256 ownership to Metal; it
  does not transplant SparkInfer's CUDA K tile or thread topology.  Within
  each H128, strides 1 through 16 use the repository's existing
  `simd_shuffle_xor` exchange and only cross-SIMD strides 32 and 64 use
  threadgroup scratch.  The ascending FP32 butterfly order and every FP16
  boundary remain unchanged.  Consecutive four-row groups map to MCG states
  `t,t+1,t+8,t+9` in the pinned tensor-core permutation.  A
  construction-sealed descriptor reconstructs the two original 19-bit
  adjacent-pair windows from shared words: fixed q0/q2 paths issue three loads
  and fixed q1/q3 paths issue two, with no optional-load branch.  This is 10
  loads per 16 K values and output panel instead of the previous 16.  The
  quad decoder preserves the four independent MCG products and ascending
  k0,k1,k2,k3 FP32 FMA order; it changes decoder window/address work, not the
  artifact arithmetic.  The retained `u4-stage16b-96x8` lane
  retains the same 48-ushort tile and 12,288-byte threadgroup layout, but stages
  it through aligned `uint4` storage.  Lanes 0 through 95 (three complete
  SIMDgroups) issue one 16-byte copy for each of the eight K tiles; expert, K,
  and N vector bases are formed before that copy loop.  Construction seals the
  16-byte vector and 96 vectors per K tile before installing the v2 kernel.
  Authentic three-bank/final-bit parity passed.  The retained
  construction-bound piecewise target route compiles only cache-free regions
  around the same eager attention calls while preserving the source BF16 head
  boundary before RMSNorm.  Only sustained 1,024-output requests are published
  as decode-throughput evidence.
- **MTPLX implementation:** `mtplx/kernels/deepseek_v4_moe_router.py` and the
  installed `EXL3SwitchGLU.direct_qmv_m6_quad` /
  `EXL3SwitchGLU.fused` paths in `mtplx/deepseek_v4_exl3.py`.  The scalar
  `EXL3SwitchGLU.direct_qmv` remains the exact oracle and generic installation
  predecessor.
- **Construction owner / installed callable:** each target `DeepseekV4MoE`
  first binds the exact scalar oracle, then installs all quad plans before
  rebinding `direct_qmv_m6_quad`; Trellis `fused` and
  `_stock_moe_tail_combine` are also fixed before `_mia_exl3_forward`, and each
  gate binds `_mia_hash_route` or `_mia_score_route`.  Each quad plan seals
  the exact Mia geometry, descriptor digest, BN256 ownership, and both compiled
  kernels at construction; the enabled M6 call contains no geometry
  eligibility, kernel factory lookup, or fallback.
  `EXL3SwitchGLU.install_trellis_runtime` binds the BM8 and BM64 plans.  The
  runtime selector uses direct QMV only when the phase is `decode_verify` and
  the physical hidden row count is exactly six; every prefill and all other
  widths use Trellis.  `_pack_trellis_routes` remains the enabled Trellis
  packer.
- **Disposition:** source-derived Metal W4A16 port.  Quad direct QMV is the
  measured production M6 arithmetic with uniform BN256 ownership and the
  source-equivalent register/SIMD H128 butterfly.  The `uint4` staging change
  is retained after guarded authentic parity and two matched short-cycle wins;
  scalar direct QMV and the generic `__call__` M-width selection remain
  explicit compatibility/oracle code outside `_mia_exl3_forward`.

### 6. WO inverse-RoPE and two projections

- **Source:** `sparkinfer/gemm/_shared/wo_mxfp8.py` and
  `sparkinfer/gemm/wo_projection/api.py`.
- **Arithmetic and layout:** a standalone producer applies inverse RoPE while
  quantizing each 32-value grouped WO-A input block to E4M3 with Mia's
  ceil-power-of-two UE8M0 scale.  It emits only `[8,M,4096]` byte values plus
  `[8,M,128]` byte scales, never a BF16 `[M,32768]` tensor.  Grouped WO-A
  reinterprets the artifact's byte-identical E4M3/expanded-UE8M0 storage,
  accumulates in FP32, and normally crosses the model boundary once as BF16
  `[M,8,1024]`.  At Spark's exact M16 route, WO-A instead applies the source
  group-32 E4M3/UE8M0 quantization directly to its FP32 accumulator and feeds
  WO-B without materializing the BF16 boundary.  At M1/M5/M6, WO-B quantizes
  the BF16 boundary inside its input stage; other bounded prefill rows use a
  standalone group-major E4M3/UE8M0 producer.  WO-B likewise accumulates in
  FP32 and returns BF16 `[M,4096]`.
- **MTPLX implementation:**
  `mtplx/kernels/deepseek_v4_wo_mxfp8.py`.
- **Construction owner / installed callable:** after both target and draft
  weights load, `install_mia_tp1_wo_projection_routes` installs 46 distinct
  `MiaTP1WOMXFP8Plan` objects directly as `_output_projection_impl`.  The
  plans retain each attention's existing `uint32` weight view and expanded
  `uint8` scales by identity and prebind the BM8, exact-M16 quantized-output,
  and BM64 kernels.  Target and draft plans carry distinct construction roles,
  and bind their WO-B owners once: target binds MLX's native group-32 MXFP8
  matmul while draft binds the exact FP32-accumulating Metal owner.  The engine
  verifies the complete plan/callable/storage/parameter-identity receipt before
  allocating serving caches or prewarming; no runtime eligibility or fallback
  branch remains in the installed plan.
- **Disposition:** source-derived TP1 Metal mapping with standalone WO-A
  producer and grouped MXFP8 GEMMs.  Target M6 verification first applies the
  exact source K32 E4M3/UE8M0 activation quantizer, reconstructs its BF16
  boundary, and feeds the already-installed native MXFP8 WO-B weights.  Against
  the exact owner on the authentic M6 layer-0 tensors, this construction-bound
  target route differs in only 3 of 24,576 BF16 outputs with maximum absolute
  drift 0.015625.  Draft and prefill retain the exact ordered BF16-tile,
  FP32-accumulating owners.  The generic
  `_MiaInverseRopeGatherOLora` route remains available only to explicitly
  constructed non-Mia models and is not reachable from Mia execution.

### 7. General non-expert FP8 linears

- **Source:** the FP8 module declarations in
  `vllm/models/deepseek_v4/nvidia/model.py` and the artifact's dynamic E4M3,
  128-by-128 E8M0 scale contract.
- **Arithmetic and layout:** E4M3 payload bytes stay unchanged.  Each 128-by-128
  E8M0 source scale is repeated into the equivalent native group-32 scale view;
  weights are viewed as `uint32` without decoding or requantizing.
- **MTPLX implementation:** `_expand_mia_fp8_block_scales`, target/draft
  sanitizers, and `_quantize_loaded_modules` in `mtplx/deepseek_v4_exl3.py`.
- **Construction owner / installed callable:** the loader replaces every
  scale-bearing non-expert linear with `nn.QuantizedLinear(mode="mxfp8")` once,
  records the complete module map, and installs the 106 GB target through five
  bounded carried shards plus one complete 2 GB EXL3 layer shard at a time.
  Source arrays are evaluated into their destination and released before the
  next shard; `build_mia_engine_plan` rejects a missing streaming-load receipt
  or any module-map mismatch.  Request execution calls the installed native
  MXFP8 operator.
- **Disposition:** existing exact native storage/operator implementation; no
  request-path dequantized weight copy exists.

### 8. K64 fixed-K5 DSpark and DFlash2

- **Source:** `vllm/models/deepseek_v4/nvidia/dspark.py`, Mia's patched
  `image-patch/vllm/models/deepseek_v4/nvidia/dspark.py`, and the pinned
  DFlash2 `SpeculativeSession` / `stream_dflash_generate` runtime.
- **Arithmetic and layout:** post-layer target taps 40/41/42 feed one main
  projection; three draft stages own K64 routed experts; proposal input is the
  accepted primary plus four noise IDs; the five sequential Markov-biased
  argmax rows are future tokens.  Persistent draft K/V contains only accepted
  target context in a chronological stock432 128-row ring.  Each proposal sees
  that context plus five temporary neural rows.  DFlash2 physically verifies
  primary plus five futures (M6) and commits only the accepted prefix.
- **MTPLX implementation:** `mtplx/models/deepseek_v4_dspark.py`,
  `mtplx/deepseek_v4_dflash2.py`, and the DeepSeek binding in
  `mtplx/benchmarks/dflash2_runtime.py`.
- **Construction owner / installed callable:** the separately validated K64
  package constructs `DeepseekV4DSparkOwner`; its three expert banks use native
  group-32 MXFP4, its gates use `_mia_score_route`, and its proposal callable is
  `_mia_propose_k5`.  Each attention stage binds `_run_k5`, the fused
  `MiaBoundQKVPrologue`, a KV-only context finalizer, and separate initial and
  incremental ring writers.  Its installed
  `_run_dspark_k5_nvfp4_mla` consumes the persistent 128-row ring and five
  proposal-local records as separate inputs to one online softmax, using
  `absolute_position % 128` for physical ring ownership.  It therefore does
  not concatenate a 133-row cache view, build visibility indices or lengths,
  allocate a temporary cache owner, or revalidate static record geometry.
  `DeepseekV4DSparkBackend.draft_greedy` is the DFlash2 backend;
  `DeepseekV4StreamingTargetFeatureStore` carries the three target taps as a
  tuple-native MLX tree and releases each prompt chunk after slicing its raw
  taps to the final 128 rows.  Only that at-most-128-row tail is concatenated
  for the main projection before inserting the three context-K/V copies; the
  full 1,024-by-12,288 tap tensor is never materialized.  Neither the projected
  DSpark context nor stage K/V is materialized for discarded prompt rows.
- **Disposition:** source-faithful fixed-K5, temperature-0 DSpark/DFlash
  protocol subset reusing the
  existing DFlash2 engine.  There is no full-prompt target-feature store.

### 9. Pages and bounded workspace ownership

- **Source:** vLLM's fixed KV block tables in
  `vllm/v1/attention/backends/mla/b12x_mla_sparse.py`, SparkInfer's paged MLA and
  indexer modules above, and `sparkinfer/attention/sparse_mla/_scratch.py`.
- **Arithmetic and layout:** request admission is fixed at 384,000 logical
  tokens.  Physical target storage and the shared target/indexer RoPE tables
  own 384,005 positions because DFlash always executes one target row plus its
  five draft rows before trimming a terminal partial block.  Each
  target layer owns an 8,416-row logical circular stock432 arena: the 8,224
  maximum input batch plus the logical 128-row attention window and installed
  64-row rollback allowance.  It is backed by 132 64-row pages (8,448
  allocated rows); the final 32 padding rows are not part of the circular
  address space.  Every query
  exposes only its causal 128-row window.  Ratio-4 and
  ratio-128 layers own `ceil(384005 / ratio)` persistent stock432 records:
  96,002 at ratio 4 and 3,001 at ratio 128.  Ratio-4 layers additionally own
  96,002 Mia132 index records.
  DSpark owns three persistent physical 128-row rings.
- **MTPLX implementation:** `mtplx/paged_cache.py`, paged owners in
  `mtplx/deepseek_v4_nvfp4_kv.py`, and `MiaDeepseekV4EnginePlan`.
- **Construction owner / installed callable:** the immutable engine plan owns
  page geometry, shared indexer carry seeds, invariant MLA operands, and the
  fixed Metal threadgroup geometry, fixed compressor state rings, one persistent
  target page lease, and one persistent DSpark ring lease.  Per-dispatch Metal
  result arrays are bounded functional outputs, not growing histories.  Cache
  writes use construction-precomputed physical maps and one functional scatter
  into the installed pages; request cleanup resets logical frontiers without
  reallocating their physical storage.
- **Disposition:** reusable fixed-capacity MTPLX paging specialized by exact
  DeepSeek record specs.  No geometric cache growth is on the Mia route.

### 10. Warmup and graph/callable ownership

- **Source:** `vllm/model_executor/warmup/b12x_sparse_indexer_warmup.py`,
  `deepseek_v4_compressor_warmup.py`, `kernel_warmup.py`, and Mia's eager
  launcher/capture configuration.
- **Arithmetic and layout:** the finite serving signatures are target M6
  prefill Trellis BM8, target M128 prefill Trellis BM64, M384 mHC post-pre BF16
  MMA BM64, exact-M16 WO-A quantized output, sparse-indexer prefill,
  sparse-indexer decode, target M6 verification direct QMV, and three-stage
  DSpark K5 BM8.  A 128-row target prefill reaches both compressor ratios; a
  direct first-layer M6 prefill compiles its distinct Trellis BM8 route; direct
  M384 mHC and M16 WO calls cover their distinct installed kernels without
  another full target pass; a 513-row synthetic paged index view reaches
  top-512 without a long model forward.
- **MTPLX implementation:** `MiaDeepseekV4EnginePlan.prewarm` and
  `mtplx/deepseek_v4_mia_piecewise.py`.
- **Construction owner / installed callable:** the loader evaluates all weights
  and prewarms every signature before returning.  `DeepseekV4TargetOps` refuses
  to bind DFlash2 without a matching immutable-plan prewarm receipt.  The
  DeepSeek DFlash context fixes batch-one prompt chunks at Mia's 1,024-token
  long-prefill threshold while workspaces retain the 8,224-token scheduler
  cap; it fixes the draft window at 128, installs the dependency's fixed-linear
  M6 lane, and retains target/draft allocator state between chunks and requests.
- **Disposition:** MLX compile/prewarm equivalent of source eager warmup and
  piecewise capture.  Cache-free regions are compiled at the fixed physical M6
  width; all attention/cache ownership stays eager.  The first request cannot
  become the compiler trigger.

## Source features intentionally absent on TP1 Metal

- **CUDA graph padding and replay repair:** these exist to keep CUDA graph
  addresses and batch buckets stable.  MTPLX has no CUDA graph or CUDA replay
  metadata.  The exact lane instead installs fixed batch-one K5/M6 callables,
  fixed page addresses, and eagerly compiles every MLX/Metal phase.  Copying the
  CUDA padding topology would add different arithmetic and state without a
  consumer.
- **TP/DCP collectives:** Mia's `entrypoint-no-download.sh` pins `TP_SIZE=1` and
  `DCP_SIZE=1`; DSpark non-causal attention also rejects DCP greater than one.
  The rank-coalesced artifact owns every K216 expert locally, so no collective
  or partial-output reduction exists in this route.
- **B12X A8 MoE activation path:** the NVIDIA `b12x-a8` lane is an SM120
  Tensor-Core/CuTe execution contract with dynamic FP8 activation tiles,
  N256/K128 geometry, and an in-place W4A8/QMMA weight representation.  Metal
  exposes neither that primitive nor that representation.  The target artifact
  separately pins EXL3 Trellis expert payloads, for which MTPLX installs the
  source-derived W4A16 Trellis route.  The K64 draft payload is native OCP MXFP4
  and maps byte-identically to MLX group-32 MXFP4.  Transplanting the A8 tile
  topology onto either format would change layout and arithmetic, so it is not
  part of the TP1 Metal port.
- **Confidence/capacity/dynamic-depth/draft-head FP8:** `start.sh` fixes
  `DSPARK_TOKENS=5` and `DSPARK_CAPACITY=0`; the packaged path does not enable
  dynamic depth or draft-head FP8.  No corresponding heads, counters, or hot
  branches are installed in MTPLX.
