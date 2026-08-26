# DeepSeek V4 Mia `stock432` NVFP4 K/V Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers-optimized:executing-plans`.  Execute serially because the record,
> arithmetic, attention, and real-model gates share one evolving contract.

**Goal:** Replace the exact Mia target and draft affine-int4 K/V lane with native
432-byte NVFP4 records and a bounded source-derived Metal attention engine.

**Architecture:** The exact record arithmetic remains authoritative, but
storage ownership is superseded by the reusable system plan in
`docs/plans/2026-08-21-system-paged-cache.md`. Target and DSpark
produce normalized latent plus a rotated tail; the writer NVFP4-quantizes the
complete BF16-rounded post-RoPE 512-wide V row and duplicates the rotated tail
as BF16 for K. Construction-installed
Metal consumers read selected records directly without whole-cache
dequantization or whole-context scores. Decode uses the measured one-head
online-softmax kernel. Prefill uses the existing M5 NAX tensor primitive with
Mia's 16-head ownership, native BF16 QK/P.V operands, and one FP32 online
softmax over the window/indexed-cache union.

**Tech Stack:** Python 3.11, MLX 0.32, `mx.fast.metal_kernel`, pytest, existing
DeepSeek V4/DFlash2 adapters.

**Assumptions:** This plan targets only Mia `stock432`, head width 512, NoPE 448,
RoPE 64, group 16, batch 1, and the pinned K216/K64 artifacts.  It will NOT work
for MLX `mxfp4`, Mia `rope368`, the community UE8M0/360-byte format, other head
geometry, or another model family.

**Design:** `docs/specs/2026-08-21-deepseek-v4-mia-nvfp4-kv-design.md`

## Execution status — 2026-08-22

The artifact, ownership, and original CPU/static portions of this plan are
complete on `feat/deepseek-v4-dspark-k5`:

- exact Mia K216 target and Sero K64 draft artifact identity, tensor ownership,
  cross-shard MXFP8 scale mapping, tokenizer identity, 384K admission, and
  DFlash commit identity are validated before model execution;
- `stock432` NVFP4 record production, compressor arithmetic, bounded indexer
  geometry, fixed MLA route installation, sealed cache snapshot/restore, Trellis
  launch geometry, carried mHC bindings, and DFlash lifecycle ownership are
  implemented without enabled hot-path fallbacks;
- the exact 48-shard target plus K64 draft construction gate passes, and the
  consolidated CPU/static integration gate passes after the final artifact and
  carried-mHC corrections.

The final source-parity review then found performance-path omissions that make
the earlier execution receipts invalid.  The construction-installed Trellis,
indexer, compressor, raw stock432, MLA, stacked input projections, shared
target/draft RoPE providers, exact mismatch gate, fused learned-Q/KV plus
Q-normalize/RoPE/stock432 prologue, and TP1 B12X MXFP8 WO chain are now
implemented and construction-bound for all 43 target and three draft owners.
The remaining implementation review is closing direct fixed-page consumption
in MLA so the exact target route does not gather its visible raw window after
each cache write.  No long model run or benchmark is permitted before that
review, the upstream merge, and the focused Metal gates complete.

The remaining gates must execute in this order:

1. finish and construction-bind the source-derived Q/KV prologue and TP1 B12X
   WO kernels for all 43 target and three draft owners, then complete the
   independent whole-branch review and consolidated static gate;
2. commit the complete implementation, merge the current upstream `main` into
   the published PR branch, resolve and verify the six overlapping paths, then
   create a clean execution worktree at that exact SHA;
3. acquire the exclusive Metal lane and run only the focused byte/parity gates
   for the Q/KV prologue, B12X WO chain, compressor, paged indexer, and every
   installed MLA route/shape, including the ratio-128 tile-crossing case;
4. run one short exact Mia/Sero generation and committed-token parity gate;
5. only after those pass, run the requested cold 1K, 16K, and 64K prefill matrix
   with 1,024 output tokens, prefill TPS, decode TPS, and peak memory;
6. update receipts and publish the PR with the exact Mia/Sero artifact and pinned
   DFlash commit named explicitly.

No long benchmark receipt is valid until the focused Metal and exact-generation
gates above pass.

---

## File Structure

- Create `mtplx/deepseek_v4_nvfp4_kv.py`: exact record constants, pack/decode
  kernels, oracle decode, and appendable row owner.
- Create `mtplx/kernels/deepseek_v4_nvfp4_mla.py`: direct record-consuming sparse
  online-softmax attention for target and fixed-window DSpark inputs.
- Modify `mtplx/models/deepseek_v4.py`: full post-RoPE target arithmetic,
  NVFP4 cache owner, selected-index route, and construction binding.
- Modify `mtplx/models/deepseek_v4_dspark.py`: three NVFP4 rings and distinct K/V.
- Modify `mtplx/deepseek_v4_dflash2.py`: construction checks and bounded prefill.
- Modify `tests/test_deepseek_v4_affine_kv.py`: replace the superseded affine
  contract with the required `stock432` record/cache contract and rename it.
- Modify `tests/test_deepseek_v4_dspark_model.py` and
  `tests/test_deepseek_v4_dflash2_adapter.py`: update only required owner and
  integration assertions.
- Modify `scripts/deepseek_v4_dspark_k5_bench.py`: report `stock432` rather than
  affine-int4 in exact-model receipts.

### Task 1: Exact `stock432` record owner

**Files:**
- Create: `mtplx/deepseek_v4_nvfp4_kv.py`
- Rename: `tests/test_deepseek_v4_affine_kv.py` to
  `tests/test_deepseek_v4_nvfp4_kv.py`

**Security flag:** `none`

**Does NOT cover:** model attention, DSpark wiring, alternate NVFP4 layouts, or
fused sparse consumption.

- [x] **Step 1: Write the failing record contract check**

```python
def test_mia_stock432_record_quantizes_full_post_rope_value_and_duplicates_key_tail():
    latent = fixed_bf16_rows(shape=(1, 2, 512))
    rope = fixed_bf16_rows(shape=(1, 2, 64))
    rows = MiaNVFP4Rows()
    rows.append(latent, rope)
    key, value = rows.decode()
    post_rope = concatenate([latent[..., :448], rope], axis=-1)
    assert rows.records.shape == (1, 2, 432)
    assert rows.records.dtype == mx.uint8
    assert mx.array_equal(rows.records[..., 288:304], mx.zeros((1, 2, 16), mx.uint8))
    assert bytes(rows.records[0, 0, 304:432]) == bf16_bytes(rope[0, 0])
    assert_allclose(key[..., :448], value[..., :448])
    assert_allclose(key[..., 448:], rope)
    assert_allclose(value[..., 448:], decode_nvfp4(post_rope)[..., 448:])
```

- [x] **Step 2: Verify RED**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_deepseek_v4_nvfp4_kv.py
```

Expected: import failure for `mtplx.deepseek_v4_nvfp4_kv`.

- [x] **Step 3: Implement the fixed codec and owner**

```python
NVFP4_HEAD_DIM = 512
NVFP4_NOPE_DIM = 448
NVFP4_ROPE_DIM = 64
NVFP4_GROUP_SIZE = 16
NVFP4_PACKED_BYTES = 256
NVFP4_SCALE_BYTES = 32
NVFP4_ROPE_OFFSET = 304
NVFP4_RECORD_BYTES = 432

class MiaNVFP4Rows:
    records: mx.array | None
    def append(self, latent: mx.array, rope: mx.array) -> None: ...
    def replace(self, start: int, latent: mx.array, rope: mx.array) -> None: ...
    def drop_first(self, count: int) -> None: ...
    def truncate(self, length: int) -> None: ...
    def decode(self, start: int = 0, stop: int | None = None) -> tuple[mx.array, mx.array]: ...
```

The Metal pack kernel substitutes the rotated tail into the complete row,
crosses the BF16 boundary, calculates each group-16 scale from `amax / 6`,
encodes it as finite E4M3, uses that decoded scale for nearest/saturating E2M1
packing, writes zero padding, and duplicates the rotated tail as BF16 bytes. The decoder
uses the fixed E2M1 table and E4M3 bit decoder already established by
`mtplx.compressed_tensors`.

- [x] **Step 4: Verify GREEN and owner mutations**

Run the same test file.  Expected: record, replacement, truncation, and state
round-trip checks pass.

- [x] **Step 5: Commit**

```bash
git add mtplx/deepseek_v4_nvfp4_kv.py tests/test_deepseek_v4_nvfp4_kv.py docs/specs docs/plans
git commit -m "feat: add Mia stock432 NVFP4 cache records"
```

### Task 2: Correct target and DSpark K/V semantics

**Files:**
- Modify: `mtplx/models/deepseek_v4.py`
- Modify: `mtplx/models/deepseek_v4_dspark.py`
- Modify: `mtplx/deepseek_v4_dflash2.py`
- Modify: `tests/test_deepseek_v4_nvfp4_kv.py`
- Modify: `tests/test_deepseek_v4_dspark_model.py`
- Modify: `tests/test_deepseek_v4_dflash2_adapter.py`

**Security flag:** `none`

**Does NOT cover:** direct sparse Metal attention; this task uses record decode as
the required arithmetic bring-up gate and is not considered the finished lane.

- [x] **Step 1: Write failing construction and arithmetic checks**

The checks require `DeepseekV4NVFP4Cache` for all 43 target layers,
`MiaNVFP4Rows` for all three draft rings, the post-RoPE NVFP4 row as V,
first-448 plus stored BF16 RoPE as K, and exact target/draft trim/replace behavior.

- [x] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest -q \
  tests/test_deepseek_v4_nvfp4_kv.py \
  tests/test_deepseek_v4_dspark_model.py \
  tests/test_deepseek_v4_dflash2_adapter.py
```

Expected: failures naming affine owners or the old shared rotated K/V result.

- [x] **Step 3: Install the source-correct owners and arithmetic**

```python
class DeepseekV4NVFP4Cache(DeepseekV4Cache):
    window: MiaNVFP4Rows
    compressed: MiaNVFP4Rows
    def update_window(self, latent, rope): ...
    def update_compressed(self, latent, rope): ...
    def attention_window_records(self): ...
    def attention_compressed_records(self): ...
```

Split attention projection into `(latent, rotated_rope)`, form and BF16-roundtrip
the complete post-RoPE row before NVFP4 packing, and retain the source model's
output inverse RoPE. Bind `DeepseekV4NVFP4Cache` and
the three NVFP4 draft rings once at exact-artifact construction.  DFlash2 rejects
any non-`stock432` owner before generation.

- [x] **Step 4: Verify GREEN**

Run the three files above.  Expected: all direct cache/arithmetic/adapter contracts
pass; no affine owner remains on the enabled exact-Mia route.

- [x] **Step 5: Commit**

```bash
git add mtplx/models/deepseek_v4.py mtplx/models/deepseek_v4_dspark.py \
  mtplx/deepseek_v4_dflash2.py tests/test_deepseek_v4_nvfp4_kv.py \
  tests/test_deepseek_v4_dspark_model.py tests/test_deepseek_v4_dflash2_adapter.py
git commit -m "fix: restore Mia NVFP4 key value arithmetic"
```

### Task 3: Complete bounded sparse Metal engine

**Files:**
- Create: `mtplx/kernels/deepseek_v4_nvfp4_mla.py`
- Modify: `mtplx/models/deepseek_v4.py`
- Modify: `mtplx/models/deepseek_v4_dspark.py`
- Modify: `tests/test_deepseek_v4_nvfp4_kv.py`

**Security flag:** `none`

**Does NOT cover:** generic attention dimensions, batching beyond one request,
other record layouts, non-M5 prefill, or a runtime fallback.

- [x] **Step 1: Write the failing direct-consumer comparison**

Use fixed `stock432` records, 64 query heads, a 128-row causal window, selected
compressed indices, learned sinks, and both M1 and M6 query shapes.  Compare the
Metal output with an online-softmax oracle reconstructed from the same records.

- [x] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest -q tests/test_deepseek_v4_nvfp4_kv.py -k sparse_attention
```

Expected: import failure for `mtplx.kernels.deepseek_v4_nvfp4_mla`.

- [x] **Step 3: Implement the fixed Metal kernel**

```python
def nvfp4_sparse_mla(
    queries: mx.array,
    window_records: mx.array,
    window_start: int,
    query_positions: mx.array,
    compressed_records: mx.array | None,
    compressed_indices: mx.array | None,
    compressed_lengths: mx.array | None,
    sinks: mx.array,
    scale: float,
) -> mx.array: ...
```

The decode kernel uses one 32-thread SIMD group per `(query, head)`. The prefill
kernel maps SparkInfer's native-NVFP4 engine to Metal's M16xN32xK16 NAX tile:
16 query heads per threadgroup, 32 candidates per tile, four 128-wide QK
splits, eight PV groups covering 512 values, BF16 tensor operands, and FP32
QK/softmax/P.V state. Both consume the 128-row causal window plus selected
compressed rows as one online-softmax union and write BF16 output. Target and
DSpark receive prebound phase callables at construction; enabled execution has
no fallback or invariant checks. A minimal installation-time dispatch compiles
the one fixed BF16 pipeline before serving, following the pinned vLLM engine's
compile/warm-before-request lifecycle.

- [x] **Step 4: Verify GREEN and the bounded-allocation contract**

Run the sparse-attention check and the target/DSpark files.  Inspect the callable's
interface to confirm it cannot receive a whole-context mask or score tensor.

- [x] **Step 5: Commit**

```bash
git add mtplx/kernels/deepseek_v4_nvfp4_mla.py mtplx/models/deepseek_v4.py \
  mtplx/models/deepseek_v4_dspark.py tests/test_deepseek_v4_nvfp4_kv.py
git commit -m "perf: consume Mia NVFP4 cache in sparse Metal MLA"
```

### Task 3A: Close the inference-engine review blockers

**Files:**
- Modify: `mtplx/deepseek_v4_exl3.py`
- Modify: `mtplx/deepseek_v4_paged_indexer.py`
- Modify: `mtplx/kernels/deepseek_v4_compressor.py`
- Modify: `mtplx/models/deepseek_v4.py`
- Modify: `mtplx/deepseek_v4_mia_engine.py`
- Modify only directly relevant DeepSeek V4 tests.

**Security flag:** artifact validation/load identity must remain exact.

**Does NOT cover:** unrelated model paths, generic optimization experiments,
new runtime fallbacks, long benchmarks before the port is complete, or tests for
hypothetical behavior outside the installed Mia route and reusable cache contract.

- [x] **Step 1: Preserve the pinned source contracts before changing kernels**

Record the exact official/vLLM/Mia/SparkInfer arithmetic, tie ordering, route
block capacity, workspace ownership, RoPE-cache ownership, and artifact
validation/load boundary used by each fix.  Where the available sources
disagree, resolve the disagreement from the packaged Mia lane before choosing
an implementation; do not create a third arithmetic definition.

- [x] **Step 2: Fix the Trellis launch and workspace geometry**

Bound descriptor capacity and MMA grid Z by the construction-proven maximum
packed route blocks, not routed task count.  Port SparkInfer's frozen caller-owned
intermediate/route workspace semantics for the installed M1..M6 and bounded
prefill routes without a host readback, hot eligibility check, or fallback.

- [x] **Step 3: Fix the indexer data plane**

Preserve lowest-logical-index selection for pivot ties in both prefill folds and
decode slices.  Replace the original K40 scorer with source-derived Q32xK256
ownership and bounded 128 MiB/512 MiB FP32 score slabs that feed the fixed
top-512 carry; never allocate query-by-full-context scores.  Use
construction-owned RoPE tables rather than per-head transcendental calls.

- [x] **Step 4: Fix compressor, cache, and lifecycle correctness**

Make stock432 and Mia132 compressor records follow the chosen packaged-source
BF16/RMSNorm/RoPE boundary exactly.  Include the fixed compressor rollback
journal in reusable cache snapshots.  Release target/draft arenas on every
prewarm exit, and bind shard validation to the bytes actually installed so cold
construction does not validate one object and load another.

- [x] **Step 5: Port the remaining packaged-Mia fused projection paths**

Construction-bind the exact TP1 stacked `wq_a+wkv` and `wkv+wgate` owners, one
shared target/draft RoPE graph, the fused learned-Q/KV RMSNorm and one-cast
Q-normalize/RoPE/stock432 prologue, and the packaged B12X inverse-RoPE/MXFP8
WO-A/WO-B chain.  Preserve target versus ephemeral K5 cache ownership, the
384K target page admission plus fixed draft lookahead, Mia's `ceil(log2())`
activation scale encoding, the BF16 WO-A bottleneck, and original checkpoint
parameter names.  Installed execution has no nullable callable, metadata
revalidation, generic QMM substitution, or silent fallback.

- [x] **Step 6: Focused red/green and integration gates**

For each encountered defect, first run the smallest deterministic reproducer,
then the fix, then the relevant DeepSeek V4 suite.  Static gates do not count as
Metal parity.  Do not run the requested long matrix until all review blockers
and subsequent whole-branch review findings are closed.

### Task 4: Exact-model execution, receipts, and PR publication

**Files:**
- Modify: `scripts/deepseek_v4_dspark_k5_bench.py`
- Create: `bench/deepseek-v4-mia/mia-k216-k64-nvfp4-b5638db-python-100.json`
- Create: `bench/deepseek-v4-mia/mia-k216-k64-nvfp4-b5638db-1024x1024.json`
- Create: `bench/deepseek-v4-mia/mia-k216-k64-nvfp4-b5638db-16384x1024-cold.json`
- Create: `bench/deepseek-v4-mia/mia-k216-k64-nvfp4-b5638db-65536x1024-cold.json`

**Security flag:** `none`

**Does NOT cover:** unrelated models, concurrency, sampling, additional context
lengths, or optimization experiments not nominated by these executions.

- [x] **Step 1: Update receipt identity and run focused non-GPU verification**

Require `kv_cache_format=nvfp4_stock432`, K216 target, K64 draft, pinned model and
source revisions. Refuse a dirty source tree before MLX/model load, commit the
complete implementation, merge and verify the live upstream PR base, and run
execution gates from a clean worktree at that exact SHA. Run lint plus only the
DeepSeek NVFP4/DSpark/DFlash2 suites.

- [x] **Step 2: Run one guarded real epoch and committed-token parity gate**

Acquire `/tmp/mtplx-gpu-exclusive.lock` through the existing service-restoration
guard.  Stop at the first record, output, rollback, or token mismatch.

- [x] **Step 3: Run the guarded Python service prompt**

Serve the exact Mia/Sero target with the packaged K64 draft through DFlash2,
generate roughly 100 tokens, then restore and verify the prior service.

- [x] **Step 4: Run the requested cold matrix**

Run exact `1024/1024`, `16384/1024`, and `65536/1024`.  Each receipt records
prefill tok/s, decode tok/s, generated count, acceptance, MLX peak/active memory,
process peak RSS, output digest, source commit, artifact revisions, and
`stock432` identity.

- [x] **Step 5: Correct and publish upstream PR #312**

Remove superseded affine/wrong-model claims, include only exact-model NVFP4
evidence, push the implementation and receipts to its existing feature head,
inspect the complete diff against upstream `main`, update the PR body, and make
it ready only after every required gate succeeds. The PR head is
`davidtai/MTPLX-1:feat/deepseek-v4-dspark-k5`, so push through the `mtplx1`
remote rather than `origin`.

- [x] **Step 6: Final verification**

Run `git diff --check`, focused ruff, the directly relevant suite, inspect the
three receipts, confirm the branch/remote SHA, and verify the original Qwen
service is healthy after the last guarded run.
