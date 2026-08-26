# DFlash2 Cache Ownership Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-optimized:subagent-driven-development (recommended) or superpowers-optimized:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep MTPLX MTP and DFlash2 loaded around one Qwen3.8 target while preventing DFlash's full-attention hook from intercepting MTPLX-owned caches.

**Architecture:** `dflash-mlx` gives its unquantized full-attention caches explicit owned subclasses and makes the existing class hook route only those types. MTPLX pins the resulting immutable fork commit, then the same-process transition and width sweep run under the canonical GPU lock.

**Tech Stack:** Python 3.12, MLX 0.32.1, mlx-lm 0.31.3, dflash-mlx 0.1.10 source, pytest, Ruff, uv, macOS Metal, canonical guarded runner.

**Assumptions:** Assumes the observed fault is caused by the full-attention class hook accepting MTPLX caches, as proven by the 1,024-token threshold and isolated-engine controls. This will not address an independent Metal fault that reproduces after the ownership route is fixed. Assumes DFlash cache subclasses preserve MLX-LM storage behavior; constructor and target-tree tests fail if that changes.

---

## File structure

- Modify `dflash_mlx/engine/target_qwen_gdn.py` in the dependency fork: define owned full-attention cache types and route the existing hook by ownership.
- Create `tests/test_target_qwen_cache_ownership.py` in the dependency fork: prove stock-cache isolation and owned-cache construction.
- Modify `pyproject.toml`, `uv.lock`, and `tests/test_dflash2_dependency.py` in MTPLX: pin and prove the immutable dependency fix.
- Update `docs/specs/2026-08-20-qwen38-dflash2-mtp-benchmark-design.md`: record the real failure, ownership correction, and tested dependency commit.

### Task 1: Add cache ownership in dflash-mlx

**Files:**
- Modify: `dflash_mlx/engine/target_qwen_gdn.py`
- Create: `tests/test_target_qwen_cache_ownership.py`

**Security flag:** none

**Does NOT cover:** This task does not change DFlash attention arithmetic, thresholds, GQA kernels, recurrent caches, quantized caches, model weights, or block-width policy.

- [ ] **Step 1: Write failing stock-cache isolation and construction tests**

Add tests that install the Qwen full-attention hook on a fake attention class,
pass a long-prefix stock cache, and require the original callable result. Add
constructor tests requiring `QwenGdnTargetOps.make_cache()` to return the owned
normal and rotating cache subclasses for DFlash full-attention layers.

- [ ] **Step 2: Verify RED**

```bash
python -m pytest -q tests/test_target_qwen_cache_ownership.py
```

Expected: failure because upstream uses base MLX-LM cache types and the hook
does not distinguish stock cache ownership.

- [ ] **Step 3: Implement the ownership route**

Define `DFlashTargetKVCache(KVCache)` and
`DFlashTargetRotatingKVCache(RotatingKVCache)`. Construct those types in
`QwenGdnTargetOps.make_cache()` and make `_install_full_attention_gqa_hook()`
delegate immediately for every cache not owned by DFlash.

- [ ] **Step 4: Verify GREEN and relevant dependency coverage**

```bash
python -m pytest -q \
  tests/test_target_qwen_cache_ownership.py \
  tests/test_target_qwen_tree.py
python -m ruff check \
  dflash_mlx/engine/target_qwen_gdn.py \
  tests/test_target_qwen_cache_ownership.py
```

Expected: all focused tests and Ruff pass.

- [ ] **Step 5: Commit and push the immutable dependency fix**

```bash
git add dflash_mlx/engine/target_qwen_gdn.py \
  tests/test_target_qwen_cache_ownership.py
git commit -m "Scope Qwen target hooks to DFlash caches"
git push -u origin fix/qwen-cache-ownership
```

### Task 2: Pin MTPLX to the ownership fix

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `tests/test_dflash2_dependency.py`

**Security flag:** none

**Does NOT cover:** This task does not update any unrelated direct dependency or accept a floating branch reference.

- [ ] **Step 1: Change the dependency test to require the exact fork commit**

Resolve `ownership_commit=$(git -C ../dflash-mlx rev-parse HEAD)` and require
the MTPLX competitor extra and installed distribution source to reference that
full commit.

- [ ] **Step 2: Verify RED**

```bash
.venv/bin/python -m pytest -q tests/test_dflash2_dependency.py
```

Expected: the old `bstnxbt/dflash-mlx@60803233...` pin fails the exact-source
assertion.

- [ ] **Step 3: Update only dflash-mlx and regenerate the lock**

Replace the dependency URL with the immutable `davidtai/dflash-mlx` ownership
commit, then run:

```bash
uv lock --upgrade-package dflash-mlx
uv sync --extra dev --extra competitors
```

- [ ] **Step 4: Verify dependency API and ownership behavior**

```bash
.venv/bin/python -m pytest -q \
  tests/test_dflash2_dependency.py \
  tests/test_dflash2_runtime.py \
  tests/test_dflash2_depth_sweep.py
```

Expected: all focused tests pass with the installed fork commit.

- [ ] **Step 5: Commit the atomic dependency integration**

```bash
git add pyproject.toml uv.lock tests/test_dflash2_dependency.py
git commit -m "Pin DFlash cache ownership fix"
```

### Task 3: Prove same-process engine transitions under the GPU lock

**Files:**
- Modify only owned MTPLX runner/tests if the fixed dependency exposes an integration regression.

**Security flag:** security

**Does NOT cover:** This task does not split engines into processes, run without the canonical lock, weaken generated-token/fallback checks, or implement a custom kernel.

- [ ] **Step 1: Run the 32/32 same-process smoke**

Use `bench/laguna/run_guarded.py` with the exact Qwen3.8 model/launcher and
widths 1 and 8. Require full token counts, requested/effective width equality,
no fallback, exact service restoration, and a free lock after exit.

- [ ] **Step 2: Run the 1,024/1,024 transition reproducer**

Run one same-process width-1/8 repetition through the guarded wrapper. Expected:
no Metal address fault when the sequence transitions DFlash-to-MTP above the
1,024-prefix hook threshold.

- [ ] **Step 3: Run the complete widths 1-8 campaign**

Run three rotated repetitions at 1,024/1,024 under the same guard and one loaded
target. Record every control and candidate row, then stop after the selected
stock width or measured tie band.

- [ ] **Step 4: Verify service and lock postflight**

Require the exact `mtplx-qwen38-27b-optimized-speed` service, a successful
completion, and nonblocking acquisition of `/tmp/mtplx-gpu-exclusive.lock`.

### Task 4: Publish the evidence to the tracking PR

**Files:**
- Modify: `docs/specs/2026-08-20-qwen38-dflash2-mtp-benchmark-design.md`
- Modify: `docs/plans/2026-08-20-dflash2-cache-ownership.md`
- Create: immutable benchmark JSON receipt under `benchmarks/results/`

**Security flag:** none

**Does NOT cover:** This task does not merge the PR, promote DFlash2 into the persistent launcher, or begin Phase B kernel optimization.

- [ ] **Step 1: Record source commits and causal evidence**

Record the upstream and fork dependency commits, MTPLX commit, real fault,
ownership proof, guarded commands, raw brackets, winner/tie band, and service
postflight.

- [ ] **Step 2: Run final repository verification**

```bash
.venv/bin/python -m pytest -q \
  tests/test_dflash2_dependency.py \
  tests/test_dflash2_contract.py \
  tests/test_dflash2_runtime.py \
  tests/test_dflash2_depth_sweep.py \
  tests/test_dflash2_cli.py \
  tests/test_qwen38_dflash2_depth_guarded.py
.venv/bin/ruff check \
  mtplx/benchmarks/dflash2_contract.py \
  mtplx/benchmarks/dflash2_runtime.py \
  mtplx/benchmarks/runners/dflash2_depth_sweep.py \
  mtplx/benchmarks/runners/competitor_baselines.py \
  mtplx/cli.py scripts/qwen38_dflash2_depth_guarded.py \
  tests/test_dflash2_dependency.py tests/test_dflash2_contract.py \
  tests/test_dflash2_runtime.py tests/test_dflash2_depth_sweep.py \
  tests/test_dflash2_cli.py tests/test_qwen38_dflash2_depth_guarded.py
git diff --check upstream/main...HEAD
```

- [ ] **Step 3: Commit and push MTPLX evidence**

```bash
git add docs/specs/2026-08-20-qwen38-dflash2-mtp-benchmark-design.md \
  docs/plans/2026-08-20-dflash2-cache-ownership.md \
  benchmarks/results/qwen38-dflash2-depth1-8-1024x1024-*.json
git commit -m "Record Qwen3.8 DFlash2 cache ownership sweep"
git push mtplx1 perf/qwen38-dflash2
```

Expected: tracking PR #304 contains the immutable dependency fix and the
same-process guarded width result. Stop before Phase B optimization.

