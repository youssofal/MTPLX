# DFlash2 full-attention cache ownership

Status: approved for implementation on 2026-08-20

## Problem

`QwenGdnTargetOps.install_speculative_hooks()` replaces the target attention
class's `__call__` method once for the process. Recurrent layers already route
only when their cache is a DFlash-owned `RecurrentRollbackCache`. Full-attention
layers instead route every non-quantized cache with a cached prefix of at least
1,024 tokens and a query length at most 16. When MTPLX MTP and DFlash2 share one
loaded Qwen3.8 target, that condition also captures MTPLX's cache.

The behavior is visible on the real 1,024/1,024 workload: MTP-only and
DFlash2-only runs succeed under the GPU lock, but MTP after a DFlash2 warmup
enters the DFlash full-attention hook and terminates with a Metal GPU address
fault. The 32-token smoke stays below the hook's prefix threshold and therefore
does not expose the conflict.

## Decision

Fix ownership in `dflash-mlx`, not with MTPLX phase-global callable swapping.

Add DFlash-owned subclasses of MLX-LM's full-attention caches. The Qwen GDN
target backend constructs those subclasses for its unquantized full and
rotating caches. Its installed full-attention hook immediately delegates to
the captured stock callable unless the runtime cache has a DFlash-owned type.
The DFlash-owned route retains its existing arithmetic, threshold, tiling, GQA
selection, tree handling, and cache operations unchanged.

This is a runtime route based on genuine cache ownership. Both engines and the
same target model remain loaded simultaneously. MTPLX caches execute the stock
target callable directly; DFlash caches execute the existing DFlash callable.
There is no try/fallback path, per-token metadata validation, environment read,
or class-callable swap between arms.

## Interfaces and invariants

- `DFlashTargetKVCache` inherits `mlx_lm.models.cache.KVCache` without changing
  storage or arithmetic.
- `DFlashTargetRotatingKVCache` inherits
  `mlx_lm.models.cache.RotatingKVCache` without changing storage or arithmetic.
- `QwenGdnTargetOps.make_cache()` constructs these types only for DFlash-owned
  unquantized full-attention cache entries.
- Quantized cache behavior remains unchanged and continues through the stock
  callable because the DFlash GQA route already excludes it.
- The full-attention hook requires a DFlash-owned cache before tree or GQA
  routing. All other caches call the pre-installation callable.
- The recurrent path remains owned by `RecurrentRollbackCache` and is not
  modified.

## Testing and rollout

1. Observe a unit test fail on upstream `dflash-mlx` because a stock cache with
   a long prefix enters the DFlash hook.
2. Add the owned cache types and make-cache routing; prove stock cache calls
   remain stock and DFlash cache construction is ownership-typed.
3. Run the Qwen target/tree suite in the dependency repository.
4. Commit the dependency fix on an immutable fork branch and pin MTPLX to that
   exact commit; regenerate only the dependency lock entry.
5. Re-run the existing 32/32 smoke and the 1,024/1,024 width-1/8 transition
   reproducer under `/tmp/mtplx-gpu-exclusive.lock` with both engines loaded.
6. Only after the transition is stable, resume the stock width 1-8 sweep. No
   custom kernel or DFlash arithmetic optimization is part of this change.

## Rejected alternatives

- Swapping class callables between benchmark arms: avoids the immediate
  conflict but introduces mutable process-global phase state and an in-flight
  dispatch hazard.
- Clearing MLX allocator caches: does not restore unchanged control arithmetic
  and cannot prevent the hook from intercepting MTPLX caches.
- Separate engine processes: violates the requirement that both engines share
  one simultaneously loaded target during the comparison.

## Failure modes

- An unowned DFlash cache would silently use stock attention. Construction and
  make-cache tests therefore require owned types for every unquantized DFlash
  full-attention entry.
- An ownership check placed after tree/GQA inspection could still touch an
  MTPLX cache. The check must be the hook's first cache-dependent route.
- Subclass construction could drift from MLX-LM constructor signatures. Tests
  instantiate normal and rotating variants using the installed MLX-LM version,
  and the MTPLX dependency remains pinned to an immutable commit.

