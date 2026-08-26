# System Paged Cache Design

**Date:** 2026-08-21

**Status:** Approved for execution

## Goal

Install one reusable vLLM-style cache ownership layer across MTPLX and use the
Mia/Sero DeepSeek V4 route as its first complete long-context adopter.  The
shared layer owns fixed capacity, physical pages, logical block tables, slot
mapping, writes, rollback, and state transfer.  Model code owns only record
geometry, encoding, retention, and attention consumption.

The immediate success condition remains the exact Mia model: native 432-byte
NVFP4 K/V, chunked prefill, DSpark through DFlash2, and the requested 1K, 16K,
and 64K executions without cache-growth high-water allocation.

## Source-Derived Contract

The pinned Mia image is not stock vLLM.  It builds
`local-inference-lab/vllm@30038602b71395f481ef4a6edfe4fcf8551d9c15` and
`local-inference-lab/sparkinfer@272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f`,
then applies the image's vLLM, NVFP4 compatibility, compact-DSpark, and
SparkInfer patches.  The final runtime has these ownership properties:

1. KV memory is sized and allocated as a fixed page pool before serving.
2. The scheduler admits a request against the pool, then schedules bounded
   prefill chunks.  Mia caps a long prefill chunk at 1,024 tokens.
3. Each chunk receives slot mappings derived from persistent block tables.
4. Target and DSpark K/V are written directly into those slots.  A full-prompt
   target-feature tensor is not retained.
5. Native `stock432` records remain packed in the pool and are read without a
   whole-cache materialization.
6. Hybrid cache specs distinguish logical token coverage from stored rows:
   sliding-window rows, ratio-4 and ratio-128 compressed rows, indexer rows,
   compressor state, and the three draft windows do not all grow at the prompt
   rate.

These mechanisms, rather than CUDA topology or kernel names, are the parts to
port.

## Architecture

### Immutable construction plan

`PagedCacheSpec` describes one logical cache lane:

- logical block size and stored rows per block;
- row shape, dtype, byte stride, and alignment;
- maximum logical tokens and retained-token bound;
- record codec identifier; and
- rollback/state-transfer policy.

`PagedCachePlan` validates a collection of specs once and calculates the exact
physical allocation.  Runtime settings are read while constructing the plan.
The installed cache objects do not read environment variables, re-check model
metadata, or select a fallback during execution.

### Pool, block tables, and views

`PagedCachePool` owns fixed physical buffers.  `PagedCacheLease` owns logical
block tables for a request, and `PagedCacheView` exposes the slots belonging to
one model lane.  Writes receive precomputed logical positions or slot mappings
and update the allocated pages directly.  Truncate, clear, snapshot, and restore
change lease metadata and the bounded rollback journal; they do not concatenate
or geometrically replace capacity-sized tensors.

The first implementation supports the serving contracts already exercised by
MTPLX: batch one, contiguous logical growth, bounded speculative rollback, and
state handoff.  The types and allocation boundary are request-neutral so the
server can later lease the same runtime pool to multiple requests.  This work
does not invent unencountered eviction, preemption, or prefix-sharing behavior.

### Existing MTPLX paged cache

`VllmMetalPagedKVCache` becomes an adapter over the shared pool instead of a
second owner.  Its attention interfaces and current model integrations remain
intact.  Capacity, quantization layout, and attention route are bound at
installation.  The enabled fixed-capacity route cannot call its current
geometric grow path or silently downgrade to another layout.

This makes the new ownership layer part of the whole system rather than a
DeepSeek utility, while avoiding a risky blanket model cutover.

### DeepSeek V4 specs

The exact Mia adapter contributes specs for:

- the bounded 128-token target sliding windows;
- ratio-4 and ratio-128 `stock432` compressed K/V;
- the ratio-4 indexer cache;
- compressor frontier state; and
- the three bounded DSpark windows.

`MiaNVFP4Rows` becomes a codec/view facade over paged storage.  Its 432-byte
arithmetic remains unchanged.  Target and draft attention continue consuming
native records.  The plan derives stored-row capacity from the real compression
ratio rather than allocating every lane at the full prompt length.

### Chunked prefill and DFlash2

The reusable DFlash2 boundary gains a streaming context-consumer capability.
For DeepSeek, each target prefill chunk is projected, immediately converted into
the three draft layers' context K/V, and written to their page slots.  The chunk
then becomes reclaimable.  Models that genuinely require retained target
features continue using the existing feature-store route selected at adapter
construction.

No enabled route contains `eligible-or-stock`, `try-custom-then-fallback`, or
per-chunk proof instrumentation.

## System Integration Boundary

The infrastructure is system-wide; adoption is explicit and incremental:

1. the existing standard paged K/V cache delegates to it;
2. DeepSeek/Mia adopts every persistent K/V lane and streaming DSpark context;
3. other model families can supply specs/codecs without another allocator; and
4. existing non-paged model caches remain unchanged until their real workload
   is migrated and measured.

This is not a port of vLLM's CUDA scheduler, request policy, or kernel library.
MTPLX retains its generation, session, GraphBank, and DFlash2 architecture.

## Direct Gates

Only gates required by the implementation or an encountered failure are added:

1. existing paged-cache tests must pass through the shared owner;
2. the exact DeepSeek record, trim, replacement, and DFlash2 adapter checks must
   pass with fixed pages;
3. the DFlash2 prefill gate must prove the encountered full-prompt feature store
   is absent on the streaming DeepSeek route;
4. one guarded exact-model epoch must preserve committed tokens and DSpark
   acceptance; and
5. the requested guarded 1K, 16K, and 64K runs decide memory and performance.

No speculative concurrency, eviction, prefix-cache, or alternate-codec tests
are part of this port.

## Non-Goals

- Reimplementing vLLM's scheduler or CUDA kernels.
- Automatically converting every model to paged storage in one pull request.
- Adding a silent fallback from an installed paged lane.
- Generic cache instrumentation in measured paths.
- Optimizing unrelated models before the Mia port and requested benchmarks work.
