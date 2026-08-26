# DeepSeek V4 Mia `stock432` NVFP4 K/V Design

**Date:** 2026-08-21

**Status:** Arithmetic contract retained; storage ownership superseded by
`docs/specs/2026-08-21-system-paged-cache-design.md`

## Goal

Replace the temporary MLX affine-int4 cache in the exact Mia/Sero K216 target
and K64 DSpark draft with Mia's native `stock432` NVFP4 cache contract, then
consume that representation directly from the bounded sparse-attention path.

This corrects both storage and arithmetic. The current affine lane cannot
represent Mia's native record. Mia's packaged writer applies RoPE to the final
64 learned-normalized latent values, crosses one BF16 boundary for the complete
post-RoPE 512-wide row, and NVFP4-quantizes all 512 values for V. A separate
BF16 field duplicates the rotated 64-wide tail used with the first 448 latent
values to reconstruct K.

## Pinned Source Contract

The authoritative layout is the 432-byte record exercised by
`MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark@d4ba142bc1d971eb73a911e207e3e963bbb3c455`
in `image-patch/selftest_padded_stride.py` and read by the patched SparkInfer sparse-MLA
prefill/decode path:

```text
bytes   0..255   512 E2M1 values, low nibble first
bytes 256..287    32 E4M3 scales, one per group of 16 latent values
bytes 288..303    16 zero padding bytes
bytes 304..431    64 BF16 GPT-J-interleaved rotated-RoPE values
```

Dequantization is:

```text
post_rope_latent[d] = e2m1(record[d / 2], d % 2) * e4m3(record[256 + d / 16])
value               = post_rope_latent[0:512]
key                 = concat(post_rope_latent[0:448], bf16(record[304:432]))
```

The record has no affine zero point and is not MLX `mxfp4`.  It is a single
`uint8` owner, not three parallel packed/scale/bias arrays.

## Architecture

### Record owner and codec

`mtplx/deepseek_v4_nvfp4_kv.py` owns the fixed constants, a Metal pack kernel,
an oracle decoder, and `MiaNVFP4Rows`.  Construction fixes width 512, NoPE 448,
RoPE 64, group size 16, and record size 432.  Invalid geometry fails before the
owner is installed.

The owner accepts normalized latent rows and their already-computed rotated
RoPE tails. It substitutes the rotated tail into the full row, crosses the BF16
boundary once, NVFP4-packs all 512 post-RoPE values, duplicates the rotated tail
as BF16 bytes, and performs append, replacement, truncation, eviction, and state
restoration on whole records.

### Target arithmetic

The target attention route computes `kv_norm(wkv(x))` and the 64-wide rotated
tail separately. The record writer installs that tail into the complete
post-RoPE row, BF16-roundtrips it, NVFP4-quantizes all 512 values, and duplicates
the rotated tail in the BF16 key field.

The attention compressor builds and BF16-roundtrips the complete post-RoPE
pooled latent before NVFP4 quantization. Attention uses that full rotated row as
V and, as the pinned DeepSeek model graph requires, inverse-rotates the 64-wide
output tail at the query position before o-LoRA. The indexer compressor and rollback
journals remain their existing auxiliary state; they are not reclassified as
NVFP4 attended K/V.

### DSpark arithmetic

Each of the three DSpark stages owns a distinct `MiaNVFP4Rows` ring. Context
prefill and authoritative-main commits insert normalized latent plus its rotated
RoPE tail; the writer packs the complete post-RoPE row and duplicates the tail
in the BF16 key field. Proposal-local rows remain ephemeral.
Attention reconstructs K and the post-RoPE V
and retains the source model's inverse output RoPE before o-LoRA.

### Direct sparse Metal consumption

`mtplx/kernels/deepseek_v4_nvfp4_mla.py` reads `stock432` records directly.
For every query/head it:

1. visits only the causal 128-row sliding interval;
2. visits the indexer's selected compressed rows, capped by the model's fixed
   `index_topk` contract;
3. decodes E2M1 and E4M3 in registers;
4. forms QK with the stored BF16 RoPE tail and accumulates PV from the NVFP4
   post-RoPE latent row;
5. includes the learned per-head sink in the online-softmax denominator; and
6. writes one BF16 512-wide output without materializing a score matrix or a
   dense dequantized cache.

The target route is installed once at construction.  It has no enabled-path
fallback, environment read, eligibility check, or engagement counter.  Prefill
continues through 1,024-query chunks, but each chunk's attention allocation is
bounded by selected rows rather than total context length.

## Migration

- Replace `DeepseekV4AffineInt4Cache` with `DeepseekV4NVFP4Cache` for the exact
  Mia artifact route.
- Replace every DSpark `AffineInt4Rows` ring with `MiaNVFP4Rows`.
- Update DFlash2 construction checks to require `stock432` owners.
- Remove the obsolete affine-int4 module and its affine-only tests once no live
  reference remains.
- Existing receipts describing affine-int4 K/V are superseded and cannot be
  published as final Mia evidence.

## Direct Verification Only

1. A fixed record contract check proves offsets, E2M1 nibble order, E4M3
   group-16 scaling, full-row post-RoPE V, and the duplicated BF16 K-RoPE bytes.
2. Target and all three draft cache constructors prove `stock432` ownership and
   exact rollback/ring replacement behavior.
3. One bounded Metal sparse-attention comparison checks the direct consumer
   against an oracle built from the same records.
4. The exact real model must pass one DSpark epoch, target-only/DSpark committed
   token parity, the Python service prompt, and the requested cold 1K/16K/64K
   matrix with peak memory.

No unrelated compatibility tests, alternate record layouts, or fallback routes
are in scope.  The record codec is now installed through the reusable paged
ownership system rather than an appendable model-local array.

## Failure-Mode Check

- **Critical: byte-compatible records but wrong values.**  The record gate checks
  decoded K and V independently and includes scale-boundary values before model
  construction.
- **Critical: long prefill still scales with full context.**  The direct consumer
  accepts selected indices and bounded window ranges; a whole-context score or
  dequantized tensor is not part of its interface.
- **Critical: rollback corrupts record alignment.**  replacement and truncation
  operate only in 432-byte row units and the existing rejection-repair gate must
  pass before benchmarking.
- **Minor: `stock432` is larger than the temporary affine record.**  This is an
  accepted cost of matching Mia's full post-RoPE V plus duplicated-BF16-tail K arithmetic and fused sparse
  consumer; it remains smaller than Mia's 584-byte padded record.

## Non-Goals

- MLX `mxfp4`, affine int4, Mia's 368-byte FP8-RoPE mode, or the alternate
  UE8M0/360-byte community layout.
- NVFP4 checkpoint-weight conversion.
- Replacing DFlash2, DSpark acceptance, event handling, or service architecture.
- Quantizing indexer scoring rows or compressor rollback journals as K/V.
- Any benchmark or PR claim from the superseded affine cache lane.
