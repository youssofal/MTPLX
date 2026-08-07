# `eschamoe` — Escha-W2 2/3-bit MoE weight decoder

Native MLX decoder for the weight format used by
[`EschaLabs/Qwen3.6-35B-A3B-Escha-W2`](https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2)
(the vendor calls the format **`eschamoe`**). Module: `mtplx/eschamoe.py`. Test:
`tests/test_eschamoe_decode.py`.

## 1. Overview

Escha-W2 is a quantized derivative of Qwen3.6-35B-A3B (Apache-2.0 base), 12.3 GB on disk. It uses
the same architecture as the existing A3B kernel — `qwen3_5_moe`, hidden 2048, 40 layers, 256
experts / 8 active, `moe_intermediate 512`, shared-expert 512, hybrid 30 GDN + 10 full-attention
layers, one declared MTP layer (`mtp_num_hidden_layers=1`) — so routing, GDN, and attention
scaffolding are reusable as-is. Two things differ from a standard checkpoint:

- **Experts** are quantized with an AQLM-family codec the vendor calls `eschamoe`: **2-bit**
  `gate_up_proj`, **3-bit** `down_proj` ("mixed 2/3-bit"). This is not the affine (`weight_int4`
  / `weight_scale` / `weight_zero`) format the existing kernel loads, and the vendor only ships a
  CUDA runtime for it (`escham_reconstruct` in a closed `_C.so`). This module exists so the model
  can be served natively on Metal/MLX without that CUDA runtime.
- **Non-expert** layers (attention, router, embeddings, `lm_head`) are int8 (`weight_int8` +
  `weight_scale`), a separate concern from this decoder.

MTP is declared in the model config but the W2 release ships no `mtp.*` tensors, so serving is
bare-autoregressive out of the box; a draft head would have to be exported/quantized separately.

**Status:** the expert weight decode is fully solved and bit-exact against the vendor kernel —
0 mismatches over 2,097,152 (K=2, `gate_up_proj`) + 1,048,576 (K=3, `down_proj`) weights — and the
full expert forward (decode → Hadamard rotation → `rin`/`rout`) matches the vendor's reference
chain to fp16 accumulation noise. The implementation in `mtplx/eschamoe.py` is pure MLX
(vectorized bit-gather), correctness-tuned, not yet a fused Metal kernel. Loader/serving
integration and a fused kernel are downstream work (§7).

## 2. On-disk format

Per FusedMoE layer, per projection `p ∈ {gate_up_proj, down_proj}`, stacked over `E=256` experts
(key prefix `model.language_model.…`):

| tensor | dtype / shape | meaning |
|---|---|---|
| `{p}.escha_code`   | int16 `[E, in_p/16, out_p/16, 16*K]` | packed codes; `K` = bits/weight (2 or 3) |
| `{p}.escha_rin`    | fp16 `[E, in_p]`  | per-input-channel scale (trained `s_in` folded in) |
| `{p}.escha_rout`   | fp16 `[E, out_p]` | per-output-channel scale (trained `s_out` folded in) |
| `{p}.escha_s_in`   | fp32 `[E, in_f]`  | all-ones when folded (ignore) |
| `{p}.escha_s_out`  | fp32 `[E, out_f]` | all-ones when folded (ignore) |
| `{p}.escha_config` | int32 `[9]` | `[tile, K, V, cb_id, E, in_f, out_f, in_p, out_p]` |

Observed `escha_config` values:

| idx | field | `gate_up_proj` | `down_proj` |
|---|---|---|---|
| 0 | tile | 16 | 16 |
| 1 | K (bits/weight) | 2 | 3 |
| 2 | V | 2 | 2 |
| 3 | cb_id | 1 | 1 |
| 4 | E (experts) | 256 | 256 |
| 5 | in_f | 2048 | 512 |
| 6 | out_f | 1024 | 2048 |
| 7 | in_p | 2048 | 512 |
| 8 | out_p | 1024 | 2048 |

Dims are all /128-aligned, so `in_p == in_f` and `out_p == out_f` for both projections; `cb_id=1`
selects the codebook this decoder implements (§6, "cbA").

The weight matrix is tiled into **16×16 blocks**. Each block (256 weights) is coded by exactly
`16*K` int16 words — i.e. exactly `K` bits per weight, with no other framing or padding.

## 3. How the decode works

### 3a. The dequant primitive (the "codebook")

There is no codebook lookup table on disk — codebook `cb_id=1` ("cbA") is a magic-multiply
bit-twiddle baked into the CUDA kernel. For a 16-bit window `w`:

```
r  = (w * 3417055213) mod 2^32                     # magic multiply
lo = ((r        & 0xffff) & 0x8fff) ^ 0x3b60       # lop3 lut 0x6a = (a & b) ^ c
hi = (((r >> 16)& 0xffff) & 0x8fff) ^ 0x3b60
value = f16(lo) + f16(hi)                           # reinterpret each half as fp16, add in fp16
```

This produces exactly **10,746 distinct fp16 values** over all 2¹⁶ possible windows, and the real
decoded weights use exactly that value set — the first confirmation that this primitive is
correct. It is context-dependent: each weight's value depends on a full 16-bit window gathered
from several code bits, not on a small index directly, i.e. a trellis-style code rather than a
flat lookup table.

### 3b. Per-tile warp assembly

The CUDA kernel decodes one 16×16 tile per warp (K=2 → 64-byte / 512-bit tile; K=3 → 96-byte /
768-bit tile). Each of the 32 lanes emits 8 weights (32 × 8 = 256 per tile). A lane assembles its
16-bit windows by **straddling two `u32` reads** of the tile — the windows are not contiguous
slices, which is what makes the packing non-obvious. `u32(tb, o)` is a little-endian `u32` read at
byte offset `o` of the tile buffer `tb`.

**K=2**, per lane `L` (0..31), per weight `m` (0..7):

```
r87 = (2*L) & 60; r89 = (r87 - 4) & 60
rd8 = u32(tb, r87); rd9 = u32(tb, r89)
r93 = ((rd8>>16) | ((rd9 & 0xffff)<<16)) & 0xffffffff if (L%2==0) else rd8
for m in range(8):
    out[L,m] = DEC[(r93 >> (2*m)) & 0xffff]
```

**K=3**, per lane `L` (0..31), per weight `m` (0..7):

```
r83=L*24; r85=(r83+755)>>5; r86=r83+791; r87=r86&2016
r88=r87-r83; r89=r88-760
r91=23 if L==0 else (r85-24); r92=r91<<2
r95=(r86>>3)&252
rd8=u32(tb, r95-96); rd9=u32(tb, r92)
rd11=(rd9<<32)|rd8
for m in range(8):
    out[L,m]=DEC[(rd11 >> (r89+3*m)) & 0xffff]
```

(Source: `reverse_packing5.py` for K=2, `reverse_packing6.py` for K=3, in `/Users/davidtai/escha-extract/`.)

### 3c. The (lane, m) → (row, col) permutation

The 256 decoded values `out[L, m]` come out in lane order, not tile-row/col order. A fixed
permutation maps `(lane, m)` slot `s = L*8 + m` to a tile position `(dr, dc)`:

```
W_tile.flat[dr*16 + dc] = tile_values[perm[dr*16 + dc]]
```

This permutation is identical across every tile and every expert (it reflects the warp's
`shfl.down` + shared-memory transpose, which doesn't depend on the data). It was recovered
empirically, not derived from the PTX (§6), and is stored as `perm_K2.npy` / `perm_K3.npy`
(256-int arrays) in `/Users/davidtai/escha-extract/`.

In the shipped `mtplx/eschamoe.py` module this permutation is **not applied as a separate step at
decode time** — it is baked into the per-position bit-gather table (`mtplx/eschamoe_gather.npz`)
at table-build time, so each of the 256 output positions already carries the correct source bits
in tile-raster order. See §4 for how that table is used.

### 3d. The full expert forward

`escham_reconstruct` (the vendor kernel) only produces the bare weight `W`. The full expert
forward wraps it in a QuaRot-style Hadamard rotation — pure elementwise ops plus a 128-block
Walsh–Hadamard transform (`T128`), all reproducible off-GPU:

```
xh = T128(x * rin)            # rin per-input-channel; T128 over 128-blocks of the input dim
y  = T128(xh @ W) * rout      # rout per-output-channel; T128 over 128-blocks of the output dim
```

Experts use contiguous SwiGLU (`silu(gate) * up`, gate/up = first/second half of the `gate_up_proj`
output), no bias. The shared expert and `shared_expert_gate` are handled by the surrounding MoE
block, not by the packed experts. `escha_s_in` / `escha_s_out` are all-ones (the trained scale is
folded into `rin` / `rout`), so they can be ignored.

## 4. Using the module (`mtplx.eschamoe`)

```python
def decode_expert_weights(code: mx.array, K: int) -> mx.array
```

Decodes packed `eschamoe` codes to bare fp16 weights.

- **`code`**: int16, shape `[..., nI, nJ, 16*K]` — the raw `escha_code` tensor (or a slice of it),
  where `nI = in_p/16`, `nJ = out_p/16`.
- **`K`**: `2` or `3`. Raises `ValueError` if `code.shape[-1] != 16*K`, or if `K` is not `2` or `3`.
- **Returns**: fp16, shape `[..., nI*16, nJ*16]` — the dequantized weight matrix `W`.

Internally: it loads (and caches) the `(word_of, bit_of)` gather tables for the given `K` from
`mtplx/eschamoe_gather.npz` (`word_of`: which of the `16*K` int16 words each of the 16 window bits
comes from; `bit_of`: which bit within that word) — these already encode the §3b assembly formulas
and the §3c permutation for all 256 output positions of a tile. It gathers those bits with
vectorized indexing, packs them into a 16-bit window per output position, looks the window up in
the lazily-built 65,536-entry `DEC` table (§3a), and reshapes/transposes the result from
`[nI, nJ, 16, 16]` into the full `[nI*16, nJ*16]` matrix (a block-tile placement, not a data-
dependent permutation — the tricky permutation is already resolved inside the gather table).

```python
def t128(x: mx.array, pre=None, post=None) -> mx.array
```

`y = post * T128(x * pre)` — a normalized 128-block Walsh–Hadamard transform over the last
dimension.

- **`x`**: shape `[..., IC]` with `IC % 128 == 0`. Cast to float32 internally.
- **`pre`**, **`post`**: optional broadcastable arrays multiplied in before / after the transform
  (typically `rin` / `rout`).
- **Returns**: float32, shape `[..., IC]`.

The 128×128 normalized Hadamard matrix is built once (`_hadamard128()`) and cached in the module
global `_HAD128`; the `DEC` table and gather tables are similarly cached on first use (`_DEC`,
`_GATHER`). Repeated calls reuse these without rebuilding them.

Usage — the full expert forward from §3d:

```python
import mlx.core as mx
from mtplx.eschamoe import decode_expert_weights, t128

W  = decode_expert_weights(code, K)   # code: int16 [nI, nJ, 16*K] -> W: fp16 [nI*16, nJ*16]
xh = t128(x, pre=rin)                 # x: [..., in_p]
y  = t128(xh @ W, post=rout)          # y: [..., out_p], fp32
```

This is pure MLX — `mx.right_shift` / `mx.bitwise_and` bit-gather plus a table lookup and a
matmul. No custom Metal kernel is required for correctness; the current implementation is
correctness-tuned, and a fused Metal decode kernel is a later performance step (§7).

## 5. Validation

Reference-decoder results (against the vendor's real `escham_reconstruct` output, layer-0 experts
0–3):

```
gate_up_proj (K=2):  decode vs golden  0 / 2,097,152  mismatches   → BIT-EXACT
down_proj    (K=3):  decode vs golden  0 / 1,048,576  mismatches   → BIT-EXACT
full chain (decode→T128→rin/rout):  max_abs_err 9e-5, rel 6e-4  → fp16 noise
```

`mtplx`'s own regression, `tests/test_eschamoe_decode.py`, checks the shipped MLX implementation
against a fixture (`tests/fixtures/eschamoe_mini.npz`) holding real Escha-W2 codes and the weights
the vendor `escham_reconstruct` kernel produced for them (2×2 tiles of layer-0 expert 0, both
projections):

- `test_decode_bit_exact[gate_up_proj]`, `test_decode_bit_exact[down_proj]` — `decode_expert_weights`
  output must be `np.array_equal` to the vendor golden, exactly (0 tolerance).
- `test_dec_table_value_count` — the built `DEC` table must have exactly 10,746 distinct fp16
  values, matching the vendor kernel.
- `test_t128_involution_scale` — `T128` is orthonormal, so applying it twice with no scales returns
  the input (loose tolerance: MLX GPU fp32 matmul is lower precision than numpy, and the model runs
  fp16 in practice, so this is well within budget).

Run it with:

```
python -m pytest tests/test_eschamoe_decode.py -v
```

All 4 tests pass.

## 6. How it was reverse-engineered

Two independent sources, cross-validated against each other:

1. **Goldens from the vendor's CUDA kernel.** The vendor runtime only ships CUDA (`_C.so`, arches
   sm_80/86/89/90/100/120). `escham_reconstruct` was run once, on a DigitalOcean RTX 6000 Ada
   droplet (`gpu-6000adax1-48gb`, tor1, torn down immediately after), to dump real `code → W`
   pairs for layer-0 experts 0–3 of both projections, plus a full-chain golden.
2. **PTX, offline, no GPU.** `cuobjdump --dump-ptx` on `_C.so` (run in a tiny amd64 container with
   just the `cuda-cuobjdump`/`cuda-nvdisasm` debs) produced the disassembled
   `escham_reconstruct_kernel<cb, K>` bodies (`recon_1_2.ptx`, `recon_1_3.ptx`), giving the exact
   bit arithmetic used in §3a/§3b.

The dequant primitive (§3a) came from the PTX and was confirmed by the 10,746-distinct-value
match against the goldens. The tile packing (§3b) was also derived from the PTX, then validated
bit-exact against the goldens: for each of the 256 tile output positions, the bit-offset where
**all four experts simultaneously** decode to their four golden values was found (a
collision-proof 4-way key across experts), which resolved the `(lane, m) → (row, col)` permutation
(§3c). The resulting decoder was then checked to reproduce every weight of every tile of all four
experts with 0 mismatches, before being trusted as general.

## 7. Limitations / next steps

- Only codebook **`cb_id=1`** ("cbA") has been reversed — the codebook this model actually uses.
  The vendor kernel also defines `cb_id` 0 and 2, a separate `mul1` codebook, and a general AQLM
  path (`escha_aqlm_*` with an explicit codebook tensor); none of those are needed for Escha-W2 and
  none have been reversed.
- The goldens cover layer-0 experts 0–3 only. The decode was validated bit-exact across all
  tiles of those four experts, and the layout is uniform (the same fixed packing/permutation
  applies to every tile and expert), so it is expected to generalize — but wider golden coverage
  would need a re-run of the GPU dump.
- The dequant (§3a–3c) is bit-exact. The `T128`/matmul chain (§3d) is float arithmetic and matches
  the vendor reference only to fp16-accumulation noise — this is expected, not a bug.
- Remaining work to actually serve the model: wiring `eschamoe` (and int8 non-expert / embedding
  loading) into the `mtplx` A3B MoE path and model registry so the HF repo loads and serves;
  porting the decode to a fused Metal kernel for performance (current implementation is pure MLX,
  correctness-tuned); and tok/s benchmarks once that's in place.
