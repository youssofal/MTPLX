# q6 paged KV cache quantization — design and review note

Status: proposed, awaiting maintainer review.

Adds an intermediate 6-bit mode to the plain paged-KV quantizer that already
ships `q8` and `q4`. Additive: no default changes, and `q8`/`q4` numerics are
untouched.

## Motivation

Large-context Qwen 3.8 27B on a 32 GB Apple Silicon Mac has no comfortable KV
setting near 262k. Unquantized fp16 KV is the quality reference but the most
memory hungry; `q8` roughly halves it but stays tight at the top of the
context; `q4` fits easily but quantizes on a much coarser grid. `q6` gives a
point between the two, so the memory/precision tradeoff is not a binary
choice between "still tight" and "most aggressive mode we have".

This is a storage mode only. It makes no claim about output quality relative
to `q8` — see [Limitations](#limitations).

## Quantizer

Identical in structure to the existing symmetric quantizer, only the clip
bound changes:

| mode | qmax | storage per code |
|------|------|------------------|
| `q8` | 127  | 1 signed byte |
| `q6` | 31   | 6 bits, packed |
| `q4` | 7    | 4 bits, packed |

Per row (one token, one KV head): `scale = max(|x|) / qmax`, floored at 1e-6
by the same minimum-scale guard the existing modes use, stored as one fp16
value. Values are divided by the scale, rounded, clipped to ±qmax, then
biased into an unsigned code — `unsigned = q + 32` for q6, giving codes 1..63
inside a 6-bit field.

## Packing: four 6-bit codes per three bytes

Unlike `q4` (two codes per byte) and `q8` (one code per byte), q6 codes cross
byte boundaries, so it packs in groups of four. Little-endian layout:

```
byte0 = u0        | (u1 & 0x03) << 6
byte1 = (u1 >> 2) | (u2 & 0x0F) << 4
byte2 = (u2 >> 4) | (u3 & 0x3F) << 2
```

Every partial field is masked before shifting, so no intermediate exceeds 8
bits and the whole path stays in `uint8`. Unpacking is the mirror image, then
`signed = unsigned - 32` and `signed.astype(float32) * scale.astype(float32)`,
matching the float behavior of the existing modes.

Consequently `packed_dim(D, 6) = D * 3 / 4`, and q6 requires `D % 4 == 0`.
Qwen 3.8's 256-wide K/V heads satisfy this naturally (256 → 192 bytes).
`packed_dim` raises on head dims that do not divide by 4 rather than
silently truncating a group.

Both directions are vectorized MLX array ops — a reshape to groups of four
plus masks and shifts. There is no Python loop over KV elements.

## Storage (calculated)

Per token per KV head, K+V, at head_dim 256. Arithmetic on the layout, not
measured process RAM:

| mode | K | V | scales | total |
|------|---|---|--------|-------|
| fp16 | 512 | 512 | — | 1024 B |
| `q8` | 256 | 256 | 4 | 516 B |
| `q6` | 192 | 192 | 4 | 388 B |
| `q4` | 128 | 128 | 4 | 260 B |

For a completely filled 262,144-token Qwen 3.8 full-attention cache — 16
full-attention layers, 4 KV heads:

| mode | cache |
|------|-------|
| fp16 | ~16.00 GiB |
| `q8` | ~8.0625 GiB |
| `q6` | ~6.0625 GiB |
| `q4` | ~4.0625 GiB |

These are storage calculations and are asserted as such in the unit tests.
They are not a measurement of total process memory, which also carries
weights, activations and allocator overhead.

## Where it plugs in

The mode reaches the runtime the same way `q8`/`q4` do, so the change is
mostly extending allowlists rather than adding a path:

- `runtime_options.normalize_paged_kv_quantization` is the single
  normalization funnel; every reader (config, CLI, server, cache) goes
  through it. Adding `q6` there is what makes the env pair, `mtplx config
  set` and `--kv-quant` all accept it at once.
- `PagedKVQuantConfig.bits` now maps mode → width explicitly. It previously
  read `4 if mode == "q4" else 8`, which would have quantized q6 onto the q8
  grid while allocating q6-sized pages. The `qmax` lookup in
  `quantize_symmetric` had the same shape (`127 if bits == 8 else 7`) and is
  now an explicit table that raises on unknown widths.
- `cache_state` allocation was already width-generic through
  `packed_dim(head_dim, bits)`; q6 shares the `uint8` packed-storage branch
  with q4, and that branch is now commented as covering every sub-byte width
  rather than reading as "q8 or else q4".
- Attention: when plain KV quant is on, the cache dequantizes through
  `_paged_range` and runs MLX SDPA. No native/Metal paged-attention kernel
  consumes the packed bytes on this path, so no kernel needed a 6-bit case.
  (TurboQuant's kernel-backed path is a separate feature and is untouched.)

Aliases are deliberately narrow: `6`, `6bit`, `int6`, `uint6`. `q6_k` and
`q6_0` are rejected — this packing is MTPLX's own and is not llama.cpp's
Q6_K or any other external format.

## Limitations

- **q6 is lossy.** It stores a finer grid than q4 and a coarser one than q8.
  No claim is made that it is quality-neutral or "close to q8"; that would
  need a quality benchmark that has not been run.
- **The packing crosses byte boundaries**, which q4 and q8 do not. Pack and
  unpack each cost several mask/shift ops plus a stack and reshape per group
  of four, against a single shift/mask pair for q4. Dequantization runs on
  every attention call over the whole active window, so this is the most
  likely place for q6 to cost decode throughput relative to q8. **No timing
  has been measured for this change** — see below.
- **Not benchmarked.** No quant/dequant timing, decode throughput, memory
  measurement or model smoke test was run for this commit. Everything above
  is either arithmetic on the layout or behavior asserted by unit tests.
- **A custom Metal q6 path is explicitly out of scope** for this change. If
  measurement shows the dequant cost matters, that is a separate follow-up.

## Tests

`tests/test_kv_quant_q6.py` covers mode parsing and aliases, `packed_dim`
including the divisibility rejection, exact integer pack/unpack round trips
(boundary codes, all 64 codes, random codes, element ordering, and the
documented byte layout), float quantize/dequantize shape/dtype/finiteness and
error bounded by half a grid step, the q8 < q6 < q4 reconstruction-error
ordering on a fixed seed, and the storage/compression arithmetic above.

`tests/test_cache_state.py` adds q6 cases mirroring the existing q8/q4 ones:
config wiring, active-state round trip with packed shapes, packed size
ordering against q8/q4, growth and trim, and an attention comparison against
unquantized SDPA within tolerance.

`tests/test_env_flag_parsing.py` and `tests/test_public_cli.py` cover the env
pair, `--kv-quant` and `mtplx config set`.
`apps/MTPLXApp/Tests` covers q6 surviving the app's command-builder
normalization.

## Model files

No model weights or files under `~/.mtplx/models/` are read, modified or
requantized by this change. q6 is a runtime KV-cache setting only.
