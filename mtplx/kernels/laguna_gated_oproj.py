"""Fused per-head softplus gate + affine o_proj GEMV for the Laguna S-2.1 decode.

Ported from the mlx.fast **Laguna XS2.1** challenge kernels
``lagunaGatedAffineOProjSource`` (the gate folded into an affine INT GEMV) and
``lagunaGateProductSoftplusSource`` (the exact softplus gate product) in
Sources/MLXFastModel/LagunaRuntimeModel.swift, synthesised into ONE
``mx.fast.metal_kernel`` and *adapted* to Laguna S-2.1:

    attention heads    XS2.1 48 full / 64 sliding  ->  S2.1 48 full / 72 sliding
    hidden (out)       2048                          ->  3072
    o_proj quant       group-32 INT8 (re-quant)      ->  affine gs64, 5- OR 8-bit
                                                          (the shipped oQ4e wire
                                                          format; layer 33's
                                                          o_proj is the promoted
                                                          8-bit row)

The challenge's fused affine variant serves its own group-32 INT8 re-quant
envelope; S-2.1's o_proj ships as affine **group_size 64** at **5 or 8 bit**
(``models/laguna_config.py`` ``_OQ4E_ATTENTION_BITS``, the 4th char of each
row).  So this port keeps the challenge's fusion shape — gate softplus in
threadgroup memory, gate the row, contract — but re-derives the affine unpack
for gs64 and both bit widths, exactly as ``laguna_qkvg_fused`` did for the q/k/v/g
projections.

## The two dispatches it replaces

The stock decode tail (``models/laguna.py`` ``Attention.__call__``, per-head
gating) is: ``g_proj`` already done, then softplus the gate logits
(``logaddexp(logits, 0)`` -> a BF16->FP32 cast, a LogAddExp, an FP32->BF16 cast),
broadcast-multiply it across each head's 128-wide slice of the attention output,
then ``o_proj`` (a ``quantized_matmul``).  This kernel folds the softplus AND the
broadcast product into the o_proj GEMV's own vector loads, so the gated
``heads*128``-wide row is never materialised and the layer spends ONE dispatch
instead of the gate chain plus the GEMV.

## Numerics

* **Softplus** is MLX's ``LogAddExp`` specialised to ``logaddexp(x, 0)`` —
  ``maxval + log1p(exp(minval - maxval))`` with the NaN/inf guards, the same form
  ``mtplx.kernels.laguna_decode`` and the shipped attn-gate kernel use — so the
  gate matches ``mx.logaddexp(logits.astype(f32), 0).astype(bf16)`` bit-for-bit.
  The gate is rounded to bf16 (``float(bfloat(gate))``) before the product, and
  the product ``bfloat(float(attn) * gate)`` rounds once, exactly where the stock
  ``output * gate`` bf16 multiply rounds.

* **The projection** dequantises each weight as ``float(code)*scale + bias`` in
  FP32 (contiguous LSB-first affine unpack — bit-exact vs ``mx.dequantize`` for
  bits 5 and 8 at gs64, verified on CPU), accumulates the dot in FP32,
  ``simd_shuffle_down`` reduces, and rounds to bf16.  That is the *same value
  class* as ``mx.quantized_matmul`` but NOT bit-for-bit: the reduction is
  reassociated, so the bar is ``allclose`` (and matching an FP32/FP64 gold), not
  bitwise equality.  NB: on CPU ``mx.quantized_matmul`` accumulates crudely
  (~1.0 abs error vs an FP64 gold), so the reference below is validated against
  the FP64 gold and the kernel-vs-``quantized_matmul`` agreement is confirmed on
  the GPU (FP32 accumulate) by the flocked check script.

Callers gate on :func:`is_gated_oproj_eligible` first; the public helper falls
back to the stock softplus-gate -> ``quantized_matmul`` chain on any shape or
quant layout it does not cover (5- and 8-bit gs64 are covered; a bits/gs the
kernel is not built for takes the fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx


_HIDDEN = 3072
_HEAD_DIM = 128
_GROUP_SIZE = 64
_SIMD_SIZE = 32
_N_SIMDS = 8  # simdgroups per threadgroup
_THREADS = _SIMD_SIZE * _N_SIMDS  # 256
_SUPPORTED_BITS = (5, 8)

# MLX's Metal LogAddExp specialised to logaddexp(x, 0) — the shipped softplus.
# Byte-identical to mtplx.kernels.laguna_decode._SOFTPLUS.
_SOFTPLUS = """
    inline float mtplx_softplus(float x) {
        if (metal::isnan(x)) {
            return metal::numeric_limits<float>::quiet_NaN();
        }
        constexpr float inf = metal::numeric_limits<float>::infinity();
        float maxval = metal::max(x, 0.0f);
        float minval = metal::min(x, 0.0f);
        if (minval == -inf || maxval == inf) {
            return maxval;
        }
        return maxval + log1p(metal::exp(minval - maxval));
    }
"""


@dataclass(frozen=True)
class GatedOProjSpec:
    """Shape + quant geometry for the fused gated-output-projection kernel."""

    n_heads: int
    bits: int
    hidden_size: int = _HIDDEN
    head_dim: int = _HEAD_DIM
    group_size: int = _GROUP_SIZE
    # Output rows each simdgroup owns per tile.  Pure perf knob; correctness is
    # independent of it (tails are guarded).
    rows_per_thread: int = 8

    @property
    def in_vec(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def out_dim(self) -> int:
        return self.hidden_size

    @property
    def words_per_row(self) -> int:
        return self.in_vec * self.bits // 32

    @property
    def n_groups(self) -> int:
        return self.in_vec // self.group_size

    @property
    def rows_per_tile(self) -> int:
        return _N_SIMDS * self.rows_per_thread


def _flat_rows(shape: tuple[int, ...]) -> int:
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    return rows


def is_gated_oproj_eligible(
    attention_output: mx.array,
    gate_logits: mx.array,
    o_codes: mx.array,
    o_scales: mx.array,
    o_biases: mx.array,
    spec: GatedOProjSpec,
) -> bool:
    """Whether the fused kernel covers this exact decode shape + quant layout."""

    if not mx.metal.is_available():
        return False
    try:
        if mx.default_device() != mx.gpu:
            return False
    except Exception:
        return False
    dtype = attention_output.dtype
    if dtype not in (mx.bfloat16, mx.float16):
        return False
    if gate_logits.dtype != dtype:
        return False
    if spec.bits not in _SUPPORTED_BITS:
        return False
    if spec.group_size != _GROUP_SIZE or spec.in_vec % spec.group_size != 0:
        return False
    if spec.in_vec % _SIMD_SIZE != 0:
        return False
    if spec.head_dim != _HEAD_DIM or spec.hidden_size != _HIDDEN:
        return False
    if spec.n_heads <= 0 or spec.rows_per_thread <= 0:
        return False
    # Decode only: one active row.
    if _flat_rows(attention_output.shape) != 1:
        return False
    if int(attention_output.shape[-1]) != spec.in_vec:
        return False
    if _flat_rows(gate_logits.shape) != 1 or int(gate_logits.shape[-1]) != spec.n_heads:
        return False
    if o_codes.dtype != mx.uint32:
        return False
    if o_scales.dtype != dtype or o_biases.dtype != dtype:
        return False
    if tuple(o_codes.shape) != (spec.out_dim, spec.words_per_row):
        return False
    if tuple(o_scales.shape) != (spec.out_dim, spec.n_groups):
        return False
    if tuple(o_biases.shape) != (spec.out_dim, spec.n_groups):
        return False
    return True


@lru_cache(maxsize=None)
def _gated_oproj_kernel(n_heads: int, bits: int, rows_per_thread: int):
    spec = GatedOProjSpec(n_heads=n_heads, bits=bits, rows_per_thread=rows_per_thread)
    header = _SOFTPLUS + f"""
        using namespace metal;
        constant constexpr uint HEAD_DIM = {spec.head_dim};
        constant constexpr uint HEAD_SHIFT = 7;   // head_dim == 128 == 1<<7
        constant constexpr uint IN_VEC = {spec.in_vec};
        constant constexpr uint OUT_DIM = {spec.out_dim};
        constant constexpr uint N_HEADS = {spec.n_heads};
        constant constexpr uint SIMD_SIZE = {_SIMD_SIZE};
        constant constexpr uint THREADS = {_THREADS};
        constant constexpr uint BITS = {spec.bits};
        constant constexpr uint GROUP_SIZE = {spec.group_size};
        constant constexpr uint N_GROUPS = {spec.n_groups};
        constant constexpr uint WORDS_PER_ROW = {spec.words_per_row};
        constant constexpr uint CODE_MASK = {(1 << spec.bits) - 1}u;
        constant constexpr uint COLS_PER_LANE = {spec.in_vec // _SIMD_SIZE};
        constant constexpr uint ROWS_PER_THREAD = {spec.rows_per_thread};
        constant constexpr uint ROWS_PER_TILE = {spec.rows_per_tile};
    """

    source = """
        uint tile = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;

        // Only the small per-head gate table lives in threadgroup memory; the
        // gated input is recomputed inline per column and reused across the
        // simdgroup's output rows, so occupancy is not throttled by staging the
        // whole IN_VEC row (the challenge's affine-oproj layout).
        threadgroup float gate_table[N_HEADS];
        for (uint h = lid; h < N_HEADS; h += THREADS) {
            float g = mtplx_softplus(float(gate_logits[h]));
            gate_table[h] = float(static_cast<T>(g));  // bf16 rounding point
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- o_proj GEMV: each simdgroup owns ROWS_PER_THREAD output rows;
        //     lane `l` walks strided columns {l, l+32, ...} of the IN_VEC row.
        //     Column loop is OUTER so the gated input (bf16) is computed once per
        //     column and shared by every output row this simdgroup owns. ---
        uint block = tile * ROWS_PER_TILE + simd_group * ROWS_PER_THREAD;
        uint rmax = (block < OUT_DIM)
            ? metal::min(ROWS_PER_THREAD, OUT_DIM - block)
            : 0u;

        float dot[ROWS_PER_THREAD];
        for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
            dot[r] = 0.0f;
        }

        for (uint kk = 0; kk < COLS_PER_LANE; ++kk) {
            uint c = simd_lane + kk * SIMD_SIZE;
            float gate = gate_table[c >> HEAD_SHIFT];
            // gated input, rounded to bf16 exactly where stock rounds the
            // `output * gate` product.
            float gv = float(static_cast<T>(float(attention_output[c]) * gate));

            // Contiguous LSB-first affine unpack (bits in {5, 8}); the offset
            // within a row is the same for every output row.  A value straddles
            // two words only mid-row (IN_VEC*BITS is a multiple of 32), so
            // word+1 stays in-row.
            uint bit_off = c * BITS;
            uint word = bit_off >> 5;
            uint shift = bit_off & 31u;
            uint grp = c / GROUP_SIZE;

            for (uint r = 0; r < rmax; ++r) {
                uint out_row = block + r;
                uint row_word_base = out_row * WORDS_PER_ROW;
                uint lo = o_codes[row_word_base + word] >> shift;
                uint hi = (shift + BITS > 32u)
                    ? (o_codes[row_word_base + word + 1] << (32u - shift))
                    : 0u;
                uint code = (lo | hi) & CODE_MASK;
                float scale = float(o_scales[out_row * N_GROUPS + grp]);
                float bias = float(o_biases[out_row * N_GROUPS + grp]);
                float w = float(code) * scale + bias;
                dot[r] += w * gv;
            }
        }

        for (uint r = 0; r < rmax; ++r) {
            float d = dot[r];
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                d += metal::simd_shuffle_down(d, delta);
            }
            if (simd_lane == 0) {
                projected[block + r] = static_cast<T>(d);
            }
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_gated_oproj_h{n_heads}_b{bits}_r{rows_per_thread}_v1",
        input_names=[
            "attention_output",
            "gate_logits",
            "o_codes",
            "o_scales",
            "o_biases",
        ],
        output_names=["projected"],
        header=header,
        source=source,
    )


def _stock_gated_oproj(
    attention_output: mx.array,
    gate_logits: mx.array,
    o_codes: mx.array,
    o_scales: mx.array,
    o_biases: mx.array,
    spec: GatedOProjSpec,
) -> mx.array:
    """Stock tail: per-head softplus gate x attention output, then affine o_proj.

    Reproduces ``models/laguna.py`` ``_stock_per_head_gate`` +
    ``o_proj``: ``gate = logaddexp(logits.f32, 0).astype(bf16)``, broadcast
    across each head's 128-wide slice, then ``quantized_matmul``.
    """

    heads, hd = spec.n_heads, spec.head_dim
    lead = tuple(int(d) for d in attention_output.shape[:-1])
    gate = mx.logaddexp(gate_logits.astype(mx.float32), mx.array(0.0)).astype(
        attention_output.dtype
    )
    gated = (
        attention_output.reshape(*lead, heads, hd) * gate[..., None]
    ).reshape(*lead, heads * hd)
    return mx.quantized_matmul(
        gated,
        o_codes,
        o_scales,
        o_biases,
        transpose=True,
        group_size=spec.group_size,
        bits=spec.bits,
    )


def gated_oproj_reference(
    attention_output: mx.array,
    gate_logits: mx.array,
    o_codes: mx.array,
    o_scales: mx.array,
    o_biases: mx.array,
    spec: GatedOProjSpec,
) -> mx.array:
    """Pure-mx reference implementing the exact math the metal kernel computes.

    Softplus gate (bf16-rounded), bf16 gate product, then a FP32-dequant /
    FP32-accumulate GEMV rounded to bf16.  Runs on CPU (no ``metal_kernel``) and
    is the value the kernel targets — matching an FP32/FP64 gold, and *more*
    accurate than CPU ``mx.quantized_matmul``.
    """

    heads, hd = spec.n_heads, spec.head_dim
    lead = tuple(int(d) for d in attention_output.shape[:-1])
    dtype = attention_output.dtype

    gate = mx.logaddexp(gate_logits.astype(mx.float32), mx.array(0.0)).astype(dtype)
    gated = (
        attention_output.reshape(*lead, heads, hd) * gate[..., None]
    ).reshape(*lead, heads * hd)  # bf16, == stock gated row

    deq = mx.dequantize(
        o_codes,
        o_scales.astype(mx.float32),
        o_biases.astype(mx.float32),
        group_size=spec.group_size,
        bits=spec.bits,
    )  # [out_dim, in_vec], fp32
    return (gated.astype(mx.float32) @ deq.T).astype(dtype)


def fused_gated_oproj(
    attention_output: mx.array,
    gate_logits: mx.array,
    o_codes: mx.array,
    o_scales: mx.array,
    o_biases: mx.array,
    spec: GatedOProjSpec,
) -> mx.array:
    """Fused per-head softplus gate + affine o_proj for one decode row.

    Returns the projected hidden state ``[*attention_output.shape[:-1],
    hidden_size]``.  Falls back to the stock softplus-gate -> ``quantized_matmul``
    chain on any shape or quant layout the kernel does not cover.
    """

    lead = tuple(int(d) for d in attention_output.shape[:-1])

    if not is_gated_oproj_eligible(
        attention_output, gate_logits, o_codes, o_scales, o_biases, spec
    ):
        projected = _stock_gated_oproj(
            attention_output, gate_logits, o_codes, o_scales, o_biases, spec
        )
    else:
        kernel = _gated_oproj_kernel(spec.n_heads, spec.bits, spec.rows_per_thread)
        tiles = (spec.out_dim + spec.rows_per_tile - 1) // spec.rows_per_tile
        attn = attention_output.reshape(1, spec.in_vec)
        glogits = gate_logits.reshape(1, spec.n_heads)
        (projected,) = kernel(
            inputs=[attn, glogits, o_codes, o_scales, o_biases],
            template=[("T", attention_output.dtype)],
            grid=(_THREADS * tiles, 1, 1),
            threadgroup=(_THREADS, 1, 1),
            output_shapes=[(1, spec.out_dim)],
            output_dtypes=[attention_output.dtype],
        )

    projected = projected.reshape(*lead, spec.out_dim)

    # Fake-speedup guard: a wrong-shaped output silently does a fraction of the
    # work and FAKES a win.  Assert the exact contract.
    assert tuple(projected.shape) == (*lead, spec.out_dim), projected.shape
    return projected
