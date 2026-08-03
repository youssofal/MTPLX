"""Fused input-RMSNorm + Q/K/V/gate projection for the Laguna S-2.1 decode step.

Ported from the mlx.fast **Laguna XS2.1** challenge kernel
``lagunaFusedQKVProjection`` (``lagunaFusedQKVProjectionSource`` in
Sources/MLXFastModel/LagunaRuntimeModel.swift), re-expressed as a Python
``mx.fast.metal_kernel`` and *adapted* to Laguna S-2.1, which is a different
model from the challenge's XS2.1:

    hidden axis        XS2.1 2048    ->  S2.1 3072
    q_proj out         heads*128     ->  heads*128 (heads 48 full / 72 sliding)
    k/v_proj out       kv_heads*128  ->  8*128 == 1024
    g_proj out         heads         ->  heads (per-head gate: one logit per head)
    attn weight quant  bf16 / MXFP8  ->  affine INT (5- or 8-bit) group_size 64

The single dispatch it replaces is the head of ``Attention.__call__`` on the
already-``input_layernorm``'d hidden state (see ``models/laguna.py``): the
decoder layer computes ``self.self_attn(self.input_layernorm(x), ...)`` and the
attention block's first act is ``q_proj(x), k_proj(x), v_proj(x)`` plus the
per-head ``g_proj(x)``.  This kernel folds ``input_layernorm`` (an ``RMSNorm``)
into the four affine QMVs so the normalised row is produced once, in
threadgroup memory, and consumed by every projection without a device
round-trip or a separate norm dispatch.

Returns ``(queries, keys, values, gate_logits)`` — the *raw* projection
outputs.  The downstream ``q_norm``/``k_norm``, RoPE, SDPA and the softplus of
the gate logits are unchanged and stay outside this kernel.

## Numerics

* The RMSNorm half is the repo's established single-pass 3072 topology
  (``768`` threads, ``N_READS == 4`` -> ``24`` simdgroups, the same shape
  ``laguna_residual_router`` widened the XS2.1 512-thread donor to).  Each tile
  recomputes it from ``residual`` into ``normalized_row`` — cheap relative to
  the projection reads and it avoids a cross-threadgroup barrier — so the
  normalised value equals ``mx.fast.rms_norm(hidden, w, eps)`` (verified
  bit-exact on CPU: ``rms_norm`` and the ``float(raw*inv)`` cast agree).

* Each projection row is a length-3072 dot: lane ``l`` walks the strided
  columns ``{l, l+32, ...}``, dequantises each weight as
  ``float(code)*scale + bias`` in FP32 (contiguous LSB-first affine unpack —
  bit-exact vs ``mx.dequantize`` for bits 5 and 8 at gs64, verified on CPU),
  accumulates in FP32, ``simd_shuffle_down`` reduces, and rounds to the output
  dtype.  That is the *same value class* as ``mx.quantized_matmul`` but not
  bit-for-bit: the reduction is reassociated, so the bar is ``allclose`` (and
  matching an FP32/FP64 gold), not bitwise equality.  NB: on CPU
  ``mx.quantized_matmul`` itself accumulates crudely (~1.0 abs error vs an FP64
  gold), so the *reference* below is validated against the FP64 gold, and the
  kernel-vs-``quantized_matmul`` agreement is confirmed on the GPU (FP32
  accumulate) by the flocked check script.

Callers gate on :func:`is_qkvg_fused_eligible` first; the public helper falls
back to the stock ``rms_norm`` + four ``quantized_matmul`` projections on any
shape or quant layout it does not cover, so it can be switched on without
owning a correctness branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mlx.core as mx


_HIDDEN = 3072
_HEAD_DIM = 128
_KV_HEADS = 8
_GROUP_SIZE = 64
_N_READS = 4
_SIMD_SIZE = 32
# Single-pass 3072 norm: 768 threads == 24 simdgroups, the repo's exact-fit
# reduction topology (identical to laguna_residual_router's widened donor).
_NORM_THREADS = _HIDDEN // _N_READS  # 768
_NORM_SIMDS = _NORM_THREADS // _SIMD_SIZE  # 24
_SUPPORTED_BITS = (5, 8)


@dataclass(frozen=True)
class QKVGSpec:
    """Shape + quant geometry the fused kernel needs.

    ``bits`` is shared by q/k/v/g: in the pinned oQ4e checkpoint every layer's
    q/k/v/g projections carry the same width (only o_proj — not in this kernel
    — is ever promoted, on layer 33), so a single ``bits`` covers the four
    banks.  The eligibility gate re-checks this against the actual weights.
    """

    n_heads: int
    bits: int
    hidden_size: int = _HIDDEN
    head_dim: int = _HEAD_DIM
    n_kv_heads: int = _KV_HEADS
    group_size: int = _GROUP_SIZE
    # Output rows each simdgroup owns per tile.  Larger => fewer tiles => the
    # per-tile RMSNorm recompute is amortised over more projection rows.  Pure
    # performance knob; correctness is independent of it (tails are guarded).
    rows_per_thread: int = 8

    @property
    def query_rows(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_rows(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def gate_rows(self) -> int:
        return self.n_heads

    @property
    def total_rows(self) -> int:
        return self.query_rows + 2 * self.kv_rows + self.gate_rows

    @property
    def rows_per_tile(self) -> int:
        return _NORM_SIMDS * self.rows_per_thread

    @property
    def words_per_row(self) -> int:
        return self.hidden_size * self.bits // 32

    @property
    def n_groups(self) -> int:
        return self.hidden_size // self.group_size


def _flat_rows(shape: tuple[int, ...]) -> int:
    rows = 1
    for dim in shape[:-1]:
        rows *= int(dim)
    return rows


def _quant_shapes_ok(
    codes: mx.array,
    scales: mx.array,
    biases: mx.array,
    out_rows: int,
    spec: QKVGSpec,
    weight_dtype,
) -> bool:
    if codes.dtype != mx.uint32:
        return False
    if scales.dtype != weight_dtype or biases.dtype != weight_dtype:
        return False
    if tuple(codes.shape) != (out_rows, spec.words_per_row):
        return False
    if tuple(scales.shape) != (out_rows, spec.n_groups):
        return False
    if tuple(biases.shape) != (out_rows, spec.n_groups):
        return False
    return True


def is_qkvg_fused_eligible(
    hidden: mx.array,
    norm_weight: mx.array,
    q_codes: mx.array,
    q_scales: mx.array,
    q_biases: mx.array,
    k_codes: mx.array,
    k_scales: mx.array,
    k_biases: mx.array,
    v_codes: mx.array,
    v_scales: mx.array,
    v_biases: mx.array,
    g_codes: mx.array,
    g_scales: mx.array,
    g_biases: mx.array,
    spec: QKVGSpec,
) -> bool:
    """Whether the fused kernel covers this exact decode shape + quant layout.

    Deliberately narrow, matching the stock pieces it fuses.  Decode only
    (a single row): the port mirrors the challenge kernel's ``[1, 1, hidden]``
    precondition.  Any miss => the public helper runs the stock chain.
    """

    if not mx.metal.is_available():
        return False
    if hidden.dtype not in (mx.bfloat16, mx.float16):
        return False
    weight_dtype = hidden.dtype
    if norm_weight.dtype != weight_dtype:
        return False
    if spec.hidden_size != _HIDDEN or int(hidden.shape[-1]) != spec.hidden_size:
        return False
    if norm_weight.ndim != 1 or int(norm_weight.shape[0]) != spec.hidden_size:
        return False
    # Decode: exactly one active row (B == T == 1).
    if _flat_rows(hidden.shape) != 1:
        return False
    if spec.bits not in _SUPPORTED_BITS:
        return False
    if spec.group_size != _GROUP_SIZE:
        return False
    if spec.hidden_size % spec.group_size != 0:
        return False
    # Exact-fit norm reduction: 24 whole simdgroups cover 3072 at N_READS == 4.
    if spec.hidden_size % (_SIMD_SIZE * _N_READS) != 0:
        return False
    if spec.hidden_size // _N_READS != _NORM_THREADS:
        return False
    if spec.head_dim != _HEAD_DIM or spec.n_kv_heads != _KV_HEADS:
        return False
    if spec.n_heads <= 0 or spec.rows_per_thread <= 0:
        return False
    if not _quant_shapes_ok(
        q_codes, q_scales, q_biases, spec.query_rows, spec, weight_dtype
    ):
        return False
    if not _quant_shapes_ok(
        k_codes, k_scales, k_biases, spec.kv_rows, spec, weight_dtype
    ):
        return False
    if not _quant_shapes_ok(
        v_codes, v_scales, v_biases, spec.kv_rows, spec, weight_dtype
    ):
        return False
    if not _quant_shapes_ok(
        g_codes, g_scales, g_biases, spec.gate_rows, spec, weight_dtype
    ):
        return False
    return True


@lru_cache(maxsize=None)
def _qkvg_kernel(n_heads: int, bits: int, rows_per_thread: int):
    spec = QKVGSpec(n_heads=n_heads, bits=bits, rows_per_thread=rows_per_thread)
    header = f"""
        using namespace metal;
        constant constexpr uint HIDDEN = {spec.hidden_size};
        constant constexpr uint N_READS = {_N_READS};
        constant constexpr uint SIMD_SIZE = {_SIMD_SIZE};
        constant constexpr uint NORM_SIMDS = {_NORM_SIMDS};
        constant constexpr uint BITS = {spec.bits};
        constant constexpr uint GROUP_SIZE = {spec.group_size};
        constant constexpr uint N_GROUPS = {spec.n_groups};
        constant constexpr uint WORDS_PER_ROW = {spec.words_per_row};
        constant constexpr uint CODE_MASK = {(1 << spec.bits) - 1}u;
        constant constexpr uint COLS_PER_LANE = {spec.hidden_size // _SIMD_SIZE};
        constant constexpr uint QUERY_ROWS = {spec.query_rows};
        constant constexpr uint KV_ROWS = {spec.kv_rows};
        constant constexpr uint GATE_ROWS = {spec.gate_rows};
        constant constexpr uint TOTAL_ROWS = {spec.total_rows};
        constant constexpr uint ROWS_PER_THREAD = {spec.rows_per_thread};
        constant constexpr uint ROWS_PER_TILE = {spec.rows_per_tile};
    """

    # normalized_row stages the RMSNorm output in threadgroup memory so every
    # projection row consumes it without re-reading from device.  Each tile
    # recomputes the (cheap) norm; there is no cross-threadgroup dependency.
    source = """
        uint tile = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;

        threadgroup float local_sums[SIMD_SIZE];
        threadgroup float local_inv_mean[1];
        threadgroup T normalized_row[HIDDEN];

        // --- input RMSNorm (single-pass, N_READS == 4, 24 simdgroups) ---
        uint norm_base = lid * N_READS;
        T raw[N_READS];
        float acc = 0.0f;
        for (uint i = 0; i < N_READS; ++i) {
            T v = residual[norm_base + i];
            raw[i] = v;
            float fv = float(v);
            acc += fv * fv;
        }
        acc = simd_sum(acc);
        if (simd_group == 0) {
            local_sums[simd_lane] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_lane == 0) {
            local_sums[simd_group] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0) {
            float a = simd_sum(local_sums[simd_lane]);
            if (simd_lane == 0) {
                local_inv_mean[0] =
                    metal::precise::rsqrt(a / float(HIDDEN) + eps);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float inv = local_inv_mean[0];
        for (uint i = 0; i < N_READS; ++i) {
            normalized_row[norm_base + i] =
                norm_weight[norm_base + i] *
                static_cast<T>(float(raw[i]) * inv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- projections: each simdgroup owns ROWS_PER_THREAD output rows;
        //     lane `l` walks strided columns {l, l+32, ...} of the 3072 row. ---
        uint block = tile * ROWS_PER_TILE + simd_group * ROWS_PER_THREAD;
        for (uint r = 0; r < ROWS_PER_THREAD; ++r) {
            uint g_out = block + r;
            if (g_out >= TOTAL_ROWS) {
                break;
            }

            const device uint32_t* codes;
            const device T* scales;
            const device T* biases;
            device T* out;
            uint local_row;
            if (g_out < QUERY_ROWS) {
                codes = q_codes; scales = q_scales; biases = q_biases;
                out = queries; local_row = g_out;
            } else if (g_out < QUERY_ROWS + KV_ROWS) {
                codes = k_codes; scales = k_scales; biases = k_biases;
                out = keys; local_row = g_out - QUERY_ROWS;
            } else if (g_out < QUERY_ROWS + 2 * KV_ROWS) {
                codes = v_codes; scales = v_scales; biases = v_biases;
                out = values; local_row = g_out - QUERY_ROWS - KV_ROWS;
            } else {
                codes = g_codes; scales = g_scales; biases = g_biases;
                out = gate_logits;
                local_row = g_out - QUERY_ROWS - 2 * KV_ROWS;
            }

            uint row_word_base = local_row * WORDS_PER_ROW;
            uint row_group_base = local_row * N_GROUPS;
            float dot = 0.0f;
            for (uint k = 0; k < COLS_PER_LANE; ++k) {
                uint c = simd_lane + k * SIMD_SIZE;
                // Contiguous LSB-first affine unpack (bits in {5, 8}).  A value
                // straddles two words only mid-row; the row is exactly
                // WORDS_PER_ROW words (HIDDEN*BITS a multiple of 32), so the
                // last value never straddles and word+1 stays in-row.
                uint bit_off = c * BITS;
                uint word = bit_off >> 5;
                uint shift = bit_off & 31u;
                uint lo = codes[row_word_base + word] >> shift;
                uint hi = (shift + BITS > 32u)
                    ? (codes[row_word_base + word + 1] << (32u - shift))
                    : 0u;
                uint code = (lo | hi) & CODE_MASK;
                uint grp = c / GROUP_SIZE;
                float scale = float(scales[row_group_base + grp]);
                float bias = float(biases[row_group_base + grp]);
                float w = float(code) * scale + bias;
                dot += w * float(normalized_row[c]);
            }
            for (ushort delta = 16; delta >= 1; delta >>= 1) {
                dot += metal::simd_shuffle_down(dot, delta);
            }
            if (simd_lane == 0) {
                out[local_row] = static_cast<T>(dot);
            }
        }
    """
    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_qkvg_fused_h{n_heads}_b{bits}_r{rows_per_thread}",
        input_names=[
            "residual",
            "norm_weight",
            "q_codes",
            "q_scales",
            "q_biases",
            "k_codes",
            "k_scales",
            "k_biases",
            "v_codes",
            "v_scales",
            "v_biases",
            "g_codes",
            "g_scales",
            "g_biases",
            "eps",
        ],
        output_names=["queries", "keys", "values", "gate_logits"],
        header=header,
        source=source,
    )


def _stock_qkvg(
    hidden,
    norm_weight,
    q_codes,
    q_scales,
    q_biases,
    k_codes,
    k_scales,
    k_biases,
    v_codes,
    v_scales,
    v_biases,
    g_codes,
    g_scales,
    g_biases,
    eps,
    spec: QKVGSpec,
):
    """Stock chain: ``rms_norm(hidden)*w`` then four affine ``quantized_matmul``."""

    normed = mx.fast.rms_norm(hidden, norm_weight, eps)

    def proj(codes, scales, biases):
        return mx.quantized_matmul(
            normed,
            codes,
            scales,
            biases,
            transpose=True,
            group_size=spec.group_size,
            bits=spec.bits,
        )

    queries = proj(q_codes, q_scales, q_biases)
    keys = proj(k_codes, k_scales, k_biases)
    values = proj(v_codes, v_scales, v_biases)
    gate_logits = proj(g_codes, g_scales, g_biases)
    return queries, keys, values, gate_logits


def fused_input_norm_qkvg(
    hidden: mx.array,
    norm_weight: mx.array,
    q_codes: mx.array,
    q_scales: mx.array,
    q_biases: mx.array,
    k_codes: mx.array,
    k_scales: mx.array,
    k_biases: mx.array,
    v_codes: mx.array,
    v_scales: mx.array,
    v_biases: mx.array,
    g_codes: mx.array,
    g_scales: mx.array,
    g_biases: mx.array,
    eps: float,
    spec: QKVGSpec,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused ``input_layernorm`` + Q/K/V/gate projection for one decode row.

    Returns ``(queries, keys, values, gate_logits)`` with leading dims matching
    ``hidden`` (``[*hidden.shape[:-1], out]``).  Falls back to the stock chain
    (``mx.fast.rms_norm`` then four ``mx.quantized_matmul``) on any unsupported
    shape or quant layout, so callers need not own a correctness branch.
    """

    leading = tuple(int(d) for d in hidden.shape[:-1])

    if not is_qkvg_fused_eligible(
        hidden,
        norm_weight,
        q_codes,
        q_scales,
        q_biases,
        k_codes,
        k_scales,
        k_biases,
        v_codes,
        v_scales,
        v_biases,
        g_codes,
        g_scales,
        g_biases,
        spec,
    ):
        queries, keys, values, gate_logits = _stock_qkvg(
            hidden,
            norm_weight,
            q_codes,
            q_scales,
            q_biases,
            k_codes,
            k_scales,
            k_biases,
            v_codes,
            v_scales,
            v_biases,
            g_codes,
            g_scales,
            g_biases,
            eps,
            spec,
        )
    else:
        kernel = _qkvg_kernel(spec.n_heads, spec.bits, spec.rows_per_thread)
        tiles = (spec.total_rows + spec.rows_per_tile - 1) // spec.rows_per_tile
        residual = hidden.reshape(1, spec.hidden_size)
        queries, keys, values, gate_logits = kernel(
            inputs=[
                residual,
                norm_weight,
                q_codes,
                q_scales,
                q_biases,
                k_codes,
                k_scales,
                k_biases,
                v_codes,
                v_scales,
                v_biases,
                g_codes,
                g_scales,
                g_biases,
                float(eps),
            ],
            template=[("T", hidden.dtype)],
            grid=(_NORM_THREADS * tiles, 1, 1),
            threadgroup=(_NORM_THREADS, 1, 1),
            output_shapes=[
                (1, spec.query_rows),
                (1, spec.kv_rows),
                (1, spec.kv_rows),
                (1, spec.gate_rows),
            ],
            output_dtypes=[hidden.dtype] * 4,
        )

    queries = queries.reshape(*leading, spec.query_rows)
    keys = keys.reshape(*leading, spec.kv_rows)
    values = values.reshape(*leading, spec.kv_rows)
    gate_logits = gate_logits.reshape(*leading, spec.gate_rows)

    # Fake-speedup guard: a wrong-shaped output silently does a fraction of the
    # work (or broadcasts) and FAKES a win.  Assert the exact contract.
    assert tuple(queries.shape) == (*leading, spec.query_rows), queries.shape
    assert tuple(keys.shape) == (*leading, spec.kv_rows), keys.shape
    assert tuple(values.shape) == (*leading, spec.kv_rows), values.shape
    assert tuple(gate_logits.shape) == (*leading, spec.gate_rows), gate_logits.shape
    return queries, keys, values, gate_logits


def fused_input_norm_qkvg_reference(
    hidden: mx.array,
    norm_weight: mx.array,
    q_codes: mx.array,
    q_scales: mx.array,
    q_biases: mx.array,
    k_codes: mx.array,
    k_scales: mx.array,
    k_biases: mx.array,
    v_codes: mx.array,
    v_scales: mx.array,
    v_biases: mx.array,
    g_codes: mx.array,
    g_scales: mx.array,
    g_biases: mx.array,
    eps: float,
    spec: QKVGSpec,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Pure-mx reference implementing the exact math the metal kernel computes.

    No ``metal_kernel`` — only primitive ``mx`` ops — so it runs on CPU and
    pins down the algorithm:

      * RMSNorm in FP32 with the ``bfloat(raw*inv)`` cast, then ``* weight``:
        verified bit-exact vs ``mx.fast.rms_norm`` on CPU.
      * Each projection dequantises to FP32 (``scale``/``bias`` upcast, no
        intermediate bf16 rounding of the weight — matching the kernel's
        ``float(code)*scale + bias``), accumulates the dot in FP32, and rounds
        to the input dtype — matching the kernel's ``static_cast<T>(dot)``.

    This is the value the kernel targets; it matches an FP32/FP64 gold to
    ~1e-6 (and is *more* accurate than CPU ``mx.quantized_matmul``, which
    accumulates crudely).
    """

    dtype = hidden.dtype
    gs = spec.group_size
    bits = spec.bits

    x = hidden.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)
    normed = norm_weight * (x * inv).astype(dtype)  # bf16, == mx.fast.rms_norm

    def proj(codes, scales, biases):
        deq = mx.dequantize(
            codes,
            scales.astype(mx.float32),
            biases.astype(mx.float32),
            group_size=gs,
            bits=bits,
        )
        return (normed.astype(mx.float32) @ deq.T).astype(dtype)

    queries = proj(q_codes, q_scales, q_biases)
    keys = proj(k_codes, k_scales, k_biases)
    values = proj(v_codes, v_scales, v_biases)
    gate_logits = proj(g_codes, g_scales, g_biases)
    return queries, keys, values, gate_logits
