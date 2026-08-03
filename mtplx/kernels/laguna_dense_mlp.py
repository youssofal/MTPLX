"""Fused dense (layer-0) SwiGLU-QMV for the Laguna S-2.1 decode step.

Ported from the mlx.fast **Laguna XS2.1** challenge's layer-0 dense-MLP fusion
(``lagunaDenseGateUpSwiGLU`` + ``lagunaDenseDownResidual`` in
``Sources/MLXFastModel/LagunaRuntimeModel.swift``) and *adapted* to Laguna
S-2.1, which differs from the challenge in both geometry and quantization:

    axis (hidden)       XS2.1 2048    ->  S2.1 3072
    dense intermediate  XS2.1 8192    ->  S2.1 12288
    layer-0 quant       XS2.1 BF16    ->  S2.1 affine, MIXED per-projection
                                          gate_proj / up_proj : 5-bit, gs 64
                                          down_proj           : 6-bit, gs 64

Layer 0 is the one decoder layer whose MLP is a plain dense
``down(silu(gate(x)) * up(x))`` (``mlp_only_layers == [0]``) rather than an MoE
block.  It is 1 of 48 layers, so its whole-model runtime share is ~2%.

## Why this is NOT a copy of the challenge kernel

The challenge's layer-0 MLP is plain BF16 ``Linear`` (never quantized), so its
two kernels load ``vec<bfloat, 4>`` weight rows directly.  S-2.1 quantizes
layer 0 to *save decode bandwidth*, and to the odd bit widths 5 and 6 at group
size 64 (see the checkpoint's ``config.json`` per-path quantization overrides).
A faithful port must therefore dequantize in-kernel from the packed ``uint32``
weight, exactly as this repo's :mod:`laguna_moe_swiglu` does for the routed
experts -- but for 5/6-bit packing, not the clean 4-bit-per-nibble case.

MLX affine packing at these widths is bit-concatenation, little-endian within
each ``uint32`` word, values in row order, straddling word boundaries.  Because
``group_size * bits`` is a whole number of 32-bit words for both projections
(``64 * 5 == 320 == 10 words``; ``64 * 6 == 384 == 12 words``), every group
starts on a word boundary and only *interior* values straddle -- so the
unpacker below can walk group by group and never read past a row's words.  This
packing is verified bit-exact against ``mx.dequantize`` in the CPU check.

## Two dispatches, like the challenge

The 12288-wide intermediate does not fit one threadgroup's memory (48 KiB of
``float`` > the 32 KiB limit), so -- as the challenge does -- this is two
dispatches, each fusing several ops:

    Kernel 1  gate & up dequant-QMV over HIDDEN, then ``silu(gate) * up``
              -> ``activated[rows, intermediate]``  (3 ops -> 1 dispatch)
    Kernel 2  down dequant-QMV over intermediate   -> ``out[rows, hidden]``

Both are thread-per-output-neuron QMVs (grid-stride), the same layout
:mod:`laguna_moe_swiglu` uses, with per-group affine dequant folded into a
single FP32 accumulator, matching stock's ``x -> bf16`` rounding at each
projection boundary.

## Numerics honesty (read before trusting any local allclose)

This kernel accumulates each dot in FP32 from FP32-dequantized weights, which
reproduces an fp64 reference to ~1e-6.  MLX's ``mx.quantized_matmul`` -- what
stock ``QuantizedLinear`` calls -- is a *lossier* approximation of the same
math: on the **CPU** backend it diverges from the fp64 gold by ~1% per
projection (reduced-precision accumulation), so a CPU ``allclose`` between this
kernel's arithmetic and stock will NOT be tight, and that gap is stock being
inexact, not this kernel.  MLX's **Metal** ``quantized_matmul`` accumulates in
FP32, so on the flocked box the kernel-vs-stock gap should be far smaller --
but it is still a *different* accumulation, and the challenge's real bar is
exact-token teacher-forced match, which only the flocked correctness run can
decide.  Treat this port as: correct algorithm, likely ALU-bound at ~2% share
(see the repo's "Metal sub-4-bit is ALU-bound" / "IQ2_XXS kernel loses to
stock" findings), correctness-on-the-token-gate = flocked-only.

Callers use :func:`dense_mlp` (drop-in for ``mlp(x)`` on layer 0) or check
:func:`is_dense_mlp_eligible` and call :func:`dense_swiglu_qmv`.
:func:`dense_mlp_reference` is the pure-mx numeric reference the kernel
implements.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


# Wired for the exact Laguna S-2.1 layer-0 dense geometry / quant.
_HIDDEN = 3072
_INTERMEDIATE = 12288
_GATE_UP_BITS = 5
_DOWN_BITS = 6
_GROUP_SIZE = 64


def _on_metal_device() -> bool:
    try:
        return mx.metal.is_available() and mx.default_device() == mx.Device(mx.gpu)
    except Exception:
        return False


def _proj_quant(proj, in_dim: int, out_dim: int, bits: int) -> bool:
    """Whether ``proj`` is an affine ``bits``-bit gs64 QuantizedLinear at shape.

    ``proj`` is a ``mlx.nn.QuantizedLinear`` (what mlx-lm builds for a
    bias-free ``nn.Linear`` under the checkpoint's quantization dict).  Requires
    the packed geometry this kernel unpacks by hand: whole ``uint32`` words per
    row *and* per group (so groups start on word boundaries).
    """

    if getattr(proj, "bits", None) != bits:
        return False
    if getattr(proj, "group_size", None) != _GROUP_SIZE:
        return False
    if getattr(proj, "mode", None) != "affine":
        return False
    for name in ("weight", "scales", "biases"):
        if not hasattr(proj, name) or getattr(proj, name) is None:
            return False
    weight, scales, biases = proj.weight, proj.scales, proj.biases
    if weight.dtype != mx.uint32 or weight.ndim != 2:
        return False
    if scales.ndim != 2 or biases.ndim != 2:
        return False
    if getattr(proj, "bias", None) is not None:
        return False
    # Whole words per group => group boundaries are word-aligned.
    if (in_dim * bits) % 32 != 0 or (_GROUP_SIZE * bits) % 32 != 0:
        return False
    if in_dim % _GROUP_SIZE != 0:
        return False
    groups = in_dim // _GROUP_SIZE
    words_per_row = (in_dim * bits) // 32
    if tuple(weight.shape) != (out_dim, words_per_row):
        return False
    if tuple(scales.shape) != (out_dim, groups) or tuple(biases.shape) != (out_dim, groups):
        return False
    # scales/biases are read as raw floats in-kernel; require a float family.
    if scales.dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if biases.dtype != scales.dtype:
        return False
    return True


def is_dense_mlp_eligible(mlp, x: mx.array) -> bool:
    """Whether the fused kernel covers this ``mlp(x)`` call on layer 0.

    Deliberately narrow: a bf16/fp16 2-D token batch and a dense MLP whose
    gate/up are affine 5-bit gs64 and down is affine 6-bit gs64 at the S-2.1
    layer-0 shape.  Anything else falls back to the stock ``mlp``.
    """

    if not _on_metal_device():
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 2:
        return False

    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    down = getattr(mlp, "down_proj", None)
    if gate is None or up is None or down is None:
        return False

    hidden = int(x.shape[-1])
    try:
        intermediate = int(gate.weight.shape[0])
    except Exception:
        return False
    if hidden <= 0 or intermediate <= 0:
        return False

    # gate/up must share bit width and group so one kernel serves both.
    if getattr(gate, "bits", None) != getattr(up, "bits", None):
        return False
    if getattr(gate, "group_size", None) != getattr(up, "group_size", None):
        return False
    if getattr(gate, "scales", None) is None or getattr(up, "scales", None) is None:
        return False
    if gate.scales.dtype != up.scales.dtype:
        return False

    if not _proj_quant(gate, hidden, intermediate, _GATE_UP_BITS):
        return False
    if not _proj_quant(up, hidden, intermediate, _GATE_UP_BITS):
        return False
    if not _proj_quant(down, intermediate, hidden, _DOWN_BITS):
        return False
    # down's scale dtype must match gate/up's (single template per kernel pair).
    if down.scales.dtype != gate.scales.dtype:
        return False
    return True


# --- unpack snippet shared by both kernels ---------------------------------
# Extract the `j`-th `BITS`-wide value of a word-aligned group.  Straddle reads
# the next word only when a value crosses a boundary; the group's last value
# always fits (group_size * BITS is a whole number of words), so `w0 + 1` never
# leaves the row.  `off >= 1` in the straddle branch, so `32 - off <= 31` (no
# undefined 32-shift).
_UNPACK = """
        // unpack value j of BITS bits from words[wbase + .], group-relative
        #define UNPACK_Q(WPTR, J, OUTQ) {                                   \\
            uint _bs = (J) * BITS;                                          \\
            uint _w0 = _bs >> 5;                                            \\
            uint _off = _bs & 31u;                                          \\
            uint _lo = (WPTR)[_w0] >> _off;                                 \\
            if (_off + BITS <= 32u) { OUTQ = _lo & MASK; }                  \\
            else { OUTQ = (_lo | ((WPTR)[_w0 + 1u] << (32u - _off))) & MASK; } \\
        }
"""


@lru_cache(maxsize=None)
def _gate_up_kernel(hidden: int, intermediate: int, bits: int, group_size: int, threads: int):
    words_per_row = (hidden * bits) // 32
    ngroups = hidden // group_size
    wpg = (group_size * bits) // 32
    header = f"""
        using namespace metal;
        constant constexpr uint HIDDEN = {hidden};
        constant constexpr uint OUT_DIM = {intermediate};
        constant constexpr uint BITS = {bits};
        constant constexpr uint GROUP = {group_size};
        constant constexpr uint MASK = {(1 << bits) - 1}u;
        constant constexpr uint NGROUPS = {ngroups};
        constant constexpr uint WORDS_ROW = {words_per_row};
        constant constexpr uint WPG = {wpg};
        constant constexpr uint TG = {threads};
        {_UNPACK}
    """

    source = """
        uint gid = thread_position_in_grid.x;
        uint total = ROWS * OUT_DIM;
        if (gid >= total) return;

        uint row = gid / OUT_DIM;
        uint m = gid - row * OUT_DIM;

        size_t g_wbase = (size_t)m * WORDS_ROW;
        size_t sb_base = (size_t)m * NGROUPS;
        size_t x_base  = (size_t)row * HIDDEN;

        float gacc = 0.0f;
        float uacc = 0.0f;
        for (uint grp = 0; grp < NGROUPS; ++grp) {
            float gsc = float(gate_s[sb_base + grp]);
            float gbi = float(gate_b[sb_base + grp]);
            float usc = float(up_s[sb_base + grp]);
            float ubi = float(up_b[sb_base + grp]);
            const device uint* gw = gate_w + g_wbase + (size_t)grp * WPG;
            const device uint* uw = up_w   + g_wbase + (size_t)grp * WPG;
            uint kbase = grp * GROUP;
            for (uint j = 0; j < GROUP; ++j) {
                float xv = float(x[x_base + kbase + j]);
                uint gq; UNPACK_Q(gw, j, gq);
                uint uq; UNPACK_Q(uw, j, uq);
                gacc += (float(gq) * gsc + gbi) * xv;
                uacc += (float(uq) * usc + ubi) * xv;
            }
        }

        // Round each projection to T (bf16) exactly where stock's
        // QuantizedLinear output rounds, then silu(gate) * up.
        T gbf = static_cast<T>(gacc);
        T ubf = static_cast<T>(uacc);
        float gf = float(gbf);
        float sig = 1.0f / (1.0f + metal::precise::exp(-gf));
        float hv = gf * sig * float(ubf);
        activated[gid] = static_cast<T>(hv);
    """
    # ROWS is templated per call via a constexpr injected in the source name;
    # bake it through an extra header constant so the grid can round up cleanly.
    return header, source, words_per_row, ngroups, wpg


@lru_cache(maxsize=None)
def _down_kernel(intermediate: int, hidden: int, bits: int, group_size: int, threads: int):
    words_per_row = (intermediate * bits) // 32
    ngroups = intermediate // group_size
    wpg = (group_size * bits) // 32
    header = f"""
        using namespace metal;
        constant constexpr uint IN_DIM = {intermediate};
        constant constexpr uint OUT_DIM = {hidden};
        constant constexpr uint BITS = {bits};
        constant constexpr uint GROUP = {group_size};
        constant constexpr uint MASK = {(1 << bits) - 1}u;
        constant constexpr uint NGROUPS = {ngroups};
        constant constexpr uint WORDS_ROW = {words_per_row};
        constant constexpr uint WPG = {wpg};
        constant constexpr uint TG = {threads};
        {_UNPACK}
    """

    source = """
        uint gid = thread_position_in_grid.x;
        uint total = ROWS * OUT_DIM;
        if (gid >= total) return;

        uint row = gid / OUT_DIM;
        uint n = gid - row * OUT_DIM;

        size_t d_wbase = (size_t)n * WORDS_ROW;
        size_t sb_base = (size_t)n * NGROUPS;
        size_t h_base  = (size_t)row * IN_DIM;

        float acc = 0.0f;
        for (uint grp = 0; grp < NGROUPS; ++grp) {
            float sc = float(down_s[sb_base + grp]);
            float bi = float(down_b[sb_base + grp]);
            const device uint* dw = down_w + d_wbase + (size_t)grp * WPG;
            uint kbase = grp * GROUP;
            for (uint j = 0; j < GROUP; ++j) {
                float hv = float(activated[h_base + kbase + j]);
                uint q; UNPACK_Q(dw, j, q);
                acc += (float(q) * sc + bi) * hv;
            }
        }
        out[gid] = static_cast<T>(acc);
    """
    return header, source, words_per_row, ngroups, wpg


def _make_kernel(name, header, source, rows, inputs_out):
    # ROWS is a per-call constant; inject it into the header so a cached kernel
    # is keyed on (shape, bits, threads, rows).
    full_header = header.replace(
        "using namespace metal;",
        f"using namespace metal;\n        constant constexpr uint ROWS = {rows};",
        1,
    )
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs_out[0],
        output_names=inputs_out[1],
        header=full_header,
        source=source,
    )


@lru_cache(maxsize=None)
def _gate_up_compiled(hidden, intermediate, bits, group_size, threads, rows):
    header, source, *_ = _gate_up_kernel(hidden, intermediate, bits, group_size, threads)
    return _make_kernel(
        f"mtplx_laguna_dense_gateup_h{hidden}_i{intermediate}_b{bits}_t{threads}_r{rows}",
        header,
        source,
        rows,
        (["x", "gate_w", "gate_s", "gate_b", "up_w", "up_s", "up_b"], ["activated"]),
    )


@lru_cache(maxsize=None)
def _down_compiled(intermediate, hidden, bits, group_size, threads, rows):
    header, source, *_ = _down_kernel(intermediate, hidden, bits, group_size, threads)
    return _make_kernel(
        f"mtplx_laguna_dense_down_i{intermediate}_h{hidden}_b{bits}_t{threads}_r{rows}",
        header,
        source,
        rows,
        (["activated", "down_w", "down_s", "down_b"], ["out"]),
    )


def _ceil_grid(total: int, threads: int) -> int:
    return ((total + threads - 1) // threads) * threads


def dense_swiglu_qmv(
    x: mx.array,
    gate_w: mx.array, gate_s: mx.array, gate_b: mx.array,
    up_w: mx.array, up_s: mx.array, up_b: mx.array,
    down_w: mx.array, down_s: mx.array, down_b: mx.array,
    *,
    hidden: int,
    intermediate: int,
    gate_up_bits: int = _GATE_UP_BITS,
    down_bits: int = _DOWN_BITS,
    group_size: int = _GROUP_SIZE,
    threads: int = 256,
) -> mx.array:
    """Fused dense-MLP output ``[rows, hidden]`` (x.dtype).

    Low-level entry: pass the raw quantized gate/up/down weight, scales, and
    biases (the arrays a ``QuantizedLinear`` holds).  Eligibility is the
    caller's responsibility here; use :func:`dense_mlp` for the guarded
    drop-in.  Returns the same dtype as ``x`` so it drops in for ``mlp(x)``.
    """

    rows = int(x.shape[0])
    threads = int(threads)
    if threads <= 0 or threads > 1024:
        threads = 256

    gu = _gate_up_compiled(hidden, intermediate, gate_up_bits, group_size, threads, rows)
    (activated,) = gu(
        inputs=[x, gate_w, gate_s, gate_b, up_w, up_s, up_b],
        template=[("T", x.dtype)],
        grid=(_ceil_grid(rows * intermediate, threads), 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, intermediate)],
        output_dtypes=[x.dtype],
    )
    # Trap guard: a wrong activation shape silently changes the work and can
    # fake a speedup; assert the geometry is exactly rows x intermediate.
    assert tuple(activated.shape) == (rows, intermediate), (
        f"dense gate/up produced {tuple(activated.shape)}, expected {(rows, intermediate)}"
    )

    dn = _down_compiled(intermediate, hidden, down_bits, group_size, threads, rows)
    (out,) = dn(
        inputs=[activated, down_w, down_s, down_b],
        template=[("T", x.dtype)],
        grid=(_ceil_grid(rows * hidden, threads), 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[x.dtype],
    )
    assert tuple(out.shape) == (rows, hidden), (
        f"dense down produced {tuple(out.shape)}, expected {(rows, hidden)}"
    )
    return out


def dense_mlp(mlp, x: mx.array) -> mx.array:
    """Drop-in for ``mlp(x)`` on the S-2.1 layer-0 dense MLP.

    Returns ``down(silu(gate(x)) * up(x))`` with the same shape/dtype the stock
    ``MLP.__call__`` returns.  Falls back to the stock ``mlp(x)`` on any
    shape/dtype/quant the fused kernel does not cover, so it can be switched on
    without owning a correctness branch.
    """

    leading = tuple(x.shape[:-1])
    hidden = int(x.shape[-1])
    x2 = x.reshape(-1, hidden)

    if not is_dense_mlp_eligible(mlp, x2):
        return mlp(x)

    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    intermediate = int(gate.weight.shape[0])
    out = dense_swiglu_qmv(
        x2,
        gate.weight, gate.scales, gate.biases,
        up.weight, up.scales, up.biases,
        down.weight, down.scales, down.biases,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_bits=int(gate.bits),
        down_bits=int(down.bits),
        group_size=int(gate.group_size),
    )
    return out.reshape(*leading, hidden)


# --- pure-mx numeric reference (what the kernel implements) -----------------

def _dequant_fp32(weight, scales, biases, group_size, bits):
    """FP32 dequant matching the kernel's ``float(q) * float(scale) + float(bias)``.

    Casting scales/biases to fp32 before ``mx.dequantize`` reproduces the
    kernel's in-register FP32 dequant (no intermediate bf16 rounding of the
    weight), which ``mx.dequantize`` would otherwise apply when scales are
    bf16.
    """

    return mx.dequantize(
        weight,
        scales.astype(mx.float32),
        biases.astype(mx.float32),
        group_size=int(group_size),
        bits=int(bits),
        mode="affine",
    )


def dense_mlp_reference(
    x: mx.array,
    gate_w, gate_s, gate_b, gate_bits, gate_gs,
    up_w, up_s, up_b, up_bits, up_gs,
    down_w, down_s, down_b, down_bits, down_gs,
) -> mx.array:
    """Pure-mx reference the Metal kernel reproduces (fp32 accumulate).

    Mirrors the kernel arithmetic exactly: FP32-dequant weights, FP32-accumulate
    each dot, round gate/up to the activation dtype at the projection boundary,
    ``silu(gate) * up`` (silu evaluated in fp32 from the rounded gate), then the
    FP32 down dot rounded to the activation dtype.  This is the fp64-accurate
    math; ``mx.quantized_matmul`` (stock) is a lossier approximation of it.
    """

    dt = x.dtype
    xf = x.astype(mx.float32)
    gw = _dequant_fp32(gate_w, gate_s, gate_b, gate_gs, gate_bits)   # [I, H]
    uw = _dequant_fp32(up_w, up_s, up_b, up_gs, up_bits)             # [I, H]
    dw = _dequant_fp32(down_w, down_s, down_b, down_gs, down_bits)   # [H, I]

    g = (xf @ gw.T)                          # fp32 [rows, I]
    u = (xf @ uw.T)
    gb = g.astype(dt).astype(mx.float32)     # round to activation dtype, widen
    ub = u.astype(dt).astype(mx.float32)
    sig = 1.0 / (1.0 + mx.exp(-gb))
    h = (gb * sig * ub).astype(dt)           # bf16 h, like stock's bf16 intermediate
    o = (h.astype(mx.float32) @ dw.T)
    return o.astype(dt)
