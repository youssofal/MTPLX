"""Fused shared-expert SwiGLU-QMV for the Laguna S-2.1 MoE decode step (D7).

Laguna's MoE block runs, on **every** token, one dense *shared expert* in
addition to the top-k routed experts:

    shared(x) = down_proj( silu(gate_proj(x)) * up_proj(x) )

(:class:`mtplx.models.laguna.MLP`, held as ``LagunaSparseMoeBlock.shared_expert``).
Unlike the routed experts this is a plain 2-D linear stack, quantized in the
oQ4e bank at **affine 8-bit, group_size 128** (see
``mtplx.models.laguna_config``: every ``mlp.shared_expert.{gate,up,down}_proj``
of the 47 sparse layers is uniform 8-bit/gs128).

## What it replaces

Stock decode runs the shared expert as three ``nn.QuantizedLinear`` calls
(``gate_proj``, ``up_proj``, ``down_proj``) with a compiled ``silu(gate)*up``
epilogue: three ``mx.quantized_matmul`` dispatches plus the elementwise glue.
This kernel collapses all of it into ONE dispatch — one threadgroup per token
computes gate & up by dequant-QMV, forms ``h = silu(gate) * up`` in threadgroup
memory (``h`` never touches device), then computes down by a second dequant-QMV
straight into the ``[rows, hidden]`` output the stock ``MLP.__call__`` returns.

## Fusion vs. the trap (expectation)

This is a sibling of :mod:`mtplx.kernels.laguna_moe_swiglu` (the routed variant).
That routed kernel already **lost ~25% at B=1** because top_k=10 lights only ten
threadgroups and under-fills the GPU. The shared expert is *worse* on that axis:
at B=1 there is exactly **ONE** token and **ONE** expert, so this kernel launches
a **single threadgroup** — the rest of the GPU is idle. Against MLX's tuned
``quantized_matmul`` (which fans a small matvec across many threadgroups) this is
expected to LOSE at decode; its only structural wins are three launches -> one
and keeping the 4 KB ``h`` row on chip. The companion check
(``scratchpad_moe_shared_check.py``) measures the gap honestly on the queued
lane; the public helper falls back to the stock ``MLP.__call__`` on any
shape/dtype/quant it does not cover, so a caller never owns a correctness branch.

Callers use :func:`shared_expert_swiglu` (drop-in for ``shared_mlp(x)``), or
check :func:`is_shared_swiglu_eligible` and call :func:`shared_swiglu_qmv`.
:func:`shared_swiglu_reference` is the pure-mx numeric reference (no
metal_kernel) the check proves the kernel and the stock module against.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


# The kernel is wired for the exact Laguna S-2.1 shared-expert geometry.
_BITS = 8
_GROUP_SIZE = 128
_PACK = 32 // _BITS  # 4 eight-bit values packed per uint32
_WORDS_PER_GROUP = _GROUP_SIZE // _PACK  # 32 uint32 words cover one 128-wide group


def _on_metal_device() -> bool:
    try:
        return mx.metal.is_available() and mx.default_device() == mx.Device(mx.gpu)
    except Exception:
        return False


def _quant_ok8(mod, in_dim: int, out_dim: int) -> bool:
    """Whether a QuantizedLinear submodule is affine 8-bit gs128 at this shape.

    ``mod`` is the parameter dict of an ``nn.QuantizedLinear`` (the object the
    stock shared ``MLP`` holds after quantization).  Checks bit width, group
    size, affine mode, and that the packed weight / scales / biases carry the
    2-D geometry this kernel unpacks by hand.
    """

    if getattr(mod, "bits", None) != _BITS:
        return False
    if getattr(mod, "group_size", None) != _GROUP_SIZE:
        return False
    if getattr(mod, "mode", None) != "affine":
        return False
    if "weight" not in mod or "scales" not in mod or "biases" not in mod:
        return False
    if mod.get("biases") is None:
        return False
    weight, scales, biases = mod["weight"], mod["scales"], mod["biases"]
    if weight.dtype != mx.uint32 or weight.ndim != 2:
        return False
    if scales.ndim != 2 or biases.ndim != 2:
        return False
    if tuple(weight.shape) != (out_dim, in_dim // _PACK):
        return False
    groups = in_dim // _GROUP_SIZE
    if tuple(scales.shape) != (out_dim, groups):
        return False
    if tuple(biases.shape) != (out_dim, groups):
        return False
    # Per-group scales/biases are read as raw float; require float32 (the oQ4e
    # export dtype) so the dequant matches mx.dequantize.
    if scales.dtype != mx.float32 or biases.dtype != mx.float32:
        return False
    return True


def is_shared_swiglu_eligible(shared_mlp, x: mx.array) -> bool:
    """Whether the fused kernel covers this exact ``shared_mlp(x)`` call.

    Deliberately narrow: bf16/fp16 token rows, an affine 8-bit gs128
    gate/up/down stack at the S-2.1 shape (hidden % 128 == 0,
    shared_intermediate % 128 == 0), and no per-projection bias.  Anything else
    falls back to the stock ``MLP.__call__``.
    """

    if not _on_metal_device():
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 2:
        return False

    gate = getattr(shared_mlp, "gate_proj", None)
    up = getattr(shared_mlp, "up_proj", None)
    down = getattr(shared_mlp, "down_proj", None)
    if gate is None or up is None or down is None:
        return False

    hidden = int(x.shape[-1])
    # shared_intermediate is the gate/up output width.
    try:
        shared_inter = int(gate["weight"].shape[0])
    except Exception:
        return False
    if hidden <= 0 or shared_inter <= 0:
        return False
    if hidden % _GROUP_SIZE != 0 or shared_inter % _GROUP_SIZE != 0:
        return False
    if hidden % _PACK != 0 or shared_inter % _PACK != 0:
        return False

    if not _quant_ok8(gate, hidden, shared_inter):
        return False
    if not _quant_ok8(up, hidden, shared_inter):
        return False
    if not _quant_ok8(down, shared_inter, hidden):
        return False
    # A per-projection bias (Linear bias=True) is not fused.
    if "bias" in gate or "bias" in up or "bias" in down:
        return False
    return True


@lru_cache(maxsize=None)
def _shared_swiglu_kernel(hidden: int, shared_inter: int, threads: int):
    in_packed = hidden // _PACK
    ng_in = hidden // _GROUP_SIZE
    si_packed = shared_inter // _PACK
    ng_si = shared_inter // _GROUP_SIZE

    header = f"""
        using namespace metal;
        constant constexpr uint HIDDEN = {hidden};
        constant constexpr uint SHARED_INTER = {shared_inter};
        constant constexpr uint TG = {threads};
        constant constexpr uint IN_PACKED = {in_packed};
        constant constexpr uint NG_IN = {ng_in};
        constant constexpr uint SI_PACKED = {si_packed};
        constant constexpr uint NG_SI = {ng_si};
        constant constexpr uint PACK = {_PACK};
        constant constexpr uint WPG = {_WORDS_PER_GROUP};
    """

    # One threadgroup per token row (tg == row). Phase 1: gate & up dequant-QMV
    # over HIDDEN, fuse silu(gate)*up into hs[SHARED_INTER] in threadgroup memory.
    # Phase 2: down dequant-QMV over SHARED_INTER straight into out[row, :].
    source = """
        uint tg = threadgroup_position_in_grid.x;   // == token row
        uint lid = thread_position_in_threadgroup.x;

        threadgroup float xs[HIDDEN];
        threadgroup float hs[SHARED_INTER];

        // --- stage the token row in threadgroup memory (bf16/fp16 -> float) ---
        for (uint k = lid; k < HIDDEN; k += TG) {
            xs[k] = float(x[(size_t)tg * HIDDEN + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 1: gate & up QMV (HIDDEN -> SHARED_INTER), fused SwiGLU ---
        for (uint m = lid; m < SHARED_INTER; m += TG) {
            const device uint*  gate_row = gate_w + (size_t)m * IN_PACKED;
            const device uint*  up_row   = up_w   + (size_t)m * IN_PACKED;
            const device float* gate_sc  = gate_s + (size_t)m * NG_IN;
            const device float* gate_bi  = gate_b + (size_t)m * NG_IN;
            const device float* up_sc    = up_s   + (size_t)m * NG_IN;
            const device float* up_bi    = up_b   + (size_t)m * NG_IN;

            float gacc = 0.0f;
            float uacc = 0.0f;
            for (uint g = 0; g < NG_IN; ++g) {
                float gsc = gate_sc[g];
                float gbi = gate_bi[g];
                float usc = up_sc[g];
                float ubi = up_bi[g];
                uint kbase = g * WPG * PACK;   // = g * 128
                uint wbase = g * WPG;
                for (uint wi = 0; wi < WPG; ++wi) {
                    uint gw = gate_row[wbase + wi];
                    uint uw = up_row[wbase + wi];
                    uint k = kbase + wi * PACK;
                    for (uint t = 0; t < PACK; ++t) {
                        float xv = xs[k + t];
                        uint gq = (gw >> (8u * t)) & 0xFFu;
                        uint uq = (uw >> (8u * t)) & 0xFFu;
                        gacc += (float(gq) * gsc + gbi) * xv;
                        uacc += (float(uq) * usc + ubi) * xv;
                    }
                }
            }
            // silu(gate) * up  ==  gate * sigmoid(gate) * up
            float sig = 1.0f / (1.0f + metal::precise::exp(-gacc));
            hs[m] = gacc * sig * uacc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 2: down QMV (SHARED_INTER -> HIDDEN) into the row output ---
        size_t out_base = (size_t)tg * HIDDEN;
        for (uint n = lid; n < HIDDEN; n += TG) {
            const device uint*  dw  = down_w + (size_t)n * SI_PACKED;
            const device float* dsc = down_s + (size_t)n * NG_SI;
            const device float* dbi = down_b + (size_t)n * NG_SI;
            float acc = 0.0f;
            for (uint g = 0; g < NG_SI; ++g) {
                float sc = dsc[g];
                float bi = dbi[g];
                uint kbase = g * WPG * PACK;
                uint wbase = g * WPG;
                for (uint wi = 0; wi < WPG; ++wi) {
                    uint w = dw[wbase + wi];
                    uint k = kbase + wi * PACK;
                    for (uint t = 0; t < PACK; ++t) {
                        uint q = (w >> (8u * t)) & 0xFFu;
                        acc += (float(q) * sc + bi) * hs[k + t];
                    }
                }
            }
            out[out_base + n] = acc;
        }
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_moe_shared_h{hidden}_i{shared_inter}_t{threads}",
        input_names=[
            "x",
            "gate_w", "gate_s", "gate_b",
            "up_w", "up_s", "up_b",
            "down_w", "down_s", "down_b",
        ],
        output_names=["out"],
        header=header,
        source=source,
    )


def shared_swiglu_qmv(
    x: mx.array,
    gate_w: mx.array, gate_s: mx.array, gate_b: mx.array,
    up_w: mx.array, up_s: mx.array, up_b: mx.array,
    down_w: mx.array, down_s: mx.array, down_b: mx.array,
    *,
    hidden: int,
    shared_intermediate: int,
    threads: int = 256,
) -> mx.array:
    """Fused shared-expert SwiGLU output ``[rows, hidden]`` (float32).

    Low-level entry: pass the raw quantized ``gate/up/down`` weight, scales, and
    biases arrays (the same objects the stock shared ``MLP`` submodules hold).
    Eligibility is the caller's responsibility here; use
    :func:`shared_expert_swiglu` for the guarded drop-in.
    """

    rows = int(x.shape[0])
    threads = int(threads)
    if threads <= 0 or threads > 1024:
        threads = 256

    kernel = _shared_swiglu_kernel(hidden, shared_intermediate, threads)
    (out,) = kernel(
        inputs=[
            x,
            gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            down_w, down_s, down_b,
        ],
        template=[("T", x.dtype)],
        grid=(threads * rows, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, hidden)],
        output_dtypes=[mx.float32],
    )
    # Hard trap guard: a wrong activation shape silently does the wrong amount of
    # work and can FAKE a speedup. Assert exactly one output row per token.
    assert tuple(out.shape) == (rows, hidden), (
        f"shared_swiglu_qmv produced {tuple(out.shape)}, expected {(rows, hidden)}"
    )
    return out


def shared_swiglu_reference(
    x: mx.array,
    gate_w: mx.array, gate_s: mx.array, gate_b: mx.array,
    up_w: mx.array, up_s: mx.array, up_b: mx.array,
    down_w: mx.array, down_s: mx.array, down_b: mx.array,
    *,
    hidden: int,
    shared_intermediate: int,
) -> mx.array:
    """Pure-mx numeric reference for :func:`shared_swiglu_qmv` (no metal_kernel).

    Reproduces the kernel's arithmetic with ``mx.dequantize`` + float32 matmuls
    and an explicit ``silu(gate)*up`` so the check can prove
    ``kernel == reference == stock`` without leaning on the kernel's own path.
    Returns ``[rows, hidden]`` float32.
    """

    xf = x.astype(mx.float32)
    gW = mx.dequantize(
        gate_w, gate_s, gate_b, group_size=_GROUP_SIZE, bits=_BITS, mode="affine"
    )
    uW = mx.dequantize(
        up_w, up_s, up_b, group_size=_GROUP_SIZE, bits=_BITS, mode="affine"
    )
    dW = mx.dequantize(
        down_w, down_s, down_b, group_size=_GROUP_SIZE, bits=_BITS, mode="affine"
    )
    g = xf @ gW.T                      # [rows, shared_inter]
    u = xf @ uW.T                      # [rows, shared_inter]
    h = (g * mx.sigmoid(g)) * u        # silu(gate) * up
    out = h @ dW.T                     # [rows, hidden]
    assert tuple(out.shape) == (int(x.shape[0]), hidden)
    return out


def shared_expert_swiglu(shared_mlp, x: mx.array) -> mx.array:
    """Drop-in for ``shared_mlp(x)`` on the S-2.1 shared-expert path.

    Returns the shared-expert SwiGLU output ``[rows, hidden]`` (matching the
    stock ``MLP.__call__`` return; float32 vs. the stock bf16, a delta below
    bf16 resolution).  Falls back to the stock ``shared_mlp(x)`` on any
    shape/dtype/quant the fused kernel does not cover, so it can be switched on
    without owning a correctness branch.
    """

    if not is_shared_swiglu_eligible(shared_mlp, x):
        return shared_mlp(x)

    gate, up, down = shared_mlp.gate_proj, shared_mlp.up_proj, shared_mlp.down_proj
    hidden = int(x.shape[-1])
    shared_inter = int(gate["weight"].shape[0])
    return shared_swiglu_qmv(
        x,
        gate["weight"], gate["scales"], gate["biases"],
        up["weight"], up["scales"], up["biases"],
        down["weight"], down["scales"], down["biases"],
        hidden=hidden,
        shared_intermediate=shared_inter,
    )
