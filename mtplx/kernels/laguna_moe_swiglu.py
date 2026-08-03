"""Fused routed-expert SwiGLU-QMV for the Laguna S-2.1 MoE decode step.

Ported from the mlx.fast **Laguna XS2.1** challenge kernel that fuses one
routed expert's SwiGLU (``down(silu(gate(x)) * up(x))``) into a single hand QMV
dispatch, re-expressed for **Laguna S-2.1**'s affine oQ4e bank instead of the
challenge's NVFP4:

    axis (hidden)       XS2.1 2048   ->  S2.1 3072
    moe_intermediate    (donor)      ->  S2.1 1024
    experts / top_k     256 / (donor)->  256 / 10
    routed quant        NVFP4        ->  affine 4-bit, group_size 128
                                         (w = q*scale + bias per group of 128)

## What it replaces

The stock decode expert path runs ``mlx_lm.models.switch_layers.SwitchGLU``,
which at B=1 (one token, top_k=10, ``indices.size < 64`` so no gather-sort)
issues THREE ``mx.gather_qmm`` dispatches (gate, up, down) plus a compiled
``silu(gate)*up`` epilogue.  This kernel collapses the three projections and the
epilogue into ONE dispatch: one threadgroup per (token, selected-expert) pair
computes gate & up by dequant-QMV, forms ``h = silu(gate) * up`` **in
threadgroup memory** (h never touches device), then computes down by a second
dequant-QMV straight into the ``[rows, top_k, hidden]`` output the stock
``switch_mlp`` returns.

## Fusion vs. the trap

At B=1 this path is bandwidth-bound on the ~4.7 MB of 4-bit weight read per
expert; the fusion's only structural wins are (a) three launches -> one and
(b) keeping the 4 KB ``h`` row on chip.  The known risk (see the project's
"Metal sub-4-bit is ALU-bound" / "IQ2_XXS kernel loses to stock" findings) is
that a hand affine-dequant QMV is ALU/occupancy-bound and can LOSE to MLX's
tuned ``mx.gather_qmm`` — at top_k=10 only 10 threadgroups are live, so a
one-threadgroup-per-expert layout under-fills the GPU.  The companion check
(:mod:`scratchpad_moe_check`) measures this honestly on the queued lane; the
public helper falls back to the stock ``switch_mlp`` on any shape/dtype it does
not cover, so a caller never owns a correctness branch.

Callers use :func:`routed_expert_swiglu` (drop-in for ``switch_mlp(x, idx)``)
or check :func:`is_routed_swiglu_eligible` and call :func:`routed_swiglu_qmv`.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx


# The kernel is wired for the exact Laguna S-2.1 routed-expert geometry.
_BITS = 4
_GROUP_SIZE = 128
_PACK = 8  # 4-bit values packed per uint32
_WORDS_PER_GROUP = _GROUP_SIZE // _PACK  # 16 uint32 words cover one 128-wide group


def _on_metal_device() -> bool:
    try:
        return mx.metal.is_available() and mx.default_device() == mx.Device(mx.gpu)
    except Exception:
        return False


def _quant_ok(mod, in_dim: int, out_dim: int) -> bool:
    """Whether a SwitchLinear submodule is affine 4-bit gs128 at the given shape.

    ``mod`` is the parameter dict of a ``QuantizedSwitchLinear`` (the object the
    stock ``SwitchGLU`` holds).  Checks bit width, group size, affine mode, and
    that the packed weight / scales / biases carry the geometry this kernel
    unpacks by hand.
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
    if weight.dtype != mx.uint32 or weight.ndim != 3:
        return False
    if scales.ndim != 3 or biases.ndim != 3:
        return False
    experts = int(weight.shape[0])
    if tuple(weight.shape) != (experts, out_dim, in_dim // _PACK):
        return False
    groups = in_dim // _GROUP_SIZE
    if tuple(scales.shape) != (experts, out_dim, groups):
        return False
    if tuple(biases.shape) != (experts, out_dim, groups):
        return False
    # Per-group scales/biases are read as raw float; require float32 (the oQ4e
    # export dtype) so the dequant matches mx.dequantize.
    if scales.dtype != mx.float32 or biases.dtype != mx.float32:
        return False
    return True


def is_routed_swiglu_eligible(switch_mlp, x: mx.array, indices: mx.array) -> bool:
    """Whether the fused kernel covers this exact ``switch_mlp(x, indices)`` call.

    Deliberately narrow: bf16/fp16 token row, an affine 4-bit gs128 gate/up/down
    bank at the S-2.1 shape (hidden % 128 == 0, moe_intermediate % 128 == 0),
    and a 2-D ``[rows, top_k]`` integer index set.  Anything else falls back to
    the stock ``switch_mlp``.
    """

    if not _on_metal_device():
        return False
    if x.dtype not in (mx.bfloat16, mx.float16):
        return False
    if x.ndim != 2:
        return False
    if indices.ndim != 2 or int(indices.shape[0]) != int(x.shape[0]):
        return False
    if indices.dtype not in (mx.uint32, mx.int32, mx.int64, mx.uint64):
        return False

    gate = getattr(switch_mlp, "gate_proj", None)
    up = getattr(switch_mlp, "up_proj", None)
    down = getattr(switch_mlp, "down_proj", None)
    if gate is None or up is None or down is None:
        return False

    hidden = int(x.shape[-1])
    # moe_intermediate is the gate/up output width.
    try:
        moe_inter = int(gate["weight"].shape[1])
    except Exception:
        return False
    if hidden <= 0 or moe_inter <= 0:
        return False
    if hidden % _GROUP_SIZE != 0 or moe_inter % _GROUP_SIZE != 0:
        return False
    if hidden % _PACK != 0 or moe_inter % _PACK != 0:
        return False

    experts = int(gate["weight"].shape[0])
    if not _quant_ok(gate, hidden, moe_inter):
        return False
    if not _quant_ok(up, hidden, moe_inter):
        return False
    if not _quant_ok(down, moe_inter, hidden):
        return False
    # gate/up/down must agree on the expert count.
    if int(up["weight"].shape[0]) != experts or int(down["weight"].shape[0]) != experts:
        return False
    # A per-expert bias term (SwitchLinear bias=True) is not fused.
    if "bias" in gate or "bias" in up or "bias" in down:
        return False

    top_k = int(indices.shape[1])
    if top_k <= 0 or top_k > experts:
        return False
    return True


@lru_cache(maxsize=None)
def _routed_swiglu_kernel(hidden: int, moe_inter: int, top_k: int, threads: int):
    in_packed = hidden // _PACK
    ng_in = hidden // _GROUP_SIZE
    mi_packed = moe_inter // _PACK
    ng_mi = moe_inter // _GROUP_SIZE

    header = f"""
        using namespace metal;
        constant constexpr uint HIDDEN = {hidden};
        constant constexpr uint MOE_INTER = {moe_inter};
        constant constexpr uint TOP_K = {top_k};
        constant constexpr uint TG = {threads};
        constant constexpr uint IN_PACKED = {in_packed};
        constant constexpr uint NG_IN = {ng_in};
        constant constexpr uint MI_PACKED = {mi_packed};
        constant constexpr uint NG_MI = {ng_mi};
        constant constexpr uint WPG = {_WORDS_PER_GROUP};
    """

    # One threadgroup per (row, slot). Phase 1: gate & up dequant-QMV over HIDDEN,
    # fuse silu(gate)*up into hs[MOE_INTER] in threadgroup memory. Phase 2: down
    # dequant-QMV over MOE_INTER straight into out[row, slot, :].
    source = """
        uint tg = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;

        uint row = tg / TOP_K;
        uint slot = tg - row * TOP_K;
        uint e = uint(indices[(size_t)row * TOP_K + slot]);

        threadgroup float xs[HIDDEN];
        threadgroup float hs[MOE_INTER];

        // --- stage the token row in threadgroup memory (bf16/fp16 -> float) ---
        for (uint k = lid; k < HIDDEN; k += TG) {
            xs[k] = float(x[(size_t)row * HIDDEN + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- Phase 1: gate & up QMV, then fused SwiGLU into hs[] ---
        size_t g_wbase = (size_t)e * MOE_INTER * IN_PACKED;
        size_t g_sbase = (size_t)e * MOE_INTER * NG_IN;
        for (uint m = lid; m < MOE_INTER; m += TG) {
            const device uint* gate_row = gate_w + g_wbase + (size_t)m * IN_PACKED;
            const device uint* up_row   = up_w   + g_wbase + (size_t)m * IN_PACKED;
            const device float* gate_sc = gate_s + g_sbase + (size_t)m * NG_IN;
            const device float* gate_bi = gate_b + g_sbase + (size_t)m * NG_IN;
            const device float* up_sc   = up_s   + g_sbase + (size_t)m * NG_IN;
            const device float* up_bi   = up_b   + g_sbase + (size_t)m * NG_IN;

            float gacc = 0.0f;
            float uacc = 0.0f;
            for (uint g = 0; g < NG_IN; ++g) {
                float gsc = gate_sc[g];
                float gbi = gate_bi[g];
                float usc = up_sc[g];
                float ubi = up_bi[g];
                uint kbase = g * WPG * 8u;      // = g * 128
                uint wbase = g * WPG;
                for (uint wi = 0; wi < WPG; ++wi) {
                    uint gw = gate_row[wbase + wi];
                    uint uw = up_row[wbase + wi];
                    uint k = kbase + wi * 8u;
                    for (uint t = 0; t < 8u; ++t) {
                        float xv = xs[k + t];
                        uint gq = (gw >> (4u * t)) & 0xFu;
                        uint uq = (uw >> (4u * t)) & 0xFu;
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

        // --- Phase 2: down QMV (MOE_INTER -> HIDDEN) into the expert output ---
        size_t d_wbase = (size_t)e * HIDDEN * MI_PACKED;
        size_t d_sbase = (size_t)e * HIDDEN * NG_MI;
        size_t out_base = (size_t)tg * HIDDEN;   // tg == row*TOP_K + slot
        for (uint n = lid; n < HIDDEN; n += TG) {
            const device uint* dw = down_w + d_wbase + (size_t)n * MI_PACKED;
            const device float* dsc = down_s + d_sbase + (size_t)n * NG_MI;
            const device float* dbi = down_b + d_sbase + (size_t)n * NG_MI;
            float acc = 0.0f;
            for (uint g = 0; g < NG_MI; ++g) {
                float sc = dsc[g];
                float bi = dbi[g];
                uint kbase = g * WPG * 8u;
                uint wbase = g * WPG;
                for (uint wi = 0; wi < WPG; ++wi) {
                    uint w = dw[wbase + wi];
                    uint k = kbase + wi * 8u;
                    for (uint t = 0; t < 8u; ++t) {
                        uint q = (w >> (4u * t)) & 0xFu;
                        acc += (float(q) * sc + bi) * hs[k + t];
                    }
                }
            }
            out[out_base + n] = acc;
        }
    """

    return mx.fast.metal_kernel(
        name=f"mtplx_laguna_moe_swiglu_h{hidden}_i{moe_inter}_k{top_k}_t{threads}",
        input_names=[
            "x",
            "indices",
            "gate_w", "gate_s", "gate_b",
            "up_w", "up_s", "up_b",
            "down_w", "down_s", "down_b",
        ],
        output_names=["out"],
        header=header,
        source=source,
    )


def routed_swiglu_qmv(
    x: mx.array,
    indices: mx.array,
    gate_w: mx.array, gate_s: mx.array, gate_b: mx.array,
    up_w: mx.array, up_s: mx.array, up_b: mx.array,
    down_w: mx.array, down_s: mx.array, down_b: mx.array,
    *,
    hidden: int,
    moe_intermediate: int,
    threads: int = 256,
) -> mx.array:
    """Fused routed-expert SwiGLU output ``[rows, top_k, hidden]`` (float32).

    Low-level entry: pass the raw quantized ``gate/up/down`` weight, scales, and
    biases arrays (the same objects the stock ``SwitchGLU`` submodules hold).
    Eligibility is the caller's responsibility here; use
    :func:`routed_expert_swiglu` for the guarded drop-in.
    """

    rows = int(x.shape[0])
    top_k = int(indices.shape[1])
    idx_u = indices if indices.dtype == mx.uint32 else indices.astype(mx.uint32)

    # threads must cover the token row in the staging loop and both QMV loops;
    # the loops are strided by TG so any positive value is correct, but pick one
    # that does not exceed the threadgroup limit.
    threads = int(threads)
    if threads <= 0 or threads > 1024:
        threads = 256

    kernel = _routed_swiglu_kernel(hidden, moe_intermediate, top_k, threads)
    groups = rows * top_k
    (out,) = kernel(
        inputs=[
            x, idx_u,
            gate_w, gate_s, gate_b,
            up_w, up_s, up_b,
            down_w, down_s, down_b,
        ],
        template=[("T", x.dtype)],
        grid=(threads * groups, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, top_k, hidden)],
        output_dtypes=[mx.float32],
    )
    # Hard trap guard: a wrong activation/index shape silently does ~8x the work
    # and can FAKE a speedup. Assert the output geometry is exactly one row per
    # (token, selected-expert).
    assert tuple(out.shape) == (rows, top_k, hidden), (
        f"routed_swiglu_qmv produced {tuple(out.shape)}, expected {(rows, top_k, hidden)}"
    )
    return out


def routed_expert_swiglu(switch_mlp, x: mx.array, indices: mx.array) -> mx.array:
    """Drop-in for ``switch_mlp(flattened, indices)`` on the S-2.1 expert path.

    Returns the routed-expert SwiGLU output ``[rows, top_k, hidden]`` (matching
    the stock ``SwitchGLU.__call__`` return, float32).  Falls back to the stock
    ``switch_mlp(x, indices)`` on any shape/dtype/quant the fused kernel does not
    cover, so it can be switched on without owning a correctness branch.
    """

    if not is_routed_swiglu_eligible(switch_mlp, x, indices):
        return switch_mlp(x, indices)

    gate, up, down = switch_mlp.gate_proj, switch_mlp.up_proj, switch_mlp.down_proj
    hidden = int(x.shape[-1])
    moe_inter = int(gate["weight"].shape[1])
    return routed_swiglu_qmv(
        x, indices,
        gate["weight"], gate["scales"], gate["biases"],
        up["weight"], up["scales"], up["biases"],
        down["weight"], down["scales"], down["biases"],
        hidden=hidden,
        moe_intermediate=moe_inter,
    )
