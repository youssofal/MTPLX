"""Merged routed+shared SwiGLU-QMV for the Laguna S-2.1 MoE decode step (D9).

The mlx.fast **Laguna XS2.1** challenge has a "9-slot merged" MoE kernel: it
fuses the token's top-8 routed experts **and** the one shared expert into a
single dispatch (8 routed + 1 shared = 9 slots). For **Laguna S-2.1** the top-k
is 10, so the merge is **11 slots** (top-10 routed + 1 shared); this module
adapts the count.

The block it replaces is the two-line combine in
:meth:`mtplx.models.laguna.LagunaSparseMoeBlock.__call__`::

    output = self.switch_mlp(flattened, indices)                    # routed
    output = MOE_COMBINE_IMPL(output, weights, self.shared_expert(flattened))

i.e. ``sum_k weights[k] * routed_expert_k(x) + shared_expert(x)``.

## What the kernel does

One threadgroup per ``(token, slot)`` for ``SLOTS = top_k + 1`` slots:

    slot < top_k : routed expert ``e = indices[row, slot]`` — affine **4-bit**
                   gs128 (the S-2.1 routed bank); its SwiGLU is pre-scaled by
                   the combine weight ``weights[row, slot]``.
    slot == top_k: the shared expert — affine **8-bit** gs128; SwiGLU un-scaled
                   (the shared term is added, not routed-weighted).

Each threadgroup computes gate & up by dequant-QMV, forms ``h = silu(gate)*up``
in threadgroup memory, then computes down by a second dequant-QMV into
``out[row, slot, :]`` (pre-scaled).  The kernel emits ``[rows, SLOTS, hidden]``;
the drop-in returns ``out.sum(axis=1)`` == the combined MoE output.  Reducing
outside the kernel keeps the fusion atomic-free and mirrors the routed sibling
:mod:`mtplx.kernels.laguna_moe_swiglu` (which returns ``[rows, top_k, hidden]``).
The slot branch is threadgroup-uniform (all threads in a group share one slot),
so the two width paths and their barriers never diverge within a group.

## Fusion vs. the trap (expectation)

Same occupancy story as the routed and shared siblings, and the merge does NOT
escape it: at B=1 this lights **SLOTS = 11 threadgroups** (10 routed + 1
shared), still far under a full GPU, against MLX's tuned ``gather_qmm`` /
``quantized_matmul`` which fan each small matvec across many threadgroups.  The
routed-only kernel already lost ~25% at B=1; folding the shared expert in adds
one more threadgroup, not occupancy, so this is expected to LOSE at decode too.
Its structural wins are launch-count (many dispatches -> one) and keeping every
slot's ``h`` on chip.  ``scratchpad_moe_merged_check.py`` measures the gap
honestly; the public helper falls back to the stock routed+shared combine on any
shape/dtype/quant it does not cover.

Callers use :func:`merged_expert_swiglu` (drop-in for the two combine lines) or
check :func:`is_merged_swiglu_eligible` and call :func:`merged_swiglu_qmv`.
:func:`merged_swiglu_reference` is the pure-mx numeric reference.
"""

from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .laguna_moe_shared import (
    _quant_ok8,
    is_shared_swiglu_eligible,
)
from .laguna_moe_swiglu import (
    is_routed_swiglu_eligible,
)


_GROUP_SIZE = 128

# Routed bank: affine 4-bit gs128.
_BITS4 = 4
_PACK4 = 32 // _BITS4          # 8
_WPG4 = _GROUP_SIZE // _PACK4  # 16

# Shared bank: affine 8-bit gs128.
_BITS8 = 8
_PACK8 = 32 // _BITS8          # 4
_WPG8 = _GROUP_SIZE // _PACK8  # 32


def _on_metal_device() -> bool:
    try:
        return mx.metal.is_available() and mx.default_device() == mx.Device(mx.gpu)
    except Exception:
        return False


def is_merged_swiglu_eligible(
    switch_mlp, shared_mlp, x: mx.array, indices: mx.array, weights: mx.array
) -> bool:
    """Whether the merged kernel covers this exact routed+shared combine.

    Requires BOTH the routed 4-bit gs128 bank (reusing the routed sibling's
    eligibility) AND the shared 8-bit gs128 stack (reusing the shared sibling's
    eligibility), a 2-D ``[rows, top_k]`` index set, and a matching 2-D float
    ``weights`` set.  Anything else falls back to the stock routed+shared path.
    """

    if not _on_metal_device():
        return False
    if not is_routed_swiglu_eligible(switch_mlp, x, indices):
        return False
    if not is_shared_swiglu_eligible(shared_mlp, x):
        return False
    if weights.ndim != 2 or tuple(weights.shape) != tuple(indices.shape):
        return False
    if weights.dtype not in (mx.float32, mx.bfloat16, mx.float16):
        return False
    return True


@lru_cache(maxsize=None)
def _merged_swiglu_kernel(
    hidden: int, moe_inter: int, shared_inter: int, top_k: int, threads: int
):
    slots = top_k + 1
    max_inter = max(moe_inter, shared_inter)

    # Routed (4-bit) geometry.
    r_in_packed = hidden // _PACK4
    r_mi_packed = moe_inter // _PACK4
    r_ng_mi = moe_inter // _GROUP_SIZE
    # Shared (8-bit) geometry.
    s_in_packed = hidden // _PACK8
    s_si_packed = shared_inter // _PACK8
    s_ng_si = shared_inter // _GROUP_SIZE
    # gate/up read HIDDEN under gs128 for both banks -> shared group count.
    ng_in = hidden // _GROUP_SIZE

    header = f"""
        using namespace metal;
        constant constexpr uint HIDDEN = {hidden};
        constant constexpr uint MOE_INTER = {moe_inter};
        constant constexpr uint SHARED_INTER = {shared_inter};
        constant constexpr uint MAX_INTER = {max_inter};
        constant constexpr uint TOP_K = {top_k};
        constant constexpr uint SLOTS = {slots};
        constant constexpr uint TG = {threads};
        constant constexpr uint NG_IN = {ng_in};
        constant constexpr uint R_IN_PACKED = {r_in_packed};
        constant constexpr uint R_MI_PACKED = {r_mi_packed};
        constant constexpr uint R_NG_MI = {r_ng_mi};
        constant constexpr uint R_WPG = {_WPG4};
        constant constexpr uint S_IN_PACKED = {s_in_packed};
        constant constexpr uint S_SI_PACKED = {s_si_packed};
        constant constexpr uint S_NG_SI = {s_ng_si};
        constant constexpr uint S_WPG = {_WPG8};
    """

    # tg == row * SLOTS + slot. slot < TOP_K -> routed 4-bit expert
    # indices[row, slot], contribution scaled by weights[row, slot]. slot ==
    # TOP_K -> shared 8-bit expert, contribution un-scaled. Each slot writes its
    # full [hidden] contribution to out[row, slot, :]; the caller sums axis 1.
    source = """
        uint tg = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;

        uint row = tg / SLOTS;
        uint slot = tg - row * SLOTS;

        threadgroup float xs[HIDDEN];
        threadgroup float hs[MAX_INTER];

        // --- stage the token row (bf16/fp16 -> float) ---
        for (uint k = lid; k < HIDDEN; k += TG) {
            xs[k] = float(x[(size_t)row * HIDDEN + k]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        size_t out_base = (size_t)tg * HIDDEN;   // out[row, slot, :]

        if (slot < TOP_K) {
            // ================= routed expert (affine 4-bit) =================
            uint e = uint(indices[(size_t)row * TOP_K + slot]);
            float wscale = float(weights[(size_t)row * TOP_K + slot]);

            size_t g_wbase = (size_t)e * MOE_INTER * R_IN_PACKED;
            size_t g_sbase = (size_t)e * MOE_INTER * NG_IN;
            for (uint m = lid; m < MOE_INTER; m += TG) {
                const device uint*  gate_row = gate_w4 + g_wbase + (size_t)m * R_IN_PACKED;
                const device uint*  up_row   = up_w4   + g_wbase + (size_t)m * R_IN_PACKED;
                const device float* gate_sc  = gate_s4 + g_sbase + (size_t)m * NG_IN;
                const device float* gate_bi  = gate_b4 + g_sbase + (size_t)m * NG_IN;
                const device float* up_sc    = up_s4   + g_sbase + (size_t)m * NG_IN;
                const device float* up_bi    = up_b4   + g_sbase + (size_t)m * NG_IN;

                float gacc = 0.0f;
                float uacc = 0.0f;
                for (uint g = 0; g < NG_IN; ++g) {
                    float gsc = gate_sc[g];
                    float gbi = gate_bi[g];
                    float usc = up_sc[g];
                    float ubi = up_bi[g];
                    uint kbase = g * R_WPG * 8u;   // = g * 128
                    uint wbase = g * R_WPG;
                    for (uint wi = 0; wi < R_WPG; ++wi) {
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
                float sig = 1.0f / (1.0f + metal::precise::exp(-gacc));
                hs[m] = gacc * sig * uacc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            size_t d_wbase = (size_t)e * HIDDEN * R_MI_PACKED;
            size_t d_sbase = (size_t)e * HIDDEN * R_NG_MI;
            for (uint n = lid; n < HIDDEN; n += TG) {
                const device uint*  dw  = down_w4 + d_wbase + (size_t)n * R_MI_PACKED;
                const device float* dsc = down_s4 + d_sbase + (size_t)n * R_NG_MI;
                const device float* dbi = down_b4 + d_sbase + (size_t)n * R_NG_MI;
                float acc = 0.0f;
                for (uint g = 0; g < R_NG_MI; ++g) {
                    float sc = dsc[g];
                    float bi = dbi[g];
                    uint kbase = g * R_WPG * 8u;
                    uint wbase = g * R_WPG;
                    for (uint wi = 0; wi < R_WPG; ++wi) {
                        uint w = dw[wbase + wi];
                        uint k = kbase + wi * 8u;
                        for (uint t = 0; t < 8u; ++t) {
                            uint q = (w >> (4u * t)) & 0xFu;
                            acc += (float(q) * sc + bi) * hs[k + t];
                        }
                    }
                }
                out[out_base + n] = wscale * acc;
            }
        } else {
            // ================= shared expert (affine 8-bit) =================
            for (uint m = lid; m < SHARED_INTER; m += TG) {
                const device uint*  gate_row = gate_w8 + (size_t)m * S_IN_PACKED;
                const device uint*  up_row   = up_w8   + (size_t)m * S_IN_PACKED;
                const device float* gate_sc  = gate_s8 + (size_t)m * NG_IN;
                const device float* gate_bi  = gate_b8 + (size_t)m * NG_IN;
                const device float* up_sc    = up_s8   + (size_t)m * NG_IN;
                const device float* up_bi    = up_b8   + (size_t)m * NG_IN;

                float gacc = 0.0f;
                float uacc = 0.0f;
                for (uint g = 0; g < NG_IN; ++g) {
                    float gsc = gate_sc[g];
                    float gbi = gate_bi[g];
                    float usc = up_sc[g];
                    float ubi = up_bi[g];
                    uint kbase = g * S_WPG * 4u;   // = g * 128
                    uint wbase = g * S_WPG;
                    for (uint wi = 0; wi < S_WPG; ++wi) {
                        uint gw = gate_row[wbase + wi];
                        uint uw = up_row[wbase + wi];
                        uint k = kbase + wi * 4u;
                        for (uint t = 0; t < 4u; ++t) {
                            float xv = xs[k + t];
                            uint gq = (gw >> (8u * t)) & 0xFFu;
                            uint uq = (uw >> (8u * t)) & 0xFFu;
                            gacc += (float(gq) * gsc + gbi) * xv;
                            uacc += (float(uq) * usc + ubi) * xv;
                        }
                    }
                }
                float sig = 1.0f / (1.0f + metal::precise::exp(-gacc));
                hs[m] = gacc * sig * uacc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint n = lid; n < HIDDEN; n += TG) {
                const device uint*  dw  = down_w8 + (size_t)n * S_SI_PACKED;
                const device float* dsc = down_s8 + (size_t)n * S_NG_SI;
                const device float* dbi = down_b8 + (size_t)n * S_NG_SI;
                float acc = 0.0f;
                for (uint g = 0; g < S_NG_SI; ++g) {
                    float sc = dsc[g];
                    float bi = dbi[g];
                    uint kbase = g * S_WPG * 4u;
                    uint wbase = g * S_WPG;
                    for (uint wi = 0; wi < S_WPG; ++wi) {
                        uint w = dw[wbase + wi];
                        uint k = kbase + wi * 4u;
                        for (uint t = 0; t < 4u; ++t) {
                            uint q = (w >> (8u * t)) & 0xFFu;
                            acc += (float(q) * sc + bi) * hs[k + t];
                        }
                    }
                }
                out[out_base + n] = acc;   // shared term un-scaled
            }
        }
    """

    return mx.fast.metal_kernel(
        name=(
            f"mtplx_laguna_moe_merged_h{hidden}_mi{moe_inter}"
            f"_si{shared_inter}_k{top_k}_t{threads}"
        ),
        input_names=[
            "x", "indices", "weights",
            "gate_w4", "gate_s4", "gate_b4",
            "up_w4", "up_s4", "up_b4",
            "down_w4", "down_s4", "down_b4",
            "gate_w8", "gate_s8", "gate_b8",
            "up_w8", "up_s8", "up_b8",
            "down_w8", "down_s8", "down_b8",
        ],
        output_names=["out"],
        header=header,
        source=source,
    )


def merged_swiglu_qmv(
    x: mx.array,
    indices: mx.array,
    weights: mx.array,
    gate_w4: mx.array, gate_s4: mx.array, gate_b4: mx.array,
    up_w4: mx.array, up_s4: mx.array, up_b4: mx.array,
    down_w4: mx.array, down_s4: mx.array, down_b4: mx.array,
    gate_w8: mx.array, gate_s8: mx.array, gate_b8: mx.array,
    up_w8: mx.array, up_s8: mx.array, up_b8: mx.array,
    down_w8: mx.array, down_s8: mx.array, down_b8: mx.array,
    *,
    hidden: int,
    moe_intermediate: int,
    shared_intermediate: int,
    threads: int = 256,
) -> mx.array:
    """Merged per-slot SwiGLU output ``[rows, top_k + 1, hidden]`` (float32).

    Slots ``0..top_k-1`` hold ``weights[:, slot] * routed_expert(x)``; slot
    ``top_k`` holds the un-scaled shared expert.  ``out.sum(axis=1)`` is the
    combined MoE output.  Eligibility is the caller's responsibility here; use
    :func:`merged_expert_swiglu` for the guarded drop-in.
    """

    rows = int(x.shape[0])
    top_k = int(indices.shape[1])
    slots = top_k + 1
    idx_u = indices if indices.dtype == mx.uint32 else indices.astype(mx.uint32)
    w_f = weights if weights.dtype == mx.float32 else weights.astype(mx.float32)

    threads = int(threads)
    if threads <= 0 or threads > 1024:
        threads = 256

    kernel = _merged_swiglu_kernel(
        hidden, moe_intermediate, shared_intermediate, top_k, threads
    )
    groups = rows * slots
    (out,) = kernel(
        inputs=[
            x, idx_u, w_f,
            gate_w4, gate_s4, gate_b4,
            up_w4, up_s4, up_b4,
            down_w4, down_s4, down_b4,
            gate_w8, gate_s8, gate_b8,
            up_w8, up_s8, up_b8,
            down_w8, down_s8, down_b8,
        ],
        template=[("T", x.dtype)],
        grid=(threads * groups, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, slots, hidden)],
        output_dtypes=[mx.float32],
    )
    # Fake-speedup guard: exactly one row per (token, slot), hidden wide.
    assert tuple(out.shape) == (rows, slots, hidden), (
        f"merged_swiglu_qmv produced {tuple(out.shape)}, expected {(rows, slots, hidden)}"
    )
    return out


def merged_swiglu_reference(
    x: mx.array,
    indices: mx.array,
    weights: mx.array,
    gate_w4: mx.array, gate_s4: mx.array, gate_b4: mx.array,
    up_w4: mx.array, up_s4: mx.array, up_b4: mx.array,
    down_w4: mx.array, down_s4: mx.array, down_b4: mx.array,
    gate_w8: mx.array, gate_s8: mx.array, gate_b8: mx.array,
    up_w8: mx.array, up_s8: mx.array, up_b8: mx.array,
    down_w8: mx.array, down_s8: mx.array, down_b8: mx.array,
    *,
    hidden: int,
    moe_intermediate: int,
    shared_intermediate: int,
):
    """Pure-mx numeric reference for the merged kernel (no metal_kernel).

    Returns ``(slots, combined)`` where ``slots`` is ``[rows, top_k+1, hidden]``
    float32 (matching :func:`merged_swiglu_qmv`'s pre-scaled per-slot output) and
    ``combined`` is ``slots.sum(axis=1)`` == the combined MoE output.  Built from
    ``mx.dequantize`` + float32 matmuls so the check can prove
    ``kernel == reference == stock`` independently of the kernel's own path.
    """

    xf = x.astype(mx.float32)
    wf = weights.astype(mx.float32)
    rows, top_k = int(indices.shape[0]), int(indices.shape[1])

    def _deq4(qw, qs, qb):
        return mx.dequantize(
            qw, qs, qb, group_size=_GROUP_SIZE, bits=_BITS4, mode="affine"
        )

    # Per slot, gather the SELECTED experts' quantized rows and dequantize only
    # those (never the whole 256-expert bank) so the reference stays light even
    # at the real expert count. Routed banks: gate/up [E, moe_inter, hidden],
    # down [E, hidden, moe_inter].
    slot_outs = []
    for s in range(top_k):
        e = indices[:, s]                              # [rows]
        ge = _deq4(gate_w4[e], gate_s4[e], gate_b4[e]) # [rows, moe_inter, hidden]
        ue = _deq4(up_w4[e], up_s4[e], up_b4[e])
        de = _deq4(down_w4[e], down_s4[e], down_b4[e]) # [rows, hidden, moe_inter]
        gv = mx.matmul(ge, xf[:, :, None])[..., 0]     # [rows, moe_inter]
        uv = mx.matmul(ue, xf[:, :, None])[..., 0]
        hv = (gv * mx.sigmoid(gv)) * uv                # [rows, moe_inter]
        ov = mx.matmul(de, hv[:, :, None])[..., 0]     # [rows, hidden]
        slot_outs.append(ov * wf[:, s][:, None])       # pre-scaled by weight

    # Shared bank dequantized to float32.
    gWs = mx.dequantize(gate_w8, gate_s8, gate_b8, group_size=_GROUP_SIZE, bits=_BITS8, mode="affine")
    uWs = mx.dequantize(up_w8, up_s8, up_b8, group_size=_GROUP_SIZE, bits=_BITS8, mode="affine")
    dWs = mx.dequantize(down_w8, down_s8, down_b8, group_size=_GROUP_SIZE, bits=_BITS8, mode="affine")
    gs = xf @ gWs.T
    us = xf @ uWs.T
    hsh = (gs * mx.sigmoid(gs)) * us
    shared_out = hsh @ dWs.T                            # [rows, hidden]
    slot_outs.append(shared_out)                        # un-scaled shared slot

    slots = mx.stack(slot_outs, axis=1)                 # [rows, top_k+1, hidden]
    combined = slots.sum(axis=1)                        # [rows, hidden]
    assert tuple(slots.shape) == (rows, top_k + 1, hidden)
    return slots, combined


def merged_expert_swiglu(
    switch_mlp, shared_mlp, x: mx.array, indices: mx.array, weights: mx.array
) -> mx.array:
    """Drop-in for the routed+shared combine on the S-2.1 MoE path.

    Replaces::

        output = switch_mlp(x, indices)
        output = (output * weights[..., None]).sum(-2) + shared_mlp(x)

    Returns the combined MoE output ``[rows, hidden]``.  Falls back to that exact
    stock combine on any shape/dtype/quant the fused kernel does not cover, so it
    can be switched on without owning a correctness branch.
    """

    if not is_merged_swiglu_eligible(switch_mlp, shared_mlp, x, indices, weights):
        routed = switch_mlp(x, indices)
        combined = (routed * weights.astype(routed.dtype)[..., None]).sum(axis=-2)
        return combined + shared_mlp(x)

    gate4, up4, down4 = switch_mlp.gate_proj, switch_mlp.up_proj, switch_mlp.down_proj
    gate8, up8, down8 = shared_mlp.gate_proj, shared_mlp.up_proj, shared_mlp.down_proj
    hidden = int(x.shape[-1])
    moe_inter = int(gate4["weight"].shape[1])
    shared_inter = int(gate8["weight"].shape[0])
    slots = merged_swiglu_qmv(
        x, indices, weights,
        gate4["weight"], gate4["scales"], gate4["biases"],
        up4["weight"], up4["scales"], up4["biases"],
        down4["weight"], down4["scales"], down4["biases"],
        gate8["weight"], gate8["scales"], gate8["biases"],
        up8["weight"], up8["scales"], up8["biases"],
        down8["weight"], down8["scales"], down8["biases"],
        hidden=hidden,
        moe_intermediate=moe_inter,
        shared_intermediate=shared_inter,
    )
    return slots.sum(axis=1)
