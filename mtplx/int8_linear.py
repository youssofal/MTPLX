"""Native int8 non-expert path for Escha-W2 (and any per-out-channel-symmetric int8 export).

The checkpoint stores non-expert weights as ``weight_int8 [out,in]`` (int8) + ``weight_scale
[out]`` (per-output-channel, symmetric). Dequantizing to bf16 at load DOUBLES resident bytes
and makes every decode gemv read 2 bytes/weight for zero quality gain. Keep them int8 and do a
FUSED int8 matvec: read the int8 weight (1 byte), convert in-register, scale per row.

    y[m, o] = scale[o] * sum_i (w_int8[o, i] * x[m, i])

────────────────────────────────────────────────────────────────────────────────────────────
ATTRIBUTION — borrowed and improved:
  The threadgroup TILING of the matvec kernel below is adapted from the ``fast_qmv`` kernel in
  dusterbloom/higgs (https://github.com/dusterbloom/higgs,
  crates/higgs-models/src/metal_kernel.rs): each simdgroup owns RESULTS_PER_SIMDGROUP output
  rows, the 32 lanes stride the contraction holding x in registers (no threadgroup memory, no
  barriers), and per-row partials are simd_sum-reduced.
  OUR CHANGES on top of that pattern:
    - retargeted from higgs's 1-bit affine packing to a per-output-channel-SYMMETRIC int8
      weight (1 byte/weight, no group biases): the inner product is a plain int8·x, scaled once
      per row — no bit-unpacking, no per-group affine term;
    - the scale is kept in f32 (the prior mlx-lm path lossily cast it to bf16), so this is
      numerically tighter than the bf16 dequant it replaces;
    - the Int8Linear wrapper + loader wiring (decode uses the matvec, prefill dequants
      transiently) are ours.
────────────────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn

_I8: dict = {}
_RPS = 4  # output rows per simdgroup


def _int8_matvec_kernel(IN: int):
    key = ("i8mv", IN)
    if key in _I8:
        return _I8[key]
    src = f"""
        uint tgx = threadgroup_position_in_grid.x;      // row-block
        uint row_of_grid_y = threadgroup_position_in_grid.y;  // token (M)
        uint sg  = simdgroup_index_in_threadgroup;
        uint lid = thread_index_in_simdgroup;
        uint nsg = simdgroups_per_threadgroup;
        int out_row = int(tgx) * (int(nsg) * {_RPS}) + int(sg) * {_RPS};
        uint m = row_of_grid_y;
        float acc[{_RPS}];
        for (int r = 0; r < {_RPS}; ++r) acc[r] = 0.0f;
        // lanes stride the contraction in chunks of 32
        for (uint i = lid; i < {IN}u; i += 32u) {{
            float xv = float(x[m * {IN}u + i]);
            for (int r = 0; r < {_RPS}; ++r) {{
                int row = out_row + r;
                if (row >= n_out) continue;
                acc[r] += xv * float(w[(uint)row * {IN}u + i]);   // w is int8 -> promoted
            }}
        }}
        for (int r = 0; r < {_RPS}; ++r) {{
            int row = out_row + r;
            float v = simd_sum(acc[r]);
            if (lid == 0u && row < n_out)
                y[m * (uint)n_out + (uint)row] = v * float(scale[row]);   // f32 out; caller casts
        }}
    """
    kern = mx.fast.metal_kernel(
        name=f"int8_matvec_{IN}",
        input_names=["x", "w", "scale", "n_out"],
        output_names=["y"],
        source=src,
    )
    _I8[key] = kern
    return kern


def int8_matvec(x: mx.array, w_int8: mx.array, scale: mx.array) -> mx.array:
    """x [M, IN] -> y [M, OUT].  w_int8 [OUT, IN] int8, scale [OUT].  Fused, no bf16 weight.

    The kernel indexes x/w as contiguous row-major and is specialized only on IN, so force a
    stable f32-contiguous x (activation is tiny at decode) — a non-contiguous view or a bf16/f32
    dtype mismatch across call sites would otherwise misread the buffer.
    """
    x = mx.contiguous(x.astype(mx.float32))
    w_int8 = mx.contiguous(w_int8)
    M, IN = x.shape
    OUT = w_int8.shape[0]
    kern = _int8_matvec_kernel(IN)
    nsg = 8
    tg = nsg * 32                                    # threads per threadgroup (8 simdgroups)
    blocks = (OUT + nsg * _RPS - 1) // (nsg * _RPS)  # each threadgroup covers nsg*RPS output rows
    (y,) = kern(
        inputs=[x, w_int8, scale.astype(mx.float32), mx.array(OUT, dtype=mx.int32)],
        output_shapes=[(M, OUT)],
        output_dtypes=[mx.float32],   # accumulate/emit f32; Int8Linear casts the small output
        grid=(blocks * tg, M, 1),     # TOTAL threads in x = blocks threadgroups * tg threads
        threadgroup=(tg, 1, 1),
    )
    return y


class Int8Linear(nn.Module):
    """Drop-in for nn.Linear whose weight stays int8 + per-out-channel scale.

    Decode (small M) uses the fused int8 matvec — no bf16 weight copy, weight-bandwidth-bound.
    Prefill (large M) is compute-bound, where a dense GEMM tiles/reuses the weight far better
    than a per-row matvec, so we transiently dequantize there. The M<=32 gate keeps decode on
    the fused path (the only path decode ever takes) and never dequantizes in the decode loop.
    """
    def __init__(self, w_int8: mx.array, scale: mx.array):
        super().__init__()
        self.weight = w_int8                        # [OUT, IN] int8, RESIDENT
        self.scale = scale.astype(mx.float32)       # [OUT]

    def __call__(self, x: mx.array) -> mx.array:
        lead = x.shape[:-1]
        IN = x.shape[-1]
        M = 1
        for d in lead:
            M *= d
        x2 = x.reshape(M, IN)
        if M <= 32:                                 # decode: fused int8 matvec, no dequant
            y = int8_matvec(x2, self.weight, self.scale).astype(x.dtype)
        else:                                       # prefill: transient dequant -> dense GEMM
            w = self.weight.astype(x.dtype) * self.scale.astype(x.dtype)[:, None]
            y = x2 @ w.T
        return y.reshape(*lead, -1)
