"""FLOCKED real-shape check for laguna_dense_mlp (D11) -- RUN UNDER THE GPU LOCK.

This runs the Metal kernel (mx.fast.metal_kernel is GPU-only), so it must NOT be
run casually -- take the GPU flock first. It:

  1. Builds a dense MLP at the exact S-2.1 layer-0 shape/quant
     (hidden 3072, intermediate 12288; gate/up affine 5-bit gs64,
      down affine 6-bit gs64, bf16 scales/biases), or loads the REAL
      layer-0 weights with --real.
  2. Asserts the fused path is eligible on Metal and checks:
       fused kernel  vs  pure-mx reference   (tight: same fp32-accumulate math)
       fused kernel  vs  stock MLP           (reported: quantized_matmul gap)
  3. Times fused kernel vs stock MLP at B=1 on the queued lane
     (per the repo's "queued vs eager microbench" note: decide µs-kernel
      promotions on the queued lane, not the eager/host-synced one).

Expectations (honest): this is a hand affine dequant-QMV; the repo's
"Metal sub-4-bit is ALU-bound" / "IQ2_XXS kernel loses to stock" findings say
such kernels are ALU/occupancy-bound and can LOSE to mx.quantized_matmul. Layer
0 is 1 of 48 layers (~2% of decode), so even a win is a ~2%-scale lever. The
kernel is fp32-accurate (more accurate than stock qmm) and therefore NUMERICALLY
DIFFERENT from stock; the challenge's real bar is exact-token teacher-forced
match, which THIS SCRIPT DOES NOT TEST -- verify that in the correctness gate.

Usage:
    python scratchpad_dense_mlp_check.py            # synthetic weights, exact shape
    python scratchpad_dense_mlp_check.py --real     # real layer-0 weights (loads shard 1)
    python scratchpad_dense_mlp_check.py --iters 500
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import mlx.core as mx
import mlx.nn as nn

from mtplx.kernels.laguna_dense_mlp import (
    dense_mlp,
    dense_mlp_reference,
    is_dense_mlp_eligible,
)

H, I = 3072, 12288
GS = 64
GATE_UP_BITS, DOWN_BITS = 5, 6
MODEL_DIR = os.environ.get(
    "MTPLX_LAGUNA_MODEL_DIR",
    str(Path.home() / ".mtplx/models/mlx-community--Laguna-S-2.1-oQ4e"),
)
SHARD1 = MODEL_DIR + "/model-00001-of-00013.safetensors"


class DenseMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def _assign_quant(ql, weight, scales, biases, bits):
    ql.weight, ql.scales, ql.biases = weight, scales, biases
    ql.bits, ql.group_size, ql.mode = bits, GS, "affine"
    return ql


def build_synthetic():
    mx.random.seed(0)
    mlp = DenseMLP(H, I)
    mlp.gate_proj.weight = mlp.gate_proj.weight * 0.5
    mlp.up_proj.weight = mlp.up_proj.weight * 0.5
    mlp.down_proj.weight = mlp.down_proj.weight * 0.5
    for name, bits in (("gate_proj", GATE_UP_BITS), ("up_proj", GATE_UP_BITS), ("down_proj", DOWN_BITS)):
        lin = getattr(mlp, name)
        ql = nn.QuantizedLinear.from_linear(lin, group_size=GS, bits=bits)
        ql.scales = ql.scales.astype(mx.bfloat16)
        ql.biases = ql.biases.astype(mx.bfloat16)
        setattr(mlp, name, ql)
    return mlp


def build_real():
    pre = "language_model.model.layers.0.mlp."
    w = mx.load(SHARD1)
    mlp = DenseMLP(H, I)
    for name, bits in (("gate_proj", GATE_UP_BITS), ("up_proj", GATE_UP_BITS), ("down_proj", DOWN_BITS)):
        in_dim, out_dim = (H, I) if name != "down_proj" else (I, H)
        ql = nn.QuantizedLinear(in_dim, out_dim, bias=False, group_size=GS, bits=bits)
        _assign_quant(
            ql,
            w[pre + name + ".weight"],
            w[pre + name + ".scales"],
            w[pre + name + ".biases"],
            bits,
        )
        setattr(mlp, name, ql)
    return mlp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="use real layer-0 weights (loads shard 1)")
    ap.add_argument("--iters", type=int, default=300)
    args = ap.parse_args()

    if not mx.metal.is_available():
        print("Metal not available; this script must run on the GPU box under the flock.")
        sys.exit(1)
    mx.set_default_device(mx.gpu)
    print("device:", mx.default_device())

    mlp = build_real() if args.real else build_synthetic()
    mx.eval(mlp.parameters())
    x = (mx.random.normal((1, H)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)

    assert is_dense_mlp_eligible(mlp, x), "fused path should be eligible at layer-0 shape on Metal"

    g, u, d = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    y_fused = dense_mlp(mlp, x)
    y_ref = dense_mlp_reference(
        x,
        g.weight, g.scales, g.biases, g.bits, g.group_size,
        u.weight, u.scales, u.biases, u.bits, u.group_size,
        d.weight, d.scales, d.biases, d.bits, d.group_size,
    )
    y_stock = mlp(x)
    mx.eval(y_fused, y_ref, y_stock)

    def gap(a, b):
        d_ = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
        return float(mx.max(d_)), float(mx.mean(d_))

    assert tuple(y_fused.shape) == (1, H), tuple(y_fused.shape)
    absmax = float(mx.max(mx.abs(y_stock.astype(mx.float32))))
    print(f"\noutput absmax (stock) = {absmax:.4e}")
    mx_ref = gap(y_fused, y_ref)
    mx_st = gap(y_fused, y_stock)
    print(f"fused vs reference : max={mx_ref[0]:.3e} mean={mx_ref[1]:.3e}  "
          f"(want tight: same fp32-accumulate math)")
    print(f"fused vs stock     : max={mx_st[0]:.3e} mean={mx_st[1]:.3e}  "
          f"(qmm gap; Metal qmm is fp32 so much tighter than CPU's ~1%)")
    # allclose verdicts (advisory, not the token gate)
    ref_ok = mx.allclose(y_fused.astype(mx.float32), y_ref.astype(mx.float32),
                         atol=max(1e-4, absmax * 1e-2), rtol=1e-2).item()
    print(f"allclose(fused, reference, atol~1%output): {ref_ok}")

    # --- timing: queued lane ---
    def timed(fn, iters):
        for _ in range(5):          # warmup
            mx.eval(fn())
        mx.synchronize()
        t0 = time.perf_counter()
        outs = [fn() for _ in range(iters)]   # enqueue all, single sync (queued lane)
        mx.eval(outs)
        mx.synchronize()
        return (time.perf_counter() - t0) / iters * 1e3   # ms/call

    ms_stock = timed(lambda: mlp(x), args.iters)
    ms_fused = timed(lambda: dense_mlp(mlp, x), args.iters)
    print(f"\nqueued-lane timing over {args.iters} iters:")
    print(f"  stock MLP      : {ms_stock:.4f} ms/tok")
    print(f"  fused kernel   : {ms_fused:.4f} ms/tok")
    print(f"  speedup (stock/fused) = {ms_stock / ms_fused:.3f}x")
    print("\nReminder: layer 0 is 1 of 48 layers (~2% of decode). A win here is")
    print("bounded by that share, and the token-exact gate is NOT tested here.")


if __name__ == "__main__":
    main()
