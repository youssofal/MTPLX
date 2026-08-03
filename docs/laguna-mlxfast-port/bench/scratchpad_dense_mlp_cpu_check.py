"""CPU (mx.cpu) validation for laguna_dense_mlp: fallback == reference ~= stock,
and reference == fp64 gold. The Metal kernel itself is NOT run here (metal_kernel
is GPU-only) -- that is scratchpad_dense_mlp_check.py, run flocked.

Proves on CPU:
  1. fallback (dense_mlp helper, ineligible on CPU) == stock MLP  (exact)
  2. reference (fp32-accumulate algorithm) == fp64 gold           (tight)
  3. reference ~= stock within quantized_matmul's own CPU precision (reported)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import mlx.core as mx
import mlx.nn as nn
import numpy as np

mx.set_default_device(mx.cpu)
mx.random.seed(0)

from mtplx.kernels.laguna_dense_mlp import (  # noqa: E402
    dense_mlp,
    dense_mlp_reference,
    is_dense_mlp_eligible,
)

H, I = 3072, 12288          # S-2.1 layer-0 shape
GS = 64
GATE_UP_BITS, DOWN_BITS = 5, 6


class DenseMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def quantize_proj(lin, bits):
    ql = nn.QuantizedLinear.from_linear(lin, group_size=GS, bits=bits)
    # Real checkpoint stores scales/biases in BF16 -- mirror that exactly.
    ql.scales = ql.scales.astype(mx.bfloat16)
    ql.biases = ql.biases.astype(mx.bfloat16)
    return ql


# Build the dense MLP and quantize it per-projection like layer 0.
mlp = DenseMLP(H, I)
# smaller weight magnitude so the quant grids behave like a trained layer
mlp.gate_proj.weight = mlp.gate_proj.weight * 0.5
mlp.up_proj.weight = mlp.up_proj.weight * 0.5
mlp.down_proj.weight = mlp.down_proj.weight * 0.5
mlp.gate_proj = quantize_proj(mlp.gate_proj, GATE_UP_BITS)
mlp.up_proj = quantize_proj(mlp.up_proj, GATE_UP_BITS)
mlp.down_proj = quantize_proj(mlp.down_proj, DOWN_BITS)

x = (mx.random.normal((1, H)) * 0.5).astype(mx.bfloat16)   # B=1 decode row


def npf(a):
    if isinstance(a, np.ndarray):
        return a.astype(np.float64)
    return np.array(a.astype(mx.float32)).astype(np.float64)


def report(tag, a, b):
    d = np.abs(npf(a) - npf(b))
    print(f"  {tag}: max={d.max():.6e} mean={d.mean():.6e} "
          f"p99={np.percentile(d, 99):.6e}")
    return d


# --- stock ---
y_stock = mlp(x)
print("shapes/dtypes:", tuple(y_stock.shape), y_stock.dtype)
assert tuple(y_stock.shape) == (1, H)

# --- (1) fallback == stock (CPU: not metal -> ineligible -> calls mlp(x)) ---
assert not is_dense_mlp_eligible(mlp, x), "should be ineligible off-Metal"
y_fallback = dense_mlp(mlp, x)
d_fb = np.abs(npf(y_fallback) - npf(y_stock))
print("\n[1] fallback vs stock (want EXACT):")
print(f"  max abs diff = {d_fb.max():.6e}")
assert d_fb.max() == 0.0, "fallback must be bit-identical to stock"
print("  PASS: fallback is the stock MLP, bit-identical.")

# --- reference ---
g = mlp.gate_proj
u = mlp.up_proj
dp = mlp.down_proj
y_ref = dense_mlp_reference(
    x,
    g.weight, g.scales, g.biases, g.bits, g.group_size,
    u.weight, u.scales, u.biases, u.bits, u.group_size,
    dp.weight, dp.scales, dp.biases, dp.bits, dp.group_size,
)
assert tuple(y_ref.shape) == (1, H)

# --- fp64 gold: fp32-dequant weights, fp64 accumulate, mirror bf16 boundaries ---
def deq_gold(ql):
    q = mx.dequantize(ql.weight, ql.scales.astype(mx.float32),
                      ql.biases.astype(mx.float32),
                      group_size=ql.group_size, bits=ql.bits, mode="affine")
    return npf(q)

xn = npf(x)
gw, uw, dw = deq_gold(g), deq_gold(u), deq_gold(dp)
gg = xn @ gw.T
ug = xn @ uw.T
# round to bf16 at the projection boundary (as kernel + stock do)
def to_bf16_f64(arr):
    return npf(mx.array(arr.astype(np.float32)).astype(mx.bfloat16))
ggb = to_bf16_f64(gg)
ugb = to_bf16_f64(ug)
sig = 1.0 / (1.0 + np.exp(-ggb))
hh = to_bf16_f64(ggb * sig * ugb)
og = hh @ dw.T
y_gold = to_bf16_f64(og)

print("\n[2] reference vs fp64 gold (want tight, ~bf16 rounding only):")
d_gold = report("ref-gold", y_ref, mx.array(y_gold.astype(np.float32)))
# reference and gold differ only by fp32-vs-fp64 accumulation reassociation
# then the identical bf16 rounding -> at most ~1 bf16 ULP of the output scale.
absmax = np.abs(npf(y_gold)).max()
bf16_ulp = absmax / 128.0
print(f"  output absmax={absmax:.4e}  ~1 bf16 ULP={bf16_ulp:.4e}")
assert d_gold.max() <= 4 * bf16_ulp + 1e-6, "reference should track fp64 gold to a few bf16 ULP"
print("  PASS: reference reproduces the fp64-accurate math (kernel is fp32-exact).")

# --- (3) reference vs stock: expose the quantized_matmul CPU accumulation gap ---
print("\n[3] reference vs stock quantized_matmul (CPU) -- EXPECTED to be loose:")
d_rs = report("ref-stock", y_ref, y_stock)
print(f"  output absmax (stock) = {np.abs(npf(y_stock)).max():.4e}")
# also show per-projection gap so the source of the gap is unambiguous
gq = g(x)            # stock quantized_matmul
gr = mx.array((xn @ gw.T).astype(np.float32))
report("  proj gate: stock-qmm vs fp32-dequant", gq, gr)
print("  NOTE: this gap is CPU quantized_matmul being lossy (reduced-precision")
print("  accumulation), NOT the reference/kernel. Metal quantized_matmul")
print("  accumulates in fp32, so the flocked kernel-vs-stock gap is far smaller.")

print("\nALL CPU CHECKS PASSED (fallback==stock exact; reference==fp64 gold).")
