"""Real-shape GPU check + timing for laguna_qkvg_fused (RUN UNDER THE FLOCK).

This is the ONLY script that dispatches the metal_kernel.  It:

  1. Builds real S-2.1 attention affine weights (exact shapes; both a
     full-attention layer -- 48 heads, 8-bit -- and a sliding layer -- 72
     heads, 5-bit -- at group_size 64), quantized from synthetic bf16 with the
     real mx.quantize format.
  2. Calls the fused kernel and the stock rms_norm + 4x quantized_matmul chain,
     asserts output shapes and allclose (fused vs stock, plus the tighter fused
     vs FP32 reference).
  3. Times both on the QUEUED lane (many dispatches, one sync -- the lane that
     matters for a B=1 decode micro-kernel; see the queued-vs-eager microbench
     note) and prints per-call ms + speedup.  An eager per-call-sync lane is
     also printed for context.

Run under the GPU flock, e.g.:
    <flock wrapper> .venv/bin/python scratchpad_qkvg_check.py

Do NOT run two model-holding GPU jobs at once; this one is small (weights well
under 1 GiB) but still takes the GPU.
"""

from __future__ import annotations

import time

import numpy as np
import mlx.core as mx

from mtplx.kernels.laguna_qkvg_fused import (
    QKVGSpec,
    fused_input_norm_qkvg,
    fused_input_norm_qkvg_reference,
    is_qkvg_fused_eligible,
    _stock_qkvg,
)

HIDDEN = 3072
KV_HEADS = 8
HEAD_DIM = 128
GS = 64
EPS = 1e-6
RNG = np.random.default_rng(0)

WARMUP = 10
ITERS = 200


def npf(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def bf16(shape, scale=1.0, center=0.0):
    a = RNG.standard_normal(shape).astype(np.float32) * scale + center
    return mx.array(a).astype(mx.bfloat16)


def quantize_bf16(rows, bits):
    return mx.quantize(bf16((rows, HIDDEN), scale=0.05), group_size=GS, bits=bits)


def build_layer(n_heads, bits):
    spec = QKVGSpec(n_heads=n_heads, bits=bits)
    hidden = bf16((1, 1, HIDDEN), scale=0.8)
    norm_weight = bf16((HIDDEN,), scale=0.1, center=1.0)
    qb = quantize_bf16(spec.query_rows, bits)
    kb = quantize_bf16(spec.kv_rows, bits)
    vb = quantize_bf16(spec.kv_rows, bits)
    gb = quantize_bf16(spec.gate_rows, bits)
    banks = (qb[0], qb[1], qb[2], kb[0], kb[1], kb[2],
             vb[0], vb[1], vb[2], gb[0], gb[1], gb[2])
    return spec, hidden, norm_weight, banks


def maxabs(a, b):
    return float(np.max(np.abs(npf(a).astype(np.float64) - npf(b).astype(np.float64))))


def correctness(spec, hidden, norm_weight, banks):
    assert is_qkvg_fused_eligible(hidden, norm_weight, *banks, spec), \
        "kernel should be eligible for the real decode shape on GPU"
    fused = fused_input_norm_qkvg(hidden, norm_weight, *banks, EPS, spec)
    stock = _stock_qkvg(hidden, norm_weight, *banks, EPS, spec)
    ref = fused_input_norm_qkvg_reference(hidden, norm_weight, *banks, EPS, spec)
    mx.eval(fused, stock, ref)

    exp = [(1, 1, spec.query_rows), (1, 1, spec.kv_rows),
           (1, 1, spec.kv_rows), (1, 1, spec.gate_rows)]
    for nm, f, e in zip(("q", "k", "v", "g"), fused, exp):
        assert tuple(f.shape) == e, f"{nm}: {f.shape} != {e}"
        assert f.dtype == mx.bfloat16
    print(f"  shapes {exp}: OK")

    ok = True
    for nm, f, s, r in zip(("q", "k", "v", "g"), fused, stock, ref):
        d_stock = maxabs(f, s)
        d_ref = maxabs(f, r)
        rng = float(np.max(np.abs(npf(s))))
        cs = bool(mx.allclose(f, s, rtol=2e-2, atol=2e-2))
        cr = bool(mx.allclose(f, r, rtol=1e-2, atol=1e-2))
        ok = ok and cs and cr
        print(f"  {nm}: vs stock max={d_stock:.3e} allclose(2e-2)={cs} | "
              f"vs ref max={d_ref:.3e} allclose(1e-2)={cr} | range={rng:.2f}")
    assert ok, "allclose FAILED -- inspect the per-projection diffs above"
    print("  allclose: OK")


def _distinct_inputs(hidden, n):
    """n distinct hidden rows (defeats common-subexpression elimination)."""
    xs = [hidden + bf16((1, 1, HIDDEN), scale=0.001) for _ in range(n)]
    mx.eval(xs)
    return xs


def time_queued(fn, hidden, rest, n):
    xs = _distinct_inputs(hidden, n)
    for i in range(WARMUP):
        mx.eval(fn(xs[i % n], *rest))
    mx.synchronize()
    t0 = time.perf_counter()
    outs = []
    for i in range(n):
        outs.extend(fn(xs[i], *rest))  # enqueue only
    mx.eval(outs)  # single sync for the whole batch -> queued lane
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3  # ms/call


def time_eager(fn, hidden, rest, n):
    xs = _distinct_inputs(hidden, n)
    for i in range(WARMUP):
        mx.eval(fn(xs[i % n], *rest))
    mx.synchronize()
    t0 = time.perf_counter()
    for i in range(n):
        mx.eval(fn(xs[i], *rest))  # per-call sync
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def bench(name, spec, hidden, norm_weight, banks):
    print(f"\n=== {name}: n_heads={spec.n_heads}, bits={spec.bits}, "
          f"rows_per_thread={spec.rows_per_thread} ===")
    correctness(spec, hidden, norm_weight, banks)

    fused_fn = lambda h, *b: fused_input_norm_qkvg(h, norm_weight, *b, EPS, spec)
    stock_fn = lambda h, *b: _stock_qkvg(h, norm_weight, *b, EPS, spec)

    fq = time_queued(fused_fn, hidden, banks, ITERS)
    sq = time_queued(stock_fn, hidden, banks, ITERS)
    fe = time_eager(fused_fn, hidden, banks, ITERS)
    se = time_eager(stock_fn, hidden, banks, ITERS)
    print(f"  queued lane: fused {fq:.4f} ms | stock {sq:.4f} ms | "
          f"speedup {sq / fq:.3f}x   <-- decision lane")
    print(f"  eager  lane: fused {fe:.4f} ms | stock {se:.4f} ms | "
          f"speedup {se / fe:.3f}x")


def main():
    if not mx.metal.is_available():
        raise SystemExit("Metal not available -- run this on the GPU box under the flock.")
    print(f"mlx {mx.__version__}  device={mx.default_device()}  "
          f"warmup={WARMUP} iters={ITERS}")
    bench("full_attention layer", *build_layer(n_heads=48, bits=8))
    bench("sliding_attention layer", *build_layer(n_heads=72, bits=5))
    print("\nDONE")


if __name__ == "__main__":
    main()
