"""REAL-SHAPE GPU check for P5: prefill MoE combine tail.

RUN UNDER THE GPU FLOCK (every Metal exec must hold it).  Do not run un-flocked.

Compares the fused metal kernel against the stock op chain
((expert_out * (w*2.5).astype(bf16)[...,None]).sum(-2) + shared + residual) at
the real S-2.1 shape (M=T=1024, top_k=10, hidden=3072), then times both.

Reports:
  * max abs diff and bit-exact fraction vs stock (expect bit-exact: the kernel
    reproduces col_reduce_small's TY=min(8,K) accumulation order and the two
    trailing BF16 adds);
  * kernel vs stock timing (queued and per-call-synchronized lanes).
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import mlx.core as mx
from mtplx.kernels.laguna_prefill_moe_combine import (
    fused_moe_combine_prefill,
    is_moe_combine_prefill_eligible,
)

assert mx.metal.is_available(), "no Metal device"
assert mx.default_device() == mx.gpu, "run on GPU (do not set cpu)"
bf16, f32 = mx.bfloat16, mx.float32
H, K = 3072, 10
SCALING = 2.5
ITERS = 30
FAIL = []


def report(tag, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + tag + ("  " + extra if extra else ""))
    if not ok:
        FAIL.append(tag)


def time_call(fn, iters=ITERS, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    mx.eval(outs)
    queued = (time.perf_counter() - t0) / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    percall = (time.perf_counter() - t0) / iters
    return queued * 1e3, percall * 1e3


def stock_combine(expert_out, weights, shared, residual, scaling):
    w = (weights * scaling).astype(expert_out.dtype)
    combined = (expert_out * w[..., None]).sum(axis=-2)
    return (combined + shared) + residual


mx.random.seed(2)
for M in (1024,):
    print(f"\n=== P5 M={M} top_k={K} hidden={H} routed_scaling={SCALING} ===")
    eo = (mx.random.normal((M, K, H)) * 0.3).astype(bf16)
    w = mx.sigmoid(mx.random.normal((M, K))).astype(f32)
    w = w / w.sum(axis=-1, keepdims=True)  # normalized, unscaled (P3 output)
    shared = (mx.random.normal((M, H)) * 0.3).astype(bf16)
    resid = (mx.random.normal((M, H)) * 0.5).astype(bf16)
    mx.eval(eo, w, shared, resid)

    report(f"P5 M={M} kernel eligible",
           is_moe_combine_prefill_eligible(eo, w, shared, resid))

    kc = fused_moe_combine_prefill(eo, w, shared, resid, SCALING)
    sc = stock_combine(eo, w, shared, resid, SCALING)
    mx.eval(kc, sc)
    d = mx.abs(kc.astype(f32) - sc.astype(f32))
    maxd = d.max().item()
    exact = mx.mean((kc == sc).astype(f32)).item()
    report(f"P5 M={M} kernel==stock", maxd <= 1.6e-2 and exact >= 0.999,
           f"maxabs={maxd:.3e} bitexact={exact:.5f}")
    report(f"P5 M={M} output shape [{M},{H}]", tuple(kc.shape) == (M, H))

    kc_ms = time_call(lambda: fused_moe_combine_prefill(eo, w, shared, resid, SCALING))
    st_ms = time_call(lambda: stock_combine(eo, w, shared, resid, SCALING))
    print(f"  timing  kernel queued={kc_ms[0]:.3f}ms percall={kc_ms[1]:.3f}ms")
    print(f"  timing  stock  queued={st_ms[0]:.3f}ms percall={st_ms[1]:.3f}ms")
    print(f"  speedup(queued)={st_ms[0]/kc_ms[0]:.3f}x  speedup(percall)={st_ms[1]/kc_ms[1]:.3f}x")

print("\n" + ("P5 ALL CHECKS PASSED" if not FAIL else f"P5 FAILURES: {FAIL}"))
