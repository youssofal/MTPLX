"""REAL-SHAPE GPU check for P1: prefill q/k RMSNorm + rope (both families).

RUN UNDER THE GPU FLOCK (every Metal exec must hold it).  Do not run un-flocked.

Compares the fused metal kernel against the stock op chain (mx.fast.rms_norm +
mx.fast.rope) at the real S-2.1 prefill shape (batch 1, ctx T=1024) for both the
FULL/YaRN (48 q heads, rot 64, theta 500000, mscale 1.4852...) and SLIDING/base
(72 q heads, rot 128, theta 10000) attention families, then times both.

Reports, per family, per tensor (q, k):
  * max abs diff and bit-exact fraction vs stock (expect near-bit-exact: the
    kernel uses metal::fast::cos/sin, the same path mx.fast.rope takes on GPU);
  * all-rows-distinct (guards the batched-rope T=1 broadcast trap);
  * kernel vs stock timing (queued and per-call-synchronized lanes).
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import mlx.core as mx
from mlx_lm.models.rope_utils import initialize_rope
from mtplx.kernels.laguna_prefill_qk_rope import (
    QkRopePrefillSpec,
    fused_qk_norm_rope_prefill,
    _stock_qk_norm_rope_prefill,
    is_qk_norm_rope_prefill_eligible,
)

assert mx.metal.is_available(), "no Metal device"
assert mx.default_device() == mx.gpu, "run on GPU (do not set cpu)"
bf16, f32 = mx.bfloat16, mx.float32
EPS = 1e-6
T = 1024
ITERS = 30
FAIL = []


def report(tag, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + tag + ("  " + extra if extra else ""))
    if not ok:
        FAIL.append(tag)


def time_call(fn, iters=ITERS, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    # queued lane: enqueue all, one sync (the lane that predicts in a chain).
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    mx.eval(outs)
    queued = (time.perf_counter() - t0) / iters
    # per-call synchronized lane.
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    percall = (time.perf_counter() - t0) / iters
    return queued * 1e3, percall * 1e3


full = initialize_rope(
    64, base=500000.0, traditional=False,
    scaling_config={"rope_type": "yarn", "factor": 128.0,
                    "original_max_position_embeddings": 8192,
                    "beta_fast": 32.0, "beta_slow": 1.0},
    max_position_embeddings=1048576,
)
FULL_SPEC = QkRopePrefillSpec(48, 8, 128, 64, full._freqs, None, full.mscale)
SLID_SPEC = QkRopePrefillSpec(72, 8, 128, 128, None, 10000.0, None)

mx.random.seed(0)
for name, spec in (("FULL/yarn", FULL_SPEC), ("SLIDING/base", SLID_SPEC)):
    print(f"\n=== P1 {name} : B=1 T={T} q_heads={spec.n_q_heads} ===")
    q = (mx.random.normal((1, T, spec.n_q_heads * 128)) * 0.5).astype(bf16)
    k = (mx.random.normal((1, T, spec.n_kv_heads * 128)) * 0.5).astype(bf16)
    qw = (mx.random.normal((128,)) * 0.3 + 1.0).astype(bf16)
    kw = (mx.random.normal((128,)) * 0.3 + 1.0).astype(bf16)
    mx.eval(q, k, qw, kw)

    report(f"P1 {name} kernel eligible", is_qk_norm_rope_prefill_eligible(q, k, qw, kw, spec))

    for offset in (0, 400):
        kq, kk = fused_qk_norm_rope_prefill(q, k, qw, kw, EPS, offset, spec)
        sq, sk = _stock_qk_norm_rope_prefill(q, k, qw, kw, EPS, offset, spec)
        mx.eval(kq, kk, sq, sk)
        for tag2, kn, st in (("q", kq, sq), ("k", kk, sk)):
            d = mx.abs(kn.astype(f32) - st.astype(f32))
            maxd = d.max().item()
            exact = mx.mean((kn == st).astype(f32)).item()
            # kernel mirrors mx.fast on GPU -> expect bit-exact (exact ~1.0).
            # Gate tolerates rare 1-ulp rounding (~0.08 at magnitude 10) while a
            # real layout/offset/mscale bug produces gross diffs and drops exact.
            ok = maxd <= 8e-2 and exact >= 0.97
            report(f"P1 {name} {tag2} off={offset} kernel==stock", ok,
                   f"maxabs={maxd:.3e} bitexact={exact:.4f}")
        # all-rows-distinct on the kernel output
        row0 = kq[0, 0, 0]
        distinct = all(
            not mx.allclose(kq[0, 0, ti], row0, atol=1e-3).item()
            for ti in range(1, T, 97)
        )
        report(f"P1 {name} off={offset} kernel all-rows-distinct", distinct)

    kq_ms = time_call(lambda: fused_qk_norm_rope_prefill(q, k, qw, kw, EPS, 0, spec))
    st_ms = time_call(lambda: _stock_qk_norm_rope_prefill(q, k, qw, kw, EPS, 0, spec))
    print(f"  timing  kernel queued={kq_ms[0]:.3f}ms percall={kq_ms[1]:.3f}ms")
    print(f"  timing  stock  queued={st_ms[0]:.3f}ms percall={st_ms[1]:.3f}ms")
    print(f"  speedup(queued)={st_ms[0]/kq_ms[0]:.3f}x  speedup(percall)={st_ms[1]/kq_ms[1]:.3f}x")

print("\n" + ("P1 ALL CHECKS PASSED" if not FAIL else f"P1 FAILURES: {FAIL}"))
