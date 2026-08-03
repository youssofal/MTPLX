"""REAL-SHAPE GPU check for P3: prefill MoE router (sigmoid+bias+top-10).

RUN UNDER THE GPU FLOCK (every Metal exec must hold it).  Do not run un-flocked.

Compares the fused metal kernel against the stock op chain
(sigmoid -> +bias -> argpartition top-10 -> gather -> normalize) at the real
S-2.1 routing shape (256 experts, top-10) over M=1024 (one prefill's tokens) and
M=10240 (a 10x batch), then times both.

Reports, per M:
  * selection SET parity per token (count of tokens whose chosen-expert SET
    differs from argpartition's) -- selection cannot be byte-identical because
    argpartition leaves order unspecified; parity is the SET;
  * normalized-weight max abs diff on tokens where the sets agree;
  * kernel vs stock timing (queued and per-call-synchronized lanes).
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import mlx.core as mx
from mtplx.kernels.laguna_prefill_router import (
    fused_router_prefill,
    is_router_prefill_eligible,
)

assert mx.metal.is_available(), "no Metal device"
assert mx.default_device() == mx.gpu, "run on GPU (do not set cpu)"
f32 = mx.float32
E, K = 256, 10
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


def stock_router(logits, bias, top_k, normalize=True, scale=1.0):
    scores = mx.sigmoid(logits)
    choice = scores + bias
    idx = mx.argpartition(-choice, kth=top_k - 1, axis=-1)[..., :top_k]
    w = mx.take_along_axis(scores, idx, axis=-1)
    if normalize:
        w = w / w.sum(axis=-1, keepdims=True)
    return idx.astype(mx.uint32), w * scale


mx.random.seed(1)
for M in (1024, 10240):
    print(f"\n=== P3 M={M} experts={E} top_k={K} ===")
    logits = (mx.random.normal((M, E)) * 2.0).astype(f32)
    bias = (mx.random.normal((E,)) * 0.1).astype(f32)
    mx.eval(logits, bias)

    report(f"P3 M={M} kernel eligible", is_router_prefill_eligible(logits, bias, K))

    ki, kw = fused_router_prefill(logits, bias, K, normalize=True, scale=1.0)
    si, sw = stock_router(logits, bias, K, normalize=True, scale=1.0)
    mx.eval(ki, kw, si, sw)

    ki_l = ki.tolist()
    si_l = si.tolist()
    flips = 0
    for i in range(M):
        if set(ki_l[i]) != set(si_l[i]):
            flips += 1
    report(f"P3 M={M} selection set-parity (flips)", flips == 0, f"flips={flips}/{M}")

    # weight comparison on the (sorted) union where sets agree: sort both by index
    ko = mx.argsort(ki, axis=-1)
    so = mx.argsort(si, axis=-1)
    kw_s = mx.take_along_axis(kw, ko, axis=-1)
    sw_s = mx.take_along_axis(sw, so, axis=-1)
    ki_s = mx.take_along_axis(ki, ko, axis=-1)
    si_s = mx.take_along_axis(si, so, axis=-1)
    same = (ki_s == si_s).all(axis=-1)  # tokens whose sorted index vecs match
    if same.sum().item() > 0:
        wd = mx.abs((kw_s - sw_s) * same[:, None].astype(f32)).max().item()
    else:
        wd = float("nan")
    report(f"P3 M={M} normalized weights match on agreeing tokens", wd < 1e-3,
           f"maxabs={wd:.2e}")

    ki_ms = time_call(lambda: fused_router_prefill(logits, bias, K, normalize=True, scale=1.0))
    st_ms = time_call(lambda: stock_router(logits, bias, K))
    print(f"  timing  kernel queued={ki_ms[0]:.3f}ms percall={ki_ms[1]:.3f}ms")
    print(f"  timing  stock  queued={st_ms[0]:.3f}ms percall={st_ms[1]:.3f}ms")
    print(f"  speedup(queued)={st_ms[0]/ki_ms[0]:.3f}x  speedup(percall)={st_ms[1]/ki_ms[1]:.3f}x")

print("\n" + ("P3 ALL CHECKS PASSED" if not FAIL else f"P3 FAILURES: {FAIL}"))
