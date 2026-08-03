"""D5 gated output projection real-shape check -- RUN UNDER THE GPU FLOCK ONLY.

3-way at the S-2.1 attention tail (per-head softplus gate x attention output,
then affine gs64 o_proj; heads 48 full / 72 sliding, hidden 3072, bits 5 and 8,
bf16):

    (a) challenge-port  -> laguna_gated_oproj.fused_gated_oproj    (1 dispatch:
                           softplus + gate product + affine GEMV fused)
    (b) MTPLX installed -> laguna_decode.fused_per_head_gate       (gate kernel)
                           then mx.quantized_matmul                (2 dispatches)
    (c) stock chain     -> laguna._stock_per_head_gate             (logaddexp)
                           then mx.quantized_matmul                (2 dispatches)

Reports allclose vs stock and timing in three lanes (queued / chained / eager).

Expectation: the port's projection is a FP32-accumulate affine GEMV, the same
value class as mx.quantized_matmul but reassociated -> bar is allclose (verified
here on GPU), NOT bitwise.  The gate half is bit-exact softplus.  MTPLX's
attn-gate kernel + quantized_matmul should be ~bit-exact vs the stock chain
(same softplus, same matmul); the port folds all three into ONE dispatch.  Judge
speedup on the CHAINED lane and confirm greedy-token parity in the model.

    .venv/bin/python scratchpad_gated_oproj_check.py
"""

import time

import mlx.core as mx

mx.set_default_device(mx.gpu)

from mtplx.kernels import laguna_gated_oproj as d5
from mtplx.kernels import laguna_decode as ld
from mtplx.models import laguna as lg

HIDDEN = 3072
HEAD_DIM = 128
GS = 64


def _mk(n_heads, bits):
    in_vec = n_heads * HEAD_DIM
    attn = (mx.random.normal((1, 1, in_vec)) * 0.5).astype(mx.bfloat16)
    glogits = (mx.random.normal((1, 1, n_heads)) * 1.0).astype(mx.bfloat16)
    w = (mx.random.normal((HIDDEN, in_vec)) * 0.05).astype(mx.bfloat16)
    codes, scales, biases = mx.quantize(w, group_size=GS, bits=bits)
    return attn, glogits, codes, scales, biases


def _mtplx_tail(attn, glogits, codes, scales, biases, spec):
    gated = ld.fused_per_head_gate(attn, glogits, spec.n_heads, spec.head_dim)
    return mx.quantized_matmul(gated, codes, scales, biases, transpose=True,
                               group_size=spec.group_size, bits=spec.bits)


def _report(name, got, ref):
    got, ref = got.astype(mx.float32), ref.astype(mx.float32)
    exact = bool(mx.all(got == ref))
    close = bool(mx.allclose(got, ref, atol=3e-2, rtol=3e-2))
    dmax = float(mx.max(mx.abs(got - ref)))
    print(f"    {name:44s} exact={exact!s:5s} allclose={close!s:5s} max|d|={dmax:.3e}")


def correctness():
    for n_heads, bits in ((48, 8), (72, 5), (48, 5), (72, 8)):
        spec = d5.GatedOProjSpec(n_heads=n_heads, bits=bits)
        print(f"\n== correctness (n_heads={n_heads}, bits={bits}, in_vec={spec.in_vec}) ==")
        attn, glogits, codes, scales, biases = _mk(n_heads, bits)
        mx.eval(attn, glogits, codes, scales, biases)

        assert d5.is_gated_oproj_eligible(attn, glogits, codes, scales, biases, spec), \
            "port ineligible at real shape -- kernel would NOT run"
        port = d5.fused_gated_oproj(attn, glogits, codes, scales, biases, spec)
        stock = lg._stock_per_head_gate(attn, glogits, n_heads, HEAD_DIM)
        stock = mx.quantized_matmul(stock, codes, scales, biases, transpose=True,
                                    group_size=GS, bits=bits)
        mtplx = _mtplx_tail(attn, glogits, codes, scales, biases, spec)
        mx.eval(port, stock, mtplx)

        _report("challenge-port vs stock", port, stock)
        _report("MTPLX gate+qmm vs stock", mtplx, stock)
        _report("challenge-port vs MTPLX", port, mtplx)


def _time_lane(fn, attn, glogits, codes, scales, biases, n, lane):
    args = (attn, glogits, codes, scales, biases)
    for _ in range(5):
        mx.eval(fn(*args))
    if lane == "eager":
        t0 = time.perf_counter()
        for _ in range(n):
            mx.eval(fn(*args))
        return (time.perf_counter() - t0) / n * 1e6
    if lane == "queued":
        outs = []
        t0 = time.perf_counter()
        for _ in range(n):
            outs.append(fn(*args))
        mx.eval(outs)
        return (time.perf_counter() - t0) / n * 1e6
    # chained: fold a scalar of the output back into the attention input.
    a = attn
    outs = []
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(a, glogits, codes, scales, biases)
        a = attn + mx.mean(out).astype(mx.bfloat16) * mx.array(1e-3, mx.bfloat16)
        outs.append(out)
    mx.eval(outs)
    return (time.perf_counter() - t0) / n * 1e6


def timing(n=300):
    for n_heads, bits in ((48, 8), (72, 5)):
        spec = d5.GatedOProjSpec(n_heads=n_heads, bits=bits)
        attn, glogits, codes, scales, biases = _mk(n_heads, bits)
        mx.eval(attn, glogits, codes, scales, biases)
        print(f"\n== timing (n_heads={n_heads}, bits={bits}, n={n}, us/call) ==")
        port = lambda a, g, c, s, b: d5.fused_gated_oproj(a, g, c, s, b, spec)
        mtp = lambda a, g, c, s, b: _mtplx_tail(a, g, c, s, b, spec)
        stk = lambda a, g, c, s, b: mx.quantized_matmul(
            lg._stock_per_head_gate(a, g, n_heads, HEAD_DIM), c, s, b,
            transpose=True, group_size=GS, bits=bits)
        for lane in ("chained", "queued", "eager"):
            p = _time_lane(port, attn, glogits, codes, scales, biases, n, lane)
            m = _time_lane(mtp, attn, glogits, codes, scales, biases, n, lane)
            s = _time_lane(stk, attn, glogits, codes, scales, biases, n, lane)
            tag = "  <- decode predictor" if lane == "chained" else ""
            print(f"  [{lane:7s}] port {p:8.2f} | mtplx {m:8.2f} | stock {s:8.2f}{tag}")


def main():
    print("device:", mx.default_device())
    print(f"S-2.1 gated o_proj: hidden={HIDDEN} head_dim={HEAD_DIM} gs={GS} "
          f"bits in (5,8)")
    correctness()
    timing()
    print("\nNOTE: the port fuses softplus + gate product + affine GEMV into ONE "
          "dispatch vs the stock 2-dispatch tail (gate kernel + quantized_matmul). "
          "The projection is FP32-accumulate (allclose to quantized_matmul, not "
          "bit-exact). Judge the CHAINED lane and confirm greedy-token parity.")


if __name__ == "__main__":
    main()
