"""D3 sliding plain-rope qk-norm+rope real-shape check -- RUN UNDER THE GPU FLOCK ONLY.

3-way at the S-2.1 sliding-attention decode shape (72 q heads + 8 kv heads,
head_dim 128, FULL rotary 128 dims, theta 10000, no mscale, bf16):

    (a) challenge-port  -> laguna_qk_rope_sliding.fused_qk_rope_sliding  (1 dispatch)
    (b) MTPLX installed -> laguna_decode.fused_qk_norm_rope              (1 dispatch)
    (c) stock chain     -> q_norm/k_norm -> transpose -> nn.RoPE         (4 dispatches)

Reports allclose + bit-exact vs stock and timing in three lanes
(queued / chained / eager); the CHAINED lane is the decode predictor.

Expectation: <= ~1 bf16 ULP from stock in the rotary region (the port does the
rotation itself via the angle table rather than calling mx.fast.rope, whose CPU
path fuses the multiply-add); RMSNorm is bit-exact.  Bar = allclose + greedy
token parity.

    .venv/bin/python scratchpad_qk_rope_sliding_check.py
"""

import math
import time

import mlx.core as mx

mx.set_default_device(mx.gpu)

from mlx_lm.models.rope_utils import initialize_rope

from mtplx.kernels import laguna_qk_rope_sliding as d3
from mtplx.kernels import laguna_decode as ld

SPEC = d3.SlidingRopeSpec()


def _mtplx_spec():
    return ld.QkRopeSpec(
        n_q_heads=SPEC.n_q_heads, n_kv_heads=SPEC.n_kv_heads,
        head_dim=SPEC.head_dim, rot_dims=SPEC.rot_dims,
        freqs=None, base_log2=math.log2(SPEC.base), mscale=None,
    )


def _mk():
    q = (mx.random.normal((1, 1, SPEC.n_q_heads * SPEC.head_dim)) * 0.8).astype(mx.bfloat16)
    k = (mx.random.normal((1, 1, SPEC.n_kv_heads * SPEC.head_dim)) * 0.8).astype(mx.bfloat16)
    qw = (mx.random.normal((SPEC.head_dim,)) * 0.1 + 1.0).astype(mx.bfloat16)
    kw = (mx.random.normal((SPEC.head_dim,)) * 0.1 + 1.0).astype(mx.bfloat16)
    return q, k, qw, kw


def _report(name, got, ref):
    got, ref = got.astype(mx.float32), ref.astype(mx.float32)
    exact = bool(mx.all(got == ref))
    close = bool(mx.allclose(got, ref, atol=2e-2, rtol=2e-2))
    dmax = float(mx.max(mx.abs(got - ref)))
    print(f"    {name:44s} exact={exact!s:5s} allclose={close!s:5s} max|d|={dmax:.3e}")


def correctness():
    mspec = _mtplx_spec()
    for offset in (0, 1, 137, 511):
        print(f"\n== correctness (offset={offset}) ==")
        q, k, qw, kw = _mk()
        angles = d3.build_sliding_rope_angles(offset, SPEC)
        mx.eval(q, k, qw, kw, angles)

        st_q, st_k = d3._stock_qk_rope_sliding(q, k, qw, kw, offset, SPEC)
        assert d3.is_qk_rope_sliding_eligible(q, k, qw, kw, angles, SPEC), \
            "port ineligible at real shape -- kernel would NOT run"
        pt_q, pt_k = d3.fused_qk_rope_sliding(q, k, qw, kw, angles, SPEC, offset=offset)
        elig_mtplx = ld.is_qk_norm_rope_eligible(q, k, qw, kw, mspec)
        mt_q, mt_k = ld.fused_qk_norm_rope(q, k, qw, kw, float(SPEC.eps), offset, mspec)
        mx.eval(st_q, st_k, pt_q, pt_k, mt_q, mt_k)

        _report("challenge-port q vs stock", pt_q, st_q)
        _report("challenge-port k vs stock", pt_k, st_k)
        print(f"    (MTPLX kernel eligible={elig_mtplx})")
        _report("MTPLX kernel  q vs stock", mt_q, st_q)
        _report("MTPLX kernel  k vs stock", mt_k, st_k)
        _report("challenge-port q vs MTPLX", pt_q, mt_q)


def _time_lane(fn, n, lane):
    q, k, qw, kw = _mk()
    args = (q, k, qw, kw)
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
    qa = q
    outs = []
    t0 = time.perf_counter()
    for _ in range(n):
        oq, ok = fn(qa, k, qw, kw)
        qa = oq.reshape(1, 1, SPEC.n_q_heads * SPEC.head_dim) * 0.001 + q
        outs.append(ok)
    mx.eval(outs)
    return (time.perf_counter() - t0) / n * 1e6


def timing(n=400):
    mspec = _mtplx_spec()
    offset = 137
    angles = d3.build_sliding_rope_angles(offset, SPEC)
    mx.eval(angles)
    print(f"\n== timing (offset={offset}, n={n}, us/call) ==")

    port = lambda q, k, qw, kw: d3.fused_qk_rope_sliding(q, k, qw, kw, angles, SPEC, offset=offset)
    mtp = lambda q, k, qw, kw: ld.fused_qk_norm_rope(q, k, qw, kw, float(SPEC.eps), offset, mspec)
    stk = lambda q, k, qw, kw: d3._stock_qk_rope_sliding(q, k, qw, kw, offset, SPEC)
    for lane in ("chained", "queued", "eager"):
        p, m, s = _time_lane(port, n, lane), _time_lane(mtp, n, lane), _time_lane(stk, n, lane)
        tag = "  <- decode predictor" if lane == "chained" else ""
        print(f"  [{lane:7s}] port {p:8.2f} | mtplx {m:8.2f} | stock {s:8.2f}{tag}")


def main():
    print("device:", mx.default_device())
    print(f"S-2.1 sliding: q_heads={SPEC.n_q_heads} kv_heads={SPEC.n_kv_heads} "
          f"head_dim={SPEC.head_dim} rot_dims={SPEC.rot_dims} theta={SPEC.base}")
    correctness()
    timing()
    print("\nNOTE: judge speedup on the CHAINED lane. The port replaces 4 stock "
          "dispatches with 1; expect allclose (<= ~1 bf16 ULP rotary), not "
          "bit-exact, vs mx.fast.rope. Confirm greedy-token parity in the model.")


if __name__ == "__main__":
    main()
