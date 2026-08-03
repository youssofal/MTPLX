"""D2 full-attention YaRN qk-norm+rope real-shape check -- RUN UNDER THE GPU FLOCK ONLY.

3-way at the S-2.1 full-attention decode shape (48 q heads + 8 kv heads, head_dim
128, partial-rotary 0.5 -> 64 rotary dims, YaRN mscale 1.4852030263919618, theta
500000, bf16):

    (a) challenge-port  -> laguna_qk_yarn_full.fused_qk_yarn_full   (1 dispatch)
    (b) MTPLX installed -> laguna_decode.fused_qk_norm_rope         (1 dispatch)
    (c) stock chain     -> q_norm/k_norm -> transpose -> YarnRoPE   (~6 dispatches)

Reports allclose + bit-exact (all-equal) vs stock, and timing in three lanes
(queued / chained / eager).  The CHAINED lane is the decode predictor for a B=1
serial link; queued overlaps independent dispatches; eager pays a host sync per
call.

Expectation: the port and MTPLX kernel both land <= ~1 bf16 ULP from stock in
the rotary region because both do the rotation themselves (angle table +
`x1*cos - x2*sin`, or recomputed cos/sin) rather than calling mx.fast.rope, whose
CPU path uses an FMA -- so the bar here is allclose (and greedy-token parity in
the model), not bitwise equality.  The RMSNorm and the non-rotary tail ARE
bit-exact.

    .venv/bin/python scratchpad_qk_yarn_check.py
"""

import math
import time

import mlx.core as mx

mx.set_default_device(mx.gpu)

from mlx_lm.models.rope_utils import initialize_rope

from mtplx.kernels import laguna_qk_yarn_full as d2
from mtplx.kernels import laguna_decode as ld

SPEC = d2.YarnFullSpec()


def _rope():
    return initialize_rope(
        SPEC.rot_dims, base=500000.0, traditional=False,
        scaling_config={"rope_type": "yarn", "factor": 128.0,
                        "original_max_position_embeddings": 8192,
                        "beta_fast": 32, "beta_slow": 1},
        max_position_embeddings=1_048_576,
    )


def _mtplx_spec(freqs):
    return ld.QkRopeSpec(
        n_q_heads=SPEC.n_q_heads, n_kv_heads=SPEC.n_kv_heads,
        head_dim=SPEC.head_dim, rot_dims=SPEC.rot_dims,
        freqs=freqs, base_log2=None,
        mscale=float(SPEC.mscale),
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
    rope = _rope()
    freqs = rope._freqs
    mspec = _mtplx_spec(freqs)
    print(f"YarnRoPE.mscale={float(rope.mscale)!r}  freqs.size={int(freqs.size)}")
    for offset in (0, 1, 137, 4095):
        print(f"\n== correctness (offset={offset}) ==")
        q, k, qw, kw = _mk()
        angles = d2.build_full_yarn_angles(freqs, offset, SPEC)
        mx.eval(q, k, qw, kw, angles)

        st_q, st_k = d2._stock_qk_yarn_full(q, k, qw, kw, freqs, offset, SPEC)
        pt_q, pt_k = d2.fused_qk_yarn_full(q, k, qw, kw, angles, SPEC,
                                           freqs=freqs, offset=offset)
        assert d2.is_qk_yarn_full_eligible(q, k, qw, kw, angles, SPEC), \
            "port ineligible at real shape -- kernel would NOT run"
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
    args = fn.pre(q, k, qw, kw)
    for _ in range(5):
        mx.eval(fn.call(*args))
    if lane == "eager":
        t0 = time.perf_counter()
        for _ in range(n):
            mx.eval(fn.call(*args))
        return (time.perf_counter() - t0) / n * 1e6
    if lane == "queued":
        outs = []
        t0 = time.perf_counter()
        for _ in range(n):
            outs.append(fn.call(*args))
        mx.eval(outs)
        return (time.perf_counter() - t0) / n * 1e6
    # chained: feed q output back into q input (data dependency).
    qa = args[0]
    outs = []
    t0 = time.perf_counter()
    for _ in range(n):
        oq, ok = fn.call(qa, *args[1:])
        qa = (oq.reshape(1, 1, SPEC.n_q_heads * SPEC.head_dim) * 0.001
              + args[0])
        outs.append(ok)
    mx.eval(outs)
    return (time.perf_counter() - t0) / n * 1e6


class _Port:
    def __init__(self, freqs, offset):
        self.freqs, self.offset = freqs, offset
        self.angles = d2.build_full_yarn_angles(freqs, offset, SPEC)
        mx.eval(self.angles)
    def pre(self, q, k, qw, kw):
        return (q, k, qw, kw)
    def call(self, q, k, qw, kw):
        return d2.fused_qk_yarn_full(q, k, qw, kw, self.angles, SPEC,
                                     freqs=self.freqs, offset=self.offset)


class _Mtplx:
    def __init__(self, mspec, offset):
        self.mspec, self.offset = mspec, offset
    def pre(self, q, k, qw, kw):
        return (q, k, qw, kw)
    def call(self, q, k, qw, kw):
        return ld.fused_qk_norm_rope(q, k, qw, kw, float(SPEC.eps), self.offset, self.mspec)


class _Stock:
    def __init__(self, freqs, offset):
        self.freqs, self.offset = freqs, offset
    def pre(self, q, k, qw, kw):
        return (q, k, qw, kw)
    def call(self, q, k, qw, kw):
        return d2._stock_qk_yarn_full(q, k, qw, kw, self.freqs, self.offset, SPEC)


def timing(n=400):
    rope = _rope()
    freqs = rope._freqs
    mspec = _mtplx_spec(freqs)
    offset = 137
    print(f"\n== timing (offset={offset}, n={n}, us/call) ==")
    port, mtp, stk = _Port(freqs, offset), _Mtplx(mspec, offset), _Stock(freqs, offset)
    for lane in ("chained", "queued", "eager"):
        p = _time_lane(port, n, lane)
        m = _time_lane(mtp, n, lane)
        s = _time_lane(stk, n, lane)
        tag = "  <- decode predictor" if lane == "chained" else ""
        print(f"  [{lane:7s}] port {p:8.2f} | mtplx {m:8.2f} | stock {s:8.2f}{tag}")


def main():
    print("device:", mx.default_device())
    print(f"S-2.1 full-attn: q_heads={SPEC.n_q_heads} kv_heads={SPEC.n_kv_heads} "
          f"head_dim={SPEC.head_dim} rot_dims={SPEC.rot_dims} mscale={SPEC.mscale}")
    correctness()
    timing()
    print("\nNOTE: judge speedup on the CHAINED lane (B=1 serial decode link). The "
          "port and MTPLX kernel replace ~6 stock dispatches with 1; expect "
          "allclose (<= ~1 bf16 ULP in the rotary region), not bit-exact, vs "
          "mx.fast.rope. Confirm greedy-token parity in the model before shipping.")


if __name__ == "__main__":
    main()
