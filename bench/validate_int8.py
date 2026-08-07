#!/usr/bin/env python3
"""Regression test for the int8 non-expert matvec (mtplx/int8_linear.py).

Locks the fused int8 matvec within tolerance of the reference dequant matmul across the input
conditions it actually sees in the model: contiguous bf16, contiguous f32, and a NON-contiguous
view (a strided slice — the case that hid the earlier wiring bug), plus a large OUT like lm_head
and a batched M>1. Run under the GPU flock.

    python bench/validate_int8.py
"""
import sys
import mlx.core as mx

sys.path.insert(0, __file__.rsplit("/bench/", 1)[0])
from mtplx.int8_linear import int8_matvec  # noqa

TOL = 5e-3


def _ref(x, w, s):
    return x.astype(mx.float32) @ (w.astype(mx.float32) * s.astype(mx.float32)[:, None]).T


def check(name, x, IN, OUT):
    w = mx.random.randint(-128, 128, (OUT, IN)).astype(mx.int8)
    s = mx.random.uniform(0.01, 1.0, (OUT,)).astype(mx.float16)
    y = int8_matvec(x, w, s)
    ref = _ref(x, w, s)
    rel = float(mx.max(mx.abs(y.astype(mx.float32) - ref)) / (mx.max(mx.abs(ref)) + 1e-6))
    ok = rel < TOL
    print(f"  {'PASS' if ok else 'FAIL'}  {name:24} IN={IN:6} OUT={OUT:7} rel={rel:.2e}")
    return ok


def main():
    mx.random.seed(0)
    cases = []
    IN, OUT = 2048, 512
    cases.append(check("decode M=1 bf16", mx.random.normal((1, IN)).astype(mx.bfloat16), IN, OUT))
    cases.append(check("batch M=30 bf16", mx.random.normal((30, IN)).astype(mx.bfloat16), IN, OUT))
    cases.append(check("f32", mx.random.normal((4, IN)).astype(mx.float32), IN, OUT))
    # non-contiguous view (strided slice of a wider array) — the model-activation case
    cases.append(check("non-contiguous", mx.random.normal((4, IN * 2)).astype(mx.bfloat16)[:, ::2], IN, OUT))
    # attention-scale OUT (q_proj) and lm_head-scale OUT
    cases.append(check("wide OUT (q_proj)", mx.random.normal((1, 2048)).astype(mx.bfloat16), 2048, 8192))
    cases.append(check("huge OUT (lm_head)", mx.random.normal((1, 2048)).astype(mx.bfloat16), 2048, 248320))
    ok = all(cases)
    print(f"\n{'ALL PASS' if ok else 'FAILURES'} ({sum(cases)}/{len(cases)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
