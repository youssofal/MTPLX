"""D10 moe-combine real-shape check -- RUN UNDER THE GPU FLOCK ONLY.

3-way at the S-2.1 combine shape (top-10 experts, hidden 3072, bf16):
    (a) challenge-port  -> laguna_moe_combine.fused_moe_combine
    (b) MTPLX installed -> laguna_decode.fused_moe_combine
    (c) stock chain     -> laguna._stock_moe_combine
plus the residual-fusing variant (the challenge's full tail) vs stock+residual.

Reports allclose + bit-exact (all-equal) vs stock, and timing in three lanes
(queued / chained / eager). The CHAINED lane is the decode-predictor for a B=1
serial link; queued overlaps independent dispatches; eager pays a host sync per
call. Checks both decode (rows=1) and prefill (rows=512) widths.

    .venv/bin/python scratchpad_moe_combine_check.py
"""

import time

import mlx.core as mx

mx.set_default_device(mx.gpu)

from mtplx.kernels import laguna_moe_combine as mc
from mtplx.kernels import laguna_decode as ld
from mtplx.models import laguna as lg

TOP_K = 10
HIDDEN = 3072
SCALE = 2.5


def _mk(rows):
    expert_out = (mx.random.normal((rows, TOP_K, HIDDEN)) * 0.1).astype(mx.bfloat16)
    raw = mx.abs(mx.random.normal((rows, TOP_K)))
    weights = ((raw / raw.sum(-1, keepdims=True)) * SCALE).astype(mx.float32)
    shared = (mx.random.normal((rows, HIDDEN)) * 0.1).astype(mx.bfloat16)
    residual = (mx.random.normal((rows, HIDDEN)) * 0.1).astype(mx.bfloat16)
    return expert_out, weights, shared, residual


def _report_close(name, got, ref):
    got = got.astype(mx.float32)
    ref = ref.astype(mx.float32)
    exact = bool(mx.all(got == ref))
    close = bool(mx.allclose(got, ref, atol=1e-3, rtol=1e-3))
    dmax = float(mx.max(mx.abs(got - ref)))
    print(f"  {name:42s} exact={exact!s:5s} allclose={close!s:5s} max|d|={dmax:.3e}")


def correctness(rows):
    print(f"\n== correctness (rows={rows}) ==")
    eo, w, sh, res = _mk(rows)
    mx.eval(eo, w, sh, res)

    stock = lg._stock_moe_combine(eo, w, sh)
    port = mc.fused_moe_combine(eo, w, sh)
    mtplx = ld.fused_moe_combine(eo, w, sh)
    mx.eval(stock, port, mtplx)
    _report_close("challenge-port combine vs stock", port, stock)
    _report_close("MTPLX installed combine vs stock", mtplx, stock)
    _report_close("challenge-port vs MTPLX", port, mtplx)

    stock_res = res + lg._stock_moe_combine(eo, w, sh)
    port_res = mc.fused_moe_combine_residual(eo, w, sh, res)
    mx.eval(stock_res, port_res)
    _report_close("challenge-port combine+residual vs stock+res", port_res, stock_res)


def _time_lane(fn, rows, n, lane, residual=False):
    eo, w, sh, res = _mk(rows)
    args = (eo, w, sh, res) if residual else (eo, w, sh)
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

    # chained: previous output feeds the next residual (data dependency).
    acc = res
    outs = []
    t0 = time.perf_counter()
    for _ in range(n):
        if residual:
            out = fn(eo, w, sh, acc)
        else:
            out = fn(eo, w, sh) + acc * 1e-4
        acc = out
        outs.append(out)
    mx.eval(outs)
    return (time.perf_counter() - t0) / n * 1e6


def timing(rows, n=300):
    print(f"\n== timing (rows={rows}, n={n}, us/call) ==")
    for lane in ("chained", "queued", "eager"):
        port = _time_lane(mc.fused_moe_combine, rows, n, lane)
        mtp = _time_lane(ld.fused_moe_combine, rows, n, lane)
        stk = _time_lane(lg._stock_moe_combine, rows, n, lane)
        tag = "  <- decode predictor" if lane == "chained" else ""
        print(f"  [{lane:7s}] port {port:8.2f} | mtplx {mtp:8.2f} | "
              f"stock {stk:8.2f}{tag}")
    # residual-fusing variant vs stock (stock has no fused residual -> compare to
    # port-combine + separate add to price the fusion)
    print(f"  -- residual-fusing variant (rows={rows}) --")
    for lane in ("chained",):
        pr = _time_lane(mc.fused_moe_combine_residual, rows, n, lane, residual=True)
        print(f"  [{lane:7s}] port combine+residual {pr:8.2f} us/call")


def main():
    print("device:", mx.default_device())
    print(f"S-2.1 combine: top_k={TOP_K} hidden={HIDDEN} scale(pre-baked)={SCALE}")
    for rows in (1, 512):
        correctness(rows)
    for rows in (1, 512):
        timing(rows)
    print("\nNOTE: the combine is a us-scale elementwise/reduce epilogue, NOT the "
          "decode bottleneck (the compute-bound expert GEMM is). The port folds "
          "the shared add (and optionally residual) into one dispatch; the "
          "ceiling is the ~2 removed dispatches. Judge on the CHAINED lane. The "
          "residual variant needs the decoder layer to pass the residual in.")


if __name__ == "__main__":
    main()
