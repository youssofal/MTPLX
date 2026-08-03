"""D12 router-topk real-shape check -- RUN UNDER THE GPU FLOCK ONLY.

3-way at the S-2.1 router shape (256 experts, top-10, norm_topk, scale 2.5):
    (a) challenge-port bitonic  -> laguna_router_topk.fused_router_topk_bitonic
    (b) MTPLX installed router   -> laguna_decode.fused_router_topk (tree select)
    (c) stock chain              -> sigmoid+bias+argpartition+gather+norm+scale

Reports SELECTION parity (index-set match vs stock argpartition, flip counts over
many random draws), weight allclose where sets match, and timing in three lanes
(queued / chained / eager). The CHAINED lane is the decode-predictive one for a
B=1 serial link (see laguna_decode router_gemv notes); queued overlaps
independent dispatches and hides single-dispatch occupancy; eager pays a host
sync per call and can invert the verdict for a us-scale op.

    export MTPLX_GPU=1   # optional convention
    .venv/bin/python scratchpad_router_topk_check.py
"""

import time

import mlx.core as mx

mx.set_default_device(mx.gpu)

from mtplx.kernels import laguna_router_topk as rt
from mtplx.kernels import laguna_decode as ld

EXPERTS = 256
TOP_K = 10
SCALE = 2.5
NORMALIZE = True


def set_of(row):
    return set(int(v) for v in row.tolist())


def selection_parity(rows, draws=64):
    """Count, per arm, how many draws diverge from stock argpartition's SET."""
    flips = {"bitonic": 0, "mtplx": 0}
    wmax = {"bitonic": 0.0, "mtplx": 0.0}
    for _ in range(draws):
        logits = mx.random.normal((rows, EXPERTS)).astype(mx.float32)
        bias = mx.random.normal((EXPERTS,)).astype(mx.float32)

        scores = mx.sigmoid(logits)
        choice = scores + bias
        ap = mx.argpartition(-choice, kth=TOP_K - 1, axis=-1)[..., :TOP_K]
        st_w = mx.take_along_axis(scores, ap, axis=-1)
        if NORMALIZE:
            st_w = st_w / st_w.sum(-1, keepdims=True)
        st_w = st_w * SCALE
        mx.eval(ap, st_w)
        st_map = [
            {int(i): float(w) for i, w in zip(ap[r].tolist(), st_w[r].tolist())}
            for r in range(rows)
        ]

        for name, fn in (
            ("bitonic", rt.fused_router_topk_bitonic),
            ("mtplx", ld.fused_router_topk),
        ):
            idx, w = fn(logits, bias, TOP_K, normalize=NORMALIZE, scale=SCALE)
            mx.eval(idx, w)
            for r in range(rows):
                got = {int(i): float(x) for i, x in zip(idx[r].tolist(), w[r].tolist())}
                if set(got) != set(st_map[r]):
                    flips[name] += 1
                else:
                    for k in got:
                        wmax[name] = max(wmax[name], abs(got[k] - st_map[r][k]))
    return flips, wmax


def _time_lane(make_call, n, lane):
    logits = mx.random.normal((make_call.rows, EXPERTS)).astype(mx.float32)
    bias = mx.random.normal((EXPERTS,)).astype(mx.float32)
    fn = make_call.fn
    # warmup
    for _ in range(5):
        idx, w = fn(logits, bias, TOP_K, normalize=NORMALIZE, scale=SCALE)
        mx.eval(idx, w)

    if lane == "eager":
        t0 = time.perf_counter()
        for _ in range(n):
            idx, w = fn(logits, bias, TOP_K, normalize=NORMALIZE, scale=SCALE)
            mx.eval(idx, w)
        return (time.perf_counter() - t0) / n * 1e6

    if lane == "queued":
        outs = []
        t0 = time.perf_counter()
        for _ in range(n):
            idx, w = fn(logits, bias, TOP_K, normalize=NORMALIZE, scale=SCALE)
            outs.append(w)
        mx.eval(outs)
        return (time.perf_counter() - t0) / n * 1e6

    # chained: each iter's selected weights perturb the next logits (true data
    # dependency, one command buffer, no host sync) -- the B=1 serial-decode
    # predictor. The scalar feedback keeps the graph shape fixed per step.
    acc = logits
    outs = []
    t0 = time.perf_counter()
    for _ in range(n):
        idx, w = fn(acc, bias, TOP_K, normalize=NORMALIZE, scale=SCALE)
        acc = acc + w.sum() * 1e-6  # iter i+1 depends on iter i's output
        outs.append(acc)
    mx.eval(outs)
    return (time.perf_counter() - t0) / n * 1e6


class _MC:
    def __init__(self, fn, rows):
        self.fn = fn
        self.rows = rows


def main():
    print("device:", mx.default_device())
    print(f"S-2.1 router: experts={EXPERTS} top_k={TOP_K} norm={NORMALIZE} scale={SCALE}")

    for rows in (1, 4):
        print(f"\n== selection parity (rows={rows}, 64 draws) ==")
        flips, wmax = selection_parity(rows)
        print(f"  bitonic: {flips['bitonic']} set-flips vs argpartition | "
              f"max|dw| matched = {wmax['bitonic']:.3e}")
        print(f"  mtplx  : {flips['mtplx']} set-flips vs argpartition | "
              f"max|dw| matched = {wmax['mtplx']:.3e}")

    n = 300
    for rows in (1, 4):
        print(f"\n== timing (rows={rows}, n={n}, us/call) ==")
        for lane in ("chained", "queued", "eager"):
            bit = _time_lane(_MC(rt.fused_router_topk_bitonic, rows), n, lane)
            mtp = _time_lane(_MC(ld.fused_router_topk, rows), n, lane)
            stk = _time_lane(_MC(rt._stock_router_topk, rows), n, lane)
            tag = "  <- decode predictor" if lane == "chained" else ""
            print(f"  [{lane:7s}] bitonic {bit:7.2f} | mtplx {mtp:7.2f} | "
                  f"stock {stk:7.2f}{tag}")

    print("\nNOTE: router selection is a us-scale epilogue over 256 floats; it is "
          "NOT the decode bottleneck (the compute-bound expert GEMM is). Expect "
          "the bitonic to land near the MTPLX selector and stock -- the ceiling "
          "here is tiny. Judge on the CHAINED lane.")


if __name__ == "__main__":
    main()
