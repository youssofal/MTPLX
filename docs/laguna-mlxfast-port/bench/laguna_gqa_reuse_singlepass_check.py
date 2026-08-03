"""GQA KV-reuse isolation benchmark for the decode SDPA-vector kernel.

Question: does sharing each K/V device read across a GROUP of query heads that
map to the same KV head (one threadgroup owns the group, reads K/V once, runs
GROUP independent online-softmax states) actually win at LONG context, where
decode attention is KV-bandwidth-bound?

The ONLY variable swept is GROUP (the KV-reuse factor). Tiling and reduction are
byte-for-byte identical across arms because every arm is the SAME kernel factory
(`_grouped_gqa_sdpa_kernel`) instantiated with a different `group`:

  GROUP=1   : one threadgroup per query head, re-reads KV per head. This is
              exactly MLX's sdpa_vector mapping (kv_head = q_head / gqa), so it
              isolates the KV-reuse theory from any userland-vs-MLX tiling gap.
  GROUP=k>1 : one threadgroup owns k query heads sharing a KV head, reads each
              K/V row ONCE -> up to k x less KV read.
  stock     : mx.fast.scaled_dot_product_attention (absolute reference).

Shapes: B=1, q_len=1 (decode), head_dim=128, bf16, no mask.
  full    geometry: HQ=48, HK=8 (gqa 6) -> GROUP in {1, 2, 3, 6}
  sliding geometry: HQ=72, HK=8 (gqa 9) -> GROUP in {1, 3, 9}
N sweep: {512, 2048, 8192, 32768, 65536}

Timing is the QUEUED lane: each measurement queues INNER_ITERS independent
kernel launches with a SINGLE host synchronize at the end (never one eval per
iter -- eager host-sync noise inverts µs-kernel verdicts). We take the median
across BATCHES of such queued runs.
"""

from __future__ import annotations

import time
from statistics import median

import mlx.core as mx

from mtplx.kernels.laguna_sdpa_pair import _grouped_gqa_sdpa_kernel

HEAD_DIM = 128
N_SWEEP = [512, 2048, 8192, 32768, 65536]
GEOMETRIES = {
    # name: (HQ, HK, [groups to sweep])
    "full": (48, 8, [1, 2, 3, 6]),
    "sliding": (72, 8, [1, 3, 9]),
}

INNER_ITERS = 50   # kernel launches queued per batch (single sync at the end)
BATCHES = 15       # queued batches -> median across these
WARMUP_BATCHES = 3


def make_inputs(hq: int, hk: int, n: int, d: int, dtype=mx.bfloat16):
    """Synthetic decode inputs: q [B,HQ,1,D], k/v [B,HK,N,D]."""
    b = 1
    mx.random.seed(1234 + n + hq)
    q = (mx.random.normal((b, hq, 1, d)) * 0.5).astype(dtype)
    k = (mx.random.normal((b, hk, n, d)) * 0.5).astype(dtype)
    v = (mx.random.normal((b, hk, n, d)) * 0.5).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def run_group(q, k, v, scale: float, group: int):
    """Call the kernel factory directly with an arbitrary GROUP (bypasses the
    _GROUP=3 public wrapper and its eligibility check)."""
    b, hq, _, d = (int(x) for x in q.shape)
    hk = int(k.shape[1])
    n = int(k.shape[2])
    gqa = hq // hk
    assert gqa % group == 0, f"group {group} must divide gqa {gqa}"
    num_groups = b * (hq // group)
    kernel = _grouped_gqa_sdpa_kernel(d, group, gqa, hq, hk)
    (out,) = kernel(
        inputs=[q, k, v, float(scale), int(n)],
        template=[("T", q.dtype)],
        grid=(num_groups * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(b, hq, 1, d)],
        output_dtypes=[q.dtype],
    )
    return out


def run_stock(q, k, v, scale: float):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)


def check_correctness(q, k, v, scale, groups):
    """allclose each GROUP arm vs stock; assert output shapes. Returns dict of
    (ok, max_abs_diff) per group."""
    ref = run_stock(q, k, v, scale)
    mx.eval(ref)
    b, hq, _, d = (int(x) for x in q.shape)
    assert tuple(ref.shape) == (b, hq, 1, d), f"stock shape {ref.shape}"
    ref32 = ref.astype(mx.float32)
    results = {}
    for g in groups:
        out = run_group(q, k, v, scale, g)
        mx.eval(out)
        # Hard shape assertion -- a wrong shape that happens to allclose is the
        # failure mode we must not accept.
        assert tuple(out.shape) == (b, hq, 1, d), f"group {g} shape {out.shape}"
        out32 = out.astype(mx.float32)
        max_abs = float(mx.max(mx.abs(out32 - ref32)))
        ok = bool(mx.allclose(out32, ref32, atol=2e-2, rtol=2e-2))
        results[g] = (ok, max_abs)
    return results


def time_queued(build_call, n_reads_hint=None):
    """Median (over BATCHES) of per-launch ms, measured in the QUEUED lane.

    `build_call()` must return a fresh lazy output array (one kernel launch).
    We queue INNER_ITERS of them into a list and mx.eval them all with a single
    host sync, then divide wall time by INNER_ITERS. Median across BATCHES.
    """
    # Warmup: forces Metal compile + steady clocks; keep out of the timing.
    for _ in range(WARMUP_BATCHES):
        outs = [build_call() for _ in range(INNER_ITERS)]
        mx.eval(outs)
    mx.synchronize()

    per_launch_ms = []
    for _ in range(BATCHES):
        outs = [build_call() for _ in range(INNER_ITERS)]
        t0 = time.perf_counter()
        mx.eval(outs)          # single queued submission + one host sync
        mx.synchronize()
        t1 = time.perf_counter()
        per_launch_ms.append((t1 - t0) / INNER_ITERS * 1e3)
    return median(per_launch_ms)


def bench_geometry(name, hq, hk, groups):
    d = HEAD_DIM
    gqa = hq // hk
    scale = 1.0 / (d ** 0.5)
    print(f"\n{'='*100}")
    print(f"GEOMETRY '{name}': HQ={hq} HK={hk} head_dim={d} gqa={gqa}  groups={groups}")
    print(f"{'='*100}")

    rows = []
    for n in N_SWEEP:
        q, k, v = make_inputs(hq, hk, n, d)

        # Correctness gate for this N.
        corr = check_correctness(q, k, v, scale, groups)
        bad = [g for g, (ok, _) in corr.items() if not ok]
        corr_str = " ".join(
            f"G{g}{'ok' if ok else 'WRONG'}(mad={mad:.2e})"
            for g, (ok, mad) in corr.items()
        )

        # Timing: stock + each group, queued lane.
        stock_ms = time_queued(lambda: run_stock(q, k, v, scale))
        group_ms = {}
        for g in groups:
            group_ms[g] = time_queued(lambda g=g: run_group(q, k, v, scale, g))

        # Delete big buffers before next N to keep memory low.
        del q, k, v
        rows.append((n, stock_ms, group_ms, corr, bad, corr_str))

    # ---- table ----
    reuse_groups = [g for g in groups if g > 1]
    g1 = 1
    header = (
        f"{'N':>7} | {'stock':>9} | "
        + " | ".join(f"{'G'+str(g):>9}" for g in groups)
        + f" | {'bestReuse/G1':>13} | {'bestReuse/stock':>15} | {'best G':>6}"
    )
    print("\n" + header)
    print("-" * len(header))
    for n, stock_ms, group_ms, corr, bad, corr_str in rows:
        # best reuse arm (group > 1)
        best_g = min(reuse_groups, key=lambda g: group_ms[g])
        best_ms = group_ms[best_g]
        r_g1 = best_ms / group_ms[g1]
        r_stock = best_ms / stock_ms
        line = (
            f"{n:>7} | {stock_ms:>9.4f} | "
            + " | ".join(f"{group_ms[g]:>9.4f}" for g in groups)
            + f" | {r_g1:>13.3f} | {r_stock:>15.3f} | {'G'+str(best_g):>6}"
        )
        print(line)
    # correctness footnote
    print("\ncorrectness (allclose vs stock, atol/rtol 2e-2; mad = max abs diff):")
    for n, stock_ms, group_ms, corr, bad, corr_str in rows:
        flag = "  <-- WRONG ARMS" if bad else ""
        print(f"  N={n:>6}: {corr_str}{flag}")
    return rows


def main():
    print("GQA KV-reuse isolation benchmark (decode SDPA-vector kernel)")
    print(f"mlx {mx.__version__}  metal={mx.metal.is_available()}")
    print(f"INNER_ITERS={INNER_ITERS} BATCHES={BATCHES} (queued lane, median of batches)")
    for name, (hq, hk, groups) in GEOMETRIES.items():
        bench_geometry(name, hq, hk, groups)


if __name__ == "__main__":
    main()
