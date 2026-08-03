"""Prefill grouped gather-GEMM check: hand affine-4bit gather-GEMM vs stock
`mx.gather_qmm(sorted_indices=True)` at the real Laguna S-2.1 MoE shape.

Answers the coordinator's question: does a HAND grouped gather-GEMM (the challenge
prefill kernel `fp_gather_qmm_rhs_nax` re-expressed for affine 4-bit gs128) beat
the stock steel gather-GEMM at prefill token counts?

Builds one real gate-proj bank (256 experts, N=moe_inter 1024, K=hidden 3072,
4-bit gs128). For each prefill T, expands to M = T*top_k sorted (token,expert)
rows and compares hand kernel vs stock for (a) max|diff| and (b) queued-lane
median ms. Also probes RUNSKIP applicability (empty-expert fraction) and a
split-K variant on stock gather_qmm.

Run:
    cd <worktree> && PYTHONPATH="$PWD" <venv python> scratchpad_moe_gather_check.py
"""

from __future__ import annotations

import time

import mlx.core as mx

from mtplx.kernels.laguna_moe_gather_gemm import (
    grouped_gather_gemm_t,
    sorted_run_layout,
    is_grouped_gather_eligible,
)

E, N, K = 256, 1024, 3072      # experts, moe_intermediate, hidden
TOPK = 10
GS, BITS = 128, 4
DTYPE = mx.bfloat16
TOL_ABS = 1e-2

T_SWEEP = (256, 512, 1024)
THREAD_CANDS = (256, 512)
ROWTILE_CANDS = (8, 16)


def build_gate_bank():
    print(f"building gate-proj bank: E={E}, N={N}, K={K}, 4-bit gs128 ...")
    w = mx.random.normal((E, N, K)) * (1.0 / (K ** 0.5))
    q, s, b = mx.quantize(w, group_size=GS, bits=BITS, mode="affine")
    mx.eval(q, s, b)
    del w
    return q, s, b


def make_sample(tokens):
    """M = tokens*TOPK sorted (token,expert) rows: each token picks TOPK experts."""
    M = tokens * TOPK
    scores = mx.random.normal((tokens, E))
    idx = mx.argpartition(-scores, kth=TOPK - 1, axis=-1)[..., :TOPK]  # [tokens, TOPK]
    row_expert = idx.reshape(M).astype(mx.uint32)
    x = mx.random.normal((M, K)).astype(DTYPE)
    order = mx.argsort(row_expert)
    re_sorted = row_expert[order]
    x_sorted = x[order]
    _, _, start, count = sorted_run_layout(re_sorted, E)
    mx.eval(x_sorted, re_sorted, start, count)
    return x_sorted, re_sorted, start, count


def stock_call(x_sorted, re_sorted, q, s, b):
    M, Kd = int(x_sorted.shape[0]), int(x_sorted.shape[1])
    y = mx.gather_qmm(
        x_sorted.reshape(M, 1, Kd), q, s, b,
        rhs_indices=re_sorted, transpose=True,
        group_size=GS, bits=BITS, mode="affine", sorted_indices=True,
    )
    return y.reshape(M, N)


def queued_median_ms(make_call, n_per_batch, repeats=7):
    for _ in range(2):
        mx.eval(make_call(0))
    mx.synchronize()
    per_call = []
    for _ in range(repeats):
        mx.synchronize()
        t0 = time.perf_counter()
        outs = [make_call(j) for j in range(n_per_batch)]
        mx.eval(outs)
        mx.synchronize()
        t1 = time.perf_counter()
        per_call.append((t1 - t0) / n_per_batch * 1e3)
    per_call.sort()
    return per_call[len(per_call) // 2]


def quick_ms(make_call, n=3):
    """Cheap single-batch estimate to pick a config without a full sweep."""
    mx.eval(make_call(0))
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [make_call(j) for j in range(n)]
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def batch_for(tokens):
    M = tokens * TOPK
    out_mb = M * N * 4 / 1e6
    return max(6, min(24, int(1500 / max(out_mb, 1.0))))


def splitk_stock(x_sorted, re_sorted, q, s, b, parts=2):
    """Split-K probe: partition K into `parts` group-aligned slices, gather_qmm
    each, sum. Tests whether adding K-parallelism helps stock at prefill."""
    M, Kd = int(x_sorted.shape[0]), int(x_sorted.shape[1])
    xr = x_sorted.reshape(M, 1, Kd)
    kg = Kd // GS
    assert kg % parts == 0
    gper = kg // parts          # groups per part
    kper = gper * GS            # K per part
    kpper = kper // 8           # packed uint32 per part
    acc = None
    for p in range(parts):
        xs = xr[:, :, p * kper:(p + 1) * kper]
        ws = q[:, :, p * kpper:(p + 1) * kpper]
        ss = s[:, :, p * gper:(p + 1) * gper]
        bs = b[:, :, p * gper:(p + 1) * gper]
        yp = mx.gather_qmm(xs, ws, ss, bs, rhs_indices=re_sorted, transpose=True,
                           group_size=GS, bits=BITS, mode="affine", sorted_indices=True)
        acc = yp if acc is None else acc + yp
    return acc.reshape(M, N)


def main():
    mx.random.seed(0)
    print("metal:", mx.metal.is_available(), "| device:", mx.default_device())
    q, s, b = build_gate_bank()

    rows = []
    for tokens in T_SWEEP:
        M = tokens * TOPK
        x_sorted, re_sorted, start, count = make_sample(tokens)
        npb = batch_for(tokens)

        assert is_grouped_gather_eligible(x_sorted, q, s, b, start, count)
        empties = int(mx.sum((count == 0).astype(mx.int32)))
        avg_per_expert = M / E

        ref = stock_call(x_sorted, re_sorted, q, s, b)
        mx.eval(ref)

        stock_ms = queued_median_ms(
            lambda j: stock_call(x_sorted, re_sorted, q, s, b), npb
        )

        # Cheap quick-pick over (threads, row_tile), with a one-time correctness
        # check per config; full median only on the winner.
        pick = None  # (threads, rt, quick_ms, max_abs)
        for threads in THREAD_CANDS:
            for rt in ROWTILE_CANDS:
                got = grouped_gather_gemm_t(
                    x_sorted, q, s, b, start, count, threads=threads, row_tile=rt
                )
                mx.eval(got)
                assert tuple(got.shape) == (M, N)
                max_abs = float(mx.max(mx.abs(got - ref)))
                qm = quick_ms(
                    lambda j, t=threads, r=rt: grouped_gather_gemm_t(
                        x_sorted, q, s, b, start, count, threads=t, row_tile=r
                    )
                )
                if pick is None or qm < pick[2]:
                    pick = (threads, rt, qm, max_abs)

        threads, rt, _, max_abs = pick
        k_ms = queued_median_ms(
            lambda j: grouped_gather_gemm_t(
                x_sorted, q, s, b, start, count, threads=threads, row_tile=rt
            ),
            npb,
        )
        best = (threads, rt, k_ms, max_abs)

        # split-K probe (2-way) on stock
        yk = splitk_stock(x_sorted, re_sorted, q, s, b, parts=2)
        mx.eval(yk)
        sk_abs = float(mx.max(mx.abs(yk - ref)))
        sk_ms = queued_median_ms(
            lambda j: splitk_stock(x_sorted, re_sorted, q, s, b, parts=2), npb
        )

        threads, rt, k_ms, max_abs = best
        rows.append({
            "T": tokens, "M": M, "stock_ms": stock_ms, "kernel_ms": k_ms,
            "threads": threads, "rt": rt, "ratio": k_ms / stock_ms,
            "max_abs": max_abs, "empties": empties, "avg": avg_per_expert,
            "splitk_ms": sk_ms, "splitk_ratio": sk_ms / stock_ms, "sk_abs": sk_abs,
        })
        print(
            f"  T={tokens:>4} (M={M:>6}) | stock {stock_ms:8.4f} | "
            f"hand {k_ms:8.4f} (t={threads},rt={rt}) ratio {k_ms/stock_ms:5.2f}x | "
            f"split-K2 {sk_ms:8.4f} ratio {sk_ms/stock_ms:5.2f}x | "
            f"empty-experts {empties}/{E} (avg {avg_per_expert:.1f}/expert) | "
            f"max|diff| {max_abs:.2e}/{sk_abs:.2e}"
        )

    print("\n=== TABLE (T, yours ms, stock ms, ratio, win?) ===")
    print(f"{'T':>5} | {'M':>7} | {'yours(ms)':>10} | {'stock(ms)':>10} | {'ratio':>7} | win?")
    for r in rows:
        print(f"{r['T']:>5} | {r['M']:>7} | {r['kernel_ms']:>10.4f} | "
              f"{r['stock_ms']:>10.4f} | {r['ratio']:>6.2f}x | "
              f"{'YES' if r['ratio'] < 1.0 else 'no'}")

    any_win = any(r["ratio"] < 1.0 for r in rows)
    all_pass = all(r["max_abs"] <= TOL_ABS for r in rows)
    any_splitk_win = any(r["splitk_ratio"] < 1.0 for r in rows)
    total_empty = sum(r["empties"] for r in rows)
    best = min(rows, key=lambda r: r["ratio"])
    print("\n=== VERDICT ===")
    print(f"allclose all T: {'PASS' if all_pass else 'FAIL'} "
          f"(worst hand {max(r['max_abs'] for r in rows):.2e}, "
          f"worst split-K {max(r['sk_abs'] for r in rows):.2e}, tol {TOL_ABS:.0e})")
    print(f"best hand ratio: {best['ratio']:.2f}x at T={best['T']} "
          f"(threads={best['threads']}, row_tile={best['rt']})")
    print(f"RUNSKIP applicability: {total_empty} empty experts across all T "
          f"-> RUNSKIP has ~nothing to skip at top-10/256 prefill")
    print(f"split-K (2-way) beats stock anywhere: {'YES' if any_splitk_win else 'NO'}")
    if any_win:
        print("VERDICT: hand grouped gather-GEMM BEATS stock at some T.")
    else:
        print("VERDICT: hand grouped gather-GEMM does NOT beat stock at ANY prefill T.")
        print("Mechanism: stock gather_qmm(sorted) is MLX's steel MMA GEMM "
              "(16x16 simdgroup_matrix, BK-double-buffered); at prefill it is "
              "compute-bound and rides the hardware matmul. The hand kernel uses "
              "scalar FMA + hand dequant, so it cannot match MMA throughput. This "
              "matches the challenge's own finding: prefill NVFP4 GEMM is the MLX "
              "builtin with NO HEADROOM; RUNSKIP inert, staging shelved-regressed, "
              "split-K +0.18% (sub-noise).")


if __name__ == "__main__":
    main()
