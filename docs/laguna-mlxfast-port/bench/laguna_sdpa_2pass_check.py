"""Fair-vehicle re-test: 2-pass KV-split decode SDPA, GROUP = KV-reuse knob.

Correctness (allclose vs stock) + queued-lane timing across geometries, N, S.
Answers: WITHIN the 2-pass vehicle (same tiling), does GROUP>1 beat GROUP=1 at
long N, and does the win grow with N? Does any 2-pass-reuse arm beat stock?
"""

from __future__ import annotations

import math
import time

import mlx.core as mx

from mtplx.kernels.laguna_sdpa_2pass import two_pass_gqa_sdpa_decode


def make_inputs(b, hq, hk, n, d, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((b, hq, 1, d)).astype(dtype)
    k = mx.random.normal((b, hk, n, d)).astype(dtype)
    v = mx.random.normal((b, hk, n, d)).astype(dtype)
    mx.eval(q, k, v)
    return q, k, v


def stock(q, k, v, scale):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=None)


def max_abs_diff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


def check_correctness(geoms, n_list, s_list):
    print("=" * 92)
    print("CORRECTNESS  (max|arm - stock|, bf16; PASS if <= 2e-2 and shape ok)")
    print("=" * 92)
    all_ok = True
    for name, (hq, hk, groups) in geoms.items():
        b, d = 1, 128
        scale = 1.0 / math.sqrt(d)
        for n in n_list:
            q, k, v = make_inputs(b, hq, hk, n, d)
            ref = stock(q, k, v, scale)
            mx.eval(ref)
            assert ref.shape == (b, hq, 1, d), ref.shape
            for s in s_list:
                for g in groups:
                    out = two_pass_gqa_sdpa_decode(
                        q, k, v, scale=scale, group=g, chunks=s
                    )
                    assert out is not None, f"ineligible {name} g={g}"
                    mx.eval(out)
                    # fake-speedup guard: shape must match exactly.
                    assert out.shape == (b, hq, 1, d), (out.shape, name, g, s)
                    diff = max_abs_diff(out, ref)
                    ok = diff <= 2e-2
                    all_ok = all_ok and ok
                    flag = "ok " if ok else "FAIL"
                    print(
                        f"  {flag} {name:8s} N={n:>6d} S={s:>4d} G={g:<2d} "
                        f"maxdiff={diff:.4e}"
                    )
    print(f"\nCORRECTNESS: {'ALL PASS' if all_ok else 'FAILURES PRESENT'}\n")
    return all_ok


def time_fn(build, iters=40, repeats=4):
    """Queued-lane timing: chain `iters` calls into one dependency chain, then a
    single eval+sync. Memory bounded because each partial set is consumed by the
    running accumulate. Returns min over `repeats` of per-call ms."""
    # warmup (compile + caches + one warm chain)
    for _ in range(3):
        w = build()
        mx.eval(w)
    acc = build()
    for _ in range(iters - 1):
        acc = acc + build()
    mx.eval(acc)
    mx.synchronize()

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        acc = build()
        for _ in range(iters - 1):
            acc = acc + build()
        mx.eval(acc)
        mx.synchronize()
        dt = (time.perf_counter() - t0) / iters
        best = min(best, dt)
    return best * 1e3  # ms per call


def run_timing(geoms, n_list, s_list):
    print("=" * 92)
    print("TIMING  (queued-lane, ms/call, min over repeats). Lower is better.")
    print("=" * 92)

    results = {}  # (name, n) -> dict
    for name, (hq, hk, groups) in geoms.items():
        b, d = 1, 128
        scale = 1.0 / math.sqrt(d)
        gqa = hq // hk
        for n in n_list:
            q, k, v = make_inputs(b, hq, hk, n, d)
            # stock reference time (no S)
            st = time_fn(lambda: stock(q, k, v, scale))
            per_s = {}
            for s in s_list:
                arm = {}
                for g in groups:
                    arm[g] = time_fn(
                        lambda g=g, s=s: two_pass_gqa_sdpa_decode(
                            q, k, v, scale=scale, group=g, chunks=s
                        )
                    )
                per_s[s] = arm
            results[(name, n)] = {
                "stock": st,
                "per_s": per_s,
                "groups": groups,
                "gqa": gqa,
            }
            print(f"  measured {name} N={n} stock={st:.4f}ms")
    return results


def report(geoms, n_list, results):
    for name, (hq, hk, groups) in geoms.items():
        gqa = hq // hk
        g1 = 1
        gg = gqa
        gmid = 3 if 3 in groups else groups[min(1, len(groups) - 1)]
        print()
        print("=" * 110)
        print(f"GEOMETRY {name}: HQ={hq} HK={hk} gqa={gqa} | arms G1(control) "
              f"G{gmid} G{gg}(max reuse) vs stock")
        print("=" * 110)
        hdr = (f"{'N':>7} {'S':>5} {'stock':>9} {'G1':>9} {'G'+str(gmid):>9} "
               f"{'G'+str(gg):>9} {'best/G1':>8} {'best/stk':>9} {'reuse win?':>11}")
        print(hdr)
        print("-" * len(hdr))
        for n in n_list:
            r = results[(name, n)]
            st = r["stock"]
            per_s = r["per_s"]
            # pick S that minimizes the max-reuse arm (sensible: best for reuse),
            # then show ALL arms at that same S (fair, same tiling).
            best_s = min(per_s.keys(), key=lambda s: per_s[s][gg])
            arm = per_s[best_s]
            t1 = arm[g1]
            tm = arm[gmid]
            tg = arm[gg]
            best_reuse = min(tm, tg)
            r_g1 = best_reuse / t1
            r_stk = best_reuse / st
            win = "YES" if best_reuse < t1 * 0.995 else "no"
            print(f"{n:>7} {best_s:>5} {st:>9.4f} {t1:>9.4f} {tm:>9.4f} "
                  f"{tg:>9.4f} {r_g1:>8.3f} {r_stk:>9.3f} {win:>11}")
        # full S transparency
        print(f"\n  --- all-S detail for {name} (ms/call) ---")
        for n in n_list:
            per_s = results[(name, n)]["per_s"]
            for s in sorted(per_s.keys()):
                arm = per_s[s]
                cells = "  ".join(f"G{g}={arm[g]:.4f}" for g in groups)
                print(f"    N={n:>6} S={s:>4}  {cells}")


def main():
    n_list = [8192, 32768, 65536, 131072]
    s_list = [128, 256, 512]
    geoms = {
        "full":    (48, 8, [1, 2, 3, 6]),   # gqa 6
        "sliding": (72, 8, [1, 3, 9]),      # gqa 9
    }

    ok = check_correctness(geoms, n_list, s_list)
    if not ok:
        print("!! correctness failures -- timing verdict is meaningless, aborting")
        return

    results = run_timing(geoms, n_list, s_list)
    report(geoms, n_list, results)


if __name__ == "__main__":
    main()
