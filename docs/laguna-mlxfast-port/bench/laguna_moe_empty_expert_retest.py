"""RE-TEST: does stock `mx.gather_qmm(sorted_indices=True)` (MLX steel MMA grouped
GEMM, what SwitchGLU uses) WASTE MMA time on EMPTY experts at REAL Laguna S-2.1
prefill routing?

## Why re-test
The earlier probe (scratchpad_moe_gather_check.py / laguna_moe_gather_check.py)
concluded "RUNSKIP inert" but built its routing with UNIFORM RANDOM scores
(top-10 of gaussian gate logits), which spreads ~evenly -> ~0 empty experts. REAL
S-2.1 prefill routing (measured on the model, ctx 1024) is heavily imbalanced:
mean 63.7 / 256 empty experts per layer, up to 115. So the empty fraction is
~25-45%, not ~0. This script re-tests RUNSKIP headroom with realistic imbalance.

## The headroom question (decides if an MLX RUNSKIP fn-const is worth it)
At a FIXED total row count M, does stock gather_qmm time depend on the number of
ACTIVE experts (equivalently, on the empty fraction)?
  - time ~CONSTANT as #active drops (empties rise)  => stock schedules per-expert
    tiles / iterates expert slots and burns MMA on 0-token experts
    => RUNSKIP headroom ~= empty fraction.
  - time SCALES DOWN as #active drops               => stock already compacts /
    only touches present segments => RUNSKIP redundant.

The full 256-expert weight bank is ALWAYS passed (so the kernel always "knows"
E=256); only the rhs_indices distribution (which experts actually receive rows)
changes. That is exactly the empty-expert axis.

## Method
S-2.1 gate-proj bank: E=256, N=1024, K=3072, affine 4-bit gs128 (same as the
earlier probe). Fixed M = 10240 (= T=1024 * top-10 prefill assignment count).
Build sorted (row->expert) streams directly from per-expert count vectors so the
empty count is EXACT and controllable:
  - EVEN distributions   (rows spread evenly among the active experts) isolate the
    pure #active-experts effect from load-shape.
  - IMBALANCED (lognormal heavy-tail among active) reflect true routing shape.
Two realism points: ~64 empty (mean) and ~115 empty (worst layer). Plus an
active-experts sweep at fixed M: active in {256,192,140,100}. Also a
blocked-vs-scattered empty-placement check (does WHERE the empties sit matter).

Timing: queued lane (warmup, many queued iters, one eval+sync), median ms. Output
shapes asserted. Also runs the hand RUNSKIP-style kernel
(laguna_moe_gather_gemm.grouped_gather_gemm_t, which early-returns empty-expert
threadgroups) at balanced vs empty-heavy routing to isolate whether RUNSKIP *the
technique* recovers work proportional to the empty fraction.

Run:
  cd <worktree> && PYTHONPATH="$PWD" <venv python> \
      docs/laguna-mlxfast-port/bench/laguna_moe_empty_expert_retest.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import numpy as np

from mtplx.kernels.laguna_moe_gather_gemm import (
    grouped_gather_gemm_t,
    sorted_run_layout,
    is_grouped_gather_eligible,
)

E, N, K = 256, 1024, 3072      # experts, moe_intermediate, hidden
GS, BITS = 128, 4
DTYPE = mx.bfloat16
TOL_ABS = 1e-2

M_FIXED = 10240                # T=1024 * top_k=10 : the real prefill assignment count
IMB_SIGMA = 1.1               # lognormal spread of the imbalanced load shape


# ---------------------------------------------------------------- bank + inputs
def build_gate_bank():
    print(f"building gate-proj bank: E={E}, N={N}, K={K}, 4-bit gs128 ...")
    w = mx.random.normal((E, N, K)) * (1.0 / (K ** 0.5))
    q, s, b = mx.quantize(w, group_size=GS, bits=BITS, mode="affine")
    mx.eval(q, s, b)
    del w
    return q, s, b


def counts_even(active_ids: np.ndarray, M: int) -> np.ndarray:
    """counts[E] with rows spread as evenly as possible over `active_ids`."""
    counts = np.zeros(E, dtype=np.int64)
    A = len(active_ids)
    base, rem = divmod(M, A)
    counts[active_ids] = base
    counts[active_ids[:rem]] += 1  # spread the remainder
    assert counts.sum() == M
    return counts


def counts_imbalanced(active_ids: np.ndarray, M: int, sigma: float,
                      rng: np.random.Generator) -> np.ndarray:
    """counts[E] with a lognormal heavy-tail load over `active_ids` (min 1 each),
    summing EXACTLY to M. Reflects real MoE routing concentration."""
    A = len(active_ids)
    w = rng.lognormal(mean=0.0, sigma=sigma, size=A)
    w = w / w.sum()
    raw = w * (M - A)                         # reserve 1 per active for the min
    c = np.floor(raw).astype(np.int64) + 1    # >= 1
    short = M - int(c.sum())
    # hand the leftover rows to the largest fractional parts (stable, sum-exact)
    frac = raw - np.floor(raw)
    order = np.argsort(-frac)
    i = 0
    while short > 0:
        c[order[i % A]] += 1
        short -= 1
        i += 1
    counts = np.zeros(E, dtype=np.int64)
    counts[active_ids] = c
    assert counts.sum() == M
    assert (counts[active_ids] >= 1).all()
    return counts


def sorted_expert_ids(counts: np.ndarray) -> np.ndarray:
    """Length-M expert id per row, ascending by id (already sorted for gather)."""
    return np.repeat(np.arange(E, dtype=np.uint32), counts)


def make_stream(counts: np.ndarray, x_pool: mx.array):
    """(x_sorted [M,K] bf16, re_sorted [M] uint32) for a given counts vector.

    x rows are irrelevant to timing, so we slice a shared random pool; the row
    ORDER already matches ascending expert id, which is what sorted gather wants.
    """
    ids = sorted_expert_ids(counts)
    M = int(ids.shape[0])
    re_sorted = mx.array(ids)               # uint32, sorted
    x_sorted = x_pool[:M]
    mx.eval(re_sorted)
    return x_sorted, re_sorted, M


# ---------------------------------------------------------------- stock + hand
def stock_call(x_sorted, re_sorted, q, s, b):
    M, Kd = int(x_sorted.shape[0]), int(x_sorted.shape[1])
    y = mx.gather_qmm(
        x_sorted.reshape(M, 1, Kd), q, s, b,
        rhs_indices=re_sorted, transpose=True,
        group_size=GS, bits=BITS, mode="affine", sorted_indices=True,
    )
    y = y.reshape(M, N)
    assert tuple(y.shape) == (M, N), f"stock produced {tuple(y.shape)} != {(M, N)}"
    return y


def hand_call(x_sorted, re_sorted, q, s, b, threads=256, row_tile=8):
    _, _, start, count = sorted_run_layout(re_sorted, E)
    y = grouped_gather_gemm_t(x_sorted, q, s, b, start, count,
                              threads=threads, row_tile=row_tile)
    return y


# ---------------------------------------------------------------- queued timing
def queued_median_ms(call, iters=15, repeats=5, warmup=3):
    for _ in range(warmup):
        mx.eval(call())
    mx.synchronize()
    per = []
    for _ in range(repeats):
        mx.synchronize()
        t0 = time.perf_counter()
        outs = [call() for _ in range(iters)]
        mx.eval(outs)
        mx.synchronize()
        per.append((time.perf_counter() - t0) / iters * 1e3)
    per.sort()
    return per[len(per) // 2]


# ---------------------------------------------------------------- experiments
def main():
    seed = 0
    mx.random.seed(seed)
    rng = np.random.default_rng(seed)
    print("metal:", mx.metal.is_available(), "| device:", mx.default_device())
    print(f"FIXED M = {M_FIXED} (T=1024 x top-10), E={E}\n")

    q, s, b = build_gate_bank()
    # Shared random x pool (values do not affect timing; order = ascending expert).
    x_pool = mx.random.normal((M_FIXED, K)).astype(DTYPE)
    mx.eval(x_pool)

    def scattered_active(n_active):
        ids = rng.choice(E, size=n_active, replace=False)
        ids.sort()
        return ids.astype(np.int64)

    def blocked_active(n_active):
        return np.arange(n_active, dtype=np.int64)

    # ---- 1) Main table: balanced vs realistic vs worst, even AND imbalanced ----
    print("=" * 96)
    print("MAIN: fixed M, stock gather_qmm ms vs empty fraction "
          "(full 256-expert bank always passed)")
    print("=" * 96)

    specs = [
        # (label, n_active, shape)
        ("balanced-even",     256, "even"),
        ("realistic-even",    192, "even"),   # 64 empty (measured mean)
        ("worst-even",        141, "even"),   # 115 empty (measured worst layer)
        ("balanced-imbal",    256, "imbal"),
        ("realistic-imbal",   192, "imbal"),
        ("worst-imbal",       141, "imbal"),
    ]

    main_rows = []
    baseline_ms = {}  # per shape-family -> balanced ms
    for label, n_active, shape in specs:
        active = scattered_active(n_active)
        if shape == "even":
            counts = counts_even(active, M_FIXED)
        else:
            counts = counts_imbalanced(active, M_FIXED, IMB_SIGMA, rng)
        n_empty = int((counts == 0).sum())
        assert n_empty == E - n_active, (n_empty, E - n_active)
        x_sorted, re_sorted, M = make_stream(counts, x_pool)
        assert M == M_FIXED

        # correctness: stock output must be finite and correctly shaped
        y = stock_call(x_sorted, re_sorted, q, s, b)
        mx.eval(y)
        assert bool(mx.all(mx.isfinite(y))), f"{label}: non-finite stock output"

        ms = queued_median_ms(lambda: stock_call(x_sorted, re_sorted, q, s, b))
        # load-shape stats
        nz = counts[counts > 0]
        row = {
            "label": label, "active": n_active, "empty": n_empty, "M": M,
            "ms": ms, "shape": shape,
            "max_run": int(nz.max()), "min_run": int(nz.min()),
        }
        main_rows.append(row)
        fam = shape
        if "balanced" in label:
            baseline_ms[fam] = ms
        print(f"  {label:<16} active={n_active:>3} empty={n_empty:>3} "
              f"({n_empty/E*100:4.1f}%) | runs[min..max]={row['min_run']:>3}..{row['max_run']:>4} "
              f"| stock {ms:8.4f} ms")

    # ---- 2) Isolation sweep: active in {256,192,140,100}, fixed M, even -------
    print("\n" + "=" * 96)
    print("ISOLATION SWEEP: fixed M, EVEN load, vary #active experts "
          "(does stock time track #active or stay flat?)")
    print("=" * 96)
    sweep_rows = []
    sweep_base = None
    for n_active in (256, 192, 140, 100):
        active = scattered_active(n_active)
        counts = counts_even(active, M_FIXED)
        n_empty = int((counts == 0).sum())
        x_sorted, re_sorted, M = make_stream(counts, x_pool)
        y = stock_call(x_sorted, re_sorted, q, s, b)
        mx.eval(y)
        ms = queued_median_ms(lambda: stock_call(x_sorted, re_sorted, q, s, b))
        if sweep_base is None:
            sweep_base = ms
        sweep_rows.append((n_active, n_empty, ms, ms / sweep_base))
        print(f"  active={n_active:>3} empty={n_empty:>3} ({n_empty/E*100:4.1f}%) "
              f"| stock {ms:8.4f} ms | vs active=256: {ms/sweep_base:5.3f}x")

    # ---- 3) Empty placement sensitivity (blocked vs scattered) ---------------
    print("\n" + "=" * 96)
    print("PLACEMENT: 192 active (64 empty), even load, blocked vs scattered empties")
    print("=" * 96)
    place_rows = []
    for name, ids in (("blocked-empties", blocked_active(192)),
                      ("scattered-empties", scattered_active(192))):
        counts = counts_even(np.sort(ids), M_FIXED)
        x_sorted, re_sorted, M = make_stream(counts, x_pool)
        ms = queued_median_ms(lambda: stock_call(x_sorted, re_sorted, q, s, b))
        place_rows.append((name, ms))
        print(f"  {name:<18} | stock {ms:8.4f} ms")

    # ---- 4) Hand RUNSKIP kernel: does empty-skip narrow the gap? -------------
    print("\n" + "=" * 96)
    print("HAND RUNSKIP KERNEL (empty-expert threadgroups early-return): "
          "balanced vs empty-heavy")
    print("=" * 96)
    hand_rows = []
    for label, n_active in (("balanced", 256), ("realistic", 192), ("worst", 141)):
        active = scattered_active(n_active)
        counts = counts_even(active, M_FIXED)
        n_empty = int((counts == 0).sum())
        x_sorted, re_sorted, M = make_stream(counts, x_pool)
        _, _, start, count = sorted_run_layout(re_sorted, E)
        mx.eval(start, count)
        assert is_grouped_gather_eligible(x_sorted, q, s, b, start, count)

        ref = stock_call(x_sorted, re_sorted, q, s, b)
        got = hand_call(x_sorted, re_sorted, q, s, b)
        mx.eval(ref, got)
        max_abs = float(mx.max(mx.abs(got - ref)))
        assert max_abs <= TOL_ABS, f"{label}: hand kernel wrong, max|diff|={max_abs:.2e}"

        stock_ms = queued_median_ms(lambda: stock_call(x_sorted, re_sorted, q, s, b))
        hand_ms = queued_median_ms(lambda: hand_call(x_sorted, re_sorted, q, s, b))
        hand_rows.append((label, n_active, n_empty, stock_ms, hand_ms, max_abs))
        print(f"  {label:<10} active={n_active:>3} empty={n_empty:>3} "
              f"| stock {stock_ms:8.4f} | hand {hand_ms:9.4f} "
              f"| gap {hand_ms/stock_ms:6.2f}x | max|diff| {max_abs:.2e}")

    # ------------------------------------------------------------------- tables
    print("\n" + "=" * 96)
    print("SUMMARY TABLE  (stock gather_qmm, fixed M=%d)" % M_FIXED)
    print("=" * 96)
    print(f"{'distribution':<16} | {'active':>6} | {'empty':>5} | {'empty%':>6} "
          f"| {'M':>6} | {'stock ms':>9} | {'vs balanced':>11}")
    print("-" * 96)
    for r in main_rows:
        base = baseline_ms[r["shape"]]
        print(f"{r['label']:<16} | {r['active']:>6} | {r['empty']:>5} "
              f"| {r['empty']/E*100:5.1f}% | {r['M']:>6} | {r['ms']:>9.4f} "
              f"| {r['ms']/base:>10.3f}x")

    # ------------------------------------------------------------------ verdict
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)

    # Compare even-family realistic/worst vs balanced.
    even = {r["label"]: r for r in main_rows if r["shape"] == "even"}
    bal = even["balanced-even"]["ms"]
    real = even["realistic-even"]["ms"]
    worst = even["worst-even"]["ms"]
    # Isolation sweep slope: ms at active=100 vs active=256.
    sw_full = sweep_rows[0][2]
    sw_100 = sweep_rows[-1][2]

    real_save = (bal - real) / bal * 100.0
    worst_save = (bal - worst) / bal * 100.0
    sweep_save = (sw_full - sw_100) / sw_full * 100.0

    print(f"stock @ balanced(256 active)   : {bal:8.4f} ms")
    print(f"stock @ realistic(192,64 empty): {real:8.4f} ms  "
          f"({real_save:+.1f}% vs balanced)")
    print(f"stock @ worst(141,115 empty)   : {worst:8.4f} ms  "
          f"({worst_save:+.1f}% vs balanced)")
    print(f"isolation active 256->100      : {sw_full:8.4f} -> {sw_100:8.4f} ms "
          f"({sweep_save:+.1f}%)")

    # Decision thresholds: if dropping ~25% of experts (64/256) barely moves time
    # (< ~5% change), stock is NOT compacting -> RUNSKIP headroom ~ empty fraction.
    # If time scales down roughly with active fraction, stock already compacts.
    real_empty_pct = even["realistic-even"]["empty"] / E * 100.0
    worst_empty_pct = even["worst-even"]["empty"] / E * 100.0
    flat = abs(real_save) < 5.0 and abs(sweep_save) < 10.0
    scales = sweep_save > 20.0
    print()
    if flat:
        print("STOCK TIME IS ~FLAT vs empty fraction at fixed M.")
        print("=> stock gather_qmm does NOT compact empty experts; it burns MMA on")
        print("   0-token expert slots. RUNSKIP headroom ~= empty fraction "
              f"(~{real_empty_pct:.0f}% typical, ~{worst_empty_pct:.0f}% worst).")
        print("   An ml-explore/mlx RUNSKIP fn-const COULD be worth it.")
    elif scales:
        print("STOCK TIME SCALES DOWN with fewer active experts at fixed M.")
        print("=> stock gather_qmm ALREADY skips/compacts empty experts.")
        print("   RUNSKIP would be REDUNDANT for MLX's steel gather GEMM.")
    else:
        print("STOCK TIME PARTIALLY tracks #active (between flat and linear).")
        print(f"   Realistic empty fraction (~25%) recovers ~{real_save:.1f}% already;")
        print("   residual RUNSKIP headroom is the gap to the empty fraction. See table.")

    print()
    print("HAND RUNSKIP-technique isolation (empty-skip vs full):")
    hb = hand_rows[0]   # balanced
    for label, na, ne, sms, hms, _ in hand_rows:
        print(f"  {label:<10}: hand {hms:9.4f} ms, gap-to-stock {hms/sms:5.2f}x, "
              f"hand-vs-hand-balanced {hms/hb[4]:5.3f}x")
    print("  (hand is scalar-FMA so it loses to stock's MMA in absolute terms; the")
    print("   hand-vs-hand-balanced column shows whether the empty-skip recovers")
    print("   work proportional to the empty fraction -- isolating RUNSKIP the technique.)")


if __name__ == "__main__":
    main()
