"""Check + benchmark harness for laguna_route_csort.

Answers: does a stable counting sort beat mx.argsort on the MoE route table at
Laguna prefill sizes, and is it exactly argsort-identical (stable)?

Route table = flattened top-10 expert-ids for T tokens over 256 experts:
    M = T * top_k uint32 keys in [0, 256).
    T in {256, 512, 1024} -> M in {2560, 5120, 10240}.
    (Decode T=1 -> M=10 is trivial and not a whole 128-key tile; noted, skipped.)

Timing is the queued lane: warmup, then many iters each scheduled with
mx.async_eval (no per-iter host sync -> kernels queue back-to-back), one
mx.synchronize per repeat, median across repeats.

Run:
    cd <worktree> && PYTHONPATH="$PWD" <venv python> scratchpad_route_csort_check.py
"""

from __future__ import annotations

import statistics
import time

import mlx.core as mx

from mtplx.kernels.laguna_route_csort import (
    is_route_csort_eligible,
    route_counting_sort,
)

TOP_K = 10
EXPERTS = 256
T_SIZES = [256, 512, 1024]
CORRECTNESS_TRIALS = 8
WARMUP = 25
ITERS = 300
REPEATS = 11


def make_route_table(T: int, seed: int) -> mx.array:
    """Flattened top-k route table: M = T*TOP_K uint32 keys in [0, EXPERTS).

    Uniform random over experts — the hardest tie stress for the stability
    check (many equal keys across tokens), and distribution-independent for the
    sort's cost.
    """
    key = mx.random.key(seed)
    keys = mx.random.randint(0, EXPERTS, shape=(T * TOP_K,), key=key)
    keys = keys.astype(mx.uint32)
    mx.eval(keys)
    return keys


def check_correctness(T: int) -> dict:
    """Verify valid permutation, sorted-key match, and argsort-identity."""
    all_valid_perm = True
    all_keys_match = True
    all_argsort_identical = True
    for trial in range(CORRECTNESS_TRIALS):
        keys = make_route_table(T, seed=1000 + trial)
        M = int(keys.shape[0])
        assert is_route_csort_eligible(keys), "expected eligible at prefill size"

        order = route_counting_sort(keys)
        mx.eval(order)

        # 1) valid permutation: sorted order == arange(M)
        perm_ok = bool(mx.all(mx.sort(order) == mx.arange(M, dtype=order.dtype)).item())
        all_valid_perm &= perm_ok

        # 2) gathered keys are the fully sorted multiset
        gathered = keys[order]
        keys_ok = bool(mx.all(gathered == mx.sort(keys)).item())
        all_keys_match &= keys_ok

        # 3) exactly argsort-identical (same permutation, ties included)
        arg = mx.argsort(keys).astype(order.dtype)
        identical = bool(mx.all(order == arg).item())
        all_argsort_identical &= identical

    return {
        "valid_perm": all_valid_perm,
        "keys_match": all_keys_match,
        "argsort_identical": all_argsort_identical,
    }


def bench(fn) -> float:
    """Queued-lane median ms/call: async_eval per iter, one sync per repeat."""
    for _ in range(WARMUP):
        mx.eval(fn())
    mx.synchronize()

    per_repeat = []
    for _ in range(REPEATS):
        mx.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            mx.async_eval(fn())
        mx.synchronize()
        t1 = time.perf_counter()
        per_repeat.append((t1 - t0) / ITERS * 1000.0)
    return statistics.median(per_repeat)


def main() -> None:
    print(f"mlx {mx.__version__}  metal={mx.metal.is_available()}")
    print(f"top_k={TOP_K} experts={EXPERTS} "
          f"warmup={WARMUP} iters={ITERS} repeats={REPEATS}\n")

    # --- note the decode case explicitly ---
    dec = make_route_table(1, seed=7)
    print(f"decode T=1 -> M={int(dec.shape[0])}: "
          f"eligible={is_route_csort_eligible(dec)} "
          f"(not a whole 128-tile; falls back to argsort — trivial)\n")

    header = (f"{'M':>7} | {'argsort ms':>11} | {'csort ms':>10} | "
              f"{'ratio(as/cs)':>12} | {'argsort-identical?':>18}")
    print(header)
    print("-" * len(header))

    rows = []
    for T in T_SIZES:
        M = T * TOP_K
        corr = check_correctness(T)

        keys = make_route_table(T, seed=42)
        as_ms = bench(lambda: mx.argsort(keys))
        cs_ms = bench(lambda: route_counting_sort(keys))
        ratio = as_ms / cs_ms

        ident = "YES" if corr["argsort_identical"] else "NO"
        if not (corr["valid_perm"] and corr["keys_match"]):
            ident += " (INVALID!)"

        print(f"{M:>7} | {as_ms:>11.4f} | {cs_ms:>10.4f} | "
              f"{ratio:>12.3f} | {ident:>18}")
        rows.append((M, as_ms, cs_ms, ratio, corr))

    print()
    # correctness detail
    for M, _, _, _, corr in rows:
        print(f"M={M}: valid_perm={corr['valid_perm']} "
              f"keys_match={corr['keys_match']} "
              f"argsort_identical={corr['argsort_identical']} "
              f"({CORRECTNESS_TRIALS} random trials)")


if __name__ == "__main__":
    main()
