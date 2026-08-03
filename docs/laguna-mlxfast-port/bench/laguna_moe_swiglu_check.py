"""Standalone correctness + queued-lane timing check for the fused routed-expert
SwiGLU-QMV kernel (mtplx/kernels/laguna_moe_swiglu.py) at the real Laguna S-2.1
MoE shape.

Builds a realistic affine 4-bit gs128 expert bank (256 experts, hidden 3072,
moe_intermediate 1024), picks top_k=10 indices, and compares the hand kernel to
the stock `SwitchGLU.__call__` for (a) max|diff| and (b) queued-lane median ms.

Run:
    cd <worktree> && PYTHONPATH="$PWD" <venv python> scratchpad_moe_check.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU

from mtplx.kernels.laguna_moe_swiglu import (
    is_routed_swiglu_eligible,
    routed_swiglu_qmv,
)

H, MI, E, TOPK = 3072, 1024, 256, 10
DTYPE = mx.bfloat16
POOL = 64            # distinct (x, idx) samples cycled during timing
TOL_ABS = 1e-2       # SwiGLU tolerance stated by the task


def build_bank():
    print(f"building bank: {E} experts, hidden {H}, moe_inter {MI}, 4-bit gs128 ...")
    sg = SwitchGLU(H, MI, E)
    nn.quantize(sg, group_size=128, bits=4)
    mx.eval(sg.parameters())
    return sg


def make_pool(n, tokens=1):
    """n distinct samples, each x=[tokens, H] and idx=[tokens, TOPK].

    Each token independently routes to top_k random experts (via argpartition of
    random scores), so per-expert token density matches a real router: with
    tokens*TOPK/E pairs on average per expert, this is what stock's sorted
    grouped-GEMM amortizes over and the per-token QMV kernel does not.
    """

    xs = [mx.random.normal((tokens, H)).astype(DTYPE) for _ in range(n)]
    idxs = []
    for _ in range(n):
        scores = mx.random.normal((tokens, E))
        idx = mx.argpartition(-scores, kth=TOPK - 1, axis=-1)[..., :TOPK]
        idxs.append(idx.astype(mx.uint32))
    mx.eval(xs, idxs)
    return xs, idxs


def kernel_call(sg, x, idx, threads):
    return routed_swiglu_qmv(
        x, idx,
        sg.gate_proj["weight"], sg.gate_proj["scales"], sg.gate_proj["biases"],
        sg.up_proj["weight"], sg.up_proj["scales"], sg.up_proj["biases"],
        sg.down_proj["weight"], sg.down_proj["scales"], sg.down_proj["biases"],
        hidden=H, moe_intermediate=MI, threads=threads,
    )


def numeric_check(sg, xs, idxs, threads, n=None):
    n = len(xs) if n is None else min(n, len(xs))
    tokens = int(xs[0].shape[0])
    max_abs = 0.0
    max_rel = 0.0
    for i in range(n):
        x, idx = xs[i], idxs[i]
        ref = sg(x, idx)                       # stock [tokens, TOPK, H] float32
        got = kernel_call(sg, x, idx, threads)
        mx.eval(ref, got)
        # fake-speedup guard: exactly one output row per (token, selected-expert)
        assert tuple(got.shape) == tuple(ref.shape) == (tokens, TOPK, H), (
            f"shape mismatch got={tuple(got.shape)} ref={tuple(ref.shape)}"
        )
        d = mx.abs(got - ref)
        denom = mx.maximum(mx.abs(ref), mx.array(1e-3))
        max_abs = max(max_abs, float(mx.max(d)))
        max_rel = max(max_rel, float(mx.max(d / denom)))
    return max_abs, max_rel


def queued_median_ms(make_call, n_per_batch=40, repeats=21):
    # warmup
    for _ in range(3):
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


# Prefill token counts to sweep. T=1 is the decode point; the rest fill the GPU
# with T*top_k threadgroups.
T_SWEEP = (1, 64, 256, 512, 1024)
# Thread-per-group candidates tried per T; the kernel's best is reported (fair
# best-case for the hand kernel). 1024 dropped at large T to bound runtime.
THREAD_CANDIDATES = (256, 512, 1024)


def _batch_for(tokens):
    """Distinct-sample pool size and per-batch call count, memory-bounded.

    Each output is tokens*TOPK*H*4 bytes; keep a timing batch under ~1.5 GB.
    """

    out_mb = tokens * TOPK * H * 4 / 1e6
    n_per_batch = max(6, min(40, int(1500 / max(out_mb, 1.0))))
    pool = min(16, max(4, n_per_batch))
    return pool, n_per_batch


def sweep_one(sg, tokens):
    pool, n_per_batch = _batch_for(tokens)
    repeats = 11
    xs, idxs = make_pool(pool, tokens=tokens)

    def call_stock(j):
        return sg(xs[j % pool], idxs[j % pool])

    stock_ms = queued_median_ms(call_stock, n_per_batch=n_per_batch, repeats=repeats)

    best = None  # (threads, ms, max_abs, max_rel)
    for threads in THREAD_CANDIDATES:
        if tokens >= 512 and threads == 1024:
            continue
        max_abs, max_rel = numeric_check(sg, xs, idxs, threads, n=min(4, pool))

        def call_kernel(j, t=threads):
            return kernel_call(sg, xs[j % pool], idxs[j % pool], t)

        k_ms = queued_median_ms(call_kernel, n_per_batch=n_per_batch, repeats=repeats)
        if best is None or k_ms < best[1]:
            best = (threads, k_ms, max_abs, max_rel)

    threads, k_ms, max_abs, max_rel = best
    return {
        "T": tokens,
        "stock_ms": stock_ms,
        "kernel_ms": k_ms,
        "threads": threads,
        "ratio": k_ms / stock_ms,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "pairs_per_expert": tokens * TOPK / E,
    }


def main():
    mx.random.seed(0)
    print("metal available:", mx.metal.is_available(), "| device:", mx.default_device())
    sg = build_bank()

    probe_x, probe_idx = make_pool(1, tokens=1)
    elig = is_routed_swiglu_eligible(sg, probe_x[0], probe_idx[0])
    print("eligible:", elig)
    if not elig:
        print("FAIL: kernel not eligible for the real S-2.1 shape")
        return

    print("\n=== prefill token-count sweep (top_k=10 of 256, moe_inter 1024, hidden 3072) ===")
    print("kernel maps ONE threadgroup per (token, selected-expert) = T*top_k groups\n")

    rows = []
    for tokens in T_SWEEP:
        r = sweep_one(sg, tokens)
        rows.append(r)
        print(
            f"  T={r['T']:>4} | stock {r['stock_ms']:8.4f} ms | "
            f"kernel {r['kernel_ms']:8.4f} ms (t={r['threads']:>4}) | "
            f"ratio(k/stock) {r['ratio']:6.3f}x | "
            f"{'WIN ' if r['ratio'] < 1.0 else 'loss'} | "
            f"max|diff| {r['max_abs']:.2e} [{'PASS' if r['max_abs'] <= TOL_ABS else 'FAIL'}]"
        )

    print("\n=== TABLE (T, yours ms, stock ms, ratio, win?) ===")
    print(f"{'T':>5} | {'yours(ms)':>10} | {'stock(ms)':>10} | {'ratio':>7} | win?  | pairs/expert")
    for r in rows:
        print(
            f"{r['T']:>5} | {r['kernel_ms']:>10.4f} | {r['stock_ms']:>10.4f} | "
            f"{r['ratio']:>6.3f}x | {'YES' if r['ratio'] < 1.0 else 'no ':>4} | "
            f"{r['pairs_per_expert']:>6.2f}"
        )

    any_win = any(r["ratio"] < 1.0 for r in rows)
    all_pass = all(r["max_abs"] <= TOL_ABS for r in rows)
    best_r = min(rows, key=lambda r: r["ratio"])
    print("\n=== VERDICT ===")
    print(f"allclose across all T: {'PASS' if all_pass else 'FAIL'} "
          f"(worst max|diff| {max(r['max_abs'] for r in rows):.2e}, tol {TOL_ABS:.0e})")
    print(f"best ratio: {best_r['ratio']:.3f}x at T={best_r['T']}")
    if any_win:
        print("VERDICT: there IS a token count where the affine SwiGLU-QMV beats stock.")
    else:
        print("VERDICT: the affine SwiGLU-QMV does NOT beat stock at ANY swept token count.")
        print("The per-token QMV re-reads each expert's weights per routed token; stock's")
        print("sorted grouped-GEMM amortizes that read across all tokens sharing an expert,")
        print("so its advantage GROWS with tokens-per-expert (prefill), not shrinks.")


if __name__ == "__main__":
    main()
