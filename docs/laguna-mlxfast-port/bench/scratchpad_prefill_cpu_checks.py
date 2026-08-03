"""CPU-only (mx.cpu) validation of the three prefill kernel references.

Validates the pure-mx references against the stock op chain (and, for P3/P5, an
independent emulation of the kernel arithmetic) without any Metal.  Run under the
CPU-only venv; no GPU/flock needed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root

import math
import mlx.core as mx
import mlx.nn as nn

mx.set_default_device(mx.cpu)
from mlx_lm.models.rope_utils import initialize_rope

from mtplx.kernels.laguna_prefill_qk_rope import (
    QkRopePrefillSpec,
    qk_norm_rope_prefill_reference,
    _stock_qk_norm_rope_prefill,
)
from mtplx.kernels.laguna_prefill_router import router_prefill_reference
from mtplx.kernels.laguna_prefill_moe_combine import moe_combine_prefill_reference

bf16 = mx.bfloat16
f32 = mx.float32
EPS = 1e-6
FAIL = []


def report(tag, ok, extra=""):
    print(("PASS " if ok else "FAIL ") + tag + ("  " + extra if extra else ""))
    if not ok:
        FAIL.append(tag)


def bf16_ulp(mag):
    # bf16 has 8-bit mantissa (7 stored); ulp near magnitude ~= mag / 128.
    return max(mag, 1.0) / 128.0


# --------------------------------------------------------------------------- P1
def check_p1():
    print("\n=== P1 prefill qk-norm + rope ===")
    mx.random.seed(1)
    T = 37  # T > 1, odd, to catch layout bugs
    B = 1
    full = initialize_rope(
        64, base=500000.0, traditional=False,
        scaling_config={"rope_type": "yarn", "factor": 128.0,
                        "original_max_position_embeddings": 8192,
                        "beta_fast": 32.0, "beta_slow": 1.0},
        max_position_embeddings=1048576,
    )
    full_spec = QkRopePrefillSpec(
        n_q_heads=48, n_kv_heads=8, head_dim=128, rot_dims=64,
        freqs=full._freqs, base=None, mscale=full.mscale,
    )
    sl_spec = QkRopePrefillSpec(
        n_q_heads=72, n_kv_heads=8, head_dim=128, rot_dims=128,
        freqs=None, base=10000.0, mscale=None,
    )

    for name, spec in (("FULL/yarn", full_spec), ("SLIDING/base", sl_spec)):
        # (a) STRICT: float32 activations remove bf16 rounding, isolating the
        # only remaining slack (cos/sin implementation) to ~1e-6 -> proves the
        # rope math, layout transpose, mscale placement and per-position offset
        # are correct.
        qf = (mx.random.normal((B, T, spec.n_q_heads * 128)) * 0.5).astype(f32)
        kf = (mx.random.normal((B, T, spec.n_kv_heads * 128)) * 0.5).astype(f32)
        qwf = (mx.random.normal((128,)) * 0.3 + 1.0).astype(f32)
        kwf = (mx.random.normal((128,)) * 0.3 + 1.0).astype(f32)
        for offset in (0, 512):
            sq, sk = _stock_qk_norm_rope_prefill(qf, kf, qwf, kwf, EPS, offset, spec)
            rq, rk = qk_norm_rope_prefill_reference(qf, kf, qwf, kwf, EPS, offset, spec)
            mx.eval(sq, sk, rq, rk)
            for tag2, s, r in (("q", sq, rq), ("k", sk, rk)):
                d = mx.abs(s - r)
                max_rel = (d.max() / mx.maximum(mx.abs(s).max(), 1.0)).item()
                report(
                    f"P1 {name} {tag2} off={offset} f32 ref==stock (strict)",
                    max_rel <= 1e-4,
                    f"max_rel={max_rel:.2e}",
                )
        # (b) bf16 activations: reference reproduces stock to bf16 precision.
        q = (mx.random.normal((B, T, spec.n_q_heads * 128)) * 0.5).astype(bf16)
        k = (mx.random.normal((B, T, spec.n_kv_heads * 128)) * 0.5).astype(bf16)
        qw = (mx.random.normal((128,)) * 0.3 + 1.0).astype(bf16)
        kw = (mx.random.normal((128,)) * 0.3 + 1.0).astype(bf16)
        for offset in (0, 512):
            sq, sk = _stock_qk_norm_rope_prefill(q, k, qw, kw, EPS, offset, spec)
            rq, rk = qk_norm_rope_prefill_reference(q, k, qw, kw, EPS, offset, spec)
            mx.eval(sq, sk, rq, rk)
            for tag2, s, r in (("q", sq, rq), ("k", sk, rk)):
                sf, rf = s.astype(f32), r.astype(f32)
                ok = mx.allclose(sf, rf, rtol=1e-2, atol=7e-2).item()
                maxd = mx.abs(sf - rf).max().item()
                exact = mx.mean((s == r).astype(f32)).item()
                report(
                    f"P1 {name} {tag2} off={offset} bf16 ref~=stock",
                    bool(ok),
                    f"maxabs={maxd:.3e} bitexact={exact:.3f}",
                )
        # all-rows-distinct: every position's q head-0 differs from position 0
        sq, _ = _stock_qk_norm_rope_prefill(q, k, qw, kw, EPS, 0, spec)
        rq, _ = qk_norm_rope_prefill_reference(q, k, qw, kw, EPS, 0, spec)
        mx.eval(sq, rq)
        for tag2, arr in (("stock", sq), ("ref", rq)):
            row0 = arr[0, 0, 0]
            distinct = all(
                not mx.allclose(arr[0, 0, ti], row0, atol=1e-3).item()
                for ti in range(1, T)
            )
            report(f"P1 {name} {tag2} all-rows-distinct", distinct)
        # cross-check: a scalar-offset stock rope really varies per position
        # (guards against the T=1 batched-rope broadcast trap re-appearing).


# --------------------------------------------------------------------------- P3
def _kernel_selection_emulation(logits, bias, top_k, normalize, scale):
    """Pure-mx emulation of the kernel's iterative top-k with lower-index ties."""
    scores = mx.sigmoid(logits)
    choice = scores + bias  # [M, E]
    M, E = choice.shape
    work = mx.array(choice)
    sel_idx = []
    sel_score = []
    ar = mx.arange(E, dtype=mx.int32)
    for _ in range(top_k):
        best_val = work.max(axis=-1, keepdims=True)  # [M,1]
        is_best = work == best_val
        # lowest index among ties: mask non-best to E, take min index
        masked_idx = mx.where(is_best, ar[None, :], mx.array(E, dtype=mx.int32))
        chosen = masked_idx.min(axis=-1)  # [M]
        sel_idx.append(chosen)
        sel_score.append(mx.take_along_axis(scores, chosen[:, None], axis=-1)[:, 0])
        # set chosen to -inf
        onehot = mx.arange(E, dtype=mx.int32)[None, :] == chosen[:, None]
        work = mx.where(onehot, mx.array(float("-inf")), work)
    idx = mx.stack(sel_idx, axis=-1).astype(mx.uint32)  # [M, top_k]
    w = mx.stack(sel_score, axis=-1)  # [M, top_k]
    if normalize:
        w = w / w.sum(axis=-1, keepdims=True)
    w = w * scale
    order = mx.argsort(idx, axis=-1)
    return mx.take_along_axis(idx, order, axis=-1), mx.take_along_axis(w, order, axis=-1)


def check_p3():
    print("\n=== P3 prefill router (sigmoid+bias+top-10) ===")
    mx.random.seed(2)
    E, K = 256, 10
    for M in (128, 1024):
        logits = (mx.random.normal((M, E)) * 2.0).astype(f32)
        bias = (mx.random.normal((E,)) * 0.1).astype(f32)
        ri, rw = router_prefill_reference(logits, bias, K, normalize=True, scale=1.0)
        ei, ew = _kernel_selection_emulation(logits, bias, K, True, 1.0)
        mx.eval(ri, rw, ei, ew)
        # set parity per token (both sorted ascending already)
        parity = mx.all(ri == ei).item()
        report(f"P3 M={M} selection set-parity ref-vs-kernel-emul", bool(parity))
        wd = mx.abs(rw.astype(f32) - ew.astype(f32)).max().item()
        report(f"P3 M={M} normalized weights match", wd < 1e-5, f"maxabs={wd:.2e}")
        # sanity: exactly K distinct experts per token, in-range
        uniq = mx.array([len(set(ri[i].tolist())) for i in range(min(M, 64))])
        report(f"P3 M={M} exactly {K} distinct experts/token",
               bool((uniq == K).all().item()))
        report(f"P3 M={M} indices in [0,{E})",
               bool((ri.astype(f32) < E).all().item() and (ri.astype(f32) >= 0).all().item()))


# --------------------------------------------------------------------------- P5
def _ty_partial_emulation(expert_out, weights, shared, residual, scaling):
    """Pure-mx emulation of the kernel's TY=min(8,K) partial-accumulator order."""
    M, K, H = expert_out.shape
    bf = expert_out.dtype
    w = (weights * scaling).astype(bf)  # bf16(w_f32 * scaling)
    TY = min(8, K)
    totals = [mx.zeros((M, H), dtype=bf) for _ in range(TY)]
    for r in range(K):
        prod = expert_out[:, r, :] * w[:, r:r + 1]  # bf16
        totals[r % TY] = prod + totals[r % TY]
    total = totals[0]
    for y in range(1, TY):
        total = totals[y] + total
    return (total + shared) + residual


def check_p5():
    print("\n=== P5 prefill MoE combine tail ===")
    mx.random.seed(3)
    H, K = 3072, 10
    scaling = 2.5
    for M in (128, 1024):
        eo = (mx.random.normal((M, K, H)) * 0.3).astype(bf16)
        w = mx.sigmoid(mx.random.normal((M, K)) * 1.0).astype(f32)
        w = w / w.sum(axis=-1, keepdims=True)  # normalized like P3 output
        shared = (mx.random.normal((M, H)) * 0.3).astype(bf16)
        resid = (mx.random.normal((M, H)) * 0.5).astype(bf16)
        ref = moe_combine_prefill_reference(eo, w, shared, resid, scaling)
        emul = _ty_partial_emulation(eo, w, shared, resid, scaling)
        mx.eval(ref, emul)
        d = mx.abs(ref.astype(f32) - emul.astype(f32))
        maxd = d.max().item()
        exact = mx.mean((ref == emul).astype(f32)).item()
        # TY-partial vs mx.sum ordering differs by <= a couple bf16 ulp on CPU.
        ok = maxd <= bf16_ulp(3.0) * 4
        report(f"P5 M={M} ref==kernel-emul(TY order)", ok,
               f"maxabs={maxd:.3e} bitexact={exact:.3f}")
        report(f"P5 M={M} output shape [{M},{H}]", tuple(ref.shape) == (M, H))


check_p1()
check_p3()
check_p5()
print("\n" + ("ALL CPU CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
sys.exit(1 if FAIL else 0)
