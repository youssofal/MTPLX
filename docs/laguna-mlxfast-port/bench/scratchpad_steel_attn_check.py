"""Real-shape correctness + timing for the P2 steel-attention prefill port.

Run this UNDER THE GPU FLOCK (it dispatches Metal kernels). It builds throwaway
Q/K/V tensors at the real Laguna S-2.1 prefill geometry and times a single
attention; it does NOT touch a live endpoint.

    python scratchpad_steel_attn_check.py

What it proves / measures, for BOTH S-2.1 head families at prefill contexts
1024 AND 8192, head_dim 128, bf16, scale = head_dim**-0.5:

  families     full_attention   : 48 q / 8 kv (gqa 6), causal, no window
               sliding_attention: 72 q / 8 kv (gqa 9), causal + window 512

  correctness  steel kernel vs stock mx.fast.scaled_dot_product_attention
               (the model's actual attention), same numeric class as the
               stock-vs-fp32-reference gap; plus a full 3-way vs the fp32
               naive reference at ctx 1024 (the 8192 fp32 score matrix is too
               large to materialize, so stock is the oracle there).
               Full layers compare against stock mask="causal" (what the model
               passes); sliding layers against the materialized boolean sliding
               mask create_attention_mask returns at N > window.

  timing       steel kernel vs stock SDPA on the QUEUED lane (the verdict lane;
               see "Queued vs eager Metal microbench") and the eager lane.
               ratio = stock_ms / kernel_ms  (>1 => kernel faster).

EXPECTATION (honest, not a prejudgement): MLX's stock fused SDPA at prefill is
the *steel* kernel doing both GEMMs with simdgroup-matrix (MMA) fragments. This
port reproduces the steel *algorithm* (tiling, KV staging, online softmax,
causal + sliding-window masking, GQA) but does its QK / PV with a cooperative
simd_sum layout, not MMA fragments -- because the steel MMA headers are not
reachable from mx.fast.metal_kernel. So this port is EXPECTED TO LOSE to stock
at prefill; the value is a correct, S-2.1-shaped reproduction whose gap is
measured, not assumed. MEASURE, then decide.
"""

from __future__ import annotations

import time

import mlx.core as mx

from mtplx.kernels.laguna_steel_attn import (
    attention_mask_bool,
    reference_masked_sdpa,
    steel_attention_prefill,
)

HEAD_DIM = 128
SCALE = HEAD_DIM ** -0.5

# name, hq, hk, window
FAMILIES = [
    ("full_attention  ", 48, 8, 0),
    ("sliding_attention", 72, 8, 512),
]
CONTEXTS = [1024, 8192]
REF_MAX_CTX = 1024   # fp32 naive reference only where the score matrix fits


def _maxrel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    d = float(mx.max(mx.abs(a - b)))
    scale = float(mx.max(mx.abs(b))) + 1e-9
    return d, d / scale


def _queued_ms(fn, iters=30, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _eager_ms(fn, iters=30, warmup=5):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    if not mx.metal.is_available():
        raise SystemExit("Metal not available; run this under the GPU flock.")
    mx.set_default_device(mx.gpu)
    print(f"device={mx.default_device()}  head_dim={HEAD_DIM}  scale={SCALE:.6f}")

    ok_all = True
    for fam, hq, hk, window in FAMILIES:
        for ctx in CONTEXTS:
            mx.random.seed(hq * 100003 + ctx)
            q = mx.random.normal((1, hq, ctx, HEAD_DIM)).astype(mx.bfloat16)
            k = mx.random.normal((1, hk, ctx, HEAD_DIM)).astype(mx.bfloat16)
            v = mx.random.normal((1, hk, ctx, HEAD_DIM)).astype(mx.bfloat16)
            mx.eval(q, k, v)

            # stock mask exactly as the model builds it for this family/ctx.
            if window > 0 and ctx > window:
                stock_mask = attention_mask_bool(
                    ctx, ctx, causal=True, window=window
                )[None, None]
            else:
                stock_mask = "causal"

            def kfn():
                return steel_attention_prefill(
                    q, k, v, scale=SCALE, causal=True, window=window
                )

            def sfn():
                return mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=SCALE, mask=stock_mask
                )

            out = kfn()
            st = sfn()
            assert out is not None, "kernel should be eligible at this shape"
            shape_ok = tuple(out.shape) == (1, hq, ctx, HEAD_DIM)
            mx.eval(out, st)

            ks_a, ks_r = _maxrel(out, st)     # kernel vs stock (bf16)
            rs_a = rs_r = kr_a = kr_r = None
            if ctx <= REF_MAX_CTX:
                ref = reference_masked_sdpa(
                    q, k, v, scale=SCALE, causal=True, window=window
                )
                mx.eval(ref)
                rs_a, rs_r = _maxrel(st, ref)   # stock vs fp32 reference
                kr_a, kr_r = _maxrel(out, ref)  # kernel vs fp32 reference

            # numeric-class gate: kernel within 4x the stock-vs-ref gap, or 5e-3.
            if rs_a is not None:
                corr_ok = shape_ok and kr_a <= max(5e-3, 4.0 * rs_a)
            else:
                corr_ok = shape_ok and ks_a <= 5e-3
            ok_all = ok_all and corr_ok

            print(f"\n[{fam} | ctx={ctx} | hq={hq} hk={hk} gqa={hq // hk} "
                  f"window={window}] shape={tuple(out.shape)} "
                  f"-> {'PASS' if corr_ok else 'FAIL'}")
            print(f"    kernel vs stock       : abs {ks_a:.3e}  rel {ks_r:.3e}")
            if rs_a is not None:
                print(f"    stock  vs fp32 ref    : abs {rs_a:.3e}  rel {rs_r:.3e}")
                print(f"    kernel vs fp32 ref    : abs {kr_a:.3e}  rel {kr_r:.3e}")

            kq, sq = _queued_ms(kfn), _queued_ms(sfn)
            ke, se = _eager_ms(kfn), _eager_ms(sfn)
            print(f"    queued  kernel {kq:8.4f} ms  stock {sq:8.4f} ms  "
                  f"ratio(stock/kernel) x{sq / kq:.3f}  (verdict lane)")
            print(f"    eager   kernel {ke:8.4f} ms  stock {se:8.4f} ms  "
                  f"ratio(stock/kernel) x{se / ke:.3f}")

            del q, k, v, out, st
            mx.clear_cache()

    print(f"\nCORRECTNESS: {'ALL PASS' if ok_all else 'FAIL'}")
    print("Timing ratios >1 mean the port beats stock; <1 mean it loses "
          "(expected at prefill -- see the header).")


if __name__ == "__main__":
    main()
