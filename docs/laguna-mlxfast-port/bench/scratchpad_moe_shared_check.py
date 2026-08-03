"""Real-shape correctness + timing for the D7 shared-expert SwiGLU-QMV kernel.

Run this UNDER THE GPU FLOCK (it dispatches Metal kernels). It does NOT hammer a
live endpoint; it builds one throwaway MoE block and times a single fused path.

    python scratchpad_moe_shared_check.py

What it proves / measures at the real Laguna S-2.1 shared-expert geometry
(hidden 3072, shared_intermediate 1024, affine 8-bit gs128):

  correctness  kernel  == reference   (both float32; fp32 accumulation-order noise)
               kernel  == stock MLP   (stock is bf16; delta at bf16 resolution)
               reference == stock MLP
               guarded drop-in == raw qmv, and eligibility is True on GPU
  timing       fused kernel vs stock ``MLP.__call__`` on the QUEUED lane
               (see the "Queued vs eager Metal microbench" finding: the eager
               lane's host-sync can invert the verdict, so promotion is decided
               on the queued lane; both are printed, queued is the verdict).

EXPECTATION (honest): at B=1 this kernel launches ONE threadgroup, so it should
LOSE to MLX's tuned ``quantized_matmul``. The batch sweep shows how the gap moves
with occupancy. This script measures the gap; it does not assume a win.
"""

from __future__ import annotations

import os
import time

import mlx.core as mx
import mlx.nn as nn

from mtplx.models.laguna import LagunaSparseMoeBlock, ModelArgs
from mtplx.kernels import laguna_moe_shared as D7

# Real Laguna S-2.1 geometry. E (expert count) only affects allocation, not the
# B=1 shared-expert work; lower it via LAGUNA_MOE_CHECK_E if GPU memory is tight.
HIDDEN = 3072
MOE_INTER = 1024
SHARED_INTER = 1024
NUM_EXPERTS = int(os.environ.get("LAGUNA_MOE_CHECK_E", "256"))
TOP_K = 10
ROWS_SWEEP = [1, 2, 4, 8]

SHARED_PATHS = {
    "shared_expert.gate_proj",
    "shared_expert.up_proj",
    "shared_expert.down_proj",
}
ROUTED_PATHS = {
    "switch_mlp.gate_proj",
    "switch_mlp.up_proj",
    "switch_mlp.down_proj",
}


def _class_predicate(path, _mod):
    if path in SHARED_PATHS:
        return {"group_size": 128, "bits": 8}
    if path in ROUTED_PATHS:
        return {"group_size": 128, "bits": 4}
    return False


def _build_block():
    args = ModelArgs(
        model_type="laguna",
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        intermediate_size=12288,
        num_attention_heads=48,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=256,
        rms_norm_eps=1e-6,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        moe_intermediate_size=MOE_INTER,
        shared_expert_intermediate_size=SHARED_INTER,
        decoder_sparse_step=1,
        norm_topk_prob=True,
        moe_routed_scaling_factor=2.5,
        mlp_only_layers=[0],
        gating="per-head",
        sliding_window=512,
        layer_types=["full_attention"],
    )
    block = LagunaSparseMoeBlock(args)
    nn.quantize(block, group_size=128, bits=4, class_predicate=_class_predicate)
    mx.eval(block.parameters())
    return block


def _maxrel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    d = float(mx.max(mx.abs(a - b)))
    scale = float(mx.max(mx.abs(b))) + 1e-9
    return d, d / scale


def _queued_ms(fn, iters=200, warmup=20):
    """Per-call ms on the queued lane: enqueue `iters` calls, one sync."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _eager_ms(fn, iters=200, warmup=20):
    """Per-call ms on the eager lane: sync after every call (host-sync heavy)."""
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    if not (mx.metal.is_available()):
        raise SystemExit("Metal not available; run this under the GPU flock.")
    mx.set_default_device(mx.gpu)
    print(f"device={mx.default_device()}  E={NUM_EXPERTS}  hidden={HIDDEN} "
          f"shared_inter={SHARED_INTER}  bits=8 gs=128")

    block = _build_block()
    se = block.shared_expert
    g, u, d = se.gate_proj, se.up_proj, se.down_proj

    ok_all = True
    for rows in ROWS_SWEEP:
        x = mx.random.normal((rows, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)

        elig = D7.is_shared_swiglu_eligible(se, x)
        kernel = D7.shared_swiglu_qmv(
            x,
            g["weight"], g["scales"], g["biases"],
            u["weight"], u["scales"], u["biases"],
            d["weight"], d["scales"], d["biases"],
            hidden=HIDDEN, shared_intermediate=SHARED_INTER,
        )
        reference = D7.shared_swiglu_reference(
            x,
            g["weight"], g["scales"], g["biases"],
            u["weight"], u["scales"], u["biases"],
            d["weight"], d["scales"], d["biases"],
            hidden=HIDDEN, shared_intermediate=SHARED_INTER,
        )
        stock = se(x)
        dropin = D7.shared_expert_swiglu(se, x)
        mx.eval(kernel, reference, stock, dropin)

        kr_a, kr_r = _maxrel(kernel, reference)      # f32 vs f32 (tight)
        ks_a, ks_r = _maxrel(kernel, stock)          # vs bf16 stock
        rs_a, rs_r = _maxrel(reference, stock)       # vs bf16 stock
        drop_ok = bool(mx.array_equal(dropin, kernel))

        shape_ok = tuple(kernel.shape) == (rows, HIDDEN)
        corr_ok = (elig and shape_ok and drop_ok
                   and kr_r < 5e-3 and ks_r < 3e-2 and rs_r < 3e-2)
        ok_all = ok_all and corr_ok
        print(f"\n[rows={rows}] eligible={elig} shape={tuple(kernel.shape)} "
              f"dropin==kernel={drop_ok}  -> {'PASS' if corr_ok else 'FAIL'}")
        print(f"    kernel vs reference : abs {kr_a:.3e}  rel {kr_r:.3e}")
        print(f"    kernel vs stock     : abs {ks_a:.3e}  rel {ks_r:.3e}")
        print(f"    reference vs stock  : abs {rs_a:.3e}  rel {rs_r:.3e}")

        kfn = lambda: D7.shared_swiglu_qmv(
            x,
            g["weight"], g["scales"], g["biases"],
            u["weight"], u["scales"], u["biases"],
            d["weight"], d["scales"], d["biases"],
            hidden=HIDDEN, shared_intermediate=SHARED_INTER,
        )
        sfn = lambda: se(x)
        kq, sq = _queued_ms(kfn), _queued_ms(sfn)
        ke, sea = _eager_ms(kfn), _eager_ms(sfn)
        print(f"    queued  kernel {kq:.4f} ms  stock {sq:.4f} ms  "
              f"speedup x{sq / kq:.3f}  (verdict lane)")
        print(f"    eager   kernel {ke:.4f} ms  stock {sea:.4f} ms  "
              f"speedup x{sea / ke:.3f}")

    print(f"\nCORRECTNESS: {'ALL PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
