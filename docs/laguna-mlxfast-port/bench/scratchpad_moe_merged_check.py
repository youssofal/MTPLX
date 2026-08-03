"""Real-shape correctness + timing for the D9 merged routed+shared SwiGLU-QMV.

Run this UNDER THE GPU FLOCK (it dispatches Metal kernels). It does NOT hammer a
live endpoint; it builds one throwaway MoE block and times a single fused path.

    python scratchpad_moe_merged_check.py

What it proves / measures at the real Laguna S-2.1 MoE geometry (hidden 3072,
moe_intermediate 1024 routed 4-bit gs128, shared_intermediate 1024 shared 8-bit
gs128, 256 experts, top-10 -> 11 merged slots):

  correctness  merged combined == reference combined  (both float32)
               merged combined == stock routed+shared combine (stock is bf16)
               per-slot merged == per-slot reference (pre-scaled contributions)
               guarded drop-in == kernel path, eligibility True on GPU
  timing       fused merged kernel (+ the axis-1 sum) vs the stock two-step
               ``switch_mlp(x, idx)`` + ``shared_expert(x)`` + combine, on the
               QUEUED lane (the verdict lane; see the "Queued vs eager Metal
               microbench" finding). Both lanes printed.

EXPECTATION (honest): at B=1 the merge lights SLOTS = 11 threadgroups (10 routed
+ 1 shared). The routed-only sibling already lost ~25% at B=1; folding in the
shared expert adds one threadgroup, not occupancy, so this is expected to LOSE at
decode. This measures the gap; it does not assume a win.
"""

from __future__ import annotations

import os
import time

import mlx.core as mx
import mlx.nn as nn

from mtplx.models.laguna import (
    LagunaSparseMoeBlock,
    ModelArgs,
    _stock_moe_combine,
)
from mtplx.kernels import laguna_moe_merged as D9

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


def _route(block, x):
    """Reproduce LagunaSparseMoeBlock's real router -> (indices, weights)."""
    logits = block.gate(x).astype(mx.float32)
    if block.softcap and block.softcap > 0.0:
        logits = mx.tanh(logits / block.softcap) * block.softcap
    scores = mx.sigmoid(logits)
    scores_for_choice = scores + block.e_score_correction_bias.astype(mx.float32)
    indices = mx.argpartition(-scores_for_choice, kth=TOP_K - 1, axis=-1)[..., :TOP_K]
    weights = mx.take_along_axis(scores, indices, axis=-1)
    if block.norm_topk_prob:
        weights = weights / weights.sum(axis=-1, keepdims=True)
    weights = (weights * block.routed_scaling_factor).astype(x.dtype)
    return indices, weights


def _maxrel(a, b):
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    d = float(mx.max(mx.abs(a - b)))
    scale = float(mx.max(mx.abs(b))) + 1e-9
    return d, d / scale


def _queued_ms(fn, iters=200, warmup=20):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize()
    t0 = time.perf_counter()
    outs = [fn() for _ in range(iters)]
    mx.eval(outs)
    mx.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def _eager_ms(fn, iters=200, warmup=20):
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
    print(f"device={mx.default_device()}  E={NUM_EXPERTS}  hidden={HIDDEN}  "
          f"moe_inter={MOE_INTER} shared_inter={SHARED_INTER}  top_k={TOP_K} "
          f"slots={TOP_K + 1}")

    block = _build_block()
    sm, se = block.switch_mlp, block.shared_expert
    g4, u4, d4 = sm.gate_proj, sm.up_proj, sm.down_proj
    g8, u8, d8 = se.gate_proj, se.up_proj, se.down_proj

    def _qmv(x, indices, weights):
        return D9.merged_swiglu_qmv(
            x, indices, weights,
            g4["weight"], g4["scales"], g4["biases"],
            u4["weight"], u4["scales"], u4["biases"],
            d4["weight"], d4["scales"], d4["biases"],
            g8["weight"], g8["scales"], g8["biases"],
            u8["weight"], u8["scales"], u8["biases"],
            d8["weight"], d8["scales"], d8["biases"],
            hidden=HIDDEN, moe_intermediate=MOE_INTER,
            shared_intermediate=SHARED_INTER,
        )

    ok_all = True
    for rows in ROWS_SWEEP:
        x = mx.random.normal((rows, HIDDEN)).astype(mx.bfloat16)
        mx.eval(x)
        indices, weights = _route(block, x)
        mx.eval(indices, weights)

        elig = D9.is_merged_swiglu_eligible(sm, se, x, indices, weights)
        slots = _qmv(x, indices, weights)
        merged = slots.sum(axis=1)
        slots_ref, comb_ref = D9.merged_swiglu_reference(
            x, indices, weights,
            g4["weight"], g4["scales"], g4["biases"],
            u4["weight"], u4["scales"], u4["biases"],
            d4["weight"], d4["scales"], d4["biases"],
            g8["weight"], g8["scales"], g8["biases"],
            u8["weight"], u8["scales"], u8["biases"],
            d8["weight"], d8["scales"], d8["biases"],
            hidden=HIDDEN, moe_intermediate=MOE_INTER,
            shared_intermediate=SHARED_INTER,
        )
        routed_stock = sm(x, indices)
        comb_stock = _stock_moe_combine(routed_stock, weights, se(x))
        dropin = D9.merged_expert_swiglu(sm, se, x, indices, weights)
        mx.eval(slots, merged, slots_ref, comb_ref, comb_stock, dropin)

        kr_a, kr_r = _maxrel(merged, comb_ref)          # f32 vs f32 (tight)
        ks_a, ks_r = _maxrel(merged, comb_stock)        # vs bf16 stock combine
        rs_a, rs_r = _maxrel(comb_ref, comb_stock)      # vs bf16 stock combine
        sl_a, sl_r = _maxrel(slots, slots_ref)          # per-slot (tight)
        drop_ok = bool(mx.array_equal(dropin, merged))

        shape_ok = tuple(slots.shape) == (rows, TOP_K + 1, HIDDEN)
        corr_ok = (elig and shape_ok and drop_ok
                   and kr_r < 5e-3 and sl_r < 5e-3 and ks_r < 3e-2 and rs_r < 3e-2)
        ok_all = ok_all and corr_ok
        print(f"\n[rows={rows}] eligible={elig} slots={tuple(slots.shape)} "
              f"dropin==merged={drop_ok}  -> {'PASS' if corr_ok else 'FAIL'}")
        print(f"    combined kernel vs reference : abs {kr_a:.3e}  rel {kr_r:.3e}")
        print(f"    combined kernel vs stock     : abs {ks_a:.3e}  rel {ks_r:.3e}")
        print(f"    combined reference vs stock  : abs {rs_a:.3e}  rel {rs_r:.3e}")
        print(f"    per-slot kernel vs reference : abs {sl_a:.3e}  rel {sl_r:.3e}")

        kfn = lambda: _qmv(x, indices, weights).sum(axis=1)
        sfn = lambda: _stock_moe_combine(sm(x, indices), weights, se(x))
        kq, sq = _queued_ms(kfn), _queued_ms(sfn)
        ke, sea = _eager_ms(kfn), _eager_ms(sfn)
        print(f"    queued  merged {kq:.4f} ms  stock {sq:.4f} ms  "
              f"speedup x{sq / kq:.3f}  (verdict lane)")
        print(f"    eager   merged {ke:.4f} ms  stock {sea:.4f} ms  "
              f"speedup x{sea / ke:.3f}")

    print(f"\nCORRECTNESS: {'ALL PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
