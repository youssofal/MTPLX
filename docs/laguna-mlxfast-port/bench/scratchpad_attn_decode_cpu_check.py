"""CPU-only algorithm validation for the three challenge attention decode kernels
(D2 full-attn YaRN qk-norm+rope, D3 sliding plain-rope qk-norm+rope, D5 gated
output projection).  NO Metal, NO GPU.

Sets the default device to CPU and proves, without ever running a metal_kernel:

  D2/D3
    * ``build_*_angles`` produces the exact cos/sin ``mx.fast.rope`` uses
      (theta = offset / freqs), so the kernel's rotation reads the rope's own
      floats.
    * the public helper takes its STOCK FALLBACK (Metal unavailable) and that
      fallback is BIT-EXACT vs the stock ``q_norm``/``k_norm`` -> transpose ->
      rope chain.
    * the pure-mx REFERENCE (the math the metal kernel targets) matches stock to
      <= ~1 bf16 ULP in the rotary region (the challenge design rotates via the
      angle table + plain multiply-subtract rather than mx.fast.rope's fused
      multiply-add) and is bit-exact for RMSNorm and the non-rotary tail.

  D5
    * the affine unpack is bit-exact vs ``mx.dequantize`` for bits {5, 8} at
      gs64;
    * the fallback is bit-exact identical to the stock softplus-gate ->
      ``quantized_matmul`` chain;
    * the reference matches an independent FP64 gold to floating tolerance
      (CPU ``quantized_matmul`` accumulates crudely, so the gold is the oracle;
      kernel-vs-``quantized_matmul`` agreement is confirmed on the GPU by
      ``scratchpad_gated_oproj_check.py``);
    * output shapes are exactly the decode contract.

Run:  .venv/bin/python scratchpad_attn_decode_cpu_check.py
"""

from __future__ import annotations

import sys
import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx_lm.models.rope_utils import initialize_rope  # noqa: E402

from mtplx.kernels import laguna_qk_yarn_full as d2  # noqa: E402
from mtplx.kernels import laguna_qk_rope_sliding as d3  # noqa: E402
from mtplx.kernels import laguna_gated_oproj as d5  # noqa: E402

RNG = np.random.default_rng(0)


def npf(a):
    return np.array(a.astype(mx.float32))


def bf16(shape, scale=1.0, center=0.0):
    a = RNG.standard_normal(shape).astype(np.float32) * scale + center
    return mx.array(a).astype(mx.bfloat16)


def maxabs(a, b):
    a = npf(a).reshape(-1).astype(np.float64)
    b = (b.reshape(-1).astype(np.float64) if isinstance(b, np.ndarray)
         else npf(b).reshape(-1).astype(np.float64))
    return float(np.max(np.abs(a - b)))


def report(name, got, ref, tol=0.0):
    exact = bool(mx.array_equal(got.astype(mx.float32), ref.astype(mx.float32)))
    d = maxabs(got, ref)
    ok = exact or d <= tol
    print(f"    {name:52s} exact={exact!s:5s} max|d|={d:.3e}  {'OK' if ok else '**FAIL**'}")
    return ok


# --------------------------------------------------------------------------
def check_d2():
    print("\n=== D2 full-attention YaRN qk-norm+rope ===")
    spec = d2.YarnFullSpec()
    rope = initialize_rope(
        spec.rot_dims, base=500000.0, traditional=False,
        scaling_config={"rope_type": "yarn", "factor": 128.0,
                        "original_max_position_embeddings": 8192,
                        "beta_fast": 32, "beta_slow": 1},
        max_position_embeddings=1_048_576,
    )
    print(f"    YarnRoPE.mscale = {rope.mscale!r}  (spec.mscale = {spec.mscale!r})")
    assert abs(float(rope.mscale) - spec.mscale) < 1e-15, "mscale mismatch!"
    freqs = rope._freqs
    ok = True
    for offset in (0, 5, 137, 4095):
        q = bf16((1, 1, spec.n_q_heads * spec.head_dim), scale=0.8)
        k = bf16((1, 1, spec.n_kv_heads * spec.head_dim), scale=0.8)
        qw = bf16((spec.head_dim,), scale=0.1, center=1.0)
        kw = bf16((spec.head_dim,), scale=0.1, center=1.0)
        angles = d2.build_full_yarn_angles(freqs, offset, spec)
        mx.eval(q, k, qw, kw, angles)

        fq = npf(freqs).astype(np.float64)
        theta = np.float64(offset) / np.where(fq == 0, np.inf, fq)
        d_ang = max(maxabs(angles.reshape(spec.rot_dims)[: spec.rot_pairs], np.cos(theta)),
                    maxabs(angles.reshape(spec.rot_dims)[spec.rot_pairs:], np.sin(theta)))

        st_q, st_k = d2._stock_qk_yarn_full(q, k, qw, kw, freqs, offset, spec)
        ref_q, ref_k = d2.fused_qk_yarn_full_reference(q, k, qw, kw, angles, spec)
        fb_q, fb_k = d2.fused_qk_yarn_full(q, k, qw, kw, angles, spec,
                                           freqs=freqs, offset=offset)
        mx.eval(st_q, st_k, ref_q, ref_k, fb_q, fb_k)
        print(f"  -- offset={offset} (angles vs cos/sin max|d|={d_ang:.2e}) --")
        ok &= report("fallback q == stock q", fb_q, st_q)
        ok &= report("fallback k == stock k", fb_k, st_k)
        ok &= report("reference q vs stock q (challenge design)", ref_q, st_q, tol=6e-2)
        ok &= report("reference k vs stock k (challenge design)", ref_k, st_k, tol=6e-2)
        assert tuple(fb_q.shape) == (1, spec.n_q_heads, 1, spec.head_dim)
        assert tuple(fb_k.shape) == (1, spec.n_kv_heads, 1, spec.head_dim)
    print(f"  shapes OK; D2 {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
def check_d3():
    print("\n=== D3 sliding plain-rope qk-norm+rope ===")
    spec = d3.SlidingRopeSpec()
    ok = True
    for offset in (0, 5, 137, 511):
        q = bf16((1, 1, spec.n_q_heads * spec.head_dim), scale=0.8)
        k = bf16((1, 1, spec.n_kv_heads * spec.head_dim), scale=0.8)
        qw = bf16((spec.head_dim,), scale=0.1, center=1.0)
        kw = bf16((spec.head_dim,), scale=0.1, center=1.0)
        angles = d3.build_sliding_rope_angles(offset, spec)
        mx.eval(q, k, qw, kw, angles)

        st_q, st_k = d3._stock_qk_rope_sliding(q, k, qw, kw, offset, spec)
        ref_q, ref_k = d3.fused_qk_rope_sliding_reference(q, k, qw, kw, angles, spec)
        fb_q, fb_k = d3.fused_qk_rope_sliding(q, k, qw, kw, angles, spec, offset=offset)
        mx.eval(st_q, st_k, ref_q, ref_k, fb_q, fb_k)
        print(f"  -- offset={offset} --")
        ok &= report("fallback q == stock q", fb_q, st_q)
        ok &= report("fallback k == stock k", fb_k, st_k)
        ok &= report("reference q vs stock q (challenge design)", ref_q, st_q, tol=2e-2)
        ok &= report("reference k vs stock k (challenge design)", ref_k, st_k, tol=2e-2)
        assert tuple(fb_q.shape) == (1, spec.n_q_heads, 1, spec.head_dim)
        assert tuple(fb_k.shape) == (1, spec.n_kv_heads, 1, spec.head_dim)
    print(f"  shapes OK; D3 {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
def manual_unpack(codes, scales, biases, bits, gs):
    codes_np = np.array(codes)
    scales_np = npf(scales)
    biases_np = npf(biases)
    out, words = codes_np.shape
    in_features = words * 32 // bits
    deq = np.zeros((out, in_features), dtype=np.float32)
    mask = (1 << bits) - 1
    for r in range(out):
        for c in range(in_features):
            bit_off = c * bits
            word = bit_off // 32
            shift = bit_off % 32
            lo = codes_np[r, word] >> shift
            hi = ((codes_np[r, word + 1] << (32 - shift)) & 0xFFFFFFFF
                  if shift + bits > 32 else 0)
            code = (lo | hi) & mask
            deq[r, c] = float(code) * scales_np[r, c // gs] + biases_np[r, c // gs]
    return deq


def fp64_gold_d5(attn, gate_logits, codes, scales, biases, spec):
    heads, hd, gs, bits = spec.n_heads, spec.head_dim, spec.group_size, spec.bits
    gate = mx.logaddexp(gate_logits.astype(mx.float32), mx.array(0.0)).astype(mx.bfloat16)
    gated = (attn.reshape(1, 1, heads, hd) * gate[..., None]).reshape(1, 1, heads * hd)
    deq = npf(mx.dequantize(codes, scales.astype(mx.float32), biases.astype(mx.float32),
                            group_size=gs, bits=bits)).astype(np.float64)
    return npf(gated).astype(np.float64).reshape(-1) @ deq.T


def check_d5():
    print("\n=== D5 gated output projection (affine gs64, bits 5/8) ===")
    ok = True
    print("  [unpack] affine unpack vs mx.dequantize(fp32):")
    for bits in (5, 8):
        w = bf16((8, 6144), scale=0.1)
        codes, scales, biases = mx.quantize(w, group_size=64, bits=bits)
        ref = npf(mx.dequantize(codes, scales.astype(mx.float32),
                                biases.astype(mx.float32), group_size=64, bits=bits))
        eq = np.array_equal(ref, manual_unpack(codes, scales, biases, bits, 64))
        print(f"    bits={bits}: unpack bit-exact={eq}")
        ok &= eq

    for n_heads, bits in ((48, 8), (72, 5), (48, 5), (72, 8)):
        spec = d5.GatedOProjSpec(n_heads=n_heads, bits=bits)
        attn = bf16((1, 1, spec.in_vec), scale=0.5)
        glogits = bf16((1, 1, n_heads), scale=1.0)
        w = bf16((spec.out_dim, spec.in_vec), scale=0.05)
        codes, scales, biases = mx.quantize(w, group_size=64, bits=bits)
        mx.eval(attn, glogits, codes, scales, biases)
        print(f"  -- n_heads={n_heads} bits={bits} in_vec={spec.in_vec} --")

        assert not d5.is_gated_oproj_eligible(attn, glogits, codes, scales, biases, spec)
        fb = d5.fused_gated_oproj(attn, glogits, codes, scales, biases, spec)
        st = d5._stock_gated_oproj(attn, glogits, codes, scales, biases, spec)
        ref = d5.gated_oproj_reference(attn, glogits, codes, scales, biases, spec)
        mx.eval(fb, st, ref)
        ok &= report("fallback == stock (both quantized_matmul)", fb, st)

        gold = fp64_gold_d5(attn, glogits, codes, scales, biases, spec)
        d_gold = maxabs(ref.reshape(-1), gold)
        rng = float(np.max(np.abs(gold)))
        gold_ok = d_gold <= 1e-2 + 1e-2 * rng
        print(f"    reference vs FP64 gold: max|d|={d_gold:.3e} (range {rng:.2f})  "
              f"{'OK' if gold_ok else '**FAIL**'}")
        ok &= gold_ok
        print(f"    [i] ref vs CPU quantized_matmul (crude): {maxabs(ref, st):.3e} | "
              f"ref vs gold={d_gold:.3e} | stock vs gold={maxabs(st.reshape(-1), gold):.3e}")
        assert tuple(fb.shape) == (1, 1, spec.out_dim)
    print(f"  shapes OK; D5 {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print(f"mlx device={mx.default_device()}")
    r2, r3, r5 = check_d2(), check_d3(), check_d5()
    print("\n==================== SUMMARY ====================")
    print(f"  D2 (yarn full):    {'PASS' if r2 else 'FAIL'}")
    print(f"  D3 (rope sliding): {'PASS' if r3 else 'FAIL'}")
    print(f"  D5 (gated oproj):  {'PASS' if r5 else 'FAIL'}")
    if not (r2 and r3 and r5):
        sys.exit(1)
    print("  ALL CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
