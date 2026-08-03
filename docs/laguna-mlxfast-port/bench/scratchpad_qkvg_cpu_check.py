"""CPU-only algorithm validation for laguna_qkvg_fused (NO Metal, NO GPU).

Sets the default device to CPU and proves, without ever running the
metal_kernel:

  1. Affine unpack formula (the kernel's inline dequant) is bit-exact vs
     mx.dequantize for bits in {5, 8} at group_size 64.
  2. On CPU the fused helper takes its STOCK FALLBACK (metal unavailable), and
     that fallback is bit-exact identical to the stock rms_norm + 4x
     quantized_matmul chain.
  3. The pure-mx REFERENCE (the math the metal kernel targets) matches an
     independent FP64 gold to floating tolerance -> the algorithm (norm +
     affine unpack + projection) is correct.
  4. Output shapes are exactly the (queries, keys, values, gate_logits)
     contract for both head counts.

The reference is deliberately compared against the FP64 gold, not against CPU
mx.quantized_matmul: CPU quantized_matmul accumulates crudely (~1.0 abs error
vs the FP64 gold), so it is a poor oracle here.  The kernel accumulates in
FP32, like the reference; kernel-vs-quantized_matmul agreement is confirmed on
the GPU by scratchpad_qkvg_check.py under the flock.

Run:  .venv/bin/python scratchpad_qkvg_cpu_check.py
"""

from __future__ import annotations

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)

from mtplx.kernels.laguna_qkvg_fused import (  # noqa: E402
    QKVGSpec,
    fused_input_norm_qkvg,
    fused_input_norm_qkvg_reference,
    is_qkvg_fused_eligible,
    _stock_qkvg,
)

HIDDEN = 3072
KV_HEADS = 8
HEAD_DIM = 128
GS = 64
EPS = 1e-6
RNG = np.random.default_rng(0)


def npf(a: mx.array) -> np.ndarray:
    return np.array(a.astype(mx.float32))


def bf16(shape, scale=1.0, center=0.0):
    a = (RNG.standard_normal(shape).astype(np.float32) * scale + center)
    return mx.array(a).astype(mx.bfloat16)


def quantize_bf16(rows, bits):
    w = bf16((rows, HIDDEN), scale=0.05)
    return mx.quantize(w, group_size=GS, bits=bits)


def manual_unpack(codes, scales, biases, bits):
    """Independent reimplementation of the kernel's inline affine unpack."""
    codes_np = np.array(codes)  # uint32
    scales_np = npf(scales)
    biases_np = npf(biases)
    out, words = codes_np.shape
    in_features = words * 32 // bits
    n_groups = in_features // GS
    deq = np.zeros((out, in_features), dtype=np.float32)
    mask = (1 << bits) - 1
    for r in range(out):
        for c in range(in_features):
            bit_off = c * bits
            word = bit_off // 32
            shift = bit_off % 32
            lo = codes_np[r, word] >> shift
            hi = (codes_np[r, word + 1] << (32 - shift)) & 0xFFFFFFFF if shift + bits > 32 else 0
            code = (lo | hi) & mask
            deq[r, c] = float(code) * scales_np[r, c // GS] + biases_np[r, c // GS]
    return deq


def fp64_gold(hidden, norm_weight, banks):
    """Independent FP64 gold: manual-verified FP32 dequant, FP64 accumulate."""
    x = npf(hidden).astype(np.float64)
    inv = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + EPS)
    # RMSNorm mirrors the kernel: cast (raw*inv) to bf16, then * weight (bf16).
    t = mx.array((x * inv).astype(np.float32)).astype(mx.bfloat16)
    normed = npf(norm_weight * t).astype(np.float64)
    outs = []
    for codes, scales, biases, bits in banks:
        deq = npf(
            mx.dequantize(
                codes, scales.astype(mx.float32), biases.astype(mx.float32),
                group_size=GS, bits=bits,
            )
        ).astype(np.float64)
        outs.append(normed @ deq.T)
    return outs


def maxabs(a, b):
    a = npf(a).reshape(-1)
    b = b.reshape(-1) if isinstance(b, np.ndarray) else npf(b).reshape(-1)
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def check_layer(name, n_heads, bits):
    print(f"\n=== {name}: n_heads={n_heads}, bits={bits} ===")
    spec = QKVGSpec(n_heads=n_heads, bits=bits)
    query_rows = n_heads * HEAD_DIM
    kv_rows = KV_HEADS * HEAD_DIM

    hidden = bf16((1, 1, HIDDEN), scale=0.8)
    norm_weight = bf16((HIDDEN,), scale=0.1, center=1.0)
    qb = quantize_bf16(query_rows, bits)
    kb = quantize_bf16(kv_rows, bits)
    vb = quantize_bf16(kv_rows, bits)
    gb = quantize_bf16(n_heads, bits)
    banks = (
        qb[0], qb[1], qb[2],
        kb[0], kb[1], kb[2],
        vb[0], vb[1], vb[2],
        gb[0], gb[1], gb[2],
    )
    args = (hidden, norm_weight, *banks, EPS, spec)

    # (2a) The gate ACCEPTS the real [1,1,H] decode shape (Metal is available on
    # this box).  We assert eligibility but do NOT run the kernel here -- the
    # GPU dispatch is exercised only by the flocked scratchpad_qkvg_check.py.
    assert is_qkvg_fused_eligible(hidden, norm_weight, *banks, spec), \
        "gate should accept the real decode shape"
    print("  [2a] gate accepts real [1,1,H] decode shape (kernel NOT run here): OK")

    # (2b) Fallback WIRING: force the stock path with an ineligible shape (T=2,
    # rows != 1) so no Metal kernel is built or dispatched, and prove the
    # helper's fallback is bit-exact identical to the stock chain.
    hidden2 = mx.concatenate([hidden, hidden * mx.array(0.5).astype(mx.bfloat16)], axis=1)
    assert tuple(hidden2.shape) == (1, 2, HIDDEN)
    assert not is_qkvg_fused_eligible(hidden2, norm_weight, *banks, spec), \
        "T=2 must be ineligible (decode-only gate)"
    fb = fused_input_norm_qkvg(hidden2, norm_weight, *banks, EPS, spec)
    st = _stock_qkvg(hidden2, norm_weight, *banks, EPS, spec)
    for nm, f, s in zip(("q", "k", "v", "g"), fb, st):
        assert mx.array_equal(f, s), f"fallback {nm} != stock chain"
    exp2 = [(1, 2, query_rows), (1, 2, kv_rows), (1, 2, kv_rows), (1, 2, n_heads)]
    for nm, f, e in zip(("q", "k", "v", "g"), fb, exp2):
        assert tuple(f.shape) == e, f"fallback {nm} shape {f.shape} != {e}"
    print("  [2b] fallback path bit-exact == stock chain, shapes OK")

    # RMSNorm bit-exactness of the reference vs mx.fast.rms_norm.
    x = hidden.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + EPS)
    normed_ref = norm_weight * (x * inv).astype(mx.bfloat16)
    normed_stock = mx.fast.rms_norm(hidden, norm_weight, EPS)
    assert mx.array_equal(normed_ref, normed_stock), "reference RMSNorm != mx.fast.rms_norm"
    print("  [3a] reference RMSNorm bit-exact == mx.fast.rms_norm: OK")

    # (3b) reference vs independent FP64 gold, and (4) shape contract.
    ref = fused_input_norm_qkvg_reference(*args)
    exp = [(1, 1, query_rows), (1, 1, kv_rows), (1, 1, kv_rows), (1, 1, n_heads)]
    for nm, f, e in zip(("q", "k", "v", "g"), ref, exp):
        assert tuple(f.shape) == e, f"reference {nm} shape {f.shape} != {e}"
        assert f.dtype == mx.bfloat16
    print(f"  [4] reference output shapes {exp} dtype bf16: OK")
    golds = fp64_gold(
        hidden, norm_weight,
        [(qb[0], qb[1], qb[2], bits), (kb[0], kb[1], kb[2], bits),
         (vb[0], vb[1], vb[2], bits), (gb[0], gb[1], gb[2], bits)],
    )
    for nm, r, g in zip(("q", "k", "v", "g"), ref, golds):
        d = maxabs(r, g)
        rng = float(np.max(np.abs(g)))
        assert d <= 1e-2 + 1e-2 * rng, f"reference {nm} vs FP64 gold too far: {d} (range {rng})"
        print(f"  [3b] reference {nm} vs FP64 gold: max|.|={d:.3e} (range {rng:.2f}) OK")

    # Informational: reference vs CPU quantized_matmul, and why they differ.
    stock11 = _stock_qkvg(*args)  # [1,1,H] stock chain, pure mx on CPU
    print("  [i] reference-vs-stock (CPU quantized_matmul) and each vs FP64 gold:")
    for nm, r, s, g in zip(("q", "k", "v", "g"), ref, stock11, golds):
        print(
            f"       {nm}: ref-vs-stock max={maxabs(r, s):.3e} | "
            f"ref-vs-gold={maxabs(r, g):.3e} | stock-vs-gold={maxabs(s, g):.3e}"
        )


def check_unpack():
    print("=== [1] affine unpack formula vs mx.dequantize (small matrix) ===")
    for bits in (5, 8):
        w = bf16((4, HIDDEN), scale=0.1)
        codes, scales, biases = mx.quantize(w, group_size=GS, bits=bits)
        # Compare against FP32-dequant (scales/biases upcast) -- the exact
        # arithmetic the kernel does (float(code)*float(scale)+float(bias)).
        ref = npf(
            mx.dequantize(
                codes, scales.astype(mx.float32), biases.astype(mx.float32),
                group_size=GS, bits=bits,
            )
        )
        mine = manual_unpack(codes, scales, biases, bits)
        assert np.array_equal(ref, mine), f"unpack mismatch bits={bits}"
        print(f"  bits={bits}: unpack bit-exact vs mx.dequantize(fp32): OK")


def main():
    assert not mx.metal.is_available() or mx.default_device() == mx.cpu
    print(f"mlx {mx.__version__}  device={mx.default_device()}\n")
    check_unpack()
    check_layer("full_attention layer", n_heads=48, bits=8)
    check_layer("sliding_attention layer", n_heads=72, bits=5)
    # cross: sliding with 8-bit and full with 5-bit too, for coverage
    check_layer("full_attention layer (5-bit)", n_heads=48, bits=5)
    check_layer("sliding_attention layer (8-bit)", n_heads=72, bits=8)
    print("\nALL CPU CHECKS PASSED")


if __name__ == "__main__":
    main()
