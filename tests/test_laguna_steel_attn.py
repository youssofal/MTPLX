"""Correctness + contract tests for the P2 steel-attention prefill port.

Two tiers:

* CPU tier (no Metal): proves the pure-mx references implement the ported steel
  algorithm correctly for BOTH S-2.1 head families (full-causal gqa 6, sliding
  gqa 9 window 512) at a small prefill shape --
  ``reference_online (flash) == reference_masked (naive) == stock mx.fast.SDPA``
  -- plus eligibility gating and the stock fallback. These run everywhere and
  never dispatch the kernel.

* Metal tier (``metal`` in the name, ``skipif(not METAL)``): the actual kernel
  vs the fp32 reference / stock SDPA. RUN THESE UNDER THE GPU FLOCK
  (``pytest -k metal`` while holding the flock); the bench-shape sweep and
  timing live in ``scratchpad_steel_attn_check.py``.
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from mtplx.kernels.laguna_steel_attn import (  # noqa: E402
    attention_mask_bool,
    is_steel_attention_eligible,
    reference_masked_sdpa,
    reference_online_sdpa,
    steel_attention_or_sdpa,
    steel_attention_prefill,
)

METAL = mx.metal.is_available()
HEAD_DIM = 128
SCALE = HEAD_DIM ** -0.5

# name, hq, hk, window
FAMILIES = [
    ("full", 48, 8, 0),
    ("sliding", 72, 8, 512),
]


def _maxabs(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


def _stock(q, k, v, mask):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE, mask=mask)


# --------------------------------------------------------------------------- #
# CPU tier: references == stock (both families, window inactive AND active)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fam,hq,hk,window", FAMILIES)
@pytest.mark.parametrize("seqlen", [40, 600])
def test_cpu_references_match_stock(fam, hq, hk, window, seqlen):
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        mx.random.seed(hq * 1000 + seqlen)
        q = mx.random.normal((1, hq, seqlen, HEAD_DIM)).astype(mx.float32)
        k = mx.random.normal((1, hk, seqlen, HEAD_DIM)).astype(mx.float32)
        v = mx.random.normal((1, hk, seqlen, HEAD_DIM)).astype(mx.float32)
        mx.eval(q, k, v)

        naive = reference_masked_sdpa(q, k, v, scale=SCALE, causal=True, window=window)
        flash = reference_online_sdpa(q, k, v, scale=SCALE, causal=True, window=window)
        keep = attention_mask_bool(seqlen, seqlen, causal=True, window=window)
        st = _stock(q, k, v, keep[None, None])
        mx.eval(naive, flash, st)

        assert _maxabs(flash, naive) < 2e-5   # flash recurrence == naive
        assert _maxabs(naive, st) < 2e-5      # reference == stock
        assert _maxabs(flash, st) < 2e-5

        # The full family (and the window-inactive sliding case) also equals the
        # stock "causal" string path the model actually passes.
        if window == 0 or seqlen <= window:
            st_causal = _stock(q, k, v, "causal")
            mx.eval(st_causal)
            assert _maxabs(naive, st_causal) < 2e-5
    finally:
        mx.set_default_device(prev)


def test_cpu_eligibility_gating():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        q = mx.random.normal((1, 48, 32, HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((1, 8, 32, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((1, 8, 32, HEAD_DIM)).astype(mx.bfloat16)
        # Shape logic is gated behind metal availability; only assert the
        # negative cases that must hold regardless of the box.
        assert is_steel_attention_eligible(q, k, v, causal=False) is False
        qd = mx.random.normal((1, 48, 32, 64)).astype(mx.bfloat16)
        kd = mx.random.normal((1, 8, 32, 64)).astype(mx.bfloat16)
        vd = mx.random.normal((1, 8, 32, 64)).astype(mx.bfloat16)
        assert is_steel_attention_eligible(qd, kd, vd, causal=True) is False  # d!=128
        qf = mx.random.normal((1, 48, 16, HEAD_DIM)).astype(mx.float32)
        kf = mx.random.normal((1, 8, 16, HEAD_DIM)).astype(mx.float32)
        vf = mx.random.normal((1, 8, 16, HEAD_DIM)).astype(mx.float32)
        assert is_steel_attention_eligible(qf, kf, vf, causal=True) is False  # fp32
        # smaller kL than qL is ineligible (causal prefill needs kL>=qL)
        qbig = mx.random.normal((1, 48, 40, HEAD_DIM)).astype(mx.bfloat16)
        ksmall = mx.random.normal((1, 8, 32, HEAD_DIM)).astype(mx.bfloat16)
        vsmall = mx.random.normal((1, 8, 32, HEAD_DIM)).astype(mx.bfloat16)
        assert is_steel_attention_eligible(qbig, ksmall, vsmall, causal=True) is False
    finally:
        mx.set_default_device(prev)


def test_cpu_fallback_is_exactly_stock():
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        qf = mx.random.normal((1, 48, 16, HEAD_DIM)).astype(mx.float32)
        kf = mx.random.normal((1, 8, 16, HEAD_DIM)).astype(mx.float32)
        vf = mx.random.normal((1, 8, 16, HEAD_DIM)).astype(mx.float32)
        mx.eval(qf, kf, vf)
        # non-causal is never covered -> kernel returns None -> stock fallback.
        assert steel_attention_prefill(qf, kf, vf, scale=SCALE, causal=False) is None
        fb = steel_attention_or_sdpa(
            qf, kf, vf, scale=SCALE, mask="causal", causal=False
        )
        st = _stock(qf, kf, vf, "causal")
        mx.eval(fb, st)
        assert _maxabs(fb, st) == 0.0
    finally:
        mx.set_default_device(prev)


# --------------------------------------------------------------------------- #
# Metal tier: the actual kernel. RUN UNDER THE GPU FLOCK.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not METAL, reason="requires Metal (run under the GPU flock)")
@pytest.mark.parametrize("fam,hq,hk,window", FAMILIES)
@pytest.mark.parametrize("seqlen", [40, 600])
def test_metal_kernel_matches_reference(fam, hq, hk, window, seqlen):
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        mx.random.seed(hq * 7 + seqlen)
        q = mx.random.normal((1, hq, seqlen, HEAD_DIM)).astype(mx.bfloat16)
        k = mx.random.normal((1, hk, seqlen, HEAD_DIM)).astype(mx.bfloat16)
        v = mx.random.normal((1, hk, seqlen, HEAD_DIM)).astype(mx.bfloat16)
        mx.eval(q, k, v)

        out = steel_attention_prefill(
            q, k, v, scale=SCALE, causal=True, window=window
        )
        assert out is not None
        assert tuple(out.shape) == (1, hq, seqlen, HEAD_DIM)

        ref = reference_masked_sdpa(q, k, v, scale=SCALE, causal=True, window=window)
        keep = attention_mask_bool(seqlen, seqlen, causal=True, window=window)
        st = _stock(q, k, v, keep[None, None])
        mx.eval(out, ref, st)

        # bf16 accumulation ordering: same numeric class as stock-vs-reference.
        stock_gap = _maxabs(st, ref)
        kernel_gap = _maxabs(out, ref)
        assert kernel_gap <= max(5e-3, 4.0 * stock_gap), (kernel_gap, stock_gap)
    finally:
        mx.set_default_device(previous)
