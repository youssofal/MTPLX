"""M2 regression gates for the DeepSeek-V4 new-math components.

Each of the four pieces V4 adds over V3.2 — Hyper-Connections (Sinkhorn),
Compressed-Sparse-Attention pooling, grouped output-LoRA, and hash routing — is
checked here against a self-contained NumPy transcription of the authoritative
reference (``deepseek-ai/DeepSeek-V4-Flash/inference/model.py`` +
``inference/kernel.py``), on small synthetic tensors.

Provenance / how the bound is justified:
  * The MLX implementation was first gated *directly against the reference torch
    classes* (the shipped ``inference/model.py`` driven on CPU with a pure-torch
    stub for the tilelang/CUDA kernels).  That gate — reproduced in the port's
    scratchpad ``gate_m2.py`` — passed formula-exact:
        sinkhorn pre/post/comb  max_abs ~1e-7
        hc_pre.y / hc_post      max_abs ~2e-7
        compressor.kv           max_abs ~7e-7
        o-LoRA                  max_abs  0.0 (bit-identical)
        MoE gate (hash+score)   index sets identical, weights max_abs ~3e-8
  * MLX's GPU fp32 matmul uses a reduced-precision fast path (~7.5e-4 relative vs
    true fp32, still sub-bf16); MLX **CPU** fp32 is bit-identical to IEEE fp32.
    This test therefore pins MLX to the CPU device so a tight bound isolates the
    algebra rather than hardware matmul precision.
This file is the always-on regression guard (NumPy only, no torch, no download);
the reference-class gate is the primary evidence.
"""
import importlib.util
import math
import os

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

@pytest.fixture(autouse=True)
def _cpu_default_device():
    # CPU-pinned by design, but the pin must stay test-scoped: a module-level
    # set_default_device leaks into every later-collected module (pytest
    # imports all test modules before running any) and flips the engine's
    # Metal bit-exactness suites onto CPU fallbacks process-wide.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "..", "mtplx", "models", "deepseek_v4.py")
_spec = importlib.util.spec_from_file_location("dsv4_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
import sys  # noqa: E402

sys.modules["dsv4_undertest"] = D
_spec.loader.exec_module(D)

RTOL, ATOL = 1e-5, 1e-6


def t2m(a):
    return mx.array(np.asarray(a, dtype=np.float32))


def m2n(a):
    return np.array(a.astype(mx.float32))


# --------------------------------------------------------------------------- oracles
def np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def np_softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def np_sinkhorn(mixes, scale, base, hc, iters, eps):
    """kernel.py L371-427 transcription."""
    mixes = mixes.astype(np.float64)
    pre = np_sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2.0 * np_sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]
    comb = comb.reshape(*comb.shape[:-1], hc, hc)
    comb = np_softmax(comb, axis=-1) + eps
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


def np_yarn_inv_freq(dim, base, orig, factor, bf, bs):
    """model.py precompute_freqs_cis (freq part) transcription."""
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    if orig and orig > 0:
        def cdim(nr):
            return dim * math.log(orig / (nr * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(cdim(bf)), 0)
        high = min(math.ceil(cdim(bs)), dim - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(dim // 2) - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    return freqs


def np_rope_interleaved(x, cos, sin):
    x0 = x[..., 0::2]
    x1 = x[..., 1::2]
    r0 = x0 * cos - x1 * sin
    r1 = x0 * sin + x1 * cos
    out = np.empty_like(x)
    out[..., 0::2] = r0
    out[..., 1::2] = r1
    return out


# --------------------------------------------------------------------------- tests
def test_yarn_inv_freq_matches_reference_formula():
    ref = np_yarn_inv_freq(64, 160000.0, 65536, 16.0, 32, 1)
    got = m2n(D._yarn_inv_freq(64, 160000.0, 65536, 16.0, 32, 1)).astype(np.float64)
    assert np.allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_sinkhorn():
    rng = np.random.default_rng(0)
    hc, iters, eps = 4, 20, 1e-6
    mix_hc = (2 + hc) * hc
    mixes = rng.standard_normal((2, 5, mix_hc))
    scale = rng.standard_normal(3)
    base = rng.standard_normal(mix_hc)
    pre_r, post_r, comb_r = np_sinkhorn(mixes, scale, base, hc, iters, eps)
    pre_m, post_m, comb_m = D.hc_split_sinkhorn(t2m(mixes), t2m(scale), t2m(base), hc, iters, eps)
    assert np.allclose(m2n(pre_m), pre_r, rtol=RTOL, atol=ATOL)
    assert np.allclose(m2n(post_m), post_r, rtol=RTOL, atol=ATOL)
    assert np.allclose(m2n(comb_m), comb_r, rtol=RTOL, atol=ATOL)
    # doubly-stochastic property (independent check on the algorithm)
    assert abs(comb_r.sum(-2).mean() - 1) < 2e-2
    assert abs(comb_r.sum(-1).mean() - 1) < 5e-2


def test_hyperconnection_pre_post():
    rng = np.random.default_rng(1)
    hc, dim, iters, eps = 4, 32, 20, 1e-6
    mix_hc = (2 + hc) * hc
    fn = rng.standard_normal((mix_hc, hc * dim)) * 0.1
    scale = rng.standard_normal(3)
    base = rng.standard_normal(mix_hc)
    x = rng.standard_normal((2, 5, hc, dim))

    # oracle hc_pre
    xf = x.reshape(2, 5, hc * dim)
    rsq = 1.0 / np.sqrt(np.mean(xf ** 2, -1, keepdims=True) + eps)
    mixes = (xf @ fn.T) * rsq
    pre_r, post_r, comb_r = np_sinkhorn(mixes, scale, base, hc, iters, eps)
    y_r = np.sum(pre_r[..., None] * x, axis=-2)

    hyper = D.HyperConnection(dim, hc, eps)
    hyper._iters = iters
    hyper.fn = t2m(fn); hyper.scale = t2m(scale); hyper.base = t2m(base)
    y_m, post_m, comb_m = hyper.pre(t2m(x))
    assert np.allclose(m2n(y_m), y_r, rtol=RTOL, atol=ATOL)
    assert np.allclose(m2n(post_m), post_r, rtol=RTOL, atol=ATOL)
    assert np.allclose(m2n(comb_m), comb_r, rtol=RTOL, atol=ATOL)

    # oracle hc_post
    xin = rng.standard_normal((2, 5, dim))
    resid = rng.standard_normal((2, 5, hc, dim))
    z_r = post_r[..., None] * xin[..., None, :] + np.einsum("...jk,...jd->...kd", comb_r, resid)
    z_m = hyper.post(t2m(xin), t2m(resid), t2m(post_r), t2m(comb_r))
    assert np.allclose(m2n(z_m), z_r, rtol=RTOL, atol=ATOL)


def test_compressor_pooling_non_overlap():
    rng = np.random.default_rng(2)
    dim, ratio, head_dim, rd, eps = 32, 8, 24, 8, 1e-6
    args = D.ModelArgs(hidden_size=dim, head_dim=head_dim, qk_rope_head_dim=rd,
                       rms_norm_eps=eps, compress_rope_theta=160000.0,
                       original_seq_len=64, rope_factor=16.0, beta_fast=32, beta_slow=1)
    comp = D.Compressor(args, ratio, head_dim)
    wkv = rng.standard_normal((head_dim, dim)) * 0.1
    wgate = rng.standard_normal((head_dim, dim)) * 0.1
    ape = rng.standard_normal((ratio, head_dim)) * 0.1
    normw = rng.standard_normal(head_dim) * 0.1 + 1.0
    comp.wkv.weight = t2m(wkv); comp.wgate.weight = t2m(wgate)
    comp.ape = t2m(ape); comp.norm.weight = t2m(normw)

    x = rng.standard_normal((2, 40, dim))
    got = m2n(comp(t2m(x)))

    # oracle
    cutoff = 40 - (40 % ratio)
    nwin = cutoff // ratio
    kv = (x[:, :cutoff] @ wkv.T).reshape(2, nwin, ratio, head_dim)
    sc = (x[:, :cutoff] @ wgate.T).reshape(2, nwin, ratio, head_dim) + ape
    pooled = np.sum(kv * np_softmax(sc, axis=2), axis=2)
    rms = 1.0 / np.sqrt(np.mean(pooled ** 2, -1, keepdims=True) + eps)
    pooled = pooled * rms * normw
    inv = np_yarn_inv_freq(rd, 160000.0, 64, 16.0, 32, 1)
    win_pos = (np.arange(nwin) * ratio)[:, None]
    ang = win_pos * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    tail = np_rope_interleaved(pooled[..., -rd:], cos[None], sin[None])
    ref = np.concatenate([pooled[..., :-rd], tail], axis=-1)
    assert np.allclose(got, ref, rtol=2e-5, atol=2e-6)


def test_o_lora_grouped():
    rng = np.random.default_rng(3)
    dim, n_heads, hd, g, r = 32, 4, 16, 2, 8
    args = D.ModelArgs(hidden_size=dim, num_attention_heads=n_heads, head_dim=hd,
                       qk_rope_head_dim=8, q_lora_rank=16, o_lora_rank=r, o_groups=g,
                       compress_ratios=[0] * 8)
    attn = D.DeepseekV4Attention(args, 0)
    per = n_heads * hd // g
    wo_a = rng.standard_normal((g * r, per)) * 0.1
    wo_b = rng.standard_normal((dim, g * r)) * 0.1
    attn.wo_a.weight = t2m(wo_a); attn.wo_b.weight = t2m(wo_b)
    o = rng.standard_normal((2, 5, n_heads * hd))
    got = m2n(attn._o_lora(t2m(o)))
    # oracle (model.py L537-542)
    o2 = o.reshape(2, 5, g, per)
    w = wo_a.reshape(g, r, per)
    o3 = np.einsum("bsgp,grp->bsgr", o2, w).reshape(2, 5, g * r)
    ref = o3 @ wo_b.T
    assert np.allclose(got, ref, rtol=RTOL, atol=ATOL)


def test_gate_sqrtsoftplus_non_hash():
    rng = np.random.default_rng(4)
    dim, n_routed, topk = 32, 8, 2
    args = D.ModelArgs(hidden_size=dim, n_routed_experts=n_routed, num_experts_per_tok=topk,
                       scoring_func="sqrtsoftplus", routed_scaling_factor=1.5, num_hash_layers=0)
    gate = D.MoEGate(args, 5)
    w = rng.standard_normal((n_routed, dim))
    bias = rng.standard_normal(n_routed) * 0.1
    gate.weight = t2m(w); gate.e_score_correction_bias = t2m(bias)
    x = rng.standard_normal((7, dim))
    idx_m, w_m = gate(t2m(x), None)
    idx_m, w_m = np.array(idx_m), m2n(w_m)
    # oracle
    s = np.sqrt(np.log1p(np.exp(x @ w.T)))  # sqrt(softplus)
    sel = np.argsort(-(s + bias), axis=-1)[:, :topk]
    ww = np.take_along_axis(s, sel, axis=-1)
    ww = ww / ww.sum(-1, keepdims=True) * 1.5
    for i in range(7):
        assert set(idx_m[i]) == set(sel[i]), (idx_m[i], sel[i])
    # align weights by expert id and compare
    def align(idx, wt):
        o = np.zeros((idx.shape[0], n_routed))
        for i in range(idx.shape[0]):
            o[i, idx[i]] = wt[i]
        return o
    assert np.allclose(align(idx_m, w_m), align(sel, ww), rtol=RTOL, atol=ATOL)


def test_gate_hash_routing():
    rng = np.random.default_rng(5)
    dim, n_routed, topk, vocab = 32, 8, 2, 50
    args = D.ModelArgs(hidden_size=dim, n_routed_experts=n_routed, num_experts_per_tok=topk,
                       scoring_func="sqrtsoftplus", routed_scaling_factor=1.5,
                       num_hash_layers=3, vocab_size=vocab)
    gate = D.MoEGate(args, 0)
    w = rng.standard_normal((n_routed, dim))
    tid2eid = rng.integers(0, n_routed, (vocab, topk)).astype(np.int32)
    gate.weight = t2m(w)
    gate.tid2eid = mx.array(tid2eid)
    x = rng.standard_normal((7, dim))
    ids = rng.integers(0, vocab, (7,)).astype(np.int64)
    idx_m, w_m = gate(t2m(x), mx.array(ids))
    idx_m, w_m = np.array(idx_m), m2n(w_m)
    # oracle: fixed expert set per token id, weights = sqrtsoftplus scores gathered
    s = np.sqrt(np.log1p(np.exp(x @ w.T)))
    sel = tid2eid[ids]
    ww = np.take_along_axis(s, sel, axis=-1)
    ww = ww / ww.sum(-1, keepdims=True) * 1.5
    assert np.array_equal(idx_m, sel)
    assert np.allclose(w_m, ww, rtol=RTOL, atol=ATOL)
