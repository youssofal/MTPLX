"""Ratio-4 indexer (top-k compressed-row filter) gates for the DeepSeek-V4 backend.

Attention on a ``compress_ratio==4`` layer may only see the ``index_topk`` compressed
rows the indexer scores highest for that query.  Below the threshold that selects
*every* causal row, so the filter is invisible until context passes
``index_topk * ratio`` tokens — which is why this file shrinks ``index_topk`` until
the sparse regime is reachable in a unit test.

Three things are gated:
  1. **The math**, against a self-contained NumPy transcription of the reference
     (``deepseek-ai/DeepSeek-V4-Flash/inference/model.py``: ``Compressor`` L279-377,
     ``Indexer`` L380-433, ``Attention.forward`` L484-543, and ``sparse_attn``
     semantics from ``inference/kernel.py`` L294-352).  The oracle is a *gather*
     implementation — it builds the reference's ``topk_idxs`` matrix, ``-1`` and all,
     and gathers rows — so it independently checks that the dense boolean mask this
     backend uses is the same computation.
  2. **Prefill/decode equivalence in the sparse regime**: one-shot logits vs
     prompt-prefill + token-by-token decode, including a run that crosses the
     dense->sparse threshold in the middle of the decode loop.
  3. **The reduction**: with ``k`` not binding, the filter must reproduce the dense
     path *bit-identically*, so the pre-existing parity golden stays valid.

Self-contained: shrunk seeded config, no downloads, no torch.  CPU device, so MLX
fp32 matmul is bit-exact (its GPU fast path carries ~7.5e-4 relative) — same
convention as the parity and decode tests.
"""
import importlib.util
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402

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
_spec = importlib.util.spec_from_file_location("dsv4_indexer_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_indexer_undertest"] = D
_spec.loader.exec_module(D)

# Shrunk config, same layer menu as the decode test (ratio 0 / 4 / 128 / 4) but with
# an index_topk small enough that ordinary test-length sequences go sparse:
# a ratio-4 layer emits one row per 4 tokens, so n_comp > INDEX_TOPK from token 28.
VOCAB = 64
DIM = 32
N_HEADS = 4
HEAD_DIM = 16
ROPE_DIM = 8
N_EXPERTS = 8
RATIOS = [0, 4, 128, 4]
WINDOW = 16
INDEX_HEAD_DIM = 16       # must be a power of two: the indexer Hadamard-rotates it
INDEX_TOPK = 6
SPARSE_FROM = (INDEX_TOPK + 1) * 4   # first token position with n_comp > INDEX_TOPK


def _args(**over):
    kwargs = dict(
        vocab_size=VOCAB,
        hidden_size=DIM,
        num_hidden_layers=len(RATIOS),
        num_hash_layers=1,
        num_attention_heads=N_HEADS,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
        q_lora_rank=16,
        o_lora_rank=8,
        o_groups=2,
        moe_intermediate_size=16,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=2,
        index_n_heads=N_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        index_topk=INDEX_TOPK,
        compress_ratios=list(RATIOS),
        compress_rope_theta=160000.0,
        sliding_window=WINDOW,
        rope_scaling={
            "original_max_position_embeddings": 65536,
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "type": "yarn",
        },
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=0.0,
    )
    kwargs.update(over)
    return D.ModelArgs(**kwargs)


def _fill(module, seed):
    """Seeded pseudo-random parameters, shaped from the module tree itself."""
    mx.random.seed(seed)
    filled = []
    for name, value in tree_flatten(module.parameters()):
        leaf = name.split(".")[-1]
        if leaf == "tid2eid":
            new = mx.random.randint(0, N_EXPERTS, value.shape).astype(mx.int32)
        elif value.ndim == 1:
            noise = mx.random.normal(value.shape) * 0.1
            centre = 1.0 if leaf == "scale" or name.endswith("norm.weight") else 0.0
            new = noise + centre
        else:
            new = mx.random.normal(value.shape) * (value.shape[-1] ** -0.5)
        filled.append((name, new.astype(value.dtype)))
    module.update(tree_unflatten(filled))
    mx.eval(module.parameters())
    return {k: np.array(v.astype(mx.float32)) for k, v in filled}


def _seeded_model(seed=0, **over):
    args = _args(**over)
    model = D.Model(args)
    _fill(model, seed)
    return args, model


def _tokens(seq_len, batch=1, seed=1234):
    mx.random.seed(seed)
    return mx.random.randint(0, VOCAB, (batch, seq_len))


def _compare(ref, got, label):
    """Per-step max relative error + exact argmax against the one-shot oracle."""
    worst_rel = 0.0
    for row in range(ref.shape[0]):
        for t in range(ref.shape[1]):
            scale = float(np.max(np.abs(ref[row, t]))) + 1e-12
            rel = float(np.max(np.abs(got[row, t] - ref[row, t]))) / scale
            worst_rel = max(worst_rel, rel)
            assert rel <= 5e-5, (
                f"{label}: row {row} step {t} logits diverge max_rel={rel:.3e}"
            )
            assert int(got[row, t].argmax()) == int(ref[row, t].argmax()), (
                f"{label}: row {row} step {t} argmax {int(got[row, t].argmax())} != "
                f"{int(ref[row, t].argmax())} (max_rel={rel:.3e})"
            )
    return worst_rel


def _prefill_then_decode(model, ids, prompt_len, prompt_chunks=1):
    total = ids.shape[1]
    cache = model.make_cache()
    pieces = []
    bounds = [round(prompt_len * (i + 1) / prompt_chunks) for i in range(prompt_chunks)]
    start = 0
    for end in bounds:
        if end == start:
            continue
        pieces.append(np.array(model(ids[:, start:end], cache=cache)))
        start = end
    for t in range(prompt_len, total):
        pieces.append(np.array(model(ids[:, t : t + 1], cache=cache)))
    assert [c.offset for c in cache] == [total] * len(cache)
    # the indexer's compressor lane must have tracked the attention lane exactly
    for c, r in zip(cache, RATIOS):
        assert c.n_index_compressed == (c.n_compressed if r == 4 else 0)
    return np.concatenate(pieces, axis=1)


def _run_case(prompt_len, total, *, prompt_chunks=1, batch=1, seed=0, label="", **over):
    args, model = _seeded_model(seed=seed, **over)
    ids = _tokens(total, batch=batch)
    ref = np.array(model(ids).astype(mx.float32))
    assert len(set(ref[0].argmax(-1).tolist())) > 1, "oracle logits are degenerate"
    got = _prefill_then_decode(model, ids, prompt_len, prompt_chunks=prompt_chunks)
    return _compare(ref, got, label or f"P={prompt_len} T={total}")


# --------------------------------------------------------------------------- oracles
def np_softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def np_yarn_inv_freq(dim, base, orig, factor, bf, bs):
    """model.py precompute_freqs_cis (frequency part)."""
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


def np_rope(x, pos, inv, inverse=False):
    """apply_rotary_emb (model.py L232-244) on the last ``2*len(inv)`` dims."""
    ang = np.asarray(pos, dtype=np.float64)[..., None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    if inverse:
        sin = -sin
    x0, x1 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * cos - x1 * sin
    out[..., 1::2] = x0 * sin + x1 * cos
    return out


def np_hadamard_matrix(n):
    """Sylvester construction, normalised — ``rotate_activation`` (model.py L247-251)."""
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def np_rms(x, w, eps=1e-6):
    return x * (1.0 / np.sqrt(np.mean(x ** 2, -1, keepdims=True) + eps)) * w


def np_compress(x, wkv, wgate, ape, normw, ratio, d, rd, inv, rotate):
    """Reference ``Compressor.forward`` at ``start_pos == 0`` (model.py L316-377)."""
    b, s, _ = x.shape
    cutoff = s - (s % ratio)
    nwin = cutoff // ratio
    if nwin == 0:
        return np.zeros((b, 0, d))
    kv = (x[:, :cutoff] @ wkv.T).reshape(b, nwin, ratio, -1)
    sc = (x[:, :cutoff] @ wgate.T).reshape(b, nwin, ratio, -1) + ape
    if ratio == 4:  # overlap_transform (L307-314)
        kv_o = np.zeros((b, nwin, 2 * ratio, d))
        sc_o = np.full((b, nwin, 2 * ratio, d), -np.inf)
        kv_o[:, :, ratio:] = kv[..., d:]
        sc_o[:, :, ratio:] = sc[..., d:]
        kv_o[:, 1:, :ratio] = kv[:, :-1, :, :d]
        sc_o[:, 1:, :ratio] = sc[:, :-1, :, :d]
        kv, sc = kv_o, sc_o
    pooled = (kv * np_softmax(sc, axis=2)).sum(axis=2)
    pooled = np_rms(pooled, normw)
    pos = np.arange(nwin) * ratio
    pooled = np.concatenate(
        [pooled[..., :-rd], np_rope(pooled[..., -rd:], pos[None, :], inv)], axis=-1
    )
    if rotate:
        pooled = pooled @ np_hadamard_matrix(pooled.shape[-1]).T
    return pooled


def np_index_scores(x, qr, wq_b, wproj, rows, n_heads, hd, rd, inv, positions):
    """Reference ``Indexer.forward`` scoring half (model.py L411-421)."""
    b, s, _ = x.shape
    q = (qr @ wq_b.T).reshape(b, s, n_heads, hd)
    q = np.concatenate(
        [q[..., :-rd], np_rope(q[..., -rd:], positions[None, :, None], inv)], axis=-1
    )
    q = q @ np_hadamard_matrix(hd).T
    w = (x @ wproj.T) * (hd ** -0.5 * n_heads ** -0.5)
    sc = np.einsum("bshd,btd->bsht", q, rows)
    return (np.maximum(sc, 0.0) * w[..., None]).sum(axis=2)


def np_topk_ds4(scores, k):
    """ds4.c ``indexer_allowed_decode_one`` selection loop: k passes, each taking the
    highest not-yet-taken entry under a strict ``>``, so ties go to the lowest index."""
    taken = np.zeros(scores.shape[0], dtype=bool)
    for _ in range(k):
        best, best_score = -1, -np.inf
        for c in range(scores.shape[0]):
            if not taken[c] and scores[c] > best_score:
                best, best_score = c, scores[c]
        if best < 0:
            break
        taken[best] = True
    return taken


def np_reference_topk_idxs(scores, seqlen, n_comp, ratio, index_topk, offset):
    """Reference ``Indexer.forward`` selection half at ``start_pos == 0`` (L424-430):
    one global ``k``, ``-inf`` on non-causal windows, then any surviving non-causal
    pick is rewritten to ``-1`` (which ``sparse_attn`` treats as "no row")."""
    k = min(index_topk, n_comp)
    out = np.full((seqlen, k), -1, dtype=np.int64)
    for i in range(seqlen):
        row = np.where(np.arange(n_comp) < (i + 1) // ratio, scores[i], -np.inf)
        taken = np_topk_ds4(row, k)
        picks = np.flatnonzero(taken)
        for j, c in enumerate(picks):
            out[i, j] = -1 if c >= (i + 1) // ratio else c + offset
    return out


def np_window_topk_idxs(window_size, seqlen):
    """``get_window_topk_idxs`` at start_pos == 0 (model.py L262-264)."""
    base = np.arange(seqlen)[:, None]
    m = np.clip(base - window_size + 1, 0, None) + np.arange(min(seqlen, window_size))
    return np.where(m > base, -1, m)


def np_sparse_attn(q, kv, sink, idxs, scale):
    """``sparse_attn`` (kernel.py L294-352): gather the listed rows (``-1`` -> a
    ``-inf`` score and a zero row), softmax with the per-head sink in the denominator."""
    b, s, h, d = q.shape
    o = np.zeros_like(q)
    for bi in range(b):
        for i in range(s):
            cols = idxs[bi, i]
            valid = cols >= 0
            rows = kv[bi, np.where(valid, cols, 0)]          # [k, d]
            logits = (q[bi, i] @ rows.T) * scale             # [h, k]
            logits = np.where(valid[None, :], logits, -np.inf)
            m = logits.max(-1, keepdims=True)
            m = np.maximum(m, sink[:, None])
            e = np.exp(logits - m)
            denom = e.sum(-1, keepdims=True) + np.exp(sink[:, None] - m)
            o[bi, i] = (e / denom) @ rows
    return o


def np_attention(P, x, args, ratio):
    """Reference ``Attention.forward`` at ``start_pos == 0`` (model.py L484-543), with
    the indexer supplying the compressed half of ``topk_idxs``."""
    b, s, _ = x.shape
    hd, rd, eps = args.head_dim, args.qk_rope_head_dim, args.rms_norm_eps
    inv = np_yarn_inv_freq(rd, args.compress_rope_theta, args.original_seq_len,
                           args.rope_factor, args.beta_fast, args.beta_slow)
    pos = np.arange(s)

    qr = np_rms(x @ P["wq_a.weight"].T, P["q_norm.weight"], eps)
    q = (qr @ P["wq_b.weight"].T).reshape(b, s, args.num_attention_heads, hd)
    q = q / np.sqrt(np.mean(q ** 2, -1, keepdims=True) + eps)
    q = np.concatenate(
        [q[..., :-rd], np_rope(q[..., -rd:], pos[None, :, None], inv)], axis=-1
    )
    kv = np_rms(x @ P["wkv.weight"].T, P["kv_norm.weight"], eps)
    kv = np.concatenate(
        [kv[..., :-rd], np_rope(kv[..., -rd:], pos[None, :], inv)], axis=-1
    )

    comp = np_compress(x, P["compressor.wkv.weight"], P["compressor.wgate.weight"],
                       P["compressor.ape"], P["compressor.norm.weight"],
                       ratio, hd, rd, inv, rotate=False)
    n_comp = comp.shape[1]
    idx_rows = np_compress(
        x, P["indexer.compressor.wkv.weight"], P["indexer.compressor.wgate.weight"],
        P["indexer.compressor.ape"], P["indexer.compressor.norm.weight"],
        ratio, args.index_head_dim, rd, inv, rotate=True,
    )
    scores = np_index_scores(
        x, qr, P["indexer.wq_b.weight"], P["indexer.weights_proj.weight"], idx_rows,
        args.index_n_heads, args.index_head_dim, rd, inv, pos,
    )

    win_idxs = np_window_topk_idxs(args.window_size, s)
    all_idxs = np.zeros((b, s, win_idxs.shape[1] + min(args.index_topk, n_comp)),
                        dtype=np.int64)
    for bi in range(b):
        comp_idxs = np_reference_topk_idxs(
            scores[bi], s, n_comp, ratio, args.index_topk, offset=s
        )
        all_idxs[bi] = np.concatenate([win_idxs, comp_idxs], axis=-1)

    o = np_sparse_attn(q, np.concatenate([kv, comp], axis=1), P["attn_sink"],
                       all_idxs, hd ** -0.5)
    o = np.concatenate(
        [o[..., :-rd], np_rope(o[..., -rd:], pos[None, :, None], inv, inverse=True)],
        axis=-1,
    )
    g, r = args.o_groups, args.o_lora_rank
    o = o.reshape(b, s, g, -1)
    o = np.einsum("bsgp,grp->bsgr", o, P["wo_a.weight"].reshape(g, r, -1))
    return o.reshape(b, s, g * r) @ P["wo_b.weight"].T


# --------------------------------------------------------------------------- tests
def test_hadamard_matches_sylvester_and_is_orthogonal():
    """``_hadamard_rotate`` is the normalised Hadamard transform, for the real model
    width (128) as well as the shrunk one."""
    rng = np.random.default_rng(0)
    for n in (16, 128):
        x = rng.standard_normal((3, 5, n))
        got = np.array(D._hadamard_rotate(mx.array(x.astype(np.float32))))
        ref = x @ np_hadamard_matrix(n).T
        assert np.allclose(got, ref, rtol=1e-5, atol=1e-6), n
        h = np_hadamard_matrix(n)
        assert np.allclose(h @ h.T, np.eye(n), atol=1e-12)
    with pytest.raises(ValueError):
        D._hadamard_rotate(mx.zeros((2, 12)))


def test_hadamard_leaves_indexer_scores_invariant():
    """Why dropping FP4 leaves the rotation cosmetic: it is applied to *both* sides of
    the indexer dot product, and it is orthogonal, so every score is unchanged.  The
    rotation is kept because it is the model graph and the slot FP4 QAT occupies —
    but no selection can turn on it alone.
    """
    rng = np.random.default_rng(1)
    q = mx.array(rng.standard_normal((2, 3, 4, 16)).astype(np.float32))
    k = mx.array(rng.standard_normal((2, 7, 16)).astype(np.float32))
    plain = np.array(mx.einsum("bshd,btd->bsht", q, k))
    rot = np.array(mx.einsum("bshd,btd->bsht",
                             D._hadamard_rotate(q), D._hadamard_rotate(k)))
    assert np.allclose(plain, rot, rtol=1e-5, atol=1e-5)


def test_topk_mask_matches_ds4_selection_including_ties():
    """``_topk_mask`` against ds4.c's selection loop, on scores deliberately seeded
    with exact ties (the realistic case: rows whose every head ReLU'd to zero)."""
    rng = np.random.default_rng(2)
    n = 12
    scores = rng.integers(0, 4, (5, n)).astype(np.float32)  # many exact collisions
    scores[2] = 0.0                                          # a fully tied row
    for k in (0, 1, 3, 7, n):
        k_row = np.full((5, 1), k, dtype=np.int32)
        got = np.array(D._topk_mask(mx.array(scores), mx.array(k_row), k))
        ref = np.stack([np_topk_ds4(scores[i], k) for i in range(5)])
        assert np.array_equal(got, ref), (k, scores, got, ref)
        assert got.sum(-1).tolist() == [min(k, n)] * 5
    # per-row k
    k_row = np.array([[0], [1], [4], [n], [2]], dtype=np.int32)
    got = np.array(D._topk_mask(mx.array(scores), mx.array(k_row), n))
    ref = np.stack([np_topk_ds4(scores[i], int(k_row[i, 0])) for i in range(5)])
    assert np.array_equal(got, ref)


def test_indexer_scores_match_reference_oracle():
    """Indexer scoring (wq_b -> rope -> Hadamard -> ReLU'd per-head dots -> weighted
    sum) against the NumPy transcription of model.py L411-421."""
    rng = np.random.default_rng(3)
    args = _args()
    attn = D.DeepseekV4Attention(args, 1)
    P = _fill(attn, seed=7)
    s = 40
    x = rng.standard_normal((2, s, DIM)).astype(np.float32)
    xm = mx.array(x)

    rows = attn.indexer.compressor(xm)
    qr = attn.q_norm(attn.wq_a(xm))
    positions = mx.arange(s)
    got = np.array(attn.indexer.scores(xm, qr, positions, rows).astype(mx.float32))

    rd = args.qk_rope_head_dim
    inv = np_yarn_inv_freq(rd, args.compress_rope_theta, args.original_seq_len,
                           args.rope_factor, args.beta_fast, args.beta_slow)
    ref_rows = np_compress(
        x, P["indexer.compressor.wkv.weight"], P["indexer.compressor.wgate.weight"],
        P["indexer.compressor.ape"], P["indexer.compressor.norm.weight"],
        4, args.index_head_dim, rd, inv, rotate=True,
    )
    assert np.allclose(np.array(rows.astype(mx.float32)), ref_rows, rtol=2e-5, atol=2e-6)
    ref_qr = np_rms(x @ P["wq_a.weight"].T, P["q_norm.weight"], args.rms_norm_eps)
    ref = np_index_scores(x, ref_qr, P["indexer.wq_b.weight"],
                          P["indexer.weights_proj.weight"], ref_rows,
                          args.index_n_heads, args.index_head_dim, rd, inv,
                          np.arange(s))
    scale = float(np.max(np.abs(ref)))
    assert np.max(np.abs(got - ref)) / scale <= 2e-5, np.max(np.abs(got - ref))
    # non-vacuous: some rows really do score zero (all heads ReLU'd away)
    assert (ref == 0).any() and (ref > 0).any()


def test_sparse_attention_matches_reference_gather():
    """The whole ratio-4 attention block against the reference's *gather* formulation:
    reference ``topk_idxs`` (window ids + indexer picks, ``-1`` for unusable) fed
    through ``sparse_attn``.  This is the gate on the dense-mask equivalence."""
    rng = np.random.default_rng(4)
    args = _args()
    attn = D.DeepseekV4Attention(args, 1)
    P = _fill(attn, seed=11)
    s = 60
    assert s // 4 > INDEX_TOPK, "config must reach the sparse regime"
    x = rng.standard_normal((2, s, DIM)).astype(np.float32) * 0.5
    got = np.array(attn(mx.array(x)).astype(mx.float32))
    ref = np_attention(P, x.astype(np.float64), args, ratio=4)
    scale = float(np.max(np.abs(ref)))
    rel = float(np.max(np.abs(got - ref))) / scale
    assert rel <= 5e-5, f"sparse attention diverges from the gather oracle: {rel:.3e}"

    # ...and the gate is not vacuous: dense-over-compressed gives a *different*
    # answer at this length, so the oracle is actually testing the filter.
    dense = D.DeepseekV4Attention(_args(index_topk=10 ** 6), 1)
    dense.update(attn.parameters())
    dense_out = np.array(dense(mx.array(x)).astype(mx.float32))
    assert float(np.max(np.abs(dense_out - ref))) / scale > 1e-3


def test_filter_reduces_to_dense_bit_identically():
    """With ``k`` not binding the selection *is* the causal mask, so running the whole
    scoring path must reproduce the dense path bit for bit — which is what keeps the
    pre-existing parity golden (index_topk=512, n_comp=40) valid."""
    args, model = _seeded_model(seed=3, index_topk=10 ** 6)
    ids = _tokens(60)
    dense = np.array(model(ids).astype(mx.float32))

    forced = D.DeepseekV4Attention._indexer_active
    try:
        # force the filter on with a k that cannot bind
        D.DeepseekV4Attention._indexer_active = lambda self, n: self.compress_ratio == 4 and n > 0
        sparse = np.array(model(ids).astype(mx.float32))
    finally:
        D.DeepseekV4Attention._indexer_active = forced
    assert np.array_equal(dense, sparse), (
        f"non-binding filter perturbed the dense path: "
        f"max_abs={float(np.max(np.abs(dense - sparse))):.3e}"
    )

    # and the selection really is every causal row
    attn = model.model.layers[1].attn
    x = mx.random.normal((1, 40, DIM))
    rows = attn.indexer.compressor(x)
    sel = np.array(attn.indexer(x, attn.q_norm(attn.wq_a(x)), mx.arange(40), rows))
    causal = np.arange(rows.shape[1])[None, :] < ((np.arange(40)[:, None] + 1) // 4)
    assert np.array_equal(sel[0], causal)


def test_indexer_actually_filters():
    """Guard the premise of every parity case below: at these lengths the filter
    excludes rows, and does so on both ratio-4 layers."""
    args, model = _seeded_model(seed=0)
    x = mx.random.normal((1, 60, DIM))
    for lid in (1, 3):
        attn = model.model.layers[lid].attn
        rows = attn.indexer.compressor(x)
        n_comp = int(rows.shape[1])
        assert attn._indexer_active(n_comp)
        sel = np.array(attn.indexer(x, attn.q_norm(attn.wq_a(x)), mx.arange(60), rows))
        counts = sel[0].sum(-1)
        assert counts.max() == INDEX_TOPK, counts
        assert counts[-1] == INDEX_TOPK < n_comp
        # early queries are below the cut and keep every causal row
        assert counts[7] == 2 == (7 + 1) // 4


def test_cache_carries_a_second_compressor_lane():
    """The indexer lane is a full peer of the attention lane: own frontier, own rows,
    and both survive a state/meta_state round trip."""
    args, model = _seeded_model()
    cache = model.make_cache()
    model(_tokens(30), cache=cache)
    assert [c.n_compressed for c in cache] == [0, 7, 0, 7]
    assert [c.n_index_compressed for c in cache] == [0, 7, 0, 7]
    c = cache[1]
    assert c.index_comp.n_emitted == 7 and c.index_comp.cur_kv.shape[1] == 30 % 4
    assert c.index_compressed.shape[-1] == INDEX_HEAD_DIM
    assert c.compressed.shape[-1] == HEAD_DIM

    state, meta = c.state, c.meta_state
    # 15 = window + (rows, cur_kv, cur_score, prev_kv, prev_score, journal kv/score)
    # for the attention lane and again for the indexer lane.
    assert len(state) == 15 and len(meta) == 5
    fresh = D.DeepseekV4Cache(WINDOW, 4, HEAD_DIM)
    fresh.state = state
    fresh.meta_state = meta
    assert fresh.n_index_compressed == 7 and fresh.index_comp.n_emitted == 7
    assert fresh.offset == 30
    fresh.state = None
    assert fresh.n_index_compressed == 0 and fresh.index_comp.n_emitted == 0
    with pytest.raises(ValueError):
        fresh.meta_state = ("mtplx-deepseek-v4-cache-v1", "0", "0", "0")


def test_sparse_decode_matches_prefill_partial_window():
    """The headline gate: prompt ends mid-window, decode runs deep into the sparse
    regime, and per-step logits must match the one-shot forward."""
    total = 60
    assert 13 % 4 != 0 and total // 4 > INDEX_TOPK
    worst = _run_case(13, total, label="sparse partial-window")
    assert worst <= 5e-5


def test_decode_crosses_dense_to_sparse_threshold():
    """The crossing itself: the prompt is short enough that the filter is inactive
    (n_comp <= index_topk) and the threshold is passed inside the decode loop."""
    prompt, total = 17, 48
    assert prompt // 4 <= INDEX_TOPK < total // 4
    assert prompt < SPARSE_FROM <= total
    worst = _run_case(prompt, total, label="dense->sparse crossing")
    assert worst <= 5e-5


def test_sparse_chunked_prefill_then_decode():
    """Chunked prompt prefill in the sparse regime: chunk boundaries land at 22/44/66,
    so a mid-chunk query sees a *shorter* compressed axis than the one-shot run does."""
    worst = _run_case(66, 90, prompt_chunks=3, label="sparse chunked-prefill")
    assert worst <= 5e-5


def test_sparse_decode_batched():
    """b > 1: each row selects its own compressed set; nothing may leak across rows."""
    worst = _run_case(13, 60, batch=3, label="sparse batched")
    assert worst <= 5e-5


def test_sparse_decode_from_single_token_prompt():
    """Everything but token 0 goes through the s == 1 path, which now needs a mask on
    the compressed half even though the window half needs none."""
    worst = _run_case(1, 44, label="sparse single-token-prompt")
    assert worst <= 5e-5


def test_sparse_decode_from_window_aligned_prompt():
    """Prompt ends exactly on a ratio-4 boundary, so both compressor frontiers start
    the decode loop empty."""
    prompt = 32
    assert prompt % 4 == 0
    worst = _run_case(prompt, 64, prompt_chunks=2, label="sparse aligned-prompt")
    assert worst <= 5e-5
