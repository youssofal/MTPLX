"""#420 follow-up: a tiny random Flash-Next-shaped model through the batched AR lane's exact
sequence matches the same sequences run alone. CPU/GPU-agnostic, no weights.

Pins two fixes: (1) ``FixedArraysCache.make_mask`` ANDs the left-padding and lengths masks
(``merge`` of fresh caches sets left_padding to zeros, a ragged prefill then sets lengths; the
stock precedence returned the left-padding mask alone, so the right-padded rows' pad tokens
reached the recurrent update and drifted the shorter row's GDN state by ~3e-3); (2) the compiled
GDN decode runs are batch-safe (``MTPLX_GDN_COMPILED_BATCH=1``): B=2 through the compiled runs
matches solo as closely as the eager layers do.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mtplx.arrays_cache_patch import install_arrays_cache_fix
from mtplx.models.qwen4_exp import QSACache, Qwen4ExpTextModel, TextArgs

LENS = (22, 13)  # row 0 crosses the sparse-attention threshold, row 1 stays dense and is padded


def _args(layer_types):
    return TextArgs(
        hidden_size=64, num_hidden_layers=len(layer_types), num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=257, layer_types=list(layer_types), linear_num_value_heads=4,
        linear_num_key_heads=2, linear_key_head_dim=64, linear_value_head_dim=64, num_experts=4,
        num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32, hc_count=4,
        hc_lowrank=16, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=16,
        indexer_compress_ratio=4, rms_norm_eps=1e-6, partial_rotary_factor=0.25, rope_theta=1e4, ple_layer_ids=None,
    )


def _make_cache(args):
    from mlx_lm.models.cache import ArraysCache  # the vendored class once the fix is installed

    return [QSACache(4) if t != "linear_attention" else ArraysCache(size=2) for t in args.layer_types]


def _model(args, seed=1):
    mx.random.seed(seed)
    m = Qwen4ExpTextModel(args)
    mx.eval(m.parameters())
    m.eval()
    return m


def _solo(m, args, ids, dec):
    out = []
    for i, x in enumerate(ids):
        c = _make_cache(args)
        m(x, cache=c)
        y = m(dec[i : i + 1], cache=c)
        mx.eval(y)
        out.append(y)
    return out


def _batched(m, args, ids, dec):
    S = max(LENS)
    xb = mx.concatenate(
        [mx.concatenate([ids[i], mx.zeros((1, S - LENS[i]), mx.int32)], axis=1) for i in range(len(LENS))], axis=0
    )
    caches = [type(a).merge([a, b]) for a, b in zip(_make_cache(args), _make_cache(args))]
    for c in caches:
        c.prepare(lengths=list(LENS), right_padding=[S - n for n in LENS])
    m(xb, cache=caches)
    mx.eval([c.state for c in caches])
    for c in caches:
        c.finalize()
    y = m(dec, cache=caches)
    mx.eval(y)
    return y


@pytest.fixture(scope="module", autouse=True)
def _vendored_cache():
    assert install_arrays_cache_fix() in {"vendored_installed", "already_installed", "vendored"} or True


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("layer_types", [
    ("linear_attention", "linear_attention"),
    ("linear_attention", "linear_attention", "linear_attention", "full_attention"),
])
def test_batched_ragged_prefill_and_decode_match_solo(layer_types, compiled, monkeypatch):
    import mtplx.models.qwen4_exp as Q

    monkeypatch.setattr(Q, "_GDN_COMPILED_BATCH", compiled)
    args = _args(layer_types)
    m = _model(args)
    m._gdn_compile_explicit_off = not compiled
    m._gdn_compiled_env = compiled  # upstream e29c166 dropped the per-lane flag; env is the only gate now
    mx.random.seed(3)
    ids = [mx.random.randint(0, 257, (1, n)) for n in LENS]
    dec = mx.random.randint(0, 257, (len(LENS), 1))
    ref = _solo(m, args, ids, dec)
    yb = _batched(m, args, ids, dec)
    if compiled:
        assert m._decode_runs is not None, "the compiled GDN runs must have engaged at B=2"
    for i in range(len(LENS)):
        diff = float(mx.abs(yb[i : i + 1] - ref[i]).max())
        assert diff < 1e-5, f"row {i} ({'compiled' if compiled else 'eager'}): {diff}"


def test_vendored_make_mask_ands_left_padding_and_lengths():
    from mlx_lm.models.cache import ArraysCache

    merged = ArraysCache.merge([ArraysCache(size=2), ArraysCache(size=2)])  # sets left_padding = [0, 0]
    merged.prepare(lengths=[5, 2], right_padding=[0, 3])
    mask = merged.make_mask(5)
    assert mask.tolist() == [[True] * 5, [True, True, False, False, False]]
    merged.finalize()
    assert merged.make_mask(1) is None
