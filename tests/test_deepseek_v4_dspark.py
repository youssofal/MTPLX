"""Synthetic contract gates for the native 0731 DSpark model layer.

These deliberately exercise structure and state transitions only.  They do not
need a checkpoint or a GPU.
"""

import importlib.util
import os
import sys
from dataclasses import fields

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


_HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = os.path.join(_HERE, "..", "mtplx", "models", "deepseek_v4.py")
_spec = importlib.util.spec_from_file_location("dsv4_dspark_undertest", _MODEL)
D = importlib.util.module_from_spec(_spec)
sys.modules["dsv4_dspark_undertest"] = D
_spec.loader.exec_module(D)


def _args(**over):
    cfg = dict(
        vocab_size=128800,
        hidden_size=8,
        num_hidden_layers=43,
        num_hash_layers=0,
        num_attention_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        moe_intermediate_size=4,
        n_routed_experts=2,
        num_experts_per_tok=1,
        index_n_heads=1,
        index_head_dim=8,
        index_topk=2,
        sliding_window=8,
        compress_ratios=[0] * 46,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_markov_rank=256,
        num_nextn_predict_layers=1,
        temperature=1.0,
    )
    cfg.update(over)
    return D.ModelArgs(**cfg)


@pytest.fixture
def tiny_model(monkeypatch):
    """Keep synthetic memory small without weakening the production manifest."""
    real_markov = D.DSparkMarkovHead
    real_confidence = D.DSparkConfidenceHead

    class TinyMarkov(real_markov):
        def __init__(self, vocab_size, _rank):
            real_markov.__init__(self, vocab_size, 3)

    class TinyConfidence(real_confidence):
        def __init__(self, hidden_size, _rank):
            real_confidence.__init__(self, hidden_size, 3)

    monkeypatch.setattr(D, "DSparkMarkovHead", TinyMarkov)
    monkeypatch.setattr(D, "DSparkConfidenceHead", TinyConfidence)
    return D.Model(_args())


_OFFICIAL_FILTERED = {
    "model_type": "deepseek_v4",
    "vocab_size": 129280,
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_hash_layers": 3,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 6,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
    "sliding_window": 128,
    "compress_ratios": [0, 0] + [4, 128] * 20 + [4, 0, 0, 0],
    "num_nextn_predict_layers": 1,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [40, 41, 42],
    "dspark_markov_rank": 256,
}


def test_real_filtered_config_derives_three_stages_without_n_mtp_layers():
    assert "n_mtp_layers" not in _OFFICIAL_FILTERED
    allowed = {f.name for f in fields(D.ModelArgs)}
    args = D.ModelArgs(**{k: v for k, v in _OFFICIAL_FILTERED.items() if k in allowed})
    D._validate_dspark_manifest(args)
    assert D.is_deepseek_v4_mtp_config(_OFFICIAL_FILTERED) is False
    assert (
        D.inject_deepseek_v4_mtp_support(object(), config=_OFFICIAL_FILTERED) is False
    )


def test_attention_exposes_exact_stock_q_projection_route():
    attention = D.DeepseekV4Attention(_args(), 0)
    qr = mx.arange(16, dtype=mx.float32).reshape(1, 2, 8).astype(mx.bfloat16)
    positions = mx.arange(2)
    cos, sin = attention._rope_tables(positions)

    projected = attention.wq_b(qr).reshape(1, 2, 1, 8)
    expected = projected * mx.rsqrt(
        mx.mean(mx.square(projected.astype(mx.float32)), axis=-1, keepdims=True)
        + attention.eps
    )
    expected = expected.astype(projected.dtype)
    expected = mx.concatenate(
        [
            expected[..., : -attention.rope_head_dim],
            D._apply_interleaved_rope(
                expected[..., -attention.rope_head_dim :],
                cos[None, :, None, :],
                sin[None, :, None, :],
            ),
        ],
        axis=-1,
    )
    actual = attention._q_projection_qhead_route(qr, cos, sin)

    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected)


def test_legacy_model_call_uses_prebound_target_route_exactly_once():
    inputs = mx.array([[7]], dtype=mx.int32)
    cache = object()
    hidden = object()
    calls = []

    class InnerModel:
        def hc_hidden(self, *_args, **_kwargs):
            raise AssertionError("legacy dispatch bypassed the installed target route")

    class Owner:
        model = InnerModel()
        _target_hidden_route = D._LegacyTargetRoute()

    owner = Owner()

    def installed_route(got_inputs, got_cache):
        calls.append((got_inputs, got_cache))
        return hidden

    owner._target_hc_hidden_route = installed_route
    logits, got_hidden = D.Model.__call__(
        owner,
        inputs,
        cache=cache,
        return_hidden=True,
        emit_logits=False,
    )

    assert logits is None
    assert got_hidden is hidden
    assert calls == [(inputs, cache)]


def test_dspark_manifest_installs_exactly_three_stages(tiny_model):
    model = tiny_model
    assert isinstance(model._dspark, D.DeepseekV4DSpark)
    assert len(model._dspark.stages) == 3
    assert model._dspark.block_size == 5
    assert model._dspark.noise_token_id == 128799
    assert model._dspark.target_layer_ids == (40, 41, 42)
    assert all(type(stage) is D.DeepseekV4DSparkStage for stage in model.mtp)
    assert not model.has_mtp  # generic preview-MTP routing is intentionally absent
    keys = {k for k, _ in tree_flatten(model.parameters())}
    assert "mtp.0.main_proj.weight" in keys
    assert "mtp.2.markov_head.markov_w1.weight" in keys
    assert "mtp.2.confidence_head.proj.weight" in keys
    assert "mtp.2.confidence_head.proj.bias" not in keys
    assert type(model._dspark.stages[0]) is D.DeepseekV4DSparkStage
    assert model._dspark.stages[0].main_proj is not None
    assert model._dspark.stages[1].main_proj is None
    assert model._dspark.stages[2].markov_head is not None


@pytest.mark.parametrize(
    "mut",
    [
        {"dspark_block_size": 4},
        {"num_nextn_predict_layers": 0},
        {"dspark_noise_token_id": 0},
        {"dspark_target_layer_ids": [39, 41, 42]},
        {"dspark_markov_rank": 255},
    ],
)
def test_dspark_manifest_fails_loudly_on_missing_or_wrong_invariants(mut):
    with pytest.raises(ValueError, match="DSpark"):
        D._validate_dspark_manifest(_args(**mut))


def test_partial_dspark_signature_never_becomes_legacy_mtp():
    corrupt = _args(dspark_block_size=0)
    with pytest.raises(ValueError, match="dspark_block_size=5"):
        D.Model(corrupt)
    partial_config = dict(_OFFICIAL_FILTERED, dspark_block_size=0)
    assert D.is_deepseek_v4_mtp_config(partial_config) is False
    assert D._has_dspark_signature(D.ModelArgs(dspark_block_size=0)) is True
    assert (
        D.is_deepseek_v4_mtp_config(
            {
                "model_type": "deepseek_v4",
                "num_nextn_predict_layers": 1,
                "dspark_block_size": 0,
            }
        )
        is False
    )


@pytest.mark.parametrize(
    "null_key",
    [
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    ],
)
def test_explicit_null_dspark_key_selects_validation_not_legacy_mtp(null_key):
    config = vars(_args()).copy()
    for key in (
        "dspark_block_size",
        "dspark_noise_token_id",
        "dspark_target_layer_ids",
        "dspark_markov_rank",
    ):
        config.pop(key)
    config[null_key] = None

    args = D.ModelArgs.from_dict(config)

    assert D._has_dspark_signature(args) is True
    with pytest.raises(ValueError, match="DSpark-0731"):
        D.Model(args)


def test_start_zero_runs_attention_only_on_all_three_stages(tiny_model):
    calls = []
    h = mx.zeros((1, 5, 4, 8), dtype=mx.float32)
    main_x = mx.zeros((1, 3, 8), dtype=mx.float32)

    class AttentionSpy:
        def __init__(self, stage_id):
            self.stage_id = stage_id

        def __call__(self, x, *, start_pos, main_x, cache):
            calls.append((self.stage_id, start_pos, cache))
            return x

    def forbidden(*_args, **_kwargs):
        raise AssertionError("HC/FFN path ran during DSpark prefill")

    for stage in tiny_model._dspark.stages:
        stage.attn = AttentionSpy(stage.stage_id)
        stage.attn_hc.pre = forbidden
        stage.ffn = forbidden
        cache = object()
        got = stage(h, start_pos=0, main_x=main_x, cache=cache)
        assert got is h
    assert [(stage_id, start_pos) for stage_id, start_pos, _ in calls] == [
        (0, 0),
        (1, 0),
        (2, 0),
    ]


def test_positive_start_runs_all_three_full_stages(tiny_model, monkeypatch):
    calls = []

    def full_stage(self, h, *, start_pos, cache=None, input_ids=None, main_x=None):
        calls.append((self.stage_id, start_pos, cache))
        return h

    monkeypatch.setattr(D.DeepseekV4DSparkStage, "__call__", full_stage)
    tiny_model._dspark.finish = lambda *_args, **_kwargs: "draft-output"
    result = tiny_model._dspark.forward(
        mx.zeros((1, 1, 24), dtype=mx.float32),
        mx.array([7], dtype=mx.int32),
        tiny_model.model.embed_tokens,
        tiny_model.lm_head,
        tiny_model.make_dspark_cache(),
        start_pos=9,
        greedy=True,
    )
    assert result == "draft-output"
    assert [(stage_id, start_pos) for stage_id, start_pos, _ in calls] == [
        (0, 9),
        (1, 9),
        (2, 9),
    ]


def test_ids_only_forward_projects_only_requested_m3_rows(tiny_model, monkeypatch):
    projected = []
    forced_primary = mx.array([19], dtype=mx.int32)
    forced_seen = []

    def full_stage(self, h, **_kwargs):
        return h

    def lm_head(rows):
        projected.append(tuple(rows.shape))
        return mx.zeros((rows.shape[0], rows.shape[1], 128800))

    monkeypatch.setattr(D.DeepseekV4DSparkStage, "__call__", full_stage)
    tiny_model._dspark.finish = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("ids-only forward used the confidence/logit-stack route")
    )

    def finish_ids(logits, _target_ids, *, width, forced_first_token_ids=None):
        forced_seen.append(forced_first_token_ids)
        return "ids-only", tuple(logits.shape), width

    tiny_model._dspark.finish_ids = finish_ids

    result = tiny_model._dspark.forward(
        mx.zeros((1, 1, 24), dtype=mx.float32),
        mx.array([7], dtype=mx.int32),
        tiny_model.model.embed_tokens,
        lm_head,
        tiny_model.make_dspark_cache(),
        start_pos=9,
        greedy=True,
        ids_only_width=3,
        forced_first_token_ids=forced_primary,
    )

    assert result == ("ids-only", (1, 3, 128800), 3)
    assert projected == [(1, 3, 8)]
    assert forced_seen == [forced_primary]


def test_zero_start_model_prefill_returns_no_draft_output(tiny_model, monkeypatch):
    calls = []

    def attention_only(self, h, *, start_pos, cache=None, input_ids=None, main_x=None):
        calls.append((self.stage_id, start_pos))
        return h

    monkeypatch.setattr(D.DeepseekV4DSparkStage, "__call__", attention_only)
    result = tiny_model._dspark.forward(
        mx.zeros((1, 3, 24), dtype=mx.float32),
        mx.array([7], dtype=mx.int32),
        tiny_model.model.embed_tokens,
        tiny_model.lm_head,
        tiny_model.make_dspark_cache(),
        start_pos=0,
    )
    assert result is None
    assert calls == [(0, 0), (1, 0), (2, 0)]


def test_dspark_prefill_seeds_all_three_stage_owned_caches(tiny_model):
    caches = tiny_model.make_dspark_cache()
    hidden = mx.arange(3 * 24, dtype=mx.float32).reshape(1, 3, 24)

    tiny_model._dspark.prefill(hidden, caches)

    mx.eval(*(cache.ring for cache in caches))
    assert [cache.prefill_length for cache in caches] == [3, 3, 3]
    assert all(cache.ring.shape == (1, 8, 8) for cache in caches)
    assert len({id(cache.ring) for cache in caches}) == 3


def test_dspark_commit_main_updates_every_stage_ring_with_accepted_prefix(tiny_model):
    dspark = tiny_model._dspark
    caches = tiny_model.make_dspark_cache()
    dspark.prefill(mx.zeros((1, 3, 24), dtype=mx.float32), caches)
    accepted = mx.arange(3 * 24, dtype=mx.float32).reshape(1, 3, 24)
    main_x = dspark.stages[0].fuse_main(accepted)
    positions = mx.arange(7, 10)
    expected = [stage.attn._kv(main_x, positions) for stage in dspark.stages]

    dspark.commit_main(accepted, caches, start_pos=7)

    mx.eval(*(cache.ring for cache in caches), *expected)
    for cache, stage_expected in zip(caches, expected):
        got = cache.ring[:, [7, 0, 1]]
        assert mx.allclose(got, stage_expected, rtol=1e-6, atol=1e-6)


def test_target_taps_are_hc_collapsed_in_exact_order(tiny_model):
    args = tiny_model.args
    model = tiny_model
    # Avoid a 43-layer numerical run: each selected layer emits a distinctive HC
    # tensor, and the target collector must collapse it at the layer boundary.
    for i in (40, 41, 42):
        value = float(i)
        model.model.layers[i] = lambda h, *a, _v=value, **k: mx.full(h.shape, _v)
    h = mx.zeros((1, 1, args.hc_mult, args.hidden_size))
    taps = model._collect_dspark_taps(h, start_layer=40)
    got = np.array(taps)
    assert got.shape == (1, 1, 3 * args.hidden_size)
    assert np.all(got[..., :8] == 40)
    assert np.all(got[..., 8:16] == 41)
    assert np.all(got[..., 16:] == 42)


def test_draft_ids_are_target_then_four_noise_tokens(tiny_model):
    dspark = tiny_model._dspark
    ids = dspark.draft_input_ids(mx.array([7, 9], dtype=mx.int32))
    assert np.array(ids).tolist() == [
        [7, 128799, 128799, 128799, 128799],
        [9, 128799, 128799, 128799, 128799],
    ]


def test_stage_caches_are_distinct_and_stage_owned(tiny_model):
    model = tiny_model
    caches = model.make_dspark_cache()
    assert len(caches) == 3
    assert len({id(c) for c in caches}) == 3
    assert all(type(c) is D.DeepseekV4DSparkCache for c in caches)
    assert [c.window_size for c in caches] == [8, 8, 8]


def test_dspark_missing_stage_weights_fail_at_load_boundary(tiny_model):
    with pytest.raises(ValueError, match=r"mtp\.1\.\*"):
        tiny_model.sanitize({"mtp.0.main_proj.weight": mx.zeros((8, 24))})


def test_markov_is_sequential_and_confidence_is_fp32(tiny_model):
    dspark = tiny_model._dspark
    # Make the Markov head depend only on the previous sampled token; this avoids
    # accidental dependence on the target logits while proving the recurrence.
    stage = dspark.stages[-1]
    stage.markov_head.markov_w1.weight = mx.arange(
        128800 * 3, dtype=mx.float32
    ).reshape(128800, 3)
    stage.markov_head.markov_w2.weight = mx.ones((128800, 3), dtype=mx.float32)
    logits = np.zeros((1, 5, 128800), dtype=np.float32)
    hidden = mx.ones((1, 5, 8), dtype=mx.float32)
    # Greedy target row selects 1, while row 1's Markov-biased output selects a
    # different token once the previous id is changed.
    logits[:, :, 1] = 1.0
    logits = mx.array(logits)
    ids_a, out_a, conf_a = dspark.finish(
        logits, hidden, mx.array([2], dtype=mx.int32), greedy=True
    )
    ids_b, out_b, conf_b = dspark.finish(
        logits, hidden, mx.array([3], dtype=mx.int32), greedy=True
    )
    assert np.array(ids_a)[:, 0].tolist() == [2]
    assert np.array(ids_b)[:, 0].tolist() == [3]
    assert not np.array_equal(np.array(out_a)[:, 0], np.array(out_b)[:, 0])
    assert conf_a.dtype == mx.float32 and conf_b.dtype == mx.float32


def test_greedy_ids_only_finish_seeds_future_recurrence_from_target_primary(
    tiny_model,
):
    dspark = tiny_model._dspark
    stage = dspark.stages[-1]

    class NextTokenMarkov:
        def __call__(self, previous):
            previous = np.asarray(previous, dtype=np.int32)
            bias = np.full((previous.shape[0], 128800), -1000.0, dtype=np.float32)
            bias[np.arange(previous.shape[0]), previous + 1] = 1000.0
            return mx.array(bias), mx.zeros((previous.shape[0], 1))

    stage.markov_head = NextTokenMarkov()
    logits = mx.zeros((1, 3, 128800), dtype=mx.float32)
    target_ids = mx.array([7], dtype=mx.int32)
    primary_ids = mx.array([19], dtype=mx.int32)

    ids = dspark.finish_ids(
        logits,
        target_ids,
        width=3,
        forced_first_token_ids=primary_ids,
    )

    mx.eval(ids)
    assert np.asarray(ids).tolist() == [[7, 19, 20, 21]]


def test_official_hc_names_map_to_exact_installed_parameter_keys(tiny_model):
    raw = {}
    expected = set()
    for stage in range(3):
        raw[f"mtp.{stage}.hc_attn_fn"] = mx.zeros((1,))
        raw[f"mtp.{stage}.hc_attn_base"] = mx.zeros((1,))
        raw[f"mtp.{stage}.hc_attn_scale"] = mx.zeros((1,))
        raw[f"mtp.{stage}.hc_ffn_fn"] = mx.zeros((1,))
        raw[f"mtp.{stage}.hc_ffn_base"] = mx.zeros((1,))
        raw[f"mtp.{stage}.hc_ffn_scale"] = mx.zeros((1,))
        expected |= {
            f"mtp.{stage}.attn_hc.fn",
            f"mtp.{stage}.attn_hc.base",
            f"mtp.{stage}.attn_hc.scale",
            f"mtp.{stage}.ffn_hc.fn",
            f"mtp.{stage}.ffn_hc.base",
            f"mtp.{stage}.ffn_hc.scale",
        }
    raw |= {
        "mtp.2.hc_head_fn": mx.zeros((1,)),
        "mtp.2.hc_head_base": mx.zeros((1,)),
        "mtp.2.hc_head_scale": mx.zeros((1,)),
    }
    expected |= {"mtp.2.hc_head.fn", "mtp.2.hc_head.base", "mtp.2.hc_head.scale"}
    mapped = tiny_model.sanitize(raw)
    assert set(mapped) == expected
    installed = {k for k, _ in tree_flatten(tiny_model.parameters())}
    assert expected <= installed


def test_dspark_grouped_o_lora_storage_flattens_at_load_boundary(tiny_model):
    raw = {
        "mtp.0.attn.wo_a.weight": mx.zeros((1, 8, 3), dtype=mx.uint32),
        "mtp.1.attn.wo_a.weight": mx.zeros((1, 8, 3), dtype=mx.uint32),
        "mtp.2.attn.wo_a.weight": mx.zeros((1, 8, 3), dtype=mx.uint32),
    }
    mapped = tiny_model.sanitize(raw)
    assert all(
        mapped[f"mtp.{stage}.attn.wo_a.weight"].shape == (8, 3) for stage in range(3)
    )


def test_dspark_visibility_is_exact_and_includes_all_five_draft_rows():
    got = np.array(D.get_dspark_topk_idxs(8, 2, 5, 3))
    expected = [0, 1, 2, 3, 8, 9, 10, 11, 12]
    assert got.shape == (2, 5, 9)
    assert got.tolist() == [[expected] * 5] * 2
    wrapped = np.array(D.get_dspark_topk_idxs(8, 1, 5, 10))
    assert wrapped.tolist() == [[list(range(8)) + [8, 9, 10, 11, 12]] * 5]


def _rms(x, eps=1e-6):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)


def _rope(x, positions, inverse=False):
    inv = 1.0 / (10000.0 ** (np.arange(0, 4, 2, dtype=np.float64) / 4))
    ang = np.asarray(positions, dtype=np.float64)[:, None] * inv[None]
    cos, sin = np.cos(ang), np.sin(ang)
    if inverse:
        sin = -sin
    out = x.copy()
    tail = x[..., -4:].reshape(*x.shape[:-1], 2, 2)
    a, b = tail[..., 0], tail[..., 1]
    trig_shape = (1, len(positions)) + (1,) * (a.ndim - 3) + (2,)
    rot = np.stack(
        [
            a * cos.reshape(trig_shape) - b * sin.reshape(trig_shape),
            a * sin.reshape(trig_shape) + b * cos.reshape(trig_shape),
        ],
        axis=-1,
    )
    out[..., -4:] = rot.reshape(*x.shape[:-1], 4)
    return out


def _identity_dspark_attention():
    attn = D.DeepseekV4DSparkAttention(_args(), 43)
    eye = mx.eye(8, dtype=mx.float32)
    attn.wq_a.weight = eye
    attn.wq_b.weight = eye
    attn.wkv.weight = eye
    attn.wo_a.weight = eye
    attn.wo_b.weight = eye
    attn.q_norm.weight = mx.ones((8,), dtype=mx.float32)
    attn.kv_norm.weight = mx.ones((8,), dtype=mx.float32)
    attn.attn_sink = mx.array([0.2], dtype=mx.float32)
    return attn


def _np_dspark_oracle(prefill, current, draft, start_pos, win=8):
    prekv = _rope(_rms(prefill), np.arange(prefill.shape[1]))
    if prefill.shape[1] <= win:
        ring = np.concatenate([prekv, np.zeros((1, win - prefill.shape[1], 8))], axis=1)
    else:
        last = prekv[:, -win:]
        cut = prefill.shape[1] % win
        ring = (
            last
            if cut == 0
            else np.concatenate([last[:, win - cut :], last[:, : win - cut]], axis=1)
        )
    mainkv = _rope(_rms(current), [start_pos])
    ring[:, start_pos % win : start_pos % win + 1] = mainkv
    positions = np.arange(start_pos + 1, start_pos + 6)
    q = _rms(_rms(draft))[:, :, None, :]
    q = _rope(q, positions)
    dkv = _rope(_rms(draft), positions)
    full = np.concatenate([ring, dkv], axis=1)
    idx = list(range(min(win, start_pos + 1))) + list(range(win, win + 5))
    visible = full[:, idx]
    scores = np.einsum("bshd,btd->bhst", q, visible) * (8**-0.5)
    sink = np.array(0.2).reshape(1, 1, 1, 1)
    maximum = np.maximum(scores.max(-1, keepdims=True), sink)
    exp = np.exp(scores - maximum)
    probs = exp / (exp.sum(-1, keepdims=True) + np.exp(sink - maximum))
    out = np.einsum("bhst,btd->bshd", probs, visible)
    return _rope(out, positions, inverse=True).reshape(1, 5, 8), ring


@pytest.mark.parametrize("prefill_len", [3, 10])
def test_dspark_attention_matches_prefill_decode_oracle_and_ring_wrap(prefill_len):
    rng = np.random.default_rng(13 + prefill_len)
    prefill = rng.normal(size=(1, prefill_len, 8)).astype(np.float32)
    current = rng.normal(size=(1, 1, 8)).astype(np.float32)
    draft = rng.normal(size=(1, 5, 8)).astype(np.float32)
    attn = _identity_dspark_attention()
    cache = D.DeepseekV4DSparkCache(8, 8)
    dummy = mx.zeros((1, 5, 8), dtype=mx.float32)
    assert tuple(
        attn(dummy, start_pos=0, main_x=mx.array(prefill), cache=cache).shape
    ) == (1, 5, 8)
    got = attn(
        mx.array(draft), start_pos=prefill_len, main_x=mx.array(current), cache=cache
    )
    ref, ref_ring = _np_dspark_oracle(prefill, current, draft, prefill_len)
    assert np.allclose(np.array(got), ref, rtol=3e-5, atol=3e-5)
    assert np.allclose(np.array(cache.ring), ref_ring, rtol=2e-6, atol=2e-6)


def test_dspark_cache_commits_an_accepted_prefix_across_ring_wrap():
    cache = D.DeepseekV4DSparkCache(4, 2)
    cache.prefill(mx.array([[[0.0, 0.5], [1.0, 1.5], [2.0, 2.5]]]))
    accepted = mx.array([[[30.0, 30.5], [40.0, 40.5], [50.0, 50.5]]])

    cache.commit_main(3, accepted)

    assert np.array(cache.ring).tolist() == [
        [[40.0, 40.5], [50.0, 50.5], [2.0, 2.5], [30.0, 30.5]]
    ]


def test_dspark_seeded_gumbel_matches_reference_and_greedy_is_explicit():
    logits = mx.array([[0.1, 1.2, -0.7, 0.8]], dtype=mx.float32)
    key = mx.random.key(90210)
    got = D._sample_dspark_token(logits, 0.7, key=key)
    uniform = mx.random.uniform(shape=logits.shape, key=key)
    uniform = mx.clip(uniform, 1e-30, 1.0 - mx.finfo(mx.float32).eps)
    ref = mx.argmax(logits / 0.7 - mx.log(-mx.log(uniform)), axis=-1)
    assert np.array_equal(np.array(got), np.array(ref))
    assert np.array(D._sample_dspark_token(logits, 0.7, greedy=True)).tolist() == [1]
