from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402

from mtplx.deepseek_v4_nvfp4_kv import FixedMiaNVFP4Ring  # noqa: E402
from mtplx.models import deepseek_v4 as target_module  # noqa: E402
from mtplx.models.deepseek_v4 import (  # noqa: E402
    DeepseekV4NVFP4Cache,
    DeepseekV4Model,
    Model,
    ModelArgs,
    is_deepseek_v4_mtp_config,
)
import mtplx.models.deepseek_v4_dspark as dspark_module  # noqa: E402
from mtplx.models.deepseek_v4_dspark import (  # noqa: E402
    DSparkTargetRoute,
    DeepseekV4DSparkAttention,
    DeepseekV4DSparkCache,
    DeepseekV4DSparkOwner,
    build_deepseek_v4_dspark,
    greedy_future_tokens,
)


class _SpyMarkovHead:
    def __init__(self) -> None:
        self.inputs: list[int] = []

    def __call__(self, token_ids: mx.array):
        self.inputs.append(int(token_ids.item()))
        batch = int(token_ids.shape[0])
        return mx.zeros((batch, 64)), mx.zeros((batch, 8))


class _SourceOrderHead:
    def __init__(self, calls: list[str], output: mx.array) -> None:
        self.calls = calls
        self.output = output

    def head(self, hidden, _head):
        self.calls.append("hc_head_bf16")
        assert hidden.dtype == mx.bfloat16
        return self.output


class _SourceOrderNorm:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def __call__(self, hidden):
        self.calls.append("rms_norm")
        assert hidden.dtype == mx.bfloat16
        return hidden + mx.array(1, dtype=mx.bfloat16)


def test_mia_target_collapse_preserves_source_bf16_then_norm_boundary() -> None:
    calls = []
    collapsed = mx.full((1, 1), 3, dtype=mx.bfloat16)
    owner = SimpleNamespace(
        _mia_mhc=_SourceOrderHead(calls, collapsed),
        hc_head=object(),
        norm=_SourceOrderNorm(calls),
        args=SimpleNamespace(hidden_size=1),
    )
    hidden = mx.zeros((1, 1, 4, 1), dtype=mx.bfloat16)

    output = DeepseekV4Model._mia_collapse(owner, hidden)
    mx.eval(output)

    assert calls == ["hc_head_bf16", "rms_norm"]
    np.testing.assert_array_equal(
        np.array(output.astype(mx.float32)),
        np.array([[[4]]], dtype=np.float32),
    )


def test_mia_dspark_head_preserves_source_bf16_then_norm_boundary() -> None:
    calls = []
    hidden_size = 1
    vocab_size = 8
    collapsed = mx.full((5, hidden_size), 3, dtype=mx.bfloat16)
    norm = _SourceOrderNorm(calls)

    def passthrough(value, **_kwargs):
        return value

    stages = [
        SimpleNamespace(
            attn_hc=object(),
            attn_norm=object(),
            ffn_hc=object(),
            ffn_norm=object(),
            attn=passthrough,
            ffn=passthrough,
            hc_head=None,
            norm=None,
            markov_head=None,
        )
        for _ in range(3)
    ]
    stages[-1].hc_head = object()
    stages[-1].norm = norm
    stages[-1].markov_head = lambda token_ids: (
        mx.zeros((token_ids.shape[0], vocab_size), dtype=mx.bfloat16),
        mx.zeros((token_ids.shape[0], 1), dtype=mx.bfloat16),
    )
    owner = DeepseekV4DSparkOwner.__new__(DeepseekV4DSparkOwner)
    owner.args = SimpleNamespace(hidden_size=hidden_size)
    owner.stages = stages
    owner._mia_mhc = _SourceOrderHead(calls, collapsed)
    owner._mia_draft_input_ids_k5 = lambda _primary: mx.zeros(
        (1, 5), dtype=mx.uint32
    )
    owner._mia_mhc.pre_broadcast = lambda embedded, *_args: (
        mx.zeros((5, 4, hidden_size), dtype=mx.bfloat16),
        mx.zeros((5, 4), dtype=mx.float32),
        mx.zeros((5, 4, 4), dtype=mx.float32),
        embedded,
    )
    owner._mia_mhc.post_pre_ffn = lambda value, residual, post, comb, *_args: (
        residual,
        post,
        comb,
        value,
    )
    owner._mia_mhc.post_pre_attn = lambda value, residual, post, comb, *_args: (
        residual,
        post,
        comb,
        value,
    )
    owner._mia_mhc.post = lambda _value, residual, _post, _comb: residual

    def lm_head(hidden):
        calls.append("lm_head")
        assert bool(mx.all(hidden == mx.array(4, dtype=mx.bfloat16)).item())
        return mx.zeros((1, 5, vocab_size), dtype=mx.bfloat16)

    DeepseekV4DSparkOwner._mia_propose_k5(
        owner,
        mx.zeros((1,), dtype=mx.uint32),
        lambda ids: mx.zeros((*ids.shape, hidden_size), dtype=mx.bfloat16),
        lm_head,
        [object(), object(), object()],
        start_pos=0,
    )

    assert calls == ["hc_head_bf16", "rms_norm", "lm_head"]


def test_primary_token_conditions_dspark_row_zero_and_returns_five_future_tokens() -> None:
    primary = mx.array([29], dtype=mx.int32)
    neural_logits = mx.full((1, 5, 64), -100.0)
    wanted = (31, 32, 33, 34, 35)
    for row, token in enumerate(wanted):
        neural_logits[:, row, token] = 100.0
    markov = _SpyMarkovHead()

    future = greedy_future_tokens(neural_logits, primary, markov)

    assert tuple(future.shape) == (1, 5)
    assert tuple(np.array(future)[0]) == wanted
    assert markov.inputs == [29, 31, 32, 33, 34]
    assert 11 not in markov.inputs
    assert 29 not in tuple(np.array(future)[0])


def test_each_dspark_stage_owns_distinct_mia_nvfp4_cache() -> None:
    caches = [DeepseekV4DSparkCache(window_size=8, head_dim=512) for _ in range(3)]
    assert len({id(cache) for cache in caches}) == 3
    assert len({id(cache.ring) for cache in caches}) == 3
    assert all(isinstance(cache.ring, FixedMiaNVFP4Ring) for cache in caches)
    assert all(cache.ring.mode == "nvfp4_stock432_fixed_ring" for cache in caches)
    assert all(cache.ring.record_bytes == 432 for cache in caches)

    prompt_latent = mx.zeros((1, 3, 512), dtype=mx.bfloat16)
    prompt_rope = mx.zeros((1, 3, 64), dtype=mx.bfloat16)
    caches[0].prefill(prompt_latent, prompt_rope)
    assert len(caches[0].ring) == 8
    assert len(caches[1].ring) == 0
    assert len(caches[2].ring) == 0


def test_dspark_cache_commits_authoritative_main_row_without_dense_owner() -> None:
    cache = DeepseekV4DSparkCache(window_size=8, head_dim=512)
    cache.prefill(
        mx.zeros((1, 8, 512), dtype=mx.bfloat16),
        mx.zeros((1, 8, 64), dtype=mx.bfloat16),
    )
    replacement_latent = ((mx.arange(512, dtype=mx.float32) % 37) / 11).reshape(
        1, 1, 512
    ).astype(mx.bfloat16)
    replacement_rope = ((mx.arange(64, dtype=mx.float32) - 11) / 9).reshape(
        1, 1, 64
    ).astype(mx.bfloat16)

    cache.commit_main(
        start_pos=2,
        main_latent=replacement_latent,
        main_rope=replacement_rope,
    )

    visible_key, visible_value = cache.visible_rows()
    expected_key, expected_value = cache.ring.decode(2, 3)
    np.testing.assert_array_equal(
        np.array(visible_value[:, 2:3].astype(mx.float32)),
        np.array(expected_value.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(visible_key[:, 2:3].astype(mx.float32)),
        np.array(expected_key.astype(mx.float32)),
    )
    np.testing.assert_array_equal(
        np.array(visible_key[:, 2:3, 448:].astype(mx.float32)),
        np.array(replacement_rope.astype(mx.float32)),
    )
    assert not hasattr(cache, "dense_ring")


def test_installed_dspark_k5_keeps_context_and_draft_records_separate() -> None:
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        captured = {}
        context_records = mx.zeros((1, 7, 432), dtype=mx.uint8)
        draft_records = mx.zeros((1, 5, 432), dtype=mx.uint8)

        def rope_tables(start, count):
            captured["rope_slice"] = (start, count)
            return (
                mx.arange(start, start + count, dtype=mx.int32),
                mx.ones((5, 32), dtype=mx.float32),
                mx.zeros((5, 32), dtype=mx.float32),
            )

        def project_learned(hidden):
            captured["projection_calls"] = (
                captured.get("projection_calls", 0) + 1
            )
            return (
                hidden,
                mx.zeros((1, 5, 512), dtype=mx.bfloat16),
            )

        def fail_old_route(*_args, **_kwargs):
            raise AssertionError("installed K5 called a legacy projection/RoPE route")

        def run_mla(*args, **kwargs):
            captured["mla_args"] = args
            captured["mla_kwargs"] = kwargs
            return mx.zeros((1, 5, 64, 512), dtype=mx.bfloat16)

        attn = DeepseekV4DSparkAttention.__new__(DeepseekV4DSparkAttention)
        attn.n_heads = 64
        attn.head_dim = 512
        attn.rope_head_dim = 64
        attn.window_size = 128
        attn.attn_sink = mx.zeros((64,), dtype=mx.float32)
        attn.softmax_scale = 512**-0.5
        attn.eps = 1e-6
        attn._mia_qkv_plan = None
        attn.wq_b = lambda value: mx.zeros(
            (*value.shape[:2], 64 * 512), dtype=mx.bfloat16
        )
        attn._mia_token_rope_tables = rope_tables
        attn._rope_tables = fail_old_route
        attn.wq_a = fail_old_route
        attn.wkv = fail_old_route
        attn.project_kv = fail_old_route
        attn._pack_draft_records = fail_old_route
        attn._dspark_k5_mla = run_mla
        attn._project_attention_output = lambda output, _cos, _sin: output

        attn.install_mia_k5_runtime()
        attn.install_mia_qkv_prologue(
            SimpleNamespace(
                project_learned=project_learned,
                proposal_records=lambda *_args: (
                    mx.zeros((1, 5, 64, 512), dtype=mx.bfloat16),
                    draft_records,
                ),
            )
        )
        output = attn(
            mx.zeros((1, 5, 8), dtype=mx.bfloat16),
            start_pos=17,
            cache=SimpleNamespace(
                ring=SimpleNamespace(records=context_records),
            ),
        )
        mx.eval(output)

        np.testing.assert_array_equal(
            np.array(attn._mia_draft_position_offsets),
            np.arange(5),
        )
        assert captured["rope_slice"] == (17, 5)
        assert captured["projection_calls"] == 1
        assert tuple(captured["mla_args"][0].shape) == (1, 5, 64, 512)
        assert captured["mla_args"][1] is context_records
        assert captured["mla_args"][2] is draft_records
        assert captured["mla_args"][3] == 17
        assert set(captured["mla_kwargs"]) == {"sinks", "scale"}
        assert attn._mia_mla_query_layout == "BMHD"
        assert attn._mia_mla_output_layout == "BMHD"
    finally:
        mx.set_default_device(previous)


def test_dspark_stages_share_boundary_rope_graph_after_provider_poison(
    monkeypatch,
) -> None:
    provider = target_module.MiaRoPETableProvider(
        mx.ones((32,), dtype=mx.float32),
        max_positions=384_005,
    )
    stages = []
    projection_calls = []

    def fail_unfused_kv(*_args, **_kwargs):
        raise AssertionError("exact DSpark context called the standalone wkv")

    for stage_id in range(3):
        attention = DeepseekV4DSparkAttention.__new__(
            DeepseekV4DSparkAttention
        )
        attention.compress_ratio = 0
        attention.rope_head_dim = 64
        attention._mia_rope_provider = None
        attention._mia_token_rope_tables = None
        attention.kv_norm = lambda value: value
        attention._mia_input_projection = (
            lambda value, stage_id=stage_id: (
                projection_calls.append(stage_id)
                or mx.zeros((*value.shape[:2], 1), dtype=value.dtype),
                value,
            )
        )
        attention.wkv = fail_unfused_kv
        attention.install_mia_rope_provider(provider)
        attention._project_kv_impl = attention._mia_project_kv
        stages.append(attention)

    provider.begin_forward()
    hidden = mx.zeros((1, 5, 512), dtype=mx.bfloat16)
    first_latent, first_rope = stages[0].project_kv(
        hidden,
        384_000,
    )
    cached_tables = provider.token_tables(384_000, 5)

    def fail_rebuild(*_args, **_kwargs):
        raise AssertionError("a DSpark stage rebuilt the shared RoPE graph")

    monkeypatch.setattr(target_module.mx, "cos", fail_rebuild)
    monkeypatch.setattr(target_module.mx, "sin", fail_rebuild)
    for stage in stages[1:]:
        latent, rope = stage.project_kv(hidden, 384_000)
        assert latent is first_latent
        np.testing.assert_array_equal(
            np.array(rope.astype(mx.float32)),
            np.array(first_rope.astype(mx.float32)),
        )
        assert provider.token_tables(384_000, 5) is cached_tables
    assert projection_calls == [0, 1, 2]


class _Layer:
    def __init__(self, layer_id: int) -> None:
        self.layer_id = layer_id

    def __call__(self, hidden, mask=None, cache=None, input_ids=None):
        del mask, cache, input_ids
        return mx.full(hidden.shape, self.layer_id, dtype=hidden.dtype)


class _Embedding:
    def __call__(self, input_ids):
        return mx.zeros((*input_ids.shape, 2), dtype=mx.float32)


def test_target_route_returns_ordered_40_41_42_taps() -> None:
    owner = SimpleNamespace(
        args=SimpleNamespace(hc_mult=2),
        model=SimpleNamespace(
            embed_tokens=_Embedding(),
            layers=[_Layer(layer_id) for layer_id in range(43)],
        ),
    )
    route = DSparkTargetRoute((40, 41, 42))

    final_hidden, taps = route(owner, mx.array([[7]], dtype=mx.int32), cache=None)

    assert tuple(final_hidden.shape) == (1, 1, 2, 2)
    assert len(taps) == 3
    assert all(tuple(tap.shape) == (1, 1, 2) for tap in taps)
    assert tuple(float(tap[0, 0, 0].item()) for tap in taps) == (40.0, 41.0, 42.0)


def test_dspark_owner_constructs_three_stages_and_primary_plus_four_noise_inputs(
    monkeypatch,
) -> None:
    class _FakeStage:
        def __init__(self, args, stage_id):
            del args
            self.stage_id = stage_id
            self.attn = SimpleNamespace(window_size=128, head_dim=512)

    monkeypatch.setattr(dspark_module, "DeepseekV4DSparkStage", _FakeStage)
    args = SimpleNamespace(
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_markov_rank=256,
        num_hidden_layers=43,
        num_nextn_predict_layers=1,
        vocab_size=129280,
        compress_ratios=[0] * 46,
    )

    owner = build_deepseek_v4_dspark(args)
    draft_inputs = owner.draft_input_ids(mx.array([29], dtype=mx.int32))
    caches = owner.make_cache()

    assert tuple(stage.stage_id for stage in owner.stages) == (0, 1, 2)
    assert tuple(np.array(draft_inputs)[0]) == (29, 128799, 128799, 128799, 128799)
    assert len(caches) == 3
    assert len({id(cache) for cache in caches}) == 3
    assert all(cache.ring.mode == "nvfp4_stock432_fixed_ring" for cache in caches)


def test_dspark_signature_installs_tap_route_and_mia_nvfp4_target_cache(monkeypatch) -> None:
    class _FakeAttention:
        window_size = 128
        compress_ratio = 0
        head_dim = 512

        def __init__(self) -> None:
            self.mia_nvfp4_installed = False

        def install_mia_nvfp4_attention(self) -> None:
            self.mia_nvfp4_installed = True

    class _FakeTarget(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.embed_tokens = _Embedding()
            self.layers = [_Layer(layer_id) for layer_id in range(args.num_hidden_layers)]
            for layer in self.layers:
                layer.attn = _FakeAttention()

        def collapse(self, hidden):
            return mx.mean(hidden, axis=2)

        def hc_hidden(self, input_ids, cache=None):
            del input_ids, cache
            raise AssertionError("the DSpark target route must replace hc_hidden")

    fake_stages = [SimpleNamespace(stage_id=stage_id) for stage_id in range(3)]
    fake_owner = SimpleNamespace(stages=fake_stages)
    monkeypatch.setattr(target_module, "DeepseekV4Model", _FakeTarget)
    monkeypatch.setattr(dspark_module, "build_deepseek_v4_dspark", lambda args: fake_owner)

    args = ModelArgs(
        vocab_size=129280,
        hidden_size=2,
        num_hidden_layers=43,
        hc_mult=2,
        compress_ratios=[0] * 46,
        dspark_block_size=5,
        dspark_noise_token_id=128799,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_markov_rank=256,
        num_nextn_predict_layers=1,
    )
    model = Model(args)

    logits, taps = model(
        mx.array([[7]], dtype=mx.int32),
        return_hidden=True,
        emit_logits=False,
    )
    caches = model.make_cache()

    assert logits is None
    assert model.dspark is fake_owner
    assert model.mtp == fake_stages
    assert model.has_mtp is False
    assert tuple(float(tap[0, 0, 0].item()) for tap in taps) == (40.0, 41.0, 42.0)
    assert len(caches) == 43
    assert all(isinstance(cache, DeepseekV4NVFP4Cache) for cache in caches)
    assert all(layer.attn.mia_nvfp4_installed for layer in model.model.layers)
    assert is_deepseek_v4_mtp_config(
        {
            "model_type": "deepseek_v4",
            "num_nextn_predict_layers": 1,
            "dspark_block_size": 5,
            "dspark_markov_rank": 256,
            "dspark_noise_token_id": 128799,
            "dspark_target_layer_ids": [40, 41, 42],
        }
    ) is False


def test_sanitize_flattens_real_dspark_grouped_o_lora_storage() -> None:
    model = SimpleNamespace(dspark=SimpleNamespace())
    grouped = mx.zeros((8, 1024, 768), dtype=mx.uint32)
    grouped_scales = mx.zeros((8, 1024, 32), dtype=mx.bfloat16)
    weights = {
        "model.layers.0.attn.wo_a.weight": grouped,
        "model.layers.0.attn.wo_a.scales": grouped_scales,
        "model.layers.0.attn.wo_a.biases": grouped_scales,
        "mtp.0.attn.wo_a.weight": grouped,
    }

    sanitized = Model.sanitize(model, weights)

    assert sanitized["model.layers.0.attn.wo_a.weight"].shape == (8192, 768)
    assert sanitized["model.layers.0.attn.wo_a.scales"].shape == (8192, 32)
    assert sanitized["model.layers.0.attn.wo_a.biases"].shape == (8192, 32)
    assert sanitized["mtp.0.attn.wo_a.weight"].shape == (8192, 768)
