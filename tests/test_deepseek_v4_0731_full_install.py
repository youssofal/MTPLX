"""Construction gates for the single receipt-backed 0731 target stack."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx  # noqa: E402

from mtplx import deepseek_v4_0731_full_install as full


@pytest.fixture(autouse=True)
def _cpu_default_device():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def full_artifact(tmp_path: Path, monkeypatch):
    config = {
        "model_type": "deepseek_v4",
        "hidden_size": 4096,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "swiglu_limit": 10.0,
        "num_nextn_predict_layers": 1,
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    index_bytes = b'{"metadata":{},"weight_map":{"model.layers.0":"a.safetensors"}}'
    (tmp_path / "config.json").write_bytes(config_bytes)
    (tmp_path / "model.safetensors.index.json").write_bytes(index_bytes)
    metadata_root = tmp_path / ".cache/huggingface/download"
    metadata_root.mkdir(parents=True)
    for name in ("config.json.metadata", "model.safetensors.index.json.metadata"):
        (metadata_root / name).write_text(
            f"{full.EXPECTED_SOURCE_REVISION}\nblob\ntimestamp\n",
            encoding="utf-8",
        )
    monkeypatch.setattr(
        full,
        "EXPECTED_FULL_CONFIG_SHA256",
        hashlib.sha256(config_bytes).hexdigest(),
    )
    monkeypatch.setattr(
        full,
        "EXPECTED_FULL_INDEX_SHA256",
        hashlib.sha256(index_bytes).hexdigest(),
    )
    return tmp_path, config


def _model():
    switches = [
        SimpleNamespace(gate_proj=object(), up_proj=object()) for _ in range(43)
    ]

    class WOBProjection(SimpleNamespace):
        def __call__(self, value):
            return value

    def attention():
        def qhead_stock(qr, _cos, _sin):
            return qr

        wq_b = SimpleNamespace(
            bits=6,
            group_size=128,
            mode="affine",
            bias=None,
            weight=SimpleNamespace(shape=(32768, 192), dtype=mx.uint32),
            scales=SimpleNamespace(shape=(32768, 8), dtype=mx.bfloat16),
            biases=SimpleNamespace(shape=(32768, 8), dtype=mx.bfloat16),
        )
        wo_b = WOBProjection(
            bits=6,
            group_size=128,
            mode="affine",
            bias=None,
            weight=SimpleNamespace(shape=(4096, 1536), dtype=mx.uint32),
            scales=SimpleNamespace(shape=(4096, 64), dtype=mx.bfloat16),
            biases=SimpleNamespace(shape=(4096, 64), dtype=mx.bfloat16),
        )
        return SimpleNamespace(
            wq_b=wq_b,
            _q_projection_qhead_route=qhead_stock,
            wo_b=wo_b,
            _o_lora_impl=SimpleNamespace(wo_b=wo_b),
        )

    layers = [
        SimpleNamespace(ffn=SimpleNamespace(switch_mlp=switch), attn=attention())
        for switch in switches
    ]
    stages = (object(), object(), object())
    stock_calls = []

    def stock(owner, input_ids, cache=None):
        stock_calls.append((owner, input_ids, cache))
        return "stock-hidden", "stock-taps"

    model = SimpleNamespace(
        model=SimpleNamespace(layers=layers),
        mtp=stages,
        _dspark=SimpleNamespace(target_layer_ids=(40, 41, 42), stages=stages),
        _target_hidden_route=stock,
    )
    return model, layers, switches, stock_calls


def test_artifact_contract_pins_config_index_and_both_hf_revisions(full_artifact):
    path, config = full_artifact

    contract = full.validate_full_0731_dspark_artifact(path, config)

    assert contract.layers == 43
    assert contract.target_layer_ids == (40, 41, 42)
    assert contract.stage_count == 3
    assert contract.source_revision == full.EXPECTED_SOURCE_REVISION
    assert contract.config_sha256 == full.EXPECTED_FULL_CONFIG_SHA256
    assert contract.index_sha256 == full.EXPECTED_FULL_INDEX_SHA256

    (path / "model.safetensors.index.json").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="index SHA-256"):
        full.validate_full_0731_dspark_artifact(path, config)


def test_artifact_contract_rejects_null_or_wrong_topology_and_revision(full_artifact):
    path, config = full_artifact
    with pytest.raises(ValueError, match="dspark_block_size"):
        full.validate_full_0731_dspark_artifact(
            path,
            {**config, "dspark_block_size": None},
        )

    metadata = (
        path / ".cache/huggingface/download/model.safetensors.index.json.metadata"
    )
    metadata.write_text("wrong-revision\nblob\ntimestamp\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata revision"):
        full.validate_full_0731_dspark_artifact(path, config)


def _install_with_spies(
    full_artifact,
    monkeypatch,
    *,
    prepare_only=False,
    fail_wob_prepare=False,
    fail_publish=None,
    fail_restore=None,
    prepare_wqb_override=None,
    prepare_wob_override=None,
):
    path, config = full_artifact
    model, layers, switches, stock_calls = _model()
    stock_route = model._target_hidden_route
    qhead_stocks = tuple(layer.attn._q_projection_qhead_route for layer in layers)
    wob_stocks = tuple(layer.attn.wo_b for layer in layers)
    replacements = [object() for _ in layers]
    validations = []
    bindings = []
    m3_layers = []
    row_owned = object()
    prepared = []
    publication = []
    combine_selfchecks = []
    projection_selfchecks = []

    monkeypatch.setattr(
        full,
        "validate_routed_q2_pair",
        lambda gate, up, **kwargs: validations.append((gate, up, kwargs)),
    )

    def build_pair(switch, **kwargs):
        index = switches.index(switch)
        return replacements[index]

    monkeypatch.setattr(full, "build_routed_q2_pair", build_pair)
    monkeypatch.setattr(
        full,
        "build_row_owned_combine_m1",
        lambda **kwargs: row_owned,
    )
    monkeypatch.setattr(
        full,
        "exact_selfcheck_row_owned_combine_m1",
        lambda combine: combine_selfchecks.append(combine),
    )

    def bind_tail(layer, **kwargs):
        index = layers.index(layer)
        assert [owned.ffn.switch_mlp for owned in layers] == switches
        assert kwargs.pop("routed_switch") is replacements[index]
        route = ("tail", index, kwargs["width"])
        bindings.append((layer, kwargs, route))
        return route

    monkeypatch.setattr(full.AI, "_bind_attention_island_layer", bind_tail)
    monkeypatch.setattr(
        full,
        "build_m3_compiled_tail_layer",
        lambda layer, tail: (
            m3_layers.append((layer, tail)) or ("m3-layer", layers.index(layer))
        ),
    )

    class M3Route:
        def __init__(self, base):
            self.base = base

        def __call__(self, owner, input_ids, cache=None):
            if tuple(input_ids.shape) == (1, 3):
                return "m3-hidden", "m3-taps"
            return self.base(owner, input_ids, cache)

    monkeypatch.setattr(
        full,
        "build_0731_m3_target_route",
        lambda owner, *, full_layer_routes, base_route: M3Route(base_route),
    )

    class Prepared:
        def __init__(self, label):
            self.label = label
            self.published_routes = tuple(object() for _ in range(43))
            self.q6_count = 43
            self.exact_selfchecked = 43
            self.o_lora_sink_count = 43

        def publish(self):
            publication.append(f"{self.label}.publish")
            if fail_publish == self.label:
                raise RuntimeError(f"{self.label} publication failed")

        def restore(self):
            publication.append(f"{self.label}.restore")
            if fail_restore == self.label:
                raise RuntimeError(f"{self.label} restoration failed")

    def prepare_wqb(layer_bank, *, exact_selfcheck):
        assert tuple(layer_bank) == tuple(layers)
        assert model._target_hidden_route is stock_route
        assert [layer.ffn.switch_mlp for layer in layers] == switches
        projection_selfchecks.append(("wqb", exact_selfcheck))
        prepared.append("wqb")
        return Prepared("wqb")

    def prepare_wob(layer_bank, *, exact_selfcheck):
        assert tuple(layer_bank) == tuple(layers)
        assert model._target_hidden_route is stock_route
        assert [layer.ffn.switch_mlp for layer in layers] == switches
        projection_selfchecks.append(("wob", exact_selfcheck))
        prepared.append("wob")
        if fail_wob_prepare:
            raise RuntimeError("wob self-check failed")
        return Prepared("wob")

    result = None
    error = None
    try:
        operation = (
            full.prepare_full_0731_dspark_compiled_tail_q2_pair
            if prepare_only
            else full.install_full_0731_dspark_compiled_tail_q2_pair
        )
        result = operation(
            model,
            config,
            path,
            prepare_wqb_qhead=prepare_wqb_override or prepare_wqb,
            prepare_wob=prepare_wob_override or prepare_wob,
        )
    except Exception as exc:  # asserted by the failure test
        error = exc
    return SimpleNamespace(
        result=result,
        error=error,
        model=model,
        layers=layers,
        switches=switches,
        replacements=replacements,
        stock_calls=stock_calls,
        stock_route=stock_route,
        qhead_stocks=qhead_stocks,
        wob_stocks=wob_stocks,
        validations=validations,
        bindings=bindings,
        m3_layers=m3_layers,
        prepared=prepared,
        publication=publication,
        combine_selfchecks=combine_selfchecks,
        projection_selfchecks=projection_selfchecks,
        row_owned=row_owned,
    )


def test_installer_publishes_only_the_receipt_backed_configuration(
    full_artifact,
    monkeypatch,
):
    state = _install_with_spies(full_artifact, monkeypatch)

    assert state.error is None
    assert len(state.validations) == 43
    assert len(state.bindings) == 86
    assert len(state.m3_layers) == 43
    assert state.prepared == ["wqb", "wob"]
    assert [label for label, _check in state.projection_selfchecks] == ["wqb", "wob"]
    assert all(callable(check) for _label, check in state.projection_selfchecks)
    assert state.combine_selfchecks == [state.row_owned]
    assert state.publication == ["wqb.publish", "wob.publish"]
    assert [layer.ffn.switch_mlp for layer in state.layers] == state.replacements
    assert all(
        kwargs
        == {
            "width": 1,
            "allowed_widths": (1,),
            "shared_bits": 8,
            "routed_pair": True,
            "routed_combine": state.row_owned,
        }
        for _layer, kwargs, _route in state.bindings[::2]
    )
    assert all(
        kwargs
        == {
            "width": 3,
            "allowed_widths": (3,),
            "shared_bits": 8,
            "routed_pair": True,
        }
        for _layer, kwargs, _route in state.bindings[1::2]
    )
    assert state.result == {
        "candidate": "mtplx-full-dspark-compiled-tail-packed-q2-pair-m1-m3",
        "artifact_label": full.RECORDED_ARTIFACT_LABEL,
        "validated_config_sha256": full.EXPECTED_FULL_CONFIG_SHA256,
        "validated_index_sha256": full.EXPECTED_FULL_INDEX_SHA256,
        "validated_metadata_revision": full.EXPECTED_SOURCE_REVISION,
        "layers_installed": 43,
        "decode_m": 1,
        "fixed_k": 2,
        "physical_target_rows": 3,
        "m3_tail": "fixed-width3-compiled-tail",
        "m3_wqb": {
            "candidate": "official-wheel-custom-fixed-m3-wqb-qhead-fused",
            "layers_installed": 43,
            "q6_g128_layers": 43,
            "exact_selfchecked_layers": 43,
            "shape": [1, 3, 1024],
            "output_shape": [1, 3, 64, 512],
        },
        "m3_wob": {
            "candidate": "official-wheel-custom-fixed-m3-affine-qmv",
            "layers_installed": 43,
            "q6_g128_layers": 43,
            "exact_selfchecked_layers": 43,
            "active_o_lora_sinks_installed": 43,
            "shape": [1, 3, 8192],
            "output_size": 4096,
        },
        "row_owned_combine": True,
        "non_m1_m3_route": "native-dspark",
        "routed_bits": 2,
        "routed_group_size": 128,
        "routed_gate_up_paired": True,
        "shared_bits": 8,
        "target_taps": (40, 41, 42),
        "dspark_stages": 3,
        "stage_ownership": "native",
    }


def test_failed_required_preparation_keeps_every_route_stock(
    full_artifact,
    monkeypatch,
):
    state = _install_with_spies(
        full_artifact,
        monkeypatch,
        fail_wob_prepare=True,
    )

    assert isinstance(state.error, RuntimeError)
    assert "wob self-check failed" in str(state.error)
    assert state.prepared == ["wqb", "wob"]
    assert [layer.ffn.switch_mlp for layer in state.layers] == state.switches
    assert state.publication == []
    ids = SimpleNamespace(shape=(1, 1))
    assert state.model._target_hidden_route(state.model, ids, "cache") == (
        "stock-hidden",
        "stock-taps",
    )


def test_prepared_target_stack_publishes_and_restores_as_one_transaction(
    full_artifact,
    monkeypatch,
):
    state = _install_with_spies(full_artifact, monkeypatch, prepare_only=True)

    assert state.error is None
    assert [layer.ffn.switch_mlp for layer in state.layers] == state.switches
    assert state.model._target_hidden_route is state.stock_route
    assert state.publication == []
    assert state.result.receipt["layers_installed"] == 43

    state.result.publish()
    assert [layer.ffn.switch_mlp for layer in state.layers] == state.replacements
    assert state.model._target_hidden_route is not state.stock_route
    assert state.publication == ["wqb.publish", "wob.publish"]

    state.result.restore()
    assert [layer.ffn.switch_mlp for layer in state.layers] == state.switches
    assert state.model._target_hidden_route is state.stock_route
    assert state.publication[-2:] == ["wob.restore", "wqb.restore"]


def test_publication_failure_restores_projection_q2_and_target_routes(
    full_artifact,
    monkeypatch,
):
    state = _install_with_spies(
        full_artifact,
        monkeypatch,
        fail_publish="wob",
    )

    assert isinstance(state.error, RuntimeError)
    assert "wob publication failed" in str(state.error)
    assert [layer.ffn.switch_mlp for layer in state.layers] == state.switches
    assert state.model._target_hidden_route is state.stock_route
    assert state.publication == [
        "wqb.publish",
        "wob.publish",
        "wob.restore",
        "wqb.restore",
    ]


def test_publication_rollback_attempts_every_restore_after_one_restore_fails(
    full_artifact,
    monkeypatch,
):
    state = _install_with_spies(
        full_artifact,
        monkeypatch,
        fail_publish="wob",
        fail_restore="wob",
    )

    assert isinstance(state.error, RuntimeError)
    assert "wob publication failed" in str(state.error)
    assert [layer.ffn.switch_mlp for layer in state.layers] == state.switches
    assert state.model._target_hidden_route is state.stock_route
    assert state.publication == [
        "wqb.publish",
        "wob.publish",
        "wob.restore",
        "wqb.restore",
    ]


def test_wob_receipt_rejects_incomplete_exact_selfcheck():
    receipt = SimpleNamespace(
        published_routes=tuple(object() for _ in range(43)),
        q6_count=43,
        exact_selfchecked=42,
        o_lora_sink_count=43,
    )

    with pytest.raises(ValueError, match="WOB preparation is not 43/43 exact"):
        full._require_wob_receipt(receipt)


def test_real_weight_projection_checks_require_three_exact_stock_m1_rows():
    qhead_rows = []

    def qhead_stock(qr, cos, sin):
        qhead_rows.append((tuple(qr.shape), tuple(cos.shape), tuple(sin.shape)))
        return qr

    qhead_check = full._m3_wqb_qhead_exact_selfcheck()
    assert qhead_check(qhead_stock, lambda qr, _cos, _sin: qr, 0) is True
    assert qhead_rows == [((1, 1, 1024), (1, 32), (1, 32))] * 3
    assert qhead_check(qhead_stock, lambda qr, _cos, _sin: qr + 1, 0) is False

    wob_check = full._m3_wob_exact_selfcheck()
    assert wob_check(lambda value: value, lambda value: value, 0) is True
    assert wob_check(lambda value: value, lambda value: value + 1, 0) is False


@pytest.mark.parametrize("failing_projection", ["wqb", "wob"])
def test_raw_projection_preparer_failure_keeps_every_live_route_stock(
    full_artifact,
    monkeypatch,
    failing_projection,
):
    from mtplx import deepseek_v4_0731_m3_wob as wob
    from mtplx import deepseek_v4_0731_m3_wqb_qnorm_rope as wqb

    built = 0

    def build_qhead(_projection):
        nonlocal built
        layer_index = built
        built += 1
        if failing_projection == "wqb" and layer_index == 9:
            return lambda qr, _cos, _sin: qr + 1
        return lambda qr, _cos, _sin: qr

    built_wob = 0

    def build_wob(_projection):
        nonlocal built_wob
        layer_index = built_wob
        built_wob += 1
        if failing_projection == "wob" and layer_index == 9:
            return lambda value: value + 1
        return lambda value: value

    monkeypatch.setattr(wqb, "build_0731_m3_wqb_qnorm_rope", build_qhead)
    monkeypatch.setattr(wob, "bind_m3_wob", build_wob)

    state = _install_with_spies(
        full_artifact,
        monkeypatch,
        prepare_wqb_override=wqb.prepare_wqb_qhead_m3,
        prepare_wob_override=wob.prepare_wob_m3,
    )

    expected_error = (
        wqb.M3WQBNormRopeContractError
        if failing_projection == "wqb"
        else wob.M3WOBContractError
    )
    assert isinstance(state.error, expected_error)
    assert "layer 9" in str(state.error)
    assert [layer.ffn.switch_mlp for layer in state.layers] == state.switches
    assert state.model._target_hidden_route is state.stock_route
    assert state.publication == []
    assert all(
        layer.attn._q_projection_qhead_route is stock
        for layer, stock in zip(state.layers, state.qhead_stocks)
    )
    assert all(
        layer.attn.wo_b is stock and layer.attn._o_lora_impl.wo_b is stock
        for layer, stock in zip(state.layers, state.wob_stocks)
    )


def test_installer_has_no_modes_or_optional_projection_fallbacks():
    signature = inspect.signature(full.install_full_0731_dspark_compiled_tail_q2_pair)
    assert tuple(signature.parameters) == (
        "model",
        "config",
        "model_path",
        "prepare_wqb_qhead",
        "prepare_wob",
    )
    assert signature.parameters["prepare_wqb_qhead"].default is inspect.Parameter.empty
    assert signature.parameters["prepare_wob"].default is inspect.Parameter.empty
    source = inspect.getsource(full)
    assert "m3_tail_mode" not in source
    assert "official-custom" not in source
    assert "row-exact-control" not in source
    assert "hybrid" not in source.lower()
    m1_route_source = inspect.getsource(full._FullDSparkTargetRoute.__call__)
    assert "tuple(" not in m1_route_source
    assert "shape[1]" in m1_route_source


def test_bound_m1_body_preserves_tap_order_and_cache_ownership(monkeypatch):
    class FakeMX:
        @staticmethod
        def broadcast_to(value, shape):
            assert shape == (1, 1, 4, 4096)
            return value

        @staticmethod
        def mean(value, axis):
            assert axis == 2
            return f"mean-{value}"

        @staticmethod
        def concatenate(values, *, axis):
            assert axis == -1
            return tuple(values)

    class Layer:
        def __init__(self, layer_id):
            self.attn_hc = SimpleNamespace(pre=lambda hidden: (hidden, "post", "comb"))
            self.attn_norm = lambda hidden: hidden
            self.cache_entries = []

            def attention(hidden, **kwargs):
                self.cache_entries.append(kwargs["cache"])
                return hidden

            self.attn = attention

    class Tail:
        def __init__(self, layer_id):
            self.layer_id = layer_id

        def __call__(self, *_args):
            return f"h{self.layer_id}"

    class Embedded:
        shape = (1, 1, 4096)

        def __getitem__(self, _key):
            return self

    monkeypatch.setattr(full, "mx", FakeMX)
    layer_tails = tuple((Layer(i), Tail(i)) for i in range(43))
    body = SimpleNamespace(
        embed_tokens=lambda _ids: Embedded(),
        hc_mult=4,
    )
    candidate = full._BoundDSparkM1Body(body, layer_tails, (40, 41, 42))

    class CacheEntries:
        def __iter__(self):
            return (f"cache-{i}" for i in range(43))

        def __len__(self):
            raise AssertionError("bound route must not revalidate cache length")

    source = inspect.getsource(full._BoundDSparkM1Body.__call__)
    assert "len(entries)" not in source
    assert "strict=True" not in source
    assert "missed a DSpark tap" not in source

    hidden, taps = candidate(SimpleNamespace(shape=(1, 1)), CacheEntries())

    assert hidden == "h42"
    assert taps == ("mean-h40", "mean-h41", "mean-h42")
    assert [layer.cache_entries for layer, _tail in layer_tails] == [
        [f"cache-{i}"] for i in range(43)
    ]
