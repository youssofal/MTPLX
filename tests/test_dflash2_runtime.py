from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace

import pytest

from mtplx.benchmarks import dflash2_runtime


EXPECTED_LAYER_IDS = (5, 19, 33, 47, 61)


@dataclass(frozen=True)
class FakeCapabilities:
    default_block_tokens: int = 5
    max_block_tokens: int = 5
    supports_copyspec: bool = False
    supports_ddtree: bool = True
    supports_early_rollback_launch: bool = False


def _draft(
    *,
    block_size=8,
    target_layer_ids=EXPECTED_LAYER_IDS,
):
    stored_layer_ids = (
        list(target_layer_ids)
        if isinstance(target_layer_ids, tuple)
        else target_layer_ids
    )
    return SimpleNamespace(
        block_size=block_size,
        target_layer_ids=stored_layer_ids,
        args=SimpleNamespace(block_size=block_size),
        capabilities=FakeCapabilities(),
    )


def _install_fakes(
    monkeypatch,
    *,
    supports_model: bool = True,
    family: str = "hybrid_gdn",
    draft_model=None,
):
    target = object()
    tokenizer = object()
    runtime = SimpleNamespace(model=target, tokenizer=tokenizer)
    target_ops = SimpleNamespace(
        supports_model=lambda model: supports_model and model is target,
        family=lambda model: family if model is target else "unexpected",
    )
    draft_model = draft_model or _draft()
    draft_meta = {"revision": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4"}
    draft_backend = object()
    calls = {"runtime": [], "draft": [], "bind": [], "backend": 0}

    def load_runtime(model_path):
        calls["runtime"].append(model_path)
        return runtime

    def load_draft(draft_ref, *, draft_quant):
        calls["draft"].append((draft_ref, draft_quant))
        return draft_model, draft_meta

    def bind_draft(bound_draft, bound_target, *, target_ops):
        calls["bind"].append((bound_draft, bound_target, target_ops))

    def make_backend():
        calls["backend"] += 1
        return draft_backend

    monkeypatch.setattr(dflash2_runtime, "load_mtplx_runtime", load_runtime)
    monkeypatch.setattr(dflash2_runtime, "load_draft", load_draft)
    monkeypatch.setattr(dflash2_runtime, "make_target_ops", lambda: target_ops)
    monkeypatch.setattr(dflash2_runtime, "bind_draft", bind_draft)
    monkeypatch.setattr(dflash2_runtime, "make_draft_backend", make_backend)
    return SimpleNamespace(
        runtime=runtime,
        target=target,
        tokenizer=tokenizer,
        target_ops=target_ops,
        draft=draft_model,
        draft_meta=draft_meta,
        draft_backend=draft_backend,
        calls=calls,
    )


def test_bundle_reuses_exact_runtime_target_and_binds_once(monkeypatch):
    values = _install_fakes(monkeypatch)

    bundle = dflash2_runtime.load_mtplx_dflash2_bundle(
        "speed",
        "z-lab/Qwen3.8-27B-DFlash2",
    )

    assert not hasattr(dflash2_runtime, "load_target")
    assert not hasattr(dflash2_runtime, "load_target_bundle")
    assert bundle.runtime is values.runtime
    assert bundle.target_model is values.runtime.model
    assert bundle.target_model is values.target
    assert bundle.tokenizer is values.tokenizer
    assert bundle.target_ops is values.target_ops
    assert bundle.draft_model is values.draft
    assert bundle.draft_backend is values.draft_backend
    assert bundle.draft_meta == values.draft_meta
    assert bundle.draft_meta is not values.draft_meta
    assert bundle.checkpoint_block_size == 8
    assert bundle.target_layer_ids == EXPECTED_LAYER_IDS
    assert bundle.draft_model.capabilities.default_block_tokens == 8
    assert bundle.draft_model.capabilities.max_block_tokens == 8
    assert bundle.draft_model.capabilities.supports_copyspec is False
    assert bundle.draft_model.capabilities.supports_ddtree is True
    assert bundle.draft_model.capabilities.supports_early_rollback_launch is False
    assert values.calls == {
        "runtime": ["speed"],
        "draft": [("z-lab/Qwen3.8-27B-DFlash2", "w4:gs64")],
        "bind": [(values.draft, values.target, values.target_ops)],
        "backend": 1,
    }
    with pytest.raises(FrozenInstanceError):
        bundle.target_model = object()


def test_mtplx_loader_loads_one_mtp_runtime(monkeypatch):
    from mtplx import runtime as runtime_module

    loaded = object()
    calls = []

    def fake_load(model_path, *, mtp):
        calls.append((model_path, mtp))
        return loaded

    monkeypatch.setattr(runtime_module, "load", fake_load)

    assert dflash2_runtime.load_mtplx_runtime("speed") is loaded
    assert calls == [("speed", True)]


def test_stock_draft_loader_and_binder_receive_exact_arguments(monkeypatch):
    from dflash_mlx.engine import target_ops as target_ops_module
    from dflash_mlx.runtime import loading as loading_module

    draft = object()
    target = object()
    target_ops = object()
    draft_result = (draft, {"revision": "pinned"})
    calls = {"draft": [], "bind": []}

    def fake_load_draft(draft_ref, *, lazy, draft_quant):
        calls["draft"].append((draft_ref, lazy, draft_quant))
        return draft_result

    def fake_bind(bound_draft, bound_target, *, target_ops):
        calls["bind"].append((bound_draft, bound_target, target_ops))

    monkeypatch.setattr(loading_module, "load_draft_bundle", fake_load_draft)
    monkeypatch.setattr(target_ops_module, "bind_draft_to_target", fake_bind)

    assert (
        dflash2_runtime.load_draft("draft-ref", draft_quant="w4:gs64")
        == draft_result
    )
    dflash2_runtime.bind_draft(draft, target, target_ops=target_ops)

    assert calls == {
        "draft": [("draft-ref", True, "w4:gs64")],
        "bind": [(draft, target, target_ops)],
    }


def test_stock_target_ops_and_draft_backend_types():
    from dflash_mlx.draft_backend import EagerDraftBackend
    from dflash_mlx.engine.target_qwen_gdn import QwenGdnTargetOps

    assert isinstance(dflash2_runtime.make_target_ops(), QwenGdnTargetOps)
    assert isinstance(dflash2_runtime.make_draft_backend(), EagerDraftBackend)


@pytest.mark.parametrize(
    ("supports_model", "family", "message"),
    [
        (False, "hybrid_gdn", "does not support"),
        (True, "pure_attention", "hybrid_gdn"),
    ],
)
def test_bundle_rejects_wrong_target_before_draft_or_bind(
    monkeypatch,
    supports_model,
    family,
    message,
):
    values = _install_fakes(
        monkeypatch,
        supports_model=supports_model,
        family=family,
    )

    with pytest.raises(ValueError, match=message):
        dflash2_runtime.load_mtplx_dflash2_bundle("speed", "draft")

    assert values.calls["draft"] == []
    assert values.calls["bind"] == []
    assert values.calls["backend"] == 0


@pytest.mark.parametrize(
    ("draft_model", "message"),
    [
        (_draft(block_size=5), "block size 8"),
        (_draft(target_layer_ids=(5, 19, 33, 47)), "target layer IDs"),
    ],
)
def test_bundle_rejects_wrong_checkpoint_geometry_before_bind(
    monkeypatch,
    draft_model,
    message,
):
    values = _install_fakes(monkeypatch, draft_model=draft_model)

    with pytest.raises(ValueError, match=message):
        dflash2_runtime.load_mtplx_dflash2_bundle("speed", "draft")

    assert values.calls["bind"] == []
    assert values.calls["backend"] == 0
    assert draft_model.capabilities.default_block_tokens == 5
    assert draft_model.capabilities.max_block_tokens == 5


@pytest.mark.parametrize(
    "draft_model",
    [
        _draft(block_size=8.0),
        _draft(block_size=8.9),
        _draft(block_size="8"),
        _draft(block_size=True),
        _draft(block_size=None),
        SimpleNamespace(
            target_layer_ids=list(EXPECTED_LAYER_IDS),
            capabilities=FakeCapabilities(),
        ),
    ],
    ids=("float-exact", "float-fractional", "string", "bool", "none", "missing"),
)
def test_bundle_rejects_non_integer_block_metadata_before_bind(
    monkeypatch,
    draft_model,
):
    values = _install_fakes(monkeypatch, draft_model=draft_model)

    with pytest.raises(ValueError, match="block size 8"):
        dflash2_runtime.load_mtplx_dflash2_bundle("speed", "draft")

    assert values.calls["bind"] == []
    assert values.calls["backend"] == 0


@pytest.mark.parametrize(
    "draft_model",
    [
        _draft(target_layer_ids=61),
        _draft(target_layer_ids=None),
        _draft(target_layer_ids=(5, 19, 33, 47, 61.0)),
        _draft(target_layer_ids=(5, 19, 33, 47, "61")),
        _draft(target_layer_ids=(5, 19, 33, 47, True)),
        SimpleNamespace(block_size=8, capabilities=FakeCapabilities()),
    ],
    ids=("integer", "none", "float-entry", "string-entry", "bool-entry", "missing"),
)
def test_bundle_rejects_malformed_layer_metadata_before_bind(
    monkeypatch,
    draft_model,
):
    values = _install_fakes(monkeypatch, draft_model=draft_model)

    with pytest.raises(ValueError, match="target layer IDs"):
        dflash2_runtime.load_mtplx_dflash2_bundle("speed", "draft")

    assert values.calls["bind"] == []
    assert values.calls["backend"] == 0
