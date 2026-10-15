"""Logit controls must not trim or bypass the injected DeepSeek trunk."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mtplx import deepseek_mtp_patch as patch
from mtplx.runtime import MTPLXRuntime


@pytest.fixture
def injected(monkeypatch, tmp_path):
    # Substitute weight storage and MLX allocation, not the injected __call__.
    mlx = ModuleType("mlx")
    core = ModuleType("mlx.core")
    core.eval = Mock()
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    monkeypatch.setattr(patch, "_deepseek_impl", lambda config: {"cache_factory": object})
    mtp = SimpleNamespace(load_weights=Mock(), parameters=list)
    monkeypatch.setattr(patch, "_make_deepseek_mtp_module", lambda config, args: mtp)
    monkeypatch.setattr(patch, "_candidate_weight_files", lambda path, config: [path / "mtp.safetensors"])
    monkeypatch.setattr(patch, "_load_weight_file", lambda path: {
        "layers.0.enorm.weight": 1,
        "layers.0.hnorm.weight": 1,
        "layers.0.eh_proj.weight": 1,
        "layers.0.mtp_block.self_attn.q_proj.weight": 1,
    })
    monkeypatch.setattr(patch, "_quantize_for_loaded_weights", lambda *args: None)

    inputs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    hidden = np.arange(24).reshape(2, 4, 3)
    cache = {"offset": 7}

    def trunk(tokens, state):
        assert tokens is inputs
        assert state is cache
        state["offset"] += tokens.shape[1]
        return hidden

    class Model:
        pass

    model = Model()
    model.args = SimpleNamespace(num_hidden_layers=2, n_routed_experts=0)
    model.model = Mock(side_effect=trunk)
    model.lm_head = Mock(side_effect=lambda rows: rows * 3 + 1)
    config = {"model_type": "deepseek_v3", "num_nextn_predict_layers": 1}
    assert patch.inject_deepseek_mtp_support(model, tmp_path, config)
    runtime = MTPLXRuntime(model, None, tmp_path, True, None)
    return SimpleNamespace(model=model, runtime=runtime, inputs=inputs, hidden=hidden, cache=cache)


@pytest.mark.parametrize("via_runtime", [False, True], ids=["injected", "runtime"])
@pytest.mark.parametrize("return_hidden", [False, True], ids=["logits", "with_hidden"])
@pytest.mark.parametrize(
    "controls, expected_rows",
    [
        ({}, 4),
        ({"emit_logits": True}, 4),
        ({"logits_keep": None}, 4),
        ({"logits_keep": 2}, 2),
        ({"logits_keep": 0}, 1),
        ({"logits_keep": -2}, 1),
        ({"logits_keep": 2.9}, 2),
        ({"logits_keep": 9}, 4),
        ({"emit_logits": False}, 0),
        ({"emit_logits": False, "logits_keep": 2}, 0),
    ],
)
def test_injected_logit_controls(injected, via_runtime, return_hidden, controls, expected_rows):
    case = injected
    call = case.runtime.forward_ar if via_runtime else case.model
    result = call(case.inputs, cache=case.cache, return_hidden=return_hidden, **controls)

    case.model.model.assert_called_once_with(case.inputs, case.cache)
    assert case.cache == {"offset": 11}
    if return_hidden:
        logits, hidden = result
        assert hidden is case.hidden
        np.testing.assert_array_equal(hidden, np.arange(24).reshape(2, 4, 3))
    else:
        logits = result

    if expected_rows == 0:
        case.model.lm_head.assert_not_called()
        assert logits is None
    else:
        case.model.lm_head.assert_called_once()
        projected = case.model.lm_head.call_args.args[0]
        expected = case.hidden[:, -expected_rows:, :]
        np.testing.assert_array_equal(projected, expected)
        np.testing.assert_array_equal(logits, expected * 3 + 1)
