"""The architectures-declared model_type alias (Qwen3.8 day-one mechanism).

``mlx_lm.utils.load`` resolves the model class from ``model_type`` alone, so
a schema-compatible checkpoint under a fresh model_type string (Qwen3.6
shipped as ``qwen3_5``; Qwen3.8 is expected to repeat the pattern) would
hard-fail even though its config names the implementing class in
``architectures``. The runtime honors that declaration via a loud module
alias; these tests pin the contract.
"""

from __future__ import annotations

import sys

import pytest

pytest.importorskip("mlx_lm")

from mtplx.runtime import _install_architectures_declared_module_alias


@pytest.fixture(autouse=True)
def _clean_alias_registrations():
    before = set(sys.modules)
    try:
        yield
    finally:
        for name in set(sys.modules) - before:
            if name.startswith("mlx_lm.models."):
                del sys.modules[name]


def test_unknown_model_type_with_declared_qwen_architecture_aliases():
    config = {
        "model_type": "qwen3_x_alias_test",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
    }

    assert _install_architectures_declared_module_alias(config) is True

    import mlx_lm.models.qwen3_5 as qwen3_5

    assert sys.modules["mlx_lm.models.qwen3_x_alias_test"] is qwen3_5


def test_unknown_model_type_and_unknown_architecture_stays_fail_loud():
    config = {
        "model_type": "totally_new_arch_test",
        "architectures": ["SomeUnknownForCausalLM"],
    }

    assert _install_architectures_declared_module_alias(config) is False
    assert "mlx_lm.models.totally_new_arch_test" not in sys.modules


def test_native_model_type_is_left_alone():
    config = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
    }

    assert _install_architectures_declared_module_alias(config) is False


def test_alias_respects_text_config_nesting():
    config = {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "text_config": {"model_type": "qwen3_x_moe_alias_test"},
    }

    assert _install_architectures_declared_module_alias(config) is True

    import mlx_lm.models.qwen3_5_moe as qwen3_5_moe

    assert sys.modules["mlx_lm.models.qwen3_x_moe_alias_test"] is qwen3_5_moe


def test_alias_is_idempotent():
    config = {
        "model_type": "qwen3_x_idem_test",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
    }

    assert _install_architectures_declared_module_alias(config) is True
    assert _install_architectures_declared_module_alias(config) is False
