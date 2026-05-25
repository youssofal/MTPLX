from __future__ import annotations

import pytest

from mtplx.mtp_patch import MTPContract, _stack_mtp_moe_experts


def test_mtp_contract_reads_config_quant_defaults() -> None:
    contract = MTPContract().with_config_defaults(
        {
            "mtplx_mtp_quantization": {
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                "prequantized": True,
            }
        }
    )

    assert contract.mtp_quant_policy == "cyankiwi"
    assert contract.mtp_quant_bits == 4
    assert contract.mtp_quant_group_size == 32
    assert contract.mtp_quant_mode == "affine"
    assert contract.mtp_prequantized is True


def test_mtp_contract_cli_bits_override_config_bits() -> None:
    contract = MTPContract(mtp_quant_bits=8).with_config_defaults(
        {
            "mtplx_mtp_quantization": {
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
            }
        }
    )

    assert contract.mtp_quant_bits == 8
    assert contract.mtp_quant_group_size == 32
    assert contract.mtp_quant_policy == "cyankiwi"


def test_mtp_contract_rejects_unknown_quant_policy() -> None:
    with pytest.raises(ValueError, match="mtp_quant_policy"):
        MTPContract(mtp_quant_policy="mystery").validate()


def test_stack_mtp_moe_experts_stacks_per_expert_into_switch_mlp() -> None:
    mx = pytest.importorskip("mlx.core")
    num_experts, out_dim, in_dim = 4, 6, 8
    config = {"text_config": {"num_experts": num_experts, "mtp_num_hidden_layers": 1}}
    weights = {"layers.0.input_layernorm.weight": mx.ones((in_dim,))}
    for e in range(num_experts):
        for proj, out in (("gate_proj", out_dim), ("up_proj", out_dim), ("down_proj", in_dim)):
            cols = in_dim if proj != "down_proj" else out_dim
            weights[f"layers.0.mlp.experts.{e}.{proj}.weight"] = mx.full((out, cols), float(e))

    stacked = _stack_mtp_moe_experts(weights, config)

    # Per-expert keys are consumed; stacked switch_mlp keys appear with a leading expert axis.
    assert not any(".experts." in k for k in stacked)
    assert tuple(stacked["layers.0.mlp.switch_mlp.gate_proj.weight"].shape) == (num_experts, out_dim, in_dim)
    assert tuple(stacked["layers.0.mlp.switch_mlp.down_proj.weight"].shape) == (num_experts, in_dim, out_dim)
    # Stacking preserves per-expert ordering (expert e was filled with value e).
    gate = stacked["layers.0.mlp.switch_mlp.gate_proj.weight"]
    for e in range(num_experts):
        assert float(gate[e, 0, 0].item()) == float(e)
    # Non-expert tensors pass through untouched.
    assert "layers.0.input_layernorm.weight" in stacked


def test_stack_mtp_moe_experts_is_noop_for_dense_head() -> None:
    mx = pytest.importorskip("mlx.core")
    config = {"text_config": {"num_experts": 0, "mtp_num_hidden_layers": 1}}
    weights = {"layers.0.mlp.gate_proj.weight": mx.ones((4, 4))}
    out = _stack_mtp_moe_experts(weights, config)
    assert "layers.0.mlp.switch_mlp.gate_proj.weight" not in out
    assert "layers.0.mlp.gate_proj.weight" in out
