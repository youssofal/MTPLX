"""Bare-Speed engagement contracts for the FR-Spec + M=4 stage-3 optimizations.

The Bare-Speed pack differs from Optimized-Speed only in its quantization
recipe: lm_head Q4/g64 (not Q8/g64), shared experts Q4/g64 (not Q8/g64), and
routed experts Q4/g64 (not Q4/g32); the geometry is identical. These tests pin
that the server pack predicates and the stage-3 shared-expert installer
contracts accept BOTH packs and still reject anything else, and that the K20
pre-scatter gate is resolved at use (not frozen at import).

Pure Python + a few tiny mx arrays; nothing dispatches a real model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import mlx.core as mx

from mtplx.server import openai
from mtplx.qwen4_m4_stage3 import (
    _SHARED_DOWN_CONTRACTS,
    _SHARED_GATE_CONTRACTS,
    _SHARED_GU_CONTRACTS,
)

_MODULES = (
    "mlp.gate",
    "mlp.shared_expert_gate",
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
    "mlp.switch_mlp.gate_proj",
    "mlp.switch_mlp.up_proj",
    "mlp.switch_mlp.down_proj",
)


def _entry(bg: tuple[int, int]) -> dict[str, object]:
    bits, group = bg
    return {"bits": bits, "group_size": group, "mode": "affine"}


def _write_config(
    tmp_path,
    *,
    lm_head: tuple[int, int],
    shared: tuple[int, int],
    routed: tuple[int, int],
    gate: tuple[int, int] = (8, 64),
    layers: int = 48,
    tie: bool = False,
) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    quant: dict[str, object] = {"bits": 4, "group_size": 64}
    quant["language_model.lm_head"] = _entry(lm_head)
    per_module = {
        "mlp.gate": gate,
        "mlp.shared_expert_gate": shared,
        "mlp.shared_expert.gate_proj": shared,
        "mlp.shared_expert.up_proj": shared,
        "mlp.shared_expert.down_proj": shared,
        "mlp.switch_mlp.gate_proj": routed,
        "mlp.switch_mlp.up_proj": routed,
        "mlp.switch_mlp.down_proj": routed,
    }
    for index in range(layers):
        prefix = f"language_model.model.layers.{index}."
        for module, bg in per_module.items():
            quant[prefix + module] = _entry(bg)
    cfg = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": layers,
            "tie_word_embeddings": tie,
        },
        "quantization": quant,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return SimpleNamespace(model=str(tmp_path))


# Canonical per-pack geometries.
_BARE = dict(lm_head=(4, 64), shared=(4, 64), routed=(4, 64))
_OPTIMIZED_SPEED = dict(lm_head=(8, 64), shared=(8, 64), routed=(4, 32))


def test_lm_head_frspec_capable_accepts_both_packs(tmp_path) -> None:
    bare = _write_config(tmp_path / "bare", **_BARE)
    opt = _write_config(tmp_path / "opt", **_OPTIMIZED_SPEED)
    assert openai._served_model_lm_head_is_frspec_capable(bare) is True
    assert openai._served_model_lm_head_is_frspec_capable(opt) is True


def test_lm_head_frspec_capable_rejects_others(tmp_path) -> None:
    # g32 head, tied embeddings, and a missing pack all refuse.
    g32 = _write_config(tmp_path / "g32", lm_head=(4, 32), shared=(4, 64), routed=(4, 64))
    tied = _write_config(
        tmp_path / "tied", lm_head=(4, 64), shared=(4, 64), routed=(4, 64), tie=True
    )
    missing = SimpleNamespace(model=str(tmp_path / "does-not-exist"))
    assert openai._served_model_lm_head_is_frspec_capable(g32) is False
    assert openai._served_model_lm_head_is_frspec_capable(tied) is False
    assert openai._served_model_lm_head_is_frspec_capable(missing) is False


def test_stage3_geometry_accepts_both_packs(tmp_path) -> None:
    bare = _write_config(tmp_path / "bare", **_BARE)
    opt = _write_config(tmp_path / "opt", **_OPTIMIZED_SPEED)
    assert openai._served_model_pack_is_stage3_geometry(bare) is True
    assert openai._served_model_pack_is_stage3_geometry(opt) is True


def test_stage3_geometry_rejects_others(tmp_path) -> None:
    # A routed group size neither pack ships, and a shared-expert bit width
    # neither ships, both refuse the whole pack.
    bad_routed = _write_config(
        tmp_path / "r128", lm_head=(4, 64), shared=(4, 64), routed=(4, 128)
    )
    bad_shared = _write_config(
        tmp_path / "s6", lm_head=(4, 64), shared=(6, 64), routed=(4, 64)
    )
    assert openai._served_model_pack_is_stage3_geometry(bad_routed) is False
    assert openai._served_model_pack_is_stage3_geometry(bad_shared) is False


def test_stage3_geometry_rejects_one_bad_layer(tmp_path) -> None:
    # Every layer is checked: a single off-contract layer refuses the pack.
    args = _write_config(tmp_path / "bare", **_BARE)
    cfg = json.loads((tmp_path / "bare" / "config.json").read_text())
    cfg["quantization"]["language_model.model.layers.31.mlp.switch_mlp.down_proj"] = {
        "bits": 4,
        "group_size": 128,
        "mode": "affine",
    }
    (tmp_path / "bare" / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    assert openai._served_model_pack_is_stage3_geometry(args) is False


def _shared_gate_contract(bits: int, group: int, weight_cols: int, groups: int):
    return (
        bits,
        group,
        "affine",
        mx.uint32,
        mx.bfloat16,
        mx.bfloat16,
        (1, weight_cols),
        (1, groups),
        (1, groups),
    )


def test_shared_expert_installer_contracts_accept_both_bit_widths() -> None:
    # Optimized-Speed Q8/g64 and Bare-Speed Q4/g64 shared gate are both in the
    # accepted set; a Q4/g32 shared gate (wrong scale count) is not.
    assert _shared_gate_contract(8, 64, 640, 40) in _SHARED_GATE_CONTRACTS
    assert _shared_gate_contract(4, 64, 320, 40) in _SHARED_GATE_CONTRACTS
    assert _shared_gate_contract(4, 32, 320, 80) not in _SHARED_GATE_CONTRACTS
    # Membership sets carry exactly the two shipped geometries.
    assert len(_SHARED_GATE_CONTRACTS) == 2
    assert len(_SHARED_GU_CONTRACTS) == 2
    assert len(_SHARED_DOWN_CONTRACTS) == 2
    for contracts in (_SHARED_GATE_CONTRACTS, _SHARED_GU_CONTRACTS, _SHARED_DOWN_CONTRACTS):
        bit_widths = {c[0] for c in contracts}
        assert bit_widths == {4, 8}
        assert all(c[1] == 64 for c in contracts)


def test_k20_prescatter_gate_resolves_at_use(monkeypatch) -> None:
    # The auto-arm stamps MTPLX_QWEN4_DRAFT_K20_PRESCATTER AFTER this module is
    # imported; is_enabled() must read the environment at use so a late stamp
    # is still seen (it froze False at import before 2026-09-07).
    from mtplx import qwen4_draft_k20_prescatter as k20

    monkeypatch.setattr(k20, "_ENABLED", None)
    monkeypatch.delenv("MTPLX_QWEN4_DRAFT_K20_PRESCATTER", raising=False)
    assert k20.is_enabled() is False
    monkeypatch.setenv("MTPLX_QWEN4_DRAFT_K20_PRESCATTER", "1")
    assert k20.is_enabled() is True
    monkeypatch.setenv("MTPLX_QWEN4_DRAFT_K20_PRESCATTER", "0")
    assert k20.is_enabled() is False
