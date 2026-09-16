"""Expert pruning inside the qwen4_exp forge lane: keep-file validation and the
tensor slice on the sanitized (stacked switch_mlp) layout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.commands.forge_qwen4_exp import (
    Qwen4ForgeError,
    load_expert_keep,
    prune_expert_tensors,
    recipe_params,
)


def _keep_file(tmp_path: Path, k: int, layers=(0, 1), experts: int = 8, bad: bool = False) -> Path:
    keep = {"k": k, "experts": experts, "layers": list(layers),
            "keep": [list(range(k)) for _ in layers], "mtp_keep": list(range(k if not bad else k - 1))}
    path = tmp_path / "keep.json"
    path.write_text(json.dumps(keep))
    return path


def test_recipe_without_keep_is_unpruned() -> None:
    assert recipe_params({"body_bits": 4})["expert_keep"] is None


def test_keep_file_loads_and_validates(tmp_path: Path) -> None:
    keep = load_expert_keep({"qwen4_expert_keep": str(_keep_file(tmp_path, 4))})
    assert keep["k"] == 4 and keep["keep"] == [[0, 1, 2, 3], [0, 1, 2, 3]]
    with pytest.raises(Qwen4ForgeError, match="exactly k"):
        load_expert_keep({"qwen4_expert_keep": str(_keep_file(tmp_path, 4, bad=True))})


def test_prune_slices_experts_and_router_rows() -> None:
    mx = pytest.importorskip("mlx.core")
    E, inter, hid = 8, 6, 5
    w = {
        "language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight": mx.arange(E * inter * hid).reshape(E, inter, hid),
        "language_model.model.layers.3.mlp.switch_mlp.up_proj.weight": mx.ones((E, inter, hid)),
        "language_model.model.layers.3.mlp.switch_mlp.down_proj.weight": mx.ones((E, hid, inter)),
        "language_model.model.layers.3.mlp.gate.weight": mx.arange(E * hid).reshape(E, hid),
        "language_model.model.layers.3.self_attn.q_proj.weight": mx.ones((4, hid)),
    }
    keep = {"layers": [3], "keep": [[1, 5, 6]]}
    assert prune_expert_tensors(w, keep, prefix="language_model.model.") == 1
    g = w["language_model.model.layers.3.mlp.switch_mlp.gate_proj.weight"]
    assert g.shape == (3, inter, hid)
    assert mx.array_equal(g[0], mx.arange(E * inter * hid).reshape(E, inter, hid)[1])
    assert mx.array_equal(g[2], mx.arange(E * inter * hid).reshape(E, inter, hid)[6])
    r = w["language_model.model.layers.3.mlp.gate.weight"]
    assert r.shape == (3, hid) and mx.array_equal(r[1], mx.arange(E * hid).reshape(E, hid)[5])
    assert w["language_model.model.layers.3.mlp.switch_mlp.down_proj.weight"].shape == (3, hid, inter)
    # untouched tensors stay untouched; layers not in the keep map are left alone
    assert w["language_model.model.layers.3.self_attn.q_proj.weight"].shape == (4, hid)
