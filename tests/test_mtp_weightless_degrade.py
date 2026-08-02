"""MTP-declared-but-weightless models must degrade to AR at load, not crash.

Root cause: a checkpoint whose config declares MTP layers
(``num_nextn_predict_layers`` > 0) but whose conversion DROPPED the MTP
sidecar/embedded weights (e.g. ``mlx-community/DeepSeek-V4-Flash-2bit-DQ``)
used to raise ``RuntimeError: MTP injection failed`` at load. The injection
functions already ``return False`` when no MTP weights are present, but the
runtime gate treated that identically to a genuine injection failure.

The unit covered here is ``mtp_weights_present_on_disk`` — the disk-level probe
that lets the gate distinguish "no MTP weights ship with this model" (degrade to
AR) from "MTP weights are present but injection could not use them" (real
failure, still raises). It must be conservative: only return ``False`` when it
can positively confirm absence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mtplx.artifacts import mtp_weights_present_on_disk


def _write_index(tmp: Path, keys: list[str]) -> None:
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}),
        encoding="utf-8",
    )


def _trunk_keys(num_layers: int) -> list[str]:
    keys = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in range(num_layers):
        keys.append(f"model.layers.{i}.self_attn.q_proj.weight")
        keys.append(f"model.layers.{i}.mlp.gate_proj.weight")
    return keys


# --- the repro: MTP declared, but no MTP weights on disk -> probe False -------

def test_deepseek_v4_style_weightless_probe_is_false(tmp_path: Path) -> None:
    # 43 trunk layers (0..42), NO layer 43, no namespaced mtp.* keys.
    _write_index(tmp_path, _trunk_keys(43))
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "num_nextn_predict_layers": 1,
    }
    assert mtp_weights_present_on_disk(tmp_path, config) is False


# --- non-regression: probe True whenever MTP weights DO ship ------------------

def test_probe_true_when_mtp_sidecar_present(tmp_path: Path) -> None:
    _write_index(tmp_path, _trunk_keys(43))  # index has no mtp keys...
    (tmp_path / "mtp.safetensors").write_bytes(b"\x00")  # ...but a sidecar ships
    config = {"model_type": "deepseek_v3", "num_hidden_layers": 43, "num_nextn_predict_layers": 1}
    assert mtp_weights_present_on_disk(tmp_path, config) is True


def test_probe_true_when_namespaced_mtp_keys_embedded(tmp_path: Path) -> None:
    # Qwen/GLM/hy3-style: MTP weights live under an "mtp." namespace in shards.
    keys = _trunk_keys(48) + ["mtp.fc.weight", "mtp.norm.weight"]
    _write_index(tmp_path, keys)
    config = {"model_type": "qwen3_5_moe", "num_hidden_layers": 48, "num_nextn_predict_layers": 1}
    assert mtp_weights_present_on_disk(tmp_path, config) is True


def test_probe_true_when_deepseek_trailing_mtp_layer_embedded(tmp_path: Path) -> None:
    # DeepSeek-V3-style: MTP is a decoder layer appended after the trunk.
    keys = _trunk_keys(61) + [
        "model.layers.61.eh_proj.weight",
        "model.layers.61.enorm.weight",
        "model.layers.61.shared_head.norm.weight",
    ]
    _write_index(tmp_path, keys)
    config = {"model_type": "deepseek_v3", "num_hidden_layers": 61, "num_nextn_predict_layers": 1}
    assert mtp_weights_present_on_disk(tmp_path, config) is True


def test_probe_conservative_true_when_no_index_and_no_sidecar(tmp_path: Path) -> None:
    # Cannot cheaply prove absence -> preserve the legacy raise-on-failure path.
    config = {"model_type": "deepseek_v3", "num_hidden_layers": 43, "num_nextn_predict_layers": 1}
    assert mtp_weights_present_on_disk(tmp_path, config) is True


# --- evidence against the real (weightless) 2bit-DQ snapshot -----------------


def _real_2bit_dq_snapshot() -> Path | None:
    """Local HF cache path for the weightless 2bit-DQ build, if it is present."""
    hub = Path(
        os.environ.get("HUGGINGFACE_HUB_CACHE")
        or Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
        / "hub"
    )
    snapshots = hub / "models--mlx-community--DeepSeek-V4-Flash-2bit-DQ" / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir()):
        if (snapshot / "config.json").is_file():
            return snapshot
    return None


@pytest.mark.skipif(
    _real_2bit_dq_snapshot() is None, reason="real 2bit-DQ snapshot not in cache"
)
def test_real_2bit_dq_snapshot_probe_is_false() -> None:
    snapshot = _real_2bit_dq_snapshot()
    assert snapshot is not None
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    # Read-only: probe reads config.json + the shard index, never the weights.
    assert mtp_weights_present_on_disk(snapshot, config) is False
