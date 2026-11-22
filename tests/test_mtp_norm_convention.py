"""MTP norm +1.0 convention is a whole-sidecar decision, never per-tensor (#301).

Synthetic gains mirror the measured fleet (2026-08-24): delta exports carry q/k
means near 0.75 and low-set means at 0.30-0.39; absolute sidecars sit at 1.73+
and 0.87+. The final norm overlaps across conventions (raw-delta 4B mean 2.58
vs absolute 3.8 mean 2.25) so it must follow the ensemble verdict.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mtplx.compressed_tensors import (
    mtp_sidecar_norms_are_delta,
    sanitize_plain_weight,
    shift_delta_mtp_norms,
)
from mtplx.mtp_patch import _heal_raw_delta_mtp_norms


def _gain(mean: float, size: int = 8) -> mx.array:
    return mx.array(np.full(size, mean, dtype=np.float32))


def _absolute_sidecar(prefix: str = "mtp.") -> dict[str, mx.array]:
    return {
        f"{prefix}layers.0.self_attn.q_norm.weight": _gain(1.79),
        f"{prefix}layers.0.self_attn.k_norm.weight": _gain(1.78),
        f"{prefix}layers.0.input_layernorm.weight": _gain(1.04),
        f"{prefix}layers.0.post_attention_layernorm.weight": _gain(1.21),
        f"{prefix}norm.weight" if prefix else "norm.weight": _gain(2.25),
        f"{prefix}layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
    }


def _delta_sidecar(prefix: str = "mtp.") -> dict[str, mx.array]:
    return {
        f"{prefix}layers.0.self_attn.q_norm.weight": _gain(0.746),
        f"{prefix}layers.0.self_attn.k_norm.weight": _gain(0.734),
        f"{prefix}layers.0.input_layernorm.weight": _gain(0.303),
        f"{prefix}layers.0.post_attention_layernorm.weight": _gain(0.387),
        f"{prefix}norm.weight" if prefix else "norm.weight": _gain(2.58),
        f"{prefix}layers.0.self_attn.q_proj.weight": mx.zeros((8, 8)),
    }


def test_absolute_sidecar_passes_through_untouched():
    weights = _absolute_sidecar()
    assert mtp_sidecar_norms_are_delta(weights) is False
    shifted = shift_delta_mtp_norms(weights)
    for key, value in weights.items():
        assert bool(mx.array_equal(shifted[key], value)), key


def test_delta_sidecar_shifts_all_seven_norm_gains():
    weights = _delta_sidecar()
    assert mtp_sidecar_norms_are_delta(weights) is True
    shifted = shift_delta_mtp_norms(weights)
    for key, value in weights.items():
        if key.endswith("proj.weight"):
            assert bool(mx.array_equal(shifted[key], value))
        else:
            assert bool(mx.array_equal(shifted[key], value + 1.0)), key
    # The final norm follows the ensemble verdict despite its high mean.
    assert float(shifted["mtp.norm.weight"].mean().item()) > 3.5


def test_stripped_key_namespace_matches_runtime_heal_shape():
    # The runtime loader strips the "mtp." namespace before healing.
    healed = _heal_raw_delta_mtp_norms(_delta_sidecar(prefix=""))
    assert float(healed["norm.weight"].mean().item()) > 3.5
    assert float(healed["layers.0.input_layernorm.weight"].mean().item()) > 1.2
    untouched = _heal_raw_delta_mtp_norms(_absolute_sidecar(prefix=""))
    assert float(untouched["norm.weight"].mean().item()) < 2.5


def test_missing_family_never_shifts():
    weights = _delta_sidecar()
    weights.pop("mtp.layers.0.self_attn.q_norm.weight")
    weights.pop("mtp.layers.0.self_attn.k_norm.weight")
    assert mtp_sidecar_norms_are_delta(weights) is False
    shifted = shift_delta_mtp_norms(weights)
    assert bool(
        mx.array_equal(shifted["mtp.norm.weight"], weights["mtp.norm.weight"])
    )


def test_sanitize_plain_weight_no_longer_shifts_mtp_norms():
    value = _gain(1.79)
    out = sanitize_plain_weight("mtp.layers.0.self_attn.q_norm.weight", value)
    assert bool(mx.array_equal(out, value))
    # Trunk suffixes keep their per-tensor shift.
    trunk = sanitize_plain_weight("model.layers.0.input_layernorm.weight", _gain(0.3))
    assert float(trunk.mean().item()) > 1.2


# --- one gain that missed its +1 ------------------------------------------------


def _unshifted_post_attention_sidecar(prefix: str = "mtp.") -> dict[str, mx.array]:
    """The shipped 35B-A3B shape: six gains absolute, post_attention zero-centred
    (mean about 0.87, one channel negative)."""

    weights = _absolute_sidecar(prefix)
    weights[f"{prefix}layers.0.post_attention_layernorm.weight"] = mx.array(
        np.array([0.87, 0.87, 0.87, 0.87, 0.87, 0.87, 1.98, -0.14], dtype=np.float32)
    )
    return weights


def test_a_single_unshifted_gain_alone_gets_its_plus_one(monkeypatch):
    from mtplx import compressed_tensors as ct

    monkeypatch.delenv(ct.MTP_NORM_REPAIR_ENV, raising=False)
    weights = _unshifted_post_attention_sidecar()
    assert mtp_sidecar_norms_are_delta(weights) is False
    assert ct.unshifted_mtp_norms(weights) == ["mtp.layers.0.post_attention_layernorm.weight"]
    healed = shift_delta_mtp_norms(weights)
    key = "mtp.layers.0.post_attention_layernorm.weight"
    assert bool(mx.array_equal(healed[key], weights[key] + 1.0))
    for other, value in weights.items():
        if other != key:
            assert bool(mx.array_equal(healed[other], value)), other
    assert [r["tensor"] for r in ct.LAST_MTP_NORM_REPAIRS] == [key]
    assert ct.LAST_MTP_NORM_REPAIRS[0]["mean_after"] > ct.LAST_MTP_NORM_REPAIRS[0]["mean_before"] + 0.99
    # The runtime heal (stripped namespace) repairs the same tensor.
    stripped = _heal_raw_delta_mtp_norms(_unshifted_post_attention_sidecar(prefix=""))
    assert float(stripped["layers.0.post_attention_layernorm.weight"].mean().item()) > 1.8


def test_the_repair_switch_loads_every_gain_as_stored(monkeypatch):
    from mtplx import compressed_tensors as ct

    monkeypatch.setenv(ct.MTP_NORM_REPAIR_ENV, "0")
    weights = _unshifted_post_attention_sidecar()
    healed = shift_delta_mtp_norms(weights)
    for key, value in weights.items():
        assert bool(mx.array_equal(healed[key], value)), key
    assert ct.LAST_MTP_NORM_REPAIRS == []


def test_healthy_and_all_delta_sidecars_get_no_per_tensor_repair(monkeypatch):
    from mtplx import compressed_tensors as ct

    monkeypatch.delenv(ct.MTP_NORM_REPAIR_ENV, raising=False)
    shift_delta_mtp_norms(_absolute_sidecar())
    assert ct.LAST_MTP_NORM_REPAIRS == []
    assert ct.unshifted_mtp_norms(_absolute_sidecar()) == []
    shift_delta_mtp_norms(_delta_sidecar())  # the set-level shift of all seven, as before
    assert ct.LAST_MTP_NORM_REPAIRS == []


def test_every_installed_head_is_repaired_only_where_a_gain_missed_its_shift():
    """Zero false positives over the heads on this Mac (skips where none are installed).

    Only the shipped Qwen3.6-35B-A3B Speed and Balance packs (and their FP16
    siblings) carry an unshifted gain; every other head is left as stored.
    Extra heads to check (a forged R1, a third-party sidecar) can be named in
    MTPLX_NORMSCAN_EXTRA, colon-separated paths to mtp safetensors files, and
    must all be left alone.
    """
    import os
    from pathlib import Path

    from mtplx import compressed_tensors as ct

    models = Path.home() / ".mtplx" / "models"
    heads = sorted(path for path in models.glob("*/mtp.safetensors") if path.is_file())
    extra = [Path(p) for p in os.environ.get("MTPLX_NORMSCAN_EXTRA", "").split(":") if p]
    if not heads and not extra:
        pytest.skip("no installed MTP heads on this machine")
    for path in heads + extra:
        weights = mx.load(str(path))
        flagged = ct.unshifted_mtp_norms(weights) if not mtp_sidecar_norms_are_delta(weights) else []
        name = path.parent.name
        expected = (
            ["mtp.layers.0.post_attention_layernorm.weight"]
            if path in heads and "Qwen3.6-35B-A3B-MTPLX-Optimized-" in name
            else []
        )
        assert flagged == expected, (str(path), flagged)
