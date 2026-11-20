"""Forge stamps verify's best depth and a served id for forge-local packs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mtplx.commands import forge
from mtplx.commands.public import _model_contract_depth
from mtplx.default_models import (
    BONSAI_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    forge_local_public_model_id,
    public_model_id_for_ref,
)

_MOE_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {"model_type": "qwen3_5_moe_text", "mtp_num_hidden_layers": 1},
}
_D2_WINS = [
    {"depth": 0, "tok_s": 90.0, "acceptance_by_position": []},
    {"depth": 1, "tok_s": 120.0, "acceptance_by_position": [0.8]},
    {"depth": 2, "tok_s": 135.0, "acceptance_by_position": [0.8, 0.55]},
    {"depth": 3, "tok_s": 125.0, "acceptance_by_position": [0.8, 0.55, 0.2]},
]
_AR_WINS = [
    {"depth": 0, "tok_s": 70.0, "acceptance_by_position": []},
    {"depth": 1, "tok_s": 40.0, "acceptance_by_position": [0.3]},
    {"depth": 2, "tok_s": 41.0, "acceptance_by_position": [0.3, 0.1]},
    {"depth": 3, "tok_s": 39.0, "acceptance_by_position": [0.3, 0.1, 0.0]},
]


def _pack(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(_MOE_CONFIG), encoding="utf-8")
    return path


def _stamp(path: Path, rows, *, existing=None, branded_name=None):
    return forge._stamp_runtime_metadata(
        path,
        branded_name=branded_name or path.name,
        source_repo="owner/source",
        source_sha="abc123",
        source_format=forge.SOURCE_MLX_AFFINE_WITH_MTP,
        recipe={},
        forge_inputs={"lane": "verify-stamp"},
        rows=forge._annotate_verify_rows([dict(row) for row in rows]),
        mtp_contract={},
        existing=existing,
    )


def test_stamp_records_verified_best_depth_as_depth_default(tmp_path):
    runtime = _stamp(_pack(tmp_path, "Qwen3.6-35B-A3B-Apodex-Balance"), _D2_WINS)

    assert runtime["mtp_depth_max"] == 3
    assert runtime["mtp_depth_default"] == 2
    assert runtime["mtp_depth_default_status"] == "forge_verified"


def test_serve_launches_a_stamped_pack_at_the_verified_depth(tmp_path):
    path = _pack(tmp_path, "Qwen3.6-35B-A3B-Apodex-Balance")
    (path / "mtplx_runtime.json").write_text(json.dumps(_stamp(path, _D2_WINS)), encoding="utf-8")
    inspection = {"model_dir": str(path), "config": _MOE_CONFIG}

    assert _model_contract_depth(inspection, profile=SimpleNamespace(name="sustained"), fallback=3) == 2


def test_stamp_keeps_a_depth_default_declared_by_the_source(tmp_path):
    path = _pack(tmp_path, "Qwen3.6-35B-A3B-Apodex-Balance")

    declared = _stamp(path, _D2_WINS, existing={"mtp_depth_default": 1})
    historical = _stamp(path, _D2_WINS, existing={"recommended_mtp_depth": 3})

    assert declared["mtp_depth_default"] == 1
    assert "mtp_depth_default_status" not in declared
    assert historical["recommended_mtp_depth"] == 3
    assert "mtp_depth_default" not in historical


def test_stamp_fills_a_null_historical_depth_key(tmp_path):
    runtime = _stamp(
        _pack(tmp_path, "Qwen3.6-35B-A3B-Apodex-Balance"),
        _D2_WINS,
        existing={"recommended_mtp_depth": None},
    )

    assert runtime["mtp_depth_default"] == 2


def test_restamp_replaces_or_drops_a_depth_forge_stamped_earlier(tmp_path):
    path = _pack(tmp_path, "Qwen3.6-35B-A3B-Apodex-Balance")
    earlier = {"mtp_depth_default": 3, "mtp_depth_default_status": "forge_verified"}

    remeasured = _stamp(path, _D2_WINS, existing=dict(earlier))
    ar_now_wins = _stamp(path, _AR_WINS, existing=dict(earlier))

    assert remeasured["mtp_depth_default"] == 2
    assert "mtp_depth_default" not in ar_now_wins
    assert "mtp_depth_default_status" not in ar_now_wins


def test_stamp_sets_no_depth_default_when_ar_wins(tmp_path):
    runtime = _stamp(_pack(tmp_path, "Qwen3.6-35B-A3B-Apodex-Balance"), _AR_WINS)

    assert "mtp_depth_default" not in runtime


def test_stamp_pins_a_served_id_from_the_branded_name(tmp_path):
    # _unique_model_dir appends "-1" when the branded name is taken; the
    # served id follows the branded name, not the directory.
    path = _pack(tmp_path, "Qwen3.6-35B-A3B-Apodex-Balance-1")
    runtime = _stamp(path, _D2_WINS, branded_name="Qwen3.6-35B-A3B-Apodex-Balance")
    (path / "mtplx_runtime.json").write_text(json.dumps(runtime), encoding="utf-8")

    assert runtime["public_model_id"] == "qwen3.6-35b-a3b-apodex-balance"
    renamed = path.rename(tmp_path / "copy of pack")
    assert public_model_id_for_ref(renamed) == "qwen3.6-35b-a3b-apodex-balance"


def test_stamp_keeps_an_existing_id_claim(tmp_path):
    path = _pack(tmp_path, "Qwen3.6-35B-A3B-Apodex-Balance")

    public = _stamp(path, _D2_WINS, existing={"public_model_id": "my-pack"})
    served = _stamp(path, _D2_WINS, existing={"served_model_id": "my-served-pack"})

    assert public["public_model_id"] == "my-pack"
    assert served["served_model_id"] == "my-served-pack"
    assert "public_model_id" not in served


def test_stamp_leaves_first_party_names_to_name_resolution(tmp_path):
    runtime = _stamp(_pack(tmp_path, "Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed"), _D2_WINS)

    assert "public_model_id" not in runtime


def test_forge_local_public_model_id_rules(tmp_path):
    assert forge_local_public_model_id("My_Pack v2", tmp_path / "x") == "my-pack-v2"
    assert forge_local_public_model_id("---", tmp_path / "x") is None
    assert forge_local_public_model_id("Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed", tmp_path / "x") is None
    # A link into the canonical store keeps its first-party identity.
    target = tmp_path / BONSAI_OPTIMIZED_SPEED_PUBLIC_MODEL_ID
    target.mkdir()
    link = tmp_path / "local-link"
    link.symlink_to(target)
    assert forge_local_public_model_id("local-link", link) is None
