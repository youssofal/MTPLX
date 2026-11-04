"""Flash-Next boots with ``--no-load-mtp`` (and ``--stock-ar``).

On the measured Flash-Next geometry the server stamps the fixed-M4 lane
defaults, and three of them are consumed by the model loader only together
with the native MTP head: the relaxed draft ties (``runtime.load`` raises
"relaxed Qwen4 draft ties require native MTP" without it), the compiled
draft-head preparation, and the FR-Spec pruned draft head. They were stamped
whatever ``--load-mtp`` said, so ``mtplx serve --generation-mode ar
--no-load-mtp`` refused to load the pack (an M1 Ultra in #400, 2.12.2). That
is the plain-decode fallback M1 and M2 owners use when speculative decoding
runs slower than plain decoding on their GPU (29 tok/s against 4 to 5 in
#400).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from mtplx.server import openai
from tests.test_server_openai import _flash_next_fixed_m4_config

MTP_HEAD_KEYS = (
    "MTPLX_QWEN4_RELAXED_DRAFT_TIES",
    "MTPLX_QWEN4_COMPILED_MTP_PREPARE",
    "MTPLX_FRSPEC_DRAFT",
    "MTPLX_FRSPEC_VOCAB",
    "MTPLX_QWEN4_DRAFT_K20_PRESCATTER",
    "MTPLX_QWEN4_SAMPLED_DRAFT_CHAIN",
)


def _optimized_speed_dir(tmp_path):
    """The fixed-M4 geometry with Optimized Speed's Q8/g64 lm_head, so the
    FR-Spec draft predicate holds as it does for the shipped pack."""

    config = _flash_next_fixed_m4_config()
    config["quantization"] = {
        "group_size": 32,
        "bits": 4,
        "mode": "affine",
        "language_model.lm_head": {"group_size": 64, "bits": 8, "mode": "affine"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    assert openai._served_model_is_qwen4_fixed_m4(SimpleNamespace(model=str(tmp_path)))
    return tmp_path


@pytest.fixture
def _clean_lane_env(monkeypatch):
    for key in (*openai._QWEN4_LANE_KEYS, *MTP_HEAD_KEYS):
        monkeypatch.delenv(key, raising=False)


def _args(model_dir, *, generation_mode: str, load_mtp: bool | None):
    args = SimpleNamespace(
        generation_mode=generation_mode,
        verify_strategy="batched",
        model=str(model_dir),
    )
    if load_mtp is not None:
        args.load_mtp = load_mtp
    return args


@pytest.mark.usefixtures("_clean_lane_env")
def test_no_load_mtp_stamps_no_gate_that_needs_the_mtp_head(tmp_path):
    model_dir = _optimized_speed_dir(tmp_path)
    overrides = openai._server_runtime_env_overrides(
        _args(model_dir, generation_mode="ar", load_mtp=False), {}
    )
    stamped = sorted(key for key in MTP_HEAD_KEYS if key in overrides)
    assert stamped == []
    # The rest of the lane still applies to plain decoding.
    assert overrides.get("MTPLX_QWEN4_FIXED_M4_VERIFY") == "1"
    assert overrides.get("MTPLX_QWEN4_PLE_PREFILL_LOOKAHEAD") == "1"


@pytest.mark.usefixtures("_clean_lane_env")
@pytest.mark.parametrize(
    ("generation_mode", "load_mtp"), [("mtp", True), ("ar", True), ("mtp", None)]
)
def test_with_the_mtp_head_loaded_the_lane_is_unchanged(tmp_path, generation_mode, load_mtp):
    model_dir = _optimized_speed_dir(tmp_path)
    overrides = openai._server_runtime_env_overrides(
        _args(model_dir, generation_mode=generation_mode, load_mtp=load_mtp), {}
    )
    for key in MTP_HEAD_KEYS:
        assert key in overrides, key
    assert overrides["MTPLX_QWEN4_RELAXED_DRAFT_TIES"] == "1"
    assert overrides["MTPLX_QWEN4_COMPILED_MTP_PREPARE"] == "1"
    assert overrides["MTPLX_FRSPEC_DRAFT"] == "1"


@pytest.mark.usefixtures("_clean_lane_env")
def test_an_operator_export_still_reaches_the_loader(monkeypatch, tmp_path):
    """An explicit export is the operator's call: it is not stamped (the
    loader reads it from the environment) and nothing here removes it."""

    model_dir = _optimized_speed_dir(tmp_path)
    monkeypatch.setenv("MTPLX_QWEN4_RELAXED_DRAFT_TIES", "1")
    overrides = openai._server_runtime_env_overrides(
        _args(model_dir, generation_mode="ar", load_mtp=False), {}
    )
    assert "MTPLX_QWEN4_RELAXED_DRAFT_TIES" not in overrides
    assert openai.os.environ["MTPLX_QWEN4_RELAXED_DRAFT_TIES"] == "1"
