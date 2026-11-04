"""``--scheduler-mode mtp_batch`` names the one model family it serves.

mtp_batch is the fixed-width MTP service for the Qwen 3.6 35B-A3B packs
(``mtplx/a3b_mtp_batch.py`` validates that exact architecture when the model
loads). On a Flash-Next pack the startup check said "mtp_batch requires
verify_strategy=target_prefix", and with ``--verify-strategy target_prefix``
Flash-Next coerces the strategy back to ``batched`` (target_prefix is a
qwen3-next structure lane), so the same refusal came back: a user on the
Flash-Next card found no flag set that works (Hugging Face discussion #4 on
Qwen3.8-Flash-Next-MTPLX-Optimized-Speed, 24 September).
"""

from __future__ import annotations

import json

import pytest

from mtplx.server import openai
from mtplx.server.openai import parse_args
from tests.test_server_openai import _flash_next_fixed_m4_config

# The reporter's launch, minus host, port and fan mode.
REPORTED_FLAGS = [
    "--profile",
    "turbo",
    "--context-window",
    "131072",
    "--scheduler-mode",
    "mtp_batch",
    "--generation-mode",
    "mtp",
    "--depth",
    "1",
    "--mtp-batch-numerics",
    "balanced",
    "--max-active-requests",
    "8",
    "--decode-batch-max",
    "8",
]
# Everything mtp_batch asks for.
FULL_CONTRACT = REPORTED_FLAGS + ["--verify-strategy", "target_prefix", "--verify-core", "stock"]


def _model_dir(tmp_path, config: dict):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(tmp_path)


def _startup_checks(argv: list[str]):
    """The two checks ServerState runs first, in its order."""

    args = parse_args(["--warmup-tokens", "0", *argv])
    openai._coerce_family_verify_strategy(args)
    openai._validate_mtp_batch_settings(args)
    return args


@pytest.fixture(autouse=True)
def _no_dependency_probe(monkeypatch):
    # The mlx-lm cache-leak probe is the last check and is not what this file tests.
    monkeypatch.setattr(openai, "_require_mlx_lm_arrays_cache_fix", lambda: None)


@pytest.mark.parametrize("argv", [REPORTED_FLAGS, FULL_CONTRACT], ids=["reported", "with_target_prefix"])
def test_flash_next_gets_one_refusal_that_names_the_family(tmp_path, argv):
    model = _model_dir(tmp_path, _flash_next_fixed_m4_config())
    with pytest.raises(RuntimeError) as refused:
        _startup_checks(["--model", model, *argv])
    message = str(refused.value)
    assert "35B-A3B" in message
    assert "qwen4_exp" in message
    assert "verify_strategy=target_prefix" not in message


def test_another_family_is_refused_before_the_model_loads(tmp_path):
    model = _model_dir(tmp_path, {"model_type": "qwen3_5", "text_config": {"model_type": "qwen3_5_text"}})
    with pytest.raises(RuntimeError, match="35B-A3B"):
        _startup_checks(["--model", model, *FULL_CONTRACT])


def test_the_35b_a3b_contract_is_unchanged(tmp_path):
    model = _model_dir(
        tmp_path,
        {"model_type": "qwen3_5_moe", "text_config": {"model_type": "qwen3_5_moe_text"}},
    )
    _startup_checks(["--model", model, *FULL_CONTRACT])
    wrong_depth = [flag if flag != "1" else "3" for flag in FULL_CONTRACT]
    with pytest.raises(RuntimeError, match="mtp_batch requires depth=1"):
        _startup_checks(["--model", model, *wrong_depth])


def test_an_unreadable_model_folder_keeps_the_contract_checks(tmp_path):
    missing = str(tmp_path / "not-downloaded")
    _startup_checks(["--model", missing, *FULL_CONTRACT])
    with pytest.raises(RuntimeError, match="mtp_batch requires verify_strategy=target_prefix"):
        _startup_checks(["--model", missing, *REPORTED_FLAGS])
