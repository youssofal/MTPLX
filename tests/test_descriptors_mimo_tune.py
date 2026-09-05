"""MiMo resolves to its own family and tunes at depth 1 only.

``mimo-mtp`` has shipped a native backend with ``can_run_verified=True``, but
``model_family_from_inspection`` had no branch returning "mimo", so every MiMo
artifact resolved to "unknown" and ``tune_policy_for_model`` refused it.
``forge build`` exited 1 at the tune gate after a successful convert, extract
and calibrate.

Depth is capped at D1: ``mimo_mtp_patch.mtp_forward`` raises on ``mtp_depth``
above 1, and vLLM's proposer is single-token as well, so offering D2+ would
advertise depths the backend refuses.  Measured on a forged MiMo-7B-RL pack on
an M3 Max: AR 69.69 tok/s, D1 91.38 tok/s (1.311x) at 69.2% acceptance.
"""

from __future__ import annotations

from mtplx.backends.descriptors import (
    model_family_from_inspection,
    tune_policy_for_model,
)

MIMO_INSPECTION = {
    "model_type": "mimo",
    "architecture": "MiMoForCausalLM",
    "mtp_arch": "mimo-mtp",
    "num_hidden_layers": 36,
    "mtp_num_hidden_layers": 1,
}


def test_mimo_resolves_to_the_mimo_family():
    assert model_family_from_inspection(MIMO_INSPECTION) == "mimo"


def test_tune_is_supported_for_mimo():
    assert tune_policy_for_model(inspection=MIMO_INSPECTION).supported is True


def test_mimo_offers_depth_one_only():
    policy = tune_policy_for_model(inspection=MIMO_INSPECTION)
    assert policy.candidates == ("AR", "D1")
    assert not any(c in policy.candidates for c in ("D2", "D3"))


def test_the_family_is_read_from_the_artifact_not_the_folder_name():
    # model_type and arch_id both feed the marker text, so a MiMo artifact is
    # recognised even when its directory has been renamed.
    assert model_family_from_inspection({"model_type": "mimo"}) == "mimo"
    assert model_family_from_inspection({"mtp_arch": "mimo-mtp"}) == "mimo"
