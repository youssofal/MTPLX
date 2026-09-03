"""Forge asks tune for the depths the model's backend actually supports.

``_run_verify`` passed a fixed ``--depths 1,2,3`` and the completeness checks
required rows for AR/D1/D2/D3.  Both assume every backend drafts three tokens.
MiMo drafts one, so tune rejected the whole run with "tune depths must be one
of 1" and ``forge build`` exited 1 after a successful convert and calibrate.

Candidates that are not D-prefixed -- Gemma 4's draft blocks -- and any lookup
failure fall back to 1,2,3, leaving existing backends unchanged.
"""

from __future__ import annotations

from pathlib import Path

from mtplx.commands.forge import DEFAULT_FORGE_VERIFY_DEPTHS, _forge_verify_depths


def test_the_default_is_the_previous_hardcoded_list():
    assert DEFAULT_FORGE_VERIFY_DEPTHS == (1, 2, 3)


def test_an_unknown_model_keeps_the_default():
    assert _forge_verify_depths(Path("/nonexistent/not-a-model")) == (1, 2, 3)


def test_gemma_style_block_candidates_fall_back_rather_than_producing_nothing():
    # Gemma 4 advertises ("AR", "Block 2", ...); none are D-prefixed, so the
    # derivation must not hand tune an empty depth list.
    assert _forge_verify_depths(Path("/nonexistent/gemma4-assistant")) == (1, 2, 3)
