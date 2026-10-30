"""The suite runs on MTPLX's defaults and never touches the developer's live state.

tests/conftest.py drops the operator's MTPLX_* switches before collection
(PR #644), including MTPLX_SSD_SESSION_CACHE=off, so the SSD session bank a
default server opens must live in the test's scratch directory, not in the
developer's real ~/.mtplx/session-bank.
"""

from __future__ import annotations

import os
from pathlib import Path

from mtplx.server.openai import parse_args


def test_a_default_server_opens_a_scratch_session_bank():
    args = parse_args(["--model", "models/example", "--warmup-tokens", "0"])
    bank = Path(args.ssd_session_cache_dir).expanduser().resolve()
    assert bank != (Path.home() / ".mtplx" / "session-bank").resolve()
    assert "hermetic-mtplx" in str(bank)


def test_operator_lane_switches_do_not_reach_the_tests():
    # Run with MTPLX_SEGMENTED_KV=1 in the shell to see this matter. The
    # value is read into a name first: a failed assert on os.environ.get()
    # prints the whole environment, tokens included.
    segmented_kv = os.environ.get("MTPLX_SEGMENTED_KV")
    assert segmented_kv is None


def test_the_m1_m4_rehearsal_switch_reaches_the_tests():
    # The release gate runs a subset with MLX_METAL_GPU_ARCH=applegpu_g16s
    # and MTPLX_FORCE_GPU_FAMILY_FALLBACK=1; MLX keeps its variable, so the
    # pair must survive together or the subset rehearses nothing.
    if os.environ.get("MLX_METAL_GPU_ARCH") == "applegpu_g16s":
        fallback = os.environ.get("MTPLX_FORCE_GPU_FAMILY_FALLBACK")
        assert fallback == "1"
