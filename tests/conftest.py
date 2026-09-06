"""Suite-wide isolation from the developer machine's live MTPLX state.

Without these guards the suite is machine-dependent: a daemon left running
by the macOS app would make ``mtplx start`` flows offer attach prompts, the
real ``~/Library/Application Support/MTPLX/settings.json`` would inject
"same as the app" options, and the real ``~/.mtplx/models`` cache would
change picker numbering. Tests that exercise those features explicitly
override these variables with their own fixtures.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config):
    # A leaked MTPLX_UPDATE_GOLDENS=1 turns every golden test into a
    # write-then-return no-op and the suite reports green with zero
    # verification. Regeneration is a deliberate local act: run it as
    # MTPLX_UPDATE_GOLDENS=1 MTPLX_UPDATE_GOLDENS_ACK=yes pytest ...
    if os.environ.get("MTPLX_UPDATE_GOLDENS") and not os.environ.get(
        "MTPLX_UPDATE_GOLDENS_ACK"
    ):
        raise pytest.UsageError(
            "MTPLX_UPDATE_GOLDENS is set: golden tests would silently skip "
            "comparison. Unset it, or acknowledge regeneration explicitly "
            "with MTPLX_UPDATE_GOLDENS_ACK=yes."
        )


@pytest.fixture(autouse=True)
def _hermetic_mtplx_state(monkeypatch, tmp_path_factory):
    isolated = tmp_path_factory.mktemp("hermetic-mtplx")
    monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "off")
    monkeypatch.setenv(
        "MTPLX_APP_SETTINGS_PATH", str(isolated / "app-settings.json")
    )
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(isolated / "models"))
    # The suite must never touch the developer's real OpenCode config. On
    # 2026-09-03 a full `pytest tests/` run rewrote
    # ~/.config/opencode/opencode.json mid-run with a fixture model id
    # (`mtplx-qwen38-27b-optimized-speed` as the only mtplx model), and every
    # `opencode run -m mtplx/mtplx-flash-next-optimized-speed` on the machine
    # failed with "Model not found" until the file was repaired by hand.
    # Tests that exercise the config writer set their own path; everyone
    # else writes into this scratch file.
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(isolated / "opencode.json"))
    # Same story for the user config: it resolves to ~/.mtplx/config.toml unless
    # MTPLX_CONFIG points elsewhere, so whatever default model the developer last
    # served silently changed tests that assert the built-in default. Observed
    # 2026-09-06: two `test_public_cli.py` legacy-path tests failed on a machine
    # whose config.toml named a Flash-Next pack, and passed under an empty HOME.
    monkeypatch.setenv("MTPLX_CONFIG", str(isolated / "config.toml"))
    # A CLI dispatch mutates the process environment: `apply_profile_env` writes a
    # whole profile into os.environ (that is how the daemon child inherits it) and
    # the one-shot paths never restore what it returned. `monkeypatch` only undoes
    # what a test set through it, so the snapshot has to cover raw writes too —
    # otherwise every later test in the session inherits knobs such as
    # MTPLX_SKIP_VERIFY_SNAPSHOT and MTPLX_LAZY_TARGET_DISTRIBUTIONS.
    saved_environment = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved_environment)


@pytest.fixture
def legacy_rewrites(monkeypatch):
    """Run one test under the full legacy agent-rewrite machinery.

    #282 made the serving endpoints passthrough by default; tests that pin
    the opt-in machinery itself (compaction forms, heuristic drops/strips,
    toolset filtering, steering contracts, injected hints) request this
    fixture and keep their historical assertions unchanged.
    """
    monkeypatch.setenv("MTPLX_AGENT_REWRITES", "on")
