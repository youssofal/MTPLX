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
    _drop_operator_switches()


# Variables that steer the suite itself (opt-in QA arms, artifacts to check,
# local model paths, golden regeneration) rather than MTPLX. Everything else
# under MTPLX_ is a runtime switch and is dropped before collection.
_SUITE_VARIABLES = frozenset(
    {
        "MTPLX_UPDATE_GOLDENS",
        "MTPLX_UPDATE_GOLDENS_ACK",
        "MTPLX_TEST_BASELINE",
        "MTPLX_TEST_TOKENIZER_DIR",
        "MTPLX_RUN_LONG_CONTEXT_QA",
        "MTPLX_LONG_CONTEXT_QA_ARTIFACT",
        "MTPLX_QA_BURST_BASELINE",
        "MTPLX_RUN_32K_MEMORY_QA",
        "MTPLX_32K_QA_MODEL",
        "MTPLX_DSV4_MTP_MODEL",
        "MTPLX_PREFILL_TPS_ARTIFACT",
        "MTPLX_PREFILL_TPS_REQUIRE_FULL",
        "MTPLX_SSE_DUMP",
    }
)

_dropped_switches: list[str] = []


def _drop_operator_switches():
    # An operator who runs the suite from the shell they serve from carries
    # their lane switches into it (MTPLX_SEGMENTED_KV=1, MTPLX_NAX_VERIFY=1,
    # chunk sizes, ...). Tests pin the defaults, and code that reads a switch
    # at import or caches it would keep the operator's value even under a
    # per-test monkeypatch, so the switches leave the process before any
    # test module is imported. Tests that exercise a switch set it themselves.
    for name in sorted(os.environ):
        if name.startswith("MTPLX_") and name not in _SUITE_VARIABLES:
            _dropped_switches.append(name)
            del os.environ[name]


def pytest_report_header(config):
    if _dropped_switches:
        return "ignored MTPLX switches from the environment: " + ", ".join(
            _dropped_switches
        )
    return None


@pytest.fixture(autouse=True)
def _hermetic_mtplx_state(monkeypatch, tmp_path_factory):
    isolated = tmp_path_factory.mktemp("hermetic-mtplx")
    monkeypatch.setenv("MTPLX_START_ATTACH_PROBE", "off")
    monkeypatch.setenv(
        "MTPLX_APP_SETTINGS_PATH", str(isolated / "app-settings.json")
    )
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(isolated / "models"))
    # `mtplx start` saves its setup to ~/.mtplx/quickstart.json and offers it
    # again on the next start; a test that reaches save_state without its own
    # path must not rewrite the developer's.
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(isolated / "quickstart.json"))
    # A real ~/.mtplx/config.toml outranks MTPLX_MODEL_DIR: its `model` changed
    # the default-model and bench dry runs, and its `model_dir` sent forge
    # builds into the user's real model cache. Tests that exercise the config
    # set their own MTPLX_CONFIG.
    monkeypatch.setenv("MTPLX_CONFIG", str(isolated / "config.toml"))
    # Synthetic server requests must not enter a live user's trace history.
    monkeypatch.setenv("MTPLX_REQUEST_LOG_JSONL", str(isolated / "requests.jsonl"))
    monkeypatch.setenv("MTPLX_FLIGHT_RECORDER", str(isolated / "flight.jsonl"))
    # The suite must never touch the developer's real OpenCode config. On
    # 2026-09-03 a full `pytest tests/` run rewrote
    # ~/.config/opencode/opencode.json mid-run with a fixture model id
    # (`mtplx-qwen38-27b-optimized-speed` as the only mtplx model), and every
    # `opencode run -m mtplx/mtplx-flash-next-optimized-speed` on the machine
    # failed with "Model not found" until the file was repaired by hand.
    # Tests that exercise the config writer set their own path; everyone
    # else writes into this scratch file.
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(isolated / "opencode.json"))
    # Pi's models.json and the extensions MTPLX installs beside it get the
    # same treatment: ~/.pi/agent is never written by the suite. So do the
    # OpenCode Desktop stores the OpenCode writer and launcher edit on macOS.
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(isolated / "pi" / "models.json"))
    monkeypatch.setenv(
        "MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(isolated / "opencode-desktop.dat")
    )
    monkeypatch.setenv(
        "MTPLX_OPENCODE_DESKTOP_APP_SUPPORT", str(isolated / "opencode-desktop")
    )
    # The system memory guard reads how much memory the kernel can still hand
    # out. On a developer machine that figure depends on what else is open,
    # so the suite sees "unknown" (the guard takes no action) unless a test
    # installs its own reading.
    import mtplx.system_memory as system_memory

    monkeypatch.setattr(system_memory, "_reader", lambda: None)
    for name in (
        "MTPLX_SYSTEM_MEMORY_GUARD",
        "MTPLX_SYSTEM_MEMORY_ABORT_FLOOR_BYTES",
        "MTPLX_SYSTEM_MEMORY_SHED_FLOOR_BYTES",
        "MTPLX_SYSTEM_MEMORY_REHEARSAL_AVAILABLE_BYTES",
        # The server writes the budget into os.environ at startup, so a test
        # that boots it with --memory-budget would size every later test's
        # session bank against that budget instead of the machine's RAM.
        "MTPLX_MEMORY_BUDGET",
        # Same for --allow-swap: a leaked value would switch it on for every
        # later server the suite boots.
        "MTPLX_ALLOW_SWAP",
    ):
        monkeypatch.delenv(name, raising=False)
    # `mtplx doctor` reads the app's failed-start report (#504). A developer
    # whose own app once failed to start must not get a different doctor
    # result from the suite than CI does.
    monkeypatch.setenv(
        "MTPLX_START_FAILURE_REPORT", str(isolated / "last-failed-start.log")
    )


@pytest.fixture
def legacy_rewrites(monkeypatch):
    """Run one test under the full legacy agent-rewrite machinery.

    #282 made the serving endpoints passthrough by default; tests that pin
    the opt-in machinery itself (compaction forms, heuristic drops/strips,
    toolset filtering, steering contracts, injected hints) request this
    fixture and keep their historical assertions unchanged.
    """
    monkeypatch.setenv("MTPLX_AGENT_REWRITES", "on")
