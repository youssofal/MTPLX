"""--cascade-threshold CLI flag + the /health cascade_acceptance entry. CPU-only.

mtplx/server/openai.py imports MLX, so the /health payload helper is compiled out
of the shipped source (the same trick test_typical_threshold_cli_health_cpu.py
uses) instead of importing the module, and the CLI wiring is asserted against the
source text.
"""
from __future__ import annotations

import argparse
import ast
import os
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVER_TEXT = (ROOT / "mtplx" / "server" / "openai.py").read_text("utf-8")


def _compile_function(source: str, name: str, namespace: dict | None = None):
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    scope: dict = {"Any": Any, "os": os, "argparse": argparse}
    scope.update(namespace or {})
    exec(compile(module, f"<{name}>", "exec"), scope)
    return scope[name]


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in ("MTPLX_FABLE_CASCADE_THRESHOLD", "MTPLX_FABLE_TYPICAL_THRESHOLD"):
        monkeypatch.delenv(key, raising=False)


def _payload():
    return _compile_function(SERVER_TEXT, "_cascade_acceptance_health_payload")()


def test_health_reports_off_by_default():
    p = _payload()
    assert p["enabled"] is False
    assert p["alpha"] is None
    assert p["distribution_exact"] is True
    assert p["conflict"] is False
    assert p["citation"] == "arXiv:2405.19261 Eq. (10)"


def test_health_reports_on_when_set_including_zero(monkeypatch):
    monkeypatch.setenv("MTPLX_FABLE_CASCADE_THRESHOLD", "0")
    p = _payload()
    assert p["enabled"] is True  # explicit 0 is ON
    assert p["alpha"] == 0.0
    assert p["distribution_exact"] is False
    monkeypatch.setenv("MTPLX_FABLE_CASCADE_THRESHOLD", "0.5")
    p = _payload()
    assert p["enabled"] is True
    assert p["alpha"] == 0.5


def test_health_flags_conflict_when_both_lanes_set(monkeypatch):
    monkeypatch.setenv("MTPLX_FABLE_CASCADE_THRESHOLD", "0.3")
    monkeypatch.setenv("MTPLX_FABLE_TYPICAL_THRESHOLD", "0.09")
    p = _payload()
    assert p["enabled"] is True
    assert p["conflict"] is True


def test_health_bad_value_is_not_fatal(monkeypatch):
    monkeypatch.setenv("MTPLX_FABLE_CASCADE_THRESHOLD", "sometimes")
    p = _payload()  # must not raise
    assert p["enabled"] is False
    assert p["alpha"] is None


def test_health_entry_is_wired_into_the_route():
    assert '"cascade_acceptance": _cascade_acceptance_health_payload(),' in SERVER_TEXT


def test_cli_flag_defined_and_stamps_env():
    assert '"--cascade-threshold",' in SERVER_TEXT
    assert 'os.environ["MTPLX_FABLE_CASCADE_THRESHOLD"] = str(float(args.cascade_threshold))' in SERVER_TEXT


def test_cli_mutual_exclusion_is_wired():
    # The flag-apply raises SystemExit when both lossy rules are set.
    assert "mutually exclusive lossy verify rules" in SERVER_TEXT
    assert "SystemExit(" in SERVER_TEXT
