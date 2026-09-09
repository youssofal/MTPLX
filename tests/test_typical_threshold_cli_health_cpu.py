"""--typical-threshold CLI flag + the /health typical_acceptance entry. CPU-only.

mtplx/server/openai.py imports MLX, so the /health payload helper is compiled out
of the shipped source (the same trick test_ngram_prewarm_option.py uses) instead
of importing the module, and the CLI wiring is asserted against the source text.
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
    for key in ("MTPLX_FABLE_TYPICAL_THRESHOLD", "MTPLX_FABLE_TYPICAL_EPS"):
        monkeypatch.delenv(key, raising=False)


def _payload():
    return _compile_function(SERVER_TEXT, "_typical_acceptance_health_payload")()


def test_health_reports_off_by_default():
    p = _payload()
    assert p["enabled"] is False
    assert p["threshold"] == 0.0
    assert p["eps"] == 1.0
    assert p["distribution_exact"] is True
    assert p["floor"] == "min(eps, delta*exp(-H))"


def test_health_reports_on_with_a_positive_threshold(monkeypatch):
    monkeypatch.setenv("MTPLX_FABLE_TYPICAL_THRESHOLD", "0.09")
    p = _payload()
    assert p["enabled"] is True
    assert p["threshold"] == 0.09
    assert p["delta"] == 0.09
    assert p["distribution_exact"] is False


def test_health_zero_threshold_is_off(monkeypatch):
    monkeypatch.setenv("MTPLX_FABLE_TYPICAL_THRESHOLD", "0")
    p = _payload()
    assert p["enabled"] is False
    assert p["distribution_exact"] is True


def test_health_bad_threshold_is_ignored_not_fatal(monkeypatch):
    monkeypatch.setenv("MTPLX_FABLE_TYPICAL_THRESHOLD", "sometimes")
    p = _payload()  # must not raise
    assert p["enabled"] is False
    assert p["threshold"] == 0.0


def test_health_entry_is_wired_into_the_route():
    assert '"typical_acceptance": _typical_acceptance_health_payload(),' in SERVER_TEXT


def test_cli_flag_is_declared_and_maps_to_the_env():
    assert '"--typical-threshold",' in SERVER_TEXT
    assert (
        'os.environ["MTPLX_FABLE_TYPICAL_THRESHOLD"] = str(float(args.typical_threshold))'
        in SERVER_TEXT
    )
    # default=None so the flag cannot silently overrule a shell-set env value.
    decl = SERVER_TEXT.split('"--typical-threshold",', 1)[1].split("    )", 1)[0]
    assert "default=None" in decl
