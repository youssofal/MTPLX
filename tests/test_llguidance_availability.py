"""Structured output reports why llguidance cannot load, not just "install it" (#643).

The reporter had llguidance 1.3.0, which predates ``get_version()``. The module
import failed inside one broad ``except``, and every ``json_schema`` request was
refused with "pip install llguidance", which pip answers with "Requirement
already satisfied". These tests build fake llguidance installs on ``sys.path``
in a child interpreter, so the import that runs is the module-level one the
server runs at startup.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_REFUSAL_PROBE = textwrap.dedent(
    """
    import json, sys
    from mtplx.constrained import (
        ResponseFormatError,
        constraint_spec_from_response_format,
        tool_call_constraint_spec,
    )
    out = {}
    try:
        constraint_spec_from_response_format(
            {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}
        )
    except ResponseFormatError as exc:
        out["json_schema"] = str(exc)
    try:
        tool_call_constraint_spec(
            [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "auto",
            object(),
        )
    except ResponseFormatError as exc:
        out["strict_tools"] = str(exc)
    print(json.dumps(out))
    """
)


def _fake_llguidance(root: Path, version: str, init: str, mlx: str = "") -> Path:
    package = root / "llguidance"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(init, encoding="utf-8")
    (package / "hf.py").write_text("", encoding="utf-8")
    (package / "mlx.py").write_text(mlx, encoding="utf-8")
    dist = root / f"llguidance-{version}.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: llguidance\nVersion: {version}\n",
        encoding="utf-8",
    )
    return root


def _refusals(path_entry: Path | None, prelude: str = "") -> dict[str, str]:
    env = dict(os.environ)
    entries = [str(REPO_ROOT)]
    if path_entry is not None:
        entries.insert(0, str(path_entry))
    env["PYTHONPATH"] = os.pathsep.join(entries)
    proc = subprocess.run(
        [sys.executable, "-c", prelude + _REFUSAL_PROBE],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_reporters_llguidance_1_3_0_is_named_with_the_upgrade_command(tmp_path):
    # llguidance 1.3.0's package exports no get_version (it arrived in 1.7.4).
    root = _fake_llguidance(tmp_path / "site", "1.3.0", "def other():\n    pass\n")
    refusals = _refusals(root)
    message = refusals["json_schema"]
    assert "llguidance 1.3.0 is installed but failed to load" in message
    assert "AttributeError" in message and "get_version" in message
    assert "MTPLX needs llguidance 1.8 or later" in message
    assert "-m pip install -U 'llguidance>=1.8'" in message
    assert "not installed" not in message
    assert message.endswith("refusing to silently return unconstrained output")
    assert "llguidance 1.3.0 is installed but failed to load" in refusals["strict_tools"]


def test_a_submodule_that_fails_to_load_is_reported_as_itself(tmp_path):
    root = _fake_llguidance(
        tmp_path / "site",
        "1.9.1",
        "def get_version():\n    return 'llguidance@1.9.1'\n",
        mlx="raise ImportError(\"cannot import name 'custom_function' from 'mlx.core'\")\n",
    )
    message = _refusals(root)["json_schema"]
    assert "llguidance 1.9.1 is installed but failed to load" in message
    assert "ImportError: cannot import name 'custom_function' from 'mlx.core'" in message
    # 1.9.1 meets the floor, so the message must not blame the version.
    assert "MTPLX needs llguidance" not in message


def test_an_absent_llguidance_is_reported_as_not_installed():
    hide = textwrap.dedent(
        """
        import importlib.metadata as _md, sys as _sys
        _sys.modules["llguidance"] = None
        _real_version = _md.version
        def _version(name):
            if name == "llguidance":
                raise _md.PackageNotFoundError(name)
            return _real_version(name)
        _md.version = _version
        """
    )
    message = _refusals(None, prelude=hide)["json_schema"]
    assert ": llguidance is not installed." in message
    assert "-m pip install -U 'llguidance>=1.8'" in message


def test_doctor_reports_the_reason_and_the_command(monkeypatch):
    from mtplx import constrained
    from mtplx.diagnostics import structured_output_check

    reason = (
        "llguidance 1.3.0 is installed but failed to load (AttributeError: "
        "module 'llguidance' has no attribute 'get_version'); MTPLX needs "
        "llguidance 1.8 or later"
    )
    monkeypatch.setattr(constrained, "LLGUIDANCE_AVAILABLE", False)
    monkeypatch.setattr(constrained, "LLGUIDANCE_VERSION", None)
    monkeypatch.setattr(constrained, "LLGUIDANCE_UNAVAILABLE_REASON", reason)
    monkeypatch.setattr(constrained, "_installed_llguidance_version", lambda: "1.3.0")
    check = structured_output_check().to_dict()
    assert check["id"] == "python.dep.llguidance"
    assert check["status"] == "warn"
    assert check["observed"]["installed_version"] == "1.3.0"
    assert check["observed"]["meets_requirement"] is False
    assert reason in check["fix"]
    assert check["command"] == constrained.llguidance_install_command()
    assert sys.executable in check["command"]


def test_doctor_text_report_prints_the_structured_output_line(monkeypatch, capsys):
    from mtplx import constrained
    from mtplx.commands.public import _render_doctor_report
    from mtplx.diagnostics import structured_output_check

    monkeypatch.setattr(constrained, "LLGUIDANCE_AVAILABLE", False)
    monkeypatch.setattr(constrained, "LLGUIDANCE_UNAVAILABLE_REASON", "llguidance is not installed")
    monkeypatch.setattr(constrained, "_installed_llguidance_version", lambda: None)
    report = {"diagnostics": {"checks": [structured_output_check().to_dict()]}}
    _render_doctor_report(SimpleNamespace(summary=False, deep=False), report)
    out = capsys.readouterr().out
    assert "structured output: Requests with response_format json_schema or json_object are refused" in out
    assert f"  fix: {constrained.llguidance_install_command()}" in out


def test_doctor_passes_on_the_installed_llguidance():
    pytest.importorskip("llguidance")
    from mtplx import constrained
    from mtplx.diagnostics import structured_output_check

    assert constrained.LLGUIDANCE_AVAILABLE, constrained.LLGUIDANCE_UNAVAILABLE_REASON
    check = structured_output_check().to_dict()
    assert check["status"] == "pass"
    assert check["observed"]["reason"] is None
    assert check["command"] is None
