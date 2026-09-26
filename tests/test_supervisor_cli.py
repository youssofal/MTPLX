"""Tests for `mtplx supervise`: CLI parsing, model resolution and the
SupervisorConfig handed to `mtplx.supervisor.service.run_supervisor`.

`mtplx/supervisor/service.py` may not exist yet (wave 2 sibling task), so
every test injects a fake module into `sys.modules` before calling
`cmd_supervise_public`. That also keeps these tests from ever booting a real
uvicorn server.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest

from mtplx.cli import build_parser
from mtplx.commands import public


def _install_fake_supervisor_service(monkeypatch, *, run_supervisor=None):
    """Insert a fake `mtplx.supervisor.service` module and return the record
    the fake `run_supervisor` appends its received config to."""

    calls: list[object] = []

    @dataclass
    class SupervisorConfig:
        host: str = "127.0.0.1"
        port: int = 8000
        model_paths: list[Path] = field(default_factory=list)
        preload: list[str] = field(default_factory=list)
        default_model_id: str | None = None
        api_key: str | None = None
        budget_bytes: int | None = None
        idle_ttl_s: float = 1800.0
        load_timeout_s: float = 600.0
        unresponsive_grace_s: float = 90.0
        drain_s: float = 15.0
        evict_to_fit: bool = False
        strict_model: bool = False
        insecure_lan: bool = False
        unload_default: bool = False
        engine_extra_args: list[str] = field(default_factory=list)
        launch_id: str | None = None

    def _run_supervisor(config):
        calls.append(config)
        if run_supervisor is not None:
            return run_supervisor(config)
        return 0

    fake_pkg = ModuleType("mtplx.supervisor")
    fake_service = ModuleType("mtplx.supervisor.service")
    fake_service.SupervisorConfig = SupervisorConfig
    fake_service.run_supervisor = _run_supervisor
    monkeypatch.setitem(sys.modules, "mtplx.supervisor", fake_pkg)
    monkeypatch.setitem(sys.modules, "mtplx.supervisor.service", fake_service)
    return calls


def _make_pack(root: Path, name: str) -> Path:
    """A minimal directory that `model_catalog.scan_installed_models` treats
    as a complete install: config.json + one non-shard weight file."""

    pack = root / name
    pack.mkdir(parents=True)
    (pack / "config.json").write_text("{}", encoding="utf-8")
    (pack / "mtplx_runtime.json").write_text("{}", encoding="utf-8")
    (pack / "model.safetensors").write_bytes(b"0" * 16)
    return pack


def _parse(args: list[str]):
    return build_parser().parse_args(["supervise", *args])


def test_supervise_resolves_models_by_path(monkeypatch, tmp_path):
    calls = _install_fake_supervisor_service(monkeypatch)
    pack = _make_pack(tmp_path, "local-model")
    args = _parse(["--models", str(pack)])
    assert public.cmd_supervise_public(args) == 0
    config = calls[0]
    assert config.model_paths == [pack.resolve()]
    assert config.default_model_id == "local-model"
    assert config.preload == ["local-model"]


def test_supervise_resolves_models_by_installed_name(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(tmp_path))
    pack_a = _make_pack(tmp_path, "Org--A")
    pack_b = _make_pack(tmp_path, "Org--B")
    calls = _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", "Org--A,Org--B"])
    assert public.cmd_supervise_public(args) == 0
    config = calls[0]
    assert config.model_paths == [pack_a, pack_b]
    assert config.default_model_id == "Org--A"


def test_supervise_models_all_picks_up_every_installed_pack(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(tmp_path))
    _make_pack(tmp_path, "Org--A")
    _make_pack(tmp_path, "Org--B")
    calls = _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", "all"])
    assert public.cmd_supervise_public(args) == 0
    config = calls[0]
    assert {p.name for p in config.model_paths} == {"Org--A", "Org--B"}


def test_supervise_missing_model_raises_systemexit_listing_installed(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(tmp_path))
    _make_pack(tmp_path, "Org--Installed")
    _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", "Org--Missing"])
    with pytest.raises(SystemExit) as excinfo:
        public.cmd_supervise_public(args)
    message = str(excinfo.value)
    assert "Org--Missing" in message
    assert "Org--Installed" in message


def test_supervise_refuses_non_localhost_bind_without_key(monkeypatch, tmp_path):
    pack = _make_pack(tmp_path, "local-model")
    _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", str(pack), "--host", "0.0.0.0"])
    assert public.cmd_supervise_public(args) == 2


def test_supervise_insecure_lan_requires_yes(monkeypatch, tmp_path):
    pack = _make_pack(tmp_path, "local-model")
    _install_fake_supervisor_service(monkeypatch)
    args = _parse(
        ["--models", str(pack), "--host", "0.0.0.0", "--insecure-lan"]
    )
    assert public.cmd_supervise_public(args) == 2


def test_supervise_insecure_lan_with_yes_starts(monkeypatch, tmp_path):
    pack = _make_pack(tmp_path, "local-model")
    calls = _install_fake_supervisor_service(monkeypatch)
    args = _parse(
        [
            "--models",
            str(pack),
            "--host",
            "0.0.0.0",
            "--insecure-lan",
            "--yes",
        ]
    )
    assert public.cmd_supervise_public(args) == 0
    assert calls[0].insecure_lan is True


def test_supervise_forwards_engine_flags_in_order(monkeypatch, tmp_path):
    pack = _make_pack(tmp_path, "local-model")
    calls = _install_fake_supervisor_service(monkeypatch)
    args = _parse(
        [
            "--models",
            str(pack),
            "--profile",
            "sustained",
            "--generation-mode",
            "ar",
            "--no-mtp",
            "--depth",
            "2",
            "--context-window",
            "4096",
            "--batching-preset",
            "agent",
            "--max-active-requests",
            "4",
            "--paged-kv-quantization",
            "q8",
            "--ssd-session-cache",
            "off",
            "--ssd-session-cache-dir",
            "/tmp/ssd",
            "--prefill-chunk-tokens",
            "128",
            "--embedding-model",
            "org/embed",
            "--reranker-model",
            "org/rank",
            "--fan-mode",
            "smart",
            "--enable-thermal-poll",
        ]
    )
    assert public.cmd_supervise_public(args) == 0
    assert calls[0].engine_extra_args == [
        "--profile",
        "sustained",
        "--generation-mode",
        "ar",
        "--no-mtp",
        "--depth",
        "2",
        "--context-window",
        "4096",
        "--batching-preset",
        "agent",
        "--max-active-requests",
        "4",
        "--paged-kv-quantization",
        "q8",
        "--ssd-session-cache",
        "off",
        "--ssd-session-cache-dir",
        "/tmp/ssd",
        "--prefill-chunk-tokens",
        "128",
        "--embedding-model",
        "org/embed",
        "--reranker-model",
        "org/rank",
        "--fan-mode",
        "smart",
        "--enable-thermal-poll",
    ]


def test_supervise_memory_budget_converts_gib_to_bytes(monkeypatch, tmp_path):
    pack = _make_pack(tmp_path, "local-model")
    calls = _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", str(pack), "--memory-budget", "40"])
    assert public.cmd_supervise_public(args) == 0
    assert calls[0].budget_bytes == 42949672960


def test_supervise_defaults_for_ttl_and_timeouts(monkeypatch, tmp_path):
    pack = _make_pack(tmp_path, "local-model")
    calls = _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", str(pack)])
    assert public.cmd_supervise_public(args) == 0
    config = calls[0]
    assert config.idle_ttl_s == 1800.0
    assert config.load_timeout_s == 600.0
    assert config.unresponsive_grace_s == 90.0
    assert config.drain_s == 15.0


def test_supervise_default_validated_against_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(tmp_path))
    _make_pack(tmp_path, "Org--A")
    _make_pack(tmp_path, "Org--B")
    _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", "Org--A", "--default", "Org--NotThere"])
    with pytest.raises(SystemExit) as excinfo:
        public.cmd_supervise_public(args)
    assert "Org--NotThere" in str(excinfo.value)


def test_supervise_preload_validated_as_subset_of_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(tmp_path))
    _make_pack(tmp_path, "Org--A")
    _make_pack(tmp_path, "Org--B")
    _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", "Org--A", "--preload", "Org--B"])
    with pytest.raises(SystemExit) as excinfo:
        public.cmd_supervise_public(args)
    assert "Org--B" in str(excinfo.value)


def test_supervise_preload_defaults_to_first_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(tmp_path))
    _make_pack(tmp_path, "Org--A")
    _make_pack(tmp_path, "Org--B")
    calls = _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", "Org--A,Org--B"])
    assert public.cmd_supervise_public(args) == 0
    assert calls[0].preload == ["Org--A"]


def test_supervise_explicit_preload_subset(monkeypatch, tmp_path):
    monkeypatch.setenv("MTPLX_MODEL_DIR", str(tmp_path))
    _make_pack(tmp_path, "Org--A")
    _make_pack(tmp_path, "Org--B")
    calls = _install_fake_supervisor_service(monkeypatch)
    args = _parse(["--models", "Org--A,Org--B", "--preload", "Org--B"])
    assert public.cmd_supervise_public(args) == 0
    assert calls[0].preload == ["Org--B"]
