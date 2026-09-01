"""Charlatan-defensibility guards for default-profile resolution (2.8).

Historic bug class: a surface resolves to sustained when the launch rule
says turbo, or displays turbo while actually running sustained. These tests
pin every fixed surface: the quickstart wizard's Auto default (no --profile
stamped anywhere), the one-shot legacy-state migration, tune's launch-rule
default, the quickstart download branch, and doctor's compiled-verify fence
line (issue #255). Suite-builder coverage lives with the original guard in
test_qwen38_family.py::test_qwen38_no_silent_sustained_side_doors.
"""

from __future__ import annotations

import argparse
import builtins
import json
from types import SimpleNamespace

from mtplx.profiles import QWEN38_BARE_SPEED_PUBLIC_MODEL_ID
from mtplx.ui import onboarding

FLAGSHIP = QWEN38_BARE_SPEED_PUBLIC_MODEL_ID
# A runtime-model path whose name components resolve to the flagship public
# id (the same first-party mapping test_qwen38_public_model_id_resolution
# pins for path refs).
FLAGSHIP_RUNTIME_DIR = "/tmp/mtplx-test/Youssofal--Qwen3.8-27B-MTPLX-Bare-Speed"


# ------------------------------------------------------------ wizard stamping


def _quickstart_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        target=None,
        model=None,
        profile="sustained",  # parser default
        max=False,
        prompt=None,
        dry_run=False,
        yes=False,
        fresh=False,
        download=False,
        cache_dir=None,
        unsafe_force_unverified=False,
        show_stats=True,
        host="127.0.0.1",
        port=8000,
        api_key=None,
        model_id=None,
        warmup_tokens=16,
        stream_interval=1,
        rate_limit=0,
        max_response_tokens=None,
        reasoning_parser="qwen3",
        strict_warmup=False,
        strict_fast_path=False,
        json=False,
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        seed=0,
        system=None,
        _cli_flags=set(),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _stub_quickstart_pipeline(monkeypatch, captured_profiles: list) -> None:
    """Stub model resolution / gating / launch so no MLX or network runs.

    ``_apply_model_contract_depth_default`` receives the profile object the
    launch resolved — capturing it observes exactly the value the engine
    will run under.
    """

    def fake_resolve_model(model, *, cache_dir, search_dirs=None, download):
        return FLAGSHIP_RUNTIME_DIR, {"model": model, "downloaded": False}

    monkeypatch.setattr(
        "mtplx.commands.public._quickstart_resolve_model", fake_resolve_model
    )

    def fake_gate(runtime_model, *, unsafe_force_unverified, yes):
        return (
            {"runtime_contract": {"verified": True}, "compatibility": "verified"},
            None,
        )

    monkeypatch.setattr("mtplx.commands.public._model_gate", fake_gate)

    def fake_depth_default(args, inspection, profile):
        captured_profiles.append(profile)

    monkeypatch.setattr(
        "mtplx.commands.public._apply_model_contract_depth_default",
        fake_depth_default,
    )
    monkeypatch.setattr(
        "mtplx.commands.public._apply_backend_serve_defaults",
        lambda args, inspection: None,
    )
    monkeypatch.setattr(
        "mtplx.commands.public._quickstart_apply_tuned_depth",
        lambda args, **kwargs: None,
    )
    monkeypatch.setattr(
        "mtplx.commands.public._quickstart_run_terminal_chat",
        lambda args, *, runtime_model, inspection: 0,
    )


def test_wizard_auto_choice_stamps_no_profile_and_resolves_turbo(
    tmp_path, monkeypatch
):
    """The 25.3 tok/s shape: wizard Auto must leave --profile unstamped so
    per-model resolution promotes the flagship to turbo."""

    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "auto.json"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    flow_calls: list[dict] = []

    def fake_flow(**kwargs):
        flow_calls.append(kwargs)
        return {
            "model": FLAGSHIP_RUNTIME_DIR,
            "profile": onboarding.PROFILE_AUTO,
            "max": False,
            "target": "terminal",
        }

    monkeypatch.setattr("mtplx.ui.onboarding.run_quickstart_flow", fake_flow)

    captured: list = []
    _stub_quickstart_pipeline(monkeypatch, captured)

    from mtplx.commands.public import cmd_quickstart_public

    args = _quickstart_args(
        cache_dir=str(tmp_path / "cache"),
        model_search_dirs=[str(tmp_path / "archive"), str(tmp_path / "shared")],
    )
    assert cmd_quickstart_public(args) == 0
    assert flow_calls[0]["cache_dir"] == str(tmp_path / "cache")
    assert flow_calls[0]["search_dirs"] == [
        str(tmp_path / "archive"),
        str(tmp_path / "shared"),
    ]
    assert "profile" not in args._cli_flags
    assert args.profile == "sustained"  # parser default untouched
    assert captured, "launch never resolved a profile"
    assert captured[-1].name == "turbo"


def test_wizard_explicit_sustained_choice_stays_pinned(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "pinned.json"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    def fake_flow(**kwargs):
        return {
            "model": FLAGSHIP_RUNTIME_DIR,
            "profile": "sustained",
            "profile_explicit": True,
            "max": False,
            "target": "terminal",
        }

    monkeypatch.setattr("mtplx.ui.onboarding.run_quickstart_flow", fake_flow)

    captured: list = []
    _stub_quickstart_pipeline(monkeypatch, captured)

    from mtplx.commands.public import cmd_quickstart_public

    args = _quickstart_args()
    assert cmd_quickstart_public(args) == 0
    assert "profile" in args._cli_flags
    assert args.profile == "sustained"
    assert captured and captured[-1].name == "sustained"


# ------------------------------------------------------- one-shot migration


def test_legacy_sustained_state_migrates_to_auto_exactly_once(
    tmp_path, monkeypatch
):
    state_file = tmp_path / "legacy.json"
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(state_file))
    onboarding.save_state(
        {
            "model": "mtplx/foo",
            "profile": "sustained",
            "max": False,
            "target": "openwebui",
        }
    )

    saves: list[dict] = []
    real_save = onboarding.save_state

    def counting_save(state):
        saves.append(dict(state))
        real_save(state)

    monkeypatch.setattr(onboarding, "save_state", counting_save)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "")

    first = onboarding.run_quickstart_flow(fresh=False)
    assert first is not None
    assert first["profile"] == onboarding.PROFILE_AUTO
    migration_saves = [s for s in saves if s.get("profile") == onboarding.PROFILE_AUTO]
    assert len(migration_saves) == 1
    persisted = json.loads(state_file.read_text(encoding="utf-8"))
    assert persisted["profile"] == onboarding.PROFILE_AUTO

    # Second run: the sentinel is already persisted — no further rewrite.
    saves.clear()
    second = onboarding.run_quickstart_flow(fresh=False)
    assert second is not None
    assert second["profile"] == onboarding.PROFILE_AUTO
    assert saves == []


def test_legacy_sustained_max_state_is_not_migrated(tmp_path, monkeypatch):
    """Sustained Max was a deliberate non-default keystroke: it stays pinned."""

    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "susmax.json"))
    onboarding.save_state(
        {
            "model": "mtplx/foo",
            "profile": "sustained",
            "max": True,
            "target": "openwebui",
        }
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "")
    monkeypatch.setattr(
        "mtplx.thermal.detect_thermal_control",
        lambda: {"available": True, "selected": {"kind": "thermalforge"}},
    )

    state = onboarding.run_quickstart_flow(fresh=False)
    assert state is not None
    assert state["profile"] == "sustained"
    assert state["max"] is True


def test_post_ship_explicit_sustained_state_is_not_migrated():
    state = {
        "model": "mtplx/foo",
        "profile": "sustained",
        "profile_explicit": True,
        "max": False,
        "target": "cli",
    }
    migrated, changed = onboarding._migrate_legacy_default_profile(state)
    assert changed is False
    assert migrated is state


def test_auto_state_is_reusable():
    assert onboarding._quickstart_state_is_reusable(
        {
            "model": "mtplx/foo",
            "profile": onboarding.PROFILE_AUTO,
            "max": False,
            "target": "cli",
        }
    )


# ------------------------------------------------------------------ tune F24


def test_tune_settings_default_follows_launch_rule():
    from mtplx.commands.public import _tune_settings

    args = SimpleNamespace(profile=None, _cli_flags=set())
    settings = _tune_settings(args, model=FLAGSHIP, depths=[1, 2, 3])
    assert settings["profile"] == "turbo"

    other = SimpleNamespace(profile=None, _cli_flags=set())
    assert (
        _tune_settings(other, model="someone/custom", depths=[1, 2, 3])["profile"]
        == "sustained"
    )

    pinned = SimpleNamespace(profile="performance-cold", _cli_flags={"profile"})
    assert (
        _tune_settings(pinned, model=FLAGSHIP, depths=[1, 2, 3])["profile"]
        == "performance-cold"
    )


def test_tune_parser_has_no_hidden_performance_cold_default():
    from mtplx.cli import build_parser

    args = build_parser().parse_args(["tune"])
    assert args.profile is None


# ------------------------------------------------- quickstart download branch


def test_quickstart_download_branch_resolves_launch_rule_profile(
    tmp_path, monkeypatch
):
    """The missing-model download branch used the raw DEFAULT_PROFILE_NAME
    fallback while its sibling already resolved per model (F27)."""

    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "dl.json"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    captured: list = []
    _stub_quickstart_pipeline(monkeypatch, captured)

    # First resolution: model not local -> the interactive "Download now?"
    # prompt fires; the second resolution (download=True) succeeds.
    calls: list[bool] = []

    def fake_resolve_model(model, *, cache_dir, search_dirs=None, download):
        calls.append(download)
        if len(calls) == 1:
            return None, {}
        return FLAGSHIP_RUNTIME_DIR, {
            "model": model,
            "downloaded": True,
            "download_ref": model,
        }

    monkeypatch.setattr(
        "mtplx.commands.public._quickstart_resolve_model", fake_resolve_model
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "y")

    from mtplx.commands.public import cmd_quickstart_public

    # Explicit model skips onboarding; the model is "missing" locally.
    args = _quickstart_args(
        model=FLAGSHIP_RUNTIME_DIR,
        target="cli",
        _cli_flags={"model"},
    )
    assert cmd_quickstart_public(args) == 0
    assert calls == [False, True], "the download prompt branch did not run"
    assert captured, "download branch never resolved a profile"
    assert captured[-1].name == "turbo"


# ---------------------------------------------------------------- doctor F15


def test_doctor_reports_compiled_verify_fence_from_profile(monkeypatch):
    from mtplx.commands import public

    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", raising=False)
    monkeypatch.setattr(
        public,
        "select_default_model",
        lambda: SimpleNamespace(model=FLAGSHIP),
    )

    fence = public._compiled_verify_fence_report(SimpleNamespace(_cli_flags=set()))
    assert fence["resolved_default_profile"] == "turbo"
    assert fence["mode"] == "on"
    assert fence["mode_source"] == "turbo profile"
    assert fence["max_context_tokens"] == 32768
    assert fence["max_context_source"] == "turbo profile"
    assert fence["fenced"] is True


def test_doctor_fence_operator_env_beats_profile(monkeypatch):
    from mtplx.commands import public

    monkeypatch.setenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", "12288")
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    monkeypatch.setattr(
        public,
        "select_default_model",
        lambda: SimpleNamespace(model=FLAGSHIP),
    )

    fence = public._compiled_verify_fence_report(SimpleNamespace(_cli_flags=set()))
    assert fence["max_context_tokens"] == 12288
    assert fence["max_context_source"] == "MTPLX_COMPILED_VERIFY_MAX_CONTEXT env"


def test_doctor_fence_engine_default_without_turbo(monkeypatch):
    from mtplx.commands import public

    monkeypatch.delenv("MTPLX_COMPILED_VERIFY", raising=False)
    monkeypatch.delenv("MTPLX_COMPILED_VERIFY_MAX_CONTEXT", raising=False)
    monkeypatch.setattr(
        public,
        "select_default_model",
        lambda: SimpleNamespace(model="someone/custom"),
    )

    fence = public._compiled_verify_fence_report(SimpleNamespace(_cli_flags=set()))
    assert fence["resolved_default_profile"] == "sustained"
    assert fence["mode"] == "off"
    assert fence["max_context_tokens"] == 6144
    assert fence["max_context_source"] == "engine default"


def test_doctor_human_render_prints_fence_line(capsys):
    from mtplx.commands.public import _render_doctor_report

    report = {
        "environment": {},
        "huggingface": {},
        "thermal_control": {},
        "tools": {},
        "compiled_verify": {
            "mode": "on",
            "mode_source": "turbo profile",
            "max_context_tokens": 32768,
            "max_context_source": "turbo profile",
            "fenced": True,
        },
    }
    args = SimpleNamespace(summary=False, deep=False)
    assert _render_doctor_report(args, report) == 0
    out = capsys.readouterr().out
    assert "compiled verify: on (turbo profile)" in out
    assert "compiled verify fence: <= 32768 tokens (turbo profile)" in out
    assert "falls back to eager" in out
