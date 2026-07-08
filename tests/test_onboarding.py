"""Smoke tests for the quickstart onboarding flow.

Verifies state persistence, fallback labels, and the screen helpers don't
raise. The interactive ``run_onboarding_screens`` is exercised with a stubbed
``input`` so the test runs non-interactively.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

from mtplx.profiles import DEFAULT_FP16_HF_MODEL_ID
from mtplx.ui import onboarding


def test_state_load_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "missing.json"))
    assert onboarding.load_state() is None


def test_state_round_trip(tmp_path, monkeypatch):
    state_file = tmp_path / "quickstart.json"
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(state_file))

    onboarding.save_state(
        {
            "model": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "profile": "performance-cold",
            "max": True,
            "target": "openwebui",
        }
    )
    loaded = onboarding.load_state()
    assert loaded is not None
    assert loaded["model"] == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    assert loaded["profile"] == "performance-cold"
    assert loaded["max"] is True
    assert loaded["target"] == "openwebui"
    assert "saved_at" in loaded
    # File on disk is valid JSON.
    with state_file.open("r", encoding="utf-8") as handle:
        json.load(handle)


def test_mode_label_covers_all_modes():
    """Mode labels explain runtime mechanics and hardware-neutral speed gain."""
    stable = onboarding.mode_label({"profile": "stable", "max": False})
    legacy = onboarding.mode_label({"profile": "performance-cold", "max": False})
    sustained = onboarding.mode_label({"profile": "sustained", "max": False})
    sustained_max = onboarding.mode_label({"profile": "sustained", "max": True})
    burst = onboarding.mode_label({"profile": "performance-cold", "max": True})
    assert "Stable" in stable and "exact/staged" in stable and "tok/s" not in stable
    assert "legacy" in legacy and "fan boost" in legacy
    assert "Sustained" in sustained and "long-context" in sustained
    assert "Sustained Max" in sustained_max and "100%" in sustained_max
    assert "Burst" in burst and "max 8K" in burst


def test_interface_label_covers_all_targets():
    assert "Web UI" in onboarding.interface_label("openwebui")
    assert "Web UI" in onboarding.interface_label("web")
    assert "API server" in onboarding.interface_label("server")
    assert "CLI" in onboarding.interface_label("cli")
    assert "CLI" in onboarding.interface_label("terminal")
    assert "Pi" in onboarding.interface_label("pi")
    assert "OpenCode" in onboarding.interface_label("opencode")


def test_run_onboarding_screens_with_stubbed_input(monkeypatch, capsys):
    """Walk all four screens with stubbed ``input`` answers.

    Screens: model, mode, interface, dashboard-companion (only asked for
    server-spawning targets; openwebui is one of them).
    """
    answers = iter(["1", "1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    expected_model = onboarding._verified_default_model()

    state = onboarding.run_onboarding_screens()
    assert state["model"] == expected_model
    assert state["profile"] == "sustained"
    assert state["max"] is False
    assert state["target"] == "openwebui"
    assert state["open_dashboard"] is False


def test_screen_interface_uses_requested_port(monkeypatch):
    captured: dict[str, object] = {}

    def fake_step_panel(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(onboarding, "_step_panel", fake_step_panel)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "1")

    target = onboarding.screen_interface(host="127.0.0.1", port=8765)

    assert target == "openwebui"
    options = captured["options"]
    rendered = " ".join(str(item) for option in options for item in option)
    assert "http://127.0.0.1:8765/" in rendered
    assert "http://127.0.0.1:8000/" not in rendered


def test_screen_dashboard_companion_uses_requested_port(monkeypatch):
    captured: dict[str, object] = {}

    def fake_step_panel(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(onboarding, "_step_panel", fake_step_panel)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "1")

    assert (
        onboarding.screen_dashboard_companion(
            "openwebui",
            host="127.0.0.1",
            port=8765,
        )
        is True
    )
    options = captured["options"]
    rendered = " ".join(str(item) for option in options for item in option)
    assert "http://127.0.0.1:8765/dashboard" in rendered
    assert "http://127.0.0.1:8000/dashboard" not in rendered


def test_run_onboarding_screens_uses_fp16_default_when_policy_selects_it(monkeypatch):
    monkeypatch.setenv("MTPLX_DEFAULT_MODEL_VARIANT", "fp16")
    # 4 answers: model + mode + interface (openwebui) + dashboard companion.
    answers = iter(["1", "1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_onboarding_screens()

    assert state["model"] == DEFAULT_FP16_HF_MODEL_ID
    assert state["model_selection"]["variant"] == "fp16"
    assert state["model_selection"]["precision"] == "FP16"


def test_run_onboarding_sustained_max_sets_max_flag_when_thermal_available(monkeypatch):
    """Picking Sustained Max + a working fan controller -> ``profile=sustained,max=True``."""
    answers = iter(["1", "2", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    monkeypatch.setattr(onboarding, "ensure_thermal_control_installed", lambda: True)
    state = onboarding.run_onboarding_screens()
    assert state["profile"] == "sustained"
    assert state["max"] is True
    assert state["target"] == "terminal"


def test_run_onboarding_fan_mode_falls_back_to_sustained_when_thermal_unavailable(monkeypatch):
    """Picking a fan-backed mode + declined/failed install -> Sustained no-fan."""
    answers = iter(["1", "3", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    monkeypatch.setattr(onboarding, "ensure_thermal_control_installed", lambda: False)
    state = onboarding.run_onboarding_screens()
    assert state["profile"] == "sustained"
    assert state["max"] is False
    assert state["target"] == "terminal"


def test_run_onboarding_sustained_mode_is_explicit(monkeypatch):
    # openwebui target → dashboard companion prompt fires; answer "No" (2).
    answers = iter(["1", "1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_onboarding_screens()

    assert state["profile"] == "sustained"
    assert state["max"] is False
    assert state["target"] == "openwebui"


def test_run_onboarding_can_select_pi(monkeypatch):
    answers = iter(["1", "1", "3", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_onboarding_screens()

    assert state["profile"] == "sustained"
    assert state["max"] is False
    assert state["target"] == "pi"
    assert state["open_dashboard"] is False


def test_run_onboarding_can_select_opencode(monkeypatch):
    answers = iter(["1", "1", "4", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_onboarding_screens()

    assert state["profile"] == "sustained"
    assert state["max"] is False
    assert state["target"] == "opencode"
    assert state["open_dashboard"] is False


def test_run_onboarding_opencode_can_open_dashboard_companion(monkeypatch):
    """User picks OpenCode AND yes to the dashboard companion."""
    answers = iter(["1", "1", "4", "1"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_onboarding_screens()

    assert state["target"] == "opencode"
    assert state["open_dashboard"] is True


def test_run_onboarding_terminal_skips_dashboard_companion(monkeypatch):
    """Terminal/CLI target has no server, so no dashboard companion prompt."""
    # Only 3 answers: model, mode, interface (terminal). No 4th input
    # consumed since terminal target skips the companion question.
    answers = iter(["1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_onboarding_screens()

    assert state["target"] == "terminal"
    assert state["open_dashboard"] is False


def test_run_onboarding_dashboard_target_skips_companion(monkeypatch):
    """Picking 'Live Dashboard' as the target skips the companion prompt."""
    answers = iter(["1", "1", "6"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_onboarding_screens()

    assert state["target"] == "dashboard"
    # The companion question is only for non-dashboard server targets.
    assert state["open_dashboard"] is False


def test_run_serve_onboarding_screens_defaults_to_api_server(monkeypatch):
    answers = iter(["1", "1", "1"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    state = onboarding.run_serve_onboarding_screens(host="127.0.0.1", port=8765)
    assert state["model"] == onboarding._verified_default_model()
    assert state["profile"] == "sustained"
    assert state["max"] is False
    assert state["target"] == "server"
    assert state["open_browser"] is False


def test_run_serve_onboarding_screens_can_open_browser(monkeypatch):
    # 4 answers: model + mode + server-surface (open chat=2) + dashboard
    # companion (No=2). When server-surface is "Open browser chat too",
    # the dashboard companion is asked too.
    answers = iter(["1", "1", "2", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    state = onboarding.run_serve_onboarding_screens(host="127.0.0.1", port=8765)
    assert state["target"] == "openwebui"
    assert state["open_browser"] is True
    assert state["open_dashboard"] is False


def test_run_serve_flow_does_not_reuse_quickstart_state(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "serve.json"))
    onboarding.save_state(
        {
            "model": "mtplx/old",
            "profile": "performance-cold",
            "max": False,
            "target": "openwebui",
        }
    )
    answers = iter(["1", "1", "1"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    state = onboarding.run_serve_flow()
    assert state is not None
    assert state["model"] == onboarding._verified_default_model()
    assert state["target"] == "server"


def test_ensure_thermal_control_installed_returns_true_when_already_present(monkeypatch):
    monkeypatch.setattr(
        "mtplx.thermal.detect_thermal_control",
        lambda: {"available": True, "selected": {"kind": "thermalforge"}},
    )
    assert onboarding.ensure_thermal_control_installed() is True


def test_ensure_thermal_control_installed_user_picks_skip(monkeypatch):
    """Choosing option 2 (skip) returns False; consistent with the rest of the
    onboarding which is always numbered, never Y/N."""
    monkeypatch.setattr(
        "mtplx.thermal.detect_thermal_control",
        lambda: {"available": False, "instructions": "..."},
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "2")
    assert onboarding.ensure_thermal_control_installed() is False


def test_ensure_thermal_control_installed_user_picks_install_success(monkeypatch):
    monkeypatch.setattr(
        "mtplx.thermal.detect_thermal_control",
        lambda: {"available": False, "instructions": "..."},
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "1")

    def fake_install(*, install_daemon: bool = True, streaming: bool = True):
        return {
            "ok": True,
            "daemon_ok": True,
            "method": "source",
            "message": "ThermalForge installed and ready.",
            "steps": [],
            "detection": {"available": True},
        }

    monkeypatch.setattr("mtplx.thermal.install_thermal_control", fake_install)
    assert onboarding.ensure_thermal_control_installed() is True


def test_ensure_thermal_control_installed_install_fails(monkeypatch):
    monkeypatch.setattr(
        "mtplx.thermal.detect_thermal_control",
        lambda: {"available": False, "instructions": "..."},
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "1")

    def fake_install(*, install_daemon: bool = True, streaming: bool = True):
        return {
            "ok": False,
            "method": None,
            "message": "all install paths failed",
            "steps": [],
            "detection": {"available": False},
        }

    monkeypatch.setattr("mtplx.thermal.install_thermal_control", fake_install)
    assert onboarding.ensure_thermal_control_installed() is False


def test_ensure_thermal_control_install_uses_numbered_prompt_not_yn(monkeypatch, capsys):
    """Regression for the user complaint that the install offer asked Y/N
    while the rest of the onboarding asked for numbered choices."""
    monkeypatch.setattr(
        "mtplx.thermal.detect_thermal_control",
        lambda: {"available": False, "instructions": "..."},
    )
    captured: list[str] = []

    def fake_input(prompt: str = "") -> str:
        captured.append(prompt)
        return "2"  # skip

    monkeypatch.setattr(builtins, "input", fake_input)
    onboarding.ensure_thermal_control_installed()
    assert captured, "install prompt was never displayed"
    rendered = " ".join(captured).lower()
    assert "type" in rendered
    assert "press enter" in rendered
    assert "y/n" not in rendered
    assert "[y" not in rendered


def test_prompt_choice_uses_explicit_type_and_press_enter_phrasing(monkeypatch, capsys):
    """Regression for the user complaint that the old ``Select [1]:`` prompt
    didn't make it obvious you had to type a number and press Enter."""
    captured: list[str] = []

    def fake_input(prompt: str = "") -> str:
        captured.append(prompt)
        return "1"

    monkeypatch.setattr(builtins, "input", fake_input)
    onboarding._prompt_choice("ignored", ["1", "2", "3"], default="1")
    assert captured, "input prompt was never displayed"
    rendered = captured[0]
    assert "Type" in rendered
    assert "press Enter" in rendered
    assert "1-3" in rendered or "1, 2, 3" in rendered


def test_run_quickstart_flow_saves_state_on_first_run(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "first.json"))
    answers = iter(["1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    state = onboarding.run_quickstart_flow(fresh=False)
    assert state is not None
    assert state["target"] == "terminal"
    # Persisted to disk.
    persisted = onboarding.load_state()
    assert persisted is not None
    assert persisted["target"] == "terminal"


def test_run_quickstart_flow_returning_user_says_same(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "returning.json"))
    onboarding.save_state(
            {
                "model": "mtplx/foo",
                "profile": "sustained",
                "max": False,
                "target": "pi",
            }
    )
    # Returning-user prompt: empty answer (default Y) should reuse last state.
    answers = iter([""])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    state = onboarding.run_quickstart_flow(fresh=False)
    assert state is not None
    assert state["model"] == "mtplx/foo"
    assert state["target"] == "pi"


def test_run_quickstart_flow_returning_user_reuses_sustained(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "returning-sustained.json"))
    onboarding.save_state(
        {
            "model": "mtplx/foo",
            "profile": "sustained",
            "max": False,
            "target": "openwebui",
        }
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "")

    state = onboarding.run_quickstart_flow(fresh=False)

    assert state is not None
    assert state["profile"] == "sustained"
    assert state["model"] == "mtplx/foo"


def test_run_quickstart_flow_refreshes_saved_verified_default(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "refresh-default.json"))
    monkeypatch.setenv("MTPLX_DEFAULT_MODEL_VARIANT", "fp16")
    onboarding.save_state(
        {
            "model": onboarding.DEFAULT_HF_MODEL,
            "profile": "sustained",
            "max": False,
            "target": "terminal",
        }
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "")

    state = onboarding.run_quickstart_flow(fresh=False)

    assert state is not None
    assert state["model"] == DEFAULT_FP16_HF_MODEL_ID
    assert state["model_selection"]["variant"] == "fp16"


def test_run_quickstart_flow_legacy_stable_state_is_not_reused(tmp_path, monkeypatch):
    """Stable is still an explicit flag, but no longer a reusable Start mode."""

    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "legacy-stable.json"))
    onboarding.save_state(
        {
            "model": "mtplx/old-stable",
            "profile": "stable",
            "max": False,
            "target": "terminal",
        }
    )
    # 4 answers: model + mode + interface (openwebui) + dashboard companion (No).
    answers = iter(["1", "1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    state = onboarding.run_quickstart_flow(fresh=False)
    assert state is not None
    assert state["model"] == onboarding._verified_default_model()
    assert state["profile"] == "sustained"
    assert state["max"] is False
    assert state["target"] == "openwebui"


def test_run_quickstart_flow_returning_user_with_stale_max_drops_max(
    tmp_path, monkeypatch
):
    """If saved state says max=True but ThermalForge has gone away (or was
    never actually installed), re-offer the install instead of silently
    boosting fans that can't be controlled."""
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "stale-max.json"))
    onboarding.save_state(
        {
            "model": "mtplx/foo",
            "profile": "performance-cold",
            "max": True,
            "target": "openwebui",
        }
    )
    # User accepts "same as last time" then declines the install offer.
    answers = iter(["", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    monkeypatch.setattr(
        "mtplx.thermal.detect_thermal_control",
        lambda: {"available": False, "instructions": "..."},
    )
    state = onboarding.run_quickstart_flow(fresh=False)
    assert state is not None
    assert state["profile"] == "sustained"
    assert state["max"] is False  # downgraded; honesty preserved
    # Persisted file is rewritten so the next run reflects reality.
    persisted = onboarding.load_state()
    assert persisted is not None and persisted["profile"] == "sustained"
    assert persisted["max"] is False


def test_run_quickstart_flow_returning_user_says_no(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "redo.json"))
    onboarding.save_state(
        {
            "model": "mtplx/old",
            "profile": "performance-cold",
            "max": False,
            "target": "terminal",
        }
    )
    # 'n' → walk onboarding again, then 4 screens (model + mode + openwebui + dashboard).
    answers = iter(["n", "1", "1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    state = onboarding.run_quickstart_flow(fresh=False)
    assert state is not None
    assert state["model"] == onboarding._verified_default_model()
    assert state["profile"] == "sustained"
    assert state["target"] == "openwebui"


def test_screen_model_surfaces_configured_path_first(tmp_path, monkeypatch):
    """A user with a configured local path should see it as option 1 so the
    'verified default' choice doesn't trigger a needless re-download."""
    configured = "/Users/test/Documents/MTPLX/models/Qwen3.6-27B-MTPLX"
    answers = iter(["1"])  # accept option 1 = configured path
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    chosen = onboarding.screen_model(configured=configured)
    assert chosen == configured


def test_screen_model_picks_verified_default_when_configured_offered(monkeypatch):
    """With a configured path shown as option 1, option 2 still maps to the
    verified default, preferring an installed local artifact when present."""
    configured = "/Users/test/Documents/MTPLX/models/Qwen3.6-27B-MTPLX"
    answers = iter(["2"])  # explicit "verified default"
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    expected_model = onboarding._verified_default_model()

    chosen = onboarding.screen_model(configured=configured)
    assert chosen == expected_model


def test_screen_model_picks_hardware_default_when_configured_offered(monkeypatch):
    monkeypatch.setenv("MTPLX_DEFAULT_MODEL_VARIANT", "fp16")
    configured = "/Users/test/Documents/MTPLX/models/Qwen3.6-27B-MTPLX"
    answers = iter(["2"])  # explicit "verified default"
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    chosen = onboarding.screen_model(configured=configured)

    assert chosen == DEFAULT_FP16_HF_MODEL_ID


def test_screen_model_no_configured_uses_default_first(monkeypatch):
    """Without a configured path, option 1 maps to the hardware default."""
    answers = iter(["1"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    expected_model = onboarding._verified_default_model()

    chosen = onboarding.screen_model(configured=None)
    assert chosen == expected_model


def test_screen_model_optimized_quality_prefers_local_model(tmp_path, monkeypatch, capsys):
    from mtplx import default_models

    local_quality = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Quality"
    local_quality.mkdir()
    (local_quality / "config.json").write_text("{}", encoding="utf-8")
    (local_quality / "mtp.safetensors").write_bytes(b"mtp")
    (local_quality / "model-00001-of-00001.safetensors").write_bytes(b"model")
    monkeypatch.setenv(default_models.QUALITY_MODEL_ENV, str(local_quality))

    answers = iter(["2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    chosen = onboarding.screen_model(configured=None)

    captured = capsys.readouterr().out
    assert chosen == str(local_quality)
    assert "Optimized Quality" in captured
    assert str(local_quality) not in captured


def test_custom_hf_repo_rejects_pasted_terminal_output(monkeypatch, capsys):
    answers = iter(
        [
            "3",
            "Last login: Mon May  4 00:55:41 on ttys000",
            "trevon/Qwen3.5-27B-MLX-MTP",
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    chosen = onboarding.screen_model(configured=None)

    captured = capsys.readouterr().out
    assert chosen == "trevon/Qwen3.5-27B-MLX-MTP"
    assert "not a Hugging Face repo id" in captured


def test_custom_hf_repo_blank_after_invalid_does_not_accept_default(monkeypatch, capsys):
    answers = iter(
        [
            "3",
            "Last login: Mon May  4 00:55:41 on ttys000",
            "",
            "trevon/Qwen3.5-27B-MLX-MTP",
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    chosen = onboarding.screen_model(configured=None)

    captured = capsys.readouterr().out
    assert chosen == "trevon/Qwen3.5-27B-MLX-MTP"
    assert captured.count("Example: trevon/Qwen3.5-27B-MLX-MTP") == 2


def test_custom_hf_repo_accepts_huggingface_url(monkeypatch):
    answers = iter(
        [
            "3",
            "https://huggingface.co/trevon/Qwen3.5-27B-MLX-MTP/tree/main",
        ]
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    chosen = onboarding.screen_model(configured=None)

    assert chosen == "trevon/Qwen3.5-27B-MLX-MTP"


def test_run_quickstart_flow_fresh_skips_returning_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "fresh.json"))
    onboarding.save_state(
        {
            "model": "mtplx/old",
            "profile": "stable",
            "max": False,
            "target": "terminal",
        }
    )
    # With fresh=True we go straight to the onboarding screens; 4 inputs
    # (model + mode + interface + dashboard-companion).
    answers = iter(["1", "1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))
    state = onboarding.run_quickstart_flow(fresh=True)
    assert state is not None
    assert state["target"] == "openwebui"


def test_run_quickstart_flow_returns_none_on_ctrlc(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "abort.json"))

    def raising_input(_prompt: str = "") -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", raising_input)
    state = onboarding.run_quickstart_flow(fresh=True)
    assert state is None


# ---------- regression: ~/.mtplx/config.toml must not silently skip the flow --
#
# The original bug report: ``mtplx start`` skipped the onboarding because
# ``apply_user_config`` had pre-filled ``args.model`` from ``~/.mtplx/config.toml``
# and the heuristic mistook that for an explicit ``--model`` on the command line.
# These tests pin the new flag-scan based detection so the regression can't
# come back.
def test_explicit_cli_flag_detection_ignores_config_file_defaults(tmp_path, monkeypatch):
    """Simulating a user with config.toml model + bare ``mtplx start`` must
    still allow onboarding (i.e. ``has_explicit_model`` must be False)."""
    from mtplx.cli import _explicit_cli_flags

    flags = _explicit_cli_flags(["start"])
    assert "model" not in flags
    assert "profile" not in flags
    assert "max" not in flags


def test_explicit_cli_flag_detection_picks_up_typed_flags():
    from mtplx.cli import _explicit_cli_flags

    flags = _explicit_cli_flags(["start", "cli", "--model", "foo", "--profile", "stable"])
    assert "model" in flags
    assert "profile" in flags

    flags_eq = _explicit_cli_flags(["start", "--model=foo"])
    assert "model" in flags_eq

    flags_max = _explicit_cli_flags(["start", "--max"])
    assert "max" in flags_max


def test_start_invokes_onboarding_when_no_explicit_flags(tmp_path, monkeypatch):
    """Bare ``mtplx start`` (with config.toml pre-filling model) must drop
    into ``run_quickstart_flow`` — this is the user-reported bug."""
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "regression.json"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    invocations: list[dict] = []

    def fake_flow(
        *,
        fresh: bool = False,
        configured_model: str | None = None,
        open_dashboard_override: bool | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        invocations.append(
            {
                "fresh": fresh,
                "configured_model": configured_model,
                "open_dashboard_override": open_dashboard_override,
                "host": host,
                "port": port,
            }
        )
        return {
            "model": "mtplx/onboarded",
            "profile": "stable",
            "max": False,
            "target": "terminal",
        }

    monkeypatch.setattr("mtplx.ui.onboarding.run_quickstart_flow", fake_flow)

    # Simulate the post-config args object that start sees.
    import argparse

    args = argparse.Namespace(
        target=None,
        model="/some/configured/path",  # pre-filled from config.toml
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
        model_id="mtplx-test",
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
        _cli_flags=set(),  # no flags typed → must trigger onboarding
    )

    # Stub the downstream model resolution to avoid hitting MLX.
    def fake_resolve_model(model, *, cache_dir, download):
        return "/tmp/fake-runtime", {"model": model, "runtime_model": "/tmp/fake-runtime"}

    monkeypatch.setattr(
        "mtplx.commands.public._quickstart_resolve_model",
        fake_resolve_model,
    )

    def fake_gate(runtime_model, *, unsafe_force_unverified, yes):
        return {"runtime_contract": {"verified": True}, "compatibility": "verified"}, None

    monkeypatch.setattr("mtplx.commands.public._model_gate", fake_gate)

    def fake_run_openwebui(args, *, runtime_model, inspection):
        return 0

    monkeypatch.setattr(
        "mtplx.commands.public._quickstart_run_openwebui",
        fake_run_openwebui,
    )

    def fake_run_terminal(args, *, runtime_model, inspection):
        return 0

    monkeypatch.setattr(
        "mtplx.commands.public._quickstart_run_terminal_chat",
        fake_run_terminal,
    )
    monkeypatch.setattr("mtplx.ui.onboarding.screen_tuning_offer", lambda: False)

    from mtplx.commands.public import cmd_quickstart_public

    rc = cmd_quickstart_public(args)
    assert rc == 0
    assert len(invocations) == 1
    assert invocations[0]["configured_model"] == "/some/configured/path"
    assert invocations[0]["host"] == "127.0.0.1"
    assert invocations[0]["port"] == 8000
    assert args._onboarded is True
    assert args.model == "mtplx/onboarded"


def test_start_skips_onboarding_with_explicit_flags(tmp_path, monkeypatch):
    """Explicit ``--model`` (or any of the gating flags) must short-circuit
    the onboarding entirely."""
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "skipped.json"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)

    invocations: list[dict] = []

    def fake_flow(*, fresh: bool = False, configured_model: str | None = None):
        invocations.append({"fresh": fresh, "configured_model": configured_model})
        return {"model": "x", "profile": "stable", "max": False, "target": "terminal"}

    monkeypatch.setattr("mtplx.ui.onboarding.run_quickstart_flow", fake_flow)

    import argparse

    args = argparse.Namespace(
        target=None,
        model="custom/explicit",
        profile="performance-cold",
        max=False,
        prompt=None,
        dry_run=True,  # also short-circuits
        yes=False,
        fresh=False,
        download=False,
        cache_dir=None,
        unsafe_force_unverified=False,
        show_stats=True,
        host="127.0.0.1",
        port=8000,
        api_key=None,
        model_id="mtplx-test",
        warmup_tokens=16,
        stream_interval=1,
        rate_limit=0,
        max_response_tokens=None,
        reasoning_parser="qwen3",
        strict_warmup=False,
        strict_fast_path=False,
        json=True,
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        seed=0,
        system=None,
        _cli_flags={"model"},  # user typed --model
    )

    from mtplx.commands.public import cmd_quickstart_public

    rc = cmd_quickstart_public(args)
    assert rc == 0
    assert invocations == []  # onboarding was NOT invoked
    assert getattr(args, "_onboarded", False) is False


# ---------- local-folder scanning: typing a parent dir lists models -------
def _make_model_dir(parent, name, *, config: dict | None = None) -> Path:
    target = parent / name
    target.mkdir(parents=True, exist_ok=True)
    payload = {
        "architectures": ["Qwen3NextForCausalLM"],
        "model_type": "qwen3_next",
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
    }
    if config is not None:
        payload.update(config)
    (target / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_scan_walks_lmstudio_publisher_layout(tmp_path):
    _make_model_dir(tmp_path, "lmstudio-community/Qwen2.5-7B-Instruct-MLX-4bit")
    _make_model_dir(tmp_path, "mlx-community/Qwen3.5-27B-4bit")
    _make_model_dir(tmp_path, "Youssofal/Qwen3.6-27B-MTPLX")

    found = onboarding._scan_for_models(tmp_path)

    assert sorted(p.name for p in found) == sorted(
        [
            "Qwen2.5-7B-Instruct-MLX-4bit",
            "Qwen3.5-27B-4bit",
            "Qwen3.6-27B-MTPLX",
        ]
    )


def test_scan_for_models_honors_timeout(tmp_path, monkeypatch):
    _make_model_dir(tmp_path, "pub/model")
    ticks = iter([0.0, 10.0])
    monkeypatch.setattr(onboarding.time, "monotonic", lambda: next(ticks, 10.0))

    found = onboarding._scan_for_models(tmp_path, timeout_s=1.0)

    assert found == []


def test_classify_scanned_model_qwen_config_only_marks_mtp_missing(tmp_path):
    target = _make_model_dir(tmp_path, "mlx-community/Qwen-config-only")

    result = onboarding._classify_scanned_model(target)

    assert result.tier == "mtp-missing"
    assert result.arch_id == "qwen3-next-mtp"
    label, _ = onboarding._tier_badge(result.tier)
    assert "MTP weights missing" in label


def test_classify_scanned_model_qwen_sidecar_without_contract_is_arch_compatible(tmp_path):
    target = _make_model_dir(tmp_path, "Youssofal/Qwen-sidecar")
    (target / "mtp.safetensors").write_bytes(b"placeholder")

    result = onboarding._classify_scanned_model(target)

    assert result.tier == "arch-compatible"
    assert result.arch_id == "qwen3-next-mtp"


def test_classify_scanned_model_glm_config_only_marks_mtp_missing(tmp_path):
    target = tmp_path / "GLM-config-only"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "num_hidden_layers": 47,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )

    result = onboarding._classify_scanned_model(target)

    assert result.tier == "mtp-missing"
    assert result.arch_id == "glm4-moe-lite-mtp"


def test_classify_scanned_model_glm_sidecar_needs_verification(tmp_path):
    target = tmp_path / "GLM-sidecar-unverified"
    target.mkdir()
    (target / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Glm4MoeLiteForCausalLM"],
                "model_type": "glm4_moe_lite",
                "num_hidden_layers": 47,
                "num_nextn_predict_layers": 1,
            }
        ),
        encoding="utf-8",
    )
    (target / "mtp.safetensors").write_bytes(b"placeholder")

    result = onboarding._classify_scanned_model(target)

    assert result.tier == "needs-verification"
    assert result.arch_id == "glm4-moe-lite-mtp"


def test_tier_badges_distinguish_runnable_from_blocked():
    label, _ = onboarding._tier_badge("arch-compatible")
    assert "Runnable" in label
    label, _ = onboarding._tier_badge("needs-verification")
    assert "Needs MTPLX verification" in label
    label, _ = onboarding._tier_badge("mtp-invalid")
    assert "MTP weights invalid" in label
    label, _ = onboarding._tier_badge("backend-pending")
    assert "Backend not runnable yet" in label


def test_scan_and_pick_prints_candidate_progress_before_picker(tmp_path, monkeypatch, capsys):
    target = _make_model_dir(tmp_path, "lmstudio-community/Qwen-A")
    monkeypatch.setattr(
        onboarding,
        "_classify_scanned_model",
        lambda path: onboarding.ScannedModel(
            path=path,
            tier="arch-compatible",
            arch_id="qwen3-next-mtp",
            architecture="Qwen3NextForCausalLM",
        ),
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "1")

    chosen = onboarding._scan_and_pick(tmp_path)

    captured = capsys.readouterr().out
    assert chosen == str(target)
    assert "Found 1 candidate model folder(s). Checking configs..." in captured


def test_screen_model_local_folder_routes_through_picker(tmp_path, monkeypatch):
    target = _make_model_dir(tmp_path, "Real")
    invocations: list[str | None] = []

    def fake_picker(*, default):
        invocations.append(default)
        return str(target)

    monkeypatch.setattr(onboarding, "_pick_local_model", fake_picker)
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "4")

    chosen = onboarding.screen_model(configured=None)

    assert chosen == str(target)
    assert invocations == [None]


def test_screen_tuning_offer_uses_numbered_choices(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _prompt="": "2")

    assert onboarding.screen_tuning_offer() is False


def test_quickstart_applies_saved_tuned_depth(monkeypatch):
    import argparse

    from mtplx.commands import public

    args = argparse.Namespace(depth=3, _explicit_depth=False, cache_dir=None)
    monkeypatch.setattr(public, "_apple_hardware_context", lambda: {"chip": "Apple M5"})
    monkeypatch.setattr(public, "_software_context", lambda: {"mtplx_version": "test", "mlx_version": "test"})
    monkeypatch.setattr(public, "_mlx_backend_context", lambda: {"stock_mlx_likely": True})
    monkeypatch.setattr(public, "_tune_state_key", lambda *_args, **_kwargs: ("key", {}))
    monkeypatch.setattr(
        public,
        "_load_tune_record",
        lambda _key: {
            "payload": {
                "best": {
                    "mode": "D2",
                    "depth": 2,
                    "multiplier_vs_ar": 1.8,
                }
            }
        },
    )

    public._quickstart_apply_tuned_depth(
        args,
        runtime_model="/tmp/model",
        target="terminal",
        can_prompt=False,
    )

    assert args.depth == 2


def test_quickstart_tuning_prompt_can_save_and_apply(monkeypatch):
    import argparse

    from mtplx.commands import public

    args = argparse.Namespace(
        depth=3,
        _explicit_depth=False,
        cache_dir=None,
        unsafe_force_unverified=False,
    )
    monkeypatch.setattr(public, "_apple_hardware_context", lambda: {"chip": "Apple M5"})
    monkeypatch.setattr(public, "_software_context", lambda: {"mtplx_version": "test", "mlx_version": "test"})
    monkeypatch.setattr(public, "_mlx_backend_context", lambda: {"stock_mlx_likely": True})
    monkeypatch.setattr(public, "_tune_state_key", lambda *_args, **_kwargs: ("key", {}))
    monkeypatch.setattr("mtplx.ui.onboarding.screen_tuning_offer", lambda: True)
    calls = []
    records = iter(
        [
            None,
            {
                "payload": {
                    "best": {
                        "mode": "D1",
                        "depth": 1,
                        "multiplier_vs_ar": 1.4,
                    }
                }
            },
        ]
    )
    monkeypatch.setattr(public, "_load_tune_record", lambda _key: next(records))

    def fake_tune(*_args, **_kwargs):
        calls.append(True)
        return 0

    monkeypatch.setattr(public, "_cmd_tune", fake_tune)

    public._quickstart_apply_tuned_depth(
        args,
        runtime_model="/tmp/model",
        target="terminal",
        can_prompt=True,
    )

    assert calls == [True]
    assert args.depth == 1


def test_quickstart_explicit_depth_never_uses_tuning(monkeypatch):
    import argparse

    from mtplx.commands import public

    args = argparse.Namespace(depth=2, _explicit_depth=True, cache_dir=None)
    monkeypatch.setattr(
        public,
        "_load_tune_record",
        lambda _key: (_ for _ in ()).throw(AssertionError("should not load tuning")),
    )

    public._quickstart_apply_tuned_depth(
        args,
        runtime_model="/tmp/model",
        target="openwebui",
        can_prompt=True,
    )

    assert args.depth == 2


# ---------- app-aware model step + attach offer ------------------------------
def _write_installed_model(root: Path, name: str) -> Path:
    target = root / name
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"weights")
    return target


def test_screen_model_lists_installed_models_first(tmp_path, monkeypatch):
    from mtplx.model_catalog import scan_installed_models

    cache = tmp_path / "models"
    speed = _write_installed_model(
        cache, "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed"
    )
    installed = scan_installed_models(cache)
    assert installed

    answers = iter(["1"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    chosen = onboarding.screen_model(configured=None, installed=installed)

    assert chosen == str(speed)


def test_screen_model_preselects_app_model_when_installed(tmp_path, monkeypatch, capsys):
    from mtplx.model_catalog import scan_installed_models

    cache = tmp_path / "models"
    _write_installed_model(cache, "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed")
    quality = _write_installed_model(
        cache, "Youssofal--Qwen3.6-27B-MTPLX-Optimized-Quality"
    )
    installed = scan_installed_models(cache)
    assert len(installed) == 2

    answers = iter([""])  # accept default
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    chosen = onboarding.screen_model(
        configured=None,
        installed=installed,
        app_model="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality",
    )

    assert chosen == str(quality)
    output = capsys.readouterr().out
    assert "installed" in output


def test_screen_model_keeps_legacy_numbering_without_installed(monkeypatch):
    answers = iter(["2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    chosen = onboarding.screen_model(configured=None, installed=[])

    assert chosen == onboarding.optimized_quality_model_ref()


def test_returning_user_can_pick_same_as_the_app(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "app-option.json"))
    app_settings = tmp_path / "app-settings.json"
    app_settings.write_text(
        json.dumps({"model": "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_APP_SETTINGS_PATH", str(app_settings))
    onboarding.save_state(
        {
            "model": "mtplx/foo",
            "profile": "sustained",
            "max": False,
            "target": "pi",
        }
    )
    answers = iter(["2"])  # pick "Same as the MTPLX app"
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_quickstart_flow(fresh=False)

    assert state is not None
    assert state["model"] == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality"
    assert state["target"] == "pi"
    persisted = onboarding.load_state()
    assert persisted is not None
    assert persisted["model"] == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality"


def test_returning_user_keeps_yes_no_prompt_without_app_settings(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "no-app.json"))
    onboarding.save_state(
        {
            "model": "mtplx/foo",
            "profile": "sustained",
            "max": False,
            "target": "pi",
        }
    )
    answers = iter([""])  # default Y on the legacy confirm
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_quickstart_flow(fresh=False)

    assert state is not None
    assert state["model"] == "mtplx/foo"


def test_quickstart_flow_offers_attach_when_daemon_running(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "attach.json"))
    from types import SimpleNamespace

    daemon = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model="mtplx-test-model",
        owned_by_app=True,
    )
    monkeypatch.setattr(onboarding, "_detect_running_daemon", lambda: daemon)
    answers = iter(["1"])  # take the attach offer
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_quickstart_flow(fresh=False)

    assert state == {
        "attach": {
            "host": "127.0.0.1",
            "port": 8000,
            "model": "mtplx-test-model",
            "owned_by_app": True,
        }
    }


def test_quickstart_flow_attach_declined_continues_to_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_QUICKSTART_STATE", str(tmp_path / "attach-no.json"))
    from types import SimpleNamespace

    daemon = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model="mtplx-test-model",
        owned_by_app=False,
    )
    monkeypatch.setattr(onboarding, "_detect_running_daemon", lambda: daemon)
    # Decline attach (2), then walk model + mode + interface + companion.
    answers = iter(["2", "1", "1", "1", "2"])
    monkeypatch.setattr(builtins, "input", lambda _prompt="": next(answers))

    state = onboarding.run_quickstart_flow(fresh=False)

    assert state is not None
    assert "attach" not in state
    assert state["model"] == onboarding._verified_default_model()
