from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from mtplx.cli import build_parser, main
from mtplx.commands import public
from mtplx.profiles import (
    DEFAULT_HF_MODEL_ID,
    DEFAULT_PUBLIC_MODEL_ID,
    LEGACY_OPTIMIZED_PUBLIC_MODEL_ID,
    QUALITY_HF_MODEL_ID,
    QUALITY_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
    QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
    QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
)
from mtplx.version import DISPLAY_VERSION, __version__


def _pin_big_apple_silicon(monkeypatch):
    """Pin the hardware probe to a big newer-generation machine.

    The auto default is device-aware: M1/M2 routes to FP16 and under
    32 GiB routes to the 9B artifact. CI runners are small M1 hosts,
    so tests that assert the big-machine 27B default must pin the
    probe instead of inheriting whatever host they run on.
    """
    monkeypatch.setattr(
        "mtplx.default_models.detect_apple_silicon",
        lambda: {
            "apple_silicon_generation": "m5",
            "chip": "Apple M5 Max",
            "memory_gib": 64.0,
        },
    )
    # Keep parser/product-default tests independent of whichever local
    # release-candidate artifacts happen to be installed on the host: the
    # default resolvers prefer a complete local build over the Hub id.
    monkeypatch.setenv("MTPLX_QWEN38_BARE_SPEED_MODEL", "off")
    monkeypatch.setenv("MTPLX_QWEN38_OPTIMIZED_SPEED_MODEL", "off")
    monkeypatch.setattr(
        "mtplx.default_models._QWEN38_OPTIMIZED_SPEED_FP16_LOCAL_CANDIDATES", ()
    )


def test_version_metadata_matches_package_metadata():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]

    assert __version__ == project["version"]
    assert DISPLAY_VERSION == __version__


def test_version_command_without_subcommand(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr().out
    assert f"mtplx {DISPLAY_VERSION} ({__version__})" in captured


def test_runtime_mode_display_respects_ar_mode():
    assert public._runtime_mode_display("sustained") == "Sustained MTP"
    assert (
        public._runtime_mode_display("sustained", generation_mode="ar")
        == "Sustained AR"
    )
    assert (
        public._runtime_mode_display(
            "sustained",
            max_mode=True,
            generation_mode="ar",
        )
        == "Sustained Max AR"
    )


def test_public_serve_applies_gemma4_backend_defaults():
    inspection = {
        "recommended_backend": "gemma4_assistant",
        "recommended_sampler": {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
        "gemma4_pair": {"benchmark": {"best_block_size": 6}},
    }
    args = SimpleNamespace(
        model_id=DEFAULT_PUBLIC_MODEL_ID,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        draft_temperature=0.6,
        draft_top_p=0.95,
        draft_top_k=20,
        reasoning_parser="qwen3",
        chat_template_profile="local_qwen36",
        adaptive_policy="expected_value",
    )

    public._apply_backend_serve_defaults(args, inspection)

    assert args.model_id == "mtplx-gemma4-31b-assistant-mtp"
    assert args.temperature == 1.0
    assert args.top_k == 64
    assert args.depth == 6
    assert args.draft_temperature == 1.0
    assert args.draft_top_k == 64
    assert args.reasoning_parser == "gemma4"
    assert args.chat_template_profile == "tokenizer"
    assert args.adaptive_policy == "none"


def test_app_parent_watchdog_stops_child_when_parent_is_gone():
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])

    started = time.monotonic()
    code = public._run_server_child_with_app_parent_watchdog(
        [
            sys.executable,
            "-c",
            "import time\ntime.sleep(30)\n",
        ],
        env=os.environ.copy(),
        cwd=Path.cwd(),
        app_parent_pid=parent.pid,
        poll_seconds=0.05,
        shutdown_grace_s=0.1,
    )

    parent.wait(timeout=5)
    assert code == 130
    assert time.monotonic() - started < 5


def test_serve_wrapper_signal_stops_child_daemon(tmp_path):
    pid_file = tmp_path / "child.pid"
    term_file = tmp_path / "child.terminated"
    child_script = tmp_path / "child.py"
    wrapper_script = tmp_path / "wrapper.py"
    child_script.write_text(
        "import signal, sys, time\n"
        "from pathlib import Path\n"
        f"pid_file = Path({str(pid_file)!r})\n"
        f"term_file = Path({str(term_file)!r})\n"
        "pid_file.write_text(str(__import__('os').getpid()))\n"
        "def stop(_signum, _frame):\n"
        "    term_file.write_text('terminated')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    wrapper_script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "from mtplx.commands import public\n"
        "code = public._run_server_child_with_app_parent_watchdog(\n"
        f"    [{sys.executable!r}, {str(child_script)!r}],\n"
        "    env=os.environ.copy(),\n"
        "    cwd=Path.cwd(),\n"
        "    app_parent_pid=None,\n"
        "    poll_seconds=0.05,\n"
        "    shutdown_grace_s=1.0,\n"
        ")\n"
        "sys.exit(code)\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    repo = str(Path.cwd())
    env["PYTHONPATH"] = repo + os.pathsep + env.get("PYTHONPATH", "")
    wrapper = subprocess.Popen([sys.executable, str(wrapper_script)], env=env)
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if pid_file.exists():
                child_pid = int(pid_file.read_text(encoding="utf-8"))
                break
            assert wrapper.poll() is None
            time.sleep(0.05)
        assert child_pid is not None

        os.kill(wrapper.pid, signal.SIGTERM)
        wrapper.wait(timeout=5)

        deadline = time.monotonic() + 5
        while public._pid_is_alive(child_pid) and time.monotonic() < deadline:
            time.sleep(0.05)

        assert wrapper.returncode == 128 + signal.SIGTERM
        assert term_file.read_text(encoding="utf-8") == "terminated"
        assert not public._pid_is_alive(child_pid)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()
        if child_pid is not None and public._pid_is_alive(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_empty_cli_shows_friendly_consumer_help(capsys):
    code = main([])

    captured = capsys.readouterr().out
    assert code == 0
    # Compact help: ASCII banner + version pill + Commands + Examples + footer.
    assert f"v{DISPLAY_VERSION}" in captured
    assert "Native MTP speculative decoding" in captured
    assert "mtplx quickstart" in captured
    assert "Prepare config and the model cache" in captured
    assert "mtplx start" in captured
    assert "mtplx help advanced" in captured
    assert "runtime-smoke" not in captured
    assert "capture-commit-equivalence" not in captured


def test_hardware_inspect_json(monkeypatch, capsys):
    import mtplx.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "inspect_hardware",
        lambda: {
            "chip": "Apple M5 Max",
            "macos_version": "26.2",
            "mlx_version": "0.31.0",
            "hardware_acceleration_eligible": True,
            "hardware_acceleration_confirmed": False,
            "warnings": ["Eligibility is not proof."],
        },
    )

    code = main(["hardware", "inspect", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["chip"] == "Apple M5 Max"
    assert payload["hardware_acceleration_eligible"] is True
    assert payload["hardware_acceleration_confirmed"] is False


def test_hardware_json_defaults_to_inspect(monkeypatch, capsys):
    import mtplx.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "inspect_hardware",
        lambda: {
            "chip": "Apple M5 Max",
            "macos_version": "26.2",
            "mlx_version": "0.31.0",
            "hardware_acceleration_eligible": True,
            "hardware_acceleration_confirmed": True,
            "warnings": [],
        },
    )

    code = main(["hardware", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["chip"] == "Apple M5 Max"
    assert payload["hardware_acceleration_confirmed"] is True


def test_bench_prefill_ladder_dry_run_json(monkeypatch, capsys):
    import mtplx.prefill_bench as prefill_bench

    monkeypatch.setattr(
        prefill_bench,
        "inspect_hardware",
        lambda: {
            "chip": "Apple M5 Max",
            "hardware_acceleration_eligible": True,
            "hardware_acceleration_confirmed": False,
        },
    )

    code = main(
        [
            "bench",
            "prefill-ladder",
            "--contexts",
            "512,1k",
            "--max-tokens",
            "8",
            "--defer-verify-hidden-eval",
            "--verify-hidden-mode",
            "logits-first-committed-slice",
            "--dry-run",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["kind"] == "prefill_ladder"
    assert payload["dry_run"] is True
    assert payload["seed"] == 0
    assert payload["vary_seed_by_context"] is False
    assert payload["inter_context_cache_cleanup"]["enabled"] is True
    assert payload["inter_context_cache_cleanup"]["events"] == 0
    assert payload["contexts"] == [512, 1024]
    assert payload["rows"] == []
    assert payload["prompt"]["style"] == "coding-agent"
    assert payload["prompt"]["format"] == "chat"
    assert payload["prompt"]["enable_thinking"] is False
    assert payload["prompt"]["policy"] == "coding_agent_tail_v2"
    assert payload["prompt"]["tail_sha256"]
    assert payload["prompt"]["release_valid"] is True
    assert payload["prefill_layout"]["requested"] == "profile"
    assert payload["prefill_layout"]["env_value"] is None
    assert payload["defer_verify_hidden_eval_override"] is True
    assert payload["verify_hidden_mode_override"] == "logits_first_committed_slice"
    assert payload["env"]["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] == "1"
    assert payload["env"]["MTPLX_VERIFY_HIDDEN_MODE"] == "logits_first_committed_slice"
    assert payload["recommended_plugged_in_commands"]
    assert "--prompt-format chat" in payload["recommended_plugged_in_commands"][0]
    assert "--disable-thinking" in payload["recommended_plugged_in_commands"][0]
    assert payload["profile"]["env"]["MTPLX_LAZY_VERIFY_LOGITS"] == "1"
    # Lazy is the active strategy; the batched lane must not be claimed
    # while the lazy gate kills it (PR #314 dead-pair fix).
    assert payload["profile"]["env"]["MTPLX_BATCH_TARGET_ARRAYS"] == "0"
    assert payload["profile"]["env"]["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1"
    assert payload["profile"]["env"]["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP"] == "1"
    assert (
        payload["profile"]["env"]["MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY"] == "auto"
    )
    assert payload["profile"]["env"]["MTPLX_PREFILL_OMLX_EXTERNAL"] == "1"
    assert payload["profile"]["env"]["MTPLX_DEFER_VERIFY_HIDDEN_EVAL"] == "1"
    assert (
        payload["profile"]["env"]["MTPLX_VERIFY_HIDDEN_MODE"]
        == "logits_first_committed_slice"
    )
    assert payload["profile"]["env"]["MTPLX_LONG_CONTEXT_MTP_DEPTH_POLICY"] == "off"
    assert (
        payload["profile"]["env"]["MTPLX_LONG_CONTEXT_MTP_DEPTH_THRESHOLD"] == "98304"
    )
    assert payload["profile"]["env"]["MTPLX_LONG_CONTEXT_MTP_DEPTH"] == "3"
    assert payload["profile"]["env"]["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] == "0"


def test_server_cli_surfaces_default_to_sustained_profile():
    parser = build_parser()

    quickstart_args = parser.parse_args(["quickstart"])
    serve_args = parser.parse_args(["serve", "--yes"])

    assert quickstart_args.profile == "sustained"
    assert serve_args.profile == "sustained"


def test_serve_parser_accepts_legacy_app_profile_strings():
    # Shipped app builds persisted profile "auto" / "sustained-max"; the
    # parser must canonicalize them instead of exiting 2 at startup.
    parser = build_parser()

    auto_args = parser.parse_args(["serve", "--yes", "--profile", "auto"])
    max_args = parser.parse_args(["serve", "--yes", "--profile", "sustained-max"])

    assert auto_args.profile == "sustained"
    assert max_args.profile == "sustained"


def test_serve_parser_accepts_auto_generation_mode_as_engine_default():
    from mtplx.commands.public import _generation_mode_from_args

    parser = build_parser()
    args = parser.parse_args(["serve", "--yes", "--generation-mode", "auto"])

    assert args.generation_mode == "auto"
    assert _generation_mode_from_args(args) == "mtp"


def test_native_ar_only_runtime_degrades_to_ar_and_disables_mtp_loading():
    inspection = {
        "compatibility": {
            "runtime_compatibility": "native-ar-only",
            "can_run": True,
        }
    }
    lines: list[str] = []
    mtp_args = SimpleNamespace(generation_mode="mtp", load_mtp=True, depth=3)

    assert (
        public._apply_runtime_compatibility_mode(
            mtp_args,
            inspection,
            printer=lines.append,
        )
        is None
    )
    assert lines == [
        "target-only AR architecture -> mtp_off: serving autoregressive "
        "(this checkpoint family has no native MTP head).",
    ]
    assert mtp_args.generation_mode == "ar"
    assert mtp_args.depth == 0
    assert mtp_args.load_mtp is False

    ar_args = SimpleNamespace(generation_mode="ar", load_mtp=True)
    assert public._apply_runtime_compatibility_mode(ar_args, inspection) is None
    assert ar_args.load_mtp is False

    auto_ar_args = SimpleNamespace(
        generation_mode="auto",
        no_mtp=True,
        load_mtp=True,
    )
    assert public._apply_runtime_compatibility_mode(auto_ar_args, inspection) is None
    assert auto_ar_args.load_mtp is False


def test_serve_cli_accepts_native_app_thermal_poll_flag():
    parser = build_parser()

    args = parser.parse_args(["serve", "--yes", "--enable-thermal-poll"])

    assert args.enable_thermal_poll is True


def test_shell_banner_env_suppresses_compact_help_ascii(monkeypatch, capsys):
    monkeypatch.setenv("MTPLX_SHELL_BANNER_SHOWN", "1")

    code = main([])

    captured = capsys.readouterr().out
    assert code == 0
    assert "███╗" not in captured
    assert "Commands" in captured
    assert "mtplx start" in captured


def test_render_banner_respects_shell_banner_env(monkeypatch, capsys):
    from mtplx.ui.banner import render_banner

    monkeypatch.setenv("MTPLX_SHELL_BANNER_SHOWN", "1")
    render_banner(no_color=True)

    assert capsys.readouterr().out == ""


def test_top_level_help_is_friendly(capsys):
    code = main(["--help"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "Commands" in captured
    assert "Examples" in captured
    assert "mtplx quickstart" in captured
    assert "mtplx help <command>" in captured
    assert "positional arguments" not in captured


def test_advanced_help_keeps_lab_tools_discoverable(capsys):
    code = main(["help", "advanced"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX advanced tools" in captured
    assert "mtplx bench" in captured
    assert "profile *" in captured
    assert "runtime-smoke" in captured


def test_help_with_no_topic_is_verbose_not_compact(capsys):
    """Bare `mtplx` shows compact help; `mtplx help` must show the verbose view."""
    capsys.readouterr()
    main([])
    compact = capsys.readouterr().out
    code = main(["help"])
    verbose = capsys.readouterr().out
    assert code == 0
    assert len(verbose) > len(compact), (
        "`mtplx help` must be more verbose than bare `mtplx`"
    )
    # Verbose-only sections.
    assert "Overview" in verbose
    assert "Help subtopics" in verbose
    assert "mtplx help commands" in verbose
    assert "mtplx help flags" in verbose
    assert "mtplx help advanced" in verbose
    # Bare `mtplx` does not list the subtopics inline.
    assert "Help subtopics" not in compact


def test_help_commands_lists_consumer_and_advanced_surfaces(capsys):
    code = main(["help", "commands"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX commands" in captured
    assert "Consumer commands" in captured
    # Consumer rows
    assert "quickstart" in captured
    assert "setup" in captured
    assert "models" in captured
    # Advanced rows
    assert "bench *" in captured
    assert "runtime-smoke" in captured


def test_help_flags_lists_every_command_flag(capsys):
    code = main(["help", "flags"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX flags" in captured
    assert "Top-level options" in captured
    # Several known flags across multiple commands must appear.
    assert "--temperature" in captured
    assert "--top-p" in captured
    assert "--top-k" in captured
    assert "--port" in captured
    assert "--profile" in captured
    assert "--max-tokens" in captured


def test_main_menu_advertises_help_command(capsys):
    """`help` must appear in the bare `mtplx` command list as one of the first commands."""
    code = main([])

    captured = capsys.readouterr().out
    assert code == 0
    # `help` appears in the Commands section, not just as part of the footer.
    assert "  help " in captured or "help         " in captured
    # Listed near the top: must come before the lab-only commands.
    quickstart_pos = captured.find("start")
    help_pos = captured.find("\n  help")
    inspect_pos = captured.find("inspect")
    assert quickstart_pos != -1 and help_pos != -1 and inspect_pos != -1
    assert quickstart_pos < help_pos < inspect_pos


def test_start_help_is_a_user_journey(capsys):
    code = main(["help", "start"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX start" in captured
    # Help leads with the interactive onboarding (model/mode/where) instead
    # of the old browser/CLI bifurcation.
    assert "Interactive end-to-end setup" in captured
    assert "What gets asked" in captured
    assert "Sustained" in captured
    assert "Sustained Max" in captured
    assert "Burst" in captured
    assert "ThermalForge" in captured  # fan-backed modes advertise fan control
    # Power-user shortcuts still showcased.
    assert "mtplx start --fresh" in captured
    assert "mtplx start --max" in captured
    assert "mtplx start cli" in captured
    assert "mtplx start pi" in captured
    assert "mtplx start opencode" in captured
    assert "mtplx start --download" in captured
    assert "/speed" in captured
    assert "Aliases:" in captured
    assert "openwebui" in captured
    assert "usage: mtplx" not in captured


def test_opencode_memory_defaults_scale_on_high_memory_darwin(monkeypatch):
    monkeypatch.setattr(public.sys, "platform", "darwin")
    monkeypatch.setattr(
        public.subprocess,
        "check_output",
        lambda *args, **kwargs: str(128 * 1024**3),
    )
    env: dict[str, str] = {}

    public._apply_opencode_memory_env_defaults(env)

    assert env["MTPLX_SESSION_BANK_MAX_ENTRIES"] == "32"
    # "auto": the engine budgets half the post-model RAM surplus at startup.
    assert env["MTPLX_SESSION_BANK_MAX_BYTES"] == "auto"
    assert env["MTPLX_SESSION_BANK_PER_SESSION_BYTES"] == "auto"
    assert env["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1"
    assert env["MTPLX_LAZY_BONUS_VERIFY"] == "1"
    assert env["MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER"] == "1"
    assert env["MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE"] == "1"
    # #282: the launcher exports no read-inspection compaction battery.
    assert "MTPLX_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES" not in env
    assert "MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS" not in env
    assert env["MTPLX_TOOL_PROMPT_MODE"] == "hybrid"


def test_opencode_memory_defaults_stay_conservative_below_high_memory(monkeypatch):
    monkeypatch.setattr(public.sys, "platform", "darwin")
    monkeypatch.setattr(
        public.subprocess,
        "check_output",
        lambda *args, **kwargs: str(64 * 1024**3),
    )
    env: dict[str, str] = {}

    public._apply_opencode_memory_env_defaults(env)

    assert env["MTPLX_SESSION_BANK_MAX_ENTRIES"] == "6"
    assert env["MTPLX_SESSION_BANK_MAX_BYTES"] == "auto"
    assert env["MTPLX_SESSION_BANK_PER_SESSION_BYTES"] == "auto"
    assert env["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1"
    assert env["MTPLX_LAZY_BONUS_VERIFY"] == "1"
    assert env["MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER"] == "1"
    assert env["MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE"] == "1"
    # #282: the launcher exports no read-inspection compaction battery.
    assert "MTPLX_ACTIVE_READ_INSPECTION_TOTAL_MAX_LINES" not in env
    assert "MTPLX_READ_ONLY_INSPECTION_FORCE_ANSWER_AFTER_TOOLS" not in env
    assert env["MTPLX_TOOL_PROMPT_MODE"] == "hybrid"


def test_unknown_command_is_targeted_not_argparse_dump(capsys):
    code = main(["wut"])

    captured = capsys.readouterr().out
    assert code == 2
    assert "Unknown command: wut" in captured
    assert "mtplx setup" in captured
    assert "usage: mtplx" not in captured


def test_start_dry_run_is_consumer_friendly(monkeypatch, tmp_path, capsys):
    """The default start opens the browser chat; the terminal flow lives behind `cli`."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    code = main(
        [
            "start",
            "cli",
            "--dry-run",
            "--model",
            "models/example",
            "--yes",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert "MTPLX start" in captured
    assert "model: models/example" in captured
    assert "profile: sustained" in captured
    assert (
        "then: load once -> chat in this terminal -> stream output -> show speed stats"
        in captured
    )


def test_start_auto_default_can_route_to_fp16(monkeypatch, tmp_path, capsys):
    _pin_big_apple_silicon(monkeypatch)
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_DEFAULT_MODEL_VARIANT", "fp16")

    code = main(["start", "cli", "--dry-run", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == QWEN38_OPTIMIZED_SPEED_FP16_HF_MODEL_ID
    assert payload["default_model_selection"]["variant"] == "fp16"
    assert payload["default_model_selection"]["precision"].startswith(
        "FP16 build for M1 and M2 Macs"
    )


def test_start_default_openwebui_dry_run_uses_resolved_model(
    monkeypatch, tmp_path, capsys
):
    _pin_big_apple_silicon(monkeypatch)
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    code = main(["start", "--dry-run", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == DEFAULT_HF_MODEL_ID
    assert payload["openwebui"]["model_id"] == DEFAULT_PUBLIC_MODEL_ID
    assert payload["openwebui"]["model_id"] != "none"
    assert f"--model {payload['model']}" in payload["openwebui"]["server_command"]
    assert "--model None" not in payload["openwebui"]["server_command"]


def test_start_explicit_model_bypasses_auto_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_DEFAULT_MODEL_VARIANT", "fp16")

    code = main(["start", "cli", "--dry-run", "--json", "--model", "local/custom"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == "local/custom"
    assert payload["default_model_selection"] is None


def test_quality_model_ref_uses_quality_public_model_id():
    value = public._public_model_id_for_ref(
        "/tmp/Qwen3.6-27B-MTPLX-Optimized-Quality",
        default_model_id="mtplx-qwen36-27b-optimized-speed",
    )

    assert value == QUALITY_PUBLIC_MODEL_ID


def test_legacy_optimized_model_ref_uses_neutral_public_model_id():
    value = public._public_model_id_for_ref(
        "/tmp/Qwen3.6-27B-MTPLX-Optimized",
        default_model_id=DEFAULT_PUBLIC_MODEL_ID,
    )

    assert value == LEGACY_OPTIMIZED_PUBLIC_MODEL_ID


def test_explicit_model_id_wins_over_loaded_artifact_identity():
    args = SimpleNamespace(
        model="/tmp/Qwen3.6-27B-MTPLX-Optimized-Quality",
        model_id="custom-served-id",
        _cli_flags={"model-id"},
    )

    assert public._public_model_id_for_args(args, args.model) == "custom-served-id"


def test_start_default_target_is_browser(monkeypatch, tmp_path):
    """`mtplx start` (no target) must dry-run as the openwebui browser path."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    args = SimpleNamespace(
        target=None,
        model="models/example",
        cache_dir=None,
        download=False,
        dry_run=True,
        json=True,
        yes=True,
        prompt=None,
        profile="performance-cold",
        show_stats=True,
        unsafe_force_unverified=False,
        host="127.0.0.1",
        port=8000,
        model_id=None,
    )
    code = public.cmd_quickstart_public(args)
    assert code == 0


def test_model_gate_force_attempts_architecture_compatible_unverified(
    monkeypatch,
    capsys,
):
    class FakeInspection:
        def to_dict(self):
            return {
                "model_dir": "/models/custom-mtp",
                "compatibility": {
                    "tier": "architecture-compatible-but-unverified",
                    "can_run": False,
                    "exit_code": 2,
                    "runtime_compatibility": "needs-contract",
                    "unsafe_force_required": False,
                },
            }

    monkeypatch.setattr(public, "inspect_model", lambda model: FakeInspection())

    inspection, exit_code = public._model_gate(
        "/models/custom-mtp",
        unsafe_force_unverified=True,
        yes=True,
    )

    assert exit_code is None
    assert inspection["compatibility"]["runtime_compatibility"] == "needs-contract"
    assert "loader result is authoritative" in capsys.readouterr().err


def test_start_target_aliases_route_correctly(monkeypatch, tmp_path, capsys):
    """Target aliases normalize to the surface MTPLX actually starts."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    def run(target: str | None) -> dict:
        argv = ["start"]
        if target:
            argv.append(target)
        argv += ["--dry-run", "--json", "--model", "models/example", "--yes"]
        capsys.readouterr()
        code = main(argv)
        captured = capsys.readouterr().out
        assert code == 0, captured
        return json.loads(captured)

    assert run(None)["target"] == "openwebui"
    assert run("web")["target"] == "openwebui"
    assert run("openwebui")["target"] == "openwebui"
    assert run("open-webui")["target"] == "openwebui"
    assert run("cli")["target"] == "terminal"
    assert run("terminal")["target"] == "terminal"
    assert run("pi")["target"] == "pi"
    assert run("pie")["target"] == "pi"
    assert run("opencode")["target"] == "opencode"
    assert run("open-code")["target"] == "opencode"
    assert run("oc")["target"] == "opencode"
    assert run("swival")["target"] == "swival"
    assert run("sv")["target"] == "swival"
    assert run("hermes")["target"] == "hermes"
    assert run("hermes-agent")["target"] == "hermes"


def test_start_opencode_dry_run_json_writes_no_hidden_cap(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.setenv(
        "MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(tmp_path / "default.dat")
    )

    code = main(
        [
            "start",
            "opencode",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--api-key",
            "1234",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "opencode"
    assert payload["opencode"]["api_base_url"] == "http://127.0.0.1:18083/v1"
    assert payload["opencode"]["model_ref"].startswith("mtplx/")
    assert payload["opencode"]["transport_headers"] == {"x-mtplx-client": "opencode"}
    assert payload["opencode"]["no_hidden_max_tokens"] is True
    assert "maxTokens" not in json.dumps(payload["opencode"]["config"])
    assert payload["opencode"]["provider"]["models"]
    command = payload["opencode"]["server_command"]
    # 2026-07-16: `mtplx start opencode` runs the app's measured OpenCode
    # lane (serial + latency). Those are the base defaults, so the command
    # omits the scheduler flags entirely; the old ar_batch/agent stack must
    # not reappear.
    assert "--scheduler-mode" not in command
    assert "--batching-preset" not in command
    assert "--decode-batch-max" not in command
    assert "--batch-wait-ms" not in command
    assert "--prefill-chunk-tokens 2048" in command
    assert "--ssd-session-cache on" in command
    assert "--ssd-session-cache-max-size 32GB" in command
    assert "--ssd-session-cache-min-prefix-tokens 1024" in command
    assert "--api-key $MTPLX_API_KEY" in command
    assert "--top-k 20" in command
    assert "--max-response-tokens" not in command
    assert "--tool-prompt-mode hybrid" in command
    assert "--chat-template-profile local_qwen36" in command
    assert "--reasoning auto" in command
    assert payload["opencode"]["tool_prompt_mode"] == "hybrid"
    assert payload["opencode"]["chat_template_profile"] == "local_qwen36"
    assert "server_max_response_tokens" not in payload["opencode"]
    model_id = payload["opencode"]["model_id"]
    assert payload["opencode"]["provider"]["options"]["apiKey"] == "<configured>"
    model = payload["opencode"]["config"]["provider"]["mtplx"]["models"][model_id]
    # The dry-run stub resolves the default qwen3_next lane: verified codec
    # (reasoning_content round-trips), no effort dial (OpenCode's built-in
    # effort picker disabled tier by tier), temperature declared so explicit
    # client-side choices transmit (nothing is injected for MTPLX ids).
    assert model["reasoning"] is True
    assert model["temperature"] is True
    assert "interleaved" not in model
    assert "options" not in model
    assert all(value == {"disabled": True} for value in model["variants"].values())


def test_start_opencode_dry_run_emits_explicit_ssd_off(monkeypatch, tmp_path, capsys):
    """Issue #140 class: a generated server_command must carry an explicit
    --ssd-session-cache off. The CLIs that re-parse these commands default
    the flag to "on" (kvcache-v2), so omitting "off" silently re-enables
    the cache the user disabled."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.setenv(
        "MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(tmp_path / "default.dat")
    )

    code = main(
        [
            "start",
            "opencode",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--api-key",
            "1234",
            "--ssd-session-cache",
            "off",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    command = payload["opencode"]["server_command"]
    assert "--ssd-session-cache off" in command
    assert "--ssd-session-cache-max-size" not in command
    assert "--ssd-session-cache-min-prefix-tokens" not in command


def test_start_opencode_dry_run_uses_qwen36_contract_draft_sampler(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.setenv(
        "MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(tmp_path / "default.dat")
    )
    model_dir = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    runtime_contract = {
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 3,
        "recommended_profile": "sustained",
        "recommended_draft_sampler": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
        },
    }

    class FakeInspection:
        def to_dict(self) -> dict[str, object]:
            return {
                "model_dir": str(model_dir),
                "recommended_backend": "qwen3_next",
                "runtime_compatibility": "native-contract-gated",
                "compatibility": {
                    "can_run": True,
                    "exit_code": 0,
                    "runtime_contract": runtime_contract,
                },
            }

    monkeypatch.setattr(public, "inspect_model", lambda _model: FakeInspection())

    code = main(
        [
            "start",
            "opencode",
            "--dry-run",
            "--json",
            "--model",
            str(model_dir),
            "--profile",
            "sustained",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    command = payload["opencode"]["server_command"]
    assert code == 0
    assert payload["opencode"]["mtp_depth"] == 3
    assert payload["opencode"]["target_sampler"] == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }
    assert payload["opencode"]["draft_sampler"] == {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
    }
    assert payload["opencode"]["draft_sampler_source"] == "model_contract_or_profile"
    assert "--depth 3" in command
    assert "--temperature 0.6" in command
    assert "--top-p 0.95" in command
    assert "--draft-temperature 0.7" in command
    assert "--draft-top-p 0.95" in command
    assert "--draft-top-k 20" in command
    assert "--max-response-tokens" not in command
    assert "maxTokens" not in json.dumps(payload["opencode"]["config"])


def test_serve_dry_run_prefers_contract_draft_sampler_over_internal_defaults(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    model_dir = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    runtime_contract = {
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 3,
        "recommended_profile": "sustained",
        "recommended_draft_sampler": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
        },
    }

    class FakeInspection:
        def to_dict(self) -> dict[str, object]:
            return {
                "model_dir": str(model_dir),
                "recommended_backend": "qwen3_next",
                "runtime_compatibility": "native-contract-gated",
                "compatibility": {
                    "can_run": True,
                    "exit_code": 0,
                    "runtime_contract": runtime_contract,
                },
            }

    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(public, "_port_is_busy", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (str(model_dir), None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda runtime_model, *, unsafe_force_unverified, yes: (
            FakeInspection().to_dict(),
            None,
        ),
    )

    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            str(model_dir),
            "--profile",
            "sustained",
            "--yes",
        ]
    )
    args.dry_run = True
    args.json = True

    code = public.cmd_serve_public(args)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["depth"] == 3
    assert "--draft-temperature 0.7" in payload["server_command"]
    assert "--draft-top-p 0.95" in payload["server_command"]
    assert "--draft-top-k 20" in payload["server_command"]
    # --profile sustained was explicitly typed: the per-model turbo default
    # must not override a user decision.
    assert payload["profile"] == "sustained"


def _serve_dry_run_payload_for_model(monkeypatch, capsys, model_dir, extra_args=()):
    """Drive `mtplx serve --dry-run --json` against a stubbed local model."""

    runtime_contract = {
        "arch_id": "qwen3-next-mtp",
        "mtp_depth_max": 3,
        "recommended_profile": "sustained",
    }
    inspection = {
        "model_dir": str(model_dir),
        "recommended_backend": "qwen3_next",
        "runtime_compatibility": "native-contract-gated",
        "compatibility": {
            "can_run": True,
            "exit_code": 0,
            "runtime_contract": runtime_contract,
        },
    }
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(public, "_port_is_busy", lambda *_a, **_k: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (str(model_dir), None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda runtime_model, *, unsafe_force_unverified, yes: (inspection, None),
    )
    args = build_parser().parse_args(
        ["serve", "--model", str(model_dir), "--yes", *extra_args]
    )
    args.dry_run = True
    args.json = True
    code = public.cmd_serve_public(args)
    assert code == 0
    return json.loads(capsys.readouterr().out)


def test_serve_defaults_quantized_27b_flagships_to_turbo(monkeypatch, tmp_path, capsys):
    """Bare `mtplx serve` on the quantized 27B flagships resolves turbo.

    This is the same launch rule the macOS app applies (Speed/Quality ->
    turbo). Before 2026-07-05 the bare CLI stayed on sustained, so every
    OpenAI-API consumer measured the slow path while the app ran the NAX +
    compiled-verify fast path.
    """

    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    for dir_name in (
        "Qwen3.6-27B-MTPLX-Optimized-Speed-V2",
        "Qwen3.6-27B-MTPLX-Optimized-Speed",
        "Qwen3.6-27B-MTPLX-Optimized-Quality",
        "Qwen3.6-27B-MTPLX-Optimized",
        # 27B Speed-FP16 (M1/M2 routing target) promoted 2026-07-07:
        # turbo measured 1.98-2.08x over true AR on the artifact itself
        # (INT4/g64 weights, fp16 activations) vs 1.34x on sustained.
        "Qwen3.6-27B-MTPLX-Optimized-Speed-FP16",
        # 27B Quality-FP16 (M1/M2 quality pick, built 2026-07-07):
        # turbo measured 2.5x over true AR on the artifact itself
        # (q8/g64 weights, fp16 activations; 43.8/42.1 vs 17.4 tok/s).
        "Qwen3.6-27B-MTPLX-Optimized-Quality-FP16",
        # 9B (6-bit) promoted 2026-07-07 with the 6-bit hexpack split-K
        # kernels: live ABBA MTP D3 110/102 vs sustained 90/69 tok/s.
        "Qwen3.5-9B-MTPLX-Optimized-Speed",
        "Qwen3.5-9B-MTPLX-Optimized-Speed-FP16",
    ):
        model_dir = tmp_path / dir_name
        model_dir.mkdir()
        payload = _serve_dry_run_payload_for_model(monkeypatch, capsys, model_dir)
        assert payload["profile"] == "turbo", dir_name
        assert "--profile turbo" in payload["server_command"], dir_name


def test_serve_default_profile_untouched_off_the_turbo_allowlist(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    for dir_name in (
        # 35B MoE: experts bypass the NAX patch; stays sustained.
        "Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
        # Unrecognized third-party artifact.
        "example-model",
    ):
        model_dir = tmp_path / dir_name
        model_dir.mkdir()
        payload = _serve_dry_run_payload_for_model(monkeypatch, capsys, model_dir)
        assert payload["profile"] == "sustained", dir_name


def test_serve_explicit_profile_beats_turbo_default(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    model_dir = tmp_path / "Qwen3.6-27B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    payload = _serve_dry_run_payload_for_model(
        monkeypatch, capsys, model_dir, extra_args=("--profile", "sustained")
    )
    assert payload["profile"] == "sustained"


def test_serve_forwards_retrieval_flags_to_the_server_command(
    monkeypatch, tmp_path, capsys
):
    """The server runs as a subprocess with a rebuilt argv; a retrieval flag
    that is not forwarded silently configures nothing."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    model_dir = tmp_path / "example-model"
    model_dir.mkdir()
    payload = _serve_dry_run_payload_for_model(
        monkeypatch,
        capsys,
        model_dir,
        extra_args=(
            "--embedding-model",
            "org/embed=e1",
            "--reranker-model",
            "org/rank",
            "--retrieval-trust-remote-code",
        ),
    )
    command = payload["server_command"]
    assert "--embedding-model org/embed=e1" in command
    assert "--reranker-model org/rank" in command
    assert "--retrieval-trust-remote-code" in command


def test_serve_no_auth_parses_and_forwards(monkeypatch, tmp_path, capsys):
    """`mtplx serve --no-auth` — the exact spelling the 2.5.4 notes promised
    (#235) — must parse at the public CLI and reach the server subprocess.
    It previously existed only on the server module's parser, so the promised
    command died with an argparse error before any handler ran."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    model_dir = tmp_path / "example-model"
    model_dir.mkdir()
    payload = _serve_dry_run_payload_for_model(
        monkeypatch, capsys, model_dir, extra_args=("--no-auth",)
    )
    assert "--no-auth" in payload["server_command"]


def test_serve_no_auth_still_requires_key_off_localhost(monkeypatch, tmp_path):
    """--no-auth is a localhost convenience only: a non-localhost bind without
    a key still refuses, exactly as the #235 close promised."""
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    args = build_parser().parse_args(
        ["serve", "--model", str(tmp_path), "--host", "0.0.0.0", "--no-auth", "--yes"]
    )
    assert public.cmd_serve_public(args) == 2


def test_serve_does_not_grant_remote_code_trust_by_default(
    monkeypatch, tmp_path, capsys
):
    """Configuring an embedder must not quietly grant checkpoint code execution."""
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    model_dir = tmp_path / "example-model"
    model_dir.mkdir()
    payload = _serve_dry_run_payload_for_model(
        monkeypatch,
        capsys,
        model_dir,
        extra_args=("--embedding-model", "org/embed"),
    )
    assert "--retrieval-trust-remote-code" not in payload["server_command"]


def test_start_opencode_dry_run_uses_step_descriptor_defaults(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.setenv(
        "MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(tmp_path / "default.dat")
    )
    step_model = tmp_path / "Step-3.7-Flash-MTPLX-step3p5"
    step_model.mkdir()

    class FakeInspection:
        def to_dict(self) -> dict[str, object]:
            return {
                "model_dir": str(step_model),
                "recommended_backend": "step3p5_mtp",
                "runtime_compatibility": "native-contract-gated",
                "compatibility": {"can_run": True, "exit_code": 0},
            }

    monkeypatch.setattr(public, "inspect_model", lambda _model: FakeInspection())

    code = main(
        [
            "start",
            "opencode",
            "--dry-run",
            "--json",
            "--model",
            str(step_model),
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    command = payload["opencode"]["server_command"]
    assert code == 0
    assert payload["opencode"]["chat_template_profile"] == "tokenizer"
    assert "--chat-template-profile tokenizer" in command
    assert "--chat-template-profile local_qwen36" not in command
    assert "--reasoning auto" in command
    assert "--reasoning-parser step3p5" in command
    assert "--reasoning-effort low" in command
    assert payload["generation_mode"] == "mtp"


def test_start_dry_run_uses_gemma_defaults_even_when_gate_reports_error(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.setenv(
        "MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(tmp_path / "default.dat")
    )
    gemma_model = tmp_path / "gemma-4-31B-it-assistant-google-q6-g64-mlx"
    gemma_model.mkdir()
    inspection = {
        "model_dir": str(gemma_model),
        "recommended_backend": "gemma4_assistant",
        "runtime_compatibility": "native-contract-gated",
        "compatibility": {"can_run": False, "exit_code": 1},
    }
    monkeypatch.setattr(
        public, "_model_gate", lambda *_args, **_kwargs: (inspection, 1)
    )

    code = main(
        [
            "start",
            "opencode",
            "--dry-run",
            "--json",
            "--model",
            str(gemma_model),
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    command = payload["opencode"]["server_command"]
    assert code == 0
    assert payload["opencode"]["chat_template_profile"] == "tokenizer"
    assert "--chat-template-profile tokenizer" in command
    assert "--chat-template-profile local_qwen36" not in command
    assert "--top-k 64" in command
    assert "--reasoning auto" in command
    assert "--reasoning-parser gemma4" in command


def test_start_opencode_dry_run_json_can_expose_adaptive_ev_policy(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.setenv(
        "MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(tmp_path / "default.dat")
    )

    code = main(
        [
            "start",
            "opencode",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--yes",
            "--adaptive-policy",
            "expected_value",
            "--adaptive-ev-warmup-full-depth-cycles",
            "5",
            "--adaptive-ev-exploration-interval",
            "17",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    command = payload["opencode"]["server_command"]
    assert "--adaptive-policy expected_value" in command
    assert "--adaptive-ev-warmup-full-depth-cycles 5" in command
    assert "--adaptive-ev-exploration-interval 17" in command


def test_start_swival_dry_run_json_emits_generic_provider_command(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    code = main(
        [
            "start",
            "swival",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "swival"
    assert payload["swival"]["base_url"] == "http://127.0.0.1:18084"
    assert payload["swival"]["api_base_url"] == "http://127.0.0.1:18084/v1"
    assert payload["swival"]["no_hidden_max_tokens"] is True
    argv = payload["swival"]["command_argv"]
    assert argv[:3] == ["swival", "--provider", "generic"]
    assert "--base-url" in argv
    assert "http://127.0.0.1:18084" in argv
    assert "--max-context-tokens" in argv
    assert "maxTokens" not in json.dumps(payload["swival"])


def test_start_hermes_dry_run_json_matches_native_agent_lane(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("HOME", str(tmp_path))

    code = main(
        [
            "start",
            "hermes",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "hermes"
    assert payload["terminal_chat"] is False
    assert payload["hermes"]["server_url"] == "http://127.0.0.1:18085"
    assert payload["hermes"]["api_base_url"] == "http://127.0.0.1:18085/v1"
    assert payload["hermes"]["profile_name"] == "mtplx"
    assert payload["hermes"]["profile_path"] == str(
        tmp_path / ".hermes" / "profiles" / "mtplx"
    )
    assert payload["hermes"]["toolsets"] == [
        "terminal",
        "file",
        "web",
        "browser",
        "messaging",
    ]
    assert payload["hermes"]["launch_command"] == (
        "hermes -p mtplx chat --model example "
        "--toolsets terminal,file,web,browser,messaging --yolo --source mtplx-cli"
    )
    command = payload["hermes"]["server_command"]
    assert "--scheduler-mode serial" in command
    assert "--batching-preset latency" in command
    assert "--decode-batch-max" not in command
    assert "--batch-wait-ms" not in command
    assert "--prefill-chunk-tokens 2048" in command
    assert "--ssd-session-cache on" in command
    assert "--ssd-session-cache-max-size 100GB" in command
    assert "--ssd-session-cache-min-prefix-tokens 512" in command
    assert "--temperature 0.6" in command
    assert "--top-p 1.0" in command
    # Draft sampler is family/stamp-owned; the serve correction chain
    # resolves it at launch, so the hermes writer emits no draft flags.
    assert "--draft-temperature" not in command
    assert "--draft-top-p" not in command
    assert "--draft-top-k" not in command
    assert "--tool-prompt-mode hybrid" in command
    assert "--chat-template-profile local_qwen36" in command
    assert "--adaptive-policy expected_value" in command
    assert "--reasoning auto" in command
    assert payload["hermes"]["api_key"] == "mtplx-local"
    assert not (tmp_path / ".hermes" / "profiles" / "mtplx" / "config.yaml").exists()


def test_start_hermes_live_path_writes_profile_and_handoff(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("HOME", str(tmp_path))

    class NonInteractiveStdin:
        def isatty(self):
            return False

    class FakeInspection:
        def to_dict(self):
            return {
                "model_dir": "models/example",
                "compatibility": {"can_run": True, "exit_code": 0},
                "context_window": 262144,
            }

    captured: dict[str, object] = {}
    monkeypatch.setattr(public.sys, "stdin", NonInteractiveStdin())
    monkeypatch.setattr(
        public,
        "_quickstart_resolve_model",
        lambda *args, **kwargs: ("models/example", {}),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda *args, **kwargs: (FakeInspection().to_dict(), None),
    )

    def fake_serve(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(public, "cmd_serve_public", fake_serve)

    code = main(["start", "hermes", "--model", "models/example", "--yes"])

    assert code == 0
    serve_args = captured["args"]
    assert serve_args.quickstart_hermes is True
    assert serve_args.port == 18085
    assert serve_args.api_key == "mtplx-local"
    assert serve_args.reasoning == "auto"
    assert serve_args.preserve_thinking == "auto"
    assert serve_args.scheduler_mode == "serial"
    assert serve_args.batching_preset == "latency"
    assert serve_args.prefill_chunk_tokens == 2048
    assert serve_args.ssd_session_cache == "on"
    assert "OPENAI_API_KEY" not in serve_args.hermes_launch_command
    assert "--source mtplx-cli" in serve_args.hermes_launch_command
    profile_dir = tmp_path / ".hermes" / "profiles" / "mtplx"
    config_text = (profile_dir / "config.yaml").read_text(encoding="utf-8")
    env_text = (profile_dir / ".env").read_text(encoding="utf-8")
    assert "provider: custom" in config_text
    assert "toolsets:" in config_text
    assert 'HERMES_MODEL="example"' in env_text
    assert 'OPENAI_API_KEY="mtplx-local"' in env_text
    assert "Hermes will open automatically" in capsys.readouterr().out


def test_terminal_quickstart_max_uses_verified_max_session(monkeypatch):
    calls: list[str] = []

    class FakeMaxSession:
        def __init__(self, **_kwargs):
            self.thermal = {"enabled": True}

        def start(self):
            calls.append("start")
            return True

        def stop(self):
            calls.append("stop")
            self.thermal["restore"] = {"ok": True}
            return {"ok": True}

    fake_runtime = ModuleType("mtplx.runtime")
    fake_runtime.load = lambda *a, **kw: SimpleNamespace(tokenizer=object())
    fake_draft = ModuleType("mtplx.draft_lm_head")
    fake_draft._install_draft_lm_head = lambda *a, **kw: {"installed": True}

    monkeypatch.setitem(sys.modules, "mtplx.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "mtplx.draft_lm_head", fake_draft)
    monkeypatch.setattr("mtplx.thermal.MaxSession", FakeMaxSession)
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    monkeypatch.setattr(
        public,
        "_quickstart_generate",
        lambda **kw: {"text": "ok", "streamed": True, "validations": [], "stats": {}},
    )

    args = SimpleNamespace(
        profile="performance-cold",
        max=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        prompt="hello",
        show_stats=False,
    )

    rc = public._quickstart_run_terminal_chat(
        args, runtime_model="/tmp/model", inspection={}
    )

    assert rc == 0
    assert calls == ["start", "stop"]


def test_depth_sweep_native60_keeps_model_runtime_env_overrides(monkeypatch):
    observed: dict[str, str | None] = {}
    fake_runner = ModuleType("mtplx.benchmarks.runners.mtp_depth_sweep")

    def fake_run_mtp_depth_sweep(*_args, **kwargs):
        observed["lazy_verify"] = os.environ.get("MTPLX_LAZY_VERIFY_LOGITS")
        observed["batch_target"] = os.environ.get("MTPLX_BATCH_TARGET_ARRAYS")
        observed["temperature"] = str(kwargs["temperature"])
        observed["top_p"] = str(kwargs["top_p"])
        observed["top_k"] = str(kwargs["top_k"])
        return {"depths": []}

    fake_runner.run_mtp_depth_sweep = fake_run_mtp_depth_sweep
    monkeypatch.setitem(
        sys.modules,
        "mtplx.benchmarks.runners.mtp_depth_sweep",
        fake_runner,
    )
    monkeypatch.delenv("MTPLX_LAZY_VERIFY_LOGITS", raising=False)
    monkeypatch.delenv("MTPLX_BATCH_TARGET_ARRAYS", raising=False)

    result = public._depth_sweep_native60(
        model="/tmp/model",
        prompt_suite="/tmp/prompts.jsonl",
        depths="1",
        max_tokens=None,
        limit=1,
        seed=0,
        temperature=0.7,
        top_p=1.0,
        top_k=13,
        runtime_env={
            "MTPLX_LAZY_VERIFY_LOGITS": "0",
            "MTPLX_BATCH_TARGET_ARRAYS": "0",
        },
    )

    assert result["depths"] == []
    # The sweep harness now records the serve-path memory preflight (#F7);
    # the receipt is additive to the runner's payload.
    assert result["memory_preflight"]["entry"] == "bench.depth_sweep"
    assert observed["temperature"] == "0.7"
    assert observed["top_p"] == "1.0"
    assert observed["top_k"] == "13"
    assert observed["lazy_verify"] == "0"
    assert observed["batch_target"] == "0"
    assert os.environ.get("MTPLX_LAZY_VERIFY_LOGITS") is None
    assert os.environ.get("MTPLX_BATCH_TARGET_ARRAYS") is None


def test_one_shot_max_uses_verified_max_session(monkeypatch):
    calls: list[str] = []

    class FakeMaxSession:
        def __init__(self, **_kwargs):
            self.thermal = {"enabled": True}

        def start(self):
            calls.append("start")
            return True

        def stop(self):
            calls.append("stop")
            self.thermal["restore"] = {"ok": True}
            return {"ok": True}

    fake_runtime = ModuleType("mtplx.runtime")
    fake_runtime.load = lambda *a, **kw: SimpleNamespace(tokenizer=object())
    fake_schema = ModuleType("mtplx.benchmarks.schema")
    fake_schema.PromptCase = lambda **kw: SimpleNamespace(**kw)
    fake_schema.encode_prompt_case = lambda *a, **kw: [1, 2, 3]
    fake_generation = ModuleType("mtplx.generation")
    fake_generation.generate_mtpk = lambda *a, **kw: SimpleNamespace(
        text="ok",
        tokens=[1],
        stats=SimpleNamespace(
            generated_tokens=1, tok_s=1.0, verify_time_s=0.0, verify_calls=0
        ),
    )
    fake_generation.generate_ar = fake_generation.generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "mtplx.benchmarks.schema", fake_schema)
    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)
    monkeypatch.setattr("mtplx.thermal.MaxSession", FakeMaxSession)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: ("/tmp/model", None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda runtime_model, *, unsafe_force_unverified, yes: ({}, None),
    )

    args = SimpleNamespace(
        prompt="hello",
        prompt_arg=None,
        model="/tmp/model",
        cache_dir=None,
        unsafe_force_unverified=False,
        yes=True,
        profile="performance-cold",
        max=True,
        system=None,
        max_tokens=8,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        seed=0,
        expect_python=False,
    )

    code, payload, _validations = public._generate_one_shot_public(args, command="run")

    assert code == 0
    assert payload["thermal"]["restore"]["ok"] is True
    assert calls == ["start", "stop"]


@pytest.mark.parametrize("command", ["run", "chat"])
def test_laguna_one_shot_uses_target_generation_defaults(monkeypatch, command):
    # One-shot really applies the sustained profile, which writes MTPLX_* keys
    # into process env for the remaining CLI lifetime; without isolation those
    # keys leak into later in-process tests (cache_state env-driven configure).
    monkeypatch.setattr(os, "environ", os.environ.copy())
    observed: dict[str, object] = {}

    fake_runtime = ModuleType("mtplx.runtime")

    def fake_load(*_args, **kwargs):
        observed["load_mtp"] = kwargs["mtp"]
        return SimpleNamespace(tokenizer=object())

    fake_runtime.load = fake_load
    fake_schema = ModuleType("mtplx.benchmarks.schema")
    fake_schema.PromptCase = lambda **kwargs: SimpleNamespace(**kwargs)

    def fake_encode(*_args, **kwargs):
        observed["enable_thinking"] = kwargs["enable_thinking"]
        return [1, 2, 3]

    fake_schema.encode_prompt_case = fake_encode
    fake_generation = ModuleType("mtplx.generation")

    def fake_generate_ar(*_args, **kwargs):
        observed["sampler"] = kwargs["sampler"]
        return SimpleNamespace(
            text="ok",
            tokens=[1],
            stats=SimpleNamespace(generated_tokens=1, tok_s=1.0),
        )

    fake_generation.generate_ar = fake_generate_ar
    fake_generation.generate_mtpk = lambda *_a, **_kw: pytest.fail(
        "Laguna one-shot reached MTP generation"
    )
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "mtplx.runtime", fake_runtime)
    monkeypatch.setitem(sys.modules, "mtplx.benchmarks.schema", fake_schema)
    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: ("/tmp/laguna", None),
    )
    inspection = {
        "recommended_backend": "laguna_ar",
        "compatibility": {
            "runtime_compatibility": "native-ar-only",
            "recommended_backend": "laguna_ar",
            "can_run": True,
        },
    }
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda *_args, **_kwargs: (inspection, None),
    )
    args = SimpleNamespace(
        prompt="hello",
        prompt_arg=None,
        model="/tmp/laguna",
        cache_dir=None,
        unsafe_force_unverified=False,
        yes=True,
        profile="sustained",
        max=False,
        system=None,
        max_tokens=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        seed=0,
        expect_python=False,
        generation_mode="ar",
        no_mtp=True,
        load_mtp=True,
        reasoning=None,
        reasoning_parser="qwen3",
        _cli_flags=set(),
    )

    code, payload, _validations = public._generate_one_shot_public(
        args,
        command=command,
    )

    sampler = observed["sampler"]
    assert code == 0
    assert payload["stats"]["generation_mode"] == "ar"
    assert payload["stats"]["max_tokens"] == 32_768
    assert observed["load_mtp"] is False
    assert observed["enable_thinking"] is True
    assert sampler.temperature == 1.0
    assert sampler.top_p == 1.0
    assert sampler.top_k == 20


def test_laguna_backend_defaults_preserve_explicit_sampler_flags():
    args = SimpleNamespace(
        temperature=0.25,
        top_p=0.8,
        top_k=7,
        depth=3,
        max_tokens=None,
        max_response_tokens=None,
        reasoning=None,
        reasoning_parser="qwen3",
        _cli_flags={"temperature", "top-p", "top-k"},
    )

    public._apply_backend_serve_defaults(
        args,
        {"recommended_backend": "laguna_ar"},
    )

    assert args.temperature == 0.25
    assert args.top_p == 0.8
    assert args.top_k == 7
    assert args.max_tokens == 32_768
    assert args.max_response_tokens == 32_768
    assert public._pi_sampler_top_p(args) == 0.8

    from mtplx.backends.descriptors import LAGUNA_AR_DESCRIPTOR

    assert LAGUNA_AR_DESCRIPTOR.context_window_policy.default == 32_768
    assert LAGUNA_AR_DESCRIPTOR.context_window_policy.maximum == 1_048_576
    assert LAGUNA_AR_DESCRIPTOR.to_dict()["default_max_response_tokens"] == 32_768


def test_laguna_opencode_payload_uses_native_tools_and_32k_context(monkeypatch):
    args = SimpleNamespace(
        host="127.0.0.1",
        port=8000,
        model="mlx-community/Laguna-S-2.1-oQ4e",
        profile="sustained",
        api_key=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        depth=3,
        no_mtp=True,
        generation_mode="ar",
        max_response_tokens=None,
        reasoning=None,
        reasoning_parser="qwen3",
        tool_prompt_mode="hybrid",
        chat_template_profile="local_qwen36",
        _cli_flags=set(),
    )
    inspection = {
        "recommended_backend": "laguna_ar",
        "compatibility": {"recommended_backend": "laguna_ar"},
    }
    monkeypatch.setattr(
        "mtplx.opencode.detect_opencode_desktop",
        lambda: {"installed": False},
    )

    public._apply_backend_serve_defaults(args, inspection)
    payload = public._quickstart_opencode_payload(args, inspection=inspection)

    assert args.tool_prompt_mode == "native"
    assert args.chat_template_profile == "tokenizer"
    assert public._inspection_context_window(inspection, args=args) == 32_768
    assert payload["context_window"] == 32_768
    assert payload["output_limit"] == 32_768
    assert payload["tool_prompt_mode"] == "native"
    assert "--tool-prompt-mode native" in payload["server_command"]
    assert "--context-window 32768" in payload["server_command"]

    args.context_window = 65_536
    expanded = public._quickstart_opencode_payload(args, inspection=inspection)

    assert expanded["context_window"] == 65_536
    assert expanded["output_limit"] == 32_768
    assert "--context-window 65536" in expanded["server_command"]
    assert "--max-response-tokens 32768" in expanded["server_command"]


@pytest.mark.parametrize(
    ("flags", "updates"),
    (
        ({"chat-template-profile"}, {"chat_template_profile": "local_qwen36"}),
        ({"chat-template-path"}, {"chat_template_path": "/tmp/qwen.jinja"}),
    ),
)
def test_laguna_rejects_conflicting_chat_template_overrides(flags, updates):
    values = {
        "reasoning": None,
        "reasoning_parser": "qwen3",
        "reasoning_effort": None,
        "tool_prompt_mode": "hybrid",
        "chat_template_profile": "tokenizer",
        "chat_template_path": None,
        "_cli_flags": flags,
    }
    values.update(updates)
    args = SimpleNamespace(**values)

    with pytest.raises(ValueError, match="requires.*tokenizer chat template"):
        public._apply_backend_serve_defaults(
            args,
            {"recommended_backend": "laguna_ar"},
        )


def test_serve_require_max_fans_fails_closed_before_child_launch(monkeypatch, capsys):
    calls: list[str] = []

    class FakeMaxSession:
        def __init__(self, **_kwargs):
            self.thermal = {
                "verified": {
                    "ok": False,
                    "message": "actual fan RPM did not ramp",
                    "actionable": "open ThermalForge",
                }
            }

        def start(self):
            calls.append("start")
            return False

        def stop(self):
            calls.append("stop")
            return {"ok": True}

    monkeypatch.setattr("mtplx.thermal.MaxSession", FakeMaxSession)
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(public, "_port_is_busy", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: ("/tmp/model", None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda runtime_model, *, unsafe_force_unverified, yes: ({}, None),
    )
    monkeypatch.setattr(public, "_model_draft_lm_head_spec", lambda *_args: None)
    monkeypatch.setattr(public, "_model_draft_sampler_spec", lambda *_args: None)
    monkeypatch.setattr(public, "os", public.os)

    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            "/tmp/model",
            "--yes",
            "--max",
            "--require-max-fans",
            "--app-launch-id",
            "native-123",
        ]
    )

    code = public.cmd_serve_public(args)
    captured = capsys.readouterr().err

    assert code == 2
    assert calls == ["start"]
    assert "strict startup requested" in captured
    assert "refusing to load the model" in captured


def test_start_parser_accepts_target_choices():
    """Parser must accept the new `web` and `cli` target literals."""
    parser = build_parser()
    for target in (
        "web",
        "cli",
        "openwebui",
        "open-webui",
        "terminal",
        "pi",
        "pie",
        "opencode",
        "open-code",
        "oc",
        "swival",
        "sv",
        "hermes",
        "hermes-agent",
        "dashboard",
        "live-dashboard",
        "live",
    ):
        args = parser.parse_args(["start", target, "--dry-run"])
        assert args.target == target
    # Default target (no positional) is now None — the absence of an explicit
    # target is what tells the handler to run the interactive onboarding flow
    # (or fall back to "web" when non-interactive). The `--fresh` flag is also
    # accepted for forcing the full onboarding.
    args_default = parser.parse_args(["start", "--dry-run"])
    assert args_default.target is None
    args_fresh = parser.parse_args(["start", "--fresh", "--dry-run"])
    assert args_fresh.fresh is True
    args_no_mtp = parser.parse_args(["start", "cli", "--no-mtp", "--dry-run"])
    assert args_no_mtp.no_mtp is True
    args_mtp = parser.parse_args(["start", "cli", "--mtp", "--dry-run"])
    assert args_mtp.no_mtp is False


def test_start_help_mentions_target_only_ar_mtp_controls(capsys):
    code = main(["help", "start"])

    captured = capsys.readouterr().out
    assert code == 0
    assert "--no-mtp" in captured
    assert "target-only AR generation" in captured
    assert "/mtp off" in captured
    assert "/stats" in captured


def test_mtp_toggle_flags_parse_on_public_generation_surfaces():
    parser = build_parser()

    cases = [
        ["start", "cli", "--no-mtp", "--dry-run"],
        ["quickstart", "--no-mtp"],
        ["serve", "--no-mtp"],
        ["ask", "hello", "--no-mtp"],
        ["run", "hello", "--no-mtp"],
        ["chat", "--prompt", "hello", "--no-mtp"],
    ]
    for argv in cases:
        assert parser.parse_args(argv).no_mtp is True

    mtp_cases = [
        ["start", "cli", "--mtp", "--dry-run"],
        ["quickstart", "--mtp"],
        ["serve", "--mtp"],
        ["ask", "hello", "--mtp"],
        ["run", "hello", "--mtp"],
        ["chat", "--prompt", "hello", "--mtp"],
    ]
    for argv in mtp_cases:
        assert parser.parse_args(argv).no_mtp is False


def test_serve_parser_accepts_top_k_aliases():
    parser = build_parser()

    for flag in ("--top-k", "--default-top-k"):
        args = parser.parse_args(["serve", "--model", "/tmp/model", flag, "20"])
        assert args.top_k == 20


def test_serve_parser_accepts_top_p_aliases():
    parser = build_parser()

    for flag in ("--top-p", "--default-top-p"):
        args = parser.parse_args(["serve", "--model", "/tmp/model", flag, "1.0"])
        assert args.top_p == 1.0


def test_serve_parser_accepts_draft_sampler_flags():
    parser = build_parser()

    args = parser.parse_args(
        [
            "serve",
            "--model",
            "/tmp/model",
            "--draft-temperature",
            "0.6",
            "--draft-top-p",
            "0.95",
            "--draft-top-k",
            "20",
        ]
    )

    assert args.draft_temperature == 0.6
    assert args.draft_top_p == 0.95
    assert args.draft_top_k == 20


def test_sustained_ignores_performance_cold_draft_contract():
    inspection = {
        "runtime_contract": {
            "recommended_profile": "performance-cold",
            "mtp_depth_max": 1,
            "recommended_draft_lm_head": {
                "bits": 3,
                "group_size": 64,
                "mode": "affine",
            },
            "recommended_draft_sampler": {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 20,
            },
        }
    }

    sustained = public.get_profile("sustained")
    burst = public.get_profile("performance-cold")

    assert public._model_draft_lm_head_spec(inspection, sustained) == {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
    }
    assert public._model_draft_sampler_spec(inspection, sustained) is None
    assert public._model_contract_depth(inspection, profile=sustained, fallback=3) == 3
    assert public._model_draft_lm_head_spec(inspection, burst) == {
        "bits": 3,
        "group_size": 64,
        "mode": "affine",
    }
    assert public._model_draft_sampler_spec(inspection, burst) == {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 20,
    }
    assert public._model_contract_depth(inspection, profile=burst, fallback=3) == 1


def test_serve_sustained_uses_profile_draft_head_and_explicit_agent_sampler(
    monkeypatch,
):
    calls = {}
    contract = {
        "recommended_profile": "performance-cold",
        "mtp_depth_max": 1,
        "recommended_draft_lm_head": {
            "bits": 3,
            "group_size": 64,
            "mode": "affine",
        },
        "recommended_draft_sampler": {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 20,
        },
    }

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {
                "compatibility": {
                    "tier": "verified",
                    "can_run": True,
                    "exit_code": 0,
                    "runtime_contract": contract,
                }
            },
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)

    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            "/tmp/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "--profile",
            "sustained",
            "--draft-temperature",
            "0.6",
            "--draft-top-p",
            "0.95",
            "--draft-top-k",
            "20",
            "--reasoning",
            "on",
            "--yes",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    cmd = calls["cmd"]
    assert exc.value.code == 0
    assert cmd[cmd.index("--depth") + 1] == "3"
    assert cmd[cmd.index("--draft-lm-head-bits") + 1] == "4"
    assert cmd[cmd.index("--draft-lm-head-group-size") + 1] == "64"
    assert cmd[cmd.index("--draft-lm-head-mode") + 1] == "affine"
    assert cmd[cmd.index("--draft-temperature") + 1] == "0.6"
    assert cmd[cmd.index("--draft-top-p") + 1] == "0.95"
    assert cmd[cmd.index("--draft-top-k") + 1] == "20"


def test_serve_parser_accepts_bridge_prompt_flags():
    parser = build_parser()

    args = parser.parse_args(
        [
            "serve",
            "--model",
            "/tmp/model",
            "--tool-prompt-mode",
            "native",
            "--chat-template-profile",
            "froggeric_v19",
        ]
    )

    assert args.tool_prompt_mode == "native"
    assert args.chat_template_profile == "froggeric_v19"

    gemma_args = parser.parse_args(
        [
            "serve",
            "--model",
            "/tmp/Gemma4-MTPLX-Optimized-Speed",
            "--temperature",
            "1.0",
            "--chat-template-profile",
            "tokenizer",
        ]
    )

    assert gemma_args.temperature == 1.0
    assert gemma_args.chat_template_profile == "tokenizer"


def test_public_depth_validation_allows_gemma4_contract_depth():
    lines: list[str] = []
    args = SimpleNamespace(
        depth=6,
        model="/tmp/Gemma4-MTPLX-Optimized-Speed",
        model_id=DEFAULT_PUBLIC_MODEL_ID,
    )

    assert public._validate_public_depth(args, printer=lines.append) is None
    assert args.depth == 6
    assert lines == []

    qwen_args = SimpleNamespace(
        depth=6,
        model="/tmp/qwen3-optimized",
        model_id=DEFAULT_PUBLIC_MODEL_ID,
    )

    assert public._validate_public_depth(qwen_args, printer=lines.append) == 2
    assert any("1 and 3" in line for line in lines)


def test_cli_response_cap_defaults_to_remaining_context():
    parser = build_parser()

    quickstart = parser.parse_args(["start", "cli", "--dry-run"])
    ask = parser.parse_args(["ask", "hello"])
    run = parser.parse_args(["run", "hello"])
    chat = parser.parse_args(["chat", "--prompt", "hello"])

    assert quickstart.max_tokens is None
    assert ask.max_tokens is None
    assert run.max_tokens is None
    assert chat.max_tokens is None


def test_cli_reasoning_flags_parse_without_being_chat_text():
    parser = build_parser()

    quickstart = parser.parse_args(["start", "cli", "--reasoning", "on"])
    run = parser.parse_args(["run", "hello", "--reasoning", "off"])
    serve = parser.parse_args(["serve", "--reasoning", "auto"])

    assert quickstart.reasoning == "on"
    assert run.reasoning == "off"
    assert serve.reasoning == "auto"


def test_start_missing_model_suggests_download(monkeypatch, capsys):
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (None, {"detail": "not cached"}),
    )
    args = SimpleNamespace(
        command="start",
        model="mtplx/example",
        cache_dir="/tmp/mtplx-models",
        download=False,
        dry_run=False,
        json=False,
        yes=True,
        prompt=None,
        profile="stable",
        show_stats=True,
        unsafe_force_unverified=False,
    )

    code = public.cmd_quickstart_public(args)

    captured = capsys.readouterr().out
    assert code == 1
    assert "MTPLX start" in captured
    assert "model is not available locally" in captured
    assert "detail: not cached" in captured
    assert "try: mtplx start --download" in captured
    assert "try: mtplx start --model /path/to/model" in captured


def test_quickstart_public_quality_alias_missing_cache_is_not_no_mtp(tmp_path, capsys):
    cache_dir = tmp_path / "cache"

    code = main(
        [
            "quickstart",
            "--max",
            "--model",
            "Qwen3.6-27B-MTPLX-Optimized-Quality",
            "--cache-dir",
            str(cache_dir),
            "--yes",
            "--warmup-tokens",
            "0",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 1
    assert "error: model is not available locally" in captured
    assert "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Quality" in captured
    assert "try: mtplx quickstart --download" in captured
    assert "error: model cannot run with MTPLX" not in captured
    assert "tier: no-MTP" not in captured


def test_quickstart_model_id_quality_without_model_loads_quality(tmp_path, capsys):
    code = main(
        [
            "quickstart",
            "--max",
            "--model-id",
            QUALITY_HF_MODEL_ID,
            "--cache-dir",
            str(tmp_path / "cache"),
            "--yes",
            "--warmup-tokens",
            "0",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 1
    assert f"model: {QUALITY_HF_MODEL_ID}" in captured
    assert f"detail: Model {QUALITY_HF_MODEL_ID} is not cached" in captured
    assert "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed" not in captured


def test_serve_model_id_quality_without_model_loads_quality(tmp_path, capsys):
    code = main(
        [
            "serve",
            "--max",
            "--model-id",
            QUALITY_HF_MODEL_ID,
            "--cache-dir",
            str(tmp_path / "cache"),
            "--yes",
            "--warmup-tokens",
            "0",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 1
    assert f"model: {QUALITY_HF_MODEL_ID}" in captured
    assert f"detail: Model {QUALITY_HF_MODEL_ID} is not cached" in captured
    assert "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed" not in captured


def test_quickstart_default_missing_cache_is_not_legacy_models_path(
    monkeypatch, tmp_path, capsys
):
    _pin_big_apple_silicon(monkeypatch)
    code = main(
        [
            "quickstart",
            "--profile",
            "sustained",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--yes",
            "--warmup-tokens",
            "0",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 1
    assert DEFAULT_HF_MODEL_ID in captured
    assert "models/Qwen3.8-27B-MTPLX-Optimized-Speed" not in captured
    assert "models/Qwen3.6-27B-MTPLX-Optimized-Speed" not in captured
    assert "error: model cannot run with MTPLX" not in captured
    assert "tier: no-MTP" not in captured


def test_tune_default_dry_run_is_not_legacy_models_path(monkeypatch, tmp_path, capsys):
    _pin_big_apple_silicon(monkeypatch)
    code = main(["tune", "--dry-run", "--json", "--cache-dir", str(tmp_path / "cache")])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == DEFAULT_HF_MODEL_ID
    first_command = payload["candidates"][0]["command"]
    assert "--model" in first_command
    assert first_command[first_command.index("--model") + 1] == payload["model"]
    assert first_command[first_command.index("--model") + 1] != (
        "models/Qwen3.6-27B-MTPLX-Optimized-Speed"
    )


def test_quickstart_short_reply_reports_decode_tps():
    line = public._quickstart_stats_line(
        {
            "profile": {"name": "performance-cold"},
            "stats": {
                "generated_tokens": 10,
                "end_to_end_tok_s": 28.48,
                "elapsed_s": 0.351,
                "prompt_eval_time_s": 0.0,
                "verify_ms_per_call": 54.8,
            },
        }
    )

    assert "10 tokens in 0.35s decode" in line
    assert "28.49 tok/s" in line
    assert "54.8 ms/verify" in line


def test_quickstart_short_reply_prefers_decode_tps_and_labels_live_window():
    line = public._quickstart_stats_line(
        {
            "profile": {"name": "performance-cold"},
            "stats": {
                "generated_tokens": 10,
                "stream_tok_s": 42.0,
                "decode_tok_s": 28.48,
                "end_to_end_tok_s": 25.0,
                "decode_elapsed_s": 0.351,
                "verify_ms_per_call": 54.8,
                "verify_calls": 5,
                "accepted_by_depth": [4, 3, 2],
                "ttft_s": 0.08,
            },
        }
    )

    assert "28.48 tok/s" in line
    assert "live_window=42.00" in line
    assert "total=25.00" in line
    assert "5 verify calls" in line
    assert "accept=[4, 3, 2]" in line
    assert "ttft=0.08s" in line
    assert "short sample" not in line


def test_quickstart_tiny_reply_prefers_decode_over_noisy_stream_window():
    line = public._quickstart_stats_line(
        {
            "profile": {"name": "performance-cold"},
            "stats": {
                "generated_tokens": 3,
                "stream_tok_s": 12.24,
                "decode_tok_s": 34.68,
                "end_to_end_tok_s": 8.73,
                "decode_elapsed_s": 0.09,
                "verify_ms_per_call": 55.8,
                "verify_calls": 1,
            },
        }
    )

    assert "34.68 tok/s" in line
    assert "live_window=12.24" in line
    assert "total=8.73" in line


def test_quickstart_long_reply_uses_decode_tps():
    line = public._quickstart_stats_line(
        {
            "profile": {"name": "performance-cold"},
            "stats": {
                "generated_tokens": 192,
                "end_to_end_tok_s": 48.0,
                "elapsed_s": 4.0,
                "prompt_eval_time_s": 0.2,
                "verify_ms_per_call": 60.2,
            },
        }
    )

    assert "192 tokens" in line
    assert "50.53 tok/s" in line
    assert "total=48.00" in line
    assert "60.2 ms/verify" in line


def test_quickstart_incremental_decoder_streams_word_boundaries():
    class TinyTokenizer:
        def decode(self, tokens, **_kwargs):
            return "".join(chr(token) for token in tokens)

    decoder = public._QuickstartIncrementalTokenDecoder(TinyTokenizer())

    assert decoder.feed([104, 101, 108, 108, 111]) == ""
    assert decoder.feed([32]) == "hello "
    assert decoder.feed([119, 111, 114, 108, 100]) == ""
    assert decoder.finish() == "world"


def test_quickstart_generation_default_uses_remaining_model_context(
    monkeypatch, tmp_path
):
    captured: dict[str, int] = {}

    class TinyTokenizer:
        model_max_length = 100

        def apply_chat_template(self, *_args, **kwargs):
            captured["enable_thinking"] = kwargs.get("enable_thinking")
            return list(range(12))

    fake_generation = ModuleType("mtplx.generation")

    def fake_generate_mtpk(*_args, **kwargs):
        captured["max_tokens"] = kwargs["max_tokens"]
        return SimpleNamespace(
            text="ok",
            stats=SimpleNamespace(
                generated_tokens=1,
                tok_s=1.0,
                elapsed_s=1.0,
                prompt_eval_time_s=0.0,
                verify_time_s=0.0,
                target_forward_time_s=1.0,
                repair_time_s=0.0,
                draft_time_s=0.0,
                verify_calls=0,
                accepted_by_depth=[],
                drafted_by_depth=[],
                correction_tokens=0,
                bonus_tokens=0,
            ),
        )

    fake_generation.generate_mtpk = fake_generate_mtpk
    fake_generation.generate_ar = fake_generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)

    rt = SimpleNamespace(tokenizer=TinyTokenizer(), model_path=tmp_path)
    payload = public._quickstart_generate(
        rt=rt,
        inspection={},
        profile=SimpleNamespace(to_dict=lambda: {"name": "stable"}),
        args=SimpleNamespace(
            system=None,
            max_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            depth=3,
            seed=0,
        ),
        prompt="hello",
        history=[],
        turn_index=0,
    )

    assert captured["max_tokens"] == 88
    assert captured["enable_thinking"] is True
    assert payload["stats"]["max_tokens"] == 88
    assert payload["stats"]["remaining_context_tokens"] == 88
    assert payload["stats"]["reasoning"] == "on"


def test_quickstart_generation_no_mtp_uses_ar(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    class TinyTokenizer:
        model_max_length = 100

        def apply_chat_template(self, *_args, **_kwargs):
            return list(range(12))

    def stats() -> SimpleNamespace:
        return SimpleNamespace(
            generated_tokens=1,
            tok_s=1.0,
            elapsed_s=1.0,
            prompt_eval_time_s=0.0,
            verify_time_s=0.0,
            target_forward_time_s=1.0,
            repair_time_s=0.0,
            draft_time_s=0.0,
            verify_calls=0,
            accepted_by_depth=[],
            drafted_by_depth=[],
            correction_tokens=0,
            bonus_tokens=0,
        )

    fake_generation = ModuleType("mtplx.generation")

    def fake_generate_ar(*_args, **kwargs):
        captured["mode"] = "ar"
        captured["max_tokens"] = kwargs["max_tokens"]
        return SimpleNamespace(text="ok", stats=stats())

    def fake_generate_mtpk(*_args, **_kwargs):  # pragma: no cover - must not be used
        captured["mode"] = "mtp"
        return SimpleNamespace(text="wrong", stats=stats())

    fake_generation.generate_ar = fake_generate_ar
    fake_generation.generate_mtpk = fake_generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)

    payload = public._quickstart_generate(
        rt=SimpleNamespace(tokenizer=TinyTokenizer(), model_path=tmp_path),
        inspection={},
        profile=SimpleNamespace(to_dict=lambda: {"name": "stable"}),
        args=SimpleNamespace(
            system=None,
            max_tokens=None,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            depth=3,
            seed=0,
            no_mtp=True,
        ),
        prompt="hello",
        history=[],
        turn_index=0,
    )

    assert captured == {"mode": "ar", "max_tokens": 88}
    assert payload["stats"]["generation_mode"] == "ar"
    assert payload["stats"]["mtp_depth"] == 0
    assert payload["stats"]["verify_calls"] == 0
    assert payload["stats"]["accepted_by_depth"] == []
    assert payload["stats"]["drafted_by_depth"] == []


def test_quickstart_mtp_slash_command_toggles_next_turn(capsys):
    args = SimpleNamespace(no_mtp=False)
    runtime = SimpleNamespace(mtp_enabled=True)

    assert (
        public._handle_quickstart_mtp_command(args, "/mtp status", runtime=runtime)
        is True
    )
    assert (
        public._handle_quickstart_mtp_command(args, "/mtp off", runtime=runtime) is True
    )
    assert args.no_mtp is True
    assert (
        public._handle_quickstart_mtp_command(args, "/mtp on", runtime=runtime) is True
    )
    assert args.no_mtp is False

    captured = capsys.readouterr().out
    assert "MTP: on" in captured
    assert "MTP: off for the next turn" in captured
    assert "MTP: on for the next turn" in captured


def test_quickstart_generation_reasoning_on_passes_enable_thinking(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    class TinyTokenizer:
        model_max_length = 64

        def apply_chat_template(self, *_args, **kwargs):
            captured["enable_thinking"] = kwargs.get("enable_thinking")
            return [1, 2, 3]

    fake_generation = ModuleType("mtplx.generation")
    fake_generation.generate_mtpk = lambda *_args, **_kwargs: SimpleNamespace(
        text="ok",
        stats=SimpleNamespace(
            generated_tokens=1,
            tok_s=1.0,
            elapsed_s=1.0,
            prompt_eval_time_s=0.0,
            verify_time_s=0.0,
            target_forward_time_s=1.0,
            repair_time_s=0.0,
            draft_time_s=0.0,
            verify_calls=0,
            accepted_by_depth=[],
            drafted_by_depth=[],
            correction_tokens=0,
            bonus_tokens=0,
        ),
    )
    fake_generation.generate_ar = fake_generation.generate_mtpk
    fake_sampling = ModuleType("mtplx.sampling")
    fake_sampling.SamplerConfig = lambda **kw: SimpleNamespace(**kw)

    monkeypatch.setitem(sys.modules, "mtplx.generation", fake_generation)
    monkeypatch.setitem(sys.modules, "mtplx.sampling", fake_sampling)

    public._quickstart_generate(
        rt=SimpleNamespace(tokenizer=TinyTokenizer(), model_path=tmp_path),
        inspection={},
        profile=SimpleNamespace(to_dict=lambda: {"name": "stable"}),
        args=SimpleNamespace(
            system=None,
            max_tokens=8,
            reasoning="on",
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            depth=3,
            seed=0,
        ),
        prompt="hello",
        history=[],
        turn_index=0,
    )

    assert captured["enable_thinking"] is True


def test_terminal_reasoning_command_updates_local_mode(capsys):
    args = SimpleNamespace(reasoning=None)

    assert public._handle_quickstart_reasoning_command(args, "--reasoning on") is True

    assert args.reasoning == "on"
    assert "Reasoning: on" in capsys.readouterr().out


def test_quickstart_openwebui_dry_run_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    code = main(
        [
            "start",
            "openwebui",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--port",
            "18012",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "openwebui"
    assert payload["terminal_chat"] is False
    assert payload["openwebui"]["server_url"] == "http://127.0.0.1:18012"
    assert payload["openwebui"]["base_url"] == "http://127.0.0.1:18012/v1"
    assert payload["openwebui"]["api_base_url"] == "http://127.0.0.1:18012/v1"
    assert payload["openwebui"]["chat_url"] == "http://127.0.0.1:18012/"
    assert (
        "Open chat UI: http://127.0.0.1:18012/"
        in payload["openwebui"]["openwebui_steps"]
    )
    assert (
        "OpenAI-compatible API base URL: http://127.0.0.1:18012/v1"
        in payload["openwebui"]["openwebui_steps"]
    )
    assert "--profile sustained" in payload["openwebui"]["server_command"]
    assert "--no-stats-footer" in payload["openwebui"]["server_command"]
    assert "--open-browser" in payload["openwebui"]["server_command"]


@pytest.mark.parametrize(
    ("target", "payload_key"),
    [
        ("openwebui", "openwebui"),
        ("pi", "pi"),
        ("opencode", "opencode"),
        ("swival", "swival"),
        ("hermes", "hermes"),
    ],
)
def test_start_client_dry_run_preserves_smart_fan_mode(
    target, payload_key, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(tmp_path / "pi" / "models.json"))
    monkeypatch.setenv("HOME", str(tmp_path))

    code = main(
        [
            "start",
            target,
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--yes",
            "--fan-mode",
            "smart",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    command = payload[payload_key]["server_command"]
    assert code == 0
    assert "--fan-mode smart" in command
    assert " --max " not in f" {command} "


def test_quickstart_pi_dry_run_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(tmp_path / "pi" / "models.json"))

    code = main(
        [
            "start",
            "pi",
            "--dry-run",
            "--json",
            "--model",
            "models/example",
            "--port",
            "18012",
            "--yes",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["target"] == "pi"
    assert payload["terminal_chat"] is False
    assert payload["pi"]["base_url"] == "http://127.0.0.1:18012/v1"
    assert payload["pi"]["model_ref"] == "mtplx/example"
    assert payload["pi"]["launch_command"] == "pi --model mtplx/example"
    assert payload["pi"]["provider"]["api"] == "openai-completions"
    assert payload["pi"]["provider"]["authHeader"] is True
    assert payload["pi"]["provider"]["headers"] == {"x-mtplx-client": "pi"}
    assert payload["pi"]["provider"]["compat"]["supportsDeveloperRole"] is False
    assert payload["pi"]["provider"]["compat"]["supportsReasoningEffort"] is True
    assert payload["pi"]["provider"]["compat"]["thinkingFormat"] == "qwen"
    assert payload["pi"]["provider"]["compat"]["maxTokensField"] == "max_tokens"
    assert payload["pi"]["provider"]["models"][0]["reasoning"] is True
    assert payload["pi"]["provider"]["models"][0]["thinkingLevelMap"] == {
        "minimal": None,
        "xhigh": "xhigh",
    }
    assert payload["pi"]["no_hidden_max_tokens"] is True
    assert payload["pi"]["provider"]["models"][0]["maxTokens"] == payload["pi"][
        "context_window"
    ]
    assert payload["pi"]["request_policy_extension_path"].endswith(
        "/extensions/mtplx-request-policy.ts"
    )
    assert "--api-key mtplx-local" in payload["pi"]["server_command"]
    assert "--default-top-p 0.95" in payload["pi"]["server_command"]
    # Draft sampler is family/stamp-owned; the old writer cross-wired the Pi
    # TARGET sampler into draft flags, pinning past the correction chain.
    assert "--draft-temperature" not in payload["pi"]["server_command"]
    assert "--draft-top-p" not in payload["pi"]["server_command"]
    assert "--draft-top-k" not in payload["pi"]["server_command"]
    # Pi rides the general auto resolution (2026-08-21): the family contract
    # owns reasoning history, replacing the 1.0.0-era Pi-only hard "off".
    assert "--preserve-thinking auto" in payload["pi"]["server_command"]


def test_start_pi_missing_cli_stops_before_model_check(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("MTPLX_CONFIG", str(tmp_path / "missing-config.toml"))

    class NonInteractiveStdin:
        def isatty(self):
            return False

    monkeypatch.setattr(public.sys, "stdin", NonInteractiveStdin())
    monkeypatch.setattr(
        public.shutil, "which", lambda name: None if name == "pi" else "/usr/bin/npm"
    )

    def fail_resolve(*_args, **_kwargs):
        raise AssertionError("Pi preflight must run before model resolution")

    monkeypatch.setattr(public, "_quickstart_resolve_model", fail_resolve)

    code = main(["start", "pi", "--model", "models/example", "--yes"])

    captured = capsys.readouterr().out
    assert code == 2
    assert "Pi is not installed" in captured
    assert "MTPLX has not loaded the model yet" in captured
    assert "npm install -g @earendil-works/pi-coding-agent" in captured
    assert "Then re-run: mtplx start pi" in captured
    assert "[1/4] Checking model" not in captured


def test_pi_models_config_merge_preserves_other_providers(tmp_path):
    from mtplx.pi import write_pi_models_config

    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "other": {
                        "baseUrl": "https://example.invalid/v1",
                        "api": "openai-completions",
                        "apiKey": "OTHER_KEY",
                        "models": [{"id": "other-model"}],
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = write_pi_models_config(
        base_url="http://127.0.0.1:18012/v1",
        model_id="mtplx-test-model",
        path=config_path,
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert result["config_path"] == str(config_path)
    assert payload["providers"]["other"]["models"][0]["id"] == "other-model"
    assert payload["providers"]["mtplx"]["baseUrl"] == "http://127.0.0.1:18012/v1"
    assert payload["providers"]["mtplx"]["headers"] == {"x-mtplx-client": "pi"}
    assert payload["providers"]["mtplx"]["models"][0]["id"] == "mtplx-test-model"
    assert payload["providers"]["mtplx"]["models"][0]["reasoning"] is True
    assert payload["providers"]["mtplx"]["models"][0]["maxTokens"] == 131072
    assert result["no_hidden_max_tokens"] is True
    extension_path = config_path.parent / "extensions" / "mtplx-request-policy.ts"
    assert result["request_policy_extension_path"] == str(extension_path)
    extension_source = extension_path.read_text(encoding="utf-8")
    assert 'delete request.max_tokens' in extension_source
    assert 'delete request.max_completion_tokens' in extension_source
    # Guarded delete: only Pi's serialized default ceiling is stripped;
    # explicit user caps must survive (the unconditional delete erased them).
    assert "mtplxPiInjectedDefaultMaxTokens" in extension_source
    assert 'event.headers["x-mtplx-session-id"]' in extension_source
    assert 'const mtplxModelID = "mtplx-test-model"' in extension_source


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_pi_extension_cap_guard_three_payload_shapes(tmp_path):
    """Execute the Pi request-policy handler under node for the three payload
    shapes: injected default (stripped), explicit cap (preserved), no cap
    (untouched)."""

    from mtplx.pi import (
        PI_INJECTED_DEFAULT_MAX_TOKENS,
        build_pi_request_policy_extension_source,
    )

    source = build_pi_request_policy_extension_source(
        "mtplx-test-model", uncapped=True
    )
    # The extension is TypeScript only by annotation; strip ": any" so node
    # can execute the real handler logic unchanged.
    module = tmp_path / "extension.mjs"
    module.write_text(source.replace(": any", ""), encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    harness.write_text(
        f"""
import register from {json.dumps(str(module))};
const handlers = {{}};
register({{ on: (name, fn) => {{ handlers[name] = fn; }} }});
const run = (payload) => handlers["before_provider_request"]({{ payload }});
const results = {{
  injected: run({{ model: "mtplx-test-model", max_tokens: {PI_INJECTED_DEFAULT_MAX_TOKENS} }}) ?? null,
  explicit: run({{ model: "mtplx-test-model", max_tokens: 8192 }}) ?? null,
  absent: run({{ model: "mtplx-test-model" }}) ?? null,
}};
console.log(JSON.stringify(results));
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, check=True
    )
    results = json.loads(proc.stdout)
    # Injected default: handler returns an override with the cap removed.
    assert results["injected"] is not None
    assert "max_tokens" not in results["injected"]
    # Explicit cap: no override returned — the deliberate cap flows through.
    assert results["explicit"] is None
    # No cap at all: nothing to strip, no override.
    assert results["absent"] is None


def test_start_pi_handoff_writes_config_and_starts_authenticated_server(
    monkeypatch,
    tmp_path,
):
    config_path = tmp_path / "pi" / "models.json"
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(config_path))
    monkeypatch.setattr(public.shutil, "which", lambda _name: "/usr/local/bin/pi")
    captured: dict[str, object] = {}

    def fake_serve(serve_args):
        captured["api_key"] = serve_args.api_key
        captured["quickstart_pi"] = serve_args.quickstart_pi
        captured["open_browser"] = serve_args.open_browser
        captured["stats_footer"] = serve_args.stats_footer
        captured["model_id"] = serve_args.model_id
        captured["top_p"] = serve_args.top_p
        captured["top_k"] = serve_args.top_k
        captured["draft_top_p"] = serve_args.draft_top_p
        captured["draft_top_k"] = serve_args.draft_top_k
        return 0

    monkeypatch.setattr(public, "cmd_serve_public", fake_serve)
    args = SimpleNamespace(
        host="127.0.0.1",
        port=18012,
        model="/models/qwen",
        model_id="mtplx-test-model",
        profile="sustained",
        max=False,
        api_key=None,
        cache_dir=None,
        unsafe_force_unverified=False,
        depth=3,
        no_mtp=False,
        rate_limit=0,
        stream_interval=1,
        warmup_tokens=16,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning=None,
        reasoning_parser="qwen3",
        strict_warmup=False,
        strict_fast_path=False,
        max_idle_min=15,
    )

    rc = public._quickstart_run_pi(args, runtime_model="/models/qwen", inspection={})

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert captured == {
        "api_key": "mtplx-local",
        "quickstart_pi": True,
        "open_browser": False,
        "stats_footer": False,
        "model_id": "mtplx-test-model",
        "top_p": 0.95,
        "top_k": 20,
        # No user draft flags: the handoff forwards None so the serve
        # correction chain resolves the family/stamp draft sampler.
        "draft_top_p": None,
        "draft_top_k": None,
    }
    assert payload["providers"]["mtplx"]["baseUrl"] == "http://127.0.0.1:18012/v1"
    assert payload["providers"]["mtplx"]["models"][0]["id"] == "mtplx-test-model"


def test_start_openwebui_handoff_uses_loaded_artifact_model_id(monkeypatch):
    captured: dict[str, object] = {}

    def fake_serve(serve_args):
        captured["model"] = serve_args.model
        captured["model_id"] = serve_args.model_id
        captured["quickstart_openwebui"] = serve_args.quickstart_openwebui
        return 0

    monkeypatch.setattr(public, "cmd_serve_public", fake_serve)
    args = SimpleNamespace(
        host="127.0.0.1",
        port=8022,
        model="/models/Qwen3.5-4B-MTPLX-Optimized-Speed",
        model_id=DEFAULT_PUBLIC_MODEL_ID,
        _cli_flags=set(),
        profile="sustained",
        max=False,
        api_key=None,
        cache_dir=None,
        unsafe_force_unverified=False,
        depth=3,
        no_mtp=False,
        rate_limit=0,
        stream_interval=1,
        warmup_tokens=0,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        draft_temperature=None,
        draft_top_p=None,
        draft_top_k=None,
        reasoning=None,
        reasoning_parser="qwen3",
        reasoning_effort=None,
        strict_warmup=False,
        strict_fast_path=False,
        open_dashboard=False,
        enable_thermal_poll=False,
        max_idle_min=15,
    )

    rc = public._quickstart_run_openwebui(
        args,
        runtime_model="/models/Qwen3.5-4B-MTPLX-Optimized-Speed",
        inspection={},
    )

    assert rc == 0
    assert captured == {
        "model": "/models/Qwen3.5-4B-MTPLX-Optimized-Speed",
        "model_id": "qwen3.5-4b-mtplx-optimized-speed",
        "quickstart_openwebui": True,
    }


def test_run_json_model_summary_excludes_heavy_inspect_fields():
    summary = public._compact_model_summary(
        {
            "source": "local",
            "model_dir": "/models/champion",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "model_type": "qwen3_5_text",
            "mtp_arch": "qwen3-next-mtp",
            "mtp_supported": "yes",
            "recommended_backend": "qwen3_next",
            "runtime_compatibility": "native",
            "runtime_contract_path": None,
            "mtp": {"tensors": [{"key": "large"}]},
            "quantization": {"language_model.model.layers.0": {"bits": 4}},
            "compatibility": {
                "tier": "verified",
                "can_run": True,
                "supported": True,
                "exit_code": 0,
                "message": "Verified MTPLX runtime contract found.",
                "arch_id": "qwen3-next-mtp",
                "recommended_profile": "stable",
                "runtime_contract_path": "/models/champion/mtplx_runtime.json",
            },
        }
    )

    assert summary["model_dir"] == "/models/champion"
    assert summary["compatibility"]["tier"] == "verified"
    assert summary["runtime_contract_path"] == "/models/champion/mtplx_runtime.json"
    assert "mtp" not in summary
    assert "quantization" not in summary


def test_inspect_human_uses_compatibility_runtime_contract(capsys):
    public._print_inspect_human(
        {
            "model_dir": "/models/champion",
            "source": "local",
            "architecture": "Qwen3_5ForConditionalGeneration",
            "mtp_num_hidden_layers": 1,
            "mtp": {"tensor_count": 29},
            "runtime_contract_path": None,
            "compatibility": {
                "tier": "verified",
                "can_run": True,
                "recognized": True,
                "runtime_contract_path": "/models/champion/mtplx_runtime.json",
                "runtime_compatibility": "native",
                "support_level": "verified-native",
                "recommended_profile": "stable",
                "message": "Verified MTPLX runtime contract found.",
            },
        }
    )

    captured = capsys.readouterr().out
    assert "runtime_contract: true" in captured


def test_public_bench_run_dry_run(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "flappy",
            "--max-tokens",
            "10000",
            "--no-fanmax",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "bench run"
    assert payload["exactness_smoke"]["automatic"] is True
    assert payload["profile"]["name"] == "sustained"
    assert payload["harness"] == "direct-http"
    assert payload["runtime_profile"] == "native_mtp_sustained"
    assert payload["runtime_env"]["MTPLX_SUSTAINED_PREFILL"] == "1"
    assert payload["direct_http_command"] is not None
    assert "--profiles" in payload["direct_http_command"]
    assert (
        payload["direct_http_command"][
            payload["direct_http_command"].index("--profiles") + 1
        ]
        == "sustained"
    )


def test_public_bench_long_context_default_is_sustained(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "long_code_uncapped",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["profile"]["name"] == "sustained"
    assert payload["harness"] == "direct-http"
    assert payload["runtime_profile"] == "native_mtp_sustained"
    assert (
        payload["direct_http_command"][
            payload["direct_http_command"].index("--profiles") + 1
        ]
        == "sustained"
    )


def test_public_bench_long_code_dash_alias_uses_sustained_direct_test(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "long-code",
            "--max-tokens",
            "1024",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    command = payload["direct_http_command"]
    assert code == 0
    assert payload["profile"]["name"] == "sustained"
    assert command[command.index("--profiles") + 1] == "sustained"
    assert command[command.index("--tests") + 1] == "long_code"


def test_tune_dry_run_prints_clean_candidate_commands(capsys):
    code = main(
        [
            "tune",
            "--model",
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "MTPLX Tune" in out
    assert "dry-run: no model will be loaded" in out
    assert "--_candidate ar" in out
    assert "--_candidate 3" in out


def test_tune_dry_run_threads_explicit_profile_to_every_candidate(capsys):
    code = main(
        [
            "tune",
            "--model",
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "--profile",
            "turbo",
            "--dry-run",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["settings"]["profile"] == "turbo"
    assert payload["candidates"]
    for row in payload["candidates"]:
        command = row["command"]
        assert command[command.index("--profile") + 1] == "turbo"


def test_bench_tune_dry_run_is_json_support_payload(capsys):
    code = main(
        [
            "bench",
            "tune",
            "--model",
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "bench tune"
    assert payload["model_family"] == "qwen3_6"
    assert payload["tune_supported"] is True
    assert payload["save_default"] is False
    assert payload["settings"]["depths"] == "1,2,3"
    assert payload["settings"]["max_tokens"] == 512
    assert [row["candidate"] for row in payload["candidates"]] == [
        "AR",
        "D1",
        "D2",
        "D3",
    ]
    assert payload["diagnostics"]["telemetry_enabled"] is True
    assert payload["diagnostics"]["model_source_notes"] == []
    assert "power" in payload["diagnostics"]["description"]


def test_tune_support_prefers_qwen36_artifact_identity_over_mlx_model_type(tmp_path):
    model_dir = tmp_path / "Qwen3.6-35B-A3B-MTPLX-Official4-CyanKiwiMTP-CleanRecipe"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5_moe_text"}),
        encoding="utf-8",
    )

    payload = public._tune_support_payload(str(model_dir))

    assert payload["model_family"] == "qwen3_6"
    assert payload["model_controls"]["model_family"] == "qwen3_6"
    assert payload["tune_supported"] is True


def test_tune_support_trusts_local_mtplx_runtime_without_inspection(
    tmp_path,
    monkeypatch,
):
    model_dir = tmp_path / "Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "arch_id": "qwen3-next-mtp",
                "mtplx_version": "1.0.0",
                "public_model_id": "mtplx-qwen35-9b-optimized-speed",
                "recommended_profile": "sustained",
                "hub": {"repo_id": "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"},
            }
        ),
        encoding="utf-8",
    )

    def fail_inspect(_model):
        raise AssertionError("MTPLX tune support should not inspect official artifacts")

    monkeypatch.setattr(public, "inspect_model", fail_inspect)

    payload = public._tune_support_payload(str(model_dir), inspect_local=True)

    assert payload["tune_supported"] is True
    assert payload["model_family"] == "qwen3_5"
    assert payload["backend_id"] == "qwen3_next"
    assert payload["model_controls"]["tune"]["control_field"] == "depth"


def test_tune_candidate_skips_primary_gate_for_local_mtplx_runtime(
    tmp_path,
    monkeypatch,
    capsys,
):
    model_dir = tmp_path / "Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "arch_id": "qwen3-next-mtp",
                "mtplx_version": "1.0.0",
                "public_model_id": "mtplx-qwen35-9b-optimized-speed",
                "hub": {"repo_id": "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "d1.json"

    def fail_gate(*_args, **_kwargs):
        raise AssertionError("MTPLX tune candidates should not run the generic gate")

    def fake_depth_sweep(**_kwargs):
        return {"ok": True}

    requested_profiles: list[str] = []
    real_get_profile = public.get_profile

    def recording_get_profile(name):
        requested_profiles.append(name)
        return real_get_profile(name)

    from mtplx.benchmarks.runners import mtp_depth_sweep

    monkeypatch.setattr(public, "_model_gate", fail_gate)
    monkeypatch.setattr(public, "_depth_sweep_native60", fake_depth_sweep)
    monkeypatch.setattr(public, "get_profile", recording_get_profile)
    monkeypatch.setattr(
        mtp_depth_sweep,
        "write_depth_sweep",
        lambda path, _result: Path(path).write_text("{}", encoding="utf-8"),
    )

    args = SimpleNamespace(
        _tune_candidate="1",
        _tune_candidate_output=str(output),
        model=str(model_dir),
        cache_dir=None,
        unsafe_force_unverified=False,
        max_tokens=1,
        limit=1,
        seed=0,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        prompt_suite=None,
        draft_temperature=None,
        draft_top_p=None,
        draft_top_k=None,
        mtp_hidden_variant=None,
        base_hidden_variant=None,
        concat_order=None,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        profile="turbo",
    )

    code = public._cmd_tune_candidate(args)
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["candidate"] == "1"
    assert output.exists()
    assert requested_profiles == ["turbo"]


def test_tune_retune_starts_max_fans_before_slow_diagnostics(
    tmp_path,
    monkeypatch,
    capsys,
):
    model_dir = tmp_path / "Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "arch_id": "qwen3-next-mtp",
                "mtplx_version": "1.0.0",
                "public_model_id": "mtplx-qwen35-9b-optimized-speed",
                "hub": {"repo_id": "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"},
            }
        ),
        encoding="utf-8",
    )
    order: list[str] = []

    class FakeMaxSession:
        def __init__(self, **_kwargs):
            self.thermal = {"enabled": True}

        def start(self):
            order.append("max-start")
            return True

        def stop(self):
            order.append("max-stop")
            self.thermal["restore"] = {"ok": True}
            return {"ok": True}

    def assert_after_max(label: str) -> None:
        assert "max-start" in order, f"{label} ran before max fans"
        order.append(label)

    def fake_run_candidates(*_args, **_kwargs):
        assert_after_max("candidates")
        return [
            {
                "candidate": "ar",
                "mode": "AR",
                "depth": None,
                "tok_s": 10.0,
                "quality_passed": True,
            },
            {
                "candidate": "1",
                "mode": "D1",
                "depth": 1,
                "tok_s": 12.0,
                "quality_passed": True,
                "acceptance_by_depth": [0.8],
            },
        ]

    monkeypatch.setattr("mtplx.thermal.MaxSession", FakeMaxSession)
    monkeypatch.setattr(public, "_run_tune_candidates", fake_run_candidates)
    monkeypatch.setattr(
        public,
        "_apple_hardware_context",
        lambda: assert_after_max("hardware") or {"chip": "Apple M5 Max"},
    )
    monkeypatch.setattr(
        public,
        "_software_context",
        lambda: assert_after_max("software") or {"mtplx_version": "1.0.0"},
    )
    monkeypatch.setattr(
        public,
        "_mlx_backend_context",
        lambda: assert_after_max("backend") or {"stock_mlx_likely": True},
    )
    monkeypatch.setenv("MTPLX_TUNE_STATE", str(tmp_path / "tune-state.json"))

    args = SimpleNamespace(
        _cli_flags={"model"},
        model=str(model_dir),
        mtplx_config={},
        run_id="retune-order",
        output_dir=str(tmp_path / "runs"),
        output=None,
        json=True,
        verbose=False,
        dry_run=False,
        cache_dir=None,
        retune=True,
        depths="1",
        max_tokens=1,
        limit=1,
        seed=0,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        no_save=True,
        no_telemetry=True,
        prompt_suite=None,
        suite=None,
        mtp_hidden_variant=None,
        base_hidden_variant=None,
        concat_order=None,
        draft_temperature=None,
        draft_top_p=None,
        draft_top_k=None,
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
    )

    code = public._cmd_tune(
        args,
        action="tune",
        save_default=False,
        verbose_default=False,
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["best"]["depth"] == 1
    assert order[:3] == ["max-start", "candidates", "max-stop"]


def test_bench_tune_dry_run_can_disable_telemetry(capsys):
    code = main(
        [
            "bench",
            "tune",
            "--model",
            "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            "--dry-run",
            "--no-telemetry",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["diagnostics"]["telemetry_enabled"] is False
    assert "telemetry disabled" in payload["diagnostics"]["description"]


def test_tune_dry_run_supports_gemma_block_candidates(capsys):
    code = main(
        [
            "tune",
            "--model",
            "Youssofal/Gemma4-MTPLX-Optimized-Speed",
            "--dry-run",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["tune_supported"] is True
    assert payload["model_family"] == "gemma4"
    assert payload["settings"]["control_field"] == "draft_block_size"
    assert payload["settings"]["depths"] == "2,3,4,5,6,7,8"
    assert payload["model_controls"]["tune"]["supported"] is True
    assert payload["model_controls"]["tune"]["control_field"] == "draft_block_size"
    assert [row["candidate"] for row in payload["candidates"]] == [
        "AR",
        "Block 2",
        "Block 3",
        "Block 4",
        "Block 5",
        "Block 6",
        "Block 7",
        "Block 8",
    ]
    assert (
        payload["candidates"][-1]["command"][
            payload["candidates"][-1]["command"].index("--_candidate") + 1
        ]
        == "8"
    )


def test_tune_dry_run_prints_gemma_block_candidates(capsys):
    code = main(
        [
            "tune",
            "--model",
            "Youssofal/Gemma4-MTPLX-Optimized-Speed",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Block 2" in captured.out
    assert "Block 8" in captured.out
    assert "--_candidate 8" in captured.out


def test_bench_tune_dry_run_warns_when_config_model_differs_from_default(
    tmp_path, monkeypatch, capsys
):
    config = tmp_path / "config.toml"
    configured_model = (
        "/Users/youssof/Documents/CustomModels/Qwen3.6-27B-MTPLX-Optimized-Speed"
    )
    config.write_text(
        f'model = "{configured_model}"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_CONFIG", str(config))
    monkeypatch.setattr(
        public,
        "select_default_model",
        lambda: SimpleNamespace(
            model="/Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized"
        ),
    )

    code = main(["bench", "tune", "--dry-run", "--json", "--no-telemetry"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == configured_model
    notes = payload["diagnostics"]["model_source_notes"]
    assert any("using configured model" in note for note in notes)
    assert any("verified default for this Mac" in note for note in notes)


def test_bench_tune_powermetrics_parser_extracts_power_frequency_and_utilization():
    parsed = public._parse_powermetrics_text(
        """
M0-Cluster HW active frequency: 1600 MHz
M0-Cluster HW active residency:  65.53%
M1-Cluster HW active frequency: 1407 MHz
M1-Cluster HW active residency:  18.71%
P-Cluster HW active frequency: 3844 MHz
P-Cluster HW active residency:  78.03%
CPU Power: 5267 mW
GPU Power: 128 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 5395 mW
Current pressure level: Nominal
GPU HW active frequency: 338 MHz
GPU HW active residency:  18.41%
GPU Power: 124 mW
"""
    )

    assert parsed["power_w"]["package"] == pytest.approx(5.395)
    assert parsed["power_w"]["cpu"] == pytest.approx(5.267)
    assert parsed["power_w"]["ane"] == 0.0
    assert parsed["power_w"]["gpu"] == pytest.approx(0.124)
    assert parsed["frequency_ghz"]["p_cluster"] == pytest.approx(3.844)
    assert parsed["frequency_ghz"]["m_cluster"] == pytest.approx((1.6 + 1.407) / 2)
    assert parsed["frequency_ghz"]["gpu"] == pytest.approx(0.338)
    assert parsed["utilization_pct"]["p_core"] == pytest.approx(78.03)
    assert parsed["utilization_pct"]["m_core"] == pytest.approx((65.53 + 18.71) / 2)
    assert parsed["utilization_pct"]["gpu"] == pytest.approx(18.41)
    assert parsed["thermal_pressure"] == "Nominal"


def test_bench_tune_thermalforge_temperature_grouping_prefers_core_sensors():
    core, gpu = public._temperature_groups_from_thermalforge(
        {
            "TAOL": 36.7,
            "TCDX": 56.1,
            "TCMb": 65.2,
            "TG0B": 36.6,
            "Tp04": 56.4,
            "Tm08": 54.4,
        }
    )

    assert core == [56.1, 65.2, 56.4, 54.4]
    assert gpu == [36.6]


def test_tune_best_multiplier_selects_depth_not_ar():
    rows = public._annotate_multipliers(
        [
            {"mode": "AR", "depth": None, "tok_s": 30.0},
            {"mode": "D1", "depth": 1, "tok_s": 48.0},
            {"mode": "D2", "depth": 2, "tok_s": 54.0},
            {"mode": "D3", "depth": 3, "tok_s": 57.0},
        ]
    )
    best = public._best_multiplier_summary(rows)

    assert rows[0]["multiplier_vs_ar"] == 1.0
    assert best["winner"]["mode"] == "D3"
    assert best["winner"]["depth"] == 3
    assert best["winner"]["multiplier_vs_ar"] == 1.9


def test_tune_no_mtp_win_has_no_saved_recommendation():
    best = public._best_multiplier_summary(
        public._annotate_multipliers(
            [
                {"mode": "AR", "depth": None, "tok_s": 50.0},
                {"mode": "D1", "depth": 1, "tok_s": 49.0},
                {"mode": "D2", "depth": 2, "tok_s": 48.0},
                {"mode": "D3", "depth": 3, "tok_s": 47.0},
            ]
        )
    )

    assert best["winner"] is None
    assert best["verdict"] == "no_mtp_depth_beat_ar"


def test_tune_no_mtp_win_labels_collapsed_acceptance():
    best = public._best_multiplier_summary(
        public._annotate_multipliers(
            [
                {"mode": "AR", "depth": None, "tok_s": 96.0},
                {"mode": "D1", "depth": 1, "tok_s": 68.0, "acceptance_by_depth": [0.0]},
                {
                    "mode": "D2",
                    "depth": 2,
                    "tok_s": 54.0,
                    "acceptance_by_depth": [0.0, 0.0],
                },
                {
                    "mode": "D3",
                    "depth": 3,
                    "tok_s": 45.0,
                    "acceptance_by_depth": [0.0, 0.0, 0.0],
                },
            ]
        )
    )

    assert best["winner"] is None
    assert best["verdict"] == "mtp_acceptance_collapsed"
    assert best["failure_reasons"] == [
        "mtp_acceptance_collapsed",
        "no_mtp_depth_beat_ar",
    ]
    assert [row["mode"] for row in best["acceptance_collapsed"]] == ["D1", "D2", "D3"]


def test_tune_rejects_quality_failed_speed_winner():
    best = public._best_multiplier_summary(
        public._annotate_multipliers(
            [
                {"mode": "AR", "depth": None, "tok_s": 65.0},
                {"mode": "D1", "depth": 1, "tok_s": 59.0, "quality_passed": True},
                {"mode": "D2", "depth": 2, "tok_s": 62.0, "quality_passed": True},
                {"mode": "D3", "depth": 3, "tok_s": 68.0, "quality_passed": False},
            ]
        )
    )

    assert best["winner"] is None
    assert best["quality_rejected"][0]["mode"] == "D3"
    assert best["verdict"] == "no_quality_passed_mtp_depth_beat_ar"


def test_tune_uses_fastest_quality_passed_depth():
    best = public._best_multiplier_summary(
        public._annotate_multipliers(
            [
                {"mode": "AR", "depth": None, "tok_s": 65.0},
                {"mode": "D1", "depth": 1, "tok_s": 66.0, "quality_passed": True},
                {"mode": "D2", "depth": 2, "tok_s": 67.0, "quality_passed": True},
                {"mode": "D3", "depth": 3, "tok_s": 70.0, "quality_passed": False},
            ]
        )
    )

    assert best["winner"]["mode"] == "D2"
    assert best["quality_rejected"][0]["mode"] == "D3"
    assert best["verdict"] == "mtp_depth_wins"


def test_tune_tie_breaker_prefers_deeper_depth_within_noise_band():
    best = public._best_multiplier_summary(
        public._annotate_multipliers(
            [
                {"mode": "AR", "depth": None, "tok_s": 29.57},
                {"mode": "D1", "depth": 1, "tok_s": 49.14},
                {"mode": "D2", "depth": 2, "tok_s": 54.67},
                {"mode": "D3", "depth": 3, "tok_s": 54.34},
            ]
        )
    )

    assert best["raw_winner"]["mode"] == "D2"
    assert best["winner"]["mode"] == "D3"
    assert best["tie_breaker"]["applied"] is True


def test_tune_state_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_TUNE_STATE", str(tmp_path / "tuning.json"))
    payload = {
        "best": {"mode": "D2", "depth": 2, "tok_s": 54.0, "multiplier_vs_ar": 1.8},
        "results": [],
    }

    public._save_tune_record("key", key_material={"model": "m"}, payload=payload)
    record = public._load_tune_record("key")

    assert record is not None
    assert record["payload"]["best"]["depth"] == 2


def test_tune_collapsed_acceptance_row_cannot_win_on_multiplier():
    # The #177 artifact: another GPU workload suppressed the AR window
    # (86.9 -> 40.8 tok/s) so a zero-acceptance depth "beat" AR on wall
    # clock. A 0.0-acceptance winner is mechanically impossible as a true
    # result, so it must not win or be saved.
    best = public._best_multiplier_summary(
        public._annotate_multipliers(
            [
                {"mode": "AR", "depth": None, "tok_s": 40.8},
                {
                    "mode": "D2",
                    "depth": 2,
                    "tok_s": 45.4,
                    "quality_passed": True,
                    "acceptance_by_depth": [0.0, 0.0],
                },
            ]
        )
    )

    assert best["winner"] is None
    assert best["verdict"] == "mtp_acceptance_collapsed"
    assert [row["mode"] for row in best["acceptance_collapsed"]] == ["D2"]


def test_tune_collapsed_row_excluded_but_healthy_depth_still_wins():
    best = public._best_multiplier_summary(
        public._annotate_multipliers(
            [
                {"mode": "AR", "depth": None, "tok_s": 30.0},
                {
                    "mode": "D1",
                    "depth": 1,
                    "tok_s": 45.0,
                    "acceptance_by_depth": [0.9],
                },
                {
                    "mode": "D2",
                    "depth": 2,
                    "tok_s": 48.0,
                    "acceptance_by_depth": [0.0, 0.0],
                },
            ]
        )
    )

    # D2 has the higher multiplier but zero measured acceptance; the honest
    # depth wins and the collapsed row stays visible in the summary.
    assert best["winner"]["mode"] == "D1"
    assert best["verdict"] == "mtp_depth_wins"
    assert [row["mode"] for row in best["acceptance_collapsed"]] == ["D2"]


def test_tune_state_load_quarantines_collapsed_winner(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_TUNE_STATE", str(tmp_path / "tuning.json"))
    poisoned = {
        "best": {"mode": "D2", "depth": 2, "tok_s": 45.4, "multiplier_vs_ar": 1.11},
        "results": [
            {"mode": "AR", "depth": None, "tok_s": 40.8},
            {
                "mode": "D2",
                "depth": 2,
                "tok_s": 45.4,
                "acceptance_by_depth": [0.0, 0.0],
            },
        ],
    }
    healthy = {
        "best": {"mode": "D3", "depth": 3, "tok_s": 57.0, "multiplier_vs_ar": 1.9},
        "results": [
            {"mode": "AR", "depth": None, "tok_s": 30.0},
            {
                "mode": "D3",
                "depth": 3,
                "tok_s": 57.0,
                "acceptance_by_depth": [0.9, 0.8, 0.7],
            },
        ],
    }

    public._save_tune_record("poisoned", key_material={"model": "m"}, payload=poisoned)
    public._save_tune_record("healthy", key_material={"model": "m"}, payload=healthy)

    # A record whose own results show a zero-acceptance winner is treated as
    # absent by every consumer instead of replaying the poisoned depth.
    assert public._load_tune_record("poisoned") is None
    record = public._load_tune_record("healthy")
    assert record is not None
    assert record["payload"]["best"]["depth"] == 3


def test_tune_clear_record_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_TUNE_STATE", str(tmp_path / "tuning.json"))
    payload = {
        "best": {"mode": "D2", "depth": 2, "tok_s": 45.4, "multiplier_vs_ar": 1.11},
        "results": [],
    }
    public._save_tune_record("key", key_material={"model": "m"}, payload=payload)

    cleared = public._clear_tune_record("key")

    assert cleared is not None
    assert cleared["previous_best"]["depth"] == 2
    assert public._load_tune_record("key") is None
    state = json.loads((tmp_path / "tuning.json").read_text(encoding="utf-8"))
    assert state["records"] == {}
    assert public._clear_tune_record("key") is None


def test_tune_measured_no_winner_clears_saved_record(tmp_path, monkeypatch, capsys):
    # End to end over the real save path: a healthy tune saves a winner, a
    # later honest retune that measures collapse clears the stored record
    # instead of leaving it immortal (#177).
    model_dir = tmp_path / "Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed"
    model_dir.mkdir()
    (model_dir / "mtplx_runtime.json").write_text(
        json.dumps(
            {
                "arch_id": "qwen3-next-mtp",
                "mtplx_version": "1.0.0",
                "public_model_id": "mtplx-qwen35-9b-optimized-speed",
                "hub": {"repo_id": "Youssofal/Qwen3.5-9B-MTPLX-Optimized-Speed"},
            }
        ),
        encoding="utf-8",
    )

    class FakeMaxSession:
        def __init__(self, **_kwargs):
            self.thermal = {"enabled": True}

        def start(self):
            return True

        def stop(self):
            self.thermal["restore"] = {"ok": True}
            return {"ok": True}

    rows_holder: dict[str, list[dict[str, object]]] = {}

    def fake_run_candidates(*_args, **_kwargs):
        return rows_holder["rows"]

    monkeypatch.setattr("mtplx.thermal.MaxSession", FakeMaxSession)
    monkeypatch.setattr(public, "_run_tune_candidates", fake_run_candidates)
    monkeypatch.setattr(
        public, "_apple_hardware_context", lambda: {"chip": "Apple M5 Max"}
    )
    monkeypatch.setattr(public, "_software_context", lambda: {"mtplx_version": "1.0.0"})
    monkeypatch.setattr(
        public, "_mlx_backend_context", lambda: {"stock_mlx_likely": True}
    )
    monkeypatch.setenv("MTPLX_TUNE_STATE", str(tmp_path / "tune-state.json"))

    def make_args(*, retune: bool) -> SimpleNamespace:
        return SimpleNamespace(
            _cli_flags={"model"},
            model=str(model_dir),
            mtplx_config={},
            run_id=f"clear-record-{int(retune)}",
            output_dir=str(tmp_path / "runs"),
            output=None,
            json=True,
            verbose=False,
            dry_run=False,
            cache_dir=None,
            retune=retune,
            depths="2",
            max_tokens=1,
            limit=1,
            seed=0,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            no_save=False,
            no_telemetry=True,
            prompt_suite=None,
            suite=None,
            mtp_hidden_variant=None,
            base_hidden_variant=None,
            concat_order=None,
            draft_temperature=None,
            draft_top_p=None,
            draft_top_k=None,
            mtp_cache_policy="persistent",
            mtp_history_policy="committed",
        )

    rows_holder["rows"] = [
        {
            "candidate": "ar",
            "mode": "AR",
            "depth": None,
            "tok_s": 40.0,
            "quality_passed": True,
        },
        {
            "candidate": "2",
            "mode": "D2",
            "depth": 2,
            "tok_s": 60.0,
            "quality_passed": True,
            "acceptance_by_depth": [0.9, 0.8],
        },
    ]
    code = public._cmd_tune(
        make_args(retune=False), action="tune", save_default=True, verbose_default=False
    )
    saved_payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert saved_payload["saved"] is True
    assert saved_payload["best"]["depth"] == 2

    rows_holder["rows"] = [
        {
            "candidate": "ar",
            "mode": "AR",
            "depth": None,
            "tok_s": 86.9,
            "quality_passed": True,
        },
        {
            "candidate": "2",
            "mode": "D2",
            "depth": 2,
            "tok_s": 51.0,
            "quality_passed": True,
            "acceptance_by_depth": [0.0, 0.0],
        },
    ]
    code = public._cmd_tune(
        make_args(retune=True), action="tune", save_default=True, verbose_default=False
    )
    retune_payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert retune_payload["saved"] is False
    assert retune_payload["best_multiplier"]["verdict"] == "mtp_acceptance_collapsed"
    assert retune_payload["cleared_saved_record"]["previous_best"]["depth"] == 2
    state = json.loads((tmp_path / "tune-state.json").read_text(encoding="utf-8"))
    assert state["records"] == {}


def test_tune_model_source_notes_warn_when_config_model_differs_from_default(
    monkeypatch,
):
    monkeypatch.setattr(
        public,
        "select_default_model",
        lambda: SimpleNamespace(
            model="/Users/youssof/.mtplx/hf-upload/Qwen3.6-27B-MTPLX-Optimized"
        ),
    )
    args = SimpleNamespace(
        _cli_flags=set(),
        mtplx_config={
            "path": "/Users/youssof/.mtplx/config.toml",
            "model": "/Users/youssof/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
        },
    )

    notes = public._tune_model_source_notes(
        args,
        runtime_model="/Users/youssof/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed",
    )

    assert any("using configured model" in note for note in notes)
    assert any("verified default for this Mac" in note for note in notes)


def test_tune_candidate_outputs_are_absolute_from_non_repo_cwd(tmp_path, monkeypatch):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    progress: list[str] = []

    def fake_run(command, *, cwd, env, text, stdout, stderr, check):
        output_arg = command[command.index("--_candidate-output") + 1]
        output = Path(output_arg)
        assert output.is_absolute()
        assert str(output).startswith(str(caller))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"ar_rows": [{"tok_s": 12.0, "generated_tokens": 2}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="candidate ok")

    monkeypatch.setattr(public.subprocess, "run", fake_run)

    rows = public._run_tune_candidates(
        SimpleNamespace(cache_dir=None, unsafe_force_unverified=False),
        runtime_model="/tmp/model",
        run_id="run",
        output_root=Path("outputs/cli/tune/run"),
        depths=[],
        settings={
            "max_tokens": 8,
            "limit": 1,
            "seed": 0,
            "depths": "",
        },
        progress=progress.append,
    )

    assert rows[0]["mode"] == "AR"
    assert rows[0]["tok_s"] == 12.0
    assert any("AR (1/1) starting" in line for line in progress)
    assert any("AR finished" in line for line in progress)


def test_tune_candidate_summary_promotes_child_json_error(tmp_path):
    stdout = tmp_path / "ar.log"
    stdout.write_text(
        json.dumps(
            {
                "error": "model failed MTP primary gate",
                "model": {
                    "model_dir": "models/Qwen3.6-27B-MTPLX-Optimized-Speed",
                    "config_exists": False,
                    "model_files": [],
                    "compatibility": {
                        "message": "Model has no MTP head. MTPLX requires an MTP-equipped model.",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    row = public._tune_candidate_summary(
        "ar",
        tmp_path / "missing-ar.json",
        returncode=2,
        stdout_path=stdout,
        command=["mtplx", "tune"],
    )

    assert row["returncode"] == 2
    assert row["model"] == "models/Qwen3.6-27B-MTPLX-Optimized-Speed"
    assert row["child_error"]["model"]["config_exists"] is False
    assert "model failed MTP primary gate: Model has no MTP head" in row["error"]
    assert row["error"] != "candidate did not write an artifact"


def test_tune_candidate_summary_prefers_decode_tok_s(tmp_path):
    ar_artifact = tmp_path / "ar.json"
    ar_artifact.write_text(
        json.dumps(
            {
                "ar_rows": [
                    {
                        "tok_s": 20.0,
                        "decode_tok_s": 20.0,
                        "decode_elapsed_s": 9.6,
                        "end_to_end_tok_s": 14.4,
                        "elapsed_s": 13.3,
                        "prompt_eval_time_s": 3.7,
                        "generated_tokens": 192,
                        "finish_reason": "length",
                        "hit_token_budget": True,
                        "validations": [
                            {"name": "no_degenerate_loop", "passed": True},
                            {"name": "balanced_delimiters", "passed": False},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    depth_artifact = tmp_path / "d1.json"
    depth_artifact.write_text(
        json.dumps(
            {
                "depths": [
                    {
                        "summary": {
                            "mean_tok_s": 21.0,
                            "mean_decode_tok_s": 21.0,
                            "mean_end_to_end_tok_s": 15.5,
                            "generated_tokens": 192,
                            "elapsed_s": 12.4,
                            "verify_calls": 2,
                            "hit_token_budget_count": 1,
                            "finish_reasons": {"length": 1},
                        },
                        "rows": [
                            {
                                "finish_reason": "length",
                                "hit_token_budget": True,
                                "validations": [
                                    {"name": "no_degenerate_loop", "passed": True},
                                    {"name": "balanced_delimiters", "passed": True},
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    ar_row = public._tune_candidate_summary(
        "ar",
        ar_artifact,
        returncode=0,
        stdout_path=tmp_path / "ar.log",
        command=["mtplx", "tune"],
    )
    depth_row = public._tune_candidate_summary(
        "1",
        depth_artifact,
        returncode=0,
        stdout_path=tmp_path / "d1.log",
        command=["mtplx", "tune"],
    )

    assert ar_row["tok_s"] == 20.0
    assert ar_row["decode_tok_s"] == 20.0
    assert ar_row["end_to_end_tok_s"] == 14.4
    assert depth_row["tok_s"] == 21.0
    assert depth_row["decode_tok_s"] == 21.0
    assert depth_row["end_to_end_tok_s"] == 15.5
    assert ar_row["hit_token_budget"] is True
    assert ar_row["quality_passed"] is True
    assert (
        ar_row["quality_inconclusive_validations"][0]["name"] == "balanced_delimiters"
    )
    assert depth_row["hit_token_budget_count"] == 1
    assert depth_row["finish_reasons"] == {"length": 1}


def test_tune_candidate_summary_still_fails_degenerate_budget_output(tmp_path):
    artifact = tmp_path / "d1.json"
    artifact.write_text(
        json.dumps(
            {
                "depths": [
                    {
                        "summary": {
                            "mean_decode_tok_s": 21.0,
                            "generated_tokens": 192,
                            "hit_token_budget_count": 1,
                            "finish_reasons": {"length": 1},
                        },
                        "rows": [
                            {
                                "finish_reason": "length",
                                "hit_token_budget": True,
                                "validations": [
                                    {"name": "no_degenerate_loop", "passed": False},
                                    {"name": "balanced_delimiters", "passed": False},
                                ],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    row = public._tune_candidate_summary(
        "1",
        artifact,
        returncode=0,
        stdout_path=tmp_path / "d1.log",
        command=["mtplx", "tune"],
    )

    assert row["quality_passed"] is False
    assert row["quality_failure_validations"][0]["name"] == "no_degenerate_loop"
    assert row["quality_inconclusive_validations"][0]["name"] == "balanced_delimiters"


def test_tune_candidate_command_passes_non_default_prompt_suite(tmp_path):
    command = public._tune_candidate_command(
        SimpleNamespace(cache_dir=None, unsafe_force_unverified=False),
        candidate="1",
        model="/tmp/model",
        output=tmp_path / "d1.json",
        settings={
            "suite": "long-code-uncapped",
            "max_tokens": 512,
            "limit": 1,
            "seed": 0,
            "depths": "1,2,3",
        },
    )

    assert command[command.index("--prompt-suite") + 1] == "long-code-uncapped"


def test_tune_candidate_command_passes_sampler_policy(tmp_path):
    command = public._tune_candidate_command(
        SimpleNamespace(cache_dir=None, unsafe_force_unverified=False),
        candidate="1",
        model="/tmp/model",
        output=tmp_path / "d1.json",
        settings={
            "profile": "turbo",
            "suite": "long-code-uncapped",
            "max_tokens": 512,
            "limit": 1,
            "seed": 0,
            "depths": "1,2,3",
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": 13,
        },
    )

    assert command[command.index("--temperature") + 1] == "0.7"
    assert command[command.index("--top-p") + 1] == "1.0"
    assert command[command.index("--top-k") + 1] == "13"
    assert command[command.index("--profile") + 1] == "turbo"


def test_tune_candidates_settle_between_depth_runs(tmp_path, monkeypatch):
    sleeps: list[float] = []
    progress: list[str] = []

    def fake_sleep(seconds):
        sleeps.append(seconds)

    def fake_run(command, *, cwd, env, text, stdout, stderr, check):
        output_arg = command[command.index("--_candidate-output") + 1]
        candidate = command[command.index("--_candidate") + 1]
        output = Path(output_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        if candidate == "ar":
            payload = {"ar_rows": [{"tok_s": 12.0, "generated_tokens": 2}]}
        else:
            payload = {
                "depths": [
                    {
                        "summary": {
                            "mean_tok_s": 13.0,
                            "generated_tokens": 2,
                            "verify_calls": 1,
                        }
                    }
                ]
            }
        output.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="candidate ok")

    monkeypatch.setattr(public.time, "sleep", fake_sleep)
    monkeypatch.setattr(public.subprocess, "run", fake_run)

    rows = public._run_tune_candidates(
        SimpleNamespace(cache_dir=None, unsafe_force_unverified=False),
        runtime_model="/tmp/model",
        run_id="run",
        output_root=tmp_path / "run",
        depths=[1],
        settings={
            "max_tokens": 8,
            "limit": 1,
            "seed": 0,
            "depths": "1",
            "candidate_settle_s": 5.0,
        },
        progress=progress.append,
    )

    assert [row["mode"] for row in rows] == ["AR", "D1"]
    assert sleeps == [5.0]
    assert any("settling 5.0s before D1" in line for line in progress)


def test_bench_tune_candidate_rows_include_hardware_telemetry(tmp_path, monkeypatch):
    progress: list[str] = []
    telemetry = {
        "enabled": True,
        "sample_count": 2,
        "power_w": {"package": {"avg": 42.0}},
        "frequency_ghz": {"p_cluster": {"avg": 4.05}},
        "temperature_c": {"core_avg": {"avg": 71.0}},
        "utilization_pct": {"gpu": {"avg": 95.0}},
        "fans_rpm": {"avg": {"avg": 7800.0}},
    }

    class FakeSampler:
        def __init__(self, *, enabled):
            self.enabled = enabled

        def start(self):
            assert self.enabled is True

        def stop(self):
            return telemetry

    def fake_run(command, *, cwd, env, text, stdout, stderr, check):
        output_arg = command[command.index("--_candidate-output") + 1]
        output = Path(output_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"ar_rows": [{"tok_s": 12.0, "generated_tokens": 2}]}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="candidate ok")

    monkeypatch.setattr(public, "_TuneTelemetrySampler", FakeSampler)
    monkeypatch.setattr(public.subprocess, "run", fake_run)

    rows = public._run_tune_candidates(
        SimpleNamespace(cache_dir=None, unsafe_force_unverified=False),
        runtime_model="/tmp/model",
        run_id="run",
        output_root=tmp_path / "run",
        depths=[],
        settings={
            "max_tokens": 8,
            "limit": 1,
            "seed": 0,
            "depths": "",
        },
        progress=progress.append,
        collect_telemetry=True,
    )

    assert rows[0]["telemetry"]["power_w"]["package"]["avg"] == 42.0
    assert any(
        "telemetry: scope=candidate | power pkg=42.0W" in line for line in progress
    )


def test_bench_tune_telemetry_prefers_generation_window_samples(tmp_path, monkeypatch):
    progress: list[str] = []
    telemetry = {
        "enabled": True,
        "scope": "candidate_process",
        "sample_count": 2,
        "samples": [
            {
                "timestamp": 100.0,
                "power_w": {"gpu": 5.0},
                "utilization_pct": {"gpu": 10.0},
            },
            {
                "timestamp": 105.0,
                "power_w": {"gpu": 40.0},
                "utilization_pct": {"gpu": 99.0},
            },
        ],
        "power_w": {"gpu": {"avg": 22.5}},
        "utilization_pct": {"gpu": {"avg": 54.5}},
    }

    class FakeSampler:
        def __init__(self, *, enabled):
            self.enabled = enabled

        def start(self):
            assert self.enabled is True

        def stop(self):
            return dict(telemetry)

    def fake_run(command, *, cwd, env, text, stdout, stderr, check):
        output_arg = command[command.index("--_candidate-output") + 1]
        output = Path(output_arg)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "ar_rows": [
                        {
                            "tok_s": 12.0,
                            "generated_tokens": 2,
                            "generation_started_at": 104.0,
                            "generation_ended_at": 106.0,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="candidate ok")

    monkeypatch.setattr(public, "_TuneTelemetrySampler", FakeSampler)
    monkeypatch.setattr(public.subprocess, "run", fake_run)

    rows = public._run_tune_candidates(
        SimpleNamespace(cache_dir=None, unsafe_force_unverified=False),
        runtime_model="/tmp/model",
        run_id="run",
        output_root=tmp_path / "run",
        depths=[],
        settings={
            "max_tokens": 8,
            "limit": 1,
            "seed": 0,
            "depths": "",
        },
        progress=progress.append,
        collect_telemetry=True,
    )

    generation = rows[0]["telemetry"]["generation"]
    assert generation["scope"] == "generation_window"
    assert generation["utilization_pct"]["gpu"]["avg"] == 99.0
    assert any("scope=generation" in line and "GPU=99.0%" in line for line in progress)


def test_tune_human_reports_candidate_errors_instead_of_false_no_win(capsys):
    payload = {
        "results": [
            {
                "mode": "AR",
                "depth": None,
                "tok_s": None,
                "multiplier_vs_ar": None,
                "error": "candidate did not write an artifact",
                "stdout": "/tmp/ar.log",
            },
            {
                "mode": "D1",
                "depth": 1,
                "tok_s": None,
                "multiplier_vs_ar": None,
                "error": "candidate did not write an artifact",
                "stdout": "/tmp/d1.log",
            },
        ],
        "best": None,
        "saved": False,
        "save_skipped_reason": "tune failed; no candidate produced usable tokens",
    }

    public._print_tune_human(payload)

    out = capsys.readouterr().out
    assert "Tune failed for one or more candidates" in out
    assert "No MTP depth beat AR" not in out
    assert "Close heavy apps" not in out
    assert "/tmp/ar.log" in out


def test_tune_human_results_do_not_give_pre_run_advice_afterward(capsys):
    payload = {
        "results": [
            {"mode": "AR", "depth": None, "tok_s": 20.0, "multiplier_vs_ar": 1.0},
            {"mode": "D1", "depth": 1, "tok_s": 30.0, "multiplier_vs_ar": 1.5},
        ],
        "best": {"mode": "D1", "depth": 1, "tok_s": 30.0, "multiplier_vs_ar": 1.5},
        "saved": False,
        "save_skipped_reason": "save disabled",
        "artifacts": {"root": "/tmp/tune"},
    }

    public._print_tune_human(payload)

    out = capsys.readouterr().out
    assert "Results written to /tmp/tune" in out
    assert "Close heavy apps" not in out
    assert "Fans may get loud" not in out
    assert "Best for this Mac: D1" in out


def test_bench_tune_human_verbose_prints_power_diagnostic_lines(capsys):
    payload = {
        "results": [
            {
                "mode": "AR",
                "depth": None,
                "tok_s": 20.0,
                "multiplier_vs_ar": 1.0,
                "telemetry": {
                    "enabled": True,
                    "sample_count": 3,
                    "power_w": {
                        "package": {"avg": 44.0},
                        "cpu": {"avg": 6.0},
                        "ane": {"avg": 0.0},
                        "gpu": {"avg": 38.0},
                    },
                    "frequency_ghz": {
                        "p_cluster": {"avg": 4.05},
                        "m_cluster": {"avg": 1.05},
                        "gpu": {"avg": 1.22},
                    },
                    "temperature_c": {
                        "core_avg": {"avg": 71.0},
                        "core_max": {"avg": 77.0},
                        "gpu_avg": {"avg": 69.0},
                    },
                    "utilization_pct": {
                        "p_core": {"avg": 17.0},
                        "m_core": {"avg": 10.0},
                        "gpu": {"avg": 99.0},
                    },
                },
            }
        ],
        "best": None,
        "saved": False,
    }

    public._print_tune_human(payload, verbose=True)

    out = capsys.readouterr().out
    assert (
        "telemetry=scope=candidate | power pkg=44.0W cpu=6.0W ane=0.0W gpu=38.0W" in out
    )
    assert "freq P=4.05GHz M=1.05GHz GPU=1.22GHz" in out
    assert "temp core_avg=71.0C core_max=77.0C gpu_avg=69.0C" in out
    assert "util P=17.0% M=10.0% GPU=99.0%" in out


def test_public_bench_run_dry_run_records_external_kernel_env(monkeypatch, capsys):
    monkeypatch.setenv("MTPLX_VERIFY_OUTPUT_DEPENDS", "recurrent")
    monkeypatch.setenv("MTPLX_VERIFY_OUTPUT_DEPENDS_AFTER_TOKENS", "1024")
    monkeypatch.setenv("MTPLX_SDPA_2PASS_BLOCKS", "64")
    monkeypatch.setenv("MTPLX_SDPA_DYNAMIC_OFFSET_ACTIVE_BLOCKS", "1")
    monkeypatch.setenv("MTPLX_EXPORT_VERIFY_DOT_DIR", "outputs/dot-probe")
    monkeypatch.setenv("MTPLX_EXPORT_VERIFY_DOT_CYCLES", "1,128")
    monkeypatch.setenv("MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE", "0")

    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "flappy",
            "--max-tokens",
            "2048",
            "--no-fanmax",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["runtime_env"]["MTPLX_VERIFY_OUTPUT_DEPENDS"] == "recurrent"
    assert payload["runtime_env"]["MTPLX_VERIFY_OUTPUT_DEPENDS_AFTER_TOKENS"] == "1024"
    assert payload["runtime_env"]["MTPLX_SDPA_2PASS_BLOCKS"] == "64"
    assert payload["runtime_env"]["MTPLX_SDPA_DYNAMIC_OFFSET_ACTIVE_BLOCKS"] == "1"
    assert payload["runtime_env"]["MTPLX_EXPORT_VERIFY_DOT_DIR"] == "outputs/dot-probe"
    assert payload["runtime_env"]["MTPLX_EXPORT_VERIFY_DOT_CYCLES"] == "1,128"
    assert payload["runtime_env"]["MTPLX_EVAL_STATE_ROOTS_INCLUDE_LIVE"] == "0"


def test_public_bench_cold_run_defaults_to_sustained_mode(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "cold-long-code-192",
            "--max-tokens",
            "192",
            "--strict-cold",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert code == 0
    assert payload["profile"]["name"] == "sustained"
    assert payload["harness"] == "direct-http"
    assert payload["seed"] == 42
    assert payload["runtime_profile"] == "native_mtp_sustained"
    assert payload["runtime_env"]["MTPLX_SUSTAINED_PREFILL"] == "1"
    assert payload["runtime_env"]["MTPLX_PREFILL_OMLX_EXTERNAL"] == "1"
    assert payload["runtime_env"]["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] == "0"
    assert payload["direct_http_command"] is not None


def test_public_bench_performance_cold_is_explicit(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "cold-long-code-192",
            "--max-tokens",
            "192",
            "--profile",
            "performance-cold",
            "--strict-cold",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    assert code == 0
    assert payload["profile"]["name"] == "performance-cold"
    assert payload["harness"] == "depth-sweep"
    assert payload["seed"] == 0
    assert payload["runtime_profile"] == "native_mtp_60_cold"
    assert payload["runtime_env"]["MTPLX_LAZY_VERIFY_LOGITS"] == "1"
    assert payload["runtime_env"]["MTPLX_LAZY_TARGET_DISTRIBUTIONS"] == "1"
    assert "MTPLX_TARGET_LAYER_EVAL_SCHEDULE" not in payload["runtime_env"]


def test_public_bench_explicit_performance_cold_overrides_long_context_default(capsys):
    code = main(
        [
            "bench",
            "run",
            "--model",
            "models/not-loaded-in-dry-run",
            "--suite",
            "long_code_uncapped",
            "--profile",
            "performance-cold",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["profile"]["name"] == "performance-cold"
    assert payload["harness"] == "depth-sweep"
    assert payload["runtime_profile"] == "native_mtp_60_cold"
    assert payload["direct_http_command"] is None


def test_public_qa_distribution_parser_dry_shape():
    parser = build_parser()
    args = parser.parse_args(
        [
            "qa",
            "distribution",
            "--model",
            "models/example",
            "--suite",
            "distribution-smoke",
        ]
    )

    assert args.command == "qa"
    assert args.qa_action == "distribution"


def test_public_profile_dispatch_without_trace_is_actionable(capsys):
    code = main(
        [
            "profile",
            "dispatch",
            "--model",
            "models/example",
            "--suite",
            "flappy",
            "--max-tokens",
            "2048",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"implemented_capture": false' in captured
    # The next-step hint must stay honest about the trace analyzer: point at
    # --trace when the research-workspace script is present, and say plainly
    # that it is not included when it is absent (the shipped package never
    # carries it).
    assert "--trace PATH" in captured or "not included in this installation" in captured


def test_reference_vllm_dry_run_includes_ssh_capture_command(capsys):
    code = main(
        [
            "bench",
            "reference-vllm",
            "--suite",
            "flappy",
            "--max-tokens",
            "6000",
            "--capture-dispatch",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"action": "bench reference-vllm"' in captured
    assert '"remote_capture_kind": "offline"' in captured
    assert "--capture-range=cudaProfilerApi" in captured
    assert "--cuda-graph-trace=graph" in captured
    assert "cuda_api_sum" in captured
    assert "mtplx-3090" in captured
    assert '"remote_prompt_override"' in captured
    assert '"max_tokens": 6000' in captured


def test_champion_bakeoff_compare_dry_run_lists_required_tasks(capsys):
    code = main(
        [
            "bench",
            "compare",
            "--models",
            "models/a",
            "models/b",
            "--suite",
            "champion-bakeoff",
            "--no-fanmax",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"action": "bench compare"' in captured
    assert '"label": "flappy-10k"' in captured
    assert '"max_tokens": 10000' in captured
    assert '"label": "python-modules-long"' in captured
    assert '"max_tokens": 6000' in captured
    assert '"label": "cold-long-code-192"' in captured
    assert '"strict_cold": true' in captured


def test_public_bench_parser_has_seed_for_live_child_runs():
    parser = build_parser()
    args = parser.parse_args(
        [
            "bench",
            "compare",
            "--models",
            "models/a",
            "models/b",
            "--suite",
            "champion-bakeoff",
        ]
    )

    assert args.seed is None


def test_bench_nightly_dry_run_lists_kernel_gate_tasks(capsys):
    code = main(
        [
            "bench",
            "nightly",
            "--model",
            "models/not-loaded-in-dry-run",
            "--run-id",
            "nightly-test",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "bench nightly"
    assert [task["label"] for task in payload["tasks"]] == [
        "cold-long-code-192",
        "flappy-6k",
        "flappy-10k",
        "python-modules-6k",
    ]
    assert payload["policy"]["fanmax_counts_for_product_gate"] is False
    assert payload["full_exactness_command"][0:2] == ["qa", "exactness"]
    assert [task["profile"] for task in payload["tasks"]] == [
        "performance-cold",
        "sustained",
        "sustained",
        "sustained",
    ]
    flappy_6k_command = payload["tasks"][1]["direct_http_command"]
    assert "--python-bin" in flappy_6k_command
    assert "--max-tokens" in flappy_6k_command
    assert flappy_6k_command[flappy_6k_command.index("--max-tokens") + 1] == "6000"


def test_bench_suite_alias_is_launch_gate_dry_run(capsys):
    code = main(
        [
            "bench",
            "suite",
            "--model",
            "models/not-loaded-in-dry-run",
            "--run-id",
            "suite-test",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "bench suite"
    assert payload["status"] == "PLAN"
    assert payload["output"] == "outputs/cli/suite/suite-test/summary.json"
    assert [task["label"] for task in payload["tasks"]] == [
        "cold-long-code-192",
        "flappy-6k",
        "flappy-10k",
        "python-modules-6k",
    ]


def test_bench_suite_quick_plans_client_contract_rows(capsys):
    code = main(
        [
            "bench",
            "suite",
            "--quick",
            "--model",
            "models/not-loaded-in-dry-run",
            "--run-id",
            "quick-suite-test",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "bench suite"
    assert payload["quick"] is True
    assert payload["status"] == "PLAN"
    assert payload["rows_jsonl"] == "outputs/cli/suite/quick-suite-test/rows.jsonl"
    assert (
        payload["full_exactness_command"][
            payload["full_exactness_command"].index("--contexts") + 1
        ]
        == "64,2048"
    )
    assert [task["label"] for task in payload["tasks"]] == [
        "short-context-384",
        "long-tool-history-1536",
        "opencode-contract-1024",
        "pi-contract-1024",
        "hermes-contract-1024",
    ]
    opencode = payload["tasks"][2]
    command = opencode["direct_http_command"]
    headers = json.loads(command[command.index("--headers-json") + 1])
    metadata = json.loads(command[command.index("--metadata-json") + 1])
    assert headers["x-mtplx-client"] == "opencode"
    assert metadata["mtplx_bench_client"] == "opencode"
    assert opencode["category"] == "client_contract"


def test_bench_suite_quick_uses_verified_local_default_when_model_omitted(
    monkeypatch, capsys
):
    local_default = (
        "/Users/youssof/Documents/MTPLX/models/Qwen3.6-27B-MTPLX-Optimized-Speed"
    )
    monkeypatch.setattr(
        public,
        "select_default_model",
        lambda: SimpleNamespace(
            model=local_default,
            hf_model="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
            variant="speed",
            precision="Q4 target with Q4 MTP sidecar",
            chip_generation="m5",
            chip="Apple M5 Max",
            reason="selected for newer Apple Silicon; installed locally",
            auto_selected=True,
            to_dict=lambda: {"model": local_default, "variant": "speed"},
        ),
    )

    code = main(
        [
            "bench",
            "suite",
            "--quick",
            "--run-id",
            "quick-suite-default-test",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["model"] == local_default
    assert payload["default_model_selection"]["model"] == local_default
    first_command = payload["tasks"][0]["direct_http_command"]
    assert first_command[first_command.index("--model") + 1] == local_default
    assert "--no-strict-mlx-fork-assert" in first_command
    assert (
        payload["full_exactness_command"][
            payload["full_exactness_command"].index("--model") + 1
        ]
        == local_default
    )


def test_bench_suite_status_classifies_hard_and_perf_gates():
    assert (
        public._bench_suite_status(
            {
                "full_exactness_passed": True,
                "quality_passed": True,
                "no_fan_product_gate": True,
                "cold_tok_s_ge_59": True,
            }
        )
        == "PASS"
    )
    assert (
        public._bench_suite_status(
            {
                "full_exactness_passed": True,
                "quality_passed": True,
                "no_fan_product_gate": True,
                "cold_tok_s_ge_59": False,
            }
        )
        == "WARN"
    )
    assert (
        public._bench_suite_status(
            {
                "full_exactness_passed": False,
                "quality_passed": True,
                "no_fan_product_gate": True,
                "cold_tok_s_ge_59": True,
            }
        )
        == "FAIL"
    )


def test_bench_suite_task_status_warns_on_speed_floor_only():
    gates = public._bench_suite_task_gates(
        {"warn_gates": {"tok_s_ge": 35.0}},
        {
            "runtime": {"tok_s": 31.0},
            "decode_trace": {},
            "quality": {"passed": True},
        },
        exit_code=0,
    )

    assert public._bench_suite_task_status(gates) == "WARN"
    assert gates["tok_s_ge_warn_floor"] is False


def test_bench_compare_envelopes_detects_cold_regression(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        json.dumps(
            {
                "suite": "cold-long-code-192",
                "runtime": {"tok_s": 60.0},
                "quality": {"passed": True},
                "correctness": {"exactness_smoke": {"passed": True}},
            }
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            {
                "suite": "cold-long-code-192",
                "runtime": {"tok_s": 58.0},
                "quality": {"passed": True},
                "correctness": {"exactness_smoke": {"passed": True}},
            }
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "bench",
            "compare",
            "--before",
            str(before),
            "--after",
            str(after),
            "--strict-cold",
            "--strict-exactness",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 5
    assert payload["action"] == "bench compare envelopes"
    assert payload["passed"] is False
    assert payload["comparisons"][0]["gates"]["cold_floor_ge_59"] is False


def test_chat_and_serve_default_to_sustained_mode():
    parser = build_parser()

    run_args = parser.parse_args(["run", "hello", "--cache-dir", "/tmp/mtplx-models"])
    chat_args = parser.parse_args(["chat", "--prompt", "hello"])
    serve_args = parser.parse_args(["serve"])
    serve_max_args = parser.parse_args(["serve", "--max"])
    serve_smart_args = parser.parse_args(["serve", "--fan-mode", "smart"])
    serve_app_args = parser.parse_args(
        ["serve", "--max", "--require-max-fans", "--app-launch-id", "native-123"]
    )
    tune_strict_args = parser.parse_args(["tune", "--require-max-fans"])
    serve_no_footer_args = parser.parse_args(["serve", "--no-stats-footer"])

    assert run_args.profile == "sustained"
    assert run_args.prompt_arg == "hello"
    assert run_args.cache_dir == "/tmp/mtplx-models"
    assert run_args.max_tokens is None
    assert run_args.reasoning is None
    assert chat_args.profile == "sustained"
    assert chat_args.max_tokens is None
    assert chat_args.reasoning is None
    assert serve_args.profile == "sustained"
    assert serve_app_args.max is True
    assert serve_app_args.require_max_fans is True
    assert tune_strict_args.require_max_fans is True
    assert serve_app_args.app_launch_id == "native-123"
    assert serve_args.reasoning is None
    assert serve_args.stock_ar is False
    assert serve_args.fan_mode == "default"
    assert serve_max_args.max is True
    assert serve_smart_args.max is False
    assert serve_smart_args.fan_mode == "smart"
    assert serve_args.stream_interval == 1
    assert serve_args.rate_limit == 0
    assert serve_args.reasoning_parser == "qwen3"
    assert serve_args.stats_footer is True
    assert serve_no_footer_args.stats_footer is False
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--max", "--fan-mode", "smart"])


def test_serve_accepts_app_runtime_mode_flags():
    parser = build_parser()

    ar_args = parser.parse_args(["serve", "--generation-mode", "ar"])
    unloaded_args = parser.parse_args(["serve", "--no-load-mtp"])

    assert ar_args.generation_mode == "ar"
    assert ar_args.load_mtp is True
    assert unloaded_args.load_mtp is False


def test_stock_ar_is_diagnostic_serve_and_bench_only():
    parser = build_parser()

    serve = parser.parse_args(["serve", "--stock-ar"])
    bench = parser.parse_args(["bench", "context", "--stock-ar", "--dry-run"])

    assert serve.stock_ar is True
    assert bench.stock_ar is True
    assert bench.bench_action == "context"
    for argv in (
        ["quickstart", "--stock-ar"],
        ["chat", "--stock-ar"],
        ["run", "hello", "--stock-ar"],
        ["start", "cli", "--stock-ar", "--dry-run"],
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)


def test_product_helper_commands_parse():
    parser = build_parser()

    start = parser.parse_args(
        ["start", "--prompt", "hello", "--max-tokens", "16", "--no-stats"]
    )
    start_openwebui = parser.parse_args(["start", "openwebui", "--port", "18012"])
    start_opencode = parser.parse_args(["start", "opencode", "--port", "18083"])
    start_swival = parser.parse_args(["start", "swival", "--port", "18084"])
    start_hermes = parser.parse_args(["start", "hermes", "--port", "18085"])
    start_openwebui_strict = parser.parse_args(
        ["start", "openwebui", "--strict-fast-path"]
    )
    quickstart = parser.parse_args(["quickstart", "--port", "18012"])
    quickstart_alias = parser.parse_args(["quick-start", "--port", "18013"])
    quickstart_dry_run = parser.parse_args(
        ["quickstart", "--dry-run", "--json", "--port", "18014"]
    )
    setup = parser.parse_args(["setup", "--dry-run"])
    pull_default = parser.parse_args(["pull"])
    ask = parser.parse_args(["ask", "hello"])
    ask_stats = parser.parse_args(["ask", "hello", "--stats"])
    serve_start = parser.parse_args(["serve", "--port", "18012"])
    tune = parser.parse_args(["tune", "--dry-run"])
    status = parser.parse_args(["status", "--deep"])
    doctor_opencode = parser.parse_args(["doctor", "opencode", "--json"])
    doctor_pi = parser.parse_args(["doctor", "pi", "--json"])
    doctor_android = parser.parse_args(
        ["doctor", "android-studio", "--port", "8008", "--json"]
    )
    connect = parser.parse_args(["connect", "openwebui", "--port", "18012"])
    connect_opencode = parser.parse_args(["connect", "opencode", "--port", "18012"])
    connect_swival = parser.parse_args(["connect", "swival", "--port", "18084"])
    models = parser.parse_args(["models", "--json"])
    report = parser.parse_args(["report", "--output-dir", "reports"])
    nightly = parser.parse_args(["bench", "nightly", "--out", "out.json"])
    suite = parser.parse_args(["bench", "suite", "--out", "suite.json"])
    bench_tune = parser.parse_args(["bench", "tune", "--dry-run"])
    nightly_json = parser.parse_args(["bench", "nightly", "--json", "--dry-run"])
    debug = parser.parse_args(["debug", "bundle", "--run-id", "debug-test"])
    hotpath = parser.parse_args(["debug", "hotpath"])
    metrics = parser.parse_args(["metrics", "watch", "--count", "1", "--json"])
    openwebui = parser.parse_args(
        ["integrate", "openwebui", "--port", "18012", "--json"]
    )
    openwebui_docker = parser.parse_args(
        ["openwebui", "docker-command", "--mtplx-port", "18012"]
    )
    claude = parser.parse_args(["integrate", "claude-code", "--port", "18012"])
    opencode = parser.parse_args(["integrate", "opencode", "--port", "18012"])
    swival = parser.parse_args(["integrate", "swival", "--port", "18084"])
    architectures = parser.parse_args(["model", "architectures", "--json"])
    qa_architectures = parser.parse_args(["model", "qa-architectures", "--json"])
    publish = parser.parse_args(
        ["model", "publish-check", "--repo-id", "mtplx/example"]
    )
    config = parser.parse_args(["config", "set", "profile", "exact", "--dry-run"])
    attribution = parser.parse_args(["profile", "eval-attribution", "--dry-run"])

    assert start.command == "start"
    assert start.prompt == "hello"
    assert start.max_tokens == 16
    assert start.show_stats is False
    assert start_openwebui.target == "openwebui"
    assert start_openwebui.port == 18012
    assert start_opencode.target == "opencode"
    assert start_opencode.port == 18083
    assert start_swival.target == "swival"
    assert start_swival.port == 18084
    assert start_hermes.target == "hermes"
    assert start_hermes.port == 18085
    assert start_openwebui.strict_fast_path is False
    assert start_openwebui_strict.strict_fast_path is True
    assert quickstart.command == "quickstart"
    assert quickstart.model == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    assert quickstart.port == 18012
    assert quickstart.profile == "sustained"
    assert quickstart_alias.command == "quick-start"
    assert quickstart_alias.model == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    assert quickstart_alias.port == 18013
    assert quickstart_alias.profile == "sustained"
    assert quickstart_dry_run.command == "quickstart"
    assert quickstart_dry_run.dry_run is True
    assert quickstart_dry_run.json is True
    assert quickstart_dry_run.port == 18014
    assert setup.command == "setup"
    assert setup.dry_run is True
    assert pull_default.command == "pull"
    assert pull_default.model == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    assert ask.command == "ask"
    assert ask.prompt_arg == "hello"
    assert ask.quiet is True
    assert ask_stats.quiet is False
    assert serve_start.command == "serve"
    assert serve_start.port == 18012
    assert serve_start.stats_footer is True
    assert tune.command == "tune"
    assert tune.model == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    assert tune.depths is None
    assert status.command == "status"
    assert status.deep is True
    assert doctor_opencode.command == "doctor"
    assert doctor_opencode.topic == "opencode"
    assert doctor_pi.command == "doctor"
    assert doctor_pi.topic == "pi"
    assert doctor_android.command == "doctor"
    assert doctor_android.topic == "android-studio"
    assert doctor_android.port == 8008
    assert connect.command == "connect"
    assert connect.integration == "openwebui"
    assert connect_opencode.integration == "opencode"
    assert connect_swival.integration == "swival"
    assert models.command == "models"
    assert report.command == "report"
    assert report.bundle is True
    assert report.deep is True
    assert nightly.bench_action == "nightly"
    assert suite.bench_action == "suite"
    assert bench_tune.bench_action == "tune"
    assert bench_tune.model == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    assert bench_tune.champion == "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed"
    assert nightly.output == "out.json"
    assert suite.output == "suite.json"
    assert nightly_json.json is True
    assert debug.debug_action == "bundle"
    assert hotpath.debug_action == "hotpath"
    assert metrics.metrics_action == "watch"
    assert openwebui.integration == "openwebui"
    assert openwebui_docker.openwebui_action == "docker-command"
    assert openwebui_docker.mtplx_port == 18012
    assert claude.integration == "claude-code"
    assert opencode.integration == "opencode"
    assert swival.integration == "swival"
    assert architectures.model_action == "architectures"
    assert qa_architectures.model_action == "qa-architectures"
    assert publish.model_action == "publish-check"
    assert config.config_action == "set"
    assert attribution.profile_action == "eval-attribution"


def test_model_architectures_json_lists_verified_and_pending(capsys):
    code = main(["model", "architectures", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    ids = {row["arch_id"] for row in payload["architectures"]}
    assert "qwen3-next-mtp" in payload["verified_runtime_arch_ids"]
    assert "deepseek-v3-mtp" in payload["verified_runtime_arch_ids"]
    assert "glm-moe-dsa-mtp" in payload["verified_runtime_arch_ids"]
    assert "glm4-moe-mtp" in payload["verified_runtime_arch_ids"]
    assert "glm4-moe-lite-mtp" in payload["verified_runtime_arch_ids"]
    assert "mimo-mtp" in payload["verified_runtime_arch_ids"]
    qwen = next(
        row for row in payload["architectures"] if row["arch_id"] == "qwen3-next-mtp"
    )
    assert "Qwen3.6" in qwen["display_name"]
    assert "qwen3_6_mtp" in qwen["aliases"]
    assert "glm4-moe-mtp" in ids
    assert "gemma-mtp" in ids


def test_model_qa_architectures_runs_contract_fixture_gates(capsys):
    code = main(["model", "qa-architectures", "--json", "--runtime-import-smoke"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "model qa-architectures"
    assert payload["passed"] is True
    assert payload["gates"]["catalog_has_main_families"] is True
    assert payload["gates"]["fixture_inspections_passed"] is True
    labels = {row["label"]: row for row in payload["fixtures"]}
    assert labels["deepseek-v3-contract-gated"]["observed"]["tier"] == "verified"
    assert (
        labels["glm4-moe-lite-contract-gated"]["observed"]["recommended_backend"]
        == "glm_mtp"
    )
    assert (
        labels["minimax-m2-num-mtp-modules-recognized-pending"]["observed"][
            "runtime_compatibility"
        ]
        == "recognized-backend-pending"
    )
    assert labels["gemma4-without-mtp-stays-no-mtp"]["observed"]["tier"] == "no-MTP"
    assert all(row["passed"] for row in payload["runtime_import_smokes"])


def test_integrate_openwebui_json(capsys):
    code = main(["integrate", "openwebui", "--port", "18012", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["integration"] == "openwebui"
    assert payload["server_url"] == "http://127.0.0.1:18012"
    assert payload["base_url"] == "http://127.0.0.1:18012/v1"
    assert payload["docker_api_base_url"] == "http://host.docker.internal:18012/v1"
    assert "host.docker.internal:18012/v1" in payload["docker_command"]
    assert (
        "OPENAI_API_BASE_URLS=http://host.docker.internal:18012/v1"
        in payload["docker_command_argv"]
    )
    assert "OPENAI_API_KEYS=$MTPLX_API_KEY" in payload["docker_command_argv"]
    assert "ENABLE_OLLAMA_API=False" in payload["docker_command_argv"]
    assert "ENABLE_TITLE_GENERATION=False" in payload["docker_command_argv"]
    assert "ENABLE_TAGS_GENERATION=False" in payload["docker_command_argv"]
    assert "ENABLE_FOLLOW_UP_GENERATION=False" in payload["docker_command_argv"]
    assert "ENABLE_AUTOCOMPLETE_GENERATION=False" in payload["docker_command_argv"]
    assert any(
        "background title/tag/follow-up/autocomplete generations" in note
        for note in payload["notes"]
    )
    assert "--no-stats-footer" in payload["server_command"]


def test_integrate_claude_code_json_uses_anthropic_root_and_auth_token(capsys):
    code = main(["integrate", "claude-code", "--port", "18012", "--json"])

    payload = json.loads(capsys.readouterr().out)
    env = payload["environment"]
    assert code == 0
    assert payload["integration"] == "claude-code"
    assert payload["base_url"] == "http://127.0.0.1:18012"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18012"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "$MTPLX_API_KEY"
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == payload["model_id"]
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == payload["model_id"]
    assert env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == payload["model_id"]
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == payload["model_id"]
    assert env["API_TIMEOUT_MS"] == "3000000"


def test_integrate_opencode_json_uses_mtplx_owned_generation_contract(capsys):
    code = main(
        [
            "integrate",
            "opencode",
            "--port",
            "18012",
            "--api-key",
            "1234",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["integration"] == "opencode"
    assert payload["api_base_url"] == "http://127.0.0.1:18012/v1"
    assert "--api-key $MTPLX_API_KEY" in payload["server_command"]
    assert "--reasoning auto" in payload["server_command"]
    model = payload["config"]["provider"]["mtplx"]["models"][payload["model_id"]]
    assert (
        payload["config"]["provider"]["mtplx"]["options"]["headers"]["x-mtplx-client"]
        == "opencode"
    )
    assert (
        payload["config"]["provider"]["mtplx"]["options"]["apiKey"] == "$MTPLX_API_KEY"
    )
    # The default model is the Qwen3.8 coding flagship: verified reasoning
    # codec (so reasoning_content round-trips), the family effort dial
    # (default medium) mirrored into options.reasoningEffort, and OpenCode's
    # built-in effort picker trimmed to the xhigh/medium/low family levels.
    assert model["reasoning"] is True
    assert model["temperature"] is True
    assert "interleaved" not in model
    assert model["options"] == {"reasoningEffort": "medium"}
    assert model["variants"] == {
        "none": {"disabled": True},
        "minimal": {"disabled": True},
        "high": {"disabled": True},
    }


def test_integrate_swival_json_emits_generic_provider_command(capsys):
    code = main(
        [
            "integrate",
            "swival",
            "--port",
            "18084",
            "--context-window",
            "131072",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["integration"] == "swival"
    assert payload["base_url"] == "http://127.0.0.1:18084"
    assert payload["api_base_url"] == "http://127.0.0.1:18084/v1"
    assert payload["context_window"] == 131072
    assert payload["command_argv"] == [
        "swival",
        "--provider",
        "generic",
        "--base-url",
        "http://127.0.0.1:18084",
        "--model",
        payload["model_id"],
        "--max-context-tokens",
        "131072",
    ]
    assert "maxTokens" not in json.dumps(payload)


def test_doctor_android_studio_json_reports_openai_compatibility(monkeypatch, capsys):
    monkeypatch.setattr(
        public,
        "_http_json",
        lambda url, timeout=15.0: {
            "object": "list",
            "data": [{"id": "mtplx-qwen36-27b-optimized-speed"}],
        },
    )
    monkeypatch.setattr(
        public,
        "_http_post_json",
        lambda url, payload, timeout=15.0: {"ok": True, "status": 200, "json": {}},
    )
    monkeypatch.setattr(
        public,
        "_http_post_text",
        lambda url, payload, timeout=15.0: {
            "ok": True,
            "status": 200,
            "preview": "data: [DONE]",
        },
    )

    code = main(["doctor", "android-studio", "--port", "8008", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    android = payload["android_studio"]
    assert android["paste_url"] == "http://127.0.0.1:8008/v1"
    assert android["url_schema"] == "OpenAI-compatible"
    assert android["model"] == "mtplx-qwen36-27b-optimized-speed"
    assert android["chat_nonstream"]["ok"] is True
    assert android["chat_stream"]["ok"] is True


def test_doctor_opencode_json_reports_provider_transport_header(
    monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "mtplx/mtplx-qwen36-27b-optimized-speed",
                "provider": {
                    "mtplx": {
                        "options": {
                            "baseURL": "http://127.0.0.1:18083/v1",
                            "headers": {"x-mtplx-client": "opencode"},
                        },
                        "models": {
                            "mtplx-qwen36-27b-optimized-speed": {
                                "reasoning": False,
                                "tool_call": True,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(config_path))
    monkeypatch.setattr(
        public,
        "_http_json",
        lambda url, timeout=1.5, api_key=None: {"ok": True, "url": url},
    )

    code = main(["doctor", "opencode", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    opencode = payload["opencode"]
    assert opencode["transport_headers"] == {"x-mtplx-client": "opencode"}
    assert opencode["mtplx_client_header_configured"] is True
    assert opencode["configured_model_id"] == "mtplx-qwen36-27b-optimized-speed"
    assert opencode["api_key_configured"] is False
    assert opencode["live_model_id"] is None
    assert opencode["model_matches_live_server"] is None
    assert opencode["deprecated_session_headers_plugin_configured"] is False
    assert opencode["session_headers_status"] == "retired"


def test_doctor_opencode_json_warns_when_config_model_is_stale(
    monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "opencode.json"
    config_path.write_text(
        json.dumps(
            {
                "model": "mtplx/gemma4-mtplx-optimized-speed",
                "provider": {
                    "mtplx": {
                        "options": {
                            "apiKey": "1234",
                            "baseURL": "http://127.0.0.1:18083/v1",
                            "headers": {"x-mtplx-client": "opencode"},
                        },
                        "models": {
                            "gemma4-mtplx-optimized-speed": {
                                "reasoning": False,
                                "tool_call": True,
                            }
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(config_path))

    seen: dict[str, str | None] = {}

    def fake_http_json(url, timeout=1.5, api_key=None):
        seen["api_key"] = api_key
        return {
            "ok": True,
            "url": url,
            "model": "mtplx-qwen36-27b-optimized-speed",
        }

    monkeypatch.setattr(
        public,
        "_http_json",
        fake_http_json,
    )

    code = main(["doctor", "opencode", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    opencode = payload["opencode"]
    assert seen["api_key"] == "1234"
    assert opencode["configured_model_id"] == "gemma4-mtplx-optimized-speed"
    assert opencode["api_key_configured"] is True
    assert opencode["live_model_id"] == "mtplx-qwen36-27b-optimized-speed"
    assert opencode["model_matches_live_server"] is False
    assert "OpenCode config points at gemma4-mtplx-optimized-speed" in (
        opencode["stale_model_warning"] or ""
    )
    assert "mtplx-qwen36-27b-optimized-speed" in (opencode["stale_model_warning"] or "")


def test_doctor_pi_json_warns_when_config_model_is_stale(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "mtplx": {
                        "baseUrl": "http://127.0.0.1:18082/v1",
                        "api": "openai-completions",
                        "apiKey": "1234",
                        "authHeader": True,
                        "headers": {"x-mtplx-client": "pi"},
                        "models": [
                            {
                                "id": "gemma4-mtplx-optimized-speed",
                                "name": "MTPLX Gemma",
                                "reasoning": True,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(config_path))

    seen: dict[str, str | None] = {}

    def fake_http_json(url, timeout=1.5, api_key=None):
        seen["api_key"] = api_key
        return {
            "ok": True,
            "url": url,
            "model": "mtplx-qwen36-27b-optimized-speed",
        }

    monkeypatch.setattr(public, "_http_json", fake_http_json)

    code = main(["doctor", "pi", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    pi = payload["pi"]
    assert seen["api_key"] == "1234"
    assert pi["model_ref"] == "mtplx/gemma4-mtplx-optimized-speed"
    assert pi["configured_model_id"] == "gemma4-mtplx-optimized-speed"
    assert pi["api_key_configured"] is True
    assert pi["auth_header"] is True
    assert pi["transport_headers"] == {"x-mtplx-client": "pi"}
    assert pi["mtplx_client_header_configured"] is True
    assert pi["live_model_id"] == "mtplx-qwen36-27b-optimized-speed"
    assert pi["model_matches_live_server"] is False
    assert "Pi config points at gemma4-mtplx-optimized-speed" in (
        pi["stale_model_warning"] or ""
    )
    assert "mtplx-qwen36-27b-optimized-speed" in (pi["stale_model_warning"] or "")


def test_doctor_model_ids_match_owner_prefixed_local_cache_alias():
    assert public._doctor_model_ids_match(
        "qwen3.5-4b-mtplx-optimized-speed",
        "youssofal-qwen3.5-4b-mtplx-optimized-speed",
    )
    assert public._doctor_model_ids_match(
        "mtplx-qwen36-27b-optimized-speed",
        "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed",
    )
    assert not public._doctor_model_ids_match(
        "gemma4-mtplx-optimized-speed",
        "youssofal-qwen3.5-4b-mtplx-optimized-speed",
    )


def test_doctor_pi_json_accepts_owner_prefixed_live_model_alias(
    monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "models.json"
    config_path.write_text(
        json.dumps(
            {
                "providers": {
                    "mtplx": {
                        "baseUrl": "http://127.0.0.1:18082/v1",
                        "api": "openai-completions",
                        "apiKey": "1234",
                        "authHeader": True,
                        "headers": {"x-mtplx-client": "pi"},
                        "models": [
                            {
                                "id": "qwen3.5-4b-mtplx-optimized-speed",
                                "name": "MTPLX Qwen 4B",
                                "reasoning": True,
                            }
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MTPLX_PI_MODELS_JSON", str(config_path))

    monkeypatch.setattr(
        public,
        "_http_json",
        lambda url, timeout=1.5, api_key=None: {
            "ok": True,
            "url": url,
            "model": "youssofal-qwen3.5-4b-mtplx-optimized-speed",
        },
    )

    code = main(["doctor", "pi", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    pi = payload["pi"]
    assert pi["configured_model_id"] == "qwen3.5-4b-mtplx-optimized-speed"
    assert pi["live_model_id"] == "youssofal-qwen3.5-4b-mtplx-optimized-speed"
    assert pi["model_matches_live_server"] is True
    assert pi["stale_model_warning"] is None


def test_config_set_dry_run_uses_selected_path(tmp_path, capsys):
    config = tmp_path / "config.toml"

    code = main(
        ["config", "set", "profile", "exact", "--config", str(config), "--dry-run"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["path"] == str(config)
    assert payload["updated"] == {"profile": "exact"}
    assert not config.exists()


def test_debug_hotpath_reports_next_kernel_boundary(capsys):
    code = main(["debug", "hotpath"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["action"] == "debug hotpath"
    names = {row["name"] for row in payload["boundaries"]}
    assert "verify_output_eval" in names
    assert "native_rowwise_mlp" in names
    assert "native_residual_mlp" in names
    assert "fused_logits_topk_distribution" in names
    assert "external_vllm_partitioned_fallback" in names
    assert payload["raw_sync_markers"]["native_mlp_is_mlx_primitive"] is True
    assert (
        "native residual MLP layer-boundary fusion" in payload["verdict"]["do_not_loop"]
    )
    assert (
        "standalone dense-logit top-k distribution kernels"
        in payload["verdict"]["do_not_loop"]
    )
    assert (
        "larger owned verify-layer or verify-cycle primitive"
        in payload["verdict"]["highest_upside_next"]
    )


def test_serve_forwards_position_ema_without_changing_none_default(monkeypatch):
    commands: list[list[str]] = []

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        commands.append(cmd)
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    parser = build_parser()
    default_args = parser.parse_args(["serve", "--model", "models/example", "--yes"])
    explicit_none_args = parser.parse_args(
        [
            "serve",
            "--model",
            "models/example",
            "--yes",
            "--adaptive-policy",
            "none",
        ]
    )
    position_args = parser.parse_args(
        [
            "serve",
            "--model",
            "models/example",
            "--yes",
            "--adaptive-policy",
            "position_ema",
            "--adaptive-position-depth-cap",
            "8",
            "--adaptive-min-depth",
            "0",
            "--qwen38-q4-mtp-block",
            "artifacts/qwen38-r17.safetensors",
        ]
    )

    assert default_args.adaptive_policy == "none"
    assert explicit_none_args.adaptive_policy == "none"
    assert position_args.adaptive_policy == "position_ema"
    assert not hasattr(position_args, "mtp_adaptive")

    for args in (default_args, explicit_none_args, position_args):
        with pytest.raises(SystemExit, match="0"):
            public.cmd_serve_public(args)

    assert "--adaptive-policy" not in commands[0]
    assert "--adaptive-policy" not in commands[1]
    assert commands[2][commands[2].index("--adaptive-policy") + 1] == "position_ema"
    assert (
        commands[2][commands[2].index("--adaptive-position-depth-cap") + 1] == "8"
    )
    assert commands[2][commands[2].index("--adaptive-min-depth") + 1] == "0"
    assert commands[2][commands[2].index("--qwen38-q4-mtp-block") + 1] == (
        "artifacts/qwen38-r17.safetensors"
    )
    assert "--qwen38-q4-mtp-block artifacts/qwen38-r17.safetensors" in (
        public._adaptive_command_suffix(position_args)
    )


def test_serve_dispatches_packaged_openai_server(monkeypatch, capsys):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["executable"] = executable
        calls["cmd"] = cmd
        calls["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key="test-key",
        rate_limit=120,
        stream_interval=4,
        max_response_tokens=512,
        temperature=0.4,
        top_p=0.9,
        adaptive_policy="expected_value",
        adaptive_min_depth=1,
        adaptive_ev_base_depth=2,
        adaptive_ev_accept_priors="0.92,0.64,0.32",
        adaptive_ev_draft_cost_s=0.0048,
        adaptive_ev_extra_verify_cost_s=0.006,
        adaptive_ev_baseline_tok_s=40.0,
        adaptive_ev_safety_margin=0.1,
        adaptive_ev_margin_center=1.0,
        adaptive_ev_margin_scale=2.0,
        adaptive_ev_confidence_weight=0.35,
        adaptive_ev_min_extra_accept_probability=0.18,
        adaptive_ev_warmup_full_depth_cycles=5,
        adaptive_ev_exploration_interval=17,
        reasoning="off",
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=8,
        strict_warmup=True,
        mtp_batch_numerics="balanced",
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr().out
    # Banner + framed status panel from the new mtplx.ui module.
    assert "MTPLX" in captured
    assert DISPLAY_VERSION in captured
    assert "127.0.0.1:8000/v1" in captured
    assert "127.0.0.1:8000/" in captured
    # Numbered handoff lines that still print before the model load.
    assert "[1/6] Server config ready" in captured
    assert "[2/6] Model resolved: models/example" in captured
    assert "[3/6] Runtime contract verified" in captured
    assert "Loading the model can take about a minute" in captured
    assert calls["cmd"][1:3] == ["-m", "mtplx.server.openai"]
    assert "--model" in calls["cmd"]
    assert calls["cmd"][calls["cmd"].index("--api-key") + 1] == "test-key"
    assert calls["cmd"][calls["cmd"].index("--rate-limit") + 1] == "120"
    assert calls["cmd"][calls["cmd"].index("--stream-interval") + 1] == "4"
    # Serial stays the default: measured 2026-07-05, serialized solo-MTP
    # beats the batched-AR lane end to end on concurrent loads because MTP
    # decode is ~4x faster per stream (see MEASUREMENTS).
    assert calls["cmd"][calls["cmd"].index("--scheduler-mode") + 1] == "serial"
    assert calls["cmd"][calls["cmd"].index("--batching-preset") + 1] == "latency"
    assert calls["cmd"][calls["cmd"].index("--mtp-batch-numerics") + 1] == ("balanced")
    assert calls["cmd"][calls["cmd"].index("--max-response-tokens") + 1] == "512"
    assert calls["cmd"][calls["cmd"].index("--adaptive-policy") + 1] == "expected_value"
    assert (
        calls["cmd"][calls["cmd"].index("--adaptive-ev-warmup-full-depth-cycles") + 1]
        == "5"
    )
    assert (
        calls["cmd"][calls["cmd"].index("--adaptive-ev-exploration-interval") + 1]
        == "17"
    )
    assert calls["cmd"][calls["cmd"].index("--model-id") + 1] == "example"
    assert calls["cmd"][calls["cmd"].index("--generation-mode") + 1] == "mtp"
    assert "--no-enable-thinking" in calls["cmd"]
    assert "--no-stats-footer" in calls["cmd"]
    assert "--strict-warmup" in calls["cmd"]


def test_serve_dispatches_step_adapter_quant_flags(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {
                "arch_id": "step3p5-mtp",
                "model_type": "step3p5",
                "recommended_backend": "step3p5_mtp",
                "compatibility": {
                    "tier": "experimental",
                    "can_run": True,
                    "exit_code": 0,
                },
            },
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["executable"] = executable
        calls["cmd"] = cmd
        calls["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            "models/Step-3.7-Flash-MTPLX-step3p5",
            "--unsafe-force-unverified",
            "--yes",
            "--verify-strategy",
            "trim_commit",
            "--verify-core",
            "stock",
            "--mtp-adapter",
            "outputs/adapters/c4-mtp-adapter-20260603-134243-r4.npz",
            "--mtp-quant-bits",
            "4",
            "--mtp-quant-group-size",
            "64",
            "--mtp-quant-mode",
            "affine",
            "--reasoning",
            "auto",
            "--reasoning-parser",
            "step3p5",
            "--reasoning-effort",
            "low",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    assert exc.value.code == 0
    assert calls["cmd"][calls["cmd"].index("--verify-strategy") + 1] == "trim_commit"
    assert calls["cmd"][calls["cmd"].index("--verify-core") + 1] == "stock"
    assert calls["cmd"][calls["cmd"].index("--backend-id") + 1] == "step3p5_mtp"
    assert (
        calls["cmd"][calls["cmd"].index("--mtp-adapter") + 1]
        == "outputs/adapters/c4-mtp-adapter-20260603-134243-r4.npz"
    )
    assert calls["cmd"][calls["cmd"].index("--mtp-quant-bits") + 1] == "4"
    assert calls["cmd"][calls["cmd"].index("--mtp-quant-group-size") + 1] == "64"
    assert calls["cmd"][calls["cmd"].index("--mtp-quant-mode") + 1] == "affine"
    assert calls["cmd"][calls["cmd"].index("--reasoning-parser") + 1] == "step3p5"
    assert calls["cmd"][calls["cmd"].index("--reasoning-effort") + 1] == "low"
    assert "--no-enable-thinking" not in calls["cmd"]


def test_serve_wildcard_host_displays_bind_and_forwards_host(monkeypatch, capsys):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="sustained",
        unsafe_force_unverified=False,
        yes=True,
        host="0.0.0.0",
        port=8000,
        depth=3,
        api_key="test-key",
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning=None,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        open_browser=False,
        stock_ar=False,
        _cli_flags={"model", "host", "api_key"},
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr().out
    assert "Listening  0.0.0.0:8000 (all interfaces)" in captured
    assert (
        "[1/6] Server config ready: listening on 0.0.0.0:8000 (all interfaces)"
        in captured
    )
    assert "Local API Base URL: http://127.0.0.1:8000/v1" in captured
    assert calls["cmd"][calls["cmd"].index("--host") + 1] == "0.0.0.0"


def test_serve_uses_quality_public_model_id_for_quality_local_path(monkeypatch):
    calls = {}
    quality_path = "/tmp/Qwen3.6-27B-MTPLX-Optimized-Quality"

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model=quality_path,
        model_id=DEFAULT_PUBLIC_MODEL_ID,
        cache_dir=None,
        profile="sustained",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning="off",
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
    )

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    assert exc.value.code == 0
    assert calls["cmd"][calls["cmd"].index("--model-id") + 1] == QUALITY_PUBLIC_MODEL_ID


def test_serve_uses_legacy_public_model_id_for_legacy_optimized_local_path(monkeypatch):
    calls = {}
    legacy_path = "/tmp/Qwen3.6-27B-MTPLX-Optimized"

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model=legacy_path,
        model_id=DEFAULT_PUBLIC_MODEL_ID,
        cache_dir=None,
        profile="sustained",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning="off",
        preserve_thinking="auto",
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
    )

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    assert exc.value.code == 0
    assert (
        calls["cmd"][calls["cmd"].index("--model-id") + 1]
        == LEGACY_OPTIMIZED_PUBLIC_MODEL_ID
    )


def test_cli_parses_api_key_file_and_kv_quant_flags():
    parser = build_parser()

    start = parser.parse_args(
        ["start", "opencode", "--api-key-file", "/tmp/key", "--kv-quant", "q8"]
    )
    quickstart = parser.parse_args(
        [
            "quickstart",
            "--api-key-file",
            "/tmp/key",
            "--paged-kv-quantization",
            "q4",
        ]
    )
    serve = parser.parse_args(
        ["serve", "--api-key-file", "/tmp/key", "--paged-kv-quant", "off"]
    )

    assert start.api_key_file == "/tmp/key"
    assert start.paged_kv_quantization == "q8"
    assert quickstart.paged_kv_quantization == "q4"
    assert serve.paged_kv_quantization == "off"


def test_quickstart_dry_run_json_previews_server_without_side_effects(
    monkeypatch, capsys
):
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: True)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )

    def fail_execvpe(*_args):
        raise AssertionError("dry-run should not exec the server")

    monkeypatch.setattr(public.os, "execvpe", fail_execvpe)

    code = main(
        [
            "quickstart",
            "--dry-run",
            "--json",
            "--model",
            "/tmp/model",
            "--yes",
            "--port",
            "18191",
            "--api-key",
            "secret-value",
            "--warmup-tokens",
            "0",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert code == 0
    assert payload["dry_run"] is True
    assert payload["command"] == "quickstart"
    assert payload["target"] == "server"
    assert payload["model"] == "/tmp/model"
    assert payload["port"] == 18191
    assert payload["api_base_url"] == "http://127.0.0.1:18191/v1"
    assert "--api-key" in payload["argv"]
    assert payload["argv"][payload["argv"].index("--api-key") + 1] == "[redacted]"
    assert "secret-value" not in payload["server_command"]


def test_config_set_show_supports_app_era_runtime_keys(tmp_path, capsys):
    config_path = tmp_path / "config.toml"

    code = main(
        [
            "config",
            "set",
            "paged_kv_quantization",
            "q8",
            "--config",
            str(config_path),
        ]
    )
    assert code == 0
    capsys.readouterr()

    code = main(["config", "show", "--config", str(config_path), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["paged_kv_quantization"] == "q8"


def test_config_set_accepts_qwen38_xhigh_reasoning_effort(tmp_path, capsys):
    config_path = tmp_path / "config.toml"

    code = main(
        [
            "config",
            "set",
            "reasoning_effort",
            "xhigh",
            "--config",
            str(config_path),
        ]
    )
    assert code == 0
    capsys.readouterr()

    code = main(["config", "show", "--config", str(config_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["reasoning_effort"] == "xhigh"

    with pytest.raises(SystemExit, match="reasoning_effort must be"):
        main(["config", "set", "reasoning_effort", "ultra", "--config", str(config_path)])


def test_public_cli_accepts_mtp_batch_scheduler_mode(tmp_path, capsys):
    serve = build_parser().parse_args(["serve", "--scheduler-mode", "mtp_batch"])
    assert serve.scheduler_mode == "mtp_batch"

    config_path = tmp_path / "config.toml"
    code = main(
        [
            "config",
            "set",
            "scheduler_mode",
            "mtp_batch",
            "--config",
            str(config_path),
        ]
    )
    assert code == 0
    capsys.readouterr()

    code = main(["config", "show", "--config", str(config_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["scheduler_mode"] == "mtp_batch"


def test_serve_threads_api_key_file_and_kv_quant_to_daemon(monkeypatch, tmp_path):
    calls: dict[str, object] = {}
    api_key_file = tmp_path / "api-key"
    api_key_file.write_text("secret-from-file\n", encoding="utf-8")

    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(public, "_print_serve_start_banner", lambda _args: None)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )

    def fake_execvpe(_executable, cmd, env):
        calls["cmd"] = cmd
        calls["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            "/tmp/model",
            "--yes",
            "--api-key-file",
            str(api_key_file),
            "--paged-kv-quantization",
            "q8",
            "--warmup-tokens",
            "0",
        ]
    )
    args._cli_flags = {
        "model",
        "yes",
        "api-key-file",
        "paged-kv-quantization",
        "warmup-tokens",
    }

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    cmd = calls["cmd"]
    env = calls["env"]
    assert exc.value.code == 0
    assert isinstance(cmd, list)
    assert "--api-key" not in cmd
    assert cmd[cmd.index("--api-key-file") + 1] == str(api_key_file)
    assert cmd[cmd.index("--paged-kv-quantization") + 1] == "q8"
    assert isinstance(env, dict)
    assert env["MTPLX_VLLM_METAL_PAGED_KV_QUANT"] == "q8"
    assert env["MTPLX_PAGED_KV_QUANT"] == "q8"


def _serve_execvpe_harness(monkeypatch):
    """Common monkeypatch set for exercising cmd_serve_public up to execvpe."""

    calls: dict[str, object] = {}
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(public, "_print_serve_start_banner", lambda _args: None)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )

    def fake_execvpe(_executable, cmd, env):
        calls["cmd"] = cmd
        calls["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    return calls


@pytest.mark.parametrize(
    "env_var",
    ["MTPLX_VLLM_METAL_PAGED_KV_QUANT", "MTPLX_PAGED_KV_QUANT"],
)
def test_serve_inherits_kv_quant_from_env_without_flag(monkeypatch, env_var):
    """App-style env KV quant survives the serve wrapper when the flag is absent.

    Regression: the wrapper used to rewrite the absent flag to an explicit
    "off", clobbering the launcher-provided env pair in both the child argv
    and the child env — the app's KV-quantization toggle was dead engine-wide.
    """

    calls = _serve_execvpe_harness(monkeypatch)
    monkeypatch.delenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", raising=False)
    monkeypatch.delenv("MTPLX_PAGED_KV_QUANT", raising=False)
    monkeypatch.setenv(env_var, "q8")

    args = build_parser().parse_args(
        ["serve", "--model", "/tmp/model", "--yes", "--warmup-tokens", "0"]
    )
    args._cli_flags = {"model", "yes", "warmup-tokens"}

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    cmd = calls["cmd"]
    env = calls["env"]
    assert exc.value.code == 0
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--paged-kv-quantization") + 1] == "q8"
    assert isinstance(env, dict)
    assert env["MTPLX_VLLM_METAL_PAGED_KV_QUANT"] == "q8"
    assert env["MTPLX_PAGED_KV_QUANT"] == "q8"


def test_serve_explicit_kv_quant_flag_beats_env(monkeypatch):
    """An explicit --paged-kv-quantization always wins over the environment."""

    calls = _serve_execvpe_harness(monkeypatch)
    monkeypatch.setenv("MTPLX_VLLM_METAL_PAGED_KV_QUANT", "q8")
    monkeypatch.setenv("MTPLX_PAGED_KV_QUANT", "q8")

    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            "/tmp/model",
            "--yes",
            "--paged-kv-quantization",
            "off",
            "--warmup-tokens",
            "0",
        ]
    )
    args._cli_flags = {"model", "yes", "paged-kv-quantization", "warmup-tokens"}

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    cmd = calls["cmd"]
    env = calls["env"]
    assert exc.value.code == 0
    assert isinstance(cmd, list)
    assert cmd[cmd.index("--paged-kv-quantization") + 1] == "off"
    assert isinstance(env, dict)
    assert env["MTPLX_VLLM_METAL_PAGED_KV_QUANT"] == "off"
    assert env["MTPLX_PAGED_KV_QUANT"] == "off"


@pytest.mark.parametrize(
    ("public_model_id", "expected_model"),
    [
        (
            QWEN35_9B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
            QWEN35_9B_OPTIMIZED_SPEED_HF_MODEL_ID,
        ),
        (
            QWEN35_9B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
            QWEN35_9B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        ),
        (
            QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
            QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
        ),
        (
            QWEN36_35B_OPTIMIZED_SPEED_FP16_PUBLIC_MODEL_ID,
            QWEN36_35B_OPTIMIZED_SPEED_FP16_HF_MODEL_ID,
        ),
        (
            QWEN36_35B_OPTIMIZED_BALANCE_PUBLIC_MODEL_ID,
            QWEN36_35B_OPTIMIZED_BALANCE_HF_MODEL_ID,
        ),
        (
            QWEN36_35B_OPTIMIZED_BALANCE_FP16_PUBLIC_MODEL_ID,
            QWEN36_35B_OPTIMIZED_BALANCE_FP16_HF_MODEL_ID,
        ),
    ],
)
def test_public_model_ids_resolve_to_hf_model_defaults(public_model_id, expected_model):
    args = SimpleNamespace(
        model_id=public_model_id,
        _cli_flags={"model-id"},
    )

    changed = public._apply_model_id_as_model_default(
        args,
        has_explicit_model=False,
    )

    assert changed is True
    assert args.model == expected_model


def test_serve_uses_measured_qwen36_35b_speed_defaults(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {
                "compatibility": {
                    "tier": "verified",
                    "can_run": True,
                    "exit_code": 0,
                    "runtime_contract": {"mtp_depth_max": 3},
                },
            },
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = build_parser().parse_args(
        [
            "serve",
            "--model",
            QWEN36_35B_OPTIMIZED_SPEED_HF_MODEL_ID,
            "--yes",
        ]
    )

    with pytest.raises(SystemExit) as exc:
        public.cmd_serve_public(args)

    cmd = calls["cmd"]
    assert exc.value.code == 0
    assert cmd[cmd.index("--depth") + 1] == "1"
    assert cmd[cmd.index("--verify-strategy") + 1] == "target_prefix"
    assert cmd[cmd.index("--draft-temperature") + 1] == "0.6"
    assert cmd[cmd.index("--draft-top-p") + 1] == "0.95"
    assert cmd[cmd.index("--draft-top-k") + 1] == "20"
    assert cmd[cmd.index("--chat-template-profile") + 1] == "local_qwen36"


def test_serve_uses_model_contract_depth_when_depth_not_explicit(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {
                "compatibility": {
                    "tier": "verified",
                    "can_run": True,
                    "exit_code": 0,
                    "runtime_contract": {"mtp_depth_max": 2},
                }
            },
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--depth") + 1] == "2"


def test_serve_no_mtp_dispatches_ar_generation_mode(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=True,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--generation-mode") + 1] == "ar"


def test_serve_native_ar_only_requires_no_mtp_and_unloads_runtime(
    monkeypatch, tmp_path, capsys
):
    inspection = {
        "model_dir": str(tmp_path),
        "architecture": "LagunaForCausalLM",
        "model_type": "laguna",
        "compatibility": {
            "tier": "AR-only",
            "arch_id": "laguna-s-2.1-ar",
            "can_run": True,
            "exit_code": 0,
            "runtime_compatibility": "native-ar-only",
            "recommended_backend": "laguna_ar",
        },
    }
    monkeypatch.setattr(public, "_serve_should_onboard", lambda _args: False)
    monkeypatch.setattr(public, "_port_is_busy", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (str(tmp_path), None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda *_args, **_kwargs: (inspection, None),
    )

    degraded = build_parser().parse_args(["serve", "--model", str(tmp_path), "--yes"])
    degraded.dry_run = True
    degraded.json = True
    assert public.cmd_serve_public(degraded) == 0
    degraded_out = capsys.readouterr().out
    assert "target-only AR architecture -> mtp_off" in degraded_out
    degraded_payload = json.loads(degraded_out[degraded_out.index("{") :])
    assert "--generation-mode ar" in degraded_payload["server_command"]
    assert "--no-load-mtp" in degraded_payload["server_command"]

    accepted = build_parser().parse_args(
        ["serve", "--model", str(tmp_path), "--yes", "--no-mtp"]
    )
    accepted.dry_run = True
    accepted.json = True
    assert public.cmd_serve_public(accepted) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "--generation-mode ar" in payload["server_command"]
    assert "--no-load-mtp" in payload["server_command"]
    assert "--backend-id laguna_ar" in payload["server_command"]
    assert "--reasoning-parser poolside_v1" in payload["server_command"]
    assert "--enable-thinking" in payload["server_command"]
    assert "--no-enable-thinking" not in payload["server_command"]
    assert "--temperature 1.0" in payload["server_command"]
    assert "--top-p 1.0" in payload["server_command"]
    assert "--backend-id qwen3_next" not in payload["server_command"]


def test_serve_generation_mode_ar_keeps_mtp_runtime_loaded(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        generation_mode="ar",
        load_mtp=True,
        stock_ar=False,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--generation-mode") + 1] == "ar"
    assert "--no-load-mtp" not in calls["cmd"]


def test_serve_rejects_mtp_generation_without_mtp_runtime(monkeypatch):
    lines = []
    monkeypatch.setattr(public, "_print_serve_start_line", lines.append)

    args = SimpleNamespace(depth=3, generation_mode="mtp", load_mtp=False)

    assert public.cmd_serve_public(args) == 2
    assert lines == [
        "error: --generation-mode mtp requires --load-mtp",
        "try: mtplx serve --generation-mode ar --no-load-mtp",
    ]


def test_serve_no_load_mtp_dispatches_unloaded_ar(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        generation_mode=None,
        load_mtp=False,
        stock_ar=False,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--generation-mode") + 1] == "ar"
    assert "--no-load-mtp" in calls["cmd"]


def test_serve_stock_ar_dispatches_unloaded_ar(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        stock_ar=True,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--generation-mode") + 1] == "ar"
    assert "--stock-ar" in calls["cmd"]


def test_quickstart_pi_passes_launch_command_to_server(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/example",
        model_id=DEFAULT_PUBLIC_MODEL_ID,
        cache_dir=None,
        profile="sustained",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        stock_ar=False,
        api_key="mtplx-local",
        rate_limit=0,
        stream_interval=1,
        context_window=262144,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        quickstart_pi=True,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert "--launch-pi" in calls["cmd"]
    assert "--server-console" in calls["cmd"]
    assert calls["cmd"][calls["cmd"].index("--preserve-thinking") + 1] == "auto"
    assert calls["cmd"][calls["cmd"].index("--context-window") + 1] == "262144"
    command = calls["cmd"][calls["cmd"].index("--pi-launch-command") + 1]
    assert command == "pi --model mtplx/example"


def test_launch_pi_in_terminal_does_not_false_positive_on_server_command(monkeypatch):
    from mtplx import pi

    calls = {}
    monkeypatch.setattr(pi.sys, "platform", "darwin")

    def fake_popen(cmd, stdout=None, stderr=None):
        calls["cmd"] = cmd
        calls["stdout"] = stdout
        calls["stderr"] = stderr

        class Proc:
            pass

        return Proc()

    monkeypatch.setattr(pi.subprocess, "Popen", fake_popen)

    result = pi.launch_pi_in_terminal(
        "pi --model mtplx/mtplx-qwen36-27b-optimized-speed",
        model_ref="mtplx/mtplx-qwen36-27b-optimized-speed",
    )

    assert result["status"] == "launched"
    assert calls["cmd"][0] == "osascript"
    script = calls["cmd"][2]
    assert "do script" in script
    assert "pi --model mtplx/mtplx-qwen36-27b-optimized-speed" in script


def test_bare_serve_invokes_server_onboarding_in_tty(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    invocations: list[dict] = []

    def fake_flow(**kwargs):
        invocations.append(kwargs)
        return {
            "model": "models/onboarded",
            "profile": "performance-cold",
            "max": False,
            "target": "server",
            "open_browser": False,
        }

    monkeypatch.setattr("mtplx.ui.onboarding.run_serve_flow", fake_flow)
    calls = {}

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/configured",
        cache_dir=None,
        download=False,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8765,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        open_browser=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert len(invocations) == 1
    assert invocations[0]["configured_model"] == "models/configured"
    assert invocations[0]["port"] == 8765
    assert args._onboarded is True
    assert args.model == "models/onboarded"
    assert calls["cmd"][calls["cmd"].index("--model") + 1] == "models/onboarded"
    assert "--open-browser" not in calls["cmd"]


def test_bare_serve_hf_choice_enables_download_and_browser(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )

    def fake_flow(**_kwargs):
        return {
            "model": "owner/repo",
            "profile": "performance-cold",
            "max": False,
            "target": "openwebui",
            "open_browser": True,
        }

    resolution_calls: list[dict] = []

    def fake_quickstart_resolve(model, *, cache_dir, download):
        resolution_calls.append({"model": model, "download": download})
        return "/tmp/runtime-model", {
            "model": model,
            "runtime_model": "/tmp/runtime-model",
            "downloaded": True,
            "download_ref": model,
        }

    monkeypatch.setattr("mtplx.ui.onboarding.run_serve_flow", fake_flow)
    monkeypatch.setattr(public, "_quickstart_resolve_model", fake_quickstart_resolve)
    calls = {}

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/configured",
        cache_dir=None,
        download=False,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8765,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        open_browser=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert resolution_calls == [{"model": "owner/repo", "download": True}]
    assert calls["cmd"][calls["cmd"].index("--model") + 1] == "/tmp/runtime-model"
    assert "--open-browser" in calls["cmd"]


def test_serve_skips_onboarding_with_explicit_model(monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )

    def fail_flow(**_kwargs):
        raise AssertionError("explicit --model should skip server onboarding")

    monkeypatch.setattr("mtplx.ui.onboarding.run_serve_flow", fail_flow)
    calls = {}

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/explicit",
        cache_dir=None,
        download=False,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        open_browser=False,
        _cli_flags={"model"},
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert calls["cmd"][calls["cmd"].index("--model") + 1] == "models/explicit"


def test_serve_never_gates_on_an_mlx_fork(monkeypatch, capsys):
    # MTPLX runs on stock PyPI MLX. The old fork gate (relax flag,
    # strict-fast-path failure, PYTHONPATH source-build autodiscovery)
    # was vestigial research residue and must never resurface in the
    # product serve path (issue #129).
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        calls["env"] = env
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    captured = capsys.readouterr().out
    assert "MLX fork" not in captured
    assert "--no-strict-mlx-fork-assert" not in calls["cmd"]
    assert "MTPLX_FAST_MLX_SOURCE_PATH_ACTIVE" not in calls["env"]


def test_serve_strict_fast_path_flag_is_an_inert_no_op(monkeypatch, capsys):
    # --strict-fast-path used to fail startup when the optional fork was
    # missing. No profile requires a fork anymore, so the flag stays
    # accepted for script compatibility but must never block a launch.
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {"compatibility": {"tier": "verified", "can_run": True, "exit_code": 0}},
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(executable, cmd, env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=True,
        max=False,
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    assert "cmd" in calls
    captured = capsys.readouterr().out
    assert "Fast MLX fork is required but not active" not in captured


def test_serve_reports_busy_port_before_model_resolution(monkeypatch, capsys):
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: True)
    from mtplx import daemon_client

    monkeypatch.setattr(
        "mtplx.daemon_client.classify_port_occupant",
        lambda _host, _port, **_kwargs: daemon_client.PortOccupant(
            kind=daemon_client.PORT_FOREIGN
        ),
    )
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (_ for _ in ()).throw(
            AssertionError("should not resolve")
        ),
    )
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=16,
        strict_warmup=False,
    )

    assert public.cmd_serve_public(args) == 2
    captured = capsys.readouterr().out
    # Banner still prints before the busy-port error path.
    assert "MTPLX" in captured
    assert DISPLAY_VERSION in captured
    assert "error: port 8000 is already in use" in captured
    # No MTPLX daemon answers on the busy port in this test, so the
    # occupant-aware copy identifies a foreign app.
    assert "Port 8000 is in use by another app (not MTPLX)." in captured
    assert "try: mtplx quickstart --profile stable --port 8001" in captured


def test_quickstart_openwebui_reuses_existing_server(monkeypatch, capsys):
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: True)
    monkeypatch.setattr(
        public,
        "_http_json",
        lambda url, timeout=15.0, **_kwargs: {
            "ok": True,
            "model": "mtplx-qwen36-27b-optimized-speed",
        },
    )
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (_ for _ in ()).throw(
            AssertionError("should not resolve")
        ),
    )
    opened = {}
    monkeypatch.setattr(
        public, "_open_browser_url", lambda url: opened.setdefault("url", url)
    )
    args = SimpleNamespace(
        model="models/example",
        model_id="mtplx-qwen36-27b-optimized-speed",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=False,
        warmup_tokens=16,
        strict_warmup=False,
        strict_fast_path=False,
        quickstart_openwebui=True,
        max=False,
    )

    assert public.cmd_serve_public(args) == 0
    captured = capsys.readouterr().out
    assert "MTPLX is already running." in captured
    assert "Chat URL: http://127.0.0.1:8000/" in captured
    assert "OpenAI API Base URL: http://127.0.0.1:8000/v1" in captured
    assert "API key: leave blank" in captured
    assert "Opening chat UI in your browser" in captured
    assert opened["url"] == "http://127.0.0.1:8000/"


def test_serve_rejects_non_localhost_without_api_key(monkeypatch, capsys):
    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (_ for _ in ()).throw(
            AssertionError("should not resolve")
        ),
    )
    args = SimpleNamespace(
        model="models/example",
        cache_dir=None,
        profile="stable",
        unsafe_force_unverified=False,
        yes=False,
        host="0.0.0.0",
        port=8000,
        depth=3,
        api_key=None,
    )

    assert public.cmd_serve_public(args) == 2
    captured = capsys.readouterr().out
    assert "--api-key or --api-key-file is required" in captured


def test_max_status_command_is_no_mlx_stable(monkeypatch, capsys):
    from mtplx import thermal

    thermal.detect_thermal_control.cache_clear()
    # Must mock both lookups: detect_thermal_control checks
    # ``~/.mtplx/bin/thermalforge`` first via ``_find_thermalforge`` so a
    # real install on the dev machine would otherwise leak in.
    monkeypatch.setattr(thermal, "_find_thermalforge", lambda: None)
    monkeypatch.setattr(thermal.shutil, "which", lambda _name: None)

    code = main(["max", "--status", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["detection"]["available"] is False
    thermal.detect_thermal_control.cache_clear()


def test_inspect_accepts_direct_model_argument():
    parser = build_parser()

    args = parser.parse_args(["inspect", "models/example", "--json"])

    assert args.command == "inspect"
    assert args.model_args == ["models/example"]
    assert args.strict_exit_code is True


def test_inspect_accepts_legacy_model_subword_form():
    parser = build_parser()

    args = parser.parse_args(["inspect", "model", "models/example", "--json"])

    assert args.command == "inspect"
    assert args.model_args == ["model", "models/example"]


def test_inspect_human_output_is_default(tmp_path, capsys):
    model = tmp_path / "plain-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type": "llama"}\n', encoding="utf-8")

    code = main(["inspect", str(model)])

    captured = capsys.readouterr().out
    assert code == 2
    assert "MTPLX inspect" in captured
    assert f"model: {model}" in captured
    assert "tier: no-MTP" in captured
    assert "can_run: false" in captured
    assert "message: Model has no MTP head." in captured


def test_start_gate_failure_is_human_readable_for_config_only_qwen(tmp_path, capsys):
    model = tmp_path / "qwen-config-only"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "num_hidden_layers": 64,
                "hidden_size": 5120,
            }
        ),
        encoding="utf-8",
    )

    code = main(["start", "cli", "--model", str(model), "--yes"])

    captured = capsys.readouterr().out
    assert code == 3
    assert "error: model cannot run with MTPLX" in captured
    assert "runtime: missing-mtp-weights" in captured
    assert "mtplx_runtime.json is optional metadata" in captured
    assert "fix: use a complete model with matching MTP weights" in captured
    assert "original source with mtplx forge" in captured
    assert "does not attach arbitrary sidecars" in captured
    assert "graft an MTP sidecar" not in captured
    assert '"model_files"' not in captured


def test_start_gate_failure_is_human_readable_for_config_only_glm(tmp_path, capsys):
    model = tmp_path / "glm-config-only"
    model.mkdir()
    (model / "config.json").write_text(
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

    code = main(["start", "cli", "--model", str(model), "--yes"])

    captured = capsys.readouterr().out
    assert code == 3
    assert "error: model cannot run with MTPLX" in captured
    assert "runtime: missing-mtp-weights" in captured
    assert "mtplx_runtime.json is optional metadata" in captured
    assert "fix: use a complete model with matching MTP weights" in captured
    assert "original source with mtplx forge" in captured
    assert "does not attach arbitrary sidecars" in captured
    assert "graft an MTP sidecar" not in captured
    assert "MTP MTP markers" not in captured
    assert '"model_files"' not in captured


def test_profiles_command_lists_default_without_mlx(capsys):
    code = main(["profiles", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["default"] == "sustained"
    assert [profile["name"] for profile in payload["profiles"]] == [
        "stable",
        "performance-cold",
        "sustained",
        "turbo",
        "exact",
        "max-diagnostic",
    ]


def test_pull_progress_json_emits_ndjson_events(tmp_path, monkeypatch, capsys):
    import mtplx.hf_loader as hf_loader

    def fake_pull_model(
        model, *, cache_dir, revision, progress_callback, progress_interval_s
    ):
        assert model == "mtplx/example"
        assert cache_dir == str(tmp_path)
        assert revision is None
        assert progress_interval_s == 0.4
        progress_callback(
            {
                "event": "start",
                "repo_id": "mtplx/example",
                "path": str(tmp_path / "mtplx--example"),
                "size_bytes": 0,
                "total_bytes": 128,
            }
        )
        progress_callback(
            {
                "event": "progress",
                "repo_id": "mtplx/example",
                "path": str(tmp_path / "mtplx--example"),
                "size_bytes": 64,
                "delta_bytes": 64,
                "rate_bps": 1024,
                "stalled_s": 0,
                "total_bytes": 128,
            }
        )
        progress_callback(
            {
                "event": "verifying",
                "repo_id": "mtplx/example",
                "path": str(tmp_path / "mtplx--example"),
                "size_bytes": 128,
                "total_bytes": 128,
            }
        )
        return {
            "ok": True,
            "repo_id": "mtplx/example",
            "path": str(tmp_path / "mtplx--example"),
            "size_bytes": 128,
        }

    monkeypatch.setattr(hf_loader, "pull_model", fake_pull_model)

    code = main(
        [
            "pull",
            "mtplx/example",
            "--cache-dir",
            str(tmp_path),
            "--progress-json",
        ]
    )

    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert code == 0
    assert [event["event"] for event in events] == [
        "resolving",
        "start",
        "progress",
        "verifying",
        "result",
    ]
    assert events[2]["rate_bps"] == 1024
    assert events[-1]["ok"] is True


_PULL_OFFLINE_ERROR = (
    "An error happened while trying to locate the file on the Hub and we "
    "cannot find the requested files in the local cache. Please check your "
    "connection and try again."
)


@pytest.mark.parametrize(
    ("error_text", "hf_endpoint", "expect_hint"),
    [
        # huggingface_hub's offline error, no mirror configured -> name the knob.
        (_PULL_OFFLINE_ERROR, None, True),
        ("HTTPSConnectionPool(host='huggingface.co'): Max retries exceeded", None, True),
        # A mirror is already configured: the mirror is what failed, no hint.
        (_PULL_OFFLINE_ERROR, "https://hf-mirror.com", False),
        # Not network-shaped (placeholder repo, #258): no hint.
        ("downloaded model is incomplete: weight shards are missing or still partial", None, False),
    ],
)
def test_pull_failure_names_hf_mirror_only_for_network_failures(
    tmp_path, monkeypatch, capsys, error_text, hf_endpoint, expect_hint
):
    import mtplx.hf_loader as hf_loader

    if hf_endpoint is None:
        monkeypatch.delenv("HF_ENDPOINT", raising=False)
    else:
        monkeypatch.setenv("HF_ENDPOINT", hf_endpoint)

    def failing_pull_model(model, **kwargs):
        raise RuntimeError(error_text)

    monkeypatch.setattr(hf_loader, "pull_model", failing_pull_model)

    code = main(["pull", "mtplx/example", "--cache-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1
    assert "error: pull failed" in out
    assert f"detail: {error_text}" in out
    assert ("hint: " in out) is expect_hint
    if expect_hint:
        assert "HF_ENDPOINT=https://hf-mirror.com" in out

    code = main(["pull", "mtplx/example", "--cache-dir", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["error"] == "pull failed"
    assert payload["detail"] == error_text
    assert ("hint" in payload) is expect_hint


def test_model_cache_commands_parse():
    parser = build_parser()

    pull_args = parser.parse_args(["pull", "mtplx/example", "--revision", "main"])
    pull_progress_args = parser.parse_args(["pull", "mtplx/example", "--progress-json"])
    list_args = parser.parse_args(["list", "--cache-dir", "/tmp/mtplx-models"])
    remove_args = parser.parse_args(["remove", "mtplx/example", "--missing-ok"])

    assert pull_args.command == "pull"
    assert pull_args.model == "mtplx/example"
    assert pull_args.revision == "main"
    assert pull_progress_args.progress_json is True
    assert list_args.command == "list"
    assert list_args.cache_dir == "/tmp/mtplx-models"
    assert remove_args.command == "remove"
    assert remove_args.missing_ok is True


def test_init_parser_exposes_model_cache_and_profile_options():
    parser = build_parser()

    args = parser.parse_args(
        [
            "init",
            "--model",
            "mtplx/example",
            "--model-dir",
            "/tmp/mtplx-models",
            "--profile",
            "exact",
            "--thermal-control",
            "none",
            "--download",
            "--write",
        ]
    )

    assert args.command == "init"
    assert args.model == "mtplx/example"
    assert args.model_dir == "/tmp/mtplx-models"
    assert args.profile == "exact"
    assert args.thermal_control == "none"
    assert args.download is True
    assert args.write is True


def test_compile_audit_dry_run_is_real_command(capsys):
    code = main(
        [
            "profile",
            "compile-audit",
            "--prefill-chunks",
            "128,256",
            "--skip-verify",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr().out
    assert code == 0
    assert '"action": "profile compile-audit"' in captured
    assert "probe_mx_compile_buckets.py" in captured
    assert "--prefill-chunks" in captured
    assert '"MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged"' in captured
    assert '"attention_impl": "mlx_vector_paged"' in captured


def test_eval_attribution_dry_run_is_real_command(capsys):
    code = main(
        [
            "profile",
            "eval-attribution",
            "--prefix-tokens",
            "64",
            "--orders",
            "outputs,recurrent;recurrent,outputs",
            "--dry-run",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "profile eval-attribution"
    from mtplx.commands.public import _research_script

    if _research_script("probe_eval_attribution.py") is None:
        # This repo does not ship the research-workspace probe script; the
        # command must say so honestly instead of printing a command whose
        # script path does not exist (dry-run included).
        assert code == 2
        assert payload["available"] is False
        assert "probe_eval_attribution.py" in payload["reason"]
        assert payload["hint"]
    else:
        assert code == 0
        assert "probe_eval_attribution.py" in " ".join(payload["command"])
        assert "--prefix-tokens" in payload["command"]
        assert "outputs,recurrent;recurrent,outputs" in payload["command"]
        assert "larger owned kernel boundary" in payload["purpose"]


@pytest.mark.parametrize("action", ["compile-audit", "eval-attribution"])
def test_mtp_only_profile_commands_reject_laguna_ar(
    monkeypatch,
    capsys,
    action,
):
    inspection = {
        "compatibility": {
            "can_run": True,
            "runtime_compatibility": "native-ar-only",
        }
    }
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda *_args, **_kwargs: (inspection, None),
    )

    code = main(["profile", action, "--model", "/models/laguna", "--dry-run"])

    assert code == 2
    assert "requires an MTP-capable runtime" in capsys.readouterr().out


def test_model_gate_error_lines_render_for_every_tier():
    # Regression for #98: the unverified-tier branch crashed with a
    # NameError instead of printing the gate explanation.
    from mtplx.commands.public import _model_gate_error_lines

    tiers = (
        "verified",
        "architecture-compatible-but-unverified",
        "incompatible-architecture",
        "no-MTP",
        None,
    )
    for tier in tiers:
        inspection = {
            "model": "/tmp/example",
            "compatibility": {
                "tier": tier,
                "can_run": False,
                "message": "example reason",
            },
            "mtp": {"exists": True, "tensor_count": 3, "passes_tensor_gate": True},
            "mtp_num_hidden_layers": 1,
        }
        lines = _model_gate_error_lines(inspection)
        assert lines, tier
        assert any("try:" in line for line in lines), (tier, lines)


# Issue #131: the Hermes profile config was regenerated from the template on
# every sync, silently dropping user-added sections (memory/providers/…) and
# child keys (model.max_tokens). Mirrors the Swift-side tests — SYNC PAIR
# with HermesIntegration.mergedConfigYAML.


def test_hermes_merged_config_owns_template_shaped_sections():
    template = (
        "model:\n"
        '  default: "m"\n'
        "  provider: custom\n"
        '  base_url: "http://127.0.0.1:9001/v1"\n'
        "toolsets:\n"
        "  - terminal\n"
        "  - file\n"
        "display:\n"
        "  streaming: true\n"
    )
    existing = (
        "# user preamble comment\n"
        "model:\n"
        '  default: "old"\n'
        "  provider: custom\n"
        '  base_url: "http://127.0.0.1:8123/v1"\n'
        "  max_tokens: 32768\n"
        '  reasoning_effort: "high"\n'
        "toolsets:\n"
        "  - terminal\n"
        "  - custom-extra\n"
        "# external memory\n"
        "memory:\n"
        "  provider: honcho\n"
    )

    merged = public._hermes_merged_config_yaml(existing, template)

    assert merged.startswith("# user preamble comment\n")
    assert 'base_url: "http://127.0.0.1:9001/v1"' in merged
    assert "8123" not in merged
    assert "  max_tokens: 32768" in merged
    # Conditionally app-owned (the app writes it while an effort is set):
    # a stale line must not be resurrected as user content.
    assert "reasoning_effort" not in merged
    # Sequence-shaped owned sections are rewritten wholly.
    assert "custom-extra" not in merged
    # Unknown sections survive verbatim, with their comments.
    assert "# external memory\nmemory:\n  provider: honcho" in merged
    # Idempotent: merging the merge result changes nothing.
    assert public._hermes_merged_config_yaml(merged, template) == merged


def test_hermes_merged_config_without_existing_is_template():
    template = 'model:\n  default: "m"\n'
    assert public._hermes_merged_config_yaml(None, template) == template
    assert public._hermes_merged_config_yaml("  \n", template) == template


def _hermes_template_blocks(template):
    _preamble, blocks, _trailing = public._hermes_parse_top_level_blocks(template)
    return {block["key"]: "\n".join(block["lines"]) for block in blocks}


def test_hermes_config_yaml_identity_header_and_agent_effort():
    template = public._hermes_config_yaml(
        model_id="m",
        base_url="http://127.0.0.1:9001/v1",
        api_key="k",
        workspace_path="/ws",
        reasoning_effort="high",
    )
    by_key = _hermes_template_blocks(template)
    # model.default_headers is the only client-side identity hook hermes
    # exposes; every hermes-conditional server branch keys on it.
    assert "  default_headers:\n    x-mtplx-client: hermes" in by_key["model"]
    # hermes reads CLI_CONFIG["agent"]["reasoning_effort"]; under model: the
    # key is silently ignored.
    assert "  reasoning_effort: 'high'" in by_key["agent"]
    assert "reasoning_effort" not in by_key["model"]

    effortless = public._hermes_config_yaml(
        model_id="m",
        base_url="http://127.0.0.1:9001/v1",
        api_key="k",
        workspace_path="/ws",
    )
    assert "reasoning_effort" not in effortless
    assert "x-mtplx-client: hermes" in effortless


def test_hermes_merge_moves_stale_model_effort_under_agent():
    template = public._hermes_config_yaml(
        model_id="m",
        base_url="http://127.0.0.1:9001/v1",
        api_key="k",
        workspace_path="/ws",
        reasoning_effort="low",
    )
    stale = (
        "model:\n"
        "  default: 'old'\n"
        "  provider: custom\n"
        "  reasoning_effort: 'high'\n"
        "agent:\n"
        "  system_prompt: 'old'\n"
    )
    merged = public._hermes_merged_config_yaml(stale, template)
    by_key = _hermes_template_blocks(merged)
    assert "reasoning_effort" not in by_key["model"]
    assert by_key["agent"].count("reasoning_effort") == 1
    assert "  reasoning_effort: 'low'" in by_key["agent"]
    # Idempotent with the nested default_headers child in place.
    assert public._hermes_merged_config_yaml(merged, template) == merged


def test_hermes_client_reasoning_effort_excludes_auto():
    from types import SimpleNamespace

    effort = public._hermes_client_reasoning_effort
    assert effort(SimpleNamespace(reasoning_effort="xhigh")) == "xhigh"
    # "auto" is the server-resolved sentinel; hermes warns and falls back to
    # medium on unknown ladder values, so it must never be written.
    assert effort(SimpleNamespace(reasoning_effort="auto")) is None
    assert effort(SimpleNamespace(reasoning_effort=None)) is None
    assert effort(SimpleNamespace()) is None


def test_sync_hermes_profile_preserves_user_sections(monkeypatch, tmp_path):
    monkeypatch.setattr(public, "_hermes_home", lambda: tmp_path / ".hermes")

    first = public._sync_hermes_profile(
        model_id="mtplx-qwen36-27b-optimized-speed",
        base_url="http://127.0.0.1:8123/v1",
        api_key="local",
        workspace_path=str(tmp_path / "ws"),
    )
    config_path = Path(first["config_path"])

    # The user adds non-template config — the issue #131 repro.
    text = config_path.read_text(encoding="utf-8")
    text = text.replace(
        "  api_mode: chat_completions\n",
        "  api_mode: chat_completions\n  max_tokens: 32768\n",
    )
    text += "# external memory (issue #131 repro)\nmemory:\n  provider: honcho\n"
    config_path.write_text(text, encoding="utf-8")

    # The next sync must preserve all of it.
    public._sync_hermes_profile(
        model_id="mtplx-qwen36-27b-optimized-speed",
        base_url="http://127.0.0.1:8123/v1",
        api_key="local",
        workspace_path=str(tmp_path / "ws"),
    )
    preserved = config_path.read_text(encoding="utf-8")
    assert (
        "# external memory (issue #131 repro)\nmemory:\n  provider: honcho" in preserved
    )
    assert "  max_tokens: 32768" in preserved

    # A repeat sync with unchanged inputs must not rewrite the file.
    repeated = public._sync_hermes_profile(
        model_id="mtplx-qwen36-27b-optimized-speed",
        base_url="http://127.0.0.1:8123/v1",
        api_key="local",
        workspace_path=str(tmp_path / "ws"),
    )
    assert repeated["did_change"] is False

    # Owned keys update in place while user content survives.
    public._sync_hermes_profile(
        model_id="mtplx-qwen36-27b-optimized-speed",
        base_url="http://127.0.0.1:9001/v1",
        api_key="local",
        workspace_path=str(tmp_path / "ws"),
    )
    final = config_path.read_text(encoding="utf-8")
    assert "9001/v1" in final
    assert "8123/v1" not in final
    assert "memory:\n  provider: honcho" in final
    assert "  max_tokens: 32768" in final


def test_ui_pretty_path_is_importable():
    # Regression: ``mtplx.ui`` must export ``pretty_path``. The dashboard
    # handoff does ``from mtplx.ui import pretty_path``, but the symbol lived
    # in ``onboarding`` as ``_pretty_path`` and was never re-exported, so the
    # import raised ImportError.
    from mtplx import ui
    from mtplx.ui import pretty_path

    assert "pretty_path" in ui.__all__
    assert pretty_path is ui.onboarding._pretty_path
    # Behaves like the underlying helper: collapse $HOME, pass HF refs through.
    assert pretty_path("Youssofal/Qwen3.6-27B") == "Youssofal/Qwen3.6-27B"
    assert pretty_path(str(Path.home() / "models" / "x")) == "~/models/x"


def test_quickstart_dashboard_handoff_does_not_crash(capsys):
    # Regression: ``mtplx start`` reached the dashboard handoff and crashed on
    # ``from mtplx.ui import pretty_path``. Drive the handoff directly and
    # assert it prints the loading line with the model path instead of raising.
    args = SimpleNamespace(host="127.0.0.1", port=8000)
    public._quickstart_print_dashboard_handoff(
        args, runtime_model=str(Path.home() / ".mtplx" / "models" / "demo")
    )
    out = capsys.readouterr().out
    assert "Loading model: ~/.mtplx/models/demo" in out
    assert "Dashboard URL:" in out


def test_model_contract_depth_uses_ceiling_when_no_default_declared():
    """``mtp_depth_max`` alone still selects the ceiling (unchanged behaviour)."""
    contract = {
        "public_model_id": "some-third-party-artifact",
        "mtp_depth_max": 3,
    }
    inspection = {"compatibility": {"runtime_contract": contract}}
    sustained = public.get_profile("sustained")

    assert public._model_contract_depth(inspection, profile=sustained, fallback=1) == 3


def test_model_contract_depth_prefers_declared_default_over_ceiling():
    """An artifact may declare a shallower default than its sidecar ceiling."""
    contract = {
        "public_model_id": "some-third-party-artifact",
        "mtp_depth_max": 3,
        "mtp_depth_default": 1,
    }
    inspection = {"compatibility": {"runtime_contract": contract}}
    sustained = public.get_profile("sustained")

    assert public._model_contract_depth(inspection, profile=sustained, fallback=3) == 1


def test_model_contract_depth_clamps_declared_default_to_ceiling():
    """A declared default never exceeds the artifact's supported depth."""
    contract = {
        "public_model_id": "some-third-party-artifact",
        "mtp_depth_max": 2,
        "mtp_depth_default": 5,
    }
    inspection = {"compatibility": {"runtime_contract": contract}}
    sustained = public.get_profile("sustained")

    assert public._model_contract_depth(inspection, profile=sustained, fallback=3) == 2


def test_qwen36_35b_a3b_defaults_to_measured_best_depth():
    """A3B measured best is D2 (M5 Max tune 145.0 tok/s, 1.54x; the reporter's
    sweep has D2 within 2% of best) while the D3 ceiling loses 9-22%."""
    contract = {
        "public_model_id": public.QWEN36_35B_OPTIMIZED_SPEED_PUBLIC_MODEL_ID,
        "recommended_profile": "sustained",
        "mtp_depth_max": 3,
    }
    inspection = {"compatibility": {"runtime_contract": contract}}
    sustained = public.get_profile("sustained")

    assert public._model_contract_depth(inspection, profile=sustained, fallback=3) == 2


def test_serve_parser_accepts_draft_core():
    from mtplx.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["serve", "--model", "m", "--draft-core", "device"])
    assert args.draft_core == "device"
    default = parser.parse_args(["serve", "--model", "m"])
    assert default.draft_core == "stock"


def test_serve_missing_mtp_degrade_reaches_child_argv(monkeypatch):
    """The missing-MTP degrade must flip the CHILD's --generation-mode too.

    Regression: the serve path snapshotted generation_mode before
    _apply_runtime_compatibility_mode mutated args, so a trunk without MTP
    heads spawned a child with the stale "--generation-mode mtp" while
    load_mtp had already been stripped — ServerState then refused with
    "--generation-mode mtp requires --load-mtp" (first live Bonsai serve).
    """
    calls = {}

    monkeypatch.setattr(
        public,
        "_resolve_runtime_model_path",
        lambda model, cache_dir=None: (model, None),
    )
    monkeypatch.setattr(
        public,
        "_model_gate",
        lambda model, unsafe_force_unverified=False, yes=False: (
            {
                "compatibility": {
                    "tier": "architecture-compatible-but-unverified",
                    "can_run": True,
                    "exit_code": 0,
                    "runtime_compatibility": "native-ar-only-missing-mtp",
                }
            },
            None,
        ),
    )
    monkeypatch.setattr(public, "_port_is_busy", lambda host, port: False)

    def fake_execvpe(_executable, cmd, _env):
        calls["cmd"] = cmd
        raise SystemExit(0)

    monkeypatch.setattr(public.os, "execvpe", fake_execvpe)
    args = SimpleNamespace(
        command="serve",
        model="models/bonsai-trunk-no-mtp",
        cache_dir=None,
        profile="performance-cold",
        unsafe_force_unverified=False,
        yes=True,
        host="127.0.0.1",
        port=8000,
        depth=3,
        no_mtp=False,
        generation_mode="mtp",
        load_mtp=True,
        stock_ar=False,
        api_key=None,
        rate_limit=0,
        stream_interval=1,
        max_response_tokens=None,
        temperature=0.6,
        top_p=0.95,
        reasoning_parser="qwen3",
        stats_footer=True,
        warmup_tokens=0,
        strict_warmup=False,
        strict_fast_path=False,
        max=False,
        _cli_flags=set(),
    )

    try:
        public.cmd_serve_public(args)
    except SystemExit as exc:
        assert exc.code == 0

    cmd = calls["cmd"]
    assert cmd[cmd.index("--generation-mode") + 1] == "ar"
    assert "--no-load-mtp" in cmd
    assert cmd[cmd.index("--depth") + 1] == "0"


def test_pi_models_config_sync_preserves_user_edits_in_mtplx_block(tmp_path):
    """#282 (intensifi): a re-sync must not clobber user edits inside the
    MTPLX provider block. Connection identity is corrected; everything the
    user changed or added wins."""
    from mtplx.pi import write_pi_models_config

    config_path = tmp_path / "models.json"
    write_pi_models_config(
        base_url="http://127.0.0.1:18012/v1",
        model_id="mtplx-test-model",
        path=config_path,
    )

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model = payload["providers"]["mtplx"]["models"][0]
    # The user teaches the model vision, tunes the thinking map, sets a cap,
    # renames it, and adds a private header plus their own second model.
    model["input"] = ["text", "image"]
    model["thinkingLevelMap"] = {
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "xhigh": "xhigh",
    }
    model["maxTokens"] = 20_000
    model["name"] = "My Local Qwen"
    payload["providers"]["mtplx"]["headers"]["x-user-header"] = "kept"
    payload["providers"]["mtplx"]["models"].append({"id": "user-second-model"})
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    # Next launch lands on a NEW port: connection identity must update while
    # every user edit survives byte-level.
    write_pi_models_config(
        base_url="http://127.0.0.1:19099/v1",
        model_id="mtplx-test-model",
        path=config_path,
    )

    merged = json.loads(config_path.read_text(encoding="utf-8"))
    provider = merged["providers"]["mtplx"]
    assert provider["baseUrl"] == "http://127.0.0.1:19099/v1"
    assert provider["headers"]["x-mtplx-client"] == "pi"
    assert provider["headers"]["x-user-header"] == "kept"
    merged_model = provider["models"][0]
    assert merged_model["input"] == ["text", "image"]
    assert merged_model["thinkingLevelMap"] == {
        "minimal": None,
        "low": "low",
        "medium": "medium",
        "xhigh": "xhigh",
    }
    assert merged_model["maxTokens"] == 20_000
    assert merged_model["name"] == "My Local Qwen"
    assert provider["models"][1]["id"] == "user-second-model"


def test_pi_request_policy_extension_respects_user_ownership(tmp_path):
    """A user who replaces the managed extension with their own content owns
    the file: MTPLX never overwrites it again."""
    from mtplx.pi import write_pi_models_config

    config_path = tmp_path / "models.json"
    write_pi_models_config(
        base_url="http://127.0.0.1:18012/v1",
        model_id="mtplx-test-model",
        path=config_path,
    )
    extension_path = config_path.parent / "extensions" / "mtplx-request-policy.ts"
    managed_source = extension_path.read_text(encoding="utf-8")
    assert "MTPLX-managed" in managed_source

    user_source = "export default function (pi) {}\n"
    extension_path.write_text(user_source, encoding="utf-8")

    write_pi_models_config(
        base_url="http://127.0.0.1:19099/v1",
        model_id="mtplx-test-model",
        path=config_path,
    )
    assert extension_path.read_text(encoding="utf-8") == user_source

    # A managed copy (marker intact) keeps receiving updates.
    extension_path.write_text(managed_source, encoding="utf-8")
    write_pi_models_config(
        base_url="http://127.0.0.1:19099/v1",
        model_id="renamed-model",
        path=config_path,
    )
    assert 'const mtplxModelID = "renamed-model"' in extension_path.read_text(
        encoding="utf-8"
    )
