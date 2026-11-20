"""One resolver, three entry points (PX.1, 2026-09-18)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from mtplx import launch_lane as ll
from mtplx.cli import build_parser, main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "apps" / "MTPLXApp" / "Tests" / "Fixtures" / "launch_lane_matrix.json"
FAMILIES = ("qwen4_exp", "qwen3_8")


# --- the table-driven parity test -------------------------------------------


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("client", ll.CLIENTS)
@pytest.mark.parametrize("seat_gib", ll.SEATS_GIB)
def test_every_entry_point_resolves_the_same_lane_except_the_declared_list(
    client, family, seat_gib
):
    lanes = {
        entry: ll.comparable_lane(
            ll.resolve_launch_lane(
                client=client,
                family=family,
                memory_bytes=seat_gib * ll.GIB,
                entry_point=entry,
            )
        )
        for entry in ll.ENTRY_POINTS
    }
    reference = lanes["app"]
    for entry, lane in lanes.items():
        for key in reference:
            if (client, key) in ll.DECLARED_ENTRY_POINT_DIFFERENCES:
                continue
            assert lane[key] == reference[key], (client, family, seat_gib, entry, key)


@pytest.mark.parametrize("entry", ll.ENTRY_POINTS)
@pytest.mark.parametrize("client", ll.CLIENTS)
def test_every_client_and_entry_point_continues_in_the_conversation_lease(
    monkeypatch, client, entry
):
    # 12 Nov: the coding-agent lane exported the switch that lets an exact
    # follow-up take its conversation's lease, so the app's agents resumed in
    # place while bare `mtplx serve`, chat and Open WebUI copied the whole
    # conversation on every follow-up (4.24 GB at 131K). Same restore for
    # every client from every entry point, whatever its env block says.
    from mtplx import generation

    for name in (
        "MTPLX_SESSION_LIVE_FRONTIER_REFERENCE_RESTORE",
        "MTPLX_OPENCODE_TOOL_HISTORY_LIVE_FRONTIER",
    ):
        monkeypatch.delenv(name, raising=False)
    lane = ll.resolve_launch_lane(
        client=client, family="qwen4_exp", memory_bytes=128 * ll.GIB, entry_point=entry
    )
    for key, value in lane["env"].items():
        monkeypatch.setenv(key, str(value))
    assert generation._session_live_frontier_reference_restore_enabled(), (client, entry)


def test_the_declared_list_is_short_named_and_real():
    assert set(ll.DECLARED_ENTRY_POINT_DIFFERENCES) == {
        ("chat", "ssd_session_cache"),
        ("openwebui", "ssd_session_cache"),
        ("pi", "ssd_session_cache"),
    }
    for (client, key), why in ll.DECLARED_ENTRY_POINT_DIFFERENCES.items():
        assert "cell" in why  # each names the measurement that settles it
        values = {
            ll.comparable_lane(
                ll.resolve_launch_lane(
                    client=client, family="qwen4_exp", memory_bytes=128 * ll.GIB,
                    entry_point=entry,
                )
            )[key]
            for entry in ll.ENTRY_POINTS
        }
        assert len(values) > 1, (client, key)  # a declared difference that is none is stale


# --- the disagreements the resolver settled ---------------------------------


def test_pi_is_serial_from_every_entry_point_behind_one_constant():
    assert ll.PI_SCHEDULER_MODE == "serial"
    for entry in ll.ENTRY_POINTS:
        lane = ll.resolve_launch_lane(
            client="pi", family="qwen3_8", memory_bytes=64 * ll.GIB, entry_point=entry
        )
        assert lane["choices"]["scheduler_mode"]["value"] == ll.PI_SCHEDULER_MODE
        assert lane["choices"]["batching_preset"]["value"] == "latency"
        assert lane["choices"]["max_active_requests"]["value"] is None
    assert ll.cli_start_defaults("pi")["scheduler_mode"] == ll.PI_SCHEDULER_MODE


def test_ssd_cap_is_auto_everywhere_and_tiers_by_ram():
    for client in ll.CLIENTS:
        assert ll.cli_start_defaults(client)["ssd_session_cache_max_size"] == "auto"
        assert ll.cli_start_defaults(client)["ssd_session_cache_min_prefix_tokens"] == 512
    assert [ll.ssd_cap_bytes_for_seat(s * ll.GIB) // ll.GIB for s in ll.SEATS_GIB] == [
        16, 24, 32, 32, 32, 100, 100,
    ]


def test_ssd_cap_mirror_matches_the_cold_tier(monkeypatch):
    from mtplx.cache_bank import cold_tier

    for seat in ll.SEATS_GIB:
        monkeypatch.setattr(cold_tier, "detect_total_ram_bytes", lambda s=seat: s * ll.GIB)
        monkeypatch.setattr(
            cold_tier.shutil, "disk_usage",
            lambda _p: type("U", (), {"free": 100 * ll.GIB})(),
        )
        assert cold_tier.default_cold_tier_max_bytes() == ll.ssd_cap_bytes_for_seat(
            seat * ll.GIB, free_disk_bytes=100 * ll.GIB
        )


def test_prefill_chunk_is_never_a_launch_flag():
    for client in ll.CLIENTS:
        assert ll.cli_start_defaults(client)["prefill_chunk_tokens"] is None
        lane = ll.resolve_launch_lane(
            client=client, family="qwen4_exp", memory_bytes=128 * ll.GIB
        )
        chunk = lane["choices"]["prefill_chunk_tokens"]
        assert chunk["source"] == "family" and chunk["value"] == 2048
        assert "inherited-27B-era, unmeasured" in chunk["reason"]


def test_depth_policy_priors_are_one_set_everywhere():
    from mtplx.server import openai

    priors = ll.depth_policy_priors("qwen3_8")
    cli = build_parser().parse_args(["start", "hermes", "--dry-run"])
    server = openai.parse_args(["--model", "models/example"])
    for namespace in (cli, server):
        assert namespace.adaptive_ev_draft_cost_s == priors["draft_cost_ms"] / 1000
        assert namespace.adaptive_ev_extra_verify_cost_s == priors["verify_extra_cost_ms"] / 1000
        assert namespace.adaptive_ev_baseline_tok_s == priors["baseline_tok_s"]
    # The stale pair is gone from every forwarder.
    source = (ROOT / "mtplx" / "commands" / "public.py").read_text()
    assert "0.0048" not in source and '"adaptive_ev_extra_verify_cost_s", 0.006' not in source


def test_hermes_env_is_the_shared_block_without_the_read_inspection_limits(monkeypatch):
    from mtplx.commands import public

    monkeypatch.setattr(
        public, "_detect_total_ram_bytes_for_opencode_defaults", lambda: 128 * ll.GIB
    )
    env: dict[str, str] = {}
    public._apply_hermes_memory_env_defaults(env)
    assert env == ll.agent_env_block("hermes", 128 * ll.GIB)
    assert env["MTPLX_CLIENT"] == "hermes"
    assert not any("READ_INSPECTION" in key or "READ_ONLY_INSPECTION" in key for key in env)
    opencode = public._opencode_memory_env_defaults()
    assert opencode == ll.agent_env_block("opencode", 128 * ll.GIB)
    assert {k: v for k, v in env.items() if k != "MTPLX_CLIENT"} == opencode


@pytest.mark.parametrize(("seat_gib", "entries"), [(16, "6"), (64, "6"), (96, "32"), (128, "32")])
def test_agent_env_block_scales_bank_entries_with_ram(seat_gib, entries):
    for client in ("opencode", "pi", "hermes"):
        block = ll.agent_env_block(client, seat_gib * ll.GIB)
        assert block["MTPLX_SESSION_BANK_MAX_ENTRIES"] == entries
    assert ll.agent_env_block("chat", seat_gib * ll.GIB) == {}


# --- the depth policy: the engine resolves it --------------------------------


def test_expected_value_on_a_one_depth_family_resolves_to_static_depth_3():
    resolved = ll.resolve_depth_policy(
        "qwen4_exp", requested_policy="expected_value", requested_depth=3
    )
    assert resolved["policy"] == "none" and resolved["depth"] == 3
    assert resolved["demoted"] is True
    assert resolved["reason"].startswith("adaptive depth has no effect on this model yet")
    # A deeper ceiling with a policy still lands on the compiled depth.
    deep = ll.resolve_depth_policy(
        "qwen4_exp", requested_policy="expected_value", requested_depth=5
    )
    assert deep["depth"] == 3
    # The 27B keeps its policy, and a static request is nobody's business.
    assert ll.resolve_depth_policy(
        "qwen3_8", requested_policy="expected_value", requested_depth=3
    )["policy"] == "expected_value"
    static = ll.resolve_depth_policy("qwen4_exp", requested_policy="none", requested_depth=5)
    assert static["demoted"] is False and static["depth"] == 5


def test_the_rule_lifts_by_itself_when_more_widths_are_compiled(monkeypatch):
    from dataclasses import replace

    from mtplx.backends import family_settings as fs

    block = dict(fs.QWEN4_EXP_SETTINGS)
    block["compiled_verify_depths"] = replace(
        block["compiled_verify_depths"], value=None
    )
    monkeypatch.setitem(fs.FAMILY_SETTINGS, "qwen4_exp", block)
    resolved = ll.resolve_depth_policy(
        "qwen4_exp", requested_policy="expected_value", requested_depth=5
    )
    assert resolved["policy"] == "expected_value" and resolved["demoted"] is False


def _served_args(tmp_path, config, **overrides):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    from mtplx.server import openai

    args = openai.parse_args(
        ["--model", str(model), "--adaptive-policy", "expected_value", "--depth", "3"]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


FLASH_NEXT_CONFIG = {"model_type": "qwen4_exp", "text_config": {"model_type": "qwen4_exp_text"}}


def test_the_server_builds_no_policy_object_for_flash_next(tmp_path):
    from mtplx.server import openai

    args = _served_args(tmp_path, FLASH_NEXT_CONFIG)
    assert openai._make_adaptive_policy(args, max_depth=3) is None
    config = openai._adaptive_config(args, max_depth=3)
    assert config["policy"] == "none"
    assert config["requested_policy"] == "expected_value"
    assert config["static_depth"] == 3
    assert "adaptive depth has no effect on this model yet" in config["reason"]


def test_the_server_keeps_the_policy_for_the_27b(tmp_path):
    from mtplx.adaptive import ExpectedValueDepthPolicy
    from mtplx.server import openai

    model = tmp_path / "Qwen3.8-27B-MTPLX-Optimized-Speed"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    args = openai.parse_args(
        ["--model", str(model), "--adaptive-policy", "expected_value", "--depth", "3"]
    )
    assert openai._served_model_family(args) == "qwen3_8"
    assert isinstance(
        openai._make_adaptive_policy(args, max_depth=3), ExpectedValueDepthPolicy
    )
    assert openai._adaptive_config(args, max_depth=3)["policy"] == "expected_value"


def test_a_live_setting_change_is_re_resolved(tmp_path):
    from mtplx.server import openai

    args = _served_args(tmp_path, FLASH_NEXT_CONFIG)
    assert openai._engine_depth_policy(args)["demoted"] is True
    args.adaptive_policy = "none"
    assert openai._engine_depth_policy(args)["demoted"] is False
    args.adaptive_policy = "expected_value"
    assert openai._engine_depth_policy(args)["policy"] == "none"


# --- `mtplx start --dry-run --json` agrees with the resolver ------------------


def _flag(command: str, name: str) -> str | None:
    parts = command.split()
    return parts[parts.index(name) + 1] if name in parts else None


@pytest.mark.parametrize("client", ["opencode", "hermes"])
@pytest.mark.parametrize("seat_gib", ll.SEATS_GIB)
def test_start_dry_run_matches_the_resolver_on_every_seat(
    client, seat_gib, monkeypatch, tmp_path, capsys
):
    from mtplx.commands import public

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MTPLX_OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.setenv("MTPLX_OPENCODE_DESKTOP_SETTINGS_STORE", str(tmp_path / "d.dat"))
    monkeypatch.setattr(
        public, "_detect_total_ram_bytes_for_opencode_defaults",
        lambda: seat_gib * ll.GIB,
    )
    code = main(
        ["start", client, "--dry-run", "--json", "--model", "models/example", "--yes"]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    command = payload[client]["server_command"]
    lane = ll.comparable_lane(
        ll.resolve_launch_lane(
            client=client, family="unknown", memory_bytes=seat_gib * ll.GIB,
            entry_point="start",
        )
    )
    assert _flag(command, "--prefill-chunk-tokens") is None
    assert _flag(command, "--ssd-session-cache") == lane["ssd_session_cache"]
    assert _flag(command, "--ssd-session-cache-max-size") == lane["ssd_session_cache_max_size"]
    assert _flag(command, "--ssd-session-cache-min-prefix-tokens") == str(
        lane["ssd_session_cache_min_prefix_tokens"]
    )
    # serial + latency are the base defaults, so the writer may omit them.
    assert _flag(command, "--scheduler-mode") in (None, lane["scheduler_mode"])
    assert _flag(command, "--batching-preset") in (None, lane["batching_preset"])
    assert "--decode-batch-max" not in command and "--batch-wait-ms" not in command
    if client == "hermes":
        assert _flag(command, "--adaptive-ev-draft-cost-s") in (None, "0.002")
        assert "0.0048" not in command and "0.006" not in command


def test_start_pi_namespace_gets_the_pi_lane(monkeypatch):
    from mtplx.commands import public

    args = argparse.Namespace(_cli_flags=set())
    public._apply_pi_lane_defaults(args)
    assert args.scheduler_mode == "serial" and args.batching_preset == "latency"
    assert args.adaptive_policy == "expected_value"
    assert args.ssd_session_cache_max_size == "auto"
    assert args.prefill_chunk_tokens is None
    assert not hasattr(args, "temperature")  # Pi's sampler helpers own the sampler
    # A flag the user typed always wins.
    typed = argparse.Namespace(_cli_flags={"scheduler-mode"}, scheduler_mode="ar_batch")
    public._apply_pi_lane_defaults(typed)
    assert typed.scheduler_mode == "ar_batch"


def test_a_client_table_sampler_never_overrides_a_family_owned_sampler():
    from mtplx.commands import public

    args = argparse.Namespace(_cli_flags=set())
    public._apply_hermes_latency_defaults(args)
    assert (args.temperature, args.top_p, args.top_k) == (0.6, 1.0, 20)
    assert args._client_table_sampler_flags == {"temperature", "top_p", "top_k"}
    assert ll.family_owned_sampler_overrides("qwen4_exp")
    assert ll.family_owned_sampler_overrides("qwen3_8")
    assert not ll.family_owned_sampler_overrides("qwen3_6")


# --- the fixture the Swift test reads -----------------------------------------


def test_the_swift_fixture_is_current():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "write_launch_lane_matrix", ROOT / "scripts" / "write_launch_lane_matrix.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert FIXTURE.read_text(encoding="utf-8") == module.render(), (
        "stale fixture: run `python scripts/write_launch_lane_matrix.py`"
    )


def test_the_fixture_carries_every_client_family_switch_and_seat():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["seats_gib"] == [16, 24, 36, 48, 64, 96, 128]
    assert set(fixture["agent_env_by_seat_gib"]) == {"16", "24", "36", "48", "64", "96", "128"}
    assert fixture["families_with_one_compiled_depth"] == ["qwen4_exp"]
    cells = {(r["family"], r["client"], r["adaptive_depth_switch"]) for r in fixture["rows"]}
    assert len(cells) == len(FAMILIES) * len(ll.CLIENTS) * 3
    for row in fixture["rows"]:
        if row["family"] == "qwen4_exp":
            assert row["effective_adaptive_policy"] == "none"
            assert row["effective_depth"] == 3
