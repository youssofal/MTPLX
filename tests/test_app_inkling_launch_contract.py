"""The app's Inkling 2 Mini launches land on the lane `mtplx start` and bare `mtplx serve` land on.

21 November: the app's OpenCode and Hermes buttons passed the Qwen coding lane to Inkling
(`--tool-prompt-mode hybrid`, `--chat-template-profile local_qwen36`, sampler 0.6 / 0.95 / 20),
and `mtplx serve` refused to start it, because Inkling's descriptor requires its native tool calls
and the tokenizer's own chat template. InklingLaunchContractTests (Swift) writes the app's real
launches, one row per pack and launch target, to tests/fixtures/app_inkling_launches.json. Here
each row goes through the CLI's own parser and the family defaults `mtplx serve` applies, and must
land where bare `mtplx serve` and `mtplx start <client>` land for the same pack: the same sampler,
tool format, chat template and reasoning, with no refusal.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtplx.cli import build_parser
from mtplx.commands import public
from mtplx.launch_lane import agent_env_block

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "app_inkling_launches.json"
ROWS = json.loads(FIXTURE.read_text(encoding="utf-8"))
PACKS = (
    "inkling2-mini-bare-speed",
    "inkling2-mini-optimized-speed",
    "inkling2-mini-optimized-quality",
    "inkling2-mini-bare-speed-fp16",
    "inkling2-mini-optimized-speed-fp16",
    "inkling2-mini-optimized-quality-fp16",
)
# Swift LaunchTarget raw values -> the CLI's client names.
APP_TARGETS = {
    "chat": "chat",
    "openWebUI": "openwebui",
    "pi": "pi",
    "openCode": "opencode",
    "hermes": "hermes",
    "claudeCode": "claude-code",
    "codex": "codex",
    "deepseekHarness": "deepseek-harness",
    "benchmark": "benchmark",
    "other": "other",
}
# What the model family owns: every surface must resolve these to the same values.
FAMILY_LANE = (
    "temperature",
    "top_p",
    "top_k",
    "tool_prompt_mode",
    "chat_template_profile",
    "chat_template_path",
    "reasoning_parser",
    "reasoning_effort",
)
INKLING_SAMPLER = {"temperature": 1.0, "top_p": 1.0, "top_k": 0}


def _inspection(pack: str) -> dict:
    # The backend answer `mtplx inspect` gives for every Inkling pack.
    return {
        "model_dir": f"models/{pack}",
        "recommended_backend": "inkling2_loop",
        "mtp_arch": "inkling2-loop",
    }


def _serve_args(pack: str, flags: list[str]):
    return build_parser().parse_args(["serve", "--model", f"models/{pack}", *flags])


def _family_lane(args, pack: str) -> dict:
    public._apply_backend_serve_defaults(args, _inspection(pack))
    return {key: getattr(args, key, None) for key in FAMILY_LANE}


def _start_lane(client: str, pack: str) -> dict:
    """`mtplx start <client>`: its client table, then the same family defaults."""

    args = _serve_args(pack, [])
    if client in public.AGENT_CLIENT_LANE_DEFAULTS:
        public._apply_client_lane_table(args, public.AGENT_CLIENT_LANE_DEFAULTS[client])
    elif client == "pi":
        public._apply_pi_lane_defaults(args)
    elif client == "opencode":
        public._apply_client_lane_table(args, public.OPENCODE_FAIR_BATCHING_DEFAULTS)
    elif client == "hermes":
        public._apply_client_lane_table(args, public.HERMES_LATENCY_DEFAULTS)
    return _family_lane(args, pack)


def test_the_fixture_covers_every_pack_and_launch_target():
    covered = {(row["pack"], row["target"]) for row in ROWS}
    assert covered == {(pack, target) for pack in PACKS for target in APP_TARGETS}
    assert len(ROWS) == len(covered)


@pytest.mark.parametrize("row", ROWS, ids=lambda row: f"{row['pack']}-{row['target']}")
def test_every_app_launch_lands_on_the_lane_bare_serve_lands_on(row):
    app = _family_lane(_serve_args(row["pack"], row["flags"]), row["pack"])
    bare = _family_lane(_serve_args(row["pack"], []), row["pack"])
    assert app == bare
    assert {key: app[key] for key in INKLING_SAMPLER} == INKLING_SAMPLER
    assert app["tool_prompt_mode"] == "native"
    assert app["chat_template_profile"] == "tokenizer"


@pytest.mark.parametrize("target", sorted(set(APP_TARGETS) - {"chat", "openWebUI", "benchmark", "other"}))
@pytest.mark.parametrize("pack", PACKS)
def test_each_agent_button_lands_where_mtplx_start_lands(pack, target):
    row = next(r for r in ROWS if r["pack"] == pack and r["target"] == target)
    app = _family_lane(_serve_args(pack, row["flags"]), pack)
    assert app == _start_lane(APP_TARGETS[target], pack)


@pytest.mark.parametrize("target", ["pi", "openCode", "hermes", "claudeCode", "codex", "deepseekHarness"])
def test_agent_buttons_export_the_start_environment(target):
    """The same coding-agent block `mtplx start` exports (the bank size follows RAM)."""

    row = next(r for r in ROWS if r["pack"] == PACKS[0] and r["target"] == target)
    expected = agent_env_block(APP_TARGETS[target], 128 * 1024**3)
    shared = {key: row["environment"].get(key) for key in expected}
    assert shared == expected


def test_the_qwen_coding_preset_is_still_refused():
    """What the OpenCode and Hermes buttons sent before the fix: the refusal this guards."""

    qwen = ["--tool-prompt-mode", "hybrid", "--chat-template-profile", "local_qwen36"]
    with pytest.raises(ValueError, match="requires --tool-prompt-mode native"):
        _family_lane(_serve_args(PACKS[0], qwen), PACKS[0])
    with pytest.raises(ValueError, match="tokenizer chat template"):
        _family_lane(_serve_args(PACKS[3], qwen[2:]), PACKS[3])


def test_inkling_owns_its_sampler_on_every_entry_point():
    """A client table's sampler is a default, never a pin, on Inkling (as on Qwen 3.8).

    `mtplx start hermes` writes top_p 1.0 and `mtplx start opencode` 0.6 / 0.95 / 20; they reached
    Inkling's own sampler only because those values match the parser's unset markers. A daemon
    started without flags (`python -m mtplx.server.openai --backend-id inkling2_loop`) served the
    generic 0.6 / 0.95 / 20.
    """

    from mtplx.backends.descriptors import INKLING_DESCRIPTOR
    from mtplx.launch_lane import family_owned_sampler_overrides
    from mtplx.server import openai as srv

    assert family_owned_sampler_overrides("inkling2")
    assert srv._model_declared_sampler_defaults(
        "models/MTPLX-Inkling-2-Mini-Bare-Speed", INKLING_DESCRIPTOR
    ) == INKLING_SAMPLER
    args = _serve_args(PACKS[0], [])
    public._apply_client_lane_table(args, {"temperature": 0.5, "top_p": 0.9, "top_k": 40})
    assert {key: _family_lane(args, PACKS[0])[key] for key in INKLING_SAMPLER} == INKLING_SAMPLER
    typed = _serve_args(PACKS[0], ["--temperature", "0.5"])
    public._apply_client_lane_table(typed, {"temperature": 0.6})
    assert _family_lane(typed, PACKS[0])["temperature"] == 0.5  # a flag the user typed wins
