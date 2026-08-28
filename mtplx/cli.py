"""MTPLX command line interface."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from .constants import DEFAULT_RUNTIME_MODEL_DIR
from .fan_mode import FAN_MODE_CHOICES
from .mtp_batch_numerics import MTP_BATCH_NUMERICS_CHOICES
from .profiles import (
    DEFAULT_HF_MODEL_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_PROFILE_NAME,
    DEFAULT_PUBLIC_MODEL_ID,
    PROFILE_CHOICES,
    get_profile,
    list_profiles,
    resolve_profile_name,
)
from .reasoning_effort import REASONING_EFFORT_CHOICES
from .runtime_options import canonicalize_flag_tokens, normalize_paged_kv_quantization
from .version import DISPLAY_VERSION, __version__

# Help/usage advertises only the canonical profiles; the parser itself
# resolves through resolve_profile_name so historical aliases that
# shipped app configs still carry ("auto", "sustained-max") keep working.
_PROFILE_METAVAR = "{" + ",".join(PROFILE_CHOICES) + "}"


def _profile_arg(value: str) -> str:
    try:
        return resolve_profile_name(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


DEFAULT_TRUTH_MODES = (
    "ar",
    "mtp1_batched",
    "mtp1_graphbank",
    "d2_batched",
    "d2_graphbank_capture_commit",
    "d2_graphbank_capture_commit_linear_gdn",
    "d2_graphbank_capture_commit_linear_gdn_committed",
    "d2_correction_cache_d2only",
    "d2_c3_blend015",
    "d3_c3_blend015",
)
DEFAULT_C3_CORRECTOR = Path(
    "outputs/correctors/logit-corrector-20260428-012607-c3-logit-r16.npz"
)


VERIFY_CORE_CHOICES = [
    "stock",
    "linear-gdn",
    "linear-gdn-len5",
    "linear-gdn-from-conv",
    "linear-gdn-from-conv-len5",
    "linear-gdn-from-conv-stream",
    "linear-gdn-from-conv-stream-len5",
    "linear-gdn-from-conv-stream-skip0",
    "linear-gdn-from-conv-stream-skip0-len5",
    "linear-gdn-from-conv-tape",
    "linear-gdn-from-conv-tape-len5",
    "linear-gdn-from-conv-inline-g",
    "linear-gdn-from-conv-inline-g-len5",
    "linear-gdn-final",
]

NATIVE_MTP_60_MODEL = DEFAULT_MODEL_ID


PUBLIC_COMMANDS = (
    (
        "start",
        "Interactive setup → chat (model · mode · web/CLI/Pi/OpenCode/Swival/Hermes/Dashboard)",
    ),
    ("tune", "Find the fastest AR/MTP draft depth for this Mac (AR, D1-D8)"),
    ("help", "Detailed help; `help commands` / `help flags` / `help <name>`"),
    ("setup", "Prepare config and the model cache"),
    ("quickstart", "Run the local OpenAI/Anthropic server"),
    ("connect", "Copy settings for Open WebUI, Claude Code, OpenCode, or Swival"),
    ("ask", "Ask the verified local model once"),
    ("status", "Check install, model, and integration health"),
    ("stop", "Stop the MTPLX daemon answering on a port"),
    ("settings", "Get or set live daemon settings"),
    ("inspect", "Check whether a model is MTPLX-compatible"),
    ("trace", "Diagnose coding sessions: timelines, TPS curves, autopsies, live status"),
    ("forge", "Forge, verify, brand, discover, and publish MTP models"),
    ("hardware", "Inspect Apple Silicon / MLX acceleration eligibility"),
    ("models", "List models in the local MTPLX cache"),
)

ADVANCED_COMMANDS = {
    "Benchmark and QA": (
        ("bench *", "Nightly gates, no-fan runs, envelope compare"),
        ("qa *", "Exactness and distribution gates"),
        (
            "profile *",
            "Compile audit; dispatch/thermal/eval-attribution need the research workspace",
        ),
    ),
    "Support": (
        ("doctor --deep", "Deep install and integration checks"),
        ("debug bundle", "Redacted support bundle"),
        ("metrics watch", "Live server metrics view"),
    ),
    "Models": (
        ("pull", "Download a model into the cache"),
        ("forge *", "Forge and publish MTPLX-branded MTP artifacts"),
        ("models", "List local cached models"),
        ("model architectures", "Architecture support matrix"),
        ("model publish-check", "HF staging readiness"),
    ),
    "Kernel Lab": (
        ("debug hotpath", "Next verify-cycle boundary map"),
        ("runtime-smoke", "Load/inject/generate smoke"),
        ("verify-profile", "Target verify section timings"),
        ("mtp-depth-sweep", "Native-MTP depth sweep"),
    ),
}


def _color_enabled() -> bool:
    return (
        sys.stdout.isatty()
        and os.environ.get("NO_COLOR") is None
        and os.environ.get("MTPLX_NO_COLOR") not in {"1", "true", "TRUE", "yes"}
    )


def _paint(text: str, code: str) -> str:
    if not _color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _heading(text: str) -> str:
    return _paint(text, "1;36")


def _command(text: str) -> str:
    return _paint(text, "1;33")


def _muted(text: str) -> str:
    return _paint(text, "2")


def _command_cell(text: str, width: int) -> str:
    return _command(text) + " " * max(1, width - len(text))


def _ascii_banner() -> str:
    """Inline copy of the MTPLX ASCII banner.

    Duplicated here (rather than importing ``mtplx.ui.banner``) so the
    top-level help survives even when ``rich`` and the rest of the runtime
    stack are not installed yet.
    """

    rows = [
        "███╗   ███╗ ████████╗ ██████╗  ██╗      ██╗  ██╗",
        "████╗ ████║ ╚══██╔══╝ ██╔══██╗ ██║      ╚██╗██╔╝",
        "██╔████╔██║    ██║    ██████╔╝ ██║       ╚███╔╝ ",
        "██║╚██╔╝██║    ██║    ██╔═══╝  ██║       ██╔██╗ ",
        "██║ ╚═╝ ██║    ██║    ██║      ███████╗ ██╔╝ ██╗",
        "╚═╝     ╚═╝    ╚═╝    ╚═╝      ╚══════╝ ╚═╝  ╚═╝",
    ]
    return "\n".join("  " + _paint(line, "1;36") for line in rows)


def _shell_banner_already_shown() -> bool:
    value = os.environ.get("MTPLX_SHELL_BANNER_SHOWN") or os.environ.get(
        "MTPLX_NO_BANNER"
    )
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _help_banner_prefix() -> str:
    if _shell_banner_already_shown():
        return ""
    return f"{_ascii_banner()}\n\n"


def _format_public_help() -> str:
    command_lines = "\n".join(
        f"  {_command_cell(name, 12)} {summary}" for name, summary in PUBLIC_COMMANDS
    )
    version_line = _muted(
        f"v{DISPLAY_VERSION}  ·  Native MTP speculative decoding on Apple Silicon"
    )
    footer = _muted(
        "more: `mtplx help <command>` · `mtplx help advanced` · `mtplx --help` · `mtplx --version`"
    )
    return f"""{_help_banner_prefix()}  {version_line}

{_heading("Commands")}
{command_lines}

{_heading("Examples")}
  mtplx start                       Interactive setup, then chat
  mtplx start --fresh               Re-run the onboarding (new model/mode/surface)
  mtplx start --max --port 8000       Sustained Max browser chat with fan boost
  mtplx start pi --port 8000           Configure Pi, then start the local server
  mtplx start opencode --port 18083    Configure OpenCode Desktop for MTPLX-owned generation
  mtplx start swival --port 18084      Print Swival generic-provider command
  mtplx start hermes --port 18085      Launch Hermes Agent against MTPLX
  mtplx quickstart --port 8000         API server only, no chat

  {footer}
"""


def _format_advanced_help() -> str:
    sections: list[str] = []
    for title, commands in ADVANCED_COMMANDS.items():
        sections.append(f"{_heading(title)}:")
        sections.extend(
            f"  {_command_cell(command, 28)} {summary}" for command, summary in commands
        )
    return (
        f"""{_heading("MTPLX advanced tools")}

Usage: mtplx <command> [options]

Commands suffixed with * have subcommands. Run `mtplx help <command>` for details.
The everyday path is start first. Servers, integrations, QA, and kernels live here when needed.

"""
        + "\n".join(sections)
        + """

Examples:
  mtplx bench nightly --json --dry-run
  mtplx doctor --deep
  mtplx model architectures
  mtplx debug hotpath

Docs: README.md
"""
    )


def _format_start_help() -> str:
    return f"""{_heading("MTPLX start")}

Interactive end-to-end setup. On first run MTPLX walks you through three
quick choices: model, runtime mode, and where to chat (browser, terminal, Pi, OpenCode, Swival, or Hermes).
On later runs it offers "same as last time?" so the chat is one keypress away.

What gets asked:
  1. Model — your configured model, the verified default, custom HF, or local
  2. Mode  — Auto (recommended; Turbo auto-selects for the quantized flagships), Sustained, Sustained Max, or Burst (Stable remains available via --profile safe)
  3. Where — Web UI (default), terminal CLI, Pi, OpenCode Desktop, Swival, or Hermes

Power-user shortcuts (any of these skip the onboarding wizard):
  mtplx start --fresh                 Walk the onboarding again from scratch
  mtplx start cli                     Skip onboarding; terminal chat directly
  mtplx start pi                      Configure Pi, then serve MTPLX for Pi
  mtplx start opencode --port 18083   Configure OpenCode Desktop for MTPLX-owned generation
  mtplx start swival --port 18084     Serve MTPLX and print the Swival command
  mtplx start hermes --port 18085     Serve MTPLX and open Hermes Agent in Terminal
  mtplx start --max                   Sustained Max: long-context mode with ThermalForge fan boost
  mtplx start --profile performance-cold --max
                                      Burst: old max-fan lane, max 8K context
  mtplx start --download              Pull the verified model from HF first
  mtplx start --model /path/...       Use a specific local or HF model
  mtplx start --prompt "hi"           One-shot ask and exit (non-interactive)
  mtplx start cli --no-mtp            Use target-only AR generation

Useful controls:
  --download       Download the selected/default model if missing
  --model PATH     Use a local model folder or HF repo id
  --profile sustained
                  Use the explicit long-context memory-safe native-MTP profile
  --profile safe   Use the compatibility long-response profile
  --mtp            Use native-MTP speculative generation (default)
  --no-mtp         Use target-only AR generation; MTP can be turned back on
  --prompt TEXT    (cli) Ask once and exit instead of opening chat
  --max-tokens N   (cli) Optional response cap; default uses remaining context
  --no-stats       Hide the TPS footer
  --dry-run        Preview without loading MLX

Inside terminal chat:
  /mtp status      Show whether the next turn uses MTP or AR
  /mtp off         Switch the next turn to target-only AR generation
  /mtp on          Switch the next turn back to MTP without reloading
  /stats           Print the last response stats again
  /speed           Run a 192-token speed sample
  /reasoning on|off|auto
                   Control reasoning for the next turns
  /exit            Quit

Aliases:
  `web` and `openwebui` -> browser chat (same as default)
  `terminal`            -> terminal chat (same as `cli`)
  `pi`                  -> Pi coding-agent connection
  `opencode`, `oc`      -> OpenCode Desktop coding-agent connection
  `swival`, `sv`        -> Swival generic-provider connection
  `hermes`              -> Hermes Agent with terminal/file/web/browser/messaging tools
  `dashboard`, `live`   -> live engine dashboard
  (hyphenated forms like `open-webui`, `open-code`, `hermes-agent` also work)
"""


def _format_verbose_help() -> str:
    """Verbose help printed by `mtplx help` (no topic).

    The bare ``mtplx`` invocation prints the compact help; ``mtplx help``
    prints this fuller version with options, more examples, and pointers to
    every help subtopic (``commands``, ``flags``, ``advanced``, ``<command>``).
    """

    public_lines = "\n".join(
        f"  {_command_cell(name, 12)} {summary}" for name, summary in PUBLIC_COMMANDS
    )
    return f"""{_help_banner_prefix()}  {_muted(f"v{DISPLAY_VERSION}  ·  Native MTP speculative decoding on Apple Silicon")}

{_heading("Overview")}

  Open the local chat in your browser, or chat in this terminal. Inference
  parameters (temperature, top-p, top-k, draft depth, max tokens) live in the
  browser sidebar and persist across sessions. The OpenAI/Anthropic-compatible
  server comes up the same way.

{_heading("Usage")}

  mtplx [options] [command] [command-options]

{_heading("Options")}

  --version        Show the installed version
  --no-color       Disable terminal colors
  -h, --help       Compact help (this view is the verbose one)

{_heading("Commands (consumer surface)")}
{public_lines}

{_heading("Examples")}

  mtplx start                       Open the local chat in your browser
  mtplx start cli                   Chat in this terminal instead
  mtplx start --download            Pull the verified model from Hugging Face
  mtplx quickstart --port 8000      Run the API server only
  mtplx connect openwebui           Print Open WebUI integration settings
  mtplx ask "Write a tiny FastAPI app"
  mtplx inspect Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed

{_heading("Help subtopics")}

  mtplx help commands         The consumer + advanced command reference (`help flags` lists every flag)
  mtplx help flags            Every flag, grouped by command
  mtplx help advanced         Benchmarks, QA, publishing, and kernel tools
  mtplx help <command>        Detailed flags for one command (argparse view)
  mtplx help start            The start user journey

  Docs: README.md
"""


def _format_commands_help() -> str:
    """Print the full list of commands across public and advanced surfaces."""

    public_lines = "\n".join(
        f"  {_command_cell(name, 16)} {summary}" for name, summary in PUBLIC_COMMANDS
    )
    advanced_sections: list[str] = []
    for title, commands in ADVANCED_COMMANDS.items():
        advanced_sections.append(_heading(title))
        advanced_sections.extend(
            f"  {_command_cell(command, 28)} {summary}" for command, summary in commands
        )
        advanced_sections.append("")
    return (
        f"""{_heading("MTPLX commands")}

{_heading("Consumer commands")}
{public_lines}

"""
        + "\n".join(advanced_sections)
        + f"""
  {_muted("Run `mtplx <command> --help` for flags on any command above (works for multi-word commands too).")}
"""
    )


def _format_flags_help() -> str:
    """Walk argparse subparsers and print every flag under every command."""

    parser = build_parser()
    sections: list[str] = []
    sections.append(_heading("MTPLX flags"))
    sections.append("")
    sections.append("  Top-level options:")
    for action in parser._actions:
        for entry in _flag_entries_for_action(action):
            sections.append(f"    {entry}")
    sections.append("")

    for sub in parser._actions:
        if not isinstance(sub, argparse._SubParsersAction):
            continue
        for command_name, sub_parser in sorted(
            sub.choices.items(), key=lambda item: item[0]
        ):
            command_section = _flag_section_for_subparser(
                command_name, sub_parser, depth=0
            )
            if command_section:
                sections.extend(command_section)
                sections.append("")

    sections.append(
        _muted("  Run `mtplx help <command>` for the argparse view of one command.")
    )
    return "\n".join(sections) + "\n"


def _flag_entries_for_action(action: argparse.Action) -> list[str]:
    if isinstance(action, (argparse._SubParsersAction, argparse._HelpAction)):
        return []
    if not action.option_strings:
        return []
    flags = ", ".join(action.option_strings)
    metavar = ""
    if action.nargs not in (0, None) or isinstance(
        action,
        (argparse._StoreAction, argparse._AppendAction),
    ):
        if not isinstance(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
                argparse._CountAction,
            ),
        ):
            metavar = " " + (action.metavar or action.dest.upper())
    summary = (action.help or "").replace("\n", " ").strip()
    line = f"{flags}{metavar}"
    if summary:
        line = f"{line:<28}  {summary}"
    return [line]


def _flag_section_for_subparser(
    command_name: str,
    sub_parser: argparse.ArgumentParser,
    *,
    depth: int,
) -> list[str]:
    indent = "  " * (depth + 1)
    lines: list[str] = []
    flag_lines: list[str] = []
    nested_sections: list[list[str]] = []
    for action in sub_parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for nested_name, nested_parser in sorted(
                action.choices.items(), key=lambda item: item[0]
            ):
                nested = _flag_section_for_subparser(
                    f"{command_name} {nested_name}",
                    nested_parser,
                    depth=depth + 1,
                )
                if nested:
                    nested_sections.append(nested)
            continue
        for entry in _flag_entries_for_action(action):
            flag_lines.append(f"{indent}    {entry}")
    if not flag_lines and not nested_sections:
        return []
    lines.append(f"{indent}{_command(command_name)}")
    lines.extend(flag_lines)
    for section in nested_sections:
        lines.extend(section)
    return lines


def _print_help_topic(topic: str | None, parser: argparse.ArgumentParser) -> int:
    if topic in (None, ""):
        print(_format_verbose_help())
        return 0
    if topic in ("commands", "all-commands"):
        print(_format_commands_help())
        return 0
    if topic in ("flags", "options", "all-flags"):
        print(_format_flags_help())
        return 0
    if topic == "start":
        print(_format_start_help())
        return 0
    if topic in ("advanced", "expert", "lab"):
        print(_format_advanced_help())
        return 0
    command_names = _parser_command_names(parser)
    if topic in command_names:
        try:
            parser.parse_args([topic, "--help"])
        except SystemExit as exc:
            return int(exc.code or 0)
        return 0
    print(f"Unknown help topic: {topic}\n")
    print(_format_verbose_help())
    return 2


def _parser_command_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def _print_unknown_command(command: str) -> int:
    print(f"Unknown command: {_command(command)}\n")
    print("Try:")
    for name, summary in PUBLIC_COMMANDS:
        print(f"  mtplx {_command_cell(name, 10)} {summary}")
    print("\nFor the full lab surface: mtplx help advanced")
    return 2


def _comma_floats(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected comma-separated floats")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def _kv_quant_arg(value: str) -> str:
    try:
        normalized = normalize_paged_kv_quantization(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    assert normalized is not None
    return normalized


def _add_reasoning_arg(
    parser: argparse.ArgumentParser, *, default: str | None = None
) -> None:
    parser.add_argument(
        "--reasoning",
        choices=["auto", "on", "off"],
        default=default,
        help=(
            "Reasoning mode. Defaults to auto; use --reasoning off for terse/non-reasoning runs."
        ),
    )


def _add_reasoning_effort_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reasoning-effort",
        choices=list(REASONING_EFFORT_CHOICES),
        default="auto",
        help=(
            "Reasoning effort for models that expose levels, such as Qwen 3.8 "
            "(xhigh/medium/low, MTPLX coding default medium) or Step-3.7 Flash."
        ),
    )


def _add_preserve_thinking_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preserve-thinking",
        choices=["auto", "on", "off", "scoped"],
        default="auto",
        help=(
            "Reasoning-history policy for Qwen chat-template history. scoped keeps "
            "reasoning only inside the active agent round (Qwen 3.6's trained "
            "contract); on preserves all; off strips all. Default auto resolves to "
            "scoped for checkpoint-capable templates, except Qwen 3.8, whose "
            "trained contract preserves thinking by default."
        ),
    )
    parser.add_argument(
        "--strip-assistant-reasoning-history",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def _add_fan_mode_args(parser: argparse.ArgumentParser, *, max_help: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--fan-mode",
        choices=FAN_MODE_CHOICES,
        default="default",
        help=(
            "Fan policy: default lets Apple manage fans, smart boosts only while "
            "requests generate, max preserves sustained verified max."
        ),
    )
    group.add_argument("--max", action="store_true", help=max_help)


def _add_bridge_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tool-prompt-mode",
        choices=["hybrid", "native"],
        default="hybrid",
        help=(
            "OpenAI bridge tool-prompt mode. native uses the model chat "
            "template's tool rendering only; hybrid preserves the legacy "
            "MTPLX contract for rollback/diagnostics."
        ),
    )
    parser.add_argument(
        "--chat-template-profile",
        choices=["local_qwen36", "froggeric_v19", "froggeric_v21_3", "tokenizer"],
        default="local_qwen36",
        help="Chat template profile for server/OpenCode paths.",
    )
    parser.add_argument(
        "--chat-template-path",
        help="Explicit chat_template.jinja path for diagnostics.",
    )


def _add_mtp_toggle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mtp",
        action="store_false",
        dest="no_mtp",
        default=False,
        help="Use native-MTP speculative generation. This is the default.",
    )
    parser.add_argument(
        "--no-mtp",
        action="store_true",
        dest="no_mtp",
        help=(
            "Use target-only AR generation while keeping the same loaded runtime "
            "where live switching is supported."
        ),
    )


SCHEDULER_MODE_CHOICES = (
    "serial",
    "cooperative",
    "ar_batch",
    "mtp_batch",
    "mtp_cohort_experimental",
)
BATCHING_PRESET_CHOICES = ("solo", "latency", "agent", "throughput")
ADAPTIVE_POLICY_CHOICES = (
    "none",
    "streak",
    "expected_value",
    "cost",
    "position_ema",
)


def _add_batching_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scheduler-mode",
        choices=SCHEDULER_MODE_CHOICES,
        default="serial",
        help=(
            "Server scheduler mode. Default serial keeps every request on "
            "the solo MTP oracle (measured 2026-07-05: serialized MTP beats "
            "the batched-AR lane end to end on prefill-heavy concurrent "
            "loads because MTP decode is ~4x faster per stream); ar_batch "
            "opts concurrent requests into the batched AR decode lane, "
            "which wins on decode-heavy many-client loads."
        ),
    )
    parser.add_argument(
        "--batching-preset",
        choices=BATCHING_PRESET_CHOICES,
        default="latency",
        help="Concurrent batching preset for coding-agent/server UX.",
    )
    parser.add_argument(
        "--mtp-batch-numerics",
        choices=MTP_BATCH_NUMERICS_CHOICES,
        default="throughput",
        help="Qwen MTP route: fast B8, balanced B8, or serial B1-exact.",
    )
    parser.add_argument("--max-active-requests", type=_positive_int)
    parser.add_argument("--decode-batch-max", type=_positive_int)
    parser.add_argument("--batch-wait-ms", type=float)
    parser.add_argument("--prefill-chunk-tokens", type=_positive_int)
    parser.add_argument(
        "--experimental-mtp-cohorts",
        action="store_true",
        help="Expose experimental batched-MTP cohort metadata; disabled by default.",
    )


def _add_ssd_session_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ssd-session-cache",
        choices=["off", "on", "write-only"],
        default="on",
        help=(
            "Persistent SessionBank SSD cold tier (default on; kvcache-v2). "
            "Budgeted by min(configured cap, free_disk/4), disabled below "
            "10 GiB free."
        ),
    )
    parser.add_argument(
        "--ssd-session-cache-dir",
        help="Directory for persistent SessionBank snapshots.",
    )
    parser.add_argument(
        "--ssd-session-cache-max-size",
        default="100GB",
        help="Soft maximum SSD SessionBank cache size.",
    )
    parser.add_argument(
        "--ssd-session-cache-min-prefix-tokens",
        type=_positive_int,
        default=512,
        help="Minimum committed prefix length before writing SSD SessionBank snapshots.",
    )


def _add_paged_kv_quant_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--paged-kv-quantization",
        "--paged-kv-quant",
        "--kv-quant",
        dest="paged_kv_quantization",
        metavar="{off,q8,q4}",
        type=_kv_quant_arg,
        default=None,
        help=(
            "Paged KV cache quantization mode. off is default; q8/q4 opt into "
            "the same runtime switch used by the app when the selected model "
            "supports it. Contract: decode-memory feature routed once per "
            "request from its starting offset. q8 at/past the two-pass "
            "threshold (default 1024 tokens) decodes through the inline-"
            "dequant kernel with no bf16 working copy; below it q8 keeps a "
            "context-sized bf16 working mirror, so its memory win starts at "
            "the threshold. q4 never kernels and keeps no mirror: smallest KV "
            "bytes at every length, decode re-dequantizes per step (slower "
            "long-context decode). Prefill runs unquantized (peak prefill "
            "memory unchanged) and compiled-verify/dense-two-pass fast paths "
            "detach while active."
        ),
    )


def _add_adaptive_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adaptive-policy",
        choices=ADAPTIVE_POLICY_CHOICES,
        default="none",
        help="Experimental per-request native-MTP depth policy.",
    )
    parser.add_argument(
        "--qwen38-q4-mtp-block",
        type=Path,
        default=None,
        help=(
            "Validated r17 Q4 native-MTP block for the measured Qwen3.8 "
            "position_ema low profile."
        ),
    )
    parser.add_argument("--adaptive-min-depth", type=_nonnegative_int, default=1)
    parser.add_argument("--adaptive-start-depth", type=_positive_int, default=1)
    parser.add_argument("--adaptive-increase-after", type=_positive_int, default=4)
    parser.add_argument("--adaptive-decrease-after", type=_positive_int, default=1)
    parser.add_argument("--adaptive-position-depth-cap", type=_positive_int, default=4)
    parser.add_argument("--adaptive-ev-base-depth", type=_positive_int, default=2)
    parser.add_argument("--adaptive-ev-accept-priors", default="0.92,0.64,0.32")
    parser.add_argument("--adaptive-ev-draft-cost-s", type=float, default=0.0048)
    parser.add_argument("--adaptive-ev-extra-verify-cost-s", type=float, default=0.006)
    parser.add_argument("--adaptive-ev-baseline-tok-s", type=float, default=40.0)
    parser.add_argument("--adaptive-ev-safety-margin", type=float, default=0.10)
    parser.add_argument("--adaptive-ev-margin-center", type=float, default=1.0)
    parser.add_argument("--adaptive-ev-margin-scale", type=float, default=2.0)
    parser.add_argument("--adaptive-ev-confidence-weight", type=float, default=0.35)
    parser.add_argument(
        "--adaptive-ev-min-extra-accept-probability",
        type=float,
        default=0.18,
    )
    parser.add_argument(
        "--adaptive-ev-warmup-full-depth-cycles",
        type=int,
        default=4,
    )
    parser.add_argument("--adaptive-ev-exploration-interval", type=int, default=32)


def cmd_bench_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_bench_public as handler

    return handler(args)


def cmd_tune_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_tune_public as handler

    return handler(args)


def cmd_stop_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_stop_public as handler

    return handler(args)


def cmd_settings_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_settings_public as handler

    return handler(args)


def cmd_hardware_public(args: argparse.Namespace) -> int:
    from .hardware import inspect_hardware

    payload = inspect_hardware()
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print("MTPLX hardware inspect")
    print(f"chip: {payload.get('chip') or 'unknown'}")
    print(
        f"Apple Silicon generation: {payload.get('apple_silicon_generation') or 'unknown'}"
    )
    print(f"macOS: {payload.get('macos_version') or 'unknown'}")
    print(f"MLX: {payload.get('mlx_version') or 'not installed'}")
    print(f"Python: {payload.get('python_version')} ({payload.get('machine')})")
    print(f"unified memory: {payload.get('unified_memory_gb') or 'unknown'} GB")
    print(
        "M5 TensorOps eligible: "
        f"{str(bool(payload.get('m5_neural_accelerator_eligible'))).lower()}"
    )
    print("hardware acceleration confirmed: false")
    for warning in payload.get("warnings") or []:
        print(f"warning: {warning}")
    return 0


def cmd_chat_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_chat_public as handler

    return handler(args)


def cmd_quickstart_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_quickstart_public as handler

    return handler(args)


def cmd_doctor(args: argparse.Namespace) -> int:
    from .commands.public import cmd_doctor as handler

    return handler(args)


def cmd_inspect_model_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_inspect_model_public as handler

    return handler(args)


def cmd_profile_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_profile_public as handler

    return handler(args)


def cmd_pull_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_pull_public as handler

    return handler(args)


def cmd_list_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_list_public as handler

    return handler(args)


def cmd_remove_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_remove_public as handler

    return handler(args)


def cmd_run_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_run_public as handler

    return handler(args)


def cmd_qa_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_qa_public as handler

    return handler(args)


def cmd_serve_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_serve_public as handler

    return handler(args)


def cmd_thermal_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_thermal_public as handler

    return handler(args)


def cmd_max_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_max_public as handler

    return handler(args)


def cmd_debug_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_debug_public as handler

    return handler(args)


def cmd_openwebui_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_openwebui_public as handler

    return handler(args)


def cmd_dashboard_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_dashboard_public as handler

    return handler(args)


def cmd_metrics_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_metrics_public as handler

    return handler(args)


def cmd_integrate_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_integrate_public as handler

    return handler(args)


def cmd_model_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_model_public as handler

    return handler(args)


def cmd_config_public(args: argparse.Namespace) -> int:
    from .commands.public import cmd_config_public as handler

    return handler(args)


def cmd_forge_public(args: argparse.Namespace) -> int:
    from .commands.forge import cmd_forge_public as handler

    return handler(args)


def cmd_trace_public(args: argparse.Namespace) -> int:
    from .commands.trace import cmd_trace as handler

    return handler(args)


def _cmd_env(args: argparse.Namespace) -> int:
    from .env import collect_environment

    snapshot = collect_environment(args.project_root)
    print(snapshot.to_json())
    return 0


def _cmd_bench_preflight(args: argparse.Namespace) -> int:
    from .benchmarks.runners.preflight import run_preflight, write_preflight

    result = run_preflight(
        args.project_root,
        top_limit=args.top_limit,
        cpu_threshold=args.cpu_threshold,
        min_free_gib=args.min_free_gib,
    )
    if args.output:
        write_preflight(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["clean"] or not args.strict else 2


def _cmd_inspect_model(args: argparse.Namespace) -> int:
    from .artifacts import inspect_model

    try:
        inspection = inspect_model(args.model)
    except Exception as exc:
        print(
            json.dumps(
                {"error": "inspect failed", "model": args.model, "detail": str(exc)},
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(inspection.to_json())
    compatibility = inspection.compatibility or {}
    if args.require_mtp or getattr(args, "strict_exit_code", True):
        return int(compatibility.get("exit_code", 0))
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    from .hf_loader import model_cache_dir, pull_model
    from .thermal import detect_thermal_control

    config_path = Path(args.config).expanduser()
    model_dir = model_cache_dir(args.model_dir)
    thermal_detection = detect_thermal_control()
    selected_thermal = thermal_detection.get("selected") or {}
    thermal_tool = selected_thermal.get("kind", "none")
    hardware = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "is_macos": platform.system() == "Darwin",
        "is_apple_silicon": platform.system() == "Darwin"
        and platform.machine() == "arm64",
    }
    profile = get_profile(args.profile)
    commands = {
        "doctor": "mtplx doctor --json",
        "pull": f"mtplx pull {args.model}",
        "inspect": f"mtplx inspect {args.model} --json",
        "run": f'mtplx run "hello" --model {args.model}',
        "serve": f"mtplx serve --model {args.model}",
    }
    report = {
        "status": "ready_for_init",
        "config_path": str(config_path),
        "dry_run": bool(args.dry_run),
        "model": args.model,
        "model_dir": str(model_dir),
        "profile": profile.to_dict(),
        "hardware": hardware,
        "thermal_control": {
            "requested": args.thermal_control,
            "detected": thermal_tool,
            "details": thermal_detection,
        },
        "download_requested": bool(args.download),
        "downloaded": False,
        "wrote_config": False,
        "commands": commands,
        "next_steps": list(commands.values()),
    }
    if args.write and not args.dry_run:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            "# MTPLX user configuration\n"
            f"model = {json.dumps(args.model)}\n"
            f"model_dir = {json.dumps(str(model_dir))}\n"
            f"profile = {json.dumps(profile.name)}\n"
            f"thermal_control = {json.dumps(args.thermal_control)}\n",
            encoding="utf-8",
        )
        report["wrote_config"] = True
    if args.download and not args.dry_run:
        try:
            progress_callback = None
            if not args.json:
                from .commands.public import _download_progress_callback

                progress_callback = _download_progress_callback()
            report["download_result"] = pull_model(
                args.model,
                cache_dir=model_dir,
                progress_callback=progress_callback,
            )
        except Exception as exc:
            report["download_error"] = str(exc)
            if args.json:
                print(json.dumps(report, indent=2, sort_keys=True))
            else:
                print("MTPLX init")
                print(f"download failed: {exc}")
            return 1
        report["downloaded"] = True
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("MTPLX init")
        print(f"config: {config_path}")
        print(f"model: {args.model}")
        print(f"profile: {profile.name}")
        print(f"model cache: {model_dir}")
        print(
            "hardware: "
            f"{hardware['system']} {hardware['release']} {hardware['machine']} "
            f"(apple_silicon={str(hardware['is_apple_silicon']).lower()})"
        )
        print(f"thermal control: {thermal_tool}")
        if args.write and not args.dry_run:
            print("wrote config")
        else:
            print("dry run: no files written")
        if args.download and not args.dry_run:
            print("downloaded model")
        print(f"next: {commands['doctor']}")
        print(f"next: {commands['pull']}")
    return 0


def _cmd_profiles(args: argparse.Namespace) -> int:
    payload = {
        "default": DEFAULT_PROFILE_NAME,
        "flagship_default": "turbo",
        "profiles": list_profiles(),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(f"library default: {DEFAULT_PROFILE_NAME}")
    print(
        "start/serve default: resolves per model — turbo for the quantized "
        "27B/9B flagships, sustained otherwise"
    )
    for profile in payload["profiles"]:
        print(f"{profile['name']}: {profile['summary']}")
    return 0


def _cmd_setup(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser()
    if config_path.exists() and not args.force and not args.dry_run:
        payload = {
            "action": "setup",
            "status": "already_configured",
            "config_path": str(config_path),
            "next_steps": [
                "mtplx status",
                "mtplx quickstart --port 8000",
                "mtplx connect openwebui",
            ],
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("MTPLX setup")
            print(f"config already exists: {config_path}")
            print("next: mtplx status")
            print("next: mtplx quickstart --port 8000")
            print("Use --force to rewrite the config.")
        return 0
    args.write = True
    return _cmd_init(args)


def _cmd_connect(args: argparse.Namespace) -> int:
    if not args.integration:
        server_command = f"mtplx quickstart --host {args.host} --port {args.port}"
        payload = {
            "action": "connect",
            "integrations": [
                {
                    "name": "openwebui",
                    "command": "mtplx connect openwebui",
                    "purpose": "Use MTPLX as an OpenAI-compatible local model in Open WebUI.",
                },
                {
                    "name": "claude-code",
                    "command": "mtplx connect claude-code",
                    "purpose": "Use MTPLX through the Anthropic-compatible Claude Code path.",
                },
                {
                    "name": "opencode",
                    "command": "mtplx connect opencode",
                    "purpose": "Use MTPLX in OpenCode with MTPLX-owned reasoning and sampler policy.",
                },
                {
                    "name": "swival",
                    "command": "mtplx connect swival",
                    "purpose": "Use MTPLX through Swival's generic OpenAI-compatible provider.",
                },
            ],
            "server": server_command,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("Connect MTPLX")
            print(f"1. Start the server: {server_command}")
            print("2. Pick a client:")
            print("   mtplx connect openwebui")
            print("   mtplx connect claude-code")
            print("   mtplx connect opencode")
            print("   mtplx connect swival")
        return 0
    return cmd_integrate_public(args)


def _cmd_bench(args: argparse.Namespace) -> int:
    if getattr(args, "bench_action", None):
        return cmd_bench_public(args)
    if args.profile:
        return _cmd_bench_profile(args)
    from .benchmarks.runners.harness import run_manifest_only
    from .benchmarks.schema import BenchmarkConfig, now_run_id

    out = (
        Path(args.output)
        if args.output
        else Path("outputs") / f"{now_run_id(args.backend)}.jsonl"
    )
    config = BenchmarkConfig(
        backend=args.backend,
        model_path=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        speculative_depth=args.speculative_depth,
        adaptive=args.adaptive,
    )
    if args.backend != "manifest":
        raise SystemExit("Only backend=manifest is implemented in this scaffold gate")
    records = run_manifest_only(args.prompts, config, out)
    print(json.dumps({"records": len(records), "output": str(out)}, indent=2))
    return 0


def _suite_to_prompts(suite: str | None, fallback: str) -> str:
    if suite is None:
        return fallback
    suites = {
        "default": "mtplx/benchmarks/prompts/default.jsonl",
        "long_code": "mtplx/benchmarks/prompts/long_code.jsonl",
        "calibration_coding": "mtplx/benchmarks/prompts/calibration_coding.jsonl",
    }
    if suite not in suites:
        raise SystemExit(f"unknown benchmark suite: {suite}")
    return suites[suite]


def _cmd_bench_profile(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_depth_sweep import (
        run_mtp_depth_sweep,
        write_depth_sweep,
    )
    from .benchmarks.runners.preflight import run_preflight
    from .benchmarks.schema import now_run_id
    from .artifacts import inspect_model
    from .draft_lm_head import draft_lm_head_spec_from_runtime_contract
    from .draft_sampling import draft_sampler_spec_from_runtime_contract
    from .profiles import apply_profile_env, runtime_env_overrides_from_contract

    profile = get_profile(args.profile)
    if profile.name != "performance-cold":
        raise SystemExit(f"unknown benchmark profile: {args.profile}")
    # Every accepted flag is honored or refused loudly — never silently
    # discarded (#285: four configs once produced byte-identical runs).
    if getattr(args, "stock_ar", False):
        raise SystemExit(
            "--stock-ar is not available on the depth-sweep harness (it always "
            "loads the MTP runtime); use --harness direct-http for stock AR, or "
            "--generation-mode ar here for a target-only AR baseline."
        )
    requested_harness = getattr(args, "harness", None)
    if requested_harness not in (None, "", "depth-sweep"):
        raise SystemExit(
            f"--harness {requested_harness!r} is not supported with "
            "--profile performance-cold; the profile runs the depth-sweep "
            "harness"
        )
    ar_baseline = getattr(args, "generation_mode", None) == "ar"
    # requested_depths / requested_seed resolve below via the shared bench
    # helpers so both harness routes agree on defaults (#285).
    model_arg = (
        NATIVE_MTP_60_MODEL
        if args.model == str(DEFAULT_RUNTIME_MODEL_DIR)
        else args.model
    )
    runtime_contract = None
    try:
        compatibility = inspect_model(model_arg).to_dict().get("compatibility") or {}
        runtime_contract = compatibility.get("runtime_contract")
    except Exception:
        runtime_contract = None
    runtime_env_overrides = runtime_env_overrides_from_contract(runtime_contract)
    apply_profile_env(profile.name, runtime_env_overrides=runtime_env_overrides)
    preflight = None
    if args.strict:
        preflight = run_preflight(
            ".",
            cpu_threshold=args.cpu_threshold,
            min_free_gib=args.min_free_gib,
        )
        if not preflight["clean"]:
            print(
                json.dumps(
                    {"profile": profile.name, "preflight": preflight},
                    indent=2,
                    sort_keys=True,
                )
            )
            return 2
    prompts = _suite_to_prompts(args.suite, args.prompts)
    out = (
        Path(args.output)
        if args.output
        else Path("outputs") / f"{now_run_id(profile.name)}.json"
    )
    fallback_draft_lm_head = (
        None
        if profile.draft_lm_head is None
        else {
            "bits": profile.draft_lm_head.bits,
            "group_size": profile.draft_lm_head.group_size,
            "mode": profile.draft_lm_head.mode,
        }
    )
    try:
        if runtime_contract is None:
            compatibility = (
                inspect_model(model_arg).to_dict().get("compatibility") or {}
            )
            runtime_contract = compatibility.get("runtime_contract")
        draft_lm_head = draft_lm_head_spec_from_runtime_contract(
            runtime_contract,
            fallback=fallback_draft_lm_head,
        )
        draft_sampler = draft_sampler_spec_from_runtime_contract(runtime_contract)
    except Exception:
        draft_lm_head = fallback_draft_lm_head
        draft_sampler = None
    # #285: honor the user's sweep knobs. depths/seed/compare-ar were
    # hardcoded ("3"/0/False) while the CLI accepted the flags — reuse the
    # same resolution helpers as `mtplx bench run --harness depth-sweep` so
    # both routes agree on defaults (depths "3", seed 0) when nothing is
    # passed.
    from .commands.public import _benchmark_seed, _depths_for_bench_run

    requested_depths = _depths_for_bench_run(args)
    requested_seed = _benchmark_seed(
        args, runtime_profile="native_mtp_60_cold", harness="depth-sweep"
    )
    result = run_mtp_depth_sweep(
        model_arg,
        prompts,
        depths=requested_depths,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=192 if args.max_tokens == 128 else args.max_tokens,
        seed=requested_seed,
        limit=args.limit,
        enable_thinking=False,
        compare_ar=ar_baseline or bool(getattr(args, "compare_ar", False)),
        ar_only=ar_baseline,
        mtp_hidden_variant="post_norm",
        mtp_cache_policy="persistent",
        mtp_history_policy="committed",
        min_speculative_depth=1,
        verify_strategy="capture_commit",
        verify_core="linear-gdn-from-conv-tape",
        draft_lm_head_bits=(
            None if draft_lm_head is None else int(draft_lm_head["bits"])
        ),
        draft_lm_head_group_size=(
            64 if draft_lm_head is None else int(draft_lm_head["group_size"])
        ),
        draft_lm_head_mode=(
            "affine" if draft_lm_head is None else str(draft_lm_head["mode"])
        ),
        draft_temperature=(
            args.draft_temperature
            if args.draft_temperature is not None
            else None if draft_sampler is None else float(draft_sampler["temperature"])
        ),
        draft_top_p=(
            args.draft_top_p
            if args.draft_top_p is not None
            else None if draft_sampler is None else float(draft_sampler["top_p"])
        ),
        draft_top_k=(
            args.draft_top_k
            if args.draft_top_k is not None
            else None if draft_sampler is None else int(draft_sampler["top_k"])
        ),
    )
    result["profile"] = {
        **profile.to_dict(),
        "fast_path_env": {**profile.env_dict(), **runtime_env_overrides},
        "model": model_arg,
        "model_id": model_arg,
        "depths": requested_depths,
        "seed": requested_seed,
        "ar_baseline": ar_baseline,
        "compare_ar": ar_baseline or bool(getattr(args, "compare_ar", False)),
        "verify_strategy": "capture_commit",
        "verify_core": "linear-gdn-from-conv-tape",
        "draft_lm_head": draft_lm_head,
        "draft_sampler": draft_sampler,
        "enable_thinking": False,
        "strict_preflight": bool(args.strict),
        "preflight": preflight,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    write_depth_sweep(out, result)
    print(
        json.dumps(
            {"profile": profile.name, "output": str(out)}, indent=2, sort_keys=True
        )
    )
    return 0


def _cmd_runtime_smoke(args: argparse.Namespace) -> int:
    from .benchmarks.runners.runtime_smoke import run_runtime_smoke

    result = run_runtime_smoke(args.model, args.prompt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["mtp_enabled"] and result["mtp_valid"] else 2


def _cmd_probe_contract(args: argparse.Namespace) -> int:
    from .benchmarks.runners.contract_probe import run_contract_probe

    result = run_contract_probe(
        args.model,
        args.prompts,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.raw_prompts,
        enable_thinking=False if args.disable_thinking else None,
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify_ratio(args: argparse.Namespace) -> int:
    from .benchmarks.runners.verify_ratio import run_verify_ratio

    result = run_verify_ratio(
        args.model,
        args.prompt,
        max_k=args.max_k,
        repeats=args.repeats,
    )
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify_profile(args: argparse.Namespace) -> int:
    from .benchmarks.runners.verify_profile import (
        run_verify_profile,
        write_verify_profile,
    )

    lengths = [int(x.strip()) for x in args.lengths.split(",") if x.strip()]
    result = run_verify_profile(
        args.model,
        args.prompts,
        lengths=lengths,
        repeats=args.repeats,
        warmup=args.warmup,
        prompt_index=args.prompt_index,
        enable_thinking=False if args.disable_thinking else None,
    )
    if args.output:
        write_verify_profile(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_verify_qmm_probe(args: argparse.Namespace) -> int:
    from .benchmarks.runners.verify_qmm_probe import (
        run_verify_qmm_probe,
        write_verify_qmm_probe,
    )

    result = run_verify_qmm_probe(
        args.model,
        m_values=args.m_values,
        repeats=args.repeats,
        warmup=args.warmup,
        include=args.include,
        dtype=args.dtype,
        mtp=not args.no_mtp,
        max_groups=args.max_groups,
        seed=args.seed,
        dense_mirror=args.dense_mirror,
    )
    if args.output:
        write_verify_qmm_probe(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_multi_qmv_probe(args: argparse.Namespace) -> int:
    from .benchmarks.runners.multi_qmv_probe import (
        run_multi_qmv_probe,
        write_multi_qmv_probe,
    )

    result = run_multi_qmv_probe(
        args.model,
        include=args.include,
        repeats=args.repeats,
        warmup=args.warmup,
        dtype=args.dtype,
        seed=args.seed,
        mtp=not args.no_mtp,
    )
    if args.output:
        write_multi_qmv_probe(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_batch_equivalence(args: argparse.Namespace) -> int:
    from .benchmarks.runners.batch_equivalence import (
        run_batch_equivalence,
        write_batch_equivalence,
    )

    result = run_batch_equivalence(
        args.model,
        args.prompts,
        suffix_len=args.suffix_len,
        limit=args.limit,
        expand_to=args.expand_to,
        enable_thinking=False if args.disable_thinking else None,
        tolerance=args.tolerance,
    )
    if args.output:
        write_batch_equivalence(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def _cmd_capture_commit_equivalence(args: argparse.Namespace) -> int:
    from .benchmarks.runners.capture_commit_equivalence import (
        run_capture_commit_equivalence,
        write_capture_commit_equivalence,
    )

    result = run_capture_commit_equivalence(
        args.model,
        args.prompts,
        suffix_len=args.suffix_len,
        min_keep_tokens=args.min_keep_tokens,
        limit=args.limit,
        expand_to=args.expand_to,
        enable_thinking=False if args.disable_thinking else None,
        tolerance=args.tolerance,
        verify_backend=args.verify_backend,
        verify_core=args.verify_core,
    )
    if args.output:
        write_capture_commit_equivalence(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def _cmd_mtp1_greedy_gate(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp1_gate import run_mtp1_greedy_gate, write_gate_result

    result = run_mtp1_greedy_gate(
        args.model,
        args.prompts,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        expand_to=args.expand_to,
        enable_thinking=False if args.disable_thinking else None,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        draft_margin_threshold=args.draft_margin_threshold,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    if args.output:
        write_gate_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


def _cmd_mtp1_sampler_smoke(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp1_sampler_smoke import (
        run_mtp1_sampler_smoke,
        write_sampler_smoke,
    )

    result = run_mtp1_sampler_smoke(
        args.model,
        args.prompts,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        compare_ar=args.compare_ar,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        draft_margin_threshold=args.draft_margin_threshold,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    if args.output:
        write_sampler_smoke(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        v for row in result["rows"] for v in row["validations"] if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_mtp_depth_sweep(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_depth_sweep import (
        run_mtp_depth_sweep,
        write_depth_sweep,
    )

    result = run_mtp_depth_sweep(
        args.model,
        args.prompts,
        depths=args.depths,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        draft_lm_head_bits=args.draft_lm_head_bits,
        draft_lm_head_group_size=args.draft_lm_head_group_size,
        draft_lm_head_mode=args.draft_lm_head_mode,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        compare_ar=args.compare_ar,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        mtp_history_policy=args.mtp_history_policy,
        draft_margin_threshold=args.draft_margin_threshold,
        min_speculative_depth=args.min_speculative_depth,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        draft_core=args.draft_core,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
        mtp_adapter_path=args.mtp_adapter,
        merge_mtp_adapter=args.merge_mtp_adapter,
        mtp_corrector_path=args.mtp_corrector,
        mtp_corrector_blend=args.mtp_corrector_blend,
        online_hidden_corrector_alpha=args.online_hidden_corrector_alpha,
        online_hidden_corrector_decay=args.online_hidden_corrector_decay,
        online_hidden_corrector_warmup=args.online_hidden_corrector_warmup,
        online_hidden_corrector_max_feed_depth=args.online_hidden_corrector_max_feed_depth,
        online_hidden_corrector_key=args.online_hidden_corrector_key,
        online_correction_cache=args.online_correction_cache,
        online_correction_cache_min_depth=args.online_correction_cache_min_depth,
        online_correction_cache_key=args.online_correction_cache_key,
        prompt_correction_cache=args.prompt_correction_cache,
        prompt_correction_cache_min_depth=args.prompt_correction_cache_min_depth,
        adapter_ensemble_q=args.adapter_ensemble_q,
        adapter_ensemble_epsilon=args.adapter_ensemble_epsilon,
        adapter_ensemble_min_depth=args.adapter_ensemble_min_depth,
        mtp_topk_reranker_calib=args.mtp_topk_reranker_calib,
        mtp_topk_reranker_depths=args.mtp_topk_reranker_depths,
        mtp_topk_reranker_topk=args.mtp_topk_reranker_topk,
        mtp_topk_reranker_q_weight=args.mtp_topk_reranker_q_weight,
        mtp_topk_reranker_token_weight=args.mtp_topk_reranker_token_weight,
        mtp_topk_reranker_rank_weight=args.mtp_topk_reranker_rank_weight,
        mtp_topk_reranker_prefix_active_only=not args.mtp_topk_reranker_all_rows,
    )
    if args.output:
        write_depth_sweep(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        v
        for depth in result["depths"]
        for row in depth["rows"]
        for v in row["validations"]
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_mtp_chain_probe(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_chain_probe import (
        run_mtp_chain_probe,
        write_mtp_chain_probe,
    )

    result = run_mtp_chain_probe(
        args.model,
        args.prompts,
        depth=args.depth,
        limit=args.limit,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.raw_prompts,
        enable_thinking=False if args.disable_thinking else None,
        windows=args.windows,
        stride=args.stride,
        top_ranks=args.top_ranks,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
        base_hidden_variants=args.base_hidden_variants,
        mtp_hidden_variants=args.mtp_hidden_variants,
        cache_policies=args.cache_policies,
        concat_orders=args.concat_orders,
        mtp_position_modes=args.mtp_position_modes,
        history_modes=args.history_modes,
        anchors=args.anchors,
    )
    if args.output:
        write_mtp_chain_probe(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_mtp_tree_probe(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_tree_probe import (
        run_mtp_tree_probe,
        write_mtp_tree_probe,
    )

    result = run_mtp_tree_probe(
        args.model,
        args.prompts,
        depth=args.depth,
        budgets=args.budgets,
        branch_factor=args.branch_factor,
        limit=args.limit,
        windows=args.windows,
        stride=args.stride,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.raw_prompts,
        enable_thinking=False if args.disable_thinking else None,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
        base_hidden_variant=args.base_hidden_variant,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        anchor=args.anchor,
    )
    if args.output:
        write_mtp_tree_probe(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_mtp_depth_grid(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_depth_grid import (
        run_mtp_depth_policy_grid,
        write_depth_grid,
    )

    result = run_mtp_depth_policy_grid(
        args.model,
        args.prompts,
        depth=args.depth,
        thresholds=args.thresholds,
        min_depths=args.min_depths,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        compare_ar=args.compare_ar,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        mtp_history_policy=args.mtp_history_policy,
        verify_strategy=args.verify_strategy,
        mtp_corrector_path=args.mtp_corrector,
        mtp_corrector_blend=args.mtp_corrector_blend,
        store_events=args.store_events,
    )
    if args.output:
        write_depth_grid(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        v
        for cell in result["grid"]
        for row in cell["rows"]
        for v in row["validations"]
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_mtp_adaptive(args: argparse.Namespace) -> int:
    from .benchmarks.runners.mtp_adaptive import run_mtp_adaptive, write_adaptive

    result = run_mtp_adaptive(
        args.model,
        args.prompts,
        max_depth=args.max_depth,
        min_depth=args.min_depth,
        start_depth=args.start_depth,
        increase_after=args.increase_after,
        decrease_after=args.decrease_after,
        policy_kind=args.policy,
        ev_base_depth=args.ev_base_depth,
        ev_accept_priors=args.ev_accept_priors,
        ev_draft_cost_s=args.ev_draft_cost_s,
        ev_extra_verify_cost_s=args.ev_extra_verify_cost_s,
        ev_baseline_tok_s=args.ev_baseline_tok_s,
        ev_safety_margin=args.ev_safety_margin,
        ev_margin_center=args.ev_margin_center,
        ev_margin_scale=args.ev_margin_scale,
        ev_confidence_weight=args.ev_confidence_weight,
        ev_min_extra_accept_probability=args.ev_min_extra_accept_probability,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        compare_ar=args.compare_ar,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        mtp_history_policy=args.mtp_history_policy,
        verify_strategy=args.verify_strategy,
        verify_core=args.verify_core,
        mtp_quant_bits=args.mtp_quant_bits,
        mtp_quant_group_size=args.mtp_quant_group_size,
        mtp_quant_mode=args.mtp_quant_mode,
    )
    if args.output:
        write_adaptive(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    failures = [
        v for row in result["rows"] for v in row["validations"] if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_dflash_mlx_baseline(args: argparse.Namespace) -> int:
    from .benchmarks.runners.competitor_baselines import (
        run_dflash_mlx_baseline,
        write_competitor_result,
    )

    result = run_dflash_mlx_baseline(
        args.model,
        args.draft_model,
        args.prompts,
        dflash_source=args.dflash_source,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        block_size=args.block_size,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        draft_sliding_window_size=args.draft_sliding_window_size,
    )
    if args.output:
        write_competitor_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("error"):
        return 2
    failures = [
        v
        for row in result["rows"]
        for v in row.get("validations", [])
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_ddtree_mlx_baseline(args: argparse.Namespace) -> int:
    from .benchmarks.runners.competitor_baselines import (
        run_ddtree_mlx_baseline,
        write_competitor_result,
    )

    result = run_ddtree_mlx_baseline(
        args.model,
        args.draft_model,
        args.prompts,
        ddtree_source=args.ddtree_source,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        tree_budget=args.tree_budget,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
    )
    if args.output:
        write_competitor_result(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("error"):
        return 2
    failures = [
        v
        for row in result["rows"]
        for v in row.get("validations", [])
        if not v["passed"]
    ]
    return 0 if not failures else 2


def _cmd_truth_report(args: argparse.Namespace) -> int:
    from .benchmarks.runners.truth import run_truth_report, write_truth_report

    result = run_truth_report(
        model_path=args.model,
        prompt_suite=args.prompts,
        modes=args.modes,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        draft_temperature=args.draft_temperature,
        draft_top_p=args.draft_top_p,
        draft_top_k=args.draft_top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
        limit=args.limit,
        enable_thinking=False if args.disable_thinking else None,
        mtp_hidden_variant=args.mtp_hidden_variant,
        mtp_cache_policy=args.mtp_cache_policy,
        mtp_history_policy=args.mtp_history_policy,
        c3_corrector=args.c3_corrector,
        c3_blend=args.c3_blend,
        project_root=args.project_root,
        min_free_gib=args.min_free_gib,
        cpu_threshold=args.cpu_threshold,
        keep_going=not args.fail_fast,
    )
    output_dir = Path(args.output_dir)
    output_json = (
        Path(args.output_json)
        if args.output_json
        else output_dir / f"{result['run_id']}.json"
    )
    output_md = (
        Path(args.output_md)
        if args.output_md
        else output_dir / f"{result['run_id']}.md"
    )
    write_truth_report(output_json, output_md, result)
    print(
        json.dumps(
            {
                "json": str(output_json),
                "markdown": str(output_md),
                "passed": result["passed"],
                "claim_label": result["claim_label"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.strict_preflight and not result["preflight"].get("clean"):
        return 2
    return 0 if result["passed"] else 2


def _cmd_session_bank(args: argparse.Namespace) -> int:
    from .benchmarks.runners.session_bank import (
        run_session_bank_benchmark,
        write_session_bank_report,
    )

    result = run_session_bank_benchmark(
        args.model,
        args.prompts,
        prompt_index=args.prompt_index,
        suffix_text=args.suffix_text,
        max_prompt_tokens=args.max_prompt_tokens,
        chat_template=not args.raw_prompts,
        enable_thinking=False if args.disable_thinking else None,
        max_entries=args.max_entries,
        tolerance=args.tolerance,
        restore_mode=args.restore_mode,
    )
    if args.output:
        write_session_bank_report(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["exact"] else 2


class _FlagRecordingArgumentParser(argparse.ArgumentParser):
    """Root parser that always records which flags were actually typed.

    Measured-default and override helpers key off ``args._cli_flags`` to
    distinguish user intent from values written onto the namespace by
    internal default appliers. Recording the flags inside ``parse_args``
    keeps that signal correct for every caller (cli.main, tests, and
    programmatic invocations), not just the console entry point.
    """

    def parse_args(self, args=None, namespace=None):  # type: ignore[override]
        raw = list(sys.argv[1:]) if args is None else list(args)
        parsed = super().parse_args(raw, namespace)
        if not hasattr(parsed, "_cli_flags"):
            parsed._cli_flags = canonicalize_flag_tokens(
                _explicit_cli_flags(raw), self, parsed
            )
        return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = _FlagRecordingArgumentParser(prog="mtplx")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable terminal colors",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"mtplx {DISPLAY_VERSION} ({__version__})",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    default_model = DEFAULT_HF_MODEL_ID

    help_p = sub.add_parser("help", help=argparse.SUPPRESS)
    help_p.add_argument("topic", nargs="?")
    help_p.set_defaults(func=lambda args: _print_help_topic(args.topic, parser))

    advanced_p = sub.add_parser("advanced", help=argparse.SUPPRESS)
    advanced_p.set_defaults(func=lambda _args: print(_format_advanced_help()) or 0)

    hardware_p = sub.add_parser("hardware", help="Inspect local Apple Silicon hardware")
    hardware_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    hardware_p.set_defaults(func=cmd_hardware_public, hardware_action="inspect")
    hardware_sub = hardware_p.add_subparsers(dest="hardware_action")
    hardware_inspect_p = hardware_sub.add_parser(
        "inspect",
        help="Report hardware and MLX acceleration eligibility",
    )
    hardware_inspect_p.add_argument("--json", action="store_true")
    hardware_inspect_p.set_defaults(func=cmd_hardware_public)

    start_flow_p = sub.add_parser(
        "start",
        help="Interactive setup → chat (model · mode · web/CLI/Pi/OpenCode/Swival/Hermes/Dashboard)",
        usage="mtplx start [cli|web|pi|opencode|swival|hermes|dashboard] [--fresh] [--max] [--profile NAME] [--model PATH_OR_REPO] [--prompt TEXT]",
        description="Walk through model / mode / surface in three quick steps, then chat. Returning users get a 'same as last time?' prompt. Use --fresh to redo the onboarding, or pass any of --model / --profile / --max / cli|web|pi|opencode|swival|hermes|dashboard to skip it entirely.",
    )
    start_flow_p.add_argument(
        "target",
        nargs="?",
        choices=[
            "web",
            "openwebui",
            "open-webui",
            "cli",
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
        ],
        default=None,
        help="Web chat, terminal chat, Pi, OpenCode Desktop, Swival, Hermes Agent, or the live MTPLX Dashboard. Without this argument, MTPLX runs an interactive onboarding (or the 'same as last time?' prompt) on first run.",
    )
    start_flow_p.add_argument(
        "--fresh",
        action="store_true",
        help="Skip the 'same as last time?' prompt and walk through the full onboarding again",
    )
    start_flow_p.add_argument(
        "--model", help="Verified model path or Hugging Face repo id"
    )
    start_flow_p.add_argument("--cache-dir")
    start_flow_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        default=DEFAULT_PROFILE_NAME,
        help="Runtime profile. Default resolves per model: Turbo for the quantized 27B and 9B flagships (the app's launch rule), Sustained otherwise. An explicit value always wins. Use --profile performance-cold --max for Burst.",
    )
    start_flow_p.add_argument(
        "--download",
        action="store_true",
        help="Download the selected/default model if it is missing",
    )
    start_flow_p.add_argument(
        "--yes",
        action="store_true",
        help="Use defaults without interactive model prompts",
    )
    start_flow_p.add_argument("--unsafe-force-unverified", action="store_true")
    start_flow_p.add_argument(
        "--prompt", help="Run one prompt and exit instead of entering the chat loop"
    )
    start_flow_p.add_argument("--system", help="Optional system prompt")
    start_flow_p.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help="Terminal response-token cap. Omit to use the model's remaining context.",
    )
    start_flow_p.add_argument("--temperature", type=float, default=0.6)
    start_flow_p.add_argument("--top-p", type=float, default=0.95)
    start_flow_p.add_argument("--top-k", type=int, default=20)
    start_flow_p.add_argument(
        "--default-presence-penalty",
        dest="default_presence_penalty",
        type=float,
        default=0.0,
    )
    start_flow_p.add_argument(
        "--default-frequency-penalty",
        dest="default_frequency_penalty",
        type=float,
        default=0.0,
    )
    start_flow_p.add_argument("--depth", type=int, default=3)
    _add_mtp_toggle_args(start_flow_p)
    start_flow_p.add_argument("--seed", type=int, default=0)
    _add_reasoning_arg(start_flow_p)
    _add_reasoning_effort_arg(start_flow_p)
    _add_preserve_thinking_arg(start_flow_p)
    _add_bridge_prompt_args(start_flow_p)
    start_flow_p.add_argument(
        "--no-stats",
        action="store_false",
        dest="show_stats",
        default=True,
        help="Hide speed stats after responses",
    )
    start_flow_p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Open WebUI server host for `mtplx start openwebui`",
    )
    start_flow_p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port for `mtplx start`; OpenCode examples use 18083 to avoid browser-chat collisions",
    )
    start_flow_p.add_argument(
        "--model-id",
        default=DEFAULT_PUBLIC_MODEL_ID,
        help="Model id to select in Open WebUI",
    )
    start_flow_p.add_argument(
        "--api-key", help="Optional API key for non-localhost Open WebUI serving"
    )
    start_flow_p.add_argument(
        "--api-key-file", help="Read the API key from a local file instead of argv/env"
    )
    start_flow_p.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="Warmup tokens for Open WebUI server startup",
    )
    start_flow_p.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Streaming chunk size for Open WebUI server",
    )
    _add_batching_args(start_flow_p)
    _add_ssd_session_cache_args(start_flow_p)
    _add_paged_kv_quant_args(start_flow_p)
    _add_adaptive_args(start_flow_p)
    start_flow_p.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Server request rate limit for Open WebUI path",
    )
    start_flow_p.add_argument(
        "--max-response-tokens",
        type=int,
        help="Server response token cap for Open WebUI path",
    )
    start_flow_p.add_argument(
        "--reasoning-parser",
        default="qwen3",
        help="Reasoning parser for Open WebUI server streaming",
    )
    start_flow_p.add_argument(
        "--strict-warmup",
        action="store_true",
        help="Fail Open WebUI startup if warmup fails",
    )
    start_flow_p.add_argument(
        "--strict-fast-path",
        action="store_true",
        help="Deprecated, no effect: MTPLX runs on stock PyPI MLX; no fork is required.",
    )
    _add_fan_mode_args(
        start_flow_p,
        max_help="Compatibility alias for --fan-mode max; combined with the sustained profile this is Sustained Max",
    )
    start_flow_p.add_argument(
        "--max-idle-min",
        type=int,
        default=15,
        help="Minutes of chat inactivity before --max drops fans back to auto (default: 15; ramps back up on next request)",
    )
    start_flow_p.add_argument(
        "--open-dashboard",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Open the live MTPLX dashboard alongside the chosen client "
            "(openwebui/pi/opencode/swival/hermes). Skips the wizard's dashboard "
            "companion prompt when set. Use --no-open-dashboard to force off."
        ),
    )
    start_flow_p.add_argument(
        "--enable-thermal-poll",
        action="store_true",
        help=(
            "Enable a 1 Hz background poll of thermal.fan_summary() so the "
            "dashboard's fan panel updates live. Off by default."
        ),
    )
    start_flow_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON for --dry-run"
    )
    start_flow_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what start will do without loading MLX",
    )
    start_flow_p.set_defaults(func=cmd_quickstart_public)

    setup_p = sub.add_parser(
        "setup", help="Set up MTPLX with a friendly guided default"
    )
    setup_p.add_argument("--config", default="~/.mtplx/config.toml")
    setup_p.add_argument(
        "--model",
        default=DEFAULT_HF_MODEL_ID,
        help="Default verified model repo id or path",
    )
    setup_p.add_argument(
        "--model-dir",
        help="Model cache directory; defaults to MTPLX_MODEL_DIR or ~/.mtplx/models",
    )
    setup_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        default=DEFAULT_PROFILE_NAME,
    )
    setup_p.add_argument("--thermal-control", choices=("auto", "none"), default="auto")
    setup_p.add_argument(
        "--download",
        action="store_true",
        help="Download the selected model into the cache",
    )
    setup_p.add_argument(
        "--force",
        action="store_true",
        help="Rewrite config even when it already exists",
    )
    setup_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    setup_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show setup actions without writing files",
    )
    setup_p.set_defaults(func=_cmd_setup)

    status_p = sub.add_parser("status", help="Check whether MTPLX is ready to run")
    status_p.add_argument("--project-root", default=".")
    status_p.add_argument("--model-cache")
    status_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    status_p.add_argument(
        "--deep",
        action="store_true",
        help="Include launchers, config, staging, release, and integration checks",
    )
    status_p.set_defaults(func=cmd_doctor)

    stop_p = sub.add_parser("stop", help="Stop the running MTPLX server")
    stop_p.add_argument("--host", default="127.0.0.1")
    stop_p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port. Omit to find the running server automatically.",
    )
    stop_p.add_argument(
        "--grace-seconds",
        type=float,
        default=10.0,
        help="Seconds to wait after SIGTERM before escalating to SIGKILL.",
    )
    stop_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    stop_p.set_defaults(func=cmd_stop_public)

    settings_p = sub.add_parser(
        "settings",
        help="Read or change live MTPLX server settings",
    )
    settings_p.add_argument(
        "settings_action",
        nargs="?",
        choices=["get", "set"],
        default="get",
        help="get prints current settings; set applies key=value pairs.",
    )
    settings_p.add_argument(
        "pairs",
        nargs="*",
        help="key=value pairs for set, e.g. depth=2 reasoning=off",
    )
    settings_p.add_argument("--host", default="127.0.0.1")
    settings_p.add_argument("--port", type=int, default=8000)
    settings_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    settings_p.set_defaults(func=cmd_settings_public)

    ask_p = sub.add_parser(
        "ask", help="Ask the verified local MTPLX model one question"
    )
    ask_p.add_argument("prompt_arg", nargs="?", help="Prompt text")
    ask_p.add_argument("--model", default=default_model)
    ask_p.add_argument("--cache-dir")
    ask_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        default=DEFAULT_PROFILE_NAME,
    )
    ask_p.add_argument("--unsafe-force-unverified", action="store_true")
    ask_p.add_argument(
        "--yes", action="store_true", help="Confirm unsafe non-interactive actions"
    )
    ask_p.add_argument(
        "--prompt", help="Prompt text, as an alternative to the positional prompt"
    )
    ask_p.add_argument("--system", help="Optional system prompt")
    ask_p.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help="Response-token cap. Omit to use the model's remaining context.",
    )
    ask_p.add_argument("--temperature", type=float, default=0.6)
    ask_p.add_argument("--top-p", type=float, default=0.95)
    ask_p.add_argument("--top-k", type=int, default=20)
    ask_p.add_argument("--depth", type=int, default=3)
    _add_mtp_toggle_args(ask_p)
    ask_p.add_argument("--seed", type=int, default=0)
    _add_reasoning_arg(ask_p)
    ask_p.add_argument(
        "--stats",
        action="store_false",
        dest="quiet",
        default=True,
        help="Show the MTPLX stats footer",
    )
    ask_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    ask_p.add_argument("--expect-python", action="store_true")
    _add_fan_mode_args(
        ask_p,
        max_help="Compatibility alias for --fan-mode max for this run",
    )
    ask_p.set_defaults(func=cmd_run_public)

    quickstart_server_p = sub.add_parser(
        "quickstart",
        aliases=["quick-start"],
        help="Start the local MTPLX server",
    )
    quickstart_server_p.add_argument("--model", default=default_model)
    quickstart_server_p.add_argument("--cache-dir")
    quickstart_server_p.add_argument(
        "--download",
        action="store_true",
        help="Download a Hugging Face model before starting if it is not cached",
    )
    quickstart_server_p.add_argument(
        "--agent-rewrites",
        choices=["on", "off"],
        default=None,
        help=(
            "Agent transcript rewriting: unset = passthrough (default), "
            "on = legacy rewrite machinery, off = hard passthrough guarantee."
        ),
    )
    quickstart_server_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        default=DEFAULT_PROFILE_NAME,
        help="Runtime profile. Default resolves per model (Turbo for the quantized 27B and 9B flagships, Sustained otherwise); use --profile performance-cold --max for Burst.",
    )
    quickstart_server_p.add_argument("--unsafe-force-unverified", action="store_true")
    quickstart_server_p.add_argument(
        "--yes", action="store_true", help="Confirm unsafe non-interactive actions"
    )
    quickstart_server_p.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Default 127.0.0.1 is this Mac only; 0.0.0.0 shares "
            "the API with other devices and VM guests (requires an API key — "
            "add --api-key-file ~/.mtplx/api-key to generate one)"
        ),
    )
    quickstart_server_p.add_argument("--port", type=int, default=8000)
    quickstart_server_p.add_argument("--model-id", default=DEFAULT_PUBLIC_MODEL_ID, help="Served OpenAI model id; defaults to the loaded artifact identity")
    quickstart_server_p.add_argument(
        "--embedding-model",
        action="append",
        default=[],
        metavar="REF[=SERVED_ID]",
        help="Serve REF on /v1/embeddings (repeatable). Loaded on first request.",
    )
    quickstart_server_p.add_argument(
        "--reranker-model",
        action="append",
        default=[],
        metavar="REF[=SERVED_ID]",
        help="Serve REF on /v1/rerank (repeatable). The same REF in both roles loads once.",
    )
    quickstart_server_p.add_argument(
        "--retrieval-max-resident",
        type=int,
        default=2,
        help="How many retrieval models stay in memory; least-recently-used are unloaded",
    )
    quickstart_server_p.add_argument(
        "--retrieval-idle-timeout",
        type=float,
        default=0.0,
        help="Unload retrieval models after this many idle seconds (0 = never)",
    )
    quickstart_server_p.add_argument(
        "--retrieval-max-tokens",
        type=int,
        default=0,
        help="Truncate retrieval inputs to this many tokens (0 = per-model default)",
    )
    quickstart_server_p.add_argument(
        "--retrieval-trust-remote-code",
        action="store_true",
        help=(
            "Allow retrieval checkpoints that ship their own Python "
            "(jina-style model.py/rerank.py) to execute it; off by default"
        ),
    )
    quickstart_server_p.add_argument("--dry-run", action="store_true", help="Preview the server launch command without loading MLX")
    quickstart_server_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON for --dry-run and errors")
    quickstart_server_p.add_argument("--depth", type=int, default=3)
    _add_mtp_toggle_args(quickstart_server_p)
    quickstart_server_p.add_argument(
        "--api-key",
        default=None,
        help="Require Bearer or X-API-Key auth. Required for non-localhost binds.",
    )
    quickstart_server_p.add_argument(
        "--api-key-file",
        help=(
            "Read the API key from a local file instead of argv/env. "
            "A missing file is created with a fresh key (printed once)."
        ),
    )
    quickstart_server_p.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Requests per minute per client/API key. Use 0 to disable.",
    )
    quickstart_server_p.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Committed-token batch size per chat SSE chunk.",
    )
    _add_batching_args(quickstart_server_p)
    _add_ssd_session_cache_args(quickstart_server_p)
    _add_paged_kv_quant_args(quickstart_server_p)
    _add_adaptive_args(quickstart_server_p)
    quickstart_server_p.add_argument(
        "--max-tokens",
        dest="max_response_tokens",
        type=int,
        help="Default server-side response-token ceiling.",
    )
    quickstart_server_p.add_argument(
        "--default-temperature", dest="temperature", type=float, default=0.6
    )
    quickstart_server_p.add_argument(
        "--default-top-p", dest="top_p", type=float, default=0.95
    )
    quickstart_server_p.add_argument(
        "--default-top-k", "--top-k", dest="top_k", type=int, default=20
    )
    quickstart_server_p.add_argument(
        "--default-presence-penalty",
        dest="default_presence_penalty",
        type=float,
        default=0.0,
    )
    quickstart_server_p.add_argument(
        "--default-frequency-penalty",
        dest="default_frequency_penalty",
        type=float,
        default=0.0,
    )
    quickstart_server_p.add_argument("--draft-temperature", type=float)
    quickstart_server_p.add_argument("--draft-top-p", type=float)
    quickstart_server_p.add_argument("--draft-top-k", type=int)
    _add_reasoning_arg(quickstart_server_p)
    _add_reasoning_effort_arg(quickstart_server_p)
    quickstart_server_p.add_argument(
        "--reasoning-parser",
        choices=["qwen3", "step3p5", "gemma4", "poolside_v1", "none"],
        default="qwen3",
    )
    _add_preserve_thinking_arg(quickstart_server_p)
    _add_bridge_prompt_args(quickstart_server_p)
    quickstart_server_p.add_argument(
        "--stats-footer",
        action="store_true",
        dest="stats_footer",
        default=False,
        help="Append visible MTPLX speed stats to returned text.",
    )
    quickstart_server_p.add_argument(
        "--no-stats-footer",
        action="store_false",
        dest="stats_footer",
        help="Keep returned text clean for UI clients. This is the default for quickstart.",
    )
    _add_fan_mode_args(
        quickstart_server_p,
        max_help=(
            "Compatibility alias for --fan-mode max for the server lifetime; "
            "combined with the sustained profile this is Sustained Max"
        ),
    )
    quickstart_server_p.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local browser chat after the server starts",
    )
    quickstart_server_p.add_argument(
        "--max-idle-min",
        type=int,
        default=15,
        help="Minutes of chat inactivity before --max drops fans back to auto (default: 15; ramps back up on next request)",
    )
    quickstart_server_p.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="Startup warmup generation length. Use 0 to disable.",
    )
    quickstart_server_p.add_argument(
        "--strict-warmup",
        action="store_true",
        help="Fail server startup if the warmup pass fails.",
    )
    quickstart_server_p.add_argument(
        "--strict-fast-path",
        action="store_true",
        help="Deprecated, no effect: MTPLX runs on stock PyPI MLX; no fork is required.",
    )
    quickstart_server_p.set_defaults(func=cmd_serve_public)

    connect_p = sub.add_parser(
        "connect",
        help="Show client setup for Open WebUI, Claude Code, OpenCode, or Swival",
    )
    connect_p.add_argument(
        "integration",
        nargs="?",
        choices=["openwebui", "claude-code", "opencode", "swival"],
    )
    connect_p.add_argument("--host", default="127.0.0.1")
    connect_p.add_argument("--port", type=int, default=8000)
    connect_p.add_argument("--model-id", default=DEFAULT_PUBLIC_MODEL_ID)
    connect_p.add_argument("--api-key-env", default="MTPLX_API_KEY")
    connect_p.add_argument(
        "--docker",
        action="store_true",
        help="Include the Dockerized Open WebUI host.docker.internal command",
    )
    connect_p.add_argument("--webui-port", type=int, default=3000)
    connect_p.add_argument(
        "--single-user",
        action="store_true",
        help="Emit WEBUI_AUTH=False for a new single-user Open WebUI data volume",
    )
    connect_p.add_argument(
        "--api-key",
        default="mtplx-local",
        help="OpenAI-compatible API key value for generated Docker command",
    )
    connect_p.add_argument("--smoke", action="store_true")
    connect_p.add_argument("--timeout", type=float, default=5.0)
    connect_p.add_argument("--context-window", type=int, default=262144)
    connect_p.add_argument("--json", action="store_true")
    connect_p.set_defaults(func=_cmd_connect)

    openwebui_p = sub.add_parser("openwebui", help="Open WebUI integration helpers")
    openwebui_sub = openwebui_p.add_subparsers(dest="openwebui_action", required=True)
    openwebui_docker_p = openwebui_sub.add_parser(
        "docker-command", help="Print the production Open WebUI Docker command"
    )
    openwebui_docker_p.add_argument("--mtplx-port", type=int, default=8000)
    openwebui_docker_p.add_argument("--webui-port", type=int, default=3000)
    openwebui_docker_p.add_argument(
        "--single-user",
        action="store_true",
        help="Add WEBUI_AUTH=False for a fresh single-user volume",
    )
    openwebui_docker_p.add_argument("--api-key", default="mtplx-local")
    openwebui_docker_p.add_argument("--json", action="store_true")
    openwebui_docker_p.set_defaults(func=cmd_openwebui_public)

    models_p = sub.add_parser(
        "models",
        help="List locally cached MTPLX models; check for and apply pack updates",
    )
    models_p.add_argument("--cache-dir")
    models_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    models_p.add_argument(
        "--check",
        action="store_true",
        help="Compare cached packs against published revisions (network)",
    )
    models_p.add_argument(
        "--update",
        nargs="*",
        metavar="REPO",
        default=None,
        help=(
            "Update model packs in place (delta download). With no REPO, "
            "updates every pack that has a newer published revision."
        ),
    )
    models_p.add_argument(
        "--progress-json",
        action="store_true",
        help="With --update: emit pull-style JSON progress events (one per line)",
    )
    models_p.add_argument(
        "--installed-path",
        help="With one --update REPO: update this exact installed pack directory",
    )
    models_p.set_defaults(func=cmd_list_public)

    env_p = sub.add_parser("env", help="Print reproducible environment snapshot")
    env_p.add_argument("--project-root", default=".")
    env_p.set_defaults(func=_cmd_env)

    doctor_p = sub.add_parser(
        "doctor", help="Check MTPLX CLI, model, thermal, and tool environment"
    )
    doctor_p.add_argument(
        "topic",
        nargs="?",
        choices=["opencode", "pi", "android-studio"],
        help="Optional focused doctor target",
    )
    doctor_p.add_argument("--project-root", default=".")
    doctor_p.add_argument("--host", default="127.0.0.1")
    doctor_p.add_argument(
        "--port",
        type=int,
        default=8008,
        help=(
            "Port for the topic bridge checks (opencode/pi/android-studio; "
            "default 8008). When passed explicitly it also aims the MTPLX "
            "server checks, which otherwise probe the shipped default :8000."
        ),
    )
    doctor_p.add_argument("--base-url")
    doctor_p.add_argument(
        "--smc-path",
        default=os.environ.get("MTPLX_SMC_PATH") or shutil.which("smc") or "",
    )
    doctor_p.add_argument(
        "--sovereign-path",
        default=os.environ.get("MTPLX_SOVEREIGN_PATH")
        or shutil.which("sovereign")
        or "",
    )
    doctor_p.add_argument("--model-cache")
    doctor_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    doctor_p.add_argument(
        "--deep",
        action="store_true",
        help="Include launchers, config, staging, release, and integration checks",
    )
    doctor_p.add_argument(
        "--summary", action="store_true", help="Print a compact check summary"
    )
    doctor_p.add_argument(
        "--bundle",
        action="store_true",
        help="Write a redacted doctor bundle under ~/.mtplx/reports",
    )
    doctor_p.add_argument("--output-dir", help="Directory for --bundle output")
    doctor_p.add_argument(
        "--include-paths",
        action="store_true",
        help="Keep local paths in --bundle output",
    )
    doctor_p.set_defaults(func=cmd_doctor)

    tune_p = sub.add_parser(
        "tune", help="Find the fastest AR/MTP draft control for this Mac"
    )
    tune_p.add_argument("--model", default=default_model)
    tune_p.add_argument("--cache-dir")
    tune_p.add_argument(
        "--depths",
        default=None,
        help="Comma-separated MTP depths or Gemma draft blocks to compare against AR",
    )
    tune_p.add_argument("--max-tokens", type=int, default=512)
    tune_p.add_argument("--limit", type=int, default=1)
    tune_p.add_argument("--seed", type=int, default=0)
    tune_p.add_argument("--run-id")
    tune_p.add_argument("--output-dir")
    tune_p.add_argument("--output")
    tune_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    tune_p.add_argument(
        "--verbose", action="store_true", help="Show verify and acceptance details"
    )
    tune_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show candidate commands without loading MLX",
    )
    tune_p.add_argument(
        "--no-save", action="store_true", help="Do not save the winning depth"
    )
    tune_p.add_argument(
        "--retune", action="store_true", help="Ignore saved tuning and measure again"
    )
    tune_p.add_argument("--unsafe-force-unverified", action="store_true")
    tune_p.add_argument(
        "--yes", action="store_true", help="Confirm unsafe non-interactive actions"
    )
    tune_p.add_argument(
        "--require-max-fans",
        action="store_true",
        help="Fail before tuning if verified max-fan mode cannot start.",
    )
    tune_p.add_argument(
        "--temperature", type=float, default=0.6, help=argparse.SUPPRESS
    )
    tune_p.add_argument("--top-p", type=float, default=0.95, help=argparse.SUPPRESS)
    tune_p.add_argument("--top-k", type=int, default=20, help=argparse.SUPPRESS)
    tune_p.add_argument(
        "--base-hidden-variant",
        choices=["pre_norm", "post_norm"],
        help="Target-model hidden contract; defaults to mtplx_runtime.json",
    )
    tune_p.add_argument(
        "--mtp-hidden-variant",
        help="MTP recursive hidden contract; defaults to mtplx_runtime.json",
    )
    tune_p.add_argument(
        "--concat-order",
        choices=["embedding_hidden", "hidden_embedding"],
        help="MTP fc concat order; defaults to mtplx_runtime.json",
    )
    tune_p.add_argument(
        "--mtp-cache-policy",
        choices=["persistent", "fresh"],
        default="persistent",
        help=argparse.SUPPRESS,
    )
    tune_p.add_argument(
        "--mtp-history-policy",
        choices=[
            "auto",
            "committed",
            "full",
            "last-window",
            "last_window",
            "cycle",
            "none",
        ],
        default="committed",
        help=argparse.SUPPRESS,
    )
    tune_p.add_argument("--draft-temperature", type=float, help=argparse.SUPPRESS)
    tune_p.add_argument(
        "--draft-core",
        choices=["stock", "device-d2", "device"],
        default="stock",
        help=argparse.SUPPRESS,
    )
    tune_p.add_argument("--draft-top-p", type=float, help=argparse.SUPPRESS)
    tune_p.add_argument("--draft-top-k", type=int, help=argparse.SUPPRESS)
    tune_p.add_argument("--prompt-suite", help=argparse.SUPPRESS)
    tune_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        # No parser default: tune's profile resolves like serve's launch
        # rule (per-model turbo for the flagships) so depth is measured
        # under the kernels the launch profile actually uses. An explicit
        # --profile always wins.
        default=None,
        help=argparse.SUPPRESS,
    )
    tune_p.add_argument(
        "--_candidate",
        choices=["ar", "1", "2", "3", "4", "5", "6", "7", "8"],
        dest="_tune_candidate",
        help=argparse.SUPPRESS,
    )
    tune_p.add_argument(
        "--_candidate-output", dest="_tune_candidate_output", help=argparse.SUPPRESS
    )
    tune_p.set_defaults(func=cmd_tune_public)

    report_p = sub.add_parser("report", help="Create a redacted MTPLX support bundle")
    report_p.add_argument("--project-root", default=".")
    report_p.add_argument(
        "--smc-path",
        default=os.environ.get("MTPLX_SMC_PATH") or shutil.which("smc") or "",
    )
    report_p.add_argument(
        "--sovereign-path",
        default=os.environ.get("MTPLX_SOVEREIGN_PATH")
        or shutil.which("sovereign")
        or "",
    )
    report_p.add_argument("--model-cache")
    report_p.add_argument("--output-dir", help="Directory for the report bundle")
    report_p.add_argument(
        "--include-paths", action="store_true", help="Keep local paths in the report"
    )
    report_p.add_argument(
        "--deep",
        action="store_true",
        default=True,
        help="Include deep integration checks",
    )
    report_p.add_argument(
        "--summary",
        action="store_true",
        help="Print compact check summary instead of JSON",
    )
    report_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    report_p.set_defaults(func=cmd_doctor, bundle=True)

    trace_p = sub.add_parser(
        "trace",
        help="Diagnose agent/coding sessions: join serve receipts, flight samples, and OpenCode history",
    )
    trace_sub = trace_p.add_subparsers(dest="trace_action", required=True)

    def _trace_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--port",
            type=int,
            default=None,
            help="serve port (default: newest request log)",
        )
        p.add_argument(
            "--db",
            default=os.path.expanduser("~/.local/share/opencode/opencode.db"),
            help="opencode.db path",
        )
        p.add_argument("--json", action="store_true", help="machine-readable output")

    trace_sessions_p = trace_sub.add_parser(
        "sessions", help="List recent OpenCode sessions with server-request matches"
    )
    _trace_common(trace_sessions_p)
    trace_sessions_p.add_argument("--limit", type=int, default=15)
    trace_sessions_p.set_defaults(func=cmd_trace_public)

    trace_session_p = trace_sub.add_parser(
        "session",
        help="Per-turn timeline for one session (cache, TPS, postcommit, canon, pathology flags)",
    )
    _trace_common(trace_session_p)
    trace_session_p.add_argument(
        "session", nargs="?", default="latest", help="ses_... id, substring, or 'latest'"
    )
    trace_session_p.set_defaults(func=cmd_trace_public)

    trace_request_p = trace_sub.add_parser(
        "request", help="Deep-dive one request receipt + per-second flight curve"
    )
    _trace_common(trace_request_p)
    trace_request_p.add_argument(
        "request",
        nargs="?",
        default="latest",
        help="request_id substring, receipt index, or 'latest'",
    )
    trace_request_p.add_argument(
        "--all", action="store_true", help="show every receipt field"
    )
    trace_request_p.set_defaults(func=cmd_trace_public)

    trace_autopsy_p = trace_sub.add_parser(
        "autopsy",
        help="Extract + analyze a turn's reasoning (loop metrics, dup paragraphs, dump to file)",
    )
    _trace_common(trace_autopsy_p)
    trace_autopsy_p.add_argument("session", nargs="?", default="latest")
    trace_autopsy_p.add_argument(
        "--turn",
        type=int,
        default=None,
        help="1-based assistant turn (default: biggest think)",
    )
    trace_autopsy_p.set_defaults(func=cmd_trace_public)

    trace_live_p = trace_sub.add_parser(
        "live", help="Live in-flight status from the serve flight endpoint"
    )
    _trace_common(trace_live_p)
    trace_live_p.add_argument(
        "--watch", action="store_true", help="poll continuously"
    )
    trace_live_p.add_argument("--interval", type=float, default=2.0)
    trace_live_p.set_defaults(func=cmd_trace_public)

    trace_report_p = trace_sub.add_parser(
        "report",
        help="Self-contained HTML report with historical graphs for a session",
    )
    _trace_common(trace_report_p)
    trace_report_p.add_argument("session", nargs="?", default="latest")
    trace_report_p.add_argument(
        "--out",
        default=None,
        help="output HTML path (default: ~/.mtplx/metrics/reports/<ses>.html)",
    )
    trace_report_p.add_argument(
        "--open", action="store_true", help="open in browser when written"
    )
    trace_report_p.set_defaults(func=cmd_trace_public)

    inspect_public_p = sub.add_parser(
        "inspect", help="Inspect a model and auto-check MTP support"
    )
    inspect_public_p.add_argument(
        "model_args",
        nargs="*",
        metavar="MODEL",
        help="Model path/repo id. Legacy form 'inspect model MODEL' is also accepted.",
    )
    inspect_public_p.add_argument("--model")
    inspect_public_p.add_argument("--require-mtp", action="store_true")
    inspect_public_p.add_argument(
        "--no-strict-exit-code",
        action="store_false",
        dest="strict_exit_code",
        help="Always exit 0 after printing the compatibility verdict.",
    )
    inspect_public_p.set_defaults(strict_exit_code=True)
    inspect_public_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    inspect_public_p.set_defaults(func=cmd_inspect_model_public)

    forge_p = sub.add_parser(
        "forge",
        help="Forge, verify, brand, discover, and publish MTPLX MTP models",
    )
    forge_sub = forge_p.add_subparsers(dest="forge_action", required=True)

    forge_probe_p = forge_sub.add_parser(
        "probe",
        help="Classify a Forge source without downloading full weights",
    )
    forge_probe_p.add_argument(
        "source", help="Hugging Face repo, HF URL, or local path"
    )
    forge_probe_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable probe result"
    )
    forge_probe_p.set_defaults(func=cmd_forge_public)

    forge_build_p = forge_sub.add_parser(
        "build",
        help="Download/convert/verify/brand a local MTPLX artifact",
    )
    forge_build_p.add_argument(
        "--repo", required=True, help="Source Hugging Face repo, HF URL, or local path"
    )
    forge_build_p.add_argument("--out", required=True, help="Progress output root")
    forge_build_p.add_argument("--run-id", required=True, help="Run id under --out")
    forge_build_p.add_argument("--recipe", required=True, help="Forge recipe JSON")
    forge_build_p.add_argument(
        "--branded-name", required=True, help="Local MTPLX artifact name"
    )
    forge_build_p.add_argument(
        "--max", action="store_true", help="Opt into max-fan verification"
    )
    forge_build_p.add_argument(
        "--max-tokens", type=int, default=2048, help="Verification response budget"
    )
    forge_build_p.add_argument("--suite", help="Verification prompt suite")
    forge_build_p.add_argument(
        "--dtype",
        choices=["auto", "bf16", "fp16"],
        help="Dtype for non-quantized parameters (overrides recipe body_dtype). "
        "fp16 prompt-processes faster on M1/M2 Macs, which have no native "
        "BF16; auto picks fp16 on those chips",
    )
    forge_build_p.add_argument(
        "--allow-degraded-mtp",
        action="store_true",
        help="Allow recipes that intentionally requantize MTP weights",
    )
    forge_build_p.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    forge_build_p.set_defaults(func=cmd_forge_public)

    forge_discover_p = forge_sub.add_parser(
        "discover",
        help="Search Hugging Face for MTPLX-branded models",
    )
    forge_discover_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable cards"
    )
    forge_discover_p.add_argument("--query", help="Search text; defaults to MTPLX")
    # 100 = the engine's per-call cap and the app wall's page size; 20 hid
    # everything below the download-rank fold (same trap as the old wall).
    forge_discover_p.add_argument("--limit", type=int, default=100)
    forge_discover_p.add_argument("--offset", type=int, default=0)
    forge_discover_p.set_defaults(func=cmd_forge_public)

    forge_publish_p = forge_sub.add_parser(
        "publish",
        help="Upload a local Forge artifact to Hugging Face",
    )
    forge_publish_p.add_argument(
        "--path", required=True, help="Local forged model directory"
    )
    forge_publish_p.add_argument(
        "--repo", required=True, help="Destination owner/name repo"
    )
    forge_publish_p.add_argument(
        "--visibility", choices=("public", "private"), required=True
    )
    forge_publish_p.add_argument("--license", required=True, help="SPDX license id")
    forge_publish_p.add_argument("--out", required=True, help="Progress output root")
    forge_publish_p.add_argument("--run-id", required=True, help="Run id under --out")
    forge_publish_p.add_argument("--token", choices=("stdin",), required=True)
    forge_publish_p.add_argument("--readme-path")
    forge_publish_p.set_defaults(func=cmd_forge_public)

    forge_inspect_p = forge_sub.add_parser(
        "inspect",
        help="Read a forged artifact's mtplx_runtime.json",
    )
    forge_inspect_p.add_argument("path")
    forge_inspect_p.add_argument("--json", action="store_true")
    forge_inspect_p.set_defaults(func=cmd_forge_public)

    forge_verify_p = forge_sub.add_parser(
        "verify",
        help="Run Forge-shaped verification rows for a local model",
    )
    forge_verify_p.add_argument("path")
    forge_verify_p.add_argument("--json", action="store_true")
    forge_verify_p.add_argument("--out")
    forge_verify_p.add_argument("--run-id")
    forge_verify_p.add_argument("--max", action="store_true")
    forge_verify_p.add_argument("--max-tokens", type=int, default=2048)
    forge_verify_p.add_argument("--suite", help="Verification prompt suite")
    forge_verify_p.set_defaults(func=cmd_forge_public)

    forge_cancel_p = forge_sub.add_parser(
        "cancel",
        help="Best-effort cancellation marker for an in-flight Forge run",
    )
    forge_cancel_p.add_argument("run_id")
    forge_cancel_p.set_defaults(func=cmd_forge_public)

    init_p = sub.add_parser(
        "init", help="Initialize MTPLX user config without importing MLX"
    )
    init_p.add_argument("--config", default="~/.mtplx/config.toml")
    init_p.add_argument(
        "--model",
        default=DEFAULT_HF_MODEL_ID,
        help="Default verified model repo id or path",
    )
    init_p.add_argument(
        "--model-dir",
        help="Model cache directory; defaults to MTPLX_MODEL_DIR or ~/.mtplx/models",
    )
    init_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        default=DEFAULT_PROFILE_NAME,
    )
    init_p.add_argument("--thermal-control", choices=("auto", "none"), default="auto")
    init_p.add_argument(
        "--download",
        action="store_true",
        help="Download the selected model into the cache",
    )
    init_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    init_p.add_argument(
        "--dry-run", action="store_true", help="Show init actions without writing files"
    )
    init_p.add_argument(
        "--write", action="store_true", help="Write the initial config file"
    )
    init_p.set_defaults(func=_cmd_init)

    profiles_p = sub.add_parser(
        "profiles", help="List MTPLX runtime profiles without importing MLX"
    )
    profiles_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    profiles_p.set_defaults(func=_cmd_profiles)

    pull_p = sub.add_parser(
        "pull", help="Download a Hugging Face model into the MTPLX cache"
    )
    pull_p.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_HF_MODEL_ID,
        help="Hugging Face repo id or URL. Defaults to the verified speed model.",
    )
    pull_p.add_argument("--cache-dir")
    pull_p.add_argument("--revision")
    pull_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    pull_p.add_argument(
        "--progress-json",
        action="store_true",
        help="Emit newline-delimited JSON progress events",
    )
    pull_p.set_defaults(func=cmd_pull_public)

    list_p = sub.add_parser("list", help="List locally cached MTPLX models")
    list_p.add_argument("--cache-dir")
    list_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    list_p.set_defaults(func=cmd_list_public)

    remove_p = sub.add_parser("remove", help="Remove a locally cached MTPLX model")
    remove_p.add_argument(
        "model", help="Hugging Face repo id, URL, or cached safe name"
    )
    remove_p.add_argument("--cache-dir")
    remove_p.add_argument("--missing-ok", action="store_true")
    remove_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    remove_p.set_defaults(func=cmd_remove_public)

    run_p = sub.add_parser("run", help="Run a one-shot verified MTPLX completion")
    run_p.add_argument("prompt_arg", nargs="?", help="Prompt text")
    run_p.add_argument("--model", default=default_model)
    run_p.add_argument("--cache-dir")
    run_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        default=DEFAULT_PROFILE_NAME,
    )
    run_p.add_argument("--unsafe-force-unverified", action="store_true")
    run_p.add_argument(
        "--yes", action="store_true", help="Confirm unsafe non-interactive actions"
    )
    run_p.add_argument(
        "--prompt", help="Prompt text, as an alternative to the positional prompt"
    )
    run_p.add_argument("--system", help="Optional system prompt")
    run_p.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help="Response-token cap. Omit to use the model's remaining context.",
    )
    run_p.add_argument("--temperature", type=float, default=0.6)
    run_p.add_argument("--top-p", type=float, default=0.95)
    run_p.add_argument("--top-k", type=int, default=20)
    run_p.add_argument("--depth", type=int, default=3)
    _add_mtp_toggle_args(run_p)
    run_p.add_argument("--seed", type=int, default=0)
    _add_reasoning_arg(run_p)
    run_p.add_argument("--quiet", action="store_true", help="Hide the stats footer")
    run_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    run_p.add_argument("--expect-python", action="store_true")
    _add_fan_mode_args(
        run_p,
        max_help="Compatibility alias for --fan-mode max for this run",
    )
    run_p.set_defaults(func=cmd_run_public)

    chat_p = sub.add_parser("chat", help="Run one native-MTP chat smoke generation")
    chat_p.add_argument("--model", default=default_model)
    chat_p.add_argument("--cache-dir")
    chat_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        default=DEFAULT_PROFILE_NAME,
    )
    chat_p.add_argument("--unsafe-force-unverified", action="store_true")
    chat_p.add_argument(
        "--yes", action="store_true", help="Confirm unsafe non-interactive actions"
    )
    chat_p.add_argument("--prompt", required=True)
    chat_p.add_argument(
        "--max-tokens",
        type=_positive_int,
        default=None,
        help="Response-token cap. Omit to use the model's remaining context.",
    )
    chat_p.add_argument("--temperature", type=float, default=0.6)
    chat_p.add_argument("--top-p", type=float, default=0.95)
    chat_p.add_argument("--top-k", type=int, default=20)
    chat_p.add_argument("--depth", type=int, default=3)
    _add_mtp_toggle_args(chat_p)
    chat_p.add_argument("--seed", type=int, default=0)
    _add_reasoning_arg(chat_p)
    chat_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    chat_p.add_argument("--expect-python", action="store_true")
    _add_fan_mode_args(
        chat_p,
        max_help="Compatibility alias for --fan-mode max for this run",
    )
    chat_p.set_defaults(func=cmd_chat_public)

    serve_p = sub.add_parser(
        "serve", help="Choose model/mode, then start the OpenAI-compatible MTPLX server"
    )
    serve_p.add_argument("--model", default=default_model)
    serve_p.add_argument("--cache-dir")
    serve_p.add_argument(
        "--download",
        action="store_true",
        help="Download a Hugging Face model before starting if it is not cached",
    )
    serve_p.add_argument(
        "--profile",
        type=_profile_arg,
        metavar=_PROFILE_METAVAR,
        default=DEFAULT_PROFILE_NAME,
        help=(
            "Runtime profile. Default resolves per model: Turbo for the "
            "quantized 27B and 9B flagships (the app's launch rule), "
            "Sustained otherwise; use --profile performance-cold "
            "--max for Burst."
        ),
    )
    serve_p.add_argument("--unsafe-force-unverified", action="store_true")
    serve_p.add_argument(
        "--agent-rewrites",
        choices=["on", "off"],
        default=None,
        help=(
            "Agent transcript rewriting. Unset (default) is passthrough: no "
            "tool-result compaction, no injected steering contracts, no "
            "heuristic toolset filtering; per-feature MTPLX_*_COMPACT_"
            "THRESHOLD_CHARS env limits can re-enable individual compactors. "
            "on restores the full legacy rewrite machinery. off is a hard "
            "passthrough guarantee that also overrides per-feature env opt-ins."
        ),
    )
    serve_p.add_argument(
        "--yes", action="store_true", help="Confirm unsafe non-interactive actions"
    )
    serve_p.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Bind address. Default 127.0.0.1 is this Mac only; 0.0.0.0 shares "
            "the API with other devices and VM guests (requires an API key — "
            "add --api-key-file ~/.mtplx/api-key to generate one)"
        ),
    )
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable API-key auth for localhost binds (non-localhost still requires a key)",
    )
    serve_p.add_argument("--depth", type=int, default=3)
    _add_mtp_toggle_args(serve_p)
    serve_p.add_argument(
        "--generation-mode",
        choices=["mtp", "ar", "auto"],
        default=None,
        help=(
            "Daemon decode mode. AR is target-only generation; MTP is native "
            "speculative decode. auto means the engine default and exists "
            "for configs written by older app builds."
        ),
    )
    serve_p.add_argument(
        "--load-mtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Load the model MTP sidecar. Use --no-load-mtp only for stock AR "
            "diagnostics or UI fallback checks."
        ),
    )
    serve_p.add_argument(
        "--stock-ar",
        action="store_true",
        help="Diagnostic only: target AR without loading the MTP sidecar.",
    )
    serve_p.add_argument(
        "--api-key",
        default=None,
        help="Require Bearer or X-API-Key auth. Required for non-localhost binds.",
    )
    serve_p.add_argument(
        "--api-key-file",
        help=(
            "Read the API key from a local file instead of argv/env. "
            "A missing file is created with a fresh key (printed once)."
        ),
    )
    serve_p.add_argument(
        "--rate-limit",
        type=int,
        default=0,
        help="Requests per minute per client/API key. Use 0 to disable.",
    )
    serve_p.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="Committed-token batch size per chat SSE chunk.",
    )
    _add_batching_args(serve_p)
    _add_ssd_session_cache_args(serve_p)
    _add_paged_kv_quant_args(serve_p)
    _add_adaptive_args(serve_p)
    serve_p.add_argument(
        "--max-tokens",
        dest="max_response_tokens",
        type=int,
        help="Default server-side response-token ceiling.",
    )
    serve_p.add_argument(
        "--context-window",
        type=_positive_int,
        help="Override context window. Default reads the model/tokenizer config.",
    )
    serve_p.add_argument(
        "--default-temperature",
        "--temperature",
        dest="temperature",
        type=float,
        default=0.6,
    )
    serve_p.add_argument(
        "--default-top-p", "--top-p", dest="top_p", type=float, default=0.95
    )
    serve_p.add_argument(
        "--default-top-k", "--top-k", dest="top_k", type=int, default=20
    )
    serve_p.add_argument(
        "--default-presence-penalty",
        dest="default_presence_penalty",
        type=float,
        default=0.0,
    )
    serve_p.add_argument(
        "--default-frequency-penalty",
        dest="default_frequency_penalty",
        type=float,
        default=0.0,
    )
    serve_p.add_argument("--draft-temperature", type=float)
    serve_p.add_argument("--draft-top-p", type=float)
    serve_p.add_argument("--draft-top-k", type=int)
    serve_p.add_argument(
        "--verify-strategy",
        choices=[
            "batched",
            "sequential",
            "capture",
            "capture_commit",
            "graphbank",
            "graphbank_capture_commit",
            "trim_commit",
            "target_prefix",
        ],
        default="capture_commit",
        help="Server MTP verification strategy.",
    )
    serve_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="linear-gdn-from-conv-tape",
        help="Server verification core.",
    )
    serve_p.add_argument(
        "--draft-core",
        choices=["stock", "device-d2", "device"],
        default="stock",
        help=(
            "Server DraftCore backend. 'device' keeps the whole draft chain "
            "on device with a single sync per cycle."
        ),
    )
    serve_p.add_argument("--mtp-adapter", type=Path)
    serve_p.add_argument(
        "--merge-mtp-adapter",
        action="store_true",
        help="Merge an installed MTP LoRA adapter into the MTP layers at load time.",
    )
    serve_p.add_argument("--mtp-quant-bits", type=int)
    serve_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    serve_p.add_argument(
        "--mtp-quant-mode", choices=["affine", "symmetric"], default="affine"
    )
    _add_reasoning_arg(serve_p)
    _add_reasoning_effort_arg(serve_p)
    serve_p.add_argument(
        "--reasoning-parser",
        choices=["qwen3", "step3p5", "gemma4", "poolside_v1", "none"],
        default="qwen3",
    )
    _add_preserve_thinking_arg(serve_p)
    _add_bridge_prompt_args(serve_p)
    serve_p.add_argument(
        "--model-id",
        default=DEFAULT_PUBLIC_MODEL_ID,
        help="Served OpenAI model id; defaults to the loaded artifact identity",
    )
    serve_p.add_argument(
        "--embedding-model",
        action="append",
        default=[],
        metavar="REF[=SERVED_ID]",
        help="Serve REF on /v1/embeddings (repeatable). Loaded on first request.",
    )
    serve_p.add_argument(
        "--reranker-model",
        action="append",
        default=[],
        metavar="REF[=SERVED_ID]",
        help="Serve REF on /v1/rerank (repeatable). The same REF in both roles loads once.",
    )
    serve_p.add_argument(
        "--retrieval-max-resident",
        type=int,
        default=2,
        help="How many retrieval models stay in memory; least-recently-used are unloaded",
    )
    serve_p.add_argument(
        "--retrieval-idle-timeout",
        type=float,
        default=0.0,
        help="Unload retrieval models after this many idle seconds (0 = never)",
    )
    serve_p.add_argument(
        "--retrieval-max-tokens",
        type=int,
        default=0,
        help="Truncate retrieval inputs to this many tokens (0 = per-model default)",
    )
    serve_p.add_argument(
        "--retrieval-trust-remote-code",
        action="store_true",
        help=(
            "Allow retrieval checkpoints that ship their own Python "
            "(jina-style model.py/rerank.py) to execute it; off by default"
        ),
    )
    serve_p.add_argument(
        "--no-stats-footer",
        action="store_false",
        dest="stats_footer",
        default=True,
        help="Do not append the visible MTPLX TPS footer to returned text.",
    )
    _add_fan_mode_args(
        serve_p,
        max_help="Compatibility alias for --fan-mode max for the server lifetime",
    )
    serve_p.add_argument(
        "--require-max-fans",
        action="store_true",
        help="Fail startup if --max cannot verify actual fan ramp before model load.",
    )
    serve_p.add_argument(
        "--enable-thermal-poll",
        action="store_true",
        help=(
            "Enable live fan telemetry polling for the dashboard/native app. "
            "Off by default because it shells out to ThermalForge."
        ),
    )
    serve_p.add_argument(
        "--app-launch-id",
        help="Opaque native-app launch id echoed by /health for daemon ownership checks.",
    )
    serve_p.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local browser chat after the server starts",
    )
    serve_p.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="Startup warmup generation length. Use 0 to disable.",
    )
    serve_p.add_argument(
        "--strict-warmup",
        action="store_true",
        help="Fail server startup if the warmup pass fails.",
    )
    serve_p.add_argument(
        "--strict-fast-path",
        action="store_true",
        help="Deprecated, no effect: MTPLX runs on stock PyPI MLX; no fork is required.",
    )
    serve_p.set_defaults(func=cmd_serve_public)

    preflight_p = sub.add_parser(
        "bench-preflight", help="Check benchmark contamination before speed runs"
    )
    preflight_p.add_argument("--project-root", default=".")
    preflight_p.add_argument("--top-limit", type=int, default=12)
    preflight_p.add_argument("--cpu-threshold", type=float, default=25.0)
    preflight_p.add_argument("--min-free-gib", type=float, default=25.0)
    preflight_p.add_argument("--strict", action="store_true")
    preflight_p.add_argument("--output")
    preflight_p.set_defaults(func=_cmd_bench_preflight)

    inspect_p = sub.add_parser("inspect-model", help="Inspect Qwen/MLX model artifacts")
    inspect_p.add_argument("model")
    inspect_p.add_argument("--require-mtp", action="store_true")
    inspect_p.add_argument(
        "--no-strict-exit-code",
        action="store_false",
        dest="strict_exit_code",
        help="Always exit 0 after printing the compatibility verdict.",
    )
    inspect_p.set_defaults(strict_exit_code=True)
    inspect_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    inspect_p.set_defaults(func=_cmd_inspect_model)

    bench_p = sub.add_parser("bench", help="Run benchmark harness")
    bench_p.add_argument(
        "bench_action",
        nargs="?",
        choices=[
            "run",
            "context",
            "tune",
            "aime",
            "prefill-ladder",
            "nightly",
            "suite",
            "compare",
            "serve",
            "reference",
            "reference-vllm",
        ],
        help="Public benchmark action. Omit for legacy benchmark flags.",
    )
    bench_p.add_argument("--backend", default="manifest")
    bench_p.add_argument(
        "--profile",
        choices=(*PROFILE_CHOICES, "native-mtp-60"),
        help=(
            "Runtime profile for product benchmark actions. Default follows the launch rule: "
            "Turbo for the quantized 27B/9B flagships across every suite; Sustained otherwise "
            "(context and long-generation suites stay Sustained for non-flagship models); "
            "native-mtp-60 is a legacy alias for performance-cold."
        ),
    )
    bench_p.add_argument(
        "--suite",
        choices=[
            "default",
            "long_code",
            "long-code",
            "long_code_uncapped",
            "long-code-uncapped",
            "calibration_coding",
            "calibration-coding",
            "flappy",
            "python_modules_long",
            "python-modules-long",
            "cold-long-code-192",
            "champion-bakeoff",
            "distribution-smoke",
            "multiturn-flappy",
        ],
    )
    bench_p.add_argument(
        "--strict",
        action="store_true",
        help="Run clean-preflight before profile benchmarks",
    )
    bench_p.add_argument(
        "--strict-cold",
        action="store_true",
        help="Enforce the cold 59 tok/s regression gate",
    )
    bench_p.add_argument(
        "--no-fanmax", action="store_true", help="Mark run as no-fan product candidate"
    )
    bench_p.add_argument(
        "--fanmax", action="store_true", help="Mark run as fan-controlled diagnostic"
    )
    bench_p.add_argument(
        "--max", action="store_true", dest="fanmax", help="Alias for --fanmax"
    )
    bench_p.add_argument(
        "--generation-mode",
        choices=["mtp", "ar"],
        default=None,
        help="Benchmark decode mode. AR here is target-only unless --stock-ar is also set.",
    )
    bench_p.add_argument(
        "--stock-ar",
        action="store_true",
        help="Diagnostic benchmark mode: AR with no MTP sidecar loaded.",
    )
    bench_p.add_argument("--unsafe-force-unverified", action="store_true")
    bench_p.add_argument(
        "--yes", action="store_true", help="Confirm unsafe non-interactive actions"
    )
    bench_p.add_argument("--dry-run", action="store_true")
    bench_p.add_argument(
        "--quick",
        action="store_true",
        help="Use the compact launch-regression task set for bench suite/nightly.",
    )
    bench_p.add_argument(
        "--harness",
        choices=["auto", "direct-http", "depth-sweep"],
        default="auto",
        help="Benchmark execution harness. auto uses the selected profile's safest harness.",
    )
    bench_p.add_argument("--run-id")
    bench_p.add_argument("--output-dir")
    bench_p.add_argument("--trace-interval-s", type=float, default=1.0)
    bench_p.add_argument("--exactness-attention-impl", default="mlx_vector_paged")
    bench_p.add_argument("--exactness-block-size", type=int, default=16)
    bench_p.add_argument("--exactness-num-blocks", type=int, default=1024)
    bench_p.add_argument("--exactness-no-partitioned", action="store_true")
    bench_p.add_argument("--exactness-partition-threshold", type=int, default=2048)
    bench_p.add_argument("--exactness-partition-size", type=int, default=512)
    bench_p.add_argument("--models", nargs="+")
    bench_p.add_argument("--record-champion", action="store_true")
    bench_p.add_argument("--champion", default=DEFAULT_HF_MODEL_ID)
    bench_p.add_argument(
        "--references", nargs="+", default=["stock_mlx_lm", "llama_cpp"]
    )
    bench_p.add_argument("--url", default="http://127.0.0.1:8000")
    bench_p.add_argument("--port", type=int, default=8041)
    bench_p.add_argument("--turns", type=int, default=5)
    bench_p.add_argument("--capture-dispatch", action="store_true")
    bench_p.add_argument("--ssh-host", default="mtplx-3090")
    bench_p.add_argument(
        "--remote-phase-dir", default="/home/youssof/ai/mtplx-phase1-v4-20260429-012151"
    )
    bench_p.add_argument("--remote-venv", default="/home/youssof/ai/vllm-venv")
    bench_p.add_argument("--remote-run-script", default="run_nsys_server_capture.sh")
    bench_p.add_argument("--remote-mode", choices=["no-mtp", "mtp5"], default="mtp5")
    bench_p.add_argument(
        "--remote-capture-kind", choices=["offline", "server"], default="offline"
    )
    bench_p.add_argument("--remote-port", type=int, default=8065)
    bench_p.add_argument("--remote-timeout-s", type=int, default=3600)
    bench_p.add_argument("--remote-output-dir")
    bench_p.add_argument("--cpu-threshold", type=float, default=25.0)
    bench_p.add_argument("--min-free-gib", type=float, default=25.0)
    bench_p.add_argument("--model", default=default_model)
    bench_p.add_argument("--cache-dir")
    bench_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    bench_p.add_argument("--output")
    bench_p.add_argument("--out", dest="output", help="Alias for --output")
    bench_p.add_argument(
        "--json",
        action="store_true",
        help="Accepted for friendly scripts; benchmark commands already print JSON",
    )
    bench_p.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed tuner diagnostics where supported",
    )
    bench_p.add_argument(
        "--no-save", action="store_true", help="Do not save tuner recommendations"
    )
    bench_p.add_argument(
        "--retune",
        action="store_true",
        help="Ignore saved tuner results where supported",
    )
    bench_p.add_argument(
        "--no-telemetry",
        action="store_true",
        help="Disable bench tune power telemetry for cleaner speed comparison",
    )
    bench_p.add_argument(
        "--before", help="Baseline envelope or nightly summary for bench compare"
    )
    bench_p.add_argument(
        "--after", help="Candidate envelope or nightly summary for bench compare"
    )
    bench_p.add_argument(
        "--strict-exactness",
        action="store_true",
        help="Require exactness gate pass in envelope compare mode",
    )
    bench_p.add_argument("--cold-regression-tolerance-pct", type=float, default=2.0)
    bench_p.add_argument("--nightly-exactness-contexts", default="64,2048,6144,10240")
    bench_p.add_argument("--temperature", type=float, default=0.6)
    bench_p.add_argument("--top-p", type=float, default=0.95)
    bench_p.add_argument("--top-k", type=int, default=20)
    bench_p.add_argument("--draft-temperature", type=float)
    bench_p.add_argument("--draft-top-p", type=float)
    bench_p.add_argument("--draft-top-k", type=int)
    bench_p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sampler seed. Defaults are harness-aware: 42 for long direct-HTTP runs, 0 for cold depth-sweep runs.",
    )
    bench_p.add_argument("--max-tokens", type=int, default=128)
    bench_p.add_argument(
        "--depths",
        help="Comma-separated MTP depths for bench tune/depth diagnostics; defaults to 1,2,3 for tune.",
    )
    bench_p.add_argument(
        "--contexts",
        help="Comma-separated prompt token contexts for bench prefill-ladder, e.g. 512,1k,32k",
    )
    bench_p.add_argument(
        "--full",
        action="store_true",
        help="Include 64k and 128k in bench prefill-ladder defaults.",
    )
    bench_p.add_argument(
        "--prompt-style",
        choices=("coding-agent", "legacy-repeat"),
        default="coding-agent",
        help=(
            "Prompt construction for bench prefill-ladder. coding-agent keeps a "
            "coherent final user request after the long filler; legacy-repeat "
            "preserves the old hard-truncated synthetic stream for diagnostics."
        ),
    )
    bench_p.add_argument(
        "--prompt-format",
        choices=("chat", "raw"),
        default="chat",
        help=(
            "Prompt envelope for bench prefill-ladder. chat matches the product "
            "API path; raw is diagnostic-only."
        ),
    )
    bench_p.add_argument(
        "--prompt-tail",
        help="Override the coherent final request used by bench prefill-ladder.",
    )
    bench_p.add_argument(
        "--prompt-tail-file",
        help="Read the coherent final request for bench prefill-ladder from a UTF-8 file.",
    )
    bench_p.add_argument(
        "--prefill-layout",
        choices=(
            "profile",
            "contiguous-then-repage",
            "contiguous-dense-decode",
            "paged",
        ),
        default="profile",
        help=(
            "Override MTPLX_SUSTAINED_PREFILL_LAYOUT for bench prefill-ladder. "
            "Use profile for the selected profile default."
        ),
    )
    bench_p.add_argument(
        "--paged-attn-impl",
        choices=(
            "mlx-vector-paged",
            "mlx_vector_paged",
            "fast-sdpa-gather",
            "fast_sdpa_gather",
            "exact-gather",
            "exact_gather",
            "sdpa-2pass-paged",
            "sdpa_2pass_paged",
            "vllm-metal",
            "vllm_metal",
            "paged",
        ),
        help="Diagnostic override for MTPLX_VLLM_METAL_PAGED_ATTN_IMPL after profile env is applied.",
    )
    bench_p.add_argument(
        "--mtp-history-policy",
        choices=(
            "auto",
            "committed",
            "full",
            "last-window",
            "last_window",
            "cycle",
            "none",
        ),
        help="Diagnostic override for MTPLX_MTP_HISTORY_POLICY after profile env is applied.",
    )
    bench_p.add_argument(
        "--mtp-history-window",
        type=int,
        help="Diagnostic override for MTPLX_MTP_HISTORY_LAST_WINDOW after profile env is applied.",
    )
    bench_p.add_argument(
        "--prefill-cache-cleanup",
        action="store_true",
        help=(
            "Diagnostic OMLX-style prefill mode: synchronize and clear MLX's "
            "cache after each prefill chunk."
        ),
    )
    bench_p.add_argument(
        "--no-prefill-cache-cleanup",
        action="store_true",
        help=(
            "Diagnostic only: disable MTPLX_PREFILL_CHUNK_CACHE_CLEANUP after "
            "profile env is applied."
        ),
    )
    bench_p.add_argument(
        "--prefill-cache-cleanup-every",
        help=(
            "Diagnostic override for MTPLX_PREFILL_CHUNK_CACHE_CLEANUP_EVERY "
            "after profile env is applied."
        ),
    )
    bench_p.add_argument(
        "--prefill-chunk-size",
        type=int,
        help=(
            "Diagnostic override for MTPLX_PREFILL_CHUNK_SIZE after profile env "
            "is applied."
        ),
    )
    bench_p.add_argument(
        "--clear-cache-every",
        type=int,
        help=(
            "Diagnostic override for MTPLX_CLEAR_CACHE_EVERY during generation "
            "after profile env is applied."
        ),
    )
    bench_p.add_argument(
        "--defer-verify-hidden-eval",
        action="store_true",
        help=(
            "Diagnostic override: force MTPLX_DEFER_VERIFY_HIDDEN_EVAL=1 after "
            "profile env is applied."
        ),
    )
    bench_p.add_argument(
        "--no-defer-verify-hidden-eval",
        action="store_true",
        help=(
            "Diagnostic override: disable MTPLX_DEFER_VERIFY_HIDDEN_EVAL after "
            "profile env is applied."
        ),
    )
    bench_p.add_argument(
        "--verify-hidden-mode",
        choices=(
            "default",
            "logits-first-committed-slice",
            "logits_first_committed_slice",
        ),
        help=("Diagnostic label for verify hidden handling in prefill-ladder JSON."),
    )
    bench_p.add_argument(
        "--no-batch-target-arrays",
        action="store_true",
        help="Diagnostic override: set MTPLX_BATCH_TARGET_ARRAYS=0 after profile env is applied.",
    )
    bench_p.add_argument(
        "--prefill-stock-cache-only",
        action="store_true",
        help=(
            "Unsafe diagnostic OMLX-style prefill mode: call the model in "
            "stock cache-only form for chunks that do not need hidden states. "
            "Requires MTPLX_ALLOW_UNSAFE_PREFILL_STOCK_CACHE_ONLY=1."
        ),
    )
    bench_p.add_argument("--limit", type=int)
    bench_p.add_argument("--disable-thinking", action="store_true")
    bench_p.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen thinking in bench prefill-ladder chat formatting.",
    )
    bench_p.add_argument(
        "--speculative-depth",
        "--depth",
        dest="speculative_depth",
        type=int,
        default=0,
        help="Speculative MTP depth for depth-sweep benchmark runs; 0 uses the model contract default.",
    )
    bench_p.add_argument(
        "--vary-seed-by-context",
        action="store_true",
        help=(
            "Diagnostic only: add the context row index to --seed. By default "
            "the ladder uses one seed so rows compare context length rather "
            "than different sampling trajectories."
        ),
    )
    bench_p.add_argument(
        "--no-inter-context-cache-cleanup",
        action="store_true",
        help=(
            "Diagnostic only: do not synchronize and clear MLX's reusable cache "
            "between prefill-ladder context rows."
        ),
    )
    bench_p.add_argument("--adaptive", action="store_true")
    bench_p.set_defaults(func=_cmd_bench)

    qa_p = sub.add_parser("qa", help="Run MTPLX correctness gates")
    qa_sub = qa_p.add_subparsers(dest="qa_action", required=True)
    qa_exact_p = qa_sub.add_parser(
        "exactness", help="Run full Phase 0H paged-verifier exactness"
    )
    qa_exact_p.add_argument("--model", default=default_model)
    qa_exact_p.add_argument("--contexts", default="64,2048,6144,10240")
    qa_exact_p.add_argument("--prompt-suite")
    qa_exact_p.add_argument("--exactness-attention-impl", default="mlx_vector_paged")
    qa_exact_p.add_argument("--exactness-block-size", type=int, default=16)
    qa_exact_p.add_argument("--exactness-num-blocks", type=int, default=1024)
    qa_exact_p.add_argument("--exactness-no-partitioned", action="store_true")
    qa_exact_p.add_argument("--exactness-partition-threshold", type=int, default=2048)
    qa_exact_p.add_argument("--exactness-partition-size", type=int, default=512)
    qa_exact_p.add_argument("--output")
    qa_exact_p.set_defaults(func=cmd_qa_public)
    qa_dist_p = qa_sub.add_parser(
        "distribution", help="Run distribution-level exactness smoke across suites"
    )
    qa_dist_p.add_argument("--model", default=default_model)
    qa_dist_p.add_argument("--reference-stack", default="stock_mlx_lm_ar")
    qa_dist_p.add_argument("--suite", default="distribution-smoke")
    qa_dist_p.add_argument("--contexts", default="2048")
    qa_dist_p.add_argument("--tolerance", default="kl=0.01,chi2_p=0.01")
    qa_dist_p.add_argument("--exactness-attention-impl", default="mlx_vector_paged")
    qa_dist_p.add_argument("--exactness-block-size", type=int, default=16)
    qa_dist_p.add_argument("--exactness-num-blocks", type=int, default=1024)
    qa_dist_p.add_argument("--exactness-no-partitioned", action="store_true")
    qa_dist_p.add_argument("--exactness-partition-threshold", type=int, default=2048)
    qa_dist_p.add_argument("--exactness-partition-size", type=int, default=512)
    qa_dist_p.add_argument("--output-dir")
    qa_dist_p.set_defaults(func=cmd_qa_public)

    profile_public_p = sub.add_parser(
        "profile", help="Profile dispatch, thermal, and compile behavior"
    )
    profile_sub = profile_public_p.add_subparsers(dest="profile_action", required=True)
    profile_dispatch_p = profile_sub.add_parser(
        "dispatch", help="Analyze or prepare dispatch-count profiling"
    )
    profile_dispatch_p.add_argument("--model", default=default_model)
    profile_dispatch_p.add_argument("--suite", default="flappy")
    profile_dispatch_p.add_argument("--max-tokens", type=int, default=2048)
    profile_dispatch_p.add_argument("--trace")
    profile_dispatch_p.add_argument("--output-dir")
    profile_dispatch_p.set_defaults(func=cmd_profile_public)
    profile_thermal_p = profile_sub.add_parser(
        "thermal", help="Run SMC Atlas / powermetrics thermal profile"
    )
    profile_thermal_p.add_argument("--model", default=default_model)
    profile_thermal_p.add_argument("--suite", default="flappy")
    profile_thermal_p.add_argument("--max-tokens", type=int, default=10000)
    profile_thermal_p.add_argument("--no-fanmax", action="store_true")
    profile_thermal_p.add_argument("--run-id")
    profile_thermal_p.add_argument("--output-dir")
    profile_thermal_p.add_argument("--dry-run", action="store_true")
    profile_thermal_p.set_defaults(func=cmd_profile_public)
    profile_compile_p = profile_sub.add_parser(
        "compile-audit", help="Audit mx.compile as a measured lever"
    )
    profile_compile_p.add_argument("--model", default=default_model)
    profile_compile_p.add_argument(
        "--prompts", default="mtplx/benchmarks/prompts/long_code.jsonl"
    )
    profile_compile_p.add_argument("--prompt-index", type=int, default=0)
    profile_compile_p.add_argument("--prefill-chunks", default="128,256,512,1024")
    profile_compile_p.add_argument("--depths", default="3,4")
    profile_compile_p.add_argument("--max-tokens", type=int, default=64)
    profile_compile_p.add_argument("--repeats", type=int, default=2)
    profile_compile_p.add_argument("--warmup", type=int, default=1)
    profile_compile_p.add_argument("--verify-core", default="linear-gdn-from-conv-tape")
    profile_compile_p.add_argument(
        "--exactness-attention-impl", default="mlx_vector_paged"
    )
    profile_compile_p.add_argument("--exactness-block-size", type=int, default=16)
    profile_compile_p.add_argument("--exactness-num-blocks", type=int, default=1024)
    profile_compile_p.add_argument("--exactness-no-partitioned", action="store_true")
    profile_compile_p.add_argument(
        "--exactness-partition-threshold", type=int, default=2048
    )
    profile_compile_p.add_argument("--exactness-partition-size", type=int, default=512)
    profile_compile_p.add_argument("--skip-prefill", action="store_true")
    profile_compile_p.add_argument("--skip-verify", action="store_true")
    profile_compile_p.add_argument("--skip-exactness-smoke", action="store_true")
    profile_compile_p.add_argument("--disable-thinking", action="store_true")
    profile_compile_p.add_argument("--output")
    profile_compile_p.add_argument("--output-dir")
    profile_compile_p.add_argument("--dry-run", action="store_true")
    profile_compile_p.set_defaults(func=cmd_profile_public)
    profile_eval_p = profile_sub.add_parser(
        "eval-attribution",
        help="Attribute verify-cycle eval debt across outputs/cache/state groups",
    )
    profile_eval_p.add_argument("--model", default=default_model)
    profile_eval_p.add_argument("--prefix-tokens", type=int, default=2048)
    profile_eval_p.add_argument("--verify-tokens", type=int, default=4)
    profile_eval_p.add_argument("--seed", type=int, default=42)
    profile_eval_p.add_argument("--depth", type=int, default=3)
    profile_eval_p.add_argument("--temperature", type=float, default=0.6)
    profile_eval_p.add_argument("--top-p", type=float, default=0.95)
    profile_eval_p.add_argument("--top-k", type=int, default=20)
    profile_eval_p.add_argument("--verify-strategy", default="capture_commit")
    profile_eval_p.add_argument("--verify-core", default="linear-gdn-from-conv-tape")
    profile_eval_p.add_argument("--mtp-history-policy", default="committed")
    profile_eval_p.add_argument(
        "--orders",
        default="outputs,attn,recurrent;attn,recurrent,outputs",
        help="Semicolon-separated eval orders, each as comma-separated group names.",
    )
    profile_eval_p.add_argument("--prompt")
    profile_eval_p.add_argument("--no-serving-fast-defaults", action="store_true")
    profile_eval_p.add_argument("--output")
    profile_eval_p.add_argument("--output-dir")
    profile_eval_p.add_argument("--dry-run", action="store_true")
    profile_eval_p.set_defaults(func=cmd_profile_public)

    thermal_p = sub.add_parser("thermal", help="Thermal diagnostic helpers")
    thermal_sub = thermal_p.add_subparsers(dest="thermal_action", required=True)
    fanmax_p = thermal_sub.add_parser(
        "fanmax-run", help="Run a diagnostic with both fans pinned to max"
    )
    fanmax_p.add_argument("--model", default=default_model)
    fanmax_p.add_argument("--suite", default="flappy")
    fanmax_p.add_argument("--max-tokens", type=int, default=10000)
    fanmax_p.add_argument("--run-id")
    fanmax_p.add_argument("--output-dir")
    fanmax_p.add_argument("--dry-run", action="store_true")
    fanmax_p.set_defaults(func=cmd_thermal_public)

    max_p = sub.add_parser(
        "max", help="Opt-in fan profile control via ThermalForge or TG Pro"
    )
    max_group = max_p.add_mutually_exclusive_group(required=True)
    max_group.add_argument(
        "--on",
        dest="max_action",
        action="store_const",
        const="performance",
        help="Set the Performance fan profile",
    )
    max_group.add_argument(
        "--max",
        dest="max_action",
        action="store_const",
        const="max",
        help="Set the Max fan profile",
    )
    max_group.add_argument(
        "--off",
        dest="max_action",
        action="store_const",
        const="silent",
        help="Restore the Silent fan profile",
    )
    max_group.add_argument(
        "--status",
        dest="max_action",
        action="store_const",
        const="status",
        help="Show thermal-control status",
    )
    max_group.add_argument(
        "--install",
        dest="max_action",
        action="store_const",
        const="install",
        help="Auto-install MTPLX's private ThermalForge source build",
    )
    max_group.add_argument(
        "--grant-sudo",
        dest="max_action",
        action="store_const",
        const="grant_sudo",
        help="Install the passwordless sudoers rule for thermalforge (run once if --install was done before this feature existed)",
    )
    max_group.add_argument(
        "--revoke-sudo",
        dest="max_action",
        action="store_const",
        const="revoke_sudo",
        help="Remove the mtplx-thermalforge sudoers rule",
    )
    max_p.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    max_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the command without changing fan state",
    )
    max_p.add_argument(
        "--no-daemon",
        action="store_true",
        help="Skip the one-time `sudo thermalforge install` daemon setup",
    )
    max_p.set_defaults(func=cmd_max_public)

    debug_p = sub.add_parser("debug", help="Create redacted support/debug artifacts")
    debug_sub = debug_p.add_subparsers(dest="debug_action", required=True)
    debug_bundle_p = debug_sub.add_parser(
        "bundle", help="Create a redacted debug bundle"
    )
    debug_bundle_p.add_argument("--run-id")
    debug_bundle_p.add_argument("--output-dir")
    debug_bundle_p.add_argument("--project-root", default=".")
    debug_bundle_p.add_argument("--model-cache")
    debug_bundle_p.add_argument("--url", default="http://127.0.0.1:8000")
    debug_bundle_p.set_defaults(func=cmd_debug_public)
    debug_hotpath_p = debug_sub.add_parser(
        "hotpath", help="Audit verifier hot-path kernel and sync boundaries"
    )
    debug_hotpath_p.add_argument("--output")
    debug_hotpath_p.set_defaults(func=cmd_debug_public)

    metrics_p = sub.add_parser(
        "metrics", help="Inspect a running MTPLX server's metrics"
    )
    metrics_sub = metrics_p.add_subparsers(dest="metrics_action", required=True)
    metrics_watch_p = metrics_sub.add_parser(
        "watch", help="Poll /metrics and print a compact live view"
    )
    metrics_watch_p.add_argument("--url", default="http://127.0.0.1:8000")
    metrics_watch_p.add_argument("--interval", type=float, default=1.0)
    metrics_watch_p.add_argument(
        "--count",
        type=int,
        default=1,
        help="Poll count. Use 0 to watch until interrupted.",
    )
    metrics_watch_p.add_argument("--timeout", type=float, default=5.0)
    metrics_watch_p.add_argument("--json", action="store_true")
    metrics_watch_p.set_defaults(func=cmd_metrics_public)

    dashboard_p = sub.add_parser(
        "dashboard",
        help="Open the live MTPLX dashboard in the browser (requires a running server)",
    )
    dashboard_p.add_argument("--host", default="127.0.0.1", help="MTPLX server host")
    dashboard_p.add_argument("--port", type=int, default=8000, help="MTPLX server port")
    dashboard_p.add_argument(
        "--timeout", type=float, default=2.5, help="Health-probe timeout in seconds"
    )
    dashboard_p.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the dashboard URL but do not open the browser",
    )
    dashboard_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON"
    )
    dashboard_p.set_defaults(func=cmd_dashboard_public)

    integrate_p = sub.add_parser("integrate", help="Print client integration settings")
    integrate_sub = integrate_p.add_subparsers(dest="integration", required=True)
    for integration_name in ("openwebui", "claude-code", "opencode", "swival"):
        integration_p = integrate_sub.add_parser(integration_name)
        integration_p.add_argument("--host", default="127.0.0.1")
        integration_p.add_argument("--port", type=int, default=8000)
        integration_p.add_argument("--model-id", default=DEFAULT_PUBLIC_MODEL_ID)
        integration_p.add_argument("--api-key-env", default="MTPLX_API_KEY")
        integration_p.add_argument(
            "--docker",
            action="store_true",
            help="Include Dockerized Open WebUI command",
        )
        integration_p.add_argument("--webui-port", type=int, default=3000)
        integration_p.add_argument("--single-user", action="store_true")
        integration_p.add_argument("--api-key", default="mtplx-local")
        integration_p.add_argument("--smoke", action="store_true")
        integration_p.add_argument("--timeout", type=float, default=5.0)
        integration_p.add_argument("--context-window", type=int, default=262144)
        integration_p.add_argument("--json", action="store_true")
        integration_p.set_defaults(func=cmd_integrate_public)

    model_p = sub.add_parser("model", help="Model publishing and compatibility helpers")
    model_sub = model_p.add_subparsers(dest="model_action", required=True)
    architectures_p = model_sub.add_parser(
        "architectures", help="List MTPLX architecture support status"
    )
    architectures_p.add_argument("--json", action="store_true")
    architectures_p.set_defaults(func=cmd_model_public)
    qa_architectures_p = model_sub.add_parser(
        "qa-architectures",
        help="Run synthetic MTP architecture compatibility QA without loading real checkpoints",
    )
    qa_architectures_p.add_argument("--json", action="store_true")
    qa_architectures_p.add_argument("--output")
    qa_architectures_p.add_argument(
        "--runtime-import-smoke",
        action="store_true",
        help="Import native backend facades and report their health metadata",
    )
    qa_architectures_p.set_defaults(func=cmd_model_public)
    publish_check_p = model_sub.add_parser(
        "publish-check", help="Validate HF staging readiness without upload"
    )
    publish_check_p.add_argument(
        "--staging-dir",
        default="hf-staging/Qwen3.6-27B-MTPLX-Optimized-Speed",
    )
    publish_check_p.add_argument("--repo-id")
    publish_check_p.set_defaults(func=cmd_model_public)

    config_p = sub.add_parser("config", help="Show or edit MTPLX user config")
    config_sub = config_p.add_subparsers(dest="config_action", required=True)
    config_show_p = config_sub.add_parser("show")
    config_show_p.add_argument("--config")
    config_show_p.add_argument("--json", action="store_true")
    config_show_p.set_defaults(func=cmd_config_public)
    config_set_p = config_sub.add_parser("set")
    config_set_p.add_argument("key")
    config_set_p.add_argument("value")
    config_set_p.add_argument("--config")
    config_set_p.add_argument("--dry-run", action="store_true")
    config_set_p.set_defaults(func=cmd_config_public)

    smoke_p = sub.add_parser(
        "runtime-smoke", help="Load model, inject MTP, and run one AR/MTP forward"
    )
    smoke_p.add_argument("--model", default=default_model)
    smoke_p.add_argument(
        "--prompt",
        default="def add(a: int, b: int) -> int:\\n    return",
    )
    smoke_p.set_defaults(func=_cmd_runtime_smoke)

    probe_p = sub.add_parser(
        "probe-contract", help="Probe MTP hidden-state and concat-order contracts"
    )
    probe_p.add_argument("--model", default=default_model)
    probe_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    probe_p.add_argument("--max-prompt-tokens", type=int, default=256)
    probe_p.add_argument("--raw-prompts", action="store_true")
    probe_p.add_argument("--disable-thinking", action="store_true")
    probe_p.add_argument("--output")
    probe_p.set_defaults(func=_cmd_probe_contract)

    ratio_p = sub.add_parser(
        "verify-ratio", help="Measure cached forward(k+1) / forward(1)"
    )
    ratio_p.add_argument("--model", default=default_model)
    ratio_p.add_argument(
        "--prompt",
        default="def add(a: int, b: int) -> int:\\n    return",
    )
    ratio_p.add_argument("--max-k", type=int, default=8)
    ratio_p.add_argument("--repeats", type=int, default=3)
    ratio_p.add_argument("--output")
    ratio_p.set_defaults(func=_cmd_verify_ratio)

    profile_p = sub.add_parser(
        "verify-profile", help="Synchronously profile target verify sections"
    )
    profile_p.add_argument("--model", default=default_model)
    profile_p.add_argument(
        "--prompts", default="mtplx/benchmarks/prompts/default.jsonl"
    )
    profile_p.add_argument("--lengths", default="1,2,3,6")
    profile_p.add_argument("--repeats", type=int, default=2)
    profile_p.add_argument("--warmup", type=int, default=1)
    profile_p.add_argument("--prompt-index", type=int, default=0)
    profile_p.add_argument("--disable-thinking", action="store_true")
    profile_p.add_argument("--output")
    profile_p.set_defaults(func=_cmd_verify_profile)

    qmm_probe_p = sub.add_parser(
        "verify-qmm-probe",
        help="Rank isolated QuantizedLinear small-M costs for VerifyCore qmm targets",
    )
    qmm_probe_p.add_argument("--model", default=default_model)
    qmm_probe_p.add_argument("--m-values", default="1,3,4,5,16")
    qmm_probe_p.add_argument("--repeats", type=int, default=5)
    qmm_probe_p.add_argument("--warmup", type=int, default=2)
    qmm_probe_p.add_argument("--include", default="mlp,gdn,attn,lm_head,mtp")
    qmm_probe_p.add_argument(
        "--dtype",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
        default="bf16",
    )
    qmm_probe_p.add_argument("--max-groups", type=int)
    qmm_probe_p.add_argument("--seed", type=int, default=0)
    qmm_probe_p.add_argument("--no-mtp", action="store_true")
    qmm_probe_p.add_argument(
        "--dense-mirror",
        action="store_true",
        help="Also time a BF16/FP16 dequantized dense-weight mirror for each sampled QuantizedLinear.",
    )
    qmm_probe_p.add_argument("--output")
    qmm_probe_p.set_defaults(func=_cmd_verify_qmm_probe)

    qmv_probe_p = sub.add_parser(
        "multi-qmv-probe",
        help="Probe the experimental M=3 multi-vector qmv VerifyCore primitive",
    )
    qmv_probe_p.add_argument("--model", default=default_model)
    qmv_probe_p.add_argument("--include", default="mlp")
    qmv_probe_p.add_argument("--repeats", type=int, default=10)
    qmv_probe_p.add_argument("--warmup", type=int, default=3)
    qmv_probe_p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    qmv_probe_p.add_argument("--seed", type=int, default=0)
    qmv_probe_p.add_argument("--no-mtp", action="store_true")
    qmv_probe_p.add_argument("--output")
    qmv_probe_p.set_defaults(func=_cmd_multi_qmv_probe)

    batch_eq_p = sub.add_parser(
        "batch-equivalence",
        help="Compare batched target forward against sequential one-token forward",
    )
    batch_eq_p.add_argument("--model", default=default_model)
    batch_eq_p.add_argument(
        "--prompts", default="mtplx/benchmarks/prompts/default.jsonl"
    )
    batch_eq_p.add_argument("--suffix-len", type=int, default=2)
    batch_eq_p.add_argument("--limit", type=int)
    batch_eq_p.add_argument("--expand-to", type=int)
    batch_eq_p.add_argument("--disable-thinking", action="store_true")
    batch_eq_p.add_argument("--tolerance", type=float, default=1e-3)
    batch_eq_p.add_argument("--output")
    batch_eq_p.set_defaults(func=_cmd_batch_equivalence)

    capture_eq_p = sub.add_parser(
        "capture-commit-equivalence",
        help="Verify captured GDN prefix commit against sequential AR state",
    )
    capture_eq_p.add_argument("--model", default=default_model)
    capture_eq_p.add_argument(
        "--prompts", default="mtplx/benchmarks/prompts/default.jsonl"
    )
    capture_eq_p.add_argument("--suffix-len", type=int, default=6)
    capture_eq_p.add_argument("--min-keep-tokens", type=int, default=1)
    capture_eq_p.add_argument("--limit", type=int)
    capture_eq_p.add_argument("--expand-to", type=int)
    capture_eq_p.add_argument("--disable-thinking", action="store_true")
    capture_eq_p.add_argument("--tolerance", type=float, default=1e-3)
    capture_eq_p.add_argument(
        "--verify-backend", choices=["direct", "graphbank"], default="direct"
    )
    capture_eq_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
        help="Capture/VerifyCore backend for GDN state capture.",
    )
    capture_eq_p.add_argument("--output")
    capture_eq_p.set_defaults(func=_cmd_capture_commit_equivalence)

    mtp1_p = sub.add_parser(
        "mtp1-greedy-gate", help="Compare MTP-1 greedy output against AR"
    )
    mtp1_p.add_argument("--model", default=default_model)
    mtp1_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    mtp1_p.add_argument("--max-tokens", type=int, default=32)
    mtp1_p.add_argument("--seed", type=int, default=0)
    mtp1_p.add_argument("--limit", type=int)
    mtp1_p.add_argument("--expand-to", type=int)
    mtp1_p.add_argument("--disable-thinking", action="store_true")
    mtp1_p.add_argument(
        "--verify-strategy",
        choices=[
            "batched",
            "sequential",
            "capture",
            "capture_commit",
            "graphbank",
            "graphbank_capture_commit",
        ],
        default="batched",
    )
    mtp1_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
        help="Capture/VerifyCore backend for capture-commit strategies.",
    )
    mtp1_p.add_argument("--draft-margin-threshold", type=float)
    mtp1_p.add_argument("--mtp-quant-bits", type=int)
    mtp1_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    mtp1_p.add_argument("--mtp-quant-mode", default="affine")
    mtp1_p.add_argument("--output")
    mtp1_p.set_defaults(func=_cmd_mtp1_greedy_gate)

    sampler_p = sub.add_parser(
        "mtp1-sampler-smoke", help="Run MTP-1 at non-greedy sampler settings"
    )
    sampler_p.add_argument("--model", default=default_model)
    sampler_p.add_argument(
        "--prompts", default="mtplx/benchmarks/prompts/default.jsonl"
    )
    sampler_p.add_argument("--temperature", type=float, default=0.6)
    sampler_p.add_argument("--top-p", type=float, default=0.95)
    sampler_p.add_argument("--top-k", type=int, default=20)
    sampler_p.add_argument("--draft-temperature", type=float)
    sampler_p.add_argument("--draft-top-p", type=float)
    sampler_p.add_argument("--draft-top-k", type=int)
    sampler_p.add_argument("--max-tokens", type=int, default=96)
    sampler_p.add_argument("--seed", type=int, default=0)
    sampler_p.add_argument("--limit", type=int)
    sampler_p.add_argument("--disable-thinking", action="store_true")
    sampler_p.add_argument("--compare-ar", action="store_true")
    sampler_p.add_argument(
        "--verify-strategy",
        choices=[
            "batched",
            "sequential",
            "capture",
            "capture_commit",
            "graphbank",
            "graphbank_capture_commit",
        ],
        default="batched",
    )
    sampler_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
        help="Capture/VerifyCore backend for capture-commit strategies.",
    )
    sampler_p.add_argument("--draft-margin-threshold", type=float)
    sampler_p.add_argument("--mtp-quant-bits", type=int)
    sampler_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    sampler_p.add_argument("--mtp-quant-mode", default="affine")
    sampler_p.add_argument("--output")
    sampler_p.set_defaults(func=_cmd_mtp1_sampler_smoke)

    depth_p = sub.add_parser("mtp-depth-sweep", help="Run fixed-depth native MTP sweep")
    depth_p.add_argument("--model", default=default_model)
    depth_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    depth_p.add_argument("--depths", default="1,2,3")
    depth_p.add_argument("--temperature", type=float, default=0.6)
    depth_p.add_argument("--top-p", type=float, default=0.95)
    depth_p.add_argument("--top-k", type=int, default=20)
    depth_p.add_argument("--draft-temperature", type=float)
    depth_p.add_argument("--draft-top-p", type=float)
    depth_p.add_argument("--draft-top-k", type=int)
    depth_p.add_argument("--draft-lm-head-bits", type=int)
    depth_p.add_argument("--draft-lm-head-group-size", type=int, default=64)
    depth_p.add_argument("--draft-lm-head-mode", default="affine")
    depth_p.add_argument("--max-tokens", type=int, default=96)
    depth_p.add_argument("--seed", type=int, default=0)
    depth_p.add_argument("--limit", type=int)
    depth_p.add_argument("--disable-thinking", action="store_true")
    depth_p.add_argument("--compare-ar", action="store_true")
    depth_p.add_argument("--mtp-hidden-variant", default="post_norm")
    depth_p.add_argument(
        "--mtp-cache-policy", choices=["persistent", "fresh"], default="persistent"
    )
    depth_p.add_argument(
        "--mtp-history-policy", choices=["cycle", "committed"], default="cycle"
    )
    depth_p.add_argument("--draft-margin-threshold", type=float)
    depth_p.add_argument(
        "--min-speculative-depth",
        type=int,
        default=1,
        help=(
            "Number of draft depths to always attempt before margin gating can "
            "skip a candidate. Default 1 keeps D1 live and gates D2+."
        ),
    )
    depth_p.add_argument(
        "--verify-strategy",
        choices=[
            "batched",
            "capture_commit",
            "graphbank",
            "graphbank_capture_commit",
            "trim_commit",
        ],
        default="batched",
    )
    depth_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
        help="Capture/VerifyCore backend for capture-commit strategies.",
    )
    depth_p.add_argument(
        "--draft-core",
        choices=["stock", "device-d2", "device"],
        default="stock",
        help=(
            "Experimental DraftCore backend. device-d2 compiles the greedy D2 "
            "native-MTP argmax chain for the exact draft-temperature 0 path."
        ),
    )
    depth_p.add_argument("--mtp-quant-bits", type=int)
    depth_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    depth_p.add_argument("--mtp-quant-mode", default="affine")
    depth_p.add_argument("--mtp-adapter", type=Path)
    depth_p.add_argument(
        "--merge-mtp-adapter",
        action="store_true",
        help=(
            "Bake an installed MTP LoRA adapter into its base MTP linears in memory "
            "before the sweep. Diagnostic for adapter overhead; depth-gated adapters "
            "are reported but not exactly representable as one static weight."
        ),
    )
    depth_p.add_argument("--mtp-corrector", type=Path)
    depth_p.add_argument("--mtp-corrector-blend", type=float)
    depth_p.add_argument(
        "--online-hidden-corrector-alpha",
        type=float,
        default=0.0,
        help=(
            "Experimental session-local EWMA residual applied to MTP hidden states "
            "before the next draft depth. Default 0 disables it."
        ),
    )
    depth_p.add_argument("--online-hidden-corrector-decay", type=float, default=0.8)
    depth_p.add_argument("--online-hidden-corrector-warmup", type=int, default=1)
    depth_p.add_argument("--online-hidden-corrector-max-feed-depth", type=int)
    depth_p.add_argument(
        "--online-hidden-corrector-key",
        choices=["global", "token"],
        default="global",
    )
    depth_p.add_argument(
        "--online-correction-cache",
        action="store_true",
        help=(
            "Experimental exact proposal override cache keyed by the local "
            "speculative prefix. Stores target top tokens after rejections."
        ),
    )
    depth_p.add_argument("--online-correction-cache-min-depth", type=int, default=1)
    depth_p.add_argument(
        "--online-correction-cache-key",
        choices=["local_prefix", "source_token", "primary_source"],
        default="local_prefix",
        help=(
            "Experimental correction-cache key policy. local_prefix preserves "
            "the original behavior; source_token and primary_source trade "
            "more hits for broader proposal reuse."
        ),
    )
    depth_p.add_argument(
        "--prompt-correction-cache",
        action="store_true",
        help=(
            "Experimental exact proposal cache seeded from prompt-local "
            "n-gram continuations. Uses the same one-hot q acceptance path."
        ),
    )
    depth_p.add_argument("--prompt-correction-cache-min-depth", type=int, default=2)
    depth_p.add_argument(
        "--adapter-ensemble-q",
        action="store_true",
        help=(
            "Experimental exact sparse-q proposal over base-vs-adapter MTP "
            "argmax tokens. Requires --mtp-adapter and greedy draft sampling."
        ),
    )
    depth_p.add_argument("--adapter-ensemble-epsilon", type=float, default=0.5)
    depth_p.add_argument("--adapter-ensemble-min-depth", type=int, default=2)
    depth_p.add_argument(
        "--mtp-topk-reranker-calib",
        type=Path,
        help=(
            "Experimental exact one-hot proposal selector fit from a hidden "
            "calibration shard. Diagnostic only."
        ),
    )
    depth_p.add_argument("--mtp-topk-reranker-depths", default="4")
    depth_p.add_argument("--mtp-topk-reranker-topk", type=int, default=32)
    depth_p.add_argument("--mtp-topk-reranker-q-weight", type=float, default=0.5)
    depth_p.add_argument("--mtp-topk-reranker-token-weight", type=float, default=1.0)
    depth_p.add_argument("--mtp-topk-reranker-rank-weight", type=float, default=0.0)
    depth_p.add_argument(
        "--mtp-topk-reranker-all-rows",
        action="store_true",
        help="Fit top-k proposal priors on all calibration rows, not just prefix-active rows.",
    )
    depth_p.add_argument("--output")
    depth_p.set_defaults(func=_cmd_mtp_depth_sweep)

    chain_p = sub.add_parser(
        "mtp-chain-probe",
        help="Probe recursive MTP agreement by history/cache contract",
    )
    chain_p.add_argument("--model", default=default_model)
    chain_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    chain_p.add_argument("--depth", type=int, default=5)
    chain_p.add_argument("--limit", type=int)
    chain_p.add_argument("--max-prompt-tokens", type=int, default=256)
    chain_p.add_argument("--windows", type=int, default=1)
    chain_p.add_argument("--stride", type=int, default=1)
    chain_p.add_argument("--top-ranks", default="1,2,4,8")
    chain_p.add_argument("--mtp-quant-bits", type=int)
    chain_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    chain_p.add_argument("--mtp-quant-mode", default="affine")
    chain_p.add_argument("--raw-prompts", action="store_true")
    chain_p.add_argument("--disable-thinking", action="store_true")
    chain_p.add_argument("--base-hidden-variants", default="post_norm")
    chain_p.add_argument("--mtp-hidden-variants", default="post_norm,pre_norm,fc")
    chain_p.add_argument("--cache-policies", default="fresh,persistent")
    chain_p.add_argument("--concat-orders", default="embedding_hidden")
    chain_p.add_argument(
        "--mtp-position-modes",
        default="local",
        help=(
            "MTP RoPE position contract to probe. 'local' preserves current "
            "MLX cache-offset behavior; 'absolute' applies prompt/window "
            "absolute positions before MTP cache update."
        ),
    )
    chain_p.add_argument(
        "--history-modes",
        default="recursive,target_forced,target_token_recursive_hidden",
    )
    chain_p.add_argument("--anchors", default="prompt_boundary,after_one_target")
    chain_p.add_argument("--output")
    chain_p.set_defaults(func=_cmd_mtp_chain_probe)

    tree_probe_p = sub.add_parser(
        "mtp-tree-probe", help="Probe native-MTP tree coverage without target verify"
    )
    tree_probe_p.add_argument("--model", default=default_model)
    tree_probe_p.add_argument(
        "--prompts", default="mtplx/benchmarks/prompts/default.jsonl"
    )
    tree_probe_p.add_argument("--depth", type=int, default=5)
    tree_probe_p.add_argument("--budgets", default="1,2,4,8,16")
    tree_probe_p.add_argument("--branch-factor", type=int, default=4)
    tree_probe_p.add_argument("--limit", type=int)
    tree_probe_p.add_argument("--windows", type=int, default=32)
    tree_probe_p.add_argument("--stride", type=int, default=1)
    tree_probe_p.add_argument("--max-prompt-tokens", type=int, default=256)
    tree_probe_p.add_argument("--raw-prompts", action="store_true")
    tree_probe_p.add_argument("--disable-thinking", action="store_true")
    tree_probe_p.add_argument("--mtp-quant-bits", type=int)
    tree_probe_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    tree_probe_p.add_argument("--mtp-quant-mode", default="affine")
    tree_probe_p.add_argument(
        "--base-hidden-variant", choices=["post_norm", "pre_norm"], default="post_norm"
    )
    tree_probe_p.add_argument("--mtp-hidden-variant", default="pre_norm")
    tree_probe_p.add_argument(
        "--mtp-cache-policy",
        choices=["fresh", "persistent_path"],
        default="fresh",
        help=(
            "MTP cache contract for branch expansion. 'persistent_path' "
            "replays each branch path into one MTP cache before expanding it."
        ),
    )
    tree_probe_p.add_argument(
        "--anchor",
        choices=["prompt_boundary", "after_one_target"],
        default="prompt_boundary",
    )
    tree_probe_p.add_argument("--output")
    tree_probe_p.set_defaults(func=_cmd_mtp_tree_probe)

    grid_p = sub.add_parser(
        "mtp-depth-grid", help="Run a sequential fixed-depth policy grid"
    )
    grid_p.add_argument("--model", default=default_model)
    grid_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    grid_p.add_argument("--depth", type=int, default=5)
    grid_p.add_argument("--thresholds", default="0.5,0.75,1.0,1.25,1.5,2.0")
    grid_p.add_argument("--min-depths", default="0,1,2,3")
    grid_p.add_argument("--temperature", type=float, default=0.6)
    grid_p.add_argument("--top-p", type=float, default=0.95)
    grid_p.add_argument("--top-k", type=int, default=20)
    grid_p.add_argument("--draft-temperature", type=float, default=0.0)
    grid_p.add_argument("--draft-top-p", type=float)
    grid_p.add_argument("--draft-top-k", type=int)
    grid_p.add_argument("--max-tokens", type=int, default=96)
    grid_p.add_argument("--seed", type=int, default=0)
    grid_p.add_argument("--limit", type=int)
    grid_p.add_argument("--disable-thinking", action="store_true")
    grid_p.add_argument("--compare-ar", action="store_true")
    grid_p.add_argument("--mtp-hidden-variant", default="pre_norm")
    grid_p.add_argument(
        "--mtp-cache-policy", choices=["persistent", "fresh"], default="fresh"
    )
    grid_p.add_argument(
        "--mtp-history-policy", choices=["cycle", "committed"], default="cycle"
    )
    grid_p.add_argument(
        "--verify-strategy",
        choices=[
            "batched",
            "capture_commit",
            "graphbank",
            "graphbank_capture_commit",
            "trim_commit",
        ],
        default="batched",
    )
    grid_p.add_argument("--mtp-corrector", type=Path)
    grid_p.add_argument("--mtp-corrector-blend", type=float)
    grid_p.add_argument("--store-events", action="store_true")
    grid_p.add_argument("--output")
    grid_p.set_defaults(func=_cmd_mtp_depth_grid)

    adaptive_p = sub.add_parser("mtp-adaptive", help="Run adaptive-depth native MTP")
    adaptive_p.add_argument("--model", default=default_model)
    adaptive_p.add_argument(
        "--prompts", default="mtplx/benchmarks/prompts/default.jsonl"
    )
    adaptive_p.add_argument("--max-depth", type=int, default=5)
    adaptive_p.add_argument("--min-depth", type=int, default=1)
    adaptive_p.add_argument("--start-depth", type=int, default=1)
    adaptive_p.add_argument("--increase-after", type=int, default=4)
    adaptive_p.add_argument("--decrease-after", type=int, default=1)
    adaptive_p.add_argument(
        "--policy", choices=["streak", "expected_value"], default="streak"
    )
    adaptive_p.add_argument("--ev-base-depth", type=int, default=2)
    adaptive_p.add_argument(
        "--ev-accept-priors", type=_comma_floats, default=(0.92, 0.64, 0.32)
    )
    adaptive_p.add_argument("--ev-draft-cost-s", type=float, default=0.0048)
    adaptive_p.add_argument("--ev-extra-verify-cost-s", type=float, default=0.0060)
    adaptive_p.add_argument("--ev-baseline-tok-s", type=float, default=40.0)
    adaptive_p.add_argument("--ev-safety-margin", type=float, default=0.10)
    adaptive_p.add_argument("--ev-margin-center", type=float, default=1.0)
    adaptive_p.add_argument("--ev-margin-scale", type=float, default=2.0)
    adaptive_p.add_argument("--ev-confidence-weight", type=float, default=0.35)
    adaptive_p.add_argument(
        "--ev-min-extra-accept-probability", type=float, default=0.18
    )
    adaptive_p.add_argument("--temperature", type=float, default=0.6)
    adaptive_p.add_argument("--top-p", type=float, default=0.95)
    adaptive_p.add_argument("--top-k", type=int, default=20)
    adaptive_p.add_argument("--draft-temperature", type=float)
    adaptive_p.add_argument("--draft-top-p", type=float)
    adaptive_p.add_argument("--draft-top-k", type=int)
    adaptive_p.add_argument("--max-tokens", type=int, default=96)
    adaptive_p.add_argument("--seed", type=int, default=0)
    adaptive_p.add_argument("--limit", type=int)
    adaptive_p.add_argument("--disable-thinking", action="store_true")
    adaptive_p.add_argument("--compare-ar", action="store_true")
    adaptive_p.add_argument("--mtp-hidden-variant", default="post_norm")
    adaptive_p.add_argument(
        "--mtp-cache-policy", choices=["persistent", "fresh"], default="persistent"
    )
    adaptive_p.add_argument(
        "--mtp-history-policy", choices=["cycle", "committed"], default="cycle"
    )
    adaptive_p.add_argument(
        "--verify-strategy",
        choices=[
            "batched",
            "capture_commit",
            "graphbank",
            "graphbank_capture_commit",
            "trim_commit",
        ],
        default="batched",
    )
    adaptive_p.add_argument(
        "--verify-core",
        choices=VERIFY_CORE_CHOICES,
        default="stock",
    )
    adaptive_p.add_argument("--mtp-quant-bits", type=int)
    adaptive_p.add_argument("--mtp-quant-group-size", type=int, default=64)
    adaptive_p.add_argument("--mtp-quant-mode", default="affine")
    adaptive_p.add_argument("--output")
    adaptive_p.set_defaults(func=_cmd_mtp_adaptive)

    dflash_p = sub.add_parser(
        "dflash-mlx-baseline", help="Run official DFlash MLX baseline"
    )
    dflash_p.add_argument("--model", default=default_model)
    dflash_p.add_argument("--draft-model", default="z-lab/Qwen3.6-27B-DFlash")
    dflash_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    dflash_p.add_argument("--dflash-source", default="REFERENCES:TOOLS/dflash")
    dflash_p.add_argument("--temperature", type=float, default=0.6)
    dflash_p.add_argument("--top-p", type=float, default=0.95)
    dflash_p.add_argument("--top-k", type=int, default=20)
    dflash_p.add_argument("--max-tokens", type=int, default=96)
    dflash_p.add_argument("--block-size", type=int)
    dflash_p.add_argument("--seed", type=int, default=0)
    dflash_p.add_argument("--limit", type=int)
    dflash_p.add_argument("--disable-thinking", action="store_true")
    dflash_p.add_argument("--draft-sliding-window-size", type=int)
    dflash_p.add_argument("--output")
    dflash_p.set_defaults(func=_cmd_dflash_mlx_baseline)

    ddtree_p = sub.add_parser("ddtree-mlx-baseline", help="Run DDTree MLX baseline")
    ddtree_p.add_argument("--model", default=default_model)
    ddtree_p.add_argument("--draft-model", default="z-lab/Qwen3.6-27B-DFlash")
    ddtree_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    ddtree_p.add_argument("--ddtree-source", default="REFERENCES:TOOLS/ddtree-mlx")
    ddtree_p.add_argument("--temperature", type=float, default=0.6)
    ddtree_p.add_argument("--top-p", type=float, default=0.95)
    ddtree_p.add_argument("--top-k", type=int, default=20)
    ddtree_p.add_argument("--max-tokens", type=int, default=96)
    ddtree_p.add_argument("--tree-budget", type=int, default=4)
    ddtree_p.add_argument("--limit", type=int)
    ddtree_p.add_argument("--disable-thinking", action="store_true")
    ddtree_p.add_argument("--output")
    ddtree_p.set_defaults(func=_cmd_ddtree_mlx_baseline)

    truth_p = sub.add_parser(
        "truth-report", help="Run the Phase 0 evidence-grade MTPLX truth harness"
    )
    truth_p.add_argument("--model", default=default_model)
    truth_p.add_argument("--prompts", default="mtplx/benchmarks/prompts/default.jsonl")
    truth_p.add_argument(
        "--modes",
        default=",".join(DEFAULT_TRUTH_MODES),
        help="Comma-separated truth modes to run",
    )
    truth_p.add_argument("--temperature", type=float, default=0.6)
    truth_p.add_argument("--top-p", type=float, default=0.95)
    truth_p.add_argument("--top-k", type=int, default=20)
    truth_p.add_argument("--draft-temperature", type=float, default=0.0)
    truth_p.add_argument("--draft-top-p", type=float)
    truth_p.add_argument("--draft-top-k", type=int, default=1)
    truth_p.add_argument("--max-tokens", type=int, default=96)
    truth_p.add_argument("--seed", type=int, default=0)
    truth_p.add_argument("--limit", type=int, default=1)
    truth_p.add_argument("--disable-thinking", action="store_true")
    truth_p.add_argument("--mtp-hidden-variant", default="pre_norm")
    truth_p.add_argument(
        "--mtp-cache-policy", choices=["persistent", "fresh"], default="persistent"
    )
    truth_p.add_argument(
        "--mtp-history-policy", choices=["cycle", "committed"], default="cycle"
    )
    truth_p.add_argument("--c3-corrector", type=Path, default=DEFAULT_C3_CORRECTOR)
    truth_p.add_argument("--c3-blend", type=float, default=0.15)
    truth_p.add_argument("--project-root", default=".")
    truth_p.add_argument("--min-free-gib", type=float, default=120.0)
    truth_p.add_argument("--cpu-threshold", type=float, default=25.0)
    truth_p.add_argument("--output-dir", default="outputs/reports/truth")
    truth_p.add_argument("--output-json")
    truth_p.add_argument("--output-md")
    truth_p.add_argument("--strict-preflight", action="store_true")
    truth_p.add_argument("--fail-fast", action="store_true")
    truth_p.set_defaults(func=_cmd_truth_report)

    session_p = sub.add_parser(
        "session-bank", help="Benchmark exact warm-prefix SessionBank prefill reuse"
    )
    session_p.add_argument("--model", default=default_model)
    session_p.add_argument(
        "--prompts", default="mtplx/benchmarks/prompts/default.jsonl"
    )
    session_p.add_argument("--prompt-index", type=int, default=0)
    session_p.add_argument(
        "--suffix-text",
        default="\n\n# Follow-up request:\nRefactor this into a cleaner implementation.\n",
    )
    session_p.add_argument("--max-prompt-tokens", type=int, default=512)
    session_p.add_argument("--raw-prompts", action="store_true")
    session_p.add_argument("--disable-thinking", action="store_true")
    session_p.add_argument("--max-entries", type=int, default=4)
    session_p.add_argument("--tolerance", type=float, default=1e-3)
    session_p.add_argument(
        "--restore-mode", choices=["clone", "reference"], default="clone"
    )
    session_p.add_argument("--output")
    session_p.set_defaults(func=_cmd_session_bank)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--no-color" in raw_args:
        os.environ["MTPLX_NO_COLOR"] = "1"
        raw_args = [arg for arg in raw_args if arg != "--no-color"]
    if not raw_args or raw_args[0] in ("-h", "--help"):
        print(_format_public_help())
        return 0
    parser = build_parser()
    if raw_args[0] == "help":
        return _print_help_topic(raw_args[1] if len(raw_args) > 1 else None, parser)
    if raw_args[0] == "advanced" and len(raw_args) == 1:
        print(_format_advanced_help())
        return 0
    command_names = _parser_command_names(parser)
    if raw_args[0] not in command_names and not raw_args[0].startswith("-"):
        return _print_unknown_command(raw_args[0])
    args = parser.parse_args(raw_args)
    args._cli_flags = canonicalize_flag_tokens(
        _explicit_cli_flags(raw_args), parser, args
    )
    from .config import apply_user_config

    apply_user_config(args)
    return int(args.func(args))


def main_tune(argv: list[str] | None = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    return main(["tune", *raw_args])


def _explicit_cli_flags(raw_args: list[str]) -> set[str]:
    """Return the set of long/short flag names actually typed on the CLI.

    This is the only reliable signal for "did the user type ``--model``?"
    because parser defaults and ``apply_user_config`` both write onto the
    parsed Namespace, masking the user's actual intent. Used by the quickstart
    onboarding to know when to fall through to the interactive flow.
    """

    flags: set[str] = set()
    for token in raw_args:
        if not token.startswith("-") or token == "-" or token == "--":
            continue
        head = token.split("=", 1)[0]
        if head.startswith("--"):
            flags.add(head[2:])
        else:
            flags.add(head[1:])
    return flags


if __name__ == "__main__":
    raise SystemExit(main())
