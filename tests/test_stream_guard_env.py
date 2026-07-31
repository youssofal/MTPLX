"""The hidden-tool stream guard ceilings honor their env overrides.

The ceilings gate how much buffered tool-call text may stream before the
runaway backstop cancels generation (f8440e7). A typo in the env names would
silently fall back to the chat-UX defaults and re-break whole-file tool calls,
so the names and defaults are pinned here via a fresh interpreter (the values
are read once at import).
"""

import os
import subprocess
import sys

_SNIPPET = (
    "from mtplx.server.openai import "
    "STREAM_HIDDEN_TOOL_GUARD_TOKENS, STREAM_HIDDEN_TOOL_GUARD_S; "
    "print(STREAM_HIDDEN_TOOL_GUARD_TOKENS, STREAM_HIDDEN_TOOL_GUARD_S)"
)


def _read_ceilings(extra_env: dict[str, str]) -> tuple[int, float]:
    env = dict(os.environ)
    env.pop("MTPLX_STREAM_HIDDEN_TOOL_GUARD_TOKENS", None)
    env.pop("MTPLX_STREAM_HIDDEN_TOOL_GUARD_S", None)
    env.update(extra_env)
    out = subprocess.run(
        [sys.executable, "-c", _SNIPPET],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout.split()
    return int(out[0]), float(out[1])


def test_stream_guard_ceilings_default_to_chat_ux_values():
    tokens, seconds = _read_ceilings({})
    assert tokens == 2048
    assert seconds == 30.0


def test_stream_guard_ceilings_honor_env_overrides():
    tokens, seconds = _read_ceilings(
        {
            "MTPLX_STREAM_HIDDEN_TOOL_GUARD_TOKENS": "16384",
            "MTPLX_STREAM_HIDDEN_TOOL_GUARD_S": "600",
        }
    )
    assert tokens == 16384
    assert seconds == 600.0
