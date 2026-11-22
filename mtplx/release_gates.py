"""What this release offers a user to pick, in one place.

``V3_AGENT_CLIENTS``: Claude Code, Codex and DeepSeek Harness as harnesses MTPLX starts and connects arrive in
MTPLX 3 (the founder, 22 November 2026): ``mtplx start claude-code|codex|deepseek-harness``, ``mtplx connect
codex|deepseek-harness``, ``mtplx integrate codex``, the client copies ``mtplx clients`` lists, and the app's Start
button. Off, a user finds them nowhere to pick, and a direct call says in one line that they arrive in MTPLX 3.
Their code, the launch lanes and the server's Anthropic and Responses endpoints stay as they are, and
``mtplx connect claude-code``, which 2.12.3 shipped, is not behind it. The app's twin is
``LaunchTarget.v3AgentClients``: turn both on together.
"""

from __future__ import annotations

V3_AGENT_CLIENTS = False

_V3_CLIENT_NAMES = {
    "claude-code": "Claude Code",
    "claude": "Claude Code",
    "claudecode": "Claude Code",
    "codex": "Codex",
    "codex-cli": "Codex",
    "codex-desktop": "Codex",
    "deepseek-harness": "DeepSeek Harness",
    "deepseek": "DeepSeek Harness",
    "dsh": "DeepSeek Harness",
}


def offered(name: str | None, *, connect: bool = False) -> bool:
    """Whether this release offers the client ``name``: a start target or its alias, a connect or integrate
    action, a client copy. ``connect``: Claude Code's ``mtplx connect`` and ``mtplx integrate`` (2.12.3's) are
    offered either way."""

    if V3_AGENT_CLIENTS or not name:
        return True
    label = _V3_CLIENT_NAMES.get(str(name).strip().lower())
    if label is None:
        return True
    return connect and label == "Claude Code"


def arrives_in_v3(name: str) -> str:
    """The one line a direct call of a client this release does not offer answers with."""

    label = _V3_CLIENT_NAMES.get(str(name).strip().lower(), str(name))
    return (
        f"{label} arrives in MTPLX 3. This release opens Pi, OpenCode, Hermes, the browser chat and "
        "the terminal chat: try `mtplx start pi`."
    )


def v3_text(when_on: str, when_off: str = "") -> str:
    """Help text that names the MTPLX 3 clients only where they are offered."""

    return when_on if V3_AGENT_CLIENTS else when_off
