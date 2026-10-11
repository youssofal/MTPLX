"""How this server process was started, for the dashboard's Settings tab.

The Settings tab used to print a restart command from a template
(``mtplx serve --model <id> --profile <name> --port 8000``). It dropped every
environment variable, the real port and flags such as ``--no-auth``, so a
copied command started a server without the operator's settings. ``/health``
only shows what the profile sets (``profile_env``, ``fast_path_env``), not
what the operator passed.

This module records, once at startup, the command line and the environment
restricted to ``MTPLX_*``, ``MLX_*`` and ``PYTHONPATH``. Each variable carries
its source:

- ``launch``: present in the environment the process was started with;
- ``profile``: set by the profile (or the model contract's runtime overrides);
- ``both``: started with a value and also set by the profile; the effective
  value is whichever won (operator-overridable keys keep the launch value);
- ``flag``: set by MTPLX itself from a command-line flag (``mtplx serve``
  translating ``--paged-kv-quantization``, or the server stamping
  ``--memory-budget``).

``mtplx serve`` execs the server as ``python -m mtplx.server.openai`` with an
expanded argument list and a few variables derived from its own flags. It
hands its own invocation to the child in ``MTPLX_SERVE_LAUNCH``, so the
restart command is the one the operator typed and the derived variables are
marked ``flag`` instead of ``launch``.

Values whose name holds the word KEY, TOKEN, SECRET or PASSWORD are shown as
``•••`` (a name ending in FILE, PATH or DIR names a location, not a secret).
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

SERVE_LAUNCH_ENV = "MTPLX_SERVE_LAUNCH"
REDACTED = "•••"

_ENV_PREFIXES = ("MTPLX_", "MLX_")
_ENV_NAMES = frozenset({"PYTHONPATH"})
_SECRET_WORDS = frozenset({"KEY", "APIKEY", "TOKEN", "SECRET", "PASSWORD"})
_LOCATION_WORDS = frozenset({"FILE", "PATH", "DIR"})
_SERVER_MODULE = "mtplx.server.openai"


def is_launch_env_name(name: str) -> bool:
    """True for the variables the Settings tab lists."""

    if name == SERVE_LAUNCH_ENV:
        return False
    return name in _ENV_NAMES or name.startswith(_ENV_PREFIXES)


def is_secret_name(name: str) -> bool:
    """True when ``name`` (an env var or a ``--flag``) holds a secret.

    Matched per word, so ``HF_TOKEN`` and ``--api-key`` are secret while
    ``MTPLX_WARMUP_TOKENS`` is not.
    """

    words = [word for word in re.split(r"[_\-\s]+", name.upper()) if word]
    if not words or words[-1] in _LOCATION_WORDS:
        return False
    return any(word in _SECRET_WORDS for word in words)


def redact_value(name: str, value: str) -> str:
    return REDACTED if is_secret_name(name) else value


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Hide the value of every secret flag (``--api-key X``, ``--api-key=X``)."""

    out: list[str] = []
    hide_next = False
    for token in argv:
        if hide_next:
            hide_next = False
            if not token.startswith("-"):
                out.append(REDACTED)
                continue
        if token.startswith("--"):
            flag, sep, _value = token.partition("=")
            if is_secret_name(flag):
                if sep:
                    out.append(f"{flag}={REDACTED}")
                    continue
                hide_next = True
        out.append(token)
    return out


def launch_env(environ: Mapping[str, str]) -> dict[str, str]:
    return {
        name: str(value) for name, value in environ.items() if is_launch_env_name(name)
    }


def serve_launch_handoff(
    *,
    argv: Sequence[str],
    parent_env: Mapping[str, str],
    child_env: Mapping[str, str],
) -> str:
    """The ``MTPLX_SERVE_LAUNCH`` value ``mtplx serve`` gives its server child.

    ``argv`` is the operator's ``mtplx serve`` command; ``env_keys`` lists the
    variables the wrapper added or changed from its own flags.
    """

    env_keys = sorted(
        name
        for name, value in child_env.items()
        if is_launch_env_name(name) and parent_env.get(name) != value
    )
    return json.dumps({"argv": list(argv), "env_keys": env_keys})


@dataclass(frozen=True)
class LaunchCapture:
    """The process start as the operator gave it, before any profile ran."""

    argv: tuple[str, ...]
    launcher: str
    env: Mapping[str, str]
    flag_env_keys: frozenset[str]


def capture_launch(
    environ: Mapping[str, str],
    server_argv: Sequence[str],
    *,
    executable: str,
) -> LaunchCapture:
    """Snapshot argv and the launch environment.

    Call before ``apply_profile_env`` writes ``os.environ``. A malformed
    ``MTPLX_SERVE_LAUNCH`` falls back to the server's own command line.
    """

    env = launch_env(environ)
    argv: tuple[str, ...] = (executable, "-m", _SERVER_MODULE, *server_argv)
    launcher = "server"
    flag_env_keys: frozenset[str] = frozenset()
    raw = environ.get(SERVE_LAUNCH_ENV)
    if raw:
        try:
            handoff = json.loads(raw)
            serve_argv = [str(token) for token in handoff["argv"]]
            keys = frozenset(str(key) for key in handoff.get("env_keys") or ())
        except (ValueError, TypeError, KeyError):
            serve_argv = []
            keys = frozenset()
        if serve_argv:
            argv = tuple(serve_argv)
            launcher = "mtplx serve"
            flag_env_keys = keys
            env = {name: value for name, value in env.items() if name not in keys}
    return LaunchCapture(
        argv=argv,
        launcher=launcher,
        env=env,
        flag_env_keys=flag_env_keys,
    )


def _env_source(
    name: str, capture: LaunchCapture, profile_env: Mapping[str, str]
) -> str:
    in_launch = name in capture.env
    in_profile = name in profile_env
    if in_launch and in_profile:
        return "both"
    if in_launch:
        return "launch"
    if in_profile:
        return "profile"
    return "flag"


def build_launch_info(
    capture: LaunchCapture,
    *,
    environ: Mapping[str, str],
    profile_env: Mapping[str, str],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    """The read-only payload of ``GET /v1/mtplx/launch``.

    ``environ`` is the environment after startup finished (profile applied,
    flags stamped); ``profile_env`` is what the profile and the model
    contract asked for.
    """

    final_env = launch_env(environ)
    names = (
        set(capture.env)
        | {name for name in profile_env if is_launch_env_name(name)}
        | set(final_env)
    )
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        value = final_env.get(name)
        if value is None:
            value = capture.env.get(name, profile_env.get(name))
        if value is None:
            continue
        source = _env_source(name, capture, profile_env)
        row: dict[str, Any] = {
            "name": name,
            "value": redact_value(name, value),
            "source": source,
        }
        if source == "both":
            row["launch_value"] = redact_value(name, capture.env[name])
            row["profile_value"] = redact_value(name, str(profile_env[name]))
        rows.append(row)
    return {
        "ok": True,
        "launcher": capture.launcher,
        "fixed": dict(fixed),
        "argv": redact_argv(capture.argv),
        "env": rows,
        "restart": {
            "env": [
                {"name": name, "value": redact_value(name, capture.env[name])}
                for name in sorted(capture.env)
            ],
            "argv": redact_argv(capture.argv),
        },
    }


def serve_invocation_argv(argv: Sequence[str], *, executable: str) -> list[str]:
    """``sys.argv`` of ``mtplx serve`` as a runnable command.

    ``python -m mtplx`` leaves ``argv[0]`` pointing at ``__main__.py``, which
    is not runnable on its own.
    """

    tokens = [str(token) for token in argv]
    if tokens and os.path.basename(tokens[0]) == "__main__.py":
        return [executable, "-m", "mtplx", *tokens[1:]]
    return tokens
