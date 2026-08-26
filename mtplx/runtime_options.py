from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


KV_QUANT_MODES = ("off", "q8", "q6", "q4")

#: The one boolean vocabulary for MTPLX env flags.
#:
#: Every reader of a boolean ``MTPLX_*`` var should go through
#: :func:`env_bool` so a spelling means the same thing everywhere. Values
#: outside these sets raise rather than being silently read as "off" by one
#: reader and "on" by another — the failure mode catalogued in
#: docs/AUDIT_2026-07-18.md, where ``=enabled`` disabled a feature in the
#: server while enabling it in the generation loop.
ENV_TRUE_VALUES = frozenset({"1", "true", "yes", "on", "enable", "enabled"})
ENV_FALSE_VALUES = frozenset({"0", "false", "no", "off", "disable", "disabled"})


def env_bool(
    name: str,
    *,
    default: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Parse a boolean ``MTPLX_*`` env var, or raise on an unknown spelling.

    Unset (and set-but-empty) yields ``default``. Anything that is neither
    a recognized true nor a recognized false value is a configuration
    error: guessing is what let one variable mean three things.
    """

    source = os.environ if env is None else env
    raw = source.get(name)
    if raw is None:
        return bool(default)
    token = str(raw).strip().lower()
    if not token:
        return bool(default)
    if token in ENV_TRUE_VALUES:
        return True
    if token in ENV_FALSE_VALUES:
        return False
    accepted = ", ".join(sorted(ENV_TRUE_VALUES | ENV_FALSE_VALUES))
    raise ValueError(
        f"{name}={raw!r} is not a boolean; expected one of: {accepted}"
    )


@dataclass(frozen=True)
class ResolvedAPIKey:
    value: str | None
    source: str

    @property
    def required(self) -> bool:
        return bool(self.value)


def parser_option_names(parser: object, namespace: object = None) -> set[str]:
    """Every option name reachable in the parse context, without dashes.

    Walks the root parser plus whichever subparsers the parse actually
    descended into (using ``namespace`` to pick the branch), which is the
    same scope argparse resolves abbreviations against.
    """

    names: set[str] = set()
    seen: set[int] = set()
    pending = [parser]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        for action in getattr(current, "_actions", ()):
            for option in getattr(action, "option_strings", ()) or ():
                names.add(str(option).lstrip("-"))
            choices = getattr(action, "choices", None)
            dest = getattr(action, "dest", None)
            if not isinstance(choices, dict) or not dest:
                continue
            # A subparsers action: follow only the branch that was taken.
            picked = getattr(namespace, str(dest), None) if namespace else None
            if picked is not None and picked in choices:
                pending.append(choices[picked])
            elif namespace is None:
                pending.extend(choices.values())
    return names


def canonicalize_flag_tokens(
    tokens: set[str],
    parser: object,
    namespace: object = None,
) -> set[str]:
    """Expand argparse abbreviations to the flag names they resolved to.

    ``--temp 0.9`` sets ``args.temperature`` but was recorded as ``temp``,
    so every ``"temperature" in cli_flags`` check read it as *not typed*
    and the config file happily overwrote the user's value. Resolving here
    (rather than setting ``allow_abbrev=False``) keeps abbreviations
    working while making the explicit-flag signal true.

    The raw token is kept alongside the expansion, so checks written
    against either spelling keep working. Ambiguous prefixes expand to
    nothing — argparse would have rejected the command anyway.
    """

    known = parser_option_names(parser, namespace)
    resolved = set(tokens)
    for token in tokens:
        if token in known:
            continue
        matches = {name for name in known if name.startswith(token)}
        if len(matches) == 1:
            resolved |= matches
    return resolved


def block_prefix_restore_enabled() -> bool:
    """The single parse of ``MTPLX_SESSION_BLOCK_PREFIX_RESTORE``.

    Default ON. Every reader — the decode loop, the engine session, the
    session bank's cold tier, and the server's settings view — goes through
    this, so one spelling cannot mean ON in one and OFF in another. Unset
    used to mean OFF in the cold tier and ON everywhere else, which
    silently disabled cold-tier block-prefix restore for library embedders
    (the CLI path masks it by force-setting "1"); and the server's
    allowlist-only read reported "off" for spellings the runtime honours as
    on, e.g. ``=enabled``.

    Lives here rather than in :mod:`mtplx.session_bank` because this module
    has no heavy imports: the server reads the setting on paths where the
    mlx-backed runtime may be unavailable.
    """

    return env_bool("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", default=True)


def normalize_paged_kv_quantization(value: object | None, *, allow_none: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        return "off"
    raw = str(value).strip().lower().replace("-", "_")
    if raw in ("", "none", "false", "0", "disabled", "disable"):
        return "off"
    if raw in ("off", "q8", "q6", "q4"):
        return raw
    if raw in ("8", "8bit", "int8", "uint8", "q8_0"):
        return "q8"
    # q6 is MTPLX's own 4-codes-per-3-bytes packing; the aliases stay limited
    # to bit-width spellings so no caller reads them as an external Q6 format.
    if raw in ("6", "6bit", "int6", "uint6"):
        return "q6"
    if raw in ("4", "4bit", "int4", "uint4", "q4_0"):
        return "q4"
    choices = ", ".join(KV_QUANT_MODES)
    raise ValueError(f"unsupported paged KV quantization mode {value!r}; expected one of: {choices}")


def paged_kv_quantization_env(mode: object | None) -> dict[str, str]:
    canonical = normalize_paged_kv_quantization(mode)
    return {
        "MTPLX_VLLM_METAL_PAGED_KV_QUANT": canonical,
        "MTPLX_PAGED_KV_QUANT": canonical,
    }


def apply_paged_kv_quantization_env(mode: object | None, env: dict[str, str] | None = None) -> str:
    canonical = normalize_paged_kv_quantization(mode)
    target = os.environ if env is None else env
    target.update(paged_kv_quantization_env(canonical))
    return canonical


def generate_api_key_file(api_key_file: str | os.PathLike[str]) -> str:
    """Create ``api_key_file`` holding a fresh random key and return the key.

    Server entrypoints call this when the user passed ``--api-key-file`` for a
    path that does not exist yet, so the recovery command our own non-localhost
    refusal prints is runnable as-is instead of dying on FileNotFoundError.
    Read-only consumers of key files (doctor, connect) must NOT call this — a
    missing file is a real error there. The file is created 0600.
    """
    import secrets

    path = Path(api_key_file).expanduser()
    key = "mtplx-" + secrets.token_hex(24)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(key + "\n")
    return key


def resolve_api_key(
    *,
    explicit_api_key: str | None = None,
    api_key_file: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> ResolvedAPIKey:
    explicit = _clean_secret(explicit_api_key)
    if explicit:
        return ResolvedAPIKey(explicit, "flag")

    if api_key_file:
        path = Path(api_key_file).expanduser()
        secret = _clean_secret(path.read_text(encoding="utf-8"))
        if not secret:
            raise ValueError(f"API key file is empty: {path}")
        return ResolvedAPIKey(secret, "file")

    source_env = os.environ if env is None else env
    api_key = _clean_secret(source_env.get("MTPLX_API_KEY"))
    if api_key:
        return ResolvedAPIKey(api_key, "env:MTPLX_API_KEY")

    legacy = _clean_secret(source_env.get("MTPLX_AUTH"))
    if legacy:
        return ResolvedAPIKey(legacy, "env:MTPLX_AUTH")

    return ResolvedAPIKey(None, "none")


def _clean_secret(value: object | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
