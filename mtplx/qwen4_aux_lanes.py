"""Two decode lanes stacked on top of the Qwen3.8 Flash-Next stack.

These are two lanes from an external optimization pass that reproduced a decode gain at the
canonical 16,384/1,024 cell and are exact by construction (timing-only,
byte-identical output):

* ``ple_cached_aux`` -- the cached async PLE auxiliary (mtplx/ple_cached_aux.py),
  which needs the ``mtplx_native_ple_cpu_rows`` extension; and
* ``qsa_pooled_rowsel`` -- the fixed-M4 pooled-key rowsel install
  (mtplx/qsa_pooled_rowsel.py), pure stock MLX.

They are armed by the server for a served Flash-Next pack exactly the way the
retained stack is: a ``setdefault`` behind the served-config predicate, which
every explicit operator export beats. This module is the single place that
names each lane, its env key and its default, and it is deliberately kept OUT
of ``mtplx.full_stack_env``'s measured 44-key stack so the PR-391 battery
counts, the committed flag files and the full-stack self-check do not move. It
shares only the operator's off switch: it registers its lane names with
``full_stack_env`` so ``--disable-optimization <lane>``, ``MTPLX_FABLE_DISABLE``
and ``all`` accept them.

The module is inert on import beyond registering its lane names; it imports no
MLX and loads no extension.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import full_stack_env


#: Boolean vocabulary, matching the other lenient ``os.environ.get`` readers.
TRUE_TOKENS = full_stack_env.TRUE_TOKENS

#: lane -> env key. One key per lane; each is its own opt-out (``=0``).
LANE_KEYS: dict[str, str] = {
    "ple_cached_aux": "MTPLX_FABLE_PLE_CACHED_AUX",
    "qsa_pooled_rowsel": "MTPLX_FABLE_QSA_POOLED_ROWSEL",
}

#: Lane names, in order.
LANES: tuple[str, ...] = tuple(LANE_KEYS)

#: env key -> lane.
KEY_LANE: dict[str, str] = {key: lane for lane, key in LANE_KEYS.items()}

#: The value the server arms each key to for a served Flash-Next pack.
STACK_VALUE = "1"

#: The whole stacked set: key -> armed value.
STACKED_ENV: dict[str, str] = {key: STACK_VALUE for key in LANE_KEYS.values()}

#: What the defaults actually armed in THIS process, key -> value. Read by
#: :func:`defaults_report` and :func:`value_source`. Empty in a process that
#: did not arm them (a driver, a test, another model family).
DEFAULTS_APPLIED: dict[str, str] = {}

SOURCE_DEFAULT = "default"
SOURCE_OPERATOR = "operator"


# Share the operator off switch with the measured stack: register the lane
# names so full_stack_env.parse_disable_lanes accepts them and ``all`` expands
# to them. Idempotent, so re-import is safe.
full_stack_env.register_extra_lanes({lane: (key,) for lane, key in LANE_KEYS.items()})


def lane_enabled(lane: str, environ: Mapping[str, str] | None = None) -> bool:
    """Is the lane armed in ``environ``? Unset is off (the stock path)."""

    import os

    if lane not in LANE_KEYS:
        raise KeyError(f"unknown auxiliary lane {lane!r}; expected one of {LANES}")
    source = os.environ if environ is None else environ
    value = str(source.get(LANE_KEYS[lane]) or "").strip().lower()
    return value in TRUE_TOKENS


def ple_cached_aux_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """MTPLX_FABLE_PLE_CACHED_AUX read the way its install site reads it."""

    return lane_enabled("ple_cached_aux", environ)


def qsa_pooled_rowsel_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """MTPLX_FABLE_QSA_POOLED_ROWSEL read the way its install site reads it."""

    return lane_enabled("qsa_pooled_rowsel", environ)


def resolve_disabled(
    environ: Mapping[str, str] | None = None,
    *,
    extra: Iterable[str] = (),
) -> frozenset[str]:
    """The stacked lanes the operator turned off, from the shared switches.

    Reads ``MTPLX_FABLE_DISABLE`` and the ``--disable-optimization`` values
    (passed in as ``extra``), the same two switches the measured stack reads.
    ``all`` turns off every stacked lane. A measured-stack lane name here is
    not this registry's concern and is ignored -- ``full_stack_env`` validates
    the whole token list and raises on a genuine typo, so this resolver does
    not need to.
    """

    import os

    source = os.environ if environ is None else environ
    tokens: list[str] = []
    raw = source.get(full_stack_env.DISABLE_ENV)
    if raw:
        tokens.extend(token.strip().lower() for token in str(raw).split(","))
    for item in extra:
        tokens.extend(token.strip().lower() for token in str(item).split(","))
    tokens = [token for token in tokens if token]
    if full_stack_env.DISABLE_ALL in tokens:
        return frozenset(LANES)
    return frozenset(token for token in tokens if token in LANE_KEYS)


def default_env(
    environ: Mapping[str, str] | None = None,
    *,
    disabled_lanes: Iterable[str] = (),
) -> dict[str, str]:
    """The keys the defaults would arm in ``environ``, and their values.

    Excludes a key the operator has already exported (any non-empty value,
    ``0`` included -- that IS the off switch) and every key of a disabled lane
    named through ``--disable-optimization`` / ``MTPLX_FABLE_DISABLE``.
    ``disabled_lanes`` holds stacked lane names, as :func:`resolve_disabled`
    returns them.
    """

    import os

    source = os.environ if environ is None else environ
    off = {LANE_KEYS[lane] for lane in disabled_lanes if lane in LANE_KEYS}
    return {
        key: value
        for key, value in STACKED_ENV.items()
        if key not in off and not str(source.get(key) or "").strip()
    }


def record_defaults_applied(applied: Mapping[str, str]) -> None:
    """Remember what the defaults armed, for the report and value_source."""

    DEFAULTS_APPLIED.update({str(k): str(v) for k, v in applied.items()})


def value_source(name: str, environ: Mapping[str, str] | None = None) -> str:
    """Who put the current value of a stacked key there: default/operator/``""``."""

    import os

    if name not in STACKED_ENV:
        return ""
    source = os.environ if environ is None else environ
    present = str(source.get(name) or "").strip()
    if not present:
        return ""
    if DEFAULTS_APPLIED.get(name) == present:
        return SOURCE_DEFAULT
    if not DEFAULTS_APPLIED:
        return ""
    return SOURCE_OPERATOR


def defaults_report(
    environ: Mapping[str, str] | None = None,
    *,
    disabled_lanes: Iterable[str] = (),
    model_gate: str = "",
) -> dict[str, Any]:
    """What the stacked defaults did, for ``GET /health`` and the startup line."""

    import os

    source = os.environ if environ is None else environ
    off_here = {lane for lane in disabled_lanes if lane in LANE_KEYS}
    armed = sorted(key for key in DEFAULTS_APPLIED if key in STACKED_ENV)
    operator_off: list[dict[str, str]] = []
    operator_pinned: list[dict[str, str]] = []
    for key, wanted in STACKED_ENV.items():
        if key in DEFAULTS_APPLIED:
            continue
        present = str(source.get(key) or "").strip()
        if not present:
            continue
        row = {"key": key, "lane": KEY_LANE.get(key, ""), "value": present}
        (operator_pinned if present == wanted else operator_off).append(row)
    return {
        "model_gate": model_gate,
        "lanes": list(LANES),
        "armed_by_default": armed,
        "operator_off": operator_off,
        "operator_pinned": operator_pinned,
        "disabled_lanes": sorted(off_here),
    }


__all__ = [
    "DEFAULTS_APPLIED",
    "KEY_LANE",
    "LANES",
    "LANE_KEYS",
    "STACKED_ENV",
    "default_env",
    "defaults_report",
    "lane_enabled",
    "ple_cached_aux_enabled",
    "qsa_pooled_rowsel_enabled",
    "record_defaults_applied",
    "value_source",
]
