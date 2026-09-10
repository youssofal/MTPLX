"""Two exact decode lanes stacked on the Qwen3.8 Flash-Next fixed-M4 stack.

These are the two aux lanes from PR #391 / PR #475, rebased onto upstream main:

* ``ple_cached_aux`` -- the cached async PLE auxiliary (``mtplx/ple_cached_aux.py``),
  which needs the ``mtplx_native_ple_cpu_rows`` extension; and
* ``qsa_pooled_rowsel`` -- the fixed-M4 pooled-key rowsel install
  (``mtplx/qsa_pooled_rowsel.py``), pure stock MLX.

Both are exact by construction (timing-only, byte-identical output). The server
auto-arms them for a served fixed-M4 Flash-Next pack the same way it arms the
other PR #391 ports: a ``setdefault`` behind the fixed-M4 predicate in
``mtplx/server/openai.py``, keyed off upstream's ``MTPLX_QWEN4_*`` / ``MTPLX_QSA_*``
namespace and honoured by the ``_QWEN4_LANE_KEYS`` ``pop`` kill-switch loop, so
an explicit operator export (``KEY=0``) always wins.

PR #391 spelled these lanes ``MTPLX_FABLE_PLE_CACHED_AUX`` /
``MTPLX_FABLE_QSA_POOLED_ROWSEL`` and armed them through ``mtplx.full_stack_env``.
Upstream main has no ``full_stack_env``; the primary keys are the ``MTPLX_QWEN4_*``
/ ``MTPLX_QSA_*`` names below and the old ``MTPLX_FABLE_*`` names are kept as
aliases so an existing launch line or pack contract keeps working.

The module is inert on import: it imports no MLX and loads no extension.
"""

from __future__ import annotations

import os
from typing import Mapping


#: Boolean vocabulary, matching upstream's lenient ``os.environ.get`` readers.
TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})

#: lane -> primary env key (upstream's namespace).
LANE_KEYS: dict[str, str] = {
    "ple_cached_aux": "MTPLX_QWEN4_PLE_CACHED_AUX",
    "qsa_pooled_rowsel": "MTPLX_QSA_POOLED_ROWSEL",
}

#: primary env key -> the PR #391 alias an operator or pack may still use.
LANE_ALIASES: dict[str, str] = {
    "MTPLX_QWEN4_PLE_CACHED_AUX": "MTPLX_FABLE_PLE_CACHED_AUX",
    "MTPLX_QSA_POOLED_ROWSEL": "MTPLX_FABLE_QSA_POOLED_ROWSEL",
}

#: Lane names, in order.
LANES: tuple[str, ...] = tuple(LANE_KEYS)


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in TRUE_TOKENS


def lane_enabled(lane: str, environ: Mapping[str, str] | None = None) -> bool:
    """Is ``lane`` armed in ``environ``? Unset is off (the stock path).

    The primary ``MTPLX_QWEN4_*`` / ``MTPLX_QSA_*`` key is authoritative when it
    is explicitly set (including ``=0`` as the off switch); an unset primary
    falls back to the PR #391 ``MTPLX_FABLE_*`` alias.
    """

    if lane not in LANE_KEYS:
        raise KeyError(f"unknown auxiliary lane {lane!r}; expected one of {LANES}")
    source = os.environ if environ is None else environ
    primary = LANE_KEYS[lane]
    raw = source.get(primary)
    if raw is not None and str(raw).strip():
        return _truthy(raw)
    alias = LANE_ALIASES.get(primary)
    if alias is not None:
        return _truthy(source.get(alias))
    return False


def ple_cached_aux_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``MTPLX_QWEN4_PLE_CACHED_AUX`` read the way its install site reads it."""

    return lane_enabled("ple_cached_aux", environ)


def qsa_pooled_rowsel_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """``MTPLX_QSA_POOLED_ROWSEL`` read the way its install site reads it."""

    return lane_enabled("qsa_pooled_rowsel", environ)


#: lane -> the runtime attribute that carries its load-time install report.
_RUNTIME_REPORT_ATTR: dict[str, str] = {
    "ple_cached_aux": "ple_cached_aux_report",
    "qsa_pooled_rowsel": "qsa_pooled_rowsel_report",
}

#: install-report keys surfaced in the read-only /health entry, when present.
_HEALTH_REPORT_KEYS: tuple[str, ...] = ("native_ext", "reason", "bank_mode")


def health_report(
    lane: str,
    runtime: Any,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    """Read-only /health entry for ``lane``, or ``None`` when it is not armed.

    Shape mirrors the upstream verify-lane reports (commit 6): ``armed`` is read
    at use (gate-able without a request), so an unarmed lane returns ``None`` and
    stays absent from ``qwen4_install_reports`` (== off). When armed, the
    load-time install report on the runtime (set by runtime.py at install) rides
    inside: ``status`` (installed/declined) plus ``native_ext`` (ple_cached_aux,
    installed), ``reason`` (ple_cached_aux, declined) or ``bank_mode``
    (qsa_pooled_rowsel). Never touches the GPU; never changes behaviour.
    """

    if lane not in LANE_KEYS:
        raise KeyError(f"unknown auxiliary lane {lane!r}; expected one of {LANES}")
    if not lane_enabled(lane, environ):
        return None
    entry: dict[str, Any] = {"armed": True}
    report = getattr(runtime, _RUNTIME_REPORT_ATTR[lane], None)
    if isinstance(report, Mapping):
        status = report.get("status")
        if status is not None:
            entry["status"] = status
        for key in _HEALTH_REPORT_KEYS:
            value = report.get(key)
            if value is not None:
                entry[key] = value
    return entry


__all__ = [
    "LANES",
    "LANE_KEYS",
    "LANE_ALIASES",
    "TRUE_TOKENS",
    "lane_enabled",
    "ple_cached_aux_enabled",
    "qsa_pooled_rowsel_enabled",
]
