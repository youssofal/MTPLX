"""Replay sites of a loaded model, whatever the model.

A site is one fixed-shape function a decode round evaluates: the verify step,
the commit of a rejected window.  The native executor records what the
function launches, checks the recording bit for bit on a later call and from
then on replays it as one command buffer; none of that knows a model.  A
family brings three small things: the attribute that holds its site, the name
the site goes by in ``/health``, and the key that says which routes one
call's trace bakes in.  Everything else is here, so a family's file is a few
lines (``sites_qwen4.py``, ``commit_qwen4.py``) and a new family writes the
same few.

A site belongs to the loaded model, so its programs serve every request of
the process; a request reports its own share through a counter snapshot.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable

from . import buffer_limits as _buffer_limits
from . import runtime as _runtime
from .site import ReplaySite, quiet_start, replay_allowed
from .switches import set_now

#: Every site of the process, for /health.
SITES: "weakref.WeakSet[ReplaySite]" = weakref.WeakSet()


def module_weights(module: Any) -> list:
    """A module's parameters, each array once: what a recorded launch reads
    and nothing writes."""

    if module is None:
        return []
    from mlx.utils import tree_flatten

    arrays = []
    seen: set[int] = set()
    for _name, value in tree_flatten(module.parameters()):
        if hasattr(value, "nbytes") and id(value) not in seen:
            seen.add(id(value))
            arrays.append(value)
    return arrays


def model_arrays(runtime: Any) -> list:
    """The weights of a runtime's model."""

    return module_weights(getattr(runtime, "model", None))


def named_model_arrays(runtime: Any) -> list[tuple[str, Any]]:
    """``(parameter path, array)`` for every weight: a census reads which layer
    and block a launch belongs to from the weights it binds."""

    model = getattr(runtime, "model", None)
    if model is None:
        return []
    from mlx.utils import tree_flatten

    return [(name, value) for name, value in tree_flatten(model.parameters()) if hasattr(value, "nbytes")]


def persistent_operands() -> list:
    """The small arrays a trace reads from memory and the process keeps for
    good (``mtplx/float32_operand.py``: the block-score divisor and its like).

    Pinned, a launch binds the array's own buffer. Unpinned, such a buffer is
    a constant of the program whose bytes are a snapshot taken while
    recording, and on 10 Nov that snapshot was seen to hold other bytes than
    the array did (the 8-byte divisor: 0.48 and 0.42 for sqrt(128) twice).
    """

    from .. import float32_operand

    return list(float32_operand._BUFFERS.values())


def site_for(
    owner: Any,
    attribute: str,
    name: str,
    *,
    weights: Callable[[Any], list],
    pin_operands: bool = False,
    names: Callable[[Any], list] | None = None,
    **site_options: Any,
) -> ReplaySite:
    """The site ``owner`` keeps under ``attribute``, made on first use.

    ``owner`` is what the site's programs live as long as: a runtime, or the
    module whose function is replayed.  ``weights(owner)`` are pinned (looked
    up through a weak reference, so the site never keeps its owner alive);
    ``pin_operands`` also pins the process's persistent operands before each
    recording, for a function whose trace reads them.  ``names(owner)`` gives
    ``(path, array)`` for the pinned weights, read only by a census
    (``ReplaySite.name_pins_from``).  ``site_options`` go to ``ReplaySite``
    (``output_names``, ``fence_strided``, ``dispatch``).
    """

    site = getattr(owner, attribute, None)
    if site is None:
        site = ReplaySite(name, **site_options)
        ref = weakref.ref(owner)
        site.pin_from(lambda: weights(ref()))
        if names is not None:
            site.name_pins_from(lambda: names(ref()))
        if pin_operands:
            site.pin_before_each_recording(persistent_operands)
        # A module files what is assigned to it as a child; a site is neither
        # a parameter nor a submodule.
        object.__setattr__(owner, attribute, site)
        SITES.add(site)
    return site


def commit_recurrent(
    site: ReplaySite | None,
    items: list[tuple[Any, tuple, Any]],
    replay: Callable[[Any, tuple, Any], tuple],
    *,
    keep_tokens: int,
    verified_tokens: int,
    committed_tokens: int | None = None,
) -> list[tuple]:
    """``(conv tail, state)`` after the kept rows for every recurrent layer, in order.

    ``items`` are ``(layer, rows, pre)``: a recurrent layer, the arrays its
    verify forward captured and its state before the verify.
    ``replay(layer, rows, pre)`` is the model's own lazy replay of one layer.
    ``committed_tokens`` is the length of the answer so far (None: unknown):
    nothing is recorded or checked inside an answer's first tokens.

    When ``site`` serves, the results are evaluated arrays and ``replay`` is
    not called. Otherwise (no site, a key warming, being recorded or refused)
    they are ``replay``'s lazy results, and on a call the site records or
    checks they are evaluated here before it returns.
    """

    ticket = None
    inputs: list = []
    if site is not None and not site.declined():
        for _layer, rows, pre in items:
            inputs.extend(rows)
            inputs.extend(pre[:2])
        if all(hasattr(value, "nbytes") for value in inputs):
            key = (int(keep_tokens), int(verified_tokens), len(items))
            ticket = site.begin(
                key,
                inputs,
                quiet=False if committed_tokens is None else quiet_start(committed_tokens),
            )
    if ticket is not None and ticket.native:
        flat = ticket.outputs
    else:
        flat = []
        for layer, rows, pre in items:
            flat.extend(replay(layer, rows, pre))
    if ticket is not None:
        # A recorded or checked call evaluates the model's graph here and
        # closes; a native call closes its times; any other does nothing.
        ticket.finish(flat)
    del inputs
    return [(flat[2 * index], flat[2 * index + 1]) for index in range(len(items))]


def process_report() -> dict[str, Any]:
    """The engine's state and every site's totals, for /health."""

    report: dict[str, Any] = {
        "mode": _runtime.mode(),
        "replay": replay_allowed(),
        # Nothing has to be set for the engine to serve; whatever is set
        # (a rollback, a diagnostic, an unmeasured lever) is said here.
        "switches_set": set_now(),
        "sites": {site.name: site.report() for site in list(SITES)},
        "buffer_rule": _buffer_limits.report(),
    }
    if _runtime.mode() == "0":
        # Off never loads the compiled module.
        report["available"] = None
        return report
    report["available"] = _runtime.available()
    reason = _runtime.unavailable()
    if reason:
        report["unavailable"] = reason
    else:
        report["module_version"] = _runtime.module_version()
        dump = _runtime.dump_dir()
        if dump is not None:
            report["dump_dir"] = str(dump)
        if any(site._engine is not None for site in list(SITES)):
            # The executor's own totals (replays, refusals by reason, waits
            # for an input MLX had not let go of, working memory).
            try:
                report["executor"] = _runtime.engine().counters()
            except Exception as exc:  # noqa: BLE001 - /health must answer
                report["executor"] = {"error": repr(exc)[:200]}
    return report


__all__ = [
    "SITES",
    "commit_recurrent",
    "model_arrays",
    "module_weights",
    "named_model_arrays",
    "persistent_operands",
    "process_report",
    "site_for",
]
