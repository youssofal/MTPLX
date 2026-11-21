"""Executed fast-path events, per process and per request (``stats.inkling_fast``).

Counted where a call is served, never inside a compiled trace (a trace runs the Python once
and replays the graph afterwards, so a counter there would count tracing, not calls).
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

_lock = Lock()
_counts: Counter = Counter()
request_receipt = ContextVar("inkling_fast_request", default=None)
_captures = ContextVar("inkling_fast_captures", default=())

#: Counts that name the path a stage took inside one step (the pack's own product, an unfused
#: launch, the reference attention): a replayed program runs no stage function, so its site
#: counts again the paths its record call took.
STAGE_PATHS = ("fallback.", "unfused.")


def count(name, n=1):
    with _lock:
        _counts[name] += n
    receipt = request_receipt.get()
    if receipt is not None:
        receipt[name] = receipt.get(name, 0) + n
    for box in _captures.get():
        box[name] = box.get(name, 0) + n


@contextmanager
def capture_counts():
    """The counts made inside the block (counted as usual too), as a dict filled on exit."""
    box: dict = {}
    token = _captures.set(_captures.get() + (box,))
    try:
        yield box
    finally:
        _captures.reset(token)


def stage_paths(box):
    """The stage-path counts of a captured call (``STAGE_PATHS``)."""
    return {name: n for name, n in box.items() if name.startswith(STAGE_PATHS)}


def fast_route_counts():
    with _lock:
        return dict(_counts)
