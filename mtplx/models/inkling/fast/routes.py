"""Executed fast-path events, per process and per request (``stats.inkling_fast``).

Counted where a call is served, never inside a compiled trace (a trace runs the Python once
and replays the graph afterwards, so a counter there would count tracing, not calls).
"""

from __future__ import annotations

from collections import Counter
from contextvars import ContextVar
from threading import Lock

_lock = Lock()
_counts: Counter = Counter()
request_receipt = ContextVar("inkling_fast_request", default=None)


def count(name, n=1):
    with _lock:
        _counts[name] += n
    receipt = request_receipt.get()
    if receipt is not None:
        receipt[name] = receipt.get(name, 0) + n


def fast_route_counts():
    with _lock:
        return dict(_counts)
