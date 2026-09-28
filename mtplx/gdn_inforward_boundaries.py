"""In-forward GDN restore boundaries for mlx-lm's Qwen3.5/3.6 and Qwen3-Next.

A restore boundary at prompt position p needs each GDN layer's state after
token p - 1. Families without in-forward hooks get it by ENDING a forward at
p (the tail ladder), and on Qwen3.6-35B-A3B every extra forward reads the
routed experts again: ~0.1 s per warm turn for prompts from 512 tokens.

These hooks record the boundary inside the wide forward instead: while a
capture is armed, each GDN layer runs its stock ``__call__`` once per
segment ``[0, p)``, ``[p, S)`` on its own cache, which is exactly what that
layer computes in the ladder's two forwards, and records its state between
the segments. The attention and MoE layers still see the wide forward, so
the rest of the trunk matches the ladder only where the forward width does
not change the rounding: installed together with the batch-invariant
prefill lane (``MTPLX_BATCH_INVARIANT_PREFILL=1``) and never without it.
``MTPLX_GDN_BOUNDARY_INFORWARD=0`` restores the ladder, as for every family.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import mlx.core as mx

from .cache_state import _is_trimmable

_CAPTURE_OFFSETS: ContextVar[tuple[int, ...] | None] = ContextVar(
    "mtplx_gdn_inforward_boundary_offsets", default=None
)
_CAPTURE_ATTR = "_mtplx_boundary_captures"


@contextmanager
def boundary_capture_scope(offsets) -> Iterator[None]:
    """Ask the GDN layers of the next forward to record their state at each
    row offset (0 < offset < rows) on their cache entry."""

    cleaned = tuple(sorted({int(offset) for offset in offsets if int(offset) > 0}))
    token = _CAPTURE_OFFSETS.set(cleaned or None)
    try:
        yield
    finally:
        _CAPTURE_OFFSETS.reset(token)


def take_boundary_captures(cache: list) -> dict:
    """Pop what the last forward captured: ``{offset: [state list or None per
    cache entry]}``. An offset is returned only when every recurrent entry
    captured it, so a partial capture can never become a boundary."""

    stores = []
    complete: set[int] | None = None
    for entry in cache:
        store = getattr(entry, _CAPTURE_ATTR, None)
        if store is not None:
            setattr(entry, _CAPTURE_ATTR, None)
        if _is_trimmable(entry):
            stores.append(None)
            continue
        stores.append(store or {})
        offsets = set(store or {})
        complete = offsets if complete is None else complete & offsets
    return {
        offset: [None if store is None else list(store[offset]) for store in stores]
        for offset in sorted(complete or ())
    }


def _armed_offsets(inputs: mx.array, mask: Any, cache: Any) -> tuple[int, ...]:
    offsets = _CAPTURE_OFFSETS.get()
    batch, rows = int(inputs.shape[0]), int(inputs.shape[1])
    if not offsets or cache is None or mask is not None or batch != 1:
        return ()
    return tuple(offset for offset in offsets if offset < rows)


def _record_capture(cache: Any, offset: int) -> None:
    store = getattr(cache, _CAPTURE_ATTR, None)
    if store is None:
        store = {}
        setattr(cache, _CAPTURE_ATTR, store)
    store[int(offset)] = tuple(cache.state)


class _BoundaryCaptureMixin:
    """GDN layer that, with a capture armed, runs one stock call per segment."""

    def __call__(self, inputs: mx.array, mask: Any = None, cache: Any = None):
        offsets = _armed_offsets(inputs, mask, cache)
        if not offsets:
            return super().__call__(inputs, mask, cache)
        pieces = []
        start = 0
        for edge in (*offsets, int(inputs.shape[1])):
            pieces.append(super().__call__(inputs[:, start:edge], None, cache))
            if edge < int(inputs.shape[1]):
                _record_capture(cache, edge)
            start = edge
        return mx.concatenate(pieces, axis=1)


def _gdn_classes() -> dict[type, type]:
    """Stock GDN class -> its capturing subclass, for the modules present."""

    classes: dict[type, type] = {}
    try:
        from mlx_lm.models import qwen3_5

        classes[qwen3_5.GatedDeltaNet] = type(  # Qwen3.5 / Qwen3.6 (A3B)
            "BoundaryCaptureGatedDeltaNet",
            (_BoundaryCaptureMixin, qwen3_5.GatedDeltaNet),
            {},
        )
    except ImportError:  # pragma: no cover - older mlx-lm
        pass
    from mlx_lm.models import qwen3_next

    classes[qwen3_next.Qwen3NextGatedDeltaNet] = type(
        "BoundaryCaptureQwen3NextGatedDeltaNet",
        (_BoundaryCaptureMixin, qwen3_next.Qwen3NextGatedDeltaNet),
        {},
    )
    return classes


_CAPTURING_CLASSES: dict[type, type] = {}


def install_gdn_inforward_boundaries(model: Any) -> int:
    """Swap every stock GDN layer of ``model`` for its capturing subclass and
    publish the two hooks the prefill loops look for. Returns the number of
    layers swapped; without GDN layers the model is left untouched."""

    if not _CAPTURING_CLASSES:
        _CAPTURING_CLASSES.update(_gdn_classes())
    swapped = 0
    for _name, module in model.named_modules():
        capturing = _CAPTURING_CLASSES.get(type(module))
        if capturing is not None:
            module.__class__ = capturing
            swapped += 1
    if swapped:
        model.boundary_capture_scope = boundary_capture_scope
        model.take_boundary_captures = take_boundary_captures
    return swapped
