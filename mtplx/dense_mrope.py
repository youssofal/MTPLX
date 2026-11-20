"""Interleaved M-RoPE for image requests on the dense Qwen3.5 / Qwen3.8 path.

The dense packs (mlx-lm ``qwen3_5`` text model, and the Bonsai loader that
subclasses it) declare ``mrope_section`` in their config, but mlx-lm builds a
plain ``nn.RoPE`` for them, so image tokens were roped at their sequence index.
The reference (transformers / mlx-vlm ``qwen3_5``) ropes an image token at its
(t, h, w) grid position, splits the rotary frequencies between the three axes
(interleaved: t, h, w, t, h, w, ...), and continues the text after an image at
``max(grid position) + 1``. Every position after the prompt is therefore the
sequence index plus one constant, the request's rope delta.

Where this hooks in
    Every attention route on the dense path (the stock mlx-lm call, the split
    / packed-GQA / paged route in ``attention_split``, the fused q|k|v route
    in ``packed_concats``, the explicit-offset draft-head layer in
    ``mtp_patch``) ropes queries and keys through ONE call,
    ``attn.rope(x, offset=...)``, before the keys enter the cache. The
    attention kernels behind it only ever see roped tensors. So the layer's
    ``rope`` attribute is the single place that has to change:
    :func:`configure_dense_mrope` replaces it, on vision-capable dense packs
    only, with a :class:`DenseMRopeAdapter` around the stock module.

Text requests
    With no state armed the adapter calls the stock module with the caller's
    own arguments, so a text request runs exactly the ops it ran before (one
    context-variable read per rope call is the whole cost). Packs without a
    vision tower, other families, and ``MTPLX_DENSE_MROPE=0`` get no adapter.

Image requests
    The serve layer builds a :class:`DenseMRopeState` (the [3, prompt] table
    and the delta) next to the vision splice; generation arms it for the whole
    request. The adapter reads the host cache offset, so it needs no plumbing
    through the forward:

    * rows past the table: the stock rope at ``offset + delta``;
    * a run of text rows inside the table: the stock rope at the run's first
      position (one call, the same kernel as a text request);
    * rows that touch an image: the stock kernel once per axis with per-row
      positions, then a per-frequency select. With equal axes all three runs
      are the same numbers, so text rows keep bit-identical keys to a text
      request; only image rows and the rows after them change.

Compiled routes
    A compiled or tensor-offset route cannot slice a host table: its offset
    is an array, and inside a shared trace it is a tracer. Those routes own
    the request's rotary origin instead (``mtplx.rope_origin``): every
    tensor-offset cache generation promotes for an admitted image request is
    stamped with the request's delta, the attention routes rotate at
    ``cache.rope_offset`` (``offset + delta``, a graph input of the verify
    trace, never a Python constant), and the adapter treats a tensor offset
    as that origin and calls the stock kernel at it. Past the prompt this is
    the eager route bit for bit, since every row there is the sequence index
    plus the delta on all three axes. The admission in ``mtplx.generation``
    (``_dense_vision_compiled_verify_admission``) names the settings that
    keep an image request eager; the same kill switch as Flash-Next,
    ``MTPLX_QWEN4_VISION_COMPILED_VERIFY=0``, governs both families.
    Dense image compilation is on by default, like Flash-Next's, with its own
    opt-out ``MTPLX_DENSE_VISION_COMPILED_VERIFY=0``. The real 27B serving gate
    found every compiled round bit-exact with the eager verifier from the same
    cache, while a free-running reply can part from a separately run eager arm
    through SDPA rounding over padded buffers (the compiled text lane's own
    rounding class); parity2's common-buffer reference is the per-round proof,
    not a proof of trajectory identity.

Fail closed
    A tensor-offset cache reached by an armed request WITHOUT a delta is a
    routing hole (a route that promoted the cache without the admission):
    the call keeps the stock positions and is counted
    (``MTPLX_DENSE_MROPE_STRICT=1`` raises instead). A table that cannot be
    built, or an attention layer the adapter does not recognise, leaves the
    whole request on sequential positions with a counted demotion: never a
    half-applied table.

Instrument
    ``host_positions_for_tensor_offsets`` makes the adapter resolve a
    CONCRETE tensor offset through the host table, as it does a host
    integer. The parity instrument of the compiled verify bank opens it
    around its reference forward, so the reference is positioned by the
    request's table and not by the delta under test. Never opened around a
    trace: a tracer has no value to resolve, and the call raises.

Rollback: ``MTPLX_DENSE_MROPE=0`` restores sequential image positions.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import demotions

# The tag that separates banked image prefixes roped this way from the ones
# roped sequentially (see mtplx.vision.splice.vision_bank_key_ids).
SCHEME = "dense_mrope_v1"

# Families whose full-attention layers are mlx-lm Qwen3NextAttention with a
# plain nn.RoPE and whose every route is covered by the map above. The MoE
# sibling (qwen3_5_moe, the 35B-A3B) is the same model with routed experts:
# its attention, its draft head and its serial routes are the dense model's.
# Its own A3B target-prefix route carries no rotary delta, so an image
# request roped here never takes it (generation.generate_mtpk), and its
# batch scheduler takes no image request at all.
DENSE_MODEL_TYPES = frozenset({"qwen3_5", "qwen3_5_moe", "prism_hadamard_qwen35"})

INSTALL_ATTR = "_mtplx_dense_mrope"

ROLE_TRUNK = "trunk"
ROLE_MTP = "mtp"

_FALLBACK_KIND = "vision_mrope_sequential_fallback"
_TENSOR_OFFSET_KIND = "vision_mrope_tensor_offset_call"
_TENSOR_OFFSET_REASON = (
    "an attention route reached a tensor-offset cache that owns no rotary "
    "origin during an image request (a compiled or tensor-offset route the "
    "admission did not hand the image delta); that call kept the stock "
    "positions"
)
_TABLE_UNBUILDABLE_REASON = (
    "the image position table could not be built (video pads, or a pad layout "
    "that does not match the images); the request used sequential positions"
)


def note_unowned_tensor_offset() -> None:
    """Count a routing hole: a delta-less tensor-offset cache under an armed
    request (``rope_origin.note_unowned_rotary_origin``). Nothing to count
    for a text request, a request that fell back to sequential positions, or
    the parity instrument's reference leg, whose delta-less containers are
    positioned from the host table on purpose."""

    if _STATE.get() is None or _HOST_PLAN.get():
        return
    demotions.note(_TENSOR_OFFSET_KIND, _TENSOR_OFFSET_REASON)
    if _strict():
        raise RuntimeError(_TENSOR_OFFSET_REASON)


_HOST_PLAN: ContextVar[bool] = ContextVar("mtplx_dense_mrope_host_plan", default=False)


@contextmanager
def host_positions_for_tensor_offsets() -> Iterator[None]:
    """Instrument mode: a concrete tensor offset is resolved through the host
    table like a host integer (see the module docstring, "Instrument")."""

    token = _HOST_PLAN.set(True)
    try:
        yield
    finally:
        _HOST_PLAN.reset(token)


def dense_mrope_enabled() -> bool:
    """``MTPLX_DENSE_MROPE=0`` is the kill switch; anything else is on."""

    return (os.environ.get("MTPLX_DENSE_MROPE") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _strict() -> bool:
    return (os.environ.get("MTPLX_DENSE_MROPE_STRICT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def mrope_axes(section: Sequence[int], interleaved: bool, freq_dim: int) -> list[int]:
    """Position axis (0 = t, 1 = h, 2 = w) of each rotary frequency.

    Interleaved is the transformers / mlx-vlm Qwen3.5 layout: h owns
    frequencies 1, 4, 7, ... below ``3 * section[1]``, w owns 2, 5, 8, ...
    below ``3 * section[2]``, t owns the rest. Chunked is contiguous blocks.
    """

    counts = [int(x) for x in section]
    if len(counts) != 3 or any(count < 0 for count in counts):
        raise ValueError(f"mrope_section must hold three counts, got {section!r}")
    freq_dim = int(freq_dim)
    axes = [0] * freq_dim
    if interleaved:
        for axis, first in ((1, 1), (2, 2)):
            for index in range(first, min(counts[axis] * 3, freq_dim), 3):
                axes[index] = axis
    else:
        cursor = counts[0]
        for axis in (1, 2):
            for index in range(cursor, min(cursor + counts[axis], freq_dim)):
                axes[index] = axis
            cursor += counts[axis]
    if [axes.count(axis) for axis in range(3)] != counts:
        raise ValueError(
            f"mrope_section {counts} does not tile {freq_dim} rotary frequencies"
        )
    return axes


class DenseMRopeState:
    """One request's positions: the prompt table and the decode delta.

    ``table`` is host memory (numpy int32 [3, n]) on purpose: every decision
    the adapter takes is made from the host cache offset without a device
    read. A pure function of the expanded prompt ids and the image grids; it
    is never stored with cache state.
    """

    __slots__ = (
        "_plans",
        "delta",
        "length",
        "mtp_aligned",
        "pad_positions",
        "table",
    )

    def __init__(
        self,
        table: np.ndarray,
        delta: int,
        *,
        pad_positions: np.ndarray | None = None,
    ) -> None:
        table = np.ascontiguousarray(table, dtype=np.int32)
        if table.ndim != 2 or table.shape[0] != 3:
            raise ValueError(f"position table must be [3, n], got {table.shape}")
        self.table = table
        self.delta = int(delta)
        self.length = int(table.shape[1])
        self.pad_positions = (
            None
            if pad_positions is None
            else np.ascontiguousarray(pad_positions, dtype=np.int64)
        )
        # The draft head's history cache ropes by ITS cache offset. That is
        # the prompt index only while the cache holds one row per committed
        # token from the start; generation clears this flag otherwise (cycle
        # or windowed history, a reset cache) and the draft head then keeps
        # the stock rope, which is what it used before.
        self.mtp_aligned = True
        self._plans: dict[tuple[int, int], tuple[str, Any]] = {}

    def check_prompt(self, prompt_ids: Sequence[int], image_token_id: int) -> None:
        """Refuse a prompt whose image tokens moved after the table was built."""

        if self.pad_positions is None:
            return
        ids = np.asarray(prompt_ids, dtype=np.int64)
        found = np.flatnonzero(ids == int(image_token_id))
        if ids.size < self.length or not np.array_equal(found, self.pad_positions):
            raise ValueError(
                "image position table does not match the prompt: the table was "
                f"built for {self.length} tokens with {self.pad_positions.size} "
                f"image tokens, the prompt has {ids.size} tokens with "
                f"{found.size} image tokens at different places"
            )

    def positions(self, start: int, length: int) -> np.ndarray:
        """[3, length] positions of rows ``start .. start + length``."""

        start = int(start)
        end = start + int(length)
        inside = self.table[:, start : min(end, self.length)]
        if end <= self.length:
            return inside
        tail = np.arange(max(start, self.length), end, dtype=np.int32) + np.int32(
            self.delta
        )
        return np.concatenate([inside, np.broadcast_to(tail, (3, tail.size))], axis=1)

    def plan(self, start: int, length: int) -> tuple[str, Any]:
        """How to rope rows ``start .. start + length``.

        ``("shift", p)``: one stock rope call at offset ``p`` (the rows are a
        consecutive run with equal axes). ``("axes", (t, h, w))``: per-row
        positions per axis, as int32 device arrays shared by every layer of
        the forward.
        """

        if start >= self.length:
            return ("shift", start + self.delta)
        key = (int(start), int(length))
        plan = self._plans.get(key)
        if plan is not None:
            return plan
        pos = self.positions(start, length)
        if pos.shape[1] == 0:
            return ("shift", int(start))
        first = int(pos[0, 0])
        run = np.arange(first, first + pos.shape[1], dtype=np.int32)
        if (
            np.array_equal(pos[0], run)
            and np.array_equal(pos[1], run)
            and np.array_equal(pos[2], run)
        ):
            plan = ("shift", first)
        else:
            import mlx.core as mx

            plan = ("axes", tuple(mx.array(pos[axis]) for axis in range(3)))
        if len(self._plans) >= 16:
            self._plans.clear()
        self._plans[key] = plan
        return plan


_STATE: ContextVar[DenseMRopeState | None] = ContextVar(
    "mtplx_dense_mrope", default=None
)


def dense_mrope_state() -> DenseMRopeState | None:
    return _STATE.get()


@contextmanager
def dense_mrope_scope(state: DenseMRopeState | None) -> Iterator[None]:
    """Arm ``state`` for every dense attention forward inside the block."""

    token = _STATE.set(state)
    try:
        yield
    finally:
        _STATE.reset(token)


def state_of(vision_splice: Any | None) -> DenseMRopeState | None:
    """The armed-able state a vision splice carries, or None."""

    if vision_splice is None:
        return None
    state = getattr(vision_splice, "dense_mrope", None)
    return state if isinstance(state, DenseMRopeState) else None


class DenseMRopeAdapter:
    """Stands in for one full-attention layer's ``nn.RoPE``.

    Deliberately not an ``nn.Module``: it owns no parameters and must never
    show up in the model's parameter tree, weight loading or quantization.
    """

    __slots__ = ("_selects", "axes", "inner", "role")

    def __init__(self, inner: Any, axes: Sequence[int], role: str) -> None:
        self.inner = inner
        self.axes = tuple(int(a) for a in axes)
        self.role = role
        self._selects: tuple[Any, Any] | None = None

    def __call__(self, x: Any, offset: Any = 0) -> Any:
        state = _STATE.get()
        if state is None:
            return self.inner(x, offset=offset)
        return self._armed(state, x, offset)

    def _armed(self, state: DenseMRopeState, x: Any, offset: Any) -> Any:
        if self.role == ROLE_MTP and not state.mtp_aligned:
            return self.inner(x, offset=offset)
        if not isinstance(offset, (int, np.integer)):
            if _HOST_PLAN.get():
                # The instrument's reference leg: a concrete offset resolved
                # through the host table. A tracer has no value and raises.
                offset = int(offset.item())
            else:
                # A tensor offset is the caller's rotary origin: the cache of
                # a compiled route rotates at ``offset + delta`` and hands
                # that array here (``rope_origin.rope_offset_of``, which also
                # counts a cache that owns no delta). The stock kernel at it
                # is the eager route's shift plan for every row past the
                # prompt, and no table is ever sliced by a tensor.
                return self.inner(x, offset=offset)
        kind, payload = state.plan(int(offset), int(x.shape[-2]))
        if kind == "shift":
            return self.inner(x, offset=payload)
        return self._rope_axes(x, payload)

    def _select_masks(self) -> tuple[Any, Any]:
        """Bool masks over the rotary features: which follow h, which w."""
        if self._selects is None:
            import mlx.core as mx

            half = len(self.axes)
            take_h = np.zeros(2 * half, dtype=bool)
            take_w = np.zeros(2 * half, dtype=bool)
            for index, axis in enumerate(self.axes):
                # Half-split pairing: frequency i rotates features i and
                # i + half; both follow that frequency's axis.
                if axis == 1:
                    take_h[index] = take_h[index + half] = True
                elif axis == 2:
                    take_w[index] = take_w[index + half] = True
            self._selects = (mx.array(take_h), mx.array(take_w))
        return self._selects

    def _rope_axes(self, x: Any, positions: tuple[Any, Any, Any]) -> Any:
        import mlx.core as mx

        inner = self.inner
        batch, heads, length, width = x.shape
        dims = int(inner.dims)
        # Only the rotary features move; the rest of the head passes through.
        # Working on that slice is the same arithmetic on a quarter of the
        # bytes (64 of 256 on the 27B). One row per token, so the stock
        # kernel takes one position per row.
        rotary = x if width == dims else x[..., :dims]
        rows = rotary.transpose(0, 2, 1, 3).reshape(batch * length, heads, 1, dims)

        def rope_at(axis_positions: Any) -> Any:
            offsets = (
                axis_positions if batch == 1 else mx.tile(axis_positions, (batch,))
            )
            return mx.fast.rope(
                rows,
                dims,
                traditional=inner.traditional,
                base=inner.base,
                scale=inner.scale,
                offset=offsets,
            )

        at_t, at_h, at_w = (rope_at(p) for p in positions)
        take_h, take_w = self._select_masks()
        out = mx.where(take_w, at_w, mx.where(take_h, at_h, at_t))
        out = out.reshape(batch, length, heads, dims).transpose(0, 2, 1, 3)
        if width == dims:
            return out
        return mx.concatenate([out, x[..., dims:]], axis=-1)


@dataclass(frozen=True)
class DenseMRopeInstall:
    """What :func:`configure_dense_mrope` did, kept on the model."""

    installed: bool
    reason: str
    trunk_layers: int = 0
    mtp_layers: int = 0
    section: tuple[int, ...] = ()
    interleaved: bool = True


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def _attention_modules(model: Any) -> tuple[list[Any], list[Any]]:
    text_model = getattr(model, "language_model", model)
    inner = getattr(text_model, "model", text_model)
    trunk = [
        layer.self_attn
        for layer in (getattr(inner, "layers", None) or [])
        if not getattr(layer, "is_linear", False)
        and getattr(layer, "self_attn", None) is not None
    ]
    mtp_module = getattr(text_model, "mtp", None)
    mtp = [
        layer.self_attn
        for layer in (getattr(mtp_module, "layers", None) or [])
        if not getattr(layer, "is_linear", False)
        and getattr(layer, "self_attn", None) is not None
    ]
    return trunk, mtp


def _per_row_offsets_exact(rope: Any) -> bool:
    """The stock kernel, one position per row, equals the stock kernel on a run.

    The adapter's per-row call rests on this (it is what keeps text rows
    bit-identical inside an image request); probed once at load on a 9-row
    tensor so an MLX build without per-row offsets fails closed.
    """

    import mlx.core as mx

    try:
        width = max(int(rope.dims), 2) + 2
        x = (mx.arange(2 * 9 * width, dtype=mx.float32) * 0.37).reshape(1, 2, 9, width)
        x = mx.sin(x)
        run = rope(x, offset=5)
        rows = x.transpose(0, 2, 1, 3).reshape(9, 2, 1, width)
        per_row = mx.fast.rope(
            rows,
            rope.dims,
            traditional=rope.traditional,
            base=rope.base,
            scale=rope.scale,
            offset=mx.arange(5, 14, dtype=mx.int32),
        )
        per_row = per_row.reshape(1, 9, 2, width).transpose(0, 2, 1, 3)
        return bool(mx.array_equal(run, per_row).item())
    except Exception:  # noqa: BLE001 - any failure of the probe means: do not install
        return False


def _stock_rope(attn: Any) -> Any:
    rope = getattr(attn, "rope", None)
    return rope.inner if isinstance(rope, DenseMRopeAdapter) else rope


def _validated_layout(
    config: dict[str, Any], attention: Sequence[Any]
) -> tuple[tuple[int, ...], bool, list[int]] | str:
    """(section, interleaved, axes) when every layer can take the adapter,
    otherwise the plain-English reason it cannot. Touches nothing."""

    import mlx.nn as nn

    rope_parameters = _text_config(config).get("rope_parameters")
    section = (
        rope_parameters.get("mrope_section")
        if isinstance(rope_parameters, dict)
        else None
    )
    if not isinstance(section, (list, tuple)) or len(section) != 3:
        return "the config declares no three-part mrope_section"
    section = tuple(int(x) for x in section)
    # The reference implementations of this family are interleaved only; an
    # absent flag means interleaved.
    interleaved = bool(rope_parameters.get("mrope_interleaved", True))
    if not attention:
        return "no full-attention layers were found"
    for attn in attention:
        rope = _stock_rope(attn)
        if (
            type(rope) is not nn.RoPE
            or bool(rope.traditional)
            or float(rope.scale) != 1.0
            or int(rope.dims) != 2 * sum(section)
        ):
            return (
                "an attention layer does not use a plain half-split rope of "
                f"{2 * sum(section)} dims ({type(rope).__name__})"
            )
    try:
        axes = mrope_axes(section, interleaved, sum(section))
    except ValueError as exc:
        return str(exc)
    if not _per_row_offsets_exact(_stock_rope(attention[0])):
        return "this MLX build does not rope per-row offsets exactly"
    return section, interleaved, axes


def configure_dense_mrope(model: Any, config: dict[str, Any]) -> DenseMRopeInstall | None:
    """Install the adapters on a vision-capable dense pack, all or nothing.

    Returns None (and touches nothing) for other families, packs without a
    vision config, and ``MTPLX_DENSE_MROPE=0``. A dense pack whose attention
    cannot take the adapter gets a not-installed record with the reason; image
    requests on it stay sequential and are counted. A model load never fails
    here: the model is only touched after every layer has been validated, and
    anything unexpected before that point is a not-installed record too.
    """

    if not dense_mrope_enabled() or not isinstance(config, dict):
        return None
    if str(config.get("model_type") or "").lower() not in DENSE_MODEL_TYPES:
        return None
    if not isinstance(config.get("vision_config"), dict):
        return None
    previous = getattr(model, INSTALL_ATTR, None)
    if isinstance(previous, DenseMRopeInstall):
        return previous

    try:
        trunk, mtp = _attention_modules(model)
        layout = _validated_layout(config, [*trunk, *mtp] if trunk else [])
    except Exception as exc:  # noqa: BLE001 - reported on the record, in the log and per request
        layout = f"the model could not be inspected ({exc!r})"
    if isinstance(layout, str):
        install = DenseMRopeInstall(False, layout)
    else:
        section, interleaved, axes = layout
        for role, modules in ((ROLE_TRUNK, trunk), (ROLE_MTP, mtp)):
            for attn in modules:
                if not isinstance(attn.rope, DenseMRopeAdapter):
                    attn.rope = DenseMRopeAdapter(attn.rope, axes, role)
        install = DenseMRopeInstall(
            True,
            "installed",
            trunk_layers=len(trunk),
            mtp_layers=len(mtp),
            section=section,
            interleaved=interleaved,
        )
    setattr(model, INSTALL_ATTR, install)
    return install


def build_request_state(
    model: Any,
    expanded_ids: Sequence[int],
    *,
    image_token_id: int,
    image_grids: Sequence[tuple[int, int, int]],
    spatial_merge_size: int,
    video_token_id: int | None = None,
) -> DenseMRopeState | None:
    """The state for one image request, or None for sequential positions.

    None without a count when the model carries no adapter record (another
    family, a text-only pack, the kill switch). None WITH a counted demotion
    when this is a dense pack and the request still has to fall back.
    """

    install = getattr(model, INSTALL_ATTR, None)
    if not isinstance(install, DenseMRopeInstall) or not dense_mrope_enabled():
        return None
    if not install.installed:
        demotions.note(_FALLBACK_KIND, install.reason)
        return None
    from .vision.mrope import build_mrope_positions

    built = build_mrope_positions(
        expanded_ids,
        image_token_id=int(image_token_id),
        image_grids=list(image_grids),
        spatial_merge_size=int(spatial_merge_size),
        video_token_id=video_token_id,
    )
    if built is None:
        demotions.note(_FALLBACK_KIND, _TABLE_UNBUILDABLE_REASON)
        return None
    table, delta = built
    ids = np.asarray(list(expanded_ids), dtype=np.int64)
    return DenseMRopeState(
        table, delta, pad_positions=np.flatnonzero(ids == int(image_token_id))
    )
