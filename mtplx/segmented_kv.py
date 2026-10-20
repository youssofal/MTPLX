"""Segmented KV cache for the full-attention layers (MTPLX_SEGMENTED_KV=1, default off).

A conversation's attention KV is a list of immutable SEGMENTS plus one growing TAIL, instead
of one contiguous buffer that the session bank aliases with a lazy snapshot:

* a sealed segment has its exact capacity and is never written again;
* the tail is the only buffer a decode, a verify forward or a prefill chunk writes into;
  verify rollback (``trim``) only moves the tail's row count;
* ``seal()`` closes the tail at the end of a turn (a copy of the tail only, never of the
  history) and a new tail starts empty at the next write;
* a snapshot of the cache (``state``) IS the list of segment references: it seals the tail
  and hands out ``SegmentedKVState`` (segment, rows) references. Nothing a later turn writes
  can alias them, so MLX's copy-on-write never copies the history. A fork or a return into the
  middle of a segment is a reference ``(segment, n')`` with n' below the segment's rows:
  no copy, and the rows past n' belong to whoever else holds the segment.

Why (receipts, 2026-10-06/07, Qwen3.8-27B, 16 full-attention layers, 64 KiB of KV per token):
the session bank's snapshot views alias the per-layer KV buffers, so the first write of the
next turn copies every buffer: +4.95 GiB at 80K (+3.1 GiB at 50K), whether the restore is a
clone or a reference lease (docs/onderzoek/2026-10-07-lease-vs-clone.md). With segments the
same follow-up turn costs the new segment only (+0.125 GiB for 1,024 tokens plus slack,
2026-10-07 phase 1 measurement) and a branch from the same history costs 0.

Attention over the segments: ``mtplx/kernels/sdpa_segmented.py`` (exact split-softmax merge,
one launch for up to 12 segments) for the verify/decode window. A cache with a single segment
(a cold prompt, or a conversation not yet sealed) runs the stock route on that segment's rows,
a zero-copy view of the buffer the stock cache would have (counter ``single_q<rows>``). Only
a call over several segments that no kernel takes (a masked or oversized window, a missing
logsumexp kernel for prefill) gathers the rows into one contiguous array (``gather_q<rows>``,
correct and slower for long histories) and runs the stock route on it. The prefill over existing segments has its own
interface, ``attend_segments_lse`` (per segment ``(out, lse)`` plus one merge), so that an
MLX kernel which returns the logsumexp can replace ``segment_sdpa_lse``.

GatedDeltaNet (recurrent) layers keep their state objects as they are.

Sharing. Segments are shared by plain Python references: a segment lives as long as a cache
or a ``SegmentedKVState`` snapshot references it, so an evicted or dropped bank entry releases
its segments by going away. The merge policy (``SegmentedKVCache.compact``) also merges small
segments that snapshots still hold: the snapshots keep their own pieces and the merged copy is
the only new memory.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx

_SEGMENT_IDS = itertools.count(1)


def _env_on(name: str, *, default: bool = False) -> bool:
    """A boolean switch: 1/true/yes/on and 0/false/no/off, anything else (or unset) is ``default``."""
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def segmented_kv_requested() -> bool:
    """MTPLX_SEGMENTED_KV (default off), read on every call: what the operator asked for."""
    return _env_on("MTPLX_SEGMENTED_KV")


def segment_route_switched_off() -> list[str]:
    """The attention switches an operator set to 0 explicitly (kill switches for the segment route).

    Segments do not need MTPLX_GQA_PACKED_SDPA or MTPLX_NAX_FLASH_ROUTE on (they run their own
    kernel, under any profile); only an explicit 0 on either turns them off.
    """
    reasons = []
    if not _env_on("MTPLX_GQA_PACKED_SDPA", default=True):
        reasons.append("gqa_packed_sdpa_off")
    if not _env_on("MTPLX_NAX_FLASH_ROUTE", default=True):
        reasons.append("nax_flash_route_off")
    return reasons


#: Verdict on the loaded model (``evaluate_model_support``); None until a model was checked.
_MODEL_SUPPORT: dict[str, Any] | None = None
_UNSUPPORTED_LOGGED = False
_LSE_NOTE_LOGGED = False


def segmented_kv_enabled() -> bool:
    """MTPLX_SEGMENTED_KV, once the loaded model was checked and can use segments.

    Without a verdict everything behaves as with the switch off (stock caches, stock bank, stock
    SSD path): the verdict is made in ``configure_split_full_attention``, and a model that loads
    without it (Laguna, a script that builds its own model) has no split-attention hook, so its
    attention would gather every segment on every call.
    """
    if not segmented_kv_requested():
        return False
    return _MODEL_SUPPORT is not None and bool(_MODEL_SUPPORT.get("supported", False))


def model_support() -> dict[str, Any] | None:
    return None if _MODEL_SUPPORT is None else dict(_MODEL_SUPPORT)


def reset_model_support() -> None:
    global _MODEL_SUPPORT, _UNSUPPORTED_LOGGED, _LSE_NOTE_LOGGED
    _MODEL_SUPPORT = None
    _UNSUPPORTED_LOGGED = False
    _LSE_NOTE_LOGGED = False


def evaluate_model_support(layers: Iterable[Any], *, hooked: Sequence[bool] | None = None) -> dict[str, Any]:
    """Whether the segmented cache can serve this model without ever gathering the history.

    A call that the segment kernel does not take runs on one gathered copy of the history
    (correct, but a full copy per layer per verify), so a model whose full-attention layers the
    kernel cannot serve keeps the stock cache instead. ``layers`` are the full-attention modules
    (``head_dim``, ``num_attention_heads``, ``num_key_value_heads``; ``hooked`` says which of
    them run the split-attention hook, the only code that reads segments). The verdict is
    stored; it switches the feature off (``segmented_kv_enabled``) and is reported on /health.
    """
    from .kernels.sdpa_segmented import SUPPORTED_HEAD_DIMS, segments_supported
    from .nax_verify import nax_available

    global _MODEL_SUPPORT, _LSE_NOTE_LOGGED
    layers = list(layers)
    flags = list(hooked) if hooked is not None else [True] * len(layers)
    reasons: list[str] = []
    shapes: set[tuple[int, int, int]] = set()
    if not layers:
        reasons.append("no_full_attention_layers")
    if not all(flags):
        reasons.append("attention_not_hooked")
    for attn in layers:
        head_dim = getattr(attn, "head_dim", None)
        if not head_dim and getattr(attn, "scale", None):
            head_dim = round(float(attn.scale) ** -2)  # mlx-lm Qwen3/Llama attention: scale = head_dim ** -0.5
        q_heads = getattr(attn, "num_attention_heads", None) or getattr(attn, "n_heads", None)
        kv_heads = getattr(attn, "num_key_value_heads", None) or getattr(attn, "n_kv_heads", None)
        if not (head_dim and q_heads and kv_heads):
            reasons.append("attention_shape_unknown")
            continue
        shapes.add((int(head_dim), int(q_heads), int(kv_heads)))
    for head_dim, q_heads, kv_heads in sorted(shapes):
        if head_dim not in SUPPORTED_HEAD_DIMS:
            reasons.append(f"head_dim_{head_dim}_unsupported")
        elif q_heads % kv_heads or not segments_supported(1, q_heads // kv_heads, head_dim):
            reasons.append(f"gqa_{q_heads}_{kv_heads}_unsupported")
    reasons.extend(segment_route_switched_off())
    if not nax_available():
        reasons.append("nax_unavailable")
    reasons = list(dict.fromkeys(reasons))
    verdict = {
        "supported": not reasons,
        "reasons": reasons,
        "head_dims": sorted({s[0] for s in shapes}),
        "gqa": sorted({s[1] // s[2] for s in shapes if s[2] and s[1] % s[2] == 0}),
        "supported_head_dims": list(SUPPORTED_HEAD_DIMS),
        "full_attention_layers": len(layers),
    }
    _MODEL_SUPPORT = verdict
    _log_unsupported(reasons)
    if not reasons and segmented_kv_requested() and prefill_route() == "gather" and not _LSE_NOTE_LOGGED:
        _LSE_NOTE_LOGGED = True
        print(
            "[mtplx] MTPLX_SEGMENTED_KV is on, but this MLX build has no logsumexp output on the fused "
            "SDPA (neither mx.fast.scaled_dot_product_attention_with_lse nor return_lse): a prefill chunk over existing segments gathers the history first "
            "(measured about 5 % slower prefill on follow-up turns; decode and verify are unaffected)",
            flush=True,
        )
    return verdict


def _log_unsupported(reasons: Sequence[str]) -> None:
    global _UNSUPPORTED_LOGGED
    if reasons and segmented_kv_requested() and not _UNSUPPORTED_LOGGED:
        _UNSUPPORTED_LOGGED = True
        print(
            "[mtplx] MTPLX_SEGMENTED_KV is on but this model cannot use it ("
            + ", ".join(reasons)
            + "): keeping the stock KV cache",
            flush=True,
        )


def kernel_probe_shapes() -> list[tuple[int, int]]:
    """(head_dim, gqa) pairs the kernel self-check probes the segment kernel at.

    Empty unless segments are requested and the loaded model passed the support check: the
    self-check then tests the kernel at the model's own shapes before any cache is built.
    """
    from .kernels.sdpa_segmented import segments_supported

    if not segmented_kv_enabled():
        return []
    support = _MODEL_SUPPORT or {}
    pairs = [(int(d), int(g)) for d in support.get("head_dims", []) for g in support.get("gqa", [])]
    return [(d, g) for d, g in pairs if segments_supported(1, g, d)]


def note_kernel_selfcheck_failed() -> None:
    """The self-check found the segment kernel wrong or failing on this GPU: stock caches from now on.

    Without the kernel every call over several segments would gather the history, so the verdict
    turns off (reason on /health, one log line) before the first cache is built.
    """
    global _MODEL_SUPPORT
    if _MODEL_SUPPORT is None:
        return
    reasons = list(dict.fromkeys([*_MODEL_SUPPORT.get("reasons", []), "segment_kernel_selfcheck_failed"]))
    _MODEL_SUPPORT = {**_MODEL_SUPPORT, "supported": False, "reasons": reasons}
    _log_unsupported(reasons)


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


#: Receipts: which route served segmented prefill chunks and which per-segment attention ran.
route_counts: dict[str, int] = {}


def count_route(route: str) -> None:
    """One more call served by ``route`` (``route_counts``, reported on /health)."""
    route_counts[route] = route_counts.get(route, 0) + 1


_SDPA_LSE: bool | None = None
_SDPA_LSE_API: str | None = None

#: The two MLX spellings of "fused SDPA that also returns the logsumexp", in order of preference.
LSE_API_FUNCTION = "scaled_dot_product_attention_with_lse"
LSE_API_KEYWORD = "scaled_dot_product_attention(return_lse=True)"


def _lse_api_name() -> str | None:
    """Which logsumexp API this MLX build offers: the separate function (proposed upstream
    public API) when present, else the ``return_lse=True`` keyword of the patched build, else None.
    The keyword form cannot be told apart by attribute: ``_probe_sdpa_lse`` runs it."""
    if hasattr(mx.fast, LSE_API_FUNCTION):
        return LSE_API_FUNCTION
    return LSE_API_KEYWORD


def sdpa_with_lse(queries, keys, values, *, scale, mask=None):
    """Fused SDPA returning ``(out, lse)``; lse is natural log, float32, (B, Hq, Q, 1).

    The one place that knows the API. Prefers ``mx.fast.scaled_dot_product_attention_with_lse``
    and falls back to ``scaled_dot_product_attention(..., return_lse=True)``. Stock released
    MLX raises TypeError on the keyword, which callers treat as "no fused lse".

    The output dtype depends on the API: ``scaled_dot_product_attention_with_lse`` returns the
    output in float32 for bfloat16/float16 queries (unrounded, so the merge rounds once),
    ``return_lse=True`` in the query dtype (a second rounding in the merge, the behaviour
    before the function existed). The callers handle both.
    """
    kwargs = {} if mask is None else {"mask": mask}
    function = getattr(mx.fast, LSE_API_FUNCTION, None)
    if function is not None:
        return function(queries, keys, values, scale=scale, **kwargs)
    return mx.fast.scaled_dot_product_attention(
        queries, keys, values, scale=scale, return_lse=True, **kwargs
    )


def _probe_sdpa_lse() -> bool:
    """Whether this MLX build's fused SDPA can return the logsumexp.

    Records which API answered in ``_SDPA_LSE_API`` (``lse_api_found``). Any failure counts as absent.
    """
    global _SDPA_LSE_API
    try:
        q = mx.zeros((1, 1, 1, 64), dtype=mx.bfloat16)
        result = sdpa_with_lse(q, q, q, scale=1.0)
        mx.eval(result)
        ok = isinstance(result, (tuple, list)) and len(result) == 2
    except Exception:  # noqa: BLE001
        ok = False
    _SDPA_LSE_API = _lse_api_name() if ok else None
    return ok


def lse_api_found() -> str | None:
    """Name of the logsumexp API the probe found (None when absent or not probed yet)."""
    return _SDPA_LSE_API if sdpa_lse_available() else None


def sdpa_lse_available() -> bool:
    """Feature detection, once per process (first use)."""
    global _SDPA_LSE
    if _SDPA_LSE is None:
        _SDPA_LSE = _probe_sdpa_lse()
    return _SDPA_LSE


def prefill_route() -> str:
    """How a prefill chunk attends over existing segments.

    ``lse``: per segment (out, lse) plus one merge. Default when the MLX kernel with LSE output
    is present; ``MTPLX_SEGMENTED_KV_PREFILL=lse`` forces it (exact reference route without the
    kernel). ``gather`` (default on stock MLX): one contiguous copy per layer and call, then the
    stock SDPA.
    """
    raw = os.environ.get("MTPLX_SEGMENTED_KV_PREFILL", "").strip().lower()
    if raw in {"lse", "gather"}:
        return raw
    return "lse" if sdpa_lse_available() else "gather"


class KVSegment:
    """One K/V buffer pair, (1, Hk, capacity, D), shared by reference.

    ``sealed`` segments hold exactly ``rows == capacity`` live rows and are never written.
    The tail of a cache is an unsealed segment: only its owner writes it.
    """

    __slots__ = ("block_digests", "id", "keys", "sealed", "values")

    def __init__(self, keys: mx.array, values: mx.array, *, sealed: bool = False) -> None:
        self.id = next(_SEGMENT_IDS)
        self.keys = keys
        self.values = values
        self.sealed = bool(sealed)
        # SSD tier: sha256 and size of the 256-row blocks of this (immutable) segment that a
        # spill already hashed, keyed (is_values, first_row, last_row) in segment rows.
        self.block_digests: dict[tuple[bool, int, int], dict[str, Any]] = {}

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[2])

    @property
    def nbytes(self) -> int:
        return int(self.keys.nbytes) + int(self.values.nbytes)


@dataclass(frozen=True)
class SegRef:
    """The first ``n`` rows of a segment."""

    segment: KVSegment
    n: int


class SegmentedKVState:
    """An immutable snapshot of a segmented cache: references, no buffers of its own.

    This is what the session bank stores as the cache snapshot of a segmented layer. Its
    references keep the segments alive for as long as it lives.
    """

    def __init__(self, refs: Iterable[SegRef]) -> None:
        self.refs: tuple[SegRef, ...] = tuple(refs)

    @property
    def rows(self) -> int:
        return sum(int(ref.n) for ref in self.refs)

    @property
    def nbytes(self) -> int:
        """Bytes of the segments this state references, counted as if it were alone."""
        return _unique_segment_bytes(self.refs)

    def to_arrays(self) -> tuple[mx.array, mx.array] | None:
        """One contiguous (keys, values) pair, the state a stock KV cache would give."""
        return _gather_refs(self.refs)


def _unique_segment_bytes(refs: Iterable[SegRef]) -> int:
    """Bytes of the segments behind ``refs``, each segment counted once."""
    seen: dict[int, int] = {}
    for ref in refs:
        seen.setdefault(ref.segment.id, ref.segment.nbytes)
    return sum(seen.values())


def unique_state_bytes(states: Iterable[Any]) -> int:
    """Bytes of the segments referenced by ``states`` (SegmentedKVState objects; anything else is
    skipped), each segment counted once however many states share it."""
    return _unique_segment_bytes(
        ref for state in states if isinstance(state, SegmentedKVState) for ref in state.refs
    )


def _rows_view(array: mx.array, n: int) -> mx.array:
    return array if int(array.shape[2]) == int(n) else array[..., : int(n), :]


def _gather_refs(refs: Sequence[SegRef]) -> tuple[mx.array, mx.array] | None:
    if not refs:
        return None
    keys = [_rows_view(ref.segment.keys, ref.n) for ref in refs]
    values = [_rows_view(ref.segment.values, ref.n) for ref in refs]
    if len(refs) == 1:
        return keys[0], values[0]
    return mx.concatenate(keys, axis=2), mx.concatenate(values, axis=2)


def _own_copy(array: mx.array) -> mx.array:
    """An evaluated array that owns its buffer (a slice of a larger buffer is copied)."""
    owned = mx.contiguous(array)
    mx.eval(owned)
    return owned


class SegmentedKVCache:
    """KV cache of one full-attention layer as sealed segments plus a growing tail.

    Drop-in for the stock ``KVCache`` where the model code needs it: ``update_and_fetch``,
    ``offset`` (the TOTAL number of rows, which is also the rotary offset and what every
    route gate must read instead of a buffer capacity), ``make_mask``, ``trim``, ``state``.
    """

    step = 256

    def __init__(self, *, step: int = 256) -> None:
        self.step = int(step)
        self._sealed: list[SegRef] = []
        self._tail: KVSegment | None = None
        self._tail_n = 0

    # ---- shape ----------------------------------------------------------------------------

    @property
    def offset(self) -> int:
        return sum(int(ref.n) for ref in self._sealed) + self._tail_n

    @offset.setter
    def offset(self, value: int) -> None:
        # Moving the offset is a trim (the stock cache has no other use for the setter here).
        value = int(value)
        current = self.offset
        if value > current:
            raise ValueError("a segmented KV cache cannot grow its offset without rows")
        self.trim(current - value)

    @property
    def segment_count(self) -> int:
        return len(self._sealed) + (1 if self._tail_n else 0)

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self.offset == 0

    def is_trimmable(self) -> bool:
        return True

    @property
    def nbytes(self) -> int:
        """Bytes this cache keeps allocated: its unique segments plus the tail buffer."""
        total = _unique_segment_bytes(self._sealed)
        if self._tail is not None:
            total += self._tail.nbytes
        return total

    @property
    def meta_state(self) -> str:
        return ""

    @meta_state.setter
    def meta_state(self, value: Any) -> None:
        del value

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)

    # ---- writing --------------------------------------------------------------------------

    def append_rows(self, keys: mx.array, values: mx.array) -> None:
        """Write new rows into the tail (the only buffer this cache ever writes)."""
        if self._tail is None:
            self.compact()
        prev = self._tail_n
        steps = int(keys.shape[2])
        tail = self._tail
        if tail is None or prev + steps > tail.capacity:
            b, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + steps - 1) // self.step
            new_k = mx.zeros((b, n_kv_heads, n_steps * self.step, k_head_dim), keys.dtype)
            new_v = mx.zeros((b, n_kv_heads, n_steps * self.step, v_head_dim), values.dtype)
            if tail is not None:
                old_k, old_v = tail.keys, tail.values
                if prev % self.step != 0:
                    old_k, old_v = old_k[..., :prev, :], old_v[..., :prev, :]
                new_k = mx.concatenate([old_k, new_k], axis=2)
                new_v = mx.concatenate([old_v, new_v], axis=2)
            tail = KVSegment(new_k, new_v)
            self._tail = tail
        tail.keys[..., prev : prev + steps, :] = keys
        tail.values[..., prev : prev + steps, :] = values
        self._tail_n = prev + steps

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        self.append_rows(keys, values)
        gathered = self.gather()
        assert gathered is not None
        return gathered

    # ---- reading --------------------------------------------------------------------------

    def attention_segments(self) -> list[tuple[mx.array, mx.array, int]]:
        """(keys, values, rows) per segment, oldest first; the last is the tail when it has rows."""
        out = [(ref.segment.keys, ref.segment.values, int(ref.n)) for ref in self._sealed]
        if self._tail is not None and self._tail_n:
            out.append((self._tail.keys, self._tail.values, self._tail_n))
        return out

    def gather(self) -> tuple[mx.array, mx.array] | None:
        """One contiguous (keys, values) over all rows (lazy; a view when there is one segment)."""
        refs = list(self._sealed)
        if self._tail is not None and self._tail_n:
            refs.append(SegRef(self._tail, self._tail_n))
        return _gather_refs(refs)

    @property
    def keys(self) -> mx.array | None:
        gathered = self.gather()
        return None if gathered is None else gathered[0]

    @property
    def values(self) -> mx.array | None:
        gathered = self.gather()
        return None if gathered is None else gathered[1]

    # ---- rollback -------------------------------------------------------------------------

    def trim(self, n: int) -> int:
        """Drop the last ``n`` rows. Verify rollback only ever moves the tail's row count."""
        n = min(self.offset, max(0, int(n)))
        left = n
        take = min(self._tail_n, left)
        self._tail_n -= take
        left -= take
        if self._tail_n == 0 and self._tail is not None and left > 0:
            self._tail = None
        while left > 0 and self._sealed:
            ref = self._sealed[-1]
            if ref.n <= left:
                self._sealed.pop()
                left -= int(ref.n)
            else:
                self._sealed[-1] = SegRef(ref.segment, int(ref.n) - left)
                left = 0
        return n

    # ---- sealing and state ----------------------------------------------------------------

    def seal(self) -> bool:
        """Close the tail: a segment of exactly its rows, then a fresh empty tail at the next write.

        Copies the tail's rows only (never a sealed segment); a tail that already has exactly
        its capacity is adopted without a copy. Returns True when a segment was added.
        """
        tail, rows = self._tail, self._tail_n
        if tail is None:
            return False
        if rows == 0:
            self._tail = None
            return False
        count_route("seal")
        if rows == tail.capacity:
            keys, values = tail.keys, tail.values
            mx.eval(keys, values)
        else:
            count_route("seal_copy")
            keys = _own_copy(tail.keys[..., :rows, :])
            values = _own_copy(tail.values[..., :rows, :])
        self._tail = None
        self._tail_n = 0
        self._sealed.append(SegRef(KVSegment(keys, values, sealed=True), rows))
        return True

    @property
    def state(self) -> SegmentedKVState:
        """The snapshot of this cache: seals the tail, then the list of references.

        Reading it has a side effect: the tail is sealed (a copy of the tail's rows). A caller
        about to replace the state does not need to read it first; restores go to fresh caches,
        where there is no tail to seal.
        """
        self.seal()
        return SegmentedKVState(self._sealed)

    @state.setter
    def state(self, value: Any) -> None:
        self._sealed = []
        self._tail = None
        self._tail_n = 0
        if value is None:
            return
        if isinstance(value, SegmentedKVState):
            refs = list(value.refs)
        elif isinstance(value, (tuple, list)) and len(value) == 2 and value[0] is not None:
            # A contiguous pair (an SSD restore, a stock snapshot): one sealed segment.
            # Rows of a stock buffer are a strided view of it (capacity beyond the rows). The
            # fused kernel takes row-contiguous inputs and MLX would copy such a view on every
            # launch (the whole history, per layer, per verify: +34 ms at 50K), so the segment
            # owns exact rows (a no-op when the pair is already row-contiguous).
            keys, values = (_own_copy(a) for a in value)
            segment = KVSegment(keys, values, sealed=True)
            refs = [SegRef(segment, int(keys.shape[2]))]
        else:
            raise ValueError(f"unsupported segmented KV state: {type(value).__name__}")
        self._sealed = refs

    # ---- merge policy ---------------------------------------------------------------------

    def compact(self) -> int:
        """Merge small recent sealed segments among themselves (geometric tiers).

        Two adjacent sealed segments merge when the older is at most ``MTPLX_SEGMENTED_KV_TIER``
        times (default 2) the newer one and the result stays within
        ``MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS`` (default 16384) rows. The big history segment is
        never merged here: a full copy would bring the duplicate back while a snapshot, a
        subagent or a lease references it. A segment that bank snapshots also hold is merged
        too: the merged copy is new memory (at most the merged rows,
        released when the snapshots holding the originals go), and without it nothing would
        ever merge, because every turn's snapshot holds that turn's segments. The segment count
        is thereby bounded by the tier structure (logarithmic in the rows added) instead of
        growing by two per turn. Copying two small segments costs 0.8 ms (512 + 512 rows) to
        4.1 ms (4096 + 4096 rows) for 16 layers (phase 1 measurement). Returns the number of
        merges.
        """
        from .kernels.sdpa_segmented import MAX_FUSED_SEGMENTS

        tier = max(1, _int_env("MTPLX_SEGMENTED_KV_TIER", 2))
        cap_rows = _int_env("MTPLX_SEGMENTED_KV_MERGE_MAX_ROWS", 16384)
        limit = _int_env("MTPLX_SEGMENTED_KV_MAX_SEGMENTS", MAX_FUSED_SEGMENTS)
        merged = 0
        while len(self._sealed) >= 2:
            older, newer = self._sealed[-2], self._sealed[-1]
            small = int(older.n) + int(newer.n) <= cap_rows
            geometric = int(older.n) <= tier * int(newer.n)
            over = len(self._sealed) > limit and small
            if not (small and (geometric or over)):
                break
            self._merge_last_two()
            merged += 1
        return merged

    def _merge_last_two(self) -> None:
        older, newer = self._sealed[-2], self._sealed[-1]
        keys = mx.concatenate(
            [_rows_view(older.segment.keys, older.n), _rows_view(newer.segment.keys, newer.n)], axis=2
        )
        values = mx.concatenate(
            [_rows_view(older.segment.values, older.n), _rows_view(newer.segment.values, newer.n)], axis=2
        )
        mx.eval(keys, values)
        merged = KVSegment(keys, values, sealed=True)
        # The merged rows are a copy of both segments' rows: a block an earlier spill hashed in
        # either keeps its digest, at rows shifted by the older segment's rows for the newer one.
        for ref, shift in ((older, 0), (newer, int(older.n))):
            for (is_values, lo, hi), blob in ref.segment.block_digests.items():
                if hi <= int(ref.n):
                    merged.block_digests[(is_values, lo + shift, hi + shift)] = dict(blob)
        self._sealed[-2:] = [SegRef(merged, int(older.n) + int(newer.n))]
        count_route("merge")


class GatheredKVView:
    """The contiguous (keys, values, offset) of a segmented cache, for the stock route ladder.

    The attention ladder in ``attention_split`` reads ``cache.keys``/``values``/``offset``;
    when no segment route takes a call it runs on this view of one gathered pair, so the
    fallback is the stock route on the same numbers.
    """

    def __init__(self, keys: mx.array, values: mx.array, offset: int) -> None:
        self.keys = keys
        self.values = values
        self.offset = int(offset)

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask

        return create_attention_mask(*args, offset=self.offset, **kwargs)


def gathered_view(cache: SegmentedKVCache) -> GatheredKVView:
    gathered = cache.gather()
    if gathered is None:
        raise ValueError("empty segmented KV cache")
    return GatheredKVView(gathered[0], gathered[1], cache.offset)


# ---- prefill over segments: per segment (out, lse) plus one merge ------------------------


def _segment_sdpa_lse_reference(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    n: int,
    *,
    scale: float,
    causal: bool,
    q_chunk_bytes: int = 256 * 1024 * 1024,
) -> tuple[mx.array, mx.array]:
    """Exact attention of ``queries`` over the first ``n`` rows of one segment, with its logsumexp.

    ``queries`` (1, Hq, Q, D); returns (out float32, lse float32 (1, Hq, Q)). The output stays
    float32 so that the merge rounds once, like the stock attention, instead of rounding every
    segment to the query dtype first (that second rounding put about one element in three one
    bf16 ulp away from the stock output at 20K, against about one in six for the exact result).
    ``causal=True`` marks the segment holding the Q new rows as its last Q rows (query row i
    sees keys up to ``n - Q + i``); otherwise every key is visible. Scores run in float32 and
    are bounded by chunking the query rows: this is the slow, exact reference route (QK^T plus
    logsumexp; phase 1 estimated about 3 times the contiguous prefill at 80K). An MLX attention
    kernel that returns the logsumexp replaces this function (``segment_sdpa_lse``).
    """
    b, hq, q_len, d = (int(x) for x in queries.shape)
    hk = int(keys.shape[1])
    gqa = hq // hk
    k = _rows_view(keys, n).astype(mx.float32)
    v = _rows_view(values, n).astype(mx.float32)
    per_row = hk * gqa * int(n) * 4
    chunk = max(1, min(q_len, q_chunk_bytes // max(1, per_row)))
    outs, lses = [], []
    for start in range(0, q_len, chunk):
        stop = min(q_len, start + chunk)
        qc = queries[:, :, start:stop, :].astype(mx.float32)
        rows = stop - start
        qg = qc.reshape(b, hk, gqa * rows, d)
        scores = (qg @ k.transpose(0, 1, 3, 2)) * scale
        if causal:
            row = mx.arange(gqa * rows) % rows + start
            limit = n - q_len + row
            visible = mx.arange(int(n))[None, :] <= limit[:, None]
            scores = mx.where(visible[None, None], scores, -mx.inf)
        lse = mx.logsumexp(scores, axis=-1)
        # A row without any visible key has lse -inf: weights 0 and output 0 (not -inf - -inf = NaN).
        p = mx.exp(scores - _finite_or_zero(lse)[..., None])
        out = (p @ v).reshape(b, hq, rows, d)
        outs.append(out)
        lses.append(lse.reshape(b, hq, rows))
    if len(outs) == 1:
        return outs[0], lses[0]
    return mx.concatenate(outs, axis=2), mx.concatenate(lses, axis=2)


def _segment_sdpa_lse_fast(queries, keys, values, n, *, scale, causal):
    """Per-segment attention on the MLX kernel that returns the logsumexp (``sdpa_with_lse``)."""
    k, v = _rows_view(keys, n), _rows_view(values, n)
    out, lse = sdpa_with_lse(queries, k, v, scale=scale, mask="causal" if causal else None)
    # The kernel's logsumexp is (B, Hq, Q, 1); the merge works on (B, Hq, Q).
    return out, lse.astype(mx.float32).reshape(out.shape[:-1])


def segment_sdpa_lse(queries, keys, values, n, *, scale, causal):
    """Per segment (out, lse): the MLX kernel when present, else the exact reference route."""
    if sdpa_lse_available():
        try:
            out = _segment_sdpa_lse_fast(queries, keys, values, n, scale=scale, causal=causal)
        except (ValueError, RuntimeError):
            # The fused kernel does not take this shape (the detection probe uses a tiny head
            # dim): the exact reference route serves the call, once counted.
            count_route("sdpa_lse_kernel_unsupported")
        else:
            count_route("sdpa_lse_kernel")
            return out
    count_route("sdpa_lse_reference")
    return _segment_sdpa_lse_reference(queries, keys, values, n, scale=scale, causal=causal)


def _finite_or_zero(x: mx.array) -> mx.array:
    """x with +-inf replaced by 0, so ``exp(lse - x)`` is 0 instead of NaN for lse = -inf."""
    return mx.where(mx.isinf(x), 0.0, x)


def merge_segment_outputs(
    outs: Sequence[mx.array], lses: Sequence[mx.array], *, dtype: mx.Dtype | None = None
) -> mx.array:
    """Exact merge of per-segment (out, lse): sum_s exp(lse_s - L) * out_s with L = logsumexp(lse).

    The sum runs in float32 and is rounded once, to ``dtype`` (default: the first output's dtype).
    """
    dtype = outs[0].dtype if dtype is None else dtype
    if len(outs) == 1:
        return outs[0] if outs[0].dtype == dtype else outs[0].astype(dtype)
    stacked = mx.stack([l.astype(mx.float32) for l in lses], axis=0)
    total = _finite_or_zero(mx.logsumexp(stacked, axis=0))
    merged = None
    for out, lse in zip(outs, lses):
        # lse -inf (no visible key in this segment): weight 0 and the segment's out is ignored,
        # even if a kernel left NaN there. A row that is -inf in every segment merges to zeros.
        lse = lse.astype(mx.float32)
        weight = mx.exp(lse - total)[..., None]
        term = mx.where(mx.isinf(lse)[..., None], 0.0, out.astype(mx.float32) * weight)
        merged = term if merged is None else merged + term
    return merged.astype(dtype)


def attend_segments_lse(queries: mx.array, cache: SegmentedKVCache, *, scale: float) -> mx.array:
    """Attention of the cache's last Q rows (``queries``) over all of its segments, via (out, lse)."""
    segments = cache.attention_segments()
    outs, lses = [], []
    for index, (keys, values, n) in enumerate(segments):
        out, lse = segment_sdpa_lse(
            queries, keys, values, n, scale=scale, causal=(index == len(segments) - 1)
        )
        outs.append(out)
        lses.append(lse)
    return merge_segment_outputs(outs, lses, dtype=queries.dtype)


# ---- the decode/verify route used by attention_split -----------------------------------


#: Verify windows (rows per call) the segmented decode route serves; bigger calls are prefill chunks.
VERIFY_WINDOW_MAX = 32


def decode_segments_attention(
    queries: mx.array,
    cache: SegmentedKVCache,
    *,
    scale: float,
    mask: Any,
    packed_threshold: int,
) -> mx.array | None:
    """Verify/decode attention straight over the segments, or None for the stock ladder.

    Engages under any profile unless MTPLX_GQA_PACKED_SDPA or MTPLX_NAX_FLASH_ROUTE is set to 0
    or the kernel self-check failed the segment kernel: a verify window of 1 to
    ``VERIFY_WINDOW_MAX`` (32) rows, a causal or no mask, and a total length of
    at least ``packed_threshold`` when the cache is one segment (several segments take the
    kernel at any length). The head-dim-split segment kernel takes a window whole when
    ``gqa * q_len <= 32`` and ``q_len <= MAX_Q_LEN`` (10), and a wider one through the wide
    route (``MTPLX_NAX_FLASH_WIDE=1``, one launch); otherwise a longer window runs as
    sub-windows of the rows the kernel takes, each over the segments, so the history is never
    gathered.
    None (the call then runs the stock ladder: on a zero-copy view when the cache is one
    segment, on gathered rows otherwise) comes from a mask, an unsupported shape or a kernel
    bail.
    """
    from .kernel_selfcheck import lane_disabled
    from .kernels.sdpa_segmented import (
        MAX_M_ROWS,
        MAX_Q_LEN,
        sdpa_nax_flash_dsplit_segments,
        segments_supported,
    )

    if segment_route_switched_off() or lane_disabled("segmented_kv_sdpa"):
        return None
    q_len = int(queries.shape[2])
    if not 1 <= q_len <= VERIFY_WINDOW_MAX:
        return None
    if cache.segment_count == 1 and cache.offset < int(packed_threshold):
        # One segment below the packed threshold: the stock route over a zero-copy view,
        # exactly what the stock cache does there (several segments always take the kernel,
        # the alternative would be a gather).
        return None
    if not (mask is None or (isinstance(mask, str) and mask == "causal")):
        return None
    hq, d = int(queries.shape[1]), int(queries.shape[3])
    segments = cache.attention_segments()
    if not segments:
        return None
    hk = int(segments[0][0].shape[1])
    if hk <= 0 or hq % hk:
        return None
    gqa = hq // hk
    if segments_supported(q_len, gqa, d):
        out = sdpa_nax_flash_dsplit_segments(queries=queries, segments=segments, scale=scale)
        if out is not None:
            count_route(f"fused_q{q_len}")
        return out
    # A window the kernel does not take whole (the wide route off or its lane disabled):
    # sub-windows of the rows it does take (both limits: gqa * rows <= 32 and rows <= 10, so
    # GQA 1 and 2 split at 10 rows, GQA 4 at 8, GQA 6 at 5), each over the keys up to its own
    # last row, so the history is read once per sub-window and never gathered. Row i of the
    # window sees keys up to n - Q + i.
    sub = max(1, min(MAX_Q_LEN, MAX_M_ROWS // gqa))
    if not segments_supported(sub, gqa, d):
        return None
    tail_keys, tail_values, tail_n = segments[-1]
    if tail_n < q_len:
        return None
    outs = []
    for r0 in range(0, q_len, sub):
        r1 = min(q_len, r0 + sub)
        trimmed = segments[:-1] + [(tail_keys, tail_values, tail_n - (q_len - r1))]
        part = sdpa_nax_flash_dsplit_segments(
            queries=queries[:, :, r0:r1, :], segments=trimmed, scale=scale
        )
        if part is None:
            return None
        outs.append(part)
    count_route(f"chunked_q{q_len}")
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


# ---- install --------------------------------------------------------------------------------


def _is_stock_plain_kv(entry: Any) -> bool:
    return type(entry).__name__ == "KVCache" and getattr(entry, "_idx", None) is None


def adapt_layer_for_restore(entry: Any, state: Any) -> Any:
    """The state to restore into ``entry``.

    A segment cache takes either snapshot kind (a stock snapshot, from the SSD tier or a stock
    bank entry, becomes its rows). A stock layer (an MTP cache, a recurrent layer, or a layer
    that already had rows when the cache was built) gets a segmented snapshot as contiguous
    rows. The entry itself is never replaced: with the switch on, the layout is decided once,
    when the cache is built (``install_segmented_attention_kv_cache``).
    """
    if isinstance(entry, SegmentedKVCache):
        if not isinstance(state, SegmentedKVState) and isinstance(state, (tuple, list)) and len(state) == 2 and state[0] is not None:
            count_route("restored_stock_as_segment")
        return state
    if isinstance(state, SegmentedKVState):
        return state.to_arrays()
    return state


def install_segmented_attention_kv_cache(cache: list[Any], *, step: int | None = None) -> dict[str, int | str]:
    """Make the empty full-attention KV layers of a target cache list segment caches.

    Plain dense ``KVCache`` layers without rows become a ``SegmentedKVCache`` from the first
    token (one path with the switch on; there is no context boundary below which the stock
    cache stays). Recurrent layers, rotating/indexed caches, layers that already hold rows and
    caches of other lists (MTP) are left alone.
    """
    stats: dict[str, int | str] = {"enabled": 1, "mode": "segmented", "entries": 0, "skipped": 0}
    from .cache_state import _is_trimmable

    for idx, entry in enumerate(cache or []):
        if entry is None:
            stats["skipped"] = int(stats["skipped"]) + 1
            continue
        if isinstance(entry, SegmentedKVCache):
            stats["entries"] = int(stats["entries"]) + 1
            continue
        rows = int(getattr(entry, "offset", 0) or 0)
        if (
            _is_trimmable(entry)
            and _is_stock_plain_kv(entry)
            and rows == 0
            and getattr(entry, "keys", None) is None
        ):
            cache[idx] = SegmentedKVCache(step=int(step or getattr(entry, "step", 256)))
            stats["entries"] = int(stats["entries"]) + 1
        else:
            stats["skipped"] = int(stats["skipped"]) + 1
    return stats


# ---- SSD tier ---------------------------------------------------------------------------------


def segmented_ssd_enabled() -> bool:
    """Segmented snapshots may go to the SSD tier (MTPLX_SEGMENTED_KV_SSD, default on with the switch)."""
    if not segmented_kv_enabled():
        return False
    return _env_on("MTPLX_SEGMENTED_KV_SSD", default=True)


class SegmentedRows:
    """One tensor of a segmented snapshot (all K or all V rows) as the codec sees an array.

    The SSD codec asks an array for ``shape``, ``dtype``, ``nbytes`` and slices along axis 2.
    A slice inside one segment is a view; a slice across a segment edge is a concatenation of
    the two small slices, so the history is never gathered. Encoded with the codec's own
    256-row blocks, the result is the spec and the block blobs of the stock cache holding the
    same rows (``docs``: SSD tier of the segmented KV cache).
    """

    def __init__(self, refs: Sequence[SegRef], *, values: bool) -> None:
        self._refs = tuple(refs)
        self._values = bool(values)
        first = self._arrays()[0]
        self.dtype = first.dtype
        self._rows = sum(int(r.n) for r in self._refs)
        self.shape = (int(first.shape[0]), int(first.shape[1]), self._rows, int(first.shape[3]))
        self.nbytes = int(first.nbytes) // max(1, int(first.shape[2])) * self._rows
        self.ndim = 4

    def _arrays(self) -> list[mx.array]:
        return [r.segment.values if self._values else r.segment.keys for r in self._refs]

    def rows_slice(self, start: int, stop: int) -> mx.array:
        """Rows [start, stop) as one array (a view when inside one segment)."""
        start, stop = max(0, int(start)), min(self._rows, int(stop))
        parts: list[mx.array] = []
        offset = 0
        for ref, array in zip(self._refs, self._arrays()):
            lo, hi = offset, offset + int(ref.n)
            if hi > start and lo < stop:
                parts.append(array[..., max(start, lo) - lo : min(stop, hi) - lo, :])
            offset = hi
            if offset >= stop:
                break
        if not parts:
            return mx.zeros((*self.shape[:2], 0, self.shape[3]), self.dtype)
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)

    def _block_home(self, start: int, stop: int) -> tuple[KVSegment, int, int] | None:
        """(segment, first, last row in segment rows) when the block lies in one sealed segment."""
        offset = 0
        for ref in self._refs:
            lo, hi = offset, offset + int(ref.n)
            if lo <= start and stop <= hi:
                seg = ref.segment
                return (seg, start - lo, stop - lo) if seg.sealed else None
            offset = hi
        return None

    def in_one_sealed_segment(self, start: int, stop: int) -> bool:
        """True when rows [start, stop) lie inside one sealed segment, which can name the block."""
        return self._block_home(start, stop) is not None

    def known_digest(self, start: int, stop: int) -> dict[str, Any] | None:
        """The blob a spill already wrote for this block, when it lies inside one sealed segment."""
        home = self._block_home(start, stop)
        if home is None:
            return None
        seg, lo, hi = home
        hit = seg.block_digests.get((self._values, lo, hi))
        return dict(hit) if hit is not None else None

    def remember_digest(self, start: int, stop: int, blob: dict[str, Any]) -> None:
        home = self._block_home(start, stop)
        if home is not None:
            seg, lo, hi = home
            seg.block_digests[(self._values, lo, hi)] = {"sha256": str(blob["sha256"]), "nbytes": int(blob["nbytes"])}

    def __getitem__(self, index: Any) -> mx.array:
        if not isinstance(index, tuple) or len(index) != 4 or not all(
            isinstance(i, slice) and i == slice(None) for k, i in enumerate(index) if k != 2
        ):
            raise TypeError("SegmentedRows only slices along axis 2")
        rows = index[2]
        start, stop, step = rows.indices(self._rows)
        if step != 1:
            raise TypeError("SegmentedRows slices have step 1")
        return self.rows_slice(start, stop)


def segmented_state_tensors(state: SegmentedKVState) -> tuple[SegmentedRows, SegmentedRows]:
    """(keys, values) of a snapshot for the SSD codec; refs dropped to zero rows are skipped."""
    refs = [r for r in state.refs if int(r.n) > 0]
    return SegmentedRows(refs, values=False), SegmentedRows(refs, values=True)


# ---- /health ------------------------------------------------------------------------------


def segmented_kv_health(entries: Iterable[Any]) -> dict[str, Any] | None:
    """The segmented-KV block of /health, None when the switch is off (the key is then absent).

    ``entries`` are the session bank's entries. Per entry: the segment count of its snapshot
    (largest over the layers) or of its live cache, and the bytes of the segments it references.
    """
    if not segmented_kv_requested():
        return None
    if not segmented_kv_enabled():
        # Asked for, but the loaded model cannot use it or was never checked: say so (and why)
        # instead of staying silent.
        support = model_support()
        if support is None:
            support = {"supported": False, "reasons": ["model_not_checked"]}
        return {"enabled": False, "requested": True, "model_support": support}
    per_entry: list[dict[str, Any]] = []
    all_states: list[SegmentedKVState] = []
    for entry in entries:
        counts: list[int] = []
        sealed_bytes = 0
        snapshot = getattr(entry, "cache_snapshot", None)
        for state in walk_segmented_states(snapshot.states) if snapshot is not None else ():
            counts.append(len(state.refs))
            sealed_bytes += int(state.nbytes)
            all_states.append(state)
        live = getattr(entry, "cache_ref", None)
        for layer in live or ():
            if isinstance(layer, SegmentedKVCache):
                counts.append(layer.segment_count)
        if counts:
            per_entry.append(
                {
                    "session_id": getattr(entry, "session_id", None),
                    "prefix_len": getattr(entry, "prefix_len", None),
                    "segments": max(counts),
                    "sealed_bytes": sealed_bytes,
                    "live": bool(live),
                }
            )
    counts_now = dict(route_counts)
    return {
        "enabled": True,
        "requested": True,
        "model_support": model_support(),
        "ssd": segmented_ssd_enabled(),
        "prefill_route": prefill_route(),
        "lse_api": lse_api_found(),
        "entries": per_entry,
        "max_segments": max((e["segments"] for e in per_entry), default=0),
        "sealed_bytes_unique": unique_state_bytes(all_states),
        "merges": counts_now.get("merge", 0),
        "seals": counts_now.get("seal", 0),
        "seal_copies": counts_now.get("seal_copy", 0),
        "route_counts": counts_now,
        "ssd_counts": {k: v for k, v in counts_now.items() if k.startswith("ssd_")},
    }


# ---- accounting -----------------------------------------------------------------------------


def walk_segmented_states(value: Any) -> Iterable[SegmentedKVState]:
    """Every SegmentedKVState inside a snapshot tree (tuples, lists, dicts)."""
    if isinstance(value, SegmentedKVState):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from walk_segmented_states(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from walk_segmented_states(item)
