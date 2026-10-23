"""One copy of the conversation's attention state (Flash-Next).

On 2026-09-29 a 138K-token Pi session held the same conversation two or three
times: the session bank's snapshot views, the live cache the next request
wrote into (a view blocks MLX's in-place write, so the first write copied the
whole buffer), and the fixed-M4 verifier's padded bank (promotion
concatenated the history into new arrays). The duplicates switched the
compiled decode route off from 98K tokens and caused both mid-answer 507s.

The one-copy store keeps a single set of buffers per conversation:

- the prefill sizes the QSA buffers once to the rows the verifier's bank will
  have (``qwen4_exp.qsa_rows_target``), so the bank adopts them instead of
  copying (``graphbank.TensorOffsetQSACache.adoptable_rows``);
- buffers a turn left at other rows (the prompt fits them and the bank's
  reserve does not, or a turn on the eager verifier grew them its own way)
  are resized to the bank's rows one layer at a time before the bank adopts
  them (``resize_qsa_buffers``);
- the bank hands the same buffers back at the end of the request
  (``demote(keep_capacity=True)``);
- the session bank keeps the conversation as a reference lease with its
  recurrent anchors and never as views of the live buffers.

Numerics do not change: every reader still reads the rows it read before,
at the widths it read them.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

_DISABLE_VALUES = {"0", "false", "no", "off"}


class QSAResizeWouldCut(ValueError):
    """The requested capacity would truncate a cache's live logical rows."""


def one_copy_enabled() -> bool:
    """The one-copy conversation store (MTPLX_ONE_COPY, on unless set off)."""

    return str(os.environ.get("MTPLX_ONE_COPY", "1")).strip().lower() not in _DISABLE_VALUES


def one_copy_runtime(rt: Any) -> bool:
    """True for a runtime whose verifier keeps the conversation in fixed QSA banks."""

    return one_copy_enabled() and bool(getattr(rt, "qwen4_fixed_m4_compiled_verify", False))


def qsa_entries(cache: Any) -> list[Any]:
    """The QSA layers of a trunk cache, stock or promoted."""

    from .graphbank import TensorOffsetQSACache
    from .models.qwen4_exp import QSACache

    return [
        entry for entry in (cache or ())
        if isinstance(entry, (QSACache, TensorOffsetQSACache))
    ]


def held_qsa_rows(cache: Any) -> int | None:
    """Rows every stock QSA layer of ``cache`` already holds, or None.

    None when the cache has no QSA layer or its layers disagree (then the
    buffers are resized to the bank's rows or the promotion copies them; see
    ``resize_qsa_buffers``).
    """

    from .graphbank import TensorOffsetQSACache

    rows: set[int] = set()
    entries = qsa_entries(cache)
    if not entries:
        return None
    for entry in entries:
        held = (
            entry.capacity
            if isinstance(entry, TensorOffsetQSACache)
            else TensorOffsetQSACache.held_rows(entry)
        )
        if held is None:
            return None
        rows.add(int(held))
    return rows.pop() if len(rows) == 1 else None


def qsa_buffers_short(cache: Any, rows: int) -> bool:
    """Whether a QSA layer of ``cache`` has a buffer shorter than a ``rows``-row bank's.

    Keys, values and raw index keys against ``rows``, the pooled index keys
    against ``rows // ratio`` blocks. The bank grows a short buffer as it is
    built; a suffix that completes no pooled block leaves the pooled buffer
    at its old size while the prefill grew the others to the bank's rows.
    """

    rows = int(rows)
    for entry in qsa_entries(cache):
        kv = getattr(entry, "kv", None)
        leaves = (
            getattr(kv, "keys", None), getattr(kv, "values", None),
            entry.raw_keys, entry.pooled,
        )
        if any(leaf is None for leaf in leaves):
            return True
        keys, values, raw, pooled = leaves
        blocks = rows // max(1, int(entry.ratio))
        if (
            min(int(keys.shape[2]), int(values.shape[2]), int(raw.shape[1])) < rows
            or int(pooled.shape[1]) < blocks
        ):
            return True
    return False


def resize_qsa_buffers(
    cache: Any, rows: int, *, admit: Callable[[int], bool] | None = None,
) -> int:
    """Give every stock QSA layer of ``cache`` buffers of exactly ``rows`` rows.

    For a conversation whose held buffers are not ones its fixed-M4 bank can
    adopt: the prompt fits them but the bank's reserve does not (the
    2026-10-01 Pi session at 147,396 tokens in 147,456 rows, against a
    1,024-row reserve), or a turn on the eager verifier, or a resize that
    stopped, left the layers at other or unequal rows. Each buffer gets what
    the bank's promotion would have padded or cut it into (its rows, then
    zeros) in an allocation of its own, which the bank then adopts.

    One layer at a time: ``admit`` is asked for the layer's new bytes before
    any of them is allocated, the new buffers are written, and only then do
    they replace the old ones, which go with their last reference. A refusal,
    or an allocation that fails while a layer is written, leaves every layer
    whole, at its old rows or at ``rows``, so the conversation's only copy
    stays good. Returns how many layers are still at other rows: 0 when all
    hold ``rows``.
    """

    pending = _pending_layers(cache, rows)
    for done, entry in enumerate(pending):
        if admit is not None and not admit(_layer_bytes(entry, rows)[0]):
            return len(pending) - done
        _resize_layer(entry, rows)
    return 0


def layers_to_resize(cache: Any, rows: int) -> int:
    """How many QSA layers ``resize_qsa_buffers(cache, rows)`` would resize now."""

    return len(_pending_layers(cache, rows))


def _pending_layers(cache: Any, rows: int) -> list[Any]:
    entries = _resizable(cache)
    if entries is None:
        raise ValueError("this cache has a QSA layer that cannot be resized in place")
    return [entry for entry in entries if _layer_bytes(entry, rows)[0] > 0]


def resize_bill(cache: Any, rows: int) -> int | None:
    """Bytes ``resize_qsa_buffers(cache, rows)`` adds, at its peak, to what ``cache`` holds.

    The rows the layers resized before the peak gained, plus one layer's new
    buffers written beside its old ones. On the 2026-10-01 turn (12 layers
    of 147,456 rows, a 148,480-row bank) that is 0.38 GB, where the padded
    copy of the same bank was priced at 4.22 GB. None when a QSA layer of
    ``cache`` cannot be resized (a promoted bank, a missing buffer, or a
    target below the live frontier): the promotion then copies, as it
    always did.
    """

    entries = _resizable(cache)
    if entries is None:
        return None
    gained = peak = 0
    for entry in entries:
        try:
            new, old = _layer_bytes(entry, rows)
        except QSAResizeWouldCut:
            # Capacity planning may decline an in-place resize. Let the
            # caller price its copy/eager path; an actual resize must still
            # reject cutting live rows.
            return None
        peak = max(peak, gained + new)
        gained += new - old
    return peak


def _resizable(cache: Any) -> list[Any] | None:
    """The QSA layers of ``cache`` when every one is stock with all its buffers."""

    from .models.qwen4_exp import QSACache

    entries = qsa_entries(cache)
    if not entries:
        return None
    for entry in entries:
        if not isinstance(entry, QSACache) or any(
            leaf is None
            for leaf in (entry.kv.keys, entry.kv.values, entry.raw_keys, entry.pooled)
        ):
            return None
    return entries


def _resized_buffers(entry: Any, rows: int) -> list[tuple[str, Any, int, int]]:
    """(name, buffer, row axis, rows) for each buffer of ``entry`` not at the bank's rows.

    Keys, values and raw index keys take ``rows`` rows and the pooled index
    keys one block per ``ratio`` of them, as in the bank. A resize never cuts
    into the rows a layer holds.
    """

    rows = int(rows)
    offset = int(entry.kv.offset)
    ratio = max(1, int(entry.ratio))
    # A rewound lease can retain its previous pooled count. Blocks beyond
    # the KV frontier are invalid, just as in QSACache.trim/state and
    # TensorOffsetQSACache.demote (#592).
    valid_pooled = min(int(entry.pooled_len), offset // ratio)
    entry.pooled_len = valid_pooled
    wanted = (
        ("keys", entry.kv.keys, 2, rows, offset),
        ("values", entry.kv.values, 2, rows, offset),
        ("raw_keys", entry.raw_keys, 1, rows, offset),
        ("pooled", entry.pooled, 1, rows // ratio, valid_pooled),
    )
    resized = []
    for name, buffer, axis, target, held in wanted:
        if int(buffer.shape[axis]) == target:
            continue
        if target < held:
            raise QSAResizeWouldCut(
                f"QSA {name} hold {held} rows; resizing to {target} would cut them"
            )
        resized.append((name, buffer, axis, target))
    return resized


def _layer_bytes(entry: Any, rows: int) -> tuple[int, int]:
    """(new, old) bytes of the buffers resizing ``entry`` to ``rows`` replaces."""

    new = old = 0
    for _name, buffer, axis, target in _resized_buffers(entry, rows):
        row_bytes = int(buffer.dtype.size)
        for dim, extent in enumerate(buffer.shape):
            if dim != axis:
                row_bytes *= int(extent)
        new += row_bytes * target
        old += int(buffer.nbytes)
    return new, old


def _resize_layer(entry: Any, rows: int) -> None:
    """One layer's new buffers, written, then swapped in for its old ones.

    Each holds the bank promotion's pad or cut (``TensorOffsetQSACache
    ._fixed_bank``, whose pad is a broadcast zero and never a block of its
    own). A cut is copied, where the bank's slice would be a view that keeps
    the old buffer. Either way only the new buffers are allocated.
    """

    import mlx.core as mx

    from .graphbank import TensorOffsetQSACache

    resized = {}
    for name, buffer, axis, target in _resized_buffers(entry, rows):
        value = TensorOffsetQSACache._fixed_bank(buffer, target, axis)
        if int(buffer.shape[axis]) > target:
            value = mx.asarray(value, copy=True)
        resized[name] = value
    mx.eval(*resized.values())
    if "keys" in resized:
        entry.kv.keys = resized["keys"]
    if "values" in resized:
        entry.kv.values = resized["values"]
    if "raw_keys" in resized:
        entry.raw_keys = resized["raw_keys"]
    if "pooled" in resized:
        entry.pooled = resized["pooled"]
        # Derived from ``pooled``; rebuilt on the next stock read.
        entry.pooled_f32_t = None
    # The old buffers are let go when the command buffer that read them
    # completes, which ``eval`` does not wait for: without this the next
    # layer is admitted and allocated beside them (two layers at once).
    mx.synchronize()


def prefill_rows_target(rt: Any, prompt_tokens: int, plan: Any) -> int:
    """Rows a request's prefill should size its QSA buffers to, or 0.

    The rows the fixed-M4 bank will be built with for this prompt
    (``FixedM4CapacityPlan.rows``), when the prompt starts on the rows-gather
    lane with the capacity bucket in force: there the capacity is a whole
    number of pages (in-place writes stay in place) and a bucket-larger
    buffer changes no value. The dense lane keeps its exact capacity and the
    stock step growth (its buffers are small and its width is arithmetic), so
    it gets 0.
    """

    if plan is None or not one_copy_runtime(rt):
        return 0
    from .graphbank import TensorOffsetQSACache
    from .models.qwen4_exp import _qsa_gather_enabled, _qsa_gather_min_context

    prompt_tokens = max(0, int(prompt_tokens))
    if not (_qsa_gather_enabled() and prompt_tokens >= _qsa_gather_min_context()):
        return 0
    rows = int(plan.rows(prompt_tokens, _qsa_ratio(rt), TensorOffsetQSACache.step))
    if int(getattr(plan, "bucket", 0) or 0) <= 0:
        return 0
    return rows


def prompt_lease_fields(
    cache: Any,
    *,
    committed_mtp_cache: Any,
    hidden: Any,
    prompt_len: int,
    boundaries: Any,
    runtime: Any = None,
) -> dict[str, Any]:
    """``SessionBank.put`` arguments that bank a finished prompt as a lease.

    The prompt's prefill used to be banked as a snapshot of views before the
    answer decoded into the same buffers, which made the first decode write
    copy the whole history. Here the bank takes a lease on the live cache
    and a recurrent anchor at the prompt's end: if the answer is committed,
    its entry replaces this lease and inherits the anchor; if it is not
    (cancelled, refused), a restore rewinds the lease to the anchor. The
    anchor is evaluated now, so it holds its own 115 MB and never pins the
    live recurrent buffers the verifier writes in place. Its size is recorded
    on ``runtime`` (``anchor_nbytes``): it is what the server's admission
    charges for publishing a prompt.
    """

    import mlx.core as mx

    from .cache_state import snapshot_untrimmable_cache

    prompt_len = int(prompt_len)
    snapshot = snapshot_untrimmable_cache(cache)
    leaves = [
        leaf
        for state in snapshot.states
        if state is not None
        for leaf in (state if isinstance(state, (list, tuple)) else [state])
        if isinstance(leaf, mx.array)
    ]
    if leaves:
        mx.eval(*leaves)
    if runtime is not None:
        _record_anchor_nbytes(runtime, sum(int(leaf.nbytes) for leaf in leaves))
    kept = [
        record for record in (boundaries or ())
        if int(record[0]) != prompt_len
    ]
    return {
        "keep_live_ref": True,
        "mtp_history_snapshot": None,
        "mtp_history_cache_ref": committed_mtp_cache,
        "gdn_boundaries": [*kept, (prompt_len, snapshot, hidden)],
    }


def anchor_nbytes(rt: Any) -> int | None:
    """Bytes one prompt anchor holds on this runtime, once one was taken.

    The recurrent state is fixed-size (115,642,384 bytes on Flash-Next), so
    the first prompt lease measures it for every later admission. None before
    that: the admission then prices the publication as the copy it used to
    be, which only ever over-charges.
    """

    value = getattr(rt, "one_copy_anchor_nbytes", None)
    return int(value) if isinstance(value, int) and value > 0 else None


def prefill_slack_rows(rt: Any, prompt_tokens: int, max_tokens: int) -> int:
    """Rows a one-copy prefill allocates past the prompt, or 0.

    The prefill sizes its QSA buffers once to the verifier bank's rows
    (``prefill_rows_target``): the answer's reserve rounded up to the
    capacity bucket, allocated when the prompt is written instead of when
    the bank is built.
    """

    if not one_copy_runtime(rt):
        return 0
    from .graphbank import FixedM4CapacityPlan

    plan = FixedM4CapacityPlan.for_request(max(1, int(max_tokens)), runtime=rt)
    target = prefill_rows_target(rt, prompt_tokens, plan)
    return max(0, int(target) - max(0, int(prompt_tokens)))


def _record_anchor_nbytes(rt: Any, nbytes: int) -> None:
    if int(nbytes) <= 0:
        return
    try:
        rt.one_copy_anchor_nbytes = int(nbytes)
    except Exception:
        # A runtime that takes no attributes keeps the copy price.
        pass


class LeaseReturn:
    """What a warm prefill needs to give its lease back if it stops early.

    Taken right after the restore handed the prefill a one-copy lease, before
    anything is written: the recurrent state at the restore point (a real
    copy, 115.6 MB on Flash-Next, so the prefill's in-place writes cannot
    touch it) and the attention and draft-history offsets there. ``give_back``
    banks the lease again at that point (``SessionBank.return_lease``); a
    later restore rewinds whatever the prefill wrote past it.
    """

    def __init__(self, bank, runtime, *, cache, mtp_history_cache, token_ids,
                 hidden, source, boundaries=()):
        import mlx.core as mx

        from .cache_state import snapshot_untrimmable_cache
        from .session_bank import _cache_kv_offset

        self.bank = bank
        self.runtime = runtime
        self.cache = cache
        self.mtp_history_cache = mtp_history_cache
        self.token_ids = tuple(int(token) for token in token_ids)
        self.source = source
        self.boundaries = list(boundaries or ())
        self.kv_offset = _cache_kv_offset(cache)
        self.mtp_offset = (
            _cache_kv_offset(mtp_history_cache) if mtp_history_cache is not None else None
        )
        snapshot = snapshot_untrimmable_cache(cache)
        leaves = [
            leaf
            for state in snapshot.states
            if state is not None
            for leaf in (state if isinstance(state, (list, tuple)) else [state])
            if isinstance(leaf, mx.array)
        ]
        if leaves:
            mx.eval(*leaves)
        self.anchor = (len(self.token_ids), snapshot, hidden)

    def give_back(self) -> Any:
        if self.kv_offset is None:
            return None
        return self.bank.return_lease(
            self.runtime,
            token_ids=self.token_ids,
            cache=self.cache,
            mtp_history_cache=self.mtp_history_cache,
            anchor=self.anchor,
            kv_offset=self.kv_offset,
            mtp_offset=self.mtp_offset,
            source=self.source,
            boundaries=self.boundaries,
        )


def lease_return(bank, runtime, *, restore_mode, cache, mtp_history_cache, token_ids,
                 hidden, source, boundaries=()) -> LeaseReturn | None:
    """A ``LeaseReturn`` when the restore handed over a one-copy lease, else None."""

    if (
        str(restore_mode) != "reference_lease"
        or token_ids is None
        or not callable(getattr(bank, "return_lease", None))
    ):
        return None
    from .session_bank import _one_copy_cache

    if not _one_copy_cache(cache):
        return None
    return LeaseReturn(
        bank, runtime, cache=cache, mtp_history_cache=mtp_history_cache,
        token_ids=token_ids, hidden=hidden, source=source, boundaries=boundaries,
    )


def hand_back_on_raise(callback, bank, live_cache):
    """``callback`` that hands the verifier's buffers back before a raise leaves.

    A client that cancels an answer raises out of the token callback. The
    prompt's lease is then the conversation's only copy, and the verifier's
    bank still holds it as tensor-offset adapters whose offset the session
    bank cannot read: the restore took the answer's rows for the prompt's own
    and resumed a recurrent state the answer had already advanced (a retry
    decoded other tokens than an uninterrupted turn, 2026-09-30). Demoting
    here, as the end of every answer does (whole buffers, no compaction),
    leaves stock containers that say how far the answer ran, and the restore
    rewinds them to the prompt's anchor. The callback runs between rounds,
    when every buffer is whole; a raise inside a round leaves the adapters,
    and the bank refuses a lease it cannot read (``session_bank._lease_advance``).
    ``live_cache`` returns the trunk cache the answer is decoding into.

    The containers are converted on a copy of that list and published in one
    step. Converted in place, a demotion that failed halfway left stock layers
    beside adapters, and the bank read the stock layers' offset as the whole
    lease's; now a failure leaves every adapter, and the lease is refused.
    """

    def emit(tokens):
        try:
            callback(tokens)
        except BaseException:
            try:
                live = live_cache()
                restored = list(live)
                bank.demote(restored, compact=False, keep_capacity=True)
                live[:] = restored
            except Exception as exc:  # the raise below is the caller's news
                import sys

                print(
                    f"[mtplx] one-copy hand-back on a cancelled answer failed ({exc}); "
                    "its lease will not be served",
                    file=sys.stderr,
                )
            raise

    return emit


def prefill_rows_scope(rows: int):
    """``qwen4_exp.qsa_rows_target(rows)``, or a no-op for 0 (no model import)."""

    import contextlib

    if int(rows or 0) <= 0:
        return contextlib.nullcontext()
    from .models.qwen4_exp import qsa_rows_target

    return qsa_rows_target(int(rows))


def _qsa_ratio(rt: Any) -> int:
    model = getattr(rt, "model", None)
    text = getattr(model, "language_model", model)
    args = getattr(text, "args", None) or getattr(getattr(text, "model", None), "args", None)
    return max(1, int(getattr(args, "indexer_compress_ratio", 4) or 4))
