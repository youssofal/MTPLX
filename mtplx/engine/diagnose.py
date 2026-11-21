"""Why a replay and the MLX path disagree: name the launch.

A program is a list of launches, so a wrong output has a first launch that made
it wrong.  The tools here find it without the model pack's source in hand:

``compare(first, second)``
    Two recordings of one call site, aligned launch by launch.  Says what one
    launches that the other does not, and names the first launch that is the
    same kernel with something different around it: its argument bytes, the
    bytes of a constant it reads, an offset, the launch whose result it reads.

``divergence(engine, first, second, fresh_inputs)``
    Both programs replayed one launch per command buffer on the same inputs.
    Names the first aligned launch whose written bytes differ.

``unwritten(engine, ir, fresh_inputs)``
    One program replayed twice with its working memory filled with two
    different bytes.  Names the first launch that computes from bytes the
    program never wrote, and every output element that depends on them.  For
    those elements the MLX path's own value is whatever its recycled buffers
    held: they are not a function of the inputs.

``explain(...)``
    All of it for one mismatch, with what each program gives against the MLX
    path's outputs and a one-sentence verdict.  ``write_report`` saves it.

Everything runs through the executor; nothing here reads a model.  The stepped
runs are slow (one command buffer per launch) and consume their inputs, so the
caller hands in ``fresh_inputs``, a callable that returns a new evaluated list
each time (``Pristine`` keeps a private copy of one call's inputs for that).
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .ir import READ, WRITE, BufferIR, ProgramIR

# Fills for the two runs of ``unwritten``.  Neither is a value arithmetic
# tends to produce, and a byte is called unwritten only when it holds exactly
# the fill of each run.
FILL_A = 0x55
FILL_B = 0xAA

# A written buffer larger than this is compared by digest, not kept.
KEEP_BYTES = 8 * 1024 * 1024


# ------------------------------------------------------------------ inputs


def copy_bits(array: Any) -> Any:
    """The same bytes in a buffer nothing else holds (lazy; evaluate it).

    Through the integer view, so no float arithmetic touches a value.
    """

    import mlx.core as mx

    views = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32,
             mx.float64: mx.uint64, mx.complex64: mx.uint32}
    shape = array.shape
    flat = array.reshape(1) if array.ndim == 0 else array
    bits = flat.view(views[array.dtype]) if array.dtype in views else flat
    if bits.dtype == mx.bool_:
        copied = mx.logical_or(bits, mx.array(False))
    else:
        copied = mx.bitwise_or(bits, mx.array(0, dtype=bits.dtype))
    if copied.dtype != array.dtype:
        copied = copied.view(array.dtype)
    return copied.reshape(shape)


class Pristine:
    """A private copy of one call's inputs; each call returns a fresh list.

    Take it before anything runs on the inputs.  ``only`` limits the copies to
    those positions (the arrays a program or the MLX path may write into); the
    others are handed out as they are.

    With ``engine`` every copy keeps its array's layout: the whole parent
    buffer is copied and the copy is the same view of it (offset, strides).
    A program recorded on views is only accepted on such copies; without
    ``engine`` the copies are plain contiguous arrays.
    """

    def __init__(self, inputs: list, only: Any = None, engine: Any = None) -> None:
        import mlx.core as mx

        chosen = set(range(len(inputs))) if only is None else {int(i) for i in only}
        if engine is None:
            from . import runtime

            if runtime.available() and hasattr(runtime.native_module().Engine, "clone_layout"):
                engine = runtime.engine()
        self._engine = engine
        self._copied = sorted(chosen)
        if engine is not None:
            mx.eval(*[inputs[i] for i in self._copied])
            self._kept = [
                engine.clone_layout(x) if i in chosen else x for i, x in enumerate(inputs)
            ]
            return
        self._kept = [copy_bits(x) if i in chosen else x for i, x in enumerate(inputs)]
        if self._copied:
            mx.eval(*[self._kept[i] for i in self._copied])

    def __call__(self) -> list:
        import mlx.core as mx

        fresh = list(self._kept)
        if self._engine is not None:
            for index in self._copied:
                fresh[index] = self._engine.clone_layout(self._kept[index])
            return fresh
        for index in self._copied:
            fresh[index] = copy_bits(self._kept[index])
        if self._copied:
            mx.eval(*[fresh[i] for i in self._copied])
        return fresh


def _bytes_of(array: Any) -> bytes:
    from .selftest import bits

    return bits(array)


# ------------------------------------------------------------ static view


@dataclass
class _View:
    """One program with what the comparisons need worked out once."""

    ir: ProgramIR
    by_id: dict[int, BufferIR]
    position: dict[int, int]  # buffer id -> index in ir.buffers (the executor's numbering)
    producer: list[dict[int, int]]  # per step: buffer id -> last earlier step that wrote it
    region: dict[int, tuple[int, int]]  # input buffer id -> (offset, bytes) of its declared arrays
    content: dict[int, str]  # const and seed buffers: a name for their bytes

    @classmethod
    def of(cls, ir: ProgramIR, content: dict[int, str] | None = None) -> "_View":
        by_id = {buf.id: buf for buf in ir.buffers}
        names = dict(content or {})
        for buf in ir.buffers:
            if buf.role in ("const", "seed") and buf.data is not None:
                names[buf.id] = hashlib.blake2b(buf.data, digest_size=8).hexdigest()
        position = {buf.id: index for index, buf in enumerate(ir.buffers)}
        producer: list[dict[int, int]] = []
        last: dict[int, int] = {}
        for index, step in enumerate(ir.steps):
            producer.append({b: last[b] for _s, b, _o, _a in step.bindings if b in last})
            for _slot, buffer_id, _offset, access in step.bindings:
                if access == WRITE:
                    last[buffer_id] = index
        region: dict[int, tuple[int, int]] = {}
        for arr in ir.inputs:
            if arr.buffer in by_id:
                start, size = region.get(arr.buffer, (arr.offset, 0))
                low = min(start, arr.offset)
                high = max(start + size, arr.offset + arr.nbytes)
                region[arr.buffer] = (low, high - low)
        return cls(ir, by_id, position, producer, region, names)

    def read_region(self, buffer_id: int) -> tuple[int, int]:
        """What to look at in a buffer: an input's declared arrays, else all of it."""

        if buffer_id in self.region:
            return self.region[buffer_id]
        return 0, int(self.by_id[buffer_id].nbytes)

    def base(self, buffer_id: int) -> int:
        buf = self.by_id[buffer_id]
        if buf.role == "input" and 0 <= buf.index < len(self.ir.inputs):
            return int(self.ir.inputs[buf.index].offset)
        return 0

    def describe(self, index: int) -> dict[str, Any]:
        step = self.ir.steps[index]
        bindings = []
        for slot, buffer_id, offset, access in sorted(step.bindings):
            buf = self.by_id[buffer_id]
            entry: dict[str, Any] = {
                "argument": slot,
                "access": "read" if access == READ else "write",
                "role": buf.role,
                "buffer": buffer_id,
                "bytes": int(buf.nbytes),
                "offset": int(offset),
            }
            if buf.role in ("input", "output", "pinned"):
                entry["index"] = int(buf.index)
            if buf.role == "input":
                entry["offset_in_input"] = int(offset) - self.base(buffer_id)
                entry["written_in_place"] = bool(buf.clobbered)
            if buf.role in ("const", "seed"):
                entry["content"] = self.content.get(buffer_id, "not in this dump")
            made = self.producer[index].get(buffer_id)
            if made is not None:
                entry["written_by_launch"] = made
                entry["written_by_kernel"] = self.ir.steps[made].kernel_name
            bindings.append(entry)
        return {
            "launch": index,
            "kernel": step.kernel_name,
            "grid": [step.dispatch[0], list(step.dispatch[1]), list(step.dispatch[2])],
            "barrier_before": bool(step.barrier_before),
            "arguments": bindings,
            "argument_bytes": [
                {"argument": slot, "bytes": len(data), "hex": data[:32].hex()}
                for slot, data in sorted(step.inline)
            ],
        }


def _shape_key(view: _View, index: int) -> tuple:
    """What makes two launches candidates for being the same launch."""

    step = view.ir.steps[index]
    return (
        step.kernel_name,
        step.dispatch,
        tuple(sorted((slot, access) for slot, _b, _o, access in step.bindings)),
        tuple(sorted((slot, len(data)) for slot, data in step.inline)),
    )


def _align(first: _View, second: _View) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Pairs of launches that correspond, and those only one program makes."""

    keys_a = [_shape_key(first, i) for i in range(len(first.ir.steps))]
    keys_b = [_shape_key(second, i) for i in range(len(second.ir.steps))]
    if keys_a == keys_b:
        return [(i, i) for i in range(len(keys_a))], [], []
    # Kernel names repeat layer after layer, so match on the ordered list.
    table: dict[tuple, int] = {}
    seq_a = [table.setdefault(k, len(table)) for k in keys_a]
    seq_b = [table.setdefault(k, len(table)) for k in keys_b]
    matcher = difflib.SequenceMatcher(None, seq_a, seq_b, autojunk=False)
    pairs: list[tuple[int, int]] = []
    only_a: list[int] = []
    only_b: list[int] = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            pairs.extend(zip(range(a0, a1), range(b0, b1)))
        else:
            only_a.extend(range(a0, a1))
            only_b.extend(range(b0, b1))
    return pairs, only_a, only_b


def _source(view: _View, index: int, buffer_id: int, offset: int, names: dict[int, int]):
    """Where a launch's argument comes from, in terms both programs share.

    ``names`` gives every paired launch the same number in both programs; a
    result of a launch only this program makes is ("result", "unpaired", ...).
    """

    buf = view.by_id[buffer_id]
    made = view.producer[index].get(buffer_id)
    if made is not None:
        extra = (buf.index, offset - view.base(buffer_id)) if buf.role == "input" else (offset,)
        return ("result", names.get(made, "unpaired")) + extra
    if buf.role == "input":
        return ("input", buf.index, offset - view.base(buffer_id))
    if buf.role == "pinned":
        return ("pinned", buf.index, offset)
    if buf.role in ("const", "seed"):
        return (buf.role, view.content.get(buffer_id, "unknown"), offset)
    return (buf.role, offset)  # working memory or an output nothing wrote yet


def compare(
    first: ProgramIR,
    second: ProgramIR,
    *,
    limit: int = 12,
    content_first: dict[int, str] | None = None,
    content_second: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Two recordings side by side.  See the module text.

    ``content_*`` name the bytes of constants for programs loaded from a dump
    that left the bytes out (``load_dump`` returns them).
    """

    a, b = _View.of(first, content_first), _View.of(second, content_second)
    pairs, only_a, only_b = _align(a, b)
    names_a = {i: number for number, (i, _j) in enumerate(pairs)}
    names_b = {j: number for number, (_i, j) in enumerate(pairs)}
    changed: list[dict[str, Any]] = []
    for i, j in pairs:
        step_a, step_b = first.steps[i], second.steps[j]
        reasons: list[dict[str, Any]] = []
        inline_a, inline_b = dict(step_a.inline), dict(step_b.inline)
        for slot in sorted(inline_a):
            if inline_a[slot] != inline_b.get(slot):
                reasons.append({
                    "what": "argument_bytes",
                    "argument": slot,
                    "first": inline_a[slot][:64].hex(),
                    "second": (inline_b.get(slot) or b"")[:64].hex(),
                })
        bind_b = {slot: (buffer_id, offset, access) for slot, buffer_id, offset, access in step_b.bindings}
        for slot, buffer_id, offset, access in sorted(step_a.bindings):
            other_id, other_offset, _other_access = bind_b[slot]
            if access == WRITE and a.producer[i].get(buffer_id) is None \
                    and b.producer[j].get(other_id) is None \
                    and a.by_id[buffer_id].role in ("temp", "output") \
                    and b.by_id[other_id].role in ("temp", "output"):
                continue  # a fresh result: which buffer receives it is the allocator's choice
            source_a = _source(a, i, buffer_id, offset, names_a)
            source_b = _source(b, j, other_id, other_offset, names_b)
            if source_a != source_b:
                reasons.append({
                    "what": "argument_source",
                    "argument": slot,
                    "access": "read" if access == READ else "write",
                    "first": list(source_a),
                    "second": list(source_b),
                })
        if step_a.barrier_before != step_b.barrier_before:
            reasons.append({"what": "barrier_before", "first": step_a.barrier_before,
                            "second": step_b.barrier_before})
        if reasons:
            changed.append({"first": i, "second": j, "kernel": step_a.kernel_name, "why": reasons})

    def kernels(view: _View, steps: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for index in steps:
            name = view.ir.steps[index].kernel_name
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1]))

    # A barrier, or an argument that is the result of a launch only one
    # program makes, follows from that launch.  The causes are the changes
    # that stand by themselves: other argument bytes, another constant,
    # another input or offset, the result of another paired launch.
    def is_cause(entry: dict[str, Any]) -> bool:
        for why in entry["why"]:
            if why["what"] == "argument_bytes":
                return True
            if why["what"] == "argument_source":
                follows = any(
                    side[0] == "result" and side[1] == "unpaired"
                    for side in (why["first"], why["second"])
                )
                if not follows:
                    return True
        return False

    causes = [entry for entry in changed if is_cause(entry)]
    report: dict[str, Any] = {
        "launches": [len(first.steps), len(second.steps)],
        "paired": len(pairs),
        "only_first": {"count": len(only_a), "kernels": kernels(a, only_a),
                       "launches": [a.describe(i) for i in only_a[:limit]]},
        "only_second": {"count": len(only_b), "kernels": kernels(b, only_b),
                        "launches": [b.describe(j) for j in only_b[:limit]]},
        "paired_but_different": len(changed),
        "different_for_a_reason_of_their_own": len(causes),
        "first_cause": None,
        "first_difference": None,
        "same_program": not only_a and not only_b and not changed,
    }
    for name, entries in (("first_cause", causes), ("first_difference", changed)):
        if entries:
            entry = entries[0]
            report[name] = {
                "why": entry["why"],
                "first": a.describe(entry["first"]),
                "second": b.describe(entry["second"]),
            }
    report["inputs_declared_differently"] = [
        index
        for index, (x, y) in enumerate(zip(first.inputs, second.inputs))
        if (x.dtype, tuple(x.shape)) != (y.dtype, tuple(y.shape))
    ]
    return report


# ------------------------------------------------------------- stepped runs


@dataclass
class Wrote:
    argument: int
    buffer: int
    role: str
    index: int
    bytes: int
    digest: int
    data: bytes | None


def trace(
    engine: Any,
    program: Any,
    inputs: list,
    *,
    fill: int | None = 0,
    lengths: dict[tuple[int, int], int] | None = None,
    keep: int = KEEP_BYTES,
    extra_refs: int = 0,
) -> tuple[list[list[Wrote]] | None, list | None, str]:
    """Replay one launch per command buffer; record what each launch wrote.

    Returns ``(per launch: [Wrote], outputs, "")`` or ``(None, None, reason)``
    when the executor declines the inputs.  ``lengths`` caps how many bytes of
    a written buffer are read at a launch: ``{(launch, argument): bytes}``.
    """

    view = _View.of(program.ir)
    written: list[list[Wrote]] = []

    def after_step(index: int) -> None:
        row: list[Wrote] = []
        seen: set[int] = set()
        for slot, buffer_id, _offset, access in sorted(view.ir.steps[index].bindings):
            if access != WRITE or buffer_id in seen:
                continue
            seen.add(buffer_id)
            buf = view.by_id[buffer_id]
            start, length = view.read_region(buffer_id)
            if lengths is not None and (index, slot) in lengths:
                length = min(length, lengths[(index, slot)])
            count, digest = engine.peek_digest(program, view.position[buffer_id], start, length)
            data = None
            if count <= keep:
                data = engine.peek(program, view.position[buffer_id], start, length)
            row.append(Wrote(slot, buffer_id, buf.role, int(buf.index), count, digest, data))
        written.append(row)

    outputs, reason = engine.step_run(
        program, inputs, after_step=after_step, fill=fill, extra_refs=extra_refs
    )
    if reason:
        return None, None, reason
    return written, list(outputs), ""


def _differing(a: bytes, b: bytes) -> tuple[int, int]:
    """(bytes that differ, position of the first) over the common length."""

    import numpy as np

    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return 0, -1
    x = np.frombuffer(a, dtype=np.uint8, count=n)
    y = np.frombuffer(b, dtype=np.uint8, count=n)
    where = np.flatnonzero(x != y)
    return int(where.size), int(where[0])


def _outputs_differing(first: list, second: list, *, limit: int = 8) -> dict[str, Any]:
    """Which declared outputs differ in their bits, and by how much."""

    rows: list[dict[str, Any]] = []
    count = 0
    for index, (x, y) in enumerate(zip(first, second)):
        if tuple(x.shape) != tuple(y.shape) or x.dtype != y.dtype:
            count += 1
            rows.append({"output": index, "shapes": [list(x.shape), list(y.shape)]})
            continue
        bytes_x, bytes_y = _bytes_of(x), _bytes_of(y)
        if bytes_x == bytes_y:
            continue
        count += 1
        if len(rows) < limit:
            differing, at = _differing(bytes_x, bytes_y)
            item = max(1, x.itemsize)
            rows.append({
                "output": index,
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "bytes_differing": differing,
                "first_element": at // item,
                "elements": int(x.size),
            })
    return {"outputs": min(len(first), len(second)), "differing": count, "first": rows}


def divergence(
    engine: Any,
    first: ProgramIR,
    second: ProgramIR,
    fresh_inputs: Callable[[], list],
    *,
    keep: int = KEEP_BYTES,
) -> dict[str, Any]:
    """The first aligned launch whose written bytes differ between two programs."""

    a, b = _View.of(first), _View.of(second)
    pairs, _only_a, _only_b = _align(a, b)
    lengths_a: dict[tuple[int, int], int] = {}
    lengths_b: dict[tuple[int, int], int] = {}
    for i, j in pairs:
        slots_b = {slot: buffer_id for slot, buffer_id, _o, access in second.steps[j].bindings
                   if access == WRITE}
        for slot, buffer_id, _offset, access in first.steps[i].bindings:
            if access != WRITE or slot not in slots_b:
                continue
            common = min(a.read_region(buffer_id)[1], b.read_region(slots_b[slot])[1])
            lengths_a[(i, slot)] = common
            lengths_b[(j, slot)] = common

    report: dict[str, Any] = {"paired": len(pairs), "first": None}
    program_a, program_b = engine.seal(first), engine.seal(second)
    inputs = fresh_inputs()
    wrote_a, outputs_a, reason = trace(engine, program_a, inputs, lengths=lengths_a, keep=keep)
    del inputs
    if reason:
        report["declined"] = {"program": "first", "reason": reason}
        return report
    inputs = fresh_inputs()
    wrote_b, outputs_b, reason = trace(engine, program_b, inputs, lengths=lengths_b, keep=keep)
    del inputs
    if reason:
        report["declined"] = {"program": "second", "reason": reason}
        return report
    report["outputs"] = _outputs_differing(outputs_a, outputs_b)

    compared = 0
    for i, j in pairs:
        by_slot = {w.argument: w for w in wrote_b[j]}
        for w in wrote_a[i]:
            other = by_slot.get(w.argument)
            if other is None:
                continue
            compared += 1
            if w.bytes == other.bytes and w.digest == other.digest:
                continue
            entry: dict[str, Any] = {
                "argument": w.argument,
                "bytes_compared": min(w.bytes, other.bytes),
                "first": a.describe(i),
                "second": b.describe(j),
            }
            if w.data is not None and other.data is not None:
                differing, at = _differing(w.data, other.data)
                if differing == 0:
                    continue  # equal over the common length
                entry["bytes_differing"] = differing
                entry["first_byte"] = at
            else:
                entry["note"] = "too large to keep: compared by digest"
            report["first"] = entry
            report["written_buffers_compared"] = compared
            return report
    report["written_buffers_compared"] = compared
    return report


def unwritten(
    engine: Any,
    ir: ProgramIR,
    fresh_inputs: Callable[[], list],
    *,
    keep: int = KEEP_BYTES,
    limit: int = 8,
) -> dict[str, Any]:
    """What a program computes from bytes it never wrote.  See the module text."""

    import numpy as np

    view = _View.of(ir)
    program = engine.seal(ir)
    report: dict[str, Any] = {"first_launch": None, "launches": 0, "outputs": [],
                              "too_large_to_classify": 0}
    runs = []
    for fill in (FILL_A, FILL_B):
        inputs = fresh_inputs()
        wrote, outputs, reason = trace(engine, program, inputs, fill=fill, keep=keep)
        del inputs
        if reason:
            report["declined"] = reason
            return report
        runs.append((wrote, outputs))
    (wrote_a, outputs_a), (wrote_b, outputs_b) = runs

    def classify(x: bytes, y: bytes) -> tuple[int, int, int]:
        """(bytes never written, bytes computed from such bytes, first of those)."""

        n = min(len(x), len(y))
        u = np.frombuffer(x, dtype=np.uint8, count=n)
        v = np.frombuffer(y, dtype=np.uint8, count=n)
        differ = u != v
        untouched = (u == FILL_A) & (v == FILL_B)
        derived = np.flatnonzero(differ & ~untouched)
        return int(untouched.sum()), int(derived.size), int(derived[0]) if derived.size else -1

    carried_first = None
    for index, (row_a, row_b) in enumerate(zip(wrote_a, wrote_b)):
        for w, other in zip(row_a, row_b):
            if w.digest == other.digest:
                continue
            if w.data is None or other.data is None:
                report["too_large_to_classify"] += 1
                continue
            untouched, derived, at = classify(w.data, other.data)
            # Bytes of an input's buffer that hold a fill were copied there:
            # nothing fills an input.
            carried = untouched if w.role == "input" else 0
            if carried and carried_first is None:
                carried_first = {"argument": w.argument, "bytes_never_written_copied_in": carried,
                                 "launch": view.describe(index)}
            if derived:
                report["launches"] += 1
                if report["first_launch"] is None:
                    report["first_launch"] = {
                        "argument": w.argument,
                        "bytes_computed_from_unwritten_memory": derived,
                        "first_byte": at,
                        "bytes_looked_at": w.bytes,
                        "launch": view.describe(index),
                    }
    report["first_copy_into_an_input"] = carried_first
    for index, (x, y) in enumerate(zip(outputs_a, outputs_b)):
        bytes_x, bytes_y = _bytes_of(x), _bytes_of(y)
        if bytes_x == bytes_y:
            continue
        untouched, derived, at = classify(bytes_x, bytes_y)
        item = max(1, x.itemsize)
        report["outputs"].append({
            "output": index,
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "elements": int(x.size),
            "bytes_never_written": untouched,
            "bytes_computed_from_unwritten_memory": derived,
            "first_element": (at if at >= 0 else _differing(bytes_x, bytes_y)[1]) // item,
        })
    report["outputs_depending_on_unwritten_memory"] = len(report["outputs"])
    report["outputs"] = report["outputs"][:limit]
    return report


def _unwritten_mask(engine: Any, ir: ProgramIR, fresh_inputs: Callable[[], list]) -> list | None:
    """Per declared output, the bytes that change with the fill (numpy bool), or None."""

    import numpy as np

    program = engine.seal(ir)
    results = []
    for fill in (FILL_A, FILL_B):
        inputs = fresh_inputs()
        outputs, reason = engine.step_run(
            program, inputs, after_step=lambda _index: None, fill=fill, extra_refs=0
        )
        del inputs
        if reason:
            return None
        results.append([_bytes_of(x) for x in outputs])
    return [
        np.frombuffer(x, dtype=np.uint8) != np.frombuffer(y, dtype=np.uint8)
        for x, y in zip(*results)
    ]


# ---------------------------------------------------------------- one report


def explain(
    engine: Any,
    first: ProgramIR,
    fresh_inputs: Callable[[], list],
    *,
    second: ProgramIR | None = None,
    reference: list | None = None,
    name: str = "",
) -> dict[str, Any]:
    """Everything about one mismatch.

    ``first`` is the stored program, ``second`` the recording of the call that
    disagreed (the MLX path of that very call), ``reference`` the MLX path's
    outputs for that call, ``fresh_inputs`` that call's inputs as they were
    before anything ran.
    """

    import numpy as np

    report: dict[str, Any] = {"name": name, "launches": len(first.steps)}
    verdict: list[str] = []

    def replay(ir: ProgramIR, *, stepped: bool) -> tuple[list | None, str]:
        program = engine.seal(ir)
        inputs = fresh_inputs()
        if stepped:
            outputs, reason = engine.step_run(
                program, inputs, after_step=lambda _index: None, fill=0, extra_refs=0
            )
        else:
            outputs, reason = engine.try_run(program, inputs, extra_refs=0)
        del inputs
        return (list(outputs) if outputs is not None else None), reason

    one_buffer, reason = replay(first, stepped=False)
    stepped, reason_stepped = replay(first, stepped=True)
    if one_buffer is None or stepped is None:
        report["declined"] = reason or reason_stepped
        report["verdict"] = (
            f"the executor declines these inputs ({report['declined']}); nothing was compared"
        )
        return report
    report["one_command_buffer_against_stepped"] = _outputs_differing(one_buffer, stepped)
    if report["one_command_buffer_against_stepped"]["differing"]:
        verdict.append(
            "the stored program gives other bits in one command buffer than launch by launch: "
            "the order of launches on the GPU matters where the recorded barriers allow overlap"
        )

    # The check on a first use hands the inputs a program writes into as plain
    # private copies.  A program recorded on views of longer buffers must give
    # the same bits on such copies, or say no to them.
    written = sorted(buf.index for buf in first.buffers if buf.role == "input" and buf.clobbered)
    if written:
        import mlx.core as mx

        def plain_inputs() -> list:
            inputs = fresh_inputs()
            for index in written:
                inputs[index] = copy_bits(inputs[index])
            mx.eval(*[inputs[index] for index in written])
            return inputs

        program = engine.seal(first)
        inputs = plain_inputs()
        plain, why = engine.try_run(program, inputs, extra_refs=0)
        del inputs
        if plain is None:
            report["written_inputs_as_plain_copies"] = {"declined": why}
        else:
            against = _outputs_differing(list(plain), stepped)
            report["written_inputs_as_plain_copies"] = against
            if against["differing"]:
                inputs = fresh_inputs()
                on_views, _o, _r = trace(engine, program, inputs)
                del inputs
                inputs = plain_inputs()
                on_copies, _o, _r = trace(engine, program, inputs)
                del inputs
                view = _View.of(first)
                found = None
                if on_views is not None and on_copies is not None:
                    for index, (row_a, row_b) in enumerate(zip(on_views, on_copies)):
                        for w, other in zip(row_a, row_b):
                            if w.digest == other.digest and w.bytes == other.bytes:
                                continue
                            entry: dict[str, Any] = {"argument": w.argument,
                                                     "launch": view.describe(index)}
                            if w.data is not None and other.data is not None:
                                differing, at = _differing(w.data, other.data)
                                entry["bytes_differing"] = differing
                                entry["first_byte"] = at
                                entry["bytes_looked_at"] = [w.bytes, other.bytes]
                            found = entry
                            break
                        if found:
                            break
                report["written_inputs_as_plain_copies"]["first_launch_that_differs"] = found
                named = ""
                if found:
                    named = (f": first at launch {found['launch']['launch']} "
                             f"({found['launch']['kernel']}), argument {found['argument']}")
                verdict.append(
                    "the stored program gives other bits when the inputs it writes into "
                    f"(positions {written}) are plain copies instead of the views it was "
                    f"recorded on{named}"
                )

    mask = _unwritten_mask(engine, first, fresh_inputs)
    fill_dependent = bool(mask) and any(bool(m.any()) for m in mask)
    if fill_dependent:
        report["unwritten"] = unwritten(engine, first, fresh_inputs)
        where = report["unwritten"].get("first_launch") or report["unwritten"].get(
            "first_copy_into_an_input"
        )
        named = ""
        if where:
            named = f" (first at launch {where['launch']['launch']}, {where['launch']['kernel']})"
        verdict.append(
            f"{report['unwritten']['outputs_depending_on_unwritten_memory']} outputs of the stored "
            f"program hold bytes that come from working memory the program never wrote{named}: "
            "for those bytes the MLX path's own value is whatever its recycled buffers held"
        )
    else:
        report["unwritten"] = {"outputs_depending_on_unwritten_memory": 0}

    if reference is not None:
        against = _outputs_differing(stepped, reference)
        report["stored_program_against_mlx"] = against
        if against["differing"] and fill_dependent and mask is not None:
            # Is every differing byte one that depends on the fill?
            outside = 0
            for index, (x, y) in enumerate(zip(stepped, reference)):
                bytes_x, bytes_y = _bytes_of(x), _bytes_of(y)
                if bytes_x == bytes_y or len(bytes_x) != len(bytes_y):
                    continue
                differ = np.frombuffer(bytes_x, dtype=np.uint8) != np.frombuffer(
                    bytes_y, dtype=np.uint8
                )
                outside += int((differ & ~mask[index]).sum())
            report["bytes_differing_from_mlx_that_do_not_depend_on_unwritten_memory"] = outside
            if outside == 0:
                verdict.append(
                    "every byte that differs from the MLX path is one of those bytes: the "
                    "mismatch is in values nothing defines, not in the replay"
                )

    if second is not None:
        report["recordings"] = compare(first, second)
        if reference is not None:
            again, reason = replay(second, stepped=True)
            if again is None:
                report["second_declined"] = reason
            else:
                report["recording_of_this_call_against_mlx"] = _outputs_differing(again, reference)
                same = report["recording_of_this_call_against_mlx"]["differing"] == 0
                stored = report["stored_program_against_mlx"]["differing"] == 0
                if same and not stored:
                    verdict.append(
                        "the recording of this very call reproduces the MLX path and the stored "
                        "program does not: the two programs differ"
                    )
                elif not same:
                    verdict.append(
                        "the recording of this very call does not reproduce the MLX path's own "
                        "outputs: either they are not a function of the inputs (see unwritten "
                        "memory) or the recorder missed something this call read"
                    )
        if not report["recordings"]["same_program"]:
            report["divergence"] = divergence(engine, first, second, fresh_inputs)
            cause = report["recordings"]["first_cause"]
            if cause is not None:
                why = cause["why"][0]
                verdict.append(
                    f"the recordings first differ for a reason of their own at launch "
                    f"{cause['first']['launch']} ({cause['first']['kernel']}): {why['what']} "
                    f"at argument {why.get('argument')}"
                )
            moved = report["divergence"].get("first")
            if moved is not None:
                verdict.append(
                    f"stepped on the same inputs, the first launch that writes other bytes is "
                    f"{moved['first']['launch']} ({moved['first']['kernel']}), argument "
                    f"{moved['argument']}"
                )
            elif "declined" not in report["divergence"]:
                verdict.append(
                    "stepped on the same inputs, every paired launch of the two recordings "
                    "writes the same bytes"
                )
    if not verdict:
        verdict.append("nothing found: the stored program reproduces everything it was compared with")
    report["verdict"] = "; ".join(verdict)
    return report


def write_report(report: dict[str, Any], path: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


def summary(report: dict[str, Any]) -> str:
    """One line for a log."""

    return str(report.get("verdict", ""))


# ------------------------------------------------------------ dumped programs


def load_dump(path: Any) -> tuple[ProgramIR, dict[int, str]]:
    """A program from a dump file, and names for the constants whose bytes the
    dump left out (it keeps their length and a hash)."""

    obj = json.loads(Path(path).read_text())
    content = {
        int(raw["id"]): str(raw["data_blake2b"])
        for raw in obj.get("buffers", [])
        if "data_blake2b" in raw
    }
    return ProgramIR.from_obj(obj), content


def describe_comparison(report: dict[str, Any]) -> str:
    """The comparison of two recordings in a few lines."""

    lines = [
        f"launches: {report['launches'][0]} and {report['launches'][1]}, {report['paired']} paired",
    ]
    for side in ("only_first", "only_second"):
        if report[side]["count"]:
            kernels = ", ".join(f"{count} x {name}" for name, count in report[side]["kernels"].items())
            where = [entry["launch"] for entry in report[side]["launches"]]
            lines.append(f"{side.replace('_', ' ')}: {report[side]['count']} launches ({kernels}) at {where}")
    lines.append(
        f"paired launches that differ: {report['paired_but_different']}, of which "
        f"{report['different_for_a_reason_of_their_own']} for a reason of their own"
    )
    for name in ("first_cause", "first_difference"):
        entry = report[name]
        if entry is None:
            continue
        lines.append(
            f"{name.replace('_', ' ')}: launch {entry['first']['launch']} / "
            f"{entry['second']['launch']} ({entry['first']['kernel']})"
        )
        for why in entry["why"]:
            lines.append(f"    {json.dumps(why)}")
        if name == "first_cause":
            break
    if report["same_program"]:
        lines.append("the same program")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare two dumped recordings launch by launch (no model, no GPU)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("compare", help="two <key>.ir.json files of one call site")
    one.add_argument("first")
    one.add_argument("second")
    one.add_argument("--out", help="write the full comparison as JSON")
    args = parser.parse_args(argv)
    first, content_first = load_dump(args.first)
    second, content_second = load_dump(args.second)
    report = compare(first, second, content_first=content_first, content_second=content_second)
    print(describe_comparison(report))
    if args.out:
        write_report(report, args.out)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
