"""From a recording to a program.

The compiled module hands over what it saw: every launch with the buffers its
kernel uses and how it accesses each one.  This file decides what each buffer
is (an input of the call, an output, a pinned weight, a constant, working
memory), refuses what a replay could not reproduce, and writes the result as
``ProgramIR``.  It is plain Python on purpose: it runs once per geometry, and
its rules are easier to read and test here than in the module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .ir import ArrayIR, BufferIR, ProgramIR, StepIR

# A buffer that is read before anything writes it, is not declared and is not
# pinned travels with the program if it is no larger than this.
CONST_LIMIT = 16 * 1024 * 1024

ROLE_CODE = {"input": 0, "output": 1, "pinned": 2, "const": 3, "seed": 4, "temp": 5}

# MLX launches these only to order work between the GPU and a CPU stream, so
# their presence means part of the evaluation ran on the host.
_CPU_BRIDGE_KERNELS = {"fence_wait", "fence_update", "input_coherent"}


# A kernel that implements both a per-item grid and a grouped grid declares
# where the GPU writes its live group counts and how many threads a tile uses.
_INDIRECT_GRIDS: dict[str, tuple[int, tuple[int, int, int]]] = {}


def register_indirect_grid(kernel_name: str, slot: int, threadgroup: tuple[int, int, int]) -> None:
    layout = (slot, threadgroup)
    previous = _INDIRECT_GRIDS.setdefault("custom_kernel_" + kernel_name, layout)
    if previous != layout:
        raise ValueError(f"{kernel_name} already declares a different indirect grid")


class Refused(Exception):
    """Raised inside the builder; ``runtime.build`` turns it into BuildRefused."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class ArrayRef:
    """Where an evaluated array lives, without keeping the array alive."""

    ptr: int  # address of the Metal buffer (0 for an empty array)
    offset: int
    shape: tuple[int, ...]
    dtype: str
    strides: tuple[int, ...]
    data_size: int
    flags: int
    nbytes: int
    itemsize: int
    buffer_length: int
    keep: Any = None  # holds the Metal buffer so its address stays unique

    @classmethod
    def from_native(cls, row: tuple, keep: Any = None) -> "ArrayRef":
        return cls(
            ptr=int(row[0]),
            offset=int(row[1]),
            shape=tuple(int(v) for v in row[2]),
            dtype=str(row[3]),
            strides=tuple(int(v) for v in row[4]),
            data_size=int(row[5]),
            flags=int(row[6]),
            nbytes=int(row[7]),
            itemsize=int(row[8]),
            buffer_length=int(row[9]),
            keep=keep,
        )


def _array_ir(ref: ArrayRef, buffer_id: int) -> ArrayIR:
    return ArrayIR(
        buffer=buffer_id,
        offset=ref.offset,
        shape=ref.shape,
        dtype=ref.dtype,
        strides=ref.strides,
        nbytes=ref.nbytes,
        data_size=ref.data_size,
        flags=ref.flags,
    )


def build_ir(
    exported: dict,
    *,
    read_buffer: Callable[[int, int], bytes | None],
    pin_slot: Callable[[int], int],
    inputs: list[ArrayRef],
    outputs: list[ArrayRef],
    name: str = "",
    const_limit: int | None = None,
) -> ProgramIR:
    """Classify every buffer of a recording and write the program.

    ``exported`` is ``Trace.export()`` of the compiled module.  ``read_buffer``
    returns a trace buffer's bytes now (or None when it is too large).
    ``const_limit`` lowers the size up to which an undeclared buffer may travel
    with the program as a constant (default ``CONST_LIMIT``): a test on a small
    model sets it low to prove that every weight is pinned.
    """

    limit = CONST_LIMIT if const_limit is None else min(int(const_limit), CONST_LIMIT)

    if exported.get("unsupported"):
        raise Refused("unsupported_call", str(exported["unsupported"]))

    raw_steps = exported["steps"]
    trace_buffers = {int(row[0]): row for row in exported["buffers"]}

    # ---- every launch must be one we can describe
    for position, step in enumerate(raw_steps):
        kernel_name = step[1]
        where = f"step {position} ({kernel_name})"
        if not step[2]:
            raise Refused(
                "unknown_kernel",
                f"{where}: compiled before the engine was loaded, its arguments are unknown",
            )
        if step[3]:
            raise Refused("kernel_" + str(step[3]), where)
        if step[12]:
            raise Refused("unbound_argument", f"{where}: slots {list(step[12])}")
        if kernel_name in _CPU_BRIDGE_KERNELS:
            raise Refused("cpu_primitive", f"{where}: part of the evaluation ran on the CPU")

    # ---- how the recording uses each buffer
    first_use: dict[int, str] = {}  # "r" when the first launch that binds it reads it
    first_step: dict[int, int] = {}
    written: dict[int, int] = {}  # buffer -> first step that can write it
    order: list[int] = []
    for position, step in enumerate(raw_steps):
        seen_here: dict[int, str] = {}
        for _slot, buffer, _offset, access in step[4]:
            buffer = int(buffer)
            if access == "r" or buffer not in seen_here:
                seen_here[buffer] = "r" if access == "r" else seen_here.get(buffer, "w")
            if access != "r":
                written.setdefault(buffer, position)
        if len(step) > 13 and step[13] is not None:
            seen_here[int(step[13][0])] = "r"
        for buffer, access in seen_here.items():
            if buffer not in first_use:
                first_use[buffer] = access
                first_step[buffer] = position
                order.append(buffer)

    # ---- what the caller declared
    input_by_ptr: dict[int, list[int]] = {}
    for index, ref in enumerate(inputs):
        if ref.ptr:
            input_by_ptr.setdefault(ref.ptr, []).append(index)
    output_by_ptr: dict[int, list[int]] = {}
    for index, ref in enumerate(outputs):
        if ref.ptr:
            output_by_ptr.setdefault(ref.ptr, []).append(index)

    def where_first(buffer: int) -> str:
        position = first_step[buffer]
        return f"step {position} ({raw_steps[position][1]})"

    def reused_constant(trace_id: int, snapshot: Any, length: int) -> BufferIR | None:
        """The constant an output's buffer held before MLX handed its memory to the output.

        Evaluating one graph, MLX frees a one-use constant (a Python scalar
        made into an array) after its last launch and may give the same Metal
        buffer to an array computed later in the same evaluation. The trace
        sees one address read first and written later. Every read before the
        first write belongs to the constant, so a copy of its first-bind bytes
        serves those reads and the output keeps its own buffer. That buffer
        starts every run with the same bytes (an output seed): MLX also donates
        a live constant to an in-place update (a slice update or a scatter),
        which writes only some elements and leaves the rest as the constant
        had them. A step that reads and writes the address at once could
        depend on the old bytes, so that recording stays refused.
        """

        first_write = written.get(trace_id)
        if first_write is None or snapshot is None or length > limit:
            return None
        step = raw_steps[first_write]
        if any(int(buffer) == trace_id and access == "r" for _slot, buffer, _offset, access in step[4]):
            return None
        if len(step) > 13 and step[13] is not None and int(step[13][0]) == trace_id:
            return None
        return BufferIR(len(buffers), "const", length, data=bytes(snapshot))

    # ---- roles
    buffers: list[BufferIR] = []
    id_of: dict[int, int] = {}  # trace buffer -> program buffer id
    id_by_ptr: dict[int, int] = {}
    # trace buffer -> (first step that writes it, program buffer serving the reads before it)
    constant_before: dict[int, tuple[int, int]] = {}
    for trace_id in order:
        row = trace_buffers[trace_id]
        _tid, ptr, length, snapshot = row[:4]
        changed = bool(row[4]) if len(row) > 4 else False
        ptr = int(ptr)
        length = int(length)
        new_id = len(buffers)
        is_written = trace_id in written
        if ptr in input_by_ptr:
            members = input_by_ptr[ptr]
            if is_written and len(members) > 1:
                raise Refused(
                    "clobbered_shared_buffer",
                    f"inputs {members} share one buffer and the program writes into it",
                )
            buf = BufferIR(new_id, "input", length, index=members[0], clobbered=is_written)
        elif pin_slot(ptr) >= 0:
            if is_written:
                position = written[trace_id]
                raise Refused(
                    "pinned_written",
                    f"step {position} ({raw_steps[position][1]}) can write a pinned array",
                )
            buf = BufferIR(new_id, "pinned", length, index=pin_slot(ptr))
        elif ptr in output_by_ptr:
            if first_use[trace_id] == "r":
                constant = reused_constant(trace_id, snapshot, length)
                if constant is None:
                    raise Refused(
                        "undeclared_state",
                        f"output {output_by_ptr[ptr][0]} lives in a buffer that {where_first(trace_id)} "
                        "reads before anything writes it, and no declared input covers it",
                    )
                buffers.append(constant)
                constant_before[trace_id] = (written[trace_id], constant.id)
                new_id = len(buffers)
                # The elements the recording's writes leave alone keep the
                # constant's bytes, as in MLX's buffer (see reused_constant).
                buf = BufferIR(new_id, "output", length, index=output_by_ptr[ptr][0], data=constant.data)
            else:
                buf = BufferIR(new_id, "output", length, index=output_by_ptr[ptr][0])
        elif first_use[trace_id] == "r":
            if is_written:
                if snapshot is None or length > limit:
                    raise Refused(
                        "const_overwritten",
                        f"{where_first(trace_id)} reads an undeclared buffer of {length} bytes "
                        "that the program later writes",
                    )
                buf = BufferIR(new_id, "seed", length, data=bytes(snapshot))
            else:
                if changed:
                    # Nothing recorded writes this buffer, and yet it held
                    # other bytes at the end of the recording than at its
                    # first bind: the host rewrites it, or work outside the
                    # recording was still writing it.  Baked in, either set of
                    # bytes would be a wrong program that could pass a check.
                    raise Refused(
                        "const_changed",
                        f"{where_first(trace_id)} reads an undeclared buffer of {length} bytes "
                        "whose bytes changed during the recording though no recorded launch "
                        "writes it: declare it as an input or pin it",
                    )
                data = None
                if length <= limit:
                    data = bytes(snapshot) if snapshot is not None else read_buffer(trace_id, limit)
                if data is None:
                    slot = next(
                        (b[0] for b in raw_steps[first_step[trace_id]][4] if int(b[1]) == trace_id),
                        -1,
                    )
                    raise Refused(
                        "undeclared_external",
                        f"{where_first(trace_id)} reads a buffer of {length} bytes at argument "
                        f"{slot} (the array in it is no larger) that is neither a declared input "
                        "nor pinned "
                        f"(constants travel with the program up to {limit} bytes)",
                    )
                buf = BufferIR(new_id, "const", length, data=data)
        else:
            buf = BufferIR(new_id, "temp", length)
        buffers.append(buf)
        id_of[trace_id] = new_id
        id_by_ptr[ptr] = new_id

    # ---- declared arrays
    input_irs = [_array_ir(ref, id_by_ptr.get(ref.ptr, -1)) for ref in inputs]
    output_irs: list[ArrayIR] = []
    output_alias: dict[int, int] = {}
    for index, ref in enumerate(outputs):
        if not ref.ptr:
            output_irs.append(_array_ir(ref, -1))
            continue
        if ref.ptr in input_by_ptr:
            source = input_by_ptr[ref.ptr][0]
            relative = ref.offset - inputs[source].offset
            if ref.itemsize <= 0 or relative % ref.itemsize:
                raise Refused("output_view_offset", f"output {index} over input {source}")
            output_alias[index] = source
            output_irs.append(_array_ir(ref, id_by_ptr.get(ref.ptr, -1)))
            continue
        buffer_id = id_by_ptr.get(ref.ptr, -1)
        if buffer_id < 0 or buffers[buffer_id].role != "output":
            raise Refused(
                "output_not_computed",
                f"output {index} was not produced by a recorded launch (a constant, a weight, "
                "or work done on the CPU)",
            )
        first = output_by_ptr[ref.ptr][0]
        if first != index and (ref.offset - outputs[first].offset) % max(ref.itemsize, 1):
            raise Refused("output_view_offset", f"output {index} over output {first}")
        output_irs.append(_array_ir(ref, buffer_id))

    def program_buffer(trace_id: int, position: int) -> int:
        early = constant_before.get(trace_id)
        return early[1] if early is not None and position < early[0] else id_of[trace_id]

    # ---- steps
    steps: list[StepIR] = []
    for position, step in enumerate(raw_steps):
        grid = step[13] if len(step) > 13 else None
        threadgroup = tuple(int(v) for v in step[9])
        if grid is None and _INDIRECT_GRIDS:
            layout = _INDIRECT_GRIDS.get(str(step[1]).partition("__")[0])
            if layout is not None:
                slot, threadgroup = layout
                grid = next(((buf, offset) for idx, buf, offset, access in step[4]
                             if idx == slot and access == "r"), None)
                if grid is None:
                    raise Refused("indirect_grid_unbound", str(step[1]))
        steps.append(
            StepIR(
                kernel=int(step[0]),
                kernel_name=str(step[1]),
                bindings=[
                    (int(slot), program_buffer(int(buffer), position), int(offset), "r" if access == "r" else "w")
                    for slot, buffer, offset, access in step[4]
                ],
                inline=[(int(slot), bytes(data)) for slot, data in step[5]],
                dispatch=(
                    "indirect" if grid is not None else ("threads" if step[7] else "groups"),
                    (0, 0, 0) if grid is not None else tuple(int(v) for v in step[8]),
                    threadgroup,
                ),
                indirect_grid=(program_buffer(int(grid[0]), position), int(grid[1])) if grid is not None else None,
                barrier_before=bool(step[10]),
                mark=str(step[11]),
                tg_memory=[(int(slot), int(length)) for slot, length in step[6]],
            )
        )

    program = ProgramIR(
        name=name,
        buffers=buffers,
        steps=steps,
        inputs=input_irs,
        outputs=output_irs,
        output_alias=output_alias,
    )
    problems = program.check()
    if problems:
        raise Refused("malformed_program", "; ".join(problems[:4]))
    return program


def to_spec(ir: ProgramIR) -> tuple:
    """The program as the plain tuples the compiled module seals."""

    position = {buf.id: index for index, buf in enumerate(ir.buffers)}
    buffers = [
        (ROLE_CODE[buf.role], int(buf.nbytes), int(buf.index), bool(buf.clobbered), buf.data)
        for buf in ir.buffers
    ]
    steps = [
        (
            int(step.kernel),
            [
                (slot, position[buffer], offset, access != "r")
                for slot, buffer, offset, access in step.bindings
            ],
            [(slot, data) for slot, data in step.inline],
            [(slot, length) for slot, length in step.tg_memory],
            step.dispatch[0] == "threads",
            tuple(step.dispatch[1]),
            tuple(step.dispatch[2]),
            bool(step.barrier_before),
            (position[step.indirect_grid[0]], step.indirect_grid[1]) if step.indirect_grid is not None else None,
        )
        for step in ir.steps
    ]

    def array(arr: ArrayIR, alias: int | None) -> tuple:
        row = (
            position.get(arr.buffer, -1),
            int(arr.offset),
            tuple(arr.shape),
            arr.dtype,
            tuple(arr.strides),
            int(arr.data_size),
            int(arr.flags),
        )
        return row if alias is None else row + (alias,)

    inputs = [array(arr, None) for arr in ir.inputs]
    outputs = [
        array(arr, int(ir.output_alias.get(index, -1))) for index, arr in enumerate(ir.outputs)
    ]
    return (ir.name, buffers, steps, inputs, outputs)
