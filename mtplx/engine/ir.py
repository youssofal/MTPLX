"""ProgramIR: the shared data model of the native executor.

A program is what one evaluation of a fixed-shape function launched on the GPU,
written down as plain data: a list of buffers with roles, and a list of steps
(one kernel launch each) that bind those buffers.  Everything here is ordinary
Python so passes, tools and tests need nothing but this file.

Roles of a buffer:

``input``   belongs to one or more declared input arrays of the call
``output``  belongs to a declared output array and to no input; freshly
            allocated from MLX's allocator on every run
``pinned``  a long-lived array (a weight) bound where MLX holds it
``const``   read, never written, not declared: its bytes travel with the program
``seed``    like ``const`` but the program also writes into it, so its bytes
            are restored before every run
``temp``    first use is a write: working memory owned by the engine
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any

IR_VERSION = 1

ROLES = ("input", "output", "pinned", "const", "seed", "temp")

# Access of one binding, as Metal's reflection reports it for the kernel.
READ = "r"
WRITE = "w"  # write-only and read-write are both "w": a "w" binding may also read


@dataclass
class ArrayIR:
    """Where one declared array lives: a buffer of the program plus a view."""

    buffer: int  # BufferIR.id
    offset: int  # byte offset of the array's first element in that buffer
    shape: tuple[int, ...]
    dtype: str  # MLX dtype name, for example "bfloat16"
    strides: tuple[int, ...] = ()  # in elements, as MLX keeps them
    nbytes: int = 0  # size * itemsize
    data_size: int = 0  # elements spanned in the buffer, as MLX keeps it
    flags: int = 0  # 1 contiguous, 2 row contiguous, 4 column contiguous

    def to_obj(self) -> dict[str, Any]:
        return {
            "buffer": self.buffer,
            "offset": self.offset,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "strides": list(self.strides),
            "nbytes": self.nbytes,
            "data_size": self.data_size,
            "flags": self.flags,
        }

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "ArrayIR":
        return cls(
            buffer=int(obj["buffer"]),
            offset=int(obj["offset"]),
            shape=tuple(int(v) for v in obj["shape"]),
            dtype=str(obj["dtype"]),
            strides=tuple(int(v) for v in obj.get("strides", ())),
            nbytes=int(obj.get("nbytes", 0)),
            data_size=int(obj.get("data_size", 0)),
            flags=int(obj.get("flags", 0)),
        )


@dataclass
class BufferIR:
    id: int
    role: str
    nbytes: int  # length of the Metal buffer as recorded
    # input: index of the first declared input in this buffer; output: index of
    # the first declared output; pinned: slot in the engine's pin table.
    index: int = -1
    clobbered: bool = False  # an input the program writes into
    # const and seed: the bytes the buffer must hold when a run starts.
    data: bytes | None = None

    def to_obj(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "id": self.id,
            "role": self.role,
            "nbytes": self.nbytes,
            "index": self.index,
            "clobbered": self.clobbered,
        }
        if self.data is not None:
            obj["data"] = base64.b64encode(self.data).decode("ascii")
        return obj

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "BufferIR":
        data = obj.get("data")
        return cls(
            id=int(obj["id"]),
            role=str(obj["role"]),
            nbytes=int(obj["nbytes"]),
            index=int(obj.get("index", -1)),
            clobbered=bool(obj.get("clobbered", False)),
            data=base64.b64decode(data) if data is not None else None,
        )


# Bytes of an inline argument that are padding of the C struct MLX sends, not
# data. MLX fills the struct's fields and leaves the padding as the stack had
# it, so two recordings of one program can differ there by whatever ran before
# each. A comparison or a hash of recordings must not read those bytes (the
# kernel never does). One entry per struct, checked against the pinned MLX's
# header: (kernel name prefix, bytes of the argument, first padding byte, end).
#   mlx/backend/metal/kernels/steel/gemm/params.h, GEMMParams: eight ints,
#   three 64-bit strides and three ints are 68 bytes in a 72-byte struct.
_INLINE_PADDING = (("steel_gemm_fused", 72, 68, 72),)


def inline_data(kernel_name: str, data: bytes) -> bytes:
    """``data`` as a comparison or a hash should see it: known padding zeroed."""

    for prefix, length, start, stop in _INLINE_PADDING:
        if len(data) == length and kernel_name.startswith(prefix):
            return data[:start] + bytes(stop - start) + data[stop:]
    return data


@dataclass
class StepIR:
    kernel: int  # handle in the engine's kernel table (valid in this process)
    kernel_name: str  # the Metal function name
    # (binding index, buffer id, byte offset, access "r" | "w")
    bindings: list[tuple[int, int, int, str]]
    # (binding index, bytes sent inline)
    inline: list[tuple[int, bytes]]
    # ("groups" | "threads" | "indirect", (x, y, z), (tx, ty, tz))
    dispatch: tuple[str, tuple[int, int, int], tuple[int, int, int]]
    barrier_before: bool = False
    mark: str = ""
    # (binding index, bytes of threadgroup memory); MLX's own kernels use none
    tg_memory: list[tuple[int, int]] = field(default_factory=list)
    indirect_grid: tuple[int, int] | None = None  # buffer id and byte offset of three uint32 counts

    def accesses(self):
        """Shader arguments and the dispatch's own read, in signature order."""
        yield from sorted(self.bindings)
        if self.indirect_grid is not None:
            yield -1, *self.indirect_grid, READ

    def to_obj(self) -> dict[str, Any]:
        return {
            "kernel": self.kernel,
            "kernel_name": self.kernel_name,
            "bindings": [list(b) for b in self.bindings],
            "inline": [
                [index, base64.b64encode(data).decode("ascii")]
                for index, data in self.inline
            ],
            "dispatch": [self.dispatch[0], list(self.dispatch[1]), list(self.dispatch[2])],
            **({"indirect_grid": list(self.indirect_grid)} if self.indirect_grid is not None else {}),
            "barrier_before": self.barrier_before,
            "mark": self.mark,
            "tg_memory": [list(t) for t in self.tg_memory],
        }

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "StepIR":
        kind, grid, group = obj["dispatch"]
        return cls(
            kernel=int(obj["kernel"]),
            kernel_name=str(obj["kernel_name"]),
            bindings=[(int(b[0]), int(b[1]), int(b[2]), str(b[3])) for b in obj["bindings"]],
            inline=[(int(i), base64.b64decode(d)) for i, d in obj["inline"]],
            dispatch=(
                str(kind),
                (int(grid[0]), int(grid[1]), int(grid[2])),
                (int(group[0]), int(group[1]), int(group[2])),
            ),
            indirect_grid=tuple(map(int, obj["indirect_grid"])) if "indirect_grid" in obj else None,
            barrier_before=bool(obj.get("barrier_before", False)),
            mark=str(obj.get("mark", "")),
            tg_memory=[(int(t[0]), int(t[1])) for t in obj.get("tg_memory", [])],
        )


@dataclass
class ProgramIR:
    name: str
    buffers: list[BufferIR]
    steps: list[StepIR]
    inputs: list[ArrayIR] = field(default_factory=list)
    outputs: list[ArrayIR] = field(default_factory=list)
    # output index -> index of the input whose data it shares (written in place
    # or passed through).  The output is returned as an array over that data.
    output_alias: dict[int, int] = field(default_factory=dict)
    version: int = IR_VERSION

    # ----------------------------------------------------------------- lookup

    def buffer(self, buffer_id: int) -> BufferIR:
        for buf in self.buffers:
            if buf.id == buffer_id:
                return buf
        raise KeyError(buffer_id)

    def launches(self) -> int:
        return len(self.steps)

    def kernel_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for step in self.steps:
            counts[step.kernel_name] = counts.get(step.kernel_name, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def bytes_by_role(self) -> dict[str, int]:
        totals = {role: 0 for role in ROLES}
        for buf in self.buffers:
            totals[buf.role] = totals.get(buf.role, 0) + buf.nbytes
        return totals

    def input_layout(self) -> tuple:
        """Shape and strides of every input exactly as recorded.

        MLX leaves an arbitrary stride on a dimension of extent 1, and some of
        its kernels are handed that stride in their arguments.  Two recordings
        of one geometry therefore hash the same only when this tuple is the
        same for both; compare signatures only then.  (A sealed program serves
        every such layout: the unused stride never addresses an element.)
        """

        return tuple((arr.dtype, tuple(arr.shape), tuple(arr.strides)) for arr in self.inputs)

    # -------------------------------------------------------------- signature

    def signature(self) -> str:
        """Structure hash, equal for two recordings of one geometry.

        MLX's allocator decides which physical buffer a temporary lands in, and
        that can differ between two evaluations of the same graph.  So the hash
        names a temporary by the step that last wrote it (its definition), not
        by its buffer.  Declared inputs, pinned and constant buffers are named
        by what they are; a declared output by the launch that last wrote it.
        Kernel names, binding slots, offsets, inline bytes, grids and barriers
        all count.
        """

        digest = hashlib.blake2b(digest_size=16)

        def feed(*parts: Any) -> None:
            for part in parts:
                if isinstance(part, bytes):
                    digest.update(struct.pack("<q", len(part)))
                    digest.update(part)
                elif isinstance(part, str):
                    raw = part.encode("utf-8")
                    digest.update(struct.pack("<q", len(raw)))
                    digest.update(raw)
                else:
                    digest.update(struct.pack("<q", int(part)))

        by_id = {buf.id: buf for buf in self.buffers}
        # An input may sit at a different offset of its buffer in another
        # recording, so offsets inside an input's buffer are taken relative to
        # the first declared input in it.
        base: dict[int, int] = {}
        for buf in self.buffers:
            if buf.role == "input" and 0 <= buf.index < len(self.inputs):
                base[buf.id] = self.inputs[buf.index].offset
        def layout(arr: ArrayIR) -> list[int]:
            # The stride of a dimension of extent 1 addresses nothing, and MLX
            # leaves whatever the array's creation route put there.
            return [
                stride if extent > 1 else 0
                for extent, stride in zip(arr.shape, arr.strides)
            ]

        feed("ir", self.version, len(self.steps), len(self.inputs), len(self.outputs))
        for arr in self.inputs:
            feed("in", arr.dtype, len(arr.shape), *arr.shape, *layout(arr))
            feed(by_id[arr.buffer].index if arr.buffer in by_id else -1)
            feed(arr.offset - base.get(arr.buffer, arr.offset))
        for index, arr in enumerate(self.outputs):
            feed("out", arr.dtype, len(arr.shape), *arr.shape, *layout(arr))
            feed(self.output_alias.get(index, -1))

        def stable_name(buf: BufferIR) -> tuple[Any, ...]:
            if buf.role == "input":
                return ("input", buf.index, int(buf.clobbered))
            if buf.role == "pinned":
                return ("pinned", buf.index)
            content = hashlib.blake2b(buf.data or b"", digest_size=8).digest()
            return (buf.role, content)

        # A read of working memory is named by how far back its producer is.
        # A write is not named at all: which physical buffer receives a fresh
        # result is the allocator's choice, and a kernel that updates in place
        # is already pinned down by its own reads.
        last_writer: dict[int, int] = {}
        for position, step in enumerate(self.steps):
            feed("step", step.kernel_name, int(step.barrier_before), step.mark)
            feed(step.dispatch[0], *step.dispatch[1], *step.dispatch[2])
            for index, data in sorted(step.inline):
                feed("inline", index, inline_data(step.kernel_name, data))
            for index, length in sorted(step.tg_memory):
                feed("tg", index, length)
            for index, buffer_id, offset, access in step.accesses():
                buf = by_id[buffer_id]
                feed("bind", index, access)
                if buf.role in ("temp", "output"):
                    # MLX recycles buffers inside one evaluation: the buffer
                    # that ends as a declared output may be working memory
                    # earlier, in one recording and not in another.  So both
                    # roles are named alike here, by the launch that made
                    # what is read; which launch feeds which declared output
                    # is hashed once, below.
                    if access == READ:
                        feed("made", offset, position - last_writer.get(buffer_id, position + 1))
                    else:
                        feed("made", offset)
                elif buf.role == "seed":
                    feed(offset, *stable_name(buf))
                    if access == READ:
                        feed(position - last_writer.get(buffer_id, position + 1))
                else:
                    feed(offset - base.get(buffer_id, 0), *stable_name(buf))
            for _index, buffer_id, _offset, access in step.bindings:
                if access == WRITE:
                    last_writer[buffer_id] = position
        for index, arr in enumerate(self.outputs):
            writer = last_writer.get(arr.buffer)
            feed(
                "out-from",
                index,
                -1 if writer is None else len(self.steps) - writer,
                arr.offset - base.get(arr.buffer, 0),
            )
        return digest.hexdigest()

    # ------------------------------------------------------------------- JSON

    def to_obj(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "name": self.name,
            "signature": self.signature(),
            "buffers": [buf.to_obj() for buf in self.buffers],
            "steps": [step.to_obj() for step in self.steps],
            "inputs": [arr.to_obj() for arr in self.inputs],
            "outputs": [arr.to_obj() for arr in self.outputs],
            "output_alias": {str(k): v for k, v in sorted(self.output_alias.items())},
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_obj(), indent=indent)

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "ProgramIR":
        return cls(
            name=str(obj.get("name", "")),
            buffers=[BufferIR.from_obj(b) for b in obj["buffers"]],
            steps=[StepIR.from_obj(s) for s in obj["steps"]],
            inputs=[ArrayIR.from_obj(a) for a in obj.get("inputs", [])],
            outputs=[ArrayIR.from_obj(a) for a in obj.get("outputs", [])],
            output_alias={int(k): int(v) for k, v in obj.get("output_alias", {}).items()},
            version=int(obj.get("version", IR_VERSION)),
        )

    @classmethod
    def from_json(cls, text: str) -> "ProgramIR":
        return cls.from_obj(json.loads(text))

    # ------------------------------------------------------------- validation

    def check(self) -> list[str]:
        """Structural problems, empty when the program is well formed."""

        problems: list[str] = []
        ids = [buf.id for buf in self.buffers]
        if len(set(ids)) != len(ids):
            problems.append("duplicate buffer ids")
        by_id = {buf.id: buf for buf in self.buffers}
        for buf in self.buffers:
            if buf.role not in ROLES:
                problems.append(f"buffer {buf.id}: unknown role {buf.role!r}")
            if buf.role in ("const", "seed") and buf.data is None:
                problems.append(f"buffer {buf.id}: {buf.role} without bytes")
            if buf.data is not None and len(buf.data) > buf.nbytes:
                problems.append(f"buffer {buf.id}: bytes longer than the buffer")
        written: set[int] = set()
        for position, step in enumerate(self.steps):
            if (step.dispatch[0] == "indirect") != (step.indirect_grid is not None):
                problems.append(f"step {position}: indirect dispatch needs its grid address")
            if step.indirect_grid is not None:
                grid_buffer, grid_offset = step.indirect_grid
                desc = by_id.get(grid_buffer)
                if desc is None or grid_offset % 4 or not 0 <= grid_offset <= desc.nbytes - 12:
                    problems.append(f"step {position}: indirect grid outside its buffer or unaligned")
            if step.dispatch[0] not in ("groups", "threads", "indirect"):
                problems.append(f"step {position}: unknown dispatch {step.dispatch[0]!r}")
            slots = [b[0] for b in step.bindings] + [i for i, _ in step.inline]
            if len(set(slots)) != len(slots):
                problems.append(f"step {position}: a binding slot is used twice")
            for index, buffer_id, offset, access in step.accesses():
                buf = by_id.get(buffer_id)
                if buf is None:
                    problems.append(f"step {position}: unknown buffer {buffer_id}")
                    continue
                if access not in (READ, WRITE):
                    problems.append(f"step {position}: unknown access {access!r}")
                if not 0 <= offset <= buf.nbytes:
                    problems.append(f"step {position}: offset outside buffer {buffer_id}")
                if access == READ and buf.role == "temp" and buffer_id not in written:
                    problems.append(
                        f"step {position} ({step.kernel_name}): reads temporary "
                        f"{buffer_id} before anything wrote it"
                    )
                if access == WRITE and buf.role in ("pinned", "const"):
                    problems.append(f"step {position}: writes {buf.role} buffer {buffer_id}")
            for _index, buffer_id, _offset, access in step.bindings:
                if access == WRITE:
                    written.add(buffer_id)
        for which, arrays in (("input", self.inputs), ("output", self.outputs)):
            for index, arr in enumerate(arrays):
                # -1: an array no launch binds (an unused input, a pure view)
                if arr.buffer != -1 and arr.buffer not in by_id:
                    problems.append(f"{which} {index}: unknown buffer {arr.buffer}")
        for out_index, in_index in self.output_alias.items():
            if not 0 <= out_index < len(self.outputs) or not 0 <= in_index < len(self.inputs):
                problems.append(f"output_alias {out_index} -> {in_index} is out of range")
        return problems
