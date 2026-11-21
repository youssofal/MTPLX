"""ReplaySite: a place in the product path where a fixed-shape function of
arrays is evaluated every round, and where that evaluation can leave MLX's
graph and be replayed by the native executor.

The MLX path stays the reference and the fallback. For each geometry key the
site walks one way:

    first calls         the MLX path serves; nothing is recorded (the first
                        call traces the graph and compiles its kernels, the
                        next runs it once settled: ``WARM_CALLS``)
    a later call        the MLX path serves and is recorded
    the call after      the same, until the recording is known to be the
                        steady program: seen twice (MLX copies a state bank
                        whole when something still holds it, so one recording
                        can carry launches the steady path does not make;
                        compared when both were made at one input layout, see
                        ``_one_program``), or launching exactly what a
                        program of the same step that is already live
                        launches (``_steady_elsewhere``: the step at another
                        bank capacity).  The call that settles it also seals
                        the program (its working memory and constants; dumped
                        when a dump folder is set), after its own outputs are
                        out.
    the next call       the program is replayed on private copies of every
                        input it writes into, the MLX path serves, and every
                        output of the replay is compared bit for bit with the
                        MLX path's
    from then on        the program serves; the MLX function is not called.

Every one of those calls is a round the MLX path serves, at about twice a
replayed round's time on the 35B-A3B (19.5 ms against 10.6 ms a verify
forward, N/rows/kernelw5-1117-0739, the image cell), so there are as few of
them as the bit check allows: for a step that is live at another capacity,
one recording and the checked call.  None of them is a request's first round
(``quiet_start``).

A key is the caller's geometry plus the layout of the inputs (type, shape and
the strides that address elements): the same state arriving as a view of a
longer buffer makes MLX launch other kernels, so it is another program.

Every call that does not replay is counted with a reason. A program that fails
any check is refused for its key, with the reason, and the MLX path keeps
serving that key.

The caller keeps its own control flow. ``begin`` returns a ticket; when the
ticket is native its ``outputs`` are the round's outputs, otherwise the caller
runs the MLX function exactly as before and hands the lazy outputs to
``finish`` after it has installed them (so MLX can still write its state in
place during a recording).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict, deque
from typing import Any, Callable

import mlx.core as mx

from . import runtime as _runtime
from .ir import ProgramIR, inline_data

# Input, axis (or a pair of adjacent axes: rows of a physical page bank),
# offset input, rows, and optionally the corresponding state output.
StateWindow = (tuple[int, int | tuple[int, int], int, int]
               | tuple[int, int | tuple[int, int], int, int, int])

REPLAY_ENV = "MTPLX_NATIVE_ENGINE_REPLAY"  # "0": record, build and dump; never replay
# "1": per-step GPU times, once per key, into the dump folder; "forms": that,
# the parameter names of the pinned buffers, and every dispatch form timed
PROFILE_ENV = "MTPLX_NATIVE_ENGINE_PROFILE"
MIXED_ENV = "MTPLX_NATIVE_ENGINE_MIXED"  # QA: "<seed>" serves live rounds from either path at random
PROGRAMS_ENV = "MTPLX_NATIVE_ENGINE_PROGRAMS"  # most programs a site keeps, whatever their size
PROGRAM_BYTES_ENV = "MTPLX_NATIVE_ENGINE_PROGRAM_MB"  # memory all of a site's programs may own, in MiB
QUIET_ENV = "MTPLX_NATIVE_ENGINE_QUIET_TOKENS"  # tokens of an answer before a recording or a check (tests: 0)
DUMP_DATA_ENV = "MTPLX_NATIVE_ENGINE_DUMP_DATA"  # "1": keep constant bytes in the dumped IR
WARM_ENV = "MTPLX_NATIVE_ENGINE_WARM_CALLS"  # measurement: calls of a key MLX serves before one is recorded
DIAGNOSE_ENV = "MTPLX_NATIVE_ENGINE_DIAGNOSE"  # "1" with a dump folder: on a mismatch, name the launch (slow; one run)

# Calls of a key the MLX path serves before one is recorded: the call that
# traces the graph and compiles its kernels, and one settled call after it.
# It was 24, so that a key showed it was long-lived before anything was spent
# on it, from the days a replayed round saved a millisecond or two.  On the
# 35B-A3B a round the MLX path serves now costs 9 ms more than a replayed one
# (19.5 ms against 10.6 ms a verify forward), and the first request of a boot
# and the first image turn each ran 24 to 30 of their rounds on MLX for it
# (N/rows/kernelw5-1117-0739: the first request 24 of 230 rounds, first 256
# tokens at 106 tok/s and the last 256 at 166; the image turn 30 of 106).
# What a new program costs on top of rounds MLX would serve anyway is small
# beside that: in the same rows a recorded call took 20.7 ms with its forward
# (``record_s`` over two) and the checked call 29.1 ms with both of its
# forwards (``verify_s``), so it is paid back within a handful of rounds, and
# the program then serves every later turn at that capacity.  A turn's first
# round still comes before any of it (``quiet``).
WARM_CALLS = 2
# Tokens of an answer before a site may record a program or make a first-use
# check on one of its calls. Either costs a few rounds' time once per geometry,
# so neither ever falls before a request's first text or inside a short answer
# (64 tokens and a margin). A program that is already live serves from the
# first round: that is what a geometry that recurs buys. A geometry of a
# family that has been live waits ``_FAMILY_QUIET_TOKENS`` instead.
QUIET_TOKENS = 96
_DEFAULT_PROGRAMS = 96
# What a program owns is small: its constants on the GPU (0.5 MB for the
# Flash-Next verify step), the same bytes again in the recording kept beside
# it, and the host memory of its launches (``_owned_bytes``): about 4 MB for
# the 1,241 launches of that step. Its temporaries are not its own (the
# executor serves every program's from one pool, as large as the largest
# program needs: 8 to 78 MiB for that step, the executor's ``pool_bytes``),
# and its outputs are allocated per run: they are the round's outputs, which
# the MLX path allocates as well. The budget is what a site may add to the
# process for all its programs; the least recently used go first.
_DEFAULT_PROGRAM_BYTES = 256 << 20
# Host memory of one launch: 1,750 bytes in the recording (measured 10 Nov on
# a dumped Flash-Next verify program: 2.18 MB for 1,241 launches, without its
# constants' bytes), and an allowance for the sealed program's own tables.
_LAUNCH_HOST_BYTES = 2200
_REMEMBERED_REFUSALS = 256
_TIMING_SAMPLES = 8192  # native calls whose times are kept for the percentiles
# One native call, start to end:
#   pre          the caller before it asks the site (window reservation, the
#                auxiliary rows, binding the program)
#   inputs       evaluating what is still pending in the inputs (a commit
#                after a rejection), and the layout key
#   site         the rest of this class and its caller around the executor
#   engine_wall  the executor's clock around one replay; inside it:
#     encode       host time writing the launches (the GPU has nothing yet)
#     gpu          the GPU's time for the command buffer
#     wait         engine_wall - encode - gpu: the commit, the wake-up, and
#                  any wait for an input MLX had not let go of
#   post         from the outputs' return to the caller's last word
#                (installing the state leaves and captures)
#   whole        all of it: what the round pays for its verify forward
_NATIVE_COLUMNS = ("engine_wall", "gpu", "encode", "wait", "pre", "inputs", "site", "post", "whole")
_CHECK_COLUMNS = ("engine_wall", "gpu", "encode")
_CHECK_RETRIES = 3  # checks that could not run (not this call) before a key is refused
_FENCED_RUN = 4  # planned calls in a row at a fenced layout before the geometry is left to MLX
_FENCED_RETRY = 64  # ... and then one call in this many is taken again
# Checks repeated because one recording had more launches than the other. A
# refusal is for the life of the process and sends every round of the key to
# MLX, so the budget is patient: on the real pack a key whose steady program
# was its first recording met three longer ones in a row (bank copies come and
# go with what still holds a bank) and was refused at three. A recording costs
# one round on the MLX path.
_RECHECKS = 8
# When that budget is spent without the shorter program being seen again, the
# key rests: MLX serves its next ``_REST_CALLS`` calls unrecorded, then the
# budget starts over, ``_RESTS`` times before the key is refused. Whatever
# held a bank through eight rounds in a row is usually gone a few dozen
# rounds later, and a refusal costs every later round of the key.
_REST_CALLS = 64
_RESTS = 3
# Warm calls of a geometry whose family already served a live program in this
# process (``plan``'s ``family``: the same step at another bank capacity). The
# warm calls are there to see a key live long before it is recorded; a family
# that has been live has shown that, and its next geometry is a capacity the
# session grew into and stays in for thousands of tokens. In the founder's Pi
# session of 12 November the bank crossed a capacity edge (every 8,192 rows
# past 16K) a dozen times between 57K and 164K, and each crossing cost every
# verify width its 24 warm calls again: 823 of his 49,853 rounds ran on MLX as
# warm calls. A copy window meets a bucket's edge with 80 to 380 copy rounds per
# bucket spread over four widths, so its warm calls would be a large part of
# them. The one call left is the one that grew the bank.
_FAMILY_WARM_CALLS = 1
# ... and the tokens of an answer such a geometry waits before it is recorded
# (``ReplaySite.quiet_start``). ``QUIET_TOKENS`` keeps a short answer from paying
# for a step that may never run long; a family that has been live has run
# long, and with that wait a conversation of short tool turns never got a
# program for the capacity it was in: in the founder's OpenCode session of 12
# November 468 rounds waited out their answers' first tokens on MLX, at about
# twice a native round's time past 100K tokens. It was 8 tokens, three rounds
# of every request that met a new capacity (N/rows/kernelw5-1117-0739, the 4K
# cell: ``request_start`` 3 of the 12 rounds that were not copy windows). A
# request's first token is out before its first verify round ends (that row:
# first text at 0.747 s, 0.730 s of it the prompt), and a round that is
# recorded is a round MLX serves either way, so the wait bought nothing but
# those rounds. What is left is the request's first round, whose inputs
# arrive as the prefill left them (see ``STRIDED_INPUTS``): the answer has no
# token of a verify round yet, and from its second round it has one.
_FAMILY_QUIET_TOKENS = 1
_REMEMBERED_FAMILIES = 64
_TRUTHY = {"1", "true", "yes", "on"}

# Fallback reason codes (ENGINE-CONTRACT.md section 4).
WARMING = "warming"
KEY_NEW = "key_new"
RECORDING = "recording"
VERIFYING = "verifying"
DISABLED = "disabled"
UNAVAILABLE = "unavailable"
RECORD_ONLY = "record_only"
SHADOW = "shadow"
MIXED = "mixed"
VERIFY_MEMORY = "verify_memory"
REQUEST_START = "request_start"  # a recording or a check put off: the request's first round
RESTING = "resting"  # between two rounds of recordings that differed in length (see _REST_CALLS)
LONG_WIDTH = "long_width"  # a rare verify width at a 64K view, kept on MLX by its caller (``refuse``)
# A FENCE (10 Nov, 06:25), not a design: a program recorded on a call whose
# inputs are strided views (a request's first verify round: the K and V banks
# arrive as views of the longer buffers the prefill left) replayed wrong on
# the next such call (bit mismatch at first use, fail-closed) and, three times
# in eight runs, never returned from the GPU. The MLX path's own result on
# those calls is a pure function of its inputs (checked), so the fault is on
# the replay's side and is agent A's to find. Until it is, such a call is one
# MLX serves: it is one round a request, counted under this reason.
STRIDED_INPUTS = "strided_inputs"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def _profile_mode() -> str:
    """``""`` (off), ``"steps"`` (per-launch GPU times) or ``"forms"`` (also
    the pinned buffers' names and every dispatch form timed: ``_time_forms``)."""

    raw = os.environ.get(PROFILE_ENV, "").strip().lower()
    if raw == "forms":
        return "forms"
    return "steps" if raw in _TRUTHY else ""


# The forms a census times (``_time_forms``), as (dispatch, barriers, commit
# every N launches). The serving default first; then the commit split, the
# barrier placements and the dispatch orders around it; last the program with
# no barrier at all (wrong results, dropped): the floor fusing could reach.
# Each effect is read with the other two settings held: barriers =
# concurrent/minimal/256 - concurrent/none/256, dispatch = hybrid/minimal/256
# - concurrent/minimal/256, split = hybrid/minimal/256 - hybrid/minimal/0
# (``census.effects``).
_CENSUS_FORMS = (
    ("hybrid", "minimal", 256),
    ("hybrid", "minimal", 0),
    ("hybrid", "minimal", 64),
    ("hybrid", "minimal", 128),
    ("hybrid", "minimal", 512),
    ("hybrid", "recorded", 256),
    ("hybrid", "named", 256),
    ("hybrid", "passes", 256),
    ("concurrent", "recorded", 0),
    ("concurrent", "recorded", 256),
    ("concurrent", "minimal", 0),
    ("concurrent", "minimal", 256),
    ("concurrent", "named", 256),
    ("concurrent", "passes", 256),
    ("serial", "minimal", 0),
    ("serial", "minimal", 256),
    ("concurrent", "none", 0),
    ("concurrent", "none", 256),
)
_CENSUS_REPS = 5  # timed replays per form, after one untimed


def load_early() -> None:
    """Load the compiled module now, unless the switch is off. Never raises.

    The module learns each Metal kernel's name and which arguments it writes
    when MLX compiles the kernel; a kernel compiled before the module was
    loaded is unknown to it and refuses every program that launches it. So
    the model loader and the server call this before anything runs on the GPU.
    """

    try:
        if _runtime.mode() != "0":
            _runtime.available()
    except Exception:  # noqa: BLE001 - loading the engine must never stop a model load
        pass


def quiet_start(tokens: int, limit: int = QUIET_TOKENS) -> bool:
    """True while an answer of ``tokens`` tokens is too short to record or
    check on: shorter than ``limit``, or than ``QUIET_ENV`` when that is set."""

    raw = os.environ.get(QUIET_ENV, "").strip()
    try:
        limit = max(0, int(raw)) if raw else limit
    except ValueError:
        pass
    return max(0, int(tokens)) < limit


def replay_allowed() -> bool:
    return os.environ.get(REPLAY_ENV, "").strip().lower() not in {"0", "off", "false", "no"}


def _bits(array: mx.array) -> mx.array:
    """The array's bytes as integers, so a comparison sees NaNs and signed zeros."""

    dtype = array.dtype
    if array.ndim == 0:
        array = array.reshape(1)
    if dtype in (mx.bfloat16, mx.float16):
        return array.view(mx.uint16)
    if dtype == mx.float32:
        return array.view(mx.uint32)
    if dtype == mx.float64:
        return array.view(mx.uint64)
    if dtype == mx.complex64:
        return array.view(mx.uint32)
    return array


def private_copy(array: mx.array) -> mx.array:
    """The same bytes in a buffer nothing else holds.

    Through the integer view, so no float arithmetic touches a value (adding
    zero would turn -0.0 into +0.0). The caller evaluates the result.
    """

    bits = _bits(array)
    if bits.dtype == mx.bool_:
        copied = mx.logical_or(bits, mx.array(False))
    else:
        copied = mx.bitwise_or(bits, mx.array(0, dtype=bits.dtype))
    if copied.dtype != array.dtype:
        copied = copied.view(array.dtype)
    return copied.reshape(array.shape)


def compare_outputs(candidate: list, reference: list) -> dict[str, Any] | None:
    """None when every output is bit-identical, else the first difference."""

    if len(candidate) != len(reference):
        return {"output": -1, "why": f"count {len(candidate)} != {len(reference)}"}
    flags = []
    for index, (got, want) in enumerate(zip(candidate, reference)):
        if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
            return {
                "output": index,
                "why": f"shape or dtype {tuple(got.shape)} {got.dtype} != "
                f"{tuple(want.shape)} {want.dtype}",
            }
        flags.append(mx.all(_bits(got) == _bits(want)))
    if not flags:
        return None
    verdict = mx.stack(flags)
    mx.eval(verdict)
    equal = verdict.tolist()
    if all(equal):
        return None
    index = equal.index(False)
    got, want = candidate[index], reference[index]
    differing = mx.sum(_bits(got) != _bits(want))
    detail: dict[str, Any] = {
        "output": index,
        "outputs_differing": int(len(equal) - sum(equal)),
        "elements": int(got.size),
        "why": "bits",
    }
    if mx.issubdtype(got.dtype, mx.floating):
        diff = mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32)))
        mx.eval(differing, diff)
        detail["max_abs_diff"] = float(diff.item())
    else:
        mx.eval(differing)
    detail["elements_differing"] = int(differing.item())
    return detail


def describe_differences(candidate: list, reference: list, limit: int = 64) -> list[dict[str, Any]]:
    """Every output that differs, with where: for whoever debugs a mismatch."""

    rows: list[dict[str, Any]] = []
    for index, (got, want) in enumerate(zip(candidate, reference)):
        if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
            rows.append({"output": index, "why": "shape or dtype",
                         "got": [list(got.shape), str(got.dtype)], "want": [list(want.shape), str(want.dtype)]})
            continue
        unequal = (_bits(got) != _bits(want)).reshape(-1)
        count = int(mx.sum(unequal).item())
        if not count:
            continue
        where = mx.argmax(unequal).item()
        last = int(unequal.size) - 1 - int(mx.argmax(unequal[::-1]).item())
        row: dict[str, Any] = {
            "output": index, "shape": list(got.shape), "dtype": str(got.dtype),
            "elements": int(got.size), "elements_differing": count,
            "first_flat_index": int(where), "last_flat_index": last,
        }
        if got.ndim >= 2 and got.size:
            # Which rows of the axis before the last hold a difference.
            per_row = mx.any((_bits(got) != _bits(want)).reshape(-1, got.shape[-1]), axis=-1)
            rows_hit = [i for i, hit in enumerate(per_row.tolist()) if hit]
            row["rows_differing"] = len(rows_hit)
            row["rows_first"] = rows_hit[:16]
        if mx.issubdtype(got.dtype, mx.floating):
            row["max_abs_diff"] = float(mx.max(mx.abs(got.astype(mx.float32) - want.astype(mx.float32))).item())
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _free_strides(ir: ProgramIR) -> set[int]:
    """Stride values that address nothing: those of extent-1 dimensions."""

    free: set[int] = set()
    for arr in (*ir.inputs, *ir.outputs):
        for extent, stride in zip(arr.shape, arr.strides or ()):
            if int(extent) == 1:
                free.add(int(stride))
    return free


def _layout(arr: Any) -> tuple:
    kept = tuple(
        int(stride) for extent, stride in zip(arr.shape, arr.strides or ()) if int(extent) > 1
    )
    return (arr.dtype, tuple(arr.shape), kept)


def _input_bases(ir: ProgramIR) -> dict[int, int]:
    bases: dict[int, int] = {}
    for arr in ir.inputs:
        if arr.buffer >= 0:
            bases[arr.buffer] = min(bases.get(arr.buffer, arr.offset), arr.offset)
    return bases


def launch_difference(first: ProgramIR, second: ProgramIR, limit: int = 4) -> str:
    """Which kernels two recordings of one geometry launch a different number
    of times, for whoever reads a recheck or a refusal: ``"copy_x 24 more"``.

    Named for the second recording against the first. Empty when the two
    launch every kernel equally often (they then differ in order or operands,
    which ``compare_structure`` says).
    """

    was, now = first.kernel_counts(), second.kernel_counts()
    parts = []
    for name in sorted(set(was) | set(now), key=lambda n: (-abs(now.get(n, 0) - was.get(n, 0)), n)):
        delta = now.get(name, 0) - was.get(name, 0)
        if delta:
            parts.append(f"{name} {abs(delta)} {'more' if delta > 0 else 'fewer'}")
    shown = parts[: max(1, int(limit))]
    if len(parts) > len(shown):
        shown.append(f"and {len(parts) - len(shown)} other kernels")
    return ", ".join(shown)


# MLX's copy kernels by name (``v_copy``, ``g1_copy``, ``gg2_dynamic_copy`` and
# their type suffixes): what MLX launches when it copies an array, either one
# it cannot write in place because something else still holds it or one that
# is not laid out as the next kernel needs it.
_COPY_KERNEL = re.compile(r"^[a-z]+\d*_(?:dynamic_)?copy")


def copies_only(shorter: ProgramIR, longer: ProgramIR) -> bool:
    """True when ``longer`` is ``shorter`` plus some of MLX's copy kernels and nothing else,
    proven structurally: with the extra copies removed and every later use of a copy's
    destination read as its source, the remaining launches are ``shorter``'s by
    ``compare_structure`` (kernel, grid, barrier, inline bytes, binding offsets, what every
    read holds, constants by their bytes). A pair that differs by a copy and also by anything
    else (an operand frozen into one recording) is not one program.

    MLX writes an array in place only when nothing else holds it at the moment the write is
    encoded. Inside one evaluation it commits a command buffer every few dozen launches, and a
    committed buffer holds the arrays it used until the GPU has run it; so whether a later
    write into the same bank finds it free depends on how far the GPU has got, and a recording
    can carry a whole copy of a bank that another recording of the same call writes in place.
    Such a pair is one program: the shorter is it (MLX's own in-place writes), and its
    first-use bit check stays the proof of its bits.
    """

    decopied = without_copies(shorter, longer)
    if decopied is None or compare_structure(shorter, decopied, same_bytes=True) is not None:
        return False
    # compare_structure places fresh results freely (the allocator's business); a write into a
    # declared input is the program's own and must land at the same place in both.
    return _input_writes(shorter) == _input_writes(decopied)


def _input_writes(ir: ProgramIR) -> list:
    """Per launch, its writes into declared inputs: (slot, input index, offset from the input)."""
    by_id = {buf.id: buf for buf in ir.buffers}
    bases = _input_bases(ir)
    rows = []
    for step in ir.steps:
        rows.append(sorted(
            (slot, by_id[b].index, offset - bases.get(b, 0))
            for slot, b, offset, access in step.bindings
            if access != "r" and b in by_id and by_id[b].role == "input"
        ))
    return rows


def without_copies(shorter: ProgramIR, longer: ProgramIR) -> ProgramIR | None:
    """``longer`` with the copy launches ``shorter`` does not make removed and each copy's
    destination read as its source from then on; None when the two do not align that way
    (a launch other than one of MLX's copies is extra, or a copy is not one read and one
    write)."""

    import dataclasses

    if len(longer.steps) <= len(shorter.steps):
        return None
    extras = set()
    position = 0
    for index, step in enumerate(longer.steps):
        if position < len(shorter.steps) and step.kernel_name == shorter.steps[position].kernel_name:
            position += 1
            continue
        if not _COPY_KERNEL.match(step.kernel_name or ""):
            return None
        extras.add(index)
    if position != len(shorter.steps):
        return None
    moved: dict[int, tuple[int, int]] = {}  # destination buffer -> (source buffer, offset shift)

    def follow(buffer_id: int, offset: int) -> tuple[int, int]:
        if buffer_id in moved:
            source, shift = moved[buffer_id]
            return source, offset + shift
        return buffer_id, offset

    steps = []
    for index, step in enumerate(longer.steps):
        if index in extras:
            reads = [(b, o) for _slot, b, o, access in step.bindings if access == "r"]
            writes = [(b, o) for _slot, b, o, access in step.bindings if access != "r"]
            if len(reads) != 1 or len(writes) != 1 or step.indirect_grid is not None:
                return None
            source, source_offset = follow(*reads[0])
            destination, destination_offset = writes[0]
            moved[destination] = (source, source_offset - destination_offset)
            continue
        bindings = [(slot, *follow(b, o), access) for slot, b, o, access in step.bindings]
        grid = None if step.indirect_grid is None else follow(*step.indirect_grid)
        steps.append(dataclasses.replace(step, bindings=bindings, indirect_grid=grid))
    outputs = []
    for array in longer.outputs:
        buffer_id, offset = follow(array.buffer, array.offset)
        outputs.append(dataclasses.replace(array, buffer=buffer_id, offset=offset))
    return dataclasses.replace(longer, steps=steps, outputs=outputs)


def copy_moved(first: ProgramIR, second: ProgramIR) -> str | None:
    """Where two recordings of one length first launch different kernels, when
    one of the two kernels there is one of MLX's copies; else None.

    A recording carries a copy the steady program does not make when an array
    it writes was still held by something else at that moment. Two such
    recordings can be of one length and still differ: each copied another
    array. Kernels launched the same way up to that point are the same
    program up to it, so the first differing kernel name tells the two cases
    apart: a copy on either side is that transient, anything else is not.
    """

    for position, (a, b) in enumerate(zip(first.steps, second.steps)):
        if a.kernel_name == b.kernel_name:
            continue
        if _COPY_KERNEL.match(a.kernel_name or "") or _COPY_KERNEL.match(b.kernel_name or ""):
            return f"launch {position}: {a.kernel_name} against {b.kernel_name}"
        return None
    return None


def compare_structure(first: ProgramIR, second: ProgramIR, *, same_bytes: bool = False) -> str | None:
    """None when two recordings of one geometry are the same program, else why not.

    ``same_bytes`` (a site that opts in, ``ReplaySite(settle_copies=True)``):
    a constant and a seed that hold the same bytes when a run starts are one
    operand. A buffer is a seed when the program later writes into it, which
    is the allocator recycling it after its last read in one recording and not
    in the other; every read before that write sees the same bytes.

    The same launches in the same order: kernel, grid, barrier, argument
    slots, inline bytes. A buffer that a launch READS is compared by what it
    holds, not by which buffer it is: a declared input by its index, a pinned
    weight by its slot, a constant by its bytes, and anything the program
    itself wrote by the launch that wrote it. MLX's allocator recycles buffers
    inside one evaluation, so the same value can sit in working memory in one
    recording and in a buffer that later becomes a declared output in the
    other; that is not a difference. A buffer a launch WRITES is compared by
    slot only, for the same reason: a fresh result can land in new memory in
    one recording and, in the other, in the buffer of a declared input that
    the function had finished with (MLX hands a freed buffer to the next
    allocation of its size, and which input's buffer that is differs from
    call to call). An update in place of a live input is still told from a
    copy followed by a write: the copy is one more launch.

    The stride of an extent-1 dimension addresses nothing, and MLX hands the
    kernel whatever stride the array's history left there. So such strides are
    left out of the layouts, an input's offset in its buffer is taken relative
    to the input, and an inline 64-bit word may differ when each recording's
    word is one of that recording's own extent-1 strides.
    """

    if len(first.steps) != len(second.steps):
        return f"{len(first.steps)} launches against {len(second.steps)}"
    if len(first.inputs) != len(second.inputs) or len(first.outputs) != len(second.outputs):
        return "declared inputs or outputs differ in number"
    if (first.steps == second.steps and first.inputs == second.inputs
            and first.outputs == second.outputs
            and [(b.role, b.index, b.data) for b in first.buffers]
            == [(b.role, b.index, b.data) for b in second.buffers]):
        # The common case: the allocator made the same choices both times.
        return None
    for name, left, right in (("input", first.inputs, second.inputs),
                              ("output", first.outputs, second.outputs)):
        for index, (a, b) in enumerate(zip(left, right)):
            if _layout(a) != _layout(b):
                return f"{name} {index}: layout {_layout(a)} against {_layout(b)}"
    programs = []
    for ir in (first, second):
        programs.append({
            "by_id": {buf.id: buf for buf in ir.buffers},
            "bases": _input_bases(ir),
            "free": _free_strides(ir),
            "writer": {},
        })
    one, two = programs
    digests: dict[int, bytes] = {}

    def held(program: dict, buffer_id: int, offset: int) -> tuple:
        buf = program["by_id"][buffer_id]
        relative = offset - program["bases"].get(buffer_id, 0) if buf.role == "input" else offset
        written = program["writer"].get(buffer_id)
        if written is not None:
            return ("made", written, relative)
        if buf.role == "input":
            return ("input", buf.index, relative)
        if buf.role == "pinned":
            return ("pinned", buf.index, relative)
        if buf.role in ("const", "seed"):
            digest = digests.get(id(buf))
            if digest is None:
                digest = digests[id(buf)] = hashlib.blake2b(buf.data or b"", digest_size=8).digest()
            return ("bytes" if same_bytes else buf.role, digest, relative)
        return ("unwritten", relative)

    writer_one, writer_two = one["writer"], two["writer"]
    for position, (a, b) in enumerate(zip(first.steps, second.steps)):
        if a == b:
            # Same kernel, arguments and buffer ids: the same launch whenever
            # every buffer it reads means the same thing in both recordings.
            same = True
            for _slot, buffer_id, offset, access in a.accesses():
                if access == "r" and held(one, buffer_id, offset) != held(two, buffer_id, offset):
                    same = False
                    break
            if same:
                for _slot, buffer_id, _offset, access in a.bindings:
                    if access != "r":
                        writer_one[buffer_id] = position
                        writer_two[buffer_id] = position
                continue
        where = f"launch {position} ({a.kernel_name})"
        if a.kernel_name != b.kernel_name:
            return f"launch {position}: kernel {a.kernel_name} against {b.kernel_name}"
        if (a.dispatch != b.dispatch or a.barrier_before != b.barrier_before or a.mark != b.mark
                or getattr(a, "tg_memory", None) != getattr(b, "tg_memory", None)):
            return f"{where}: grid, barrier or mark differs"
        inline_a, inline_b = sorted(a.inline), sorted(b.inline)
        if [(i, len(d)) for i, d in inline_a] != [(i, len(d)) for i, d in inline_b]:
            return f"{where}: inline arguments differ in slot or length"
        for (slot, bytes_a), (_slot, bytes_b) in zip(inline_a, inline_b):
            # (Struct padding MLX leaves unset is not part of the program.)
            bytes_a = inline_data(a.kernel_name, bytes_a)
            bytes_b = inline_data(b.kernel_name, bytes_b)
            if bytes_a == bytes_b:
                continue
            if len(bytes_a) % 8:
                return f"{where}: inline bytes of slot {slot} differ"
            for start in range(0, len(bytes_a), 8):
                word_a = int.from_bytes(bytes_a[start:start + 8], "little", signed=True)
                word_b = int.from_bytes(bytes_b[start:start + 8], "little", signed=True)
                if word_a != word_b and not (word_a in one["free"] and word_b in two["free"]):
                    return f"{where}: inline word {start // 8} of slot {slot}: {word_a} against {word_b}"
        bind_a, bind_b = list(a.accesses()), list(b.accesses())
        if [(i, acc) for i, _b, _o, acc in bind_a] != [(i, acc) for i, _b, _o, acc in bind_b]:
            return f"{where}: bound slots or access differ"
        for (slot, id_a, off_a, access), (_s, id_b, off_b, _acc) in zip(bind_a, bind_b):
            if access != "r":
                continue  # where a result lands is the allocator's business
            left, right = held(one, id_a, off_a), held(two, id_b, off_b)
            if left != right:
                operand = "indirect grid" if slot == -1 else f"slot {slot}"
                return f"{where}: {operand} reads {left} against {right}"
        for program, bindings in ((one, bind_a), (two, bind_b)):
            for _slot, buffer_id, _offset, access in bindings:
                if access != "r":
                    program["writer"][buffer_id] = position
    for index, (a, b) in enumerate(zip(first.outputs, second.outputs)):
        left, right = one["writer"].get(a.buffer), two["writer"].get(b.buffer)
        if left != right:
            return f"output {index}: last written by launch {left} against {right}"
        if left is None and held(one, a.buffer, a.offset) != held(two, b.buffer, b.offset):
            # Handed through unwritten: the same input or constant in both.
            return f"output {index}: passes through {held(one, a.buffer, a.offset)[:2]} against {held(two, b.buffer, b.offset)[:2]}"
    return None


def key_digest(key: Any) -> str:
    return hashlib.blake2b(repr(key).encode("utf-8"), digest_size=6).hexdigest()


class _Geometry:
    """What a site knows about a caller's geometry before its inputs are
    evaluated (``ReplaySite.plan``): how often it was called, and the keys
    (the geometry at one input layout each) seen under it."""

    __slots__ = ("calls", "entries", "fenced_run")

    def __init__(self) -> None:
        self.calls = 0
        self.entries: dict[Any, "_Entry"] = {}
        self.fenced_run = 0  # planned calls in a row that arrived at a fenced layout


class _Entry:
    """What a site knows about one geometry key."""

    __slots__ = (
        "calls", "state", "program", "ir", "signature", "refused", "detail",
        "native_runs", "clobbered", "profiled", "digest", "check_blocked", "rechecks",
        "owned_bytes", "rests", "rest_until", "kin", "cut_dispatches", "forward_routes", "checks_left",
    )

    def __init__(self, digest: str) -> None:
        self.calls = 0
        # warming | candidate (one recording held) | recorded (known steady and
        # sealed, not yet bit-checked) | live | refused | record_only | fenced
        self.state = "warming"
        self.program: Any = None
        self.ir: ProgramIR | None = None
        self.signature = ""
        self.refused = ""
        self.detail = ""
        self.native_runs = 0
        self.cut_dispatches: dict[str, int] = {}
        self.forward_routes = None  # model bindings captured for this particular recorded graph
        self.clobbered: tuple[int, ...] = ()
        self.profiled = False
        self.digest = digest
        self.check_blocked = 0
        self.rechecks = 0
        self.owned_bytes = 0
        self.rests = 0  # times the recheck budget was spent and started over
        self.rest_until = 0  # the call count at which the key is recorded again
        self.kin: Any = None  # the step this key is one layout of (``begin``'s ``kin``)
        # Bit checks still to pass before the program goes live: one, or two
        # for a program settled from recordings that differed by MLX copies.
        self.checks_left = 1


# An output over a bank checked in place that is not the bank itself (a view
# of the round's rows, say) is copied whole after the replay, up to this size.
_IN_PLACE_SMALL_ALIAS = 16 << 20


class _Bank:
    """One state bank of an in-place check (``ReplaySite._begin_check``)."""

    __slots__ = ("index", "output", "small", "chunks", "before", "native_window", "small_copies")

    def __init__(self, index: int, output: int, small: list[int], chunks: list[int]) -> None:
        self.index = index  # the input
        self.output = output  # the output that is the bank after the round
        self.small = small  # other outputs over the bank, copied whole
        self.chunks = chunks  # the 32 KiB chunks holding the round's rows
        self.before: Any = None  # digests before the replay
        self.native_window: Any = None  # the replay's chunks, copied
        self.small_copies: dict[int, Any] = {}


class _InPlace:
    __slots__ = ("banks", "relocated")

    def __init__(self) -> None:
        self.banks: dict[int, _Bank] = {}
        self.relocated = False


def room_for_check_copies(need: int) -> bool:
    """Whether a first-use check's private copies fit beside the request now.

    A check is optional work, so its copies are charged against the shed floor
    (as the draft step's are), not the abort floor a request's own growth may
    reach: on an 8 or 16 GB Mac a copy the size of several KV banks must not
    take the room the conversation needs to grow. A Mac the guard cannot read
    reports no shortfall, as for every request."""

    from ..system_memory import admission_shortfall_bytes, read_system_memory

    return admission_shortfall_bytes(read_system_memory(), growth_bytes=int(need), floor="shed") == 0


_room_for_relocated_banks = room_for_check_copies


def _changed_chunks(before: mx.array, after: mx.array, window: list[int]) -> list[int]:
    """Chunks whose digest differs, outside ``window``."""

    differs = mx.any(before != after, axis=1)
    mx.eval(differs)
    skip = set(window)
    return [index for index, flag in enumerate(differs.tolist()) if flag and index not in skip]


class Ticket:
    """One call of a site. ``native`` says which path produced the outputs."""

    __slots__ = ("site", "entry", "kind", "native", "outputs", "reason",
                 "_replayed", "_described", "_started", "_key", "_sample", "_returned",
                 "_pristine", "_in_place")

    def __init__(self, site: "ReplaySite", entry: _Entry | None, kind: str, reason: str) -> None:
        self.site = site
        self.entry = entry
        self.kind = kind  # native | fallback | record | verify | shadow
        self.native = kind == "native"
        self.outputs: list | None = None
        self.reason = reason
        self._replayed: list | None = None
        self._described: list | None = None
        self._started = 0.0
        self._key: Any = None
        self._sample: tuple | None = None
        self._returned = 0.0
        self._pristine: Any = None  # the call's inputs as they arrived (diagnosis runs only)
        self._in_place: "_InPlace | None" = None  # a check run on the live banks (see _begin_check)

    def finish(self, outputs: Any) -> None:
        """After the outputs are installed (and, on the MLX path, handed to MLX).

        A native round closes its timing sample here; a plain fallback round
        does nothing. A recorded or checked round waits for the outputs,
        closes the recording, builds or compares, and moves the key on.
        """

        if self.kind == "native":
            self.site._close_native(self)
            return
        if self.kind == "fallback":
            return
        self.site._finish(self, list(outputs))

    def abandon(self, reason: str, detail: str) -> None:
        """End a failed library call before its caller executes and counts a fallback."""
        site = self.site
        if self.native and self.entry.clobbered:
            raise _runtime.ReplayLost(f"failed:outputs: {reason} ({detail})")
        if site._open is self:
            site._abandon_open(site._engine)
        # No outputs were served through this ticket. The bank accounts for
        # the round once when its eager forward takes over.
        site.counters["calls"] -= 1
        if self.native:
            site.counters["native"] -= 1
        else:
            fallback = site.counters["fallback"]
            fallback[self.reason] -= 1
            if not fallback[self.reason]:
                del fallback[self.reason]
        abandoned = site.counters["abandoned"]
        abandoned[reason] = abandoned.get(reason, 0) + 1
        site._say(f"abandon:{reason}", f"the verify bank took back this call: {reason} ({detail})")


class ReplaySite:
    """Record policy, first-use bit check, legality, fallback and counters."""

    def __init__(
        self,
        name: str,
        *,
        engine: Any = None,
        output_names: Callable[[int], str] | None = None,
        log: Callable[[str], None] | None = None,
        fence_strided: bool = True,
        dispatch: tuple[str, str] | None = None,
        split: int | None = None,
        settle_copies: bool = False,
    ) -> None:
        self.name = str(name)
        # ``copies_only`` settling (two recordings that differ only by MLX
        # copies are one program, the shorter): for a site that opts in (the
        # round program's). Every other site keeps the stricter rule, two
        # recordings that agree in full structure.
        self._settle_copies = bool(settle_copies)
        # How this site's programs order their launches on the GPU, as
        # (dispatch, barriers) of ``runtime.DISPATCH`` and ``runtime.BARRIERS``;
        # None is the executor's own setting. Every form computes the same bits.
        self._dispatch = dispatch
        self._split = split
        # The fence of STRIDED_INPUTS. A site whose function only reads its
        # inputs turns it off: what went wrong behind that fence was a write
        # past the end of an input's buffer (the executor now refuses or
        # repairs that itself, ``input_buffer_small``), and a program that
        # writes into no input cannot do it. Such a site refuses a recording
        # that does write into one (``writes_strided_input``).
        self._fence_strided = bool(fence_strided)
        self._engine = engine  # None: the process engine, resolved lazily
        self._engine_error = ""
        self._entries: "OrderedDict[Any, _Entry]" = OrderedDict()
        self._key_times: deque[float] = deque()
        self._geometries: "OrderedDict[Any, _Geometry]" = OrderedDict()
        # Families (``plan``'s ``family``) one of whose geometries has served a
        # live program in this process: a new geometry of one of them has
        # already shown that it is long-lived (see ``_FAMILY_WARM_CALLS``).
        self._live_families: "OrderedDict[Any, None]" = OrderedDict()
        # Families one of whose geometries has been called past its warm calls
        # in this process: shown long-lived, so their callers' ``quiet`` ends
        # as soon as a live family's (``quiet_start``).
        self._warmed_families: "OrderedDict[Any, None]" = OrderedDict()
        # Per step (``begin``'s ``kin``): what each program of it that passed
        # its bit check launches, as a digest of its kernels in order
        # (``_steady_elsewhere``).
        self._steady: "OrderedDict[Any, set[bytes]]" = OrderedDict()
        self._open: Ticket | None = None
        self._pins: list = []
        self._pin_sources: list[Callable[[], list]] = []
        self._pin_live: list[Callable[[], list]] = []
        # (name, array) pairs for the pinned arrays: written beside a profiled
        # program so a census can place each launch in its layer and block.
        self._pin_names: Callable[[], list] | None = None
        self._pinned = False
        self._output_names = output_names
        self._log = log or (lambda line: print(line, file=sys.stderr, flush=True))
        self._mixed_rng: random.Random | None = None
        self._mixed_seed: str | None = None
        self._logged: dict[str, int] = {}
        self.counters: dict[str, Any] = {
            "calls": 0,
            "keys_created": 0,
            "native": 0,
            "fallback": {},
            "abandoned": {},
            "programs_built": 0,
            "programs_live": 0,
            "programs_refused": {},
            "programs_evicted": 0,
            "shadow_checks": 0,
            "shadow_mismatches": 0,
            "stability_checked": 0,
            "stability_not_comparable": 0,
            # Recordings taken as the steady program because a live program of
            # the same step launches the same kernels in the same order.
            "steady_by_kin": 0,
            "programs_rerecorded": 0,
            "rechecks": 0,
            "rests": 0,
            # Recording pairs that differed only by MLX's copies, settled on the shorter.
            "copies_settled": 0,
            "copies_settled_rechecked": 0,
            "timed": 0,
            "check_timed": 0,
            "record_s": 0.0,
            "seal_s": 0.0,
            "verify_s": 0.0,
            "native_wall_s": 0.0,
            "launches": 0,
            "cut_dispatches": {},
            "command_buffers": 0,
            "encode_ns": 0,
            "gpu_ns": 0,
            "cpu_ns": 0,
            # Native rounds on which a state leaf arrived in a buffer shorter
            # than the recorded one and the executor moved it into a longer
            # one first (a copy of a few kilobytes; the round stays native).
            "inputs_grown": 0,
            "inputs_grown_bytes": 0,
            # Written inputs a first-use check showed the program leaves the
            # room past alone (``_relax_tails``): later calls may hand them in
            # a buffer of their own size.
            "inputs_relaxed": 0,
            # First-use checks run on the live state banks (``_begin_check``).
            "checks_in_place": 0,
            "relocated_bank_checks": 0,
        }
        self.last_refusal: dict[str, Any] | None = None
        # Why the last ``plan`` that declined did (for a caller that counts
        # the call under its own reason).
        self.last_decline = ""
        # Per native call, in ns: see _NATIVE_COLUMNS.
        self._samples: "deque[tuple]" = deque(maxlen=_TIMING_SAMPLES)
        # Replays made beside the MLX path (first-use and shadow checks, on
        # private copies): the executor's own three figures.
        self._check_samples: "deque[tuple]" = deque(maxlen=_TIMING_SAMPLES)

    # ------------------------------------------------------------ configuration

    def pin(self, arrays: list) -> None:
        """Weights and other long-lived arrays the recorded launches read."""

        self._pins.extend(arrays)
        self._pinned = False

    def pin_from(self, source: Callable[[], list]) -> None:
        """Like ``pin``, collected only when the first recording is made."""

        self._pin_sources.append(source)
        self._pinned = False

    def pin_before_each_recording(self, source: Callable[[], list]) -> None:
        """Like ``pin_from``, asked again before every recording: for a set of
        long-lived arrays that grows (small persistent operands a trace makes
        the first time it needs one). Pinning an array twice is harmless."""

        self._pin_live.append(source)

    def name_pins_from(self, source: Callable[[], list]) -> None:
        """``source()`` gives ``(name, array)`` for the arrays this site pins
        (a model's parameters by path). Read only for a census
        (``MTPLX_NATIVE_ENGINE_PROFILE=forms``): which pinned buffer a launch
        reads says which layer and block it belongs to."""

        self._pin_names = source

    def release_idle_temporaries(self) -> int:
        """Release the existing executor's scratch after a served request."""
        release = getattr(self._engine, "release_idle_temporaries", None)
        return int(release()) if release is not None else 0

    def _resolve_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        if self._engine_error:
            return None
        if not _runtime.available():
            self._engine_error = _runtime.unavailable() or "module_missing"
            return None
        try:
            self._engine = _runtime.engine()
        except Exception as exc:  # the module loaded but the engine cannot start
            self._engine_error = f"engine_start_failed: {exc!r}"[:200]
            return None
        return self._engine

    # ------------------------------------------------------------------- counts

    def _count(self, reason: str) -> None:
        fallback = self.counters["fallback"]
        fallback[reason] = fallback.get(reason, 0) + 1

    def _say(self, topic: str, line: str, limit: int = 3) -> None:
        seen = self._logged.get(topic, 0)
        self._logged[topic] = seen + 1
        if seen < limit:
            self._log(f"[mtplx-engine] {self.name}: {line}")

    def snapshot(self) -> dict[str, Any]:
        """Counters now, for a per-request difference (``report(since=...)``)."""

        return json.loads(json.dumps(self.counters))

    def report(self, since: dict[str, Any] | None = None) -> dict[str, Any]:
        """Counters for the request log and /health.

        With ``since`` (a ``snapshot()`` taken at the start of a request) the
        counts are that request's; without it they are the process totals.
        """

        now = self.counters
        base = since or {}

        def delta(name: str) -> Any:
            value, before = now[name], base.get(name, 0)
            if isinstance(value, dict):
                before = before or {}
                out = {k: v - before.get(k, 0) for k, v in value.items()}
                return {k: v for k, v in out.items() if v}
            return value - before

        calls = delta("calls")
        native = delta("native")
        mode = _runtime.mode()
        # Off never loads the compiled module, so it cannot say more.
        off = mode == "0" and self._engine is None
        minute_ago = time.monotonic() - 60
        report: dict[str, Any] = {
            "site": self.name,
            "mode": mode,
            "replay": replay_allowed(),
            "available": None if off else (self._engine is not None or _runtime.available()),
            "calls": calls,
            "native": native,
            "native_share": round(native / calls, 4) if calls else None,
            "fallback": delta("fallback"),
            "programs_built": delta("programs_built"),
            "programs_live": delta("programs_live"),
            "programs_refused": delta("programs_refused"),
            "programs_evicted": delta("programs_evicted"),
            "programs_held": sum(1 for e in self._entries.values() if e.program is not None),
            "programs_mb": round(
                sum(e.owned_bytes for e in self._entries.values() if e.ir is not None) / 2**20, 1
            ),
            "programs_budget_mb": self._program_budget() >> 20,
            "keys": len(self._entries),
            "keys_created": delta("keys_created"),
            "keys_last_minute": sum(t >= minute_ago for t in tuple(self._key_times)),
        }
        unavailable = self._engine_error or (
            "" if self._engine is not None or off else _runtime.unavailable()
        )
        if unavailable:
            report["unavailable"] = unavailable
        for name in ("shadow_checks", "shadow_mismatches", "inputs_grown", "inputs_grown_bytes",
                     "inputs_relaxed", "checks_in_place", "relocated_bank_checks"):
            value = delta(name)
            if value:
                report[name] = value
        if native:
            if cut_dispatches := delta("cut_dispatches"):
                report["cut_dispatches"] = cut_dispatches
            for name in ("launches", "command_buffers"):
                report[f"{name}_per_run"] = round(delta(name) / native, 2)
            for name in ("encode_ns", "gpu_ns", "cpu_ns"):
                report[f"{name[:-3]}_ms_per_run"] = round(delta(name) / native / 1e6, 4)
            report["native_wall_ms_per_run"] = round(delta("native_wall_s") / native * 1e3, 4)
            timing = self._timing(self._samples, delta("timed"), _NATIVE_COLUMNS)
            if timing:
                report["timing_ms"] = timing
        timing = self._timing(self._check_samples, delta("check_timed"), _CHECK_COLUMNS)
        if timing:
            report["check_timing_ms"] = timing
        for name in ("stability_checked", "stability_not_comparable", "steady_by_kin",
                     "programs_rerecorded", "rechecks", "rests", "copies_settled", "copies_settled_rechecked"):
            value = delta(name)
            if value:
                report[name] = value
        for name in ("record_s", "seal_s", "verify_s"):
            value = delta(name)
            if value:
                report[name] = round(value, 6)
        if self.last_refusal is not None:
            report["last_refusal"] = self.last_refusal
        refused = [{"program": e.digest, "code": e.refused, "detail": e.detail}
                   for e in self._entries.values() if e.state == "refused"]
        if refused:
            # Every refused program's facts (the launch where its recordings differed), newest last.
            report["refusals"] = refused[-8:]
        abandoned = delta("abandoned")
        if abandoned:
            report["abandoned"] = abandoned
        return report

    @staticmethod
    def _timing(samples: Any, newest: int, columns: tuple[str, ...]) -> dict[str, Any] | None:
        """Mean, median and 95th percentile over the newest replays, in ms
        (the columns are described at ``_NATIVE_COLUMNS``)."""

        count = min(int(newest), len(samples))
        if count <= 0:
            return None
        rows = list(samples)[-count:]
        out: dict[str, Any] = {"samples": count}
        for column, name in enumerate(columns):
            values = sorted(row[column] for row in rows)
            out[name] = {
                "mean": round(sum(values) / count / 1e6, 4),
                "p50": round(values[(count - 1) // 2] / 1e6, 4),
                "p95": round(values[min(count - 1, int(count * 0.95))] / 1e6, 4),
            }
        return out

    # -------------------------------------------------------------------- calls

    def declined(self) -> bool:
        """True (and counted) when no call can replay: the switch is off or
        the compiled module is not there. The caller then skips gathering the
        inputs, so the MLX path runs with nothing added to it."""

        if _runtime.mode() == "0":
            reason = DISABLED
        elif self._resolve_engine() is None:
            reason = UNAVAILABLE
            self._say("unavailable", f"native engine unavailable ({self._engine_error})", limit=1)
        else:
            return False
        self.counters["calls"] += 1
        self._count(reason)
        return True

    def quiet_start(self, tokens: int, family: Any = None) -> bool:
        """``quiet_start`` for a call of this site: what the caller passes
        ``plan`` and ``begin`` as ``quiet``. A geometry whose ``family`` has
        served a live program here, or has been called past its warm calls
        here, waits ``_FAMILY_QUIET_TOKENS``; every other one ``QUIET_TOKENS``.

        The second case is the stream of short turns an agent sends (tool
        calls of 50 to 200 tokens): each is shorter than ``QUIET_TOKENS``, so
        with the long wait a geometry first met in such a stream was never
        recorded in any of them, and every one of its rounds ran on MLX
        (V3 night 2 baseline, bars-1117-0448: the first request of a cell 384
        of 445 rounds native, request_start 45). A geometry past its warm
        calls has run long in this process; the recording then falls inside
        an answer once, after its first text, and every later round of every
        later turn replays.
        """

        live = family is not None and (
            family in self._live_families or family in self._warmed_families
        )
        return quiet_start(tokens, _FAMILY_QUIET_TOKENS if live else QUIET_TOKENS)

    def plan(
        self, geometry: Any, *, quiet: bool = False, count_decline: bool = True, family: Any = None
    ) -> bool:
        """Before the caller dispatches anything: does the site take this call?

        A caller that pipelines (it hands its pending inputs to MLX early and
        builds the next graph while the GPU works) must know before that
        point whether the site will serve, record or check the call, because
        those evaluate the inputs themselves. ``geometry`` is the caller's
        key without the layout of the inputs, which only evaluated inputs
        have: routes and shapes.

        True: the caller makes no dispatch of its own and calls ``begin`` with
        the same ``geometry`` and ``planned=True``. False: the call is the
        MLX path's, untouched; it is counted here with its reason and
        ``begin`` is not called. The site says True only for a geometry with
        a live program (or, in shadow mode, a program to check), and for one
        whose turn it is to be recorded, sealed or checked: past its warm
        calls and outside an answer's first tokens (``quiet``).

        ``count_decline=False`` is for a caller that serves a declined call
        by a path of its own and counts it under its own reason (a copy
        window runs its eager forward, cheaper than the compiled MLX path, and
        is counted as a copy round): the call still counts toward the
        geometry's warm calls, but no fallback reason and no call is counted
        here for a decline.

        ``family`` is the caller's name for the geometries that are one step
        at different bank capacities (None: no family). Once one of them has
        served a live program, a new one waits ``_FAMILY_WARM_CALLS`` calls
        instead of the warm calls, and its caller's ``quiet`` ends sooner
        (``quiet_start``); the rest of its life (two recordings, the seal, the
        first-use bit check) is the same.
        """

        def decline(reason: str) -> bool:
            self.last_decline = reason
            return self._decline(reason) if count_decline else False

        if _runtime.mode() == "0":
            return decline(DISABLED)
        engine = self._resolve_engine()
        if engine is None:
            self._say("unavailable", f"native engine unavailable ({self._engine_error})", limit=1)
            return decline(UNAVAILABLE)
        if self._open is not None:
            # A recorded call that never finished (the MLX path raised) must
            # not leave the recorder running over the calls the site declines.
            self._abandon_open(engine)
        known = self._geometries.get(geometry)
        if known is None:
            first = not self._geometries
            known = self._geometries[geometry] = _Geometry()
            while len(self._geometries) > _REMEMBERED_REFUSALS:
                self._geometries.popitem(last=False)
        else:
            first = False
            self._geometries.move_to_end(geometry)
        known.calls += 1
        if any(entry.state == "evicted" for entry in known.entries.values()):
            # An evicted program's key starts again, as a key never seen.
            known.entries = {k: e for k, e in known.entries.items() if e.state != "evicted"}
        # A fenced layout (a request's first round) says nothing about the
        # layout the rounds after it arrive at.
        steady = [entry for entry in known.entries.values() if entry.state != "fenced"]
        if any(entry.state == "live" for entry in steady):
            if family is not None:
                self._live_families[family] = None
                self._live_families.move_to_end(family)
                while len(self._live_families) > _REMEMBERED_FAMILIES:
                    self._live_families.popitem(last=False)
            return True
        warm = self._warm_calls()
        if family is not None and family in self._live_families:
            warm = min(warm, _FAMILY_WARM_CALLS)
        if family is not None and known.calls >= warm and family not in self._warmed_families:
            # From its last warm call on: the caller works out ``quiet`` before
            # it asks, so the call after the warm calls already sees the
            # family as one that has run here.
            self._warmed_families[family] = None
            while len(self._warmed_families) > _REMEMBERED_FAMILIES:
                self._warmed_families.popitem(last=False)
        if known.calls <= warm:
            return decline(WARMING if first or known.calls > 1 else KEY_NEW)
        if quiet:
            return decline(REQUEST_START)
        if not steady:
            # Nothing but fenced layouts so far. The steady one is expected on
            # the next call; a geometry that only ever arrives fenced is left
            # to MLX, and looked at again now and then.
            if known.fenced_run < _FENCED_RUN or known.calls % _FENCED_RETRY == 0:
                return True
            return decline(STRIDED_INPUTS)
        if any(entry.state in ("warming", "candidate", "recorded") for entry in steady):
            return True
        # Every steady layout this geometry has shown is closed: say why the last one is.
        last = steady[-1]
        if last.state == "refused":
            return decline(f"refused:{last.refused}")
        return decline(RECORD_ONLY)

    def _decline(self, reason: str) -> bool:
        self.counters["calls"] += 1
        self._count(reason)
        return False

    def refuse(self, reason: str) -> None:
        """Count a call its caller keeps on the MLX path by a rule of its own,
        without asking ``plan`` (so it never warms a geometry toward a
        recording): one call, under ``reason``."""

        self.last_decline = reason
        self._decline(reason)

    def begin(
        self,
        key: Any,
        inputs: list,
        *,
        admit: Callable[[int], bool] | None = None,
        extra_refs: int = 1,
        started: float | None = None,
        entered: float | None = None,
        quiet: bool = False,
        planned: bool = False,
        should_stop: Callable[[], bool] | None = None,
        windows: list[StateWindow] | None = None,
        protect_banks: bool = False,
        kin: Any = None,
    ) -> Ticket:
        """Decide which path serves this call.

        ``inputs`` are the call's arrays in the function's order. The site
        keeps no reference to them: a recorded MLX evaluation must be free to
        write its state in place. ``admit(nbytes)`` says whether the private
        copies a check needs fit in memory now; None admits. ``extra_refs`` is
        how many places besides ``inputs`` hold each array the program writes
        into (the cache entry it lives in, a caller's list): the executor
        refuses to write into an array anyone else could still be reading.
        ``started`` is the caller's ``time.perf_counter()`` from before it
        gathered the inputs, so the site's reported overhead includes that;
        ``entered`` is the caller's clock at the start of its whole call.
        ``quiet`` says this call is one of a request's first rounds: a live
        program still serves it, but no recording and no check is made on it
        (they cost tens of milliseconds, and a request's first text must
        never wait for them); MLX serves and the call is counted
        ``request_start``. ``planned`` says the caller asked ``plan`` with
        this ``key`` first and was told True: the geometry's warm calls were
        counted there, so a key first seen here is recorded at once.
        ``should_stop`` is the request's own cancel flag, looked at while a
        live replay is waited for. ``windows`` names the state banks the call
        writes only a few rows of, as ``(input, axis, offset input, rows)``:
        input ``input`` is written at rows ``[o, o + rows)`` of ``axis``,
        where ``o`` is the value of the scalar input ``offset input``, and
        read only after those rows are written. A pair of adjacent axes
        addresses rows in a physical page bank. A fifth element names the
        logical output for that bank. A bank whose buffer now holds another
        logical state requires a private copy for the check. Banks that keep
        their own output can be checked in place at any context length
        (see ``_begin_check``). With ``protect_banks``, the check declines if
        it would clone a named bank.
        ``kin`` names the step this key is one layout of, when ``key`` says
        more than that (a caller whose key carries the bank's capacity passes
        the key without it); None: ``key`` itself, which the site completes
        with the layout of the inputs. A first recording that launches what a
        live program of the same kin launches needs no second one
        (``_steady_elsewhere``).

        Raises ``runtime.ReplayLost`` when a live replay was committed, writes
        into its inputs and gave no outputs: those inputs are of unknown
        content, so no path can serve this call and the caller's request ends
        (see ``_begin_native``).
        """

        if started is None:
            started = time.perf_counter()
        if entered is None:
            entered = started
        counters = self.counters
        counters["calls"] += 1
        mode = _runtime.mode()
        if mode == "0":
            self._count(DISABLED)
            return Ticket(self, None, "fallback", DISABLED)
        engine = self._resolve_engine()
        if engine is None:
            self._count(UNAVAILABLE)
            self._say("unavailable", f"native engine unavailable ({self._engine_error})", limit=1)
            return Ticket(self, None, "fallback", UNAVAILABLE)
        if self._open is not None:
            self._abandon_open(engine)
        try:
            # Whatever is still pending in an input (a commit after a
            # rejection, gathered rows) is not part of the function, and an
            # array has its real strides only once it is evaluated.
            before_inputs = time.perf_counter()
            mx.eval(*inputs)
            geometry = key
            key = (key, self._layout_key(engine, inputs))
            inputs_s = time.perf_counter() - before_inputs
        except Exception as exc:  # noqa: BLE001 - the MLX path raises it again, in its own words
            self._count("input_error")
            self._say("input", f"inputs could not be evaluated: {exc!r}")
            return Ticket(self, None, "fallback", "input_error")

        known = self._geometries.get(geometry) if planned else None
        entry = self._entries.get(key)
        if entry is None:
            first_key = not self._entries
            now = time.monotonic()
            while self._key_times and self._key_times[0] < now - 60:
                self._key_times.popleft()
            self._key_times.append(now)
            counters["keys_created"] += 1
            entry = _Entry(key_digest(key))
            entry.kin = geometry if kin is None else kin
            self._entries[key] = entry
            self._trim_entries()
            entry.calls = 1
            if known is not None:
                known.entries[key] = entry
            if (
                self._fence_strided
                and self._strided(engine, inputs)
                and not (mode == "shadow" and _env_flag(DIAGNOSE_ENV))
            ):
                # (A shadow run with the diagnosis on is the one place the
                # fence is lifted: that run exists to catch this case.)
                entry.state = "fenced"
                if known is not None:
                    known.fenced_run += 1
                self._count(STRIDED_INPUTS)
                return Ticket(self, entry, "fallback", STRIDED_INPUTS)
            if not planned:
                reason = WARMING if first_key else KEY_NEW
                self._count(reason)
                return Ticket(self, entry, "fallback", reason)
            # The geometry's warm calls were the plan's.
            entry.calls = self._warm_calls() + 1
        else:
            self._entries.move_to_end(key)
            entry.calls += 1

        state = entry.state
        if known is not None:
            known.fenced_run = known.fenced_run + 1 if state == "fenced" else 0
        if state == "fenced":
            self._count(STRIDED_INPUTS)
            return Ticket(self, entry, "fallback", STRIDED_INPUTS)
        if state == "refused":
            reason = f"refused:{entry.refused}"
            self._count(reason)
            return Ticket(self, entry, "fallback", reason)
        if state == "record_only":
            self._count(RECORD_ONLY)
            return Ticket(self, entry, "fallback", RECORD_ONLY)
        if state == "warming" and entry.calls <= self._warm_calls():
            self._count(WARMING)
            return Ticket(self, entry, "fallback", WARMING)
        if state in ("warming", "candidate"):
            # A geometry is recorded on consecutive calls until a recording is
            # known to be the steady program: MLX copies a state bank whole
            # when something still holds it, so one recording can carry
            # launches the steady path does not make (``_take_recording``).
            if quiet:
                self._count(REQUEST_START)
                return Ticket(self, entry, "fallback", REQUEST_START)
            if entry.calls <= entry.rest_until:
                self._count(RESTING)
                return Ticket(self, entry, "fallback", RESTING)
            return self._begin_record(engine, key, entry, inputs)
        if state == "recorded":
            if quiet:
                self._count(REQUEST_START)
                return Ticket(self, entry, "fallback", REQUEST_START)
            if not replay_allowed():
                # Dump-only: the program is built and written; profile it once
                # on private copies when asked, and keep serving from MLX.
                self._maybe_profile(engine, entry, inputs, admit)
                entry.state = "record_only"
                self._count(RECORD_ONLY)
                return Ticket(self, entry, "fallback", RECORD_ONLY)
            return self._begin_check(
                engine, key, entry, inputs, admit, "verify", windows=windows, extra_refs=extra_refs,
                protect_banks=protect_banks,
            )
        # live
        if mode == "shadow":
            return self._begin_check(engine, key, entry, inputs, admit, "shadow",
                                     windows=windows, protect_banks=protect_banks)
        if self._mixed_declines():
            self._count(MIXED)
            return Ticket(self, entry, "fallback", MIXED)
        return self._begin_native(
            engine, entry, inputs, extra_refs, started, entered, inputs_s, should_stop
        )

    @staticmethod
    def _strided(engine: Any, inputs: list) -> bool:
        """True when some input is not laid out row by row (see STRIDED_INPUTS)."""

        try:
            described = engine.describe(inputs)
        except Exception:  # noqa: BLE001 - the recording would say why; this only fences
            return False
        for ref in described:
            flags = getattr(ref, "flags", None)
            if flags is not None and getattr(ref, "nbytes", 1) and not int(flags) & 2:
                return True
        return False

    @staticmethod
    def _layout_key(engine: Any, inputs: list) -> Any:
        native = getattr(engine, "layout_key", None)
        if native is not None:
            return (len(inputs), native(inputs))
        return tuple((tuple(a.shape), str(a.dtype)) for a in inputs)

    # ---------------------------------------------------------------- recording

    def _ensure_pinned(self, engine: Any) -> None:
        if self._pinned:
            return
        sources, self._pin_sources = self._pin_sources, []
        for source in sources:
            self._pins.extend(source())
        pending, self._pins = self._pins, []
        if pending:
            try:
                engine.pin(pending)
                pending = []
            except ValueError:
                # Some array has never been evaluated (a weight no forward has
                # read yet). No recorded launch can have read it either, so it
                # waits; it is offered again before the next recording.
                waiting = []
                for array in pending:
                    try:
                        engine.pin([array])
                    except ValueError:
                        waiting.append(array)
                pending = waiting
        self._pins = pending
        self._pinned = not pending
        if pending:
            self._say("pin", f"{len(pending)} arrays not evaluated yet were not pinned", limit=1)

    def _pin_grown(self, engine: Any) -> None:
        for source in self._pin_live:
            arrays = [a for a in source() if a is not None]
            if arrays:
                engine.pin(arrays)

    def _begin_record(self, engine: Any, key: Any, entry: _Entry, inputs: list) -> Ticket:
        ticket = Ticket(self, entry, "record", RECORDING)
        ticket._key = key
        ticket._started = time.perf_counter()
        try:
            self._ensure_pinned(engine)
            self._pin_grown(engine)
            self._settle(engine, inputs)
            ticket._described = engine.describe(inputs)
            engine.record_begin(label=f"{self.name}:{entry.digest}", inputs=ticket._described)
        except Exception as exc:
            self._refuse(entry, "record_error", repr(exc))
            reason = f"refused:{entry.refused}"
            self._count(reason)
            return Ticket(self, entry, "fallback", reason)
        self._open = ticket
        self._count(RECORDING)
        return ticket

    def _begin_check(
        self, engine: Any, key: Any, entry: _Entry, inputs: list,
        admit: Callable[[int], bool] | None, kind: str,
        *, windows: list[StateWindow] | None = None, extra_refs: int = 1,
        protect_banks: bool = False,
    ) -> Ticket:
        """The first-use check: the program replays beside the MLX path, whose
        outputs it must give bit for bit (``_finish`` compares).

        The program writes into some of its inputs, and the MLX path must see
        them as they were. Small ones get a private copy each. The state banks
        a caller names in ``windows`` (keys, values and raw index keys of every
        attention layer: 24 KiB of the 28 KiB a token of context costs) are
        not copied: a copy is the size of the conversation (6.3 GB at 262K
        tokens) and does not fit beside it at long context, which kept every
        new program of the founder's 156K session off the engine
        (``verify_memory``, 4,264 rounds). The replay runs on the banks
        themselves. Each round writes its own rows of a bank before it reads
        them, so the MLX path that follows computes the same result from the
        bank as the replay left it; what must be shown is that the replay
        wrote nothing else, and that its rows are MLX's. So every bank is
        digested before and after the replay (``digest``: 64 bits per 32 KiB,
        read on the GPU), the chunks holding the round's rows are kept, and
        after the MLX path those chunks are compared byte for byte and the
        rest of the bank by digest. A replay that changed a bank outside its
        rows has written state nobody can restore: the request ends, as for a
        replay lost on the GPU (``runtime.ReplayLost``), and the key is
        refused. The check's memory is the small copies and a few chunks, at
        any context.
        """

        reason = VERIFYING if kind == "verify" else SHADOW
        ticket = Ticket(self, entry, kind, reason)
        ticket._key = key
        ticket._started = time.perf_counter()
        in_place: _InPlace | None = None
        try:
            if not protect_banks and _env_flag(DIAGNOSE_ENV) and _runtime.dump_dir() is not None:
                # Agent A's diagnosis: every input copied before anything runs
                # on it, so a mismatch can be stepped launch by launch.
                from . import diagnose as _diagnose

                ticket._pristine = _diagnose.Pristine(inputs)
            # A measuring run that asks for this key's profile checks on
            # private copies of every written input, banks included, as every
            # check did before the in-place plan: the profile runs on copies.
            profiling = not protect_banks and bool(_profile_mode()) and not entry.profiled and _runtime.dump_dir() is not None
            in_place = (
                self._in_place_plan(engine, entry, inputs, windows)
                if kind == "verify" and ticket._pristine is None and not profiling
                else None
            )
            if in_place is not None and in_place.relocated:
                # Copies of relocated KV banks can grow with the conversation.
                # Charge the actual private workspace before cloning any input.
                if admit is None:
                    admit = _room_for_relocated_banks
                if not in_place.banks:
                    in_place = None  # replay will have only private writable inputs
            kept = tuple(in_place.banks) if in_place is not None else ()
            if protect_banks and any(index in entry.clobbered and index not in kept for index, *_ in windows or ()):
                self._count("uncheckable_history_window")
                return Ticket(self, entry, "fallback", "uncheckable_history_window")
            replay_inputs = self._replay_inputs(engine, entry, inputs, admit, keep=kept)
            if replay_inputs is None:
                # No room for the private copies now: the MLX path serves and
                # the check waits for a later call.
                self._count(VERIFY_MEMORY)
                return Ticket(self, entry, "fallback", VERIFY_MEMORY)
            if kind == "verify" and not entry.profiled and _profile_mode() and kept:
                # The profile writes into its inputs, and these are live banks
                # the check runs on in place: no census for this program, and
                # it says why instead of returning nothing.
                self._say(
                    f"profile_skipped:{entry.digest}",
                    f"census profile skipped for program {entry.digest}: the check keeps "
                    f"{len(kept)} live banks in place; boot with {REPLAY_ENV}=0",
                    limit=1,
                )
            if kind == "verify" and not entry.profiled and _profile_mode() and not kept:
                # The profile run writes into the copies; the check gets fresh ones.
                self._profile(engine, entry, replay_inputs)
                self._census_extras(engine, entry, inputs, admit)
                replay_inputs = self._replay_inputs(engine, entry, inputs, admit)
                if replay_inputs is None:
                    self._count(VERIFY_MEMORY)
                    return Ticket(self, entry, "fallback", VERIFY_MEMORY)
            # The room past each written input in its buffer, marked: whether
            # the program leaves it alone decides how long a buffer later
            # calls must hand that input in (``_relax_tails``).
            marks = self._mark_tails(engine, entry, replay_inputs)
            if in_place is not None:
                self._in_place_before(engine, in_place, replay_inputs)
            # The private copies are held by this list alone; a bank checked
            # in place is held as the live path holds it, and by this list.
            replayed, illegal = engine.try_run(
                entry.program, replay_inputs, extra_refs=(extra_refs + 1) if kept else 0
            )
            if not illegal:
                self._relax_tails(engine, entry, replay_inputs, marks)
            if illegal:
                del replay_inputs
                # Nothing was touched. A reason that belongs to this call
                # alone lets a later call try again, a few times.
                entry.check_blocked += 1
                if entry.check_blocked < _CHECK_RETRIES:
                    reason = f"check_blocked:{illegal}"
                    self._count(reason)
                    self._say("blocked", f"check of {entry.digest} could not run: {illegal}")
                    return Ticket(self, entry, "fallback", reason)
                self._refuse(entry, "check_illegal", illegal)
                refused = f"refused:{entry.refused}"
                self._count(refused)
                return Ticket(self, entry, "fallback", refused)
            replayed = list(replayed)
            if in_place is not None:
                outside = self._in_place_after(engine, entry, in_place, replay_inputs, replayed)
                if outside:
                    del replay_inputs, replayed
                    self._refuse(entry, "wrote_outside_window", outside)
                    self._count("check_wrote_outside_window")
                    raise _runtime.ReplayLost(
                        f"failed:check_wrote_outside_window: {outside}", wrote_inputs=True
                    )
                ticket._in_place = in_place
            del replay_inputs
            ticket._replayed = replayed
            self._sample(engine, self._check_samples, "check_timed", None)
        except _runtime.ReplayLost as exc:
            if in_place is None:
                # A check on private copies lost nothing the caller holds.
                self._refuse(entry, "check_error", repr(exc))
                refused = f"refused:{entry.refused}"
                self._count(refused)
                return Ticket(self, entry, "fallback", refused)
            # On the live banks: their content is not known any more.
            if entry.state != "refused":
                self._refuse(entry, f"replay_lost_{exc.code}", repr(exc))
            raise
        except Exception as exc:
            self._refuse(entry, "check_error", repr(exc))
            refused = f"refused:{entry.refused}"
            self._count(refused)
            return Ticket(self, entry, "fallback", refused)
        if in_place is not None:
            self.counters["checks_in_place"] += 1
        self._open = ticket
        self._count(reason)
        return ticket

    def _in_place_plan(
        self, engine: Any, entry: _Entry, inputs: list,
        windows: list[StateWindow] | None,
    ) -> "_InPlace | None":
        """Which written banks of this call the check runs on in place, and the
        chunks that hold the rows the round writes. Relocated banks need a
        private check with memory admission. None leaves the usual private
        checks for callers without eligible or relocated banks."""

        if not windows or entry.ir is None or not hasattr(engine, "describe"):
            return None
        if getattr(engine, "mark_tail", None) is None:
            return None  # the stand-in executor of the policy tests
        from . import digest as _digest

        ir = entry.ir
        members: dict[int, int] = {}
        for arr in ir.inputs:
            members[arr.buffer] = members.get(arr.buffer, 0) + 1
        aliased: dict[int, list[int]] = {}
        for out, source in ir.output_alias.items():
            aliased.setdefault(int(source), []).append(int(out))
        clobbered = set(entry.clobbered)
        plan = _InPlace()
        for window in windows:
            index, axis, start_index, rows = window[:4]
            state_output = window[4] if len(window) == 5 else None
            index, start_index = int(index), int(start_index)
            if index not in clobbered or not 0 <= start_index < len(inputs):
                continue
            # The allocator may recycle this buffer for another layer's
            # identically shaped bank. That recording can still replay, but
            # its whole-bank copy would overwrite the reference's live history.
            if state_output is not None and ir.output_alias.get(state_output) != index:
                plan.relocated = True
                self.counters["relocated_bank_checks"] += 1
                continue
            spec = ir.inputs[index]
            array = inputs[index]
            if members.get(spec.buffer, 0) != 1 or not int(spec.flags) & 2:
                continue
            if not _digest.supported(array):
                continue
            leaf, small = None, []
            usable = True
            for out in aliased.get(index, ()):
                arr = ir.outputs[out]
                if (tuple(arr.shape), arr.dtype, int(arr.offset), tuple(arr.strides)) == (
                    tuple(spec.shape), spec.dtype, int(spec.offset), tuple(spec.strides)
                ) and leaf is None and (state_output is None or out == state_output):
                    leaf = out
                elif int(arr.nbytes) <= _IN_PLACE_SMALL_ALIAS:
                    small.append(out)
                else:
                    usable = False
            if not usable or leaf is None:
                continue
            start = int(inputs[start_index].item())
            shape = tuple(array.shape)
            if isinstance(axis, tuple):
                first, second = axis
                if second != first + 1:
                    raise ValueError("a paged write window requires adjacent axes")
                shape = (*shape[:first], shape[first] * shape[second], *shape[second + 1:])
                axis = first
            chunks = _digest.window_chunks(shape, int(array.dtype.size), axis, start, int(rows))
            if not chunks:
                continue
            plan.banks[index] = _Bank(index, leaf, small, chunks)
        return plan if plan.banks or plan.relocated else None

    def _in_place_before(self, engine: Any, plan: "_InPlace", replay_inputs: list) -> None:
        from . import digest as _digest

        banks = [replay_inputs[index] for index in plan.banks]
        digests = [_digest.digest(bank) for bank in banks]
        mx.eval(*digests)
        for bank, value in zip(plan.banks.values(), digests):
            bank.before = value
        del digests
        # The digest's command buffer holds the banks for a moment: the
        # replay writes only into data nobody else holds.
        self._settle(engine, banks)

    def _in_place_after(
        self, engine: Any, entry: _Entry, plan: "_InPlace", replay_inputs: list, replayed: list
    ) -> str:
        """After the replay on the live banks: keep what the comparison needs,
        drop the replay's arrays over the banks (MLX must be free to write
        them in place next), and say where the replay wrote outside its rows
        ("" when nowhere)."""

        from . import digest as _digest

        after, windows, smalls = [], [], []
        for bank in plan.banks.values():
            live = replay_inputs[bank.index]
            after.append(_digest.digest(live))
            windows.append(_digest.chunk_copy(live, bank.chunks))
            for out in bank.small:
                smalls.append((bank, out, private_copy(replayed[out])))
        mx.eval(*after, *windows, *[copy for _bank, _out, copy in smalls])
        outside = ""
        for bank, now, rows in zip(plan.banks.values(), after, windows):
            bank.native_window = rows
            changed = _changed_chunks(bank.before, now, bank.chunks)
            if changed and not outside:
                outside = f"input {bank.index}: chunks {changed[:4]} outside rows {bank.chunks[:4]}"
        for bank, out, copy in smalls:
            bank.small_copies[out] = copy
        for bank in plan.banks.values():
            replayed[bank.output] = None
            for out in bank.small:
                replayed[out] = None
        del after, windows, smalls
        self._settle(engine, [replay_inputs[index] for index in plan.banks])
        return outside

    def _replay_inputs(
        self, engine: Any, entry: _Entry, inputs: list, admit: Callable[[int], bool] | None,
        *, keep: tuple[int, ...] = (),
    ) -> list | None:
        """The call's inputs with a private copy of each one the program writes into.

        A program writes into the whole buffer its input was recorded in, not
        only where the array is (MLX hands a freed input's buffer to later
        launches as working memory), so the executor never writes an input
        whose buffer is shorter than the recorded one: it moves the array
        into a longer buffer first, or declines a large one
        (``input_buffer_small``). So the copy keeps the layout: the
        executor's ``clone_layout`` copies the whole buffer and leaves the
        array the same view of it. An executor without one (the tests'
        stand-in) gets a plain copy.
        """

        if not entry.clobbered:
            return list(inputs)
        copied = [index for index in entry.clobbered if index not in keep]
        need = sum(int(inputs[index].nbytes) for index in copied)
        if admit is not None and need and not admit(need):
            return None
        clone = getattr(engine, "clone_layout", None)
        recorded = self._recorded_lengths(entry)
        replay_inputs = list(inputs)
        for index in copied:
            source = inputs[index]
            if clone is None:
                replay_inputs[index] = private_copy(source)
                continue
            # At least as long as the buffer the program was recorded with:
            # the copy is then never too short to run on, and the room past
            # the array is there to be marked.
            length = recorded.get(index, 0)
            if length:
                length += max(0, int(engine.describe([source])[0].offset))
                replay_inputs[index] = clone(source, min_length=length)
            else:
                replay_inputs[index] = clone(source)
        # No other name may hold a copy: the executor writes only into arrays
        # that this list alone references.
        if copied:
            mx.eval(*[replay_inputs[index] for index in copied])
        return replay_inputs

    @staticmethod
    def _recorded_lengths(entry: _Entry) -> dict[int, int]:
        """Per written input: the length of the buffer it was recorded in."""

        ir = entry.ir
        if ir is None:
            return {}
        by_id = {buf.id: buf for buf in ir.buffers}
        out: dict[int, int] = {}
        for index in entry.clobbered:
            arr = ir.inputs[index]
            buf = by_id.get(arr.buffer)
            if buf is not None:
                # Counted from the array's start: the copy keeps the array's
                # offset inside its buffer, and adds as much room before it.
                out[index] = max(0, int(buf.nbytes) - max(0, int(arr.offset)))
        return out

    def _mark_tails(self, engine: Any, entry: _Entry, replay_inputs: list) -> dict[int, tuple[int, int]]:
        """Mark the room past every written input of a check (its private copy),
        so the run can show whether the program uses it: ``{index: (seed, bytes)}``."""

        mark = getattr(engine, "mark_tail", None)
        if mark is None or not entry.clobbered:
            return {}
        arrays = [replay_inputs[index] for index in entry.clobbered]
        # Right after an evaluation MLX's command buffer still holds what it
        # touched; a tail is only marked on data nobody else holds.
        self._settle(engine, arrays)
        del arrays
        marks: dict[int, tuple[int, int]] = {}
        for index in entry.clobbered:
            seed = (hash((entry.digest, index, entry.calls)) & 0xFFFFFFFFFFFF) | 1
            marked = int(mark(replay_inputs[index], seed) or 0)
            if marked:
                marks[index] = (seed, marked)
        return marks

    def _relax_tails(
        self, engine: Any, entry: _Entry, replay_inputs: list, marks: dict[int, tuple[int, int]]
    ) -> None:
        """After a run on marked inputs: an input whose marked room is untouched
        may arrive later in a buffer without that room (the program needs
        nothing past the array), so a bank in a buffer of its own size (a
        restored state, a round MLX served) stays on the engine.

        The room is the allocator's: MLX hands out a cached buffer up to two
        pages longer than asked for, and writes an array in place only while
        its buffer is at most 16 KiB longer than it. So two buffers of the
        same bank can differ by 16 KiB, and a program recorded in the longer
        one declined the shorter (``input_buffer_small``) for every round of
        a request once the bank was larger than the executor copies (8 MiB):
        98 of 154 rounds of a cold 130K prompt on 12 November.
        """

        intact = getattr(engine, "tail_intact", None)
        relax = getattr(engine, "relax_input", None)
        if not marks or intact is None or relax is None:
            return
        try:
            grown = int((engine.last_run() or {}).get("grown", 0) or 0)
        except Exception:  # noqa: BLE001 - without the run's figures nothing is relaxed
            return
        if grown:
            # The run worked on a longer copy of some input, not on the marked
            # one: its marks say nothing.
            return
        relaxed = 0
        kept = []
        for index, (seed, marked) in marks.items():
            if intact(replay_inputs[index], seed, marked):
                if relax(entry.program, index):
                    relaxed += 1
            else:
                kept.append(index)
        if relaxed:
            self.counters["inputs_relaxed"] += relaxed
        if kept:
            self._say("tail", f"program {entry.digest} writes past the array of inputs "
                              f"{kept[:8]}: they keep their recorded buffer length", limit=4)

    def _begin_native(
        self, engine: Any, entry: _Entry, inputs: list, extra_refs: int, started: float,
        entered: float, inputs_s: float, should_stop: Callable[[], bool] | None = None,
    ) -> Ticket:
        # The watched wait (agent A's executor, module 0.6.0; an older module
        # or a stand-in runs ``try_run``): a slow replay is waited for in
        # slices and served when it finishes late; only at the hard limit or
        # on a stopped request is it given up.
        run = getattr(engine, "run_patient", None)
        try:
            if run is not None:
                outputs, illegal = run(
                    entry.program, inputs, extra_refs=extra_refs,
                    slice_ms=2000, limit_ms=30_000, should_stop=should_stop,
                )
            else:
                outputs, illegal = engine.try_run(entry.program, inputs, extra_refs=extra_refs)
        except _runtime.ReplayLost as exc:
            # Committed, no outputs. The key is refused for the life of the process.
            self._refuse(entry, f"replay_lost_{exc.code}", repr(exc))
            if exc.wrote_inputs:
                # The live state is of unknown content, and after a timeout
                # the GPU may still be writing it: the MLX path must not run
                # on it. The request fails; its cache keeps the verifier's
                # adapters, which the session bank does not serve again.
                self._count("replay_lost")
                raise
            reason = f"refused:{entry.refused}"
            self._count(reason)
            return Ticket(self, entry, "fallback", reason)
        except Exception as exc:
            # Nothing was committed (an argument the module would not take).
            self._refuse(entry, "run_error", repr(exc))
            reason = f"refused:{entry.refused}"
            self._count(reason)
            return Ticket(self, entry, "fallback", reason)
        if illegal:
            # Not this call, and nothing was touched (for example the state
            # has another holder, so MLX itself would copy instead of writing
            # in place): MLX serves it.
            self._count(illegal)
            if illegal == "input_buffer_small":
                detail = getattr(engine, "last_short_detail", None)
                self._say("short", f"program {entry.digest}: a written input arrived in a "
                                   f"buffer shorter than the program needs "
                                   f"({detail() if detail is not None else 'no detail'}); MLX serves",
                          limit=3)
            return Ticket(self, entry, "fallback", illegal)
        ticket = Ticket(self, entry, "native", "")
        ticket.outputs = list(outputs)
        counters = self.counters
        counters["native"] += 1
        entry.native_runs += 1
        dispatched = counters["cut_dispatches"]
        for cut, count in entry.cut_dispatches.items():
            dispatched[cut] = dispatched.get(cut, 0) + count
        try:
            last = engine.last_run()
        except Exception:  # noqa: BLE001 - times are a report, never a reason to fail a call
            last = None
        now = time.perf_counter()
        if last:
            for name in ("launches", "command_buffers", "encode_ns", "gpu_ns", "cpu_ns"):
                counters[name] += int(last.get(name, 0) or 0)
            counters["inputs_grown"] += int(last.get("grown", 0) or 0)
            counters["inputs_grown_bytes"] += int(last.get("grown_bytes", 0) or 0)
            wall = int(last.get("wall_ns", 0) or 0)
            gpu = int(last.get("gpu_ns", 0) or 0)
            encode = int(last.get("encode_ns", 0) or 0)
            inputs_ns = int(inputs_s * 1e9)
            ticket._sample = (
                wall, gpu, encode, max(0, wall - gpu - encode),
                max(0, int((started - entered) * 1e9)), inputs_ns,
                max(0, int((now - started) * 1e9) - wall - inputs_ns),
            )
        counters["native_wall_s"] += now - started
        ticket._started = entered
        ticket._returned = now
        return ticket

    def _close_native(self, ticket: Ticket) -> None:
        """The caller has installed a native round's outputs: close its times."""

        sample, ticket._sample = ticket._sample, None
        if sample is None:
            return
        now = time.perf_counter()
        self._samples.append((
            *sample, max(0, int((now - ticket._returned) * 1e9)),
            max(0, int((now - ticket._started) * 1e9)),
        ))
        self.counters["timed"] += 1

    def _sample(self, engine: Any, samples: Any, counter: str, started: float | None) -> dict | None:
        """Keep the executor's times for a replay made beside the MLX path."""

        del started
        try:
            last = engine.last_run()
        except Exception:  # noqa: BLE001 - times are a report, never a reason to fail a call
            return None
        if not last:
            return None
        samples.append((
            int(last.get("wall_ns", 0) or 0), int(last.get("gpu_ns", 0) or 0),
            int(last.get("encode_ns", 0) or 0),
        ))
        self.counters[counter] += 1
        return last

    @staticmethod
    def _settle(engine: Any, inputs: list) -> None:
        """Wait until MLX's last command buffer has let go of the inputs.

        For a moment after an evaluation returns, MLX's command buffer still
        holds every array it produced, and MLX does not write into an array in
        that state: a recording made then contains a copy of a state array
        that the settled path writes in place.
        """

        settle = getattr(engine, "settle", None)
        if settle is not None:
            settle(inputs)

    def _abandon_open(self, engine: Any) -> None:
        """A call that never reached ``finish`` (the MLX path raised)."""

        ticket, self._open = self._open, None
        if ticket is None:
            return
        if ticket._described is not None:
            try:
                engine.record_end()
            except Exception as exc:
                self._say("abandoned_record", f"closing an abandoned recording raised {exc!r}")
        if ticket.kind == "record" and ticket.entry is not None:
            ticket.entry.state = "warming"

    # ------------------------------------------------------------------ finish

    def _finish(self, ticket: Ticket, outputs: list) -> None:
        if self._open is not ticket:
            return
        self._open = None
        engine = self._engine
        entry = ticket.entry
        assert entry is not None
        ir: ProgramIR | None = None
        try:
            mx.eval(*outputs)
            if ticket._described is not None:
                trace = engine.record_end()
                ir = engine.build(
                    trace,
                    inputs=ticket._described,
                    outputs=outputs,
                    name=f"{self.name}:{entry.digest}",
                )
        except _runtime.BuildRefused as exc:
            self._refuse(entry, exc.code, exc.detail)
            return
        except Exception as exc:
            if ticket._described is not None:
                try:
                    engine.record_end()
                except Exception:
                    pass
            self._refuse(entry, "build_error", repr(exc))
            return
        elapsed = time.perf_counter() - ticket._started
        if ticket.kind == "record":
            self.counters["record_s"] += elapsed
            self._take_recording(engine, ticket, entry, ir)
            return
        self.counters["verify_s"] += elapsed
        if ticket._in_place is not None:
            difference = self._compare_in_place(ticket, list(outputs))
        else:
            difference = compare_outputs(ticket._replayed or [], outputs)
        if difference is not None:
            self._dump_mismatch(entry, ticket, ir, outputs)
            self._diagnose(engine, entry, ticket, ir, outputs)
        ticket._replayed = None
        ticket._pristine = None
        ticket._in_place = None
        if ticket.kind == "shadow":
            self.counters["shadow_checks"] += 1
            if difference is not None:
                self.counters["shadow_mismatches"] += 1
                self._note_difference(entry, difference, shadow=True)
            return
        if difference is not None:
            self._note_difference(entry, difference, shadow=False)
            return
        if entry.checks_left > 1:
            # Settled from recordings that differed by MLX copies: the next
            # call checks it again before it goes live.
            entry.checks_left -= 1
            self.counters["copies_settled_rechecked"] += 1
            return
        entry.state = "live"
        self.counters["programs_live"] += 1
        self._note_steady(entry)
        self._say(
            "live",
            f"program {entry.digest} live: {entry.ir.launches()} launches, "
            f"{len(entry.clobbered)} inputs written in place, bit-identical on first use",
        )

    def _compare_in_place(self, ticket: Ticket, outputs: list) -> dict[str, Any] | None:
        """``compare_outputs`` for a check run on the live banks: every output
        the replay made in its own memory bit for bit; each bank by the chunks
        holding the round's rows (byte for byte) and by digest everywhere
        else; the small outputs over a bank against their copies."""

        from . import digest as _digest

        plan = ticket._in_place
        assert plan is not None
        replayed = list(ticket._replayed or [])
        candidate, reference = [], []
        covered: set[int] = set()
        for bank in plan.banks.values():
            covered.add(bank.output)
            for out in bank.small:
                covered.add(out)
                replayed[out] = bank.small_copies.get(out)
        for index, (got, want) in enumerate(zip(replayed, outputs)):
            if index in covered and all(index != bank.output for bank in plan.banks.values()):
                candidate.append(got)
                reference.append(want)
            elif index in covered:
                # Placeholders that agree, so the output numbers stay the call's own.
                candidate.append(mx.array(0, dtype=mx.uint8))
                reference.append(mx.array(0, dtype=mx.uint8))
            else:
                candidate.append(got)
                reference.append(want)
        if len(replayed) != len(outputs):
            return {"output": -1, "why": f"count {len(replayed)} != {len(outputs)}"}
        difference = compare_outputs(candidate, reference)
        if difference is not None:
            return difference
        engine = self._engine
        for bank in plan.banks.values():
            mlx_bank = outputs[bank.output]
            # The chunks are of a bank laid out row by row, as the replay's was.
            flags = int(engine.describe([mlx_bank])[0].flags) if engine is not None else 0
            if not _digest.supported(mlx_bank) or not flags & 2:
                return {"output": bank.output, "why": "bank layout", "flags": flags}
            after = _digest.digest(mlx_bank)
            rows = _digest.chunk_copy(mlx_bank, bank.chunks)
            mx.eval(after, rows)
            if not bool(mx.array_equal(rows, bank.native_window).item()):
                differing = mx.sum(rows != bank.native_window)
                mx.eval(differing)
                return {
                    "output": bank.output,
                    "why": "bits in the round's rows",
                    "elements": int(rows.size),
                    "elements_differing": int(differing.item()),
                }
            changed = _changed_chunks(bank.before, after, bank.chunks)
            if changed:
                # MLX wrote this bank outside the rows the caller named: the
                # call writes more than its windows say, and nothing proves
                # the replay wrote the same (its own writes were inside them).
                return {"output": bank.output, "why": "MLX wrote outside the named rows",
                        "chunks": changed[:8]}
        return None

    def _one_program(self, entry: _Entry, ir: ProgramIR) -> str:
        """The second recording against the first: "same", "shorter",
        "shorter_settled", "longer" or "refused" (the key was refused here).

        MLX writes a state bank in place only when nothing else holds it at
        the moment the write is encoded; otherwise it copies the whole bank
        first, one more launch. So one recording can be longer than another
        of the same geometry. The shorter one is the steady program. When the
        longer launches exactly the shorter's kernels plus MLX copies
        (``copies_only``), the pair settles on the shorter at once ("same"
        when the held one is the shorter, "shorter_settled" when the new one
        is); any other length difference is checked again (a few times) until
        two recordings agree.

        MLX leaves an arbitrary stride on a dimension of extent 1 and hands it
        to some kernels inside their arguments, in a form that follows the
        array's history (a result of the function itself, or the view a commit
        left). Both programs are right for both layouts, but they are not the
        same bytes. So two recordings are held to one program only when their
        inputs were laid out the same; otherwise the bit check of this call
        is the proof, and the call is counted as not comparable.
        """

        first = entry.ir
        same_layout = first.input_layout() == ir.input_layout()
        was, now = len(first.steps), len(ir.steps)
        difference: str | None = None
        if was == now:
            # Only a copies-settling site (the round program) names a constant
            # and a seed by their bytes; every other site calls the comparison
            # exactly as before.
            difference = (
                compare_structure(first, ir, same_bytes=True)
                if self._settle_copies
                else compare_structure(first, ir)
            )
            if difference is None:
                self.counters["stability_checked"] += 1
                return "same"
            if not same_layout:
                self.counters["stability_not_comparable"] += 1
                self._say("layout", f"second recording of {entry.digest} made at another "
                                    f"input layout, not compared: {difference}", limit=2)
                return "same"
            # One length, and the first launch where the two differ is a copy
            # in one of them: the copy MLX makes of an array it cannot write
            # in place (something still held it) moved to another array, so
            # each recording carries one transient copy where the steady
            # program has none (the real-pack refusal of 12 November: "launch
            # 546: kernel g1_copybfloat16bfloat16 against" a GDN kernel, after
            # which MLX served 2,377 rounds of that answer). That is the same
            # transient as a length difference and takes the same patience;
            # any other difference is the program's own and refuses the key.
            moved = copy_moved(first, ir)
            if moved is not None:
                what = f"have one length ({was} launches) and a copy in another place ({moved})"
                spent = f"{was} launches each, a copy in another place ({moved})"
                difference = None
        else:
            # Which kernels make the difference: a whole-bank copy reads as
            # one copy kernel per bank, anything else is another question.
            extra = launch_difference(first, ir)
            shorter, longer = (ir, first) if now < was else (first, ir)
            if self._settle_copies and copies_only(shorter, longer):
                # The longer one is the shorter plus copies of arrays something
                # (a command buffer still running) held when they were written:
                # the shorter is the program. One pair settles it, no rests.
                # The kernel names are all this compared, so the program is
                # bit-checked on two calls, not one: a value the recorder kept
                # as a constant that a call should have passed in shows when
                # the next call's inputs move.
                entry.checks_left = 2
                self.counters["copies_settled"] += 1
                self._say("canonical", f"recordings of {entry.digest} differ only by MLX copies "
                                       f"({was} launches, then {now}: {extra}); the shorter, written "
                                       f"in place, is the program", limit=4)
                return "shorter_settled" if now < was else "same"
            what = f"differ in length ({was} launches, then {now}: {extra})"
            spent = f"{was} launches against {now} ({extra})"
        if difference is None:
            entry.rechecks += 1
            if entry.rechecks <= _RECHECKS:
                self.counters["rechecks"] += 1
                self._say("recheck", f"recordings of {entry.digest} {what}; the shorter "
                                     f"is kept and the next call is recorded again", limit=4)
                return "shorter" if now < was else "longer"
            if entry.rests < _RESTS:
                entry.rests += 1
                entry.rechecks = 0
                entry.rest_until = entry.calls + _REST_CALLS
                self.counters["rests"] += 1
                self._say("rest", f"recordings of {entry.digest} still {what} after "
                                  f"{_RECHECKS} more; MLX serves its next {_REST_CALLS} calls, "
                                  f"then it is recorded again", limit=4)
                return "shorter" if now < was else "longer"
            difference = f"{spent}, after {_RECHECKS} more recordings {_RESTS + 1} times over"
        # Keep both recordings for whoever reads the refusal (the first was
        # never sealed, so nothing else has written it).
        second = self._dump_path(entry, "second.ir.json")
        if second is not None:
            second.write_text(ir.to_json())
            self._dump_path(entry, "first.ir.json").write_text(first.to_json())
        self._refuse(entry, "signature_unstable", f"second recording: {difference}")
        return "refused"

    def _take_recording(
        self, engine: Any, ticket: Ticket, entry: _Entry, ir: ProgramIR | None
    ) -> None:
        """A recorded call ended: hold its program, or seal the one held.

        The first recording of a geometry is held unsealed (``candidate``),
        unless a live program of the same step already launches exactly what
        it launches (``_steady_elsewhere``): then it is the steady program
        and is sealed here. Every later recording is compared with the one
        held: the same program seals it; a shorter one replaces it (and is
        sealed at once when a live program of the step vouches for it); a
        longer one is this call's own extra launches and changes nothing.
        Recordings that keep disagreeing refuse the key. The seal falls at the
        end of the recorded call, after its outputs are out, so the call
        after it is the bit check. Dump-only (replay switched off) takes the
        first recording as it is: the dump is what that run is for.
        """

        if entry.state == "candidate" and replay_allowed():
            if ir is None:
                self._refuse(entry, "build_error", "no program")
                return
            verdict = self._one_program(entry, ir)
            if verdict == "same":
                self._seal(engine, ticket._key, entry)
            elif verdict == "shorter":
                self.counters["programs_rerecorded"] += 1
                if self._hold_recording(entry, ir) and self._steady_elsewhere(entry, ir):
                    self._seal(engine, ticket._key, entry)
            elif verdict == "shorter_settled":
                self.counters["programs_rerecorded"] += 1
                if self._hold_recording(entry, ir):
                    self._seal(engine, ticket._key, entry)
            return
        if not self._hold_recording(entry, ir):
            return
        if not replay_allowed() or self._steady_elsewhere(entry, ir):
            self._seal(engine, ticket._key, entry)

    @staticmethod
    def _launched(ir: ProgramIR) -> bytes:
        """What a program launches, in order: a digest of its kernels' names."""

        names = "\n".join(step.kernel_name or "" for step in ir.steps)
        return hashlib.blake2b(names.encode("utf-8"), digest_size=16).digest()

    def _steady_elsewhere(self, entry: _Entry, ir: ProgramIR) -> bool:
        """True (and counted) when a program of this key's step that passed
        its bit check launches the kernels of ``ir`` in the same order.

        The second recording of a key exists for one question: does the first
        one carry launches the steady path does not make (the copy MLX makes
        of a state bank something still holds)? Such a recording is longer
        than the steady program by that copy, or has it where the steady one
        has another kernel. A program of the same step at another bank
        capacity that went live was itself seen twice and replayed bit for
        bit, so it is the steady program of the step, and a recording with
        its kernels in its order carries no such copy. Whether the program
        computes the right bits is not decided here: that stays the checked
        call's, as for every program. Sizes, strides and grids are not
        compared (they are what a capacity changes); a step that launches
        other kernels at another capacity simply is not vouched for and is
        recorded twice, as before.
        """

        known = self._steady.get(entry.kin)
        if not known or self._launched(ir) not in known:
            return False
        self.counters["steady_by_kin"] += 1
        return True

    def _note_steady(self, entry: _Entry) -> None:
        """A program went live: remember what it launches, for its step."""

        if entry.ir is None:
            return
        known = self._steady.get(entry.kin)
        if known is None:
            known = self._steady[entry.kin] = set()
            while len(self._steady) > _REMEMBERED_FAMILIES:
                self._steady.popitem(last=False)
        else:
            self._steady.move_to_end(entry.kin)
        known.add(self._launched(entry.ir))

    def _hold_recording(self, entry: _Entry, ir: ProgramIR | None) -> bool:
        if ir is None:
            self._refuse(entry, "build_error", "no program")
            return False
        problems = ir.check()
        if problems:
            self._refuse(entry, "ir_malformed", "; ".join(problems[:3]))
            return False
        entry.ir = ir
        entry.signature = ""  # hashed only when a dump asks for it (tens of ms in Python)
        by_id = {buf.id: buf for buf in ir.buffers}
        entry.clobbered = tuple(
            index for index, arr in enumerate(ir.inputs)
            # (-1: an input no launch binds, never written)
            if arr.buffer in by_id and by_id[arr.buffer].clobbered
        )
        if not self._fence_strided and any(
            not int(ir.inputs[index].flags) & 2 for index in entry.clobbered
        ):
            # (See ``fence_strided``: this site reads strided inputs, and a
            # program that writes into one is the case the fence exists for.)
            self._refuse(entry, "writes_strided_input", f"inputs {list(entry.clobbered)[:8]}")
            return False
        entry.owned_bytes = self._owned_bytes(ir)
        entry.state = "candidate"
        return True

    @staticmethod
    def _owned_bytes(ir: ProgramIR, program: Any = None) -> int:
        """What one program adds to the process (see ``_DEFAULT_PROGRAM_BYTES``).

        Its constants on the GPU (the sealed program's own count when it has
        one), their bytes again in the recording, and its launches' host
        memory. Not its temporaries (one pool for all programs) and not its
        outputs (the round's own, allocated per run).
        """

        const = getattr(getattr(program, "native", None), "const_bytes", None)
        if const is None:
            const = sum(int(buf.nbytes) for buf in ir.buffers if buf.role in ("const", "seed"))
        held = sum(len(buf.data) for buf in ir.buffers if buf.data)
        return int(const) + held + _LAUNCH_HOST_BYTES * len(ir.steps)

    def _seal(self, engine: Any, key: Any, entry: _Entry) -> None:
        """The held recording becomes a program (working memory, constants).

        At the end of the recorded call that settled it, after that call's
        outputs are out: host time on one round (``seal_s`` in the report;
        about a round and a half on the Flash-Next step), where it used to be
        a call of its own that MLX served. The bit check is the next call's.
        """

        started = time.perf_counter()
        try:
            self._dump(entry, key)
            entry.program = engine.seal(entry.ir)
            if self._dispatch is not None:
                entry.program.dispatch, entry.program.barriers_form = self._dispatch
            if self._split is not None:
                entry.program.split = self._split
        except _runtime.BuildRefused as exc:
            self._refuse(entry, exc.code, exc.detail)
        except Exception as exc:
            self._refuse(entry, "seal_error", repr(exc))
        else:
            entry.state = "recorded"
            # Dispatches the program makes of the kernels a model family wants
            # counted in the report: none named on this line, so only the
            # indirect ones (their grids are written by the GPU).
            entry.cut_dispatches = {}
            indirect = sum(step.indirect_grid is not None for step in entry.ir.steps)
            if indirect:
                entry.cut_dispatches["indirect_dispatches"] = indirect
            entry.owned_bytes = self._owned_bytes(entry.ir, entry.program)
            self.counters["programs_built"] += 1
            self._evict_programs()
        self.counters["seal_s"] += time.perf_counter() - started

    def _note_difference(self, entry: _Entry, difference: dict[str, Any], *, shadow: bool) -> None:
        index = int(difference.get("output", -1))
        name = self._output_names(index) if self._output_names and index >= 0 else f"output[{index}]"
        detail = {**difference, "name": name, "key": entry.digest}
        if shadow:
            self.last_refusal = {"code": "shadow_mismatch", **detail}
            self._say("shadow", f"SHADOW MISMATCH {json.dumps(detail)}", limit=20)
            return
        self._refuse(entry, "bit_mismatch", json.dumps(detail))

    def _refuse(self, entry: _Entry, code: str, detail: str) -> None:
        was_live = entry.state == "live"
        entry.state = "refused"
        entry.refused = str(code)
        entry.detail = str(detail)[:600]
        entry.program = None
        entry.ir = None  # (what a reader needs was dumped before the refusal)
        entry.owned_bytes = 0
        refused = self.counters["programs_refused"]
        refused[entry.refused] = refused.get(entry.refused, 0) + 1
        if was_live:
            self.counters["programs_live"] -= 1
        self.last_refusal = {"code": entry.refused, "detail": entry.detail, "key": entry.digest}
        self._say("refused", f"program {entry.digest} REFUSED ({entry.refused}): {entry.detail}", limit=10)

    # ------------------------------------------------------------ housekeeping

    @staticmethod
    def _warm_calls() -> int:
        raw = os.environ.get(WARM_ENV, "").strip()
        try:
            return max(1, int(raw)) if raw else WARM_CALLS
        except ValueError:
            return WARM_CALLS

    def _program_limit(self) -> int:
        raw = os.environ.get(PROGRAMS_ENV, "").strip()
        try:
            return max(1, int(raw)) if raw else _DEFAULT_PROGRAMS
        except ValueError:
            return _DEFAULT_PROGRAMS

    def _program_budget(self) -> int:
        raw = os.environ.get(PROGRAM_BYTES_ENV, "").strip()
        try:
            return max(1, int(raw)) << 20 if raw else _DEFAULT_PROGRAM_BYTES
        except ValueError:
            return _DEFAULT_PROGRAM_BYTES

    def _evict_programs(self) -> None:
        """Keep the most recently used programs inside the byte budget (and
        the count limit); an evicted key starts again. The newest stays.

        A recording held but not sealed yet counts as its program does: it is
        most of what a program owns."""

        limit, budget = self._program_limit(), self._program_budget()
        holders = [key for key, e in self._entries.items() if e.ir is not None]
        owned = sum(self._entries[key].owned_bytes for key in holders)
        while len(holders) > 1 and (len(holders) > limit or owned > budget):
            key = holders.pop(0)
            entry = self._entries.pop(key)
            owned -= entry.owned_bytes
            if entry.state == "live":
                self.counters["programs_live"] -= 1
            if entry.program is not None:
                self.counters["programs_evicted"] += 1
            entry.program = None
            entry.ir = None
            entry.state = "evicted"  # (a geometry that still lists it starts again)

    def _trim_entries(self) -> None:
        while len(self._entries) > _REMEMBERED_REFUSALS:
            key, entry = next(iter(self._entries.items()))
            if entry.state == "live":
                self.counters["programs_live"] -= 1
            entry.program = None
            entry.ir = None
            entry.state = "evicted"
            del self._entries[key]

    def _mixed_declines(self) -> bool:
        seed = os.environ.get(MIXED_ENV, "").strip()
        if not seed:
            return False
        if self._mixed_rng is None or seed != self._mixed_seed:
            self._mixed_rng = random.Random(f"{self.name}:{seed}")
            self._mixed_seed = seed
        return self._mixed_rng.random() < 0.5

    # --------------------------------------------------------- dump and profile

    def _dump_path(self, entry: _Entry, suffix: str):
        folder = _runtime.dump_dir()
        if folder is None:
            return None
        folder.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)
        return folder / f"{safe}-{entry.digest}.{suffix}"

    def _dump(self, entry: _Entry, key: Any) -> None:
        path = self._dump_path(entry, "ir.json")
        if path is None or entry.ir is None:
            return
        ir = entry.ir
        entry.signature = ir.signature()
        obj = ir.to_obj()
        if not _env_flag(DUMP_DATA_ENV):
            # Constants can be megabytes of weights; tools need their size and
            # identity, not their bytes.
            for raw, buf in zip(obj["buffers"], ir.buffers):
                if buf.data is not None and len(buf.data) > 256:
                    raw.pop("data", None)
                    raw["data_len"] = len(buf.data)
                    raw["data_blake2b"] = hashlib.blake2b(buf.data, digest_size=8).hexdigest()
        path.write_text(json.dumps(obj))
        by_id = {buf.id: buf for buf in ir.buffers}
        summary = {
            "site": self.name,
            "key": repr(key),
            "key_digest": entry.digest,
            "signature": entry.signature,
            "launches": ir.launches(),
            "kernel_counts": ir.kernel_counts(),
            "bytes_by_role": ir.bytes_by_role(),
            "buffers_by_role": {
                role: sum(1 for buf in ir.buffers if buf.role == role)
                for role in sorted({buf.role for buf in ir.buffers})
            },
            "inputs": len(ir.inputs),
            "outputs": len(ir.outputs),
            "inputs_written_in_place": [
                {"input": index, "shape": list(arr.shape), "dtype": arr.dtype, "nbytes": arr.nbytes}
                for index, arr in enumerate(ir.inputs) if by_id[arr.buffer].clobbered
            ],
            "outputs_in_input_buffers": {str(k): v for k, v in sorted(ir.output_alias.items())},
            "marks": sorted({step.mark for step in ir.steps if step.mark}),
            "barriers": sum(1 for step in ir.steps if step.barrier_before),
            "recorded_at_call": entry.calls,
        }
        self._dump_path(entry, "summary.json").write_text(json.dumps(summary, indent=1))
        self._say("dump", f"program {entry.digest} dumped to {path} ({ir.launches()} launches)")

    def _dump_mismatch(self, entry: _Entry, ticket: Ticket, ir: ProgramIR | None, outputs: list) -> None:
        """A replay differed from the MLX path: leave what a reader needs.

        Into the dump folder (nothing without one): `<key>.mismatch.json`
        with every differing output and where, the layout of every input at
        the recorded call and at this call, and which inputs share a buffer;
        and, for a first-use check, this call's own recording as
        `<key>.second.ir.json`, to be set against the first.
        """

        path = self._dump_path(entry, "mismatch.json")
        if path is None:
            return
        try:
            def layouts(program: ProgramIR | None) -> list | None:
                if program is None:
                    return None
                return [
                    {"input": index, "buffer": arr.buffer, "offset": arr.offset, "shape": list(arr.shape),
                     "dtype": arr.dtype, "strides": list(arr.strides), "nbytes": arr.nbytes,
                     "data_size": arr.data_size, "flags": arr.flags}
                    for index, arr in enumerate(program.inputs)
                ]

            def shared(program: ProgramIR | None) -> list | None:
                if program is None:
                    return None
                by_buffer: dict[int, list[int]] = {}
                for index, arr in enumerate(program.inputs):
                    by_buffer.setdefault(arr.buffer, []).append(index)
                return [members for members in by_buffer.values() if len(members) > 1]

            report = {
                "site": self.name,
                "key_digest": entry.digest,
                "kind": ticket.kind,
                "call": entry.calls,
                "launches_first": len(entry.ir.steps) if entry.ir is not None else None,
                "launches_this_call": len(ir.steps) if ir is not None else None,
                "inputs_written_by_program": list(entry.clobbered),
                "differences": describe_differences(ticket._replayed or [], outputs),
                "inputs_at_recording": layouts(entry.ir),
                "inputs_at_this_call": layouts(ir),
                "inputs_sharing_a_buffer_at_recording": shared(entry.ir),
                "inputs_sharing_a_buffer_at_this_call": shared(ir),
            }
            path.write_text(json.dumps(report, indent=1, default=str))
            if ir is not None:
                second = self._dump_path(entry, "second.ir.json")
                if second is not None:
                    second.write_text(ir.to_json())
            self._say("mismatch", f"mismatch of {entry.digest} written to {path}")
        except Exception as exc:  # noqa: BLE001 - a report never fails a round
            self._say("mismatch", f"mismatch report of {entry.digest} failed: {exc!r}")

    def _diagnose(self, engine: Any, entry: _Entry, ticket: Ticket, ir: ProgramIR | None, outputs: list) -> None:
        """MTPLX_NATIVE_ENGINE_DIAGNOSE=1: agent A's stepped replays name the
        launch behind a mismatch (`<key>.diagnosis.json`, one log line)."""

        pristine = ticket._pristine
        if pristine is None:
            return
        try:
            from . import diagnose as _diagnose

            report = _diagnose.explain(
                engine, entry.ir, pristine,
                second=ir,  # this call's own recording; None in shadow mode
                reference=outputs,  # the MLX path's outputs of this call
                name=f"{self.name}:{entry.digest}",
            )
            path = self._dump_path(entry, "diagnosis.json")
            _diagnose.write_report(report, path)
            self._say("diagnosis", f"{entry.digest}: {_diagnose.summary(report)} ({path})", limit=8)
        except Exception as exc:  # noqa: BLE001 - a diagnosis never fails a round
            self._say("diagnosis", f"diagnosis of {entry.digest} failed: {exc!r}", limit=8)

    def _maybe_profile(
        self, engine: Any, entry: _Entry, inputs: list, admit: Callable[[int], bool] | None
    ) -> None:
        if entry.profiled or not _profile_mode() or _runtime.dump_dir() is None:
            return
        try:
            mx.eval(*inputs)
            replay_inputs = self._replay_inputs(engine, entry, inputs, admit)
            if replay_inputs is None:
                return
            self._profile(engine, entry, replay_inputs)
            del replay_inputs
            self._census_extras(engine, entry, inputs, admit)
        except Exception as exc:
            entry.profiled = True
            self._say("profile", f"profile of {entry.digest} failed: {exc!r}")

    def _census_extras(
        self, engine: Any, entry: _Entry, inputs: list, admit: Callable[[int], bool] | None
    ) -> None:
        """With ``MTPLX_NATIVE_ENGINE_PROFILE=forms``: the pinned buffers' names
        and the program timed in every form of ``_CENSUS_FORMS``. Diagnostic:
        a failure is said once and never fails the round."""

        if _profile_mode() != "forms":
            return
        try:
            self._write_pins(engine, entry)
        except Exception as exc:  # noqa: BLE001 - a census never fails a round
            self._say("census", f"pin names of {entry.digest} failed: {exc!r}")
        try:
            self._time_forms(engine, entry, inputs, admit)
        except Exception as exc:  # noqa: BLE001 - a census never fails a round
            self._say("census", f"form timing of {entry.digest} failed: {exc!r}")

    def _write_pins(self, engine: Any, entry: _Entry) -> None:
        """``<key>.pins.json``: pin slot -> the parameter names held there."""

        path = self._dump_path(entry, "pins.json")
        if path is None or entry.ir is None or self._pin_names is None:
            return
        wanted = {buf.index for buf in entry.ir.buffers if buf.role == "pinned"}
        slots: dict[str, list[str]] = {}
        for name, array in self._pin_names():
            if not hasattr(array, "nbytes"):
                continue
            try:
                ref = engine.describe([array])[0]
            except Exception:  # noqa: BLE001 - an array never evaluated was never pinned
                continue
            slot = engine.pin_slot(ref.ptr) if ref.ptr else -1
            if slot in wanted:
                slots.setdefault(str(slot), []).append(str(name))
        path.write_text(json.dumps({
            "site": self.name,
            "key_digest": entry.digest,
            "pinned_buffers": len(wanted),
            "named": len(slots),
            "slots": slots,
        }))

    def _time_forms(
        self, engine: Any, entry: _Entry, inputs: list, admit: Callable[[int], bool] | None
    ) -> None:
        """``<key>.forms.json``: the program's GPU and wall time in each form of
        ``_CENSUS_FORMS``, every replay on fresh private copies, its outputs
        dropped.

        Every form runs with the working memory and fresh outputs zeroed
        (``fill=0``): without barriers a launch can read an index buffer before
        the launch that writes it, and a recycled buffer's bytes read as an
        expert id or an offset would index out of bounds; zero is in range.
        The serving form is timed once more without the fill (what serving
        pays). The forms are interleaved, one replay of each in turn, so a GPU
        clock that ramps or is cut over the census weighs on every form alike.
        Before and after, the serving form replays on fresh copies and the two
        results are compared bit for bit; a difference or a stuck engine marks
        the file ``floor_suspect`` and says so.

        Short context only: every replay copies each input the program writes,
        the KV banks included (about 150 MB at 4K tokens on the 35B, linear in
        context), 115 times. The census runs inside the first verify of each
        key, so that request's times are not serving figures.
        """

        path = self._dump_path(entry, "forms.json")
        if path is None or entry.program is None:
            return
        program = entry.program
        rows: list[dict[str, Any]] = []
        encode: dict[str, Any] = {}
        copies = self._replay_inputs(engine, entry, inputs, admit)
        if copies is not None:
            try:
                encode = engine.null_run(program, copies)
            except Exception as exc:  # noqa: BLE001 - reported in the file
                encode = {"error": repr(exc)[:200]}
            del copies
        reference = self._census_serving_outputs(engine, entry, inputs, admit)
        forms = [(*form, 0) for form in _CENSUS_FORMS] + [(*_CENSUS_FORMS[0], None)]
        figures: list[list[dict]] = [[] for _ in forms]
        declined = [""] * len(forms)
        for rep in range(1 + _CENSUS_REPS):
            for index, (dispatch, barriers, split, fill) in enumerate(forms):
                if declined[index]:
                    continue
                copies = self._replay_inputs(engine, entry, inputs, admit)
                if copies is None:
                    declined[index] = "no_memory_for_copies"
                    continue
                stats, reason = engine.time_form(
                    program, copies, dispatch=dispatch, barriers=barriers, split=split, fill=fill
                )
                del copies
                if stats is None:
                    declined[index] = reason or "declined"
                elif rep:
                    figures[index].append(stats)
        after = self._census_serving_outputs(engine, entry, inputs, admit)
        if reference is None or after is None:
            check: dict[str, Any] = {"compared": False}
        else:
            difference = compare_outputs(after, reference)
            check = {"compared": True, "equal": difference is None}
            if difference is not None:
                check["first_difference"] = difference
        check["engine_stuck"] = bool(engine.stuck())
        suspect = check["engine_stuck"] or not check.get("equal", False)
        for (dispatch, barriers, split, fill), reason, timed in zip(forms, declined, figures):
            row: dict[str, Any] = {"dispatch": dispatch, "barriers": barriers, "split": split, "fill": fill}
            if reason:
                row["declined"] = reason
            if timed:
                for column in ("gpu_ns", "wall_ns", "encode_ns", "wait_ns"):
                    values = sorted(int(f.get(column, 0) or 0) for f in timed)
                    row[column] = {"min": values[0], "median": values[len(values) // 2], "max": values[-1]}
                last = timed[-1]
                row["command_buffers"] = int(last.get("command_buffers", 0) or 0)
                row["passes"] = int(last.get("passes", 0) or 0)
                row["barriers_encoded"] = int(last.get("barriers", 0) or 0)
            rows.append(row)
        path.write_text(json.dumps({
            "site": self.name,
            "key_digest": entry.digest,
            "launches": program.launches,
            "barriers_by_form": dict(getattr(program, "barriers", {}) or {}),
            "reps": _CENSUS_REPS,
            "order": "interleaved",
            "fill": 0,
            "null_encoder": encode,
            "after_check": check,
            "floor_suspect": suspect,
            "forms": rows,
        }, indent=1))
        if suspect:
            self._say("census", f"forms of {entry.digest}: the serving replay after the barrier-free forms "
                      f"was not proven unchanged ({check}); the floor is suspect ({path})", limit=8)
        else:
            self._say("census", f"forms of {entry.digest} timed into {path}", limit=8)

    def _census_serving_outputs(
        self, engine: Any, entry: _Entry, inputs: list, admit: Callable[[int], bool] | None
    ) -> list | None:
        """The program's outputs in its serving form on fresh copies (None when
        the copies or the replay are refused): the census's before-and-after
        proof that the barrier-free forms left nothing behind."""

        copies = self._replay_inputs(engine, entry, inputs, admit)
        if copies is None or entry.program is None:
            return None
        outputs, _reason = engine.try_run(entry.program, copies)
        del copies
        if outputs is None:
            return None
        mx.eval(*outputs)
        return outputs

    def _profile(self, engine: Any, entry: _Entry, replay_inputs: list) -> None:
        """Per-step GPU time of the program, once per key, on private copies."""

        if entry.profiled or not _profile_mode():
            return
        entry.profiled = True
        path = self._dump_path(entry, "profile.json")
        if path is None:
            return
        try:
            steps = engine.profile(entry.program, replay_inputs)
        except Exception as exc:
            self._say("profile", f"profile of {entry.digest} failed: {exc!r}")
            return
        by_kernel: dict[str, dict[str, float]] = {}
        total = 0
        for step in steps:
            name = str(step.get("kernel_name") or step.get("kernel") or "?")
            ns = int(step.get("gpu_ns", 0) or 0)
            total += ns
            slot = by_kernel.setdefault(name, {"launches": 0, "gpu_ns": 0})
            slot["launches"] += 1
            slot["gpu_ns"] += ns
        ranked = dict(sorted(by_kernel.items(), key=lambda item: -item[1]["gpu_ns"]))
        path.write_text(json.dumps({
            "site": self.name,
            "key_digest": entry.digest,
            "launches": len(steps),
            "gpu_ns": total,
            "by_kernel": ranked,
            "steps": steps,
        }))
        self._say("profile", f"profile of {entry.digest} written to {path}")


__all__ = [
    "ReplaySite",
    "Ticket",
    "compare_outputs",
    "private_copy",
    "replay_allowed",
    "key_digest",
    "compare_structure",
    "copy_moved",
    "launch_difference",
]
