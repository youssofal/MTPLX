"""A stand-in executor for the site policy tests.

It "replays" a recording by calling the recorded function again, so the policy
of ``mtplx.engine.site.ReplaySite`` (which call is recorded, which is checked,
what is counted, what is refused) can be tested without the compiled module.
Faults are injected by name.
"""

from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx

from mtplx.engine.site import _COPY_KERNEL as _COPY_NAME
from mtplx.engine.ir import ArrayIR, BufferIR, ProgramIR, StepIR
from mtplx.engine.runtime import BuildRefused


class FakeEngine:
    def __init__(
        self,
        fn: Callable[[list], list],
        *,
        clobbered: tuple[int, ...] = (),
    ) -> None:
        self.fn = fn
        self.clobbered = tuple(clobbered)
        self.recording: dict[str, Any] | None = None
        self.recordings = 0
        self.builds = 0
        self.runs = 0
        self.run_inputs: list[list] = []
        self.pinned: list = []
        self.profiles = 0
        self.settles = 0
        self.extra_refs_seen: list[int] = []
        # faults
        self.build_refuse: str | None = None
        self.unstable_signature = False  # every build has one more launch than the last
        self.unstable_inline = False  # every build has other inline bytes
        self.launches_by_build: list[int] | None = None  # launches of build 1, 2, ...
        self.kernels_by_build: list[list[str]] | None = None  # kernel names of build 1, 2, ...
        self.wrong_on_run: int | None = None  # 1-based run number that returns wrong bits
        self.wrong_always = False
        self.can_run_reason = ""
        self.run_raises = False
        self.record_raises = False

    # -- recording -----------------------------------------------------------

    def describe(self, arrays: list) -> list:
        return [(index, tuple(a.shape), str(a.dtype)) for index, a in enumerate(arrays)]

    def record_begin(self, *, label: str = "", inputs: list | None = None) -> None:
        if self.record_raises:
            raise RuntimeError("recorder refused")
        if self.recording is not None:
            raise RuntimeError("a recording is already open")
        self.recording = {"label": label, "inputs": inputs}

    def record_mark(self, label: str) -> None:
        pass

    def record_end(self) -> Any:
        if self.recording is None:
            raise RuntimeError("no recording is open")
        trace, self.recording = self.recording, None
        self.recordings += 1
        return trace

    # -- building ------------------------------------------------------------

    def pin(self, arrays: list) -> None:
        self.pinned.extend(arrays)

    def build(self, trace: Any, *, inputs: list, outputs: list, name: str = "") -> ProgramIR:
        if self.build_refuse:
            raise BuildRefused(self.build_refuse, "step 3 (some_kernel) reads an undeclared buffer")
        self.builds += 1
        buffers = []
        input_arrays = []
        for index, (_i, shape, dtype) in enumerate(inputs):
            buffers.append(
                BufferIR(id=index, role="input", nbytes=64, index=index,
                         clobbered=index in self.clobbered)
            )
            input_arrays.append(ArrayIR(buffer=index, offset=0, shape=shape, dtype=dtype))
        base = len(buffers)
        output_arrays = []
        for index, out in enumerate(outputs):
            buffers.append(BufferIR(id=base + index, role="output", nbytes=64, index=index))
            output_arrays.append(
                ArrayIR(buffer=base + index, offset=0, shape=tuple(out.shape), dtype=str(out.dtype))
            )
        const_id = base + len(outputs)
        buffers.append(BufferIR(id=const_id, role="const", nbytes=4096, data=b"\x01" * 1024))
        launches = 3 + (self.builds if self.unstable_signature else 0)
        if self.launches_by_build is not None:
            launches = self.launches_by_build[min(self.builds, len(self.launches_by_build)) - 1]
        inline = bytes([self.builds % 251, 1]) if self.unstable_inline else b"\x00\x01"
        names = [f"fake_kernel_{n % 2}" for n in range(launches)]
        if self.kernels_by_build is not None:
            names = list(self.kernels_by_build[min(self.builds, len(self.kernels_by_build)) - 1])
        steps = []
        for name in names:
            if _COPY_NAME.match(name):
                # An MLX copy as MLX launches it: one source read, one fresh destination written
                # (an array it could not write in place, copied aside); the program reads on as before.
                temp = len(buffers)
                buffers.append(BufferIR(id=temp, role="temp", nbytes=64))
                steps.append(StepIR(kernel=1, kernel_name=name, bindings=[(0, 0, 0, "r"), (1, temp, 0, "w")],
                                    inline=[], dispatch=("threads", (4, 1, 1), (4, 1, 1))))
                continue
            steps.append(
                StepIR(
                    kernel=1,
                    kernel_name=name,
                    bindings=[(0, 0, 0, "r"), (1, base, 0, "w"), (2, const_id, 0, "r")],
                    inline=[(3, inline)],
                    dispatch=("threads", (4, 1, 1), (4, 1, 1)),
                )
            )
        return ProgramIR(
            name=name, buffers=buffers, steps=steps, inputs=input_arrays, outputs=output_arrays,
        )

    def seal(self, ir: ProgramIR) -> Any:
        return {"ir": ir}

    # -- running -------------------------------------------------------------

    def can_run(self, program: Any, inputs: list) -> str:
        return self.can_run_reason

    def try_run(self, program: Any, inputs: list, *, profile: bool = False, extra_refs: int = 1):
        self.extra_refs_seen.append(extra_refs)
        if self.can_run_reason:
            return None, self.can_run_reason
        return self.run(program, inputs), ""

    def settle(self, arrays: list, *, timeout_ms: float = 5.0) -> bool:
        self.settles += 1
        return True

    def run(self, program: Any, inputs: list, *, then_prepare: Any = None) -> list:
        self.runs += 1
        self.run_inputs.append(list(inputs))
        if self.run_raises:
            raise RuntimeError("command buffer failed")
        outputs = list(self.fn(list(inputs)))
        mx.eval(*outputs)
        if self.wrong_always or self.wrong_on_run == self.runs:
            first = outputs[0]
            # One bit of one element.
            flipped = first.reshape(-1)
            bits = flipped.view(mx.uint32 if first.dtype == mx.float32 else mx.uint16)
            bits = mx.concatenate([bits[:1] ^ 1, bits[1:]])
            outputs[0] = bits.view(first.dtype).reshape(first.shape)
            mx.eval(outputs[0])
        return outputs

    def last_run(self) -> dict:
        return {"launches": 3, "command_buffers": 1, "encode_ns": 2000, "gpu_ns": 5000,
                "wall_ns": 9000, "cpu_ns": 1000}

    def profile(self, program: Any, inputs: list) -> list[dict]:
        self.profiles += 1
        self.run_inputs.append(list(inputs))
        return [
            {"kernel_name": step.kernel_name, "gpu_ns": 1000 * (n + 1)}
            for n, step in enumerate(program["ir"].steps)
        ]

    def counters(self) -> dict:
        return {"runs": self.runs}

    def synchronize(self) -> None:
        pass
