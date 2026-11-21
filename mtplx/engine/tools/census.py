"""Launch census: what an evaluation launches on the GPU, by block and by kernel.

A launch is one Metal dispatch.  The census answers "how many, of which
kernels, in which part of the round", which is the work list for cutting
launches and the check after every cut.

Three sources feed one table:

* ``dump``: a directory of ProgramIR JSON files written by the native executor
  (``MTPLX_NATIVE_ENGINE_DUMP``).  Works for any model, the real pack included,
  because the files are plain data.
* ``engine``: a live recording through the executor's recorder
  (``record_begin`` / ``record_mark`` / ``record_end``).
* ``dtrace``: a live tape of MLX's Metal encoder taken with DTrace (every
  dispatch MLX makes on its compute encoder, named by the kernel MLX looked up
  last on the same thread).  It needs ``sudo -n dtrace`` and is for developer
  Macs: an independent count, and the count before the executor is built.

In code::

    from mtplx.engine.tools import census
    with census.tape() as t:          # engine recorder when it runs, else dtrace
        t.mark("verify")
        mx.eval(verify(...))
        t.mark("draft")
        mx.eval(draft(...))
    print(census.render(t.result()))

Command line::

    python -m mtplx.engine.tools.census dump DIR [--out FILE.md]
    python -m mtplx.engine.tools.census synth [--rounds 6] [--recipe bare-speed] [--out FILE.md]

A launch count is never a speed claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

HOLD_ENV = "MTPLX_TOOLS_HOLD"  # colon-separated files; while one exists no GPU job starts
MARK_PREFIX = "/.mtplx-census/"


def held() -> str:
    """The hold file that exists now, or ''."""

    for raw in (os.environ.get(HOLD_ENV) or "").split(":"):
        if raw and os.path.exists(raw):
            return raw
    return ""


def refuse_if_held() -> None:
    path = held()
    if path:
        raise SystemExit(f"census: {path} exists, not starting")


# ------------------------------------------------------------------ the tape


@dataclass
class Launch:
    kernel: str
    mark: str = ""
    kind: str = ""  # "groups" | "threads" | ""
    grid: tuple[int, int, int] = (0, 0, 0)
    group: tuple[int, int, int] = (0, 0, 0)

    def threads(self) -> int:
        """Threads the launch starts (0 when the source carries no sizes)."""

        x, y, z = self.grid
        n = x * y * z
        if self.kind == "groups":
            tx, ty, tz = self.group
            n *= tx * ty * tz
        return n


@dataclass
class Result:
    source: str
    launches: list[Launch] = field(default_factory=list)
    command_buffers: int | None = None
    notes: list[str] = field(default_factory=list)

    def marks(self) -> list[str]:
        seen: list[str] = []
        for launch in self.launches:
            if launch.mark not in seen:
                seen.append(launch.mark)
        return seen

    def by_mark(self) -> dict[str, list[Launch]]:
        out: dict[str, list[Launch]] = {}
        for launch in self.launches:
            out.setdefault(launch.mark, []).append(launch)
        return out

    def counts(self, mark: str | None = None) -> Counter:
        return Counter(
            launch.kernel for launch in self.launches if mark is None or launch.mark == mark
        )

    def to_obj(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "command_buffers": self.command_buffers,
            "notes": self.notes,
            "launches": [
                [l.mark, l.kernel, l.kind, list(l.grid), list(l.group)] for l in self.launches
            ],
        }

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "Result":
        return cls(
            source=str(obj.get("source", "")),
            command_buffers=obj.get("command_buffers"),
            notes=list(obj.get("notes", [])),
            launches=[
                Launch(kernel=k, mark=m, kind=kind, grid=tuple(grid), group=tuple(group))
                for m, k, kind, grid, group in obj.get("launches", [])
            ],
        )


# -------------------------------------------------------------- kernel families

# First match wins.  Names are Metal function names: MLX's own kernels, the
# kernels of mtplx/kernels (MLX prefixes them ``custom_kernel_``), and the
# elementwise chains mx.compile fuses (named by their op letters and a hash).
FAMILIES: list[tuple[str, str]] = [
    ("hyper-connection read", r"mtplx_hc_|mtplx_hyper"),
    ("gdn", r"gated_delta|mtplx_gdn|gdn_|depthwise_conv|implicit_gemm_conv"),
    ("routed experts", r"m4_route|routed_glu|routed_down|moe_glu|mtplx_moe_|gather_qmv|gather_qmm|gather_mm"),
    ("attention", r"sdpa|mtplx_qsa|m4_rope|qsa_"),
    ("head and sampling", r"lm_head|logits_topk|mbsort|block_sort|logsumexp|softmax|scan_|categorical"),
    ("quantized product", r"affine_q|qmv4|verify_qmv|ternary"),
    ("dense product", r"gemv|steel_gemm|gemm"),
    ("norm", r"^rms|layer_norm"),
    ("fused elementwise", r"^[A-Z][A-Za-z0-9]*_[VSC]+_.*(contiguous|strided)|V2I|f4I|i4I"),
    ("copy", r"copy"),
    ("index build", r"arange|compute_dynamic_offset|gather_front|gather_axis|scatter|eye|full_|zeros"),
    ("reduce", r"reduce|all_|any_"),
    ("elementwise", r"^(v|vv|vs|sv|ss|vn|vvn|vsn|svn|gn?\d|g\d)_|^g\d_|Sigmoid|Multiply|Add|Divide|Select"),
]
_FAMILY_RE = [(name, re.compile(pattern)) for name, pattern in FAMILIES]


def family_of(kernel: str) -> str:
    name = kernel[len("custom_kernel_"):] if kernel.startswith("custom_kernel_") else kernel
    for family, pattern in _FAMILY_RE:
        if pattern.search(name):
            return family
    return "other"


def is_ours(kernel: str) -> bool:
    """A kernel compiled from our own Metal source (mx.fast.metal_kernel)."""

    return kernel.startswith("custom_kernel_")


def short(kernel: str, width: int = 96) -> str:
    name = kernel[len("custom_kernel_"):] if kernel.startswith("custom_kernel_") else kernel
    name = re.sub(r"(_+(bfloat16_t|float|uint32_t|int32_t|bool|floatc|int32_tc|bfloat16_tc|int32_ts)c?)+$", "", name)
    return name if len(name) <= width else name[: width - 1] + "~"


# ------------------------------------------------------------------- DTrace

_DTRACE = r"""
#pragma D option quiet
#pragma D option strsize=320
#pragma D option bufsize=64m
#pragma D option switchrate=100hz
BEGIN { printf("B %d\n", timestamp); }
/* libc++ std::string on Apple arm64: bit 7 of byte 23 is the long flag; a short
   string's characters start at byte 0, a long string's pointer is at byte 0. */
pid$target:libmlx.dylib:*Device*get_kernel*:entry /(*(uint8_t *)copyin(arg1 + 23, 1) & 0x80) == 0/ {
    self->k = copyinstr(arg1);
}
pid$target:libmlx.dylib:*Device*get_kernel*:entry /(*(uint8_t *)copyin(arg1 + 23, 1) & 0x80) != 0/ {
    self->k = copyinstr(*(uint64_t *)copyin(arg1, 8));
}
pid$target:libmlx.dylib:*CommandEncoder*dispatch_threadgroups*:entry {
    this->g = (uint64_t *)copyin(arg1, 24);
    this->t = (uint64_t *)copyin(arg2, 24);
    printf("D %d G %d %d %d %d %d %d %s\n", timestamp, this->g[0], this->g[1], this->g[2],
           this->t[0], this->t[1], this->t[2], self->k != NULL ? self->k : "<unnamed>");
}
pid$target:libmlx.dylib:*CommandEncoder*dispatch_threads*:entry {
    this->g = (uint64_t *)copyin(arg1, 24);
    this->t = (uint64_t *)copyin(arg2, 24);
    printf("D %d T %d %d %d %d %d %d %s\n", timestamp, this->g[0], this->g[1], this->g[2],
           this->t[0], this->t[1], this->t[2], self->k != NULL ? self->k : "<unnamed>");
}
/* macOS shows demangled names; "?" stands for the colons a probe description cannot hold. */
pid$target:libmlx.dylib:*CommandEncoder??commit?std*:entry { printf("C %d\n", timestamp); }
syscall::access:entry /pid == $target/ {
    self->path = copyinstr(arg0);
}
syscall::access:entry /pid == $target && strstr(self->path, "/.mtplx-census/") != NULL/ {
    printf("M %d %s\n", timestamp, self->path);
}
"""


class DTraceTape:
    """Every Metal dispatch MLX makes in this process between start and stop."""

    source = "dtrace"

    def __init__(self) -> None:
        self._dir = Path(tempfile.mkdtemp(prefix="mtplx-census-"))
        self._out = self._dir / "tape.txt"
        self._proc: subprocess.Popen | None = None
        self._result: Result | None = None

    @staticmethod
    def usable() -> str:
        """'' when DTrace can attach to this process, otherwise why not."""

        if sys.platform != "darwin":
            return "not macOS"
        try:
            ok = subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"sudo unavailable: {exc}"
        if ok.returncode != 0:
            return "sudo needs a password"
        return ""

    def start(self) -> "DTraceTape":
        why = self.usable()
        if why:
            raise RuntimeError(f"dtrace tape unavailable: {why}")
        import mlx.core as mx

        # MLX loads its Metal library on first use; the probes need it mapped.
        mx.eval(mx.zeros((1,)) + 1)
        script = self._dir / "tape.d"
        script.write_text(_DTRACE)
        self._proc = subprocess.Popen(
            ["sudo", "-n", "dtrace", "-p", str(os.getpid()), "-s", str(script), "-o", str(self._out)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=open(self._dir / "stderr.txt", "w"),
        )
        deadline = time.time() + 60
        while time.time() < deadline:
            if self._out.exists() and self._out.read_text(errors="replace").startswith("B "):
                return self
            if self._proc.poll() is not None:
                break
            time.sleep(0.05)
        detail = (self._dir / "stderr.txt").read_text(errors="replace")[-400:]
        self.stop()
        raise RuntimeError(f"dtrace did not attach: {detail}")

    def mark(self, label: str) -> None:
        os.access(MARK_PREFIX + label.replace("/", "|"), os.F_OK)

    def stop(self) -> Result:
        if self._result is not None:
            return self._result
        proc, self._proc = self._proc, None
        if proc is not None:
            self.mark("__end__")
            end = MARK_PREFIX + "__end__"
            deadline = time.time() + 20
            while time.time() < deadline and proc.poll() is None:
                text = self._out.read_text(errors="replace") if self._out.exists() else ""
                if end in text:
                    break
                time.sleep(0.05)
            subprocess.run(["sudo", "-n", "kill", "-INT", str(proc.pid)], capture_output=True)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                subprocess.run(["sudo", "-n", "kill", "-KILL", str(proc.pid)], capture_output=True)
        self._result = parse_dtrace(self._out)
        return self._result

    def result(self) -> Result:
        return self.stop()

    def __enter__(self) -> "DTraceTape":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def parse_dtrace(path: Path) -> Result:
    events: list[tuple[int, str, list[str]]] = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            parts = line.split(" ")
            if len(parts) >= 2 and parts[0] in ("D", "M", "C") and parts[1].isdigit():
                events.append((int(parts[1]), parts[0], parts[2:]))
    # DTrace drains one buffer per CPU; the timestamps restore the order.
    events.sort(key=lambda event: event[0])
    result = Result(source="dtrace")
    mark = ""
    commits = 0
    for _stamp, kind, rest in events:
        if kind == "M":
            label = " ".join(rest)
            label = label[label.index(MARK_PREFIX) + len(MARK_PREFIX):] if MARK_PREFIX in label else label
            if label == "__end__":
                break
            mark = label.replace("|", "/")
        elif kind == "C":
            commits += 1
        elif len(rest) >= 8:
            sizes = [int(v) for v in rest[1:7]]
            result.launches.append(
                Launch(
                    kernel=" ".join(rest[7:]),
                    mark=mark,
                    kind="groups" if rest[0] == "G" else "threads",
                    grid=tuple(sizes[:3]),
                    group=tuple(sizes[3:]),
                )
            )
    result.command_buffers = commits
    return result


# -------------------------------------------------------- the executor's recorder


class EngineTape:
    """The same tape through the native executor's recorder."""

    source = "engine"

    def __init__(self) -> None:
        from mtplx.engine import runtime

        self._engine = runtime.engine()
        self._result: Result | None = None
        self._open = False

    @staticmethod
    def usable() -> str:
        try:
            from mtplx.engine import runtime
        except Exception as exc:  # noqa: BLE001 - reported as the reason
            return f"{type(exc).__name__}: {exc}"
        return "" if runtime.available() else (runtime.unavailable() or "unavailable")

    def start(self) -> "EngineTape":
        self._engine.record_begin(label="census")
        self._open = True
        return self

    def mark(self, label: str) -> None:
        self._engine.record_mark(label)

    def stop(self) -> Result:
        if self._result is None:
            trace = self._engine.record_end() if self._open else None
            self._open = False
            self._result = result_from_steps(_trace_steps(trace), source="engine")
        return self._result

    def result(self) -> Result:
        return self.stop()

    def __enter__(self) -> "EngineTape":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def _trace_steps(trace: Any) -> list[Any]:
    """The launches of a recorder trace, whatever container it uses."""

    if trace is None:
        return []
    exported = getattr(trace, "exported", None)
    if callable(exported):
        # The executor's Trace: every step is a tuple, kernel name second, its mark twelfth.
        return [{"kernel_name": step[1], "mark": step[11]} for step in exported().get("steps", [])]
    for name in ("steps", "launches", "events"):
        steps = getattr(trace, name, None)
        if callable(steps):
            steps = steps()
        if steps is not None:
            return list(steps)
    if isinstance(trace, dict):
        return list(trace.get("steps") or [])
    return list(trace)


def _field(step: Any, name: str, default: Any = None) -> Any:
    if isinstance(step, dict):
        return step.get(name, default)
    return getattr(step, name, default)


def result_from_steps(steps: Iterable[Any], *, source: str, name: str = "") -> Result:
    """A census of ProgramIR steps (or recorder steps): marks carry forward."""

    result = Result(source=source)
    mark = name
    for step in steps:
        label = _field(step, "mark", "") or ""
        if label:
            mark = f"{name}/{label}" if name else label
        kernel = _field(step, "kernel_name", None)
        if not kernel:
            continue
        dispatch = _field(step, "dispatch", None) or ("", (0, 0, 0), (0, 0, 0))
        result.launches.append(
            Launch(
                kernel=str(kernel),
                mark=mark,
                kind=str(dispatch[0]),
                grid=tuple(int(v) for v in dispatch[1]),
                group=tuple(int(v) for v in dispatch[2]),
            )
        )
    return result


def tape(backend: str = "auto"):
    """A tape on the best source this process has: the recorder, else DTrace."""

    refuse_if_held()
    if backend in ("auto", "engine") and not EngineTape.usable():
        return EngineTape()
    if backend == "engine":
        raise RuntimeError(f"engine recorder unavailable: {EngineTape.usable()}")
    return DTraceTape()


# -------------------------------------------------------------------- dumps


def from_dump(folder: Path) -> Result:
    """Every ProgramIR JSON under ``folder`` as one census, one mark per program."""

    from mtplx.engine.ir import ProgramIR

    result = Result(source=f"dump {folder}")
    files = sorted(Path(folder).rglob("*.json"))
    for path in files:
        try:
            obj = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(obj, dict) or "steps" not in obj or "buffers" not in obj:
            continue
        program = ProgramIR.from_obj(obj)
        name = program.name or path.stem
        part = result_from_steps(program.steps, source="dump", name=name)
        result.launches.extend(part.launches)
        result.notes.append(
            f"{name}: {program.launches()} launches, signature {program.signature()[:12]}, "
            f"temp bytes {program.bytes_by_role().get('temp', 0):,} ({path.name})"
        )
    if not result.launches:
        result.notes.append("no ProgramIR JSON found")
    return result


# -------------------------------------------------------------------- blocks
#
# A profiled program dumped with MTPLX_NATIVE_ENGINE_PROFILE=forms comes with
# ``<key>.pins.json``: the parameter names held in each pinned buffer. A launch
# that binds a weight belongs to that weight's layer and block, exactly; a
# launch that binds none (a fused elementwise chain, a copy) belongs to the
# block of the launch before it. Launches after the output head that bind no
# weight are the state writes the program returns last ("tail").

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
BLOCK_ORDER = ("prelude", "embedding", "recurrent", "attention", "experts", "head", "draft", "tail", "other")


def _param_block(name: str, recurrent_layers: set[int]) -> tuple[int | None, str]:
    """(layer or None, block) of one parameter path."""

    parts = name.split(".")
    if "mtp" in parts:
        return None, "draft"
    match = _LAYER_RE.search(name)
    if match is None:
        if "embed_tokens" in parts:
            return None, "embedding"
        if "lm_head" in parts or parts[-2:] == ["norm", "weight"]:
            return None, "head"
        return None, "other"
    layer = int(match.group(1))
    rest = name[match.end():].split(".")
    if rest[0] in ("mlp", "post_attention_layernorm"):
        return layer, "experts"
    if rest[0] == "linear_attn":
        return layer, "recurrent"
    if rest[0] == "self_attn":
        return layer, "attention"
    if rest[0] == "input_layernorm":
        return layer, "recurrent" if layer in recurrent_layers else "attention"
    return layer, "other"


def _recurrent_layers(names: Iterable[str]) -> set[int]:
    found: set[int] = set()
    for name in names:
        match = _LAYER_RE.search(name)
        if match is not None and ".linear_attn." in name and "mtp" not in name.split("."):
            found.add(int(match.group(1)))
    return found


def attribute(program: Any, pins: dict[str, list[str]]) -> list[tuple[int | None, str, bool]]:
    """Per step of ``program`` (a ProgramIR): (layer, block, binds a named weight)."""

    all_names = [name for names in pins.values() for name in names]
    recurrent = _recurrent_layers(all_names)
    slot_block: dict[int, tuple[int | None, str]] = {}
    for slot, names in pins.items():
        places = {_param_block(name, recurrent) for name in names}
        # A buffer shared by several parameters: the last one in model order.
        slot_block[int(slot)] = max(places, key=lambda p: (-1 if p[0] is None else p[0], BLOCK_ORDER.index(p[1])))
    pinned = {buf.id: buf.index for buf in program.buffers if buf.role == "pinned"}
    out: list[tuple[int | None, str, bool]] = []
    current: tuple[int | None, str] = (None, "prelude")
    seen_head = False
    for step in program.steps:
        places = [slot_block[pinned[b[1]]] for b in step.bindings if b[1] in pinned and pinned[b[1]] in slot_block]
        if places:
            current = max(places, key=lambda p: (-1 if p[0] is None else p[0], BLOCK_ORDER.index(p[1])))
            seen_head = seen_head or current[1] == "head"
            out.append((current[0], current[1], True))
        elif seen_head:
            out.append((None, "tail", False))
        else:
            out.append((current[0], current[1], False))
    return out


def _width(program: Any) -> int | None:
    """Rows of the call: the token input ``[1, W]`` (int32) the verify step takes first."""

    for arr in program.inputs[:2]:
        if arr.dtype in ("int32", "uint32") and len(arr.shape) == 2 and arr.shape[0] == 1:
            return int(arr.shape[1])
    return None


def blocks_report(folder: Path) -> dict[str, Any]:
    """Every program under ``folder`` with its profile, pins and forms, by block and layer."""

    from mtplx.engine.ir import ProgramIR

    programs: list[dict[str, Any]] = []
    for ir_path in sorted(Path(folder).rglob("*.ir.json")):
        if ir_path.name.endswith(".second.ir.json"):
            continue
        stem = ir_path.name[: -len(".ir.json")]

        def sibling(suffix: str) -> dict[str, Any] | None:
            path = ir_path.with_name(f"{stem}.{suffix}")
            try:
                return json.loads(path.read_text())
            except (OSError, ValueError):
                return None

        try:
            program = ProgramIR.from_obj(json.loads(ir_path.read_text()))
        except (OSError, ValueError, KeyError):
            continue
        profile = sibling("profile.json") or {}
        pins = (sibling("pins.json") or {}).get("slots") or {}
        forms = sibling("forms.json") or {}
        summary = sibling("summary.json") or {}
        gpu = [0] * len(program.steps)
        for row in profile.get("steps") or []:
            index = int(row.get("step", -1))
            if 0 <= index < len(gpu):
                gpu[index] = int(row.get("gpu_ns", 0) or 0)
        places = attribute(program, pins) if pins else [(None, "other", False)] * len(program.steps)
        by_block: dict[str, dict[str, int]] = {}
        per_layer: dict[int, dict[str, Any]] = {}
        by_kernel: dict[tuple[str, str], dict[str, int]] = {}
        for step, (layer, block, weighted), ns in zip(program.steps, places, gpu):
            slot = by_block.setdefault(block, {"launches": 0, "gpu_ns": 0, "launches_binding_a_weight": 0})
            slot["launches"] += 1
            slot["gpu_ns"] += ns
            slot["launches_binding_a_weight"] += int(weighted)
            if layer is not None:
                row = per_layer.setdefault(layer, {"layer": layer, "kind": "", "mixer_launches": 0, "mixer_gpu_ns": 0,
                                                   "expert_launches": 0, "expert_gpu_ns": 0})
                if block == "experts":
                    row["expert_launches"] += 1
                    row["expert_gpu_ns"] += ns
                elif block in ("recurrent", "attention"):
                    row["kind"] = block
                    row["mixer_launches"] += 1
                    row["mixer_gpu_ns"] += ns
            kslot = by_kernel.setdefault((step.kernel_name, block), {"launches": 0, "gpu_ns": 0})
            kslot["launches"] += 1
            kslot["gpu_ns"] += ns
        layers = [per_layer[k] for k in sorted(per_layer)]

        def mean(kind: str, column: str) -> float | None:
            values = [row[column] for row in layers if row["kind"] == kind] if kind != "experts" else [
                row[column] for row in layers if row["expert_launches"]]
            return round(sum(values) / len(values), 2) if values else None

        form_rows = list(forms.get("forms") or [])
        default = next((r for r in form_rows if "gpu_ns" in r), None)
        floor = next((r for r in form_rows if r.get("barriers") == "none" and "gpu_ns" in r), None)
        effects = form_effects(form_rows)
        programs.append({
            "program": stem,
            "site": summary.get("site") or profile.get("site"),
            "width": _width(program),
            "launches": len(program.steps),
            "barriers_recorded_in_ir": sum(1 for s in program.steps if s.barrier_before),
            "profile_gpu_ns": sum(gpu),
            "profiled": bool(profile),
            "pins_named": len(pins),
            "by_block": {k: by_block[k] for k in BLOCK_ORDER if k in by_block},
            "per_layer_mean": {
                "recurrent_mixer_launches": mean("recurrent", "mixer_launches"),
                "recurrent_mixer_gpu_ns": mean("recurrent", "mixer_gpu_ns"),
                "attention_mixer_launches": mean("attention", "mixer_launches"),
                "attention_mixer_gpu_ns": mean("attention", "mixer_gpu_ns"),
                "expert_block_launches": mean("experts", "expert_launches"),
                "expert_block_gpu_ns": mean("experts", "expert_gpu_ns"),
            },
            "per_layer": layers,
            "by_kernel": [
                {"kernel": kernel, "family": family_of(kernel), "block": block, **value}
                for (kernel, block), value in sorted(by_kernel.items(), key=lambda item: -item[1]["gpu_ns"])
            ],
            "forms": form_rows,
            "forms_order": forms.get("order", "form by form"),
            "forms_fill": forms.get("fill"),
            "after_check": forms.get("after_check"),
            "floor_suspect": forms.get("floor_suspect"),
            "effects": effects,
            "null_encoder": forms.get("null_encoder"),
            "barriers_by_form": forms.get("barriers_by_form"),
            "barrier_free_floor": None if not (default and floor) else {
                "serving_form": [default["dispatch"], default["barriers"], default["split"]],
                "serving_gpu_ns_median": default["gpu_ns"]["median"],
                "no_barrier_gpu_ns_median": floor["gpu_ns"]["median"],
                "most_fusion_could_save_ns": default["gpu_ns"]["median"] - floor["gpu_ns"]["median"],
                "serving_wall_ns_median": default["wall_ns"]["median"],
                "no_barrier_wall_ns_median": floor["wall_ns"]["median"],
            },
        })
    return {"folder": str(folder), "programs": programs}


def form_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Each setting's own share of a replay's GPU time, the other two held:
    ``barriers`` = concurrent/minimal/256 - concurrent/none/256 (the most that
    fusing dependent launches could save), ``barriers_one_buffer`` the same in
    one command buffer, ``dispatch`` = hybrid/minimal/256 - concurrent/minimal/
    256, ``split`` = hybrid/minimal/256 - hybrid/minimal/0, ``fill`` = the
    serving form without the zero fill minus with it. Medians, in ns; a
    positive figure is time the first form spends over the second. Rows of a
    census with the fill (``fill`` 0) are compared with each other; a file
    from before the fill compares its plain rows."""

    timed = [r for r in rows if "gpu_ns" in r]
    filled = any(r.get("fill") == 0 for r in timed)

    def median(dispatch: str, barriers: str, split: int, fill: Any = 0) -> int | None:
        want = fill if filled else None
        for row in timed:
            if (row.get("dispatch"), row.get("barriers"), row.get("split")) == (dispatch, barriers, split) \
                    and row.get("fill") == want:
                return int(row["gpu_ns"]["median"])
        return None

    def delta(first: int | None, second: int | None) -> int | None:
        return None if first is None or second is None else first - second

    serving = median("hybrid", "minimal", 256)
    concurrent = median("concurrent", "minimal", 256)
    out = {
        "barriers": delta(concurrent, median("concurrent", "none", 256)),
        "barriers_one_buffer": delta(median("concurrent", "minimal", 0), median("concurrent", "none", 0)),
        "dispatch": delta(serving, concurrent),
        "split": delta(serving, median("hybrid", "minimal", 0)),
        "fill": None,
        "serving_gpu_ns": serving,
        "rows_filled": filled,
    }
    if filled:
        plain = next((int(r["gpu_ns"]["median"]) for r in timed if r.get("fill") is None
                      and (r.get("dispatch"), r.get("barriers"), r.get("split")) == ("hybrid", "minimal", 256)), None)
        out["fill"] = delta(serving, plain)
        out["serving_gpu_ns_without_fill"] = plain
    return out


def render_blocks(report: dict[str, Any]) -> str:
    lines = ["# Census by block", "", f"Source: {report['folder']}. GPU times are the per-launch profile "
             "(one command buffer a launch); the forms table times whole replays.", ""]
    for prog in report["programs"]:
        lines += [f"## {prog['program']}: width {prog['width']}, {prog['launches']} launches, "
                  f"profile {prog['profile_gpu_ns'] / 1e6:.2f} ms", "",
                  "| block | launches | binding a weight | GPU ms (profile) |", "|---|---:|---:|---:|"]
        for block, value in prog["by_block"].items():
            lines.append(f"| {block} | {value['launches']} | {value['launches_binding_a_weight']} | "
                         f"{value['gpu_ns'] / 1e6:.3f} |")
        means = prog["per_layer_mean"]
        lines += ["", "Per layer (mean): " + ", ".join(f"{k} {v}" for k, v in means.items() if v is not None), ""]
        if prog["forms"]:
            lines += [f"Forms timed {prog.get('forms_order')}, fill {prog.get('forms_fill')}; serving replay "
                      f"after the barrier-free forms: {prog.get('after_check')}"
                      + (" **FLOOR SUSPECT**" if prog.get("floor_suspect") else ""), "",
                      "| dispatch | barriers | split | fill | GPU ms median | wall ms median | command buffers |",
                      "|---|---|---:|---:|---:|---:|---:|"]
            for row in prog["forms"]:
                if "gpu_ns" not in row:
                    lines.append(f"| {row['dispatch']} | {row['barriers']} | {row['split']} | {row.get('fill')} | "
                                 f"declined: {row.get('declined')} | | |")
                    continue
                lines.append(f"| {row['dispatch']} | {row['barriers']} | {row['split']} | {row.get('fill')} | "
                             f"{row['gpu_ns']['median'] / 1e6:.3f} | {row['wall_ns']['median'] / 1e6:.3f} | "
                             f"{row.get('command_buffers')} |")
            effects = prog.get("effects") or {}
            ms = lambda ns: "n/a" if ns is None else f"{ns / 1e6:.3f} ms"  # noqa: E731
            lines += ["", "Each setting's own share of a replay's GPU time (the other two held): "
                          f"barriers {ms(effects.get('barriers'))} (concurrent, minimal against none, split 256; "
                          f"in one command buffer {ms(effects.get('barriers_one_buffer'))}), "
                          f"dispatch form {ms(effects.get('dispatch'))} (hybrid against concurrent), "
                          f"commit split {ms(effects.get('split'))} (every 256 launches against one buffer), "
                          f"zero fill {ms(effects.get('fill'))}."]
            if prog["barrier_free_floor"]:
                floor = prog["barrier_free_floor"]
                lines += ["", f"Most that fusing dependent launches could save: "
                              f"{floor['most_fusion_could_save_ns'] / 1e6:.3f} ms of GPU time a replay "
                              "(serving form against concurrent with no barrier in one buffer: all three "
                              "settings at once)."]
        lines += ["", "| launches | GPU ms | block | kernel |", "|---:|---:|---|---|"]
        for row in prog["by_kernel"][:30]:
            lines.append(f"| {row['launches']} | {row['gpu_ns'] / 1e6:.3f} | {row['block']} | `{short(row['kernel'])}` |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- rendering


def render(result: Result, *, title: str = "Launch census", top: int = 40) -> str:
    marks = result.marks()
    by_mark = result.by_mark()
    lines = [f"# {title}", "", f"Source: {result.source}. A launch is one Metal dispatch.", ""]
    for note in result.notes:
        lines.append(f"- {note}")
    if result.notes:
        lines.append("")

    families = [name for name, _ in FAMILIES] + ["other"]
    table: dict[str, Counter] = {mark: Counter() for mark in marks}
    ours: dict[str, int] = {mark: 0 for mark in marks}
    for mark in marks:
        for launch in by_mark[mark]:
            table[mark][family_of(launch.kernel)] += 1
            ours[mark] += int(is_ours(launch.kernel))
    head = "| family | " + " | ".join(mark or "(unmarked)" for mark in marks) + " | all |"
    lines += ["## Launches by family", "", head, "|---|" + "---:|" * (len(marks) + 1)]
    for family in families:
        row = [table[mark][family] for mark in marks]
        if not any(row):
            continue
        lines.append(f"| {family} | " + " | ".join(str(v) for v in row) + f" | {sum(row)} |")
    totals = [len(by_mark[mark]) for mark in marks]
    lines.append("| **total** | " + " | ".join(f"**{v}**" for v in totals) + f" | **{sum(totals)}** |")
    lines.append(
        "| of which kernels of ours | " + " | ".join(str(ours[mark]) for mark in marks)
        + f" | {sum(ours.values())} |"
    )
    if result.command_buffers is not None:
        lines += ["", f"Command buffers committed while the tape ran: {result.command_buffers}."]

    for mark in marks:
        counts = Counter(launch.kernel for launch in by_mark[mark])
        lines += ["", f"## {mark or '(unmarked)'}: {len(by_mark[mark])} launches, {len(counts)} kernels", "",
                  "| launches | family | kernel |", "|---:|---|---|"]
        for kernel, count in counts.most_common(top):
            lines.append(f"| {count} | {family_of(kernel)} | `{short(kernel)}` |")
        if len(counts) > top:
            rest = sum(count for _k, count in counts.most_common()[top:])
            lines.append(f"| {rest} | | ({len(counts) - top} more kernels) |")
    return "\n".join(lines) + "\n"


def render_tape(result: Result, mark: str | None = None) -> str:
    """The launches in the order they were encoded: the map from a count to the code."""

    lines = []
    last = None
    for index, launch in enumerate(result.launches):
        if mark is not None and launch.mark != mark:
            continue
        if launch.mark != last:
            lines.append(f"--- {launch.mark or '(unmarked)'}")
            last = launch.mark
        grid = "x".join(str(v) for v in launch.grid)
        group = "x".join(str(v) for v in launch.group)
        lines.append(f"{index:5d}  {launch.kind[:1] or '-'} {grid:>14} / {group:<10}  {short(launch.kernel, 150)}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------- command line


def _cmd_dump(args: argparse.Namespace) -> int:
    result = from_dump(Path(args.dir))
    text = render(result, title=f"Launch census of {args.dir}")
    if args.out:
        Path(args.out).write_text(text)
        if args.tape:
            Path(args.out).with_suffix(".tape.txt").write_text(render_tape(result))
        Path(args.out).with_suffix(".json").write_text(json.dumps(result.to_obj()))
    print(text)
    return 0


def _cmd_blocks(args: argparse.Namespace) -> int:
    report = blocks_report(Path(args.dir))
    text = render_blocks(report)
    if args.out:
        Path(args.out).with_suffix(".md").write_text(text)
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(text)
    return 0


def _cmd_synth(args: argparse.Namespace) -> int:
    refuse_if_held()
    from mtplx.backends import candidate_shapes
    from mtplx.engine.tools import synth_flash_next as synth

    shape = candidate_shapes.BY_NAME[args.shape] if args.shape else None
    result, report = synth.census_round(
        rounds=args.rounds, backend=args.backend, layers=args.layers, vocab=args.vocab,
        prompt_tokens=args.prompt_tokens, seed=args.seed, recipe=args.recipe, shape=shape,
    )
    result.notes = report + result.notes
    text = render(result, title=f"Launch census, synthetic {args.shape or 'Flash-Next'} at the real geometry")
    if args.out:
        Path(args.out).write_text(text)
        Path(args.out).with_suffix(".tape.txt").write_text(render_tape(result))
        Path(args.out).with_suffix(".json").write_text(json.dumps(result.to_obj()))
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mtplx.engine.tools.census", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    dump = sub.add_parser("dump", help="census of a directory of ProgramIR JSON")
    dump.add_argument("dir")
    dump.add_argument("--out")
    dump.add_argument("--tape", action="store_true", help="also write the ordered launch tape")
    dump.set_defaults(fn=_cmd_dump)
    blocks = sub.add_parser("blocks", help="a profiled dump (MTPLX_NATIVE_ENGINE_PROFILE=forms) by block, layer and form")
    blocks.add_argument("dir")
    blocks.add_argument("--out", help="writes OUT.md and OUT.json")
    blocks.set_defaults(fn=_cmd_blocks)
    synth = sub.add_parser("synth", help="census of a round on the synthetic real-geometry model")
    synth.add_argument("--rounds", type=int, default=6)
    synth.add_argument("--layers", type=int, default=48)
    synth.add_argument("--vocab", type=int, default=248_320)
    synth.add_argument("--prompt-tokens", type=int, default=48)
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument("--backend", choices=("auto", "engine", "dtrace"), default="auto")
    synth.add_argument("--recipe", choices=("optimized-speed", "bare-speed"), default="optimized-speed",
                       help="whose packing the synthetic model carries")
    synth.add_argument("--shape", help="a candidate shape at its release sizes (mtplx.backends.candidate_shapes) instead of Flash-Next")
    synth.add_argument("--out")
    synth.set_defaults(fn=_cmd_synth)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
