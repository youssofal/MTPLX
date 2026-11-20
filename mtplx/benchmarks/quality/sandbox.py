"""Running code the model wrote, on the machine that serves it.

Grading a coding task means executing model-written Python, unattended, on a
Mac that also holds the user's files and apps. Each run therefore gets:

- its own scratch folder as the working directory, ``HOME`` and ``TMPDIR``,
  and an environment holding nothing else;
- a wall-clock limit, after which its whole process group is killed;
- limits on CPU time, memory, core dumps and the size of any file it writes
  (its output goes to files in the scratch folder, so a program that prints
  forever fills neither memory nor the disk);
- on macOS, a sandbox profile: no network; nothing read from the user's
  folders or external drives but the scratch folder and the Python
  installation; nothing written outside the scratch folder; no new processes,
  no programs other than Python; no signals to any other process (a model
  once ran ``pkill`` in an unattended session).
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
OUTPUT_LIMIT = 16_384  # characters kept from the end of each of stdout and stderr
FILE_LIMIT = 64 * 1024**2  # bytes any one file the program writes may reach
LAUNCHER = ".launch.py"
# Sets the limits inside the child, then runs ``<file>`` or ``-m <module>``
# exactly as ``python`` would, with the project folder first on sys.path.
_LAUNCHER_SOURCE = """\
import resource, runpy, sys
for name, value in (("RLIMIT_CPU", {cpu}), ("RLIMIT_AS", {memory}), ("RLIMIT_CORE", 0),
                    ("RLIMIT_FSIZE", {file_bytes})):
    limit = getattr(resource, name, None)
    if limit is not None:
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass  # macOS refuses some of these; the sandbox and the timeout still hold
if sys.argv[1] == "-m":
    sys.argv = sys.argv[2:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")
"""


@dataclass(frozen=True)
class RunResult:
    status: str  # "ok" | "failed" | "timeout"
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    sandboxed: bool


def _profile(workdir: Path) -> str:
    # Later rules win in a sandbox profile: each deny is followed by the
    # narrow allowances the child needs to run Python in its scratch folder.
    python_roots = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(), Path(sys.executable).resolve().parent}
    python = " ".join(f'(subpath "{root}")' for root in sorted(python_roots))
    # File metadata stays readable everywhere: Python resolves its own path at
    # start by stat-ing each folder above it, and a stat shows no content (a
    # folder's listing is a data read, so still refused).
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "/Users") (subpath "/Volumes"))
(allow file-read-metadata)
(allow file-read* (subpath "{workdir}") {python})
(deny file-write*)
(allow file-write* (subpath "{workdir}") (literal "/dev/null") (literal "/dev/tty"))
(deny process-fork)
(deny process-exec)
(allow process-exec {python})
(deny signal)
(allow signal (target self))
"""


def run_python(
    workdir: Path,
    args: list[str],
    *,
    timeout_s: float = 20.0,
    memory_bytes: int = 2 * 1024**3,
) -> RunResult:
    """Run ``python <args>`` inside ``workdir`` under the limits above.

    ``args`` is a script name or ``["-m", module, ...]``, as on a command line.
    """

    workdir = workdir.resolve()
    launcher = workdir / LAUNCHER
    # The CPU limit is a backstop behind the wall clock, never ahead of it, so
    # a program that spins is always reported as a timeout.
    cpu_seconds = 2 * int(timeout_s) + 5
    launcher.write_text(
        _LAUNCHER_SOURCE.format(cpu=cpu_seconds, memory=int(memory_bytes), file_bytes=FILE_LIMIT)
    )
    argv = [sys.executable, "-B", "-E", "-s", LAUNCHER, *args]
    sandboxed = SANDBOX_EXEC.exists()
    if sandboxed:
        argv = [str(SANDBOX_EXEC), "-p", _profile(workdir), *argv]
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "TMPDIR": str(workdir), "PYTHONDONTWRITEBYTECODE": "1"}
    outputs = workdir / ".stdout", workdir / ".stderr"
    started = time.monotonic()
    with outputs[0].open("wb") as out, outputs[1].open("wb") as err:
        proc = subprocess.Popen(argv, cwd=workdir, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
            status = "ok" if proc.returncode == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            status = "timeout"
    stdout, stderr = (_tail(path) for path in outputs)
    for path in (launcher, *outputs):
        path.unlink(missing_ok=True)
    return RunResult(
        status=status,
        returncode=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        seconds=time.monotonic() - started,
        sandboxed=sandboxed,
    )


def _tail(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 4 * OUTPUT_LIMIT))
        return handle.read().decode(errors="replace")[-OUTPUT_LIMIT:]


class Workspace:
    """A scratch project folder whose files the model reads and writes by name."""

    def __init__(self, root: Path, files: dict[str, str]):
        self.root = root.resolve()
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        for name, content in files.items():
            self.write(name, content)

    def path(self, name: str) -> Path:
        candidate = (self.root / name).resolve()
        if candidate == self.root or self.root not in candidate.parents:
            raise ValueError(f"{name!r} is outside the project")
        return candidate

    def read(self, name: str) -> str:
        return self.path(name).read_text()

    def write(self, name: str, content: str) -> None:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def names(self) -> list[str]:
        return sorted(
            str(p.relative_to(self.root))
            for p in self.root.rglob("*")
            if p.is_file() and not p.name.startswith(".")
        )
