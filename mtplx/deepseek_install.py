"""Install the reviewed harness and its Node runtime privately on first open."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

NODE_VERSION = "22.23.4"
HARNESS_VERSION = "0.1.2-rc.1"
HARNESS_BIN = Path("node_modules/.bin/dsh")
# SHASUMS256.txt from the official Node distribution for this release.
NODE_SHA256 = {
    "arm64": "c9e1dbc748b33ed2d14f42445191ecaf14c56259c7a9a26d0042c49ec2b66428",
    "x86_64": "efcc2b0db6f94bf53eddb038fe2b370277a042bd29e4f19b8f8e23598795b819",
}


def runtime_directory(root: Path) -> Path:
    return root / f"runtime-{HARNESS_VERSION}-node{NODE_VERSION}"


def executable(root: Path) -> Path:
    return runtime_directory(root) / HARNESS_BIN


def installed(root: Path) -> bool:
    runtime = runtime_directory(root)
    return os.access(runtime / "node/bin/node", os.X_OK) and os.access(runtime / HARNESS_BIN, os.X_OK)


def _install_node(stage: Path) -> Path:
    machine = platform.machine()
    if platform.system() != "Darwin" or machine not in NODE_SHA256:
        raise OSError(f"DeepSeek Harness first-use setup requires an Intel or Apple Silicon Mac (got {machine}).")
    arch = "arm64" if machine == "arm64" else "x64"
    name = f"node-v{NODE_VERSION}-darwin-{arch}"
    url = f"https://nodejs.org/dist/v{NODE_VERSION}/{name}.tar.gz"
    with tempfile.TemporaryFile() as archive:
        with urllib.request.urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, archive)
        archive.seek(0)
        if hashlib.file_digest(archive, "sha256").hexdigest() != NODE_SHA256[machine]:
            raise OSError(f"Node {NODE_VERSION} failed its download checksum. Retry setup.")
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:gz") as bundle:
            bundle.extractall(stage, filter="data")
    (stage / name).rename(stage / "node")
    return stage / "node" / "bin"


def install(root: Path, telemetry_env: dict[str, str]) -> Path:
    """Serialize concurrent starts and publish only a complete, runnable installation."""
    root.mkdir(parents=True, exist_ok=True)
    target = executable(root)
    with (root / "install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if installed(root):
            return target
        lockfile = Path(__file__).with_name("data") / "deepseek-harness-lock.json"
        with tempfile.TemporaryDirectory(dir=root, prefix=".setup-") as temp:
            stage = Path(temp) / "runtime"
            stage.mkdir()
            node_bin = _install_node(stage)
            shutil.copyfile(lockfile, stage / "package-lock.json")
            manifest = {"private": True, "dependencies": {"@deepseek-ai/dsh": HARNESS_VERSION}}
            (stage / "package.json").write_text(json.dumps(manifest))
            env = {**os.environ, **telemetry_env}
            env["PATH"] = os.pathsep.join((str(node_bin), env.get("PATH", os.defpath)))
            env.pop("DEEPSEEK_API_KEY", None)
            log_path = root / "install.log"
            with log_path.open("w") as log:
                result = subprocess.run(
                    [str(node_bin / "npm"), "ci", "--no-audit", "--no-fund"],
                    cwd=stage, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, timeout=300, check=False,
                )
            if result.returncode or not os.access(stage / HARNESS_BIN, os.X_OK):
                raise OSError(f"DeepSeek Harness setup failed. See {log_path} and retry.")
            runtime = runtime_directory(root)
            if runtime.exists():
                runtime.rename(Path(temp) / "previous")
            stage.rename(runtime)
    return target
