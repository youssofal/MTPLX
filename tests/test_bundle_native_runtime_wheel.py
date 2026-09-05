"""The runtime wheel's native kernels are signed before they are packaged.

Notarization inspects every Mach-O inside the app bundle, wheels included,
and the app's own signing pass cannot reach into a zip: 2.11.2's first
submission was rejected on the two linker-signed QSA kernels. The bundler
signs them the way the app signs its binaries and the release script
verifies the result before anything is uploaded.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("wheel.wheelfile")
from wheel.wheelfile import WheelFile  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "bundle_native_runtime_wheel",
    Path(__file__).parents[1] / "scripts/bundle_native_runtime_wheel.py",
)
bundler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundler)

EXT = "mtplx_qsa_kernels/_ext.cpython-314-darwin.so"
DYLIB = "mtplx_qsa_kernels/libmtplx_qsa_kernel_ops.dylib"
METALLIB = "mtplx_qsa_kernels/kernels.metallib"


def _write_wheel(path: Path, members: dict[str, bytes]) -> Path:
    with WheelFile(path, "w") as wheel:
        for name, data in members.items():
            wheel.writestr(name, data)
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    pure = _write_wheel(
        tmp_path / "mtplx-9.9.9-py3-none-any.whl",
        {
            "mtplx/__init__.py": b"",
            "mtplx-9.9.9.dist-info/METADATA": b"Metadata-Version: 2.1\nName: mtplx\nVersion: 9.9.9\n",
            "mtplx-9.9.9.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
            ),
            "mtplx-9.9.9.dist-info/top_level.txt": b"mtplx\n",
        },
    )
    native = _write_wheel(
        tmp_path / "mtplx_qsa_kernels-9.9.9-cp314-cp314-macosx_15_0_arm64.whl",
        {
            "mtplx_qsa_kernels/__init__.py": b"",
            EXT: b"MACHO-EXT",
            DYLIB: b"MACHO-DYLIB",
            METALLIB: b"METALLIB",
            "mtplx_qsa_kernels/NOTICE": b"notice",
            "mtplx_qsa_kernels/LICENSE.txt": b"license",
            "mtplx_qsa_kernels/MLX_LICENSE.txt": b"mlx license",
            "mtplx_qsa_kernels-9.9.9.dist-info/METADATA": (
                b"Metadata-Version: 2.1\nName: mtplx-qsa-kernels\nVersion: 9.9.9\n"
                b"Requires-Dist: mlx==0.32.2\n"
            ),
            "mtplx_qsa_kernels-9.9.9.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
                b"Tag: cp314-cp314-macosx_15_0_arm64\n"
            ),
        },
    )
    return pure, native


def _run_bundler(monkeypatch, pure: Path, native: Path, out: Path, *extra: str) -> Path:
    monkeypatch.setattr(sys, "argv", ["bundle", str(pure), str(native), "--out", str(out), *extra])
    bundler.main()
    return out / "mtplx-9.9.9-cp314-cp314-macosx_15_0_arm64.whl"


def _fake_codesign(calls: list[list[str]], *, timestamp: bool = True):
    def run(cmd, **kwargs):
        assert cmd[0] == "/usr/bin/codesign", cmd
        calls.append(list(cmd))
        target = Path(cmd[-1])
        if "--sign" in cmd:
            target.write_bytes(b"SIGNED:" + target.read_bytes())
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "--verify" in cmd:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "-dvvv" in cmd:
            details = "Authority=Developer ID Application: Test (TEAMID)\nAuthority=Apple Root CA\n"
            if timestamp:
                details += "Timestamp=Sep 5, 2026 at 4:37:34 pm\n"
            return subprocess.CompletedProcess(cmd, 0, "", details)
        raise AssertionError(cmd)

    return run


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def test_native_mach_o_members_are_signed_and_the_record_follows(tmp_path, monkeypatch) -> None:
    pure, native = _inputs(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(bundler.subprocess, "run", _fake_codesign(calls))
    bundled = _run_bundler(
        monkeypatch, pure, native, tmp_path / "out", "--codesign-identity", "Developer ID Application: Test"
    )
    with zipfile.ZipFile(bundled) as archive:
        assert archive.read(EXT) == b"SIGNED:MACHO-EXT"
        assert archive.read(DYLIB) == b"SIGNED:MACHO-DYLIB"
        assert archive.read(METALLIB) == b"METALLIB"
        record = archive.read("mtplx-9.9.9.dist-info/RECORD").decode().splitlines()
    hashes = {line.split(",")[0]: line.split(",")[1] for line in record if line}
    assert hashes[EXT] == _record_hash(b"SIGNED:MACHO-EXT")
    assert hashes[DYLIB] == _record_hash(b"SIGNED:MACHO-DYLIB")
    with WheelFile(bundled) as reopened:  # the rewritten RECORD verifies
        assert reopened.read(EXT) == b"SIGNED:MACHO-EXT"
    sign_calls = [call for call in calls if "--sign" in call]
    assert [Path(call[-1]).name for call in sign_calls] == [
        "_ext.cpython-314-darwin.so",
        "libmtplx_qsa_kernel_ops.dylib",
    ]
    for call in sign_calls:
        assert call[1:6] == ["--force", "--options", "runtime", "--timestamp", "--sign"]
        assert call[6] == "Developer ID Application: Test"


def test_without_an_identity_the_members_are_packaged_unchanged(tmp_path, monkeypatch) -> None:
    pure, native = _inputs(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(bundler.subprocess, "run", _fake_codesign(calls))
    bundled = _run_bundler(monkeypatch, pure, native, tmp_path / "out")
    with zipfile.ZipFile(bundled) as archive:
        assert archive.read(EXT) == b"MACHO-EXT"
        assert archive.read(DYLIB) == b"MACHO-DYLIB"
    assert calls == []


def test_a_signature_without_a_secure_timestamp_fails_the_build(tmp_path, monkeypatch) -> None:
    pure, native = _inputs(tmp_path)
    monkeypatch.setattr(bundler.subprocess, "run", _fake_codesign([], timestamp=False))
    with pytest.raises(RuntimeError, match="secure timestamp"):
        _run_bundler(
            monkeypatch, pure, native, tmp_path / "out", "--codesign-identity", "Developer ID Application: Test"
        )
