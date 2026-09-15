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
PLE_EXT = "mtplx_native_ple_cpu_rows/_ext.cpython-314-darwin.so"
PLE_DYLIB = "mtplx_native_ple_cpu_rows/libmtplx_native_ple_cpu_rows.dylib"

# PR #391 remainder port: the split-K QSA sparse-GQA decode extension is a
# second native wheel (mtplx_native_qsa) that also carries a metallib.
QSA_EXT = "mtplx_native_qsa/_ext.cpython-314-darwin.so"
QSA_DYLIB = "mtplx_native_qsa/libmtplx_native_qsa.dylib"
QSA_METALLIB = "mtplx_native_qsa/mtplx_native_qsa.metallib"


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


def _ple_input(tmp_path: Path) -> Path:
    return _write_wheel(
        tmp_path / "mtplx_native_ple_cpu_rows-9.9.9-cp314-cp314-macosx_15_0_arm64.whl",
        {
            "mtplx_native_ple_cpu_rows/__init__.py": b"",
            PLE_EXT: b"PLE-MACHO-EXT",
            PLE_DYLIB: b"PLE-MACHO-DYLIB",
            "mtplx_native_ple_cpu_rows-9.9.9.dist-info/METADATA": (
                b"Metadata-Version: 2.1\nName: mtplx-native-ple-cpu-rows\nVersion: 9.9.9\n"
                b"Requires-Dist: mlx==0.32.2\n"
            ),
            "mtplx_native_ple_cpu_rows-9.9.9.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
                b"Tag: cp314-cp314-macosx_15_0_arm64\n"
            ),
        },
    )


def _run_bundler(monkeypatch, pure: Path, native: Path, out: Path, *extra: str) -> Path:
    monkeypatch.setattr(sys, "argv", ["bundle", str(pure), str(native), "--out", str(out), *extra])
    bundler.main()
    return out / "mtplx-9.9.9-cp314-cp314-macosx_15_0_arm64.whl"


def _run_bundler_multi(monkeypatch, pure: Path, natives: list[Path], out: Path, *extra: str) -> Path:
    monkeypatch.setattr(
        sys, "argv",
        ["bundle", str(pure), *[str(n) for n in natives], "--out", str(out), *extra],
    )
    bundler.main()
    return out / "mtplx-9.9.9-cp314-cp314-macosx_15_0_arm64.whl"


def _qsa_native_input(tmp_path: Path) -> Path:
    return _write_wheel(
        tmp_path / "mtplx_native_qsa-9.9.9-cp314-cp314-macosx_15_0_arm64.whl",
        {
            "mtplx_native_qsa/__init__.py": b"",
            QSA_EXT: b"QSA-MACHO-EXT",
            QSA_DYLIB: b"QSA-MACHO-DYLIB",
            QSA_METALLIB: b"QSA-METALLIB",
            "mtplx_native_qsa-9.9.9.dist-info/METADATA": (
                b"Metadata-Version: 2.1\nName: mtplx-native-qsa\nVersion: 9.9.9\n"
                b"Requires-Dist: mlx==0.32.2\n"
            ),
            "mtplx_native_qsa-9.9.9.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
                b"Tag: cp314-cp314-macosx_15_0_arm64\n"
            ),
        },
    )


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


def test_qsa_sparse_gqa_extension_is_bundled_and_signed_with_the_qsa_kernels(tmp_path, monkeypatch) -> None:
    # The PR #391 remainder QSA split-K decode lane ships a second native
    # extension, mtplx_native_qsa (its own .so + .dylib + .metallib). It must
    # be bundled and Developer-ID signed alongside the QSA kernels, or
    # notarization rejects its ad-hoc-signed Mach-O the way it rejected the
    # QSA kernels before ea2560a2.
    pure, native = _inputs(tmp_path)
    qsa = _qsa_native_input(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(bundler.subprocess, "run", _fake_codesign(calls))
    bundled = _run_bundler_multi(
        monkeypatch, pure, [native, qsa], tmp_path / "out",
        "--codesign-identity", "Developer ID Application: Test",
    )
    with zipfile.ZipFile(bundled) as archive:
        assert archive.read(EXT) == b"SIGNED:MACHO-EXT"
        assert archive.read(QSA_EXT) == b"SIGNED:QSA-MACHO-EXT"
        assert archive.read(QSA_DYLIB) == b"SIGNED:QSA-MACHO-DYLIB"
        assert archive.read(QSA_METALLIB) == b"QSA-METALLIB"  # metallib is not a Mach-O
        top_level = archive.read("mtplx-9.9.9.dist-info/top_level.txt").decode()
        record = archive.read("mtplx-9.9.9.dist-info/RECORD").decode().splitlines()
    assert "mtplx_qsa_kernels" in top_level.split()
    assert "mtplx_native_qsa" in top_level.split()
    hashes = {line.split(",")[0]: line.split(",")[1] for line in record if line}
    assert hashes[QSA_EXT] == _record_hash(b"SIGNED:QSA-MACHO-EXT")
    with WheelFile(bundled) as reopened:  # the rewritten RECORD verifies
        assert reopened.read(QSA_EXT) == b"SIGNED:QSA-MACHO-EXT"
    signed = [Path(call[-1]).name for call in calls if "--sign" in call]
    assert "_ext.cpython-314-darwin.so" in signed  # QSA kernels
    assert "libmtplx_native_qsa.dylib" in signed  # split-K decode ext


def test_qsa_native_alone_is_rejected_without_the_required_qsa_kernels(tmp_path, monkeypatch) -> None:
    # mtplx_native_qsa is an OPTIONAL second extension; the required base is
    # mtplx_qsa_kernels. The bundler refuses a native input set that omits it.
    pure, _native = _inputs(tmp_path)
    qsa = _qsa_native_input(tmp_path)
    with pytest.raises(SystemExit):
        _run_bundler_multi(monkeypatch, pure, [qsa], tmp_path / "out")


def test_qsa_native_missing_its_metallib_is_rejected(tmp_path, monkeypatch) -> None:
    # mtplx_native_qsa declares require_metallib=True (it carries a split-K
    # metallib), so a build that shipped only the .so is refused.
    pure, native = _inputs(tmp_path)
    qsa = _write_wheel(
        tmp_path / "mtplx_native_qsa-9.9.9-cp314-cp314-macosx_15_0_arm64.whl",
        {
            "mtplx_native_qsa/__init__.py": b"",
            QSA_EXT: b"QSA-MACHO-EXT",
            "mtplx_native_qsa-9.9.9.dist-info/METADATA": (
                b"Metadata-Version: 2.1\nName: mtplx-native-qsa\nVersion: 9.9.9\n"
                b"Requires-Dist: mlx==0.32.2\n"
            ),
            "mtplx_native_qsa-9.9.9.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\n"
                b"Tag: cp314-cp314-macosx_15_0_arm64\n"
            ),
        },
    )
    with pytest.raises(SystemExit):
        _run_bundler_multi(monkeypatch, pure, [native, qsa], tmp_path / "out")


def test_ple_cpu_rows_extension_is_bundled_and_signed_with_the_qsa_kernels(tmp_path, monkeypatch) -> None:
    # PR #475's cached async PLE lane ships a second native extension,
    # mtplx_native_ple_cpu_rows. It must be bundled and Developer-ID signed
    # alongside the QSA kernels, or notarization rejects its ad-hoc-signed
    # Mach-O the way it rejected the QSA kernels before ea2560a2.
    pure, native = _inputs(tmp_path)
    ple = _ple_input(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(bundler.subprocess, "run", _fake_codesign(calls))
    bundled = _run_bundler_multi(
        monkeypatch, pure, [native, ple], tmp_path / "out",
        "--codesign-identity", "Developer ID Application: Test",
    )
    with zipfile.ZipFile(bundled) as archive:
        assert archive.read(EXT) == b"SIGNED:MACHO-EXT"
        assert archive.read(PLE_EXT) == b"SIGNED:PLE-MACHO-EXT"
        assert archive.read(PLE_DYLIB) == b"SIGNED:PLE-MACHO-DYLIB"
        top_level = archive.read("mtplx-9.9.9.dist-info/top_level.txt").decode()
        record = archive.read("mtplx-9.9.9.dist-info/RECORD").decode().splitlines()
    assert "mtplx_qsa_kernels" in top_level.split()
    assert "mtplx_native_ple_cpu_rows" in top_level.split()
    hashes = {line.split(",")[0]: line.split(",")[1] for line in record if line}
    assert hashes[PLE_EXT] == _record_hash(b"SIGNED:PLE-MACHO-EXT")
    with WheelFile(bundled) as reopened:  # the rewritten RECORD verifies
        assert reopened.read(PLE_EXT) == b"SIGNED:PLE-MACHO-EXT"
    signed = [Path(call[-1]).name for call in calls if "--sign" in call]
    assert "_ext.cpython-314-darwin.so" in signed  # QSA
    assert "libmtplx_native_ple_cpu_rows.dylib" in signed  # PLE


def test_ple_cpu_rows_alone_is_rejected_without_the_required_qsa_kernels(tmp_path, monkeypatch) -> None:
    pure, _native = _inputs(tmp_path)
    ple = _ple_input(tmp_path)
    with pytest.raises(SystemExit):
        _run_bundler_multi(monkeypatch, pure, [ple], tmp_path / "out")
