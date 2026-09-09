#!/usr/bin/env python3
"""Package the tested QSA extension inside a platform-specific MTPLX wheel.

The pure Python wheel remains the fallback for other Python/OS platforms.
This builds a local artifact only; it never uploads or installs anything.
"""

import argparse
import contextlib
from email.parser import BytesParser
from email.policy import compat32
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from packaging.tags import Tag
from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
from wheel.wheelfile import WheelFile


# The native extensions the runtime wheel may carry, each in its own platform
# wheel. mtplx_qsa_kernels is the metallib-bearing QSA lane (ea2560a2);
# mtplx_native_qsa is the metallib-bearing split-K QSA sparse-GQA decode
# extension the MTPLX_QSA_SPARSE_DECODE lane needs; mtplx_native_ple_cpu_rows is
# the CPU-stream PLE row extension the cached async PLE lane (PR #475) needs.
# Every Mach-O member (.so/.dylib) of each is Developer-ID + hardened-runtime +
# secure-timestamp signed before packaging, so notarization does not reject an
# ad-hoc-signed member found inside the zip.
_KNOWN_NATIVE = {
    "mtplx_qsa_kernels": {
        "required_files": ("NOTICE", "LICENSE.txt", "MLX_LICENSE.txt"),
        "require_metallib": True,
    },
    "mtplx_native_qsa": {
        "required_files": (),
        "require_metallib": True,
    },
    "mtplx_native_ple_cpu_rows": {
        "required_files": (),
        "require_metallib": False,
    },
}
_REQUIRED_NATIVE = "mtplx_qsa_kernels"


def sign_mach_o(name: str, data: bytes, identity: str) -> bytes:
    """Return the Mach-O member re-signed the way the app bundle signs its binaries.

    Developer ID, hardened runtime and a secure timestamp: the notary service
    checks every Mach-O it finds inside the bundle, archives included, and
    rejected 2.11.2's first submission on the two linker-signed kernels here.
    """
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / Path(name).name
        path.write_bytes(data)
        subprocess.run(
            ["/usr/bin/codesign", "--force", "--options", "runtime", "--timestamp",
             "--sign", identity, str(path)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["/usr/bin/codesign", "--verify", "--strict", str(path)],
            check=True, capture_output=True, text=True,
        )
        shown = subprocess.run(
            ["/usr/bin/codesign", "-dvvv", str(path)],
            check=True, capture_output=True, text=True,
        )
        details = shown.stdout + shown.stderr
        if "Authority=Developer ID Application" not in details or "Timestamp=" not in details:
            raise RuntimeError(
                f"{name}: signature lacks a Developer ID authority or a secure timestamp\n{details}"
            )
        return path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    parser.add_argument(
        "native",
        type=Path,
        nargs="+",
        help="One or more tested native platform wheels "
        f"({', '.join(sorted(_KNOWN_NATIVE))}); {_REQUIRED_NATIVE} is required, "
        "every other must be a known native extension. All must share one "
        "Apple Silicon platform tag and the same exact MLX ABI pin.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--codesign-identity",
        default=None,
        help="Sign the native Mach-O members (Developer ID, hardened runtime, secure "
        "timestamp) before packaging. The app's signing pass cannot reach inside the "
        "wheel, and notarization rejects linker-signed kernels found there.",
    )
    args = parser.parse_args()
    name, version, _, core_tags = parse_wheel_filename(args.runtime.name)
    if name != "mtplx" or core_tags != frozenset({Tag("py3", "none", "any")}):
        parser.error("The runtime input must be the pure Python MTPLX wheel")

    # Validate every native wheel: a known extension, one Apple Silicon tag
    # shared across all, and an exact MLX ABI pin that agrees across all.
    natives: list[tuple[Path, str, Tag]] = []
    for native_path in args.native:
        native_name, _, _, native_tags = parse_wheel_filename(native_path.name)
        pkg = native_name.replace("-", "_")
        if pkg not in _KNOWN_NATIVE or len(native_tags) != 1:
            parser.error(
                "Each native input must be one tested platform wheel of a known "
                f"extension ({', '.join(sorted(_KNOWN_NATIVE))}); got {native_path.name}"
            )
        native_tag = next(iter(native_tags))
        if not native_tag.platform.startswith("macosx_") or not native_tag.platform.endswith("_arm64"):
            parser.error("Each native wheel must target Apple Silicon")
        natives.append((native_path, pkg, native_tag))
    if not any(pkg == _REQUIRED_NATIVE for _, pkg, _ in natives):
        parser.error(f"Expected the {_REQUIRED_NATIVE} platform wheel among the native inputs")
    seen_pkgs = [pkg for _, pkg, _ in natives]
    if len(set(seen_pkgs)) != len(seen_pkgs):
        parser.error("A native extension was passed more than once")
    tags = {native_tag for _, _, native_tag in natives}
    if len(tags) != 1:
        parser.error("All native wheels must share one platform tag")
    tag = next(iter(tags))

    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f"mtplx-{version}-{tag}.whl"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")

    with contextlib.ExitStack() as stack:
        core = stack.enter_context(WheelFile(args.runtime))
        native_requirements: list[str] = []
        mlx_pin: str | None = None
        # (WheelFile, pkg, members) per native, in input order.
        opened: list[tuple[WheelFile, str, list[str]]] = []
        for native_path, pkg, _native_tag in natives:
            native = stack.enter_context(WheelFile(native_path))
            metadata_name = next(n for n in native.namelist() if n.endswith(".dist-info/METADATA"))
            native_metadata = BytesParser().parsebytes(native.read(metadata_name))
            requirements = native_metadata.get_all("Requires-Dist", [])
            mlx_pins = [Requirement(r) for r in requirements if Requirement(r).name == "mlx"]
            if len(mlx_pins) != 1 or not str(mlx_pins[0].specifier).startswith("=="):
                parser.error(f"{pkg} must declare its exact MLX runtime ABI dependency")
            this_pin = str(mlx_pins[0].specifier)
            if mlx_pin is None:
                mlx_pin = this_pin
            elif this_pin != mlx_pin:
                parser.error("Native wheels disagree on the MLX ABI pin")
            for requirement in requirements:
                if requirement not in native_requirements:
                    native_requirements.append(requirement)
            members = [n for n in native.namelist() if n.startswith(f"{pkg}/")]
            spec = _KNOWN_NATIVE[pkg]
            for required in spec["required_files"]:
                if f"{pkg}/{required}" not in members:
                    parser.error(f"{pkg} attribution is missing: {required}")
            if not any(n.endswith(".so") for n in members):
                parser.error(f"{pkg} wheel lacks its extension")
            if spec["require_metallib"] and not any(n.endswith(".metallib") for n in members):
                parser.error(f"{pkg} wheel lacks its Metal library")
            opened.append((native, pkg, members))

        provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (args.runtime, *args.native)}
        native_pkgs = [pkg for _native, pkg, _members in opened]
        with WheelFile(output, "w") as bundled:
            sources: list[tuple[WheelFile, list[str], bool]] = [
                (core, core.namelist(), False)
            ]
            sources.extend((native, members, True) for native, _pkg, members in opened)
            for archive, names, is_native in sources:
                for name in names:
                    if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                        continue
                    if name.startswith("/") or ".." in Path(name).parts:
                        raise ValueError(f"Unsafe archive path: {name}")
                    data = archive.read(name)  # WheelFile verifies source RECORD hashes.
                    if is_native and args.codesign_identity and name.endswith((".so", ".dylib")):
                        data = sign_mach_o(name, data, args.codesign_identity)
                    if archive is core and name.endswith(".dist-info/WHEEL"):
                        lines = [line for line in data.decode().splitlines()
                                 if not line.startswith(("Tag:", "Root-Is-Purelib:"))]
                        data = ("\n".join([*lines, "Root-Is-Purelib: false", f"Tag: {tag}", ""])).encode()
                    elif archive is core and name.endswith(".dist-info/top_level.txt"):
                        data += ("".join(f"{pkg}\n" for pkg in native_pkgs)).encode()
                    elif archive is core and name.endswith(".dist-info/METADATA"):
                        metadata = BytesParser().parsebytes(data)
                        for requirement in native_requirements:
                            if requirement not in metadata.get_all("Requires-Dist", []):
                                metadata["Requires-Dist"] = requirement
                        # Core Metadata requirements must stay on one line;
                        # email's default folding inserts a newline inside
                        # environment markers that wheel validators reject.
                        data = metadata.as_bytes(policy=compat32.clone(max_line_length=0))
                    bundled.writestr(archive.getinfo(name), data)
            bundled.writestr("mtplx/native_build_receipt.json", json.dumps(provenance, indent=2))
            # WheelFile writes a new RECORD for the complete distribution.
    print(output)


if __name__ == "__main__":
    main()
