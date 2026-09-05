#!/usr/bin/env python3
"""Package the tested QSA extension inside a platform-specific MTPLX wheel.

The pure Python wheel remains the fallback for other Python/OS platforms.
This builds a local artifact only; it never uploads or installs anything.
"""

import argparse
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
    parser.add_argument("native", type=Path)
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
    native_name, _, _, native_tags = parse_wheel_filename(args.native.name)
    if name != "mtplx" or core_tags != frozenset({Tag("py3", "none", "any")}):
        parser.error("The runtime input must be the pure Python MTPLX wheel")
    if native_name.replace("-", "_") != "mtplx_qsa_kernels" or len(native_tags) != 1:
        parser.error("Expected one tested mtplx_qsa_kernels platform wheel")
    tag = next(iter(native_tags))
    if not tag.platform.startswith("macosx_") or not tag.platform.endswith("_arm64"):
        parser.error("The native wheel must target Apple Silicon")
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f"mtplx-{version}-{tag}.whl"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    with WheelFile(args.runtime) as core, WheelFile(args.native) as native:
        metadata_name = next(n for n in native.namelist() if n.endswith(".dist-info/METADATA"))
        native_metadata = BytesParser().parsebytes(native.read(metadata_name))
        native_requirements = native_metadata.get_all("Requires-Dist", [])
        mlx_pins = [Requirement(r) for r in native_requirements if Requirement(r).name == "mlx"]
        if len(mlx_pins) != 1 or not str(mlx_pins[0].specifier).startswith("=="):
            parser.error("Native wheel must declare its exact MLX runtime ABI dependency")
        members = [n for n in native.namelist() if n.startswith("mtplx_qsa_kernels/")]
        for required in ("NOTICE", "LICENSE.txt", "MLX_LICENSE.txt"):
            if f"mtplx_qsa_kernels/{required}" not in members:
                parser.error(f"Native attribution is missing: {required}")
        if not any(n.endswith(".metallib") for n in members) or not any(n.endswith(".so") for n in members):
            parser.error("Native wheel lacks its extension or Metal library")
        provenance = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (args.runtime, args.native)}
        with WheelFile(output, "w") as bundled:
            for archive, names in ((core, core.namelist()), (native, members)):
                for name in names:
                    if name.endswith("/") or name.endswith(".dist-info/RECORD"):
                        continue
                    if name.startswith("/") or ".." in Path(name).parts:
                        raise ValueError(f"Unsafe archive path: {name}")
                    data = archive.read(name)  # WheelFile verifies source RECORD hashes.
                    if archive is native and args.codesign_identity and name.endswith((".so", ".dylib")):
                        data = sign_mach_o(name, data, args.codesign_identity)
                    if archive is core and name.endswith(".dist-info/WHEEL"):
                        lines = [line for line in data.decode().splitlines()
                                 if not line.startswith(("Tag:", "Root-Is-Purelib:"))]
                        data = ("\n".join([*lines, "Root-Is-Purelib: false", f"Tag: {tag}", ""])).encode()
                    elif archive is core and name.endswith(".dist-info/top_level.txt"):
                        data += b"mtplx_qsa_kernels\n"
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
