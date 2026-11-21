#!/usr/bin/env python3
"""Put the native executor (mtplx._engine) into an MTPLX wheel.

The compiled module is built once per Python version, against the exact MLX
the package pins, and travels inside a platform wheel. An Apple Silicon source install builds
the matching module through setuptools. The pure wheel is a release assembly input.

    native/package.py pins
        The build requirements, for pip:  mlx==<pin> nanobind==<version>

    native/package.py build --out DIR [--no-selftest]
        Build for the interpreter running this script and print the module's
        path.  The interpreter must have the pinned mlx and nanobind.

    native/package.py wheel BASE MODULE (--out DIR | --replace)
                            [--platform TAG] [--codesign-identity ID]
        BASE is the pure wheel (a new platform wheel is written) or a platform
        wheel of the same Python (the module is added to it).  Prints the path
        of the wheel written.

    native/package.py check WHEEL
        Verify a wheel that carries the module: record hashes, tags, machine
        type, oldest macOS, library search path, signature.

    <an environment with the wheel installed>/bin/python native/package.py smoke
                            [--require-selftest]
        The installed module loads, shares MLX's array type and replays the
        self-test graphs bit for bit.  Without --require-selftest a failed
        replay is reported and does not fail the command (a build machine whose
        GPU is virtual); a module that does not load always fails it.

Builds local artifacts only; uploads and installs nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# mlx.core and this module share MLX's array type through nanobind, which only
# works when both were built with the same nanobind.  One entry per MLX the
# package has pinned; the version is the one MLX's own build fetches.
NANOBIND_FOR_MLX = {"0.32.2": "2.15.0"}

MODULE_NAME = re.compile(r"^_engine\.cpython-(3)(\d+)-darwin\.so$")
MLX_LIBRARY_PATH = "@loader_path/../mlx/lib"


class Problem(Exception):
    pass


def mlx_pin() -> str:
    from packaging.requirements import Requirement

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    pins = []
    for line in project["dependencies"]:
        requirement = Requirement(line)
        if requirement.name == "mlx":
            pins += [s.version for s in requirement.specifier if s.operator == "=="]
    if len(pins) != 1:
        raise Problem("pyproject.toml must pin mlx to one exact version (mlx==X)")
    return pins[0]


def nanobind_pin(mlx_version: str) -> str:
    try:
        return NANOBIND_FOR_MLX[mlx_version]
    except KeyError:
        raise Problem(
            f"no nanobind version on file for mlx {mlx_version}: add the one MLX {mlx_version} "
            "is built with to NANOBIND_FOR_MLX in native/package.py"
        ) from None


def run(command: list[str]) -> str:
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise Problem(f"{' '.join(command)}\n{done.stdout}{done.stderr}")
    return done.stdout + done.stderr


def mach_o_facts(path: Path) -> dict:
    """Machine type, oldest macOS and library search paths of a built module."""

    commands = run(["/usr/bin/otool", "-l", str(path)])
    minos = re.search(r"cmd LC_BUILD_VERSION\n(?:.*\n){1,3}?\s+minos (\d+)\.(\d+)", commands)
    rpaths = re.findall(r"cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.+?) \(offset \d+\)", commands)
    kinds = run(["/usr/bin/lipo", "-archs", str(path)]).split()
    return {
        "archs": kinds,
        "minos": (int(minos.group(1)), int(minos.group(2))) if minos else None,
        "rpaths": rpaths,
    }


def check_module(path: Path) -> dict:
    facts = mach_o_facts(path)
    if facts["archs"] != ["arm64"]:
        raise Problem(f"{path.name}: built for {facts['archs']}, expected arm64 only")
    if facts["minos"] is None:
        raise Problem(f"{path.name}: no oldest-macOS stamp")
    absolute = [p for p in facts["rpaths"] if Path(p).is_absolute()]
    if absolute:
        raise Problem(f"{path.name}: library search path leaves the wheel: {absolute}")
    if MLX_LIBRARY_PATH not in facts["rpaths"]:
        raise Problem(f"{path.name}: does not look for MLX's library in {MLX_LIBRARY_PATH}")
    return facts


def signature(path: Path) -> str:
    """'developer-id', 'adhoc' or raises when the signature does not verify."""

    run(["/usr/bin/codesign", "--verify", "--strict", str(path)])
    details = run(["/usr/bin/codesign", "-dvvv", str(path)])
    if "Authority=Developer ID Application" in details and "Timestamp=" in details:
        return "developer-id"
    if "Signature=adhoc" in details:
        return "adhoc"
    return "other"


def sign(path: Path, identity: str) -> None:
    """Developer ID, hardened runtime, secure timestamp: what the notary service
    asks of every Mach-O it finds inside the app, wheels included."""

    command = ["/usr/bin/codesign", "--force", "--options", "runtime", "--sign", identity]
    if identity != "-":
        command.append("--timestamp")
    run([*command, str(path)])
    kind = signature(path)
    if identity != "-" and kind != "developer-id":
        raise Problem(f"{path.name}: signature lacks a Developer ID authority or a secure timestamp")


def platform_version(platform: str) -> tuple[int, int]:
    match = re.fullmatch(r"macosx_(\d+)_(\d+)_arm64", platform)
    if not match:
        raise Problem(f"platform tag {platform!r} is not macosx_<major>_<minor>_arm64")
    return int(match.group(1)), int(match.group(2))


# ---------------------------------------------------------------- commands


def cmd_pins(_args) -> int:
    pin = mlx_pin()
    print(f"mlx=={pin} nanobind=={nanobind_pin(pin)}")
    return 0


def cmd_build(args) -> int:
    from importlib import metadata

    pin = mlx_pin()
    have = metadata.version("mlx")
    if have != pin:
        # The loader refuses a module built against another MLX, so a module
        # built here would ship and never run.
        raise Problem(f"this interpreter has mlx {have}; the package pins mlx=={pin}")
    if not os.environ.get("MTPLX_NANOBIND_CMAKE_DIR"):
        want = nanobind_pin(pin)
        try:
            found = metadata.version("nanobind")
        except metadata.PackageNotFoundError:
            found = "none"
        if found != want:
            raise Problem(
                f"this interpreter has nanobind {found}; mlx {pin} needs nanobind=={want} "
                "(or point MTPLX_NANOBIND_CMAKE_DIR at that version's cmake directory)"
            )
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = [str(HERE / "build.sh"), sys.executable, "--stage", str(out)]
    command.append("--no-selftest" if args.no_selftest else "--no-swap")
    done = subprocess.run(command, stdout=sys.stderr)
    if done.returncode != 0:
        return done.returncode
    module = (
        out
        / sysconfig.get_config_var("SOABI")
        / "out"
        / f"_engine{sysconfig.get_config_var('EXT_SUFFIX')}"
    )
    check_module(module)
    print(module)
    return 0


def cmd_wheel(args) -> int:
    from packaging.tags import Tag
    from packaging.utils import parse_wheel_filename
    from wheel.wheelfile import WheelFile

    base = Path(args.base).resolve()
    module = Path(args.module).resolve()
    named = MODULE_NAME.match(module.name)
    if not named:
        raise Problem(f"{module.name} is not _engine.cpython-3X-darwin.so")
    python_tag = f"cp{named.group(1)}{named.group(2)}"
    facts = check_module(module)

    name, version, _build, tags = parse_wheel_filename(base.name)
    if name != "mtplx":
        raise Problem("the base wheel must be an MTPLX wheel")
    pure = tags == frozenset({Tag("py3", "none", "any")})
    if pure:
        platform = args.platform or f"macosx_{facts['minos'][0]}_0_arm64"
        tag = Tag(python_tag, python_tag, platform)
    else:
        if len(tags) != 1:
            raise Problem(f"{base.name}: expected one tag")
        tag = next(iter(tags))
        if (tag.interpreter, tag.abi) != (python_tag, python_tag):
            raise Problem(f"{base.name} is for {tag.interpreter}; the module is for {python_tag}")
        if args.platform and args.platform != tag.platform:
            raise Problem(f"{base.name} is already tagged {tag.platform}")
    if platform_version(tag.platform) < facts["minos"]:
        oldest = ".".join(map(str, facts["minos"]))
        raise Problem(f"the module needs macOS {oldest}; the wheel is tagged {tag.platform}")

    if args.replace:
        if pure:
            raise Problem("--replace is for adding the module to a platform wheel")
        destination = base
    else:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        destination = Path(args.out).resolve() / f"mtplx-{version}-{tag}.whl"
        if destination.exists():
            raise Problem(f"refusing to overwrite {destination}")

    member = f"mtplx/{module.name}"
    with tempfile.TemporaryDirectory() as scratch:
        staged_module = Path(scratch) / module.name
        staged_module.write_bytes(module.read_bytes())
        if args.codesign_identity:
            sign(staged_module, args.codesign_identity)
        else:
            signature(staged_module)  # the linker's own signature must verify
        # WheelFile names the metadata directory after the file it writes, so
        # the staged wheel carries its final name.
        staged_wheel = Path(scratch) / "wheel" / destination.name
        staged_wheel.parent.mkdir()
        with WheelFile(base) as source, WheelFile(staged_wheel, "w") as target:
            for info in source.infolist():
                entry = info.filename
                if entry.endswith("/") or entry.endswith(".dist-info/RECORD"):
                    continue
                if entry.startswith("/") or ".." in Path(entry).parts:
                    raise Problem(f"unsafe archive path: {entry}")
                if entry == member:
                    continue  # a rebuilt module replaces the one already there
                data = source.read(entry)  # WheelFile verifies the recorded hash
                if entry.endswith(".dist-info/WHEEL"):
                    # Headers only: a blank line would end them for a reader,
                    # and hide the tag written after it.
                    lines = [
                        line
                        for line in data.decode().splitlines()
                        if line.strip() and not line.startswith(("Tag:", "Root-Is-Purelib:"))
                    ]
                    data = "\n".join([*lines, "Root-Is-Purelib: false", f"Tag: {tag}", ""]).encode()
                target.writestr(info, data)
            entry_info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            entry_info.compress_type = zipfile.ZIP_DEFLATED
            entry_info.external_attr = (0o100755 & 0xFFFF) << 16
            target.writestr(entry_info, staged_module.read_bytes())
        os.replace(staged_wheel, destination)
    print(destination)
    return 0


def inspect_wheel(path: Path) -> dict:
    from email.parser import BytesParser

    from packaging.utils import parse_wheel_filename
    from wheel.wheelfile import WheelFile

    name, version, _build, tags = parse_wheel_filename(path.name)
    report: dict = {"wheel": path.name, "version": str(version), "modules": [], "problems": []}
    problems = report["problems"]
    if name != "mtplx" or len(tags) != 1:
        raise Problem(f"{path.name}: expected an MTPLX wheel with one tag")
    tag = next(iter(tags))
    with WheelFile(path) as wheel, tempfile.TemporaryDirectory() as scratch:
        for entry in wheel.namelist():
            if not entry.endswith("/"):
                wheel.read(entry)  # raises when a recorded hash does not match
        metadata = BytesParser().parsebytes(
            wheel.read(next(n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")))
        )
        requirements = [r.replace(" ", "") for r in metadata.get_all("Requires-Dist", [])]
        report["mlx"] = sorted({r.split(";")[0] for r in requirements if r.startswith("mlx==")})
        declared = BytesParser().parsebytes(
            wheel.read(next(n for n in wheel.namelist() if n.endswith(".dist-info/WHEEL")))
        )
        if declared.get_all("Tag", []) != [str(tag)]:
            problems.append(f"file name says {tag}, WHEEL says {declared.get_all('Tag', [])}")
        if str(declared.get("Root-Is-Purelib", "")).lower() != "false":
            problems.append("WHEEL does not say Root-Is-Purelib: false")
        members = [n for n in wheel.namelist() if n.startswith("mtplx/_engine.")]
        if not members:
            problems.append("no mtplx/_engine module inside")
        for member in members:
            named = MODULE_NAME.match(Path(member).name)
            extracted = Path(scratch) / Path(member).name
            extracted.write_bytes(wheel.read(member))
            entry: dict = {"member": member}
            try:
                facts = check_module(extracted)
                entry["minos"] = ".".join(map(str, facts["minos"]))
                entry["signature"] = signature(extracted)
                if not named or tag.interpreter != f"cp{named.group(1)}{named.group(2)}":
                    problems.append(f"{member} does not match the wheel's Python tag {tag.interpreter}")
                if tag.abi != tag.interpreter:
                    problems.append(f"abi tag {tag.abi} is not {tag.interpreter}")
                if platform_version(tag.platform) < facts["minos"]:
                    problems.append(f"{member} needs macOS {entry['minos']}, tag is {tag.platform}")
                mode = wheel.getinfo(member).external_attr >> 16
                if not mode & 0o111:
                    problems.append(f"{member} is not executable in the archive")
            except Problem as exc:
                problems.append(str(exc))
            report["modules"].append(entry)
    if len(report["mlx"]) != 1:
        problems.append(f"the wheel must require one exact mlx, found {report['mlx']}")
    return report


def cmd_check(args) -> int:
    report = inspect_wheel(Path(args.wheel).resolve())
    print(json.dumps(report, indent=2))
    return 1 if report["problems"] else 0


def cmd_smoke(args) -> int:
    import mtplx

    package = Path(mtplx.__file__).resolve().parent
    if package.parent == ROOT:
        raise Problem("mtplx was imported from the source tree; run this with an environment "
                      "where the wheel is installed")
    try:
        import mlx.core as mx

        mx.eval(mx.arange(4) + 1)
    except Exception as exc:  # a machine MLX itself cannot run on judges nothing
        print(f"::warning::MLX does not run on this machine ({exc!r}); the module was not judged")
        return 0

    from mtplx.engine import runtime, selftest

    if not runtime.available():
        raise Problem(f"the installed module does not load: {runtime.unavailable()}")
    module = runtime.native_module()
    if Path(module.__file__).resolve().parent != package:
        raise Problem(f"the module was loaded from {module.__file__}, not from the installed package")
    print(f"{module.__file__}: version {module.__version__}, built against mlx "
          f"{module.BUILT_AGAINST_MLX} and nanobind {module.BUILT_AGAINST_NANOBIND}")
    status = selftest.main([])
    if status != 0 and not args.require_selftest:
        print("::warning::the module loads, but the self-test did not pass on this machine's GPU")
        return 0
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("pins", help="print the build requirements for pip")

    build = commands.add_parser("build", help="build the module for this interpreter")
    build.add_argument("--out", required=True, help="build directory (outside the source tree)")
    build.add_argument(
        "--no-selftest",
        action="store_true",
        help="skip the bit-for-bit self-test (a build machine without an Apple GPU)",
    )

    wheel = commands.add_parser("wheel", help="write a platform wheel that carries the module")
    wheel.add_argument("base", help="the pure MTPLX wheel, or a platform wheel of the same Python")
    wheel.add_argument("module", help="the module printed by 'build'")
    where = wheel.add_mutually_exclusive_group(required=True)
    where.add_argument("--out", help="directory for the new wheel")
    where.add_argument("--replace", action="store_true", help="add the module to BASE in place")
    wheel.add_argument("--platform", help="platform tag for a wheel made from the pure one")
    wheel.add_argument(
        "--codesign-identity",
        help="sign the module (hardened runtime, secure timestamp) before it is packaged; "
        "the app's signing pass cannot reach inside a wheel",
    )

    check = commands.add_parser("check", help="verify a wheel that carries the module")
    check.add_argument("wheel")

    smoke = commands.add_parser("smoke", help="load and self-test the installed module")
    smoke.add_argument("--require-selftest", action="store_true")

    args = parser.parse_args(argv)
    try:
        handlers = {
            "pins": cmd_pins,
            "build": cmd_build,
            "wheel": cmd_wheel,
            "check": cmd_check,
            "smoke": cmd_smoke,
        }
        return handlers[args.command](args)
    except Problem as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
