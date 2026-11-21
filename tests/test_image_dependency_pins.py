"""Every install channel gets the image library the vision gates ran on.

pip, Homebrew and the app resolve `pyproject.toml`'s requirement from PyPI; only `uv sync` reads `uv.lock`. A
Pillow release can change the pixels a resized screenshot gives the vision tower (12.3 to 13 did, for images the
grid rounding enlarges), so the requirement must admit the locked version and stop before the next major one.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]
PIXEL_DEPENDENCIES = ("pillow",)


def _requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    found = {}
    for line in project["dependencies"]:
        req = Requirement(line)
        found[req.name.lower()] = req
    return found


def _locked():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    return {pkg["name"].lower(): Version(pkg["version"]) for pkg in lock["package"]}


def test_install_channels_resolve_the_locked_image_library():
    requirements, locked = _requirements(), _locked()
    for name in PIXEL_DEPENDENCIES:
        spec = requirements[name].specifier
        version = locked[name]
        assert spec.contains(version), f"{name}: uv.lock has {version}, pyproject asks for {spec}"
        next_major = Version(f"{version.major + 1}")
        assert not spec.contains(next_major), (
            f"{name}: pyproject ({spec}) lets pip, Homebrew and the app install {next_major}, "
            f"while every gate ran on the locked {version}")
