from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_toml(name: str) -> dict:
    return tomllib.loads((ROOT / name).read_text(encoding="utf-8"))


def test_mlx_runtime_and_lock_are_pinned_to_0322() -> None:
    project = _read_toml("pyproject.toml")["project"]
    lock_packages = {
        package["name"]: package for package in _read_toml("uv.lock")["package"]
    }

    assert (
        "mlx==0.32.2; sys_platform == 'darwin' and platform_machine == 'arm64'"
        in project["dependencies"]
    )

    mlx = lock_packages["mlx"]
    mlx_metal = lock_packages["mlx-metal"]
    assert mlx["version"] == "0.32.2"
    assert mlx_metal["version"] == mlx["version"]
    assert any(dependency["name"] == "mlx-metal" for dependency in mlx["dependencies"])

    mtplx_metadata = lock_packages["mtplx"]["metadata"]["requires-dist"]
    mlx_requirement = next(
        requirement for requirement in mtplx_metadata if requirement["name"] == "mlx"
    )
    assert mlx_requirement["specifier"] == "==0.32.2"
