"""External ``mlx-serve`` launch contract for DeepSeek V4 target-only MLX.

MTPLX does not implement the DeepSeek V4 target graph itself.  This module
keeps that boundary explicit: artifact recognition remains in MTPLX while the
OpenAI-compatible server process is the native ``mlx-serve`` executable.
"""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Mapping


BACKEND_ID = "deepseek_v4_mlxserve_ar"
BINARY_ENV = "MTPLX_MLX_SERVE_BIN"
CWD_ENV = "MTPLX_MLX_SERVE_CWD"
DEFAULT_CONTEXT_WINDOW = 8_192
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_CACHE_LIMIT_BYTES = 256 * 1024 * 1024


class DeepSeekV4MlxServeError(RuntimeError):
    """The external runtime could not be admitted safely."""


def _direct_executable(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        info = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise DeepSeekV4MlxServeError(
            f"mlx-serve executable is unavailable: {path}"
        ) from exc
    if not stat.S_ISREG(info.st_mode) or not os.access(resolved, os.X_OK):
        raise DeepSeekV4MlxServeError(
            f"mlx-serve path is not an executable regular file: {resolved}"
        )
    return resolved


def resolve_binary(env: Mapping[str, str] | None = None) -> Path:
    values = os.environ if env is None else env
    configured = str(values.get(BINARY_ENV) or "").strip()
    if configured:
        return _direct_executable(Path(configured))
    discovered = shutil.which("mlx-serve", path=values.get("PATH"))
    if not discovered:
        raise DeepSeekV4MlxServeError(
            "DeepSeek V4 requires mlx-serve on PATH or MTPLX_MLX_SERVE_BIN"
        )
    return _direct_executable(Path(discovered))


def resolve_working_directory(
    binary: Path,
    env: Mapping[str, str] | None = None,
) -> Path:
    values = os.environ if env is None else env
    configured = str(values.get(CWD_ENV) or "").strip()
    if configured:
        candidate = Path(configured).expanduser().resolve(strict=True)
        if not candidate.is_dir():
            raise DeepSeekV4MlxServeError(
                f"MTPLX_MLX_SERVE_CWD is not a directory: {candidate}"
            )
        return candidate

    # Source builds currently use cwd-relative dylib search paths.  Detect the
    # standard <repo>/zig-out/bin/mlx-serve layout without baking in a user path.
    parents = binary.parents
    if len(parents) >= 3 and parents[0].name == "bin" and parents[1].name == "zig-out":
        source_root = parents[2]
        if (source_root / "lib" / "mlx" / "lib" / "libmlx.dylib").is_file():
            return source_root
    return Path.cwd()


def child_environment(env: Mapping[str, str] | None = None) -> dict[str, str]:
    source = os.environ if env is None else env
    child = {
        str(key): str(value)
        for key, value in source.items()
        if not str(key).startswith(("MLX_SERVE_", "MLXSERVE_"))
        and key != "MTPLX_DSV4_WIRED"
    }
    # ``fit`` is the previously exercised residency setting for this roughly
    # 100 GB target-only artifact.  Keep it explicit instead of inheriting a
    # caller's ambient MLX_SERVE_WIRED state. This is a launch-safety default,
    # not a representative streaming-performance claim; callers may make an
    # explicit override via the non-filtered MTPLX_DSV4_WIRED variable.
    child["MLX_SERVE_WIRED"] = str(
        source.get("MTPLX_DSV4_WIRED", "fit")
    ).strip() or "fit"
    child["MLX_SERVE_CACHE_LIMIT"] = str(DEFAULT_CACHE_LIMIT_BYTES)
    return child


def build_command(
    *,
    binary: Path,
    model: str,
    host: str,
    port: int,
    context_window: int | None,
    api_key: str | None,
) -> list[str]:
    context = DEFAULT_CONTEXT_WINDOW if context_window is None else int(context_window)
    if context <= 0 or context > 1_048_576:
        raise DeepSeekV4MlxServeError(
            "DeepSeek V4 context window must be between 1 and 1048576"
        )
    command = [
        str(binary),
        "--model",
        str(model),
        "--serve",
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--no-pld",
        "--no-decode-attn-quant",
        "--no-vision",
        "--ctx-size",
        str(context),
        "--timeout",
        str(DEFAULT_TIMEOUT_SECONDS),
        "--max-resident-models",
        "1",
        "--max-resident-mem",
        "110GB",
    ]
    if api_key:
        command.extend(("--api-key", str(api_key)))
    return command
