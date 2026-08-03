#!/usr/bin/env python3
"""Guard the attention-M3 bracket and attest Qwen restoration on every exit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _shared():
    path = HERE / "deepseek_v4_adaptive_width_guarded.py"
    spec = importlib.util.spec_from_file_location("_dsv4_shared_guard", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _configure(module) -> None:
    module.WRAPPER_ENV = "MTPLX_DSV4_ATTN_PROJ_WIDE_M3_POSTFLIGHT_WRAPPER"
    module.INVALID_TAG_RECEIPT_PREFIX = "attn-proj-wide-m3-invalid-tag-"

    def command(tag: str) -> list[str]:
        return [
            str(module.VENV),
            str(module.RUN_GUARDED),
            "--plist",
            str(module.PLIST),
            "--timeout-seconds",
            "300",
            "--lock-timeout-seconds",
            "3600",
            "--child-timeout-seconds",
            "7200",
            "--",
            "/bin/zsh",
            str(HERE / "deepseek_v4_attn_proj_wide_m3_arms.sh"),
            tag,
        ]

    def read_primary(path: Path):
        try:
            encoded = path.read_bytes()
            payload = json.loads(encoded)
            if not isinstance(payload, dict):
                raise TypeError("primary receipt is not an object")
            if payload.get("receipt_role") != "attn_proj_wide_m3_performance_bracket":
                raise ValueError("primary receipt role is invalid")
            if payload.get("status") != 0:
                raise ValueError("primary receipt status is not zero")
            return payload, hashlib.sha256(encoded).hexdigest(), None
        except Exception as error:
            return None, None, f"{type(error).__name__}: {error}"

    module._command = command
    module._read_primary = read_primary


def run(tag: str, **kwargs) -> int:
    module = _shared()
    _configure(module)
    return module.run(tag, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tag",
        nargs="?",
        default=f"attn-proj-wide-m3-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
    )
    return run(parser.parse_args().tag)


if __name__ == "__main__":
    raise SystemExit(main())
