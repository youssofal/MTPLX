#!/usr/bin/env python3
"""Consume one GPU-guard attestation and run the closed DFlash2 sweep."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--widths", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--smoke-tokens", type=int, choices=(32,))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def write_atomic_json(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(
                receipt,
                stream,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from deepseek_v4_guard_window import issue_guard_window

    guard_path, guard_sha256 = issue_guard_window(
        expected_lock=Path("/tmp/mtplx-gpu-exclusive.lock")
    )
    from mtplx.benchmarks.runners.dflash2_depth_sweep import run_cli_sweep

    token_count = args.smoke_tokens or 1024
    receipt = run_cli_sweep(args, token_count=token_count)
    receipt["guard_window"] = {
        "path": str(guard_path),
        "sha256": guard_sha256,
    }
    write_atomic_json(args.output, receipt)
    return 0 if receipt.get("selection") is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
