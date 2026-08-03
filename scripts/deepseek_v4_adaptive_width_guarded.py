#!/usr/bin/env python3
"""Run the guarded adaptive-width bracket and persist restoration postflight."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
# Guard deployment locations are supplied by the operator.  These defaults are
# intentionally relative so an unconfigured checkout fails in the guarded
# runner rather than encoding a particular developer machine.
VENV = Path(os.environ.get("MTPLX_DSV4_PYTHON", "python3"))
RUN_GUARDED = Path(os.environ.get("MTPLX_DSV4_GUARDED_RUNNER", "run_guarded.py"))
PLIST = Path(os.environ.get("MTPLX_DSV4_QUALITY_PLIST", "com.tea.qwen.plist"))
BENCH = Path(os.environ.get("MTPLX_DSV4_BENCH_DIR", "bench/deepseek-v4"))
WRAPPER_ENV = "MTPLX_DSV4_ADAPTIVE_WIDTH_POSTFLIGHT_WRAPPER"
RECEIPT_KIND = "deepseek_v4_adaptive_width_guarded_postflight"
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
INVALID_TAG_RECEIPT_PREFIX = "adaptive-width-invalid-tag-"
REQUIRED_PROBES = (
    "lock_free",
    "wired_limit_mb",
    "quality_models",
    "quality_ready_chat",
)


def _validate_tag(tag: str) -> str:
    if (
        not isinstance(tag, str)
        or tag in {"", ".", ".."}
        or TAG_PATTERN.fullmatch(tag) is None
    ):
        raise ValueError("invalid bracket tag: expected a safe basename")
    return tag


def _tag_sha256(tag: object) -> str:
    encoded = (
        tag.encode("utf-8", errors="surrogatepass")
        if isinstance(tag, str)
        else repr(tag).encode("utf-8", errors="surrogatepass")
    )
    return hashlib.sha256(encoded).hexdigest()


def _invalid_tag_receipt_path(bench_dir: Path, tag: object) -> Path:
    digest = _tag_sha256(tag)
    name = f"{INVALID_TAG_RECEIPT_PREFIX}{digest}-pid-{os.getpid()}-postflight.json"
    return bench_dir / name


def _postflight_collector():
    path = HERE / "deepseek_v4_moe_tail_guarded_bracket.py"
    spec = importlib.util.spec_from_file_location("_dsv4_shared_postflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    collector = getattr(module, "collect_postflight", None)
    if not callable(collector):
        raise TypeError("shared postflight module has no collector")
    return collector()


def _command(tag: str) -> list[str]:
    return [
        str(VENV),
        str(RUN_GUARDED),
        "--plist",
        str(PLIST),
        "--timeout-seconds",
        "300",
        "--lock-timeout-seconds",
        "3600",
        "--child-timeout-seconds",
        "7200",
        "--",
        "/bin/zsh",
        str(HERE / "deepseek_v4_adaptive_width_arms.sh"),
        tag,
    ]


def _write_receipt(path: Path, receipt: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _read_primary(path: Path) -> tuple[dict | None, str | None, str | None]:
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise TypeError("primary receipt is not an object")
        if payload.get("receipt_role") != "adaptive_width_performance_bracket":
            raise ValueError("primary receipt role is invalid")
        if payload.get("status") != 0:
            raise ValueError("primary receipt status is not zero")
        return payload, hashlib.sha256(encoded).hexdigest(), None
    except Exception as error:
        return None, None, f"{type(error).__name__}: {error}"


def _validate_postflight(payload) -> tuple[dict, list[str]]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return {}, ["postflight result is not an object"]
    normalized = {}
    for name in REQUIRED_PROBES:
        row = payload.get(name)
        if not isinstance(row, dict) or type(row.get("ok")) is not bool:
            errors.append(f"postflight probe {name} is missing or malformed")
            continue
        normalized[name] = row
        if row["ok"] is not True:
            errors.append(f"postflight probe {name} failed")
    return normalized, errors


def run(
    tag: str,
    *,
    run_command=subprocess.run,
    postflight_collector=_postflight_collector,
    bench_dir: Path = BENCH,
) -> int:
    bench_dir = Path(bench_dir)
    tag_sha256 = _tag_sha256(tag)
    tag_validation_error = None
    try:
        valid_tag = _validate_tag(tag)
    except ValueError as error:
        valid_tag = None
        tag_validation_error = f"{type(error).__name__}: {error}"

    child_started = valid_tag is not None
    child_error = None
    if child_started:
        environment = dict(os.environ)
        environment[WRAPPER_ENV] = "1"
        try:
            completed = run_command(_command(valid_tag), check=False, env=environment)
            child_exit_code = int(completed.returncode)
        except Exception as error:
            child_exit_code = 1
            child_error = f"{type(error).__name__}: {error}"
        primary, primary_sha256, primary_error = _read_primary(
            bench_dir / f"{valid_tag}.json"
        )
        receipt_path = bench_dir / f"{valid_tag}-postflight.json"
    else:
        child_exit_code = None
        primary = None
        primary_sha256 = None
        primary_error = "primary receipt skipped because bracket tag is invalid"
        receipt_path = _invalid_tag_receipt_path(bench_dir, tag)

    try:
        raw_postflight = postflight_collector()
        postflight, postflight_errors = _validate_postflight(raw_postflight)
    except Exception as error:
        postflight = {}
        postflight_errors = [f"{type(error).__name__}: {error}"]

    errors = list(postflight_errors)
    if tag_validation_error is not None:
        errors.append(tag_validation_error)
    if child_started and child_exit_code != 0:
        errors.append(f"guarded child exited {child_exit_code}")
    if child_error is not None:
        errors.append(child_error)
    if primary_error is not None:
        errors.append(primary_error)
    status = int(bool(errors))
    receipt = {
        "kind": RECEIPT_KIND,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "tag": valid_tag,
        "tag_valid": valid_tag is not None,
        "tag_sha256": tag_sha256,
        "tag_validation_error": tag_validation_error,
        "guarded_child_started": child_started,
        "guarded_child_exit_code": child_exit_code,
        "primary_receipt": primary,
        "primary_receipt_sha256": primary_sha256,
        "primary_receipt_error": primary_error,
        "postflight": postflight,
        "postflight_ok": not postflight_errors,
        "validation_errors": errors,
        "status": status,
    }
    _write_receipt(receipt_path, receipt)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tag",
        nargs="?",
        default=f"adaptive-width-policy-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
    )
    args = parser.parse_args()
    return run(args.tag)


if __name__ == "__main__":
    raise SystemExit(main())
