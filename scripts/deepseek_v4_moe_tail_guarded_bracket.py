#!/usr/bin/env python3
"""Run the MoE-tail bracket, then attest that the shared service is restored.

``run_guarded.py`` is deliberately the only process that owns the Quality
service lifecycle.  This wrapper only waits for that canonical child to exit
and performs read-only postflight checks while holding the canonical GPU lock;
it never starts, stops, or repairs a service itself.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


VENV_PYTHON = Path("/Users/davidtai/projects/OpenSourceWTF/mtplx-hy3-ssd/.venv/bin/python")
RUN_GUARDED = Path("/Users/davidtai/projects/OpenSourceWTF/bench/laguna/run_guarded.py")
QUALITY_PLIST = Path("/Users/davidtai/Library/LaunchAgents/com.tea.qwen.plist")
LOCK_PATH = Path("/tmp/mtplx-gpu-exclusive.lock")
ARMS = Path(__file__).with_name("deepseek_v4_moe_tail_arms.sh")
DEFAULT_BENCH_DIR = Path("/Users/davidtai/projects/OpenSourceWTF/bench/deepseek-v4")
QUALITY_MODEL = "mtplx-qwen36-27b-optimized-quality"
WIRED_LIMIT_MB = 114688
WRAPPER_ENV = "MTPLX_DSV4_MOE_TAIL_POSTFLIGHT_WRAPPER"


def _failed_check(error: BaseException, *, context: str | None = None) -> dict[str, Any]:
    message = str(error)
    if context is not None:
        message = f"{context}: {message}"
    return {
        "ok": False,
        "error": message,
        "error_type": type(error).__name__,
    }


def _safe_check(check: Any) -> dict[str, Any]:
    try:
        result = check()
    except Exception as error:
        return _failed_check(error)
    if not isinstance(result, dict):
        return {
            "ok": False,
            "error": f"probe returned {type(result).__name__}, expected an object",
            "error_type": "MalformedProbeResult",
        }
    return result


def _check_wired_limit() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
            check=False,
            capture_output=True,
            text=True,
        )
        value = completed.stdout.strip()
        if completed.returncode != 0:
            return {
                "ok": False,
                "error": completed.stderr.strip() or "sysctl failed",
                "exit_code": completed.returncode,
            }
        observed = int(value)
        return {"ok": observed == WIRED_LIMIT_MB, "value": observed}
    except (OSError, ValueError) as error:
        return {"ok": False, "error": str(error)}


def _request_json(path: str, *, payload: dict[str, Any] | None, timeout: float) -> Any:
    request = urllib.request.Request(
        f"http://127.0.0.1:8080{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={} if payload is None else {"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _check_quality_models() -> dict[str, Any]:
    try:
        payload = _request_json("/v1/models", payload=None, timeout=10)
        models = [entry["id"] for entry in payload["data"]]
        return {"ok": models == [QUALITY_MODEL], "models": models}
    except Exception as error:
        return _failed_check(error, context="malformed /v1/models response")


def _check_quality_ready_chat() -> dict[str, Any]:
    try:
        payload = _request_json(
            "/v1/chat/completions",
            payload={
                "model": QUALITY_MODEL,
                "messages": [{"role": "user", "content": "Say READY"}],
                "max_tokens": 8,
                "temperature": 0,
            },
            timeout=60,
        )
        choice = payload["choices"][0]
        content_value = choice["message"]["content"]
        if not isinstance(content_value, str):
            raise TypeError(
                "chat choice message content must be a string, got "
                f"{type(content_value).__name__}"
            )
        content = content_value.strip()
        finish_reason = choice["finish_reason"]
        return {
            "ok": content == "READY" and finish_reason == "stop",
            "content": content,
            "finish_reason": finish_reason,
        }
    except Exception as error:
        return _failed_check(error, context="malformed READY chat response")


def collect_postflight() -> dict[str, dict[str, Any]]:
    """Hold the canonical lock across all read-only restoration probes."""

    postflight: dict[str, dict[str, Any]] = {}
    lock_file = None
    try:
        lock_file = LOCK_PATH.open("rb")
        opened = os.fstat(lock_file.fileno())
        resolved = LOCK_PATH.resolve(strict=True)
        path_stat = resolved.stat()
        if (opened.st_dev, opened.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            raise RuntimeError("canonical lock identity changed while opening")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception as error:
        if lock_file is not None:
            lock_file.close()
        postflight["lock_free"] = {
            **_failed_check(error, context="canonical lock acquisition failed"),
            "requested_path": str(LOCK_PATH),
            "acquired_nonblocking": False,
            "held_through_probes": False,
            "released_after_probes": True,
        }
        skipped = {
            "ok": False,
            "skipped": True,
            "error": "canonical lock was not held; restoration probe is unsafe",
            "error_type": "LockNotHeld",
        }
        postflight["wired_limit_mb"] = dict(skipped)
        postflight["quality_models"] = dict(skipped)
        postflight["quality_ready_chat"] = dict(skipped)
        return postflight

    identity = {
        "device": opened.st_dev,
        "inode": opened.st_ino,
    }
    lock_check = {
        "ok": False,
        "requested_path": str(LOCK_PATH),
        "resolved_path": str(resolved),
        "identity": identity,
        "mode": "exclusive_nonblocking",
        "acquired_nonblocking": True,
        "held_through_probes": False,
        "released_after_probes": False,
    }
    postflight["lock_free"] = lock_check
    try:
        postflight["wired_limit_mb"] = _safe_check(_check_wired_limit)
        postflight["quality_models"] = _safe_check(_check_quality_models)
        postflight["quality_ready_chat"] = _safe_check(_check_quality_ready_chat)
        after = os.fstat(lock_file.fileno())
        resolved_after = LOCK_PATH.resolve(strict=True)
        current = resolved_after.stat()
        lock_check["held_through_probes"] = (
            (after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino)
            and resolved_after == resolved
            and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
        )
    except Exception as error:
        postflight["collector"] = _failed_check(
            error, context="unexpected postflight collector failure"
        )
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception as error:
            lock_check["release_error"] = str(error)
            lock_check["release_error_type"] = type(error).__name__
        finally:
            lock_file.close()
            lock_check["released_after_probes"] = True
        lock_check["ok"] = (
            lock_check["acquired_nonblocking"]
            and lock_check["held_through_probes"]
            and lock_check["released_after_probes"]
            and "release_error" not in lock_check
        )
    return postflight


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as receipt_file:
            receipt_file.write(encoded)
            receipt_file.flush()
            os.fsync(receipt_file.fileno())
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _command(tag: str) -> list[str]:
    return [
        str(VENV_PYTHON),
        str(RUN_GUARDED),
        "--plist",
        str(QUALITY_PLIST),
        "--timeout-seconds",
        "300",
        "--lock-timeout-seconds",
        "3600",
        "--child-timeout-seconds",
        "3600",
        "--",
        "/bin/zsh",
        str(ARMS),
        tag,
    ]


def run(tag: str, *, bench_dir: Path = DEFAULT_BENCH_DIR) -> int:
    """Run the sole service owner, then persist postflight on every outcome."""

    try:
        child_exit_code = subprocess.run(
            _command(tag),
            check=False,
            env={**os.environ, WRAPPER_ENV: "1"},
        ).returncode
    except OSError as error:
        child_exit_code = 127
        child_error: str | None = str(error)
    else:
        child_error = None
    try:
        postflight = collect_postflight()
    except Exception as error:
        postflight = {
            "collector": _failed_check(
                error, context="unexpected postflight collector failure"
            )
        }
    postflight_ok = all(result.get("ok") is True for result in postflight.values())
    exit_code = child_exit_code if child_exit_code != 0 else (0 if postflight_ok else 1)
    receipt = {
        "schema_version": 1,
        "kind": "deepseek_v4_moe_tail_guarded_postflight",
        "tag": tag,
        "run_guarded_command": _command(tag),
        "child_exit_code": child_exit_code,
        "child_error": child_error,
        "postflight": postflight,
        "postflight_ok": postflight_ok,
        "exit_code": exit_code,
        "completed_utc": datetime.now(UTC).isoformat(),
    }
    _write_receipt(bench_dir / f"{tag}-postflight.json", receipt)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tag",
        nargs="?",
        default=f"moe-tail-k3-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
    )
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    arguments = parser.parse_args()
    return run(arguments.tag, bench_dir=arguments.bench_dir)


if __name__ == "__main__":
    raise SystemExit(main())
