#!/usr/bin/env python3
"""Guard the canonical attention-island bracket and restore Qwen Quality."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
# Guard deployment locations are supplied by the operator.  These defaults are
# intentionally relative so an unconfigured checkout fails in the guarded
# runner rather than encoding a particular developer machine (mirrors the
# adaptive-width pair; PR #223 review edit).
VENV_PYTHON = Path(os.environ.get("MTPLX_DSV4_PYTHON", "python3"))
RUN_GUARDED = Path(os.environ.get("MTPLX_DSV4_GUARDED_RUNNER", "run_guarded.py"))
QUALITY_PLIST = Path(os.environ.get("MTPLX_DSV4_QUALITY_PLIST", "com.tea.qwen.plist"))
QUALITY_PLIST_SHA256 = os.environ.get(
    "MTPLX_DSV4_QUALITY_PLIST_SHA256",
    "a504ddfc6893a2ac7cef3d6072bdc49e1626b926638169de151530e311281e10",
)
QUALITY_PLIST_SIZE = int(os.environ.get("MTPLX_DSV4_QUALITY_PLIST_SIZE", "888"))
BENCH_DIR = Path(os.environ.get("MTPLX_DSV4_BENCH_DIR", "bench/deepseek-v4"))
LOCK_PATH = Path("/tmp/mtplx-gpu-exclusive.lock")
ARMS = HERE / "deepseek_v4_attention_island_arms.sh"
WRAPPER_ENV = "MTPLX_DSV4_ATTENTION_ISLAND_POSTFLIGHT_WRAPPER"
QUALITY_MODEL = "mtplx-qwen36-27b-optimized-quality"
# 0 disables the exact wired-limit gate (it encodes one operator's sysctl
# tuning, not a portable invariant); set it to pin a deployment's value.
EXPECTED_WIRED_LIMIT_MB = int(
    os.environ.get("MTPLX_DSV4_EXPECTED_WIRED_LIMIT_MB", "114688")
)
PRIMARY_RECEIPT_ROLE = "attention_island_performance_bracket"
POSTFLIGHT_KIND = "deepseek_v4_attention_island_guarded_postflight"
CHILD_STATUS_KIND = "attention_island_child_status"
INVALID_TAG_PREFIX = "attention-island-invalid-tag-"
TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
REQUIRED_PROBES = (
    "lock_free",
    "wired_limit_mb",
    "quality_models",
    "quality_ready_chat",
    "quality_plist",
)


def _validate_tag(tag: str) -> str:
    if (
        not isinstance(tag, str)
        or tag in {"", ".", ".."}
        or TAG_PATTERN.fullmatch(tag) is None
    ):
        raise ValueError("invalid attention-island tag: expected a safe basename")
    return tag


def _tag_sha256(tag: object) -> str:
    encoded = (
        tag.encode("utf-8", errors="surrogatepass")
        if isinstance(tag, str)
        else repr(tag).encode("utf-8", errors="surrogatepass")
    )
    return hashlib.sha256(encoded).hexdigest()


def _receipt_path(bench_dir: Path, tag: object, valid_tag: str | None) -> Path:
    if valid_tag is not None:
        return bench_dir / f"{valid_tag}-postflight.json"
    return bench_dir / (
        f"{INVALID_TAG_PREFIX}{_tag_sha256(tag)}-pid-{os.getpid()}-postflight.json"
    )


def _validate_commit_sha(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be an exact lowercase 40-hex commit SHA")
    return value


def _source_commit() -> str:
    observed = subprocess.check_output(
        ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"], text=True
    ).strip()
    return _validate_commit_sha(observed, label="observed source commit")


def _source_clean() -> bool:
    status = subprocess.check_output(
        ["git", "-C", str(WORKTREE), "status", "--porcelain"], text=True
    )
    return not status.strip()


def _command(tag: str, expected_commit: str, observed_commit: str) -> list[str]:
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
        "7200",
        "--",
        "/bin/zsh",
        str(ARMS),
        tag,
        expected_commit,
        observed_commit,
    ]


def _failed_check(error: BaseException, *, context: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": f"{context}: {error}",
        "error_type": type(error).__name__,
    }


def _safe_check(check: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        result = check()
    except Exception as error:
        return _failed_check(error, context="probe raised")
    if not isinstance(result, dict):
        return {
            "ok": False,
            "error": f"probe returned {type(result).__name__}, expected an object",
            "error_type": "MalformedProbeResult",
        }
    return result


def _check_wired_limit() -> dict[str, Any]:
    completed = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "iogpu.wired_limit_mb"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        return {
            "ok": False,
            "error": completed.stderr.strip() or "sysctl failed",
            "exit_code": int(completed.returncode),
        }
    value = int(completed.stdout.strip())
    ok = EXPECTED_WIRED_LIMIT_MB <= 0 or value == EXPECTED_WIRED_LIMIT_MB
    return {"ok": ok, "value": value}


def _request_json(path: str, *, payload: dict | None, timeout: float):
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
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("READY chat content is not a string")
        content = content.strip()
        finish_reason = choice["finish_reason"]
        return {
            "ok": content == "READY" and finish_reason == "stop",
            "content": content,
            "finish_reason": finish_reason,
        }
    except Exception as error:
        return _failed_check(error, context="malformed READY chat response")


def _attest_quality_plist() -> dict[str, Any]:
    try:
        path_status = QUALITY_PLIST.lstat()
        if not stat.S_ISREG(path_status.st_mode):
            raise ValueError("Quality plist is not a regular file")
        encoded = QUALITY_PLIST.read_bytes()
        completed = subprocess.run(
            ["/usr/bin/plutil", "-lint", str(QUALITY_PLIST)],
            check=False,
            capture_output=True,
            text=True,
        )
        sha256 = hashlib.sha256(encoded).hexdigest()
        size = len(encoded)
        plutil_valid = completed.returncode == 0
        return {
            "ok": (
                sha256 == QUALITY_PLIST_SHA256
                and size == QUALITY_PLIST_SIZE
                and plutil_valid
            ),
            "path": str(QUALITY_PLIST),
            "sha256": sha256,
            "size": size,
            "plutil_valid": plutil_valid,
            "plutil_exit_code": int(completed.returncode),
            "plutil_stdout": str(completed.stdout or "").strip(),
            "plutil_stderr": str(completed.stderr or "").strip(),
        }
    except Exception as error:
        return {
            **_failed_check(error, context="Quality plist attestation failed"),
            "path": str(QUALITY_PLIST),
            "sha256": None,
            "size": None,
            "plutil_valid": False,
        }


def _collect_guarded_probes() -> dict[str, dict[str, Any]]:
    """Hold the canonical lock across literal knob, service, and plist probes."""

    probes: dict[str, dict[str, Any]] = {}
    lock_file = None
    try:
        lock_file = LOCK_PATH.open("rb")
        opened = os.fstat(lock_file.fileno())
        resolved = LOCK_PATH.resolve(strict=True)
        path_status = resolved.stat()
        if (opened.st_dev, opened.st_ino) != (
            path_status.st_dev,
            path_status.st_ino,
        ):
            raise RuntimeError("canonical lock identity changed while opening")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception as error:
        if lock_file is not None:
            lock_file.close()
        probes["lock_free"] = {
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
        for name in REQUIRED_PROBES[1:]:
            probes[name] = dict(skipped)
        return probes

    lock_check = {
        "ok": False,
        "requested_path": str(LOCK_PATH),
        "resolved_path": str(resolved),
        "identity": {"device": opened.st_dev, "inode": opened.st_ino},
        "mode": "exclusive_nonblocking",
        "acquired_nonblocking": True,
        "held_through_probes": False,
        "released_after_probes": False,
    }
    probes["lock_free"] = lock_check
    try:
        probes["wired_limit_mb"] = _safe_check(_check_wired_limit)
        probes["quality_models"] = _safe_check(_check_quality_models)
        probes["quality_ready_chat"] = _safe_check(_check_quality_ready_chat)
        probes["quality_plist"] = _safe_check(_attest_quality_plist)
        after = os.fstat(lock_file.fileno())
        resolved_after = LOCK_PATH.resolve(strict=True)
        current = resolved_after.stat()
        lock_check["held_through_probes"] = (
            (after.st_dev, after.st_ino) == (opened.st_dev, opened.st_ino)
            and resolved_after == resolved
            and (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino)
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
    return probes


def _validate_probes(payload: object, *, phase: str) -> tuple[dict, list[str]]:
    if not isinstance(payload, dict):
        return {}, [f"{phase} result is not an object"]
    normalized: dict[str, dict] = {}
    errors: list[str] = []
    unexpected = sorted(set(payload) - set(REQUIRED_PROBES))
    if unexpected:
        errors.append(f"{phase} contains unexpected probes: {unexpected}")
    for name in REQUIRED_PROBES:
        row = payload.get(name)
        if not isinstance(row, dict) or type(row.get("ok")) is not bool:
            errors.append(f"{phase} probe {name} is missing or malformed")
            continue
        normalized[name] = row
        if row["ok"] is not True:
            errors.append(f"{phase} probe {name} failed")

    lock = normalized.get("lock_free", {})
    if (
        lock.get("requested_path") != str(LOCK_PATH)
        or lock.get("acquired_nonblocking") is not True
        or lock.get("held_through_probes") is not True
        or lock.get("released_after_probes") is not True
    ):
        errors.append(f"{phase} lock receipt is not canonical")
    if (
        EXPECTED_WIRED_LIMIT_MB > 0
        and normalized.get("wired_limit_mb", {}).get("value") != EXPECTED_WIRED_LIMIT_MB
    ):
        errors.append(f"{phase} wired limit changed")
    if normalized.get("quality_models", {}).get("models") != [QUALITY_MODEL]:
        errors.append(f"{phase} Quality model identity changed")
    chat = normalized.get("quality_ready_chat", {})
    if chat.get("content") != "READY" or chat.get("finish_reason") != "stop":
        errors.append(f"{phase} READY chat is not a real natural stop")
    plist = normalized.get("quality_plist", {})
    if (
        plist.get("path") != str(QUALITY_PLIST)
        or plist.get("sha256") != QUALITY_PLIST_SHA256
        or plist.get("size") != QUALITY_PLIST_SIZE
        or plist.get("plutil_valid") is not True
    ):
        errors.append(f"{phase} Quality plist identity changed")
    return normalized, errors


def _collect(collector: Callable, *, phase: str) -> tuple[dict, list[str]]:
    try:
        return _validate_probes(collector(), phase=phase)
    except Exception as error:
        return {}, [f"{phase} collector failed: {type(error).__name__}: {error}"]


def _read_child_status(
    path: Path, *, tag: str, expected_commit: str, observed_commit: str
) -> tuple[dict | None, str | None, str | None]:
    try:
        status = path.lstat()
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("benchmark child status is not a regular file")
        if stat.S_IMODE(status.st_mode) != 0o600:
            raise ValueError("benchmark child status mode is not 0600")
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        if payload != {
            "schema_version": 1,
            "kind": CHILD_STATUS_KIND,
            "tag": tag,
            "expected_source_commit": expected_commit,
            "observed_source_commit": observed_commit,
            "benchmark_exit_code": payload.get("benchmark_exit_code"),
        }:
            raise ValueError("benchmark child status identity is invalid")
        exit_code = payload["benchmark_exit_code"]
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise TypeError("benchmark child exit code is not an integer")
        if not 0 <= exit_code <= 255:
            raise ValueError("benchmark child exit code is outside [0, 255]")
        return payload, hashlib.sha256(encoded).hexdigest(), None
    except Exception as error:
        return None, None, f"{type(error).__name__}: {error}"


def _read_primary(path: Path) -> tuple[dict | None, str | None, str | None]:
    """Preserve a valid nonzero primary receipt as failure diagnostics."""

    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise TypeError("primary receipt is not an object")
        if payload.get("receipt_role") != PRIMARY_RECEIPT_ROLE:
            raise ValueError("primary receipt role is invalid")
        status = payload.get("status")
        if isinstance(status, bool) or not isinstance(status, int):
            raise TypeError("primary receipt status is not an integer")
        error = None if status == 0 else f"ValueError: primary receipt status is {status}"
        return payload, hashlib.sha256(encoded).hexdigest(), error
    except Exception as error:
        return None, None, f"{type(error).__name__}: {error}"


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
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


def run(
    tag: str,
    *,
    expected_commit: str,
    source_commit_reader: Callable[[], str] = _source_commit,
    source_clean_reader: Callable[[], bool] = _source_clean,
    run_command=subprocess.run,
    preflight_collector: Callable = _collect_guarded_probes,
    postflight_collector: Callable = _collect_guarded_probes,
    bench_dir: Path = BENCH_DIR,
) -> int:
    """Require the caller-authorized clean commit before service shutdown."""

    bench_dir = Path(bench_dir)
    errors: list[str] = []
    try:
        valid_tag = _validate_tag(tag)
        tag_error = None
    except ValueError as error:
        valid_tag = None
        tag_error = f"{type(error).__name__}: {error}"
        errors.append(tag_error)

    try:
        expected = _validate_commit_sha(expected_commit, label="expected source commit")
    except ValueError as error:
        expected = None
        errors.append(f"{type(error).__name__}: {error}")
    try:
        observed = _validate_commit_sha(
            source_commit_reader(), label="observed source commit"
        )
    except Exception as error:
        observed = None
        errors.append(f"source commit read failed: {type(error).__name__}: {error}")
    try:
        source_clean = source_clean_reader()
        if type(source_clean) is not bool:
            raise TypeError("source clean reader did not return bool")
    except Exception as error:
        source_clean = False
        errors.append(f"source clean check failed: {type(error).__name__}: {error}")
    commit_match = expected is not None and observed is not None and expected == observed
    if expected is not None and observed is not None and not commit_match:
        errors.append("expected source commit does not match observed worktree HEAD")
    if not source_clean:
        errors.append("source worktree is dirty before guarded launch")

    preflight, preflight_errors = _collect(preflight_collector, phase="preflight")
    errors.extend(preflight_errors)
    child_started = (
        valid_tag is not None and commit_match and source_clean and not preflight_errors
    )
    guarded_runner_exit_code = None
    guarded_child_error = None
    primary = None
    primary_sha256 = None
    primary_error = None
    child_status = None
    child_status_sha256 = None
    child_status_error = None
    benchmark_child_exit_code = None
    if child_started:
        sidecar_path = bench_dir / f"{valid_tag}-child-status.json"
        sidecar_path.unlink(missing_ok=True)
        environment = {**os.environ, WRAPPER_ENV: "1"}
        try:
            completed = run_command(
                _command(valid_tag, expected, observed),
                check=False,
                env=environment,
            )
            guarded_runner_exit_code = int(completed.returncode)
        except Exception as error:
            guarded_runner_exit_code = 1
            guarded_child_error = f"{type(error).__name__}: {error}"
        child_status, child_status_sha256, child_status_error = _read_child_status(
            sidecar_path,
            tag=valid_tag,
            expected_commit=expected,
            observed_commit=observed,
        )
        if child_status is not None:
            benchmark_child_exit_code = child_status["benchmark_exit_code"]
        primary, primary_sha256, primary_error = _read_primary(
            bench_dir / f"{valid_tag}.json"
        )
        if primary is not None and primary.get("source_commit_attestation") != {
            "expected": expected,
            "observed": observed,
            "match": True,
            "clean": True,
        }:
            primary_error = "ValueError: primary source commit attestation is invalid"
    else:
        primary_error = (
            "primary receipt skipped because guarded child was not eligible to start"
        )

    postflight, postflight_errors = _collect(postflight_collector, phase="postflight")
    errors.extend(postflight_errors)
    if child_started:
        if child_status_error is not None:
            errors.append(f"benchmark child status failed: {child_status_error}")
        if benchmark_child_exit_code not in (None, 0):
            errors.append(f"benchmark child exited {benchmark_child_exit_code}")
        if (
            benchmark_child_exit_code is not None
            and guarded_runner_exit_code != benchmark_child_exit_code
        ):
            errors.append(
                f"guarded lifecycle returned {guarded_runner_exit_code} after "
                f"benchmark returned {benchmark_child_exit_code}"
            )
        elif guarded_runner_exit_code not in (None, 0):
            errors.append(f"guarded child exited {guarded_runner_exit_code}")
    if guarded_child_error is not None:
        errors.append(guarded_child_error)
    if primary_error is not None:
        errors.append(primary_error)

    pre_plist = preflight.get("quality_plist", {})
    post_plist = postflight.get("quality_plist", {})
    quality_plist_unchanged = (
        pre_plist.get("ok") is True
        and post_plist.get("ok") is True
        and pre_plist.get("path") == post_plist.get("path") == str(QUALITY_PLIST)
        and pre_plist.get("sha256")
        == post_plist.get("sha256")
        == QUALITY_PLIST_SHA256
        and pre_plist.get("size")
        == post_plist.get("size")
        == QUALITY_PLIST_SIZE
        and pre_plist.get("plutil_valid") is True
        and post_plist.get("plutil_valid") is True
    )
    if not quality_plist_unchanged:
        errors.append("Quality plist pre/post identity is not unchanged")

    receipt = {
        "schema_version": 1,
        "kind": POSTFLIGHT_KIND,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "tag": valid_tag,
        "tag_sha256": _tag_sha256(tag),
        "tag_validation_error": tag_error,
        "source_commit_attestation": {
            "expected": expected,
            "observed": observed,
            "match": commit_match,
            "clean": source_clean,
        },
        "run_guarded_command": (
            _command(valid_tag, expected, observed) if child_started else None
        ),
        "guarded_child_started": child_started,
        "guarded_runner_exit_code": guarded_runner_exit_code,
        "guarded_child_exit_code": guarded_runner_exit_code,
        "guarded_child_error": guarded_child_error,
        "benchmark_child_exit_code": benchmark_child_exit_code,
        "benchmark_child_status": child_status,
        "benchmark_child_status_sha256": child_status_sha256,
        "benchmark_child_status_error": child_status_error,
        "primary_receipt": primary,
        "primary_receipt_sha256": primary_sha256,
        "primary_receipt_error": primary_error,
        "preflight": preflight,
        "preflight_ok": not preflight_errors,
        "postflight": postflight,
        "postflight_ok": not postflight_errors,
        "quality_plist_unchanged": quality_plist_unchanged,
        "validation_errors": errors,
        "status": int(bool(errors)),
    }
    _write_receipt(_receipt_path(bench_dir, tag, valid_tag), receipt)
    return int(receipt["status"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tag",
        nargs="?",
        default=f"attention-island-{datetime.now(UTC):%Y%m%dT%H%M%SZ}",
    )
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="exact 40-hex commit SHA authorized for this GPU bracket",
    )
    parser.add_argument("--bench-dir", type=Path, default=BENCH_DIR)
    args = parser.parse_args()
    return run(
        args.tag,
        expected_commit=args.expected_commit,
        bench_dir=args.bench_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
