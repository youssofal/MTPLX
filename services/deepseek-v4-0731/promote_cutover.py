#!/usr/bin/env python3
"""Guarded, deliberately explicit 0731 candidate promotion workflow.

This is an operator workflow, not an auto-promotion hook.  It has no default
action, takes an exclusive nonblocking GPU lock, and refuses receipts that can
contain request content or process secrets.  The lock spans both cutover and
rollback, so an unrelated GPU user cannot be interrupted or raced.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import plistlib
import re
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


LOCK_PATH = Path("/tmp/mtplx-gpu-exclusive.lock")
CANDIDATE_LABEL = "com.tea.deepseek-v4-0731.candidate"
PRIOR_LIVE_LABEL = "com.tea.qwen"
PRODUCTION_LABEL = "com.tea.deepseek-v4-0731.production"
CANDIDATE_PORT = 8081
LIVE_PORT = 8080
SENSITIVE_KEY = re.compile(r"(?:prompt|message|tool|secret|token|authorization|argv|env|stdout|stderr)", re.I)
SENSITIVE_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bBearer\s+\S+|\bsk-[A-Za-z0-9_-]{12,}|"
    r"[\"']?(?:prompt|messages|tools|secret|authorization)[\"']?\s*[:=])",
    re.I,
)
PATH_LIKE_VALUE = re.compile(
    r"(?:^~[\\/]|(?:^|[\\/])\.\.(?:[\\/]|$)|\b[A-Za-z]:[\\/]|"
    r"[A-Za-z][A-Za-z0-9+.-]*://|(?:^|\s)/)"
)
ALLOWED_SIGNERS = Path("/Users/davidtai/.config/mtplx/deepseek-v4-0731-allowed-signers")
# Digest of the reviewed dedicated public signer list.
ALLOWED_SIGNERS_SHA256 = "003f258613fe308134ef184e52988a082a3376655d6b44f526017d7d71c7f843"
SIGNING_IDENTITY = "mtplx-deepseek-v4-0731-candidate"
SIGNING_NAMESPACE = "mtplx-deepseek-v4-0731"
ALLOWED_CANDIDATE_MODEL_IDS = frozenset({"deepseek-v4-0731-candidate"})
ALLOWED_PRIOR_MODEL_IDS = ("mtplx-qwen36-27b-optimized-quality",)
ALLOWED_LAUNCHD_LABELS = frozenset({PRIOR_LIVE_LABEL, PRODUCTION_LABEL})
SNAPSHOT_DIR_NAME = ".mtplx-dsv4-0731-snapshots"
CANDIDATE_WORKTREE = Path("/Users/davidtai/projects/OpenSourceWTF/.worktrees/dsv4-0731-service")
REVIEWED_REF = "refs/tags/mtplx-dsv4-0731-reviewed"
CANDIDATE_PLIST_SHA256 = "93eac0d4eaac491c7f2f1d3a293ba38a3144ade59ee3afdf52b35cc9ec9bb101"
ENCODING_ASSET_SET_SHA256 = "6758dfda8a39afdd00d907606c42c1a268289c463351b9628ac07f4f916d7d0a"
MODEL_CONFIG_SHA256 = "6d0297a4329d55dccf3cd48fd168efea8044996245195d518a9e8aaa14906d3e"
MODEL_INDEX_SHA256 = "9edcd0db7e6b8f0b8e02978d73c30083b2aa64c2e3a8fd77d3b776a4efb4bc91"


class PromotionError(RuntimeError):
    pass


@dataclass
class PlistSnapshot:
    source_path: Path
    path: Path
    raw: bytes
    sha256: str
    label: str
    program_arguments: tuple[str, ...]
    source_device: int
    source_inode: int

    def assert_source_unchanged(self) -> None:
        raw, metadata = _read_regular_file(self.source_path, "source plist")
        if (
            metadata.st_dev != self.source_device
            or metadata.st_ino != self.source_inode
            or hashlib.sha256(raw).hexdigest() != self.sha256
        ):
            raise PromotionError("source plist changed since snapshot")

    def assert_snapshot_intact(self) -> None:
        _assert_snapshot_directory(self.path.parent, source_device=self.source_device)
        raw, metadata = _read_regular_file(self.path, "durable plist snapshot")
        if (
            raw != self.raw
            or hashlib.sha256(raw).hexdigest() != self.sha256
            or metadata.st_dev != self.source_device
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise PromotionError("rollback plist snapshot changed")

    def cleanup_if_unloaded(self) -> None:
        """Remove this snapshot only after launchd no longer references it."""
        job = _launchctl_job_if_loaded(self.label)
        if job is not None and job["path"] == self.path:
            raise PromotionError("durable plist snapshot is still loaded")
        self.assert_snapshot_intact()
        self.path.unlink()
        _fsync_directory(self.path.parent)


def _read_regular_file(path: Path, context: str) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise PromotionError(f"{context} is missing or unsafe") from error
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise PromotionError(f"{context} is not a regular file")
        chunks: list[bytes] = []
        while chunk := os.read(fd, 1024 * 1024):
            chunks.append(chunk)
        return b"".join(chunks), metadata
    finally:
        os.close(fd)


def _parse_plist_identity(raw: bytes) -> tuple[str, tuple[str, ...]]:
    try:
        payload = plistlib.loads(raw)
    except plistlib.InvalidFileException as error:
        raise PromotionError("attested plist is not valid") from error
    if not isinstance(payload, dict):
        raise PromotionError("attested plist root is not a dictionary")
    label = payload.get("Label")
    arguments = payload.get("ProgramArguments")
    if label not in ALLOWED_LAUNCHD_LABELS:
        raise PromotionError("attested plist Label is not allowlisted")
    if (
        not isinstance(arguments, list)
        or not arguments
        or not all(isinstance(argument, str) and argument for argument in arguments)
    ):
        raise PromotionError("attested plist ProgramArguments are invalid")
    return str(label), tuple(arguments)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise PromotionError("snapshot directory is missing or unsafe") from error
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _assert_snapshot_directory(path: Path, *, source_device: int) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise PromotionError("snapshot directory is missing or unsafe") from error
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or path.is_symlink()
        or metadata.st_dev != source_device
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise PromotionError("snapshot directory metadata is unsafe")


def _materialize_durable_snapshot(
    *, source: Path, raw: bytes, metadata: os.stat_result, label: str, digest: str
) -> Path:
    if source.parent.name == SNAPSHOT_DIR_NAME:
        snapshot_dir = source.parent
    else:
        snapshot_dir = source.parent / SNAPSHOT_DIR_NAME
        directory_created = False
        try:
            snapshot_dir.mkdir(mode=0o700)
            directory_created = True
        except FileExistsError:
            pass
        if directory_created:
            _fsync_directory(source.parent)
    _assert_snapshot_directory(snapshot_dir, source_device=metadata.st_dev)
    snapshot_path = snapshot_dir / f"{label}-{digest}.plist"
    if snapshot_path.exists() or snapshot_path.is_symlink():
        existing, existing_metadata = _read_regular_file(
            snapshot_path, "durable plist snapshot"
        )
        if (
            existing != raw
            or existing_metadata.st_dev != metadata.st_dev
            or existing_metadata.st_uid != os.getuid()
            or stat.S_IMODE(existing_metadata.st_mode) != 0o400
        ):
            raise PromotionError("existing durable plist snapshot is unsafe")
        return snapshot_path

    fd, temporary_name = tempfile.mkstemp(
        prefix=".mtplx-dsv4-0731-write-", suffix=".tmp", dir=snapshot_dir
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o400)
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise PromotionError("durable plist snapshot write did not progress")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary_path, snapshot_path)
        _fsync_directory(snapshot_dir)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return snapshot_path


@contextmanager
def plist_snapshot(source: Path) -> Iterator[PlistSnapshot]:
    """Materialize exact plist bytes durably beside the reviewed source."""
    source = Path(os.path.abspath(source))
    raw, metadata = _read_regular_file(source, "source plist")
    label, program_arguments = _parse_plist_identity(raw)
    digest = hashlib.sha256(raw).hexdigest()
    snapshot_path = _materialize_durable_snapshot(
        source=source,
        raw=raw,
        metadata=metadata,
        label=label,
        digest=digest,
    )
    snapshot = PlistSnapshot(
        source_path=source,
        path=snapshot_path,
        raw=raw,
        sha256=digest,
        label=label,
        program_arguments=program_arguments,
        source_device=metadata.st_dev,
        source_inode=metadata.st_ino,
    )
    snapshot.assert_snapshot_intact()
    yield snapshot


def _command(*argv: str) -> str:
    result = subprocess.run(argv, check=False, capture_output=True, text=True)
    if result.returncode:
        raise PromotionError(f"required identity probe failed: {argv[0]}")
    return result.stdout


def _http_json(url: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError) as error:
        raise PromotionError("required HTTP readiness probe failed") from error
    if not isinstance(payload, dict):
        raise PromotionError("required HTTP readiness response is malformed")
    return payload


def _smoke_stop(model_id: str) -> None:
    """Run a real, unrecorded readiness completion and require normal stop."""
    body = json.dumps(
        {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly READY."}],
            "temperature": 0,
            "max_tokens": 8,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{LIVE_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choice = payload["choices"][0]
        content = choice["message"]["content"]
        if (
            choice.get("finish_reason") != "stop"
            or not isinstance(content, str)
            or content.strip() != "READY"
        ):
            raise ValueError("required READY/stop evidence absent")
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise PromotionError("service smoke did not return READY with finish_reason=stop") from error


def _listener_pid(port: int) -> int:
    output = _command("/usr/sbin/lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpn")
    current_pid: int | None = None
    loopback_pids: set[int] = set()
    wildcard_pids: set[int] = set()
    loopback_endpoint = f"127.0.0.1:{port}"
    wildcard_endpoints = {
        f"*:{port}",
        f"0.0.0.0:{port}",
        f"[::]:{port}",
        f":::{port}",
    }
    for line in output.splitlines():
        if line.startswith("p"):
            current_pid = int(line[1:]) if line[1:].isdigit() else None
        elif line.startswith("n") and current_pid is not None:
            endpoint = line[1:]
            if endpoint == loopback_endpoint:
                loopback_pids.add(current_pid)
            elif endpoint in wildcard_endpoints:
                wildcard_pids.add(current_pid)
    if wildcard_pids or len(loopback_pids) != 1:
        raise PromotionError("exact loopback listener identity is absent or ambiguous")
    return loopback_pids.pop()


def _parse_launchctl_job(output: str) -> dict[str, Any]:
    def scalar(name: str) -> str:
        match = re.search(rf"^\s*{re.escape(name)} = (.+?)\s*$", output, re.MULTILINE)
        if not match:
            raise PromotionError(f"launchd job has no {name}")
        return match.group(1)

    arguments_match = re.search(
        r"^\s*arguments = \{\s*$(.*?)^\s*\}\s*$",
        output,
        re.MULTILINE | re.DOTALL,
    )
    if not arguments_match:
        raise PromotionError("launchd job has no arguments")
    arguments = tuple(
        line.strip()
        for line in arguments_match.group(1).splitlines()
        if line.strip()
    )
    pid_text = scalar("pid")
    if not pid_text.isdigit():
        raise PromotionError("launchd service has no single running PID")
    return {
        "pid": int(pid_text),
        "path": Path(scalar("path")),
        "program": scalar("program"),
        "arguments": arguments,
    }


def _launchctl_job(label: str) -> dict[str, Any]:
    if label not in ALLOWED_LAUNCHD_LABELS:
        raise PromotionError("launchd label is not allowlisted")
    domain = f"gui/{os.getuid()}/{label}"
    return _parse_launchctl_job(_command("/bin/launchctl", "print", domain))


def _launchctl_job_if_loaded(label: str) -> dict[str, Any] | None:
    """Return a loaded job, distinguishing absence from an unsafe probe failure."""
    if label not in ALLOWED_LAUNCHD_LABELS:
        raise PromotionError("launchd label is not allowlisted")
    domain = f"gui/{os.getuid()}/{label}"
    result = subprocess.run(
        ["/bin/launchctl", "print", domain],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        diagnostic = f"{result.stdout}\n{result.stderr}"
        if "Could not find service" in diagnostic:
            return None
        raise PromotionError("could not safely determine whether launchd still references snapshot")
    return _parse_launchctl_job(result.stdout)


def _attest_process_snapshot(
    *,
    label: str,
    snapshot: PlistSnapshot,
    loaded_path: Path,
) -> dict[str, Any]:
    """Bind one launchd label, plist, and 8080 listener before any HTTP probe."""
    if label != snapshot.label:
        raise PromotionError("supplied label differs from the plist Label")
    snapshot.assert_snapshot_intact()
    job = _launchctl_job(label)
    if job["path"] != loaded_path:
        raise PromotionError("launchd job path differs from the supplied plist")
    if (
        job["program"] != snapshot.program_arguments[0]
        or job["arguments"] != snapshot.program_arguments
    ):
        raise PromotionError("launchd ProgramArguments differ from the supplied plist")
    launch_pid = int(job["pid"])
    listener_pid = _listener_pid(LIVE_PORT)
    if launch_pid != listener_pid:
        raise PromotionError("launchd PID and 8080 listener PID differ")
    return {
        "label": label,
        "pid": launch_pid,
        "listener_port": LIVE_PORT,
        "plist_sha256": snapshot.sha256,
    }


def attest_process_identity(*, label: str, plist: Path) -> dict[str, Any]:
    with plist_snapshot(plist) as snapshot:
        identity = _attest_process_snapshot(
            label=label,
            snapshot=snapshot,
            loaded_path=snapshot.source_path,
        )
        snapshot.assert_source_unchanged()
        return identity


def attest_live(
    *,
    label: str,
    plist: Path | PlistSnapshot,
    loaded_path: Path | None = None,
) -> dict[str, Any]:
    """Capture exact live identity without sending a generation prompt."""
    if isinstance(plist, PlistSnapshot):
        process = _attest_process_snapshot(
            label=label,
            snapshot=plist,
            loaded_path=loaded_path or plist.source_path,
        )
    else:
        process = attest_process_identity(label=label, plist=plist)
    models = _http_json(f"http://127.0.0.1:{LIVE_PORT}/v1/models")
    model_ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
    if not model_ids or not all(isinstance(model_id, str) for model_id in model_ids):
        raise PromotionError("live /v1/models is not a valid service identity")
    return {
        "schema": "mtplx.live-identity.v1",
        **process,
        "model_ids": model_ids,
    }


def _wait_for_process_identity(
    *,
    label: str,
    plist: Path | PlistSnapshot,
    loaded_path: Path | None = None,
) -> dict[str, Any]:
    """Wait for launchd and the listener to converge without probing HTTP."""
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            if isinstance(plist, PlistSnapshot):
                return _attest_process_snapshot(
                    label=label,
                    snapshot=plist,
                    loaded_path=loaded_path or plist.source_path,
                )
            return attest_process_identity(label=label, plist=plist)
        except PromotionError:
            time.sleep(0.5)
    raise PromotionError("promoted launchd process did not acquire the 8080 listener")


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise PromotionError("receipt is missing or unsafe")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as error:
        raise PromotionError("receipt is not valid JSON") from error
    if not isinstance(payload, dict):
        raise PromotionError("receipt root must be an object")
    return payload, raw


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_bytes(path)[0]


def _verify_candidate_signature(receipt_bytes: bytes, signature: Path) -> None:
    _assert_allowed_signers_trusted()
    if not signature.is_file() or signature.is_symlink():
        raise PromotionError("detached candidate signature is missing or unsafe")
    result = subprocess.run(
        [
            "/usr/bin/ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(ALLOWED_SIGNERS),
            "-I",
            SIGNING_IDENTITY,
            "-n",
            SIGNING_NAMESPACE,
            "-s",
            str(signature),
        ],
        input=receipt_bytes,
        check=False,
        capture_output=True,
    )
    if result.returncode:
        raise PromotionError("candidate receipt signature verification failed")


def _assert_allowed_signers_trusted() -> None:
    """Trust exactly the reviewed signer list, owned by this operator."""
    if not ALLOWED_SIGNERS.is_file() or ALLOWED_SIGNERS.is_symlink():
        raise PromotionError("pinned candidate allowed-signers file is missing or unsafe")
    try:
        metadata = ALLOWED_SIGNERS.stat()
    except OSError as error:
        raise PromotionError("pinned candidate allowed-signers metadata is unavailable") from error
    if metadata.st_uid != os.getuid():
        raise PromotionError("pinned candidate allowed-signers owner is unsafe")
    if stat.S_IMODE(metadata.st_mode) & (stat.S_IWGRP | stat.S_IWOTH):
        raise PromotionError("pinned candidate allowed-signers permissions are unsafe")
    if hashlib.sha256(ALLOWED_SIGNERS.read_bytes()).hexdigest() != ALLOWED_SIGNERS_SHA256:
        raise PromotionError("pinned candidate allowed-signers digest changed")


def _contains_sensitive(value: Any) -> bool:
    if isinstance(value, dict):
        return any(SENSITIVE_KEY.search(str(key)) or _contains_sensitive(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive(item) for item in value)
    if isinstance(value, str):
        decoded = urllib.parse.unquote(value)
        if PATH_LIKE_VALUE.search(decoded):
            return True
        if SENSITIVE_VALUE.search(value):
            return True
    return False


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise PromotionError(f"{context} fields do not match the strict receipt schema")
    return value


def assert_candidate_receipt(payload: dict[str, Any]) -> None:
    """Accept only a previously passing, scrubbed candidate preflight+smoke receipt."""
    if _contains_sensitive(payload):
        raise PromotionError("candidate receipt includes prohibited sensitive capture")
    _require_exact_keys(payload, {"schema", "candidate_preflight", "candidate_smoke"}, "candidate receipt")
    if payload["schema"] != "mtplx.dsv4-0731-candidate.v1":
        raise PromotionError("candidate receipt schema is not pinned")
    preflight = _require_exact_keys(
        payload["candidate_preflight"],
        {
            "ok", "label", "port", "plist_sha256", "encoding_source_revision",
            "encoding_asset_set_sha256", "reviewed_commit", "model_config_sha256",
            "model_index_sha256", "promotion_target",
        },
        "candidate preflight",
    )
    smoke = _require_exact_keys(
        payload["candidate_smoke"],
        {"ok", "models_ok", "ready", "finish_reason", "candidate_model_ids"},
        "candidate smoke",
    )
    if preflight.get("ok") is not True or smoke.get("ok") is not True:
        raise PromotionError("candidate preflight and smoke must already pass")
    if preflight.get("label") != CANDIDATE_LABEL or preflight.get("port") != CANDIDATE_PORT:
        raise PromotionError("candidate identity does not match the pinned isolated service")
    if smoke.get("models_ok") is not True or smoke.get("ready") is not True or smoke.get("finish_reason") != "stop":
        raise PromotionError("candidate smoke receipt lacks models/READY/stop evidence")
    if preflight.get("encoding_source_revision") != "7872f01b1d1fe23eabc4c98b48bffcef5a386062":
        raise PromotionError("candidate encoding source revision changed")
    for field in (
        "plist_sha256", "encoding_asset_set_sha256", "model_config_sha256", "model_index_sha256"
    ):
        if not isinstance(preflight.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", preflight[field]):
            raise PromotionError(f"candidate preflight has invalid {field}")
    if not isinstance(preflight.get("reviewed_commit"), str) or not re.fullmatch(r"[0-9a-f]{40}", preflight["reviewed_commit"]):
        raise PromotionError("candidate preflight has invalid reviewed_commit")
    target = _require_exact_keys(preflight["promotion_target"], {"label", "plist_sha256"}, "promotion target")
    if target.get("label") != PRODUCTION_LABEL:
        raise PromotionError("candidate preflight has a disallowed production label")
    digest = target.get("plist_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PromotionError("candidate preflight lacks a valid promotion plist digest")
    candidate_model_ids = smoke.get("candidate_model_ids")
    if (
        not isinstance(candidate_model_ids, list)
        or not candidate_model_ids
        or not all(
            isinstance(model_id, str) and model_id in ALLOWED_CANDIDATE_MODEL_IDS
            for model_id in candidate_model_ids
        )
    ):
        raise PromotionError("candidate smoke has a disallowed model ID")


def assert_live_identity(expected: dict[str, Any], current: dict[str, Any]) -> None:
    _require_exact_keys(
        expected,
        {"schema", "label", "pid", "listener_port", "plist_sha256", "model_ids"},
        "live attestation",
    )
    fields = ("schema", "label", "pid", "listener_port", "plist_sha256", "model_ids")
    if expected.get("schema") != "mtplx.live-identity.v1":
        raise PromotionError("live attestation schema is not allowlisted")
    if expected.get("label") != PRIOR_LIVE_LABEL:
        raise PromotionError("live attestation label is not allowlisted")
    if expected.get("model_ids") != list(ALLOWED_PRIOR_MODEL_IDS):
        raise PromotionError("live attestation model IDs are not allowlisted")
    if (
        not isinstance(expected.get("pid"), int)
        or isinstance(expected.get("pid"), bool)
        or expected["pid"] <= 0
    ):
        raise PromotionError("live attestation has invalid pid")
    if expected.get("listener_port") != LIVE_PORT:
        raise PromotionError("live attestation has invalid listener port")
    if not isinstance(expected.get("plist_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected["plist_sha256"]
    ):
        raise PromotionError("live attestation has invalid plist digest")
    if any(expected.get(field) != current.get(field) for field in fields):
        raise PromotionError("live service identity changed since its attestation")


@contextmanager
def exclusive_gpu_lock() -> Iterator[None]:
    """Take the shared lock once, nonblocking, and retain it through rollback."""
    fd = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PromotionError("GPU lock is already held; no service action was taken") from error
        yield
    finally:
        os.close(fd)


def _bootstrap(plist: Path) -> None:
    _command("/bin/launchctl", "bootstrap", f"gui/{os.getuid()}", str(plist))


def _bootout(label: str) -> None:
    _command("/bin/launchctl", "bootout", f"gui/{os.getuid()}/{label}")


def _verify_live_ready(expected_model_ids: list[str]) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            payload = _http_json(f"http://127.0.0.1:{LIVE_PORT}/v1/models")
            ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
            if ids == expected_model_ids:
                _smoke_stop(expected_model_ids[0])
                return
        except PromotionError:
            pass
        time.sleep(0.5)
    raise PromotionError("restored service did not recover its exact /v1/models identity")


def promote(args: argparse.Namespace) -> None:
    if args.promote is not True:
        raise PromotionError("refusing promotion without --promote")
    if args.production_label != PRODUCTION_LABEL:
        raise PromotionError("production label is not allowlisted")
    candidate, candidate_bytes = _read_json_bytes(args.candidate_receipt)
    _verify_candidate_signature(candidate_bytes, args.candidate_signature)
    expected_live = _read_json(args.live_attestation)
    if _contains_sensitive(expected_live):
        raise PromotionError("live attestation includes prohibited sensitive capture")
    assert_candidate_receipt(candidate)
    preflight = candidate["candidate_preflight"]
    reviewed_commit = _command(
        "/usr/bin/git",
        "-C",
        str(CANDIDATE_WORKTREE),
        "rev-parse",
        "--verify",
        f"{REVIEWED_REF}^{{commit}}",
    ).strip()
    pinned_candidate = {
        "reviewed_commit": reviewed_commit,
        "plist_sha256": CANDIDATE_PLIST_SHA256,
        "encoding_asset_set_sha256": ENCODING_ASSET_SET_SHA256,
        "model_config_sha256": MODEL_CONFIG_SHA256,
        "model_index_sha256": MODEL_INDEX_SHA256,
    }
    if any(preflight[field] != expected for field, expected in pinned_candidate.items()):
        raise PromotionError("signed candidate receipt does not match the reviewed installation")
    # The production target must be a separately reviewed 8080 plist.  The
    # candidate plist stays isolated on 8081 and is never edited in place.
    target = args.production_plist
    if target == args.live_plist or not target.is_absolute():
        raise PromotionError("an absolute separately reviewed production plist is required")
    if not target.is_file() or target.is_symlink():
        raise PromotionError("production plist is missing or unsafe")
    prior_plist = args.live_plist
    if not prior_plist.is_absolute():
        raise PromotionError("live attestation does not name an absolute prior plist")
    with plist_snapshot(target) as target_snapshot, plist_snapshot(prior_plist) as prior_snapshot:
        promotion_target = preflight["promotion_target"]
        if (
            promotion_target["label"] != args.production_label
            or promotion_target["plist_sha256"] != target_snapshot.sha256
        ):
            raise PromotionError("production plist identity does not match the passing candidate preflight")
        if (
            target_snapshot.label != args.production_label
            or args.production_label == str(expected_live.get("label"))
        ):
            raise PromotionError("production label is unsafe or does not match its plist")

        with exclusive_gpu_lock():
            current = attest_live(
                label=str(expected_live.get("label", "")),
                plist=prior_snapshot,
                loaded_path=prior_snapshot.source_path,
            )
            assert_live_identity(expected_live, current)
            prior_snapshot.assert_source_unchanged()
            target_snapshot.assert_source_unchanged()
            # No service is stopped until every receipt and identity check above
            # has passed under the lock. Any post-cutover exception restores
            # the exact descriptor-read prior snapshot under that same lock.
            try:
                _bootout(current["label"])
                target_snapshot.assert_source_unchanged()
                _bootstrap(target_snapshot.path)
                promoted = _wait_for_process_identity(
                    label=args.production_label,
                    plist=target_snapshot,
                    loaded_path=target_snapshot.path,
                )
                if promoted["plist_sha256"] != promotion_target["plist_sha256"]:
                    raise PromotionError("promoted service plist identity changed during cutover")
                _verify_live_ready(candidate["candidate_smoke"]["candidate_model_ids"])
            except BaseException:
                try:
                    _bootout(args.production_label)
                finally:
                    prior_snapshot.assert_snapshot_intact()
                    _bootstrap(prior_snapshot.path)
                    _wait_for_process_identity(
                        label=current["label"],
                        plist=prior_snapshot,
                        loaded_path=prior_snapshot.path,
                    )
                    _verify_live_ready(current["model_ids"])
                    target_snapshot.cleanup_if_unloaded()
                raise
            # Readiness is the cutover commit point. Snapshot reclamation is
            # post-commit housekeeping and must never re-enter rollback after
            # it has removed the only prior rollback path.
            try:
                prior_snapshot.cleanup_if_unloaded()
            except Exception as error:
                print(
                    f"promotion committed; prior snapshot cleanup was incomplete: {error}",
                    file=sys.stderr,
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true", help="explicitly authorize guarded service action")
    parser.add_argument("--candidate-receipt", type=Path, required=True)
    parser.add_argument("--candidate-signature", type=Path, required=True)
    parser.add_argument("--live-attestation", type=Path, required=True)
    parser.add_argument("--live-plist", type=Path, required=True)
    parser.add_argument("--production-plist", type=Path, required=True)
    parser.add_argument("--production-label", required=True)
    args = parser.parse_args(argv)
    try:
        promote(args)
    except PromotionError as error:
        print(f"promotion refused: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
