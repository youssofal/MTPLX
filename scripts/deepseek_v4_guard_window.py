#!/usr/bin/env python3
"""Bridge one guard attestation to a fixed DeepSeek-V4 benchmark bracket.

The guard pipe is deliberately consumed once, by ``issue``.  The resulting
canonical, read-only receipt can then be checked by each benchmark grandchild
without attempting to read the pipe again.  Every check binds the receipt to
the still-live guarded process ancestry and the still-held lock before MLX is
imported.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


WINDOW_PATH_ENV = "MTPLX_DSV4_GUARD_WINDOW_PATH"
WINDOW_SHA256_ENV = "MTPLX_DSV4_GUARD_WINDOW_SHA256"
DEFAULT_LOCK_PATH = Path("/tmp/mtplx-gpu-exclusive.lock")
LAGUNA_BENCH = Path(
    os.environ.get(
        "MTPLX_DSV4_GUARD_VERIFIER",
        "/Users/davidtai/projects/OpenSourceWTF/bench/laguna/laguna_fixed_m2_bench.py",
    )
)
_MAX_RECEIPT_BYTES = 16 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")


def _assert_mlx_not_imported() -> None:
    if any(name == "mlx" or name.startswith("mlx.") for name in sys.modules):
        raise RuntimeError("guard verification must run before any MLX import")


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )


def _load_repository_guard() -> ModuleType:
    """Load the repository's authoritative verifier without importing MLX."""

    _assert_mlx_not_imported()
    if not LAGUNA_BENCH.is_file():
        raise RuntimeError(f"repository guard verifier is missing: {LAGUNA_BENCH}")
    module_dir = str(LAGUNA_BENCH.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    name = "_mtplx_repository_guard_verifier"
    spec = importlib.util.spec_from_file_location(name, LAGUNA_BENCH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load repository guard verifier: {LAGUNA_BENCH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    _assert_mlx_not_imported()
    return module


def _checked_attestation(
    attestation: Mapping[str, Any], expected_lock: Path
) -> dict[str, Any]:
    integers = (
        attestation.get("guard_pid"),
        attestation.get("child_pid"),
        attestation.get("issued_monotonic_ns"),
        attestation.get("expires_monotonic_ns"),
        attestation.get("lock_device"),
        attestation.get("lock_inode"),
    )
    if (
        attestation.get("schema_version") != 1
        or any(
            isinstance(value, bool) or not isinstance(value, int) for value in integers
        )
        or not _valid_digest(attestation.get("nonce_sha256"))
    ):
        raise RuntimeError("repository guard attestation receipt is malformed")
    issued = int(attestation["issued_monotonic_ns"])
    expires = int(attestation["expires_monotonic_ns"])
    if issued > expires or expires - issued > 60_000_000_000:
        raise RuntimeError("repository guard attestation expiry is malformed")
    lock_path = attestation.get("lock_path")
    resolved_lock = expected_lock.resolve(strict=True)
    if (
        not isinstance(lock_path, str)
        or Path(lock_path).resolve(strict=True) != resolved_lock
    ):
        raise RuntimeError(
            f"guard attested {lock_path!r}, expected lock {str(expected_lock)!r}"
        )
    observed = resolved_lock.stat()
    if (observed.st_dev, observed.st_ino) != (
        attestation["lock_device"],
        attestation["lock_inode"],
    ):
        raise RuntimeError("guard attestation lock device/inode no longer matches")
    return {
        "requested_path": str(expected_lock),
        "resolved_path": str(resolved_lock),
        "device": observed.st_dev,
        "inode": observed.st_ino,
    }


def issue_guard_window(*, expected_lock: Path = DEFAULT_LOCK_PATH) -> tuple[Path, str]:
    """Consume the repository attestation and publish an immutable receipt."""

    repository = _load_repository_guard()
    attestation = repository.verify_guard_attestation()
    verified = time.monotonic_ns()
    lock_identity = _checked_attestation(attestation, expected_lock)
    if not (
        attestation["issued_monotonic_ns"]
        <= verified
        <= attestation["expires_monotonic_ns"]
    ):
        raise RuntimeError("guard attestation expired before receipt publication")
    window_id = _sha256(_canonical_json(attestation))
    document = {
        "schema_version": 1,
        "kind": "mtplx_verified_guard_window",
        "verified": True,
        "verified_monotonic_ns": verified,
        "window_id": window_id,
        "attestation": attestation,
        "lock_identity": lock_identity,
    }
    encoded = _canonical_json(document)
    directory = Path(tempfile.mkdtemp(prefix="mtplx-dsv4-guard-window-"))
    os.chmod(directory, 0o700)
    path = directory / "window.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o400)
    try:
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path, _sha256(encoded)


def _read_private_receipt(path: Path) -> bytes:
    parent = path.parent.lstat()
    observed = path.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or parent.st_uid != os.getuid()
        or stat.S_IMODE(parent.st_mode) != 0o700
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_uid != os.getuid()
        or stat.S_IMODE(observed.st_mode) != 0o400
    ):
        raise RuntimeError("verified guard window receipt permissions are unsafe")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino):
            raise RuntimeError("verified guard window receipt changed while opening")
        payload = bytearray()
        while len(payload) <= _MAX_RECEIPT_BYTES:
            chunk = os.read(descriptor, _MAX_RECEIPT_BYTES + 1 - len(payload))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > _MAX_RECEIPT_BYTES:
            raise RuntimeError("verified guard window receipt is oversized")
        return bytes(payload)
    finally:
        os.close(descriptor)


def load_verified_guard_window(
    *, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Verify the inherited static receipt against this live descendant."""

    _assert_mlx_not_imported()
    environ = os.environ if environment is None else environment
    path_text = environ.get(WINDOW_PATH_ENV)
    expected_digest = environ.get(WINDOW_SHA256_ENV)
    if (
        not isinstance(path_text, str)
        or not Path(path_text).is_absolute()
        or not _valid_digest(expected_digest)
    ):
        raise RuntimeError("verified guard window environment is absent or malformed")
    path = Path(path_text)
    encoded = _read_private_receipt(path)
    if _sha256(encoded) != expected_digest:
        raise RuntimeError("verified guard window receipt digest mismatch")
    try:
        document = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"verified guard window receipt is malformed: {error}"
        ) from error
    if not isinstance(document, dict) or _canonical_json(document) != encoded:
        raise RuntimeError("verified guard window receipt is not canonical")
    attestation = document.get("attestation")
    if not isinstance(attestation, dict):
        raise RuntimeError("verified guard window attestation is absent")
    lock_path = attestation.get("lock_path")
    if not isinstance(lock_path, str):
        raise RuntimeError("verified guard window lock path is absent")
    lock_identity = document.get("lock_identity")
    if not isinstance(lock_identity, dict):
        raise RuntimeError("verified guard window lock identity is absent")
    requested_path = lock_identity.get("requested_path")
    if not isinstance(requested_path, str):
        raise RuntimeError("verified guard window requested lock path is absent")
    repository = _load_repository_guard()
    observed_lock_identity = _checked_attestation(attestation, Path(requested_path))
    if observed_lock_identity != lock_identity:
        raise RuntimeError("verified guard window lock identity changed")
    verified = document.get("verified_monotonic_ns")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "mtplx_verified_guard_window"
        or document.get("verified") is not True
        or isinstance(verified, bool)
        or not isinstance(verified, int)
        or not attestation.get("issued_monotonic_ns")
        <= verified
        <= attestation.get("expires_monotonic_ns")
        or document.get("window_id") != _sha256(_canonical_json(attestation))
    ):
        raise RuntimeError("verified guard window identity or expiry is invalid")
    # The repository verifier returns a tuple, but this object is both consumed
    # live by the validator and persisted as JSON.  Canonicalize before either
    # path sees it so live and serialized guard semantics are identical.
    ancestry = list(repository._current_process_ancestry())
    child_pid = attestation["child_pid"]
    guard_pid = attestation["guard_pid"]
    if (
        child_pid not in ancestry
        or guard_pid not in ancestry
        or ancestry.index(guard_pid) <= ancestry.index(child_pid)
    ):
        raise RuntimeError("verified guard window process ancestry check failed")
    lock_held = repository._lock_is_held_by_other_process(
        Path(attestation["lock_path"]),
        attestation["lock_device"],
        attestation["lock_inode"],
    )
    if not lock_held:
        raise RuntimeError("verified guard window lock is not held")
    _assert_mlx_not_imported()
    return {
        **document,
        "receipt_path": str(path),
        "receipt_sha256": expected_digest,
        "consumer_verification": {
            "consumer_pid": os.getpid(),
            "ancestry": ancestry,
            "child_pid_index": ancestry.index(child_pid),
            "guard_pid_index": ancestry.index(guard_pid),
            "lock_held": lock_held,
            "observed_lock_device": observed_lock_identity["device"],
            "observed_lock_inode": observed_lock_identity["inode"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--expected-lock", type=Path, default=DEFAULT_LOCK_PATH)
    subparsers.add_parser("verify")
    args = parser.parse_args()
    try:
        if args.action == "issue":
            path, digest = issue_guard_window(expected_lock=args.expected_lock)
            print(f"{path}\t{digest}")
        else:
            print(json.dumps(load_verified_guard_window(), sort_keys=True))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[deepseek-v4-guard-window] {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
