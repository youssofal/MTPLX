#!/usr/bin/env python3
"""Four-process ABBA gate for process-latched Qwen 3.8 candidates."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import math
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.qwen35b_mtp_batch_numerics_attribution import (  # noqa: E402
    _verify_parent_guard_attestation,
)
_GATE_SCRIPT = ROOT / "scripts/qwen38_challenge_port_gate.py"
_GATE_SPEC = importlib.util.spec_from_file_location(
    "qwen38_challenge_port_gate",
    _GATE_SCRIPT,
)
if _GATE_SPEC is None or _GATE_SPEC.loader is None:
    raise RuntimeError(f"cannot load gate module: {_GATE_SCRIPT}")
gate = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(gate)


BUFFER_ENV = ("MLX_MAX_MB_PER_BUFFER", "MLX_MAX_OPS_PER_BUFFER")
GUARD_FD_ENV = "MTPLX_GUARD_ATTEST_FD"
GUARD_NONCE_ENV = "MTPLX_GUARD_ATTEST_NONCE"


class NativeMTPBundleDelta:
    def __init__(
        self,
        *,
        control_features: frozenset[str],
        candidate_features_set: frozenset[str],
        candidate_feature: str,
        candidate_features: tuple[str, ...],
        added: frozenset[str],
        removed: frozenset[str],
    ) -> None:
        self.control_features = control_features
        self.candidate_features_set = candidate_features_set
        self.candidate_feature = candidate_feature
        self.candidate_features = candidate_features
        self.added = added
        self.removed = removed


def _environment_for_route(
    route_id: str,
    inherited: Mapping[str, str],
) -> dict[str, str]:
    features = gate._validate_route_id(route_id)
    environment = dict(inherited)
    for name in (*BUFFER_ENV, GUARD_FD_ENV, GUARD_NONCE_ENV):
        environment.pop(name, None)
    if "r53_command_buffers" in features:
        environment["MLX_MAX_MB_PER_BUFFER"] = "512"
        environment["MLX_MAX_OPS_PER_BUFFER"] = "50"
    return environment


@contextmanager
def _gpu_lock_scope(lock_path: Path) -> Iterator[str]:
    if _verify_parent_guard_attestation(lock_path):
        yield "attested_parent"
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"GPU lock is busy: {lock_path}") from exc
        yield "direct"
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _run_attested_child(
    command: list[str],
    *,
    environment: Mapping[str, str],
    lock_path: Path,
    owns_process_group: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Delegate the already-verified lock attestation to one direct child."""

    read_fd, write_fd = os.pipe()
    nonce = secrets.token_hex(32)
    child_env = dict(environment)
    child_env[GUARD_FD_ENV] = str(read_fd)
    child_env[GUARD_NONCE_ENV] = nonce
    process: subprocess.Popen[str] | None = None

    def close_fd(fd: int | None) -> None:
        if fd is None:
            return
        try:
            os.close(fd)
        except OSError:
            pass

    try:
        try:
            process = subprocess.Popen(
                command,
                env=child_env,
                pass_fds=(read_fd,),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=owns_process_group,
            )
        except BaseException:
            close_fd(read_fd)
            close_fd(write_fd)
            raise
        close_fd(read_fd)
        read_fd = None
        issued = time.monotonic_ns()
        resolved_lock = lock_path.resolve(strict=True)
        observed = resolved_lock.stat()
        payload = {
            "schema_version": 1,
            "nonce": nonce,
            "guard_pid": os.getpid(),
            "child_pid": process.pid,
            "lock_path": str(resolved_lock),
            "lock_device": observed.st_dev,
            "lock_inode": observed.st_ino,
            "issued_monotonic_ns": issued,
            "expires_monotonic_ns": issued + 60_000_000_000,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        view = memoryview(encoded)
        while view:
            written = os.write(write_fd, view)
            if written <= 0:
                raise BrokenPipeError("attestation pipe accepted no bytes")
            view = view[written:]
        close_fd(write_fd)
        write_fd = None
        stdout, _ = process.communicate()
    except BaseException:
        close_fd(read_fd)
        close_fd(write_fd)
        _terminate_process_group(
            process, owns_process_group=owns_process_group
        )
        raise
    if owns_process_group:
        _terminate_process_group(process, owns_process_group=True)
    return subprocess.CompletedProcess(command, process.returncode, stdout, None)


def _terminate_process_group(
    process: subprocess.Popen[str] | None,
    *,
    owns_process_group: bool = True,
) -> None:
    """Terminate and reap the isolated child session before releasing the lock."""

    if process is None:
        return
    if not owns_process_group:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5.0)
            return
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return

    leader_running = process.poll() is None
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if leader_running:
            process.wait()
        return
    if leader_running:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    if process.poll() is None:
        process.wait()
    deadline = time.monotonic() + 5.0
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"owned child process group {process.pid} survived SIGKILL"
            )
        time.sleep(0.01)


def _positive_wall_mean(arms: list[dict[str, Any]]) -> float:
    if not arms:
        raise ValueError("wall aggregation requires at least one arm")
    values = [float(arm.get("wall_s")) for arm in arms]
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("wall_s must be finite and positive")
    mean = math.fsum(values) / len(values)
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError("mean wall_s must be finite and positive")
    return mean


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=gate.DEFAULT_MODEL)
    parser.add_argument("--prompt-file", type=Path, default=gate.DEFAULT_PROMPT)
    parser.add_argument("--prompt-tokens", type=int, default=16_384)
    parser.add_argument("--context-file", type=Path, default=gate.DEFAULT_CONTEXT)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--warmup-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--draft-temperature", type=float)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "xhigh"),
        default="low",
    )
    parser.add_argument("--order", required=True)
    parser.add_argument("--control-route", required=True)
    parser.add_argument("--candidate-route", required=True)
    parser.add_argument("--allow-frozen-candidate", action="store_true")
    parser.add_argument(
        "--candidate-bundle",
        help="Ordered comma-separated atomic feature bundle added by the candidate.",
    )
    parser.add_argument("--row17-artifact", type=Path)
    parser.add_argument("--row28-artifact", type=Path)
    parser.add_argument("--row36-artifact", type=Path)
    parser.add_argument("--lock", type=Path, default=gate.DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _validate_route_delta(args: argparse.Namespace) -> Any:
    raw_bundle = str(getattr(args, "candidate_bundle", None) or "").strip()
    if raw_bundle:
        bundle = tuple(item.strip() for item in raw_bundle.split(",") if item.strip())
        if len(bundle) < 2 or len(bundle) != len(set(bundle)):
            raise gate.NativeMTPRouteError(
                "candidate bundle must contain at least two unique ordered features"
            )
        control = gate.canonicalize_native_mtp_route(args.control_route)
        candidate = gate.canonicalize_native_mtp_route(args.candidate_route)
        added = candidate - control
        removed = control - candidate
        if added != frozenset(bundle) or removed:
            raise gate.NativeMTPRouteError(
                "candidate bundle must exactly match the added route features"
            )
        current = set(control - {"control"})
        for feature in bundle:
            next_features = (*sorted(current), feature)
            gate.validate_native_mtp_route_delta(
                "+".join(sorted(current)) if current else "control",
                "+".join(next_features),
                allow_frozen_candidate=bool(
                    getattr(args, "allow_frozen_candidate", False)
                ),
            )
            current.add(feature)
        return NativeMTPBundleDelta(
            control_features=control,
            candidate_features_set=candidate,
            candidate_feature="+".join(bundle),
            candidate_features=bundle,
            added=added,
            removed=removed,
        )
    return gate.validate_native_mtp_route_delta(
        args.control_route,
        args.candidate_route,
        allow_frozen_candidate=bool(
            getattr(args, "allow_frozen_candidate", False)
        ),
    )


def _child_command(
    args: argparse.Namespace,
    *,
    route_id: str,
    output: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/qwen38_challenge_port_gate.py"),
        "--model",
        str(args.model),
        "--prompt-file",
        str(args.prompt_file),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--context-file",
        str(args.context_file),
        "--max-tokens",
        str(args.max_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--seed",
        str(args.seed),
        "--target-temperature",
        str(args.target_temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--reasoning-effort",
        str(args.reasoning_effort),
        "--order",
        route_id,
        "--lock",
        str(args.lock),
        "--output",
        str(output),
    ]
    for flag, value in (
        ("--draft-temperature", args.draft_temperature),
        ("--row17-artifact", args.row17_artifact),
        ("--row28-artifact", args.row28_artifact),
        ("--row36-artifact", args.row36_artifact),
    ):
        if value is not None:
            command.extend((flag, str(value)))
    return command


def _receipt_invariant_errors(
    args: argparse.Namespace,
    child_receipts: list[dict[str, Any]],
) -> list[str]:
    """Validate exact workload and frozen identities outside measured children."""

    errors: list[str] = []
    if len(child_receipts) != 4:
        errors.append("isolated bracket requires exactly four child receipts")
        return errors
    expected_prompt_id, _ = gate._read_prompt(args.prompt_file)
    expected_draft_temperature = (
        float(args.draft_temperature)
        if args.draft_temperature is not None
        else float(child_receipts[0].get("draft_temperature", 1.0))
    )
    exact_values = {
        "prompt_file": str(args.prompt_file.resolve()),
        "context_file": str(args.context_file.resolve()),
        "prompt_id": expected_prompt_id,
        "prompt_token_target": int(args.prompt_tokens),
        "prompt_tokens": int(args.prompt_tokens),
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed),
        "target_temperature": float(args.target_temperature),
        "draft_temperature": expected_draft_temperature,
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "enable_thinking": True,
        "reasoning_effort": str(args.reasoning_effort),
        "mlx_version": gate.REQUIRED_MLX_VERSION,
        "mlx_metal_version": gate.REQUIRED_MLX_METAL_VERSION,
        "gpu_lock_path": str(args.lock.resolve()),
    }
    for index, receipt in enumerate(child_receipts):
        for key, expected in exact_values.items():
            if receipt.get(key) != expected:
                label = "mlx-metal" if key == "mlx_metal_version" else key
                errors.append(
                    f"child {index} {label} mismatch: {receipt.get(key)!r} != {expected!r}"
                )
        if receipt.get("source_status"):
            errors.append(f"child {index} source tree is not clean")
        if receipt.get("gpu_lock_scope") != "attested_parent":
            errors.append(f"child {index} did not attest the parent GPU lock")
        if len(receipt.get("warmups") or ()) != 1 or any(
            int(run.get("generated_tokens", -1)) != int(args.warmup_tokens)
            for run in receipt.get("warmups") or ()
        ):
            errors.append(f"child {index} conditioner token count is not exact")
        if len(receipt.get("arms") or ()) != 1 or any(
            int(run.get("generated_tokens", -1)) != int(args.max_tokens)
            for run in receipt.get("arms") or ()
        ):
            errors.append(f"child {index} timed output token count is not exact")
        expected_route = (
            args.control_route,
            args.candidate_route,
            args.candidate_route,
            args.control_route,
        )[index]
        if (receipt.get("arms") or [{}])[0].get("route_id") != expected_route:
            errors.append(f"child {index} timed route does not match ABBA order")

    for key in (
        "source_commit",
        "model_artifact_hashes",
        "context_sha256",
        "prompt_token_sha256",
    ):
        if len(
            {
                json.dumps(item.get(key), sort_keys=True, allow_nan=False)
                for item in child_receipts
            }
        ) != 1:
            errors.append(f"child {key} values are not identical")
    if not child_receipts[0].get("model_artifact_hashes"):
        errors.append("model artifact hashes are missing")

    by_route: dict[str, set[str]] = {}
    frozen_fingerprints: dict[str, set[str]] = {}
    route_fingerprints: dict[str, set[str]] = {}
    installed_route_ids: dict[str, set[str]] = {}
    for receipt in child_receipts:
        arm = (receipt.get("arms") or [{}])[0]
        route = str(arm.get("route_id") or "")
        route_fingerprint = str(arm.get("route_fingerprint") or "")
        installed_route_id = str(arm.get("installed_route_id") or "")
        frozen_fingerprint = str(
            receipt.get("frozen_substrate_fingerprint") or ""
        )
        if not frozen_fingerprint:
            errors.append(f"{route} frozen_substrate_fingerprint is missing")
        if not route_fingerprint:
            errors.append(f"{route} route_fingerprint is missing")
        if not installed_route_id:
            errors.append(f"{route} installed_route_id is missing")
        route_fingerprints.setdefault(route, set()).add(route_fingerprint)
        frozen_fingerprints.setdefault(route, set()).add(frozen_fingerprint)
        installed_route_ids.setdefault(route, set()).add(installed_route_id)
        by_route.setdefault(route, set()).add(
            json.dumps(
                arm.get("candidate_artifact_hashes") or {},
                sort_keys=True,
                allow_nan=False,
            )
        )
    if any(len(values) != 1 for values in route_fingerprints.values()):
        errors.append("route_fingerprint is not identical within route")
    if any(len(values) != 1 for values in frozen_fingerprints.values()):
        errors.append(
            "frozen_substrate_fingerprint is not identical within route"
        )
    unique_frozen = set().union(*frozen_fingerprints.values())
    if bool(getattr(args, "allow_frozen_candidate", False)):
        if len(unique_frozen) != 2:
            errors.append(
                "frozen candidate requires distinct control and candidate "
                "frozen_substrate_fingerprint values"
            )
    elif len(unique_frozen) != 1:
        errors.append(
            "child frozen_substrate_fingerprint values are not identical"
        )
    if any(len(values) != 1 for values in installed_route_ids.values()):
        errors.append("installed_route_id is not identical within route")
    if any(len(values) != 1 for values in by_route.values()):
        errors.append("candidate artifact hashes are not deterministic within route")

    try:
        delta = _validate_route_delta(args)
    except gate.NativeMTPRouteError:
        return errors
    delta_features = getattr(
        delta, "candidate_features", (delta.candidate_feature,)
    )
    for delta_feature in delta_features:
        expected_artifact = gate._expected_candidate_artifact_hashes(delta_feature)
        if expected_artifact is None:
            continue
        candidate_arms = [
            (receipt.get("arms") or [{}])[0]
            for receipt in child_receipts
            if (receipt.get("arms") or [{}])[0].get("route_id")
            == args.candidate_route
        ]
        if not candidate_arms or any(
            (arm.get("candidate_artifact_hashes") or {}).get(
                delta_feature
            )
            != expected_artifact
            for arm in candidate_arms
        ):
            errors.append(
                f"{delta_feature} artifact hashes do not match the registry"
            )
    return errors


def _aggregate(
    args: argparse.Namespace,
    *,
    order: list[str],
    child_receipts: list[dict[str, Any]],
    lock_scope: str,
) -> dict[str, Any]:
    arms = [receipt["arms"][0] for receipt in child_receipts]
    warmups = [receipt["warmups"][0] for receipt in child_receipts]
    unique_routes = list(dict.fromkeys(order))
    correctness = gate._correctness_summary(
        arms,
        route_ids=unique_routes,
        max_tokens=args.max_tokens,
    )
    means = {
        route_id: _positive_wall_mean(
            [arm for arm in arms if arm["route_id"] == route_id]
        )
        for route_id in unique_routes
    }
    improvement_pct = (
        means[args.control_route] / means[args.candidate_route] - 1.0
    ) * 100.0
    if not math.isfinite(improvement_pct):
        raise ValueError("candidate wall improvement must be finite")
    source_status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    ).splitlines()
    engagement_errors = gate._candidate_engagement_errors(
        args.candidate_route,
        warmups,
        arms,
    )
    delta = _validate_route_delta(args)
    promotion = gate._promotion_decision(
        order=order,
        control_id=args.control_route,
        candidate_id=args.candidate_route,
        improvement_pct=improvement_pct,
        correctness=correctness,
        source_status=source_status,
        engagement_errors=engagement_errors,
        allow_frozen_candidate=bool(
            getattr(args, "allow_frozen_candidate", False)
        ),
        validated_route_delta=delta,
    )
    receipt_invariant_errors = _receipt_invariant_errors(args, child_receipts)
    phase_summary = gate._phase_summary(
        arms,
        control_id=args.control_route,
        candidate_id=args.candidate_route,
    )
    if receipt_invariant_errors:
        promotion = {
            **promotion,
            "passed": False,
            "errors": [*promotion["errors"], *receipt_invariant_errors],
        }
    first = child_receipts[0]
    return {
        **{
            key: first[key]
            for key in (
                "model",
                "prompt_file",
                "context_file",
                "context_sha256",
                "prompt_id",
                "prompt_tokens",
                "prompt_token_sha256",
                "prompt_token_target",
                "max_tokens",
                "seed",
                "target_temperature",
                "draft_temperature",
                "top_p",
                "top_k",
                "enable_thinking",
                "reasoning_effort",
                "optimized_speed_stack",
                "platform",
                "python",
                "mlx_version",
                "mlx_metal_version",
                "source_commit",
                "frozen_substrate_fingerprint",
                "model_artifact_hashes",
                "gpu_lock_path",
            )
        },
        "kind": "qwen38_challenge_port_isolated_gate",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "isolation_reason": (
            "row53_process_latched_command_buffer_environment"
            if any(
                "r53_command_buffers" in gate._validate_route_id(route_id)
                for route_id in order
            )
            else "fresh_process_per_abba_arm"
        ),
        "conditioning_scope": "one_1024_token_generation_per_isolated_arm_process",
        "timed_arm_count": 4,
        "order": order,
        "gpu_lock_scope": lock_scope,
        "gpu_lock_path": str(args.lock.resolve()),
        "source_status": source_status,
        "exact": bool(
            correctness["cross_route_token_exact"]
            and correctness["cross_route_schedule_exact"]
        ),
        "token_exact": correctness["cross_route_token_exact"],
        "schedule_exact": correctness["cross_route_schedule_exact"],
        "correctness": correctness,
        "control_route_id": args.control_route,
        "candidate_route_id": args.candidate_route,
        "candidate_feature": delta.candidate_feature,
        "mean_wall_s": means,
        "candidate_improvement_pct": improvement_pct,
        "phase_summary": phase_summary,
        "receipt_invariant_errors": receipt_invariant_errors,
        "candidate_engagement_errors": engagement_errors,
        "promotion": promotion,
        "warmups": warmups,
        "arms": arms,
    }


def main() -> int:
    args = _parse_args()
    order = [item.strip() for item in args.order.split(",") if item.strip()]
    expected = [
        args.control_route,
        args.candidate_route,
        args.candidate_route,
        args.control_route,
    ]
    if order != expected:
        raise ValueError("isolated gate requires exactly four ABBA routes")
    for route_id in order:
        gate._validate_route_id(route_id)
    _validate_route_delta(args)

    child_receipts: list[dict[str, Any]] = []
    with _gpu_lock_scope(args.lock) as lock_scope:
        if lock_scope == "direct":
            model_artifact_hashes = gate._model_artifact_hashes(
                args.model.expanduser().resolve()
            )
        else:
            model_artifact_hashes = gate._attested_model_artifact_hashes(
                args.model.expanduser().resolve(),
                guarded_by_parent=True,
            )
        child_environment = dict(os.environ)
        child_environment[gate.MODEL_ARTIFACT_HASHES_ENV] = json.dumps(
            model_artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with tempfile.TemporaryDirectory(prefix="qwen38-r53-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, route_id in enumerate(order):
                child_output = temp_root / f"arm-{index}.json"
                result = _run_attested_child(
                    _child_command(args, route_id=route_id, output=child_output),
                    environment=_environment_for_route(
                        route_id, child_environment
                    ),
                    lock_path=args.lock,
                    owns_process_group=lock_scope == "direct",
                )
                if result.returncode not in (0, 2) or not child_output.is_file():
                    raise RuntimeError(
                        f"isolated arm {index} failed ({result.returncode}):\n"
                        f"{result.stdout}"
                    )
                child_receipts.append(
                    json.loads(child_output.read_text(encoding="utf-8"))
                )

    receipt = _aggregate(
        args,
        order=order,
        child_receipts=child_receipts,
        lock_scope=lock_scope,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "exact": receipt["exact"],
                "candidate_improvement_pct": receipt[
                    "candidate_improvement_pct"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if receipt["promotion"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
