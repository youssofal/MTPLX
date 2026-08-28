from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1] / "scripts/qwen38_challenge_port_isolated_gate.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "qwen38_challenge_port_isolated_gate_focused", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "model": Path("/model"),
        "prompt_file": Path("/repo/mtplx/benchmarks/prompts/python_modules_long.jsonl"),
        "prompt_tokens": 1024,
        "context_file": Path("/repo/mtplx/generation.py"),
        "max_tokens": 1024,
        "warmup_tokens": 1024,
        "seed": 42,
        "target_temperature": 1.0,
        "draft_temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "reasoning_effort": "low",
        "order": "control,r08_device_draft,r08_device_draft,control",
        "control_route": "control",
        "candidate_route": "r08_device_draft",
        "allow_frozen_candidate": False,
        "candidate_bundle": None,
        "row17_artifact": None,
        "row28_artifact": None,
        "row36_artifact": None,
        "lock": Path("/tmp/mtplx-gpu-exclusive.lock"),
        "output": Path("/tmp/output.json"),
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_rebench_delta_flag_allows_one_frozen_feature() -> None:
    isolated = _module()
    control = "r08_device_draft+r10_compact_vocab"
    candidate = control + "+r18_gdn_decay_memo"

    delta = isolated._validate_route_delta(
        _args(
            control_route=control,
            candidate_route=candidate,
            allow_frozen_candidate=True,
        )
    )

    assert delta.candidate_feature == "r18_gdn_decay_memo"


@pytest.mark.parametrize(
    ("bundle", "candidate"),
    [
        (
            "r08_device_draft,r10_compact_vocab",
            "r20_kv_only_history+r53_command_buffers+"
            "r08_device_draft+r10_compact_vocab",
        ),
        (
            "r21_qk_rms_rope,r24_eval_ladder,r26_prefill_ladder_3",
            "r20_kv_only_history+r53_command_buffers+"
            "r21_qk_rms_rope+r24_eval_ladder+r26_prefill_ladder_3",
        ),
    ],
)
def test_bundle_delta_validates_each_dependency_in_order(bundle, candidate) -> None:
    isolated = _module()
    control = "r20_kv_only_history+r53_command_buffers"

    delta = isolated._validate_route_delta(
        _args(
            control_route=control,
            candidate_route=candidate,
            candidate_bundle=bundle,
            allow_frozen_candidate="r21_qk_rms_rope" in bundle,
        )
    )

    assert delta.candidate_features == tuple(bundle.split(","))
    assert delta.candidate_feature == "+".join(bundle.split(","))


def test_bundle_delta_rejects_features_not_exactly_added() -> None:
    isolated = _module()
    control = "r20_kv_only_history+r53_command_buffers"

    with pytest.raises(isolated.gate.NativeMTPRouteError, match="exactly match"):
        isolated._validate_route_delta(
            _args(
                control_route=control,
                candidate_route=control + "+r08_device_draft",
                candidate_bundle="r08_device_draft,r10_compact_vocab",
            )
        )


def test_frozen_rebench_accepts_stable_route_specific_substrate_fingerprints() -> None:
    isolated = _module()
    control = "control"
    candidate = "r18_gdn_decay_memo"
    args = _args(
        prompt_file=isolated.gate.DEFAULT_PROMPT,
        context_file=isolated.gate.DEFAULT_CONTEXT,
        order=f"{control},{candidate},{candidate},{control}",
        control_route=control,
        candidate_route=candidate,
        allow_frozen_candidate=True,
    )
    base = {
        "prompt_file": str(args.prompt_file),
        "context_file": str(args.context_file),
        "prompt_id": isolated.gate._read_prompt(args.prompt_file)[0],
        "prompt_token_target": 1024,
        "prompt_tokens": 1024,
        "max_tokens": 1024,
        "seed": 42,
        "target_temperature": 1.0,
        "draft_temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "low",
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "source_status": [],
        "source_commit": "abc",
        "gpu_lock_scope": "attested_parent",
        "gpu_lock_path": str(args.lock.resolve()),
        "model_artifact_hashes": {"config.json": "123"},
        "warmups": [{"generated_tokens": 1024}],
    }
    routes = (control, candidate, candidate, control)
    receipts = [
        {
            **base,
            "frozen_substrate_fingerprint": f"frozen-{route}",
            "arms": [
                {
                    "generated_tokens": 1024,
                    "route_id": route,
                    "route_fingerprint": f"route-{route}",
                    "installed_route_id": f"installed-{route}",
                    "feature_receipt": {
                        "r18_gdn_decay_memo": {"active_modules": 48}
                    }
                    if route == candidate
                    else {},
                }
            ],
        }
        for route in routes
    ]

    assert isolated._receipt_invariant_errors(args, receipts) == []
    receipts[2] = {
        **receipts[2],
        "frozen_substrate_fingerprint": "candidate-drift",
    }
    assert any(
        "frozen_substrate_fingerprint" in error
        for error in isolated._receipt_invariant_errors(args, receipts)
    )


def test_child_command_preserves_the_exact_sampler_and_conditioner() -> None:
    isolated = _module()

    command = isolated._child_command(
        _args(), route_id="control", output=Path("/tmp/arm.json")
    )

    joined = " ".join(command)
    assert "--prompt-tokens 1024" in joined
    assert "--max-tokens 1024" in joined
    assert "--warmup-tokens 1024" in joined
    assert "--target-temperature 1.0" in joined
    assert "--draft-temperature 1.0" in joined
    assert "--top-p 0.95" in joined
    assert "--top-k 20" in joined
    assert "--seed 42" in joined


def test_receipt_invariants_require_versions_fingerprints_and_exact_counts() -> None:
    isolated = _module()
    args = _args(
        prompt_file=isolated.gate.DEFAULT_PROMPT,
        context_file=isolated.gate.DEFAULT_CONTEXT,
    )
    base = {
        "prompt_file": str(args.prompt_file),
        "context_file": str(args.context_file),
        "prompt_id": isolated.gate._read_prompt(args.prompt_file)[0],
        "prompt_token_target": 1024,
        "prompt_tokens": 1024,
        "max_tokens": 1024,
        "seed": 42,
        "target_temperature": 1.0,
        "draft_temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "low",
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "source_status": [],
        "source_commit": "abc",
        "gpu_lock_scope": "attested_parent",
        "gpu_lock_path": str(args.lock.resolve()),
        "frozen_substrate_fingerprint": "frozen",
        "model_artifact_hashes": {"config.json": "123"},
        "warmups": [{"generated_tokens": 1024}],
        "arms": [
            {
                "generated_tokens": 1024,
                "route_fingerprint": "fingerprint",
                "installed_route_id": "installed",
            }
        ],
    }
    receipts = [{**base, "arms": [{**base["arms"][0], "route_id": route}]} for route in (
        "control",
        "r08_device_draft",
        "r08_device_draft",
        "control",
    )]

    assert isolated._receipt_invariant_errors(args, receipts) == []
    receipts[2] = {**receipts[2], "mlx_metal_version": "0.32.1"}
    errors = isolated._receipt_invariant_errors(args, receipts)
    assert any("mlx-metal" in error for error in errors)


def test_receipt_invariants_require_stable_installed_route_identity() -> None:
    isolated = _module()
    args = _args(
        prompt_file=isolated.gate.DEFAULT_PROMPT,
        context_file=isolated.gate.DEFAULT_CONTEXT,
    )
    base = {
        "prompt_file": str(args.prompt_file),
        "context_file": str(args.context_file),
        "prompt_id": isolated.gate._read_prompt(args.prompt_file)[0],
        "prompt_token_target": 1024,
        "prompt_tokens": 1024,
        "max_tokens": 1024,
        "seed": 42,
        "target_temperature": 1.0,
        "draft_temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "enable_thinking": True,
        "reasoning_effort": "low",
        "mlx_version": "0.32.2",
        "mlx_metal_version": "0.32.2",
        "source_status": [],
        "source_commit": "abc",
        "gpu_lock_scope": "attested_parent",
        "gpu_lock_path": str(args.lock.resolve()),
        "frozen_substrate_fingerprint": "frozen",
        "model_artifact_hashes": {"mtp.safetensors": "123"},
        "warmups": [{"generated_tokens": 1024}],
    }
    routes = ("control", "r08_device_draft", "r08_device_draft", "control")
    receipts = [
        {
            **base,
            "arms": [
                {
                    "generated_tokens": 1024,
                    "route_id": route,
                    "route_fingerprint": f"fp-{route}",
                    "installed_route_id": f"installed-{route}",
                }
            ],
        }
        for route in routes
    ]
    assert isolated._receipt_invariant_errors(args, receipts) == []

    missing = [*receipts]
    missing[1] = {
        **missing[1],
        "arms": [{**missing[1]["arms"][0], "route_fingerprint": ""}],
    }
    assert any(
        "route_fingerprint" in error
        for error in isolated._receipt_invariant_errors(args, missing)
    )

    mismatch = [*receipts]
    mismatch[2] = {
        **mismatch[2],
        "arms": [
            {**mismatch[2]["arms"][0], "installed_route_id": "different"}
        ],
    }
    assert any(
        "installed_route_id" in error
        for error in isolated._receipt_invariant_errors(args, mismatch)
    )


def test_gpu_lock_scope_attestation_uses_the_exact_same_lock(monkeypatch) -> None:
    isolated = _module()
    seen = []
    monkeypatch.setattr(
        isolated,
        "_verify_parent_guard_attestation",
        lambda path: seen.append(path) or True,
    )

    lock = Path("/tmp/mtplx-gpu-exclusive.lock")
    with isolated._gpu_lock_scope(lock) as scope:
        assert scope == "attested_parent"

    assert seen == [lock]


def test_positive_wall_mean_rejects_zero_and_nonfinite() -> None:
    isolated = _module()

    for invalid in (0.0, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            isolated._positive_wall_mean([{"wall_s": invalid}])


def test_attested_child_popen_failure_closes_both_pipe_fds(monkeypatch, tmp_path) -> None:
    isolated = _module()
    observed_fds = []
    real_pipe = os.pipe

    def tracked_pipe():
        fds = real_pipe()
        observed_fds.extend(fds)
        return fds

    monkeypatch.setattr(isolated.os, "pipe", tracked_pipe)
    monkeypatch.setattr(
        isolated.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("popen failed")),
    )

    with pytest.raises(OSError, match="popen failed"):
        isolated._run_attested_child(
            ["child"], environment={}, lock_path=tmp_path / "lock"
        )

    for fd in observed_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_attested_child_stat_failure_terminates_reaps_and_closes_fds(
    monkeypatch, tmp_path
) -> None:
    isolated = _module()
    observed_fds = []
    terminated = []
    real_pipe = os.pipe

    class Process:
        pid = 12345
        returncode = None

    process = Process()

    def tracked_pipe():
        fds = real_pipe()
        observed_fds.extend(fds)
        return fds

    monkeypatch.setattr(isolated.os, "pipe", tracked_pipe)
    monkeypatch.setattr(isolated.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        isolated,
        "_terminate_process_group",
        lambda child, **_kwargs: terminated.append(child),
    )

    with pytest.raises(FileNotFoundError):
        isolated._run_attested_child(
            ["child"], environment={}, lock_path=tmp_path / "missing-lock"
        )

    assert terminated == [process]
    for fd in observed_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_attested_child_write_failure_terminates_reaps_and_closes_fds(
    monkeypatch, tmp_path
) -> None:
    isolated = _module()
    lock = tmp_path / "lock"
    lock.touch()
    observed_fds = []
    terminated = []
    real_pipe = os.pipe

    class Process:
        pid = 12345
        returncode = None

    process = Process()

    def tracked_pipe():
        fds = real_pipe()
        observed_fds.extend(fds)
        return fds

    monkeypatch.setattr(isolated.os, "pipe", tracked_pipe)
    monkeypatch.setattr(isolated.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        isolated.os,
        "write",
        lambda *args, **kwargs: (_ for _ in ()).throw(BrokenPipeError("write")),
    )
    monkeypatch.setattr(
        isolated,
        "_terminate_process_group",
        lambda child, **_kwargs: terminated.append(child),
    )

    with pytest.raises(BrokenPipeError, match="write"):
        isolated._run_attested_child(["child"], environment={}, lock_path=lock)

    assert terminated == [process]
    for fd in observed_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_attested_child_interrupted_communicate_terminates_and_reaps(
    monkeypatch, tmp_path
) -> None:
    isolated = _module()
    lock = tmp_path / "lock"
    lock.touch()
    observed_fds = []
    terminated = []
    popen_kwargs = {}
    real_pipe = os.pipe

    class Process:
        pid = 12345
        returncode = None

        @staticmethod
        def communicate():
            raise KeyboardInterrupt

    process = Process()

    def popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return process

    def tracked_pipe():
        fds = real_pipe()
        observed_fds.extend(fds)
        return fds

    monkeypatch.setattr(isolated.os, "pipe", tracked_pipe)
    monkeypatch.setattr(isolated.subprocess, "Popen", popen)
    monkeypatch.setattr(isolated.os, "write", lambda _fd, view: len(view))
    monkeypatch.setattr(
        isolated,
        "_terminate_process_group",
        lambda child, **_kwargs: terminated.append(child),
    )

    with pytest.raises(KeyboardInterrupt):
        isolated._run_attested_child(["child"], environment={}, lock_path=lock)

    assert popen_kwargs["start_new_session"] is True
    assert terminated == [process]
    for fd in observed_fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_terminate_process_group_escalates_and_reaps(monkeypatch) -> None:
    isolated = _module()
    signals = []

    class Process:
        pid = 12345
        waits = 0

        @staticmethod
        def poll():
            return None

        def wait(self, *, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            return -signal.SIGKILL

    process = Process()
    def killpg(pid, sig):
        if sig == 0:
            raise ProcessLookupError
        signals.append((pid, sig))

    monkeypatch.setattr(isolated.os, "killpg", killpg)

    isolated._terminate_process_group(process)

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.waits == 2


def test_owned_group_cleanup_kills_descendants_after_leader_exit(monkeypatch) -> None:
    isolated = _module()
    signals = []

    class Process:
        pid = 12345

        @staticmethod
        def poll():
            return 0

    def killpg(pid, sig):
        if sig == 0:
            raise ProcessLookupError
        signals.append((pid, sig))

    monkeypatch.setattr(isolated.os, "killpg", killpg)

    isolated._terminate_process_group(Process(), owns_process_group=True)

    assert (12345, signal.SIGTERM) in signals
    assert (12345, signal.SIGKILL) in signals


def test_nested_owned_group_interrupt_removes_real_grandchild(monkeypatch, tmp_path) -> None:
    isolated = _module()
    lock = tmp_path / "lock"
    lock.touch()
    grandchild_pid = []
    real_popen = subprocess.Popen
    child_code = (
        "import subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "print(p.pid, flush=True); time.sleep(60)"
    )

    def interrupting_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)

        def interrupted_communicate():
            assert process.stdout is not None
            grandchild_pid.append(int(process.stdout.readline().strip()))
            raise KeyboardInterrupt

        process.communicate = interrupted_communicate
        return process

    monkeypatch.setattr(isolated.subprocess, "Popen", interrupting_popen)

    with pytest.raises(KeyboardInterrupt):
        isolated._run_attested_child(
            [sys.executable, "-c", child_code],
            environment=os.environ,
            lock_path=lock,
            owns_process_group=True,
        )

    assert grandchild_pid
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid[0], 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"grandchild {grandchild_pid[0]} survived owned-group cleanup")
