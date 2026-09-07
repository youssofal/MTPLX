from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import threading

import pytest

from mtplx.agent_workspace import WorkspaceStore
from mtplx.workspace_tools import (
    FIRST_PARTY_TOOL_NAMES,
    WorkspaceToolPermissionError,
    WorkspaceToolService,
    first_party_tool_definitions,
)


def _git(project, *args):
    return subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("# MTPLX\n", encoding="utf-8")
    _git(project, "init", "-q")
    _git(project, "add", "README.md")
    _git(
        project,
        "-c",
        "user.name=MTPLX Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "initial",
    )
    return project


def _approve(store, response):
    approval = response["approval"]
    store.resolve_approval(approval["id"], "approved", resolved_by="test")
    return approval["id"]


def test_first_party_tool_catalog_is_exact():
    definitions = first_party_tool_definitions()
    assert tuple(item["function"]["name"] for item in definitions) == FIRST_PARTY_TOOL_NAMES
    assert len(definitions) == 10


def test_write_requires_exact_one_time_approval_and_records_events(tmp_path):
    project = _repository(tmp_path)
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project))
    run = store.create_run(workspace.id, title="Write and verify")
    service = WorkspaceToolService(store, sandbox_mode="off")
    arguments = {"path": "result.txt", "content": "verified\n"}

    pending = service.execute(workspace.id, "write_file", arguments, run_id=run.id)
    assert pending["status"] == "approval_required"
    assert not (project / "result.txt").exists()
    approval_id = _approve(store, pending)

    mismatched = service.execute(
        workspace.id,
        "write_file",
        {**arguments, "content": "changed\n"},
        run_id=run.id,
        approval_id=approval_id,
    )
    assert mismatched["status"] == "approval_invalid"
    assert not (project / "result.txt").exists()

    other_run = store.create_run(workspace.id, title="Different session")
    wrong_run = service.execute(
        workspace.id,
        "write_file",
        arguments,
        run_id=other_run.id,
        approval_id=approval_id,
    )
    assert wrong_run["status"] == "approval_invalid"
    assert not (project / "result.txt").exists()

    completed = service.execute(
        workspace.id,
        "write_file",
        arguments,
        run_id=run.id,
        approval_id=approval_id,
    )
    assert completed["ok"] is True
    assert (project / "result.txt").read_text(encoding="utf-8") == "verified\n"

    replayed = service.execute(
        workspace.id,
        "write_file",
        arguments,
        run_id=run.id,
        approval_id=approval_id,
    )
    assert replayed["status"] == "approval_invalid"
    kinds = [event.kind for event in store.list_events(run.id)]
    assert kinds == [
        "run_created",
        "approval_requested",
        "approval_resolved",
        "approval_consumed",
        "tool_call",
        "tool_result",
        "file_changed",
    ]


def test_path_and_network_boundaries_are_enforced(tmp_path):
    project = _repository(tmp_path)
    (project / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace(
        "MTPLX",
        str(project),
        tool_policy={"terminal": "allow", "network": "deny"},
    )
    service = WorkspaceToolService(store, sandbox_mode="off")

    outside = service.execute(workspace.id, "read_file", {"path": "../outside"})
    assert outside["ok"] is False
    assert "outside workspace" in outside["result"]["error"]
    secret = service.execute(workspace.id, "read_file", {"path": ".env"})
    assert secret["ok"] is False
    assert "sensitive path" in secret["result"]["error"]

    denied = service.execute(
        workspace.id,
        "run_command",
        {"command": "true", "network": True},
    )
    assert denied["status"] == "denied"
    allowed = service.execute(
        workspace.id,
        "run_command",
        {"command": "printf ok", "network": False},
    )
    assert allowed["ok"] is True
    assert allowed["result"]["stdout"] == "ok"
    with pytest.raises(WorkspaceToolPermissionError):
        service.preview(
            workspace.id,
            "run_command",
            {"command": "true", "network": True},
            permissions=["terminal"],
        )
    browser_denied = service.authorize_external_action(
        workspace.id,
        "web_search",
        {"query": "MTPLX"},
    )
    assert browser_denied["status"] == "denied"


def test_external_browser_action_consumes_exact_network_authorization(tmp_path):
    project = _repository(tmp_path)
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project))
    run = store.create_run(workspace.id, title="browser authorization")
    service = WorkspaceToolService(store, sandbox_mode="off")
    arguments = {"query": "MTPLX local agent"}

    pending = service.authorize_external_action(
        workspace.id,
        "web_search",
        arguments,
        run_id=run.id,
    )
    assert pending["status"] == "approval_required"
    approval_id = _approve(store, pending)
    authorized = service.authorize_external_action(
        workspace.id,
        "web_search",
        arguments,
        run_id=run.id,
        approval_id=approval_id,
        executor_id="desktop-test",
    )
    assert authorized["ok"] is True
    assert authorized["status"] == "authorized"
    replayed = service.authorize_external_action(
        workspace.id,
        "web_search",
        arguments,
        run_id=run.id,
        approval_id=approval_id,
        executor_id="desktop-test",
    )
    assert replayed["status"] == "approval_invalid"
    assert "external_action_authorized" in [
        event.kind for event in store.list_events(run.id)
    ]


def test_side_effect_idempotency_replays_result_without_reexecuting(tmp_path):
    project = _repository(tmp_path)
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace(
        "MTPLX",
        str(project),
        tool_policy={"write": "allow"},
    )
    service = WorkspaceToolService(store, sandbox_mode="off")
    arguments = {"path": "guarded.txt", "content": "first\n"}

    first = service.execute(
        workspace.id,
        "write_file",
        arguments,
        idempotency_key="graph-run-1:node-write:attempt-1",
    )
    assert first["ok"] is True
    (project / "guarded.txt").write_text("externally changed\n", encoding="utf-8")
    replay = service.execute(
        workspace.id,
        "write_file",
        arguments,
        idempotency_key="graph-run-1:node-write:attempt-1",
    )
    assert replay["ok"] is True
    assert replay["replayed"] is True
    assert (project / "guarded.txt").read_text(encoding="utf-8") == "externally changed\n"
    conflict = service.execute(
        workspace.id,
        "write_file",
        {**arguments, "content": "second\n"},
        idempotency_key="graph-run-1:node-write:attempt-1",
    )
    assert conflict["status"] == "idempotency_conflict"


def test_concurrent_services_dispatch_one_idempotent_mutation(tmp_path, monkeypatch):
    project = _repository(tmp_path)
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace(
        "MTPLX",
        str(project),
        tool_policy={"write": "allow"},
    )
    run = store.create_run(workspace.id, title="Concurrent idempotency")
    first_service = WorkspaceToolService(store, sandbox_mode="off")
    second_service = WorkspaceToolService(store, sandbox_mode="off")
    arguments = {"path": "concurrent.txt", "content": "written once\n"}
    idempotency_key = f"{run.id}:node-write:attempt-1"

    dispatch_started = threading.Event()
    allow_dispatch = threading.Event()
    dispatch_lock = threading.Lock()
    dispatch_count = 0

    def instrument_write(original):
        def write_once(write_arguments, root):
            nonlocal dispatch_count
            with dispatch_lock:
                dispatch_count += 1
            dispatch_started.set()
            assert allow_dispatch.wait(timeout=5), "timed out releasing write dispatch"
            return original(write_arguments, root)

        return write_once

    monkeypatch.setattr(
        first_service,
        "_write_file",
        instrument_write(first_service._write_file),
    )
    monkeypatch.setattr(
        second_service,
        "_write_file",
        instrument_write(second_service._write_file),
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_future = executor.submit(
            first_service.execute,
            workspace.id,
            "write_file",
            arguments,
            run_id=run.id,
            idempotency_key=idempotency_key,
        )
        try:
            assert dispatch_started.wait(timeout=5), "write dispatch did not start"
            duplicate = second_service.execute(
                workspace.id,
                "write_file",
                arguments,
                run_id=run.id,
                idempotency_key=idempotency_key,
            )
            conflict = second_service.execute(
                workspace.id,
                "write_file",
                {**arguments, "content": "conflicting write\n"},
                run_id=run.id,
                idempotency_key=idempotency_key,
            )
        finally:
            allow_dispatch.set()
        first = first_future.result(timeout=5)

    replay = second_service.execute(
        workspace.id,
        "write_file",
        arguments,
        run_id=run.id,
        idempotency_key=idempotency_key,
    )

    assert first["ok"] is True
    assert duplicate["status"] == "execution_indeterminate"
    assert conflict["status"] == "idempotency_conflict"
    assert replay["ok"] is True
    assert replay["replayed"] is True
    assert dispatch_count == 1
    assert [
        event.kind for event in store.list_events(run.id) if event.kind == "tool_call"
    ] == ["tool_call"]
    assert (project / "concurrent.txt").read_text(encoding="utf-8") == "written once\n"


def test_real_write_test_diff_and_restart_acceptance_path(tmp_path):
    project = _repository(tmp_path)
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project), model="local-test-model")
    run = store.create_run(workspace.id, title="Acceptance workflow")
    store.update_run(run.id, status="running")
    store.append_event(run.id, "plan_created", {"steps": ["write", "test", "review"]})
    service = WorkspaceToolService(store, sandbox_mode="off")

    write_args = {"path": "answer.py", "content": "ANSWER = 42\n"}
    write_pending = service.execute(workspace.id, "write_file", write_args, run_id=run.id)
    write_approval = _approve(store, write_pending)
    assert service.execute(
        workspace.id,
        "write_file",
        write_args,
        run_id=run.id,
        approval_id=write_approval,
    )["ok"]

    test_args = {
        "command": "python -c 'import answer; assert answer.ANSWER == 42'",
        "network": False,
    }
    test_pending = service.execute(workspace.id, "run_tests", test_args, run_id=run.id)
    test_approval = _approve(store, test_pending)
    tested = service.execute(
        workspace.id,
        "run_tests",
        test_args,
        run_id=run.id,
        approval_id=test_approval,
    )
    assert tested["ok"] is True, tested["result"]
    diff = service.execute(workspace.id, "git_diff", {"scope": "unstaged"}, run_id=run.id)
    assert diff["ok"] is True
    assert "answer.py" not in diff["result"]["scopes"][0]["stdout"]
    status = service.execute(workspace.id, "git_status", {}, run_id=run.id)
    assert "?? answer.py" in status["result"]["stdout"]

    restarted = WorkspaceStore(tmp_path / "state")
    recovered = restarted.recover_incomplete_runs()
    assert recovered[0].status == "paused"
    assert restarted.resume_run(run.id).status == "queued"
    events = restarted.list_events(run.id)
    kinds = [event.kind for event in events]
    for required in (
        "run_created",
        "plan_created",
        "approval_requested",
        "approval_resolved",
        "tool_call",
        "tool_result",
        "file_changed",
        "test_started",
        "test_completed",
        "run_paused",
        "run_resumed",
    ):
        assert required in kinds
    assert any(
        event.kind == "test_completed" and event.payload["passed"] is True
        for event in events
    )


@pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS-specific")
def test_required_sandbox_blocks_terminal_writes_outside_workspace(tmp_path):
    project = _repository(tmp_path)
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace(
        "MTPLX",
        str(project),
        tool_policy={"terminal": "allow", "network": "deny"},
    )
    service = WorkspaceToolService(store, sandbox_mode="required")

    harmless = service.execute(
        workspace.id,
        "run_command",
        {"command": "printf ok", "network": False},
    )
    assert harmless["result"]["exit_code"] == 0, harmless["result"]["stderr"]
    assert harmless["ok"] is True
    assert harmless["result"]["sandboxed"] is True

    inside = service.execute(
        workspace.id,
        "run_command",
        {"command": "printf inside > inside.txt", "network": False},
    )
    assert inside["ok"] is True, inside
    assert (project / "inside.txt").read_text(encoding="utf-8") == "inside"

    escape = service.execute(
        workspace.id,
        "run_command",
        {"command": "printf escape > ../escape.txt", "network": False},
    )
    assert escape["ok"] is False
    assert not (tmp_path / "escape.txt").exists()
