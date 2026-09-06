import json
import subprocess
import threading
import time

import pytest

from mtplx.agent_orchestrator import AgentDelegationError, AgentOrchestrator
from mtplx.agent_workspace import WorkspaceStore
from mtplx.workspace_tools import WorkspaceToolService


def _git(project, *args):
    return subprocess.run(
        ["git", "-C", str(project), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_profiles_and_isolated_reviewer_delegation_are_durable(tmp_path):
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

    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project), model="test-model")
    parent = store.create_run(workspace.id, title="parent")
    orchestrator = AgentOrchestrator(store)
    try:
        profiles = {item["id"] for item in orchestrator.profiles()}
        assert {"planner", "implementer", "reviewer", "tester", "research"} <= profiles

        delegation = orchestrator.delegate(
            workspace.id,
            role="reviewer",
            prompt="Review the current change.",
            parent_run_id=parent.id,
            budget=4096,
            start=False,
        )
        assert delegation.status == "queued"
        assert delegation.worktree_path
        assert delegation.worktree_commit
        assert delegation.permissions == ("read", "search")
        assert delegation.budget == 4096
        assert (tmp_path / "state" / "delegations" / f"{delegation.id}.json").exists()

        child = store.get_run(delegation.child_run_id)
        assert child.status == "queued"
        assert [event.kind for event in store.list_events(parent.id)] == [
            "run_created",
            "agent_delegated",
        ]
        assert [event.kind for event in store.list_events(child.id)] == ["run_created"]
    finally:
        if delegation.worktree_path:
            _git(project, "worktree", "remove", "--force", delegation.worktree_path)
        orchestrator.close()


def test_reviewer_records_evidence_and_parent_completion_event(tmp_path):
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

    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project), model="test-model")
    parent = store.create_run(workspace.id, title="parent")
    orchestrator = AgentOrchestrator(
        store,
        tool_service=WorkspaceToolService(store, sandbox_mode="off"),
    )
    responses = iter(
        [
            {
                "model": "test-model",
                "usage": {"completion_tokens": 20},
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_inspect",
                                    "type": "function",
                                    "function": {
                                        "name": "inspect_repo",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "model": "test-model",
                "usage": {"completion_tokens": 30},
                "choices": [
                    {
                        "message": {
                            "content": (
                                "No blocking findings.\n"
                                'MTPLX_REVIEW: {"verdict":"approved",'
                                '"blocking_findings":[],"notes":"clean"}'
                            )
                        }
                    }
                ],
            },
        ]
    )
    orchestrator._chat_completion = lambda **_: next(responses)
    try:
        delegation = orchestrator.delegate(
            workspace.id,
            role="reviewer",
            parent_run_id=parent.id,
            start=True,
        )
        for _ in range(60):
            current = orchestrator.get(delegation.id)
            if current.status in {"completed", "failed"}:
                break
            time.sleep(0.05)
        current = orchestrator.get(delegation.id)
        assert current.status == "completed"
        assert current.evidence["summary"].startswith("No blocking findings.")
        assert current.evidence["review"]["verdict"] == "approved"
        assert current.evidence["completion_evidence"]["verified"] is True
        child_events = [event.kind for event in store.list_events(current.child_run_id)]
        assert "review_started" in child_events
        assert "tool_call" in child_events
        assert "tool_result" in child_events
        assert "agent_completed" in child_events
        assert "run_completed" in child_events
        assert any(
            event.kind == "agent_completed"
            for event in store.list_events(parent.id)
        )
    finally:
        if delegation.worktree_path:
            _git(project, "worktree", "remove", "--force", delegation.worktree_path)
        orchestrator.close()


def test_tester_runs_shared_tests_and_requires_successful_evidence(tmp_path):
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

    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project), model="test-model")
    parent = store.create_run(workspace.id, title="parent")
    service = WorkspaceToolService(store, sandbox_mode="off")
    orchestrator = AgentOrchestrator(store, tool_service=service)
    test_command = (
        'python -c "from pathlib import Path; '
        "assert Path('README.md').read_text(encoding='utf-8') == '# MTPLX\\n'; "
        "print('TESTER_BOUNDARY_OK')\""
    )
    expected_tools = {
        "list_files",
        "read_file",
        "search_files",
        "inspect_repo",
        "git_status",
        "git_diff",
        "run_tests",
        "run_command",
    }
    offered_tool_sets = []
    responses = iter(
        [
            {
                "model": "test-model",
                "usage": {"completion_tokens": 20},
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_tester_run",
                                    "type": "function",
                                    "function": {
                                        "name": "run_tests",
                                        "arguments": json.dumps(
                                            {
                                                "command": test_command,
                                                "network": False,
                                            }
                                        ),
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "model": "test-model",
                "usage": {"completion_tokens": 25},
                "choices": [
                    {
                        "message": {
                            "content": (
                                "Ran the repository probe through run_tests. "
                                "It passed with TESTER_BOUNDARY_OK."
                            )
                        }
                    }
                ],
            },
        ]
    )

    def successful_completion(**kwargs):
        assert kwargs["agent_role"] == "tester"
        offered_tool_sets.append(
            {item["function"]["name"] for item in kwargs["tools"]}
        )
        return next(responses)

    orchestrator._chat_completion = successful_completion
    try:
        successful = orchestrator.delegate(
            workspace.id,
            role="tester",
            prompt="Run the repository verification probe.",
            parent_run_id=parent.id,
            start=True,
        )
        assert successful.permissions == ("read", "search", "terminal")

        approved_ids = set()
        for _ in range(200):
            for approval in store.list_approvals(
                workspace_id=workspace.id,
                run_id=successful.child_run_id,
                status="pending",
            ):
                if approval.id not in approved_ids:
                    store.resolve_approval(
                        approval.id,
                        "approved",
                        resolved_by="test-user",
                    )
                    approved_ids.add(approval.id)
            current = orchestrator.get(successful.id)
            if current.status in {"completed", "failed"}:
                break
            time.sleep(0.05)

        successful = orchestrator.get(successful.id)
        assert successful.status == "completed", successful.error
        assert offered_tool_sets
        assert all(tool_set == expected_tools for tool_set in offered_tool_sets)
        assert "write_file" not in expected_tools
        assert "apply_patch" not in expected_tools
        assert len(approved_ids) == 1
        approval = store.get_approval(next(iter(approved_ids)))
        assert approval.tool == "run_tests"
        assert approval.status == "consumed"
        assert approval.consumed_by == successful.id

        persisted = WorkspaceStore(tmp_path / "state")
        events = persisted.list_events(successful.child_run_id)
        tool_call = next(event for event in events if event.kind == "tool_call")
        tool_result = next(event for event in events if event.kind == "tool_result")
        test_completed = next(event for event in events if event.kind == "test_completed")
        assert tool_call.payload["tool"] == "run_tests"
        assert tool_call.payload["arguments"]["command"] == test_command
        assert tool_call.payload["arguments"]["network"] is False
        assert tool_result.payload["tool"] == "run_tests"
        assert tool_result.payload["ok"] is True
        assert "TESTER_BOUNDARY_OK" in tool_result.payload["result"]["stdout"]
        assert test_completed.payload["command"] == test_command
        assert test_completed.payload["passed"] is True
        assert test_completed.payload["exit_code"] == 0
        assert successful.evidence["tests"][0]["payload"] == test_completed.payload
        assert successful.evidence["completion_evidence"] == {
            "verified": True,
            "reasons": [],
            "successful_tool_results": 1,
            "file_changes": 0,
            "passed_tests": 1,
            "review_verdict": None,
        }
        assert persisted.get_run(successful.child_run_id).status == "completed"

        false_tool_sets = []

        def false_completion(**kwargs):
            assert kwargs["agent_role"] == "tester"
            false_tool_sets.append(
                {item["function"]["name"] for item in kwargs["tools"]}
            )
            return {
                "model": "test-model",
                "usage": {"completion_tokens": 20},
                "choices": [
                    {
                        "message": {
                            "content": "All tests passed. Everything is complete."
                        }
                    }
                ],
            }

        orchestrator._chat_completion = false_completion
        unsupported = orchestrator.delegate(
            workspace.id,
            role="tester",
            prompt="Verify the repository and report completion.",
            parent_run_id=parent.id,
            start=True,
        )
        for _ in range(100):
            unsupported = orchestrator.get(unsupported.id)
            if unsupported.status in {"completed", "failed"}:
                break
            time.sleep(0.05)

        assert unsupported.status == "failed"
        assert false_tool_sets == [expected_tools]
        assert unsupported.evidence["summary"] == (
            "All tests passed. Everything is complete."
        )
        assert unsupported.evidence["tests"] == []
        completion = unsupported.evidence["completion_evidence"]
        assert completion["verified"] is False
        assert completion["passed_tests"] == 0
        assert "tester recorded no successful test" in completion["reasons"]
        assert "tester recorded no successful test" in unsupported.error

        persisted = WorkspaceStore(tmp_path / "state")
        assert persisted.get_run(unsupported.child_run_id).status == "failed"
        unsupported_events = persisted.list_events(unsupported.child_run_id)
        assert not any(event.kind == "test_completed" for event in unsupported_events)
        assert not any(event.kind == "run_completed" for event in unsupported_events)
        failed_event = next(
            event
            for event in unsupported_events
            if event.kind == "agent_completed"
        )
        assert failed_event.payload["status"] == "failed"
    finally:
        for item in orchestrator.list(workspace_id=workspace.id):
            if item.worktree_path:
                _git(project, "worktree", "remove", "--force", item.worktree_path)
        orchestrator.close()


def test_restart_pauses_delegation_and_retry_requeues_child(tmp_path):
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

    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project), model="test-model")
    orchestrator = AgentOrchestrator(store)
    delegation = orchestrator.delegate(workspace.id, role="reviewer", start=False)
    delegation = orchestrator._replace(
        delegation,
        status="running",
        owner_id="dead-orchestrator",
        generation=1,
        lease_expires_at="2000-01-01T00:00:00Z",
    )
    store.update_run(delegation.child_run_id, status="running")
    orchestrator.close()

    restarted = AgentOrchestrator(store)
    try:
        paused = restarted.get(delegation.id)
        assert paused.status == "paused"
        assert store.get_run(paused.child_run_id).status == "paused"
        assert any(
            event.kind == "agent_paused"
            for event in store.list_events(paused.child_run_id)
        )

        queued = restarted.retry(paused.id)
        assert queued.status == "queued"
        assert store.get_run(paused.child_run_id).status == "queued"
    finally:
        if delegation.worktree_path:
            _git(project, "worktree", "remove", "--force", delegation.worktree_path)
        restarted.close()


def test_durable_execution_claim_prevents_duplicate_worker_and_allows_expired_claim(
    tmp_path,
):
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

    state_root = tmp_path / "state"
    store_a = WorkspaceStore(state_root)
    workspace = store_a.create_workspace("MTPLX", str(project), model="test-model")
    orchestrator_a = AgentOrchestrator(store_a, owner_id="owner-a")
    orchestrator_b = AgentOrchestrator(
        WorkspaceStore(state_root),
        owner_id="owner-b",
    )
    delegation = orchestrator_a.delegate(
        workspace.id,
        role="reviewer",
        start=False,
    )
    try:
        claimed = orchestrator_a._claim(delegation.id)
        assert claimed is not None
        assert claimed.owner_id == "owner-a"
        assert claimed.status == "running"
        assert claimed.generation == 1
        assert claimed.attempts == 1
        assert orchestrator_b._claim(delegation.id) is None

        queued = orchestrator_a._replace(
            claimed,
            status="queued",
            owner_id="dead-owner",
            lease_expires_at="2000-01-01T00:00:00Z",
        )
        reclaimed = orchestrator_b._claim(queued.id)
        assert reclaimed is not None
        assert reclaimed.owner_id == "owner-b"
        assert reclaimed.generation == 2
        assert reclaimed.attempts == 2
    finally:
        if delegation.worktree_path:
            _git(project, "worktree", "remove", "--force", delegation.worktree_path)
        orchestrator_a.close()
        orchestrator_b.close()


def test_retry_preserves_durable_token_budget_and_attempt_accounting(tmp_path):
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

    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project), model="test-model")
    orchestrator = AgentOrchestrator(
        store,
        tool_service=WorkspaceToolService(store, sandbox_mode="off"),
    )
    requested_budgets: list[int] = []
    responses = iter(
        [
            {
                "model": "test-model",
                "usage": {"completion_tokens": 200},
                "choices": [
                    {"message": {"content": "No structured review evidence."}}
                ],
            },
            {
                "model": "test-model",
                "usage": {"completion_tokens": 50},
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_inspect_retry",
                                    "type": "function",
                                    "function": {
                                        "name": "inspect_repo",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
            {
                "model": "test-model",
                "usage": {"completion_tokens": 100},
                "choices": [
                    {
                        "message": {
                            "content": (
                                "No blocking findings.\n"
                                'MTPLX_REVIEW: {"verdict":"approved",'
                                '"blocking_findings":[],"notes":"verified"}'
                            )
                        }
                    }
                ],
            },
        ]
    )

    def completion(**kwargs):
        requested_budgets.append(kwargs["max_tokens"])
        return next(responses)

    orchestrator._chat_completion = completion
    try:
        delegation = orchestrator.delegate(
            workspace.id,
            role="reviewer",
            budget=512,
            start=True,
        )
        for _ in range(100):
            failed = orchestrator.get(delegation.id)
            if failed.status == "failed":
                break
            time.sleep(0.05)

        failed = orchestrator.get(delegation.id)
        assert failed.status == "failed"
        assert failed.tokens_used == 200
        assert failed.attempts == 1

        orchestrator.retry(failed.id)
        for _ in range(100):
            completed = orchestrator.get(delegation.id)
            if completed.status in {"completed", "failed"}:
                break
            time.sleep(0.05)

        completed = orchestrator.get(delegation.id)
        assert completed.status == "completed", completed.error
        assert requested_budgets == [512, 312, 262]
        assert completed.tokens_used == 350
        assert completed.attempts == 2
        assert completed.evidence["completion_tokens_used"] == 150
        assert completed.evidence["cumulative_completion_tokens_used"] == 350
        assert completed.evidence["remaining_token_budget"] == 162
        budget_events = [
            event
            for event in store.list_events(completed.child_run_id)
            if event.kind == "agent_budget_updated"
        ]
        assert [event.payload["tokens_used"] for event in budget_events] == [
            200,
            250,
            350,
        ]

        exhausted = orchestrator._replace(
            completed,
            status="failed",
            tokens_used=completed.budget,
        )
        with pytest.raises(AgentDelegationError, match="token budget is exhausted"):
            orchestrator.retry(exhausted.id)
    finally:
        if delegation.worktree_path:
            _git(project, "worktree", "remove", "--force", delegation.worktree_path)
        orchestrator.close()


def test_cancel_interrupts_in_flight_agent_model_request(tmp_path, monkeypatch):
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

    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project), model="test-model")
    orchestrator = AgentOrchestrator(store)
    model_started = threading.Event()
    model_cancelled = threading.Event()
    request_hints: list[str] = []
    cancelled_ids: list[str] = []

    class FakeResponse:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(self.body).encode("utf-8")

    def fake_urlopen(request, timeout):
        if "/v1/chat/completions" in request.full_url:
            request_hint = request.get_header("X-mtplx-request-id")
            request_hints.append(request_hint)
            model_started.set()
            assert model_cancelled.wait(timeout=2)
            return FakeResponse(
                {
                    "model": "test-model",
                    "usage": {"completion_tokens": 16},
                    "choices": [{"message": {"content": "cancelled"}}],
                }
            )
        assert "/v1/mtplx/cancel/" in request.full_url
        cancelled_ids.append(request.full_url.rsplit("/", 1)[-1])
        model_cancelled.set()
        return FakeResponse({"ok": True, "cancelled": True})

    monkeypatch.setattr("mtplx.agent_orchestrator.urlopen", fake_urlopen)
    try:
        delegation = orchestrator.delegate(
            workspace.id,
            role="reviewer",
            budget=512,
            start=True,
        )
        assert model_started.wait(timeout=2)
        cancelled = orchestrator.cancel(delegation.id)
        assert cancelled.status == "cancelled"
        assert model_cancelled.is_set()

        for _ in range(100):
            current = orchestrator.get(delegation.id)
            if not orchestrator._active_requests:
                break
            time.sleep(0.02)
        current = orchestrator.get(delegation.id)
        assert current.status == "cancelled"
        assert store.get_run(current.child_run_id).status == "cancelled"
        assert request_hints
        assert cancelled_ids == [f"chatcmpl-{request_hints[0]}"]
        cancellation_event = next(
            event
            for event in store.list_events(current.child_run_id)
            if event.kind == "agent_model_cancel_requested"
        )
        assert cancellation_event.payload["cancel_endpoint_reached"] is True
        assert cancellation_event.payload["cancelled"] is True
        assert cancellation_event.payload["request_id"] == cancelled_ids[0]
    finally:
        if delegation.worktree_path:
            _git(project, "worktree", "remove", "--force", delegation.worktree_path)
        orchestrator.close()


def test_implementer_approval_tool_loop_review_and_guarded_integration(tmp_path):
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

    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("MTPLX", str(project), model="test-model")
    parent = store.create_run(workspace.id, title="parent")
    service = WorkspaceToolService(store, sandbox_mode="off")
    orchestrator = AgentOrchestrator(store, tool_service=service)
    role_rounds = {"implementer": 0, "reviewer": 0}

    def completion(**kwargs):
        role = kwargs["agent_role"]
        role_rounds[role] += 1
        round_number = role_rounds[role]
        if role == "implementer" and round_number == 1:
            call = {
                "id": "call_write",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": json.dumps(
                        {"path": "answer.py", "content": "ANSWER = 42\n"}
                    ),
                },
            }
            content = ""
        elif role == "implementer" and round_number == 2:
            call = {
                "id": "call_test",
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "arguments": json.dumps(
                        {
                            "command": (
                                "python -c 'import answer; assert answer.ANSWER == 42'"
                            ),
                            "network": False,
                        }
                    ),
                },
            }
            content = ""
        elif role == "reviewer" and round_number == 1:
            call = {
                "id": "call_review_inspect",
                "type": "function",
                "function": {"name": "inspect_repo", "arguments": "{}"},
            }
            content = ""
        elif role == "reviewer":
            call = None
            content = (
                "The patch is scoped and the recorded test passed.\n"
                'MTPLX_REVIEW: {"verdict":"approved","blocking_findings":[],'
                '"notes":"verified source patch and evidence"}'
            )
        else:
            call = None
            content = "Changed answer.py and verified its import."
        message = {"content": content}
        if call is not None:
            message["tool_calls"] = [call]
        return {
            "model": "test-model",
            "usage": {"completion_tokens": 25},
            "choices": [{"message": message}],
        }

    orchestrator._chat_completion = completion
    try:
        implementation = orchestrator.delegate(
            workspace.id,
            role="implementer",
            prompt="Add answer.py and test it.",
            parent_run_id=parent.id,
            start=True,
        )
        approved_ids: set[str] = set()
        for _ in range(200):
            pending = store.list_approvals(
                workspace_id=workspace.id,
                run_id=implementation.child_run_id,
                status="pending",
            )
            for approval in pending:
                if approval.id not in approved_ids:
                    store.resolve_approval(
                        approval.id,
                        "approved",
                        resolved_by="test-user",
                    )
                    approved_ids.add(approval.id)
            current = orchestrator.get(implementation.id)
            if current.status in {"completed", "failed"}:
                break
            time.sleep(0.05)
        implementation = orchestrator.get(implementation.id)
        assert implementation.status == "completed", implementation.error
        assert len(approved_ids) == 2
        assert implementation.evidence["completion_evidence"]["verified"] is True
        assert implementation.evidence["tests"][0]["payload"]["passed"] is True
        assert "answer.py" in implementation.evidence["git_diff"]["stdout"]
        assert (implementation.evidence["worktree_path"] and not (project / "answer.py").exists())

        review = orchestrator.delegate(
            workspace.id,
            role="reviewer",
            prompt="Independently review the implementation.",
            parent_run_id=parent.id,
            source_delegation_id=implementation.id,
            start=True,
        )
        for _ in range(100):
            review = orchestrator.get(review.id)
            if review.status in {"completed", "failed"}:
                break
            time.sleep(0.05)
        assert review.status == "completed", review.error
        assert review.evidence["review"]["verdict"] == "approved"

        readiness = orchestrator.integration_check(
            implementation.id,
            reviewer_delegation_id=review.id,
        )
        assert readiness["ready"] is True, readiness["reasons"]
        pending_integration = orchestrator.integrate(
            implementation.id,
            reviewer_delegation_id=review.id,
        )
        assert pending_integration["status"] == "approval_required"
        integration_approval = pending_integration["approval"]["id"]
        store.resolve_approval(
            integration_approval,
            "approved",
            resolved_by="test-user",
        )
        integrated = orchestrator.integrate(
            implementation.id,
            reviewer_delegation_id=review.id,
            approval_id=integration_approval,
            executor_id="test-user",
        )
        assert integrated["ok"] is True, integrated
        assert integrated["integration_performed"] is True
        assert (project / "answer.py").read_text(encoding="utf-8") == "ANSWER = 42\n"
        parent_kinds = [event.kind for event in store.list_events(parent.id)]
        assert "agent_integrated" in parent_kinds
        consumed = store.get_approval(integration_approval)
        assert consumed.status == "consumed"
        assert consumed.consumed_by == "test-user"
    finally:
        for item in orchestrator.list(workspace_id=workspace.id):
            if item.worktree_path:
                _git(project, "worktree", "remove", "--force", item.worktree_path)
        orchestrator.close()
