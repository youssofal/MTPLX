from __future__ import annotations

import json
import subprocess

import pytest
from fastapi.testclient import TestClient

from mtplx.dreaming import DreamingService
from mtplx.agent_workspace import WorkspaceConflictError, WorkspaceStore
from mtplx.memory import (
    MemoryConflictError,
    MemoryPermissionError,
    MemoryPrincipal,
    MemoryStore,
)
from mtplx.server.openai import create_app


def test_memory_writes_are_versioned_and_hash_guarded(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.initialize()

    first = store.write("shared/deploy.md", "Deploy via make ship.", author="sess-a")
    second = store.write(
        "shared/deploy.md",
        "Deploy via make ship, not ./deploy.sh.",
        expected_sha256=first.content_sha256,
        author="sess-b",
    )

    assert second.version == 2
    assert store.read("shared/deploy.md").content == second.content
    assert list((tmp_path / "memory" / ".history").rglob("*.json"))
    history = store.history("shared/deploy.md")
    assert [item["version"] for item in history["versions"]] == [1]
    restored = store.restore("shared/deploy.md", 1, author="rollback")
    assert restored.version == 3
    assert restored.content == "Deploy via make ship."
    with pytest.raises(MemoryConflictError):
        store.write(
            "shared/deploy.md",
            "stale write",
            expected_sha256=second.content_sha256,
        )


def test_memory_progressive_search_and_scoped_permissions(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    store.write("shared/style.md", "Use concise release notes.")
    store.write("working/agent-a/notes.md", "Agent A is fixing the release path.")
    store.write("working/agent-b/notes.md", "Agent B owns the benchmark path.")

    agent_a = MemoryPrincipal(agent_id="agent-a")
    assert store.search("release", principal=agent_a)[0].document.path == "shared/style.md"
    assert store.read("working/agent-a/notes.md", principal=agent_a).content.startswith("Agent A")
    with pytest.raises(MemoryPermissionError):
        store.read("working/agent-b/notes.md", principal=agent_a)
    with pytest.raises(MemoryPermissionError):
        store.write("shared/new.md", "not allowed", principal=agent_a)


def test_dreaming_proposes_explicit_markers_and_applies_with_snapshot(tmp_path):
    store = MemoryStore(tmp_path / "memory")
    principal = MemoryPrincipal(agent_id="agent-a", session_id="session-1")
    store.record_transcript(
        "session-1",
        [{"role": "user", "content": "Remember: deploy with make ship."}],
        agent_id="agent-a",
        principal=principal,
    )
    service = DreamingService(store)
    try:
        result = service.run()
        assert result["status"] == "completed"
        assert result["proposals"][0]["path"] == "working/agent-a/dreaming-candidates.md"
        applied = service.apply(result["run_id"])
        assert applied["status"] == "applied"
        assert service.get(result["run_id"])["status"] == "applied"
        assert "deploy with make ship." in store.read(
            "working/agent-a/dreaming-candidates.md"
        ).content
    finally:
        service.close()


def test_memory_http_surface_uses_same_store_and_agent_scope(tmp_path):
    from test_server_openai import _fake_state

    state = _fake_state()
    state.memory_store = MemoryStore(tmp_path / "memory")
    state.memory_store.initialize()
    state.dreaming_service = DreamingService(state.memory_store)
    try:
        with TestClient(create_app(state)) as client:
            written = client.put(
                "/v1/mtplx/memory/working/agent-a/preferences.md",
                headers={"X-MTPLX-Agent-Id": "agent-a"},
                json={
                    "content": "Prefer compact diffs.",
                    "agent_id": "agent-a",
                    "author": "agent-a",
                },
            )
            assert written.status_code == 200
            fetched = client.get(
                "/v1/mtplx/memory/working/agent-a/preferences.md",
                headers={"X-MTPLX-Agent-Id": "agent-a"},
            )
            assert fetched.status_code == 200
            assert fetched.json()["content"] == "Prefer compact diffs."
            appended = client.post(
                "/v1/mtplx/memory/append",
                params={
                    "path": "working/agent-a/preferences.md",
                },
                headers={"X-MTPLX-Agent-Id": "agent-a"},
                json={
                    "content": "\nPrefer explicit conflicts.",
                    "expected_sha256": fetched.json()["content_sha256"],
                    "agent_id": "agent-a",
                },
            )
            assert appended.status_code == 200
            assert "explicit conflicts" in appended.json()["content"]
            history = client.get(
                "/v1/mtplx/memory/history/working/agent-a/preferences.md",
                headers={"X-MTPLX-Agent-Id": "agent-a"},
            )
            assert history.status_code == 200
            assert history.json()["versions"][0]["version"] == 1
            restored = client.post(
                "/v1/mtplx/memory/history/working/agent-a/preferences.md/restore",
                headers={"X-MTPLX-Agent-Id": "agent-a"},
                json={"version": 1, "agent_id": "agent-a"},
            )
            assert restored.status_code == 200
            assert restored.json()["content"] == "Prefer compact diffs."
            denied = client.put(
                "/v1/mtplx/memory/shared/unsafe.md",
                headers={"X-MTPLX-Agent-Id": "agent-a"},
                json={"content": "must not write shared memory", "agent_id": "agent-a"},
            )
            assert denied.status_code == 403
            transcript = client.post(
                "/v1/mtplx/memory/transcripts",
                json={
                    "session_id": "session-2",
                    "agent_id": "agent-a",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert transcript.status_code == 200
            listed = client.get(
                "/v1/mtplx/memory",
                params={"agent_id": "agent-a", "scope": "working"},
            )
            assert listed.status_code == 200
            assert listed.json()["documents"][0]["path"].startswith("working/agent-a/")
    finally:
        state.dreaming_service.close()


def test_workspace_store_persists_runs_events_and_approval_decisions(tmp_path):
    store = WorkspaceStore(tmp_path / "workspaces")
    project = tmp_path / "project"
    project.mkdir()
    workspace = store.create_workspace(
        "MTPLX",
        str(project),
        model="local-model",
    )
    run = store.create_run(workspace.id, title="Implement agent surface")
    event = store.append_event(run.id, "user_message", {"content": "Build the shell"})
    assert event.sequence == 2
    assert store.get_run(run.id).event_count == 2
    approval = store.create_approval(
        workspace.id,
        run_id=run.id,
        tool="terminal",
        action="run command",
        description="Run the focused test suite",
        target="pytest -q tests/test_memory.py",
    )
    assert approval.status == "pending"
    resolved = store.resolve_approval(approval.id, "approved", resolved_by="pjb")
    assert resolved.status == "approved"
    assert [item.kind for item in store.list_events(run.id)] == [
        "run_created",
        "user_message",
        "approval_requested",
        "approval_resolved",
    ]
    assert store.list_approvals(workspace_id=workspace.id, status="approved")[0].id == approval.id
    with pytest.raises(WorkspaceConflictError):
        store.resolve_approval(approval.id, "denied")


def test_workspace_restart_recovery_and_expiring_approvals_are_durable(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = WorkspaceStore(tmp_path / "workspaces")
    workspace = store.create_workspace("MTPLX", str(project))
    run = store.create_run(workspace.id)
    recovered = store.recover_incomplete_runs()
    assert recovered[0].status == "paused"
    assert store.get_run(run.id).status == "paused"
    resumed = store.resume_run(run.id)
    assert resumed.status == "queued"
    assert [event.kind for event in store.list_events(run.id)] == [
        "run_created",
        "run_paused",
        "run_resumed",
    ]
    approval = store.create_approval(
        workspace.id,
        run_id=run.id,
        tool="terminal",
        action="run command",
        description="short lived approval",
    )
    raw = approval.to_dict()
    raw["expires_at"] = "2000-01-01T00:00:00Z"
    store._approval_path(approval.id).write_text(
        json.dumps(raw), encoding="utf-8"
    )
    assert store.get_approval(approval.id).status == "expired"
    assert store.get_approval(approval.id).resolved_by == "system"


def test_approval_is_bound_to_exact_arguments_and_consumed_once(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = WorkspaceStore(tmp_path / "workspaces")
    workspace = store.create_workspace("MTPLX", str(project))
    run = store.create_run(workspace.id)
    arguments = {"path": "result.txt", "content": "verified\n"}
    approval = store.create_approval(
        workspace.id,
        run_id=run.id,
        tool="write_file",
        action="Write result.txt",
        description="Write 9 bytes to result.txt",
        target="result.txt",
        arguments=arguments,
    )
    store.resolve_approval(approval.id, "approved", resolved_by="tester")

    with pytest.raises(WorkspaceConflictError, match="arguments do not match"):
        store.consume_approval(
            approval.id,
            workspace_id=workspace.id,
            run_id=run.id,
            tool="write_file",
            arguments={**arguments, "content": "different\n"},
        )

    consumed = store.consume_approval(
        approval.id,
        workspace_id=workspace.id,
        run_id=run.id,
        tool="write_file",
        arguments=arguments,
    )
    assert consumed.status == "consumed"
    assert consumed.arguments == arguments
    assert consumed.arguments_sha256
    with pytest.raises(WorkspaceConflictError, match="not executable"):
        store.consume_approval(
            approval.id,
            workspace_id=workspace.id,
            run_id=run.id,
            tool="write_file",
            arguments=arguments,
        )
    assert [event.kind for event in store.list_events(run.id)][-1] == "approval_consumed"


def test_approved_authorization_cannot_be_consumed_after_its_deadline(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = WorkspaceStore(tmp_path / "workspaces")
    workspace = store.create_workspace("MTPLX", str(project))
    run = store.create_run(workspace.id)
    arguments = {"path": "result.txt", "content": "verified\n"}
    approval = store.create_approval(
        workspace.id,
        run_id=run.id,
        tool="write_file",
        action="Write result.txt",
        description="Write the verified result",
        arguments=arguments,
    )
    approved = store.resolve_approval(
        approval.id,
        "approved",
        resolved_by="tester",
    )
    raw = approved.to_dict()
    raw["expires_at"] = "2000-01-01T00:00:00Z"
    store._approval_path(approval.id).write_text(
        json.dumps(raw), encoding="utf-8"
    )

    with pytest.raises(WorkspaceConflictError, match="approval is expired"):
        store.consume_approval(
            approval.id,
            workspace_id=workspace.id,
            run_id=run.id,
            tool="write_file",
            arguments=arguments,
        )

    expired = store.get_approval(approval.id)
    assert expired.status == "expired"
    assert expired.resolved_by == "system"
    assert expired.reason == "approval expired before execution"
    assert [event.kind for event in store.list_events(run.id)][-1] == "approval_resolved"


def test_workspace_http_surface_exposes_local_agent_state(tmp_path):
    from test_server_openai import _fake_state

    state = _fake_state()
    state.workspace_store = WorkspaceStore(tmp_path / "workspaces")
    state.memory_store = MemoryStore(tmp_path / "memory")
    state.memory_store.initialize()
    state.dreaming_service = DreamingService(state.memory_store)
    try:
        with TestClient(create_app(state)) as client:
            project = tmp_path / "project"
            project.mkdir()
            created = client.post(
                "/v1/mtplx/workspaces",
                json={
                    "name": "MTPLX",
                    "root_path": str(project),
                },
            )
            assert created.status_code == 200
            workspace = created.json()
            assert workspace["model"] == state.model_id
            models = client.get("/v1/mtplx/agent/models")
            assert models.status_code == 200
            assert models.json()["provider"]["id"] == "mtplx"
            run_response = client.post(
                f"/v1/mtplx/workspaces/{workspace['id']}/runs",
                json={"title": "first turn"},
            )
            assert run_response.status_code == 200
            run = run_response.json()
            event_response = client.post(
                f"/v1/mtplx/runs/{run['id']}/events",
                json={"kind": "assistant_message", "payload": {"content": "ok"}},
            )
            assert event_response.status_code == 200
            assert event_response.json()["sequence"] == 2
            tool_catalog = client.get("/v1/mtplx/agent/tools")
            assert tool_catalog.status_code == 200
            assert tool_catalog.json()["names"] == [
                "list_files",
                "read_file",
                "search_files",
                "inspect_repo",
                "git_status",
                "git_diff",
                "write_file",
                "apply_patch",
                "run_tests",
                "run_command",
            ]
            tool_arguments = {"path": "evidence.txt", "content": "recorded\n"}
            tool_pending = client.post(
                f"/v1/mtplx/workspaces/{workspace['id']}/tools/write_file",
                json={"run_id": run["id"], "arguments": tool_arguments},
            )
            assert tool_pending.status_code == 200
            assert tool_pending.json()["status"] == "approval_required"
            tool_approval = tool_pending.json()["approval"]
            approved = client.post(
                f"/v1/mtplx/approvals/{tool_approval['id']}",
                json={"decision": "approved", "reason": "exact arguments reviewed"},
            )
            assert approved.status_code == 200
            tool_completed = client.post(
                f"/v1/mtplx/workspaces/{workspace['id']}/tools/write_file",
                json={
                    "run_id": run["id"],
                    "arguments": tool_arguments,
                    "approval_id": tool_approval["id"],
                },
            )
            assert tool_completed.status_code == 200
            assert tool_completed.json()["ok"] is True
            assert (project / "evidence.txt").read_text(encoding="utf-8") == "recorded\n"
            approval_response = client.post(
                f"/v1/mtplx/workspaces/{workspace['id']}/approvals",
                json={
                    "run_id": run["id"],
                    "tool": "write",
                    "action": "edit file",
                    "description": "Update README",
                    "target": "README.md",
                },
            )
            assert approval_response.status_code == 200
            approval = approval_response.json()
            resolved = client.post(
                f"/v1/mtplx/approvals/{approval['id']}",
                json={"decision": "denied", "reason": "review first"},
            )
            assert resolved.status_code == 200
            assert resolved.json()["status"] == "denied"
            context = client.get(
                "/v1/mtplx/memory/context",
                params={"query": "MTPLX"},
            )
            assert context.status_code == 200
            capabilities = client.get("/v1/mtplx/app/capabilities").json()
            assert capabilities["features"]["agent_workspaces"] is True
            assert "workspace_runs" in capabilities["endpoints"]
    finally:
        state.dreaming_service.close()


def test_agent_http_surface_exposes_profiles_skills_and_worktree_evidence(tmp_path):
    from test_server_openai import _fake_state

    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("# MTPLX\n", encoding="utf-8")
    skill = project / ".mtplx" / "skills" / "reviewer"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Review\nInspect the diff.\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(project), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(project), "add", "README.md", ".mtplx"], check=True)
    subprocess.run(
        [
            "git", "-C", str(project),
            "-c", "user.name=MTPLX Test",
            "-c", "user.email=test@example.invalid",
            "commit", "-qm", "initial",
        ],
        check=True,
    )
    state = _fake_state()
    state.workspace_store = WorkspaceStore(tmp_path / "workspaces")
    state.memory_store = MemoryStore(tmp_path / "memory")
    state.memory_store.initialize()
    state.dreaming_service = DreamingService(state.memory_store)
    delegation = None
    try:
        with TestClient(create_app(state)) as client:
            created = client.post(
                "/v1/mtplx/workspaces",
                json={"name": "MTPLX", "root_path": str(project)},
            )
            assert created.status_code == 200
            workspace_id = created.json()["id"]
            assert client.get("/v1/mtplx/agent/profiles").json()["profiles"]
            profile = client.post(
                "/v1/mtplx/agent/profiles",
                json={
                    "id": "security-auditor",
                    "name": "Security auditor",
                    "permissions": ["read", "search"],
                    "instructions": "Inspect trust boundaries.",
                    "token_budget": 2048,
                    "context_window": 32768,
                },
            )
            assert profile.status_code == 200
            assert profile.json()["built_in"] is False
            updated_profile = client.patch(
                "/v1/mtplx/agent/profiles/security-auditor",
                json={"token_budget": 3072},
            )
            assert updated_profile.status_code == 200
            assert updated_profile.json()["token_budget"] == 3072
            skills = client.get(
                f"/v1/mtplx/workspaces/{workspace_id}/skills"
            ).json()["skills"]
            assert skills[0]["name"] == "reviewer"
            delegation_response = client.post(
                f"/v1/mtplx/workspaces/{workspace_id}/delegations",
                json={
                    "role": "security-auditor",
                    "context_window": 49152,
                    "start": False,
                },
            )
            assert delegation_response.status_code == 200
            delegation = delegation_response.json()
            assert delegation["role"] == "security-auditor"
            assert delegation["context_window"] == 49152
            assert delegation["profile_sha256"] == updated_profile.json()["sha256"]
            evidence = client.get(
                f"/v1/mtplx/delegations/{delegation['id']}/worktree"
            )
            assert evidence.status_code == 200
            assert evidence.json()["merge_check"]["mergeable"] is True
    finally:
        if delegation and delegation.get("worktree_path"):
            subprocess.run(
                [
                    "git", "-C", str(project), "worktree", "remove", "--force",
                    delegation["worktree_path"],
                ],
                check=False,
                capture_output=True,
            )
        state.dreaming_service.close()
