"""CLI client for the MTPLX local agent control plane."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _request(
    args: argparse.Namespace,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_url = str(
        args.base_url or os.environ.get("MTPLX_BASE_URL") or "http://127.0.0.1:8000"
    ).rstrip("/")
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    api_key = args.api_key or os.environ.get("MTPLX_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(f"{base_url}{path}", data=payload, headers=headers, method=method)
    try:
        with urlopen(request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"MTPLX agent API unavailable: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("MTPLX agent API returned a non-object response")
    return value


def cmd_agent(args: argparse.Namespace) -> int:
    try:
        if args.agent_action == "profiles":
            result = _request(args, "GET", "/v1/mtplx/agent/profiles")
        elif args.agent_action == "profile-add":
            result = _request(
                args,
                "POST",
                "/v1/mtplx/agent/profiles",
                {
                    "id": args.profile_id,
                    "name": args.name,
                    "description": args.description,
                    "permissions": _comma_values(args.permissions),
                    "instructions": args.instructions,
                    "token_budget": args.budget,
                    "context_window": args.context_window,
                    "model": args.model,
                },
            )
        elif args.agent_action == "profile-update":
            body = {
                key: value
                for key, value in {
                    "name": args.name,
                    "description": args.description,
                    "permissions": (
                        _comma_values(args.permissions)
                        if args.permissions is not None
                        else None
                    ),
                    "instructions": args.instructions,
                    "token_budget": args.budget,
                    "context_window": args.context_window,
                    "model": args.model,
                }.items()
                if value is not None
            }
            result = _request(
                args,
                "PATCH",
                f"/v1/mtplx/agent/profiles/{args.profile_id}",
                body,
            )
        elif args.agent_action == "tools":
            result = _request(args, "GET", "/v1/mtplx/agent/tools")
        elif args.agent_action == "workspace-list":
            result = _request(args, "GET", "/v1/mtplx/workspaces")
        elif args.agent_action == "workspace-add":
            result = _request(
                args,
                "POST",
                "/v1/mtplx/workspaces",
                {"name": args.name, "root_path": args.root},
            )
        elif args.agent_action == "run-list":
            result = _request(
                args,
                "GET",
                f"/v1/mtplx/workspaces/{args.workspace_id}/runs",
            )
        elif args.agent_action == "events":
            result = _request(args, "GET", f"/v1/mtplx/runs/{args.run_id}/events")
        elif args.agent_action == "resume":
            result = _request(args, "POST", f"/v1/mtplx/runs/{args.run_id}/resume")
        elif args.agent_action in {"review", "delegate"}:
            role = "reviewer" if args.agent_action == "review" else args.role
            result = _request(
                args,
                "POST",
                f"/v1/mtplx/workspaces/{args.workspace_id}/delegations",
                {
                    "role": role,
                    "prompt": args.prompt,
                    "parent_run_id": args.parent_run_id,
                    "model": args.model,
                    "budget": args.budget,
                    "context_window": args.context_window,
                    "source_delegation_id": args.source_delegation_id,
                    "start": True,
                },
            )
        elif args.agent_action == "delegation":
            result = _request(
                args,
                "GET",
                f"/v1/mtplx/delegations/{args.delegation_id}",
            )
        elif args.agent_action == "retry":
            result = _request(
                args,
                "POST",
                f"/v1/mtplx/delegations/{args.delegation_id}/retry",
            )
        elif args.agent_action == "worktree":
            result = _request(
                args,
                "GET",
                f"/v1/mtplx/delegations/{args.delegation_id}/worktree",
            )
        elif args.agent_action == "integration-check":
            result = _request(
                args,
                "GET",
                (
                    f"/v1/mtplx/delegations/{args.delegation_id}/integration"
                    f"?reviewer_delegation_id={args.reviewer_delegation_id}"
                ),
            )
        elif args.agent_action == "integrate":
            result = _request(
                args,
                "POST",
                f"/v1/mtplx/delegations/{args.delegation_id}/integrate",
                {
                    "reviewer_delegation_id": args.reviewer_delegation_id,
                    "approval_id": args.approval_id,
                    "executor_id": "cli",
                },
            )
        elif args.agent_action == "approvals":
            query = []
            if args.run_id:
                query.append(f"run_id={args.run_id}")
            if args.status:
                query.append(f"status={args.status}")
            suffix = f"?{'&'.join(query)}" if query else ""
            result = _request(
                args,
                "GET",
                f"/v1/mtplx/workspaces/{args.workspace_id}/approvals{suffix}",
            )
        elif args.agent_action in {"approve", "deny"}:
            result = _request(
                args,
                "POST",
                f"/v1/mtplx/approvals/{args.approval_id}",
                {
                    "decision": "approved" if args.agent_action == "approve" else "denied",
                    "resolved_by": "cli",
                    "reason": args.reason,
                },
            )
        elif args.agent_action == "tool":
            try:
                arguments = json.loads(args.arguments)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"--arguments must be a JSON object: {exc.msg}") from exc
            if not isinstance(arguments, dict):
                raise RuntimeError("--arguments must decode to a JSON object")
            result = _request(
                args,
                "POST",
                f"/v1/mtplx/workspaces/{args.workspace_id}/tools/{args.tool_name}",
                {
                    "run_id": args.run_id,
                    "arguments": arguments,
                    "approval_id": args.approval_id,
                    "executor_id": "cli",
                },
            )
        elif args.agent_action == "skills":
            result = _request(
                args,
                "GET",
                f"/v1/mtplx/workspaces/{args.workspace_id}/skills",
            )
        else:
            raise RuntimeError(f"unknown agent action: {args.agent_action}")
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _comma_values(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _json_object(value: str, *, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} must be a JSON object: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{label} must decode to a JSON object")
    return decoded


def _json_file(path: str) -> dict[str, Any]:
    source = Path(path).expanduser()
    try:
        return _json_object(source.read_text(encoding="utf-8"), label=str(source))
    except OSError as exc:
        raise RuntimeError(f"cannot read Graph definition {source}: {exc}") from exc


def cmd_graph(args: argparse.Namespace) -> int:
    try:
        action = args.graph_action
        if action == "list":
            query = urlencode(
                {
                    key: value
                    for key, value in {
                        "workspace_id": args.workspace_id,
                        "limit": args.limit,
                    }.items()
                    if value is not None
                }
            )
            result = _request(args, "GET", f"/v1/mtplx/graphs?{query}")
        elif action in {"create", "validate"}:
            payload = _json_file(args.file)
            path = "/v1/mtplx/graphs" + ("/validate" if action == "validate" else "")
            result = _request(args, "POST", path, payload)
        elif action == "update":
            query = (
                f"?{urlencode({'expected_revision': args.expected_revision})}"
                if args.expected_revision is not None
                else ""
            )
            result = _request(
                args,
                "PATCH",
                f"/v1/mtplx/graphs/{args.graph_id}{query}",
                _json_file(args.file),
            )
        elif action == "show":
            query = (
                f"?{urlencode({'revision': args.revision})}"
                if args.revision is not None
                else ""
            )
            result = _request(args, "GET", f"/v1/mtplx/graphs/{args.graph_id}{query}")
        elif action == "revisions":
            result = _request(
                args,
                "GET",
                f"/v1/mtplx/graphs/{args.graph_id}/revisions?limit={args.limit}",
            )
        elif action == "run":
            result = _request(
                args,
                "POST",
                "/v1/mtplx/graph-runs",
                {
                    "graph_id": args.graph_id,
                    "revision": args.revision,
                    "inputs": _json_object(args.inputs, label="--inputs"),
                    "model": args.model,
                    "runtime_profile": args.runtime_profile,
                    "run_id": args.run_id,
                    "start": not args.no_start,
                },
            )
        elif action == "runs":
            query = urlencode(
                {
                    key: value
                    for key, value in {
                        "graph_id": args.graph_id,
                        "workspace_id": args.workspace_id,
                        "limit": args.limit,
                    }.items()
                    if value is not None
                }
            )
            result = _request(args, "GET", f"/v1/mtplx/graph-runs?{query}")
        elif action in {"status", "pause", "resume", "cancel"}:
            method = "GET" if action == "status" else "POST"
            suffix = "" if action == "status" else f"/{action}"
            result = _request(
                args,
                method,
                f"/v1/mtplx/graph-runs/{args.run_id}{suffix}",
            )
        elif action == "retry":
            result = _request(
                args,
                "POST",
                f"/v1/mtplx/graph-runs/{args.run_id}/retry",
                {
                    "node_id": args.node_id,
                    "allow_side_effect_retry": args.allow_side_effect_retry,
                    "force_new_side_effect": args.force_new_side_effect,
                },
            )
        elif action == "approve":
            result = _request(
                args,
                "POST",
                f"/v1/mtplx/graph-runs/{args.run_id}/approve",
                {
                    "approval_id": args.approval_id,
                    "decision": args.decision,
                    "resolved_by": "cli",
                    "reason": args.reason,
                    "resume": not args.no_resume,
                },
            )
        elif action == "logs":
            query = urlencode({"after": args.after, "limit": args.limit})
            result = _request(
                args,
                "GET",
                f"/v1/mtplx/graph-runs/{args.run_id}/logs?{query}",
            )
        elif action == "approvals":
            query = urlencode(
                {
                    key: value
                    for key, value in {
                        "status": args.status,
                        "limit": args.limit,
                    }.items()
                    if value is not None
                }
            )
            result = _request(
                args,
                "GET",
                f"/v1/mtplx/graph-runs/{args.run_id}/approvals?{query}",
            )
        else:
            raise RuntimeError(f"unknown Graph action: {action}")
    except RuntimeError as exc:
        print(f"error: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def add_agent_parser(sub: Any) -> None:
    agent = sub.add_parser(
        "agent",
        help="Control local agent workspaces, runs, and delegation",
    )
    agent.add_argument(
        "--base-url",
        default=None,
        help="MTPLX server URL, defaulting to MTPLX_BASE_URL or localhost:8000",
    )
    agent.add_argument("--api-key", default=None, help="Bearer key or MTPLX_API_KEY")
    actions = agent.add_subparsers(dest="agent_action", required=True)

    actions.add_parser("profiles", help="List available agent roles").set_defaults(func=cmd_agent)
    profile_add = actions.add_parser("profile-add", help="Create a user-defined agent role")
    profile_add.add_argument("profile_id")
    profile_add.add_argument("name")
    profile_add.add_argument("--description", default="")
    profile_add.add_argument("--permissions", default="read,search")
    profile_add.add_argument("--instructions", default="")
    profile_add.add_argument("--budget", type=int, default=2400)
    profile_add.add_argument("--context-window", type=int, default=65536)
    profile_add.add_argument("--model", default=None)
    profile_add.set_defaults(func=cmd_agent)

    profile_update = actions.add_parser("profile-update", help="Update a user-defined role")
    profile_update.add_argument("profile_id")
    profile_update.add_argument("--name", default=None)
    profile_update.add_argument("--description", default=None)
    profile_update.add_argument("--permissions", default=None)
    profile_update.add_argument("--instructions", default=None)
    profile_update.add_argument("--budget", type=int, default=None)
    profile_update.add_argument("--context-window", type=int, default=None)
    profile_update.add_argument("--model", default=None)
    profile_update.set_defaults(func=cmd_agent)

    actions.add_parser("tools", help="List first-party coding tools").set_defaults(func=cmd_agent)
    actions.add_parser("workspace-list", help="List local workspaces").set_defaults(func=cmd_agent)

    add_workspace = actions.add_parser("workspace-add", help="Register a local workspace")
    add_workspace.add_argument("name")
    add_workspace.add_argument("root")
    add_workspace.set_defaults(func=cmd_agent)

    run_list = actions.add_parser("run-list", help="List durable runs for a workspace")
    run_list.add_argument("workspace_id")
    run_list.set_defaults(func=cmd_agent)

    events = actions.add_parser("events", help="Read durable events for a run")
    events.add_argument("run_id")
    events.set_defaults(func=cmd_agent)

    resume = actions.add_parser("resume", help="Resume a restart-paused run")
    resume.add_argument("run_id")
    resume.set_defaults(func=cmd_agent)

    review = actions.add_parser("review", help="Delegate a read-only reviewer")
    review.add_argument("workspace_id")
    review.add_argument("--prompt", default="")
    review.add_argument("--parent-run-id", default=None)
    review.add_argument("--model", default=None)
    review.add_argument("--budget", type=int, default=None, help="Maximum delegated output tokens")
    review.add_argument("--context-window", type=int, default=None)
    review.add_argument("--source-delegation-id", default=None)
    review.set_defaults(func=cmd_agent)

    delegate = actions.add_parser("delegate", help="Start any built-in or user-defined agent")
    delegate.add_argument("workspace_id")
    delegate.add_argument("role")
    delegate.add_argument("--prompt", default="")
    delegate.add_argument("--parent-run-id", default=None)
    delegate.add_argument("--source-delegation-id", default=None)
    delegate.add_argument("--model", default=None)
    delegate.add_argument("--budget", type=int, default=None)
    delegate.add_argument("--context-window", type=int, default=None)
    delegate.set_defaults(func=cmd_agent)

    delegation = actions.add_parser("delegation", help="Read delegated agent evidence")
    delegation.add_argument("delegation_id")
    delegation.set_defaults(func=cmd_agent)

    retry = actions.add_parser("retry", help="Retry a failed or cancelled delegation")
    retry.add_argument("delegation_id")
    retry.set_defaults(func=cmd_agent)

    worktree = actions.add_parser("worktree", help="Check delegated worktree evidence")
    worktree.add_argument("delegation_id")
    worktree.set_defaults(func=cmd_agent)

    integration_check = actions.add_parser(
        "integration-check",
        help="Evaluate tests, review, conflict, and policy gates",
    )
    integration_check.add_argument("delegation_id")
    integration_check.add_argument("reviewer_delegation_id")
    integration_check.set_defaults(func=cmd_agent)

    integrate = actions.add_parser(
        "integrate",
        help="Apply a reviewed child patch after all gates pass",
    )
    integrate.add_argument("delegation_id")
    integrate.add_argument("reviewer_delegation_id")
    integrate.add_argument("--approval-id", default=None)
    integrate.set_defaults(func=cmd_agent)

    approvals = actions.add_parser("approvals", help="List workspace approvals")
    approvals.add_argument("workspace_id")
    approvals.add_argument("--run-id", default=None)
    approvals.add_argument("--status", default=None)
    approvals.set_defaults(func=cmd_agent)

    approve = actions.add_parser("approve", help="Approve one exact pending action")
    approve.add_argument("approval_id")
    approve.add_argument("--reason", default=None)
    approve.set_defaults(func=cmd_agent)

    deny = actions.add_parser("deny", help="Deny one exact pending action")
    deny.add_argument("approval_id")
    deny.add_argument("--reason", default=None)
    deny.set_defaults(func=cmd_agent)

    tool = actions.add_parser("tool", help="Execute one first-party workspace tool")
    tool.add_argument("workspace_id")
    tool.add_argument("tool_name")
    tool.add_argument("--arguments", default="{}")
    tool.add_argument("--run-id", default=None)
    tool.add_argument("--approval-id", default=None)
    tool.set_defaults(func=cmd_agent)

    skills = actions.add_parser("skills", help="List workspace-local skills")
    skills.add_argument("workspace_id")
    skills.set_defaults(func=cmd_agent)


def add_graph_parser(sub: Any) -> None:
    graph = sub.add_parser(
        "graph",
        help="Create and control durable Graph workflows with bounded Loop nodes",
    )
    graph.add_argument(
        "--base-url",
        default=None,
        help="MTPLX server URL, defaulting to MTPLX_BASE_URL or localhost:8000",
    )
    graph.add_argument("--api-key", default=None, help="Bearer key or MTPLX_API_KEY")
    actions = graph.add_subparsers(dest="graph_action", required=True)

    list_parser = actions.add_parser("list", help="List Graph definitions")
    list_parser.add_argument("--workspace-id", default=None)
    list_parser.add_argument("--limit", type=int, default=100)
    list_parser.set_defaults(func=cmd_graph)

    for action in ("create", "validate"):
        parser = actions.add_parser(action, help=f"{action.title()} a Graph JSON definition")
        parser.add_argument("file")
        parser.set_defaults(func=cmd_graph)

    update = actions.add_parser("update", help="Create an immutable Graph revision")
    update.add_argument("graph_id")
    update.add_argument("file")
    update.add_argument("--expected-revision", type=int, default=None)
    update.set_defaults(func=cmd_graph)

    show = actions.add_parser("show", help="Show one Graph revision")
    show.add_argument("graph_id")
    show.add_argument("--revision", type=int, default=None)
    show.set_defaults(func=cmd_graph)

    revisions = actions.add_parser("revisions", help="List immutable Graph revisions")
    revisions.add_argument("graph_id")
    revisions.add_argument("--limit", type=int, default=100)
    revisions.set_defaults(func=cmd_graph)

    run = actions.add_parser("run", help="Start one pinned Graph revision")
    run.add_argument("graph_id")
    run.add_argument("--revision", type=int, default=None)
    run.add_argument("--inputs", default="{}")
    run.add_argument("--model", default=None)
    run.add_argument("--runtime-profile", default="auto")
    run.add_argument("--run-id", default=None)
    run.add_argument("--no-start", action="store_true")
    run.set_defaults(func=cmd_graph)

    runs = actions.add_parser("runs", help="List durable Graph runs")
    runs.add_argument("--graph-id", default=None)
    runs.add_argument("--workspace-id", default=None)
    runs.add_argument("--limit", type=int, default=100)
    runs.set_defaults(func=cmd_graph)

    for action in ("status", "pause", "resume", "cancel"):
        parser = actions.add_parser(action, help=f"{action.title()} a Graph run")
        parser.add_argument("run_id")
        parser.set_defaults(func=cmd_graph)

    retry = actions.add_parser("retry", help="Retry one failed Graph node")
    retry.add_argument("run_id")
    retry.add_argument("--node-id", default=None)
    retry.add_argument("--allow-side-effect-retry", action="store_true")
    retry.add_argument("--force-new-side-effect", action="store_true")
    retry.set_defaults(func=cmd_graph)

    approve = actions.add_parser("approve", help="Resolve an exact Graph approval")
    approve.add_argument("run_id")
    approve.add_argument("approval_id")
    approve.add_argument("--decision", choices=("approved", "denied"), default="approved")
    approve.add_argument("--reason", default=None)
    approve.add_argument("--no-resume", action="store_true")
    approve.set_defaults(func=cmd_graph)

    logs = actions.add_parser("logs", help="Read the durable Graph event log")
    logs.add_argument("run_id")
    logs.add_argument("--after", type=int, default=0)
    logs.add_argument("--limit", type=int, default=500)
    logs.set_defaults(func=cmd_graph)

    approvals = actions.add_parser(
        "approvals",
        help="List approvals bound to one Graph run",
    )
    approvals.add_argument("run_id")
    approvals.add_argument(
        "--status",
        choices=("pending", "approved", "denied", "expired", "consumed"),
        default=None,
    )
    approvals.add_argument("--limit", type=int, default=100)
    approvals.set_defaults(func=cmd_graph)


__all__ = ["add_agent_parser", "add_graph_parser", "cmd_agent", "cmd_graph"]
