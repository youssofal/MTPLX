"""Durable Graph executor backed by the MTPLX runtime.

The executor never loads a model. Model nodes call the already-running MTPLX
OpenAI-compatible endpoint, so routing, scheduler admission, session-bank use,
context policy, memory pressure, and thermal controls stay owned by MTPLX.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .agent_workspace import (
    WorkspaceConflictError,
    WorkspaceStoreError,
    safe_id,
    utc_now,
)
from .graphs import (
    SIDE_EFFECT_NODE_TYPES,
    GraphDefinition,
    GraphError,
    GraphNode,
    GraphRun,
    GraphStore,
    GraphValidationError,
    validate_graph_contract_value,
)
from .memory import (
    MemoryConflictError,
    MemoryNotFoundError,
    MemoryPermissionError,
    MemoryPrincipal,
    MemoryStore,
    content_sha256,
)
from .workspace_tools import (
    MUTATING_TOOLS,
    TOOL_POLICY_KEY,
    WorkspaceToolService,
)


_TEMPLATE = re.compile(r"\{\{\s*([A-Za-z0-9_.-]+)\s*\}\}")
_TERMINAL_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})


@dataclass(frozen=True)
class NodeOutcome:
    status: str
    output: Any = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    approval_id: str | None = None
    retryable: bool = False
    state_updates: dict[str, Any] = field(default_factory=dict)


class GraphExecutionError(GraphError):
    pass


ModelRunner = Callable[..., Mapping[str, Any]]
ResourceSnapshot = Callable[[], Mapping[str, Any]]


class GraphExecutor:
    """Runs one pinned Graph revision with durable per-node checkpoints."""

    def __init__(
        self,
        graph_store: GraphStore,
        tool_service: WorkspaceToolService,
        memory_store: MemoryStore,
        *,
        model_runner: ModelRunner | None = None,
        resource_snapshot: ResourceSnapshot | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_workers: int = 4,
    ) -> None:
        self.graph_store = graph_store
        self.workspace_store = graph_store.workspace_store
        self.tool_service = tool_service
        self.memory_store = memory_store
        self.base_url = str(
            base_url
            or os.environ.get("MTPLX_BASE_URL")
            or "http://127.0.0.1:8000"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("MTPLX_API_KEY")
        self._uses_http_model_runner = model_runner is None
        self.model_runner = model_runner or self._http_model_runner
        self.resource_snapshot = resource_snapshot or (
            lambda: {
                "provider": "mtplx",
                "runtime_loaded": True,
                "backend_id": "mtplx",
                "runtime_capabilities": [],
                "memory_pressure_level": 0,
                "thermal_throttled": False,
                "active_memory_bytes": 0,
                "context_window": 1_048_576,
                "runtime_profile": "auto",
            }
        )
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, min(int(max_workers), 16)),
            thread_name_prefix="mtplx-graph",
        )
        self._node_executor = ThreadPoolExecutor(
            max_workers=max(2, min(int(max_workers), 16)),
            thread_name_prefix="mtplx-graph-node",
        )
        self._model_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mtplx-graph-model",
        )
        self._model_lease = threading.Semaphore(1)
        self._run_locks: dict[str, threading.Lock] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._run_locks_guard = threading.RLock()
        self.recover_incomplete_runs()

    def close(self) -> None:
        with self._run_locks_guard:
            for event in self._cancel_events.values():
                event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._node_executor.shutdown(wait=False, cancel_futures=True)
        self._model_executor.shutdown(wait=False, cancel_futures=True)

    def _run_lock(self, run_id: str) -> threading.Lock:
        with self._run_locks_guard:
            return self._run_locks.setdefault(run_id, threading.Lock())

    def _cancel_event(self, run_id: str) -> threading.Event:
        with self._run_locks_guard:
            return self._cancel_events.setdefault(run_id, threading.Event())

    def start(
        self,
        graph_id: str,
        *,
        revision: int | None = None,
        inputs: Mapping[str, Any] | None = None,
        model: str | None = None,
        runtime_profile: str = "auto",
        run_id: str | None = None,
        start: bool = True,
    ) -> GraphRun:
        graph = self.graph_store.get(graph_id, revision=revision)
        run = self.graph_store.create_run(
            graph,
            inputs=inputs,
            model=model,
            runtime_profile=runtime_profile,
            run_id=run_id,
        )
        self._cancel_event(run.id).clear()
        if start:
            self._executor.submit(self._run, run.id)
        return run

    def pause(self, run_id: str) -> GraphRun:
        run = self.graph_store.get_run(run_id)
        if run.status in _TERMINAL_RUN_STATUSES:
            return run
        if run.resource_metrics.get("completion_prepared"):
            return run
        immediate = run.status in {"queued", "waiting_approval", "paused"}
        updated = self.graph_store.update_run(
            run.id,
            status="paused" if immediate else run.status,
            resource_metrics=(
                self._suspend_active_metrics(run)
                if immediate
                else run.resource_metrics
            ),
            pause_requested=True,
        )
        self.workspace_store.append_event(
            run.id,
            "graph_pause_requested",
            {"current_node_id": run.current_node_id},
        )
        return updated

    def resume(self, run_id: str) -> GraphRun:
        run = self.graph_store.get_run(run_id)
        if run.status in _TERMINAL_RUN_STATUSES:
            raise GraphExecutionError(f"cannot resume graph run in {run.status} state")
        current_state = run.node_states.get(run.current_node_id or "", {})
        if current_state.get("status") == "failed":
            raise GraphExecutionError(
                "the current Graph node is failed; use retry_failed_node so the "
                "side-effect and idempotency policy is explicit"
            )
        if run.pending_approval_id:
            approval = self.workspace_store.get_approval(run.pending_approval_id)
            if approval.status == "pending":
                return run
            if approval.status != "approved":
                return self._fail_run(
                    run,
                    f"pending approval {approval.status}: {approval.id}",
                )
        updated = self.graph_store.update_run(
            run.id,
            status="queued",
            pause_requested=False,
            error=None,
        )
        self._cancel_event(run.id).clear()
        self.workspace_store.append_event(
            run.id,
            "graph_resumed",
            {
                "current_node_id": run.current_node_id,
                "successful_nodes_preserved": [
                    node_id
                    for node_id, state in run.node_states.items()
                    if state.get("status") == "completed"
                ],
            },
        )
        self._executor.submit(self._run, run.id)
        return updated

    def cancel(self, run_id: str) -> GraphRun:
        run = self.graph_store.get_run(run_id)
        if run.status in _TERMINAL_RUN_STATUSES:
            return run
        if run.resource_metrics.get("completion_prepared"):
            return run
        self._cancel_event(run.id).set()
        cancelled_approval_id = run.pending_approval_id
        if cancelled_approval_id:
            try:
                approval = self.workspace_store.get_approval(cancelled_approval_id)
                if approval.status == "pending":
                    self.workspace_store.resolve_approval(
                        cancelled_approval_id,
                        "denied",
                        resolved_by="system",
                        reason="Graph run cancelled",
                    )
            except WorkspaceStoreError:
                pass
        states = {key: dict(value) for key, value in run.node_states.items()}
        if run.graph_revision and run.current_node_id and run.current_node_id in states:
            state = states[run.current_node_id]
            if state.get("status") not in {"completed", "failed"}:
                state["status"] = "cancelled"
                state["completed_at"] = utc_now()
                state["pending_approval_id"] = None
        try:
            graph = self._pinned_graph(run)
        except GraphError:
            graph = None
        if graph is not None and graph.schema_version >= 2:
            for state in states.values():
                if state.get("status") in {"pending", "running", "waiting_approval"}:
                    state["status"] = "cancelled"
                    state["completed_at"] = utc_now()
                    state["pending_approval_id"] = None
        updated = self.graph_store.update_run(
            run.id,
            status="cancelled",
            node_states=states,
            resource_metrics=self._suspend_active_metrics(run),
            pending_approval_id=None,
            pause_requested=False,
            error="cancelled by user",
        )
        self.workspace_store.append_event(
            run.id,
            "graph_cancelled",
            {
                "kind": "graph",
                "current_node_id": run.current_node_id,
                "cancelled_approval_id": cancelled_approval_id,
            },
        )
        return updated

    def approve(
        self,
        run_id: str,
        approval_id: str,
        decision: str,
        *,
        resolved_by: str = "user",
        reason: str | None = None,
        resume: bool = True,
    ) -> GraphRun:
        run = self.graph_store.get_run(run_id)
        if run.status in _TERMINAL_RUN_STATUSES:
            raise GraphExecutionError(
                f"cannot resolve approval for graph run in {run.status} state"
            )
        if run.pending_approval_id != approval_id:
            raise WorkspaceConflictError(
                f"approval {approval_id} is not pending for graph run {run.id}"
            )
        resolved = self.workspace_store.resolve_approval(
            approval_id,
            decision,
            resolved_by=resolved_by,
            reason=reason,
        )
        if resolved.status == "approved" and resume:
            return self.resume(run.id)
        if resolved.status != "approved":
            error = f"graph approval {resolved.status}: {approval_id}"
            current_id = run.current_node_id
            if current_id:
                graph = self._pinned_graph(run)
                node = graph.node_map.get(current_id)
                state = run.node_states.get(current_id)
                if node is not None and state is not None:
                    cleared = dict(state)
                    cleared["pending_approval_id"] = None
                    run = self.graph_store.update_run(
                        run.id,
                        pending_approval_id=None,
                    )
                    return self._fail_node_and_run(run, node, cleared, error)
            run = self.graph_store.update_run(
                run.id,
                pending_approval_id=None,
            )
            return self._fail_run(run, error)
        return self.graph_store.get_run(run.id)

    def retry_failed_node(
        self,
        run_id: str,
        node_id: str | None = None,
        *,
        allow_side_effect_retry: bool = False,
        force_new_side_effect: bool = False,
    ) -> GraphRun:
        run = self.graph_store.get_run(run_id)
        graph = self._pinned_graph(run)
        selected = node_id or run.current_node_id
        if not selected or selected not in graph.node_map:
            raise GraphExecutionError("failed node id is required")
        state = dict(run.node_states.get(selected) or {})
        if state.get("status") != "failed":
            raise GraphExecutionError(f"node {selected} is not failed")
        node = graph.node_map[selected]
        has_side_effect = self._node_has_side_effect(node)
        if has_side_effect and not allow_side_effect_retry:
            raise GraphExecutionError(
                "retrying this failed side-effect node requires explicit "
                "allow_side_effect_retry=true"
            )
        if force_new_side_effect and not has_side_effect:
            raise GraphExecutionError(
                "force_new_side_effect is valid only for a side-effect node"
            )
        recovery_guard = str(state.get("recovery_guard") or "")
        preserve_execution_key = bool(
            has_side_effect
            and recovery_guard == "interrupted_side_effect"
            and not force_new_side_effect
        )
        if node.type == "loop" and has_side_effect and not preserve_execution_key:
            state["side_effect_retry_generation"] = (
                int(state.get("side_effect_retry_generation") or 0) + 1
            )
        state.update(
            {
                "status": "pending",
                "error": None,
                "pending_approval_id": None,
                "idempotency_key": (
                    state.get("idempotency_key") if preserve_execution_key else None
                ),
                "started_at": None,
                "completed_at": None,
                "recovery_guard": (
                    recovery_guard if preserve_execution_key else None
                ),
            }
        )
        states = {key: dict(value) for key, value in run.node_states.items()}
        states[selected] = state
        updated = self.graph_store.update_run(
            run.id,
            status="queued",
            node_states=states,
            current_node_id=selected,
            pending_approval_id=None,
            pause_requested=False,
            error=None,
        )
        self._cancel_event(run.id).clear()
        self.workspace_store.append_event(
            run.id,
            "graph_node_retry_requested",
            {
                "node_id": selected,
                "completed_nodes_preserved": [
                    key
                    for key, value in states.items()
                    if value.get("status") == "completed"
                ],
                "preserved_execution_key": preserve_execution_key,
                "force_new_side_effect": force_new_side_effect,
            },
        )
        self._executor.submit(self._run, run.id)
        return updated

    def recover_incomplete_runs(self) -> list[str]:
        recovered: list[str] = []
        for run in self.graph_store.list_runs(limit=10_000):
            if run.status not in {"queued", "running", "waiting_approval"}:
                continue
            try:
                graph = self._pinned_graph(run)
            except GraphError as exc:
                error = f"cannot recover pinned Graph revision: {exc}"
                self.graph_store.update_run(
                    run.id,
                    status="failed",
                    resource_metrics=self._suspend_active_metrics(
                        run,
                        stop_at=run.updated_at,
                    ),
                    pause_requested=False,
                    error=error,
                )
                self.workspace_store.append_event(
                    run.id,
                    "graph_recovery_failed",
                    {"reason": error},
                )
                recovered.append(run.id)
                continue
            if run.resource_metrics.get("completion_prepared"):
                output_node = next(
                    node for node in graph.nodes if node.type == "output"
                )
                output_state = dict(run.node_states.get(output_node.id) or {})
                self._complete_run(
                    run,
                    graph,
                    output_state.get("output"),
                    output_state=output_state,
                    output_metrics=dict(output_state.get("metrics") or {}),
                    attempt=max(1, int(output_state.get("attempts") or 1)),
                    already_checkpointed=True,
                )
                recovered.append(run.id)
                continue
            states = {key: dict(value) for key, value in run.node_states.items()}
            if graph.schema_version >= 2:
                interrupted: list[str] = []
                for node_id, state in states.items():
                    if state.get("status") != "running":
                        continue
                    node = graph.node_map.get(node_id)
                    if node and self._node_has_side_effect(node):
                        state["status"] = "failed"
                        state["recovery_guard"] = "interrupted_side_effect"
                        state["error"] = (
                            "side-effect execution was interrupted after its durable start guard; "
                            "MTPLX will not execute it again automatically"
                        )
                    else:
                        state["status"] = "pending"
                        state["error"] = None
                        state["idempotency_key"] = None
                        state["pending_approval_id"] = None
                    interrupted.append(node_id)
                error = "MTPLX restarted before this Graph scheduler wave reached a checkpoint"
                self.graph_store.update_run(
                    run.id,
                    status="paused",
                    node_states=states,
                    resource_metrics=self._suspend_active_metrics(
                        run,
                        stop_at=run.updated_at,
                    ),
                    pause_requested=True,
                    error=error,
                )
                self.workspace_store.append_event(
                    run.id,
                    "graph_recovered",
                    {
                        "current_node_id": run.current_node_id,
                        "pending_approval_id": run.pending_approval_id,
                        "interrupted_node_ids": interrupted,
                        "reason": error,
                    },
                )
                recovered.append(run.id)
                continue
            current = states.get(run.current_node_id or "")
            error = "MTPLX restarted before this Graph run reached a checkpoint"
            if current and current.get("status") == "running":
                node = graph.node_map.get(run.current_node_id or "")
                if node and self._node_has_side_effect(node):
                    current["status"] = "failed"
                    current["recovery_guard"] = "interrupted_side_effect"
                    current["error"] = (
                        "side-effect execution was interrupted after its durable start guard; "
                        "MTPLX will not execute it again automatically"
                    )
                    error = current["error"]
                else:
                    current["status"] = "pending"
                    current["error"] = None
                    current["idempotency_key"] = None
                    current["pending_approval_id"] = None
            self.graph_store.update_run(
                run.id,
                status="paused",
                node_states=states,
                resource_metrics=self._suspend_active_metrics(
                    run,
                    stop_at=run.updated_at,
                ),
                pause_requested=True,
                error=error,
            )
            self.workspace_store.append_event(
                run.id,
                "graph_recovered",
                {
                    "current_node_id": run.current_node_id,
                    "pending_approval_id": run.pending_approval_id,
                    "reason": error,
                },
            )
            recovered.append(run.id)
        return recovered

    def _run(self, run_id: str) -> None:
        lock = self._run_lock(run_id)
        if not lock.acquire(blocking=False):
            return
        try:
            self._run_locked(run_id)
        finally:
            lock.release()

    def _run_locked(self, run_id: str) -> None:
        run = self.graph_store.get_run(run_id)
        if run.status in _TERMINAL_RUN_STATUSES or run.pause_requested:
            return
        if self._cancel_event(run.id).is_set():
            return
        graph = self._pinned_graph(run)
        if run.graph_sha256 != graph.content_sha256:
            self._fail_run(run, "pinned Graph revision hash no longer matches")
            return
        if graph.schema_version >= 2:
            self._run_parallel_locked(run, graph)
            return
        if run.current_node_id is None:
            input_node = next(node for node in graph.nodes if node.type == "input")
            run = self.graph_store.update_run(run.id, current_node_id=input_node.id)
        run = self.graph_store.update_run(
            run.id,
            status="running",
            resource_metrics=self._start_active_metrics(run),
            pause_requested=False,
            error=None,
        )
        self.workspace_store.append_event(
            run.id,
            "graph_started",
            {
                "graph_id": graph.id,
                "graph_revision": graph.revision,
                "pinned_model": run.pinned_model,
                "runtime_profile": run.runtime_profile,
            },
        )
        max_steps = int(graph.limits["max_steps"])
        while True:
            run = self.graph_store.get_run(run.id)
            if run.status == "cancelled":
                return
            if self._cancel_event(run.id).is_set():
                self.cancel(run.id)
                return
            if run.pause_requested:
                self.graph_store.update_run(
                    run.id,
                    status="paused",
                    resource_metrics=self._suspend_active_metrics(run),
                )
                self.workspace_store.append_event(
                    run.id,
                    "graph_paused",
                    {"current_node_id": run.current_node_id},
                )
                return
            if self._deadline_exceeded(run, graph):
                self._fail_run(run, "Graph run timeout exceeded")
                return
            steps = int(run.resource_metrics.get("steps_completed") or 0)
            if steps >= max_steps:
                self._fail_run(run, f"Graph step limit exceeded: {max_steps}")
                return
            node_id = run.current_node_id
            if not node_id or node_id not in graph.node_map:
                self._fail_run(run, f"Graph has no executable node: {node_id}")
                return
            node = graph.node_map[node_id]
            state = dict(run.node_states[node.id])
            if state.get("status") == "completed":
                if node.type == "output":
                    self._complete_run(
                        run,
                        graph,
                        state.get("output"),
                        output_state=state,
                        output_metrics=dict(state.get("metrics") or {}),
                        attempt=int(state.get("attempts") or 1),
                        already_checkpointed=True,
                    )
                    return
                next_id = self._next_node(graph, node, state.get("output"), run)
                if next_id is None:
                    self._fail_run(run, f"node {node.id} completed without a route to output")
                    return
                self.graph_store.update_run(run.id, current_node_id=next_id)
                continue

            resume_approval_id = None
            if state.get("status") == "waiting_approval":
                resume_approval_id = str(state.get("pending_approval_id") or "") or None
                if not resume_approval_id:
                    self._fail_run(run, f"node {node.id} lost its pending approval id")
                    return
                approval = self.workspace_store.get_approval(resume_approval_id)
                if approval.status == "pending":
                    self.graph_store.update_run(run.id, status="waiting_approval")
                    return
                if approval.status != "approved":
                    self._fail_node_and_run(
                        run,
                        node,
                        state,
                        f"approval {approval.status}: {resume_approval_id}",
                    )
                    return

            resource = self._check_resources(run, graph, node)
            run = self.graph_store.get_run(run.id)
            if resource is not None:
                if resource.status == "waiting_approval":
                    self.graph_store.update_run(
                        run.id,
                        status="paused",
                        resource_metrics=self._suspend_active_metrics(run),
                        pause_requested=True,
                        error=resource.error,
                    )
                    self.workspace_store.append_event(
                        run.id,
                        "graph_resource_paused",
                        {"node_id": node.id, "reason": resource.error},
                    )
                else:
                    self._fail_node_and_run(run, node, state, resource.error or "resource failure")
                return

            attempt = max(1, int(state.get("attempts") or 0)) if resume_approval_id else int(
                state.get("attempts") or 0
            ) + 1
            execution_key = str(state.get("idempotency_key") or "")
            if not execution_key:
                execution_key = f"{run.id}:{node.id}:attempt:{attempt}"
            state.update(
                {
                    "status": "running",
                    "attempts": attempt,
                    "started_at": state.get("started_at") or utc_now(),
                    "completed_at": None,
                    "error": None,
                    "idempotency_key": execution_key,
                    "pending_approval_id": resume_approval_id,
                }
            )
            states = {key: dict(value) for key, value in run.node_states.items()}
            states[node.id] = state
            run = self.graph_store.update_run(
                run.id,
                node_states=states,
                current_node_id=node.id,
                pending_approval_id=resume_approval_id,
            )
            self.workspace_store.append_event(
                run.id,
                "graph_node_started",
                {
                    "node_id": node.id,
                    "node_type": node.type,
                    "attempt": attempt,
                    "idempotency_key": execution_key,
                },
            )
            try:
                outcome = self._execute_node(
                    run,
                    graph,
                    node,
                    state,
                    execution_key=execution_key,
                    approval_id=resume_approval_id,
                )
            except Exception as exc:
                outcome = NodeOutcome(
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                    retryable=node.type not in SIDE_EFFECT_NODE_TYPES,
                )
            run = self.graph_store.get_run(run.id)
            if run.status == "cancelled":
                return
            state = dict(run.node_states[node.id])
            state.update(outcome.state_updates)
            if outcome.status == "paused":
                state.update(
                    {
                        "status": "pending",
                        "error": outcome.error,
                        "pending_approval_id": None,
                    }
                )
                states = {key: dict(value) for key, value in run.node_states.items()}
                states[node.id] = state
                self.graph_store.update_run(
                    run.id,
                    status="paused",
                    node_states=states,
                    resource_metrics=self._suspend_active_metrics(run),
                    pending_approval_id=None,
                    pause_requested=True,
                    error=outcome.error,
                )
                self.workspace_store.append_event(
                    run.id,
                    "graph_paused",
                    {
                        "current_node_id": node.id,
                        "iterations_completed": state.get("iterations_completed"),
                    },
                )
                return
            if outcome.status == "waiting_approval":
                state.update(
                    {
                        "status": "waiting_approval",
                        "pending_approval_id": outcome.approval_id,
                        "error": None,
                    }
                )
                states = {key: dict(value) for key, value in run.node_states.items()}
                states[node.id] = state
                self.graph_store.update_run(
                    run.id,
                    status="waiting_approval",
                    node_states=states,
                    resource_metrics=self._suspend_active_metrics(run),
                    pending_approval_id=outcome.approval_id,
                )
                self.workspace_store.append_event(
                    run.id,
                    "graph_waiting_approval",
                    {"node_id": node.id, "approval_id": outcome.approval_id},
                )
                return
            if outcome.status != "completed":
                state.update(
                    {
                        "status": "failed",
                        "error": outcome.error or "node failed",
                        "completed_at": utc_now(),
                        "metrics": dict(outcome.metrics),
                    }
                )
                states = {key: dict(value) for key, value in run.node_states.items()}
                states[node.id] = state
                run = self.graph_store.update_run(
                    run.id,
                    node_states=states,
                    pending_approval_id=None,
                )
                if outcome.retryable and attempt < self._max_attempts(graph, node):
                    state["status"] = "pending"
                    state["idempotency_key"] = None
                    states[node.id] = state
                    self.graph_store.update_run(run.id, node_states=states)
                    self.workspace_store.append_event(
                        run.id,
                        "graph_node_retry_scheduled",
                        {
                            "node_id": node.id,
                            "attempt": attempt + 1,
                            "backoff_seconds": self._retry_backoff(graph, node),
                        },
                    )
                    if not self._wait_retry_backoff(
                        run.id,
                        self._retry_backoff(graph, node),
                    ):
                        continue
                    continue
                self._fail_node_and_run(
                    run,
                    node,
                    state,
                    outcome.error or "node failed",
                    persist_node=False,
                )
                return

            if node.type == "output":
                self._complete_run(
                    run,
                    graph,
                    outcome.output,
                    output_state=state,
                    output_metrics=outcome.metrics,
                    attempt=attempt,
                )
                return

            state.update(
                {
                    "status": "completed",
                    "output": outcome.output,
                    "error": None,
                    "completed_at": utc_now(),
                    "pending_approval_id": None,
                    "metrics": dict(outcome.metrics),
                    "recovery_guard": None,
                }
            )
            states = {key: dict(value) for key, value in run.node_states.items()}
            states[node.id] = state
            metrics = dict(run.resource_metrics)
            if node.type != "loop":
                metrics["steps_completed"] = int(
                    metrics.get("steps_completed") or 0
                ) + 1
            metrics["last_node_metrics"] = dict(outcome.metrics)
            metrics["last_completed_node_id"] = node.id
            run = self.graph_store.update_run(
                run.id,
                node_states=states,
                resource_metrics=metrics,
                pending_approval_id=None,
            )
            self.workspace_store.append_event(
                run.id,
                "graph_node_completed",
                {
                    "node_id": node.id,
                    "node_type": node.type,
                    "attempt": attempt,
                    "metrics": outcome.metrics,
                },
            )
            next_id = self._next_node(graph, node, outcome.output, run)
            if next_id is None:
                self._fail_run(run, f"node {node.id} has no route to the output node")
                return
            self.graph_store.update_run(run.id, current_node_id=next_id)

    def _run_parallel_locked(self, run: GraphRun, graph: GraphDefinition) -> None:
        """Run a schema-v2 DAG in checkpointed scheduler waves.

        The scheduler may overlap independent pure and model nodes, but retains
        one exclusive lane for side effects, approvals, and loops. Model work
        still goes through MTPLX's single admission lease, so a Graph never
        bypasses the runtime's memory and thermal controls.
        """
        run = self.graph_store.update_run(
            run.id,
            status="running",
            resource_metrics=self._start_active_metrics(run),
            pause_requested=False,
            error=None,
        )
        self.workspace_store.append_event(
            run.id,
            "graph_started",
            {
                "graph_id": graph.id,
                "graph_revision": graph.revision,
                "pinned_model": run.pinned_model,
                "runtime_profile": run.runtime_profile,
                "scheduler": dict(graph.schedule),
            },
        )
        max_steps = int(graph.limits["max_steps"])
        while True:
            run = self.graph_store.get_run(run.id)
            if run.status == "cancelled":
                return
            if self._cancel_event(run.id).is_set():
                self.cancel(run.id)
                return
            if run.pause_requested:
                self.graph_store.update_run(
                    run.id,
                    status="paused",
                    resource_metrics=self._suspend_active_metrics(run),
                )
                self.workspace_store.append_event(
                    run.id,
                    "graph_paused",
                    {"current_node_id": run.current_node_id},
                )
                return
            if self._deadline_exceeded(run, graph):
                self._fail_run(run, "Graph run timeout exceeded")
                return
            if int(run.resource_metrics.get("steps_completed") or 0) >= max_steps:
                self._fail_run(run, f"Graph step limit exceeded: {max_steps}")
                return

            ready, skipped = self._parallel_ready_nodes(run, graph)
            if skipped:
                states = {key: dict(value) for key, value in run.node_states.items()}
                for node_id in skipped:
                    state = states[node_id]
                    state.update(
                        {
                            "status": "skipped",
                            "completed_at": utc_now(),
                            "error": None,
                            "pending_approval_id": None,
                        }
                    )
                    self.workspace_store.append_event(
                        run.id,
                        "graph_node_skipped",
                        {"node_id": node_id, "reason": "no selected incoming route"},
                    )
                self.graph_store.update_run(run.id, node_states=states)
                continue
            if not ready:
                terminal = {"completed", "skipped", "failed", "cancelled"}
                if all(
                    state.get("status") in terminal
                    for state in run.node_states.values()
                ):
                    self._fail_run(run, "Graph reached no output through the selected routes")
                else:
                    self._fail_run(run, "Graph scheduler has no executable node")
                return

            wave = self._parallel_wave(graph, ready)
            states = {key: dict(value) for key, value in run.node_states.items()}
            prepared: list[tuple[GraphNode, dict[str, Any], str]] = []
            for node in wave:
                resource = self._check_resources(replace(run, current_node_id=node.id), graph, node)
                if resource is not None:
                    if resource.status == "waiting_approval":
                        self.graph_store.update_run(
                            run.id,
                            status="paused",
                            resource_metrics=self._suspend_active_metrics(run),
                            pause_requested=True,
                            error=resource.error,
                        )
                        self.workspace_store.append_event(
                            run.id,
                            "graph_resource_paused",
                            {"node_id": node.id, "reason": resource.error},
                        )
                    else:
                        self._fail_node_and_run(
                            run,
                            node,
                            states[node.id],
                            resource.error or "resource failure",
                        )
                    return
                state = states[node.id]
                attempt = int(state.get("attempts") or 0) + 1
                execution_key = str(state.get("idempotency_key") or "")
                if not execution_key:
                    execution_key = f"{run.id}:{node.id}:attempt:{attempt}"
                state.update(
                    {
                        "status": "running",
                        "attempts": attempt,
                        "started_at": state.get("started_at") or utc_now(),
                        "completed_at": None,
                        "error": None,
                        "idempotency_key": execution_key,
                        "pending_approval_id": None,
                    }
                )
                prepared.append((node, dict(state), execution_key))
                self.workspace_store.append_event(
                    run.id,
                    "graph_node_queued",
                    {
                        "node_id": node.id,
                        "node_type": node.type,
                        "scheduler_policy": graph.schedule.get("policy", "fifo"),
                        "wave_size": len(wave),
                    },
                )
            run = self.graph_store.update_run(
                run.id,
                node_states=states,
                current_node_id=prepared[0][0].id,
                pending_approval_id=None,
            )
            futures: list[tuple[GraphNode, dict[str, Any], str, Any]] = []
            for node, state, execution_key in prepared:
                self.workspace_store.append_event(
                    run.id,
                    "graph_node_started",
                    {
                        "node_id": node.id,
                        "node_type": node.type,
                        "attempt": state["attempts"],
                        "idempotency_key": execution_key,
                    },
                )
                snapshot = replace(run, current_node_id=node.id)
                future = self._node_executor.submit(
                    self._execute_node,
                    snapshot,
                    graph,
                    node,
                    state,
                    execution_key=execution_key,
                    approval_id=None,
                )
                futures.append((node, state, execution_key, future))

            outcomes: list[tuple[GraphNode, dict[str, Any], NodeOutcome]] = []
            for node, state, _execution_key, future in futures:
                try:
                    outcome = future.result()
                except Exception as exc:
                    outcome = NodeOutcome(
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        retryable=node.type not in SIDE_EFFECT_NODE_TYPES,
                    )
                outcomes.append((node, state, outcome))

            run = self.graph_store.get_run(run.id)
            states = {key: dict(value) for key, value in run.node_states.items()}
            metrics = dict(run.resource_metrics)
            failure: tuple[GraphNode, dict[str, Any], NodeOutcome] | None = None
            waiting: tuple[GraphNode, dict[str, Any], NodeOutcome] | None = None
            output: tuple[GraphNode, dict[str, Any], NodeOutcome] | None = None
            retry_seconds = 0.0
            for node, _started_state, outcome in outcomes:
                state = states[node.id]
                state.update(outcome.state_updates)
                if outcome.status == "waiting_approval":
                    state.update(
                        {
                            "status": "waiting_approval",
                            "pending_approval_id": outcome.approval_id,
                            "error": None,
                        }
                    )
                    states[node.id] = state
                    waiting = (node, state, outcome)
                    continue
                if outcome.status != "completed":
                    state.update(
                        {
                            "status": "failed",
                            "error": outcome.error or "node failed",
                            "completed_at": utc_now(),
                            "metrics": dict(outcome.metrics),
                        }
                    )
                    if outcome.retryable and int(state.get("attempts") or 0) < self._max_attempts(graph, node):
                        state["status"] = "pending"
                        state["idempotency_key"] = None
                        retry_seconds = max(retry_seconds, self._retry_backoff(graph, node))
                        self.workspace_store.append_event(
                            run.id,
                            "graph_node_retry_scheduled",
                            {
                                "node_id": node.id,
                                "attempt": int(state.get("attempts") or 0) + 1,
                                "backoff_seconds": self._retry_backoff(graph, node),
                            },
                        )
                    elif failure is None:
                        failure = (node, state, outcome)
                    states[node.id] = state
                    continue
                state.update(
                    {
                        "status": "completed",
                        "output": outcome.output,
                        "error": None,
                        "completed_at": utc_now(),
                        "pending_approval_id": None,
                        "metrics": dict(outcome.metrics),
                        "recovery_guard": None,
                        "selected_targets": self._selected_targets(graph, node, outcome.output, run),
                    }
                )
                states[node.id] = state
                metrics["steps_completed"] = int(metrics.get("steps_completed") or 0) + 1
                metrics["last_node_metrics"] = dict(outcome.metrics)
                metrics["last_completed_node_id"] = node.id
                if node.type == "output":
                    output = (node, state, outcome)
                self.workspace_store.append_event(
                    run.id,
                    "graph_node_completed",
                    {
                        "node_id": node.id,
                        "node_type": node.type,
                        "attempt": state["attempts"],
                        "metrics": outcome.metrics,
                    },
                )

            run = self.graph_store.update_run(
                run.id,
                node_states=states,
                resource_metrics=metrics,
                current_node_id=(output[0].id if output else run.current_node_id),
                pending_approval_id=(waiting[2].approval_id if waiting else None),
            )
            if failure is not None:
                self._fail_node_and_run(
                    run,
                    failure[0],
                    failure[1],
                    failure[2].error or "node failed",
                    persist_node=False,
                )
                return
            if waiting is not None:
                self.graph_store.update_run(
                    run.id,
                    status="waiting_approval",
                    resource_metrics=self._suspend_active_metrics(run),
                    pending_approval_id=waiting[2].approval_id,
                )
                self.workspace_store.append_event(
                    run.id,
                    "graph_waiting_approval",
                    {"node_id": waiting[0].id, "approval_id": waiting[2].approval_id},
                )
                return
            if output is not None:
                self._complete_run(
                    run,
                    graph,
                    output[2].output,
                    output_state=output[1],
                    output_metrics=output[2].metrics,
                    attempt=int(output[1].get("attempts") or 1),
                )
                return
            if retry_seconds > 0 and not self._wait_retry_backoff(run.id, retry_seconds):
                continue

    def _parallel_ready_nodes(
        self,
        run: GraphRun,
        graph: GraphDefinition,
    ) -> tuple[list[GraphNode], list[str]]:
        incoming: dict[str, list[Any]] = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            incoming.setdefault(edge.target, []).append(edge)
        ready: list[GraphNode] = []
        skipped: list[str] = []
        for node in graph.nodes:
            state = run.node_states.get(node.id, {})
            if state.get("status") != "pending":
                continue
            edges = incoming.get(node.id, [])
            if not edges:
                if node.type == "input":
                    ready.append(node)
                continue
            source_states = [run.node_states.get(edge.source, {}) for edge in edges]
            terminal = {"completed", "skipped", "failed", "cancelled"}
            if not all(source.get("status") in terminal for source in source_states):
                continue
            active = [
                edge
                for edge in edges
                if self._edge_is_active(graph, edge, run.node_states.get(edge.source, {}))
            ]
            if not active:
                skipped.append(node.id)
                continue
            if any(
                run.node_states.get(edge.source, {}).get("status") != "completed"
                for edge in active
            ):
                continue
            if node.type == "join" and str(node.config.get("mode") or "all") == "any":
                ready.append(node)
            elif node.type != "join" and len(active) != 1:
                continue
            else:
                ready.append(node)
        return ready, skipped

    def _parallel_wave(
        self,
        graph: GraphDefinition,
        ready: list[GraphNode],
    ) -> list[GraphNode]:
        distances = self._critical_path_distances(graph)
        policy = str(graph.schedule.get("policy") or "fifo")
        ordered = sorted(
            ready,
            key=lambda node: (
                -node.priority,
                -distances.get(node.id, 0) if policy == "critical_path" else 0,
                node.id,
            ),
        )
        exclusive = [
            node
            for node in ordered
            if node.type in {"loop", "tool", "human_approval", "memory_write", "memory_curate"}
        ]
        if exclusive:
            return [exclusive[0]]
        return ordered[: max(1, int(graph.limits.get("max_concurrency") or 1))]

    @staticmethod
    def _critical_path_distances(graph: GraphDefinition) -> dict[str, int]:
        outgoing: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
        for edge in graph.edges:
            outgoing.setdefault(edge.source, []).append(edge.target)
        distances: dict[str, int] = {}

        def visit(node_id: str) -> int:
            if node_id in distances:
                return distances[node_id]
            children = outgoing.get(node_id, [])
            distances[node_id] = 0 if not children else 1 + max(visit(item) for item in children)
            return distances[node_id]

        for node in graph.nodes:
            visit(node.id)
        return distances

    def _selected_targets(
        self,
        graph: GraphDefinition,
        node: GraphNode,
        output: Any,
        run: GraphRun,
    ) -> list[str]:
        outgoing = [edge for edge in graph.edges if edge.source == node.id]
        if node.type != "conditional":
            return [edge.target for edge in outgoing]
        context = {**self._context(run), "condition": output}
        default: str | None = None
        for edge in outgoing:
            condition = edge.condition or {}
            if condition.get("default"):
                default = edge.target
                continue
            selector = str(condition.get("path") or "condition")
            if self._condition_matches(self._lookup(context, selector), condition):
                return [edge.target]
        return [default] if default else []

    @staticmethod
    def _edge_is_active(
        graph: GraphDefinition,
        edge: Any,
        source_state: Mapping[str, Any],
    ) -> bool:
        if source_state.get("status") != "completed":
            return False
        selected = source_state.get("selected_targets")
        if isinstance(selected, list):
            return edge.target in selected
        source = graph.node_map.get(edge.source)
        return source is not None and source.type != "conditional"

    def _execute_node(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        node: GraphNode,
        state: Mapping[str, Any],
        *,
        execution_key: str,
        approval_id: str | None,
    ) -> NodeOutcome:
        context = self._context(run)
        if node.type == "input":
            return NodeOutcome(status="completed", output=dict(run.inputs))
        if node.type == "output":
            mapping = node.config.get("mapping")
            output = (
                self._render(mapping, context)
                if isinstance(mapping, Mapping)
                else context.get("last_output")
            )
            return NodeOutcome(status="completed", output=output)
        if node.type == "join":
            active_inputs = {
                edge.source: run.node_states.get(edge.source, {}).get("output")
                for edge in graph.edges
                if edge.target == node.id
                and self._edge_is_active(
                    graph,
                    edge,
                    run.node_states.get(edge.source, {}),
                )
            }
            mapping = node.config.get("mapping")
            output = (
                self._render(mapping, {**context, "join": active_inputs})
                if isinstance(mapping, Mapping)
                else active_inputs
            )
            return NodeOutcome(status="completed", output=output)
        if node.type == "model":
            return self._execute_model(run, graph, node, context)
        if node.type == "tool":
            return self._execute_tool(
                run,
                graph,
                node,
                context,
                execution_key=execution_key,
                approval_id=approval_id,
            )
        if node.type == "human_approval":
            return self._execute_human_approval(
                run,
                node,
                context,
                approval_id=approval_id,
            )
        if node.type == "memory_read":
            return self._execute_memory_read(run, node, context)
        if node.type == "memory_write":
            return self._execute_memory_write(
                run,
                graph,
                node,
                context,
                approval_id=approval_id,
            )
        if node.type == "memory_curate":
            return self._execute_memory_curate(
                run,
                graph,
                node,
                context,
                state,
                approval_id=approval_id,
            )
        if node.type == "conditional":
            selector = str(node.config.get("selector") or "last_output")
            return NodeOutcome(status="completed", output=self._lookup(context, selector))
        if node.type == "loop":
            return self._execute_loop(
                run,
                graph,
                node,
                state,
                approval_id=approval_id,
            )
        return NodeOutcome(status="failed", error=f"unsupported node type: {node.type}")

    def _execute_model(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        node: GraphNode,
        context: Mapping[str, Any],
    ) -> NodeOutcome:
        prompt = str(self._render(node.config.get("prompt") or "", context))
        if node.config.get("prompt_path"):
            prompt = str(self._lookup(context, str(node.config["prompt_path"])) or "")
        context_limit = int(graph.limits["max_context_tokens"])
        prompt_limit = max(4_096, context_limit * 4)
        truncated_chars = max(0, len(prompt) - prompt_limit)
        if truncated_chars:
            prompt = prompt[-prompt_limit:]
        maximum = max(
            1,
            min(
                int(node.config.get("max_tokens") or 1024),
                context_limit,
                16_384,
            ),
        )
        requested_model = str(node.config.get("model") or run.pinned_model or "") or None
        session_id = f"graph:{run.id}:{node.id}"
        request_id = (
            f"chatcmpl-graph-{safe_id(run.id, fallback='run')[:32]}-"
            f"{safe_id(node.id, fallback='node')[:24]}-{uuid.uuid4().hex[:8]}"
        )
        started = time.monotonic()
        self.workspace_store.append_event(
            run.id,
            "graph_model_started",
            {
                "node_id": node.id,
                "model": requested_model,
                "session_id": session_id,
                "request_id": request_id,
                "runtime_profile": run.runtime_profile,
            },
        )
        response: Mapping[str, Any]
        model_wait_started = time.monotonic()
        timing: dict[str, int] = {}
        timeout_seconds = max(1, int(node.timeout_seconds or graph.timeout_seconds))
        model_cancel_event = threading.Event()

        def invoke() -> tuple[Mapping[str, Any], bool]:
            fallback_used = False
            with self._model_lease:
                if model_cancel_event.is_set():
                    raise GraphExecutionError("model node cancelled before admission")
                timing["model_wait_ms"] = int(
                    (time.monotonic() - model_wait_started) * 1000
                )
                try:
                    value = self.model_runner(
                        model=requested_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=maximum,
                        session_id=session_id,
                        runtime_profile=run.runtime_profile,
                        graph_run_id=run.id,
                        node_id=node.id,
                        timeout_seconds=timeout_seconds,
                        request_id=request_id,
                        cancellation_event=model_cancel_event,
                    )
                except Exception:
                    if model_cancel_event.is_set() or self._cancel_event(run.id).is_set():
                        raise GraphExecutionError("model node cancelled")
                    allow_fallback = bool(
                        graph.runtime_requirements.get("allow_model_fallback", True)
                    )
                    if not allow_fallback or requested_model is None:
                        raise
                    value = self.model_runner(
                        model=None,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=maximum,
                        session_id=session_id,
                        runtime_profile=run.runtime_profile,
                        graph_run_id=run.id,
                        node_id=node.id,
                        timeout_seconds=timeout_seconds,
                        request_id=request_id,
                        cancellation_event=model_cancel_event,
                    )
                    fallback_used = True
            return value, fallback_used

        future = self._model_executor.submit(invoke)
        deadline = time.monotonic() + timeout_seconds
        while True:
            if self._cancel_event(run.id).is_set():
                model_cancel_event.set()
                self._cancel_model_request(request_id)
                future.cancel()
                return NodeOutcome(status="cancelled", error="model node cancelled")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                model_cancel_event.set()
                self._cancel_model_request(request_id)
                future.cancel()
                self.workspace_store.append_event(
                    run.id,
                    "graph_model_timed_out",
                    {"node_id": node.id, "timeout_seconds": timeout_seconds},
                )
                return NodeOutcome(
                    status="failed",
                    error=f"model node timeout exceeded: {timeout_seconds} seconds",
                    retryable=True,
                )
            try:
                response, fallback_used = future.result(timeout=min(0.1, remaining))
                break
            except FutureTimeoutError:
                continue
        model_wait_ms = timing.get("model_wait_ms", 0)
        text = self._model_text(response)
        if not text:
            return NodeOutcome(
                status="failed",
                error="MTPLX model node returned no assistant content",
                retryable=True,
            )
        usage = dict(response.get("usage") or {}) if isinstance(response.get("usage"), Mapping) else {}
        mtplx_value = response.get("mtplx_stats") or response.get("mtplx")
        mtplx = dict(mtplx_value) if isinstance(mtplx_value, Mapping) else {}
        actual_model = str(response.get("model") or requested_model or "") or None
        required_model = str(graph.runtime_requirements.get("model") or "").strip()
        allowed_models = {
            str(item) for item in graph.runtime_requirements.get("allowed_models") or []
        }
        allow_fallback = bool(
            graph.runtime_requirements.get("allow_model_fallback", True)
        )
        if actual_model is None:
            return NodeOutcome(
                status="failed",
                error="MTPLX model response did not identify the served model",
                retryable=True,
            )
        if required_model and actual_model != required_model:
            return NodeOutcome(
                status="failed",
                error=(
                    f"MTPLX served model {actual_model}, required {required_model}"
                ),
                retryable=False,
            )
        if allowed_models and actual_model not in allowed_models:
            return NodeOutcome(
                status="failed",
                error=f"MTPLX served model is not allowed: {actual_model}",
                retryable=False,
            )
        if requested_model and actual_model != requested_model and not allow_fallback:
            return NodeOutcome(
                status="failed",
                error=(
                    f"MTPLX served model {actual_model}, pinned {requested_model}"
                ),
                retryable=False,
            )
        accepted = mtplx.get("accepted_by_depth")
        drafted = mtplx.get("drafted_by_depth")
        accepted_total = sum(
            float(item)
            for item in accepted
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ) if isinstance(accepted, list) else 0.0
        drafted_total = sum(
            float(item)
            for item in drafted
            if isinstance(item, (int, float)) and not isinstance(item, bool)
        ) if isinstance(drafted, list) else 0.0
        scheduler_wait_ms = int(float(mtplx.get("lock_wait_time_s") or 0.0) * 1000)
        metrics = {
            "latency_ms": int((time.monotonic() - started) * 1000),
            "model_load_wait_ms": scheduler_wait_ms or model_wait_ms,
            "model_lease_wait_ms": model_wait_ms,
            "requested_model": requested_model,
            "actual_model": actual_model,
            "fallback_used": fallback_used
            or (requested_model is not None and actual_model != requested_model),
            "prompt_truncated_chars": truncated_chars,
            "usage": usage,
            "session_cache_hit": bool(
                mtplx.get("session_cache_hit")
                or mtplx.get("cache_hit")
                or response.get("session_cache_hit")
            ),
            "ssd_cache_hit": bool(mtplx.get("ssd_cache_hit")),
            "cache_source": mtplx.get("cache_source"),
            "decode_tok_s": mtplx.get("decode_tok_s"),
            "ttft_s": mtplx.get("ttft_s"),
            "mtp_depth": mtplx.get("mtp_depth"),
            "verify_calls": mtplx.get("verify_calls"),
            "accepted_by_depth": accepted if isinstance(accepted, list) else [],
            "drafted_by_depth": drafted if isinstance(drafted, list) else [],
            "acceptance_rate": (
                accepted_total / drafted_total if drafted_total > 0 else None
            ),
            "scheduler_lane": mtplx.get("scheduler_lane"),
            "active_memory_bytes": mtplx.get("active_memory_bytes"),
            "runtime_profile": run.runtime_profile,
            "request_id": request_id,
        }
        self.workspace_store.append_event(
            run.id,
            "graph_model_completed",
            {"node_id": node.id, **metrics},
        )
        return NodeOutcome(status="completed", output=text, metrics=metrics)

    def _execute_tool(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        node: GraphNode,
        context: Mapping[str, Any],
        *,
        execution_key: str,
        approval_id: str | None,
    ) -> NodeOutcome:
        name = str(node.config.get("tool") or "")
        rendered_arguments = self._render(
            dict(node.config.get("arguments") or {}),
            context,
        )
        if not isinstance(rendered_arguments, Mapping):
            return NodeOutcome(
                status="failed",
                error=f"tool node {node.id} rendered arguments are not an object",
            )
        arguments = dict(rendered_arguments)
        if name in {"run_tests", "run_command"} and node.timeout_seconds is not None:
            current_timeout = int(arguments.get("timeout_seconds") or node.timeout_seconds)
            arguments["timeout_seconds"] = min(current_timeout, node.timeout_seconds)
        overrides = dict(graph.policies)
        category = TOOL_POLICY_KEY.get(name)
        categories = [category] if category else []
        if name in {"run_tests", "run_command"} and bool(arguments.get("network")):
            categories.append("network")
        for policy_category in categories:
            if self._side_effect_approval_required(
                graph,
                node,
                tool=name,
                category=policy_category,
            ):
                overrides[policy_category] = "ask"
        result = self.tool_service.execute(
            run.workspace_id,
            name,
            arguments,
            run_id=run.id,
            approval_id=approval_id,
            root_override=run.workspace_root,
            executor_id=f"graph:{run.id}:{node.id}",
            idempotency_key=execution_key,
            policy_overrides=overrides,
            cancellation_event=self._cancel_event(run.id),
            execution_timeout_seconds=node.timeout_seconds,
        )
        if result.get("status") == "approval_required":
            approval = dict(result.get("approval") or {})
            return NodeOutcome(
                status="waiting_approval",
                approval_id=str(approval.get("id") or "") or None,
                state_updates={"tool_arguments": arguments},
            )
        if not result.get("ok"):
            return NodeOutcome(
                status="failed",
                error=str(result.get("error") or result.get("result") or "tool failed"),
                metrics={"tool_status": result.get("status")},
                retryable=(name not in MUTATING_TOOLS and result.get("status") != "execution_indeterminate"),
            )
        return NodeOutcome(
            status="completed",
            output=result.get("result"),
            metrics={
                "elapsed_ms": result.get("elapsed_ms"),
                "replayed": bool(result.get("replayed")),
                "arguments_sha256": result.get("arguments_sha256"),
            },
        )

    def _execute_human_approval(
        self,
        run: GraphRun,
        node: GraphNode,
        context: Mapping[str, Any],
        *,
        approval_id: str | None,
    ) -> NodeOutcome:
        arguments = {
            "graph_run_id": run.id,
            "node_id": node.id,
            "payload": self._render(node.config.get("payload") or {}, context),
        }
        if not approval_id:
            approval = self.workspace_store.create_approval(
                run.workspace_id,
                run_id=run.id,
                tool="graph_human_approval",
                action=str(node.config.get("action") or node.name),
                description=str(
                    self._render(node.config.get("description") or node.name, context)
                ),
                target=node.id,
                risk=str(node.config.get("risk") or "medium"),
                arguments=arguments,
                expires_in_seconds=int(node.config.get("expires_in_seconds") or 3600),
            )
            return NodeOutcome(status="waiting_approval", approval_id=approval.id)
        try:
            self.workspace_store.consume_approval(
                approval_id,
                workspace_id=run.workspace_id,
                run_id=run.id,
                tool="graph_human_approval",
                arguments=arguments,
                consumed_by=f"graph:{run.id}:{node.id}",
            )
        except WorkspaceStoreError as exc:
            return NodeOutcome(status="failed", error=str(exc))
        return NodeOutcome(
            status="completed",
            output={"approved": True, "approval_id": approval_id},
        )

    def _execute_memory_read(
        self,
        run: GraphRun,
        node: GraphNode,
        context: Mapping[str, Any],
    ) -> NodeOutcome:
        path = str(self._render(node.config.get("path") or "", context))
        try:
            document = self.memory_store.read(path, principal=self._memory_principal(run))
        except MemoryNotFoundError as exc:
            if bool(node.config.get("optional")):
                return NodeOutcome(status="completed", output=None)
            return NodeOutcome(status="failed", error=str(exc), retryable=False)
        return NodeOutcome(status="completed", output=document.to_dict())

    def _execute_memory_write(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        node: GraphNode,
        context: Mapping[str, Any],
        *,
        approval_id: str | None,
    ) -> NodeOutcome:
        path = str(self._render(node.config.get("path") or "", context))
        content = str(self._render(node.config.get("content") or "", context))
        expected = str(self._render(node.config.get("expected_sha256") or "", context))
        principal = self._memory_principal(run)
        try:
            path = self.memory_store.validate_write_path(path, principal=principal)
        except MemoryPermissionError as exc:
            return NodeOutcome(status="failed", error=str(exc), retryable=False)
        arguments = {
            "path": path,
            "content_sha256": content_sha256(content),
            "expected_sha256": expected,
        }
        mode = str(graph.policies.get("memory") or "ask")
        if mode == "deny":
            return NodeOutcome(status="failed", error="Graph policy denies memory writes")
        if mode == "ask" or self._side_effect_approval_required(
            graph,
            node,
            category="memory",
        ):
            approval = self._memory_approval(
                run,
                node,
                arguments,
                approval_id=approval_id,
            )
            if isinstance(approval, NodeOutcome):
                return approval
        try:
            current = self.memory_store.read(path, principal=principal)
        except MemoryNotFoundError:
            current = None
        except MemoryPermissionError as exc:
            return NodeOutcome(status="failed", error=str(exc), retryable=False)
        if current is not None and current.content_sha256 == content_sha256(content):
            return NodeOutcome(
                status="completed",
                output={**current.to_dict(), "idempotent_replay": True},
            )
        try:
            document = self.memory_store.write(
                path,
                content,
                expected_sha256=expected,
                author=f"graph:{run.graph_id}",
                session_id=run.id,
                agent_id=principal.agent_id,
                metadata={
                    "graph_id": run.graph_id,
                    "node_id": node.id,
                    "project_id": run.project_id,
                },
                principal=principal,
            )
        except (MemoryConflictError, MemoryPermissionError) as exc:
            return NodeOutcome(status="failed", error=str(exc), retryable=False)
        self.workspace_store.append_event(
            run.id,
            "graph_memory_written",
            {
                "node_id": node.id,
                "path": document.path,
                "version": document.version,
                "content_sha256": document.content_sha256,
            },
        )
        return NodeOutcome(status="completed", output=document.to_dict())

    def _execute_memory_curate(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        node: GraphNode,
        context: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        approval_id: str | None,
    ) -> NodeOutcome:
        curated_content = state.get("curated_content")
        curation_metrics = dict(state.get("curation_model_metrics") or {})
        if not isinstance(curated_content, str):
            query = str(self._render(node.config.get("query") or run.graph_id, context))
            memory_context = self.memory_store.build_context(
                query,
                principal=self._memory_principal(run),
                max_chars=int(node.config.get("max_context_chars") or 12_000),
            )
            model_node = GraphNode(
                id=f"{node.id}-curation-model",
                type="model",
                name="Memory curation model",
                config={
                    "prompt": (
                        "Extract concise durable lessons from this memory context. Preserve provenance "
                        "and do not invent facts.\n\n"
                        + memory_context.context
                    ),
                    "max_tokens": int(node.config.get("max_tokens") or 1200),
                },
                timeout_seconds=node.timeout_seconds,
            )
            generated = self._execute_model(run, graph, model_node, context)
            if generated.status != "completed":
                return generated
            curated_content = str(generated.output)
            curation_metrics = dict(generated.metrics)
        write_node = GraphNode(
            id=node.id,
            type="memory_write",
            name=node.name,
            config={
                "path": node.config.get("path"),
                "content": curated_content,
                "expected_sha256": node.config.get("expected_sha256", ""),
            },
            approval=node.approval,
        )
        outcome = self._execute_memory_write(
            run,
            graph,
            write_node,
            context,
            approval_id=approval_id,
        )
        if outcome.status == "waiting_approval":
            return NodeOutcome(
                status=outcome.status,
                error=outcome.error,
                approval_id=outcome.approval_id,
                retryable=outcome.retryable,
                metrics=outcome.metrics,
                state_updates={
                    **outcome.state_updates,
                    "curated_content": curated_content,
                    "curation_model_metrics": curation_metrics,
                },
            )
        if outcome.status == "completed":
            return NodeOutcome(
                status="completed",
                output=outcome.output,
                metrics={**outcome.metrics, "curation_model": curation_metrics},
                state_updates={
                    "curated_content": None,
                    "curation_model_metrics": {},
                },
            )
        return outcome

    def _memory_approval(
        self,
        run: GraphRun,
        node: GraphNode,
        arguments: Mapping[str, Any],
        *,
        approval_id: str | None,
    ) -> bool | NodeOutcome:
        if not approval_id:
            approval = self.workspace_store.create_approval(
                run.workspace_id,
                run_id=run.id,
                tool="memory_write",
                action=f"Write memory {arguments['path']}",
                description=(
                    "Write the exact approved memory content hash "
                    f"{arguments['content_sha256']}."
                ),
                target=str(arguments["path"]),
                risk="medium",
                arguments=arguments,
            )
            return NodeOutcome(status="waiting_approval", approval_id=approval.id)
        try:
            self.workspace_store.consume_approval(
                approval_id,
                workspace_id=run.workspace_id,
                run_id=run.id,
                tool="memory_write",
                arguments=arguments,
                consumed_by=f"graph:{run.id}:{node.id}",
            )
        except WorkspaceStoreError as exc:
            return NodeOutcome(status="failed", error=str(exc))
        return True

    def _execute_loop(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        node: GraphNode,
        state: Mapping[str, Any],
        *,
        approval_id: str | None,
    ) -> NodeOutcome:
        maximum = int(node.config["max_iterations"])
        body = dict(node.config["body"])
        body_type = str(body["type"])
        body_approval = dict(body.get("approval") or {})
        if bool(node.approval.get("required")) or node.id in set(
            graph.approval_requirements.get("required_node_ids") or []
        ):
            body_approval["required"] = True
        body_node = GraphNode(
            id=f"{node.id}-body",
            type=body_type,
            name=f"{node.name} body",
            config=dict(body.get("config") or {}),
            timeout_seconds=node.timeout_seconds,
            retry=dict(body.get("retry") or {}),
            approval=body_approval,
        )
        completed = int(state.get("iterations_completed") or 0)
        outputs = list(state.get("loop_outputs") or [])
        iteration_metrics = list(state.get("iteration_metrics") or [])
        retry_generation = int(state.get("side_effect_retry_generation") or 0)
        body_has_side_effect = self._node_has_side_effect(body_node)
        prior_active_seconds = float(state.get("loop_active_elapsed_seconds") or 0.0)
        invocation_started = time.monotonic()

        def active_seconds() -> float:
            return prior_active_seconds + max(
                0.0,
                time.monotonic() - invocation_started,
            )

        def checkpoint(
            index: int,
            *,
            active_iteration: int | None,
            active_body_state: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
            return {
                "iterations_completed": index,
                "loop_outputs": outputs,
                "iteration_metrics": iteration_metrics,
                "active_iteration": active_iteration,
                "active_body_state": dict(active_body_state or {}),
                "loop_active_elapsed_seconds": active_seconds(),
            }

        for index in range(completed, maximum):
            current = self.graph_store.get_run(run.id)
            if current.status == "cancelled":
                return NodeOutcome(
                    status="cancelled",
                    error="Loop cancelled",
                    state_updates=checkpoint(index, active_iteration=index + 1),
                )
            if current.pause_requested:
                return NodeOutcome(
                    status="paused",
                    state_updates=checkpoint(index, active_iteration=index + 1),
                )
            if self._deadline_exceeded(current, graph):
                return NodeOutcome(
                    status="failed",
                    error="Graph run timeout exceeded inside Loop",
                    state_updates=checkpoint(index, active_iteration=index + 1),
                )
            if node.timeout_seconds is not None and active_seconds() >= node.timeout_seconds:
                return NodeOutcome(
                    status="failed",
                    error=(
                        f"Loop node timeout exceeded: {node.timeout_seconds} seconds"
                    ),
                    state_updates=checkpoint(index, active_iteration=index + 1),
                )
            steps = int(current.resource_metrics.get("steps_completed") or 0)
            if steps >= int(graph.limits["max_steps"]):
                return NodeOutcome(
                    status="failed",
                    error=f"Graph step limit exceeded inside Loop: {graph.limits['max_steps']}",
                    state_updates=checkpoint(index, active_iteration=index + 1),
                )
            resource = self._check_resources(current, graph, body_node)
            current = self.graph_store.get_run(run.id)
            if resource is not None:
                return NodeOutcome(
                    status=("paused" if resource.status == "waiting_approval" else "failed"),
                    error=resource.error,
                    state_updates=checkpoint(index, active_iteration=index + 1),
                )
            context = self._context(current)
            context = {**context, "loop": {"index": index, "outputs": outputs}}
            base_key = (
                f"{run.id}:{node.id}:iteration:{index + 1}:"
                f"generation:{retry_generation}"
            )
            active_iteration = int(state.get("active_iteration") or 0)
            active_body_state = (
                dict(state.get("active_body_state") or {})
                if active_iteration == index + 1
                else {}
            )
            body_approval_id = approval_id if index == completed else None
            body_attempt = int(active_body_state.get("attempts") or 0)
            resuming_approval = bool(
                body_approval_id
                and body_approval_id == active_body_state.get("pending_approval_id")
            )
            while True:
                if not resuming_approval:
                    body_attempt += 1
                resuming_approval = False
                attempt_node = body_node
                if node.timeout_seconds is not None:
                    remaining_timeout = node.timeout_seconds - active_seconds()
                    if remaining_timeout <= 0:
                        return NodeOutcome(
                            status="failed",
                            error=(
                                f"Loop node timeout exceeded: "
                                f"{node.timeout_seconds} seconds"
                            ),
                            state_updates=checkpoint(
                                index,
                                active_iteration=index + 1,
                                active_body_state=active_body_state,
                            ),
                        )
                    attempt_node = replace(
                        body_node,
                        timeout_seconds=max(1, int(remaining_timeout + 0.999)),
                    )
                execution_key = (
                    base_key
                    if body_has_side_effect
                    else f"{base_key}:attempt:{body_attempt}"
                )
                try:
                    outcome = self._execute_inline(
                        current,
                        graph,
                        attempt_node,
                        context,
                        active_body_state,
                        execution_key=execution_key,
                        approval_id=body_approval_id,
                    )
                except Exception as exc:
                    outcome = NodeOutcome(
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        retryable=not body_has_side_effect,
                    )
                if outcome.status == "completed":
                    break
                body_state = {
                    **active_body_state,
                    **outcome.state_updates,
                    "attempts": body_attempt,
                    "pending_approval_id": outcome.approval_id,
                }
                if (
                    outcome.status == "failed"
                    and outcome.retryable
                    and body_attempt < self._max_attempts(graph, body_node)
                ):
                    backoff = self._retry_backoff(graph, body_node)
                    self.workspace_store.append_event(
                        run.id,
                        "loop_body_retry_scheduled",
                        {
                            "node_id": node.id,
                            "iteration": index + 1,
                            "attempt": body_attempt + 1,
                            "backoff_seconds": backoff,
                        },
                    )
                    if not self._wait_retry_backoff(run.id, backoff):
                        latest = self.graph_store.get_run(run.id)
                        return NodeOutcome(
                            status=(
                                "cancelled"
                                if latest.status == "cancelled"
                                else "paused"
                            ),
                            error=(
                                "Loop cancelled"
                                if latest.status == "cancelled"
                                else None
                            ),
                            state_updates=checkpoint(
                                index,
                                active_iteration=index + 1,
                                active_body_state=body_state,
                            ),
                        )
                    body_approval_id = None
                    active_body_state = body_state
                    continue
                return NodeOutcome(
                    status=outcome.status,
                    output=outputs,
                    error=outcome.error,
                    metrics=outcome.metrics,
                    approval_id=outcome.approval_id,
                    retryable=outcome.retryable,
                    state_updates=checkpoint(
                        index,
                        active_iteration=index + 1,
                        active_body_state=body_state,
                    ),
                )
            outputs.append(outcome.output)
            iteration_metrics.append(
                {**dict(outcome.metrics), "attempts": body_attempt}
            )
            latest = self.graph_store.get_run(run.id)
            states = {key_: dict(value) for key_, value in latest.node_states.items()}
            loop_state = dict(states[node.id])
            loop_state.update(
                checkpoint(index + 1, active_iteration=None)
            )
            states[node.id] = loop_state
            metrics = dict(latest.resource_metrics)
            metrics["steps_completed"] = int(metrics.get("steps_completed") or 0) + 1
            metrics["loop_body_steps_completed"] = int(
                metrics.get("loop_body_steps_completed") or 0
            ) + 1
            self.graph_store.update_run(
                latest.id,
                node_states=states,
                resource_metrics=metrics,
            )
            self.workspace_store.append_event(
                run.id,
                "loop_iteration_completed",
                {
                    "node_id": node.id,
                    "iteration": index + 1,
                    "attempts": body_attempt,
                    "steps_completed": metrics["steps_completed"],
                },
            )
            latest = self.graph_store.get_run(run.id)
            if self._deadline_exceeded(latest, graph):
                return NodeOutcome(
                    status="failed",
                    error="Graph run timeout exceeded after Loop iteration",
                    state_updates=checkpoint(index + 1, active_iteration=None),
                )
            if node.timeout_seconds is not None and active_seconds() >= node.timeout_seconds:
                return NodeOutcome(
                    status="failed",
                    error=(
                        f"Loop node timeout exceeded: {node.timeout_seconds} seconds"
                    ),
                    state_updates=checkpoint(index + 1, active_iteration=None),
                )
            until = node.config.get("until")
            if isinstance(until, Mapping):
                selector = str(until.get("path") or "loop.output")
                probe_context = {
                    **self._context(self.graph_store.get_run(run.id)),
                    "loop": {"index": index, "output": outcome.output, "outputs": outputs},
                }
                if self._condition_matches(
                    self._lookup(probe_context, selector),
                    until,
                ):
                    break
        return NodeOutcome(
            status="completed",
            output=outputs,
            metrics={
                "iterations": len(outputs),
                "iteration_metrics": iteration_metrics,
            },
            state_updates={
                "iterations_completed": len(outputs),
                "loop_outputs": outputs,
                "iteration_metrics": iteration_metrics,
                "active_iteration": None,
                "active_body_state": {},
                "loop_active_elapsed_seconds": active_seconds(),
            },
        )

    def _execute_inline(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        node: GraphNode,
        context: Mapping[str, Any],
        state: Mapping[str, Any],
        *,
        execution_key: str,
        approval_id: str | None,
    ) -> NodeOutcome:
        if node.type == "model":
            return self._execute_model(run, graph, node, context)
        if node.type == "tool":
            return self._execute_tool(
                run,
                graph,
                node,
                context,
                execution_key=execution_key,
                approval_id=approval_id,
            )
        if node.type == "memory_read":
            return self._execute_memory_read(run, node, context)
        if node.type == "memory_write":
            return self._execute_memory_write(
                run,
                graph,
                node,
                context,
                approval_id=approval_id,
            )
        if node.type == "memory_curate":
            return self._execute_memory_curate(
                run,
                graph,
                node,
                context,
                state,
                approval_id=approval_id,
            )
        return NodeOutcome(status="failed", error=f"unsupported loop body: {node.type}")

    def _check_resources(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        node: GraphNode,
    ) -> NodeOutcome | None:
        try:
            snapshot = dict(self.resource_snapshot() or {})
        except Exception as exc:
            snapshot = {"snapshot_error": f"{type(exc).__name__}: {exc}"}
        metrics = dict(run.resource_metrics)
        metrics["latest_resource_snapshot"] = snapshot
        metrics["latest_resource_snapshot_at"] = utc_now()
        self.graph_store.update_run(run.id, resource_metrics=metrics)
        self.workspace_store.append_event(
            run.id,
            "graph_resource_snapshot",
            {"node_id": node.id, **snapshot},
        )
        requirements = graph.runtime_requirements
        if snapshot.get("snapshot_error"):
            return NodeOutcome(
                status="failed",
                error=f"MTPLX runtime telemetry failed: {snapshot['snapshot_error']}",
            )
        snapshot_provider = str(snapshot.get("provider") or "").strip().lower()
        if not snapshot_provider:
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing provider",
            )
        if snapshot_provider != str(requirements.get("provider") or "mtplx"):
            return NodeOutcome(
                status="failed",
                error=f"runtime provider is not MTPLX: {snapshot_provider}",
            )
        required_model = str(requirements.get("model") or "").strip()
        allowed_models = {
            str(item) for item in requirements.get("allowed_models") or []
        }
        strict_pinned_model = bool(
            run.pinned_model
            and not bool(requirements.get("allow_model_fallback", True))
        )
        node_needs_model = node.type in {"model", "memory_curate"}
        if node.type == "loop":
            body = node.config.get("body")
            node_needs_model = bool(
                isinstance(body, Mapping)
                and str(body.get("type") or "") in {"model", "memory_curate"}
            )
        loaded_state_required = bool(
            requirements.get("require_loaded_model")
            or required_model
            or allowed_models
            or strict_pinned_model
            or node_needs_model
        )
        if loaded_state_required and not isinstance(snapshot.get("runtime_loaded"), bool):
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing runtime_loaded",
            )
        if loaded_state_required and snapshot.get("runtime_loaded") is not True:
            return NodeOutcome(status="failed", error="MTPLX has no loaded model")
        backend = str(snapshot.get("backend_id") or "").strip()
        required_backend = str(requirements.get("backend") or "").strip()
        allowed_backends = {
            str(item) for item in requirements.get("allowed_backends") or []
        }
        if (
            (required_backend and required_backend not in {"auto", "mtplx"})
            or allowed_backends
        ) and not backend:
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing backend_id",
            )
        if (
            required_backend
            and required_backend not in {"auto", "mtplx"}
            and backend != required_backend
        ):
            return NodeOutcome(
                status="failed",
                error=(
                    f"loaded MTPLX backend {backend} does not satisfy "
                    f"required backend {required_backend}"
                ),
            )
        if allowed_backends and backend not in allowed_backends:
            return NodeOutcome(
                status="failed",
                error=f"loaded MTPLX backend is not allowed: {backend}",
            )
        loaded_model = str(snapshot.get("loaded_model") or "").strip()
        if (required_model or allowed_models or strict_pinned_model) and not loaded_model:
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing loaded_model",
            )
        if required_model and loaded_model != required_model:
            return NodeOutcome(
                status="failed",
                error=(
                    f"loaded MTPLX model {loaded_model} does not satisfy required model "
                    f"{required_model}"
                ),
            )
        if allowed_models and loaded_model not in allowed_models:
            return NodeOutcome(
                status="failed",
                error=f"loaded MTPLX model is not allowed: {loaded_model}",
            )
        if (
            strict_pinned_model
            and loaded_model != run.pinned_model
        ):
            return NodeOutcome(
                status="failed",
                error=(
                    f"loaded MTPLX model {loaded_model} does not match pinned model "
                    f"{run.pinned_model}"
                ),
            )
        required_capabilities = {
            str(item) for item in requirements.get("required_capabilities") or []
        }
        available_capabilities = {
            str(item) for item in snapshot.get("runtime_capabilities") or []
        }
        if required_capabilities and not isinstance(
            snapshot.get("runtime_capabilities"), (list, tuple, set)
        ):
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing runtime_capabilities",
            )
        missing_capabilities = sorted(
            required_capabilities - available_capabilities
        )
        if missing_capabilities:
            return NodeOutcome(
                status="failed",
                error=(
                    "loaded MTPLX runtime lacks required capabilities: "
                    + ", ".join(missing_capabilities)
                ),
            )
        if not isinstance(snapshot.get("memory_pressure_level"), (int, float)) or isinstance(
            snapshot.get("memory_pressure_level"), bool
        ):
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing memory_pressure_level",
            )
        if not isinstance(snapshot.get("thermal_throttled"), bool):
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing thermal_throttled",
            )
        pressure = int(snapshot["memory_pressure_level"])
        if pressure >= 2:
            return NodeOutcome(
                status="waiting_approval",
                error="elevated Apple Silicon memory pressure",
            )
        if bool(snapshot.get("thermal_throttled")):
            return NodeOutcome(status="waiting_approval", error="thermal throttling active")
        maximum_memory = int(graph.limits.get("max_memory_bytes") or 0)
        if maximum_memory and (
            not isinstance(snapshot.get("active_memory_bytes"), (int, float))
            or isinstance(snapshot.get("active_memory_bytes"), bool)
        ):
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing active_memory_bytes",
            )
        active_memory = int(snapshot.get("active_memory_bytes") or 0)
        if maximum_memory and active_memory > maximum_memory:
            return NodeOutcome(
                status="failed",
                error=(
                    f"active memory {active_memory} exceeds Graph budget {maximum_memory}"
                ),
            )
        if (
            not isinstance(snapshot.get("context_window"), (int, float))
            or isinstance(snapshot.get("context_window"), bool)
            or int(snapshot.get("context_window") or 0) <= 0
        ):
            return NodeOutcome(
                status="failed",
                error="MTPLX runtime telemetry is missing context_window",
            )
        runtime_context = int(snapshot["context_window"])
        minimum_context = int(requirements.get("min_context_tokens") or 0)
        if minimum_context and runtime_context < minimum_context:
            return NodeOutcome(
                status="failed",
                error=(
                    f"loaded MTPLX context {runtime_context} is below required context "
                    f"{minimum_context}"
                ),
            )
        if int(graph.limits["max_context_tokens"]) > runtime_context:
            return NodeOutcome(
                status="failed",
                error=(
                    f"Graph context budget {graph.limits['max_context_tokens']} exceeds "
                    f"loaded MTPLX context {runtime_context}"
                ),
            )
        required_profile = str(requirements.get("profile") or "").strip()
        allowed_profiles = {
            str(item) for item in requirements.get("allowed_profiles") or []
        }
        selected_profile = str(run.runtime_profile or "auto")
        profile_contract = (
            required_profile
            if required_profile and required_profile != "auto"
            else selected_profile if selected_profile != "auto" else ""
        )
        if profile_contract or allowed_profiles:
            actual_profile = str(snapshot.get("runtime_profile") or "").strip()
            if not actual_profile:
                return NodeOutcome(
                    status="failed",
                    error="MTPLX runtime telemetry is missing runtime_profile",
                )
            if profile_contract and actual_profile != profile_contract:
                return NodeOutcome(
                    status="failed",
                    error=(
                        f"loaded MTPLX profile {actual_profile} does not satisfy "
                        f"required profile {profile_contract}"
                    ),
                )
            if allowed_profiles and actual_profile not in allowed_profiles:
                return NodeOutcome(
                    status="failed",
                    error=f"loaded MTPLX profile is not allowed: {actual_profile}",
                )
        return None

    def _next_node(
        self,
        graph: GraphDefinition,
        node: GraphNode,
        output: Any,
        run: GraphRun,
    ) -> str | None:
        outgoing = [edge for edge in graph.edges if edge.source == node.id]
        if not outgoing:
            return None
        if node.type != "conditional":
            return outgoing[0].target
        context = {**self._context(run), "condition": output}
        default_target = None
        for edge in outgoing:
            condition = edge.condition or {}
            if condition.get("default"):
                default_target = edge.target
                continue
            selector = str(condition.get("path") or "condition")
            if self._condition_matches(self._lookup(context, selector), condition):
                return edge.target
        return default_target

    @staticmethod
    def _condition_matches(value: Any, condition: Mapping[str, Any]) -> bool:
        if "equals" in condition:
            return value == condition["equals"]
        if "not_equals" in condition:
            return value != condition["not_equals"]
        if "in" in condition and isinstance(condition["in"], list):
            return value in condition["in"]
        if condition.get("truthy") is True:
            return bool(value)
        if condition.get("falsy") is True:
            return not bool(value)
        return False

    def _complete_run(
        self,
        run: GraphRun,
        graph: GraphDefinition,
        output: Any,
        *,
        output_state: Mapping[str, Any],
        output_metrics: Mapping[str, Any],
        attempt: int,
        already_checkpointed: bool = False,
    ) -> GraphRun:
        states = {key: dict(value) for key, value in run.node_states.items()}
        output_node = next(node for node in graph.nodes if node.type == "output")
        outputs = output if isinstance(output, Mapping) else {"result": output}
        try:
            validate_graph_contract_value(
                graph.outputs,
                dict(outputs),
                field_name="outputs",
            )
        except GraphValidationError as exc:
            error = str(exc)
            failed = dict(states[output_node.id])
            failed.update(dict(output_state))
            failed.update(
                {
                    "status": "failed",
                    "error": error,
                    "completed_at": utc_now(),
                }
            )
            states[output_node.id] = failed
            run = self.graph_store.update_run(
                run.id,
                node_states=states,
                resource_metrics=self._suspend_active_metrics(run),
            )
            self.workspace_store.append_event(
                run.id,
                "graph_output_contract_failed",
                {"node_id": output_node.id, "error": error},
            )
            return self._fail_run(run, error)
        completed_output = dict(states[output_node.id])
        completed_output.update(dict(output_state))
        completed_output.update(
            {
                "status": "completed",
                "output": output,
                "error": None,
                "completed_at": utc_now(),
                "pending_approval_id": None,
                "metrics": dict(output_metrics),
                "recovery_guard": None,
            }
        )
        states[output_node.id] = completed_output
        for state in states.values():
            if state.get("status") == "pending":
                state["status"] = "skipped"
                state["completed_at"] = utc_now()
        metrics = dict(run.resource_metrics)
        if not already_checkpointed:
            metrics["steps_completed"] = int(metrics.get("steps_completed") or 0) + 1
        metrics["last_node_metrics"] = dict(output_metrics)
        metrics["last_completed_node_id"] = output_node.id
        metrics["completion_prepared"] = True
        metrics["completion_prepared_at"] = utc_now()
        prepared = self.graph_store.update_run(
            run.id,
            expected_state_version=run.state_version,
            outputs=dict(outputs),
            node_states=states,
            current_node_id=output_node.id,
            pending_approval_id=None,
            resource_metrics=metrics,
            pause_requested=False,
            error=None,
        )
        events = self.workspace_store.list_events(run.id, limit=5000)
        output_event_exists = any(
            event.kind == "graph_node_completed"
            and event.payload.get("node_id") == output_node.id
            for event in events
        )
        completion_event_exists = any(
            event.kind == "graph_completed" for event in events
        )
        if not output_event_exists:
            self.workspace_store.append_event(
                run.id,
                "graph_node_completed",
                {
                    "node_id": output_node.id,
                    "node_type": output_node.type,
                    "attempt": attempt,
                    "metrics": dict(output_metrics),
                },
            )
        if not completion_event_exists:
            self.workspace_store.append_event(
                run.id,
                "graph_completed",
                {
                    "graph_id": graph.id,
                    "graph_revision": graph.revision,
                    "outputs": dict(outputs),
                    "successful_nodes": [
                        key
                        for key, state in states.items()
                        if state.get("status") == "completed"
                    ],
                },
            )
        try:
            self.workspace_store.update_run(
                run.id,
                status="completed",
                error="",
            )
        except WorkspaceStoreError as exc:
            failed_metrics = dict(prepared.resource_metrics)
            failed_metrics["mirror_sync_error"] = f"{type(exc).__name__}: {exc}"
            self.graph_store.update_run(
                prepared.id,
                expected_state_version=prepared.state_version,
                resource_metrics=failed_metrics,
            )
            return self.graph_store.get_run(prepared.id)
        current = self.graph_store.get_run(prepared.id)
        final_metrics = dict(current.resource_metrics)
        final_metrics.pop("completion_prepared", None)
        final_metrics["completion_committed_at"] = utc_now()
        updated = self.graph_store.update_run(
            current.id,
            expected_state_version=current.state_version,
            status="completed",
            resource_metrics=self._suspend_active_metrics(
                replace(current, resource_metrics=final_metrics)
            ),
            pause_requested=False,
            error=None,
        )
        return updated

    def _fail_node_and_run(
        self,
        run: GraphRun,
        node: GraphNode,
        state: Mapping[str, Any],
        error: str,
        *,
        persist_node: bool = True,
    ) -> GraphRun:
        if persist_node:
            states = {key: dict(value) for key, value in run.node_states.items()}
            failed = dict(state)
            failed.update({"status": "failed", "error": error, "completed_at": utc_now()})
            states[node.id] = failed
            run = self.graph_store.update_run(run.id, node_states=states)
        self.workspace_store.append_event(
            run.id,
            "graph_node_failed",
            {"node_id": node.id, "node_type": node.type, "error": error},
        )
        return self._fail_run(run, error)

    def _fail_run(self, run: GraphRun, error: str) -> GraphRun:
        updated = self.graph_store.update_run(
            run.id,
            status="failed",
            resource_metrics=self._suspend_active_metrics(run),
            pause_requested=False,
            error=str(error),
        )
        self.workspace_store.append_event(
            run.id,
            "graph_failed",
            {
                "kind": "graph",
                "graph_id": run.graph_id,
                "current_node_id": run.current_node_id,
                "error": str(error),
            },
        )
        return updated

    def _pinned_graph(self, run: GraphRun) -> GraphDefinition:
        graph = self.graph_store.get(run.graph_id, revision=run.graph_revision)
        if graph.content_sha256 != run.graph_sha256:
            raise GraphExecutionError("pinned Graph revision hash mismatch")
        return graph

    def _context(self, run: GraphRun) -> dict[str, Any]:
        node_outputs = {
            node_id: {"output": state.get("output"), "status": state.get("status")}
            for node_id, state in run.node_states.items()
        }
        last_output = None
        if run.current_node_id:
            last_output = run.node_states.get(run.current_node_id, {}).get("output")
        if last_output is None:
            last_completed = str(
                run.resource_metrics.get("last_completed_node_id") or ""
            )
            if last_completed:
                last_output = run.node_states.get(last_completed, {}).get("output")
        if last_output is None:
            completed = [
                state
                for state in run.node_states.values()
                if state.get("status") == "completed"
            ]
            if completed:
                last_output = completed[-1].get("output")
        return {
            "inputs": dict(run.inputs),
            "outputs": dict(run.outputs),
            "nodes": node_outputs,
            "last_output": last_output,
            "run": run.to_dict(),
        }

    def _render(self, value: Any, context: Mapping[str, Any]) -> Any:
        if isinstance(value, Mapping):
            return {str(key): self._render(item, context) for key, item in value.items()}
        if isinstance(value, list):
            return [self._render(item, context) for item in value]
        if not isinstance(value, str):
            return value
        full = _TEMPLATE.fullmatch(value)
        if full:
            return self._lookup(context, full.group(1))

        def replace(match: re.Match[str]) -> str:
            item = self._lookup(context, match.group(1))
            if isinstance(item, (dict, list)):
                return json.dumps(item, ensure_ascii=False, sort_keys=True)
            return "" if item is None else str(item)

        return _TEMPLATE.sub(replace, value)

    @staticmethod
    def _lookup(context: Mapping[str, Any], path: str) -> Any:
        value: Any = context
        for part in str(path or "").split("."):
            if not part:
                continue
            if isinstance(value, Mapping):
                value = value.get(part)
            elif isinstance(value, list) and part.isdigit():
                index = int(part)
                value = value[index] if 0 <= index < len(value) else None
            else:
                return None
        return value

    @staticmethod
    def _model_text(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        first = choices[0]
        if not isinstance(first, Mapping):
            return ""
        message = first.get("message")
        if not isinstance(message, Mapping):
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text") or "")
                for item in content
                if isinstance(item, Mapping)
            )
        return str(content or "")

    def _http_model_runner(self, **kwargs: Any) -> Mapping[str, Any]:
        body = {
            "model": kwargs.get("model"),
            "messages": kwargs.get("messages") or [],
            "max_tokens": int(kwargs.get("max_tokens") or 1024),
            "stream": False,
            "metadata": {
                "graph_run_id": kwargs.get("graph_run_id"),
                "graph_node_id": kwargs.get("node_id"),
                "runtime_profile": kwargs.get("runtime_profile"),
            },
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-MTPLX-Session-Id": str(kwargs.get("session_id") or ""),
            "X-MTPLX-Request-Id": str(kwargs.get("request_id") or ""),
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(
                request,
                timeout=max(1, min(int(kwargs.get("timeout_seconds") or 180), 3600)),
            ) as response:
                value = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:2000]
            raise GraphExecutionError(
                f"MTPLX model request failed ({exc.code}): {detail}"
            ) from exc
        except (URLError, OSError, TimeoutError, json.JSONDecodeError) as exc:
            raise GraphExecutionError(f"MTPLX model request unavailable: {exc}") from exc
        if not isinstance(value, Mapping) or value.get("error"):
            raise GraphExecutionError(f"MTPLX model returned an error: {value}")
        return value

    def _cancel_model_request(self, request_id: str) -> bool:
        if not self._uses_http_model_runner or not request_id:
            return False
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/v1/mtplx/cancel/{request_id}",
            data=b"{}",
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=2) as response:
                value = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError):
            return False
        return bool(isinstance(value, Mapping) and value.get("cancelled"))

    @staticmethod
    def _memory_principal(run: GraphRun) -> MemoryPrincipal:
        return MemoryPrincipal(
            agent_id=f"project-{safe_id(run.project_id, fallback='workspace')}",
            session_id=run.id,
        )

    def _side_effect_approval_required(
        self,
        graph: GraphDefinition,
        node: GraphNode,
        *,
        tool: str | None = None,
        category: str | None = None,
    ) -> bool:
        requirements = graph.approval_requirements
        if bool(node.approval.get("required")):
            return True
        if node.id in set(requirements.get("required_node_ids") or []):
            return True
        if tool and tool in set(requirements.get("required_tool_names") or []):
            return True
        if category and category in set(
            requirements.get("required_policy_categories") or []
        ):
            return True
        if category == "memory" and bool(requirements.get("memory_writes")):
            return True
        return bool(requirements.get("all_side_effects")) and self._node_has_side_effect(
            node
        )

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    def _start_active_metrics(self, run: GraphRun) -> dict[str, Any]:
        metrics = dict(run.resource_metrics)
        if not metrics.get("active_started_at"):
            metrics["active_started_at"] = utc_now()
            metrics["active_resume_count"] = int(
                metrics.get("active_resume_count") or 0
            ) + 1
        metrics.setdefault("active_elapsed_seconds", 0.0)
        return metrics

    def _suspend_active_metrics(
        self,
        run: GraphRun,
        *,
        stop_at: str | None = None,
    ) -> dict[str, Any]:
        metrics = dict(run.resource_metrics)
        started = self._parse_timestamp(str(metrics.get("active_started_at") or ""))
        stopped = self._parse_timestamp(stop_at) or datetime.now(timezone.utc)
        elapsed = float(metrics.get("active_elapsed_seconds") or 0.0)
        if started is not None:
            elapsed += max(0.0, (stopped - started).total_seconds())
        metrics["active_elapsed_seconds"] = elapsed
        metrics["active_started_at"] = None
        return metrics

    def _active_elapsed_seconds(self, run: GraphRun) -> float:
        metrics = run.resource_metrics
        elapsed = float(metrics.get("active_elapsed_seconds") or 0.0)
        started = self._parse_timestamp(str(metrics.get("active_started_at") or ""))
        if started is not None:
            elapsed += max(
                0.0,
                (datetime.now(timezone.utc) - started).total_seconds(),
            )
        return elapsed

    @staticmethod
    def _retry_backoff(graph: GraphDefinition, node: GraphNode) -> float:
        value = node.retry.get(
            "backoff_seconds",
            graph.retry.get("backoff_seconds", 0),
        )
        return max(0.0, min(float(value or 0), 300.0))

    def _wait_retry_backoff(self, run_id: str, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            event = self._cancel_event(run_id)
            if event.wait(timeout=min(0.1, deadline - time.monotonic())):
                return False
            run = self.graph_store.get_run(run_id)
            if run.pause_requested or run.status in _TERMINAL_RUN_STATUSES:
                return False
        return True

    @staticmethod
    def _max_attempts(graph: GraphDefinition, node: GraphNode) -> int:
        value = node.retry.get("max_attempts", graph.retry.get("max_attempts", 1))
        return max(1, min(int(value or 1), 10))

    def _deadline_exceeded(self, run: GraphRun, graph: GraphDefinition) -> bool:
        return self._active_elapsed_seconds(run) > graph.timeout_seconds

    @staticmethod
    def _node_has_side_effect(node: GraphNode) -> bool:
        if node.type in SIDE_EFFECT_NODE_TYPES:
            if node.type == "tool":
                return str(node.config.get("tool") or "") in MUTATING_TOOLS
            return True
        if node.type == "loop":
            body = node.config.get("body")
            if isinstance(body, Mapping):
                body_type = str(body.get("type") or "")
                if body_type in {"memory_write", "memory_curate"}:
                    return True
                if body_type == "tool":
                    config = body.get("config")
                    return isinstance(config, Mapping) and str(config.get("tool") or "") in MUTATING_TOOLS
        return False


__all__ = ["GraphExecutionError", "GraphExecutor", "NodeOutcome"]
