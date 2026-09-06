import copy
import json

import pytest

from mtplx.agent_workspace import WorkspaceConflictError, WorkspaceStore
from mtplx.graphs import (
    GraphStore,
    GraphValidationError,
    _canonical_sha256,
    validate_graph_payload,
)


def _graph_payload(workspace_id):
    return {
        "schema_version": 1,
        "id": "verification-graph",
        "project_id": workspace_id,
        "workspace_id": workspace_id,
        "name": "Verification graph",
        "inputs": {"goal": {"type": "string"}},
        "outputs": {"answer": {"type": "string"}},
        "limits": {
            "max_steps": 20,
            "max_context_tokens": 32768,
            "max_concurrency": 1,
        },
        "policies": {"write": "ask", "terminal": "ask"},
        "runtime_requirements": {"backend": "mtplx", "profile": "auto"},
        "nodes": [
            {"id": "start", "type": "input", "name": "Input"},
            {
                "id": "model",
                "type": "model",
                "name": "Local model",
                "config": {"prompt": "Solve: {{inputs.goal}}", "max_tokens": 256},
            },
            {
                "id": "repeat",
                "type": "loop",
                "name": "Bounded loop",
                "config": {
                    "max_iterations": 2,
                    "body": {
                        "type": "memory_read",
                        "config": {"path": "shared/project.md"},
                    },
                },
            },
            {"id": "finish", "type": "output", "name": "Output"},
        ],
        "edges": [
            {"source": "start", "target": "model"},
            {"source": "model", "target": "repeat"},
            {"source": "repeat", "target": "finish"},
        ],
    }


def _conditional_graph_payload(workspace_id):
    payload = _graph_payload(workspace_id)
    payload["nodes"][2] = {
        "id": "route",
        "type": "conditional",
        "name": "Route",
        "config": {"selector": "last_output"},
    }
    payload["nodes"].insert(
        -1,
        {
            "id": "fallback",
            "type": "memory_read",
            "name": "Fallback",
            "config": {"path": "shared/fallback.md", "optional": True},
        },
    )
    payload["edges"] = [
        {"source": "start", "target": "model"},
        {"source": "model", "target": "route"},
        {
            "source": "route",
            "target": "finish",
            "condition": {"equals": True},
        },
        {
            "source": "route",
            "target": "fallback",
            "condition": {"default": True},
        },
        {"source": "fallback", "target": "finish"},
    ]
    return payload


def test_graph_schema_accepts_loop_but_rejects_general_cycles(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("Project", str(project))
    payload = _graph_payload(workspace.id)

    graph = validate_graph_payload(payload)
    assert graph.schema_version == 1
    assert graph.node_map["repeat"].type == "loop"
    assert graph.limits["max_concurrency"] == 1
    assert graph.content_sha256

    cyclic = copy.deepcopy(payload)
    cyclic["edges"].append({"source": "repeat", "target": "model"})
    with pytest.raises(GraphValidationError, match="cycles"):
        validate_graph_payload(cyclic)


def test_graph_schema_rejects_dangling_nodes_joins_and_nested_loops(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("Project", str(project))
    payload = _graph_payload(workspace.id)
    payload["nodes"].insert(
        -1,
        {
            "id": "join",
            "type": "memory_read",
            "config": {"path": "shared/project.md"},
        },
    )
    payload["edges"][-1] = {"source": "repeat", "target": "join"}
    payload["edges"].append({"source": "join", "target": "finish"})
    payload["edges"].append({"source": "model", "target": "join"})
    payload["edges"].append({"source": "missing", "target": "finish"})
    payload["nodes"][2]["config"]["body"] = {
        "type": "loop",
        "config": {"max_iterations": 2},
    }
    with pytest.raises(GraphValidationError) as error:
        validate_graph_payload(payload)
    message = str(error.value)
    assert "dangling source" in message
    assert "joins are not supported" in message
    assert "body type must be one of" in message


def test_graph_schema_validates_loop_body_and_policy_names(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("Project", str(project))
    payload = _graph_payload(workspace.id)
    payload["policies"]["typo-write"] = "allow"
    payload["nodes"][2]["config"]["body"] = {
        "type": "tool",
        "config": {"tool": "not-a-tool", "arguments": "wrong"},
    }
    with pytest.raises(GraphValidationError) as error:
        validate_graph_payload(payload)
    message = str(error.value)
    assert "unknown Graph policy key" in message
    assert "body uses unknown tool" in message
    assert "body tool arguments must be an object" in message


def test_graph_store_pins_revisions_hashes_and_run_checkpoints(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    workspace_store = WorkspaceStore(tmp_path / "state")
    workspace = workspace_store.create_workspace(
        "Project",
        str(project),
        model="local-model",
    )
    store = GraphStore(workspace_store)
    created = store.create(_graph_payload(workspace.id))
    assert created.revision == 1
    assert store.get(created.id).content_sha256 == created.content_sha256

    updated = store.update(
        created.id,
        {"description": "Second immutable revision"},
        expected_revision=1,
    )
    assert updated.revision == 2
    assert updated.content_sha256 != created.content_sha256
    assert store.get(created.id, revision=1).description == ""
    with pytest.raises(WorkspaceConflictError, match="revision conflict"):
        store.update(created.id, {"description": "stale"}, expected_revision=1)

    run = store.create_run(
        updated,
        inputs={"goal": "verify"},
        runtime_profile="balanced",
    )
    assert run.graph_revision == 2
    assert run.graph_sha256 == updated.content_sha256
    assert run.pinned_model == "local-model"
    assert workspace_store.get_run(run.id).id == run.id
    changed_states = copy.deepcopy(run.node_states)
    changed_states["start"]["status"] = "completed"
    checkpoint = store.update_run(
        run.id,
        expected_state_version=run.state_version,
        status="running",
        current_node_id="model",
        node_states=changed_states,
    )
    assert checkpoint.state_version == run.state_version + 1
    assert store.get_run(run.id).node_states["start"]["status"] == "completed"
    with pytest.raises(WorkspaceConflictError, match="state conflict"):
        store.update_run(
            run.id,
            expected_state_version=run.state_version,
            status="paused",
        )


def test_graph_store_loads_legacy_v1_revision_after_v2_serializer_upgrade(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    workspace_store = WorkspaceStore(tmp_path / "state")
    workspace = workspace_store.create_workspace("Project", str(project))
    store = GraphStore(workspace_store)
    created = store.create(_graph_payload(workspace.id))

    legacy = created.to_dict(include_hash=False)
    legacy.pop("schedule", None)
    legacy.pop("layout", None)
    for node in legacy["nodes"]:
        node.pop("priority", None)
    legacy["content_sha256"] = _canonical_sha256(legacy)
    store._revision_path(created.id, created.revision).write_text(
        json.dumps(legacy, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    loaded = store.get(created.id)
    assert loaded.schema_version == 1
    assert loaded.content_sha256 == legacy["content_sha256"]
    assert "schedule" not in loaded.to_dict(include_hash=False)
    assert "layout" not in loaded.to_dict(include_hash=False)
    assert "priority" not in loaded.to_dict(include_hash=False)["nodes"][0]


def test_graph_contracts_apply_defaults_and_reject_invalid_run_inputs(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    workspace_store = WorkspaceStore(tmp_path / "state")
    workspace = workspace_store.create_workspace("Project", str(project))
    store = GraphStore(workspace_store)
    payload = _graph_payload(workspace.id)
    payload["inputs"] = {
        "goal": {"type": "string", "minLength": 3},
        "attempts": {
            "type": "integer",
            "minimum": 1,
            "default": 2,
        },
    }
    graph = store.create(payload)

    run = store.create_run(graph, inputs={"goal": "ship it"})
    assert run.inputs == {"goal": "ship it", "attempts": 2}

    with pytest.raises(GraphValidationError, match="inputs.goal must be of type string"):
        store.create_run(graph, inputs={"goal": 42})
    with pytest.raises(GraphValidationError, match="inputs.extra is not declared"):
        store.create_run(graph, inputs={"goal": "ship it", "extra": True})
    assert len(workspace_store.list_runs(workspace.id)) == 1


def test_graph_contracts_accept_explicit_object_roots():
    payload = _graph_payload("workspace")
    payload["inputs"] = {
        "type": "object",
        "properties": {"goal": {"type": "string"}},
        "required": ["goal"],
        "additionalProperties": False,
    }
    payload["outputs"] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }

    graph = validate_graph_payload(payload)

    assert graph.inputs["type"] == "object"
    assert graph.outputs["type"] == "object"


@pytest.mark.parametrize("field_name", ["inputs", "outputs"])
def test_graph_contracts_reject_non_object_roots(field_name):
    payload = _graph_payload("workspace")
    payload[field_name] = {
        "type": "array",
        "items": {"type": "string"},
    }

    with pytest.raises(
        GraphValidationError,
        match=rf"{field_name} root type must be object",
    ):
        validate_graph_payload(payload)


def test_graph_contracts_reject_nullable_object_roots():
    payload = _graph_payload("workspace")
    payload["outputs"] = {
        "type": "object",
        "nullable": True,
        "properties": {"answer": {"type": "string"}},
    }

    with pytest.raises(GraphValidationError, match="outputs root may not be nullable"):
        validate_graph_payload(payload)


def test_graph_schema_rejects_approval_required_on_unsupported_nodes():
    payload = _graph_payload("workspace")
    payload["nodes"][1]["approval"] = {"required": True}

    with pytest.raises(
        GraphValidationError,
        match="node model type model does not support approval.required",
    ):
        validate_graph_payload(payload)


def test_conditional_schema_requires_exactly_one_default_edge():
    valid = _conditional_graph_payload("workspace")
    validate_graph_payload(valid)

    missing_default = copy.deepcopy(valid)
    missing_default["edges"][3]["condition"] = {"equals": False}
    with pytest.raises(
        GraphValidationError,
        match="conditional node route requires exactly one default edge",
    ):
        validate_graph_payload(missing_default)

    duplicate_default = copy.deepcopy(valid)
    duplicate_default["edges"][2]["condition"] = {"default": True}
    with pytest.raises(
        GraphValidationError,
        match="conditional node route requires exactly one default edge",
    ):
        validate_graph_payload(duplicate_default)


def test_conditional_schema_rejects_duplicate_predicates():
    payload = _conditional_graph_payload("workspace")
    payload["edges"][3]["condition"] = {"equals": True}
    payload["edges"].append(
        {
            "source": "route",
            "target": "finish",
            "condition": {"default": True},
        }
    )

    with pytest.raises(
        GraphValidationError,
        match="conditional node route has duplicate predicates",
    ):
        validate_graph_payload(payload)


def test_graph_schema_requires_project_and_workspace_identity_equality():
    payload = _graph_payload("workspace")
    graph = validate_graph_payload(payload)
    assert graph.project_id == graph.workspace_id == "workspace"

    payload["project_id"] = "different-project"
    with pytest.raises(
        GraphValidationError,
        match="schema version 1 requires project_id to equal workspace_id",
    ):
        validate_graph_payload(payload)


def test_model_node_override_requires_runtime_model_binding():
    payload = _graph_payload("workspace")
    payload["runtime_requirements"] = {
        "provider": "mtplx",
        "allowed_models": ["model-a"],
    }
    payload["nodes"][1]["config"]["model"] = "model-a"
    validate_graph_payload(payload)

    unbound = copy.deepcopy(payload)
    unbound["runtime_requirements"].pop("allowed_models")
    with pytest.raises(
        GraphValidationError,
        match="model node model model override requires runtime_requirements.allowed_models",
    ):
        validate_graph_payload(unbound)

    disallowed = copy.deepcopy(payload)
    disallowed["nodes"][1]["config"]["model"] = "model-b"
    with pytest.raises(
        GraphValidationError,
        match="model node model model is not in runtime_requirements.allowed_models",
    ):
        validate_graph_payload(disallowed)

    exact_mismatch = copy.deepcopy(payload)
    exact_mismatch["runtime_requirements"]["model"] = "model-a"
    exact_mismatch["runtime_requirements"]["allowed_models"] = ["model-a", "model-b"]
    exact_mismatch["nodes"][1]["config"]["model"] = "model-b"
    with pytest.raises(
        GraphValidationError,
        match="model node model model must equal runtime_requirements.model",
    ):
        validate_graph_payload(exact_mismatch)


def test_loop_model_body_override_uses_runtime_model_binding():
    payload = _graph_payload("workspace")
    payload["runtime_requirements"] = {
        "provider": "mtplx",
        "allowed_models": ["model-a"],
    }
    payload["nodes"][2]["config"]["body"] = {
        "type": "model",
        "config": {"prompt": "Refine the answer", "model": "model-a"},
    }
    validate_graph_payload(payload)

    payload["nodes"][2]["config"]["body"]["config"]["model"] = "model-b"
    with pytest.raises(
        GraphValidationError,
        match=(
            "loop node repeat model body model is not in "
            "runtime_requirements.allowed_models"
        ),
    ):
        validate_graph_payload(payload)


def test_graph_schema_limits_node_timeouts_to_supported_types():
    payload = _graph_payload("workspace")
    payload["nodes"][1]["timeout_seconds"] = 30
    payload["nodes"][2]["timeout_seconds"] = 60
    validate_graph_payload(payload)

    unsupported = copy.deepcopy(payload)
    unsupported["nodes"][0]["timeout_seconds"] = 10
    with pytest.raises(
        GraphValidationError,
        match="node start type input does not support timeout_seconds",
    ):
        validate_graph_payload(unsupported)

    out_of_range = copy.deepcopy(payload)
    out_of_range["nodes"][1]["timeout_seconds"] = 0
    with pytest.raises(
        GraphValidationError,
        match="node model timeout_seconds must be from 1 to 86400",
    ):
        validate_graph_payload(out_of_range)


def test_loop_body_accepts_valid_approval_and_retry_contracts():
    payload = _graph_payload("workspace")
    payload["nodes"][2]["config"]["body"] = {
        "type": "tool",
        "config": {
            "tool": "write_file",
            "arguments": {"path": "answer.txt", "content": "done"},
        },
        "approval": {"required": True},
        "retry": {"max_attempts": 2, "backoff_seconds": 0.25},
    }

    graph = validate_graph_payload(payload)

    body = graph.node_map["repeat"].config["body"]
    assert body["approval"] == {"required": True}
    assert body["retry"] == {"max_attempts": 2, "backoff_seconds": 0.25}


def test_loop_body_rejects_unsupported_approval_and_invalid_retry():
    payload = _graph_payload("workspace")
    payload["nodes"][2]["config"]["body"] = {
        "type": "memory_read",
        "config": {"path": "shared/project.md"},
        "approval": {"required": True, "unexpected": True},
        "retry": {
            "max_attempts": 0,
            "backoff_seconds": 301,
            "unexpected": True,
        },
    }

    with pytest.raises(GraphValidationError) as error:
        validate_graph_payload(payload)
    message = str(error.value)
    assert "loop node repeat body approval has unknown keys: unexpected" in message
    assert "body type memory_read does not support approval.required" in message
    assert "loop node repeat body retry has unknown keys: unexpected" in message
    assert "loop node repeat body retry.max_attempts must be from 1 to 10" in message
    assert "loop node repeat body retry.backoff_seconds must be from 0 to 300" in message


def test_loop_body_requires_approval_and_retry_objects():
    payload = _graph_payload("workspace")
    payload["nodes"][2]["config"]["body"]["approval"] = []
    payload["nodes"][2]["config"]["body"]["retry"] = []

    with pytest.raises(GraphValidationError) as error:
        validate_graph_payload(payload)
    message = str(error.value)
    assert "loop node repeat body approval must be an object" in message
    assert "loop node repeat body retry must be an object" in message


def test_graph_schema_rejects_unknown_fields_invalid_numbers_and_conditions(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("Project", str(project))
    payload = _graph_payload(workspace.id)
    payload["typo"] = True
    payload["limits"]["max_steps"] = "many"
    payload["nodes"][1]["timeout_seconds"] = "later"
    payload["nodes"].insert(
        -1,
        {
            "id": "route",
            "type": "conditional",
            "config": {"selector": "last_output"},
        },
    )
    payload["edges"][-1] = {"source": "repeat", "target": "route"}
    payload["edges"].extend(
        [
            {
                "source": "route",
                "target": "finish",
                "condition": {"equals": True, "truthy": True},
            },
            {
                "source": "route",
                "target": "finish",
                "condition": {"default": True, "equals": False},
            },
        ]
    )
    with pytest.raises(GraphValidationError) as error:
        validate_graph_payload(payload)
    message = str(error.value)
    assert "unknown Graph fields: typo" in message
    assert "limits.max_steps must be an integer" in message
    assert "node model timeout_seconds must be an integer" in message
    assert "requires exactly one predicate" in message
    assert "may contain only default" in message


def test_graph_schema_validates_runtime_and_approval_requirements(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    store = WorkspaceStore(tmp_path / "state")
    workspace = store.create_workspace("Project", str(project))
    payload = _graph_payload(workspace.id)
    payload["runtime_requirements"] = {
        "provider": "other",
        "allowed_models": "not-an-array",
        "unknown": True,
    }
    payload["approval_requirements"] = {
        "required_node_ids": ["missing", "model"],
        "required_tool_names": ["missing_tool"],
        "required_policy_categories": ["missing_category"],
        "unknown": True,
    }
    with pytest.raises(GraphValidationError) as error:
        validate_graph_payload(payload)
    message = str(error.value)
    assert "runtime_requirements.provider must be mtplx" in message
    assert "runtime_requirements has unknown keys" in message
    assert "approval_requirements has unknown keys" in message
    assert "references unknown nodes: missing" in message
    assert "node model is not a side-effect node" in message
    assert "has unknown tools: missing_tool" in message
    assert "has unknown categories: missing_category" in message
