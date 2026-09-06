"""Versioned Graph definitions and durable Graph run state for MTPLX.

Graphs are the agent workflow engine. They are deliberately separate from
``mtplx.graphbank``, which remains an inference optimization subsystem.
Iteration is represented only by the bounded ``loop`` node type.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agent_workspace import (
    POLICY_MODES,
    WorkspaceConflictError,
    WorkspaceStore,
    WorkspaceStoreError,
    _atomic_write,
    safe_id,
    utc_now,
)
from .workspace_tools import (
    FIRST_PARTY_TOOL_NAMES,
    MUTATING_TOOLS,
    first_party_tool_definitions,
)


GRAPH_SCHEMA_VERSION = 1
GRAPH_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
GRAPH_NODE_TYPES = (
    "input",
    "output",
    "loop",
    "model",
    "tool",
    "conditional",
    "human_approval",
    "memory_read",
    "memory_write",
    "memory_curate",
    "join",
)
GRAPH_RUN_STATUSES = (
    "queued",
    "running",
    "waiting_approval",
    "paused",
    "completed",
    "failed",
    "cancelled",
)
GRAPH_NODE_STATUSES = (
    "pending",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "skipped",
    "cancelled",
)
_GRAPH_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"queued", "running", "paused", "failed", "cancelled"}),
    "running": frozenset(
        {"running", "waiting_approval", "paused", "completed", "failed", "cancelled"}
    ),
    "waiting_approval": frozenset(
        {"waiting_approval", "queued", "paused", "failed", "cancelled"}
    ),
    "paused": frozenset({"paused", "queued", "failed", "cancelled"}),
    "completed": frozenset({"completed"}),
    "failed": frozenset({"failed", "queued"}),
    "cancelled": frozenset({"cancelled"}),
}
LOOP_BODY_TYPES = frozenset(
    {"model", "tool", "memory_read", "memory_write", "memory_curate"}
)
SIDE_EFFECT_NODE_TYPES = frozenset({"tool", "memory_write", "memory_curate"})
GRAPH_POLICY_KEYS = frozenset(
    {"read", "search", "write", "terminal", "browser", "network", "memory"}
)
GRAPH_TOP_LEVEL_KEYS = frozenset(
    {
        "id",
        "project_id",
        "workspace_id",
        "name",
        "description",
        "schema_version",
        "revision",
        "inputs",
        "outputs",
        "nodes",
        "edges",
        "limits",
        "policies",
        "runtime_requirements",
        "retry",
        "timeout_seconds",
        "approval_requirements",
        "schedule",
        "layout",
        "created_at",
        "updated_at",
        "content_sha256",
    }
)
GRAPH_RUNTIME_REQUIREMENT_KEYS = frozenset(
    {
        "provider",
        "backend",
        "allowed_backends",
        "model",
        "allowed_models",
        "profile",
        "allowed_profiles",
        "required_capabilities",
        "require_loaded_model",
        "allow_model_fallback",
        "min_context_tokens",
    }
)
GRAPH_APPROVAL_REQUIREMENT_KEYS = frozenset(
    {
        "required_node_ids",
        "required_tool_names",
        "required_policy_categories",
        "all_side_effects",
        "memory_writes",
    }
)
_CONTRACT_SCHEMA_KEYS = frozenset(
    {
        "$schema",
        "type",
        "description",
        "default",
        "enum",
        "const",
        "nullable",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
    }
)
_CONTRACT_TYPES = frozenset(
    {"object", "array", "string", "integer", "number", "boolean", "null"}
)
_TOOL_PARAMETER_SCHEMAS = {
    str(item["function"]["name"]): dict(item["function"]["parameters"])
    for item in first_party_tool_definitions()
}
_NODE_CONFIG_KEYS: dict[str, frozenset[str]] = {
    "input": frozenset(),
    "output": frozenset({"mapping"}),
    "loop": frozenset({"max_iterations", "body", "until"}),
    "model": frozenset({"prompt", "prompt_path", "max_tokens", "model"}),
    "tool": frozenset({"tool", "arguments"}),
    "conditional": frozenset({"selector"}),
    "human_approval": frozenset(
        {"action", "description", "payload", "risk", "expires_in_seconds"}
    ),
    "memory_read": frozenset({"path", "optional"}),
    "memory_write": frozenset({"path", "content", "expected_sha256"}),
    "memory_curate": frozenset(
        {"path", "expected_sha256", "query", "max_context_chars", "max_tokens"}
    ),
    "join": frozenset({"mode", "mapping"}),
}


class GraphError(WorkspaceStoreError):
    pass


class GraphValidationError(GraphError):
    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = tuple(str(item) for item in issues)
        super().__init__("invalid graph: " + "; ".join(self.issues))


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_object(value: Any, *, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise GraphValidationError([f"{field_name} must be an object"])
    try:
        return json.loads(
            json.dumps(dict(value), allow_nan=False, ensure_ascii=False)
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GraphValidationError([f"{field_name} must be JSON-compatible"]) from exc


def _validated_int(
    value: Any,
    *,
    field_name: str,
    default: int,
    issues: list[str],
) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        issues.append(f"{field_name} must be an integer")
        return int(default)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        issues.append(f"{field_name} must be an integer")
        return int(default)
    if isinstance(value, str) and re.fullmatch(r"[+-]?[0-9]+", value.strip()):
        try:
            return int(value)
        except (ValueError, OverflowError):
            pass
    issues.append(f"{field_name} must be an integer")
    return int(default)


def _string_list(
    value: Any,
    *,
    field_name: str,
    issues: list[str],
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        issues.append(f"{field_name} must be an array of strings")
        return []
    return [str(item) for item in value]


def _contract_root_schema(
    contract: Mapping[str, Any],
    *,
    field_name: str,
    issues: list[str],
) -> dict[str, Any]:
    """Convert the compact field map or object JSON schema into one object schema."""
    if not contract:
        return {}
    full_schema = (
        isinstance(contract.get("type"), (str, list))
        or "properties" in contract
        or isinstance(contract.get("required"), list)
        or "additionalProperties" in contract
        or "$schema" in contract
    )
    if full_schema:
        schema = dict(contract)
        schema.setdefault("type", "object")
    else:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for key, raw in contract.items():
            name = str(key)
            if not isinstance(raw, Mapping):
                issues.append(f"{field_name}.{name} must be a schema object")
                continue
            item = dict(raw)
            required_value = item.pop("required", True)
            if not isinstance(required_value, bool):
                issues.append(f"{field_name}.{name}.required must be a boolean")
                required_value = True
            properties[name] = item
            if required_value and "default" not in item:
                required.append(name)
        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    raw_root_type = schema.get("type")
    root_types = (
        [raw_root_type]
        if isinstance(raw_root_type, str)
        else list(raw_root_type or [])
    )
    if root_types != ["object"]:
        issues.append(f"{field_name} root type must be object")
    if bool(schema.get("nullable")):
        issues.append(f"{field_name} root may not be nullable")
    _validate_contract_schema(schema, path=field_name, issues=issues, depth=0)
    return schema


def _validate_contract_schema(
    schema: Mapping[str, Any],
    *,
    path: str,
    issues: list[str],
    depth: int,
) -> None:
    if depth > 12:
        issues.append(f"{path} schema nesting exceeds 12 levels")
        return
    unknown = set(schema) - _CONTRACT_SCHEMA_KEYS
    if unknown:
        issues.append(f"{path} has unsupported schema keys: {', '.join(sorted(unknown))}")
    raw_type = schema.get("type")
    types: list[str] = []
    if raw_type is not None:
        if isinstance(raw_type, str):
            types = [raw_type]
        elif isinstance(raw_type, list) and raw_type and all(
            isinstance(item, str) for item in raw_type
        ):
            types = [str(item) for item in raw_type]
        else:
            issues.append(f"{path}.type must be a string or non-empty string array")
        for item in types:
            if item not in _CONTRACT_TYPES:
                issues.append(f"{path}.type has unsupported value: {item}")
    if "enum" in schema and not isinstance(schema.get("enum"), list):
        issues.append(f"{path}.enum must be an array")
    if "nullable" in schema and not isinstance(schema.get("nullable"), bool):
        issues.append(f"{path}.nullable must be a boolean")
    properties = schema.get("properties")
    if properties is not None:
        if not isinstance(properties, Mapping):
            issues.append(f"{path}.properties must be an object")
        else:
            for key, raw in properties.items():
                if not isinstance(raw, Mapping):
                    issues.append(f"{path}.properties.{key} must be a schema object")
                    continue
                _validate_contract_schema(
                    raw,
                    path=f"{path}.{key}",
                    issues=issues,
                    depth=depth + 1,
                )
    required = schema.get("required")
    if required is not None:
        values = _string_list(required, field_name=f"{path}.required", issues=issues)
        if len(values) != len(set(values)):
            issues.append(f"{path}.required may not contain duplicates")
        if isinstance(properties, Mapping):
            missing = sorted(set(values) - {str(key) for key in properties})
            if missing:
                issues.append(
                    f"{path}.required references unknown properties: {', '.join(missing)}"
                )
    additional = schema.get("additionalProperties")
    if additional is not None and not isinstance(additional, (bool, Mapping)):
        issues.append(f"{path}.additionalProperties must be a boolean or schema object")
    elif isinstance(additional, Mapping):
        _validate_contract_schema(
            additional,
            path=f"{path}.additionalProperties",
            issues=issues,
            depth=depth + 1,
        )
    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            issues.append(f"{path}.items must be a schema object")
        else:
            _validate_contract_schema(
                items,
                path=f"{path}.items",
                issues=issues,
                depth=depth + 1,
            )
    for key in ("minimum", "maximum"):
        if key in schema and (
            isinstance(schema[key], bool) or not isinstance(schema[key], (int, float))
        ):
            issues.append(f"{path}.{key} must be a number")
    for key in ("minLength", "maxLength", "minItems", "maxItems"):
        if key in schema and (
            isinstance(schema[key], bool)
            or not isinstance(schema[key], int)
            or int(schema[key]) < 0
        ):
            issues.append(f"{path}.{key} must be a non-negative integer")
    if (
        isinstance(schema.get("minimum"), (int, float))
        and isinstance(schema.get("maximum"), (int, float))
        and not isinstance(schema.get("minimum"), bool)
        and not isinstance(schema.get("maximum"), bool)
        and schema["minimum"] > schema["maximum"]
    ):
        issues.append(f"{path}.minimum may not exceed maximum")
    if (
        isinstance(schema.get("minLength"), int)
        and isinstance(schema.get("maxLength"), int)
        and schema["minLength"] > schema["maxLength"]
    ):
        issues.append(f"{path}.minLength may not exceed maxLength")
    if (
        isinstance(schema.get("minItems"), int)
        and isinstance(schema.get("maxItems"), int)
        and schema["minItems"] > schema["maxItems"]
    ):
        issues.append(f"{path}.minItems may not exceed maxItems")
    if "pattern" in schema:
        if not isinstance(schema.get("pattern"), str):
            issues.append(f"{path}.pattern must be a string")
        else:
            try:
                re.compile(str(schema["pattern"]))
            except re.error as exc:
                issues.append(f"{path}.pattern is invalid: {exc}")
    if "default" in schema:
        _validate_contract_value(
            schema,
            schema["default"],
            path=f"{path}.default",
            issues=issues,
        )


def _value_matches_contract_type(value: Any, type_name: str) -> bool:
    if type_name == "null":
        return value is None
    if type_name == "object":
        return isinstance(value, Mapping)
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def _validate_contract_value(
    schema: Mapping[str, Any],
    value: Any,
    *,
    path: str,
    issues: list[str],
) -> None:
    if value is None and bool(schema.get("nullable")):
        return
    raw_type = schema.get("type")
    types = [raw_type] if isinstance(raw_type, str) else list(raw_type or [])
    if types and not any(_value_matches_contract_type(value, str(item)) for item in types):
        issues.append(f"{path} must be of type {' or '.join(str(item) for item in types)}")
        return
    if "const" in schema and value != schema["const"]:
        issues.append(f"{path} must equal its declared constant")
    if isinstance(schema.get("enum"), list) and value not in schema["enum"]:
        issues.append(f"{path} is not one of the declared enum values")
    if isinstance(value, Mapping):
        properties = schema.get("properties")
        property_map = dict(properties) if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        for key in required if isinstance(required, list) else []:
            if str(key) not in value:
                issues.append(f"{path}.{key} is required")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            name = str(key)
            child = property_map.get(name)
            if isinstance(child, Mapping):
                _validate_contract_value(child, item, path=f"{path}.{name}", issues=issues)
            elif additional is False:
                issues.append(f"{path}.{name} is not declared")
            elif isinstance(additional, Mapping):
                _validate_contract_value(
                    additional,
                    item,
                    path=f"{path}.{name}",
                    issues=issues,
                )
    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            issues.append(f"{path} requires at least {minimum} items")
        if isinstance(maximum, int) and len(value) > maximum:
            issues.append(f"{path} allows at most {maximum} items")
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                _validate_contract_value(
                    items,
                    item,
                    path=f"{path}.{index}",
                    issues=issues,
                )
    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            issues.append(f"{path} requires at least {minimum} characters")
        if isinstance(maximum, int) and len(value) > maximum:
            issues.append(f"{path} allows at most {maximum} characters")
        pattern = schema.get("pattern")
        if isinstance(pattern, str) and re.search(pattern, value) is None:
            issues.append(f"{path} does not match the declared pattern")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            issues.append(f"{path} must be at least {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            issues.append(f"{path} must be at most {maximum}")


def apply_graph_contract_defaults(
    contract: Mapping[str, Any],
    value: Mapping[str, Any] | None,
    *,
    field_name: str,
) -> dict[str, Any]:
    issues: list[str] = []
    schema = _contract_root_schema(contract, field_name=field_name, issues=issues)
    if issues:
        raise GraphValidationError(issues)
    result = _json_object(value, field_name=field_name)
    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        for key, raw in properties.items():
            if str(key) not in result and isinstance(raw, Mapping) and "default" in raw:
                result[str(key)] = json.loads(
                    json.dumps(raw["default"], allow_nan=False, ensure_ascii=False)
                )
    return result


def validate_graph_contract_value(
    contract: Mapping[str, Any],
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> None:
    if not contract:
        return
    issues: list[str] = []
    schema = _contract_root_schema(contract, field_name=field_name, issues=issues)
    if not issues:
        _validate_contract_value(schema, value, path=field_name, issues=issues)
    if issues:
        raise GraphValidationError(issues)


def _validate_retry_policy(
    retry: Mapping[str, Any],
    *,
    field_name: str,
    issues: list[str],
    include_defaults: bool = False,
) -> dict[str, Any]:
    result = dict(retry)
    unknown = set(result) - {"max_attempts", "backoff_seconds"}
    if unknown:
        issues.append(f"{field_name} has unknown keys: {', '.join(sorted(unknown))}")
    normalized: dict[str, Any] = {}
    if "max_attempts" in result or include_defaults:
        maximum = _validated_int(
            result.get("max_attempts"),
            field_name=f"{field_name}.max_attempts",
            default=1,
            issues=issues,
        )
        if not 1 <= maximum <= 10:
            issues.append(f"{field_name}.max_attempts must be from 1 to 10")
        normalized["max_attempts"] = maximum
    if "backoff_seconds" in result or include_defaults:
        backoff = result.get("backoff_seconds", 0)
        if isinstance(backoff, bool) or not isinstance(backoff, (int, float)):
            issues.append(f"{field_name}.backoff_seconds must be a number")
            backoff = 0
        backoff_value = float(backoff)
        if not 0 <= backoff_value <= 300:
            issues.append(f"{field_name}.backoff_seconds must be from 0 to 300")
        normalized["backoff_seconds"] = backoff_value
    return normalized


def _validate_node_approval(
    approval: Mapping[str, Any],
    *,
    field_name: str,
    issues: list[str],
) -> dict[str, Any]:
    result = dict(approval)
    unknown = set(result) - {"required"}
    if unknown:
        issues.append(f"{field_name} has unknown keys: {', '.join(sorted(unknown))}")
    required = result.get("required", False)
    if not isinstance(required, bool):
        issues.append(f"{field_name}.required must be a boolean")
        required = False
    return {"required": bool(required)}


def _validate_tool_arguments(
    tool: str,
    arguments: Any,
    *,
    field_name: str,
    issues: list[str],
) -> None:
    if arguments is not None and not isinstance(arguments, Mapping):
        issues.append(f"{field_name} must be an object")
        return
    if tool not in _TOOL_PARAMETER_SCHEMAS:
        return
    values = dict(arguments or {})
    schema = _TOOL_PARAMETER_SCHEMAS[tool]
    required = {str(item) for item in schema.get("required") or []}
    missing = sorted(required - set(values))
    if missing:
        issues.append(f"{field_name} is missing required keys: {', '.join(missing)}")
    properties = schema.get("properties")
    allowed = {str(key) for key in properties} if isinstance(properties, Mapping) else set()
    unknown = sorted(set(values) - allowed)
    if unknown:
        issues.append(f"{field_name} has unknown keys: {', '.join(unknown)}")
    if "network" in values and not isinstance(values["network"], bool):
        issues.append(f"{field_name}.network must be a boolean")
    if "timeout_seconds" in values:
        timeout = _validated_int(
            values["timeout_seconds"],
            field_name=f"{field_name}.timeout_seconds",
            default=1,
            issues=issues,
        )
        maximum = 900 if tool == "run_tests" else 300
        if not 1 <= timeout <= maximum:
            issues.append(
                f"{field_name}.timeout_seconds must be from 1 to {maximum}"
            )


def _validate_node_config(
    node_type: str,
    config: Mapping[str, Any],
    *,
    field_name: str,
    issues: list[str],
) -> None:
    allowed = _NODE_CONFIG_KEYS.get(node_type)
    if allowed is not None:
        unknown = sorted(set(config) - allowed)
        if unknown:
            issues.append(f"{field_name} has unknown keys: {', '.join(unknown)}")
    if node_type == "model" and "max_tokens" in config:
        maximum = _validated_int(
            config["max_tokens"],
            field_name=f"{field_name}.max_tokens",
            default=1,
            issues=issues,
        )
        if not 1 <= maximum <= 16_384:
            issues.append(f"{field_name}.max_tokens must be from 1 to 16384")
    if node_type == "human_approval" and "expires_in_seconds" in config:
        expires = _validated_int(
            config["expires_in_seconds"],
            field_name=f"{field_name}.expires_in_seconds",
            default=1,
            issues=issues,
        )
        if not 1 <= expires <= 86_400:
            issues.append(
                f"{field_name}.expires_in_seconds must be from 1 to 86400"
            )
    if node_type == "memory_read" and "optional" in config and not isinstance(
        config["optional"], bool
    ):
        issues.append(f"{field_name}.optional must be a boolean")
    if node_type == "memory_curate":
        for key, upper in (("max_context_chars", 1_000_000), ("max_tokens", 16_384)):
            if key not in config:
                continue
            value = _validated_int(
                config[key],
                field_name=f"{field_name}.{key}",
                default=1,
                issues=issues,
            )
            if not 1 <= value <= upper:
                issues.append(f"{field_name}.{key} must be from 1 to {upper}")
    if node_type == "join":
        mode = str(config.get("mode") or "all").strip().lower()
        if mode not in {"all", "any"}:
            issues.append(f"{field_name}.mode must be all or any")
        mapping = config.get("mapping")
        if mapping is not None and not isinstance(mapping, Mapping):
            issues.append(f"{field_name}.mapping must be an object")


@dataclass(frozen=True)
class GraphNode:
    id: str
    type: str
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int | None = None
    retry: dict[str, Any] = field(default_factory=dict)
    approval: dict[str, Any] = field(default_factory=dict)
    priority: int = 0

    def to_dict(self, *, schema_version: int = GRAPH_SCHEMA_VERSION) -> dict[str, Any]:
        value = {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "config": dict(self.config),
            "timeout_seconds": self.timeout_seconds,
            "retry": dict(self.retry),
            "approval": dict(self.approval),
        }
        if schema_version >= 2:
            value["priority"] = self.priority
        return value


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    condition: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "condition": dict(self.condition) if self.condition is not None else None,
        }


@dataclass(frozen=True)
class GraphDefinition:
    id: str
    project_id: str
    workspace_id: str
    name: str
    description: str
    schema_version: int
    revision: int
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    limits: dict[str, Any]
    policies: dict[str, str]
    runtime_requirements: dict[str, Any]
    retry: dict[str, Any]
    timeout_seconds: int
    approval_requirements: dict[str, Any]
    schedule: dict[str, Any]
    layout: dict[str, Any]
    created_at: str
    updated_at: str
    content_sha256: str

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "id": self.id,
            "project_id": self.project_id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "schema_version": self.schema_version,
            "revision": self.revision,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "nodes": [
                node.to_dict(schema_version=self.schema_version) for node in self.nodes
            ],
            "edges": [edge.to_dict() for edge in self.edges],
            "limits": dict(self.limits),
            "policies": dict(self.policies),
            "runtime_requirements": dict(self.runtime_requirements),
            "retry": dict(self.retry),
            "timeout_seconds": self.timeout_seconds,
            "approval_requirements": dict(self.approval_requirements),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.schema_version >= 2:
            value["schedule"] = dict(self.schedule)
            value["layout"] = dict(self.layout)
        if include_hash:
            value["content_sha256"] = self.content_sha256
        return value

    @property
    def node_map(self) -> dict[str, GraphNode]:
        return {node.id: node for node in self.nodes}


@dataclass(frozen=True)
class GraphRun:
    id: str
    graph_id: str
    graph_revision: int
    graph_sha256: str
    workspace_id: str
    project_id: str
    workspace_root: str
    status: str
    pinned_model: str | None
    runtime_profile: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    node_states: dict[str, dict[str, Any]]
    current_node_id: str | None
    pending_approval_id: str | None
    resource_metrics: dict[str, Any]
    created_at: str
    updated_at: str
    state_version: int
    pause_requested: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "graph_id": self.graph_id,
            "graph_revision": self.graph_revision,
            "graph_sha256": self.graph_sha256,
            "workspace_id": self.workspace_id,
            "project_id": self.project_id,
            "workspace_root": self.workspace_root,
            "status": self.status,
            "pinned_model": self.pinned_model,
            "runtime_profile": self.runtime_profile,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "node_states": {key: dict(value) for key, value in self.node_states.items()},
            "current_node_id": self.current_node_id,
            "pending_approval_id": self.pending_approval_id,
            "resource_metrics": dict(self.resource_metrics),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state_version": self.state_version,
            "pause_requested": self.pause_requested,
            "error": self.error,
        }


def validate_graph_payload(
    payload: Mapping[str, Any],
    *,
    graph_id: str | None = None,
    revision: int | None = None,
    created_at: str | None = None,
    updated_at: str | None = None,
) -> GraphDefinition:
    """Validate and normalize one immutable Graph revision."""
    issues: list[str] = []
    unknown_top_level = set(payload) - GRAPH_TOP_LEVEL_KEYS
    if unknown_top_level:
        issues.append(
            "unknown Graph fields: " + ", ".join(sorted(unknown_top_level))
        )
    schema_version = _validated_int(
        payload.get("schema_version"),
        field_name="schema_version",
        default=GRAPH_SCHEMA_VERSION,
        issues=issues,
    )
    if schema_version not in GRAPH_SUPPORTED_SCHEMA_VERSIONS:
        issues.append(
            "schema_version must be one of: "
            + ", ".join(str(item) for item in sorted(GRAPH_SUPPORTED_SCHEMA_VERSIONS))
            + f", found {schema_version}"
        )
    identifier = safe_id(
        graph_id or str(payload.get("id") or f"graph_{uuid.uuid4().hex}"),
        fallback="graph",
    )
    workspace_id = safe_id(str(payload.get("workspace_id") or ""), fallback="")
    project_id = safe_id(
        str(payload.get("project_id") or workspace_id),
        fallback="",
    )
    if not workspace_id:
        issues.append("workspace_id is required")
    if not project_id:
        issues.append("project_id is required")
    elif workspace_id and project_id != workspace_id:
        issues.append(
            f"schema version {schema_version} requires project_id to equal workspace_id"
        )
    name = str(payload.get("name") or "").strip()
    if not name:
        issues.append("name is required")

    inputs = _json_object(payload.get("inputs"), field_name="inputs")
    outputs = _json_object(payload.get("outputs"), field_name="outputs")
    _contract_root_schema(inputs, field_name="inputs", issues=issues)
    _contract_root_schema(outputs, field_name="outputs", issues=issues)

    raw_nodes = payload.get("nodes")
    raw_edges = payload.get("edges")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        issues.append("nodes must be a non-empty array")
        raw_nodes = []
    if not isinstance(raw_edges, list):
        issues.append("edges must be an array")
        raw_edges = []
    if len(raw_nodes) > 200:
        issues.append("graphs may contain at most 200 nodes")
    if len(raw_edges) > 400:
        issues.append("graphs may contain at most 400 edges")

    nodes: list[GraphNode] = []
    node_ids: set[str] = set()
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, Mapping):
            issues.append(f"nodes[{index}] must be an object")
            continue
        node_id = safe_id(str(raw.get("id") or ""), fallback="")
        node_type = str(raw.get("type") or "").strip().lower()
        if not node_id:
            issues.append(f"nodes[{index}].id is required")
            continue
        if node_id in node_ids:
            issues.append(f"duplicate node id: {node_id}")
            continue
        node_ids.add(node_id)
        if node_type not in GRAPH_NODE_TYPES:
            issues.append(f"node {node_id} has unknown type: {node_type}")
        config = (
            _json_object(raw.get("config"), field_name=f"node {node_id} config")
            if isinstance(raw.get("config"), Mapping)
            else {}
        )
        if raw.get("config") is not None and not isinstance(raw.get("config"), Mapping):
            issues.append(f"node {node_id} config must be an object")
        _validate_node_config(
            node_type,
            config,
            field_name=f"node {node_id} config",
            issues=issues,
        )
        unknown_node_fields = set(raw) - {
            "id",
            "type",
            "name",
            "config",
            "timeout_seconds",
            "retry",
            "approval",
            "priority",
        }
        if unknown_node_fields:
            issues.append(
                f"node {node_id} has unknown fields: "
                + ", ".join(sorted(unknown_node_fields))
            )
        retry_value = raw.get("retry")
        if retry_value is not None and not isinstance(retry_value, Mapping):
            issues.append(f"node {node_id} retry must be an object")
        retry_config = _validate_retry_policy(
            dict(retry_value) if isinstance(retry_value, Mapping) else {},
            field_name=f"node {node_id} retry",
            issues=issues,
        )
        approval_value = raw.get("approval")
        if approval_value is not None and not isinstance(approval_value, Mapping):
            issues.append(f"node {node_id} approval must be an object")
        approval = _validate_node_approval(
            dict(approval_value) if isinstance(approval_value, Mapping) else {},
            field_name=f"node {node_id} approval",
            issues=issues,
        )
        priority = _validated_int(
            raw.get("priority", 0),
            field_name=f"node {node_id} priority",
            default=0,
            issues=issues,
        )
        if not -100 <= priority <= 100:
            issues.append(f"node {node_id} priority must be from -100 to 100")
        timeout = raw.get("timeout_seconds")
        timeout_value = (
            _validated_int(
                timeout,
                field_name=f"node {node_id} timeout_seconds",
                default=1,
                issues=issues,
            )
            if timeout is not None
            else None
        )
        if timeout_value is not None and not 1 <= timeout_value <= 86_400:
            issues.append(f"node {node_id} timeout_seconds must be from 1 to 86400")
        if timeout_value is not None and node_type not in {
            "model",
            "tool",
            "loop",
            "memory_curate",
        }:
            issues.append(
                f"node {node_id} type {node_type} does not support timeout_seconds"
            )
        if bool(approval.get("required")) and node_type not in {
            "tool",
            "loop",
            "memory_write",
            "memory_curate",
        }:
            issues.append(
                f"node {node_id} type {node_type} does not support approval.required"
            )
        nodes.append(
            GraphNode(
                id=node_id,
                type=node_type,
                name=str(raw.get("name") or node_id),
                config=config,
                timeout_seconds=timeout_value,
                retry=retry_config,
                approval=approval,
                priority=priority,
            )
        )

    edges: list[GraphEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    outgoing: dict[str, list[GraphEdge]] = {node_id: [] for node_id in node_ids}
    incoming: dict[str, list[GraphEdge]] = {node_id: [] for node_id in node_ids}
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, Mapping):
            issues.append(f"edges[{index}] must be an object")
            continue
        source = safe_id(str(raw.get("source") or ""), fallback="")
        target = safe_id(str(raw.get("target") or ""), fallback="")
        unknown_edge_fields = set(raw) - {"source", "target", "condition"}
        if unknown_edge_fields:
            issues.append(
                f"edge {index} has unknown fields: "
                + ", ".join(sorted(unknown_edge_fields))
            )
        condition_value = raw.get("condition")
        condition = (
            _json_object(
                condition_value,
                field_name=f"edge {index} condition",
            )
            if isinstance(condition_value, Mapping)
            else None
        )
        if condition_value is not None and not isinstance(condition_value, Mapping):
            issues.append(f"edge {index} condition must be an object")
        if source not in node_ids:
            issues.append(f"edge {index} has dangling source: {source}")
        if target not in node_ids:
            issues.append(f"edge {index} has dangling target: {target}")
        if source == target and source:
            issues.append(f"self-cycle is not allowed: {source}")
        edge = GraphEdge(source=source, target=target, condition=condition)
        edge_key = (
            source,
            target,
            json.dumps(condition, sort_keys=True, default=str),
        )
        if edge_key in edge_keys:
            issues.append(f"duplicate edge: {source}->{target}")
        edge_keys.add(edge_key)
        edges.append(edge)
        if source in outgoing:
            outgoing[source].append(edge)
        if target in incoming:
            incoming[target].append(edge)

    input_nodes = [node for node in nodes if node.type == "input"]
    output_nodes = [node for node in nodes if node.type == "output"]
    if len(input_nodes) != 1:
        issues.append("a graph requires exactly one input node")
    if len(output_nodes) != 1:
        issues.append("a graph requires exactly one output node")
    if input_nodes and incoming.get(input_nodes[0].id):
        issues.append("the input node may not have incoming edges")
    if output_nodes and outgoing.get(output_nodes[0].id):
        issues.append("the output node may not have outgoing edges")

    for node in nodes:
        node_outgoing = outgoing.get(node.id, [])
        node_incoming = incoming.get(node.id, [])
        if node.type != "output" and not node_outgoing:
            issues.append(f"node {node.id} is a dead end")
        if (
            schema_version == 1
            and node.type != "conditional"
            and len(node_outgoing) > 1
        ):
            issues.append(
                f"node {node.id} branches but is not a conditional node"
            )
        if (
            schema_version == 1
            and node.type not in {"input", "output"}
            and len(node_incoming) > 1
        ):
            issues.append(
                f"node {node.id} has multiple incoming edges; joins are not supported yet"
            )
        if (
            schema_version == 2
            and node.type != "join"
            and len(node_incoming) > 1
        ):
            issues.append(
                f"node {node.id} has multiple incoming edges; use a join node"
            )
        if node.type == "join" and len(node_incoming) < 2:
            issues.append(f"join node {node.id} requires at least two incoming edges")
        if node.type == "conditional":
            if len(node_outgoing) < 1:
                issues.append(f"conditional node {node.id} requires outgoing edges")
            default_count = 0
            predicate_keys: set[str] = set()
            for edge in node_outgoing:
                if edge.condition is None:
                    issues.append(
                        f"conditional edge {node.id}->{edge.target} requires a condition"
                    )
                elif bool(edge.condition.get("default")):
                    default_count += 1
                    if set(edge.condition) != {"default"}:
                        issues.append(
                            f"conditional default edge {node.id}->{edge.target} "
                            "may contain only default"
                        )
                else:
                    predicate_key = json.dumps(
                        edge.condition,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if predicate_key in predicate_keys:
                        issues.append(
                            f"conditional node {node.id} has duplicate predicates"
                        )
                    predicate_keys.add(predicate_key)
                    operators = {
                        key
                        for key in ("equals", "not_equals", "in", "truthy", "falsy")
                        if key in edge.condition
                    }
                    unknown = set(edge.condition) - {
                        "path",
                        "equals",
                        "not_equals",
                        "in",
                        "truthy",
                        "falsy",
                    }
                    if unknown:
                        issues.append(
                            f"conditional edge {node.id}->{edge.target} has unknown "
                            f"keys: {', '.join(sorted(unknown))}"
                        )
                    if len(operators) != 1:
                        issues.append(
                            f"conditional edge {node.id}->{edge.target} requires "
                            "exactly one predicate"
                        )
                    if "in" in edge.condition and not isinstance(
                        edge.condition.get("in"), list
                    ):
                        issues.append(
                            f"conditional edge {node.id}->{edge.target} in must be an array"
                        )
                    for key in ("truthy", "falsy"):
                        if key in edge.condition and edge.condition[key] is not True:
                            issues.append(
                                f"conditional edge {node.id}->{edge.target} {key} must be true"
                            )
            if default_count != 1:
                issues.append(
                    f"conditional node {node.id} requires exactly one default edge"
                )
        elif any(edge.condition is not None for edge in node_outgoing):
            issues.append(f"only conditional nodes may use conditional edges: {node.id}")

        config = node.config
        if node.type == "loop":
            maximum = _validated_int(
                config.get("max_iterations"),
                field_name=f"loop node {node.id} max_iterations",
                default=0,
                issues=issues,
            )
            body = config.get("body")
            until = config.get("until")
            if until is not None and not isinstance(until, Mapping):
                issues.append(f"loop node {node.id} until must be an object")
            elif isinstance(until, Mapping):
                operators = {
                    key
                    for key in ("equals", "not_equals", "in", "truthy", "falsy")
                    if key in until
                }
                unknown_until = set(until) - {
                    "path",
                    "equals",
                    "not_equals",
                    "in",
                    "truthy",
                    "falsy",
                }
                if unknown_until:
                    issues.append(
                        f"loop node {node.id} until has unknown keys: "
                        + ", ".join(sorted(unknown_until))
                    )
                if len(operators) != 1:
                    issues.append(
                        f"loop node {node.id} until requires exactly one predicate"
                    )
                if "in" in until and not isinstance(until.get("in"), list):
                    issues.append(f"loop node {node.id} until in must be an array")
            if not 1 <= maximum <= 100:
                issues.append(f"loop node {node.id} max_iterations must be from 1 to 100")
            if not isinstance(body, Mapping):
                issues.append(f"loop node {node.id} requires one inline body object")
            else:
                unknown_body_fields = set(body) - {
                    "type",
                    "config",
                    "retry",
                    "approval",
                }
                if unknown_body_fields:
                    issues.append(
                        f"loop node {node.id} body has unknown fields: "
                        + ", ".join(sorted(unknown_body_fields))
                    )
                body_type = str(body.get("type") or "").strip().lower()
                if body_type not in LOOP_BODY_TYPES:
                    issues.append(
                        f"loop node {node.id} body type must be one of: "
                        + ", ".join(sorted(LOOP_BODY_TYPES))
                    )
                body_config_value = body.get("config")
                body_config = (
                    _json_object(
                        body_config_value,
                        field_name=f"loop node {node.id} body config",
                    )
                    if isinstance(body_config_value, Mapping)
                    else {}
                )
                if body_config_value is not None and not isinstance(
                    body_config_value, Mapping
                ):
                    issues.append(f"loop node {node.id} body config must be an object")
                _validate_node_config(
                    body_type,
                    body_config,
                    field_name=f"loop node {node.id} body config",
                    issues=issues,
                )
                body_retry_value = body.get("retry")
                if body_retry_value is not None and not isinstance(
                    body_retry_value, Mapping
                ):
                    issues.append(f"loop node {node.id} body retry must be an object")
                _validate_retry_policy(
                    dict(body_retry_value)
                    if isinstance(body_retry_value, Mapping)
                    else {},
                    field_name=f"loop node {node.id} body retry",
                    issues=issues,
                )
                body_approval_value = body.get("approval")
                if body_approval_value is not None and not isinstance(
                    body_approval_value, Mapping
                ):
                    issues.append(f"loop node {node.id} body approval must be an object")
                _validate_node_approval(
                    dict(body_approval_value)
                    if isinstance(body_approval_value, Mapping)
                    else {},
                    field_name=f"loop node {node.id} body approval",
                    issues=issues,
                )
                body_approval = (
                    dict(body_approval_value)
                    if isinstance(body_approval_value, Mapping)
                    else {}
                )
                body_supports_approval = body_type in {
                    "tool",
                    "memory_write",
                    "memory_curate",
                }
                if bool(body_approval.get("required")) and not body_supports_approval:
                    issues.append(
                        f"loop node {node.id} body type {body_type} does not support "
                        "approval.required"
                    )
                if bool(node.approval.get("required")) and not body_supports_approval:
                    issues.append(
                        f"loop node {node.id} approval.required needs a side-effect body"
                    )
                if body_type == "tool":
                    tool = str(body_config.get("tool") or "")
                    if tool not in FIRST_PARTY_TOOL_NAMES:
                        issues.append(
                            f"loop node {node.id} body uses unknown tool: {tool}"
                        )
                    _validate_tool_arguments(
                        tool,
                        body_config.get("arguments"),
                        field_name=f"loop node {node.id} body tool arguments",
                        issues=issues,
                    )
                elif body_type == "model":
                    if not str(
                        body_config.get("prompt") or body_config.get("prompt_path") or ""
                    ).strip():
                        issues.append(
                            f"loop node {node.id} model body requires prompt or prompt_path"
                        )
                elif body_type in {"memory_read", "memory_write", "memory_curate"}:
                    if not str(body_config.get("path") or "").strip():
                        issues.append(
                            f"loop node {node.id} {body_type} body requires path"
                        )
                    if body_type in {"memory_write", "memory_curate"} and (
                        "expected_sha256" not in body_config
                    ):
                        issues.append(
                            f"loop node {node.id} {body_type} body requires "
                            "expected_sha256 for conflict detection"
                        )
        elif node.type == "tool":
            tool = str(config.get("tool") or "")
            if tool not in FIRST_PARTY_TOOL_NAMES:
                issues.append(f"tool node {node.id} uses unknown tool: {tool}")
            _validate_tool_arguments(
                tool,
                config.get("arguments"),
                field_name=f"tool node {node.id} arguments",
                issues=issues,
            )
        elif node.type == "model":
            if not str(config.get("prompt") or config.get("prompt_path") or "").strip():
                issues.append(f"model node {node.id} requires prompt or prompt_path")
        elif node.type in {"memory_read", "memory_write", "memory_curate"}:
            if not str(config.get("path") or "").strip():
                issues.append(f"{node.type} node {node.id} requires path")
            if node.type in {"memory_write", "memory_curate"} and (
                "expected_sha256" not in config
            ):
                issues.append(
                    f"{node.type} node {node.id} requires expected_sha256 "
                    "for conflict detection"
                )
        elif node.type == "human_approval":
            if not str(config.get("action") or "").strip():
                issues.append(f"human approval node {node.id} requires action")
            if str(config.get("risk") or "medium") not in {
                "low",
                "medium",
                "high",
                "critical",
            }:
                issues.append(
                    f"human approval node {node.id} risk must be low, medium, high, or critical"
                )

    # General graph cycles are rejected. Loop iterations are internal to one node.
    indegree = {node_id: len(incoming[node_id]) for node_id in node_ids}
    queue = sorted(node_id for node_id, count in indegree.items() if count == 0)
    visited: list[str] = []
    while queue:
        current = queue.pop(0)
        visited.append(current)
        for edge in outgoing.get(current, []):
            if edge.target not in indegree:
                continue
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                queue.append(edge.target)
                queue.sort()
    if len(visited) != len(node_ids):
        cyclic = sorted(node_ids - set(visited))
        issues.append("general graph cycles are not allowed: " + ", ".join(cyclic))

    if input_nodes:
        reachable: set[str] = set()
        pending = [input_nodes[0].id]
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(edge.target for edge in outgoing.get(current, []))
        missing = sorted(node_ids - reachable)
        if missing:
            issues.append("nodes are unreachable from input: " + ", ".join(missing))

    limits = _json_object(payload.get("limits"), field_name="limits")
    unknown_limits = set(limits) - {
        "max_steps",
        "max_context_tokens",
        "max_concurrency",
        "max_memory_bytes",
    }
    if unknown_limits:
        issues.append(f"limits has unknown keys: {', '.join(sorted(unknown_limits))}")
    max_steps = _validated_int(
        limits.get("max_steps"),
        field_name="limits.max_steps",
        default=max(1, len(nodes) * 4),
        issues=issues,
    )
    max_context_tokens = _validated_int(
        limits.get("max_context_tokens"),
        field_name="limits.max_context_tokens",
        default=65_536,
        issues=issues,
    )
    max_concurrency = _validated_int(
        limits.get("max_concurrency"),
        field_name="limits.max_concurrency",
        default=1,
        issues=issues,
    )
    max_memory_bytes = _validated_int(
        limits.get("max_memory_bytes"),
        field_name="limits.max_memory_bytes",
        default=0,
        issues=issues,
    )
    if not 1 <= max_steps <= 10_000:
        issues.append("limits.max_steps must be from 1 to 10000")
    if not 1_024 <= max_context_tokens <= 1_048_576:
        issues.append("limits.max_context_tokens must be from 1024 to 1048576")
    if schema_version == 1 and max_concurrency != 1:
        issues.append("version 1 Graphs require limits.max_concurrency = 1")
    if schema_version == 2 and not 1 <= max_concurrency <= 16:
        issues.append("version 2 Graphs require limits.max_concurrency from 1 to 16")
    if max_memory_bytes < 0:
        issues.append("limits.max_memory_bytes may not be negative")
    limits.update(
        {
            "max_steps": max_steps,
            "max_context_tokens": max_context_tokens,
            "max_concurrency": max_concurrency,
            "max_memory_bytes": max_memory_bytes,
        }
    )
    schedule = _json_object(payload.get("schedule"), field_name="schedule")
    unknown_schedule = set(schedule) - {"policy", "max_parallel_model_requests"}
    if unknown_schedule:
        issues.append(
            "schedule has unknown keys: " + ", ".join(sorted(unknown_schedule))
        )
    schedule_policy = str(schedule.get("policy") or "fifo").strip().lower()
    if schedule_policy not in {"fifo", "critical_path"}:
        issues.append("schedule.policy must be fifo or critical_path")
    max_parallel_models = _validated_int(
        schedule.get("max_parallel_model_requests"),
        field_name="schedule.max_parallel_model_requests",
        default=1,
        issues=issues,
    )
    if max_parallel_models != 1:
        issues.append(
            "schedule.max_parallel_model_requests must be 1 because MTPLX owns model admission"
        )
    schedule = {
        "policy": schedule_policy,
        "max_parallel_model_requests": 1,
    }

    layout = _json_object(payload.get("layout"), field_name="layout")
    unknown_layout = set(layout) - {"nodes", "viewport"}
    if unknown_layout:
        issues.append("layout has unknown keys: " + ", ".join(sorted(unknown_layout)))
    layout_nodes = layout.get("nodes", {})
    if not isinstance(layout_nodes, Mapping):
        issues.append("layout.nodes must be an object")
        layout_nodes = {}
    normalized_layout_nodes: dict[str, dict[str, float]] = {}
    for node_id, position in layout_nodes.items():
        normalized_id = safe_id(str(node_id), fallback="")
        if normalized_id not in node_ids:
            issues.append(f"layout.nodes references unknown node: {node_id}")
            continue
        if not isinstance(position, Mapping):
            issues.append(f"layout.nodes.{normalized_id} must be an object")
            continue
        if set(position) - {"x", "y"}:
            issues.append(f"layout.nodes.{normalized_id} has unknown keys")
            continue
        x, y = position.get("x"), position.get("y")
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not -100_000 <= float(x) <= 100_000
            or not -100_000 <= float(y) <= 100_000
        ):
            issues.append(f"layout.nodes.{normalized_id} requires finite x and y coordinates")
            continue
        normalized_layout_nodes[normalized_id] = {"x": float(x), "y": float(y)}
    layout = {"nodes": normalized_layout_nodes}
    viewport = layout.get("viewport")
    if isinstance(payload.get("layout"), Mapping):
        raw_viewport = payload["layout"].get("viewport")
        if raw_viewport is not None:
            if not isinstance(raw_viewport, Mapping) or set(raw_viewport) - {"x", "y", "scale"}:
                issues.append("layout.viewport must contain only x, y, and scale")
            else:
                values = {key: raw_viewport.get(key) for key in ("x", "y", "scale")}
                if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values.values()):
                    issues.append("layout.viewport x, y, and scale must be numbers")
                elif not 0.1 <= float(values["scale"]) <= 4:
                    issues.append("layout.viewport.scale must be from 0.1 to 4")
                else:
                    layout["viewport"] = {key: float(item) for key, item in values.items()}
    policies_raw = payload.get("policies")
    if policies_raw is not None and not isinstance(policies_raw, Mapping):
        issues.append("policies must be an object")
    policies = {
        str(key): str(value).strip().lower()
        for key, value in (
            policies_raw.items() if isinstance(policies_raw, Mapping) else []
        )
    }
    for key, mode in policies.items():
        if key not in GRAPH_POLICY_KEYS:
            issues.append(f"unknown Graph policy key: {key}")
        if mode not in POLICY_MODES:
            issues.append(f"unknown Graph policy mode for {key}: {mode}")
    timeout_seconds = _validated_int(
        payload.get("timeout_seconds"),
        field_name="timeout_seconds",
        default=3600,
        issues=issues,
    )
    if not 1 <= timeout_seconds <= 604_800:
        issues.append("timeout_seconds must be from 1 to 604800")

    runtime_requirements = _json_object(
        payload.get("runtime_requirements"),
        field_name="runtime_requirements",
    )
    unknown_runtime = set(runtime_requirements) - GRAPH_RUNTIME_REQUIREMENT_KEYS
    if unknown_runtime:
        issues.append(
            "runtime_requirements has unknown keys: "
            + ", ".join(sorted(unknown_runtime))
        )
    provider = str(runtime_requirements.get("provider") or "mtplx").strip().lower()
    if provider != "mtplx":
        issues.append("runtime_requirements.provider must be mtplx")
    runtime_requirements["provider"] = provider
    for key in ("backend", "model", "profile"):
        if key in runtime_requirements and not isinstance(runtime_requirements[key], str):
            issues.append(f"runtime_requirements.{key} must be a string")
        elif key in runtime_requirements:
            runtime_requirements[key] = str(runtime_requirements[key]).strip()
    for key in (
        "allowed_backends",
        "allowed_models",
        "allowed_profiles",
        "required_capabilities",
    ):
        if key in runtime_requirements:
            runtime_requirements[key] = _string_list(
                runtime_requirements[key],
                field_name=f"runtime_requirements.{key}",
                issues=issues,
            )
            if len(runtime_requirements[key]) != len(set(runtime_requirements[key])):
                issues.append(f"runtime_requirements.{key} may not contain duplicates")
    for key, default in (
        ("require_loaded_model", False),
        ("allow_model_fallback", True),
    ):
        value = runtime_requirements.get(key, default)
        if not isinstance(value, bool):
            issues.append(f"runtime_requirements.{key} must be a boolean")
            value = default
        runtime_requirements[key] = bool(value)
    minimum_context = _validated_int(
        runtime_requirements.get("min_context_tokens"),
        field_name="runtime_requirements.min_context_tokens",
        default=0,
        issues=issues,
    )
    if minimum_context < 0 or minimum_context > 1_048_576:
        issues.append(
            "runtime_requirements.min_context_tokens must be from 0 to 1048576"
        )
    runtime_requirements["min_context_tokens"] = minimum_context
    required_model = str(runtime_requirements.get("model") or "")
    allowed_models = runtime_requirements.get("allowed_models") or []
    if required_model and allowed_models and required_model not in allowed_models:
        issues.append("runtime_requirements.model must appear in allowed_models")
    required_profile = str(runtime_requirements.get("profile") or "")
    allowed_profiles = runtime_requirements.get("allowed_profiles") or []
    if (
        required_profile
        and required_profile != "auto"
        and allowed_profiles
        and required_profile not in allowed_profiles
    ):
        issues.append("runtime_requirements.profile must appear in allowed_profiles")
    required_backend = str(runtime_requirements.get("backend") or "")
    allowed_backends = runtime_requirements.get("allowed_backends") or []
    if (
        required_backend
        and required_backend not in {"auto", "mtplx"}
        and allowed_backends
        and required_backend not in allowed_backends
    ):
        issues.append("runtime_requirements.backend must appear in allowed_backends")
    for node in nodes:
        model_configs: list[tuple[str, Mapping[str, Any]]] = []
        if node.type == "model":
            model_configs.append((f"model node {node.id}", node.config))
        elif node.type == "loop":
            body = node.config.get("body")
            if isinstance(body, Mapping) and str(body.get("type") or "") == "model":
                body_config = body.get("config")
                if isinstance(body_config, Mapping):
                    model_configs.append(
                        (f"loop node {node.id} model body", body_config)
                    )
        for label, model_config in model_configs:
            node_model = str(model_config.get("model") or "").strip()
            if not node_model:
                continue
            if required_model and node_model != required_model:
                issues.append(
                    f"{label} model must equal runtime_requirements.model"
                )
            elif allowed_models and node_model not in allowed_models:
                issues.append(
                    f"{label} model is not in runtime_requirements.allowed_models"
                )
            elif not required_model and not allowed_models:
                issues.append(
                    f"{label} model override requires runtime_requirements.allowed_models"
                )

    retry_raw = _json_object(payload.get("retry"), field_name="retry")
    retry = _validate_retry_policy(
        retry_raw,
        field_name="retry",
        issues=issues,
        include_defaults=True,
    )

    approval_requirements = _json_object(
        payload.get("approval_requirements"),
        field_name="approval_requirements",
    )
    unknown_approval_requirements = (
        set(approval_requirements) - GRAPH_APPROVAL_REQUIREMENT_KEYS
    )
    if unknown_approval_requirements:
        issues.append(
            "approval_requirements has unknown keys: "
            + ", ".join(sorted(unknown_approval_requirements))
        )
    for key in (
        "required_node_ids",
        "required_tool_names",
        "required_policy_categories",
    ):
        approval_requirements[key] = _string_list(
            approval_requirements.get(key),
            field_name=f"approval_requirements.{key}",
            issues=issues,
        )
        if len(approval_requirements[key]) != len(set(approval_requirements[key])):
            issues.append(f"approval_requirements.{key} may not contain duplicates")
    for key in ("all_side_effects", "memory_writes"):
        value = approval_requirements.get(key, False)
        if not isinstance(value, bool):
            issues.append(f"approval_requirements.{key} must be a boolean")
            value = False
        approval_requirements[key] = bool(value)
    unknown_required_nodes = sorted(
        set(approval_requirements["required_node_ids"]) - node_ids
    )
    if unknown_required_nodes:
        issues.append(
            "approval_requirements.required_node_ids references unknown nodes: "
            + ", ".join(unknown_required_nodes)
        )
    for node_id in approval_requirements["required_node_ids"]:
        node = next((item for item in nodes if item.id == node_id), None)
        if node is not None:
            is_side_effect = node.type in {"memory_write", "memory_curate", "tool"}
            if node.type == "loop":
                body = node.config.get("body")
                if isinstance(body, Mapping):
                    body_type = str(body.get("type") or "")
                    body_config = body.get("config")
                    is_side_effect = body_type in {"memory_write", "memory_curate"} or (
                        body_type == "tool"
                        and isinstance(body_config, Mapping)
                        and str(body_config.get("tool") or "") in MUTATING_TOOLS
                    )
            if not is_side_effect:
                issues.append(
                    f"approval_requirements node {node_id} is not a side-effect node"
                )
    unknown_required_tools = sorted(
        set(approval_requirements["required_tool_names"]) - set(FIRST_PARTY_TOOL_NAMES)
    )
    if unknown_required_tools:
        issues.append(
            "approval_requirements.required_tool_names has unknown tools: "
            + ", ".join(unknown_required_tools)
        )
    unknown_required_categories = sorted(
        set(approval_requirements["required_policy_categories"]) - GRAPH_POLICY_KEYS
    )
    if unknown_required_categories:
        issues.append(
            "approval_requirements.required_policy_categories has unknown categories: "
            + ", ".join(unknown_required_categories)
        )

    revision_value = _validated_int(
        revision if revision is not None else payload.get("revision"),
        field_name="revision",
        default=1,
        issues=issues,
    )
    if revision_value < 1:
        issues.append("revision must be at least 1")

    if issues:
        raise GraphValidationError(issues)
    now = utc_now()
    definition = GraphDefinition(
        id=identifier,
        project_id=project_id,
        workspace_id=workspace_id,
        name=name,
        description=str(payload.get("description") or ""),
        schema_version=schema_version,
        revision=max(1, revision_value),
        inputs=inputs,
        outputs=outputs,
        nodes=tuple(nodes),
        edges=tuple(edges),
        limits=limits,
        policies=policies,
        runtime_requirements=runtime_requirements,
        retry=retry,
        timeout_seconds=timeout_seconds,
        approval_requirements=approval_requirements,
        schedule=schedule,
        layout=layout,
        created_at=created_at or str(payload.get("created_at") or now),
        updated_at=updated_at or str(payload.get("updated_at") or now),
        content_sha256="",
    )
    digest = _canonical_sha256(definition.to_dict(include_hash=False))
    return GraphDefinition(**{**definition.__dict__, "content_sha256": digest})


class GraphStore:
    """File-backed immutable Graph revisions plus mutable checkpointed runs."""

    def __init__(self, workspace_store: WorkspaceStore) -> None:
        self.workspace_store = workspace_store
        self.root = workspace_store.root / "graphs"
        self.definitions_root = self.root / "definitions"
        self.runs_root = self.root / "runs"
        self._lock = threading.RLock()
        self.ensure_layout()

    def ensure_layout(self) -> None:
        self.definitions_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def _definition_dir(self, graph_id: str) -> Path:
        return self.definitions_root / safe_id(graph_id, fallback="graph")

    def _revision_path(self, graph_id: str, revision: int) -> Path:
        return self._definition_dir(graph_id) / f"r{int(revision):08d}.json"

    def _latest_path(self, graph_id: str) -> Path:
        return self._definition_dir(graph_id) / "latest.json"

    def _run_path(self, run_id: str) -> Path:
        return self.runs_root / f"{safe_id(run_id, fallback='graph-run')}.json"

    def create(self, payload: Mapping[str, Any]) -> GraphDefinition:
        definition = validate_graph_payload(payload, revision=1)
        self.workspace_store.get_workspace(definition.workspace_id)
        with self.workspace_store._exclusive():
            directory = self._definition_dir(definition.id)
            if directory.exists():
                raise WorkspaceConflictError(f"graph already exists: {definition.id}")
            directory.mkdir(parents=True)
            self._write_definition(definition)
        return definition

    def update(
        self,
        graph_id: str,
        payload: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> GraphDefinition:
        current = self.get(graph_id)
        if expected_revision is not None and current.revision != int(expected_revision):
            raise WorkspaceConflictError(
                f"graph revision conflict: expected {expected_revision}, found {current.revision}"
            )
        merged = {
            **current.to_dict(include_hash=False),
            **dict(payload),
            "id": current.id,
            "workspace_id": current.workspace_id,
            "project_id": current.project_id,
        }
        definition = validate_graph_payload(
            merged,
            graph_id=current.id,
            revision=current.revision + 1,
            created_at=current.created_at,
            updated_at=utc_now(),
        )
        self.workspace_store.get_workspace(definition.workspace_id)
        with self.workspace_store._exclusive():
            if self._revision_path(definition.id, definition.revision).exists():
                raise WorkspaceConflictError(
                    f"graph revision already exists: {definition.id}@{definition.revision}"
                )
            self._write_definition(definition)
        return definition

    def _write_definition(self, definition: GraphDefinition) -> None:
        directory = self._definition_dir(definition.id)
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self._revision_path(definition.id, definition.revision),
            json.dumps(definition.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        _atomic_write(
            self._latest_path(definition.id),
            json.dumps(
                {
                    "id": definition.id,
                    "revision": definition.revision,
                    "content_sha256": definition.content_sha256,
                    "updated_at": definition.updated_at,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def get(self, graph_id: str, revision: int | None = None) -> GraphDefinition:
        if revision is None:
            try:
                latest = json.loads(self._latest_path(graph_id).read_text(encoding="utf-8"))
                revision = int(latest["revision"])
            except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise GraphError(f"graph not found: {graph_id}") from exc
        try:
            value = json.loads(
                self._revision_path(graph_id, int(revision)).read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GraphError(f"graph revision not found: {graph_id}@{revision}") from exc
        if not isinstance(value, Mapping):
            raise GraphError(f"invalid graph revision: {graph_id}@{revision}")
        definition = validate_graph_payload(
            value,
            graph_id=str(value.get("id") or graph_id),
            revision=int(value.get("revision") or revision),
            created_at=str(value.get("created_at") or utc_now()),
            updated_at=str(value.get("updated_at") or utc_now()),
        )
        stored_hash = str(value.get("content_sha256") or "")
        if stored_hash and stored_hash != definition.content_sha256:
            raise GraphError(f"graph revision hash mismatch: {graph_id}@{revision}")
        return definition

    def list(self, *, workspace_id: str | None = None, limit: int = 100) -> list[GraphDefinition]:
        result: list[GraphDefinition] = []
        self.ensure_layout()
        for path in sorted(self.definitions_root.glob("*/latest.json")):
            try:
                definition = self.get(path.parent.name)
            except GraphError:
                continue
            if workspace_id and definition.workspace_id != workspace_id:
                continue
            result.append(definition)
        result.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
        return result[: max(1, min(int(limit), 1000))]

    def list_revisions(self, graph_id: str, *, limit: int = 100) -> list[GraphDefinition]:
        result: list[GraphDefinition] = []
        directory = self._definition_dir(graph_id)
        if not directory.is_dir():
            raise GraphError(f"graph not found: {graph_id}")
        for path in sorted(directory.glob("r*.json"), reverse=True):
            try:
                revision = int(path.stem[1:])
                result.append(self.get(graph_id, revision=revision))
            except (GraphError, TypeError, ValueError):
                continue
            if len(result) >= max(1, min(int(limit), 1000)):
                break
        return result

    def create_run(
        self,
        graph: GraphDefinition,
        *,
        inputs: Mapping[str, Any] | None = None,
        model: str | None = None,
        runtime_profile: str = "auto",
        run_id: str | None = None,
    ) -> GraphRun:
        workspace = self.workspace_store.get_workspace(graph.workspace_id)
        normalized_inputs = apply_graph_contract_defaults(
            graph.inputs,
            inputs,
            field_name="inputs",
        )
        validate_graph_contract_value(
            graph.inputs,
            normalized_inputs,
            field_name="inputs",
        )
        requirements = graph.runtime_requirements
        required_model = str(requirements.get("model") or "").strip()
        selected_model = str(model or required_model or workspace.model or "").strip() or None
        allowed_models = {
            str(item) for item in requirements.get("allowed_models") or []
        }
        if required_model and selected_model != required_model:
            raise GraphValidationError(
                [
                    "run model does not satisfy runtime_requirements.model: "
                    f"expected {required_model}, found {selected_model or '<none>'}"
                ]
            )
        if allowed_models and selected_model not in allowed_models:
            raise GraphValidationError(
                [
                    "run model is not allowed by runtime_requirements.allowed_models: "
                    f"{selected_model or '<none>'}"
                ]
            )
        required_profile = str(requirements.get("profile") or "").strip()
        selected_profile = str(runtime_profile or "auto").strip() or "auto"
        if required_profile and required_profile != "auto":
            if selected_profile == "auto":
                selected_profile = required_profile
            elif selected_profile != required_profile:
                raise GraphValidationError(
                    [
                        "run runtime_profile does not satisfy "
                        f"runtime_requirements.profile: expected {required_profile}, "
                        f"found {selected_profile}"
                    ]
                )
        allowed_profiles = {
            str(item) for item in requirements.get("allowed_profiles") or []
        }
        if allowed_profiles and selected_profile not in allowed_profiles:
            raise GraphValidationError(
                [
                    "run runtime_profile is not allowed by "
                    f"runtime_requirements.allowed_profiles: {selected_profile}"
                ]
            )
        if len(selected_profile) > 200 or any(ord(char) < 32 for char in selected_profile):
            raise GraphValidationError(["runtime_profile is invalid"])
        identifier = safe_id(
            run_id or f"graph_run_{uuid.uuid4().hex}",
            fallback="graph-run",
        )
        if self._run_path(identifier).exists():
            raise WorkspaceConflictError(f"graph run already exists: {identifier}")
        durable_run = self.workspace_store.create_run(
            workspace.id,
            run_id=identifier,
            session_id=f"graph:{identifier}",
            title=f"Graph: {graph.name}",
            model=selected_model,
        )
        now = utc_now()
        node_states = {
            node.id: {
                "status": "pending",
                "attempts": 0,
                "iterations_completed": 0,
                "loop_outputs": [],
                "iteration_metrics": [],
                "active_iteration": None,
                "active_body_state": {},
                "side_effect_retry_generation": 0,
                "recovery_guard": None,
                "output": None,
                "error": None,
                "started_at": None,
                "completed_at": None,
                "pending_approval_id": None,
                "idempotency_key": None,
                "metrics": {},
            }
            for node in graph.nodes
        }
        run = GraphRun(
            id=durable_run.id,
            graph_id=graph.id,
            graph_revision=graph.revision,
            graph_sha256=graph.content_sha256,
            workspace_id=graph.workspace_id,
            project_id=graph.project_id,
            workspace_root=workspace.root_path,
            status="queued",
            pinned_model=selected_model,
            runtime_profile=selected_profile,
            inputs=normalized_inputs,
            outputs={},
            node_states=node_states,
            current_node_id=None,
            pending_approval_id=None,
            resource_metrics={},
            created_at=now,
            updated_at=now,
            state_version=1,
        )
        with self.workspace_store._exclusive():
            path = self._run_path(run.id)
            if path.exists():
                raise WorkspaceConflictError(f"graph run already exists: {run.id}")
            _atomic_write(path, json.dumps(run.to_dict(), indent=2, sort_keys=True) + "\n")
        self.workspace_store.append_event(
            run.id,
            "graph_run_created",
            {
                "graph_id": graph.id,
                "graph_revision": graph.revision,
                "graph_sha256": graph.content_sha256,
                "pinned_model": run.pinned_model,
                "runtime_profile": run.runtime_profile,
            },
        )
        return self.get_run(run.id)

    def _decode_run(self, value: Mapping[str, Any]) -> GraphRun:
        run = GraphRun(
            id=str(value["id"]),
            graph_id=str(value["graph_id"]),
            graph_revision=int(value["graph_revision"]),
            graph_sha256=str(value["graph_sha256"]),
            workspace_id=str(value["workspace_id"]),
            project_id=str(value["project_id"]),
            workspace_root=str(
                value.get("workspace_root")
                or self.workspace_store.get_workspace(str(value["workspace_id"])).root_path
            ),
            status=str(value.get("status") or "queued"),
            pinned_model=str(value["pinned_model"]) if value.get("pinned_model") else None,
            runtime_profile=str(value.get("runtime_profile") or "auto"),
            inputs=dict(value.get("inputs") or {}),
            outputs=dict(value.get("outputs") or {}),
            node_states={
                str(key): dict(item)
                for key, item in dict(value.get("node_states") or {}).items()
                if isinstance(item, Mapping)
            },
            current_node_id=(
                str(value["current_node_id"]) if value.get("current_node_id") else None
            ),
            pending_approval_id=(
                str(value["pending_approval_id"])
                if value.get("pending_approval_id")
                else None
            ),
            resource_metrics=dict(value.get("resource_metrics") or {}),
            created_at=str(value.get("created_at") or utc_now()),
            updated_at=str(value.get("updated_at") or utc_now()),
            state_version=max(1, int(value.get("state_version") or 1)),
            pause_requested=bool(value.get("pause_requested")),
            error=str(value["error"]) if value.get("error") else None,
        )
        if run.status not in GRAPH_RUN_STATUSES:
            raise GraphError(f"unknown graph run status: {run.status}")
        invalid_nodes = sorted(
            node_id
            for node_id, state in run.node_states.items()
            if str(state.get("status") or "pending") not in GRAPH_NODE_STATUSES
        )
        if invalid_nodes:
            raise GraphError(
                "unknown graph node status for: " + ", ".join(invalid_nodes)
            )
        return run

    def get_run(self, run_id: str) -> GraphRun:
        try:
            value = json.loads(self._run_path(run_id).read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GraphError(f"graph run not found: {run_id}") from exc
        if not isinstance(value, Mapping):
            raise GraphError(f"invalid graph run: {run_id}")
        return self._decode_run(value)

    def update_run(
        self,
        run_id: str,
        *,
        expected_state_version: int | None = None,
        **changes: Any,
    ) -> GraphRun:
        allowed = {
            "status",
            "outputs",
            "node_states",
            "current_node_id",
            "pending_approval_id",
            "resource_metrics",
            "pause_requested",
            "error",
        }
        unknown = set(changes) - allowed
        if unknown:
            raise GraphError(f"unknown graph run fields: {', '.join(sorted(unknown))}")
        with self.workspace_store._exclusive():
            current = self.get_run(run_id)
            if (
                expected_state_version is not None
                and current.state_version != int(expected_state_version)
            ):
                raise WorkspaceConflictError(
                    f"graph run state conflict: expected {expected_state_version}, "
                    f"found {current.state_version}"
                )
            if current.status in {"completed", "failed", "cancelled"}:
                current_value = current.to_dict()
                changed = [
                    key
                    for key, item in changes.items()
                    if current_value.get(key) != item
                ]
                if changed:
                    raise WorkspaceConflictError(
                        f"terminal graph run is immutable: {current.id}"
                    )
                return current
            value = current.to_dict()
            value.update(changes)
            if "status" in changes:
                metrics = dict(value.get("resource_metrics") or {})
                metrics.pop("mirror_sync_error", None)
                value["resource_metrics"] = metrics
            value["updated_at"] = utc_now()
            value["state_version"] = current.state_version + 1
            updated = self._decode_run(value)
            if updated.status not in GRAPH_RUN_STATUSES:
                raise GraphError(f"unknown graph run status: {updated.status}")
            if updated.status not in _GRAPH_STATUS_TRANSITIONS[current.status]:
                raise WorkspaceConflictError(
                    f"invalid graph run transition: {current.status} -> {updated.status}"
                )
            _atomic_write(
                self._run_path(updated.id),
                json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
            )
        mirror_sync_error: str | None = None
        if "status" in changes:
            mapped = {
                "waiting_approval": "paused",
                "paused": "paused",
                "running": "running",
                "queued": "queued",
                "completed": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
            }[updated.status]
            try:
                self.workspace_store.update_run(
                    updated.id,
                    status=mapped,
                    error=updated.error or "",
                )
            except WorkspaceStoreError as exc:
                mirror_sync_error = f"{type(exc).__name__}: {exc}"
        if mirror_sync_error is not None:
            value = updated.to_dict()
            metrics = dict(updated.resource_metrics)
            metrics["mirror_sync_error"] = mirror_sync_error
            value["resource_metrics"] = metrics
            updated = self._decode_run(value)
            with self.workspace_store._exclusive():
                _atomic_write(
                    self._run_path(updated.id),
                    json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n",
                )
        return updated

    def list_runs(
        self,
        *,
        graph_id: str | None = None,
        workspace_id: str | None = None,
        limit: int = 100,
    ) -> list[GraphRun]:
        result: list[GraphRun] = []
        self.ensure_layout()
        for path in self.runs_root.glob("*.json"):
            try:
                run = self.get_run(path.stem)
            except GraphError:
                continue
            if graph_id and run.graph_id != graph_id:
                continue
            if workspace_id and run.workspace_id != workspace_id:
                continue
            result.append(run)
        result.sort(key=lambda item: (item.updated_at, item.id), reverse=True)
        return result[: max(1, min(int(limit), 1000))]


__all__ = [
    "GRAPH_APPROVAL_REQUIREMENT_KEYS",
    "GRAPH_NODE_STATUSES",
    "GRAPH_NODE_TYPES",
    "GRAPH_POLICY_KEYS",
    "GRAPH_RUN_STATUSES",
    "GRAPH_RUNTIME_REQUIREMENT_KEYS",
    "GRAPH_SCHEMA_VERSION",
    "GraphDefinition",
    "GraphEdge",
    "GraphError",
    "GraphNode",
    "GraphRun",
    "GraphStore",
    "GraphValidationError",
    "LOOP_BODY_TYPES",
    "SIDE_EFFECT_NODE_TYPES",
    "apply_graph_contract_defaults",
    "validate_graph_contract_value",
    "validate_graph_payload",
]
