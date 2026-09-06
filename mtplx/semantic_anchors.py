"""Exact semantic checkpoint planning for long agent transcripts.

The planner never guesses a cache boundary from character offsets.  Every
candidate carries the complete tokenized prefix produced by the active chat
template and is admitted only when it is an exact prefix of the final rendered
prompt.  Callers can therefore use the returned offsets as mandatory recurrent
checkpoint edges without weakening SessionBank's existing fail-closed restore
contract.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence


class SemanticAnchorKind(str, Enum):
    INSTRUCTIONS_END = "instructions_end"
    USER_TURN_END = "user_turn_end"
    REASONING_START = "reasoning_start"
    REASONING_END = "reasoning_end"
    TOOL_CALL_END = "tool_call_end"
    TOOL_RESULT_END = "tool_result_end"
    ASSISTANT_TURN_END = "assistant_turn_end"
    COMPACTION_END = "compaction_end"
    EXPLICIT = "explicit"


_KIND_PRIORITY: dict[SemanticAnchorKind, int] = {
    SemanticAnchorKind.TOOL_RESULT_END: 100,
    SemanticAnchorKind.TOOL_CALL_END: 95,
    SemanticAnchorKind.ASSISTANT_TURN_END: 90,
    SemanticAnchorKind.COMPACTION_END: 85,
    SemanticAnchorKind.USER_TURN_END: 80,
    SemanticAnchorKind.REASONING_END: 70,
    SemanticAnchorKind.REASONING_START: 65,
    SemanticAnchorKind.INSTRUCTIONS_END: 55,
    SemanticAnchorKind.EXPLICIT: 50,
}

_KIND_ALIASES: dict[str, SemanticAnchorKind] = {
    "system_end": SemanticAnchorKind.INSTRUCTIONS_END,
    "developer_end": SemanticAnchorKind.INSTRUCTIONS_END,
    "instruction_end": SemanticAnchorKind.INSTRUCTIONS_END,
    "instructions_end": SemanticAnchorKind.INSTRUCTIONS_END,
    "user_end": SemanticAnchorKind.USER_TURN_END,
    "user_turn_end": SemanticAnchorKind.USER_TURN_END,
    "thinking_start": SemanticAnchorKind.REASONING_START,
    "reasoning_start": SemanticAnchorKind.REASONING_START,
    "thinking_end": SemanticAnchorKind.REASONING_END,
    "reasoning_end": SemanticAnchorKind.REASONING_END,
    "tool_call": SemanticAnchorKind.TOOL_CALL_END,
    "tool_call_end": SemanticAnchorKind.TOOL_CALL_END,
    "tool_result": SemanticAnchorKind.TOOL_RESULT_END,
    "tool_result_end": SemanticAnchorKind.TOOL_RESULT_END,
    "assistant_end": SemanticAnchorKind.ASSISTANT_TURN_END,
    "assistant_turn_end": SemanticAnchorKind.ASSISTANT_TURN_END,
    "summary_end": SemanticAnchorKind.COMPACTION_END,
    "compaction_end": SemanticAnchorKind.COMPACTION_END,
    "explicit": SemanticAnchorKind.EXPLICIT,
}


def _message_value(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _message_metadata(message: Any) -> dict[str, Any]:
    raw = _message_value(message, "metadata", None)
    metadata = dict(raw) if isinstance(raw, Mapping) else {}
    for key in (
        "semantic_kind",
        "reasoning_boundary",
        "is_reasoning",
        "is_compaction_summary",
        "compacted",
        "summary",
        "tool_calls",
        "tool_call_id",
    ):
        value = _message_value(message, key, None)
        if value is not None and key not in metadata:
            metadata[key] = value
    return metadata


def classify_semantic_kind(
    role: str,
    metadata: Mapping[str, Any] | None = None,
) -> SemanticAnchorKind:
    """Classify a rendered message boundary without inspecting its text."""

    metadata = metadata or {}
    explicit = metadata.get("semantic_kind")
    if explicit is not None:
        normalized = str(explicit).strip().lower().replace("-", "_")
        if normalized in _KIND_ALIASES:
            return _KIND_ALIASES[normalized]
        try:
            return SemanticAnchorKind(normalized)
        except ValueError:
            return SemanticAnchorKind.EXPLICIT

    reasoning_boundary = str(metadata.get("reasoning_boundary") or "").strip().lower()
    if reasoning_boundary in {"start", "begin", "open"}:
        return SemanticAnchorKind.REASONING_START
    if reasoning_boundary in {"end", "close", "stop"}:
        return SemanticAnchorKind.REASONING_END

    normalized_role = str(role or "").strip().lower()
    if bool(
        metadata.get("is_compaction_summary")
        or metadata.get("compacted")
        or metadata.get("summary")
    ):
        return SemanticAnchorKind.COMPACTION_END
    if normalized_role in {"system", "developer"}:
        return SemanticAnchorKind.INSTRUCTIONS_END
    if normalized_role == "user":
        return SemanticAnchorKind.USER_TURN_END
    if normalized_role in {"tool", "function"} or metadata.get("tool_call_id"):
        return SemanticAnchorKind.TOOL_RESULT_END
    if normalized_role == "assistant":
        if metadata.get("tool_calls"):
            return SemanticAnchorKind.TOOL_CALL_END
        if metadata.get("is_reasoning"):
            return SemanticAnchorKind.REASONING_END
        return SemanticAnchorKind.ASSISTANT_TURN_END
    return SemanticAnchorKind.EXPLICIT


def _token_tuple(value: Sequence[int] | Iterable[int]) -> tuple[int, ...]:
    return tuple(int(token) for token in value)


def _token_prefix_hash(tokens: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token in tokens:
        digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


def _first_mismatch(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (lhs, rhs) in enumerate(zip(left, right)):
        if int(lhs) != int(rhs):
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


@dataclass(frozen=True)
class RenderedMessagePrefix:
    """One exact chat-template prefix ending at a semantic boundary."""

    message_index: int
    role: str
    token_ids: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    estimated_checkpoint_bytes: int = 0
    survival_probability: float = 1.0
    source: str = "message_prefix"

    @classmethod
    def from_tokens(
        cls,
        *,
        message_index: int,
        role: str,
        token_ids: Sequence[int] | Iterable[int],
        metadata: Mapping[str, Any] | None = None,
        estimated_checkpoint_bytes: int = 0,
        survival_probability: float = 1.0,
        source: str = "message_prefix",
    ) -> "RenderedMessagePrefix":
        return cls(
            message_index=int(message_index),
            role=str(role),
            token_ids=_token_tuple(token_ids),
            metadata=dict(metadata or {}),
            estimated_checkpoint_bytes=max(0, int(estimated_checkpoint_bytes)),
            survival_probability=max(0.0, min(1.0, float(survival_probability))),
            source=str(source),
        )


@dataclass(frozen=True)
class SemanticAnchor:
    kind: SemanticAnchorKind
    token_offset: int
    message_index: int
    token_prefix_hash: str
    template_hash: str | None
    estimated_checkpoint_bytes: int
    survival_probability: float
    priority: int
    utility: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "token_offset": int(self.token_offset),
            "message_index": int(self.message_index),
            "token_prefix_hash": self.token_prefix_hash,
            "template_hash": self.template_hash,
            "estimated_checkpoint_bytes": int(self.estimated_checkpoint_bytes),
            "survival_probability": float(self.survival_probability),
            "priority": int(self.priority),
            "utility": float(self.utility),
            "source": self.source,
        }


@dataclass(frozen=True)
class RejectedSemanticBoundary:
    message_index: int
    role: str
    reason: str
    token_offset: int
    mismatch_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_index": int(self.message_index),
            "role": self.role,
            "reason": self.reason,
            "token_offset": int(self.token_offset),
            "mismatch_index": self.mismatch_index,
        }


@dataclass(frozen=True)
class SemanticAnchorPlan:
    anchors: tuple[SemanticAnchor, ...]
    rejected: tuple[RejectedSemanticBoundary, ...]
    final_token_count: int
    template_hash: str | None
    max_anchors: int
    max_checkpoint_bytes: int | None

    @property
    def edges(self) -> tuple[int, ...]:
        return tuple(anchor.token_offset for anchor in self.anchors)

    @property
    def estimated_checkpoint_bytes(self) -> int:
        return sum(anchor.estimated_checkpoint_bytes for anchor in self.anchors)

    def to_metrics(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for anchor in self.anchors:
            by_kind[anchor.kind.value] = by_kind.get(anchor.kind.value, 0) + 1
        rejected: dict[str, int] = {}
        for boundary in self.rejected:
            rejected[boundary.reason] = rejected.get(boundary.reason, 0) + 1
        return {
            "semantic_anchor_count": len(self.anchors),
            "semantic_anchor_edges": list(self.edges),
            "semantic_anchor_kinds": by_kind,
            "semantic_anchor_rejected": rejected,
            "semantic_anchor_estimated_bytes": self.estimated_checkpoint_bytes,
            "semantic_anchor_final_tokens": int(self.final_token_count),
        }


def plan_semantic_anchors(
    final_token_ids: Sequence[int] | Iterable[int],
    rendered_prefixes: Iterable[RenderedMessagePrefix],
    *,
    template_hash: str | None = None,
    max_anchors: int = 8,
    max_checkpoint_bytes: int | None = None,
    default_checkpoint_bytes: int = 1,
    min_token_offset: int = 1,
) -> SemanticAnchorPlan:
    """Select exact, deterministic semantic checkpoint edges.

    A candidate is rejected unless its complete rendered token prefix exactly
    matches the same-length prefix of ``final_token_ids``.  Selection is then
    utility-ranked under count/byte budgets and returned in token order.
    """

    final_tokens = _token_tuple(final_token_ids)
    max_count = max(0, int(max_anchors))
    byte_budget = (
        None if max_checkpoint_bytes is None else max(0, int(max_checkpoint_bytes))
    )
    default_bytes = max(1, int(default_checkpoint_bytes))
    lower = max(1, int(min_token_offset))

    rejected: list[RejectedSemanticBoundary] = []
    by_offset: dict[int, SemanticAnchor] = {}

    for prefix in rendered_prefixes:
        tokens = _token_tuple(prefix.token_ids)
        offset = len(tokens)
        if offset < lower:
            rejected.append(
                RejectedSemanticBoundary(
                    prefix.message_index,
                    prefix.role,
                    "below_minimum",
                    offset,
                )
            )
            continue
        if offset > len(final_tokens):
            rejected.append(
                RejectedSemanticBoundary(
                    prefix.message_index,
                    prefix.role,
                    "longer_than_final_prompt",
                    offset,
                    len(final_tokens),
                )
            )
            continue
        expected = final_tokens[:offset]
        mismatch = _first_mismatch(tokens, expected)
        if mismatch is not None:
            rejected.append(
                RejectedSemanticBoundary(
                    prefix.message_index,
                    prefix.role,
                    "not_exact_prefix",
                    offset,
                    mismatch,
                )
            )
            continue

        kind = classify_semantic_kind(prefix.role, prefix.metadata)
        priority = _KIND_PRIORITY[kind]
        estimated_bytes = max(
            1,
            int(prefix.estimated_checkpoint_bytes or default_bytes),
        )
        survival = max(0.0, min(1.0, float(prefix.survival_probability)))
        # Token offset approximates avoided re-prefill work.  log2 prevents a
        # single huge prefix from crowding every high-value tool/turn boundary.
        utility = (
            float(priority)
            * (1.0 + math.log2(float(offset) + 1.0))
            * max(0.01, survival)
            / max(1.0, estimated_bytes / float(1024**2))
        )
        candidate = SemanticAnchor(
            kind=kind,
            token_offset=offset,
            message_index=int(prefix.message_index),
            token_prefix_hash=_token_prefix_hash(tokens),
            template_hash=template_hash,
            estimated_checkpoint_bytes=estimated_bytes,
            survival_probability=survival,
            priority=priority,
            utility=utility,
            source=str(prefix.source),
        )
        incumbent = by_offset.get(offset)
        if incumbent is None or (
            candidate.priority,
            candidate.utility,
            candidate.message_index,
        ) > (
            incumbent.priority,
            incumbent.utility,
            incumbent.message_index,
        ):
            by_offset[offset] = candidate

    ranked = sorted(
        by_offset.values(),
        key=lambda anchor: (
            -anchor.utility,
            -anchor.priority,
            -anchor.token_offset,
            anchor.kind.value,
        ),
    )
    # The newest complete-message boundary is the live conversation frontier.
    # Reserve it before utility ranking so a tool-heavy history cannot spend
    # the entire count budget on older tool-call/result pairs.  This is the
    # boundary needed when the next tool-result request re-serializes the
    # assistant generation marker and invalidates the stored prompt tail.
    frontier = max(
        (
            anchor
            for anchor in ranked
            if int(anchor.token_offset) < len(final_tokens)
        ),
        key=lambda anchor: (
            anchor.token_offset,
            anchor.priority,
            anchor.kind.value,
        ),
        default=None,
    )
    selection_order = (
        [frontier, *(anchor for anchor in ranked if anchor is not frontier)]
        if frontier is not None
        else ranked
    )
    selected: list[SemanticAnchor] = []
    used_bytes = 0
    for candidate in selection_order:
        if len(selected) >= max_count:
            rejected.append(
                RejectedSemanticBoundary(
                    candidate.message_index,
                    candidate.kind.value,
                    "count_budget",
                    candidate.token_offset,
                )
            )
            continue
        projected = used_bytes + candidate.estimated_checkpoint_bytes
        if byte_budget is not None and projected > byte_budget:
            rejected.append(
                RejectedSemanticBoundary(
                    candidate.message_index,
                    candidate.kind.value,
                    "byte_budget",
                    candidate.token_offset,
                )
            )
            continue
        selected.append(candidate)
        used_bytes = projected

    selected.sort(key=lambda anchor: (anchor.token_offset, anchor.kind.value))
    return SemanticAnchorPlan(
        anchors=tuple(selected),
        rejected=tuple(rejected),
        final_token_count=len(final_tokens),
        template_hash=template_hash,
        max_anchors=max_count,
        max_checkpoint_bytes=byte_budget,
    )


def render_message_prefixes(
    messages: Sequence[Any],
    render_prefix: Callable[[Sequence[Any]], Sequence[int] | Iterable[int]],
    *,
    message_indexes: Iterable[int] | None = None,
    estimated_checkpoint_bytes: int = 0,
    survival_probability: float = 1.0,
) -> tuple[RenderedMessagePrefix, ...]:
    """Render every complete message prefix through the active template.

    Rendering failures are omitted; the final exact-prefix validation remains
    authoritative.  This function is intentionally request-local and carries
    no tokenizer or server dependency.
    """

    prefixes: list[RenderedMessagePrefix] = []
    if message_indexes is None:
        indexes = range(len(messages))
    else:
        indexes = sorted(
            {
                int(index)
                for index in message_indexes
                if 0 <= int(index) < len(messages)
            }
        )
    for index in indexes:
        message = messages[index]
        try:
            tokens = render_prefix(messages[: index + 1])
        except Exception:
            continue
        prefixes.append(
            RenderedMessagePrefix.from_tokens(
                message_index=index,
                role=str(_message_value(message, "role", "")),
                token_ids=tokens,
                metadata=_message_metadata(message),
                estimated_checkpoint_bytes=estimated_checkpoint_bytes,
                survival_probability=survival_probability,
            )
        )
    return tuple(prefixes)


def mandatory_semantic_edges(
    plan: SemanticAnchorPlan | None,
    *,
    lower_bound: int = 0,
    upper_bound: int | None = None,
) -> tuple[int, ...]:
    if plan is None:
        return ()
    lower = max(0, int(lower_bound))
    upper = plan.final_token_count if upper_bound is None else int(upper_bound)
    return tuple(
        sorted(
            {
                int(anchor.token_offset)
                for anchor in plan.anchors
                if lower < int(anchor.token_offset) < upper
            }
        )
    )


def merge_mandatory_edges(
    *edge_groups: Iterable[int] | None,
    lower_bound: int = 0,
    upper_bound: int | None = None,
) -> tuple[int, ...]:
    """Normalize stable-prefix and semantic edges for prefill chunk splitting."""

    lower = max(0, int(lower_bound))
    upper = None if upper_bound is None else int(upper_bound)
    merged: set[int] = set()
    for group in edge_groups:
        if group is None:
            continue
        for raw in group:
            edge = int(raw)
            if edge <= lower:
                continue
            if upper is not None and edge >= upper:
                continue
            merged.add(edge)
    return tuple(sorted(merged))


__all__ = [
    "RenderedMessagePrefix",
    "RejectedSemanticBoundary",
    "SemanticAnchor",
    "SemanticAnchorKind",
    "SemanticAnchorPlan",
    "classify_semantic_kind",
    "mandatory_semantic_edges",
    "merge_mandatory_edges",
    "plan_semantic_anchors",
    "render_message_prefixes",
]
