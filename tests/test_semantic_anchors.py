from __future__ import annotations

from dataclasses import replace

import pytest

from mtplx.semantic_anchors import (
    RenderedMessagePrefix,
    SemanticAnchorKind,
    classify_semantic_kind,
    mandatory_semantic_edges,
    merge_mandatory_edges,
    plan_semantic_anchors,
    render_message_prefixes,
)


def _prefix(index: int, role: str, end: int, **metadata):
    return RenderedMessagePrefix.from_tokens(
        message_index=index,
        role=role,
        token_ids=range(1, end + 1),
        metadata=metadata,
        estimated_checkpoint_bytes=2,
    )


def test_role_and_metadata_classification():
    assert classify_semantic_kind("system") == SemanticAnchorKind.INSTRUCTIONS_END
    assert classify_semantic_kind("developer") == SemanticAnchorKind.INSTRUCTIONS_END
    assert classify_semantic_kind("user") == SemanticAnchorKind.USER_TURN_END
    assert classify_semantic_kind("tool") == SemanticAnchorKind.TOOL_RESULT_END
    assert classify_semantic_kind("assistant", {"tool_calls": [{}]}) == SemanticAnchorKind.TOOL_CALL_END
    assert classify_semantic_kind("assistant") == SemanticAnchorKind.ASSISTANT_TURN_END
    assert classify_semantic_kind("assistant", {"is_reasoning": True}) == SemanticAnchorKind.REASONING_END
    assert classify_semantic_kind("user", {"summary": True}) == SemanticAnchorKind.COMPACTION_END
    assert classify_semantic_kind("assistant", {"semantic_kind": "reasoning_start"}) == SemanticAnchorKind.REASONING_START


def test_exact_prefixes_are_selected_and_mismatches_fail_closed():
    final = tuple(range(1, 25))
    prefixes = [
        _prefix(0, "system", 4),
        _prefix(1, "user", 8),
        _prefix(2, "assistant", 12, tool_calls=[{"id": "x"}]),
        _prefix(3, "tool", 18, tool_call_id="x"),
    ]
    bad = replace(prefixes[-1], token_ids=prefixes[-1].token_ids[:-1] + (999,))
    plan = plan_semantic_anchors(final, [*prefixes[:-1], bad], max_anchors=8)
    assert plan.edges == (4, 8, 12)
    assert [item.reason for item in plan.rejected] == ["not_exact_prefix"]
    assert plan.rejected[0].mismatch_index == 17


@pytest.mark.parametrize(
    ("mutation_offset", "expected_edges"),
    [
        # Remove/replace an old reasoning block: only earlier exact message prefixes survive.
        (9, (4, 8)),
        # Replace an old tool result: the tool-call anchor survives, later history does not.
        (14, (4, 8, 12)),
        # Truncate an observation near the tail.
        (19, (4, 8, 12, 18)),
        # Final-user-only extension: every prior semantic boundary remains exact.
        (25, (4, 8, 12, 18, 24)),
    ],
)
def test_agent_history_mutation_matrix(mutation_offset: int, expected_edges: tuple[int, ...]):
    base = list(range(1, 31))
    if mutation_offset < len(base):
        base[mutation_offset - 1] = 9999
    final = tuple(base)
    prefixes = [
        _prefix(0, "system", 4),
        _prefix(1, "user", 8),
        _prefix(2, "assistant", 12, reasoning_boundary="end"),
        _prefix(3, "assistant", 18, tool_calls=[{"id": "call"}]),
        _prefix(4, "tool", 24, tool_call_id="call"),
    ]
    plan = plan_semantic_anchors(final, prefixes, max_anchors=8)
    assert plan.edges == expected_edges


def test_compaction_summary_is_prioritized_under_count_budget():
    final = tuple(range(1, 41))
    prefixes = [
        _prefix(0, "system", 4),
        _prefix(1, "user", 10),
        _prefix(2, "assistant", 16),
        _prefix(3, "user", 24, summary=True),
        _prefix(4, "assistant", 32, tool_calls=[{"id": "x"}]),
        _prefix(5, "tool", 38, tool_call_id="x"),
    ]
    plan = plan_semantic_anchors(final, prefixes, max_anchors=3)
    assert {anchor.kind for anchor in plan.anchors} == {
        SemanticAnchorKind.COMPACTION_END,
        SemanticAnchorKind.TOOL_CALL_END,
        SemanticAnchorKind.TOOL_RESULT_END,
    }
    assert sum(item.reason == "count_budget" for item in plan.rejected) == 3


def test_latest_message_frontier_survives_tool_heavy_count_budget():
    prefixes = []
    token_offset = 4
    for index in range(14):
        role = "assistant" if index % 2 == 0 else "tool"
        metadata = (
            {"tool_calls": [{"id": f"call-{index // 2}"}]}
            if role == "assistant"
            else {"tool_call_id": f"call-{index // 2}"}
        )
        prefixes.append(_prefix(index, role, token_offset, **metadata))
        token_offset += 4
    prefixes.append(_prefix(14, "user", token_offset))

    final = tuple(range(1, token_offset + 6))
    plan = plan_semantic_anchors(final, prefixes, max_anchors=8)

    assert token_offset in plan.edges
    frontier = next(
        anchor for anchor in plan.anchors if anchor.token_offset == token_offset
    )
    assert frontier.kind == SemanticAnchorKind.USER_TURN_END
    assert sum(item.reason == "count_budget" for item in plan.rejected) == 7


def test_byte_budget_is_strict_and_deterministic():
    final = tuple(range(1, 31))
    prefixes = [
        replace(_prefix(0, "system", 5), estimated_checkpoint_bytes=8),
        replace(_prefix(1, "assistant", 15), estimated_checkpoint_bytes=8),
        replace(_prefix(2, "tool", 25), estimated_checkpoint_bytes=8),
    ]
    first = plan_semantic_anchors(
        final,
        prefixes,
        max_anchors=8,
        max_checkpoint_bytes=16,
    )
    second = plan_semantic_anchors(
        final,
        reversed(prefixes),
        max_anchors=8,
        max_checkpoint_bytes=16,
    )
    assert first.anchors == second.anchors
    assert first.estimated_checkpoint_bytes == 16
    assert SemanticAnchorKind.TOOL_RESULT_END in {anchor.kind for anchor in first.anchors}
    assert any(item.reason == "byte_budget" for item in first.rejected)


def test_render_message_prefixes_uses_complete_template_prefixes():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a", "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "r", "tool_call_id": "x"},
    ]

    def render(prefix):
        return tuple(range(1, len(prefix) * 4 + 1))

    prefixes = render_message_prefixes(messages, render)
    plan = plan_semantic_anchors(render(messages), prefixes)
    assert plan.edges == (4, 8, 12, 16)
    assert plan.anchors[-1].kind == SemanticAnchorKind.TOOL_RESULT_END


def test_128k_tool_turn_keeps_complete_user_frontier_when_prompt_tail_changes():
    stable_prefix = tuple(range(1, 126_684))
    base_prompt = (*stable_prefix, 900_001, 900_002, 900_003, 900_004, 900_005)
    tool_turn_prompt = (
        *stable_prefix,
        900_001,
        900_002,
        900_003,
        900_004,
        999_999,
        *range(1_000_000, 1_001_038),
    )
    pre_tool = RenderedMessagePrefix.from_tokens(
        message_index=1,
        role="user",
        token_ids=stable_prefix,
    )

    plan = plan_semantic_anchors(base_prompt, [pre_tool])

    assert len(base_prompt) == 126_688
    assert len(tool_turn_prompt) - len(stable_prefix) == 1_043
    assert base_prompt[:126_687] == tool_turn_prompt[:126_687]
    assert base_prompt[126_687] != tool_turn_prompt[126_687]
    assert stable_prefix == tool_turn_prompt[: len(stable_prefix)]
    assert plan.edges == (len(stable_prefix),)


def test_mandatory_edges_are_strictly_inside_requested_span():
    final = tuple(range(1, 21))
    plan = plan_semantic_anchors(
        final,
        [_prefix(0, "system", 4), _prefix(1, "user", 10), _prefix(2, "assistant", 20)],
    )
    assert mandatory_semantic_edges(plan, lower_bound=4, upper_bound=20) == (10,)
    assert merge_mandatory_edges((2, 8), (8, 12), None, lower_bound=2, upper_bound=12) == (8,)


def test_metrics_are_json_primitive_only():
    plan = plan_semantic_anchors(
        tuple(range(1, 20)),
        [_prefix(0, "system", 4), _prefix(1, "tool", 12)],
    )
    metrics = plan.to_metrics()
    assert metrics["semantic_anchor_count"] == 2
    assert metrics["semantic_anchor_edges"] == [4, 12]
    assert metrics["semantic_anchor_kinds"]["tool_result_end"] == 1


def test_render_message_prefixes_can_limit_expensive_candidates():
    calls = []
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
    ]

    def render(prefix):
        calls.append(len(prefix))
        return list(range(len(prefix)))

    prefixes = render_message_prefixes(
        messages,
        render,
        message_indexes=(0, 3, 3, 99),
    )
    assert calls == [1, 4]
    assert [prefix.message_index for prefix in prefixes] == [0, 3]
