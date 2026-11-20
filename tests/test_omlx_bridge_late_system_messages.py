"""Issue #477: a mid-conversation system message must not rewrite message 0."""

from __future__ import annotations

from mtplx.server.omlx_bridge.adapter import _consolidate_system_messages, _merge_consecutive_roles


def _convo(n: int) -> list[dict]:
    messages = [
        {"role": "system", "content": "S0"},
        {"role": "user", "content": "turn 1"},
    ]
    for i in range(1, n + 1):
        messages += [
            {"role": "assistant", "content": f"answer {i}"},
            {"role": "system", "content": f"<system-reminder>note {i}</system-reminder>"},
            {"role": "user", "content": f"turn {i + 1}"},
        ]
    return messages


def test_leading_system_messages_are_joined_into_message_zero():
    out = _consolidate_system_messages(
        [{"role": "system", "content": "A"}, {"role": "system", "content": "B"}, {"role": "user", "content": "hi"}]
    )
    assert out[0] == {"role": "system", "content": "A\n\nB"}
    assert out[1:] == [{"role": "user", "content": "hi"}]


def test_a_late_system_message_stays_in_place_as_a_user_turn():
    out = _consolidate_system_messages(_convo(1))
    assert out[0]["content"] == "S0"  # message 0 untouched by the reminder
    roles = [m["role"] for m in out]
    assert roles == ["system", "user", "assistant", "user", "user"]
    assert out[3]["content"] == "<system-reminder>note 1</system-reminder>"
    merged = _merge_consecutive_roles(out)
    assert [m["role"] for m in merged] == ["system", "user", "assistant", "user"]
    assert merged[3]["content"] == "<system-reminder>note 1</system-reminder>\n\nturn 2"


def test_the_rendered_prefix_grows_by_whole_turns():
    """Consecutive turns share everything up to the newest reminder."""

    previous = None
    for n in range(1, 5):
        current = _merge_consecutive_roles(_consolidate_system_messages(_convo(n)))
        if previous is not None:
            # every message of the previous conversation is byte-identical
            # in the new one except the final user turn, which grew
            assert current[: len(previous) - 1] == previous[:-1]
        previous = current


def test_the_prefix_still_grows_by_whole_turns_when_late_system_turns_are_wrapped():
    """An unwrapped late system turn is wrapped the same way on every render,
    so each completed turn stays byte-identical on the next request."""

    def convo(n: int) -> list[dict]:
        messages = [{"role": "system", "content": "S0"}, {"role": "user", "content": "turn 1"}]
        for i in range(1, n + 1):
            messages += [
                {"role": "assistant", "content": f"answer {i}"},
                {"role": "system", "content": f"# Environment\nnote {i}"},
                {"role": "user", "content": f"turn {i + 1}"},
            ]
        return messages

    previous = None
    for n in range(1, 5):
        current = _merge_consecutive_roles(_consolidate_system_messages(convo(n)))
        assert current[-1]["content"].startswith("<system-reminder>\n# Environment\n")
        if previous is not None:
            assert current[: len(previous) - 1] == previous[:-1]
        previous = current


def test_no_system_messages_is_a_no_op():
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert _consolidate_system_messages(messages) == messages


def test_only_late_system_messages_without_a_leading_one():
    out = _consolidate_system_messages(
        [{"role": "user", "content": "hi"}, {"role": "system", "content": "note"}, {"role": "user", "content": "go"}]
    )
    assert [m["role"] for m in out] == ["user", "user", "user"]
    assert out[1]["content"] == "<system-reminder>\nnote\n</system-reminder>"


def test_an_unwrapped_late_system_turn_reads_as_system_context_not_the_users_words():
    """Claude Code 2.1.3xx sends its session context (environment, MCP
    instructions, skill listing) as one unwrapped system turn right after the
    first user message. On 16 Nov it was folded into "yo" as plain text, and
    a skill description's "run this grep FIRST" read as the user's order."""

    context = "# Environment\nPlatform: darwin\n\nThe following skills are available: ... (run this grep FIRST ...)"
    rendered = _merge_consecutive_roles(
        _consolidate_system_messages(
            [
                {"role": "system", "content": "You are Claude Code."},
                {"role": "user", "content": "yo"},
                {"role": "system", "content": context},
                {"role": "assistant", "content": "Hey!"},
            ]
        )
    )
    assert [m["role"] for m in rendered] == ["system", "user", "assistant"]
    assert rendered[0]["content"] == "You are Claude Code."
    assert rendered[1]["content"] == f"yo\n\n<system-reminder>\n{context}\n</system-reminder>"


def test_a_reminder_block_is_not_wrapped_twice():
    reminder = "<system-reminder>note</system-reminder>"
    out = _consolidate_system_messages(
        [{"role": "user", "content": "hi"}, {"role": "system", "content": f"  {reminder}\n"}]
    )
    assert out[1]["content"] == f"  {reminder}\n"


def test_a_conversation_without_late_system_messages_renders_as_before():
    """Requests with no mid-conversation system turn do not change at all."""

    messages = [
        {"role": "system", "content": "S0"},
        {"role": "user", "content": "turn 1"},
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "<system-reminder>a client's own note</system-reminder>\n\nturn 2"},
    ]
    assert _consolidate_system_messages(messages) == messages
