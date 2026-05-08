"""Regression tests for tool_specs plumbing through the postcommit path.

The next-turn prompt is encoded with `tools=tool_specs` so the chat
template injects tool definitions into the system message. Pre-fix the
postcommit pathways (`_history_ids_for_postcommit`,
`_store_retokenized_history_snapshot`, `_schedule_idle_postcommit_snapshot`)
encoded the synthetic history WITHOUT `tools=`, which produced a token
sequence that no longer matched the next prompt's prefix - SessionBank
lookups failed with `cache_miss_reason=prefix_divergence_at_token` even
when the snapshot itself stored successfully.

These tests cover:

1. The reachable-prefix invariant: storing a snapshot for messages M
   plus tools T produces a token sequence that is a STRICT prefix of
   the next prompt encoded from M' (M with one new appended user
   message) plus the same tools T.
2. The async idle-postcommit path forwards tool_specs through to
   `_store_retokenized_history_snapshot` so the stored sequence
   contains the tool-definition tokens.
"""

from concurrent.futures import ThreadPoolExecutor
import threading
from threading import Lock
from types import SimpleNamespace

import pytest

from mtplx.server import openai
from mtplx.server.openai import (
    ChatMessage,
    _encode_messages,
    _history_ids_for_postcommit,
    _schedule_idle_postcommit_snapshot,
    _store_retokenized_history_snapshot,
    parse_args,
)


def _ids(text: str) -> list[int]:
    return [ord(ch) for ch in text]


class ToolAwareTokenizer:
    """A tiny tokenizer that mimics Qwen-style tool-definition injection.

    When `tools=` is passed to `apply_chat_template`, tool definitions are
    serialised into a synthetic system message prefix. This produces a
    DIFFERENT, LONGER token sequence than encoding without `tools=` -
    matching the real-world Qwen3.6 chat template behaviour that triggered
    `prefix_divergence_at_token`.
    """

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        tools=None,
        **_kwargs,
    ):
        assert tokenize is True
        text = ""
        if tools:
            # Serialise each tool spec into a deterministic prefix so the
            # tokens are stable across calls and clearly larger than the
            # no-tools encoding.
            tool_lines = ["<tools>"]
            for spec in tools:
                fn = spec.get("function") or spec
                name = fn.get("name") or spec.get("name") or "fn"
                tool_lines.append(f"<tool name={name}/>")
            tool_lines.append("</tools>")
            text += "\n".join(tool_lines) + "\n"
        text += "\n".join(
            f"{message['role']}:{message.get('content') or ''}" for message in messages
        )
        if add_generation_prompt:
            text = f"{text}\nassistant:" if text else "assistant:"
        return _ids(text)

    def encode(self, text, **_kwargs):
        return _ids(str(text))

    def decode(self, tokens, **_kwargs):
        return "".join(chr(int(t)) for t in tokens)


class RecordingBank:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]),
            nbytes=123,
            token_hash="test-token-hash",
        )


def _postcommit_state(*, tokenizer=None):
    args = parse_args(["--warmup-tokens", "0"])
    bank = RecordingBank()
    return SimpleNamespace(
        args=args,
        runtime=SimpleNamespace(
            tokenizer=tokenizer or ToolAwareTokenizer(),
            model_path="models/test",
            mtp_enabled=True,
        ),
        sessions=SimpleNamespace(bank=bank),
        template_hash="template",
        draft_head_identity="draft-head",
        lock=Lock(),
        generation_executor=SimpleNamespace(
            submit=lambda fn, *args, **kwargs: fn(*args, **kwargs)
        ),
        postcommit_executor=SimpleNamespace(
            submit=lambda fn, *args, **kwargs: fn(*args, **kwargs)
        ),
        has_foreground=lambda: False,
        begin_foreground=lambda: None,
        end_foreground=lambda: None,
    )


_TOOL_SPECS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
]


def test_history_ids_with_tools_is_strict_prefix_of_next_prompt():
    """The reachable-prefix invariant.

    If turn 1 stores a snapshot for `messages + assistant_response` with
    `tool_specs=T`, then turn 2's prompt - encoded from the SAME messages
    plus the assistant response plus a new user message, with the SAME
    `tool_specs=T` and `add_generation_prompt=True` - must START with the
    stored token sequence. Otherwise SessionBank's strict-prefix lookup
    can never reach the snapshot.

    Pre-fix `_history_ids_for_postcommit` did NOT pass `tools=`, so the
    stored sequence diverged from the next prompt at the system-message
    boundary (where the chat template injects tool definitions).
    """
    state = _postcommit_state()
    messages = [
        ChatMessage(role="system", content="You are a helpful agent."),
        ChatMessage(role="user", content="Find the bug in main.py."),
    ]
    assistant_content = "I will search for it."

    history_ids = _history_ids_for_postcommit(
        state,
        messages=messages,
        assistant_content=assistant_content,
        assistant_tool_calls=None,
        thinking_enabled=False,
        tool_specs=_TOOL_SPECS,
    )

    # Turn 2: same conversation + assistant reply + new user turn, with
    # the same tool_specs and add_generation_prompt=True (the real
    # request path).
    next_messages = list(messages) + [
        ChatMessage(role="assistant", content=assistant_content),
        ChatMessage(role="user", content="Now fix it."),
    ]
    next_prompt_ids = _encode_messages(
        state.runtime.tokenizer,
        next_messages,
        enable_thinking=False,
        tools=_TOOL_SPECS,
    )

    assert history_ids, "history_ids must not be empty"
    assert len(history_ids) <= len(next_prompt_ids)
    assert next_prompt_ids[: len(history_ids)] == history_ids, (
        "stored history snapshot must be a strict token prefix of the "
        "next-turn prompt; otherwise SessionBank lookups will diverge"
    )


def test_history_ids_without_tools_diverges_from_next_prompt_with_tools():
    """The bug this fix addresses: pre-fix code stored history WITHOUT
    `tools=` while the next prompt was encoded WITH `tools=`. Show that
    the no-tools encoding is NOT a prefix of the with-tools next prompt.
    This is the exact failure mode that produced
    `cache_miss_reason=prefix_divergence_at_token`.
    """
    state = _postcommit_state()
    messages = [
        ChatMessage(role="system", content="You are a helpful agent."),
        ChatMessage(role="user", content="Find the bug in main.py."),
    ]
    assistant_content = "I will search for it."

    # Pre-fix encoding (no tools) - what the old `_history_ids_for_postcommit`
    # produced.
    history_ids_no_tools = _history_ids_for_postcommit(
        state,
        messages=messages,
        assistant_content=assistant_content,
        assistant_tool_calls=None,
        thinking_enabled=False,
        # tool_specs left as default None == pre-fix behaviour.
    )

    next_messages = list(messages) + [
        ChatMessage(role="assistant", content=assistant_content),
        ChatMessage(role="user", content="Now fix it."),
    ]
    next_prompt_ids_with_tools = _encode_messages(
        state.runtime.tokenizer,
        next_messages,
        enable_thinking=False,
        tools=_TOOL_SPECS,
    )

    assert history_ids_no_tools, "history_ids must not be empty"
    # The no-tools encoding is NOT a prefix of the with-tools next prompt
    # - this is the divergence the fix targets.
    assert (
        next_prompt_ids_with_tools[: len(history_ids_no_tools)]
        != history_ids_no_tools
    ), (
        "regression guard: encoding without tools must diverge from a "
        "next prompt encoded with tools - this is the prefix_divergence_at_token "
        "failure mode"
    )


def test_store_retokenized_history_snapshot_includes_tool_definition_tokens(
    monkeypatch,
):
    """`_store_retokenized_history_snapshot` must encode the synthetic
    history WITH `tools=` so the stored token sequence includes the
    tool-definition tokens. Encoding the same messages without tools
    produces a strictly shorter sequence.
    """
    state = _postcommit_state()
    messages = [
        ChatMessage(role="system", content="You are a helpful agent."),
        ChatMessage(role="user", content="Find the bug in main.py."),
    ]
    assistant_content = "I will search for it."

    # Mock the runtime-heavy prefill/cache calls; we are validating
    # ENCODING + bank.put(token_ids=...), not the runtime KV-cache path.
    fake_prompt_state = SimpleNamespace(
        trunk_cache="trunk-cache",
        logits="logits",
        hidden="hidden",
        committed_mtp_cache=None,
    )
    monkeypatch.setattr(
        openai,
        "restore_or_prefill_prompt_state",
        lambda *args, **kwargs: fake_prompt_state,
    )
    monkeypatch.setattr(openai, "snapshot_cache", lambda cache: None)

    result = _store_retokenized_history_snapshot(
        state,
        session_id="test-session",
        messages=messages,
        assistant_content=assistant_content,
        thinking_enabled=False,
        policy_fingerprint="test-policy",
        tool_specs=_TOOL_SPECS,
    )

    assert result["stored"] is True, result
    assert state.sessions.bank.puts, "bank.put must be called"
    stored_ids = state.sessions.bank.puts[0]["token_ids"]

    # Encode the same history without tools and confirm the stored
    # sequence is STRICTLY LONGER (has the tool-definition prefix).
    history_messages_no_tools = list(messages) + [
        ChatMessage(role="assistant", content=assistant_content),
    ]
    no_tools_ids = _encode_messages(
        state.runtime.tokenizer,
        history_messages_no_tools,
        enable_thinking=False,
        add_generation_prompt=False,
    )

    assert len(stored_ids) > len(no_tools_ids), (
        f"stored sequence ({len(stored_ids)} tokens) must be longer than "
        f"the no-tools encoding ({len(no_tools_ids)} tokens) - the tool "
        "definitions should add tokens"
    )

    # And it must be reachable as a prefix of the next-turn prompt.
    next_messages = list(messages) + [
        ChatMessage(role="assistant", content=assistant_content),
        ChatMessage(role="user", content="Now fix it."),
    ]
    next_prompt_ids = _encode_messages(
        state.runtime.tokenizer,
        next_messages,
        enable_thinking=False,
        tools=_TOOL_SPECS,
    )
    assert next_prompt_ids[: len(stored_ids)] == stored_ids


def _async_state(*, foreground_always: bool = False, tokenizer=None):
    """A more permissive state for async-path testing that exposes the
    same surface as `_schedule_idle_postcommit_snapshot` expects."""

    def has_foreground() -> bool:
        return True if foreground_always else False

    args = parse_args(["--warmup-tokens", "0"])
    bank = RecordingBank()
    return SimpleNamespace(
        args=args,
        runtime=SimpleNamespace(
            tokenizer=tokenizer or ToolAwareTokenizer(),
            model_path="models/test",
            mtp_enabled=True,
        ),
        sessions=SimpleNamespace(bank=bank),
        template_hash="template",
        draft_head_identity="draft-head",
        lock=threading.Lock(),
        has_foreground=has_foreground,
        postcommit_executor=None,
        generation_executor=ThreadPoolExecutor(max_workers=1),
    )


def test_idle_postcommit_async_path_forwards_tool_specs(monkeypatch):
    """The async idle path must forward `tool_specs` to
    `_store_retokenized_history_snapshot`. Without the forwarding, the
    stored sequence would diverge from the next prompt's prefix
    (`prefix_divergence_at_token`).
    """
    state = _async_state(foreground_always=True)
    captured: list[dict] = []

    def fake_store(_state, **kwargs):
        captured.append(kwargs)
        return {
            "stored": True,
            "mode": "retokenized_history",
            "prefix_len": 17,
            "nbytes": 4242,
        }

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    pending = _schedule_idle_postcommit_snapshot(
        state,
        session_id="opencode-researcher",
        messages=[ChatMessage(role="user", content="hi")],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy",
        unsafe_reason="retokenized_history_mismatch",
        tool_specs=_TOOL_SPECS,
    )
    state.generation_executor.shutdown(wait=True)

    assert pending["mode"] == "async_pending"
    assert len(captured) == 1
    assert captured[0]["tool_specs"] == _TOOL_SPECS, (
        "scheduled async commit must receive the same tool_specs that "
        "produced the request's prompt; otherwise the stored snapshot "
        "diverges from the next prompt's prefix"
    )


def test_idle_postcommit_default_tool_specs_is_none(monkeypatch):
    """Backwards compat: the four pre-existing tests in
    `test_idle_postcommit_subagent.py` do not pass `tool_specs`. Confirm
    the default value remains `None` so the legacy callers see no
    behaviour change.
    """
    state = _async_state(foreground_always=False)
    captured: list[dict] = []

    def fake_store(_state, **kwargs):
        captured.append(kwargs)
        return {"stored": True, "mode": "retokenized_history"}

    monkeypatch.setattr(openai, "_store_retokenized_history_snapshot", fake_store)
    monkeypatch.setattr(openai, "_server_console_enabled", lambda _state: True)

    _schedule_idle_postcommit_snapshot(
        state,
        session_id="legacy-session",
        messages=[ChatMessage(role="user", content="hi")],
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy",
        unsafe_reason="retokenized_history_mismatch",
    )
    state.generation_executor.shutdown(wait=True)

    assert captured and captured[0]["tool_specs"] is None
