"""Opt out of the no-tools markup strip (MTPLX_KEEP_ORPHAN_TOOL_MARKUP=on).

A text-only client that drives the model with its own plain-text protocol
(`RUN: <cmd>`) declares no tools, exactly like a chat window. When the model
wraps that intent in its trained `<tool_call>` syntax — which it does
intermittently — the strip deletes the command the client was waiting for and
substitutes the "no tools are active" notice. The client cannot tell an eaten
command from an empty answer, re-issues the turn, and the notice also lands in
the assistant history the client replays.

Stripping stays the default: the app chat must never render raw
`<tool_call>` XML (#160), and a swallowed call must never be silent (#349).
The opt-out is for callers that parse the text themselves, where the markup is
not dead syntax but the payload.
"""

from __future__ import annotations

import mtplx.server.openai as srv

NOTICE = "no tools are active"
CALL = '<tool_call>\n{"name": "run", "arguments": {"cmd": "echo hi"}}\n</tool_call>'


def _keep(monkeypatch):
    monkeypatch.setenv("MTPLX_KEEP_ORPHAN_TOOL_MARKUP", "on")


def _drop(monkeypatch):
    monkeypatch.delenv("MTPLX_KEEP_ORPHAN_TOOL_MARKUP", raising=False)


def test_default_is_still_strip(monkeypatch):
    _drop(monkeypatch)
    assert srv._keep_orphan_tool_markup() is False
    cleaned, count = srv._strip_orphan_tool_markup("Checking.\n" + CALL + "\nDone.")
    assert count == 1
    assert "<tool_call>" not in cleaned


def test_flag_parses_the_usual_spellings(monkeypatch):
    for spelling in ("on", "1", "true", "YES", " On "):
        monkeypatch.setenv("MTPLX_KEEP_ORPHAN_TOOL_MARKUP", spelling)
        assert srv._keep_orphan_tool_markup() is True, spelling
    for spelling in ("", "0", "off", "no", "false"):
        monkeypatch.setenv("MTPLX_KEEP_ORPHAN_TOOL_MARKUP", spelling)
        assert srv._keep_orphan_tool_markup() is False, spelling


def test_flag_does_not_change_the_strip_primitive(monkeypatch):
    # The opt-out is applied at the call sites, so the helper keeps its
    # #160 contract and stays testable on its own.
    _keep(monkeypatch)
    cleaned, count = srv._strip_orphan_tool_markup(CALL)
    assert count == 1
    assert "<tool_call>" not in cleaned


def test_stream_gate_reads_the_flag(monkeypatch):
    tools_active = False

    def _stream(keep: bool):
        splitter = srv._ThinkingContentStreamSplitter(
            thinking_enabled=False,
            suppress_orphan_tool_markup=(
                not tools_active and not srv._keep_orphan_tool_markup()
            ),
        )
        chunks = list(splitter.feed("Checking.\n" + CALL))
        chunks.extend(splitter.finish())
        return "".join(text for field, text in chunks if field == "content")

    _drop(monkeypatch)
    assert "<tool_call>" not in _stream(False)   # default posture unchanged

    _keep(monkeypatch)
    assert "<tool_call>" in _stream(True)


def test_opt_out_skips_the_notice(monkeypatch):
    _keep(monkeypatch)
    # The notice is produced only on the strip path; with the flag on,
    # orphan_blocks stays 0 and neither the notice nor the stats stamp fires.
    display_text = "Checking.\n" + CALL
    orphan_blocks = 0
    if not srv._keep_orphan_tool_markup():
        display_text, orphan_blocks = srv._strip_orphan_tool_markup(display_text)
    assert orphan_blocks == 0
    assert NOTICE not in display_text
    assert "<tool_call>" in display_text
