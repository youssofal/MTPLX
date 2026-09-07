"""Aligned-boundary stable prompt prefix (2026-08-06 design).

The transient trailing tool-continuation hint (shipped since 1.0.0)
makes every tool-turn prompt diverge from the previous turn's stored
prompt at the hint's start (~215 tokens before the entry end), forcing
boundary restores that snap to the 512 grid. This change is
metadata-only: the encoder reports where the hint's user turn begins
(stable_prefix_len, token-exact via a merge-safe split at the turn's
<|im_start|> special token), and prefill span planning makes that
position a mandatory chunk edge so the EXISTING gdn-boundary capture
records recurrent state exactly there. Keys, snapshots, epochs,
rendered bytes, and tool UX are untouched; without the metadata every
span is byte-identical to before.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.cache_state import CacheSnapshot
from mtplx.generation import (
    _iter_prefill_chunk_spans,
    _prefill_spans_with_tail_grid,
    _split_spans_at,
    _thin_gdn_boundary_records,
)
from mtplx.server import openai as oa
from mtplx.session_bank import SessionBank


def _assert_contiguous(spans, total):
    assert spans[0][0] == 0
    assert spans[-1][1] == total
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert b == c, f"gap/overlap between {(a, b)} and {(c, d)}"
        assert a < b and c < d


def test_split_spans_at_inserts_edges_without_gaps_or_overlap():
    spans = [(0, 100), (100, 250)]
    out = _split_spans_at(spans, (40, 100, 170, 0, 250, 999))
    _assert_contiguous(out, 250)
    ends = [e for _, e in out]
    assert 40 in ends and 170 in ends
    assert out == [(0, 40), (40, 100), (100, 170), (170, 250)]


def test_split_spans_at_noop_without_edges():
    spans = [(0, 7), (7, 9)]
    assert _split_spans_at(spans, ()) == spans


def test_cold_chunk_spans_include_mandatory_edge(monkeypatch):
    monkeypatch.delenv("MTPLX_SUSTAINED_PREFILL", raising=False)
    base = _iter_prefill_chunk_spans(600)
    with_edge = _iter_prefill_chunk_spans(600, mandatory_edges=(217,))
    _assert_contiguous(with_edge, 600)
    assert 217 in [e for _, e in with_edge]
    # No-metadata behavior is byte-identical.
    assert _iter_prefill_chunk_spans(600) == base


def test_tail_grid_spans_include_mandatory_edge():
    base = _prefill_spans_with_tail_grid(2000, tail_interval=512)
    with_edge = _prefill_spans_with_tail_grid(
        2000, tail_interval=512, mandatory_edges=(1723,)
    )
    _assert_contiguous(with_edge, 2000)
    assert 1723 in [e for _, e in with_edge]
    assert _prefill_spans_with_tail_grid(2000, tail_interval=512) == base
    # Edge already on a grid end: layout unchanged.
    on_grid = _prefill_spans_with_tail_grid(
        2000, tail_interval=512, mandatory_edges=(base[-1][0],)
    )
    assert on_grid == base


def test_segmented_encode_reports_cumulative_counts(monkeypatch):
    monkeypatch.setattr(
        oa, "_encode_rendered_chat_text", lambda tok, text: [0] * len(text)
    )
    rendered = "A" * 30 + "B" * 20 + "C" * 10
    counts = {30: -1, 50: -1}
    ids = oa._encode_rendered_chat_text_segmented(
        None, rendered, [30, 50], token_counts_at=counts
    )
    assert len(ids) == 60
    assert counts == {30: 30, 50: 50}


def test_trailing_hint_boundary_includes_shared_marker():
    """Both renders share the hint turn's <|im_start|> and diverge on the
    FOLLOWING role token, so the reported boundary must sit immediately
    AFTER the shared marker — the captured edge then equals the live
    common prefix (actual_restore_point == requested matched), leaving no
    valid token behind."""
    hint = oa._mtplx_tool_result_continuation_hint_text()
    tail = "<|im_start|>user\n" + hint + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
    rendered = "<|im_start|>user\nreal question<|im_end|>\n" + tail
    pos = oa._trailing_tool_hint_char_boundary(rendered)
    turn_at = rendered.rfind("<|im_start|>user\n" + hint[:48])
    assert pos == turn_at + len("<|im_start|>")
    assert rendered[pos : pos + 5] == "user\n"
    # Absent hint -> None.
    assert oa._trailing_tool_hint_char_boundary("<|im_start|>user\nhi<|im_end|>\n") is None
    # Lookalike with additional turns after it -> rejected.
    echo = rendered + "<|im_start|>user\nmore<|im_end|>\n<|im_start|>assistant\n"
    assert oa._trailing_tool_hint_char_boundary(echo) is None


def test_production_metadata_path_reports_stable_prefix(monkeypatch):
    """Exercise the real writer (_encode_with_stable_hint_boundary): render
    via the tokenizer's template call, split at the shared marker, and write
    template_observability[stable_prefix_len] with shared-marker inclusion."""
    hint = oa._mtplx_tool_result_continuation_hint_text()
    rendered = (
        "<|im_start|>user\nreal question<|im_end|>\n"
        "<|im_start|>user\n" + hint + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )

    class StubTokenizer:
        def apply_chat_template(self, normalized, **kwargs):
            assert kwargs.get("tokenize") is False
            return rendered

    monkeypatch.setattr(
        oa, "_encode_rendered_chat_text", lambda tok, text: [0] * len(text)
    )
    observability: dict = {}
    normalized = [
        {"role": "user", "content": "real question"},
        {"role": "user", "content": hint},
    ]
    ids = oa._encode_with_stable_hint_boundary(
        StubTokenizer(),
        normalized,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort=None,
        preserve_thinking=False,
        tools=None,
        template_observability=observability,
    )
    assert ids is not None and len(ids) == len(rendered)
    expected = rendered.rfind("<|im_start|>user\n" + hint[:48]) + len("<|im_start|>")
    # Char-per-token stub: the token count at the boundary equals the char
    # position, proving shared-marker inclusion end to end.
    assert observability["stable_prefix_len"] == expected


AGENT_TOOLS = [
    {"type": "function", "function": {"name": name, "parameters": {"type": "object"}}}
    for name in ("read", "bash", "edit", "write")
]


class _TemplateStub:
    """Deterministic template tokenizer: role-tagged concatenation, char ids."""

    def apply_chat_template(self, normalized, **kwargs):
        rendered = "".join(
            f"<|im_start|>{m.get('role')}\n{m.get('content') or ''}<|im_end|>\n"
            for m in normalized
        )
        if kwargs.get("add_generation_prompt"):
            rendered += "<|im_start|>assistant\n<think>\n"
        if kwargs.get("tokenize") is False:
            return rendered
        return [0] * len(rendered)


def _encode_production(monkeypatch, messages, *, tool_prompt_mode="hybrid"):
    monkeypatch.setattr(
        oa, "_encode_rendered_chat_text", lambda tok, text: [0] * len(text)
    )
    observability: dict = {}
    ids = oa._encode_messages_uncached(
        _TemplateStub(),
        [oa.ChatMessage(**m) for m in messages],
        enable_thinking=True,
        scoped_reasoning_history=True,
        add_generation_prompt=True,
        tools=AGENT_TOOLS,
        tool_choice="auto",
        tool_prompt_mode=tool_prompt_mode,
        template_observability=observability,
    )
    return ids, observability


def test_real_trailing_tool_injection_writes_stable_metadata(legacy_rewrites, monkeypatch):
    """Production path: a genuine trailing tool result triggers the injector,
    the explicit append signal gates the stable-boundary encoder, and
    stable_prefix_len lands."""
    messages = [
        {"role": "user", "content": "run the check"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "12 passed"},
    ]
    ids, observability = _encode_production(monkeypatch, messages)
    assert observability.get("tool_result_continuation_hint_injected") is True
    stable = observability.get("stable_prefix_len")
    assert isinstance(stable, int) and 0 < stable < len(ids)


def test_user_authored_lookalike_writes_no_stable_metadata(monkeypatch):
    """Injection-only contract: a user-authored final message that exactly
    starts with the internal hint text must not perturb chunk layout or
    telemetry — no injected flag, no stable_prefix_len."""
    hint = oa._mtplx_tool_result_continuation_hint_text()
    messages = [
        {"role": "user", "content": "run the check"},
        {"role": "user", "content": hint},
    ]
    ids, observability = _encode_production(monkeypatch, messages)
    assert ids, "encode must still succeed via the untouched plain path"
    assert "tool_result_continuation_hint_injected" not in observability
    assert "stable_prefix_len" not in observability


def test_native_tail_records_explicit_append_signal(legacy_rewrites):
    observability: dict = {}
    messages = [
        {"role": "user", "content": "task"},
        {"role": "tool", "tool_call_id": "c", "content": "out"},
    ]
    out, tail_added = oa._with_mtplx_native_agent_tail(
        messages, tools=AGENT_TOOLS, observability=observability
    )
    assert observability.get("tool_result_continuation_hint_injected") is True
    assert out[-1]["role"] == "user"
    # Callers without a sink keep working (optional default).
    out2, _ = oa._with_mtplx_native_agent_tail(messages, tools=AGENT_TOOLS)
    assert out2[-1]["role"] == "user"


def test_boundary_restore_route_at_restore_point_equal_matched_with_hidden():
    """The amended design's serving route: a boundary captured exactly at the
    stable edge restores at restore_point == matched WITH boundary_hidden."""
    bank = SessionBank(max_entries=4, max_bytes=4096, per_session_max_bytes=4096)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    entry = bank.put(
        runtime=runtime,
        token_ids=list(range(1, 61)),
        cache=[],
        logits=None,
        hidden=None,
        session_id="s",
        nbytes_override=64,
    )
    assert entry is not None
    entry.has_recurrent = True
    sentinel_hidden = object()
    entry.gdn_boundaries = [
        (40, CacheSnapshot(states=(), meta_states=()), sentinel_hidden)
    ]
    result = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        40,  # matched exactly at the captured stable edge
        mode="clone",
        cache_factory=list,
    )
    assert result is not None
    cache, mtp_cache, mode, restore_point, boundary_hidden = result
    assert restore_point == 40
    assert boundary_hidden is sentinel_hidden
    assert mode == "clone"


def test_thinning_retains_tail_adjacent_stable_edge():
    snap = CacheSnapshot(states=(), meta_states=())
    records = [(pos, snap, None) for pos in range(512, 15873, 512)]
    stable_edge = (15723, snap, "hidden")
    records.append(stable_edge)
    thinned = _thin_gdn_boundary_records(sorted(records, key=lambda r: r[0]), 8)
    assert len(thinned) <= 8
    assert any(r[0] == 15723 for r in thinned), (
        "tail-adjacent stable edge must survive geometric thinning"
    )


def test_128k_pretool_anchor_survives_capture_grid_thinning():
    prompt_body_tokens = 126_687
    pretool_edge = 126_683
    spans = _prefill_spans_with_tail_grid(
        prompt_body_tokens,
        tail_interval=512,
        mandatory_edges=(pretool_edge,),
        chunk_size=8192,
    )
    records = []
    for _start, end in spans:
        records.append((end, object(), None))
        if len(records) > 8:
            records = _thin_gdn_boundary_records(records, 8)

    assert pretool_edge in {record[0] for record in records}
    assert prompt_body_tokens in {record[0] for record in records}


def test_oversized_live_frontier_retains_pretool_boundary_for_restore():
    pretool_edge = 126_683
    stored_tokens = [*range(pretool_edge), *range(200_000, 200_032)]
    followup_tokens = [*range(pretool_edge), *range(300_000, 301_043)]
    boundary_state = CacheSnapshot(states=(), meta_states=())
    boundary_hidden = object()
    recurrent_cache = SimpleNamespace(is_trimmable=lambda: False)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)
    bank = SessionBank(max_entries=4, max_bytes=4096, per_session_max_bytes=64)

    entry = bank.put(
        runtime=runtime,
        token_ids=stored_tokens,
        cache=[recurrent_cache],
        logits=None,
        hidden=None,
        keep_live_ref=True,
        session_id="tool-turn",
        nbytes_override=1024,
        gdn_boundaries=[(pretool_edge, boundary_state, boundary_hidden)],
    )

    assert entry is not None and entry.live_ref_only
    assert [record[0] for record in entry.gdn_boundaries] == [pretool_edge]
    candidates = bank.near_prefix_candidates(followup_tokens)
    assert candidates == [(entry, pretool_edge)]
    restored = bank.restore_entry_prefix_cache(
        runtime,
        entry,
        pretool_edge,
        mode="reference",
    )
    assert restored is not None
    _cache, _mtp_cache, mode, restore_point, hidden = restored
    assert mode == "reference_lease"
    assert restore_point == pretool_edge
    assert hidden is boundary_hidden


class _NearPrefixProbe(Exception):
    """Raised by the recorder to stop restore_or_prefill before any real work."""


def test_near_prefix_lane_one_forwards_stable_prefix_len(monkeypatch):
    """Defect A regression (2026-08-16): the FIRST near-prefix lane — the one
    every warm tool round takes when a shorter entry shadows the prompt —
    must forward stable_prefix_len so its suffix prefill captures the
    pre-nudge recurrent boundary. Without it each banked tool-round entry
    lacks the stable-edge snapshot and the NEXT round block-rounds down."""
    import mtplx.generation as generation

    captured: dict[str, object] = {}

    def _recorder(rt, prompt_ids, **kwargs):
        captured.update(kwargs)
        raise _NearPrefixProbe()

    monkeypatch.setattr(generation, "_restore_near_prefix_prompt_state", _recorder)

    bank = SessionBank(max_entries=4, max_bytes=4096, per_session_max_bytes=4096)
    runtime = SimpleNamespace(
        model_path=Path("models/example"),
        mtp_enabled=False,
        contract=SimpleNamespace(),
    )
    prompt_ids = list(range(1, 121))
    entry = bank.put(
        runtime=runtime,
        token_ids=prompt_ids[:60],  # strict prefix -> exact_prefix_len < len
        cache=[],
        logits=None,
        hidden=None,
        session_id="s",
        nbytes_override=64,
    )
    assert entry is not None

    try:
        generation.restore_or_prefill_prompt_state(
            runtime,
            prompt_ids,
            mtp_history_policy="cycle",
            session_bank=bank,
            session_id="s",
            stable_prefix_len=97,
        )
    except _NearPrefixProbe:
        pass
    else:
        raise AssertionError("near-prefix lane 1 never fired for a shorter entry")

    assert captured.get("stable_prefix_len") == 97, (
        "lane-1 near-prefix restore dropped stable_prefix_len: "
        f"forwarded kwargs {sorted(captured)}"
    )
