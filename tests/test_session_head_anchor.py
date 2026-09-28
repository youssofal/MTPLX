"""Session-head anchor (``MTPLX_SESSION_HEAD_ANCHOR``, default off).

New agent sessions share a fixed head (tools and system text) and diverge at
the first user message. Before the anchor, a new session restored only from
the oldest GDN boundary (2,048): the geometric thinning to 8 boundaries per
entry keeps the oldest and records near the tail, and the donor entry is
nearly always a long conversation. These tests pin:

* the head end is found in the prompt tokens (the first non-system
  ``<|im_start|>``), for fake and real ChatML templates, with and without
  tools or a system turn, for chat completions and ``/v1/messages``;
* the anchor becomes a span end of the prefill plan, survives thinning, the
  sink's retention, inheritance between entries and the SSD tier;
* a bank simulation of a long chat (real planner, retention and SessionBank):
  a new session restores at the anchor instead of 2,048;
* on a tiny A3B model with the invariant lane: the anchor does not change the
  prefill result, and a new session restored at the anchor (RAM or SSD) gives
  the cold prefill's logits bit for bit;
* switch off: no anchors, the old behaviour.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx import generation, runtime_options
from mtplx.cache_bank import SessionBankColdTier
from mtplx.cache_state import snapshot_cache
from mtplx.session_bank import CacheSnapshot, SessionBank
from mtplx.session_head_anchor import (
    TurnMarkers,
    session_head_length,
    turn_markers,
)

SWITCH = runtime_options.SESSION_HEAD_ANCHOR_ENV
IM, SYSTEM, USER, ASSISTANT, END = 5, 6, 7, 8, 9
MARKERS = TurnMarkers(turn_open=IM, system_role=(SYSTEM,))


@dataclass
class _AddedToken:
    content: str
    normalized: bool = False
    lstrip: bool = False
    rstrip: bool = False


class _Tokenizer:
    """ChatML markers only: ``<|im_start|>`` is added token IM, "system" is SYSTEM."""

    def __init__(self, turn_open: _AddedToken | None = None, im: int = IM, system: int = SYSTEM):
        self.added_tokens_decoder = {} if turn_open is None else {im: turn_open}
        self._system = system

    def encode(self, text, add_special_tokens=False):
        return [self._system] if text == "system" else [0]


def _turn(role: int, body: list[int]) -> list[int]:
    return [IM, role, *body, END]


def _filler(count: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 30000) for _ in range(count)]


@pytest.fixture
def anchor_on(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    return monkeypatch


# ---------------------------------------------------------------------------
# Where the head ends
# ---------------------------------------------------------------------------


def test_head_ends_where_the_first_non_system_turn_opens():
    system = _turn(SYSTEM, [11, 12, 13])
    ids = system + _turn(USER, [21, 22]) + _turn(ASSISTANT, [31])
    assert session_head_length(ids, MARKERS) == len(system)
    assert session_head_length(tuple(ids), MARKERS) == len(system)


def test_consecutive_system_turns_are_all_head():
    head = _turn(SYSTEM, [11]) + _turn(SYSTEM, [12, 13])
    assert session_head_length(head + _turn(USER, [21]), MARKERS) == len(head)


def test_no_head_without_a_system_turn_or_without_a_later_turn():
    assert session_head_length(_turn(USER, [21]) + _turn(ASSISTANT, [31]), MARKERS) is None
    assert session_head_length(_turn(SYSTEM, [11, 12]), MARKERS) is None
    assert session_head_length([11, 12, 13], MARKERS) is None


def test_markers_need_an_atomic_turn_open():
    assert turn_markers(_Tokenizer(_AddedToken("<|im_start|>"))) == MARKERS
    assert turn_markers(_Tokenizer(_AddedToken("<|im_start|>", normalized=True))) is None
    assert turn_markers(_Tokenizer(None)) is None
    assert turn_markers(SimpleNamespace()) is None
    assert turn_markers(None) is None


def _rt(tokenizer=None):
    return SimpleNamespace(tokenizer=tokenizer or _Tokenizer(_AddedToken("<|im_start|>")))


def test_the_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    ids = _turn(SYSTEM, _filler(600, 1)) + _turn(USER, [21])
    assert generation._session_head_anchors(_rt(), ids) == ()


def test_anchors_with_the_switch_on(anchor_on):
    system = _turn(SYSTEM, _filler(600, 1))
    ids = system + _turn(USER, [21])
    assert generation._session_head_anchors(_rt(), ids) == (len(system),)
    # Image prompts, a tokenizer without ChatML turns and a head below the
    # block-restore minimum get none.
    assert generation._session_head_anchors(_rt(), ids, vision_splice=object()) == ()
    assert generation._session_head_anchors(_rt(_Tokenizer(None)), ids) == ()
    short = _turn(SYSTEM, _filler(100, 1)) + _turn(USER, [21])
    assert generation._session_head_anchors(_rt(), short) == ()


# --- real templates --------------------------------------------------------

MODELS = Path.home() / ".mtplx/models"
REAL_TEMPLATES = [
    "Youssofal--Qwen3.6-35B-A3B-MTPLX-Optimized-Balance",
    "Youssofal--Qwen3.5-9B-MTPLX-Optimized-Speed",
    "Youssofal--Ternary-Bonsai-2-27B-MTPLX-Optimized-Speed",
]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} over the workspace. " * 20,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Target."}},
                "required": ["path"],
            },
        },
    }
    for name in ("read_file", "write_file", "list_dir")
]
SYSTEM_TEXT = "You are a careful assistant for a documentation team. " * 40


def _real_tokenizer(name: str):
    model_dir = MODELS / name
    if not (model_dir / "tokenizer.json").exists():
        pytest.skip(f"{name} not cached locally")
    from mtplx.runtime import _load_tokenizer_resilient

    return _load_tokenizer_resilient(model_dir, json.loads((model_dir / "config.json").read_text()))


def _encode(tok, messages, *, tools=None, thinking=True):
    import mtplx.server.openai as oa

    request = oa.ChatCompletionRequest(model="m", messages=messages)
    return oa._encode_messages(
        tok,
        request.messages,
        enable_thinking=thinking,
        tools=tools,
        template_observability={},
    )


def _assert_head_then_user(tok, ids) -> int:
    head = session_head_length(ids, turn_markers(tok))
    assert head is not None
    assert tok.decode(ids[:head]).startswith("<|im_start|>system")
    assert tok.decode(ids[:head]).endswith("<|im_end|>\n")
    assert tok.decode(ids[head : head + 3]).startswith("<|im_start|>user")
    return head


@pytest.mark.parametrize("name", REAL_TEMPLATES)
@pytest.mark.parametrize("thinking", [True, False])
@pytest.mark.parametrize(
    "system, tools",
    [(True, True), (True, False), (False, True)],
    ids=["system+tools", "system", "tools"],
)
def test_real_templates_anchor_the_first_user_turn(name, thinking, system, tools, monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tok = _real_tokenizer(name)
    lead = [{"role": "system", "content": SYSTEM_TEXT}] if system else []
    tool_specs = TOOLS if tools else None
    first = _encode(tok, [*lead, {"role": "user", "content": "Summarize the notes."}], tools=tool_specs, thinking=thinking)
    other = _encode(tok, [*lead, {"role": "user", "content": "Plan the week, please."}], tools=tool_specs, thinking=thinking)
    head = _assert_head_then_user(tok, first)
    assert _assert_head_then_user(tok, other) == head
    assert first[:head] == other[:head]
    later = _encode(
        tok,
        [
            *lead,
            {"role": "user", "content": "Summarize the notes."},
            {"role": "assistant", "content": "Here is a summary."},
            {"role": "user", "content": "Shorter, please."},
        ],
        tools=tool_specs,
        thinking=thinking,
    )
    assert _assert_head_then_user(tok, later) == head
    assert later[:head] == first[:head]


@pytest.mark.parametrize("name", REAL_TEMPLATES)
def test_real_template_without_system_input(name, monkeypatch):
    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tok = _real_tokenizer(name)
    ids = _encode(tok, [{"role": "user", "content": "Hello there."}])
    head = session_head_length(ids, turn_markers(tok))
    if tok.decode(ids[:3]).startswith("<|im_start|>system"):
        # The template's own default system turn (Bonsai) is a fixed head too.
        assert head == _assert_head_then_user(tok, ids)
    else:
        assert head is None


def test_messages_endpoint_anchors_the_first_user_turn(monkeypatch):
    """``/v1/messages`` as the Claude Agent SDK sends it: system string,
    Anthropic tools, a user turn with reminder blocks, later a tool round."""

    import mtplx.server.openai as oa

    monkeypatch.setenv("MTPLX_CHAT_ENCODE_CACHE", "off")
    tok = _real_tokenizer(REAL_TEMPLATES[0])
    tools = [
        {
            "name": spec["function"]["name"],
            "description": spec["function"]["description"],
            "input_schema": spec["function"]["parameters"],
        }
        for spec in TOOLS
    ]

    def prompt(messages):
        request = oa.AnthropicMessagesRequest(
            model="m", max_tokens=64, system=SYSTEM_TEXT, tools=tools, messages=messages
        )
        chat = oa._anthropic_to_chat_request(request)
        return _encode(tok, chat.messages, tools=chat.tools)

    def user(text):
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": "<system-reminder>Today is a weekday.</system-reminder>"},
                {"type": "text", "text": text},
            ],
        }

    first = prompt([user("Read the notes file.")])
    second = prompt([user("List the drafts folder.")])
    tool_round = prompt(
        [
            user("Read the notes file."),
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "notes.md"}}],
            },
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Notes."}]},
        ]
    )
    head = _assert_head_then_user(tok, first)
    assert _assert_head_then_user(tok, second) == head
    assert _assert_head_then_user(tok, tool_round) == head
    assert first[:head] == second[:head] == tool_round[:head]


def test_gemma_has_no_chatml_turns():
    tok_dir = MODELS / "Youssofal--Gemma4-MTPLX-Optimized-Speed" / "target"
    if not (tok_dir / "tokenizer.json").exists():
        pytest.skip("Gemma 4 pack not cached locally")
    from transformers import AutoTokenizer

    assert turn_markers(AutoTokenizer.from_pretrained(str(tok_dir))) is None


# ---------------------------------------------------------------------------
# Planning, thinning, inheritance
# ---------------------------------------------------------------------------


def _rec(position: int):
    return (position, f"snap-{position}", None)


def test_mandatory_edges_merge_stable_prefix_and_anchors():
    edge = generation._mandatory_prefill_edges
    assert edge(None, (), limit=100) == ()
    assert edge(40, (), limit=100) == (40,)
    assert edge(40, (60,), limit=100) == (40, 60)
    assert edge(40, (60,), limit=100, offset=50) == (10,)
    assert edge(None, (0, 100, 150), limit=100) == ()


def test_the_anchor_is_recorded_inside_the_last_wide_forward():
    spans, interior = generation._prefill_boundary_plan(
        15_816,
        capture_boundaries=True,
        inforward=True,
        tail_interval=256,
        mandatory_edges=(13_024,),
        chunk_size=2048,
    )
    assert spans == generation._iter_prefill_chunk_spans(15_816, chunk_size=2048)
    assert 13_024 in interior


def test_thinning_keeps_the_anchor_within_the_cap():
    records = [_rec(position) for position in range(2048, 60_000, 256)]
    plain = generation._thin_gdn_boundary_records(records, 8)
    kept = generation._thin_gdn_boundary_records(records, 8, keep=(13_056,))
    assert 13_056 not in [record[0] for record in plain]
    assert 13_056 in [record[0] for record in kept]
    assert len(kept) <= 8
    assert kept[0][0] == 2048 and kept[-1][0] == records[-1][0]
    assert generation._thin_gdn_boundary_records(records, 8, keep=()) == plain


def test_the_sink_never_thins_its_anchor(monkeypatch):
    monkeypatch.delenv("MTPLX_GDN_BOUNDARY_MAX", raising=False)
    anchored = generation.GdnBoundarySink(anchors=(13_056,))
    plain: list = []
    for position in range(2048, 60_000, 256):
        generation._append_gdn_boundary_record(anchored, position, f"s{position}", None)
        generation._append_gdn_boundary_record(plain, position, f"s{position}", None)
    assert 13_056 in [record[0] for record in anchored]
    assert 13_056 not in [record[0] for record in plain]
    assert len(anchored) <= 8 and isinstance(anchored, generation.GdnBoundarySink)


def test_inheritance_keeps_the_new_requests_anchor():
    entry = SimpleNamespace(gdn_boundaries=[_rec(p) for p in range(2048, 60_000, 256)])
    kept = generation._inherited_gdn_boundaries(entry, 50_000, (13_056,))
    assert 13_056 in [record[0] for record in kept]
    assert 13_056 not in [r[0] for r in generation._inherited_gdn_boundaries(entry, 50_000)]


# ---------------------------------------------------------------------------
# Bank simulation: a long chat, then a new session with the same head
# ---------------------------------------------------------------------------

HEAD = 13_024


class _BankRuntime:
    model_path = Path("/tmp/fake-model")
    mtp_enabled = True
    tokenizer = _Tokenizer(_AddedToken("<|im_start|>"))


class _Recurrent:
    def __init__(self):
        self.state = [mx.ones((2, 2)), None]
        self.meta_state = ("owned_recurrent_state", "persistent_eval")

    def is_trimmable(self):
        return False

    def replace_state(self, value):
        self.state = list(value)


def _snapshot(position: int) -> CacheSnapshot:
    return CacheSnapshot(states=(mx.full((2, 2), float(position)),), meta_states=(None,))


def _session(first_user_seed: int, length: int) -> list[int]:
    """Same head for every session (system turn to HEAD), then its own chat."""

    head = [IM, SYSTEM, *_filler(HEAD - 3, seed=1), END]
    assert len(head) == HEAD
    return head + [IM, USER, *_filler(length - HEAD - 2, seed=first_user_seed)]


def _simulated_prefill(sink, start: int, prompt_len: int, anchors) -> None:
    """The boundary records the real warm/cold loops append, in their order."""

    body = prompt_len - 1 - start
    spans, interior = generation._prefill_boundary_plan(
        body,
        capture_boundaries=True,
        inforward=True,
        tail_interval=generation._gdn_boundary_tail_interval(),
        mandatory_edges=generation._mandatory_prefill_edges(
            None, anchors, limit=body, offset=start
        ),
    )
    for span_start, span_end in spans:
        for row in sorted(p for p in interior if span_start < p < span_end):
            generation._append_gdn_boundary_record(sink, start + row, _snapshot(start + row), None)
        generation._append_gdn_boundary_record(sink, start + span_end, _snapshot(start + span_end), None)


def _put(bank, tokens, boundaries=None):
    return bank.put(
        runtime=_BankRuntime(),
        token_ids=tokens,
        cache=[_Recurrent()],
        logits=mx.zeros((1, 4)),
        hidden=None,
        session_id="chat",
        template_hash="t",
        policy_fingerprint="p",
        snapshot_epoch=len(tokens),
        gdn_boundaries=boundaries,
    )


def _run_chat(bank, conversation: list[int], turn_ends: list[int]) -> None:
    """Each turn: exact restore from the bank, prefill the suffix, bank the
    prompt entry (with the sink) and the generation-final entry (inherits)."""

    rt = _BankRuntime()
    for prompt_len, final_len in zip(turn_ends[::2], turn_ends[1::2]):
        prompt = conversation[:prompt_len]
        anchors = generation._session_head_anchors(rt, prompt)
        donor = bank.longest_prefix(prompt)
        start = donor.prefix_len if donor is not None else 0
        inherited = (
            generation._inherited_gdn_boundaries(donor, start, anchors) if donor else []
        )
        sink = generation.GdnBoundarySink(inherited, anchors=anchors)
        _simulated_prefill(sink, start, prompt_len, anchors)
        _put(bank, prompt, list(sink))
        _put(bank, conversation[:final_len])


def _turn_ends(first_prompt: int, total: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    ends: list[int] = []
    position = first_prompt
    while position < total:
        final = position + rng.randrange(150, 1500)
        ends += [position, final]
        position = final + rng.randrange(300, 4000)
    return ends


def _new_session_restore_point(bank) -> int:
    prompt = _session(first_user_seed=99, length=15_800)
    candidates = bank.near_prefix_candidates(
        prompt,
        block_size=256,
        block_min_matched_tokens=512,
        allow_block_prefix=True,
        model_path=str(_BankRuntime.model_path),
        mtp_enabled=True,
        template_hash="t",
        policy_fingerprint="p",
    )
    assert candidates
    best = 0
    for entry, matched in candidates:
        boundary = entry.recurrent_boundary_at_or_below(matched)
        if boundary is not None:
            best = max(best, int(boundary[0]))
    return best


@pytest.fixture
def bank_env(monkeypatch):
    for name, value in (
        ("MTPLX_SUSTAINED_PREFILL", "1"),
        ("MTPLX_PREFILL_CHUNK_SIZE", "2048"),
        ("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1"),
    ):
        monkeypatch.setenv(name, value)
    for name in (
        "MTPLX_GDN_BOUNDARY_MAX",
        "MTPLX_GDN_BOUNDARY_TAIL_INTERVAL",
        "MTPLX_GDN_BOUNDARY_TAIL_MIN_RUNG",
        "MTPLX_GDN_BOUNDARY_TAIL_BACKOFF",
        "MTPLX_GDN_BOUNDARY_TAIL_LAYOUT",
        "MTPLX_SESSION_BLOCK_PREFIX_MIN_MATCH_TOKENS",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _chat_bank(cold_tier=None) -> tuple[SessionBank, list[int]]:
    conversation = _session(first_user_seed=2, length=90_000)
    bank = SessionBank(
        max_entries=8, max_bytes=1 << 34, per_session_max_bytes=1 << 34, cold_tier=cold_tier
    )
    _run_chat(bank, conversation, _turn_ends(15_817, 60_000, seed=3))
    return bank, conversation


@pytest.mark.parametrize("anchor", [False, True], ids=["off", "on"])
def test_a_new_session_restores_at_the_anchor_after_a_long_chat(bank_env, anchor):
    bank_env.setenv(SWITCH, "1" if anchor else "0")
    bank, _conversation = _chat_bank()
    newest = max(bank._entries.values(), key=lambda entry: entry.prefix_len)
    assert newest.prefix_len > 55_000
    positions = [record[0] for record in newest.gdn_boundaries]
    assert len(positions) <= 8
    assert (HEAD in positions) is anchor
    # Without the anchor only the oldest boundary is left under the head:
    # the 2,048 the request log shows for most session starts.
    assert _new_session_restore_point(bank) == (HEAD if anchor else 2048)


def test_the_anchor_survives_the_ssd_tier(bank_env, tmp_path):
    bank_env.setenv(SWITCH, "1")
    cold = SessionBankColdTier(base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=2)
    try:
        bank, _conversation = _chat_bank(cold_tier=cold)
        assert cold.flush(timeout_s=30.0) is True
        bank.clear()
        assert not bank._entries
        assert _new_session_restore_point(bank) == HEAD
    finally:
        cold.close()


# ---------------------------------------------------------------------------
# Tiny A3B model on the invariant lane: output and restore, bit for bit
# ---------------------------------------------------------------------------

gib_tests = pytest.importorskip("tests.test_gdn_inforward_boundaries")
lane = gib_tests.lane
needs_metal = gib_tests.needs_metal


def _model_rt(model):
    from mtplx.mtp_patch import MTPContract

    rt = gib_tests._Runtime(model)
    rt.tokenizer = _Tokenizer(_AddedToken("<|im_start|>"), im=250, system=251)
    rt.contract = MTPContract()
    return rt


def _chat_prompt(user_seed: int, user_len: int) -> list[int]:
    """Tokens 250/251/252 play ``<|im_start|>``/system/user: the head is 60."""

    def text(count: int, seed: int) -> list[int]:
        return [min(token, 249) for token in gib_tests._prompt(count, seed=seed)]

    return [250, 251, *text(58, 1), 250, 252, *text(user_len, user_seed)]


def _prefill(rt, prompt, bank, session_id="s"):
    state = generation.restore_or_prefill_prompt_state(
        rt,
        prompt,
        mtp_history_policy="committed",
        base_hidden_variant="post_norm",
        mtp_hidden_variant="post_norm",
        session_bank=bank,
        session_id=session_id,
    )
    mx.eval(state.logits, state.hidden)
    return state


def _bank_state(bank, rt, prompt, state):
    bank.put(
        runtime=rt,
        token_ids=prompt,
        cache=state.trunk_cache,
        logits=state.logits,
        hidden=state.hidden,
        hidden_variant="post_norm",
        session_id="first",
        mtp_history_policy="committed",
        mtp_history_snapshot=snapshot_cache(state.committed_mtp_cache),
        snapshot_epoch=len(prompt),
        mtp_snapshot_epoch=len(prompt),
        gdn_boundaries=state.gdn_boundaries,
    )


def _logits_equal(left, right) -> bool:
    return np.array_equal(gib_tests._np(left.logits), gib_tests._np(right.logits)) and np.array_equal(
        gib_tests._np(left.hidden), gib_tests._np(right.hidden)
    )


@pytest.fixture
def tiny(lane, monkeypatch):
    monkeypatch.setenv("MTPLX_SESSION_BLOCK_PREFIX_RESTORE", "1")
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL", "0")
    # 16-token blocks, so the lane's 16-token restore minimum applies as
    # given (the block minimum is at least one block).
    monkeypatch.setenv("MTPLX_SESSION_PREFIX_BLOCK_SIZE", "16")
    model = gib_tests._tiny_model()
    gib_tests._install(model)
    return model


@needs_metal
def test_the_anchor_does_not_change_the_prefill(tiny, monkeypatch):
    prompt = _chat_prompt(3, 100)
    monkeypatch.setenv(SWITCH, "0")
    off = _prefill(_model_rt(tiny), prompt, SessionBank(max_entries=4))
    monkeypatch.setenv(SWITCH, "1")
    on = _prefill(_model_rt(tiny), prompt, SessionBank(max_entries=4))
    assert 60 not in [record[0] for record in off.gdn_boundaries]
    assert 60 in [record[0] for record in on.gdn_boundaries]
    assert _logits_equal(on, off)
    gib_tests._assert_bit_equal(
        gib_tests._cache_leaves(on.trunk_cache), gib_tests._cache_leaves(off.trunk_cache)
    )


@needs_metal
@pytest.mark.parametrize("anchor", [False, True], ids=["off", "on"])
def test_a_new_session_restores_at_the_anchor_bit_for_bit(tiny, monkeypatch, anchor):
    monkeypatch.setenv(SWITCH, "1" if anchor else "0")
    rt = _model_rt(tiny)
    bank = SessionBank(max_entries=4, max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    first = _chat_prompt(3, 100)
    _bank_state(bank, rt, first, _prefill(rt, first, bank))
    second = _chat_prompt(5, 90)
    warm = _prefill(rt, second, bank, session_id="second")
    cold = _prefill(_model_rt(tiny), second, None)
    assert warm.cached_tokens == (60 if anchor else 0)
    assert _logits_equal(warm, cold)


@needs_metal
@pytest.mark.parametrize("anchor", [False, True], ids=["off", "on"])
def test_a_new_session_restores_at_the_anchor_from_the_ssd(tiny, monkeypatch, tmp_path, anchor):
    """16-token SSD blocks: the shared 62 tokens align down to 48, below the
    anchor at 60; with the switch the SSD lane restores to the exact match."""

    monkeypatch.setenv(SWITCH, "1" if anchor else "0")
    rt = _model_rt(tiny)
    cold_tier = SessionBankColdTier(
        base_dir=tmp_path / "bank", mode="on", min_prefix_tokens=2, block_size=16
    )
    try:
        bank = SessionBank(
            max_entries=4, max_bytes=1 << 32, per_session_max_bytes=1 << 32, cold_tier=cold_tier
        )
        first = _chat_prompt(3, 100)
        _bank_state(bank, rt, first, _prefill(rt, first, bank))
        assert cold_tier.flush(timeout_s=30.0) is True
        bank.clear()
        second = _chat_prompt(5, 90)
        warm = _prefill(rt, second, bank, session_id="second")
        assert warm.cached_tokens == (60 if anchor else 0)
        assert warm.ssd_cache_hit is anchor
        assert _logits_equal(warm, _prefill(_model_rt(tiny), second, None))
    finally:
        cold_tier.close()


@needs_metal
def test_an_anchor_inside_a_warm_suffix_does_not_change_the_result(tiny):
    """Restored below the anchor (an entry from before the switch): the warm
    suffix records the anchor inside its forward, with the same result."""

    prompt = _chat_prompt(3, 150)

    def warm(sink):
        rt = _model_rt(tiny)
        cache = rt.make_cache()
        mx.eval(rt.forward_ar(mx.array([prompt[:30]]), cache=cache))
        restored = SimpleNamespace(
            cache=cache, mtp_history_cache=[], hidden=None, entry=SimpleNamespace(prefix_len=30)
        )
        logits, hidden, _forward_s, _history_s = generation._prefill_restored_prompt_suffix(
            rt,
            restored,
            prompt[30:],
            base_hidden_variant="post_norm",
            mtp_hidden_variant="post_norm",
            mtp_history_policy="committed",
            cached_tokens=30,
            gdn_boundary_sink=sink,
        )
        mx.eval(logits, hidden)
        return rt, sink, cache, logits, hidden

    plain_rt, plain, plain_cache, plain_logits, plain_hidden = warm([])
    rt, anchored, cache, logits, hidden = warm(generation.GdnBoundarySink(anchors=(60,)))
    assert 60 not in [record[0] for record in plain]
    assert 60 in [record[0] for record in anchored]
    assert rt.forwards == plain_rt.forwards
    gib_tests._assert_bit_equal(
        gib_tests._leaves([logits, hidden]), gib_tests._leaves([plain_logits, plain_hidden])
    )
    gib_tests._assert_bit_equal(gib_tests._cache_leaves(cache), gib_tests._cache_leaves(plain_cache))
