"""Grammar-constrained decoding (response_format), issue #186 phase 1.

Covers: the request-validation surface (bad shapes 400 instead of silent
non-enforcement), the generate_ar wiring (mask before sampling, advance per
token, grammar-terminal early stop, stats counters), end-to-end schema
enforcement against adversarial logits with a tiny single-byte tokenizer,
and public-envelope exposure of the constraint counters.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mtplx.constrained import (
    ResponseFormatError,
    constraint_spec_from_response_format,
)
from mtplx.generation import generate_ar
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.sampling import SamplerConfig


# --- response_format validation surface (no llguidance required) ----------


def test_absent_and_text_response_formats_apply_no_constraint():
    assert constraint_spec_from_response_format(None) is None
    assert constraint_spec_from_response_format({"type": "text"}) is None


@pytest.mark.parametrize(
    "response_format",
    [
        "json",
        ["json_object"],
        {},
        {"type": "json"},
        {"type": "grammar"},
    ],
)
def test_invalid_response_formats_are_rejected(response_format):
    with pytest.raises(ResponseFormatError):
        constraint_spec_from_response_format(response_format)


# --- generate_ar wiring (scripted model, fake constraint) ------------------


class _Tokenizer:
    def decode(self, tokens, **_kwargs):
        return "".join(f"<{int(token)}>" for token in tokens)


class _RampModel:
    """Unconstrained argmax always walks t -> t+1 over an 8-token vocab."""

    vocab = 8

    def __init__(self):
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        return []

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        tokens = [int(token) for token in np.asarray(input_ids).reshape(-1)]
        row = [0.0] * self.vocab
        row[(tokens[-1] + 1) % self.vocab] = 10.0
        logits = mx.array([[row]], dtype=mx.float32)
        hidden = mx.zeros((1, len(tokens), 2), dtype=mx.float32)
        if return_hidden:
            return logits, hidden
        return logits


class _ForcingConstraint:
    """Duck-typed constraint that forces a scripted token path, then stops."""

    def __init__(self, forced: list[int]):
        self.forced = list(forced)
        self.advanced: list[int] = []
        self.masked_steps = 0
        self.mask_time_s = 0.0

    def mask_logits_row(self, row):
        self.masked_steps += 1
        wanted = self.forced[len(self.advanced)]
        mask = mx.full(row.shape, -np.inf, dtype=row.dtype)
        return mx.where(
            mx.arange(row.shape[-1]) == wanted, mx.array(100.0, dtype=row.dtype), mask
        )

    def advance(self, token_id: int) -> None:
        self.advanced.append(int(token_id))

    @property
    def stopped(self) -> bool:
        return len(self.advanced) >= len(self.forced)

    @property
    def completed(self) -> bool:
        return self.stopped


def _runtime(model, backend_id: str | None = None) -> MTPLXRuntime:
    rt = MTPLXRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        model_path=Path("tiny-constrained"),
        mtp_enabled=False,
        contract=MTPContract(),
    )
    if backend_id is not None:
        rt.backend_id = backend_id
    return rt


def test_generate_ar_constraint_overrides_model_preference():
    constraint = _ForcingConstraint([5, 2, 7])
    out = generate_ar(
        _runtime(_RampModel()),
        [1, 2],
        max_tokens=10,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        seed=0,
        stop_token_ids=set(),
        constraint=constraint,
    )
    # The ramp model wants 3,4,5...; the mask forces the scripted path, and
    # generation halts at the grammar terminal instead of running to
    # max_tokens.
    assert out.tokens == [5, 2, 7]
    assert constraint.advanced == [5, 2, 7]
    assert constraint.masked_steps == 3
    assert out.finish_reason == "stop"
    assert out.stats.constraint_active is True
    assert out.stats.constraint_completed is True
    assert out.stats.constraint_masked_steps == 3
    assert any("constraint_stop" in event for event in out.stats.events)


def test_generate_ar_without_constraint_reports_inactive():
    out = generate_ar(
        _runtime(_RampModel()),
        [1],
        max_tokens=3,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        seed=0,
        stop_token_ids=set(),
    )
    assert out.stats.constraint_active is False
    assert out.stats.constraint_completed is None


def test_generate_ar_rejects_constraint_on_gemma4_assistant_backend():
    with pytest.raises(ValueError, match="gemma4_assistant"):
        generate_ar(
            _runtime(_RampModel(), backend_id="gemma4_assistant"),
            [1],
            max_tokens=3,
            sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
            seed=0,
            stop_token_ids=set(),
            constraint=_ForcingConstraint([1]),
        )


# --- generate_mtpk composition (#186 phase 3): scripted model, fake grammar -


class _EvensOnlyConstraint:
    """Duck-typed grammar allowing only even token ids, stopping after `limit`.

    The scripted ramp model always prefers odd successors, so every even
    committed token is the mask's or the clamp's doing.
    """

    def __init__(self, limit: int = 6):
        self.limit = limit
        self.advanced: list[int] = []
        self.masked_steps = 0
        self.mask_time_s = 0.0

    def _legal(self, token_id: int) -> bool:
        return token_id % 2 == 0 and len(self.advanced) < self.limit

    def mask_logits_row(self, row):
        self.masked_steps += 1
        ids = mx.arange(row.shape[-1])
        legal = (ids % 2) == 0
        return mx.where(legal, row, mx.array(-np.inf, dtype=row.dtype))

    def validate_prefix(self, token_ids):
        count = 0
        pos = len(self.advanced)
        for token in token_ids:
            if pos + count >= self.limit or int(token) % 2 != 0:
                break
            count += 1
        return count

    def advance(self, token_id: int) -> None:
        self.advanced.append(int(token_id))

    def advance_many(self, token_ids) -> None:
        for token in token_ids:
            self.advance(token)

    @property
    def stopped(self) -> bool:
        return len(self.advanced) >= self.limit

    @property
    def completed(self) -> bool:
        return self.stopped


class _MTPScriptedModel:
    """Deterministic mtpk stub: after token t, both trunk and MTP head want
    t+1 (mod vocab) — always odd successors from even tokens and vice versa."""

    def __init__(self, vocab: int = 8):
        self.vocab = vocab
        self.mtp = SimpleNamespace(_mtplx_lora_targets=[])

    def make_cache(self):
        return []

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def _logits_for(self, last_tokens):
        rows = []
        for token in last_tokens:
            row = [0.0] * self.vocab
            row[(int(token) + 1) % self.vocab] = 10.0
            rows.append(row)
        return mx.array([rows], dtype=mx.float32)

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
    ):
        toks = [int(t) for t in np.asarray(input_ids).reshape(-1)]
        keep = len(toks) if logits_keep is None else min(len(toks), max(1, int(logits_keep)))
        logits = self._logits_for(toks[-keep:]) if emit_logits else None
        hidden = mx.zeros((1, len(toks), 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        return (logits, hidden) if return_hidden else logits

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        *,
        mtp_cache=None,
        concat_order=None,
        return_hidden: bool = False,
        mtp_hidden_variant: str | None = None,
        position_offset=None,
    ):
        toks = [int(t) for t in np.asarray(next_token_ids).reshape(-1)]
        logits = self._logits_for(toks)
        hidden = mx.zeros((1, len(toks), 2), dtype=mx.float32)
        return (logits, hidden) if return_hidden else logits


def _mtpk_constrained(constraint, *, max_tokens: int = 12, depth: int = 2):
    from mtplx.generation import generate_mtpk

    rt = MTPLXRuntime(
        model=_MTPScriptedModel(),
        tokenizer=_Tokenizer(),
        model_path=Path("tiny-constrained-mtpk"),
        mtp_enabled=True,
        contract=MTPContract(),
    )
    return generate_mtpk(
        rt,
        [0, 1, 2, 3],
        max_tokens=max_tokens,
        sampler=SamplerConfig(temperature=0.0, top_p=1.0, top_k=0),
        speculative_depth=depth,
        seed=0,
        stop_token_ids=set(),
        verify_strategy="capture_commit",
        constraint=constraint,
    )


def test_generate_mtpk_masks_and_clamps_to_grammar(monkeypatch):
    monkeypatch.delenv("MTPLX_CONTEXT_COPY", raising=False)
    constraint = _EvensOnlyConstraint(limit=6)
    out = _mtpk_constrained(constraint)
    # The ramp model always wants odd successors; only the mask (primary) and
    # the legality clamp (draft window / bonus) can keep the stream even.
    assert out.tokens, "no tokens generated"
    assert all(t % 2 == 0 for t in out.tokens), out.tokens
    # The matcher advanced through exactly the committed stream, in order.
    assert constraint.advanced == out.tokens
    assert out.stats.constraint_active is True
    assert out.stats.constraint_completed is True
    assert out.stats.constraint_masked_steps >= 1


def test_generate_mtpk_stops_at_grammar_terminal(monkeypatch):
    monkeypatch.delenv("MTPLX_CONTEXT_COPY", raising=False)
    constraint = _EvensOnlyConstraint(limit=3)
    out = _mtpk_constrained(constraint, max_tokens=20)
    assert len(out.tokens) == 3, out.tokens
    assert out.finish_reason == "stop"
    assert out.stats.constraint_completed is True


def test_generate_mtpk_unconstrained_reports_inactive(monkeypatch):
    monkeypatch.delenv("MTPLX_CONTEXT_COPY", raising=False)
    out = _mtpk_constrained(None, max_tokens=6)
    assert out.stats.constraint_active is False
    assert out.stats.constraint_completed is None


# --- strict tool-call constraint spec (phase 2) -----------------------------


def test_tool_call_spec_paths_without_llguidance_dependency():
    from mtplx.constrained import tool_call_constraint_spec

    assert tool_call_constraint_spec(None, None, object()) is None
    assert tool_call_constraint_spec([], "auto", object()) is None
    assert (
        tool_call_constraint_spec(
            [{"type": "function", "function": {"name": "f"}}], "none", object()
        )
        is None
    )


# --- end-to-end with llguidance (tiny single-byte tokenizer) ---------------

llguidance = pytest.importorskip("llguidance")


def _tiny_hf_tokenizer():
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    # No space token: raw space is not its own symbol in the byte-level
    # alphabet (it's 'Ġ'), and JSON for the test schema needs no whitespace.
    vocab = {chr(i): i - 32 for i in range(33, 127)}
    vocab["<eos>"] = 0
    # Merge-free BPE tokenizes any byte string char-by-char, which llguidance
    # needs to canonically tokenize forced-byte runs like '"age"' (WordLevel
    # would return UNK for multi-char lookups and break the mask).
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<eos>"))
    # llguidance derives token byte representations from the decoder type;
    # ByteLevel is one it recognizes, and every remaining printable-ASCII
    # token maps to itself under the byte-level alphabet.
    backend.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>"), {
        v: k for k, v in vocab.items()
    }


_SCHEMA = {
    "type": "object",
    "properties": {
        "age": {"type": "integer", "minimum": 0},
        "tag": {"type": "string", "enum": ["a", "b"]},
    },
    "required": ["age", "tag"],
    "additionalProperties": False,
}


def test_grammar_forces_schema_valid_json_under_adversarial_logits():
    hf_tok, id_to_text = _tiny_hf_tokenizer()
    spec = constraint_spec_from_response_format(
        {"type": "json_schema", "json_schema": {"name": "t", "schema": _SCHEMA}}
    )
    assert spec is not None
    constraint = spec.build(hf_tok)

    # Logits width deliberately exceeds the tokenizer vocab (padded lm_head).
    # The unmasked argmax is always a padding token, and among legal tokens
    # the driver prefers the lowest id — never what the schema wants next —
    # so any schema-valid output is purely the mask's doing.
    n_vocab = 128
    tokens: list[int] = []
    for _ in range(200):
        if constraint.stopped:
            break
        row = -mx.arange(n_vocab, dtype=mx.float32)
        masked = constraint.mask_logits_row(row)
        arr = np.array(masked)
        assert np.all(arr[95:] < -1e30), "mask leaked padding/out-of-vocab tokens"
        token = int(mx.argmax(masked).item())
        constraint.advance(token)
        tokens.append(token)

    text = "".join(id_to_text[t] for t in tokens if id_to_text[t] != "<eos>")
    assert constraint.completed, f"grammar never completed: {text!r}"
    parsed = json.loads(text)
    assert set(parsed) == {"age", "tag"}
    assert isinstance(parsed["age"], int) and parsed["age"] >= 0
    assert parsed["tag"] in {"a", "b"}
    assert constraint.masked_steps == len(tokens)


def test_json_object_and_lenient_schema_shapes_accepted():
    assert (
        constraint_spec_from_response_format({"type": "json_object"}).source_type
        == "json_object"
    )
    lenient = constraint_spec_from_response_format(
        {"type": "json_schema", "schema": {"type": "object"}}
    )
    assert lenient is not None and lenient.source_type == "json_schema"
    with pytest.raises(ResponseFormatError, match="json_schema.schema"):
        constraint_spec_from_response_format({"type": "json_schema"})


def _tiny_tool_tokenizer():
    """Tiny char tokenizer plus the Qwen-family special markers, so the
    strict tool-call grammar is exercisable in CI without model weights."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {chr(i): i - 32 for i in range(33, 127)}
    vocab["<eos>"] = 0
    # Space and newline via their byte-level alphabet symbols (raw space is
    # not its own symbol there); the forced envelope head needs both.
    vocab["Ġ"] = 95
    vocab["Ċ"] = 96
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<eos>"))
    # The ByteLevel pre-tokenizer makes encode() map raw bytes to the
    # byte-level symbols — llguidance canonically tokenizes forced-byte runs
    # through the tokenizer, so "\n" must encode to Ċ, not UNK.
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>")
    hf_tok.add_special_tokens(
        {
            "additional_special_tokens": [
                "<tool_call>",
                "</tool_call>",
                "<think>",
                "</think>",
            ]
        }
    )
    return hf_tok


def test_strict_tool_call_grammar_end_to_end():
    from mtplx.constrained import tool_call_constraint_spec

    hf_tok = _tiny_tool_tokenizer()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "pick",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer", "minimum": 0, "maximum": 9}
                    },
                    "required": ["n"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    spec = tool_call_constraint_spec(tools, "auto", hf_tok)
    assert spec is not None and spec.source_type == "tool_call_strict"

    n_vocab = 128
    constraint = spec.build(hf_tok)
    constraint.mask_logits_row(mx.zeros((n_vocab,)))  # bind

    def ids(text):
        return hf_tok.encode(text, add_special_tokens=False)

    # Free text is unconstrained; a lone </think> is legal (the chat template
    # opens the think block inside the generation prompt).
    assert constraint.validate_prefix(ids("hello.")) == len(ids("hello."))
    prelude = ids("reasoning...") + ids("</think>") + ids("ok.")
    assert constraint.validate_prefix(prelude) == len(prelude)

    # Inside the envelope everything is forced: walk the adversarial argmax
    # (unmasked argmax is always a padding token; among legal tokens the
    # lowest id wins, never what the schema wants).
    constraint.advance_many(prelude)
    trigger = ids("<tool_call>")
    assert len(trigger) == 1
    constraint.advance_many(trigger)
    out = []
    for _ in range(80):
        if constraint.stopped or constraint.completed and out and out[-1] == trigger[0]:
            break
        row = -mx.arange(n_vocab, dtype=mx.float32)
        masked = constraint.mask_logits_row(row)
        token = int(mx.argmax(masked).item())
        constraint.advance(token)
        out.append(token)
        if token == ids("</tool_call>")[0]:
            break
    text = hf_tok.decode(out).replace("</tool_call>", "")
    payload = json.loads(text)
    assert payload["name"] == "pick"
    assert isinstance(payload["arguments"]["n"], int)
    assert 0 <= payload["arguments"]["n"] <= 9
    # Back in free text after the envelope closes.
    assert constraint.completed
    assert constraint.validate_prefix(ids("done.")) == len(ids("done."))

    # A tool the request never declared is unreachable.
    fresh = spec.build(hf_tok)
    fresh.mask_logits_row(mx.zeros((n_vocab,)))
    fresh.advance_many(trigger)
    bad = ids('\n{"name": "rm_rf"')
    assert fresh.validate_prefix(bad) < len(bad)


def test_json_prelude_gated_on_open_think_block():
    hf_tok = _tiny_tool_tokenizer()
    spec = constraint_spec_from_response_format(
        {"type": "json_object"}, tokenizer=hf_tok
    )
    assert spec.grammar_with_prelude is not None
    think_open = hf_tok.encode("<think>", add_special_tokens=False)[0]
    think_close = hf_tok.encode("</think>", add_special_tokens=False)[0]
    n_vocab = 160
    prose = hf_tok.encode("hello", add_special_tokens=False)

    # Prompt ends inside an open think block -> prelude grammar: reasoning
    # text is legal before the document.
    inside = spec.build(hf_tok, prompt_ids=[5, think_open])
    inside.mask_logits_row(mx.zeros((n_vocab,)))
    assert inside.validate_prefix(prose) == len(prose)

    # Think block already closed -> plain grammar: prose is illegal, the
    # document must start immediately (the prelude would otherwise allow
    # unbounded free text on non-thinking runs).
    closed = spec.build(hf_tok, prompt_ids=[5, think_open, 6, think_close])
    closed.mask_logits_row(mx.zeros((n_vocab,)))
    assert closed.validate_prefix(prose) == 0
    brace = hf_tok.encode("{", add_special_tokens=False)
    assert closed.validate_prefix(brace) == 1

    # No prompt information -> conservative plain grammar.
    unknown = spec.build(hf_tok)
    unknown.mask_logits_row(mx.zeros((n_vocab,)))
    assert unknown.validate_prefix(prose) == 0


def test_grammar_cache_canonicalizes_key_order():
    from mtplx import constrained as mod

    a = {"type": "object", "properties": {"x": {"type": "integer"}}}
    b = {"properties": {"x": {"type": "integer"}}, "type": "object"}
    before = len(mod._GRAMMAR_CACHE)
    spec_a = constraint_spec_from_response_format(
        {"type": "json_schema", "json_schema": {"schema": a}}
    )
    grown = len(mod._GRAMMAR_CACHE)
    spec_b = constraint_spec_from_response_format(
        {"type": "json_schema", "json_schema": {"schema": b}}
    )
    assert spec_a.grammar == spec_b.grammar
    assert len(mod._GRAMMAR_CACHE) == grown >= before


# --- public envelope --------------------------------------------------------


def test_public_mtplx_stats_expose_constraint_counters():
    from mtplx.server.openai import PUBLIC_MTPLX_STATS_KEYS, _public_mtplx_stats

    keys = {
        "constraint_active",
        "constraint_completed",
        "constraint_masked_steps",
        "constraint_mask_time_s",
    }
    assert keys <= set(PUBLIC_MTPLX_STATS_KEYS)
    generated = {"stats": {key: 1 for key in keys}}
    public = _public_mtplx_stats(generated)
    assert keys <= set(public)


def test_masked_row_through_real_sparse_topk_sampler():
    # Grammar masks must survive the PRODUCT sampler path (temp 0.6,
    # top_p 0.95, top_k 20 — the sparse top-k lane), not only argmax:
    # -inf entries may never be sampled, and every legal token must stay
    # reachable once the mask removes the illegal mass, because top-k
    # selection runs on the MASKED row (mask-then-shape).
    import mlx.core as mx
    import numpy as np

    from mtplx.generation import _sample_from_logits
    from mtplx.sampling import SamplerConfig

    vocab = 512
    legal = {3: 2.0, 17: 1.8, 400: 1.6, 401: 1.4}
    row = np.full(vocab, -np.inf, dtype=np.float32)
    for token, logit in legal.items():
        row[token] = logit

    sampled = SamplerConfig(temperature=0.6, top_p=0.95, top_k=20)
    rng = np.random.default_rng(7)
    draws = {
        _sample_from_logits(mx.array(row), sampled, rng)[0] for _ in range(400)
    }
    assert draws <= set(legal), draws
    assert draws == set(legal), draws  # comparable masses: all four reachable

    greedy = SamplerConfig(temperature=0.0, top_p=1.0, top_k=0)
    token, _ = _sample_from_logits(mx.array(row), greedy, rng)
    assert token == 3


# --- bounded think prelude (the unbounded prelude is a legal runaway) -------


def _schema_json():
    return json.dumps(
        {
            "type": "object",
            "properties": {"a": {"type": "number"}},
            "required": ["a"],
            "additionalProperties": False,
        }
    )


def _prelude_line(grammar):
    return next(
        line for line in grammar.splitlines() if line.startswith("PRELUDE_TEXT")
    )


def test_think_prelude_is_bounded_by_default(monkeypatch):
    from mtplx.constrained import _cached_grammar_for_schema

    monkeypatch.delenv("MTPLX_THINK_PRELUDE_MAX_CHARS", raising=False)
    grammar = _cached_grammar_for_schema(_schema_json(), think_prelude=True)
    assert _prelude_line(grammar) == r"PRELUDE_TEXT: /(.|\n){0,4000}/"


def test_think_prelude_bound_is_configurable_and_disableable(monkeypatch):
    from mtplx.constrained import _cached_grammar_for_schema

    monkeypatch.setenv("MTPLX_THINK_PRELUDE_MAX_CHARS", "600")
    assert (
        _prelude_line(_cached_grammar_for_schema(_schema_json(), think_prelude=True))
        == r"PRELUDE_TEXT: /(.|\n){0,600}/"
    )

    # 0 restores the previous unbounded behaviour verbatim.
    monkeypatch.setenv("MTPLX_THINK_PRELUDE_MAX_CHARS", "0")
    assert (
        _prelude_line(_cached_grammar_for_schema(_schema_json(), think_prelude=True))
        == r"PRELUDE_TEXT: /(.|\n)*/"
    )


def test_prelude_bound_participates_in_the_grammar_cache_key(monkeypatch):
    """Without this the second bound would silently reuse the first grammar."""
    from mtplx.constrained import _cached_grammar_for_schema

    monkeypatch.setenv("MTPLX_THINK_PRELUDE_MAX_CHARS", "600")
    first = _cached_grammar_for_schema(_schema_json(), think_prelude=True)
    monkeypatch.setenv("MTPLX_THINK_PRELUDE_MAX_CHARS", "1200")
    second = _cached_grammar_for_schema(_schema_json(), think_prelude=True)
    assert first != second


def test_tool_call_prelude_bounded_but_tail_stays_free(monkeypatch):
    """The cap must not leak into the assistant's visible answer."""
    from mtplx.constrained import _tool_call_lark_grammar

    monkeypatch.delenv("MTPLX_THINK_PRELUDE_MAX_CHARS", raising=False)
    grammar = _tool_call_lark_grammar(
        [("write_file", json.loads(_schema_json()))], include_think=True
    )
    assert _prelude_line(grammar) == r"PRELUDE_TEXT: /(.|\n){0,4000}/"
    assert r"TAG_TEXT: /(.|\n)*/" in grammar  # tail/free text unchanged
    assert "tail: TAG_TEXT" in grammar


def test_no_prelude_terminal_when_thinking_is_off(monkeypatch):
    from mtplx.constrained import _tool_call_lark_grammar

    monkeypatch.delenv("MTPLX_THINK_PRELUDE_MAX_CHARS", raising=False)
    grammar = _tool_call_lark_grammar(
        [("write_file", json.loads(_schema_json()))], include_think=False
    )
    assert "PRELUDE_TEXT" not in grammar


@pytest.mark.parametrize("bound", ["4000", "600", "0"])
def test_bounded_prelude_grammars_compile(monkeypatch, bound):
    from mtplx.constrained import _cached_grammar_for_schema, _tool_call_lark_grammar

    monkeypatch.setenv("MTPLX_THINK_PRELUDE_MAX_CHARS", bound)
    for grammar in (
        _cached_grammar_for_schema(_schema_json(), think_prelude=True),
        _tool_call_lark_grammar(
            [("write_file", json.loads(_schema_json()))], include_think=True
        ),
    ):
        assert not llguidance.LLMatcher.validate_grammar(grammar)
