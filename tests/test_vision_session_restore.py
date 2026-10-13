"""An image turn restores the session's text history (2026-09-18).

The receipt: a Pi session attached an image at token 158,043 of a long text
conversation. The session bank served an old 5,984-token entry and 154,899
tokens were prefilled again, 184 seconds. The request prologue skipped the
session resolution and the committed-id splice for every request carrying an
image, so the raw re-encode of the history parted from the banked ids at the
first assistant turn.

Both now run on the TEXT ids, before the image pads are expanded. These tests
drive the real endpoint prologue and then the REAL restore
(``restore_or_prefill_prompt_state`` on a real ``SessionBank``) with a toy
runtime whose logits depend on every input row, image rows included, so a
restore that aliased pixels would change the logits:

  (a) an image turn after a text history restores up to the committed
      frontier, which sits under the first image pad, and its logits equal a
      cold prefill's;
  (b) an image turn after an earlier image turn restores past the earlier
      image only when that image's digest matches; with other pixels in the
      old slot the restore stays under the old image's first pad;
  (c) a request without an image produces the same ids with the switch on
      and off (the text path is untouched);
  (d) ``MTPLX_VISION_SESSION_RESTORE=0`` restores the old behaviour;
  (e) a conversation whose first message already carries an image restores
      nothing and does not fail.

Every end-to-end test runs twice, once per way a resent history parts from
the committed stream: ``token_seam`` (thinking off: the model sampled a merged
token the tokenizer never produces, so only the committed-id splice can serve
it) and ``stripped_reasoning`` (thinking on: the client resends the answer
without its reasoning, so the committed think bytes are substituted first and
the splice then serves the merged token inside them).

CPU-sized: toy model, 48-id vocabulary, no model pack, no tower.
"""

from __future__ import annotations

import base64
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache
from starlette.testclient import TestClient
from test_server_openai import FakeExecutor, _fake_streaming_session_state

from mtplx.generation import restore_or_prefill_prompt_state
from mtplx.mtp_patch import MTPContract
from mtplx.runtime import MTPLXRuntime
from mtplx.server import openai
from mtplx.server.openai import create_app
from mtplx.session_bank import SessionBank
from mtplx.vision.splice import VisionSplice, vision_bank_key_ids, vision_image_spans

VOCAB = 48
ALPHABET = "abcdefghijklmnopqrstuvwxyz .:"
NEWLINE, IM_START, IM_END, VISION_START, VISION_END = 29, 30, 31, 32, 33
THINK_OPEN, THINK_CLOSE = 35, 36
# "ab" as ONE token: the model sampled it, the tokenizer never produces it.
# This is the non-canonical seam that parts a client's re-tokenized history
# from the committed stream on identical text.
MERGED_AB = 34
PAD = 47
SPECIALS = {
    "<|im_start|>": IM_START,
    "<|im_end|>": IM_END,
    "<|vision_start|>": VISION_START,
    "<|image_pad|>": PAD,
    "<|vision_end|>": VISION_END,
    "<think>": THINK_OPEN,
    "</think>": THINK_CLOSE,
}
SESSION = "image-after-text"
HEADERS = {"x-mtplx-session-id": SESSION}

_MIX = mx.array(
    [[((i * 7 + j * 13) % 31) - 15 for j in range(VOCAB)] for i in range(VOCAB)],
    dtype=mx.float32,
)


class SeamTokenizer:
    """Character-level ChatML tokenizer with one merged token it never emits."""

    all_special_ids = sorted(SPECIALS.values())

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids: list[int] = []
        text = str(text)
        position = 0
        while position < len(text):
            for literal, token in SPECIALS.items():
                if text.startswith(literal, position):
                    ids.append(token)
                    position += len(literal)
                    break
            else:
                char = text[position]
                ids.append(NEWLINE if char == "\n" else ALPHABET.index(char))
                position += 1
        return ids

    def decode(self, tokens, **_kwargs):
        literals = {token: literal for literal, token in SPECIALS.items()}
        parts: list[str] = []
        for token in tokens:
            token = int(token)
            if token == MERGED_AB:
                parts.append("ab")
            elif token == NEWLINE:
                parts.append("\n")
            elif token in literals:
                parts.append(literals[token])
            else:
                parts.append(ALPHABET[token])
        return "".join(parts)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        **_kwargs,
    ):
        turns: list[str] = []
        for message in messages:
            body = message.get("content") or ""
            reasoning = message.get("reasoning_content")
            if message["role"] == "assistant" and reasoning:
                body = f"<think>\n{reasoning}\n</think>\n\n{body}"
            turns.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
        rendered = "".join(turns)
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n"
            if enable_thinking:
                rendered += "<think>\n"
        return self.encode(rendered) if tokenize else rendered


class _OneHotEmbedding:
    def __call__(self, ids):
        return mx.eye(VOCAB, dtype=mx.float32)[ids]


class PixelSensitiveModel:
    """Causal toy model: logits are exact integer sums over EVERY input row,
    vision rows included, through a fixed mixing matrix. A restored KV that
    held another image's rows cannot reproduce a cold prefill's logits."""

    def __init__(self) -> None:
        self.model = SimpleNamespace(embed_tokens=_OneHotEmbedding())

    def make_cache(self):
        return [KVCache()]

    def make_mtp_cache(self):
        return []

    def mtp_update_cache(self, hidden_states, next_token_ids, **_kwargs):
        return hidden_states

    def __call__(
        self,
        input_ids,
        *,
        cache=None,
        return_hidden: bool = False,
        hidden_variant: str | None = None,
        emit_logits: bool = True,
        logits_keep: int | None = None,
        input_embeddings=None,
    ):
        del hidden_variant
        batch, length = int(input_ids.shape[0]), int(input_ids.shape[1])
        rows = (
            input_embeddings
            if input_embeddings is not None
            else mx.eye(VOCAB, dtype=mx.float32)[input_ids]
        )
        keys, _values = cache[0].update_and_fetch(
            rows[:, None, :, :], rows[:, None, :, :]
        )
        hidden = mx.zeros((batch, length, 2), dtype=mx.float32)
        if not emit_logits:
            return (None, hidden) if return_hidden else None
        counts = mx.cumsum(keys[:, 0, :, :], axis=1)[:, -length:, :]
        logits = counts @ _MIX
        keep = length if logits_keep is None else min(length, max(1, int(logits_keep)))
        logits = logits[:, -keep:, :]
        if return_hidden:
            return logits, hidden[:, -keep:, :]
        return logits


def _toy_runtime() -> MTPLXRuntime:
    return MTPLXRuntime(
        model=PixelSensitiveModel(),
        tokenizer=SeamTokenizer(),
        model_path=Path("models/vision-session-restore"),
        mtp_enabled=True,
        contract=MTPContract(),
    )


def _fresh(splice: VisionSplice | None) -> VisionSplice | None:
    return None if splice is None else dataclasses.replace(splice, cursor=0)


def _toy_splice(images: list[bytes]) -> VisionSplice:
    """The real content digests; the tower is a pure function of the digest
    (no pack, no GPU). Rows differ between digests, so pixels reach the
    logits."""

    digests = [openai._image_content_digest(raw) for raw in images]
    # The pad count follows the payload size, as a real grid follows the
    # image size: two images of one size share their pad layout.
    pad_counts = [4 + len(raw) % 3 for raw in images]
    rows = [
        [float((digest % 9973 + 31 * row + column) % 7) for column in range(VOCAB)]
        for digest, count in zip(digests, pad_counts)
        for row in range(count)
    ]
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.array(rows, dtype=mx.float32),
        image_digests=tuple(digests),
        pad_counts=tuple(pad_counts),
    )


def _fake_materialize(_state, images, prompt_ids, *, timing=None):
    """``_materialize_vision_splice`` without a tower: the real pad expansion
    over the toy splice."""

    del timing
    splice = _toy_splice(list(images))
    expanded = openai._expand_image_pads(
        list(prompt_ids), image_pad_id=PAD, pad_counts=list(splice.pad_counts)
    )
    return expanded, splice


class _Engine:
    """Stands in for generation. The prefill is the REAL one: the prompt the
    endpoint built goes through ``restore_or_prefill_prompt_state`` with the
    bank, session and fingerprint the endpoint passed. The reply tokens are
    scripted. The frontier (prompt + reply) is then banked the way the
    generation-final commit banks it: under the content-keyed view."""

    def __init__(self) -> None:
        self.runtime = _toy_runtime()
        self.replies: list[list[int]] = []
        self.calls: list[dict] = []

    def __call__(self, _state, prompt_ids, **kwargs):
        reply = self.replies.pop(0)
        splice = kwargs.get("vision_splice")
        lookup = {
            "session_bank": kwargs.get("session_bank"),
            "session_id": kwargs.get("session_id"),
            "template_hash": kwargs.get("session_template_hash"),
            "draft_head_identity": kwargs.get("session_draft_head_identity"),
            "policy_fingerprint": kwargs.get("session_policy_fingerprint"),
        }
        prompt_state = restore_or_prefill_prompt_state(
            self.runtime,
            list(prompt_ids),
            vision_splice=_fresh(splice),
            store_prefix_snapshot=True,
            **lookup,
        )
        self.calls.append(
            {
                "prompt_ids": list(prompt_ids),
                "splice": splice,
                "cached_tokens": int(prompt_state.cached_tokens),
                # Read out on the engine's own thread: an MLX array belongs
                # to the stream of the thread that built it.
                "logits": prompt_state.logits.tolist(),
                "observability": dict(kwargs.get("request_observability") or {}),
            }
        )
        if lookup["session_bank"] is not None:
            restore_or_prefill_prompt_state(
                self.runtime,
                list(prompt_ids) + reply,
                vision_splice=_fresh(splice),
                store_prefix_snapshot=True,
                **lookup,
            )
        tokenizer = self.runtime.tokenizer
        return {
            "text": tokenizer.decode([t for t in reply if t != IM_END]),
            "tokens": list(reply),
            "stats": {
                "generation_mode": kwargs["generation_mode"],
                "mtp_depth": kwargs["depth"],
                "completion_tokens": len(reply),
                "cached_tokens": int(prompt_state.cached_tokens),
            },
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(reply),
            "finish_reason": "stop",
        }


def _cold_logits(prompt_ids: list[int], images: list[bytes]) -> list:
    """The same prompt prefilled from nothing: no bank, a fresh runtime, and
    a splice built on this thread (see _Engine.__call__)."""

    return restore_or_prefill_prompt_state(
        _toy_runtime(),
        list(prompt_ids),
        vision_splice=_toy_splice(images) if images else None,
    ).logits.tolist()


def _image(payload: bytes) -> dict:
    data = base64.b64encode(payload).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


IMAGE_A = b"screenshot with a blue window"
# Same size, so the same pad layout and the SAME token ids: only the pixels
# (the digest, the embedding rows) differ. This is the alias case.
IMAGE_A_OTHER_PIXELS = b"screenshot with a pink window"
IMAGE_B = b"second screenshot"
# The first answer is long (the text history worth restoring) and starts with
# the merged token, so a re-tokenized history parts from the committed stream
# two tokens into the first answer.
FIRST_ANSWER = [MERGED_AB] + SeamTokenizer().encode(
    " " + ("the quick brown fox jumps over the lazy dog. " * 30).rstrip()
)
SECOND_ANSWER = SeamTokenizer().encode("the window title is cut off on the right.")
SHORT_ANSWER = SeamTokenizer().encode("done.")


@pytest.fixture(params=[False, True], ids=["token_seam", "stripped_reasoning"])
def harness(request, monkeypatch):
    thinking = bool(request.param)
    # The toy prompts are far below the production store threshold.
    monkeypatch.setenv("MTPLX_SESSION_STORE_ON_PREFILL_MIN_SUFFIX", "1")
    monkeypatch.delenv("MTPLX_VISION_SESSION_RESTORE", raising=False)
    state = _fake_streaming_session_state()
    state.runtime.tokenizer = SeamTokenizer()
    state.sessions = openai.EngineSessionManager(
        bank=SessionBank(
            max_entries=16, max_bytes=1 << 30, per_session_max_bytes=1 << 30
        )
    )
    state.generation_executor = ThreadPoolExecutor(max_workers=1)
    state.postcommit_executor = FakeExecutor()
    state.args.enable_thinking = thinking
    state._vision_spec_cache = SimpleNamespace(image_token_id=PAD)
    engine = _Engine()
    monkeypatch.setattr(openai, "_run_generation", engine)
    monkeypatch.setattr(openai, "_materialize_vision_splice", _fake_materialize)
    # The scripted replies carry no engine final state; the frontier is
    # banked by the stand-in engine above instead.
    monkeypatch.setattr(
        openai,
        "_store_generation_final_history_snapshot",
        lambda *_args, **_kwargs: {"stored": False, "mode": "unsafe", "reason": "test"},
    )
    monkeypatch.setattr(
        openai,
        "_schedule_idle_postcommit_snapshot",
        lambda *_args, **_kwargs: {"stored": False, "mode": "async_pending"},
    )
    with TestClient(create_app(state)) as client:
        yield SimpleNamespace(
            state=state, engine=engine, client=client, thinking=thinking
        )
    state.generation_executor.shutdown(wait=False)


def _reply(harness, answer: list[int]) -> list[int]:
    """The tokens the model emits after the generation prompt: with thinking
    on the prompt ends in ``<think>``, so the reasoning comes first."""

    reasoning = (
        SeamTokenizer().encode("the fox looks quick.\n</think>\n\n")
        if harness.thinking
        else []
    )
    return [*reasoning, *answer, IM_END]


def _chat(harness, messages, answer, *, headers=HEADERS):
    reply = _reply(harness, answer)
    harness.engine.replies.append(reply)
    response = harness.client.post(
        "/v1/chat/completions",
        headers=headers,
        json={
            "messages": messages,
            "max_tokens": 64,
            "enable_thinking": harness.thinking,
        },
    )
    assert response.status_code == 200, response.text
    call = harness.engine.calls[-1]
    call["reply"] = reply
    # The client resends the visible answer only, as agent clients do.
    echoed = {
        "role": "assistant",
        "content": response.json()["choices"][0]["message"]["content"],
    }
    return echoed, call


def _text_turn(harness) -> tuple[list[dict], list[int]]:
    """One text turn; returns the history the client resends and the ids the
    session committed (prompt + the reply with its merged token)."""

    messages = [{"role": "user", "content": "describe the fox."}]
    echoed, call = _chat(harness, messages, FIRST_ANSWER)
    assert call["cached_tokens"] == 0
    committed = call["prompt_ids"] + call["reply"]
    session = harness.state.sessions.peek(SESSION)
    assert list(session.committed_token_ids) == committed
    return [*messages, echoed], committed


def _image_message(text: str, payload: bytes) -> dict:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text}, _image(payload)],
    }


def _first_pad(prompt_ids: list[int]) -> int:
    return prompt_ids.index(PAD)


def test_image_turn_after_text_history_restores_to_the_pad_boundary(harness):
    history, committed = _text_turn(harness)
    assert len(committed) > 1300  # a history worth restoring

    _echoed, call = _chat(
        harness, [*history, _image_message("look at this: ", IMAGE_A)], SECOND_ANSWER
    )

    # The served ids extend the committed stream: the canonicalization ran on
    # the text ids, and the pads were expanded afterwards.
    prompt_ids = call["prompt_ids"]
    assert prompt_ids[: len(committed)] == committed
    first_pad = _first_pad(prompt_ids)
    assert prompt_ids.count(PAD) == call["splice"].pad_counts[0]
    # The whole text history is restored; nothing at or past a pad is.
    assert call["cached_tokens"] == len(committed)
    assert call["cached_tokens"] <= first_pad
    assert first_pad - call["cached_tokens"] <= 256  # within one block
    assert call["logits"] == _cold_logits(prompt_ids, [IMAGE_A])

    observed = call["observability"]
    assert observed["first_image_pad_position"] == first_pad
    assert observed["request_vision_session_restore"] == {
        "enabled": True,
        "canonicalized": True,
        "refused": None,
        "session_committed_tokens": len(committed),
        "committed_prefix_tokens": len(committed),
    }
    canonicalization = observed["committed_reasoning_canonicalization"]
    assert canonicalization["applied"] is True
    assert canonicalization["cp_spliced"] == len(committed)
    assert canonicalization["cp_raw"] < 64  # where the old path stopped
    assert canonicalization.get("turns_substituted", 0) == (
        1 if harness.thinking else 0
    )
    # (This dict is what generation spreads into the request log row, next to
    # its own cached_tokens: "held N, shares M, first pad at P, restored K".)
    assert observed["request_vision_images"] == 1
    # The image turn advances the session frontier in the content-keyed
    # view: its image rows stand for image A, never for the raw pad id.
    frontier = harness.state.sessions.peek(SESSION).committed_token_ids
    assert list(frontier) == vision_bank_key_ids(
        prompt_ids + call["reply"], call["splice"]
    )
    assert PAD not in frontier


def _image_turn(harness) -> tuple[list[dict], list[int], dict]:
    history, committed = _text_turn(harness)
    messages = [*history, _image_message("look at this: ", IMAGE_A)]
    echoed, call = _chat(harness, messages, SECOND_ANSWER)
    return [*messages, echoed], committed, call


def _second_image_request(history: list[dict], first_image: bytes) -> list[dict]:
    messages = [dict(message) for message in history]
    messages[2] = _image_message("look at this: ", first_image)
    return [*messages, _image_message("and now this: ", IMAGE_B)]


def test_second_image_turn_restores_past_the_first_image_on_a_matching_digest(harness):
    history, committed, first_call = _image_turn(harness)
    if harness.thinking:
        # A client that keeps the image turn's reasoning resends the turn
        # byte for byte. (One that strips it: see the floor test below.)
        history[-1]["reasoning_content"] = "the fox looks quick."
    banked_frontier = len(first_call["prompt_ids"]) + len(first_call["reply"])

    _echoed, call = _chat(
        harness, _second_image_request(history, IMAGE_A), SHORT_ANSWER
    )

    prompt_ids, splice = call["prompt_ids"], call["splice"]
    assert prompt_ids[: len(committed)] == committed
    (first_start, first_end), (second_start, _second_end) = vision_image_spans(
        prompt_ids, splice
    )
    # Same pixels in the old slot: the banked prefix that contains the first
    # image is reused whole, and the restore stops under the NEW image.
    assert call["cached_tokens"] == banked_frontier
    assert first_end <= call["cached_tokens"] <= second_start
    assert first_start == _first_pad(prompt_ids)
    assert call["logits"] == _cold_logits(prompt_ids, [IMAGE_A, IMAGE_B])
    assert call["observability"]["request_vision_session_restore"]["refused"] is None


def test_other_pixels_in_the_old_slot_never_restore_past_the_old_image(harness):
    history, committed, _first_call = _image_turn(harness)
    if harness.thinking:
        history[-1]["reasoning_content"] = "the fox looks quick."

    _echoed, call = _chat(
        harness, _second_image_request(history, IMAGE_A_OTHER_PIXELS), SHORT_ANSWER
    )

    prompt_ids, splice = call["prompt_ids"], call["splice"]
    assert splice.image_digests[0] != openai._image_content_digest(IMAGE_A)
    # The text under the old image is still restored (the banked entry that
    # holds the first image serves it, cut back to the first pad); the old
    # image's KV and everything after it are not.
    assert len(committed) <= call["cached_tokens"] <= _first_pad(prompt_ids)
    images = [IMAGE_A_OTHER_PIXELS, IMAGE_B]
    assert call["logits"] == _cold_logits(prompt_ids, images)
    # The harness can tell the two apart: the ids are the ones the banked
    # image produced, and with ITS rows in the old slot they give other
    # logits, so its KV cannot have served this request.
    assert splice.pad_counts == _toy_splice([IMAGE_A, IMAGE_B]).pad_counts
    assert call["logits"] != _cold_logits(prompt_ids, [IMAGE_A, IMAGE_B])


def test_a_client_that_strips_the_image_turns_reasoning_restores_all_of_it(harness):
    """The image turn committed its keyed view, so the session holds the
    image turn's own answer. A client that strips that answer's reasoning
    resends a history that parts from the banked ids there; the repair puts
    the committed reasoning back past the image, and the follow-up restores
    the whole image turn. Never less than the text history."""

    history, committed, first_call = _image_turn(harness)

    _echoed, call = _chat(
        harness, _second_image_request(history, IMAGE_A), SHORT_ANSWER
    )

    prompt_ids = call["prompt_ids"]
    assert prompt_ids[: len(committed)] == committed
    assert call["cached_tokens"] == len(first_call["prompt_ids"]) + len(
        first_call["reply"]
    )
    assert call["logits"] == _cold_logits(prompt_ids, [IMAGE_A, IMAGE_B])
    receipt = call["observability"]["request_vision_session_restore"]
    assert receipt["canonicalized"] is True and receipt["refused"] is None


# The image turn's answer starts with the merged token too, so the resent
# history parts from the session's stream AFTER the image.
IMAGE_ANSWER = [MERGED_AB] + SeamTokenizer().encode(" the window is cut off on the right.")


def test_the_repair_crosses_an_image_its_session_holds(harness):
    history, _committed = _text_turn(harness)
    messages = [*history, _image_message("look at this: ", IMAGE_A)]
    echoed, call = _chat(harness, messages, IMAGE_ANSWER)
    # The image turn committed the content-keyed view of what it was served.
    frontier = vision_bank_key_ids(call["prompt_ids"] + call["reply"], call["splice"])
    assert list(harness.state.sessions.peek(SESSION).committed_token_ids) == frontier

    _echoed, follow = _chat(
        harness, _second_image_request([*messages, echoed], IMAGE_A), SHORT_ANSWER
    )

    # The repair re-expressed the image turn's answer in the model's own
    # tokens past the image, so the served prompt extends the banked turn
    # and restores all of it.
    served = vision_bank_key_ids(follow["prompt_ids"], follow["splice"])
    assert served[: len(frontier)] == frontier
    assert follow["cached_tokens"] == len(frontier)
    assert follow["logits"] == _cold_logits(follow["prompt_ids"], [IMAGE_A, IMAGE_B])
    receipt = follow["observability"]["request_vision_session_restore"]
    assert receipt["canonicalized"] is True and receipt["refused"] is None
    assert receipt["committed_prefix_tokens"] == len(frontier)


def test_a_pi_session_restores_all_but_the_new_turn_after_a_screenshot(harness):
    """Consecutive Pi-shaped prompts: two text turns, a screenshot, then an
    edit, a write and a read turn. Every answer starts with the token the
    model sampled and the tokenizer never produces (and, with thinking on,
    the client drops its reasoning), so every history the client re-sends
    parts from the session's stream inside the previous answer, after the
    screenshot. On 2026-09-29 each such turn restored nothing past the
    screenshot and read 123K to 138K tokens again. Every turn after the
    screenshot restores its prompt minus the new turn (the previous turn's
    prompt and answer), and its logits equal a cold prefill."""

    messages: list[dict] = [{"role": "user", "content": "read the game file."}]
    echoed, _call = _chat(harness, messages, FIRST_ANSWER)
    messages += [echoed, {"role": "user", "content": "run it."}]
    echoed, _call = _chat(harness, messages, IMAGE_ANSWER)
    messages += [echoed, _image_message("screenshot: ", IMAGE_A)]
    echoed, previous = _chat(harness, messages, IMAGE_ANSWER)
    messages.append(echoed)
    for request in ("edit the bird speed.", "write the file.", "read it back."):
        messages.append({"role": "user", "content": request})
        echoed, call = _chat(harness, messages, IMAGE_ANSWER)
        messages.append(echoed)
        held = len(previous["prompt_ids"]) + len(previous["reply"])
        assert call["cached_tokens"] >= held, request
        assert call["logits"] == _cold_logits(call["prompt_ids"], [IMAGE_A])
        receipt = call["observability"]["request_vision_session_restore"]
        assert receipt["canonicalized"] is True and receipt["refused"] is None
        previous = call


def test_the_splice_never_copies_across_image_rows_that_differ():
    tokenizer = SeamTokenizer()
    a = tokenizer.encode("look ") + [VISION_START, PAD, PAD, VISION_END]
    committed = _keyed([*a, MERGED_AB, 26], (DIGEST_A, 2))  # "ab" as one token
    same = _keyed([*a, 0, 1, 26], (DIGEST_A, 2))  # "a", "b": a seam after the image
    other = _keyed([*a, 0, 1, 26], (DIGEST_C, 2))  # the same text, other pixels
    spliced, receipt = openai._splice_committed_token_ids(same, committed, tokenizer)
    assert spliced == committed and receipt["spans"] == 1
    spliced, receipt = openai._splice_committed_token_ids(other, committed, tokenizer)
    assert spliced == other and receipt["spans"] == 0


def _follow_up_text_ids(harness, monkeypatch, switch: str | None) -> list[int]:
    if switch is None:
        monkeypatch.delenv("MTPLX_VISION_SESSION_RESTORE", raising=False)
    else:
        monkeypatch.setenv("MTPLX_VISION_SESSION_RESTORE", switch)
    history, committed = _text_turn(harness)
    _echoed, call = _chat(
        harness, [*history, {"role": "user", "content": "and the dog."}], SHORT_ANSWER
    )
    assert call["splice"] is None
    assert call["prompt_ids"][: len(committed)] == committed
    assert call["cached_tokens"] == len(committed)
    assert "request_vision_session_restore" not in call["observability"]
    assert "first_image_pad_position" not in call["observability"]
    return call["prompt_ids"]


def test_text_only_ids_do_not_depend_on_the_switch(harness, monkeypatch):
    default_ids = _follow_up_text_ids(harness, monkeypatch, None)
    harness.state.sessions.clear_all()
    switched_off_ids = _follow_up_text_ids(harness, monkeypatch, "0")
    assert default_ids == switched_off_ids


def test_kill_switch_restores_the_old_image_turn(harness, monkeypatch):
    monkeypatch.setenv("MTPLX_VISION_SESSION_RESTORE", "0")
    history, committed = _text_turn(harness)

    _echoed, call = _chat(
        harness, [*history, _image_message("look at this: ", IMAGE_A)], SECOND_ANSWER
    )

    # The raw re-encode parts from the committed stream at the first
    # assistant turn (its reasoning is gone, or its merged token came back
    # as two), so nothing past that point can be restored: the whole text
    # history is prefilled again.
    prompt_ids = call["prompt_ids"]
    parted_at = openai.common_prefix_len(prompt_ids, committed)
    assert parted_at < 64
    assert call["cached_tokens"] <= parted_at
    assert call["logits"] == _cold_logits(prompt_ids, [IMAGE_A])
    observed = call["observability"]
    assert "committed_reasoning_canonicalization" not in observed
    assert observed["request_vision_session_restore"] == {
        "enabled": False,
        "canonicalized": False,
        "refused": "kill_switch",
    }


def test_first_message_with_an_image_restores_nothing(harness):
    _echoed, call = _chat(
        harness,
        [_image_message("what is this: ", IMAGE_A)],
        SECOND_ANSWER,
        headers={"x-mtplx-session-id": "image-first"},
    )

    assert call["cached_tokens"] == 0
    assert call["prompt_ids"].count(PAD) == call["splice"].pad_counts[0]
    assert call["observability"]["request_vision_session_restore"] == {
        "enabled": True,
        "canonicalized": False,
        "refused": "no_committed_stream",
        "session_committed_tokens": 0,
        "committed_prefix_tokens": 0,
    }
    assert "committed_reasoning_canonicalization" not in call["observability"]


def _record_sweep_order(harness, monkeypatch) -> list[str]:
    """Order of the cross-session postcommit sweep and the pad expansion."""

    events: list[str] = []
    real_sweep = harness.state.sessions.abort_cross_session_postcommits

    def sweep(*, except_session_id=None, **kwargs):
        events.append(f"sweep:{except_session_id}")
        return real_sweep(except_session_id=except_session_id, **kwargs)

    def materialize(*args, **kwargs):
        events.append("expand")
        return _fake_materialize(*args, **kwargs)

    monkeypatch.setattr(
        harness.state.sessions, "abort_cross_session_postcommits", sweep
    )
    monkeypatch.setattr(openai, "_materialize_vision_splice", materialize)
    return events


def test_a_named_session_sweeps_before_the_gate_like_a_text_request(
    harness, monkeypatch
):
    events = _record_sweep_order(harness, monkeypatch)
    _chat(harness, [_image_message("see: ", IMAGE_A)], SHORT_ANSWER)
    assert events == [f"sweep:{SESSION}", "expand"]


def test_an_inferred_session_sweeps_once_its_lineage_is_known(harness, monkeypatch):
    """Without a session header the image request may still move onto the
    lineage of its pixel-keyed bank entry after the pads are expanded. Its
    sweep stays at the original site, so it spares the FINAL session (as it
    always did) and cannot abort that lineage's own pending commit."""

    events = _record_sweep_order(harness, monkeypatch)
    _echoed, call = _chat(
        harness, [_image_message("see: ", IMAGE_A)], SHORT_ANSWER, headers={}
    )
    assert events[0] == "expand"
    assert len(events) == 2 and events[1].startswith("sweep:")
    assert call["observability"]["request_session_source"] == "new"
    # The gate still ran; there was simply nothing to run it against.
    assert call["observability"]["request_vision_session_restore"]["refused"] == (
        "no_committed_stream"
    )


# ---------------------------------------------------------------------------
# the guard on the canonicalized TEXT ids


def _keyed(ids: list[int], *images: tuple[int, int]) -> list[int]:
    """``ids`` in the content-keyed view, ``(digest, pad count)`` per image."""

    splice = VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.zeros((sum(count for _digest, count in images), 1)),
        image_digests=tuple(digest for digest, _count in images),
        pad_counts=tuple(count for _digest, count in images),
    )
    return vision_bank_key_ids(list(ids), splice)


# Text, image A (two rows), text, image B (two rows), text.
TWO_IMAGES = [1, 2, 3, 4, VISION_START, PAD, PAD, VISION_END, 5, 6,
              VISION_START, PAD, PAD, VISION_END, 8, 9]
DIGEST_A, DIGEST_B, DIGEST_C = 0xA, 0xB, 0xC


def test_the_repair_stops_at_the_first_image_its_session_does_not_hold():
    refusal = openai._vision_canonicalization_refusal
    raw = _keyed(TWO_IMAGES, (DIGEST_A, 2), (DIGEST_B, 2))
    between = [*raw[:9], 7, *raw[10:]]  # the text between the images changed
    after = [*raw[:15], 7]  # the text after the second image changed
    # A stream that holds no image (a text session, or one from before image
    # sessions committed): legal only before the first image, as it was.
    text_only = [1, 2, 3, 4]
    assert refusal(raw, [1, 9, 4, *raw[4:]], text_only, PAD) is None
    assert refusal(raw, raw, text_only, PAD) is None
    assert refusal(raw, between, text_only, PAD) == (
        "canonicalization_crossed_an_image_not_in_the_session"
    )
    # A stream that holds image A: the stretch after it is the session's own.
    holds_a = _keyed([*TWO_IMAGES[:8], 5, 7], (DIGEST_A, 2))
    assert refusal(raw, between, holds_a, PAD) is None
    assert refusal(raw, after, holds_a, PAD) == (
        "canonicalization_crossed_an_image_not_in_the_session"
    )
    # A stream that holds both images: every stretch may be repaired.
    holds_both = _keyed(TWO_IMAGES[:14], (DIGEST_A, 2), (DIGEST_B, 2))
    assert refusal(raw, after, holds_both, PAD) is None
    # A stream whose first image has other pixels holds none of the request's.
    holds_c = _keyed([*TWO_IMAGES[:8], 5, 7], (DIGEST_C, 2))
    assert refusal(raw, between, holds_c, PAD) == (
        "canonicalization_crossed_an_image_not_in_the_session"
    )
    # An image the session holds must come back byte-identical.
    other_rows = _keyed(TWO_IMAGES, (DIGEST_C, 2), (DIGEST_B, 2))
    assert refusal(raw, [*other_rows[:9], 7, *raw[10:]], holds_a, PAD) == (
        "canonicalization_changed_an_image"
    )
    # A placeholder the raw encode did not have would shift every image.
    assert refusal(raw, [1, PAD, *raw[2:]], text_only, PAD) == (
        "canonicalization_added_image_placeholder"
    )
    assert refusal([1, 2, 3], [1, 2, 3], text_only, PAD) == "image_placeholder_missing"
    assert refusal(raw, raw, text_only, None) == "image_pad_token_unknown"


def _gate_state(committed):
    return SimpleNamespace(
        _vision_spec_cache=SimpleNamespace(image_token_id=PAD),
        sessions=SimpleNamespace(
            peek=lambda _sid: SimpleNamespace(committed_token_ids=tuple(committed))
        ),
    )


def _one_image_splice() -> VisionSplice:
    return VisionSplice(
        image_pad_token_id=PAD,
        embeddings=mx.zeros((1, 1)),
        image_digests=(DIGEST_A,),
        pad_counts=(1,),
    )


def test_a_refused_canonicalization_serves_the_raw_encode_and_says_why():
    raw = [1, 2, 9, VISION_START, PAD, VISION_END, 5]
    raw_keyed = _keyed(raw, (DIGEST_A, 1))
    crossed = [1, 2, 3, 4, *raw_keyed[3:6], 6]
    template_observability = {"chat_encode_cache": "miss"}
    gate_observability = {
        "stable_prefix_len": 3,
        "committed_reasoning_canonicalization": {"applied": True, "cp_raw": 2},
    }
    receipt = {"enabled": True, "canonicalized": False, "refused": None}

    served = openai._vision_gate_canonicalization(
        _gate_state((1, 2, 3, 4)),
        raw_keyed=raw_keyed,
        canonicalized=(["canon"], crossed),
        splice=_one_image_splice(),
        canon_observability=gate_observability,
        template_observability=template_observability,
        receipt=receipt,
        session_id="s",
    )

    assert served is None
    # The raw encode's observability stands; only the gate's record is kept.
    assert template_observability == {
        "chat_encode_cache": "miss",
        "committed_reasoning_canonicalization": {
            "applied": False,
            "cp_raw": 2,
            "refused_reason": "canonicalization_crossed_an_image_not_in_the_session",
        },
    }
    assert receipt == {
        "enabled": True,
        "canonicalized": False,
        "refused": "canonicalization_crossed_an_image_not_in_the_session",
        "session_committed_tokens": 4,
        "committed_prefix_tokens": 2,
    }


def test_a_served_canonicalization_hands_the_model_its_ids_and_splice():
    raw = [1, 2, 9, VISION_START, PAD, VISION_END, 5]
    raw_keyed = _keyed(raw, (DIGEST_A, 1))
    repaired = [1, 2, 3, 4, *raw_keyed[3:]]  # a seam before the image
    template_observability = {"chat_encode_cache": "miss"}
    gate_observability = {"committed_reasoning_canonicalization": {"applied": True}}
    receipt = {"enabled": True, "canonicalized": False, "refused": None}

    messages, served_ids, splice = openai._vision_gate_canonicalization(
        _gate_state((1, 2, 3, 4)),
        raw_keyed=raw_keyed,
        canonicalized=(["canon"], repaired),
        splice=_one_image_splice(),
        canon_observability=gate_observability,
        template_observability=template_observability,
        receipt=receipt,
        session_id="s",
    )

    assert messages == ["canon"]
    # The model reads the pad id; the rows ride the splice, from its start.
    assert served_ids == [1, 2, 3, 4, VISION_START, PAD, VISION_END, 5]
    assert splice.cursor == 0 and splice.pad_counts == (1,)
    assert template_observability == gate_observability
    assert receipt == {
        "enabled": True,
        "canonicalized": True,
        "refused": None,
        "session_committed_tokens": 4,
        "committed_prefix_tokens": 4,
    }


def test_switch_is_on_by_default(monkeypatch):
    monkeypatch.delenv("MTPLX_VISION_SESSION_RESTORE", raising=False)
    assert openai._vision_session_restore_enabled()
    for spelling in ("0", "off", "false", "no"):
        monkeypatch.setenv("MTPLX_VISION_SESSION_RESTORE", spelling)
        assert not openai._vision_session_restore_enabled()
    monkeypatch.setenv("MTPLX_VISION_SESSION_RESTORE", "1")
    assert openai._vision_session_restore_enabled()
