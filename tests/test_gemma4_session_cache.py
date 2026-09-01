from __future__ import annotations

from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import mtplx.backends.gemma4_assistant as gemma4
from mtplx.backends.descriptors import descriptor_for_backend_id
from mtplx.server import openai
from mtplx.session_bank import SessionBank


class _GemmaTokenizer:
    bos_token = "<bos>"
    model_specific_special_tokens = {
        "think_token": "<|think|>",
        "soc_token": "<|channel>",
        "eoc_token": "<channel|>",
    }

    def encode(self, text, **_kwargs):
        return [ord(char) for char in str(text)]

    def decode(self, token_ids, **_kwargs):
        return "".join(chr(int(token)) for token in token_ids)


class _RecordingBank:
    def __init__(self) -> None:
        self.puts: list[dict] = []

    def put(self, **kwargs):
        self.puts.append(kwargs)
        return SimpleNamespace(
            prefix_len=len(kwargs["token_ids"]),
            nbytes=123,
            token_hash="gemma4-test-prefix",
        )


def _postcommit_state(tokenizer: _GemmaTokenizer, bank: object) -> SimpleNamespace:
    return SimpleNamespace(
        args=SimpleNamespace(
            reasoning_parser="gemma4",
            strip_assistant_reasoning_history=False,
        ),
        backend_descriptor=descriptor_for_backend_id("gemma4_assistant"),
        runtime=SimpleNamespace(
            tokenizer=tokenizer,
            model_path=Path("models/gemma4"),
            mtp_enabled=True,
        ),
        sessions=SimpleNamespace(bank=bank),
        template_hash="gemma4-template",
        draft_head_identity="gemma4-assistant-head",
        lock=Lock(),
    )


def test_gemma4_reasoning_history_is_token_identical_and_uses_native_bank_state():
    tokenizer = _GemmaTokenizer()
    bank = _RecordingBank()
    state = _postcommit_state(tokenizer, bank)
    messages = [openai.ChatMessage(role="user", content="Explain the result.")]
    prompt_ids = openai._encode_messages(
        tokenizer,
        messages,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    reasoning = "Check the invariant first."
    content = "The invariant holds."
    generated_ids = tokenizer.encode(
        f"<|channel>thought\n{reasoning}<channel|>{content}<turn|>"
    )
    generated = {
        "tokens": generated_ids,
        "_final_state": SimpleNamespace(
            final_trunk_cache=["gemma4-cache"],
            final_logits="gemma4-logits",
            final_hidden="gemma4-pre-norm-hidden",
            final_committed_mtp_cache=None,
            generated_token_ids=tuple(generated_ids),
            safe_to_commit=False,
            finish_reason="stop",
            extra_state={
                "gemma4_shared_kv_states": {"layer": "shared-kv"},
                "gemma4_kv_offset": len(prompt_ids) + len(generated_ids),
                "gemma4_session_state_policy": "assistant_shared_kv",
            },
            prompt_boundary_cache=["gemma4-prompt-cache"],
            prompt_boundary_logits="gemma4-prompt-logits",
            prompt_boundary_hidden="gemma4-prompt-pre-norm-hidden",
            prompt_boundary_extra_state={
                "gemma4_shared_kv_states": {"layer": "prompt-shared-kv"},
                "gemma4_kv_offset": len(prompt_ids),
                "gemma4_session_state_policy": "assistant_shared_kv",
            },
        ),
    }

    echoed_history_ids = openai._encode_messages(
        tokenizer,
        [
            *messages,
            openai.ChatMessage(
                role="assistant",
                content=content,
                reasoning_content=reasoning,
            ),
        ],
        enable_thinking=True,
        add_generation_prompt=False,
    )
    committed_ids = prompt_ids + generated_ids

    assert echoed_history_ids[: len(committed_ids)] == committed_ids
    assert openai._generation_final_bank_commit_safe(
        state,
        generated["_final_state"],
    )

    result = openai._store_generation_final_history_snapshot(
        state,
        session_id="gemma4-session",
        prompt_ids=prompt_ids,
        generated=generated,
        messages=messages,
        assistant_content=content,
        thinking_enabled=True,
        policy_fingerprint="gemma4-policy",
    )

    assert result["stored"] is True
    assert result["mode"] == "generation_prompt_boundary"
    assert result["reason"] == "prompt_boundary_before_unsafe_history"
    assert result["prefix_len"] == len(prompt_ids)
    assert result["history_suffix_tokens"] == 0
    assert bank.puts[0]["hidden_variant"] == "gemma4_pre_norm"
    assert bank.puts[0]["mtp_history_policy"] == "assistant_shared_kv"
    assert bank.puts[0]["mtp_history_snapshot"] is None
    assert bank.puts[0]["token_ids"] == prompt_ids
    assert bank.puts[0]["cache"] == ["gemma4-prompt-cache"]
    assert bank.puts[0]["logits"] == "gemma4-prompt-logits"
    assert bank.puts[0]["hidden"] == "gemma4-prompt-pre-norm-hidden"
    assert (
        bank.puts[0]["extra_state"]
        == generated["_final_state"].prompt_boundary_extra_state
    )


class _TokenCache:
    def __init__(self, token_ids: list[int] | None = None) -> None:
        self.token_ids = list(token_ids or [])
        self.offset = len(self.token_ids)

    @property
    def state(self):
        raise RuntimeError("live reference only")

    def is_trimmable(self) -> bool:
        return True

    def trim(self, count: int) -> int:
        count = int(count)
        if count:
            del self.token_ids[-count:]
        self.offset = len(self.token_ids)
        return count


class _GemmaRuntime:
    model_path = Path("models/gemma4")
    mtp_enabled = True

    def __init__(self) -> None:
        self.target = SimpleNamespace(cache_offset=lambda cache: cache[0].offset)

    def make_cache(self):
        return [_TokenCache()]


class _SliceValue:
    def __init__(self, kind: str, prompt: tuple[int, ...]) -> None:
        self.kind = kind
        self.prompt = prompt

    def __getitem__(self, _key):
        return self

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, _SliceValue)
            and self.kind == other.kind
            and self.prompt == other.prompt
        )


def test_gemma4_two_turn_memory_rewrite_restores_common_prefix_with_cold_parity(
    monkeypatch,
):
    tokenizer = _GemmaTokenizer()
    stable_context = "stable project context " * 35
    first_messages = [
        openai.ChatMessage(role="system", content=stable_context),
        openai.ChatMessage(
            role="user",
            content="<memory_context>volatile one</memory_context>\nFirst question",
        ),
    ]
    first_prompt = openai._encode_messages(
        tokenizer,
        first_messages,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    reasoning = "Use the stable context."
    content = "First answer."
    first_generated = tokenizer.encode(
        f"<|channel>thought\n{reasoning}<channel|>{content}<turn|>"
    )
    committed_ids = first_prompt + first_generated

    second_messages = [
        openai.ChatMessage(role="system", content=stable_context),
        openai.ChatMessage(role="user", content="First question"),
        openai.ChatMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning,
        ),
        openai.ChatMessage(
            role="user",
            content="<memory_context>volatile two</memory_context>\nSecond question",
        ),
    ]
    second_prompt = openai._encode_messages(
        tokenizer,
        second_messages,
        enable_thinking=True,
        add_generation_prompt=True,
    )
    common = openai._common_prefix_len(committed_ids, second_prompt)
    assert common > 512
    assert common < len(first_prompt)

    runtime = _GemmaRuntime()
    bank = SessionBank(max_entries=2, max_bytes=1024, per_session_max_bytes=512)
    entry = bank.put(
        runtime=runtime,
        token_ids=first_prompt,
        cache=[_TokenCache(first_prompt)],
        logits=("stored", len(first_prompt)),
        hidden=("stored", len(first_prompt)),
        hidden_variant="gemma4_pre_norm",
        keep_live_ref=True,
        session_id="gemma4-session",
        template_hash="gemma4-template",
        mtp_history_policy="assistant_shared_kv",
        draft_head_identity="gemma4-assistant-head",
        policy_fingerprint="gemma4-policy",
        snapshot_epoch=len(first_prompt),
        nbytes_override=1024,
        extra_state={
            "gemma4_shared_kv_states": {"stored": True},
            "gemma4_kv_offset": len(first_prompt),
            "gemma4_session_state_policy": "assistant_shared_kv",
        },
    )
    assert entry is not None and entry.live_ref_only

    prefill_calls = []

    def fake_prefill(_runtime, prompt_ids, *, cache, phase):
        assert phase == "prefill"
        prefill_calls.append(tuple(prompt_ids))
        cache[0].token_ids.extend(int(token) for token in prompt_ids)
        cache[0].offset = len(cache[0].token_ids)
        full_prompt = tuple(cache[0].token_ids)
        return (
            SimpleNamespace(
                logits=_SliceValue("logits", full_prompt),
                hidden=_SliceValue("hidden", full_prompt),
                shared_kv_states={"prompt": full_prompt},
                cache_offset=len(full_prompt),
            ),
            0.01,
        )

    monkeypatch.setattr(gemma4, "_gemma4_prefill_prompt", fake_prefill)
    cold = gemma4._restore_or_prefill_gemma4_prompt(
        runtime,
        second_prompt,
        require_shared_kv=True,
    )
    warm = gemma4._restore_or_prefill_gemma4_prompt(
        runtime,
        second_prompt,
        session_bank=bank,
        session_restore_mode="reference",
        session_template_hash="gemma4-template",
        session_draft_head_identity="gemma4-assistant-head",
        session_policy_fingerprint="gemma4-policy",
        require_shared_kv=True,
    )

    assert warm.cache_hit is True
    assert warm.cached_tokens == common
    assert warm.suffix_tokens == len(second_prompt) - common
    assert warm.logits == cold.logits
    assert warm.hidden == cold.hidden
    assert warm.shared_kv_states == cold.shared_kv_states
    assert warm.kv_offset == cold.kv_offset == len(second_prompt)
    assert prefill_calls == [
        tuple(second_prompt),
        tuple(second_prompt[common:]),
    ]
