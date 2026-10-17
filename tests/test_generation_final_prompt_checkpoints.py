"""The generation-final commit banks the prompt's recurrent checkpoints.

When the prefill admission skips the prompt-prefix commit (a cold prefill
near the memory line), the generation-final entry was the turn's only entry
and it carried no checkpoints: put() found no donor, the prefill's
checkpoints were dropped, and the next restore below the previous prompt's
end (a new session with the same head, a compacted history) fell back to an
older entry's checkpoint or went cold. Measured 2026-10-17 on Qwen3.8-27B
with Claude Code agents: a shared 11,843-token head restored 1,024 tokens.

Engine always faked: no model loads, no GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

from mtplx.server.openai import (
    ChatMessage,
    _encode_messages,
    _generation_final_prompt_boundaries,
    _store_generation_final_history_snapshot,
)
from tests.test_openai_bridge import _final_state, _ids, _postcommit_state


def _record(position: int) -> tuple:
    return (position, f"state@{position}", None)


def _commit(state, *, boundaries) -> tuple[dict, list[int]]:
    messages = [ChatMessage(role="user", content="hi")]
    prompt_ids = _encode_messages(
        state.runtime.tokenizer,
        messages,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    tokens = _ids("ok")
    final_state = _final_state(tokens)
    final_state.prompt_gdn_boundaries = tuple(boundaries(len(prompt_ids)))
    result = _store_generation_final_history_snapshot(
        state,
        session_id="session-1",
        prompt_ids=prompt_ids,
        generated={"tokens": tokens, "_final_state": final_state},
        messages=messages,
        assistant_content="ok",
        thinking_enabled=False,
        policy_fingerprint="policy",
    )
    return result, prompt_ids


def test_the_commit_banks_the_prompt_checkpoints():
    state = _postcommit_state()

    result, prompt_ids = _commit(
        state, boundaries=lambda n: [_record(2), _record(n - 1)]
    )

    assert result["stored"] is True
    put = state.sessions.bank.puts[-1]
    assert [record[0] for record in put["gdn_boundaries"]] == [2, len(prompt_ids) - 1]


def test_a_commit_without_prompt_checkpoints_keeps_the_old_put_contract():
    state = _postcommit_state()

    result, _ = _commit(state, boundaries=lambda n: [])

    assert result["stored"] is True
    assert "gdn_boundaries" not in state.sessions.bank.puts[-1]


def test_only_checkpoints_inside_the_shared_prefix_are_kept():
    final_state = SimpleNamespace(
        prompt_gdn_boundaries=(_record(2), _record(4), _record(6), _record(0))
    )

    kept = _generation_final_prompt_boundaries(
        final_state,
        prompt_ids=[1, 2, 3, 4, 5, 6],
        final_token_ids=[1, 2, 3, 4, 9, 9, 9],
    )

    assert [record[0] for record in kept] == [2, 4]


def test_a_final_state_without_the_field_gives_no_checkpoints():
    kept = _generation_final_prompt_boundaries(
        SimpleNamespace(), prompt_ids=[1, 2], final_token_ids=[1, 2, 3]
    )

    assert kept == []
