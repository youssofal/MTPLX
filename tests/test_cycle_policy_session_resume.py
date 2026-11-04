"""Two turns under MTPLX_MTP_HISTORY_POLICY=cycle, banked the way the server banks them.

The label fix (tests/test_bank_history_policy_operator_override.py) pins the
strings; this test runs the path: the tiny fixed-M4 pack decodes a turn with
the policy the server hands generation, the server's own generation-final
metadata banks it, and the next turn of the same anonymous session must resume
from it. Before the fix the server labelled that save ``committed`` while
generation looked up ``cycle``, so the follow-up could not resume the answer.
"""

from __future__ import annotations

from types import SimpleNamespace

import mtplx.generation as generation
from mtplx.qwen4_fixed_verify import install_qwen4_fixed_verify_route
from mtplx.server import openai as server
from mtplx.session_bank import SessionBank
from test_qwen4_fixed_m4_capacity_bucket import NATIVE, SEED, lane, pack  # noqa: F401


def _cycle_turns(pack, monkeypatch, *, one_copy: bool):
    smoke, model = pack
    monkeypatch.setenv("MTPLX_MTP_HISTORY_POLICY", "cycle")
    monkeypatch.setenv("MTPLX_ONE_COPY", "1" if one_copy else "0")
    monkeypatch.setenv("MTPLX_COMPILED_VERIFY", "1")
    rt = smoke._tiny_runtime(model)
    install_qwen4_fixed_verify_route(rt)
    hidden_variant = generation._resolve_runtime_base_hidden_variant(rt, None)
    state = SimpleNamespace(
        runtime=rt, backend_descriptor=SimpleNamespace(backend_id="native_mtp")
    )
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    prompt = [(7 * i + 3) % 128 for i in range(48)]
    turns = []
    for turn in range(2):
        result = generation.generate_mtpk(
            rt, list(prompt), max_tokens=24, sampler=NATIVE, draft_sampler=NATIVE,
            speculative_depth=3, seed=SEED + turn, mtp_cache_policy="persistent",
            # What the server passes (_bank_history_policy), as for a real request.
            mtp_history_policy=server._bank_history_policy(state),
            verify_strategy="batched", stop_token_ids=set(), capture_final_state=True,
            session_bank=bank, session_id="anon-cycle", session_restore_mode="reference",
            commit_prompt_state_to_bank=True,
        )
        final = result.final_state
        assert final is not None and final.safe_to_commit
        tokens = list(prompt) + list(result.tokens)
        # The server's generation-final save, with its own labels.
        metadata = server._generation_final_bank_metadata(
            state, final, token_count=len(tokens)
        )
        metadata["hidden_variant"] = hidden_variant
        entry = bank.put(
            runtime=rt, token_ids=tokens, cache=final.final_trunk_cache,
            logits=final.final_logits, hidden=final.final_hidden, keep_live_ref=False,
            session_id="anon-cycle", snapshot_epoch=len(tokens), **metadata,
        )
        assert entry is not None
        turns.append((tokens, result))
        prompt = tokens + [5, 6, 7, 8]
    return turns


def test_a_cycle_session_resumes_the_answer_it_saved(pack, lane):
    turns = _cycle_turns(pack, lane, one_copy=True)
    (first_tokens, first), (_second_tokens, second) = turns
    assert first.stats.mtp_history_policy == "cycle"
    assert second.stats.mtp_history_policy == "cycle"
    # The follow-up resumes the whole previous turn, answer included.
    assert second.stats.cached_tokens == len(first_tokens)
    assert second.stats.session_restore_mode != "cold"


def test_the_copying_store_resumes_the_same_way_under_cycle(pack, lane):
    turns = _cycle_turns(pack, lane, one_copy=False)
    (first_tokens, _first), (_second_tokens, second) = turns
    assert second.stats.cached_tokens == len(first_tokens)
