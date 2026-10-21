"""One request, one compare per RAM entry: shared_prefix_len memo.

Results and diagnostics of the bank lookups must be identical whether the
prompt arrives as a list (every call converts and compares on its own) or as
one normalized TokenTuple (compares shared across the lookups).
"""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

from mtplx import session_bank as sb
from mtplx.session_bank import SessionBank, common_prefix_len, shared_prefix_len
from mtplx.token_ids import as_token_tuple


def _bank(rng):
    bank = SessionBank(max_entries=32, max_bytes=1 << 30, per_session_max_bytes=1 << 30)
    runtime = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=False)
    base = [rng.randrange(1000, 60000) for _ in range(600)]
    for i, n in enumerate((80, 150, 300, 420, 600)):
        toks = base[:n] if i % 2 == 0 else base[: n - 20] + [90000 + i] * 20
        bank.put(
            runtime=runtime, token_ids=toks, cache=[], logits=None, hidden=None,
            session_id=f"s{i % 2}", nbytes_override=100,
        )
    return bank, base


def test_shared_prefix_len_matches_plain():
    rng = random.Random(3)
    for _ in range(50):
        a = [rng.randrange(5) for _ in range(rng.randrange(0, 40))]
        b = a[: rng.randrange(0, len(a) + 1)] + [rng.randrange(5) for _ in range(rng.randrange(0, 5))]
        ta, tb = as_token_tuple(a), as_token_tuple(b)
        want = common_prefix_len(a, b)
        assert shared_prefix_len(ta, tb) == want
        assert shared_prefix_len(ta, tb) == want  # memo hit
        assert shared_prefix_len(a, b) == want  # plain inputs bypass the memo
        assert shared_prefix_len(tuple(a), tb) == want


def test_memo_is_identity_keyed_and_bounded():
    a = as_token_tuple([1, 2, 3, 4])
    b = as_token_tuple([1, 2, 9])
    assert shared_prefix_len(a, b) == 2
    # An equal but different object is a separate key with the same answer.
    assert shared_prefix_len(as_token_tuple([1, 2, 3, 4]), b) == 2
    for i in range(sb._SHARED_LEN_MEMO_MAX + 20):
        shared_prefix_len(as_token_tuple([i]), b)
    assert len(sb._shared_len_memo) <= sb._SHARED_LEN_MEMO_MAX


def test_bank_lookups_identical_for_list_and_token_tuple():
    rng = random.Random(11)
    bank, base = _bank(rng)
    kw = dict(max_token_gap=8, min_matched_tokens=16, block_size=16, block_min_matched_tokens=16)
    prompts = [
        base[:n] + [rng.randrange(70000, 80000) for _ in range(k)]
        for n in (50, 150, 299, 420, 600)
        for k in (0, 5, 60)
    ]
    for prompt in prompts:
        for session in (None, "s0", "s1"):
            norm = as_token_tuple(prompt)
            assert bank.longest_shared_prefix_tokens(prompt, session_id=session) == \
                bank.longest_shared_prefix_tokens(norm, session_id=session)
        assert bank.longest_prefix(prompt) is bank.longest_prefix(as_token_tuple(prompt))
        got_list = bank.near_prefix_candidates(list(prompt), **kw)
        diag_list = dict(bank.last_prefix_diagnostic or {})
        norm = as_token_tuple(prompt)
        got_norm = bank.near_prefix_candidates(norm, **kw)
        diag_norm = dict(bank.last_prefix_diagnostic or {})
        got_norm_again = bank.near_prefix_candidates(norm, **kw)
        assert [(e.token_hash, m) for e, m in got_list] == [(e.token_hash, m) for e, m in got_norm]
        assert [(e.token_hash, m) for e, m in got_norm] == [
            (e.token_hash, m) for e, m in got_norm_again
        ]
        assert diag_list == diag_norm
        assert bank.same_conversation_entries("s0", prompt) == bank.same_conversation_entries(
            "s0", norm
        )
