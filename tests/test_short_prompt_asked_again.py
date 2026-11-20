"""A short prompt asked again is served from what its first answer left.

In the final comparison of 12 November the founder's 76-token prompt reached
first text in 0.23 s every time it was sent, cached 0 of 76, where another
engine took 0.09 s. The bank held everything the repeat needed: the first
answer's entry starts with the prompt and keeps a checkpoint at its end. But
an entry that runs on past the prompt is offered through the block lane,
which wants 512 matched tokens, so a prompt under that length was read again
whole (one forward, about 0.2 s whatever its width). The lane now takes a
prompt the entry holds whole at any length; a prompt that only shares its
opening with a stored conversation is still read, as before.

Host tests on the matcher and the lane, then the same turn on the tiny pack.
"""

from __future__ import annotations

import mtplx.generation as generation
from mtplx.session_bank import SessionBank, _near_prefix_candidate_len
from test_lease_survives_aborted_prefill import _prompt, _runtime
from test_near_lane_releases_refused import _near, _put, _WatchingModel
from test_near_lane_releases_refused import _runtime as _toy_runtime
from test_one_copy_conversation import _one_turn
from test_qwen4_fixed_m4_capacity_bucket import NATIVE, SEED, lane, pack  # noqa: F401
from test_vision_keyed_restore_rules import _entry

# What the server asks the matcher with (generation._restore_near_prefix_prompt_state).
SHIPPED = dict(gap_limit=8, min_match=64, block=256, block_min_match=512)
QUESTION = [(5 * i + 2) % 400 for i in range(76)]
ANSWER = [(3 * i + 7) % 400 for i in range(300)]


def _offered(entry, prompt):
    return _near_prefix_candidate_len(
        entry, tuple(prompt), allow_block_prefix=True, **SHIPPED
    )


def test_the_entry_of_its_first_answer_offers_the_whole_prompt():
    # The first ask's prefill left its last checkpoint one token before the
    # prompt's end: that is how the entry knows a prompt ended here.
    entry = _entry(QUESTION + ANSWER, boundaries=(11, 75))
    assert _offered(entry, QUESTION) == len(QUESTION)


def test_a_prompt_that_ends_inside_what_the_model_wrote_is_still_read():
    """The entry holds it whole, but no prompt of the conversation ended
    there: what follows the stored prompt is the model's own text, and a
    prompt is never served from that (tests/test_session_bank.py guards the
    same for the tiny-gap lane)."""

    entry = _entry(QUESTION + ANSWER, boundaries=(11, 75))
    assert _offered(entry, QUESTION + ANSWER[:10]) is None
    assert _offered(entry, QUESTION[:40]) is None


def test_an_entry_that_cannot_say_where_its_prompt_ended_offers_nothing():
    """A model without recurrent state keeps no checkpoints, so its entry
    does not know where the prompt stopped and the answer began."""

    assert _offered(_entry(QUESTION + ANSWER, recurrent=False), QUESTION) is None


def test_a_prompt_that_only_shares_its_opening_is_still_read():
    entry = _entry(QUESTION + ANSWER, boundaries=(75, 76))
    other = QUESTION[:60] + [401] * 16
    assert _offered(entry, other) is None
    # And a longer question that starts with this one.
    assert _offered(entry, QUESTION + [401] * 16) is None


def test_a_recurrent_entry_without_a_checkpoint_offers_nothing():
    assert _offered(_entry(QUESTION + ANSWER), QUESTION) is None


def test_the_lane_does_not_serve_a_prompt_from_an_entry_without_checkpoints():
    model = _WatchingModel()
    rt = _toy_runtime(model)
    bank = SessionBank(max_entries=8, max_bytes=4096, per_session_max_bytes=4096)
    assert _put(bank, rt, QUESTION + ANSWER, "first") is not None

    assert _near(rt, QUESTION, bank, session_id="second") is None
    assert model.alive_per_forward == []


def _answer(rt, bank, prompt, *, turn):
    return generation.generate_mtpk(
        rt, list(prompt), max_tokens=24, sampler=NATIVE, draft_sampler=NATIVE,
        speculative_depth=3, seed=SEED + turn, mtp_cache_policy="persistent",
        mtp_history_policy="committed", verify_strategy="batched",
        stop_token_ids=set(), capture_final_state=True, session_bank=bank,
        session_id="s", session_restore_mode="reference",
        commit_prompt_state_to_bank=True,
    )


def test_a_repeat_on_the_model_writes_what_a_first_ask_writes(pack, lane):
    """The 48-token prompt of a finished turn, asked again with another seed:
    nothing is read, and the answer is the one a server that never saw the
    prompt writes with that seed."""
    rt, hidden_variant = _runtime(pack, lane)
    bank = SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32)
    question = _prompt(48)
    _one_turn(rt, bank, question, 0, True, hidden_variant)

    again = _answer(rt, bank, question, turn=3)
    assert again.stats.cached_tokens == len(question)
    assert again.stats.suffix_tokens == 0

    first_ask = _answer(
        rt, SessionBank(max_bytes=1 << 32, per_session_max_bytes=1 << 32), question, turn=3
    )
    assert first_ask.stats.cached_tokens == 0
    assert list(again.tokens) == list(first_ask.tokens)
