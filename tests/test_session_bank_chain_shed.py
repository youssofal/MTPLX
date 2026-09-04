"""#447: SessionBank.shrink_for_admission — escalating eviction for the
admission shed. Phase 1 walks non-terminal chain entries (every session
keeps its highest-prefix entry); phase 2, only if the deficit stands,
takes remaining entries in take-anything order with active sessions last.
Both phases spare the entry the imminent prompt restores from.

Pure host tests -- synthetic entries, ``cache=[]`` plus ``nbytes_override``,
no MLX, no model, no Metal.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mtplx.session_bank import SessionBank


RUNTIME = SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True)


def _bank(**kwargs) -> SessionBank:
    defaults = dict(max_entries=16, max_bytes=10_000, per_session_max_bytes=10_000)
    defaults.update(kwargs)
    return SessionBank(**defaults)


def _put(bank: SessionBank, tokens, *, session_id: str, nbytes: int):
    return bank.put(
        runtime=RUNTIME,
        token_ids=list(tokens),
        cache=[],
        logits=None,
        hidden=None,
        session_id=session_id,
        nbytes_override=nbytes,
    )


def _keys(bank: SessionBank):
    return set(bank._entries.keys())


def test_phase_one_evicts_the_sibling_and_stops():
    bank = _bank()
    _put(bank, (1, 2, 9, 9), session_id="a", nbytes=80)      # sibling fork
    _put(bank, (1, 2, 3, 4, 5), session_id="a", nbytes=100)  # terminal of a
    _put(bank, (7, 7, 7), session_id="b", nbytes=50)         # terminal of b
    assert bank.shrink_for_admission(150) == (1, 0)
    assert _keys(bank) == {(1, 2, 3, 4, 5), (7, 7, 7)}


def test_phase_two_takes_terminals_when_siblings_are_not_enough():
    bank = _bank()
    _put(bank, (1, 2, 9, 9), session_id="a", nbytes=80)
    _put(bank, (1, 2, 3, 4, 5), session_id="a", nbytes=100)
    _put(bank, (7, 7, 7), session_id="b", nbytes=50)
    non_terminal, terminal = bank.shrink_for_admission(
        0, protect_tokens=(1, 2, 3, 4, 5, 6)
    )
    assert non_terminal == 1
    assert terminal == 1
    # The imminent prompt's restore source survives both phases.
    assert _keys(bank) == {(1, 2, 3, 4, 5)}


def test_protect_tokens_shields_the_restore_source():
    bank = _bank()
    _put(bank, (1, 2, 9, 9), session_id="a", nbytes=80)
    _put(bank, (1, 2, 3, 4, 5), session_id="a", nbytes=100)
    non_terminal, terminal = bank.shrink_for_admission(
        0, protect_tokens=(1, 2, 9, 9, 10)
    )
    # The shorter fork is what this prompt restores from: phase 1 skips it,
    # phase 2 takes the session's terminal instead.
    assert (non_terminal, terminal) == (0, 1)
    assert _keys(bank) == {(1, 2, 9, 9)}


def test_phase_two_takes_active_sessions_last():
    import time

    bank = _bank()
    _put(bank, (1, 2, 3), session_id="a", nbytes=100)
    _put(bank, (7, 7), session_id="b", nbytes=100)
    # put() stamps the active pin for both; age b out of the TTL so only
    # a is active when phase 2 orders its victims.
    bank._session_last_active["b"] = time.monotonic() - 100_000
    non_terminal, terminal = bank.shrink_for_admission(100)
    assert (non_terminal, terminal) == (0, 1)
    assert _keys(bank) == {(1, 2, 3)}


def test_lru_order_among_siblings():
    bank = _bank()
    _put(bank, (1, 9), session_id="a", nbytes=60)            # oldest sibling
    _put(bank, (1, 8, 8), session_id="a", nbytes=60)         # newer sibling
    _put(bank, (1, 2, 3, 4, 5, 6), session_id="a", nbytes=100)
    assert bank.shrink_for_admission(160) == (1, 0)
    assert (1, 9) not in _keys(bank)
    assert (1, 8, 8) in _keys(bank)


def test_target_already_met_is_a_no_op():
    bank = _bank()
    _put(bank, (1, 9), session_id="a", nbytes=60)
    _put(bank, (1, 2, 3), session_id="a", nbytes=100)
    assert bank.shrink_for_admission(1_000) == (0, 0)
    assert len(bank._entries) == 2


def test_eviction_reason_is_recorded():
    bank = _bank()
    _put(bank, (1, 9), session_id="a", nbytes=60)
    _put(bank, (1, 2, 3), session_id="a", nbytes=100)
    bank.shrink_for_admission(100, reason="prefill_admission_chain")
    assert bank.eviction_log[-1]["reason"] == "prefill_admission_chain"


def test_live_referenced_entries_are_never_victims():
    bank = _bank()
    fork = _put(bank, (1, 2, 9, 9), session_id="a", nbytes=80)
    _put(bank, (1, 2, 3, 4, 5), session_id="a", nbytes=100)
    # The fork's arrays are the live session's own state (a live-ref lease):
    # walking it frees nothing and costs the running session its state.
    fork.cache_ref = object()
    assert bank.shrink_for_admission(0) == (0, 1)
    assert (1, 2, 9, 9) in _keys(bank)
    assert (1, 2, 3, 4, 5) not in _keys(bank)
