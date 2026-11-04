"""The session bank names the MTP-history policy generation actually runs.

``MTPLX_MTP_HISTORY_POLICY`` (or ``mtplx serve --mtp-history-policy``) is the
documented way around the committed-history verify stall on M1 and M2 GPUs
(#400: 4 to 5 tok/s with ``committed``, 29 to 33 with ``cycle``).
``restore_or_prefill_prompt_state`` resolves that override over the requested
``committed``, so its own lookups and prompt store used ``cycle``, while every
store and lookup the server makes (the generation-final save, the block-prefix
save, the near-prefix probe, the fingerprint) kept saying ``committed``. The
bank refuses an entry whose policy does not match the lookup's, so with the
override on, the next turn never found the conversation it had just saved and
read the whole prompt again (an M2 Ultra in #422: every turn ``cache_source:
none``, first token 166 s at 76K tokens and 292 s at 108K).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mtplx.generation import _resolve_mtp_history_policy
from mtplx.server.openai import (
    _bank_history_policy,
    _generation_final_bank_metadata,
    _policy_fingerprint,
)
from mtplx.session_bank import _mtp_history_policy_compatible

ENV = "MTPLX_MTP_HISTORY_POLICY"
CYCLE_FAMILY = ("cycle", "none", "off")
COMMITTED_FAMILY = (None, "committed", "full", "last_window", "window", "auto")


def _state(*, mtp_enabled: bool = True) -> SimpleNamespace:
    args = SimpleNamespace(
        strip_assistant_reasoning_history=False,
        adaptive_policy="none",
        online_correction_cache=False,
        online_correction_cache_min_depth=1,
        online_correction_cache_key="local_prefix",
        prompt_correction_cache=False,
        prompt_correction_cache_min_depth=2,
        online_hidden_corrector_alpha=0.25,
        online_hidden_corrector_decay=0.7,
        online_hidden_corrector_warmup=2,
        online_hidden_corrector_max_feed_depth=2,
        online_hidden_corrector_key="token",
    )
    return SimpleNamespace(
        args=args,
        template_hash="template",
        draft_head_identity="draft",
        runtime=SimpleNamespace(mtp_enabled=mtp_enabled),
    )


def _set_override(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)


@pytest.mark.parametrize("override", CYCLE_FAMILY + COMMITTED_FAMILY)
@pytest.mark.parametrize("prompt_tokens", [512, 20_000])
def test_the_bank_identity_is_the_policy_generation_resolves(
    monkeypatch: pytest.MonkeyPatch, override: str | None, prompt_tokens: int
) -> None:
    _set_override(monkeypatch, override)
    banked = _bank_history_policy(_state())
    # The server hands generation this same string as the requested policy;
    # generation applies the operator's override on top of it.
    resolved = _resolve_mtp_history_policy(banked, prompt_tokens)
    assert _mtp_history_policy_compatible(banked, resolved), (override, banked, resolved)
    assert _mtp_history_policy_compatible(resolved, banked), (override, banked, resolved)


@pytest.mark.parametrize("override", CYCLE_FAMILY)
def test_a_generation_final_save_under_cycle_is_found_by_the_next_turn(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    _set_override(monkeypatch, override)
    state = _state()
    # Under cycle there is no committed draft-head history to save.
    saved = _generation_final_bank_metadata(
        state, SimpleNamespace(final_committed_mtp_cache=None), token_count=40_000
    )
    next_turn_lookup = _resolve_mtp_history_policy(_bank_history_policy(state), 40_100)
    assert next_turn_lookup == "cycle"
    assert saved["mtp_history_policy"] == "cycle"
    assert _mtp_history_policy_compatible(saved["mtp_history_policy"], next_turn_lookup)
    assert "mtp_history_policy=cycle" in _policy_fingerprint(state, thinking_enabled=True)


@pytest.mark.parametrize("override", COMMITTED_FAMILY)
def test_without_a_cycle_override_nothing_changes(
    monkeypatch: pytest.MonkeyPatch, override: str | None
) -> None:
    """Every saved conversation keeps its identity unless the operator asked
    for cycle: the label and the fingerprint are the committed ones."""
    monkeypatch.delenv(ENV, raising=False)
    baseline = _policy_fingerprint(_state(), thinking_enabled=True)
    _set_override(monkeypatch, override)
    assert _bank_history_policy(_state()) == "committed"
    assert _policy_fingerprint(_state(), thinking_enabled=True) == baseline
    assert "mtp_history_policy=committed" in baseline


@pytest.mark.parametrize("override", CYCLE_FAMILY + COMMITTED_FAMILY)
def test_an_ar_only_runtime_banks_under_cycle_whatever_the_override(
    monkeypatch: pytest.MonkeyPatch, override: str | None
) -> None:
    _set_override(monkeypatch, override)
    assert _bank_history_policy(_state(mtp_enabled=False)) == "cycle"
