"""Flash-Next entries saved before 2.12.3 are never restored by this build (#635).

Until 2.12.3 a restore from the SSD tier below a saved entry's end kept that
entry's pooled index table and block count, and the next commit saved the
table again, so a saved Flash-Next conversation can hold another
conversation's block keys. Nothing in a saved row tells a clean row from
such a one, so Flash-Next cache identity carries a pooled-index version:
rows saved before it are not served, and the cold tier's least recently used
eviction reclaims them as this build writes its own. No other model family's
identity changes, and the tier's on-disk format version stays, so nothing is
archived or rewritten when the store opens.
"""

from __future__ import annotations

from test_qsa_ssd_prefix_restore import (  # noqa: F401  (fixtures)
    DONOR_IDS,
    SHARED_IDS,
    TEMPLATE,
    _bank,
    _candidates,
    _Runtime,
    _stored_conversation,
    _tier,
    attn,
    rows,
)


def _fingerprint(monkeypatch, family: str) -> str:
    """The server's cache identity for a request on a model of ``family``."""

    from test_bank_history_policy import _state

    from mtplx.server import openai

    monkeypatch.setattr(openai, "_model_family_for_state", lambda _state: family)
    return openai._policy_fingerprint(_state(mtp_enabled=True), thinking_enabled=True)


def test_flash_next_entries_carry_the_pooled_index_version(monkeypatch):
    assert "qsa_pooled_index=v2" in _fingerprint(monkeypatch, "qwen4_exp").split(";")
    # Every other family keeps the identity its stored entries already have.
    assert "qsa_pooled_index" not in _fingerprint(monkeypatch, "qwen3_next")


def test_an_entry_stored_before_the_fix_is_never_served_and_eviction_reclaims_it(
    tmp_path, attn, rows, monkeypatch,
):
    current = _fingerprint(monkeypatch, "qwen4_exp")
    # The same request's identity on 2.12.2: every part but the new one.
    before = ";".join(part for part in current.split(";") if part != "qsa_pooled_index=v2")
    assert before != current
    tier = _tier(tmp_path)
    try:
        bank = _bank(tier)
        _stored_conversation(attn, rows, bank, tier, policy=before)
        assert tier.stats()["entries"] == 1
        # The row is whole: a request with its old identity would be served.
        assert _candidates(bank, policy=before)
        bank.clear()
        # This build's lookups never reach it, by either restore lane, and
        # nothing fails on the way: the turn prefills cold.
        assert _candidates(bank, policy=current) == []
        assert (
            bank.restore(
                _Runtime(), SHARED_IDS + DONOR_IDS + [1],
                template_hash=TEMPLATE, policy_fingerprint=current,
            )
            is None
        )
        assert tier.stats().get("restore_failures", 0) == 0
        # Eviction is least recently used first, whatever an entry's
        # identity: the first entries this build writes past the store's cap
        # retire the old one.
        tier.max_bytes = int(tier.stats()["bytes"]) + 1
        _stored_conversation(attn, rows, bank, tier, policy=current)
        assert tier.stats()["entries"] == 1
        assert _candidates(bank, policy=before) == []
        assert _candidates(bank, policy=current)
    finally:
        tier.close()
