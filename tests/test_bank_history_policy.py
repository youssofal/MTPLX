"""#465: one MTP-history policy per runtime, used by every bank store, lookup
and the policy fingerprint.

A target-only AR runtime (``--no-load-mtp``) banked its postcommit prefix
under ``cycle`` while the lookups, the prefill store and the fingerprint
said ``committed``. The bank compares the two strings, so the longest entry
(the postcommit one, which the next turn matches) was refused with
``policy_mismatch`` and a Hermes/Pi session re-prefilled its whole 14.5k
prompt on every top-level turn (~2 minutes on an M1 Max).
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from mtplx.server.openai import _bank_history_policy, _policy_fingerprint
from mtplx.session_bank import _mtp_history_policy_compatible


def _state(*, mtp_enabled: bool | None) -> SimpleNamespace:
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
    state = SimpleNamespace(args=args, template_hash="template", draft_head_identity="draft")
    if mtp_enabled is not None:
        state.runtime = SimpleNamespace(mtp_enabled=mtp_enabled)
    return state


def test_mtp_runtime_banks_under_committed_and_ar_only_runtime_under_cycle():
    assert _bank_history_policy(_state(mtp_enabled=True)) == "committed"
    assert _bank_history_policy(_state(mtp_enabled=False)) == "cycle"
    # A state without a runtime (unit fixtures) keeps the MTP default.
    assert _bank_history_policy(_state(mtp_enabled=None)) == "committed"


def test_fingerprint_carries_the_runtime_policy():
    mtp = _policy_fingerprint(_state(mtp_enabled=True), thinking_enabled=True)
    ar_only = _policy_fingerprint(_state(mtp_enabled=False), thinking_enabled=True)
    assert "mtp_history_policy=committed" in mtp
    assert "mtp_history_policy=cycle" in ar_only
    assert mtp.replace("mtp_history_policy=committed", "") == ar_only.replace(
        "mtp_history_policy=cycle", ""
    )


def test_ar_only_postcommit_entry_is_compatible_with_the_ar_only_lookup():
    policy = _bank_history_policy(_state(mtp_enabled=False))
    # The postcommit store always banked AR-only runtimes under "cycle"; the
    # lookup now asks for the same string, so the bank's identity gate passes.
    assert _mtp_history_policy_compatible("cycle", policy)
    # The old lookup literal was the mismatch the reporter saw.
    assert not _mtp_history_policy_compatible("cycle", "committed")


def test_server_has_no_hardcoded_history_policy_literal():
    source = Path(__file__).resolve().parents[1].joinpath(
        "mtplx", "server", "openai.py"
    ).read_text()
    literal_stores = re.findall(r'mtp_history_policy="committed"', source)
    literal_fingerprint = re.findall(r'"mtp_history_policy=committed"', source)
    assert literal_stores == [], "store/lookup sites must derive from _bank_history_policy"
    assert literal_fingerprint == [], "the fingerprint must derive from _bank_history_policy"
