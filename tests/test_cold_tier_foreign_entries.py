"""SSD cold tier: entries this server can never restore go first.

A store shared across restarts keeps entries no request of the running
server can match: written by another model or draft head, or under a
server-wide setting that has since changed (a server-level component of the
session policy fingerprint). Eviction was pure LRU and ran only under space
pressure, so such entries could fill the cap while every SSD restore
missed.

Contract pinned here:

- only server-level identity makes an entry foreign: another model, another
  draft head, or a server-level fingerprint component that differs or is
  present on one side only; request-level components never do;
- an entry whose fingerprint carries none of the server-level components
  (empty, or a format the server does not know) is judged on model and
  draft head alone;
- under space pressure foreign entries are evicted before the server's own,
  each group in least-recently-used order; without an identity eviction is
  plain LRU;
- foreign entries unused for MTPLX_SSD_FOREIGN_MAX_AGE_DAYS are removed
  when the store opens, through the eviction path, so a blob another entry
  still references survives; 0 turns the removal off;
- /health stats and the reconcile summary report what was found and removed.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mtplx.cache_bank.cold_tier import (
    ServerIdentity,
    SessionBankColdTier,
    parse_policy_fingerprint,
)
from mtplx.server.openai import (
    _SERVER_LEVEL_POLICY_COMPONENTS,
    _policy_fingerprint,
    _session_bank_server_identity,
)

DAY_S = 86400.0


@pytest.fixture(autouse=True)
def _default_age_limit(monkeypatch):
    monkeypatch.delenv("MTPLX_SSD_FOREIGN_MAX_AGE_DAYS", raising=False)

MODEL = "models/own"
DRAFT = "draft-a"
SERVER_COMPONENTS = {
    "template": "tpl-a",
    "openai_bridge": "bridge-v4",
    "hidden_variant": "post_norm",
    "mtp_history_policy": "committed",
    "draft_head": DRAFT,
    "gdn_prefill": "route-b",
    "mtp_batch_route": None,
}


def fingerprint(**overrides: str | None) -> str:
    parts = {
        "template": "tpl-a",
        "thinking": "0",
        "reasoning_history": "preserve_echo",
        "openai_bridge": "bridge-v4",
        "tool_contract": "none",
        "depth": "3",
        "hidden_variant": "post_norm",
        "mtp_history_policy": "committed",
        "draft_head": DRAFT,
        "gdn_prefill": "route-b",
    }
    parts.update(overrides)
    return ";".join(f"{key}={value}" for key, value in parts.items() if value is not None)


IDENTITY = ServerIdentity(
    model_path=MODEL,
    draft_head_identity=DRAFT,
    fingerprint_components=dict(SERVER_COMPONENTS),
)


def foreign(**kwargs) -> bool:
    row = {"model_path": MODEL, "draft_head_identity": DRAFT, "policy_fingerprint": fingerprint()}
    row.update(kwargs)
    return IDENTITY.is_foreign(**row)


# --- the definition ----------------------------------------------------------


def test_own_entry_is_not_foreign():
    assert foreign() is False


def test_another_model_or_draft_head_is_foreign():
    assert foreign(model_path="models/other") is True
    assert foreign(draft_head_identity="draft-b") is True
    # No draft head recorded on either side is no evidence either way.
    assert foreign(draft_head_identity=None) is False


def test_a_differing_server_level_component_is_foreign():
    assert foreign(policy_fingerprint=fingerprint(gdn_prefill="route-a")) is True
    assert foreign(policy_fingerprint=fingerprint(template="tpl-b")) is True
    assert foreign(policy_fingerprint=fingerprint(draft_head="draft-b")) is True


def test_a_server_level_component_on_one_side_only_is_foreign():
    # Written before the server emitted the component (an older build):
    # the whole-string lookup can never match it.
    assert foreign(policy_fingerprint=fingerprint(gdn_prefill=None)) is True
    # Written by a server that has a component this one never emits.
    assert foreign(policy_fingerprint=fingerprint() + ";mtp_batch_route=lane-1") is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"thinking": "1"},
        {"reasoning_history": "scoped"},
        {"tool_contract": "soft_schema_contract:v13"},
        {"depth": "5"},
        {"cache_scope": "opencode_process_cache:v1:pid:1"},
    ],
)
def test_request_level_differences_are_not_foreign(overrides):
    assert foreign(policy_fingerprint=fingerprint(**overrides)) is False


def test_fingerprint_without_server_components_is_judged_on_model_alone():
    assert foreign(policy_fingerprint=None) is False
    assert foreign(policy_fingerprint="") is False
    assert foreign(policy_fingerprint="policy-a") is False
    assert foreign(policy_fingerprint="policy-a", model_path="models/other") is True


def test_parse_skips_parts_without_a_key():
    assert parse_policy_fingerprint("a=1;junk;b=x=y;=z") == {"a": "1", "b": "x=y"}


# --- the server's identity ---------------------------------------------------


def _server_state() -> SimpleNamespace:
    args = SimpleNamespace(
        strip_assistant_reasoning_history=False,
        adaptive_policy="none",
        online_correction_cache=False,
        online_correction_cache_min_depth=1,
        online_correction_cache_key="local_prefix",
        prompt_correction_cache=False,
        prompt_correction_cache_min_depth=2,
        online_hidden_corrector_alpha=0.0,
        online_hidden_corrector_decay=0.8,
        online_hidden_corrector_warmup=1,
        online_hidden_corrector_max_feed_depth=None,
        online_hidden_corrector_key="global",
    )
    return SimpleNamespace(
        args=args,
        template_hash="template",
        draft_head_identity="draft",
        runtime=SimpleNamespace(model_path=Path("models/example"), mtp_enabled=True),
    )


def test_server_level_components_do_not_vary_with_request_choices():
    state = _server_state()
    variants = [
        _policy_fingerprint(state, thinking_enabled=False),
        _policy_fingerprint(state, thinking_enabled=True, depth=5),
        _policy_fingerprint(state, thinking_enabled=False, generation_mode="ar"),
        _policy_fingerprint(
            state,
            thinking_enabled=True,
            tools_active=True,
            no_tools_contract_active=True,
            simple_chat_contract_active=True,
            opencode_prompt_contract_profile="compact",
            cache_scope="opencode_process_cache:v1:pid:1",
        ),
    ]
    picked = [
        {key: parse_policy_fingerprint(fp).get(key) for key in _SERVER_LEVEL_POLICY_COMPONENTS}
        for fp in variants
    ]
    assert all(item == picked[0] for item in picked)
    identity = _session_bank_server_identity(state)
    assert identity is not None
    for fp in variants:
        assert not identity.is_foreign(
            model_path="models/example",
            draft_head_identity="draft",
            policy_fingerprint=fp,
        )


def test_server_identity_changes_when_a_server_level_setting_does():
    state = _server_state()
    entry_fp = _policy_fingerprint(state, thinking_enabled=True)
    state.draft_head_identity = "other-draft"
    identity = _session_bank_server_identity(state)
    assert identity.is_foreign(
        model_path="models/example", draft_head_identity="draft", policy_fingerprint=entry_fp
    )


def test_no_runtime_means_no_identity():
    state = _server_state()
    state.runtime = None
    assert _session_bank_server_identity(state) is None


# --- eviction and removal ----------------------------------------------------


def seed_entry(
    tier: SessionBankColdTier,
    name: str,
    *,
    age_days: float,
    model_path: str = MODEL,
    policy: str | None = None,
    shared: tuple[tuple[str, bytes], ...] = (),
    blobs: int = 2,
) -> dict[str, bytes]:
    tensors = {digest: raw for digest, raw in shared}
    for i in range(blobs):
        raw = f"{name}-{i}".encode() * 64
        tensors[hashlib.sha256(raw).hexdigest()] = raw
    entry = SimpleNamespace(
        token_ids=(1, 2, 3),
        nbytes=sum(map(len, tensors.values())),
        model_path=model_path,
        mtp_enabled=False,
        draft_head_identity=DRAFT,
        policy_fingerprint=fingerprint() if policy is None else policy,
        snapshot_epoch=0,
        session_id=name,
    )
    metadata = tier._metadata_for_entry(entry, capabilities=(), payload_nbytes=entry.nbytes)
    metadata["entry_id"] = name
    metadata["entry_dir"] = f"entries/{name[:2]}/{name}"
    directory = tier.base_dir / metadata["entry_dir"]
    directory.mkdir(parents=True)
    for digest, raw in tensors.items():
        tier._write_blob(digest, raw)
    payload = {"tensor_blobs": {key: {"sha256": key} for key in tensors}}
    (directory / "payload.json").write_text(json.dumps(payload))
    tier._insert_manifest(metadata)
    with tier._connect() as conn:
        conn.execute(
            "UPDATE entries SET last_access_s = ? WHERE entry_id = ?",
            (time.time() - age_days * DAY_S, name),
        )
    tier._invalidate_disk_usage_cache()
    return tensors


def entry_ids(tier: SessionBankColdTier) -> set[str]:
    with tier._connect() as conn:
        return {str(row[0]) for row in conn.execute("SELECT entry_id FROM entries")}


def blob_exists(tier: SessionBankColdTier, digest: str) -> bool:
    return tier._blob_path(digest).exists()


def open_tier(tmp_path: Path, *, identity=IDENTITY, **kwargs) -> SessionBankColdTier:
    return SessionBankColdTier(
        base_dir=tmp_path / "session-bank",
        mode="on",
        server_identity=(lambda: identity) if identity is not None else None,
        **kwargs,
    )


def seed_mixed(tier: SessionBankColdTier) -> None:
    # Oldest first: own, foreign (other model), own (other request choice),
    # foreign (changed server setting), newest own.
    seed_entry(tier, "own-old", age_days=5)
    seed_entry(tier, "frn-model", age_days=1, model_path="models/other")
    seed_entry(tier, "own-mid", age_days=4, policy=fingerprint(thinking="1"))
    seed_entry(tier, "frn-route", age_days=2, policy=fingerprint(gdn_prefill="route-a"))
    seed_entry(tier, "own-new", age_days=0.5)


def entry_bytes(tier: SessionBankColdTier, name: str) -> int:
    with tier._connect() as conn:
        row = conn.execute(
            "SELECT physical_nbytes FROM entries WHERE entry_id = ?", (name,)
        ).fetchone()
    return int(row[0])


def test_eviction_takes_foreign_entries_before_own_ones(tmp_path):
    tier = open_tier(tmp_path)
    try:
        seed_mixed(tier)
        total = tier._current_bytes()
        # Room for all but two entries: both foreign ones go, although
        # three own entries are older than one of them.
        cap = total - entry_bytes(tier, "frn-model") - entry_bytes(tier, "frn-route")
        assert tier._evict_until_room(0, cap_bytes=cap) is True
        assert entry_ids(tier) == {"own-old", "own-mid", "own-new"}
        stats = tier.stats()
        assert stats["foreign_entries_evicted"] == 2
        assert stats["entries_evicted"] == 2
        assert stats["foreign_entries"] == 0
    finally:
        tier.close()


def test_own_entries_keep_lru_order_after_the_foreign_ones(tmp_path):
    tier = open_tier(tmp_path)
    try:
        seed_mixed(tier)
        cap = (
            tier._current_bytes()
            - entry_bytes(tier, "frn-model")
            - entry_bytes(tier, "frn-route")
            - entry_bytes(tier, "own-old")
        )
        assert tier._evict_until_room(0, cap_bytes=cap) is True
        # Both foreign entries, then the oldest own one. own-mid differs
        # only in a request-level choice (thinking), so it is not foreign
        # and keeps its LRU place behind own-old.
        assert entry_ids(tier) == {"own-mid", "own-new"}
        stats = tier.stats()
        assert stats["foreign_entries_evicted"] == 2
        assert stats["entries_evicted"] == 3
    finally:
        tier.close()


def test_without_an_identity_eviction_is_plain_lru(tmp_path):
    tier = open_tier(tmp_path, identity=None)
    try:
        seed_mixed(tier)
        cap = tier._current_bytes() - entry_bytes(tier, "own-old") - 1
        assert tier._evict_until_room(0, cap_bytes=cap) is True
        # The two oldest go, whatever they are.
        assert entry_ids(tier) == {"frn-model", "frn-route", "own-new"}
        stats = tier.stats()
        assert stats["foreign_entries_evicted"] == 0
        assert stats["foreign_entries"] is None
    finally:
        tier.close()


def test_age_limit_removes_only_old_foreign_entries(tmp_path):
    tier = open_tier(tmp_path)
    try:
        seed_entry(tier, "own-ancient", age_days=30)
        seed_entry(tier, "frn-old", age_days=8, model_path="models/other")
        seed_entry(tier, "frn-route-old", age_days=9, policy=fingerprint(gdn_prefill=None))
        seed_entry(tier, "frn-recent", age_days=6, model_path="models/other")
        expected_bytes = entry_bytes(tier, "frn-old") + entry_bytes(tier, "frn-route-old")
        result = tier._expire_foreign_entries()
        assert result is not None
        assert result["entries"] == 2
        assert result["bytes"] == expected_bytes
        assert entry_ids(tier) == {"own-ancient", "frn-recent"}
        stats = tier.stats()
        assert stats["foreign_entries_expired"] == 2
        assert stats["foreign_bytes_expired"] == expected_bytes
        assert stats["foreign_entries"] == 1
        assert stats["foreign_max_age_days"] == 7.0
        assert stats["last_foreign_expiry"]["entries"] == 2
    finally:
        tier.close()


def test_age_limit_zero_disables_the_removal(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_FOREIGN_MAX_AGE_DAYS", "0")
    tier = open_tier(tmp_path)
    try:
        seed_entry(tier, "frn-old", age_days=100, model_path="models/other")
        assert tier._expire_foreign_entries() is None
        assert entry_ids(tier) == {"frn-old"}
        assert tier.stats()["foreign_max_age_days"] == 0.0
    finally:
        tier.close()


def test_custom_age_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("MTPLX_SSD_FOREIGN_MAX_AGE_DAYS", "2")
    tier = open_tier(tmp_path)
    try:
        seed_entry(tier, "frn-3d", age_days=3, model_path="models/other")
        seed_entry(tier, "frn-1d", age_days=1, model_path="models/other")
        tier._expire_foreign_entries()
        assert entry_ids(tier) == {"frn-1d"}
    finally:
        tier.close()


def test_shared_blob_survives_while_an_own_entry_references_it(tmp_path):
    tier = open_tier(tmp_path)
    try:
        shared_raw = b"shared-prefix-block" * 64
        shared = ((hashlib.sha256(shared_raw).hexdigest(), shared_raw),)
        foreign_tensors = seed_entry(
            tier, "frn-old", age_days=10, model_path="models/other", shared=shared
        )
        seed_entry(tier, "own-new", age_days=0, shared=shared)
        tier._expire_foreign_entries()
        assert entry_ids(tier) == {"own-new"}
        shared_digest = shared[0][0]
        assert blob_exists(tier, shared_digest)
        for digest in foreign_tensors:
            if digest != shared_digest:
                assert not blob_exists(tier, digest)
    finally:
        tier.close()


def test_entry_used_again_meanwhile_is_kept(tmp_path, monkeypatch):
    # Another server sharing the directory touches the entry between the
    # selection and the removal: recency is re-read under the store lock.
    tier = open_tier(tmp_path)
    try:
        seed_entry(tier, "frn-old", age_days=10, model_path="models/other")
        original = tier._pause_for_foreground

        def touch_then_pause():
            tier._touch_entry("frn-old")
            original()

        monkeypatch.setattr(tier, "_pause_for_foreground", touch_then_pause)
        result = tier._expire_foreign_entries()
        assert result["entries"] == 0
        assert entry_ids(tier) == {"frn-old"}
    finally:
        tier.close()


def test_legacy_entries_without_server_components_stay_on_lru_footing(tmp_path):
    # Fingerprints that carry no server-level component (an unknown format)
    # give no evidence; only the model decides.
    tier = open_tier(tmp_path)
    try:
        seed_entry(tier, "legacy-own", age_days=30, policy="policy-a")
        seed_entry(tier, "legacy-other-model", age_days=30, policy="policy-a", model_path="m/x")
        tier._expire_foreign_entries()
        assert entry_ids(tier) == {"legacy-own"}
    finally:
        tier.close()


def test_store_open_removes_old_foreign_entries_and_reports_them(tmp_path):
    seeder = open_tier(tmp_path, identity=None)
    try:
        seed_entry(seeder, "own-old", age_days=20)
        seed_entry(seeder, "frn-old", age_days=20, policy=fingerprint(gdn_prefill="route-a"))
        removed_bytes = entry_bytes(seeder, "frn-old")
    finally:
        seeder.close()
    summaries: list[dict] = []
    tier = open_tier(tmp_path, reconcile_listener=summaries.append)
    try:
        deadline = time.time() + 10
        while not summaries and time.time() < deadline:
            time.sleep(0.01)
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary["foreign_entries_expired"] == 1
        assert summary["foreign_bytes_expired"] == removed_bytes
        assert summary["foreign_max_age_days"] == 7.0
        assert summary["entries"] == 1
        assert entry_ids(tier) == {"own-old"}
        assert tier.stats()["startup_reconcile"]["foreign_entries_expired"] == 1
    finally:
        tier.close()


def test_periodic_pass_runs_at_most_once_per_interval(tmp_path, monkeypatch):
    tier = open_tier(tmp_path)
    calls = []
    try:
        monkeypatch.setattr(
            tier, "_expire_foreign_entries", lambda **_: calls.append(1) or None
        )
        tier._maybe_expire_foreign_entries()
        tier._last_foreign_expiry_monotonic = time.monotonic()
        tier._maybe_expire_foreign_entries()
        assert calls == [1]
    finally:
        tier.close()


def test_failing_identity_provider_classifies_nothing(tmp_path):
    def broken():
        raise AttributeError("state not ready")

    tier = SessionBankColdTier(
        base_dir=tmp_path / "session-bank", mode="on", server_identity=broken
    )
    try:
        seed_entry(tier, "frn-old", age_days=30, model_path="models/other")
        assert tier._expire_foreign_entries() is None
        assert entry_ids(tier) == {"frn-old"}
    finally:
        tier.close()
