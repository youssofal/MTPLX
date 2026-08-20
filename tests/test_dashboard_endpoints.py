"""Tests for the MTPLX dashboard HTTP surface.

These tests intentionally use the existing ``_fake_state`` helper from
``test_server_openai`` so they exercise the same FastAPI app the server
actually builds. They cover:

- The new ``/v1/mtplx/...`` endpoints (snapshot, prefill_history,
  settings, cancel, metrics/stream).
- ``/health`` additions (``machine_model``, ``unified_memory_bytes``).
- The dashboard static mount (or HTML fallback when the bundle is missing).
- Regression: ``GET /`` chat UI route is untouched.
- Regression: ``PUBLIC_MTPLX_STATS_KEYS`` keeps every previously exposed key.
"""

from __future__ import annotations

import sys
import time
from threading import Event, Thread
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from mtplx.dashboard import has_static_bundle
from mtplx.benchmarks.runners.aime import AIMEProblem
from mtplx.server import openai
from mtplx.server.dashboard_state import (
    InFlightHandle,
    ProgressEventGate,
    RollingMetrics,
)
from mtplx.server.openai import (
    DASHBOARD_MUTABLE_SETTINGS_KEYS,
    DASHBOARD_READ_ONLY_SETTINGS_KEYS,
    DASHBOARD_RESTART_REQUIRED_KEYS,
    DASHBOARD_SNAPSHOT_INTERVAL_DEFAULT_MS,
    DASHBOARD_SNAPSHOT_INTERVAL_IDLE_MIN_MS,
    DASHBOARD_SNAPSHOT_INTERVAL_MAX_MS,
    DASHBOARD_SNAPSHOT_INTERVAL_MIN_MS,
    PUBLIC_MTPLX_STATS_KEYS,
    _dashboard_snapshot_interval_for_activity_s,
    _dashboard_snapshot_interval_s,
    create_app,
)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from test_server_openai import _fake_state  # noqa: E402


# ---- regression: existing public surface ----------------------------------


def test_public_stats_keys_keep_previously_exposed_keys():
    """If this assertion fires, a contributor has removed a key from the
    public whitelist that downstream clients depend on. Adding keys is OK;
    removing them is a breaking change."""

    must_keep = {
        "mode",
        "generation_mode",
        "generated_tokens",
        "prompt_tokens",
        "completion_tokens",
        "elapsed_s",
        "tok_s",
        "prompt_eval_time_s",
        "prefill_tok_s",
        "ttft_s",
        "decode_elapsed_s",
        "request_elapsed_s",
        "request_tok_s",
        "decode_tok_s",
        "sliding_decode_tok_s_first_32",
        "sliding_decode_tok_s_first_64",
        "sliding_decode_tok_s_first_128",
        "sliding_decode_tok_s_first_256",
        "sliding_decode_tok_s_last_32",
        "sliding_decode_tok_s_last_64",
        "sliding_decode_tok_s_last_128",
        "sliding_decode_tok_s_last_256",
        "accepted_drafts",
        "rejected_drafts",
        "drafted_tokens",
        "verify_calls",
        "accepted_by_depth",
        "drafted_by_depth",
        "mean_accept_probability_by_depth",
        "correction_tokens",
        "bonus_tokens",
        "verify_time_s",
        "draft_time_s",
        "accept_time_s",
        "repair_time_s",
        "session_cache_hit",
        "cached_tokens",
        "new_prefill_tokens",
        "cache_miss_reason",
        "session_restore_mode",
        "session_id",
        "context_len",
        "lock_wait_time_s",
        "request_max_tokens",
        "server_max_response_tokens",
        "effective_max_tokens",
        "remaining_context_tokens",
        "server_cap_applied",
        "context_cap_applied",
        "server_elapsed_s",
        "server_tok_s",
        "server_seed",
        "server_attempts",
        "server_blank_retries",
        "server_blank_retry_suppressed",
        "mtp_depth",
        "speculative_depth",
        "peak_memory_bytes",
        "reasoning_reentries",
        "reasoning_tokens",
        "answer_tokens",
        "paged_active_array_time_s",
        "paged_kv_quant_dequant_calls",
        "paged_kv_quant_dequant_time_s",
        "paged_kv_quant_dequant_tokens",
    }
    missing = must_keep - set(PUBLIC_MTPLX_STATS_KEYS)
    assert not missing, f"public stats keys regressed; lost: {sorted(missing)}"


def test_public_stats_keys_expose_mtp_batch_execution_truth():
    assert {
        "mtp_batch_real_width",
        "mtp_batch_fixed_width",
        "mtp_batch_route_id",
        "target_verify_cycles",
        "mtp_batch_session_cache_bypass",
    } <= set(PUBLIC_MTPLX_STATS_KEYS)


def test_public_stats_keys_includes_verify_decomposition():
    new_keys = {
        "target_forward_time_s",
        "verify_forward_time_s",
        "verify_eval_time_s",
        "verify_logits_eval_time_s",
        "verify_hidden_eval_time_s",
        "verify_target_distribution_time_s",
        "verify_eval_unattributed_time_s",
        "mtp_history_policy",
        "mtp_history_window_tokens",
        "mtp_history_position_base",
        "snapshot_time_s",
        "commit_time_s",
        "capture_commit_time_s",
        "rollback_time_s",
        "graphbank",
        "repair_time_by_reject_depth_s",
        "mtp_history_materialize_every",
        "mtp_history_materialize_events",
        "clear_cache_every",
        "clear_cache_events",
        "clear_cache_time_s",
        "trunk_cache_materialize_every",
        "trunk_cache_materialize_events",
        "trunk_cache_materialize_time_s",
        "dirty_detach_components",
        "dirty_detach_mode",
        "dirty_detach_gdn_every",
        "dirty_detach_conv_every",
        "dirty_detach_attn_every",
        "dirty_detach_events",
        "dirty_detach_time_s",
        "dirty_detach_arrays",
        "dirty_detach_bytes",
        "live_output_detach_enabled",
        "live_output_detach_mode",
        "live_output_detach_events",
        "live_output_detach_time_s",
        "live_output_detach_arrays",
        "live_output_detach_bytes",
        "state_rebase_every",
        "state_rebase_events",
        "state_rebase_time_s",
        "state_root_eval_enabled",
        "state_root_eval_include_mtp",
        "state_root_eval_events",
        "state_root_eval_time_s",
        "state_root_eval_arrays",
        "capture_commit_detach_components",
        "capture_commit_detach_mode",
        "capture_commit_detach_gdn_every",
        "capture_commit_detach_conv_every",
        "capture_commit_detach_events",
        "capture_commit_detach_time_s",
        "capture_commit_detach_arrays",
        "capture_commit_detach_bytes",
        "trace_accounting_time_s",
        "dashboard_progress_published_events",
        "dashboard_progress_throttled_events",
        "dashboard_progress_last_completion_tokens",
        "dashboard_progress_decision_time_s",
        "dashboard_progress_registry_update_time_s",
        "dashboard_progress_rolling_update_time_s",
        "dashboard_progress_bus_publish_time_s",
    }
    missing = new_keys - set(PUBLIC_MTPLX_STATS_KEYS)
    assert not missing, f"verify-decomposition keys not whitelisted: {sorted(missing)}"


# ---- /health additions ----------------------------------------------------


def test_health_exposes_chip_machine_model_and_unified_memory_bytes():
    client = TestClient(create_app(_fake_state()))
    payload = client.get("/health").json()
    assert "chip" in payload
    assert "machine_model" in payload
    assert "unified_memory_bytes" in payload
    # On macOS dev workstations chip/machine_model are non-empty strings; on
    # CI runners these fields may be ``None`` — both shapes are acceptable.
    assert payload["chip"] is None or isinstance(payload["chip"], str)
    assert payload["machine_model"] is None or isinstance(payload["machine_model"], str)
    assert payload["unified_memory_bytes"] is None or isinstance(
        payload["unified_memory_bytes"], int
    )
    assert payload["scheduler"]["mode"] == "serial"
    assert payload["scheduler"]["preset"] == "latency"
    assert payload["scheduler"]["config"]["max_active_requests"] == 1
    assert payload["scheduler"]["config"]["decode_batch_max"] == 1
    assert payload["scheduler"]["config"]["batch_wait_ms"] == 0.0
    assert payload["scheduler"]["scheduler_policy"] == "solo_mtp_oracle"
    assert payload["scheduler"]["path_a"]["solo_mtp_protected"] is True


def test_health_counts_dashboard_in_flight_requests():
    state = _fake_state()
    state.dashboard.in_flight.register(
        InFlightHandle(
            request_id="chatcmpl-active",
            cancel_event=Event(),
            started_s=time.time(),
            session_id="session-active",
            model="mtplx-test-model",
            prompt_preview="still working",
            prompt_tokens=12,
        )
    )
    client = TestClient(create_app(state))

    payload = client.get("/health").json()

    assert payload["foreground_active"] == 0
    assert payload["dashboard_active_requests"] == 1
    assert payload["active_requests"] == 1


# ---- /v1/mtplx/snapshot ---------------------------------------------------


def test_dashboard_snapshot_returns_expected_shape():
    client = TestClient(create_app(_fake_state()))
    payload = client.get("/v1/mtplx/snapshot").json()
    for required in (
        "ts",
        "model_id",
        "profile",
        "context_window",
        "active_requests",
        "in_flight",
        "latest",
        "recent",
        "rolling",
        "lifetime",
        "sessions",
        "session_bank",
        "mem",
        "settings",
        "scheduler",
        "machine",
        "uptime_s",
    ):
        assert required in payload, f"missing snapshot key: {required}"
    assert payload["model_id"] == "mtplx-test-model"
    assert payload["context_window"] == 4096
    assert payload["active_requests"] == 0
    assert isinstance(payload["in_flight"], list)
    assert isinstance(payload["rolling"], dict)
    assert payload["lifetime"]["requests_total"] == 0
    assert set(payload["settings"].keys()) >= set(DASHBOARD_MUTABLE_SETTINGS_KEYS)
    assert "chip" in payload["machine"]
    assert payload["scheduler"]["active_lane"] == "solo_mtp"
    assert payload["scheduler"]["path"] == "path_a"
    assert payload["scheduler"]["scheduler_policy"] == "solo_mtp_oracle"


# ---- /v1/mtplx/benchmarks/aime/start --------------------------------------


def test_aime_start_question_limit_uses_bounded_problem_slice(monkeypatch):
    from mtplx.benchmarks.runners import aime as aime_runner

    problems = [
        AIMEProblem(
            id=f"2026-I-{i}",
            set="AIME I",
            year=2026,
            index=i,
            problem=f"Q{i}",
            answer=i,
            source="test",
        )
        for i in range(1, 6)
    ]
    captured: dict[str, object] = {}

    monkeypatch.setattr(aime_runner, "load_dataset", lambda: problems)

    async def fake_start_run(*, year, problems=None, **kwargs):
        captured["year"] = year
        captured["problems"] = problems
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            run_id="aime-2026-quick-test",
            total=len(problems or []),
            model_id="mtplx-test-model",
            year=year,
            state=SimpleNamespace(value="running"),
            snapshot=lambda: {"started_at": "2026-06-06T10:15:00Z"},
        )

    monkeypatch.setattr(aime_runner, "start_run", fake_start_run)

    client = TestClient(create_app(_fake_state()))
    response = client.post(
        "/v1/mtplx/benchmarks/aime/start",
        json={"year": 2026, "question_limit": 3},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 3
    assert captured["year"] == 2026
    assert [p.id for p in captured["problems"]] == [
        "2026-I-1",
        "2026-I-2",
        "2026-I-3",
    ]


def test_aime_start_rejects_invalid_question_limit():
    client = TestClient(create_app(_fake_state()))
    response = client.post(
        "/v1/mtplx/benchmarks/aime/start",
        json={"year": 2026, "question_limit": 0},
    )

    assert response.status_code == 400
    assert "question_limit must be between 1 and 30" in response.text


# ---- /v1/mtplx/prefill_history --------------------------------------------


def test_prefill_history_returns_capacity_and_history():
    state = _fake_state()
    state.dashboard.prefill_history.append(
        {"t": time.time(), "prefill_tok_s": 800.0, "ttft_s": 0.1}
    )
    client = TestClient(create_app(state))
    payload = client.get("/v1/mtplx/prefill_history").json()
    assert payload["capacity"] == state.dashboard.prefill_history.capacity()
    assert len(payload["history"]) == 1
    assert payload["history"][0]["prefill_tok_s"] == 800.0


# ---- /v1/mtplx/settings (mutate + restart-required guard) -----------------


def test_settings_post_mutates_mutable_keys():
    state = _fake_state()
    client = TestClient(create_app(state))
    response = client.post(
        "/v1/mtplx/settings",
        json={
            "generation_mode": "ar",
            "depth": 2,
            "temperature": 0.5,
            "top_k": 40,
            "prefill_chunk_tokens": 4096,
        },
    )
    assert response.status_code == 200
    body = response.json()
    # v0.3.7 release wraps the settings endpoint with reasoning metadata;
    # the dashboard-applied keys land under `applied`.
    assert body.get("applied") == {
        "generation_mode": "ar",
        "depth": 2,
        "temperature": 0.5,
        "top_k": 40,
        "prefill_chunk_tokens": 4096,
    }
    assert state.args.generation_mode == "ar"
    assert state.args.depth == 2
    assert state.args.temperature == 0.5
    assert state.args.top_k == 40
    assert state.args.prefill_chunk_tokens == 4096
    assert body["prefill_chunk_tokens"] == 4096
    assert body["generation_mode"] == "ar"


def test_settings_post_rejects_restart_required_keys():
    client = TestClient(create_app(_fake_state()))
    response = client.post("/v1/mtplx/settings", json={"profile": "safe"})
    # Structured HTTPException details ride as error.detail and the
    # human message is the detail's own message (QA-105) — clients no
    # longer parse a repr out of error.message.
    assert response.status_code == 400
    error = response.json()["error"]
    assert "profile" in error["message"]
    assert error["detail"]["error"] == "restart_required"
    assert "profile" in error["detail"]["keys"]


def test_settings_post_rejects_unknown_keys():
    client = TestClient(create_app(_fake_state()))
    response = client.post("/v1/mtplx/settings", json={"banana": 7})
    assert response.status_code == 400
    message = response.json()["error"]["message"]
    assert "unknown_settings" in message
    assert "banana" in message


def test_settings_restart_keys_cover_protected_runtime_surface():
    # Sanity check the explicit policy; future contributors should think
    # twice before broadening the mutable surface.
    for protected in ("profile", "model", "host", "port", "load_mtp", "verify_core"):
        assert protected in DASHBOARD_RESTART_REQUIRED_KEYS
    assert "prefill_chunk_tokens" in DASHBOARD_MUTABLE_SETTINGS_KEYS
    assert "prefill_chunk_tokens" not in DASHBOARD_RESTART_REQUIRED_KEYS
    assert "generation_mode" in DASHBOARD_MUTABLE_SETTINGS_KEYS
    assert "generation_mode" not in DASHBOARD_RESTART_REQUIRED_KEYS


def test_settings_get_payload_keys_are_all_classified():
    """Every key the settings GET returns must be classified as mutable,
    read-only, or restart-required.

    The dashboard Settings page diffs its draft against the GET payload and
    POSTs the diff back. Object-valued keys always differ by reference after
    a store refresh, so ANY unclassified GET key eventually rides a POST and
    the all-or-nothing unknown_settings 400 silently kills the whole write
    (the 2026-07-02 presence-penalty-persistence flag). This test pins the
    contract so a new GET field cannot reintroduce the bug.
    """
    client = TestClient(create_app(_fake_state()))
    payload = client.get("/v1/mtplx/settings").json()
    classified = (
        set(DASHBOARD_MUTABLE_SETTINGS_KEYS)
        | set(DASHBOARD_READ_ONLY_SETTINGS_KEYS)
        | set(DASHBOARD_RESTART_REQUIRED_KEYS)
    )
    unclassified = sorted(set(payload.keys()) - classified)
    assert unclassified == [], (
        f"settings GET returns unclassified keys {unclassified}; add them to "
        "one of the DASHBOARD_*_SETTINGS_KEYS tuples (read-only for "
        "informational echo-back keys) or the dashboard settings POST 400s"
    )


def test_settings_post_tolerates_dashboard_echo_back_diff():
    """The dashboard-shaped write: one changed primitive + every
    object/array-valued GET key echoed back unchanged (reference-diff
    artifact). Must apply the primitive and drop the echoes — this is the
    exact payload that used to 400 and made presence_penalty (and every
    other dashboard settings write) silently non-persistent.
    """
    state = _fake_state()
    client = TestClient(create_app(state))
    snapshot = client.get("/v1/mtplx/settings").json()
    echo = {
        key: value
        for key, value in snapshot.items()
        if isinstance(value, (dict, list))
    }
    assert echo, "expected object/array-valued keys in the settings GET"
    response = client.post(
        "/v1/mtplx/settings", json={**echo, "presence_penalty": 0.5}
    )
    assert response.status_code == 200, response.text
    assert response.json().get("applied") == {"presence_penalty": 0.5}
    # presence/frequency penalties apply as server-side request DEFAULTS
    # (default_-prefixed on args), mirroring the completion-request override.
    assert state.args.default_presence_penalty == 0.5


def test_settings_post_rejects_invalid_depth_prefill_chunk_and_generation_mode():
    client = TestClient(create_app(_fake_state()))

    impossible_model_depth = client.post("/v1/mtplx/settings", json={"depth": 4})
    assert impossible_model_depth.status_code == 400
    assert "depth must be between 1 and 3 for the loaded model" in (
        impossible_model_depth.json()["error"]["message"]
    )

    bad_depth = client.post("/v1/mtplx/settings", json={"depth": 9})
    assert bad_depth.status_code == 400
    assert "depth must be between 1 and 8" in bad_depth.json()["error"]["message"]

    bad_prefill = client.post(
        "/v1/mtplx/settings", json={"prefill_chunk_tokens": 64}
    )
    assert bad_prefill.status_code == 400
    assert "prefill_chunk_tokens must be between 128 and 32768" in bad_prefill.json()[
        "error"
    ]["message"]

    bad_generation_mode = client.post(
        "/v1/mtplx/settings", json={"generation_mode": "off"}
    )
    assert bad_generation_mode.status_code == 400
    assert "generation_mode must be 'mtp' or 'ar'" in bad_generation_mode.json()[
        "error"
    ]["message"]


def test_settings_post_rejects_mtp_mode_when_runtime_has_no_mtp():
    state = _fake_state()
    state.runtime.mtp_enabled = False
    client = TestClient(create_app(state))

    ar_response = client.post("/v1/mtplx/settings", json={"generation_mode": "ar"})
    assert ar_response.status_code == 200
    assert state.args.generation_mode == "ar"

    mtp_response = client.post("/v1/mtplx/settings", json={"generation_mode": "mtp"})
    assert mtp_response.status_code == 400
    assert "requires a runtime loaded with MTP" in mtp_response.json()["error"][
        "message"
    ]


# ---- /v1/mtplx/cancel/{id} ------------------------------------------------


def test_cancel_endpoint_flips_handle_event():
    state = _fake_state()
    ev = Event()
    state.dashboard.in_flight.register(
        InFlightHandle(
            request_id="r1",
            cancel_event=ev,
            started_s=time.time(),
            model="m",
            prompt_preview="hi",
            prompt_tokens=2,
        )
    )
    client = TestClient(create_app(state))
    response = client.post("/v1/mtplx/cancel/r1")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["cancelled"] is True
    assert ev.is_set()


def test_cancel_endpoint_returns_ok_false_for_unknown_request():
    client = TestClient(create_app(_fake_state()))
    response = client.post("/v1/mtplx/cancel/does-not-exist")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert body["cancelled"] is False


# ---- /v1/mtplx/app/capabilities ------------------------------------------


def test_app_capabilities_returns_stable_native_backend_contract():
    client = TestClient(create_app(_fake_state()))
    response = client.get("/v1/mtplx/app/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["api_version"] == 1
    assert body["endpoints"]["snapshot"] == "/v1/mtplx/snapshot"
    assert body["endpoints"]["metrics_stream"] == "/v1/mtplx/metrics/stream"
    assert body["endpoints"]["settings"] == "/v1/mtplx/settings"
    assert body["endpoints"]["cancel"] == "/v1/mtplx/cancel/{request_id}"
    assert body["mutable_settings"] == list(DASHBOARD_MUTABLE_SETTINGS_KEYS)
    assert body["restart_required_settings"] == list(DASHBOARD_RESTART_REQUIRED_KEYS)
    assert body["features"]["sse_metrics"] is True
    assert body["features"]["request_cancel"] is True
    assert body["features"]["cache_clear"] is True
    assert body["features"]["ssd_session_cache"] is True
    assert body["features"]["ssd_cache_archive"] is True
    assert body["features"]["session_clear"] is True
    assert body["features"]["prefill_history"] is True
    assert body["features"]["thermal_polling"] is True
    assert body["features"]["scheduler_telemetry"] is True
    assert body["features"]["cooperative_scheduler_core"] is True
    assert body["features"]["ar_batching_live"] is True
    assert body["features"]["concurrent_mtp_ar_fallback"] is True
    assert body["features"]["mtp_cohorts_default_enabled"] is False
    assert body["scheduler"]["modes"] == [
        "serial",
        "cooperative",
        "ar_batch",
        "mtp_batch",
        "mtp_cohort_experimental",
    ]
    assert body["scheduler"]["default_ux"] == "coding_agents"
    assert body["scheduler"]["default_policy"] == "solo_mtp_oracle"
    assert body["snapshot_interval"]["default_ms"] == DASHBOARD_SNAPSHOT_INTERVAL_DEFAULT_MS
    assert body["snapshot_interval"]["min_ms"] == DASHBOARD_SNAPSHOT_INTERVAL_MIN_MS
    assert body["snapshot_interval"]["max_ms"] == DASHBOARD_SNAPSHOT_INTERVAL_MAX_MS
    assert body["snapshot_interval"]["idle_min_ms"] == DASHBOARD_SNAPSHOT_INTERVAL_IDLE_MIN_MS
    assert body["snapshot_interval"]["native_default_ms"] == 500
    assert body["snapshot_interval"]["performance_lock_ms"] == 1000


# ---- /v1/mtplx/metrics/stream --------------------------------------------


def test_metrics_stream_route_is_registered_and_streams():
    """The SSE endpoint is registered on the FastAPI app with the right
    media type. We deliberately do *not* iterate the body here because
    the SSE handler is an infinite loop and TestClient's sync
    ``stream(...)`` does not propagate disconnect into the coroutine
    fast enough to make per-iteration assertions reliable. Behavior of
    the underlying helpers (snapshot shape, bus deliveries) is covered
    by direct unit tests below."""

    app = create_app(_fake_state())
    routes = [(getattr(r, "path", None), getattr(r, "name", None)) for r in app.routes]
    assert ("/v1/mtplx/metrics/stream", "mtplx_metrics_stream") in routes


def test_metrics_stream_respects_api_key_auth_before_streaming():
    client = TestClient(create_app(_fake_state(api_key="test-key")))

    missing = client.get("/v1/mtplx/metrics/stream?snapshot_interval_ms=500")
    assert missing.status_code == 401
    assert missing.json()["error"]["type"] == "authentication_error"

    wrong = client.get(
        "/v1/mtplx/metrics/stream?snapshot_interval_ms=500",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert wrong.status_code == 401
    assert wrong.json()["error"]["type"] == "authentication_error"


def test_metrics_stream_accepts_bounded_snapshot_interval():
    assert _dashboard_snapshot_interval_s(None) == 0.2
    assert _dashboard_snapshot_interval_s(500) == 0.5
    assert _dashboard_snapshot_interval_s(1000) == 1.0
    assert _dashboard_snapshot_interval_s(1) == 0.1
    assert _dashboard_snapshot_interval_s(999_999) == 5.0


def test_metrics_stream_relaxes_full_snapshots_only_while_idle():
    assert _dashboard_snapshot_interval_for_activity_s(0.1, active_requests=1) == 0.1
    assert _dashboard_snapshot_interval_for_activity_s(0.1, active_requests=0) == 1.0
    assert _dashboard_snapshot_interval_for_activity_s(2.0, active_requests=0) == 2.0


def test_metrics_bus_round_trips_completed_event():
    """Direct asyncio-level coverage of the bus + subscriber contract."""

    import asyncio

    state = _fake_state()
    bus = state.dashboard.bus

    async def scenario():
        bus.attach_loop(asyncio.get_running_loop())
        queue = bus.subscribe()
        try:
            assert bus.subscriber_count() == 1

            def publish_from_thread() -> None:
                # Mimics the generation-thread publish path used by
                # `_dashboard_record_completion`.
                bus.publish({"kind": "completed", "envelope": {"decode_tok_s": 42.0}})

            Thread(target=publish_from_thread, daemon=True).start()
            event = await asyncio.wait_for(queue.get(), timeout=2.0)
            assert event["kind"] == "completed"
            assert event["envelope"]["decode_tok_s"] == 42.0
        finally:
            bus.unsubscribe(queue)
            assert bus.subscriber_count() == 0

    asyncio.run(scenario())


# ---- Static mount --------------------------------------------------------


def test_dashboard_static_mount_responds_with_html():
    client = TestClient(create_app(_fake_state()))
    response = client.get("/dashboard/")
    assert response.status_code in (200, 307, 308)
    if response.status_code in (307, 308):
        # Follow the redirect to the index path.
        response = client.get(response.headers["location"])
        assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    text = response.text.lower()
    if has_static_bundle():
        assert "<div id=\"root\"" in response.text or "mtplx" in text
    else:
        assert "dashboard" in text
        assert "bun" in text or "bundle is missing" in text


# ---- Existing public surface untouched -----------------------------------


def test_root_chat_ui_still_renders():
    client = TestClient(create_app(_fake_state()))
    root = client.get("/")
    assert root.status_code == 200
    assert "MTPLX" in root.text
    assert 'id="messages"' in root.text


def test_metrics_endpoint_unchanged_shape():
    client = TestClient(create_app(_fake_state()))
    payload = client.get("/metrics").json()
    assert "latest" in payload
    assert "recent" in payload


# ---- Helper-function unit coverage ---------------------------------------


def test_dashboard_record_completion_records_lifetime_and_rolling():
    state = _fake_state()
    envelope = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "cached_tokens": 10,
        "decode_tok_s": 55.0,
        "session_id": "sess-x",
        "new_prefill_tokens": 90,
        "prompt_eval_time_s": 0.1,
        "prefill_tok_s": 900.0,
        "ttft_s": 0.12,
        "context_len": 150,
    }
    openai._dashboard_record_completion(state, envelope=envelope, stats={})
    lifetime = state.dashboard.lifetime.snapshot()
    assert lifetime["requests_total"] == 1
    assert lifetime["prompt_tokens_total"] == 100
    assert lifetime["completion_tokens_total"] == 50
    rolling = state.dashboard.rolling.snapshot()
    assert rolling["count"] == 1
    assert rolling["sticky_all_time_max"] == 55.0
    assert state.dashboard.prefill_history.snapshot()[-1]["prefill_tok_s"] == 900.0


def test_dashboard_record_completion_prefers_raw_decode_for_records():
    state = _fake_state()

    openai._dashboard_record_completion(
        state,
        envelope={
            "prompt_tokens": 16,
            "completion_tokens": 318,
            "decode_tok_s": 53.5,
            "display_decode_tok_s": 52.2,
            "session_id": "sess-chat",
        },
        stats={},
    )

    rolling = state.dashboard.rolling.snapshot()
    assert rolling["max"] == 53.5
    assert rolling["sticky_all_time_max"] == 53.5


def test_dashboard_progress_does_not_set_sticky_speed_record():
    state = _fake_state()
    queue = state.dashboard.bus.subscribe()

    openai._dashboard_publish_progress(
        state,
        request_id="chatcmpl-live",
        payload={
            "decode_tok_s": 85.8,
            "display_decode_tok_s": 64.7,
            "completion_tokens": 12,
            "session_id": "sess-live",
        },
    )

    rolling = state.dashboard.rolling.snapshot()
    assert rolling["sticky_all_time_max"] == 0.0
    assert rolling["live_history"][-1]["tok_s"] == 85.8
    assert queue.get_nowait()["kind"] == "progress"
    assert queue.empty()


def test_dashboard_live_history_is_bounded_and_downsampled():
    rolling = RollingMetrics()
    original_interval = rolling.LIVE_SAMPLE_MIN_INTERVAL_S
    try:
        rolling.LIVE_SAMPLE_MIN_INTERVAL_S = 0.0
        for idx in range(rolling.LIVE_HISTORY_MAX_POINTS + 75):
            rolling.observe_progress(40.0 + idx, "sess-live")
    finally:
        rolling.LIVE_SAMPLE_MIN_INTERVAL_S = original_interval

    live_history = rolling.snapshot()["live_history"]
    assert len(live_history) == rolling.LIVE_HISTORY_MAX_POINTS
    assert live_history[-1]["tok_s"] == 40.0 + rolling.LIVE_HISTORY_MAX_POINTS + 74


def test_dashboard_progress_events_are_ui_rate_limited():
    state = _fake_state()
    queue = state.dashboard.bus.subscribe()

    for token in range(1, 20):
        openai._dashboard_publish_progress(
            state,
            request_id="chatcmpl-live",
            payload={
                "decode_tok_s": 50.0,
                "completion_tokens": token,
                "session_id": "sess-live",
            },
        )

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    assert [event["kind"] for event in events] == ["progress"]
    assert state.dashboard.in_flight.snapshot() == []


def test_dashboard_progress_publishes_early_token_milestones():
    state = _fake_state()
    state.dashboard.in_flight.register(
        InFlightHandle(
            request_id="chatcmpl-live",
            cancel_event=Event(),
            started_s=time.time(),
            session_id="sess-live",
            model="mtplx-test-model",
            prompt_preview="aime",
            prompt_tokens=12,
        )
    )
    queue = state.dashboard.bus.subscribe()

    for token in (1, 7, 20, 32, 64, 80, 128, 256):
        openai._dashboard_publish_progress(
            state,
            request_id="chatcmpl-live",
            payload={
                "decode_tok_s": 50.0 + token,
                "completion_tokens": token,
                "session_id": "sess-live",
            },
        )

    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    published_tokens = [
        event["progress"]["completion_tokens"]
        for event in events
        if event["kind"] == "progress"
    ]
    assert published_tokens == [1, 32, 64, 128, 256]
    snapshot = state.dashboard.in_flight.snapshot()
    assert snapshot[0]["last_progress"]["completion_tokens"] == 256
    stats = state.dashboard.progress_events.stats_for("chatcmpl-live")
    assert stats["dashboard_progress_published_events"] == 5
    assert stats["dashboard_progress_throttled_events"] == 3


def test_dashboard_live_history_keys_stateless_requests_by_request_id():
    state = _fake_state()

    openai._dashboard_publish_progress(
        state,
        request_id="chatcmpl-stateless",
        payload={
            "decode_tok_s": 52.0,
            "completion_tokens": 12,
        },
    )

    live_history = state.dashboard.rolling.snapshot()["live_history"]
    assert live_history[-1]["session_id"] == "chatcmpl-stateless"


def test_dashboard_progress_gate_resets_per_request():
    gate = ProgressEventGate()

    assert gate.should_publish("chatcmpl-live", now=10.0)
    assert not gate.should_publish("chatcmpl-live", now=10.1)
    assert gate.should_publish("chatcmpl-live", now=10.21)

    gate.forget("chatcmpl-live")
    assert gate.should_publish("chatcmpl-live", now=10.22)


def test_dashboard_record_completion_deregisters_finished_request():
    state = _fake_state()
    state.dashboard.in_flight.register(
        InFlightHandle(
            request_id="chatcmpl-finished",
            cancel_event=Event(),
            started_s=time.time(),
            session_id="sess-x",
            model="mtplx-test-model",
            prompt_preview="done",
            prompt_tokens=12,
        )
    )

    openai._dashboard_record_completion(
        state,
        envelope={
            "request_id": "chatcmpl-finished",
            "prompt_tokens": 12,
            "completion_tokens": 6,
            "decode_tok_s": 42.0,
            "session_id": "sess-x",
        },
        stats={},
    )

    assert state.dashboard.in_flight.count() == 0


def test_dashboard_prefill_chunk_exposes_live_and_cumulative_rates():
    state = _fake_state()
    state.dashboard.in_flight.register(
        InFlightHandle(
            request_id="prefill-1",
            cancel_event=Event(),
            started_s=time.time(),
            session_id="sess-prefill",
        )
    )

    openai._dashboard_publish_prefill(
        state,
        request_id="prefill-1",
        session_id="sess-prefill",
        payload={
            "phase": "chunk",
            "tokens_done": 4096,
            "tokens_total": 10000,
            "elapsed_s": 16.0,
            "chunk_size": 2048,
            "chunk_elapsed_s": 4.0,
            "chunk_prefill_tok_s": 512.0,
        },
    )

    handle = state.dashboard.in_flight.get("prefill-1")
    assert handle is not None
    assert handle.prefill_state is not None
    assert handle.prefill_state["prefill_tok_s"] == 256.0
    assert handle.prefill_state["cumulative_prefill_tok_s"] == 256.0
    assert handle.prefill_state["prefill_wall_tok_s"] == 256.0
    assert handle.prefill_state["live_prefill_tok_s"] == 512.0


def test_dashboard_prompt_preview_truncates_long_messages():
    long_text = "abcdefghij" * 20
    request = type(
        "FakeReq",
        (),
        {
            "messages": [type("M", (), {"role": "user", "content": long_text})()],
            "prompt": None,
        },
    )()
    preview = openai._dashboard_prompt_preview(request, tokenizer=None, max_chars=40)
    assert len(preview) <= 40
    assert preview.endswith("...")
    assert preview.startswith("abc")


def test_rolling_metrics_per_session_map_is_lru_bounded():
    # Agent clients mint fresh session ids freely; the per-session max map is
    # copied whole into every dashboard snapshot, so it must stay bounded for
    # the daemon's lifetime (external review F5).
    metrics = RollingMetrics()
    for index in range(200):
        metrics.append(10.0 + index, f"session-{index}")
    snapshot = metrics.snapshot()
    per_session = snapshot["max_per_session"]
    assert len(per_session) == RollingMetrics.MAX_PER_SESSION_ENTRIES
    # Most-recent sessions survive; the oldest were evicted.
    assert "session-199" in per_session
    assert "session-0" not in per_session
