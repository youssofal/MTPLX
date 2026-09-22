"""The Splash bridge must answer with exactly what MTPLX clients decode.

The native app's Swift DTOs are the contract: a field declared non-optional
there fails the whole decode when the server omits it, which would blank the
dashboard rather than degrade it. Rather than restate that contract here and
let the two drift, these tests parse `DashboardModels.swift` and assert the
bridge's live payloads carry every required key.

The engine itself is stubbed by a fake Splash HTTP server, so the suite runs
without a 17 GB model package, without Metal, and without MLX.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mtplx.server.splash_bridge import SplashInstall, create_app
from mtplx.server.splash_bridge.supervisor import SplashEngine
from mtplx.server.splash_bridge.telemetry import BridgeTelemetry

MODEL = "incoai/Qwen3.8-27B-Splash"
SWIFT_DTOS = (
    Path(__file__).resolve().parents[1]
    / "apps/MTPLXApp/Sources/MTPLXAppCore/Models/DashboardModels.swift"
)

# What a healthy Splash reports; keys mirror its prometheus metric paths.
FAKE_STATUS = {
    "ready": 1,
    "maximum_context_tokens": 262144,
    "metal": {"healthy": 1},
    "frontend": {"active": 0, "waiting": 0},
    "requests": {"submitted": 3, "completed": 3, "cancelled": 0, "failed": 0},
    "scheduler": {"queued": 0, "prefilling": 0, "decoding": 0},
    "cache": {"hits": 7, "cold_misses": 1, "reused_tokens": 4096},
    "kv": {
        "blocks": 12,
        "pages_total": 4096,
        "pages_free": 3900,
        "pages_active": 96,
        "pages_cache": 100,
        "resident_backing_bytes": 2_147_483_648,
        "reclaimable_backing_bytes": 536_870_912,
    },
}


# --------------------------------------------------------------------------
# a fake Splash engine over HTTP
# --------------------------------------------------------------------------

# Every generation body the fake engine received, newest last.
RECEIVED: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):  # keep pytest output clean
        pass

    def _send(self, code, body, content_type="application/json"):
        payload = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/ready":
            self._send(200, json.dumps({"status": "ready"}))
        elif self.path == "/status":
            self._send(200, json.dumps(FAKE_STATUS))
        elif self.path == "/metrics":
            self._send(200, "splash_kv_pages_total 4096\n", "text/plain")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path.startswith("/v1/"):
            RECEIVED.append(request)
        messages = request.get("messages") or [{}]
        # Only generation is slow; templating and tokenizing run no model.
        if self.path.startswith("/v1/") and messages[-1].get("content") == "slow":
            time.sleep(1.5)
        if self.path == "/apply-template":
            self._send(200, json.dumps({"prompt": "<|im_start|>user\nhi<|im_end|>"}))
            return
        if self.path == "/tokenize":
            self._send(200, json.dumps({"tokens": list(range(11))}))
            return
        if not self.path.startswith(("/v1/chat/completions", "/v1/messages")):
            self._send(404, json.dumps({"error": "not found"}))
            return
        if not request.get("stream"):
            self._send(
                200,
                json.dumps(
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "hi"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 11,
                            "completion_tokens": 5,
                            "prompt_tokens_details": {"cached_tokens": 4},
                        },
                    }
                ),
            )
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # The shape a live Splash 1.0.2 stream has: an id on every frame,
        # thinking in reasoning_content, then a bare finish frame.
        deltas = [{"reasoning_content": "Hm"}] + [
            {"content": token} for token in ("He", "llo", "!")
        ]
        for delta in deltas:
            frame = {"id": "chatcmpl-up", "choices": [{"delta": delta, "index": 0}]}
            self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.flush()
        finish = {
            "id": "chatcmpl-up",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        self.wfile.write(f"data: {json.dumps(finish)}\n\n".encode())
        # Splash only emits the trailing usage frame when it is asked to.
        if (request.get("stream_options") or {}).get("include_usage"):
            final = {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4}}
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture(scope="module")
def fake_splash():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


class _StubEngine(SplashEngine):
    """A SplashEngine pointed at the fake server, with no process to spawn."""

    def start(self) -> None:
        self.state.phase = "ready"
        self.state.ready_at = time.time()

    def stop(self) -> None:
        self.state.phase = "stopped"

    def is_running(self) -> bool:
        return self.state.phase == "ready"


@pytest.fixture()
def client(fake_splash):
    install = SplashInstall.discover()
    engine = _StubEngine(install, MODEL, port=fake_splash)
    engine.start()
    app = create_app(
        install=install,
        engine=engine,
        telemetry=BridgeTelemetry(MODEL),
    )
    with TestClient(app) as http:
        yield http


# --------------------------------------------------------------------------
# the Swift contract, read from source
# --------------------------------------------------------------------------


def _required_fields(struct: str) -> set[str]:
    """Non-optional stored properties of a Swift struct, by their JSON key."""
    lines = SWIFT_DTOS.read_text(encoding="utf-8").split("\n")
    body: list[str] = []
    for index, line in enumerate(lines):
        if not re.match(rf"\s*public struct {struct}\b", line):
            continue
        depth = 0
        for cursor in range(index, len(lines)):
            depth += lines[cursor].count("{") - lines[cursor].count("}")
            body.append(lines[cursor])
            if depth == 0 and cursor > index:
                break
        break
    keys = {}
    for line in body:
        match = re.match(r'\s*case (\w+) = "([^"]+)"', line)
        if match:
            keys[match.group(1)] = match.group(2)
    required = set()
    for line in body:
        match = re.match(r"^    public (?:let|var) (\w+): ([^={\n]+?)\s*(?:=.*)?$", line)
        if match and not match.group(2).strip().endswith("?"):
            required.add(keys.get(match.group(1), match.group(1)))
    return required


def test_swift_dtos_are_present():
    assert SWIFT_DTOS.is_file(), "the native app's DTOs define the contract"
    assert _required_fields("HealthPayload"), "parser found no required fields"


@pytest.mark.parametrize(
    "path,struct",
    [
        ("/health", "HealthPayload"),
        ("/v1/mtplx/snapshot", "DashboardSnapshot"),
        ("/v1/mtplx/app/capabilities", "AppCapabilities"),
        ("/v1/mtplx/prefill_history", "PrefillHistoryPayload"),
        ("/admin/sessions", "SessionsPayload"),
    ],
)
def test_endpoint_satisfies_swift_contract(client, path, struct):
    response = client.get(path)
    assert response.status_code == 200, response.text
    payload = response.json()
    missing = _required_fields(struct) - set(payload)
    assert not missing, f"{path} omits required {struct} keys: {sorted(missing)}"


def test_snapshot_nested_structs_satisfy_contract(client):
    snapshot = client.get("/v1/mtplx/snapshot").json()
    for key, struct in (
        ("rolling", "RollingMetrics"),
        ("lifetime", "LifetimeSnapshot"),
        ("sessions", "SessionsPayload"),
        ("mem", "MemSnapshot"),
    ):
        missing = _required_fields(struct) - set(snapshot[key])
        assert not missing, f"snapshot.{key} omits {sorted(missing)}"


# --------------------------------------------------------------------------
# behaviour
# --------------------------------------------------------------------------


def test_kv_quantization_is_declared_fixed_not_offered(client):
    """Splash's KV width is a kernel property, so the control must be off."""
    policy = client.get("/v1/mtplx/settings").json()["kv_quant_policy"]
    assert policy["supported"] is False
    assert policy["modes"] == ["q8"]
    assert "8-bit" in policy["disabled_reason"]
    features = client.get("/v1/mtplx/app/capabilities").json()["features"]
    assert features["kv_quantization"] is False
    assert features["mtp"] is False, "Splash drafts with DFlash 2, not MTPLX MTP"


def test_engine_stats_pass_through_real_splash_numbers(client):
    stats = client.get("/v1/mtplx/snapshot").json()["engine_stats"]
    assert stats["engine"] == "splash"
    assert stats["kv_quantization_bits"] == 8
    assert stats["kv_pages_total"] == 4096
    assert stats["cache_reused_tokens"] == 4096
    assert stats["kv_resident_backing_bytes"] == 2_147_483_648


def test_context_window_comes_from_the_engine(client):
    assert client.get("/health").json()["context_window"] == 262144


def test_models_lists_the_loaded_package(client):
    body = client.get("/v1/models").json()
    assert [entry["id"] for entry in body["data"]] == [MODEL]
    assert body["data"][0]["owned_by"] == "splash"


def test_chat_completion_proxies_and_counts_usage(client):
    response = client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "hi"
    lifetime = client.get("/v1/mtplx/snapshot").json()["lifetime"]
    assert lifetime["requests_total"] == 1
    assert lifetime["prompt_tokens_total"] == 11
    assert lifetime["completion_tokens_total"] == 5
    assert lifetime["cached_tokens_total"] == 4


def test_streaming_chat_forwards_every_frame(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert "He" in body and "llo" in body
    assert body.rstrip().endswith("[DONE]")


def test_unloaded_engine_refuses_inference_with_503(fake_splash):
    install = SplashInstall.discover()
    engine = _StubEngine(install, MODEL, port=fake_splash)  # never started
    app = create_app(
        install=install,
        engine=engine,
        telemetry=BridgeTelemetry(MODEL),
    )
    with TestClient(app) as http:
        assert http.get("/ready").status_code == 503
        response = http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 503


def test_api_key_is_enforced_when_set(fake_splash):
    install = SplashInstall.discover()
    engine = _StubEngine(install, MODEL, port=fake_splash)
    engine.start()
    app = create_app(
        install=install,
        engine=engine,
        telemetry=BridgeTelemetry(MODEL),
        api_key="secret",
    )
    with TestClient(app) as http:
        body = {"messages": [{"role": "user", "content": "hi"}]}
        assert http.post("/v1/chat/completions", json=body).status_code == 401
        ok = http.post(
            "/v1/chat/completions",
            json=body,
            headers={"Authorization": "Bearer secret"},
        )
        assert ok.status_code == 200


def test_install_discovery_reports_official_packages():
    install = SplashInstall.discover()
    install.validate()
    models = install.official_models()
    assert MODEL in models
    assert all(candidate.count("/") == 1 for candidate in models)


def test_only_frames_carrying_text_count_as_tokens():
    """Role, finish and ping frames must not inflate the dashboard's tok/s."""
    from mtplx.server.splash_bridge.app import _has_text_delta

    assert _has_text_delta({"choices": [{"delta": {"content": "Hi"}}]})
    assert _has_text_delta({"delta": {"text": "Hi"}})  # Anthropic
    assert _has_text_delta({"delta": {"partial_json": '{"a"'}})  # tool arguments

    assert not _has_text_delta({"choices": [{"delta": {"role": "assistant"}}]})
    assert not _has_text_delta({"choices": [{"delta": {}, "finish_reason": "stop"}]})
    assert not _has_text_delta({"choices": [], "usage": {"completion_tokens": 3}})
    assert not _has_text_delta({"type": "ping"})


def test_machine_info_is_resolved_once():
    """/health and the SSE stream poll constantly; this shells out to sysctl."""
    from mtplx.server.splash_bridge.app import _machine_info

    _machine_info.cache_clear()
    first = _machine_info()
    assert _machine_info() is first, "machine info must be cached, not re-shelled"


def test_a_slow_generation_does_not_block_the_event_loop(client):
    """A blocking proxy call on the loop would stall /health.

    The app's liveness probe polls /health; if one non-streaming generation
    froze the loop, the probe would time out and the app would decide the
    daemon had died mid-answer. The proxy must therefore run off the loop.
    """
    import threading

    done = threading.Event()

    def slow_call():
        try:
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "slow"}]},
            )
        finally:
            done.set()

    worker = threading.Thread(target=slow_call, daemon=True)
    worker.start()
    time.sleep(0.4)  # let the generation be genuinely in flight
    assert not done.is_set(), "fixture too fast to prove anything"

    started = time.monotonic()
    assert client.get("/health").status_code == 200
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"/health blocked for {elapsed:.2f}s behind a generation"

    worker.join(timeout=10)
    assert done.is_set()


def test_boolean_status_flags_are_not_read_as_zero():
    """Splash reports flags as JSON booleans; its exporter maps them to 1/0.

    Excluding `bool` from the numeric reader (because `isinstance(True, int)`
    is True in Python) made a healthy Metal device report as unhealthy.
    """
    from mtplx.server.splash_bridge.telemetry import _num

    assert _num({"metal": {"healthy": True}}, "metal", "healthy") == 1.0
    assert _num({"metal": {"healthy": False}}, "metal", "healthy") == 0.0
    assert _num({"ready": True}, "ready") == 1.0
    assert _num({"kv": {"pages_total": 32768}}, "kv", "pages_total") == 32768.0
    # absent and non-numeric still fall back
    assert _num({}, "kv", "pages_total") == 0.0
    assert _num({"kv": {"pages_total": "many"}}, "kv", "pages_total") == 0.0


def test_streaming_counts_prompt_tokens_without_changing_the_client_stream(client):
    """The bridge asks Splash for usage; the client must not see the extra frame.

    Splash omits usage on a stream unless asked, which left every streamed
    turn with prompt_tokens=0 on the dashboard. The bridge opts in on the
    client's behalf and drops the usage-only frame again on the way out.
    """
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        body = "".join(response.iter_text())

    assert '"choices": []' not in body, "client did not ask for a usage frame"
    assert body.rstrip().endswith("[DONE]")

    lifetime = client.get("/v1/mtplx/snapshot").json()["lifetime"]
    assert lifetime["prompt_tokens_total"] == 11, "prompt tokens must be accounted"
    assert lifetime["completion_tokens_total"] == 4


def test_a_client_that_asks_for_usage_still_receives_it(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        body = "".join(response.iter_text())
    assert '"usage"' in body, "an explicit include_usage must be forwarded"


def _frames(body):
    return [
        json.loads(line[5:])
        for line in body.splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    ]


def test_the_chat_stream_carries_the_stats_the_app_renders(client):
    """The app's chat footer and tok/s chip read mtplx_stats and usage off the
    finish frame; a Splash reply without them rendered a blank footer."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        body = "".join(response.iter_text())
    assert body.rstrip().endswith("data: [DONE]")
    frames = _frames(body)
    finishes = [f for f in frames if (f.get("choices") or [{}])[0].get("finish_reason")]
    assert len(finishes) == 1
    finish = finishes[0]
    assert finish is frames[-1], "the finish frame must stay last"
    assert finish["usage"]["prompt_tokens"] == 11
    assert finish["usage"]["completion_tokens"] == 4
    stats = finish["mtplx_stats"]
    assert stats["completion_tokens"] == 4
    assert stats["generation_mode"] == "splash"
    # The chat page's stats line reads "N thinking" from this.
    assert stats["reasoning_tokens"] == 1
    # We asked Splash for usage; the client did not, so no usage-only frame.
    assert not any(f.get("choices") == [] for f in frames)


def test_a_client_that_asks_for_usage_gets_one_usage_frame(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        body = "".join(response.iter_text())
    frames = _frames(body)
    assert [f for f in frames if f.get("choices") == []] == [frames[-1]]


def test_thinking_tokens_move_the_live_gauge():
    from mtplx.server.splash_bridge.app import _has_text_delta

    assert _has_text_delta({"choices": [{"delta": {"reasoning_content": "Hm"}}]})


def test_cancel_by_the_id_the_client_saw():
    """The app stops a reply by the chatcmpl id on the stream, which is
    Splash's, not the bridge's own request id."""
    telemetry = BridgeTelemetry("incoai/Qwen3.8-27B-Splash")
    trace = telemetry.begin("hi")
    telemetry.alias(trace, "chatcmpl-up")
    assert telemetry.cancel("chatcmpl-up")
    assert trace.cancelled
    telemetry.finish(trace)
    assert not telemetry.cancel("chatcmpl-up"), "aliases die with the request"


# --------------------------------------------------------------------------
# the browser chat page: MTPLX's own, for both engines
# --------------------------------------------------------------------------


def _element_ids(page):
    return set(re.findall(r'id="([^"]+)"', page))


def test_the_chat_page_is_the_mlx_engines_page(client):
    """Splash served Inco's own chat page, so switching engines swapped the
    whole browser UI. It must be MTPLX's page, element for element."""
    from mtplx.server.chat_page import chat_ui_html

    page = client.get("/").text
    mlx_page = chat_ui_html(
        model_id=MODEL,
        server_url="http://testserver",
        api_key_required=False,
        default_settings={"depth": 3, "depth_max": 3},
    )
    assert "<title>MTPLX</title>" in page
    assert _element_ids(page) == _element_ids(mlx_page)
    assert f"const MODEL_ID = {json.dumps(MODEL)};" in page
    for section in ("Sampling", "Speculative", "Output", "System prompt"):
        assert f'<p class="sb-title">{section}</p>' in page


def test_the_chat_page_shows_what_splash_runs(client):
    page = client.get("/").text
    # Speculation is Splash's DFlash 2 draft, fixed, not MTPLX's MTP.
    assert '<label for="ctl-mtp">DFlash 2 <span' in page
    assert '<label for="ctl-depth">Draft tokens <span' in page
    assert '"depth": 7' in page and '"mtp_enabled": true' in page
    # Sliders span exactly what Splash accepts, so none can fail a request.
    assert '<input id="ctl-top-k" type="range" min="1" max="32"' in page
    assert '<input id="ctl-top-p" type="range" min="0.01" max="1"' in page
    assert '"presence_penalty": 0' in page


def test_every_route_the_chat_page_calls_answers(client):
    page = client.get("/").text
    routes = set(re.findall(r'fetch\("(/[^"]*)"', page))
    assert routes == {"/health", "/v1/mtplx/settings", "/v1/chat/completions"}
    assert client.get("/health").status_code == 200
    assert client.get("/v1/mtplx/settings").status_code == 200
    health = client.get("/health").json()
    assert health["runtime_mode"] == "Splash · DFlash 2"
    assert health["context_window"] > 0


# --------------------------------------------------------------------------
# live settings: shared by the app's panel and the chat page, applied to turns
# --------------------------------------------------------------------------


def test_settings_carry_the_fields_the_chat_page_reads(client):
    settings = client.get("/v1/mtplx/settings").json()
    for key in ("temperature", "top_p", "top_k", "presence_penalty", "reasoning"):
        assert key in settings
    assert settings["generation_mode"] == "splash"
    assert 1 <= settings["top_k"] <= 32


def test_a_settings_write_lands_and_snaps_to_what_splash_runs(client):
    written = client.post(
        "/v1/mtplx/settings",
        json={
            "temperature": 0.7,
            "top_k": 50,
            "presence_penalty": 1.5,
            "generation_mode": "mtp",
            "depth": 3,
            "max_response_tokens": 4096,
            "reasoning": "off",
        },
    ).json()
    assert written["temperature"] == 0.7
    assert written["top_k"] == 32, "top_k past Splash's 32 must snap, not fail turns"
    assert written["presence_penalty"] == 0.0
    assert set(written["adjusted"]) == {"top_k", "presence_penalty"}
    assert set(written["ignored"]) == {"generation_mode", "depth"}
    assert written["reasoning"] == "off"
    # And it stays written: the chat page polls this every 1.5 s.
    again = client.get("/v1/mtplx/settings").json()
    assert (again["temperature"], again["top_k"], again["max_response_tokens"]) == (
        0.7,
        32,
        4096,
    )


def test_top_k_off_becomes_splashs_widest_not_greedy():
    """MTPLX's top_k 0 means no filter; top-1 would silently make it greedy."""
    from mtplx.server.splash_bridge.settings import SamplingSettings

    settings = SamplingSettings()
    report = settings.update({"top_k": 0})
    assert settings.top_k == 32
    assert "top_k" in report["adjusted"]
    settings.update({"top_k": -3})
    assert settings.top_k == 1


def test_reasoning_wins_over_the_legacy_thinking_flag():
    from mtplx.server.splash_bridge.settings import SamplingSettings

    settings = SamplingSettings()
    settings.update({"reasoning": "auto", "enable_thinking": True})
    assert settings.reasoning == "auto"
    settings.update({"enable_thinking": False})
    assert settings.reasoning == "off"


QWEN38_EFFORTS = (("xhigh", "medium", "low"), "xhigh")


@pytest.fixture()
def effort_client(fake_splash):
    """A bridge whose package template takes Qwen 3.8's effort levels."""
    from mtplx.server.splash_bridge.settings import SamplingSettings

    install = SplashInstall.discover()
    engine = _StubEngine(install, MODEL, port=fake_splash)
    engine.start()
    app = create_app(
        install=install,
        engine=engine,
        telemetry=BridgeTelemetry(MODEL),
        sampling=SamplingSettings(efforts=lambda: QWEN38_EFFORTS),
    )
    with TestClient(app) as http:
        yield http


def test_the_packages_thinking_levels_are_advertised(effort_client):
    """The app's panel shows an effort picker, and the chat bar its Thinking
    selector, only when the reasoning policy lists levels."""
    policy = effort_client.get("/v1/mtplx/settings").json()["reasoning_policy"]
    assert policy["supported"] is True
    assert policy["modes"] == ["auto", "on", "off"]
    assert policy["effort_levels"] == ["xhigh", "medium", "low"]
    assert policy["default_effort"] == "xhigh", "Splash's template default"
    controls = effort_client.get("/health").json()["model_controls"]
    assert controls["reasoning"] == policy


def test_a_chosen_thinking_level_reaches_splash(effort_client):
    written = effort_client.post(
        "/v1/mtplx/settings", json={"reasoning_effort": "medium"}
    ).json()
    assert written["reasoning_effort"] == "medium"
    RECEIVED.clear()
    effort_client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert RECEIVED[-1]["reasoning_effort"] == "medium"
    # Off still wins: a level is how hard to think, not whether to.
    effort_client.post("/v1/mtplx/settings", json={"reasoning": "off"})
    RECEIVED.clear()
    effort_client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert RECEIVED[-1]["reasoning_effort"] == "none"


def test_effort_vocabulary_matches_splash(effort_client):
    report = effort_client.post(
        "/v1/mtplx/settings", json={"reasoning_effort": "high"}
    ).json()
    assert report["reasoning_effort"] == "xhigh", "Splash folds high into xhigh"
    report = effort_client.post(
        "/v1/mtplx/settings", json={"reasoning_effort": "banana"}
    ).json()
    assert "reasoning_effort" in report["ignored"]
    assert report["reasoning_effort"] == "xhigh"
    # MTPLX's "auto" means the default level; Splash would refuse the word.
    effort_client.post("/v1/mtplx/settings", json={"reasoning_effort": "auto"})
    RECEIVED.clear()
    effort_client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "auto",
        },
    )
    assert "reasoning_effort" not in RECEIVED[-1]


def test_a_model_without_levels_takes_no_effort():
    from mtplx.server.splash_bridge.settings import SamplingSettings

    settings = SamplingSettings()
    report = settings.update({"reasoning_effort": "low"})
    assert "reasoning_effort" in report["ignored"]
    sent = settings.apply("/v1/chat/completions", {"messages": []})
    assert "reasoning_effort" not in sent


def test_the_chat_bar_has_the_thinking_selector_on_both_engines(client):
    from mtplx.server.chat_page import chat_ui_html

    mlx_page = chat_ui_html(
        model_id=MODEL,
        server_url="http://testserver",
        api_key_required=False,
        default_settings={"depth": 3, "depth_max": 3},
    )
    for page in (client.get("/").text, mlx_page):
        assert 'id="think-pill"' in page and 'id="composer-think"' in page
        # Built from the loaded model's policy, and sent with each turn.
        assert "payload.reasoning_policy" in page
        assert "requestBody.reasoning_effort = settingsNow.reasoning_effort" in page


def test_live_settings_fill_what_the_app_chat_leaves_out(client):
    """The app's chat sends no sampling fields; the panel's values must reach
    the engine rather than Splash's built-in defaults."""
    client.post(
        "/v1/mtplx/settings",
        json={"temperature": 0.4, "top_p": 0.8, "top_k": 8, "reasoning": "off"},
    )
    RECEIVED.clear()
    client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
    )
    sent = RECEIVED[-1]
    assert (sent["temperature"], sent["top_p"], sent["top_k"]) == (0.4, 0.8, 8)
    # Splash ignores enable_thinking; "off" has to arrive as reasoning_effort.
    assert sent["reasoning_effort"] == "none"


def test_a_clients_own_values_win_over_live_settings(client):
    client.post("/v1/mtplx/settings", json={"temperature": 0.4, "reasoning": "off"})
    RECEIVED.clear()
    client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 1.2,
            "enable_thinking": True,
        },
    )
    sent = RECEIVED[-1]
    assert sent["temperature"] == 1.2
    assert "reasoning_effort" not in sent, "an explicit thinking request stands"


def test_hide_thinking_from_the_chat_page_reaches_splash(client):
    RECEIVED.clear()
    client.post(
        "/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": False,
        },
    )
    assert RECEIVED[-1]["reasoning_effort"] == "none"


def test_chat_page_is_served_and_drives_the_bridge(client):
    """One chat page, on the bridge's port, posting through the bridge."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    # A page that posted to an absolute URL would bypass the bridge entirely,
    # skipping auth and leaving every turn missing from the dashboard.
    assert "/v1/chat/completions" in body
    assert "http://127.0.0.1:" not in body


# --------------------------------------------------------------------------
# `mtplx splash` — the seam the native app's Settings drives
# --------------------------------------------------------------------------


def _run_splash(*args):
    import subprocess, sys as _sys
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[1]
    return subprocess.run(
        [_sys.executable, "-c",
         "import sys; from mtplx.cli import main; sys.argv=['mtplx']+sys.argv[1:]; "
         "sys.exit(main())", "splash", *args],
        capture_output=True, text=True, cwd=repo, timeout=120,
    )


def test_splash_list_json_is_machine_readable():
    done = _run_splash("list", "--json")
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    assert payload["ok"] is True
    assert payload["engine"] == "splash"
    ids = {p["id"] for p in payload["packages"]}
    assert MODEL in ids
    for package in payload["packages"]:
        # The Swift Package DTO requires exactly these keys.
        assert set(package) >= {"id", "installed", "path", "download_bytes"}
        assert isinstance(package["installed"], bool)


def test_splash_install_refuses_an_unknown_package():
    done = _run_splash("install", "--model", "bogus/not-a-package")
    assert done.returncode == 2
    assert "not a Splash package" in done.stderr


def test_splash_install_is_idempotent_for_an_installed_package():
    """Selecting an already-verified package must not re-download 17 GB."""
    listing = json.loads(_run_splash("list", "--json").stdout.strip().splitlines()[-1])
    installed = [p["id"] for p in listing["packages"] if p["installed"]]
    if not installed:
        pytest.skip("no Splash package installed on this machine")
    done = _run_splash("install", "--model", installed[0], "--json")
    assert done.returncode == 0, done.stderr
    payload = json.loads(done.stdout.strip().splitlines()[-1])
    assert payload["installed"] is True
    assert payload["changed"] is False, "an installed package must not be refetched"


def test_engine_argv_is_all_strings_whatever_the_caller_passes():
    """The native app always sends --context-window, which argparse types int.

    An int in the engine's argv raised TypeError before Splash even spawned,
    so the daemon died at launch from the app while every hand-run (which
    never passed the flag) worked.
    """
    install = SplashInstall.discover()
    engine = SplashEngine(install, MODEL, port=1, max_context=131072, max_memory=None)
    assert engine.max_context == "131072"
    assert engine.max_memory == "auto"
    assert SplashEngine(install, MODEL, port=1, max_context=0).max_context == "auto"
    assert SplashEngine(install, MODEL, port=1, max_context="100K").max_context == "100K"
    assert SplashEngine(install, MODEL, port=1, max_memory="28G").max_memory == "28G"


def test_health_echoes_the_apps_launch_id(fake_splash, monkeypatch):
    """The app rejects a daemon that cannot prove it is the one just launched.

    DaemonSupervisor sets MTPLX_APP_LAUNCH_ID and compares it to
    /health -> startup.launch_id; a missing block read as a stale foreign
    daemon and surfaced as "startup didn't match what we expected" even though
    the engine was up and serving.
    """
    monkeypatch.setenv("MTPLX_APP_LAUNCH_ID", "launch-abc123")
    monkeypatch.setenv("MTPLX_APP_PARENT_PID", "4242")
    install = SplashInstall.discover()
    engine = _StubEngine(install, MODEL, port=fake_splash)
    engine.start()
    app = create_app(install=install, engine=engine, telemetry=BridgeTelemetry(MODEL))
    with TestClient(app) as http:
        startup = http.get("/health").json()["startup"]
    assert startup["launch_id"] == "launch-abc123"
    assert startup["app_parent_pid"] == 4242
    assert startup["model_id"] == MODEL
    assert startup["pid"] > 0
    assert startup["model_controls"]["backend_id"] == "splash"
    assert startup["model_controls"]["kv_quant"]["supported"] is False


# --------------------------------------------------------------------------
# dashboard telemetry
#
# The app's hero tiles follow the metrics stream's `progress` / `completed`
# events, the min/max/mean/p95 row comes from RollingMetrics, and the request
# table from the last completion envelopes. The bridge drives the MLX server's
# own DashboardState, so these tests pin behaviour, not a copied shape.
# --------------------------------------------------------------------------


def _status(*, prefill_tokens=0, prefill_ms=0.0, decode_tokens=0, decode_ms=0.0,
            drafted=0, accepted=0, reused=0, **extra):
    """A Splash /status with its cumulative counters at the given values."""
    return {
        **FAKE_STATUS,
        "cache": {**FAKE_STATUS["cache"], "reused_tokens": reused},
        "memory_actual": {
            "dense_bytes": 18_720_882_688,
            "sparse_resident_bytes": 136_314_880,
            "current_bytes": 18_858_983_424,
            "peak_bytes": 19_909_656_576,
        },
        "metrics": {
            "prefill_input_tokens": prefill_tokens,
            "prefill_wall_ms": prefill_ms,
            "decode_output_tokens": decode_tokens,
            "decode_wall_ms": decode_ms,
            "drafted_tokens": drafted,
            "accepted_draft_tokens": accepted,
            "prefill_tokens_per_second": 0,
            "decode_tokens_per_second": 0,
        },
        **extra,
    }


def _complete(telemetry, *, before, after, prompt=None, completion=None, frames=0):
    trace = telemetry.begin("hi", status=before)
    for _ in range(frames):
        telemetry.token(trace)
    return telemetry.finish(
        trace, status=after, prompt_tokens=prompt, completion_tokens=completion
    )


def test_request_speeds_are_the_engines_counters_across_the_request():
    """Splash's counters are cumulative; a request's speed is their difference.

    An earlier cut divided tokens by wall time from request start, which
    folded time-to-first-token into "decode speed" and read ~25 tok/s for an
    engine decoding far faster.
    """
    telemetry = BridgeTelemetry(MODEL)
    envelope = _complete(
        telemetry,
        before=_status(prefill_tokens=100, prefill_ms=1000, decode_tokens=500, decode_ms=10_000),
        after=_status(prefill_tokens=2100, prefill_ms=2000, decode_tokens=700, decode_ms=12_500),
        prompt=2000,
        completion=200,
    )
    assert envelope["decode_tok_s"] == pytest.approx(200 / 2.5)      # 80 tok/s
    assert envelope["prefill_tok_s"] == pytest.approx(2000 / 1.0)    # 2000 tok/s
    assert envelope["new_prefill_tokens"] == 2000
    assert envelope["decode_elapsed_s"] == pytest.approx(2.5)
    assert envelope["generation_mode"] == "splash"


def test_draft_acceptance_is_per_request_and_labelled_dflash2():
    telemetry = BridgeTelemetry(MODEL)
    envelope = _complete(
        telemetry,
        before=_status(drafted=1000, accepted=400, decode_tokens=10, decode_ms=100),
        after=_status(drafted=1200, accepted=550, decode_tokens=110, decode_ms=1100),
        completion=100,
    )
    assert envelope["draft"] == "dflash2"
    assert envelope["drafted_tokens"] == 200
    assert envelope["accepted_drafts"] == 150
    assert envelope["draft_acceptance_rate"] == pytest.approx(0.75)


def test_rolling_row_tracks_min_max_mean_like_the_mlx_engine():
    """The dashboard's peak / min / mean / p95 row reads RollingMetrics."""
    telemetry = BridgeTelemetry(MODEL)
    for tokens, ms in ((100, 2000), (100, 1000), (100, 4000)):   # 50, 100, 25 tok/s
        _complete(
            telemetry,
            before=_status(),
            after=_status(decode_tokens=tokens, decode_ms=ms),
            completion=tokens,
        )
    rolling = telemetry.snapshot(_status(), profile={}, machine={})["rolling"]
    assert rolling["count"] == 3
    assert rolling["max"] == pytest.approx(100.0)
    assert rolling["min"] == pytest.approx(25.0)
    assert rolling["mean"] == pytest.approx((50 + 100 + 25) / 3)
    assert rolling["sticky_all_time_max"] == pytest.approx(100.0)
    assert rolling["p50"] is not None and rolling["p95"] is not None
    assert len(rolling["history"]) == 3


def _drain(queue):
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def test_completion_publishes_the_events_the_hero_tiles_follow():
    telemetry = BridgeTelemetry(MODEL)
    queue = telemetry.dashboard.bus.subscribe()
    _complete(
        telemetry,
        before=_status(),
        after=_status(decode_tokens=90, decode_ms=1000, prefill_tokens=30, prefill_ms=100),
        prompt=30,
        completion=90,
    )
    events = _drain(queue)
    kinds = [event["kind"] for event in events]
    assert "completed" in kinds
    completed = next(e for e in events if e["kind"] == "completed")
    # The app reads payload["envelope"], then envelope["decode_tok_s"].
    assert completed["envelope"]["decode_tok_s"] == pytest.approx(90.0)
    assert completed["envelope"]["prefill_tok_s"] == pytest.approx(300.0)
    assert completed["envelope"]["request_id"].startswith("splash-")
    # First completion is, by definition, the fastest so far.
    new_max = next(e for e in events if e["kind"] == "new_max_tps")
    assert new_max["tok_s"] == pytest.approx(90.0)


def test_streaming_publishes_live_progress_with_a_decode_rate(monkeypatch):
    from mtplx.server.splash_bridge import telemetry as telemetry_module

    monkeypatch.setattr(telemetry_module, "PROGRESS_INTERVAL_S", 0.0)
    telemetry = BridgeTelemetry(MODEL)
    queue = telemetry.dashboard.bus.subscribe()
    trace = telemetry.begin("hi", status=_status())
    telemetry.token(trace)
    time.sleep(0.12)
    telemetry.token(trace)
    telemetry.token(trace)
    progress = [e for e in _drain(queue) if e["kind"] == "progress"]
    assert progress, "a stream must move the live gauge before it completes"
    latest = progress[-1]
    assert latest["request_id"] == trace.request_id
    # The app reads payload["progress"], then progress["decode_tok_s"].
    assert latest["progress"]["decode_tok_s"] > 0
    assert latest["progress"]["completion_tokens"] == 3
    assert telemetry.active_requests == 1
    telemetry.finish(trace, status=_status())
    assert telemetry.active_requests == 0


def test_recent_holds_request_envelopes_newest_last():
    telemetry = BridgeTelemetry(MODEL)
    idle = telemetry.snapshot(_status(), profile={}, machine={})["recent"]
    assert len(idle) == 1 and "request_id" not in idle[0], "engine record until a request completes"
    for tokens in (10, 20):
        _complete(telemetry, before=_status(),
                  after=_status(decode_tokens=tokens, decode_ms=1000), completion=tokens)
    recent = telemetry.snapshot(_status(), profile={}, machine={})["recent"]
    assert [row["completion_tokens"] for row in recent] == [10, 20]


def test_prefill_history_gets_one_row_per_request(client):
    client.post("/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}]})
    body = client.get("/v1/mtplx/prefill_history").json()
    assert body["capacity"] > 0
    assert len(body["history"]) == 1
    assert body["history"][0]["prompt_tokens"] == 11


def test_cancel_trips_the_requests_flag():
    telemetry = BridgeTelemetry(MODEL)
    trace = telemetry.begin("hi", status=_status())
    assert telemetry.cancel("splash-does-not-exist") is False
    assert telemetry.cancel(trace.request_id) is True
    assert trace.cancelled, "the proxy loop watches this to close the upstream"
    envelope = telemetry.finish(trace, status=_status())
    assert envelope["cancelled"] is True
    lifetime = telemetry.snapshot(_status(), profile={}, machine={})["lifetime"]
    assert lifetime["cancelled_total"] == 1


def test_memory_reports_the_process_not_just_the_kv_pages():
    """Reading kv.resident_backing_bytes alone showed 0.14 GB of an 18.9 GB engine."""
    mem = BridgeTelemetry(MODEL).snapshot(_status(), profile={}, machine={})["mem"]
    assert mem["active_memory_bytes"] == 18_858_983_424
    assert mem["peak_memory_bytes"] == 19_909_656_576
    assert mem["model_weights_bytes"] == 18_720_882_688
    assert mem["cache_memory_bytes"] == 136_314_880
    assert mem["cache_memory_bytes"] < mem["active_memory_bytes"] / 10


# --------------------------------------------------------------------------
# the tiles that read optional snapshot blocks: Avg Prefill, Cached, Context
# --------------------------------------------------------------------------


def test_avg_prefill_tile_is_fed_by_prefill_rates():
    """The tile reads snapshot.prefill_rates, not the per-request envelopes."""
    telemetry = BridgeTelemetry(MODEL)
    assert telemetry.snapshot(_status(), profile={}, machine={})["prefill_rates"]["samples"] == 0
    _complete(telemetry, before=_status(),
              after=_status(prefill_tokens=2000, prefill_ms=1000, decode_tokens=10, decode_ms=100),
              prompt=2000, completion=10)
    _complete(telemetry, before=_status(),
              after=_status(prefill_tokens=1000, prefill_ms=1000, decode_tokens=10, decode_ms=100),
              prompt=1000, completion=10)
    rates = telemetry.snapshot(_status(), profile={}, machine={})["prefill_rates"]
    assert rates["samples"] == 2
    assert rates["tokens"] == 3000
    assert rates["compute_time_s"] == pytest.approx(2.0)     # app shows 3000/2 = 1500 tok/s
    assert rates["peak_tok_s"] == pytest.approx(2000.0)


def test_cached_tokens_do_not_inflate_the_prefill_average():
    """A warm turn computes few tokens; only those count as prefill work."""
    telemetry = BridgeTelemetry(MODEL)
    trace = telemetry.begin("hi", status=_status())
    telemetry.finish(
        trace,
        # 1934-token prompt, 1856 of it reused: the engine prefilled only 78.
        status=_status(prefill_tokens=78, prefill_ms=500, decode_tokens=40, decode_ms=800, reused=1856),
        prompt_tokens=1934, completion_tokens=40, cached_tokens=1856,
    )
    snapshot = telemetry.snapshot(_status(), profile={}, machine={})
    assert snapshot["prefill_rates"]["tokens"] == 78
    newest = snapshot["recent"][-1]
    assert newest["cached_tokens"] == 1856
    assert newest["session_cache_hit"] is True, "the Cached tile's caption reads this key"
    assert snapshot["latest"] == newest
    assert snapshot["lifetime"]["cached_tokens_total"] == 1856


def test_turns_of_one_conversation_share_a_session_row():
    from mtplx.server.splash_bridge.telemetry import conversation_key

    turn1 = {"messages": [{"role": "user", "content": "explain paging"}]}
    turn2 = {"messages": turn1["messages"] + [
        {"role": "assistant", "content": "..."}, {"role": "user", "content": "more"}]}
    other = {"messages": [{"role": "user", "content": "a different chat"}]}
    assert conversation_key(turn1) == conversation_key(turn2)
    assert conversation_key(turn1) != conversation_key(other)
    assert conversation_key({"session_id": "abc", **turn1}) == "abc"

    telemetry = BridgeTelemetry(MODEL)
    for payload, prompt in ((turn1, 100), (turn2, 180)):
        trace = telemetry.begin("x", status=_status(), session_id=conversation_key(payload))
        telemetry.finish(trace, status=_status(decode_tokens=20, decode_ms=400),
                         prompt_tokens=prompt, completion_tokens=20)
    sessions = telemetry.sessions(_status())
    assert sessions["count"] == 1
    assert sessions["sessions"][0]["prefix_len"] == 200      # the latest turn's context
    assert sessions["sessions"][0]["bytes"] > 0


def test_sessions_are_not_claimed_once_the_engine_holds_no_pages():
    telemetry = BridgeTelemetry(MODEL)
    trace = telemetry.begin("x", status=_status(), session_id="s1")
    telemetry.finish(trace, status=_status(decode_tokens=5, decode_ms=100),
                     prompt_tokens=50, completion_tokens=5)
    evicted = {**_status(), "kv": {**FAKE_STATUS["kv"], "pages_cache": 0, "pages_active": 0}}
    assert telemetry.sessions(evicted) == {"sessions": [], "count": 0}


def test_utility_calls_are_not_counted_as_chat_requests(client):
    assert client.post("/tokenize", json={"content": "hi"}).status_code == 200
    assert client.post("/apply-template",
                       json={"messages": [{"role": "user", "content": "hi"}]}).status_code == 200
    snapshot = client.get("/v1/mtplx/snapshot").json()
    assert snapshot["lifetime"]["requests_total"] == 0
    assert all("request_id" not in row for row in snapshot["recent"])


def test_prompt_is_sized_beside_the_request_for_the_context_tile(client):
    """The in-flight row needs prompt_tokens for the Context tile to move."""
    import threading

    seen = {}

    def slow_call():
        client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "slow"}]})

    worker = threading.Thread(target=slow_call, daemon=True)
    worker.start()
    deadline = time.monotonic() + 1.3
    while time.monotonic() < deadline and not seen:
        for row in client.get("/v1/mtplx/snapshot").json()["in_flight"]:
            if row.get("prompt_tokens"):
                seen = row
        time.sleep(0.05)
    worker.join(timeout=10)
    assert seen.get("prompt_tokens") == 11, "counted via /apply-template + /tokenize"


def test_non_streamed_requests_still_report_time_to_first_token():
    """No frames arrive to time, so use the engine's prefill time instead."""
    telemetry = BridgeTelemetry(MODEL)
    envelope = _complete(
        telemetry, before=_status(),
        after=_status(prefill_tokens=500, prefill_ms=2500, decode_tokens=20, decode_ms=400),
        prompt=500, completion=20, frames=0,
    )
    assert envelope["ttft_s"] == pytest.approx(2.5)
