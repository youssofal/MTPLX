# SPDX-License-Identifier: Apache-2.0
"""The MTPLX contract, served over a Splash engine.

Every client MTPLX already has — the native app, the web dashboard, OpenCode,
any OpenAI or Anthropic client — talks to the endpoints below. This app
implements them against a supervised Splash process, so selecting the Splash
engine changes which kernels run and nothing about how clients talk to it.

Inference endpoints are proxied to Splash, which speaks OpenAI Chat
Completions, OpenAI Responses and Anthropic Messages natively. Dashboard
endpoints are translated from Splash's /status by `telemetry`. Endpoints that
describe MLX-only machinery answer honestly that the feature is absent rather
than pretending.

Streaming is a plain blocking generator handed to StreamingResponse: Starlette
runs a sync iterator on a worker thread, which lets the proxy use the standard
library and keeps the bridge free of a new HTTP dependency.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)

from mtplx.server.chat_page import chat_ui_html

from .install import SplashInstall
from .supervisor import SplashEngine
from .settings import (
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    TOP_K_MAX,
    TOP_K_MIN,
    TOP_P_MIN,
    SamplingSettings,
)
from .telemetry import BridgeTelemetry, conversation_key

PROXY_TIMEOUT_S = 1800.0
# How often the chat stream carries a live mtplx_progress frame.
PROGRESS_FRAME_S = 0.2
# The draft length both official packages compile in; shown only if a
# package's manifest does not state its own.
DEFAULT_DRAFT_TOKENS = 7


def splash_engine_ui(draft_tokens: int) -> dict[str, Any]:
    """The chat page's engine slots, filled for Splash.

    Same page and layout as the MLX engine's. The speculative controls show
    what Splash runs and stay fixed, and the sampling sliders cover exactly
    the range Splash accepts, so no setting on the page can fail a request.
    """
    return {
        "speculative_label": "DFlash 2",
        "depth_label": "Draft tokens",
        "depth_help": (
            "Splash always speculates: its trained DFlash 2 draft proposes "
            f"{draft_tokens} tokens per verify."
        ),
        "top_p_min": TOP_P_MIN,
        "top_k_min": TOP_K_MIN,
        "top_k_max": TOP_K_MAX,
        "top_k_help": f"Splash samples from the top {TOP_K_MIN}&ndash;{TOP_K_MAX} tokens.",
        "presence_help": "Splash samples without penalties.",
        "locked_controls": {
            "mtp_enabled": True,
            "depth": draft_tokens,
            "presence_penalty": 0,
        },
    }


@functools.lru_cache(maxsize=1)
def _machine_info() -> dict[str, Any]:
    """Cached: this is fixed hardware, and the SSE stream asks twice a tick."""

    def sysctl(name: str) -> str:
        try:
            done = subprocess.run(
                ["sysctl", "-n", name], capture_output=True, text=True, timeout=5
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return done.stdout.strip()

    info: dict[str, Any] = {}
    if chip := sysctl("machdep.cpu.brand_string"):
        info["chip"] = chip
    if model := sysctl("hw.model"):
        info["machine_model"] = model
    try:
        info["unified_memory_bytes"] = int(sysctl("hw.memsize"))
    except ValueError:
        pass
    if not info:
        info["chip"] = platform.processor() or "unknown"
    return info


def _int_env(name: str) -> int | None:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return None


def _prompt_preview(payload: dict[str, Any]) -> str:
    """A short human label for the dashboard's in-flight row."""
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                return content[:200]
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        return part["text"][:200]
    prompt = payload.get("prompt")
    return prompt[:200] if isinstance(prompt, str) else ""


def _has_text_delta(event: dict[str, Any]) -> bool:
    """True when a stream frame actually carried generated text.

    Role frames, finish frames and Anthropic's ping/start frames arrive as
    events with no content; counting them as tokens inflates the tokens/s the
    dashboard shows at the start of every turn.
    """
    choices = event.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            # Thinking tokens are decoded like any other; skipping them left
            # the live gauge at zero for the whole reasoning phase.
            if isinstance(delta, dict) and (
                delta.get("content") or delta.get("reasoning_content")
            ):
                return True
    delta = event.get("delta")  # Anthropic content_block_delta
    if isinstance(delta, dict) and (delta.get("text") or delta.get("partial_json")):
        return True
    return False


def _is_reasoning_delta(event: dict[str, Any]) -> bool:
    choices = event.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        delta = choices[0].get("delta")
        return isinstance(delta, dict) and bool(delta.get("reasoning_content"))
    return False


def _finish_reason(event: dict[str, Any]) -> str | None:
    choices = event.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return reason if isinstance(reason, str) else None
    return None


def _is_usage_only(event: dict[str, Any]) -> bool:
    """The trailing `choices: []` frame that carries only a usage block."""
    return event.get("choices") == [] and isinstance(event.get("usage"), dict)


def _usage(obj: Any) -> tuple[int, int, int]:
    """(prompt, completion, cached) from an OpenAI or Anthropic usage block."""
    if not isinstance(obj, dict):
        return 0, 0, 0
    usage = obj.get("usage")
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None and completion is None:  # Anthropic spelling
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if cached is None:
        cached = usage.get("cache_read_input_tokens")
    return (
        int(prompt or 0),
        int(completion or 0),
        int(cached or 0),
    )


def create_app(
    *,
    install: SplashInstall,
    engine: SplashEngine,
    telemetry: BridgeTelemetry,
    api_key: str | None = None,
    sampling: SamplingSettings | None = None,
) -> FastAPI:

    # The caller starts and stops the engine: a server that quietly booted one
    # on first request would hide a failed load behind an unexplained 503.
    app = FastAPI(title="MTPLX (Splash engine)", version="1")
    started_at = time.time()
    # Resolved once: /health is polled continuously by the app's liveness
    # probe, and this shells out to `splash --version`.
    splash_version = install.version()
    # A package's files never change while it is installed, and the chat page
    # polls settings every 1.5 s: read each package's template and manifest
    # once.
    efforts_for = functools.lru_cache(maxsize=8)(install.reasoning_efforts)
    draft_tokens_for = functools.lru_cache(maxsize=8)(install.draft_tokens)
    sampling = sampling or SamplingSettings(
        efforts=lambda: efforts_for(engine.model_id)
    )

    # -- helpers ----------------------------------------------------------

    def _authorize(request: Request) -> None:
        if not api_key:
            return
        header = request.headers.get("authorization", "")
        presented = header[7:] if header.lower().startswith("bearer ") else ""
        if not presented:
            presented = request.headers.get("x-api-key", "")
        if presented != api_key:
            raise HTTPException(status_code=401, detail="invalid API key")

    def _upstream_headers() -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if engine.api_key:
            headers["Authorization"] = f"Bearer {engine.api_key}"
        return headers

    def _require_ready() -> None:
        if engine.is_ready():
            return
        phase = engine.state.phase
        if phase in {"installing", "starting"}:
            raise HTTPException(
                status_code=503,
                detail=f"Splash is {phase}: {engine.state.detail}",
            )
        raise HTTPException(
            status_code=503,
            detail=engine.state.error or "no model is loaded; POST /v1/mtplx/engine/load",
        )

    def _model_controls() -> dict[str, Any]:
        """What this model's controls can do, in the app's ModelControls shape.

        Settings reads `model_controls.kv_quant` before any other source, so
        this is where the fixed 8-bit KV width is declared to the UI.
        """
        return {
            "schema_version": 1,
            "model_ref": engine.model_id,
            "model_family": "qwen3_8" if "3.8" in engine.model_id else "qwen3_6",
            "backend_id": "splash",
            "support_level": "verified",
            "display_name": engine.model_id.split("/")[-1],
            "draft_control": {"supported": False},
            "reasoning": sampling.reasoning_policy(),
            "sampling": {
                "temperature": DEFAULT_TEMPERATURE,
                "top_p": DEFAULT_TOP_P,
                "top_k": DEFAULT_TOP_K,
                "family_default_reason": "Splash engine defaults",
            },
            "kv_quant": telemetry.kv_quant_policy(),
            "context_window": {"supported": False, "source": "engine"},
        }

    def _profile() -> dict[str, Any]:
        return {
            "name": "splash",
            "label": "Splash",
            "engine": "splash",
            "model_id": engine.model_id,
            "speculative_decoding": "dflash2",
            "kv_quantization": "q8",
        }

    # -- inference proxy ---------------------------------------------------

    def _engine_post(path: str, body: dict[str, Any], timeout: float) -> Any:
        request = urllib.request.Request(
            engine.base_url + path,
            data=json.dumps(body).encode(),
            headers=_upstream_headers(),
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def _count_prompt_tokens(payload: dict[str, Any], entry: Any) -> None:
        """Render and tokenize the prompt with Splash's own template."""
        try:
            payload.pop("stream", None)
            payload.pop("stream_options", None)
            rendered = _engine_post("/apply-template", payload, timeout=60)
            tokens = _engine_post(
                "/tokenize", {"content": rendered.get("prompt", "")}, timeout=60
            )
            counted = tokens.get("tokens")
            if isinstance(counted, list):
                telemetry.set_prompt_tokens(entry, len(counted))
        except Exception:  # a tile's figure must never disturb a generation
            return

    def _forward(path: str, payload: dict[str, Any]) -> Response:
        """Relay a utility call untracked.

        /tokenize and /apply-template run no model. Routing them through the
        tracked proxy counted each one as a chat request, polluting the
        lifetime totals and the recent-requests table.
        """
        _require_ready()
        try:
            body = _engine_post(path, payload, timeout=120)
        except urllib.error.HTTPError as error:
            return Response(content=error.read(), status_code=int(error.code),
                            media_type="application/json")
        except (OSError, urllib.error.URLError) as error:
            raise HTTPException(status_code=502, detail=f"Splash: {error}") from None
        return JSONResponse(body)

    def _proxy(path: str, payload: dict[str, Any]) -> Response:
        """Blocking: always reached through `run_in_threadpool`."""
        _require_ready()
        # The live settings fill whatever the client left out, as the MLX
        # server does: the app's own chat sends no sampling fields at all.
        payload = sampling.apply(path, payload)
        # Splash's counters are cumulative, so a reading either side of the
        # request yields the engine's own per-request speeds and acceptance.
        entry = telemetry.begin(
            _prompt_preview(payload),
            status=engine.status(),
            session_id=conversation_key(payload),
        )
        # Size the prompt beside the request, never in front of it: the count
        # costs CPU proportional to prompt length and only feeds a tile.
        if path == "/v1/chat/completions":
            threading.Thread(
                target=_count_prompt_tokens,
                args=(dict(payload), entry),
                name="splash-prompt-count",
                daemon=True,
            ).start()
        url = engine.base_url + path

        # Splash sends no usage on a stream unless asked, which left prompt
        # tokens unknown for every streamed turn. Ask for it when the client
        # did not, and drop the extra usage-only frame on the way out so the
        # client's stream is byte-for-byte what it would have been.
        options = payload.get("stream_options")
        swallow_usage_frame = False
        if payload.get("stream") and path == "/v1/chat/completions":
            if not isinstance(options, dict) or "include_usage" not in options:
                payload = dict(payload)
                payload["stream_options"] = {
                    **(options if isinstance(options, dict) else {}),
                    "include_usage": True,
                }
                swallow_usage_frame = True

        body = json.dumps(payload).encode()
        upstream = urllib.request.Request(
            url, data=body, headers=_upstream_headers(), method="POST"
        )

        if not payload.get("stream"):
            try:
                with urllib.request.urlopen(upstream, timeout=PROXY_TIMEOUT_S) as response:
                    raw = response.read()
                    code = int(response.status)
            except urllib.error.HTTPError as error:
                telemetry.finish(entry, status=engine.status(), cancelled=True)
                return Response(
                    content=error.read(),
                    status_code=int(error.code),
                    media_type="application/json",
                )
            except (OSError, urllib.error.URLError) as error:
                telemetry.finish(entry, status=engine.status(), cancelled=True)
                raise HTTPException(status_code=502, detail=f"Splash: {error}") from None
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            prompt, completion, cached = _usage(parsed)
            telemetry.finish(
                entry,
                status=engine.status(),
                prompt_tokens=prompt or None,
                completion_tokens=completion or None,
                cached_tokens=cached,
            )
            return Response(content=raw, status_code=code, media_type="application/json")

        # The MLX engine decorates its chat stream with `mtplx_progress`
        # frames and an `mtplx_stats` + usage block on the finish frame; the
        # app's chat reads both for its live tok/s chip and per-reply footer.
        # Splash sends neither, so the bridge adds them.
        decorate = path == "/v1/chat/completions"

        def progress_frame(upstream_id: str, created: Any, model: Any) -> bytes:
            now = time.perf_counter()
            first = entry.first_token_perf or now
            decode_s = now - first
            progress: dict[str, Any] = {
                "phase": "generating",
                "completion_tokens": entry.frames,
                "decode_elapsed_s": decode_s,
                "elapsed_s": now - entry.started_perf,
                "ttft_s": first - entry.started_perf,
                "generation_mode": "splash",
            }
            if decode_s > 0.05 and entry.frames > 1:
                progress["decode_tok_s"] = (entry.frames - 1) / decode_s
            frame = {
                "id": upstream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
                "mtplx_progress": progress,
            }
            return f"data: {json.dumps(frame)}\n\n".encode()

        def stream() -> Iterator[bytes]:
            prompt = completion = cached = 0
            cancelled = False
            finished = False
            held_finish: dict[str, Any] | None = None
            held_usage: bytes | None = None
            # Splash streams one token per frame, so thinking frames count
            # the thinking tokens (less the two delimiters it never emits).
            reasoning_frames = 0
            upstream_id: str | None = None
            last_progress = 0.0

            def close_out() -> dict[str, Any]:
                nonlocal finished
                finished = True
                return telemetry.finish(
                    entry,
                    status=engine.status(),
                    prompt_tokens=prompt or None,
                    completion_tokens=completion or None,
                    cached_tokens=cached,
                    cancelled=cancelled,
                )

            try:
                with urllib.request.urlopen(upstream, timeout=PROXY_TIMEOUT_S) as response:
                    for line in response:
                        if entry.cancelled:
                            # POST /v1/mtplx/cancel tripped the flag. Leaving
                            # the `with` closes the upstream socket, which is
                            # how Splash learns the client is gone.
                            cancelled = True
                            break
                        if line.startswith(b"data:"):
                            chunk = line[5:].strip()
                            if decorate and chunk == b"[DONE]" and held_finish is None:
                                if held_usage is not None:
                                    yield held_usage
                                    held_usage = None
                            if decorate and chunk == b"[DONE]" and held_finish is not None:
                                # Every frame is in: stamp the held finish
                                # frame with this request's numbers.
                                envelope = close_out()
                                event = held_finish
                                if "usage" not in event:
                                    event["usage"] = {
                                        "prompt_tokens": envelope["prompt_tokens"],
                                        "completion_tokens": envelope["completion_tokens"],
                                        "total_tokens": envelope["prompt_tokens"]
                                        + envelope["completion_tokens"],
                                        "prompt_tokens_details": {
                                            "cached_tokens": envelope["cached_tokens"]
                                        },
                                    }
                                stats = telemetry.chat_stats(envelope)
                                if reasoning_frames:
                                    stats["reasoning_tokens"] = reasoning_frames
                                event["mtplx_stats"] = stats
                                yield f"data: {json.dumps(event)}\n\n".encode()
                                held_finish = None
                                if held_usage is not None:
                                    yield held_usage
                                    held_usage = None
                                yield line
                                continue
                            if chunk and chunk != b"[DONE]":
                                try:
                                    event = json.loads(chunk)
                                except ValueError:
                                    event = None
                                if isinstance(event, dict):
                                    if upstream_id is None and isinstance(event.get("id"), str):
                                        upstream_id = event["id"]
                                        telemetry.alias(entry, upstream_id)
                                    text = _has_text_delta(event)
                                    if _is_reasoning_delta(event):
                                        reasoning_frames += 1
                                    if text:
                                        telemetry.token(entry)
                                    got = _usage(event)
                                    if any(got):
                                        prompt, completion, cached = got
                                    if swallow_usage_frame and _is_usage_only(event):
                                        continue  # we asked for it, not the client
                                    if decorate and _is_usage_only(event):
                                        held_usage = line + b"\n"
                                        continue
                                    if decorate and _finish_reason(event):
                                        held_finish = event
                                        continue
                                    if decorate and text:
                                        yield line
                                        now = time.perf_counter()
                                        if now - last_progress >= PROGRESS_FRAME_S:
                                            last_progress = now
                                            yield b"\n" + progress_frame(
                                                upstream_id or entry.request_id,
                                                event.get("created"),
                                                event.get("model"),
                                            )
                                        continue
                        yield line
                # Upstream closed without [DONE]; still deliver what was held.
                if held_finish is not None:
                    yield f"data: {json.dumps(held_finish)}\n\n".encode()
                if held_usage is not None:
                    yield held_usage
            except urllib.error.HTTPError as error:
                yield b"data: " + error.read() + b"\n\n"
            except (OSError, urllib.error.URLError) as error:
                message = json.dumps({"error": {"message": f"Splash: {error}"}})
                yield f"data: {message}\n\n".encode()
            except GeneratorExit:
                # The client hung up mid-stream.
                cancelled = True
                raise
            finally:
                if not finished:
                    close_out()

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def _json_body(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(status_code=400, detail="body must be JSON") from None
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        return payload

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        _authorize(request)
        payload = await _json_body(request)
        return await run_in_threadpool(_proxy, "/v1/chat/completions", payload)

    @app.post("/v1/responses")
    async def responses(request: Request) -> Response:
        _authorize(request)
        payload = await _json_body(request)
        return await run_in_threadpool(_proxy, "/v1/responses", payload)

    @app.post("/v1/messages")
    async def messages(request: Request) -> Response:
        _authorize(request)
        payload = await _json_body(request)
        return await run_in_threadpool(_proxy, "/v1/messages", payload)

    @app.post("/tokenize")
    async def tokenize(request: Request) -> Response:
        _authorize(request)
        payload = await _json_body(request)
        return await run_in_threadpool(_forward, "/tokenize", payload)

    @app.post("/apply-template")
    async def apply_template(request: Request) -> Response:
        _authorize(request)
        payload = await _json_body(request)
        return await run_in_threadpool(_forward, "/apply-template", payload)

    # -- identity and health ----------------------------------------------

    @app.get("/health")
    def health() -> dict[str, Any]:
        status = engine.status()
        return {
            "ok": engine.is_ready(),
            "model": engine.model_id,
            "model_path": str(install.package_root(engine.model_id)),
            "generation_mode": "splash",
            # Splash speculates with its DFlash 2 draft on every decode, but
            # that is not MTPLX's MTP: report the MTP fields as absent.
            "load_mtp": False,
            "mtp_enabled": False,
            "depth": 0,
            "profile": _profile(),
            "context_window": telemetry.context_window(status),
            "active_requests": telemetry.active_requests,
            "reasoning_parser": "qwen3",
            "engine": "splash",
            "engine_phase": engine.state.phase,
            "engine_detail": engine.state.detail,
            "engine_error": engine.state.error,
            "splash_version": splash_version,
            # The native app stamps every launch with MTPLX_APP_LAUNCH_ID and
            # refuses a daemon whose /health does not echo it back: that is
            # how it tells the process it just started from a stale one that
            # happens to hold the port. Without this block the engine came up
            # fine and the app still reported "startup didn't match".
            "startup": {
                "launch_id": os.environ.get("MTPLX_APP_LAUNCH_ID"),
                "pid": os.getpid(),
                "app_parent_pid": _int_env("MTPLX_APP_PARENT_PID"),
                "started_at": started_at,
                "model_id": engine.model_id,
                "api_key_required": bool(api_key),
                "paged_kv_quantization": "q8",
                "model_controls": _model_controls(),
            },
            "model_controls": _model_controls(),
            "chip_name": _machine_info().get("chip"),
            "unified_memory_bytes": _machine_info().get("unified_memory_bytes"),
            "speculative_decoding": "dflash2",
            # The chat page's runtime pill reads this before anything else.
            "runtime_mode": "Splash · DFlash 2",
        }

    @app.get("/", response_class=HTMLResponse)
    def chat_page(request: Request) -> HTMLResponse:
        """The MTPLX chat page, the one the MLX engine serves, set for Splash.

        The bridge starts the engine with `--no-webui`, so this is the only
        chat page. It posts to this server's relative routes, so every turn
        passes through the bridge and lands in the dashboard's telemetry.
        """
        settings = sampling.payload()
        draft = draft_tokens_for(engine.model_id) or DEFAULT_DRAFT_TOKENS
        return HTMLResponse(
            chat_ui_html(
                model_id=engine.model_id,
                server_url=str(request.base_url).rstrip("/"),
                api_key_required=bool(api_key),
                default_settings={
                    "temperature": settings["temperature"],
                    "top_p": settings["top_p"],
                    "top_k": settings["top_k"],
                    "presence_penalty": 0.0,
                    "depth": draft,
                    "depth_max": draft,
                    "mtp_enabled": True,
                    "max_tokens": settings["max_response_tokens"] or 16384,
                    "reasoning": settings["reasoning"],
                    "system": "",
                },
                engine_ui=splash_engine_ui(draft),
            )
        )

    @app.get("/ready")
    def ready() -> Response:
        ok = engine.is_ready()
        return JSONResponse(
            {"status": "ready" if ok else "unavailable"}, status_code=200 if ok else 503
        )

    @app.get("/v1/models")
    def list_models(capability: str | None = None) -> dict[str, Any]:
        now = int(time.time())
        return {
            "object": "list",
            "id": engine.model_id,
            "data": [
                {
                    "id": engine.model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "splash",
                    "capability": "chat",
                }
            ],
        }

    # -- dashboard contract -------------------------------------------------

    @app.get("/v1/mtplx/snapshot")
    def snapshot() -> dict[str, Any]:
        return telemetry.snapshot(
            engine.status(), profile=_profile(), machine=_machine_info()
        )

    @app.get("/v1/mtplx/metrics/stream")
    async def metrics_stream(snapshot_interval_ms: int | None = None) -> StreamingResponse:
        """Snapshots on a cadence, with bus events delivered as they happen.

        The app's hero tiles follow `progress` and `completed`, not snapshots,
        so a snapshot-only stream left them blank however complete each
        snapshot was. This mirrors the MLX server's stream: same bus, same
        event names, same framing.
        """
        interval = max(0.1, min(5.0, (snapshot_interval_ms or 500) / 1000.0))
        bus = telemetry.dashboard.bus
        bus.attach_loop(asyncio.get_running_loop())
        queue = bus.subscribe()

        async def current_snapshot() -> str:
            # engine.status() is a blocking HTTP call; keep it off the loop.
            payload = await run_in_threadpool(
                lambda: telemetry.snapshot(
                    engine.status(), profile=_profile(), machine=_machine_info()
                )
            )
            return f"event: snapshot\ndata: {json.dumps(payload)}\n\n"

        async def events():
            try:
                yield await current_snapshot()
                last = time.perf_counter()
                while True:
                    # Snapshot faster while a request is live, as the MLX
                    # server does, so in-flight rows stay fresh.
                    active = telemetry.active_requests > 0
                    effective = min(interval, 0.25) if active else interval
                    timeout = max(0.01, effective - (time.perf_counter() - last))
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=timeout)
                        kind = event.get("kind", "event")
                        yield f"event: {kind}\ndata: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        pass
                    if (time.perf_counter() - last) >= effective:
                        yield await current_snapshot()
                        last = time.perf_counter()
            finally:
                bus.unsubscribe(queue)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/mtplx/prefill_history")
    def prefill_history() -> dict[str, Any]:
        # One row per completed request, built from the engine's prefill
        # counters across that request.
        history = telemetry.dashboard.prefill_history
        return {"capacity": history.capacity(), "history": history.snapshot()}

    def _settings_payload() -> dict[str, Any]:
        return {
            "ok": True,
            "model": engine.model_id,
            **telemetry.settings(),
            **sampling.payload(),
            "draft_tokens": draft_tokens_for(engine.model_id),
        }

    @app.get("/v1/mtplx/settings")
    def get_settings() -> dict[str, Any]:
        return _settings_payload()

    @app.post("/v1/mtplx/settings")
    async def post_settings(request: Request) -> dict[str, Any]:
        _authorize(request)
        report = sampling.update(await _json_body(request))
        # The reply carries the values now in force plus what moved or was
        # ignored, so a panel that wrote an unsupported value snaps to the
        # one Splash will actually use rather than silently diverging.
        return {**_settings_payload(), **report}

    @app.post("/v1/mtplx/cancel/{request_id}")
    def cancel(request_id: str) -> dict[str, Any]:
        return {"ok": telemetry.cancel(request_id), "request_id": request_id}

    @app.get("/v1/mtplx/app/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "ok": True,
            "name": "mtplx-splash-bridge",
            "api_version": 1,
            "engine": "splash",
            "endpoints": {
                "health": "/health",
                "snapshot": "/v1/mtplx/snapshot",
                "metrics_stream": "/v1/mtplx/metrics/stream",
                "chat_completions": "/v1/chat/completions",
                "messages": "/v1/messages",
                "models": "/v1/models",
                "engine_load": "/v1/mtplx/engine/load",
                "engine_unload": "/v1/mtplx/engine/unload",
            },
            "mutable_settings": [],
            "restart_required_settings": ["model", "max_memory", "max_context"],
            "snapshot_interval": {
                "default_ms": 500,
                "min_ms": 100,
                "max_ms": 5000,
                "native_default_ms": 500,
                "performance_lock_ms": 1000,
            },
            "features": {
                "chat": True,
                "streaming": True,
                "tool_calls": True,
                "vision": True,
                "mtp": False,
                "adaptive_depth": False,
                "kv_quantization": False,
                "engine_load_unload": True,
            },
        }

    @app.get("/v1/mtplx/thermal/status")
    def thermal_status() -> dict[str, Any]:
        return {"ok": True, "supported": False, "fan_mode": "auto"}

    @app.get("/metrics")
    def metrics() -> Response:
        text = engine.metrics_text()
        return PlainTextResponse(
            text, media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    @app.get("/admin/sessions")
    def sessions() -> dict[str, Any]:
        return telemetry.sessions(engine.status())

    @app.post("/admin/cache/clear")
    def clear_cache() -> dict[str, Any]:
        # Splash's prefix cache is owned by the engine and has no clear hook;
        # restarting the engine is the only way to drop it.
        return {"ok": False, "reason": "Splash manages its own prefix cache"}

    @app.post("/admin/sessions/{session_id}/clear")
    def clear_session(session_id: str) -> dict[str, Any]:
        return {"ok": False, "session_id": session_id}

    # -- engine control (load / unload) -------------------------------------

    @app.get("/v1/mtplx/engine")
    def engine_state() -> dict[str, Any]:
        return {
            "engine": "splash",
            "model_id": engine.model_id,
            "phase": engine.state.phase,
            "detail": engine.state.detail,
            "error": engine.state.error,
            "pid": engine.state.pid,
            "port": engine.port,
            "ready": engine.is_ready(),
            "installed": install.is_installed(engine.model_id),
            "available_models": [
                {
                    "id": model_id,
                    "installed": install.is_installed(model_id),
                }
                for model_id in install.official_models()
            ],
            "logs": list(engine.state.logs)[-40:],
            "uptime_s": max(0.0, time.time() - started_at),
        }

    @app.post("/v1/mtplx/engine/load")
    async def engine_load(request: Request) -> dict[str, Any]:
        _authorize(request)
        try:
            payload = await request.json()
        except (ValueError, UnicodeDecodeError):
            payload = {}
        model_id = (payload or {}).get("model") or engine.model_id
        if model_id != engine.model_id:
            engine.stop()
            engine.model_id = model_id
            engine.state.model_id = model_id
            telemetry.model_id = model_id
        try:
            # Downloads a 17 GB package on a cold load; never on the loop.
            await run_in_threadpool(engine.start)
        except Exception as error:  # surfaced to the caller, not swallowed
            raise HTTPException(status_code=500, detail=str(error)) from None
        return engine_state()

    @app.post("/v1/mtplx/engine/unload")
    async def engine_unload(request: Request) -> dict[str, Any]:
        _authorize(request)
        # Waits out the SIGTERM grace period before escalating.
        await run_in_threadpool(engine.stop)
        return engine_state()

    return app
