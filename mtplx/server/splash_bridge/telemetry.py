# SPDX-License-Identifier: Apache-2.0
"""Splash engine telemetry, in the shapes the MTPLX clients already decode.

The dashboard is not driven by snapshots alone. The hero tiles follow the
metrics stream's `progress` and `completed` events, the min/max/mean/p95 row
comes from `RollingMetrics`, the "new record" toast from `new_max_tps`, and
the per-request table from the last 32 completion envelopes. The MLX server
produces all of that through `mtplx.server.dashboard_state`, which is plain
stdlib, so the bridge drives the very same objects rather than imitating
their output: identical shapes by construction, not by transcription.

Where each number comes from:

* Speeds, token counts and draft acceptance for a finished request are the
  engine's own: the difference in Splash's cumulative `/status` counters
  across the request. (With overlapping requests those counters are shared,
  so concurrent requests each see the aggregate for the overlap.)
* The live decode gauge during a stream is measured here, from frame arrival
  times, because the engine only publishes counters, not a live rate.
* Memory is Splash's `memory_actual`, the process footprint, not the KV pages
  alone.
* Numbers that only exist for the MLX engine (MTP depth, per-depth
  acceptance) stay absent. Splash's draft acceptance is real and is reported,
  labelled `draft: dflash2`.
"""

from __future__ import annotations

import collections
import hashlib
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from mtplx.server.dashboard_state import DashboardState, InFlightHandle

# Splash's engine reports this, and it is not configurable: its Metal kernels
# read 8-bit KV directly, so there is no bit width to choose.
SPLASH_KV_BITS = 8
SPLASH_KV_QUANTIZATION = "q8"

# Matches the MLX server's recent-request window.
RECENT_REQUESTS = 32
# Live gauge cadence. The app throttles progress frames itself; this only
# bounds how often a fast stream touches the bus.
PROGRESS_INTERVAL_S = 0.2
# Conversations whose cached prefix is worth listing; Splash holds many more
# blocks than this, but the dashboard only needs the recent ones.
TRACKED_SESSIONS = 8
# Splash pages KV in blocks of this many tokens (identity.cache.block_tokens).
KV_BLOCK_TOKENS = 32


def conversation_key(payload: dict[str, Any]) -> str | None:
    """A stable id for a conversation, so its turns share one session row.

    Splash reuses a prefix by content, not by any id the client sends, so the
    honest notion of a "session" is the conversation's opening: every later
    turn extends the same prefix chain. A client-supplied id wins when there
    is one.
    """
    for field_name in ("session_id", "user"):
        value = payload.get(field_name)
        if isinstance(value, str) and value:
            return value
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    # The opening is every leading system message plus the first message that
    # is not one. Slicing a fixed count is not stable: a one-message first
    # turn and its three-message follow-up would differ in their second item.
    opening = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        opening.append(f"{message.get('role')}:{message.get('content')!r}"[:2000])
        if message.get("role") != "system":
            break
    digest = hashlib.sha1("|".join(opening).encode("utf-8", "replace")).hexdigest()
    return f"conv-{digest[:10]}"


def _num(source: Any, *path: str, default: float = 0.0) -> float:
    """Read a nested numeric out of Splash's /status, tolerating absence."""
    cursor: Any = source
    for key in path:
        if not isinstance(cursor, dict):
            return default
        cursor = cursor.get(key)
    # Splash reports flags as JSON booleans (metal.healthy, ready) and its own
    # prometheus exporter maps them to 1/0; do the same. Excluding bool here
    # silently reported a healthy Metal device as unhealthy.
    if isinstance(cursor, bool):
        return 1.0 if cursor else 0.0
    if not isinstance(cursor, (int, float)):
        return default
    return float(cursor)


@dataclass
class EngineCounters:
    """Splash's cumulative counters at one instant; two of these bracket a request."""

    prefill_tokens: float = 0.0
    prefill_wall_ms: float = 0.0
    decode_tokens: float = 0.0
    decode_wall_ms: float = 0.0
    drafted: float = 0.0
    accepted: float = 0.0
    reused_tokens: float = 0.0

    @classmethod
    def read(cls, status: dict[str, Any]) -> "EngineCounters":
        metrics = status.get("metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        return cls(
            prefill_tokens=_num(metrics, "prefill_input_tokens"),
            prefill_wall_ms=_num(metrics, "prefill_wall_ms"),
            decode_tokens=_num(metrics, "decode_output_tokens"),
            decode_wall_ms=_num(metrics, "decode_wall_ms"),
            drafted=_num(metrics, "drafted_tokens"),
            accepted=_num(metrics, "accepted_draft_tokens"),
            reused_tokens=_num(status, "cache", "reused_tokens"),
        )


@dataclass
class RequestTrace:
    """One proxied request, from first byte sent to last frame forwarded."""

    handle: InFlightHandle
    started_perf: float
    before: EngineCounters
    first_token_perf: float | None = None
    frames: int = 0
    last_progress_perf: float = 0.0

    @property
    def request_id(self) -> str:
        return self.handle.request_id

    @property
    def cancelled(self) -> bool:
        return self.handle.cancel_event.is_set()


class BridgeTelemetry:
    """The dashboard state for one bridge process, plus the engine translation."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.started_at = time.time()
        self.dashboard = DashboardState()
        self.recent: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=RECENT_REQUESTS
        )
        self._ids = itertools.count(1)
        self._lock = threading.RLock()
        # session id -> the prefix Splash most recently cached for it
        self._sessions: collections.OrderedDict[str, dict[str, Any]] = (
            collections.OrderedDict()
        )
        # Splash's own response id (chatcmpl-...) -> our request id. Clients
        # cancel by the id they saw on the stream, which is Splash's.
        self._aliases: dict[str, str] = {}

    # -- request accounting ------------------------------------------------

    def begin(
        self,
        prompt_preview: str,
        status: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> RequestTrace:
        request_id = f"splash-{next(self._ids)}"
        handle = InFlightHandle(
            request_id=request_id,
            cancel_event=threading.Event(),
            started_s=time.time(),
            session_id=session_id or request_id,
            model=self.model_id,
            prompt_preview=prompt_preview[:200],
        )
        self.dashboard.in_flight.register(handle)
        return RequestTrace(
            handle=handle,
            started_perf=time.perf_counter(),
            before=EngineCounters.read(status or {}),
        )

    def set_prompt_tokens(self, trace: RequestTrace, prompt_tokens: int) -> None:
        """Record the prompt size once it is known.

        Counted beside the request rather than in front of it, so the Context
        tile can show a live figure without the count ever delaying the first
        token.
        """
        if prompt_tokens > 0:
            trace.handle.prompt_tokens = int(prompt_tokens)

    def token(self, trace: RequestTrace, count: int = 1) -> None:
        """A frame of generated text arrived; keep the live gauge moving."""
        now = time.perf_counter()
        if trace.first_token_perf is None:
            trace.first_token_perf = now
        trace.frames += count
        if now - trace.last_progress_perf < PROGRESS_INTERVAL_S:
            return
        trace.last_progress_perf = now
        since_first = now - trace.first_token_perf
        # The first frame marks the clock's start, so it carries no interval.
        live = ((trace.frames - 1) / since_first) if since_first > 0.05 else 0.0
        progress = {
            "request_id": trace.request_id,
            "session_id": trace.handle.session_id,
            "model_id": self.model_id,
            "generation_mode": "splash",
            "phase": "decode",
            "completion_tokens": trace.frames,
            "generated_tokens": trace.frames,
            "elapsed_s": now - trace.started_perf,
            "ttft_s": trace.first_token_perf - trace.started_perf,
            "draft": "dflash2",
        }
        if live > 0:
            progress["decode_tok_s"] = live
        self.dashboard.in_flight.update_progress(trace.request_id, progress)
        is_new_max = False
        if live > 0:
            is_new_max = self.dashboard.rolling.observe_progress(
                live, trace.handle.session_id
            )
        self.dashboard.bus.publish(
            {
                "kind": "progress",
                "when_s": time.time(),
                "request_id": trace.request_id,
                "progress": progress,
            }
        )
        if is_new_max:
            self._publish_new_max(live, trace.handle.session_id)

    def finish(
        self,
        trace: RequestTrace,
        *,
        status: dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        cached_tokens: int = 0,
        cancelled: bool = False,
    ) -> dict[str, Any]:
        """Close the request out and publish its completion envelope."""
        now = time.perf_counter()
        self.dashboard.in_flight.deregister(trace.request_id)
        after = EngineCounters.read(status or {})
        before = trace.before

        d_prefill_tokens = max(0.0, after.prefill_tokens - before.prefill_tokens)
        d_prefill_ms = max(0.0, after.prefill_wall_ms - before.prefill_wall_ms)
        d_decode_tokens = max(0.0, after.decode_tokens - before.decode_tokens)
        d_decode_ms = max(0.0, after.decode_wall_ms - before.decode_wall_ms)
        d_drafted = max(0.0, after.drafted - before.drafted)
        d_accepted = max(0.0, after.accepted - before.accepted)
        d_reused = max(0.0, after.reused_tokens - before.reused_tokens)

        prompt = int(prompt_tokens or 0)
        completion = int(completion_tokens or 0) or int(d_decode_tokens) or trace.frames
        cached = int(cached_tokens or 0) or int(d_reused)
        elapsed = now - trace.started_perf
        ttft = (
            trace.first_token_perf - trace.started_perf
            if trace.first_token_perf is not None
            else None
        )
        if ttft is None and d_prefill_ms > 0:
            # A non-streamed response has no first frame to time. The engine's
            # prefill time for this request is what that wait consisted of, so
            # report it rather than leave time-to-first-token blank.
            ttft = d_prefill_ms / 1000.0

        # Prefer the engine's own clock; fall back to arrival timing only when
        # its counters did not move (an upstream error, or a status miss).
        decode_tok_s: float | None = None
        if d_decode_ms > 0 and d_decode_tokens > 0:
            decode_tok_s = d_decode_tokens / (d_decode_ms / 1000.0)
        elif trace.first_token_perf is not None and trace.frames > 1:
            window = now - trace.first_token_perf
            if window > 0.05:
                decode_tok_s = (trace.frames - 1) / window
        prefill_tok_s: float | None = None
        if d_prefill_ms > 0 and d_prefill_tokens > 0:
            prefill_tok_s = d_prefill_tokens / (d_prefill_ms / 1000.0)
        elif ttft and prompt:
            prefill_tok_s = prompt / ttft

        envelope: dict[str, Any] = {
            "request_id": trace.request_id,
            "session_id": trace.handle.session_id,
            "model_id": self.model_id,
            "generation_mode": "splash",
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "generated_tokens": completion,
            "cached_tokens": cached,
            "new_prefill_tokens": int(d_prefill_tokens) or max(0, prompt - cached),
            "decode_tok_s": decode_tok_s,
            "prefill_tok_s": prefill_tok_s,
            "prefill_compute_tok_s": prefill_tok_s,
            "prefill_wall_tok_s": (prompt / ttft) if (ttft and prompt) else prefill_tok_s,
            "ttft_s": ttft,
            "prompt_eval_time_s": (d_prefill_ms / 1000.0) or ttft,
            "decode_elapsed_s": (d_decode_ms / 1000.0) or None,
            "elapsed_s": elapsed,
            "cache_hit": cached > 0,
            # The Cached tile's "hit" caption reads this exact key.
            "session_cache_hit": cached > 0,
            "cache_source": "splash-prefix" if cached > 0 else None,
            "context_len": prompt + completion,
            "draft": "dflash2",
            "drafted_tokens": int(d_drafted),
            "accepted_drafts": int(d_accepted),
            "draft_acceptance_rate": (d_accepted / d_drafted) if d_drafted > 0 else None,
            "kv_quantization_bits": SPLASH_KV_BITS,
            "cancelled": bool(cancelled or trace.cancelled),
            "completed_at_s": time.time(),
        }

        if envelope["cancelled"]:
            self.dashboard.lifetime.record_cancellation()
        self.dashboard.lifetime.record_completion(
            prompt_tokens=prompt, completion_tokens=completion, cached_tokens=cached
        )
        is_new_max = False
        if decode_tok_s and not envelope["cancelled"]:
            is_new_max = self.dashboard.rolling.append(
                decode_tok_s, trace.handle.session_id
            )
        self.dashboard.prefill_history.append(
            {
                "when_s": time.time(),
                "request_id": trace.request_id,
                "session_id": trace.handle.session_id,
                "prompt_tokens": prompt,
                "cached_tokens": cached,
                "new_prefill_tokens": envelope["new_prefill_tokens"],
                "prompt_eval_time_s": envelope["prompt_eval_time_s"],
                "prefill_tok_s": prefill_tok_s,
                "prefill_compute_tok_s": prefill_tok_s,
                "prefill_wall_tok_s": envelope["prefill_wall_tok_s"],
                "ttft_s": ttft,
                "session_cache_hit": cached > 0,
                "context_len": prompt + completion,
                "model_id": self.model_id,
            }
        )
        # The Avg Prefill tile reads prefill_rates, which sums these samples:
        # tokens actually computed over the time spent computing them. Cached
        # tokens are excluded, so a warm turn does not inflate the average.
        if d_prefill_tokens > 0 and d_prefill_ms > 0:
            self.dashboard.prefill_history.record_chunk(
                int(d_prefill_tokens), d_prefill_ms / 1000.0
            )
        elif ttft and prompt > cached:
            self.dashboard.prefill_history.record_chunk(prompt - cached, ttft)
        with self._lock:
            self.recent.append(envelope)
            if not envelope["cancelled"] and trace.handle.session_id:
                self._sessions[trace.handle.session_id] = {
                    "prefix_len": prompt + completion,
                    "last_access_s": time.time(),
                }
                self._sessions.move_to_end(trace.handle.session_id)
                while len(self._sessions) > TRACKED_SESSIONS:
                    self._sessions.popitem(last=False)
        self.dashboard.progress_events.forget(trace.request_id)
        with self._lock:
            for alias in [k for k, v in self._aliases.items() if v == trace.request_id]:
                del self._aliases[alias]
        self.dashboard.bus.publish(
            {"kind": "completed", "when_s": time.time(), "envelope": dict(envelope)}
        )
        if is_new_max and decode_tok_s:
            self._publish_new_max(decode_tok_s, trace.handle.session_id)
        return envelope

    def _publish_new_max(self, tok_s: float, session_id: str | None) -> None:
        self.dashboard.bus.publish(
            {
                "kind": "new_max_tps",
                "when_s": time.time(),
                "tok_s": float(tok_s),
                "session_id": session_id,
            }
        )

    def alias(self, trace: RequestTrace, upstream_id: str) -> None:
        """Let a cancel addressed to Splash's response id reach this request."""
        with self._lock:
            self._aliases[upstream_id] = trace.request_id

    def cancel(self, request_id: str) -> bool:
        """Trip the request's cancel flag; the proxy loop closes the upstream."""
        with self._lock:
            request_id = self._aliases.get(request_id, request_id)
        return self.dashboard.in_flight.cancel(request_id)

    @staticmethod
    def chat_stats(envelope: dict[str, Any]) -> dict[str, Any]:
        """The `mtplx_stats` block the MLX engine puts on its finish frame.

        The app's chat reads it for the per-reply footer (tok/s, TTFT) and the
        held header reading, so a Splash reply carries the same keys.
        """
        stats: dict[str, Any] = {
            "generation_mode": "splash",
            "draft": "dflash2",
            "prompt_tokens": envelope.get("prompt_tokens"),
            "completion_tokens": envelope.get("completion_tokens"),
            "cached_tokens": envelope.get("cached_tokens"),
            "new_prefill_tokens": envelope.get("new_prefill_tokens"),
            "ttft_s": envelope.get("ttft_s"),
            "request_elapsed_s": envelope.get("elapsed_s"),
            "decode_elapsed_s": envelope.get("decode_elapsed_s"),
            "prompt_eval_time_s": envelope.get("prompt_eval_time_s"),
            "prefill_tok_s": envelope.get("prefill_tok_s"),
            "session_cache_hit": envelope.get("session_cache_hit"),
            "drafted_tokens": envelope.get("drafted_tokens"),
            "accepted_drafts": envelope.get("accepted_drafts"),
            "draft_acceptance_rate": envelope.get("draft_acceptance_rate"),
            "kv_quantization_bits": envelope.get("kv_quantization_bits"),
        }
        tok_s = envelope.get("decode_tok_s")
        if tok_s:
            stats["raw_decode_tok_s"] = tok_s
            stats["decode_tok_s"] = tok_s
            stats["display_decode_tok_s"] = tok_s
        return {key: value for key, value in stats.items() if value is not None}

    @property
    def active_requests(self) -> int:
        return self.dashboard.in_flight.count()

    # -- engine translation --------------------------------------------------

    @staticmethod
    def _phase(status: dict[str, Any]) -> str:
        if _num(status, "scheduler", "prefilling") > 0:
            return "prefill"
        if _num(status, "scheduler", "decoding") > 0:
            return "decode"
        return "idle"

    def _engine_record(self, status: dict[str, Any]) -> dict[str, Any]:
        """Engine-lifetime figures, shown until a request has completed."""
        metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
        prefill = _num(metrics, "prefill_tokens_per_second")
        decode = _num(metrics, "decode_tokens_per_second")
        record: dict[str, Any] = {
            "generation_mode": "splash",
            "model_id": self.model_id,
            "phase": self._phase(status),
            "draft": "dflash2",
            "kv_quantization_bits": SPLASH_KV_BITS,
            "drafted_tokens": _num(metrics, "drafted_tokens"),
            "accepted_drafts": _num(metrics, "accepted_draft_tokens"),
            "draft_acceptance_rate": _num(metrics, "draft_acceptance_rate"),
            "cached_tokens": _num(status, "cache", "reused_tokens"),
        }
        if prefill > 0:
            record["prefill_tok_s"] = prefill
            record["prefill_compute_tok_s"] = prefill
        if decode > 0:
            record["decode_tok_s"] = decode
        return record

    def engine_stats(self, status: dict[str, Any]) -> dict[str, Any]:
        """Splash-specific counters, for a raw view and for debugging."""
        metrics = status.get("metrics") if isinstance(status.get("metrics"), dict) else {}
        return {
            "engine": "splash",
            "kv_quantization_bits": SPLASH_KV_BITS,
            "kv_pages_total": _num(status, "kv", "pages_total"),
            "kv_pages_free": _num(status, "kv", "pages_free"),
            "kv_pages_active": _num(status, "kv", "pages_active"),
            "kv_resident_backing_bytes": _num(status, "kv", "resident_backing_bytes"),
            "cache_hits": _num(status, "cache", "hits"),
            "cache_cold_misses": _num(status, "cache", "cold_misses"),
            "cache_reused_tokens": _num(status, "cache", "reused_tokens"),
            "scheduler_queued": _num(status, "scheduler", "queued"),
            "scheduler_prefilling": _num(status, "scheduler", "prefilling"),
            "scheduler_decoding": _num(status, "scheduler", "decoding"),
            "requests_completed": _num(status, "requests", "completed"),
            "requests_failed": _num(status, "requests", "failed"),
            "metal_healthy": bool(_num(status, "metal", "healthy")),
            "memory_pressure": status.get("memory_pressure") or "unknown",
            "ttft_ms_p50": _num(metrics, "ttft_ms", "p50"),
            "ttft_ms_p95": _num(metrics, "ttft_ms", "p95"),
            "itl_ms_p50": _num(metrics, "itl_ms", "p50"),
            "itl_ms_p95": _num(metrics, "itl_ms", "p95"),
            "prefill_tokens_per_second": _num(metrics, "prefill_tokens_per_second"),
            "decode_tokens_per_second": _num(metrics, "decode_tokens_per_second"),
            "draft_acceptance_rate": _num(metrics, "draft_acceptance_rate"),
        }

    def sessions(self, status: dict[str, Any]) -> dict[str, Any]:
        """Recent conversations and the prefix Splash holds for each.

        Splash owns its prefix cache and may evict under pressure, so rows are
        only listed while the engine reports cached pages at all, and the byte
        figure is an estimate from the engine's own page size.
        """
        cached_pages = _num(status, "kv", "pages_cache") + _num(status, "kv", "pages_active")
        resident = _num(status, "kv", "resident_backing_bytes")
        per_token = (resident / cached_pages / KV_BLOCK_TOKENS) if cached_pages > 0 else 0.0
        live = set(self.dashboard.in_flight.session_ids())
        rows = []
        with self._lock:
            sessions = list(self._sessions.items())
        if cached_pages > 0:
            for session_id, info in reversed(sessions):
                rows.append(
                    {
                        "session_id": session_id,
                        "prefix_len": int(info["prefix_len"]),
                        "bytes": int(info["prefix_len"] * per_token),
                        "in_flight": session_id in live,
                        "last_access_s": info["last_access_s"],
                    }
                )
        return {"sessions": rows, "count": len(rows)}

    def context_window(self, status: dict[str, Any]) -> int:
        reported = _num(status, "maximum_context_tokens")
        return int(reported) if reported > 0 else 0

    def mem(self, status: dict[str, Any]) -> dict[str, Any]:
        """The engine's real footprint, from Splash's memory_actual block.

        Reading only `kv.resident_backing_bytes` here reported the KV pages
        alone — about 0.14 GB against an 18.9 GB process — so the memory tile
        was accurate about the wrong quantity.
        """
        payload: dict[str, Any] = {"ok": True}
        current = _num(status, "memory_actual", "current_bytes")
        peak = _num(status, "memory_actual", "peak_bytes")
        weights = _num(status, "memory_actual", "dense_bytes")
        kv_resident = _num(status, "memory_actual", "sparse_resident_bytes")
        if current:
            payload["active_memory_bytes"] = int(current)
        if peak:
            payload["peak_memory_bytes"] = int(peak)
        if weights:
            payload["model_weights_bytes"] = int(weights)
        if kv_resident:
            # KV pages are the cache portion of that footprint.
            payload["cache_memory_bytes"] = int(kv_resident)
        if not current:
            # Pre-warmup the block is absent; say so rather than imply zero.
            payload["ok"] = bool(_num(status, "ready"))
        return payload

    def kv_quant_policy(self) -> dict[str, Any]:
        """Splash's KV precision is a property of its kernels, not a setting."""
        return {
            "supported": False,
            "modes": [SPLASH_KV_QUANTIZATION],
            "restart_required": False,
            "proof_level": "engine-fixed",
            "disabled_reason": (
                "Splash reads 8-bit KV directly in its precompiled Metal "
                "kernels, so the bit width is fixed at int8. Use the MLX "
                "engine for 4-bit, 8-bit or unquantized KV."
            ),
        }

    def settings(self) -> dict[str, Any]:
        return {
            "generation_mode": "splash",
            "reasoning_parser": "qwen3",
            "kv_quant_policy": self.kv_quant_policy(),
            "context_window_policy": {"supported": False, "source": "engine"},
            "adaptive_depth_supported": False,
        }

    def snapshot(
        self,
        status: dict[str, Any],
        *,
        profile: dict[str, Any],
        machine: dict[str, Any],
    ) -> dict[str, Any]:
        now = time.time()
        with self._lock:
            recent = list(self.recent)
        if not recent:
            # Nothing has completed yet: show the engine's lifetime figures
            # (warmup runs included) rather than an empty table.
            recent = [self._engine_record(status)]
        in_flight = self.dashboard.in_flight.snapshot()
        return {
            "ts": now,
            "model_id": self.model_id,
            "profile": profile,
            "context_window": self.context_window(status),
            "active_requests": len(in_flight),
            "in_flight": in_flight,
            "recent": recent,
            "rolling": self.dashboard.rolling.snapshot(),
            "lifetime": self.dashboard.lifetime.snapshot(),
            "latest": recent[-1],
            "prefill_rates": self.dashboard.prefill_history.rates(),
            "sessions": self.sessions(status),
            "session_bank": {},
            "mem": self.mem(status),
            "settings": self.settings(),
            "machine": machine,
            "uptime_s": max(0.0, now - self.started_at),
            "engine_stats": self.engine_stats(status),
        }
