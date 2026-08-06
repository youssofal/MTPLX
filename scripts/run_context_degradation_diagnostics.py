#!/usr/bin/env python3
"""Cross-system long-response serving diagnostics.

This runner is intentionally provider-neutral. It talks to an OpenAI-compatible
server, records raw SSE timing, and writes one JSONL summary row per request.
It is meant to compare:

- MTPLX/MLX direct HTTP
- MTPLX/MLX through Open WebUI request shapes
- vLLM direct HTTP
- vLLM through Open WebUI request shapes

It does not hide max-token behavior. If --max-tokens is omitted, the request
payload omits max_tokens and the output row records that fact.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


FLAPPY_PROMPT = (
    "Create a single-file HTML5 Canvas flappy bird game. All visuals drawn "
    "procedurally. Animated bird with distinct up-stroke and down-stroke wing "
    "shapes, body tilt, squash-and-stretch on flap, feather particles from "
    "wing tips. Pipes with gradient shading, cap/lip, cylindrical highlight. "
    "Three-layer parallax background: sky with day/night colour cycle and "
    "stars, clouds with bobbing, rolling hills. Death explosion, +1 score pop, "
    "ambient floating motes. Start screen, death screen with best score in "
    "localStorage. Delta-time physics. Make it gorgeous."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a precise coding assistant. Return working code when asked."
)


def _comma_floats(value: str) -> tuple[float, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _json_loads(value: str | None, *, default: Any) -> Any:
    if not value:
        return default
    return json.loads(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _server_python_env(env: dict[str, str], repo_root: Path) -> dict[str, str]:
    next_env = dict(env)
    repo_path = str(repo_root.resolve())
    existing = next_env.get("PYTHONPATH")
    if existing:
        parts = existing.split(os.pathsep)
        next_env["PYTHONPATH"] = os.pathsep.join(
            [repo_path, *[part for part in parts if part != repo_path]]
        )
    else:
        next_env["PYTHONPATH"] = repo_path
    return next_env


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def _run(cmd: list[str], *, timeout: int = 60) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_s": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "cmd": cmd,
            "returncode": None,
            "stdout": "",
            "stderr": repr(exc),
            "elapsed_s": time.perf_counter() - started,
        }


def _http_json(url: str, *, timeout: int = 10) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(raw)
            except Exception:
                parsed = {"raw": raw}
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "elapsed_s": time.perf_counter() - started,
                "body": parsed,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = {"raw": raw}
        return {
            "ok": False,
            "status": exc.code,
            "elapsed_s": time.perf_counter() - started,
            "body": parsed,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "elapsed_s": time.perf_counter() - started,
            "body": {"error": repr(exc)},
        }


def collect_remote_env(args: argparse.Namespace) -> int:
    output_dir = args.output_dir / f"{args.label}-{_now_stamp()}"
    commands = {
        "identity": "hostname; date -Is; uname -a",
        "gpu_query": (
            "nvidia-smi --query-gpu=index,name,memory.total,memory.used,"
            "utilization.gpu,power.draw --format=csv"
        ),
        "gpu_topology": "nvidia-smi topo -m",
        "processes": "pgrep -af 'vllm serve|VLLM::Worker|open-webui|uvicorn' || true",
        "files": (
            "test -d /home/youssof/ai/models/Qwen3.6-27B-AWQ-BF16-INT4 && "
            "echo model_ok || echo model_missing; "
            "test -f /home/youssof/ai/mtplx-phase1-v4-20260429-012151/"
            "run_qwen36_optstyle_compare.sh && echo qwen36_optstyle_ok || "
            "echo qwen36_optstyle_missing; "
            "test -f /home/youssof/ai/launch-qwen-optimized.sh && "
            "echo legacy_optimized_ok || echo legacy_optimized_missing"
        ),
        "versions": (
            "source /home/youssof/ai/vllm-venv/bin/activate && python - <<'PY'\n"
            "import importlib, json\n"
            "mods = ['torch', 'vllm', 'flashinfer', 'transformers']\n"
            "out = {}\n"
            "for name in mods:\n"
            "    try:\n"
            "        mod = importlib.import_module(name)\n"
            "        out[name] = getattr(mod, '__version__', 'unknown')\n"
            "    except Exception as exc:\n"
            "        out[name] = repr(exc)\n"
            "print(json.dumps(out, sort_keys=True))\n"
            "PY"
        ),
    }
    rows = {
        name: _run(["ssh", "-o", "BatchMode=yes", args.ssh_host, cmd], timeout=args.timeout)
        for name, cmd in commands.items()
    }
    probes = {}
    for port in args.probe_ports:
        probes[str(port)] = {
            "models": _run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    args.ssh_host,
                    f"curl -sS --max-time 5 http://127.0.0.1:{port}/v1/models || true",
                ],
                timeout=10,
            )
        }
    result = {
        "label": args.label,
        "ssh_host": args.ssh_host,
        "created_at": datetime.now().isoformat(),
        "commands": rows,
        "port_probes": probes,
    }
    _write_json(output_dir / "remote-env.json", result)
    print(json.dumps({"output_dir": str(output_dir), "remote_env": str(output_dir / "remote-env.json")}, indent=2))
    return 0


def _prompt_rows(tests: list[str]) -> list[dict[str, str]]:
    rows = []
    for name in tests:
        if name == "hi":
            rows.append({"id": "hi", "prompt": "hi"})
        elif name == "flappy":
            rows.append({"id": "flappy", "prompt": FLAPPY_PROMPT})
        else:
            raise ValueError(f"unknown test: {name}")
    return rows


def _build_payload(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    system_prompt: str,
    stream: bool,
    max_tokens: int | None,
    temperature: float,
    top_p: float,
    top_k: int | None,
    seed: int | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if endpoint == "chat":
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "stream_options": {"include_usage": True},
            "metadata": metadata,
        }
    elif endpoint == "completions":
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "stream_options": {"include_usage": True},
        }
    else:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    if top_k is not None:
        payload["top_k"] = top_k
    if seed is not None:
        payload["seed"] = seed
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return payload


def _delta_text(event: dict[str, Any], endpoint: str) -> tuple[str, str]:
    choices = event.get("choices") or []
    if not choices:
        return "", ""
    choice = choices[0]
    if endpoint == "completions":
        return "content", str(choice.get("text") or "")
    delta = choice.get("delta") or {}
    for field in ("content", "reasoning_content", "reasoning", "thinking"):
        value = delta.get(field)
        if value:
            return field, str(value)
    return "", ""


def _finish_reason(event: dict[str, Any]) -> str | None:
    choices = event.get("choices") or []
    if not choices:
        return None
    return choices[0].get("finish_reason")


def _token_window_rate(times: list[float], *, size: int) -> float | None:
    if len(times) < max(2, size):
        return None
    window = times[:size]
    elapsed = window[-1] - window[0]
    if elapsed <= 0:
        return None
    return (len(window) - 1) / elapsed


def _last_token_window_rate(times: list[float], *, size: int) -> float | None:
    if len(times) < max(2, size):
        return None
    window = times[-size:]
    elapsed = window[-1] - window[0]
    if elapsed <= 0:
        return None
    return (len(window) - 1) / elapsed


def _extract_footer_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    match = re.search(r"MTPLX TPS:\s*([0-9.]+)\s*tok/s\s*.\s*([0-9]+)\s*tokens\s*.\s*([0-9.]+)s decode", text)
    if match:
        metrics["footer_tok_s"] = float(match.group(1))
        metrics["footer_tokens"] = int(match.group(2))
        metrics["footer_decode_s"] = float(match.group(3))
    return metrics


def _mean_rate(rows: list[dict[str, Any]]) -> float | None:
    elapsed = 0.0
    for row in rows:
        if row.get("bucket_elapsed_s") is not None:
            elapsed += float(row.get("bucket_elapsed_s") or 0.0)
        elif row.get("elapsed_s") is not None:
            elapsed += float(row.get("elapsed_s") or 0.0)
        elif row.get("t_end_s") is not None and row.get("t_start_s") is not None:
            elapsed += float(row.get("t_end_s") or 0.0) - float(row.get("t_start_s") or 0.0)
    tokens = sum(int(row.get("generated_tokens_delta") or 0) for row in rows)
    if elapsed <= 0 or tokens <= 0:
        return None
    return tokens / elapsed


def _mean_verify_ms(rows: list[dict[str, Any]]) -> float | None:
    calls = sum(int(row.get("verify_calls_delta") or 0) for row in rows)
    verify_s = sum(float(row.get("verify_time_s_delta") or 0.0) for row in rows)
    if calls <= 0:
        return None
    return 1000.0 * verify_s / calls


def _gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0**3)


def _mtplx_trace_summary(stats: dict[str, Any]) -> dict[str, Any]:
    path = stats.get("decode_trace_path")
    run_id = stats.get("decode_trace_run_id")
    if not path or not run_id:
        return {}
    trace_path = Path(path)
    if not trace_path.exists():
        return {"trace_path": str(trace_path), "trace_run_id": run_id, "trace_missing": True}
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("run_id") == run_id:
                rows.append(row)
    usable = [
        row
        for row in rows
        if not row.get("final")
        and int(row.get("generated_tokens_delta") or 0) >= 0
        and int(row.get("generated_tokens_total") or 0) > 0
    ]
    positive = [
        row
        for row in rows
        if int(row.get("generated_tokens_total") or 0) > 0
        and int(row.get("generated_tokens_delta") or 0) >= 0
    ]
    if not usable:
        return {
            "trace_path": str(trace_path),
            "trace_run_id": run_id,
            "trace_rows": len(rows),
            "trace_usable_rows": 0,
        }
    first10 = usable[:10]
    last10 = usable[-10:]
    mid_start = max(0, len(usable) // 2 - 15)
    mid30 = usable[mid_start : mid_start + 30]
    first_mem = (first10[-1].get("mlx_memory") or {}) if first10 else {}
    last_mem = (last10[-1].get("mlx_memory") or {}) if last10 else {}
    last10_cache_start = (last10[0].get("mlx_memory") or {}).get("cache_memory_bytes") if last10 else None
    last10_cache_end = last_mem.get("cache_memory_bytes")
    last10_cache_delta_gib = None
    if last10_cache_start is not None and last10_cache_end is not None:
        last10_cache_delta_gib = _gib(float(last10_cache_end) - float(last10_cache_start))
    return {
        "trace_path": str(trace_path),
        "trace_run_id": run_id,
        "trace_rows": len(rows),
        "trace_usable_rows": len(usable),
        "trace_generated_tokens": max(int(row.get("generated_tokens_total") or 0) for row in positive),
        "trace_context_len_last": int(positive[-1].get("context_len") or 0),
        "trace_elapsed_s": float(positive[-1].get("t_end_s") or positive[-1].get("elapsed_s") or 0.0),
        "trace_first10_tok_s": _mean_rate(first10),
        "trace_mid30_tok_s": _mean_rate(mid30),
        "trace_last10_tok_s": _mean_rate(last10),
        "trace_first10_verify_ms": _mean_verify_ms(first10),
        "trace_mid30_verify_ms": _mean_verify_ms(mid30),
        "trace_last10_verify_ms": _mean_verify_ms(last10),
        "trace_first_cache_gib": _gib(first_mem.get("cache_memory_bytes")),
        "trace_last_cache_gib": _gib(last_mem.get("cache_memory_bytes")),
        "trace_last10_cache_delta_gib": last10_cache_delta_gib,
        "trace_cache_plateaued": (
            abs(last10_cache_delta_gib) <= 0.25
            if last10_cache_delta_gib is not None
            else None
        ),
        "trace_first_active_gib": _gib(first_mem.get("active_memory_bytes")),
        "trace_last_active_gib": _gib(last_mem.get("active_memory_bytes")),
    }


def _stream_request(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    endpoint: str,
    timeout: int,
    events_path: Path,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    started = time.perf_counter()
    first_event_s = None
    first_text_s = None
    last_text_s = None
    token_like_times: list[float] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    usage = None
    mtplx_stats = None
    finish_reason = None
    event_count = 0
    nonempty_delta_count = 0
    error = None
    status = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            for raw in response:
                now = time.perf_counter()
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if first_event_s is None:
                    first_event_s = now - started
                if body == "[DONE]":
                    _append_jsonl(events_path, {"t_s": now - started, "done": True})
                    break
                try:
                    event = json.loads(body)
                except Exception as exc:
                    _append_jsonl(events_path, {"t_s": now - started, "raw": body, "parse_error": repr(exc)})
                    continue
                event_count += 1
                field, text = _delta_text(event, endpoint)
                if text:
                    nonempty_delta_count += 1
                    token_like_times.append(now - started)
                    last_text_s = now - started
                    if first_text_s is None:
                        first_text_s = now - started
                    if field == "content":
                        content_parts.append(text)
                    else:
                        reasoning_parts.append(text)
                if event.get("usage"):
                    usage = event.get("usage")
                if event.get("mtplx_stats"):
                    mtplx_stats = event.get("mtplx_stats")
                finish_reason = _finish_reason(event) or finish_reason
                _append_jsonl(
                    events_path,
                    {
                        "t_s": now - started,
                        "event_index": event_count,
                        "field": field or None,
                        "delta_chars": len(text),
                        "finish_reason": _finish_reason(event),
                        "has_usage": bool(event.get("usage")),
                        "has_mtplx_stats": bool(event.get("mtplx_stats")),
                        "error": event.get("error"),
                    },
                )
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read().decode("utf-8", errors="replace")
        error = {"type": "HTTPError", "status": exc.code, "body": raw[:4000]}
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": repr(exc)}
    ended = time.perf_counter()
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    decode_elapsed_s = None
    if first_text_s is not None and last_text_s is not None:
        decode_elapsed_s = max(last_text_s - first_text_s, 0.0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    total_decode_tok_s = None
    if completion_tokens and decode_elapsed_s and decode_elapsed_s > 0:
        total_decode_tok_s = max(completion_tokens - 1, 0) / decode_elapsed_s
    first64 = _token_window_rate(token_like_times, size=64)
    last64 = _last_token_window_rate(token_like_times, size=64)
    return {
        "status": status,
        "error": error,
        "request_elapsed_s": ended - started,
        "ttft_s": first_text_s,
        "first_event_s": first_event_s,
        "decode_elapsed_s": decode_elapsed_s,
        "finish_reason": finish_reason,
        "event_count": event_count,
        "nonempty_delta_count": nonempty_delta_count,
        "token_like_first64_s": first64,
        "token_like_last64_s": last64,
        "token_like_last_over_first": (last64 / first64) if first64 and last64 else None,
        "usage": usage,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_decode_tok_s_from_usage": total_decode_tok_s,
        "content_chars": len(content),
        "reasoning_chars": len(reasoning),
        "content_prefix": content[:500],
        "content_suffix": content[-500:],
        "reasoning_prefix": reasoning[:500],
        "mtplx_stats": mtplx_stats,
        "footer_metrics": _extract_footer_metrics(content),
        "content": content,
        "reasoning": reasoning,
    }


def _nonstream_request(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            body = json.loads(raw)
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except Exception:
            body = {"raw": raw}
        status = exc.code
    except Exception as exc:
        return {
            "status": None,
            "error": {"type": type(exc).__name__, "message": repr(exc)},
            "request_elapsed_s": time.perf_counter() - started,
        }
    elapsed = time.perf_counter() - started
    content = ""
    try:
        choice = body["choices"][0]
        if "message" in choice:
            content = str(choice["message"].get("content") or "")
        else:
            content = str(choice.get("text") or "")
    except Exception:
        content = ""
    usage = body.get("usage") if isinstance(body, dict) else None
    stats = body.get("mtplx_stats") if isinstance(body, dict) else None
    return {
        "status": status,
        "error": body.get("error") if isinstance(body, dict) else None,
        "request_elapsed_s": elapsed,
        "usage": usage,
        "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
        "completion_tokens": int((usage or {}).get("completion_tokens") or 0),
        "content_chars": len(content),
        "content_prefix": content[:500],
        "content_suffix": content[-500:],
        "mtplx_stats": stats,
        "footer_metrics": _extract_footer_metrics(content),
        "raw_response": body,
        "content": content,
    }


def run_direct(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"{args.label}-{_now_stamp()}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    request_dir = output_dir / "requests"
    text_dir = output_dir / "texts"
    event_dir = output_dir / "sse-events"
    headers = _json_loads(args.headers_json, default={})
    metadata_base = _json_loads(args.metadata_json, default={})
    tests = [item.strip() for item in args.tests.split(",") if item.strip()]
    base_url = args.base_url.rstrip("/")
    endpoint_path = "/v1/chat/completions" if args.endpoint == "chat" else "/v1/completions"
    url = base_url + endpoint_path
    probes = {
        "models": _http_json(base_url + "/v1/models", timeout=10),
        "health": _http_json(base_url + "/health", timeout=10),
        "metrics_before": _http_json(base_url + "/metrics", timeout=10),
    }
    _write_json(output_dir / "probes-before.json", probes)
    config = {
        "run_id": run_id,
        "label": args.label,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "model": args.model,
        "tests": tests,
        "stream": args.stream,
        "max_tokens_present": args.max_tokens is not None,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "headers": headers,
        "metadata_base": metadata_base,
    }
    _write_json(output_dir / "run-config.json", config)
    for row in _prompt_rows(tests):
        request_id = f"{row['id']}-{uuid.uuid4().hex[:8]}"
        metadata = {
            **metadata_base,
            "diagnostic_run_id": run_id,
            "diagnostic_label": args.label,
            "diagnostic_test": row["id"],
        }
        payload = _build_payload(
            endpoint=args.endpoint,
            model=args.model,
            prompt=row["prompt"],
            system_prompt=args.system_prompt,
            stream=args.stream,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            metadata=metadata,
        )
        request_path = request_dir / f"{request_id}.json"
        _write_json(request_path, {"url": url, "headers": headers, "payload": payload})
        events_path = event_dir / f"{request_id}.jsonl"
        print(f"[{args.label}] {row['id']} stream={args.stream} max_tokens={args.max_tokens}", flush=True)
        if args.stream:
            result = _stream_request(
                url=url,
                payload=payload,
                headers=headers,
                endpoint=args.endpoint,
                timeout=args.timeout,
                events_path=events_path,
            )
        else:
            result = _nonstream_request(
                url=url,
                payload=payload,
                headers=headers,
                timeout=args.timeout,
            )
        content = str(result.pop("content", "") or "")
        reasoning = str(result.pop("reasoning", "") or "")
        content_path = text_dir / f"{request_id}.content.txt"
        reasoning_path = text_dir / f"{request_id}.reasoning.txt"
        content_path.parent.mkdir(parents=True, exist_ok=True)
        content_path.write_text(content, encoding="utf-8")
        if reasoning:
            reasoning_path.write_text(reasoning, encoding="utf-8")
        result_row = {
            "run_id": run_id,
            "label": args.label,
            "test": row["id"],
            "request_id": request_id,
            "created_at": datetime.now().isoformat(),
            "request_path": str(request_path),
            "events_path": str(events_path) if args.stream else None,
            "content_path": str(content_path),
            "reasoning_path": str(reasoning_path) if reasoning else None,
            "prompt_chars": len(row["prompt"]),
            "request_max_tokens": args.max_tokens,
            "request_max_tokens_present": args.max_tokens is not None,
            **result,
        }
        _append_jsonl(results_path, result_row)
        print(json.dumps({k: result_row.get(k) for k in [
            "test",
            "status",
            "completion_tokens",
            "total_decode_tok_s_from_usage",
            "token_like_first64_s",
            "token_like_last64_s",
            "token_like_last_over_first",
            "request_elapsed_s",
            "finish_reason",
            "error",
        ]}, sort_keys=True), flush=True)
        if result_row.get("status") and int(result_row["status"]) >= 400:
            break
    after = {
        "metrics_after": _http_json(base_url + "/metrics", timeout=10),
    }
    _write_json(output_dir / "probes-after.json", after)
    print(json.dumps({"output_dir": str(output_dir), "results": str(results_path)}, indent=2))
    return 0


def summarize(args: argparse.Namespace) -> int:
    rows: list[dict[str, Any]] = []
    for path in args.results:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    summary_rows = []
    for row in rows:
        stats = row.get("mtplx_stats") or {}
        trace_summary = _mtplx_trace_summary(stats)
        first64 = (
            stats.get("sliding_decode_tok_s_first_64")
            or stats.get("sliding_decode_tok_s_last_64")
            or row.get("token_like_first64_s")
        )
        last64 = stats.get("sliding_decode_tok_s_last_64") or row.get("token_like_last64_s")
        ratio = (last64 / first64) if first64 and last64 else row.get("token_like_last_over_first")
        completion_tokens = (
            row.get("completion_tokens")
            or stats.get("completion_tokens")
            or stats.get("generated_tokens")
            or trace_summary.get("trace_generated_tokens")
        )
        trace_decode_tok_s = None
        if trace_summary.get("trace_generated_tokens") and trace_summary.get("trace_elapsed_s"):
            trace_decode_tok_s = float(trace_summary["trace_generated_tokens"]) / float(
                trace_summary["trace_elapsed_s"]
            )
        summary_rows.append(
            {
                "label": row.get("label"),
                "test": row.get("test"),
                "status": row.get("status"),
                "prompt_tokens": row.get("prompt_tokens") or stats.get("prompt_tokens"),
                "completion_tokens": completion_tokens,
                "decode_tok_s": stats.get("decode_tok_s") or row.get("total_decode_tok_s_from_usage") or trace_decode_tok_s,
                "first64": first64,
                "last64": last64,
                "last_over_first": ratio,
                "ttft_s": row.get("ttft_s") or stats.get("ttft_s"),
                "request_elapsed_s": row.get("request_elapsed_s"),
                "request_max_tokens_present": row.get("request_max_tokens_present"),
                "request_max_tokens": row.get("request_max_tokens"),
                "finish_reason": row.get("finish_reason"),
                "content_chars": row.get("content_chars"),
                "error": row.get("error"),
                "trace_summary": trace_summary,
            }
        )
    grouped: dict[str, list[float]] = {}
    for row in summary_rows:
        key = f"{row.get('label')}:{row.get('test')}"
        value = row.get("last_over_first")
        if isinstance(value, (int, float)):
            grouped.setdefault(key, []).append(float(value))
    aggregate = {
        key: {
            "count": len(values),
            "mean_last_over_first": statistics.mean(values),
            "abnormal_under_0_80": any(value < 0.80 for value in values),
        }
        for key, values in grouped.items()
    }
    result = {"rows": summary_rows, "aggregate": aggregate}
    if args.output:
        _write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


FAST_PATH_FLAGS = [
    "MTPLX_LAZY_VERIFY_LOGITS",
    "MTPLX_BATCH_TARGET_ARRAYS",
    "MTPLX_LAZY_MTP_HISTORY_APPEND",
    "MTPLX_DROP_EVENTS",
    "MTPLX_SKIP_VERIFY_SNAPSHOT",
]


def _local_profile_env(profile: str, base_env: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    env = dict(base_env)
    for key in [
        *FAST_PATH_FLAGS,
        "MTPLX_MTP_HISTORY_MATERIALIZE_EVERY",
        "MTPLX_CLEAR_CACHE_EVERY",
        "MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY",
        "MTPLX_STATE_REBASE_EVERY",
        "MTPLX_DETACH_COMPONENTS",
        "MTPLX_DETACH_MODE",
        "MTPLX_DETACH_EVERY",
        "MTPLX_DETACH_GDN_EVERY",
        "MTPLX_DETACH_CONV_EVERY",
        "MTPLX_DETACH_ATTN_EVERY",
        "MTPLX_CAPTURE_COMMIT_DETACH_COMPONENTS",
        "MTPLX_CAPTURE_COMMIT_DETACH_MODE",
        "MTPLX_CAPTURE_COMMIT_DETACH_EVERY",
        "MTPLX_CAPTURE_COMMIT_DETACH_GDN_EVERY",
        "MTPLX_CAPTURE_COMMIT_DETACH_CONV_EVERY",
        "MTPLX_DETACH_LIVE_OUTPUTS",
        "MTPLX_DETACH_LIVE_OUTPUTS_MODE",
        "MTPLX_OWNED_ATTN_KV",
        "MTPLX_OWNED_ATTN_KV_MODE",
        "MTPLX_OWNED_ATTN_KV_STEP",
        "MTPLX_OWNED_ATTN_KV_BLOCK_SIZE",
        "MTPLX_VLLM_METAL_PAGED_ATTN",
        "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE",
        "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS",
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL",
        "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_LAST_N",
        "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_INDICES",
        "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW",
        "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD",
        "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE",
        "MTPLX_SPLIT_FULL_ATTN",
        "MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE",
        "MTPLX_SPLIT_FULL_ATTN_THRESHOLD",
        "MTPLX_BLOCKWISE_ATTN",
        "MTPLX_BLOCKWISE_ATTN_THRESHOLD",
        "MTPLX_NATIVE_GDN_TAIL",
        "MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS",
        "MTPLX_MLX_CACHE_LIMIT",
        "MTPLX_SPLIT_VERIFY_EVAL",
        "MTPLX_SUSTAINED_PREFILL",
        "MTPLX_PREFILL_CHUNK_SIZE",
        "MTPLX_PREFILL_CHUNK_SIZE_DENSE",
        "MTPLX_PREFILL_CHUNK_SIZE_REPAGE",
        "MTPLX_SUSTAINED_PREFILL_LAYOUT",
        "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT",
        "MTPLX_DEFER_VERIFY_HIDDEN_EVAL",
        "MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS",
        "MTPLX_DYNAMIC_PAGED_KV",
        "MTPLX_DYNAMIC_PAGED_KV_TOKENS",
        "MTPLX_DYNAMIC_PAGED_KV_MARGIN",
        "MTPLX_DYNAMIC_PAGED_KV_MIN_BLOCKS",
    ]:
        env.pop(key, None)
    fast_defaults = {
        "MTPLX_LAZY_VERIFY_LOGITS": "1",
        "MTPLX_BATCH_TARGET_ARRAYS": "1",
        "MTPLX_LAZY_MTP_HISTORY_APPEND": "1",
        "MTPLX_DROP_EVENTS": "1",
        "MTPLX_SKIP_VERIFY_SNAPSHOT": "1",
    }
    info: dict[str, Any] = {"profile": profile, "profile_type": "lazy_ablation"}
    if profile in {"current_baseline", "baseline"}:
        env.update(fast_defaults)
        info["profile_type"] = "baseline"
    elif profile == "sustained":
        env.update(fast_defaults)
        env.update(
            {
                "MTPLX_SUSTAINED_PREFILL": "1",
                "MTPLX_SUSTAINED_PREFILL_LAYOUT": "auto",
                "MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT": "131072",
                "MTPLX_PREFILL_CHUNK_SIZE": "auto",
                "MTPLX_PREFILL_CHUNK_SIZE_DENSE": "2048",
                "MTPLX_PREFILL_CHUNK_SIZE_REPAGE": "2048",
                "MTPLX_DEFER_VERIFY_HIDDEN_EVAL": "auto",
                "MTPLX_TARGET_EMIT_FULL_PREFILL_LOGITS": "0",
                "MTPLX_DYNAMIC_PAGED_KV": "1",
                "MTPLX_VLLM_METAL_PAGED_ATTN": "1",
                "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE": "16",
                "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL": "mlx_vector_paged",
                "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN": "1",
                "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD": "2048",
                "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE": "512",
                "MTPLX_CLEAR_CACHE_EVERY": "0",
            }
        )
        info["profile_type"] = "sustained"
    elif profile == "split_verify_eval":
        env.update(fast_defaults)
        env["MTPLX_SPLIT_VERIFY_EVAL"] = "1"
        info["profile_type"] = "eval_split"
    elif profile == "lazy_verify_off":
        env.update({key: value for key, value in fast_defaults.items() if key != "MTPLX_LAZY_VERIFY_LOGITS"})
    elif profile == "lazy_verify_off_split_eval":
        env.update({key: value for key, value in fast_defaults.items() if key != "MTPLX_LAZY_VERIFY_LOGITS"})
        env["MTPLX_SPLIT_VERIFY_EVAL"] = "1"
        info["profile_type"] = "eval_split"
    elif profile == "lazy_mtp_history_off":
        env.update({key: value for key, value in fast_defaults.items() if key != "MTPLX_LAZY_MTP_HISTORY_APPEND"})
    elif profile == "both_lazy_off":
        env.update(
            {
                key: value
                for key, value in fast_defaults.items()
                if key not in {"MTPLX_LAZY_VERIFY_LOGITS", "MTPLX_LAZY_MTP_HISTORY_APPEND"}
            }
        )
    elif profile == "all_fast_off":
        info["profile_type"] = "eager_control"
    else:
        normalized_profile = profile.lower().replace("-", "_")
        native_gdn_tail = False
        native_gdn_tail_simdgroups = None
        native_gdn_tail_match = re.search(r"_native_gdn_tail(?:_sg([24]))?$", normalized_profile)
        if native_gdn_tail_match:
            native_gdn_tail = True
            native_gdn_tail_simdgroups = native_gdn_tail_match.group(1)
            normalized_profile = normalized_profile[: native_gdn_tail_match.start()]
        detach_mode_re = r"eval_only|contiguous_eval|selected_slice_contiguous_eval|metal_copy_leaf"
        match = re.fullmatch(r"materialize[_-](\d+)", profile)
        clear_match = re.fullmatch(r"clear[_-]cache[_-](\d+)", profile)
        trunk_match = re.fullmatch(r"trunk[_-]materialize[_-](\d+)", profile)
        trunk_clear_match = re.fullmatch(
            r"trunk[_-]materialize[_-](\d+)[_-]clear[_-]cache[_-](\d+)",
            profile,
        )
        state_rebase_match = re.fullmatch(
            (
                r"state_rebase_every_(\d+)"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        cache_limit_match = re.fullmatch(
            r"cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?)",
            normalized_profile,
        )
        detach_match = re.fullmatch(
            (
                r"detach_(gdn|conv|attn|gdn_conv)_every_(\d+)"
                rf"(?:_mode_({detach_mode_re}))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        capture_commit_detach_match = re.fullmatch(
            (
                r"capture_commit_detach_(gdn|conv|gdn_conv)_every_(\d+)"
                rf"(?:_mode_({detach_mode_re}))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        owned_recurrent_match = re.fullmatch(
            (
                r"owned_recurrent_state"
                r"(?:_mode_(persistent_eval))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        owned_attn_tail_match = re.fullmatch(
            (
                r"owned_attn_tail"
                rf"(?:_mode_({detach_mode_re}))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        owned_attn_block_match = re.fullmatch(
            (
                r"owned_attn_block(?:_size_(\d+))?"
                rf"(?:_mode_({detach_mode_re}))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        live_outputs_match = re.fullmatch(
            (
                r"detach_live_outputs"
                rf"(?:_mode_({detach_mode_re}))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        split_full_attn_match = re.fullmatch(
            (
                r"split_full_attn"
                r"(?:_chunk_(\d+))?"
                r"(?:_threshold_(\d+))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        sdpa_2pass_match = re.fullmatch(
            (
                r"sdpa_2pass"
                r"(?:_threshold_(\d+))?"
                r"(?:_max_q_(\d+))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        blockwise_attn_match = re.fullmatch(
            (
                r"blockwise_attn(?:_block_size_(\d+))?"
                r"(?:_threshold_(\d+))?"
                rf"(?:_mode_({detach_mode_re}))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
            ),
            normalized_profile,
        )
        vllm_metal_paged_match = re.fullmatch(
            (
                r"vllm_metal_paged_attn"
                r"(?:_(partitioned))?"
                r"(?:_block(?:_size)?_(\d+))?"
                r"(?:_blocks_(\d+))?"
                r"(?:_partition_threshold_(\d+))?"
                r"(?:_impl_(fast_sdpa_gather|sdpa_gather|exact_gather|fp32_paged|paged_fp32|sdpa_2pass_paged|mlx_vector_paged))?"
                r"(?:_exact_gather_last_(\d+))?"
                r"(?:_exact_gather_indices_([0-9]+(?:_[0-9]+)*))?"
                r"(_native_mlp(?:_after_(\d+))?)?"
                r"(?:_mlp_variant_"
                r"(compiled_shapeless|tiled_gateup|rowwise_sg4|split_gateup|native_rowwise|native_full_rowwise)"
                r"(?:_after_(\d+))?)?"
                r"(_mtp_paged)?"
                rf"(?:_detach_live_outputs(?:_mode_({detach_mode_re}))?)?"
                r"(?:_trunk_materialize_(\d+))?"
                r"(?:_cache_limit_(0|[0-9]+(?:mb|mib|gb|gib)?))?"
                r"(_snapshot)?"
                r"(?:_window_(\d+))?"
                r"(?:_turboquant(?:_k_([a-z0-9]+(?:_[0-9]+)?))?"
                r"(?:_v_([a-z0-9]+(?:_[0-9]+)?))?)?"
            ),
            normalized_profile,
        )
        if match:
            every = int(match.group(1))
            env.update(fast_defaults)
            env["MTPLX_MTP_HISTORY_MATERIALIZE_EVERY"] = str(every)
            info["profile_type"] = "materialize"
            info["materialize_every"] = every
        elif clear_match:
            every = int(clear_match.group(1))
            env.update(fast_defaults)
            env["MTPLX_CLEAR_CACHE_EVERY"] = str(every)
            info["profile_type"] = "clear_cache_probe"
            info["clear_cache_every"] = every
        elif trunk_match:
            every = int(trunk_match.group(1))
            env.update(fast_defaults)
            env["MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY"] = str(every)
            info["profile_type"] = "trunk_materialize_probe"
            info["trunk_cache_materialize_every"] = every
        elif trunk_clear_match:
            trunk_every = int(trunk_clear_match.group(1))
            clear_every = int(trunk_clear_match.group(2))
            env.update(fast_defaults)
            env["MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY"] = str(trunk_every)
            env["MTPLX_CLEAR_CACHE_EVERY"] = str(clear_every)
            info["profile_type"] = "trunk_materialize_clear_cache_probe"
            info["trunk_cache_materialize_every"] = trunk_every
            info["clear_cache_every"] = clear_every
        elif state_rebase_match:
            every = int(state_rebase_match.group(1))
            cache_limit = state_rebase_match.group(2)
            env.update(fast_defaults)
            env["MTPLX_STATE_REBASE_EVERY"] = str(every)
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "state_rebase_probe"
            info["state_rebase_every"] = every
            info["mlx_cache_limit"] = cache_limit
        elif cache_limit_match:
            limit = cache_limit_match.group(1)
            env.update(fast_defaults)
            env["MTPLX_MLX_CACHE_LIMIT"] = limit
            info["profile_type"] = "cache_limit_probe"
            info["mlx_cache_limit"] = limit
        elif detach_match:
            component_group = detach_match.group(1)
            every = int(detach_match.group(2))
            mode = detach_match.group(3) or "selected_slice_contiguous_eval"
            cache_limit = detach_match.group(4)
            components = (
                ["gdn", "conv"]
                if component_group == "gdn_conv"
                else [component_group]
            )
            env.update(fast_defaults)
            env["MTPLX_DETACH_COMPONENTS"] = ",".join(components)
            env["MTPLX_DETACH_MODE"] = mode
            if "gdn" in components:
                env["MTPLX_DETACH_GDN_EVERY"] = str(every)
            if "conv" in components:
                env["MTPLX_DETACH_CONV_EVERY"] = str(every)
            if "attn" in components:
                env["MTPLX_DETACH_ATTN_EVERY"] = str(every)
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "dirty_detach_probe"
            info["detach_components"] = components
            info["detach_every"] = every
            info["detach_mode"] = mode
            info["mlx_cache_limit"] = cache_limit
        elif capture_commit_detach_match:
            component_group = capture_commit_detach_match.group(1)
            every = int(capture_commit_detach_match.group(2))
            mode = capture_commit_detach_match.group(3) or "selected_slice_contiguous_eval"
            cache_limit = capture_commit_detach_match.group(4)
            components = (
                ["gdn", "conv"]
                if component_group == "gdn_conv"
                else [component_group]
            )
            env.update(fast_defaults)
            env["MTPLX_CAPTURE_COMMIT_DETACH_COMPONENTS"] = ",".join(components)
            env["MTPLX_CAPTURE_COMMIT_DETACH_MODE"] = mode
            if "gdn" in components:
                env["MTPLX_CAPTURE_COMMIT_DETACH_GDN_EVERY"] = str(every)
            if "conv" in components:
                env["MTPLX_CAPTURE_COMMIT_DETACH_CONV_EVERY"] = str(every)
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "capture_commit_detach_probe"
            info["capture_commit_detach_components"] = components
            info["capture_commit_detach_every"] = every
            info["capture_commit_detach_mode"] = mode
            info["mlx_cache_limit"] = cache_limit
        elif owned_recurrent_match:
            mode = owned_recurrent_match.group(1) or "persistent_eval"
            cache_limit = owned_recurrent_match.group(2)
            env.update(fast_defaults)
            env["MTPLX_OWNED_RECURRENT_STATE"] = "1"
            env["MTPLX_OWNED_RECURRENT_STATE_MODE"] = mode
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "owned_recurrent_state_probe"
            info["owned_recurrent_state"] = True
            info["owned_recurrent_state_mode"] = mode
            info["mlx_cache_limit"] = cache_limit
        elif owned_attn_tail_match:
            mode = owned_attn_tail_match.group(1) or "contiguous_eval"
            cache_limit = owned_attn_tail_match.group(2)
            env.update(fast_defaults)
            env["MTPLX_OWNED_ATTN_KV"] = "tail"
            env["MTPLX_OWNED_ATTN_KV_MODE"] = mode
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "owned_attn_tail_probe"
            info["owned_attn_kv"] = "tail"
            info["owned_attn_kv_mode"] = mode
            info["mlx_cache_limit"] = cache_limit
        elif owned_attn_block_match:
            block_size = owned_attn_block_match.group(1) or "1024"
            mode = owned_attn_block_match.group(2) or "contiguous_eval"
            cache_limit = owned_attn_block_match.group(3)
            env.update(fast_defaults)
            env["MTPLX_OWNED_ATTN_KV"] = "block"
            env["MTPLX_OWNED_ATTN_KV_BLOCK_SIZE"] = block_size
            env["MTPLX_OWNED_ATTN_KV_MODE"] = mode
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "owned_attn_block_probe"
            info["owned_attn_kv"] = "block"
            info["owned_attn_kv_block_size"] = int(block_size)
            info["owned_attn_kv_mode"] = mode
            info["mlx_cache_limit"] = cache_limit
        elif live_outputs_match:
            mode = live_outputs_match.group(1) or "contiguous_eval"
            cache_limit = live_outputs_match.group(2)
            env.update(fast_defaults)
            env["MTPLX_DETACH_LIVE_OUTPUTS"] = "1"
            env["MTPLX_DETACH_LIVE_OUTPUTS_MODE"] = mode
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "live_output_detach_probe"
            info["live_output_detach"] = True
            info["live_output_detach_mode"] = mode
            info["mlx_cache_limit"] = cache_limit
        elif split_full_attn_match:
            chunk_size = split_full_attn_match.group(1) or "1"
            threshold = split_full_attn_match.group(2) or "1024"
            cache_limit = split_full_attn_match.group(3)
            env.update(fast_defaults)
            env["MTPLX_SPLIT_FULL_ATTN"] = "1"
            env["MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE"] = chunk_size
            env["MTPLX_SPLIT_FULL_ATTN_THRESHOLD"] = threshold
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "split_full_attn_probe"
            info["split_full_attn"] = True
            info["split_full_attn_chunk_size"] = int(chunk_size)
            info["split_full_attn_threshold"] = int(threshold)
            info["mlx_cache_limit"] = cache_limit
        elif sdpa_2pass_match:
            threshold = sdpa_2pass_match.group(1) or "1024"
            max_q = sdpa_2pass_match.group(2) or "16"
            cache_limit = sdpa_2pass_match.group(3)
            env.update(fast_defaults)
            env["MTPLX_SDPA_2PASS"] = "1"
            env["MTPLX_SDPA_2PASS_THRESHOLD"] = threshold
            env["MTPLX_SDPA_2PASS_MAX_Q"] = max_q
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "sdpa_2pass_probe"
            info["sdpa_2pass"] = True
            info["sdpa_2pass_threshold"] = int(threshold)
            info["sdpa_2pass_max_q"] = int(max_q)
            info["mlx_cache_limit"] = cache_limit
        elif blockwise_attn_match:
            block_size = blockwise_attn_match.group(1) or "1024"
            threshold = blockwise_attn_match.group(2) or "1024"
            mode = blockwise_attn_match.group(3) or "contiguous_eval"
            cache_limit = blockwise_attn_match.group(4)
            env.update(fast_defaults)
            env["MTPLX_OWNED_ATTN_KV"] = "block"
            env["MTPLX_OWNED_ATTN_KV_BLOCK_SIZE"] = block_size
            env["MTPLX_OWNED_ATTN_KV_MODE"] = mode
            env["MTPLX_BLOCKWISE_ATTN"] = "1"
            env["MTPLX_BLOCKWISE_ATTN_THRESHOLD"] = threshold
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            info["profile_type"] = "blockwise_attn_probe"
            info["blockwise_attn"] = True
            info["blockwise_attn_threshold"] = int(threshold)
            info["owned_attn_kv"] = "block"
            info["owned_attn_kv_block_size"] = int(block_size)
            info["owned_attn_kv_mode"] = mode
            info["mlx_cache_limit"] = cache_limit
        elif vllm_metal_paged_match:
            partitioned = vllm_metal_paged_match.group(1) is not None
            block_size = vllm_metal_paged_match.group(2) or "16"
            num_blocks = vllm_metal_paged_match.group(3) or "1024"
            partition_threshold = vllm_metal_paged_match.group(4)
            attention_impl = vllm_metal_paged_match.group(5)
            exact_gather_last_n = vllm_metal_paged_match.group(6)
            exact_gather_indices = vllm_metal_paged_match.group(7)
            native_mlp = vllm_metal_paged_match.group(8) is not None
            native_mlp_after = vllm_metal_paged_match.group(9)
            mlp_call_variant = vllm_metal_paged_match.group(10)
            mlp_call_variant_after = vllm_metal_paged_match.group(11)
            mtp_paged = vllm_metal_paged_match.group(12) is not None
            live_output_mode = vllm_metal_paged_match.group(13)
            live_output_detach = "_detach_live_outputs" in normalized_profile
            trunk_materialize_every = vllm_metal_paged_match.group(14)
            cache_limit = vllm_metal_paged_match.group(15)
            snapshot_required = vllm_metal_paged_match.group(16) is not None
            sliding_window = vllm_metal_paged_match.group(17)
            turboquant_k = vllm_metal_paged_match.group(18)
            turboquant_v = vllm_metal_paged_match.group(19)
            turboquant_enabled = (
                "_turboquant" in normalized_profile
                or turboquant_k is not None
                or turboquant_v is not None
            )
            env.update(fast_defaults)
            if snapshot_required:
                env.pop("MTPLX_SKIP_VERIFY_SNAPSHOT", None)
            if native_mlp:
                env["MTPLX_NATIVE_MLP_ROWWISE"] = "1"
                if native_mlp_after is not None:
                    env["MTPLX_NATIVE_MLP_CONTEXT_THRESHOLD"] = native_mlp_after
            if mlp_call_variant is not None:
                env["MTPLX_MLP_CALL_VARIANT"] = mlp_call_variant
                if mlp_call_variant_after is not None:
                    env["MTPLX_NATIVE_MLP_CONTEXT_THRESHOLD"] = (
                        mlp_call_variant_after
                    )
            if live_output_detach:
                env["MTPLX_DETACH_LIVE_OUTPUTS"] = "1"
                env["MTPLX_DETACH_LIVE_OUTPUTS_MODE"] = (
                    live_output_mode or "contiguous_eval"
                )
            env["MTPLX_VLLM_METAL_PAGED_ATTN"] = "1"
            env["MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE"] = block_size
            env["MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"] = num_blocks
            if attention_impl is not None:
                env["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] = attention_impl
            if exact_gather_last_n is not None:
                env["MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_LAST_N"] = (
                    exact_gather_last_n
                )
            if exact_gather_indices is not None:
                env["MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_INDICES"] = (
                    exact_gather_indices.replace("_", ",")
                )
            if mtp_paged:
                env["MTPLX_VLLM_METAL_PAGED_MTP_ATTN"] = "1"
                env["MTPLX_VLLM_METAL_PAGED_MTP_BLOCK_SIZE"] = block_size
                env["MTPLX_VLLM_METAL_PAGED_MTP_NUM_BLOCKS"] = num_blocks
            if partitioned:
                env["MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"] = "1"
            if partition_threshold is not None:
                env["MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"] = partition_threshold
            if trunk_materialize_every is not None:
                env["MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY"] = trunk_materialize_every
            if cache_limit is not None:
                env["MTPLX_MLX_CACHE_LIMIT"] = cache_limit
            if sliding_window is not None:
                env["MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW"] = sliding_window
            if turboquant_enabled:
                env["MTPLX_VLLM_METAL_PAGED_TURBOQUANT"] = "1"
                env["MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT"] = (
                    turboquant_k or "q8_0"
                )
                env["MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT"] = (
                    turboquant_v or "q3_0"
                )
            if native_gdn_tail:
                env["MTPLX_NATIVE_GDN_TAIL"] = "1"
                if native_gdn_tail_simdgroups is not None:
                    env["MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS"] = native_gdn_tail_simdgroups
            info["profile_type"] = "vllm_metal_paged_attn_probe"
            info["vllm_metal_paged_attn"] = True
            info["vllm_metal_paged_partitioned_attn"] = bool(partitioned)
            info["vllm_metal_paged_block_size"] = int(block_size)
            info["vllm_metal_paged_num_blocks"] = int(num_blocks)
            info["vllm_metal_paged_partition_threshold"] = (
                int(partition_threshold) if partition_threshold is not None else None
            )
            info["vllm_metal_paged_attn_impl"] = attention_impl
            info["vllm_metal_paged_exact_gather_last_n"] = (
                int(exact_gather_last_n) if exact_gather_last_n is not None else None
            )
            info["vllm_metal_paged_exact_gather_indices"] = (
                [int(item) for item in exact_gather_indices.split("_")]
                if exact_gather_indices is not None
                else None
            )
            info["vllm_metal_paged_mtp_attn"] = bool(mtp_paged)
            info["vllm_metal_paged_sliding_window"] = (
                int(sliding_window) if sliding_window is not None else None
            )
            info["vllm_metal_paged_turboquant"] = bool(turboquant_enabled)
            info["vllm_metal_paged_turboquant_k_quant"] = (
                turboquant_k or "q8_0" if turboquant_enabled else None
            )
            info["vllm_metal_paged_turboquant_v_quant"] = (
                turboquant_v or "q3_0" if turboquant_enabled else None
            )
            info["native_gdn_tail"] = bool(native_gdn_tail)
            info["native_gdn_tail_simdgroups"] = (
                int(native_gdn_tail_simdgroups)
                if native_gdn_tail_simdgroups is not None
                else (2 if native_gdn_tail else None)
            )
            info["native_mlp_rowwise"] = bool(native_mlp)
            info["native_mlp_context_threshold"] = (
                int(native_mlp_after) if native_mlp_after is not None else None
            )
            info["mlp_call_variant"] = mlp_call_variant
            info["mlp_call_variant_context_threshold"] = (
                int(mlp_call_variant_after)
                if mlp_call_variant_after is not None
                else None
            )
            info["live_output_detach"] = bool(live_output_detach)
            info["live_output_detach_mode"] = (
                live_output_mode or "contiguous_eval"
                if live_output_detach
                else None
            )
            info["snapshot_required"] = bool(snapshot_required)
            info["trunk_cache_materialize_every"] = (
                int(trunk_materialize_every)
                if trunk_materialize_every is not None
                else None
            )
            info["mlx_cache_limit"] = cache_limit
        else:
            raise ValueError(f"unknown local ablation profile: {profile}")
    info["fast_path_env"] = {key: env.get(key) for key in FAST_PATH_FLAGS}
    info["mtp_history_materialize_every"] = env.get("MTPLX_MTP_HISTORY_MATERIALIZE_EVERY")
    info["clear_cache_every_env"] = env.get("MTPLX_CLEAR_CACHE_EVERY")
    info["trunk_cache_materialize_every_env"] = env.get("MTPLX_TRUNK_CACHE_MATERIALIZE_EVERY")
    info["state_rebase_every_env"] = env.get("MTPLX_STATE_REBASE_EVERY")
    info["detach_components_env"] = env.get("MTPLX_DETACH_COMPONENTS")
    info["detach_mode_env"] = env.get("MTPLX_DETACH_MODE")
    info["detach_gdn_every_env"] = env.get("MTPLX_DETACH_GDN_EVERY")
    info["detach_conv_every_env"] = env.get("MTPLX_DETACH_CONV_EVERY")
    info["detach_attn_every_env"] = env.get("MTPLX_DETACH_ATTN_EVERY")
    info["capture_commit_detach_components_env"] = env.get(
        "MTPLX_CAPTURE_COMMIT_DETACH_COMPONENTS"
    )
    info["capture_commit_detach_mode_env"] = env.get(
        "MTPLX_CAPTURE_COMMIT_DETACH_MODE"
    )
    info["capture_commit_detach_gdn_every_env"] = env.get(
        "MTPLX_CAPTURE_COMMIT_DETACH_GDN_EVERY"
    )
    info["capture_commit_detach_conv_every_env"] = env.get(
        "MTPLX_CAPTURE_COMMIT_DETACH_CONV_EVERY"
    )
    info["owned_attn_kv_env"] = env.get("MTPLX_OWNED_ATTN_KV")
    info["owned_attn_kv_mode_env"] = env.get("MTPLX_OWNED_ATTN_KV_MODE")
    info["owned_attn_kv_step_env"] = env.get("MTPLX_OWNED_ATTN_KV_STEP")
    info["owned_attn_kv_block_size_env"] = env.get("MTPLX_OWNED_ATTN_KV_BLOCK_SIZE")
    info["vllm_metal_paged_attn_env"] = env.get("MTPLX_VLLM_METAL_PAGED_ATTN")
    info["vllm_metal_paged_block_size_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_BLOCK_SIZE"
    )
    info["vllm_metal_paged_num_blocks_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_NUM_BLOCKS"
    )
    info["vllm_metal_paged_attn_impl_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"
    )
    info["vllm_metal_paged_exact_gather_last_n_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_LAST_N"
    )
    info["vllm_metal_paged_exact_gather_indices_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_ATTN_EXACT_GATHER_INDICES"
    )
    info["vllm_metal_paged_sliding_window_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_SLIDING_WINDOW"
    )
    info["vllm_metal_paged_partitioned_attn_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_PARTITIONED_ATTN"
    )
    info["vllm_metal_paged_partition_threshold_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"
    )
    info["vllm_metal_paged_partition_size_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_PARTITION_SIZE"
    )
    info["vllm_metal_paged_mtp_attn_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_MTP_ATTN"
    )
    info["vllm_metal_paged_mtp_block_size_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_MTP_BLOCK_SIZE"
    )
    info["vllm_metal_paged_mtp_num_blocks_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_MTP_NUM_BLOCKS"
    )
    info["vllm_metal_paged_turboquant_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT"
    )
    info["vllm_metal_paged_turboquant_k_quant_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_K_QUANT"
    )
    info["vllm_metal_paged_turboquant_v_quant_env"] = env.get(
        "MTPLX_VLLM_METAL_PAGED_TURBOQUANT_V_QUANT"
    )
    info["native_gdn_tail_env"] = env.get("MTPLX_NATIVE_GDN_TAIL")
    info["native_gdn_tail_simdgroups_env"] = env.get("MTPLX_NATIVE_GDN_TAIL_SIMDGROUPS")
    info["owned_recurrent_state_env"] = env.get("MTPLX_OWNED_RECURRENT_STATE")
    info["owned_recurrent_state_mode_env"] = env.get("MTPLX_OWNED_RECURRENT_STATE_MODE")
    info["live_output_detach_env"] = env.get("MTPLX_DETACH_LIVE_OUTPUTS")
    info["live_output_detach_mode_env"] = env.get("MTPLX_DETACH_LIVE_OUTPUTS_MODE")
    info["split_full_attn_env"] = env.get("MTPLX_SPLIT_FULL_ATTN")
    info["split_full_attn_chunk_size_env"] = env.get("MTPLX_SPLIT_FULL_ATTN_CHUNK_SIZE")
    info["split_full_attn_threshold_env"] = env.get("MTPLX_SPLIT_FULL_ATTN_THRESHOLD")
    info["sdpa_2pass_env"] = env.get("MTPLX_SDPA_2PASS")
    info["sdpa_2pass_threshold_env"] = env.get("MTPLX_SDPA_2PASS_THRESHOLD")
    info["sdpa_2pass_max_q_env"] = env.get("MTPLX_SDPA_2PASS_MAX_Q")
    info["blockwise_attn_env"] = env.get("MTPLX_BLOCKWISE_ATTN")
    info["blockwise_attn_threshold_env"] = env.get("MTPLX_BLOCKWISE_ATTN_THRESHOLD")
    info["target_layer_eval_every_env"] = env.get("MTPLX_TARGET_LAYER_EVAL_EVERY")
    info["target_layer_eval_schedule_env"] = env.get("MTPLX_TARGET_LAYER_EVAL_SCHEDULE")
    info["target_layer_eval_context_threshold_env"] = env.get(
        "MTPLX_TARGET_LAYER_EVAL_CONTEXT_THRESHOLD"
    )
    info["target_layer_eval_max_q_env"] = env.get("MTPLX_TARGET_LAYER_EVAL_MAX_Q")
    info["graphbank_preserve_paged_kv_env"] = env.get(
        "MTPLX_GRAPHBANK_PRESERVE_PAGED_KV"
    )
    info["graphbank_paged_static_max_offset_env"] = env.get(
        "MTPLX_GRAPHBANK_PAGED_STATIC_MAX_OFFSET"
    )
    info["late_depth_switch_after_tokens_env"] = env.get(
        "MTPLX_LATE_DEPTH_SWITCH_AFTER_TOKENS"
    )
    info["late_depth_before_env"] = env.get("MTPLX_LATE_DEPTH_BEFORE")
    info["late_depth_after_env"] = env.get("MTPLX_LATE_DEPTH_AFTER")
    info["mtp_position_mode_env"] = env.get("MTPLX_MTP_POSITION_MODE")
    info["mtp_position_cap_env"] = env.get("MTPLX_MTP_POSITION_CAP")
    info["mtp_position_period_env"] = env.get("MTPLX_MTP_POSITION_PERIOD")
    info["mtp_position_base_env"] = env.get("MTPLX_MTP_POSITION_BASE")
    info["mlx_cache_limit_env"] = env.get("MTPLX_MLX_CACHE_LIMIT")
    info["split_verify_eval"] = env.get("MTPLX_SPLIT_VERIFY_EVAL")
    info["defer_verify_hidden_eval_env"] = env.get("MTPLX_DEFER_VERIFY_HIDDEN_EVAL")
    return env, info


def _wait_for_server(base_url: str, proc: subprocess.Popen[str], *, timeout_s: float) -> dict[str, Any]:
    deadline = time.perf_counter() + timeout_s
    last_probe: dict[str, Any] | None = None
    while time.perf_counter() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before readiness with code {proc.returncode}")
        last_probe = _http_json(base_url.rstrip("/") + "/health", timeout=5)
        if last_probe.get("ok"):
            return last_probe
        time.sleep(2.0)
    raise TimeoutError(f"server did not become ready within {timeout_s:.0f}s; last_probe={last_probe}")


def _read_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _diagnostic_summary_from_row(row: dict[str, Any]) -> dict[str, Any]:
    stats = row.get("mtplx_stats") or {}
    trace_summary = _mtplx_trace_summary(stats)
    first64 = (
        stats.get("sliding_decode_tok_s_first_64")
        or row.get("token_like_first64_s")
    )
    last64 = (
        stats.get("sliding_decode_tok_s_last_64")
        or row.get("token_like_last64_s")
    )
    ratio = (last64 / first64) if first64 and last64 else row.get("token_like_last_over_first")
    first10 = trace_summary.get("trace_first10_tok_s")
    last10 = trace_summary.get("trace_last10_tok_s")
    late_verify = trace_summary.get("trace_last10_verify_ms")
    cache_last = trace_summary.get("trace_last_cache_gib")
    cache_plateaued = trace_summary.get("trace_cache_plateaued")
    hard_gates = {
        "last64_over_first64_ge_0_90": bool(ratio is not None and ratio >= 0.90),
        "last10_ge_0_85_first10": bool(
            first10 is not None and last10 is not None and last10 >= 0.85 * first10
        ),
        "late_verify_le_75ms": bool(late_verify is not None and late_verify <= 75.0),
        "cache_lt_4gib_or_plateaued": bool(
            (cache_last is not None and cache_last < 4.0) or bool(cache_plateaued)
        ),
        "no_hidden_max_tokens": bool(not row.get("request_max_tokens_present")),
        "http_ok": bool(row.get("status") and int(row.get("status")) < 400 and not row.get("error")),
    }
    return {
        "label": row.get("label"),
        "test": row.get("test"),
        "status": row.get("status"),
        "finish_reason": row.get("finish_reason"),
        "request_elapsed_s": row.get("request_elapsed_s"),
        "prompt_tokens": row.get("prompt_tokens") or stats.get("prompt_tokens"),
        "completion_tokens": (
            row.get("completion_tokens")
            or stats.get("completion_tokens")
            or stats.get("generated_tokens")
            or trace_summary.get("trace_generated_tokens")
        ),
        "decode_tok_s": (
            stats.get("decode_tok_s")
            or row.get("total_decode_tok_s_from_usage")
            or (
                float(trace_summary["trace_generated_tokens"]) / float(trace_summary["trace_elapsed_s"])
                if trace_summary.get("trace_generated_tokens") and trace_summary.get("trace_elapsed_s")
                else None
            )
        ),
        "first64": first64,
        "last64": last64,
        "last64_over_first64": ratio,
        "first10_tok_s": first10,
        "last10_tok_s": last10,
        "last10_over_first10": (last10 / first10) if first10 and last10 else None,
        "late_verify_ms": late_verify,
        "last_cache_gib": cache_last,
        "cache_plateaued": cache_plateaued,
        "request_max_tokens_present": row.get("request_max_tokens_present"),
        "trace_summary": trace_summary,
        "hard_gates": hard_gates,
        "hard_gate_pass": all(hard_gates.values()),
        "content_path": row.get("content_path"),
        "events_path": row.get("events_path"),
        "error": row.get("error"),
    }


def _select_materialization_winner(rows: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [
        row
        for row in rows
        if row.get("profile_type")
        in {
            "materialize",
            "trunk_materialize_probe",
            "trunk_materialize_clear_cache_probe",
            "cache_limit_probe",
            "state_rebase_probe",
            "dirty_detach_probe",
            "capture_commit_detach_probe",
            "owned_attn_tail_probe",
            "owned_attn_block_probe",
            "live_output_detach_probe",
        }
        and row.get("test") == "flappy"
        and row.get("hard_gate_pass")
    ]
    if passing:
        winner = max(passing, key=lambda row: float(row.get("decode_tok_s") or 0.0))
        reason = "fastest profile satisfying all strict long-response gates"
        return {"winner_profile": winner.get("profile"), "winner": winner, "reason": reason}
    fallback = next(
        (
            row
            for row in rows
            if row.get("profile") == "both_lazy_off"
            and row.get("test") == "flappy"
            and row.get("hard_gate_pass")
        ),
        None,
    )
    if fallback is not None:
        return {
            "winner_profile": "both_lazy_off",
            "winner": fallback,
            "reason": "no materialization cadence passed; both-lazy-off passes as eager fallback",
        }
    return {
        "winner_profile": None,
        "winner": None,
        "reason": "no ablation profile passed and eager fallback did not satisfy all strict gates",
    }


def _local_adaptive_config(args: argparse.Namespace) -> dict[str, Any]:
    policy = str(getattr(args, "adaptive_policy", "none") or "none")
    if policy == "none":
        return {"policy": "none"}
    config: dict[str, Any] = {
        "policy": policy,
        "min_depth": int(args.adaptive_min_depth),
    }
    if policy == "streak":
        config.update(
            {
                "start_depth": int(args.adaptive_start_depth),
                "increase_after": int(args.adaptive_increase_after),
                "decrease_after": int(args.adaptive_decrease_after),
            }
        )
    elif policy == "expected_value":
        config.update(
            {
                "base_depth": int(args.adaptive_ev_base_depth),
                "accept_priors": [float(v) for v in args.adaptive_ev_accept_priors],
                "draft_cost_s": float(args.adaptive_ev_draft_cost_s),
                "extra_verify_cost_s": float(args.adaptive_ev_extra_verify_cost_s),
                "baseline_tok_s": float(args.adaptive_ev_baseline_tok_s),
                "safety_margin": float(args.adaptive_ev_safety_margin),
                "margin_center": float(args.adaptive_ev_margin_center),
                "margin_scale": float(args.adaptive_ev_margin_scale),
                "confidence_weight": float(args.adaptive_ev_confidence_weight),
                "min_extra_accept_probability": float(
                    args.adaptive_ev_min_extra_accept_probability
                ),
            }
        )
    return config


def _local_adaptive_server_args(args: argparse.Namespace) -> list[str]:
    config = _local_adaptive_config(args)
    if config["policy"] == "none":
        return []
    server_args = [
        "--adaptive-policy",
        str(config["policy"]),
        "--adaptive-min-depth",
        str(config["min_depth"]),
    ]
    if config["policy"] == "streak":
        server_args.extend(
            [
                "--adaptive-start-depth",
                str(config["start_depth"]),
                "--adaptive-increase-after",
                str(config["increase_after"]),
                "--adaptive-decrease-after",
                str(config["decrease_after"]),
            ]
        )
    elif config["policy"] == "expected_value":
        server_args.extend(
            [
                "--adaptive-ev-base-depth",
                str(config["base_depth"]),
                "--adaptive-ev-accept-priors",
                ",".join(f"{float(v):.8g}" for v in config["accept_priors"]),
                "--adaptive-ev-draft-cost-s",
                str(config["draft_cost_s"]),
                "--adaptive-ev-extra-verify-cost-s",
                str(config["extra_verify_cost_s"]),
                "--adaptive-ev-baseline-tok-s",
                str(config["baseline_tok_s"]),
                "--adaptive-ev-safety-margin",
                str(config["safety_margin"]),
                "--adaptive-ev-margin-center",
                str(config["margin_center"]),
                "--adaptive-ev-margin-scale",
                str(config["margin_scale"]),
                "--adaptive-ev-confidence-weight",
                str(config["confidence_weight"]),
                "--adaptive-ev-min-extra-accept-probability",
                str(config["min_extra_accept_probability"]),
            ]
        )
    return server_args


def _local_proposal_cache_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "online_correction_cache": bool(args.online_correction_cache),
        "online_correction_cache_min_depth": int(args.online_correction_cache_min_depth),
        "online_correction_cache_key": str(args.online_correction_cache_key),
        "prompt_correction_cache": bool(args.prompt_correction_cache),
        "prompt_correction_cache_min_depth": int(args.prompt_correction_cache_min_depth),
    }


def _local_proposal_cache_server_args(args: argparse.Namespace) -> list[str]:
    server_args: list[str] = []
    if args.online_correction_cache:
        server_args.extend(
            [
                "--online-correction-cache",
                "--online-correction-cache-min-depth",
                str(args.online_correction_cache_min_depth),
                "--online-correction-cache-key",
                str(args.online_correction_cache_key),
            ]
        )
    if args.prompt_correction_cache:
        server_args.extend(
            [
                "--prompt-correction-cache",
                "--prompt-correction-cache-min-depth",
                str(args.prompt_correction_cache_min_depth),
            ]
        )
    return server_args


def _local_online_hidden_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "alpha": float(getattr(args, "online_hidden_corrector_alpha", 0.0)),
        "decay": float(getattr(args, "online_hidden_corrector_decay", 0.8)),
        "warmup": int(getattr(args, "online_hidden_corrector_warmup", 1)),
        "max_feed_depth": getattr(args, "online_hidden_corrector_max_feed_depth", None),
        "key": str(getattr(args, "online_hidden_corrector_key", "global")),
    }


def _local_online_hidden_server_args(args: argparse.Namespace) -> list[str]:
    config = _local_online_hidden_config(args)
    if config["alpha"] <= 0.0:
        return []
    server_args = [
        "--online-hidden-corrector-alpha",
        str(config["alpha"]),
        "--online-hidden-corrector-decay",
        str(config["decay"]),
        "--online-hidden-corrector-warmup",
        str(config["warmup"]),
        "--online-hidden-corrector-key",
        str(config["key"]),
    ]
    if config["max_feed_depth"] is not None:
        server_args.extend(
            [
                "--online-hidden-corrector-max-feed-depth",
                str(config["max_feed_depth"]),
            ]
        )
    return server_args


def run_local_ablation(args: argparse.Namespace) -> int:
    run_id = args.run_id or f"{args.label}-{_now_stamp()}"
    output_dir = args.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    python_bin = args.python_bin or str(Path(".venv/bin/python"))
    base_url = f"http://{args.host}:{args.port}"
    rows: list[dict[str, Any]] = []
    repo_root = Path.cwd().resolve()
    server_script = repo_root / "scripts" / "serve_openai_mtplx.py"
    _write_json(
        output_dir / "ablation-config.json",
        {
            "run_id": run_id,
            "label": args.label,
            "profiles": profiles,
            "model": args.model,
            "model_id": args.model_id,
            "generation_mode": args.generation_mode,
            "load_mtp": bool(args.load_mtp),
            "depth": args.depth,
            "tests": args.tests,
            "max_tokens_present": args.max_tokens is not None,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "cache_mode": args.cache_mode,
            "port": args.port,
            "adaptive": _local_adaptive_config(args),
            "proposal_cache": _local_proposal_cache_config(args),
            "online_hidden": _local_online_hidden_config(args),
        },
    )
    for profile in profiles:
        profile_dir = output_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        env, profile_info = _local_profile_env(profile, os.environ)
        env["MTPLX_DECODE_TRACE_JSONL"] = str(profile_dir / "decode-trace.jsonl")
        env["MTPLX_DECODE_TRACE_INTERVAL_S"] = str(args.trace_interval_s)
        server_log = profile_dir / "server.log"
        cmd = [
            python_bin,
            str(server_script),
            "--model",
            args.model,
            "--model-id",
            args.model_id,
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--generation-mode",
            args.generation_mode,
            "--load-mtp" if args.load_mtp else "--no-load-mtp",
            "--depth",
            str(args.depth),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--verify-strategy",
            args.verify_strategy,
            "--verify-core",
            args.verify_core,
            "--no-strict-startup-asserts",
            "--diagnostic-env-ablation",
        ]
        cmd.extend(_local_adaptive_server_args(args))
        cmd.extend(_local_proposal_cache_server_args(args))
        cmd.extend(_local_online_hidden_server_args(args))
        if not args.strict_mlx_fork_assert:
            cmd.append("--no-strict-mlx-fork-assert")
        _write_json(profile_dir / "profile-env.json", profile_info)
        print(f"[local-ablation] starting {profile} on {base_url}", flush=True)
        log_handle = server_log.open("w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=repo_root,
            env=_server_python_env(env, repo_root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        profile_summary: dict[str, Any] = {
            **profile_info,
            "profile": profile,
            "server_log": str(server_log),
            "server_cmd": cmd,
            "generation_mode": args.generation_mode,
            "load_mtp": bool(args.load_mtp),
            "cache_mode": args.cache_mode,
            "adaptive": _local_adaptive_config(args),
            "proposal_cache": _local_proposal_cache_config(args),
            "online_hidden": _local_online_hidden_config(args),
        }
        try:
            health = _wait_for_server(base_url, proc, timeout_s=args.startup_timeout_s)
            _write_json(profile_dir / "health-ready.json", health)
            direct_headers = _json_loads(args.headers_json, default={})
            if args.cache_mode != "default":
                direct_headers = {
                    **direct_headers,
                    "x-mtplx-cache-mode": args.cache_mode,
                }
            direct_metadata = {
                **_json_loads(args.metadata_json, default={}),
                "ablation_profile": profile,
                "ablation_run_id": run_id,
            }
            direct_args = argparse.Namespace(
                base_url=base_url,
                label=f"{args.label}-{profile}",
                model=args.model_id,
                endpoint="chat",
                tests=args.tests,
                output_dir=profile_dir,
                run_id="direct",
                system_prompt=args.system_prompt,
                stream=True,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                timeout=args.request_timeout_s,
                headers_json=json.dumps(direct_headers) if direct_headers else None,
                metadata_json=json.dumps(direct_metadata),
            )
            code = run_direct(direct_args)
            profile_summary["direct_returncode"] = code
            results_path = profile_dir / "direct" / "results.jsonl"
            for result_row in _read_results(results_path):
                summary = _diagnostic_summary_from_row(result_row)
                rows.append({**profile_summary, **summary, "results_path": str(results_path)})
        except Exception as exc:
            profile_summary.update({"error": {"type": type(exc).__name__, "message": str(exc)}})
            rows.append(profile_summary)
            print(f"[local-ablation] {profile} failed: {exc}", flush=True)
        finally:
            if proc.poll() is None:
                proc.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=30)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=30)
            log_handle.close()
    decision = _select_materialization_winner(rows)
    result = {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "rows": rows,
        "decision": decision,
    }
    _write_json(output_dir / "ablation-summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    env = sub.add_parser("remote-env")
    env.add_argument("--ssh-host", default="mtplx-3090")
    env.add_argument("--label", default="vllm-linux")
    env.add_argument("--output-dir", type=Path, default=Path("outputs/context-diagnostics"))
    env.add_argument("--timeout", type=int, default=60)
    env.add_argument("--probe-ports", type=int, nargs="*", default=[5004, 5001, 8000])
    env.set_defaults(func=collect_remote_env)

    direct = sub.add_parser("direct")
    direct.add_argument("--base-url", required=True)
    direct.add_argument("--label", required=True)
    direct.add_argument("--model", required=True)
    direct.add_argument("--endpoint", choices=["chat", "completions"], default="chat")
    direct.add_argument("--tests", default="hi,flappy")
    direct.add_argument("--output-dir", type=Path, default=Path("outputs/context-diagnostics"))
    direct.add_argument("--run-id")
    direct.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    direct.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True)
    direct.add_argument("--max-tokens", type=int)
    direct.add_argument("--temperature", type=float, default=0.6)
    direct.add_argument("--top-p", type=float, default=0.95)
    direct.add_argument("--top-k", type=int, default=20)
    direct.add_argument("--seed", type=int, default=42)
    direct.add_argument("--timeout", type=int, default=1800)
    direct.add_argument("--headers-json")
    direct.add_argument("--metadata-json")
    direct.set_defaults(func=run_direct)

    summ = sub.add_parser("summarize")
    summ.add_argument("results", type=Path, nargs="+")
    summ.add_argument("--output", type=Path)
    summ.set_defaults(func=summarize)

    local = sub.add_parser("local-ablation")
    local.add_argument("--label", default="mlx-lazy-ablation")
    local.add_argument("--output-dir", type=Path, default=Path("outputs/context-diagnostics"))
    local.add_argument("--run-id")
    local.add_argument("--python-bin", default=str(Path(".venv/bin/python")))
    local.add_argument("--host", default="127.0.0.1")
    local.add_argument("--port", type=int, default=8011)
    local.add_argument("--model", default="models/Qwen3.6-27B-MTPLX-GDN8-Speed4-CyanKiwiMTP")
    local.add_argument("--model-id", default="mtplx-qwen36-27b-native-mtp")
    local.add_argument(
        "--generation-mode",
        choices=["mtp", "ar"],
        default="mtp",
        help="Run the local server in native-MTP mode or target AR-only diagnostic mode.",
    )
    local.add_argument(
        "--load-mtp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the native MTP sidecar. Use --no-load-mtp for stock AR diagnostics.",
    )
    local.add_argument("--depth", type=int, default=3)
    local.add_argument("--verify-strategy", default="capture_commit")
    local.add_argument("--verify-core", default="linear-gdn-from-conv-tape")
    local.add_argument(
        "--profiles",
        default=(
            "current_baseline,lazy_verify_off,lazy_mtp_history_off,"
            "both_lazy_off,all_fast_off,materialize_16,materialize_32,"
            "materialize_64,materialize_128"
        ),
    )
    local.add_argument("--tests", default="flappy")
    local.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    local.add_argument("--max-tokens", type=int)
    local.add_argument("--temperature", type=float, default=0.6)
    local.add_argument("--top-p", type=float, default=0.95)
    local.add_argument("--top-k", type=int, default=20)
    local.add_argument("--seed", type=int, default=42)
    local.add_argument(
        "--cache-mode",
        choices=["default", "bypass", "stateless", "off"],
        default="default",
        help=(
            "Optional x-mtplx-cache-mode header for local ablations. Use "
            "'bypass' to isolate within-response decode from SessionBank "
            "postcommit snapshots."
        ),
    )
    local.add_argument(
        "--headers-json",
        help="Extra HTTP headers to forward to direct requests, as a JSON object.",
    )
    local.add_argument(
        "--metadata-json",
        help="Extra OpenAI request metadata to forward to direct requests, as a JSON object.",
    )
    local.add_argument(
        "--adaptive-policy",
        choices=["none", "streak", "expected_value"],
        default="none",
    )
    local.add_argument("--adaptive-min-depth", type=int, default=1)
    local.add_argument("--adaptive-start-depth", type=int, default=1)
    local.add_argument("--adaptive-increase-after", type=int, default=4)
    local.add_argument("--adaptive-decrease-after", type=int, default=1)
    local.add_argument("--adaptive-ev-base-depth", type=int, default=2)
    local.add_argument(
        "--adaptive-ev-accept-priors",
        type=_comma_floats,
        default=(0.92, 0.64, 0.32),
    )
    local.add_argument("--adaptive-ev-draft-cost-s", type=float, default=0.0048)
    local.add_argument("--adaptive-ev-extra-verify-cost-s", type=float, default=0.0060)
    local.add_argument("--adaptive-ev-baseline-tok-s", type=float, default=40.0)
    local.add_argument("--adaptive-ev-safety-margin", type=float, default=0.10)
    local.add_argument("--adaptive-ev-margin-center", type=float, default=1.0)
    local.add_argument("--adaptive-ev-margin-scale", type=float, default=2.0)
    local.add_argument("--adaptive-ev-confidence-weight", type=float, default=0.35)
    local.add_argument(
        "--adaptive-ev-min-extra-accept-probability",
        type=float,
        default=0.18,
    )
    local.add_argument(
        "--online-correction-cache",
        action="store_true",
        help=(
            "Forward the exact target-feedback proposal cache to the local MTPLX "
            "server. Diagnostic only; target verification remains authoritative."
        ),
    )
    local.add_argument("--online-correction-cache-min-depth", type=int, default=1)
    local.add_argument(
        "--online-correction-cache-key",
        choices=["local_prefix", "source_token", "primary_source"],
        default="local_prefix",
    )
    local.add_argument(
        "--prompt-correction-cache",
        action="store_true",
        help="Forward the prompt-seeded exact proposal cache to the local server.",
    )
    local.add_argument("--prompt-correction-cache-min-depth", type=int, default=2)
    local.add_argument(
        "--online-hidden-corrector-alpha",
        type=float,
        default=0.0,
        help=(
            "Forward online hidden residual correction to the local server. "
            "Diagnostic only; target verification remains authoritative."
        ),
    )
    local.add_argument("--online-hidden-corrector-decay", type=float, default=0.8)
    local.add_argument("--online-hidden-corrector-warmup", type=int, default=1)
    local.add_argument("--online-hidden-corrector-max-feed-depth", type=int)
    local.add_argument(
        "--online-hidden-corrector-key",
        choices=["global", "token"],
        default="global",
    )
    local.add_argument("--trace-interval-s", type=float, default=1.0)
    local.add_argument("--startup-timeout-s", type=float, default=900.0)
    local.add_argument("--request-timeout-s", type=int, default=2400)
    local.add_argument(
        "--strict-mlx-fork-assert",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    local.set_defaults(func=run_local_ablation)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
