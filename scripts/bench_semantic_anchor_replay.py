#!/usr/bin/env python3
"""Replay frozen chat requests for a same-build semantic-anchor ON/OFF comparison.

This measures a running server; it neither starts a model nor resets any cache.
See docs/validation/semantic-anchor-ab.md for isolation and manifest requirements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
from typing import Any, Iterable
import urllib.error
import urllib.parse
import urllib.request

MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_EVENT_BYTES = 4 * 1024 * 1024


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path: str) -> Any:
    with open(path, "rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Input exceeds the 32 MiB bound")
    return json.loads(data)


def sse_payloads(lines: Iterable[bytes]) -> Iterable[dict[str, Any]]:
    data: list[str] = []
    size = 0
    for raw in lines:
        if len(raw) > MAX_EVENT_BYTES:
            raise ValueError("Oversized SSE line")
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if not data:
                continue
            text = "\n".join(data)
            data, size = [], 0
            if text == "[DONE]":
                return
            payload = json.loads(text)
            if not isinstance(payload, dict) or "error" in payload:
                raise ValueError("Server emitted an invalid/error SSE payload")
            yield payload
        elif line.startswith("data:"):
            piece = line[5:].removeprefix(" ")
            size += len(raw)
            if size > MAX_EVENT_BYTES:
                raise ValueError("Oversized SSE event")
            data.append(piece)
    # A closed socket is not an explicit successful end-of-stream.
    raise ValueError("SSE stream ended without [DONE]")


def count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Missing or invalid {name}")
    return value


def measure(payloads: Iterable[dict[str, Any]], started: float,
            clock=time.perf_counter) -> dict[str, Any]:
    first_generated = first_content = None
    content: list[str] = []
    reasoning: list[str] = []
    tools: dict[int, dict[str, str]] = {}
    usage: dict[str, Any] | None = None
    finish = None
    for payload in payloads:
        now = clock()
        if isinstance(payload.get("usage"), dict):
            usage = payload["usage"]
        for choice in payload.get("choices") or []:
            if choice.get("index", 0) != 0:
                raise ValueError("Only single-choice requests are supported")
            delta = choice.get("delta") or {}
            generated = False
            for key, destination in (("content", content), ("reasoning_content", reasoning)):
                value = delta.get(key)
                if isinstance(value, str) and value:
                    destination.append(value)
                    generated = True
                    if key == "content" and first_content is None:
                        first_content = now - started
            for tool in delta.get("tool_calls") or []:
                index = count(tool.get("index"), "tool index")
                target = tools.setdefault(index, {"name": "", "arguments": ""})
                for key in ("name", "arguments"):
                    value = (tool.get("function") or {}).get(key)
                    if isinstance(value, str) and value:
                        target[key] += value
                        generated = True
            if generated and first_generated is None:
                first_generated = now - started
            if choice.get("finish_reason") is not None:
                finish = choice["finish_reason"]
    if first_generated is None or finish not in {"stop", "length", "tool_calls"}:
        raise ValueError("Missing generated output or unsuccessful finish reason")
    if usage is None:
        raise ValueError("Final usage is missing; cache hits cannot be inferred")
    prompt = count(usage.get("prompt_tokens"), "prompt_tokens")
    completion = count(usage.get("completion_tokens"), "completion_tokens")
    cached = count((usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                   "prompt_tokens_details.cached_tokens")
    if prompt == 0 or cached > prompt:
        raise ValueError("Invalid cached/prompt token relationship")
    return {
        "ttft_generated_s": first_generated,
        "ttft_content_s": first_content,
        "wall_s": clock() - started,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "cached_fraction": cached / prompt,
        "completion_tokens": completion,
        "finish_reason": finish,
        "output_sha256": digest({"content": "".join(content),
                                 "reasoning": "".join(reasoning),
                                 "tools": [tools[i] for i in sorted(tools)]}),
    }


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirects are not permitted for benchmark requests")


def run(args: argparse.Namespace) -> dict[str, Any]:
    parsed = urllib.parse.urlsplit(args.base_url)
    if (parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("base-url must be a loopback HTTP origin, without a path or credentials")
    manifest = read_json(args.manifest)
    required = ("server_commit", "model", "model_revision", "tokenizer_revision",
                "mlx_version", "hardware", "server_settings")
    if not isinstance(manifest, dict) or any(not manifest.get(k) for k in required):
        raise ValueError("Manifest lacks a required reproducibility field")
    if type(manifest.get("anchors_enabled")) is not bool:
        raise ValueError("Manifest must explicitly declare anchors_enabled")
    transcript = read_json(args.transcript)
    requests = transcript.get("requests") if isinstance(transcript, dict) else None
    if not isinstance(requests, list) or not 2 <= len(requests) <= 100:
        raise ValueError("A frozen transcript needs 2 to 100 complete request bodies")
    session = transcript.get("session_id")
    if (not isinstance(session, str) or not session or len(session) > 200
            or not session.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in session)):
        raise ValueError("Transcript needs a printable ASCII session_id")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
               "X-MTPLX-Client": "semantic-anchor-replay", "X-MTPLX-Session-ID": session}
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            raise ValueError("The requested API-key environment variable is empty")
        headers["Authorization"] = "Bearer " + key
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    bodies = []
    for original in requests:
        body = json.loads(canonical(original))
        if not isinstance(body, dict) or body.get("model") != manifest["model"]:
            raise ValueError("Every request must name the manifest's model")
        if not body.get("messages") or body.get("temperature") != 0 or body.get("n", 1) != 1:
            raise ValueError("Every request needs messages, temperature=0, and n=1")
        body["stream"] = True
        body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        bodies.append(body)
    rows = []
    for index, body in enumerate(bodies):
        request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
                                         data=canonical(body), headers=headers, method="POST")
        started = time.perf_counter()
        with opener.open(request, timeout=args.timeout_s) as response:
            if response.headers.get_content_type() != "text/event-stream":
                raise ValueError("Server did not return text/event-stream")
            row = measure(sse_payloads(response), started)
        if index == 0 and row["cached_tokens"] != 0:
            raise ValueError("First turn was cached; restart with an isolated empty session bank")
        rows.append({"turn": index, "warm_turn": index > 0,
                     "request_sha256": digest(body), **row})
    return {"schema_version": 1, "kind": "measured_http_replay",
            "manifest_attestation": "operator-supplied; not a server-verified feature-flag assertion",
            "manifest": manifest, "transcript_sha256": digest(transcript),
            "session_sha256": digest(session), "turns": rows}


def compare(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    left, right = dict(off["manifest"]), dict(on["manifest"])
    if left.pop("anchors_enabled") is not False or right.pop("anchors_enabled") is not True:
        raise ValueError("Expected OFF receipt followed by ON receipt")
    if left != right:
        raise ValueError("Arms differ in reproducibility metadata beyond the anchor flag")
    if off["transcript_sha256"] != on["transcript_sha256"]:
        raise ValueError("Arms used different frozen transcripts")
    before, after = off["turns"], on["turns"]
    if len(before) != len(after) or len(before) < 2:
        raise ValueError("Arms have different or insufficient turn counts")
    rows = []
    for a, b in zip(before, after):
        for key in ("turn", "request_sha256", "prompt_tokens"):
            if a[key] != b[key]:
                raise ValueError(f"Arms differ in {key}")
        parity = all(a[key] == b[key] for key in
                     ("output_sha256", "finish_reason", "completion_tokens"))
        rows.append({"turn": a["turn"], "warm_turn": a["turn"] > 0,
                     "output_parity": parity,
                     "off_cached_tokens": a["cached_tokens"],
                     "on_cached_tokens": b["cached_tokens"],
                     "cached_tokens_delta": b["cached_tokens"] - a["cached_tokens"],
                     "off_ttft_generated_s": a["ttft_generated_s"],
                     "on_ttft_generated_s": b["ttft_generated_s"],
                     "ttft_generated_delta_s": b["ttft_generated_s"] - a["ttft_generated_s"]})
    return {"schema_version": 1, "kind": "paired_http_comparison",
            "output_parity": all(row["output_parity"] for row in rows),
            "performance_claim_requires_repeated_isolated_runs": True, "turns": rows}


def write_receipt(path: Path, result: dict[str, Any]) -> None:
    data = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".anchor-receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes atomically without replacing a receipt.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("run")
    replay.add_argument("--base-url", default="http://127.0.0.1:8000")
    replay.add_argument("--transcript", required=True)
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--timeout-s", type=float, default=1800)
    replay.add_argument("--api-key-env")
    pair = sub.add_parser("compare")
    pair.add_argument("--off", required=True)
    pair.add_argument("--on", required=True)
    for command in (replay, pair):
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "run" and not 0 < args.timeout_s <= 7200:
            raise ValueError("timeout-s must be greater than zero and at most 7200")
        result = run(args) if args.command == "run" else compare(read_json(args.off), read_json(args.on))
        write_receipt(Path(args.output), result)
        return 0 if result.get("output_parity", True) else 1
    except (ValueError, KeyError, TypeError, OSError, urllib.error.URLError) as exc:
        parser.exit(2, f"Benchmark incomplete: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
