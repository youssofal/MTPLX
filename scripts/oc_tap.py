#!/usr/bin/env python3
"""OpenCode-to-MTPLX recording tap: a transparent streaming HTTP proxy.

Sits between an OpenCode-style client and the MTPLX server, forwarding
byte-for-byte (including SSE streams) while journaling every request and
response summary as ms-timestamped JSONL. Built 2026-08-01 so client-to-engine
sessions can be diagnosed from raw wire truth:
engine-side request-log records numeric telemetry only, so cross-referencing
WHICH bytes of the conversation a client rewrote between requests needs a
content-visible tap at the HTTP boundary. Point the client's baseURL at this
tap; nothing else changes.

Usage:
  python3 scripts/oc_tap.py --listen 8002 --upstream 127.0.0.1:8001 \
      --journal ~/.mtplx/logs/oc-tap.jsonl

Journal record (one line per request):
  {ts_ms, id, method, path, request_headers_subset, request_body_sha256,
   request_body (chat completions only), status, response_ms, sse_events,
   response_bytes, first_token_ms}
Chat request bodies are stored complete (they are the object of study);
non-chat bodies store only the hash. stdlib only.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time

JOURNAL_LOCK = asyncio.Lock()


def now_ms() -> int:
    return int(time.time() * 1000)


async def journal_write(path: str, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str)
    async with JOURNAL_LOCK:
        with open(path, "a", encoding="utf-8") as sink:
            sink.write(line + "\n")


async def read_http_message(reader: asyncio.StreamReader):
    """Read one HTTP/1.1 message head + body (Content-Length or chunked)."""
    head = await reader.readuntil(b"\r\n\r\n")
    head_text = head.decode("latin-1")
    lines = head_text.split("\r\n")
    start_line = lines[0]
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip().lower()] = value.strip()
    body = b""
    if headers.get("transfer-encoding", "").lower() == "chunked":
        while True:
            size_line = await reader.readuntil(b"\r\n")
            size = int(size_line.strip() or b"0", 16)
            chunk = await reader.readexactly(size + 2)
            body += chunk[:-2]
            if size == 0:
                break
    elif "content-length" in headers:
        body = await reader.readexactly(int(headers["content-length"]))
    return start_line, headers, head, body


async def handle_client(client_reader, client_writer, args):
    peer = client_writer.get_extra_info("peername")
    try:
        while True:
            try:
                start_line, headers, raw_head, body = await read_http_message(
                    client_reader
                )
            except (asyncio.IncompleteReadError, ConnectionResetError):
                return
            method, _, rest = start_line.partition(" ")
            path = rest.rsplit(" ", 1)[0]
            rid = f"tap-{now_ms()}-{os.urandom(3).hex()}"
            t0 = time.monotonic()
            record = {
                "ts_ms": now_ms(),
                "id": rid,
                "peer": str(peer),
                "method": method,
                "path": path,
                "request_body_bytes": len(body),
                "request_body_sha256": hashlib.sha256(body).hexdigest(),
            }
            if "/chat/completions" in path or "/messages" in path:
                try:
                    record["request_body"] = json.loads(body.decode("utf-8"))
                except Exception:
                    record["request_body_raw_prefix"] = body[:2048].decode(
                        "utf-8", "replace"
                    )
            interesting = (
                "x-mtplx-client",
                "x-mtplx-request-id",
                "user-agent",
                "content-length",
            )
            record["request_headers"] = {
                k: headers.get(k) for k in interesting if headers.get(k)
            }

            upstream_host, upstream_port = args.upstream.split(":")
            try:
                up_reader, up_writer = await asyncio.open_connection(
                    upstream_host, int(upstream_port)
                )
            except OSError as exc:
                record["error"] = f"upstream_connect: {exc}"
                await journal_write(args.journal, record)
                client_writer.close()
                return
            up_writer.write(raw_head)
            if body:
                up_writer.write(body)
            await up_writer.drain()

            # Relay the response transparently while counting SSE events.
            status = None
            response_bytes = 0
            sse_events = 0
            first_token_ms = None
            try:
                # status + headers. Force Connection: close toward the client:
                # undici/fetch keep-alive pools reuse sockets that died with an
                # upstream restart and then drop the next request SILENTLY
                # (observed twice live 2026-08-01: chat.headers fired, no
                # bytes ever egressed). Fresh socket per request removes the
                # class; localhost connect cost is nil.
                resp_head = await up_reader.readuntil(b"\r\n\r\n")
                status = resp_head.decode("latin-1").split("\r\n")[0]
                head_text = resp_head.decode("latin-1")
                lines = [
                    l for l in head_text.split("\r\n")
                    if not l.lower().startswith("connection:")
                ]
                lines.insert(1, "Connection: close")
                resp_head = "\r\n".join(lines).encode("latin-1")
                client_writer.write(resp_head)
                await client_writer.drain()
                resp_headers = resp_head.decode("latin-1").lower()
                chunked = "transfer-encoding: chunked" in resp_headers
                content_length = None
                for line in resp_headers.split("\r\n"):
                    if line.startswith("content-length:"):
                        content_length = int(line.split(":", 1)[1].strip())
                if chunked:
                    while True:
                        size_line = await up_reader.readuntil(b"\r\n")
                        client_writer.write(size_line)
                        size = int(size_line.strip() or b"0", 16)
                        chunk = await up_reader.readexactly(size + 2)
                        client_writer.write(chunk)
                        await client_writer.drain()
                        response_bytes += size
                        if size and b"data:" in chunk:
                            sse_events += chunk.count(b"data:")
                            if first_token_ms is None:
                                first_token_ms = int(
                                    (time.monotonic() - t0) * 1000
                                )
                        if size == 0:
                            break
                elif content_length is not None:
                    remaining = content_length
                    while remaining > 0:
                        chunk = await up_reader.read(min(65536, remaining))
                        if not chunk:
                            break
                        client_writer.write(chunk)
                        await client_writer.drain()
                        remaining -= len(chunk)
                        response_bytes += len(chunk)
                else:
                    while True:
                        chunk = await up_reader.read(65536)
                        if not chunk:
                            break
                        client_writer.write(chunk)
                        await client_writer.drain()
                        response_bytes += len(chunk)
            except (asyncio.IncompleteReadError, ConnectionResetError) as exc:
                record["relay_error"] = str(exc)
            finally:
                up_writer.close()

            record["status"] = status
            record["response_ms"] = int((time.monotonic() - t0) * 1000)
            record["response_bytes"] = response_bytes
            record["sse_events"] = sse_events
            record["first_token_ms"] = first_token_ms
            await journal_write(args.journal, record)
    except Exception as exc:  # tap must never take the session down loudly
        sys.stderr.write(f"[oc-tap] handler error: {exc}\n")
    finally:
        try:
            client_writer.close()
        except Exception:
            pass


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen", type=int, default=8002)
    parser.add_argument("--upstream", default="127.0.0.1:8001")
    parser.add_argument(
        "--journal",
        default=os.path.expanduser("~/.mtplx/logs/oc-tap.jsonl"),
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.journal), exist_ok=True)
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, args), "127.0.0.1", args.listen
    )
    print(
        f"[oc-tap] listening on 127.0.0.1:{args.listen} -> {args.upstream}; "
        f"journal {args.journal}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
