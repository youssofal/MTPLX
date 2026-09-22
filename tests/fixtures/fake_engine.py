"""A tiny stdlib HTTP server standing in for a real `mtplx serve` engine.

Used only by ``tests/test_supervisor_process.py`` to exercise
``mtplx.supervisor.process.EngineProcess`` against a real subprocess without
loading an actual model. Speaks just enough of the daemon's `/health`
contract (``{"ok": true, ...}``) for the supervisor's liveness probe.

Flags:
  --port PORT            required, TCP port to bind
  --ready-after-s FLOAT   /health answers 503 "not ready" until this many
                          seconds after start (default 0: ready immediately)
  --hang-health           /health never responds (simulates a wedged engine)
  --exit-after-s FLOAT    process exits cleanly this many seconds after start
  --oom                   print a Metal OOM line to stderr and exit(1)
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

START_TIME = time.monotonic()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--ready-after-s", type=float, default=0.0)
    parser.add_argument("--hang-health", action="store_true")
    parser.add_argument("--exit-after-s", type=float, default=None)
    parser.add_argument("--oom", action="store_true")
    return parser.parse_args(argv)


def _make_handler(args: argparse.Namespace) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a: object) -> None:  # quiet
            pass

        def do_GET(self) -> None:  # noqa: N802 - stdlib method name
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            if args.hang_health:
                # Never respond; the client's timeout is what ends this.
                while True:
                    time.sleep(1.0)
            elapsed = time.monotonic() - START_TIME
            if elapsed < args.ready_after_s:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"ok": False}).encode("utf-8"))
                return
            payload = json.dumps({"ok": True, "model": "fake"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.oom:
        print(
            "RuntimeError: cannot load inside the available Metal memory budget",
            file=sys.stderr,
            flush=True,
        )
        return 1

    server = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(args))

    if args.exit_after_s is not None:
        def _stop_after() -> None:
            time.sleep(args.exit_after_s)
            server.shutdown()

        threading.Thread(target=_stop_after, daemon=True).start()

    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
