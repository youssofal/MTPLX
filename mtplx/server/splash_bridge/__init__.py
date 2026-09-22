# SPDX-License-Identifier: Apache-2.0
"""Serve MTPLX's API and dashboard contract over Splash's inference engine.

`mtplx serve --engine splash` lands here. The bridge supervises one Splash
process, proxies inference to it, and translates its telemetry into the
payloads MTPLX clients already decode, so switching engines changes the
kernels underneath and nothing above.

What the two engines actually differ on is worth stating plainly, because the
bridge reports it rather than papering over it: Splash is specialized per
model with precompiled Metal kernels and a trained DFlash 2 draft, and its KV
cache is fixed at 8-bit. The MLX engine runs many architectures and supports
4-bit, 8-bit and unquantized paged KV. The bridge advertises Splash's KV
policy as unsupported-with-a-reason so clients disable the control instead of
offering a choice the engine cannot honor.
"""

from __future__ import annotations

import signal
import sys
from typing import Any

from .app import create_app
from .install import SplashInstall, SplashUnavailable
from .supervisor import SplashEngine, free_port
from .telemetry import BridgeTelemetry

__all__ = [
    "BridgeTelemetry",
    "SplashEngine",
    "SplashInstall",
    "SplashUnavailable",
    "create_app",
    "free_port",
    "serve",
]

DEFAULT_MODEL = "incoai/Qwen3.8-27B-Splash"


def _line(text: str = "") -> None:
    print(text, flush=True)


def serve(
    *,
    model: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_memory: str | None = None,
    max_context: str | None = None,
    api_key: str | None = None,
    splash_port: int | None = None,
    splash_prefix: str | None = None,
    download: bool = True,
) -> int:
    """Run the bridge in the foreground. Returns a process exit code."""
    model_id = model or DEFAULT_MODEL
    try:
        install = SplashInstall.discover(splash_prefix)
        install.validate()
    except SplashUnavailable as error:
        _line(f"error: {error}")
        return 2

    known = install.official_models()
    if model_id not in known and not install.is_installed(model_id):
        _line(f"error: {model_id} is not a Splash package.")
        _line("Splash packages on this install:")
        for candidate in known:
            _line(f"  {candidate}")
        return 2

    if not install.is_installed(model_id) and not download:
        _line(f"error: {model_id} is not downloaded; re-run without --no-download")
        return 2

    engine = SplashEngine(
        install,
        model_id,
        port=splash_port or free_port(),
        max_memory=max_memory,
        max_context=max_context,
        api_key=api_key,
        on_log=lambda line: _line(f"  splash | {line}"),
    )
    telemetry = BridgeTelemetry(model_id)

    _line(f"MTPLX {port} -> Splash engine (private port {engine.port})")
    _line(f"Model: {model_id}")
    if not install.is_installed(model_id):
        _line("The package is not downloaded yet; Splash will fetch and verify it.")

    try:
        engine.start()
    except SplashUnavailable as error:
        _line(f"error: {error}")
        engine.stop()
        return 1
    except KeyboardInterrupt:
        engine.stop()
        return 130

    app = create_app(
        install=install,
        engine=engine,
        telemetry=telemetry,
        api_key=api_key,
    )

    status = engine.status()
    context = telemetry.context_window(status)
    _line("")
    _line("MTPLX is ready.")
    # install.version() already reads "Splash 1.0"; do not prefix it again.
    _line(f"Engine: {install.version()} (DFlash 2 speculative decoding)")
    _line("KV cache: 8-bit int8, fixed by the engine's Metal kernels")
    if context:
        _line(f"Context: {context:,} tokens")
    _line(f"OpenAI API Base URL: http://{host}:{port}/v1")
    _line(f"Anthropic Messages:  http://{host}:{port}/v1/messages")
    _line(f"Health check: http://{host}:{port}/health")
    _line("Keep this terminal open. Press Ctrl-C to stop.")
    _line("")

    def _stop(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)

    import uvicorn

    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        _line("Stopping the Splash engine...")
        engine.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Standalone entry point, for running the bridge without the MTPLX CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="mtplx-splash", description="Serve the MTPLX contract over Splash"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-memory", default=None)
    parser.add_argument("--max-context", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--splash-port", type=int, default=None)
    parser.add_argument("--splash-prefix", default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    return serve(
        model=args.model,
        host=args.host,
        port=args.port,
        max_memory=args.max_memory,
        max_context=args.max_context,
        api_key=args.api_key,
        splash_port=args.splash_port,
        splash_prefix=args.splash_prefix,
        download=not args.no_download,
    )


if __name__ == "__main__":
    sys.exit(main())
