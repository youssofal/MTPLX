#!/usr/bin/env python3
"""Production entrypoint for the optimized DeepSeek-V4-Flash-0731 service."""

from __future__ import annotations

import sys
from pathlib import Path

from candidate_entry import CandidateConstructionError, install_candidate_surface


MODEL = Path("/Users/davidtai/models/DeepSeek-V4-Flash-0731-2.4bit-mixed")
MODEL_ID = "mtplx-deepseek-v4-flash-0731-2.4bit-k3"


def serve_argv() -> list[str]:
    return [
        "serve",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--model",
        str(MODEL),
        "--model-id",
        MODEL_ID,
        "--backend-id",
        "deepseek_mtp",
        "--context-window",
        "262144",
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--top-k",
        "0",
        "--generation-mode",
        "mtp",
        "--load-mtp",
        "--depth",
        "3",
        "--deepseek-v4-0731-optimized",
        "--verify-strategy",
        "batched",
        "--verify-core",
        "stock",
        "--warmup-tokens",
        "0",
        "--max-active-requests",
        "1",
        "--session-cache-mode",
        "off",
        "--ssd-session-cache",
        "off",
        "--prefill-chunk-tokens",
        "512",
        "--mlx-cache-limit",
        "536870912",
        "--reasoning",
        "on",
        "--reasoning-effort",
        "low",
        "--reasoning-parser",
        "qwen3",
        "--tool-prompt-mode",
        "native",
        "--chat-template-profile",
        "tokenizer",
        "--no-stats-footer",
    ]


def main() -> int:
    if sys.argv[1:]:
        raise CandidateConstructionError("production entrypoint accepts no arguments")
    from mtplx.server import openai as server

    install_candidate_surface(server)
    server.main(serve_argv()[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
