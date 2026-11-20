"""The instrument's only way into a model: an OpenAI-compatible server.

Measuring through the server measures what users run (the pack, its kernels,
the server's settings, the chat template and the reasoning and tool parsers),
and lets the same suite score other engines that speak the same API.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

# A busy or restarting server; every other status is a real answer.
_TRANSIENT = frozenset({408, 425, 429, 500, 502, 503, 504})
# What an MTPLX server's /health says about the thing being measured; other
# servers simply lack these keys.
_IDENTITY_KEYS = (
    "model",
    "model_path",
    "vision",
    "generation_mode",
    "mtp_enabled",
    "depth",
    "context_window",
    "paged_kv_quantization",
    "reasoning_parser",
    "enable_thinking",
    "preserve_thinking_effective",
    "draft_lm_head",
    "draft_head_identity",
    "tokenizer_template_hash",
)


class ServerError(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


class Server:
    def __init__(self, base_url: str, *, model: str | None = None, timeout_s: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[: -len("/v1")]
        self.model = model
        self.timeout_s = timeout_s

    def _request(self, path: str, body: dict[str, Any] | None = None, *, attempts: int = 3) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode()
        for attempt in range(attempts):
            request = urllib.request.Request(
                self.base_url + path,
                data=data,
                headers={"Content-Type": "application/json"},
                method="GET" if body is None else "POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:500]
                if error.code not in _TRANSIENT or attempt == attempts - 1:
                    raise ServerError(f"{path}: HTTP {error.code}: {detail}", error.code) from error
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                if attempt == attempts - 1:
                    raise ServerError(f"{path}: {error}") from error
            time.sleep(5.0 * (attempt + 1))
        raise AssertionError("unreachable")

    def identity(self) -> dict[str, Any]:
        """What is being measured: the served model and the server's own report."""

        models = self._request("/v1/models").get("data") or []
        identity: dict[str, Any] = {"base_url": self.base_url, "models": [m.get("id") for m in models]}
        try:
            health = self._request("/health", attempts=1)
        except ServerError:
            health = {}
        for key in _IDENTITY_KEYS:
            if key in health:
                identity[key] = health[key]
        sampler = (health.get("profile") or {}).get("sampler")
        if sampler:
            identity["default_sampler"] = sampler
        if self.model is None and models:
            self.model = models[0].get("id")
        identity["model_requested"] = self.model
        return identity

    def routes(self) -> dict[str, Any]:
        """Which fast lanes the server's requests took and every demotion off
        one, as the server counts them (empty for servers without the report)."""

        try:
            return dict(self._request("/health", attempts=1).get("degradation") or {})
        except ServerError:
            return {}

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._request("/v1/chat/completions", {"model": self.model, **body, "stream": False})

    def score(self, token_ids: list[int], *, top_k: int) -> dict[str, Any]:
        """Teacher-forced next-token distributions over ``token_ids``."""

        return self._request(
            "/v1/completions",
            {"model": self.model, "prompt": token_ids, "echo": True, "logprobs": top_k, "max_tokens": 0},
        )
