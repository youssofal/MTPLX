"""Lightweight attention-phase telemetry context."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

VALID_ATTENTION_PHASES = {
    "prefill",
    "decode_verify",
    "ar_decode",
    "postcommit",
    "unknown",
}
VALID_MODEL_FORWARD_KINDS = {
    "target_verify",
    "repair",
    "other",
}

_ATTENTION_PHASE: ContextVar[str] = ContextVar(
    "mtplx_attention_phase",
    default="unknown",
)
_MODEL_FORWARD_KIND: ContextVar[str] = ContextVar(
    "mtplx_model_forward_kind",
    default="other",
)
_REQUEST_PROMPT_TOKENS: ContextVar[int] = ContextVar(
    "mtplx_request_prompt_tokens",
    default=0,
)


def normalize_attention_phase(phase: str | None) -> str:
    value = (phase or "unknown").strip().lower()
    return value if value in VALID_ATTENTION_PHASES else "unknown"


def current_attention_phase() -> str:
    return normalize_attention_phase(_ATTENTION_PHASE.get())


def normalize_model_forward_kind(kind: str | None) -> str:
    value = (kind or "other").strip().lower()
    return value if value in VALID_MODEL_FORWARD_KINDS else "other"


def current_model_forward_kind() -> str:
    return normalize_model_forward_kind(_MODEL_FORWARD_KIND.get())


def current_request_prompt_tokens() -> int:
    return max(0, int(_REQUEST_PROMPT_TOKENS.get()))


@contextmanager
def attention_phase(phase: str | None) -> Iterator[None]:
    token = _ATTENTION_PHASE.set(normalize_attention_phase(phase))
    try:
        yield
    finally:
        _ATTENTION_PHASE.reset(token)


@contextmanager
def model_forward_kind(kind: str | None) -> Iterator[None]:
    """Identify whether one decode-verify-phase target call verifies or repairs."""

    token = _MODEL_FORWARD_KIND.set(normalize_model_forward_kind(kind))
    try:
        yield
    finally:
        _MODEL_FORWARD_KIND.reset(token)


@contextmanager
def request_prompt_tokens(count: int) -> Iterator[None]:
    token = _REQUEST_PROMPT_TOKENS.set(max(0, int(count)))
    try:
        yield
    finally:
        _REQUEST_PROMPT_TOKENS.reset(token)
