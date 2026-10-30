"""User-facing MTPLX error types.

The CLI should only show tracebacks when the user explicitly asks for debug
output.  These exceptions carry enough structure for both human and JSON
responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


EXIT_GENERAL = 1
EXIT_INVALID_MTP = 2
EXIT_EXACTNESS = 3
EXIT_UNSUPPORTED_ARCHITECTURE = 4
EXIT_ENVIRONMENT = 5
EXIT_RESOURCE = 6
EXIT_INTEGRATION = 7
EXIT_THERMAL = 8


@dataclass
class MTPLXError(Exception):
    message: str
    code: int = EXIT_GENERAL
    kind: str = "mtplx_error"
    detail: str | None = None
    fix: str | None = None
    command: str | None = None

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.kind,
            "message": self.message,
            "exit_code": self.code,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.fix:
            payload["fix"] = self.fix
        if self.command:
            payload["command"] = self.command
        return payload


class MissingMLXError(MTPLXError):
    def __init__(self, detail: str | None = None) -> None:
        super().__init__(
            "MLX is not installed for this native arm64 Python environment.",
            code=EXIT_ENVIRONMENT,
            kind="missing_mlx",
            detail=detail,
            fix=(
                "Install MLX from a native arm64 Python on Apple Silicon. "
                "If platform.processor() reports i386, switch out of Rosetta."
            ),
            command="python3 -m pip install mlx",
        )


class MissingModelError(MTPLXError):
    def __init__(self, model: str, pull_command: str) -> None:
        super().__init__(
            f"Model {model} is not cached locally.",
            code=EXIT_RESOURCE,
            kind="missing_model",
            fix="Download the verified MTPLX model before running.",
            command=pull_command,
        )


class IntegrationError(MTPLXError):
    def __init__(self, message: str, *, fix: str | None = None, command: str | None = None) -> None:
        super().__init__(
            message,
            code=EXIT_INTEGRATION,
            kind="integration_error",
            fix=fix,
            command=command,
        )



# Allocation failures the daemon must survive as per-request errors. MLX's
# Metal allocator raises plain RuntimeErrors ("[metal::malloc] ... maximum
# allowed buffer size", "Insufficient Memory",
# kIOGPUCommandBufferCallbackErrorOutOfMemory); before 2.9.4 these reached
# clients as anonymous internal_error 500s with no cache shed, so the very
# next request hit the same wall (#348).
_ALLOCATION_FAILURE_MARKERS = (
    "insufficient memory",
    "out of memory",
    "failed to allocate",
    "kiogpucommandbuffercallbackerroroutofmemory",
    "metal::malloc",
    "maximum allowed buffer size",
)


def is_allocation_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is the memory refusing an allocation.

    The server answers such a request with a memory error instead of an
    internal one, and an optional allocation (a faster route's buffers) falls
    back to the path that needs none. Anything else, a cancellation or a
    programming error, is not one.
    """

    if isinstance(exc, MemoryError):
        return True
    if not isinstance(exc, (RuntimeError, OSError)):
        return False
    text = str(exc).lower()
    return any(marker in text for marker in _ALLOCATION_FAILURE_MARKERS)


def is_resource_count_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is Metal's limit on live buffer objects, not memory.

    MLX's allocator refuses a new buffer once 499,000 of them are alive
    (``[metal::malloc] Resource limit (499000) exceeded.``), whatever memory
    is free, and it has already released its cache of free buffers before
    it raises. Halving a byte budget or clearing that cache cannot answer it;
    only letting go of arrays can (#640: 1,124 identical failures over a
    7.1 GB bank under its byte budget, until a restart).
    """

    if not is_allocation_failure(exc):
        return False
    text = str(exc).lower()
    return "resource limit" in text and "exceeded" in text
