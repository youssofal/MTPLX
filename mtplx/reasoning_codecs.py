"""Backend-neutral reasoning text codecs.

Public surfaces use this module to split backend-native hidden reasoning from
visible assistant content without importing the HTTP server or MLX runtime.
Each backend owns its wire format; callers only ask for a codec by parser id.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from mtplx.chat_encoding import (
    GEMMA4_THINK_CLOSE,
    GEMMA4_THINK_OPEN,
    GEMMA4_THOUGHT_PREFIX,
    strip_gemma4_thinking_text,
)


QWEN_THINK_OPEN = "<think>"
QWEN_THINK_CLOSE = "</think>"
QWEN_STYLE_REASONING_PARSERS = {"qwen3", "step3p5"}
QWEN_STYLE_REASONING_TAG_NAMES = (
    "think",
    "thinks",
    "thinking",
    "thought",
    "thoughts",
    "reason",
    "reasoning",
)
_QWEN_STYLE_TAG_NAME_PATTERN = "|".join(
    re.escape(name) for name in QWEN_STYLE_REASONING_TAG_NAMES
)
QWEN_STYLE_REASONING_OPEN_RE = re.compile(
    rf"<\s*(?:{_QWEN_STYLE_TAG_NAME_PATTERN})\b[^>\n]*>",
    re.IGNORECASE,
)
QWEN_STYLE_REASONING_CLOSE_RE = re.compile(
    rf"</\s*(?:{_QWEN_STYLE_TAG_NAME_PATTERN})\s*>",
    re.IGNORECASE,
)
QWEN_STYLE_REASONING_CONTROL_RE = re.compile(
    rf"</?\s*(?:{_QWEN_STYLE_TAG_NAME_PATTERN})\b[^>\n]*>",
    re.IGNORECASE,
)
QWEN_STYLE_REASONING_BLOCK_RE = re.compile(
    rf"<\s*(?:{_QWEN_STYLE_TAG_NAME_PATTERN})\b[^>\n]*>"
    rf".*?</\s*(?:{_QWEN_STYLE_TAG_NAME_PATTERN})\s*>",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class ReasoningTextParts:
    reasoning: str
    content: str


def _strip_gemma4_channel_markup(text: str) -> str:
    return (
        str(text)
        .replace(GEMMA4_THINK_OPEN, "")
        .replace(GEMMA4_THINK_CLOSE, "")
        .replace(GEMMA4_THOUGHT_PREFIX, "")
        .strip()
    )


def split_gemma4_reasoning_text(
    text: str,
    *,
    thinking_enabled: bool,
) -> ReasoningTextParts:
    raw = str(text or "")
    if not thinking_enabled:
        return ReasoningTextParts(
            "",
            _strip_gemma4_channel_markup(strip_gemma4_thinking_text(raw)),
        )
    if GEMMA4_THINK_OPEN not in raw and GEMMA4_THINK_CLOSE not in raw:
        return ReasoningTextParts("", raw.strip())

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    position = 0
    while position < len(raw):
        open_index = raw.find(GEMMA4_THINK_OPEN, position)
        if open_index < 0:
            content_parts.append(raw[position:])
            break
        content_parts.append(raw[position:open_index])
        thought_start = open_index + len(GEMMA4_THINK_OPEN)
        close_index = raw.find(GEMMA4_THINK_CLOSE, thought_start)
        if close_index < 0:
            reasoning = raw[thought_start:]
            position = len(raw)
        else:
            reasoning = raw[thought_start:close_index]
            position = close_index + len(GEMMA4_THINK_CLOSE)
        if reasoning.startswith(GEMMA4_THOUGHT_PREFIX):
            reasoning = reasoning[len(GEMMA4_THOUGHT_PREFIX) :]
        if reasoning:
            reasoning_parts.append(reasoning)
    return ReasoningTextParts(
        "".join(reasoning_parts).strip(),
        _strip_gemma4_channel_markup("".join(content_parts)),
    )


def split_qwen_reasoning_text(
    text: str,
    *,
    thinking_enabled: bool,
) -> ReasoningTextParts:
    raw = str(text or "")
    if not thinking_enabled:
        return ReasoningTextParts("", strip_qwen_style_reasoning_from_content(raw))

    reasoning_parts: list[str] = []

    def _capture(match: re.Match[str]) -> str:
        reasoning_parts.append(match.group(1))
        return ""

    content = re.sub(
        rf"<\s*(?:{_QWEN_STYLE_TAG_NAME_PATTERN})\b[^>\n]*>"
        rf"(.*?)</\s*(?:{_QWEN_STYLE_TAG_NAME_PATTERN})\s*>",
        _capture,
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    first_close = QWEN_STYLE_REASONING_CLOSE_RE.search(content)
    if first_close and not QWEN_STYLE_REASONING_OPEN_RE.search(content):
        before = content[: first_close.start()]
        after = content[first_close.end() :]
        reasoning_parts.insert(0, before)
        content = after
    return ReasoningTextParts(
        "\n".join(part.strip() for part in reasoning_parts if part.strip()).strip(),
        strip_qwen_style_reasoning_control_markup(content).strip(),
    )


def strip_qwen_style_reasoning_control_markup(text: str) -> str:
    """Remove visible control tags used by Qwen/Step-style reasoning parsers."""

    return QWEN_STYLE_REASONING_CONTROL_RE.sub("", str(text or ""))


def strip_qwen_style_reasoning_from_content(text: str) -> str:
    """Return visible content when Qwen/Step reasoning is disabled.

    The Step 3.7 template always opens a ``<think>`` block for generation.
    Runtime integrations normally pre-close an empty thought block when
    reasoning is disabled, but models can still emit orphan close markers such
    as ``</think>`` or ``</thinks>``. Those are backend control markers, not
    assistant text. If a standalone close marker appears and has visible text
    after it, keep the text after the last close marker because that is the
    post-thought answer position in Qwen/Step templates.
    """

    cleaned = QWEN_STYLE_REASONING_BLOCK_RE.sub("", str(text or ""))
    close_matches = list(QWEN_STYLE_REASONING_CLOSE_RE.finditer(cleaned))
    if close_matches:
        after_last_close = cleaned[close_matches[-1].end() :]
        if after_last_close.strip():
            cleaned = after_last_close
    cleaned = strip_qwen_style_reasoning_control_markup(cleaned)
    return cleaned.strip()


def _split_qwen_prefilled_thinking_text(
    text: str,
    *,
    thinking_enabled: bool,
) -> ReasoningTextParts:
    """Split Qwen text when the chat template already opened `<think>`.

    Qwen can begin generated text inside a thinking block because the assistant
    generation prompt emitted the opening tag.  This path is intentionally
    separate from `split_qwen_reasoning_text`, which parses complete public
    response text after generation.
    """

    if not thinking_enabled:
        return ReasoningTextParts("", strip_qwen_style_reasoning_from_content(text))
    raw = str(text or "")
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    def append_reasoning(segment: str) -> None:
        if not segment:
            return
        if (
            reasoning_parts
            and not reasoning_parts[-1].endswith(("\n", " "))
            and not segment.startswith(("\n", " "))
        ):
            reasoning_parts.append("\n")
        reasoning_parts.append(segment)

    position = 0
    inside_thinking = True
    while position < len(raw):
        if inside_thinking:
            close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(raw, position)
            if close_match is None:
                segment = (
                    raw[position:]
                )
                segment = strip_qwen_style_reasoning_control_markup(segment)
                append_reasoning(segment)
                break
            segment = raw[position : close_match.start()]
            segment = strip_qwen_style_reasoning_control_markup(segment)
            append_reasoning(segment)
            position = close_match.end()
            inside_thinking = False
            continue

        open_match = QWEN_STYLE_REASONING_OPEN_RE.search(raw, position)
        if open_match is None:
            segment = strip_qwen_style_reasoning_control_markup(raw[position:])
            if segment:
                content_parts.append(segment)
            break
        segment = strip_qwen_style_reasoning_control_markup(raw[position : open_match.start()])
        if segment:
            content_parts.append(segment)
        position = open_match.end()
        inside_thinking = True
    return ReasoningTextParts(
        "".join(reasoning_parts).strip(),
        "".join(content_parts).strip(),
    )


def _format_qwen_reasoning_history(parts: ReasoningTextParts) -> str:
    pieces: list[str] = []
    if parts.reasoning:
        pieces.append(f"{QWEN_THINK_OPEN}\n{parts.reasoning}\n{QWEN_THINK_CLOSE}")
    if parts.content:
        pieces.append(parts.content)
    return "\n\n".join(pieces)


def normalize_qwen_thinking_tags(text: str, *, thinking_enabled: bool) -> str:
    """Return Qwen-compatible history text for a Qwen generation."""

    parts = _split_qwen_prefilled_thinking_text(
        text,
        thinking_enabled=thinking_enabled,
    )
    if not thinking_enabled:
        return parts.content
    return _format_qwen_reasoning_history(parts)


def normalize_reasoning_tags(
    text: str,
    *,
    parser: str,
    thinking_enabled: bool,
) -> str:
    """Return Qwen-style history tags from any backend-native reasoning codec."""

    parser_id = str(parser or "none").lower()
    if parser_id == "gemma4":
        parts = split_gemma4_reasoning_text(text, thinking_enabled=thinking_enabled)
        if not thinking_enabled:
            return parts.content
        return _format_qwen_reasoning_history(parts)
    if parser_id in QWEN_STYLE_REASONING_PARSERS:
        return normalize_qwen_thinking_tags(
            text,
            thinking_enabled=thinking_enabled,
        )
    return str(text or "").strip()


def split_reasoning_text(
    text: str,
    *,
    parser: str,
    thinking_enabled: bool,
) -> ReasoningTextParts:
    parser_id = str(parser or "none").lower()
    if parser_id == "gemma4":
        return split_gemma4_reasoning_text(text, thinking_enabled=thinking_enabled)
    if parser_id in QWEN_STYLE_REASONING_PARSERS:
        return split_qwen_reasoning_text(text, thinking_enabled=thinking_enabled)
    return ReasoningTextParts("", str(text or "").strip())


class ReasoningContentStreamSplitter:
    """Incrementally split backend-native reasoning from visible content."""

    def __init__(self, *, thinking_enabled: bool) -> None:
        self._thinking_enabled = thinking_enabled
        self._reentry_count = 0

    @property
    def reentry_count(self) -> int:
        return self._reentry_count

    def start(self) -> list[tuple[str, str]]:
        return []

    def feed(self, text: str) -> list[tuple[str, str]]:
        raise NotImplementedError

    def finish(self) -> list[tuple[str, str]]:
        raise NotImplementedError


class QwenThinkingContentStreamSplitter(ReasoningContentStreamSplitter):
    def __init__(self, *, thinking_enabled: bool) -> None:
        super().__init__(thinking_enabled=thinking_enabled)
        self._inside_thinking = thinking_enabled
        self._disabled_inside_reasoning = False
        self._disabled_visible_started = False
        self._pending = ""

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        if not self._thinking_enabled:
            self._pending += text
            return self._drain_disabled(final=False)
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        chunks = (
            self._drain(final=True)
            if self._thinking_enabled
            else self._drain_disabled(final=True)
        )
        self._inside_thinking = False
        return chunks

    def _append_chunk(
        self,
        chunks: list[tuple[str, str]],
        field: str,
        text: str,
    ) -> None:
        cleaned = strip_qwen_style_reasoning_control_markup(text)
        if cleaned:
            chunks.append((field, cleaned))

    def _drain_disabled(self, *, final: bool) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        keep = max(
            max(len(name) for name in QWEN_STYLE_REASONING_TAG_NAMES) + len("</>"),
            16,
        )
        initial_hold = 384
        while self._pending:
            if self._disabled_inside_reasoning:
                close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(self._pending)
                if close_match is None:
                    if final:
                        self._pending = ""
                    else:
                        self._pending = self._pending[-keep:]
                    break
                self._pending = self._pending[close_match.end() :].lstrip()
                self._disabled_inside_reasoning = False
                self._disabled_visible_started = True
                continue
            block_cleaned = QWEN_STYLE_REASONING_BLOCK_RE.sub("", self._pending)
            if block_cleaned != self._pending:
                self._pending = block_cleaned
                if not self._pending:
                    break
                continue
            close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(self._pending)
            if close_match is not None:
                self._pending = self._pending[close_match.end() :].lstrip()
                self._disabled_visible_started = True
                continue
            open_match = QWEN_STYLE_REASONING_OPEN_RE.search(self._pending)
            if open_match is not None:
                before = self._pending[: open_match.start()]
                if before:
                    self._append_chunk(chunks, "content", before)
                    self._disabled_visible_started = True
                self._reentry_count += 1
                self._pending = self._pending[open_match.end() :]
                self._disabled_inside_reasoning = True
                continue
            if (
                not final
                and not self._disabled_visible_started
                and len(self._pending) < initial_hold
            ):
                break
            emit_len = len(self._pending) if final else max(0, len(self._pending) - keep)
            if emit_len <= 0:
                break
            self._append_chunk(chunks, "content", self._pending[:emit_len])
            self._disabled_visible_started = True
            self._pending = self._pending[emit_len:]
            break
        return chunks

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        # Hold back enough trailing bytes to cover the longest reasoning tag
        # (e.g. "</reasoning>"); using only the "<think>"/"</think>" lengths
        # let a longer alias tag split across chunks leak into visible content.
        # Mirrors _drain_disabled's window.
        keep = max(
            max(len(name) for name in QWEN_STYLE_REASONING_TAG_NAMES) + len("</>"),
            16,
        )
        while self._pending:
            if self._inside_thinking:
                close_match = QWEN_STYLE_REASONING_CLOSE_RE.search(self._pending)
                if close_match is None:
                    emit_len = (
                        len(self._pending)
                        if final
                        else max(0, len(self._pending) - keep)
                    )
                    if emit_len <= 0:
                        break
                    self._append_chunk(
                        chunks,
                        "reasoning_content",
                        self._pending[:emit_len],
                    )
                    self._pending = self._pending[emit_len:]
                    break
                self._append_chunk(
                    chunks,
                    "reasoning_content",
                    self._pending[: close_match.start()],
                )
                self._pending = self._pending[close_match.end() :].lstrip()
                self._inside_thinking = False
                continue

            open_match = QWEN_STYLE_REASONING_OPEN_RE.search(self._pending)
            if open_match is None:
                emit_len = (
                    len(self._pending) if final else max(0, len(self._pending) - keep)
                )
                if emit_len <= 0:
                    break
                self._append_chunk(chunks, "content", self._pending[:emit_len])
                self._pending = self._pending[emit_len:]
                break
            self._append_chunk(chunks, "content", self._pending[: open_match.start()])
            self._pending = self._pending[open_match.end() :]
            self._inside_thinking = True
            self._reentry_count += 1
        return chunks


class Gemma4ThinkingContentStreamSplitter(ReasoningContentStreamSplitter):
    def __init__(self, *, thinking_enabled: bool) -> None:
        super().__init__(thinking_enabled=thinking_enabled)
        self._inside_thinking = False
        self._pending = ""
        self._reasoning_prefix_buffer = ""
        self._reasoning_prefix_stripped = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        self._pending += text
        return self._drain(final=False)

    def finish(self) -> list[tuple[str, str]]:
        chunks = self._drain(final=True)
        self._inside_thinking = False
        return chunks

    def _append_content(
        self,
        chunks: list[tuple[str, str]],
        text: str,
    ) -> None:
        cleaned = self._strip_channel_markup(text)
        if cleaned:
            chunks.append(("content", cleaned))

    def _append_reasoning(
        self,
        chunks: list[tuple[str, str]],
        text: str,
    ) -> None:
        if not text:
            return
        if not self._thinking_enabled:
            return
        if not self._reasoning_prefix_stripped:
            self._reasoning_prefix_buffer += text
            if GEMMA4_THOUGHT_PREFIX.startswith(self._reasoning_prefix_buffer):
                return
            if self._reasoning_prefix_buffer.startswith(GEMMA4_THOUGHT_PREFIX):
                text = self._reasoning_prefix_buffer[len(GEMMA4_THOUGHT_PREFIX) :]
            else:
                text = self._reasoning_prefix_buffer
            self._reasoning_prefix_buffer = ""
            self._reasoning_prefix_stripped = True
        if text:
            chunks.append(("reasoning_content", text))

    def _drain(self, *, final: bool) -> list[tuple[str, str]]:
        chunks: list[tuple[str, str]] = []
        keep = max(len(GEMMA4_THINK_OPEN), len(GEMMA4_THINK_CLOSE)) - 1
        while self._pending:
            if self._inside_thinking:
                close_index = self._pending.find(GEMMA4_THINK_CLOSE)
                if close_index < 0:
                    emit_len = (
                        len(self._pending)
                        if final
                        else max(0, len(self._pending) - keep)
                    )
                    if emit_len <= 0:
                        break
                    self._append_reasoning(chunks, self._pending[:emit_len])
                    self._pending = self._pending[emit_len:]
                    break
                self._append_reasoning(chunks, self._pending[:close_index])
                self._pending = self._pending[
                    close_index + len(GEMMA4_THINK_CLOSE) :
                ].lstrip()
                self._inside_thinking = False
                continue

            open_index = self._pending.find(GEMMA4_THINK_OPEN)
            if open_index < 0:
                emit_len = (
                    len(self._pending) if final else max(0, len(self._pending) - keep)
                )
                if emit_len <= 0:
                    break
                self._append_content(chunks, self._pending[:emit_len])
                self._pending = self._pending[emit_len:]
                break
            self._append_content(chunks, self._pending[:open_index])
            self._pending = self._pending[open_index + len(GEMMA4_THINK_OPEN) :]
            self._inside_thinking = True
            self._reentry_count += 1
        return chunks

    @staticmethod
    def _strip_channel_markup(text: str) -> str:
        return (
            text.replace(GEMMA4_THINK_OPEN + GEMMA4_THOUGHT_PREFIX, "")
            .replace(GEMMA4_THINK_OPEN, "")
            .replace(GEMMA4_THINK_CLOSE, "")
        )


class ThinkingContentStreamNormalizer(QwenThinkingContentStreamSplitter):
    """Compatibility wrapper returning only text chunks."""

    def start(self) -> list[str]:
        return [text for _, text in super().start()]

    def feed(self, text: str) -> list[str]:
        return [chunk for _, chunk in super().feed(text)]

    def finish(self) -> list[str]:
        return [chunk for _, chunk in super().finish()]


def stream_splitter_for_parser(
    parser: str,
    *,
    thinking_enabled: bool,
) -> ReasoningContentStreamSplitter:
    parser_id = str(parser or "none").lower()
    if parser_id == "gemma4":
        return Gemma4ThinkingContentStreamSplitter(
            thinking_enabled=thinking_enabled,
        )
    if parser_id in QWEN_STYLE_REASONING_PARSERS:
        return QwenThinkingContentStreamSplitter(thinking_enabled=thinking_enabled)
    return QwenThinkingContentStreamSplitter(thinking_enabled=False)
