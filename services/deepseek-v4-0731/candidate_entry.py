#!/usr/bin/env python3
"""Construction-only entrypoint for the isolated V4-Flash-0731 service.

This module verifies and self-tests the official encoder before replacing the
MTPLX request-path call sites that own prompt encoding and DSML completion
parsing. There is no tokenizer-template, stock-prompt, or stock completion
parser fallback after install.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parent
ENCODING_ROOT = ROOT / "encoding"
MANIFEST = ENCODING_ROOT / "SHA256SUMS"
SOURCE_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
ENCODER_NAME = "deepseek-v4-flash-0731-official"
REQUIRED_ASSETS = (
    "encoding_dsv4.py",
    "tests/test_input_1.json",
    "tests/test_input_2.json",
    "tests/test_input_3.json",
    "tests/test_input_4.json",
    "tests/test_output_1.txt",
    "tests/test_output_2.txt",
    "tests/test_output_3.txt",
    "tests/test_output_4.txt",
)


class CandidateConstructionError(RuntimeError):
    """The isolated service cannot install its reviewed request surface."""


def _manifest_entries() -> dict[str, str]:
    if not MANIFEST.is_file() or MANIFEST.is_symlink():
        raise CandidateConstructionError("official encoding manifest is missing or unsafe")
    entries: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_./-]+)", line)
        if match is None or match.group(2) in entries:
            raise CandidateConstructionError("official encoding manifest is malformed")
        entries[match.group(2)] = match.group(1)
    if tuple(entries) != REQUIRED_ASSETS:
        raise CandidateConstructionError("official encoding manifest asset set changed")
    return entries


def verify_official_assets() -> dict[str, str]:
    """Verify the exact official source and vector set once at construction."""
    entries = _manifest_entries()
    root = ENCODING_ROOT.resolve()
    for relative, expected in entries.items():
        path = ENCODING_ROOT / relative
        if not path.is_file() or path.is_symlink() or path.resolve().parent != (root / relative).parent:
            raise CandidateConstructionError("official encoding asset is missing or unsafe")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise CandidateConstructionError("official encoding asset digest mismatch")
    return entries


def _load_official_encoder() -> ModuleType:
    verify_official_assets()
    path = ENCODING_ROOT / "encoding_dsv4.py"
    module = ModuleType("mtplx_dsv4_0731_official_encoding")
    module.__file__ = str(path)
    # Compile the already-verified source bytes directly. This cannot select an
    # ignored or stale __pycache__ artifact in place of the reviewed encoder.
    code = compile(path.read_bytes(), str(path), "exec")
    exec(code, module.__dict__)
    return module


def _self_test(encoding: ModuleType) -> None:
    """Run all four official byte vectors before installing the request path."""
    for case in range(1, 5):
        payload = json.loads((ENCODING_ROOT / f"tests/test_input_{case}.json").read_text(encoding="utf-8"))
        if case == 1:
            messages = payload["messages"]
            messages[0]["tools"] = payload["tools"]
        else:
            messages = payload
        mode = "chat" if case == 4 else "thinking"
        expected = (ENCODING_ROOT / f"tests/test_output_{case}.txt").read_text(encoding="utf-8")
        if encoding.encode_messages(messages, thinking_mode=mode) != expected:
            raise CandidateConstructionError(f"official encoding vector {case} failed")


def _message_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    dump = getattr(message, "model_dump", None)
    if callable(dump):
        value = dump(exclude_none=True)
        if isinstance(value, dict):
            return value
    value: dict[str, Any] = {}
    for field in ("role", "content", "name", "tool_call_id", "tool_calls", "reasoning_content"):
        if hasattr(message, field):
            item = getattr(message, field)
            if item is not None:
                value[field] = item
    if "role" not in value:
        raise CandidateConstructionError("request message has no role")
    return value


def _install_encoder(server: ModuleType, encoding: ModuleType):
    encode_text = server._encode_rendered_chat_text

    def encode_messages(
        tokenizer: Any,
        messages: list[Any],
        *,
        enable_thinking: bool,
        reasoning_effort: str | None = None,
        strip_assistant_reasoning_history: bool = False,
        scoped_reasoning_history: bool = False,
        add_generation_prompt: bool = True,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        tool_prompt_mode: str = "native",
        template_observability: dict[str, Any] | None = None,
    ) -> list[int]:
        del strip_assistant_reasoning_history, scoped_reasoning_history, tool_prompt_mode
        if tool_choice not in (None, "auto"):
            raise CandidateConstructionError("V4-0731 candidate does not support forced tool_choice")
        effort = reasoning_effort or "low"
        if effort not in encoding.REASONING_EFFORT_PROMPTS:
            raise CandidateConstructionError("reasoning_effort must be one of: low, high, max")
        prepared = [_message_dict(message) for message in messages]
        if not prepared:
            prepared = [{"role": "user", "content": ""}]
        if tools:
            if prepared[0].get("role") != "system":
                prepared.insert(0, {"role": "system", "content": "", "tools": tools})
            else:
                prepared[0] = {**prepared[0], "tools": tools}
        mode = "thinking" if enable_thinking else "chat"
        rendered = encoding.encode_messages(
            prepared,
            thinking_mode=mode,
            reasoning_effort=effort,
        )
        if not add_generation_prompt:
            suffix = encoding.ASSISTANT_SP_TOKEN + (
                encoding.thinking_start_token if enable_thinking else encoding.thinking_end_token
            )
            if rendered.endswith(suffix):
                rendered = rendered[: -len(suffix)]
        if template_observability is not None:
            template_observability.update(
                {
                    "backend_chat_encoding": ENCODER_NAME,
                    "encoding_source_revision": SOURCE_REVISION,
                }
            )
        return encode_text(tokenizer, rendered)

    return encode_messages


def _install_completion_parser(server: ModuleType, encoding: ModuleType):
    dsml_marker = f"<{encoding.dsml_token}{encoding.tool_calls_block_name}>"

    def parse_generated_tool_calls_or_content(
        text: str,
        *,
        tools: list[dict[str, Any]],
        tokenizer: Any | None = None,
        state: Any | None = None,
        response_id: str | None = None,
        stream: bool = False,
    ):
        del tools, tokenizer, state, response_id, stream
        completion = text if text.endswith(encoding.eos_token) else text + encoding.eos_token
        mode = "thinking" if encoding.thinking_end_token in completion.split(dsml_marker, 1)[0] else "chat"
        try:
            parsed = encoding.parse_message_from_completion_text(completion, thinking_mode=mode)
        except (AssertionError, ValueError) as error:
            raise CandidateConstructionError("malformed V4-0731 DSML completion") from error
        calls = parsed.get("tool_calls") or None
        return calls, None

    return parse_generated_tool_calls_or_content


def _install_actual_tool_extractor(server: ModuleType, encoding: ModuleType) -> None:
    """Install official DSML completion parsing at the live response call site."""
    from mtplx.server.omlx_bridge import ToolCallExtraction

    def extract(
        thinking_content: str,
        regular_content: str,
        tokenizer: Any | None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ToolCallExtraction:
        del tokenizer, tools
        mode = "thinking" if thinking_content else "chat"
        completion = (
            thinking_content + encoding.thinking_end_token + regular_content
            if mode == "thinking"
            else regular_content
        )
        if not completion.endswith(encoding.eos_token):
            completion += encoding.eos_token
        try:
            parsed = encoding.parse_message_from_completion_text(completion, thinking_mode=mode)
        except (AssertionError, ValueError) as error:
            raise CandidateConstructionError("malformed V4-0731 DSML completion") from error
        calls = parsed.get("tool_calls") or None
        if calls:
            stable_calls = []
            for index, call in enumerate(calls):
                canonical = json.dumps(call, sort_keys=True, separators=(",", ":"))
                call_id = "call_" + hashlib.sha256(
                    f"{index}:{canonical}".encode("utf-8")
                ).hexdigest()[:24]
                stable_calls.append({**call, "id": str(call.get("id") or call_id)})
            calls = stable_calls
        return ToolCallExtraction(
            cleaned_text=str(parsed.get("content") or ""),
            tool_calls=calls,
            cleaned_thinking=str(parsed.get("reasoning_content") or ""),
            parser_source="deepseek_v4_0731_official",
            status="parsed" if calls else "no_tool",
            raw_tool_markup_suppressed=bool(calls),
        )

    server.omlx_extract_tool_calls_with_thinking = extract


def _install_no_tools_nonstream_sanitizer(server: ModuleType, encoding: ModuleType) -> None:
    """Route the candidate's no-tools JSON response through official parsing."""
    dsml_marker = f"<{encoding.dsml_token}"

    def strip_orphan_tool_markup(text: str) -> tuple[str, int]:
        try:
            extraction = server.omlx_extract_tool_calls_with_thinking(
                "", text, None, []
            )
        except CandidateConstructionError:
            marker = text.find(dsml_marker)
            return (text[:marker].rstrip(), 1) if marker >= 0 else ("", 0)
        cleaned = extraction.cleaned_text.strip()
        if dsml_marker in cleaned:
            raise CandidateConstructionError("official nonstream parser retained DSML markup")
        return cleaned, int(dsml_marker in text)

    server._strip_orphan_tool_markup = strip_orphan_tool_markup


def _install_stream_translator(server: ModuleType, encoding: ModuleType) -> None:
    """Keep scanning after visible prose while holding split DSML prefixes."""

    dsml_marker = f"<{encoding.dsml_token}"
    dsml_envelope_start = "\n\n" + dsml_marker

    class DSV40731StreamTranslator:
        def __init__(self, *, tools, argument_chunk_chars, tokenizer=None, **kwargs) -> None:
            del kwargs
            self._tools = tools
            self._argument_chunk_chars = max(1, int(argument_chunk_chars))
            self._tokenizer = tokenizer
            self._pending = ""
            self._all_content = ""
            self._emitted_content = ""
            self._inside_dsml = False
            self.tool_calls = None
            self.fallback_reason = None
            self.tool_parser_dialect = "deepseek_v4_0731_official"
            self._suppressed = False
            self._emitted_tool_deltas = False

        @property
        def has_tool_calls(self):
            return bool(self.tool_calls)

        @property
        def has_emitted_tool_deltas(self):
            return self._emitted_tool_deltas

        @property
        def suppressed_tool_markup(self):
            return self._suppressed

        @property
        def buffering_tool_call(self):
            return self._inside_dsml

        @property
        def tool_argument_in_progress(self):
            return self._inside_dsml

        @property
        def ready_to_finish_tool_turn(self):
            return False

        @property
        def invalid_trailing_after_tool_call(self):
            return False

        def feed(self, field: str, text: str):
            if not text:
                return []
            if field != "content":
                return [{field: text}]
            self._all_content += text
            if self._inside_dsml:
                return []
            self._pending += text
            marker = self._pending.find(dsml_marker)
            if marker >= 0:
                visible = self._pending[:marker]
                if visible.endswith("\n\n"):
                    visible = visible[:-2]
                self._pending = ""
                self._inside_dsml = True
                self._suppressed = True
                self._emitted_content += visible
                return [{"content": visible}] if visible else []
            hold = 0
            prefix_limit = max(len(dsml_marker), len(dsml_envelope_start)) - 1
            for size in range(min(len(self._pending), prefix_limit), 0, -1):
                suffix = self._pending[-size:]
                if dsml_marker.startswith(suffix) or dsml_envelope_start.startswith(suffix):
                    hold = size
                    break
            visible = self._pending[:-hold] if hold else self._pending
            self._pending = self._pending[-hold:] if hold else ""
            self._emitted_content += visible
            return [{"content": visible}] if visible else []

        def finish(self, *, defer_content_resolution: bool = False):
            del defer_content_resolution
            extraction = server.omlx_extract_tool_calls_with_thinking(
                "", self._all_content, self._tokenizer, self._tools
            )
            self.tool_calls = extraction.tool_calls
            self._pending = ""
            self._all_content = ""
            deltas = []
            if not extraction.cleaned_text.startswith(self._emitted_content):
                raise CandidateConstructionError("official stream parse changed emitted content")
            remaining_content = extraction.cleaned_text[len(self._emitted_content):]
            if remaining_content:
                deltas.append({"content": remaining_content})
                self._emitted_content += remaining_content
            if self.tool_calls:
                self._suppressed = True
                tool_deltas = list(
                    server._stream_tool_call_deltas(
                        self.tool_calls,
                        argument_chunk_chars=self._argument_chunk_chars,
                    )
                )
                self._emitted_tool_deltas = bool(tool_deltas)
                deltas.extend(tool_deltas)
            return deltas

        def resolve_deferred_content(self, *, has_tool_calls: bool):
            del has_tool_calls
            return []

    server._ToolAwareContentStreamTranslator = DSV40731StreamTranslator


def _install_no_tools_stream_sanitizer(server: ModuleType) -> None:
    """Apply the official DSML sanitizer to no-tools SSE content too."""
    stock_factory = server._stream_splitter_for_state

    class DSV40731NoToolsStreamSplitter:
        def __init__(self, stock: Any) -> None:
            self._stock = stock
            self._translator = server._ToolAwareContentStreamTranslator(
                tools=[], argument_chunk_chars=1
            )

        @property
        def reentry_count(self) -> int:
            return int(self._stock.reentry_count)

        def start(self):
            return self._translate(self._stock.start())

        def feed(self, text: str):
            return self._translate(self._stock.feed(text))

        def finish(self, **kwargs):
            chunks = self._translate(self._stock.finish(**kwargs))
            try:
                chunks.extend(self._content_pairs(self._translator.finish()))
            except CandidateConstructionError:
                # A malformed completion cannot release the DSML suffix that
                # the translator withheld while it waited for official parsing.
                return chunks
            return chunks

        def _translate(self, chunks: list[tuple[str, str]]):
            translated: list[tuple[str, str]] = []
            for field, text in chunks:
                if field == "content":
                    translated.extend(self._content_pairs(self._translator.feed(field, text)))
                else:
                    translated.append((field, text))
            return translated

        @staticmethod
        def _content_pairs(deltas: list[dict[str, Any]]):
            return [
                ("content", text)
                for delta in deltas
                if isinstance((text := delta.get("content")), str) and text
            ]

    def stream_splitter_for_state(*args, **kwargs):
        stock = stock_factory(*args, **kwargs)
        if kwargs.get("suppress_orphan_tool_markup"):
            return DSV40731NoToolsStreamSplitter(stock)
        return stock

    server._stream_splitter_for_state = stream_splitter_for_state


def _install_reasoning_policy(server: ModuleType) -> None:
    def normalize(value: Any, *, default: str = "low") -> str:
        effort = str(value or default).strip().lower()
        if effort not in {"auto", "low", "high", "max"}:
            raise ValueError("reasoning_effort must be one of: auto, low, high, max")
        return effort

    def for_state(
        state: Any,
        *,
        thinking_enabled: bool,
        request_effort: str | None = None,
        allow_client_controls: bool = True,
    ) -> str | None:
        if not thinking_enabled:
            return None
        raw = request_effort if request_effort is not None and allow_client_controls else state.args.reasoning_effort
        effort = normalize(raw, default="low")
        return "low" if effort == "auto" else effort

    server._normalize_reasoning_effort = normalize
    server._reasoning_effort_for_state = for_state


def _install_construction_identity(server: ModuleType) -> None:
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()

    def apply_profile(_tokenizer: Any, _args: Any) -> dict[str, Any]:
        return {
            "profile": ENCODER_NAME,
            "source": "official_python_encoder",
            "path": None,
            "applied": True,
            "sha256": manifest_digest,
        }

    server._apply_chat_template_profile = apply_profile
    server._template_hash = lambda _tokenizer: f"{ENCODER_NAME}:{manifest_digest}"
    server._template_supports_scoped_reasoning = lambda _tokenizer: True


def install_candidate_surface(server: ModuleType) -> dict[str, str]:
    """Install the verified encoder/parser directly into the imported server."""
    encoding = _load_official_encoder()
    _self_test(encoding)
    if not hasattr(server, "_encode_rendered_chat_text"):
        # Unit fixture: retain the same strict no-special-token encoding contract.
        server._encode_rendered_chat_text = lambda tokenizer, text: list(
            tokenizer.encode(text, add_special_tokens=False)
        )
    server._encode_messages = _install_encoder(server, encoding)
    server._parse_generated_tool_calls_or_content = _install_completion_parser(server, encoding)
    if hasattr(server, "omlx_extract_tool_calls_with_thinking"):
        _install_actual_tool_extractor(server, encoding)
    if hasattr(server, "_strip_orphan_tool_markup"):
        _install_no_tools_nonstream_sanitizer(server, encoding)
    if hasattr(server, "_ToolAwareContentStreamTranslator"):
        _install_stream_translator(server, encoding)
    if hasattr(server, "_stream_splitter_for_state"):
        _install_no_tools_stream_sanitizer(server)
    _install_reasoning_policy(server)
    _install_construction_identity(server)
    server._DSV4_0731_ENCODER_INSTALLED = True
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    return {
        "encoder": ENCODER_NAME,
        "source_revision": SOURCE_REVISION,
        "asset_set_sha256": manifest_digest,
    }


def main() -> int:
    if sys.argv[1:]:
        raise CandidateConstructionError("candidate entrypoint accepts no arguments")
    from mtplx.server import openai as server
    from mtplx.cli import main as mtplx_main

    install_candidate_surface(server)
    return mtplx_main(
        [
            "serve",
            "--host", "127.0.0.1",
            "--port", "8081",
            "--model", "/Users/davidtai/models/DeepSeek-V4-Flash-0731-oQ2e-mtp",
            "--model-id", "deepseek-v4-0731-candidate",
            "--reasoning", "on",
            "--reasoning-effort", "low",
            "--reasoning-parser", "qwen3",
            "--tool-prompt-mode", "native",
            "--chat-template-profile", "tokenizer",
            "--warmup-tokens", "0",
            "--no-stats-footer",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
