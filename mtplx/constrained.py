"""Grammar-constrained decoding (structured output) for the serial lanes.

Issue #186: ``response_format`` of type ``json_object`` / ``json_schema`` (and
opt-in strict tool calls) is enforced with llguidance token bitmasks applied
to target logits before sampling. Constrained requests run on the serial AR
lane and the serial MTP lane and bypass the batched AR pump. The AR lane masks
each step's row; an MTP verify window masks each of its rows at that row's own
grammar position (``mask_window_logits``), so both lanes commit every token
from the same masked law.

llguidance is an optional dependency: requests that do not use
``response_format`` never touch it, and requests that do get a clear 400 when
it is missing instead of silent non-enforcement (which is what shipped before
this module existed).

The mask must hit the logits row before any shaping (temperature, top-p/k,
penalties) so that both the greedy argmax branch and the sampled branch of
``_sample_from_logits`` operate on the constrained distribution. Illegal
tokens are set to -inf, which survives every downstream shaping step.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shlex
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

# The floor the [server] extra declares in pyproject.toml (raised for #547).
LLGUIDANCE_REQUIREMENT = "llguidance>=1.8"
_LLGUIDANCE_MIN_RELEASE = (1, 8)


def _installed_llguidance_version() -> str | None:
    """The installed llguidance distribution's version, read without importing it."""
    try:
        return importlib.metadata.version("llguidance")
    except importlib.metadata.PackageNotFoundError:
        return None


def _below_llguidance_floor(version: str | None) -> bool:
    match = re.match(r"(\d+)\.(\d+)", version or "")
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) < _LLGUIDANCE_MIN_RELEASE


def _llguidance_unavailable_reason(exc: BaseException, installed: str | None) -> str:
    """Why llguidance cannot be used, in words that point at the fix.

    "Not installed" is only true when the package itself is missing. An
    installed llguidance can still fail to load: 1.3.0 has no get_version()
    (#643), and a submodule can fail on its own imports. Saying "not installed"
    then sends the user to a pip install that reports it is already satisfied.
    """
    if isinstance(exc, ModuleNotFoundError) and exc.name == "llguidance":
        return "llguidance is not installed"
    version = f" {installed}" if installed else ""
    reason = (
        f"llguidance{version} is installed but failed to load "
        f"({type(exc).__name__}: {exc})"
    )
    if _below_llguidance_floor(installed):
        reason += "; MTPLX needs llguidance 1.8 or later"
    return reason


def llguidance_install_command() -> str:
    """The pip command that gives the Python running MTPLX a usable llguidance.

    It names this interpreter because a bare ``pip`` often belongs to another
    Python (Homebrew installs MTPLX into its own environment).
    """
    return f"{shlex.quote(sys.executable)} -m pip install -U '{LLGUIDANCE_REQUIREMENT}'"


try:  # pragma: no cover - exercised via LLGUIDANCE_AVAILABLE branches
    import llguidance as _llg
    import llguidance.hf as _llg_hf
    import llguidance.mlx as _llg_mlx

    LLGUIDANCE_VERSION = str(_llg.get_version())
    LLGUIDANCE_AVAILABLE = True
    LLGUIDANCE_UNAVAILABLE_REASON: str | None = None
except Exception as _llg_import_error:  # pragma: no cover
    # llguidance is optional: no failure to load it may stop the server from
    # starting. Every failure is kept as the reason the refusals and
    # `mtplx doctor` report.
    _llg = None
    _llg_hf = None
    _llg_mlx = None
    LLGUIDANCE_AVAILABLE = False
    LLGUIDANCE_VERSION = None
    LLGUIDANCE_UNAVAILABLE_REASON = _llguidance_unavailable_reason(
        _llg_import_error, _installed_llguidance_version()
    )


def llguidance_unavailable_message(feature: str) -> str:
    """The refusal text for a feature that needs llguidance when it cannot load."""
    return (
        f"{feature} cannot be enforced without the optional llguidance "
        f"dependency ({LLGUIDANCE_REQUIREMENT}): {LLGUIDANCE_UNAVAILABLE_REASON}. "
        f"Install it with: {llguidance_install_command()}"
    )


def llguidance_status() -> dict[str, Any]:
    """Structured-output readiness for `mtplx doctor`."""
    installed = _installed_llguidance_version()
    if LLGUIDANCE_AVAILABLE:
        # The loaded library's own version is the truth when it loaded.
        release = _llguidance_release()
        meets_requirement = release is not None and release[:2] >= _LLGUIDANCE_MIN_RELEASE
    else:
        meets_requirement = installed is not None and not _below_llguidance_floor(installed)
    return {
        "available": LLGUIDANCE_AVAILABLE,
        "version": LLGUIDANCE_VERSION,
        "installed_version": installed,
        "required": LLGUIDANCE_REQUIREMENT,
        "meets_requirement": meets_requirement,
        "reason": LLGUIDANCE_UNAVAILABLE_REASON,
        "install_command": llguidance_install_command(),
    }

SUPPORTED_RESPONSE_FORMAT_TYPES = ("text", "json_object", "json_schema")

# ``json_object`` promises a JSON object (OpenAI semantics), not merely any
# JSON value, so the generic grammar pins the top-level type.
_JSON_OBJECT_SCHEMA: dict[str, Any] = {"type": "object"}

# Whitespace the JSON grammars admit wherever JSON allows it: the `space` rule
# of llama.cpp's json-schema-to-grammar (`| " " | "\n"{1,2} [ \t]{0,20}`), i.e.
# one space, or one or two newlines followed by at most 20 spaces or tabs.
# llguidance's default is `[\x20\x0A\x0D\x09]+` at every such point: unbounded,
# CR included. Whitespace leaves the grammar exactly where it was, so once a
# whitespace token narrowly beats every legal value token under greedy decoding
# it wins again at the next step, and the document never finishes (#547:
# `"tests_run":`, two spaces, then 640 CR tokens until max_tokens).
_JSON_WHITESPACE_PATTERN = r"\x20|\x0A{1,2}[\x20\x09]{0,20}"


def _llguidance_release() -> tuple[int, int, int] | None:
    match = re.search(r"llguidance@(\d+)\.(\d+)\.(\d+)", LLGUIDANCE_VERSION or "")
    if match is None:
        return None
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


# llguidance matches the JSON whitespace pattern at most once per position from
# 1.8.0 on (guidance-ai/llguidance#370, "single-match skip lexemes"). Earlier
# releases re-apply it any number of times at one position, so a bound written
# into the pattern bounds nothing there.
_JSON_WHITESPACE_SINGLE_MATCH = (_llguidance_release() or (0, 0, 0)) >= (1, 8, 0)


def _json_compile_options() -> dict[str, Any]:
    """llguidance JSON options under which no whitespace run is unbounded."""
    if _JSON_WHITESPACE_SINGLE_MATCH:
        return {"whitespace_pattern": _JSON_WHITESPACE_PATTERN}
    # Without single-match whitespace the only finite layout is whitespace the
    # grammar matches exactly once: one optional space after each separator,
    # as in {"a": 1, "b": [true, false]} (no pretty-printed newlines).
    return {
        "whitespace_flexible": False,
        "item_separator": r",\x20?",
        "key_separator": r":\x20?",
    }


# The x-guidance keys that together decide JSON whitespace. They interact:
# whitespace_pattern overrides whitespace_flexible, and with flexible
# whitespace off the separators carry whatever whitespace is allowed. A client
# that sets any of them therefore gets exactly its own policy, none of ours.
_WHITESPACE_OPTION_KEYS = frozenset(
    {"whitespace_flexible", "whitespace_pattern", "item_separator", "key_separator"}
)


def _grammar_schema_json(schema: dict[str, Any]) -> str:
    """The JSON Schema text llguidance compiles for a request.

    llguidance fixes object properties to the order they appear in the schema
    (and resolves allOf/anyOf/additionalProperties precedence by position), so
    key order is part of the grammar: this text keeps the client's order, and
    the grammar caches key on exactly this text. Sorting the keys here used to
    force alphabetical properties (#547).

    The compile options travel as the schema's own top-level ``x-guidance``
    object, the one place both ``grammar_from_json_schema`` and a lark
    ``%json`` block read them. The bounded-whitespace defaults apply only when
    the client's ``x-guidance`` sets no whitespace option of its own.
    """
    client_options = schema.get("x-guidance", {})
    if not isinstance(client_options, dict):
        raise ResponseFormatError("JSON Schema 'x-guidance' must be an object")
    if _WHITESPACE_OPTION_KEYS.isdisjoint(client_options):
        options = {**_json_compile_options(), **client_options}
    else:
        options = dict(client_options)
    return json.dumps({**schema, "x-guidance": options}, separators=(",", ":"))


# Strict tool-call constraint markers. These are the Qwen/Hermes-family
# native tool-call and thinking tokens; strict mode only activates when the
# runtime tokenizer encodes each marker as a single (special) token, so the
# grammar can reference it as a hard boundary rather than bytes.
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
THINK_START = "<think>"
THINK_END = "</think>"


def tool_call_strict_enabled() -> bool:
    """Opt-in via MTPLX_TOOL_CALL_STRICT=1/true/on. Default off."""
    return (os.environ.get("MTPLX_TOOL_CALL_STRICT") or "").strip().lower() in {
        "1",
        "true",
        "on",
    }


class ResponseFormatError(ValueError):
    """Invalid or unsupported ``response_format``; message is client-safe."""


@dataclass(frozen=True)
class ConstraintSpec:
    """A validated, tokenizer-independent grammar for one request.

    Built once at request-validation time (so bad schemas 400 before any
    model work) and bound to the runtime tokenizer lazily via ``build`` —
    once per generation attempt, because matcher state is consumed by a
    generation and blank-retry attempts must start fresh.

    ``grammar_with_prelude`` exists for response_format grammars on
    thinking templates: it accepts a leading ``TEXT </think>`` so the model
    can close reasoning the chat template opened inside the prompt. It is
    selected only when the prompt actually ends inside an open think block —
    otherwise the prelude's free-text rule would let the model write prose
    forever without ever starting the document.
    """

    grammar: str
    source_type: str
    grammar_with_prelude: str | None = None
    think_start_id: int | None = None
    think_end_id: int | None = None

    def build(
        self, tokenizer: Any, prompt_ids: list[int] | None = None
    ) -> "GrammarConstraint":
        grammar = self.grammar
        if self.grammar_with_prelude is not None and _prompt_ends_inside_think(
            prompt_ids, self.think_start_id, self.think_end_id
        ):
            grammar = self.grammar_with_prelude
        return GrammarConstraint(grammar, tokenizer)


def _prompt_ends_inside_think(
    prompt_ids: list[int] | None,
    think_start_id: int | None,
    think_end_id: int | None,
) -> bool:
    if not prompt_ids or think_start_id is None:
        return False
    for token in reversed(prompt_ids):
        if token == think_start_id:
            return True
        if think_end_id is not None and token == think_end_id:
            return False
    return False


def constraint_spec_from_response_format(
    response_format: Any,
    tokenizer: Any | None = None,
) -> ConstraintSpec | None:
    """Parse/validate a request's ``response_format`` into a ConstraintSpec.

    Returns None when no constraint applies (absent or ``type: text``).
    Raises ResponseFormatError for anything the server cannot honestly
    enforce — the caller turns that into a 400.

    When a tokenizer is provided and its template family uses native
    thinking markers, the grammar accepts an optional leading thinking
    close (chat templates open ``<think>`` inside the generation prompt, so
    the model's reasoning must be allowed to finish before the document).
    """
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise ResponseFormatError(
            "response_format must be an object with a 'type' field"
        )
    format_type = response_format.get("type")
    if format_type not in SUPPORTED_RESPONSE_FORMAT_TYPES:
        raise ResponseFormatError(
            "unsupported response_format type "
            f"{format_type!r}; supported: {', '.join(SUPPORTED_RESPONSE_FORMAT_TYPES)}"
        )
    if format_type == "text":
        return None
    if not LLGUIDANCE_AVAILABLE:
        raise ResponseFormatError(
            llguidance_unavailable_message(f"response_format type {format_type!r}")
            + "; refusing to silently return unconstrained output"
        )
    if format_type == "json_object":
        schema = _JSON_OBJECT_SCHEMA
    else:
        wrapper = response_format.get("json_schema")
        if wrapper is None and isinstance(response_format.get("schema"), dict):
            # Lenient shape some clients send: {"type": "json_schema",
            # "schema": {...}} without the OpenAI wrapper object.
            schema = response_format["schema"]
        elif isinstance(wrapper, dict):
            schema = wrapper.get("schema")
        else:
            schema = None
        if not isinstance(schema, dict):
            raise ResponseFormatError(
                "response_format type 'json_schema' requires json_schema.schema "
                "to be a JSON Schema object"
            )
    schema_json = _grammar_schema_json(schema)
    grammar = _cached_grammar_for_schema(schema_json, think_prelude=False)
    think_start_id = (
        _single_token_id(tokenizer, THINK_START) if tokenizer is not None else None
    )
    think_end_id = (
        _single_token_id(tokenizer, THINK_END) if tokenizer is not None else None
    )
    grammar_with_prelude = (
        _cached_grammar_for_schema(schema_json, think_prelude=True)
        if think_start_id is not None and think_end_id is not None
        else None
    )
    return ConstraintSpec(
        grammar=grammar,
        source_type=str(format_type),
        grammar_with_prelude=grammar_with_prelude,
        think_start_id=think_start_id,
        think_end_id=think_end_id,
    )


def tool_call_constraint_spec(
    tools: Any,
    tool_choice: Any,
    tokenizer: Any,
) -> ConstraintSpec | None:
    """Build a strict tool-call ConstraintSpec from a request's tools.

    The grammar allows free text (and native thinking blocks) but forces any
    tool-call envelope the model opens to carry a declared tool name and
    schema-valid arguments. Returns None when no constraint applies
    (tool_choice "none", or no function tools declared). Raises
    ResponseFormatError for shapes strict mode cannot honestly enforce.
    """
    if isinstance(tool_choice, str) and tool_choice == "none":
        return None
    functions = _function_tools(tools)
    if not functions:
        return None
    if tool_choice is not None and tool_choice != "auto":
        raise ResponseFormatError(
            "strict tool calls support tool_choice 'auto' or 'none' only; "
            f"got {tool_choice!r}"
        )
    if not LLGUIDANCE_AVAILABLE:
        raise ResponseFormatError(llguidance_unavailable_message("strict tool calls"))
    if (
        _single_token_id(tokenizer, TOOL_CALL_START) is None
        or _single_token_id(tokenizer, TOOL_CALL_END) is None
    ):
        raise ResponseFormatError(
            "strict tool calls require the chat template's tool-call markers "
            f"({TOOL_CALL_START} / {TOOL_CALL_END}) to be single special "
            "tokens; this model's template is not supported yet"
        )
    include_think = (
        _single_token_id(tokenizer, THINK_START) is not None
        and _single_token_id(tokenizer, THINK_END) is not None
    )
    # Key order inside each tool schema is part of its grammar (see
    # _grammar_schema_json), so the key keeps it rather than sorting it away.
    cache_key = "structtool:" + json.dumps(
        {
            "functions": [[name, schema] for name, schema in functions],
            "think": include_think,
            "llg": LLGUIDANCE_VERSION,
            "json_options": _json_compile_options(),
        },
        separators=(",", ":"),
    )
    with _CACHE_LOCK:
        cached = _GRAMMAR_CACHE.get(cache_key)
        if cached is not None:
            _GRAMMAR_CACHE.move_to_end(cache_key)
            return ConstraintSpec(grammar=cached, source_type="tool_call_strict")
    grammar = _tool_call_lark_grammar(functions, include_think=include_think)
    err = _llg.LLMatcher.validate_grammar(grammar)
    if err:
        raise ResponseFormatError(f"unsupported tool schema: {err}")
    with _CACHE_LOCK:
        _GRAMMAR_CACHE[cache_key] = grammar
        while len(_GRAMMAR_CACHE) > _GRAMMAR_CACHE_MAX:
            _GRAMMAR_CACHE.popitem(last=False)
    return ConstraintSpec(grammar=grammar, source_type="tool_call_strict")


def _function_tools(tools: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(tools, list):
        return []
    functions: list[tuple[str, dict[str, Any]]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise ResponseFormatError("each tool must be an object")
        function = tool.get("function") if tool.get("type") == "function" else None
        if function is None and "name" in tool:
            function = tool
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ResponseFormatError("each function tool must declare a name")
        parameters = function.get("parameters")
        if parameters is None:
            parameters = {"type": "object"}
        if not isinstance(parameters, dict):
            raise ResponseFormatError(
                f"tool {name!r} parameters must be a JSON Schema object"
            )
        functions.append((name, parameters))
    return functions


def _lark_string(text: str) -> str:
    """A lark string literal; JSON escaping is a valid subset."""
    return json.dumps(text)


_THINK_PRELUDE_DEFAULT_MAX_CHARS = 4000

def _think_prelude_max_chars() -> int:
    """Character cap on the reasoning segment that precedes constrained output.

    The think prelude exists because Qwen-style templates open ``<think>`` inside
    the generation prompt, so generation starts mid-reasoning and the grammar must
    allow the model back out (see #186). That prelude was unbounded free text, which
    makes one failure mode *legal*: a model that never emits ``</think>`` stays inside
    the prelude and fills ``max_tokens`` with prose, returning no document at all.
    Reported symptom on the tool-call side in #196 ("the content channel fills with
    the model's reasoning narration ... until finish: length, no tool call emitted").

    Bounding the prelude regex makes the grammar itself force the close. Because the
    bound is carried by the sampling-time token mask rather than by scheduler state,
    it cannot go stale under speculative decoding -- the failure mode that silently
    disabled vLLM's thinking budget whenever MTP was on, fixed only in vLLM 0.21.0.

    ``MTPLX_THINK_PRELUDE_MAX_CHARS=0`` restores the previous unbounded behaviour.
    """
    raw = os.environ.get("MTPLX_THINK_PRELUDE_MAX_CHARS")
    if raw is None or raw.strip() == "":
        return _THINK_PRELUDE_DEFAULT_MAX_CHARS
    try:
        value = int(raw)
    except ValueError:
        return _THINK_PRELUDE_DEFAULT_MAX_CHARS
    return value if value > 0 else 0


def _prelude_terminal(max_chars: int) -> str:
    """The prelude's own terminal, so bounding it never touches tail/free text."""
    if max_chars <= 0:
        return "PRELUDE_TEXT: /(.|\\n)*/\n"
    return f"PRELUDE_TEXT: /(.|\\n){{0,{max_chars}}}/\n"


def _tool_call_lark_grammar(
    functions: list[tuple[str, dict[str, Any]]],
    *,
    include_think: bool,
) -> str:
    """Free text + forced tool-call envelopes as a lark grammar.

    Special tokens must appear at the rule level (llguidance rejects them
    inside terminals), and the closing marker must be a bare special-token
    reference — inside a quoted string it would match bytes the special
    token never produces.
    """
    alternatives = []
    for name, schema in functions:
        name_inner = json.dumps(name)[1:-1]
        head = f'\n{{"name": "{name_inner}", "arguments": '
        alternatives.append(
            f"TAG_TEXT <tool_call> {_lark_string(head)} %json "
            f"{_grammar_schema_json(schema)} {_lark_string('}')} "
            f"{_lark_string(chr(10))} </tool_call>"
        )
    if include_think:
        alternatives.append("TAG_TEXT <think> TAG_TEXT </think>")
    seg = "seg: " + "\n   | ".join(alternatives)
    # The optional prelude closes a thinking block the chat template opened
    # inside the generation prompt (Qwen renders `<|im_start|>assistant\n
    # <think>\n`, so generation begins mid-think and must be allowed out).
    prelude = "prelude: PRELUDE_TEXT </think>\n" if include_think else ""
    prelude_terminal = (
        _prelude_terminal(_think_prelude_max_chars()) if include_think else ""
    )
    start = "start: prelude? (seg)* tail\n" if include_think else "start: (seg)* tail\n"
    return (
        "%llguidance {}\n"
        f"{start}"
        f"{prelude}"
        "tail: TAG_TEXT\n"
        "TAG_TEXT: /(.|\\n)*/\n"
        f"{prelude_terminal}"
        f"{seg}\n"
    )


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    unwrapped = _unwrap_hf_tokenizer(tokenizer)
    try:
        ids = unwrapped.encode(text, add_special_tokens=False)
    except Exception:
        return None
    return int(ids[0]) if len(ids) == 1 else None


class GrammarConstraint:
    """Per-generation matcher state: mask logits rows, advance per token.

    The llguidance tokenizer wrap needs the model's logits width (which can
    exceed the tokenizer vocab on padded lm_heads), so binding is deferred to
    the first ``mask_logits_row`` call, where the row's shape provides it.
    Tokens beyond the tokenizer vocab are always masked out.
    """

    def __init__(self, grammar: str, tokenizer: Any):
        self._grammar = grammar
        self._tokenizer = tokenizer
        self._matcher: Any | None = None
        self._bitmask: Any | None = None
        self.masked_steps = 0
        self.mask_time_s = 0.0

    def _bind(self, n_vocab: int) -> None:
        ll_tokenizer = _cached_ll_tokenizer(self._tokenizer, n_vocab)
        matcher = _llg.LLMatcher(ll_tokenizer, self._grammar)
        err = matcher.get_error()
        if err:
            raise ResponseFormatError(f"response_format grammar rejected: {err}")
        self._matcher = matcher
        self._bitmask = _llg_mlx.allocate_token_bitmask(1, n_vocab)

    def mask_logits_row(
        self, row: Any, *, prefix: list[int] | tuple[int, ...] = ()
    ) -> Any:
        """Return a masked copy of a 1-D logits row (mx.array).

        The mask is the grammar's at the current step or, with ``prefix``, at
        the step after those tokens (the bonus row that follows accepted
        drafts). ``row`` itself is left raw: a row that outlives this request
        (the next cycle's row, the session bank's final state) must never
        carry its grammar. The matcher ends where it started.
        """
        if self._matcher is None:
            self._bind(int(row.shape[-1]))
        if self._matcher.is_stopped():
            return row
        prefix = [int(t) for t in prefix]
        if prefix and self._matcher.validate_tokens(prefix) != len(prefix):
            return row
        started = time.perf_counter()
        if prefix and not self._matcher.consume_tokens(prefix):
            raise RuntimeError(
                f"constrained decoding desync on a mask prefix: {self._matcher.get_error()}"
            )
        try:
            _llg_mlx.fill_next_token_bitmask(self._matcher, self._bitmask)
        finally:
            if prefix and not self._matcher.rollback(len(prefix)):
                raise RuntimeError("constrained decoding could not roll back a mask prefix")
        masked = _llg_mlx.apply_token_bitmask(row.reshape(1, -1), self._bitmask)
        self.mask_time_s += time.perf_counter() - started
        self.masked_steps += 1
        return masked.reshape(row.shape)

    def mask_window_logits(self, logits: Any, window_tokens: list[int]) -> Any:
        """Return a copy of a speculative window's rows, each masked at its own
        grammar position.

        ``logits`` is ``(1, T, V)``: row ``j`` scores the token after the
        committed stream and ``window_tokens[:j]``. Every row up to the
        grammar-legal prefix of the window gets exactly the mask
        ``mask_logits_row`` computes at that position, so a verify window
        accepts, corrects and samples from the same masked law the AR lane
        samples each step from. Rows past an illegal window token are copied
        as they are: the row before them gives that token zero probability, so
        no window commits past it. ``logits`` itself is left raw (see
        ``mask_logits_row``); use the copy only to verify and sample. The
        matcher ends where it started (llguidance advances through the window
        and rolls back).
        """
        import mlx.core as mx

        if self._matcher is None:
            self._bind(int(logits.shape[-1]))
        if self._matcher.is_stopped():
            return logits
        started = time.perf_counter()
        rows = int(logits.shape[1])
        drafts = [int(t) for t in window_tokens[: max(0, rows - 1)]]
        # One mask row per window position; rows llguidance does not reach
        # (past an illegal token, or after the document ends) stay all-legal.
        bitmask = _llg_mlx.allocate_token_bitmask(len(drafts) + 1, int(logits.shape[-1]))
        self._matcher.unsafe_compute_mask_ptr_with_draft_token(
            bitmask.ctypes.data, bitmask.shape[1] * bitmask.itemsize, drafts
        )
        head = _llg_mlx.apply_token_bitmask(logits[0, : len(drafts) + 1, :], bitmask)
        masked = (
            mx.concatenate([head[None], logits[:, len(drafts) + 1 :, :]], axis=1)
            if len(drafts) + 1 < rows
            else head[None]
        )
        self.mask_time_s += time.perf_counter() - started
        self.masked_steps += len(drafts) + 1
        return masked

    def advance(self, token_id: int) -> None:
        if self._matcher is None or self._matcher.is_stopped():
            return
        self._matcher.consume_token(int(token_id))
        err = self._matcher.get_error()
        if err:
            # A committed token the grammar rejects means the decode loop
            # desynced from the matcher — fail loudly rather than stream
            # unconstrained output labeled as completed.
            raise RuntimeError(
                f"constrained decoding desync on token {int(token_id)}: {err}"
            )

    def advance_many(self, token_ids: list[int]) -> None:
        for token_id in token_ids:
            self.advance(token_id)

    def validate_prefix(self, token_ids: list[int]) -> int:
        """How many of token_ids extend the current state legally (no mutation).

        Speculative windows get clamped to this prefix; the matcher itself
        only ever advances through tokens that were actually committed.
        """
        if not token_ids:
            return 0
        if self._matcher is None or self._matcher.is_stopped():
            return 0
        return int(self._matcher.validate_tokens([int(t) for t in token_ids]))

    @property
    def stopped(self) -> bool:
        return self._matcher is not None and bool(self._matcher.is_stopped())

    @property
    def completed(self) -> bool:
        """True when the emitted text is a complete document per the grammar."""
        if self._matcher is None or self._matcher.get_error():
            return False
        return bool(self._matcher.is_accepting() or self._matcher.is_stopped())


# --- caches ---------------------------------------------------------------
#
# A compiled grammar is schema- and engine-version-specific (key order
# included: two schemas that differ only in property order are different
# grammars); the LLTokenizer wrap is tokenizer-object- and vocab-width-specific.
# Both caches hold strong references (the server keeps one tokenizer for its
# lifetime) and are bounded, so id() reuse after GC cannot alias a live entry.

_GRAMMAR_CACHE: OrderedDict[str, str] = OrderedDict()
_GRAMMAR_CACHE_MAX = 64
_TOKENIZER_CACHE: OrderedDict[tuple[int, int], tuple[Any, Any]] = OrderedDict()
_TOKENIZER_CACHE_MAX = 4
_CACHE_LOCK = threading.Lock()


def _cached_grammar_for_schema(schema_json: str, *, think_prelude: bool = False) -> str:
    prelude_max = _think_prelude_max_chars() if think_prelude else 0
    key = (
        f"{LLGUIDANCE_VERSION}:think={int(think_prelude)}"
        f":pmax={prelude_max}:{schema_json}"
    )
    with _CACHE_LOCK:
        cached = _GRAMMAR_CACHE.get(key)
        if cached is not None:
            _GRAMMAR_CACHE.move_to_end(key)
            return cached
    try:
        if think_prelude:
            grammar = (
                "%llguidance {}\n"
                "start: prelude? doc\n"
                "prelude: PRELUDE_TEXT </think>\n"
                f"{_prelude_terminal(prelude_max)}"
                f"doc: %json {schema_json}\n"
            )
        else:
            grammar = _llg.LLMatcher.grammar_from_json_schema(schema_json)
    except Exception as exc:
        raise ResponseFormatError(f"unsupported JSON Schema: {exc}") from exc
    err = _llg.LLMatcher.validate_grammar(grammar)
    if err:
        raise ResponseFormatError(f"unsupported JSON Schema: {err}")
    with _CACHE_LOCK:
        _GRAMMAR_CACHE[key] = grammar
        while len(_GRAMMAR_CACHE) > _GRAMMAR_CACHE_MAX:
            _GRAMMAR_CACHE.popitem(last=False)
    return grammar


def _unwrap_hf_tokenizer(tokenizer: Any) -> Any:
    """Return the underlying fast tokenizer llguidance requires.

    The runtime hands us mlx_lm's TokenizerWrapper, which delegates
    attribute access to the fast tokenizer it holds but fails llguidance's
    strict isinstance check; unwrap it when present.
    """
    import transformers

    if isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
        return tokenizer
    inner = getattr(tokenizer, "_tokenizer", None)
    if inner is not None and isinstance(inner, transformers.PreTrainedTokenizerFast):
        return inner
    return tokenizer


def _cached_ll_tokenizer(tokenizer: Any, n_vocab: int) -> Any:
    tokenizer = _unwrap_hf_tokenizer(tokenizer)
    key = (id(tokenizer), int(n_vocab))
    with _CACHE_LOCK:
        entry = _TOKENIZER_CACHE.get(key)
        if entry is not None and entry[0] is tokenizer:
            _TOKENIZER_CACHE.move_to_end(key)
            return entry[1]
    ll_tokenizer = _llg_hf.from_tokenizer(tokenizer, n_vocab=int(n_vocab))
    with _CACHE_LOCK:
        _TOKENIZER_CACHE[key] = (tokenizer, ll_tokenizer)
        while len(_TOKENIZER_CACHE) > _TOKENIZER_CACHE_MAX:
            _TOKENIZER_CACHE.popitem(last=False)
    return ll_tokenizer
