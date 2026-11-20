"""The fixed continuations every arm is scored on.

A corpus entry is one finished conversation from a reference run, rendered
with the model's own chat template into token ids, with the spans the model
wrote marked: thinking, answers and tool calls, never the prompt or tool
results. Scoring those spans on two servers compares their next-token
distributions exactly where the model writes.

Conversations with pictures are left out (scoring takes token ids, and the
image positions would need the pixels); text turns are what this measures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

CORPUS_SCHEMA = "mtplx.quality.corpus/1"


def _templated(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Messages as the chat template reads them: tool arguments as mappings."""

    out = []
    for message in messages:
        calls = message.get("tool_calls")
        if calls:
            fixed = []
            for call in calls:
                function = dict(call.get("function") or {})
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
                fixed.append({**call, "function": function})
            message = {**message, "tool_calls": fixed}
        out.append(message)
    return out


def _render(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, open_turn: bool) -> str:
    return tokenizer.apply_chat_template(
        messages, tools=tools or None, tokenize=False, add_generation_prompt=open_turn
    )


def entry_from_row(tokenizer: Any, row: dict[str, Any], *, max_tokens: int) -> dict[str, Any] | None:
    """One corpus entry from a task row, or None when nothing can be scored."""

    if row.get("kind") == "image" or row.get("outcome") == "error":
        return None
    messages = _templated(row["messages"])
    tools = row.get("tools") or []
    full_text = _render(tokenizer, messages, tools, open_turn=False)
    char_spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        start = len(_render(tokenizer, messages[:index], tools, open_turn=True))
        through = _render(tokenizer, messages[: index + 1], tools, open_turn=False)
        if not full_text.startswith(through):
            raise ValueError(f"{row['task']}: the template does not render turns as prefixes")
        char_spans.append((start, len(through)))
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    spans = []
    for char_start, char_end in char_spans:
        tokens = [t for t, (a, _) in enumerate(offsets) if char_start <= a < char_end]
        if tokens and tokens[0] < max_tokens:  # a span past the limit is cut there
            spans.append((tokens[0], min(tokens[-1] + 1, max_tokens)))
    if not spans:
        return None
    end = spans[-1][1]
    return {
        "schema": CORPUS_SCHEMA,
        "id": f"{row['task']}-s{row['seed']}",
        "task": row["task"],
        "token_ids": ids[:end],
        "spans": spans,
        "scored_tokens": sum(b - a for a, b in spans),
    }


def build_corpus(tokenizer_dir: Path, rows: Iterable[dict[str, Any]], *, max_tokens: int = 8192) -> list[dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    entries = []
    for row in rows:
        entry = entry_from_row(tokenizer, row, max_tokens=max_tokens)
        if entry is not None:
            entries.append(entry)
    return entries
