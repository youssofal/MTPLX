"""Build an FR-Spec draft vocabulary table from the user's own text.

The built-in ``qwen38-code-64k`` table is ranked on code. A token outside the
table can never be drafted, so text in other languages loses acceptance where
the table misses its tokens. This module ranks token ids by their frequency in
local text (plain text, Markdown, code, JSON and chat/session JSONL) and
writes a table in the format ``frspec_draft.load_frspec_ids`` reads, plus a
``<name>.meta.json`` sidecar with the tokenizer fingerprint and row order.

Only CPU and the model's ``tokenizer.json`` are used; no weights are loaded.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .frspec_draft import (
    _BUILTIN_VOCABS,
    tokenizer_vocab_fingerprint,
    vocab_sidecar_path,
)

SIDECAR_FORMAT = "mtplx-frspec-vocab"
SIDECAR_VERSION = 1
DEFAULT_ROWS = 65_536
DEFAULT_MAX_FILE_BYTES = 8 * 1024 * 1024
DEFAULT_ROLES = ("assistant",)
# Single strings above this size are pasted files or dumps, not model text.
MAX_TEXT_CHARS = 200_000
ENCODE_BATCH = 256

# Tokenizer vocabulary each built-in table was ranked with
# (``frspec_draft.tokenizer_vocab_fingerprint``). Every Qwen3.5 / 3.6 / 3.8
# pack shares this mapping, whatever its pre-tokenizer settings.
BUILTIN_TOKENIZER_FINGERPRINTS = {
    "qwen38-code-64k": "2bac0d5354ab7df80bc41f42d69ef2a4c8e5f9408eee83d9950606341051dd54",
}

TEXT_SUFFIXES = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".adoc", ".tex", ".csv", ".tsv", ".html",
    ".htm", ".xml", ".css", ".scss", ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx",
    ".mjs", ".cjs", ".go", ".rs", ".java", ".kt", ".kts", ".scala", ".swift", ".m",
    ".mm", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".pl", ".lua", ".r",
    ".jl", ".dart", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".sql", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".jinja", ".vue", ".svelte",
})
JSON_SUFFIXES = frozenset({".json"})
JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})

PRIVACY_NOTICE = (
    "Note: the table reveals which tokens are frequent in the input text. "
    "Treat it, and its .meta.json, as derived from that input."
)


class FrspecBuildError(ValueError):
    """A build request that cannot produce a valid table."""


# ---------------------------------------------------------------------------
# Corpus: files -> text pieces
# ---------------------------------------------------------------------------


@dataclass
class CorpusStats:
    files_read: int = 0
    files_skipped: int = 0
    files_truncated: int = 0


def collect_files(inputs: Iterable[str | Path]) -> list[Path]:
    """Expand files and directories into a sorted, de-duplicated file list.

    Directories contribute files with a known text, code or JSON suffix;
    hidden entries (``.git``, ``.venv``) are skipped. A file named directly is
    always read, as plain text unless its suffix says JSON.
    """

    found: dict[Path, None] = {}
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_file():
            found[path.resolve()] = None
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                relative = child.relative_to(path).parts
                if any(part.startswith(".") for part in relative):
                    continue
                if child.is_file() and _known_suffix(child):
                    found[child.resolve()] = None
        else:
            raise FrspecBuildError(f"input not found: {path}")
    return sorted(found)


def _known_suffix(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in TEXT_SUFFIXES or suffix in JSON_SUFFIXES or suffix in JSONL_SUFFIXES


def read_capped(path: Path, max_bytes: int, stats: CorpusStats) -> str | None:
    """File contents up to ``max_bytes``; None for unreadable or binary files."""

    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError:
        stats.files_skipped += 1
        return None
    if b"\x00" in data[:8192]:
        stats.files_skipped += 1
        return None
    if len(data) > max_bytes:
        data = data[:max_bytes]
        stats.files_truncated += 1
    stats.files_read += 1
    return data.decode("utf-8", errors="replace")


def file_texts(path: Path, content: str, roles: Iterable[str]) -> Iterator[str]:
    """Text pieces of one file, by format."""

    suffix = path.suffix.lower()
    if suffix in JSONL_SUFFIXES:
        yield from jsonl_texts(content, roles)
    elif suffix in JSON_SUFFIXES:
        try:
            yield from json_texts(json.loads(content), roles)
        except ValueError:
            yield content
    else:
        yield content


def jsonl_texts(content: str, roles: Iterable[str]) -> Iterator[str]:
    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue  # a truncated last line, or a non-JSON line
    yield from json_texts(records, roles)


def json_texts(payload: Any, roles: Iterable[str]) -> Iterator[str]:
    """Chat content when the payload holds chat records, else every string value.

    Chat records are Claude Code / Anthropic session lines
    (``{"message": {"role", "content"}}``), OpenAI messages
    (``{"role", "content"}``) and conversations (``{"messages": [...]}``).
    When any record is a chat record, the other records (summaries, ids,
    snapshots) are skipped.
    """

    records = payload if isinstance(payload, list) else [payload]
    messages = [m for record in records for m in _chat_messages(record)]
    if not messages:
        yield from _string_values(payload)
        return
    wanted = {role.lower() for role in roles}
    for message in messages:
        if wanted and str(message.get("role") or "").lower() not in wanted:
            continue
        yield from _message_texts(message)


def _chat_messages(record: Any) -> list[dict[str, Any]]:
    if not isinstance(record, dict):
        return []
    if _is_message(record):
        return [record]
    inner = record.get("message")
    if _is_message(inner):
        return [inner]
    conversation = record.get("messages")
    if isinstance(conversation, list):
        return [m for m in conversation if _is_message(m)]
    return []


def _is_message(value: Any) -> bool:
    return isinstance(value, dict) and "role" in value and (
        "content" in value or "tool_calls" in value
    )


def _message_texts(message: dict[str, Any]) -> Iterator[str]:
    yield from _content_texts(message.get("content"))
    for call in message.get("tool_calls") or []:
        arguments = (call.get("function") or {}).get("arguments") if isinstance(call, dict) else None
        if isinstance(arguments, str):
            yield arguments


def _content_texts(content: Any) -> Iterator[str]:
    if isinstance(content, str):
        yield content
        return
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, str):
            yield block
        elif isinstance(block, dict):
            kind = block.get("type")
            if kind == "text":
                yield str(block.get("text") or "")
            elif kind == "thinking":
                yield str(block.get("thinking") or "")
            elif kind == "tool_use":
                yield json.dumps(block.get("input") or {}, ensure_ascii=False)
            elif kind == "tool_result":
                yield from _content_texts(block.get("content"))


def _string_values(payload: Any) -> Iterator[str]:
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _string_values(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _string_values(value)


def corpus_texts(
    files: Iterable[Path],
    *,
    max_file_bytes: int,
    roles: Iterable[str],
    stats: CorpusStats,
) -> Iterator[str]:
    roles = tuple(roles)
    for path in files:
        content = read_capped(path, max_file_bytes, stats)
        if content is None:
            continue
        for text in file_texts(path, content, roles):
            if text and len(text) <= MAX_TEXT_CHARS:
                yield text


def split_holdout(files: list[Path], fraction: float) -> tuple[list[Path], list[Path]]:
    """Deterministic split by file: (ranking files, held-out files)."""

    if not 0.0 <= fraction < 1.0:
        raise FrspecBuildError("--holdout must be in [0, 1)")
    if fraction == 0.0:
        return list(files), []
    cut = round(fraction * 10_000)
    held = [p for p in files if zlib.crc32(str(p).encode("utf-8")) % 10_000 < cut]
    held_set = set(held)
    return [p for p in files if p not in held_set], held


# ---------------------------------------------------------------------------
# Tokenizer facts
# ---------------------------------------------------------------------------


def load_tokenizer(tokenizer_json: Path) -> Any:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_json))


def count_tokens(tokenizer: Any, texts: Iterable[str], vocab_size: int) -> np.ndarray:
    """Token frequency per id over ``texts`` (no special tokens added)."""

    counts = np.zeros(vocab_size, dtype=np.int64)
    batch: list[str] = []

    def flush() -> None:
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        ids = [token_id for encoding in encodings for token_id in encoding.ids]
        if ids:
            counts[:] += np.bincount(ids, minlength=vocab_size)[:vocab_size]
        batch.clear()

    for text in texts:
        batch.append(text)
        if len(batch) >= ENCODE_BATCH:
            flush()
    if batch:
        flush()
    return counts


def added_token_ids(tokenizer_json: Path) -> list[int]:
    """Ids of every added token: specials plus markers such as ``<think>`` or ``<tool_call>``."""

    payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    return sorted(int(token["id"]) for token in payload.get("added_tokens") or [])


def byte_token_ids(tokenizer: Any) -> list[int]:
    """Ids that spell one byte: byte-level BPE's 256-symbol alphabet and ``<0xNN>`` fallbacks.

    With these in the table the draft can propose any byte sequence.
    """

    candidates = list(_byte_level_alphabet()) + [f"<0x{b:02X}>" for b in range(256)]
    ids = {tokenizer.token_to_id(token) for token in candidates}
    return sorted(i for i in ids if i is not None)


def _byte_level_alphabet() -> list[str]:
    """GPT-2 ``bytes_to_unicode``: the printable stand-in for each byte."""

    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = list(printable)
    extra = 0
    for byte in range(256):
        if byte not in printable:
            printable.append(byte)
            chars.append(256 + extra)
            extra += 1
    return [chr(c) for c in chars]


def excluded_token_ids(tokenizer: Any, strings: Iterable[str]) -> tuple[set[int], list[str]]:
    """Ids for strings that are one token (as written, or with a leading space).

    Returns (ids, strings that are not a single token and were not excluded).
    """

    ids: set[int] = set()
    unmatched: list[str] = []
    for text in strings:
        matched = False
        exact = tokenizer.token_to_id(text)
        if exact is not None:
            ids.add(exact)
            matched = True
        for variant in (text, " " + text):
            encoded = tokenizer.encode(variant, add_special_tokens=False).ids
            if len(encoded) == 1:
                ids.add(encoded[0])
                matched = True
        if not matched:
            unmatched.append(text)
    return ids, unmatched


def read_exclude_file(path: str | Path) -> list[str]:
    lines = Path(path).expanduser().read_text(encoding="utf-8").splitlines()
    return [line.rstrip("\r") for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def builtin_ids(name: str) -> list[int]:
    path = _BUILTIN_VOCABS.get(name)
    if path is None:
        raise FrspecBuildError(
            f"unknown built-in table {name!r}; known: {', '.join(sorted(_BUILTIN_VOCABS))}"
        )
    return np.load(path, allow_pickle=False).reshape(-1).astype(int).tolist()


def builtin_for_fingerprint(fingerprint: str) -> str | None:
    for name, expected in BUILTIN_TOKENIZER_FINGERPRINTS.items():
        if expected == fingerprint and name in _BUILTIN_VOCABS:
            return name
    return None


@dataclass
class RankedTable:
    ids: list[int]
    from_input: int
    filled: int
    forced: int
    excluded: int = 0
    notes: list[str] = field(default_factory=list)


def rank_table(
    counts: np.ndarray,
    *,
    rows: int,
    forced: Iterable[int],
    fill: Iterable[int],
    exclude: Iterable[int] = (),
) -> RankedTable:
    """Most-frequent-first table of at most ``rows`` ids.

    Order: forced ids (added and byte tokens), then ids seen in the input by
    descending count (ties by id), then ``fill`` ids not yet present. Excluded
    ids never enter, except forced ones.
    """

    forced_ids = list(dict.fromkeys(int(i) for i in forced))
    blocked = {int(i) for i in exclude} - set(forced_ids)
    chosen: dict[int, None] = dict.fromkeys(forced_ids)
    seen = np.flatnonzero(counts)
    order = seen[np.lexsort((seen, -counts[seen]))]
    from_input = 0
    for token_id in order.tolist():
        if len(chosen) >= rows:
            break
        if token_id in blocked or token_id in chosen:
            continue
        chosen[token_id] = None
        from_input += 1
    filled = 0
    for token_id in fill:
        if len(chosen) >= rows:
            break
        token_id = int(token_id)
        if token_id in blocked or token_id in chosen:
            continue
        chosen[token_id] = None
        filled += 1
    ids = list(chosen)[:rows]
    return RankedTable(
        ids=ids,
        from_input=from_input,
        filled=filled,
        forced=len(forced_ids),
        excluded=len(blocked),
    )


def coverage(counts: np.ndarray, ids: Iterable[int]) -> float | None:
    total = int(counts.sum())
    if total == 0:
        return None
    index = np.asarray([i for i in ids if 0 <= i < counts.shape[0]], dtype=np.int64)
    return float(counts[index].sum()) / total


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_table(
    out: Path,
    table: RankedTable,
    *,
    order: str,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    """Write ``<out>.npy`` (int32) and its ``.meta.json`` sidecar.

    ``ascending`` matches the built-in table (row-sorted for gathering; the
    pre-scatter draft read requires it). ``ranked`` keeps frequency order so
    ``MTPLX_FRSPEC_N`` can cut it.
    """

    if out.suffix != ".npy":
        raise FrspecBuildError("--out must end in .npy")
    ids = sorted(table.ids) if order == "ascending" else list(table.ids)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(ids, dtype=np.int32), allow_pickle=False)
    sidecar = vocab_sidecar_path(out)
    payload = {
        "format": SIDECAR_FORMAT,
        "version": SIDECAR_VERSION,
        "rows": len(ids),
        "order": order,
        **metadata,
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out, sidecar


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class BuildRequest:
    tokenizer_json: Path
    inputs: list[str]
    out: Path
    rows: int = DEFAULT_ROWS
    fill_from: str | None = None  # None: family built-in if any, else "ids"
    order: str = "ascending"
    holdout: float = 0.0
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES
    roles: tuple[str, ...] = DEFAULT_ROLES
    exclude_strings: list[str] = field(default_factory=list)


def build_frspec_table(request: BuildRequest) -> dict[str, Any]:
    """Build and write the table; return the report printed by the CLI."""

    tokenizer = load_tokenizer(request.tokenizer_json)
    vocab_size = int(tokenizer.get_vocab_size(with_added_tokens=True))
    if not 0 < request.rows < vocab_size:
        raise FrspecBuildError(f"--rows must be in [1, {vocab_size - 1}] for this tokenizer")
    fingerprint = tokenizer_vocab_fingerprint(request.tokenizer_json)
    family_builtin = builtin_for_fingerprint(fingerprint)
    fill_name, fill = _resolve_fill(request.fill_from, family_builtin, fingerprint, vocab_size)

    files = collect_files(request.inputs)
    if not files:
        raise FrspecBuildError("no readable input files")
    ranking_files, held_files = split_holdout(files, request.holdout)
    stats = CorpusStats()
    counts = count_tokens(
        tokenizer,
        corpus_texts(ranking_files, max_file_bytes=request.max_file_bytes,
                     roles=request.roles, stats=stats),
        vocab_size,
    )
    if int(counts.sum()) == 0:
        raise FrspecBuildError("the input produced no tokens")

    forced = sorted(set(added_token_ids(request.tokenizer_json) + byte_token_ids(tokenizer)))
    if request.rows < len(forced):
        raise FrspecBuildError(f"--rows must be at least {len(forced)} (added and byte tokens)")
    excluded, unmatched = excluded_token_ids(tokenizer, request.exclude_strings)
    table = rank_table(counts, rows=request.rows, forced=forced, fill=fill, exclude=excluded)

    report: dict[str, Any] = {
        "out": str(request.out),
        "rows": len(table.ids),
        "rows_requested": request.rows,
        "order": request.order,
        "fill_from": fill_name,
        "files_read": stats.files_read,
        "files_skipped": stats.files_skipped,
        "files_truncated": stats.files_truncated,
        "tokens": int(counts.sum()),
        "unique_ids": int(np.count_nonzero(counts)),
        "forced_ids": table.forced,
        "from_input": table.from_input,
        "filled": table.filled,
        "excluded_ids": table.excluded,
        "exclude_unmatched": unmatched,
        "coverage_ranking_text": _round(coverage(counts, table.ids)),
    }
    if held_files:
        report["holdout"] = _holdout_report(
            tokenizer, held_files, request, vocab_size, table.ids, family_builtin
        )
    metadata = {
        "tokenizer_vocab_sha256": fingerprint,
        "tokenizer_json_sha256": _file_sha256(request.tokenizer_json),
        "tokenizer_vocab_size": vocab_size,
        "fill_from": fill_name,
        "from_input": table.from_input,
        "filled": table.filled,
        "forced": table.forced,
        "tokens": report["tokens"],
    }
    write_table(request.out, table, order=request.order, metadata=metadata)
    report["meta"] = str(vocab_sidecar_path(request.out))
    return report


def _resolve_fill(
    fill_from: str | None,
    family_builtin: str | None,
    fingerprint: str,
    vocab_size: int,
) -> tuple[str, list[int]]:
    """Fill source after the input's own ids.

    ``builtin:<name>`` adds that table's ids, ``ids`` adds unseen ids in
    tokenizer id order (for BPE: earlier merges first), ``none`` adds nothing
    (the table may then be shorter than ``--rows``).
    """

    choice = fill_from or (f"builtin:{family_builtin}" if family_builtin else "ids")
    if choice == "none":
        return choice, []
    if choice == "ids":
        return choice, list(range(vocab_size))
    if choice.startswith("builtin:"):
        name = choice.removeprefix("builtin:")
        ids = builtin_ids(name)
        expected = BUILTIN_TOKENIZER_FINGERPRINTS.get(name)
        if expected is not None and expected != fingerprint:
            raise FrspecBuildError(
                f"built-in table {name!r} was ranked with a different tokenizer vocabulary"
            )
        return choice, ids
    raise FrspecBuildError("--fill-from must be builtin:<name>, ids or none")


def _holdout_report(
    tokenizer: Any,
    held_files: list[Path],
    request: BuildRequest,
    vocab_size: int,
    table_ids: list[int],
    family_builtin: str | None,
) -> dict[str, Any]:
    stats = CorpusStats()
    counts = count_tokens(
        tokenizer,
        corpus_texts(held_files, max_file_bytes=request.max_file_bytes,
                     roles=request.roles, stats=stats),
        vocab_size,
    )
    report: dict[str, Any] = {
        "files": stats.files_read,
        "tokens": int(counts.sum()),
        "coverage_table": _round(coverage(counts, table_ids)),
    }
    if family_builtin is not None:
        report["builtin"] = family_builtin
        report["coverage_builtin"] = _round(coverage(counts, builtin_ids(family_builtin)))
    return report


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 5)


@dataclass
class TokenizerGroup:
    fingerprint: str
    builtin: str | None
    packs: list[tuple[str, Path]] = field(default_factory=list)


def group_packs_by_tokenizer(
    packs: Iterable[tuple[str, Path]],
) -> tuple[list[TokenizerGroup], list[tuple[str, Path]]]:
    """Group model packs by tokenizer vocabulary: one table serves a whole group.

    Returns (groups, packs without a readable tokenizer.json). Identical files
    are fingerprinted once.
    """

    groups: dict[str, TokenizerGroup] = {}
    by_file_hash: dict[str, str] = {}
    unusable: list[tuple[str, Path]] = []
    for name, path in packs:
        tokenizer_json = Path(path) / "tokenizer.json"
        try:
            file_hash = _file_sha256(tokenizer_json)
            fingerprint = by_file_hash.get(file_hash) or tokenizer_vocab_fingerprint(tokenizer_json)
        except (OSError, ValueError, KeyError, TypeError):
            unusable.append((name, Path(path)))
            continue
        by_file_hash[file_hash] = fingerprint
        group = groups.setdefault(
            fingerprint, TokenizerGroup(fingerprint, builtin_for_fingerprint(fingerprint))
        )
        group.packs.append((name, Path(path)))
    ordered = sorted(groups.values(), key=lambda g: (-len(g.packs), g.fingerprint))
    return ordered, unusable


def format_tokenizer_groups(
    groups: list[TokenizerGroup], unusable: list[tuple[str, Path]]
) -> str:
    if not groups:
        return "No local model packs with a tokenizer.json (see `mtplx models`)."
    lines = ["Local packs by tokenizer (one FR-Spec table serves every pack in a group):"]
    for group in groups:
        builtin = f"built-in table: {group.builtin}" if group.builtin else "no built-in table"
        lines.append(f"  tokenizer {group.fingerprint[:12]} ({builtin})")
        for name, path in group.packs:
            lines.append(f"    - {name}  {path}")
    if unusable:
        lines.append(f"  {len(unusable)} more without a tokenizer.json (not usable here)")
    return "\n".join(lines)


def format_report(report: dict[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100 * value:.2f}%"

    read = (
        f"{report['files_read']:,} files, {report['tokens']:,} tokens, "
        f"{report['unique_ids']:,} unique ids"
    )
    if report["files_truncated"]:
        read += f", {report['files_truncated']} truncated at the size cap"
    if report["files_skipped"]:
        read += f", {report['files_skipped']} skipped (binary or unreadable)"
    rows = (
        f"{report['forced_ids']:,} added/byte tokens, {report['from_input']:,} from input, "
        f"{report['filled']:,} filled from {report['fill_from']}"
    )
    lines = [
        f"FR-Spec table: {report['out']} ({report['rows']:,} ids, {report['order']})",
        f"  metadata:     {report['meta']}",
        f"  input:        {read}",
        f"  rows:         {rows}",
    ]
    if report["rows"] < report["rows_requested"]:
        lines.append(f"  note:         fewer rows than the requested {report['rows_requested']:,}")
    if report["excluded_ids"] or report["exclude_unmatched"]:
        lines.append(f"  excluded:     {report['excluded_ids']} ids")
        for text in report["exclude_unmatched"]:
            lines.append(f"  not excluded: {text!r} is not a single token")
    lines.append(f"  coverage:     {pct(report['coverage_ranking_text'])} of the ranking text")
    held = report.get("holdout")
    if held:
        line = (
            f"  held out:     {held['files']:,} files, {held['tokens']:,} tokens: "
            f"table {pct(held['coverage_table'])}"
        )
        if "coverage_builtin" in held:
            line += f", builtin:{held['builtin']} {pct(held['coverage_builtin'])}"
        lines.append(line)
    lines.append(
        f"  use:          MTPLX_FRSPEC_DRAFT=1 MTPLX_FRSPEC_VOCAB={report['out']}"
    )
    lines.append(PRIVACY_NOTICE)
    return "\n".join(lines)
