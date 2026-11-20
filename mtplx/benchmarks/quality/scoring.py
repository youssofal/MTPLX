"""Scoring the corpus on one server, and comparing two servers' scores.

A score row keeps, for each position the model wrote, the server's top-K map
and the log-probability of the token that follows: everything the divergence
needs, and nothing of the prompt positions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from mtplx.benchmarks.quality.client import Server
from mtplx.benchmarks.quality.divergence import (
    DivergenceSummary,
    PositionDivergence,
    position_divergence,
    summarize,
)

SCORE_SCHEMA = "mtplx.quality.score/1"


def tokens_digest(token_ids: list[int]) -> str:
    """What two score rows must share to be compared: the text they scored."""

    return hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()[:16]


def top_map(ranked: list[tuple[int, float]], scored: tuple[int, float], text_of: Any) -> dict[str, float]:
    """A position's top-K map as a server's scoring returns it: keyed by each
    token's text, the higher log-probability kept where two tokens share a
    text, and the token that follows always present with its own value."""

    row: dict[str, float] = {}
    for token_id, logprob in ranked:
        row.setdefault(text_of(token_id), round(float(logprob), 5))
    row[text_of(scored[0])] = round(float(scored[1]), 5)
    return row


def score_entry(server: Server, entry: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    response = server.score(entry["token_ids"], top_k=top_k)
    logprobs = response["choices"][0]["logprobs"]
    if logprobs.get("token_ids") != entry["token_ids"]:
        raise ValueError(f"{entry['id']}: the server scored different tokens than were sent")
    positions = []
    for start, end in entry["spans"]:
        for index in range(start, end):
            top = logprobs["top_logprobs"][index]
            positions.append(
                [index, round(float(logprobs["token_logprobs"][index]), 5), {k: round(float(v), 5) for k, v in top.items()}]
            )
    return {
        "schema": SCORE_SCHEMA,
        "id": entry["id"],
        "tokens_digest": tokens_digest(entry["token_ids"]),
        "top_k": top_k,
        "positions": positions,
        "prompt_eval_time_s": (response.get("mtplx_stats") or {}).get("prompt_eval_time_s"),
    }


def write_scores(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def compare(reference: Iterable[dict[str, Any]], arm: Iterable[dict[str, Any]]) -> tuple[DivergenceSummary, dict[str, float]]:
    """Divergence of ``arm`` from ``reference`` over the entries both scored."""

    by_id = {row["id"]: row for row in reference}
    entries: list[list[PositionDivergence]] = []
    per_entry: dict[str, float] = {}
    for row in arm:
        ref = by_id.get(row["id"])
        if ref is None:
            continue
        if ref["tokens_digest"] != row["tokens_digest"]:
            raise ValueError(f"{row['id']}: the reference and the arm scored different text")
        divergences = [
            position_divergence(rt, at, reference_logprob=rl, arm_logprob=al)
            for (ri, rl, rt), (ai, al, at) in zip(ref["positions"], row["positions"], strict=True)
            if ri == ai
        ]
        entries.append(divergences)
        per_entry[row["id"]] = sum(d.kl for d in divergences) / max(1, len(divergences))
    return summarize(entries), per_entry
