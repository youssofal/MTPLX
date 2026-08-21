#!/usr/bin/env python3
"""Build frequency-ranked token-id subsets for the FR-Spec reduced draft head.

Counts Qwen3.8-27B tokenizer ids over a mixed corpus (English web/edu, English
wikipedia, math/reasoning, multi-language code, Chinese wikipedia,
chat-templated dialogue), unions them with a hard-coded always-keep set
(specials, control ids >= 248000, digits, punctuation, whitespace/indent,
single-byte tokens), and writes sorted int32 ``ids_16k/32k/48k/64k.npy`` plus a
coverage report.  Coverage is measured on a per-domain held-out 10% split.

Reference implementation of the method: the corpus and tokenizer paths below
are local to the machine this was built on.  Needs ``tokenizers``, ``numpy``,
``pyarrow`` and ``datasets``::

    python scripts/build_vocab_subset.py --tokenizer <model dir> [--offline]
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import unicodedata
import urllib.request
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CORPUS_ROOT = Path(os.environ.get("VOCAB_CORPUS_ROOT", "corpus"))
DEFAULT_TOKENIZER = CORPUS_ROOT / "models/Qwen3.8-27B"
VOCAB_SIZE = 248320          # text_config.vocab_size of Qwen3.8-27B
CONTROL_FLOOR = 248000       # everything above is chat/control specials

SIZES = {"16k": 16384, "32k": 32768, "48k": 49152, "64k": 65536}

CHUNK_CHARS = 4000
BATCH_DOCS = 256


def _read_text_file(path: Path, budget: int):
    if not path.is_file():
        return
    got = 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while got < budget:
            block = handle.read(1 << 20)
            if not block:
                break
            got += len(block)
            yield block


def _arrow_column(path: Path, column: str, budget: int):
    try:
        import pyarrow as pa
    except ImportError:
        return
    if not path.is_file():
        return
    got = 0
    with pa.memory_map(str(path), "rb") as source:
        try:
            reader = pa.ipc.open_stream(source)
        except pa.ArrowInvalid:
            source.seek(0)
            reader = pa.ipc.open_file(source)
        batches = (
            reader
            if hasattr(reader, "__iter__")
            else (reader.get_batch(i) for i in range(reader.num_record_batches))
        )
        for batch in batches:
            if column not in batch.schema.names:
                return
            for value in batch.column(column).to_pylist():
                if not value:
                    continue
                got += len(value)
                yield value
                if got >= budget:
                    return


def _parquet_column(path: Path, column: str, budget: int):
    import pyarrow.parquet as pq

    if not path.is_file():
        return
    handle = pq.ParquetFile(str(path))
    got = 0
    for rg in range(handle.num_row_groups):
        table = handle.read_row_group(rg, columns=[column])
        for value in table.column(column).to_pylist():
            if not value:
                continue
            got += len(value)
            yield value
            if got >= budget:
                return


class _HttpRangeFile(io.RawIOBase):
    """Minimal random-access file over HTTP range requests (for parquet)."""

    def __init__(self, url: str):
        self.url = url
        self.pos = 0
        self.downloaded = 0
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=60
        ) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.pos
        if size == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + size, self.size) - 1
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{end}"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
        self.pos += len(data)
        self.downloaded += len(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _http_parquet_column(url: str, column: str, budget: int):
    import pyarrow.parquet as pq

    handle = _HttpRangeFile(url)
    reader = pq.ParquetFile(handle)
    got = 0
    for rg in range(reader.num_row_groups):
        table = reader.read_row_group(rg, columns=[column])
        for value in table.column(column).to_pylist():
            if not value:
                continue
            got += len(value)
            yield value
            if got >= budget:
                return


def _http_gz_jsonl(url: str, field: str, byte_budget: int, char_budget: int):
    request = urllib.request.Request(url, headers={"Range": f"bytes=0-{byte_budget}"})
    with urllib.request.urlopen(request, timeout=180) as response:
        raw = response.read()
    text = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(raw).decode(
        "utf-8", "ignore"
    )
    got = 0
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        value = record.get(field) or ""
        if not value:
            continue
        got += len(value)
        yield value
        if got >= char_budget:
            return


def _json_strings(path: Path, budget: int):
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return
    got = 0
    stack = [data]
    while stack and got < budget:
        node = stack.pop()
        if isinstance(node, str):
            if node:
                got += len(node)
                yield node
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)


def _local_code(budget: int):
    roots = [
        (CORPUS_ROOT / "llama.cpp", ("*.c", "*.cpp", "*.h", "*.hpp", "*.cu", "*.metal",
                              "*.cmake", "*.sh", "*.py")),
        (CORPUS_ROOT / "mtplx/repo", ("*.py", "*.ts", "*.js", "*.metal", "*.toml",
                               "*.yaml", "*.yml", "*.json", "*.sh")),
        (CORPUS_ROOT / "train", ("*.py", "*.sh")),
        (CORPUS_ROOT / "scripts", ("*.py", "*.sh")),
    ]
    got = 0
    for root, patterns in roots:
        if not root.is_dir():
            continue
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if got >= budget:
                    return
                try:
                    if path.stat().st_size > 400_000:
                        continue
                    body = path.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                if not body.strip():
                    continue
                got += len(body)
                yield body


def _chat_docs(tokenizer_dir: Path, budget: int):
    """Chat-template rendered dialogue so control/role tokens get counted."""
    try:
        import transformers

        tok = transformers.AutoTokenizer.from_pretrained(
            str(tokenizer_dir), trust_remote_code=False
        )
    except Exception as exc:
        print(f"  [chat] transformers unavailable ({exc}); using raw template", file=sys.stderr)
        tok = None
    seeds: list[tuple[str, str]] = []
    for path in [
        CORPUS_ROOT / "mtplx/eval/data/gsm8k_200.json",
        CORPUS_ROOT / "mtplx/eval/data/mmlu_redux_300.json",
        CORPUS_ROOT / "mtplx/eval/data/aime25_30.json",
        CORPUS_ROOT / "eval/data/arc_c_200.json",
    ]:
        if not path.is_file():
            continue
        try:
            rows = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(rows, dict):
            rows = rows.get("data") or rows.get("rows") or []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question") or row.get("prompt") or row.get("problem") or "")
            answer = str(row.get("answer") or row.get("solution") or row.get("target") or "")
            if question:
                seeds.append((question, answer))
    got = 0
    for question, answer in seeds:
        messages = [{"role": "user", "content": question}]
        if answer:
            messages.append({"role": "assistant", "content": answer})
        if tok is not None:
            try:
                rendered = tok.apply_chat_template(messages, tokenize=False)
            except Exception:
                rendered = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
        else:
            rendered = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"
        got += len(rendered)
        yield rendered
        if got >= budget:
            return


def build_sources(tokenizer_dir: Path, offline: bool, scale: float):
    """Return {domain: iterable-of-text-generators}."""
    mb = lambda x: int(x * 1024 * 1024 * scale)  # noqa: E731
    wt103 = (
        Path.home()
        / ".cache/huggingface/datasets/Salesforce___wikitext/wikitext-103-raw-v1/0.0.0"
        / "b08601e04326c79dfdd32d625aee71d232d685c3"
    )
    gsm8k = (
        Path.home()
        / ".cache/huggingface/datasets/openai___gsm8k/main/0.0.0"
        / "740312add88f781978c0658806c59bc2815b9866"
    )
    math500 = (
        Path.home()
        / ".cache/huggingface/datasets/HuggingFaceH4___math-500/default/0.0.0"
        / "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    )
    mmlu = (
        Path.home()
        / ".cache/huggingface/datasets/cais___mmlu/all/0.0.0"
        / "c30699e8356da336a370243923dbaf21066bb9fe"
    )

    sources: dict[str, list] = {
        "web_en": [
            ("fineweb-edu train.txt", _read_text_file(CORPUS_ROOT / "train/data/train.txt", mb(26))),
            (
                "fineweb-edu parquet",
                _parquet_column(CORPUS_ROOT / "train/data/fwe_000.parquet", "text", mb(24)),
            ),
        ],
        "wiki_en": [
            (
                "wikitext-103 raw (shard 0)",
                _arrow_column(wt103 / "wikitext-train-00000-of-00002.arrow", "text", mb(30)),
            ),
            (
                "wikitext-2 raw",
                _read_text_file(
                    CORPUS_ROOT / "eval/data/wikitext-2-raw/wiki.train.raw", mb(10)
                ),
            ),
        ],
        "math": [
            ("gsm8k train", _arrow_column(gsm8k / "gsm8k-train.arrow", "answer", mb(3))),
            ("gsm8k questions", _arrow_column(gsm8k / "gsm8k-train.arrow", "question", mb(3))),
            ("math-500 solutions", _arrow_column(math500 / "math-500-test.arrow", "solution", mb(2))),
            ("mmlu (all)", _arrow_column(mmlu / "mmlu-test.arrow", "question", mb(3))),
            ("mmlu auxiliary_train", _arrow_column(mmlu / "mmlu-auxiliary_train.arrow", "question", mb(4))),
            ("mtplx eval gsm8k", _json_strings(CORPUS_ROOT / "mtplx/eval/data/gsm8k_200.json", mb(1))),
            ("mtplx eval aime25", _json_strings(CORPUS_ROOT / "mtplx/eval/data/aime25_30.json", mb(1))),
            ("mtplx eval gpqa", _json_strings(CORPUS_ROOT / "mtplx/eval/data/gpqa_diamond_198.json", mb(1))),
            ("local mmlu_500", _json_strings(CORPUS_ROOT / "eval/data/mmlu_500.json", mb(1))),
        ],
        "code": [("local repos (c/c++/cuda/metal/py/ts/sh)", _local_code(mb(36)))],
        "zh": [],
        "chat": [("chat-template rendered eval prompts", _chat_docs(tokenizer_dir, mb(6)))],
    }

    if not offline:
        sources["code"].append(
            (
                "rosetta-code (multi-language)",
                _http_parquet_column(
                    "https://huggingface.co/datasets/christopher/rosetta-code/"
                    "resolve/main/data/train-00000-of-00001-8b4da49264116bbf.parquet",
                    "code",
                    mb(12),
                ),
            )
        )
        sources["code"].append(
            (
                "codeparrot-clean-valid (python)",
                _http_gz_jsonl(
                    "https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid/"
                    "resolve/main/file-000000000054.json.gz",
                    "content",
                    12_000_000,
                    mb(14),
                ),
            )
        )
        sources["zh"].append(
            (
                "wikipedia zh 20231101 (row groups)",
                _http_parquet_column(
                    "https://huggingface.co/datasets/wikimedia/wikipedia/"
                    "resolve/main/20231101.zh/train-00002-of-00006.parquet",
                    "text",
                    mb(24),
                ),
            )
        )
    return sources


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / Qwen byte-level BPE byte <-> printable-unicode map."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def decode_vocab(tokenizer_json: Path) -> tuple[dict[int, str], set[int]]:
    """Return {id: decoded-text} and the set of added/special ids."""
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    byte_decoder = {v: k for k, v in _bytes_to_unicode().items()}
    decoded: dict[int, str] = {}
    for token, idx in vocab.items():
        try:
            raw = bytes(byte_decoder[ch] for ch in token)
        except KeyError:
            decoded[int(idx)] = token
            continue
        decoded[int(idx)] = raw.decode("utf-8", errors="replace")
    special: set[int] = set()
    for entry in data.get("added_tokens") or []:
        special.add(int(entry["id"]))
        decoded.setdefault(int(entry["id"]), str(entry.get("content", "")))
    return decoded, special


_PUNCT_CATS = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sm", "Sc", "Sk", "So"}


def always_keep(decoded: dict[int, str], special: set[int]) -> dict[str, set[int]]:
    groups: dict[str, set[int]] = {
        "special_added": set(special),
        "control_ge_248000": {i for i in range(CONTROL_FLOOR, VOCAB_SIZE)},
        "single_byte": set(),
        "digits": set(),
        "whitespace_indent": set(),
        "punct_symbol": set(),
        "code_symbol_runs": set(),
    }
    for idx, text in decoded.items():
        if idx >= VOCAB_SIZE:
            continue
        if not text:
            continue
        if len(text.encode("utf-8", "ignore")) == 1:
            groups["single_byte"].add(idx)
        core = text.strip(" ")
        if core and all(ch.isdigit() for ch in core):
            groups["digits"].add(idx)
        if all(ch in " \t\n\r\x0b\x0c" for ch in text):
            groups["whitespace_indent"].add(idx)
        if core and all(unicodedata.category(ch) in _PUNCT_CATS for ch in core):
            groups["punct_symbol"].add(idx)
        if text and all(
            (33 <= ord(ch) <= 47) or (58 <= ord(ch) <= 64)
            or (91 <= ord(ch) <= 96) or (123 <= ord(ch) <= 126)
            or ch in " \t\n\r"
            for ch in text
        ):
            groups["code_symbol_runs"].add(idx)
    return groups


def chunks(text: str, size: int = CHUNK_CHARS):
    for start in range(0, len(text), size):
        yield text[start : start + size]


def count_domain(tokenizer, docs, vocab_size: int, holdout_every: int = 10):
    """Return (train_counts, holdout_counts, n_tokens_train, n_tokens_holdout)."""
    train = np.zeros(vocab_size, dtype=np.int64)
    holdout = np.zeros(vocab_size, dtype=np.int64)
    batch: list[str] = []
    flags: list[bool] = []
    doc_index = 0
    encode = getattr(tokenizer, "encode_batch_fast", None) or tokenizer.encode_batch

    def flush():
        if not batch:
            return
        encodings = encode(batch, add_special_tokens=False)
        train_ids: list[np.ndarray] = []
        hold_ids: list[np.ndarray] = []
        for encoding, is_hold in zip(encodings, flags):
            arr = np.asarray(encoding.ids, dtype=np.int64)
            (hold_ids if is_hold else train_ids).append(arr)
        if train_ids:
            flat = np.concatenate(train_ids)
            train[: vocab_size] += np.bincount(flat, minlength=vocab_size)[:vocab_size]
        if hold_ids:
            flat = np.concatenate(hold_ids)
            holdout[: vocab_size] += np.bincount(flat, minlength=vocab_size)[:vocab_size]
        batch.clear()
        flags.clear()

    for doc in docs:
        is_hold = (doc_index % holdout_every) == (holdout_every - 1)
        doc_index += 1
        for piece in chunks(doc):
            batch.append(piece)
            flags.append(is_hold)
            if len(batch) >= BATCH_DOCS:
                flush()
    flush()
    return train, holdout, int(train.sum()), int(holdout.sum())


def coverage(counts: np.ndarray, ids: np.ndarray) -> float:
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    return float(counts[ids].sum()) / total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--out-dir", default=str(HERE / "ids"))
    parser.add_argument("--report", default=str(HERE / "vocab_coverage.md"))
    parser.add_argument("--offline", action="store_true", help="skip network sources")
    parser.add_argument("--scale", type=float, default=1.0, help="scale all budgets")
    parser.add_argument(
        "--weights",
        default="web_en=0.22,wiki_en=0.12,code=0.34,math=0.16,chat=0.06,zh=0.10",
        help=(
            "per-domain share of the ranking mass (default: a coding/agent "
            "workload mix). Use 'equal' for a flat 1/N split."
        ),
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore the cached token counts"
    )
    args = parser.parse_args()

    from tokenizers import Tokenizer

    tokenizer_dir = Path(args.tokenizer)
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"tokenizer      : {tokenizer_json}")
    print(f"vocab_size     : {VOCAB_SIZE}")

    decoded, special = decode_vocab(tokenizer_json)
    groups = always_keep(decoded, special)
    keep = set()
    for name, ids in groups.items():
        keep |= ids
        print(f"  always-keep {name:<20} {len(ids):>6}")
    keep_ids = np.array(sorted(keep), dtype=np.int64)
    print(f"  always-keep TOTAL              {keep_ids.size}")

    cache_path = out_dir / "counts_cache.npz"
    meta_cache = out_dir / "counts_cache.json"
    domain_train: dict[str, np.ndarray] = {}
    domain_hold: dict[str, np.ndarray] = {}
    domain_meta: dict[str, dict] = {}

    if cache_path.is_file() and meta_cache.is_file() and not args.refresh:
        blob = np.load(cache_path)
        domain_meta = json.loads(meta_cache.read_text())
        for domain in domain_meta:
            domain_train[domain] = blob[f"{domain}__train"]
            domain_hold[domain] = blob[f"{domain}__hold"]
        print(f"reusing cached counts from {cache_path} (--refresh to recount)")
        sources = {}
    else:
        sources = build_sources(tokenizer_dir, args.offline, args.scale)

    for domain, entries in sources.items():
        started = time.perf_counter()
        train = np.zeros(VOCAB_SIZE, dtype=np.int64)
        hold = np.zeros(VOCAB_SIZE, dtype=np.int64)
        used: list[str] = []
        for label, generator in entries:
            try:
                t, h, nt, nh = count_domain(tokenizer, generator, VOCAB_SIZE)
            except Exception as exc:
                print(f"  [{domain}] SKIP {label}: {type(exc).__name__}: {exc}")
                continue
            if nt + nh == 0:
                print(f"  [{domain}] empty {label}")
                continue
            train += t
            hold += h
            used.append(f"{label} ({nt + nh:,} tok)")
            print(f"  [{domain}] {label}: {nt + nh:,} tokens")
        domain_train[domain] = train
        domain_hold[domain] = hold
        domain_meta[domain] = {
            "sources": used,
            "train_tokens": int(train.sum()),
            "holdout_tokens": int(hold.sum()),
            "distinct": int((train > 0).sum()),
            "seconds": round(time.perf_counter() - started, 1),
        }
        print(
            f"  [{domain}] total {int(train.sum()) + int(hold.sum()):,} tokens, "
            f"{int((train > 0).sum()):,} distinct, "
            f"{domain_meta[domain]['seconds']}s"
        )

    if sources:
        np.savez_compressed(
            cache_path,
            **{f"{d}__train": domain_train[d] for d in domain_train},
            **{f"{d}__hold": domain_hold[d] for d in domain_hold},
        )
        meta_cache.write_text(json.dumps(domain_meta, indent=2))
        print(f"cached counts -> {cache_path}")

    # balance domains so a huge web corpus cannot bury code/zh tokens
    if args.weights.strip().lower() == "equal":
        weights = {d: 1.0 for d in domain_train}
    else:
        weights = {}
        for item in args.weights.split(","):
            if not item.strip():
                continue
            key, _, value = item.partition("=")
            weights[key.strip()] = float(value)
    live = {d: weights.get(d, 0.0) for d in domain_train if domain_train[d].sum() > 0}
    scale_w = sum(live.values()) or 1.0
    live = {d: w / scale_w for d, w in live.items()}
    print("ranking weights: " + ", ".join(f"{d}={w:.3f}" for d, w in live.items()))

    weighted = np.zeros(VOCAB_SIZE, dtype=np.float64)
    for domain, counts in domain_train.items():
        total = float(counts.sum())
        if total <= 0 or live.get(domain, 0.0) <= 0:
            continue
        weighted += live[domain] * (counts.astype(np.float64) / total)

    order = np.lexsort((np.arange(VOCAB_SIZE), -weighted))  # freq desc, id asc

    subsets: dict[str, np.ndarray] = {}
    for name, size in SIZES.items():
        if size < keep_ids.size:
            raise SystemExit(
                f"always-keep set ({keep_ids.size}) exceeds requested size {size}"
            )
        chosen = list(keep_ids)
        seen = set(keep_ids.tolist())
        for idx in order:
            if len(chosen) >= size:
                break
            token_id = int(idx)
            if token_id in seen:
                continue
            if weighted[token_id] <= 0:
                break
            seen.add(token_id)
            chosen.append(token_id)
        if len(chosen) < size:
            for token_id in range(VOCAB_SIZE):
                if len(chosen) >= size:
                    break
                if token_id not in seen:
                    seen.add(token_id)
                    chosen.append(token_id)
        ids = np.array(sorted(chosen), dtype=np.int32)
        subsets[name] = ids
        path = out_dir / f"ids_{name}.npy"
        np.save(path, ids)
        print(f"wrote {path}  ({ids.size} ids)")

    lines: list[str] = []
    lines.append("# Reduced-vocabulary draft head: corpus coverage\n")
    lines.append(
        f"Tokenizer `{tokenizer_json}` - vocab_size **{VOCAB_SIZE}**. "
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `ext/build_vocab_subset.py`.\n"
    )
    lines.append(
        "Coverage = fraction of corpus tokens whose id is inside the subset S. "
        "It is measured on a **held-out 10% document split** per domain "
        "(documents whose index mod 10 == 9 were excluded from the counts that "
        "produced the ranking).\n"
    )
    lines.append(
        "Ranking weights (share of probability mass each domain contributes to "
        "the frequency ranking): "
        + ", ".join(f"`{d}`={w:.2f}" for d, w in live.items())
        + ".\n"
    )
    lines.append("## Always-keep set\n")
    lines.append("| group | ids |")
    lines.append("|---|---:|")
    for name, ids in sorted(groups.items()):
        lines.append(f"| `{name}` | {len(ids):,} |")
    lines.append(f"| **union** | **{keep_ids.size:,}** |")
    lines.append("")
    lines.append("## Corpus\n")
    lines.append("| domain | train tokens | held-out tokens | distinct ids | sources |")
    lines.append("|---|---:|---:|---:|---|")
    for domain, meta in domain_meta.items():
        lines.append(
            f"| `{domain}` | {meta['train_tokens']:,} | {meta['holdout_tokens']:,} | "
            f"{meta['distinct']:,} | " + "; ".join(meta["sources"]) + " |"
        )
    lines.append("")
    lines.append("## Held-out coverage by subset size\n")
    header = "| domain | " + " | ".join(f"|S|={n}" for n in SIZES) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(SIZES))
    for domain in domain_meta:
        hold = domain_hold[domain]
        if hold.sum() <= 0:
            continue
        cells = [f"{100 * coverage(hold, subsets[n].astype(np.int64)):.3f}%" for n in SIZES]
        lines.append(f"| `{domain}` | " + " | ".join(cells) + " |")
    all_hold = np.zeros(VOCAB_SIZE, dtype=np.int64)
    for hold in domain_hold.values():
        all_hold += hold
    cells = [f"{100 * coverage(all_hold, subsets[n].astype(np.int64)):.3f}%" for n in SIZES]
    lines.append("| **all (pooled)** | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## In-sample coverage (ranking split)\n")
    lines.append(header)
    lines.append("|---|" + "---:|" * len(SIZES))
    for domain in domain_meta:
        train = domain_train[domain]
        if train.sum() <= 0:
            continue
        cells = [f"{100 * coverage(train, subsets[n].astype(np.int64)):.3f}%" for n in SIZES]
        lines.append(f"| `{domain}` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Draft-head cost per size\n")
    lines.append(
        "4-bit / group-size-64 affine head, K=5120: "
        "`weight = |S| x 640 uint32`, `scales = biases = |S| x 80` (fp16).\n"
    )
    lines.append("| |S| | head bytes | vs full 248,320 | read @340 GB/s |")
    lines.append("|---:|---:|---:|---:|")
    full_bytes = VOCAB_SIZE * (640 * 4 + 80 * 2 * 2)
    for name, size in SIZES.items():
        nbytes = size * (640 * 4 + 80 * 2 * 2)
        lines.append(
            f"| {size:,} ({name}) | {nbytes / 1e6:.0f} MB | "
            f"{100 * nbytes / full_bytes:.1f}% | {1000 * nbytes / 340e9:.2f} ms |"
        )
    lines.append(
        f"| {VOCAB_SIZE:,} (full) | {full_bytes / 1e6:.0f} MB | 100.0% | "
        f"{1000 * full_bytes / 340e9:.2f} ms |"
    )
    lines.append("")
    Path(args.report).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.report}")

    meta_path = out_dir / "build_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "vocab_size": VOCAB_SIZE,
                "tokenizer": str(tokenizer_json),
                "always_keep": {k: len(v) for k, v in groups.items()},
                "always_keep_total": int(keep_ids.size),
                "ranking_weights": live,
                "domains": domain_meta,
                "sizes": {k: int(v.size) for k, v in subsets.items()},
                "holdout_coverage": {
                    k: {
                        d: coverage(domain_hold[d], subsets[k].astype(np.int64))
                        for d in domain_meta
                        if domain_hold[d].sum() > 0
                    }
                    for k in SIZES
                },
            },
            indent=2,
        )
    )
    print(f"wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
