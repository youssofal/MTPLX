"""``mtplx frspec build``: FR-Spec tables from local text (tiny tokenizer, no model load)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mtplx import frspec_build
from mtplx.cli import build_parser
from mtplx.frspec_build import (
    BuildRequest,
    FrspecBuildError,
    build_frspec_table,
    json_texts,
    load_tokenizer,
    rank_table,
    split_holdout,
)
from mtplx.frspec_draft import (
    frspec_tokenizer_mismatch,
    load_frspec_ids,
    tokenizer_vocab_fingerprint,
    vocab_sidecar_path,
)

ADDED = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>"]
TRAIN_TEXT = [
    "def build_table(rows): return sorted(rows)",
    "Het concept wordt morgen besproken in het overleg.",
    "The draft proposes tokens and the target verifies them.",
] * 20


def _make_tokenizer(path: Path, extra_text: str = "") -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=420,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(TRAIN_TEXT + ([extra_text] * 20 if extra_text else []), trainer)
    tokenizer.add_special_tokens(ADDED[:3])
    tokenizer.add_tokens([ADDED[3]])
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path / "tokenizer.json"))
    return path / "tokenizer.json"


@pytest.fixture
def tokenizer_json(tmp_path: Path) -> Path:
    return _make_tokenizer(tmp_path / "pack")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "notes.md").write_text("# Overleg\n\nHet concept wordt morgen besproken.\n" * 5)
    (root / "code.py").write_text("def build_table(rows):\n    return sorted(rows)\n" * 5)
    (root / "image.png").write_bytes(b"\x89PNG\x00\x00binary")
    hidden = root / ".git"
    hidden.mkdir()
    (hidden / "config.md").write_text("secret remote url")
    return root


def _request(tokenizer_json: Path, corpus: Path, tmp_path: Path, **overrides) -> BuildRequest:
    values = {
        "tokenizer_json": tokenizer_json,
        "inputs": [str(corpus)],
        "out": tmp_path / "out" / "table.npy",
        "rows": 300,
        "fill_from": "ids",
    }
    values.update(overrides)
    return BuildRequest(**values)


def _forced_ids(tokenizer_json: Path) -> set[int]:
    tokenizer = load_tokenizer(tokenizer_json)
    added = {tokenizer.token_to_id(token) for token in ADDED}
    return added | set(frspec_build.byte_token_ids(tokenizer))


def test_table_is_int32_ascending_with_sidecar(tokenizer_json, corpus, tmp_path) -> None:
    report = build_frspec_table(_request(tokenizer_json, corpus, tmp_path))

    table = np.load(tmp_path / "out" / "table.npy")
    assert table.dtype == np.int32
    assert table.shape == (300,)
    assert np.all(np.diff(table) > 0)
    meta = json.loads(vocab_sidecar_path(tmp_path / "out" / "table.npy").read_text())
    assert meta["format"] == "mtplx-frspec-vocab"
    assert meta["order"] == "ascending"
    assert meta["rows"] == 300
    assert meta["tokenizer_vocab_sha256"] == tokenizer_vocab_fingerprint(tokenizer_json)
    assert report["files_read"] == 2  # notes.md and code.py; png and .git skipped
    assert report["from_input"] > 0


def test_ranked_order_puts_forced_then_frequent_ids_first(tokenizer_json, corpus, tmp_path) -> None:
    build_frspec_table(_request(tokenizer_json, corpus, tmp_path, order="ranked"))

    table = np.load(tmp_path / "out" / "table.npy").tolist()
    forced = _forced_ids(tokenizer_json)
    assert set(table[: len(forced)]) == forced
    tokenizer = load_tokenizer(tokenizer_json)
    frequent = tokenizer.encode("Het", add_special_tokens=False).ids[0]
    assert table.index(frequent) < len(forced) + 40


def test_added_and_byte_tokens_are_always_included(tokenizer_json, tmp_path) -> None:
    text = tmp_path / "one.txt"
    text.write_text("zzz")
    forced = _forced_ids(tokenizer_json)
    assert len(forced) == 256 + len(ADDED)

    build_frspec_table(
        _request(tokenizer_json, text, tmp_path, rows=len(forced) + 1, fill_from="none")
    )

    table = set(np.load(tmp_path / "out" / "table.npy").tolist())
    assert forced <= table
    with pytest.raises(FrspecBuildError, match="at least"):
        build_frspec_table(_request(tokenizer_json, text, tmp_path, rows=len(forced) - 1))


def test_fill_from_builtin_is_the_family_default(tokenizer_json, corpus, tmp_path, monkeypatch) -> None:
    fingerprint = tokenizer_vocab_fingerprint(tokenizer_json)
    builtin = tmp_path / "builtin.npy"
    np.save(builtin, np.arange(400, dtype=np.int32)[::-1].copy())
    monkeypatch.setattr(frspec_build, "_BUILTIN_VOCABS", {"tiny-code": builtin})
    monkeypatch.setattr(frspec_build, "BUILTIN_TOKENIZER_FINGERPRINTS", {"tiny-code": fingerprint})

    report = build_frspec_table(_request(tokenizer_json, corpus, tmp_path, fill_from=None))

    assert report["fill_from"] == "builtin:tiny-code"
    assert report["rows"] == 300
    assert report["filled"] == 300 - report["forced_ids"] - report["from_input"]
    table = set(np.load(tmp_path / "out" / "table.npy").tolist())
    assert 399 in table  # the builtin's first entries fill the remaining rows


def test_fill_from_builtin_refuses_another_tokenizer(tokenizer_json, corpus, tmp_path, monkeypatch) -> None:
    builtin = tmp_path / "builtin.npy"
    np.save(builtin, np.arange(10, dtype=np.int32))
    monkeypatch.setattr(frspec_build, "_BUILTIN_VOCABS", {"tiny-code": builtin})
    monkeypatch.setattr(frspec_build, "BUILTIN_TOKENIZER_FINGERPRINTS", {"tiny-code": "0" * 64})

    with pytest.raises(FrspecBuildError, match="different tokenizer"):
        build_frspec_table(_request(tokenizer_json, corpus, tmp_path, fill_from="builtin:tiny-code"))


def test_fill_none_writes_only_forced_and_seen_ids(tokenizer_json, corpus, tmp_path) -> None:
    report = build_frspec_table(_request(tokenizer_json, corpus, tmp_path, rows=349, fill_from="none"))

    assert report["filled"] == 0
    assert report["rows"] == report["forced_ids"] + report["from_input"] < 349


def test_exclude_tokens_drops_single_token_strings(tokenizer_json, corpus, tmp_path) -> None:
    tokenizer = load_tokenizer(tokenizer_json)
    word = "concept"
    word_ids = set(tokenizer.encode(" " + word, add_special_tokens=False).ids)
    assert len(word_ids) == 1

    report = build_frspec_table(
        _request(
            tokenizer_json,
            corpus,
            tmp_path,
            exclude_strings=[word, "<|im_end|>", "no such multi token phrase qqq"],
        )
    )

    table = set(np.load(tmp_path / "out" / "table.npy").tolist())
    assert not word_ids & table
    assert tokenizer.token_to_id("<|im_end|>") in table  # added tokens stay in
    assert report["exclude_unmatched"] == ["no such multi token phrase qqq"]


def test_holdout_reports_coverage_on_unseen_files(tokenizer_json, tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "many"
    corpus.mkdir()
    for index in range(20):
        (corpus / f"doc{index:02d}.txt").write_text(TRAIN_TEXT[index % 3] + f" item {index}")
    builtin = tmp_path / "builtin.npy"
    np.save(builtin, np.arange(260, dtype=np.int32))
    fingerprint = tokenizer_vocab_fingerprint(tokenizer_json)
    monkeypatch.setattr(frspec_build, "_BUILTIN_VOCABS", {"tiny-code": builtin})
    monkeypatch.setattr(frspec_build, "BUILTIN_TOKENIZER_FINGERPRINTS", {"tiny-code": fingerprint})

    report = build_frspec_table(_request(tokenizer_json, corpus, tmp_path, holdout=0.3))

    held = report["holdout"]
    assert held["files"] > 0
    assert held["files"] + report["files_read"] == 20
    assert held["tokens"] > 0
    assert 0.0 < held["coverage_builtin"] < held["coverage_table"] <= 1.0


def test_split_holdout_is_deterministic_and_disjoint(tmp_path) -> None:
    files = [tmp_path / f"f{index}.md" for index in range(50)]

    kept, held = split_holdout(files, 0.2)

    assert (kept, held) == split_holdout(list(files), 0.2)
    assert not set(kept) & set(held)
    assert len(kept) + len(held) == 50
    assert 0 < len(held) < 50
    with pytest.raises(FrspecBuildError):
        split_holdout(files, 1.0)


def test_rank_table_orders_by_count_then_id() -> None:
    counts = np.zeros(10, dtype=np.int64)
    counts[[7, 3, 5]] = [4, 9, 4]

    table = rank_table(counts, rows=6, forced=[0], fill=[9, 8, 7, 1], exclude=[5])

    assert table.ids == [0, 3, 7, 9, 8, 1]
    assert (table.forced, table.from_input, table.filled) == (1, 2, 3)


def test_session_jsonl_reads_assistant_messages_only() -> None:
    records = [
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "tool output"}]}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "plan"},
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "input": {"cmd": "ls"}},
        ]}},
        {"type": "summary", "summary": "not model text", "uuid": "abc"},
    ]

    assert list(json_texts(records, ("assistant",))) == ["plan", "answer", '{"cmd": "ls"}']
    assert "tool output" in list(json_texts(records, ("assistant", "user")))


def test_openai_conversation_and_plain_json() -> None:
    conversation = {"messages": [
        {"role": "system", "content": "be brief"},
        {"role": "assistant", "content": "hello", "tool_calls": [
            {"function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
    ]}

    assert list(json_texts(conversation, ("assistant",))) == ["hello", "{\"a\": 1}"]
    assert list(json_texts({"title": "x", "items": [{"body": "y"}, 3]}, ("assistant",))) == ["x", "y"]


def test_file_size_cap_truncates(tokenizer_json, tmp_path) -> None:
    big = tmp_path / "big.txt"
    big.write_text("word " * 1000)

    report = build_frspec_table(
        _request(tokenizer_json, big, tmp_path, max_file_bytes=100, fill_from="ids")
    )

    assert report["files_truncated"] == 1
    assert report["tokens"] < 60


def test_loader_round_trip_and_truncation_rules(tokenizer_json, corpus, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MTPLX_FRSPEC_N", raising=False)
    ascending = tmp_path / "asc.npy"
    ranked = tmp_path / "ranked.npy"
    build_frspec_table(_request(tokenizer_json, corpus, tmp_path, out=ascending))
    build_frspec_table(_request(tokenizer_json, corpus, tmp_path, out=ranked, order="ranked"))

    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(ascending))
    assert load_frspec_ids() == np.load(ascending).tolist()
    monkeypatch.setenv("MTPLX_FRSPEC_N", "100")
    assert load_frspec_ids() is None  # a row-sorted table cannot be cut

    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(ranked))
    assert load_frspec_ids() == np.load(ranked).tolist()[:100]


def test_loader_detects_tokenizer_mismatch(tokenizer_json, corpus, tmp_path, monkeypatch) -> None:
    table = tmp_path / "table.npy"
    build_frspec_table(_request(tokenizer_json, corpus, tmp_path, out=table))
    other = _make_tokenizer(tmp_path / "other-pack", extra_text="völlig andere Wörter hier")
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(table))

    assert frspec_tokenizer_mismatch(tokenizer_json.parent) is None
    mismatch = frspec_tokenizer_mismatch(other.parent)
    assert mismatch is not None and mismatch["reason"] == "tokenizer_mismatch"
    assert frspec_tokenizer_mismatch(None) is None


def test_install_refuses_a_table_built_for_another_tokenizer(tmp_path, monkeypatch) -> None:
    import mlx.core as mx
    from mlx import nn

    from mtplx.frspec_draft import install_frspec_draft_head

    linear = nn.Linear(64, 8, bias=False)
    head = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)
    mx.eval(head.parameters())
    table = tmp_path / "table.npy"
    np.save(table, np.asarray([1, 6], dtype=np.int32))
    vocab_sidecar_path(table).write_text(json.dumps({"tokenizer_vocab_sha256": "0" * 64}))
    model_dir = tmp_path / "pack"
    _make_tokenizer(model_dir)
    monkeypatch.setenv("MTPLX_FRSPEC_VOCAB", str(table))
    monkeypatch.delenv("MTPLX_FRSPEC_N", raising=False)
    monkeypatch.delenv("MTPLX_FRSPEC_LEGACY", raising=False)
    text = SimpleNamespace(_mtplx_draft_lm_head=head)

    report = install_frspec_draft_head(text, model_path=model_dir)

    assert report["installed"] is False
    assert report["reason"] == "tokenizer_mismatch"
    assert install_frspec_draft_head(text)["installed"] is True  # no model path: no check


def test_fingerprint_ignores_pre_tokenizer_settings(tokenizer_json, tmp_path) -> None:
    payload = json.loads(tokenizer_json.read_text())
    payload["pre_tokenizer"] = None
    copy = tmp_path / "copy" / "tokenizer.json"
    copy.parent.mkdir()
    copy.write_text(json.dumps(payload, indent=1))

    assert tokenizer_vocab_fingerprint(copy) == tokenizer_vocab_fingerprint(tokenizer_json)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def library(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.delenv("MTPLX_MODEL_DIRS", raising=False)
    monkeypatch.delenv("MTPLX_MODEL_DIR", raising=False)
    root = tmp_path / "models"
    _make_tokenizer(root / "Org--Family-A-Speed")
    _make_tokenizer(root / "Org--Family-A-Quality")
    _make_tokenizer(root / "Other-Model", extra_text="völlig andere Wörter hier")
    (root / "No-Tokenizer").mkdir()
    return root


def _run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


def test_cli_list_models_groups_packs_by_tokenizer(library, capsys) -> None:
    assert _run(["frspec", "build", "--list-models", "--cache-dir", str(library)]) == 0

    out = capsys.readouterr().out
    groups = out.split("  tokenizer ")[1:]
    assert len(groups) == 2
    assert "Org/Family-A-Speed" in groups[0] and "Org/Family-A-Quality" in groups[0]
    assert "Other-Model" in groups[1]
    assert "1 more without a tokenizer.json" in out


def test_cli_unknown_model_lists_usable_packs(library, tmp_path, capsys) -> None:
    code = _run([
        "frspec", "build", "--model", "no-such-model", "--cache-dir", str(library),
        "--input", str(tmp_path), "--out", str(tmp_path / "t.npy"),
    ])

    assert code == 2
    err = capsys.readouterr().err
    assert "no-such-model" in err
    assert "Org/Family-A-Speed" in err and "Other-Model" in err
    assert not (tmp_path / "t.npy").exists()


def test_cli_builds_from_a_name_in_the_model_listing(library, corpus, tmp_path, capsys) -> None:
    out = tmp_path / "t.npy"

    code = _run([
        "frspec", "build", "--model", "Other-Model", "--cache-dir", str(library),
        "--input", str(corpus), "--out", str(out), "--rows", "300", "--fill-from", "ids",
    ])

    assert code == 0
    printed = capsys.readouterr().out
    assert "300 ids" in printed
    assert "derived from that input" in printed
    fingerprint = tokenizer_vocab_fingerprint(library / "Other-Model" / "tokenizer.json")
    assert json.loads(vocab_sidecar_path(out).read_text())["tokenizer_vocab_sha256"] == fingerprint
