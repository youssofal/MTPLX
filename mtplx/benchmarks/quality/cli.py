"""``python -m mtplx.benchmarks.quality``: run the suite, build and score the
corpus, and print the tables.

    run     BASE_URL OUT_DIR   the tasks against one server   -> tasks.jsonl
    corpus  ROWS_DIR...        a reference run's conversations -> corpus.jsonl
    score   BASE_URL OUT_DIR   the corpus on one server        -> scores.jsonl.gz
    report  --arm NAME=DIR...  the table, from rows only

``run`` and ``score`` append and skip what their output already holds, so a
run that was cut short continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mtplx.benchmarks.quality.client import Server
from mtplx.benchmarks.quality.corpus import build_corpus
from mtplx.benchmarks.quality.episode import append_row, run_episode
from mtplx.benchmarks.quality.report import Arm, build_report, markdown
from mtplx.benchmarks.quality.scoring import read_jsonl, score_entry, write_scores
from mtplx.benchmarks.quality.suite import DEFAULT_SUITE_DIR, load_suite

_PACKAGE = Path(__file__).resolve().parent


def _instrument_identity() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(_PACKAGE), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    dirty = git("status", "--porcelain", "--", ".", "../prompts/quality")
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(dirty), "python": sys.version.split()[0]}


def _list(value: str | None) -> list[str] | None:
    return [part.strip() for part in value.split(",") if part.strip()] if value else None


def _done(path: Path, key: Any) -> set[Any]:
    return {key(row) for row in read_jsonl(path)} if path.exists() else set()


def _write_routes(server: Server, out: Path) -> None:
    # The server's lane counters after this command's requests (cumulative
    # since the server started), kept beside the rows they explain.
    (out / f"routes-{int(time.time())}.json").write_text(json.dumps(server.routes(), indent=1))


def cmd_run(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    server = Server(args.base_url, model=args.model)
    extras: dict[str, Any] = {}
    if args.sampler:
        temperature, top_p, top_k = (float(v) for v in args.sampler.split(","))
        extras.update(temperature=temperature, top_p=top_p, top_k=int(top_k))
    if args.depth is not None:
        extras["depth"] = args.depth
    identity = {"server": server.identity(), "instrument": _instrument_identity(), "suite": suite.name,
                "request_extras": extras, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / f"identity-{int(time.time())}.json").write_text(json.dumps(identity, indent=1))

    seeds = [int(s) for s in _list(args.seeds) or ["1"]]
    jobs = [(task, seed) for task in suite.select(_list(args.tasks), _list(args.kinds)) for seed in seeds]
    index, count = (int(v) for v in args.shard.split("/"))
    jobs = jobs[index - 1 :: count]
    rows_path = out / "tasks.jsonl"
    done = _done(rows_path, lambda r: (r["task"], r["seed"]))
    started = time.monotonic()
    pending = [(task, seed) for task, seed in jobs if (task.id, seed) not in done]
    for position, (task, seed) in enumerate(pending):
        if args.deadline_s and time.monotonic() - started > args.deadline_s:
            # A caller with a hard time limit reruns the same command for the rest.
            print(f"deadline reached: {len(pending) - position} of {len(pending)} left for the next run", flush=True)
            break
        row = run_episode(server, suite, task, seed=seed, scratch=out / "scratch", request_extras=extras)
        row["arm"] = args.label
        append_row(rows_path, row)
        print(f"{task.id} s{seed}: {row['status']} {row['tokens']['completion']} tokens {row['wall_s']:.0f}s", flush=True)
    _write_routes(server, out)
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    seeds = {int(s) for s in _list(args.seeds) or []}
    rows = [
        row
        for folder in args.rows
        for path in sorted(Path(folder).rglob("tasks*.jsonl"))
        for row in read_jsonl(path)
        if not seeds or row["seed"] in seeds
    ]
    entries = build_corpus(Path(args.tokenizer), rows, max_tokens=args.max_tokens)
    with open(args.out, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    scored = sum(e["scored_tokens"] for e in entries)
    print(f"{len(entries)} entries, {scored:,} scored tokens -> {args.out}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    server = Server(args.base_url, model=args.model)
    (out / f"identity-score-{int(time.time())}.json").write_text(
        json.dumps({"server": server.identity(), "instrument": _instrument_identity(), "corpus": args.corpus}, indent=1)
    )
    path = out / "scores.jsonl.gz"
    done = _done(path, lambda r: r["id"])
    for entry in read_jsonl(Path(args.corpus)):
        if entry["id"] in done:
            continue
        row = score_entry(server, entry, top_k=args.top_k)
        write_scores(path, [row])
        print(f"{entry['id']}: {len(row['positions'])} positions", flush=True)
    _write_routes(server, out)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    arms = []
    kinds = _list(args.kinds)
    for spec in args.arm:
        name, _, folder = spec.partition("=")
        arm = Arm.load(name, Path(folder))
        if kinds:
            arm.tasks = [row for row in arm.tasks if row["kind"] in kinds]
        arms.append(arm)
    report = build_report(arms, base=args.base, reference=args.reference)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    sys.stdout.write(markdown(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mtplx.benchmarks.quality", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the tasks against one server")
    run.add_argument("base_url")
    run.add_argument("out_dir")
    run.add_argument("--suite", default=str(DEFAULT_SUITE_DIR))
    run.add_argument("--model", help="model id to request (default: the server's first)")
    run.add_argument("--label", default="", help="arm name written into each row")
    run.add_argument("--tasks", help="comma-separated task ids (default: all)")
    run.add_argument("--kinds", help="comma-separated kinds (default: all)")
    run.add_argument("--seeds", default="1", help="comma-separated sampler seeds")
    run.add_argument("--shard", default="1/1", help="I/N: every N-th job starting at the I-th")
    run.add_argument("--depth", type=int, help="draft depth to request (default: the server's)")
    run.add_argument("--sampler", help="TEMPERATURE,TOP_P,TOP_K to send (default: the server's)")
    run.add_argument("--deadline-s", type=float, help="start no new task after this many seconds")
    run.set_defaults(func=cmd_run)

    corpus = sub.add_parser("corpus", help="build the divergence corpus from a reference run")
    corpus.add_argument("rows", nargs="+", help="folders holding the reference run's tasks.jsonl")
    corpus.add_argument("--tokenizer", required=True, help="the model folder whose tokenizer and template to use")
    corpus.add_argument("--out", required=True)
    corpus.add_argument("--seeds", help="only these seeds (default: all)")
    corpus.add_argument("--max-tokens", type=int, default=8192, help="the server's scoring limit per entry")
    corpus.set_defaults(func=cmd_corpus)

    score = sub.add_parser("score", help="score the corpus on one server")
    score.add_argument("base_url")
    score.add_argument("out_dir")
    score.add_argument("--corpus", required=True)
    score.add_argument("--model")
    score.add_argument("--top-k", type=int, default=64)
    score.set_defaults(func=cmd_score)

    report = sub.add_parser("report", help="print the table from rows")
    report.add_argument("--arm", action="append", required=True, help="NAME=FOLDER, in column order")
    report.add_argument("--base", help="arm the others are paired with (tokens, pass flips)")
    report.add_argument("--reference", help="arm the divergence is measured from")
    report.add_argument("--kinds", help="only these task kinds, comma-separated (default: all)")
    report.add_argument("--json", help="also write the full report here")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return int(args.func(args))
