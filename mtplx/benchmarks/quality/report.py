"""Tables from rows: one arm per column, every figure computed here.

Rows are found under each arm's folder (``tasks*.jsonl`` and
``scores*.jsonl.gz``, at any depth, so a measurement window's boot folder
works as it is). A base arm, when given, is the one the others are paired
with: the same task and seed on both sides, so the token ratio and the pass
flips compare like with like.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mtplx.benchmarks.quality.scoring import compare, read_jsonl


@dataclass
class Arm:
    name: str
    tasks: list[dict[str, Any]]
    scores: list[dict[str, Any]]

    @classmethod
    def load(cls, name: str, folder: Path) -> "Arm":
        tasks = [row for path in sorted(folder.rglob("tasks*.jsonl")) for row in read_jsonl(path)]
        scores = [row for path in sorted(folder.rglob("scores*.jsonl.gz")) for row in read_jsonl(path)]
        if not tasks and not scores:
            raise ValueError(f"{name}: no rows under {folder}")
        return cls(name, tasks, scores)


def _acceptance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted: list[int] = []
    drafted: list[int] = []
    rounds = 0
    for row in rows:
        for turn in row["turns"]:
            rounds += int(turn.get("verify_calls") or 0)
            for depth, (a, d) in enumerate(zip(turn.get("accepted_by_depth") or [], turn.get("drafted_by_depth") or [])):
                if depth == len(accepted):
                    accepted.append(0)
                    drafted.append(0)
                accepted[depth] += int(a)
                drafted[depth] += int(d)
    # by_step is the share of rounds whose first d drafts were all accepted;
    # conditional_by_step is the chance step d is accepted once step d-1 was.
    conditional = []
    for step, (a, d) in enumerate(zip(accepted, drafted)):
        before = accepted[step - 1] if step else d
        conditional.append(a / before if before else None)
    return {
        "by_step": [a / d if d else None for a, d in zip(accepted, drafted)],
        "conditional_by_step": conditional,
        "accepted_per_round": sum(accepted) / rounds if rounds else None,
        "rounds": rounds,
    }


def task_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [r for r in rows if r["outcome"] != "error"]
    passed = [r for r in finished if r["passed"]]
    completion = [r["tokens"]["completion"] for r in finished]
    effective = {
        (t.get("effective_temperature"), t.get("effective_top_p"), t.get("effective_top_k"))
        for r in finished
        for t in r["turns"]
    }
    by_kind: dict[str, list[int]] = {}
    for r in finished:
        by_kind.setdefault(r["kind"], [0, 0])
        by_kind[r["kind"]][0] += r["passed"]
        by_kind[r["kind"]][1] += 1
    turns = [t for r in finished for t in r["turns"]]
    demotions: dict[str, int] = {}
    for turn in turns:
        for kind, count in (turn.get("demotions") or {}).items():
            demotions[kind] = demotions.get(kind, 0) + int(count)
    return {
        "rows": len(rows),
        "errors": len(rows) - len(finished),
        "passed": len(passed),
        "pass_rate": len(passed) / len(finished) if finished else None,
        "by_kind": {k: f"{p}/{n}" for k, (p, n) in sorted(by_kind.items())},
        "runaways": sum(r["outcome"] == "runaway" for r in finished),
        "unfinished": sum(r["outcome"] == "unfinished" for r in finished),
        "mean_tokens": statistics.fmean(completion) if completion else None,
        "median_tokens": statistics.median(completion) if completion else None,
        "mean_thinking": statistics.fmean(r["tokens"]["reasoning"] for r in finished) if finished else None,
        "mean_answer": statistics.fmean(r["tokens"]["answer"] for r in finished) if finished else None,
        "tokens_per_pass": sum(completion) / len(passed) if passed else None,
        "acceptance": _acceptance(finished),
        "sampler_seen": sorted(str(e) for e in effective),
        "requests": len(turns),
        "demoted_requests": sum(bool(t.get("demotions")) for t in turns),
        "demotions": dict(sorted(demotions.items())),
    }


def paired(base: list[dict[str, Any]], arm: list[dict[str, Any]]) -> dict[str, Any]:
    """The same task and seed on both arms: token ratio and pass flips."""

    key = lambda r: (r["task"], r["seed"])  # noqa: E731
    base_by = {key(r): r for r in base if r["outcome"] != "error"}
    pairs = [(base_by[key(r)], r) for r in arm if r["outcome"] != "error" and key(r) in base_by]
    if not pairs:
        return {"pairs": 0}
    base_tokens = sum(b["tokens"]["completion"] for b, _ in pairs)
    arm_tokens = sum(a["tokens"]["completion"] for _, a in pairs)
    log_ratios = [
        math.log(a["tokens"]["completion"] / b["tokens"]["completion"])
        for b, a in pairs
        if a["tokens"]["completion"] and b["tokens"]["completion"]
    ]
    return {
        "pairs": len(pairs),
        "token_ratio": arm_tokens / base_tokens if base_tokens else None,
        "geomean_token_ratio": math.exp(statistics.fmean(log_ratios)) if log_ratios else None,
        "gained": sum(a["passed"] and not b["passed"] for b, a in pairs),
        "lost": sum(b["passed"] and not a["passed"] for b, a in pairs),
    }


def build_report(arms: list[Arm], *, base: str | None, reference: str | None) -> dict[str, Any]:
    names = [a.name for a in arms]
    for wanted in (base, reference):
        if wanted and wanted not in names:
            raise ValueError(f"{wanted!r} is not one of the arms {names}")
    by_name = {a.name: a for a in arms}
    report: dict[str, Any] = {"arms": {}}
    for arm in arms:
        entry: dict[str, Any] = {}
        if arm.tasks:
            entry["tasks"] = task_summary(arm.tasks)
            if base and arm.name != base and by_name[base].tasks:
                entry["paired_with_base"] = paired(by_name[base].tasks, arm.tasks)
        if reference and arm.name != reference and arm.scores and by_name[reference].scores:
            summary, _ = compare(by_name[reference].scores, arm.scores)
            entry["divergence"] = summary.as_dict()
        report["arms"][arm.name] = entry
    report["base"], report["reference"] = base, reference
    return report


def _cell(value: Any, fmt: str = "{:.3g}") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return fmt.format(value)
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    arms = report["arms"]
    names = list(arms)
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]

    def row(label: str, pick: Any, fmt: str = "{:.3g}") -> None:
        values = []
        for name in names:
            try:
                values.append(_cell(pick(arms[name]), fmt))
            except (KeyError, TypeError, IndexError):
                values.append("-")
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    row("tasks passed", lambda a: f"{a['tasks']['passed']}/{a['tasks']['rows'] - a['tasks']['errors']}")
    row("pass rate", lambda a: a["tasks"]["pass_rate"], "{:.1%}")
    row("passed by kind", lambda a: " ".join(f"{k} {v}" for k, v in a["tasks"]["by_kind"].items()))
    row("runaways / unfinished / errors", lambda a: f"{a['tasks']['runaways']} / {a['tasks']['unfinished']} / {a['tasks']['errors']}")
    row("tokens written per task (mean)", lambda a: a["tasks"]["mean_tokens"], "{:,.0f}")
    row("  of which thinking", lambda a: a["tasks"]["mean_thinking"], "{:,.0f}")
    row("  of which answer", lambda a: a["tasks"]["mean_answer"], "{:,.0f}")
    row("tokens per passed task", lambda a: a["tasks"]["tokens_per_pass"], "{:,.0f}")
    if report.get("base"):
        row(f"tokens vs {report['base']} (same tasks and seeds)", lambda a: a["paired_with_base"]["token_ratio"], "{:.3f}")
        row(f"passes gained / lost vs {report['base']}", lambda a: f"{a['paired_with_base']['gained']} / {a['paired_with_base']['lost']}")
    for step in range(3):
        row(f"rounds accepting drafts 1 to {step + 1}", lambda a, s=step: a["tasks"]["acceptance"]["by_step"][s], "{:.3f}")
    for step in range(1, 3):
        row(f"step {step + 1} accepted after step {step}",
            lambda a, s=step: a["tasks"]["acceptance"]["conditional_by_step"][s], "{:.3f}")
    row("accepted drafts per round", lambda a: a["tasks"]["acceptance"]["accepted_per_round"], "{:.3f}")
    if report.get("reference"):
        ref = report["reference"]
        row(f"KL vs {ref}, mean (nats)", lambda a: a["divergence"]["mean_kl"], "{:.4f}")
        row("  standard error", lambda a: a["divergence"]["mean_kl_stderr"], "{:.4f}")
        row("  99th percentile", lambda a: a["divergence"]["p99_kl"], "{:.3f}")
        row(f"top-1 agreement with {ref}", lambda a: a["divergence"]["top1_agreement"], "{:.2%}")
        row("NLL increase per token (nats)", lambda a: a["divergence"]["mean_nll_delta"], "{:.4f}")
        row("scored positions", lambda a: a["divergence"]["positions"], "{:,}")
    row("sampler seen", lambda a: "; ".join(a["tasks"]["sampler_seen"]))
    row("requests that left a fast lane", lambda a: f"{a['tasks']['demoted_requests']}/{a['tasks']['requests']}"
        + "".join(f"; {k} {n}" for k, n in a["tasks"]["demotions"].items()))
    return "\n".join(lines) + "\n"
