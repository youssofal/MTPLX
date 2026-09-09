#!/usr/bin/env python3
"""HumanEval / MBPP correctness gate against an OpenAI-compatible server.

Every other benchmark in this repo measures throughput or acceptance. This one
measures whether the output is still *correct*, which is what makes a
quantization arm comparable: a decode win that silently costs pass@1 is not a
win.

This is the driver half. It gets completions from a served model and hands them
to :mod:`mtplx.benchmarks.code_eval`, which owns loading, program assembly,
sandboxed execution, and pass@k. Splitting it this way keeps the scoring half
importable with no network and no model, and keeps this half free of any
opinion about how a candidate is executed.

Two endpoints are supported. ``--endpoint chat`` posts to
``/v1/chat/completions`` and expects an instruct-tuned model to answer with a
fenced code block. ``--endpoint completions`` posts to ``/v1/completions`` with
the raw HumanEval prompt, which is the base-model continuation protocol the
original Codex paper used; the completion is then a bare function body.

Executing model-generated code is the entire point of the benchmark and also
its only real hazard, so it never happens implicitly: pass
``--allow-code-execution`` or the run refuses before it sends a single request.

Determinism is the default (temperature 0, n=1, fixed seed) and every request
parameter is recorded in the report, so two arms scored on different days are
still comparable.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtplx.benchmarks.code_eval import (  # noqa: E402
    CodeTask,
    TaskResult,
    extract_code,
    load_tasks,
    pass_at_k,
    run_candidate,
    summarize,
)

SCHEMA = "mtplx.code_eval_gate/1"

# Retry these and nothing else. A 400 or a 404 is a misconfigured run and
# retrying it just burns wall clock three times before reporting the same
# thing; a 503 or a dropped socket is the server being busy, which is exactly
# what a long eval run should ride out.
_TRANSIENT_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

_HUMANEVAL_SYSTEM = (
    "You are a Python programming assistant. Complete the function you are "
    "given. Reply with a single fenced Python code block containing the "
    "complete function definition -- the signature, the body, and any imports "
    "it needs. Do not include tests, example usage, or explanation."
)

_MBPP_SYSTEM = (
    "You are a Python programming assistant. Write a Python function that "
    "solves the task and satisfies the given tests. Reply with a single "
    "fenced Python code block containing the complete function definition and "
    "any imports it needs. Use exactly the function name the tests call. Do "
    "not include the tests or any explanation."
)

# Base-model continuation has no stop token of its own: the model happily keeps
# emitting the *next* top-level definition after finishing ours, which then
# shadows or breaks the candidate. These are the standard HumanEval stops.
_COMPLETION_STOPS = ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]


class GateHTTPError(RuntimeError):
    """An HTTP-layer failure, carrying the status when there was one."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def transient(self) -> bool:
        # No status at all means the socket died or timed out before a
        # response existed -- always worth another attempt.
        return self.status is None or self.status in _TRANSIENT_STATUS


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _dataset_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mtplx_version() -> str | None:
    """Read the version without importing the package root (which pulls MLX)."""

    try:
        from mtplx.version import __version__

        return str(__version__)
    except Exception:
        return None


def _client_request_id(task_id: str, sample: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "-", task_id).strip("-")[:48]
    suffix = time.time_ns() % 1_000_000_000
    return f"codeeval-{safe or 'task'}-s{sample}-{os.getpid()}-{suffix}"


# --------------------------------------------------------------------------
# prompt / payload construction
# --------------------------------------------------------------------------


def build_messages(task: CodeTask) -> list[dict[str, str]]:
    """Chat messages for one task."""

    if task.suite == "humaneval":
        system = _HUMANEVAL_SYSTEM
        user = f"Complete this function:\n\n```python\n{task.prompt.rstrip()}\n```"
    else:
        system = _MBPP_SYSTEM
        user = task.prompt.rstrip()
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_payload(
    args: argparse.Namespace,
    task: CodeTask,
    *,
    sample: int,
    client_request_id: str,
) -> dict[str, Any]:
    """The exact request body for one sample of one task."""

    payload: dict[str, Any] = {
        "model": args.model,
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "stream": False,
        "metadata": {
            "mtplx_benchmark": "code_eval_gate",
            "mtplx_request_id": client_request_id,
            "task_id": task.task_id,
            "suite": task.suite,
            "sample": sample,
        },
    }
    if args.endpoint == "chat":
        payload["messages"] = build_messages(task)
    else:
        # Base-model continuation: the raw prompt IS the request, verbatim.
        payload["prompt"] = task.prompt
        if task.suite == "humaneval":
            payload["stop"] = list(_COMPLETION_STOPS)
    if args.top_p is not None:
        payload["top_p"] = float(args.top_p)
    if args.seed is not None:
        # Distinct seed per sample, or n>1 just draws the same completion n
        # times at a nonzero temperature and pass@k is a lie.
        payload["seed"] = int(args.seed) + sample
    # Optional server-extension passthrough (default empty == byte-identical to
    # the classic payload). Used to carry e.g. enable_thinking:false or top_k
    # for engines that read them off the request body. Applied last so it is
    # explicit in the recorded request args.
    for key, value in (args.extra_body or {}).items():
        payload[key] = value
    return payload


def endpoint_url(base_url: str, endpoint: str) -> str:
    tail = "/v1/chat/completions" if endpoint == "chat" else "/v1/completions"
    return f"{base_url.rstrip('/')}{tail}"


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: str | None,
    client_request_id: str,
) -> dict[str, Any]:
    """POST one request. This is the seam the tests replace."""

    headers = {
        "Content-Type": "application/json",
        "X-MTPLX-Request-ID": client_request_id,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise GateHTTPError(f"HTTP {exc.code}: {body}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise GateHTTPError(f"connection failed: {exc.reason!r}") from exc
    except (TimeoutError, OSError) as exc:
        raise GateHTTPError(f"transport failed: {exc!r}") from exc
    except json.JSONDecodeError as exc:
        raise GateHTTPError(f"malformed JSON response: {exc}") from exc


def request_with_retries(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_s: float,
    api_key: str | None,
    client_request_id: str,
    retries: int,
    backoff_s: float = 0.5,
    sleep=time.sleep,
) -> tuple[dict[str, Any], int]:
    """Post with bounded retries. Returns the response and the attempt count.

    Looks ``_post_json`` up on the module rather than closing over it so a test
    (or a caller) can swap the transport by patching the module attribute.
    """

    attempts = 0
    last: Exception | None = None
    for attempt in range(max(0, int(retries)) + 1):
        attempts += 1
        try:
            return (
                globals()["_post_json"](
                    url,
                    payload,
                    timeout_s=timeout_s,
                    api_key=api_key,
                    client_request_id=client_request_id,
                ),
                attempts,
            )
        except GateHTTPError as exc:
            last = exc
            if not exc.transient or attempt >= int(retries):
                break
            # Jittered backoff: a whole thread pool retrying a 503 in lockstep
            # is just the same thundering herd one second later.
            sleep(backoff_s * (2**attempt) * (0.5 + random.random() / 2))
        except Exception as exc:  # noqa: BLE001 - a bad transport must not kill the run
            last = exc
            break
    raise last if last is not None else GateHTTPError("request failed")


def extract_completion_text(response: dict[str, Any], endpoint: str) -> tuple[str, str | None]:
    """Pull the generated text and finish_reason out of either response shape."""

    choices = response.get("choices") or []
    if not choices:
        return "", None
    choice = choices[0] or {}
    finish_reason = choice.get("finish_reason")
    if endpoint == "chat":
        message = choice.get("message") or {}
        text = message.get("content")
    else:
        text = choice.get("text")
    return (text if isinstance(text, str) else ""), finish_reason


# --------------------------------------------------------------------------
# generate
# --------------------------------------------------------------------------


def generate_one(config: dict[str, Any], task: CodeTask, sample: int) -> dict[str, Any]:
    """Fetch one completion. Never raises: a dead request becomes an error row.

    One task failing to generate must not take the other 163 down with it, so
    the failure is recorded and the run continues. It is still counted as a
    non-pass in the summary -- a run that could not ask the model is not a run
    that scored zero by accident.
    """

    args = argparse.Namespace(**config["request_args"])
    client_request_id = _client_request_id(task.task_id, sample)
    payload = build_payload(
        args, task, sample=sample, client_request_id=client_request_id
    )
    url = endpoint_url(config["base_url"], args.endpoint)
    started = time.perf_counter()
    try:
        response, attempts = request_with_retries(
            url,
            payload,
            timeout_s=float(config["timeout_s"]),
            api_key=config.get("api_key"),
            client_request_id=client_request_id,
            retries=int(config["retries"]),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "task_id": task.task_id,
            "sample": sample,
            "completion": None,
            "error": repr(exc),
            "attempts": int(config["retries"]) + 1,
            "finish_reason": None,
            "request_seconds": time.perf_counter() - started,
            "usage": None,
            "mtplx_stats": {},
        }
    text, finish_reason = extract_completion_text(response, args.endpoint)
    return {
        "task_id": task.task_id,
        "sample": sample,
        "completion": text,
        "error": None,
        "attempts": attempts,
        "finish_reason": finish_reason,
        "request_seconds": time.perf_counter() - started,
        "usage": response.get("usage"),
        "mtplx_stats": response.get("mtplx_stats") or {},
    }


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def score_one(
    task: CodeTask,
    generation: dict[str, Any],
    *,
    allow_execution: bool,
    timeout_s: float,
) -> TaskResult:
    """Score one generation, mapping a failed request onto its own status."""

    if generation.get("error") is not None:
        return TaskResult(
            task.task_id, False, "request_error", str(generation["error"])[:400], 0.0
        )
    completion = generation.get("completion") or ""
    try:
        return run_candidate(
            task,
            completion,
            allow_execution=allow_execution,
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - a broken candidate is a row, not a crash
        return TaskResult(task.task_id, False, "error", repr(exc)[:400], 0.0)


def pass_at_k_report(
    results: Sequence[TaskResult], *, n: int, ks: Sequence[int]
) -> dict[str, float]:
    """Per-task pass@k, averaged over tasks.

    ``summarize`` deliberately is not used for this: its ``n>1`` branch treats
    the whole run as one n-sample draw, which is only correct for a single
    task. pass@k is defined per problem and then averaged.
    """

    by_task: dict[str, list[TaskResult]] = {}
    for result in results:
        by_task.setdefault(result.task_id, []).append(result)
    report: dict[str, float] = {}
    for k in ks:
        if k > n or k <= 0:
            continue
        scores = [
            pass_at_k(len(rows), sum(1 for r in rows if r.passed), k)
            for rows in by_task.values()
            if len(rows) >= k
        ]
        if scores:
            report[f"pass@{k}"] = sum(scores) / len(scores)
    return report


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def _request_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": args.model,
        "endpoint": args.endpoint,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "extra_body": dict(getattr(args, "extra_body", {}) or {}),
    }


def build_report(
    *,
    args: argparse.Namespace,
    tasks: Sequence[CodeTask],
    generations: Sequence[dict[str, Any]],
    results: Sequence[TaskResult],
    wall_s: float,
    dataset_sha256: str,
) -> dict[str, Any]:
    by_id = {task.task_id: task for task in tasks}
    rows: list[dict[str, Any]] = []
    for result, generation in zip(results, generations):
        rows.append(
            {
                "task_id": result.task_id,
                "sample": generation["sample"],
                "suite": by_id[result.task_id].suite
                if result.task_id in by_id
                else args.suite,
                "status": result.status,
                "passed": result.passed,
                "seconds": result.seconds,
                "request_seconds": generation["request_seconds"],
                "attempts": generation["attempts"],
                "finish_reason": generation["finish_reason"],
                "detail": result.detail[:400],
                "error": generation["error"],
                "usage": generation["usage"],
            }
        )
    ks = sorted({1, int(args.n)})
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "summary": summarize(list(results)),
        "rows": rows,
        "provenance": {
            "base_url": args.base_url.rstrip("/"),
            "model": args.model,
            "suite": args.suite,
            "endpoint": args.endpoint,
            "dataset_path": str(args.dataset_path),
            "dataset_sha256": dataset_sha256,
            "tasks": len(tasks),
            "samples_per_task": int(args.n),
            "completions": len(generations),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mtplx_version": _mtplx_version(),
            "python": sys.version.split()[0],
            "wall_s": wall_s,
        },
        "params": {
            **_request_args(args),
            "n": int(args.n),
            "limit": args.limit,
            "workers": int(args.workers),
            "score_workers": int(args.score_workers),
            "retries": int(args.retries),
            "timeout_s": float(args.timeout_s),
            "execution_timeout_s": float(args.execution_timeout_s),
            "allow_code_execution": bool(args.allow_code_execution),
        },
        "request_errors": sum(1 for r in results if r.status == "request_error"),
    }
    if int(args.n) > 1:
        report["pass_at_k"] = pass_at_k_report(results, n=int(args.n), ks=ks)
    return report


def run(args: argparse.Namespace) -> int:
    if not args.allow_code_execution:
        print(
            "REFUSING: scoring HumanEval/MBPP means executing code this model "
            "wrote, in a subprocess on this machine. Pass --allow-code-execution "
            "to acknowledge that. Nothing was requested and nothing was run.",
            file=sys.stderr,
        )
        return 2

    dataset_path = Path(args.dataset_path)
    tasks = load_tasks(args.suite, dataset_path)
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2

    config = {
        "base_url": args.base_url.rstrip("/"),
        "api_key": args.api_key,
        "timeout_s": args.timeout_s,
        "retries": args.retries,
        "request_args": _request_args(args),
    }

    units = [(task, sample) for task in tasks for sample in range(int(args.n))]
    started = time.perf_counter()

    # Generation is network-bound and scoring is process-bound, so they run as
    # two phases with independently bounded pools. Interleaving them would put
    # `workers` sockets and `workers` subprocesses in flight at once, which is
    # how a 164-task run turns into a fork bomb on a laptop.
    generations: list[dict[str, Any]] = [None] * len(units)  # type: ignore[list-item]
    with futures.ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        pending = {
            pool.submit(generate_one, config, task, sample): index
            for index, (task, sample) in enumerate(units)
        }
        done = 0
        for future in futures.as_completed(pending):
            index = pending[future]
            generations[index] = future.result()
            done += 1
            if args.progress:
                _print_progress("generate", done, len(units), generations[index])

    results: list[TaskResult] = [None] * len(units)  # type: ignore[list-item]
    with futures.ThreadPoolExecutor(max_workers=int(args.score_workers)) as pool:
        pending = {
            pool.submit(
                score_one,
                task,
                generations[index],
                allow_execution=True,
                timeout_s=float(args.execution_timeout_s),
            ): index
            for index, (task, sample) in enumerate(units)
        }
        done = 0
        for future in futures.as_completed(pending):
            index = pending[future]
            results[index] = future.result()
            done += 1
            if args.progress:
                _print_progress("score", done, len(units), results[index])

    if getattr(args, "save_completions", None):
        _write_completions_sidecar(
            Path(args.save_completions), units, generations, args.suite
        )

    report = build_report(
        args=args,
        tasks=tasks,
        generations=generations,
        results=results,
        wall_s=time.perf_counter() - started,
        dataset_sha256=_dataset_sha256(dataset_path),
    )

    output_path = _resolve_output_path(args)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        report.setdefault("artifacts", {})["report_json"] = str(output_path)

    print(
        json.dumps(
            {
                "schema": report["schema"],
                "summary": report["summary"],
                "pass_at_k": report.get("pass_at_k"),
                "request_errors": report["request_errors"],
                "provenance": report["provenance"],
                "artifacts": report.get("artifacts", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )

    if report["request_errors"]:
        return 1
    if args.min_pass_rate is not None:
        rate = report["summary"].get("pass@1", 0.0)
        if rate < float(args.min_pass_rate):
            print(
                f"FAIL: pass@1 {rate:.4f} < --min-pass-rate {args.min_pass_rate}",
                file=sys.stderr,
            )
            return 1
    return 0


def _write_completions_sidecar(
    path: Path,
    units: Sequence[tuple[CodeTask, int]],
    generations: Sequence[dict[str, Any]],
    suite: str,
) -> None:
    """Persist one JSON-Lines record per (task_id, sample).

    Each record carries the raw model ``completion`` and the ``solution`` that
    the gate actually executed (``extract_code`` applied identically to the
    scoring path), so a later extended-test rescore (HumanEval+) can run offline
    on the exact same completions without re-hitting the model. Writing this file
    does not change the report or any pass rate.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for (task, sample), gen in zip(units, generations):
            gen = gen or {}
            completion = gen.get("completion") or ""
            try:
                solution = extract_code(completion, suite=suite)
            except Exception:  # noqa: BLE001 - a bad completion is still a record
                solution = ""
            handle.write(
                json.dumps(
                    {
                        "task_id": task.task_id,
                        "sample": int(sample),
                        "completion": completion,
                        "solution": solution,
                        "finish_reason": gen.get("finish_reason"),
                        "error": gen.get("error"),
                    }
                )
                + "\n"
            )


def _resolve_output_path(args: argparse.Namespace) -> Path | None:
    if args.output_json:
        return Path(args.output_json)
    if args.output_dir:
        return Path(args.output_dir) / f"code-eval-{args.suite}-{args.endpoint}.json"
    return None


def _print_progress(phase: str, done: int, total: int, item: Any) -> None:
    if isinstance(item, TaskResult):
        detail = {"task_id": item.task_id, "status": item.status}
    else:
        detail = {
            "task_id": item.get("task_id"),
            "error": item.get("error"),
            "finish": item.get("finish_reason"),
        }
    print(
        json.dumps({"phase": phase, "done": done, "total": total, **detail}),
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:18183")
    parser.add_argument("--api-key", default=os.environ.get("MTPLX_API_KEY"))
    parser.add_argument("--model", default="mtplx-qwen36-27b-optimized-speed")
    parser.add_argument("--suite", choices=("humaneval", "mbpp"), default="humaneval")
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="Path to HumanEval.jsonl, mbpp.jsonl, or sanitized-mbpp.json.",
    )
    parser.add_argument(
        "--endpoint",
        choices=("chat", "completions"),
        default="chat",
        help="chat = /v1/chat/completions (instruct); "
        "completions = /v1/completions (base-model continuation).",
    )
    parser.add_argument("--limit", type=int, help="Score only the first N tasks.")
    parser.add_argument(
        "--workers", type=int, default=4, help="Concurrent in-flight requests."
    )
    parser.add_argument(
        "--score-workers",
        type=int,
        default=4,
        help="Concurrent scoring subprocesses. Each one is a real process.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 by default so two arms are comparable. Raise it for pass@k.",
    )
    parser.add_argument("--top-p", type=float)
    parser.add_argument(
        "--extra-body",
        action="append",
        default=[],
        metavar="KEY=JSONVALUE",
        help="Extra request-body field(s) to send, repeatable. VALUE is parsed "
        "as JSON (falls back to a string). E.g. --extra-body enable_thinking=false "
        "--extra-body top_k=20. Default empty == unchanged payload.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Samples per task. >1 enables pass@k and needs a nonzero temperature.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--execution-timeout-s",
        type=float,
        default=15.0,
        help="Wall-clock limit per candidate subprocess.",
    )
    parser.add_argument("--output-json", help="Write the full report here.")
    parser.add_argument("--output-dir", help="Write the report into this directory.")
    parser.add_argument(
        "--save-completions",
        help=(
            "Also write a completions sidecar (JSON Lines) here: one object per "
            "(task_id, sample) with the raw completion and the extracted solution. "
            "This is what lets a later HumanEval+ / extended-test rescore run "
            "offline (CPU only) on the SAME completions. Omit == unchanged "
            "behaviour; the report JSON is byte-for-byte identical either way."
        ),
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        help="Exit nonzero if pass@1 falls below this.",
    )
    parser.add_argument("--progress", action="store_true", default=False)
    parser.add_argument(
        "--allow-code-execution",
        action="store_true",
        default=False,
        help="Required. Acknowledges that scoring executes model-written code.",
    )
    args = parser.parse_args(argv)
    if int(args.n) < 1:
        parser.error("--n must be >= 1")
    if int(args.n) > 1 and float(args.temperature) == 0.0:
        parser.error(
            "--n > 1 with --temperature 0 draws the same completion n times; "
            "pass@k would be meaningless. Raise --temperature."
        )
    # Parse repeated --extra-body KEY=JSONVALUE into a dict (JSON value, string
    # fallback). Empty list -> {} -> payload unchanged.
    extra_body: dict[str, Any] = {}
    for item in args.extra_body or []:
        if "=" not in item:
            parser.error(f"--extra-body entry is not KEY=VALUE: {item!r}")
        key, _, raw = item.partition("=")
        try:
            extra_body[key] = json.loads(raw)
        except json.JSONDecodeError:
            extra_body[key] = raw
    args.extra_body = extra_body
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
