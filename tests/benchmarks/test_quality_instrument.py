"""The instrument end to end, against stand-in servers and tokenizers: no
model, no GPU. The suite's own checks are proven in test_quality_suite."""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mtplx.benchmarks.quality import cli
from mtplx.benchmarks.quality.corpus import entry_from_row
from mtplx.benchmarks.quality.divergence import position_divergence, summarize
from mtplx.benchmarks.quality.episode import run_episode
from mtplx.benchmarks.quality.report import Arm, build_report, markdown
from mtplx.benchmarks.quality.scoring import compare
from mtplx.benchmarks.quality.suite import Suite, load_suite

SUITE = load_suite()
TASK = {t.id: t for t in SUITE.tasks}


# --------------------------------------------------------------------------
# divergence
# --------------------------------------------------------------------------


def _logs(probabilities: dict[str, float]) -> dict[str, float]:
    return {k: math.log(v) for k, v in probabilities.items()}


def test_identical_distributions_do_not_diverge():
    top = _logs({"a": 0.6, "b": 0.3, "c": 0.05})
    d = position_divergence(top, dict(top), reference_logprob=math.log(0.6), arm_logprob=math.log(0.6))
    assert d.kl == pytest.approx(0.0, abs=1e-12) and d.top1_agrees and d.nll_delta == 0.0
    assert d.reference_rest_mass == pytest.approx(0.05)


def test_partition_kl_matches_the_analytic_value():
    p = {"a": 0.5, "b": 0.3, "x": 0.15}
    q = {"a": 0.4, "b": 0.4, "y": 0.1}
    d = position_divergence(_logs(p), _logs(q), reference_logprob=math.log(0.5), arm_logprob=math.log(0.4))
    # Shared cells a and b, then one rest cell: 0.2 for the reference, 0.2 for the arm.
    expected = 0.5 * math.log(0.5 / 0.4) + 0.3 * math.log(0.3 / 0.4) + 0.2 * math.log(0.2 / 0.2)
    assert d.kl == pytest.approx(expected)
    assert d.nll_delta == pytest.approx(math.log(0.5) - math.log(0.4))


def test_collapsed_pieces_fall_into_the_rest_cell():
    p = {"a": 0.7, "�": 0.2}
    q = {"a": 0.7, "�": 0.05}
    d = position_divergence(_logs(p), _logs(q), reference_logprob=0.0, arm_logprob=0.0)
    assert d.kl == pytest.approx(0.0, abs=1e-12)  # 0.3 rest on both sides


def test_an_empty_arm_rest_is_floored_and_counted():
    p = {"a": 0.5}
    q = {"a": 1.0}
    d = position_divergence(_logs(p), _logs(q), reference_logprob=0.0, arm_logprob=0.0)
    assert d.rest_floored and math.isfinite(d.kl) and d.kl > 5


def test_summary_pools_positions_and_blocks_entries():
    same = position_divergence({"a": 0.0}, {"a": 0.0}, reference_logprob=0.0, arm_logprob=0.0)
    far = position_divergence(_logs({"a": 0.9}), _logs({"a": 0.1}), reference_logprob=0.0, arm_logprob=-1.0)
    summary = summarize([[same, same, same], [far]])
    assert summary.positions == 4 and summary.entries == 2
    assert summary.mean_kl == pytest.approx(far.kl / 4)
    assert summary.top1_agreement == 1.0  # one key each side: the argmax agrees
    assert summary.mean_kl_stderr > 0


# --------------------------------------------------------------------------
# episodes against a scripted server
# --------------------------------------------------------------------------


class ScriptedServer:
    """Replies in order; records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    def chat(self, body):
        self.sent.append(json.loads(json.dumps(body)))
        message, finish, completion, reasoning = self.replies.pop(0)
        return {
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": completion,
                      "completion_tokens_details": {"reasoning_tokens": reasoning}},
            "mtplx_stats": {"accepted_by_depth": [8, 5, 2], "drafted_by_depth": [10, 10, 10], "verify_calls": 10,
                            "effective_temperature": 0.6, "effective_top_p": 0.95, "effective_top_k": 20},
        }


def _call(call_id, name, **arguments):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def test_agent_episode_fixes_the_project(tmp_path):
    task = TASK["agent-stats"]
    replies = [
        ({"role": "assistant", "content": "", "reasoning_content": "look first",
          "tool_calls": [_call("1", "read_file", path="stats.py")]}, "tool_calls", 50, 30),
        ({"role": "assistant", "content": "", "reasoning_content": "fix both",
          "tool_calls": [_call("2", "write_file", path="stats.py", content=task.reference_files["stats.py"]),
                         _call("3", "run_tests")]}, "tool_calls", 400, 100),
        ({"role": "assistant", "content": "Fixed median and mode."}, "stop", 20, 10),
    ]
    server = ScriptedServer(replies)
    row = run_episode(server, SUITE, task, seed=7, scratch=tmp_path, request_extras={"depth": 3})
    assert row["passed"] and row["outcome"] == "finished" and row["requests"] == 3
    assert row["tokens"] == {"completion": 470, "reasoning": 140, "answer": 330}
    # The tool results went back: the file the model read, then the passing test run.
    third = server.sent[2]["messages"]
    assert "def median" in third[3]["content"]
    assert third[-1]["role"] == "tool" and third[-1]["content"].startswith("exit status 0")
    assert third[2]["reasoning_content"] == "look first"  # thinking is sent back
    assert all(b["seed"] == 7 and b["depth"] == 3 and b["max_tokens"] == SUITE.budget_tokens for b in server.sent)


def test_a_reply_that_hits_the_budget_is_a_runaway(tmp_path):
    server = ScriptedServer([({"role": "assistant", "content": "wait, wait, wait"}, "length", SUITE.budget_tokens, 0)])
    row = run_episode(server, SUITE, TASK["code-lru-cache"], seed=1, scratch=tmp_path, request_extras={})
    assert row["outcome"] == "runaway" and not row["passed"] and row["tokens"]["completion"] == SUITE.budget_tokens


def test_an_agent_that_never_stops_is_unfinished(tmp_path):
    small = Suite(name="t", budget_tokens=1000, max_agent_turns=2, tasks=SUITE.tasks)
    loop = ({"role": "assistant", "content": "", "tool_calls": [_call("1", "list_files")]}, "tool_calls", 10, 0)
    row = run_episode(ScriptedServer([loop, loop, loop]), small, TASK["agent-pager"], seed=1, scratch=tmp_path,
                      request_extras={})
    assert row["outcome"] == "unfinished" and row["requests"] == 2


def test_image_tasks_send_the_picture(tmp_path):
    server = ScriptedServer([({"role": "assistant", "content": "5"}, "stop", 3, 0)])
    row = run_episode(server, SUITE, TASK["image-count-squares"], seed=1, scratch=tmp_path, request_extras={})
    sent = server.sent[0]["messages"][-1]["content"]
    assert sent[0]["image_url"]["url"].startswith("data:image/png;base64,iVBOR")
    assert row["passed"] and row["messages"][0]["content"][0]["image_url"]["url"] == "(task image)"


# --------------------------------------------------------------------------
# corpus spans, with a character-level stand-in tokenizer
# --------------------------------------------------------------------------


class CharTokenizer:
    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m.get('reasoning_content', '')}|{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_corpus_marks_only_what_the_model_wrote():
    row = {"task": "t", "seed": 1, "kind": "agent", "outcome": "finished", "tools": [], "messages": [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "reasoning_content": "hm", "tool_calls": [_call("1", "run_tests")]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]}
    entry = entry_from_row(CharTokenizer(), row, max_tokens=8192)
    text = "".join(chr(i) for i in entry["token_ids"])
    written = ["".join(text[a:b]) for a, b in entry["spans"]]
    assert written == ["hm|</assistant>", "|done</assistant>"]
    assert entry["scored_tokens"] == sum(len(w) for w in written)
    assert entry_from_row(CharTokenizer(), {**row, "kind": "image"}, max_tokens=8192) is None


# --------------------------------------------------------------------------
# scores and the report
# --------------------------------------------------------------------------


def _score_row(entry_id, positions):
    return {"id": entry_id, "tokens_digest": "d", "positions": positions}


def test_compare_pairs_positions_by_entry():
    ref = [_score_row("e", [[5, -0.1, {"a": -0.1, "b": -2.4}], [6, -0.5, {"c": -0.5}]])]
    arm = [_score_row("e", [[5, -0.3, {"a": -0.3, "b": -1.4}], [6, -0.5, {"c": -0.5}]])]
    summary, per_entry = compare(ref, arm)
    assert summary.positions == 2 and summary.mean_kl > 0 and per_entry["e"] == pytest.approx(summary.mean_kl)
    with pytest.raises(ValueError):
        compare(ref, [{**arm[0], "tokens_digest": "other"}])


def _task_row(task, seed, passed, completion, reasoning, outcome="finished"):
    return {"task": task, "seed": seed, "kind": "code", "passed": passed, "outcome": outcome,
            "tokens": {"completion": completion, "reasoning": reasoning, "answer": completion - reasoning},
            "turns": [{"verify_calls": 10, "accepted_by_depth": [8, 4, 1], "drafted_by_depth": [10, 10, 10],
                       "effective_temperature": 0.6, "effective_top_p": 0.95, "effective_top_k": 20}]}


def test_report_pairs_arms_and_builds_the_table(tmp_path):
    base = Arm("old", [_task_row("a", 1, True, 1000, 800), _task_row("b", 1, False, 3000, 2900)], [])
    new = Arm("new", [_task_row("a", 1, True, 800, 600), _task_row("b", 1, True, 1500, 1300)], [])
    report = build_report([base, new], base="old", reference=None)
    pairing = report["arms"]["new"]["paired_with_base"]
    assert pairing["token_ratio"] == pytest.approx(2300 / 4000) and pairing["gained"] == 1 and pairing["lost"] == 0
    tasks = report["arms"]["new"]["tasks"]
    assert tasks["pass_rate"] == 1.0 and tasks["acceptance"]["by_step"] == [0.8, 0.4, 0.1]
    assert tasks["acceptance"]["conditional_by_step"] == pytest.approx([0.8, 0.5, 0.25])
    assert tasks["acceptance"]["accepted_per_round"] == pytest.approx(1.3)
    table = markdown(report)
    assert "| tokens per passed task | 4,000 | 1,150 |" in table
    assert "| rounds accepting drafts 1 to 2 | 0.400 | 0.400 |" in table
    assert "| step 3 accepted after step 2 | 0.250 | 0.250 |" in table


# --------------------------------------------------------------------------
# the command line against a stand-in HTTP server
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send({"data": [{"id": "stand-in"}]})
        else:
            self._send({"model": "stand-in", "model_path": "/models/x", "depth": 3,
                        "degradation": {"compiled_verify": {"mode": "on", "permanent_eager": False}}})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/completions":
            ids = body["prompt"]
            self._send({"choices": [{"logprobs": {"token_ids": ids, "token_logprobs": [None] + [-0.5] * (len(ids) - 1),
                                                  "top_logprobs": [{}] + [{"x": -0.5, "y": -1.5}] * (len(ids) - 1)}}]})
            return
        code = TASK["code-to-roman"].reference
        self._send({"choices": [{"message": {"role": "assistant", "content": f"```python\n{code}\n```",
                                             "reasoning_content": "numerals"}, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": 120, "completion_tokens_details": {"reasoning_tokens": 40}},
                    "mtplx_stats": {"accepted_by_depth": [3], "drafted_by_depth": [4], "verify_calls": 4,
                                    "demotions": {"verify_eager": 1}}})


@pytest.fixture()
def stand_in_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_run_score_and_report_from_the_command_line(stand_in_server, tmp_path, capsys):
    out = tmp_path / "arm"
    # Past its deadline a run starts nothing, and says what it left.
    assert cli.main(["run", stand_in_server, str(out), "--tasks", "code-to-roman", "--deadline-s", "1e-9"]) == 0
    assert not (out / "tasks.jsonl").exists()
    assert "deadline reached: 1 of 1 left" in capsys.readouterr().out
    assert cli.main(["run", stand_in_server, str(out), "--tasks", "code-to-roman", "--seeds", "1,2",
                     "--depth", "3", "--sampler", "0.6,0.95,20", "--label", "x"]) == 0
    rows = [json.loads(line) for line in (out / "tasks.jsonl").read_text().splitlines()]
    assert [r["seed"] for r in rows] == [1, 2] and all(r["passed"] for r in rows)
    assert rows[0]["request_extras"] == {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "depth": 3}
    # A second run skips what is already there.
    assert cli.main(["run", stand_in_server, str(out), "--tasks", "code-to-roman", "--seeds", "1,2"]) == 0
    assert len((out / "tasks.jsonl").read_text().splitlines()) == 2
    # Each request's route is in its row; the server's lane counters beside them.
    assert rows[0]["turns"][0]["demotions"] == {"verify_eager": 1}
    routes = sorted(out.glob("routes-*.json"))
    assert routes and json.loads(routes[-1].read_text())["compiled_verify"]["mode"] == "on"

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"id": "e1", "token_ids": [1, 2, 3, 4], "spans": [[2, 4]], "scored_tokens": 2}) + "\n")
    assert cli.main(["score", stand_in_server, str(out), "--corpus", str(corpus), "--top-k", "2"]) == 0
    assert cli.main(["report", "--arm", f"ref={out}", "--arm", f"same={out}", "--reference", "ref"]) == 0
    table = capsys.readouterr().out
    assert "| KL vs ref, mean (nats) | - | 0.0000 |" in table
    assert "| tasks passed | 2/2 | 2/2 |" in table
    assert "| requests that left a fast lane | 2/2; verify_eager 2 | 2/2; verify_eager 2 |" in table
    # A report over some kinds only: the rows' own kind keeps them, another leaves none.
    assert cli.main(["report", "--arm", f"ref={out}", "--kinds", "code"]) == 0
    assert "| tasks passed | 2/2 |" in capsys.readouterr().out
    assert cli.main(["report", "--arm", f"ref={out}", "--kinds", "agent"]) == 0
    assert "| tasks passed | - |" in capsys.readouterr().out
