import json

from mtplx.agent_cli import cmd_graph
from mtplx.cli import build_parser


def test_graph_cli_lists_runs_with_api_parity(monkeypatch, capsys):
    captured = {}

    def request(args, method, path, body=None):
        captured.update(method=method, path=path, body=body)
        return {"runs": []}

    monkeypatch.setattr("mtplx.agent_cli._request", request)
    args = build_parser().parse_args(
        [
            "graph",
            "runs",
            "--graph-id",
            "verification-graph",
            "--workspace-id",
            "project",
            "--limit",
            "25",
        ]
    )

    assert args.func is cmd_graph
    assert args.func(args) == 0
    assert captured == {
        "method": "GET",
        "path": (
            "/v1/mtplx/graph-runs?graph_id=verification-graph"
            "&workspace_id=project&limit=25"
        ),
        "body": None,
    }
    assert json.loads(capsys.readouterr().out) == {"runs": []}


def test_graph_cli_lists_run_approvals_with_status_filter(monkeypatch, capsys):
    captured = {}

    def request(args, method, path, body=None):
        captured.update(method=method, path=path, body=body)
        return {"approvals": []}

    monkeypatch.setattr("mtplx.agent_cli._request", request)
    args = build_parser().parse_args(
        [
            "graph",
            "approvals",
            "graph-run-1",
            "--status",
            "pending",
            "--limit",
            "10",
        ]
    )

    assert args.func is cmd_graph
    assert args.func(args) == 0
    assert captured == {
        "method": "GET",
        "path": (
            "/v1/mtplx/graph-runs/graph-run-1/approvals"
            "?status=pending&limit=10"
        ),
        "body": None,
    }
    assert json.loads(capsys.readouterr().out) == {"approvals": []}
