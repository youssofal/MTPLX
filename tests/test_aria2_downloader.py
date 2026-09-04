from __future__ import annotations

import hashlib
import re
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from mtplx import hf_loader
from mtplx.aria2_downloader import (
    Aria2Download,
    Aria2DownloadError,
    _build_input,
    _command,
    run_aria2_downloads,
)
from mtplx.cli import build_parser


def test_aria2_job_input_carries_resume_checksum_and_secret_header(tmp_path: Path):
    output = tmp_path / "weights.safetensors.incomplete"
    download = Aria2Download(
        url="https://huggingface.co/org/repo/resolve/commit/weights.safetensors",
        output=output,
        headers=("authorization: Bearer hf_secret",),
        sha256="a" * 64,
    )

    payload = _build_input([download], {output: "0123456789abcdef"})
    argv = _command(
        "/opt/homebrew/bin/aria2c",
        rpc_port=12345,
        rpc_secret="rpc-secret",
        connections=16,
        concurrent_downloads=4,
    )

    assert "continue=true" in payload
    assert "checksum=sha-256=" + "a" * 64 in payload
    assert "header=authorization: Bearer hf_secret" in payload
    assert "weights.safetensors.incomplete" in payload
    assert all("hf_secret" not in argument for argument in argv)


def test_aria2_job_input_rejects_header_newlines(tmp_path: Path):
    output = tmp_path / "weights.incomplete"
    download = Aria2Download(
        url="https://example.invalid/weights",
        output=output,
        headers=("authorization: safe\nX-Injected: unsafe",),
    )

    with pytest.raises(Aria2DownloadError, match="header must fit on one line"):
        _build_input([download], {output: "0123456789abcdef"})


@pytest.mark.parametrize(
    ("requested", "found", "expected"),
    [
        ("python", "/opt/homebrew/bin/aria2c", ("python", None)),
        ("auto", None, ("python", None)),
        (
            "auto",
            "/opt/homebrew/bin/aria2c",
            ("aria2", "/opt/homebrew/bin/aria2c"),
        ),
        (
            "aria2",
            "/opt/homebrew/bin/aria2c",
            ("aria2", "/opt/homebrew/bin/aria2c"),
        ),
    ],
)
def test_resolve_download_backend(monkeypatch, requested, found, expected):
    monkeypatch.setattr(hf_loader.shutil, "which", lambda _name: found)

    assert hf_loader._resolve_download_backend(requested) == expected


def test_resolve_download_backend_requires_installed_aria2(monkeypatch):
    monkeypatch.setattr(hf_loader.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="aria2c is not installed"):
        hf_loader._resolve_download_backend("aria2")


def test_aria2_backend_lands_verified_partials_and_emits_progress(
    monkeypatch, tmp_path: Path
):
    content = b"parallel weights"
    sha256 = hashlib.sha256(content).hexdigest()
    repo_file = hf_loader.RepoFile(
        path="weights/model.safetensors",
        size_bytes=len(content),
        sha256=sha256,
    )
    destination = tmp_path / "model"
    destination.mkdir()
    events: list[dict] = []
    captured: dict[str, object] = {}

    def fake_run(downloads, **kwargs):
        jobs = list(downloads)
        captured["jobs"] = jobs
        captured["kwargs"] = kwargs
        for job in jobs:
            job.output.parent.mkdir(parents=True, exist_ok=True)
            job.output.write_bytes(content)
        kwargs["progress_callback"](len(content), 4096.0, repo_file.path)

    monkeypatch.setattr(
        "mtplx.aria2_downloader.run_aria2_downloads",
        fake_run,
    )

    hf_loader._download_repo_files_with_aria2(
        [repo_file],
        executable="/opt/homebrew/bin/aria2c",
        repo_id="org/repo",
        revision="commit",
        destination=destination,
        hf_hub_url=lambda **_kwargs: "https://example.invalid/weights",
        build_hf_headers=lambda token=None: {
            "authorization": f"Bearer {token}",
        },
        token="hf_secret",
        callback=events.append,
        total_bytes=len(content),
        started_at=time.monotonic(),
        progress_interval_s=0.1,
        last_emit_at=time.monotonic(),
        last_emit_size=0,
    )

    jobs = captured["jobs"]
    assert isinstance(jobs, list)
    assert jobs[0].sha256 == sha256
    assert jobs[0].headers == ("authorization: Bearer hf_secret",)
    assert (destination / repo_file.path).read_bytes() == content
    assert not (destination / f"{repo_file.path}.incomplete").exists()
    assert any(event["event"] == "progress" for event in events)


def test_pull_parser_accepts_aria2_backend():
    args = build_parser().parse_args(
        ["pull", "org/repo", "--download-backend", "aria2"]
    )

    assert args.download_backend == "aria2"


@pytest.mark.skipif(shutil.which("aria2c") is None, reason="aria2c is not installed")
def test_aria2_runner_resumes_against_range_server(tmp_path: Path):
    content = bytes(range(256)) * 8192
    range_requests: list[str] = []

    class RangeHandler(BaseHTTPRequestHandler):
        def do_HEAD(self):
            self._send(include_body=False)

        def do_GET(self):
            self._send(include_body=True)

        def _send(self, *, include_body: bool):
            start = 0
            end = len(content) - 1
            requested = self.headers.get("Range")
            if requested:
                range_requests.append(requested)
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                assert match is not None
                start = int(match.group(1))
                if match.group(2):
                    end = min(end, int(match.group(2)))
                if start >= len(content):
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{len(content)}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            else:
                self.send_response(200)
            body = content[start : end + 1]
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = tmp_path / "weights.incomplete"
    output.write_bytes(content[: 512 * 1024])
    progress: list[tuple[int, float, str | None]] = []
    try:
        run_aria2_downloads(
            [
                Aria2Download(
                    url=f"http://127.0.0.1:{server.server_port}/weights",
                    output=output,
                    expected_size=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                    display_name="weights",
                )
            ],
            executable=shutil.which("aria2c") or "aria2c",
            progress_callback=lambda *event: progress.append(event),
            progress_interval_s=0.1,
            connections=4,
            concurrent_downloads=1,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert output.read_bytes() == content
    assert range_requests
    assert progress[-1][0] == len(content)
