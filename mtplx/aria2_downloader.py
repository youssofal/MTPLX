"""Small, dependency-free aria2c runner for Hugging Face downloads."""

from __future__ import annotations

import http.client
import json
import secrets
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path


class Aria2DownloadError(RuntimeError):
    """Raised when aria2c cannot complete a requested transfer."""


@dataclass(frozen=True)
class Aria2Download:
    """One aria2 download and the metadata needed to verify it."""

    url: str
    output: Path
    headers: tuple[str, ...] = ()
    expected_size: int | None = None
    sha256: str | None = None
    display_name: str | None = None


Aria2ProgressCallback = Callable[[int, float, str | None], None]


def _single_line(value: str, *, label: str) -> str:
    if "\n" in value or "\r" in value:
        raise Aria2DownloadError(f"aria2 {label} must fit on one line")
    return value


def _build_input(
    downloads: Iterable[Aria2Download],
    gids: dict[Path, str],
) -> str:
    lines: list[str] = []
    for download in downloads:
        url = _single_line(download.url, label="URL")
        output = download.output.resolve()
        directory = _single_line(str(output.parent), label="destination")
        filename = _single_line(output.name, label="filename")
        lines.extend(
            [
                url,
                f"  gid={gids[download.output]}",
                f"  dir={directory}",
                f"  out={filename}",
                "  continue=true",
                "  auto-file-renaming=false",
                "  allow-overwrite=false",
            ]
        )
        if download.sha256:
            checksum = _single_line(download.sha256, label="checksum")
            lines.append(f"  checksum=sha-256={checksum}")
        for header in download.headers:
            lines.append(f"  header={_single_line(header, label='header')}")
    return "\n".join(lines) + "\n"


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _command(
    executable: str,
    *,
    rpc_port: int,
    rpc_secret: str,
    connections: int,
    concurrent_downloads: int,
) -> list[str]:
    return [
        executable,
        "--input-file=-",
        "--enable-rpc=true",
        "--rpc-listen-all=false",
        f"--rpc-listen-port={rpc_port}",
        f"--rpc-secret={rpc_secret}",
        f"--max-concurrent-downloads={max(1, concurrent_downloads)}",
        f"--max-connection-per-server={max(1, connections)}",
        f"--split={max(1, connections)}",
        "--min-split-size=1M",
        "--file-allocation=none",
        "--max-tries=5",
        "--retry-wait=2",
        "--connect-timeout=30",
        "--timeout=60",
        "--summary-interval=0",
        "--console-log-level=warn",
        "--download-result=hide",
    ]


def _rpc_call(
    *,
    port: int,
    secret: str,
    method: str,
    params: list[object],
    timeout: float = 0.5,
) -> object:
    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "mtplx",
            "method": f"aria2.{method}",
            "params": [f"token:{secret}", *params],
        }
    )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        connection.request(
            "POST",
            "/jsonrpc",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        body = json.loads(response.read().decode("utf-8"))
    finally:
        connection.close()
    if response.status != 200 or "error" in body:
        detail = body.get("error", {}).get("message", f"HTTP {response.status}")
        raise Aria2DownloadError(f"aria2 RPC failed: {detail}")
    return body.get("result")


def _status_bytes(
    *,
    port: int,
    secret: str,
    gids: dict[Path, str],
    downloads: list[Aria2Download],
    initial_sizes: dict[Path, int],
) -> tuple[int, float, str | None, bool, str | None, int]:
    completed = 0
    rate = 0.0
    active_name: str | None = None
    fastest = -1.0
    terminal = 0
    seen = 0
    errors: list[str] = []
    by_output = {download.output: download for download in downloads}
    for output, gid in gids.items():
        try:
            result = _rpc_call(
                port=port,
                secret=secret,
                method="tellStatus",
                params=[
                    gid,
                    [
                        "status",
                        "completedLength",
                        "downloadSpeed",
                        "errorCode",
                        "errorMessage",
                    ],
                ],
            )
        except (Aria2DownloadError, OSError, ValueError):
            completed += initial_sizes.get(output, 0)
            continue
        if not isinstance(result, dict):
            completed += initial_sizes.get(output, 0)
            continue
        seen += 1
        item_completed = max(
            initial_sizes.get(output, 0),
            int(result.get("completedLength") or 0),
        )
        item_rate = float(result.get("downloadSpeed") or 0)
        completed += item_completed
        rate += item_rate
        status = result.get("status")
        if status in {"complete", "error", "removed"}:
            terminal += 1
        if status in {"error", "removed"}:
            message = result.get("errorMessage") or result.get("errorCode") or status
            errors.append(f"{by_output[output].display_name or output.name}: {message}")
        if status == "active" and item_rate > fastest:
            fastest = item_rate
            active_name = by_output[output].display_name
    all_terminal = seen == len(gids) and terminal == len(gids)
    return completed, rate, active_name, all_terminal, "; ".join(errors) or None, seen


def run_aria2_downloads(
    downloads: Iterable[Aria2Download],
    *,
    executable: str,
    progress_callback: Aria2ProgressCallback | None = None,
    progress_interval_s: float = 0.4,
    connections: int = 16,
    concurrent_downloads: int = 4,
) -> None:
    """Run one aria2c process and report byte-accurate aggregate progress."""

    jobs = list(downloads)
    if not jobs:
        return
    for job in jobs:
        job.output.parent.mkdir(parents=True, exist_ok=True)
    gids = {job.output: secrets.token_hex(8) for job in jobs}
    initial_sizes = {
        job.output: job.output.stat().st_size if job.output.is_file() else 0
        for job in jobs
    }
    rpc_port = _reserve_loopback_port()
    rpc_secret = secrets.token_urlsafe(24)
    argv = _command(
        executable,
        rpc_port=rpc_port,
        rpc_secret=rpc_secret,
        connections=connections,
        concurrent_downloads=concurrent_downloads,
    )
    job_input = _build_input(jobs, gids)

    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as error_log:
        process = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=error_log,
            text=True,
        )
        try:
            assert process.stdin is not None
            process.stdin.write(job_input)
            process.stdin.close()
            interval = max(0.1, float(progress_interval_s))
            next_progress = time.monotonic() + interval
            status_started_at = time.monotonic()
            download_error: str | None = None
            while process.poll() is None:
                now = time.monotonic()
                if now >= next_progress:
                    (
                        completed,
                        rate,
                        active_name,
                        all_terminal,
                        status_error,
                        seen,
                    ) = _status_bytes(
                        port=rpc_port,
                        secret=rpc_secret,
                        gids=gids,
                        downloads=jobs,
                        initial_sizes=initial_sizes,
                    )
                    if progress_callback is not None:
                        progress_callback(completed, rate, active_name)
                    if all_terminal:
                        download_error = status_error
                        _rpc_call(
                            port=rpc_port,
                            secret=rpc_secret,
                            method="shutdown",
                            params=[],
                        )
                        break
                    if seen == 0 and now - status_started_at >= 5.0:
                        download_error = "aria2c did not register any download jobs"
                        _rpc_call(
                            port=rpc_port,
                            secret=rpc_secret,
                            method="shutdown",
                            params=[],
                        )
                        break
                    next_progress = now + interval
                time.sleep(min(0.1, interval))
            return_code = process.wait(timeout=5)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            raise

        if return_code != 0 or download_error:
            error_log.seek(0)
            detail = error_log.read()[-4096:].strip()
            for job in jobs:
                for header in job.headers:
                    detail = detail.replace(header, "<redacted header>")
            reasons = [reason for reason in (download_error, detail) if reason]
            suffix = f": {'; '.join(reasons)}" if reasons else ""
            raise Aria2DownloadError(f"aria2c exited with status {return_code}{suffix}")

    if progress_callback is not None:
        completed = sum(
            job.output.stat().st_size if job.output.is_file() else 0 for job in jobs
        )
        progress_callback(completed, 0.0, None)
