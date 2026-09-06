#!/usr/bin/env python3
"""Run an agent CLI with an honest exit receipt and bounded process cleanup.

Example: python scripts/run_harness_check.py --out outputs/pi --timeout 1200 -- pi -p "Build a game"
This bounds the QA process, never the model's output or reasoning budget.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", required=True, type=float)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.timeout <= 0:
        parser.error("a command and positive timeout are required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    timed_out = False
    with args.out.with_suffix(".log").open("x") as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            code = 124
        finally:
            # A client may exit while a tool child remains alive. Reap only
            # this run's process group; never match unrelated processes by name.
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    receipt = dict(command=command, started_at_s=started,
                   elapsed_s=time.time() - started, exit_code=code,
                   timed_out=timed_out, process_succeeded=code == 0)
    args.out.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
