#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from mtplx.modelopt_nvfp4 import convert_modelopt_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    parser.add_argument("--source-repo")
    parser.add_argument("--source-sha")
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    def progress(event):
        if event["event"] == "shard_complete":
            print(f"[{event['completed']}/{event['total']}] {event['filename']}", flush=True)

    report = convert_modelopt_checkpoint(
        args.source,
        args.output,
        group_size=args.group_size,
        source_repo=args.source_repo,
        source_sha=args.source_sha,
        progress_callback=progress,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
