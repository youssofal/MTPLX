#!/usr/bin/env python3
"""Measure default-off and privacy-safe request capture over fixed records."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _load_capture_module():
    sys.path.insert(0, str(ROOT))
    from mtplx import request_capture

    return request_capture


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def _round_metric(value: float) -> float:
    return round(value, 3)


def measure(records: int) -> dict[str, Any]:
    if records < 1:
        raise ValueError("records must be positive")

    capture = _load_capture_module()
    prompt_tokens = list(range(256))
    completion_tokens = list(range(64))
    prompt_text = "private-prompt-content-" * 32
    response_text = "private-response-content-" * 16
    message_text = "private-message-content"
    secret_text = "Bearer measurement-secret"
    request_payload = {
        "prompt_token_ids": prompt_tokens,
        "prompt_text": prompt_text,
        "messages": [{"role": "user", "content": message_text}],
        "observability": {"authorization": secret_text, "mode": "mtp"},
        "max_tokens": len(completion_tokens),
    }
    environment_names = (
        "MTPLX_REQUEST_CAPTURE_DIR",
        "MTPLX_REQUEST_CAPTURE_KEEP",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TOKENS",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_COMPLETION_TOKENS",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_PROMPT_TEXT",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_RESPONSE_TEXT",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_MESSAGES",
        "MTPLX_REQUEST_CAPTURE_INCLUDE_EXCEPTION_TEXT",
    )
    saved_environment = {name: os.environ.get(name) for name in environment_names}

    try:
        for name in environment_names:
            os.environ.pop(name, None)
        capture._PATHS_BY_ID.clear()
        outcome_payload = {
            **capture.completion_token_ids(completion_tokens),
            "completion_tokens": len(completion_tokens),
            "finish_reason": "stop",
            **capture.clip_text_head_tail(response_text),
        }

        disabled_start = time.perf_counter_ns()
        for index in range(records):
            request_id = f"measure-disabled-{index:06d}"
            capture.capture_request(request_id, request_payload)
            capture.capture_outcome(request_id, outcome_payload)
        disabled_ns = time.perf_counter_ns() - disabled_start

        with tempfile.TemporaryDirectory(prefix="mtplx-request-capture-") as directory:
            os.environ["MTPLX_REQUEST_CAPTURE_DIR"] = directory
            os.environ["MTPLX_REQUEST_CAPTURE_KEEP"] = str(records)
            capture._PATHS_BY_ID.clear()

            enabled_start = time.perf_counter_ns()
            for index in range(records):
                request_id = f"measure-enabled-{index:06d}"
                capture.capture_request(
                    request_id,
                    {**request_payload, "request_index": index},
                )
                capture.capture_outcome(request_id, outcome_payload)
            enabled_ns = time.perf_counter_ns() - enabled_start

            paths = sorted(Path(directory).glob("req-*.json"))
            serialized_records = [path.read_text(encoding="utf-8") for path in paths]
            decoded_records = [json.loads(value) for value in serialized_records]
            total_bytes = sum(path.stat().st_size for path in paths)

        prompt_digest = capture.stable_json_digest(prompt_tokens)
        completion_digest = capture.stable_json_digest(completion_tokens)
        raw_token_ids_absent = all(
            not _contains_key(record, "prompt_token_ids")
            and not _contains_key(record, "completion_token_ids")
            for record in decoded_records
        )
        content_absent = all(
            prompt_text not in serialized
            and response_text not in serialized
            and message_text not in serialized
            and secret_text not in serialized
            for serialized in serialized_records
        )
        counts_and_digests_present = all(
            record.get("prompt_token_count") == len(prompt_tokens)
            and record.get("prompt_tokens_sha256") == prompt_digest
            and record.get("outcome", {}).get("completion_token_count")
            == len(completion_tokens)
            and record.get("outcome", {}).get("completion_tokens_sha256")
            == completion_digest
            for record in decoded_records
        )
        all_records_completed = all(
            record.get("phase") == "completed" for record in decoded_records
        )
        file_count_matches = len(paths) == records
        assertions = {
            "raw_prompt_and_completion_token_ids_absent": raw_token_ids_absent,
            "prompt_response_message_and_secret_content_absent": content_absent,
            "token_counts_and_sha256_digests_present": counts_and_digests_present,
            "all_records_completed": all_records_completed,
            "file_count_matches": file_count_matches,
        }
        if not all(assertions.values()):
            raise RuntimeError(f"privacy assertion failed: {assertions}")

        disabled_seconds = disabled_ns / 1_000_000_000
        enabled_seconds = enabled_ns / 1_000_000_000
        return {
            "schema_version": 1,
            "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_revision": _source_revision(),
            "command": f"python scripts/measure_request_capture.py --records {records}",
            "platform": {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "measurement_scope": (
                "One dispatch and one completion write per enabled record, plus "
                "the same calls with capture disabled. Validation is outside timing."
            ),
            "records": records,
            "payload": {
                "prompt_token_count": len(prompt_tokens),
                "completion_token_count": len(completion_tokens),
                "prompt_chars": len(prompt_text),
                "response_chars": len(response_text),
            },
            "disabled": {
                "total_ms": _round_metric(disabled_seconds * 1000),
                "microseconds_per_record": _round_metric(
                    disabled_seconds * 1_000_000 / records
                ),
            },
            "enabled_default_privacy": {
                "total_ms": _round_metric(enabled_seconds * 1000),
                "microseconds_per_record": _round_metric(
                    enabled_seconds * 1_000_000 / records
                ),
                "records_per_second": _round_metric(records / enabled_seconds),
                "files": len(paths),
                "total_bytes": total_bytes,
                "bytes_per_record": _round_metric(total_bytes / records),
            },
            "privacy_assertions": assertions,
        }
    finally:
        capture._PATHS_BY_ID.clear()
        for name, value in saved_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = measure(args.records)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
