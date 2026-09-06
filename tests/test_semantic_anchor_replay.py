"""No-model tests for the frozen-transcript measurement contract."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "bench_semantic_anchor_replay.py"
spec = importlib.util.spec_from_file_location("semantic_anchor_replay", MODULE)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def events(text="OK", cached=3):
    return [
        {"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "delta": {"reasoning_content": "think"}}]},
        {"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3,
                                    "prompt_tokens_details": {"cached_tokens": cached}}},
    ]


def measured():
    clock = iter([101.0, 102.0, 103.0, 104.0, 105.0])
    return bench.measure(events(), 100.0, lambda: next(clock))


def receipt(enabled=False):
    row = measured()
    return {"manifest": {"anchors_enabled": enabled, "server_commit": "same"},
            "transcript_sha256": "same",
            "turns": [{"turn": i, "request_sha256": str(i), **row} for i in range(2)]}


class MeasurementTests(unittest.TestCase):
    def test_role_only_event_does_not_start_ttft(self):
        row = measured()
        self.assertEqual(row["ttft_generated_s"], 2.0)
        self.assertEqual(row["ttft_content_s"], 3.0)
        self.assertEqual(row["wall_s"], 5.0)
        self.assertEqual(row["cached_fraction"], 0.3)

    def test_missing_cache_usage_is_not_zero(self):
        data = events()
        del data[-1]["usage"]["prompt_tokens_details"]
        with self.assertRaisesRegex(ValueError, "cached_tokens"):
            bench.measure(data, 0)

    def test_invalid_token_counts_fail(self):
        for value in (-1, True, "3", 11, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.measure(events(cached=value), 0)

    def test_error_finish_is_rejected(self):
        data = events()
        data[2]["choices"][0]["finish_reason"] = "error"
        with self.assertRaises(ValueError):
            bench.measure(data, 0)

    def test_tool_only_output_is_measured_without_visible_text(self):
        data = events()
        data[1]["choices"][0]["delta"] = {"tool_calls": [
            {"index": 0, "id": "random", "function": {"name": "read", "arguments": "{}"}}
        ]}
        data[2]["choices"][0] = {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
        one = bench.measure(data, 0)
        data[1]["choices"][0]["delta"]["tool_calls"][0]["id"] = "different"
        two = bench.measure(data, 0)
        self.assertIsNone(one["ttft_content_s"])
        self.assertEqual(one["output_sha256"], two["output_sha256"])

    def test_output_hash_is_independent_of_sse_segmentation(self):
        split = events()
        split[2]["choices"][0]["delta"]["content"] = "O"
        split.insert(3, {"choices": [{"index": 0, "delta": {"content": "K"}}]})
        self.assertEqual(bench.measure(split, 0)["output_sha256"], measured()["output_sha256"])

    def test_sse_comments_multiline_and_done(self):
        payloads = list(bench.sse_payloads([
            b": keepalive\n", b"\n", b'data: {"choices":\n', b"data: []}\n", b"\n",
            b"data: [DONE]\n", b"\n",
        ]))
        self.assertEqual(payloads, [{"choices": []}])

    def test_truncated_stream_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "without \\[DONE\\]"):
            list(bench.sse_payloads([b'data: {"choices": []}\n', b"\n"]))

    def test_sse_error_is_rejected(self):
        with self.assertRaises(ValueError):
            list(bench.sse_payloads([b'data: {"error": "failure"}\n', b"\n"]))

    def test_compare_reports_cache_and_latency_deltas(self):
        off, on = receipt(), receipt(True)
        on["turns"][1]["cached_tokens"] = 8
        on["turns"][1]["ttft_generated_s"] = 1.0
        pair = bench.compare(off, on)
        self.assertTrue(pair["output_parity"])
        self.assertEqual(pair["turns"][1]["cached_tokens_delta"], 5)
        self.assertEqual(pair["turns"][1]["ttft_generated_delta_s"], -1.0)

    def test_comparison_rejects_drift(self):
        for target, key in (("manifest", "server_commit"), ("root", "transcript_sha256"),
                            ("turn", "request_sha256"), ("turn", "prompt_tokens")):
            off, on = receipt(), receipt(True)
            obj = on if target == "root" else on["manifest"] if target == "manifest" else on["turns"][1]
            obj[key] = "different"
            with self.subTest(key=key), self.assertRaises(ValueError):
                bench.compare(off, on)

    def test_output_mismatch_is_not_a_passing_comparison(self):
        off, on = receipt(), receipt(True)
        on["turns"][1]["output_sha256"] = "different"
        self.assertFalse(bench.compare(off, on)["output_parity"])

    def test_atomic_receipt_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            bench.write_receipt(path, {"first": True})
            with self.assertRaises(FileExistsError):
                bench.write_receipt(path, {"second": True})
            self.assertEqual(json.loads(path.read_text()), {"first": True})
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["receipt.json"])

    def test_nonfinite_receipt_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                bench.write_receipt(Path(tmp) / "receipt.json", {"bad": float("nan")})
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_frozen_requests_are_sent_without_generated_history_injection(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                response = b"".join(b"data: " + bench.canonical(item) + b"\n\n" for item in events(cached=0 if len(captured) == 1 else 3)) + b"data: [DONE]\n\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                body = {"model": "test", "temperature": 0, "max_tokens": 8,
                        "messages": [{"role": "user", "content": "fixed"}]}
                manifest = {key: "test" for key in ("server_commit", "model", "model_revision",
                             "tokenizer_revision", "mlx_version", "hardware", "server_settings")}
                manifest["anchors_enabled"] = False
                (root / "manifest.json").write_text(json.dumps(manifest))
                (root / "transcript.json").write_text(json.dumps({"session_id": "fixed-session", "requests": [body, copy.deepcopy(body)]}))
                result = bench.run(argparse.Namespace(base_url=f"http://127.0.0.1:{server.server_port}",
                    manifest=str(root / "manifest.json"), transcript=str(root / "transcript.json"),
                    timeout_s=2, api_key_env=None))
                self.assertEqual(len(result["turns"]), 2)
                self.assertEqual(captured[0], captured[1])
                self.assertEqual(captured[0]["messages"], body["messages"])
                self.assertNotIn("fixed", json.dumps(result))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
