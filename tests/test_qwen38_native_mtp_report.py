from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from scripts import qwen38_native_mtp_report as report


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs/perf/receipts/qwen38-native-mtp-four-series-data.json"
LOW_CHART = ROOT / "docs/perf/qwen38-native-mtp-four-series-decode-tps.svg"
XHIGH_CHART = ROOT / "docs/perf/qwen38-native-mtp-xhigh-decode-tps.svg"
REPORT = ROOT / "docs/perf/receipts/qwen38-native-mtp-adaptive.md"

LABELS = [
    "v2.9.2 fixed K3",
    "Optimized fixed K3",
    "Adaptive BF16",
    "Adaptive Q4",
    "DFlash2",
]


def _chart_values(path: Path) -> dict[tuple[int, str], float]:
    root = ET.parse(path).getroot()
    return {
        (int(node.attrib["data-context-tokens"]), node.attrib["data-candidate"]): float(
            node.attrib["data-value"]
        )
        for node in root.iter()
        if node.attrib.get("data-metric") == "decode_tok_s"
    }


def _chart_winners(path: Path) -> set[tuple[int, str]]:
    root = ET.parse(path).getroot()
    return {
        (int(node.attrib["data-context-tokens"]), node.attrib["data-candidate"])
        for node in root.iter()
        if node.attrib.get("data-winner") == "true"
    }


def _expected_chart_values(workload: dict) -> dict[tuple[int, str], float]:
    return {
        (int(row["input_tokens"]), str(row["candidate"])): float(row["decode_tok_s"])
        for row in workload["rows"]
    }


def test_canonical_report_data_has_the_final_1024_output_contract() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))

    assert data["schema_version"] == 2
    assert [candidate["label"] for candidate in data["candidates"]] == LABELS
    assert data["vanity"]["conditioner_tokens"] == 0
    assert data["vanity"]["input_tokens"] == 100
    assert data["vanity"]["output_contract"] == "natural_stop"
    assert {row["candidate"] for row in data["vanity"]["rows"]} == set(LABELS)

    for workload_name in ("low", "xhigh"):
        workload = data[workload_name]
        assert workload["conditioner_tokens"] == 1_024
        assert workload["output_tokens"] == 1_024
        assert sorted({row["input_tokens"] for row in workload["rows"]}) == [
            1_024,
            16_384,
            65_536,
            131_072,
        ]
        for row in workload["rows"]:
            assert row["generated_tokens"] == 1_024
            for metric in ("prefill_tok_s", "decode_tok_s", "wall_s", "peak_memory_gib"):
                assert float(row[metric]) > 0

    xhigh_dflash = [
        row for row in data["xhigh"]["rows"] if row["candidate"] == "DFlash2"
    ]
    assert [row["input_tokens"] for row in xhigh_dflash] == [1_024]


def test_charts_are_mechanically_equal_to_the_canonical_data() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))

    assert _chart_values(LOW_CHART) == {
        **_expected_chart_values(data["vanity"]),
        **_expected_chart_values(data["low"]),
    }
    assert _chart_values(XHIGH_CHART) == _expected_chart_values(data["xhigh"])


def test_every_input_size_highlights_the_lowest_wall_time() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    expected: dict[str, set[tuple[int, str]]] = {"low": set(), "xhigh": set()}

    for workload_name in ("vanity", "low", "xhigh"):
        rows = data[workload_name]["rows"]
        for context in {int(row["input_tokens"]) for row in rows}:
            context_rows = [row for row in rows if int(row["input_tokens"]) == context]
            winner = min(context_rows, key=lambda row: float(row["wall_s"]))
            assert [row["candidate"] for row in context_rows if row["winner"]] == [
                winner["candidate"]
            ]
            chart = "low" if workload_name in {"vanity", "low"} else "xhigh"
            expected[chart].add((context, winner["candidate"]))

    assert _chart_winners(LOW_CHART) == expected["low"]
    assert _chart_winners(XHIGH_CHART) == expected["xhigh"]


def test_winner_marker_prefers_wall_time_when_decode_throughput_disagrees() -> None:
    rows = [
        {"input_tokens": 1_024, "candidate": "decode", "decode_tok_s": 100.0, "wall_s": 11.0},
        {"input_tokens": 1_024, "candidate": "wall", "decode_tok_s": 90.0, "wall_s": 10.0},
    ]

    report._mark_wall_winners(rows)

    assert [row["candidate"] for row in rows if row["winner"]] == ["wall"]


def test_report_omits_machine_local_benchmark_mechanics() -> None:
    report = REPORT.read_text(encoding="utf-8")

    for forbidden in ("/tmp/", "launchd", "scratch path", "GPU-lock"):
        assert forbidden not in report
