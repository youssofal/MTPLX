#!/usr/bin/env python3
"""Build the PR #347 canonical data, tables, and charts from validated receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape


V292_COMMIT = "bbc67427e88288001e4b90ecb44708dc0222154c"
DFLASH_COMMIT = "9a6f48e69f9c8c6932d0f005c364844b2bf33e9c"
CONTEXTS = (1_024, 16_384, 65_536, 131_072)
NATIVE_LANES = (
    "v2.9.2-mlx0322",
    "full-fixed-k3",
    "full-adaptive",
    "full-q4-adaptive",
)
CANDIDATES = (
    ("v2.9.2-mlx0322", "v2.9.2 fixed K3"),
    ("full-fixed-k3", "Optimized fixed K3"),
    ("full-adaptive", "Adaptive BF16"),
    ("full-q4-adaptive", "Adaptive Q4"),
    ("dflash2", "DFlash2"),
)
LABEL_BY_ID = dict(CANDIDATES)
COLOR_BY_ID = {
    "v2.9.2-mlx0322": "#8B95A7",
    "full-fixed-k3": "#2E90FA",
    "full-adaptive": "#12B76A",
    "full-q4-adaptive": "#7A5AF8",
    "dflash2": "#F79009",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = payload.get("invariant_errors")
    if errors != []:
        raise ValueError(f"receipt does not validate: {path}: {errors}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generated_tokens(receipt: dict[str, Any], lane_id: str) -> int:
    values = {
        int(arm["generated_tokens"])
        for arm in receipt["arms"]
        if arm.get("lane_id") == lane_id
    }
    if len(values) != 1:
        raise ValueError(f"{lane_id} generated-token counts disagree: {values}")
    return values.pop()


def _native_rows(path: Path, *, workload: str) -> tuple[list[dict[str, Any]], dict]:
    receipt = _read(path)
    if receipt.get("workload") != workload:
        raise ValueError(f"wrong workload in {path}")
    if tuple(receipt.get("lanes") or ()) != NATIVE_LANES:
        raise ValueError(f"native lane set is incomplete in {path}")
    context = int(receipt["context_tokens"])
    conditioner = int(receipt["conditioner_output_tokens"])
    output = int(receipt["timed_output_tokens"])
    if conditioner != (0 if workload == "vanity" else 1_024) or output != 1_024:
        raise ValueError(f"token contract mismatch in {path}")

    rows = []
    for lane_id in NATIVE_LANES:
        summary = receipt["summary"][lane_id]
        source_commit = str(summary["source_commit"])
        if lane_id == "v2.9.2-mlx0322" and source_commit != V292_COMMIT:
            raise ValueError("v2.9.2 lane source mismatch")
        generated = _generated_tokens(receipt, lane_id)
        if workload != "vanity" and generated != 1_024:
            raise ValueError(f"{lane_id} did not generate exactly 1024 tokens")
        rows.append(
            {
                "candidate_id": lane_id,
                "candidate": LABEL_BY_ID[lane_id],
                "input_tokens": context,
                "conditioner_tokens": conditioner,
                "generated_tokens": generated,
                "output_contract": "natural_stop" if workload == "vanity" else "exact",
                "arms": int(summary["arms"]),
                "source_commit": source_commit,
                "prefill_tok_s": float(summary["prefill_tok_s_mean"]),
                "decode_tok_s": float(summary["decode_tok_s_mean"]),
                "wall_s": float(summary["wall_s_mean"]),
                "peak_memory_gib": float(summary["peak_memory_gib_max"]),
                "deterministic": summary["per_lane_token_deterministic"],
            }
        )
    return rows, receipt


def _dflash_row(path: Path, *, workload: str) -> tuple[dict[str, Any], dict]:
    receipt = _read(path)
    if receipt.get("workload") != workload:
        raise ValueError(f"wrong DFlash2 workload in {path}")
    context = int(receipt["context_tokens"])
    conditioner = int(receipt["conditioner_output_tokens"])
    output = int(receipt["timed_output_tokens"])
    if conditioner != (0 if workload == "vanity" else 1_024) or output != 1_024:
        raise ValueError(f"DFlash2 token contract mismatch in {path}")
    summary = receipt["summary"]
    if summary["source_commit"] != DFLASH_COMMIT:
        raise ValueError("DFlash2 source mismatch")
    generated_values = {int(value) for value in summary["generated_tokens"]}
    if len(generated_values) != 1:
        raise ValueError("DFlash2 generated-token counts disagree")
    generated = generated_values.pop()
    if workload != "vanity" and generated != 1_024:
        raise ValueError("DFlash2 did not generate exactly 1024 tokens")
    return (
        {
            "candidate_id": "dflash2",
            "candidate": LABEL_BY_ID["dflash2"],
            "input_tokens": context,
            "conditioner_tokens": conditioner,
            "generated_tokens": generated,
            "output_contract": "natural_stop" if workload == "vanity" else "exact",
            "arms": int(summary["arms"]),
            "source_commit": str(summary["source_commit"]),
            "prefill_tok_s": float(summary["prefill_tok_s_mean"]),
            "decode_tok_s": float(summary["decode_tok_s_mean"]),
            "wall_s": float(summary["wall_s_mean"]),
            "peak_memory_gib": float(summary["peak_memory_gib_max"]),
            "deterministic": summary["paired_token_deterministic"],
        },
        receipt,
    )


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {candidate_id: index for index, (candidate_id, _) in enumerate(CANDIDATES)}
    return sorted(rows, key=lambda row: (int(row["input_tokens"]), order[row["candidate_id"]]))


def _add_wall_deltas(rows: list[dict[str, Any]]) -> None:
    fixed = {
        int(row["input_tokens"]): float(row["wall_s"])
        for row in rows
        if row["candidate_id"] == "full-fixed-k3"
    }
    for row in rows:
        baseline = fixed.get(int(row["input_tokens"]))
        row["wall_vs_optimized_fixed_pct"] = (
            (baseline / float(row["wall_s"]) - 1.0) * 100.0
            if baseline is not None
            else None
        )


def _mark_wall_winners(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["winner"] = False
    for context in {int(row["input_tokens"]) for row in rows}:
        context_rows = [row for row in rows if int(row["input_tokens"]) == context]
        min(context_rows, key=lambda row: float(row["wall_s"]))["winner"] = True


def build_data(args: argparse.Namespace) -> dict[str, Any]:
    source_receipts: list[dict[str, Any]] = []

    vanity_rows, _ = _native_rows(args.vanity_native, workload="vanity")
    vanity_dflash, _ = _dflash_row(args.vanity_dflash, workload="vanity")
    vanity_rows.append(vanity_dflash)
    for path in (args.vanity_native, args.vanity_dflash):
        source_receipts.append({"workload": "vanity", "path_label": path.name, "sha256": _sha256(path)})

    low_rows: list[dict[str, Any]] = []
    low_128_receipt = None
    for path in args.low_native:
        rows, receipt = _native_rows(path, workload="low")
        low_rows.extend(rows)
        if int(receipt["context_tokens"]) == 131_072:
            low_128_receipt = receipt
        source_receipts.append({"workload": "low", "input_tokens": receipt["context_tokens"], "candidate_group": "native", "sha256": _sha256(path)})
    for path in args.low_dflash:
        row, receipt = _dflash_row(path, workload="low")
        low_rows.append(row)
        source_receipts.append({"workload": "low", "input_tokens": receipt["context_tokens"], "candidate_group": "dflash2", "sha256": _sha256(path)})

    xhigh_rows: list[dict[str, Any]] = []
    xhigh_128_receipt = None
    for path in args.xhigh_native:
        rows, receipt = _native_rows(path, workload="xhigh")
        xhigh_rows.extend(rows)
        if int(receipt["context_tokens"]) == 131_072:
            xhigh_128_receipt = receipt
        source_receipts.append({"workload": "xhigh", "input_tokens": receipt["context_tokens"], "candidate_group": "native", "sha256": _sha256(path)})
    xhigh_dflash, xhigh_dflash_receipt = _dflash_row(args.xhigh_dflash, workload="xhigh")
    if xhigh_dflash["input_tokens"] != 1_024:
        raise ValueError("DFlash2 xhigh must be the 1K-input comparator")
    xhigh_rows.append(xhigh_dflash)
    source_receipts.append({"workload": "xhigh", "input_tokens": 1_024, "candidate_group": "dflash2", "sha256": _sha256(args.xhigh_dflash)})

    if {row["input_tokens"] for row in low_rows} != set(CONTEXTS):
        raise ValueError("low contexts are incomplete")
    if {row["input_tokens"] for row in xhigh_rows} != set(CONTEXTS):
        raise ValueError("xhigh contexts are incomplete")
    if len(low_rows) != 20 or len(xhigh_rows) != 17:
        raise ValueError("candidate matrix is incomplete")
    if low_128_receipt is None or xhigh_128_receipt is None:
        raise ValueError("128K adaptive telemetry receipt is missing")

    low_rows = _ordered_rows(low_rows)
    xhigh_rows = _ordered_rows(xhigh_rows)
    vanity_rows = _ordered_rows(vanity_rows)
    _add_wall_deltas(vanity_rows)
    _add_wall_deltas(low_rows)
    _add_wall_deltas(xhigh_rows)
    _mark_wall_winners(vanity_rows)
    _mark_wall_winners(low_rows)
    _mark_wall_winners(xhigh_rows)

    depth: dict[str, Any] = {}
    for workload, receipt in (("low", low_128_receipt), ("xhigh", xhigh_128_receipt)):
        depth[workload] = {
            LABEL_BY_ID[lane_id]: receipt["summary"][lane_id]["depth_usage"]
            for lane_id in ("full-adaptive", "full-q4-adaptive")
        }
        if any(value is None for value in depth[workload].values()):
            raise ValueError(f"{workload} 128K depth telemetry is missing")

    all_rows = vanity_rows + low_rows + xhigh_rows
    if any(
        not math.isfinite(float(row[key])) or float(row[key]) <= 0
        for row in all_rows
        for key in ("prefill_tok_s", "decode_tok_s", "wall_s", "peak_memory_gib")
    ):
        raise ValueError("non-finite or non-positive benchmark metric")

    return {
        "schema_version": 2,
        "kind": "qwen38_native_mtp_pr347_matched_campaign",
        "software": {"mlx": "0.32.2", "mlx_metal": "0.32.2"},
        "candidates": [
            {"id": candidate_id, "label": label} for candidate_id, label in CANDIDATES
        ],
        "vanity": {
            "input_tokens": 100,
            "conditioner_tokens": 0,
            "output_contract": "natural_stop",
            "temperature": 0.0,
            "rows": vanity_rows,
        },
        "low": {
            "reasoning_effort": "low",
            "conditioner_tokens": 1_024,
            "output_tokens": 1_024,
            "rows": low_rows,
        },
        "xhigh": {
            "reasoning_effort": "xhigh",
            "conditioner_tokens": 1_024,
            "output_tokens": 1_024,
            "rows": xhigh_rows,
        },
        "depth_usage_128k": depth,
        "source_receipts": source_receipts,
        "dflash_receipt_kind": xhigh_dflash_receipt["kind"],
    }


def render_chart(data: dict[str, Any], workload: str) -> str:
    section = data[workload]
    rows = (
        [*data["vanity"]["rows"], *section["rows"]]
        if workload == "low"
        else section["rows"]
    )
    contexts = sorted({int(row["input_tokens"]) for row in rows})
    width, height = 1280, 720
    left, right, top, bottom = 90, 36, 104, 126
    plot_w, plot_h = width - left - right, height - top - bottom
    max_value = max(float(row["decode_tok_s"]) for row in rows)
    y_max = math.ceil(max_value / 10.0) * 10.0
    group_w = plot_w / len(contexts)
    bar_w = min(42.0, group_w / 6.5)
    order = [candidate_id for candidate_id, _ in CANDIDATES]
    lookup = {(int(row["input_tokens"]), row["candidate_id"]): row for row in rows}
    title = (
        "Qwen3.8 MTP decode throughput — vanity + thinking=low"
        if workload == "low"
        else "Qwen3.8 MTP decode throughput — thinking=xhigh"
    )
    subtitle = (
        "100-token vanity has no conditioner and stops naturally · thinking rows: 1,024 conditioner and output"
        if workload == "low"
        else "1,024 conditioner · input-prefill on x-axis · 1,024 generated output"
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0B1220"/>',
        f'<text x="{left}" y="44" fill="#F8FAFC" font-size="24" font-family="system-ui" font-weight="700">{escape(title)}</text>',
        f'<text x="{left}" y="72" fill="#94A3B8" font-size="14" font-family="system-ui">{subtitle}</text>',
    ]
    for tick in range(0, int(y_max) + 1, 10):
        y = top + plot_h - (tick / y_max * plot_h)
        out.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#233047" stroke-width="1"/>')
        out.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" fill="#94A3B8" font-size="12" font-family="system-ui">{tick}</text>')
    for context_index, context in enumerate(contexts):
        center = left + (context_index + 0.5) * group_w
        for lane_index, candidate_id in enumerate(order):
            row = lookup.get((context, candidate_id))
            if row is None:
                continue
            value = float(row["decode_tok_s"])
            winner = bool(row["winner"])
            bar_h = value / y_max * plot_h
            x = center + (lane_index - 2) * (bar_w + 5) - bar_w / 2
            y = top + plot_h - bar_h
            out.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="4" fill="{COLOR_BY_ID[candidate_id]}" stroke="{("#FDE68A" if winner else "none")}" stroke-width="{(4 if winner else 0)}" data-metric="decode_tok_s" data-context-tokens="{context}" data-candidate="{escape(LABEL_BY_ID[candidate_id])}" data-value="{value!r}" data-winner="{str(winner).lower()}"/>'
            )
        label = {
            100: "100",
            1_024: "1K",
            16_384: "16K",
            65_536: "64K",
            131_072: "128K",
        }[context]
        out.append(f'<text x="{center:.2f}" y="{top + plot_h + 28}" text-anchor="middle" fill="#E2E8F0" font-size="14" font-family="system-ui">{label}</text>')
    legend_y = height - 58
    legend_x = left
    for candidate_id, label in CANDIDATES:
        out.append(f'<rect x="{legend_x}" y="{legend_y - 12}" width="14" height="14" rx="3" fill="{COLOR_BY_ID[candidate_id]}"/>')
        out.append(f'<text x="{legend_x + 22}" y="{legend_y}" fill="#CBD5E1" font-size="13" font-family="system-ui">{escape(label)}</text>')
        legend_x += 220
    out.append('</svg>\n')
    return "\n".join(out)


def _context_label(tokens: int) -> str:
    return {100: "100", 1_024: "1K", 16_384: "16K", 65_536: "64K", 131_072: "128K"}[tokens]


def _table(rows: list[dict[str, Any]], *, vanity: bool = False) -> str:
    lines = [
        "| Input | Candidate | Output | Prefill tok/s | Decode tok/s | Wall (s) | vs optimized fixed | Peak GiB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = row["wall_vs_optimized_fixed_pct"]
        delta_text = "baseline" if row["candidate_id"] == "full-fixed-k3" else f"{delta:+.2f}%"
        output = str(row["generated_tokens"]) + (" natural" if vanity else "")
        candidate = (
            f'**★ {row["candidate"]}**' if row["winner"] else row["candidate"]
        )
        lines.append(
            f'| {_context_label(int(row["input_tokens"]))} | {candidate} | {output} | '
            f'{row["prefill_tok_s"]:.2f} | {row["decode_tok_s"]:.2f} | {row["wall_s"]:.3f} | '
            f'{delta_text} | {row["peak_memory_gib"]:.3f} |'
        )
    return "\n".join(lines)


def _depth_table(depth: dict[str, Any]) -> str:
    lines = [
        "| Candidate | Cycles | Attempt D0 | D1 | D2 | D3 | Accept D0 | D1 | D2 | D3 | Mean attempted | Mean accepted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("Adaptive BF16", "Adaptive Q4"):
        usage = depth[label]
        attempted = usage["attempted_share_pct"]
        accepted = usage["accepted_share_pct"]
        lines.append(
            f'| {label} | {usage["decode_cycles"]} | '
            + " | ".join(f'{attempted[f"D{i}"]:.2f}%' for i in range(4))
            + " | "
            + " | ".join(f'{accepted[f"D{i}"]:.2f}%' for i in range(4))
            + f' | {usage["mean_attempted_depth"]:.3f} | {usage["mean_accepted_depth"]:.3f} |'
        )
    return "\n".join(lines)


def render_report(data: dict[str, Any]) -> str:
    return f"""# Qwen3.8 native-MTP optimized profiles

This receipt replaces the earlier campaign. Every measured candidate uses MLX and Metal 0.32.2. The thinking matrices use a 1,024-token same-prompt conditioner, the Qwen thinking template, temperature 1.0, top-p 0.95, top-k 20, seed 42, and exactly 1,024 generated tokens. The x-axis is input-prefill length—not output length.

The optimized fixed-K3 lane uses the matched optimized route and remains pinned at K=3 without executing adaptive depth. Adaptive BF16 and Adaptive Q4 use the same workload-specific optimized profile plus the existing `--adaptive-policy position_ema` toggle. The v2.9.2 lane is exact source `{V292_COMMIT}` with only MLX/Metal upgraded. DFlash2 is exact source `{DFLASH_COMMIT}`.

Every current native lane other than v2.9.2 uses a measured optimized shared profile. Low uses `r20_kv_only_history+r53_command_buffers+r08_device_draft+r10_compact_vocab+r21_qk_rms_rope+r24_eval_ladder+r26_prefill_ladder_3`; xhigh uses `r20_kv_only_history+r24_eval_ladder+r26_prefill_ladder_3+r50_wired_residency+r53_command_buffers`. Fixed K3 uses the applicable shared profile without `r11`; Adaptive BF16 adds `r11_position_ema`; Adaptive Q4 adds `r11_position_ema+r17_q4_mtp_block`. DFlash2 uses its separate PR335 optimized comparator path.

The custom Q4 head is retained for further benchmarking but is not published: it wins low at 1K and 16K, then loses low at 64K and 128K and loses three of four xhigh rows. That matched evidence does not justify a supported artifact yet.

Winner highlights use lowest wall time at each input/prefill size. The charts still plot decode tok/s; their gold outline marks the wall-time winner.

## 100-token temperature-zero vanity prompt

No conditioner or prefill-generation pass is used. All five candidates stop naturally at the same 102-token output.

{_table(data["vanity"]["rows"], vanity=True)}

## Thinking=low — 1,024 output tokens

![Low-reasoning decode throughput](../qwen38-native-mtp-four-series-decode-tps.svg)

{_table(data["low"]["rows"])}

## Thinking=xhigh — 1,024 output tokens

![Xhigh-reasoning decode throughput](../qwen38-native-mtp-xhigh-decode-tps.svg)

{_table(data["xhigh"]["rows"])}

DFlash2 is intentionally measured only at the 1K-input xhigh row.

## 128K adaptive-depth telemetry

Attempted and accepted shares are speculative decode-cycle shares derived from the recorded schedule events; they are not shares of wall time. Fixed K3 is excluded because it remains pinned at depth 3 and never executes the adaptive policy.

### Thinking=low

{_depth_table(data["depth_usage_128k"]["low"])}

### Thinking=xhigh

{_depth_table(data["depth_usage_128k"]["xhigh"])}

## Reproducibility

[`qwen38-native-mtp-four-series-data.json`](qwen38-native-mtp-four-series-data.json) is the canonical source for every number in these tables and both charts. The JSON records the SHA-256 identity of every aggregate receipt. The chart bars carry the exact canonical decode value in `data-value`, and the focused test mechanically compares every plotted bar with the JSON row.

```bash
.venv/bin/python -m pytest -q \
  tests/test_qwen38_native_mtp_matrix.py \
  tests/test_qwen38_dflash2_matrix.py \
  tests/test_qwen38_fixed_k3_xhigh_gate.py \
  tests/test_qwen38_native_mtp_report.py
.venv/bin/python -m ruff check \
  scripts/qwen38_native_mtp_matrix.py \
  scripts/qwen38_native_mtp_report.py \
  tests/test_qwen38_native_mtp_matrix.py \
  tests/test_qwen38_native_mtp_report.py
```
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanity-native", type=Path, required=True)
    parser.add_argument("--vanity-dflash", type=Path, required=True)
    parser.add_argument("--low-native", type=Path, nargs=4, required=True)
    parser.add_argument("--low-dflash", type=Path, nargs=4, required=True)
    parser.add_argument("--xhigh-native", type=Path, nargs=4, required=True)
    parser.add_argument("--xhigh-dflash", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--low-chart", type=Path, required=True)
    parser.add_argument("--xhigh-chart", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    data = build_data(args)
    _write(args.data, json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _write(args.report, render_report(data))
    _write(args.low_chart, render_chart(data, "low"))
    _write(args.xhigh_chart, render_chart(data, "xhigh"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
