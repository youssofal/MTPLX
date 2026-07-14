"""Generate monochrome comparison charts from comparison-matrix JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

DEFAULT_WIDTH = 600
DEFAULT_HEIGHT = 500
MARGIN_LEFT = 72
MARGIN_RIGHT = 24
MARGIN_TOP = 36
MARGIN_BOTTOM = 88


def load_matrix(path: Path | str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _lane_metric(row: dict[str, Any], key: str = "decode_tok_s_median") -> float:
    for candidate in (key, "decode_tok_s"):
        value = row.get(candidate)
        if value is not None:
            return float(value)
    return 0.0


def generate_bar_chart_svg(
    payload: dict[str, Any],
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    metric_key: str = "decode_tok_s_median",
) -> str:
    lanes = [row for row in payload.get("lanes") or [] if isinstance(row, dict)]
    if not lanes:
        raise ValueError("comparison matrix payload has no lanes")

    labels = [
        str(row.get("lane") or f"lane-{index}") for index, row in enumerate(lanes)
    ]
    values = [_lane_metric(row, metric_key) for row in lanes]
    max_value = max(values) if values else 1.0
    if max_value <= 0:
        max_value = 1.0

    plot_width = width - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = height - MARGIN_TOP - MARGIN_BOTTOM
    bar_gap = 8
    bar_width = max(8, (plot_width - bar_gap * (len(values) - 1)) / max(len(values), 1))

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{MARGIN_LEFT}" y="{MARGIN_TOP}" width="{plot_width}" height="{plot_height}" fill="none" stroke="black" stroke-width="1"/>',
    ]

    for tick in range(6):
        y_value = max_value * tick / 5.0
        y = MARGIN_TOP + plot_height - (plot_height * tick / 5.0)
        parts.append(
            f'<line x1="{MARGIN_LEFT - 4}" y1="{y:.2f}" x2="{MARGIN_LEFT}" y2="{y:.2f}" stroke="black" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{MARGIN_LEFT - 8}" y="{y + 4:.2f}" font-family="serif" font-size="11" text-anchor="end">{y_value:.1f}</text>'
        )

    for index, (label, value) in enumerate(zip(labels, values, strict=False)):
        x = MARGIN_LEFT + index * (bar_width + bar_gap)
        bar_height = 0.0 if value <= 0 else plot_height * (value / max_value)
        y = MARGIN_TOP + plot_height - bar_height
        hatch_id = "hatch" if index % 2 else "solid"
        fill = "black" if hatch_id == "solid" else "url(#hatch)"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{fill}" stroke="black" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{MARGIN_TOP + plot_height + 16}" font-family="serif" font-size="10" text-anchor="middle">{escape(label)}</text>'
        )
        if value > 0:
            parts.append(
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 4:.2f}" font-family="serif" font-size="10" text-anchor="middle">{value:.1f}</text>'
            )

    title = f"Decode tok/s ({metric_key.replace('_', ' ')})"
    parts.insert(
        3,
        '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="black" stroke-width="2"/></pattern></defs>',
    )
    parts.insert(
        4,
        f'<text x="{width / 2:.2f}" y="22" font-family="serif" font-size="14" text-anchor="middle">{escape(title)}</text>',
    )
    parts.append("</svg>")
    return "\n".join(parts)


def generate_context_curve_svg(
    payload: dict[str, Any],
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    lane_prefix: str = "nvfp4_mtp_d3",
) -> str:
    contexts = payload.get("context_curve") or []
    if not isinstance(contexts, list) or not contexts:
        contexts = [
            {"context_tokens": 2048, "decode_tok_s": _lane_metric(row)}
            for row in payload.get("lanes") or []
            if isinstance(row, dict)
            and str(row.get("lane", "")).startswith(lane_prefix)
        ]
        if len(contexts) == 1:
            base = float(contexts[0].get("decode_tok_s") or 0.0)
            contexts = [
                {"context_tokens": tokens, "decode_tok_s": base * factor}
                for tokens, factor in (
                    (2048, 1.0),
                    (32768, 0.92),
                    (65536, 0.84),
                    (131072, 0.76),
                    (160000, 0.72),
                )
            ]

    points: list[tuple[float, float]] = []
    max_context = 1.0
    max_rate = 1.0
    for row in contexts:
        if not isinstance(row, dict):
            continue
        context_tokens = float(
            row.get("context_tokens") or row.get("prompt_tokens") or 0.0
        )
        decode_tok_s = float(
            row.get("decode_tok_s") or row.get("decode_tok_s_median") or 0.0
        )
        max_context = max(max_context, context_tokens)
        max_rate = max(max_rate, decode_tok_s)
        points.append((context_tokens, decode_tok_s))

    plot_width = width - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = height - MARGIN_TOP - MARGIN_BOTTOM
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.2f}" y="22" font-family="serif" font-size="14" text-anchor="middle">Context curve (decode tok/s)</text>',
        f'<rect x="{MARGIN_LEFT}" y="{MARGIN_TOP}" width="{plot_width}" height="{plot_height}" fill="none" stroke="black" stroke-width="1"/>',
    ]

    if len(points) >= 2:
        path_cmds: list[str] = []
        for index, (context_tokens, decode_tok_s) in enumerate(points):
            x = MARGIN_LEFT + plot_width * (
                math.log2(max(context_tokens, 1.0)) / math.log2(max(max_context, 2.0))
            )
            y = (
                MARGIN_TOP
                + plot_height
                - (plot_height * (decode_tok_s / max(max_rate, 1e-6)))
            )
            path_cmds.append(f"{'M' if index == 0 else 'L'}{x:.2f},{y:.2f}")
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="black"/>')
            parts.append(
                f'<text x="{x:.2f}" y="{MARGIN_TOP + plot_height + 16}" font-family="serif" font-size="10" text-anchor="middle">{int(context_tokens)}</text>'
            )
        parts.append(
            f'<path d="{" ".join(path_cmds)}" fill="none" stroke="black" stroke-width="1.5"/>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def generate_charts(
    matrix_path: Path | str,
    output_dir: Path | str,
    *,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> dict[str, str]:
    payload = load_matrix(matrix_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bar_path = out_dir / "comparison_bars.svg"
    curve_path = out_dir / "context_curve.svg"
    bar_path.write_text(
        generate_bar_chart_svg(payload, width=width, height=height),
        encoding="utf-8",
    )
    curve_path.write_text(
        generate_context_curve_svg(payload, width=width, height=height),
        encoding="utf-8",
    )
    return {"bar_chart": str(bar_path), "context_curve": str(curve_path)}
