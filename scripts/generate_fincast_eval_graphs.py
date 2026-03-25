#!/usr/bin/env python3
"""Generate dependency-free SVG evaluation graphs for FinCast metrics."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "models" / "fincast_runtime_local"
FIG_DIR = ROOT / "docs" / "figures"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _y_ticks(y_min: float, y_max: float, count: int = 6) -> list[float]:
    if count <= 1:
        return [y_min, y_max]
    step = (y_max - y_min) / (count - 1)
    return [y_min + i * step for i in range(count)]


def write_grouped_bar_chart(
    output_path: Path,
    *,
    title: str,
    y_label: str,
    categories: list[str],
    series: dict[str, list[float]],
    colors: dict[str, str],
    floor_at_zero: bool = False,
) -> None:
    width, height = 1100, 640
    margin_left, margin_right, margin_top, margin_bottom = 100, 40, 90, 140
    chart_x = margin_left
    chart_y = margin_top
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    all_values: list[float] = []
    for values in series.values():
        all_values.extend(values)

    y_min = min(all_values + [0.0])
    y_max = max(all_values + [0.0])
    if abs(y_max - y_min) < 1e-12:
        y_max += 1.0
        y_min -= 1.0

    pad = max((y_max - y_min) * 0.12, 1e-6)
    if floor_at_zero:
        y_min = 0.0
        y_max += pad
    else:
        y_min -= pad
        y_max += pad

    def to_y(value: float) -> float:
        return chart_y + (y_max - value) * chart_h / (y_max - y_min)

    axis_y = to_y(0.0)

    series_items = list(series.items())
    n_series = len(series_items)
    n_cats = len(categories)
    group_w = chart_w / max(n_cats, 1)
    bar_w = min(56.0, group_w / max(n_series * 1.8, 2.0))
    bar_gap = bar_w * 0.22
    bars_total_w = n_series * bar_w + max(n_series - 1, 0) * bar_gap

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect width="100%" height="100%" fill="#ffffff"/>')

    # Title
    lines.append(
        f'<text x="{width / 2:.1f}" y="42" text-anchor="middle" font-family="Arial, sans-serif" '
        f'font-size="26" fill="#0f172a" font-weight="700">{escape(title)}</text>'
    )

    # Axes
    lines.append(
        f'<line x1="{chart_x}" y1="{chart_y}" x2="{chart_x}" y2="{chart_y + chart_h}" stroke="#334155" stroke-width="1.5"/>'
    )
    lines.append(
        f'<line x1="{chart_x}" y1="{chart_y + chart_h}" x2="{chart_x + chart_w}" y2="{chart_y + chart_h}" stroke="#334155" stroke-width="1.5"/>'
    )

    # Zero line
    if chart_y <= axis_y <= chart_y + chart_h:
        lines.append(
            f'<line x1="{chart_x}" y1="{axis_y:.2f}" x2="{chart_x + chart_w}" y2="{axis_y:.2f}" '
            'stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4 4"/>'
        )

    # Y ticks + grid
    for tick in _y_ticks(y_min, y_max, count=6):
        y = to_y(tick)
        lines.append(
            f'<line x1="{chart_x}" y1="{y:.2f}" x2="{chart_x + chart_w}" y2="{y:.2f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{chart_x - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" '
            f'font-size="13" fill="#334155">{_fmt(tick)}</text>'
        )

    # Y label
    lines.append(
        f'<text x="26" y="{chart_y + chart_h / 2:.1f}" text-anchor="middle" '
        'transform="rotate(-90 26,320)" font-family="Arial, sans-serif" font-size="14" fill="#334155">'
        f'{escape(y_label)}</text>'
    )

    # Bars
    for ci, category in enumerate(categories):
        group_x = chart_x + ci * group_w + (group_w - bars_total_w) / 2

        # X label
        cat_center = chart_x + ci * group_w + group_w / 2
        lines.append(
            f'<text x="{cat_center:.2f}" y="{chart_y + chart_h + 34}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="14" fill="#0f172a">{escape(category)}</text>'
        )

        for si, (series_name, values) in enumerate(series_items):
            value = values[ci]
            x = group_x + si * (bar_w + bar_gap)
            y_val = to_y(value)

            if value >= 0:
                rect_y = y_val
                rect_h = max(axis_y - y_val, 0.8)
                value_label_y = rect_y - 8
            else:
                rect_y = axis_y
                rect_h = max(y_val - axis_y, 0.8)
                value_label_y = rect_y + rect_h + 16

            color = colors.get(series_name, "#475569")
            lines.append(
                f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_w:.2f}" height="{rect_h:.2f}" '
                f'fill="{color}" rx="5"/>'
            )
            lines.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{value_label_y:.2f}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="12" fill="#0f172a">{_fmt(value)}</text>'
            )

    # Legend
    lx = chart_x + chart_w - 250
    ly = chart_y - 42
    for idx, (series_name, _values) in enumerate(series_items):
        y = ly + idx * 24
        color = colors.get(series_name, "#475569")
        lines.append(f'<rect x="{lx}" y="{y}" width="14" height="14" fill="{color}" rx="2"/>')
        lines.append(
            f'<text x="{lx + 22}" y="{y + 12}" font-family="Arial, sans-serif" font-size="13" fill="#0f172a">{escape(series_name)}</text>'
        )

    lines.append('</svg>')
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    frozen_summary = _load_json(METRICS_DIR / "frozen_fincast_summary.json")
    lora_validation = _load_json(METRICS_DIR / "custom_lora_validation_metrics.json")
    lora_holdout = _load_json(METRICS_DIR / "custom_lora_holdout_metrics.json")

    frozen_validation = frozen_summary["validation_metrics"]
    frozen_holdout = frozen_summary["holdout_metrics"]

    colors = {
        "Frozen FinCast": "#64748b",
        "Fine-tuned LoRA": "#0ea5e9",
    }

    # 1) Pooled directional accuracy
    write_grouped_bar_chart(
        FIG_DIR / "fincast_eval_pooled_directional_accuracy.svg",
        title="Pooled Directional Accuracy (Frozen vs Fine-tuned)",
        y_label="Directional Accuracy",
        categories=["Validation", "Holdout"],
        series={
            "Frozen FinCast": [
                float(frozen_validation["pooled::all"]["directional_accuracy"]),
                float(frozen_holdout["pooled::all"]["directional_accuracy"]),
            ],
            "Fine-tuned LoRA": [
                float(lora_validation["pooled::all"]["directional_accuracy"]),
                float(lora_holdout["pooled::all"]["directional_accuracy"]),
            ],
        },
        colors=colors,
        floor_at_zero=True,
    )

    # 2) Pooled rank IC
    write_grouped_bar_chart(
        FIG_DIR / "fincast_eval_pooled_rank_ic.svg",
        title="Pooled Rank IC (Frozen vs Fine-tuned)",
        y_label="Rank IC",
        categories=["Validation", "Holdout"],
        series={
            "Frozen FinCast": [
                float(frozen_validation["pooled::all"]["rank_ic"]),
                float(frozen_holdout["pooled::all"]["rank_ic"]),
            ],
            "Fine-tuned LoRA": [
                float(lora_validation["pooled::all"]["rank_ic"]),
                float(lora_holdout["pooled::all"]["rank_ic"]),
            ],
        },
        colors=colors,
    )

    # 3) Pooled turnover proxy
    write_grouped_bar_chart(
        FIG_DIR / "fincast_eval_pooled_turnover.svg",
        title="Pooled Turnover Proxy (Frozen vs Fine-tuned)",
        y_label="Turnover Proxy",
        categories=["Validation", "Holdout"],
        series={
            "Frozen FinCast": [
                float(frozen_validation["pooled::all"]["turnover_proxy"]),
                float(frozen_holdout["pooled::all"]["turnover_proxy"]),
            ],
            "Fine-tuned LoRA": [
                float(lora_validation["pooled::all"]["turnover_proxy"]),
                float(lora_holdout["pooled::all"]["turnover_proxy"]),
            ],
        },
        colors=colors,
        floor_at_zero=True,
    )

    # 4) Holdout directional accuracy by asset class
    slices = ["asset_class::commodities", "asset_class::equities", "asset_class::rates"]
    labels = ["Commodities", "Equities", "Rates"]
    write_grouped_bar_chart(
        FIG_DIR / "fincast_eval_holdout_asset_class_directional_accuracy.svg",
        title="Holdout Directional Accuracy by Asset Class",
        y_label="Directional Accuracy",
        categories=labels,
        series={
            "Frozen FinCast": [float(frozen_holdout[s]["directional_accuracy"]) for s in slices],
            "Fine-tuned LoRA": [float(lora_holdout[s]["directional_accuracy"]) for s in slices],
        },
        colors=colors,
        floor_at_zero=True,
    )

    # 5) Holdout high-confidence slices (all assets)
    conf_slices = ["confidence_top_20pct::all", "confidence_top_10pct::all"]
    conf_labels = ["Top 20% Confidence", "Top 10% Confidence"]
    write_grouped_bar_chart(
        FIG_DIR / "fincast_eval_holdout_confidence_directional_accuracy.svg",
        title="Holdout Directional Accuracy at High Confidence Slices",
        y_label="Directional Accuracy",
        categories=conf_labels,
        series={
            "Frozen FinCast": [float(frozen_holdout[s]["directional_accuracy"]) for s in conf_slices],
            "Fine-tuned LoRA": [float(lora_holdout[s]["directional_accuracy"]) for s in conf_slices],
        },
        colors=colors,
        floor_at_zero=True,
    )

    print("Generated SVG charts in", FIG_DIR)


if __name__ == "__main__":
    main()
