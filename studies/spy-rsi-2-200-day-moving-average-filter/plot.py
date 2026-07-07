#!/usr/bin/env python3
"""Generate SVG charts from the SPY RSI(2) 200DMA-filter CSV outputs.

This script reads outputs produced by backtest.py. It does not recompute RSI,
signals, positions, returns, or strategy equity.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CHART_DIR = ROOT / "charts"

WIDTH = 960
HEIGHT = 520
PLOT_X = 78
PLOT_Y = 62
PLOT_W = 804
PLOT_H = 340
BASE_COST_BPS = 5.0

VARIANT_RSI_ONLY = "RSI(2) only"
VARIANT_FILTER = "RSI(2) + 200DMA filter"

COLORS = {
    "bg": "#ffffff",
    "fg": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "rsi_only": "#0f766e",
    "filter": "#2563eb",
    "benchmark": "#334155",
    "drawdown": "#dc2626",
}

VARIANT_COLORS = {
    VARIANT_RSI_ONLY: COLORS["rsi_only"],
    VARIANT_FILTER: COLORS["filter"],
}


@dataclass
class SeriesSpec:
    name: str
    values: pd.Series
    color: str
    width: float = 2.0
    dash: str | None = None


def _read_equity() -> pd.DataFrame:
    path = OUTPUT_DIR / "spy-rsi2-200dma-filter-equity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run backtest.py first")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["Strategy variant", "Date"])


def _read_summary() -> pd.DataFrame:
    path = OUTPUT_DIR / "spy-rsi2-200dma-filter-summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run backtest.py first")
    return pd.read_csv(path)


def _variant_series(df: pd.DataFrame, variant: str, column: str) -> pd.Series:
    subset = df[df["Strategy variant"] == variant].copy()
    return subset.set_index("Date")[column]


def _index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(df["Date"].unique()))


def _downsample(points: list[tuple[float, float]], limit: int = 1200) -> list[tuple[float, float]]:
    if len(points) <= limit:
        return points
    step = max(1, math.ceil(len(points) / limit))
    sampled = points[::step]
    if sampled[-1] != points[-1]:
        sampled.append(points[-1])
    return sampled


def _date_to_x(index: pd.DatetimeIndex, date: pd.Timestamp) -> float:
    start = index[0].toordinal()
    end = index[-1].toordinal()
    if start == end:
        return PLOT_X
    return PLOT_X + (date.toordinal() - start) / (end - start) * PLOT_W


def _scale_value(value: float, min_value: float, max_value: float, log_scale: bool) -> float:
    if log_scale:
        value = math.log(value)
        min_value = math.log(min_value)
        max_value = math.log(max_value)
    if max_value == min_value:
        return PLOT_Y + PLOT_H / 2
    return PLOT_Y + (max_value - value) / (max_value - min_value) * PLOT_H


def _polyline(
    index: pd.DatetimeIndex,
    values: pd.Series,
    min_value: float,
    max_value: float,
    log_scale: bool,
) -> str:
    points: list[tuple[float, float]] = []
    for date, value in values.dropna().items():
        if log_scale and value <= 0:
            continue
        x = _date_to_x(index, pd.Timestamp(date))
        y = _scale_value(float(value), min_value, max_value, log_scale)
        points.append((x, y))
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in _downsample(points))


def _year_ticks(index: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, str]]:
    start_year = index[0].year
    end_year = index[-1].year
    step = 5 if end_year - start_year > 12 else 2
    first = start_year + ((step - start_year % step) % step)
    ticks = []
    for year in range(first, end_year + 1, step):
        tick = pd.Timestamp(year=year, month=1, day=1)
        if index[0] <= tick <= index[-1]:
            ticks.append((tick, str(year)))
    if not ticks:
        ticks.append((index[0], str(start_year)))
        ticks.append((index[-1], str(end_year)))
    return ticks


def _money_label(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:.0f}"


def _axis_svg(
    index: pd.DatetimeIndex,
    y_ticks: list[tuple[float, str]],
    min_value: float,
    max_value: float,
    log_scale: bool = False,
) -> str:
    parts = [
        f'<rect x="{PLOT_X}" y="{PLOT_Y}" width="{PLOT_W}" height="{PLOT_H}" fill="#ffffff" stroke="{COLORS["grid"]}" />'
    ]
    for value, label in y_ticks:
        if log_scale and value <= 0:
            continue
        y = _scale_value(value, min_value, max_value, log_scale)
        parts.append(
            f'<line x1="{PLOT_X}" x2="{PLOT_X + PLOT_W}" y1="{y:.1f}" y2="{y:.1f}" stroke="{COLORS["grid"]}" />'
        )
        parts.append(
            f'<text x="{PLOT_X - 10}" y="{y + 4:.1f}" text-anchor="end" class="axis">{html.escape(label)}</text>'
        )
    for tick, label in _year_ticks(index):
        x = _date_to_x(index, tick)
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{PLOT_Y}" y2="{PLOT_Y + PLOT_H}" stroke="{COLORS["grid"]}" />'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{PLOT_Y + PLOT_H + 28}" text-anchor="middle" class="axis">{label}</text>'
        )
    return "\n".join(parts)


def _legend(specs: list[SeriesSpec], x: int = 78, y: int = 438) -> str:
    parts = []
    cursor = x
    for spec in specs:
        dash = f' stroke-dasharray="{spec.dash}"' if spec.dash else ""
        parts.append(
            f'<line x1="{cursor}" x2="{cursor + 22}" y1="{y}" y2="{y}" stroke="{spec.color}" stroke-width="{spec.width}"{dash} />'
        )
        parts.append(
            f'<text x="{cursor + 30}" y="{y + 4}" class="legend">{html.escape(spec.name)}</text>'
        )
        cursor += 225
    return "\n".join(parts)


def _base_svg(title: str, desc: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(desc)}</desc>
  <style>
    text {{ font-family: Inter, Arial, sans-serif; fill: {COLORS["fg"]}; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .subtitle {{ font-size: 13px; fill: {COLORS["muted"]}; }}
    .axis {{ font-size: 12px; fill: {COLORS["muted"]}; }}
    .legend {{ font-size: 13px; fill: {COLORS["fg"]}; }}
    .bar-label {{ font-size: 13px; fill: {COLORS["fg"]}; font-weight: 600; }}
  </style>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["bg"]}" />
  <text x="78" y="28" class="title">{html.escape(title)}</text>
  <text x="78" y="48" class="subtitle">{html.escape(desc)}</text>
  {body}
</svg>
'''


def write_equity_chart(df: pd.DataFrame) -> None:
    index = _index(df)
    specs = [
        SeriesSpec(VARIANT_RSI_ONLY, _variant_series(df, VARIANT_RSI_ONLY, "StrategyEquity"), COLORS["rsi_only"], 2.4),
        SeriesSpec(VARIANT_FILTER, _variant_series(df, VARIANT_FILTER, "StrategyEquity"), COLORS["filter"], 2.3),
        SeriesSpec("SPY buy and hold", _variant_series(df, VARIANT_RSI_ONLY, "BenchmarkEquity"), COLORS["benchmark"], 1.8, "7 5"),
    ]
    minimum = min(spec.values.min() for spec in specs) * 0.92
    maximum = max(spec.values.max() for spec in specs) * 1.08
    ticks = [
        tick
        for tick in [5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
        if minimum <= tick <= maximum
    ]
    if not ticks:
        ticks = [minimum, maximum]
    body = [_axis_svg(index, [(tick, _money_label(tick)) for tick in ticks], minimum, maximum, True)]
    body.append(_legend(specs))
    for spec in specs:
        dash = f' stroke-dasharray="{spec.dash}"' if spec.dash else ""
        body.append(
            f'<polyline fill="none" stroke="{spec.color}" stroke-width="{spec.width}"{dash} points="{_polyline(index, spec.values, minimum, maximum, True)}" />'
        )
    body.append('<text x="78" y="484" class="subtitle">Log scale. Strategy variants use 5 bps base costs and lagged post-close signal states.</text>')

    svg = _base_svg(
        "SPY RSI(2) With and Without 200DMA Filter",
        "Equity curve, $10,000 initial capital",
        "\n".join(body),
    )
    path = CHART_DIR / "spy-rsi2-200dma-filter-equity-curve.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"Wrote {path}")


def write_drawdown_chart(df: pd.DataFrame) -> None:
    index = _index(df)
    specs = [
        SeriesSpec(VARIANT_RSI_ONLY, _variant_series(df, VARIANT_RSI_ONLY, "StrategyDrawdown") * 100, COLORS["rsi_only"], 2.3),
        SeriesSpec(VARIANT_FILTER, _variant_series(df, VARIANT_FILTER, "StrategyDrawdown") * 100, COLORS["filter"], 2.2),
        SeriesSpec("SPY buy and hold", _variant_series(df, VARIANT_RSI_ONLY, "BenchmarkDrawdown") * 100, COLORS["benchmark"], 1.8, "7 5"),
    ]
    minimum = min(spec.values.min() for spec in specs) * 1.08
    maximum = 0.0
    rounded_min = math.floor(minimum / 10) * 10
    ticks = [(value, f"{value:.0f}%") for value in range(int(rounded_min), 1, 10)]
    body = [_axis_svg(index, ticks, rounded_min, maximum)]
    body.append(_legend(specs))
    zero_y = _scale_value(0, rounded_min, maximum, False)
    body.append(
        f'<line x1="{PLOT_X}" x2="{PLOT_X + PLOT_W}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="{COLORS["muted"]}" />'
    )
    for spec in specs:
        dash = f' stroke-dasharray="{spec.dash}"' if spec.dash else ""
        body.append(
            f'<polyline fill="none" stroke="{spec.color}" stroke-width="{spec.width}"{dash} points="{_polyline(index, spec.values, rounded_min, maximum, False)}" />'
        )

    svg = _base_svg(
        "SPY RSI(2) Strategy Drawdowns",
        "Peak-to-trough drawdown by strategy variant",
        "\n".join(body),
    )
    path = CHART_DIR / "spy-rsi2-200dma-filter-drawdowns.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"Wrote {path}")


def write_position_changes_chart(summary: pd.DataFrame) -> None:
    base = summary[summary["Cost bps"] == BASE_COST_BPS].copy()
    variants = [VARIANT_RSI_ONLY, VARIANT_FILTER]
    values = [
        float(base.loc[base["Strategy variant"] == variant, "Position changes"].iloc[0])
        for variant in variants
    ]
    max_value = max(values) if values else 1.0

    plot_x = 190
    plot_y = 116
    plot_w = 640
    plot_h = 190
    bar_h = 48
    gap = 46
    parts = [
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="{COLORS["grid"]}" />'
    ]
    for i in range(5):
        value = max_value * i / 4
        x = plot_x + value / max_value * plot_w
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{plot_y}" y2="{plot_y + plot_h}" stroke="{COLORS["grid"]}" />'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{plot_y + plot_h + 28}" text-anchor="middle" class="axis">{value:.0f}</text>'
        )

    for idx, (variant, value) in enumerate(zip(variants, values)):
        y = plot_y + 32 + idx * (bar_h + gap)
        width = value / max_value * plot_w
        color = VARIANT_COLORS[variant]
        parts.append(
            f'<text x="{plot_x - 18}" y="{y + 30}" text-anchor="end" class="bar-label">{html.escape(variant)}</text>'
        )
        parts.append(
            f'<rect x="{plot_x}" y="{y}" width="{width:.1f}" height="{bar_h}" fill="{color}" />'
        )
        parts.append(
            f'<text x="{plot_x + width + 10:.1f}" y="{y + 30}" class="bar-label">{value:.0f}</text>'
        )

    parts.append('<text x="78" y="438" class="subtitle">Position changes count cash-to-SPY and SPY-to-cash transitions in the 5 bps base case.</text>')

    svg = _base_svg(
        "SPY RSI(2) Position Changes",
        "Turnover comparison for RSI-only and filtered variants",
        "\n".join(parts),
    )
    path = CHART_DIR / "spy-rsi2-200dma-filter-position-changes.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    equity = _read_equity()
    summary = _read_summary()
    write_equity_chart(equity)
    write_drawdown_chart(equity)
    write_position_changes_chart(summary)


if __name__ == "__main__":
    main()
