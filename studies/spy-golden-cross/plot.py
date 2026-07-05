#!/usr/bin/env python3
"""Generate SVG charts from the SPY golden-cross backtest CSV outputs.

This script reads outputs produced by backtest.py. It does not recompute the
strategy, which keeps the charting layer auditable.
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

COLORS = {
    "bg": "#ffffff",
    "fg": "#0f172a",
    "muted": "#64748b",
    "grid": "#e2e8f0",
    "strategy": "#0f766e",
    "benchmark": "#334155",
    "sma50": "#0891b2",
    "sma200": "#d97706",
    "risk_on": "#ccfbf1",
    "drawdown": "#dc2626",
    "benchmark_drawdown": "#475569",
}


@dataclass
class SeriesSpec:
    name: str
    values: pd.Series
    color: str
    width: float = 2.0
    dash: str | None = None


def _read_equity() -> pd.DataFrame:
    path = OUTPUT_DIR / "spy-golden-cross-equity.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run backtest.py first")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date")


def _downsample(
    points: list[tuple[float, float]], limit: int = 1200
) -> list[tuple[float, float]]:
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


def _scale_value(
    value: float, min_value: float, max_value: float, log_scale: bool
) -> float:
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


def _legend(specs: list[SeriesSpec], x: int = 78, y: int = 30) -> str:
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
        cursor += 190
    return "\n".join(parts)


def _base_svg(title: str, desc: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">{html.escape(title)}</title>
  <desc id="desc">{html.escape(desc)}</desc>
  <style>
    text {{ font-family: Inter, Arial, sans-serif; fill: {COLORS["fg"]}; }}
    .title {{ font-size: 20px; font-weight: 700; }}
    .subtitle {{ font-size: 13px; fill: {COLORS["muted"]}; }}
    .axis {{ font-size: 12px; fill: {COLORS["muted"]}; }}
    .legend {{ font-size: 13px; fill: {COLORS["fg"]}; }}
  </style>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="{COLORS["bg"]}" />
  <text x="78" y="28" class="title">{html.escape(title)}</text>
  <text x="78" y="48" class="subtitle">{html.escape(desc)}</text>
  {body}
</svg>
"""


def write_equity_chart(df: pd.DataFrame) -> None:
    specs = [
        SeriesSpec(
            "50/200 SMA crossover, 5 bps costs",
            df["StrategyEquity"],
            COLORS["strategy"],
            2.4,
        ),
        SeriesSpec(
            "SPY buy and hold",
            df["BenchmarkEquity"],
            COLORS["benchmark"],
            1.9,
            "7 5",
        ),
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
    body = [
        _axis_svg(
            df.index,
            [(tick, _money_label(tick)) for tick in ticks],
            minimum,
            maximum,
            True,
        )
    ]
    body.append(_legend(specs, y=438))
    for spec in specs:
        dash = f' stroke-dasharray="{spec.dash}"' if spec.dash else ""
        body.append(
            f'<polyline fill="none" stroke="{spec.color}" stroke-width="{spec.width}"{dash} points="{_polyline(df.index, spec.values, minimum, maximum, True)}" />'
        )
    body.append(
        '<text x="78" y="484" class="subtitle">Log scale. Strategy is risk-on when prior SMA50 is above prior SMA200; cash return is 0%.</text>'
    )

    svg = _base_svg(
        "SPY Golden Cross Backtest (50/200 SMA)",
        "Equity curve, $10,000 initial capital",
        "\n".join(body),
    )
    path = CHART_DIR / "spy-golden-cross-equity-curve.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"Wrote {path}")


def write_drawdown_chart(df: pd.DataFrame) -> None:
    specs = [
        SeriesSpec(
            "Strategy drawdown",
            df["StrategyDrawdown"] * 100,
            COLORS["drawdown"],
            2.2,
        ),
        SeriesSpec(
            "Benchmark drawdown",
            df["BenchmarkDrawdown"] * 100,
            COLORS["benchmark_drawdown"],
            1.8,
            "7 5",
        ),
    ]
    minimum = min(spec.values.min() for spec in specs) * 1.08
    maximum = 0.0
    rounded_min = math.floor(minimum / 10) * 10
    ticks = [(value, f"{value:.0f}%") for value in range(int(rounded_min), 1, 10)]
    body = [_axis_svg(df.index, ticks, rounded_min, maximum)]
    body.append(_legend(specs, y=438))
    zero_y = _scale_value(0, rounded_min, maximum, False)
    body.append(
        f'<line x1="{PLOT_X}" x2="{PLOT_X + PLOT_W}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="{COLORS["muted"]}" />'
    )
    for spec in specs:
        dash = f' stroke-dasharray="{spec.dash}"' if spec.dash else ""
        body.append(
            f'<polyline fill="none" stroke="{spec.color}" stroke-width="{spec.width}"{dash} points="{_polyline(df.index, spec.values, rounded_min, maximum, False)}" />'
        )

    svg = _base_svg(
        "SPY Golden Cross Drawdowns (50/200 SMA)",
        "Peak-to-trough drawdown, strategy versus buy and hold",
        "\n".join(body),
    )
    path = CHART_DIR / "spy-golden-cross-drawdowns.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"Wrote {path}")


def _risk_on_rectangles(
    df: pd.DataFrame, min_value: float, max_value: float
) -> str:
    parts = []
    in_segment = False
    start_date = None
    previous_date = None
    for date, position in df["Position"].items():
        if position == 1 and not in_segment:
            start_date = date
            in_segment = True
        elif position == 0 and in_segment:
            end_date = previous_date or date
            x1 = _date_to_x(df.index, pd.Timestamp(start_date))
            x2 = _date_to_x(df.index, pd.Timestamp(end_date))
            parts.append(
                f'<rect x="{x1:.1f}" y="{PLOT_Y}" width="{max(1.0, x2 - x1):.1f}" height="{PLOT_H}" fill="{COLORS["risk_on"]}" opacity="0.45" />'
            )
            in_segment = False
        previous_date = date

    if in_segment and start_date is not None:
        x1 = _date_to_x(df.index, pd.Timestamp(start_date))
        x2 = _date_to_x(df.index, pd.Timestamp(df.index[-1]))
        parts.append(
            f'<rect x="{x1:.1f}" y="{PLOT_Y}" width="{max(1.0, x2 - x1):.1f}" height="{PLOT_H}" fill="{COLORS["risk_on"]}" opacity="0.45" />'
        )
    return "\n".join(parts)


def write_price_chart(df: pd.DataFrame) -> None:
    specs = [
        SeriesSpec("Adjusted close", df["AdjustedClose"], COLORS["benchmark"], 1.4),
        SeriesSpec("SMA50", df["SMA50"], COLORS["sma50"], 2.6),
        SeriesSpec("SMA200", df["SMA200"], COLORS["sma200"], 1.8),
    ]
    minimum = min(spec.values.min(skipna=True) for spec in specs) * 0.86
    maximum = max(spec.values.max(skipna=True) for spec in specs) * 1.10
    ticks = [
        tick
        for tick in [20, 50, 100, 200, 400, 800, 1200]
        if minimum <= tick <= maximum
    ]
    body = [
        _axis_svg(
            df.index,
            [(tick, _money_label(tick)) for tick in ticks],
            minimum,
            maximum,
            True,
        )
    ]
    body.append(_risk_on_rectangles(df, minimum, maximum))
    body.append(_legend(specs, y=438))
    body.append(
        '<text x="592" y="442" class="legend">Shaded area: risk-on</text>'
    )
    for spec in specs:
        body.append(
            f'<polyline fill="none" stroke="{spec.color}" stroke-width="{spec.width}" points="{_polyline(df.index, spec.values, minimum, maximum, True)}" />'
        )

    svg = _base_svg(
        "SPY Adjusted Close, SMA50, and SMA200",
        "Risk-on when SMA50 is above SMA200; crosses mark state changes",
        "\n".join(body),
    )
    path = CHART_DIR / "spy-golden-cross-price-sma.svg"
    path.write_text(svg, encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    df = _read_equity()
    write_equity_chart(df)
    write_drawdown_chart(df)
    write_price_chart(df)


if __name__ == "__main__":
    main()
