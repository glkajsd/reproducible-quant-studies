#!/usr/bin/env python3
"""Generate dependency-free SVG charts from the relative/absolute study CSVs."""
from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CHARTS = ROOT / "charts"
STEM = "relative-vs-absolute-momentum-etf-rotation"
PORTFOLIOS = (
    "Top 1 relative momentum",
    "Top 1 absolute momentum",
    "Top 3 relative momentum",
    "Top 3 absolute momentum",
    "Equal weight",
    "SPY buy and hold",
)
PRIMARY = (
    "Top 1 relative momentum",
    "Top 1 absolute momentum",
    "Top 3 relative momentum",
    "Top 3 absolute momentum",
    "SPY buy and hold",
)
COLORS = {
    "Top 1 relative momentum": "#0f766e",
    "Top 1 absolute momentum": "#14b8a6",
    "Top 3 relative momentum": "#2563eb",
    "Top 3 absolute momentum": "#60a5fa",
    "Equal weight": "#d97706",
    "SPY buy and hold": "#334155",
}


def base(title: str, subtitle: str, body: str, height: int = 540) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 {height}" role="img">
<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>
<style>text{{font-family:Arial,sans-serif;fill:#0f172a}}.a{{font-size:12px;fill:#64748b}}.l{{font-size:13px}}</style>
<rect width="980" height="{height}" fill="white"/><text x="70" y="30" font-size="20" font-weight="700">{html.escape(title)}</text>
<text x="70" y="50" class="a">{html.escape(subtitle)}</text>{body}</svg>"""


def line_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    portfolios: tuple[str, ...],
    percent: bool = False,
    log: bool = False,
) -> None:
    df = df[df["Portfolio"].isin(portfolios)].copy()
    groups = {name: g.set_index("Date")[column] for name, g in df.groupby("Portfolio", sort=False)}
    dates = pd.DatetimeIndex(sorted(df.Date.unique()))
    vals = pd.concat(groups.values())
    lo, hi = float(vals.min()), float(vals.max())
    if log:
        transform = math.log
        lo, hi = transform(lo), transform(hi)
    else:
        transform = lambda x: x
    x = lambda d: 75 + (pd.Timestamp(d).toordinal() - dates[0].toordinal()) / (
        dates[-1].toordinal() - dates[0].toordinal()
    ) * 830
    y = lambda v: 65 + (hi - transform(float(v))) / (hi - lo or 1.0) * 350
    parts = ['<rect x="75" y="65" width="830" height="350" fill="white" stroke="#cbd5e1"/>']
    for i in range(5):
        yy = 65 + i * 87.5
        value = hi - i * (hi - lo) / 4
        label = f"{value:.0%}" if percent else (f"${math.exp(value):,.0f}" if log else f"{value:,.0f}")
        parts += [
            f'<line x1="75" x2="905" y1="{yy}" y2="{yy}" stroke="#e2e8f0"/>',
            f'<text x="67" y="{yy + 4}" text-anchor="end" class="a">{label}</text>',
        ]
    for name in portfolios:
        values = groups[name]
        step = max(1, len(values) // 1000)
        pts = " ".join(f"{x(date):.1f},{y(value):.1f}" for date, value in values.iloc[::step].items())
        parts.append(f'<polyline fill="none" stroke="{COLORS[name]}" stroke-width="2.1" points="{pts}"/>')
    for i, name in enumerate(portfolios):
        xx = 80 + i % 3 * 295
        yy = 452 + i // 3 * 25
        parts += [
            f'<line x1="{xx}" x2="{xx + 24}" y1="{yy}" y2="{yy}" stroke="{COLORS[name]}" stroke-width="3"/>',
            f'<text x="{xx + 30}" y="{yy + 5}" class="l">{html.escape(name)}</text>',
        ]
    parts.append(f'<text x="75" y="520" class="a">{dates[0].date()} to {dates[-1].date()}; 5 bps costs</text>')
    (CHARTS / filename).write_text(
        base(title, "Common complete-universe sample; month-end signal applied one trading day later", "".join(parts)),
        encoding="utf-8",
    )


def turnover_chart() -> None:
    summary = pd.read_csv(OUT / f"{STEM}-summary.csv")
    s = summary[summary["Cost bps"].eq(5) & summary["Portfolio"].isin(PORTFOLIOS)]
    maximum = float(s["Total turnover"].max())
    parts = []
    for i, (_, row) in enumerate(s.iterrows()):
        name = row["Portfolio"]
        value = float(row["Total turnover"])
        y = 85 + i * 56
        width = value / maximum * 570 if maximum else 0
        parts += [
            f'<text x="70" y="{y + 20}" class="l">{html.escape(name)}</text>',
            f'<rect x="285" y="{y}" width="{width:.1f}" height="28" fill="{COLORS[name]}"/>',
            f'<text x="{295 + width:.1f}" y="{y + 19}" class="l">{value:.1f}</text>',
        ]
    (CHARTS / f"{STEM}-turnover.svg").write_text(
        base("Total Turnover", "Sum of absolute ETF weight changes in the 5 bps scenario", "".join(parts), height=455),
        encoding="utf-8",
    )


def main() -> None:
    CHARTS.mkdir(exist_ok=True)
    df = pd.read_csv(OUT / f"{STEM}-equity.csv", parse_dates=["Date"])
    line_chart(
        df,
        "Equity",
        "Relative vs Absolute Momentum ETF Rotation",
        f"{STEM}-equity-curve.svg",
        PRIMARY,
        log=True,
    )
    line_chart(
        df,
        "Drawdown",
        "Relative vs Absolute Momentum Drawdowns",
        f"{STEM}-drawdowns.svg",
        PRIMARY,
        percent=True,
    )
    line_chart(
        df,
        "ActualCashWeight",
        "Cash Weight Under the Absolute Momentum Rule",
        f"{STEM}-cash-weight.svg",
        ("Top 1 absolute momentum", "Top 3 absolute momentum"),
        percent=True,
    )
    turnover_chart()
    print(f"Wrote four SVG charts to {CHARTS}")


if __name__ == "__main__":
    main()
