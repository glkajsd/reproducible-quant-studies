#!/usr/bin/env python3
"""Generate dependency-free SVG charts from the backtest CSV outputs."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT, CHARTS = ROOT / "outputs", ROOT / "charts"
STEM = "multi-asset-etf-momentum-rotation"
COLORS = {"Top 1 momentum": "#0f766e", "Top 3 momentum": "#2563eb",
          "Equal weight": "#d97706", "SPY buy and hold": "#334155"}
TICKERS = ("SPY", "EFA", "EEM", "IEF", "GLD", "DBC", "VNQ")


def base(title: str, subtitle: str, body: str, height: int = 520) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" role="img">
<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>
<style>text{{font-family:Arial,sans-serif;fill:#0f172a}}.a{{font-size:12px;fill:#64748b}}.l{{font-size:13px}}</style>
<rect width="960" height="{height}" fill="white"/><text x="70" y="30" font-size="20" font-weight="700">{html.escape(title)}</text>
<text x="70" y="50" class="a">{html.escape(subtitle)}</text>{body}</svg>"""


def line_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    portfolios: tuple[str, ...],
    percent=False,
    log=False,
) -> None:
    df = df[df["Portfolio"].isin(portfolios)].copy()
    groups = {name: g.set_index("Date")[column] for name, g in df.groupby("Portfolio")}
    dates = pd.DatetimeIndex(sorted(df.Date.unique()))
    vals = pd.concat(groups.values())
    lo, hi = float(vals.min()), float(vals.max())
    if log:
        import math
        transform = math.log
        lo, hi = transform(lo), transform(hi)
    else:
        transform = lambda x: x
    x = lambda d: 75 + (pd.Timestamp(d).toordinal() - dates[0].toordinal()) / (dates[-1].toordinal() - dates[0].toordinal()) * 810
    y = lambda v: 65 + (hi - transform(float(v))) / (hi - lo or 1) * 350
    parts = ['<rect x="75" y="65" width="810" height="350" fill="white" stroke="#cbd5e1"/>']
    for i in range(5):
        yy = 65 + i * 87.5
        value = hi - i * (hi - lo) / 4
        label = f"{value:.0%}" if percent else (f"${__import__('math').exp(value):,.0f}" if log else f"{value:,.0f}")
        parts += [f'<line x1="75" x2="885" y1="{yy}" y2="{yy}" stroke="#e2e8f0"/>',
                  f'<text x="67" y="{yy+4}" text-anchor="end" class="a">{label}</text>']
    for name, values in groups.items():
        pts = " ".join(f"{x(d):.1f},{y(v):.1f}" for d, v in values.iloc[::max(1, len(values)//1000)].items())
        parts.append(f'<polyline fill="none" stroke="{COLORS[name]}" stroke-width="2.2" points="{pts}"/>')
    for i, name in enumerate(groups):
        xx = 80 + i * 205
        parts += [f'<line x1="{xx}" x2="{xx+24}" y1="452" y2="452" stroke="{COLORS[name]}" stroke-width="3"/>',
                  f'<text x="{xx+30}" y="457" class="l">{name}</text>']
    parts.append(f'<text x="75" y="492" class="a">{dates[0].date()} to {dates[-1].date()}; 5 bps costs</text>')
    (CHARTS / filename).write_text(base(title, "Common complete-universe sample; one-day-lagged weights", "".join(parts)), encoding="utf-8")


def allocation(df: pd.DataFrame) -> None:
    g = df[df.Portfolio.eq("Top 3 momentum")].copy()
    dates = pd.DatetimeIndex(g.Date)
    x = lambda d: 75 + (pd.Timestamp(d).toordinal() - dates[0].toordinal()) / (dates[-1].toordinal() - dates[0].toordinal()) * 810
    colors = ["#2563eb", "#0f766e", "#dc2626", "#7c3aed", "#d97706", "#64748b", "#db2777"]
    parts = ['<rect x="75" y="65" width="810" height="350" fill="white" stroke="#cbd5e1"/>']
    lower = pd.Series(0.0, index=g.index)
    for ticker, color in zip(TICKERS, colors):
        upper = lower + g[f"ActualWeight_{ticker}"]
        points = [(x(d), 415 - 350 * u) for d, u in zip(dates, upper)]
        points += [(x(d), 415 - 350 * l) for d, l in reversed(list(zip(dates, lower)))]
        parts.append(f'<polygon fill="{color}" opacity=".82" points="{" ".join(f"{a:.1f},{b:.1f}" for a,b in points[::max(1,len(points)//1800)])}"/>')
        lower = upper
    for i, (ticker, color) in enumerate(zip(TICKERS, colors)):
        xx = 80 + i % 4 * 190; yy = 452 + i // 4 * 25
        parts += [f'<rect x="{xx}" y="{yy-10}" width="14" height="14" fill="{color}"/>',
                  f'<text x="{xx+20}" y="{yy+2}" class="l">{ticker}</text>']
    (CHARTS / f"{STEM}-allocation.svg").write_text(base("Top 3 Monthly Allocation", "Actual weights after the one-trading-day delay", "".join(parts)), encoding="utf-8")


def turnover(df: pd.DataFrame, portfolios: tuple[str, ...]) -> None:
    summary = pd.read_csv(OUT / f"{STEM}-summary.csv")
    s = summary[
        summary["Cost bps"].eq(5) & summary["Portfolio"].isin(portfolios)
    ]
    parts = []
    maximum = s["Total turnover"].max()
    for i, (_, row) in enumerate(s.iterrows()):
        name = row["Portfolio"]
        value = float(row["Total turnover"])
        y = 95 + i * 82
        width = value / maximum * 600
        parts += [f'<text x="70" y="{y+20}" class="l">{html.escape(name)}</text>',
                  f'<rect x="220" y="{y}" width="{width:.1f}" height="30" fill="{COLORS[name]}"/>',
                  f'<text x="{230+width:.1f}" y="{y+20}" class="l">{value:.1f}</text>']
    height = 95 + len(s) * 82
    (CHARTS / f"{STEM}-turnover.svg").write_text(
        base(
            "Total Turnover",
            "Sum of absolute daily weight changes; 5 bps scenario",
            "".join(parts),
            height=height,
        ),
        encoding="utf-8",
    )


def main() -> None:
    CHARTS.mkdir(exist_ok=True)
    df = pd.read_csv(OUT / f"{STEM}-daily-equity.csv", parse_dates=["Date"])
    primary = ("Top 1 momentum", "Top 3 momentum", "SPY buy and hold")
    line_chart(
        df, "Equity", "Multi-Asset ETF Momentum Rotation",
        f"{STEM}-equity-curve.svg", primary, log=True,
    )
    line_chart(
        df, "Drawdown", "Multi-Asset ETF Rotation Drawdowns",
        f"{STEM}-drawdowns.svg", primary, percent=True,
    )
    allocation(df)
    turnover(df, primary)
    print(f"Wrote four SVG charts to {CHARTS}")


if __name__ == "__main__":
    main()
