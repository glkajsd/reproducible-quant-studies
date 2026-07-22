#!/usr/bin/env python3
"""Generate dependency-free SVG charts for the monthly contribution study."""
from __future__ import annotations

import html
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CHARTS = ROOT / "charts"
STEM = "monthly-contributions-spy-qqq-portfolio-backtest"
PORTFOLIOS = (
    "SPY 100%",
    "QQQ 100%",
    "60/40 SPY IEF",
    "Multi-asset 40/40/15/5",
)
COLORS = {
    "SPY 100%": "#2563eb",
    "QQQ 100%": "#7c3aed",
    "60/40 SPY IEF": "#0f766e",
    "Multi-asset 40/40/15/5": "#d97706",
    "Total contributed": "#64748b",
}


def base(title: str, subtitle: str, body: str, height: int = 540) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 {height}" role="img">
<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>
<style>text{{font-family:Arial,sans-serif;fill:#0f172a}}.a{{font-size:12px;fill:#64748b}}.l{{font-size:13px}}</style>
<rect width="980" height="{height}" fill="white"/><text x="70" y="30" font-size="20" font-weight="700">{html.escape(title)}</text>
<text x="70" y="50" class="a">{html.escape(subtitle)}</text>{body}</svg>"""


def _coords(dates: pd.DatetimeIndex, lo: float, hi: float):
    def x(date) -> float:
        denom = dates[-1].toordinal() - dates[0].toordinal()
        return 75 + (pd.Timestamp(date).toordinal() - dates[0].toordinal()) / (denom or 1) * 830

    def y(value: float) -> float:
        return 65 + (hi - float(value)) / (hi - lo or 1.0) * 350

    return x, y


def line_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
    portfolios: tuple[str, ...] = PORTFOLIOS,
    percent: bool = False,
    include_contributions: bool = False,
) -> None:
    df = df[df["Portfolio"].isin(portfolios)].copy()
    groups = {name: g.set_index("Date")[column] for name, g in df.groupby("Portfolio", sort=False)}
    dates = pd.DatetimeIndex(sorted(df["Date"].unique()))
    values = pd.concat(groups.values())
    if include_contributions:
        contribution = df[df["Portfolio"].eq(portfolios[0])].set_index("Date")["TotalContributed"]
        values = pd.concat([values, contribution])
    lo, hi = float(values.min()), float(values.max())
    if not percent:
        lo = min(0.0, lo)
    x, y = _coords(dates, lo, hi)
    parts = ['<rect x="75" y="65" width="830" height="350" fill="white" stroke="#cbd5e1"/>']
    for i in range(5):
        yy = 65 + i * 87.5
        value = hi - i * (hi - lo) / 4
        label = f"{value:.0%}" if percent else f"${value:,.0f}"
        parts += [
            f'<line x1="75" x2="905" y1="{yy}" y2="{yy}" stroke="#e2e8f0"/>',
            f'<text x="67" y="{yy + 4}" text-anchor="end" class="a">{label}</text>',
        ]
    for name in portfolios:
        series = groups[name]
        step = max(1, len(series) // 1000)
        pts = " ".join(f"{x(date):.1f},{y(value):.1f}" for date, value in series.iloc[::step].items())
        parts.append(f'<polyline fill="none" stroke="{COLORS[name]}" stroke-width="2.1" points="{pts}"/>')
    if include_contributions:
        series = contribution
        step = max(1, len(series) // 1000)
        pts = " ".join(f"{x(date):.1f},{y(value):.1f}" for date, value in series.iloc[::step].items())
        parts.append(f'<polyline fill="none" stroke="{COLORS["Total contributed"]}" stroke-width="2" stroke-dasharray="5 5" points="{pts}"/>')
    legend = list(portfolios) + (["Total contributed"] if include_contributions else [])
    for i, name in enumerate(legend):
        xx = 80 + i % 2 * 390
        yy = 452 + i // 2 * 25
        dash = ' stroke-dasharray="5 5"' if name == "Total contributed" else ""
        parts += [
            f'<line x1="{xx}" x2="{xx + 24}" y1="{yy}" y2="{yy}" stroke="{COLORS[name]}" stroke-width="3"{dash}/>',
            f'<text x="{xx + 30}" y="{yy + 5}" class="l">{html.escape(name)}</text>',
        ]
    parts.append(f'<text x="75" y="520" class="a">{dates[0].date()} to {dates[-1].date()}; $500 monthly contributions</text>')
    (CHARTS / filename).write_text(
        base(title, "Common ETF sample; first trading day monthly contributions", "".join(parts)),
        encoding="utf-8",
    )


def sensitivity_chart() -> None:
    df = pd.read_csv(OUT / f"{STEM}-start-year-sensitivity.csv")
    piv = df.pivot(index="Start year", columns="Portfolio", values="Money-weighted return")
    years = list(piv.index)
    lo = float(piv.min().min())
    hi = float(piv.max().max())
    x = lambda year: 85 + (year - years[0]) / (years[-1] - years[0] or 1) * 800
    y = lambda value: 65 + (hi - value) / (hi - lo or 1.0) * 330
    parts = ['<rect x="75" y="65" width="830" height="330" fill="white" stroke="#cbd5e1"/>']
    for i in range(5):
        yy = 65 + i * 82.5
        value = hi - i * (hi - lo) / 4
        parts += [
            f'<line x1="75" x2="905" y1="{yy}" y2="{yy}" stroke="#e2e8f0"/>',
            f'<text x="67" y="{yy + 4}" text-anchor="end" class="a">{value:.0%}</text>',
        ]
    for name in PORTFOLIOS:
        series = piv[name].dropna()
        pts = " ".join(f"{x(year):.1f},{y(value):.1f}" for year, value in series.items())
        parts.append(f'<polyline fill="none" stroke="{COLORS[name]}" stroke-width="2.1" points="{pts}"/>')
    for i, name in enumerate(PORTFOLIOS):
        xx = 80 + i % 2 * 390
        yy = 430 + i // 2 * 25
        parts += [
            f'<line x1="{xx}" x2="{xx + 24}" y1="{yy}" y2="{yy}" stroke="{COLORS[name]}" stroke-width="3"/>',
            f'<text x="{xx + 30}" y="{yy + 5}" class="l">{html.escape(name)}</text>',
        ]
    parts.append('<text x="75" y="510" class="a">Each point is a 10-year monthly contribution window by starting year</text>')
    (CHARTS / f"{STEM}-start-year-sensitivity.svg").write_text(
        base("Start-Year Sensitivity", "Annualized money-weighted return by 10-year start window", "".join(parts)),
        encoding="utf-8",
    )


def main() -> None:
    CHARTS.mkdir(exist_ok=True)
    df = pd.read_csv(OUT / f"{STEM}-daily.csv", parse_dates=["Date"])
    line_chart(
        df,
        "PortfolioValue",
        "Monthly Contributions Account Value",
        f"{STEM}-account-value.svg",
        include_contributions=True,
    )
    line_chart(
        df,
        "Drawdown",
        "Account-Value Drawdowns",
        f"{STEM}-drawdowns.svg",
        percent=True,
    )
    line_chart(
        df,
        "ProfitOverContributions",
        "Value Above Total Contributions",
        f"{STEM}-profit-over-contributions.svg",
    )
    sensitivity_chart()
    print(f"Wrote four SVG charts to {CHARTS}")


if __name__ == "__main__":
    main()
