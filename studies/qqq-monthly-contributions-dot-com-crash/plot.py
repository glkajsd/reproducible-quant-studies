#!/usr/bin/env python3
"""Generate SVG charts for the QQQ dot-com monthly contribution study."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
CHARTS = ROOT / "charts"
STEM = "qqq-monthly-contributions-dot-com-crash"
PORTFOLIOS = ("QQQ monthly contributions", "SPY monthly contributions")
COLORS = {
    "QQQ monthly contributions": "#7c3aed",
    "SPY monthly contributions": "#2563eb",
    "Total contributed": "#64748b",
    "QQQ price drawdown": "#a855f7",
    "QQQ monthly investment account drawdown": "#0f766e",
}


def base(title: str, subtitle: str, body: str, height: int = 540) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 980 {height}" role="img">
<title>{html.escape(title)}</title><desc>{html.escape(subtitle)}</desc>
<style>text{{font-family:Arial,sans-serif;fill:#0f172a}}.a{{font-size:12px;fill:#64748b}}.l{{font-size:13px}}</style>
<rect width="980" height="{height}" fill="white"/><text x="70" y="30" font-size="20" font-weight="700">{html.escape(title)}</text>
<text x="70" y="50" class="a">{html.escape(subtitle)}</text>{body}</svg>"""


def coords(dates: pd.DatetimeIndex, lo: float, hi: float):
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
    x, y = coords(dates, lo, hi)
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
    legend = list(portfolios)
    if include_contributions:
        series = contribution
        step = max(1, len(series) // 1000)
        pts = " ".join(f"{x(date):.1f},{y(value):.1f}" for date, value in series.iloc[::step].items())
        parts.append(f'<polyline fill="none" stroke="{COLORS["Total contributed"]}" stroke-width="2" stroke-dasharray="5 5" points="{pts}"/>')
        legend.append("Total contributed")
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
        base(title, "Adjusted close data; first trading day monthly contributions", "".join(parts)),
        encoding="utf-8",
    )


def qqq_drawdown_comparison(df: pd.DataFrame) -> None:
    qqq = df[df["Portfolio"].eq("QQQ monthly contributions")].set_index("Date")
    dates = pd.DatetimeIndex(qqq.index)
    series = {
        "QQQ price drawdown": qqq["PriceDrawdown"],
        "QQQ monthly investment account drawdown": qqq["Drawdown"],
    }
    values = pd.concat(series.values())
    lo, hi = float(values.min()), 0.0
    x, y = coords(dates, lo, hi)
    parts = ['<rect x="75" y="65" width="830" height="350" fill="white" stroke="#cbd5e1"/>']
    for i in range(5):
        yy = 65 + i * 87.5
        value = hi - i * (hi - lo) / 4
        parts += [
            f'<line x1="75" x2="905" y1="{yy}" y2="{yy}" stroke="#e2e8f0"/>',
            f'<text x="67" y="{yy + 4}" text-anchor="end" class="a">{value:.0%}</text>',
        ]
    for name, data in series.items():
        step = max(1, len(data) // 1000)
        pts = " ".join(f"{x(date):.1f},{y(value):.1f}" for date, value in data.iloc[::step].items())
        parts.append(f'<polyline fill="none" stroke="{COLORS[name]}" stroke-width="2.1" points="{pts}"/>')
    for i, name in enumerate(series):
        xx = 80 + i * 390
        yy = 452
        parts += [
            f'<line x1="{xx}" x2="{xx + 24}" y1="{yy}" y2="{yy}" stroke="{COLORS[name]}" stroke-width="3"/>',
            f'<text x="{xx + 30}" y="{yy + 5}" class="l">{html.escape(name)}</text>',
        ]
    parts.append('<text x="75" y="520" class="a">Price drawdown is not the same as monthly investment account drawdown</text>')
    (CHARTS / f"{STEM}-qqq-price-vs-account-drawdown.svg").write_text(
        base("QQQ Price Drawdown vs Monthly Investment Account Drawdown", "Monthly contributions change the account path", "".join(parts)),
        encoding="utf-8",
    )


def main() -> None:
    CHARTS.mkdir(exist_ok=True)
    df = pd.read_csv(OUT / f"{STEM}-daily.csv", parse_dates=["Date"])
    line_chart(
        df,
        "PortfolioValue",
        "QQQ and SPY Monthly Contribution Account Value",
        f"{STEM}-account-value.svg",
        include_contributions=True,
    )
    line_chart(
        df,
        "ProfitOverContributions",
        "Value Above Total Contributions",
        f"{STEM}-profit-over-contributions.svg",
    )
    line_chart(
        df,
        "Drawdown",
        "Account-Value Drawdowns",
        f"{STEM}-account-drawdowns.svg",
        percent=True,
    )
    qqq_drawdown_comparison(df)
    print(f"Wrote four SVG charts to {CHARTS}")


if __name__ == "__main__":
    main()
