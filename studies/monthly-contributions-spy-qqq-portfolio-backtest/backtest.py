#!/usr/bin/env python3
"""Monthly contribution backtest for several ETF portfolios.

The study invests a fixed dollar contribution on each month's first available
trading day. Contributions are allocated to target portfolio weights on that
date, and weights drift with daily adjusted-close returns until the next
contribution/rebalance date.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ("SPY", "QQQ", "IEF", "GLD", "DBC")
PORTFOLIOS: dict[str, dict[str, float]] = {
    "SPY 100%": {"SPY": 1.0},
    "QQQ 100%": {"QQQ": 1.0},
    "60/40 SPY IEF": {"SPY": 0.60, "IEF": 0.40},
    "Multi-asset 40/40/15/5": {"SPY": 0.40, "IEF": 0.40, "GLD": 0.15, "DBC": 0.05},
}
ETF_ROLES = {
    "SPY": "US large-cap equities",
    "QQQ": "Nasdaq-100 equities",
    "IEF": "7-10 year US Treasuries",
    "GLD": "gold",
    "DBC": "broad commodities",
}
DATA_SOURCE = "Yahoo Finance via yfinance"
START_DATE = "2000-01-01"
END_DATE_EXCLUSIVE = "2026-07-08"
INITIAL_CAPITAL = 0.0
MONTHLY_CONTRIBUTION = 500.0
TRADING_DAYS = 252
SENSITIVITY_YEARS = 10

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
STEM = "monthly-contributions-spy-qqq-portfolio-backtest"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    if ticker in df.columns.get_level_values(-1):
        return df.xs(ticker, axis=1, level=-1)
    df = df.copy()
    df.columns = df.columns.get_level_values(0)
    return df


def download(ticker: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance before refreshing data") from exc

    yf.set_tz_cache_location("/tmp/reproquant-yfinance-cache")
    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE_EXCLUSIVE,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")
    df = _flatten_yfinance_columns(df, ticker)
    df = df.rename_axis("Date").reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker} data missing columns: {missing}")
    return df[cols].dropna(subset=["Close"]).sort_values("Date")


def load_universe(refresh: bool = False) -> dict[str, pd.DataFrame]:
    ensure_dirs()
    data = {}
    for ticker in TICKERS:
        path = DATA_DIR / f"{ticker}.csv"
        if refresh or not path.exists():
            download(ticker).to_csv(path, index=False, float_format="%.10f")
        data[ticker] = pd.read_csv(path, parse_dates=["Date"])
    return data


def close_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    missing = set(TICKERS) - set(data)
    if missing:
        raise ValueError(f"Missing tickers: {sorted(missing)}")
    series = []
    for ticker in TICKERS:
        frame = data[ticker]
        if not {"Date", "Close"} <= set(frame.columns):
            raise ValueError(f"{ticker} requires Date and Close")
        close = (
            frame.assign(Date=pd.to_datetime(frame["Date"]))
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .set_index("Date")["Close"]
            .astype(float)
            .rename(ticker)
        )
        series.append(close)
    close = pd.concat(series, axis=1, join="inner").dropna().sort_index()
    if close.empty:
        raise ValueError("No common valid dates across the ETF universe")
    return close


def first_trading_day_mask(index: pd.DatetimeIndex) -> pd.Series:
    dates = pd.Series(index=index, data=index)
    first = dates.groupby(index.to_period("M")).transform("min")
    return dates.eq(first)


def _weight_series(weights: dict[str, float]) -> pd.Series:
    missing = set(weights) - set(TICKERS)
    if missing:
        raise ValueError(f"Unknown tickers in portfolio: {sorted(missing)}")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("Portfolio weights must sum to 1")
    return pd.Series({ticker: weights.get(ticker, 0.0) for ticker in TICKERS}, dtype=float)


def simulate_monthly_contributions(
    close: pd.DataFrame,
    name: str,
    weights: dict[str, float],
    monthly_contribution: float = MONTHLY_CONTRIBUTION,
    initial_capital: float = INITIAL_CAPITAL,
) -> pd.DataFrame:
    if monthly_contribution < 0 or initial_capital < 0:
        raise ValueError("Contributions and initial capital must be non-negative")
    target = _weight_series(weights)
    returns = close.pct_change().fillna(0.0)
    contribution_dates = first_trading_day_mask(close.index)

    holdings = pd.Series(0.0, index=TICKERS)
    rows: list[dict[str, object]] = []
    total_contributed = 0.0
    prior_value = float(initial_capital)

    for date in close.index:
        contribution = float(monthly_contribution if contribution_dates.loc[date] else 0.0)
        if date == close.index[0] and initial_capital:
            contribution += float(initial_capital)
        total_contributed += contribution

        pre_trade_value = prior_value + contribution
        if contribution_dates.loc[date] or (date == close.index[0] and initial_capital):
            pre_trade_weights = (
                holdings / prior_value if prior_value > 0 else pd.Series(0.0, index=TICKERS)
            )
            holdings = pre_trade_value * target
            turnover = float((target - pre_trade_weights).abs().sum()) if pre_trade_value else 0.0
            rebalance_event = True
        else:
            turnover = 0.0
            rebalance_event = False

        starting_value = float(holdings.sum())
        gross_return = (
            float((holdings * returns.loc[date]).sum() / starting_value)
            if starting_value > 0
            else 0.0
        )
        holdings = holdings * (1.0 + returns.loc[date])
        ending_value = float(holdings.sum())
        actual_weights = (
            holdings / ending_value if ending_value > 0 else pd.Series(0.0, index=TICKERS)
        )

        row: dict[str, object] = {
            "Date": date,
            "Portfolio": name,
            "Contribution": contribution,
            "TotalContributed": total_contributed,
            "StartingValueAfterContribution": starting_value,
            "GrossReturn": gross_return,
            "PortfolioValue": ending_value,
            "ProfitOverContributions": ending_value - total_contributed,
            "RebalanceEvent": int(rebalance_event),
            "Turnover": turnover,
        }
        for ticker in TICKERS:
            row[f"AdjustedClose_{ticker}"] = close.at[date, ticker]
            row[f"Return_{ticker}"] = returns.at[date, ticker]
            row[f"TargetWeight_{ticker}"] = target[ticker]
            row[f"ActualWeight_{ticker}"] = actual_weights[ticker]
        rows.append(row)
        prior_value = ending_value

    frame = pd.DataFrame(rows).set_index("Date")
    frame["Drawdown"] = frame["PortfolioValue"] / frame["PortfolioValue"].cummax() - 1.0
    frame["LinkedReturnIndex"] = (1.0 + frame["GrossReturn"]).cumprod()
    return frame


def annualized_time_weighted_return(returns: pd.Series, index: pd.DatetimeIndex) -> float:
    years = (index[-1] - index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((1.0 + returns).prod() ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * math.sqrt(TRADING_DAYS))


def money_weighted_return(cashflows: pd.Series, final_value: float) -> float:
    """Annualized XIRR from dated contributions and final ending value."""
    flows = cashflows[cashflows != 0].copy()
    if final_value <= 0 or flows.empty:
        return float("nan")
    flows = -flows
    end_date = cashflows.index[-1]
    flows.loc[end_date] = flows.get(end_date, 0.0) + final_value
    start = flows.index[0]
    years = np.array([(date - start).days / 365.25 for date in flows.index], dtype=float)
    amounts = flows.to_numpy(dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(amounts / np.power(1.0 + rate, years)))

    low, high = -0.9999, 10.0
    low_value, high_value = npv(low), npv(high)
    while low_value * high_value > 0 and high < 1_000:
        high *= 2
        high_value = npv(high)
    if low_value * high_value > 0:
        return float("nan")
    for _ in range(160):
        mid = (low + high) / 2
        mid_value = npv(mid)
        if abs(mid_value) < 1e-7:
            return mid
        if low_value * mid_value <= 0:
            high = mid
            high_value = mid_value
        else:
            low = mid
            low_value = mid_value
    return (low + high) / 2


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    returns = frame["GrossReturn"]
    twr = annualized_time_weighted_return(returns, frame.index)
    vol = annualized_volatility(returns)
    mwr = money_weighted_return(frame["Contribution"], float(frame["PortfolioValue"].iloc[-1]))
    total_contributed = float(frame["TotalContributed"].iloc[-1])
    final_value = float(frame["PortfolioValue"].iloc[-1])
    contribution_count = int((frame["Contribution"] > 0).sum())
    return {
        "Portfolio": frame["Portfolio"].iloc[0],
        "Data source": DATA_SOURCE,
        "Start date": frame.index[0].date().isoformat(),
        "End date": frame.index[-1].date().isoformat(),
        "Monthly contribution": MONTHLY_CONTRIBUTION,
        "Initial capital": INITIAL_CAPITAL,
        "Contribution timing": "First trading day of each month",
        "Contribution count": contribution_count,
        "Total contributed": total_contributed,
        "Final value": final_value,
        "Profit over contributions": final_value - total_contributed,
        "Money-weighted return": mwr,
        "Time-weighted CAGR": twr,
        "Annualized volatility": vol,
        "Sharpe ratio, 0% risk-free": returns.mean() / returns.std(ddof=0) * math.sqrt(TRADING_DAYS)
        if returns.std(ddof=0)
        else np.nan,
        "Max account drawdown": float(frame["Drawdown"].min()),
        "Total turnover": float(frame["Turnover"].sum()),
    }


def build_backtests(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close = close_matrix(data)
    frames = [
        simulate_monthly_contributions(close, name, weights)
        for name, weights in PORTFOLIOS.items()
    ]
    summary = pd.DataFrame([summarize(frame) for frame in frames])
    daily = pd.concat([frame.reset_index() for frame in frames], ignore_index=True)
    sensitivity = build_start_year_sensitivity(close)
    return summary, daily, sensitivity


def build_start_year_sensitivity(close: pd.DataFrame, years: int = SENSITIVITY_YEARS) -> pd.DataFrame:
    rows = []
    first_dates = close.index[first_trading_day_mask(close.index)]
    last_date = close.index[-1]
    for start_year in sorted(set(first_dates.year)):
        start_candidates = first_dates[first_dates.year == start_year]
        if len(start_candidates) == 0:
            continue
        start = start_candidates[0]
        end = start + pd.DateOffset(years=years)
        if end > last_date:
            continue
        window = close.loc[(close.index >= start) & (close.index < end)]
        if window.empty:
            continue
        for name, weights in PORTFOLIOS.items():
            frame = simulate_monthly_contributions(window, name, weights)
            info = summarize(frame)
            rows.append(
                {
                    "Portfolio": name,
                    "Start year": int(start_year),
                    "Window years": years,
                    "Start date": frame.index[0].date().isoformat(),
                    "End date": frame.index[-1].date().isoformat(),
                    "Contribution count": info["Contribution count"],
                    "Total contributed": info["Total contributed"],
                    "Final value": info["Final value"],
                    "Profit over contributions": info["Profit over contributions"],
                    "Money-weighted return": info["Money-weighted return"],
                    "Max account drawdown": info["Max account drawdown"],
                }
            )
    return pd.DataFrame(rows)


def contribution_log(daily: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Date",
        "Portfolio",
        "Contribution",
        "TotalContributed",
        "PortfolioValue",
        "RebalanceEvent",
        "Turnover",
    ]
    return daily[daily["Contribution"] > 0][cols].copy()


def write_outputs(data: dict[str, pd.DataFrame]) -> None:
    ensure_dirs()
    summary, daily, sensitivity = build_backtests(data)
    contributions = contribution_log(daily)
    summary.to_csv(OUTPUT_DIR / f"{STEM}-summary.csv", index=False, float_format="%.10f")
    daily.to_csv(OUTPUT_DIR / f"{STEM}-daily.csv", index=False, float_format="%.10f")
    sensitivity.to_csv(OUTPUT_DIR / f"{STEM}-start-year-sensitivity.csv", index=False, float_format="%.10f")
    contributions.to_csv(OUTPUT_DIR / f"{STEM}-contributions.csv", index=False, float_format="%.10f")
    print(
        summary[
            [
                "Portfolio",
                "Total contributed",
                "Final value",
                "Money-weighted return",
                "Max account drawdown",
            ]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    write_outputs(load_universe(refresh=args.refresh_data))


if __name__ == "__main__":
    main()
