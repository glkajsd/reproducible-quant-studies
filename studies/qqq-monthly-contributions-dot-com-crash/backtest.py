#!/usr/bin/env python3
"""Monthly QQQ contribution backtest through the dot-com crash."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ("QQQ", "SPY")
DATA_SOURCE = "Yahoo Finance via yfinance"
START_DATE = "2000-01-01"
END_DATE_EXCLUSIVE = "2026-07-08"
MONTHLY_CONTRIBUTION = 500.0
TRADING_DAYS = 252
DOT_COM_HIGH_END = "2000-12-31"

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
STEM = "qqq-monthly-contributions-dot-com-crash"


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
    series = []
    for ticker in TICKERS:
        if ticker not in data:
            raise ValueError(f"Missing ticker: {ticker}")
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
    close = close.loc[close.index >= pd.Timestamp(START_DATE)]
    if close.empty:
        raise ValueError("No common valid dates across QQQ and SPY")
    return close


def first_trading_day_mask(index: pd.DatetimeIndex) -> pd.Series:
    dates = pd.Series(index=index, data=index)
    first = dates.groupby(index.to_period("M")).transform("min")
    return dates.eq(first)


def simulate_monthly_contributions(
    prices: pd.Series,
    ticker: str,
    monthly_contribution: float = MONTHLY_CONTRIBUTION,
) -> pd.DataFrame:
    if monthly_contribution <= 0:
        raise ValueError("monthly_contribution must be positive")
    contribution_dates = first_trading_day_mask(prices.index)
    returns = prices.pct_change().fillna(0.0)
    shares = 0.0
    total_contributed = 0.0
    rows: list[dict[str, object]] = []

    for date, price in prices.items():
        contribution = float(monthly_contribution if contribution_dates.loc[date] else 0.0)
        if contribution:
            shares += contribution / float(price)
            total_contributed += contribution
        value = shares * float(price)
        rows.append(
            {
                "Date": date,
                "Portfolio": f"{ticker} monthly contributions",
                "Ticker": ticker,
                "AdjustedClose": float(price),
                "AssetReturn": float(returns.loc[date]),
                "Contribution": contribution,
                "Shares": shares,
                "TotalContributed": total_contributed,
                "PortfolioValue": value,
                "ProfitOverContributions": value - total_contributed,
            }
        )
    frame = pd.DataFrame(rows).set_index("Date")
    frame["Drawdown"] = frame["PortfolioValue"] / frame["PortfolioValue"].cummax() - 1.0
    frame["PriceDrawdown"] = frame["AdjustedClose"] / frame["AdjustedClose"].cummax() - 1.0
    return frame


def money_weighted_return(cashflows: pd.Series, final_value: float) -> float:
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


def annualized_time_weighted_return(returns: pd.Series, index: pd.DatetimeIndex) -> float:
    years = (index[-1] - index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((1.0 + returns).prod() ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * math.sqrt(TRADING_DAYS))


def last_below_contributions(frame: pd.DataFrame) -> pd.Timestamp | None:
    below = frame.index[frame["ProfitOverContributions"] < 0]
    return below[-1] if len(below) else None


def durable_recovery_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    last_below = last_below_contributions(frame)
    if last_below is None:
        return frame.index[0]
    after = frame.index[frame.index > last_below]
    return after[0] if len(after) else None


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    final_value = float(frame["PortfolioValue"].iloc[-1])
    total_contributed = float(frame["TotalContributed"].iloc[-1])
    min_profit_date = frame["ProfitOverContributions"].idxmin()
    last_below = last_below_contributions(frame)
    recovery = durable_recovery_date(frame)
    returns = frame["AssetReturn"]
    return {
        "Portfolio": frame["Portfolio"].iloc[0],
        "Ticker": frame["Ticker"].iloc[0],
        "Data source": DATA_SOURCE,
        "Start date": frame.index[0].date().isoformat(),
        "End date": frame.index[-1].date().isoformat(),
        "Monthly contribution": MONTHLY_CONTRIBUTION,
        "Contribution timing": "First trading day of each month",
        "Contribution count": int((frame["Contribution"] > 0).sum()),
        "Total contributed": total_contributed,
        "Final value": final_value,
        "Profit over contributions": final_value - total_contributed,
        "Money-weighted return": money_weighted_return(frame["Contribution"], final_value),
        "Time-weighted CAGR": annualized_time_weighted_return(returns, frame.index),
        "Annualized volatility": annualized_volatility(returns),
        "Max account drawdown": float(frame["Drawdown"].min()),
        "Max price drawdown": float(frame["PriceDrawdown"].min()),
        "Worst dollar loss vs contributions": float(frame["ProfitOverContributions"].min()),
        "Worst dollar loss date": min_profit_date.date().isoformat(),
        "Last date below contributions": last_below.date().isoformat() if last_below is not None else "",
        "Durable recovery date": recovery.date().isoformat() if recovery is not None else "",
    }


def qqq_price_stress(close: pd.DataFrame) -> dict[str, object]:
    qqq = close["QQQ"]
    pre_end = pd.Timestamp(DOT_COM_HIGH_END)
    high_date = qqq.loc[:pre_end].idxmax()
    high = float(qqq.loc[high_date])
    after_high = qqq.loc[qqq.index > high_date]
    recovery = after_high[after_high >= high]
    recovery_date = recovery.index[0] if len(recovery) else None
    stress_window = after_high.loc[:recovery_date] if recovery_date is not None else after_high
    trough_date = stress_window.idxmin()
    trough = float(stress_window.loc[trough_date])
    return {
        "Metric": "QQQ dot-com price path",
        "Data source": DATA_SOURCE,
        "High date": high_date.date().isoformat(),
        "High adjusted close": high,
        "Trough date": trough_date.date().isoformat(),
        "Trough adjusted close": trough,
        "Price drawdown from high": trough / high - 1.0,
        "Recovery date": recovery_date.date().isoformat() if recovery_date is not None else "",
        "Years to recovery": (recovery_date - high_date).days / 365.25 if recovery_date is not None else np.nan,
    }


def build_outputs(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    close = close_matrix(data)
    frames = [
        simulate_monthly_contributions(close["QQQ"], "QQQ"),
        simulate_monthly_contributions(close["SPY"], "SPY"),
    ]
    summary = pd.DataFrame([summarize(frame) for frame in frames])
    daily = pd.concat([frame.reset_index() for frame in frames], ignore_index=True)
    stress = pd.DataFrame([qqq_price_stress(close)])
    return summary, daily, stress


def write_outputs(data: dict[str, pd.DataFrame]) -> None:
    ensure_dirs()
    summary, daily, stress = build_outputs(data)
    summary.to_csv(OUTPUT_DIR / f"{STEM}-summary.csv", index=False, float_format="%.10f")
    daily.to_csv(OUTPUT_DIR / f"{STEM}-daily.csv", index=False, float_format="%.10f")
    stress.to_csv(OUTPUT_DIR / f"{STEM}-price-stress.csv", index=False, float_format="%.10f")
    print(
        summary[
            [
                "Portfolio",
                "Total contributed",
                "Final value",
                "Money-weighted return",
                "Max account drawdown",
                "Worst dollar loss vs contributions",
                "Durable recovery date",
            ]
        ].to_string(index=False)
    )
    print(stress.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    write_outputs(load_universe(refresh=args.refresh_data))


if __name__ == "__main__":
    main()
