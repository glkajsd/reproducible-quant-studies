#!/usr/bin/env python3
"""Relative versus absolute momentum ETF rotation backtest."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ("SPY", "EFA", "EEM", "IEF", "GLD", "DBC", "VNQ")
ETF_ROLES = {
    "SPY": "US equities",
    "EFA": "developed-market equities",
    "EEM": "emerging-market equities",
    "IEF": "7-10 year US Treasuries",
    "GLD": "gold",
    "DBC": "broad commodities",
    "VNQ": "US REITs",
}
LOOKBACK = 126
INITIAL_CAPITAL = 10_000.0
COST_SCENARIOS = (0.0, 5.0, 10.0)
BASE_COST_BPS = 5.0
TRADING_DAYS = 252
CASH_RETURN = 0.0
DATA_SOURCE = "Yahoo Finance via yfinance"
START_DATE = "2000-01-01"
END_DATE_EXCLUSIVE = "2026-07-08"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
STEM = "relative-vs-absolute-momentum-etf-rotation"

PORTFOLIOS: dict[str, tuple[int, bool]] = {
    "Top 1 relative momentum": (1, False),
    "Top 1 absolute momentum": (1, True),
    "Top 3 relative momentum": (3, False),
    "Top 3 absolute momentum": (3, True),
    "Equal weight": (len(TICKERS), False),
}
BENCHMARK = "SPY buy and hold"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download(ticker: str) -> pd.DataFrame:
    import yfinance as yf

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
    if isinstance(df.columns, pd.MultiIndex):
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            df.columns = df.columns.get_level_values(0)
    df = df.rename_axis("Date").reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
    return df[cols].dropna(subset=["Close"]).sort_values("Date")


def load_universe(refresh: bool = False) -> dict[str, pd.DataFrame]:
    ensure_dirs()
    result = {}
    for ticker in TICKERS:
        path = DATA_DIR / f"{ticker}.csv"
        if refresh or not path.exists():
            download(ticker).to_csv(path, index=False, float_format="%.10f")
        result[ticker] = pd.read_csv(path, parse_dates=["Date"])
    return result


def close_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    missing = set(TICKERS) - set(data)
    if missing:
        raise ValueError(f"Missing tickers: {sorted(missing)}")
    series = []
    for ticker in TICKERS:
        df = data[ticker]
        if not {"Date", "Close"} <= set(df.columns):
            raise ValueError(f"{ticker} requires Date and Close")
        close = (
            df.assign(Date=pd.to_datetime(df["Date"]))
            .sort_values("Date")
            .drop_duplicates("Date", keep="last")
            .set_index("Date")["Close"]
            .astype(float)
            .rename(ticker)
        )
        series.append(close)
    close = pd.concat(series, axis=1, join="inner").dropna().sort_index()
    if close.empty:
        raise ValueError("No common valid dates across the complete ETF universe")
    return close


def compute_momentum(close: pd.DataFrame, lookback: int = LOOKBACK) -> pd.DataFrame:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    return close / close.shift(lookback) - 1.0


def last_trading_day_mask(index: pd.DatetimeIndex) -> pd.Series:
    dates = pd.Series(index=index, data=index)
    return dates.eq(dates.groupby(index.to_period("M")).transform("max"))


def common_analysis_window(
    close: pd.DataFrame, lookback: int = LOOKBACK
) -> tuple[pd.DataFrame, pd.DataFrame]:
    momentum = compute_momentum(close, lookback)
    valid = momentum.notna().all(axis=1)
    if not valid.any():
        raise ValueError("Insufficient common history for momentum")
    start = valid.index[valid][0]
    return close.loc[start:].copy(), momentum.loc[start:].copy()


def target_weights(
    momentum: pd.DataFrame,
    top_n: int,
    absolute: bool = False,
    ticker_order: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Month-end targets; absolute mode sends non-positive selected slots to cash."""
    if not 1 <= top_n <= len(momentum.columns):
        raise ValueError("Invalid top_n")
    out = pd.DataFrame(np.nan, index=momentum.index, columns=momentum.columns)
    dates = momentum.index[last_trading_day_mask(momentum.index) & momentum.notna().all(axis=1)]
    ticker_order = ticker_order or tuple(momentum.columns)
    if set(ticker_order) != set(momentum.columns):
        raise ValueError("ticker_order must match momentum columns")
    order = {ticker: i for i, ticker in enumerate(ticker_order)}
    for date in dates:
        ranked = sorted(momentum.columns, key=lambda t: (-float(momentum.at[date, t]), order[t]))
        out.loc[date] = 0.0
        for ticker in ranked[:top_n]:
            if not absolute or float(momentum.at[date, ticker]) > 0:
                out.loc[date, ticker] = 1.0 / top_n
    return out.ffill().fillna(0.0)


def portfolio_targets(momentum: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    if name == BENCHMARK:
        targets = pd.DataFrame(0.0, index=momentum.index, columns=TICKERS)
        targets["SPY"] = 1.0
        events = pd.Series(False, index=momentum.index)
        if len(events) > 1:
            events.iloc[1] = True
        return targets, events
    if name not in PORTFOLIOS:
        raise ValueError(f"Unknown portfolio: {name}")
    top_n, absolute = PORTFOLIOS[name]
    targets = target_weights(momentum, top_n, absolute=absolute)
    signal = last_trading_day_mask(momentum.index) & momentum.notna().all(axis=1)
    events = signal.shift(1, fill_value=False)
    return targets.shift(1).fillna(0.0), events


def simulate_weights(
    desired: pd.DataFrame, returns: pd.DataFrame, rebalance_events: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Apply targets on event dates and let ETF weights drift against cash."""
    pretrade_weights = pd.DataFrame(0.0, index=desired.index, columns=desired.columns)
    weights = pd.DataFrame(0.0, index=desired.index, columns=desired.columns)
    turnover = pd.Series(0.0, index=desired.index)
    gross = pd.Series(0.0, index=desired.index)
    pretrade = pd.Series(0.0, index=desired.columns)
    for date in desired.index:
        pretrade_weights.loc[date] = pretrade
        applied = desired.loc[date].copy() if rebalance_events.loc[date] else pretrade.copy()
        if rebalance_events.loc[date]:
            turnover.loc[date] = float((applied - pretrade).abs().sum())
        weights.loc[date] = applied
        gross.loc[date] = float((applied * returns.loc[date]).sum())
        ending = applied * (1.0 + returns.loc[date])
        denominator = 1.0 + gross.loc[date]
        pretrade = ending / denominator if denominator else applied
    return pretrade_weights, weights, turnover, gross


def _cash_weight(weights: pd.DataFrame) -> pd.Series:
    return (1.0 - weights.sum(axis=1)).clip(lower=0.0)


def prepare(data: dict[str, pd.DataFrame], name: str, cost_bps: float) -> pd.DataFrame:
    close, momentum = common_analysis_window(close_matrix(data))
    returns = close.pct_change().fillna(0.0)
    targets, events = portfolio_targets(momentum, name)
    pretrade_weights, weights, turnover, gross = simulate_weights(targets, returns, events)
    cost = turnover * cost_bps / 10_000
    net = gross - cost
    equity = INITIAL_CAPITAL * (1.0 + net).cumprod()

    frame = pd.DataFrame(index=close.index)
    frame["Portfolio"] = name
    frame["CostBps"] = cost_bps
    frame["SignalDate"] = last_trading_day_mask(close.index).astype(int)
    frame["RebalanceEvent"] = events.astype(int)
    for ticker in TICKERS:
        frame[f"AdjustedClose_{ticker}"] = close[ticker]
        frame[f"Momentum_{ticker}"] = momentum[ticker]
        frame[f"TargetWeight_{ticker}"] = targets[ticker]
        frame[f"PreTradeWeight_{ticker}"] = pretrade_weights[ticker]
        frame[f"ActualWeight_{ticker}"] = weights[ticker]
        frame[f"Return_{ticker}"] = returns[ticker]
    frame["TargetCashWeight"] = _cash_weight(targets)
    frame["PreTradeCashWeight"] = _cash_weight(pretrade_weights)
    frame["ActualCashWeight"] = _cash_weight(weights)
    frame["GrossReturn"] = gross
    frame["Turnover"] = turnover
    frame["TradingCost"] = cost
    frame["NetReturn"] = net
    frame["Equity"] = equity
    frame["Drawdown"] = equity / equity.cummax() - 1.0
    return frame


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / years) - 1.0)


def metrics(frame: pd.DataFrame) -> dict[str, object]:
    returns = frame["NetReturn"]
    volatility = returns.std(ddof=0)
    annualized_vol = float(volatility * math.sqrt(TRADING_DAYS))
    sharpe = float(returns.mean() / volatility * math.sqrt(TRADING_DAYS)) if volatility else np.nan
    growth = cagr(frame["Equity"])
    max_drawdown = float(frame["Drawdown"].min())
    weight_cols = [f"ActualWeight_{ticker}" for ticker in TICKERS]
    return {
        "Portfolio": frame["Portfolio"].iloc[0],
        "Cost bps": frame["CostBps"].iloc[0],
        "ETF universe": " ".join(TICKERS),
        "Data source": DATA_SOURCE,
        "Start date": frame.index[0].date().isoformat(),
        "End date": frame.index[-1].date().isoformat(),
        "Momentum lookback trading days": LOOKBACK,
        "Momentum definition": "adjusted_close[t] / adjusted_close[t-126] - 1",
        "Rebalance timing": "Month-end close target; next trading day effective",
        "Cash return annual": CASH_RETURN,
        "Initial capital": INITIAL_CAPITAL,
        "CAGR": growth,
        "Annualized volatility": annualized_vol,
        "Sharpe ratio, 0% risk-free": sharpe,
        "Max drawdown": max_drawdown,
        "Calmar ratio": growth / abs(max_drawdown) if max_drawdown < 0 else np.nan,
        "Total turnover": float(frame["Turnover"].sum()),
        "Rebalance count": int(frame["RebalanceEvent"].sum()),
        "Average number of ETF holdings": float((frame[weight_cols] > 1e-12).sum(axis=1).mean()),
        "Average cash weight": float(frame["ActualCashWeight"].mean()),
        "Final equity": float(frame["Equity"].iloc[-1]),
    }


def build_summary(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    names = [*PORTFOLIOS, BENCHMARK]
    return pd.DataFrame([metrics(prepare(data, name, cost)) for name in names for cost in COST_SCENARIOS])


def _weight_text(row: pd.Series, prefix: str, cash_key: str) -> str:
    parts = [f"{ticker}:{row[prefix + ticker]:.12f}" for ticker in TICKERS if row[prefix + ticker] > 1e-12]
    if row[cash_key] > 1e-12:
        parts.append(f"cash:{row[cash_key]:.12f}")
    return "; ".join(parts) or "cash"


def trade_log(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, (date, row) in enumerate(frame.iterrows()):
        if not row["RebalanceEvent"]:
            continue
        signal_date = frame.index[i - 1] if i else date
        rank_row = frame.iloc[i - 1] if i else row
        ranks = sorted(TICKERS, key=lambda t: (-rank_row[f"Momentum_{t}"], TICKERS.index(t)))
        log_row = {
            "Portfolio": row["Portfolio"],
            "Signal date": signal_date.date().isoformat(),
            "Effective date": date.date().isoformat(),
            "Momentum ranking": " > ".join(ranks),
            "Old weights": _weight_text(row, "PreTradeWeight_", "PreTradeCashWeight"),
            "New weights": _weight_text(row, "ActualWeight_", "ActualCashWeight"),
            "OldCashWeight": row["PreTradeCashWeight"],
            "NewCashWeight": row["ActualCashWeight"],
            "Turnover": row["Turnover"],
            "Cost bps": row["CostBps"],
            "Cost as portfolio return": row["TradingCost"],
        }
        for ticker in TICKERS:
            log_row[f"OldWeight_{ticker}"] = row[f"PreTradeWeight_{ticker}"]
            log_row[f"NewWeight_{ticker}"] = row[f"ActualWeight_{ticker}"]
            log_row[f"SignalMomentum_{ticker}"] = rank_row[f"Momentum_{ticker}"]
        rows.append(log_row)
    return pd.DataFrame(rows)


def write_outputs(data: dict[str, pd.DataFrame]) -> None:
    ensure_dirs()
    names = [*PORTFOLIOS, BENCHMARK]
    frames = {(name, cost): prepare(data, name, cost) for name in names for cost in COST_SCENARIOS}
    summary = pd.DataFrame([metrics(frames[name, cost]) for name in names for cost in COST_SCENARIOS])
    equity = pd.concat(
        [frames[name, BASE_COST_BPS].reset_index(names="Date") for name in names],
        ignore_index=True,
    )
    trades = pd.concat([trade_log(frames[name, BASE_COST_BPS]) for name in names], ignore_index=True)
    summary.to_csv(OUTPUT_DIR / f"{STEM}-summary.csv", index=False, float_format="%.10f")
    equity.to_csv(OUTPUT_DIR / f"{STEM}-equity.csv", index=False, float_format="%.10f")
    trades.to_csv(OUTPUT_DIR / f"{STEM}-trades.csv", index=False, float_format="%.10f")
    print(
        summary[summary["Cost bps"].eq(BASE_COST_BPS)][
            ["Portfolio", "CAGR", "Max drawdown", "Average cash weight", "Final equity"]
        ].to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()
    write_outputs(load_universe(args.refresh_data))


if __name__ == "__main__":
    main()
