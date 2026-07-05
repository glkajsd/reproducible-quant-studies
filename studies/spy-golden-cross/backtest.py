#!/usr/bin/env python3
"""SPY golden-cross backtest (50-day / 200-day SMA crossover).

The strategy holds SPY when the previous 50-day simple moving average is above
the previous 200-day simple moving average. Entries and exits occur when that
relationship flips. Returns are modeled with a close-to-close approximation
using adjusted close data from Yahoo Finance via yfinance.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TICKER = "SPY"
DATA_SOURCE = "Yahoo Finance via yfinance"
START_DATE = "1993-01-29"
END_DATE = None
INITIAL_CAPITAL = 10_000.0
SMA_FAST = 50
SMA_SLOW = 200
CASH_RETURN_ANNUAL = 0.0
TRADING_DAYS = 252
COST_BPS_SCENARIOS = (0.0, 5.0, 10.0)
BASE_COST_BPS = 5.0

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CACHE_PATH = DATA_DIR / f"{TICKER}.csv"


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _flatten_yfinance_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a single-ticker OHLCV frame from yfinance output."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    if ticker in df.columns.get_level_values(-1):
        return df.xs(ticker, axis=1, level=-1)

    if len(set(df.columns.get_level_values(0))) <= 8:
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
        return df

    raise ValueError("Could not flatten yfinance MultiIndex columns")


def download_adjusted_ohlcv(
    ticker: str = TICKER,
    start: str = START_DATE,
    end: str | None = END_DATE,
) -> pd.DataFrame:
    """Download adjusted OHLCV from yfinance.

    yfinance with auto_adjust=True applies Yahoo's adjustment factor to OHLC
    prices. The resulting Close column is used as the adjusted close series for
    the strategy and benchmark returns.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance before downloading data") from exc

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=False,
    )
    if df.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")

    df = _flatten_yfinance_columns(df, ticker)
    df = df.rename_axis("Date").reset_index()
    if "Date" not in df.columns:
        raise RuntimeError("Downloaded data does not contain a Date column")

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing columns: {missing}")

    df = df[["Date", *required]].dropna(subset=["Close"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    return df


def load_adjusted_ohlcv(refresh: bool = False) -> pd.DataFrame:
    """Load cached adjusted OHLCV, downloading it if necessary."""
    _ensure_dirs()
    if CACHE_PATH.exists() and not refresh:
        return pd.read_csv(CACHE_PATH, parse_dates=["Date"])

    df = download_adjusted_ohlcv()
    df.to_csv(CACHE_PATH, index=False, float_format="%.10f")
    return df


def prepare_signals(price_df: pd.DataFrame) -> pd.DataFrame:
    """Add fast/slow SMAs, risk-on signal, lagged position, and SPY returns."""
    df = price_df.copy()
    if "Date" not in df.columns:
        raise ValueError("price_df must contain a Date column")
    if "Close" not in df.columns:
        raise ValueError("price_df must contain a Close column")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")
    df["AdjustedClose"] = df["Close"].astype(float)

    df["SMA50"] = (
        df["AdjustedClose"]
        .rolling(SMA_FAST, min_periods=SMA_FAST)
        .mean()
    )
    df["SMA200"] = (
        df["AdjustedClose"]
        .rolling(SMA_SLOW, min_periods=SMA_SLOW)
        .mean()
    )

    # Risk-on state: SMA50 above SMA200. Both must be valid (after warmup).
    df["Signal"] = (
        (df["SMA50"].notna())
        & (df["SMA200"].notna())
        & (df["SMA50"] > df["SMA200"])
    ).astype(int)

    # Close-to-close approximation: a signal computed from close[t-1] is modeled
    # as the target position for the close[t-1] to close[t] return interval.
    df["Position"] = df["Signal"].shift(1).fillna(0).astype(int)
    df["SPYReturn"] = df["AdjustedClose"].pct_change().fillna(0.0)
    return df


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def run_backtest(price_df: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Run the strategy for one transaction-cost scenario."""
    df = prepare_signals(price_df)
    previous_position = df["Position"].shift(1).fillna(0).astype(int)
    df["PreviousPosition"] = previous_position
    df["PositionChange"] = (df["Position"] - previous_position).abs()
    df["CashReturn"] = CASH_RETURN_ANNUAL / TRADING_DAYS
    df["StrategyReturnGross"] = (
        df["Position"] * df["SPYReturn"]
        + (1 - df["Position"]) * df["CashReturn"]
    )
    df["TradingCost"] = df["PositionChange"] * (cost_bps / 10_000.0)
    df["StrategyReturnNet"] = df["StrategyReturnGross"] - df["TradingCost"]
    df["StrategyEquity"] = INITIAL_CAPITAL * (1 + df["StrategyReturnNet"]).cumprod()
    df["BenchmarkEquity"] = INITIAL_CAPITAL * (1 + df["SPYReturn"]).cumprod()
    df["StrategyDrawdown"] = _drawdown(df["StrategyEquity"])
    df["BenchmarkDrawdown"] = _drawdown(df["BenchmarkEquity"])
    df["CostBps"] = cost_bps
    return df


def annualized_return(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=0) * math.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series) -> float:
    vol = returns.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return float(returns.mean() / vol * math.sqrt(TRADING_DAYS))


def calmar_ratio(cagr: float, max_drawdown: float) -> float:
    if max_drawdown >= 0 or np.isnan(max_drawdown):
        return float("nan")
    return cagr / abs(max_drawdown)


def summarize(result: pd.DataFrame, cost_bps: float) -> dict[str, object]:
    strategy_cagr = annualized_return(result["StrategyEquity"])
    benchmark_cagr = annualized_return(result["BenchmarkEquity"])
    strategy_mdd = float(result["StrategyDrawdown"].min())
    benchmark_mdd = float(result["BenchmarkDrawdown"].min())
    trade_count = int(result["PositionChange"].sum())
    start_date = result.index[0].date().isoformat()
    end_date = result.index[-1].date().isoformat()
    first_valid_sma = result.index[result["SMA50"].notna() & result["SMA200"].notna()]
    first_valid_sma_date = (
        first_valid_sma[0].date().isoformat() if len(first_valid_sma) else ""
    )

    return {
        "scenario": f"{cost_bps:g} bps",
        "ticker": TICKER,
        "data_source": DATA_SOURCE,
        "execution_model": "close-to-close same-close approximation",
        "start_date": start_date,
        "end_date": end_date,
        "last_available_date": end_date,
        "first_valid_sma_date": first_valid_sma_date,
        "metrics_include_sma_warmup": True,
        "sma_fast": SMA_FAST,
        "sma_slow": SMA_SLOW,
        "cash_return_annual": CASH_RETURN_ANNUAL,
        "cost_bps_per_position_change": cost_bps,
        "initial_capital": INITIAL_CAPITAL,
        "cagr": strategy_cagr,
        "annualized_volatility": annualized_volatility(result["StrategyReturnNet"]),
        "sharpe_ratio_0rf": sharpe_ratio(result["StrategyReturnNet"]),
        "max_drawdown": strategy_mdd,
        "calmar_ratio": calmar_ratio(strategy_cagr, strategy_mdd),
        "time_in_market": float(result["Position"].mean()),
        "number_of_trades": trade_count,
        "turnover_count": trade_count,
        "final_equity": float(result["StrategyEquity"].iloc[-1]),
        "benchmark_cagr": benchmark_cagr,
        "benchmark_annualized_volatility": annualized_volatility(result["SPYReturn"]),
        "benchmark_sharpe_ratio_0rf": sharpe_ratio(result["SPYReturn"]),
        "benchmark_max_drawdown": benchmark_mdd,
        "benchmark_calmar_ratio": calmar_ratio(benchmark_cagr, benchmark_mdd),
        "benchmark_final_equity": float(result["BenchmarkEquity"].iloc[-1]),
    }


def build_trade_log(result: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = list(result.index)
    for idx, date in enumerate(dates):
        row = result.loc[date]
        if row["PositionChange"] == 0:
            continue

        signal_date = dates[idx - 1] if idx > 0 else date
        signal_row = result.loc[signal_date]
        action = "BUY" if row["Position"] == 1 else "SELL"
        rows.append(
            {
                "Date": date.date().isoformat(),
                "Action": action,
                "PreviousPosition": int(row["PreviousPosition"]),
                "NewPosition": int(row["Position"]),
                "SignalDate": signal_date.date().isoformat(),
                "SignalAdjustedClose": float(signal_row["AdjustedClose"]),
                "SignalSMA50": float(signal_row["SMA50"]),
                "SignalSMA200": float(signal_row["SMA200"]),
                "ExecutionModel": "close-to-close same-close approximation",
                "CostBps": float(row["CostBps"]),
                "CostAsPortfolioReturn": float(row["TradingCost"]),
            }
        )

    return pd.DataFrame(rows)


def write_outputs(price_df: pd.DataFrame) -> None:
    _ensure_dirs()
    results_by_cost = {
        cost_bps: run_backtest(price_df, cost_bps)
        for cost_bps in COST_BPS_SCENARIOS
    }
    base = results_by_cost[BASE_COST_BPS]
    summary = pd.DataFrame(
        [summarize(results_by_cost[cost_bps], cost_bps) for cost_bps in COST_BPS_SCENARIOS]
    )

    equity_columns = [
        "AdjustedClose",
        "SMA50",
        "SMA200",
        "Signal",
        "Position",
        "PreviousPosition",
        "PositionChange",
        "SPYReturn",
        "CashReturn",
        "StrategyReturnGross",
        "TradingCost",
        "StrategyReturnNet",
        "StrategyEquity",
        "BenchmarkEquity",
        "StrategyDrawdown",
        "BenchmarkDrawdown",
    ]
    equity = base[equity_columns].reset_index()
    trades = build_trade_log(base)

    summary_path = OUTPUT_DIR / "spy-golden-cross-summary.csv"
    equity_path = OUTPUT_DIR / "spy-golden-cross-equity.csv"
    trades_path = OUTPUT_DIR / "spy-golden-cross-trades.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10f")
    equity.to_csv(equity_path, index=False, float_format="%.10f")
    trades.to_csv(trades_path, index=False, float_format="%.10f")

    base_summary = summary.loc[
        summary["cost_bps_per_position_change"] == BASE_COST_BPS
    ].iloc[0]
    print(f"Wrote {summary_path}")
    print(f"Wrote {equity_path}")
    print(f"Wrote {trades_path}")
    print()
    print("Base case: 5 bps per position change")
    print(f"  Date range: {base_summary['start_date']} to {base_summary['end_date']}")
    print(f"  Strategy CAGR: {base_summary['cagr']:.2%}")
    print(f"  Strategy max drawdown: {base_summary['max_drawdown']:.2%}")
    print(f"  Strategy Sharpe, 0% rf: {base_summary['sharpe_ratio_0rf']:.2f}")
    print(f"  Time in market: {base_summary['time_in_market']:.2%}")
    print(f"  Trades: {int(base_summary['number_of_trades'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh yfinance data instead of using data/SPY.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_df = load_adjusted_ohlcv(refresh=args.refresh_data)
    write_outputs(price_df)


if __name__ == "__main__":
    main()
