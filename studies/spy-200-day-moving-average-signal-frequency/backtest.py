#!/usr/bin/env python3
"""SPY 200-day moving-average signal-frequency backtest.

The rule always uses a 200-trading-day simple moving average of SPY adjusted
close. The study changes only how often the signal is checked: daily, weekly,
or on the last trading day of each month. Returns are modeled with the same
close-to-close approximation used in the earlier SPY moving-average studies.
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
SMA_WINDOW = 200
CASH_RETURN_ANNUAL = 0.0
TRADING_DAYS = 252
COST_BPS_SCENARIOS = (0.0, 5.0, 10.0)
BASE_COST_BPS = 5.0

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
CACHE_PATH = DATA_DIR / f"{TICKER}.csv"

VARIANTS = {
    "daily": "Daily",
    "weekly": "Weekly",
    "month_end": "Month-end",
}


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
    """Download adjusted OHLCV from yfinance."""
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


def _last_trading_day_mask(index: pd.DatetimeIndex, frequency: str) -> pd.Series:
    dates = pd.Series(index=index, data=index)
    if frequency == "weekly":
        groups = index.to_period("W-FRI")
    elif frequency == "month_end":
        groups = index.to_period("M")
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    last_dates = dates.groupby(groups).transform("max")
    return pd.Series(index=index, data=index == last_dates)


def _update_mask(index: pd.DatetimeIndex, variant: str) -> pd.Series:
    if variant == "daily":
        return pd.Series(True, index=index)
    if variant in {"weekly", "month_end"}:
        return _last_trading_day_mask(index, variant)
    raise ValueError(f"Unknown strategy variant: {variant}")


def prepare_signals(price_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Add SMA, update mask, signal, lagged position, and SPY returns."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown strategy variant: {variant}")

    df = price_df.copy()
    if "Date" not in df.columns:
        raise ValueError("price_df must contain a Date column")
    if "Close" not in df.columns:
        raise ValueError("price_df must contain a Close column")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")
    df["AdjustedClose"] = df["Close"].astype(float)
    df["SMA200"] = (
        df["AdjustedClose"]
        .rolling(SMA_WINDOW, min_periods=SMA_WINDOW)
        .mean()
    )

    above_sma = (df["SMA200"].notna()) & (df["AdjustedClose"] > df["SMA200"])
    update_mask = _update_mask(df.index, variant)

    df["SignalUpdate"] = update_mask.astype(int)
    df["RawSignal"] = np.nan
    df.loc[update_mask, "RawSignal"] = above_sma.loc[update_mask].astype(int)
    df["Signal"] = df["RawSignal"].ffill().fillna(0).astype(int)

    # A signal checked at close[t-1] determines the modeled position for the
    # close[t-1] to close[t] return interval. This avoids using update-date
    # information in the same close-to-close return.
    df["Position"] = df["Signal"].shift(1).fillna(0).astype(int)
    df["SPYReturn"] = df["AdjustedClose"].pct_change().fillna(0.0)
    df["StrategyVariant"] = VARIANTS[variant]
    return df


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def run_backtest(price_df: pd.DataFrame, variant: str, cost_bps: float) -> pd.DataFrame:
    """Run one strategy variant for one transaction-cost scenario."""
    df = prepare_signals(price_df, variant)
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


def summarize(result: pd.DataFrame, variant: str, cost_bps: float) -> dict[str, object]:
    strategy_cagr = annualized_return(result["StrategyEquity"])
    benchmark_cagr = annualized_return(result["BenchmarkEquity"])
    strategy_mdd = float(result["StrategyDrawdown"].min())
    benchmark_mdd = float(result["BenchmarkDrawdown"].min())
    position_changes = int(result["PositionChange"].sum())
    start_date = result.index[0].date().isoformat()
    end_date = result.index[-1].date().isoformat()
    first_valid_sma = result.index[result["SMA200"].notna()]
    first_valid_sma_date = (
        first_valid_sma[0].date().isoformat() if len(first_valid_sma) else ""
    )

    return {
        "Strategy variant": VARIANTS[variant],
        "Cost bps": cost_bps,
        "Ticker": TICKER,
        "Data source": DATA_SOURCE,
        "Execution model": "close-to-close same-close approximation",
        "Start date": start_date,
        "End date": end_date,
        "Last available date": end_date,
        "First valid SMA200 date": first_valid_sma_date,
        "Metrics include SMA warmup": True,
        "SMA window": SMA_WINDOW,
        "Cash return annual": CASH_RETURN_ANNUAL,
        "Initial capital": INITIAL_CAPITAL,
        "CAGR": strategy_cagr,
        "Annualized volatility": annualized_volatility(result["StrategyReturnNet"]),
        "Sharpe ratio, 0% risk-free": sharpe_ratio(result["StrategyReturnNet"]),
        "Max drawdown": strategy_mdd,
        "Calmar ratio": calmar_ratio(strategy_cagr, strategy_mdd),
        "Time in market": float(result["Position"].mean()),
        "Position changes": position_changes,
        "Final equity": float(result["StrategyEquity"].iloc[-1]),
        "Benchmark CAGR": benchmark_cagr,
        "Benchmark annualized volatility": annualized_volatility(result["SPYReturn"]),
        "Benchmark Sharpe ratio, 0% risk-free": sharpe_ratio(result["SPYReturn"]),
        "Benchmark max drawdown": benchmark_mdd,
        "Benchmark Calmar ratio": calmar_ratio(benchmark_cagr, benchmark_mdd),
        "Benchmark final equity": float(result["BenchmarkEquity"].iloc[-1]),
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
                "Strategy variant": row["StrategyVariant"],
                "Date": date.date().isoformat(),
                "Action": action,
                "Previous position": int(row["PreviousPosition"]),
                "New position": int(row["Position"]),
                "Signal date": signal_date.date().isoformat(),
                "Signal update on date": int(signal_row["SignalUpdate"]),
                "Signal adjusted close": float(signal_row["AdjustedClose"]),
                "Signal SMA200": float(signal_row["SMA200"]),
                "Execution model": "close-to-close same-close approximation",
                "Cost bps": float(row["CostBps"]),
                "Cost as portfolio return": float(row["TradingCost"]),
            }
        )

    return pd.DataFrame(rows)


def run_all_scenarios(price_df: pd.DataFrame) -> dict[tuple[str, float], pd.DataFrame]:
    return {
        (variant, cost_bps): run_backtest(price_df, variant, cost_bps)
        for variant in VARIANTS
        for cost_bps in COST_BPS_SCENARIOS
    }


def build_summary(price_df: pd.DataFrame) -> pd.DataFrame:
    results = run_all_scenarios(price_df)
    rows = [
        summarize(results[(variant, cost_bps)], variant, cost_bps)
        for variant in VARIANTS
        for cost_bps in COST_BPS_SCENARIOS
    ]
    return pd.DataFrame(rows)


def write_outputs(price_df: pd.DataFrame) -> None:
    _ensure_dirs()
    results_by_scenario = run_all_scenarios(price_df)
    summary = pd.DataFrame(
        [
            summarize(results_by_scenario[(variant, cost_bps)], variant, cost_bps)
            for variant in VARIANTS
            for cost_bps in COST_BPS_SCENARIOS
        ]
    )

    equity_columns = [
        "StrategyVariant",
        "AdjustedClose",
        "SMA200",
        "SignalUpdate",
        "RawSignal",
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
        "CostBps",
    ]
    equity = pd.concat(
        [
            results_by_scenario[(variant, BASE_COST_BPS)][equity_columns]
            .reset_index()
            .rename(columns={"StrategyVariant": "Strategy variant"})
            for variant in VARIANTS
        ],
        ignore_index=True,
    )
    trades = pd.concat(
        [
            build_trade_log(results_by_scenario[(variant, BASE_COST_BPS)])
            for variant in VARIANTS
        ],
        ignore_index=True,
    )

    summary_path = OUTPUT_DIR / "spy-200dma-signal-frequency-summary.csv"
    equity_path = OUTPUT_DIR / "spy-200dma-signal-frequency-equity.csv"
    trades_path = OUTPUT_DIR / "spy-200dma-signal-frequency-trades.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10f")
    equity.to_csv(equity_path, index=False, float_format="%.10f")
    trades.to_csv(trades_path, index=False, float_format="%.10f")

    print(f"Wrote {summary_path}")
    print(f"Wrote {equity_path}")
    print(f"Wrote {trades_path}")
    print()
    print("Base case: 5 bps per position change")
    for variant_label in VARIANTS.values():
        row = summary[
            (summary["Strategy variant"] == variant_label)
            & (summary["Cost bps"] == BASE_COST_BPS)
        ].iloc[0]
        print(
            f"  {variant_label}: CAGR {row['CAGR']:.2%}, "
            f"max drawdown {row['Max drawdown']:.2%}, "
            f"position changes {int(row['Position changes'])}, "
            f"final equity ${row['Final equity']:,.0f}"
        )


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
