#!/usr/bin/env python3
"""SPY RSI(2) mean-reversion backtest with an optional 200DMA filter.

The study compares a short-term RSI(2) state machine with the same state
machine gated by a 200-trading-day simple moving average trend filter. Returns
are modeled with the close-to-close same-close approximation used in the
earlier ReproQuant SPY moving-average studies.
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
RSI_WINDOW = 2
ENTRY_RSI_THRESHOLD = 10.0
EXIT_RSI_THRESHOLD = 70.0
SMA_WINDOW = 200
CASH_RETURN_ANNUAL = 0.0
TRADING_DAYS = 252
COST_BPS_SCENARIOS = (0.0, 5.0, 10.0)
BASE_COST_BPS = 5.0

VARIANT_RSI_ONLY = "RSI(2) only"
VARIANT_FILTER = "RSI(2) + 200DMA filter"
STRATEGY_VARIANTS = (VARIANT_RSI_ONLY, VARIANT_FILTER)

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
    RSI, SMA, strategy returns, and benchmark returns.
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


def compute_wilder_rsi(close: pd.Series, window: int = RSI_WINDOW) -> pd.Series:
    """Compute Wilder-style RSI from adjusted-close differences.

    The first valid average gain and loss are simple means of the first
    ``window`` close-to-close differences. Subsequent averages use Wilder's
    recursive smoothing:

    avg[t] = (avg[t - 1] * (window - 1) + value[t]) / window
    """
    if window <= 0:
        raise ValueError("window must be positive")

    close = pd.Series(close, dtype=float)
    rsi = pd.Series(np.nan, index=close.index, name=f"RSI{window}", dtype=float)
    if len(close) <= window:
        return rsi

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = float(gain.iloc[1 : window + 1].mean())
    avg_loss = float(loss.iloc[1 : window + 1].mean())

    def value_from_averages(mean_gain: float, mean_loss: float) -> float:
        if mean_loss == 0:
            if mean_gain == 0:
                return 50.0
            return 100.0
        rs = mean_gain / mean_loss
        return 100.0 - (100.0 / (1.0 + rs))

    rsi.iloc[window] = value_from_averages(avg_gain, avg_loss)

    for idx in range(window + 1, len(close)):
        current_gain = float(gain.iloc[idx])
        current_loss = float(loss.iloc[idx])
        avg_gain = (avg_gain * (window - 1) + current_gain) / window
        avg_loss = (avg_loss * (window - 1) + current_loss) / window
        rsi.iloc[idx] = value_from_averages(avg_gain, avg_loss)

    return rsi


def build_state_machine(
    entry_signal: pd.Series,
    exit_signal: pd.Series,
) -> pd.Series:
    """Build the post-close strategy state from entry and exit signals."""
    if len(entry_signal) != len(exit_signal):
        raise ValueError("entry_signal and exit_signal must have the same length")

    entry = entry_signal.fillna(False).astype(bool)
    exit_ = exit_signal.fillna(False).astype(bool)
    state_values: list[int] = []
    current_state = 0

    for enter, leave in zip(entry, exit_):
        if current_state == 0 and enter:
            current_state = 1
        elif current_state == 1 and leave:
            current_state = 0
        state_values.append(current_state)

    return pd.Series(state_values, index=entry_signal.index, name="SignalState", dtype=int)


def apply_strategy_rules(price_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Apply RSI-only or RSI-plus-filter rules to a prepared indicator frame."""
    if variant not in STRATEGY_VARIANTS:
        raise ValueError(f"Unknown strategy variant: {variant}")

    required = ["AdjustedClose", "RSI2", "SMA200"]
    missing = [col for col in required if col not in price_df.columns]
    if missing:
        raise ValueError(f"price_df is missing columns: {missing}")

    df = price_df.copy()
    if variant == VARIANT_RSI_ONLY:
        entry_signal = df["RSI2"] < ENTRY_RSI_THRESHOLD
        exit_signal = df["RSI2"] > EXIT_RSI_THRESHOLD
    else:
        has_sma = df["SMA200"].notna()
        entry_signal = (
            (df["RSI2"] < ENTRY_RSI_THRESHOLD)
            & has_sma
            & (df["AdjustedClose"] > df["SMA200"])
        )
        exit_signal = (
            (df["RSI2"] > EXIT_RSI_THRESHOLD)
            | (has_sma & (df["AdjustedClose"] < df["SMA200"]))
        )

    df["RawEntrySignal"] = entry_signal.fillna(False).astype(int)
    df["RawExitSignal"] = exit_signal.fillna(False).astype(int)
    df["SignalState"] = build_state_machine(entry_signal, exit_signal)

    # A state updated after close[t-1] is the modeled position for the
    # close[t-1] to close[t] return interval.
    df["Position"] = df["SignalState"].shift(1).fillna(0).astype(int)
    df["StrategyVariant"] = variant
    return df


def prepare_signals(price_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Add RSI(2), SMA200, state, lagged position, and SPY returns."""
    df = price_df.copy()
    if "Date" not in df.columns:
        raise ValueError("price_df must contain a Date column")
    if "Close" not in df.columns:
        raise ValueError("price_df must contain a Close column")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")
    df["AdjustedClose"] = df["Close"].astype(float)
    df["RSI2"] = compute_wilder_rsi(df["AdjustedClose"], RSI_WINDOW)
    df["SMA200"] = (
        df["AdjustedClose"]
        .rolling(SMA_WINDOW, min_periods=SMA_WINDOW)
        .mean()
    )
    df = apply_strategy_rules(df, variant)
    df["SPYReturn"] = df["AdjustedClose"].pct_change().fillna(0.0)
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


def holding_period_lengths(position: pd.Series) -> list[int]:
    """Return trading-day lengths for each modeled risk-on holding period."""
    lengths: list[int] = []
    current_length = 0
    for value in position.astype(int):
        if value == 1:
            current_length += 1
        elif current_length > 0:
            lengths.append(current_length)
            current_length = 0

    if current_length > 0:
        lengths.append(current_length)
    return lengths


def average_holding_days(position: pd.Series) -> float:
    lengths = holding_period_lengths(position)
    if not lengths:
        return 0.0
    return float(np.mean(lengths))


def summarize(result: pd.DataFrame, variant: str, cost_bps: float) -> dict[str, object]:
    strategy_cagr = annualized_return(result["StrategyEquity"])
    benchmark_cagr = annualized_return(result["BenchmarkEquity"])
    strategy_mdd = float(result["StrategyDrawdown"].min())
    benchmark_mdd = float(result["BenchmarkDrawdown"].min())
    position_changes = int(result["PositionChange"].sum())
    start_date = result.index[0].date().isoformat()
    end_date = result.index[-1].date().isoformat()
    first_valid_rsi = result.index[result["RSI2"].notna()]
    first_valid_sma = result.index[result["SMA200"].notna()]
    first_valid_rsi_date = (
        first_valid_rsi[0].date().isoformat() if len(first_valid_rsi) else ""
    )
    first_valid_sma_date = (
        first_valid_sma[0].date().isoformat() if len(first_valid_sma) else ""
    )

    return {
        "Strategy variant": variant,
        "Cost bps": cost_bps,
        "Ticker": TICKER,
        "Data source": DATA_SOURCE,
        "Execution model": "close-to-close same-close approximation",
        "Start date": start_date,
        "End date": end_date,
        "Last available date": end_date,
        "RSI window": RSI_WINDOW,
        "RSI formula": "Wilder smoothing on adjusted-close differences",
        "Entry RSI threshold": ENTRY_RSI_THRESHOLD,
        "Exit RSI threshold": EXIT_RSI_THRESHOLD,
        "SMA window": SMA_WINDOW,
        "First valid RSI date": first_valid_rsi_date,
        "First valid SMA200 date": first_valid_sma_date,
        "Metrics include indicator warmup": True,
        "Cash return annual": CASH_RETURN_ANNUAL,
        "Initial capital": INITIAL_CAPITAL,
        "CAGR": strategy_cagr,
        "Annualized volatility": annualized_volatility(result["StrategyReturnNet"]),
        "Sharpe ratio, 0% risk-free": sharpe_ratio(result["StrategyReturnNet"]),
        "Max drawdown": strategy_mdd,
        "Calmar ratio": calmar_ratio(strategy_cagr, strategy_mdd),
        "Time in market": float(result["Position"].mean()),
        "Position changes": position_changes,
        "Average holding days": average_holding_days(result["Position"]),
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
                "Signal adjusted close": float(signal_row["AdjustedClose"]),
                "Signal RSI2": float(signal_row["RSI2"]),
                "Signal SMA200": float(signal_row["SMA200"]),
                "Signal raw entry": int(signal_row["RawEntrySignal"]),
                "Signal raw exit": int(signal_row["RawExitSignal"]),
                "Signal state after close": int(signal_row["SignalState"]),
                "Execution model": "close-to-close same-close approximation",
                "Cost bps": float(row["CostBps"]),
                "Cost as portfolio return": float(row["TradingCost"]),
            }
        )

    return pd.DataFrame(rows)


def run_all_scenarios(price_df: pd.DataFrame) -> dict[tuple[str, float], pd.DataFrame]:
    return {
        (variant, cost_bps): run_backtest(price_df, variant, cost_bps)
        for variant in STRATEGY_VARIANTS
        for cost_bps in COST_BPS_SCENARIOS
    }


def build_summary(price_df: pd.DataFrame) -> pd.DataFrame:
    results = run_all_scenarios(price_df)
    rows = [
        summarize(results[(variant, cost_bps)], variant, cost_bps)
        for variant in STRATEGY_VARIANTS
        for cost_bps in COST_BPS_SCENARIOS
    ]
    return pd.DataFrame(rows)


def write_outputs(price_df: pd.DataFrame) -> None:
    _ensure_dirs()
    results_by_scenario = run_all_scenarios(price_df)
    summary = pd.DataFrame(
        [
            summarize(results_by_scenario[(variant, cost_bps)], variant, cost_bps)
            for variant in STRATEGY_VARIANTS
            for cost_bps in COST_BPS_SCENARIOS
        ]
    )

    equity_columns = [
        "StrategyVariant",
        "AdjustedClose",
        "RSI2",
        "SMA200",
        "RawEntrySignal",
        "RawExitSignal",
        "SignalState",
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
            for variant in STRATEGY_VARIANTS
        ],
        ignore_index=True,
    )
    trades = pd.concat(
        [
            build_trade_log(results_by_scenario[(variant, BASE_COST_BPS)])
            for variant in STRATEGY_VARIANTS
        ],
        ignore_index=True,
    )

    summary_path = OUTPUT_DIR / "spy-rsi2-200dma-filter-summary.csv"
    equity_path = OUTPUT_DIR / "spy-rsi2-200dma-filter-equity.csv"
    trades_path = OUTPUT_DIR / "spy-rsi2-200dma-filter-trades.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10f")
    equity.to_csv(equity_path, index=False, float_format="%.10f")
    trades.to_csv(trades_path, index=False, float_format="%.10f")

    print(f"Wrote {summary_path}")
    print(f"Wrote {equity_path}")
    print(f"Wrote {trades_path}")
    print()
    print("Base case: 5 bps per position change")
    for variant in STRATEGY_VARIANTS:
        row = summary[
            (summary["Strategy variant"] == variant)
            & (summary["Cost bps"] == BASE_COST_BPS)
        ].iloc[0]
        print(
            f"  {variant}: CAGR {row['CAGR']:.2%}, "
            f"max drawdown {row['Max drawdown']:.2%}, "
            f"position changes {int(row['Position changes'])}, "
            f"average holding days {row['Average holding days']:.1f}, "
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
