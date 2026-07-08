#!/usr/bin/env python3
"""SPY, QQQ, and sector ETF monthly momentum rotation backtest.

The strategy ranks ETFs by trailing 126-trading-day adjusted-close return on
the last available trading day of each month. The selected holdings are applied
with a one-trading-day lag, so month-end signal information is not used for the
same close-to-close return interval.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DATA_SOURCE = "Yahoo Finance via yfinance"
START_DATE = "1998-12-01"
END_DATE = None
INITIAL_CAPITAL = 10_000.0
CASH_RETURN_ANNUAL = 0.0
TRADING_DAYS = 252
MOMENTUM_LOOKBACK_DAYS = 126
COST_BPS_SCENARIOS = (0.0, 5.0, 10.0)
BASE_COST_BPS = 5.0

ETF_UNIVERSE: dict[str, str] = {
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "XLK": "Technology sector ETF",
    "XLF": "Financial sector ETF",
    "XLV": "Health Care sector ETF",
    "XLY": "Consumer Discretionary sector ETF",
    "XLI": "Industrial sector ETF",
    "XLP": "Consumer Staples sector ETF",
    "XLE": "Energy sector ETF",
    "XLU": "Utilities sector ETF",
    "XLB": "Materials sector ETF",
}
TICKERS = tuple(ETF_UNIVERSE)

VARIANT_TOP_1 = "Top 1 momentum"
VARIANT_TOP_3 = "Top 3 momentum"
STRATEGY_VARIANTS = {
    VARIANT_TOP_1: 1,
    VARIANT_TOP_3: 3,
}

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"


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

    raise ValueError(f"Could not flatten yfinance MultiIndex columns for {ticker}")


def download_adjusted_ohlcv(
    ticker: str,
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
        raise RuntimeError(f"Downloaded data for {ticker} does not contain a Date column")

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Downloaded data for {ticker} is missing columns: {missing}")

    df = df[["Date", *required]].dropna(subset=["Close"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    return df


def load_adjusted_ohlcv(ticker: str, refresh: bool = False) -> pd.DataFrame:
    """Load cached adjusted OHLCV for one ETF, downloading it if necessary."""
    _ensure_dirs()
    cache_path = DATA_DIR / f"{ticker}.csv"
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path, parse_dates=["Date"])

    df = download_adjusted_ohlcv(ticker)
    df.to_csv(cache_path, index=False, float_format="%.10f")
    return df


def load_universe(refresh: bool = False) -> dict[str, pd.DataFrame]:
    """Load adjusted OHLCV for every ETF in the universe."""
    return {ticker: load_adjusted_ohlcv(ticker, refresh=refresh) for ticker in TICKERS}


def close_matrix(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build an inner-joined adjusted-close matrix for the full ETF universe."""
    close_series: list[pd.Series] = []
    missing_tickers = [ticker for ticker in TICKERS if ticker not in price_data]
    if missing_tickers:
        raise ValueError(f"price_data is missing tickers: {missing_tickers}")

    for ticker in TICKERS:
        df = price_data[ticker].copy()
        if "Date" not in df.columns or "Close" not in df.columns:
            raise ValueError(f"price_data[{ticker!r}] must contain Date and Close columns")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        close_series.append(df.set_index("Date")["Close"].astype(float).rename(ticker))

    close = pd.concat(close_series, axis=1, join="inner").sort_index()
    close = close.dropna(subset=list(TICKERS))
    if close.empty:
        raise ValueError("No overlapping adjusted-close rows across the ETF universe")
    return close


def compute_momentum(
    close: pd.DataFrame,
    lookback: int = MOMENTUM_LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Trailing trading-day return used for ETF ranking."""
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    return close / close.shift(lookback) - 1.0


def last_trading_day_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True on the last available trading day of each calendar month."""
    dates = pd.Series(index=index, data=index)
    last_dates = dates.groupby(index.to_period("M")).transform("max")
    return pd.Series(index=index, data=index == last_dates)


def strategy_start_date(momentum: pd.DataFrame) -> pd.Timestamp:
    """First date when every ETF has a valid momentum value."""
    valid = momentum.notna().all(axis=1)
    if not valid.any():
        raise ValueError("Not enough overlapping data for the momentum lookback")
    return pd.Timestamp(valid.index[valid][0])


def target_weights_from_momentum(
    momentum: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Create post-close target weights on month-end ranking dates."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if top_n > len(momentum.columns):
        raise ValueError("top_n cannot exceed the number of ETFs")

    weights = pd.DataFrame(np.nan, index=momentum.index, columns=momentum.columns)
    rebalance_mask = last_trading_day_mask(momentum.index)
    ranking_dates = momentum.index[rebalance_mask & momentum.notna().all(axis=1)]

    for date in ranking_dates:
        selected = list(momentum.loc[date].sort_values(ascending=False).index[:top_n])
        weights.loc[date, :] = 0.0
        weights.loc[date, selected] = 1.0 / top_n

    return weights.ffill().fillna(0.0)


def actual_weights_from_targets(target_weights: pd.DataFrame) -> pd.DataFrame:
    """Lag post-close targets by one trading day before applying returns."""
    return target_weights.shift(1).fillna(0.0)


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def prepare_strategy_frame(
    price_data: dict[str, pd.DataFrame],
    variant: str,
) -> pd.DataFrame:
    """Build prices, momentum, month-end targets, lagged weights, and returns."""
    if variant not in STRATEGY_VARIANTS:
        raise ValueError(f"Unknown strategy variant: {variant}")

    close = close_matrix(price_data)
    momentum = compute_momentum(close)
    start = strategy_start_date(momentum)
    close = close.loc[start:].copy()
    momentum = momentum.loc[start:].copy()
    returns = close.pct_change().fillna(0.0)

    target_weights = target_weights_from_momentum(momentum, STRATEGY_VARIANTS[variant])
    weights = actual_weights_from_targets(target_weights)

    result = pd.DataFrame(index=close.index)
    result["StrategyVariant"] = variant
    result["SignalUpdate"] = last_trading_day_mask(close.index).astype(int)
    result["CashReturn"] = CASH_RETURN_ANNUAL / TRADING_DAYS
    result["SPYReturn"] = returns["SPY"]

    for ticker in TICKERS:
        result[f"AdjustedClose_{ticker}"] = close[ticker]
        result[f"Momentum_{ticker}"] = momentum[ticker]
        result[f"TargetWeight_{ticker}"] = target_weights[ticker]
        result[f"Weight_{ticker}"] = weights[ticker]
        result[f"Return_{ticker}"] = returns[ticker]

    return result


def run_backtest(
    price_data: dict[str, pd.DataFrame],
    variant: str,
    cost_bps: float,
) -> pd.DataFrame:
    """Run one strategy variant for one transaction-cost scenario."""
    df = prepare_strategy_frame(price_data, variant)
    weight_cols = [f"Weight_{ticker}" for ticker in TICKERS]
    return_cols = [f"Return_{ticker}" for ticker in TICKERS]

    weights = df[weight_cols].rename(columns=lambda col: col.replace("Weight_", ""))
    returns = df[return_cols].rename(columns=lambda col: col.replace("Return_", ""))
    previous_weights = weights.shift(1).fillna(0.0)
    turnover = (weights - previous_weights).abs().sum(axis=1)

    df["Turnover"] = turnover
    df["TradingCost"] = turnover * (cost_bps / 10_000.0)
    df["StrategyReturnGross"] = (weights * returns).sum(axis=1)
    df["StrategyReturnNet"] = df["StrategyReturnGross"] - df["TradingCost"]
    df["StrategyEquity"] = INITIAL_CAPITAL * (1.0 + df["StrategyReturnNet"]).cumprod()
    df["BenchmarkEquity"] = INITIAL_CAPITAL * (1.0 + df["SPYReturn"]).cumprod()
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


def average_number_of_holdings(result: pd.DataFrame) -> float:
    weight_cols = [f"Weight_{ticker}" for ticker in TICKERS]
    return float((result[weight_cols] > 0).sum(axis=1).mean())


def time_in_market(result: pd.DataFrame) -> float:
    weight_cols = [f"Weight_{ticker}" for ticker in TICKERS]
    return float(result[weight_cols].sum(axis=1).mean())


def summarize(result: pd.DataFrame, variant: str, cost_bps: float) -> dict[str, object]:
    strategy_cagr = annualized_return(result["StrategyEquity"])
    benchmark_cagr = annualized_return(result["BenchmarkEquity"])
    strategy_mdd = float(result["StrategyDrawdown"].min())
    benchmark_mdd = float(result["BenchmarkDrawdown"].min())

    return {
        "Strategy variant": variant,
        "Cost bps": cost_bps,
        "ETF universe": " ".join(TICKERS),
        "Data source": DATA_SOURCE,
        "Execution model": "close-to-close same-close approximation",
        "Start date": result.index[0].date().isoformat(),
        "End date": result.index[-1].date().isoformat(),
        "Last available date": result.index[-1].date().isoformat(),
        "Momentum lookback trading days": MOMENTUM_LOOKBACK_DAYS,
        "Momentum definition": "adjusted_close[t] / adjusted_close[t-126] - 1",
        "Rebalance frequency": "Monthly, last available trading day",
        "Cash return annual": CASH_RETURN_ANNUAL,
        "Initial capital": INITIAL_CAPITAL,
        "CAGR": strategy_cagr,
        "Annualized volatility": annualized_volatility(result["StrategyReturnNet"]),
        "Sharpe ratio, 0% risk-free": sharpe_ratio(result["StrategyReturnNet"]),
        "Max drawdown": strategy_mdd,
        "Calmar ratio": calmar_ratio(strategy_cagr, strategy_mdd),
        "Time in market": time_in_market(result),
        "Rebalance count": int((result["Turnover"] > 0).sum()),
        "Total turnover": float(result["Turnover"].sum()),
        "Average number of holdings": average_number_of_holdings(result),
        "Final equity": float(result["StrategyEquity"].iloc[-1]),
        "Benchmark CAGR": benchmark_cagr,
        "Benchmark annualized volatility": annualized_volatility(result["SPYReturn"]),
        "Benchmark Sharpe ratio, 0% risk-free": sharpe_ratio(result["SPYReturn"]),
        "Benchmark max drawdown": benchmark_mdd,
        "Benchmark Calmar ratio": calmar_ratio(benchmark_cagr, benchmark_mdd),
        "Benchmark final equity": float(result["BenchmarkEquity"].iloc[-1]),
    }


def _weights_to_text(row: pd.Series, prefix: str) -> str:
    parts = []
    for ticker in TICKERS:
        value = float(row[f"{prefix}{ticker}"])
        if abs(value) > 1e-12:
            parts.append(f"{ticker}:{value:.6f}")
    return "; ".join(parts) if parts else "cash"


def build_trade_log(result: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = list(result.index)

    for idx, date in enumerate(dates):
        row = result.loc[date]
        if float(row["Turnover"]) <= 1e-12:
            continue

        previous_weights = {}
        for ticker in TICKERS:
            previous_weights[f"Weight_{ticker}"] = (
                float(result.iloc[idx - 1][f"Weight_{ticker}"]) if idx > 0 else 0.0
            )
        previous_row = pd.Series(previous_weights)
        signal_date = dates[idx - 1] if idx > 0 else date
        selected = [
            ticker
            for ticker in TICKERS
            if float(row[f"Weight_{ticker}"]) > 1e-12
        ]

        rows.append(
            {
                "Strategy variant": row["StrategyVariant"],
                "Rebalance date": signal_date.date().isoformat(),
                "Effective date": date.date().isoformat(),
                "Selected ETFs": " ".join(selected) if selected else "cash",
                "Previous weights": _weights_to_text(previous_row, "Weight_"),
                "New weights": _weights_to_text(row, "Weight_"),
                "Turnover": float(row["Turnover"]),
                "Cost bps": float(row["CostBps"]),
                "Cost as portfolio return": float(row["TradingCost"]),
                "Execution model": "close-to-close same-close approximation",
            }
        )

    return pd.DataFrame(rows)


def run_all_scenarios(
    price_data: dict[str, pd.DataFrame],
) -> dict[tuple[str, float], pd.DataFrame]:
    return {
        (variant, cost_bps): run_backtest(price_data, variant, cost_bps)
        for variant in STRATEGY_VARIANTS
        for cost_bps in COST_BPS_SCENARIOS
    }


def build_summary(price_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    results = run_all_scenarios(price_data)
    rows = [
        summarize(results[(variant, cost_bps)], variant, cost_bps)
        for variant in STRATEGY_VARIANTS
        for cost_bps in COST_BPS_SCENARIOS
    ]
    return pd.DataFrame(rows)


def write_outputs(price_data: dict[str, pd.DataFrame]) -> None:
    _ensure_dirs()
    results_by_scenario = run_all_scenarios(price_data)
    summary = pd.DataFrame(
        [
            summarize(results_by_scenario[(variant, cost_bps)], variant, cost_bps)
            for variant in STRATEGY_VARIANTS
            for cost_bps in COST_BPS_SCENARIOS
        ]
    )

    equity_columns = [
        "StrategyVariant",
        "SignalUpdate",
        "SPYReturn",
        "CashReturn",
        "StrategyReturnGross",
        "TradingCost",
        "Turnover",
        "StrategyReturnNet",
        "StrategyEquity",
        "BenchmarkEquity",
        "StrategyDrawdown",
        "BenchmarkDrawdown",
        "CostBps",
        *[f"Weight_{ticker}" for ticker in TICKERS],
    ]
    equity = pd.concat(
        [
            results_by_scenario[(variant, BASE_COST_BPS)][equity_columns]
            .reset_index()
            .rename(
                columns={
                    "index": "Date",
                    "StrategyVariant": "Strategy variant",
                    "StrategyReturnGross": "Daily strategy return gross",
                    "StrategyReturnNet": "Daily strategy return net",
                    "TradingCost": "Trading cost",
                    "Turnover": "Turnover",
                    "StrategyEquity": "Strategy equity",
                    "BenchmarkEquity": "Benchmark equity",
                    "StrategyDrawdown": "Strategy drawdown",
                    "BenchmarkDrawdown": "Benchmark drawdown",
                }
            )
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

    summary_path = OUTPUT_DIR / "spy-qqq-sector-momentum-rotation-summary.csv"
    equity_path = OUTPUT_DIR / "spy-qqq-sector-momentum-rotation-equity.csv"
    trades_path = OUTPUT_DIR / "spy-qqq-sector-momentum-rotation-trades.csv"
    summary.to_csv(summary_path, index=False, float_format="%.10f")
    equity.to_csv(equity_path, index=False, float_format="%.10f")
    trades.to_csv(trades_path, index=False, float_format="%.10f")

    print(f"Wrote {summary_path}")
    print(f"Wrote {equity_path}")
    print(f"Wrote {trades_path}")
    print()
    print("Base case: 5 bps per dollar traded / turnover")
    for variant in STRATEGY_VARIANTS:
        row = summary[
            (summary["Strategy variant"] == variant)
            & (summary["Cost bps"] == BASE_COST_BPS)
        ].iloc[0]
        print(
            f"  {variant}: CAGR {row['CAGR']:.2%}, "
            f"max drawdown {row['Max drawdown']:.2%}, "
            f"total turnover {row['Total turnover']:.1f}, "
            f"final equity ${row['Final equity']:,.0f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download fresh yfinance data instead of using data/{TICKER}.csv caches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    price_data = load_universe(refresh=args.refresh_data)
    write_outputs(price_data)


if __name__ == "__main__":
    main()
