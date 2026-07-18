# SPY 200-Day Moving Average Signal Frequency Backtest

Published article:
[Daily vs Weekly vs Month-End SPY 200DMA Backtest](https://www.reproquant.com/studies/spy-200-day-moving-average-signal-frequency/).

Research question:

> How sensitive is the SPY 200-day moving average strategy to signal-check frequency?

This study compares three versions of the same SPY 200-day SMA rule:

- Daily: check the signal every trading day.
- Weekly: check the signal only on the last trading day of each week, then carry that signal forward.
- Month-end: check the signal only on the last trading day of each month, then carry that signal forward.

The moving average is always a 200-trading-day SMA. This is not a 200-week or 200-month moving average test.

## Method

- Data: SPY adjusted OHLCV from Yahoo Finance via `yfinance`.
- Cache: `data/SPY.csv` is used when present; pass `--refresh-data` to download fresh data.
- Price series: adjusted close.
- Initial capital: `$10,000`.
- Cash return: `0%`.
- Base transaction cost: `5 bps` per position change.
- Cost sensitivity: `0`, `5`, and `10` bps.
- Execution model: close-to-close same-close approximation, matching the earlier ReproQuant SPY moving-average studies.

Signal timing is intentionally lagged. A signal checked on day `t` becomes the modeled position for day `t+1`. The position for any return interval uses only day `t-1` or earlier information. The first 199 trading days have no valid SMA200 and cannot produce a risk-on signal.

## Reproduce

```bash
cd studies/spy-200-day-moving-average-signal-frequency
pip install -r requirements.txt
python3 -B -m unittest discover -s . -p "test_*.py"
python3 backtest.py
python3 plot.py
```

Refresh the local Yahoo Finance cache:

```bash
python3 backtest.py --refresh-data
```

## Outputs

| File | Purpose |
|---|---|
| `data/SPY.csv` | Cached adjusted OHLCV from yfinance |
| `outputs/spy-200dma-signal-frequency-summary.csv` | Summary metrics for daily, weekly, and month-end variants at 0/5/10 bps costs |
| `outputs/spy-200dma-signal-frequency-equity.csv` | Base-case daily equity, signal, position, returns, costs, and drawdowns |
| `outputs/spy-200dma-signal-frequency-trades.csv` | Base-case position-change log |
| `charts/spy-200dma-signal-frequency-equity-curve.svg` | Equity curve comparison |
| `charts/spy-200dma-signal-frequency-drawdowns.svg` | Drawdown comparison |
| `charts/spy-200dma-signal-frequency-position-changes.svg` | Position-change comparison |

## Disclaimer

Educational only. Not investment advice. The scripts are a reproducible research example and do not model taxes, market impact, real order routing, or suitability for any account.
