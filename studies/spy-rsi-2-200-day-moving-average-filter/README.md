# SPY RSI(2) Backtest With a 200-Day Moving Average Filter

Published article:
[SPY RSI(2) Backtest With 200DMA Trend Filter](https://www.reproquant.com/studies/spy-rsi-2-200-day-moving-average-filter/).

Research question:

> Does a 200-day moving average trend filter improve a short-term SPY RSI(2) mean reversion strategy?

This study compares two SPY strategy variants:

- RSI(2) only: a short-term mean reversion rule based only on RSI(2).
- RSI(2) + 200DMA filter: the same RSI(2) rule, but entries are allowed only when adjusted close is above the 200-day SMA.

The 200-day SMA is used as a trend filter. It is not a standalone entry trigger in this study. The filtered strategy still needs RSI(2) below the entry threshold before it can enter.

## Method

- Data: SPY adjusted OHLCV from Yahoo Finance via `yfinance`.
- Cache: `data/SPY.csv` is used when present; pass `--refresh-data` to download fresh data.
- Price series: adjusted close.
- Initial capital: `$10,000`.
- Cash return: `0%`.
- Base transaction cost: `5 bps` per position change.
- Cost sensitivity: `0`, `5`, and `10` bps.
- Execution model: close-to-close same-close approximation, matching the earlier ReproQuant SPY moving-average studies.

Signal timing is intentionally lagged. A signal state updated after the close of day `t` becomes the modeled position for day `t+1`. The position for any return interval uses only day `t-1` or earlier information.

## RSI Calculation

RSI(2) is computed directly in pandas/numpy from adjusted-close differences. TA-Lib is not required.

For window `n = 2`:

```text
delta[t] = adjusted_close[t] - adjusted_close[t - 1]
gain[t] = max(delta[t], 0)
loss[t] = max(-delta[t], 0)
```

The first valid average gain and loss are the simple means of the first `n` close-to-close gains and losses. After that, the script uses Wilder smoothing:

```text
avg_gain[t] = (avg_gain[t - 1] * (n - 1) + gain[t]) / n
avg_loss[t] = (avg_loss[t - 1] * (n - 1) + loss[t]) / n
RSI[t] = 100 - 100 / (1 + avg_gain[t] / avg_loss[t])
```

If both average gain and average loss are zero, RSI is set to `50`. If average loss is zero and average gain is positive, RSI is set to `100`.

## Strategy Rules

### RSI(2) only

- Raw entry signal: `RSI(2) < 10`.
- Raw exit signal: `RSI(2) > 70`.
- If currently in cash and the entry signal is true, the post-close state becomes invested.
- If currently invested and the exit signal is true, the post-close state becomes cash.
- Otherwise the previous state is carried forward.

### RSI(2) + 200DMA filter

- Raw entry signal: `RSI(2) < 10` and `adjusted close > SMA200`.
- Raw exit signal: `RSI(2) > 70` or `adjusted close < SMA200`.
- The first 199 trading days have no valid SMA200 and cannot produce a filtered entry.
- The state machine and one-day position shift are the same as the RSI-only variant.

For both variants:

```text
position[t] = signal_state[t - 1]
```

The first row position is `0`.

## Reproduce

```bash
cd studies/spy-rsi-2-200-day-moving-average-filter
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
| `outputs/spy-rsi2-200dma-filter-summary.csv` | Summary metrics for both variants at 0/5/10 bps costs |
| `outputs/spy-rsi2-200dma-filter-equity.csv` | Base-case daily indicators, signals, positions, returns, costs, equity, and drawdowns |
| `outputs/spy-rsi2-200dma-filter-trades.csv` | Base-case position-change log |
| `charts/spy-rsi2-200dma-filter-equity-curve.svg` | Equity curve comparison |
| `charts/spy-rsi2-200dma-filter-drawdowns.svg` | Drawdown comparison |
| `charts/spy-rsi2-200dma-filter-position-changes.svg` | Position-change comparison |

## Disclaimer

Educational only. Not investment advice. The scripts are a reproducible research example and do not model taxes, market impact, real order routing, intraday fills, or suitability for any account.
