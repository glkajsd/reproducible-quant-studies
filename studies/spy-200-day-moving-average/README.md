# SPY 200-Day Moving Average Backtest

This study reproduces a simple SPY 200-day moving-average rule in Python.

## Rule

- Compute `SMA200[t]` from the adjusted close series.
- If `adjusted_close[t-1] > SMA200[t-1]`, the strategy holds SPY for day `t`.
- Otherwise it holds cash.
- The first 199 trading days do not produce a valid signal.

The model is a close-to-close approximation. A signal computed from one day's adjusted close is modeled as the target position for the return interval from that close to the next close. It does not simulate next-open fills, intraday fills, bid/ask queue position, or market impact.

## Data

The script downloads adjusted OHLCV for SPY from Yahoo Finance through `yfinance` and caches it to `data/SPY.csv`. Later runs use the cache unless `--refresh-data` is passed.

Yahoo Finance and yfinance are suitable for educational reproduction, but they are not institutional-grade data sources.

## Cost Model

The base case deducts 5 bps each time the position changes. The summary output also includes 0 bps and 10 bps sensitivity rows. Cash return is 0% in the base study.

## Reproduce

```bash
pip install -r requirements.txt
python3 backtest.py
python3 plot.py
python3 -m unittest discover -s . -p "test_*.py"
```

## Outputs

- `outputs/spy-200dma-summary.csv`: metrics for 0, 5, and 10 bps cost scenarios. Metrics include the 200-trading-day SMA warmup period.
- `outputs/spy-200dma-equity.csv`: daily base-case equity, benchmark, signal, position, return, cost, and drawdown columns.
- `outputs/spy-200dma-trades.csv`: base-case position-change log.
- `charts/spy-200dma-equity-curve.svg`: strategy versus buy-and-hold equity curve.
- `charts/spy-200dma-drawdowns.svg`: strategy versus buy-and-hold drawdowns.
- `charts/spy-200dma-price-sma.svg`: adjusted close, SMA200, and risk-on periods.

## Disclaimer

For educational purposes only. This is not financial advice, investment advice, or a trading signal.
