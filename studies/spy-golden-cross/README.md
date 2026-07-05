# SPY Golden Cross Backtest (50/200 SMA Crossover)

This study reproduces a classic 50/200 SMA golden-cross crossover rule on SPY in Python.

## Rule

- Compute `SMA50[t]` and `SMA200[t]` from the adjusted close series.
- If `SMA50[t-1] > SMA200[t-1]`, the strategy is risk-on and holds SPY for day `t`.
- If `SMA50[t-1] <= SMA200[t-1]`, the strategy is risk-off and holds cash.
- The first 200 trading days do not produce a valid signal.

Golden crosses and death crosses are the entry and exit events. The daily
position is a state rule: stay invested while SMA50 remains above SMA200.

The model is a close-to-close approximation. A signal computed from one day's adjusted close is modeled as the target position for the return interval from that close to the next close. It does not simulate next-open fills, intraday fills, bid/ask queue position, or market impact.

## Data

The script downloads adjusted OHLCV for SPY from Yahoo Finance through `yfinance` and caches it to `data/SPY.csv`. Later runs use the cache unless `--refresh-data` is passed.

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

- `outputs/spy-golden-cross-summary.csv`: metrics for 0, 5, and 10 bps cost scenarios.
- `outputs/spy-golden-cross-equity.csv`: daily base-case equity, benchmark, signal, position, return, cost, and drawdown columns.
- `outputs/spy-golden-cross-trades.csv`: base-case position-change log.
- `charts/spy-golden-cross-equity-curve.svg`: strategy versus buy-and-hold equity curve.
- `charts/spy-golden-cross-drawdowns.svg`: strategy versus buy-and-hold drawdowns.
- `charts/spy-golden-cross-price-sma.svg`: adjusted close, SMA50, SMA200, and risk-on periods.

## Disclaimer

For educational purposes only. This is not financial advice, investment advice, or a trading signal.
