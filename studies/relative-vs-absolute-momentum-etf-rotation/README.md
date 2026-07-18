# Relative vs Absolute Momentum in ETF Rotation

This study compares two monthly ETF rotation rules on the same multi-asset
universe used by the existing cross-asset momentum note:

- SPY, EFA, EEM, IEF, GLD, DBC, VNQ
- 126-trading-day adjusted-close momentum
- month-end signal dates
- next-trading-day implementation
- drifted weights between rebalances
- 0, 5, and 10 bps turnover cost scenarios

The relative-momentum rule always holds the highest-ranked ETF for Top 1 or the
three highest-ranked ETFs for Top 3. The absolute-momentum rule first uses the
same ranking, then holds only selected ETFs whose momentum is greater than 0.
Selected non-positive momentum slots remain in cash. Cash return is 0%.

For Top 3 absolute momentum, selected ETF slots keep their original one-third
target weight. If only two selected ETFs have positive momentum, the portfolio
holds those two ETFs at one-third each and cash at one-third. The remaining ETF
weights are not rescaled to 100%.

Published article:
[Relative vs Absolute Momentum in ETF Rotation](https://www.reproquant.com/studies/relative-vs-absolute-momentum-etf-rotation/).

## Run

```bash
pip install -r requirements.txt
python3 -B -m unittest discover -s . -p "test_*.py"
python3 backtest.py
python3 plot.py
```

Use `python3 backtest.py --refresh-data` to replace the committed local Yahoo
Finance/yfinance adjusted-OHLCV cache.

Outputs are written to `outputs/`:

- `relative-vs-absolute-momentum-etf-rotation-summary.csv`
- `relative-vs-absolute-momentum-etf-rotation-equity.csv`
- `relative-vs-absolute-momentum-etf-rotation-trades.csv`

Charts are written to `charts/`:

- equity curve
- drawdowns
- cash weight
- turnover

The test suite includes checks for one-trading-day signal lag, complete-universe
date alignment, cash drift, turnover accounting, and the Top 3 absolute
momentum rule without rescaling.
