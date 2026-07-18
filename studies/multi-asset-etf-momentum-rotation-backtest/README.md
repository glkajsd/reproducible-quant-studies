# Multi-Asset ETF Momentum Rotation Backtest

This study ranks SPY, EFA, EEM, IEF, GLD, DBC, and VNQ by trailing
126-trading-day adjusted-close momentum on each calendar month's last actual
trading day. It compares Top 1 and equal-weighted Top 3 rotation with SPY buy
and hold and a monthly rebalanced equal-weight portfolio.

All series use the complete universe's inner-joined date range, and signals
begin only after momentum is valid for every ETF. Month-end targets take effect
one trading day later, and weights drift with returns between rebalances. Cash
earns 0%. Costs are `sum(abs(w_after-w_before))` times 0, 5, or 10 bps; a
complete ETF switch therefore has turnover 2.

Ties are broken by the fixed universe order shown above. IEF is used instead
of TLT to reduce dependence on long-duration risk; DBC is used instead of PDBC
for its longer history. These choices do not remove duration, roll-yield,
fund-structure, or survivorship limitations.

Published article:
[Multi-Asset ETF Momentum Rotation Backtest in Python](https://www.reproquant.com/studies/multi-asset-etf-momentum-rotation-backtest/).

## Run

```bash
pip install -r requirements.txt
python3 -B -m unittest discover -s . -p "test_*.py"
python3 backtest.py
python3 plot.py
```

Use `python3 backtest.py --refresh-data` to replace the committed local Yahoo
Finance/yfinance adjusted-OHLCV cache.

Outputs include summary, daily audit, and trade CSVs under `outputs/`, plus
equity, drawdown, allocation, and turnover SVGs under `charts/`.

`outputs/multi-asset-vs-sector-momentum-common-window.csv` rebases this study
and the existing sector-rotation study to the shared 2006-08-07 to 2026-07-07
window. It reads the sibling study's cached ETF prices and recomputes sector
Top 1 and Top 3 with the same drift, rebalance, turnover, delay, and cost
accounting used here. It does not change that study or its original results.
