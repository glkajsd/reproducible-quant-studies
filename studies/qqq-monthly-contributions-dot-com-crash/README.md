# QQQ Monthly Contributions Through the Dot-Com Crash

This study tests a fixed monthly contribution schedule that starts near the
dot-com bubble period:

- $500 into QQQ on each month's first available trading day
- $500 into SPY on the same dates as a broad-market comparison
- adjusted close data from Yahoo Finance via yfinance
- no transaction costs, taxes, bid/ask spreads, or fractional-share limits

The note separates two related ideas:

- QQQ's adjusted price path from its 2000 high to its later recovery.
- The account path for a monthly contribution investor who continued buying
  through that period.

Because the account receives external cash flows, the summary reports
money-weighted return, total contributed, final value, account-value drawdown,
worst dollar loss versus cumulative contributions, and the last date the
account was below cumulative contributions.

## Run

```bash
pip install -r requirements.txt
python3 -B -m unittest discover -s . -p "test_*.py"
python3 backtest.py
python3 plot.py
```

By default, the script uses cached QQQ and SPY data in this study's own
`data/` folder. If a required ticker cache is missing, the script downloads it
with yfinance. Use `python3 backtest.py --refresh-data` to replace this study's
local cache with fresh yfinance data.

Outputs are written to `outputs/`:

- `qqq-monthly-contributions-dot-com-crash-summary.csv`
- `qqq-monthly-contributions-dot-com-crash-daily.csv`
- `qqq-monthly-contributions-dot-com-crash-price-stress.csv`

Charts are written to `charts/`:

- account value versus cumulative contributions
- value above total contributions
- account-value drawdowns
- QQQ price drawdown versus QQQ contribution-account drawdown
