# Monthly Contributions in SPY, QQQ, and ETF Portfolios

This study compares fixed monthly contributions into four ETF portfolios:

- SPY 100%
- QQQ 100%
- 60% SPY / 40% IEF
- 40% SPY / 40% IEF / 15% GLD / 5% DBC

The script uses adjusted close data, invests a fixed contribution on each
month's first available trading day, and resets the portfolio to target weights
on that date. Weights drift with daily returns until the next monthly
contribution date. Transaction costs, taxes, cash drag between paycheck and
investment date, fractional share constraints, and tax-lot accounting are not
modeled.

Because the portfolio receives external cash flows, the main return metric is
money-weighted return. The output also includes a linked time-weighted CAGR,
account-value drawdown, total contributed, final value, and a 10-year
start-year sensitivity table.

## Run

```bash
pip install -r requirements.txt
python3 -B -m unittest discover -s . -p "test_*.py"
python3 backtest.py
python3 plot.py
```

By default, the script uses cached data in this study's own `data/` folder.
If a required ticker cache is missing, the script downloads it with yfinance.
Use `python3 backtest.py --refresh-data` to replace this study's local cache
with fresh Yahoo Finance/yfinance data.

Outputs are written to `outputs/`:

- `monthly-contributions-spy-qqq-portfolio-backtest-summary.csv`
- `monthly-contributions-spy-qqq-portfolio-backtest-daily.csv`
- `monthly-contributions-spy-qqq-portfolio-backtest-contributions.csv`
- `monthly-contributions-spy-qqq-portfolio-backtest-start-year-sensitivity.csv`

Charts are written to `charts/`:

- account value versus cumulative contributions
- account-value drawdowns
- value above total contributions
- 10-year start-year sensitivity
