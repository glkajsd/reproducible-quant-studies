# SPY QQQ Sector ETF Momentum Rotation Backtest

This study compares a monthly US equity ETF and sector momentum rotation rule
with SPY buy and hold. It rotates only among SPY, QQQ, and older sector ETFs.
It is not a stocks, bonds, gold, commodities, REITs, or international equities
multi-asset rotation strategy.

Published article:
[ETF Momentum Rotation Backtest: SPY, QQQ, and Sector ETFs](https://www.reproquant.com/studies/spy-qqq-sector-momentum-rotation-backtest/).

## Research question

What happens if a portfolio rotates monthly into the strongest ETF or strongest
three ETFs from a US equity universe of SPY, QQQ, and sector ETFs, using trailing
126-trading-day momentum, 5 bps base transaction costs, and SPY buy and hold as
the benchmark?

## US equity ETF universe

- SPY: S&P 500 ETF, used as the broad large-cap benchmark.
- QQQ: Nasdaq 100 ETF, a growth and technology-heavy index ETF.
- XLK: Technology sector ETF.
- XLF: Financial sector ETF.
- XLV: Health Care sector ETF.
- XLY: Consumer Discretionary sector ETF.
- XLI: Industrial sector ETF.
- XLP: Consumer Staples sector ETF.
- XLE: Energy sector ETF.
- XLU: Utilities sector ETF.
- XLB: Materials sector ETF.

XLC and XLRE are deliberately excluded for now because their available histories
are shorter than the older sector SPDR ETFs. The backtest starts only after every
included ETF has valid adjusted close data and a valid momentum lookback.

This is intentionally an equity-only universe. Bonds, gold, broad commodities,
REITs, and international equity ETFs are left for a separate multi-asset
rotation study.

## Momentum calculation

Momentum is the trailing 126-trading-day adjusted-close return:

```text
momentum[t] = adjusted_close[t] / adjusted_close[t - 126] - 1
```

Why 126 trading days? The study uses 126 trading days as a practical
approximation for six months, based on the common 252-trading-day convention for
one US trading year. Half of 252 is 126, so the lookback is a simple half-year
trading-day approximation.

This is not strict calendar-month momentum. It is not the same as measuring
performance from one calendar month-end to another. The 126-day lookback is a
fixed assumption, not an optimized parameter. A future robustness study could
compare 63-, 126-, and 252-day momentum windows, or use strict calendar
month-end returns.

## Monthly rebalance timing

The ranking is calculated only on the last available trading day of each month.
The target weights are determined after that close and then shifted by one
trading day before returns are applied:

```text
actual_weights = target_weights.shift(1)
```

The first row of actual weights is zero. This keeps signal-date information out
of the same close-to-close return interval.

## Strategy variants

- Top 1 momentum: hold the single ETF with the highest trailing 126-trading-day return.
- Top 3 momentum: hold the top three ETFs by the same ranking, equal weighted.

Cash earns 0%. Month-to-month holdings are forward-filled until the next month-end
ranking date.

Because the universe is equity-only, broad equity bear markets can still produce
large drawdowns. Sector rotation can change US equity exposure, but it does not
provide cross-asset diversification.

## Transaction costs

Costs are turnover-based:

```text
turnover[t] = sum(abs(weights[t] - weights[t - 1]))
trading_cost[t] = turnover[t] * cost_bps / 10000
```

For example, cash to 100% SPY has turnover 1, while a full switch from 100% SPY
to 100% QQQ has turnover 2. The summary output includes 0, 5, and 10 bps cost
sensitivity rows.

## Reproduce

```bash
pip install -r requirements.txt
python3 -B -m unittest discover -s . -p "test_*.py"
python3 backtest.py
python3 plot.py
```

Use `python3 backtest.py --refresh-data` to replace the local per-ETF CSV caches
with fresh yfinance downloads.

## Outputs

- `data/{TICKER}.csv`: cached adjusted OHLCV from yfinance for each ETF.
- `outputs/spy-qqq-sector-momentum-rotation-summary.csv`: summary metrics for
  both variants and 0/5/10 bps cost scenarios.
- `outputs/spy-qqq-sector-momentum-rotation-equity.csv`: base-case daily returns,
  costs, turnover, equity, drawdowns, and ETF weights.
- `outputs/spy-qqq-sector-momentum-rotation-trades.csv`: base-case rebalance log.
- `charts/spy-qqq-sector-momentum-rotation-equity-curve.svg`: equity chart.
- `charts/spy-qqq-sector-momentum-rotation-drawdowns.svg`: drawdown chart.
- `charts/spy-qqq-sector-momentum-rotation-turnover.svg`: base-case turnover chart.

Educational only. The code does not model taxes, bid/ask spreads, market impact,
fund changes, or real order fills.

## Possible follow-up

A separate study could test stocks, bonds, gold, commodities, REITs, and
international equities, for example across SPY, TLT or IEF, GLD, DBC or PDBC,
VNQ, EFA, and EEM. That would answer a different question from this US
equity-sector rotation test.
