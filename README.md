# Reproducible Quant Studies

Python backtest code for the research notes published at
[Repro Quant](https://www.reproquant.com/).

The goal is small, inspectable, reproducible strategy research: each study keeps
the tested rule, data source, assumptions, tests, CSV outputs, and chart
generation code close together. These are research scripts, not trading signals.

## Published Studies

| Study | Article | Main topics |
|---|---|---|
| SPY 200-Day Moving Average Backtest | [Article](https://www.reproquant.com/studies/spy-200-day-moving-average/) | SPY, 200DMA, trend following |
| Daily vs Weekly vs Month-End SPY 200DMA | [Article](https://www.reproquant.com/studies/spy-200-day-moving-average-signal-frequency/) | signal frequency, moving average timing |
| SPY Golden Cross Backtest | [Article](https://www.reproquant.com/studies/spy-golden-cross/) | 50/200 SMA crossover, golden cross |
| SPY RSI(2) With 200DMA Trend Filter | [Article](https://www.reproquant.com/studies/spy-rsi-2-200-day-moving-average-filter/) | RSI(2), mean reversion, trend filter |
| SPY, QQQ, and Sector ETF Momentum Rotation | [Article](https://www.reproquant.com/studies/spy-qqq-sector-momentum-rotation-backtest/) | sector ETF rotation, cross-sectional momentum |
| Multi-Asset ETF Momentum Rotation | [Article](https://www.reproquant.com/studies/multi-asset-etf-momentum-rotation-backtest/) | cross-asset ETF momentum, tactical allocation |
| Relative vs Absolute Momentum in ETF Rotation | [Article](https://www.reproquant.com/studies/relative-vs-absolute-momentum-etf-rotation/) | relative momentum, absolute momentum, cash filter |

## Reproduce a Study

From a study folder:

```bash
pip install -r requirements.txt
python3 -B -m unittest discover -s . -p "test_*.py"
python3 backtest.py
python3 plot.py
```

Most studies use local CSV caches under `data/` when present. To refresh data
from Yahoo Finance through `yfinance`, run:

```bash
python3 backtest.py --refresh-data
```

## Repository Layout

```text
reproducible-quant-studies/
  studies/
    study-slug/
      README.md
      requirements.txt
      backtest.py
      plot.py
      test_backtest.py
      data/
      outputs/
      charts/
  shared/
```

## Data and Assumptions

The current studies use adjusted OHLCV data from Yahoo Finance through
`yfinance`. They are intended for educational reproduction, so the models keep
execution deliberately simple: adjusted-close return series, lagged signal
timing, simple transaction-cost sensitivity, and cash return assumptions stated
inside each study.

The scripts do not model taxes, real order routing, bid/ask spreads, market
impact, borrow constraints, account restrictions, or suitability for any
investor.

## License

Code in this repository is licensed under the Mozilla Public License 2.0. See
[`LICENSE`](LICENSE).

## Disclaimer

For learning and reference only. Nothing in this repository is financial advice,
investment advice, or a recommendation to buy or sell any security.
