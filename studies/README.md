# Studies

Each folder contains one reproducible backtest study. The web versions are
published at [Repro Quant](https://www.reproquant.com/studies/).

| Folder | Article |
|---|---|
| `spy-200-day-moving-average` | [SPY 200-Day Moving Average Backtest](https://www.reproquant.com/studies/spy-200-day-moving-average/) |
| `spy-200-day-moving-average-signal-frequency` | [Daily vs Weekly vs Month-End SPY 200DMA Backtest](https://www.reproquant.com/studies/spy-200-day-moving-average-signal-frequency/) |
| `spy-golden-cross` | [SPY Golden Cross Backtest](https://www.reproquant.com/studies/spy-golden-cross/) |
| `spy-rsi-2-200-day-moving-average-filter` | [SPY RSI(2) Backtest With 200DMA Trend Filter](https://www.reproquant.com/studies/spy-rsi-2-200-day-moving-average-filter/) |
| `spy-qqq-sector-momentum-rotation-backtest` | [ETF Momentum Rotation Backtest: SPY, QQQ, and Sector ETFs](https://www.reproquant.com/studies/spy-qqq-sector-momentum-rotation-backtest/) |
| `multi-asset-etf-momentum-rotation-backtest` | [Multi-Asset ETF Momentum Rotation Backtest](https://www.reproquant.com/studies/multi-asset-etf-momentum-rotation-backtest/) |
| `relative-vs-absolute-momentum-etf-rotation` | [Relative vs Absolute Momentum in ETF Rotation](https://www.reproquant.com/studies/relative-vs-absolute-momentum-etf-rotation/) |

Folder shape:

```text
study-slug/
  README.md
  requirements.txt
  backtest.py
  plot.py
  data/
  outputs/
  charts/
```
