import unittest

import numpy as np
import pandas as pd

import backtest


def synthetic_universe(close_by_ticker, dates=None):
    if dates is None:
        dates = pd.bdate_range("2020-01-02", periods=len(next(iter(close_by_ticker.values()))))
    price_data = {}
    for ticker in backtest.TICKERS:
        values = np.asarray(close_by_ticker[ticker], dtype=float)
        price_data[ticker] = pd.DataFrame(
            {
                "Date": pd.to_datetime(dates),
                "Open": values,
                "High": values,
                "Low": values,
                "Close": values,
                "Volume": 1_000_000,
            }
        )
    return price_data


def rising_universe(periods=150):
    base = {}
    for idx, ticker in enumerate(backtest.TICKERS):
        base[ticker] = np.linspace(100 + idx, 120 + idx, periods)
    return synthetic_universe(base)


class MomentumRotationBacktestTests(unittest.TestCase):
    def test_month_end_detection_uses_last_actual_trading_day(self):
        dates = pd.to_datetime(
            [
                "2020-01-29",
                "2020-01-30",
                "2020-02-03",
                "2020-02-27",
                "2020-02-28",
                "2020-03-02",
            ]
        )
        mask = backtest.last_trading_day_mask(pd.DatetimeIndex(dates))

        expected = pd.Series(
            [False, True, False, False, True, True],
            index=pd.DatetimeIndex(dates),
        )
        pd.testing.assert_series_equal(mask, expected)

    def test_momentum_calculation_is_trailing_return(self):
        dates = pd.bdate_range("2020-01-02", periods=5)
        close = pd.DataFrame({"SPY": [100, 110, 121, 100, 150]}, index=dates)
        momentum = backtest.compute_momentum(close, lookback=2)

        self.assertTrue(np.isnan(momentum["SPY"].iloc[0]))
        self.assertTrue(np.isnan(momentum["SPY"].iloc[1]))
        self.assertAlmostEqual(momentum["SPY"].iloc[2], 0.21)
        self.assertAlmostEqual(momentum["SPY"].iloc[3], -0.0909090909)
        self.assertAlmostEqual(momentum["SPY"].iloc[4], 0.2396694215)

    def test_top_1_selects_highest_momentum_etf(self):
        date = pd.Timestamp("2020-01-31")
        momentum = pd.DataFrame(
            {ticker: [idx / 100] for idx, ticker in enumerate(backtest.TICKERS)},
            index=[date],
        )
        weights = backtest.target_weights_from_momentum(momentum, top_n=1)

        best = backtest.TICKERS[-1]
        self.assertEqual(float(weights.loc[date, best]), 1.0)
        self.assertAlmostEqual(float(weights.loc[date].sum()), 1.0)

    def test_top_3_selects_highest_three_and_equal_weights(self):
        date = pd.Timestamp("2020-01-31")
        momentum = pd.DataFrame(
            {ticker: [idx / 100] for idx, ticker in enumerate(backtest.TICKERS)},
            index=[date],
        )
        weights = backtest.target_weights_from_momentum(momentum, top_n=3)

        selected = list(weights.loc[date][weights.loc[date] > 0].index)
        self.assertEqual(selected, list(backtest.TICKERS[-3:]))
        for ticker in selected:
            self.assertAlmostEqual(float(weights.loc[date, ticker]), 1 / 3)

    def test_actual_weights_are_shifted_targets(self):
        dates = pd.bdate_range("2020-01-30", periods=3)
        targets = pd.DataFrame(0.0, index=dates, columns=backtest.TICKERS)
        targets.loc[dates[0], "SPY"] = 1.0
        targets.loc[dates[1], "QQQ"] = 1.0
        actual = backtest.actual_weights_from_targets(targets)

        self.assertAlmostEqual(float(actual.iloc[0].sum()), 0.0)
        self.assertAlmostEqual(float(actual.loc[dates[1], "SPY"]), 1.0)
        self.assertAlmostEqual(float(actual.loc[dates[2], "QQQ"]), 1.0)

    def test_monthly_holdings_forward_fill_between_rebalances(self):
        dates = pd.bdate_range("2020-01-02", "2020-02-28")
        momentum = pd.DataFrame(0.01, index=dates, columns=backtest.TICKERS)
        momentum["SPY"] = 0.20
        momentum.loc[dates >= "2020-02-03", "QQQ"] = 0.30

        targets = backtest.target_weights_from_momentum(momentum, top_n=1)
        jan_end = pd.Timestamp("2020-01-31")

        self.assertAlmostEqual(float(targets.loc[jan_end, "SPY"]), 1.0)
        self.assertAlmostEqual(float(targets.loc[pd.Timestamp("2020-02-03"), "SPY"]), 1.0)
        self.assertAlmostEqual(float(targets.loc[pd.Timestamp("2020-02-10"), "SPY"]), 1.0)
        self.assertAlmostEqual(float(targets.loc[pd.Timestamp("2020-02-10")].sum()), 1.0)

    def test_strategy_start_date_requires_all_momentum_valid(self):
        price_data = rising_universe(periods=140)
        frame = backtest.prepare_strategy_frame(price_data, backtest.VARIANT_TOP_1)
        close = backtest.close_matrix(price_data)
        momentum = backtest.compute_momentum(close)
        expected_start = momentum.index[momentum.notna().all(axis=1)][0]

        self.assertEqual(frame.index[0], expected_start)
        self.assertTrue(frame[[f"Momentum_{ticker}" for ticker in backtest.TICKERS]].iloc[0].notna().all())

    def test_turnover_cost_only_when_weights_change(self):
        dates = pd.bdate_range("2020-01-30", periods=4)
        targets = pd.DataFrame(0.0, index=dates, columns=backtest.TICKERS)
        targets.loc[dates[0]:, "SPY"] = 1.0
        weights = backtest.actual_weights_from_targets(targets)
        previous = weights.shift(1).fillna(0.0)
        turnover = (weights - previous).abs().sum(axis=1)
        cost = turnover * 0.0005

        self.assertEqual(list(turnover), [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(int((cost > 0).sum()), 1)

    def test_full_switch_turnover_is_two(self):
        weights = pd.DataFrame(0.0, index=pd.bdate_range("2020-01-02", periods=2), columns=backtest.TICKERS)
        weights.iloc[0, weights.columns.get_loc("SPY")] = 1.0
        weights.iloc[1, weights.columns.get_loc("QQQ")] = 1.0
        previous = weights.shift(1).fillna(0.0)
        turnover = (weights - previous).abs().sum(axis=1)

        self.assertAlmostEqual(float(turnover.iloc[1]), 2.0)

    def test_cost_sensitivity_rows_exist_and_are_reproducible(self):
        price_data = rising_universe(periods=150)
        summary_1 = backtest.build_summary(price_data)
        summary_2 = backtest.build_summary(price_data)

        expected_costs = {0.0, 5.0, 10.0}
        self.assertEqual(set(summary_1["Cost bps"]), expected_costs)
        self.assertEqual(set(summary_1["Strategy variant"]), set(backtest.STRATEGY_VARIANTS))
        self.assertEqual(len(summary_1), len(backtest.STRATEGY_VARIANTS) * len(expected_costs))
        pd.testing.assert_frame_equal(summary_1, summary_2)


if __name__ == "__main__":
    unittest.main()
