import unittest

import numpy as np
import pandas as pd

import backtest


def universe(periods=190, dates=None, starts=None):
    dates = dates if dates is not None else pd.bdate_range("2020-01-02", periods=periods)
    starts = starts or {}
    result = {}
    for i, ticker in enumerate(backtest.TICKERS):
        start = starts.get(ticker, 0)
        d = dates[start:]
        values = 100 * (1 + (i + 1) * 0.0002) ** np.arange(len(d))
        result[ticker] = pd.DataFrame({"Date": d, "Close": values})
    return result


class BacktestTests(unittest.TestCase):
    def test_126_day_momentum(self):
        dates = pd.bdate_range("2020-01-01", periods=128)
        close = pd.DataFrame({"SPY": np.arange(100, 228)}, index=dates)
        momentum = backtest.compute_momentum(close)
        self.assertTrue(momentum.iloc[:126].isna().all().all())
        self.assertAlmostEqual(momentum.iloc[126, 0], 226 / 100 - 1)

    def test_last_actual_trading_day(self):
        dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2020-03-02"])
        self.assertEqual(
            backtest.last_trading_day_mask(pd.DatetimeIndex(dates)).tolist(),
            [False, True, False, True, True],
        )

    def test_relative_target_weights_use_fixed_tie_order(self):
        date = pd.Timestamp("2020-01-31")
        momentum = pd.DataFrame([[1, 1, 0.8, 0.7, 0.6, 0.5, 0.4]], index=[date], columns=backtest.TICKERS)
        top1 = backtest.target_weights(momentum, 1)
        top3 = backtest.target_weights(momentum, 3)
        self.assertEqual(top1.loc[date, "SPY"], 1)
        self.assertEqual(list(top3.loc[date][top3.loc[date] > 0].index), ["SPY", "EFA", "EEM"])

    def test_top3_absolute_does_not_renormalize_remaining_positive_assets(self):
        date = pd.Timestamp("2020-01-31")
        values = [[0.30, 0.20, 0.00, -0.01, -0.02, -0.03, -0.04]]
        momentum = pd.DataFrame(values, index=[date], columns=backtest.TICKERS)
        target = backtest.target_weights(momentum, 3, absolute=True)
        self.assertAlmostEqual(target.loc[date, "SPY"], 1 / 3)
        self.assertAlmostEqual(target.loc[date, "EFA"], 1 / 3)
        self.assertEqual(target.loc[date, "EEM"], 0)
        self.assertAlmostEqual(target.loc[date].sum(), 2 / 3)

    def test_top1_absolute_moves_non_positive_selection_to_cash(self):
        date = pd.Timestamp("2020-01-31")
        momentum = pd.DataFrame([[-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07]], index=[date], columns=backtest.TICKERS)
        target = backtest.target_weights(momentum, 1, absolute=True)
        self.assertEqual(target.loc[date].sum(), 0)

    def test_signal_lag_keeps_month_end_return_out_of_strategy(self):
        dates = pd.bdate_range("2020-01-30", periods=3)
        momentum = pd.DataFrame(1.0, index=dates, columns=backtest.TICKERS)
        targets, events = backtest.portfolio_targets(momentum, "Top 1 absolute momentum")
        self.assertEqual(targets.iloc[0].sum(), 0)
        self.assertFalse(events.iloc[1])
        self.assertTrue(events.iloc[2])
        self.assertEqual(targets.iloc[2]["SPY"], 1)

    def test_cash_weight_drifts_with_asset_returns(self):
        dates = pd.bdate_range("2020-01-02", periods=2)
        desired = pd.DataFrame({"SPY": [0.5, 0.5]}, index=dates)
        returns = pd.DataFrame({"SPY": [0.10, 0.00]}, index=dates)
        events = pd.Series([True, False], index=dates)
        pretrade, actual, turnover, gross = backtest.simulate_weights(desired, returns, events)
        self.assertEqual(turnover.iloc[0], 0.5)
        self.assertAlmostEqual(gross.iloc[0], 0.05)
        self.assertAlmostEqual(actual.iloc[0, 0], 0.5)
        self.assertAlmostEqual(pretrade.iloc[1, 0], 0.55 / 1.05)

    def test_common_window_inner_join_and_full_validity(self):
        data = universe(starts={"DBC": 8, "VNQ": 3})
        close = backtest.close_matrix(data)
        self.assertEqual(close.index[0], data["DBC"]["Date"].iloc[0])
        window, momentum = backtest.common_analysis_window(close)
        self.assertEqual(window.index[0], close.index[126])
        self.assertTrue(momentum.iloc[0].notna().all())

    def test_cost_scenarios_and_portfolios_are_complete(self):
        summary = backtest.build_summary(universe())
        self.assertEqual(set(summary["Cost bps"]), {0, 5, 10})
        self.assertEqual(set(summary["Portfolio"]), {*backtest.PORTFOLIOS, backtest.BENCHMARK})
        self.assertEqual(len(summary), 18)

    def test_trade_log_reproduces_turnover_without_cash_renormalization(self):
        frame = backtest.prepare(universe(periods=190), "Top 3 absolute momentum", 5)
        log = backtest.trade_log(frame)
        for _, row in log.iterrows():
            calculated = sum(
                abs(row[f"NewWeight_{ticker}"] - row[f"OldWeight_{ticker}"])
                for ticker in backtest.TICKERS
            )
            self.assertAlmostEqual(calculated, row["Turnover"], delta=1e-12)


if __name__ == "__main__":
    unittest.main()
