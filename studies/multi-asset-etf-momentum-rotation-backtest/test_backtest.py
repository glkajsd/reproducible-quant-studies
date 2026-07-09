import unittest

import numpy as np
import pandas as pd

import backtest


def universe(periods=180, dates=None, starts=None):
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
        m = backtest.compute_momentum(close)
        self.assertTrue(m.iloc[:126].isna().all().all())
        self.assertAlmostEqual(m.iloc[126, 0], 226 / 100 - 1)

    def test_last_actual_trading_day(self):
        dates = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-27", "2020-02-28", "2020-03-02"])
        self.assertEqual(backtest.last_trading_day_mask(pd.DatetimeIndex(dates)).tolist(),
                         [False, True, False, True, True])

    def test_top1_top3_equal_weight_and_tie_order(self):
        date = pd.Timestamp("2020-01-31")
        m = pd.DataFrame([[1, 1, .8, .7, .6, .5, .4]], index=[date], columns=backtest.TICKERS)
        top1 = backtest.target_weights(m, 1)
        top3 = backtest.target_weights(m, 3)
        equal = backtest.target_weights(m, 7)
        self.assertEqual(top1.loc[date, "SPY"], 1)
        self.assertEqual(list(top3.loc[date][top3.loc[date] > 0].index), ["SPY", "EFA", "EEM"])
        np.testing.assert_allclose(equal.loc[date], np.repeat(1 / 7, 7))

    def test_signal_lag_and_initial_cash(self):
        dates = pd.bdate_range("2020-01-30", periods=3)
        target = pd.DataFrame(0.0, index=dates, columns=backtest.TICKERS)
        target.loc[:, "SPY"] = 1
        actual = backtest.actual_weights(target)
        self.assertEqual(actual.iloc[0].sum(), 0)
        self.assertEqual(actual.iloc[1]["SPY"], 1)

    def test_targets_hold_during_month(self):
        dates = pd.bdate_range("2020-01-01", "2020-02-28")
        m = pd.DataFrame(0.0, index=dates, columns=backtest.TICKERS)
        m["SPY"] = 1
        target = backtest.target_weights(m, 1)
        self.assertTrue((target.loc["2020-02-03":"2020-02-27", "SPY"] == 1).all())

    def test_common_window_inner_join_and_full_validity(self):
        data = universe(starts={"DBC": 8, "VNQ": 3})
        close = backtest.close_matrix(data)
        self.assertEqual(close.index[0], data["DBC"]["Date"].iloc[0])
        window, momentum = backtest.common_analysis_window(close)
        self.assertEqual(window.index[0], close.index[126])
        self.assertTrue(momentum.iloc[0].notna().all())

    def test_turnover_cash_entry_and_full_switch(self):
        dates = pd.bdate_range("2020-01-01", periods=3)
        w = pd.DataFrame(0.0, index=dates, columns=backtest.TICKERS)
        w.loc[dates[1], "SPY"] = 1
        w.loc[dates[2], "EFA"] = 1
        turnover = w.diff().abs().sum(axis=1)
        self.assertEqual(turnover.iloc[1], 1)
        self.assertEqual(turnover.iloc[2], 2)

    def test_cost_scenarios_complete_reproducible(self):
        data = universe()
        a, b = backtest.build_summary(data), backtest.build_summary(data)
        self.assertEqual(set(a["Cost bps"]), {0, 5, 10})
        self.assertEqual(set(a["Portfolio"]), {*backtest.PORTFOLIOS, backtest.BENCHMARK})
        self.assertEqual(len(a), 12)
        pd.testing.assert_frame_equal(a, b)

    def test_equal_weight_benchmark_is_lagged(self):
        data = universe()
        close, momentum = backtest.common_analysis_window(backtest.close_matrix(data))
        targets, events = backtest.portfolio_targets(momentum, "Equal weight")
        returns = close.pct_change().fillna(0)
        _, weights, _, _ = backtest.simulate_weights(targets, returns, events)
        signal_dates = targets.index[backtest.last_trading_day_mask(targets.index)]
        first_signal = signal_dates[0]
        loc = targets.index.get_loc(first_signal) + 1
        self.assertEqual(weights.iloc[loc - 1].sum(), 0)
        self.assertAlmostEqual(weights.iloc[loc].sum(), 1)

    def test_equal_weight_rebalances_after_drift(self):
        data = universe(periods=190)
        frame = backtest.prepare(data, "Equal weight", 5)
        self.assertGreater((frame["Turnover"] > 0).sum(), 1)

    def test_top3_drifts_and_rebalances_from_pretrade_weights(self):
        dates = pd.bdate_range("2020-01-02", periods=4)
        columns = backtest.TICKERS[:3]
        desired = pd.DataFrame(1 / 3, index=dates, columns=columns)
        returns = pd.DataFrame(
            [[0.10, 0.00, -0.05], [0.02, -0.01, 0.00], [0, 0, 0], [0, 0, 0]],
            index=dates, columns=columns,
        )
        events = pd.Series([True, False, True, False], index=dates)
        pretrade, actual, turnover, _ = backtest.simulate_weights(desired, returns, events)

        self.assertFalse(np.allclose(actual.iloc[1], np.repeat(1 / 3, 3)))
        np.testing.assert_allclose(actual.iloc[1], pretrade.iloc[1])
        expected = float((desired.iloc[2] - pretrade.iloc[2]).abs().sum())
        self.assertAlmostEqual(turnover.iloc[2], expected, places=12)
        np.testing.assert_allclose(actual.iloc[2], desired.iloc[2])

    def test_trade_log_weights_exactly_reproduce_turnover(self):
        frame = backtest.prepare(universe(periods=190), "Top 3 momentum", 5)
        log = backtest.trade_log(frame)
        for _, row in log.iterrows():
            calculated = sum(
                abs(row[f"NewWeight_{ticker}"] - row[f"OldWeight_{ticker}"])
                for ticker in backtest.TICKERS
            )
            self.assertAlmostEqual(calculated, row["Turnover"], delta=1e-12)

    def test_simulated_cash_entry_and_single_asset_switch_turnover(self):
        dates = pd.bdate_range("2020-01-02", periods=3)
        desired = pd.DataFrame(0.0, index=dates, columns=backtest.TICKERS[:2])
        desired.loc[dates[0], "SPY"] = 1
        desired.loc[dates[1]:, "EFA"] = 1
        returns = pd.DataFrame(0.0, index=dates, columns=desired.columns)
        events = pd.Series([True, True, False], index=dates)
        _, _, turnover, _ = backtest.simulate_weights(desired, returns, events)
        self.assertEqual(turnover.iloc[0], 1)
        self.assertEqual(turnover.iloc[1], 2)

    def test_sector_top3_drifts_delays_signal_and_starts_in_cash(self):
        close = backtest.load_sector_close()
        start = pd.Timestamp("2006-08-07")
        end = pd.Timestamp("2006-10-31")
        frame = backtest.prepare_rotation_from_close(
            close, "Sector Top 3", 3, 5, start, end
        )
        first_event = frame.index[frame["RebalanceEvent"].eq(1)][0]
        first_loc = frame.index.get_loc(first_event)
        actual_cols = [f"ActualWeight_{ticker}" for ticker in backtest.SECTOR_TICKERS]
        self.assertTrue(frame.iloc[:first_loc][actual_cols].eq(0).all().all())
        self.assertEqual(first_event, frame.index[first_loc - 1] + pd.offsets.BDay(1))
        self.assertAlmostEqual(frame.loc[first_event, "Turnover"], 1.0)
        after = frame.iloc[first_loc + 1]
        self.assertFalse(np.allclose(
            after[actual_cols][after[actual_cols] > 0].to_numpy(),
            np.repeat(1 / 3, 3),
        ))

    def test_common_window_comparison_is_reproducible(self):
        data = backtest.load_universe()
        daily = pd.concat(
            [
                backtest.prepare(data, name, backtest.BASE_COST_BPS)
                .reset_index(names="Date")
                for name in [*backtest.PORTFOLIOS, backtest.BENCHMARK]
            ],
            ignore_index=True,
        )
        first = backtest.build_common_window_comparison(daily)
        second = backtest.build_common_window_comparison(daily)
        pd.testing.assert_frame_equal(first, second)


if __name__ == "__main__":
    unittest.main()
