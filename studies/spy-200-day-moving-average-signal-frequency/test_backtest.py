import unittest

import numpy as np
import pandas as pd

import backtest


def synthetic_prices(close_values, dates=None):
    if dates is None:
        dates = pd.bdate_range("2020-01-02", periods=len(close_values))
    close = np.asarray(close_values, dtype=float)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(dates),
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        }
    )


class SignalFrequencyBacktestTests(unittest.TestCase):
    def test_less_than_200_days_has_no_risk_on_signal(self):
        prices = synthetic_prices(np.linspace(100, 120, 199))

        for variant in backtest.VARIANTS:
            result = backtest.run_backtest(prices, variant=variant, cost_bps=5)
            self.assertEqual(int(result["Signal"].sum()), 0)
            self.assertEqual(int(result["Position"].sum()), 0)
            self.assertAlmostEqual(float(result["TradingCost"].sum()), 0.0)
            self.assertAlmostEqual(
                float(result["StrategyEquity"].iloc[-1]),
                backtest.INITIAL_CAPITAL,
            )

    def test_all_variants_use_lagged_position(self):
        prices = synthetic_prices([100] * 199 + list(np.linspace(110, 130, 30)))

        for variant in backtest.VARIANTS:
            with self.subTest(variant=variant):
                result = backtest.run_backtest(prices, variant=variant, cost_bps=0)
                expected_position = result["Signal"].shift(1).fillna(0).astype(int)
                pd.testing.assert_series_equal(
                    result["Position"],
                    expected_position,
                    check_names=False,
                )

    def test_daily_first_risk_on_position_follows_signal_by_one_day(self):
        prices = synthetic_prices([100] * 199 + [110, 111])
        result = backtest.run_backtest(prices, variant="daily", cost_bps=0)
        first_signal_date = result.index[result["Signal"] == 1][0]
        first_position_date = result.index[result["Position"] == 1][0]
        expected_position_date = result.index[result.index.get_loc(first_signal_date) + 1]

        self.assertEqual(first_position_date, expected_position_date)
        self.assertEqual(int(result.loc[first_signal_date, "Position"]), 0)

    def test_weekly_updates_only_on_last_trading_day_and_forward_fills(self):
        prices = synthetic_prices([100] * 199 + list(np.linspace(110, 130, 20)))
        result = backtest.prepare_signals(prices, variant="weekly")

        dates = pd.Series(result.index, index=result.index)
        expected_updates = dates == dates.groupby(result.index.to_period("W-FRI")).transform("max")
        pd.testing.assert_series_equal(
            result["SignalUpdate"].astype(bool),
            expected_updates,
            check_names=False,
        )

        above_sma = (result["SMA200"].notna()) & (result["AdjustedClose"] > result["SMA200"])
        update_dates = result.index[result["SignalUpdate"] == 1]
        pd.testing.assert_series_equal(
            result.loc[update_dates, "Signal"],
            above_sma.loc[update_dates].astype(int),
            check_names=False,
        )

        non_update_dates = result.index[result["SignalUpdate"] == 0][1:]
        previous_signal = result["Signal"].shift(1).loc[non_update_dates].astype(int)
        pd.testing.assert_series_equal(
            result.loc[non_update_dates, "Signal"],
            previous_signal,
            check_names=False,
        )

    def test_month_end_updates_only_on_last_trading_day_and_forward_fills(self):
        prices = synthetic_prices([100] * 199 + list(np.linspace(110, 150, 65)))
        result = backtest.prepare_signals(prices, variant="month_end")

        dates = pd.Series(result.index, index=result.index)
        expected_updates = dates == dates.groupby(result.index.to_period("M")).transform("max")
        pd.testing.assert_series_equal(
            result["SignalUpdate"].astype(bool),
            expected_updates,
            check_names=False,
        )

        above_sma = (result["SMA200"].notna()) & (result["AdjustedClose"] > result["SMA200"])
        update_dates = result.index[result["SignalUpdate"] == 1]
        pd.testing.assert_series_equal(
            result.loc[update_dates, "Signal"],
            above_sma.loc[update_dates].astype(int),
            check_names=False,
        )

        non_update_dates = result.index[result["SignalUpdate"] == 0][1:]
        previous_signal = result["Signal"].shift(1).loc[non_update_dates].astype(int)
        pd.testing.assert_series_equal(
            result.loc[non_update_dates, "Signal"],
            previous_signal,
            check_names=False,
        )

    def test_cost_is_charged_only_on_position_changes(self):
        prices = synthetic_prices([100] * 199 + [110, 111, 90, 89, 120, 121])
        result = backtest.run_backtest(prices, variant="daily", cost_bps=5)
        expected_cost = result["PositionChange"] * 0.0005

        pd.testing.assert_series_equal(
            result["TradingCost"],
            expected_cost,
            check_names=False,
        )
        self.assertEqual(
            int((result["TradingCost"] > 0).sum()),
            int((result["PositionChange"] > 0).sum()),
        )

    def test_cost_sensitivity_rows_exist_and_are_reproducible(self):
        prices = synthetic_prices([100] * 199 + [110, 111, 90, 89, 120, 121, 122])
        summary_1 = backtest.build_summary(prices)
        summary_2 = backtest.build_summary(prices)

        expected_costs = {0.0, 5.0, 10.0}
        self.assertEqual(set(summary_1["Cost bps"]), expected_costs)
        self.assertEqual(set(summary_1["Strategy variant"]), set(backtest.VARIANTS.values()))
        self.assertEqual(len(summary_1), len(backtest.VARIANTS) * len(expected_costs))
        pd.testing.assert_frame_equal(summary_1, summary_2)


if __name__ == "__main__":
    unittest.main()
