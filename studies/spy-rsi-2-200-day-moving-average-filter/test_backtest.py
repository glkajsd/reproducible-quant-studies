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


def indicator_frame(close_values, rsi_values, sma_values):
    dates = pd.bdate_range("2020-01-02", periods=len(close_values))
    return pd.DataFrame(
        {
            "AdjustedClose": np.asarray(close_values, dtype=float),
            "RSI2": np.asarray(rsi_values, dtype=float),
            "SMA200": np.asarray(sma_values, dtype=float),
        },
        index=dates,
    )


class Rsi2FilterBacktestTests(unittest.TestCase):
    def test_wilder_rsi_2_is_reproducible_on_simple_sample(self):
        close = pd.Series([100, 102, 101, 100, 103], dtype=float)
        rsi = backtest.compute_wilder_rsi(close, window=2)

        self.assertTrue(np.isnan(rsi.iloc[0]))
        self.assertTrue(np.isnan(rsi.iloc[1]))
        self.assertAlmostEqual(rsi.iloc[2], 66.6666666667)
        self.assertAlmostEqual(rsi.iloc[3], 40.0)
        self.assertAlmostEqual(rsi.iloc[4], 82.3529411765)

    def test_rsi_only_entry_exit_state_machine_is_correct(self):
        frame = indicator_frame(
            close_values=[100, 99, 98, 102, 101, 97],
            rsi_values=[50, 5, 20, 80, 50, 5],
            sma_values=[np.nan] * 6,
        )
        result = backtest.apply_strategy_rules(frame, backtest.VARIANT_RSI_ONLY)

        expected_state = pd.Series(
            [0, 1, 1, 0, 0, 1],
            index=result.index,
            name="SignalState",
            dtype=int,
        )
        expected_position = expected_state.shift(1).fillna(0).astype(int)

        pd.testing.assert_series_equal(result["SignalState"], expected_state)
        pd.testing.assert_series_equal(
            result["Position"],
            expected_position,
            check_names=False,
        )

    def test_filter_has_no_entry_before_sma200_warmup(self):
        prices = synthetic_prices(np.linspace(120, 80, 199))
        result = backtest.run_backtest(prices, backtest.VARIANT_FILTER, cost_bps=5)

        self.assertEqual(int(result["RawEntrySignal"].sum()), 0)
        self.assertEqual(int(result["SignalState"].sum()), 0)
        self.assertEqual(int(result["Position"].sum()), 0)
        self.assertAlmostEqual(float(result["TradingCost"].sum()), 0.0)

    def test_filter_entry_requires_close_above_sma_and_low_rsi(self):
        frame = indicator_frame(
            close_values=[100, 99, 101, 102],
            rsi_values=[50, 5, 5, 50],
            sma_values=[100, 100, 100, 100],
        )
        result = backtest.apply_strategy_rules(frame, backtest.VARIANT_FILTER)

        self.assertEqual(int(result["RawEntrySignal"].iloc[1]), 0)
        self.assertEqual(int(result["RawEntrySignal"].iloc[2]), 1)
        self.assertEqual(int(result["SignalState"].iloc[2]), 1)
        self.assertEqual(int(result["Position"].iloc[2]), 0)
        self.assertEqual(int(result["Position"].iloc[3]), 1)

    def test_filter_close_below_sma_triggers_exit_state(self):
        frame = indicator_frame(
            close_values=[100, 101, 102, 99, 101],
            rsi_values=[50, 5, 50, 50, 50],
            sma_values=[100, 100, 100, 100, 100],
        )
        result = backtest.apply_strategy_rules(frame, backtest.VARIANT_FILTER)

        self.assertEqual(int(result["SignalState"].iloc[1]), 1)
        self.assertEqual(int(result["RawExitSignal"].iloc[3]), 1)
        self.assertEqual(int(result["SignalState"].iloc[3]), 0)
        self.assertEqual(int(result["Position"].iloc[3]), 1)
        self.assertEqual(int(result["Position"].iloc[4]), 0)

    def test_position_is_shifted_state_to_avoid_lookahead(self):
        frame = indicator_frame(
            close_values=[100, 101, 102, 103, 98],
            rsi_values=[50, 5, 20, 80, 50],
            sma_values=[100, 100, 100, 100, 100],
        )

        for variant in backtest.STRATEGY_VARIANTS:
            with self.subTest(variant=variant):
                result = backtest.apply_strategy_rules(frame, variant)
                expected_position = result["SignalState"].shift(1).fillna(0).astype(int)
                pd.testing.assert_series_equal(
                    result["Position"],
                    expected_position,
                    check_names=False,
                )
                first_entry_date = result.index[result["RawEntrySignal"] == 1][0]
                self.assertEqual(int(result.loc[first_entry_date, "Position"]), 0)

    def test_cost_is_charged_only_on_position_changes(self):
        prices = synthetic_prices([100, 99, 98, 120, 121, 90, 89, 130, 131])
        result = backtest.run_backtest(prices, backtest.VARIANT_RSI_ONLY, cost_bps=5)
        expected_cost = result["PositionChange"] * 0.0005

        self.assertGreater(int(result["PositionChange"].sum()), 0)
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
        prices = synthetic_prices(
            [100] * 199 + [140, 130, 120, 110, 105, 104, 106, 120, 90, 88, 130, 131]
        )
        summary_1 = backtest.build_summary(prices)
        summary_2 = backtest.build_summary(prices)

        expected_costs = {0.0, 5.0, 10.0}
        self.assertEqual(set(summary_1["Cost bps"]), expected_costs)
        self.assertEqual(set(summary_1["Strategy variant"]), set(backtest.STRATEGY_VARIANTS))
        self.assertEqual(len(summary_1), len(backtest.STRATEGY_VARIANTS) * len(expected_costs))
        pd.testing.assert_frame_equal(summary_1, summary_2)


if __name__ == "__main__":
    unittest.main()
