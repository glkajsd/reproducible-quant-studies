import unittest

import numpy as np
import pandas as pd

import backtest


def synthetic_prices(close_values):
    dates = pd.bdate_range("2020-01-02", periods=len(close_values))
    close = np.asarray(close_values, dtype=float)
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Volume": 1_000_000,
        }
    )


class GoldenCrossBacktestTests(unittest.TestCase):
    def test_less_than_200_days_has_no_position(self):
        prices = synthetic_prices(np.linspace(100, 120, 199))
        result = backtest.run_backtest(prices, cost_bps=5)

        self.assertEqual(int(result["Signal"].sum()), 0)
        self.assertEqual(int(result["Position"].sum()), 0)
        self.assertAlmostEqual(float(result["TradingCost"].sum()), 0.0)
        self.assertAlmostEqual(
            float(result["StrategyEquity"].iloc[-1]),
            backtest.INITIAL_CAPITAL,
        )

    def test_position_uses_lagged_signal(self):
        # Flat prices for the first 200 days so SMA50 and SMA200 are equal.
        # Then a jump that lifts SMA50 above SMA200 after a few days.
        prices = synthetic_prices([100] * 201 + [105] * 50)
        result = backtest.run_backtest(prices, cost_bps=0)

        first_risk_on_signal_date = result.index[result["Signal"] == 1][0]
        first_position_date = result.index[result["Position"] == 1][0]
        expected_position_date = result.index[
            result.index.get_loc(first_risk_on_signal_date) + 1
        ]

        self.assertEqual(first_position_date, expected_position_date)
        self.assertEqual(
            int(result.loc[first_risk_on_signal_date, "Position"]), 0
        )

    def test_signal_is_zero_when_sma50_below_sma200(self):
        # Prices trending down so SMA50 stays below SMA200 after warmup.
        close = [100] * 200 + list(np.linspace(100, 90, 50))
        prices = synthetic_prices(close)
        result = backtest.run_backtest(prices, cost_bps=0)

        # After warmup, all signals should be 0 since price is declining.
        post_warmup = result.iloc[200:]
        self.assertEqual(int(post_warmup["Signal"].sum()), 0)

    def test_golden_cross_triggers_buy(self):
        # Start flat, then a strong uptrend so SMA50 crosses above SMA200.
        close = [100] * 200 + list(np.linspace(100, 150, 80))
        prices = synthetic_prices(close)
        result = backtest.run_backtest(prices, cost_bps=0)

        # After warmup + uptrend, the signal should eventually become 1.
        post_warmup = result.iloc[220:]
        self.assertGreater(int(post_warmup["Signal"].sum()), 0)
        self.assertGreater(int(post_warmup["Position"].sum()), 0)

    def test_cost_is_charged_only_on_position_changes(self):
        # Create a regime where SMA50 oscillates around SMA200.
        close = [100] * 200 + [120, 100, 120, 100, 120, 100, 120, 100]
        prices = synthetic_prices(close)
        result = backtest.run_backtest(prices, cost_bps=5)
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

    def test_cost_sensitivity_reduces_equity_when_trades_exist(self):
        close = [100] * 200 + [120, 100, 120, 100, 120, 100, 120, 100]
        prices = synthetic_prices(close)
        result_0 = backtest.run_backtest(prices, cost_bps=0)
        result_5 = backtest.run_backtest(prices, cost_bps=5)
        result_10 = backtest.run_backtest(prices, cost_bps=10)

        self.assertGreater(int(result_0["PositionChange"].sum()), 0)
        self.assertGreater(
            result_0["StrategyEquity"].iloc[-1],
            result_5["StrategyEquity"].iloc[-1],
        )
        self.assertGreater(
            result_5["StrategyEquity"].iloc[-1],
            result_10["StrategyEquity"].iloc[-1],
        )


if __name__ == "__main__":
    unittest.main()
