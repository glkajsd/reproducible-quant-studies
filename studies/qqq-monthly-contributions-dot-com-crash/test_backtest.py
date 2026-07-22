import math
import unittest

import numpy as np
import pandas as pd

import backtest


class QqqDotComDcaTests(unittest.TestCase):
    def test_first_trading_day_mask(self):
        dates = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01", "2000-02-02"])
        mask = backtest.first_trading_day_mask(pd.DatetimeIndex(dates))
        self.assertEqual(mask.tolist(), [True, False, True, False])

    def test_constant_price_value_equals_contributions(self):
        dates = pd.bdate_range("2000-01-03", periods=45)
        prices = pd.Series(100.0, index=dates)
        frame = backtest.simulate_monthly_contributions(prices, "QQQ", monthly_contribution=500)
        self.assertAlmostEqual(frame["PortfolioValue"].iloc[-1], frame["TotalContributed"].iloc[-1])
        self.assertAlmostEqual(frame["ProfitOverContributions"].iloc[-1], 0.0)

    def test_monthly_contribution_share_count(self):
        dates = pd.to_datetime(["2000-01-03", "2000-01-04", "2000-02-01"])
        prices = pd.Series([100.0, 110.0, 50.0], index=dates)
        frame = backtest.simulate_monthly_contributions(prices, "QQQ", monthly_contribution=500)
        self.assertAlmostEqual(frame["Shares"].iloc[-1], 15.0)
        self.assertAlmostEqual(frame["TotalContributed"].iloc[-1], 1000.0)

    def test_money_weighted_return_constant_value_is_zero(self):
        dates = pd.to_datetime(["2000-01-03", "2001-01-03"])
        flows = pd.Series([500.0, 0.0], index=dates)
        irr = backtest.money_weighted_return(flows, 500.0)
        self.assertTrue(math.isclose(irr, 0.0, abs_tol=1e-7))

    def test_durable_recovery_after_last_negative_profit(self):
        dates = pd.bdate_range("2000-01-03", periods=4)
        frame = pd.DataFrame({"ProfitOverContributions": [0.0, -2.0, -1.0, 3.0]}, index=dates)
        self.assertEqual(backtest.last_below_contributions(frame), dates[2])
        self.assertEqual(backtest.durable_recovery_date(frame), dates[3])

    def test_price_stress_identifies_recovery_after_high(self):
        dates = pd.to_datetime(["2000-01-03", "2000-03-01", "2002-10-09", "2015-02-20"])
        close = pd.DataFrame({"QQQ": [90.0, 100.0, 20.0, 101.0], "SPY": [100.0, 101.0, 80.0, 120.0]}, index=dates)
        stress = backtest.qqq_price_stress(close)
        self.assertEqual(stress["High date"], "2000-03-01")
        self.assertEqual(stress["Trough date"], "2002-10-09")
        self.assertEqual(stress["Recovery date"], "2015-02-20")
        self.assertAlmostEqual(stress["Price drawdown from high"], -0.8)

    def test_build_outputs_has_two_monthly_series(self):
        dates = pd.bdate_range("2000-01-03", periods=400)
        qqq_values = np.r_[
            np.linspace(100, 120, 20),
            np.full(230, 60.0),
            np.linspace(121, 130, 150),
        ]
        data = {
            "QQQ": pd.DataFrame({"Date": dates, "Close": qqq_values}),
            "SPY": pd.DataFrame({"Date": dates, "Close": 100 * (1.0005 ** np.arange(len(dates)))}),
        }
        summary, daily, stress = backtest.build_outputs(data)
        self.assertEqual(set(summary["Ticker"]), {"QQQ", "SPY"})
        self.assertEqual(set(daily["Ticker"]), {"QQQ", "SPY"})
        self.assertEqual(len(stress), 1)


if __name__ == "__main__":
    unittest.main()
