import math
import unittest

import numpy as np
import pandas as pd

import backtest


def close_frame(periods=70):
    dates = pd.bdate_range("2020-01-02", periods=periods)
    data = {}
    for i, ticker in enumerate(backtest.TICKERS):
        data[ticker] = 100 * (1 + (i + 1) * 0.0005) ** np.arange(periods)
    return pd.DataFrame(data, index=dates)


class MonthlyContributionTests(unittest.TestCase):
    def test_first_trading_day_mask(self):
        dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-02-03", "2020-02-04"])
        mask = backtest.first_trading_day_mask(pd.DatetimeIndex(dates))
        self.assertEqual(mask.tolist(), [True, False, True, False])

    def test_weight_series_requires_sum_to_one(self):
        with self.assertRaises(ValueError):
            backtest._weight_series({"SPY": 0.5})
        weights = backtest._weight_series({"SPY": 0.6, "IEF": 0.4})
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertEqual(weights["QQQ"], 0.0)

    def test_monthly_contributions_are_added_on_first_trading_day(self):
        close = close_frame()
        frame = backtest.simulate_monthly_contributions(close, "SPY 100%", {"SPY": 1.0}, monthly_contribution=100)
        contributions = frame[frame["Contribution"] > 0]
        expected_dates = backtest.first_trading_day_mask(close.index).sum()
        self.assertEqual(len(contributions), expected_dates)
        self.assertAlmostEqual(frame["TotalContributed"].iloc[-1], expected_dates * 100)

    def test_single_asset_constant_price_keeps_value_equal_to_contributions(self):
        dates = pd.bdate_range("2020-01-02", periods=45)
        close = pd.DataFrame(100.0, index=dates, columns=backtest.TICKERS)
        frame = backtest.simulate_monthly_contributions(close, "SPY 100%", {"SPY": 1.0}, monthly_contribution=250)
        self.assertAlmostEqual(frame["PortfolioValue"].iloc[-1], frame["TotalContributed"].iloc[-1])
        self.assertAlmostEqual(frame["ProfitOverContributions"].iloc[-1], 0.0)

    def test_top_level_build_outputs_all_portfolios(self):
        dates = pd.bdate_range("2020-01-02", periods=90)
        data = {}
        for ticker in backtest.TICKERS:
            values = 100 * (1.0002 ** np.arange(len(dates)))
            data[ticker] = pd.DataFrame({"Date": dates, "Close": values})
        close = backtest.close_matrix(data)
        frames = [
            backtest.simulate_monthly_contributions(close, name, weights)
            for name, weights in backtest.PORTFOLIOS.items()
        ]
        summary = pd.DataFrame([backtest.summarize(frame) for frame in frames])
        daily = pd.concat([frame.reset_index() for frame in frames], ignore_index=True)
        self.assertEqual(set(summary["Portfolio"]), set(backtest.PORTFOLIOS))
        self.assertEqual(set(daily["Portfolio"]), set(backtest.PORTFOLIOS))

    def test_start_year_sensitivity_can_use_shorter_windows(self):
        dates = pd.bdate_range("2020-01-02", periods=520)
        close = pd.DataFrame(
            {ticker: 100 * (1.0002 ** np.arange(len(dates))) for ticker in backtest.TICKERS},
            index=dates,
        )
        sensitivity = backtest.build_start_year_sensitivity(close, years=1)
        self.assertFalse(sensitivity.empty)
        self.assertEqual(set(sensitivity["Portfolio"]), set(backtest.PORTFOLIOS))

    def test_money_weighted_return_for_constant_value_is_zero(self):
        dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
        flows = pd.Series([100.0, 0.0], index=dates)
        irr = backtest.money_weighted_return(flows, 100.0)
        self.assertTrue(math.isclose(irr, 0.0, abs_tol=1e-7))

    def test_close_matrix_inner_join(self):
        dates = pd.bdate_range("2020-01-02", periods=5)
        data = {}
        for ticker in backtest.TICKERS:
            ticker_dates = dates[1:] if ticker == "DBC" else dates
            data[ticker] = pd.DataFrame({"Date": ticker_dates, "Close": np.arange(len(ticker_dates)) + 100})
        close = backtest.close_matrix(data)
        self.assertEqual(close.index[0], dates[1])


if __name__ == "__main__":
    unittest.main()
