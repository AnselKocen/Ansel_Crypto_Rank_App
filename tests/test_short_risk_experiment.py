import unittest

import numpy as np
import pandas as pd

from replay_short_risk_experiment import allocate_short_weights, continuous_metrics, trailing_risk


class ShortRiskExperimentTests(unittest.TestCase):
    def test_future_prices_cannot_change_decision_risk(self):
        dates = pd.date_range("2026-01-01", periods=50)
        prices = pd.DataFrame({"symbol": "X", "date": dates,
                               "close": np.exp(np.sin(np.arange(50))/10)})
        decision = dates[35]
        before = trailing_risk(prices, [decision])
        altered = prices.copy()
        altered.loc[altered.date.gt(decision), "close"] *= 1000
        pd.testing.assert_frame_equal(before, trailing_risk(altered, [decision]))

    def test_missing_days_are_not_multiday_returns_and_old_returns_expire(self):
        dates = pd.date_range("2026-01-01", periods=60)
        prices = pd.DataFrame({"symbol": "X", "date": dates, "close": np.exp(np.arange(60)/100)})
        prices = prices[~prices.date.eq(dates[-3])]
        result = trailing_risk(prices, [dates[-1]]).iloc[0]
        # The missing price destroys both its own day's and the next day's return.
        self.assertEqual(result.valid_daily_returns, 26)
        self.assertAlmostEqual(result.volatility_28d, 0, places=12)
        # Plenty of old data cannot compensate for sparse current-window prices.
        sparse = prices[prices.date.lt(dates[-25]) | prices.date.eq(dates[-1])]
        result = trailing_risk(sparse, [dates[-1]]).iloc[0]
        self.assertEqual(result.valid_daily_returns, 3)
        self.assertEqual(result.risk_status, "insufficient_daily_returns")
        self.assertTrue(pd.isna(result.volatility_28d))

    def test_sizing_control_matches_exposure_and_never_adds_risk(self):
        weights = allocate_short_weights([.1, .2, .4, np.nan], .2, [-.05]*4)
        np.testing.assert_allclose(weights["volatility_reduced"], [-.05, -.05, -.025, 0])
        np.testing.assert_allclose(weights["matched_gross_equal"], [-.03125]*4)
        self.assertAlmostEqual(sum(weights["matched_gross_equal"]), sum(weights["volatility_reduced"]))
        self.assertTrue((abs(weights["volatility_reduced"]) <= .05).all())
        zero = allocate_short_weights([0., .1], 0., [-.05, -.05])
        np.testing.assert_allclose(zero["volatility_reduced"], [-.05, 0])
        missing = allocate_short_weights([.1], np.nan, [-.05])
        np.testing.assert_allclose(missing["volatility_reduced"], [0])

    def test_continuous_account_links_legs_and_rejects_calendar_gaps(self):
        frame = pd.DataFrame({"decision_date": pd.to_datetime(["2026-01-07", "2026-01-14"]),
                              "weekly_return": [.05, .3], "long_contribution": [.1, .2],
                              "short_contribution": [-.05, .1]})
        result = continuous_metrics(frame)
        self.assertAlmostEqual(result["total_return"], .365)
        self.assertAlmostEqual(result["linked_long_contribution"] + result["linked_short_contribution"], .365)
        frame.loc[1, "decision_date"] += pd.Timedelta(days=7)
        with self.assertRaises(ValueError):
            continuous_metrics(frame)


if __name__ == "__main__":
    unittest.main()
