import unittest

import numpy as np
import pandas as pd

from replay_portfolio_report import ranked_weights, fusion_weights, longest_common_segment, cumulative_statistics


class RankedPortfolioTests(unittest.TestCase):
    def test_rank_selects_twenty_even_when_all_predictions_negative(self):
        frame = pd.DataFrame({"symbol": [f"C{i}" for i in range(50)],
                              "predicted_rank": range(1, 51), "predicted_return": np.linspace(-.01, -.5, 50)})
        positions = ranked_weights(frame.sample(frac=1, random_state=4), "EW")
        self.assertEqual([s for s, _, _ in positions], [f"C{i}" for i in range(20)])
        self.assertAlmostEqual(sum(w for _, _, w in positions), 1.)
        ls = ranked_weights(frame, "LS")
        self.assertEqual({s for s, side, _ in ls if side == "short"}, {f"C{i}" for i in range(30, 50)})
        self.assertAlmostEqual(sum(w for _, side, w in ls if side == "short"), -1.)

    def test_consensus_fusion_uses_original_weights_and_normalizes_each_side(self):
        a = pd.DataFrame({"symbol": ["A", "B", "C", "D"], "is_top20": [True, True, False, False], "is_bottom20": [False, False, True, True]})
        b = pd.DataFrame({"symbol": ["A", "E", "C", "F"], "is_top20": [True, True, False, False], "is_bottom20": [False, False, True, True]})
        positions = fusion_weights(a, b)
        long = {s: w for s, side, w in positions if side == "long"}
        self.assertAlmostEqual(long["A"], .65/(.65+.35+.35))
        self.assertAlmostEqual(sum(long.values()), 1.)
        self.assertAlmostEqual(sum(w for _, side, w in positions if side == "short"), -1.)

    def test_missing_week_is_a_gap_and_window_is_chosen_by_availability(self):
        f = pd.DataFrame({"decision_date": pd.date_range("2025-01-01", periods=6, freq="W-WED"),
                          "status": ["evaluated", "evaluated", "missing", "evaluated", "evaluated", "evaluated"],
                          "weekly_return": [10., 10., np.nan, -.1, -.1, -.1]})
        common, longest, segments = longest_common_segment(f)
        self.assertEqual(len(common), 5)
        self.assertEqual(len(longest), 3)
        self.assertEqual(longest[0], pd.Timestamp("2025-01-22"))
        self.assertEqual(len(segments), 2)

    def test_drawdown_includes_initial_capital(self):
        stats, nav, _ = cumulative_statistics([-.1, .2])
        self.assertAlmostEqual(stats["total_return"], .08)
        self.assertAlmostEqual(stats["max_drawdown"], -.1)
        self.assertEqual(nav[0], 1.)
        with self.assertRaises(ValueError):
            cumulative_statistics([.1, np.nan])


if __name__ == "__main__":
    unittest.main()
