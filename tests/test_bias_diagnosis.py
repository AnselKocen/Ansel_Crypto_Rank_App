import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from diagnose_replay_bias import training_data, center_labels, evaluate_scores


class BiasDiagnosisTests(unittest.TestCase):
    def history(self):
        return pd.DataFrame([
            {"date": d, "symbol": f"C{j}", "clipped_weekly_return": .01*i+.001*j,
             "raw_weekly_return": .02*i+.002*j}
            for i, d in enumerate(pd.date_range("2024-01-03", periods=60, freq="W-WED"))
            for j in range(3)])

    def quotes(self):
        return pd.DataFrame({"symbol": [f"C{i}" for i in range(60)],
                             "score": np.linspace(-.2, .3, 60),
                             "actual_return_raw": np.sin(np.arange(60))*.1,
                             "evaluation_status": "evaluated"})

    def test_future_returns_cannot_affect_training_or_labels(self):
        h = self.history()
        decision = h.date.unique()[-4]
        a, _ = training_data(h, decision)
        h.loc[h.date.gt(decision), ["clipped_weekly_return", "raw_weekly_return"]] = 1e9
        b, _ = training_data(h, decision)
        assert_frame_equal(a, b)
        self.assertEqual(a.date.nunique(), 52)
        self.assertEqual(a.date.max()+pd.Timedelta(days=7), decision)

    def test_missing_coin_week_does_not_become_two_week_label(self):
        h = self.history()
        missing = h.date.unique()[-8]
        h = h[~(h.symbol.eq("C0") & h.date.eq(missing))]
        train, _ = training_data(h, h.date.max())
        self.assertFalse((train.symbol.eq("C0") & train.date.eq(missing-pd.Timedelta(days=7))).any())

    def test_training_market_component_removed_for_each_date(self):
        train, _ = training_data(self.history(), self.history().date.max())
        centered = center_labels(train)
        self.assertTrue(centered.groupby(train.date).mean().abs().lt(1e-14).all())
        self.assertTrue(np.allclose(centered[train.symbol.eq("C2")], .001))

    def test_additive_market_component_cannot_change_selection_or_rank_ic(self):
        f = self.quotes()
        original = evaluate_scores(f, "score")
        f["score"] += 3.
        f["actual_return_raw"] += .5
        shifted = evaluate_scores(f, "score")
        for metric in ["rank_ic", "ls_return", "top_excess", "bottom_excess"]:
            self.assertAlmostEqual(original[metric], shifted[metric])
        self.assertAlmostEqual(shifted["top_return"]-original["top_return"], .5)

    def test_unavailable_future_price_cannot_silently_change_universe(self):
        f = self.quotes()
        f.loc[59, ["actual_return_raw", "evaluation_status"]] = [np.nan, "missing_price"]
        result = evaluate_scores(f, "score")
        self.assertEqual(result, {"status": "incomplete_universe"})

    def test_identical_scores_cannot_become_arbitrary_profitable_selection(self):
        f = self.quotes()
        f["score"] = -.01
        first = evaluate_scores(f, "score")
        second = evaluate_scores(f.iloc[::-1], "score")
        self.assertEqual(first["status"], "no_ranking_signal")
        self.assertEqual(second["status"], "no_ranking_signal")
        self.assertNotIn("ls_return", first)


if __name__ == "__main__":
    unittest.main()
