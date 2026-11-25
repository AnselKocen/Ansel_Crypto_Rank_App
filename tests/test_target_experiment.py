import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from replay_core import ALL_FEATURES, MODEL_PARAMS, ROOT, train_predict
from replay_target_experiment import exact_weekly_returns, quality_reasons, target_frame
import json


class TargetExperimentTests(unittest.TestCase):
    def history(self):
        rng = np.random.default_rng(9)
        rows = []
        for i, date in enumerate(pd.date_range("2024-01-03", periods=60, freq="W-WED")):
            for j, symbol in enumerate(["AAA", "BBB", "CCC"]):
                raw = .2 * np.sin(i / 3 + j)
                row = {"date": date, "symbol": symbol, "raw_weekly_return": raw,
                       "clipped_weekly_return": np.clip(raw, -.04, .04),
                       "return": np.clip(raw, -.04, .04), "sentiment_available": True}
                row.update({f: float(rng.normal()) for f in ALL_FEATURES})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_raw_returns_do_not_bridge_missing_calendar_week(self):
        f = pd.DataFrame({"symbol": ["AAA"] * 3,
                          "date": pd.to_datetime(["2025-01-01", "2025-01-08", "2025-01-22"]),
                          "close": [100., 120., 240.]})
        values = exact_weekly_returns(f)
        self.assertTrue(pd.isna(values.iloc[0]))
        self.assertAlmostEqual(values.iloc[1], .2)
        self.assertTrue(pd.isna(values.iloc[2]))

    def test_target_switch_preserves_every_input_feature_and_original(self):
        h = self.history()
        before = h.copy(deep=True)
        clipped, raw = target_frame(h, "legacy_clipped"), target_frame(h, "raw")
        assert_frame_equal(clipped.drop(columns="return"), raw.drop(columns="return"))
        assert_frame_equal(h, before)
        self.assertTrue((raw["return"] == h.raw_weekly_return).all())
        self.assertTrue((clipped.strev_weekly == raw.strev_weekly).all())

    def test_raw_target_retrains_all_four_estimators_and_changes_predictions(self):
        h = self.history()
        old, _ = train_predict(target_frame(h, "legacy_clipped"), h.date.max())
        new, _ = train_predict(target_frame(h, "raw"), h.date.max())
        self.assertEqual(set(new.model_name), set(MODEL_PARAMS))
        key = ["model_name", "symbol"]
        delta = old.set_index(key).predicted_return - new.set_index(key).predicted_return
        self.assertTrue(delta.abs().gt(1e-6).all())
        self.assertTrue(old.groupby("model_name").training_rows.first().equals(
            new.groupby("model_name").training_rows.first()))

    def test_raw_future_labels_are_invisible(self):
        h = self.history()
        decision = h.date.unique()[-5]
        a, _ = train_predict(target_frame(h, "raw"), decision)
        h.loc[h.date.gt(decision), "raw_weekly_return"] = 1e9
        b, _ = train_predict(target_frame(h, "raw"), decision)
        assert_frame_equal(a, b)

    def test_migration_quarantine_does_not_remove_earlier_observations(self):
        rules = json.loads((ROOT / "replay_quality_rules.json").read_text())
        f = pd.DataFrame({"symbol": ["OM", "OM", "QNT", "MSQ", "DEXE"],
                          "date": pd.to_datetime(["2026-03-01", "2026-03-02", "2026-09-30", "2026-05-13", "2026-07-21"])})
        reason = quality_reasons(f, rules)
        self.assertEqual(reason.ne("").tolist(), [False, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
