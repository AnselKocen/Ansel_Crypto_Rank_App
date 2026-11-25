import unittest
import ast
import contextlib
import io
import tempfile
from unittest.mock import patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from replay_core import ALL_FEATURES, MODEL_PARAMS, train_predict, weekly_market, weekly_sentiment
from replay_weekly_predictions import compare_actual, strategy_results
from prepare_replay_data import matched_yahoo_prices


class HistoricalReplayTests(unittest.TestCase):
    def history(self):
        rng = np.random.default_rng(42)
        rows = []
        for date in pd.date_range("2024-01-03", periods=60, freq="W-WED"):
            for symbol in ["AAA", "BBB", "CCC"]:
                row = {"date": date, "symbol": symbol, "return": float(rng.normal(0, .03)),
                       "sentiment_available": True}
                row.update({f: float(rng.normal()) for f in ALL_FEATURES})
                rows.append(row)
        return pd.DataFrame(rows)

    def test_all_four_models_and_future_data_isolation(self):
        history = self.history()
        decision = pd.Timestamp("2025-01-01")
        first, events = train_predict(history, decision)
        modified = history.copy()
        modified.loc[modified.date > decision, ALL_FEATURES + ["return"]] = 1e12
        second, _ = train_predict(modified, decision)
        assert_frame_equal(first, second)
        self.assertEqual(set(first.model_name), set(MODEL_PARAMS))
        self.assertEqual(len(events), 4)
        self.assertTrue((first.train_end < first.decision_date).all())
        self.assertTrue((first.target_end - first.decision_date == pd.Timedelta(days=7)).all())

    def test_missing_calendar_week_is_not_next_observation(self):
        history = self.history()
        history = history[history.date != pd.Timestamp("2024-08-07")]
        with self.assertRaisesRegex(ValueError, "missing calendar weeks"):
            train_predict(history, pd.Timestamp("2025-01-01"))

    def test_unchanged_original_entry_training_parity(self):
        history = self.history().drop(columns=["sentiment_available"])
        # Reverse symbols inside every week to catch accidental row-order changes.
        history = history.sort_values(["date", "symbol"], ascending=[True, False])
        decision = history.date.max()
        source = Path(__file__).resolve().parents[1] / "customer_side.py"
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "predict_from_history")
        module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
        with tempfile.TemporaryDirectory() as directory:
            scope = {"Path": Path}
            exec(compile(module, str(source), "exec"), scope)
            with contextlib.redirect_stdout(io.StringIO()):
                scope[fn.name](history.copy(), decision, Path(directory), [])
            ours, _ = train_predict(history, decision)
            for model in MODEL_PARAMS:
                original = pd.read_csv(Path(directory) / f"pred_{model}.csv").set_index("symbol").y_pred
                actual = ours[ours.model_name == model].set_index("symbol").predicted_return
                self.assertLess(float((original - actual).abs().max()), 1e-10)

    def test_homonym_rejected_using_pre_start_reference(self):
        days = pd.to_datetime(["2025-07-09", "2025-07-16", "2025-07-23", "2025-07-30"])
        baseline = pd.DataFrame({"date": days, "symbol": ["CORE"] * 4, "close": [1.] * 4})
        wrong = pd.DataFrame({"date": days, "symbol": ["CORE"] * 4, "close": [100.] * 4})
        right = wrong.copy()
        right["close"] = 1.
        with patch("prepare_replay_data.get_prices", side_effect=[wrong, right]):
            _, symbol, error = matched_yahoo_prices(
                "CORE", days[0], days[-1], Path("."), baseline,
                [{"symbol": "CORE23254-USD"}])
        self.assertEqual(symbol, "CORE23254-USD")
        self.assertEqual(error, 0.)

    def test_market_features_do_not_use_future_prices(self):
        rows = []
        for symbol in ["AAA", "BBB"]:
            for i, day in enumerate(pd.date_range("2024-01-01", periods=180)):
                close = 100 + i * .15 + np.sin(i / 3)
                rows.append(dict(symbol=symbol, date=day, open=close-.1, high=close+1,
                                 low=close-1, close=close, usd_volume=10000+i,
                                 base_volume=(10000+i)/close))
        prices = pd.DataFrame(rows)
        decision = pd.Timestamp("2024-05-01")
        a = weekly_market(prices, decision)
        modified = prices.copy()
        modified.loc[modified.date > decision, ["close", "usd_volume"]] = 1e12
        b = weekly_market(modified, decision)
        assert_frame_equal(a, b)

    def test_sentiment_thresholds_do_not_use_future_articles(self):
        news = pd.DataFrame({"published_at": pd.date_range("2024-01-01", periods=30),
                             "compound": np.linspace(-1, 1, 30)})
        a, count = weekly_sentiment(news, pd.Timestamp("2024-01-10"))
        modified = news.copy()
        modified.loc[modified.published_at >= pd.Timestamp("2024-01-11"), "compound"] = 1e9
        b, count2 = weekly_sentiment(modified, pd.Timestamp("2024-01-10"))
        self.assertEqual(a, b)
        self.assertEqual(count, 7)
        self.assertEqual(count2, 7)

    def test_stale_quote_is_not_promoted_to_decision_day(self):
        rows = []
        decision = pd.Timestamp("2024-05-01")
        for symbol in ["AAA", "BBB"]:
            for i, day in enumerate(pd.date_range("2024-01-01", decision)):
                if symbol == "BBB" and day == decision:
                    continue
                rows.append(dict(symbol=symbol, date=day, open=100+i, high=102+i,
                                 low=99+i, close=101+i, usd_volume=10000+i, base_volume=100))
        result = weekly_market(pd.DataFrame(rows), decision)
        self.assertEqual(set(result.symbol), {"AAA"})
        # Even a current quote must not turn a multi-week gap into a weekly return.
        prices = pd.DataFrame(rows)
        prices = pd.concat([prices, pd.DataFrame([dict(
            symbol="BBB", date=decision, open=220, high=222, low=219,
            close=221, usd_volume=10120, base_volume=100)])], ignore_index=True)
        prices = prices[~((prices.symbol == "BBB") &
                          (prices.date == decision - pd.Timedelta(days=7)))]
        result = weekly_market(prices, decision)
        self.assertEqual(set(result.symbol), {"AAA"})

    def test_actual_is_exact_target_week_and_pending_stays_missing(self):
        decision = pd.Timestamp("2025-07-30")
        forecast = pd.DataFrame([dict(symbol="BTC", decision_date=decision,
                                     target_end=decision+pd.Timedelta(days=7),
                                     predicted_return=.05, model_name="enet_all")])
        prices = pd.DataFrame(dict(
            symbol=["BTC"]*3,
            date=[decision, decision+pd.Timedelta(days=7), decision+pd.Timedelta(days=14)],
            close=[100., 110., 50.], source=["test"]*3))
        history = pd.DataFrame(dict(symbol=["BTC"]*2, date=prices.date[:2], **{"return":[0., .08]}))
        result = compare_actual(forecast, prices, history, decision+pd.Timedelta(days=7))
        self.assertAlmostEqual(result.actual_return_raw.iloc[0], .1)
        self.assertAlmostEqual(result.actual_return_model_target.iloc[0], .08)
        pending = compare_actual(forecast, prices, history, decision)
        self.assertEqual(pending.evaluation_status.iloc[0], "pending")
        self.assertTrue(pd.isna(pending.actual_return_raw.iloc[0]))

    def test_missing_selected_asset_does_not_renormalize_portfolio(self):
        f = pd.DataFrame([dict(
            decision_date=pd.Timestamp("2025-07-30"), target_end=pd.Timestamp("2025-08-06"),
            model_name="enet_all", symbol=f"COIN{i}", actual_return_raw=.01,
            is_top20=i<20, is_bottom20=i>=20, evaluation_status="evaluated") for i in range(40)])
        f.loc[0, ["actual_return_raw", "evaluation_status"]] = [np.nan, "missing_price"]
        result, _ = strategy_results(f)
        self.assertTrue(result.weekly_return.isna().all())


if __name__ == "__main__":
    unittest.main()
