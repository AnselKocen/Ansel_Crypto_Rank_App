"""Verify persisted diagnosis tables and probe the sentiment definition."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_core import DEFAULT_OUTPUT, ROOT, digest, weekly_sentiment, write_csv, write_json


def run():
    out = DEFAULT_OUTPUT / "bias_diagnosis_v2"
    manifest = json.loads((out / "diagnosis_manifest.json").read_text())
    identity = manifest["identity"]
    history = pd.read_csv(DEFAULT_OUTPUT / "target_experiment/common_feature_history.csv",
                          float_precision="round_trip", parse_dates=["date"])
    source = DEFAULT_OUTPUT / "target_experiment/runs" / manifest["source_run"]
    actual = pd.read_csv(source / "forecast_actual_comparison.csv", float_precision="round_trip")
    actual = actual[actual.target_variant.eq("legacy_clipped")]
    scores = pd.read_csv(out / "centered_label_predictions.csv", float_precision="round_trip")
    metrics = pd.read_csv(out / "weekly_selection_metrics.csv", float_precision="round_trip")
    level = pd.read_csv(out / "weekly_prediction_levels.csv", float_precision="round_trip")
    coefficients = pd.read_csv(out / "linear_feature_contributions.csv", float_precision="round_trip")
    keys = ["decision_date", "model_name", "symbol"]
    combined = scores.merge(actual, on=keys, suffixes=("_saved", "_original"), validate="one_to_one")
    checks = {"all_68_decisions_four_models": len(level) == 68*4,
              "same_forecast_keys_as_original": len(combined) == len(scores) == len(actual),
              "original_scores_unchanged": np.array_equal(combined.predicted_return_saved, combined.predicted_return_original),
              "all_scores_finite": bool(np.isfinite(scores.centered_label_score).all()),
              "code_matches_manifest": digest(ROOT / "diagnose_replay_bias.py") == identity["code"],
              "protected_inputs_unchanged": all(digest(ROOT/p) == h for p,h in identity["inputs"].items()),
              "all_internal_checks_pass": all(manifest["checks"].values())}
    daily_lookup = history.set_index(["date", "symbol"])[["clipped_weekly_return", "raw_weekly_return"]]
    train_errors, metric_errors = [], []
    for date, rows in level.groupby("decision_date"):
        decision = pd.Timestamp(date)
        required = pd.date_range(end=decision-pd.Timedelta(days=7), periods=52, freq="W-WED")
        train = history[history.date.isin(required)]
        future_keys = pd.MultiIndex.from_arrays([train.date+pd.Timedelta(days=7), train.symbol], names=["date", "symbol"])
        labels = daily_lookup.reindex(future_keys)
        mean = labels.clipped_weekly_return.mean()
        raw_mean = labels.loc[labels.clipped_weekly_return.notna(), "raw_weekly_return"].mean()
        train_errors += (rows.train_label_mean-mean).abs().tolist()
        train_errors += (rows.train_raw_label_mean-raw_mean).abs().tolist()
    for (date, model), group in combined.groupby(["decision_date", "model_name"], sort=False):
        for arm, score in [("original", "predicted_return_saved"), ("date_centered_training_label", "centered_label_score")]:
            row = metrics[(metrics.decision_date == date) & (metrics.model_name == model) & (metrics.arm == arm)].iloc[0]
            complete = group.evaluation_status.eq("evaluated").all()
            flat = np.ptp(group[score].to_numpy()) <= 1e-12
            expected = "incomplete_universe" if not complete else "no_ranking_signal" if flat else "evaluated"
            checks.setdefault("no_future_price_filter_or_arbitrary_flat_selection", True)
            checks["no_future_price_filter_or_arbitrary_flat_selection"] &= row.status == expected
            if expected != "evaluated":
                continue
            values = group.actual_return_raw.to_numpy()
            order = (np.argsort(group.predicted_rank_saved.to_numpy(), kind="stable") if arm == "original"
                     else np.argsort(-group[score].to_numpy(), kind="stable"))
            top, bottom, market = values[order[:20]].mean(), values[order[-20:]].mean(), values.mean()
            for metric, value in [("top_return", top), ("bottom_return", bottom), ("universe_return", market),
                                  ("top_excess", top-market), ("bottom_excess", bottom-market), ("ls_return", top-bottom)]:
                metric_errors.append(abs(value-row[metric]))
    checks["training_means_rebuilt_by_exact_calendar_lookup"] = max(train_errors) < 1e-12
    checks["all_portfolio_metrics_rebuilt_from_saved_scores_and_actuals"] = max(metric_errors) < 1e-12
    contribution = coefficients.groupby(["decision_date", "model_name"]).mean_prediction_contribution.sum()
    linear = level[level.model_name.str.startswith("enet")].set_index(["decision_date", "model_name"])
    error = (linear.predicted_universe_mean-linear.linear_intercept-contribution).abs().max()
    checks["linear_forecast_equals_training_mean_plus_feature_contributions"] = error < 1e-12

    # Same-week quantiles track relative article positions, not an absolute
    # sentiment threshold. A uniform increase in sentiment leaves ratios intact.
    news = pd.DataFrame({"published_at": pd.date_range("2025-01-01", periods=20, freq="h"),
                         "compound": np.linspace(-.4, .5, 20)})
    before, _ = weekly_sentiment(news, pd.Timestamp("2025-01-01"))
    news.compound += .2
    after, _ = weekly_sentiment(news, pd.Timestamp("2025-01-01"))
    checks["quantile_probe_mean_increases_but_tail_proportions_stay_equal"] = bool(
        np.isclose(after["compound_weekly"]-before["compound_weekly"], .2) and
        before["extremely_positive_ratio"] == after["extremely_positive_ratio"] and
        before["extremely_negative_ratio"] == after["extremely_negative_ratio"])
    sentiment = history.groupby("date")[["compound_weekly", "extremely_positive_ratio", "extremely_negative_ratio"]].first()
    stats = []
    for period, start, end in [("52_weeks_before_replay", "2024-07-31", "2025-07-23"),
                                ("68_replay_weeks", "2025-07-30", "2026-11-11")]:
        g = sentiment.loc[start:end]
        for feature in g.columns:
            stats.append({"period": period, "weeks": len(g), "feature": feature,
                          "mean": float(g[feature].mean()), "std": float(g[feature].std()),
                          "min": float(g[feature].min()), "max": float(g[feature].max())})
    write_csv(out / "sentiment_feature_regimes.csv", pd.DataFrame(stats))
    write_json(out / "sentiment_definition_probe.json", {
        "before_uniform_shift": before, "after_uniform_shift": after,
        "interpretation": "Week-specific q10/q90 tail proportions stay near ten percent despite a uniform change in sentiment. The mean sentiment feature remains responsive. Old-versus-new distribution differences do not isolate publisher changes, threshold conventions and actual sentiment changes.",
        "code_locations": {"old_global_quantiles": "crypto_project_pipeline.py:833",
                           "weekly_quantiles": "replay_core.py:weekly_sentiment"}})
    result = {"passed": all(checks.values()), "checks": {k: bool(v) for k,v in checks.items()},
              "max_training_mean_error": max(train_errors), "max_portfolio_metric_error": max(metric_errors),
              "max_linear_mean_decomposition_error": float(error), "score_rows": len(scores),
              "limitations": "Independent arithmetic and timing audit; no fresh holdout, causal identification of market regime, beta-neutral portfolio or execution certification."}
    write_json(out / "independent_verification.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Independent diagnosis verification failed")


if __name__ == "__main__":
    run()
