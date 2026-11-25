#!/usr/bin/env python3
"""Diagnose the archived weekly forecast level and cross-sectional selection.

The experimental arm changes only the training label: subtract its same-date
cross-sectional mean. All labels must have matured by the decision close.
This is retrospective diagnosis, not a new untouched out-of-sample result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from replay_core import (ROOT, DEFAULT_OUTPUT, ALL_FEATURES, MARKET_FEATURES,
                         MODEL_PARAMS, digest, write_csv, write_json)

OUTPUT = DEFAULT_OUTPUT / "bias_diagnosis_v2"
MODELS = list(MODEL_PARAMS)


def training_data(history, decision):
    """Preserve original order and exact seven-calendar-day label maturity."""
    decision = pd.Timestamp(decision)
    visible = history[history.date.le(decision)].drop_duplicates(["date", "symbol"], keep="last").copy()
    columns = ["clipped_weekly_return", "raw_weekly_return"]
    labels = visible[["symbol", "date"]+columns].copy()
    labels["date"] -= pd.Timedelta(days=7)
    labels = labels.rename(columns={c: "next_"+c for c in columns})
    labeled = visible.merge(labels, on=["date", "symbol"], how="left", validate="one_to_one")
    required = pd.date_range(end=decision-pd.Timedelta(days=7), periods=52, freq="W-WED")
    if len(required.difference(visible.date.unique())):
        raise ValueError("Missing training calendar weeks")
    train = labeled[labeled.date.isin(required)].dropna(subset=["next_clipped_weekly_return"])
    test = labeled[labeled.date.eq(decision)].copy()
    return train, test


def center_labels(train):
    y = train.next_clipped_weekly_return
    return y-train.groupby("date").next_clipped_weekly_return.transform("mean")


def fit_model(train, test, model, target):
    features = ALL_FEATURES if model.endswith("_all") else MARKET_FEATURES
    estimator = (ElasticNet(max_iter=10000, random_state=42, **MODEL_PARAMS[model])
                 if model.startswith("enet") else ExtraTreesRegressor(random_state=42, **MODEL_PARAMS[model]))
    pipe = Pipeline([("impute", SimpleImputer(strategy="mean")),
                     ("scale", StandardScaler()), ("model", estimator)])
    xtrain = train[features].replace([np.inf, -np.inf], np.nan)
    xtest = test[features].replace([np.inf, -np.inf], np.nan)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", RuntimeWarning)
        pipe.fit(xtrain, target)
        prediction = pipe.predict(xtest)
    pipe.diagnostic_warnings_ = sorted(set(str(w.message) for w in captured))
    pipe.diagnostic_reconstruction_error_ = 0.
    if model.startswith("enet"):
        imputed = pipe.named_steps["impute"].transform(xtest)
        scaled = pipe.named_steps["scale"].transform(imputed)
        rebuilt = (scaled*pipe.named_steps["model"].coef_).sum(axis=1)+pipe.named_steps["model"].intercept_
        pipe.diagnostic_reconstruction_error_ = float(np.max(np.abs(prediction-rebuilt)))
        if not np.isfinite(rebuilt).all() or pipe.diagnostic_reconstruction_error_ > 1e-12:
            raise ValueError("Linear prediction failed independent elementwise reconstruction")
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite predictions")
    return prediction, pipe, features


def linear_decomposition(pipe, xtest, features):
    imputed = pipe.named_steps["impute"].transform(xtest[features].replace([np.inf, -np.inf], np.nan))
    scaled = pipe.named_steps["scale"].transform(imputed)
    estimator = pipe.named_steps["model"]
    contribution = scaled*estimator.coef_
    rebuilt = estimator.intercept_+contribution.sum(axis=1)
    return float(estimator.intercept_), contribution.mean(axis=0), scaled.mean(axis=0), rebuilt


def evaluate_scores(frame, score, rank=None):
    """Evaluate the whole preselected universe, never filter by future prices."""
    if len(frame) < 40:
        raise ValueError("Top/bottom 20 needs at least 40 assets")
    ordered = (frame.sort_values(rank, kind="stable") if rank else
               frame.sort_values(score, ascending=False, kind="stable"))
    if not frame.evaluation_status.eq("evaluated").all():
        return {"status": "incomplete_universe"}
    if float(np.ptp(frame[score])) <= 1e-12:
        return {"status": "no_ranking_signal", "universe_return": float(frame.actual_return_raw.mean())}
    top, bottom = ordered.head(20), ordered.tail(20)
    market = float(frame.actual_return_raw.mean())
    raw_ic = frame[score].corr(frame.actual_return_raw, method="spearman")
    centered_ic = (frame[score]-frame[score].mean()).corr(frame.actual_return_raw-market, method="spearman")
    return {"status": "evaluated", "universe_return": market,
            "rank_ic": float(raw_ic), "rank_ic_after_demeaning": float(centered_ic),
            "top_return": float(top.actual_return_raw.mean()),
            "bottom_return": float(bottom.actual_return_raw.mean()),
            "top_excess": float(top.actual_return_raw.mean()-market),
            "bottom_excess": float(bottom.actual_return_raw.mean()-market),
            "ls_return": float(top.actual_return_raw.mean()-bottom.actual_return_raw.mean())}


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    experiment = DEFAULT_OUTPUT / "target_experiment"
    source = experiment / json.loads((experiment / "latest_run.json").read_text())["path"]
    inputs = [experiment / "common_feature_history.csv", source / "forecast_actual_comparison.csv",
              DEFAULT_OUTPUT / "portfolio_evaluation/portfolio_weekly_returns.csv",
              ROOT / "replay_core.py", ROOT / "replay_target_experiment.py",
              ROOT / "crypto_project_pipeline.py"]
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    identity = {"inputs": hashes, "code": digest(Path(__file__)), "params": MODEL_PARAMS}
    previous = output / "identity.json"
    if previous.exists() and json.loads(previous.read_text()) != identity:
        raise ValueError("Different code/input identity; use a separate --output directory")
    write_json(previous, identity)
    history = pd.read_csv(inputs[0], float_precision="round_trip", parse_dates=["date"])
    forecast = pd.read_csv(inputs[1], float_precision="round_trip", parse_dates=["decision_date", "target_end"])
    forecast = forecast[forecast.target_variant.eq("legacy_clipped")].copy()
    weekly = pd.read_csv(inputs[2], float_precision="round_trip", parse_dates=["decision_date"])
    common = set(weekly.loc[weekly.common_week, "decision_date"])
    continuous = set(weekly.loc[weekly.longest_continuous_window, "decision_date"])
    decisions = sorted(forecast.decision_date.unique())
    if args.limit:
        decisions = decisions[:args.limit]
    max_original_diff, max_linear_diff = 0., 0.
    numeric_warnings = set()
    levels, coef_rows, metrics, centered_forecasts = [], [], [], []
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for i, decision in enumerate(decisions, 1):
        decision = pd.Timestamp(decision)
        path = checkpoints / f"{decision:%Y-%m-%d}.json"
        if path.exists():
            saved = json.loads(path.read_text())
        else:
            train, test = training_data(history, decision)
            y = train.next_clipped_weekly_return
            train_mean = float(y.mean())
            saved = {"levels": [], "coefficients": [], "metrics": [], "forecasts": [],
                     "max_original_diff": 0., "max_linear_diff": 0., "numeric_warnings": []}
            for model in MODELS:
                actual = forecast[forecast.decision_date.eq(decision) & forecast.model_name.eq(model)].copy()
                eligible = test[test.symbol.isin(actual.symbol)].copy()
                if set(eligible.symbol) != set(actual.symbol):
                    raise ValueError("Different prediction universe")
                data = eligible[["symbol"]].merge(actual, on="symbol", validate="one_to_one")
                row = {"decision_date": str(decision.date()), "model_name": model,
                       "training_rows": len(train), "training_dates": train.date.nunique(),
                       "train_label_mean": train_mean,
                       "train_raw_label_mean": float(train.next_raw_weekly_return.mean()),
                       "train_equal_week_label_mean": float(train.groupby("date").next_clipped_weekly_return.mean().mean()),
                       "predicted_universe_mean": float(data.predicted_return.mean()),
                       "feature_model_adjustment_to_train_mean": float(data.predicted_return.mean()-train_mean),
                       "actual_universe_return": float(data.actual_return_raw.mean()) if data.evaluation_status.eq("evaluated").all() else None,
                       "common_week": decision in common, "continuous_window": decision in continuous}
                if model.startswith("enet"):
                    prediction, pipe, features = fit_model(train, eligible, model, y)
                    saved["numeric_warnings"] += pipe.diagnostic_warnings_
                    delta = float(np.max(np.abs(prediction-data.predicted_return.to_numpy())))
                    saved["max_original_diff"] = max(saved["max_original_diff"], delta)
                    intercept, contributions, zmeans, rebuilt = linear_decomposition(pipe, eligible, features)
                    linear_delta = float(np.max(np.abs(rebuilt-prediction)))
                    saved["max_linear_diff"] = max(saved["max_linear_diff"], linear_delta)
                    if abs(intercept-train_mean) > 1e-12:
                        raise ValueError("Linear intercept does not equal training target mean")
                    row["linear_intercept"] = intercept
                    row["nonzero_coefficients"] = int(np.count_nonzero(pipe.named_steps["model"].coef_))
                    row["linear_mean_feature_contribution"] = float(contributions.sum())
                    for feature, coef, contribution, zmean in zip(features, pipe.named_steps["model"].coef_, contributions, zmeans):
                        saved["coefficients"].append({"decision_date": str(decision.date()), "model_name": model,
                            "feature": feature, "standardized_coefficient": float(coef),
                            "test_mean_standardized_feature": float(zmean), "mean_prediction_contribution": float(contribution)})
                # Reproduce original trees at three dates as a real-input control.
                elif i in {1, len(decisions)//2, len(decisions)}:
                    prediction, original_pipe, _ = fit_model(train, eligible, model, y)
                    saved["numeric_warnings"] += original_pipe.diagnostic_warnings_
                    delta = float(np.max(np.abs(prediction-data.predicted_return.to_numpy())))
                    saved["max_original_diff"] = max(saved["max_original_diff"], delta)
                prediction, centered_pipe, _ = fit_model(train, eligible, model, center_labels(train))
                saved["numeric_warnings"] += centered_pipe.diagnostic_warnings_
                saved["max_linear_diff"] = max(saved["max_linear_diff"], centered_pipe.diagnostic_reconstruction_error_)
                data["centered_label_score"] = prediction
                saved["levels"].append(row)
                for arm, score, rank in [("original", "predicted_return", "predicted_rank"),
                                         ("date_centered_training_label", "centered_label_score", None)]:
                    metric = evaluate_scores(data, score, rank)
                    metric.update(decision_date=str(decision.date()), model_name=model, arm=arm,
                                  common_week=decision in common, continuous_window=decision in continuous)
                    saved["metrics"].append(metric)
                for row in data[["symbol", "predicted_return", "predicted_rank", "centered_label_score"]].to_dict("records"):
                    saved["forecasts"].append({"decision_date": str(decision.date()), "model_name": model, **row})
            if saved["max_original_diff"] > 1e-12 or saved["max_linear_diff"] > 1e-12:
                raise ValueError("Original forecast or linear decomposition reproduction failed")
            write_json(path, saved)
        levels.extend(saved["levels"])
        coef_rows.extend(saved["coefficients"])
        metrics.extend(saved["metrics"])
        centered_forecasts.extend(saved["forecasts"])
        max_original_diff = max(max_original_diff, saved["max_original_diff"])
        max_linear_diff = max(max_linear_diff, saved["max_linear_diff"])
        numeric_warnings.update(saved.get("numeric_warnings", []))
        print(f"DIAGNOSE {i}/{len(decisions)} {decision:%Y-%m-%d} label level + relative-return refit", flush=True)
    level = pd.DataFrame(levels)
    metric = pd.DataFrame(metrics)
    coef = pd.DataFrame(coef_rows)
    summaries, level_summaries, paired_summaries = [], [], []
    for scope, column in [("all_common_59_weeks", "common_week"), ("continuous_27_weeks", "continuous_window")]:
        for (model, arm), g in metric[metric[column]].groupby(["model_name", "arm"]):
            if not g.status.isin(["evaluated", "no_ranking_signal"]).all():
                raise ValueError("New portfolio missing an outcome on original shared mask")
            r = {"scope": scope, "model_name": model, "arm": arm, "weeks": len(g),
                 "signal_weeks": int(g.status.eq("evaluated").sum()),
                 "no_signal_weeks": int(g.status.eq("no_ranking_signal").sum())}
            # Do not report a complete-window portfolio return for an arm that
            # supplies no ranking on some dates. No guessed tiebreak or cash fill.
            if not g.status.eq("evaluated").all():
                summaries.append(r)
                continue
            for c in ["rank_ic", "top_return", "bottom_return", "top_excess", "bottom_excess", "ls_return"]:
                r["mean_"+c] = float(g[c].mean())
            r["top_beats_universe_ratio"] = float(g.top_excess.gt(0).mean())
            r["top_beats_bottom_ratio"] = float(g.ls_return.gt(0).mean())
            if column == "continuous_window":
                d = pd.to_datetime(g.sort_values("decision_date").decision_date)
                if len(d) > 1 and not d.diff().dropna().eq(pd.Timedelta(days=7)).all():
                    raise ValueError("Cannot compound across gaps")
                r["continuous_long_total_return"] = float((1+g.top_return).prod()-1)
                r["continuous_ls_total_return"] = float((1+g.ls_return).prod()-1)
            summaries.append(r)
        # Paired descriptive statistics use the SAME signal dates for both arms.
        # These subsets can have gaps and therefore receive no cumulative NAV.
        for model, group in metric[metric[column]].groupby("model_name"):
            usable = group.groupby("decision_date").status.apply(lambda s: s.eq("evaluated").all())
            dates = set(usable[usable].index)
            for arm, g in group[group.decision_date.isin(dates)].groupby("arm"):
                r = {"scope": scope, "model_name": model, "arm": arm, "paired_signal_weeks": len(g)}
                for c in ["rank_ic", "top_excess", "bottom_excess", "ls_return"]:
                    r["mean_"+c] = float(g[c].mean())
                paired_summaries.append(r)
        for model, g in level[level[column]].groupby("model_name"):
            r = {"scope": scope, "model_name": model, "weeks": len(g),
                 "negative_train_label_mean_weeks": int(g.train_label_mean.lt(0).sum()),
                 "negative_train_raw_label_mean_weeks": int(g.train_raw_label_mean.lt(0).sum()),
                 "negative_predicted_mean_weeks": int(g.predicted_universe_mean.lt(0).sum()),
                 "actual_positive_weeks": int(g.actual_universe_return.gt(0).sum()),
                 "train_mean_direction_matches_prediction_ratio": float(np.sign(g.train_label_mean).eq(np.sign(g.predicted_universe_mean)).mean()),
                 "train_mean_forecast_correlation": float(g.train_label_mean.corr(g.predicted_universe_mean)),
                 "mean_train_label": float(g.train_label_mean.mean()),
                 "mean_train_raw_label": float(g.train_raw_label_mean.mean()),
                 "mean_prediction": float(g.predicted_universe_mean.mean()),
                 "mean_feature_model_adjustment": float(g.feature_model_adjustment_to_train_mean.mean()),
                 "mean_actual_universe_return": float(g.actual_universe_return.mean()),
                 "market_direction_hit_ratio": float(np.sign(g.predicted_universe_mean).eq(np.sign(g.actual_universe_return)).mean()),
                 "always_up_direction_hit_ratio": float(g.actual_universe_return.gt(0).mean()),
                 "mae_universe_forecast": float((g.predicted_universe_mean-g.actual_universe_return).abs().mean()),
                 "mae_universe_train_mean_baseline": float((g.train_label_mean-g.actual_universe_return).abs().mean()),
                 "mae_universe_zero_baseline": float(g.actual_universe_return.abs().mean())}
            level_summaries.append(r)
    summary = pd.DataFrame(summaries)
    checks = {"original_forecasts_reproduced_at_checked_dates": max_original_diff < 1e-12,
              "linear_mean_decomposition_exact": max_linear_diff < 1e-12,
              "centering_predictions_and_outcomes_does_not_change_same_week_rank_ic": bool(
                  np.allclose(metric.rank_ic, metric.rank_ic_after_demeaning, equal_nan=True, atol=1e-12)),
              "same_common_observed_weeks_all_arms": bool(summary[summary.scope.eq("all_common_59_weeks")].weeks.nunique() <= 1),
              "same_continuous_window_all_arms": bool(summary[summary.scope.eq("continuous_27_weeks")].weeks.nunique() <= 1),
              "source_files_unchanged": all(digest(ROOT/p) == h for p,h in hashes.items()),
              "finite_experimental_scores": bool(np.isfinite(pd.DataFrame(centered_forecasts).centered_label_score).all())}
    checks["no_complete_window_returns_for_missing_signal_arms"] = bool(
        summary.loc[summary.no_signal_weeks.gt(0), "mean_ls_return"].isna().all())
    for filename, f in [("weekly_prediction_levels.csv", level), ("linear_feature_contributions.csv", coef),
                        ("weekly_selection_metrics.csv", metric), ("selection_summary.csv", summary),
                        ("level_summary.csv", pd.DataFrame(level_summaries)),
                        ("paired_signal_summary.csv", pd.DataFrame(paired_summaries)),
                        ("centered_label_predictions.csv", pd.DataFrame(centered_forecasts))]:
        write_csv(output / filename, f)
    manifest = {"status": "complete_retrospective_diagnosis" if len(decisions) == forecast.decision_date.nunique() else "partial",
                "source_run": source.name, "decisions": len(decisions), "checks": checks,
                "passed": all(checks.values()), "max_original_prediction_error": max_original_diff,
                "max_linear_reconstruction_error": max_linear_diff, "identity": identity,
                "captured_numeric_warning_messages": sorted(numeric_warnings),
                "numeric_warning_validation": "Every linear prediction, including centered-label models, was finite and independently matched an elementwise sum rather than BLAS matmul to 1e-12. Warning root cause in the numeric environment was not established.",
                "flat_score_policy": "Score range <= 1e-12 means no ranking signal. Do not select by row order, guess returns, fill with cash or report complete-window NAV. Paired descriptive tables compare both arms only on identical nonflat dates.",
                "training_label_control": "Same original clipped training returns minus each feature date's mean over identical matured labeled rows. No new raw-return arm, universe change or parameter search.",
                "evaluation": "Raw realized returns on the unchanged 59 shared observed weeks and 27 continuous weeks. Subtracting same-week market mean changes neither original ordering nor same-week Rank IC.",
                "limitations": ["Retrospective experiment after viewing outcomes, no fresh holdout or statistical significance claim.",
                    "Date-mean subtraction removes the common additive return component; it is not beta-neutralization.",
                    "Training and evaluation universe means refer to their respective eligible samples.",
                    "All prior quote-quality, source changes, fixed-universe, execution and cost limitations remain.",
                    "Negative training mean can explain model level mechanically, not identify the cause of economic regime changes."]}
    write_json(output / "diagnosis_manifest.json", manifest)
    render_chart(output, level, summary)
    print(json.dumps({"passed": manifest["passed"], "decisions": len(decisions),
                      "checks": checks, "output": str(output)}, indent=2), flush=True)
    if not manifest["passed"]:
        raise SystemExit("Diagnosis checks failed")


def render_chart(output, level, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    g = level[level.model_name.eq("enet_market")].copy()
    d = pd.to_datetime(g.decision_date)
    for column, label, color in [("actual_universe_return", "Realized universe weekly return", "#aaa"),
                                 ("train_label_mean", "Mean of matured training labels", "#e3a229"),
                                 ("predicted_universe_mean", "Mean ElasticNet market forecast", "#2468b4")]:
        axes[0].plot(d, g[column], label=label, color=color, linewidth=1.5)
    axes[0].axhline(0, color="black", linewidth=.5)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8)
    axes[0].set_title("Forecast level versus the previous 52 weeks' training-label mean")
    for j, arm in enumerate(["original", "date_centered_training_label"]):
        s = summary[summary.scope.eq("all_common_59_weeks") & summary.arm.eq(arm)].set_index("model_name").reindex(MODELS)
        axes[1].bar(np.arange(4)+(j-.5)*.35, s.mean_ls_return, width=.35,
                    label="Original target" if j == 0 else "Same-date centered training target")
    axes[1].set_xticks(np.arange(4), MODELS)
    axes[1].axhline(0, color="black", linewidth=.5)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title("Mean Top-20 minus Bottom-20 weekly return: same 59 observed weeks")
    axes[1].legend(fontsize=8)
    fig.suptitle("Retrospective diagnosis; gross returns, prior coverage and execution limits remain")
    fig.tight_layout()
    fig.savefig(output / "bias_and_relative_selection.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int)
    run(parser.parse_args())
