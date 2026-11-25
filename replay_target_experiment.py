#!/usr/bin/env python3
"""Paired weekly trial: unchanged estimators/features, clipped versus raw labels.

The archived first replay is never overwritten. Both arms use the same reviewed
input, calendar joins, training rows and feature values. Unresolved instruments
are quarantined rather than silently adjusted or selected for their returns.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from replay_core import (ROOT, DEFAULT_OUTPUT, ALL_FEATURES, MARKET_FEATURES,
                         MODEL_PARAMS, SENTIMENT_FEATURES, digest, legacy_market_function,
                         train_predict, weekly_market, write_csv, write_json)
from replay_weekly_predictions import compare_actual, model_metrics, strategy_results

VARIANTS = ("legacy_clipped", "raw")


def quality_reasons(frame, config, date_column="date"):
    reasons = pd.Series("", index=frame.index, dtype=object)
    dates = pd.to_datetime(frame[date_column])
    for rule in config["rules"]:
        mask = frame.symbol.eq(rule["symbol"])
        if "start" in rule:
            mask &= dates.ge(pd.Timestamp(rule["start"]))
        if "end" in rule:
            mask &= dates.le(pd.Timestamp(rule["end"]))
        reasons.loc[mask] = rule["reason"]
    return reasons


def exact_weekly_returns(history):
    """Never bridge a missing week, including one removed for data quality."""
    previous = history[["symbol", "date", "close"]].copy()
    previous["date"] += pd.Timedelta(days=7)
    previous = previous.rename(columns={"close": "previous_close"})
    joined = history[["symbol", "date", "close"]].merge(
        previous, on=["symbol", "date"], how="left", validate="one_to_one")
    raw = joined.close / joined.previous_close - 1
    raw = raw.where((joined.close > 0) & (joined.previous_close > 0))
    return pd.Series(raw.to_numpy(), index=history.index)


def target_frame(history, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    result = history.copy()
    # 'return' is excluded from both original feature sets. In particular,
    # strev_weekly and all momentum/volatility features stay unchanged.
    result["return"] = history["raw_weekly_return" if variant == "raw" else "clipped_weekly_return"]
    return result


def prepare(source, output, start, end, rules):
    output.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(source / "raw_prices.csv", parse_dates=["date"])
    prices = prices[prices.date.le(end)].copy()
    repairs = []
    for repair in rules.get("repairs", []):
        evidence = output / repair["evidence"]
        if not evidence.exists():
            raise FileNotFoundError(f"Required independently retrieved repair evidence: {evidence}")
        candles = {pd.to_datetime(row[0], unit="s"): row for row in json.loads(evidence.read_text())}
        mask = (prices.symbol.eq(repair["symbol"]) &
                prices.date.between(repair["start"], repair["end"]))
        for idx in prices.index[mask]:
            row = candles.get(prices.at[idx, "date"])
            if row is None:
                raise ValueError("Missing exact-date Coinbase repair candle")
            for field, value in zip(["open", "high", "low", "close"], [row[3], row[2], row[1], row[4]]):
                repairs.append({"symbol": prices.at[idx, "symbol"], "date": prices.at[idx, "date"],
                                "field": field, "original": prices.at[idx, field], "corrected": value,
                                "evidence": str(evidence.relative_to(output))})
                prices.at[idx, field] = value
            prices.at[idx, "base_volume"] = prices.at[idx, "usd_volume"] / prices.at[idx, "close"]
            prices.at[idx, "source"] = "coinbase_ohlc_yahoo_volume"
    prices["quality_reason"] = quality_reasons(prices, rules)
    write_csv(output / "quarantined_daily_prices.csv", prices[prices.quality_reason.ne("")])
    prices = prices[prices.quality_reason.eq("")].copy()
    write_csv(output / "reviewed_prices.csv", prices)
    write_csv(output / "price_repairs.csv", pd.DataFrame(repairs))

    baseline = pd.read_csv(source / "snapshot/baseline_history_2025-07-30.csv", parse_dates=["date"])
    baseline = baseline.drop(columns=["ret_lead1"], errors="ignore")
    baseline["sentiment_available"] = True
    baseline = baseline[baseline.date.lt(start)].copy()
    baseline["quality_reason"] = quality_reasons(baseline, rules)
    write_csv(output / "quarantined_baseline_rows.csv", baseline[baseline.quality_reason.ne("")])
    baseline = baseline[baseline.quality_reason.eq("")].copy()
    old_new = pd.read_csv(source / "weekly_feature_history.csv", parse_dates=["date"])
    first = pd.read_csv(source / "snapshot/baseline_history_2025-07-30.csv", parse_dates=["date"])
    sentiments = pd.concat([first[first.date.eq(start)], old_new], ignore_index=True).drop_duplicates("date")
    sentiments = sentiments.set_index("date")
    market_function, additions = legacy_market_function(), []
    weeks = pd.date_range(start, end, freq="W-WED")
    for number, date in enumerate(weeks, 1):
        frame = weekly_market(prices, date, market_function)
        if date not in sentiments.index:
            raise ValueError(f"No original weekly news features for {date}")
        for column in SENTIMENT_FEATURES:
            frame[column] = sentiments.loc[date, column]
        frame["sentiment_available"] = frame[SENTIMENT_FEATURES].notna().all(axis=1)
        frame["quality_reason"] = quality_reasons(frame, rules)
        frame = frame[frame.quality_reason.eq("")].copy()
        additions.append(frame)
        print(f"PREPARE {number}/{len(weeks)} {date:%Y-%m-%d} eligible={len(frame)}", flush=True)
    history = pd.concat([baseline, *additions], ignore_index=True).drop_duplicates(["date", "symbol"], keep="last")
    history["raw_weekly_return"] = exact_weekly_returns(history)
    # For replay dates use the identical provider's explicit daily close pair.
    # This avoids a baseline-CADLI / Yahoo boundary return at the replay start.
    closes = prices.set_index(["symbol", "date"]).close.to_dict()
    for idx in history.index[history.date.ge(start)]:
        s, d = history.at[idx, "symbol"], history.at[idx, "date"]
        a, b = closes.get((s, d)), closes.get((s, d - pd.Timedelta(days=7)))
        history.at[idx, "raw_weekly_return"] = a / b - 1 if a and b else np.nan
    history["clipped_weekly_return"] = history["return"].where(history.raw_weekly_return.notna())
    write_csv(output / "common_feature_history.csv", history)
    manifest = {"status": "prepared_restricted_data_quality_trial", "start": str(start.date()),
                "end": str(end.date()), "quality_rules": rules,
                "original_prices_sha256": digest(source / "raw_prices.csv"),
                "original_features_sha256": digest(source / "weekly_feature_history.csv"),
                "common_features_sha256": digest(output / "common_feature_history.csv"),
                "reviewed_prices_sha256": digest(output / "reviewed_prices.csv"),
                "repairs_count": len(repairs), "rules_sha256": digest(ROOT / "replay_quality_rules.json"),
                "preparation_code_sha256": digest(Path(__file__)),
                "limitations": ["Retrospective quality-controlled subset, not an independently verified full-universe trading backtest.",
                    "All previously stated Yahoo volume estimates, publisher changes and fixed-universe limitations remain.",
                    "Baseline features before replay start remain frozen originals except explicitly quarantined rows.",
                    "Do not compare this trial to the old unreviewed run to attribute effects solely to target clipping.",
                    "Quality exclusions were determined in this retrospective audit; this is not a claim of contemporaneous data availability.",
                    "No guessed split adjustment, no automatic deletion solely for large returns, no parameter optimization."]}
    write_json(output / "preparation_manifest.json", manifest)
    return history, prices, manifest


def summarize(comparison):
    rows = []
    valid = comparison[comparison.evaluation_status.eq("evaluated")]
    for (variant, model), g in valid.groupby(["target_variant", "model_name"]):
        p, a = g.predicted_return, g.actual_return_raw
        ic = g.groupby("decision_date").apply(
            lambda w: w.predicted_return.corr(w.actual_return_raw, method="spearman"), include_groups=False)
        rows.append({"target_variant": variant, "model_name": model, "evaluated_rows": len(g),
                     "evaluated_weeks": g.decision_date.nunique(), "symbols": g.symbol.nunique(),
                     "mae_raw": float((p-a).abs().mean()), "rmse_raw": float(np.sqrt(((p-a)**2).mean())),
                     "mae_clipped": float((p-g.actual_return_model_target).abs().mean()),
                     "zero_prediction_mae_raw": float(a.abs().mean()),
                     "last_week_prediction_mae_raw": float((g.last_week_return-a).abs().mean()),
                     "direction_accuracy": float((np.sign(p)==np.sign(a)).mean()),
                     "mean_weekly_rank_ic": float(ic.mean()), "prediction_std": float(p.std()),
                     "prediction_abs_median": float(p.abs().median()),
                     "prediction_within_2pct": float(p.abs().le(.02).mean())})
    return pd.DataFrame(rows)


def run(args):
    source, output = DEFAULT_OUTPUT, args.output
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    rules = json.loads((ROOT / "replay_quality_rules.json").read_text())
    required = [output / f for f in ["common_feature_history.csv", "reviewed_prices.csv", "preparation_manifest.json"]]
    current = json.loads(required[2].read_text()) if required[2].exists() else {}
    if (not all(p.exists() for p in required) or args.rebuild_data or
            current.get("start") != args.start or current.get("end") != args.end or
            current.get("preparation_code_sha256") != digest(Path(__file__)) or
            current.get("rules_sha256") != digest(ROOT / "replay_quality_rules.json")):
        history, prices, prep = prepare(source, output, start, end, rules)
    else:
        prep = current
    # Canonical persisted inputs for first runs and resumes alike. The default
    # CSV parser can round nearly tied values differently, changing tree splits.
    history = pd.read_csv(required[0], float_precision="round_trip", parse_dates=["date"])
    prices = pd.read_csv(required[1], float_precision="round_trip", parse_dates=["date"])
    if args.prepare_only:
        return
    identity = {"start": args.start, "end": args.end, "input": digest(required[0]),
                "prices": digest(required[1]), "rules": digest(ROOT / "replay_quality_rules.json"),
                "core": digest(ROOT / "replay_core.py"), "experiment": digest(Path(__file__)),
                "params": MODEL_PARAMS}
    run_id = "targets_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    directory = output / "runs" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    forecasts, events = [], []
    weeks = pd.date_range(start, end, freq="W-WED")
    for number, decision in enumerate(weeks, 1):
        for variant in VARIANTS:
            file = directory / "weekly_forecasts" / f"{decision:%Y-%m-%d}_{variant}.csv"
            event_file = file.with_suffix(".json")
            if file.exists() and event_file.exists() and not args.force:
                prediction = pd.read_csv(file, float_precision="round_trip", parse_dates=[
                    "decision_date", "target_start", "target_end", "train_start", "train_end"])
                week_events = json.loads(event_file.read_text())
            else:
                prediction, week_events = train_predict(target_frame(history, variant), decision)
                prediction["target_variant"] = variant
                prediction["forecast_id"] = variant + ":" + prediction.forecast_id
                prediction["run_id"] = run_id
                write_csv(file, prediction)
                write_json(event_file, week_events)
            forecasts.append(prediction)
            events.extend({"decision_date": str(decision.date()), "target_variant": variant, **e} for e in week_events)
        print(f"TRIAL {number}/{len(weeks)} {decision:%Y-%m-%d} four_models_x_two_targets", flush=True)
        write_csv(directory / "run_events.csv", pd.DataFrame(events))
    forecast = pd.concat(forecasts, ignore_index=True)
    comparison = compare_actual(forecast, prices, history, end)
    # Rank separately within each experiment arm.
    comparison["actual_rank"] = comparison.groupby(["target_variant", "decision_date", "model_name"]).actual_return_raw.rank(ascending=False, method="min")
    raw_map = history.set_index(["symbol", "date"]).raw_weekly_return.to_dict()
    comparison["last_week_return"] = [raw_map.get((s, d), np.nan) for s, d in zip(comparison.symbol, comparison.decision_date)]
    reasons = quality_reasons(comparison, rules, "target_end")
    comparison.loc[comparison.evaluation_status.eq("missing_price") & reasons.ne(""), "evaluation_status"] = "quarantined_price"
    comparison.loc[comparison.evaluation_status.eq("quarantined_price"), "missing_reason"] = reasons
    metrics, strategies, strategy_summaries = [], [], []
    for variant, frame in comparison.groupby("target_variant"):
        metric = model_metrics(frame); metric["target_variant"] = variant; metrics.append(metric)
        strategy, summary = strategy_results(frame)
        strategy["target_variant"], summary["target_variant"] = variant, variant
        strategies.append(strategy); strategy_summaries.append(summary)
    summary = summarize(comparison)
    for name, frame in [("forecast_history.csv", forecast), ("forecast_actual_comparison.csv", comparison),
                        ("model_summary.csv", summary), ("weekly_model_metrics.csv", pd.concat(metrics)),
                        ("strategy_returns.csv", pd.concat(strategies)), ("strategy_summary.csv", pd.concat(strategy_summaries))]:
        write_csv(directory / name, frame)
    paired = comparison.pivot(index=["decision_date", "model_name", "symbol"], columns="target_variant", values="predicted_return")
    valid_keys = [set(map(tuple, g[g.evaluation_status.eq("evaluated")][["decision_date", "model_name", "symbol"]].to_numpy()))
                  for _, g in comparison.groupby("target_variant")]
    checks = {"same_forecast_keys": not paired.isna().any().any(), "same_evaluated_keys": valid_keys[0] == valid_keys[1],
              "all_model_week_runs_complete": len(events) == len(weeks)*8 and all(e["status"] == "complete" for e in events),
              "finite_predictions": bool(np.isfinite(forecast.predicted_return).all()),
              "actual_exact_7_days": bool((forecast.target_end-forecast.decision_date).eq(pd.Timedelta(days=7)).all()),
              "features_identical_between_arms": target_frame(history, "legacy_clipped")[ALL_FEATURES].equals(target_frame(history, "raw")[ALL_FEATURES]),
              "same_training_label_availability": bool(history.raw_weekly_return.isna().equals(history.clipped_weekly_return.isna())),
              "old_core_unchanged": identity["core"] == digest(ROOT / "replay_core.py")}
    if not all(checks.values()):
        raise RuntimeError(f"Experiment audit failed: {checks}")
    manifest = {"run_id": run_id, "status": "complete_restricted_quality_trial", "start": args.start, "end": args.end,
                "identity": identity, "preparation": prep, "model_params": MODEL_PARAMS,
                "variants": {"legacy_clipped": "Original stored/corrected-feature clipped weekly targets", "raw": "Unclipped exact-calendar-week close returns"},
                "decision_weeks": len(weeks), "model_week_runs": len(events), "forecast_rows": len(forecast),
                "status_counts": comparison.evaluation_status.value_counts().to_dict(), "checks": checks,
                "changes": "Only the training label differs between paired arms. Features, eligible rows, algorithms and fixed parameters are identical.",
                "source_note": "Current archived publishers and quote sources; no publication-time data vintages. Research comparison, not contemporaneously saved live predictions."}
    write_json(directory / "run_manifest.json", manifest)
    write_json(output / "latest_run.json", {"run_id": run_id, "path": str(directory.relative_to(output))})
    from replay_target_report import build_report
    build_report(directory, output)
    print("TRIAL_COMPLETE " + str(directory), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-07-30")
    parser.add_argument("--end", default="2026-11-11")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT / "target_experiment")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())
