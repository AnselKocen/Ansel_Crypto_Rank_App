#!/usr/bin/env python3
"""Replay the unchanged four weekly models, then score explicitly dated outcomes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from replay_core import (
    ROOT, DEFAULT_OUTPUT, ALL_FEATURES, MARKET_FEATURES, MODEL_PARAMS,
    digest, train_predict, write_csv, write_json,
)


def compare_actual(forecasts, prices, history, end):
    result = forecasts.copy()
    daily = prices.drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])
    closes = daily.close.to_dict()
    sources = daily.source.to_dict()
    targets = history.drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"])["return"].to_dict()
    result["reference_close"] = [closes.get((s, d), np.nan) for s, d in zip(result.symbol, result.decision_date)]
    result["target_close"] = [closes.get((s, d), np.nan) for s, d in zip(result.symbol, result.target_end)]
    result["price_source"] = [sources.get((s, d), "") for s, d in zip(result.symbol, result.decision_date)]
    result["target_price_source"] = [sources.get((s, d), "") for s, d in zip(result.symbol, result.target_end)]
    valid = ((result.reference_close > 0) & (result.target_close > 0) &
             result.price_source.eq(result.target_price_source))
    result["evaluation_status"] = np.where(
        result.target_end > end, "pending", np.where(valid, "evaluated", "missing_price"))
    result["actual_return_raw"] = np.where(
        result.evaluation_status.eq("evaluated"), result.target_close / result.reference_close - 1, np.nan)
    result["actual_return_model_target"] = [
        targets.get((s, d), np.nan) if d <= end else np.nan
        for s, d in zip(result.symbol, result.target_end)
    ]
    result["prediction_error_raw"] = result.predicted_return - result.actual_return_raw
    result["prediction_error_model_target"] = result.predicted_return - result.actual_return_model_target
    result["direction_hit"] = np.where(
        result.evaluation_status.eq("evaluated"),
        (np.sign(result.predicted_return) == np.sign(result.actual_return_raw)).astype(float), np.nan)
    result["actual_rank"] = result.groupby(["decision_date", "model_name"]).actual_return_raw.rank(
        ascending=False, method="min")
    result["missing_reason"] = np.where(
        result.evaluation_status.eq("pending"), "target_week_not_finished",
        np.where(result.evaluation_status.eq("missing_price"), "missing_or_incompatible_source_price", ""))
    return result


def model_metrics(comparison):
    rows = []
    for (date, model), group in comparison.groupby(["decision_date", "model_name"], sort=True):
        valid = group[group.evaluation_status == "evaluated"]
        r = {"decision_date": date, "target_end": group.target_end.iloc[0], "model_name": model,
             "forecast_count": len(group), "evaluated_count": len(valid),
             "pending_count": int(group.evaluation_status.eq("pending").sum()),
             "missing_count": int(group.evaluation_status.eq("missing_price").sum())}
        if len(valid):
            error = valid.prediction_error_raw
            r.update(mae=float(error.abs().mean()), rmse=float(np.sqrt((error ** 2).mean())),
                     direction_accuracy=float(valid.direction_hit.mean()),
                     mean_actual_return=float(valid.actual_return_raw.mean()),
                     mean_predicted_return=float(valid.predicted_return.mean()))
            if len(valid) > 2 and valid.predicted_return.nunique() > 1 and valid.actual_return_raw.nunique() > 1:
                r["rank_ic"] = float(valid.predicted_return.corr(valid.actual_return_raw, method="spearman"))
        rows.append(r)
    return pd.DataFrame(rows)


def strategy_results(comparison):
    rows = []
    for (date, model), group in comparison.groupby(["decision_date", "model_name"], sort=True):
        if group.target_end.iloc[0] > comparison.target_end.min() and group.evaluation_status.eq("pending").all():
            continue
        top, bottom = group[group.is_top20], group[group.is_bottom20]
        for strategy, selected in [("EW", top), ("LS", pd.concat([top, bottom]))]:
            complete = len(top) == 20 and (strategy == "EW" or len(bottom) == 20)
            complete = complete and selected.evaluation_status.eq("evaluated").all()
            ret = np.nan
            if complete:
                ret = top.actual_return_raw.mean()
                if strategy == "LS":
                    ret -= bottom.actual_return_raw.mean()
            rows.append({"decision_date": date, "target_end": group.target_end.iloc[0],
                         "model_name": model, "strategy": strategy,
                         "weekly_return": ret, "status": "evaluated" if complete else "missing_constituents",
                         "gross_exposure": 1 if strategy == "EW" else 2,
                         "cost_assumption": "gross_no_fees_or_funding"})
        complete = group.evaluation_status.eq("evaluated").all()
        rows.append({"decision_date": date, "target_end": group.target_end.iloc[0],
                     "model_name": model, "strategy": "baseline_EW",
                     "weekly_return": group.actual_return_raw.mean() if complete else np.nan,
                     "status": "evaluated" if complete else "missing_constituents",
                     "gross_exposure": 1, "cost_assumption": "gross_no_fees_or_funding"})
    # Preserve the original two-model fusion weights, evaluated on matching symbols.
    for date, g in comparison.groupby("decision_date", sort=True):
        for tag in ["all", "market"]:
            a, b = g[g.model_name == f"enet_{tag}"], g[g.model_name == f"extra_{tag}"]
            if a.empty or b.empty or a.evaluation_status.eq("pending").all():
                continue
            weights = {}
            for field, sign in [("is_top20", 1), ("is_bottom20", -1)]:
                sa, sb = set(a[a[field]].symbol), set(b[b[field]].symbol)
                raw = {s: (0.65 if s in sa & sb else 0.35) for s in sorted(sa | sb)}
                total = sum(raw.values())
                for s, w in raw.items():
                    weights[s] = weights.get(s, 0) + sign * w / total
            actual = g.drop_duplicates("symbol").set_index("symbol")
            valid = all(s in actual.index and actual.loc[s, "evaluation_status"] == "evaluated" for s in weights)
            ret = sum(w * actual.loc[s, "actual_return_raw"] for s, w in weights.items()) if valid else np.nan
            rows.append({"decision_date": date, "target_end": a.target_end.iloc[0],
                         "model_name": f"fusion_{tag}", "strategy": "LS", "weekly_return": ret,
                         "status": "evaluated" if valid else "missing_constituents",
                         "gross_exposure": 2, "cost_assumption": "gross_no_fees_or_funding"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, pd.DataFrame()
    frame = frame.sort_values(["model_name", "strategy", "target_end"]).reset_index(drop=True)
    frame["cumulative_value"], frame["drawdown"], frame["segment_id"] = np.nan, np.nan, 0
    summaries = []
    for (model, strategy), group in frame.groupby(["model_name", "strategy"]):
        segment, cumulative, peak, previous = 0, 1.0, 1.0, None
        for idx, row in group.iterrows():
            if pd.isna(row.weekly_return):
                previous = None
                continue
            if previous is None or row.target_end - previous != pd.Timedelta(days=7):
                segment += 1
                cumulative, peak = 1.0, 1.0
            cumulative *= 1 + row.weekly_return
            peak = max(peak, cumulative)
            frame.loc[idx, ["cumulative_value", "drawdown", "segment_id"]] = [cumulative, cumulative / peak - 1, segment]
            previous = row.target_end
        valid = group.weekly_return.dropna()
        std = valid.std()
        summaries.append({"model_name": model, "strategy": strategy, "evaluated_weeks": len(valid),
                          "missing_weeks": int(group.weekly_return.isna().sum()),
                          "mean_weekly_return": valid.mean(),
                          "weekly_volatility": std, "annualized_volatility": std * np.sqrt(52),
                          "annualized_sharpe_rf_zero": valid.mean() / std * np.sqrt(52) if std > 0 else np.nan,
                          "minimum_segment_drawdown": frame.loc[group.index, "drawdown"].min(),
                          "continuity": "each contiguous segment restarts at 1; gaps are never zero-filled"})
    return frame, pd.DataFrame(summaries)


def run(args):
    out, start, end = args.output, pd.Timestamp(args.start), pd.Timestamp(args.end)
    preparation = json.loads((out / "preparation_manifest.json").read_text())
    baseline_file = out / "snapshot/baseline_history_2025-07-30.csv"
    history = pd.read_csv(baseline_file, parse_dates=["date"]).drop(columns=["ret_lead1"], errors="ignore")
    history = history[history.date <= start].copy()
    history["sentiment_available"] = True
    history["sentiment_source"] = "original_baseline"
    new = pd.read_csv(out / "weekly_feature_history.csv", parse_dates=["date"])
    new["sentiment_available"] = new.sentiment_available.astype(str).str.lower().eq("true")
    prices = pd.read_csv(out / "raw_prices.csv")
    prices["date"] = pd.to_datetime(prices.date, format="mixed", utc=True).dt.tz_localize(None)
    identity = {
        "start": args.start, "end": args.end, "baseline": digest(baseline_file),
        "features": digest(out / "weekly_feature_history.csv"),
        "prices": digest(out / "raw_prices.csv"), "models": MODEL_PARAMS,
        "core": digest(ROOT / "replay_core.py"), "runner": digest(Path(__file__)),
    }
    run_id = f"replay_{args.start}_{args.end}_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    run_dir = out / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    events, forecasts = [], []
    weeks = pd.date_range(start, end, freq="W-WED")
    if args.limit_weeks:
        weeks = weeks[:args.limit_weeks]
    for number, decision in enumerate(weeks, 1):
        added = new[new.date == decision]
        if decision > start:
            history = pd.concat([history, added], ignore_index=True)
            history = history.drop_duplicates(["date", "symbol"], keep="last")
        checkpoint = run_dir / "weekly_forecasts" / f"{decision:%Y-%m-%d}.csv"
        event_file = run_dir / "weekly_events" / f"{decision:%Y-%m-%d}.json"
        try:
            if checkpoint.exists() and event_file.exists() and not args.force:
                pred = pd.read_csv(checkpoint, float_precision="round_trip",
                                   parse_dates=["decision_date", "target_start", "target_end",
                                                "train_start", "train_end"])
                week_events = json.loads(event_file.read_text())
            else:
                pred, week_events = train_predict(history, decision)
                if not pred.empty:
                    pred["run_id"] = run_id
                    pred["model_config_hash"] = hashlib.sha256(json.dumps(MODEL_PARAMS, sort_keys=True).encode()).hexdigest()
                    pred["feature_schema_version"] = "original_24_all_16_market_v1"
                    write_csv(checkpoint, pred)
                    write_json(event_file, week_events)
            forecasts.append(pred)
            for event in week_events:
                events.append({"decision_date": str(decision.date()), **event})
            print(f"REPLAY {number}/{len(weeks)} {decision:%Y-%m-%d} " +
                  " ".join(f"{x['model']}={x['rows']}" for x in week_events), flush=True)
        except Exception as exc:
            events.extend({"decision_date": str(decision.date()), "model": name,
                           "status": "error", "error": str(exc), "rows": 0} for name in MODEL_PARAMS)
            print(f"REPLAY_ERROR {decision:%Y-%m-%d} {exc}", flush=True)
        write_csv(run_dir / "run_events.csv", pd.DataFrame(events))
    if not forecasts or all(f.empty for f in forecasts):
        raise RuntimeError("No predictions produced; inspect run_events.csv")
    forecast = pd.concat([f for f in forecasts if not f.empty], ignore_index=True)
    comparison = compare_actual(forecast, prices, history, end)
    metrics = model_metrics(comparison)
    strategies, strategy_summary = strategy_results(comparison)
    for filename, f in [
        ("forecast_history.csv", forecast), ("forecast_actual_comparison.csv", comparison),
        ("weekly_model_metrics.csv", metrics), ("strategy_returns.csv", strategies),
        ("strategy_summary.csv", strategy_summary),
    ]:
        write_csv(run_dir / filename, f)
    manifest = {
        "run_id": run_id, "start": args.start, "end": args.end,
        "status": "complete" if len(weeks) * 4 == sum(e["status"] == "complete" for e in events) else "partial",
        "simulation_not_live_predictions": True, "prepared_data": preparation,
        "identity": identity, "model_params": MODEL_PARAMS,
        "market_features": MARKET_FEATURES, "all_features": ALL_FEATURES,
        "decision_weeks": len(weeks), "model_week_statuses": len(events),
        "prediction_rows": len(forecast), "symbols": forecast.symbol.nunique(),
        "evaluated_rows": int(comparison.evaluation_status.eq("evaluated").sum()),
        "pending_rows": int(comparison.evaluation_status.eq("pending").sum()),
        "missing_price_rows": int(comparison.evaluation_status.eq("missing_price").sum()),
        "source_limitations": preparation.get("source_limitations", []),
        "costs": "Original gross portfolios: no fees, funding or slippage; LS gross exposure 200%.",
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "scikit-learn", "nltk", "plotly"]},
    }
    write_json(run_dir / "run_manifest.json", manifest)
    if not args.limit_weeks:
        write_json(out / "latest_run.json", {"run_id": run_id, "path": str(run_dir.relative_to(out))})
    from replay_report import build_report
    build_report(run_dir, out, prices)
    print(f"REPLAY_COMPLETE {run_dir} predictions={len(forecast)} status={manifest['status']}", flush=True)
    return run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2025-07-30")
    p.add_argument("--end", default="2026-11-11")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--limit-weeks", type=int)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
