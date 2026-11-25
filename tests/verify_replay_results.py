"""Audit saved real-run artifacts without rerunning model selection or training."""
from __future__ import annotations
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_core import ROOT, DEFAULT_OUTPUT, MODEL_PARAMS, digest, write_json


def audit():
    out = DEFAULT_OUTPUT
    pointer = json.loads((out / "latest_run.json").read_text())
    run = out / pointer["path"]
    manifest = json.loads((run / "run_manifest.json").read_text())
    forecast = pd.read_csv(run / "forecast_history.csv")
    comparison = pd.read_csv(run / "forecast_actual_comparison.csv")
    events = pd.read_csv(run / "run_events.csv")
    for c in ["decision_date", "target_start", "target_end", "train_start", "train_end"]:
        forecast[c] = pd.to_datetime(forecast[c])
    for c in ["decision_date", "target_end"]:
        comparison[c] = pd.to_datetime(comparison[c])
    start, end = pd.Timestamp(manifest["start"]), pd.Timestamp(manifest["end"])
    expected = pd.date_range(start, end, freq="W-WED")
    valid = comparison[comparison.evaluation_status == "evaluated"]
    pending = comparison[comparison.evaluation_status == "pending"]
    checks = {
        "all_expected_decision_weeks": set(forecast.decision_date) == set(expected),
        "four_models_each_week": forecast.groupby("decision_date").model_name.nunique().eq(4).all(),
        "all_model_week_runs_complete": len(events) == len(expected) * 4 and events.status.eq("complete").all(),
        "unique_forecasts": not forecast.duplicated(["decision_date", "model_name", "symbol"]).any(),
        "finite_predictions": np.isfinite(forecast.predicted_return).all(),
        "target_is_next_calendar_week": (forecast.target_end - forecast.decision_date).eq(pd.Timedelta(days=7)).all(),
        "target_starts_next_day": (forecast.target_start - forecast.decision_date).eq(pd.Timedelta(days=1)).all(),
        "training_window_bounds": (
            (forecast.decision_date - forecast.train_start).eq(pd.Timedelta(weeks=52)).all()
            and (forecast.decision_date - forecast.train_end).eq(pd.Timedelta(days=7)).all()),
        "actual_return_formula": np.allclose(
            valid.actual_return_raw, valid.target_close / valid.reference_close - 1, atol=1e-10, rtol=1e-10),
        "evaluated_targets_finished": valid.target_end.le(end).all(),
        "pending_targets_unfinished": pending.target_end.gt(end).all(),
        "pending_returns_missing": pending.actual_return_raw.isna().all(),
        "same_source_for_actual_pair": valid.price_source.eq(valid.target_price_source).all(),
        "fixed_model_parameters": manifest["model_params"] == MODEL_PARAMS,
        "core_version_matches": manifest["identity"]["core"] == digest(ROOT / "replay_core.py"),
        "runner_version_matches": manifest["identity"]["runner"] == digest(ROOT / "replay_weekly_predictions.py"),
        "interactive_report_exists": (run / "report/index.html").stat().st_size > 100000,
        "exportable_figures_exist": all((run / "report" / f).is_file() for f in [
            "btc_prediction_vs_actual.png", "calibration_and_weekly_error.png", "strategy_nav.png"]),
    }
    original = json.loads((out / "snapshot/source_manifest.json").read_text())
    protected = [p for p, details in original.items()
                 if isinstance(details, dict) and p != "HISTORICAL_REPLAY_PLAN.md"]
    checks["original_models_and_data_unchanged"] = all(
        digest(ROOT / p) == original[p]["sha256"] for p in protected)
    payload_html = (run / "report/index.html").read_text()
    marker = '<script type="application/json" id="payload">'
    payload = json.loads(payload_html.split(marker, 1)[1].split("</script>", 1)[0])
    checks["html_rows_match_csv"] = len(payload["comparison"]) == len(comparison)
    checks["html_run_matches"] = payload["manifest"]["run_id"] == manifest["run_id"]
    result = {
        "run_id": manifest["run_id"], "checks": {k: bool(v) for k, v in checks.items()},
        "passed": all(checks.values()), "prediction_rows": len(forecast),
        "evaluated_rows": len(valid), "pending_rows": len(pending),
        "missing_price_rows": int(comparison.evaluation_status.eq("missing_price").sum()),
    }
    write_json(out / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
