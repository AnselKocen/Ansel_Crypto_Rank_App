"""Independent saved-result and original-artifact audit for the target trial."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_core import ROOT, DEFAULT_OUTPUT, ALL_FEATURES, digest, train_predict, write_json
from replay_target_experiment import quality_reasons, target_frame


def audit():
    out = DEFAULT_OUTPUT / "target_experiment"
    run = out / json.loads((out / "latest_run.json").read_text())["path"]
    manifest = json.loads((run / "run_manifest.json").read_text())
    history = pd.read_csv(out / "common_feature_history.csv", float_precision="round_trip", parse_dates=["date"])
    prices = pd.read_csv(out / "reviewed_prices.csv", float_precision="round_trip", parse_dates=["date"])
    comparison = pd.read_csv(run / "forecast_actual_comparison.csv", float_precision="round_trip", parse_dates=["decision_date", "target_end"])
    summary = pd.read_csv(run / "model_summary.csv")
    events = pd.read_csv(run / "run_events.csv")
    valid = comparison[comparison.evaluation_status.eq("evaluated")]
    keys = ["decision_date", "model_name", "symbol"]
    old = valid[valid.target_variant.eq("legacy_clipped")].set_index(keys).sort_index()
    new = valid[valid.target_variant.eq("raw")].set_index(keys).sort_index()
    quotes = prices.set_index(["symbol", "date"]).close.to_dict()
    independently_calculated = np.array([quotes[(r.symbol, r.target_end)] / quotes[(r.symbol, r.decision_date)] - 1
                                        for r in valid.itertuples()])
    rules = json.loads((ROOT / "replay_quality_rules.json").read_text())
    old_run = DEFAULT_OUTPUT / json.loads((DEFAULT_OUTPUT / "latest_run.json").read_text())["path"]
    old_manifest = json.loads((old_run / "run_manifest.json").read_text())
    old_audit = json.loads((DEFAULT_OUTPUT / "prediction_calculation_audit.json").read_text())
    checks = {
        "paired_evaluation_keys_identical": old.index.equals(new.index),
        "paired_actual_returns_identical": np.array_equal(old.actual_return_raw, new.actual_return_raw),
        "all_68_weeks_and_eight_models_per_week": events.groupby("decision_date").size().eq(8).all() and events.decision_date.nunique() == 68,
        "all_model_runs_complete": events.status.eq("complete").all(),
        "unique_predictions": not comparison.duplicated(keys + ["target_variant"]).any(),
        "independent_close_return_formula": np.allclose(valid.actual_return_raw, independently_calculated, atol=1e-14, rtol=1e-14),
        "pending_actuals_missing": comparison.loc[comparison.evaluation_status.eq("pending"), "actual_return_raw"].isna().all(),
        "no_quarantined_feature_rows": quality_reasons(history, rules).eq("").all(),
        "no_quarantined_daily_rows": quality_reasons(prices, rules).eq("").all(),
        "qnt_real_extreme_retained": valid[valid.symbol.eq("QNT") & valid.target_end.eq("2026-09-30")].actual_return_raw.gt(3).all(),
        "msq_real_extreme_retained": valid[valid.symbol.eq("MSQ") & valid.target_end.eq("2026-05-13")].actual_return_raw.gt(2).all(),
        "no_om_post_migration_forecasts": not ((comparison.symbol.eq("OM")) & comparison.decision_date.ge("2026-03-02")).any(),
        "no_prelauch_pi_splice_in_training_history": not ((history.symbol.eq("PI")) & history.date.lt("2025-06-24")).any(),
        "old_core_file_unchanged": digest(ROOT / "replay_core.py") == old_manifest["identity"]["core"],
        "old_forecasts_unchanged": digest(old_run / "forecast_history.csv") == old_audit["source_hashes"]["forecast"],
        "old_comparison_unchanged": digest(old_run / "forecast_actual_comparison.csv") == old_audit["source_hashes"]["comparison"],
        "old_raw_prices_unchanged": digest(DEFAULT_OUTPUT / "raw_prices.csv") == old_audit["source_hashes"]["raw_prices"],
        "experiment_code_matches_manifest": digest(ROOT / "replay_target_experiment.py") == manifest["identity"]["experiment"],
        "identical_input_feature_columns": target_frame(history, "raw")[ALL_FEATURES].equals(target_frame(history, "legacy_clipped")[ALL_FEATURES]),
    }
    for row in summary.itertuples():
        g = valid[valid.target_variant.eq(row.target_variant) & valid.model_name.eq(row.model_name)]
        checks[f"summary_{row.target_variant}_{row.model_name}"] = bool(
            len(g) == row.evaluated_rows and
            np.isclose((g.predicted_return-g.actual_return_raw).abs().mean(), row.mae_raw, atol=1e-14) and
            np.isclose(g.actual_return_raw.abs().mean(), row.zero_prediction_mae_raw, atol=1e-14))
    # Refit a late, fully out-of-sample period from the saved input, rather than
    # trusting that checkpoint existence proves that models were trained.
    refit = []
    decision = pd.Timestamp("2026-11-04")
    for variant in ["legacy_clipped", "raw"]:
        fresh, _ = train_predict(target_frame(history, variant), decision)
        saved = comparison[comparison.decision_date.eq(decision) & comparison.target_variant.eq(variant)]
        merged = fresh.merge(saved, on=["symbol", "model_name"], suffixes=("_fresh", "_saved"), validate="one_to_one")
        delta = float((merged.predicted_return_fresh-merged.predicted_return_saved).abs().max())
        refit.append({"variant": variant, "rows": len(merged), "max_absolute_difference": delta})
        checks[f"saved_predictions_reproduced_{variant}"] = delta < 1e-12 and len(merged) == len(saved)
    page = (run / "report/index.html").read_text()
    payload = json.loads(page.split('<script type="application/json" id="payload">', 1)[1].split('</script>', 1)[0])
    checks["html_payload_matches_run"] = payload["manifest"]["run_id"] == run.name and len(payload["rows"]) == len(comparison)
    result = {"run_id": run.name, "checks": {k: bool(v) for k, v in checks.items()}, "passed": all(checks.values()),
              "independent_refits": refit, "checks_count": len(checks), "evaluated_rows_per_variant": len(old),
              "limitations": "Audits arithmetic and controlled pairing; does not certify all third-party historical quotes."}
    write_json(out / "verification.json", result)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    audit()
