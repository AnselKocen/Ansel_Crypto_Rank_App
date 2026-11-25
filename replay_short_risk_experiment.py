#!/usr/bin/env python3
"""Post-hoc short sizing trial using saved forecasts; no model refits or charts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from replay_core import DEFAULT_OUTPUT, digest, write_csv, write_json

MODELS = ("enet_all", "enet_market", "extra_all", "extra_market")
VARIANT = "legacy_clipped"
ARMS = ("original_equal", "matched_gross_equal", "volatility_reduced")
LOOKBACK = 28
MIN_RETURNS = 14


def trailing_risk(prices, decisions):
    """28 calendar days of log returns, with exact daily joins and no future data.

    At least 14 valid returns and a decision-date close are required. Missing
    calendar dates stay missing; adjacent observations cannot bridge a gap.
    """
    if prices.duplicated(["symbol", "date"]).any():
        raise ValueError("Daily price keys must be unique")
    decisions = pd.DatetimeIndex(sorted(set(decisions)))
    if not len(decisions):
        raise ValueError("No decision dates")
    rows = []
    for symbol, part in prices.groupby("symbol", sort=True):
        s = part.set_index("date").close.sort_index()
        calendar = pd.date_range(min(s.index.min(), decisions.min()-pd.Timedelta(days=LOOKBACK)),
                                 decisions.max(), freq="D")
        s = s.reindex(calendar).where(lambda x: np.isfinite(x) & x.gt(0))
        returns = np.log(s).diff()
        counts = returns.rolling(LOOKBACK, min_periods=1).count().fillna(0)
        volatility = returns.rolling(LOOKBACK, min_periods=MIN_RETURNS).std(ddof=1)
        for decision in decisions:
            n = int(counts.loc[decision])
            close_present = bool(pd.notna(s.loc[decision]))
            reason = ("missing_decision_close" if not close_present else
                      "insufficient_daily_returns" if n < MIN_RETURNS else "available")
            vol = float(volatility.loc[decision]) if reason == "available" else np.nan
            window = returns.loc[decision-pd.Timedelta(days=LOOKBACK-1):decision].dropna()
            rows.append({"symbol": symbol, "decision_date": decision,
                         "risk_window_start": decision-pd.Timedelta(days=LOOKBACK-1),
                         "risk_price_window_start": decision-pd.Timedelta(days=LOOKBACK),
                         "valid_daily_returns": n, "decision_close_present": close_present,
                         "volatility_28d": vol, "risk_status": reason,
                         "latest_risk_return_date": window.index.max() if len(window) else pd.NaT})
    return pd.DataFrame(rows)


def short_scales(volatility, reference):
    """Only reduce each original short. Unknown risk opens no short position."""
    vol = np.asarray(volatility, dtype=float)
    scale = np.zeros(len(vol), dtype=float)
    if not np.isfinite(reference) or reference < 0:
        return scale
    valid = np.isfinite(vol) & (vol >= 0)
    scale[valid & (vol == 0)] = 1.
    positive = valid & (vol > 0)
    scale[positive] = np.minimum(1., reference/vol[positive])
    return scale


def allocate_short_weights(volatility, reference, original_weights):
    weights = np.asarray(original_weights, dtype=float)
    if (weights >= 0).any() or not np.isfinite(weights).all():
        raise ValueError("Expected finite negative original short weights")
    reduced = weights * short_scales(volatility, reference)
    # Control keeps all the same Bottom 20 names, at identical total short gross.
    matched = np.full(len(weights), -np.abs(reduced).sum()/len(weights))
    return {"original_equal": weights, "matched_gross_equal": matched,
            "volatility_reduced": reduced}


def continuous_metrics(frame):
    frame = frame.sort_values("decision_date")
    if len(frame) > 1 and not frame.decision_date.diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("Cannot compound across missing weeks")
    returns = frame.weekly_return.to_numpy(float)
    if not np.isfinite(returns).all():
        raise ValueError("Cannot compound unknown returns")
    if (returns <= -1).any():
        raise ValueError("Account exhausted: linear continuation requires an execution model")
    nav = np.r_[1., np.cumprod(1+returns)]
    before = nav[:-1]
    return {"total_return": float(nav[-1]-1),
            "max_drawdown": float((nav/np.maximum.accumulate(nav)-1).min()),
            "linked_long_contribution": float((before*frame.long_contribution).sum()),
            "linked_short_contribution": float((before*frame.short_contribution).sum())}


def build(root=DEFAULT_OUTPUT):
    root = Path(root)
    portfolio = root / "portfolio_evaluation"
    meta = json.loads((portfolio / "portfolio_manifest.json").read_text())
    source = root / "target_experiment/runs" / meta["source_run_id"]
    out = root / "short_risk_experiment"
    inputs = [portfolio / "portfolio_holdings.csv", portfolio / "portfolio_weekly_returns.csv",
              portfolio / "portfolio_manifest.json", source / "forecast_actual_comparison.csv",
              root / "target_experiment/reviewed_prices.csv"]
    hashes = {str(p.relative_to(root)): digest(p) for p in inputs}
    dates = ["decision_date", "target_end"]
    holdings = pd.read_csv(inputs[0], float_precision="round_trip", parse_dates=dates)
    holdings = holdings[holdings.target_variant.eq(VARIANT) & holdings.strategy.eq("LS")
                        & holdings.model_name.isin(MODELS)].copy()
    weekly = pd.read_csv(inputs[1], float_precision="round_trip", parse_dates=dates)
    weekly = weekly[weekly.target_variant.eq(VARIANT) & weekly.strategy.eq("LS")
                    & weekly.model_name.isin(MODELS)].copy()
    forecasts = pd.read_csv(inputs[3], float_precision="round_trip", parse_dates=dates)
    forecasts = forecasts[forecasts.target_variant.eq(VARIANT) & forecasts.model_name.isin(MODELS)]
    prices = pd.read_csv(inputs[4], float_precision="round_trip", parse_dates=["date"])
    risk = trailing_risk(prices, holdings.decision_date.unique())
    universe = forecasts[forecasts.model_name.eq(MODELS[0])][["decision_date", "symbol"]]
    for model in MODELS[1:]:
        other = forecasts[forecasts.model_name.eq(model)][["decision_date", "symbol"]]
        if set(map(tuple, other.to_numpy())) != set(map(tuple, universe.to_numpy())):
            raise ValueError("Different eligible forecast universes")
    risk = universe.merge(risk, on=["decision_date", "symbol"], how="left", validate="one_to_one")
    risk["risk_status"] = risk.risk_status.fillna("no_daily_price_history")
    risk["valid_daily_returns"] = risk.valid_daily_returns.fillna(0).astype(int)
    risk["decision_close_present"] = risk.decision_close_present.fillna(False).astype(bool)
    risk["market_median_volatility"] = risk.groupby("decision_date").volatility_28d.transform("median")
    risk["universe_with_risk"] = risk.groupby("decision_date").volatility_28d.transform("count")
    risk = risk.sort_values(["decision_date", "symbol"])
    positions = holdings.merge(risk, on=["decision_date", "symbol"], how="left", validate="many_to_one")
    if positions.risk_status.isna().any():
        raise ValueError("A selected holding is absent from the forecast universe")
    positions = positions.merge(
        forecasts[["model_name", "decision_date", "symbol", "predicted_rank", "predicted_return"]],
        on=["model_name", "decision_date", "symbol"], validate="one_to_one")
    saved = weekly.set_index(["model_name", "decision_date"])
    holding_parts, week_rows = [], []
    for (model, decision), group in positions.groupby(["model_name", "decision_date"], sort=True):
        group = group.sort_values(["side", "symbol"])
        short = group.side.eq("short")
        if len(group) != 40 or int(short.sum()) != 20:
            raise ValueError("Expected the original Top/Bottom 20")
        allocations = allocate_short_weights(group.loc[short, "volatility_28d"],
                                             group.market_median_volatility.iloc[0],
                                             group.loc[short, "weight"])
        if (model, decision) in saved.index:
            original = saved.loc[(model, decision)]
            status = original.status
            common = bool(original.common_week)
            continuous = bool(original.longest_continuous_window)
        else:
            if group.target_end.iloc[0] <= pd.Timestamp(meta["data_end"]):
                raise ValueError("An ended original portfolio week is missing")
            status, common, continuous = "pending", False, False
        for arm in ARMS:
            part = group.rename(columns={"weight": "original_weight",
                                         "weighted_contribution": "original_contribution"}).copy()
            part["arm"] = arm
            part["weight"] = part.original_weight
            part.loc[short, "weight"] = allocations[arm]
            # Unknown outcomes stay unknown, including when risk weight is zero.
            part["weighted_contribution"] = part.weight * part.actual_return_raw
            holding_parts.append(part)
            complete = status == "evaluated"
            if complete and (not part.evaluation_status.eq("evaluated").all()
                             or not np.isfinite(part.actual_return_raw).all()):
                raise ValueError("A supposedly evaluated week contains an unknown actual return")
            long = float(part.loc[~short, "weighted_contribution"].sum()) if complete else np.nan
            short_pnl = float(part.loc[short, "weighted_contribution"].sum()) if complete else np.nan
            gross = float(-part.loc[short, "weight"].sum())
            week_rows.append({"target_variant": VARIANT, "model_name": model, "decision_date": decision,
                              "target_end": part.target_end.iloc[0], "arm": arm,
                              "weekly_return": long+short_pnl if complete else np.nan,
                              "long_contribution": long, "short_contribution": short_pnl,
                              "long_gross": float(part.loc[~short, "weight"].sum()), "short_gross": gross,
                              "total_gross": 1.+gross, "net_long_exposure": 1.-gross,
                              "status": status, "common_week": common,
                              "longest_continuous_window": continuous,
                              "shorts_missing_risk": int(part.loc[short, "volatility_28d"].isna().sum()),
                              "shorts_with_zero_weight": int(part.loc[short, "weight"].eq(0).sum())})
    trial_holdings = pd.concat(holding_parts, ignore_index=True)
    trial_weekly = pd.DataFrame(week_rows)
    summary_rows = []
    for scope, flag, continuous in [("common_observed_weeks", "common_week", False),
                                     ("continuous_common_window", "longest_continuous_window", True)]:
        for (model, arm), group in trial_weekly[trial_weekly[flag]].groupby(["model_name", "arm"]):
            row = {"scope": scope, "target_variant": VARIANT, "model_name": model, "arm": arm,
                   "weeks": len(group), "mean_weekly_return": float(group.weekly_return.mean()),
                   "weekly_std": float(group.weekly_return.std(ddof=1)),
                   "positive_week_ratio": float(group.weekly_return.gt(0).mean()),
                   "worst_weekly_return": float(group.weekly_return.min()),
                   "mean_long_contribution": float(group.long_contribution.mean()),
                   "mean_short_contribution": float(group.short_contribution.mean()),
                   "mean_short_gross": float(group.short_gross.mean()),
                   "min_short_gross": float(group.short_gross.min()),
                   "max_short_gross": float(group.short_gross.max()),
                   "mean_total_gross": float(group.total_gross.mean()),
                   "mean_net_long_exposure": float(group.net_long_exposure.mean()),
                   "short_position_observations_missing_risk": int(group.shorts_missing_risk.sum()),
                   "total_return": np.nan, "max_drawdown": np.nan,
                   "linked_long_contribution": np.nan, "linked_short_contribution": np.nan}
            if continuous:
                row.update(continuous_metrics(group))
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    keys = ["model_name", "decision_date", "target_end", "status", "common_week", "longest_continuous_window"]
    contrasts = trial_weekly.pivot(index=keys, columns="arm", values="weekly_return").reset_index()
    contrasts["equal_deleveraging_effect"] = contrasts.matched_gross_equal - contrasts.original_equal
    contrasts["allocation_effect_at_matched_gross"] = contrasts.volatility_reduced - contrasts.matched_gross_equal

    def max_error(a, b):
        difference = np.abs(np.asarray(a, float)-np.asarray(b, float))
        return float(np.nanmax(difference))

    original = trial_weekly[trial_weekly.arm.eq("original_equal")].merge(
        weekly[["model_name", "decision_date", "weekly_return", "status"]],
        on=["model_name", "decision_date"], suffixes=("", "_saved"), validate="one_to_one")
    evaluated = original.status.eq("evaluated")
    baseline_error = max_error(original.loc[evaluated, "weekly_return"],
                               original.loc[evaluated, "weekly_return_saved"])
    exposures = trial_weekly.pivot(index=["model_name", "decision_date"], columns="arm", values="short_gross")
    gross_error = max_error(exposures.matched_gross_equal, exposures.volatility_reduced)
    long_spread = trial_holdings[trial_holdings.side.eq("long")].groupby(
        ["model_name", "decision_date", "symbol"]).weight.agg(lambda s: s.max()-s.min()).max()
    scales = trial_holdings[trial_holdings.arm.eq("volatility_reduced") & trial_holdings.side.eq("short")]
    weight_excess = float((scales.weight.abs()-scales.original_weight.abs()).max())
    flat_weekly = trial_weekly[trial_weekly.status.eq("evaluated")]
    pnl_error = max_error(flat_weekly.weekly_return,
                          flat_weekly.long_contribution+flat_weekly.short_contribution)
    # With unchanged equal-weight shorts, B-A must be exactly overall deleveraging.
    pivot_pnl = flat_weekly.pivot(index=["model_name", "decision_date"], columns="arm", values="short_contribution")
    identity_error = max_error(pivot_pnl.matched_gross_equal,
                              pivot_pnl.original_equal*exposures.loc[pivot_pnl.index, "matched_gross_equal"])
    linked = summary[summary.scope.eq("continuous_common_window")]
    linked_error = max_error(linked.total_return, linked.linked_long_contribution+linked.linked_short_contribution)
    risk_dates = risk[risk.latest_risk_return_date.notna()]
    checks = {
        "baseline_weekly_returns_reproduced": baseline_error < 1e-12,
        "original_evaluation_mask_preserved": bool(original.status.eq(original.status_saved).all()),
        "all_arms_preserve_identical_evaluation_mask": bool(trial_weekly.groupby(
            ["model_name", "decision_date"]).status.nunique().eq(1).all()),
        "matched_control_has_same_short_gross": gross_error < 1e-12,
        "long_weights_identical_all_arms": bool(long_spread == 0),
        "no_short_weight_increased": weight_excess < 1e-12,
        "leg_contributions_reconcile": pnl_error < 1e-12,
        "equal_control_only_changes_gross": identity_error < 1e-12,
        "linked_leg_pnl_reconciles": linked_error < 1e-12,
        "risk_uses_no_future_return": bool(risk_dates.latest_risk_return_date.le(risk_dates.decision_date).all()),
        "risk_minimum_history_enforced": bool(risk.loc[risk.volatility_28d.notna(), "valid_daily_returns"].ge(MIN_RETURNS).all()),
        "all_arms_same_common_week_count": bool(summary[summary.scope.eq("common_observed_weeks")].weeks.eq(meta["common_evaluated_weeks"]).all()),
        "continuous_window_unchanged": bool(linked.weeks.eq(meta["longest_window_weeks"]).all()),
        "source_files_unchanged": all(digest(p) == hashes[str(p.relative_to(root))] for p in inputs),
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    manifest = {
        "status": "complete_exploratory_short_sizing_trial", "source_run_id": meta["source_run_id"],
        "target_variant": VARIANT, "models": MODELS, "model_retraining": False,
        "decision_start": str(holdings.decision_date.min().date()),
        "last_decision": str(holdings.decision_date.max().date()), "data_end": meta["data_end"],
        "common_observed_weeks": meta["common_evaluated_weeks"],
        "continuous_window_start": meta["longest_window_start"],
        "continuous_window_end": meta["longest_window_end"],
        "continuous_window_weeks": meta["longest_window_weeks"],
        "arms": {"original_equal": "Original 100% long and 100% short, equal 5% per name.",
                 "matched_gross_equal": "Same weekly short gross as volatility_reduced, equally across original Bottom 20.",
                 "volatility_reduced": "Original short weight times min(1, median universe volatility / asset volatility)."},
        "risk_rule": "28 calendar-day daily log-return sample standard deviation, min 14 exact consecutive-day return observations, decision-date close required. Median uses the eligible forecast universe without outcome filtering. Unknown risk means zero short weight; original selected names remain in the audit. Never increase or renormalize short weights.",
        "timing": "Decision day UTC daily close is available at next midnight, as assumed in the original close-to-close replay. No executable fill is validated.",
        "missing_policy": "Preserve original portfolio evaluation mask even for a risk-zero holding. No missing actual return filled or dropped. No compounding across noncontinuous observed weeks.",
        "limitations": ["Post-hoc design after inspecting losses; not independent out-of-sample evidence.",
                        "Reducing shorts introduces net long exposure and benefits from rising markets; the exposure-matched control is required.",
                        "Unknown-risk exclusions are included in the allocation effect; do not attribute all of it to volatility estimates.",
                        "Current source history and retrospective quality filters are not point-in-time data vintages.",
                        "No verified historical shortability, execution, fees, funding, borrow costs, margin calls or liquidation model.",
                        "Full-period continuous portfolio performance still has unresolved price gaps. No daily rebalancing or new visualization in this trial."],
        "conceptual_reference": "https://www.nber.org/papers/w22208 (motivation only; this is not a replication or validation for crypto)",
        "source_hashes": hashes, "checks": checks,
        "max_errors": {"baseline_return": baseline_error, "matched_gross": gross_error,
                       "leg_pnl": pnl_error, "deleveraging_identity": identity_error, "linked_attribution": linked_error},
    }
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "decision_risk_inputs.csv", risk)
    write_csv(out / "portfolio_holdings.csv", trial_holdings)
    write_csv(out / "portfolio_weekly_returns.csv", trial_weekly)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "weekly_contrasts.csv", contrasts)
    write_json(out / "experiment_manifest.json", manifest)
    print(summary[["scope", "model_name", "arm", "weeks", "mean_short_gross", "mean_weekly_return", "total_return", "max_drawdown", "short_position_observations_missing_risk"]].to_string(index=False))
    print(f"Checks: {sum(checks.values())}/{len(checks)}. Outputs: {out}")
    return manifest


if __name__ == "__main__":
    build()
