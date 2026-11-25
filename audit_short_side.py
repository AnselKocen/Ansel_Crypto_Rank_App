#!/usr/bin/env python3
"""Explain saved long/short losses without changing portfolios or filling gaps."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from replay_core import DEFAULT_OUTPUT, digest, write_csv, write_json


def linked_leg_attribution(legs):
    """Additive final-PnL contributions on one continuous account equity path."""
    if len(legs) > 1 and not legs.index.to_series().diff().dropna().eq(pd.Timedelta(days=7)).all():
        raise ValueError("Cannot link contributions across missing calendar weeks")
    if not np.isfinite(legs[["long", "short"]]).all().all():
        raise ValueError("Missing leg returns cannot be zero-filled")
    total = legs.long + legs.short
    before = pd.Series(np.r_[1., np.cumprod(1+total.to_numpy())[:-1]], index=legs.index)
    return before, float((before*legs.long).sum()), float((before*legs.short).sum()), float(np.prod(1+total)-1)


def build():
    portfolio = DEFAULT_OUTPUT / "portfolio_evaluation"
    out = DEFAULT_OUTPUT / "short_side_audit"
    out.mkdir(exist_ok=True)
    manifest = json.loads((portfolio / "portfolio_manifest.json").read_text())
    source = DEFAULT_OUTPUT / "target_experiment/runs" / manifest["source_run_id"]
    sources = {str(p.relative_to(DEFAULT_OUTPUT)): digest(p) for p in [
        portfolio / "portfolio_holdings.csv", portfolio / "portfolio_weekly_returns.csv",
        source / "forecast_actual_comparison.csv"]}
    holdings = pd.read_csv(portfolio / "portfolio_holdings.csv", float_precision="round_trip",
                           parse_dates=["decision_date", "target_end"])
    weekly = pd.read_csv(portfolio / "portfolio_weekly_returns.csv", float_precision="round_trip",
                        parse_dates=["decision_date", "target_end"])
    comparison = pd.read_csv(source / "forecast_actual_comparison.csv", float_precision="round_trip",
                            parse_dates=["decision_date", "target_end"])
    short_holdings = holdings[holdings.strategy.eq("LS")].copy()
    quotes = comparison[comparison.model_name.eq("enet_all")].drop_duplicates(["target_variant", "decision_date", "symbol"])
    audited = short_holdings.merge(quotes[["target_variant", "decision_date", "symbol", "reference_close", "target_close"]],
                                   on=["target_variant", "decision_date", "symbol"], validate="many_to_one")
    audited["independent_contribution"] = audited.weight * (audited.target_close/audited.reference_close-1)
    observed = audited[audited.evaluation_status.eq("evaluated")]
    formula_error = float((observed.independent_contribution-observed.weighted_contribution).abs().max())
    valid_weekly = weekly[weekly.strategy.eq("LS")]
    legs_rows = []
    for key, g in short_holdings.groupby(["target_variant", "model_name", "decision_date"], sort=True):
        variant, model, decision = key
        if g.target_end.iloc[0] > pd.Timestamp(manifest["data_end"]):
            continue
        row = {"target_variant": variant, "model_name": model, "decision_date": decision,
               "target_end": g.target_end.iloc[0]}
        for side in ["long", "short"]:
            part = g[g.side.eq(side)]
            complete = part.evaluation_status.eq("evaluated").all()
            row[side] = float(part.weighted_contribution.sum()) if complete else np.nan
            row[side+"_missing_symbols"] = ",".join(sorted(part.loc[part.evaluation_status.ne("evaluated"), "symbol"].unique()))
        legs_rows.append(row)
    legs = pd.DataFrame(legs_rows)
    joined = legs.merge(valid_weekly[["target_variant", "model_name", "decision_date", "weekly_return", "status", "common_week", "longest_continuous_window"]],
                        on=["target_variant", "model_name", "decision_date"], validate="one_to_one")
    joined["bottom_basket_return"] = -joined.short
    joined["leg_sum"] = joined.long + joined.short
    available = joined[joined.status.eq("evaluated")]
    leg_error = float((available.leg_sum-available.weekly_return).abs().max())
    write_csv(out / "leg_weekly_returns.csv", joined)
    summary_rows, coin_rows = [], []
    for scope, flag, continuous in [("continuous_27_weeks", "longest_continuous_window", True),
                                     ("all_59_common_observed_weeks", "common_week", False),
                                     ("all_individually_evaluable_weeks", None, False)]:
        selected = joined[joined[flag]] if flag else joined[joined.status.eq("evaluated")]
        for (variant, model), group in selected.groupby(["target_variant", "model_name"]):
            frame = group.set_index("decision_date").sort_index()
            baseline = weekly[weekly.target_variant.eq(variant) & weekly.model_name.eq("universe")].set_index("decision_date").weekly_return.reindex(frame.index)
            both = baseline.notna()
            long_beta = frame.loc[both, "long"].cov(baseline[both])/baseline[both].var()
            bottom_beta = frame.loc[both, "bottom_basket_return"].cov(baseline[both])/baseline[both].var()
            r = {"scope": scope, "target_variant": variant, "model_name": model, "weeks": len(frame),
                 "mean_long_contribution": float(frame.long.mean()), "mean_short_contribution": float(frame.short.mean()),
                 "mean_ls_return": float(frame.weekly_return.mean()),
                 "short_losing_weeks": int(frame.short.lt(0).sum()), "ls_losing_weeks": int(frame.weekly_return.lt(0).sum()),
                 "long_beta_sample": float(long_beta), "bottom_beta_sample": float(bottom_beta),
                 "net_beta_sample": float(long_beta-bottom_beta),
                 "mean_short_when_market_rises": float(frame.loc[baseline.gt(0), "short"].mean()),
                 "mean_short_when_market_falls": float(frame.loc[baseline.lt(0), "short"].mean()),
                 "continuous_total_return": None, "linked_long_contribution": None, "linked_short_contribution": None}
            if continuous:
                before, long_pnl, short_pnl, total = linked_leg_attribution(frame)
                r.update(continuous_total_return=total, linked_long_contribution=long_pnl, linked_short_contribution=short_pnl)
                positions = short_holdings[short_holdings.target_variant.eq(variant) & short_holdings.model_name.eq(model) & short_holdings.decision_date.isin(frame.index)].copy()
                positions["linked_contribution"] = positions.weighted_contribution * positions.decision_date.map(before)
                for (side, symbol), f in positions.groupby(["side", "symbol"]):
                    coin_rows.append({"target_variant": variant, "model_name": model, "side": side, "symbol": symbol,
                                      "held_weeks": len(f), "linked_pnl_per_initial_dollar": float(f.linked_contribution.sum()),
                                      "unlinked_weekly_contribution_sum": float(f.weighted_contribution.sum())})
            summary_rows.append(r)
    summary = pd.DataFrame(summary_rows)
    coin = pd.DataFrame(coin_rows)
    write_csv(out / "leg_summary.csv", summary)
    write_csv(out / "coin_attribution_continuous_27_weeks.csv", coin)
    write_csv(out / "unsettled_short_positions.csv", audited[audited.side.eq("short") & ~audited.evaluation_status.isin(["evaluated", "pending"])])

    checks = []
    prices = pd.read_csv(DEFAULT_OUTPUT / "target_experiment/reviewed_prices.csv", float_precision="round_trip", parse_dates=["date"]).set_index(["symbol", "date"])
    for file in (out / "source_checks").glob("coinbase_*.json"):
        symbol = file.name.split("_")[1]
        rows = json.loads(file.read_text())
        independent = {pd.to_datetime(x[0], unit="s"): float(x[4]) for x in rows}
        for date, close in independent.items():
            if (symbol, date) not in prices.index:
                continue
            replay_close = float(prices.loc[(symbol, date), "close"])
            checks.append({"symbol": symbol, "date": date, "replay_close": replay_close, "independent_close": close,
                           "relative_close_difference": replay_close/close-1, "source": "Coinbase Exchange USD daily candle",
                           "evidence": str(file.relative_to(DEFAULT_OUTPUT))})
    # MSQ's key event was already independently retrieved from Gate spot API.
    gate_file = DEFAULT_OUTPUT / "target_experiment/source_checks/msq_gate.json"
    for row in json.loads(gate_file.read_text()):
        date, close = pd.to_datetime(int(row[0]), unit="s"), float(row[2])
        if ("MSQ", date) in prices.index:
            replay_close = float(prices.loc[("MSQ", date), "close"])
            checks.append({"symbol": "MSQ", "date": date, "replay_close": replay_close, "independent_close": close,
                           "relative_close_difference": replay_close/close-1, "source": "Gate MSQ-USDT; cross-currency venue check, not USD replacement",
                           "evidence": str(gate_file.relative_to(DEFAULT_OUTPUT))})
    cross = pd.DataFrame(checks)
    anchor_dates = {"USELESS": ["2026-08-26", "2026-09-02", "2026-09-09"],
                    "ENA": ["2026-09-16", "2026-09-23"], "NEAR": ["2026-09-16", "2026-09-23"],
                    "MINA": ["2026-09-16", "2026-09-23"], "SAND": ["2026-09-30", "2026-10-07"],
                    "MSQ": ["2026-05-06", "2026-05-13"]}
    cross["key_weekly_endpoint"] = [str(pd.Timestamp(d).date()) in anchor_dates.get(s, [])
                                     for s, d in zip(cross.symbol, cross.date)]
    write_csv(out / "independent_price_checks.csv", cross)
    sample = cross[cross.symbol.ne("MSQ")]
    endpoints = sample[sample.key_weekly_endpoint]
    verified = {
        "holding_formula_matches_independent_close_calculation": formula_error < 1e-12,
        "long_plus_short_matches_saved_ls_returns": leg_error < 1e-12,
        "short_weights_negative": bool(short_holdings.loc[short_holdings.side.eq("short"), "weight"].lt(0).all()),
        "missing_ls_returns_not_zero_filled": bool(joined.loc[joined.status.ne("evaluated"), "weekly_return"].isna().all()),
        "linked_contributions_add_to_continuous_total": bool(np.allclose(
            summary.loc[summary.scope.eq("continuous_27_weeks"), "linked_long_contribution"] + summary.loc[summary.scope.eq("continuous_27_weeks"), "linked_short_contribution"],
            summary.loc[summary.scope.eq("continuous_27_weeks"), "continuous_total_return"], atol=1e-12)),
        "no_fake_compounding_across_noncontinuous_scopes": bool(summary.loc[summary.scope.ne("continuous_27_weeks"), "continuous_total_return"].isna().all()),
        "five_coinbase_usd_series_checked": sample.symbol.nunique() == 5,
        "all_eleven_key_usd_weekly_endpoints_checked": len(endpoints) == 11,
        "key_usd_weekly_endpoints_within_one_percent": bool(endpoints.relative_close_difference.abs().lt(.01).all()),
        "source_predictions_and_portfolios_unchanged": all(digest(DEFAULT_OUTPUT / file) == value for file, value in sources.items())}
    findings = {"source_run_id": manifest["source_run_id"], "source_hashes": sources,
                "formula_max_absolute_difference": formula_error, "leg_sum_max_absolute_difference": leg_error,
                "checks": verified, "passed": all(verified.values()),
                "price_validation_scope": "Numerical pass is limited to the 11 selected USD weekly endpoints, not every daily quote.",
                "additional_daily_price_discrepancies_above_one_percent": json.loads(cross[cross.relative_close_difference.abs().gt(.01)].to_json(orient="records", date_format="iso")),
                "scopes": {"continuous_27_weeks": "2026-04-01 to 2026-10-07; selected by availability, not performance",
                           "all_59_common_observed_weeks": "2025-07-30 to 2026-11-11 with gaps; averages only, no continuous cumulative return",
                           "all_individually_evaluable_weeks": "Each model's own complete weeks; different dates, not a fair cross-model comparison"},
                "delisting_findings": [{"symbol": "VENOM", "venue": "KuCoin spot", "effective_utc": "2026-10-14 08:00:00",
                                        "source": "https://www.kucoin.com/announcement/th-kucoin-will-delist-venom-and-NFP-on-20261014",
                                        "affected_decision": "2026-10-07", "treatment": "quarantined/missing, not booked as a loss",
                                        "outside_continuous_27_week_window": True}],
                "unresolved": ["No complete historical venue-specific borrowing/perpetual eligibility universe; spot prices do not prove short executability.",
                               "Funding, borrow costs, liquidation, recalls, slippage and forced delisting settlements are not simulated.",
                               "Missing-price weeks may hide profitable or losing shorts. Their absence can bias results; it is not evidence that missing assets were delisted.",
                               "The independent source audit is a targeted sample, not certification of every quote, especially illiquid instruments such as MWC.",
                               "Full-period portfolio data gaps remain; no visualization changes or model/portfolio rule changes were made."]}
    write_json(out / "audit_manifest.json", findings)
    if not findings["passed"]:
        raise RuntimeError(f"Audit failed: {verified}")
    print(json.dumps({"passed": findings["passed"], "checks": verified, "price_pairs_checked": len(cross)}, indent=2))
    print(summary[summary.target_variant.eq("legacy_clipped") & summary.model_name.isin(["enet_all", "enet_market", "extra_all", "extra_market"])].to_string(index=False))


if __name__ == "__main__":
    build()
