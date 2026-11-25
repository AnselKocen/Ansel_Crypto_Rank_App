#!/usr/bin/env python3
"""Independently audit saved ranking portfolios and inventory daily inputs.

Reads archived results; never retrains, repairs quotes, or changes strategy files.
The legacy probe executes the original selection/accounting code with synthetic
prices, removing only its plotting/writing tail. No project module is imported.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "artifacts/historical_replay"
OUTPUT = ARCHIVE / "simulation_chain_audit"
KEYS = ["target_variant", "decision_date", "model_name", "strategy"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_probes():
    path = ROOT / "crypto_pipeline_stage3.py"
    tree = ast.parse(path.read_text())
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "rolling_weekly_prediction")
    loop = next(n for n in ast.walk(train) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "i")
    fits = [n.lineno for n in ast.walk(loop) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "fit"]
    full_fit = [n.lineno for n in ast.walk(train) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "fit"]
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "rolling_backtest_with_baseline_combined")
    # Preserve the actual calculation and return before any CSV/plot side effects.
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Expr)
               and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
               and n.value.func.attr == "to_csv")
    fn.body = fn.body[:end] + [ast.Return(value=ast.Name(id="df_all", ctx=ast.Load()))]
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    scope = {"pd": pd, "np": np, "warnings": warnings, "Path": Path}
    exec(compile(module, str(path), "exec"), scope)
    dates = pd.date_range("2025-01-01", periods=4, freq="W-WED")
    realized = [.01, .02, .30, -.40]
    history = pd.DataFrame([
        {"date": d, "symbol": f"C{i:02}", "return": realized[j]}
        for j, d in enumerate(dates) for i in range(40)])
    pred = pd.DataFrame([
        {"date": d, "symbol": f"C{i:02}", "y_pred": 40-i,
         "y_true": realized[j+1] if j+1 < len(dates) else np.nan}
        for j, d in enumerate(dates) for i in range(40)])
    result = scope[fn.name](pred, pred, "all", 20, Path("."), Path("."), history)
    wrong = float(result[result.strategy.eq("enet_EW")].iloc[0]["return"])
    baseline = float(result[result.strategy.eq("baseline_EW")].iloc[0]["return"])

    path = ROOT / "web/services/paper_trading_service.py"
    paper_tree = ast.parse(path.read_text())
    paper = next(n for n in paper_tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_compute_weekly_strategy_returns")
    paper.returns = None
    for arg in paper.args.args:
        arg.annotation = None
    scope = {"pd": pd, "np": np, "TOP_N": 20}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[paper], type_ignores=[])),
                 str(path), "exec"), scope)
    # Current top asset has no next-week quote. The original code excludes it
    # BEFORE choosing positions, which uses future outcome availability.
    sparse = pd.DataFrame([
        {"date": d, "symbol": f"C{i:02}", "return": .5-i*.001 if j == 0 else .01}
        for j, d in enumerate(dates) for i in range(41)
        if not (i == 0 and j > 0)])
    proxy = scope[paper.name](sparse)
    return {
        "rolling_training": {"fit_call_lines": full_fit, "fit_calls_inside_rolling_loop": fits,
                             "confirmed_full_sample_fit_reused": bool(full_fit and not fits)},
        "one_week_offset": {"decision": str(dates[0].date()),
            "correct_holding_return": realized[1], "legacy_reported_return": wrong,
            "legacy_baseline_return": baseline,
            "confirmed": bool(np.isclose(wrong, .30) and np.isclose(baseline, .30))},
        "paper_trading_proxy": {"ranks_realized_return_instead_of_model_forecasts":
            any(isinstance(n, ast.Constant) and n.value == "return" for n in ast.walk(paper)),
            "first_eligible_count_after_future_missing_quote_exclusion": int(proxy.n_coins.iloc[0]),
            "assets_known_at_decision": 41,
            "confirmed_future_availability_filter": int(proxy.n_coins.iloc[0]) == 40},
    }


def run():
    portfolio = ARCHIVE / "portfolio_evaluation"
    exp = ARCHIVE / "target_experiment"
    run_path = exp / json.loads((exp / "latest_run.json").read_text())["path"]
    protected = [run_path / x for x in ["forecast_history.csv", "forecast_actual_comparison.csv"]]
    protected += [portfolio / x for x in ["portfolio_holdings.csv", "portfolio_weekly_returns.csv",
                                          "portfolio_summary.csv", "portfolio_nav_segments.csv"]]
    protected += [exp / "reviewed_prices.csv", ARCHIVE / "raw_prices.csv",
                  ROOT / "crypto_pipeline_stage3.py", ROOT / "crypto_project_pipeline.py",
                  ROOT / "web/services/paper_trading_service.py", ROOT / "replay_core.py"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    dates = ["decision_date", "target_end"]
    comparison = pd.read_csv(run_path / "forecast_actual_comparison.csv",
                             float_precision="round_trip", parse_dates=dates)
    forecast = pd.read_csv(run_path / "forecast_history.csv", float_precision="round_trip",
                          parse_dates=dates + ["train_start", "train_end"])
    prices = pd.read_csv(exp / "reviewed_prices.csv", float_precision="round_trip", parse_dates=["date"])
    holdings = pd.read_csv(portfolio / "portfolio_holdings.csv", float_precision="round_trip", parse_dates=dates)
    weekly = pd.read_csv(portfolio / "portfolio_weekly_returns.csv", float_precision="round_trip", parse_dates=dates)
    summary = pd.read_csv(portfolio / "portfolio_summary.csv", float_precision="round_trip")
    manifest = json.loads((portfolio / "portfolio_manifest.json").read_text())
    checks, errors = {}, {}
    checks["unique_daily_instrument_dates"] = not prices.duplicated(["symbol", "date"]).any()
    checks["unique_forecast_keys"] = not forecast.duplicated(KEYS[:3] + ["symbol"]).any()
    checks["all_predictions_finite"] = np.isfinite(forecast.predicted_return).all()
    checks["all_targets_exactly_seven_calendar_days"] = (forecast.target_end-forecast.decision_date).eq(pd.Timedelta(days=7)).all()
    checks["training_features_end_before_decision"] = forecast.train_end.lt(forecast.decision_date).all()
    checks["latest_training_label_matures_at_decision_close"] = (forecast.train_end+pd.Timedelta(days=7)).eq(forecast.decision_date).all()

    a = prices[["symbol", "date", "close", "source"]].rename(columns={"date": "decision_date", "close": "entry", "source": "entry_source"})
    b = prices[["symbol", "date", "close", "source"]].rename(columns={"date": "target_end", "close": "exit", "source": "exit_source"})
    quotes = comparison.merge(a, on=["symbol", "decision_date"], how="left", validate="many_to_one").merge(
        b, on=["symbol", "target_end"], how="left", validate="many_to_one")
    quotes["recomputed_return"] = quotes.exit/quotes.entry-1
    known = quotes.evaluation_status.eq("evaluated")
    errors["price_pair_return"] = float((quotes.loc[known, "recomputed_return"]-quotes.loc[known, "actual_return_raw"]).abs().max())
    checks["returns_independently_joined_from_daily_quotes"] = errors["price_pair_return"] < 1e-12
    checks["evaluated_quotes_same_source"] = quotes.loc[known, "entry_source"].eq(quotes.loc[known, "exit_source"]).all()
    checks["pending_returns_unknown"] = comparison.loc[comparison.evaluation_status.eq("pending"), "actual_return_raw"].isna().all()

    expected = []
    for (variant, date), g in comparison.groupby(["target_variant", "decision_date"]):
        frames = {m: f.sort_values("predicted_rank") for m, f in g.groupby("model_name")}
        for model, f in frames.items():
            checks.setdefault("saved_ranks_match_prediction_order", True)
            checks["saved_ranks_match_prediction_order"] &= bool(f.predicted_return.diff().dropna().le(0).all())
            for strategy in ["EW", "LS"]:
                positions = [(s, "long", .05) for s in f.head(20).symbol]
                if strategy == "LS":
                    positions += [(s, "short", -.05) for s in f.tail(20).symbol]
                expected += [(variant, date, model, strategy, s, side, w) for s, side, w in positions]
        for tag in ["all", "market"]:
            for side, sign in [("long", 1), ("short", -1)]:
                sets = [set((frames[m].head(20) if side == "long" else frames[m].tail(20)).symbol)
                        for m in ["enet_"+tag, "extra_"+tag]]
                raw = {s: .65 if s in sets[0] & sets[1] else .35 for s in sets[0] | sets[1]}
                total = sum(raw.values())
                expected += [(variant, date, "fusion_"+tag, "LS", s, side, sign*w/total) for s, w in raw.items()]
        universe = set(next(iter(frames.values())).symbol)
        checks.setdefault("same_forecast_universe_all_models", True)
        checks["same_forecast_universe_all_models"] &= all(set(f.symbol) == universe for f in frames.values())
        expected += [(variant, date, "universe", "EW", s, "long", 1/len(universe)) for s in universe]
    pos_keys = KEYS + ["symbol", "side"]
    expected = pd.DataFrame(expected, columns=pos_keys+["expected_weight"])
    joined = holdings.merge(expected, on=pos_keys, how="outer", validate="one_to_one", indicator=True)
    errors["position_weight"] = float((joined.weight-joined.expected_weight).abs().max())
    checks["all_selected_assets_sides_and_weights_match_forecasts"] = joined._merge.eq("both").all() and errors["position_weight"] < 1e-12
    # Price return map is unique across models and target variants.
    price_map = quotes.drop_duplicates(["decision_date", "symbol"])[["decision_date", "symbol", "recomputed_return"]]
    joined = joined.merge(price_map, on=["decision_date", "symbol"], how="left", validate="many_to_one")
    joined["expected_contribution"] = joined.weight*joined.recomputed_return
    valid_h = joined.evaluation_status.eq("evaluated")
    errors["holding_contribution"] = float((joined.loc[valid_h, "expected_contribution"]-joined.loc[valid_h, "weighted_contribution"]).abs().max())
    checks["all_holding_pnl_matches_price_pair_times_signed_weight"] = errors["holding_contribution"] < 1e-12
    rebuilt = joined.groupby(KEYS).agg(recomputed_weekly_return=("expected_contribution", "sum"),
                                      all_known=("evaluation_status", lambda s: s.eq("evaluated").all()))
    match = weekly.merge(rebuilt, on=KEYS, validate="one_to_one")
    known_w = match.status.eq("evaluated")
    errors["weekly_portfolio_return"] = float((match.loc[known_w, "weekly_return"]-match.loc[known_w, "recomputed_weekly_return"]).abs().max())
    checks["all_finished_portfolios_rebuilt"] = len(match) == len(weekly)
    checks["missing_positions_never_become_evaluated_portfolios"] = match.all_known.eq(known_w).all()
    checks["incomplete_portfolio_returns_remain_unknown"] = match.loc[~known_w, "weekly_return"].isna().all()
    checks["all_weekly_portfolio_pnl_matches_independent_rebuild"] = errors["weekly_portfolio_return"] < 1e-12
    # Recompute the shared mask and select segments without any return criterion.
    complete = weekly.groupby("decision_date").status.apply(lambda s: s.eq("evaluated").all())
    common_dates = complete[complete].index.sort_values()
    segments = []
    for d in common_dates:
        if not segments or d-segments[-1][-1] != pd.Timedelta(days=7):
            segments.append([])
        segments[-1].append(d)
    longest = max(segments, key=len)
    checks["common_week_flags_recomputed"] = weekly.common_week.eq(weekly.decision_date.isin(common_dates)).all()
    checks["continuous_window_flags_recomputed"] = weekly.longest_continuous_window.eq(weekly.decision_date.isin(longest)).all()
    summary_diffs = []
    for row in summary[summary.scope.eq("longest_continuous_common_window")].itertuples():
        f = match[match.target_variant.eq(row.target_variant) & match.model_name.eq(row.model_name) &
                  match.strategy.eq(row.strategy) & match.decision_date.isin(longest)].sort_values("decision_date")
        nav = np.r_[1., np.cumprod(1+f.recomputed_weekly_return.to_numpy())]
        summary_diffs += [abs(nav[-1]-1-row.total_return), abs((nav/np.maximum.accumulate(nav)-1).min()-row.max_drawdown)]
    errors["continuous_return_and_drawdown"] = float(max(summary_diffs))
    checks["continuous_compounding_and_initial_capital_drawdown_match"] = errors["continuous_return_and_drawdown"] < 1e-12

    # Diagnose the CURRENT replay, independently of the old webpage simulation.
    diagnostics, attribution = [], []
    for scope_name, scope_dates in [("all_common_observed_weeks", common_dates),
                                    ("continuous_common_window", longest)]:
        selected_quotes = quotes[quotes.target_variant.eq("legacy_clipped") &
                                 quotes.decision_date.isin(scope_dates) & known]
        for model, g in selected_quotes.groupby("model_name"):
            ic = g.groupby("decision_date").apply(
                lambda x: x.predicted_return.corr(x.recomputed_return, method="spearman"),
                include_groups=False)
            top = g[g.predicted_rank.le(20)]
            bottom = g[g.is_bottom20]
            positions = joined[joined.target_variant.eq("legacy_clipped") & joined.model_name.eq(model) &
                               joined.strategy.eq("LS") & joined.decision_date.isin(scope_dates)].copy()
            leg = positions.groupby(["decision_date", "side"]).expected_contribution.sum().unstack()
            ls = leg.sum(axis=1)
            row = {"scope": scope_name, "model_name": model, "weeks": len(ic),
                   "mean_weekly_rank_ic": float(ic.mean()), "rank_ic_positive_week_ratio": float(ic.gt(0).mean()),
                   "mean_top20_asset_return": float(top.recomputed_return.mean()),
                   "mean_bottom20_asset_return": float(bottom.recomputed_return.mean()),
                   "mean_ls_return": float(ls.mean()), "short_losing_weeks": int(leg.short.lt(0).sum()),
                   "predicted_direction_accuracy_descriptive": float((np.sign(g.predicted_return) == np.sign(g.recomputed_return)).mean())}
            if scope_name == "continuous_common_window":
                before = (1+ls).cumprod().shift(1, fill_value=1.)
                row["linked_long_pnl_per_initial_dollar"] = float((before*leg.long).sum())
                row["linked_short_pnl_per_initial_dollar"] = float((before*leg.short).sum())
                row["continuous_total_return"] = float((1+ls).prod()-1)
                positions["linked_pnl_per_initial_dollar"] = positions.expected_contribution*positions.decision_date.map(before)
                attrib = positions.groupby(["model_name", "side", "symbol"], as_index=False).linked_pnl_per_initial_dollar.sum()
                attribution.append(attrib)
                checks.setdefault("linked_long_and_short_pnl_sum_to_compounded_account", True)
                checks["linked_long_and_short_pnl_sum_to_compounded_account"] &= bool(np.isclose(
                    row["linked_long_pnl_per_initial_dollar"]+row["linked_short_pnl_per_initial_dollar"],
                    row["continuous_total_return"], atol=1e-12))
            diagnostics.append(row)

    # Inventory calendar coverage without inventing daily values from weekly data.
    coverage = prices.groupby("date").symbol.nunique().rename("assets_with_daily_quote").to_frame()
    news = pd.read_csv(ARCHIVE / "news_articles.csv", usecols=["published_at", "source"])
    news["date"] = pd.to_datetime(news.published_at, utc=True).dt.tz_localize(None).dt.normalize()
    ncounts = news.groupby(["date", "source"]).size().unstack(fill_value=0).add_prefix("news_")
    coverage = coverage.join(ncounts).fillna(0).astype(int)
    earliest = prices.date.min()
    start = pd.Timestamp(manifest["decision_start"])
    # For a close(t)->close(t+1) RESEARCH label: first usable feature is
    # earliest+warmup, and the last training label must mature by decision t.
    budgets = [{"feature_warmup_days": warm, "labeled_training_dates": train,
                "earliest_theoretical_close_signal": str((earliest+pd.Timedelta(days=warm+train)).date()),
                "daily_quotes_needed_to_start_on_original_date": str((start-pd.Timedelta(days=warm+train)).date())}
               for warm, train in [(90,365), (90,180), (28,90)]]
    probes = legacy_probes()
    checks["legacy_bug_probe_reproduced_offset"] = probes["one_week_offset"]["confirmed"]
    checks["legacy_bug_probe_confirms_no_rolling_refit"] = probes["rolling_training"]["confirmed_full_sample_fit_reused"]
    checks["legacy_proxy_future_availability_probe_reproduced"] = probes["paper_trading_proxy"]["confirmed_future_availability_filter"]
    checks["all_protected_sources_unchanged"] = all(sha256(ROOT/p) == h for p, h in hashes.items())
    result = {
        "status": "arithmetic_audited_execution_not_certified", "source_run": run_path.name,
        "passed": bool(all(checks.values())), "checks": {k: bool(v) for k,v in checks.items()},
        "max_absolute_errors": errors, "legacy_probes": probes,
        "rows_checked": {"forecasts": len(forecast), "evaluated_price_pairs": int(known.sum()),
                         "holdings": len(holdings), "finished_portfolios": len(weekly)},
        "coverage": {"finished_weeks": int(weekly.decision_date.nunique()), "common_weeks": len(common_dates),
            "missing_common_weeks": int((~complete).sum()), "continuous_segments": len(segments),
            "longest_start": str(longest[0].date()), "longest_target_end": str((longest[-1]+pd.Timedelta(days=7)).date()),
            "longest_weeks": len(longest), "reviewed_daily_price_start": str(earliest.date()),
            "daily_price_end": str(prices.date.max().date()), "daily_price_rows": len(prices),
            "daily_price_symbols": int(prices.symbol.nunique()), "daily_price_dates": int(prices.date.nunique()),
            "news_start": str(news.date.min().date()), "news_end": str(news.date.max().date())},
        "daily_training_budgets": budgets,
        "unresolved": [
            "Current portfolio ranks are relative: top 20 are bought even if every forecast is negative.",
            "Decision-day close and full-day news are consumed, then entry is assumed at that same close; no post-signal executable fill.",
            "Gross PnL excludes turnover costs, spread, slippage, funding, borrow availability, margin and liquidation.",
            "Historical daily paths are not simulated for short margin risk; weekly terminal prices do not establish survival.",
            "Frozen pre-replay weekly features cannot establish point-in-time preprocessing of their original raw history.",
            "The fixed start-date universe and retrospective quality exclusions are not a historical investable-universe reconstruction.",
            "Missing outcomes remove whole comparison weeks, so observed-window statistics can have selection bias.",
            "Yahoo estimated base volume differs from original source native volume; news publisher mix changes over time.",
            "Existing 2025-2026 outcomes have already guided experiments and cannot now be called an untouched holdout.",
            "Full original daily training timeline requires additional daily prices/news; weekly history is not a substitute.",
        ], "protected_source_hashes": hashes,
    }
    OUTPUT.mkdir(exist_ok=True)
    coverage.rename_axis("date").to_csv(OUTPUT / "daily_input_coverage.csv")
    pd.DataFrame(diagnostics).to_csv(OUTPUT / "current_replay_diagnostics.csv", index=False)
    pd.concat(attribution, ignore_index=True).to_csv(OUTPUT / "current_replay_coin_attribution.csv", index=False)
    losses = joined[joined.target_variant.eq("legacy_clipped") & joined.strategy.eq("LS") &
                    joined.model_name.ne("universe") & ~joined.model_name.str.startswith("fusion") &
                    joined.decision_date.isin(longest)]
    losses.nsmallest(30, "expected_contribution")[pos_keys+["target_end", "recomputed_return", "weight", "expected_contribution"]].to_csv(
        OUTPUT / "largest_position_losses.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: result[k] for k in ["status", "passed", "rows_checked", "coverage", "max_absolute_errors"]}, indent=2))
    if not result["passed"]:
        raise SystemExit("Audit failed; see audit.json")


if __name__ == "__main__":
    run()
