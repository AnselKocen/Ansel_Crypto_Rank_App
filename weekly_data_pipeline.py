#!/usr/bin/env python3
"""Run the original manual workflows on a schedule and publish their outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd

from web.services.weekly_updates import WeeklyUpdates, PIPELINE_REVISION, DATA_FILES

ROOT = Path(__file__).resolve().parent
MODELS = ("enet_all", "enet_market", "extra_all", "extra_market")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def seed_history_path(service):
    active = service.published()
    path = active["path"] / "df_merged_history.csv" if active else ROOT / "df_merged_history.csv"
    if not path.is_file():
        raise ValueError("没有可供旧流程增量更新的历史数据")
    return path


def run_legacy_workflows(directory, period, history_path, *, service=None):
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts/historical_replay/mpl_config"))
    os.environ.setdefault("NLTK_DATA", str(ROOT / "artifacts/historical_replay/nltk_data"))
    from customer_side import run_for_client
    from sentiment_update import update_sentiment_and_gauge

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    loaders, sentiment_loaders = {}, {}
    if service is not None:
        from web.services.legacy_data_sources import make_legacy_loaders
        price_loader, news_loader = make_legacy_loaders(service, period, history_path)
        loaders = {"price_loader": price_loader, "news_loader": news_loader}
        sentiment_loaders = {"news_loader": news_loader}
    if Path(history_path).is_file():
        history = pd.read_csv(history_path, float_precision="round_trip")
        # The old ALL model uses every non-excluded column. Audit metadata from
        # replay archives must not silently become predictors.
        columns = pd.read_csv(ROOT / "df_merged_history.csv", nrows=0).columns
        history = history[[c for c in columns if c != "ret_lead1"]]
        history_path = directory / "df_merged_history.csv"
        history.to_csv(history_path, index=False)
    key = next((os.environ[k] for k in ("COINDESK_API_KEY", "CRYPTOCOMPARE_API_KEY", "CC_API_KEY")
                if os.environ.get(k)), "")
    run_for_client(key, str(history_path), output_dir=directory,
                   decision_date=period, refresh=True, **loaders)
    update_sentiment_and_gauge(output_dir=directory, as_of=period,
                               api_key=key, **sentiment_loaders)


def validate_outputs(directory, period):
    for relative in DATA_FILES:
        path = directory / relative
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"旧流程未生成完整输出：{relative}")
    history = pd.read_csv(directory / "df_merged_history.csv")
    if not {"date", "symbol", "return"}.issubset(history.columns):
        raise ValueError("旧流程历史数据字段不完整")
    history["date"] = pd.to_datetime(history.date)
    if history.date.max() != period or history.duplicated(["date", "symbol"]).any():
        raise ValueError("旧流程历史截止日期或唯一性不正确")
    assets = None
    for model in MODELS:
        frame = pd.read_csv(directory / f"pred_{model}.csv")
        if (not {"symbol", "y_pred"}.issubset(frame.columns) or frame.empty
                or frame.symbol.isna().any() or frame.symbol.duplicated().any()
                or not np.isfinite(pd.to_numeric(frame.y_pred, errors="coerce")).all()):
            raise ValueError(f"旧流程预测结果不完整：{model}")
        assets = len(frame) if assets is None else assets
    for relative in DATA_FILES:
        if relative.endswith(".png"):
            with (directory / relative).open("rb") as stream:
                if stream.read(8) != b"\x89PNG\r\n\x1a\n":
                    raise ValueError(f"情绪图片不是有效 PNG：{relative}")
    prices = pd.read_csv(directory / "stage_1_crypto_data.csv")
    prices["date"] = pd.to_datetime(prices.date)
    daily = prices.groupby("symbol").date.agg(["min", "max", "nunique"]).reset_index()
    daily["unquoted_days_inside_saved_span"] = (daily["max"]-daily["min"]).dt.days+1-daily["nunique"]
    daily.to_csv(directory / "daily_coverage.csv", index=False)
    news = pd.read_csv(directory / "figures/stage_1_news_raw.csv")
    return {"history_start": str(history.date.min().date()), "history_end": str(period.date()),
            "history_weeks": int(history.date.nunique()), "history_rows": len(history),
            "latest_forecast_assets": assets, "daily_start": str(prices.date.min().date()),
            "daily_end": str(prices.date.max().date()), "daily_rows": len(prices),
            "news_week_articles": len(news), "news_week_days": int(pd.to_datetime(news.date).dt.normalize().nunique()),
            "daily_gap_days": int(daily.unquoted_days_inside_saved_span.sum()),
            "note": "原手动计算流程；保留已替换的数据接口与已有历史，仅重算目标周。"}


def generate_report(directory, period):
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return "not_configured", "AI 服务未配置；数据、情绪与预测已更新"
    if os.environ.get("ANSEL_REPORT_OUTBOUND_ENABLED", "0") != "1":
        return "approval_required", "AI 周报外部连接保持关闭；数据、情绪与预测已更新"
    spec = importlib.util.spec_from_file_location("snapshot_market_report", ROOT / "ai_market_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name, relative in {
        "REPORT_FILE": "figures/weekly_report.md", "REPORT_UPDATE_FILE": "last_updated_report.txt",
        "HISTORY_FILE": "df_merged_history.csv", "SENTIMENT_WEEKLY_FILE": "figures/stage_2_sentiment_weekly.csv",
        "SENTIMENT_WEEKLY_FULL": "stage_2_sentiment_weekly.csv", "COMMON_WORDS_FILE": "figures/stage_1_common_words.csv",
        "COMMON_WORDS_FULL": "stage_1_common_words.csv", "NEWS_RAW_FILE": "figures/stage_1_news_raw.csv",
        "NEWS_RAW_FULL": "stage_1_news_raw.csv"}.items():
        setattr(module, name, directory / relative)
    module.PRED_FILES = {m: directory / f"pred_{m}.csv" for m in MODELS}
    module.get_latest_wednesday = lambda _: period.to_pydatetime()
    library = module.anthropic
    module.anthropic = SimpleNamespace(APIError=library.APIError,
        Anthropic=lambda **kwargs: library.Anthropic(**kwargs, max_retries=0, timeout=180))
    _, generated, message = module.generate_analysis_report(key, force_update=True)
    if not generated:
        return "failed", "AI 报告生成失败；数据与预测已保留"
    return "complete", None


def build(service, job_id, period):
    period = pd.Timestamp(period).normalize()
    active = service.published()
    stage = service.root / "staging" / job_id
    stage.mkdir(parents=True, exist_ok=False)
    if (active and active["period"] == str(period.date())
            and active["manifest"].get("pipeline_revision") == PIPELINE_REVISION
            and all((active["path"] / f).is_file() for f in DATA_FILES)):
        shutil.copytree(active["path"], stage, dirs_exist_ok=True)
        manifest = dict(active["manifest"])
    else:
        service.progress(job_id, "执行原手动更新流程")
        history_path = seed_history_path(service)
        run_legacy_workflows(stage, period, history_path, service=service)
        coverage = validate_outputs(stage, period)
        manifest = {"period": str(period.date()), "pipeline_revision": PIPELINE_REVISION,
            "generated_at": datetime.now(timezone.utc).isoformat(), "coverage": coverage,
            "workflow": "customer_side.run_for_client + sentiment_update.update_sentiment_and_gauge",
            "seed_history": str(history_path),
            "data_providers": "Existing Yahoo prices and configured news publisher adapters",
            "limitations": ["已有历史保留；目标周按旧函数重新计算。",
                             "数据接口替换保留，不宣称为原 CoinDesk 数据。"]}
    service.progress(job_id, "检查共享 AI 周报配置")
    try:
        report_status, report_error = generate_report(stage, period)
    except Exception:
        report_status, report_error = "failed", "AI 报告生成失败，数据与预测已保留"
    manifest.update(report_status=report_status, report_error=report_error)
    write_json(stage / "manifest.json", manifest)
    destination = service.root / "versions" / f"{period:%Y-%m-%d}_{job_id}"
    destination.parent.mkdir(exist_ok=True)
    stage.rename(destination)
    service.publish(str(period.date()), destination, manifest)
    result = {"period": str(period.date()), "report_status": report_status, "report_error": report_error}
    write_json(service.root / f"{job_id}.result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--period", required=True)
    args = parser.parse_args()
    build(WeeklyUpdates(args.root), args.job, args.period)
