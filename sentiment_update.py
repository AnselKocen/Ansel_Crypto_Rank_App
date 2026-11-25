from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

from crypto_project_pipeline import (
    build_week_dirs,
    stage1_etl,
    stage1_load_news,
    stage1_clean_text,
    stage2_feature_engineering,
    stage2_sentiment,
    merge_sentiment_feature
)
# 这个文件全部存在figures里
BASE_DIR = Path(__file__).resolve().parent
def update_sentiment_and_gauge(*, output_dir=None, as_of=None, api_key="",
                               news_loader=None):
    """Run the original sentiment workflow with an optional clock/output root.

    ``as_of`` replaces the old ``today`` clock; a UTC Wednesday at midnight
    produces the same seven-day window ending at Thursday 00:00 UTC. Outputs
    are isolated below ``output_dir`` when supplied.
    """
    # Use timezone-naive UTC datetime to avoid pandas comparison issues.
    if as_of is None:
        today = datetime.now(timezone.utc).replace(tzinfo=None)
    else:
        today = pd.Timestamp(as_of)
        if today.tzinfo is not None:
            today = today.tz_convert("UTC").tz_localize(None)
        today = today.to_pydatetime()

    news_start_date = today - timedelta(days=6)  # 包含今天在内，共7天
    news_end_date = today + timedelta(days=1)    # 为了包含今天，加1天（有些API是exclusive）

    print(f"Get news from {news_start_date.date()} to {news_end_date.date()}")

    output_root = Path(output_dir) if output_dir is not None else BASE_DIR
    save_dir = output_root / "figures"
    save_dir.mkdir(parents=True, exist_ok=True)
    df_news = (news_loader or stage1_load_news)(api_key, news_start_date, news_end_date, save_dir)
    df_clean = stage1_clean_text(df_news, save_dir)
    stage2_sentiment(df_clean, save_dir, save_dir)

    from crypto_project_pipeline import generate_sentiment_wordclouds
    df_sent = pd.read_csv(save_dir / "stage_2_news_sentiment.csv")
    # Also make news_end_date timezone-naive for wordcloud generation
    generate_sentiment_wordclouds(df_sent, save_dir, news_end_date)

    df_weekly_sent = pd.read_csv(save_dir / "stage_2_sentiment_weekly.csv")
    from crypto_project_pipeline import draw_fear_greed_gauge_from_latest
    draw_fear_greed_gauge_from_latest(df_weekly_sent, save_dir / "fear_greed_gauge.png")

    update_time = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    (output_root / "last_updated_wordcloud.txt").write_text(update_time)


if __name__ == "__main__":
    import sys
    try:
        update_sentiment_and_gauge()
        print("Sentiment update completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)