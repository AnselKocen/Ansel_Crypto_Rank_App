import sys
from pathlib import Path

# 设置 BASE_DIR 并添加到 sys.path（必须在导入项目模块之前）
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from datetime import datetime, timedelta, timezone

from crypto_project_pipeline import (
    build_week_dirs,
    stage1_etl,
    stage1_load_news,
    stage1_clean_text,
    stage2_feature_engineering,
    stage2_sentiment,
    merge_sentiment_feature
)
# 这个文件更新全部存在当前脚本目录下。
# ==============================================
#  Step 0: 自动确定最近的周三（返回 datetime）
# ==============================================
def get_latest_wednesday(today: datetime) -> datetime:
    days_since_wed = (today.weekday() - 2) % 7
    return today - timedelta(days=days_since_wed)

# ==============================================
#  Step 1: 增量更新 df_merged（只下载新的一周）
# ==============================================
def update_df_merged(api_key: str, history_path: str, output_root = BASE_DIR.name,
                     *, output_dir=None, decision_date=None, refresh=False,
                     price_loader=None, news_loader=None):
    """Run the original updater, optionally writing an isolated weekly snapshot.

    ``decision_date`` fixes the UTC Wednesday cutoff. ``refresh`` rebuilds that
    week from a truncated in-memory history; an output directory keeps the input
    history untouched. Optional loaders adapt the already configured data
    providers to the original stage schemas. With no optional arguments the
    manual entry point retains its existing paths and freshness checks.
    """
    notice_list = []
    today = datetime.now(timezone.utc).replace(tzinfo=None)
    latest_wed = get_latest_wednesday(today)  # 保持为 datetime 类型
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import joblib
    # 最新的要保证我们传入的df是没有ret_lead1的
    # 读取历史数据并检查是否已更新
    df_hist = pd.read_csv(history_path)
    df_hist["date"] = pd.to_datetime(df_hist["date"], utc=True).dt.tz_localize(None)
    if decision_date is not None:
        cutoff = pd.Timestamp(decision_date)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        latest_wed = cutoff.normalize().to_pydatetime()
        if latest_wed.weekday() != 2:
            raise ValueError("decision_date must be a Wednesday in UTC")
        df_hist = df_hist[df_hist["date"] <= latest_wed].copy()
    if refresh:
        df_hist = df_hist[df_hist["date"] < latest_wed].copy()
    if df_hist.empty:
        raise ValueError("Historical data before the decision week is required")
    last_date = df_hist["date"].max().date()
    next_day = last_date + timedelta(days=1)

    data_dir = Path(output_dir) if output_dir is not None else BASE_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = data_dir
    history_output = (data_dir / "df_merged_history.csv"
                      if output_dir is not None else Path(history_path))

    print(f"###NOTICE### Latest Date in Historical Data: {last_date}，Latest Wednesday: {latest_wed.date()}")
    notice_list.append(f"Latest Date in Historical Data: {last_date}，Latest Wednesday: {latest_wed.date()}")
    if last_date >= latest_wed.date():
        print("Data already includes the most recent week. No update needed.")
        notice_list.append("Data already includes the most recent week. No update needed.")
        if output_dir is not None:
            df_hist.to_csv(history_output, index=False)
        return df_hist.copy(), latest_wed, (data_dir if output_dir is not None else Path(history_path).parent),notice_list

    #elif today.weekday() == 2:  # Wednesday
        #notice_list.append("Today is Wednesday(UTC)! Please wait until Thursday to ensure complete data.")
        #return df_hist.copy(), latest_wed, Path(history_path).parent,notice_list
    # === 如果今天是周三，且历史数据已经包含了上一周的完整数据，则不更新
    elif decision_date is None and today.weekday() == 2 and last_date >= (latest_wed - timedelta(days=7)).date():
        print("Today is Wednesday (UTC), and last week’s data has already been fully updated. Please wait until Thursday to get the latest week’s complete data.")
        notice_list.append(
            "Today is Wednesday (UTC), and last week’s data has already been fully updated. Please wait until Thursday to get the latest week’s complete data.")
        if output_dir is not None:
            df_hist.to_csv(history_output, index=False)
        return df_hist.copy(), latest_wed, (data_dir if output_dir is not None else Path(history_path).parent), notice_list

    # === 拉取最近的价格数据并构建 market 特征 ===
    (price_loader or stage1_etl)(
        api_key=api_key,
        pages=[1,2],
        top_limit=100,
        history_limit=110,# 不改如果真的很久没有抓他的话 120天就不够了 这个我们之后再想办法
        currency="USD",
        data_dir=data_dir
    )
    # 去除 latest_wed 之后的数据，避免出现未来行情
    #df_prices = pd.read_csv(data_dir / "stage_1_crypto_data.csv")
    #df_prices.columns = [c.strip().lower() for c in df_prices.columns]

    #df_prices.columns = [col.strip().lower() for col in df_prices.columns]
    #print(f"是否包含 'date'：{'date' in df_prices.columns}")
    print('-------------------------------------------------------------------------------------------------------')

    #df_prices["date"] = pd.to_datetime(df_prices["date"])
   # df_prices = df_prices[df_prices["date"] <= latest_wed]
    #print(df_prices)
    #print("📅 df_prices 最大日期：", df_prices["date"].max())
    last_date_df = pd.to_datetime(last_date)
    df_prices = pd.read_csv(data_dir / "stage_1_crypto_data.csv")
    df_prices["date"] = pd.to_datetime(df_prices["date"], utc=True).dt.tz_localize(None) ###
    df_prices = df_prices[df_prices["date"] <= latest_wed] ###
    #print(last_date_df)
    df_features = stage2_feature_engineering(df_prices, data_dir)

    print(df_features)
    # 统一为 Python date 类型进行比较
    df_features["date"] = pd.to_datetime(df_features["date"]).dt.date
    print(df_features)
    # last_date 已经是 date 类型（第42行），直接比较
    df_features = df_features[df_features["date"].apply(lambda x: x > last_date)]
    print(df_features)

    # === 抓取新闻并计算 sentiment 特征 === 修改了抓取时间
    #news_start_date = latest_wed - timedelta(days=6)
    #news_end_date = latest_wed + timedelta(days=1)
    print(last_date)
    print(latest_wed)
    #news_start_date = last_date + timedelta(days=1)
    news_start_date = datetime.combine(last_date + timedelta(days=1), datetime.min.time())
    news_end_date = latest_wed + timedelta(days=1)# +1 是为了包含周三
    print(f"抓取新闻时间范围: {news_start_date} 至 {news_end_date}")
    df_news = (news_loader or stage1_load_news)(api_key, news_start_date, news_end_date, data_dir)
    df_clean = stage1_clean_text(df_news, data_dir)
    stage2_sentiment(df_clean, data_dir, fig_dir)

    # 更新词云
    # from crypto_project_pipeline import generate_sentiment_wordclouds
    # generate_sentiment_wordclouds(df_sent, fig_dir, latest_wed)
    # 存到更新路径
    #renew_path = BASE_DIR / "figures"
    #generate_sentiment_wordclouds(df_sent, renew_path, latest_wed)
    # === 合并 market 和 sentiment 特征 ===
    df_weekly_sent = pd.read_csv(data_dir / "stage_2_sentiment_weekly.csv")
    df_new = merge_sentiment_feature(df_features, df_weekly_sent, save_path=data_dir)
    #df_new["ret_lead1"] = df_new.groupby("symbol")["return"].shift(-1) #不应该在这里mergeret_lead1，删除

    # === 合并到历史数据并保存 ===
    df_combined = pd.concat([df_hist, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["date", "symbol"]).sort_values("date")
    print(f"⚠即将覆盖历史文件：{history_output}，保存最新的 df_merged，共 {df_combined['date'].nunique()} 周数据。")
    df_combined.to_csv(history_output, index=False)
    # 生成仪表盘
    # from crypto_project_pipeline import draw_fear_greed_gauge_from_latest
    # df_latest = pd.read_csv(BASE_DIR / "df_merged_history.csv")
    # draw_fear_greed_gauge_from_latest(df_latest, fig_dir / "fear_greed_gauge1.png")
    # 更新仪表盘
    #renew_path = BASE_DIR/"figures"
    #draw_fear_greed_gauge_from_latest(df_latest, renew_path / "fear_greed_gauge1.png")
    return df_combined, latest_wed, data_dir,notice_list




# ==============================================
# Step 2: 模型训练 + 预测（只预测最新一周）
# ==============================================
def run_prediction_pipeline(api_key: str, history_path: str, *, output_dir=None,
                            decision_date=None, refresh=False,
                            price_loader=None, news_loader=None):
    df_merged, latest_wed, data_dir, notice_list = update_df_merged(
        api_key, history_path, output_dir=output_dir,
        decision_date=decision_date, refresh=refresh,
        price_loader=price_loader, news_loader=news_loader)
    return predict_from_history(df_merged, latest_wed, data_dir, notice_list)


def predict_from_history(df_merged, latest_wed, data_dir, notice_list=None):
    """Run the unchanged manual training/prediction block on supplied history."""
    notice_list = [] if notice_list is None else notice_list
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    df_merged = df_merged.copy()
    import pandas as pd
    # Ensure date column is datetime and latest_wed is comparable
    # Remove timezone info from latest_wed to match tz-naive df_merged dates
    df_merged["date"] = pd.to_datetime(df_merged["date"], utc=True).dt.tz_localize(None)
    latest_wed = pd.Timestamp(latest_wed)
    if latest_wed.tzinfo is not None:
        latest_wed = latest_wed.tz_convert("UTC").tz_localize(None)
    df_merged = df_merged[df_merged["date"] <= latest_wed].copy()
    import numpy as np
    from sklearn.linear_model import ElasticNet
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import joblib

    weeks = sorted(df_merged["date"].unique())
    train_weeks = weeks[-53:-1]  # 52周训练
    test_week = weeks[-1]        # 最新一周预测

    # 应该在这里加一句，为我们读取的df_merged 构建ret_lead1，后续 都是正常的操作，不需要更改了。下面这段是新添加的
    df_merged["ret_lead1"] = (
        df_merged.sort_values(["symbol", "date"])
        .groupby("symbol")["return"]
        .shift(-1)
    )

    df_train = df_merged[df_merged["date"].isin(train_weeks)].dropna(subset=["ret_lead1"])
    df_test = df_merged[df_merged["date"] == test_week]

    feature_sets = {
        "all": [col for col in df_train.columns if col not in ["date", "symbol", "return", "ret_lead1", "open", "high", "low", "close"]],
        "market": [col for col in df_train.columns if any(k in col for k in ["momentum", "volatility", "usd_volume", "base_volume", "return_sign", "long_candle", "strev"])]
    }

    fixed_params = {
        "enet_all": {"alpha": 0.01, "l1_ratio": 0.5},
        "enet_market": {"alpha": 0.01, "l1_ratio": 0.5},
        "extra_all": {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt"},
        "extra_market": {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt"}
    }

    results = {}

    for model_key in ["enet", "extra"]:
        for tag in ["all", "market"]:
            name = f"{model_key}_{tag}"
            print(f"\n 正在训练模型 {name} ...")

            X_train = df_train[feature_sets[tag]].copy()
            y_train = df_train["ret_lead1"].fillna(0)
            X_test = df_test[feature_sets[tag]].copy()

            imputer = SimpleImputer(strategy="mean")
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)

            if model_key == "enet":
                model = ElasticNet(max_iter=10000, random_state=42, **fixed_params[name])
            else:
                model = ExtraTreesRegressor(random_state=42, **fixed_params[name])

            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            df_out = df_test[["symbol"]].copy()
            df_out["y_pred"] = y_pred
            df_out = df_out.sort_values("y_pred", ascending=False)

            results[name] = df_out
            out_path = data_dir / f"pred_{name}.csv"
            df_out.to_csv(out_path, index=False)
            print(f" {name} 输出保存至 {out_path}")

    # === 打印推荐币种 + 构造返回结果 ===
    results_dict = {}

    for name, df in results.items():
        print(f"\n 模型: {name}")
        print("Top 20 推荐币种:")
        print(df.head(20).to_string(index=False))
        print("Bottom 20 做空建议:")
        print(df.tail(20).to_string(index=False))

        top_list = df.head(20)["symbol"].tolist()
        bot_list = df.tail(20)["symbol"].tolist()

        print(f">>>TOP20_{name}: " + ",      ".join(top_list))
        print(f">>>BOT20_{name}: " + ",      ".join(bot_list))

        results_dict[name] = {
            "top": top_list,
            "bot": bot_list
        }
    return results_dict, notice_list


def run_for_client(api_key: str, history_path: str, *, output_dir=None,
                   decision_date=None, refresh=False,
                   price_loader=None, news_loader=None):
    return run_prediction_pipeline(api_key, history_path, output_dir=output_dir,
                                   decision_date=decision_date, refresh=refresh,
                                   price_loader=price_loader, news_loader=news_loader)


if __name__ == "__main__":
    import sys
    API_KEY = ""  # Coindesk 免费额度无需 key
    HISTORY_PATH = BASE_DIR / "df_merged_history.csv"
    try:
        results, notices = run_for_client(API_KEY, str(HISTORY_PATH))
        print("Prediction pipeline completed!")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

