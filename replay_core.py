"""Shared, deterministic helpers for the project's historical replay."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "historical_replay"
MARKET_FEATURES = [
    "usd_volume", "base_volume",
    *[x for n in [14, 21, 28, 42, 90] for x in [f"momentum_{n}", f"volatility_{n}"]],
    "return_sign_consistency_3d", "long_candle_ratio_5d", "strev_daily", "strev_weekly",
]
ALL_FEATURES = [
    "usd_volume", "base_volume", *[f"v_{n}d" for n in [7, 14, 21, 28, 42]],
    *MARKET_FEATURES[2:], "compound_weekly", "extremely_positive_ratio",
    "extremely_negative_ratio",
]
SENTIMENT_FEATURES = ["compound_weekly", "extremely_positive_ratio", "extremely_negative_ratio"]
MODEL_PARAMS = {
    "enet_all": {"alpha": 0.01, "l1_ratio": 0.5},
    "enet_market": {"alpha": 0.01, "l1_ratio": 0.5},
    "extra_all": {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt"},
    "extra_market": {"n_estimators": 200, "max_depth": 10, "max_features": "sqrt"},
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=10))


def legacy_market_function():
    """Load the exact existing function without import-time downloads or writes."""
    source = ROOT / "crypto_project_pipeline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "stage2_feature_engineering")
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn,
    ], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"np": np, "pd": pd, "Path": Path, "logging": logging}
    exec(compile(module, str(source), "exec"), scope)
    return scope[fn.name]


def load_sentiment_tools(data_dir: Path):
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    nltk_dir = data_dir / "nltk_data"
    nltk_dir.mkdir(parents=True, exist_ok=True)
    nltk.data.path.insert(0, str(nltk_dir))
    for package, resource in [
        ("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet.zip"),
        ("vader_lexicon", "sentiment/vader_lexicon.zip"),
    ]:
        try:
            nltk.data.find(resource)
        except LookupError:
            if not nltk.download(package, download_dir=str(nltk_dir), quiet=True):
                raise RuntimeError(f"Unable to download required NLTK data: {package}")
    tree = ast.parse((ROOT / "crypto_project_pipeline.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "crypto_lexicon" for t in n.targets))
    analyzer = SentimentIntensityAnalyzer()
    analyzer.lexicon.update(ast.literal_eval(node.value))
    stop = set(stopwords.words("english"))
    lem = WordNetLemmatizer()

    def score(text: str) -> float:
        # Exactly the existing preprocessing: including its stopword/negation behavior.
        tokens = [t.lower() for t in word_tokenize(str(text))]
        cleaned = " ".join(lem.lemmatize(t) for t in tokens if t.isalpha() and t not in stop)
        return analyzer.polarity_scores(cleaned)["compound"]
    return score


def weekly_market(prices: pd.DataFrame, decision: pd.Timestamp, function=None) -> pd.DataFrame:
    """Reproduce a weekly run's 110-day input, not whole-replay quantiles."""
    decision = pd.Timestamp(decision).normalize()
    frame = prices[(prices.date <= decision) &
                   (prices.date >= decision - pd.Timedelta(days=109))].copy()
    if frame.empty:
        return pd.DataFrame()
    fn = function or legacy_market_function()
    result = fn(frame[["symbol", "date", "open", "high", "low", "close",
                      "usd_volume", "base_volume"]], data_dir=None)
    result = result[result.date == decision].copy()
    # resample(...).last() can label a stale Tuesday observation as Wednesday.
    # An asset without an actual decision-day daily close is not eligible.
    quoted_today = set(frame.loc[frame.date == decision, "symbol"])
    quoted_previous = set(frame.loc[frame.date == decision - pd.Timedelta(days=7), "symbol"])
    result = result[result.symbol.isin(quoted_today & quoted_previous)].copy()
    # The original uses zero for warm-up NaNs; retain that rule and expose coverage.
    counts = frame.groupby("symbol").date.nunique()
    result["price_days_in_window"] = result.symbol.map(counts)
    return result


def weekly_sentiment(news: pd.DataFrame, decision: pd.Timestamp) -> tuple[dict, int]:
    lo = pd.Timestamp(decision).normalize() - pd.Timedelta(days=6)
    hi = pd.Timestamp(decision).normalize() + pd.Timedelta(days=1)
    week = news[(news.published_at >= lo) & (news.published_at < hi)]
    values = pd.to_numeric(week.compound, errors="coerce").dropna()
    if values.empty:
        return {x: np.nan for x in SENTIMENT_FEATURES}, 0
    # Original update downloads one new week. Quantiles are fitted on that week.
    qlo, qhi = values.quantile([0.1, 0.9])
    return {
        "compound_weekly": float(values.mean()),
        "extremely_positive_ratio": float((values > qhi).mean()),
        "extremely_negative_ratio": float((values < qlo).mean()),
        "sentiment_q10": float(qlo), "sentiment_q90": float(qhi),
    }, len(values)


def train_predict(history: pd.DataFrame, decision: pd.Timestamp) -> tuple[pd.DataFrame, list[dict]]:
    decision = pd.Timestamp(decision).normalize()
    visible = history[history.date <= decision].copy()
    visible = visible.drop(columns=["ret_lead1"], errors="ignore")
    # Preserve original row order: ExtraTrees can change tie-breaking with a new
    # training order even when its seed and hyperparameters are unchanged.
    visible = visible.drop_duplicates(["date", "symbol"], keep="last")
    required = pd.date_range(end=decision - pd.Timedelta(days=7), periods=52, freq="W-WED")
    missing = required.difference(pd.DatetimeIndex(visible.date.unique()))
    if len(missing):
        raise ValueError(f"Training window missing calendar weeks: {missing.strftime('%Y-%m-%d').tolist()}")
    # Date join, rather than shift to next observation across a missing week.
    labels = visible[["symbol", "date", "return"]].copy()
    labels["date"] -= pd.Timedelta(days=7)
    labels = labels.rename(columns={"return": "ret_lead1"})
    labeled = visible.merge(labels, on=["symbol", "date"], how="left", validate="one_to_one")
    train = labeled[labeled.date.isin(required)].dropna(subset=["ret_lead1"])
    test = labeled[labeled.date == decision]
    if train.empty or test.empty:
        raise ValueError("Empty training data or decision-week data")
    outputs, events = [], []
    for name, params in MODEL_PARAMS.items():
        features = ALL_FEATURES if name.endswith("_all") else MARKET_FEATURES
        eligible = test.copy()
        if name.endswith("_all") and "sentiment_available" in eligible:
            eligible = eligible[eligible.sentiment_available.fillna(False).astype(bool)]
        if eligible.empty:
            events.append({"model": name, "status": "missing_sentiment_or_inputs", "rows": 0})
            continue
        xtrain = train[features].replace([np.inf, -np.inf], np.nan)
        xtest = eligible[features].replace([np.inf, -np.inf], np.nan)
        imputer = SimpleImputer(strategy="mean")
        xtrain = imputer.fit_transform(xtrain)
        xtest = imputer.transform(xtest)
        estimator = (ElasticNet(max_iter=10000, random_state=42, **params)
                     if name.startswith("enet") else
                     ExtraTreesRegressor(random_state=42, **params))
        pipeline = Pipeline([("scale", StandardScaler()), ("model", estimator)])
        pipeline.fit(xtrain, train.ret_lead1.to_numpy())
        out = eligible[["symbol"]].copy()
        out["predicted_return"] = pipeline.predict(xtest)
        out = out.sort_values("predicted_return", ascending=False, kind="stable").reset_index(drop=True)
        out["predicted_rank"] = np.arange(1, len(out) + 1)
        out["is_top20"] = out.predicted_rank <= 20
        out["is_bottom20"] = out.predicted_rank > max(len(out) - 20, 20)
        out["model_name"] = name
        out["decision_date"] = decision
        out["target_start"] = decision + pd.Timedelta(days=1)
        out["target_end"] = decision + pd.Timedelta(days=7)
        out["data_cutoff_utc"] = (decision + pd.Timedelta(days=1)).isoformat() + "Z (exclusive)"
        out["train_start"], out["train_end"] = required[0], required[-1]
        out["training_rows"] = len(train)
        out["forecast_id"] = [f"{decision:%Y-%m-%d}:{name}:{s}" for s in out.symbol]
        outputs.append(out)
        events.append({"model": name, "status": "complete", "rows": len(out),
                       "training_rows": len(train)})
    if not outputs:
        return pd.DataFrame(), events
    return pd.concat(outputs, ignore_index=True), events
