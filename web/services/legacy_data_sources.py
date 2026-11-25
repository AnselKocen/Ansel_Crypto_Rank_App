"""Adapt the configured saved/Yahoo/publisher inputs to the original ETL schemas.

This module only obtains and stores observations. The original pipeline remains
responsible for text cleaning, sentiment, market features, and model fitting.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RESEARCH = PROJECT / "artifacts/historical_replay"
PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close",
                 "usd_volume", "base_volume"]
NEWS_COLUMNS = ["article_id", "published_at", "title", "body", "url", "source"]
ORIGINAL_NEWS = {"coindesk", "coindesk_cached"}
PUBLIC_NEWS = {"bitcoinmagazine", "cryptopotato"}


def _utc_naive(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_convert("UTC").tz_localize(None) if stamp.tzinfo else stamp


def _read(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def _save(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _existing(paths):
    # Preserve precedence, without reading the same active/history directory twice.
    return list(dict.fromkeys(Path(p) for p in paths if Path(p).is_file()))


def _prices(paths):
    frames = []
    for path in _existing(paths):
        frame = _read(path)
        if not set(PRICE_COLUMNS).issubset(frame.columns):
            raise ValueError(f"Saved price schema is incomplete: {path.name}")
        frame["date"] = pd.to_datetime(frame.date, format="mixed", utc=True).dt.tz_localize(None)
        if "source" not in frame:
            frame["source"] = "coindesk_cached"
        frames.append(frame)
    if not frames:
        empty = pd.DataFrame(columns=PRICE_COLUMNS + ["source", "source_symbol"])
        empty["date"] = pd.Series(dtype="datetime64[ns]")
        return empty
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates(["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"]).reset_index(drop=True))


def _news(paths):
    frames = []
    for path in _existing(paths):
        frame = _read(path)
        if "published_at" not in frame and "date" in frame:
            frame = frame.rename(columns={"date": "published_at"})
        if not {"published_at", "body"}.issubset(frame.columns):
            raise ValueError(f"Saved news schema is incomplete: {path.name}")
        frame["published_at"] = pd.to_datetime(frame.published_at, format="mixed", utc=True).dt.tz_localize(None)
        if "source" not in frame:
            frame["source"] = "coindesk_cached"
        if "article_id" not in frame:
            if "id" not in frame:
                raise ValueError(f"Saved news has no article identifier: {path.name}")
            identifiers = frame.id.astype(str)
            prefix = frame.source.replace("coindesk_cached", "coindesk").astype(str)
            frame["article_id"] = identifiers.where(identifiers.str.contains(":"), prefix + ":" + identifiers)
        for column in ["title", "body", "url"]:
            frame[column] = frame[column].fillna("") if column in frame else ""
        frames.append(frame)
    if not frames:
        empty = pd.DataFrame(columns=NEWS_COLUMNS)
        empty["published_at"] = pd.Series(dtype="datetime64[ns]")
        return empty
    return (pd.concat(frames, ignore_index=True)
            .drop_duplicates("article_id", keep="last")
            .sort_values("published_at").reset_index(drop=True))


def make_legacy_loaders(service, period, history_path):
    """Return replacements for ``stage1_etl`` and ``stage1_load_news`` only.

    The established replacement-provider coin universe comes from the last
    saved feature week before ``period``. Cached observations keep their source
    metadata; only the original input columns are exposed to the old pipeline.
    """
    period = _utc_naive(period).normalize()
    history_path = Path(history_path)
    cache = Path(service.root) / "cache"
    publication = service.published()
    active = Path(publication["path"]) if publication else None
    price_paths = [PROJECT / "stage_1_crypto_data.csv",
                   RESEARCH / "target_experiment/reviewed_prices.csv",
                   history_path.parent / "stage_1_crypto_data.csv",
                   history_path.parent / "daily_prices.csv"]
    news_paths = [PROJECT / "stage_1_news_raw.csv",
                  PROJECT / "figures/stage_1_news_raw.csv",
                  RESEARCH / "news_articles.csv",
                  history_path.parent / "stage_1_news_raw.csv",
                  history_path.parent / "news_articles.csv"]
    if active:
        price_paths += [active / "stage_1_crypto_data.csv", active / "daily_prices.csv"]
        news_paths += [active / "stage_1_news_raw.csv", active / "news_articles.csv"]
    price_paths.append(cache / "prices.csv")
    news_paths.append(cache / "news.csv")
    saved_prices = None
    saved_news = None

    def price_loader(api_key, pages, top_limit=100, history_limit=2000,
                     currency="USD", sleep_sec=0.5, data_dir=None,
                     filename="stage_1_crypto_data.csv"):
        nonlocal saved_prices
        if currency.upper() != "USD":
            raise ValueError("The configured Yahoo observations use USD")
        history = _read(history_path)
        history["date"] = pd.to_datetime(history.date, format="mixed", utc=True).dt.tz_localize(None)
        before = history[history.date.lt(period)]
        if before.empty:
            raise ValueError("A saved feature week before the decision date is required")
        symbols = sorted(before.loc[before.date.eq(before.date.max()), "symbol"].dropna().unique())
        if not symbols or int(history_limit) < 1:
            raise ValueError("The price universe and history window must be nonempty")
        start = period - pd.Timedelta(days=int(history_limit)-1)
        if saved_prices is None:
            saved_prices = _prices(price_paths)
        additions = []
        for symbol in symbols:
            old = saved_prices[saved_prices.symbol.eq(symbol) & saved_prices.date.le(period)]
            if old.date.eq(period).any():
                continue
            # Only obtain the extension; existing source observations are not
            # rescaled, repaired, filtered, or recomputed here.
            from prepare_replay_data import ALIASES, get_prices
            fetch_start = max(start, old.date.max() + pd.Timedelta(days=1)) if len(old) else start
            codes = old.get("source_symbol", pd.Series(dtype=str)).dropna().astype(str)
            codes = codes[codes.str.endswith("-USD")]
            code = codes.iloc[-1] if len(codes) else ALIASES.get(symbol, symbol)+"-USD"
            try:
                fetched = get_prices(symbol, fetch_start, period, "yahoo", cache, "", yahoo_symbol=code)
            except Exception as exc:
                raise RuntimeError(f"Yahoo prices unavailable for {symbol}") from exc
            if fetched.empty or not fetched.date.eq(period).any():
                raise ValueError(f"No completed decision-day price for {symbol}")
            additions.append(fetched)
        if additions:
            saved_prices = (pd.concat([saved_prices, *additions], ignore_index=True)
                            .drop_duplicates(["symbol", "date"], keep="last")
                            .sort_values(["symbol", "date"]).reset_index(drop=True))
        selected = saved_prices[saved_prices.symbol.isin(symbols) &
                                saved_prices.date.between(start, period)].copy()
        available = set(selected.loc[selected.date.eq(period), "symbol"])
        if selected.empty or set(symbols) - available:
            raise ValueError("Saved prices do not cover the decision-day universe")
        _save(cache / "prices.csv", saved_prices)
        result = selected[PRICE_COLUMNS].copy()
        result["usd_volume_mil"] = result.usd_volume / 1e6
        result = result.sort_values(["symbol", "date"]).reset_index(drop=True)
        if data_dir is not None:
            _save(Path(data_dir) / filename, result)
        return result.set_index(["symbol", "date"])

    def news_loader(api_key, start_dt, end_dt, data_dir,
                    filename="stage_1_news_raw.csv"):
        nonlocal saved_news
        start, end = _utc_naive(start_dt), _utc_naive(end_dt)
        if start >= end or end > period + pd.Timedelta(days=1):
            raise ValueError("News range must end by the completed decision week")
        if saved_news is None:
            saved_news = _news(news_paths)
        days = set(pd.date_range(start.normalize(), (end-pd.Timedelta(nanoseconds=1)).normalize(), freq="D"))

        def window(sources):
            return saved_news[saved_news.source.isin(sources) &
                              saved_news.published_at.ge(start) & saved_news.published_at.lt(end)].copy()

        def covered(frame):
            usable = frame[frame.body.fillna("").astype(str).str.strip().ne("")]
            return set(usable.published_at.dt.normalize())

        chosen = window(ORIGINAL_NEWS)
        if not days.issubset(covered(chosen)):
            chosen = window(PUBLIC_NEWS)
            missing = days - covered(chosen)
            if missing:
                from prepare_replay_data import PUBLISHERS, wordpress_news
                additions = []
                for name, url in PUBLISHERS.items():
                    fetched, _ = wordpress_news(name, url, min(missing), max(missing), cache)
                    if not fetched.empty:
                        additions.append(fetched)
                if additions:
                    saved_news = (pd.concat([saved_news, *additions], ignore_index=True)
                                  .drop_duplicates("article_id", keep="last")
                                  .sort_values("published_at").reset_index(drop=True))
                chosen = window(PUBLIC_NEWS)
        if chosen.empty or not days.issubset(covered(chosen)):
            raise ValueError("News does not cover every requested UTC day; update was not published")
        _save(cache / "news.csv", saved_news)
        result = chosen[NEWS_COLUMNS].rename(columns={"article_id": "id", "published_at": "date"})
        result["positive"] = np.nan
        result = result[["date", "id", "title", "body", "source", "url", "positive"]].reset_index(drop=True)
        _save(Path(data_dir) / filename, result)
        return result

    return price_loader, news_loader
