#!/usr/bin/env python3
"""Download/cache research data and build point-in-time weekly feature packages."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from replay_core import (
    ROOT, DEFAULT_OUTPUT, SENTIMENT_FEATURES, digest, legacy_market_function,
    load_sentiment_tools, weekly_market, weekly_sentiment, write_csv, write_json,
)

ALIASES = {"MANTLE": "MNT", "MIOTA": "IOTA", "RONIN": "RON"}
PUBLISHERS = {
    "bitcoinmagazine": "https://bitcoinmagazine.com/wp-json/wp/v2/posts",
    "cryptopotato": "https://cryptopotato.com/wp-json/wp/v2/posts",
}
SEARCH_NAMES = {
    "BTT": "BitTorrent", "BUIDL": "BlackRock BUIDL", "CHEEMS": "Cheems",
    "CKB": "Nervos Network", "FXS": "Frax Share", "GALA": "Gala",
    "IOTX": "IoTeX", "IP": "Story", "MATIC": "Polygon", "MSQ": "MSQUARE",
    "NFT": "APENFT", "OKB": "OKB", "QUBIC": "Qubic", "RNDR": "Render",
    "RYO": "Ryo Currency", "TRIBE": "Tribe", "VELO": "Velo",
}
AGENT = {"User-Agent": "AnselCryptoResearch/1.0"}


def api_key() -> str:
    names = ["COINDESK_API_KEY", "CRYPTOCOMPARE_API_KEY", "CC_API_KEY"]
    for name in names:
        if os.environ.get(name):
            return os.environ[name]
    for filename in [".env", ".streamlit/secrets.toml"]:
        p = ROOT / filename
        if p.exists():
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    if k.strip() in names:
                        return v.strip().strip("\"'")
    return ""


def get_json(url, params, cache: Path, key=""):
    if cache.exists():
        return json.loads(cache.read_text()), {}
    cache.parent.mkdir(parents=True, exist_ok=True)
    headers = dict(AGENT)
    if key:
        headers["authorization"] = f"Apikey {key}"
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=35)
            if response.status_code == 400:
                try:
                    code = response.json().get("code", "")
                except ValueError:
                    code = ""
                raise RuntimeError(f"HTTP 400 {code} at {url.split('?')[0]}")
            if response.status_code in [401, 403, 404, 451]:
                raise RuntimeError(f"HTTP {response.status_code} at {url.split('?')[0]}")
            response.raise_for_status()
            value = response.json()
            write_json(cache, value)
            return value, dict(response.headers)
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt * 2)


def text_only(value):
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1>", "", value or "", flags=re.S | re.I)
    return html.unescape(re.sub(r"<[^>]+>", " ", text)).strip()


def get_prices(symbol, start, end, provider, out, key, yahoo_symbol=None):
    suffix = f"__{yahoo_symbol}" if yahoo_symbol else ""
    cache = out / "raw" / "prices" / provider / f"{symbol}{suffix}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if provider == "coindesk":
        obj, _ = get_json(
            "https://data-api.coindesk.com/index/cc/v1/historical/days",
            {"market": "cadli", "instrument": f"{symbol}-USD",
             "limit": (end - start).days + 2, "aggregate": 1, "fill": "true",
             "apply_mapping": "true", "to_ts": int((end + pd.Timedelta(days=1)).timestamp()) - 1},
            cache, key,
        )
        data = obj.get("Data")
        if not isinstance(data, list) or not data:
            raise RuntimeError("No price observations")
        frame = pd.DataFrame(data).rename(columns={
            "OPEN": "open", "HIGH": "high", "LOW": "low", "CLOSE": "close",
            "VOLUME": "base_volume", "QUOTE_VOLUME": "usd_volume",
        })
        frame["date"] = pd.to_datetime(frame.TIMESTAMP, unit="s")
        frame["source_symbol"] = symbol
        frame["base_volume_estimated"] = False
    else:
        source_symbol = yahoo_symbol or ALIASES.get(symbol, symbol) + "-USD"
        obj, _ = get_json(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{source_symbol}",
            {"period1": int(start.timestamp()),
             "period2": int((end + pd.Timedelta(days=1)).timestamp()), "interval": "1d"},
            cache,
        )
        result = (obj.get("chart", {}).get("result") or [None])[0]
        if not result or not result.get("timestamp"):
            raise RuntimeError("No Yahoo observations")
        quote = result["indicators"]["quote"][0]
        frame = pd.DataFrame({c: quote.get(c) for c in ["open", "high", "low", "close", "volume"]})
        frame["date"] = pd.to_datetime(result["timestamp"], unit="s")
        # Yahoo cryptocurrency volume is quote-currency volume, not native base volume.
        frame["usd_volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame["base_volume"] = frame.usd_volume / pd.to_numeric(frame.close, errors="coerce")
        frame["source_symbol"] = source_symbol
        frame["source_name"] = result.get("meta", {}).get("shortName", "")
        frame["base_volume_estimated"] = True
    frame["symbol"], frame["source"] = symbol, provider
    frame = frame[(frame.date >= start) & (frame.date <= end)].copy()
    # A stale intraday quote appended by Yahoo is not a completed UTC daily bar.
    frame = frame[frame.date.eq(frame.date.dt.normalize())]
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame = frame[(frame.close > 0) & (frame.open > 0)]
    columns = ["symbol", "date", "open", "high", "low", "close", "usd_volume",
               "base_volume", "source", "source_symbol", "base_volume_estimated"]
    return frame[columns].drop_duplicates(["symbol", "date"]).sort_values("date")


def yahoo_catalog(out):
    quotes, seen = [], set()
    for offset in range(0, 1000, 250):
        obj, _ = get_json("https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved",
                          {"scrIds": "all_cryptocurrencies_us", "count": 250, "start": offset},
                          out / "raw" / f"crypto_catalog_{offset}.json")
        batch = (obj.get("finance", {}).get("result") or [{}])[0].get("quotes", [])
        fresh = [q for q in batch if q.get("symbol") not in seen]
        if not fresh:
            break
        quotes.extend(fresh)
        seen.update(q["symbol"] for q in fresh)
    return quotes


def matched_yahoo_prices(symbol, start, end, out, baseline, catalog):
    """Resolve public numeric tickers; reject homonyms against pre-replay closes."""
    alias = ALIASES.get(symbol, symbol)
    candidates = [alias + "-USD"]
    for q in catalog:
        code = q.get("symbol", "")
        base = re.sub(r"\d+$", "", code.removesuffix("-USD"))
        if code.endswith("-USD") and base == alias:
            candidates.append(code)
    candidates = list(dict.fromkeys(candidates))
    reference_end = baseline.date.max()
    ref = baseline[(baseline.symbol == symbol) &
                   (baseline.date >= reference_end - pd.Timedelta(days=35))]
    ref = ref.set_index("date").close
    errors = []
    index, searched = 0, False
    while index < len(candidates) or not searched:
        if index >= len(candidates):
            searched = True
            if symbol not in SEARCH_NAMES:
                break
            obj, _ = get_json(
                "https://query1.finance.yahoo.com/v1/finance/search",
                {"q": SEARCH_NAMES[symbol], "quotesCount": 25, "newsCount": 0},
                out / "raw/symbol_search" / f"{symbol}_name.json")
            for q in obj.get("quotes", []):
                code = q.get("symbol", "")
                if q.get("quoteType") == "CRYPTOCURRENCY" and code.endswith("-USD") and code not in candidates:
                    candidates.append(code)
            if index >= len(candidates):
                break
        candidate = candidates[index]
        index += 1
        try:
            # Reuse the initial direct-code response; resolved codes have separate caches.
            code = None if candidate == alias + "-USD" else candidate
            frame = get_prices(symbol, start, end, "yahoo", out, "", code)
            paired = pd.concat([ref.rename("reference"), frame.set_index("date").close.rename("public")],
                               axis=1, join="inner").dropna()
            paired = paired[(paired.reference > 0) & (paired.public > 0)]
            error = float(np.median(np.abs(np.log(paired.public / paired.reference)))) if len(paired) else np.inf
            if len(paired) >= 3 and error <= np.log(1.25):
                return frame, candidate, error
            errors.append(f"{candidate}: pre-start reference mismatch ({len(paired)} pairs)")
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("; ".join(errors)[:600])


def local_news(out):
    frames = []
    for filename in ["stage_2_news_sentiment.csv", "figures/stage_2_news_sentiment.csv"]:
        path = ROOT / filename
        if not path.exists():
            continue
        f = pd.read_csv(path, low_memory=False)
        f["published_at"] = pd.to_datetime(f["date"], utc=True).dt.tz_localize(None)
        f["article_id"] = "coindesk:" + f.id.astype(str)
        f["source"] = "coindesk_cached"
        f["url"] = f.get("url", "")
        f["body"] = f.body.fillna("")
        frames.append(f[["article_id", "published_at", "title", "body", "url", "source", "compound"]])
    if not frames:
        return pd.DataFrame(columns=["article_id", "published_at", "title", "body", "url", "source", "compound"])
    return pd.concat(frames).drop_duplicates("article_id", keep="first")


def cached_project_prices(symbols, start, end, out):
    """Reuse authentic saved project observations where public mapping is unavailable."""
    if not symbols:
        return pd.DataFrame()
    paths = [ROOT / "stage_1_crypto_data.csv"]
    for revision in ["HEAD", "fb8cf67", "2e523b1", "b6cb437", "73d9be9"]:
        destination = out / "raw/original_snapshots" / f"{revision}_prices.csv"
        if not destination.exists():
            result = subprocess.run(["git", "show", f"{revision}:stage_1_crypto_data.csv"],
                                    cwd=ROOT, capture_output=True)
            if result.returncode:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(result.stdout)
        paths.append(destination)
    frames = []
    for path in paths:
        if not path.exists():
            continue
        f = pd.read_csv(path)
        if not {"symbol", "date", "close", "base_volume", "usd_volume"}.issubset(f.columns):
            continue
        f = f[f.symbol.isin(symbols)].copy()
        f["date"] = pd.to_datetime(f.date, format="mixed", utc=True).dt.tz_localize(None)
        f = f[(f.date >= start) & (f.date <= end) & (f.close > 0)]
        f["source"] = "coindesk_cached"
        f["source_symbol"] = f.symbol
        f["base_volume_estimated"] = False
        f["source_file"] = str(path.relative_to(ROOT))
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(["symbol", "date"], keep="first")


def wordpress_news(name, url, start, end, out):
    """Public publisher's own API; archive every page and preserve provenance."""
    posts, ranges = [], []
    month = start.replace(day=1)
    while month <= end:
        following = month + pd.offsets.MonthBegin(1)
        low, high = max(start, month), min(end + pd.Timedelta(days=1), following)
        page, total = 1, None
        while True:
            cache = out / "raw" / "news" / name / f"{low:%Y%m%d}_{high:%Y%m%d}_{page:03}.json"
            params = {"after": (low - pd.Timedelta(seconds=1)).isoformat(),
                      "before": high.isoformat(), "per_page": 100, "page": page,
                      "orderby": "date", "order": "asc",
                      "_fields": "id,date_gmt,link,title,content,excerpt"}
            try:
                obj, headers = get_json(url, params, cache)
            except Exception as exc:
                if page > 1 and "rest_post_invalid_page_number" in str(exc):
                    ranges.append({"source": name, "start": str(low), "end_exclusive": str(high),
                                   "status": "complete", "pages": page - 1})
                    break
                ranges.append({"source": name, "start": str(low), "end_exclusive": str(high),
                               "status": "error", "error": str(exc)})
                break
            if not isinstance(obj, list):
                raise RuntimeError(f"Unexpected publisher result: {name}")
            for post in obj:
                body = text_only(post.get("content", {}).get("rendered", ""))
                if not body:
                    body = text_only(post.get("excerpt", {}).get("rendered", ""))
                posts.append({"article_id": f"{name}:{post['id']}",
                              "published_at": post.get("date_gmt"),
                              "title": text_only(post.get("title", {}).get("rendered", "")),
                              "body": body, "url": post.get("link", ""), "source": name})
            if total is None and headers.get("X-WP-TotalPages"):
                total = int(headers["X-WP-TotalPages"])
            if len(obj) < 100 or (total is not None and page >= total):
                ranges.append({"source": name, "start": str(low), "end_exclusive": str(high),
                               "status": "complete", "pages": page})
                break
            page += 1
            time.sleep(0.15)
        print(f"NEWS {name} {month:%Y-%m} cumulative={len(posts)}", flush=True)
        month = following
    frame = pd.DataFrame(posts)
    if not frame.empty:
        frame["published_at"] = pd.to_datetime(frame.published_at, utc=True).dt.tz_localize(None)
        frame = frame[(frame.published_at >= start) & (frame.published_at < end + pd.Timedelta(days=1))]
    return frame, ranges


def coindesk_news(start, end, out, key):
    posts, cursor = [], int((end + pd.Timedelta(days=1)).timestamp()) - 1
    while cursor >= int(start.timestamp()):
        obj, _ = get_json("https://data-api.coindesk.com/news/v1/article/list",
                          {"lang": "EN", "to_ts": cursor, "limit": 100},
                          out / "raw/news/coindesk" / f"{cursor}.json", key)
        data = obj.get("Data")
        if not isinstance(data, list) or not data:
            break
        for p in data:
            posts.append({"article_id": f"coindesk:{p['ID']}",
                          "published_at": pd.Timestamp(p["PUBLISHED_ON"], unit="s"),
                          "title": p.get("TITLE", ""), "body": p.get("BODY", ""),
                          "url": p.get("URL", ""), "source": "coindesk"})
        nxt = min(p["PUBLISHED_ON"] for p in data) - 1
        if nxt >= cursor:
            raise RuntimeError("News cursor failed to move backwards")
        cursor = nxt
        if len(posts) % 1000 == 0:
            print(f"NEWS coindesk rows={len(posts)}", flush=True)
    return pd.DataFrame(posts)


def download(args):
    out, start, end = args.output, pd.Timestamp(args.start), pd.Timestamp(args.end)
    baseline_path = out / "snapshot/baseline_history_2025-07-30.csv"
    if not baseline_path.exists():
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        old = pd.read_csv(ROOT / "df_merged_history.csv", parse_dates=["date"])
        write_csv(baseline_path, old[old.date <= start])
    baseline = pd.read_csv(baseline_path, parse_dates=["date"])
    symbols = sorted(baseline[baseline.date == start].symbol.unique())
    if args.symbols:
        symbols = args.symbols.split(",")
    key = api_key()
    provider = args.price_source
    if provider == "auto":
        try:
            get_prices("BTC", end - pd.Timedelta(days=2), end, "coindesk", out, key)
            provider = "coindesk"
        except Exception as exc:
            print(f"ORIGINAL_SOURCE_UNAVAILABLE {exc}; using explicitly labelled Yahoo USD data", flush=True)
            provider = "yahoo"
    price_start = start - pd.Timedelta(days=109)
    catalog = yahoo_catalog(out) if provider == "yahoo" else []
    frames, coverage = [], []
    def task(symbol):
        try:
            code, reference_error = symbol, None
            if provider == "yahoo":
                f, code, reference_error = matched_yahoo_prices(
                    symbol, price_start, end, out, baseline, catalog)
            else:
                f = get_prices(symbol, price_start, end, provider, out, key)
            if f.empty:
                raise RuntimeError("Empty price history")
            return f, {"kind": "price", "symbol": symbol, "source": provider,
                       "source_symbol": code, "pre_start_reference_log_error": reference_error,
                       "status": "complete" if f.date.max() == end else "partial",
                       "rows": len(f), "first_date": str(f.date.min().date()),
                       "last_date": str(f.date.max().date()), "error": ""}
        except Exception as exc:
            return None, {"kind": "price", "symbol": symbol, "source": provider,
                          "status": "unavailable", "rows": 0, "error": str(exc)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (f, c) in enumerate(pool.map(task, symbols), 1):
            coverage.append(c)
            if f is not None:
                frames.append(f)
            if i % 10 == 0 or i == len(symbols):
                print(f"PRICES {i}/{len(symbols)} usable={len(frames)}", flush=True)
                write_csv(out / "data_coverage.csv", pd.DataFrame(coverage))
    downloaded_symbols = len(frames)
    missing_symbols = {c["symbol"] for c in coverage if c["status"] == "unavailable"}
    cached = cached_project_prices(missing_symbols, price_start, end, out)
    if not cached.empty:
        frames.append(cached)
        for c in coverage:
            local = cached[cached.symbol == c["symbol"]]
            if c["status"] == "unavailable" and not local.empty:
                c["public_source_error"] = c["error"]
                c.update(status="partial_cached", source="coindesk_cached",
                         source_symbol=c["symbol"], rows=len(local),
                         first_date=str(local.date.min().date()), last_date=str(local.date.max().date()),
                         error="Only existing project cache is available; gaps remain outside these dates.")
        print(f"REUSED_PROJECT_PRICE_CACHE symbols={cached.symbol.nunique()} rows={len(cached)}", flush=True)
    write_csv(out / "data_coverage.csv", pd.DataFrame(coverage))
    if not frames:
        raise RuntimeError("No usable price observations; cached errors retained")
    prices = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])
    write_csv(out / "raw_prices.csv", prices)
    write_csv(out / "symbol_mapping.csv", pd.DataFrame(coverage))
    news = local_news(out)
    news_source = args.news_source if args.news_source != "auto" else ("coindesk" if key else "public")
    ranges, fetched = [], []
    news_start = start + pd.Timedelta(days=1)
    if news_source == "coindesk":
        try:
            f = coindesk_news(news_start, end, out, key)
            if not f.empty:
                fetched.append(f)
            ranges.append({"source": "coindesk", "start": str(news_start),
                           "end_exclusive": str(end + pd.Timedelta(days=1)), "status": "complete"})
        except Exception as exc:
            print(f"NEWS_ORIGINAL_UNAVAILABLE {exc}; trying public publishers", flush=True)
            news_source = "public"
    if news_source == "public":
        for name, url in PUBLISHERS.items():
            f, r = wordpress_news(name, url, news_start, end, out)
            if not f.empty:
                fetched.append(f)
            ranges.extend(r)
    if fetched:
        news = pd.concat([news, *fetched], ignore_index=True).drop_duplicates("article_id", keep="last")
    news = news[(news.published_at >= start - pd.Timedelta(days=6)) &
                (news.published_at < end + pd.Timedelta(days=1))].copy()
    news["body"] = news.body.fillna("")
    if "compound" not in news:
        news["compound"] = np.nan
    write_csv(out / "news_articles.csv", news.sort_values("published_at"))
    write_csv(out / "news_fetch_coverage.csv", pd.DataFrame(ranges))
    aliases = {}
    for row in coverage:
        if row.get("source_symbol") and row.get("status") != "unavailable":
            aliases.setdefault(row["source_symbol"], []).append(row["symbol"])
    write_json(out / "preparation_manifest.json", {
        "start": args.start, "end": args.end, "price_source": provider,
        "news_source": news_source, "universe": symbols,
        "universe_policy": "fixed starting-date project universe; not reconstructed weekly top-200",
        "shared_source_aliases": {k: v for k, v in aliases.items() if len(v) > 1},
        "base_volume_policy": "Yahoo quote volume / close estimate" if provider == "yahoo" else "source native base volume",
        "price_rows": len(prices), "price_symbols": prices.symbol.nunique(), "news_rows": len(news),
        "downloaded_price_symbols": downloaded_symbols,
        "complete_price_symbols": sum(c["status"] == "complete" for c in coverage),
        "cached_only_price_symbols": int(cached.symbol.nunique()) if not cached.empty else 0,
        "downloaded_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "legacy_source_sha256": digest(ROOT / "crypto_project_pipeline.py"),
        "baseline_sha256": digest(baseline_path), "price_window_days": 110,
        "source_limitations": [
            "Original CoinDesk endpoints require an API key when anonymous access returns 401.",
            "Public fallback inputs differ from CoinDesk CADLI and its news sample.",
            "Historical top-list membership was not archived; fixed starting universe is explicit.",
            "Yahoo native base volume is unavailable and is estimated from quote volume and close.",
            "Ticker identities are checked against at least three pre-start project weekly closes; unresolved homonyms are excluded.",
            "Historical articles are current archive copies, not preserved publication-time text snapshots; later edits may be unrecoverable.",
            "Unavailable public quotes reuse original project price archives only on their observed dates; other dates stay missing.",
        ] if provider == "yahoo" else [],
    })
    print(f"DOWNLOAD_COMPLETE prices={len(prices)} news={len(news)}", flush=True)


def build_features(args):
    out, start, end = args.output, pd.Timestamp(args.start), pd.Timestamp(args.end)
    prices = pd.read_csv(out / "raw_prices.csv")
    prices["date"] = pd.to_datetime(prices.date, format="mixed", utc=True).dt.tz_localize(None)
    non_daily = ~prices.date.eq(prices.date.dt.normalize())
    if non_daily.any():
        write_csv(out / "excluded_intraday_quotes.csv", prices[non_daily])
        prices = prices[~non_daily].copy()
    write_csv(out / "raw_prices.csv", prices)
    news = pd.read_csv(out / "news_articles.csv", low_memory=False)
    news["published_at"] = pd.to_datetime(news.published_at, format="mixed", utc=True).dt.tz_localize(None)
    news["body"] = news.body.fillna("")
    news["compound"] = pd.to_numeric(news.compound, errors="coerce")
    score_cache_path = out / "sentiment_scores.csv"
    if score_cache_path.exists():
        cached = pd.read_csv(score_cache_path).set_index("article_id").compound
        news["compound"] = news.compound.fillna(news.article_id.map(cached))
    pending = news.compound.isna()
    if pending.any():
        scorer = load_sentiment_tools(out)
        for i, idx in enumerate(news.index[pending], 1):
            news.loc[idx, "compound"] = scorer(news.loc[idx, "body"])
            if i % 500 == 0:
                print(f"SENTIMENT {i}/{pending.sum()}", flush=True)
                write_csv(score_cache_path, news[["article_id", "compound"]].dropna())
        write_csv(score_cache_path, news[["article_id", "compound"]].dropna())
    write_csv(out / "news_articles.csv", news)
    fn = legacy_market_function()
    packs, states = [], []
    for decision in pd.date_range(start + pd.Timedelta(days=7), end, freq="W-WED"):
        package = out / "features" / f"{decision:%Y-%m-%d}.csv"
        market = weekly_market(prices, decision, fn)
        if market.empty:
            states.append({"decision_date": str(decision.date()), "status": "missing_prices"})
            continue
        low, high = decision - pd.Timedelta(days=6), decision + pd.Timedelta(days=1)
        original = news[(news.published_at >= low) & (news.published_at < high) &
                        news.source.isin(["coindesk", "coindesk_cached"])]
        original_days = original.published_at.dt.normalize().nunique()
        chosen = original if original_days == 7 else news[~news.source.isin(["coindesk", "coindesk_cached"])]
        values, count = weekly_sentiment(chosen, decision)
        week_news = chosen[(chosen.published_at >= low) & (chosen.published_at < high)]
        days = week_news.published_at.dt.normalize().nunique()
        available = bool(count and days == 7)
        for c in SENTIMENT_FEATURES:
            market[c] = values[c] if available else np.nan
        market["sentiment_available"] = available
        market["news_count"] = count
        market["news_days"] = days
        market["sentiment_source"] = "|".join(sorted(week_news.source.unique()))
        market["data_cutoff_utc"] = high.isoformat() + "Z (exclusive)"
        write_csv(package, market)
        packs.append(market)
        state = {"decision_date": str(decision.date()), "status": "complete" if available else "partial_news",
                 "symbols": len(market), "news_count": count, "news_days": days,
                 "sentiment_source": market.sentiment_source.iloc[0],
                 "sentiment_q10": values.get("sentiment_q10"), "sentiment_q90": values.get("sentiment_q90")}
        states.append(state)
        print(f"FEATURES {decision:%Y-%m-%d} symbols={len(market)} news={count} days={days}", flush=True)
    if packs:
        write_csv(out / "weekly_feature_history.csv", pd.concat(packs, ignore_index=True))
    write_csv(out / "weekly_input_status.csv", pd.DataFrame(states))
    manifest_path = out / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(price_rows=len(prices), excluded_intraday_quotes=int(non_daily.sum()))
    if manifest.get("news_source") == "public":
        note = "Missing original news weeks use public publisher samples; not the original CoinDesk news universe."
        if note not in manifest["source_limitations"]:
            manifest["source_limitations"].append(note)
    write_json(manifest_path, manifest)
    print("FEATURE_PREPARATION_COMPLETE", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2025-07-30")
    p.add_argument("--end", default="2026-11-11")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--price-source", choices=["auto", "coindesk", "yahoo"], default="auto")
    p.add_argument("--news-source", choices=["auto", "coindesk", "public"], default="auto")
    p.add_argument("--stage", choices=["all", "download", "features"], default="all")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--symbols", help="Optional comma-separated subset for a small dry run")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if pd.Timestamp(args.start).dayofweek != 2 or pd.Timestamp(args.end).dayofweek != 2:
        p.error("Start and end must be Wednesdays")
    if args.stage in ["all", "download"]:
        download(args)
    if args.stage in ["all", "features"]:
        build_features(args)


if __name__ == "__main__":
    main()
