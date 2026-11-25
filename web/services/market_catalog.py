"""Cache public exchange instrument directories, never synthesize a listing.

Only unauthenticated, full public directories are requested. No predictions,
historical features, credentials or chat messages are sent to either exchange.
"""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import threading
import time

import requests

from web.services.weekly_updates import get_updates

ALIASES = {"MANTLE":"MNT", "MIOTA":"IOTA", "RONIN":"RON", "XBT":"BTC", "XDG":"DOGE", "BTT":"BTTC"}
DIRECTORIES = {
    "BINANCE": "https://data-api.binance.vision/api/v3/exchangeInfo",
    "COINBASE": "https://api.exchange.coinbase.com/products",
}
TTL = 6*3600
RETRY = 900


def catalog_file():
    return get_updates().root / "market_catalog.json"


def read_catalog():
    try:
        data = json.loads(catalog_file().read_text())
        return data if data.get("schema") == 1 else {"providers": {}}
    except (FileNotFoundError, ValueError):
        return {"providers": {}}


def parse_pairs(provider, payload):
    pairs = {}
    if provider == "BINANCE":
        if not isinstance(payload, dict) or not isinstance(payload.get("symbols"), list):
            raise ValueError("Invalid exchange instrument directory")
        for row in payload["symbols"]:
            if row.get("status") == "TRADING" and row.get("isSpotTradingAllowed", True) and row.get("quoteAsset") == "USDT":
                pairs[row["baseAsset"]] = {"symbol": row["symbol"], "quote": "USDT"}
    else:
        if not isinstance(payload, list):
            raise ValueError("Invalid exchange instrument directory")
        for row in payload:
            if row.get("status") == "online" and not row.get("trading_disabled", False) and row.get("quote_currency") == "USD":
                pairs[row["base_currency"]] = {"symbol": row["id"].replace("-", ""), "quote": "USD"}
    if not pairs:
        raise ValueError("Empty exchange directory; keep previous cache")
    return pairs


def resolve(symbol, catalog=None):
    symbol = str(symbol).strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,29}", symbol):
        return None
    base = ALIASES.get(symbol, symbol)
    catalog = catalog if catalog is not None else read_catalog()
    for provider in DIRECTORIES:
        entry = catalog.get("providers", {}).get(provider, {})
        if time.time()-entry.get("updated_at", 0) > 48*3600:
            continue
        pair = entry.get("pairs", {}).get(base)
        if pair:
            return {"tradingview": f"{provider}:{pair['symbol']}", "chart_provider": provider,
                    "quote_currency": pair["quote"], "catalog_updated_at": entry.get("updated_at")}
    return None


def refresh_catalog(force=False):
    path = catalog_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return read_catalog()
        data = read_catalog()
        providers = data.setdefault('providers', {})
        now = time.time()
        for provider, url in DIRECTORIES.items():
            old = providers.get(provider, {})
            if not force and (now-old.get('checked_at', 0) < RETRY or now-old.get('updated_at', 0) < TTL):
                continue
            try:
                response = requests.get(url, timeout=12, headers={'User-Agent':'AnselCryptoMarket/1.0'})
                response.raise_for_status()
                pairs = parse_pairs(provider, response.json())
                providers[provider] = {'pairs':pairs, 'updated_at':now, 'checked_at':now, 'source':url, 'status':'ready'}
            except (requests.RequestException, ValueError, KeyError, TypeError):
                providers[provider] = {**old, 'checked_at':now, 'source':url, 'status':'unavailable'}
        data['schema'] = 1
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False))
        temporary.replace(path)
        return data


class CatalogRefresher:
    def __init__(self):
        self.stop = threading.Event()
        self.thread = None

    def start(self):
        def loop():
            while not self.stop.is_set():
                refresh_catalog()
                self.stop.wait(900)
        self.thread = threading.Thread(target=loop, name='public-market-catalog', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=1)
