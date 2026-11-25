"""
Prediction Service
Handles loading predictions and symbol mapping for TradingView
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import re
from datetime import date, timedelta
from web.services.weekly_updates import get_updates
from web.services.market_catalog import read_catalog, resolve

# Add parent directory for imports
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Model configurations
MODELS = {
    "enet_all": {"name": "ElasticNet (All Features)", "file": "pred_enet_all.csv"},
    "enet_market": {"name": "ElasticNet (Market Only)", "file": "pred_enet_market.csv"},
    "extra_all": {"name": "ExtraTrees (All Features)", "file": "pred_extra_all.csv"},
    "extra_market": {"name": "ExtraTrees (Market Only)", "file": "pred_extra_market.csv"}
}

def get_available_models() -> Dict:
    """Get list of available prediction models"""
    return MODELS


def map_symbol_to_tradingview(symbol: str, catalog=None) -> Optional[str]:
    """Convert prediction symbol to TradingView format"""
    symbol = symbol.upper().strip()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]{0,29}", symbol):
        return None

    pair = resolve(symbol, catalog if catalog is not None else read_catalog())
    return pair["tradingview"] if pair else None


def load_predictions(model_key: str) -> Optional[Dict]:
    """
    Load predictions for a model

    Returns:
        {
            "model": model_key,
            "model_name": "Model Display Name",
            "top_20": [{"symbol": "BTC", "tradingview": "BINANCE:BTCUSDT", ...}],
            "bottom_20": [{"symbol": "XYZ", "tradingview": "BINANCE:XYZUSDT", ...}]
        }
    """
    if model_key not in MODELS:
        return None

    model_info = MODELS[model_key]
    publication = get_updates().published()
    # Read data AND its date from the same publication, even if an update
    # finishes while this request is being served. Never label a legacy fallback
    # file as belonging to a newer publication.
    pred_file = (publication["path"] / model_info["file"] if publication
                 else BASE_DIR / model_info["file"])

    if not pred_file.exists():
        return None

    try:
        df = pd.read_csv(pred_file, float_precision="round_trip")

        # Assume columns: symbol, predicted_return (or similar)
        if 'symbol' not in df.columns:
            return None

        # Metadata columns must never accidentally become the ranking score.
        value_col = next((c for c in ["y_pred", "predicted_return", "value"] if c in df), None)
        if value_col is None or df.symbol.isna().any() or df.symbol.duplicated().any():
            return None
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        if not np.isfinite(df[value_col]).all():
            return None
        df_sorted = df.sort_values(by=value_col, ascending=False, kind="stable").copy()
        df_sorted["predicted_rank"] = np.arange(1, len(df_sorted)+1)

        # Get top 20 and bottom 20
        top_20 = df_sorted.head(20)
        bottom_20 = df_sorted.tail(20)

        catalog = read_catalog()
        catalog_ready = any(entry.get("pairs") for entry in catalog.get("providers", {}).values())

        def format_rows(df_rows):
            result = []
            for _, row in df_rows.iterrows():
                symbol = row['symbol']
                pair = resolve(symbol, catalog)
                tv_symbol = pair["tradingview"] if pair else None
                result.append({
                    "symbol": symbol,
                    "tradingview": tv_symbol,
                    "value": float(row[value_col]),
                    "rank": int(row["predicted_rank"]),
                    "chart_available": tv_symbol is not None,
                    "chart_unavailable_reason": (None if tv_symbol else "当前未确认可用交易对，预测仍可查看"),
                    "chart_provider": pair["chart_provider"] if pair else None,
                    "quote_currency": pair["quote_currency"] if pair else None
                })
            return result

        period = publication["period"] if publication else None
        decision = date.fromisoformat(period) if period else None
        return {
            "model": model_key,
            "model_name": model_info["name"],
            "top_20": format_rows(top_20),
            "bottom_20": format_rows(bottom_20),
            "data_period": period,
            "forecast_start": (decision+timedelta(days=1)).isoformat() if decision else None,
            "forecast_end": (decision+timedelta(days=7)).isoformat() if decision else None,
            "generated_at": publication["manifest"].get("generated_at") if publication else None,
            "data_version": publication["path"].name if publication else None,
            "universe_count": len(df_sorted),
            "value_unit": "decimal_return",
            "chart_catalog_ready": catalog_ready
        }
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None
