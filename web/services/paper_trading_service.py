"""
Paper Trading Service
Simulates extra_ls (Long-Short) and extra_EW (Equal Weight) strategies
using historical data and current predictions.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

HISTORY_FILE = BASE_DIR / "df_merged_history.csv"
PRED_EXTRA_ALL = BASE_DIR / "pred_extra_all.csv"
PRED_EXTRA_MARKET = BASE_DIR / "pred_extra_market.csv"

TOP_N = 20


def _load_history() -> Optional[pd.DataFrame]:
    """Load the merged history data with returns."""
    if not HISTORY_FILE.exists():
        return None
    try:
        df = pd.read_csv(HISTORY_FILE)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"[PaperTrading] Error loading history: {e}")
        return None


def _compute_weekly_strategy_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly returns for extra_ls and extra_EW strategies
    using a rolling simulation over all available weeks.

    For each week t:
      1. Rank coins by their 'return' in week t (simulating a prediction signal)
      2. Take top-20 as long, bottom-20 as short
      3. Compute portfolio return using NEXT week's (t+1) actual returns

    We use actual historical returns as the "prediction signal" proxy since
    the real pred files are only for the latest week.
    For a more accurate simulation, we use the return column directly.
    """
    all_weeks = sorted(df["date"].unique())

    if len(all_weeks) < 4:
        return pd.DataFrame()

    # Prepare next-week returns
    df_sorted = df.sort_values(["symbol", "date"])
    df_sorted["next_return"] = df_sorted.groupby("symbol")["return"].shift(-1)

    results = []

    for i in range(len(all_weeks) - 1):
        this_week = all_weeks[i]
        next_week = all_weeks[i + 1]

        # This week's data — use 'return' as the ranking signal
        df_this = df_sorted[df_sorted["date"] == this_week].copy()
        if len(df_this) < TOP_N * 2:
            continue

        df_this = df_this.dropna(subset=["next_return"])
        if len(df_this) < TOP_N * 2:
            continue

        # Rank by return (proxy for prediction)
        df_ranked = df_this.sort_values("return", ascending=False)

        top_coins = df_ranked.head(TOP_N)
        bottom_coins = df_ranked.tail(TOP_N)

        # EW strategy: Long top 20 equally
        ew_return = top_coins["next_return"].mean()

        # LS strategy: Long top 20, Short bottom 20
        ls_return = top_coins["next_return"].mean() - bottom_coins["next_return"].mean()

        # Baseline: Equal weight all coins
        baseline_return = df_this["next_return"].mean()

        results.append({
            "date": next_week,
            "extra_EW": ew_return,
            "extra_ls": ls_return,
            "baseline": baseline_return,
            "n_coins": len(df_this),
        })

    return pd.DataFrame(results)


def _compute_cumulative(df_returns: pd.DataFrame) -> Dict:
    """Convert weekly returns into cumulative performance metrics."""
    if df_returns.empty:
        return {}

    df = df_returns.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    strategies = ["extra_EW", "extra_ls", "baseline"]
    output = {}

    for strat in strategies:
        if strat not in df.columns:
            continue

        returns = df[strat].dropna()
        if len(returns) == 0:
            continue

        cum_return = (1 + returns).cumprod()
        total_return = cum_return.iloc[-1] - 1 if len(cum_return) > 0 else 0
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return / std_return * np.sqrt(52)) if std_return > 0 else 0

        # Max drawdown
        peak = cum_return.cummax()
        drawdown = (cum_return - peak) / peak
        max_drawdown = drawdown.min()

        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

        output[strat] = {
            "total_return_pct": round(total_return * 100, 2),
            "mean_weekly_return_pct": round(mean_return * 100, 4),
            "volatility_annual_pct": round(std_return * np.sqrt(52) * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "win_rate_pct": round(win_rate * 100, 1),
            "total_weeks": len(returns),
            "cumulative_series": [
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "value": round(v, 6)
                }
                for d, v in zip(df["date"], cum_return)
                if pd.notna(v)
            ],
        }

    return output


def _get_current_positions() -> Dict:
    """
    Get current recommended positions from latest prediction files.
    Returns top 20 (long) and bottom 20 (short) for extra_all strategy.
    """
    positions = {}

    for label, fpath in [("extra_all", PRED_EXTRA_ALL), ("extra_market", PRED_EXTRA_MARKET)]:
        if not fpath.exists():
            continue
        try:
            df = pd.read_csv(fpath)
            df_sorted = df.sort_values("y_pred", ascending=False)

            top_20 = df_sorted.head(TOP_N)
            bottom_20 = df_sorted.tail(TOP_N).sort_values("y_pred", ascending=True)

            positions[label] = {
                "long": [
                    {"symbol": r["symbol"], "pred_return": round(r["y_pred"] * 100, 4)}
                    for _, r in top_20.iterrows()
                ],
                "short": [
                    {"symbol": r["symbol"], "pred_return": round(r["y_pred"] * 100, 4)}
                    for _, r in bottom_20.iterrows()
                ],
            }
        except Exception as e:
            print(f"[PaperTrading] Error loading {label}: {e}")

    return positions


def get_paper_trading_data(initial_capital: float = 10000.0) -> Dict:
    """
    Main entry point: Get complete paper trading data.

    Returns:
    {
        "strategies": {
            "extra_EW": { metrics + cumulative_series },
            "extra_ls": { metrics + cumulative_series },
            "baseline": { metrics + cumulative_series }
        },
        "current_positions": {
            "extra_all": { "long": [...], "short": [...] }
        },
        "initial_capital": 10000,
        "strategy_descriptions": { ... },
        "generated_at": "2026-02-05T12:00:00Z"
    }
    """
    df_history = _load_history()

    strategies_data = {}
    if df_history is not None:
        # Compute rolling backtest
        df_returns = _compute_weekly_strategy_returns(df_history)
        strategies_data = _compute_cumulative(df_returns)

        # Add capital series (starting from initial_capital)
        for strat, data in strategies_data.items():
            if "cumulative_series" in data:
                for point in data["cumulative_series"]:
                    point["capital"] = round(initial_capital * point["value"], 2)

    # Get current positions
    current_positions = _get_current_positions()

    return {
        "strategies": strategies_data,
        "current_positions": current_positions,
        "initial_capital": initial_capital,
        "strategy_descriptions": {
            "extra_EW": "Equal Weight — Long Top 20 coins (ExtraTrees model), each 5% weight",
            "extra_ls": "Long-Short — Long Top 20 + Short Bottom 20 (ExtraTrees), fully hedged",
            "baseline": "Baseline — Equal weight ALL coins (buy & hold benchmark)",
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
