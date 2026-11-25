"""
AI Market Analysis Module (Enhanced)
Generates comprehensive weekly crypto market analysis reports using Claude Opus.

Data Sources Used:
  1. Price Trends    - df_merged_history.csv (OHLCV + technical indicators)
  2. Sentiment       - stage_2_sentiment_weekly.csv (VADER compound scores)
  3. Wordcloud Data  - stage_1_common_words.csv (hot keyword frequencies)
  4. Model Predictions - pred_*.csv (ElasticNet & ExtraTrees forecasts)
  5. News Headlines  - stage_1_news_raw.csv (CoinDesk headlines)
  6. Skills KB       - skills/*.md (39 crypto knowledge modules)

This enhanced version feeds ALL available project data into the Claude prompt,
producing a deeply data-driven, skill-backed analysis report.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Tuple
import pandas as pd
import anthropic

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent
REPORT_FILE = BASE_DIR / "figures" / "weekly_report.md"
REPORT_UPDATE_FILE = BASE_DIR / "last_updated_report.txt"
HISTORY_FILE = BASE_DIR / "df_merged_history.csv"

# === Additional Data Paths ===
SENTIMENT_WEEKLY_FILE = BASE_DIR / "figures" / "stage_2_sentiment_weekly.csv"
SENTIMENT_WEEKLY_FULL = BASE_DIR / "stage_2_sentiment_weekly.csv"
COMMON_WORDS_FILE = BASE_DIR / "figures" / "stage_1_common_words.csv"
COMMON_WORDS_FULL = BASE_DIR / "stage_1_common_words.csv"
NEWS_RAW_FILE = BASE_DIR / "figures" / "stage_1_news_raw.csv"
NEWS_RAW_FULL = BASE_DIR / "stage_1_news_raw.csv"
SKILLS_DIR = BASE_DIR / "skills"
PRED_FILES = {
    "enet_all": BASE_DIR / "pred_enet_all.csv",
    "enet_market": BASE_DIR / "pred_enet_market.csv",
    "extra_all": BASE_DIR / "pred_extra_all.csv",
    "extra_market": BASE_DIR / "pred_extra_market.csv",
}


def get_latest_wednesday(dt: datetime) -> datetime:
    """Get the most recent Wednesday (UTC)"""
    days_since_wed = (dt.weekday() - 2) % 7
    return dt - timedelta(days=days_since_wed)


def should_update_report() -> Tuple[bool, str]:
    """
    Check if the report should be updated (every Wednesday UTC)
    Returns: (should_update, reason)
    """
    now = datetime.now(timezone.utc)
    this_wednesday = get_latest_wednesday(now)

    # Check if report exists
    if not REPORT_FILE.exists():
        return True, "Report file does not exist"

    # Check last update time
    if not REPORT_UPDATE_FILE.exists():
        return True, "Update timestamp file does not exist"

    try:
        last_update_str = REPORT_UPDATE_FILE.read_text().strip()
        last_update = datetime.strptime(last_update_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

        # If last update is before this Wednesday, need to update
        if last_update.date() < this_wednesday.date():
            return True, f"Last update ({last_update.date()}) is before this Wednesday ({this_wednesday.date()})"

        return False, f"Report is up to date (last updated: {last_update.date()})"
    except Exception as e:
        return True, f"Error reading update timestamp: {e}"


# =========================================================================
# DATA LOADERS — Each function reads one data source for the prompt
# =========================================================================

def load_recent_price_data(weeks: int = 4) -> Optional[pd.DataFrame]:
    """
    Load recent price data for Top 20 coins from history.
    Extended to 4 weeks for richer trend context.
    """
    if not HISTORY_FILE.exists():
        return None

    try:
        df = pd.read_csv(HISTORY_FILE)
        df['date'] = pd.to_datetime(df['date'])
        max_date = df['date'].max()
        cutoff_date = max_date - timedelta(weeks=weeks)
        df_recent = df[df['date'] >= cutoff_date].copy()

        # Get top 20 coins by average volume
        top_coins = df_recent.groupby('symbol')['usd_volume'].mean().nlargest(20).index.tolist()
        df_top = df_recent[df_recent['symbol'].isin(top_coins)]
        return df_top
    except Exception as e:
        print(f"[DataLoader] Error loading price data: {e}")
        return None


def format_price_summary(df: pd.DataFrame) -> str:
    """Format price data into a detailed summary string with multi-week trends."""
    if df is None or df.empty:
        return "No price data available."

    summary_lines = []

    for symbol in df['symbol'].unique():
        df_sym = df[df['symbol'] == symbol].sort_values('date')
        if len(df_sym) >= 2:
            latest = df_sym.iloc[-1]
            prev = df_sym.iloc[-2]

            weekly_return = ((latest['close'] - prev['close']) / prev['close'] * 100) if prev['close'] > 0 else 0

            # Calculate 4-week trend if available
            if len(df_sym) >= 4:
                four_weeks_ago = df_sym.iloc[-4]
                monthly_return = ((latest['close'] - four_weeks_ago['close']) / four_weeks_ago['close'] * 100) if four_weeks_ago['close'] > 0 else 0
                trend_str = f"4W: {monthly_return:+.2f}%"
            else:
                trend_str = "4W: N/A"

            # Volatility from high/low
            if 'high' in latest.index and 'low' in latest.index and latest['low'] > 0:
                week_range = ((latest['high'] - latest['low']) / latest['low'] * 100)
                vol_str = f"Range: {week_range:.1f}%"
            else:
                vol_str = ""

            summary_lines.append(
                f"- {symbol}: ${latest['close']:.4f} | 1W: {weekly_return:+.2f}% | {trend_str} | "
                f"Vol: ${latest['usd_volume']/1e6:.1f}M | {vol_str}"
            )

    return "\n".join(summary_lines[:20])


def load_technical_indicators(df: pd.DataFrame) -> str:
    """Extract technical indicators (momentum, volatility, volume shock) from the data."""
    if df is None or df.empty:
        return "No technical indicator data available."

    lines = []
    # Detect available indicator columns
    indicator_cols = [c for c in df.columns if any(
        kw in c for kw in ['momentum', 'volatility', 'volume_shock', 'long_candle', 'reversal']
    )]

    if not indicator_cols:
        return "No technical indicator columns found in data."

    for symbol in df['symbol'].unique()[:10]:  # Top 10 for brevity
        df_sym = df[df['symbol'] == symbol].sort_values('date')
        if len(df_sym) == 0:
            continue
        latest = df_sym.iloc[-1]

        vals = []
        for col in indicator_cols[:6]:  # Up to 6 indicators per coin
            val = latest.get(col, None)
            if pd.notna(val):
                short_name = col.replace('_', ' ').title()
                vals.append(f"{short_name}: {val:.4f}")

        if vals:
            lines.append(f"- {symbol}: {' | '.join(vals)}")

    return "\n".join(lines) if lines else "Technical indicators not available."


def load_sentiment_data() -> str:
    """
    Load sentiment weekly data (VADER compound, extreme ratios).
    Tries figures/ first, then root directory.
    """
    for fpath in [SENTIMENT_WEEKLY_FILE, SENTIMENT_WEEKLY_FULL]:
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                if df.empty:
                    continue

                lines = ["**Sentiment Trend (Recent Weeks):**"]
                lines.append("| Date | Compound Score | Extreme Positive % | Extreme Negative % | Mood |")
                lines.append("|------|---------------|--------------------|--------------------|------|")

                # Show last 6 weeks for trend
                for _, row in df.tail(6).iterrows():
                    compound = row.get('compound_weekly', 0)
                    pos_ratio = row.get('extremely_positive_ratio', 0) * 100
                    neg_ratio = row.get('extremely_negative_ratio', 0) * 100

                    # Determine mood
                    score = (compound + 1.0) * 50.0
                    if score < 20:
                        mood = "😱 Extreme Fear"
                    elif score < 40:
                        mood = "😟 Fear"
                    elif score < 60:
                        mood = "😐 Neutral"
                    elif score < 80:
                        mood = "😊 Greed"
                    else:
                        mood = "🤑 Extreme Greed"

                    lines.append(
                        f"| {row.get('date', 'N/A')} | {compound:.4f} | {pos_ratio:.1f}% | {neg_ratio:.1f}% | {mood} |"
                    )

                # Latest summary
                latest = df.iloc[-1]
                latest_score = (latest.get('compound_weekly', 0) + 1.0) * 50.0
                lines.append(f"\n**Current Fear & Greed Score: {latest_score:.0f}/100**")

                return "\n".join(lines)
            except Exception as e:
                print(f"[DataLoader] Error reading sentiment data from {fpath}: {e}")
                continue

    return "No sentiment data available."


def load_wordcloud_keywords() -> str:
    """
    Load top keywords from the wordcloud / common words data.
    This reflects what the crypto community is talking about.
    """
    for fpath in [COMMON_WORDS_FILE, COMMON_WORDS_FULL]:
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                if df.empty:
                    continue

                lines = ["**Top 30 Hot Keywords from News (Word Frequency):**"]
                for _, row in df.head(30).iterrows():
                    word = row.get('word', '')
                    count = row.get('count', 0)
                    lines.append(f"- **{word}**: {count} mentions")

                # Categorize keywords
                categories = {
                    "Market/Trading": ["market", "price", "trading", "exchange", "volume", "trade", "rally", "bearish", "bullish"],
                    "Technology": ["blockchain", "protocol", "network", "platform", "technology", "smart", "contract", "defi"],
                    "Regulation": ["regulatory", "regulation", "sec", "government", "compliance", "legal", "law"],
                    "Sentiment": ["risk", "growth", "fear", "bullish", "bearish", "uncertainty", "confidence"],
                    "Major Coins": ["bitcoin", "ethereum", "btc", "eth", "crypto", "cryptocurrency", "token", "digital"],
                }

                lines.append("\n**Keyword Category Breakdown:**")
                words_set = set(df['word'].str.lower().tolist()[:50])
                for cat, keywords in categories.items():
                    matched = [w for w in keywords if w in words_set]
                    if matched:
                        total_count = sum(
                            df[df['word'].str.lower() == w]['count'].sum() for w in matched
                        )
                        lines.append(f"- {cat}: {len(matched)} keywords, ~{total_count} total mentions ({', '.join(matched)})")

                return "\n".join(lines)
            except Exception as e:
                print(f"[DataLoader] Error reading wordcloud keywords from {fpath}: {e}")
                continue

    return "No wordcloud keyword data available."


def load_recent_news_headlines(max_headlines: int = 20) -> str:
    """
    Load recent news headlines from raw news data.
    These provide context on what's happening in the market.
    """
    for fpath in [NEWS_RAW_FILE, NEWS_RAW_FULL]:
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                if df.empty:
                    continue

                # Detect headline/title column
                title_col = None
                for col in ['title', 'headline', 'Title', 'Headline', 'name']:
                    if col in df.columns:
                        title_col = col
                        break

                if title_col is None:
                    # Fallback: use first text-like column
                    for col in df.columns:
                        if df[col].dtype == 'object' and col != 'body' and col != 'date':
                            title_col = col
                            break

                if title_col is None:
                    return "News data found but no title column detected."

                # Sort by date if available
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.sort_values('date', ascending=False)

                lines = ["**Recent Crypto News Headlines:**"]
                seen = set()
                count = 0
                for _, row in df.iterrows():
                    title = str(row.get(title_col, '')).strip()
                    if title and title not in seen and len(title) > 10:
                        date_str = str(row.get('date', ''))[:10] if 'date' in row.index else ''
                        lines.append(f"- [{date_str}] {title}")
                        seen.add(title)
                        count += 1
                        if count >= max_headlines:
                            break

                return "\n".join(lines)
            except Exception as e:
                print(f"[DataLoader] Error reading news headlines from {fpath}: {e}")
                continue

    return "No recent news data available."


def load_model_predictions() -> str:
    """
    Load prediction results from all 4 models.
    Shows top-ranked and bottom-ranked coins across models.
    """
    lines = ["**Model Prediction Summary (4 Models):**\n"]
    models_loaded = 0

    for model_name, fpath in PRED_FILES.items():
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                if df.empty:
                    continue

                models_loaded += 1
                df = df.sort_values('y_pred', ascending=False)

                top5 = df.head(5)
                bottom5 = df.tail(5).sort_values('y_pred', ascending=True)

                lines.append(f"**{model_name.upper()}** (Algorithm: {'ElasticNet' if 'enet' in model_name else 'ExtraTrees'}, "
                             f"Features: {'Market+Sentiment' if 'all' in model_name else 'Market Only'}):")
                lines.append(f"  🟢 Top 5 (Buy): {', '.join(f'{r.symbol}({r.y_pred:+.4f})' for _, r in top5.iterrows())}")
                lines.append(f"  🔴 Bottom 5 (Sell): {', '.join(f'{r.symbol}({r.y_pred:+.4f})' for _, r in bottom5.iterrows())}")
                lines.append("")
            except Exception as e:
                print(f"[DataLoader] Error reading predictions from {fpath}: {e}")

    if models_loaded == 0:
        return "No prediction data available."

    # Cross-model consensus
    try:
        all_tops = {}
        all_bottoms = {}
        for model_name, fpath in PRED_FILES.items():
            if fpath.exists():
                df = pd.read_csv(fpath)
                if df.empty:
                    continue
                df_sorted = df.sort_values('y_pred', ascending=False)
                for sym in df_sorted.head(10)['symbol']:
                    all_tops[sym] = all_tops.get(sym, 0) + 1
                for sym in df_sorted.tail(10)['symbol']:
                    all_bottoms[sym] = all_bottoms.get(sym, 0) + 1

        consensus_buy = [f"{s}({c}/4)" for s, c in sorted(all_tops.items(), key=lambda x: -x[1]) if c >= 2][:5]
        consensus_sell = [f"{s}({c}/4)" for s, c in sorted(all_bottoms.items(), key=lambda x: -x[1]) if c >= 2][:5]

        if consensus_buy:
            lines.append(f"**🎯 Cross-Model Consensus Buy (≥2/4 models agree):** {', '.join(consensus_buy)}")
        if consensus_sell:
            lines.append(f"**⚠️ Cross-Model Consensus Sell (≥2/4 models agree):** {', '.join(consensus_sell)}")
    except Exception:
        pass

    return "\n".join(lines)


def load_relevant_skills() -> str:
    """
    Load key skills knowledge for market analysis context.
    Selects the most relevant skills for weekly report analysis.
    """
    relevant_skill_names = [
        "bitcoin-market-analysis-and-trading-strategy",
        "bitcoin-market-cycle-analysis",
        "cryptocurrency-investment-strategies",
        "cryptocurrency-risk-assessment",
        "cryptocurrency-valuation-and-price-determinants",
        "bull-market-trading-strategy",
        "historical-crypto-market-timeline",
    ]

    skill_contents = []

    if not SKILLS_DIR.exists():
        return "No skills knowledge base available."

    loaded_count = 0
    for skill_name in relevant_skill_names:
        skill_file = SKILLS_DIR / skill_name / "SKILL.md"
        if skill_file.exists():
            try:
                content = skill_file.read_text(encoding='utf-8')
                # Keep full content (up to 4000 chars) so Claude can deeply apply each skill
                if len(content) > 4000:
                    content = content[:4000] + "\n... [truncated]"
                skill_contents.append(f"### Skill: {skill_name}\n{content}")
                loaded_count += 1
            except Exception:
                continue

    if not skill_contents:
        return "No relevant skills loaded."

    header = f"**Knowledge Base ({loaded_count} skills loaded):**\n"
    return header + "\n\n---\n\n".join(skill_contents)


# =========================================================================
# REPORT GENERATION — The main orchestration function
# =========================================================================

def generate_analysis_report(api_key: str, force_update: bool = False) -> Tuple[str, bool, str]:
    """
    Generate a comprehensive market analysis report using Claude Opus.

    Enhanced version that feeds ALL project data:
      - Price trends (4 weeks)
      - Technical indicators (momentum, volatility, volume shock)
      - Sentiment data (fear/greed index, compound scores)
      - Wordcloud keywords (hot topics)
      - News headlines (recent events)
      - Model predictions (4-model consensus)
      - Skills knowledge base (expert analysis framework)

    Args:
        api_key: Anthropic API key
        force_update: Force regeneration even if report is current

    Returns:
        (report_content, was_generated, message)
    """
    # Check if update needed
    should_update, reason = should_update_report()

    if not force_update and not should_update:
        if REPORT_FILE.exists():
            return REPORT_FILE.read_text(encoding='utf-8'), False, reason
        else:
            return "No report available.", False, "Report file not found"

    # === STEP 1: Load ALL data sources ===
    print("[Report] Loading price data...")
    df_prices = load_recent_price_data(weeks=4)
    price_summary = format_price_summary(df_prices)

    print("[Report] Loading technical indicators...")
    technical_summary = load_technical_indicators(df_prices)

    print("[Report] Loading sentiment data...")
    sentiment_summary = load_sentiment_data()

    print("[Report] Loading wordcloud keywords...")
    wordcloud_summary = load_wordcloud_keywords()

    print("[Report] Loading news headlines...")
    news_summary = load_recent_news_headlines(max_headlines=15)

    print("[Report] Loading model predictions...")
    prediction_summary = load_model_predictions()

    print("[Report] Loading skills knowledge base...")
    skills_summary = load_relevant_skills()

    # Track data sources used
    data_sources_used = []
    if "No price data" not in price_summary:
        data_sources_used.append("✅ Price Trends (4 weeks, Top 20 coins)")
    if "No technical" not in technical_summary:
        data_sources_used.append("✅ Technical Indicators (momentum, volatility, volume shock)")
    if "No sentiment" not in sentiment_summary:
        data_sources_used.append("✅ Sentiment Analysis (VADER compound, fear/greed)")
    if "No wordcloud" not in wordcloud_summary:
        data_sources_used.append("✅ Wordcloud Keywords (news frequency analysis)")
    if "No recent news" not in news_summary:
        data_sources_used.append("✅ News Headlines (CoinDesk)")
    if "No prediction" not in prediction_summary:
        data_sources_used.append("✅ ML Predictions (ElasticNet + ExtraTrees × 4 models)")
    if "No skills" not in skills_summary and "No relevant" not in skills_summary:
        data_sources_used.append("✅ Skills Knowledge Base (7 expert analysis modules)")

    data_sources_str = "\n".join(data_sources_used)

    now = datetime.now(timezone.utc)
    this_wednesday = get_latest_wednesday(now)

    # === STEP 2: Build enhanced prompt ===
    system_prompt = """You are a senior cryptocurrency market analyst with deep expertise in:
- Technical analysis (momentum, volatility, support/resistance)
- Sentiment analysis (NLP-based fear/greed indicators)
- Machine learning prediction interpretation
- Blockchain fundamentals and market cycle theory

You have been provided with REAL data from a professional crypto analysis system that includes:
price data, technical indicators, VADER sentiment scores, wordcloud keyword frequencies,
news headlines, ML model predictions, and expert knowledge modules (skills).

Your task is to synthesize ALL of these data sources into a comprehensive, deeply analytical
weekly market report. This is NOT a generic report — every section must reference specific
data points, numbers, and trends from the provided data.

Guidelines:
- Be objective and data-driven — ALWAYS cite specific numbers and data
- Cross-reference multiple data sources (e.g., price + sentiment + predictions)
- Highlight divergences between data sources (e.g., sentiment vs price action)
- Include specific actionable insights based on model consensus
- Format the report in clean Markdown with tables where appropriate
- Write in both English and Chinese (bilingual sections)
- Make the report comprehensive: 1500-2500 words
- Every claim must be backed by data provided

CRITICAL — Skills Integration Requirement:
You MUST deeply integrate the Skills knowledge base into your analysis. Do NOT just mention
skill names — you must APPLY the specific frameworks, formulas, and strategies from each skill.
For each skill used, you must:
1. Name the skill explicitly (e.g., "Applying: bitcoin-market-cycle-analysis")
2. Quote or paraphrase a specific concept/framework from that skill
3. Apply that concept to the CURRENT market data
4. Draw a concrete conclusion

Example of GOOD skills usage:
  "Applying **bitcoin-market-analysis-and-trading-strategy**: The skill states 'Volatility is
   our friend — rapid movements enable profit generation.' With BTC showing only 1.9% daily
   range (historically low), we are in a volatility compression phase. The skill's core strategy
   of 'wait for price to drop, then buy into it' suggests accumulating during this quiet period."

Example of BAD skills usage (DO NOT do this):
  "Based on the bull market trading strategy skill, we recommend buying dips."
  (Too vague, no specific framework quoted, no data connection)
"""

    user_prompt = f"""Generate a comprehensive weekly cryptocurrency market analysis report.

**Report Date:** {now.strftime('%Y-%m-%d')} (UTC)
**Data Week:** Ending {this_wednesday.strftime('%Y-%m-%d')}

---

## 📊 DATA SOURCE 1: Price Trends (Top 20 Coins, Last 4 Weeks)

{price_summary}

---

## 📈 DATA SOURCE 2: Technical Indicators (Latest Week)

{technical_summary}

---

## 😊 DATA SOURCE 3: Sentiment Analysis (VADER-based, Weekly Trend)

{sentiment_summary}

---

## ☁️ DATA SOURCE 4: Wordcloud / Hot Keywords (News Word Frequency)

{wordcloud_summary}

---

## 📰 DATA SOURCE 5: Recent News Headlines

{news_summary}

---

## 🤖 DATA SOURCE 6: ML Model Predictions (4 Models)

{prediction_summary}

---

## 📚 DATA SOURCE 7: Expert Knowledge Base (Skills)

{skills_summary}

---

Now, synthesize ALL the above data sources and generate a comprehensive report with these sections:

## Report Structure Required:

### 1. 📊 Executive Summary / 执行摘要
- One-paragraph overview synthesizing price, sentiment, and predictions
- Current Fear & Greed status with numerical score
- Key theme of the week (derived from wordcloud keywords)

### 2. 💰 Price Trend Analysis / 价格趋势分析
- Top 20 coins performance table with 1W and 4W returns
- Identify trends: which coins are in uptrend/downtrend/consolidation
- Volume analysis: unusual volume spikes or declines
- Compare with 4-week momentum

### 3. 🏆 Top Performers & Losers / 涨跌排行深度分析
- Top 3 gainers with possible factors (cross-reference with news headlines)
- Top 3 losers with possible factors
- Cross-reference with wordcloud keywords to find narrative drivers

### 4. 📈 Technical Analysis / 技术分析
- Momentum indicators interpretation
- Volatility regime assessment (high/low/normal)
- Volume shock analysis
- Key support/resistance levels based on data
- 🔑 MUST apply: **bitcoin-market-analysis-and-trading-strategy** skill
  → Use its "Volatility is our friend" framework and daily swing thresholds (+/-15%)
  → Apply its "Lower Average Price" strategy to current conditions
- 🔑 MUST apply: **bitcoin-market-cycle-analysis** skill
  → Identify current market cycle phase based on the skill's framework

### 5. 😊 Sentiment Deep Dive / 情绪深度分析
- Fear & Greed trend over recent weeks — is sentiment improving or deteriorating?
- Extreme positive/negative ratios — what do they signal?
- Wordcloud keyword analysis: what topics dominate the news narrative?
- Sentiment-Price divergence: is sentiment leading or lagging price action?

### 6. 🤖 ML Model Insights / 机器学习模型预测
- Summary of 4-model predictions
- Cross-model consensus: which coins are agreed upon?
- Model divergence: where do ElasticNet and ExtraTrees disagree? Why?
- Compare sentiment-inclusive vs market-only models

### 7. 🔮 Market Outlook & Strategy / 市场展望与策略建议
- Short-term outlook (1-2 weeks) based on data synthesis
- 🔑 MUST apply: **bull-market-trading-strategy** skill
  → Quote specific strategies (Value Cost Average, 20% profit rule, etc.)
  → Apply to current top picks from ML models
- 🔑 MUST apply: **cryptocurrency-investment-strategies** skill
  → Reference specific portfolio allocation frameworks
- Specific tactical suggestions based on model consensus
- Key events/catalysts to watch (from news data)

### 8. ⚠️ Risk Assessment / 风险评估
- Current risk level (based on volatility + sentiment)
- Key risk factors identified from data
- 🔑 MUST apply: **cryptocurrency-risk-assessment** skill
  → Use its specific risk categories and scoring framework
  → Quote its risk mitigation strategies and apply to current data
- 🔑 MUST apply: **cryptocurrency-valuation-and-price-determinants** skill
  → Reference valuation methods to assess over/undervaluation
- Portfolio sizing recommendations based on volatility regime

### 9. 📋 Data Sources & Methodology / 数据来源与方法论
- List all data sources used in this report
- Methodology notes
- Standard risk disclaimer

Format the report in Markdown with proper headers, bullet points, and tables.
Make every section data-rich — reference specific numbers, symbols, and trends.

⚠️ FINAL REMINDER: You MUST use at least 5 out of 7 skills in the report, each with:
- Explicit skill name in bold (e.g., **Applying: skill-name**)
- A quoted concept/framework from that skill
- Application to current data with specific numbers
The report will be rejected if skills are only mentioned by name without deep application.
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)

        print("[Report] Calling Claude Opus API with enhanced prompt...")
        response = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=8192,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        report_content = response.content[0].text

        # === STEP 3: Add rich metadata header ===
        full_report = f"""# 📊 Weekly Crypto Market Analysis Report
# 加密货币市场周度深度分析报告

**Generated / 生成时间:** {now.strftime('%Y-%m-%d %H:%M')} UTC
**Data Period / 数据周期:** {(this_wednesday - timedelta(days=7)).strftime('%Y-%m-%d')} to {this_wednesday.strftime('%Y-%m-%d')}
**Model / 模型:** Claude Opus (Enhanced Multi-Source Analysis)

### 📂 Data Sources Used / 使用的数据源:
{data_sources_str}

---

{report_content}

---

### 📂 Report Metadata / 报告元数据

| Item | Value |
|------|-------|
| Generation Time | {now.strftime('%Y-%m-%d %H:%M:%S')} UTC |
| Data Period | {(this_wednesday - timedelta(days=7)).strftime('%Y-%m-%d')} ~ {this_wednesday.strftime('%Y-%m-%d')} |
| Price Data Scope | Top 20 coins by volume (4 weeks) |
| Sentiment Engine | VADER + 700 crypto-specific terms |
| ML Models | ElasticNet (×2) + ExtraTrees (×2) |
| Skills Applied | 7 expert knowledge modules |
| News Source | CoinDesk API |
| AI Model | Claude Opus |

---

*This report is generated automatically by AI using real market data, ML predictions, and NLP sentiment analysis. It does not constitute investment advice.*
*本报告由AI自动生成，基于真实市场数据、机器学习预测和NLP情绪分析。仅供参考，不构成投资建议。*
"""

        # Save report
        REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPORT_FILE.write_text(full_report, encoding='utf-8')

        # Update timestamp
        REPORT_UPDATE_FILE.write_text(now.strftime('%Y-%m-%d'), encoding='utf-8')

        sources_count = len(data_sources_used)
        return full_report, True, f"Report generated with {sources_count} data sources at {now.strftime('%Y-%m-%d %H:%M')} UTC"

    except anthropic.APIError as e:
        return f"API Error: {str(e)}", False, f"Failed to generate report: {e}"
    except Exception as e:
        return f"Error: {str(e)}", False, f"Failed to generate report: {e}"


def load_existing_report() -> Optional[str]:
    """Load existing report if available"""
    if REPORT_FILE.exists():
        return REPORT_FILE.read_text(encoding='utf-8')
    return None


def get_report_last_update() -> Optional[str]:
    """Get the last update timestamp"""
    if REPORT_UPDATE_FILE.exists():
        return REPORT_UPDATE_FILE.read_text().strip()
    return None


def export_report_to_file(content: str, filename: str = "crypto_report.md") -> str:
    """
    Export report content to a downloadable file
    Returns the full path of the exported file
    """
    export_path = BASE_DIR / "figures" / filename
    export_path.write_text(content, encoding='utf-8')
    return str(export_path)
