"""
Claude AI Assistant Module for Crypto Rank App
Integrates Claude API with cryptocurrency skills knowledge base
Enhanced version with detailed project context and preset questions
"""

import os
from pathlib import Path
from typing import Optional, List, Dict
import anthropic

# Skills directory
SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# === Fixed Project Introduction (for preset question) ===
FIXED_PROJECT_INTRO = """
# Ansel Crypto Rank App 项目介绍
# Project Introduction

## 🎯 项目目标 / Project Goal
这是一个**机器学习驱动的加密货币周度推荐系统**，帮助投资者发现潜在的投资机会。

This is a **machine learning-driven weekly cryptocurrency recommendation system** that helps investors discover potential investment opportunities.

---

## 🔧 实现方式 / Implementation

### 数据源 / Data Sources
- **价格数据 / Price Data**: CoinDesk API，获取Top 100币种的每日OHLCV（开高低收量）
- **新闻数据 / News Data**: CoinDesk新闻API，用于情绪分析

### 特征工程 / Feature Engineering
系统计算以下技术指标作为预测特征：
The system calculates the following technical indicators as prediction features:

| 特征类型 / Feature Type | 指标 / Indicators |
|------------------------|-------------------|
| 📈 **动量 / Momentum** | 14/21/28/42/90天动量指标 |
| 📊 **波动率 / Volatility** | 14/21/28/42/90天年化波动率 |
| 💹 **交易量冲击 / Volume Shock** | 7/14/21/28/42天交易量变化 |
| 🕯️ **技术形态 / Technical Patterns** | 长烛线比率、短期反转信号 |
| 😊 **情绪指标 / Sentiment** | VADER情绪分析（含700+加密专用词汇）|

### 模型训练 / Model Training
- **训练窗口 / Training Window**: 滚动52周历史数据
- **预测目标 / Prediction Target**: 下周收益率
- **更新频率 / Update Frequency**: 每周四（基于周三收盘数据）

---

## 🤖 四个预测模型 / Four Prediction Models

| 模型名称 / Model | 算法 / Algorithm | 特征集 / Features | 特点 / Characteristics |
|-----------------|------------------|-------------------|------------------------|
| `enet_all` | ElasticNet | 市场+情绪 | L1/L2正则化，自动特征选择 |
| `enet_market` | ElasticNet | 仅市场 | 纯技术面，排除情绪噪声 |
| `extra_all` | ExtraTrees | 市场+情绪 | 集成学习，捕捉非线性关系 |
| `extra_market` | ExtraTrees | 仅市场 | 纯技术面，稳定性更高 |

### 模型对比 / Model Comparison
- **ElasticNet**: 线性模型，结合L1和L2正则化，自动进行特征选择，适合解释性需求
- **ExtraTrees**: 极端随机树集成，能捕捉复杂的非线性关系，泛化能力强

---

## 📊 输出结果 / Output Results
- **Top 20**: 预测收益最高的币种（做多建议）/ Highest predicted returns (Long recommendations)
- **Bottom 20**: 预测收益最低的币种（做空建议）/ Lowest predicted returns (Short recommendations)

---

## 📁 项目文件结构 / Project Structure

```
Ansel_Crypto_Rank_App/
├── main_app.py              # Streamlit Web界面
├── customer_side.py         # 预测管道
├── crypto_project_pipeline.py # ETL和特征工程
├── claude_assistant.py      # AI助手模块
├── ai_market_analysis.py    # AI市场分析报告
├── sentiment_update.py      # 情绪更新脚本
├── skills/                  # 39个教育知识模块
├── figures/                 # 可视化图表
├── text/                    # 配置文本
└── df_merged_history.csv    # 历史数据（~19M）
```

---

## 🕐 更新周期 / Update Schedule
| 功能 / Feature | 更新时间 / Update Time |
|---------------|----------------------|
| 预测推荐 / Predictions | 每周四 / Every Thursday |
| 情绪词云 / Sentiment Wordcloud | 每日 / Daily |
| AI分析报告 / AI Analysis Report | 每周三 / Every Wednesday |

---

## 💡 使用建议 / Usage Recommendations
根据回测结果，推荐策略排名：
Based on backtesting results, recommended strategy ranking:

1. **`extra_all` + Long-Short** - Sharpe比率最高 (2.20)
2. **`fusion_all` + Long-Short** - 平衡型 (Sharpe: 2.03)
3. **`extra_market` + Long-Short** - 纯技术面 (Sharpe: 1.90)

---

⚠️ **免责声明 / Disclaimer**
本系统仅供研究和参考，不构成投资建议。加密货币投资风险极高，请谨慎决策。

This system is for research and reference only and does not constitute investment advice. Cryptocurrency investments carry high risks. Please make decisions carefully.
"""

# === Preset Questions ===
PRESET_QUESTIONS = [
    {
        "id": "intro",
        "label": "📋 介绍这个项目",
        "label_en": "📋 About This Project",
        "question": "请详细介绍这个加密货币推荐项目",
        "type": "fixed",  # Returns fixed content, no API call
        "fixed_response": FIXED_PROJECT_INTRO
    },
    {
        "id": "interpret",
        "label": "🎯 如何使用推荐",
        "label_en": "🎯 How to Use Recommendations",
        "question": "如何解读和使用这个系统的推荐结果？",
        "type": "context"  # Uses context to answer
    },
    {
        "id": "risk",
        "label": "⚠️ 投资风险提示",
        "label_en": "⚠️ Investment Risks",
        "question": "加密货币投资有哪些主要风险？",
        "type": "context"
    },
    {
        "id": "models",
        "label": "🤖 模型区别说明",
        "label_en": "🤖 Model Differences",
        "question": "这四个模型（enet_all, enet_market, extra_all, extra_market）有什么区别？我该选哪个？",
        "type": "context"
    }
]


def load_all_skills() -> str:
    """Load all skill markdown files and combine into a single knowledge base string."""
    skills_content = []

    if not SKILLS_DIR.exists():
        return ""

    # Load each SKILL.md file
    for skill_folder in sorted(SKILLS_DIR.iterdir()):
        if skill_folder.is_dir():
            skill_file = skill_folder / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text(encoding="utf-8")
                skills_content.append(f"## {skill_folder.name}\n\n{content}")

                # Also load reference files if they exist
                refs_dir = skill_folder / "references"
                if refs_dir.exists():
                    for ref_file in refs_dir.glob("*.md"):
                        ref_content = ref_file.read_text(encoding="utf-8")
                        skills_content.append(f"### Reference: {ref_file.stem}\n\n{ref_content}")

    return "\n\n---\n\n".join(skills_content)


def get_skills_list() -> List[Dict[str, str]]:
    """Get a list of all available skills with names and descriptions."""
    skills = []

    if not SKILLS_DIR.exists():
        return skills

    for skill_folder in sorted(SKILLS_DIR.iterdir()):
        if skill_folder.is_dir():
            skill_file = skill_folder / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text(encoding="utf-8")

                # Extract name and description from frontmatter
                name = skill_folder.name
                description = ""

                # Parse YAML frontmatter
                if content.startswith("---"):
                    try:
                        end_idx = content.index("---", 3)
                        frontmatter = content[3:end_idx]
                        for line in frontmatter.split("\n"):
                            if line.startswith("description:"):
                                description = line.replace("description:", "").strip()
                                break
                    except ValueError:
                        pass

                # Fallback: get first paragraph after title
                if not description:
                    lines = content.split("\n")
                    for line in lines:
                        if line and not line.startswith("#") and not line.startswith("---"):
                            description = line[:200] + "..." if len(line) > 200 else line
                            break

                skills.append({
                    "name": name,
                    "title": name.replace("-", " ").title(),
                    "description": description
                })

    return skills


def get_skill_content(skill_name: str) -> Optional[str]:
    """Get the full content of a specific skill."""
    skill_folder = SKILLS_DIR / skill_name
    if not skill_folder.exists():
        return None

    skill_file = skill_folder / "SKILL.md"
    if not skill_file.exists():
        return None

    content = skill_file.read_text(encoding="utf-8")

    # Also include reference files
    refs_dir = skill_folder / "references"
    if refs_dir.exists():
        for ref_file in refs_dir.glob("*.md"):
            ref_content = ref_file.read_text(encoding="utf-8")
            content += f"\n\n### Reference: {ref_file.stem}\n\n{ref_content}"

    return content


def get_preset_questions() -> List[Dict]:
    """Get list of preset questions for UI"""
    return PRESET_QUESTIONS


def handle_preset_question(question_id: str) -> Optional[str]:
    """
    Handle preset question - returns fixed response if type is 'fixed'
    Returns None if question should be handled by AI
    """
    for q in PRESET_QUESTIONS:
        if q["id"] == question_id and q["type"] == "fixed":
            return q.get("fixed_response", None)
    return None


def build_system_prompt(project_context: str = "") -> str:
    """Build the system prompt for Claude with skills knowledge."""

    skills_knowledge = load_all_skills()

    system_prompt = f"""You are an expert cryptocurrency and blockchain assistant for the Ansel Crypto Rank App.

## Your Identity
- You are a knowledgeable crypto market expert with deep understanding of blockchain technology, trading strategies, and market analysis.
- You speak in a friendly, professional manner and can explain complex concepts in simple terms.
- You are bilingual and can respond in both English and Chinese based on user's language.

## About This Project (Detailed)

### Project Overview
The Ansel Crypto Rank App is a machine learning-based cryptocurrency recommendation system that:
- Uses ElasticNet and ExtraTrees models to predict weekly crypto returns
- Analyzes market features (momentum, volatility, volume) and sentiment features from news
- Provides Top 20 (long) and Bottom 20 (short) token recommendations every week
- Updates sentiment analysis (word clouds, fear & greed index) daily
- Updates predictions every Thursday after Wednesday market data is complete

### The Four Models
1. **enet_all**: ElasticNet with all features (market + sentiment)
   - L1/L2 regularization for automatic feature selection
   - Good interpretability

2. **enet_market**: ElasticNet with market features only
   - Pure technical analysis, excludes sentiment noise
   - More stable in volatile news environments

3. **extra_all**: ExtraTrees with all features (market + sentiment)
   - Ensemble learning, captures non-linear relationships
   - Highest Sharpe ratio in backtests (2.20)

4. **extra_market**: ExtraTrees with market features only
   - Pure technical approach
   - Most stable across different market conditions

### Feature Categories
- **Momentum**: 14/21/28/42/90-day momentum indicators
- **Volatility**: Annualized volatility over various windows
- **Volume Shock**: Volume changes vs historical averages
- **Technical Patterns**: Long candle ratio, reversal signals
- **Sentiment**: VADER compound scores with 700+ crypto-specific terms

### Update Schedule
- Predictions: Every Thursday (based on Wednesday close)
- Sentiment/Wordcloud: Daily
- AI Analysis Report: Every Wednesday

{project_context}

## Your Knowledge Base (Skills)
You have been loaded with comprehensive cryptocurrency investment and blockchain knowledge. Use this knowledge to answer questions accurately:

{skills_knowledge}

## Response Guidelines
1. Always provide accurate, helpful information based on your skills knowledge
2. When discussing investments, always include appropriate risk disclaimers
3. Be specific with data and examples when available
4. If asked about the project's recommendations, explain the methodology
5. If you don't know something, say so honestly
6. Format responses with markdown for better readability
7. Keep responses concise but comprehensive
8. Respond in the same language the user uses (Chinese/English)

## Risk Disclaimer
Always remind users that cryptocurrency investments carry significant risks. Past performance does not guarantee future results. Never provide specific financial advice - only educational information."""

    return system_prompt


class CryptoAssistant:
    """Claude-based cryptocurrency assistant."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-20250514"):
        """Initialize the assistant with API key."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = None
        self.conversation_history = []
        self.model = model

        if self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)

    def is_available(self) -> bool:
        """Check if the assistant is properly configured."""
        return bool(self.api_key and self.client)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def chat(self, user_message: str, project_context: str = "") -> str:
        """Send a message and get a response."""
        if not self.is_available():
            return "Error: Claude API key not configured. Please set ANTHROPIC_API_KEY environment variable."

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=build_system_prompt(project_context),
                messages=self.conversation_history
            )

            # Extract assistant response
            assistant_message = response.content[0].text

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            return assistant_message

        except anthropic.APIError as e:
            return f"API Error: {str(e)}"
        except Exception as e:
            return f"Error: {str(e)}"


# Singleton instance for Streamlit
_assistant_instance = None

def get_assistant(api_key: Optional[str] = None, model: str = "claude-opus-4-20250514") -> CryptoAssistant:
    """Get or create the assistant singleton."""
    global _assistant_instance
    if _assistant_instance is None or (api_key and api_key != _assistant_instance.api_key):
        _assistant_instance = CryptoAssistant(api_key, model)
    return _assistant_instance


# === Skills Categorization Helper ===
SKILL_CATEGORIES = {
    "📖 Fundamentals": ["cryptocurrency-fundamentals", "currency-evolution", "early-money", "monetary-properties"],
    "₿ Bitcoin & Blockchain": ["bitcoin-", "blockchain-", "proof-of-work", "ethereum", "transaction-", "double-spend"],
    "📈 Trading & Investment": ["investment-strateg", "bull-market", "market-analysis", "market-cycle", "valuation", "portfolio"],
    "🔒 Security & Wallets": ["wallet-", "security", "risk-"],
    "🪙 Tokens & ICO": ["ico-", "token-", "exchange-token"],
    "🔗 Other": []
}


def categorize_skills(skills: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize skills into groups for UI display"""
    categorized = {cat: [] for cat in SKILL_CATEGORIES}

    for skill in skills:
        placed = False
        for cat, keywords in SKILL_CATEGORIES.items():
            if cat == "🔗 Other":
                continue
            for kw in keywords:
                if kw in skill["name"]:
                    categorized[cat].append(skill)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            categorized["🔗 Other"].append(skill)

    return categorized
