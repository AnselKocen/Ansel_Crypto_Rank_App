import pandas as pd
import streamlit as st
from PIL import Image

from pathlib import Path
import subprocess
import base64
import json
import streamlit.components.v1 as components
from datetime import datetime, timezone
from functools import lru_cache

# === NLTK 数据包管理（优化版）===
import nltk

@st.cache_resource
def ensure_nltk_data():
    """确保NLTK数据包已下载，使用Streamlit缓存只执行一次"""
    required_packages = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'vader_lexicon': 'sentiment/vader_lexicon'
    }

    for package, path in required_packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(package, quiet=True)
            except Exception:
                pass  # 静默处理下载失败
    return True

# 只在需要时调用（延迟初始化）
_nltk_initialized = False

def get_nltk_ready():
    """获取NLTK就绪状态，首次调用时初始化"""
    global _nltk_initialized
    if not _nltk_initialized:
        ensure_nltk_data()
        _nltk_initialized = True
    return True


# === 设置页面信息 ===
st.set_page_config(page_title="Crypto Investment Strategy Hub", layout="wide")
st.title("🪙 Cryptocurrency Market Analysis & Strategy Visualization")

# === 路径设置 ===
BASE_DIR = Path(__file__).resolve().parent
fig_dir = BASE_DIR / "figures"
text_dir = BASE_DIR / "text"

# === 辅助函数：居中显示图像 ===
def show_centered_img(img_path, caption="", width_percent=70):
    if not Path(img_path).exists():
        st.warning(f"no pictures: {img_path}")
        return

    with open(img_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{img_base64}"
                 style="width: {width_percent}%; border-radius: 10px;" />
            <div style="font-size: 14px; color: gray; margin-top: 8px;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# === 辅助函数：TradingView 符号转换 ===
# 符号映射表：预测文件符号 -> 币安交易对符号
SYMBOL_MAPPING = {
    # 常见映射
    "MANTLE": "MNT",
    "BTT": "BTTC",
    "IOTA": "IOTA",
    "LUNA": "LUNC",      # Luna Classic
    "TERRA": "LUNA",     # Terra 2.0
    "MIOTA": "IOTA",
    # 稳定币相关
    "USTC": "USTC",
    # 其他常见映射
    "XBT": "BTC",
    "XDG": "DOGE",
}

# 币安上不存在的币种（跳过）- 扩展列表
UNSUPPORTED_SYMBOLS = {
    # 交易所代币（非Binance）
    "HTX", "XT", "GT", "KCS", "OKB", "FTT", "CRO",
    # 不在Binance的币种
    "XDC", "BSV", "FLR", "TIBBIR", "ZANO", "XPL", "WMTX",
    "TRIBE", "WFI", "HYPE", "ASTER", "MON", "PUMP", "JLP",
    # 隐私币（部分地区受限）
    "XMR", "ZEC", "DASH",
    # 其他
    "BDX", "XAUT", "GAS", "NEXO", "LEO", "QNT", "DCR",
    "SPX", "PI", "TFUEL", "RSR", "GNO", "TEL", "LPT",
}

def map_symbol_to_tradingview(symbol: str) -> str:
    """将预测文件中的符号转换为 TradingView 格式"""
    symbol = symbol.upper().strip()
    if not symbol.isascii() or not symbol:
        return None

    # 跳过不支持的币种
    if symbol in UNSUPPORTED_SYMBOLS:
        return None

    # 应用符号映射
    mapped_symbol = SYMBOL_MAPPING.get(symbol, symbol)

    return f"BINANCE:{mapped_symbol}USDT"


# === 辅助函数：生成 TradingView Ticker Widget ===
def generate_tradingview_ticker_widget(symbols: list, widget_id: str) -> str:
    """生成 TradingView Ticker Tape 小部件 HTML"""
    symbols = [s for s in symbols[:15] if s]
    symbols_json = [{"proName": s, "title": s.split(":")[-1].replace("USDT", "")}
                    for s in symbols]
    return f'''
    <div class="tradingview-widget-container" id="{widget_id}">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
      {{
        "symbols": {json.dumps(symbols_json)},
        "showSymbolLogo": true,
        "colorTheme": "light",
        "isTransparent": false,
        "displayMode": "adaptive",
        "locale": "en"
      }}
      </script>
    </div>
    '''


# === 辅助函数：运行外部脚本并提取关键输出 ===
def run_external_script(script_path: str):
    result_lines = []
    top_outputs = []
    bot_outputs = []
    notices = []   # ✅ 用于保存 ###NOTICE###

    process = subprocess.Popen(
        ["python", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    with st.expander("📄 Streaming Log Output (Click to Expand) ", expanded=False):
        for line in process.stdout:
            st.write(line.strip())
            result_lines.append(line.strip())

            # ✅ 提取关键输出
            if line.strip().startswith("###NOTICE###"):
                notices.append(line.strip().replace("###NOTICE###", "").strip())
            elif line.strip().startswith(">>>TOP20_"):
                top_outputs.append(line.strip().replace(">>>TOP20_", ""))
            elif line.strip().startswith(">>>BOT20_"):
                bot_outputs.append(line.strip().replace(">>>BOT20_", ""))

    process.wait()
    if process.returncode == 0:
        st.success("✅ Script Execution Completed! ")
    else:
        st.error(f"❌ Script Execution Failed, Return Code:{process.returncode}")

    return top_outputs, bot_outputs, notices


# === 辅助函数：渲染Skills浏览器（三栏布局的左两栏）===
def render_skills_browser(col_width_ratio=[1, 1], key_prefix=""):
    """渲染Skills浏览器组件，返回API key"""
    from claude_assistant import get_skills_list, get_skill_content, categorize_skills

    col_skills, col_detail = st.columns(col_width_ratio)

    # Session state keys with prefix to avoid conflicts
    skill_key = f"{key_prefix}_selected_skill"
    if skill_key not in st.session_state:
        st.session_state[skill_key] = None

    # === Left Column: Skills List ===
    with col_skills:
        st.subheader("📚 Knowledge Base")

        # API Key handling
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""

        if default_key:
            api_key = default_key
            st.success("✅ API Key loaded")
        else:
            api_key = st.text_input(
                "🔑 Anthropic API Key",
                type="password",
                help="Enter your Anthropic API key",
                key=f"{key_prefix}_api_key"
            )
            if api_key:
                st.success("✅ API Key configured")

        # Skills list
        skills = get_skills_list()
        categorized = categorize_skills(skills)

        # Display categorized skills
        for category, cat_skills in categorized.items():
            if cat_skills:
                with st.expander(f"{category} ({len(cat_skills)})", expanded=False):
                    for skill in cat_skills:
                        if st.button(
                            skill["title"][:25] + "..." if len(skill["title"]) > 25 else skill["title"],
                            key=f"{key_prefix}_skill_{skill['name']}",
                            help=skill["description"][:80] if skill["description"] else "",
                            use_container_width=True
                        ):
                            st.session_state[skill_key] = skill["name"]

    # === Middle Column: Skill Detail ===
    with col_detail:
        st.subheader("📄 Skill Detail")

        if st.session_state[skill_key]:
            content = get_skill_content(st.session_state[skill_key])
            if content:
                with st.container(height=500):
                    st.markdown(content)
        else:
            st.info("👈 Click a skill on the left to view details")

    return api_key


# ============================================================
# === 顶部 Tabs 页面结构 (重新排序 - 11个Tab) ===
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "🎓 Educational",           # 1. 教育（保持不变）
    "💭 Sentiment",             # 2. 情绪词云展示
    "🤖 AI Analysis",           # 3. AI市场分析报告
    "💬 AI Q&A",                # 4. AI问答
    "📈 Live Prices",           # 5. 价格展示
    "💰 Investment",            # 6. 建议
    "📝 Model Strategy",        # 7. 模型策略
    "🛠️ Hyper-Params",          # 8. 超参数
    "🔍 Features",              # 9. 特征选择
    "📊 Backtest",              # 10. 回测结果
    "▶️ Prediction"             # 11. 运行预测
])


# ============================================================
# === Tab 1：教育资源 ===
# ============================================================
with tab1:
    st.header("📚 Crypto Basics & Educational Resources")

    st.markdown("Here are some beginner-friendly learning resources. You can also use our 💬 AI Q&A for support.")

    # 🔗 教学链接
    st.markdown("🔗 [Binance Academy – What Is Cryptocurrency?](https://academy.binance.com/en/articles/what-is-a-cryptocurrency)")
    st.markdown("🔗 [CoinMarketCap：Stay Updated on the Latest Cryptocurrency Price Trends](https://coinmarketcap.com/)")
    st.markdown("🔗 [educational YouTuber：@simplyexplained](https://www.youtube.com/@simplyexplained)")
    st.markdown("🔗 [What is long and short?](https://www.youtube.com/watch?v=fXnCtGcvqdk&t=31s)")

    # 🎥 教学视频嵌入
    st.subheader("🎥 What is crypto currency🪙?")
    st.video("https://www.youtube.com/watch?v=Zoz9gvhLgpM")


# ============================================================
# === Tab 2：市场情绪指数 ===
# ============================================================
with tab2:
    st.header("📰 Market Sentiment Index over the last 7-days")
    update_file = BASE_DIR / "last_updated_wordcloud.txt"

    # 如果文件存在，读取日期
    if update_file.exists():
        last_updated_str = update_file.read_text().strip()
        st.info(f"📅 Last updated: {last_updated_str}")
        # 转为日期进行比较
        try:
            last_updated_date = datetime.strptime(last_updated_str, "%Y-%m-%d").date()
            today = datetime.now(timezone.utc).date()

            # 如果不是今天，提示用户需要更新
            if last_updated_date < today:
                st.warning("⚠️ This data may be outdated.")
        except ValueError:
            st.error("❌ Invalid update date format in last_updated_wordcloud.txt.")
    else:
        st.info("ℹ️ Word cloud and Gauge have not been generated yet.")

    if st.button("▶️ Update Word Cloud and Fear & Greed Gauge", key="update_sentiment"):
        # 初始化NLTK
        get_nltk_ready()
        with st.spinner("Generating... This may take up to 5 minutes."):
            try:
                from sentiment_update import update_sentiment_and_gauge
                update_sentiment_and_gauge()
                st.success("✅ Updated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error occurred during update: {e}")

    option = st.radio(
        label="Display Options",
        options=["Word Cloud", "Fear & Greed Index"],
        horizontal=True,
        label_visibility="collapsed"
    )
    if option == "Word Cloud":
        st.subheader("☁️ Sentiment Word Cloud")

        col1, col2 = st.columns(2)
        with col1:
            wc_pos_path = fig_dir / "wordcloud_positive.png"
            show_centered_img(wc_pos_path, caption="Positive wordcloud", width_percent=91)
        with col2:
            wc_neg_path = fig_dir / "wordcloud_negative.png"
            show_centered_img(wc_neg_path, caption="Negative wordcloud", width_percent=91)
    elif option == "Fear & Greed Index":
        st.subheader("🧭 Fear & Greed Gauge")
        gauge_path = fig_dir / "fear_greed_gauge.png"
        show_centered_img(gauge_path, caption="Fear & Greed Gauge this week", width_percent=61)


# ============================================================
# === Tab 3：AI市场分析报告 ===
# ============================================================
with tab3:
    st.header("🤖 AI Market Analysis Report")
    st.markdown("Weekly market analysis generated by Claude Opus, based on price data and market trends.")

    from ai_market_analysis import (
        generate_analysis_report,
        load_existing_report,
        get_report_last_update,
        should_update_report
    )
    from claude_assistant import get_skills_list, get_skill_content, categorize_skills

    # 三栏布局：Skills列表(25%) | Skills详情(25%) | 报告(50%)
    col_skills, col_detail, col_report = st.columns([1, 1, 2])

    # Session state for this tab
    if "analysis_selected_skill" not in st.session_state:
        st.session_state.analysis_selected_skill = None

    # === Left Column: Skills List ===
    with col_skills:
        st.subheader("📚 Knowledge Base")

        # API Key handling
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""

        if default_key:
            api_key_analysis = default_key
            st.success("✅ API Key loaded")
        else:
            api_key_analysis = st.text_input(
                "🔑 Anthropic API Key",
                type="password",
                help="Required for generating reports",
                key="analysis_api_key"
            )
            if api_key_analysis:
                st.success("✅ API Key configured")

        # Skills list
        skills = get_skills_list()
        categorized = categorize_skills(skills)

        for category, cat_skills in categorized.items():
            if cat_skills:
                with st.expander(f"{category} ({len(cat_skills)})", expanded=False):
                    for skill in cat_skills:
                        if st.button(
                            skill["title"][:22] + "..." if len(skill["title"]) > 22 else skill["title"],
                            key=f"analysis_skill_{skill['name']}",
                            help=skill["description"][:60] if skill["description"] else "",
                            use_container_width=True
                        ):
                            st.session_state.analysis_selected_skill = skill["name"]

    # === Middle Column: Skill Detail ===
    with col_detail:
        st.subheader("📄 Skill Detail")

        if st.session_state.analysis_selected_skill:
            content = get_skill_content(st.session_state.analysis_selected_skill)
            if content:
                with st.container(height=500):
                    st.markdown(content)
        else:
            st.info("👈 Click a skill to view")

    # === Right Column: Report ===
    with col_report:
        st.subheader("📊 Weekly Analysis Report")

        # Report status
        last_update = get_report_last_update()
        needs_update, reason = should_update_report()

        if last_update:
            st.info(f"📅 Last updated: {last_update}")

        if needs_update:
            st.warning(f"⚠️ {reason}")

        # Action buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Generate/Update Report", key="gen_report", use_container_width=True):
                if not api_key_analysis:
                    st.error("Please enter API Key first")
                else:
                    with st.spinner("Generating report with Claude Opus... This may take 1-2 minutes."):
                        report, was_generated, message = generate_analysis_report(
                            api_key_analysis,
                            force_update=True
                        )
                        if was_generated:
                            st.success(f"✅ {message}")
                            st.rerun()
                        else:
                            st.error(f"❌ {message}")

        with col_btn2:
            # Download button
            existing_report = load_existing_report()
            if existing_report:
                st.download_button(
                    label="📥 Download Report",
                    data=existing_report,
                    file_name=f"crypto_report_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    use_container_width=True
                )

        # Display report
        st.markdown("---")
        existing_report = load_existing_report()
        if existing_report:
            with st.container(height=450):
                st.markdown(existing_report)
        else:
            st.info("No report available. Click 'Generate/Update Report' to create one.")


# ============================================================
# === Tab 4：AI问答 ===
# ============================================================
with tab4:
    st.header("💬 AI Q&A Assistant")
    st.markdown("Ask anything about cryptocurrency, trading strategies, or this project.")

    from claude_assistant import (
        get_assistant, get_skills_list, get_skill_content,
        categorize_skills, get_preset_questions, handle_preset_question
    )

    # Initialize session state
    if "qa_chat_history" not in st.session_state:
        st.session_state.qa_chat_history = []
    if "qa_selected_skill" not in st.session_state:
        st.session_state.qa_selected_skill = None

    # 三栏布局：Skills列表(25%) | Skills详情(25%) | 对话(50%)
    col_skills, col_detail, col_chat = st.columns([1, 1, 2])

    # === Left Column: Skills List ===
    with col_skills:
        st.subheader("📚 Knowledge Base")

        # API Key handling
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""

        if default_key:
            api_key_qa = default_key
            st.success("✅ API Key loaded")
        else:
            api_key_qa = st.text_input(
                "🔑 Anthropic API Key",
                type="password",
                help="Required for AI chat",
                key="qa_api_key"
            )
            if api_key_qa:
                st.success("✅ API Key configured")

        # Skills list
        skills = get_skills_list()
        categorized = categorize_skills(skills)

        for category, cat_skills in categorized.items():
            if cat_skills:
                with st.expander(f"{category} ({len(cat_skills)})", expanded=False):
                    for skill in cat_skills:
                        if st.button(
                            skill["title"][:22] + "..." if len(skill["title"]) > 22 else skill["title"],
                            key=f"qa_skill_{skill['name']}",
                            help=skill["description"][:60] if skill["description"] else "",
                            use_container_width=True
                        ):
                            st.session_state.qa_selected_skill = skill["name"]

    # === Middle Column: Skill Detail ===
    with col_detail:
        st.subheader("📄 Skill Detail")

        if st.session_state.qa_selected_skill:
            content = get_skill_content(st.session_state.qa_selected_skill)
            if content:
                with st.container(height=500):
                    st.markdown(content)
        else:
            st.info("👈 Click a skill to view")

    # === Right Column: Chat Interface ===
    with col_chat:
        st.subheader("💬 Chat")

        # Preset question buttons
        st.markdown("**Quick Questions / 快捷问题:**")
        preset_questions = get_preset_questions()

        preset_cols = st.columns(2)
        for i, pq in enumerate(preset_questions):
            with preset_cols[i % 2]:
                if st.button(pq["label"], key=f"preset_{pq['id']}", use_container_width=True):
                    # Check if it's a fixed response
                    fixed_response = handle_preset_question(pq["id"])
                    if fixed_response:
                        # Add to history directly without API call
                        st.session_state.qa_chat_history.append({
                            "role": "user",
                            "content": pq["question"]
                        })
                        st.session_state.qa_chat_history.append({
                            "role": "assistant",
                            "content": fixed_response
                        })
                        st.rerun()
                    else:
                        st.session_state.pending_qa_question = pq["question"]

        st.markdown("---")

        # Chat container
        chat_container = st.container(height=350)

        with chat_container:
            # Display chat history
            for msg in st.session_state.qa_chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask anything about crypto or this project...", key="qa_input")

        # Handle pending question from preset buttons
        if "pending_qa_question" in st.session_state:
            user_input = st.session_state.pending_qa_question
            del st.session_state.pending_qa_question

        if user_input:
            # Add user message to history
            st.session_state.qa_chat_history.append({"role": "user", "content": user_input})

            # Get AI response
            if api_key_qa:
                assistant = get_assistant(api_key_qa, model="claude-opus-4-20250514")
                # Sync history
                assistant.conversation_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.qa_chat_history[:-1]
                ]

                # Project context
                project_context = """
Current project status:
- Predictions are updated weekly on Thursdays
- Sentiment (word clouds, fear & greed index) is updated daily
- AI Analysis Report is updated every Wednesday
- Uses 4 models: enet_all, enet_market, extra_all, extra_market
- Top 20 tokens = Long recommendations, Bottom 20 = Short recommendations
"""
                with st.spinner("Thinking..."):
                    response = assistant.chat(user_input, project_context)
            else:
                response = "⚠️ Please enter your Anthropic API Key to enable AI chat.\n\nGet one at [console.anthropic.com](https://console.anthropic.com)"

            # Add response to history
            st.session_state.qa_chat_history.append({"role": "assistant", "content": response})
            st.rerun()

        # Clear chat button
        if st.button("🗑️ Clear Chat", key="clear_qa_chat"):
            st.session_state.qa_chat_history = []
            if api_key_qa:
                get_assistant(api_key_qa).clear_history()
            st.rerun()


# ============================================================
# === Tab 5：实时价格追踪 ===
# ============================================================
with tab5:
    st.header("📈 Real-time Price Tracking")
    st.markdown("📈 Real-time Price Tracking by @TradingView")

    # 模型选择器
    selected_model = st.selectbox(
        "Select Prediction Model",
        ["enet_all", "enet_market", "extra_all", "extra_market"],
        index=0,
        key="price_model_select"
    )

    # 加载预测结果
    pred_file = BASE_DIR / f"pred_{selected_model}.csv"
    if pred_file.exists():
        df_pred = pd.read_csv(pred_file)

        # 获取 Top 20 和 Bottom 20
        top_20 = df_pred.head(20)["symbol"].tolist()
        bot_20 = df_pred.tail(20)["symbol"].tolist()

        # 转换为 TradingView 格式
        top_tv = [s for s in [map_symbol_to_tradingview(sym) for sym in top_20] if s]
        bot_tv = [s for s in [map_symbol_to_tradingview(sym) for sym in bot_20] if s]

        # 做多推荐
        st.subheader("🟢 Long Recommendations (Top 20)")
        st.caption("Tokens with highest predicted returns")
        if top_tv:
            components.html(generate_tradingview_ticker_widget(top_tv[:15], "long1"), height=80)
            if len(top_tv) > 15:
                components.html(generate_tradingview_ticker_widget(top_tv[15:], "long2"), height=80)

        st.markdown("---")

        # 做空推荐
        st.subheader("🔴 Short Recommendations (Bottom 20)")
        st.caption("Tokens with lowest predicted returns")
        if bot_tv:
            components.html(generate_tradingview_ticker_widget(bot_tv[:15], "short1"), height=80)
            if len(bot_tv) > 15:
                components.html(generate_tradingview_ticker_widget(bot_tv[15:], "short2"), height=80)

        # 符号列表
        with st.expander("📋 View Full List"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🟢 Long:**")
                st.write(", ".join(top_20))
            with col2:
                st.markdown("**🔴 Short:**")
                st.write(", ".join(bot_20))
    else:
        st.warning("Prediction file not found. Please run the ▶️ Prediction tab first to generate recommendations.")


# ============================================================
# === Tab 6：投资建议 ===
# ============================================================
with tab6:
    st.markdown("## 🎯 Which Strategy Suits Me Best? Let Us Help You Choose!")
    st.markdown(
        "Our platform recommends the most suitable strategy combinations based on various indicators, so your investment decisions are no longer a guess!")
    st.markdown(
        "👉 *Match your selected Model Feature tag, then head to ▶️ Prediction to see the Top 20 and Bottom 20 token picks*"
        " and start building your portfolio according to the **corresponding strategy**!")

    # === 🥇 Top 3 Sharpe Ratio Strategies ===
    st.markdown("### 🏅 Top Pick: Sharpe Ratio Champions (Best Risk-Adjusted Return)")
    st.markdown(
        "💡 **If you aim to maximize excess return per unit of risk**, start by considering the strategies with the highest Sharpe Ratios:")

    sharpe_df = pd.DataFrame({
        "🏆 Rank": ["🥇 1st ", "🥈 2nd ", "🥉 3rd "],
        "Model Feature Tag": ["extra_all", "fusion_all", "extra_market"],
        "Strategy": ["ls", "ls", "ls"],
        "Sharpe Ratio": [2.20, 2.03, 1.90]
    })
    st.dataframe(sharpe_df, hide_index=True)

    # === 👤 Risk Preference Guide ===
    st.markdown("### 😎 Strategy Recommendations Based on Your Risk Preference")

    st.markdown("#### 🔥 High Risk Appetite (Maximize Return)")
    high_risk_df = pd.DataFrame({
        "🏆 Rank": ["🥇 1st ", "🥈 2nd ", "🥉 3rd "],
        "Model Feature Tag": ["extra_all", "extra_market", "extra_all"],
        "Strategy": ["EW", "EW", "ls"],
        "Annualized Return": [2.10, 1.80, 1.75]
    })
    st.dataframe(high_risk_df, hide_index=True)

    st.markdown(
        "📝 Note: These strategies offer high returns but also higher volatility — perfect for aggressive investors!")

    st.markdown("#### 🧊 Low Risk Appetite (Stability First)")
    low_risk_df = pd.DataFrame({
        "🏆 Rank": ["🥇 1st ", "🥈 2nd ", "🥉 3rd "],
        "Model Feature Tag": ["fusion_market", "fusion_all", "extra_market"],
        "Strategy": ["ls", "ls", "ls"],
        "Annualized Volatility": [0.36, 0.38, 0.42]
    })
    st.dataframe(low_risk_df, hide_index=True)

    # === Allocation Recommendation ===
    st.markdown("### 🧮 What is the difference between ls and EW?")
    st.markdown("👉 **Equal Weight (EW)** allocation is the simplest and most effective method.")
    st.info(
        "In fact, very few weighted models consistently outperform EW over the long run! Just allocate equal capital to the top tokens weekly for both long and short sides.")

    st.warning(
        "⚠️ If you're using **`ls`** strategies, make sure you understand how **shorting** works! Visit the 🎓 Education tab to watch the video!")


# ============================================================
# === Tab 7：模型策略介绍 ===
# ============================================================
emoji_map = {
    "market": "🔵",
    "all": "🟢",
    "enet_EW": "🟡",
    "extra_EW": "🟠",
    "enet_ls": "🔴",
    "extra_ls": "🟣",
    "fusion_ls": "🟤"
}

with tab7:
    st.header("📘 Model Strategy Overview")
    intro_path = text_dir / "strategy_intro.txt"
    if intro_path.exists():
        with open(intro_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if ":" in line:
                    key = line.split(":")[0]
                    emoji = emoji_map.get(key, "")
                    st.markdown(f"{emoji} **{line}**")
                else:
                    st.markdown(line)
    else:
        st.info("no txt")


# ============================================================
# === Tab 8：最优参数展示 ===
# ============================================================
with tab8:
    st.header("⚙️ Optimal Hyper-Parameters Display")
    param_path = text_dir / "best_params.txt"
    if param_path.exists():
        with open(param_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="yaml")
    else:
        st.info("no best_params.txt ")


# ============================================================
# === Tab 9：特征选择 ===
# ============================================================
with tab9:
    st.header("💡 Feature Selection")
    st.markdown(
        "Below are the optimal features selected based on data from 2020-01-01 to 2025-07-30."
    )
    # 展示 enet 特征选择 txt
    enet_feat_path = text_dir / "enet_features.txt"
    if enet_feat_path.exists():
        st.subheader("🔴 ElasticNet Selected Features")
        st.markdown(
            "The ElasticNet model automatically selects features it considers important and contributive, while compressing others."
        )
        with open(enet_feat_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")
    else:
        st.info("no enet_selected_features.txt")

    # 展示 ExtraTrees 特征图像
    st.subheader("🔵 ExtraTrees Features Importance")
    extra_fig_all = fig_dir / "extra_all_feature_importance.png"
    extra_fig_market = fig_dir / "extra_market_feature_importance.png"

    col1, col2 = st.columns(2)
    with col1:
        show_centered_img(extra_fig_all, caption="All Features", width_percent=91)
    with col2:
        show_centered_img(extra_fig_market, caption="Market Features", width_percent=91)


# ============================================================
# === Tab 10：策略回测结果对比 ===
# ============================================================
with tab10:
    st.header("📈 Strategy Backtest Results Comparison")
    backtest_option = st.radio(
        label="Comparison Metric Options",
        options=["Cumulative Return", "Volatility", "Average Return", "Sharpe Ratio", "Maximum Drawdown"],
        horizontal=True,
        label_visibility="collapsed",
        key="backtest_metric"
    )
    base_name_map = {
        "Cumulative Return": "cum_return_comparison",
        "Volatility": "volatility_bar",
        "Average Return": "mean_return_bar",
        "Sharpe Ratio": "sharpe_bar",
        "Maximum Drawdown": "max_drawdown_bar"
    }
    base_name = base_name_map[backtest_option]
    variants = ["all", "market"]
    for variant in variants:
        img_name = f"{base_name}_{variant}_with_baseline.png"
        img_path = fig_dir / img_name
        if "cum_return" in img_name:
            show_centered_img(img_path, width_percent=71)
        else:
            show_centered_img(img_path, width_percent=51)


# ============================================================
# === Tab 11：运行预测 ===
# ============================================================
from customer_side import run_for_client

with tab11:
    st.header("🧠 Get Next Week's Recommended Portfolio!")
    st.markdown(
        "🟦 Click the button to automatically fetch crypto market and news data up to the most recent Wednesday (t), merge it with historical data, and perform ETL.  \n"
        "🟨 Models are trained on the past 52 weeks (t−52 to t−1), and using features observed in week t to rank expected returns for week t+1.  \n"
        "🟥 **Disclaimer: For reference only. Not financial advice.**"
    )

    if st.button("▶️ Click to Get Recommended Tokens", key="run_prediction"):
        # 初始化NLTK
        get_nltk_ready()
        st.session_state["log_expanded"] = False

        with st.spinner("Running, please wait... Might take 5–10 mins if you haven't run this page in a while"):
            try:
                API_KEY = ""  # ← 可改为 st.secrets["API_KEY"]
                HISTORY_PATH = BASE_DIR / "df_merged_history.csv"
                results_dict, notice_list = run_for_client(API_KEY, str(HISTORY_PATH))
            except Exception as e:
                st.error(f"❌ Error: {e}")
                raise

        # 显示提示语（如"今天是周三…"）
        if notice_list:
            st.subheader("📢")
            for notice in notice_list:
                st.warning(notice)

        # 显示每个模型的 top 和 bot 币种推荐
        for model_name, result in results_dict.items():
            st.subheader(f"📊 Model: {model_name}")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("🟢 **Top 20 Long Strategy Suggestions**")
                for token in result["top"]:
                    st.markdown(
                        f"<div style='background-color:#eafaf1; color:#1e5631; padding:10px; border-radius:8px; margin:4px 0; text-align:left; font-size:16px;'>{token}</div>",
                        unsafe_allow_html=True
                    )

            with col2:
                st.markdown("🔴 **Bottom 20 Short Suggestions**")
                for token in result["bot"]:
                    st.markdown(
                        f"<div style='background-color:#fdecea; color:#8a1c1c; padding:10px; border-radius:8px; margin:4px 0; text-align:left; font-size:16px;'>{token}</div>",
                        unsafe_allow_html=True
                    )


# ============================================================
# === Footer ===
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 12px;'>"
    "Built with ❤️ using Streamlit | Powered by Claude AI | "
    "⚠️ This is not financial advice"
    "</div>",
    unsafe_allow_html=True
)

# streamlit run main_app.py
