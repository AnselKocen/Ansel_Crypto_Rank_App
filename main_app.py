import pandas as pd
import streamlit as st
from PIL import Image

from pathlib import Path
import subprocess
import base64
import streamlit.components.v1 as components
from datetime import datetime
from customer_side import run_prediction_pipeline




import nltk
# 定义需要的所有数据包
required_nltk_packages = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon']

# 直接循环下载，NLTK会自动跳过已存在的数据包
print("正在检查并准备NLTK数据包...")
for package in required_nltk_packages:
    nltk.download(package)

print("所有数据包准备就绪！")
try:
    nltk.download('punkt_tab')
except Exception as e:
    st.error(f"fail downloading 'punkt_tab' : {e}")


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


# === 顶部 Tabs 页面结构 ===
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "🎓Educational",
    "📝Model Strategy",
    "🛠️Hyber-Parameters",
    "🔍Feature Selection",
    "📊Backtest Results",
    "💭Market Sentiment",
    "▶️Prediction",
    "💰Investment",
    "😊💬Assistant"
])

# === 页面 1：加密货币基础 ===
with tab1:
    st.header("📚 Crypto Basics & Educational Resources")

    st.markdown("Here are some beginner-friendly learning resources. You can also use our 😊💬Assistant for simple Q&A support.")

    # 🔗 教学链接
    st.markdown("🔗 [Binance Academy – What Is Cryptocurrency?](https://academy.binance.com/en/articles/what-is-a-cryptocurrency)")
    st.markdown("🔗 [CoinMarketCap：Stay Updated on the Latest Cryptocurrency Price Trends](https://coinmarketcap.com/)")
    st.markdown("🔗 [educational YouTuber：@simplyexplained](https://www.youtube.com/@simplyexplained)")
    st.markdown("🔗 [What is long and short?](https://www.youtube.com/watch?v=fXnCtGcvqdk&t=31s)")

    # 🎥 教学视频嵌入
    st.subheader("🎥 What is crypto currency🪙?")
    st.video("https://www.youtube.com/watch?v=Zoz9gvhLgpM")


# === 页面 2：模型策略介绍 ===
emoji_map = {
    "market": "🔵",
    "all": "🟢",
    "enet_EW": "🟡",
    "extra_EW": "🟠",
    "enet_ls": "🔴",
    "extra_ls": "🟣",
    "fusion_ls": "🟤"
}

with tab2:
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


# === 页面 3：最优参数展示 ===
with tab3:
    st.header("⚙️ Optimal Hyber-Parameters Display")
    param_path = text_dir / "best_params.txt"
    if param_path.exists():
        with open(param_path, "r", encoding="utf-8") as f:
            st.code(f.read(), language="yaml")
    else:
        st.info("no best_params.txt ")



# === 页面 4：特征选择 ===
with tab4:
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

# === 页面 5：策略回测结果对比 ===
with tab5:
    st.header("📈 Strategy Backtest Results Comparison")
    tab = st.radio(
        label="Comparison Metric Options",
        options=["Cumulative Return", "Volatility", "Average Return", "Sharpe Ratio", "Maximum Drawdown"],
        horizontal=True,
        label_visibility="collapsed"
    )
    base_name_map = {
        "Cumulative Return": "cum_return_comparison",
        "Volatility": "volatility_bar",
        "Average Return": "mean_return_bar",
        "Sharpe Ratio": "sharpe_bar",
        "Maximum Drawdown": "max_drawdown_bar"
    }
    base_name = base_name_map[tab]
    variants = ["all", "market"]
    for variant in variants:
        img_name = f"{base_name}_{variant}_with_baseline.png"
        img_path = fig_dir / img_name
        if "cum_return" in img_name:
            show_centered_img(img_path, width_percent=71)
        else:
            show_centered_img(img_path, width_percent=51)

# === 页面 6：市场情绪指数 ===
with tab6:
    st.header("📰 Market Sentiment Index over the last 7-days ")
    update_file = BASE_DIR / "last_updated_wordcloud.txt"

    # 如果文件存在，读取日期
    if update_file.exists():
        last_updated_str = update_file.read_text().strip()
        st.info(f"📅 Last updated: {last_updated_str}")  # ✅ 永久显示
        # 转为日期进行比较
        try:
            last_updated_date = datetime.strptime(last_updated_str, "%Y-%m-%d").date()
            today = datetime.today().date()

            # ✅ 如果不是今天，提示用户需要更新
            if last_updated_date < today:
                st.warning("⚠️ This data may be outdated.")
        except ValueError:
            st.error("❌ Invalid update date format in last_updated_wordcloud.txt.")
    else:
        st.info("ℹ️ Word cloud and Gauge have not been generated yet.")

    if st.button("▶️ Update Word Cloud and Fear & Greed Gauge - If you are in mainland China, you may need a **VPN** to perform the update."):
        with st.spinner("Generating... This may take up to 5 minutes."):
            try:
                from sentiment_update import update_sentiment_and_gauge
                update_sentiment_and_gauge()
                st.success("✅ Updated successfully!")
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
# === 页面 7：运行函数 ===
from customer_side import run_for_client  # ✅ 直接导入函数

with tab7:
    st.header("🧠 Get Next Week's Recommended Portfolio ! ")
    st.markdown(
        "🟦 Click the button to automatically fetch crypto market and news data up to the most recent Wednesday (t), merge it with historical data, and perform ETL.  \n"
        "🟨 Models are trained on the past 52 weeks (t−52 to t−1), and using features observed in week t to rank expected returns for week t+1.  \n"
        "🟥 **Disclaimer: For reference only. Not financial advice.**"
    )

    if st.button("▶️ Click to Get Recommended Tokens"):
        st.session_state["log_expanded"] = False

        with st.spinner("Running, please wait... Might take 5–10 mins if you haven't run this page in a while"):
            try:
                API_KEY = ""  # ← 可改为 st.secrets["API_KEY"]
                HISTORY_PATH = BASE_DIR / "df_merged_history.csv"
                results_dict, notice_list = run_for_client(API_KEY, str(HISTORY_PATH))
            except Exception as e:
                st.error(f"❌ Error: {e}")
                raise

        # ✅ 显示提示语（如“今天是周三…”）
        if notice_list:
            st.subheader("📢")
            for notice in notice_list:
                st.warning(notice)

        # ✅ 显示每个模型的 top 和 bot 币种推荐
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

with tab8:
    import pandas as pd

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
#############################################
with tab9:
    import streamlit as st

    import streamlit as st

    st.title("💡 Strategy Assistant ChatBot ✨")
    st.markdown("Click a question below to quickly understand how our platform works and get friendly strategy advice!")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Full Q&A dictionary with updated questions
    qa_pairs = {
        "📈 What is long / short / LS / EW?":
            "Long means buying assets expected to rise 📈, short means betting on decline 📉. LS combines both! EW (Equal Weight) splits positions evenly for stability 🧺",

        "🧠 What's the difference between Enet and Extra?":
            "`Enet` is a linear model for simple relationships, while `Extra` is nonlinear and captures complex market patterns 🤖🌐",

        "🏷️ What do all and market tags mean?":
            "`all` includes sentiment features; `market` only uses market features like volatility and momentum 💬📊",

        "📚 How can I learn about crypto by myself?":
            "You can check out these beginner-friendly resources — I’ve handpicked them just for you! 🎓💡\n\n"
            "🔗 [How Cryptocurrency ACTUALLY works](https://www.youtube.com/watch?v=rYQgy8QDEBI)\n"
            "🔗 [How To Invest in Crypto as A COMPLETE Beginner [2025 GUIDE]](https://www.youtube.com/watch?v=LGHsNaIv5os)\n"
            "🔗 [Blockchain Technology Explained (2 Hour Course)](https://www.youtube.com/watch?v=qOVAbKKSH10)",

        "💸 How do I start crypto trading?":
            "Register on an exchange like **Binance**, deposit fiat or USDT, and you’re ready to trade! 🚀",

        "🕒 When are weekly token recommendations updated?":
            "Every Thursday at 12:00am ✅ (we wait for complete Wednesday data) 📅",

        "🧊 I don’t like risk":
            "We recommend low-volatility strategies like `fusion_ls` or `market_EW` — calm and steady! 🛡️",

        "🔥 I want high returns":
            "Try high-return strategies like `extra_EW` or `extra_ls`, but watch out for volatility! ⚡",

        "🤷 I’m not sure about my risk preference":
            "No worries! Start with the strategy that has the highest Sharpe ratio — it balances return and risk 📈⚖️",

        "❓ I don’t understand crypto at all":
            "Crypto are decentralized digital assets like BTC and ETH. We recommend watching our 🎓 Education videos for a quick start!",

        # ✅ Newly added questions
        "💰 How many recommended tokens should I buy?":
            "👉 We suggest using the **Top 20 tokens** from your selected strategy and allocating them **equally (EW)**. This helps diversify risk and keep things simple 🧺📊",

        "🪙 How do I actually buy the recommended tokens?":
            "You’ll need to create an account on a crypto exchange like **Binance**, deposit funds (like USDT), and search for each token to trade. Super easy once you're set up! 🚀",

        "⚠️ Important to know - ☁️ Why need word cloud & sentiment gauge?":
            "They reflect overall market sentiment! Remember Buffett's quote: *Be fearful when others are greedy, greedy when others are fearful* 😬🧭",

        "⚠️ Important to know - 💬 Why do we use sentiment features?":
            "Because **crypto markets are emotional!** Sentiment helps capture **non-structural signals** like hype or panic, boosting the model’s prediction power 📈🧠  \n"
            "You can also refer to the **word cloud** and **sentiment gauge** to judge market mood and entry timing 🧭☁️  \n"
            "But remember — while sentiment may help forecast returns, chasing highs or panic-selling isn’t always smart. Always invest with caution! ⚠️"
    }

    # Button layout
    cols = st.columns(3)
    buttons = list(qa_pairs.keys())

    for i, key in enumerate(buttons):
        with cols[i % 3]:
            if st.button(key):
                st.session_state.chat_history.append(("user", key))
                st.session_state.chat_history.append(("bot", qa_pairs[key]))

    # Display chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(msg)


#################### 页面8 #######################


# streamlit run C:\Users\10526\PycharmProjects\Ansel_Crypto_Rank_App\main_app.py  [ARGUMENTS]