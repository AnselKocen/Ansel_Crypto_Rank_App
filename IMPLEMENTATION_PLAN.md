# Ansel Crypto Rank App 重构实施计划

## 概述
本计划将对现有Streamlit应用进行优化和重构，包括Bug修复、Tab重排序、AI助手模块拆分。

---

## 任务1：Bug修复（精简版）

### 1.1 NLTK下载优化
**问题：** 当前NLTK下载代码在模块顶层执行（`main_app.py:16-29`），每次import都会触发检查。

**修改文件：** `main_app.py`

**修改方案：**
```python
# 将NLTK下载逻辑封装为函数，只在首次需要时执行
import nltk
from functools import lru_cache

@lru_cache(maxsize=1)
def ensure_nltk_data():
    """确保NLTK数据包已下载，只执行一次"""
    required_packages = ['punkt', 'stopwords', 'wordnet', 'vader_lexicon', 'punkt_tab']
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
        except LookupError:
            nltk.download(package, quiet=True)

# 在实际需要时才调用
# ensure_nltk_data()  # 移到sentiment更新函数内部调用
```

### 1.2 TradingView符号映射完善
**问题：** `UNSUPPORTED_SYMBOLS`和`SYMBOL_MAPPING`不够完整，需要更新。

**修改文件：** `main_app.py`

**修改方案：**
- 扩展`SYMBOL_MAPPING`字典，添加更多常见映射
- 更新`UNSUPPORTED_SYMBOLS`集合
- 添加动态检测机制：如果符号在Binance不存在，自动跳过

### 1.3 时区明确为UTC
**问题：** 代码中使用`datetime.today()`没有明确时区。

**修改文件：** `main_app.py`, `sentiment_update.py`, `customer_side.py`

**修改方案：**
```python
from datetime import datetime, timezone

# 替换所有 datetime.today() 为
datetime.now(timezone.utc)

# 替换所有 datetime.strptime(...).date() 后添加时区意识
```

---

## 任务2：Tab顺序重排

### 新顺序
| 序号 | Tab名称 | 说明 |
|-----|---------|-----|
| 1 | 🎓 Educational | 保持不变 |
| 2 | 💭 Market Sentiment | 情绪词云展示 |
| 3 | 🤖 AI Analysis | AI市场分析报告（新模块A）|
| 4 | 💬 AI Q&A | AI问答（新模块B）|
| 5 | 📈 Live Prices | 价格展示 |
| 6 | 💰 Investment | 投资建议 |
| 7 | 📝 Model Strategy | 模型策略介绍 |
| 8 | 🛠️ Hyper-Parameters | 超参数展示 |
| 9 | 🔍 Feature Selection | 特征选择 |
| 10 | 📊 Backtest Results | 回测结果 |
| 11 | ▶️ Prediction | 运行预测 |

**修改文件：** `main_app.py`

---

## 任务3：AI助手模块重构

### 3.1 模块A - AI市场分析报告

#### 布局设计（三栏）
```
┌────────────────────────────────────────────────────────────────────┐
│ Skills列表 (~25%)  │ Skills详情 (~25%)  │ AI分析报告 (~50%)       │
│                    │                    │                          │
│ [可折叠分类列表]    │ [Markdown渲染]     │ [Markdown报告预览]       │
│ - 📖 Fundamentals  │                    │                          │
│ - ₿ Bitcoin       │ 点击左侧skill后    │ - 自动生成的周度报告      │
│ - 📈 Trading      │ 在此显示详细内容    │ - 基于最近2周价格数据     │
│ - 🔒 Security     │                    │ - 包含新闻分析            │
│ - 🪙 Tokens       │                    │                          │
│                    │                    │ [📥 下载报告] 按钮        │
└────────────────────────────────────────────────────────────────────┘
```

#### 功能实现

**新建文件：** `ai_market_analysis.py`

```python
# 核心功能：
1. load_price_history() - 从df_merged_history.csv读取最近2周Top20币价格
2. search_crypto_news() - 调用Claude搜索最近2周加密货币新闻
3. generate_analysis_report() - 生成Markdown分析报告
4. check_report_freshness() - 检查报告是否需要更新（每周三）
5. save_report() / load_report() - 报告持久化
```

**报告内容结构：**
```markdown
# 加密货币市场周度分析报告
**生成日期：** 2024-XX-XX (UTC)
**数据范围：** 2024-XX-XX 至 2024-XX-XX

## 一、市场概览
- Top 20 币种整体表现
- 涨跌幅排名

## 二、价格走势分析
- 本周vs上周对比
- 关键价格变动

## 三、市场新闻摘要
- 重要新闻事件（5-10条）
- 市场情绪解读

## 四、技术指标信号
- 基于模型特征的分析
- 动量/波动率变化

## 五、下周展望
- 潜在关注点
- 风险提示

---
*本报告由AI自动生成，仅供参考，不构成投资建议*
```

**Claude API调用：**
```python
# 使用 claude-sonnet-4-20250514 (你说的opus4.5，实际最新是claude-sonnet-4-20250514)
# 系统提示词包含：
# - 项目背景和数据来源
# - Skills知识库作为参考
# - 搜索最近2周新闻的指令
# - 报告格式要求
```

**自动更新逻辑：**
```python
def should_update_report():
    """判断是否需要更新报告（每周三UTC）"""
    now = datetime.now(timezone.utc)
    # 找到本周三
    days_since_wednesday = (now.weekday() - 2) % 7
    this_wednesday = now - timedelta(days=days_since_wednesday)

    # 检查上次更新时间
    last_update = load_last_update_time()
    return last_update < this_wednesday
```

#### 修改文件清单
| 文件 | 操作 | 说明 |
|-----|------|-----|
| `ai_market_analysis.py` | 新建 | 市场分析报告核心逻辑 |
| `main_app.py` | 修改 | 添加Tab3 UI |
| `figures/weekly_report.md` | 新建 | 存储最新报告 |
| `last_updated_report.txt` | 新建 | 记录报告更新时间 |

---

### 3.2 模块B - AI问答

#### 布局设计（三栏）
```
┌────────────────────────────────────────────────────────────────────┐
│ Skills列表 (~25%)  │ Skills详情 (~25%)  │ 对话框 (~50%)           │
│                    │                    │                          │
│ [可折叠分类列表]    │ [Markdown渲染]     │ ┌────────────────────┐  │
│ - 📖 Fundamentals  │                    │ │ 对话历史            │  │
│ - ₿ Bitcoin       │ 点击左侧skill后    │ │ User: ...          │  │
│ - 📈 Trading      │ 在此显示详细内容    │ │ AI: ...            │  │
│ - 🔒 Security     │                    │ │                    │  │
│ - 🪙 Tokens       │                    │ └────────────────────┘  │
│                    │                    │                          │
│                    │                    │ [预设问题按钮行]         │
│                    │                    │ [💬 输入框] [发送]       │
└────────────────────────────────────────────────────────────────────┘
```

#### 功能实现

**修改文件：** `claude_assistant.py`

**增强系统提示词：**
```python
ENHANCED_SYSTEM_PROMPT = """
## 项目详细介绍

### 项目名称
Ansel Crypto Rank App - ML驱动的加密货币周度推荐系统

### 实现方式
1. **数据获取**：通过CoinDesk API获取Top 100币种的OHLCV数据和新闻
2. **特征工程**：计算技术指标（动量、波动率、交易量冲击等）+ VADER情绪分析
3. **模型训练**：使用52周滚动窗口训练ElasticNet和ExtraTrees模型
4. **预测输出**：每周四生成Top 20（做多）和Bottom 20（做空）推荐

### 四个模型及其意义
| 模型 | 特征集 | 说明 |
|-----|-------|-----|
| enet_all | 市场+情绪 | ElasticNet使用全部特征，自动特征选择 |
| enet_market | 仅市场 | ElasticNet仅用技术指标，排除情绪噪声 |
| extra_all | 市场+情绪 | ExtraTrees使用全部特征，捕捉非线性 |
| extra_market | 仅市场 | ExtraTrees仅用技术指标 |

### 项目文件结构
- main_app.py: Streamlit Web UI
- customer_side.py: 预测管道
- crypto_project_pipeline.py: ETL和特征工程
- claude_assistant.py: AI助手
- skills/: 39个教育知识模块
- figures/: 可视化图表
- df_merged_history.csv: 历史数据（~19M）

### 更新周期
- 预测：每周四
- 情绪词云：每日
- 分析报告：每周三

### 功能模块
1. 教育资源 - 加密货币基础知识
2. 情绪分析 - 词云和恐惧贪婪指数
3. AI分析 - 自动生成的市场报告
4. AI问答 - 你正在使用的功能
5. 实时价格 - TradingView行情
6. 投资建议 - 策略推荐
7. 模型详情 - 超参数、特征、回测

{skills_knowledge}
"""
```

**预设问题：**
```python
PRESET_QUESTIONS = [
    {
        "label": "📋 介绍这个项目",
        "question": "请详细介绍这个加密货币推荐项目，包括实现方式、功能和各模型的意义",
        "type": "fixed"  # 固定回答，不调用API
    },
    {
        "label": "📰 最新市场动态",
        "question": "请介绍最新的加密货币市场动态，总结一段话并列出5-10篇相关新闻链接",
        "type": "search"  # 需要联网搜索
    },
    {
        "label": "🎯 如何使用推荐",
        "question": "如何解读和使用这个系统的推荐结果？",
        "type": "context"  # 基于上下文回答
    },
    {
        "label": "⚠️ 风险提示",
        "question": "加密货币投资有哪些风险？",
        "type": "context"
    }
]
```

**固定回答（介绍项目）：**
```python
FIXED_PROJECT_INTRO = """
# Ansel Crypto Rank App 项目介绍

## 🎯 项目目标
这是一个**机器学习驱动的加密货币周度推荐系统**，帮助投资者发现潜在的投资机会。

## 🔧 实现方式

### 数据源
- **价格数据**：CoinDesk API，获取Top 100币种的每日OHLCV（开高低收量）
- **新闻数据**：CoinDesk新闻API，用于情绪分析

### 特征工程
系统计算以下技术指标作为预测特征：
- 📈 **动量指标**：14/21/28/42/90天动量
- 📊 **波动率**：14/21/28/42/90天年化波动率
- 💹 **交易量冲击**：7/14/21/28/42天交易量变化
- 🕯️ **技术形态**：长烛线比率、短期反转信号
- 😊 **情绪指标**：VADER情绪分析（含700+加密专用词汇）

### 模型训练
- **训练窗口**：滚动52周历史数据
- **预测目标**：下周收益率
- **更新频率**：每周四（基于周三收盘数据）

## 🤖 四个预测模型

| 模型名称 | 算法 | 特征集 | 特点 |
|---------|------|-------|------|
| `enet_all` | ElasticNet | 市场+情绪 | L1/L2正则化，自动特征选择 |
| `enet_market` | ElasticNet | 仅市场 | 纯技术面，排除情绪噪声 |
| `extra_all` | ExtraTrees | 市场+情绪 | 集成学习，捕捉非线性关系 |
| `extra_market` | ExtraTrees | 仅市场 | 纯技术面，稳定性更高 |

## 📊 输出结果
- **Top 20**：预测收益最高的币种（做多建议）
- **Bottom 20**：预测收益最低的币种（做空建议）

## 💡 使用建议
根据回测结果，`extra_all` + Long-Short策略的Sharpe比率最高（2.20），适合风险调整后收益最大化。

---
⚠️ **免责声明**：本系统仅供研究和参考，不构成投资建议。加密货币投资风险极高，请谨慎决策。
"""
```

#### 修改文件清单
| 文件 | 操作 | 说明 |
|-----|------|-----|
| `claude_assistant.py` | 修改 | 增强系统提示词、添加预设问题处理 |
| `main_app.py` | 修改 | 添加Tab4 UI |

---

## 实施顺序

### Phase 1: 基础修复（任务1）
1. [x] 修改 `main_app.py` - NLTK优化 ✅
2. [x] 修改 `main_app.py` - TradingView映射 ✅
3. [x] 修改多文件 - UTC时区统一 ✅

### Phase 2: Tab重排（任务2）
4. [x] 修改 `main_app.py` - 调整Tab顺序和变量名 ✅

### Phase 3: AI模块A（任务3.1）
5. [x] 新建 `ai_market_analysis.py` - 核心逻辑 ✅
6. [x] 修改 `main_app.py` - 添加AI Analysis Tab UI ✅

### Phase 4: AI模块B（任务3.2）
7. [x] 修改 `claude_assistant.py` - 增强系统提示词 ✅
8. [x] 修改 `main_app.py` - 添加AI Q&A Tab UI ✅

### Phase 5: 测试验证
9. [x] 语法检查通过 ✅
10. [ ] 用户本地运行测试

---

## 风险和注意事项

### API成本
- Claude API调用需要付费
- 市场分析报告每周生成1次，成本可控
- 问答功能按对话计费，建议设置max_tokens限制

### 性能考虑
- `df_merged_history.csv`（19M）读取可能较慢
- 建议只读取需要的列和最近的数据
- 报告生成建议异步处理，避免阻塞UI

### 兼容性
- Streamlit Cloud部署时需要secrets配置API Key
- 确保所有新依赖添加到requirements.txt

---

## 预计工时

| 阶段 | 预计时间 |
|-----|---------|
| Phase 1 | 30分钟 |
| Phase 2 | 15分钟 |
| Phase 3 | 2小时 |
| Phase 4 | 1.5小时 |
| Phase 5 | 30分钟 |
| **总计** | **~4.5小时** |

---

## 确认清单

请确认以下内容后开始实施：

- [ ] Tab顺序确认（11个Tab）
- [ ] AI模型使用 `claude-sonnet-4-20250514`（当前最新sonnet）
- [ ] 报告自动更新时间：每周三 UTC
- [ ] Skills在模块A和模块B都显示（左25% + 中25%）
- [ ] 对话框/报告占右侧50%
- [ ] 继续使用Streamlit，暂不迁移

---

*计划编写日期：2024-XX-XX*
*版本：v1.0*
