"""Reviewed bilingual quick answers. Rendering these never calls a model."""
from copy import deepcopy


PRESETS = [
    {
        "id": "intro", "label": "📋 介绍这个项目", "label_en": "📋 About This Project",
        "type": "fixed",
        "response_zh": """### 👋 认识 Ansel Crypto Rank
这是一个用机器学习整理**加密货币每周排名**的研究网站。

- 📊 **看排名**：比较四个模型对币种的预测排序。
- 🌤️ **看情绪**：通过新闻词云和情绪仪表了解市场氛围。
- 📚 **学知识**：阅读加密基础资料，或向 Kimi 提问。

🗓️ 每周四北京时间 **09:00 起**自动更新；实际数据日期以页面标注为准。

💡 排名和情绪指标供学习研究参考，不保证未来收益。""",
        "response_en": """### 👋 Meet Ansel Crypto Rank
A research website that uses machine learning to create **weekly cryptocurrency rankings**.

- 📊 **Explore rankings**: compare predictions from four models.
- 🌤️ **Check sentiment**: view news word clouds and a sentiment gauge.
- 📚 **Learn**: browse crypto guides or ask Kimi a question.

🗓️ Automatic updates become due on **Thursday at 09:00 Beijing time**. Check the date shown on the page for the published data.

💡 Rankings and sentiment indicators are research tools, with no guarantee of future returns.""",
    },
    {
        "id": "interpret", "label": "🎯 如何使用推荐", "label_en": "🎯 How to Use Recommendations",
        "type": "fixed",
        "response_zh": """### 🎯 推荐结果这样看
1. 🤖 **先选模型**：`all` 包含市场与情绪特征；`market` 只用市场特征。
2. 📈 **再看排序**：Top 20 是预测相对靠前的币；Bottom 20 是相对靠后的币，不代表必须做空。
3. 🕒 **核对日期**：确认数据截止日和预测区间，避免把旧结果当成实时行情。
4. 🔎 **综合判断**：结合流动性、项目信息和自身风险承受能力研究。

💡 预测收益率不是已实现收益，也不是上涨概率；排名不能替代你的决策。""",
        "response_en": """### 🎯 Reading the recommendations
1. 🤖 **Choose a model**: `all` includes market and sentiment features; `market` uses market features only.
2. 📈 **Read the ranking**: Top 20 contains higher-ranked predictions; Bottom 20 contains lower-ranked ones, not an instruction to short them.
3. 🕒 **Check the dates**: confirm the data cutoff and forecast period. Older results are not live prices.
4. 🔎 **Do your research**: consider liquidity, project information and your own risk tolerance.

💡 A predicted return is neither a realized return nor a probability of a price increase. Rankings do not replace your judgment.""",
    },
    {
        "id": "risk", "label": "⚠️ 投资风险提示", "label_en": "⚠️ Investment Risks",
        "type": "fixed",
        "response_zh": """### ⚠️ 先了解风险
- 🌊 **价格与流动性**：价格可能剧烈波动，成交价格也可能偏离预期。
- 🧮 **模型有误差**：预测会出错，历史表现不代表未来收益。
- ⚖️ **杠杆放大风险**：可能加速亏损并触发强制平仓。
- 🔐 **资产安全**：留意平台、私钥、助记词和诈骗风险。

💡 本网站提供学习与研究信息，不构成投资建议。投入前，请先判断自己是否能够承受相应损失。""",
        "response_en": """### ⚠️ Understand the risks first
- 🌊 **Price and liquidity**: sharp swings and slippage can affect your results.
- 🧮 **Model error**: predictions can be wrong; past performance does not guarantee future returns.
- ⚖️ **Leverage**: it can amplify losses and trigger liquidation.
- 🔐 **Asset security**: consider platform, private-key, recovery-phrase and fraud risks.

💡 This website provides educational and research information, not investment advice. Consider whether you can afford the possible loss before committing funds.""",
    },
    {
        "id": "models", "label": "🤖 模型区别说明", "label_en": "🤖 Model Differences",
        "type": "fixed",
        "response_zh": """### 🤖 四个模型，两个区别
**算法不同**：`enet` 是 ElasticNet 线性模型；`extra` 是 ExtraTrees 树模型，可捕捉非线性关系。

**输入不同**：`all` 包含市场与情绪特征；`market` 只用市场特征。

| 模型 | 算法 | 输入 |
|---|---|---|
| `enet_all` | ElasticNet | 市场 + 情绪 |
| `enet_market` | ElasticNet | 市场 |
| `extra_all` | ExtraTrees | 市场 + 情绪 |
| `extra_market` | ExtraTrees | 市场 |

🔎 可以比较不同模型的排名，观察一致与分歧。没有一个模型能保证始终最好。""",
        "response_en": """### 🤖 Four models, two differences
**Algorithm**: `enet` uses the linear ElasticNet model; `extra` uses ExtraTrees, which can capture nonlinear relationships.

**Inputs**: `all` includes market and sentiment features; `market` uses market features only.

| Model | Algorithm | Inputs |
|---|---|---|
| `enet_all` | ElasticNet | Market + sentiment |
| `enet_market` | ElasticNet | Market |
| `extra_all` | ExtraTrees | Market + sentiment |
| `extra_market` | ExtraTrees | Market |

🔎 Compare rankings to see where models agree or differ. No model is guaranteed to perform best all the time.""",
    },
    {
        "id": "crypto_basics", "label": "🪙 到底什么是加密货币", "label_en": "🪙 What Is Cryptocurrency?",
        "type": "fixed",
        "response_zh": """### 🪙 到底什么是加密货币？
你可以先把它理解为：**一种在计算机网络上记录和转移、用密码学验证控制权的数字资产。** 比特币（BTC）和以太坊的 ETH 都是例子，但不同币的用途、规则和风险并不相同。

#### 📒 区块链：大家共同核对的账本
银行转账由银行更新账本；在比特币等公有链上，许多独立计算机按照共同规则验证交易、保存记录。区块把交易按顺序连接起来，因此叫“区块链”。

例如小李给小王转 BTC：小李的钱包用私钥签名，网络验证后记录转账，小王等待确认。通常会有网络手续费，确认速度和费用取决于网络情况，并不总是即时或免费。

“加密”主要指使用密码学保护和验证控制权，**不等于交易记录全部保密**；很多公有链的地址与交易记录都能公开查询。

#### 🧩 常见的几类资产
- **BTC**：比特币网络的原生资产，可用于转账和持有。
- **ETH**：以太坊网络的原生资产，用于支付网络操作费用；以太坊还支持智能合约和应用。
- **稳定币**：尝试跟踪美元等资产的价值，但可能脱锚，也有发行方、储备或机制风险，不能当作无风险存款。

#### 👛 钱包与币是什么关系？
币的归属记录在网络账本上，钱包负责查看余额、管理密钥和签署交易。钱包地址用于收款；私钥用于授权操作；助记词通常用于恢复钱包，必须保密。

#### 💡 为什么有人使用它？
用途包括数字资产转账、参与链上应用和资产交易。买卖需求、用途和市场预期等会影响价格，技术有用不代表某个币一定升值。

先弄懂“**资产、网络、钱包、密钥**”这四件事，就已经迈出了第一步，不必急着购买。

📚 延伸阅读：[比特币如何运作](https://bitcoin.org/en/how-it-works) · [认识以太坊](https://ethereum.org/what-is-ethereum/) · [稳定币介绍](https://ethereum.org/stablecoins/)""",
        "response_en": """### 🪙 What is cryptocurrency?
Start with this idea: **a digital asset recorded and transferred over a computer network, with cryptography used to verify control.** Bitcoin (BTC) and Ethereum's ETH are examples, but assets differ in purpose, rules and risk.

#### 📒 Blockchain: a shared ledger
A bank updates its own transfer records. On public networks such as Bitcoin, independent computers follow shared rules to check transactions and retain records. Transactions are grouped into linked blocks—the blockchain.

Suppose Alice sends BTC to Bob: her wallet signs with a private key, the network validates and records the transfer, and Bob waits for confirmation. Network fees and confirmation times vary; transfers are not always instant or free.

“Crypto” refers to cryptographic verification. **It does not mean every transaction is secret.** Addresses and transactions on many public blockchains can be inspected publicly.

#### 🧩 Common asset types
- **BTC**: Bitcoin's native asset, which can be transferred or held.
- **ETH**: Ethereum's native asset, used to pay for network operations. Ethereum also supports smart contracts and applications.
- **Stablecoins**: assets designed to track a value such as the US dollar. They can lose their peg and carry issuer, reserve or mechanism risks; they are not risk-free deposits.

#### 👛 Where does a wallet fit?
Ownership is recorded on the network. A wallet displays balances, manages keys and signs transactions. An address receives assets; a private key authorizes actions; a recovery phrase commonly restores a wallet and must remain secret.

#### 💡 Why use it?
Uses include transfers, blockchain applications and asset trading. Demand, utility and expectations influence prices; useful technology does not guarantee an asset will appreciate.

Understanding **assets, networks, wallets and keys** is a useful first step. There is no need to rush into a purchase.

📚 Further reading: [How Bitcoin works](https://bitcoin.org/en/how-it-works) · [Ethereum explained](https://ethereum.org/what-is-ethereum/) · [Stablecoins](https://ethereum.org/stablecoins/)""",
    },
    {
        "id": "get_crypto", "label": "👛 如何获得加密货币", "label_en": "👛 How Can I Get Crypto?",
        "type": "fixed",
        "response_zh": """### 👛 如何获得加密货币？从钱包说起
常见方式是**购买**，或**接收别人转给你的资产**，例如礼物或商品、服务的付款。创建钱包本身不会自动获得币；先了解所在地规则，以及哪些服务对你可用。

#### 1️⃣ 先分清：平台账户与自己的钱包
- **托管平台账户**：平台管理密钥，你通过账户操作，使用方便，但要承担平台风险及其提现限制。
- **自托管钱包**：自己管理密钥和备份，更直接地控制资产，也要自己承担保管、转账和恢复责任。

软件钱包适合日常操作；硬件钱包可让私钥在设备内离线保存，但仍需保护备份并核对签名内容。不是所有钱包都支持所有币或网络。

#### 2️⃣ 如果使用自托管钱包，先做好备份
通过产品官网核对下载入口。创建钱包后，按照官方流程安全备份；若使用助记词，**不要截图上云、发给别人，或交给自称客服的人**。设备丢失不一定意味着资产丢失，有效备份通常可用于恢复；设备和所有有效备份都无法使用时，可能永久失去访问权。

#### 3️⃣ 购买或收款
若选择购买，先了解可用平台的费用、验证要求、支持币种和提币规则。购买后留在托管平台的余额，与提到自托管钱包的链上资产，管理方式不同。

常见操作顺序是：**注册账户 → 按要求完成身份验证 → 使用支持的付款方式入金 → 核对币种、金额和费用后购买**。能否直接用当地货币购买某个币，取决于平台服务与交易对。

若接收转账，向对方提供**正确网络上的收款地址**；正常收款不需要告诉对方私钥或助记词。到账以自己的钱包记录和网络确认为准，不凭截图判断。

#### 4️⃣ 转账前，核对三件事
**币种、网络、地址**都要匹配，平台要求的 Memo／Tag 也不能遗漏。即使地址长得一样，也不代表任意网络都能接收。条件允许时先小额测试，再按需转账；转错地址或不兼容网络，可能无法找回。

链上发送通常需要网络手续费；例如以太坊主网上的操作使用 ETH 支付 gas，不一定能直接从要转出的代币中扣费。

💡 可以先熟悉钱包和小额收发流程，不必把学习入门变成大额投资。

📚 延伸阅读：[获取比特币](https://bitcoin.org/en/getting-started) · [钱包介绍](https://ethereum.org/wallets/) · [安全基础](https://ethereum.org/security/) · [备份与恢复](https://trezor.io/support/troubleshooting/trezor-suite-issues/troubleshoot-wallet-backup-and-recovery-problems)""",
        "response_en": """### 👛 How can I get crypto? Start with wallets
Common routes are **buying it** or **receiving a transfer**, such as a gift or payment for goods or services. Creating a wallet does not give you coins. First check local rules and which services are available to you.

#### 1️⃣ Platform account or your own wallet?
- **Custodial platform account**: the provider manages the keys. You operate through an account and depend on its security and withdrawal rules.
- **Self-custody wallet**: you manage the keys and backups, taking responsibility for safekeeping, transfers and recovery.

Software wallets support everyday use. Hardware wallets can keep private keys offline inside the device, but backups and transaction checks still matter. Wallets do not support every asset or network.

#### 2️⃣ Back up a self-custody wallet first
Verify downloads through the product's official website and follow its backup process. If it uses a recovery phrase, **do not upload screenshots, share it, or give it to someone claiming to be support**. A valid backup can usually restore access after device loss; losing access to every device and usable backup may make recovery impossible.

#### 3️⃣ Buy or receive
If buying, check the available platform's fees, verification requirements, supported assets and withdrawal rules. A custodial account balance is managed differently from assets withdrawn to your own wallet.

A common sequence is: **create an account → complete any required identity checks → fund it using a supported method → confirm the asset, amount and fees before buying**. Direct purchases with local currency depend on the service and available trading pairs.

To receive, share your **receiving address on the correct network**, not your private key or recovery phrase. Confirm receipt in your own wallet and through network confirmations, not someone else's screenshot.

#### 4️⃣ Check before sending
Match the **asset, network and address**, plus any required Memo/Tag. An identical-looking address does not mean every network is supported. Where practical, test a small transfer before sending more; mistakes may be unrecoverable.

Sending on-chain usually requires a network fee. For example, operations on Ethereum mainnet use ETH for gas; fees cannot necessarily be deducted from the token you are sending.

💡 Learn the wallet and transfer process before committing significant funds.

📚 Further reading: [Getting Bitcoin](https://bitcoin.org/en/getting-started) · [Wallets](https://ethereum.org/wallets/) · [Security basics](https://ethereum.org/security/) · [Backup and recovery](https://trezor.io/support/troubleshooting/trezor-suite-issues/troubleshoot-wallet-backup-and-recovery-problems)""",
    },
]


def get_presets() -> list[dict]:
    return deepcopy(PRESETS)


def get_preset(preset_id: str) -> dict | None:
    return next((deepcopy(preset) for preset in PRESETS if preset["id"] == preset_id), None)


def fixed_answer(preset_id: str, language: str) -> str | None:
    preset = get_preset(preset_id)
    return preset["response_en" if language == "en" else "response_zh"] if preset else None
