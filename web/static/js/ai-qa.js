/* Localize the assistant interface; conversation content remains in its original language. */
(() => {
    'use strict';
    const $ = id => document.getElementById(id);
    const textFor = (zh, en) => window.SiteLocale?.text(zh, en) ?? zh;
    const currentLang = () => window.SiteLocale?.language === 'en' ? 'en' : 'zh';
    const strings = {
        welcome: ["👋 你好！我是由 Kimi 提供支持的加密货币助手。", "👋 Welcome! I'm your crypto assistant powered by Kimi."],
        welcomeHint: ['可以点击上方预设问题，或在下方输入你的问题。', 'Click a preset question above or type your own question below.'],
        skillsFailed: ['暂时无法加载知识库，请重试。', 'The knowledge base could not be loaded. Please retry.'],
        skillsEmpty: ['暂无知识内容。', 'No articles are available.'],
        presetsFailed: ['暂时无法加载预设问题。', 'Preset questions could not be loaded.'],
        retry: ['重试', 'Retry'],
        you: ['👤 你', '👤 You'],
        thinking: ['Kimi 正在思考…', 'Kimi is thinking…'],
        noResponse: ['暂未收到回复，请重试。', 'No response received. Please try again.'],
        notConfigured: ['Kimi 服务尚未配置，请联系管理员。', 'The Kimi service is not configured. Please contact the administrator.'],
        providerTimeout: ['Kimi 回复超时，请稍后重试。', 'Kimi took too long to respond. Please try again.'],
        unavailable: ['Kimi 服务暂时不可用，请稍后重试。', 'The Kimi service is temporarily unavailable. Please try again later.'],
        rateLimited: ['请求较多，请稍等片刻再试。', 'Too many requests. Please wait a moment and try again.'],
        projectGuide: ['📖 项目说明', '📖 Project guide'],
        presetSyncFailed: ['说明已显示，但暂未同步到对话。稍后发送追问时会先重试同步。', 'The guide is shown, but could not be added to the conversation. Your next question will retry synchronization first.'],
        requestFailed: ['请求未完成，请稍后重试。', 'The request could not be completed. Please try again.'],
    };
    const categories = {
        '📖 Fundamentals': '📖 基础知识',
        '₿ Bitcoin & Blockchain': '₿ 比特币与区块链',
        '📈 Trading & Investment': '📈 交易与投资',
        '🔒 Security & Wallets': '🔒 安全与钱包',
        '🪙 Tokens & ICO': '🪙 代币与首次发行',
        '🔗 Other': '🔗 其他知识',
    };
    // Presentation metadata only. The underlying articles and their source are unchanged.
    const skillsZh = {
        'author-profile-crypto-education': ['作者与加密货币教育', '了解 Marcus de Maria 的背景、教育理念与课程内容。'],
        'bitcoin-economic-incentives': ['比特币经济激励机制', '了解挖矿奖励、交易手续费与网络安全的经济机制。'],
        'bitcoin-market-analysis-and-trading-strategy': ['比特币市场分析与交易策略', '了解比特币市场结构、波动特征和交易分析方法。'],
        'bitcoin-market-cycle-analysis': ['比特币市场周期分析', '了解减半机制、历史周期与宏观市场因素。'],
        'bitcoin-proof-of-work-consensus': ['比特币工作量证明共识', '了解工作量证明、分布式账本与双花防范。'],
        'bitcoin-transaction-model': ['比特币交易模型', '了解电子货币转账、交易链验证与双花问题。'],
        'blockchain-oracle-integration': ['区块链预言机', '了解智能合约如何获取和验证链外数据。'],
        'blockchain-platform-evaluation': ['区块链平台评估', '比较不同区块链平台的机制、性能与用途。'],
        'blockchain-storage-optimization-and-spv-verification': ['区块链存储优化与轻量验证', '了解默克尔树、存储裁剪与简化支付验证。'],
        'bull-market-trading-strategy': ['牛市交易策略', '了解牛市中的回调买入与分批止盈思路。'],
        'cryptocurrency-decentralized-transfer': ['加密货币去中心化转账', '了解无需中介的点对点转账与跨境支付。'],
        'cryptocurrency-fundamentals-and-blockchain-mechanics': ['加密货币基础与区块链原理', '了解数字资产、区块链、挖矿和比特币的基本概念。'],
        'cryptocurrency-fundamentals-and-classification': ['加密货币基础与分类', '了解货币属性，以及原生币和智能合约代币的区别。'],
        'cryptocurrency-investment-strategies': ['加密货币投资策略', '了解仓位管理、分批买入、止盈与区间交易。'],
        'cryptocurrency-investment-strategy-and-due-diligence': ['投资策略与尽职调查', '了解加密项目评估方法与常见风险信号。'],
        'cryptocurrency-monetary-properties': ['加密货币的货币属性', '从稀缺性、便携性等角度理解加密货币的价值储藏属性。'],
        'cryptocurrency-portfolio-tracking': ['加密货币投资组合跟踪', '了解跨钱包和交易所的资产记录、跟踪与税务管理。'],
        'cryptocurrency-purchase-and-exchange-selection': ['加密货币购买与交易所选择', '了解法币与币币交易，以及交易平台的选择标准。'],
        'cryptocurrency-risk-assessment': ['加密货币风险评估', '了解价格波动、交易不可逆和市场结构等风险。'],
        'cryptocurrency-security-and-fraud-prevention': ['加密货币安全与防诈骗', '了解交易安全、身份保护和区块链安全机制。'],
        'cryptocurrency-valuation-and-price-determinants': ['加密货币估值与价格因素', '了解价格驱动因素、网络价值与基本面分析。'],
        'cryptocurrency-wallet-management': ['加密货币钱包管理', '了解热钱包、冷存储、备份与双重验证。'],
        'cryptocurrency-wallet-risk-management': ['加密货币钱包风险管理', '了解钱包技术风险、资产损失和实际使用中的限制。'],
        'currency-evolution-analysis': ['货币演变分析', '了解货币如何从金属货币发展到现代数字形式。'],
        'double-spend-attack-probability': ['双花攻击概率', '了解确认次数、攻击者算力与双花风险之间的关系。'],
        'early-money-systems-evolution': ['早期货币制度演变', '了解以物易物、商品货币与早期交换制度。'],
        'ethereum-smart-contract-platform': ['以太坊智能合约平台', '了解去中心化应用、智能合约与去中心化自治组织。'],
        'exchange-token-utility-analysis': ['交易所代币用途分析', '了解交易所代币的功能、费用折扣与生态价值。'],
        'historical-crypto-market-timeline': ['加密货币市场历史时间线', '回顾 2017 至 2025 年的市场阶段与关键事件。'],
        'ico-sto-ieo-investment-evaluation': ['代币发行项目投资评估', '了解 ICO、STO、IEO 的区别与项目尽职调查。'],
        'ico-sto-ieo-investment-execution': ['代币发行项目投资流程', '了解项目评估后的购买、转账与投后管理流程。'],
        'investment-risk-disclaimer-policy': ['投资风险与教育用途说明', '了解投资资料的教育性质、风险提示与使用边界。'],
        'proof-of-work-consensus': ['工作量证明与分布式共识', '了解交易历史共识、挖矿过程和链的安全性。'],
        'ripple-payment-network-protocol': ['瑞波支付网络协议', '了解 Ripple、XRP 与跨境银行支付网络。'],
        'token-utility-classification': ['代币用途分类', '了解支付型、利润分享型和债务型代币的区别。'],
        'transaction-structure-privacy': ['交易结构与隐私', '了解交易输入输出、金额合并拆分与隐私保护。'],
    };
    let chatHistory = [];
    let sending = false;
    let generation = 0;
    let activeRequest = null;
    const presets = new Map();
    let pendingPresetSync = [];

    function localNode(tag, key, className = '') {
        const node = document.createElement(tag);
        node.className = className;
        [node.dataset.zh, node.dataset.en] = strings[key];
        node.textContent = textFor(...strings[key]);
        return node;
    }
    function storageGet(key) { try { return localStorage.getItem(key); } catch { return null; } }
    function storageSet(key, value) { try { localStorage.setItem(key, value); } catch { /* Optional storage. */ } }
    function createSessionId() {
        return 'sess_' + Math.random().toString(36).slice(2) + Date.now().toString(36);
    }
    function getSessionId() {
        let id = storageGet('chat_session_id');
        if (!id) {
            id = createSessionId();
            storageSet('chat_session_id', id);
        }
        return id;
    }
    function escapeHtml(text) {
        const node = document.createElement('div');
        node.textContent = text;
        return node.innerHTML;
    }
    function markdown(text) {
        return window.marked?.parse ? window.marked.parse(text) : escapeHtml(text).replace(/\n/g, '<br>');
    }
    function localResponse(key) {
        return textFor(...(strings[key] || strings.noResponse));
    }
    function renderWelcome() {
        const welcome = document.createElement('div');
        welcome.className = 'welcome-message';
        welcome.append(localNode('p', 'welcome'), localNode('p', 'welcomeHint'));
        $('chat-messages').replaceChildren(welcome);
    }
    function appendMessage(message) {
        const container = $('chat-messages');
        container.querySelector('.welcome-message')?.remove();
        const node = document.createElement('div');
        node.className = `message ${message.role}`;
        const fixed = message.role === 'assistant' && validFixedReply(message.fixed) ? message.fixed : null;
        const guide = fixed || message.localKey === 'presetSyncFailed';
        const label = message.role === 'user' ? localNode('div', 'you', 'label')
            : guide ? localNode('div', 'projectGuide', 'label') : document.createElement('div');
        if (message.role !== 'user' && !guide) { label.className = 'label'; label.textContent = '🤖 Kimi'; }
        const content = document.createElement('div');
        content.className = `content${message.isError ? ' error' : ''}`;
        const localKey = message.role === 'assistant' && Object.hasOwn(strings, message.localKey) ? message.localKey : null;
        const value = fixed ? textFor(fixed.zh, fixed.en) : localKey ? localResponse(localKey) : message.content || '';
        if (localKey) content.dataset.localResponse = localKey;
        if (fixed) {
            content.dataset.fixedResponse = fixed.id;
            content.dataset.fixedZh = fixed.zh;
            content.dataset.fixedEn = fixed.en;
        }
        content.innerHTML = message.role === 'assistant' ? markdown(value) : escapeHtml(value);
        node.append(label, content);
        container.appendChild(node);
        if (fixed) {
            container.scrollTop += node.getBoundingClientRect().top - container.getBoundingClientRect().top - container.clientTop;
        } else container.scrollTop = container.scrollHeight;
    }
    function addMessage(role, content, isError = false, localKey = null, fixed = null) {
        const message = {role, content, isError};
        if (localKey) message.localKey = localKey;
        if (fixed) message.fixed = fixed;
        chatHistory.push(message);
        appendMessage(message);
        storageSet('chat_history', JSON.stringify(chatHistory));
        return message;
    }
    function addLocalReply(key, error = false) { addMessage('assistant', localResponse(key), error, key); }
    function busy(value, showThinking = true) {
        sending = value;
        $('send-btn').disabled = value;
        $('loading-indicator')?.remove();
        if (value && showThinking) {
            const loading = document.createElement('div');
            loading.id = 'loading-indicator';
            loading.className = 'loading-indicator';
            const spinner = document.createElement('span');
            spinner.className = 'spinner';
            spinner.setAttribute('aria-hidden', 'true');
            loading.append(spinner, localNode('span', 'thinking'));
            $('chat-messages').appendChild(loading);
            $('chat-messages').scrollTop = $('chat-messages').scrollHeight;
        }
    }
    function requestErrorKey(data, status) {
        const code = String(data?.error_code || data?.code || data?.detail?.code || '').toLowerCase();
        if (['not_configured', 'provider_not_configured'].includes(code)) return 'notConfigured';
        if (['provider_timeout', 'timeout', 'upstream_timeout'].includes(code) || status === 504) return 'providerTimeout';
        if (['rate_limited', 'rate_limit'].includes(code) || status === 429) return 'rateLimited';
        if (['provider_unavailable', 'unavailable'].includes(code) || status === 502 || status === 503) return 'unavailable';
        const legacyMessage = typeof data?.response === 'string' ? data.response : '';
        if (/not configured/i.test(legacyMessage)) return 'notConfigured';
        if (/timed?\s*out|timeout/i.test(legacyMessage)) return 'providerTimeout';
        if (/unavailable/i.test(legacyMessage)) return 'unavailable';
        return 'requestFailed';
    }
    function validFixedReply(fixed) {
        return fixed && ['intro', 'interpret', 'risk', 'models', 'crypto_basics', 'get_crypto'].includes(fixed.id)
            && typeof fixed.zh === 'string' && typeof fixed.en === 'string';
    }
    async function syncFixedReplies() {
        if (!pendingPresetSync.length) return true;
        const request = generation;
        const sessionId = getSessionId();
        activeRequest = new AbortController();
        busy(true, false);
        try {
            while (pendingPresetSync.length) {
                const message = pendingPresetSync[0];
                const fixed = message.fixed;
                const language = fixed.lang === 'en' ? 'en' : 'zh';
                const preset = presets.get(fixed.id);
                const response = await fetch('/api/chat/message', {
                    method: 'POST', headers: {'Content-Type': 'application/json'}, signal: activeRequest.signal,
                    body: JSON.stringify({message: (language === 'en' ? preset?.label_en : preset?.label) || fixed.id,
                        session_id: sessionId, preset_id: fixed.id, lang: language})
                });
                const data = await response.json().catch(() => null);
                if (request !== generation) return false;
                if (!response.ok || data?.error || data?.is_preset !== true
                        || data?.provider !== 'local' || data?.preset_id !== fixed.id) throw new Error('Preset synchronization failed');
                fixed.synced = true;
                pendingPresetSync.shift();
                storageSet('chat_history', JSON.stringify(chatHistory));
            }
            return true;
        } catch (error) {
            if (request === generation && error.name !== 'AbortError') addLocalReply('presetSyncFailed', true);
            return false;
        } finally {
            if (request === generation) { activeRequest = null; busy(false); }
        }
    }
    async function requestMessage(message) {
        if (pendingPresetSync.length && !await syncFixedReplies()) return;
        const request = generation;
        activeRequest = new AbortController();
        busy(true);
        try {
            const selectedDocumentId = window.SkillReader?.getSelectedId?.();
            const response = await fetch('/api/chat/message', {
                method: 'POST', headers: {'Content-Type': 'application/json'}, signal: activeRequest.signal,
                body: JSON.stringify({message, session_id: getSessionId(), lang: currentLang(),
                    ...(selectedDocumentId ? {selected_document_id: selectedDocumentId} : {})})
            });
            const data = await response.json().catch(() => null);
            if (request !== generation) return;
            if (!response.ok || data?.error || /^(?:API\s+)?Error:/i.test(data?.response || '')) {
                addLocalReply(requestErrorKey(data, response.status), true);
            } else if (!data?.response) addLocalReply('noResponse', true);
            else addMessage('assistant', data.response);
        } catch (error) {
            if (request === generation && error.name !== 'AbortError') addLocalReply('requestFailed', true);
        } finally {
            if (request === generation) { activeRequest = null; busy(false); }
        }
    }
    function askPreset(id, label) {
        if (sending) return;
        const preset = presets.get(id);
        if (preset?.type !== 'fixed' || typeof preset.response_zh !== 'string' || typeof preset.response_en !== 'string') {
            addLocalReply('presetsFailed', true);
            return;
        }
        addMessage('user', label);
        const fixed = {id, zh: preset.response_zh, en: preset.response_en, lang: currentLang(), synced: false};
        pendingPresetSync.push(addMessage('assistant', textFor(fixed.zh, fixed.en), false, null, fixed));
        syncFixedReplies();
    }
    async function sendMessage() {
        const message = $('chat-input').value.trim();
        if (!message || sending) return;
        if (pendingPresetSync.length && !await syncFixedReplies()) return;
        addMessage('user', message);
        $('chat-input').value = '';
        requestMessage(message);
    }
    function handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
            event.preventDefault();
            sendMessage();
        }
    }
    function clearChat() {
        const previousSessionId = getSessionId();
        storageSet('chat_session_id', createSessionId());
        generation++;
        activeRequest?.abort();
        activeRequest = null;
        busy(false);
        pendingPresetSync = [];
        chatHistory = [];
        try { localStorage.removeItem('chat_history'); } catch { /* Optional storage. */ }
        renderWelcome();
        fetch('/api/chat/clear', {method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({session_id: previousSessionId})}).catch(() => {});
    }
    function failure(container, key, retry) {
        const button = localNode('button', 'retry', 'skill-reader-button');
        button.type = 'button';
        button.addEventListener('click', retry);
        container.replaceChildren(localNode('p', key, 'error'), button);
    }
    async function loadSkills() {
        const container = $('skills-list');
        try {
            const response = await fetch('/api/skills');
            if (!response.ok) throw new Error('skills unavailable');
            const data = await response.json();
            const registered = {};
            const groups = [];
            for (const [category, skills] of Object.entries(data.categories || {})) {
                if (!skills.length) continue;
                const categoryZh = categories[category] || '🔗 其他知识';
                registered[category] = skills.map(skill => ({...skill,
                    title_zh: skillsZh[skill.name]?.[0] || skill.title,
                    description_zh: skillsZh[skill.name]?.[1] || skill.description,
                    category_zh: categoryZh}));
                const group = document.createElement('div');
                group.className = 'skill-category';
                const header = document.createElement('div');
                header.className = 'category-header';
                const label = document.createElement('span');
                label.dataset.zh = `${categoryZh}（${skills.length}）`;
                label.dataset.en = `${category} (${skills.length})`;
                label.textContent = textFor(label.dataset.zh, label.dataset.en);
                const arrow = document.createElement('span');
                arrow.className = 'arrow'; arrow.textContent = '▶';
                const items = document.createElement('div');
                items.className = 'category-items';
                header.addEventListener('click', () => {
                    items.classList.toggle('expanded');
                    arrow.textContent = items.classList.contains('expanded') ? '▼' : '▶';
                });
                header.append(label, arrow);
                for (const skill of registered[category]) {
                    const item = document.createElement('div');
                    item.className = 'skill-item';
                    Object.assign(item.dataset, {name: skill.name, zh: skill.title_zh, en: skill.title,
                        titleZh: skill.description_zh, titleEn: skill.description});
                    item.textContent = textFor(skill.title_zh, skill.title);
                    item.title = textFor(skill.description_zh, skill.description);
                    item.addEventListener('click', () => window.SkillReader.load(skill.name));
                    items.appendChild(item);
                }
                group.append(header, items); groups.push(group);
            }
            window.SkillReader.register(registered);
            container.replaceChildren(...(groups.length ? groups : [localNode('p', 'skillsEmpty', 'placeholder')]));
        } catch { failure(container, 'skillsFailed', loadSkills); }
    }
    async function loadPresets() {
        const container = $('preset-questions');
        try {
            const response = await fetch('/api/chat/presets');
            if (!response.ok) throw new Error('presets unavailable');
            const data = await response.json();
            const buttons = data.presets.map(preset => {
                presets.set(preset.id, preset);
                const button = document.createElement('button');
                button.className = 'preset-btn'; button.type = 'button';
                button.dataset.zh = preset.label;
                button.dataset.en = preset.label_en || preset.label;
                button.textContent = textFor(button.dataset.zh, button.dataset.en);
                button.addEventListener('click', () => askPreset(preset.id, button.textContent));
                return button;
            });
            container.replaceChildren(...buttons);
        } catch { failure(container, 'presetsFailed', loadPresets); }
    }
    function translateLocalReplies() {
        document.title = textFor('AI 问答 - Ansel Crypto Rank App', 'AI Q&A - Ansel Crypto Rank App');
        for (const node of document.querySelectorAll('[data-local-response]')) {
            node.innerHTML = markdown(localResponse(node.dataset.localResponse));
        }
        for (const node of document.querySelectorAll('[data-fixed-response]')) {
            node.innerHTML = markdown(textFor(node.dataset.fixedZh, node.dataset.fixedEn));
        }
    }
    document.addEventListener('DOMContentLoaded', () => {
        try {
            const saved = JSON.parse(storageGet('chat_history') || '[]');
            if (Array.isArray(saved)) chatHistory = saved.filter(message => ['user', 'assistant'].includes(message.role) && typeof message.content === 'string');
        } catch { chatHistory = []; }
        pendingPresetSync = chatHistory.filter(message => validFixedReply(message.fixed) && message.fixed.synced !== true);
        if (chatHistory.length) {
            $('chat-messages').replaceChildren();
            chatHistory.forEach(appendMessage);
        }
        loadSkills(); loadPresets(); translateLocalReplies();
    });
    window.addEventListener('site:localechange', translateLocalReplies);
    Object.assign(window, {sendMessage, handleKeyDown, clearChat});
})();
