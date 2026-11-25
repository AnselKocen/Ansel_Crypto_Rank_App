/* Localize model documentation while keeping source parameters and field IDs intact. */
(() => {
    'use strict';

    const text = (zh, en) => window.SiteLocale?.text(zh, en) ?? zh;
    const pageTitles = {
        '/transparency': ['模型透明度', 'Transparency'],
        '/model-strategy': ['模型策略', 'Model Strategy'],
        '/hyper-params': ['超参数', 'Hyperparameters'],
        '/features': ['特征选择', 'Features'],
        '/backtest': ['策略回测', 'Backtest']
    };
    const strategyTranslations = new Map([
        ['market: Only market variables were used for modeling, including momentum, volatility, usd_volume, base_volume, return_sign, long_candle, strev.',
            'market（市场特征）：仅使用市场变量建模，包括动量、波动率、美元成交量、基础币成交量、收益方向、长实体 K 线与短期反转指标。'],
        ['all: All variables were used for modeling, including market variables plus weekly_sentiment and extreme_sentiment.',
            'all（全部特征）：使用全部变量建模，包括市场变量、周度情绪与极端情绪指标。'],
        ['enet_EW: Top 20 coins ranked by enet predictions are selected and held equally weighted (long only).',
            'enet_EW（ElasticNet 等权做多）：选择 ElasticNet 预测排名前 20 的币种，以相同权重持有，仅做多。'],
        ['extra_EW: Top 20 coins ranked by extra predictions are selected and held equally weighted (long only).',
            'extra_EW（ExtraTrees 等权做多）：选择 ExtraTrees 预测排名前 20 的币种，以相同权重持有，仅做多。'],
        ['enet_ls: Long top 20 and short bottom 20 coins based on enet ranking, equally weighted on both sides.',
            'enet_ls（ElasticNet 多空）：根据 ElasticNet 排名，做多前 20 个币种、做空后 20 个币种，两侧各自等权配置。'],
        ['extra_ls: Long top 20 and short bottom 20 coins based on extra ranking, equally weighted on both sides.',
            'extra_ls（ExtraTrees 多空）：根据 ExtraTrees 排名，做多前 20 个币种、做空后 20 个币种，两侧各自等权配置。'],
        ['fusion_ls: For coins in both enet and extra top 20, assign long weight of 0.65; if only in one, assign 0.35. Similarly, for bottom 20, assign short weight of -0.65 (both models) or -0.35 (only one).',
            'fusion_ls（融合多空）：同时位于两个模型前 20 名的币种，做多权重为 0.65；仅入选一个模型时为 0.35。后 20 名同理，同时入选时做空权重为 -0.65，仅入选一个模型时为 -0.35。']
    ]);
    const emojiMap = {market: '🔵', all: '🟢', enet_EW: '🟡', extra_EW: '🟠', enet_ls: '🔴', extra_ls: '🟣', fusion_ls: '🟤'};
    const modelNames = {
        enet_all: 'ElasticNet · 全部特征', enet_market: 'ElasticNet · 市场特征',
        extra_all: 'ExtraTrees · 全部特征', extra_market: 'ExtraTrees · 市场特征',
        ElasticNet_all: 'ElasticNet · 全部特征', ElasticNet_market: 'ElasticNet · 市场特征'
    };
    const featureNames = {
        usd_volume: '美元成交量', base_volume: '基础币成交量',
        return_sign_consistency_3d: '近 3 天正收益占比', long_candle_ratio_5d: '近 5 天长实体 K 线占比',
        strev_daily: '日度短期反转指标', strev_weekly: '周度短期反转指标',
        compound_weekly: '周度综合情绪', extremely_positive_ratio: '极端正面情绪占比',
        extremely_negative_ratio: '极端负面情绪占比'
    };

    function bilingual(element, zh, en) {
        element.dataset.zh = zh;
        element.dataset.en = en;
        element.textContent = text(zh, en);
        return element;
    }

    function localizeMetadata() {
        const title = pageTitles[location.pathname.replace(/\/$/, '')];
        if (title) document.title = `${text(...title)} - Ansel Crypto Rank App`;
        document.querySelectorAll('[data-alt-zh][data-alt-en]').forEach(image => {
            image.alt = text(image.dataset.altZh, image.dataset.altEn);
        });
    }

    function renderStrategy(container, content) {
        const fragment = document.createDocumentFragment();
        content.split(/\r?\n/).forEach(source => {
            const line = source.trim();
            if (!line) return;
            const key = line.split(':', 1)[0];
            const row = document.createElement('div');
            row.className = `strategy-line${line.includes(':') ? ' highlighted' : ''}`;
            if (emojiMap[key]) {
                const marker = document.createElement('span');
                marker.className = 'emoji-marker';
                marker.setAttribute('aria-hidden', 'true');
                marker.textContent = emojiMap[key];
                row.append(marker);
            }
            row.append(bilingual(document.createElement(line.includes(':') ? 'strong' : 'span'),
                strategyTranslations.get(line) || line, line));
            fragment.append(row);
        });
        container.replaceChildren(fragment);
    }

    function localizeModelLabels(content) {
        return content.replace(/^(enet_all|enet_market|extra_all|extra_market|ElasticNet_all|ElasticNet_market)([：:])/gm,
            (match, key, colon) => `${modelNames[key]}（${key}）${colon}`);
    }

    function renderCode(container, content, kind) {
        let chinese = localizeModelLabels(content);
        if (kind === 'params') {
            chinese = chinese.replace(
                /The following are the optimal hyperparameters for each model after tuning\. Based on data from (\d{4}-\d{2}-\d{2}) to (\d{4}-\d{2}-\d{2})/,
                '以下为各模型调参后的最优超参数，使用的数据范围为 $1 至 $2。');
        } else {
            chinese = chinese.replace(/^(\s*-\s*)([A-Za-z0-9_]+)/gm, (match, prefix, key) => {
                let label = featureNames[key];
                const period = key.match(/^(momentum|volatility)_(\d+)$/);
                const volume = key.match(/^v_(\d+)d$/);
                if (period) label = `${period[2]} 天${period[1] === 'momentum' ? '动量' : '波动率'}`;
                if (volume) label = `${volume[1]} 天成交量冲击`;
                return label ? `${prefix}${key}（${label}）` : match;
            });
        }
        const code = bilingual(document.createElement('pre'), chinese, content);
        code.className = 'code-block';
        container.replaceChildren(code);
    }

    async function loadDocument(id, filename, kind) {
        const container = document.getElementById(id);
        if (!container) return;
        try {
            const response = await fetch(`/api/predictions/text/${filename}`);
            if (!response.ok) throw new Error('Document unavailable');
            const data = await response.json();
            if (typeof data.content !== 'string') throw new Error('Invalid document');
            if (kind === 'strategy') renderStrategy(container, data.content);
            else renderCode(container, data.content, kind);
        } catch {
            const names = {strategy: ['策略信息', 'strategy information'], params: ['超参数', 'hyperparameters'], features: ['特征', 'features']};
            const [zh, en] = names[kind];
            const error = bilingual(document.createElement('p'), `${zh}加载失败，请稍后重试。`, `Failed to load ${en}. Please try again later.`);
            error.className = 'error';
            container.replaceChildren(error);
        }
    }

    window.loadBacktestImages = () => {
        const selected = document.querySelector('input[name="metric"]:checked');
        if (!selected) return;
        const timestamp = Date.now();
        for (const source of ['all', 'market']) {
            const image = document.getElementById(`backtest-${source}`);
            if (image) image.src = `/figures/${selected.value}_${source}_with_baseline.png?t=${timestamp}`;
        }
    };

    window.addEventListener('site:localechange', localizeMetadata);
    document.addEventListener('DOMContentLoaded', () => {
        localizeMetadata();
        loadDocument('strategy-content', 'strategy_intro.txt', 'strategy');
        loadDocument('params-content', 'best_params.txt', 'params');
        loadDocument('enet-features', 'enet_features.txt', 'features');
        window.loadBacktestImages();
    });
})();
