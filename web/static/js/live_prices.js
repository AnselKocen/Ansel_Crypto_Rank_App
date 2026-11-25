/* Market page: read shared predictions, keep selection stable, embed one chart. */
(() => {
    'use strict';
    const MODEL_KEYS = ['extra_all', 'extra_market', 'enet_all', 'enet_market'];
    const INTERVALS = ['15', '60', '240', 'D', 'W'];
    const STORAGE_KEY = 'ansel:live-prices:v1';
    const $ = id => document.getElementById(id);
    const textFor = (zh, en) => window.SiteLocale?.text(zh, en) ?? en;
    const element = (tag, className, text) => {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    };
    function percent(value) {
        if (value === null || value === undefined || !Number.isFinite(Number(value))) return '—';
        const p = Number(value) * 100;
        if (p !== 0 && Math.abs(p) < 0.0001) return p > 0 ? '+<0.0001%' : '−<0.0001%';
        return `${p > 0 ? '+' : ''}${p.toFixed(Math.abs(p) > 0 && Math.abs(p) < .01 ? 4 : 2)}%`;
    }
    function direction(value) {
        return Number(value) > 0 ? 'prediction-positive' : Number(value) < 0 ? 'prediction-negative' : '';
    }
    function readPreferences() {
        let saved = {};
        try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {}; } catch (_) { /* Private browsing is supported. */ }
        const query = new URLSearchParams(location.search);
        const pick = (key, choices, fallback) => {
            const value = query.get(key) || saved[key];
            return choices.includes(value) ? value : fallback;
        };
        const coin = query.get('coin') || saved.coin || '';
        return {model: pick('model', MODEL_KEYS, 'extra_all'), side: pick('side', ['top', 'bottom'], 'top'),
            interval: pick('interval', INTERVALS, '60'),
            coin: typeof coin === 'string' && /^[A-Z0-9][A-Z0-9._-]{0,29}$/.test(coin) ? coin : null,
            supportedOnly: saved.supportedOnly === true};
    }

    class LiveMarket {
        constructor() {
            this.state = readPreferences();
            this.selectionIsAuto = !this.state.coin;
            this.data = null;
            this.query = '';
            this.cache = new Map();
            this.requestSequence = 0;
            this.controller = null;
            this.loading = false;
            this.chartKey = null;
            this.chartTimer = null;
            this.chartLoads = 0;
            this.expectedPeriod = null;
            this.overviewObserver = null;
            this.overviewStarted = false;
        }
        init() {
            this.syncControls();
            const positionNavigation = () => {
                if (!window.matchMedia('(max-width:800px)').matches) return;
                const strip = document.querySelector('.nav-tabs');
                const active = strip?.querySelector('.nav-tab.active');
                if (active) strip.scrollLeft = active.offsetLeft - strip.offsetLeft - strip.clientWidth / 2 + active.clientWidth / 2;
            };
            positionNavigation();
            window.matchMedia('(max-width:800px)').addEventListener('change', positionNavigation);
            $('model-select').addEventListener('change', event => {
                this.state.model = event.target.value;
                this.loadPredictions();
            });
            document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
                this.state.side = button.dataset.view;
                this.syncControls();
                if (this.data?.model === this.state.model) this.applyData();
            }));
            document.querySelectorAll('[data-interval]').forEach(button => button.addEventListener('click', () => {
                this.state.interval = button.dataset.interval;
                this.syncControls();
                this.persist();
                this.renderChart();
            }));
            $('coin-search').addEventListener('input', event => {
                this.query = event.target.value.trim().toUpperCase();
                this.renderList();
            });
            $('chart-available-filter').addEventListener('click', () => {
                this.state.supportedOnly = !this.state.supportedOnly;
                this.syncControls();
                this.renderList();
                this.persist();
            });
            $('refresh-market-list').addEventListener('click', () => this.loadPredictions(true));
            $('retry-chart').addEventListener('click', () => this.renderChart(true));
            $('retry-overview').addEventListener('click', () => this.loadOverview(true));
            $('copy-market-link').addEventListener('click', () => this.copyLink());
            document.addEventListener('weekly:status', event => {
                const period = event.detail.data_period;
                if (!period || !event.detail.data_current) return;
                this.expectedPeriod = period;
                if (this.data && this.data.data_period !== period && !this.loading) {
                    this.cache.clear();
                    this.loadPredictions(true);
                }
            });
            window.addEventListener('popstate', () => {
                this.state = readPreferences();
                this.selectionIsAuto = !this.state.coin;
                this.syncControls();
                this.loadPredictions();
            });
            window.addEventListener('site:localechange', () => {
                this.syncControls();
                if (this.data && !this.loading) this.applyData();
                if (this.overviewStarted) this.loadOverview(true);
                if ($('ticker-tape-container').childElementCount) this.loadTicker();
            });
            this.loadPredictions();
            this.prepareOverview();
            if ('requestIdleCallback' in window) window.requestIdleCallback(() => this.loadTicker(), {timeout: 1800});
            else setTimeout(() => this.loadTicker(), 600);
        }
        syncControls() {
            $('model-select').value = this.state.model;
            document.querySelectorAll('[data-view]').forEach(button => {
                const active = button.dataset.view === this.state.side;
                button.classList.toggle('active', active);
                button.setAttribute('aria-pressed', String(active));
            });
            document.querySelectorAll('[data-interval]').forEach(button => {
                const active = button.dataset.interval === this.state.interval;
                button.classList.toggle('active', active);
                button.setAttribute('aria-pressed', String(active));
            });
            $('chart-available-filter').setAttribute('aria-pressed', String(this.state.supportedOnly));
            $('coins-title').textContent = this.state.side === 'top' ? textFor('多头 Top 20', 'Long · Top 20') : textFor('空头 Bottom 20', 'Short · Bottom 20');
            $('list-sort-label').textContent = this.state.side === 'top' ? textFor('由强到弱', 'Strongest first') : textFor('由弱到强', 'Weakest first');
        }
        persist() {
            try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.state)); } catch (_) { /* No storage is required. */ }
            const url = new URL(location.href);
            for (const key of ['model', 'side', 'interval']) url.searchParams.set(key, this.state[key]);
            if (this.state.coin) url.searchParams.set('coin', this.state.coin);
            else url.searchParams.delete('coin');
            history.replaceState(null, '', url);
        }
        coins() {
            if (this.data?.model !== this.state.model) return [];
            const coins = [...(this.state.side === 'top' ? this.data.top_20 : this.data.bottom_20)];
            return coins.sort((a, b) => this.state.side === 'top' ? a.rank - b.rank : b.rank - a.rank);
        }
        async loadPredictions(force = false) {
            const sequence = ++this.requestSequence;
            const model = this.state.model;
            this.controller?.abort();
            this.syncControls();
            if (!force && this.cache.has(model)) {
                this.loading = false;
                this.data = this.cache.get(model);
                this.applyData();
                return;
            }
            this.loading = true;
            $('refresh-market-list').disabled = true;
            $('coins-grid').setAttribute('aria-busy', 'true');
            $('coins-grid').replaceChildren(...Array.from({length: 6}, () => element('li', 'list-skeleton')));
            $('market-feedback').textContent = textFor('正在读取已发布名单…', 'Loading the published list…');
            $('selected-prediction').textContent = '—';
            const controller = new AbortController();
            this.controller = controller;
            const timeout = setTimeout(() => controller.abort(), 12000);
            try {
                const response = await fetch(`/api/predictions/data/${encodeURIComponent(model)}`, {signal: controller.signal, cache: 'no-store'});
                if (!response.ok) throw new Error('Unavailable predictions');
                const data = await response.json();
                if (sequence !== this.requestSequence) return;
                if (data.model !== model || !Array.isArray(data.top_20) || !Array.isArray(data.bottom_20)) throw new Error('Invalid predictions');
                this.cache.set(model, data);
                this.data = data;
                this.loading = false;
                this.applyData();
                $('market-feedback').textContent = force ? textFor('已重新读取共享名单，无需重新训练。', 'Shared list refreshed; no model retraining needed.') : '';
                if (data.chart_catalog_ready === false && (this.catalogRetries || 0) < 5) {
                    this.catalogRetries = (this.catalogRetries || 0) + 1;
                    $('market-feedback').textContent = textFor('正在准备交易对目录，名单与预测已可查看。', 'Preparing trading pairs; token lists and forecasts are available.');
                    clearTimeout(this.catalogTimer);
                    this.catalogTimer = setTimeout(() => {
                        if (!this.loading && this.data?.chart_catalog_ready === false) this.loadPredictions(true);
                    }, 3500);
                }
                if (this.expectedPeriod && data.data_period !== this.expectedPeriod && this.lastVersionRetry !== this.expectedPeriod) {
                    this.lastVersionRetry = this.expectedPeriod;
                    this.cache.clear();
                    this.loadPredictions(true);
                }
            } catch (error) {
                if (sequence !== this.requestSequence) return;
                this.loading = false;
                this.listMessage(error.name === 'AbortError' ? textFor('名单加载超时，请重试。', 'List loading timed out. Please retry.') : textFor('暂时无法读取该模型名单，请重试。', 'This model list is unavailable. Please retry.'), () => this.loadPredictions(true));
                $('market-feedback').textContent = textFor('图表保留上次选币，当前模型名单尚未加载完成。', 'The chart keeps the previous selection while the model list loads.');
                $('coin-count').textContent = '—';
            } finally {
                clearTimeout(timeout);
                if (sequence === this.requestSequence) {
                    $('refresh-market-list').disabled = false;
                    $('coins-grid').setAttribute('aria-busy', 'false');
                }
            }
        }
        applyData() {
            const coins = this.coins();
            const selected = coins.find(coin => coin.symbol === this.state.coin);
            if (!selected || (this.selectionIsAuto && !selected.tradingview && coins.some(coin => coin.tradingview))) {
                this.state.coin = (coins.find(coin => coin.tradingview) || coins[0])?.symbol || null;
                this.selectionIsAuto = true;
            }
            const data = this.data;
            $('market-data-period').textContent = data.data_period ? textFor(`预测数据截至 ${data.data_period} UTC`, `Predictions through ${data.data_period} UTC`) : textFor('预测日期未知，请以模型原始记录为准', 'Prediction date unavailable; refer to the original model record.');
            $('market-forecast-window').textContent = data.forecast_start && data.forecast_end ? textFor(`预测区间 ${data.forecast_start} → ${data.forecast_end}`, `Forecast ${data.forecast_start} → ${data.forecast_end}`) : textFor('预测区间未标注', 'Forecast window not specified');
            $('refresh-market-list').disabled = false;
            $('coins-grid').setAttribute('aria-busy', 'false');
            this.renderList();
            this.renderSelected();
            this.persist();
            this.renderChart();
        }
        listMessage(message, retry) {
            const li = element('li', 'list-message');
            li.appendChild(element('p', '', message));
            if (retry) {
                const button = element('button', 'market-button', textFor('重新加载', 'Reload'));
                button.type = 'button';
                button.addEventListener('click', retry);
                li.appendChild(button);
            }
            $('coins-grid').replaceChildren(li);
        }
        renderList() {
            if (this.loading) return;
            const all = this.coins();
            const coins = all.filter(coin => (!this.state.supportedOnly || coin.tradingview) &&
                (!this.query || coin.symbol.toUpperCase().includes(this.query)));
            $('coin-count').textContent = `${coins.length} / ${all.length}`;
            if (!coins.length) {
                this.listMessage(all.length ? textFor('没有匹配币种，试试清空搜索或关闭图表筛选。', 'No matching tokens. Clear the search or disable the chart filter.') : textFor('当前名单暂无币种。', 'No tokens in the current list.'));
                return;
            }
            const nodes = coins.map(coin => {
                const li = element('li');
                const button = element('button', `coin-row${coin.symbol === this.state.coin ? ' active' : ''}${coin.tradingview ? '' : ' unavailable'}`);
                button.type = 'button';
                button.dataset.symbol = coin.symbol;
                button.setAttribute('aria-pressed', String(coin.symbol === this.state.coin));
                button.setAttribute('aria-label', textFor(`${coin.symbol}，全币池第${coin.rank}名，预测周收益${percent(coin.value)}${coin.tradingview ? '' : '，暂无图表映射'}`, `${coin.symbol}, universe rank ${coin.rank}, predicted weekly return ${percent(coin.value)}${coin.tradingview ? '' : ', no chart mapping'}`));
                button.appendChild(element('span', 'rank', `#${coin.rank}`));
                const identity = element('span', 'coin-identity');
                identity.appendChild(element('strong', '', coin.symbol));
                identity.appendChild(element('small', '', coin.tradingview ? coin.tradingview.replace(':', ' · ') : textFor('图表暂未配置', 'Chart not configured')));
                button.appendChild(identity);
                const value = element('span', `coin-prediction ${direction(coin.value)}`, percent(coin.value));
                value.title = textFor(`预测周收益 ${(Number(coin.value) * 100).toPrecision(6)}%`, `Predicted weekly return ${(Number(coin.value) * 100).toPrecision(6)}%`);
                button.appendChild(value);
                button.addEventListener('click', () => this.selectCoin(coin.symbol));
                li.appendChild(button);
                return li;
            });
            $('coins-grid').replaceChildren(...nodes);
        }
        selectCoin(symbol) {
            if (!this.coins().some(coin => coin.symbol === symbol)) return;
            this.state.coin = symbol;
            this.selectionIsAuto = false;
            this.renderList();
            this.renderSelected();
            this.persist();
            this.renderChart();
        }
        selected() { return this.coins().find(coin => coin.symbol === this.state.coin); }
        renderSelected() {
            const coin = this.selected();
            $('chart-title').textContent = coin?.symbol || textFor('暂无选币', 'No token selected');
            $('selected-pair').textContent = coin?.tradingview ? `${coin.tradingview.replace(':', ' · ')} · UTC`
                : coin ? textFor('该币保留在模型名单中，图表暂未配置', 'This token remains in the model list; its chart is not configured.') : textFor('选择名单中的币种查看图表', 'Select a token from the list to view its chart.');
            $('selected-prediction').textContent = coin ? percent(coin.value) : '—';
            $('selected-prediction').className = coin ? direction(coin.value) : '';
            $('selected-rank').textContent = coin ? textFor(`全币池 #${coin.rank} / ${this.data.universe_count}`, `Universe #${coin.rank} / ${this.data.universe_count}`) : textFor('全币池排名 —', 'Universe rank —');
        }
        renderChart(force = false) {
            const coin = this.selected();
            const symbol = coin?.tradingview;
            const external = $('open-tradingview');
            const validSymbol = symbol && /^[A-Z0-9_]+:[A-Z0-9._/-]+$/.test(symbol);
            $('retry-chart').disabled = !validSymbol;
            external.hidden = !validSymbol;
            if (!validSymbol) {
                clearTimeout(this.chartTimer);
                this.chartKey = null;
                $('main-chart').replaceChildren();
                $('chart-loading').hidden = true;
                $('chart-placeholder').hidden = false;
                $('chart-placeholder').querySelector('h3').textContent = coin ? textFor('该币暂无可靠图表映射', 'No reliable chart mapping for this token') : textFor('暂无可显示行情', 'No market chart available');
                $('chart-placeholder-text').textContent = coin?.chart_unavailable_reason
                    ? textFor(coin.chart_unavailable_reason, 'No verified TradingView pair is available for this token; its model ranking and forecast remain visible.')
                    : textFor('可从名单选择其他币种；模型预测仍可查看。', 'Select another token from the list; model forecasts remain available.');
                $('chart-connection-status').textContent = textFor('暂无图表', 'No chart');
                $('chart-canvas').setAttribute('aria-busy', 'false');
                return;
            }
            external.href = `https://www.tradingview.com/chart/?symbol=${encodeURIComponent(symbol)}`;
            const key = `${symbol}|${this.state.interval}|${window.SiteLocale?.language || 'en'}`;
            if (!force && this.chartKey === key && $('main-chart').querySelector('iframe')) return;
            this.chartKey = key;
            clearTimeout(this.chartTimer);
            const frame = document.createElement('iframe');
            const params = new URLSearchParams({symbol, interval: this.state.interval, theme: 'dark', style: '1',
                timezone: 'Etc/UTC', locale: window.SiteLocale?.language === 'zh' ? 'zh_CN' : 'en', hidesidetoolbar: '0', symboledit: '0', saveimage: '1',
                toolbarbg: '0f1c2e', withdateranges: '1', showvolume: '1', studies: '[]'});
            frame.src = `https://s.tradingview.com/widgetembed/?${params}`;
            frame.title = textFor(`${coin.symbol} TradingView 交易对行情`, `${coin.symbol} TradingView market chart`);
            frame.allowFullscreen = true;
            frame.referrerPolicy = 'strict-origin-when-cross-origin';
            const finish = () => {
                if (this.chartKey !== key || !$('main-chart').contains(frame)) return;
                clearTimeout(this.chartTimer);
                $('chart-loading').hidden = true;
                $('chart-canvas').setAttribute('aria-busy', 'false');
                // Cross-origin frame load does not certify price freshness.
                $('chart-connection-status').textContent = textFor('TradingView 图表', 'TradingView chart');
            };
            frame.addEventListener('load', finish, {once: true});
            frame.addEventListener('error', () => {
                if ($('main-chart').contains(frame)) this.chartSlow(key);
            }, {once: true});
            $('chart-placeholder').hidden = true;
            $('chart-loading').hidden = false;
            $('chart-loading-text').textContent = textFor('正在连接图表…', 'Connecting to chart…');
            $('chart-connection-status').textContent = textFor('连接中…', 'Connecting…');
            $('chart-canvas').setAttribute('aria-busy', 'true');
            $('main-chart').replaceChildren(frame);
            this.chartLoads += 1;
            this.chartTimer = setTimeout(() => this.chartSlow(key), 14000);
        }
        chartSlow(key) {
            if (this.chartKey !== key) return;
            $('chart-loading').hidden = false;
            $('chart-loading-text').textContent = textFor('图表加载较慢，可重新连接或在 TradingView 打开。', 'The chart is taking longer to load. Reconnect or open TradingView.');
            $('chart-connection-status').textContent = textFor('等待行情服务', 'Waiting for market service');
            $('chart-canvas').setAttribute('aria-busy', 'false');
        }
        prepareOverview() {
            if ('IntersectionObserver' in window) {
                this.overviewObserver = new IntersectionObserver(entries => {
                    if (entries.some(entry => entry.isIntersecting)) {
                        this.loadOverview();
                        this.overviewObserver.disconnect();
                    }
                }, {rootMargin: '180px'});
                this.overviewObserver.observe($('market-overview'));
            } else this.loadOverview();
        }
        embed(container, source, config, onReady, onError) {
            const wrapper = element('div', 'tradingview-widget-container');
            wrapper.style.cssText = 'width:100%;height:100%';
            const widget = element('div', 'tradingview-widget-container__widget');
            widget.style.cssText = 'width:100%;height:100%';
            const script = document.createElement('script');
            script.src = source;
            script.async = true;
            script.textContent = JSON.stringify(config);
            script.addEventListener('error', onError, {once: true});
            wrapper.append(widget, script);
            const observer = new MutationObserver(() => {
                const frame = wrapper.querySelector('iframe');
                if (frame) {
                    frame.addEventListener('load', onReady, {once: true});
                    observer.disconnect();
                }
            });
            observer.observe(wrapper, {childList: true, subtree: true});
            setTimeout(() => observer.disconnect(), 20000);
            container.replaceChildren(wrapper);
        }
        loadOverview(force = false) {
            if (this.overviewStarted && !force) return;
            this.overviewStarted = true;
            const attempt = this.overviewAttempt = (this.overviewAttempt || 0) + 1;
            clearTimeout(this.overviewTimer);
            $('overview-status').hidden = false;
            $('overview-status').textContent = textFor('正在连接 TradingView 市场概览…', 'Connecting to the TradingView market overview…');
            this.embed($('tradingview-market-widget'), 'https://s3.tradingview.com/external-embedding/embed-widget-crypto-coins-heatmap.js',
                {dataSource: 'Crypto', blockSize: 'market_cap_calc', blockColor: 'change', locale: window.SiteLocale?.language === 'zh' ? 'zh_CN' : 'en', colorTheme: 'dark',
                 hasTopBar: true, isDataSet498Enabled: true, isZoomEnabled: true, hasSymbolTooltip: true,
                 isTransparent: true, width: '100%', height: '100%'},
                () => { if (attempt === this.overviewAttempt) { clearTimeout(this.overviewTimer); $('overview-status').hidden = true; } },
                () => { if (attempt === this.overviewAttempt) { clearTimeout(this.overviewTimer); $('overview-status').textContent = textFor('市场概览暂未连接，可点击重新加载。', 'Market overview unavailable. Select Reload to retry.'); } });
            this.overviewTimer = setTimeout(() => {
                if (attempt === this.overviewAttempt && !$('overview-status').hidden) {
                    $('overview-status').textContent = textFor('市场概览加载较慢，可点击重新加载或稍后查看。', 'The market overview is taking longer to load. Reload or check again later.');
                }
            }, 14000);
        }
        loadTicker() {
            const symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'AVAX', 'DOT'];
            this.embed($('ticker-tape-container'), 'https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js',
                {symbols: symbols.map(symbol => ({proName: `BINANCE:${symbol}USDT`, title: symbol})),
                 showSymbolLogo: true, colorTheme: 'dark', isTransparent: true, displayMode: 'adaptive', locale: window.SiteLocale?.language === 'zh' ? 'zh_CN' : 'en'},
                () => {}, () => { $('ticker-tape-container').hidden = true; });
        }
        async copyLink() {
            this.persist();
            try {
                await navigator.clipboard.writeText(location.href);
                $('market-feedback').textContent = textFor('已复制当前模型、名单、币种与K线周期的链接。', 'Copied a link with the current model, list, token and chart interval.');
            } catch (_) {
                $('market-feedback').textContent = textFor('可复制浏览器地址栏，当前选币已保存在链接中。', 'Copy the browser address; the current token selection is saved in the link.');
            }
        }
    }
    const init = () => {
        if (!$('main-chart') || window.liveMarket) return;
        window.liveMarket = new LiveMarket();
        window.liveMarket.init();
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init, {once: true});
    else init();
})();
